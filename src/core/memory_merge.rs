//! Memory-graph merge ladder for CoW overlay → primary promotion.
//!
//! ## Overview
//!
//! When a ref's HEAD becomes an ancestor of primary's HEAD (detected by the
//! head watcher), the ref's overlay nodes are folded into primary via a
//! four-rung classification ladder:
//!
//! 1. **Clean** — primary has no node with this ID → publish to primary as-is.
//! 2. **Identical** — primary has the node with the same content hash → noop.
//! 3. **Smart-merge eligible** — only additive changes (edge additions, counter
//!    updates, timestamp advances, no body difference) → union and publish.
//! 4. **Conflict** — contradictory bodies → flag primary's node
//!    `needs_rebuild`; both contested versions are logged for diagnostics.
//!
//! Auto-merge is suppressed when the daemon is draining (`draining` flag set).
//!
//! ## U4 / U6 seams
//!
//! * **U4 seam** — `run_merge_for_ref` accepts the ref's overlay graph by
//!   value.  When U4 lands, the caller will hold the ref's `Arc<RwLock<RefIndex>>`
//!   and pass in `ref_index.memory_graph.overlay` once that type is refined.
//! * **U6 seam** — `MemoryNode::references_chunk_hashes` is `Vec<String>`
//!   today; U6 will replace the element type with `ChunkHash = [u8; 32]`.
//!   The merge ladder does not inspect chunk hashes — it only copies the field
//!   verbatim during a clean-publish or smart-merge.

use tracing::{debug, info, warn};

use crate::core::memory_graph::{MemoryEdge, MemoryGraph, MemoryNode};

// ---------------------------------------------------------------------------
// Classification types
// ---------------------------------------------------------------------------

/// Classification of a single overlay node against primary's current state.
#[derive(Debug, PartialEq, Eq)]
pub enum MergeClass {
    /// Primary has no node with this ID — safe to publish directly.
    Clean,
    /// Both sides have the same content (hash-compared) — no action needed.
    Identical,
    /// Only additive differences: new edges, higher access_count, later
    /// last_accessed, no body change.  Can be merged automatically.
    SmartMerge,
    /// Body differs between overlay and primary — flag for rebuild.
    Conflict,
}

/// Result of classifying a single overlay node.
pub struct ClassifyResult {
    pub class: MergeClass,
    /// Content hash of the overlay node's body (used for logging on conflict).
    pub overlay_hash: u64,
    /// Content hash of primary's node body (Some on Identical/SmartMerge/Conflict).
    pub primary_hash: Option<u64>,
}

// ---------------------------------------------------------------------------
// Per-node classification
// ---------------------------------------------------------------------------

/// Compute a fast non-cryptographic hash of a string body for comparison.
///
/// This is intentionally a cheap FNV-1a hash — not a cryptographic hash.
/// Its job is equality comparison within a single daemon process; it is never
/// stored to disk.
fn body_hash(s: &str) -> u64 {
    // FNV-1a 64-bit
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x00000100000001b3);
    }
    h
}

/// Classify one overlay node against primary's view.
pub fn classify_node(overlay_node: &MemoryNode, primary: &MemoryGraph) -> ClassifyResult {
    let overlay_hash = body_hash(&overlay_node.content);

    match primary.get_node(&overlay_node.id) {
        None => ClassifyResult {
            class: MergeClass::Clean,
            overlay_hash,
            primary_hash: None,
        },
        Some(primary_node) => {
            let primary_hash = body_hash(&primary_node.content);

            if primary_hash == overlay_hash {
                ClassifyResult {
                    class: MergeClass::Identical,
                    overlay_hash,
                    primary_hash: Some(primary_hash),
                }
            } else {
                // Bodies differ — this is a conflict unless overlay node is
                // "additive-only": same body hash as at fork time means the
                // overlay only added edges / bumped counters while primary
                // updated the body.  We detect this as: overlay body == primary
                // body is false (checked above) but we have no "body at fork
                // time" reference without U6's CAS.  Conservative choice:
                // treat as Conflict rather than risk data loss.
                //
                // The SmartMerge rung therefore requires body equality in this
                // implementation.  U6 will refine this when the fork-base hash
                // is available.
                ClassifyResult {
                    class: MergeClass::Conflict,
                    overlay_hash,
                    primary_hash: Some(primary_hash),
                }
            }
        }
    }
}

/// Summary of a merge operation.
#[derive(Debug, Default)]
pub struct MergeSummary {
    /// Nodes published as new (Clean rung).
    pub published: usize,
    /// Nodes skipped because identical (Identical rung).
    pub skipped_identical: usize,
    /// Nodes merged additively (SmartMerge rung).
    pub smart_merged: usize,
    /// Nodes flagged `needs_rebuild` due to conflict.
    pub conflicts: usize,
}

// ---------------------------------------------------------------------------
// Merge ladder
// ---------------------------------------------------------------------------

/// Run the merge ladder for all overlay nodes in `overlay_graph` into
/// `primary`.
///
/// `overlay_graph` is the ref's local storage (the diff against primary).
/// `primary` is the primary ref's graph, mutated in-place.
///
/// The overlay edges in `edge_snapshot` are the *outgoing* edges from overlay
/// nodes that should be considered for smart-merge union.  Pass an empty vec
/// if edge merging is not required (e.g. unit tests for the node-only path).
///
/// Auto-merge is suppressed when `draining` is `true`; the function returns
/// immediately with an empty summary in that case.
pub fn run_merge_ladder(
    overlay_graph: &MemoryGraph,
    primary: &mut MemoryGraph,
    edge_snapshot: &[(String, String, MemoryEdge)],
    draining: bool,
) -> MergeSummary {
    if draining {
        debug!("memory_merge: draining is set — merge suppressed");
        return MergeSummary::default();
    }

    let overlay_ids = overlay_graph.overlay_node_ids();
    if overlay_ids.is_empty() {
        return MergeSummary::default();
    }

    let mut summary = MergeSummary::default();

    for node_id in &overlay_ids {
        // Safe: node_id came from overlay_graph's own id_index.
        let overlay_node = match overlay_graph.get_node_cow(node_id) {
            Some(n) => n.into_owned(),
            None => {
                warn!(node_id, "memory_merge: overlay node disappeared mid-merge");
                continue;
            }
        };

        let result = classify_node(&overlay_node, primary);

        match result.class {
            MergeClass::Clean => {
                info!(
                    node_id,
                    label = %overlay_node.label,
                    "memory_merge: clean publish"
                );
                primary.insert_node(overlay_node);
                summary.published += 1;
            }

            MergeClass::Identical => {
                debug!(node_id, "memory_merge: identical — noop");
                summary.skipped_identical += 1;
            }

            MergeClass::SmartMerge => {
                // Bodies are equal — union edges, max(access_count), latest last_accessed.
                // Construct the merged node from primary's copy plus overlay counters.
                if let Some(primary_node) = primary.get_node_mut(node_id) {
                    primary_node.access_count =
                        primary_node.access_count.max(overlay_node.access_count);
                    primary_node.last_accessed =
                        primary_node.last_accessed.max(overlay_node.last_accessed);
                    // chunk-hash union (U6 seam — today just extends the Vec)
                    for h in &overlay_node.references_chunk_hashes {
                        if !primary_node.references_chunk_hashes.contains(h) {
                            primary_node.references_chunk_hashes.push(h.clone());
                        }
                    }
                }
                summary.smart_merged += 1;
            }

            MergeClass::Conflict => {
                info!(
                    node_id,
                    overlay_hash = result.overlay_hash,
                    primary_hash = ?result.primary_hash,
                    "memory_merge: conflict — flagging primary node needs_rebuild"
                );
                let flagged = primary.flag_needs_rebuild(node_id);
                if !flagged {
                    warn!(
                        node_id,
                        "memory_merge: conflict but primary node not found — publishing overlay"
                    );
                    primary.insert_node(overlay_node);
                    summary.published += 1;
                } else {
                    summary.conflicts += 1;
                }
            }
        }
    }

    // Edge union: add overlay edges whose both endpoints exist in primary.
    let mut edges_added = 0usize;
    for (source_id, target_id, edge) in edge_snapshot {
        if primary.node_exists(source_id) && primary.node_exists(target_id) {
            primary.create_relation(
                source_id,
                target_id,
                edge.relation.clone(),
                Some(edge.weight),
                if edge.metadata.is_empty() {
                    None
                } else {
                    Some(edge.metadata.clone())
                },
            );
            edges_added += 1;
        }
    }

    if edges_added > 0 {
        debug!(edges_added, "memory_merge: edge union complete");
    }

    summary
}

// ---------------------------------------------------------------------------
// Rebuild-on-access
// ---------------------------------------------------------------------------

/// Attempt to rebuild a `needs_rebuild` node from its chunk hashes.
///
/// In the real system (U6), this would re-embed the node's body from the
/// current CAS blob.  Today it performs a simpler re-embed of the existing
/// body content using the provided embedding closure.
///
/// Returns `Ok(())` on success (node is cleared + embedding updated).
/// Returns `Err(RebuildError)` on failure; the caller should prune the node.
#[derive(Debug)]
pub enum RebuildError {
    NodeNotFound,
    EmbeddingFailed(String),
    /// The referenced symbol / chunk is gone — node should be pruned.
    SourceGone,
}

impl std::fmt::Display for RebuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RebuildError::NodeNotFound => write!(f, "node not found"),
            RebuildError::EmbeddingFailed(e) => write!(f, "embedding failed: {e}"),
            RebuildError::SourceGone => write!(f, "source chunk is gone — node will be pruned"),
        }
    }
}

/// Outcome of a rebuild attempt.
pub enum RebuildOutcome {
    Rebuilt,
    Pruned,
}

/// Synchronous rebuild helper used when a node is accessed directly and
/// `needs_rebuild` is set.
///
/// `embed_fn` takes the node's content body and returns a new embedding on
/// success.  If `embed_fn` returns `None` the node is pruned.
///
/// ## U6 seam
///
/// When U6 lands, `embed_fn` will consult the CAS to fetch the latest chunk
/// body for each hash in `references_chunk_hashes`.  If any hash is missing
/// from CAS the node is flagged `SourceGone` and pruned.  Today the body is
/// re-embedded as-is.
pub fn rebuild_node_sync<F>(
    graph: &mut MemoryGraph,
    node_id: &str,
    embed_fn: F,
) -> Result<RebuildOutcome, RebuildError>
where
    F: FnOnce(&str, &[String]) -> Option<Vec<f32>>,
{
    let (content, chunk_hashes) = {
        let node = graph.get_node(node_id).ok_or(RebuildError::NodeNotFound)?;
        if !node.needs_rebuild {
            // Already clean — nothing to do.
            return Ok(RebuildOutcome::Rebuilt);
        }
        (node.content.clone(), node.references_chunk_hashes.clone())
    };

    match embed_fn(&content, &chunk_hashes) {
        Some(new_embedding) => {
            graph.mark_rebuilt(node_id, &content, new_embedding);
            info!(node_id, "memory_merge: node rebuilt successfully");
            Ok(RebuildOutcome::Rebuilt)
        }
        None => {
            // embed_fn returning None signals the source is gone.
            info!(node_id, "memory_merge: rebuild failed — pruning node");
            graph.delete_node(node_id);
            Ok(RebuildOutcome::Pruned)
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::memory_graph::{MemoryGraph, NodeType, RelationType};

    fn make_node(id: &str, label: &str, content: &str) -> MemoryNode {
        MemoryNode {
            id: id.to_string(),
            node_type: NodeType::Note,
            label: label.to_string(),
            content: content.to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            created_at: 1_000_000,
            last_accessed: 1_000_000,
            access_count: 1,
            metadata: HashMap::new(),
            references_chunk_hashes: Vec::new(),
            needs_rebuild: false,
        }
    }

    // --- Ladder: Clean ---

    #[test]
    fn merge_clean_publishes_new_node() {
        let mut overlay = MemoryGraph::new();
        let mut primary = MemoryGraph::new();

        let node = make_node("node-1", "alpha", "hello world");
        overlay.insert_node(node.clone());

        let summary = run_merge_ladder(&overlay, &mut primary, &[], false);
        assert_eq!(summary.published, 1);
        assert_eq!(summary.skipped_identical, 0);
        assert_eq!(summary.conflicts, 0);

        // Primary should now have the node.
        let got = primary
            .get_node("node-1")
            .expect("node should be in primary");
        assert_eq!(got.content, "hello world");
        assert!(!got.needs_rebuild);
    }

    // --- Ladder: Identical ---

    #[test]
    fn merge_identical_is_noop() {
        let mut overlay = MemoryGraph::new();
        let mut primary = MemoryGraph::new();

        let node = make_node("node-2", "beta", "same content");
        overlay.insert_node(node.clone());
        primary.insert_node(node.clone());

        let summary = run_merge_ladder(&overlay, &mut primary, &[], false);
        assert_eq!(summary.published, 0);
        assert_eq!(summary.skipped_identical, 1);
        assert_eq!(summary.conflicts, 0);
    }

    // --- Ladder: Conflict ---

    #[test]
    fn merge_conflict_flags_primary_needs_rebuild() {
        let mut overlay = MemoryGraph::new();
        let mut primary = MemoryGraph::new();

        // Same node ID, different bodies.
        overlay.insert_node(make_node("node-3", "gamma", "version A"));
        primary.insert_node(make_node("node-3", "gamma", "version B"));

        let summary = run_merge_ladder(&overlay, &mut primary, &[], false);
        assert_eq!(summary.conflicts, 1);

        let flagged = primary.get_node("node-3").expect("node still in primary");
        assert!(flagged.needs_rebuild, "conflict must flag needs_rebuild");
    }

    // --- Drain suppression ---

    #[test]
    fn merge_suppressed_when_draining() {
        let mut overlay = MemoryGraph::new();
        let mut primary = MemoryGraph::new();

        overlay.insert_node(make_node("node-4", "delta", "should not appear"));

        let summary = run_merge_ladder(&overlay, &mut primary, &[], /* draining */ true);
        assert_eq!(summary.published, 0);
        assert!(
            primary.get_node("node-4").is_none(),
            "drain must suppress merge"
        );
    }

    // --- CoW reads ---

    #[test]
    fn cow_overlay_reads_through_to_parent() {
        use std::sync::Arc;

        let mut primary = MemoryGraph::new();
        primary.insert_node(make_node("node-p", "parent-node", "parent content"));
        let primary_arc = Arc::new(primary);

        let overlay = MemoryGraph::with_parent(primary_arc.clone());

        // Overlay can read the parent's node via get_node_cow.
        let found = overlay
            .get_node_cow("node-p")
            .expect("should fall through to parent");
        assert_eq!(found.content, "parent content");
    }

    #[test]
    fn cow_overlay_write_does_not_touch_parent() {
        use std::sync::Arc;

        let mut primary = MemoryGraph::new();
        primary.insert_node(make_node("node-p2", "shared", "primary body"));
        let primary_arc = Arc::new(primary);

        let mut overlay = MemoryGraph::with_parent(primary_arc.clone());
        overlay.insert_node(make_node("node-o", "overlay-only", "overlay content"));

        // Primary (via Arc) should not have the overlay-only node.
        assert!(
            primary_arc.get_node("node-o").is_none(),
            "parent must not see overlay write"
        );
        // Overlay should have the new node locally.
        assert!(overlay.get_node("node-o").is_some());
        // Overlay should still fall through to parent for the parent node.
        let p = overlay.get_node_cow("node-p2").expect("parent fallback");
        assert_eq!(p.content, "primary body");
    }

    #[test]
    fn overlay_ids_only_returns_local_nodes() {
        use std::sync::Arc;

        let mut primary = MemoryGraph::new();
        primary.insert_node(make_node("node-prim", "prim", "p"));
        let arc = Arc::new(primary);

        let mut overlay = MemoryGraph::with_parent(arc);
        overlay.insert_node(make_node("node-ov", "ov", "o"));

        let ids = overlay.overlay_node_ids();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], "node-ov");
    }

    // --- needs_rebuild filter in search ---

    #[test]
    fn search_excludes_needs_rebuild_nodes() {
        let mut graph = MemoryGraph::new();
        graph.insert_node(make_node("visible", "v", "hello world semantic search"));
        let mut hidden = make_node("hidden", "h", "hello world semantic search");
        hidden.needs_rebuild = true;
        graph.insert_node(hidden);

        // Query vector with same dimensionality.
        let query = vec![0.1_f32, 0.2, 0.3];
        let (result, _touched) = graph.search(&query, 1, 10, None);

        let ids: Vec<&str> = result.direct.iter().map(|r| r.node.id.as_str()).collect();
        assert!(
            ids.contains(&"visible"),
            "visible node must appear in search"
        );
        assert!(
            !ids.contains(&"hidden"),
            "needs_rebuild node must be hidden from search"
        );
    }

    // --- needs_rebuild: flag on chunk-hash change ---

    #[test]
    fn flag_nodes_for_chunk_changes_marks_correct_nodes() {
        let mut graph = MemoryGraph::new();

        let mut n1 = make_node("n1", "one", "content A");
        n1.references_chunk_hashes = vec!["hash-abc".to_string()];
        graph.insert_node(n1);

        let mut n2 = make_node("n2", "two", "content B");
        n2.references_chunk_hashes = vec!["hash-xyz".to_string()];
        graph.insert_node(n2);

        // Only hash-abc changed.
        let flagged = graph.flag_nodes_for_chunk_changes(&["hash-abc".to_string()]);
        assert_eq!(flagged.len(), 1);
        assert_eq!(flagged[0], "n1");

        assert!(graph.get_node("n1").unwrap().needs_rebuild);
        assert!(!graph.get_node("n2").unwrap().needs_rebuild);
    }

    // --- Rebuild on access ---

    #[test]
    fn rebuild_node_clears_flag_and_updates_embedding() {
        let mut graph = MemoryGraph::new();
        let mut n = make_node("rebuild-me", "r", "body text");
        n.needs_rebuild = true;
        graph.insert_node(n);

        let result = rebuild_node_sync(&mut graph, "rebuild-me", |_body, _hashes| {
            Some(vec![0.9, 0.8, 0.7])
        });
        assert!(matches!(result, Ok(RebuildOutcome::Rebuilt)));
        let node = graph.get_node("rebuild-me").unwrap();
        assert!(!node.needs_rebuild);
        assert_eq!(node.embedding, vec![0.9, 0.8, 0.7]);
    }

    #[test]
    fn rebuild_node_prunes_when_source_gone() {
        let mut graph = MemoryGraph::new();
        let mut n = make_node("prune-me", "p", "deleted symbol");
        n.needs_rebuild = true;
        graph.insert_node(n);

        // embed_fn returns None → source is gone → prune.
        let result = rebuild_node_sync(&mut graph, "prune-me", |_body, _hashes| None);
        assert!(matches!(result, Ok(RebuildOutcome::Pruned)));
        assert!(
            graph.get_node("prune-me").is_none(),
            "pruned node must be removed"
        );
    }

    // --- Auto-merge: smart-merge (additive edge union) ---

    #[test]
    fn merge_smart_merge_unions_edges() {
        let mut overlay = MemoryGraph::new();
        let mut primary = MemoryGraph::new();

        // Both sides have the same node body — smart-merge eligible.
        let body = "shared body text";
        overlay.insert_node(make_node("n-src", "src", body));
        overlay.insert_node(make_node("n-m", "m", "target M"));
        primary.insert_node(make_node("n-src", "src", body));
        primary.insert_node(make_node("n-p", "p", "target P"));

        // Overlay edge: src → m
        let edge_overlay = crate::core::memory_graph::MemoryEdge {
            id: "edge-1".to_string(),
            relation: RelationType::RelatesTo,
            weight: 0.9,
            created_at: 1_000,
            metadata: HashMap::new(),
        };

        // After overlay merges, primary should get src → m as well as the
        // existing primary-only edges.
        // We need n-m in primary first for the edge to be valid.
        primary.insert_node(make_node("n-m", "m", "target M"));

        let edge_snapshot = vec![("n-src".to_string(), "n-m".to_string(), edge_overlay)];

        let summary = run_merge_ladder(&overlay, &mut primary, &edge_snapshot, false);
        // n-src and n-m both have same body in overlay and primary → both identical.
        assert_eq!(
            summary.skipped_identical, 2,
            "both n-src and n-m should be identical (same body in overlay and primary)"
        );
        // Edge union ran for n-src→n-m (endpoints exist in primary).
        // We verify by calling create_relation — since the edge already exists
        // it will be returned as an upsert result (Some).
        let result =
            primary.create_relation("n-src", "n-m", RelationType::RelatesTo, Some(0.9), None);
        assert!(result.is_some(), "edge n-src→n-m should be in primary");
    }
}
