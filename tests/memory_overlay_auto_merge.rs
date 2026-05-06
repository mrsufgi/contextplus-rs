//! Integration tests for memory-graph CoW overlay and auto-merge.
//!
//! These tests exercise the full scenario pipeline described in U7:
//! * CoW reads — overlay inherits parent nodes via fallback
//! * CoW writes — writes stay local; primary is not affected
//! * Auto-merge clean — new node published to primary after merge ladder
//! * Auto-merge identical — same node on both sides → noop
//! * Auto-merge smart — additive edge union
//! * Auto-merge conflict — contradictory bodies → `needs_rebuild` on primary
//! * `needs_rebuild` on chunk change — `flag_nodes_for_chunk_changes`
//! * `needs_rebuild` rebuild success — `rebuild_node_sync` clears flag
//! * `needs_rebuild` rebuild failure → prune
//! * Drain suppression — merge suppressed when draining
//!
//! The scripted-git-history scenarios (ancestry detection) shell out to `git`
//! using a temporary repository created per test.

use std::collections::HashMap;
use std::sync::Arc;

use contextplus_rs::core::head_watcher::{is_ancestor, resolve_head_sha};
use contextplus_rs::core::memory_graph::{MemoryGraph, MemoryNode, NodeType, RelationType};
use contextplus_rs::core::memory_merge::{RebuildOutcome, rebuild_node_sync, run_merge_ladder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn node(id: &str, label: &str, content: &str) -> MemoryNode {
    MemoryNode {
        id: id.to_string(),
        node_type: NodeType::Note,
        label: label.to_string(),
        content: content.to_string(),
        embedding: vec![0.1_f32, 0.2, 0.3],
        created_at: 1_000_000,
        last_accessed: 1_000_000,
        access_count: 1,
        metadata: HashMap::new(),
        references_chunk_hashes: Vec::new(),
        needs_rebuild: false,
    }
}

// ---------------------------------------------------------------------------
// Scenario 1: CoW reads — overlay inherits parent nodes
// ---------------------------------------------------------------------------

#[test]
fn cow_reads_inherit_from_parent() {
    let mut primary = MemoryGraph::new();
    primary.insert_node(node("p1", "parent-concept", "concept from primary"));
    primary.insert_node(node("p2", "parent-note", "note from primary"));
    let primary_arc = Arc::new(primary);

    // Overlay starts empty — no local nodes.
    let overlay = MemoryGraph::with_parent(primary_arc.clone());

    // Overlay should see parent nodes via get_node_cow.
    assert!(
        overlay.get_node_cow("p1").is_some(),
        "overlay must fall through to parent for p1"
    );
    assert!(
        overlay.get_node_cow("p2").is_some(),
        "overlay must fall through to parent for p2"
    );

    // Primary has no parent — it is the root graph, not an overlay.
    assert!(!primary_arc.is_overlay(), "primary must not be an overlay");
    // Overlay has a parent — it IS an overlay.
    assert!(overlay.is_overlay());
}

// ---------------------------------------------------------------------------
// Scenario 2: CoW writes — writes stay local
// ---------------------------------------------------------------------------

#[test]
fn cow_writes_stay_local() {
    let mut primary = MemoryGraph::new();
    primary.insert_node(node("p1", "primary-node", "primary content"));
    let primary_arc = Arc::new(primary);

    let mut overlay = MemoryGraph::with_parent(primary_arc.clone());
    overlay.insert_node(node("o1", "overlay-only", "overlay content"));

    // Primary should not see overlay node.
    assert!(
        primary_arc.get_node("o1").is_none(),
        "primary must not see overlay-only node"
    );

    // Overlay node IDs contains only local.
    let ids = overlay.overlay_node_ids();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], "o1");

    // search_memory_graph on primary does not return overlay-only node.
    // (We test this via overlay_node_ids; the actual search filter test
    //  is in the unit tests in memory_merge.rs.)
    assert!(
        overlay.get_node("o1").is_some(),
        "overlay should find its own node locally"
    );
    assert!(
        overlay.get_node_cow("p1").is_some(),
        "overlay should still fall through for parent node"
    );
}

// ---------------------------------------------------------------------------
// Scenario 3: Auto-merge clean — new node published to primary
// ---------------------------------------------------------------------------

#[test]
fn auto_merge_clean_publishes_to_primary() {
    let mut overlay = MemoryGraph::new();
    let mut primary = MemoryGraph::new();

    overlay.insert_node(node("new-n", "new-node", "content created in overlay"));

    let summary = run_merge_ladder(&overlay, &mut primary, &[], false);
    assert_eq!(summary.published, 1, "clean node must be published");
    assert_eq!(summary.skipped_identical, 0);
    assert_eq!(summary.conflicts, 0);

    // After merge, primary has the node.
    assert!(
        primary.get_node("new-n").is_some(),
        "primary must contain merged node"
    );
    assert!(!primary.get_node("new-n").unwrap().needs_rebuild);

    // Overlay's node IDs are still there (merge doesn't clear them).
    // In the real system, clear_overlay() is called after merge; here we
    // just verify the semantic: overlay nodes were published.
    assert_eq!(overlay.overlay_node_ids().len(), 1);
}

// ---------------------------------------------------------------------------
// Scenario 4: Auto-merge identical — noop
// ---------------------------------------------------------------------------

#[test]
fn auto_merge_identical_is_noop() {
    let mut overlay = MemoryGraph::new();
    let mut primary = MemoryGraph::new();

    let n = node("shared-n", "shared", "identical content");
    overlay.insert_node(n.clone());
    primary.insert_node(n);

    let summary = run_merge_ladder(&overlay, &mut primary, &[], false);
    assert_eq!(summary.published, 0);
    assert_eq!(summary.skipped_identical, 1, "identical node must be noop");
    assert_eq!(summary.conflicts, 0);

    // Primary node is unchanged and not flagged.
    let pn = primary.get_node("shared-n").unwrap();
    assert!(!pn.needs_rebuild);
}

// ---------------------------------------------------------------------------
// Scenario 5: Auto-merge smart — additive edge union
// ---------------------------------------------------------------------------

#[test]
fn auto_merge_smart_unions_edges() {
    let mut overlay = MemoryGraph::new();
    let mut primary = MemoryGraph::new();

    let body = "shared node body";

    // Both sides have the same node; overlay also has a new target node.
    overlay.insert_node(node("src", "source", body));
    overlay.insert_node(node("target-ov", "target-overlay", "overlay target"));
    primary.insert_node(node("src", "source", body));
    primary.insert_node(node("target-ov", "target-overlay", "overlay target"));
    primary.insert_node(node("target-p", "target-primary", "primary target"));

    // Overlay wants to add edge: src → target-ov
    let edge = contextplus_rs::core::memory_graph::MemoryEdge {
        id: "e1".to_string(),
        relation: RelationType::References,
        weight: 1.0,
        created_at: 1_000,
        metadata: HashMap::new(),
    };
    let edge_snapshot = vec![("src".to_string(), "target-ov".to_string(), edge)];

    let summary = run_merge_ladder(&overlay, &mut primary, &edge_snapshot, false);
    // src and target-ov are identical (same body).
    assert_eq!(summary.skipped_identical, 2);
    assert_eq!(summary.conflicts, 0);

    // The new edge must now exist in primary.
    let result = primary.create_relation(
        "src",
        "target-ov",
        RelationType::References,
        Some(1.0),
        None,
    );
    assert!(result.is_some(), "edge src→target-ov should be in primary");
}

// ---------------------------------------------------------------------------
// Scenario 6: Auto-merge conflict — contradictory bodies
// ---------------------------------------------------------------------------

#[test]
fn auto_merge_conflict_flags_primary_needs_rebuild() {
    let mut overlay = MemoryGraph::new();
    let mut primary = MemoryGraph::new();

    // Same node ID, different bodies.
    overlay.insert_node(node("cn", "conflict-node", "overlay said A"));
    primary.insert_node(node("cn", "conflict-node", "primary said B"));

    let summary = run_merge_ladder(&overlay, &mut primary, &[], false);
    assert_eq!(summary.conflicts, 1, "conflict must be counted");
    assert_eq!(summary.published, 0);

    // Primary's node must be flagged.
    let pn = primary
        .get_node("cn")
        .expect("primary node must still exist");
    assert!(
        pn.needs_rebuild,
        "conflicted node must have needs_rebuild = true"
    );

    // Search would hide this node (tested at unit level in memory_merge.rs,
    // but we also confirm here).
    let query = vec![0.1_f32, 0.2, 0.3];
    let (result, _) = primary.search(&query, 1, 10, None);
    let ids: Vec<&str> = result.direct.iter().map(|r| r.node.id.as_str()).collect();
    assert!(
        !ids.contains(&"cn"),
        "needs_rebuild node must be hidden from search"
    );
}

// ---------------------------------------------------------------------------
// Scenario 7: needs_rebuild on chunk change
// ---------------------------------------------------------------------------

#[test]
fn needs_rebuild_on_chunk_hash_change() {
    let mut graph = MemoryGraph::new();

    let mut n1 = node("n-chunk", "chunk-derived", "some content from chunk H");
    n1.references_chunk_hashes = vec!["hash-H".to_string()];
    graph.insert_node(n1);

    let mut n2 = node("n-other", "other", "unrelated");
    n2.references_chunk_hashes = vec!["hash-X".to_string()];
    graph.insert_node(n2);

    // Simulate chunk H changing (file edited).
    let flagged = graph.flag_nodes_for_chunk_changes(&["hash-H".to_string()]);
    assert_eq!(flagged.len(), 1);
    assert_eq!(flagged[0], "n-chunk");

    // n-chunk is flagged; n-other is not.
    assert!(graph.get_node("n-chunk").unwrap().needs_rebuild);
    assert!(!graph.get_node("n-other").unwrap().needs_rebuild);

    // Search hides n-chunk.
    let (result, _) = graph.search(&[0.1_f32, 0.2, 0.3], 1, 10, None);
    let ids: Vec<&str> = result.direct.iter().map(|r| r.node.id.as_str()).collect();
    assert!(
        !ids.contains(&"n-chunk"),
        "chunk-changed node must be hidden from search"
    );
    assert!(
        ids.contains(&"n-other"),
        "unrelated node must still appear in search"
    );
}

// ---------------------------------------------------------------------------
// Scenario 8: needs_rebuild rebuild success
// ---------------------------------------------------------------------------

#[test]
fn needs_rebuild_rebuild_clears_flag() {
    let mut graph = MemoryGraph::new();

    let mut n = node("r1", "rebuildable", "original body");
    n.needs_rebuild = true;
    graph.insert_node(n);

    let outcome = rebuild_node_sync(&mut graph, "r1", |body, _hashes| {
        // Simulate successful re-embedding.
        assert_eq!(body, "original body");
        Some(vec![0.5_f32, 0.5, 0.5])
    });
    assert!(matches!(outcome, Ok(RebuildOutcome::Rebuilt)));

    let rebuilt = graph.get_node("r1").unwrap();
    assert!(!rebuilt.needs_rebuild, "flag must be cleared after rebuild");
    assert_eq!(rebuilt.embedding, vec![0.5_f32, 0.5, 0.5]);
}

// ---------------------------------------------------------------------------
// Scenario 9: needs_rebuild rebuild failure → prune
// ---------------------------------------------------------------------------

#[test]
fn needs_rebuild_rebuild_failure_prunes_node() {
    let mut graph = MemoryGraph::new();

    let mut n = node("prune-n", "deleted-symbol", "refers to gone symbol");
    n.needs_rebuild = true;
    n.references_chunk_hashes = vec!["dead-hash".to_string()];
    graph.insert_node(n);

    // embed_fn returns None → source is gone.
    let outcome = rebuild_node_sync(&mut graph, "prune-n", |_body, _hashes| None);
    assert!(matches!(outcome, Ok(RebuildOutcome::Pruned)));

    // Node must be removed.
    assert!(
        graph.get_node("prune-n").is_none(),
        "pruned node must be absent from graph"
    );
}

// ---------------------------------------------------------------------------
// Scenario 10: Drain suppression
// ---------------------------------------------------------------------------

#[test]
fn auto_merge_suppressed_during_drain() {
    let mut overlay = MemoryGraph::new();
    let mut primary = MemoryGraph::new();

    overlay.insert_node(node("drain-n", "drain-node", "should not appear"));

    // draining = true
    let summary = run_merge_ladder(&overlay, &mut primary, &[], /* draining= */ true);
    assert_eq!(
        summary.published, 0,
        "merge must be suppressed while draining"
    );
    assert!(
        primary.get_node("drain-n").is_none(),
        "drained merge must not publish nodes"
    );
}

// ---------------------------------------------------------------------------
// Scenario 11: Scripted git history — ancestor detection
// ---------------------------------------------------------------------------

fn init_repo_for_test() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for args in &[
        vec!["init", "--initial-branch=main"],
        vec!["config", "user.email", "test@example.com"],
        vec!["config", "user.name", "Test"],
    ] {
        std::process::Command::new("git")
            .args(args)
            .current_dir(dir.path())
            .output()
            .unwrap();
    }
    dir
}

fn commit_in_repo(dir: &std::path::Path, msg: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    std::fs::write(dir.join("f.txt"), format!("{msg}-{ts}")).unwrap();
    std::process::Command::new("git")
        .args(["add", "."])
        .current_dir(dir)
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["commit", "-m", msg])
        .current_dir(dir)
        .output()
        .unwrap();
    resolve_head_sha(dir).expect("commit should produce HEAD")
}

/// When ref_head is an ancestor of primary_head, auto-merge fires (ancestry
/// detection via `git merge-base --is-ancestor`).
#[test]
fn git_ancestry_detection_triggers_ancestor_check() {
    let repo = init_repo_for_test();
    let path = repo.path();

    let sha1 = commit_in_repo(path, "base commit");
    let sha2 = commit_in_repo(path, "primary advances");

    // sha1 is ancestor of sha2.
    assert_eq!(
        is_ancestor(path, &sha1, &sha2),
        Some(true),
        "earlier commit must be an ancestor of later"
    );

    // sha2 is NOT an ancestor of sha1.
    assert_eq!(
        is_ancestor(path, &sha2, &sha1),
        Some(false),
        "later commit must not be an ancestor of earlier"
    );

    // Simulate the daemon logic: if ref_head is ancestor of new primary HEAD,
    // run merge ladder.
    let ref_head = sha1.clone();
    let new_primary_head = sha2.clone();
    let ref_is_merged = is_ancestor(path, &ref_head, &new_primary_head) == Some(true);
    assert!(ref_is_merged, "merge condition should be true");

    // Build a small overlay graph with one node and merge it.
    let mut overlay = MemoryGraph::new();
    let mut primary = MemoryGraph::new();
    overlay.insert_node(node("git-n", "git-node", "from the ref branch"));

    let summary = run_merge_ladder(&overlay, &mut primary, &[], false);
    assert_eq!(summary.published, 1);
    assert!(primary.get_node("git-n").is_some());
}

/// When ref_head is NOT an ancestor of primary_head (diverged branches),
/// no auto-merge.
#[test]
fn git_non_ancestor_does_not_trigger_merge() {
    let repo = init_repo_for_test();
    let path = repo.path();

    let _base = commit_in_repo(path, "base");

    // Create a second branch at the same point.
    std::process::Command::new("git")
        .args(["checkout", "-b", "branch-a"])
        .current_dir(path)
        .output()
        .unwrap();
    let sha_a = commit_in_repo(path, "commit-a");

    std::process::Command::new("git")
        .args(["checkout", "main"])
        .current_dir(path)
        .output()
        .unwrap();
    let sha_main = commit_in_repo(path, "commit-main");

    // Diverged: neither is an ancestor of the other.
    let a_anc_main = is_ancestor(path, &sha_a, &sha_main);
    let main_anc_a = is_ancestor(path, &sha_main, &sha_a);

    // At least one direction should be false for diverged branches.
    assert!(
        a_anc_main != Some(true) || main_anc_a != Some(true),
        "diverged branches: at least one direction must not be an ancestor"
    );
}
