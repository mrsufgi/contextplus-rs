// In-memory property graph backed by petgraph with rkyv binary persistence.
// Supports temporal decay, auto-similarity linking, BFS traversal, and pruning.

use petgraph::Direction;
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Notify, RwLock};

use crate::error::{ContextPlusError, Result};

// --- Constants ---

const GRAPH_FILE_JSON: &str = "memory-graph.json";
const CACHE_DIR: &str = ".mcp_data";
const DECAY_LAMBDA: f64 = 0.05;
const SIMILARITY_THRESHOLD: f64 = 0.72;
const STALE_THRESHOLD: f64 = 0.15;
const ORPHAN_AGE_DAYS: u64 = 7;

// --- Types ---

/// Node types in the memory graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    Concept,
    File,
    Symbol,
    Note,
}

impl NodeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeType::Concept => "concept",
            NodeType::File => "file",
            NodeType::Symbol => "symbol",
            NodeType::Note => "note",
        }
    }

    pub fn parse_str(s: &str) -> Option<NodeType> {
        match s {
            "concept" => Some(NodeType::Concept),
            "file" => Some(NodeType::File),
            "symbol" => Some(NodeType::Symbol),
            "note" => Some(NodeType::Note),
            _ => None,
        }
    }
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Relation types for edges in the memory graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    RelatesTo,
    DependsOn,
    Implements,
    References,
    SimilarTo,
    Contains,
}

impl RelationType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RelationType::RelatesTo => "relates_to",
            RelationType::DependsOn => "depends_on",
            RelationType::Implements => "implements",
            RelationType::References => "references",
            RelationType::SimilarTo => "similar_to",
            RelationType::Contains => "contains",
        }
    }

    pub fn parse_str(s: &str) -> Option<RelationType> {
        match s {
            "relates_to" => Some(RelationType::RelatesTo),
            "depends_on" => Some(RelationType::DependsOn),
            "implements" => Some(RelationType::Implements),
            "references" => Some(RelationType::References),
            "similar_to" => Some(RelationType::SimilarTo),
            "contains" => Some(RelationType::Contains),
            _ => None,
        }
    }
}

impl std::fmt::Display for RelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A node in the memory graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: String,
    pub node_type: NodeType,
    pub label: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub created_at: u64,
    pub last_accessed: u64,
    pub access_count: u32,
    pub metadata: HashMap<String, String>,
}

/// An edge in the memory graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub id: String,
    pub relation: RelationType,
    pub weight: f32,
    pub created_at: u64,
    pub metadata: HashMap<String, String>,
}

/// Result of a graph traversal operation.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    pub node: MemoryNode,
    pub depth: usize,
    pub path_relations: Vec<String>,
    pub relevance_score: f64,
}

/// Result of a graph search operation.
#[derive(Debug, Clone)]
pub struct GraphSearchResult {
    pub direct: Vec<TraversalResult>,
    pub neighbors: Vec<TraversalResult>,
    pub total_nodes: usize,
    pub total_edges: usize,
}

/// Stats about the graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub nodes: usize,
    pub edges: usize,
    pub types: HashMap<String, usize>,
    pub relations: HashMap<String, usize>,
}

/// Result of a prune operation.
#[derive(Debug, Clone)]
pub struct PruneResult {
    pub removed: usize,
    pub remaining_edges: usize,
}

/// Result of an interlinked context operation.
#[derive(Debug, Clone)]
pub struct InterlinkResult {
    pub nodes: Vec<MemoryNode>,
    pub edges: Vec<MemoryEdge>,
}

// --- Core Graph ---

/// In-memory graph structure backed by petgraph's StableGraph.
/// StableGraph preserves node/edge indices on removal.
pub struct MemoryGraph {
    graph: StableGraph<MemoryNode, MemoryEdge>,
    /// Map (label, node_type_str) -> NodeIndex for fast lookup
    node_index: HashMap<(String, String), NodeIndex>,
    /// Map node_id -> NodeIndex
    id_index: HashMap<String, NodeIndex>,
    dirty: bool,
}

impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            graph: StableGraph::new(),
            node_index: HashMap::new(),
            id_index: HashMap::new(),
            dirty: false,
        }
    }

    /// Generate a unique ID with prefix.
    fn generate_id(prefix: &str) -> String {
        let ts = now_millis();
        let rand_part: u32 = rand::random();
        format!("{}-{}-{:06x}", prefix, ts, rand_part & 0xFFFFFF)
    }

    /// Get a node by its ID.
    pub fn get_node(&self, id: &str) -> Option<&MemoryNode> {
        self.id_index
            .get(id)
            .and_then(|&idx| self.graph.node_weight(idx))
    }

    /// Get a mutable node by its ID.
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut MemoryNode> {
        self.id_index
            .get(id)
            .copied()
            .and_then(|idx| self.graph.node_weight_mut(idx))
    }

    /// Check if a node exists by ID.
    pub fn node_exists(&self, id: &str) -> bool {
        self.id_index.contains_key(id)
    }

    /// Find a node by label and type.
    pub fn find_node(&self, label: &str, node_type: &NodeType) -> Option<&MemoryNode> {
        let key = (label.to_string(), node_type.as_str().to_string());
        self.node_index
            .get(&key)
            .and_then(|&idx| self.graph.node_weight(idx))
    }

    /// Upsert a node: update if exists (by label+type), else create.
    pub fn upsert_node(
        &mut self,
        node_type: NodeType,
        label: &str,
        content: &str,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> MemoryNode {
        let key = (label.to_string(), node_type.as_str().to_string());

        if let Some(&idx) = self.node_index.get(&key)
            && let Some(node) = self.graph.node_weight_mut(idx)
        {
            node.content = content.to_string();
            node.last_accessed = now_millis();
            node.access_count += 1;
            node.embedding = embedding;
            if let Some(meta) = metadata {
                node.metadata.extend(meta);
            }
            self.dirty = true;
            return node.clone();
        }

        let now = now_millis();
        let node = MemoryNode {
            id: Self::generate_id("mn"),
            node_type: node_type.clone(),
            label: label.to_string(),
            content: content.to_string(),
            embedding,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            metadata: metadata.unwrap_or_default(),
        };

        let idx = self.graph.add_node(node.clone());
        self.node_index.insert(key, idx);
        self.id_index.insert(node.id.clone(), idx);
        self.dirty = true;
        node
    }

    /// Create a relation (edge) between two nodes by ID.
    /// Returns None if either node doesn't exist.
    /// Updates weight/metadata if a duplicate edge (same source, target, relation) exists.
    pub fn create_relation(
        &mut self,
        source_id: &str,
        target_id: &str,
        relation: RelationType,
        weight: Option<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Option<MemoryEdge> {
        let source_idx = *self.id_index.get(source_id)?;
        let target_idx = *self.id_index.get(target_id)?;

        // Check for existing duplicate edge
        let existing_edge = self
            .graph
            .edges_connecting(source_idx, target_idx)
            .find(|e| e.weight().relation == relation);

        if let Some(edge_ref) = existing_edge {
            let edge_idx = edge_ref.id();
            if let Some(edge) = self.graph.edge_weight_mut(edge_idx) {
                if let Some(w) = weight {
                    edge.weight = w;
                }
                if let Some(meta) = metadata {
                    edge.metadata.extend(meta);
                }
                self.dirty = true;
                return Some(edge.clone());
            }
        }

        let edge = MemoryEdge {
            id: Self::generate_id("me"),
            relation,
            weight: weight.unwrap_or(1.0),
            created_at: now_millis(),
            metadata: metadata.unwrap_or_default(),
        };

        self.graph.add_edge(source_idx, target_idx, edge.clone());
        self.dirty = true;
        Some(edge)
    }

    /// Get all edges connected to a node (both directions).
    fn edges_for_node(&self, node_id: &str) -> Vec<(MemoryEdge, String, String)> {
        let Some(&idx) = self.id_index.get(node_id) else {
            return Vec::new();
        };

        let mut edges = Vec::new();

        for edge_ref in self.graph.edges_directed(idx, Direction::Outgoing) {
            let target = self.graph.node_weight(edge_ref.target());
            if let Some(t) = target {
                edges.push((edge_ref.weight().clone(), node_id.to_string(), t.id.clone()));
            }
        }
        for edge_ref in self.graph.edges_directed(idx, Direction::Incoming) {
            let source = self.graph.node_weight(edge_ref.source());
            if let Some(s) = source {
                edges.push((edge_ref.weight().clone(), s.id.clone(), node_id.to_string()));
            }
        }

        edges
    }

    /// Get neighbor node ID from an edge, given the current node.
    fn get_neighbor_id<'a>(edge_source: &'a str, edge_target: &'a str, from_id: &str) -> &'a str {
        if edge_source == from_id {
            edge_target
        } else {
            edge_source
        }
    }

    /// Compute cosine similarity between two f32 vectors.
    /// Delegates to simsimd for SIMD-accelerated computation.
    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        crate::core::embeddings::cosine_similarity_simsimd(a, b) as f64
    }

    /// Compute temporal decay weight: weight * exp(-lambda * days_since_creation)
    fn decay_weight(edge: &MemoryEdge) -> f64 {
        let days = (now_millis().saturating_sub(edge.created_at)) as f64 / 86_400_000.0;
        edge.weight as f64 * (-DECAY_LAMBDA * days).exp()
    }

    /// Semantic search + BFS traversal.
    pub fn search(
        &mut self,
        query_vec: &[f32],
        max_depth: usize,
        top_k: usize,
        edge_filter: Option<&[RelationType]>,
    ) -> GraphSearchResult {
        // Score all nodes by index — no cloning until top_k is known.
        let mut scored: Vec<(NodeIndex, f64)> = self
            .graph
            .node_indices()
            .filter_map(|idx| {
                let node = self.graph.node_weight(idx)?;
                let score = Self::cosine(query_vec, &node.embedding);
                Some((idx, score))
            })
            .collect();

        if scored.is_empty() {
            return GraphSearchResult {
                direct: Vec::new(),
                neighbors: Vec::new(),
                total_nodes: 0,
                total_edges: 0,
            };
        }

        // Partial sort to find top_k
        let top_k = top_k.min(scored.len());
        scored.select_nth_unstable_by(top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Clone only top-k nodes
        let direct_hits: Vec<TraversalResult> = scored
            .iter()
            .filter_map(|(idx, score)| {
                let node = self.graph.node_weight(*idx)?.clone();
                // Update last_accessed
                if let Some(n) = self.get_node_mut(&node.id) {
                    n.last_accessed = now_millis();
                }
                Some(TraversalResult {
                    node,
                    depth: 0,
                    path_relations: Vec::new(),
                    relevance_score: (score * 1000.0).round() / 10.0,
                })
            })
            .collect();

        // BFS neighbors
        let mut neighbor_results: Vec<TraversalResult> = Vec::new();
        let mut visited: HashSet<String> = direct_hits.iter().map(|h| h.node.id.clone()).collect();

        for hit in &direct_hits {
            self.traverse_neighbors(
                &hit.node.id,
                query_vec,
                1,
                max_depth,
                std::slice::from_ref(&hit.node.label),
                &mut visited,
                &mut neighbor_results,
                edge_filter,
            );
        }

        neighbor_results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let truncated_neighbors: Vec<TraversalResult> =
            neighbor_results.into_iter().take(top_k * 2).collect();

        self.dirty = true;

        let total_nodes = self.graph.node_count();
        let total_edges = self.graph.edge_count();

        GraphSearchResult {
            direct: direct_hits,
            neighbors: truncated_neighbors,
            total_nodes,
            total_edges,
        }
    }

    /// BFS traversal collecting neighbor nodes with relevance scoring.
    #[allow(clippy::too_many_arguments)]
    fn traverse_neighbors(
        &mut self,
        node_id: &str,
        query_vec: &[f32],
        depth: usize,
        max_depth: usize,
        path_labels: &[String],
        visited: &mut HashSet<String>,
        results: &mut Vec<TraversalResult>,
        edge_filter: Option<&[RelationType]>,
    ) {
        if depth > max_depth {
            return;
        }

        let edges = self.edges_for_node(node_id);

        for (edge, source_id, target_id) in edges {
            if let Some(filter) = edge_filter
                && !filter.contains(&edge.relation)
            {
                continue;
            }

            let neighbor_id = Self::get_neighbor_id(&source_id, &target_id, node_id).to_string();
            if visited.contains(&neighbor_id) {
                continue;
            }

            let neighbor = match self.get_node(&neighbor_id) {
                Some(n) => n.clone(),
                None => continue,
            };

            visited.insert(neighbor_id.clone());

            let similarity = Self::cosine(query_vec, &neighbor.embedding);
            let edge_decay = Self::decay_weight(&edge);
            let relevance = similarity * 0.6 + (edge_decay / (edge.weight as f64).max(0.01)) * 0.4;

            let mut new_path = path_labels.to_vec();
            new_path.push(format!("--[{}]-->", edge.relation));
            new_path.push(neighbor.label.clone());

            results.push(TraversalResult {
                node: neighbor.clone(),
                depth,
                path_relations: new_path.clone(),
                relevance_score: (relevance * 1000.0).round() / 10.0,
            });

            // Update last_accessed
            if let Some(n) = self.get_node_mut(&neighbor_id) {
                n.last_accessed = now_millis();
            }

            self.traverse_neighbors(
                &neighbor_id,
                query_vec,
                depth + 1,
                max_depth,
                &new_path,
                visited,
                results,
                edge_filter,
            );
        }
    }

    /// Prune stale links (decayed weight < threshold) and orphan nodes.
    pub fn prune_stale_links(&mut self, threshold: Option<f64>) -> PruneResult {
        let cutoff = threshold.unwrap_or(STALE_THRESHOLD);
        let mut edges_to_remove = Vec::new();

        // Find stale edges
        for edge_idx in self.graph.edge_indices() {
            if let Some(edge) = self.graph.edge_weight(edge_idx)
                && Self::decay_weight(edge) < cutoff
            {
                edges_to_remove.push(edge_idx);
            }
        }

        for idx in &edges_to_remove {
            self.graph.remove_edge(*idx);
        }

        // Find orphan nodes: no edges, access_count <= 1, last_accessed > 7 days ago
        let now = now_millis();
        let age_threshold = ORPHAN_AGE_DAYS * 86_400_000;
        let mut nodes_to_remove = Vec::new();

        for node_idx in self.graph.node_indices() {
            let has_edges = self
                .graph
                .edges_directed(node_idx, Direction::Outgoing)
                .next()
                .is_some()
                || self
                    .graph
                    .edges_directed(node_idx, Direction::Incoming)
                    .next()
                    .is_some();

            if !has_edges
                && let Some(node) = self.graph.node_weight(node_idx)
                && node.access_count <= 1
                && now.saturating_sub(node.last_accessed) > age_threshold
            {
                nodes_to_remove.push((
                    node_idx,
                    node.id.clone(),
                    node.label.clone(),
                    node.node_type.clone(),
                ));
            }
        }

        for (idx, id, label, node_type) in &nodes_to_remove {
            self.graph.remove_node(*idx);
            self.id_index.remove(id);
            self.node_index
                .remove(&(label.clone(), node_type.as_str().to_string()));
        }

        self.dirty = true;
        let remaining_edges = self.graph.edge_count();

        PruneResult {
            removed: edges_to_remove.len() + nodes_to_remove.len(),
            remaining_edges,
        }
    }

    /// Add multiple interlinked context nodes, optionally auto-linking by similarity.
    #[allow(clippy::type_complexity)]
    pub fn add_interlinked_context(
        &mut self,
        items: Vec<(
            NodeType,
            String,
            String,
            Vec<f32>,
            Option<HashMap<String, String>>,
        )>,
        auto_link: bool,
    ) -> InterlinkResult {
        let mut created_nodes: Vec<MemoryNode> = Vec::new();

        for (node_type, label, content, embedding, metadata) in items {
            let node = self.upsert_node(node_type, &label, &content, embedding, metadata);
            created_nodes.push(node);
        }

        let mut created_edges: Vec<MemoryEdge> = Vec::new();

        if auto_link && created_nodes.len() > 1 {
            // Link new nodes to each other by similarity
            for i in 0..created_nodes.len() {
                for j in (i + 1)..created_nodes.len() {
                    let similarity =
                        Self::cosine(&created_nodes[i].embedding, &created_nodes[j].embedding);
                    if similarity >= SIMILARITY_THRESHOLD
                        && let Some(edge) = self.create_relation(
                            &created_nodes[i].id,
                            &created_nodes[j].id,
                            RelationType::SimilarTo,
                            Some(similarity as f32),
                            None,
                        )
                    {
                        created_edges.push(edge);
                    }
                }
            }

            // Link new nodes to existing nodes by similarity
            let new_ids: HashSet<String> = created_nodes.iter().map(|n| n.id.clone()).collect();
            let existing_nodes: Vec<MemoryNode> = self
                .graph
                .node_indices()
                .filter_map(|idx| self.graph.node_weight(idx))
                .filter(|n| !new_ids.contains(&n.id))
                .take(200)
                .cloned()
                .collect();

            for new_node in &created_nodes {
                for existing in &existing_nodes {
                    let similarity = Self::cosine(&new_node.embedding, &existing.embedding);
                    if similarity >= SIMILARITY_THRESHOLD
                        && let Some(edge) = self.create_relation(
                            &new_node.id,
                            &existing.id,
                            RelationType::SimilarTo,
                            Some(similarity as f32),
                            None,
                        )
                    {
                        created_edges.push(edge);
                    }
                }
            }
        }

        InterlinkResult {
            nodes: created_nodes,
            edges: created_edges,
        }
    }

    /// Retrieve nodes via BFS traversal from a specific starting node.
    pub fn retrieve_with_traversal(
        &mut self,
        start_node_id: &str,
        max_depth: usize,
        edge_filter: Option<&[RelationType]>,
    ) -> Vec<TraversalResult> {
        let start_node = match self.get_node(start_node_id) {
            Some(n) => n.clone(),
            None => return Vec::new(),
        };

        // Update access info
        if let Some(n) = self.get_node_mut(start_node_id) {
            n.last_accessed = now_millis();
            n.access_count += 1;
        }

        let mut results = vec![TraversalResult {
            node: start_node.clone(),
            depth: 0,
            path_relations: vec![start_node.label.clone()],
            relevance_score: 100.0,
        }];

        let mut visited = HashSet::new();
        visited.insert(start_node_id.to_string());

        self.collect_traversal(
            start_node_id,
            1,
            max_depth,
            std::slice::from_ref(&start_node.label),
            &mut visited,
            &mut results,
            edge_filter,
        );

        self.dirty = true;
        results
    }

    /// BFS traversal collecting nodes with depth-penalized scoring.
    #[allow(clippy::too_many_arguments)]
    fn collect_traversal(
        &mut self,
        node_id: &str,
        depth: usize,
        max_depth: usize,
        path_labels: &[String],
        visited: &mut HashSet<String>,
        results: &mut Vec<TraversalResult>,
        edge_filter: Option<&[RelationType]>,
    ) {
        if depth > max_depth {
            return;
        }

        let edges = self.edges_for_node(node_id);

        for (edge, source_id, target_id) in edges {
            if let Some(filter) = edge_filter
                && !filter.contains(&edge.relation)
            {
                continue;
            }

            let neighbor_id = Self::get_neighbor_id(&source_id, &target_id, node_id).to_string();
            if visited.contains(&neighbor_id) {
                continue;
            }

            let neighbor = match self.get_node(&neighbor_id) {
                Some(n) => n.clone(),
                None => continue,
            };

            visited.insert(neighbor_id.clone());

            // Update last_accessed
            if let Some(n) = self.get_node_mut(&neighbor_id) {
                n.last_accessed = now_millis();
            }

            let decayed = Self::decay_weight(&edge);
            let depth_penalty = 1.0 / (1.0 + depth as f64 * 0.3);
            let score = decayed * depth_penalty * 100.0;

            let mut new_path = path_labels.to_vec();
            new_path.push(format!("--[{}]-->", edge.relation));
            new_path.push(neighbor.label.clone());

            results.push(TraversalResult {
                node: neighbor.clone(),
                depth,
                path_relations: new_path.clone(),
                relevance_score: (score * 10.0).round() / 10.0,
            });

            self.collect_traversal(
                &neighbor_id,
                depth + 1,
                max_depth,
                &new_path,
                visited,
                results,
                edge_filter,
            );
        }
    }

    /// Delete a node by ID along with all edges that reference it.
    /// Returns `Some((label, node_type, edges_removed))` if the node existed,
    /// or `None` if the ID was not found (idempotent — not an error).
    pub fn delete_node(&mut self, node_id: &str) -> Option<(String, NodeType, usize)> {
        let &idx = self.id_index.get(node_id)?;

        // Collect all edge indices touching this node (incoming + outgoing).
        // Two separate collects avoid holding a borrow across the chain.
        let mut edges_to_remove: Vec<_> = self
            .graph
            .edges_directed(idx, Direction::Incoming)
            .map(|e| e.id())
            .collect();
        edges_to_remove.extend(
            self.graph
                .edges_directed(idx, Direction::Outgoing)
                .map(|e| e.id()),
        );

        let edge_count = edges_to_remove.len();
        for eid in edges_to_remove {
            self.graph.remove_edge(eid);
        }

        // Remove the node itself and clean up the two lookup indices.
        let node = self.graph.remove_node(idx).expect("index must exist");
        self.id_index.remove(node_id);
        self.node_index
            .remove(&(node.label.clone(), node.node_type.as_str().to_string()));

        self.dirty = true;
        Some((node.label, node.node_type, edge_count))
    }

    /// Get statistics about the graph.
    pub fn stats(&self) -> GraphStats {
        let mut types: HashMap<String, usize> = HashMap::new();
        let mut relations: HashMap<String, usize> = HashMap::new();

        for idx in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(idx) {
                *types
                    .entry(node.node_type.as_str().to_string())
                    .or_insert(0) += 1;
            }
        }
        for idx in self.graph.edge_indices() {
            if let Some(edge) = self.graph.edge_weight(idx) {
                *relations
                    .entry(edge.relation.as_str().to_string())
                    .or_insert(0) += 1;
            }
        }

        GraphStats {
            nodes: self.graph.node_count(),
            edges: self.graph.edge_count(),
            types,
            relations,
        }
    }

    /// Get all nodes (for embedding operations).
    pub fn all_nodes(&self) -> Vec<MemoryNode> {
        self.graph
            .node_indices()
            .filter_map(|idx| self.graph.node_weight(idx).cloned())
            .collect()
    }

    /// Check if the graph has been modified since last save.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark graph as clean (after saving).
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }
}

// --- Persistence ---

/// Thread-safe graph store with lazy loading and debounced persistence.
/// Uses RwLock so concurrent read-only operations (search, traverse) don't block each other.
///
/// Debounced save: after every mutation, a 500ms timer is (re-)started. When it
/// fires the dirty graph is written to disk. `flush()` persists immediately and
/// cancels any pending debounce — use it on graceful shutdown.
pub struct GraphStore {
    graphs: RwLock<HashMap<String, MemoryGraph>>,
    save_notify: Arc<Notify>,
    shutdown_notify: Arc<Notify>,
}

impl Default for GraphStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Debounce interval for saves after mutations.
const DEBOUNCE_MS: u64 = 500;

impl GraphStore {
    pub fn new() -> Self {
        Self {
            graphs: RwLock::new(HashMap::new()),
            save_notify: Arc::new(Notify::new()),
            shutdown_notify: Arc::new(Notify::new()),
        }
    }

    /// Spawn the background debounce task.
    pub fn spawn_debounce_task(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let store = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = store.save_notify.notified() => {
                        loop {
                            tokio::select! {
                                _ = tokio::time::sleep(tokio::time::Duration::from_millis(DEBOUNCE_MS)) => break,
                                _ = store.save_notify.notified() => continue,
                            }
                        }
                        if let Err(e) = store.persist_all_dirty().await {
                            tracing::warn!("Debounced graph save failed: {e}");
                        }
                    }
                    _ = store.shutdown_notify.notified() => {
                        if let Err(e) = store.persist_all_dirty().await {
                            tracing::warn!("Shutdown graph save failed: {e}");
                        }
                        return;
                    }
                }
            }
        })
    }

    fn schedule_save(&self) {
        self.save_notify.notify_one();
    }

    /// Get or load the graph for a given root directory.
    /// Automatically schedules a debounced save if the graph becomes dirty.
    pub async fn get_graph<F, R>(&self, root_dir: &str, f: F) -> Result<R>
    where
        F: FnOnce(&mut MemoryGraph) -> R,
    {
        let result;
        {
            let graphs = self.graphs.read().await;
            if graphs.contains_key(root_dir) {
                drop(graphs);
                let mut graphs = self.graphs.write().await;
                let graph = graphs.get_mut(root_dir).ok_or_else(|| {
                    ContextPlusError::Other("Graph not found after check".to_string())
                })?;
                let was_dirty = graph.is_dirty();
                result = f(graph);
                if !was_dirty && graph.is_dirty() {
                    self.schedule_save();
                }
                return Ok(result);
            }
        }
        let mut graphs = self.graphs.write().await;
        if !graphs.contains_key(root_dir) {
            let graph = load_graph_from_disk(root_dir).await?;
            graphs.insert(root_dir.to_string(), graph);
        }
        let graph = graphs.get_mut(root_dir).ok_or_else(|| {
            ContextPlusError::Other("Graph not found after insertion".to_string())
        })?;
        let was_dirty = graph.is_dirty();
        result = f(graph);
        if !was_dirty && graph.is_dirty() {
            self.schedule_save();
        }
        Ok(result)
    }

    /// Persist the graph for a root directory to disk.
    pub async fn persist(&self, root_dir: &str) -> Result<()> {
        let mut graphs = self.graphs.write().await;
        if let Some(graph) = graphs.get_mut(root_dir)
            && graph.is_dirty()
        {
            persist_graph_to_disk(root_dir, graph).await?;
            graph.mark_clean();
        }
        Ok(())
    }

    /// Persist ALL dirty graphs to disk.
    async fn persist_all_dirty(&self) -> Result<()> {
        let mut graphs = self.graphs.write().await;
        for (root_dir, graph) in graphs.iter_mut() {
            if graph.is_dirty() {
                persist_graph_to_disk(root_dir, graph).await?;
                graph.mark_clean();
            }
        }
        Ok(())
    }

    /// Flush: persist immediately and signal the debounce task to shut down.
    pub async fn flush(&self) -> Result<()> {
        self.persist_all_dirty().await?;
        self.shutdown_notify.notify_one();
        Ok(())
    }
}

/// Load a graph from JSON on disk.
async fn load_graph_from_disk(root_dir: &str) -> Result<MemoryGraph> {
    let path = Path::new(root_dir).join(CACHE_DIR).join(GRAPH_FILE_JSON);

    let content = match tokio::fs::read_to_string(&path).await {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(MemoryGraph::new());
        }
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "Failed to read memory graph file — file exists but is unreadable; \
                 returning error to prevent overwriting potentially valid data"
            );
            return Err(ContextPlusError::Io(e));
        }
    };

    let raw: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        ContextPlusError::Serialization(format!("Failed to parse graph JSON: {}", e))
    })?;

    let mut graph = MemoryGraph::new();

    // Load nodes
    if let Some(nodes_obj) = raw.get("nodes").and_then(|v| v.as_object()) {
        for (node_key, node_val) in nodes_obj {
            match serde_json::from_value::<MemoryNode>(node_val.clone()) {
                Ok(node) => {
                    let key = (node.label.clone(), node.node_type.as_str().to_string());
                    let idx = graph.graph.add_node(node.clone());
                    graph.node_index.insert(key, idx);
                    graph.id_index.insert(node.id.clone(), idx);
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        node_key = %node_key,
                        error = %e,
                        "Skipping malformed node in memory graph — data may be partially corrupted"
                    );
                }
            }
        }
    }

    // Load edges
    if let Some(edges_obj) = raw.get("edges").and_then(|v| v.as_object()) {
        for (_id, edge_val) in edges_obj {
            let source = edge_val
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let target = edge_val
                .get("target")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if let (Some(&source_idx), Some(&target_idx)) =
                (graph.id_index.get(source), graph.id_index.get(target))
                && let Ok(edge) = serde_json::from_value::<MemoryEdge>(edge_val.clone())
            {
                graph.graph.add_edge(source_idx, target_idx, edge);
            }
        }
    }

    Ok(graph)
}

/// Persist a graph to JSON on disk.
async fn persist_graph_to_disk(root_dir: &str, graph: &MemoryGraph) -> Result<()> {
    let dir_path = Path::new(root_dir).join(CACHE_DIR);
    tokio::fs::create_dir_all(&dir_path).await?;

    let path = dir_path.join(GRAPH_FILE_JSON);

    // Build serialized format
    let mut nodes: HashMap<String, MemoryNode> = HashMap::new();
    let mut edges: HashMap<String, serde_json::Value> = HashMap::new();

    for node_idx in graph.graph.node_indices() {
        if let Some(node) = graph.graph.node_weight(node_idx) {
            nodes.insert(node.id.clone(), node.clone());
        }
    }

    for edge_idx in graph.graph.edge_indices() {
        if let Some((source_idx, target_idx)) = graph.graph.edge_endpoints(edge_idx)
            && let (Some(edge), Some(source_node), Some(target_node)) = (
                graph.graph.edge_weight(edge_idx),
                graph.graph.node_weight(source_idx),
                graph.graph.node_weight(target_idx),
            )
        {
            let mut edge_json = serde_json::to_value(edge).unwrap_or_default();
            if let Some(obj) = edge_json.as_object_mut() {
                obj.insert(
                    "source".to_string(),
                    serde_json::Value::String(source_node.id.clone()),
                );
                obj.insert(
                    "target".to_string(),
                    serde_json::Value::String(target_node.id.clone()),
                );
            }
            edges.insert(edge.id.clone(), edge_json);
        }
    }

    let store = serde_json::json!({
        "nodes": nodes,
        "edges": edges,
    });

    let json = serde_json::to_string_pretty(&store).map_err(|e| {
        ContextPlusError::Serialization(format!("Failed to serialize graph: {}", e))
    })?;

    tokio::fs::write(&path, json).await?;
    Ok(())
}

/// Get current time in milliseconds since UNIX epoch.
fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(seed: f32) -> Vec<f32> {
        vec![seed, seed * 0.5, 1.0 - seed, seed * 0.3]
    }

    #[test]
    fn temporal_decay_formula_verification() {
        // decay_weight computes: weight * exp(-DECAY_LAMBDA * days)
        // For 10 days with weight 1.0: exp(-0.05 * 10) = exp(-0.5) ~ 0.6065
        let ten_days_ago = now_millis() - (10 * 86_400_000);
        let edge = MemoryEdge {
            id: "test".to_string(),
            relation: RelationType::RelatesTo,
            weight: 1.0,
            created_at: ten_days_ago,
            metadata: HashMap::new(),
        };
        let decayed = MemoryGraph::decay_weight(&edge);
        let expected = (-0.05_f64 * 10.0).exp(); // ~0.6065
        assert!(
            (decayed - expected).abs() < 0.02,
            "expected ~{:.4}, got {:.4}",
            expected,
            decayed
        );
    }

    #[test]
    fn temporal_decay_scales_with_weight() {
        let one_day_ago = now_millis() - 86_400_000;
        let edge = MemoryEdge {
            id: "test".to_string(),
            relation: RelationType::RelatesTo,
            weight: 2.0,
            created_at: one_day_ago,
            metadata: HashMap::new(),
        };
        let decayed = MemoryGraph::decay_weight(&edge);
        let expected = 2.0 * (-0.05_f64).exp(); // ~1.9025
        assert!(
            (decayed - expected).abs() < 0.02,
            "expected ~{:.4}, got {:.4}",
            expected,
            decayed
        );
    }

    #[test]
    fn memory_graph_new_is_empty() {
        let graph = MemoryGraph::new();
        assert_eq!(graph.graph.node_count(), 0);
        assert_eq!(graph.graph.edge_count(), 0);
        assert!(graph.node_index.is_empty());
        assert!(graph.id_index.is_empty());
        assert!(!graph.is_dirty());
    }

    #[test]
    fn memory_graph_default_equals_new() {
        let from_new = MemoryGraph::new();
        let from_default = MemoryGraph::default();
        assert_eq!(from_new.graph.node_count(), from_default.graph.node_count());
        assert_eq!(from_new.graph.edge_count(), from_default.graph.edge_count());
        assert_eq!(from_new.is_dirty(), from_default.is_dirty());
    }

    #[test]
    fn node_type_roundtrip() {
        for nt in [
            NodeType::Concept,
            NodeType::File,
            NodeType::Symbol,
            NodeType::Note,
        ] {
            let s = nt.as_str();
            let parsed = NodeType::parse_str(s);
            assert_eq!(parsed, Some(nt));
        }
    }

    #[test]
    fn relation_type_roundtrip() {
        for rt in [
            RelationType::RelatesTo,
            RelationType::DependsOn,
            RelationType::Implements,
            RelationType::References,
            RelationType::SimilarTo,
            RelationType::Contains,
        ] {
            let s = rt.as_str();
            let parsed = RelationType::parse_str(s);
            assert_eq!(parsed, Some(rt));
        }
    }

    #[test]
    fn upsert_creates_new_node() {
        let mut graph = MemoryGraph::new();
        let node = graph.upsert_node(
            NodeType::Concept,
            "auth",
            "authentication module",
            make_embedding(0.5),
            None,
        );
        assert_eq!(node.label, "auth");
        assert_eq!(node.access_count, 1);
        assert!(node.id.starts_with("mn-"));
    }

    #[test]
    fn upsert_updates_existing_node() {
        let mut graph = MemoryGraph::new();
        let first = graph.upsert_node(NodeType::Concept, "auth", "v1", make_embedding(0.5), None);
        let second = graph.upsert_node(NodeType::Concept, "auth", "v2", make_embedding(0.6), None);
        assert_eq!(first.id, second.id);
        assert_eq!(second.content, "v2");
        assert_eq!(second.access_count, 2);
    }

    #[test]
    fn upsert_different_types_are_separate() {
        let mut graph = MemoryGraph::new();
        let concept = graph.upsert_node(
            NodeType::Concept,
            "auth",
            "concept",
            make_embedding(0.5),
            None,
        );
        let file = graph.upsert_node(NodeType::File, "auth", "file", make_embedding(0.6), None);
        assert_ne!(concept.id, file.id);
    }

    #[test]
    fn create_relation_basic() {
        let mut graph = MemoryGraph::new();
        let a = graph.upsert_node(NodeType::Concept, "A", "node A", make_embedding(0.1), None);
        let b = graph.upsert_node(NodeType::Concept, "B", "node B", make_embedding(0.9), None);

        let edge = graph.create_relation(&a.id, &b.id, RelationType::DependsOn, None, None);
        assert!(edge.is_some());
        let e = edge.expect("edge should exist");
        assert_eq!(e.relation, RelationType::DependsOn);
        assert_eq!(e.weight, 1.0);
    }

    #[test]
    fn create_relation_invalid_node_returns_none() {
        let mut graph = MemoryGraph::new();
        let a = graph.upsert_node(NodeType::Concept, "A", "a", make_embedding(0.1), None);
        let edge = graph.create_relation(&a.id, "nonexistent", RelationType::DependsOn, None, None);
        assert!(edge.is_none());
    }

    #[test]
    fn create_relation_duplicate_updates() {
        let mut graph = MemoryGraph::new();
        let a = graph.upsert_node(NodeType::Concept, "A", "a", make_embedding(0.1), None);
        let b = graph.upsert_node(NodeType::Concept, "B", "b", make_embedding(0.9), None);

        let e1 = graph.create_relation(&a.id, &b.id, RelationType::DependsOn, Some(0.5), None);
        let e2 = graph.create_relation(&a.id, &b.id, RelationType::DependsOn, Some(0.8), None);
        assert!(e1.is_some());
        assert!(e2.is_some());
        // Should be same edge, updated weight
        assert_eq!(e1.expect("e1").id, e2.expect("e2").id);
        // Only 1 edge in graph
        assert_eq!(graph.graph.edge_count(), 1);
    }

    #[test]
    fn get_node_by_id() {
        let mut graph = MemoryGraph::new();
        let node = graph.upsert_node(
            NodeType::File,
            "main.rs",
            "entry",
            make_embedding(0.5),
            None,
        );
        let found = graph.get_node(&node.id);
        assert!(found.is_some());
        assert_eq!(found.expect("node").label, "main.rs");
    }

    #[test]
    fn find_node_by_label_and_type() {
        let mut graph = MemoryGraph::new();
        graph.upsert_node(
            NodeType::Symbol,
            "MyStruct",
            "a struct",
            make_embedding(0.5),
            None,
        );
        let found = graph.find_node("MyStruct", &NodeType::Symbol);
        assert!(found.is_some());
        let not_found = graph.find_node("MyStruct", &NodeType::File);
        assert!(not_found.is_none());
    }

    #[test]
    fn cosine_identical() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let sim = MemoryGraph::cosine(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = MemoryGraph::cosine(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn decay_weight_new_edge() {
        let edge = MemoryEdge {
            id: "test".to_string(),
            relation: RelationType::RelatesTo,
            weight: 1.0,
            created_at: now_millis(),
            metadata: HashMap::new(),
        };
        let decayed = MemoryGraph::decay_weight(&edge);
        // Should be very close to 1.0 for a brand new edge
        assert!((decayed - 1.0).abs() < 0.01);
    }

    #[test]
    fn decay_weight_old_edge() {
        let thirty_days_ago = now_millis() - (30 * 86_400_000);
        let edge = MemoryEdge {
            id: "test".to_string(),
            relation: RelationType::RelatesTo,
            weight: 1.0,
            created_at: thirty_days_ago,
            metadata: HashMap::new(),
        };
        let decayed = MemoryGraph::decay_weight(&edge);
        // exp(-0.05 * 30) = exp(-1.5) ~ 0.223
        assert!(decayed < 0.3);
        assert!(decayed > 0.1);
    }

    #[test]
    fn search_empty_graph() {
        let mut graph = MemoryGraph::new();
        let result = graph.search(&[1.0, 0.0, 0.0], 1, 5, None);
        assert!(result.direct.is_empty());
        assert_eq!(result.total_nodes, 0);
    }

    #[test]
    fn search_finds_similar_nodes() {
        let mut graph = MemoryGraph::new();
        graph.upsert_node(
            NodeType::Concept,
            "auth",
            "authentication",
            vec![1.0, 0.0, 0.0],
            None,
        );
        graph.upsert_node(
            NodeType::Concept,
            "db",
            "database",
            vec![0.0, 1.0, 0.0],
            None,
        );

        let result = graph.search(&[0.9, 0.1, 0.0], 0, 5, None);
        assert!(!result.direct.is_empty());
        // Auth should be the top hit
        assert_eq!(result.direct[0].node.label, "auth");
    }

    #[test]
    fn search_with_neighbors() {
        let mut graph = MemoryGraph::new();
        let a = graph.upsert_node(
            NodeType::Concept,
            "auth",
            "authentication",
            vec![1.0, 0.0, 0.0],
            None,
        );
        let b = graph.upsert_node(
            NodeType::Concept,
            "tokens",
            "JWT tokens",
            vec![0.5, 0.5, 0.0],
            None,
        );
        graph.create_relation(&a.id, &b.id, RelationType::DependsOn, None, None);

        let result = graph.search(&[0.9, 0.1, 0.0], 1, 1, None);
        assert_eq!(result.direct.len(), 1);
        assert_eq!(result.direct[0].node.label, "auth");
        // Should find tokens as neighbor
        assert!(!result.neighbors.is_empty());
    }

    #[test]
    fn prune_removes_nothing_from_fresh_graph() {
        let mut graph = MemoryGraph::new();
        let a = graph.upsert_node(NodeType::Concept, "A", "a", make_embedding(0.1), None);
        let b = graph.upsert_node(NodeType::Concept, "B", "b", make_embedding(0.9), None);
        graph.create_relation(&a.id, &b.id, RelationType::RelatesTo, None, None);

        let result = graph.prune_stale_links(None);
        assert_eq!(result.removed, 0);
        assert_eq!(result.remaining_edges, 1);
    }

    #[test]
    fn prune_removes_old_stale_edges() {
        let mut graph = MemoryGraph::new();
        let a = graph.upsert_node(NodeType::Concept, "A", "a", make_embedding(0.1), None);
        let b = graph.upsert_node(NodeType::Concept, "B", "b", make_embedding(0.9), None);

        // Create edge, then manually age it
        graph.create_relation(&a.id, &b.id, RelationType::RelatesTo, Some(0.1), None);

        // Manually set edge created_at to 100 days ago
        let edge_indices: Vec<_> = graph.graph.edge_indices().collect();
        for edge_idx in edge_indices {
            if let Some(edge) = graph.graph.edge_weight_mut(edge_idx) {
                edge.created_at = now_millis() - (100 * 86_400_000);
            }
        }

        let result = graph.prune_stale_links(None);
        // 1 edge removed. Nodes were just created (lastAccessed = now), so they won't be orphaned.
        assert!(result.removed >= 1);
    }

    #[test]
    fn retrieve_with_traversal_basic() {
        let mut graph = MemoryGraph::new();
        let a = graph.upsert_node(
            NodeType::Concept,
            "root",
            "root node",
            make_embedding(0.5),
            None,
        );
        let b = graph.upsert_node(
            NodeType::Concept,
            "child",
            "child node",
            make_embedding(0.6),
            None,
        );
        graph.create_relation(&a.id, &b.id, RelationType::Contains, None, None);

        let results = graph.retrieve_with_traversal(&a.id, 2, None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].node.label, "root");
        assert_eq!(results[0].depth, 0);
        assert_eq!(results[1].node.label, "child");
        assert_eq!(results[1].depth, 1);
    }

    #[test]
    fn retrieve_with_traversal_missing_node() {
        let mut graph = MemoryGraph::new();
        let results = graph.retrieve_with_traversal("nonexistent", 2, None);
        assert!(results.is_empty());
    }

    #[test]
    fn retrieve_with_edge_filter() {
        let mut graph = MemoryGraph::new();
        let a = graph.upsert_node(NodeType::Concept, "A", "a", make_embedding(0.1), None);
        let b = graph.upsert_node(NodeType::Concept, "B", "b", make_embedding(0.5), None);
        let c = graph.upsert_node(NodeType::Concept, "C", "c", make_embedding(0.9), None);
        graph.create_relation(&a.id, &b.id, RelationType::DependsOn, None, None);
        graph.create_relation(&a.id, &c.id, RelationType::Contains, None, None);

        let results = graph.retrieve_with_traversal(&a.id, 2, Some(&[RelationType::DependsOn]));
        // Should only find A -> B (depends_on), not A -> C (contains)
        assert_eq!(results.len(), 2);
        assert_eq!(results[1].node.label, "B");
    }

    #[test]
    fn add_interlinked_context_auto_links() {
        let mut graph = MemoryGraph::new();
        // Two very similar embeddings should auto-link
        let items = vec![
            (
                NodeType::Concept,
                "A".to_string(),
                "a".to_string(),
                vec![1.0_f32, 0.0, 0.0],
                None,
            ),
            (
                NodeType::Concept,
                "B".to_string(),
                "b".to_string(),
                vec![0.99_f32, 0.01, 0.0],
                None,
            ),
            (
                NodeType::Concept,
                "C".to_string(),
                "c".to_string(),
                vec![0.0_f32, 1.0, 0.0],
                None,
            ),
        ];

        let result = graph.add_interlinked_context(items, true);
        assert_eq!(result.nodes.len(), 3);
        // A and B are very similar (cosine > 0.72), should be linked
        // A/B and C are orthogonal, should not be linked
        assert!(!result.edges.is_empty());
    }

    #[test]
    fn add_interlinked_context_no_auto_link() {
        let mut graph = MemoryGraph::new();
        let items = vec![
            (
                NodeType::Concept,
                "A".to_string(),
                "a".to_string(),
                vec![1.0_f32, 0.0],
                None,
            ),
            (
                NodeType::Concept,
                "B".to_string(),
                "b".to_string(),
                vec![0.99_f32, 0.01],
                None,
            ),
        ];

        let result = graph.add_interlinked_context(items, false);
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.edges.len(), 0);
    }

    #[test]
    fn stats_basic() {
        let mut graph = MemoryGraph::new();
        graph.upsert_node(NodeType::Concept, "A", "a", make_embedding(0.1), None);
        graph.upsert_node(NodeType::File, "B", "b", make_embedding(0.5), None);
        let a = graph
            .find_node("A", &NodeType::Concept)
            .expect("A")
            .id
            .clone();
        let b = graph.find_node("B", &NodeType::File).expect("B").id.clone();
        graph.create_relation(&a, &b, RelationType::References, None, None);

        let stats = graph.stats();
        assert_eq!(stats.nodes, 2);
        assert_eq!(stats.edges, 1);
        assert_eq!(stats.types.get("concept"), Some(&1));
        assert_eq!(stats.types.get("file"), Some(&1));
        assert_eq!(stats.relations.get("references"), Some(&1));
    }

    #[test]
    fn dirty_tracking() {
        let mut graph = MemoryGraph::new();
        assert!(!graph.is_dirty());
        graph.upsert_node(NodeType::Concept, "A", "a", make_embedding(0.1), None);
        assert!(graph.is_dirty());
        graph.mark_clean();
        assert!(!graph.is_dirty());
    }

    #[tokio::test]
    async fn graph_store_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();

        let store = GraphStore::new();

        // Create nodes and edge
        let (_node_a_id, _node_b_id) = store
            .get_graph(&root, |graph| {
                let a = graph.upsert_node(
                    NodeType::Concept,
                    "auth",
                    "authentication",
                    vec![1.0, 0.0],
                    None,
                );
                let b =
                    graph.upsert_node(NodeType::File, "auth.rs", "auth file", vec![0.0, 1.0], None);
                graph.create_relation(&a.id, &b.id, RelationType::Implements, None, None);
                (a.id, b.id)
            })
            .await
            .expect("get_graph");

        // Persist
        store.persist(&root).await.expect("persist");

        // Load fresh store and verify
        let store2 = GraphStore::new();
        let stats = store2
            .get_graph(&root, |graph| graph.stats())
            .await
            .expect("reload");

        assert_eq!(stats.nodes, 2);
        assert_eq!(stats.edges, 1);
    }
    #[tokio::test]
    async fn graph_save_load_roundtrip_json_format() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_string_lossy().to_string();
        let store = Arc::new(GraphStore::new());
        store
            .get_graph(&root, |graph| {
                let a = graph.upsert_node(
                    NodeType::Concept,
                    "billing",
                    "billing module",
                    vec![0.5, 0.5],
                    Some(HashMap::from([(
                        "domain".to_string(),
                        "finance".to_string(),
                    )])),
                );
                let b = graph.upsert_node(
                    NodeType::Symbol,
                    "charge_card",
                    "charges a credit card",
                    vec![0.3, 0.7],
                    None,
                );
                graph.create_relation(&a.id, &b.id, RelationType::Contains, Some(0.9), None);
            })
            .await
            .expect("create");
        store.flush().await.expect("flush");
        let json_path = dir.path().join(".mcp_data").join("memory-graph.json");
        assert!(json_path.exists());
        let store2 = Arc::new(GraphStore::new());
        let stats = store2
            .get_graph(&root, |graph| {
                let s = graph.stats();
                let billing = graph.find_node("billing", &NodeType::Concept);
                assert!(billing.is_some());
                assert_eq!(
                    billing.unwrap().metadata.get("domain"),
                    Some(&"finance".to_string())
                );
                s
            })
            .await
            .expect("reload");
        assert_eq!(stats.nodes, 2);
        assert_eq!(stats.edges, 1);
    }

    #[tokio::test]
    async fn debounce_does_not_write_immediately() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_string_lossy().to_string();
        let store = Arc::new(GraphStore::new());
        let _handle = store.spawn_debounce_task();
        store
            .get_graph(&root, |graph| {
                graph.upsert_node(NodeType::Note, "test", "a note", vec![1.0], None);
            })
            .await
            .expect("upsert");
        let json_path = dir.path().join(".mcp_data").join("memory-graph.json");
        assert!(!json_path.exists(), "should NOT exist immediately");
        tokio::time::sleep(tokio::time::Duration::from_millis(700)).await;
        assert!(json_path.exists(), "should exist after debounce");
        store.flush().await.expect("flush");
    }

    #[tokio::test]
    async fn flush_persists_immediately() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_string_lossy().to_string();
        let store = Arc::new(GraphStore::new());
        store
            .get_graph(&root, |graph| {
                graph.upsert_node(NodeType::Concept, "quick", "fast save", vec![0.1], None);
            })
            .await
            .expect("upsert");
        store.flush().await.expect("flush");
        let json_path = dir.path().join(".mcp_data").join("memory-graph.json");
        assert!(json_path.exists());
        let content = tokio::fs::read_to_string(&json_path).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(parsed.get("nodes").is_some());
        assert!(parsed.get("edges").is_some());
    }

    #[tokio::test]
    async fn load_graph_missing_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_string_lossy().to_string();
        // No file written — should silently return empty graph (first-run case)
        let graph = load_graph_from_disk(&root)
            .await
            .expect("should succeed on missing file");
        assert_eq!(graph.graph.node_count(), 0);
        assert_eq!(graph.graph.edge_count(), 0);
    }

    #[tokio::test]
    async fn load_graph_unreadable_file_returns_error() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join(CACHE_DIR);
        fs::create_dir_all(&cache_dir).unwrap();
        let graph_path = cache_dir.join(GRAPH_FILE_JSON);
        // Write a valid file, then revoke read permissions
        fs::write(&graph_path, b"{\"nodes\":{},\"edges\":{}}").unwrap();
        fs::set_permissions(&graph_path, fs::Permissions::from_mode(0o000)).unwrap();

        let root = dir.path().to_string_lossy().to_string();
        let result = load_graph_from_disk(&root).await;

        // Restore permissions so tempdir cleanup doesn't fail
        fs::set_permissions(&graph_path, fs::Permissions::from_mode(0o644)).unwrap();

        // Must propagate as error — not silently return empty graph
        assert!(result.is_err(), "expected Err on unreadable file, got Ok");
        assert!(
            matches!(result.err().unwrap(), ContextPlusError::Io(_)),
            "expected Io variant"
        );
    }

    #[tokio::test]
    async fn load_graph_partial_malformed_nodes_loads_valid_ones() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join(CACHE_DIR);
        tokio::fs::create_dir_all(&cache_dir).await.unwrap();
        let graph_path = cache_dir.join(GRAPH_FILE_JSON);

        // JSON with one valid node and one garbage node (missing required fields)
        let json = serde_json::json!({
            "nodes": {
                "good-id": {
                    "id": "good-id",
                    "label": "auth",
                    "node_type": "concept",
                    "content": "auth module",
                    "embedding": [0.1, 0.2],
                    "created_at": 0u64,
                    "updated_at": 0u64,
                    "access_count": 0u64,
                    "last_accessed": 0u64,
                    "metadata": {}
                },
                "bad-id": {
                    "this_is": "garbage"
                }
            },
            "edges": {}
        });
        tokio::fs::write(&graph_path, json.to_string())
            .await
            .unwrap();

        let root = dir.path().to_string_lossy().to_string();
        let graph = load_graph_from_disk(&root)
            .await
            .expect("should succeed despite malformed node");
        // Only the valid node should be loaded
        assert_eq!(
            graph.graph.node_count(),
            1,
            "expected 1 valid node, got {}",
            graph.graph.node_count()
        );
    }
}
