// MCP tool wrappers for memory graph operations and interlinked RAG.
// 6 tools: upsert, relate, search, prune, interlink, traverse.

use crate::core::embeddings::OllamaClient;
use crate::core::memory_graph::{GraphStore, NodeType, RelationType, TraversalResult};
use crate::error::{ContextPlusError, Result};
use std::collections::HashMap;

// --- Options structs ---

pub struct UpsertMemoryNodeOptions {
    pub root_dir: String,
    pub node_type: String,
    pub label: String,
    pub content: String,
    pub metadata: Option<HashMap<String, String>>,
}

pub struct CreateRelationOptions {
    pub root_dir: String,
    /// Direct node ID (TS API: source_id). Bypasses label+type lookup.
    pub source_id: Option<String>,
    /// Fallback: find node by label+type if source_id is not provided.
    pub source_label: Option<String>,
    pub source_type: String,
    /// Direct node ID (TS API: target_id). Bypasses label+type lookup.
    pub target_id: Option<String>,
    /// Fallback: find node by label+type if target_id is not provided.
    pub target_label: Option<String>,
    pub target_type: String,
    pub relation: String,
    pub weight: Option<f32>,
    pub metadata: Option<HashMap<String, String>>,
}

pub struct SearchMemoryGraphOptions {
    pub root_dir: String,
    pub query: String,
    pub max_depth: Option<usize>,
    pub top_k: Option<usize>,
    pub edge_filter: Option<Vec<String>>,
}

pub struct PruneStaleLinksOptions {
    pub root_dir: String,
    pub threshold: Option<f64>,
}

pub struct AddInterlinkedContextOptions {
    pub root_dir: String,
    pub items: Vec<InterlinkedItem>,
    pub auto_link: Option<bool>,
}

pub struct InterlinkedItem {
    pub node_type: String,
    pub label: String,
    pub content: String,
    pub metadata: Option<HashMap<String, String>>,
}

pub struct RetrieveWithTraversalOptions {
    pub root_dir: String,
    pub node_id: String,
    pub max_depth: Option<usize>,
    pub edge_filter: Option<Vec<String>>,
}

// --- Helpers ---

fn format_traversal_result(result: &TraversalResult) -> String {
    let content_preview = if result.node.content.len() > 120 {
        format!(
            "{}...",
            crate::core::parser::truncate_to_char_boundary(&result.node.content, 120)
        )
    } else {
        result.node.content.clone()
    };

    let mut lines = vec![
        format!(
            "  [{}] {} (depth: {}, score: {})",
            result.node.node_type, result.node.label, result.depth, result.relevance_score
        ),
        format!("    Content: {}", content_preview),
    ];

    if result.path_relations.len() > 1 {
        lines.push(format!("    Path: {}", result.path_relations.join(" ")));
    }

    lines.push(format!(
        "    ID: {} | Accessed: {}x",
        result.node.id, result.node.access_count
    ));

    lines.join("\n")
}

fn parse_node_type(s: &str) -> Result<NodeType> {
    NodeType::parse_str(s).ok_or_else(|| {
        ContextPlusError::Other(format!(
            "Invalid node type: '{}'. Valid types: concept, file, symbol, note",
            s
        ))
    })
}

fn parse_relation_type(s: &str) -> Result<RelationType> {
    RelationType::parse_str(s).ok_or_else(|| {
        ContextPlusError::Other(format!(
            "Invalid relation: '{}'. Valid: relates_to, depends_on, implements, references, similar_to, contains",
            s
        ))
    })
}

fn parse_edge_filter(filter: &Option<Vec<String>>) -> Result<Option<Vec<RelationType>>> {
    match filter {
        None => Ok(None),
        Some(strs) => {
            let mut types = Vec::new();
            for s in strs {
                types.push(parse_relation_type(s)?);
            }
            Ok(Some(types))
        }
    }
}

/// Fetch a single embedding via OllamaClient.
async fn fetch_embedding(ollama: &OllamaClient, text: &str) -> Result<Vec<f32>> {
    let results = ollama.embed(&[text.to_string()]).await?;
    results
        .into_iter()
        .next()
        .ok_or_else(|| ContextPlusError::Ollama("No embedding returned".to_string()))
}

// --- Tool implementations ---

/// Tool 1: Upsert a memory node. Creates or updates by (label, type).
pub async fn tool_upsert_memory_node(
    store: &GraphStore,
    ollama: &OllamaClient,
    options: UpsertMemoryNodeOptions,
) -> Result<String> {
    let node_type = parse_node_type(&options.node_type)?;
    let embed_text = format!("{} {}", options.label, options.content);
    let embedding = fetch_embedding(ollama, &embed_text).await?;

    let node = store
        .get_graph(&options.root_dir, |graph| {
            graph.upsert_node(
                node_type,
                &options.label,
                &options.content,
                embedding,
                options.metadata,
            )
        })
        .await?;

    let stats = store
        .get_graph(&options.root_dir, |graph| graph.stats())
        .await?;

    store.persist(&options.root_dir).await?;

    Ok(format!(
        "Memory node upserted: {}\n  ID: {}\n  Type: {}\n  Access count: {}\n\nGraph: {} nodes, {} edges",
        node.label, node.id, node.node_type, node.access_count, stats.nodes, stats.edges
    ))
}

/// Tool 2: Create a relation between two nodes.
/// Supports both direct ID lookup (TS API: source_id/target_id) and label+type fallback.
pub async fn tool_create_relation(
    store: &GraphStore,
    options: CreateRelationOptions,
) -> Result<String> {
    let relation = parse_relation_type(&options.relation)?;

    // Resolve source: prefer direct ID, fall back to label+type lookup
    let source_id = if let Some(ref id) = options.source_id {
        let exists = store
            .get_graph(&options.root_dir, |graph| graph.node_exists(id))
            .await?;
        if !exists {
            return Ok(format!("Failed: source node not found (id: '{}')", id));
        }
        id.clone()
    } else if let Some(ref label) = options.source_label {
        let source_type = parse_node_type(&options.source_type)?;
        let found = store
            .get_graph(&options.root_dir, |graph| {
                graph.find_node(label, &source_type).map(|n| n.id.clone())
            })
            .await?;
        match found {
            Some(id) => id,
            None => {
                return Ok(format!(
                    "Failed: source node not found (label: '{}', type: '{}')",
                    label, options.source_type
                ));
            }
        }
    } else {
        return Ok("Failed: either source_id or source_label is required".to_string());
    };

    // Resolve target: prefer direct ID, fall back to label+type lookup
    let target_id = if let Some(ref id) = options.target_id {
        let exists = store
            .get_graph(&options.root_dir, |graph| graph.node_exists(id))
            .await?;
        if !exists {
            return Ok(format!("Failed: target node not found (id: '{}')", id));
        }
        id.clone()
    } else if let Some(ref label) = options.target_label {
        let target_type = parse_node_type(&options.target_type)?;
        let found = store
            .get_graph(&options.root_dir, |graph| {
                graph.find_node(label, &target_type).map(|n| n.id.clone())
            })
            .await?;
        match found {
            Some(id) => id,
            None => {
                return Ok(format!(
                    "Failed: target node not found (label: '{}', type: '{}')",
                    label, options.target_type
                ));
            }
        }
    } else {
        return Ok("Failed: either target_id or target_label is required".to_string());
    };

    let edge = store
        .get_graph(&options.root_dir, |graph| {
            graph.create_relation(
                &source_id,
                &target_id,
                relation,
                options.weight,
                options.metadata,
            )
        })
        .await?;

    let stats = store
        .get_graph(&options.root_dir, |graph| graph.stats())
        .await?;

    store.persist(&options.root_dir).await?;

    match edge {
        Some(e) => Ok(format!(
            "Relation created: {} --[{}]--> {}\n  Edge ID: {}\n  Weight: {}\n\nGraph: {} nodes, {} edges",
            source_id, e.relation, target_id, e.id, e.weight, stats.nodes, stats.edges
        )),
        None => Ok(format!(
            "Failed: one or both node IDs not found (source: {}, target: {})",
            source_id, target_id
        )),
    }
}

/// Tool 3: Search memory graph semantically + BFS traversal.
pub async fn tool_search_memory_graph(
    store: &GraphStore,
    ollama: &OllamaClient,
    options: SearchMemoryGraphOptions,
) -> Result<String> {
    let max_depth = options.max_depth.unwrap_or(1);
    let top_k = options.top_k.unwrap_or(5);
    let edge_filter = parse_edge_filter(&options.edge_filter)?;

    let query_embedding = fetch_embedding(ollama, &options.query).await?;

    let result = store
        .get_graph(&options.root_dir, |graph| {
            graph.search(&query_embedding, max_depth, top_k, edge_filter.as_deref())
        })
        .await?;

    store.persist(&options.root_dir).await?;

    if result.direct.is_empty() {
        return Ok(format!(
            "No memory nodes found for: \"{}\"\nGraph has {} nodes, {} edges.",
            options.query, result.total_nodes, result.total_edges
        ));
    }

    let mut sections = vec![
        format!("Memory Graph Search: \"{}\"", options.query),
        format!(
            "Graph: {} nodes, {} edges\n",
            result.total_nodes, result.total_edges
        ),
        "Direct Matches:".to_string(),
    ];

    for hit in &result.direct {
        sections.push(format_traversal_result(hit));
    }

    if !result.neighbors.is_empty() {
        sections.push("\nLinked Neighbors:".to_string());
        for neighbor in &result.neighbors {
            sections.push(format_traversal_result(neighbor));
        }
    }

    Ok(sections.join("\n"))
}

/// Tool 4: Prune stale links and orphan nodes.
pub async fn tool_prune_stale_links(
    store: &GraphStore,
    options: PruneStaleLinksOptions,
) -> Result<String> {
    let result = store
        .get_graph(&options.root_dir, |graph| {
            graph.prune_stale_links(options.threshold)
        })
        .await?;

    store.persist(&options.root_dir).await?;

    Ok(format!(
        "Pruning complete\n  Removed: {} stale links/orphan nodes\n  Remaining edges: {}",
        result.removed, result.remaining_edges
    ))
}

/// Tool 5: Add interlinked context (batch upsert + auto-similarity linking).
pub async fn tool_add_interlinked_context(
    store: &GraphStore,
    ollama: &OllamaClient,
    options: AddInterlinkedContextOptions,
) -> Result<String> {
    let auto_link = options.auto_link.unwrap_or(true);

    // Parse node types upfront before batch embedding
    let mut node_types = Vec::with_capacity(options.items.len());
    for item in &options.items {
        node_types.push(parse_node_type(&item.node_type)?);
    }

    // Batch embed all items in a single HTTP call instead of N separate calls
    let embed_texts: Vec<String> = options
        .items
        .iter()
        .map(|item| format!("{} {}", item.label, item.content))
        .collect();
    let embeddings = ollama.embed(&embed_texts).await?;

    // Zip node types, items, and embeddings into prepared tuples
    let mut prepared_items = Vec::with_capacity(options.items.len());
    for (i, item) in options.items.iter().enumerate() {
        let embedding = embeddings.get(i).cloned().ok_or_else(|| {
            crate::error::ContextPlusError::Ollama("Missing embedding for item".to_string())
        })?;
        prepared_items.push((
            node_types[i].clone(),
            item.label.clone(),
            item.content.clone(),
            embedding,
            item.metadata.clone(),
        ));
    }

    let result = store
        .get_graph(&options.root_dir, |graph| {
            graph.add_interlinked_context(prepared_items, auto_link)
        })
        .await?;

    let stats = store
        .get_graph(&options.root_dir, |graph| graph.stats())
        .await?;

    store.persist(&options.root_dir).await?;

    let mut sections = vec![format!("Added {} interlinked nodes", result.nodes.len())];

    if result.edges.is_empty() {
        sections.push("  No auto-links above threshold".to_string());
    } else {
        sections.push(format!(
            "  Auto-linked: {} similarity edges (threshold >= 0.72)",
            result.edges.len()
        ));
    }

    sections.push("\nNodes:".to_string());
    for node in &result.nodes {
        sections.push(format!(
            "  [{}] {} -> {}",
            node.node_type, node.label, node.id
        ));
    }

    if !result.edges.is_empty() {
        sections.push("\nEdges:".to_string());
        for edge in &result.edges {
            sections.push(format!("  --[{} w:{:.2}]-->", edge.relation, edge.weight));
        }
    }

    sections.push(format!(
        "\nGraph total: {} nodes, {} edges",
        stats.nodes, stats.edges
    ));

    Ok(sections.join("\n"))
}

/// Tool 6: Retrieve with BFS traversal from a specific node.
pub async fn tool_retrieve_with_traversal(
    store: &GraphStore,
    options: RetrieveWithTraversalOptions,
) -> Result<String> {
    let max_depth = options.max_depth.unwrap_or(2);
    let edge_filter = parse_edge_filter(&options.edge_filter)?;

    let results = store
        .get_graph(&options.root_dir, |graph| {
            graph.retrieve_with_traversal(&options.node_id, max_depth, edge_filter.as_deref())
        })
        .await?;

    store.persist(&options.root_dir).await?;

    if results.is_empty() {
        return Ok(format!("Node not found: {}", options.node_id));
    }

    let mut sections = vec![format!(
        "Traversal from: {} (depth limit: {})\n",
        results[0].node.label, max_depth
    )];

    for result in &results {
        sections.push(format_traversal_result(result));
    }

    Ok(sections.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_graph::MemoryNode;

    fn make_embedding(seed: f32) -> Vec<f32> {
        vec![seed, seed * 0.5, 1.0 - seed, seed * 0.3]
    }

    #[allow(clippy::too_many_arguments)]
    fn make_traversal(
        id: &str,
        node_type: NodeType,
        label: &str,
        content: &str,
        depth: usize,
        path: Vec<String>,
        score: f64,
        access_count: u32,
    ) -> TraversalResult {
        TraversalResult {
            node: MemoryNode {
                id: id.to_string(),
                node_type,
                label: label.to_string(),
                content: content.to_string(),
                embedding: vec![],
                created_at: 0,
                last_accessed: 0,
                access_count,
                metadata: HashMap::new(),
            },
            depth,
            path_relations: path,
            relevance_score: score,
        }
    }

    // ---------------------------------------------------------------
    // parse_node_type
    // ---------------------------------------------------------------

    #[test]
    fn parse_node_type_valid() {
        assert!(parse_node_type("concept").is_ok());
        assert!(parse_node_type("file").is_ok());
        assert!(parse_node_type("symbol").is_ok());
        assert!(parse_node_type("note").is_ok());
    }

    #[test]
    fn parse_node_type_returns_correct_variant() {
        assert_eq!(parse_node_type("concept").unwrap(), NodeType::Concept);
        assert_eq!(parse_node_type("file").unwrap(), NodeType::File);
        assert_eq!(parse_node_type("symbol").unwrap(), NodeType::Symbol);
        assert_eq!(parse_node_type("note").unwrap(), NodeType::Note);
    }

    #[test]
    fn parse_node_type_invalid() {
        assert!(parse_node_type("invalid").is_err());
    }

    #[test]
    fn parse_node_type_invalid_error_message() {
        let err = parse_node_type("bogus").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Invalid node type: 'bogus'"), "got: {msg}");
        assert!(msg.contains("concept, file, symbol, note"), "got: {msg}");
    }

    #[test]
    fn parse_node_type_case_sensitive() {
        assert!(parse_node_type("Concept").is_err());
        assert!(parse_node_type("FILE").is_err());
        assert!(parse_node_type("Symbol").is_err());
        assert!(parse_node_type("NOTE").is_err());
    }

    #[test]
    fn parse_node_type_empty_string() {
        assert!(parse_node_type("").is_err());
    }

    #[test]
    fn parse_node_type_whitespace() {
        assert!(parse_node_type(" concept").is_err());
        assert!(parse_node_type("concept ").is_err());
    }

    // ---------------------------------------------------------------
    // parse_relation_type
    // ---------------------------------------------------------------

    #[test]
    fn parse_relation_type_valid() {
        assert!(parse_relation_type("relates_to").is_ok());
        assert!(parse_relation_type("depends_on").is_ok());
        assert!(parse_relation_type("implements").is_ok());
        assert!(parse_relation_type("references").is_ok());
        assert!(parse_relation_type("similar_to").is_ok());
        assert!(parse_relation_type("contains").is_ok());
    }

    #[test]
    fn parse_relation_type_returns_correct_variant() {
        assert_eq!(
            parse_relation_type("relates_to").unwrap(),
            RelationType::RelatesTo
        );
        assert_eq!(
            parse_relation_type("depends_on").unwrap(),
            RelationType::DependsOn
        );
        assert_eq!(
            parse_relation_type("implements").unwrap(),
            RelationType::Implements
        );
        assert_eq!(
            parse_relation_type("references").unwrap(),
            RelationType::References
        );
        assert_eq!(
            parse_relation_type("similar_to").unwrap(),
            RelationType::SimilarTo
        );
        assert_eq!(
            parse_relation_type("contains").unwrap(),
            RelationType::Contains
        );
    }

    #[test]
    fn parse_relation_type_invalid() {
        assert!(parse_relation_type("invalid").is_err());
    }

    #[test]
    fn parse_relation_type_invalid_error_message() {
        let err = parse_relation_type("foo_bar").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Invalid relation: 'foo_bar'"), "got: {msg}");
        assert!(msg.contains("relates_to"), "got: {msg}");
        assert!(msg.contains("depends_on"), "got: {msg}");
        assert!(msg.contains("implements"), "got: {msg}");
        assert!(msg.contains("references"), "got: {msg}");
        assert!(msg.contains("similar_to"), "got: {msg}");
        assert!(msg.contains("contains"), "got: {msg}");
    }

    #[test]
    fn parse_relation_type_case_sensitive() {
        assert!(parse_relation_type("RelatesTo").is_err());
        assert!(parse_relation_type("DEPENDS_ON").is_err());
    }

    #[test]
    fn parse_relation_type_empty_string() {
        assert!(parse_relation_type("").is_err());
    }

    // ---------------------------------------------------------------
    // parse_edge_filter
    // ---------------------------------------------------------------

    #[test]
    fn parse_edge_filter_none() {
        assert!(parse_edge_filter(&None).unwrap().is_none());
    }

    #[test]
    fn parse_edge_filter_valid() {
        let filter = Some(vec!["depends_on".to_string(), "implements".to_string()]);
        let result = parse_edge_filter(&filter).unwrap();
        assert!(result.is_some());
        let types = result.unwrap();
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn parse_edge_filter_valid_preserves_order() {
        let filter = Some(vec![
            "contains".to_string(),
            "relates_to".to_string(),
            "similar_to".to_string(),
        ]);
        let types = parse_edge_filter(&filter).unwrap().unwrap();
        assert_eq!(types[0], RelationType::Contains);
        assert_eq!(types[1], RelationType::RelatesTo);
        assert_eq!(types[2], RelationType::SimilarTo);
    }

    #[test]
    fn parse_edge_filter_single_element() {
        let filter = Some(vec!["references".to_string()]);
        let types = parse_edge_filter(&filter).unwrap().unwrap();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0], RelationType::References);
    }

    #[test]
    fn parse_edge_filter_all_six_types() {
        let filter = Some(vec![
            "relates_to".to_string(),
            "depends_on".to_string(),
            "implements".to_string(),
            "references".to_string(),
            "similar_to".to_string(),
            "contains".to_string(),
        ]);
        let types = parse_edge_filter(&filter).unwrap().unwrap();
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn parse_edge_filter_invalid() {
        let filter = Some(vec!["invalid".to_string()]);
        assert!(parse_edge_filter(&filter).is_err());
    }

    #[test]
    fn parse_edge_filter_mixed_valid_invalid_fails() {
        let filter = Some(vec!["depends_on".to_string(), "nope".to_string()]);
        assert!(parse_edge_filter(&filter).is_err());
    }

    #[test]
    fn parse_edge_filter_empty_vec() {
        let filter = Some(vec![]);
        let result = parse_edge_filter(&filter).unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }

    // ---------------------------------------------------------------
    // format_traversal_result
    // ---------------------------------------------------------------

    #[test]
    fn format_traversal_result_basic() {
        let result = TraversalResult {
            node: MemoryNode {
                id: "mn-123".to_string(),
                node_type: NodeType::Concept,
                label: "auth".to_string(),
                content: "authentication module".to_string(),
                embedding: vec![],
                created_at: 0,
                last_accessed: 0,
                access_count: 3,
                metadata: HashMap::new(),
            },
            depth: 0,
            path_relations: vec![],
            relevance_score: 95.0,
        };
        let formatted = format_traversal_result(&result);
        assert!(formatted.contains("[concept] auth"));
        assert!(formatted.contains("depth: 0"));
        assert!(formatted.contains("score: 95"));
        assert!(formatted.contains("Accessed: 3x"));
    }

    #[test]
    fn format_traversal_result_with_path() {
        let result = TraversalResult {
            node: MemoryNode {
                id: "mn-456".to_string(),
                node_type: NodeType::File,
                label: "auth.rs".to_string(),
                content: "auth file".to_string(),
                embedding: vec![],
                created_at: 0,
                last_accessed: 0,
                access_count: 1,
                metadata: HashMap::new(),
            },
            depth: 1,
            path_relations: vec![
                "root".to_string(),
                "--[contains]-->".to_string(),
                "auth.rs".to_string(),
            ],
            relevance_score: 72.5,
        };
        let formatted = format_traversal_result(&result);
        assert!(formatted.contains("Path: root --[contains]--> auth.rs"));
    }

    #[test]
    fn format_traversal_result_truncates_long_content() {
        let long_content = "a".repeat(200);
        let result = TraversalResult {
            node: MemoryNode {
                id: "mn-789".to_string(),
                node_type: NodeType::Note,
                label: "note".to_string(),
                content: long_content,
                embedding: vec![],
                created_at: 0,
                last_accessed: 0,
                access_count: 1,
                metadata: HashMap::new(),
            },
            depth: 0,
            path_relations: vec![],
            relevance_score: 50.0,
        };
        let formatted = format_traversal_result(&result);
        assert!(formatted.contains("..."));
    }

    #[test]
    fn format_traversal_result_no_path_when_single_element() {
        let result = make_traversal(
            "mn-1",
            NodeType::Concept,
            "root",
            "content",
            0,
            vec!["root".to_string()],
            100.0,
            1,
        );
        let formatted = format_traversal_result(&result);
        // path_relations.len() == 1, so "Path:" should NOT appear
        assert!(!formatted.contains("Path:"), "got: {formatted}");
    }

    #[test]
    fn format_traversal_result_no_path_when_empty() {
        let result = make_traversal(
            "mn-1",
            NodeType::Concept,
            "root",
            "content",
            0,
            vec![],
            100.0,
            1,
        );
        let formatted = format_traversal_result(&result);
        assert!(!formatted.contains("Path:"), "got: {formatted}");
    }

    #[test]
    fn format_traversal_result_exact_120_chars_no_truncation() {
        let content = "b".repeat(120);
        let result = make_traversal(
            "mn-boundary",
            NodeType::Note,
            "boundary",
            &content,
            0,
            vec![],
            50.0,
            1,
        );
        let formatted = format_traversal_result(&result);
        // Exactly 120 chars is NOT > 120, so no truncation
        assert!(!formatted.contains("..."), "got: {formatted}");
        assert!(formatted.contains(&content), "got: {formatted}");
    }

    #[test]
    fn format_traversal_result_121_chars_truncates() {
        let content = "c".repeat(121);
        let result = make_traversal(
            "mn-over",
            NodeType::Note,
            "over",
            &content,
            0,
            vec![],
            50.0,
            1,
        );
        let formatted = format_traversal_result(&result);
        assert!(formatted.contains("..."), "got: {formatted}");
        // Should not contain the full 121-char string
        assert!(!formatted.contains(&content), "got: {formatted}");
    }

    #[test]
    fn format_traversal_result_contains_id() {
        let result = make_traversal(
            "mn-unique-id-42",
            NodeType::Symbol,
            "my_fn",
            "a function",
            2,
            vec![],
            33.3,
            7,
        );
        let formatted = format_traversal_result(&result);
        assert!(
            formatted.contains("ID: mn-unique-id-42"),
            "got: {formatted}"
        );
    }

    #[test]
    fn format_traversal_result_all_node_types() {
        for (nt, expected) in [
            (NodeType::Concept, "[concept]"),
            (NodeType::File, "[file]"),
            (NodeType::Symbol, "[symbol]"),
            (NodeType::Note, "[note]"),
        ] {
            let result = make_traversal("mn-1", nt, "x", "x", 0, vec![], 0.0, 1);
            let formatted = format_traversal_result(&result);
            assert!(formatted.contains(expected), "got: {formatted}");
        }
    }

    #[test]
    fn format_traversal_result_short_content_not_truncated() {
        let result = make_traversal(
            "mn-short",
            NodeType::Note,
            "short",
            "hello world",
            0,
            vec![],
            80.0,
            2,
        );
        let formatted = format_traversal_result(&result);
        assert!(
            formatted.contains("Content: hello world"),
            "got: {formatted}"
        );
        assert!(!formatted.contains("..."), "got: {formatted}");
    }

    #[test]
    fn format_traversal_result_depth_and_score() {
        let result = make_traversal("mn-1", NodeType::Concept, "deep", "c", 5, vec![], 12.34, 1);
        let formatted = format_traversal_result(&result);
        assert!(formatted.contains("depth: 5"), "got: {formatted}");
        assert!(formatted.contains("score: 12.34"), "got: {formatted}");
    }

    // ---------------------------------------------------------------
    // tool_create_relation
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn tool_create_relation_source_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        // Create only the target node
        store
            .get_graph(&root, |graph| {
                graph.upsert_node(
                    NodeType::Concept,
                    "target",
                    "target node",
                    make_embedding(0.5),
                    None,
                );
            })
            .await
            .expect("ok");

        let result = tool_create_relation(
            &store,
            CreateRelationOptions {
                root_dir: root,
                source_id: None,
                source_label: Some("nonexistent".to_string()),
                source_type: "concept".to_string(),
                target_id: None,
                target_label: Some("target".to_string()),
                target_type: "concept".to_string(),
                relation: "depends_on".to_string(),
                weight: None,
                metadata: None,
            },
        )
        .await
        .expect("ok");

        assert!(result.contains("source node not found"), "got: {result}");
        assert!(result.contains("nonexistent"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_create_relation_target_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        // Create only the source node
        store
            .get_graph(&root, |graph| {
                graph.upsert_node(
                    NodeType::Concept,
                    "source",
                    "source node",
                    make_embedding(0.5),
                    None,
                );
            })
            .await
            .expect("ok");

        let result = tool_create_relation(
            &store,
            CreateRelationOptions {
                root_dir: root,
                source_id: None,
                source_label: Some("source".to_string()),
                source_type: "concept".to_string(),
                target_id: None,
                target_label: Some("missing".to_string()),
                target_type: "concept".to_string(),
                relation: "depends_on".to_string(),
                weight: None,
                metadata: None,
            },
        )
        .await
        .expect("ok");

        assert!(result.contains("target node not found"), "got: {result}");
        assert!(result.contains("missing"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_create_relation_success() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        store
            .get_graph(&root, |graph| {
                graph.upsert_node(
                    NodeType::Concept,
                    "auth",
                    "auth module",
                    make_embedding(0.3),
                    None,
                );
                graph.upsert_node(
                    NodeType::File,
                    "auth.rs",
                    "auth file",
                    make_embedding(0.7),
                    None,
                );
            })
            .await
            .expect("ok");

        let result = tool_create_relation(
            &store,
            CreateRelationOptions {
                root_dir: root,
                source_id: None,
                source_label: Some("auth".to_string()),
                source_type: "concept".to_string(),
                target_id: None,
                target_label: Some("auth.rs".to_string()),
                target_type: "file".to_string(),
                relation: "implements".to_string(),
                weight: Some(0.9),
                metadata: None,
            },
        )
        .await
        .expect("ok");

        assert!(result.contains("Relation created"), "got: {result}");
        assert!(result.contains("implements"), "got: {result}");
        assert!(
            result.contains("1 nodes, 1 edges") || result.contains("2 nodes, 1 edges"),
            "got: {result}"
        );
    }

    #[tokio::test]
    async fn tool_create_relation_invalid_relation_type() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_create_relation(
            &store,
            CreateRelationOptions {
                root_dir: root,
                source_id: None,
                source_label: Some("a".to_string()),
                source_type: "concept".to_string(),
                target_id: None,
                target_label: Some("b".to_string()),
                target_type: "concept".to_string(),
                relation: "invalid_relation".to_string(),
                weight: None,
                metadata: None,
            },
        )
        .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid relation"),);
    }

    #[tokio::test]
    async fn tool_create_relation_invalid_source_type() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_create_relation(
            &store,
            CreateRelationOptions {
                root_dir: root,
                source_id: None,
                source_label: Some("a".to_string()),
                source_type: "bad_type".to_string(),
                target_id: None,
                target_label: Some("b".to_string()),
                target_type: "concept".to_string(),
                relation: "depends_on".to_string(),
                weight: None,
                metadata: None,
            },
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn tool_create_relation_invalid_target_type() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        // Create a source node so source lookup succeeds
        store
            .get_graph(&root, |graph| {
                graph.upsert_node(NodeType::Concept, "a", "test", vec![0.1, 0.2], None)
            })
            .await
            .unwrap();

        let result = tool_create_relation(
            &store,
            CreateRelationOptions {
                root_dir: root,
                source_id: None,
                source_label: Some("a".to_string()),
                source_type: "concept".to_string(),
                target_id: None,
                target_label: Some("b".to_string()),
                target_type: "bad_type".to_string(),
                relation: "depends_on".to_string(),
                weight: None,
                metadata: None,
            },
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn tool_create_relation_with_metadata() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        store
            .get_graph(&root, |graph| {
                graph.upsert_node(NodeType::Concept, "A", "node A", make_embedding(0.2), None);
                graph.upsert_node(NodeType::Concept, "B", "node B", make_embedding(0.8), None);
            })
            .await
            .expect("ok");

        let mut meta = HashMap::new();
        meta.insert("reason".to_string(), "test".to_string());

        let result = tool_create_relation(
            &store,
            CreateRelationOptions {
                root_dir: root,
                source_id: None,
                source_label: Some("A".to_string()),
                source_type: "concept".to_string(),
                target_id: None,
                target_label: Some("B".to_string()),
                target_type: "concept".to_string(),
                relation: "relates_to".to_string(),
                weight: Some(0.75),
                metadata: Some(meta),
            },
        )
        .await
        .expect("ok");

        assert!(result.contains("Relation created"), "got: {result}");
        assert!(result.contains("Weight: 0.75"), "got: {result}");
    }

    // ---------------------------------------------------------------
    // tool_prune_stale_links
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn tool_prune_stale_links_empty_graph() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_prune_stale_links(
            &store,
            PruneStaleLinksOptions {
                root_dir: root,
                threshold: None,
            },
        )
        .await
        .expect("ok");

        assert!(result.contains("Pruning complete"), "got: {result}");
        assert!(result.contains("Removed: 0"), "got: {result}");
        assert!(result.contains("Remaining edges: 0"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_prune_stale_links_with_threshold() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        // Add nodes and a fresh edge
        store
            .get_graph(&root, |graph| {
                let a = graph.upsert_node(NodeType::Concept, "A", "a", make_embedding(0.1), None);
                let b = graph.upsert_node(NodeType::Concept, "B", "b", make_embedding(0.9), None);
                graph.create_relation(&a.id, &b.id, RelationType::DependsOn, None, None);
            })
            .await
            .expect("ok");

        let result = tool_prune_stale_links(
            &store,
            PruneStaleLinksOptions {
                root_dir: root,
                threshold: Some(0.1),
            },
        )
        .await
        .expect("ok");

        assert!(result.contains("Pruning complete"), "got: {result}");
        // Fresh edge should not be pruned (decay ~ 1.0, well above 0.1)
        assert!(result.contains("Remaining edges: 1"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_prune_stale_links_format() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_prune_stale_links(
            &store,
            PruneStaleLinksOptions {
                root_dir: root,
                threshold: Some(0.5),
            },
        )
        .await
        .expect("ok");

        // Verify output format structure
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines[0], "Pruning complete");
        assert!(lines[1].contains("Removed:"), "got: {}", lines[1]);
        assert!(
            lines[1].contains("stale links/orphan nodes"),
            "got: {}",
            lines[1]
        );
        assert!(lines[2].contains("Remaining edges:"), "got: {}", lines[2]);
    }

    // ---------------------------------------------------------------
    // tool_retrieve_with_traversal
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn tool_retrieve_with_traversal_node_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_retrieve_with_traversal(
            &store,
            RetrieveWithTraversalOptions {
                root_dir: root,
                node_id: "nonexistent-id".to_string(),
                max_depth: None,
                edge_filter: None,
            },
        )
        .await
        .expect("ok");

        assert_eq!(result, "Node not found: nonexistent-id");
    }

    #[tokio::test]
    async fn tool_retrieve_with_traversal_success() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let node_id = store
            .get_graph(&root, |graph| {
                let a = graph.upsert_node(
                    NodeType::Concept,
                    "root_node",
                    "the root",
                    make_embedding(0.5),
                    None,
                );
                let b = graph.upsert_node(
                    NodeType::File,
                    "child.rs",
                    "child file",
                    make_embedding(0.6),
                    None,
                );
                graph.create_relation(&a.id, &b.id, RelationType::Contains, None, None);
                a.id
            })
            .await
            .expect("ok");

        let result = tool_retrieve_with_traversal(
            &store,
            RetrieveWithTraversalOptions {
                root_dir: root,
                node_id,
                max_depth: Some(2),
                edge_filter: None,
            },
        )
        .await
        .expect("ok");

        assert!(
            result.contains("Traversal from: root_node"),
            "got: {result}"
        );
        assert!(result.contains("depth limit: 2"), "got: {result}");
        assert!(result.contains("[concept] root_node"), "got: {result}");
        assert!(result.contains("[file] child.rs"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_retrieve_with_traversal_default_max_depth() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let node_id = store
            .get_graph(&root, |graph| {
                let a = graph.upsert_node(
                    NodeType::Concept,
                    "solo",
                    "standalone node",
                    make_embedding(0.5),
                    None,
                );
                a.id
            })
            .await
            .expect("ok");

        let result = tool_retrieve_with_traversal(
            &store,
            RetrieveWithTraversalOptions {
                root_dir: root,
                node_id,
                max_depth: None, // defaults to 2
                edge_filter: None,
            },
        )
        .await
        .expect("ok");

        // Default max_depth is 2
        assert!(result.contains("depth limit: 2"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_retrieve_with_traversal_with_edge_filter() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let node_id = store
            .get_graph(&root, |graph| {
                let a = graph.upsert_node(NodeType::Concept, "A", "a", make_embedding(0.1), None);
                let b = graph.upsert_node(NodeType::Concept, "B", "b", make_embedding(0.5), None);
                let c = graph.upsert_node(NodeType::File, "C", "c", make_embedding(0.9), None);
                graph.create_relation(&a.id, &b.id, RelationType::DependsOn, None, None);
                graph.create_relation(&a.id, &c.id, RelationType::Contains, None, None);
                a.id
            })
            .await
            .expect("ok");

        let result = tool_retrieve_with_traversal(
            &store,
            RetrieveWithTraversalOptions {
                root_dir: root,
                node_id,
                max_depth: Some(1),
                edge_filter: Some(vec!["depends_on".to_string()]),
            },
        )
        .await
        .expect("ok");

        // Should contain A and B (depends_on) but not C (contains)
        assert!(result.contains("[concept] A"), "got: {result}");
        assert!(result.contains("[concept] B"), "got: {result}");
        assert!(!result.contains("[file] C"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_retrieve_with_traversal_invalid_edge_filter() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_retrieve_with_traversal(
            &store,
            RetrieveWithTraversalOptions {
                root_dir: root,
                node_id: "any-id".to_string(),
                max_depth: None,
                edge_filter: Some(vec!["not_a_valid_type".to_string()]),
            },
        )
        .await;

        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // tool_upsert_memory_node (requires mock Ollama)
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn tool_upsert_memory_node_with_mock_ollama() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        // Mock the /api/embed endpoint
        let embed_response = serde_json::json!({
            "embeddings": [[0.1, 0.2, 0.3, 0.4]]
        });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&embed_response))
            .mount(&mock_server)
            .await;

        let config = crate::config::Config {
            ollama_host: mock_server.uri(),
            ollama_embed_model: "test-model".to_string(),
            ollama_chat_model: "test-chat".to_string(),
            ollama_api_key: None,
            embed_batch_size: 32,
            embed_tracker_mode: crate::config::TrackerMode::Off,
            embed_tracker_debounce_ms: 0,
            embed_tracker_max_files: 0,
            ignore_dirs: std::collections::HashSet::new(),
            cache_ttl_secs: 300,
            max_embed_file_size: 50 * 1024,
            embed_num_gpu: None,
            embed_main_gpu: None,
            embed_num_thread: None,
            embed_num_batch: None,
            embed_num_ctx: None,
            embed_low_vram: None,
            idle_timeout_ms: 0,
            parent_poll_ms: 5000,
            embed_chunk_chars: 2000,
        };
        let ollama = OllamaClient::new(&config);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_upsert_memory_node(
            &store,
            &ollama,
            UpsertMemoryNodeOptions {
                root_dir: root,
                node_type: "concept".to_string(),
                label: "authentication".to_string(),
                content: "handles user login and token verification".to_string(),
                metadata: None,
            },
        )
        .await
        .expect("ok");

        assert!(
            result.contains("Memory node upserted: authentication"),
            "got: {result}"
        );
        assert!(result.contains("Type: concept"), "got: {result}");
        assert!(result.contains("1 nodes, 0 edges"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_upsert_memory_node_invalid_type() {
        use wiremock::MockServer;

        let mock_server = MockServer::start().await;
        let config = crate::config::Config {
            ollama_host: mock_server.uri(),
            ollama_embed_model: "test-model".to_string(),
            ollama_chat_model: "test-chat".to_string(),
            ollama_api_key: None,
            embed_batch_size: 32,
            embed_tracker_mode: crate::config::TrackerMode::Off,
            embed_tracker_debounce_ms: 0,
            embed_tracker_max_files: 0,
            ignore_dirs: std::collections::HashSet::new(),
            cache_ttl_secs: 300,
            max_embed_file_size: 50 * 1024,
            embed_num_gpu: None,
            embed_main_gpu: None,
            embed_num_thread: None,
            embed_num_batch: None,
            embed_num_ctx: None,
            embed_low_vram: None,
            idle_timeout_ms: 0,
            parent_poll_ms: 5000,
            embed_chunk_chars: 2000,
        };
        let ollama = OllamaClient::new(&config);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_upsert_memory_node(
            &store,
            &ollama,
            UpsertMemoryNodeOptions {
                root_dir: root,
                node_type: "invalid_type".to_string(),
                label: "test".to_string(),
                content: "test".to_string(),
                metadata: None,
            },
        )
        .await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid node type"),
        );
    }

    #[tokio::test]
    async fn tool_upsert_memory_node_with_metadata() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let embed_response = serde_json::json!({
            "embeddings": [[0.5, 0.5, 0.5, 0.5]]
        });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&embed_response))
            .mount(&mock_server)
            .await;

        let config = crate::config::Config {
            ollama_host: mock_server.uri(),
            ollama_embed_model: "test-model".to_string(),
            ollama_chat_model: "test-chat".to_string(),
            ollama_api_key: None,
            embed_batch_size: 32,
            embed_tracker_mode: crate::config::TrackerMode::Off,
            embed_tracker_debounce_ms: 0,
            embed_tracker_max_files: 0,
            ignore_dirs: std::collections::HashSet::new(),
            cache_ttl_secs: 300,
            max_embed_file_size: 50 * 1024,
            embed_num_gpu: None,
            embed_main_gpu: None,
            embed_num_thread: None,
            embed_num_batch: None,
            embed_num_ctx: None,
            embed_low_vram: None,
            idle_timeout_ms: 0,
            parent_poll_ms: 5000,
            embed_chunk_chars: 2000,
        };
        let ollama = OllamaClient::new(&config);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "test".to_string());

        let result = tool_upsert_memory_node(
            &store,
            &ollama,
            UpsertMemoryNodeOptions {
                root_dir: root,
                node_type: "note".to_string(),
                label: "my-note".to_string(),
                content: "some important note".to_string(),
                metadata: Some(meta),
            },
        )
        .await
        .expect("ok");

        assert!(
            result.contains("Memory node upserted: my-note"),
            "got: {result}"
        );
        assert!(result.contains("Access count: 1"), "got: {result}");
    }

    // ---------------------------------------------------------------
    // tool_search_memory_graph (requires mock Ollama)
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn tool_search_memory_graph_empty_graph() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let embed_response = serde_json::json!({
            "embeddings": [[0.5, 0.5, 0.5]]
        });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&embed_response))
            .mount(&mock_server)
            .await;

        let config = crate::config::Config {
            ollama_host: mock_server.uri(),
            ollama_embed_model: "test-model".to_string(),
            ollama_chat_model: "test-chat".to_string(),
            ollama_api_key: None,
            embed_batch_size: 32,
            embed_tracker_mode: crate::config::TrackerMode::Off,
            embed_tracker_debounce_ms: 0,
            embed_tracker_max_files: 0,
            ignore_dirs: std::collections::HashSet::new(),
            cache_ttl_secs: 300,
            max_embed_file_size: 50 * 1024,
            embed_num_gpu: None,
            embed_main_gpu: None,
            embed_num_thread: None,
            embed_num_batch: None,
            embed_num_ctx: None,
            embed_low_vram: None,
            idle_timeout_ms: 0,
            parent_poll_ms: 5000,
            embed_chunk_chars: 2000,
        };
        let ollama = OllamaClient::new(&config);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_search_memory_graph(
            &store,
            &ollama,
            SearchMemoryGraphOptions {
                root_dir: root,
                query: "test query".to_string(),
                max_depth: None,
                top_k: None,
                edge_filter: None,
            },
        )
        .await
        .expect("ok");

        assert!(result.contains("No memory nodes found"), "got: {result}");
        assert!(result.contains("test query"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_search_memory_graph_with_results() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let embed_response = serde_json::json!({
            "embeddings": [[1.0, 0.0, 0.0]]
        });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&embed_response))
            .mount(&mock_server)
            .await;

        let config = crate::config::Config {
            ollama_host: mock_server.uri(),
            ollama_embed_model: "test-model".to_string(),
            ollama_chat_model: "test-chat".to_string(),
            ollama_api_key: None,
            embed_batch_size: 32,
            embed_tracker_mode: crate::config::TrackerMode::Off,
            embed_tracker_debounce_ms: 0,
            embed_tracker_max_files: 0,
            ignore_dirs: std::collections::HashSet::new(),
            cache_ttl_secs: 300,
            max_embed_file_size: 50 * 1024,
            embed_num_gpu: None,
            embed_main_gpu: None,
            embed_num_thread: None,
            embed_num_batch: None,
            embed_num_ctx: None,
            embed_low_vram: None,
            idle_timeout_ms: 0,
            parent_poll_ms: 5000,
            embed_chunk_chars: 2000,
        };
        let ollama = OllamaClient::new(&config);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        // Pre-populate graph
        store
            .get_graph(&root, |graph| {
                graph.upsert_node(
                    NodeType::Concept,
                    "auth",
                    "authentication logic",
                    vec![1.0, 0.0, 0.0],
                    None,
                );
                graph.upsert_node(
                    NodeType::Concept,
                    "db",
                    "database layer",
                    vec![0.0, 1.0, 0.0],
                    None,
                );
            })
            .await
            .expect("ok");

        let result = tool_search_memory_graph(
            &store,
            &ollama,
            SearchMemoryGraphOptions {
                root_dir: root,
                query: "authentication".to_string(),
                max_depth: Some(0),
                top_k: Some(5),
                edge_filter: None,
            },
        )
        .await
        .expect("ok");

        assert!(
            result.contains("Memory Graph Search: \"authentication\""),
            "got: {result}"
        );
        assert!(result.contains("Direct Matches:"), "got: {result}");
        assert!(result.contains("[concept] auth"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_search_memory_graph_with_neighbors() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let embed_response = serde_json::json!({
            "embeddings": [[1.0, 0.0, 0.0]]
        });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&embed_response))
            .mount(&mock_server)
            .await;

        let config = crate::config::Config {
            ollama_host: mock_server.uri(),
            ollama_embed_model: "test-model".to_string(),
            ollama_chat_model: "test-chat".to_string(),
            ollama_api_key: None,
            embed_batch_size: 32,
            embed_tracker_mode: crate::config::TrackerMode::Off,
            embed_tracker_debounce_ms: 0,
            embed_tracker_max_files: 0,
            ignore_dirs: std::collections::HashSet::new(),
            cache_ttl_secs: 300,
            max_embed_file_size: 50 * 1024,
            embed_num_gpu: None,
            embed_main_gpu: None,
            embed_num_thread: None,
            embed_num_batch: None,
            embed_num_ctx: None,
            embed_low_vram: None,
            idle_timeout_ms: 0,
            parent_poll_ms: 5000,
            embed_chunk_chars: 2000,
        };
        let ollama = OllamaClient::new(&config);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        // Pre-populate with linked nodes
        store
            .get_graph(&root, |graph| {
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
            })
            .await
            .expect("ok");

        let result = tool_search_memory_graph(
            &store,
            &ollama,
            SearchMemoryGraphOptions {
                root_dir: root,
                query: "auth".to_string(),
                max_depth: Some(1),
                top_k: Some(1),
                edge_filter: None,
            },
        )
        .await
        .expect("ok");

        assert!(result.contains("Direct Matches:"), "got: {result}");
        assert!(result.contains("Linked Neighbors:"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_search_memory_graph_invalid_edge_filter() {
        use wiremock::MockServer;

        let mock_server = MockServer::start().await;
        let config = crate::config::Config {
            ollama_host: mock_server.uri(),
            ollama_embed_model: "test-model".to_string(),
            ollama_chat_model: "test-chat".to_string(),
            ollama_api_key: None,
            embed_batch_size: 32,
            embed_tracker_mode: crate::config::TrackerMode::Off,
            embed_tracker_debounce_ms: 0,
            embed_tracker_max_files: 0,
            ignore_dirs: std::collections::HashSet::new(),
            cache_ttl_secs: 300,
            max_embed_file_size: 50 * 1024,
            embed_num_gpu: None,
            embed_main_gpu: None,
            embed_num_thread: None,
            embed_num_batch: None,
            embed_num_ctx: None,
            embed_low_vram: None,
            idle_timeout_ms: 0,
            parent_poll_ms: 5000,
            embed_chunk_chars: 2000,
        };
        let ollama = OllamaClient::new(&config);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_search_memory_graph(
            &store,
            &ollama,
            SearchMemoryGraphOptions {
                root_dir: root,
                query: "test".to_string(),
                max_depth: None,
                top_k: None,
                edge_filter: Some(vec!["bad_filter".to_string()]),
            },
        )
        .await;

        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // tool_add_interlinked_context (requires mock Ollama)
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn tool_add_interlinked_context_single_item() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let embed_response = serde_json::json!({
            "embeddings": [[0.5, 0.5, 0.5]]
        });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&embed_response))
            .mount(&mock_server)
            .await;

        let config = crate::config::Config {
            ollama_host: mock_server.uri(),
            ollama_embed_model: "test-model".to_string(),
            ollama_chat_model: "test-chat".to_string(),
            ollama_api_key: None,
            embed_batch_size: 32,
            embed_tracker_mode: crate::config::TrackerMode::Off,
            embed_tracker_debounce_ms: 0,
            embed_tracker_max_files: 0,
            ignore_dirs: std::collections::HashSet::new(),
            cache_ttl_secs: 300,
            max_embed_file_size: 50 * 1024,
            embed_num_gpu: None,
            embed_main_gpu: None,
            embed_num_thread: None,
            embed_num_batch: None,
            embed_num_ctx: None,
            embed_low_vram: None,
            idle_timeout_ms: 0,
            parent_poll_ms: 5000,
            embed_chunk_chars: 2000,
        };
        let ollama = OllamaClient::new(&config);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_add_interlinked_context(
            &store,
            &ollama,
            AddInterlinkedContextOptions {
                root_dir: root,
                items: vec![InterlinkedItem {
                    node_type: "concept".to_string(),
                    label: "solo-node".to_string(),
                    content: "just one node".to_string(),
                    metadata: None,
                }],
                auto_link: Some(true),
            },
        )
        .await
        .expect("ok");

        assert!(
            result.contains("Added 1 interlinked nodes"),
            "got: {result}"
        );
        assert!(result.contains("[concept] solo-node"), "got: {result}");
        assert!(result.contains("Graph total:"), "got: {result}");
    }

    #[tokio::test]
    async fn tool_add_interlinked_context_no_auto_link() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let embed_response = serde_json::json!({
            "embeddings": [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]
        });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&embed_response))
            .mount(&mock_server)
            .await;

        let config = crate::config::Config {
            ollama_host: mock_server.uri(),
            ollama_embed_model: "test-model".to_string(),
            ollama_chat_model: "test-chat".to_string(),
            ollama_api_key: None,
            embed_batch_size: 32,
            embed_tracker_mode: crate::config::TrackerMode::Off,
            embed_tracker_debounce_ms: 0,
            embed_tracker_max_files: 0,
            ignore_dirs: std::collections::HashSet::new(),
            cache_ttl_secs: 300,
            max_embed_file_size: 50 * 1024,
            embed_num_gpu: None,
            embed_main_gpu: None,
            embed_num_thread: None,
            embed_num_batch: None,
            embed_num_ctx: None,
            embed_low_vram: None,
            idle_timeout_ms: 0,
            parent_poll_ms: 5000,
            embed_chunk_chars: 2000,
        };
        let ollama = OllamaClient::new(&config);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_add_interlinked_context(
            &store,
            &ollama,
            AddInterlinkedContextOptions {
                root_dir: root,
                items: vec![
                    InterlinkedItem {
                        node_type: "concept".to_string(),
                        label: "A".to_string(),
                        content: "node a".to_string(),
                        metadata: None,
                    },
                    InterlinkedItem {
                        node_type: "concept".to_string(),
                        label: "B".to_string(),
                        content: "node b".to_string(),
                        metadata: None,
                    },
                ],
                auto_link: Some(false),
            },
        )
        .await
        .expect("ok");

        assert!(
            result.contains("Added 2 interlinked nodes"),
            "got: {result}"
        );
        assert!(
            result.contains("No auto-links above threshold"),
            "got: {result}"
        );
    }

    #[tokio::test]
    async fn tool_add_interlinked_context_invalid_node_type() {
        use wiremock::MockServer;

        let mock_server = MockServer::start().await;
        let config = crate::config::Config {
            ollama_host: mock_server.uri(),
            ollama_embed_model: "test-model".to_string(),
            ollama_chat_model: "test-chat".to_string(),
            ollama_api_key: None,
            embed_batch_size: 32,
            embed_tracker_mode: crate::config::TrackerMode::Off,
            embed_tracker_debounce_ms: 0,
            embed_tracker_max_files: 0,
            ignore_dirs: std::collections::HashSet::new(),
            cache_ttl_secs: 300,
            max_embed_file_size: 50 * 1024,
            embed_num_gpu: None,
            embed_main_gpu: None,
            embed_num_thread: None,
            embed_num_batch: None,
            embed_num_ctx: None,
            embed_low_vram: None,
            idle_timeout_ms: 0,
            parent_poll_ms: 5000,
            embed_chunk_chars: 2000,
        };
        let ollama = OllamaClient::new(&config);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_add_interlinked_context(
            &store,
            &ollama,
            AddInterlinkedContextOptions {
                root_dir: root,
                items: vec![InterlinkedItem {
                    node_type: "invalid_type".to_string(),
                    label: "test".to_string(),
                    content: "test".to_string(),
                    metadata: None,
                }],
                auto_link: None,
            },
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn tool_add_interlinked_context_default_auto_link_is_true() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        // Return identical embeddings so auto-linking triggers (2 items = 2 embeddings)
        let embed_response = serde_json::json!({
            "embeddings": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        });
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&embed_response))
            .mount(&mock_server)
            .await;

        let config = crate::config::Config {
            ollama_host: mock_server.uri(),
            ollama_embed_model: "test-model".to_string(),
            ollama_chat_model: "test-chat".to_string(),
            ollama_api_key: None,
            embed_batch_size: 32,
            embed_tracker_mode: crate::config::TrackerMode::Off,
            embed_tracker_debounce_ms: 0,
            embed_tracker_max_files: 0,
            ignore_dirs: std::collections::HashSet::new(),
            cache_ttl_secs: 300,
            max_embed_file_size: 50 * 1024,
            embed_num_gpu: None,
            embed_main_gpu: None,
            embed_num_thread: None,
            embed_num_batch: None,
            embed_num_ctx: None,
            embed_low_vram: None,
            idle_timeout_ms: 0,
            parent_poll_ms: 5000,
            embed_chunk_chars: 2000,
        };
        let ollama = OllamaClient::new(&config);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = tool_add_interlinked_context(
            &store,
            &ollama,
            AddInterlinkedContextOptions {
                root_dir: root,
                items: vec![
                    InterlinkedItem {
                        node_type: "concept".to_string(),
                        label: "X".to_string(),
                        content: "x content".to_string(),
                        metadata: None,
                    },
                    InterlinkedItem {
                        node_type: "concept".to_string(),
                        label: "Y".to_string(),
                        content: "y content".to_string(),
                        metadata: None,
                    },
                ],
                auto_link: None, // defaults to true
            },
        )
        .await
        .expect("ok");

        // With identical embeddings (cosine similarity = 1.0), auto-link should fire
        assert!(result.contains("Auto-linked:"), "got: {result}");
        assert!(result.contains("similarity edges"), "got: {result}");
    }

    // ---------------------------------------------------------------
    // Options structs construction
    // ---------------------------------------------------------------

    #[test]
    fn upsert_memory_node_options_construction() {
        let opts = UpsertMemoryNodeOptions {
            root_dir: "/tmp/test".to_string(),
            node_type: "concept".to_string(),
            label: "test".to_string(),
            content: "test content".to_string(),
            metadata: None,
        };
        assert_eq!(opts.root_dir, "/tmp/test");
        assert_eq!(opts.node_type, "concept");
        assert_eq!(opts.label, "test");
        assert_eq!(opts.content, "test content");
        assert!(opts.metadata.is_none());
    }

    #[test]
    fn create_relation_options_construction() {
        let opts = CreateRelationOptions {
            root_dir: "/tmp/test".to_string(),
            source_id: None,
            source_label: Some("src".to_string()),
            source_type: "concept".to_string(),
            target_id: None,
            target_label: Some("tgt".to_string()),
            target_type: "file".to_string(),
            relation: "depends_on".to_string(),
            weight: Some(0.5),
            metadata: None,
        };
        assert_eq!(opts.source_label, Some("src".to_string()));
        assert_eq!(opts.target_label, Some("tgt".to_string()));
        assert_eq!(opts.weight, Some(0.5));
    }

    #[test]
    fn search_memory_graph_options_construction() {
        let opts = SearchMemoryGraphOptions {
            root_dir: "/tmp/test".to_string(),
            query: "find auth".to_string(),
            max_depth: Some(3),
            top_k: Some(10),
            edge_filter: Some(vec!["depends_on".to_string()]),
        };
        assert_eq!(opts.query, "find auth");
        assert_eq!(opts.max_depth, Some(3));
        assert_eq!(opts.top_k, Some(10));
    }

    #[test]
    fn prune_stale_links_options_construction() {
        let opts = PruneStaleLinksOptions {
            root_dir: "/tmp/test".to_string(),
            threshold: Some(0.25),
        };
        assert_eq!(opts.threshold, Some(0.25));
    }

    #[test]
    fn add_interlinked_context_options_construction() {
        let item = InterlinkedItem {
            node_type: "file".to_string(),
            label: "main.rs".to_string(),
            content: "entry point".to_string(),
            metadata: Some(HashMap::from([("lang".to_string(), "rust".to_string())])),
        };
        assert_eq!(item.node_type, "file");
        assert_eq!(item.label, "main.rs");
        assert!(item.metadata.is_some());

        let opts = AddInterlinkedContextOptions {
            root_dir: "/tmp/test".to_string(),
            items: vec![item],
            auto_link: Some(false),
        };
        assert_eq!(opts.items.len(), 1);
        assert_eq!(opts.auto_link, Some(false));
    }

    #[test]
    fn retrieve_with_traversal_options_construction() {
        let opts = RetrieveWithTraversalOptions {
            root_dir: "/tmp/test".to_string(),
            node_id: "mn-12345".to_string(),
            max_depth: Some(4),
            edge_filter: None,
        };
        assert_eq!(opts.node_id, "mn-12345");
        assert_eq!(opts.max_depth, Some(4));
        assert!(opts.edge_filter.is_none());
    }

    // ---------------------------------------------------------------
    // Existing integration tests
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn graph_store_search_without_ollama() {
        // Test that we can use the graph store directly without Ollama
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        // Directly manipulate the graph
        let stats = store
            .get_graph(&root, |graph| {
                graph.upsert_node(
                    NodeType::Concept,
                    "test",
                    "test content",
                    make_embedding(0.5),
                    None,
                );
                graph.stats()
            })
            .await
            .expect("ok");

        assert_eq!(stats.nodes, 1);
    }

    #[tokio::test]
    async fn graph_store_prune_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let result = store
            .get_graph(&root, |graph| graph.prune_stale_links(None))
            .await
            .expect("ok");

        assert_eq!(result.removed, 0);
        assert_eq!(result.remaining_edges, 0);
    }

    #[tokio::test]
    async fn graph_store_traverse_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_string_lossy().to_string();
        let store = GraphStore::new();

        let results = store
            .get_graph(&root, |graph| {
                graph.retrieve_with_traversal("nonexistent", 2, None)
            })
            .await
            .expect("ok");

        assert!(results.is_empty());
    }
}
