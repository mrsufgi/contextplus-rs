// MCP tool wrappers for memory graph operations and interlinked RAG.
// 6 tools: upsert, relate, search, prune, interlink, traverse.

use crate::core::memory_graph::{
    GraphStore, NodeType, RelationType, TraversalResult,
};
use crate::error::{ContextPlusError, Result};
use reqwest::Client;
use serde::Deserialize;
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
    pub source_label: String,
    pub source_type: String,
    pub target_label: String,
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
        format!("{}...", &result.node.content[..120])
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
    NodeType::from_str(s).ok_or_else(|| {
        ContextPlusError::Other(format!(
            "Invalid node type: '{}'. Valid types: concept, file, symbol, note",
            s
        ))
    })
}

fn parse_relation_type(s: &str) -> Result<RelationType> {
    RelationType::from_str(s).ok_or_else(|| {
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

/// Fetch embedding for a text string from Ollama.
async fn fetch_embedding(text: &str) -> Result<Vec<f32>> {
    let ollama_host =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let embed_model =
        std::env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());

    let client = Client::new();
    let body = serde_json::json!({
        "model": embed_model,
        "input": [text],
    });

    let resp = client
        .post(format!("{}/api/embed", ollama_host))
        .json(&body)
        .send()
        .await
        .map_err(|e| ContextPlusError::Ollama(format!("Embedding request failed: {}", e)))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(ContextPlusError::Ollama(format!(
            "Ollama returned {}: {}",
            status, text
        )));
    }

    #[derive(Deserialize)]
    struct EmbedResponse {
        embeddings: Vec<Vec<f32>>,
    }

    let embed_resp: EmbedResponse = resp.json().await.map_err(|e| {
        ContextPlusError::Ollama(format!("Failed to parse embedding response: {}", e))
    })?;

    embed_resp
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| ContextPlusError::Ollama("No embedding returned".to_string()))
}

// --- Tool implementations ---

/// Tool 1: Upsert a memory node. Creates or updates by (label, type).
pub async fn tool_upsert_memory_node(
    store: &GraphStore,
    options: UpsertMemoryNodeOptions,
) -> Result<String> {
    let node_type = parse_node_type(&options.node_type)?;
    let embed_text = format!("{} {}", options.label, options.content);
    let embedding = fetch_embedding(&embed_text).await?;

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

/// Tool 2: Create a relation between two nodes found by label+type.
pub async fn tool_create_relation(
    store: &GraphStore,
    options: CreateRelationOptions,
) -> Result<String> {
    let relation = parse_relation_type(&options.relation)?;
    let source_type = parse_node_type(&options.source_type)?;
    let target_type = parse_node_type(&options.target_type)?;

    // Look up node IDs by label+type
    let (source_id, target_id) = store
        .get_graph(&options.root_dir, |graph| {
            let source = graph
                .find_node(&options.source_label, &source_type)
                .map(|n| n.id.clone());
            let target = graph
                .find_node(&options.target_label, &target_type)
                .map(|n| n.id.clone());
            (source, target)
        })
        .await?;

    let source_id = match source_id {
        Some(id) => id,
        None => {
            return Ok(format!(
                "Failed: source node not found (label: '{}', type: '{}')",
                options.source_label, options.source_type
            ));
        }
    };
    let target_id = match target_id {
        Some(id) => id,
        None => {
            return Ok(format!(
                "Failed: target node not found (label: '{}', type: '{}')",
                options.target_label, options.target_type
            ));
        }
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
    options: SearchMemoryGraphOptions,
) -> Result<String> {
    let max_depth = options.max_depth.unwrap_or(1);
    let top_k = options.top_k.unwrap_or(5);
    let edge_filter = parse_edge_filter(&options.edge_filter)?;

    let query_embedding = fetch_embedding(&options.query).await?;

    let result = store
        .get_graph(&options.root_dir, |graph| {
            graph.search(
                &query_embedding,
                max_depth,
                top_k,
                edge_filter.as_deref(),
            )
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
    options: AddInterlinkedContextOptions,
) -> Result<String> {
    let auto_link = options.auto_link.unwrap_or(true);

    // Prepare items with embeddings
    let mut prepared_items = Vec::new();
    for item in &options.items {
        let node_type = parse_node_type(&item.node_type)?;
        let embed_text = format!("{} {}", item.label, item.content);
        let embedding = fetch_embedding(&embed_text).await?;
        prepared_items.push((
            node_type,
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
            sections.push(format!(
                "  --[{} w:{:.2}]-->",
                edge.relation, edge.weight
            ));
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

    #[test]
    fn parse_node_type_valid() {
        assert!(parse_node_type("concept").is_ok());
        assert!(parse_node_type("file").is_ok());
        assert!(parse_node_type("symbol").is_ok());
        assert!(parse_node_type("note").is_ok());
    }

    #[test]
    fn parse_node_type_invalid() {
        assert!(parse_node_type("invalid").is_err());
    }

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
    fn parse_relation_type_invalid() {
        assert!(parse_relation_type("invalid").is_err());
    }

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
    fn parse_edge_filter_invalid() {
        let filter = Some(vec!["invalid".to_string()]);
        assert!(parse_edge_filter(&filter).is_err());
    }

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
