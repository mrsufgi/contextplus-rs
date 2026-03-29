// Tool definitions — built once via LazyLock, returned as &'static [Tool].
// Extracted from server.rs (Round 11D) to keep server.rs focused on dispatch logic.

use std::sync::{Arc, LazyLock};

use rmcp::model::Tool;
use serde_json::Value;

/// All 17 tool definitions. Built once at first access, reused for every list_tools call.
static TOOL_DEFINITIONS: LazyLock<Vec<Tool>> = LazyLock::new(build_tool_definitions);

/// Returns the static tool definitions slice. O(1) after first call.
pub fn tool_definitions() -> &'static [Tool] {
    &TOOL_DEFINITIONS
}

fn build_tool_definitions() -> Vec<Tool> {
    vec![
        make_tool(
            "get_context_tree",
            "Build a token-aware context tree showing file structure and symbols. Prunes detail levels based on max_tokens budget.",
            &[
                (
                    "target_path",
                    "string",
                    false,
                    "Specific directory or file to analyze (relative to project root)",
                ),
                (
                    "depth_limit",
                    "integer",
                    false,
                    "How many folder levels deep to scan. Use 1-2 for large projects.",
                ),
                (
                    "include_symbols",
                    "boolean",
                    false,
                    "Include function/class/enum names in the tree (default true)",
                ),
                (
                    "max_tokens",
                    "integer",
                    false,
                    "Maximum tokens for output. Auto-prunes if exceeded (default 20000)",
                ),
            ],
        ),
        make_tool(
            "get_file_skeleton",
            "Get function signatures, class definitions, and line ranges for a file without reading full content.",
            &[(
                "file_path",
                "string",
                true,
                "Path to the file to inspect (relative to project root)",
            )],
        ),
        make_tool(
            "get_blast_radius",
            "Find every file that imports or references a symbol. Maps the full impact of changing it.",
            &[
                (
                    "symbol_name",
                    "string",
                    true,
                    "The function, class, or variable name to trace across the codebase",
                ),
                (
                    "file_context",
                    "string",
                    false,
                    "The file where the symbol is defined. Excludes the definition line from results.",
                ),
            ],
        ),
        make_tool(
            "semantic_code_search",
            "Search code files semantically using natural language queries. Combines embedding similarity with keyword matching for hybrid ranking.",
            &[
                (
                    "query",
                    "string",
                    true,
                    "Natural language description of what you're looking for",
                ),
                (
                    "top_k",
                    "integer",
                    false,
                    "Number of matches to return (default 5, max 50)",
                ),
                (
                    "semantic_weight",
                    "number",
                    false,
                    "Weight for embedding similarity in hybrid ranking (default 0.72)",
                ),
                (
                    "keyword_weight",
                    "number",
                    false,
                    "Weight for keyword overlap in hybrid ranking (default 0.28)",
                ),
                (
                    "min_semantic_score",
                    "number",
                    false,
                    "Minimum semantic score filter (0-1 or 0-100)",
                ),
                (
                    "min_keyword_score",
                    "number",
                    false,
                    "Minimum keyword score filter (0-1 or 0-100)",
                ),
                (
                    "min_combined_score",
                    "number",
                    false,
                    "Minimum final score filter (0-1 or 0-100)",
                ),
                (
                    "require_keyword_match",
                    "boolean",
                    false,
                    "When true, only return files with keyword overlap",
                ),
                (
                    "require_semantic_match",
                    "boolean",
                    false,
                    "When true, only return files with positive semantic similarity",
                ),
            ],
        ),
        make_tool(
            "semantic_identifier_search",
            "Search for functions, classes, and variables by semantic meaning. Returns identifiers with call-site rankings.",
            &[
                (
                    "query",
                    "string",
                    true,
                    "Natural language intent to match identifiers and usages",
                ),
                (
                    "top_k",
                    "integer",
                    false,
                    "How many identifiers to return (default 5)",
                ),
                (
                    "top_calls_per_identifier",
                    "integer",
                    false,
                    "How many ranked call sites per identifier (default 10)",
                ),
                (
                    "include_kinds",
                    "array",
                    false,
                    "Optional kinds filter, e.g. [\"function\", \"method\", \"variable\"]",
                ),
                (
                    "semantic_weight",
                    "number",
                    false,
                    "Weight for semantic similarity score (default 0.78)",
                ),
                (
                    "keyword_weight",
                    "number",
                    false,
                    "Weight for keyword overlap score (default 0.22)",
                ),
            ],
        ),
        make_tool(
            "semantic_navigate",
            "Cluster files by semantic similarity using spectral clustering. Returns labeled groups for codebase navigation. Pass rootDir to scope to a subdirectory.",
            &[
                (
                    "rootDir",
                    "string",
                    false,
                    "Directory to navigate (default: workspace root). Must be within the workspace.",
                ),
                (
                    "max_depth",
                    "integer",
                    false,
                    "Maximum nesting depth of clusters (default 3)",
                ),
                (
                    "max_clusters",
                    "integer",
                    false,
                    "Maximum sub-clusters per group at depth 1+ (default 20). Top-level groups are based on directory structure and not limited by this parameter.",
                ),
                (
                    "min_clusters",
                    "integer",
                    false,
                    "Minimum sub-clusters per group (default 2). Increase to force finer-grained splitting.",
                ),
                (
                    "mode",
                    "string",
                    false,
                    "Clustering mode: 'hybrid' (default, directory-based + spectral, best for CPU), 'semantic' (pure spectral clustering like original contextplus, best with GPU), or 'imports' (blends embedding similarity with import-graph adjacency for structure-aware clustering).",
                ),
            ],
        ),
        make_tool(
            "get_feature_hub",
            "Navigate Obsidian-style wikilinks to discover feature hubs and their connections.",
            &[
                (
                    "hub_path",
                    "string",
                    false,
                    "Path to a specific hub .md file (relative to root)",
                ),
                (
                    "feature_name",
                    "string",
                    false,
                    "Feature name to search for. Finds matching hub file automatically.",
                ),
                (
                    "show_orphans",
                    "boolean",
                    false,
                    "If true, lists all source files not linked to any hub.",
                ),
            ],
        ),
        make_tool(
            "run_static_analysis",
            "Run available linters (tsc, eslint, cargo check, ruff) on the project or a specific file.",
            &[(
                "target_path",
                "string",
                false,
                "Specific file or folder to lint (relative to root). Omit for full project.",
            )],
        ),
        make_tool(
            "propose_commit",
            "Write a file with validation (header, comments, nesting, line count) and create a shadow restore point for undo.",
            &[
                (
                    "file_path",
                    "string",
                    true,
                    "Where to save the file (relative to project root)",
                ),
                (
                    "new_content",
                    "string",
                    true,
                    "The complete file content to save",
                ),
                ("description", "string", false, "Description of the change"),
            ],
        ),
        make_tool(
            "list_restore_points",
            "List all shadow restore points created by propose_commit.",
            &[],
        ),
        make_tool(
            "undo_change",
            "Restore files from a shadow restore point created by propose_commit.",
            &[(
                "point_id",
                "string",
                true,
                "The restore point ID (format: rp-timestamp-hash). Get from list_restore_points.",
            )],
        ),
        make_tool(
            "upsert_memory_node",
            "Create or update a memory graph node. Nodes are uniquely identified by (label, type).",
            &[
                (
                    "type",
                    "string",
                    true,
                    "Node type: concept, file, symbol, note",
                ),
                (
                    "label",
                    "string",
                    true,
                    "Short identifier for the node. Used for deduplication with type.",
                ),
                (
                    "content",
                    "string",
                    true,
                    "Detailed content for the node. Used for embedding generation.",
                ),
                (
                    "metadata",
                    "object",
                    false,
                    "Optional key-value metadata pairs",
                ),
            ],
        ),
        make_tool(
            "create_relation",
            "Create or update a relation between two memory graph nodes.",
            &[
                ("source_id", "string", true, "ID of the source memory node"),
                ("target_id", "string", true, "ID of the target memory node"),
                (
                    "relation",
                    "string",
                    true,
                    "Relationship type: relates_to, depends_on, implements, references, similar_to, contains",
                ),
                (
                    "weight",
                    "number",
                    false,
                    "Edge weight 0-1. Higher = stronger relationship (default 1.0)",
                ),
                (
                    "metadata",
                    "object",
                    false,
                    "Optional key-value metadata for the edge",
                ),
            ],
        ),
        make_tool(
            "search_memory_graph",
            "Search the memory graph by semantic similarity and BFS traversal.",
            &[
                (
                    "query",
                    "string",
                    true,
                    "Natural language query to search the memory graph",
                ),
                (
                    "max_depth",
                    "integer",
                    false,
                    "How many hops to traverse from direct matches (default 1)",
                ),
                (
                    "top_k",
                    "integer",
                    false,
                    "Number of direct matches to return (default 5)",
                ),
                (
                    "edge_filter",
                    "array",
                    false,
                    "Only traverse edges of these types. Omit for all types.",
                ),
            ],
        ),
        make_tool(
            "prune_stale_links",
            "Remove memory graph edges with decayed weight below threshold, and orphan nodes.",
            &[(
                "threshold",
                "number",
                false,
                "Minimum decayed weight to keep an edge (default 0.15). Lower = keep more edges.",
            )],
        ),
        make_tool(
            "add_interlinked_context",
            "Add multiple memory nodes at once with optional auto-linking by semantic similarity.",
            &[
                (
                    "items",
                    "array",
                    true,
                    "Array of nodes to add. Each needs type, label, and content.",
                ),
                (
                    "auto_link",
                    "boolean",
                    false,
                    "Whether to auto-create similarity edges (default true)",
                ),
            ],
        ),
        make_tool(
            "retrieve_with_traversal",
            "Retrieve a memory node and its neighborhood via BFS traversal with depth penalty.",
            &[
                (
                    "start_node_id",
                    "string",
                    true,
                    "ID of the memory node to start traversal from",
                ),
                (
                    "max_depth",
                    "integer",
                    false,
                    "Maximum traversal depth from start node (default 2)",
                ),
                (
                    "edge_filter",
                    "array",
                    false,
                    "Only traverse edges of these types. Omit for all.",
                ),
            ],
        ),
    ]
}

pub fn make_tool(name: &str, description: &str, params: &[(&str, &str, bool, &str)]) -> Tool {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    for (pname, ptype, is_required, pdesc) in params {
        let mut prop = serde_json::Map::new();
        prop.insert("type".into(), Value::String(ptype.to_string()));
        prop.insert("description".into(), Value::String(pdesc.to_string()));
        properties.insert(pname.to_string(), Value::Object(prop));
        if *is_required {
            required.push(Value::String(pname.to_string()));
        }
    }

    let mut schema = serde_json::Map::new();
    schema.insert("type".into(), Value::String("object".into()));
    schema.insert("properties".into(), Value::Object(properties));
    if !required.is_empty() {
        schema.insert("required".into(), Value::Array(required));
    }

    Tool::new_with_raw(
        name.to_string(),
        Some(description.to_string().into()),
        Arc::new(schema),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_returns_17_tools() {
        let defs = tool_definitions();
        assert_eq!(defs.len(), 17, "expected 17 tools, got {}", defs.len());
        for tool in defs {
            assert!(!tool.name.is_empty(), "tool name must not be empty");
            assert!(
                tool.description.is_some(),
                "tool '{}' must have a description",
                tool.name
            );
        }
    }

    #[test]
    fn tool_definitions_is_static_pointer_stable() {
        // Calling twice should return the same pointer (LazyLock)
        let a = tool_definitions().as_ptr();
        let b = tool_definitions().as_ptr();
        assert_eq!(a, b, "LazyLock should return the same allocation");
    }

    #[test]
    fn make_tool_sets_required_params() {
        let tool = make_tool(
            "test_tool",
            "A test tool",
            &[
                ("required_param", "string", true, "A required param"),
                ("optional_param", "integer", false, "An optional param"),
            ],
        );
        assert_eq!(tool.name.as_ref(), "test_tool");
        assert_eq!(tool.description.as_deref(), Some("A test tool"));

        let schema = tool.input_schema.as_ref();
        let required = schema.get("required").and_then(|v| v.as_array()).unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].as_str(), Some("required_param"));
    }
}
