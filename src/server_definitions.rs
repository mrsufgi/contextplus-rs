// Tool definitions — built once via LazyLock, returned as &'static [Tool].
// Extracted from server.rs (Round 11D) to keep server.rs focused on dispatch logic.

use std::sync::{Arc, LazyLock};

use rmcp::model::Tool;
use serde_json::Value;

/// All 26 tool definitions. Built once at first access, reused for every list_tools call.
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
            "Find every file that imports or references a symbol. Maps the full impact of changing it. Scans one worktree's indexed tree, so a zero-usage result only means \"unused in the scanned worktree\" — not globally unused.",
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
                (
                    "path",
                    "string",
                    false,
                    "Root of an attached worktree to scan instead of the current one. Must already be attached via attach_worktree. Use this to trace a symbol on a feature branch from the primary checkout.",
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
                "Specific file or folder to lint. Relative to the active ref's root, or absolute — absolute paths under a registered worktree's root auto-route the linter to that worktree. Omit for full project.",
            )],
        ),
        make_tool(
            "attach_worktree",
            "Register a worktree directory as a ref that inherits the primary ref's embedding cache via CoW (CAS parent pointer + memory-graph overlay), then spawns per-ref warmup. Required for analyzing worktrees outside the daemon's primary root without a per-worktree MCP handshake. Idempotent.",
            &[(
                "path",
                "string",
                true,
                "Absolute or relative path to the worktree directory; will be canonicalized. Must exist and be a directory.",
            )],
        ),
        make_tool(
            "detach_worktree",
            "Detach a previously-attached worktree. Decrements its session count; once it reaches zero the ref enters the TTL eviction queue. Refuses to detach the primary ref.",
            &[(
                "path",
                "string",
                true,
                "Worktree path used at attach time (or any path that canonicalizes to the same root).",
            )],
        ),
        make_tool(
            "list_worktrees",
            "List every ref currently in the registry — the primary plus any attached worktrees — with their canonical roots, session counts, and HEAD SHAs.",
            &[],
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
        // --- 5 new tools wired in this PR ---
        make_tool(
            "find_dead_code",
            "Heuristic scan for potentially unused symbols. Reports symbols whose names do not appear as tokens in any other indexed file. Advisory only — scans one worktree's indexed tree, so a symbol used on another branch/worktree can show up as a false positive.",
            &[
                (
                    "ignore_kinds",
                    "array",
                    false,
                    "Symbol kinds to skip (default: [\"mod\",\"impl\",\"trait\",\"test\"]). Pass [] to include all.",
                ),
                (
                    "ignore_names",
                    "array",
                    false,
                    "Symbol names to skip (default: common entry-points like \"main\",\"new\",\"default\"). Pass [] to include all.",
                ),
                (
                    "max_results",
                    "integer",
                    false,
                    "Cap on number of reported candidates (default 200).",
                ),
                (
                    "path",
                    "string",
                    false,
                    "Root of an attached worktree to scan instead of the current one. Must already be attached via attach_worktree.",
                ),
            ],
        ),
        make_tool(
            "review_pr_diff",
            "Analyse a unified diff and produce a risk-ranked impact report. Identifies changed symbols, expands to 2-hop dependent files, and ranks all affected files by composite risk score.",
            &[
                (
                    "diff",
                    "string",
                    true,
                    "Unified diff text (output of `git diff` or similar).",
                ),
                (
                    "max_hops",
                    "integer",
                    false,
                    "Dependency expansion depth (default 2).",
                ),
                (
                    "max_files",
                    "integer",
                    false,
                    "Cap on total files surfaced (default 500).",
                ),
            ],
        ),
        make_tool(
            "detect_dependency_loops",
            "Detect import cycles in the project using Tarjan SCC algorithm. Returns all strongly-connected components with >= 2 files, plus self-importing files.",
            &[],
        ),
        make_tool(
            "check_embedding_quality",
            "Diagnose the in-memory embedding cache: reports zero vectors, NaN/Inf values, dimension mismatches, and duplicate vectors.",
            &[(
                "expected_dim",
                "integer",
                false,
                "Expected embedding dimensionality. Auto-detected from first cached vector if omitted.",
            )],
        ),
        make_tool(
            "lexical_search",
            "Fast in-process TF-IDF lexical search over all indexed files. Complements semantic_code_search for exact-keyword and camelCase identifier queries.",
            &[
                (
                    "query",
                    "string",
                    true,
                    "Keyword or identifier query (camelCase is split into sub-tokens automatically).",
                ),
                (
                    "top_k",
                    "integer",
                    false,
                    "Number of results to return (default 10).",
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
