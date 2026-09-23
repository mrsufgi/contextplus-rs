// Tool definitions — built once via LazyLock, returned as &'static [Tool].
// Extracted from server.rs (Round 11D) to keep server.rs focused on dispatch logic.

use std::sync::{Arc, LazyLock};

use rmcp::model::Tool;
use serde_json::Value;

/// The five tools the server lists. Built once at first access, reused for every list_tools call.
/// The pre-facade names still dispatch (see `ContextPlusServer::dispatch_inner`) but are not listed.
static TOOL_DEFINITIONS: LazyLock<Vec<Tool>> = LazyLock::new(build_tool_definitions);

/// Returns the static tool definitions slice. O(1) after first call.
pub fn tool_definitions() -> &'static [Tool] {
    &TOOL_DEFINITIONS
}

fn build_tool_definitions() -> Vec<Tool> {
    vec![
        make_tool(
            "explore",
            "Find code by what it does. Start here for any question about where something lives or how a concept is implemented; results carry file paths and line ranges to pass to outline.",
            &[
                (
                    "query",
                    "string",
                    true,
                    "What you are looking for: plain words for match = meaning, the exact identifier or keyword for match = keywords.",
                ),
                (
                    "kind",
                    "string",
                    false,
                    "files (default): ranked source files; identifiers: functions, classes and variables with their call sites; clusters: the codebase grouped by topic.",
                ),
                (
                    "match",
                    "string",
                    false,
                    "meaning (default): embedding similarity plus keyword overlap; keywords: exact tokens and camelCase parts only, no embeddings, fastest.",
                ),
                (
                    "top_k",
                    "integer",
                    false,
                    "Number of results (default 5, max 50).",
                ),
                (
                    "path",
                    "string",
                    false,
                    "Directory to scope the search, or a path inside another worktree of this repo to search there.",
                ),
            ],
        ),
        make_tool(
            "outline",
            "Structure without reading bodies: a file's signatures and line ranges, or a directory's file and symbol tree. Call on a file before reading it and on a directory before exploring it.",
            &[
                (
                    "path",
                    "string",
                    true,
                    "A file or a directory, relative to the worktree root or absolute.",
                ),
                (
                    "depth",
                    "integer",
                    false,
                    "Directories only: how many folder levels to include (1-2 for large trees).",
                ),
                (
                    "max_tokens",
                    "integer",
                    false,
                    "Directories only: output budget; detail is pruned to fit (default 20000).",
                ),
            ],
        ),
        make_tool(
            "impact",
            "What breaks if this changes. Call before modifying or deleting any symbol: every file and line that imports or references it. Give it a unified diff instead to risk-rank a whole change (changed symbols, dependents two hops out, files to read first). Also: the project's import cycles, or symbols nothing references.",
            &[
                (
                    "symbol",
                    "string",
                    false,
                    "Function, class, type or variable name (required for what = symbol unless diff is given).",
                ),
                (
                    "diff",
                    "string",
                    false,
                    "A unified diff to rank instead of a single symbol.",
                ),
                (
                    "max_hops",
                    "integer",
                    false,
                    "diff only: how far to follow dependents (default 2).",
                ),
                (
                    "max_files",
                    "integer",
                    false,
                    "diff only: cap on files in the report.",
                ),
                (
                    "file",
                    "string",
                    false,
                    "File that defines the symbol, when the name is ambiguous.",
                ),
                (
                    "what",
                    "string",
                    false,
                    "symbol (default): users of the symbol; cycles: import cycles (Tarjan SCC); dead: symbols no other indexed file names, advisory only.",
                ),
                (
                    "path",
                    "string",
                    false,
                    "Root of another worktree of this repo to scan instead of the current one.",
                ),
                (
                    "max_results",
                    "integer",
                    false,
                    "what = dead: cap on reported symbols.",
                ),
            ],
        ),
        make_tool(
            "check",
            "Run the project's own linters and compilers after an edit (tsc, eslint, cargo check, ruff), or audit the search index when results look wrong.",
            &[
                (
                    "path",
                    "string",
                    false,
                    "File or directory to check; omit for the whole project. A path inside another worktree checks that worktree.",
                ),
                (
                    "what",
                    "string",
                    false,
                    "lint (default): linters and compilers on the target; embeddings: zero, NaN, mismatched or duplicate vectors in the embedding cache.",
                ),
            ],
        ),
        make_tool(
            "worktrees",
            "Show or pin git worktrees. Any call whose path lies inside a worktree attaches it automatically; attach pins one for the session, detach releases it.",
            &[
                (
                    "action",
                    "string",
                    false,
                    "list (default), attach or detach.",
                ),
                (
                    "path",
                    "string",
                    false,
                    "Worktree directory for attach and detach.",
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
