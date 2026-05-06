//! Path translation primitives for the dispatch boundary.
//!
//! When a daemon serves multiple worktrees it must ensure that:
//! - **Inputs**: absolute paths supplied by a caller are stripped of the
//!   caller's worktree prefix and stored as repo-relative form before being
//!   forwarded to tool implementations.
//! - **Outputs**: repo-relative paths embedded in a tool's text output are
//!   prefixed with the calling session's worktree absolute root so the caller
//!   sees paths that are valid inside its own worktree.
//! - **Out-of-tree paths**: a caller-supplied absolute path that does _not_
//!   start with the caller's worktree root is rejected with a clear error
//!   rather than silently mangled.
//!
//! All functions in this module are pure (no I/O, no async). They are
//! exercised by unit tests that use synthetic worktree roots; the dispatch
//! layer in `src/transport/dispatch.rs` applies them at the tool-call
//! boundary.
//!
//! ## Seam for U4 integration
//!
//! Today the caller's worktree root is obtained from
//! `SharedState.default_ref().root_dir`. When U4 lands and introduces
//! `session.ref_id`, the dispatch layer will replace that with
//! `state.ref_index(session.ref_id).root_dir` — the translation primitives
//! themselves don't change.

use std::path::Path;

/// Errors returned by path-translation functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathTranslationError {
    /// The supplied absolute path is not inside the caller's worktree root.
    OutOfTree {
        /// The path that was rejected.
        path: String,
        /// The worktree root the path was expected to be inside.
        worktree_root: String,
    },
}

impl std::fmt::Display for PathTranslationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfTree {
                path,
                worktree_root,
            } => write!(
                f,
                "Caller-provided path {path:?} is outside the caller's worktree root \
                 {worktree_root:?}; rejecting to prevent cross-session path leakage"
            ),
        }
    }
}

impl std::error::Error for PathTranslationError {}

/// Translate a caller-supplied *input* path into repo-relative form.
///
/// If `raw` is already relative, it is returned as-is (it was already
/// repo-relative — the tool will prepend its own `root_dir`).
///
/// If `raw` is absolute it must start with `caller_root`; the prefix is
/// stripped and the remainder returned. If the absolute path is outside
/// `caller_root` an [`PathTranslationError::OutOfTree`] is returned.
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use contextplus::core::path_translation::translate_input_path;
///
/// let root = Path::new("/home/user/repo");
/// assert_eq!(
///     translate_input_path("/home/user/repo/src/foo.rs", root).unwrap(),
///     "src/foo.rs"
/// );
/// // Relative paths are passed through.
/// assert_eq!(
///     translate_input_path("src/foo.rs", root).unwrap(),
///     "src/foo.rs"
/// );
/// ```
pub fn translate_input_path(raw: &str, caller_root: &Path) -> Result<String, PathTranslationError> {
    let p = Path::new(raw);
    if p.is_absolute() {
        match p.strip_prefix(caller_root) {
            Ok(rel) => Ok(rel.to_string_lossy().to_string()),
            Err(_) => Err(PathTranslationError::OutOfTree {
                path: raw.to_string(),
                worktree_root: caller_root.to_string_lossy().to_string(),
            }),
        }
    } else {
        // Already relative — pass through.
        Ok(raw.to_string())
    }
}

/// Translate a repo-relative path from tool output into the caller's
/// absolute path by prepending `caller_root`.
///
/// This is the inverse of [`translate_input_path`]: the tool stores paths
/// as repo-relative; the caller expects absolute paths anchored at its own
/// worktree root.
///
/// `rel` must be a relative path. If it is absolute it is returned
/// unchanged (it was already resolved to an absolute path by some earlier
/// stage; the caller would not expect further prefixing).
pub fn translate_output_path(rel: &str, caller_root: &Path) -> String {
    let p = Path::new(rel);
    if p.is_relative() {
        caller_root.join(p).to_string_lossy().to_string()
    } else {
        // Already absolute — preserve.
        rel.to_string()
    }
}

/// Rewrite every occurrence of a repo-relative path inside a free-form
/// text string, replacing it with an absolute path rooted at `caller_root`.
///
/// "Occurrence" means the exact relative-path string appears in the text
/// surrounded by either:
/// - a non-path character (`\n`, ` `, `(`, `)`, `"`, `'`, `,`, `[`, `]`,
///   `{`, `}`), or
/// - the start / end of the string.
///
/// This is deliberately conservative: we only rewrite paths that look like
/// file-system paths (`foo/bar`, `src/main.rs`, etc.) — sequences that
/// contain `/` and no absolute-path marker. Paths that are already absolute
/// are left untouched (they were produced from the server's own `root_dir`
/// which matches the caller's worktree in the current single-ref path; and
/// in the multi-ref future they will already have been rewritten before
/// this point).
///
/// ## Stdio no-op
///
/// When `caller_root` is the same path that the tool used internally
/// (single-daemon, single-worktree), prepending the root to relative paths
/// and then the caller reading them from under the same root produces the
/// correct absolute path. The translation is effectively a no-op _for the
/// caller_ because the absolute path is identical to what the tool would
/// have produced if it had emitted absolute paths directly.
pub fn rewrite_paths_in_text(text: &str, caller_root: &Path) -> String {
    // Fast path: if there is nothing that looks like a path separator there
    // is nothing to rewrite.
    if !text.contains('/') {
        return text.to_string();
    }

    let caller_root_str = caller_root.to_string_lossy();

    // If the text already contains the caller's absolute root we don't need
    // to rewrite — all occurrences are already correct absolute paths.
    // (This is the common case for stdio mode where root_dir == caller_root.)
    if text.contains(caller_root_str.as_ref()) {
        return text.to_string();
    }

    // Walk through the text and replace relative path-like tokens.
    // A "path-like token" starts at a word boundary and matches the regex
    // pattern [a-zA-Z0-9_.\-][a-zA-Z0-9_./-]*/[a-zA-Z0-9_./\-]+ .
    // We implement this with a simple hand-rolled scanner to avoid a regex dep.
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut result = String::with_capacity(text.len() + 32);
    let mut i = 0;

    while i < len {
        // Check if we're at the start of a relative path.
        // A relative path candidate must:
        //   1. Start at position 0 or after a delimiter character.
        //   2. Not start with `/` (that would be absolute).
        //   3. Not start with `~` (home-dir shorthand, not handled).
        //   4. Contain at least one `/`.
        let at_boundary = i == 0 || is_path_delimiter(bytes[i - 1]);
        if at_boundary
            && bytes[i] != b'/'
            && bytes[i] != b'~'
            && let Some(end) = find_path_end(bytes, i)
        {
            let candidate = &text[i..end];
            // Only rewrite if it contains `/` (it's a path, not a bare word).
            // Also skip if it looks like a URL scheme ("https://", etc.)
            if candidate.contains('/') && !candidate.contains("://") {
                let abs = format!("{}/{}", caller_root_str, candidate);
                result.push_str(&abs);
                i = end;
                continue;
            }
        }
        // Copy current byte verbatim.
        result.push(bytes[i] as char);
        i += 1;
    }

    result
}

/// Returns `true` if `c` is a character that can appear immediately before
/// a path in free-form tool output (whitespace or common punctuation).
fn is_path_delimiter(c: u8) -> bool {
    matches!(
        c,
        b'\n'
            | b'\r'
            | b'\t'
            | b' '
            | b'('
            | b')'
            | b'"'
            | b'\''
            | b','
            | b'['
            | b']'
            | b'{'
            | b'}'
    )
}

/// Scan forward from `start` to find the end of a path-like token.
///
/// A path character is: alphanumeric, `_`, `.`, `-`, `/`.
/// We stop at the first character that is not a path character.
fn find_path_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    while i < bytes.len() && is_path_char(bytes[i]) {
        i += 1;
    }
    if i > start { Some(i) } else { None }
}

fn is_path_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'-' | b'/')
}

/// Check whether a JSON value (recursively) contains a given string anywhere
/// in its text content (string values or object keys).
///
/// Used by tests to assert absence of cross-session path leakage: a caller
/// must never receive in a tool response any string that belongs to another
/// session's worktree root.
pub fn json_contains_string(value: &serde_json::Value, needle: &str) -> bool {
    match value {
        serde_json::Value::String(s) => s.contains(needle),
        serde_json::Value::Array(arr) => arr.iter().any(|v| json_contains_string(v, needle)),
        serde_json::Value::Object(map) => map
            .iter()
            .any(|(k, v)| k.contains(needle) || json_contains_string(v, needle)),
        _ => false,
    }
}

/// Apply input-path translation to all string arguments that look like
/// file-system paths in an MCP tool argument map.
///
/// The arguments checked are: `file_path`, `path`, `root_dir`, `target_path`,
/// and `rootDir` (the camelCase variant some clients send). Any string value
/// whose key is in the set above is run through [`translate_input_path`].
///
/// Returns `Ok(translated_args)` on success, or the first
/// [`PathTranslationError`] encountered.
pub fn translate_input_args(
    mut args: serde_json::Map<String, serde_json::Value>,
    caller_root: &Path,
) -> Result<serde_json::Map<String, serde_json::Value>, PathTranslationError> {
    const PATH_KEYS: &[&str] = &["file_path", "path", "root_dir", "target_path", "rootDir"];

    for key in PATH_KEYS {
        if let Some(serde_json::Value::String(raw)) = args.get(*key) {
            let raw_clone = raw.clone();
            let translated = translate_input_path(&raw_clone, caller_root)?;
            args.insert(key.to_string(), serde_json::Value::String(translated));
        }
    }

    Ok(args)
}

/// Apply output-path translation to the text content of a [`rmcp::model::CallToolResult`].
///
/// All `TextContent` items in the result have their text run through
/// [`rewrite_paths_in_text`] so that repo-relative paths become absolute
/// paths anchored at `caller_root`.
///
/// In stdio mode (no daemon, single worktree) this is a near-no-op: the
/// text already contains `caller_root`-anchored paths because the server's
/// own `root_dir` matches `caller_root`, so the fast-path in
/// [`rewrite_paths_in_text`] returns the text unchanged.
pub fn translate_output_result(
    result: rmcp::model::CallToolResult,
    caller_root: &Path,
) -> rmcp::model::CallToolResult {
    use rmcp::model::{Annotated, RawContent};

    let new_content: Vec<Annotated<RawContent>> = result
        .content
        .into_iter()
        .map(|annotated| {
            let new_raw = match annotated.raw {
                RawContent::Text(text_content) => {
                    let rewritten = rewrite_paths_in_text(&text_content.text, caller_root);
                    RawContent::Text(rmcp::model::RawTextContent {
                        text: rewritten,
                        meta: text_content.meta,
                    })
                }
                other => other,
            };
            Annotated {
                raw: new_raw,
                annotations: annotated.annotations,
            }
        })
        .collect();

    // Reconstruct the CallToolResult preserving is_error / meta.
    // Use the `success` / `error` constructor + `with_meta` since
    // CallToolResult is `#[non_exhaustive]`.
    let base = if result.is_error == Some(true) {
        rmcp::model::CallToolResult::error(new_content)
    } else {
        rmcp::model::CallToolResult::success(new_content)
    };
    base.with_meta(result.meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // ── translate_input_path ──────────────────────────────────────────────────

    #[test]
    fn absolute_path_inside_root_strips_prefix() {
        let root = Path::new("/home/user/repo");
        let result = translate_input_path("/home/user/repo/src/foo.rs", root).unwrap();
        assert_eq!(result, "src/foo.rs");
    }

    #[test]
    fn absolute_path_at_root_itself_strips_to_empty() {
        let root = Path::new("/home/user/repo");
        let result = translate_input_path("/home/user/repo", root).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn absolute_path_outside_root_returns_out_of_tree_error() {
        let root = Path::new("/home/user/repo-a");
        let err = translate_input_path("/home/user/repo-b/src/foo.rs", root).unwrap_err();
        assert!(matches!(err, PathTranslationError::OutOfTree { .. }));
        let msg = err.to_string();
        assert!(msg.contains("repo-b"));
        assert!(msg.contains("repo-a"));
    }

    #[test]
    fn relative_path_passes_through_unchanged() {
        let root = Path::new("/home/user/repo");
        let result = translate_input_path("src/foo.rs", root).unwrap();
        assert_eq!(result, "src/foo.rs");
    }

    #[test]
    fn deep_nested_absolute_path_strips_correctly() {
        let root = Path::new("/workspace/primary");
        let result = translate_input_path("/workspace/primary/apps/api/src/main.rs", root).unwrap();
        assert_eq!(result, "apps/api/src/main.rs");
    }

    // ── translate_output_path ─────────────────────────────────────────────────

    #[test]
    fn relative_output_path_gets_prefix() {
        let root = Path::new("/home/user/repo");
        assert_eq!(
            translate_output_path("src/foo.rs", root),
            "/home/user/repo/src/foo.rs"
        );
    }

    #[test]
    fn absolute_output_path_passes_through_unchanged() {
        let root = Path::new("/home/user/repo");
        assert_eq!(
            translate_output_path("/some/other/path/foo.rs", root),
            "/some/other/path/foo.rs"
        );
    }

    // ── rewrite_paths_in_text ─────────────────────────────────────────────────

    #[test]
    fn rewrites_relative_path_in_plain_text() {
        let root = Path::new("/workspace/wt-a");
        let text = "1. src/auth.rs (90% total)";
        let result = rewrite_paths_in_text(text, root);
        assert!(
            result.contains("/workspace/wt-a/src/auth.rs"),
            "got: {result}"
        );
    }

    #[test]
    fn does_not_rewrite_when_caller_root_already_present() {
        let root = Path::new("/workspace/wt-a");
        let text = "Found /workspace/wt-a/src/auth.rs";
        // Fast path: root already present → no change.
        let result = rewrite_paths_in_text(text, root);
        assert_eq!(result, text);
    }

    #[test]
    fn does_not_rewrite_bare_words_without_slash() {
        let root = Path::new("/workspace/wt-a");
        let text = "query result: foo bar baz";
        let result = rewrite_paths_in_text(text, root);
        // No path separators → fast-path return, text unchanged.
        assert_eq!(result, text);
    }

    #[test]
    fn does_not_rewrite_url_scheme() {
        let root = Path::new("/workspace/wt-a");
        let text = "see https://example.com/foo/bar for details";
        let result = rewrite_paths_in_text(text, root);
        // URLs must not be rewritten.
        assert!(
            result.contains("https://example.com/foo/bar"),
            "got: {result}"
        );
        assert!(!result.contains("/workspace/wt-a/https:"), "got: {result}");
    }

    #[test]
    fn rewrites_multiple_paths_in_text() {
        let root = Path::new("/workspace/wt-b");
        let text = "1. src/auth.rs\n2. src/db/client.rs";
        let result = rewrite_paths_in_text(text, root);
        assert!(
            result.contains("/workspace/wt-b/src/auth.rs"),
            "got: {result}"
        );
        assert!(
            result.contains("/workspace/wt-b/src/db/client.rs"),
            "got: {result}"
        );
    }

    // ── json_contains_string ──────────────────────────────────────────────────

    #[test]
    fn json_contains_string_finds_in_nested_value() {
        let v: serde_json::Value = serde_json::json!({
            "content": [{"type": "text", "text": "/worktree-a/src/foo.rs (90%)"}]
        });
        assert!(json_contains_string(&v, "/worktree-a/"));
        assert!(!json_contains_string(&v, "/worktree-b/"));
    }

    #[test]
    fn json_contains_string_finds_in_array() {
        let v: serde_json::Value = serde_json::json!(["one", "two /wt-a/src/main.rs three"]);
        assert!(json_contains_string(&v, "/wt-a/"));
    }

    #[test]
    fn json_contains_string_absent_returns_false() {
        let v: serde_json::Value = serde_json::json!({"text": "no path here"});
        assert!(!json_contains_string(&v, "/wt-a/"));
    }

    // ── translate_input_args ──────────────────────────────────────────────────

    #[test]
    fn translate_input_args_rewrites_file_path_key() {
        let root = Path::new("/workspace/wt-a");
        let mut args = serde_json::Map::new();
        args.insert(
            "file_path".to_string(),
            serde_json::Value::String("/workspace/wt-a/src/main.rs".to_string()),
        );
        let translated = translate_input_args(args, root).unwrap();
        assert_eq!(
            translated.get("file_path").and_then(|v| v.as_str()),
            Some("src/main.rs")
        );
    }

    #[test]
    fn translate_input_args_rejects_out_of_tree_path() {
        let root = Path::new("/workspace/wt-a");
        let mut args = serde_json::Map::new();
        args.insert(
            "file_path".to_string(),
            serde_json::Value::String("/workspace/wt-b/src/main.rs".to_string()),
        );
        let err = translate_input_args(args, root).unwrap_err();
        assert!(matches!(err, PathTranslationError::OutOfTree { .. }));
    }

    #[test]
    fn translate_input_args_leaves_non_path_keys_unchanged() {
        let root = Path::new("/workspace/wt-a");
        let mut args = serde_json::Map::new();
        args.insert(
            "query".to_string(),
            serde_json::Value::String("find auth handlers".to_string()),
        );
        args.insert("top_k".to_string(), serde_json::Value::Number(10.into()));
        let translated = translate_input_args(args, root).unwrap();
        assert_eq!(
            translated.get("query").and_then(|v| v.as_str()),
            Some("find auth handlers")
        );
        assert_eq!(translated.get("top_k").and_then(|v| v.as_u64()), Some(10));
    }

    // ── leakage invariant tests (multi-worktree scenarios) ────────────────────

    /// Happy path: session for worktree A gets output paths rooted at A's dir.
    #[test]
    fn output_paths_are_rooted_at_caller_worktree() {
        let worktree_a = Path::new("/workspace/wt-a");
        // Simulate tool output with a repo-relative path.
        let tool_output = "1. src/auth.rs (95% total)\n   Snippet: fn verify_token";
        let rewritten = rewrite_paths_in_text(tool_output, worktree_a);
        assert!(
            rewritten.contains("/workspace/wt-a/src/auth.rs"),
            "expected absolute path rooted at wt-a, got: {rewritten}"
        );
    }

    /// Happy path: session for worktree B gets the same content prefixed with B's root.
    #[test]
    fn different_sessions_get_different_absolute_roots() {
        let tool_output = "1. src/auth.rs (95% total)";
        let rewritten_a = rewrite_paths_in_text(tool_output, Path::new("/workspace/wt-a"));
        let rewritten_b = rewrite_paths_in_text(tool_output, Path::new("/workspace/wt-b"));
        assert!(rewritten_a.contains("/workspace/wt-a/"), "A: {rewritten_a}");
        assert!(rewritten_b.contains("/workspace/wt-b/"), "B: {rewritten_b}");
        assert!(
            !rewritten_b.contains("/workspace/wt-a/"),
            "B must not contain A's root: {rewritten_b}"
        );
    }

    /// THE LEAKAGE TEST: verify that B's result JSON never contains A's worktree path string.
    #[test]
    fn leakage_test_b_result_never_contains_a_worktree_path() {
        let wt_a = Path::new("/workspace/worktree-a");
        let wt_b = Path::new("/workspace/worktree-b");

        // Simulate a shared chunk found by both sessions — the raw tool output
        // uses repo-relative paths (as tool implementations produce).
        let shared_chunk_output = "1. src/shared/utils.rs (88% total)\n   Snippet: helper fn";

        // Session A's result: rewrite for A's worktree.
        let result_for_a = rewrite_paths_in_text(shared_chunk_output, wt_a);
        // Session B's result: rewrite for B's worktree.
        let result_for_b = rewrite_paths_in_text(shared_chunk_output, wt_b);

        // A's result must contain A's absolute path.
        assert!(result_for_a.contains("/workspace/worktree-a/src/shared/utils.rs"));

        // B's result must contain B's absolute path.
        assert!(result_for_b.contains("/workspace/worktree-b/src/shared/utils.rs"));

        // *** The leakage invariant ***: B's serialized JSON must not contain
        // any string from A's worktree path namespace.
        let b_json = serde_json::json!({
            "content": [{"type": "text", "text": result_for_b}]
        });
        assert!(
            !json_contains_string(&b_json, "/workspace/worktree-a"),
            "LEAKAGE: B's result contains A's worktree path!\n\
             B result JSON: {b_json}"
        );
    }

    /// Edge case: out-of-tree input path is rejected cleanly.
    #[test]
    fn out_of_tree_input_path_produces_clear_error() {
        let root = Path::new("/workspace/wt-a");
        let err =
            translate_input_path("/workspace/wt-b/secrets/credentials.json", root).unwrap_err();
        let msg = err.to_string();
        // Must mention the rejected path.
        assert!(msg.contains("wt-b"), "msg: {msg}");
        // Must mention the worktree root.
        assert!(msg.contains("wt-a"), "msg: {msg}");
        // Must not panic.
    }

    /// Edge case: output path learned from parent ref is rewritten to caller's
    /// worktree prefix (simulated by calling the rewrite function).
    #[test]
    fn output_path_from_parent_ref_rewritten_to_caller_root() {
        // The "parent ref" is the primary worktree at wt-primary.
        // A caller is a secondary worktree at wt-secondary.
        // The tool output contains a repo-relative path (canonical form).
        let primary_relative_output = "1. src/core/engine.rs (92% total)";
        let caller_root = Path::new("/workspace/wt-secondary");
        let rewritten = rewrite_paths_in_text(primary_relative_output, caller_root);
        assert!(
            rewritten.contains("/workspace/wt-secondary/src/core/engine.rs"),
            "got: {rewritten}"
        );
        // Must NOT contain the primary worktree path.
        assert!(
            !rewritten.contains("/workspace/wt-primary"),
            "leaked primary path: {rewritten}"
        );
    }

    /// Edge case: stdio mode — when caller_root matches the server's own root_dir,
    /// the translation is a no-op (text already contains the absolute paths).
    #[test]
    fn stdio_mode_translation_is_noop_when_root_present() {
        let root = Path::new("/workspace/primary");
        // In stdio mode the tool output already contains absolute paths
        // rooted at root — the fast-path in rewrite_paths_in_text returns unchanged.
        let text = "1. /workspace/primary/src/main.rs (99% total)";
        let result = rewrite_paths_in_text(text, root);
        assert_eq!(
            result, text,
            "stdio mode must not alter already-absolute paths"
        );
    }
}
