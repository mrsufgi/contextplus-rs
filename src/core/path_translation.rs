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
/// use contextplus_rs::core::path_translation::translate_input_path;
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

/// Rewrite occurrences of *foreign* worktree absolute prefixes inside a
/// free-form text string, replacing them with `caller_root`.
///
/// The protection target: when daemon serves multiple refs and a tool result
/// is built from a CACHE entry that was authored under another ref's
/// worktree path, the response text might contain that other worktree's
/// absolute prefix. Returning that to a different session would leak
/// implementation detail (and on Linux, possibly file paths the caller has
/// no access to). This function rewrites those foreign prefixes to the
/// caller's own root.
///
/// **Single-ref no-op.** Pass an empty `foreign_roots` slice to skip text
/// rewriting entirely. Today's daemon serves one ref; there are no foreign
/// prefixes to translate. When real multi-ref dispatch lands the daemon
/// passes the list of OTHER attached refs' canonical worktree paths.
///
/// **Why not regex-rewrite repo-relative paths?** The previous design
/// scanned for any token matching `[a-zA-Z0-9_./-]+` containing `/` and
/// prefixed it with `caller_root`. That mangles:
///
/// 1. Tool labels that happen to look like paths (e.g. `semantic_navigate`
///    cluster headers like `prior-auth-ai/app` or `[/workspace/billing]`
///    auto-generated descriptive tokens).
/// 2. Scope-relative paths from tools called with a sub-scope `rootDir` —
///    the binary returns paths relative to the scope, not to the workspace
///    root, so prefixing with `caller_root` produces the wrong absolute
///    path (e.g. `/workspace/billing/...` instead of
///    `/workspace/packages/domains/billing/...`).
///
/// Both classes of bug were observed in the live MCP. The fix: only rewrite
/// strings that start with a known foreign-worktree absolute prefix, which
/// are the only strings whose contents are unambiguously foreign-worktree
/// paths. Everything else is left untouched.
///
/// ## Edge case: caller_root in foreign_roots
///
/// `foreign_roots` should NEVER contain `caller_root` itself; the caller is
/// responsible for filtering. If it does, this function safely no-ops on
/// those entries (it would be replacing `caller_root` with `caller_root`).
pub fn rewrite_paths_in_text(text: &str, caller_root: &Path, foreign_roots: &[&Path]) -> String {
    // Fast path: empty list means single-ref or no cross-ref leakage risk.
    if foreign_roots.is_empty() {
        return text.to_string();
    }
    // Fast path: nothing that looks like a path separator.
    if !text.contains('/') {
        return text.to_string();
    }

    let caller_root_str = caller_root.to_string_lossy();
    let mut output = text.to_string();
    for foreign in foreign_roots {
        let foreign_str = foreign.to_string_lossy();
        if foreign_str == caller_root_str {
            // Caller passed its own root in the foreign list — defensively skip.
            continue;
        }
        if !output.contains(foreign_str.as_ref()) {
            continue;
        }
        // Replace every occurrence. Substring replace is safe here because
        // a canonical absolute prefix is unique enough to not collide with
        // legitimate text — paths that *don't* belong to a worktree won't
        // start with a worktree's canonical absolute path.
        output = output.replace(foreign_str.as_ref(), &caller_root_str);
    }
    output
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
    args: serde_json::Map<String, serde_json::Value>,
    caller_root: &Path,
) -> Result<serde_json::Map<String, serde_json::Value>, PathTranslationError> {
    translate_input_args_with_allowed_roots(args, caller_root, &[])
}

/// Like [`translate_input_args`] but also accepts absolute paths that fall
/// under any of `allowed_extra_roots` — those paths are passed through
/// **unchanged** (not rewritten relative).
///
/// The intent: some tools route to OTHER registered worktrees (e.g.
/// `run_static_analysis` with an absolute `target_path` under a foreign ref).
/// Those calls are not a leakage risk because:
///
/// 1. The path is opt-in — the caller supplied it.
/// 2. The destination ref is one the daemon already has in its registry
///    (registered explicitly via `attach_worktree`, or via a per-worktree
///    session handshake).
///
/// Translation strips the caller_root prefix, which would corrupt foreign
/// absolute paths; instead we pass them through and let the tool handler do
/// the registry-aware routing.
pub fn translate_input_args_with_allowed_roots(
    mut args: serde_json::Map<String, serde_json::Value>,
    caller_root: &Path,
    allowed_extra_roots: &[&Path],
) -> Result<serde_json::Map<String, serde_json::Value>, PathTranslationError> {
    const PATH_KEYS: &[&str] = &["file_path", "path", "root_dir", "target_path", "rootDir"];

    for key in PATH_KEYS {
        if let Some(serde_json::Value::String(raw)) = args.get(*key) {
            let raw_clone = raw.clone();
            // Try caller_root first (preserves the relative-rewrite behavior).
            match translate_input_path(&raw_clone, caller_root) {
                Ok(translated) => {
                    args.insert(key.to_string(), serde_json::Value::String(translated));
                }
                Err(e) => {
                    // Out of caller's tree — accept only if the absolute path
                    // lives under a registered foreign ref. Pass through verbatim.
                    let p = Path::new(&raw_clone);
                    let allowed =
                        p.is_absolute() && allowed_extra_roots.iter().any(|r| p.starts_with(r));
                    if !allowed {
                        return Err(e);
                    }
                    // Pass through unchanged.
                }
            }
        }
    }

    Ok(args)
}

/// Apply output-path translation to the text content of a [`rmcp::model::CallToolResult`].
///
/// `foreign_roots` is the list of OTHER attached refs' canonical worktree
/// paths whose absolute prefix may appear in cached tool output and would
/// leak across the session boundary if returned verbatim. In single-ref
/// mode pass `&[]` — the function becomes identity, preserving every
/// tool's native output formatting.
///
/// **Why output translation is identity in single-ref mode.** Today the
/// daemon serves one ref. All paths in tool output were authored against
/// that single ref's `root_dir`, which equals `caller_root`. There is no
/// foreign prefix to rewrite; relative paths stay relative; absolute
/// paths with the only-known root are already correct.
pub fn translate_output_result(
    result: rmcp::model::CallToolResult,
    caller_root: &Path,
    foreign_roots: &[&Path],
) -> rmcp::model::CallToolResult {
    use rmcp::model::{Annotated, RawContent};

    let new_content: Vec<Annotated<RawContent>> = result
        .content
        .into_iter()
        .map(|annotated| {
            let new_raw = match annotated.raw {
                RawContent::Text(text_content) => {
                    let rewritten =
                        rewrite_paths_in_text(&text_content.text, caller_root, foreign_roots);
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
    fn empty_foreign_roots_is_identity() {
        let root = Path::new("/workspace/wt-a");
        let text = "1. src/auth.rs (90% total)\n[plugins + observability + config]";
        let result = rewrite_paths_in_text(text, root, &[]);
        assert_eq!(result, text, "single-ref / empty foreign_roots → identity");
    }

    #[test]
    fn rewrites_when_foreign_root_present() {
        let caller = Path::new("/workspace/wt-b");
        let foreign = Path::new("/workspace/wt-a");
        let text = "1. /workspace/wt-a/src/auth.rs (90% total)";
        let result = rewrite_paths_in_text(text, caller, &[foreign]);
        assert!(
            result.contains("/workspace/wt-b/src/auth.rs"),
            "got: {result}"
        );
        assert!(
            !result.contains("/workspace/wt-a"),
            "foreign prefix must be gone: {result}"
        );
    }

    #[test]
    fn does_not_rewrite_when_foreign_not_present() {
        let caller = Path::new("/workspace/wt-b");
        let foreign = Path::new("/workspace/wt-a");
        let text = "1. src/auth.rs (95% total)\n2. /workspace/wt-c/src/x.rs";
        let result = rewrite_paths_in_text(text, caller, &[foreign]);
        assert_eq!(
            result, text,
            "no foreign prefix in text → no rewrite, even if other absolutes exist"
        );
    }

    #[test]
    fn defensive_skips_caller_root_in_foreign_list() {
        let caller = Path::new("/workspace/wt-b");
        let text = "1. /workspace/wt-b/src/auth.rs";
        // Caller passed its own root in foreign list — must no-op safely.
        let result = rewrite_paths_in_text(text, caller, &[caller]);
        assert_eq!(result, text);
    }

    #[test]
    fn does_not_rewrite_url_scheme_when_foreign_unrelated() {
        let caller = Path::new("/workspace/wt-a");
        let foreign = Path::new("/workspace/wt-b");
        let text = "see https://example.com/foo/bar for details";
        let result = rewrite_paths_in_text(text, caller, &[foreign]);
        // No foreign prefix in URL → no rewrite.
        assert_eq!(result, text);
    }

    #[test]
    fn rewrites_multiple_occurrences_of_one_foreign() {
        let caller = Path::new("/workspace/wt-b");
        let foreign = Path::new("/workspace/wt-a");
        let text = "1. /workspace/wt-a/src/auth.rs\n2. /workspace/wt-a/src/db/client.rs";
        let result = rewrite_paths_in_text(text, caller, &[foreign]);
        assert!(result.contains("/workspace/wt-b/src/auth.rs"), "{result}");
        assert!(
            result.contains("/workspace/wt-b/src/db/client.rs"),
            "{result}"
        );
        assert!(!result.contains("wt-a"), "{result}");
    }

    #[test]
    fn rewrites_each_of_multiple_foreigns() {
        let caller = Path::new("/workspace/wt-c");
        let foreigns: &[&Path] = &[Path::new("/workspace/wt-a"), Path::new("/workspace/wt-b")];
        let text = "1. /workspace/wt-a/x.rs\n2. /workspace/wt-b/y.rs\n3. local/z.rs";
        let result = rewrite_paths_in_text(text, caller, foreigns);
        assert!(result.contains("/workspace/wt-c/x.rs"), "{result}");
        assert!(result.contains("/workspace/wt-c/y.rs"), "{result}");
        assert!(
            result.contains("local/z.rs"),
            "relative untouched: {result}"
        );
    }

    #[test]
    fn does_not_mangle_navigate_cluster_labels() {
        // Cluster labels (descriptive tokens shaped like paths) and
        // scope-relative file paths must NOT be mangled in single-ref mode.
        let caller = Path::new("/workspace");
        let navigate_text = "Semantic Navigator: 25 files\n  \
                             [plugins + observability + config] (10 files)\n    \
                             config/config.test.ts\n    \
                             plugins/health.ts";
        let result = rewrite_paths_in_text(navigate_text, caller, &[]);
        assert_eq!(result, navigate_text);
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

    /// THE LEAKAGE TEST: when ref A's absolute prefix appears in cached
    /// output served to ref B's session, B's response must have it
    /// rewritten to B's caller_root.
    #[test]
    fn leakage_test_b_result_never_contains_a_worktree_path() {
        let wt_a = Path::new("/workspace/worktree-a");
        let wt_b = Path::new("/workspace/worktree-b");

        // Simulate a cached chunk authored under A's worktree (absolute path
        // contains A's prefix, as cached SearchIndex entries do).
        let cached_output = "1. /workspace/worktree-a/src/shared/utils.rs (88% total)";

        // B's session: dispatch passes A as a foreign root because A is
        // currently attached.
        let result_for_b = rewrite_paths_in_text(cached_output, wt_b, &[wt_a]);

        // B's result must contain B's absolute path, not A's.
        assert!(result_for_b.contains("/workspace/worktree-b/src/shared/utils.rs"));

        let b_json = serde_json::json!({
            "content": [{"type": "text", "text": result_for_b}]
        });
        assert!(
            !json_contains_string(&b_json, "/workspace/worktree-a"),
            "LEAKAGE: B's result contains A's worktree path!\n{b_json}"
        );
    }

    /// Edge case: out-of-tree input path is rejected cleanly.
    #[test]
    fn out_of_tree_input_path_produces_clear_error() {
        let root = Path::new("/workspace/wt-a");
        let err =
            translate_input_path("/workspace/wt-b/secrets/credentials.json", root).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("wt-b"), "msg: {msg}");
        assert!(msg.contains("wt-a"), "msg: {msg}");
    }

    /// In single-ref / stdio mode caller_root matches server root_dir;
    /// `foreign_roots` is empty and the function is identity.
    #[test]
    fn single_ref_translation_is_identity_for_absolute_paths() {
        let root = Path::new("/workspace/primary");
        let text = "1. /workspace/primary/src/main.rs (99% total)";
        let result = rewrite_paths_in_text(text, root, &[]);
        assert_eq!(result, text);
    }
}
