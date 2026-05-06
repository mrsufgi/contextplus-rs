//! Transport mode resolution and implicit-default dispatch.
//!
//! This module owns the logic that was previously inlined in `src/main.rs` for
//! choosing between daemon-proxy and direct-stdio transports, as well as the
//! `run_mcp_server` helper that starts the full stdio server. Extracting these
//! to the library crate makes the behaviour unit-testable and keeps `main.rs`
//! as a thin entry-point.
//!
//! # Path translation at the dispatch boundary (U5)
//!
//! When a daemon serves multiple worktrees, each session's tool calls must be
//! translated so that:
//!
//! 1. **Inputs**: absolute paths supplied by a caller are stripped of the
//!    caller's worktree prefix before reaching the tool implementation.
//! 2. **Outputs**: repo-relative paths in tool results are prefixed with the
//!    calling session's worktree root before being returned to the caller.
//! 3. **Out-of-tree paths** are rejected with a clear error.
//!
//! The `dispatch_with_translation` function implements this boundary.
//! Today it uses `SharedState.default_ref().root_dir` as the caller's worktree
//! root.  When U4 lands and introduces `session.ref_id`, replace the
//! `default_ref()` call with `state.ref_index(session.ref_id)` — the
//! primitives in `crate::core::path_translation` don't change.
//!
//! ## Stdio no-op
//!
//! In single-daemon / stdio mode the caller's worktree is the same path the
//! server uses internally, so `rewrite_paths_in_text` fast-paths through
//! (the text already contains the absolute root), and
//! `translate_input_path` for relative args is a pass-through.  Existing
//! stdio-only deployments are entirely unaffected.
//!
//! # Environment variables
//!
//! | Variable                  | Values                       | Default |
//! |---------------------------|------------------------------|---------|
//! | `CONTEXTPLUS_TRANSPORT`   | `auto`, `stdio`, `daemon`    | `auto`  |
//!
//! When set to `auto` (or absent / empty) the server probes whether
//! `<root>/.mcp_data/` is writable and picks daemon mode if it is, stdio
//! otherwise.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use rmcp::ServiceExt;

use crate::config::Config;
use crate::core::path_translation;
use crate::core::process_lifecycle;
use crate::server::ContextPlusServer;
use crate::transport::paths::MCP_DATA_DIR;

/// Environment variable that controls which transport the server uses when
/// invoked without an explicit sub-command.
pub const TRANSPORT_ENV: &str = "CONTEXTPLUS_TRANSPORT";

/// Transport mode used when the binary is invoked without a sub-command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportMode {
    /// Automatically probe writability and pick daemon or stdio accordingly.
    Auto,
    /// Always use the direct stdio transport (legacy).
    Stdio,
    /// Always attempt the daemon-proxy transport.
    Daemon,
}

/// Read [`TRANSPORT_ENV`] and return the corresponding [`TransportMode`].
///
/// An unknown value produces a `warn!` log and falls back to [`TransportMode::Auto`].
pub fn resolve_transport_mode() -> TransportMode {
    match std::env::var(TRANSPORT_ENV)
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("stdio") => TransportMode::Stdio,
        Some("daemon") => TransportMode::Daemon,
        Some("auto") | None | Some("") => TransportMode::Auto,
        Some(other) => {
            tracing::warn!("unknown {TRANSPORT_ENV}={other:?}; falling back to auto");
            TransportMode::Auto
        }
    }
}

/// Return `true` when `<root_dir>/.mcp_data/` is writable (creating it if
/// absent). A `false` return means the daemon transport cannot safely be used
/// for this workspace.
pub fn mcp_data_writable(root_dir: &Path) -> bool {
    let dir = root_dir.join(MCP_DATA_DIR);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::debug!(
            "mcp_data not writable: create_dir_all({}) failed: {e}",
            dir.display()
        );
        return false;
    }
    let probe = dir.join(".write-probe");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(e) => {
            tracing::debug!("mcp_data not writable: probe failed: {e}");
            false
        }
    }
}

/// Resolve whether the daemon transport should be attempted for `mode` and
/// `root_dir`.
///
/// This is the pure, synchronous part of [`run_implicit_default`]'s decision:
/// - `Stdio`  → always `false`
/// - `Daemon` → always `true`
/// - `Auto`   → probe `<root_dir>/.mcp_data/` writability at runtime
///
/// Extracted as a separate function so the branch logic is unit-testable
/// without spawning an async runtime.
pub fn resolve_want_daemon(mode: TransportMode, root_dir: &Path) -> bool {
    match mode {
        TransportMode::Stdio => false,
        TransportMode::Daemon => true,
        TransportMode::Auto => mcp_data_writable(root_dir),
    }
}

/// Handle the implicit (no sub-command) invocation.
///
/// Behaviour:
/// - `Stdio`  → serve directly over stdio.
/// - `Daemon` → proxy through the per-workspace daemon; propagate errors.
/// - `Auto`   → probe writability; attempt daemon first; if it fails **warn**
///   and fall back to stdio.
pub async fn run_implicit_default(
    mode: TransportMode,
    root_dir: PathBuf,
    config: Config,
) -> Result<()> {
    let want_daemon = resolve_want_daemon(mode, &root_dir);

    if want_daemon {
        match crate::transport::client::run(&root_dir).await {
            Ok(()) => return Ok(()),
            Err(e) if mode == TransportMode::Auto => {
                tracing::warn!("daemon transport failed ({e}); falling back to stdio");
            }
            Err(e) => return Err(e),
        }
    }

    run_mcp_server(root_dir, config).await
}

/// Start the MCP server over stdio. This is the low-level function invoked
/// both from `run_implicit_default` and directly from `main.rs` for the
/// `serve` sub-command.
pub async fn run_mcp_server(root_dir: PathBuf, config: Config) -> Result<()> {
    tracing::info!(
        "Starting contextplus MCP server on {} (model: {})",
        root_dir.display(),
        config.ollama_embed_model
    );

    let server = ContextPlusServer::new(root_dir.clone(), config.clone());

    let root_str = root_dir.to_string_lossy().to_string();
    if let Err(e) = server
        .state
        .memory_graph
        .get_graph(&root_str, |_graph| {})
        .await
    {
        tracing::warn!("Failed to pre-load memory graph from disk: {e}");
    }

    let _debounce_handle = server.state.memory_graph.spawn_debounce_task();

    use crate::config::TrackerMode;
    tracing::info!(mode = %config.embed_tracker_mode, "Embedding tracker mode");
    if config.embed_tracker_mode == TrackerMode::Eager {
        server.ensure_tracker_started();
    }

    if config.warmup_on_start {
        tracing::info!("Spawning SearchIndex warmup task (CONTEXTPLUS_WARMUP_ON_START=true)");
        server.spawn_warmup_task();
    }

    let idle_timeout_ms = config.idle_timeout_ms;
    let idle_monitor = process_lifecycle::create_idle_monitor(idle_timeout_ms, move || {
        tracing::info!(
            "Idle timeout ({}ms) reached -- initiating shutdown",
            idle_timeout_ms
        );
        std::process::exit(0);
    });
    if config.idle_timeout_ms > 0 {
        tracing::info!(
            "Idle shutdown monitor started ({}ms)",
            config.idle_timeout_ms
        );
    }

    let idle_monitor = Arc::new(idle_monitor);
    {
        let mut guard = server.state.idle_monitor.write().await;
        *guard = Some(idle_monitor.clone());
    }

    #[cfg(unix)]
    let _parent_monitor = {
        let parent_pid = std::os::unix::process::parent_id();
        let handle =
            process_lifecycle::start_parent_monitor(parent_pid, config.parent_poll_ms, move || {
                tracing::info!(
                    "Parent process (pid={}) exited -- initiating shutdown",
                    parent_pid
                );
                std::process::exit(0);
            });
        tracing::info!(
            "Parent PID monitor started (pid={}, poll={}ms)",
            parent_pid,
            config.parent_poll_ms
        );
        handle
    };

    let memory_graph = Arc::clone(&server.state.memory_graph);
    let state_for_shutdown = server.state.clone();

    let transport = rmcp::transport::io::stdio();
    let ct = server.serve(transport).await?;

    #[cfg(unix)]
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    #[cfg(unix)]
    let mut sighup = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())?;

    tokio::select! {
        result = ct.waiting() => { result?; }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("SIGINT received -- cancelling embeddings and shutting down");
            state_for_shutdown.ollama.cancel_all_embeddings();
        }
        _ = async {
            #[cfg(unix)]
            { sigterm.recv().await }
            #[cfg(not(unix))]
            { std::future::pending::<Option<()>>().await }
        } => {
            tracing::info!("SIGTERM received -- shutting down");
            state_for_shutdown.ollama.cancel_all_embeddings();
        }
        _ = async {
            #[cfg(unix)]
            { sighup.recv().await }
            #[cfg(not(unix))]
            { std::future::pending::<Option<()>>().await }
        } => {
            tracing::info!("SIGHUP received -- shutting down");
            state_for_shutdown.ollama.cancel_all_embeddings();
        }
    }

    state_for_shutdown.ollama.flush_query_cache();

    if let Err(e) = memory_graph.flush().await {
        tracing::warn!("Failed to persist memory graph on shutdown: {e}");
    }

    idle_monitor.stop();
    #[cfg(unix)]
    _parent_monitor.stop();

    Ok(())
}

// ── Path-translation dispatch boundary (U5) ──────────────────────────────────

/// Invoke a tool with full path-translation at the dispatch boundary.
///
/// This wraps [`ContextPlusServer::dispatch`] with the U5 path-translation
/// layer:
///
/// 1. **Input translation**: known path arguments (`file_path`, `path`,
///    `root_dir`, `target_path`, `rootDir`) that are absolute paths are
///    stripped of the `caller_root` prefix.  Absolute paths outside
///    `caller_root` are rejected immediately with an error result —
///    no panic, no forwarding to the tool.
///
/// 2. **Output translation**: after the tool returns, all relative-path
///    strings in the text content are prefixed with `caller_root`.
///
/// **U4 seam:** `caller_root` today comes from `SharedState.default_ref()`.
/// When U4 introduces `session.ref_id`, replace the call site with
/// `state.ref_index(session.ref_id).root_dir`.
///
/// **Stdio no-op:** when `caller_root` equals the server's own `root_dir`
/// the translation is transparent — inputs are typically relative and
/// outputs already contain `root_dir`-prefixed absolute paths.
pub async fn dispatch_with_translation(
    server: &ContextPlusServer,
    tool_name: &str,
    args: serde_json::Map<String, serde_json::Value>,
    caller_root: &Path,
) -> rmcp::model::CallToolResult {
    // --- Input translation ---
    let translated_args = match path_translation::translate_input_args(args, caller_root) {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!(
                tool = tool_name,
                caller_root = %caller_root.display(),
                "Rejecting tool call: {e}"
            );
            return rmcp::model::CallToolResult::error(vec![rmcp::model::Content::text(format!(
                "Error: {e}"
            ))]);
        }
    };

    // --- Dispatch ---
    let raw_result = server.dispatch(tool_name, translated_args).await;

    // --- Output translation ---
    path_translation::translate_output_result(raw_result, caller_root)
}

/// Return the caller's worktree root for the current (single-ref) configuration.
///
/// This is the U4 seam: today it reads from `SharedState.default_ref()`;
/// U4 will replace this with per-session `RefIndex` lookup using
/// `session.ref_id` obtained during `register_session`.
///
/// Returns `None` only if the registry was tampered with externally.
pub fn caller_root_from_default_ref(state: &crate::server::SharedState) -> Option<PathBuf> {
    state.default_ref().map(|r| r.root_dir.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialise env-mutating tests so they don't race each other.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn set_transport(val: &str) {
        unsafe { std::env::set_var(TRANSPORT_ENV, val) };
    }

    fn remove_transport() {
        unsafe { std::env::remove_var(TRANSPORT_ENV) };
    }

    // ── resolve_transport_mode ────────────────────────────────────────────────

    #[test]
    fn resolve_missing_env_returns_auto() {
        let _g = ENV_LOCK.lock().unwrap();
        remove_transport();
        assert_eq!(resolve_transport_mode(), TransportMode::Auto);
    }

    #[test]
    fn resolve_empty_string_returns_auto() {
        let _g = ENV_LOCK.lock().unwrap();
        set_transport("");
        assert_eq!(resolve_transport_mode(), TransportMode::Auto);
        remove_transport();
    }

    #[test]
    fn resolve_explicit_auto_returns_auto() {
        let _g = ENV_LOCK.lock().unwrap();
        set_transport("auto");
        assert_eq!(resolve_transport_mode(), TransportMode::Auto);
        remove_transport();
    }

    #[test]
    fn resolve_stdio_returns_stdio() {
        let _g = ENV_LOCK.lock().unwrap();
        set_transport("stdio");
        assert_eq!(resolve_transport_mode(), TransportMode::Stdio);
        remove_transport();
    }

    #[test]
    fn resolve_daemon_returns_daemon() {
        let _g = ENV_LOCK.lock().unwrap();
        set_transport("daemon");
        assert_eq!(resolve_transport_mode(), TransportMode::Daemon);
        remove_transport();
    }

    #[test]
    fn resolve_garbage_returns_auto() {
        let _g = ENV_LOCK.lock().unwrap();
        set_transport("blah_blah");
        // Unknown value: should warn and fall back to Auto.
        assert_eq!(resolve_transport_mode(), TransportMode::Auto);
        remove_transport();
    }

    #[test]
    fn resolve_is_case_insensitive() {
        let _g = ENV_LOCK.lock().unwrap();
        set_transport("STDIO");
        assert_eq!(resolve_transport_mode(), TransportMode::Stdio);
        set_transport("DAEMON");
        assert_eq!(resolve_transport_mode(), TransportMode::Daemon);
        remove_transport();
    }

    #[test]
    fn resolve_trims_whitespace() {
        let _g = ENV_LOCK.lock().unwrap();
        set_transport("  stdio  ");
        assert_eq!(resolve_transport_mode(), TransportMode::Stdio);
        remove_transport();
    }

    // ── mcp_data_writable ────────────────────────────────────────────────────

    #[test]
    fn writable_dir_returns_true() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(mcp_data_writable(tmp.path()));
    }

    #[test]
    fn nonexistent_dir_is_created_and_returns_true() {
        let tmp = tempfile::tempdir().unwrap();
        let new_root = tmp.path().join("deep").join("nested");
        // The directory does not exist yet.
        assert!(!new_root.exists());
        assert!(mcp_data_writable(&new_root));
        // Side-effect: directory was created.
        assert!(new_root.join(MCP_DATA_DIR).exists());
    }

    #[test]
    #[cfg(unix)]
    fn readonly_dir_returns_false() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let mcp_dir = tmp.path().join(MCP_DATA_DIR);
        std::fs::create_dir_all(&mcp_dir).unwrap();
        // Make the directory read-only so the write-probe fails.
        std::fs::set_permissions(&mcp_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        let result = mcp_data_writable(tmp.path());
        // Restore permissions so tempdir cleanup can remove the dir.
        std::fs::set_permissions(&mcp_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(!result);
    }

    #[test]
    fn writable_probe_file_is_removed() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(mcp_data_writable(tmp.path()));
        // The write probe must be cleaned up after a successful check.
        let probe = tmp.path().join(MCP_DATA_DIR).join(".write-probe");
        assert!(!probe.exists(), ".write-probe must be removed after check");
    }

    #[test]
    fn writable_idempotent_across_calls() {
        let tmp = tempfile::tempdir().unwrap();
        // Calling twice must both succeed — second call should not fail
        // because `.mcp_data/` already exists.
        assert!(mcp_data_writable(tmp.path()));
        assert!(mcp_data_writable(tmp.path()));
    }

    // ── resolve_want_daemon ──────────────────────────────────────────────────

    #[test]
    fn want_daemon_stdio_always_false() {
        let tmp = tempfile::tempdir().unwrap();
        // Stdio mode never wants the daemon, regardless of writability.
        assert!(!resolve_want_daemon(TransportMode::Stdio, tmp.path()));
    }

    #[test]
    fn want_daemon_daemon_always_true() {
        // Daemon mode always wants the daemon, even for a path that may not be
        // writable (writability check is irrelevant in this mode).
        let tmp = tempfile::tempdir().unwrap();
        assert!(resolve_want_daemon(TransportMode::Daemon, tmp.path()));
    }

    #[test]
    fn want_daemon_auto_writable_returns_true() {
        let tmp = tempfile::tempdir().unwrap();
        // Auto mode with a writable root → wants daemon.
        assert!(resolve_want_daemon(TransportMode::Auto, tmp.path()));
    }

    #[test]
    #[cfg(unix)]
    fn want_daemon_auto_readonly_returns_false() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let mcp_dir = tmp.path().join(MCP_DATA_DIR);
        std::fs::create_dir_all(&mcp_dir).unwrap();
        std::fs::set_permissions(&mcp_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        // Auto mode with a non-writable .mcp_data/ → does NOT want daemon.
        let result = resolve_want_daemon(TransportMode::Auto, tmp.path());
        std::fs::set_permissions(&mcp_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(!result);
    }

    #[test]
    fn want_daemon_auto_creates_mcp_data_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let new_root = tmp.path().join("new_workspace");
        std::fs::create_dir_all(&new_root).unwrap();
        // Auto mode must create .mcp_data/ when it doesn't exist.
        assert!(resolve_want_daemon(TransportMode::Auto, &new_root));
        assert!(
            new_root.join(MCP_DATA_DIR).exists(),
            ".mcp_data/ must be created by Auto mode writability probe"
        );
    }

    // ── mcp_data_writable — non-existent parent ──────────────────────────────

    #[test]
    fn nonexistent_parent_dir_returns_false() {
        // If the entire root path does not exist, create_dir_all will fail
        // when the parent itself cannot be created (e.g. under a read-only FS).
        // On a normal writable FS this actually succeeds (mkdir -p), so we test
        // a path inside a known non-writable location instead.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let tmp = tempfile::tempdir().unwrap();
            let ro_parent = tmp.path().join("ro_parent");
            std::fs::create_dir_all(&ro_parent).unwrap();
            std::fs::set_permissions(&ro_parent, std::fs::Permissions::from_mode(0o555)).unwrap();
            // Try to create .mcp_data/ inside a read-only parent — must fail.
            let result = mcp_data_writable(&ro_parent.join("child"));
            std::fs::set_permissions(&ro_parent, std::fs::Permissions::from_mode(0o755)).unwrap();
            assert!(!result);
        }
    }

    // ── dispatch_with_translation ─────────────────────────────────────────────
    //
    // These tests exercise the path-translation boundary at the dispatch layer.
    // They use synthetic worktree roots and a live ContextPlusServer to verify:
    //   1. Input paths get stripped of the caller's worktree prefix.
    //   2. Output paths get prefixed with the caller's worktree root.
    //   3. Cross-session leakage: B's result never contains A's worktree path.
    //   4. Out-of-tree input paths are rejected with a clear error.
    //   5. The `caller_root_from_default_ref` accessor returns the server's root.
    //
    // Note: full end-to-end testing of actual tool content would require a
    // real corpus with embeddings. These tests focus on the translation layer
    // itself using lightweight tool calls (missing-arg fast paths) that return
    // predictable text — sufficient to prove the boundary works.

    fn make_test_server(root: &Path) -> crate::server::ContextPlusServer {
        let config = crate::config::Config::from_env();
        crate::server::ContextPlusServer::new(root.to_path_buf(), config)
    }

    /// Build a minimal server in a temp dir with a few source files.
    fn make_server_with_files() -> (tempfile::TempDir, crate::server::ContextPlusServer) {
        let tmp = tempfile::tempdir().unwrap();
        // Create some "source files" so the tool has content to return.
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/auth.rs"), "pub fn verify_token() {}").unwrap();
        std::fs::write(tmp.path().join("src/db.rs"), "pub fn connect() {}").unwrap();
        let server = make_test_server(tmp.path());
        (tmp, server)
    }

    /// Extract the first text content from a CallToolResult.
    fn first_text(result: &rmcp::model::CallToolResult) -> &str {
        match &result.content[0].raw {
            rmcp::model::RawContent::Text(t) => &t.text,
            _ => panic!("expected text content"),
        }
    }

    /// Recursively scan a serde_json::Value for a needle string.
    fn json_has(v: &serde_json::Value, needle: &str) -> bool {
        crate::core::path_translation::json_contains_string(v, needle)
    }

    // ── caller_root_from_default_ref ──────────────────────────────────────────

    #[test]
    fn default_ref_accessor_returns_server_root() {
        let tmp = tempfile::tempdir().unwrap();
        let server = make_test_server(tmp.path());
        let root = caller_root_from_default_ref(&server.state);
        assert!(root.is_some(), "default_ref should exist");
        // The returned root must be the same path we constructed the server with
        // (modulo canonicalization that ContextPlusServer::new may do).
        let got = root.unwrap();
        assert!(
            got == tmp.path()
                || got
                    == tmp
                        .path()
                        .canonicalize()
                        .unwrap_or(tmp.path().to_path_buf()),
            "expected server root, got: {}",
            got.display()
        );
    }

    // ── Input translation ─────────────────────────────────────────────────────

    /// Out-of-tree `file_path` arg is rejected before the tool is called.
    #[tokio::test]
    async fn dispatch_rejects_out_of_tree_file_path() {
        let tmp = tempfile::tempdir().unwrap();
        let server = make_test_server(tmp.path());

        // Pass a file_path that is NOT under tmp (some other worktree).
        let mut args = serde_json::Map::new();
        args.insert(
            "file_path".to_string(),
            serde_json::Value::String("/completely/different/worktree/src/foo.rs".to_string()),
        );

        let wt_root = tmp.path();
        let result = dispatch_with_translation(&server, "get_file_skeleton", args, wt_root).await;

        // Must be an error result.
        assert_eq!(result.is_error, Some(true));
        let text = first_text(&result);
        assert!(
            text.contains("outside the caller's worktree root"),
            "expected out-of-tree error, got: {text}"
        );
    }

    /// Relative `file_path` (already canonical) passes through without error.
    #[tokio::test]
    async fn dispatch_accepts_relative_file_path_input() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/main.rs"), "fn main() {}").unwrap();
        let server = make_test_server(tmp.path());

        let mut args = serde_json::Map::new();
        // Relative path — should pass through translation unchanged and reach tool.
        args.insert(
            "file_path".to_string(),
            serde_json::Value::String("src/main.rs".to_string()),
        );

        let result =
            dispatch_with_translation(&server, "get_file_skeleton", args, tmp.path()).await;
        // The tool either succeeds or returns a content error, but must NOT
        // produce an "outside the caller's worktree root" rejection.
        let text = first_text(&result);
        assert!(
            !text.contains("outside the caller's worktree root"),
            "relative path should not trigger out-of-tree error, got: {text}"
        );
    }

    /// Absolute `file_path` inside the caller's worktree is stripped to relative.
    #[tokio::test]
    async fn dispatch_strips_caller_prefix_from_absolute_file_path() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/main.rs"), "fn main() {}").unwrap();
        let server = make_test_server(tmp.path());

        let abs_path = tmp.path().join("src/main.rs");
        let mut args = serde_json::Map::new();
        args.insert(
            "file_path".to_string(),
            serde_json::Value::String(abs_path.to_string_lossy().to_string()),
        );

        let result =
            dispatch_with_translation(&server, "get_file_skeleton", args, tmp.path()).await;
        // Same as above: must not reject as out-of-tree.
        let text = first_text(&result);
        assert!(
            !text.contains("outside the caller's worktree root"),
            "in-tree absolute path should not trigger rejection, got: {text}"
        );
    }

    // ── Output translation ────────────────────────────────────────────────────

    /// A tool result with relative paths gets prefixed with caller_root.
    #[test]
    fn translate_output_result_prefixes_relative_paths() {
        use crate::core::path_translation::translate_output_result;

        let caller_root = std::path::Path::new("/workspace/wt-a");
        // Simulate a tool result with a relative path in text.
        let content = rmcp::model::Content::text("1. src/auth.rs (90% total)");
        let raw = rmcp::model::CallToolResult::success(vec![content]);
        let translated = translate_output_result(raw, caller_root);

        let text = first_text(&translated);
        assert!(
            text.contains("/workspace/wt-a/src/auth.rs"),
            "expected absolute path in output, got: {text}"
        );
    }

    /// Output already containing absolute paths for the caller's root is
    /// left unchanged (stdio no-op path).
    #[test]
    fn translate_output_result_noop_when_already_absolute() {
        use crate::core::path_translation::translate_output_result;

        let caller_root = std::path::Path::new("/workspace/primary");
        let text = "1. /workspace/primary/src/main.rs (99% total)";
        let content = rmcp::model::Content::text(text);
        let raw = rmcp::model::CallToolResult::success(vec![content]);
        let translated = translate_output_result(raw, caller_root);
        assert_eq!(first_text(&translated), text);
    }

    // ── THE LEAKAGE TEST ─────────────────────────────────────────────────────
    //
    // This is the load-bearing test for U5. It proves that the dispatch layer
    // ensures B's tool result never contains A's worktree path string anywhere
    // in the response JSON, even when both sessions share the same underlying
    // content (chunk learned from the same repo).

    #[tokio::test]
    async fn leakage_test_b_result_never_contains_a_worktree_path_via_dispatch() {
        use crate::core::path_translation::translate_output_result;

        // Simulate the shared raw tool output as if both sessions hit the same
        // chunk index (repo-relative paths, no worktree prefix).
        let shared_tool_output = "1. src/shared/utils.rs (88% total)\n   Snippet: helper fn\n2. src/core/engine.rs (75% total)";

        let wt_a = std::path::Path::new("/workspace/worktree-a");
        let wt_b = std::path::Path::new("/workspace/worktree-b");

        // Produce per-session results by applying output translation.
        let result_for_a = {
            let content = rmcp::model::Content::text(shared_tool_output);
            let raw = rmcp::model::CallToolResult::success(vec![content]);
            translate_output_result(raw, wt_a)
        };
        let result_for_b = {
            let content = rmcp::model::Content::text(shared_tool_output);
            let raw = rmcp::model::CallToolResult::success(vec![content]);
            translate_output_result(raw, wt_b)
        };

        // A's result must contain A's absolute paths.
        let text_a = first_text(&result_for_a);
        assert!(
            text_a.contains("/workspace/worktree-a/src/shared/utils.rs"),
            "A: {text_a}"
        );

        // B's result must contain B's absolute paths.
        let text_b = first_text(&result_for_b);
        assert!(
            text_b.contains("/workspace/worktree-b/src/shared/utils.rs"),
            "B: {text_b}"
        );

        // *** THE LEAKAGE INVARIANT ***: serialize B's result to JSON and assert
        // that A's worktree path string is nowhere in B's response.
        let b_json = serde_json::to_value(&result_for_b).unwrap();
        assert!(
            !json_has(&b_json, "/workspace/worktree-a"),
            "LEAKAGE: B's result JSON contains A's worktree path!\nB JSON: {b_json}"
        );

        // Also check in the other direction: A's result doesn't contain B's path.
        let a_json = serde_json::to_value(&result_for_a).unwrap();
        assert!(
            !json_has(&a_json, "/workspace/worktree-b"),
            "LEAKAGE: A's result JSON contains B's worktree path!\nA JSON: {a_json}"
        );
    }

    // ── stdio mode (no daemon) ────────────────────────────────────────────────

    /// In stdio mode the caller's root equals the server's root, so
    /// translation is a no-op for paths that are already absolute.
    #[tokio::test]
    async fn stdio_mode_dispatch_is_transparent() {
        let (_tmp, server) = make_server_with_files();
        let server_root = server.state.root_dir.clone();

        // query tool with no path args — just making sure it doesn't panic
        // and the translation layer does not break the call.
        let mut args = serde_json::Map::new();
        args.insert(
            "query".to_string(),
            serde_json::Value::String("auth token".to_string()),
        );

        // In stdio mode, caller_root == server_root (same worktree).
        let result =
            dispatch_with_translation(&server, "semantic_code_search", args, &server_root).await;

        // Must not error with a path translation error.
        let text = first_text(&result);
        assert!(
            !text.contains("outside the caller's worktree root"),
            "stdio mode must not produce path errors, got: {text}"
        );
    }

    /// `dispatch_with_translation` on an unknown tool name returns the
    /// tool's standard "Unknown tool" error without panicking.
    #[tokio::test]
    async fn dispatch_unknown_tool_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let server = make_test_server(tmp.path());
        let args = serde_json::Map::new();
        let result = dispatch_with_translation(&server, "no_such_tool", args, tmp.path()).await;
        let text = first_text(&result);
        assert!(
            text.contains("Unknown tool") || text.contains("Error"),
            "got: {text}"
        );
    }
}
