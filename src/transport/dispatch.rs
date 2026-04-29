//! Transport mode resolution and implicit-default dispatch.
//!
//! This module owns the logic that was previously inlined in `src/main.rs` for
//! choosing between daemon-proxy and direct-stdio transports, as well as the
//! `run_mcp_server` helper that starts the full stdio server. Extracting these
//! to the library crate makes the behaviour unit-testable and keeps `main.rs`
//! as a thin entry-point.
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
    let want_daemon = match mode {
        TransportMode::Stdio => false,
        TransportMode::Daemon => true,
        TransportMode::Auto => mcp_data_writable(&root_dir),
    };

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
}
