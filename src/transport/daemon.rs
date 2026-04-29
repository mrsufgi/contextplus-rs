//! Per-workspace daemon. Binds a Unix socket and serves any number of
//! concurrent MCP sessions from a single shared [`crate::server::SharedState`].
//!
//! Lifecycle (lock → bind → listen → accept loop):
//!
//! ```text
//! try_acquire_lock() ---fail---> caller becomes a client
//!     |
//!  success
//!     v
//! bind UnixListener (re-bind after stale-socket cleanup)
//!     |
//!     v
//! write pid file                              <--- best-effort, advisory
//!     |
//!     v
//! accept() ----new conn----> spawn task: server.clone().serve(stream)
//!     |        \                           |
//!     |         `--> client_count++       client_count--
//!     |                                    |
//!     |                                    v
//!     |                          if 0: arm idle-timer (default 30 min)
//!     |                                    |
//!     v                                    v
//! signal / drain ---------- run_cleanup → unlink socket, exit
//! ```
//!
//! Single-instance is enforced by a `flock(LOCK_EX|LOCK_NB)` on
//! `<root>/.mcp_data/contextplus.daemon.lock`. The lock is bound to the
//! file-descriptor lifetime: keep [`LockGuard`] alive and the lock holds.

use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use rmcp::ServiceExt;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Notify;

use crate::config::Config;
use crate::core::process_lifecycle;
use crate::server::ContextPlusServer;
use crate::transport::paths;

/// Default idle shutdown when running as a daemon — 30 minutes after the last
/// client disconnects. Override with `CONTEXTPLUS_DAEMON_IDLE_SECS`.
pub const DEFAULT_DAEMON_IDLE_SECS: u64 = 30 * 60;

/// Override env var for daemon idle shutdown seconds. `0` disables the timer
/// entirely (daemon stays up forever).
pub const DAEMON_IDLE_SECS_ENV: &str = "CONTEXTPLUS_DAEMON_IDLE_SECS";

/// Outcome of attempting to acquire the per-workspace daemon lock.
pub enum AcquireOutcome {
    /// We hold the lock and own the daemon for this workspace. The held
    /// `LockGuard` releases the advisory lock on drop.
    Acquired(LockGuard),
    /// Another process holds the lock — we should connect as a client.
    AlreadyRunning,
}

/// RAII container for an `flock(LOCK_EX|LOCK_NB)` advisory lock. The lock is
/// held by the file descriptor; closing the file (i.e. dropping this guard)
/// releases it.
pub struct LockGuard {
    _file: std::fs::File,
}

/// Try to acquire the daemon lock. Non-blocking: returns immediately whether
/// we got it or another daemon is running.
pub fn acquire_lock(root_dir: &Path) -> Result<AcquireOutcome> {
    let lock_path = paths::daemon_lock_path(root_dir);
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create daemon dir: {}", parent.display()))?;
    }

    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("failed to open lock file: {}", lock_path.display()))?;

    #[cfg(unix)]
    {
        // SAFETY: `file` is a valid open file. `flock` is documented as safe to
        // call on any open fd; success returns 0, failure returns -1 with errno
        // set. We close `file` automatically on Drop, which also releases the
        // lock per flock(2).
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            return Ok(AcquireOutcome::Acquired(LockGuard { _file: file }));
        }
        let err = std::io::Error::last_os_error();
        // Linux returns EWOULDBLOCK (== EAGAIN); macOS returns the same.
        // std::io::ErrorKind::WouldBlock covers both portably.
        if err.kind() == std::io::ErrorKind::WouldBlock {
            return Ok(AcquireOutcome::AlreadyRunning);
        }
        Err(err).context("flock(LOCK_EX|LOCK_NB) failed unexpectedly")
    }
    #[cfg(not(unix))]
    {
        // No daemon mode on non-Unix targets — caller falls back to stdio.
        let _ = file;
        Ok(AcquireOutcome::AlreadyRunning)
    }
}

/// Bind the Unix listener, removing any stale socket file from a crashed
/// previous daemon. Caller must already hold the daemon lock.
pub fn bind_listener(root_dir: &Path) -> Result<UnixListener> {
    let socket_path = paths::daemon_socket_path(root_dir);
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create socket dir: {}", parent.display()))?;
    }

    // Stale socket cleanup: a previous daemon may have crashed without
    // unlinking. Since we already hold the lock, removing the file is safe.
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)
            .with_context(|| format!("failed to remove stale socket: {}", socket_path.display()))?;
    }

    UnixListener::bind(&socket_path)
        .with_context(|| format!("failed to bind socket: {}", socket_path.display()))
}

/// Best-effort: write our PID to the daemon pid file. Failures are logged and
/// ignored — the lock file is the real source of truth.
pub fn write_pid_file(root_dir: &Path) {
    let pid_path = paths::daemon_pid_path(root_dir);
    let pid = std::process::id();
    if let Err(e) = std::fs::write(&pid_path, format!("{pid}\n")) {
        tracing::warn!("failed to write pid file {}: {e}", pid_path.display());
    }
}

/// Resolve daemon idle timeout from the env. `0` means disabled.
pub fn idle_secs_from_env() -> u64 {
    match std::env::var(DAEMON_IDLE_SECS_ENV) {
        Ok(v) => v.trim().parse::<u64>().unwrap_or(DEFAULT_DAEMON_IDLE_SECS),
        Err(_) => DEFAULT_DAEMON_IDLE_SECS,
    }
}

/// Run the daemon serve loop on `listener`. Returns when a shutdown signal
/// fires (SIGINT/SIGTERM/SIGHUP) or the idle timer expires.
///
/// `_lock` is consumed only to tie its lifetime to this function — drop on
/// return releases the advisory lock.
pub async fn run(
    server: ContextPlusServer,
    listener: UnixListener,
    socket_path: PathBuf,
    pid_path: PathBuf,
    idle_secs: u64,
    _lock: LockGuard,
) -> Result<()> {
    let client_count = Arc::new(AtomicUsize::new(0));
    let shutdown = Arc::new(Notify::new());
    let shutdown_flag = Arc::new(AtomicBool::new(false));

    // Idle timer: when client count hits 0, after `idle_secs` of zero clients
    // we trigger drain. `idle_notify` is poked on every (dis)connect so the
    // timer resets when a new client arrives.
    let idle_notify = Arc::new(Notify::new());
    if idle_secs > 0 {
        let cc = Arc::clone(&client_count);
        let n = Arc::clone(&idle_notify);
        let sd = Arc::clone(&shutdown);
        let sf = Arc::clone(&shutdown_flag);
        let timeout = Duration::from_secs(idle_secs);
        tokio::spawn(async move {
            loop {
                if sf.load(Ordering::Acquire) {
                    return;
                }
                // Wait until count == 0, then start the timeout.
                if cc.load(Ordering::Acquire) > 0 {
                    n.notified().await;
                    continue;
                }
                tokio::select! {
                    _ = tokio::time::sleep(timeout) => {
                        if cc.load(Ordering::Acquire) == 0 {
                            tracing::info!(
                                "daemon idle for {}s with no clients — initiating shutdown",
                                timeout.as_secs(),
                            );
                            sf.store(true, Ordering::Release);
                            sd.notify_waiters();
                            return;
                        }
                    }
                    _ = n.notified() => {
                        // New client connected — loop and re-check.
                    }
                }
            }
        });
    }

    // Drain watcher: when `state.draining` flips true (signal handler or
    // idle timer), wait for inflight==0 (or grace expiry) then notify
    // shutdown. Reuses Tier 1's drain primitives so in-flight tool calls
    // get to finish before the daemon exits.
    let drain_grace_secs = process_lifecycle::get_drain_grace_secs(
        std::env::var("CONTEXTPLUS_DRAIN_GRACE_SECS")
            .ok()
            .as_deref(),
    );
    {
        let draining = Arc::clone(&server.state.draining);
        let inflight = Arc::clone(&server.state.inflight);
        let sd = Arc::clone(&shutdown);
        let sf = Arc::clone(&shutdown_flag);
        // Background watcher; we never join its handle — process exit (or
        // the daemon shutdown branch) reaps it.
        drop(process_lifecycle::start_drain_watcher(
            draining,
            inflight,
            Duration::from_secs(drain_grace_secs),
            move |reason| {
                tracing::info!(?reason, "daemon drain watcher fired");
                sf.store(true, Ordering::Release);
                sd.notify_waiters();
            },
        ));
    }

    // Spawn signal handlers (SIGINT / SIGTERM / SIGHUP) — flip the shared
    // `draining` flag so the drain watcher above fires, and the dispatch
    // path rejects new tool calls.
    spawn_signal_listener(Arc::clone(&server.state.draining));

    tracing::info!(
        socket = %socket_path.display(),
        idle_secs,
        drain_grace_secs,
        "contextplus daemon listening",
    );

    // Accept loop, racing against shutdown.
    let accept_loop = {
        let server = server.clone();
        let client_count = Arc::clone(&client_count);
        let idle_notify = Arc::clone(&idle_notify);
        let shutdown_flag = Arc::clone(&shutdown_flag);
        async move {
            loop {
                let (stream, _addr) = match listener.accept().await {
                    Ok(s) => s,
                    Err(e) => {
                        if shutdown_flag.load(Ordering::Acquire) {
                            return;
                        }
                        tracing::warn!("accept() error: {e}");
                        continue;
                    }
                };
                if server.state.draining.load(Ordering::Acquire) {
                    // Refuse new clients while draining.
                    drop(stream);
                    continue;
                }
                client_count.fetch_add(1, Ordering::AcqRel);
                idle_notify.notify_waiters();
                let server_for_conn = server.clone();
                let cc = Arc::clone(&client_count);
                let n = Arc::clone(&idle_notify);
                tokio::spawn(async move {
                    serve_connection(server_for_conn, stream).await;
                    let prev = cc.fetch_sub(1, Ordering::AcqRel);
                    tracing::debug!("client disconnected (active before={prev})");
                    n.notify_waiters();
                });
            }
        }
    };

    tokio::select! {
        _ = accept_loop => {}
        _ = shutdown.notified() => {
            tracing::info!("daemon shutdown signal — exiting accept loop");
        }
    }

    // Cleanup: unlink socket, remove pid file. Lock guard drops at end of fn.
    if let Err(e) = std::fs::remove_file(&socket_path)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!("failed to unlink socket {}: {e}", socket_path.display());
    }
    let _ = std::fs::remove_file(&pid_path);

    // Final flush: persist memory graph + query embeddings before exit.
    server.state.ollama.flush_query_cache();
    if let Err(e) = server.state.memory_graph.flush().await {
        tracing::warn!("memory graph flush failed at daemon shutdown: {e}");
    }

    Ok(())
}

/// Serve one MCP session over an accepted Unix stream. Logs and swallows
/// errors — a bad client must not take down the daemon.
async fn serve_connection(server: ContextPlusServer, stream: UnixStream) {
    let (read_half, write_half) = stream.into_split();
    match server.serve((read_half, write_half)).await {
        Ok(running) => match running.waiting().await {
            Ok(reason) => {
                tracing::debug!(?reason, "client session ended");
            }
            Err(e) => {
                tracing::debug!("client session join error: {e}");
            }
        },
        Err(e) => {
            tracing::warn!("MCP handshake failed on Unix stream: {e}");
        }
    }
}

/// Spawn signal listeners that flip the drain flag. The drain watcher then
/// triggers shutdown once in-flight calls finish (or grace expires).
fn spawn_signal_listener(draining: Arc<AtomicBool>) {
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let mut sigterm = match signal(SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("failed to install SIGTERM handler: {e}");
                    return;
                }
            };
            let mut sighup = match signal(SignalKind::hangup()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("failed to install SIGHUP handler: {e}");
                    return;
                }
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("daemon received SIGINT — entering drain");
                }
                _ = sigterm.recv() => {
                    tracing::info!("daemon received SIGTERM — entering drain");
                }
                _ = sighup.recv() => {
                    tracing::info!("daemon received SIGHUP — entering drain");
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("daemon received Ctrl-C — entering drain");
        }
        draining.store(true, Ordering::Release);
    });
}

/// Top-level entry called from `main`. Acquire lock → bind → write pid → run.
/// Returns `Ok(false)` if another daemon is already running (caller falls
/// back to client mode).
pub async fn run_if_owner(root_dir: PathBuf, config: Config) -> Result<bool> {
    let lock = match acquire_lock(&root_dir)? {
        AcquireOutcome::Acquired(l) => l,
        AcquireOutcome::AlreadyRunning => return Ok(false),
    };

    let listener = bind_listener(&root_dir)?;
    write_pid_file(&root_dir);

    let socket_path = paths::daemon_socket_path(&root_dir);
    let pid_path = paths::daemon_pid_path(&root_dir);
    let idle_secs = idle_secs_from_env();

    let server = ContextPlusServer::new(root_dir.clone(), config.clone());

    // Pre-load memory graph + spawn debounce + tracker, just like the stdio path.
    let root_str = root_dir.to_string_lossy().to_string();
    if let Err(e) = server
        .state
        .memory_graph
        .get_graph(&root_str, |_g| {})
        .await
    {
        tracing::warn!("daemon: pre-load memory graph failed: {e}");
    }
    let _debounce = server.state.memory_graph.spawn_debounce_task();

    use crate::config::TrackerMode;
    if config.embed_tracker_mode == TrackerMode::Eager {
        server.ensure_tracker_started();
    }
    if config.warmup_on_start {
        server.spawn_warmup_task();
    }

    run(server, listener, socket_path, pid_path, idle_secs, lock).await?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both env-driven cases live in a single test so they don't race against
    /// each other on the shared `DAEMON_IDLE_SECS_ENV` var. cargo runs tests
    /// in parallel by default and `set_var`/`remove_var` aren't isolated.
    #[test]
    fn idle_secs_env_handling() {
        // SAFETY: test-only; we restore on exit. Env mutations from
        // separate tests on the same var are racy, so this combined test
        // owns the variable end-to-end.
        unsafe {
            std::env::remove_var(DAEMON_IDLE_SECS_ENV);
        }
        assert_eq!(idle_secs_from_env(), DEFAULT_DAEMON_IDLE_SECS);

        unsafe {
            std::env::set_var(DAEMON_IDLE_SECS_ENV, "not-a-number");
        }
        assert_eq!(idle_secs_from_env(), DEFAULT_DAEMON_IDLE_SECS);

        unsafe {
            std::env::remove_var(DAEMON_IDLE_SECS_ENV);
        }
    }

    #[cfg(unix)]
    #[test]
    fn lock_is_exclusive() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let g1 = match acquire_lock(root).unwrap() {
            AcquireOutcome::Acquired(g) => g,
            AcquireOutcome::AlreadyRunning => panic!("expected first attempt to win"),
        };
        match acquire_lock(root).unwrap() {
            AcquireOutcome::AlreadyRunning => {}
            AcquireOutcome::Acquired(_) => panic!("second attempt should be contended"),
        }
        drop(g1);
        // After dropping the first guard the lock should be free again.
        match acquire_lock(root).unwrap() {
            AcquireOutcome::Acquired(_) => {}
            AcquireOutcome::AlreadyRunning => {
                panic!("lock not released after first guard dropped")
            }
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bind_clears_stale_socket() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mcp_data = root.join(paths::MCP_DATA_DIR);
        std::fs::create_dir_all(&mcp_data).unwrap();
        let stale = paths::daemon_socket_path(root);
        std::fs::write(&stale, b"stale junk").unwrap();
        let _listener = bind_listener(root).expect("bind should clean up stale socket");
        assert!(stale.exists(), "fresh socket should be created");
    }
}
