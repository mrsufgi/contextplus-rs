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
use crate::core::head_watcher::{HeadEvent, start_head_watcher};
use crate::core::memory_graph::MemoryGraph;
use crate::core::memory_merge::run_merge_ladder;
use crate::core::process_lifecycle;
use crate::ref_index::{RefId, RefIndex};
use crate::server::ContextPlusServer;
use crate::transport::client::{RegisterSession, SessionReady, read_frame, write_frame};
use crate::transport::paths;

/// Environment variable for the ref TTL (seconds) after the last session
/// disconnects. Default 24 h. `0` means immediate eviction.
pub const REF_TTL_SECS_ENV: &str = "CONTEXTPLUS_REF_TTL_SECS";
/// Default TTL: 24 hours.
pub const DEFAULT_REF_TTL_SECS: u64 = 24 * 60 * 60;

/// Read the ref TTL from the environment, falling back to the 24 h default.
pub fn ref_ttl_from_env() -> u64 {
    match std::env::var(REF_TTL_SECS_ENV) {
        Ok(s) if !s.trim().is_empty() => match s.trim().parse::<u64>() {
            Ok(n) => n,
            Err(_) => {
                tracing::warn!(
                    "{REF_TTL_SECS_ENV}={s:?} is not a valid u64; using default {DEFAULT_REF_TTL_SECS}"
                );
                DEFAULT_REF_TTL_SECS
            }
        },
        _ => DEFAULT_REF_TTL_SECS,
    }
}

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
///
/// After a successful bind the socket file is `chmod 600` so other users on
/// a shared host cannot connect to this workspace's daemon.
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

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("failed to bind socket: {}", socket_path.display()))?;

    // Restrict socket to owner-only so other users on the same host cannot
    // connect to this workspace's daemon.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&socket_path)
            .with_context(|| format!("stat({}) failed", socket_path.display()))?
            .permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(&socket_path, perms)
            .with_context(|| format!("chmod 600 {} failed", socket_path.display()))?;
    }

    Ok(listener)
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
        Ok(s) if !s.trim().is_empty() => match s.trim().parse::<u64>() {
            Ok(n) => n,
            Err(_) => {
                tracing::warn!(
                    "CONTEXTPLUS_DAEMON_IDLE_SECS={s:?} is not a valid u64; using default {DEFAULT_DAEMON_IDLE_SECS}"
                );
                DEFAULT_DAEMON_IDLE_SECS
            }
        },
        _ => DEFAULT_DAEMON_IDLE_SECS,
    }
}

/// Probe whether the daemon lock for `root_dir` is currently held by another
/// process. Returns `true` if the lock is held (daemon alive), `false` if it
/// is free (no daemon).
///
/// Callers should treat an `Err` as "unknown / don't unlink" to be safe.
pub(crate) fn probe_lock_held(root_dir: &Path) -> Result<bool> {
    match acquire_lock(root_dir)? {
        AcquireOutcome::Acquired(_guard) => {
            // We got the lock — no daemon is alive.
            // _guard drops here, releasing the lock immediately.
            Ok(false)
        }
        AcquireOutcome::AlreadyRunning => Ok(true),
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

/// Serve one MCP session over an accepted Unix stream. Performs the
/// `register_session` handshake, attaches the appropriate `RefIndex`, then
/// hands the stream off to rmcp. Logs and swallows errors — a bad client must
/// not take down the daemon.
async fn serve_connection(server: ContextPlusServer, mut stream: UnixStream) {
    let ttl_secs = ref_ttl_from_env();

    // ── Step 1: register_session handshake ──────────────────────────────────
    let reg: RegisterSession = match read_frame(&mut stream).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("register_session read failed: {e}");
            return;
        }
    };
    tracing::debug!(
        client_root = %reg.client_root.display(),
        head_sha = %reg.head_sha,
        client_pid = reg.client_pid,
        "register_session received"
    );

    // Reject immediately if draining.
    if server.state.draining.load(Ordering::Acquire) {
        let _ = write_frame(&mut stream, &SessionReady::RejectedDraining).await;
        return;
    }

    // ── Step 2: resolve ref_id from client_root ──────────────────────────────
    let canonical_root = reg
        .client_root
        .canonicalize()
        .unwrap_or_else(|_| reg.client_root.clone());
    let ref_id = RefId::for_canonical_path(&canonical_root);

    // Determine parent ref: find merge-base between client HEAD and primary.
    // For now: if the client root differs from the primary root, the primary
    // ref is the parent (CoW-fork). U6 will wire in proper merge-base lookup.
    let parent_ref_id = if ref_id != server.state.default_ref_id {
        Some(server.state.default_ref_id)
    } else {
        None
    };

    let head_sha = reg.head_sha.clone();
    let client_root = reg.client_root.clone();

    let ref_arc = server
        .state
        .attach_ref(ref_id, || {
            Arc::new(RefIndex::new_with_head(
                client_root.clone(),
                canonical_root.clone(),
                parent_ref_id,
                head_sha.clone(),
            ))
        })
        .await;

    tracing::debug!(
        ref_id = ref_id.0,
        sessions = ref_arc.session_count.load(Ordering::Acquire),
        "ref attached"
    );

    // ── Step 3: send session_ready ───────────────────────────────────────────
    let session_id = format!("{}-{}", ref_id.0, reg.client_pid);
    // TODO(U6): detect warming state once CAS/diff embedding is wired up.
    // For now we always reply Ready.
    let reply = SessionReady::Ready {
        session_id,
        ref_id: ref_id.0,
    };
    if let Err(e) = write_frame(&mut stream, &reply).await {
        tracing::warn!("session_ready write failed: {e}");
        server.state.detach_ref(ref_id, 0).await;
        return;
    }

    // ── Step 4: serve MCP over the remainder of the stream ──────────────────
    // Clone here so `server.state` is still accessible after `.serve()` moves
    // the server into the transport.
    let state = Arc::clone(&server.state);
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

    // ── Step 5: detach ref (decrement refcount, schedule eviction if 0) ─────
    state.detach_ref(ref_id, ttl_secs).await;
    tracing::debug!(ref_id = ref_id.0, "ref detached");
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
            // We intentionally treat SIGHUP as a drain signal here, not the
            // conventional "reload config". The daemon has no on-disk config to
            // reload; SIGHUP from a terminal close (parent shell exit) means our
            // stdio is gone anyway, so a clean drain is the right response. If
            // config-reload is added later, split this listener so SIGHUP routes
            // elsewhere.
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

    // HEAD watcher: track primary HEAD advances and trigger memory-graph
    // merge for overlay refs whose HEAD has become an ancestor.
    //
    // ## U4 / U7 seam
    //
    // Today the registry holds only the primary ref (U3 scaffolding) so the
    // merge loop is a no-op: there are no overlay graphs to fold in.  When
    // U4 lands and multiple refs populate `server.state.refs`, this task
    // will iterate over them and call `run_merge_ladder` for each that
    // qualifies.
    //
    // The `_head_watcher_handle` binding keeps the watcher alive for the
    // daemon's lifetime; dropping it shuts down the background thread.
    let _head_watcher_handle = spawn_head_watcher_task(&server, root_dir.clone());

    run(server, listener, socket_path, pid_path, idle_secs, lock).await?;
    Ok(true)
}

/// Spawn the HEAD-watcher + merge-dispatch background task.
///
/// Returns the watcher handle (drop = shutdown).  If the gitdir cannot be
/// found or the watcher fails to start, logs a warning and returns `None`
/// so the rest of the daemon still operates.
fn spawn_head_watcher_task(
    server: &ContextPlusServer,
    root_dir: PathBuf,
) -> Option<crate::core::head_watcher::HeadWatcherHandle> {
    // Locate the gitdir.  For a primary worktree `.git` is a directory; for
    // a linked worktree `.git` is a file — `git_worktree::resolve_primary_worktree`
    // already handles both, but here we need the gitdir itself, not just the
    // primary root.  We shell out to `git rev-parse --git-dir` to cover both
    // cases.
    let gitdir_out = std::process::Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .current_dir(&root_dir)
        .output();

    let gitdir = match gitdir_out {
        Ok(out) if out.status.success() => {
            let raw = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if std::path::Path::new(&raw).is_absolute() {
                std::path::PathBuf::from(raw)
            } else {
                root_dir.join(raw)
            }
        }
        _ => {
            tracing::warn!(
                root = %root_dir.display(),
                "head_watcher: could not resolve gitdir — HEAD-advance events disabled"
            );
            return None;
        }
    };

    let (handle, mut event_rx) = match start_head_watcher(root_dir.clone(), gitdir) {
        Ok(pair) => pair,
        Err(e) => {
            tracing::warn!("head_watcher: failed to start: {e} — merge events disabled");
            return None;
        }
    };

    // Capture what we need for the async task.
    let state = Arc::clone(&server.state);
    let root_str = root_dir.to_string_lossy().to_string();

    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            let HeadEvent::Advanced {
                old_sha: _,
                new_sha,
            } = event;

            // Suppress merges during drain.
            if state.draining.load(std::sync::atomic::Ordering::Acquire) {
                tracing::debug!(
                    "head_watcher: daemon draining — skipping merge for HEAD {}",
                    new_sha
                );
                continue;
            }

            // Iterate over all registered refs and check ancestry.
            // U3: the registry has only one ref (primary); no overlay nodes to
            // merge.  U4 will add secondary refs; this loop body will process
            // them once they carry real overlay memory graphs.
            let refs_snapshot: Vec<(crate::ref_index::RefId, Arc<crate::ref_index::RefIndex>)> = {
                let reg = state.refs.read().await;
                reg.iter().map(|(id, r)| (*id, Arc::clone(r))).collect()
            };

            for (ref_id, ref_index) in &refs_snapshot {
                let ref_head = match &ref_index.head_sha {
                    Some(h) => h.clone(),
                    None => continue, // no HEAD recorded yet — skip
                };

                // Skip the primary ref itself.
                if ref_index.parent_ref_id.is_none() {
                    continue;
                }

                // Check ancestry.
                let is_anc = crate::core::head_watcher::is_ancestor(&root_dir, &ref_head, &new_sha);
                if is_anc != Some(true) {
                    continue;
                }

                tracing::info!(
                    ref_id = ?ref_id,
                    ref_head = %ref_head,
                    primary_head = %new_sha,
                    "head_watcher: ref HEAD is ancestor of primary — entering merge ladder"
                );

                // U4/U7 seam: when RefIndex carries its own MemoryGraph
                // overlay, the merge runs here.  Today there is no overlay
                // stored on RefIndex so we run a no-op merge (empty overlay).
                //
                // Pattern to extend when U4 lands:
                //   let overlay = ref_index.memory_graph_overlay();
                //   let summary = run_merge_ladder(&overlay, &mut primary_graph, &[], draining);
                let overlay = MemoryGraph::new(); // empty — no-op
                let _ = state
                    .memory_graph
                    .get_graph(&root_str, |primary_graph| {
                        run_merge_ladder(
                            &overlay,
                            primary_graph,
                            &[],
                            state.draining.load(std::sync::atomic::Ordering::Acquire),
                        )
                    })
                    .await;
            }
        }
        tracing::debug!("head_watcher: event channel closed — merge task exiting");
    });

    Some(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All env-driven cases live in a single test so they don't race against
    /// each other on the shared `DAEMON_IDLE_SECS_ENV` var. cargo runs tests
    /// in parallel by default and `set_var`/`remove_var` aren't isolated.
    #[test]
    fn idle_secs_env_handling() {
        // SAFETY: test-only; we restore on exit. Env mutations from
        // separate tests on the same var are racy, so this combined test
        // owns the variable end-to-end.

        // Case: var absent → default
        unsafe {
            std::env::remove_var(DAEMON_IDLE_SECS_ENV);
        }
        assert_eq!(idle_secs_from_env(), DEFAULT_DAEMON_IDLE_SECS);

        // Case: non-numeric value → falls back to default
        unsafe {
            std::env::set_var(DAEMON_IDLE_SECS_ENV, "not-a-number");
        }
        assert_eq!(idle_secs_from_env(), DEFAULT_DAEMON_IDLE_SECS);

        // Case: "0" → disabled (returns 0)
        unsafe {
            std::env::set_var(DAEMON_IDLE_SECS_ENV, "0");
        }
        assert_eq!(idle_secs_from_env(), 0);

        // Case: positive integer is returned verbatim
        unsafe {
            std::env::set_var(DAEMON_IDLE_SECS_ENV, "300");
        }
        assert_eq!(idle_secs_from_env(), 300);

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

    /// `write_pid_file` writes the current PID to the pid file path.
    #[test]
    fn write_pid_file_writes_current_pid() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mcp_data = root.join(paths::MCP_DATA_DIR);
        std::fs::create_dir_all(&mcp_data).unwrap();

        write_pid_file(root);

        let pid_path = paths::daemon_pid_path(root);
        let content = std::fs::read_to_string(&pid_path).expect("pid file should exist");
        let parsed: u32 = content.trim().parse().expect("should be a valid pid");
        assert_eq!(parsed, std::process::id());
    }

    /// `bind_listener` on a fresh directory creates a usable UnixListener.
    #[cfg(unix)]
    #[tokio::test]
    async fn bind_listener_fresh_dir() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let listener = bind_listener(root).expect("bind on fresh dir should succeed");
        let socket_path = paths::daemon_socket_path(root);
        assert!(
            socket_path.exists(),
            "socket file should exist after bind_listener"
        );
        drop(listener);
    }

    /// Verify `acquire_lock` returns `AlreadyRunning` when same dir is locked.
    #[cfg(unix)]
    #[test]
    fn acquire_lock_already_running_arm() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        let _guard = match acquire_lock(root).unwrap() {
            AcquireOutcome::Acquired(g) => g,
            AcquireOutcome::AlreadyRunning => panic!("first acquire should succeed"),
        };

        match acquire_lock(root).unwrap() {
            AcquireOutcome::AlreadyRunning => {} // expected arm
            AcquireOutcome::Acquired(_) => panic!("should have been AlreadyRunning"),
        }
    }
}
