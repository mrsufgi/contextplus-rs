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

use std::collections::HashMap;
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
use crate::ref_index::{RefId, RefIndex};
use crate::server::ContextPlusServer;
use crate::transport::client::{
    RegisterSession, SearchConfig, SessionReady, read_frame, write_frame,
};
use crate::transport::paths;

pub const DAEMON_LOG_MAX_BYTES: u64 = 4 * 1024 * 1024;

pub fn daemon_log_path(root_dir: &Path) -> PathBuf {
    paths::daemon_dir(root_dir).join("logs/daemon.log")
}

pub fn open_daemon_log(root_dir: &Path) -> Result<std::fs::File> {
    open_log_file(&daemon_log_path(root_dir))
}

pub fn open_log_file(path: &Path) -> Result<std::fs::File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if file.metadata()?.len() > DAEMON_LOG_MAX_BYTES {
        file.set_len(0)?;
    }
    Ok(file)
}

pub fn bounded_log_writer(path: &Path) -> Result<impl std::io::Write + Send + 'static> {
    struct BoundedLog(std::fs::File);
    impl std::io::Write for BoundedLog {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.0.metadata()?.len() + bytes.len() as u64 > DAEMON_LOG_MAX_BYTES {
                self.0.set_len(0)?;
            }
            self.0.write(bytes)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.0.flush()
        }
    }
    Ok(BoundedLog(open_log_file(path)?))
}

pub fn install_panic_hook(log_path: &Path) -> Result<()> {
    use std::io::Write;
    let file = std::sync::Mutex::new(bounded_log_writer(log_path)?);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Ok(mut file) = file.lock() {
            let backtrace = std::backtrace::Backtrace::force_capture();
            let _ = writeln!(file, "PANIC: {info}\n{backtrace}");
            let _ = file.flush();
        }
        previous(info);
    }));
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ConfigDifference {
    pub field: &'static str,
    pub daemon_value: String,
    pub bridge_value: String,
}

pub(crate) fn compare_search_config(
    daemon: &Config,
    bridge: &SearchConfig,
) -> Vec<ConfigDifference> {
    let daemon = SearchConfig::from(daemon);
    daemon
        .reported_fields()
        .into_iter()
        .zip(bridge.reported_fields())
        .filter_map(|((field, daemon_value), (bridge_field, bridge_value))| {
            debug_assert_eq!(field, bridge_field);
            (daemon_value != bridge_value).then_some(ConfigDifference {
                field,
                daemon_value,
                bridge_value,
            })
        })
        .collect()
}

pub(crate) struct AppliedConfigValue {
    pub key: String,
    pub inherited_value: Option<String>,
    pub configured_value: String,
}

pub(crate) struct DaemonConfigResolution {
    pub config: Config,
    pub applied_values: Vec<AppliedConfigValue>,
}

const SEARCH_CONFIG_KEYS: &[&str] = &[
    "CONTEXTPLUS_EMBED_PROVIDER",
    "CONTEXTPLUS_CHAT_PROVIDER",
    "OLLAMA_HOST",
    "OLLAMA_EMBED_MODEL",
    "OLLAMA_CHAT_MODEL",
    "CONTEXTPLUS_OPENAI_BASE_URL",
    "CONTEXTPLUS_OPENAI_EMBED_MODEL",
    "CONTEXTPLUS_OPENAI_CHAT_MODEL",
    "CONTEXTPLUS_CHAT_BASE_URL",
    "CONTEXTPLUS_CLAUDE_PATH",
    "CONTEXTPLUS_CLAUDE_MODEL",
    "CONTEXTPLUS_ANTHROPIC_CHAT_MODEL",
    "CONTEXTPLUS_EMBED_QUERY_PREFIX",
    "CONTEXTPLUS_EMBED_DOC_PREFIX",
    "CONTEXTPLUS_EMBED_DOC_SHAPE",
    "CONTEXTPLUS_EMBED_BATCH_SIZE",
    "CONTEXTPLUS_EMBED_BUDGET_MS",
    "CONTEXTPLUS_EMBED_FILL_BATCH_TIMEOUT_MS",
    "CONTEXTPLUS_EMBED_TRACKER",
    "CONTEXTPLUS_EMBED_TRACKER_DEBOUNCE_MS",
    "CONTEXTPLUS_EMBED_TRACKER_MAX_FILES",
    "CONTEXTPLUS_IGNORE_DIRS",
    "CONTEXTPLUS_CACHE_TTL_SECS",
    "CONTEXTPLUS_MAX_EMBED_FILE_SIZE",
    "CONTEXTPLUS_EMBED_NUM_GPU",
    "CONTEXTPLUS_EMBED_MAIN_GPU",
    "CONTEXTPLUS_EMBED_NUM_THREAD",
    "CONTEXTPLUS_EMBED_NUM_BATCH",
    "CONTEXTPLUS_EMBED_NUM_CTX",
    "CONTEXTPLUS_EMBED_LOW_VRAM",
    "CONTEXTPLUS_EMBED_CHUNK_CHARS",
    "CONTEXTPLUS_QUERY_BATCH_SIZE",
    "CONTEXTPLUS_WARMUP_ON_START",
    "CONTEXTPLUS_HNSW_EF_CONSTRUCTION",
    "CONTEXTPLUS_HNSW_EF_SEARCH",
    "CONTEXTPLUS_REF_WARMUP_MODE",
    "CONTEXTPLUS_OLLAMA_MAX_CONCURRENT",
];

fn matching_server(document: &serde_json::Value) -> Option<&serde_json::Value> {
    let servers = document.get("mcpServers")?.as_object()?;
    servers.get("contextplus").or_else(|| {
        servers.values().find(|server| {
            server
                .get("command")
                .and_then(serde_json::Value::as_str)
                .and_then(|command| Path::new(command).file_name())
                .is_some_and(|name| name == "contextplus-rs")
        })
    })
}

pub(crate) fn resolve_daemon_config_contents(
    contents: Option<&str>,
    inherited_env: &HashMap<String, String>,
) -> serde_json::Result<DaemonConfigResolution> {
    let document = contents
        .map(serde_json::from_str::<serde_json::Value>)
        .transpose()?;
    let mut env = inherited_env.clone();
    let mut applied_values = Vec::new();
    if let Some(values) = document
        .as_ref()
        .and_then(matching_server)
        .and_then(|server| server.get("env"))
        .and_then(serde_json::Value::as_object)
    {
        for &key in SEARCH_CONFIG_KEYS {
            if let Some(value) = values.get(key).and_then(serde_json::Value::as_str) {
                if inherited_env.get(key).map(String::as_str) != Some(value) {
                    applied_values.push(AppliedConfigValue {
                        key: key.to_string(),
                        inherited_value: inherited_env.get(key).cloned(),
                        configured_value: value.to_string(),
                    });
                }
                env.insert(key.to_string(), value.to_string());
            }
        }
    }
    Ok(DaemonConfigResolution {
        config: Config::from_env_map(&env),
        applied_values,
    })
}

pub(crate) fn resolve_daemon_startup_config(
    root_dir: &Path,
    inherited_env: &HashMap<String, String>,
) -> DaemonConfigResolution {
    let path = crate::core::git_worktree::resolve_primary_worktree(root_dir).join(".mcp.json");
    let contents = std::fs::read_to_string(&path).ok();
    let mut resolved = match resolve_daemon_config_contents(contents.as_deref(), inherited_env) {
        Ok(resolved) => resolved,
        Err(error) => {
            tracing::warn!(path = %path.display(), reason = %error, "cannot parse daemon config JSON; using process environment");
            resolve_daemon_config_contents(None, inherited_env).expect("no file is valid")
        }
    };
    if contents
        .as_deref()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
        .as_ref()
        .and_then(matching_server)
        .is_some()
    {
        resolved.config.config_source = Some(path.clone());
    }
    let changes = resolved
        .applied_values
        .iter()
        .map(|value| {
            format!(
                "{}: inherited={:?}, configured={:?}",
                value.key,
                value.inherited_value.as_deref().unwrap_or("<unset>"),
                value.configured_value
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    tracing::info!(path = %path.display(), source = %resolved.config.config_source.as_deref()
        .map(|p| p.display().to_string()).unwrap_or_else(|| "process environment".into()),
        overrides = %changes, "daemon search config source");
    resolved
}

fn config_warning(
    differences: &[ConfigDifference],
    config_source: Option<&Path>,
) -> Option<String> {
    if differences.is_empty() {
        return None;
    }
    let details = differences
        .iter()
        .map(|difference| {
            format!(
                "{}={} but your MCP config sets {}",
                difference.field, difference.daemon_value, difference.bridge_value
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let source = config_source
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "process environment".into());
    Some(format!(
        "contextplus warning: daemon config comes from {source}: {details}; differing bridge values are ignored."
    ))
}

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

    // Final flush: persist query embeddings and unwritten snapshots before exit.
    server.state.ollama.flush_query_cache();
    server.state.flush_snapshots().await;

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

    let differences = reg
        .search_config
        .as_ref()
        .map(|bridge| compare_search_config(&server.state.config, bridge))
        .unwrap_or_default();
    for difference in &differences {
        tracing::warn!(
            field = difference.field,
            daemon_value = difference.daemon_value,
            bridge_value = difference.bridge_value,
            source = %server.state.config.config_source.as_deref().map(|path| path.display().to_string()).unwrap_or_else(|| "process environment".into()),
            "bridge search configuration differs from daemon config source; bridge values are ignored"
        );
    }
    let warning = config_warning(&differences, server.state.config.config_source.as_deref());

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

    // ── Step 2b: initialize CAS on-disk layout for this ref ─────────────────
    // For the primary ref this is a no-op (idempotent empty-manifest creation).
    // For non-primary (worktree) refs this creates the ref directory and writes
    // the parent pointer so chunk lookups can chain through the primary's
    // manifest (U12 diff-only embedding).
    {
        let mcp_data = server.state.root_dir.join(paths::MCP_DATA_DIR);
        let model = server.state.config.document_cache_identity();
        let parent_ref_opt = match parent_ref_id {
            Some(pid) => server.state.ref_index(pid).await,
            None => None,
        };
        if let Err(e) = ref_arc.fork_from(&mcp_data, &model, parent_ref_opt.as_deref()) {
            tracing::warn!(ref_id = ref_id.0, "CAS fork_from failed (non-fatal): {e}");
        }
    }

    // ── Step 2c: per-ref warmup (U18) ────────────────────────────────────────
    // Fire-and-forget: errors inside `spawn_ref_warmup` are logged and never
    // propagate here.  Idempotent — safe to call for every connection including
    // reconnects from the same worktree root.
    server.spawn_ref_warmup(ref_id);

    // ── Step 2d: per-ref embedding tracker (U11) ─────────────────────────────
    // Mirror the daemon's startup behaviour for the default ref so attached
    // worktree refs also pick up live file changes in Eager mode. Lazy mode
    // defers tracker startup until the first tool call on the session, which
    // resolves to this ref via `session_ref_id` and calls
    // `ensure_tracker_started` automatically.
    if server.state.config.embed_tracker_mode == crate::config::TrackerMode::Eager {
        server.ensure_tracker_started_for(ref_id).await;
    }

    // ── Step 3: send session_ready ───────────────────────────────────────────
    let session_id = format!("{}-{}", ref_id.0, reg.client_pid);
    // For now we always reply Ready.
    let reply = SessionReady::Ready {
        session_id,
        ref_id: ref_id.0,
    };
    if let Err(e) = write_frame(&mut stream, &reply).await {
        tracing::warn!("session_ready write failed: {e}");
        server
            .state
            .detach_ref(ref_id, std::time::Duration::ZERO)
            .await;
        return;
    }

    // ── Step 4: serve MCP over the remainder of the stream ──────────────────
    // Build a session-scoped server: clone shares the Arc<SharedState> but
    // sets session_ref_id so every subsequent tool call resolves to this
    // connection's registered worktree (U9).
    let session_server = server.with_session_config_warning(ref_id, warning);
    let state = Arc::clone(&session_server.state);
    let (read_half, write_half) = stream.into_split();
    match session_server.serve((read_half, write_half)).await {
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
    state
        .detach_ref(ref_id, std::time::Duration::from_secs(ttl_secs))
        .await;
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
fn daemon_server(root_dir: &Path, config: Config) -> ContextPlusServer {
    let primary_root = crate::core::git_worktree::resolve_primary_worktree(root_dir);
    ContextPlusServer::new(primary_root, config)
}

pub async fn run_if_owner(root_dir: PathBuf, _config: Config) -> Result<bool> {
    let lock = match acquire_lock(&root_dir)? {
        AcquireOutcome::Acquired(l) => l,
        AcquireOutcome::AlreadyRunning => return Ok(false),
    };

    let config = resolve_daemon_startup_config(&root_dir, &crate::config::env_snapshot()).config;
    let listener = bind_listener(&root_dir)?;
    write_pid_file(&root_dir);

    let socket_path = paths::daemon_socket_path(&root_dir);
    let pid_path = paths::daemon_pid_path(&root_dir);
    let idle_secs = idle_secs_from_env();

    let server = daemon_server(&root_dir, config.clone());

    use crate::config::TrackerMode;
    if config.embed_tracker_mode == TrackerMode::Eager {
        server.ensure_tracker_started().await;
    }
    if config.warmup_on_start {
        server.spawn_warmup_task(true);
    }

    run(server, listener, socket_path, pid_path, idle_secs, lock).await?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn linked_worktree_fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        let primary = temp.path().join("primary");
        let linked = temp.path().join("linked");
        let linked_gitdir = primary.join(".git/worktrees/linked");
        std::fs::create_dir_all(&linked_gitdir).unwrap();
        std::fs::create_dir_all(&linked).unwrap();
        std::fs::write(linked_gitdir.join("commondir"), "../..").unwrap();
        std::fs::write(
            linked.join(".git"),
            format!("gitdir: {}\n", linked_gitdir.display()),
        )
        .unwrap();
        (temp, primary, linked)
    }

    fn env_map(values: &[(&str, &str)]) -> HashMap<String, String> {
        values
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
    }

    fn named_contextplus_config(env: serde_json::Value) -> String {
        serde_json::json!({
            "mcpServers": {
                "contextplus": {
                    "command": "/opt/contextplus-rs",
                    "env": env
                }
            }
        })
        .to_string()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn daemon_started_from_linked_worktree_uses_primary_and_registers_linked_child() {
        let (_temp, primary, linked) = linked_worktree_fixture();
        let primary = primary.canonicalize().unwrap();
        let linked = linked.canonicalize().unwrap();
        let mut config = Config::from_env();
        config.embed_tracker_mode = crate::config::TrackerMode::Off;
        config.ref_warmup_mode = crate::config::RefWarmupMode::Off;

        let server = daemon_server(&linked, config);
        let inspection = server.clone();
        let (mut bridge_stream, daemon_stream) = UnixStream::pair().unwrap();
        let connection = tokio::spawn(serve_connection(server, daemon_stream));

        write_frame(
            &mut bridge_stream,
            &RegisterSession {
                client_root: linked.clone(),
                head_sha: "linked-head".into(),
                client_pid: 42,
                search_config: None,
            },
        )
        .await
        .unwrap();
        let ready: SessionReady =
            tokio::time::timeout(Duration::from_secs(1), read_frame(&mut bridge_stream))
                .await
                .expect("daemon did not register the linked-worktree session")
                .unwrap();
        assert!(matches!(ready, SessionReady::Ready { .. }));

        let default_ref = inspection
            .state
            .default_ref()
            .expect("daemon default ref must exist");
        assert_eq!(default_ref.canonical_root, primary);
        assert_eq!(inspection.state.root_dir, primary);

        let linked_id = RefId::for_canonical_path(&linked);
        let refs = inspection.state.refs.read().await;
        let linked_ref = refs.get(&linked_id).expect("linked ref was not registered");
        assert_eq!(linked_ref.canonical_root, linked);
        assert_eq!(
            linked_ref.parent_ref_id,
            Some(inspection.state.default_ref_id)
        );
        assert_ne!(linked_id, inspection.state.default_ref_id);

        drop(refs);
        drop(bridge_stream);
        connection.abort();
    }

    #[test]
    fn daemon_root_keeps_primary_repo_and_non_git_directory_unchanged() {
        let temp = tempfile::tempdir().unwrap();
        let primary = temp.path().join("primary");
        let non_git = temp.path().join("non-git");
        std::fs::create_dir_all(primary.join(".git")).unwrap();
        std::fs::create_dir_all(&non_git).unwrap();

        for root in [primary, non_git] {
            let canonical = root.canonicalize().unwrap();
            let server = daemon_server(&root, Config::from_env());
            let default_ref = server
                .state
                .default_ref()
                .expect("daemon default ref must exist");
            assert_eq!(server.state.root_dir, canonical);
            assert_eq!(default_ref.canonical_root, canonical);
        }
    }

    #[cfg(unix)]
    #[test]
    fn panic_log_child_process_helper() {
        let Some(root_dir) = std::env::var_os("CONTEXTPLUS_TEST_PANIC_LOG_ROOT") else {
            return;
        };
        let log_path = daemon_log_path(Path::new(&root_dir));
        install_panic_hook(&log_path).unwrap();
        panic!("daemon panic sentinel");
    }

    #[cfg(unix)]
    #[test]
    fn daemon_log_is_size_bounded_and_panic_hook_records_message_and_location() {
        use std::process::{Command, Stdio};

        let dir = tempfile::tempdir().unwrap();
        let canonical_root = dir.path().canonicalize().unwrap();
        let root = canonical_root.as_path();
        let log_path = daemon_log_path(root);
        assert_eq!(
            log_path,
            root.join(".mcp_data").join("logs").join("daemon.log")
        );
        std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
        std::fs::write(&log_path, vec![b'x'; DAEMON_LOG_MAX_BYTES as usize + 1]).unwrap();
        drop(open_daemon_log(root).unwrap());
        assert!(
            std::fs::metadata(&log_path).unwrap().len() <= DAEMON_LOG_MAX_BYTES,
            "oversized daemon log was not truncated or rotated"
        );

        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "transport::daemon::tests::panic_log_child_process_helper",
                "--nocapture",
            ])
            .env("CONTEXTPLUS_TEST_PANIC_LOG_ROOT", root)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(!status.success(), "panic helper unexpectedly exited zero");

        let contents = std::fs::read_to_string(&log_path).unwrap();
        assert!(
            contents.contains("daemon panic sentinel"),
            "panic payload missing from daemon log: {contents}"
        );
        assert!(
            contents.contains("daemon.rs"),
            "panic location missing from daemon log: {contents}"
        );
    }

    #[test]
    fn search_config_comparison_lists_exactly_differing_fields() {
        // Compare the full protocol snapshot, not selected daemon fields.
        let daemon = Config::from_env();
        let mut bridge = crate::transport::client::SearchConfig::from(&daemon);
        bridge.ollama_embed_model = "nomic-embed-text".into();
        bridge.ollama_host = "http://bridge-ollama:11434".into();
        bridge.max_embed_file_size += 1;

        let differences = compare_search_config(&daemon, &bridge);
        let names: Vec<_> = differences.iter().map(|diff| diff.field).collect();

        assert_eq!(
            names,
            [
                "OLLAMA_EMBED_MODEL",
                "OLLAMA_HOST",
                "CONTEXTPLUS_MAX_EMBED_FILE_SIZE"
            ]
        );
        let warning = config_warning(&differences, None).unwrap();
        for field in names {
            assert!(
                warning.contains(field),
                "warning omitted {field}: {warning}"
            );
        }
    }

    #[test]
    fn primary_mcp_file_overrides_inherited_search_environment() {
        let inherited = env_map(&[
            ("OLLAMA_EMBED_MODEL", "model-from-spawning-session"),
            ("OLLAMA_HOST", "http://inherited:11434"),
        ]);
        let contents = named_contextplus_config(serde_json::json!({
            "OLLAMA_EMBED_MODEL": "model-from-primary-file"
        }));

        let resolved = resolve_daemon_config_contents(Some(&contents), &inherited).unwrap();

        assert_eq!(
            resolved.config.ollama_embed_model,
            "model-from-primary-file"
        );
        assert_eq!(resolved.config.ollama_host, "http://inherited:11434");
        assert_eq!(resolved.applied_values.len(), 1);
        assert_eq!(resolved.applied_values[0].key, "OLLAMA_EMBED_MODEL");
        assert_eq!(
            resolved.applied_values[0].inherited_value.as_deref(),
            Some("model-from-spawning-session")
        );
        assert_eq!(
            resolved.applied_values[0].configured_value,
            "model-from-primary-file"
        );
    }

    #[test]
    fn secret_named_file_values_are_ignored_and_inherited_secrets_are_preserved() {
        let inherited = env_map(&[
            ("CONTEXTPLUS_EMBED_PROVIDER", "openai"),
            ("CONTEXTPLUS_OPENAI_API_KEY", "inherited-contextplus-key"),
            ("OPENAI_API_KEY", "inherited-openai-key"),
            ("OLLAMA_API_KEY", "inherited-ollama-key"),
            ("OLLAMA_EMBED_MODEL", "inherited-model"),
        ]);
        let contents = named_contextplus_config(serde_json::json!({
            "CONTEXTPLUS_OPENAI_API_KEY": "file-contextplus-key",
            "OPENAI_API_KEY": "file-openai-key",
            "OLLAMA_API_KEY": "file-ollama-key",
            "CONTEXTPLUS_AUTH_TOKEN": "file-token",
            "CONTEXTPLUS_CLIENT_SECRET": "file-secret",
            "CONTEXTPLUS_PASSWORD": "file-password",
            "CONTEXTPLUS_TYPO": "file-unknown-setting",
            "OLLAMA_EMBED_MODEL": "file-model"
        }));

        let resolved = resolve_daemon_config_contents(Some(&contents), &inherited).unwrap();

        assert_eq!(resolved.config.ollama_embed_model, "file-model");
        assert_eq!(
            resolved.config.openai_api_key.as_deref(),
            Some("inherited-contextplus-key")
        );
        assert_eq!(
            resolved.config.ollama_api_key.as_deref(),
            Some("inherited-ollama-key")
        );
        let applied_keys: Vec<_> = resolved
            .applied_values
            .iter()
            .map(|value| value.key.as_str())
            .collect();
        assert_eq!(applied_keys, ["OLLAMA_EMBED_MODEL"]);
    }

    #[test]
    fn absent_file_key_falls_back_to_inherited_environment() {
        let inherited = env_map(&[
            ("OLLAMA_EMBED_MODEL", "inherited-model"),
            ("OLLAMA_CHAT_MODEL", "inherited-chat-model"),
            ("CONTEXTPLUS_WARMUP_ON_START", "false"),
        ]);
        let contents = named_contextplus_config(serde_json::json!({
            "OLLAMA_EMBED_MODEL": "file-model"
        }));

        let resolved = resolve_daemon_config_contents(Some(&contents), &inherited).unwrap();

        assert_eq!(resolved.config.ollama_embed_model, "file-model");
        assert_eq!(resolved.config.ollama_chat_model, "inherited-chat-model");
        assert!(!resolved.config.warmup_on_start);
    }

    #[test]
    fn no_mcp_file_keeps_inherited_environment_without_warning() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".git")).unwrap();
        let inherited = env_map(&[
            ("OLLAMA_EMBED_MODEL", "inherited-model"),
            ("OLLAMA_HOST", "http://inherited:11434"),
        ]);
        let (logs, _guard) = crate::test_logs::captured_info_logs();

        let resolved = resolve_daemon_startup_config(root.path(), &inherited);
        let logs = crate::test_logs::logs_as_string(&logs);

        assert_eq!(resolved.config.ollama_embed_model, "inherited-model");
        assert_eq!(resolved.config.ollama_host, "http://inherited:11434");
        assert!(
            !logs.lines().any(|line| line.contains(" WARN ")),
            "missing .mcp.json should not warn: {logs}"
        );
    }

    #[test]
    fn no_matching_server_entry_keeps_inherited_environment_without_warning() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".git")).unwrap();
        std::fs::write(
            root.path().join(".mcp.json"),
            serde_json::json!({
                "mcpServers": {
                    "other-server": {
                        "command": "/opt/not-contextplus",
                        "env": {"OLLAMA_EMBED_MODEL": "file-model"}
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        let inherited = env_map(&[("OLLAMA_EMBED_MODEL", "inherited-model")]);
        let (logs, _guard) = crate::test_logs::captured_info_logs();

        let resolved = resolve_daemon_startup_config(root.path(), &inherited);
        let logs = crate::test_logs::logs_as_string(&logs);

        assert_eq!(resolved.config.ollama_embed_model, "inherited-model");
        assert!(
            !logs.lines().any(|line| line.contains(" WARN ")),
            "unmatched .mcp.json should not warn: {logs}"
        );
    }

    #[test]
    fn malformed_mcp_file_keeps_environment_and_warns_once_with_path_and_reason() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".git")).unwrap();
        let config_path = root.path().join(".mcp.json");
        std::fs::write(&config_path, r#"{"secret":"must-not-be-logged""#).unwrap();
        let inherited = env_map(&[("OLLAMA_EMBED_MODEL", "inherited-model")]);
        let (logs, _guard) = crate::test_logs::captured_info_logs();

        let resolved = resolve_daemon_startup_config(root.path(), &inherited);
        let logs = crate::test_logs::logs_as_string(&logs);
        let warnings: Vec<_> = logs
            .lines()
            .filter(|line| line.contains(" WARN "))
            .collect();

        assert_eq!(resolved.config.ollama_embed_model, "inherited-model");
        assert_eq!(warnings.len(), 1, "expected one warning: {logs}");
        assert!(warnings[0].contains(&config_path.display().to_string()));
        assert!(
            warnings[0].contains("parse") || warnings[0].contains("JSON"),
            "warning did not name the reason: {}",
            warnings[0]
        );
        assert!(!logs.contains("must-not-be-logged"));
    }

    #[test]
    fn contextplus_entry_is_matched_by_name() {
        let contents = serde_json::json!({
            "mcpServers": {
                "contextplus": {
                    "command": "/opt/some-other-binary",
                    "env": {
                        "CONTEXTPLUS_OPENAI_BASE_URL": "https://file.example/v1",
                        "CONTEXTPLUS_OPENAI_EMBED_MODEL": "file-embed-model",
                        "CONTEXTPLUS_CLAUDE_MODEL": "file-claude-model",
                        "CONTEXTPLUS_ANTHROPIC_CHAT_MODEL": "file-anthropic-model"
                    }
                },
                "also-matches-command": {
                    "command": "/opt/contextplus-rs",
                    "env": {
                        "CONTEXTPLUS_OPENAI_BASE_URL": "https://wrong.example/v1",
                        "CONTEXTPLUS_OPENAI_EMBED_MODEL": "wrong-embed-model",
                        "CONTEXTPLUS_CLAUDE_MODEL": "wrong-claude-model",
                        "CONTEXTPLUS_ANTHROPIC_CHAT_MODEL": "wrong-anthropic-model"
                    }
                }
            }
        })
        .to_string();

        let resolved = resolve_daemon_config_contents(Some(&contents), &HashMap::new()).unwrap();

        assert_eq!(resolved.config.openai_base_url, "https://file.example/v1");
        assert_eq!(resolved.config.openai_embed_model, "file-embed-model");
        assert_eq!(resolved.config.claude_model, "file-claude-model");
        assert_eq!(resolved.config.anthropic_chat_model, "file-anthropic-model");
    }

    #[test]
    fn contextplus_entry_is_matched_by_command_basename() {
        let contents = serde_json::json!({
            "mcpServers": {
                "legacy-name": {
                    "command": "/opt/contextplus/bin/contextplus-rs",
                    "env": {"OLLAMA_EMBED_MODEL": "file-model"}
                }
            }
        })
        .to_string();

        let resolved = resolve_daemon_config_contents(Some(&contents), &HashMap::new()).unwrap();

        assert_eq!(resolved.config.ollama_embed_model, "file-model");
    }

    #[test]
    fn linked_worktree_loads_config_from_primary_worktree() {
        let temp = tempfile::tempdir().unwrap();
        let primary = temp.path().join("primary");
        let linked = temp.path().join("linked");
        let linked_gitdir = primary.join(".git/worktrees/linked");
        std::fs::create_dir_all(&linked_gitdir).unwrap();
        std::fs::create_dir_all(&linked).unwrap();
        std::fs::write(linked_gitdir.join("commondir"), "../..").unwrap();
        std::fs::write(
            linked.join(".git"),
            format!("gitdir: {}\n", linked_gitdir.display()),
        )
        .unwrap();
        let config_path = primary.join(".mcp.json");
        std::fs::write(
            &config_path,
            named_contextplus_config(serde_json::json!({
                "OLLAMA_EMBED_MODEL": "model-from-primary-file"
            })),
        )
        .unwrap();
        let inherited = env_map(&[("OLLAMA_EMBED_MODEL", "model-from-linked-session")]);

        let resolved = resolve_daemon_startup_config(&linked, &inherited);

        assert_eq!(
            resolved.config.ollama_embed_model,
            "model-from-primary-file"
        );
        assert_eq!(
            resolved.config.config_source.as_deref(),
            Some(config_path.canonicalize().unwrap().as_path())
        );
    }

    #[test]
    fn startup_log_names_source_and_changed_values_without_secrets() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".git")).unwrap();
        let config_path = root.path().join(".mcp.json");
        std::fs::write(
            &config_path,
            named_contextplus_config(serde_json::json!({
                "OLLAMA_EMBED_MODEL": "file-model",
                "OLLAMA_HOST": "http://same:11434",
                "OPENAI_API_KEY": "file-openai-secret",
                "CONTEXTPLUS_OPENAI_API_KEY": "file-contextplus-secret"
            })),
        )
        .unwrap();
        let inherited = env_map(&[
            ("OLLAMA_EMBED_MODEL", "inherited-model"),
            ("OLLAMA_HOST", "http://same:11434"),
            ("OPENAI_API_KEY", "inherited-openai-secret"),
            ("CONTEXTPLUS_OPENAI_API_KEY", "inherited-contextplus-secret"),
        ]);
        let (logs, _guard) = crate::test_logs::captured_info_logs();

        let _resolved = resolve_daemon_startup_config(root.path(), &inherited);
        let logs = crate::test_logs::logs_as_string(&logs);
        let source_lines: Vec<_> = logs
            .lines()
            .filter(|line| line.contains("daemon search config source"))
            .collect();

        assert_eq!(source_lines.len(), 1, "startup source log: {logs}");
        let source_line = source_lines[0];
        assert!(source_line.contains(&config_path.display().to_string()));
        assert!(source_line.contains("OLLAMA_EMBED_MODEL"));
        assert!(source_line.contains("inherited-model"));
        assert!(source_line.contains("file-model"));
        assert!(
            !source_line.contains("OLLAMA_HOST"),
            "unchanged values must not be listed: {source_line}"
        );
        for forbidden in [
            "OPENAI_API_KEY",
            "CONTEXTPLUS_OPENAI_API_KEY",
            "file-openai-secret",
            "file-contextplus-secret",
            "inherited-openai-secret",
            "inherited-contextplus-secret",
        ] {
            assert!(!logs.contains(forbidden), "log leaked {forbidden}: {logs}");
        }
    }

    #[test]
    fn mismatch_warning_names_daemon_source_and_ignored_bridge_values() {
        let daemon = Config::from_env();
        let mut bridge = crate::transport::client::SearchConfig::from(&daemon);
        bridge.ollama_embed_model = "bridge-model".into();
        let differences = compare_search_config(&daemon, &bridge);
        let source = Path::new("/primary/repo/.mcp.json");

        let warning = config_warning(&differences, Some(source)).unwrap();

        assert!(warning.contains(&source.display().to_string()));
        assert!(warning.contains("OLLAMA_EMBED_MODEL"));
        assert!(warning.contains("bridge-model"));
        assert!(warning.contains("ignored"));
        assert!(!warning.contains("session with the right config"));
    }

    #[test]
    fn search_config_comparison_includes_provider_identity_without_secrets() {
        let mut daemon = Config::from_env();
        daemon.embed_provider = crate::config::EmbedProvider::Ollama;
        daemon.chat_provider = crate::config::ChatProvider::Ollama;
        let mut bridge_config = daemon.clone();
        bridge_config.embed_provider = crate::config::EmbedProvider::OpenAi;
        bridge_config.chat_provider = crate::config::ChatProvider::Claude;
        bridge_config.openai_embed_model = "bridge-embed".into();
        bridge_config.openai_base_url = "https://embed.example/v1".into();
        bridge_config.claude_model = "bridge-claude".into();
        bridge_config.claude_path = "/opt/claude".into();
        bridge_config.openai_api_key = Some("must-not-appear".into());

        let bridge = crate::transport::client::SearchConfig::from(&bridge_config);
        let differences = compare_search_config(&daemon, &bridge);
        let names: Vec<_> = differences
            .iter()
            .map(|difference| difference.field)
            .collect();
        let rendered = format!("{bridge:?} {differences:?}");

        assert!(names.contains(&"CONTEXTPLUS_EMBED_PROVIDER"));
        assert!(names.contains(&"CONTEXTPLUS_CHAT_PROVIDER"));
        assert!(names.contains(&"CONTEXTPLUS_OPENAI_EMBED_MODEL"));
        assert!(names.contains(&"CONTEXTPLUS_OPENAI_BASE_URL"));
        assert!(names.contains(&"CONTEXTPLUS_CLAUDE_MODEL"));
        assert!(names.contains(&"CONTEXTPLUS_CLAUDE_PATH"));
        assert!(!rendered.contains("must-not-appear"));
    }

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
