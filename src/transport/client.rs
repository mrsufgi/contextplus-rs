//! Stdio↔socket bridge. The MCP host (Claude Code, Cursor, …) spawns a
//! contextplus binary expecting it to speak MCP over stdin/stdout; in client
//! mode we just shovel JSON-RPC frames between our own stdio and a connected
//! per-workspace daemon.
//!
//! Boot sequence:
//!
//! ```text
//!  connect(.mcp_data/contextplus.sock)
//!         |
//!     +---+---+
//!     | ok    | ECONNREFUSED / ENOENT
//!     v       |
//!   bridge    v
//!           spawn daemon (self-fork with --daemon, setsid'd)
//!             v
//!         poll for socket up to SPAWN_TIMEOUT
//!             v
//!         connect → bridge
//! ```
//!
//! Host EOF terminates the bridge; daemon disconnects restore the MCP session.

use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, copy};
use tokio::net::UnixStream;

use crate::transport::{daemon, paths};

// ── Bridge↔Daemon framing ────────────────────────────────────────────────────
//
// Before forwarding raw MCP stdio the bridge sends a single length-prefixed
// JSON frame:
//   [u32 big-endian length][JSON bytes]
//
// The daemon replies with the same framing (session_ready or rejected_draining).
// After that exchange the stream reverts to plain stdio passthrough.

/// Message the bridge sends immediately after connecting.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct RegisterSession {
    /// `--root-dir` as resolved by the bridge process.
    pub client_root: std::path::PathBuf,
    /// Output of `git rev-parse HEAD` in `client_root`. Empty string when
    /// the worktree has no commits yet.
    pub head_sha: String,
    /// PID of the bridge process (advisory; used for logging).
    pub client_pid: u32,
    /// Search configuration seen by the bridge. Optional for compatibility
    /// with bridges built before configuration reporting was added.
    #[serde(default)]
    pub search_config: Option<SearchConfig>,
}

/// Configuration that can change search results or embedding-cache contents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchConfig {
    #[serde(default)]
    pub embed_provider: String,
    #[serde(default)]
    pub chat_provider: String,
    pub ollama_embed_model: String,
    pub ollama_chat_model: String,
    pub ollama_host: String,
    #[serde(default)]
    pub openai_embed_model: String,
    #[serde(default)]
    pub openai_chat_model: String,
    #[serde(default)]
    pub openai_base_url: String,
    #[serde(default)]
    pub chat_base_url: Option<String>,
    #[serde(default)]
    pub claude_path: String,
    #[serde(default)]
    pub claude_model: String,
    #[serde(default)]
    pub anthropic_chat_model: String,
    pub embed_tracker_mode: String,
    pub ignore_dirs: Vec<String>,
    pub max_embed_file_size: usize,
    pub embed_num_gpu: Option<i32>,
    pub embed_main_gpu: Option<i32>,
    pub embed_num_thread: Option<i32>,
    pub embed_num_batch: Option<i32>,
    pub embed_num_ctx: Option<i32>,
    pub embed_low_vram: Option<bool>,
    pub embed_chunk_chars: usize,
    pub warmup_on_start: bool,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub ref_warmup_mode: String,
}

impl From<&crate::config::Config> for SearchConfig {
    fn from(config: &crate::config::Config) -> Self {
        let mut ignore_dirs: Vec<_> = config.ignore_dirs.iter().cloned().collect();
        ignore_dirs.sort();
        Self {
            embed_provider: config.embed_provider.to_string(),
            chat_provider: config.chat_provider.to_string(),
            ollama_embed_model: config.ollama_embed_model.clone(),
            ollama_chat_model: config.ollama_chat_model.clone(),
            ollama_host: config.ollama_host.clone(),
            openai_embed_model: config.openai_embed_model.clone(),
            openai_chat_model: config.openai_chat_model.clone(),
            openai_base_url: config.openai_base_url.clone(),
            chat_base_url: config.chat_base_url.clone(),
            claude_path: config.claude_path.clone(),
            claude_model: config.claude_model.clone(),
            anthropic_chat_model: config.anthropic_chat_model.clone(),
            embed_tracker_mode: config.embed_tracker_mode.to_string(),
            ignore_dirs,
            max_embed_file_size: config.max_embed_file_size,
            embed_num_gpu: config.embed_num_gpu,
            embed_main_gpu: config.embed_main_gpu,
            embed_num_thread: config.embed_num_thread,
            embed_num_batch: config.embed_num_batch,
            embed_num_ctx: config.embed_num_ctx,
            embed_low_vram: config.embed_low_vram,
            embed_chunk_chars: config.embed_chunk_chars,
            warmup_on_start: config.warmup_on_start,
            hnsw_ef_construction: config.hnsw_ef_construction,
            hnsw_ef_search: config.hnsw_ef_search,
            ref_warmup_mode: config.ref_warmup_mode.to_string(),
        }
    }
}

impl SearchConfig {
    pub(crate) fn reported_fields(&self) -> Vec<(&'static str, String)> {
        fn option<T: ToString>(value: Option<T>) -> String {
            value
                .map(|value| value.to_string())
                .unwrap_or_else(|| "<unset>".to_string())
        }

        vec![
            ("CONTEXTPLUS_EMBED_PROVIDER", self.embed_provider.clone()),
            ("CONTEXTPLUS_CHAT_PROVIDER", self.chat_provider.clone()),
            ("OLLAMA_EMBED_MODEL", self.ollama_embed_model.clone()),
            ("OLLAMA_CHAT_MODEL", self.ollama_chat_model.clone()),
            ("OLLAMA_HOST", self.ollama_host.clone()),
            (
                "CONTEXTPLUS_OPENAI_EMBED_MODEL",
                self.openai_embed_model.clone(),
            ),
            (
                "CONTEXTPLUS_OPENAI_CHAT_MODEL",
                self.openai_chat_model.clone(),
            ),
            ("CONTEXTPLUS_OPENAI_BASE_URL", self.openai_base_url.clone()),
            (
                "CONTEXTPLUS_CHAT_BASE_URL",
                self.chat_base_url
                    .clone()
                    .unwrap_or_else(|| "<unset>".to_string()),
            ),
            ("CONTEXTPLUS_CLAUDE_PATH", self.claude_path.clone()),
            ("CONTEXTPLUS_CLAUDE_MODEL", self.claude_model.clone()),
            (
                "CONTEXTPLUS_ANTHROPIC_CHAT_MODEL",
                self.anthropic_chat_model.clone(),
            ),
            ("CONTEXTPLUS_EMBED_TRACKER", self.embed_tracker_mode.clone()),
            ("CONTEXTPLUS_IGNORE_DIRS", self.ignore_dirs.join(",")),
            (
                "CONTEXTPLUS_MAX_EMBED_FILE_SIZE",
                self.max_embed_file_size.to_string(),
            ),
            ("CONTEXTPLUS_EMBED_NUM_GPU", option(self.embed_num_gpu)),
            ("CONTEXTPLUS_EMBED_MAIN_GPU", option(self.embed_main_gpu)),
            (
                "CONTEXTPLUS_EMBED_NUM_THREAD",
                option(self.embed_num_thread),
            ),
            ("CONTEXTPLUS_EMBED_NUM_BATCH", option(self.embed_num_batch)),
            ("CONTEXTPLUS_EMBED_NUM_CTX", option(self.embed_num_ctx)),
            ("CONTEXTPLUS_EMBED_LOW_VRAM", option(self.embed_low_vram)),
            (
                "CONTEXTPLUS_EMBED_CHUNK_CHARS",
                self.embed_chunk_chars.to_string(),
            ),
            (
                "CONTEXTPLUS_WARMUP_ON_START",
                self.warmup_on_start.to_string(),
            ),
            (
                "CONTEXTPLUS_HNSW_EF_CONSTRUCTION",
                self.hnsw_ef_construction.to_string(),
            ),
            (
                "CONTEXTPLUS_HNSW_EF_SEARCH",
                self.hnsw_ef_search.to_string(),
            ),
            ("CONTEXTPLUS_REF_WARMUP_MODE", self.ref_warmup_mode.clone()),
        ]
    }
}

#[cfg(test)]
mod register_session_tests {
    use super::*;

    #[derive(Deserialize)]
    struct LegacyRegisterSession {
        client_root: std::path::PathBuf,
        head_sha: String,
        client_pid: u32,
    }

    #[test]
    fn register_session_round_trips_search_config() {
        let config = crate::config::Config::from_env();
        let register = RegisterSession {
            client_root: "/tmp/project".into(),
            head_sha: "deadbeef".into(),
            client_pid: 42,
            search_config: Some(SearchConfig::from(&config)),
        };

        let encoded = serde_json::to_vec(&register).unwrap();
        let decoded: RegisterSession = serde_json::from_slice(&encoded).unwrap();
        let legacy: LegacyRegisterSession = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded, register);
        assert_eq!(legacy.client_root, register.client_root);
        assert_eq!(legacy.head_sha, register.head_sha);
        assert_eq!(legacy.client_pid, register.client_pid);
    }

    #[test]
    fn register_session_deserializes_old_frame_without_search_config() {
        let old_frame = br#"{
            "client_root":"/tmp/project",
            "head_sha":"deadbeef",
            "client_pid":42
        }"#;

        let decoded: RegisterSession = serde_json::from_slice(old_frame).unwrap();

        assert_eq!(
            decoded.client_root,
            std::path::PathBuf::from("/tmp/project")
        );
        assert_eq!(decoded.head_sha, "deadbeef");
        assert_eq!(decoded.client_pid, 42);
        assert_eq!(decoded.search_config, None);

        let reencoded = serde_json::to_vec(&decoded).unwrap();
        let round_tripped: RegisterSession = serde_json::from_slice(&reencoded).unwrap();
        assert_eq!(round_tripped, decoded);
    }
}

/// Response the daemon sends back.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status")]
pub enum SessionReady {
    /// Daemon accepted the session and assigned a ref.
    Ready { session_id: String, ref_id: u64 },
    /// Daemon is draining; reconnecting bridges should retry.
    RejectedDraining,
    /// Ref is being warmed (initial embedding); calls will succeed but may
    /// observe stale index until warming finishes.
    Warming {
        session_id: String,
        ref_id: u64,
        eta_ms: u64,
    },
}

/// Write a length-prefixed JSON frame onto `w`.
pub async fn write_frame<W, T>(w: &mut W, msg: &T) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
    T: Serialize,
{
    let payload = serde_json::to_vec(msg)?;
    let len = payload.len() as u32;
    w.write_all(&len.to_be_bytes()).await?;
    w.write_all(&payload).await?;
    w.flush().await?;
    Ok(())
}

/// Read a length-prefixed JSON frame from `r`.
pub async fn read_frame<R, T>(r: &mut R) -> Result<T>
where
    R: tokio::io::AsyncRead + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)
        .await
        .context("read frame length")?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)
        .await
        .context("read frame payload")?;
    let msg: T = serde_json::from_slice(&payload).context("deserialize frame")?;
    Ok(msg)
}

/// Resolve `git rev-parse HEAD` for a given directory. Returns an empty string
/// if the directory is not a git repo or has no commits yet.
pub fn resolve_head_sha(root_dir: &Path) -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root_dir)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_default()
}

/// Back-off when a live daemon's accept queue is momentarily full and we
/// receive a spurious `ECONNREFUSED`.
const BACKOFF_ON_LIVE_DAEMON: Duration = Duration::from_millis(50);

/// Maximum time to wait for a freshly-spawned daemon to bind its socket.
pub const SPAWN_TIMEOUT: Duration = Duration::from_secs(5);
/// Polling interval while waiting for the daemon socket to appear.
pub const SPAWN_POLL: Duration = Duration::from_millis(50);

/// Connect to the workspace daemon and restore the session after disconnects.
pub async fn run(root_dir: &Path) -> Result<()> {
    run_with_config(root_dir, &crate::config::Config::from_env()).await
}

/// Connect using an already-resolved bridge configuration.
pub async fn run_with_config(root_dir: &Path, config: &crate::config::Config) -> Result<()> {
    run_with_io(
        root_dir,
        config,
        tokio::io::stdin(),
        tokio::io::stdout(),
        BridgeOptions::default(),
    )
    .await
}

#[derive(Clone, Copy, Debug)]
pub struct BridgeOptions {
    pub reconnect_timeout: Duration,
    pub reconnect_backoff: Duration,
}

impl Default for BridgeOptions {
    fn default() -> Self {
        Self {
            reconnect_timeout: Duration::from_secs(60),
            reconnect_backoff: Duration::from_millis(50),
        }
    }
}

pub async fn run_with_io<R, W>(
    root_dir: &Path,
    config: &crate::config::Config,
    host_input: R,
    mut host_output: W,
    options: BridgeOptions,
) -> Result<()>
where
    R: tokio::io::AsyncRead + Unpin + Send,
    W: tokio::io::AsyncWrite + Unpin + Send,
{
    let (sender, mut pending) = tokio::sync::mpsc::unbounded_channel();
    let read_host = async {
        let mut input = tokio::io::BufReader::new(host_input).lines();
        while let Some(line) = input.next_line().await? {
            if sender.send(line).is_err() {
                break;
            }
        }
        Ok::<(), anyhow::Error>(())
    };
    let forward = async {
        let mut initialize: Option<String> = None;
        let mut initialized: Option<String> = None;
        let mut inflight = std::collections::BTreeMap::new();
        let mut reconnecting = false;
        loop {
            let restore = async {
                loop {
                    let attempt = async {
                        let mut stream = connect_or_spawn(root_dir).await?;
                        let reg = RegisterSession {
                            client_root: root_dir.to_path_buf(),
                            head_sha: resolve_head_sha(root_dir),
                            client_pid: std::process::id(),
                            search_config: Some(SearchConfig::from(config)),
                        };
                        write_frame(&mut stream, &reg).await?;
                        if read_frame::<_, SessionReady>(&mut stream).await?
                            == SessionReady::RejectedDraining
                        {
                            bail!("daemon is draining");
                        }
                        let mut stream = tokio::io::BufReader::new(stream);
                        if let Some(line) = &initialize {
                            send_host_line(stream.get_mut(), line).await?;
                            let id = serde_json::from_str::<serde_json::Value>(line)?["id"].clone();
                            loop {
                                let mut response = String::new();
                                if stream.read_line(&mut response).await? == 0 {
                                    bail!("daemon closed during initialization replay");
                                }
                                let value: serde_json::Value = serde_json::from_str(&response)?;
                                if value.get("id") == Some(&id) && value.get("method").is_none() {
                                    if value.get("error").is_some() {
                                        bail!("daemon rejected initialization replay");
                                    }
                                    break;
                                }
                                host_output.write_all(response.as_bytes()).await?;
                                host_output.flush().await?;
                            }
                        }
                        if let Some(line) = &initialized {
                            send_host_line(stream.get_mut(), line).await?;
                        }
                        if reconnecting {
                            tracing::info!("session re-registered after reconnect; initialization replay complete");
                        }
                        Ok::<_, anyhow::Error>(stream)
                    }
                    .await;
                    match attempt {
                        Ok(stream) => return Ok::<_, anyhow::Error>(stream),
                        Err(error) => tracing::debug!(%error, "daemon reconnect failed; retrying"),
                    }
                    tokio::time::sleep(options.reconnect_backoff).await;
                }
            };
            let stream = tokio::time::timeout(options.reconnect_timeout, restore)
                .await
                .context("daemon reconnect deadline exceeded")??;
            let (reader, mut writer) = tokio::io::split(stream);
            let mut responses = tokio::io::BufReader::new(reader).lines();
            let mut outgoing = Vec::new();
            let mut written = 0;
            loop {
                tokio::select! {
                    biased;
                    result = responses.next_line() => {
                        match result {
                            Ok(None) | Err(_) => break,
                            Ok(Some(response)) => {
                                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&response)
                                    && value.get("method").is_none()
                                    && let Some(id) = value.get("id") {
                                    inflight.remove(&id.to_string());
                                }
                                host_output.write_all(format!("{response}\n").as_bytes()).await?;
                                host_output.flush().await?;
                            }
                        }
                    }
                    result = writer.write(&outgoing[written..]), if written < outgoing.len() => {
                        match result {
                            Ok(0) | Err(_) => break,
                            Ok(count) => written += count,
                        }
                    }
                    Some(line) = pending.recv(), if written == outgoing.len() => {
                        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                            match value["method"].as_str() {
                                Some("initialize") if initialize.is_none() => initialize = Some(line.clone()),
                                Some("notifications/initialized") if initialized.is_none() => initialized = Some(line.clone()),
                                _ => {}
                            }
                            if value.get("method").is_some() && let Some(id) = value.get("id") {
                                inflight.insert(id.to_string(), id.clone());
                            }
                        }
                        let cwd = crate::core::client_cwd::parent_process_cwd();
                        outgoing = crate::core::client_cwd::inject_cwd(
                            format!("{line}\n").as_bytes(), cwd.as_deref(),
                        );
                        written = 0;
                    }
                }
            }
            reconnecting = true;
            tracing::info!("daemon connection lost; reconnecting session");
            for (_, id) in std::mem::take(&mut inflight) {
                let error = serde_json::json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": {"code": -32000, "message": "contextplus daemon restarted; retry the call"}
                });
                host_output
                    .write_all(format!("{error}\n").as_bytes())
                    .await?;
            }
            host_output.flush().await?;
        }
    };
    tokio::select! {
        biased;
        result = read_host => result,
        result = forward => result,
    }
}

async fn send_host_line<W: tokio::io::AsyncWrite + Unpin>(
    writer: &mut W,
    line: &str,
) -> Result<()> {
    let cwd = crate::core::client_cwd::parent_process_cwd();
    let line = format!("{line}\n");
    let out = crate::core::client_cwd::inject_cwd(line.as_bytes(), cwd.as_deref());
    writer.write_all(&out).await?;
    writer.flush().await?;
    Ok(())
}

/// Perform the register_session handshake and then bridge stdio. Exposed for
/// tests so they can inject a pre-connected stream.
pub async fn run_with_handshake(root_dir: &Path, stream: UnixStream) -> Result<()> {
    run_with_handshake_config(root_dir, &crate::config::Config::from_env(), stream).await
}

/// Perform the handshake using an already-resolved bridge configuration.
pub async fn run_with_handshake_config(
    root_dir: &Path,
    config: &crate::config::Config,
    mut stream: UnixStream,
) -> Result<()> {
    let head_sha = resolve_head_sha(root_dir);
    let reg = RegisterSession {
        client_root: root_dir.to_path_buf(),
        head_sha,
        client_pid: std::process::id(),
        search_config: Some(SearchConfig::from(config)),
    };
    write_frame(&mut stream, &reg)
        .await
        .context("send register_session")?;

    let reply: SessionReady = read_frame(&mut stream)
        .await
        .context("read session_ready")?;

    match reply {
        SessionReady::RejectedDraining => {
            tracing::info!("daemon is draining — bridge exiting cleanly");
            return Ok(());
        }
        SessionReady::Ready { session_id, ref_id } => {
            tracing::debug!(%session_id, ref_id, "session registered with daemon");
        }
        SessionReady::Warming {
            session_id,
            ref_id,
            eta_ms,
        } => {
            tracing::debug!(
                %session_id,
                ref_id,
                eta_ms,
                "daemon accepted session — ref is warming"
            );
        }
    }

    bridge(stream).await
}

/// Try to connect to the daemon socket; if it isn't there or has gone stale,
/// spawn a daemon and wait for it to come up.
pub async fn connect_or_spawn(root_dir: &Path) -> Result<UnixStream> {
    let socket = paths::daemon_socket_path(root_dir);

    match UnixStream::connect(&socket).await {
        Ok(s) => return Ok(s),
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
            ) =>
        {
            if e.kind() == std::io::ErrorKind::ConnectionRefused {
                // Before unlinking the socket, probe the daemon lock to
                // distinguish a truly dead daemon from a live one whose accept
                // queue is momentarily full (ECONNREFUSED under high load).
                let lock_path = paths::daemon_lock_path(root_dir);
                match daemon::probe_lock_held(root_dir) {
                    Ok(true) => {
                        // A daemon IS alive — the ECONNREFUSED was spurious
                        // (full accept queue). Do NOT unlink the socket; just
                        // back off and retry once.
                        tracing::debug!(
                            "spurious ECONNREFUSED at {} — daemon lock held, retrying after backoff",
                            socket.display()
                        );
                        tokio::time::sleep(BACKOFF_ON_LIVE_DAEMON).await;
                        return UnixStream::connect(&socket).await.with_context(|| {
                            format!(
                                "connect retry after spurious ECONNREFUSED at {}",
                                socket.display()
                            )
                        });
                    }
                    Ok(false) => {
                        // No daemon. Safe to remove the stale socket and spawn.
                        tracing::debug!(
                            "daemon lock is free — removing stale socket {}",
                            socket.display()
                        );
                        let _ = std::fs::remove_file(&socket);
                    }
                    Err(e) => {
                        // Could not probe the lock — err on the side of caution:
                        // do not unlink. Log and fall through to spawn attempt
                        // (spawn will fail to bind but that surfaces a clear error).
                        tracing::warn!(
                            "could not probe daemon lock at {}: {e} — skipping socket removal",
                            lock_path.display()
                        );
                    }
                }
            }
            tracing::debug!("no daemon at {} — spawning", socket.display());
        }
        Err(e) => {
            return Err(e).with_context(|| format!("connect({}) failed", socket.display()));
        }
    }

    spawn_daemon(root_dir)?;

    // Poll for socket appearance.
    let deadline = Instant::now() + SPAWN_TIMEOUT;
    loop {
        if Instant::now() >= deadline {
            bail!(
                "timed out after {:?} waiting for daemon socket at {}",
                SPAWN_TIMEOUT,
                socket.display(),
            );
        }
        match UnixStream::connect(&socket).await {
            Ok(s) => return Ok(s),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
                ) =>
            {
                tokio::time::sleep(SPAWN_POLL).await;
            }
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("post-spawn connect({}) failed", socket.display()));
            }
        }
    }
}

/// Re-exec ourselves with `--daemon` flag and detach via `setsid` so the
/// child survives client (Claude Code) termination.
/// The binary to launch as the daemon. After an in-place upgrade Linux reports
/// our own executable as "<path> (deleted)"; launch the new file at that path.
fn daemon_executable(current: std::path::PathBuf) -> std::path::PathBuf {
    match current.to_str().and_then(|p| p.strip_suffix(" (deleted)")) {
        Some(replaced) if Path::new(replaced).is_file() => std::path::PathBuf::from(replaced),
        _ => current,
    }
}

fn spawn_daemon(root_dir: &Path) -> Result<()> {
    let exe = daemon_executable(std::env::current_exe().context("current_exe() failed")?);
    let log_path = std::env::var_os("CONTEXTPLUS_DAEMON_LOG")
        .filter(|path| !path.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| daemon::daemon_log_path(root_dir));
    let log = daemon::open_log_file(&log_path)?;
    let mut cmd = daemon_spawn_command(&exe, root_dir, &log_path, log);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn daemon: {}", exe.display()))?;

    tracing::debug!("spawned daemon pid={}", child.id());
    // Reap the detached daemon without blocking bridge shutdown.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

fn daemon_spawn_command(
    exe: &Path,
    root_dir: &Path,
    log_path: &Path,
    log: std::fs::File,
) -> std::process::Command {
    let primary_root = crate::core::git_worktree::resolve_primary_worktree(root_dir);
    let mut cmd = std::process::Command::new(exe);
    cmd.env("CONTEXTPLUS_DAEMON_LOG", log_path);
    cmd.arg("--root-dir")
        .arg(primary_root)
        .arg("daemon")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log));

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: `setsid` is async-signal-safe and sets the child's session
        // id so it doesn't share the parent's controlling terminal. This is
        // the standard daemonization trick — the child is reparented to PID 1
        // (or the session leader) once the parent exits.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    cmd
}

/// Pump bytes between our stdio and the daemon socket. Returns when either
/// half closes (EOF on stdin, or daemon disconnect).
///
/// Host → daemon traffic is forwarded line by line so every `tools/call`
/// can carry the host's current working directory (see
/// [`crate::core::client_cwd`]); the daemon uses it to run the call against
/// the worktree the agent is in.
pub async fn bridge(stream: UnixStream) -> Result<()> {
    let (mut sock_r, mut sock_w) = stream.into_split();
    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());
    let mut stdout = tokio::io::stdout();

    let to_daemon = async move {
        let mut total = 0u64;
        let mut line = Vec::new();
        loop {
            line.clear();
            if stdin.read_until(b'\n', &mut line).await? == 0 {
                break;
            }
            let cwd = crate::core::client_cwd::parent_process_cwd();
            let out = crate::core::client_cwd::inject_cwd(&line, cwd.as_deref());
            sock_w.write_all(&out).await?;
            total += out.len() as u64;
        }
        // Half-close so the daemon sees EOF on its read side.
        let _ = sock_w.shutdown().await;
        Ok::<u64, std::io::Error>(total)
    };

    let to_stdout = async move {
        let n = copy(&mut sock_r, &mut stdout).await?;
        let _ = stdout.flush().await;
        Ok::<u64, std::io::Error>(n)
    };

    // First side to finish ends the session — the MCP host is exiting or the
    // daemon disconnected.
    tokio::select! {
        r = to_daemon => {
            r.context("client→daemon copy failed")?;
        }
        r = to_stdout => {
            r.context("daemon→client copy failed")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::path::PathBuf;
    use tokio::io::{AsyncBufReadExt, BufReader, DuplexStream};
    use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
    use tokio::sync::oneshot;

    struct FakeSession {
        reader: BufReader<OwnedReadHalf>,
        writer: OwnedWriteHalf,
    }

    impl FakeSession {
        async fn read_json(&mut self) -> Value {
            let mut line = String::new();
            let read = self.reader.read_line(&mut line).await.unwrap();
            assert_ne!(read, 0, "fake daemon expected another JSON-RPC line");
            serde_json::from_str(&line).unwrap()
        }

        async fn write_json(&mut self, value: Value) {
            let mut line = serde_json::to_vec(&value).unwrap();
            line.push(b'\n');
            self.writer.write_all(&line).await.unwrap();
            self.writer.flush().await.unwrap();
        }
    }

    struct FakeHost {
        input: DuplexStream,
        output: BufReader<DuplexStream>,
    }

    impl FakeHost {
        async fn write_json(&mut self, value: Value) {
            let mut line = serde_json::to_vec(&value).unwrap();
            line.push(b'\n');
            self.input.write_all(&line).await.unwrap();
            self.input.flush().await.unwrap();
        }

        async fn read_json(&mut self) -> Value {
            let mut line = String::new();
            let read = self.output.read_line(&mut line).await.unwrap();
            assert_ne!(read, 0, "host expected another JSON-RPC line");
            serde_json::from_str(&line).unwrap()
        }
    }

    fn host_io() -> (DuplexStream, DuplexStream, FakeHost) {
        let (host_input, bridge_input) = tokio::io::duplex(16 * 1024);
        let (bridge_output, host_output) = tokio::io::duplex(16 * 1024);
        (
            bridge_input,
            bridge_output,
            FakeHost {
                input: host_input,
                output: BufReader::new(host_output),
            },
        )
    }

    async fn accept_session(
        listener: &tokio::net::UnixListener,
        reply: SessionReady,
    ) -> FakeSession {
        let (mut stream, _) = listener.accept().await.unwrap();
        let _: RegisterSession = read_frame(&mut stream).await.unwrap();
        write_frame(&mut stream, &reply).await.unwrap();
        let (reader, writer) = stream.into_split();
        FakeSession {
            reader: BufReader::new(reader),
            writer,
        }
    }

    fn ready(name: &str) -> SessionReady {
        SessionReady::Ready {
            session_id: name.into(),
            ref_id: 1,
        }
    }

    fn short_reconnect_policy() -> BridgeOptions {
        BridgeOptions {
            reconnect_timeout: Duration::from_millis(250),
            reconnect_backoff: Duration::from_millis(10),
        }
    }

    fn assert_cwd_injected(message: &Value) {
        // parent_process_cwd reads /proc, so only Linux injects a cwd.
        if !cfg!(target_os = "linux") {
            return;
        }
        assert!(
            message["params"]["arguments"][crate::core::client_cwd::CWD_ARG]
                .as_str()
                .is_some(),
            "tools/call did not contain injected cwd: {message}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn bridge_child_process_helper() {
        let Some(root_dir) = std::env::var_os("CONTEXTPLUS_TEST_BRIDGE_CHILD_ROOT") else {
            return;
        };
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime
            .block_on(run_with_io(
                Path::new(&root_dir),
                &crate::config::Config::from_env(),
                tokio::io::stdin(),
                tokio::io::stdout(),
                BridgeOptions::default(),
            ))
            .expect("bridge child should stop cleanly when stdin closes");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn host_stdin_close_exits_bridge_process_zero_promptly() {
        use std::process::{Command, Stdio};
        use tokio::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let socket_path = paths::daemon_socket_path(&root);
        std::fs::create_dir_all(socket_path.parent().unwrap()).unwrap();
        let listener = UnixListener::bind(&socket_path).unwrap();

        let daemon = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _: RegisterSession = read_frame(&mut stream).await.unwrap();
            write_frame(
                &mut stream,
                &SessionReady::Ready {
                    session_id: "stdin-close".into(),
                    ref_id: 1,
                },
            )
            .await
            .unwrap();
            tokio::time::sleep(Duration::from_secs(2)).await;
        });

        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "transport::client::tests::bridge_child_process_helper",
                "--nocapture",
            ])
            .env("CONTEXTPLUS_TEST_BRIDGE_CHILD_ROOT", root)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();

        tokio::time::timeout(Duration::from_secs(1), async {
            while !socket_path.exists() {
                tokio::task::yield_now().await;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(child.stdin.take());
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "bridge child exited with {status}");
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            let _ = child.kill();
            let _ = child.wait();
            panic!("bridge child did not exit within 1s after host stdin closed")
        });

        daemon.abort();
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn reconnect_replays_session_fails_inflight_and_flushes_buffered_request() {
        use tokio::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let socket_path = paths::daemon_socket_path(&root);
        std::fs::create_dir_all(socket_path.parent().unwrap()).unwrap();
        let listener = UnixListener::bind(&socket_path).unwrap();
        let replacement_path = socket_path.clone();
        let (dropped_tx, dropped_rx) = oneshot::channel();
        let (second_registered_tx, second_registered_rx) = oneshot::channel();
        let (allow_second_tx, allow_second_rx) = oneshot::channel();

        let daemon = tokio::spawn(async move {
            let mut first = accept_session(&listener, ready("daemon-a")).await;
            let initialize = first.read_json().await;
            assert_eq!(initialize["method"], "initialize");
            first
                .write_json(json!({"jsonrpc":"2.0","id":1,"result":{"serverInfo":{"name":"a"}}}))
                .await;

            let initialized = first.read_json().await;
            assert_eq!(initialized["method"], "notifications/initialized");

            let completed = first.read_json().await;
            assert_eq!(completed["id"], 2);
            assert_cwd_injected(&completed);
            first
                .write_json(json!({"jsonrpc":"2.0","id":2,"result":{"ok":"a"}}))
                .await;

            let interrupted = first.read_json().await;
            assert_eq!(interrupted["id"], 3);
            assert_cwd_injected(&interrupted);
            drop(listener);
            std::fs::remove_file(&replacement_path).unwrap();
            let second_listener = UnixListener::bind(&replacement_path).unwrap();
            drop(first);
            dropped_tx.send(()).unwrap();

            let (mut stream, _) = second_listener.accept().await.unwrap();
            let _: RegisterSession = read_frame(&mut stream).await.unwrap();
            second_registered_tx.send(()).unwrap();
            allow_second_rx.await.unwrap();
            write_frame(&mut stream, &ready("daemon-b")).await.unwrap();
            let (reader, writer) = stream.into_split();
            let mut second = FakeSession {
                reader: BufReader::new(reader),
                writer,
            };

            let replayed_initialize = second.read_json().await;
            assert_eq!(replayed_initialize, initialize);
            second
                .write_json(json!({"jsonrpc":"2.0","id":1,"result":{"serverInfo":{"name":"b"}}}))
                .await;
            let replayed_initialized = second.read_json().await;
            assert_eq!(replayed_initialized, initialized);

            let buffered = second.read_json().await;
            assert_eq!(buffered["id"], 4);
            assert_cwd_injected(&buffered);
            second
                .write_json(json!({"jsonrpc":"2.0","id":4,"result":{"ok":"b"}}))
                .await;
        });

        let (bridge_input, bridge_output, mut host) = host_io();
        let config = crate::config::Config::from_env();
        let bridge_root = root.clone();
        let bridge = tokio::spawn(async move {
            run_with_io(
                &bridge_root,
                &config,
                bridge_input,
                bridge_output,
                short_reconnect_policy(),
            )
            .await
        });

        host.write_json(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}))
            .await;
        let initialize_response = host.read_json().await;
        assert_eq!(initialize_response["id"], 1);
        host.write_json(json!({"jsonrpc":"2.0","method":"notifications/initialized"}))
            .await;
        host.write_json(json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"first","arguments":{}}}))
            .await;
        assert_eq!(host.read_json().await["id"], 2);

        host.write_json(json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"interrupted","arguments":{}}}))
            .await;
        dropped_rx.await.unwrap();
        second_registered_rx.await.unwrap();
        host.write_json(json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"buffered","arguments":{}}}))
            .await;
        allow_second_tx.send(()).unwrap();

        let mut responses = vec![initialize_response];
        responses.push(host.read_json().await);
        responses.push(host.read_json().await);
        let interrupted = responses.iter().find(|value| value["id"] == 3).unwrap();
        assert_eq!(interrupted["error"]["code"], -32000);
        assert_eq!(
            interrupted["error"]["message"],
            "contextplus daemon restarted; retry the call"
        );
        assert!(
            responses
                .iter()
                .any(|value| value["id"] == 4 && value["result"]["ok"] == "b")
        );
        assert_eq!(
            responses.iter().filter(|value| value["id"] == 1).count(),
            1,
            "host received a duplicate initialize response"
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(50), host.read_json())
                .await
                .is_err(),
            "host received an unexpected extra response"
        );

        drop(host.input);
        tokio::time::timeout(Duration::from_secs(1), bridge)
            .await
            .expect("bridge did not stop after host input closed")
            .unwrap()
            .unwrap();
        daemon.await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn reconnect_timeout_returns_error_while_host_input_is_open() {
        use tokio::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let socket_path = paths::daemon_socket_path(&root);
        std::fs::create_dir_all(socket_path.parent().unwrap()).unwrap();
        let listener = UnixListener::bind(&socket_path).unwrap();

        let daemon = tokio::spawn(async move {
            let mut first = accept_session(&listener, ready("daemon-a")).await;
            assert_eq!(first.read_json().await["method"], "initialize");
            first
                .write_json(json!({"jsonrpc":"2.0","id":1,"result":{}}))
                .await;
            assert_eq!(first.read_json().await["id"], 9);
            drop(first);

            let (mut reconnect, _) = listener.accept().await.unwrap();
            let _: RegisterSession = read_frame(&mut reconnect).await.unwrap();
            tokio::time::sleep(Duration::from_secs(2)).await;
        });

        let (bridge_input, bridge_output, mut host) = host_io();
        let config = crate::config::Config::from_env();
        let started = Instant::now();
        let bridge_root = root.clone();
        let bridge = tokio::spawn(async move {
            run_with_io(
                &bridge_root,
                &config,
                bridge_input,
                bridge_output,
                BridgeOptions {
                    reconnect_timeout: Duration::from_millis(100),
                    reconnect_backoff: Duration::from_millis(10),
                },
            )
            .await
        });

        host.write_json(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}))
            .await;
        assert_eq!(host.read_json().await["id"], 1);
        host.write_json(json!({"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"never-finishes","arguments":{}}}))
            .await;

        let result = tokio::time::timeout(Duration::from_secs(1), bridge)
            .await
            .expect("bridge exceeded its reconnect deadline")
            .unwrap();
        assert!(
            result.is_err(),
            "bridge returned success after reconnect timeout"
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(host.input);
        daemon.abort();
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn daemon_responses_progress_while_large_host_request_is_still_writing() {
        use tokio::net::UnixListener;

        const PAYLOAD_SIZE: usize = 4 * 1024 * 1024;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let socket_path = paths::daemon_socket_path(&root);
        std::fs::create_dir_all(socket_path.parent().unwrap()).unwrap();
        let listener = UnixListener::bind(&socket_path).unwrap();
        let (prior_request_tx, prior_request_rx) = oneshot::channel();
        let (request_received_tx, request_received_rx) = oneshot::channel();
        let (close_daemon_tx, close_daemon_rx) = oneshot::channel();

        let daemon = tokio::spawn(async move {
            let mut session = accept_session(&listener, ready("backpressure")).await;
            let prior_request = session.read_json().await;
            assert_eq!(prior_request["id"], 41);
            prior_request_tx.send(()).unwrap();

            let mut first_byte = [0_u8; 1];
            session.reader.read_exact(&mut first_byte).await.unwrap();

            session
                .write_json(json!({
                    "jsonrpc": "2.0",
                    "id": 41,
                    "result": {"payload": "r".repeat(PAYLOAD_SIZE)}
                }))
                .await;

            let mut request_tail = String::new();
            session.reader.read_line(&mut request_tail).await.unwrap();
            let request: Value =
                serde_json::from_slice(&[first_byte.as_slice(), request_tail.as_bytes()].concat())
                    .unwrap();
            assert_eq!(request["id"], 42);
            assert_eq!(
                request["params"]["payload"].as_str().unwrap().len(),
                PAYLOAD_SIZE
            );
            session
                .write_json(json!({"jsonrpc": "2.0", "id": 42, "result": {"received": true}}))
                .await;
            request_received_tx.send(()).unwrap();
            close_daemon_rx.await.unwrap();
        });

        let (bridge_input, bridge_output, mut host) = host_io();
        let config = crate::config::Config::from_env();
        let bridge_root = root.clone();
        let bridge = tokio::spawn(async move {
            run_with_io(
                &bridge_root,
                &config,
                bridge_input,
                bridge_output,
                short_reconnect_policy(),
            )
            .await
        });

        host.write_json(json!({
            "jsonrpc": "2.0",
            "id": 41,
            "method": "test/prior-request",
            "params": {}
        }))
        .await;
        prior_request_rx.await.unwrap();
        host.write_json(json!({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "test/large-request",
            "params": {"payload": "q".repeat(PAYLOAD_SIZE)}
        }))
        .await;

        let response = tokio::time::timeout(Duration::from_secs(5), host.read_json())
            .await
            .expect("daemon response stalled behind the host-to-daemon write");
        assert_eq!(response["id"], 41);
        assert_eq!(
            response["result"]["payload"].as_str().unwrap().len(),
            PAYLOAD_SIZE
        );
        tokio::time::timeout(Duration::from_secs(5), request_received_rx)
            .await
            .expect("daemon did not receive the complete host request")
            .unwrap();
        let request_response = tokio::time::timeout(Duration::from_secs(1), host.read_json())
            .await
            .expect("host did not receive the large request's response");
        assert_eq!(request_response["id"], 42);
        assert_eq!(request_response["result"]["received"], true);

        drop(host.input);
        tokio::time::timeout(Duration::from_secs(1), bridge)
            .await
            .expect("bridge did not stop after host EOF")
            .unwrap()
            .unwrap();
        close_daemon_tx.send(()).unwrap();
        daemon.await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn rejected_draining_is_retried_until_ready_daemon_resumes_session() {
        use tokio::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let socket_path = paths::daemon_socket_path(&root);
        std::fs::create_dir_all(socket_path.parent().unwrap()).unwrap();
        let listener = UnixListener::bind(&socket_path).unwrap();
        let (dropped_tx, dropped_rx) = oneshot::channel();

        let daemon = tokio::spawn(async move {
            let mut first = accept_session(&listener, ready("daemon-a")).await;
            let initialize = first.read_json().await;
            first
                .write_json(json!({"jsonrpc":"2.0","id":1,"result":{"daemon":"a"}}))
                .await;
            let initialized = first.read_json().await;
            drop(first);

            let _draining = accept_session(&listener, SessionReady::RejectedDraining).await;
            // The bridge is reconnecting now, so the next host request is buffered, not in flight.
            dropped_tx.send(()).unwrap();
            let mut ready_daemon = accept_session(&listener, ready("daemon-c")).await;
            assert_eq!(ready_daemon.read_json().await, initialize);
            ready_daemon
                .write_json(json!({"jsonrpc":"2.0","id":1,"result":{"daemon":"c"}}))
                .await;
            assert_eq!(ready_daemon.read_json().await, initialized);
            let resumed = ready_daemon.read_json().await;
            assert_eq!(resumed["id"], 5);
            assert_cwd_injected(&resumed);
            ready_daemon
                .write_json(json!({"jsonrpc":"2.0","id":5,"result":{"daemon":"c"}}))
                .await;
        });

        let (bridge_input, bridge_output, mut host) = host_io();
        let config = crate::config::Config::from_env();
        let bridge_root = root.clone();
        let bridge = tokio::spawn(async move {
            run_with_io(
                &bridge_root,
                &config,
                bridge_input,
                bridge_output,
                short_reconnect_policy(),
            )
            .await
        });

        host.write_json(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}))
            .await;
        let initialize_response = host.read_json().await;
        assert_eq!(initialize_response["result"]["daemon"], "a");
        host.write_json(json!({"jsonrpc":"2.0","method":"notifications/initialized"}))
            .await;
        dropped_rx.await.unwrap();
        host.write_json(json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"after-drain","arguments":{}}}))
            .await;
        let resumed = tokio::time::timeout(Duration::from_secs(1), host.read_json())
            .await
            .expect("session did not resume after RejectedDraining");
        assert_eq!(resumed["id"], 5);
        assert_eq!(resumed["result"]["daemon"], "c");

        drop(host.input);
        tokio::time::timeout(Duration::from_secs(1), bridge)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        daemon.await.unwrap();
    }

    #[test]
    fn daemon_executable_follows_an_upgraded_binary() {
        let dir = tempfile::tempdir().unwrap();
        let installed = dir.path().join("contextplus-rs");
        std::fs::write(&installed, b"new build").unwrap();
        let deleted = std::path::PathBuf::from(format!("{} (deleted)", installed.display()));
        assert_eq!(daemon_executable(deleted), installed);

        let missing =
            std::path::PathBuf::from(format!("{} (deleted)", dir.path().join("gone").display()));
        assert_eq!(daemon_executable(missing.clone()), missing);
        assert_eq!(daemon_executable(installed.clone()), installed);
    }

    #[test]
    fn daemon_spawn_command_passes_primary_root_for_linked_worktree() {
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
        let log_path = temp.path().join("daemon.log");
        let log = std::fs::File::create(&log_path).unwrap();

        let command =
            daemon_spawn_command(Path::new("/opt/contextplus-rs"), &linked, &log_path, log);
        let args: Vec<_> = command.get_args().collect();

        assert_eq!(args[0], std::ffi::OsStr::new("--root-dir"));
        assert_eq!(
            PathBuf::from(args[1]),
            primary.canonicalize().unwrap(),
            "spawned daemon must receive the primary checkout as --root-dir"
        );
        assert_eq!(args[2], std::ffi::OsStr::new("daemon"));
    }

    #[tokio::test]
    async fn connect_to_missing_socket_yields_error_kind() {
        let dir = tempfile::tempdir().unwrap();
        // Point straight at a non-existent socket without spawning a daemon —
        // we just want to verify the error mapping.
        let path = paths::daemon_socket_path(dir.path());
        let res = UnixStream::connect(&path).await;
        match res {
            Err(e) => assert!(matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
            )),
            Ok(_) => panic!("nothing should be listening"),
        }
    }

    /// Verify that `bridge` correctly copies bytes between two UnixStream
    /// endpoints (socketpair-style). We create a listener, connect, then let
    /// `bridge` move data from one side to the other.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn bridge_pumps_bytes_from_socket_to_socket() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("bridge_unit.sock");

        let listener = UnixListener::bind(&sock_path).unwrap();
        // Server side: write some bytes then close.
        let server_task = tokio::spawn(async move {
            let (mut conn, _) = listener.accept().await.unwrap();
            conn.write_all(b"hello").await.unwrap();
            drop(conn); // EOF to client
        });

        let client_stream = UnixStream::connect(&sock_path).await.unwrap();
        let (mut sock_r, sock_w) = client_stream.into_split();

        // Drop write side immediately — we're only testing the read direction.
        drop(sock_w);

        let mut buf = Vec::new();
        sock_r.read_to_end(&mut buf).await.unwrap();
        assert_eq!(buf, b"hello");

        let _ = server_task.await;
    }

    /// `connect_or_spawn` against a live daemon (socket already exists)
    /// should return Ok immediately without attempting to spawn.
    ///
    /// We bind the listener at the exact path `paths::daemon_socket_path`
    /// returns for the temp root so no env-var override is needed, avoiding
    /// any race with other tests that also read `SOCKET_PATH_ENV`.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn connect_or_spawn_connects_to_live_socket() {
        use tokio::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // Ensure no SOCKET_PATH_ENV leak from another test affects path resolution.
        // Compute the default socket path for this root (env must be unset for
        // this to be deterministic — we rely on the test harness not setting it).
        let sock_path = {
            // Temporarily ensure env is clear for path computation.
            let saved = std::env::var_os(paths::SOCKET_PATH_ENV);
            unsafe { std::env::remove_var(paths::SOCKET_PATH_ENV) };
            let p = paths::daemon_socket_path(root);
            if let Some(v) = saved {
                unsafe { std::env::set_var(paths::SOCKET_PATH_ENV, v) };
            }
            p
        };

        // Create the .mcp_data directory so bind succeeds.
        if let Some(parent) = sock_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }

        // Stand up a trivial listener at the computed path.
        let listener = UnixListener::bind(&sock_path).unwrap();
        let _accept_task = tokio::spawn(async move {
            let _ = listener.accept().await;
        });

        // connect_or_spawn should see the live socket and connect directly,
        // with no env override needed.
        let result = connect_or_spawn(root).await;

        assert!(
            result.is_ok(),
            "connect_or_spawn should succeed against a live socket: {result:?}"
        );
    }
}
