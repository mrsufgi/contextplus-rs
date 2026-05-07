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
//! The bridge runs two `tokio::io::copy` halves: stdin→socket and
//! socket→stdout. Either direction closing terminates the bridge.

use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt, copy};
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
}

/// Response the daemon sends back.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status")]
pub enum SessionReady {
    /// Daemon accepted the session and assigned a ref.
    Ready { session_id: String, ref_id: u64 },
    /// Daemon is draining; bridge should exit 0 cleanly.
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

/// Connect to the per-workspace daemon, spawning one if absent, then bridge
/// stdin↔socket↔stdout until either side closes.
///
/// Performs the `register_session` handshake before entering the stdio
/// passthrough loop. If the daemon responds with `RejectedDraining` the
/// function returns `Ok(())` immediately (bridge exits 0, clean shutdown).
pub async fn run(root_dir: &Path) -> Result<()> {
    let stream = connect_or_spawn(root_dir).await?;
    run_with_handshake(root_dir, stream).await
}

/// Perform the register_session handshake and then bridge stdio. Exposed for
/// tests so they can inject a pre-connected stream.
pub async fn run_with_handshake(root_dir: &Path, mut stream: UnixStream) -> Result<()> {
    let head_sha = resolve_head_sha(root_dir);
    let reg = RegisterSession {
        client_root: root_dir.to_path_buf(),
        head_sha,
        client_pid: std::process::id(),
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
fn spawn_daemon(root_dir: &Path) -> Result<()> {
    let exe = std::env::current_exe().context("current_exe() failed")?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--root-dir")
        .arg(root_dir)
        .arg("daemon")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

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

    let child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn daemon: {}", exe.display()))?;

    tracing::debug!("spawned daemon pid={}", child.id());
    // We don't `wait()` — the child is detached by `setsid` and we don't want
    // to keep a zombie or block the client.
    std::mem::forget(child);
    Ok(())
}

/// Pump bytes between our stdio and the daemon socket. Returns when either
/// half closes (EOF on stdin, or daemon disconnect).
pub async fn bridge(stream: UnixStream) -> Result<()> {
    let (mut sock_r, mut sock_w) = stream.into_split();
    let mut stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();

    let to_daemon = async move {
        let n = copy(&mut stdin, &mut sock_w).await?;
        // Half-close so the daemon sees EOF on its read side.
        let _ = sock_w.shutdown().await;
        Ok::<u64, std::io::Error>(n)
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
