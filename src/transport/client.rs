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
use tokio::io::{AsyncWriteExt, copy};
use tokio::net::UnixStream;

use crate::transport::paths;

/// Maximum time to wait for a freshly-spawned daemon to bind its socket.
pub const SPAWN_TIMEOUT: Duration = Duration::from_secs(5);
/// Polling interval while waiting for the daemon socket to appear.
pub const SPAWN_POLL: Duration = Duration::from_millis(50);

/// Connect to the per-workspace daemon, spawning one if absent, then bridge
/// stdin↔socket↔stdout until either side closes.
pub async fn run(root_dir: &Path) -> Result<()> {
    let stream = connect_or_spawn(root_dir).await?;
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
            // Stale socket file (daemon crashed) or never started — clean up
            // and respawn. Removal is best-effort; a missing file is fine.
            if e.kind() == std::io::ErrorKind::ConnectionRefused {
                let _ = std::fs::remove_file(&socket);
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
}
