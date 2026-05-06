//! On-disk path layout for the per-workspace daemon.
//!
//! ```text
//! <root>/.mcp_data/
//!   contextplus.daemon.lock   advisory fd-lock — single-instance enforcement
//!   contextplus.daemon.pid    PID of the live daemon (best-effort, advisory)
//!   contextplus.sock          Unix-domain socket clients connect to
//! ```
//!
//! The socket path can be overridden with `CONTEXTPLUS_DAEMON_SOCKET_PATH` for
//! callers that need to escape the workspace `.mcp_data/` (e.g. workspaces
//! mounted read-only). When overridden, the lock + pid files live in the same
//! parent directory as the socket.

use std::path::{Path, PathBuf};

/// Subdirectory inside the workspace root where contextplus stores caches +
/// daemon coordination files. Mirrors `.mcp_data` used by the cache layer.
pub const MCP_DATA_DIR: &str = ".mcp_data";

const DAEMON_LOCK_FILE: &str = "contextplus.daemon.lock";
const DAEMON_PID_FILE: &str = "contextplus.daemon.pid";
const DAEMON_SOCKET_FILE: &str = "contextplus.sock";

/// Override env var for the socket path. When set, `socket_path` returns the
/// override verbatim and lock/pid live next to it.
pub const SOCKET_PATH_ENV: &str = "CONTEXTPLUS_DAEMON_SOCKET_PATH";

/// Resolve the directory holding daemon coordination files for `root_dir`.
///
/// If `CONTEXTPLUS_DAEMON_SOCKET_PATH` is set, returns its parent; otherwise
/// `<root>/.mcp_data`.
pub fn daemon_dir(root_dir: &Path) -> PathBuf {
    if let Some(parent) = socket_override().and_then(|p| p.parent().map(Path::to_path_buf)) {
        return parent;
    }
    root_dir.join(MCP_DATA_DIR)
}

/// Path to the advisory lock file enforcing single-instance per workspace.
pub fn daemon_lock_path(root_dir: &Path) -> PathBuf {
    daemon_dir(root_dir).join(DAEMON_LOCK_FILE)
}

/// Path to the PID file written by the daemon. Best-effort — never relied on
/// for correctness; the lock file is the source of truth.
pub fn daemon_pid_path(root_dir: &Path) -> PathBuf {
    daemon_dir(root_dir).join(DAEMON_PID_FILE)
}

/// Path to the Unix-domain socket clients connect to. Honors
/// `CONTEXTPLUS_DAEMON_SOCKET_PATH` if set.
pub fn daemon_socket_path(root_dir: &Path) -> PathBuf {
    if let Some(p) = socket_override() {
        return p;
    }
    daemon_dir(root_dir).join(DAEMON_SOCKET_FILE)
}

fn socket_override() -> Option<PathBuf> {
    std::env::var_os(SOCKET_PATH_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn clear_env() {
        // SAFETY: test-only — these env vars are scoped to this process and we
        // don't have other threads racing on them in this unit test.
        unsafe {
            std::env::remove_var(SOCKET_PATH_ENV);
        }
    }

    #[test]
    fn default_paths_live_under_mcp_data() {
        clear_env();
        let root = Path::new("/tmp/x");
        assert_eq!(daemon_dir(root), Path::new("/tmp/x/.mcp_data"));
        assert_eq!(
            daemon_lock_path(root),
            Path::new("/tmp/x/.mcp_data/contextplus.daemon.lock")
        );
        assert_eq!(
            daemon_pid_path(root),
            Path::new("/tmp/x/.mcp_data/contextplus.daemon.pid")
        );
        assert_eq!(
            daemon_socket_path(root),
            Path::new("/tmp/x/.mcp_data/contextplus.sock")
        );
    }

    #[test]
    fn socket_override_redirects_paths() {
        // SAFETY: test-only env mutation; the harness runs unit tests on a
        // single thread per `cargo test` binary and we restore state below.
        unsafe {
            std::env::set_var(SOCKET_PATH_ENV, "/var/run/cp/cp.sock");
        }
        let root = Path::new("/tmp/x");
        assert_eq!(daemon_socket_path(root), Path::new("/var/run/cp/cp.sock"));
        assert_eq!(daemon_dir(root), Path::new("/var/run/cp"));
        assert_eq!(
            daemon_lock_path(root),
            Path::new("/var/run/cp/contextplus.daemon.lock")
        );
        clear_env();
    }
}
