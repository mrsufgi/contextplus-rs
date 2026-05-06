//! On-disk path layout for the per-workspace daemon.
//!
//! ```text
//! <primary>/.mcp_data/
//!   contextplus.daemon.lock   advisory fd-lock — single-instance enforcement
//!   contextplus.daemon.pid    PID of the live daemon (best-effort, advisory)
//!   contextplus.sock          Unix-domain socket clients connect to
//! ```
//!
//! `<primary>` is the **primary worktree** of the calling root. Linked git
//! worktrees (`<primary>/.git/worktrees/<name>/`, materialized at any path
//! the user picks) resolve their daemon paths back to the primary worktree
//! so every worktree of the same repo connects to one daemon. See
//! [`crate::core::git_worktree`] for the resolution rules and edge cases.
//!
//! The socket path can be overridden with `CONTEXTPLUS_DAEMON_SOCKET_PATH` for
//! callers that need to escape the workspace `.mcp_data/` (e.g. workspaces
//! mounted read-only). When overridden, the lock + pid files live in the same
//! parent directory as the socket and primary-worktree resolution is bypassed.

use std::path::{Path, PathBuf};

use crate::core::git_worktree;

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
/// Resolution order:
/// 1. If `CONTEXTPLUS_DAEMON_SOCKET_PATH` is set, return its parent.
/// 2. Otherwise resolve `root_dir` to its primary worktree via
///    [`git_worktree::resolve_primary_worktree`] and join `.mcp_data`.
///
/// Step 2 ensures every linked worktree of a repo points at the same daemon
/// directory. If git resolution fails (no `.git`, malformed pointer, etc.),
/// it gracefully falls back to using `root_dir` itself.
pub fn daemon_dir(root_dir: &Path) -> PathBuf {
    if let Some(parent) = socket_override().and_then(|p| p.parent().map(Path::to_path_buf)) {
        return parent;
    }
    git_worktree::resolve_primary_worktree(root_dir).join(MCP_DATA_DIR)
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
    use std::sync::Mutex;

    /// Serialise tests that mutate process env vars so they cannot race each
    /// other under `--test-threads`. Same pattern as `transport::dispatch`.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear_env() {
        // SAFETY: test-only env mutation; ENV_LOCK ensures no racing reader.
        unsafe {
            std::env::remove_var(SOCKET_PATH_ENV);
        }
    }

    #[test]
    fn default_paths_live_under_mcp_data() {
        let _g = ENV_LOCK.lock().unwrap();
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
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: test-only env mutation; ENV_LOCK ensures no racing reader.
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

    /// When the daemon is invoked from inside a linked git worktree, the
    /// daemon directory must resolve to the primary worktree's `.mcp_data/`
    /// — that's how all worktrees of the same repo end up connecting to one
    /// daemon. This is the load-bearing invariant for U2.
    #[test]
    fn linked_worktree_daemon_dir_lives_under_primary() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_env();
        let td = tempfile::TempDir::new().unwrap();

        // Build a fake primary repo with a linked worktree.
        let primary = td.path().join("primary");
        std::fs::create_dir_all(primary.join(".git/worktrees/feat-x")).unwrap();
        std::fs::write(primary.join(".git/worktrees/feat-x/commondir"), "../..").unwrap();
        let wt_root = td.path().join("wt-feat-x");
        std::fs::create_dir_all(&wt_root).unwrap();
        std::fs::write(
            wt_root.join(".git"),
            format!(
                "gitdir: {}\n",
                primary.join(".git/worktrees/feat-x").display()
            ),
        )
        .unwrap();

        // Both the primary AND the linked worktree resolve to the SAME
        // daemon directory. We compare logical paths after canonicalizing
        // the *parent* (which exists), then appending MCP_DATA_DIR. We do
        // NOT canonicalize the daemon dir directly because the test does
        // not create `.mcp_data` on disk.
        let from_primary = daemon_dir(&primary);
        let from_worktree = daemon_dir(&wt_root);
        assert_eq!(
            from_primary,
            from_worktree,
            "primary and linked worktree must share daemon dir; got {} vs {}",
            from_primary.display(),
            from_worktree.display(),
        );
        // And it lives under the primary, not the worktree.
        let primary_canon = primary.canonicalize().unwrap();
        assert!(
            from_primary.starts_with(&primary_canon),
            "daemon_dir must live under primary; got {} (primary {})",
            from_primary.display(),
            primary_canon.display(),
        );
        assert_eq!(
            from_primary.file_name().unwrap(),
            std::ffi::OsStr::new(MCP_DATA_DIR)
        );
    }

    /// The `CONTEXTPLUS_DAEMON_SOCKET_PATH` override must continue to win
    /// over primary-worktree resolution — used for read-only or
    /// non-FS-default deployments.
    #[test]
    fn override_env_wins_over_primary_resolution() {
        let _g = ENV_LOCK.lock().unwrap();
        let td = tempfile::TempDir::new().unwrap();
        let primary = td.path().join("primary");
        std::fs::create_dir_all(primary.join(".git/worktrees/feat-y")).unwrap();
        std::fs::write(primary.join(".git/worktrees/feat-y/commondir"), "../..").unwrap();
        let wt_root = td.path().join("wt-feat-y");
        std::fs::create_dir_all(&wt_root).unwrap();
        std::fs::write(
            wt_root.join(".git"),
            format!(
                "gitdir: {}\n",
                primary.join(".git/worktrees/feat-y").display()
            ),
        )
        .unwrap();

        // SAFETY: test-only env mutation; ENV_LOCK held.
        unsafe {
            std::env::set_var(SOCKET_PATH_ENV, "/var/run/cp/override.sock");
        }
        // Daemon dir for the linked worktree should be the override's parent,
        // NOT the primary. Override is absolute precedence.
        assert_eq!(daemon_dir(&wt_root), Path::new("/var/run/cp"));
        clear_env();
    }
}
