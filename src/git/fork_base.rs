//! The fork base: a checkout of the primary's repository the daemon owns,
//! detached at the commit a configured ref such as `origin/main` names, so a
//! worktree cut from that ref can fork an index of it rather than of the
//! primary.
//!
//! The checkout is a linked worktree locked with the reason
//! [`LOCK_REASON`], so `git worktree prune` leaves it and a changed
//! directory setting can find the old one. Every git call runs with hooks off.

use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::error::Result;

/// The lock reason that marks a linked worktree as a fork base checkout.
pub const LOCK_REASON: &str = "contextplus-fork-base";

/// A fork base checkout this daemon holds the lock of.
pub struct ForkBase {
    /// The checkout's working tree.
    pub dir: PathBuf,
    /// The ref the checkout tracks.
    pub reference: String,
    /// The commit the checkout is at.
    pub head: String,
    _lock: fd_lock::RwLock<std::fs::File>,
}

/// The directory of the fork base checkout of the repository at
/// `primary_root`, named for its git common dir.
pub fn fork_base_dir(config: &Config, primary_root: &Path) -> Option<PathBuf> {
    let dirs = crate::core::git_worktree::git_dirs(primary_root)?;
    let hash = blake3::hash(dirs.common_dir.as_os_str().as_encoded_bytes());
    Some(config.fork_base_dir.as_ref()?.join(&hash.to_hex()[..16]))
}

/// The fork base checkout of the repository at `primary_root`, created at the
/// commit the configured ref names when missing. `None`, with the reason
/// logged, when the fork base is off or cannot be used.
pub fn ensure_fork_base(config: &Config, primary_root: &Path) -> Result<Option<ForkBase>> {
    let Some(reference) = config.fork_base.clone() else {
        return Ok(None);
    };
    let refused = |reason: &str| {
        tracing::info!(
            phase = "fork_base",
            reason,
            root = %primary_root.display(),
            "fork base off"
        );
        Ok(None)
    };
    let (Some(dirs), Some(dir)) = (
        crate::core::git_worktree::git_dirs(primary_root),
        fork_base_dir(config, primary_root),
    ) else {
        return refused("no_checkout_dir");
    };
    let parent = dir.parent().unwrap_or(&dir);
    std::fs::create_dir_all(parent)?;
    let mut lock = fd_lock::RwLock::new(
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.with_extension("lock"))?,
    );
    match lock.try_write() {
        // Held until the checkout is dropped and its file closed.
        Ok(guard) => std::mem::forget(guard),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            return refused("locked_elsewhere");
        }
        Err(error) => return Err(error.into()),
    }
    let Some(sha) = resolve(primary_root, &reference) else {
        return refused("unresolved_ref");
    };
    remove_other_checkouts(primary_root, &dir);
    if dir.exists() {
        let ours = crate::core::git_worktree::git_dirs(&dir)
            .is_some_and(|checkout| checkout.common_dir == dirs.common_dir);
        if !ours {
            return refused("foreign_checkout");
        }
    } else {
        let target = dir.to_string_lossy();
        // Twice forced: a registration left locked by a removed checkout
        // would otherwise refuse the path.
        let added = git(
            primary_root,
            &[
                "worktree",
                "add",
                "-f",
                "-f",
                "--detach",
                "--lock",
                "--reason",
                LOCK_REASON,
                &target,
                &sha,
            ],
        );
        if added.is_none() {
            return refused("checkout_failed");
        }
    }
    let Some(head) = resolve(&dir, "HEAD") else {
        return refused("unresolved_head");
    };
    Ok(Some(ForkBase {
        dir,
        reference,
        head,
        _lock: lock,
    }))
}

/// The commit `reference` names in the repository at `root`.
pub(crate) fn resolve(root: &Path, reference: &str) -> Option<String> {
    git(
        root,
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("{reference}^{{commit}}"),
        ],
    )
}

/// Removes the fork base checkouts of the repository at `primary_root` other
/// than `dir`, as a changed directory setting leaves them.
fn remove_other_checkouts(primary_root: &Path, dir: &Path) {
    let Some(listed) = git(primary_root, &["worktree", "list", "--porcelain"]) else {
        return;
    };
    let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let mut path = None;
    for line in listed.lines() {
        if let Some(listed) = line.strip_prefix("worktree ") {
            path = Some(PathBuf::from(listed));
        } else if line.strip_prefix("locked ").map(str::trim) == Some(LOCK_REASON)
            && let Some(path) = path.take().filter(|path| *path != canonical && path != dir)
        {
            let removed = git(
                primary_root,
                &["worktree", "remove", "-f", "-f", &path.to_string_lossy()],
            );
            tracing::info!(
                phase = "fork_base",
                path = %path.display(),
                removed = removed.is_some(),
                "removed an old fork base checkout"
            );
        }
    }
}

/// The trimmed output of `git <args>` in `cwd` with hooks off, `None` when it fails.
fn git(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["-c", "core.hooksPath=/dev/null"])
        .args(args)
        .env("LEFTHOOK", "0")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|out| out.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn git(cwd: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(cwd)
            .args(args)
            .env("GIT_AUTHOR_NAME", "t")
            .env("GIT_AUTHOR_EMAIL", "t@t")
            .env("GIT_COMMITTER_NAME", "t")
            .env("GIT_COMMITTER_EMAIL", "t@t")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .output()
            .expect("git runs");
        assert!(output.status.success(), "git {args:?}: {output:?}");
        String::from_utf8(output.stdout).unwrap().trim().to_string()
    }

    /// A repository with one commit and `refs/remotes/origin/main` at it.
    fn repository() -> (tempfile::TempDir, String) {
        let primary = tempfile::tempdir().unwrap();
        git(primary.path(), &["init", "-q", "-b", "main"]);
        std::fs::write(primary.path().join("lib.rs"), "pub fn base() {}\n").unwrap();
        std::fs::write(primary.path().join(".gitignore"), ".mcp_data/\n").unwrap();
        git(primary.path(), &["add", "-A"]);
        git(primary.path(), &["commit", "-qm", "base"]);
        let sha = git(primary.path(), &["rev-parse", "HEAD"]);
        git(
            primary.path(),
            &["update-ref", "refs/remotes/origin/main", &sha],
        );
        (primary, sha)
    }

    fn config(bases: &Path) -> Config {
        Config::from_env_map(&HashMap::from([
            (
                "CONTEXTPLUS_FORK_BASE".to_string(),
                "origin/main".to_string(),
            ),
            (
                "CONTEXTPLUS_FORK_BASE_DIR".to_string(),
                bases.to_string_lossy().into_owned(),
            ),
        ]))
    }

    /// `(path, locked reason)` of each linked worktree of `primary`.
    fn worktrees(primary: &Path) -> Vec<(PathBuf, Option<String>)> {
        let mut listed = Vec::new();
        for line in git(primary, &["worktree", "list", "--porcelain"]).lines() {
            if let Some(path) = line.strip_prefix("worktree ") {
                listed.push((PathBuf::from(path), None));
            } else if let Some(reason) = line.strip_prefix("locked") {
                listed.last_mut().unwrap().1 = Some(reason.trim().to_string());
            }
        }
        listed
    }

    #[test]
    fn fork_base_is_a_locked_detached_checkout_of_the_ref() {
        let (primary, sha) = repository();
        let sentinel = primary.path().join("hook-ran");
        let hook = primary.path().join(".git/hooks/post-checkout");
        std::fs::write(&hook, format!("#!/bin/sh\ntouch {}\n", sentinel.display())).unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let bases = tempfile::tempdir().unwrap();

        let base = ensure_fork_base(&config(bases.path()), primary.path())
            .unwrap()
            .expect("a fork base");
        assert!(base.dir.starts_with(bases.path()));
        assert_eq!(base.head, sha);
        assert_eq!(git(&base.dir, &["rev-parse", "HEAD"]), sha);
        assert_eq!(
            git(&base.dir, &["rev-parse", "--abbrev-ref", "HEAD"]),
            "HEAD",
            "the fork base is on a branch"
        );
        let dir = base.dir.canonicalize().unwrap();
        assert!(
            worktrees(primary.path())
                .iter()
                .any(|(path, reason)| *path == dir && reason.as_deref() == Some(LOCK_REASON)),
            "{:?}",
            worktrees(primary.path())
        );
        assert!(!sentinel.exists(), "a git hook ran for the fork base");
    }

    #[test]
    fn fork_base_lock_held_elsewhere_gives_none() {
        let (primary, _) = repository();
        let bases = tempfile::tempdir().unwrap();
        let config = config(bases.path());
        let held = ensure_fork_base(&config, primary.path())
            .unwrap()
            .expect("a fork base");

        assert!(ensure_fork_base(&config, primary.path()).unwrap().is_none());
        assert!(held.dir.exists());
    }

    #[test]
    fn fork_base_repairs_a_locked_registration_with_no_checkout() {
        let (primary, sha) = repository();
        let bases = tempfile::tempdir().unwrap();
        let config = config(bases.path());
        let dir = fork_base_dir(&config, primary.path()).unwrap();
        std::fs::create_dir_all(bases.path()).unwrap();
        git(
            primary.path(),
            &[
                "worktree",
                "add",
                "-q",
                "--detach",
                "--lock",
                "--reason",
                LOCK_REASON,
                &dir.to_string_lossy(),
                &sha,
            ],
        );
        std::fs::remove_dir_all(&dir).unwrap();

        let base = ensure_fork_base(&config, primary.path())
            .unwrap()
            .expect("a repaired fork base");
        assert_eq!(base.dir, dir);
        assert_eq!(git(&base.dir, &["rev-parse", "HEAD"]), sha);
    }

    #[test]
    fn fork_base_in_a_new_directory_removes_the_old_checkout() {
        let (primary, _) = repository();
        let old_bases = tempfile::tempdir().unwrap();
        let old = ensure_fork_base(&config(old_bases.path()), primary.path())
            .unwrap()
            .expect("a fork base")
            .dir;
        let new_bases = tempfile::tempdir().unwrap();

        let base = ensure_fork_base(&config(new_bases.path()), primary.path())
            .unwrap()
            .expect("a fork base");
        assert!(!old.exists(), "the old fork base checkout remains");
        let listed: Vec<_> = worktrees(primary.path())
            .into_iter()
            .filter(|(_, reason)| reason.as_deref() == Some(LOCK_REASON))
            .map(|(path, _)| path)
            .collect();
        assert_eq!(listed, vec![base.dir.canonicalize().unwrap()]);
    }

    #[test]
    fn fork_base_refuses_a_checkout_of_another_repository() {
        let (primary, _) = repository();
        let (other, _) = repository();
        let bases = tempfile::tempdir().unwrap();
        let config = config(bases.path());
        let dir = ensure_fork_base(&config, other.path())
            .unwrap()
            .expect("a fork base")
            .dir;
        let name = dir.file_name().unwrap().to_owned();
        let foreign = fork_base_dir(&config, primary.path()).unwrap();
        std::fs::rename(&dir, &foreign).unwrap();
        assert_ne!(foreign.file_name().unwrap(), name);

        assert!(ensure_fork_base(&config, primary.path()).unwrap().is_none());
    }
}
