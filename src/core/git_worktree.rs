//! Resolve git worktree relationships.
//!
//! When `<root>/.git` is a directory, `root` is the primary worktree. When it
//! is a file, the file contains a `gitdir: <path>` pointer into
//! `<primary>/.git/worktrees/<name>/`, and that directory holds a `commondir`
//! file (relative or absolute) pointing back to the primary's `.git/`.
//!
//! ```text
//! primary repo (root):
//!   /repos/foo/
//!     .git/                     ← directory
//!       worktrees/
//!         feat-bar/             ← linked-worktree gitdir
//!           commondir → ../..   (relative to /repos/foo/.git/worktrees/feat-bar/)
//!           HEAD, refs/, ...
//!   /repos/foo-feat-bar/        ← linked worktree's working tree
//!     .git                      ← file: `gitdir: /repos/foo/.git/worktrees/feat-bar`
//! ```
//!
//! Both `/repos/foo/` and `/repos/foo-feat-bar/` should resolve to
//! `/repos/foo/` as the **primary worktree root**. That's the directory the
//! daemon places `.mcp_data/` and the unix socket next to, so every worktree
//! of the same repo connects to one daemon.
//!
//! Resolution deliberately avoids invoking `git`. A subprocess call
//! per daemon-resolve would slow startup and require git in PATH.

use std::fs;
use std::path::{Path, PathBuf};

/// Resolve the primary worktree root for `root`.
///
/// **Always returns a canonicalized path** (when canonicalize succeeds). This
/// matters because `daemon_dir(primary)` and `daemon_dir(linked_worktree)`
/// must compare equal byte-for-byte for two bridges to converge on one
/// daemon. On macOS where `/var` is a symlink to `/private/var`, comparing
/// non-canonical paths from one branch with canonical paths from another
/// would silently break worktree convergence.
///
/// **Returns `root` on any error**, after emitting a `tracing::warn!`. This
/// function never panics and never bails — a half-broken worktree pointer
/// should degrade to "treat this directory as its own primary" rather than
/// crash the daemon.
pub fn resolve_primary_worktree(root: &Path) -> PathBuf {
    // Canonicalize once up-front so every return path produces the same form.
    // The cost is one stat-and-symlink-resolve syscall, observable but small;
    // critically, both fast and slow paths must agree on path shape.
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());

    let dot_git = canonical_root.join(".git");
    match fs::symlink_metadata(&dot_git) {
        Ok(m) if m.is_dir() => {
            // Fast path: standard repo. `canonical_root` IS primary.
            return canonical_root;
        }
        Err(_) => {
            // Fast path: no `.git` at all. Non-git folder; canonical root is fine.
            return canonical_root;
        }
        Ok(_) => {
            // `.git` is a file (or symlink to file): need slow path.
        }
    }

    // Slow path: linked worktree. Parse `gitdir:` pointer.
    match resolve_linked_worktree_primary(&canonical_root) {
        Ok(primary) => primary,
        Err(e) => {
            tracing::warn!(
                root = %canonical_root.display(),
                error = %e,
                "could not resolve primary worktree; treating root as primary"
            );
            canonical_root
        }
    }
}

/// The git directories behind a working tree: its own gitdir (HEAD,
/// ORIG_HEAD, index) and the common dir that holds `refs/heads` and
/// `packed-refs`. Equal for a primary checkout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GitDirs {
    pub gitdir: PathBuf,
    pub common_dir: PathBuf,
}

impl GitDirs {
    /// Whether an event at `path` can mean HEAD moved: a HEAD-like file or
    /// a branch ref, in this worktree's gitdir or the shared common dir.
    pub fn is_ref_update(&self, path: &Path) -> bool {
        if !(path.starts_with(&self.gitdir) || path.starts_with(&self.common_dir)) {
            return false;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.ends_with("HEAD")
            || name == "packed-refs"
            || path.starts_with(self.common_dir.join("refs").join("heads"))
            || path.starts_with(self.gitdir.join("refs").join("heads"))
    }
}

/// Resolve [`GitDirs`] for `root` without invoking git. `None` when `root` is
/// not a git working tree or its pointers do not resolve.
pub fn git_dirs(root: &Path) -> Option<GitDirs> {
    let dot_git = root.join(".git");
    let meta = fs::symlink_metadata(&dot_git).ok()?;
    if meta.is_dir() {
        let gitdir = dot_git.canonicalize().unwrap_or(dot_git);
        return Some(GitDirs {
            common_dir: gitdir.clone(),
            gitdir,
        });
    }
    let pointer = parse_gitdir_pointer(&fs::read_to_string(&dot_git).ok()?)?;
    let gitdir = if pointer.is_absolute() {
        pointer
    } else {
        root.join(pointer)
    };
    let gitdir = gitdir.canonicalize().ok()?;
    let common_dir = match fs::read_to_string(gitdir.join("commondir")) {
        Ok(contents) => {
            let raw = Path::new(contents.trim());
            let common = if raw.is_absolute() {
                raw.to_path_buf()
            } else {
                gitdir.join(raw)
            };
            common.canonicalize().unwrap_or(common)
        }
        Err(_) => gitdir.clone(),
    };
    Some(GitDirs { gitdir, common_dir })
}

/// Slow path: `<root>/.git` is a file. Parse the `gitdir:` pointer and walk
/// to the primary's `.git/` via the linked-worktree gitdir's `commondir` file.
fn resolve_linked_worktree_primary(root: &Path) -> Result<PathBuf, ResolveError> {
    let dot_git = root.join(".git");
    // `.git` is a file: parse `gitdir: <path>` pointer.
    let contents = fs::read_to_string(&dot_git).map_err(ResolveError::ReadDotGit)?;
    let worktree_gitdir =
        parse_gitdir_pointer(&contents).ok_or(ResolveError::MalformedGitdirPointer)?;

    // The pointer is usually absolute; if relative, it's relative to the
    // worktree directory itself (per git docs).
    let worktree_gitdir = if worktree_gitdir.is_absolute() {
        worktree_gitdir
    } else {
        root.join(&worktree_gitdir)
    };

    let worktree_gitdir = worktree_gitdir
        .canonicalize()
        .map_err(|e| ResolveError::GitdirNotResolvable(e.to_string()))?;

    // Read the `commondir` file inside the worktree gitdir. Its contents are
    // a path (relative to worktree gitdir, or absolute) that points at the
    // primary's `.git/`.
    let commondir_file = worktree_gitdir.join("commondir");
    let commondir_raw = fs::read_to_string(&commondir_file)
        .map_err(|e| ResolveError::ReadCommondir(e.to_string()))?;
    let commondir_raw = commondir_raw.trim();
    if commondir_raw.is_empty() {
        return Err(ResolveError::EmptyCommondir);
    }

    let commondir_path = PathBuf::from(commondir_raw);
    let primary_gitdir = if commondir_path.is_absolute() {
        commondir_path
    } else {
        worktree_gitdir.join(&commondir_path)
    };
    let primary_gitdir = primary_gitdir
        .canonicalize()
        .map_err(|e| ResolveError::CommondirNotResolvable(e.to_string()))?;

    // Primary repo root = parent of `<primary>/.git/`.
    primary_gitdir
        .parent()
        .map(Path::to_path_buf)
        .ok_or(ResolveError::CommondirHasNoParent)
}

/// Parse the first `gitdir: <path>` line from a `.git`-as-file's content.
fn parse_gitdir_pointer(contents: &str) -> Option<PathBuf> {
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("gitdir:") {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
    }
    None
}

#[derive(Debug)]
enum ResolveError {
    ReadDotGit(std::io::Error),
    MalformedGitdirPointer,
    GitdirNotResolvable(String),
    ReadCommondir(String),
    EmptyCommondir,
    CommondirNotResolvable(String),
    CommondirHasNoParent,
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadDotGit(e) => write!(f, "could not read .git pointer file: {e}"),
            Self::MalformedGitdirPointer => {
                write!(f, ".git file is missing or has malformed gitdir: line")
            }
            Self::GitdirNotResolvable(e) => write!(f, "gitdir pointer does not resolve: {e}"),
            Self::ReadCommondir(e) => {
                write!(f, "could not read commondir file in worktree gitdir: {e}")
            }
            Self::EmptyCommondir => write!(f, "commondir file is empty"),
            Self::CommondirNotResolvable(e) => {
                write!(f, "commondir does not resolve to a real path: {e}")
            }
            Self::CommondirHasNoParent => write!(f, "resolved primary gitdir has no parent"),
        }
    }
}

impl std::error::Error for ResolveError {}

/// Relative path to blob id of the files git shows clean in one working tree.
pub type CleanBlobs = std::collections::HashMap<String, String>;

/// The entries with the same blob in both: across two trees, the files that
/// are byte-identical in both; across two reads of one tree, the files that
/// stayed clean and unchanged in between.
pub fn common_blobs(a: &CleanBlobs, b: &CleanBlobs) -> CleanBlobs {
    a.iter()
        .filter(|(path, blob)| b.get(*path) == Some(*blob))
        .map(|(path, blob)| (path.clone(), blob.clone()))
        .collect()
}

/// Regular files tracked at stage 0 whose working copy matches the index, with
/// their blob ids. Files git is told not to check (`assume-unchanged`,
/// `skip-worktree`) are left out. `None` when `root` is not a git working tree.
pub fn clean_blobs(root: &Path) -> Option<CleanBlobs> {
    let git = |args: &[&str]| {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .env("GIT_OPTIONAL_LOCKS", "0")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .output()
            .ok()?;
        output.status.success().then_some(output.stdout)
    };
    let (staged, modified) = std::thread::scope(|scope| {
        let staged = scope.spawn(|| git(&["ls-files", "--stage", "-v", "-z"]));
        let modified = git(&["diff-files", "--name-only", "--relative", "-z"]);
        (staged.join().ok().flatten(), modified)
    });
    let (staged, modified) = (staged?, modified?);
    let modified: std::collections::HashSet<&[u8]> = modified
        .split(|&byte| byte == 0)
        .filter(|path| !path.is_empty())
        .collect();
    Some(
        staged
            .split(|&byte| byte == 0)
            .filter_map(|record| {
                let record = std::str::from_utf8(record).ok()?;
                let (meta, path) = record.split_once('\t')?;
                let mut fields = meta.split(' ');
                let (tag, mode, blob, stage) = (
                    fields.next()?,
                    fields.next()?,
                    fields.next()?,
                    fields.next()?,
                );
                (tag == "H"
                    && mode.starts_with("100")
                    && stage == "0"
                    && !modified.contains(path.as_bytes()))
                .then(|| (path.to_string(), blob.to_string()))
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn git(cwd: &Path, args: &[&str]) {
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
    }

    #[test]
    fn common_blobs_lists_only_clean_equal_blobs() {
        let primary = TempDir::new().unwrap();
        let root = primary.path();
        git(root, &["init", "-q", "-b", "main"]);
        fs::create_dir_all(root.join("sub")).unwrap();
        for name in [
            "same.rs",
            "dirty.rs",
            "committed.rs",
            "primary_dirty.rs",
            "sub/nested.rs",
        ] {
            fs::write(root.join(name), format!("// {name}\n")).unwrap();
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink("committed.rs", root.join("link.rs")).unwrap();
        git(root, &["add", "-A"]);
        git(root, &["commit", "-qm", "base"]);
        let holder = TempDir::new().unwrap();
        let worktree = holder.path().join("wt");
        git(
            root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "wt",
                worktree.to_str().unwrap(),
            ],
        );
        fs::write(worktree.join("dirty.rs"), "// changed\n").unwrap();
        fs::write(worktree.join("committed.rs"), "// committed\n").unwrap();
        git(&worktree, &["commit", "-qam", "change"]);
        fs::write(worktree.join("dirty.rs"), "// changed again\n").unwrap();
        fs::write(worktree.join("untracked.rs"), "// new\n").unwrap();
        fs::write(root.join("primary_dirty.rs"), "// edited\n").unwrap();

        let same = common_blobs(
            &clean_blobs(root).unwrap(),
            &clean_blobs(&worktree).unwrap(),
        );
        let mut same: Vec<String> = same.into_keys().collect();
        same.sort();
        assert_eq!(same, ["same.rs", "sub/nested.rs"]);
        assert!(clean_blobs(holder.path()).is_none());
    }

    #[test]
    fn clean_blobs_drops_assume_unchanged_and_skip_worktree_entries() {
        let repo = TempDir::new().unwrap();
        let root = repo.path();
        git(root, &["init", "-q", "-b", "main"]);
        for name in ["plain.rs", "assumed.rs", "skipped.rs"] {
            fs::write(root.join(name), format!("// {name}\n")).unwrap();
        }
        git(root, &["add", "-A"]);
        git(root, &["commit", "-qm", "base"]);
        git(root, &["update-index", "--assume-unchanged", "assumed.rs"]);
        git(root, &["update-index", "--skip-worktree", "skipped.rs"]);
        fs::write(root.join("assumed.rs"), "// edited\n").unwrap();
        fs::remove_file(root.join("skipped.rs")).unwrap();

        let mut clean: Vec<String> = clean_blobs(root).unwrap().into_keys().collect();
        clean.sort();
        assert_eq!(clean, ["plain.rs"]);
    }

    /// Build a fake "primary repo" with a `.git/` directory and return the
    /// repo root.
    fn make_primary(td: &TempDir) -> PathBuf {
        let root = td.path().join("primary");
        fs::create_dir_all(root.join(".git/worktrees")).unwrap();
        // git also writes `commondir`-related files but we only need the dir.
        root
    }

    /// Build a fake linked worktree pointing at the given primary's gitdir
    /// with an absolute `gitdir:` pointer. Returns (worktree_root,
    /// worktree_gitdir).
    fn make_linked_worktree(
        td: &TempDir,
        primary_root: &Path,
        worktree_name: &str,
    ) -> (PathBuf, PathBuf) {
        let primary_gitdir = primary_root.join(".git");
        let worktree_gitdir = primary_gitdir.join("worktrees").join(worktree_name);
        fs::create_dir_all(&worktree_gitdir).unwrap();

        // commondir points back to the primary `.git/` (relative `..`).
        fs::write(worktree_gitdir.join("commondir"), "../..").unwrap();

        let worktree_root = td.path().join(format!("wt-{}", worktree_name));
        fs::create_dir_all(&worktree_root).unwrap();
        fs::write(
            worktree_root.join(".git"),
            format!("gitdir: {}\n", worktree_gitdir.display()),
        )
        .unwrap();
        (worktree_root, worktree_gitdir)
    }

    #[test]
    fn primary_repo_resolves_to_self() {
        let td = TempDir::new().unwrap();
        let primary = make_primary(&td);
        assert_eq!(
            resolve_primary_worktree(&primary).canonicalize().unwrap(),
            primary.canonicalize().unwrap()
        );
    }

    #[test]
    fn no_dot_git_returns_root() {
        let td = TempDir::new().unwrap();
        let dir = td.path().join("plain");
        fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            resolve_primary_worktree(&dir).canonicalize().unwrap(),
            dir.canonicalize().unwrap()
        );
    }

    #[test]
    fn linked_worktree_absolute_pointer_resolves_to_primary() {
        let td = TempDir::new().unwrap();
        let primary = make_primary(&td);
        let (wt_root, _) = make_linked_worktree(&td, &primary, "feat-x");
        assert_eq!(
            resolve_primary_worktree(&wt_root).canonicalize().unwrap(),
            primary.canonicalize().unwrap()
        );
    }

    #[test]
    fn relative_gitdir_pointer_resolves_correctly() {
        // Hand-crafted relative pointer: write a `.git` file containing a
        // `gitdir:` path computed manually from the temp tree. Verifies that
        // the relative-path branch in resolve_primary_inner works without
        // needing a third-party `pathdiff` dependency.
        let td = TempDir::new().unwrap();
        let primary = make_primary(&td);
        let wt_gitdir = primary.join(".git/worktrees/feat-rel");
        fs::create_dir_all(&wt_gitdir).unwrap();
        fs::write(wt_gitdir.join("commondir"), "../..").unwrap();

        // The worktree root sits at td.path().join("wt-feat-rel").
        // Manually compute a relative path from worktree root to wt_gitdir.
        let wt_root = td.path().join("wt-feat-rel");
        fs::create_dir_all(&wt_root).unwrap();
        // Both paths are direct children of `td.path()` — relative path is
        // `../primary/.git/worktrees/feat-rel`.
        let rel = "../primary/.git/worktrees/feat-rel";
        fs::write(wt_root.join(".git"), format!("gitdir: {}\n", rel)).unwrap();

        assert_eq!(
            resolve_primary_worktree(&wt_root).canonicalize().unwrap(),
            primary.canonicalize().unwrap()
        );
    }

    #[test]
    fn malformed_gitdir_pointer_falls_back_to_root() {
        let td = TempDir::new().unwrap();
        let dir = td.path().join("bad-pointer");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(".git"), "gitdir:\n").unwrap();
        // Should NOT panic; should fall back to the root itself.
        assert_eq!(
            resolve_primary_worktree(&dir).canonicalize().unwrap(),
            dir.canonicalize().unwrap()
        );
    }

    #[test]
    fn unparseable_dot_git_file_falls_back_to_root() {
        let td = TempDir::new().unwrap();
        let dir = td.path().join("garbage");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(".git"), "this is not a gitdir pointer\n").unwrap();
        assert_eq!(
            resolve_primary_worktree(&dir).canonicalize().unwrap(),
            dir.canonicalize().unwrap()
        );
    }

    #[test]
    fn pointer_to_nonexistent_path_falls_back_to_root() {
        let td = TempDir::new().unwrap();
        let dir = td.path().join("dangling");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join(".git"),
            "gitdir: /nonexistent/path/.git/worktrees/x\n",
        )
        .unwrap();
        assert_eq!(
            resolve_primary_worktree(&dir).canonicalize().unwrap(),
            dir.canonicalize().unwrap()
        );
    }

    #[test]
    fn worktree_with_missing_commondir_falls_back_to_root() {
        let td = TempDir::new().unwrap();
        let primary = make_primary(&td);
        let bad_wt_gitdir = primary.join(".git/worktrees/no-commondir");
        fs::create_dir_all(&bad_wt_gitdir).unwrap();
        // intentionally do not write `commondir`

        let wt_root = td.path().join("wt-no-commondir");
        fs::create_dir_all(&wt_root).unwrap();
        fs::write(
            wt_root.join(".git"),
            format!("gitdir: {}\n", bad_wt_gitdir.display()),
        )
        .unwrap();

        assert_eq!(
            resolve_primary_worktree(&wt_root).canonicalize().unwrap(),
            wt_root.canonicalize().unwrap()
        );
    }

    #[test]
    fn parse_gitdir_pointer_skips_blank_lines() {
        assert_eq!(
            parse_gitdir_pointer("\n\ngitdir: /a/b/c\n"),
            Some(PathBuf::from("/a/b/c"))
        );
    }

    #[test]
    fn git_dirs_for_primary_and_linked_worktree() {
        let td = TempDir::new().unwrap();
        let primary = make_primary(&td);
        let (wt_root, wt_gitdir) = make_linked_worktree(&td, &primary, "feat-y");

        let p = git_dirs(&primary).unwrap();
        assert_eq!(p.gitdir, primary.join(".git").canonicalize().unwrap());
        assert_eq!(p.common_dir, p.gitdir);

        let w = git_dirs(&wt_root).unwrap();
        assert_eq!(w.gitdir, wt_gitdir.canonicalize().unwrap());
        assert_eq!(w.common_dir, primary.join(".git").canonicalize().unwrap());

        assert!(w.is_ref_update(&w.gitdir.join("HEAD")));
        assert!(w.is_ref_update(&w.gitdir.join("ORIG_HEAD")));
        assert!(w.is_ref_update(&w.common_dir.join("packed-refs")));
        assert!(w.is_ref_update(&w.common_dir.join("refs/heads/feat/y")));
        assert!(!w.is_ref_update(&w.gitdir.join("index")));
        assert!(!w.is_ref_update(&w.common_dir.join("refs/remotes/origin/main")));
        assert!(!w.is_ref_update(&wt_root.join("src/HEAD")));

        let plain = td.path().join("plain");
        fs::create_dir_all(&plain).unwrap();
        assert_eq!(git_dirs(&plain), None);
    }

    #[test]
    fn parse_gitdir_pointer_returns_none_when_absent() {
        assert_eq!(parse_gitdir_pointer("nothing here\n"), None);
        assert_eq!(parse_gitdir_pointer(""), None);
    }
}
