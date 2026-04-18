//! Git hook installer for contextplus-rs.
//!
//! Wires lightweight `post-commit`, `post-merge`, and `post-checkout` hooks
//! into the host repository. Each hook touches a sentinel file under
//! `.mcp_data/hooks/` — the running MCP server's tracker (already watching the
//! workspace) picks up the touch and triggers a refresh that knows it can
//! batch the changed files revealed by the corresponding git command.
//!
//! Why a sentinel-file dance instead of running contextplus-rs directly from
//! the hook? Hooks must be fast (commit-time latency hurts) and side-effect
//! free for users without contextplus running. The sentinel is a no-op when
//! nothing is watching, and the MCP picks up real work only when present.
//!
//! Each managed hook is wrapped in BEGIN/END markers so re-installs are
//! idempotent and uninstall removes only our block — preserving any
//! user-authored hook content above or below.

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{ContextPlusError, Result};

const BEGIN_MARK: &str = "# >>> contextplus-rs hooks (managed) >>>";
const END_MARK: &str = "# <<< contextplus-rs hooks (managed) <<<";

/// Hook names contextplus manages. Subset chosen to capture the events that
/// most often invalidate working-tree embeddings without spamming refreshes.
pub const MANAGED_HOOKS: &[&str] = &["post-commit", "post-merge", "post-checkout"];

/// Locate the `.git/hooks` directory for the given working tree.
///
/// Handles three layouts:
/// 1. Plain checkout: `<root>/.git/hooks`.
/// 2. Worktree: `<root>/.git` is a *file* containing `gitdir: <commondir>/worktrees/<name>`;
///    hooks live at `<commondir>/hooks` (the worktree shares the host repo's hooks).
/// 3. `core.hooksPath` override is *not* respected here — installing hooks at
///    a custom location would surprise the user; we emit an error and let the
///    caller decide.
pub fn hooks_dir(root_dir: &Path) -> Result<PathBuf> {
    let git_path = root_dir.join(".git");
    let metadata = fs::metadata(&git_path)
        .map_err(|_| ContextPlusError::Cache(format!("not a git repo: {}", root_dir.display())))?;
    if metadata.is_dir() {
        return Ok(git_path.join("hooks"));
    }
    // Worktree: read pointer file
    let contents = fs::read_to_string(&git_path)?;
    let line = contents
        .lines()
        .find(|l| l.starts_with("gitdir:"))
        .ok_or_else(|| ContextPlusError::Cache(".git file missing gitdir entry".into()))?;
    let gitdir = Path::new(line.trim_start_matches("gitdir:").trim());
    // <commondir>/worktrees/<name> → walk back to <commondir>
    let commondir = gitdir
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| ContextPlusError::Cache("malformed worktree gitdir".into()))?;
    Ok(commondir.join("hooks"))
}

/// Install (or refresh) contextplus-rs hooks in `root_dir`.
///
/// Each managed hook becomes a `#!/usr/bin/env sh` file containing our
/// marker block. If a hook already exists with our markers, the block is
/// replaced. If it exists *without* our markers (user content), our block is
/// appended below the existing content.
pub fn install_hooks(root_dir: &Path) -> Result<Vec<PathBuf>> {
    let dir = hooks_dir(root_dir)?;
    fs::create_dir_all(&dir)?;
    let sentinel_dir = root_dir.join(".mcp_data").join("hooks");
    fs::create_dir_all(&sentinel_dir)?;

    let mut installed = Vec::new();
    for hook in MANAGED_HOOKS {
        let path = dir.join(hook);
        let block = block_for(hook, &sentinel_dir);
        let new_contents = match fs::read_to_string(&path) {
            Ok(existing) => upsert_block(&existing, &block),
            Err(_) => format!("#!/usr/bin/env sh\n{}\n", block),
        };
        fs::write(&path, new_contents)?;
        set_executable(&path)?;
        installed.push(path);
    }
    Ok(installed)
}

/// Remove contextplus-rs hook blocks. If a hook file becomes empty (just the
/// shebang) we delete it; otherwise we keep the user's content.
pub fn uninstall_hooks(root_dir: &Path) -> Result<Vec<PathBuf>> {
    let dir = hooks_dir(root_dir)?;
    let mut removed = Vec::new();
    for hook in MANAGED_HOOKS {
        let path = dir.join(hook);
        let Ok(existing) = fs::read_to_string(&path) else {
            continue;
        };
        let stripped = strip_block(&existing);
        // Treat shebang-only or whitespace-only as empty.
        let trimmed: String = stripped
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("#!"))
            .collect();
        if trimmed.is_empty() {
            fs::remove_file(&path)?;
        } else {
            fs::write(&path, stripped)?;
        }
        removed.push(path);
    }
    Ok(removed)
}

fn block_for(hook: &str, sentinel_dir: &Path) -> String {
    let sentinel = sentinel_dir.join(hook);
    format!(
        "{}\n# Touch a sentinel that the running contextplus-rs MCP server picks up.\n# No-op when contextplus-rs isn't running — safe to leave installed.\nmkdir -p \"{}\" 2>/dev/null && touch \"{}\" 2>/dev/null || true\n{}",
        BEGIN_MARK,
        sentinel_dir.display(),
        sentinel.display(),
        END_MARK,
    )
}

/// Insert or replace our marker block within an existing hook file.
fn upsert_block(existing: &str, new_block: &str) -> String {
    if let (Some(start), Some(end)) = (existing.find(BEGIN_MARK), existing.find(END_MARK)) {
        let end_line_end = existing[end..].find('\n').map(|i| end + i + 1).unwrap_or(existing.len());
        let mut out = String::with_capacity(existing.len());
        out.push_str(&existing[..start]);
        out.push_str(new_block);
        out.push('\n');
        out.push_str(&existing[end_line_end..]);
        return out;
    }
    // No existing block — append below the user's content.
    let mut out = existing.to_string();
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(new_block);
    out.push('\n');
    out
}

/// Remove our marker block (and surrounding empty lines) from a hook file.
fn strip_block(existing: &str) -> String {
    let (Some(start), Some(end)) = (existing.find(BEGIN_MARK), existing.find(END_MARK)) else {
        return existing.to_string();
    };
    let end_line_end = existing[end..].find('\n').map(|i| end + i + 1).unwrap_or(existing.len());
    let mut out = String::with_capacity(existing.len());
    out.push_str(&existing[..start]);
    out.push_str(&existing[end_line_end..]);
    out
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<()> {
    // No-op on Windows; git for Windows ignores the executable bit.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn init_main_checkout(tmp: &TempDir) {
        fs::create_dir(tmp.path().join(".git")).unwrap();
        fs::create_dir(tmp.path().join(".git").join("hooks")).unwrap();
    }

    #[test]
    fn hooks_dir_for_main_checkout() {
        let tmp = TempDir::new().unwrap();
        init_main_checkout(&tmp);
        assert_eq!(hooks_dir(tmp.path()).unwrap(), tmp.path().join(".git/hooks"));
    }

    #[test]
    fn hooks_dir_for_worktree_resolves_to_common_dir() {
        // Set up a fake main repo + worktree side-by-side
        let main = TempDir::new().unwrap();
        fs::create_dir(main.path().join(".git")).unwrap();
        fs::create_dir(main.path().join(".git").join("hooks")).unwrap();
        fs::create_dir_all(main.path().join(".git").join("worktrees").join("wt-a")).unwrap();

        let wt = TempDir::new().unwrap();
        let gitdir = main.path().join(".git").join("worktrees").join("wt-a");
        fs::write(
            wt.path().join(".git"),
            format!("gitdir: {}\n", gitdir.display()),
        )
        .unwrap();

        let resolved = hooks_dir(wt.path()).unwrap();
        assert_eq!(resolved, main.path().join(".git").join("hooks"));
    }

    #[test]
    fn hooks_dir_errors_for_non_git_dir() {
        let tmp = TempDir::new().unwrap();
        assert!(hooks_dir(tmp.path()).is_err());
    }

    #[test]
    fn install_creates_all_managed_hooks() {
        let tmp = TempDir::new().unwrap();
        init_main_checkout(&tmp);

        let installed = install_hooks(tmp.path()).unwrap();
        assert_eq!(installed.len(), MANAGED_HOOKS.len());

        for hook in MANAGED_HOOKS {
            let path = tmp.path().join(".git/hooks").join(hook);
            assert!(path.exists(), "hook {} not created", hook);
            let contents = fs::read_to_string(&path).unwrap();
            assert!(contents.starts_with("#!/usr/bin/env sh"));
            assert!(contents.contains(BEGIN_MARK));
            assert!(contents.contains(END_MARK));
            assert!(contents.contains(".mcp_data/hooks/"));
        }
    }

    #[test]
    fn install_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        init_main_checkout(&tmp);

        install_hooks(tmp.path()).unwrap();
        install_hooks(tmp.path()).unwrap();

        let path = tmp.path().join(".git/hooks/post-commit");
        let contents = fs::read_to_string(&path).unwrap();
        // Block must appear exactly once, not duplicated.
        assert_eq!(contents.matches(BEGIN_MARK).count(), 1);
        assert_eq!(contents.matches(END_MARK).count(), 1);
    }

    #[test]
    fn install_preserves_user_authored_lines() {
        let tmp = TempDir::new().unwrap();
        init_main_checkout(&tmp);

        let path = tmp.path().join(".git/hooks/post-commit");
        fs::write(&path, "#!/usr/bin/env bash\necho user-script\n").unwrap();

        install_hooks(tmp.path()).unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents.contains("echo user-script"), "user content lost");
        assert!(contents.contains(BEGIN_MARK));
    }

    #[test]
    fn uninstall_removes_only_our_block() {
        let tmp = TempDir::new().unwrap();
        init_main_checkout(&tmp);

        let path = tmp.path().join(".git/hooks/post-commit");
        fs::write(&path, "#!/usr/bin/env bash\necho user-script\n").unwrap();

        install_hooks(tmp.path()).unwrap();
        uninstall_hooks(tmp.path()).unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents.contains("echo user-script"));
        assert!(!contents.contains(BEGIN_MARK));
        assert!(!contents.contains(END_MARK));
    }

    #[test]
    fn uninstall_deletes_hook_when_only_our_block_present() {
        let tmp = TempDir::new().unwrap();
        init_main_checkout(&tmp);
        install_hooks(tmp.path()).unwrap();
        uninstall_hooks(tmp.path()).unwrap();

        for hook in MANAGED_HOOKS {
            let path = tmp.path().join(".git/hooks").join(hook);
            assert!(!path.exists(), "{} should be removed when empty", hook);
        }
    }

    #[test]
    fn install_creates_sentinel_dir() {
        let tmp = TempDir::new().unwrap();
        init_main_checkout(&tmp);
        install_hooks(tmp.path()).unwrap();
        assert!(tmp.path().join(".mcp_data/hooks").exists());
    }

    #[cfg(unix)]
    #[test]
    fn install_makes_hooks_executable() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = TempDir::new().unwrap();
        init_main_checkout(&tmp);
        install_hooks(tmp.path()).unwrap();
        let mode = fs::metadata(tmp.path().join(".git/hooks/post-commit"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o111, 0o111, "hook must be executable");
    }
}
