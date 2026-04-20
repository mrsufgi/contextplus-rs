// Path validation to prevent directory traversal attacks.
// All user-supplied paths must be resolved through resolve_safe_path before use.

use std::path::{Path, PathBuf};

use crate::error::{ContextPlusError, Result};

/// Resolve a user-supplied path against root, ensuring it stays within root.
///
/// Rejects paths that attempt directory traversal (e.g., `../../etc/passwd`).
/// Returns an absolute `PathBuf` within `root`.
///
/// Round 9C P0: uses `std::fs::canonicalize` on the root and
/// `std::env::current_dir()` with a proper error (no `unwrap_or_default()`).
pub fn resolve_safe_path(root: &Path, user_path: &str) -> Result<PathBuf> {
    // Resolve the root to an absolute path.
    //
    // We canonicalize the root to resolve symlinks — without this, a symlinked
    // root (e.g. /tmp/link -> /etc) would let a caller write through the link
    // while the final `starts_with` check compared only the uncanonicalized
    // prefix. Only NotFound is tolerated as a "root doesn't exist yet" case;
    // any other IO error (permission denied, broken symlink, etc.) propagates
    // so we never silently fall through to the weaker raw-prefix comparison.
    let canonical_root = if root.is_absolute() {
        match root.canonicalize() {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => root.to_path_buf(),
            Err(e) => {
                return Err(ContextPlusError::Io(std::io::Error::new(
                    e.kind(),
                    format!("failed to canonicalize root '{}': {}", root.display(), e),
                )));
            }
        }
    } else {
        let cwd = std::env::current_dir().map_err(|e| {
            ContextPlusError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to resolve current directory: {}", e),
            ))
        })?;
        let joined = cwd.join(root);
        match joined.canonicalize() {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => joined,
            Err(e) => {
                return Err(ContextPlusError::Io(std::io::Error::new(
                    e.kind(),
                    format!("failed to canonicalize root '{}': {}", joined.display(), e),
                )));
            }
        }
    };

    // Build the candidate path by joining root with the user-supplied path.
    // `Path::join` with an absolute component replaces the base — guard against this.
    let user = Path::new(user_path);
    if user.is_absolute() {
        return Err(ContextPlusError::PathTraversal(format!(
            "absolute paths are not allowed: {}",
            user_path
        )));
    }

    let candidate = canonical_root.join(user_path);

    // Canonicalize the candidate if it exists; otherwise normalise manually.
    let resolved = if candidate.exists() {
        candidate.canonicalize().map_err(|e| {
            ContextPlusError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to canonicalize path: {}", e),
            ))
        })?
    } else {
        // File doesn't exist yet — normalise by collapsing `.` and `..`.
        normalize_path(&candidate)
    };

    // The resolved path must start with the canonical root.
    if !resolved.starts_with(&canonical_root) {
        return Err(ContextPlusError::PathTraversal(format!(
            "path traversal detected: '{}' escapes root '{}'",
            user_path,
            canonical_root.display()
        )));
    }

    Ok(resolved)
}

/// Normalise a path without requiring it to exist on disk.
/// Collapses `.` and `..` components. Does not resolve symlinks.
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::CurDir => {} // skip `.`
            Component::ParentDir => {
                // Pop last non-root component; never go above root
                if matches!(components.last(), Some(Component::Normal(_))) {
                    components.pop();
                }
            }
            other => components.push(other),
        }
    }
    components.iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_root() -> TempDir {
        TempDir::new().expect("tempdir")
    }

    /// Canonicalize the TempDir path to handle macOS /tmp -> /private/tmp symlink.
    fn canonical_root(dir: &TempDir) -> PathBuf {
        dir.path().canonicalize().unwrap()
    }

    #[test]
    fn resolve_simple_relative_path() {
        let dir = make_root();
        let root = canonical_root(&dir);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();

        let resolved = resolve_safe_path(&root, "src/main.rs").unwrap();
        assert!(resolved.starts_with(&root));
        assert!(resolved.ends_with("main.rs"));
    }

    #[test]
    fn resolve_nonexistent_path_within_root() {
        let dir = make_root();
        let root = canonical_root(&dir);

        let resolved = resolve_safe_path(&root, "new_file.rs").unwrap();
        assert!(resolved.starts_with(&root));
    }

    #[test]
    fn rejects_path_traversal() {
        let dir = make_root();
        let root = canonical_root(&dir);

        let result = resolve_safe_path(&root, "../../etc/passwd");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ContextPlusError::PathTraversal(_)));
    }

    #[test]
    fn rejects_absolute_path() {
        let dir = make_root();
        let root = canonical_root(&dir);

        let result = resolve_safe_path(&root, "/etc/passwd");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ContextPlusError::PathTraversal(_)
        ));
    }

    #[test]
    fn allows_nested_path() {
        let dir = make_root();
        let root = canonical_root(&dir);

        let resolved = resolve_safe_path(&root, "a/b/c/d.ts").unwrap();
        assert!(resolved.starts_with(&root));
    }

    #[test]
    fn normalizes_dotdot_that_stays_inside() {
        let dir = make_root();
        let root = canonical_root(&dir);
        std::fs::create_dir_all(root.join("src/core")).unwrap();
        std::fs::write(root.join("src/core/lib.rs"), "").unwrap();

        // src/core/../core/lib.rs -> src/core/lib.rs — still inside root
        let resolved = resolve_safe_path(&root, "src/core/../core/lib.rs").unwrap();
        assert!(resolved.starts_with(&root));
    }
}
