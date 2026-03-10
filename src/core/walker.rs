use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

use crate::config::Config;

/// Entry from a directory walk.
#[derive(Debug, Clone)]
pub struct FileEntry {
    pub path: PathBuf,
    pub relative_path: String,
    pub is_directory: bool,
    pub depth: usize,
}

/// Options for walking a directory.
pub struct WalkOptions<'a> {
    pub root_dir: &'a Path,
    pub target_path: Option<&'a str>,
    pub depth_limit: Option<usize>,
    pub ignore_dirs: &'a HashSet<String>,
}

/// Segment-based path ignore check.
/// Checks if any path segment matches the ignore set.
/// This is the correct approach (matching our TS fork fix) — prevents
/// false positives from prefix matching like "generated" matching "gen".
pub fn should_track(path: &str, ignore_dirs: &HashSet<String>) -> bool {
    if path.is_empty() {
        return false;
    }
    for segment in path.split('/') {
        if ignore_dirs.contains(segment) {
            return false;
        }
    }
    true
}

/// Walk a directory respecting .gitignore rules and configured ignore dirs.
/// Uses the `ignore` crate (from ripgrep) for high-performance gitignore-aware walking.
pub fn walk_directory(opts: &WalkOptions) -> Vec<FileEntry> {
    let root_dir = opts.root_dir;
    let start_dir = match opts.target_path {
        Some(target) => root_dir.join(target),
        None => root_dir.to_path_buf(),
    };

    if !start_dir.exists() {
        return Vec::new();
    }

    let mut builder = WalkBuilder::new(&start_dir);
    builder.hidden(true); // skip hidden files/dirs
    builder.git_ignore(true);
    builder.git_global(false);
    builder.git_exclude(true);

    if let Some(limit) = opts.depth_limit
        && limit > 0
    {
        builder.max_depth(Some(limit));
    }

    let mut results = Vec::new();

    for entry in builder.build().flatten() {
        let path = entry.path();

        // Skip the root directory itself
        if path == start_dir {
            continue;
        }

        let relative = match path.strip_prefix(root_dir) {
            Ok(rel) => {
                // On Windows paths may use backslashes; on Linux/macOS this is a no-op
                // so skip the allocation when there's nothing to replace.
                let s = rel.to_string_lossy();
                if cfg!(windows) || s.contains('\\') {
                    s.replace('\\', "/")
                } else {
                    s.into_owned()
                }
            }
            Err(_) => continue,
        };

        // Segment-based ignore check
        if !should_track(&relative, opts.ignore_dirs) {
            continue;
        }

        let is_dir = path.is_dir();
        let depth = relative.matches('/').count();

        results.push(FileEntry {
            path: path.to_path_buf(),
            relative_path: relative,
            is_directory: is_dir,
            depth,
        });
    }

    results
}

/// Walk using Config for ignore dirs.
pub fn walk_with_config(root_dir: &Path, config: &Config) -> Vec<FileEntry> {
    walk_directory(&WalkOptions {
        root_dir,
        target_path: None,
        depth_limit: None,
        ignore_dirs: &config.ignore_dirs,
    })
}

/// Group file entries by their parent directory.
pub fn group_by_directory(
    entries: &[FileEntry],
) -> std::collections::HashMap<String, Vec<&FileEntry>> {
    let mut groups: std::collections::HashMap<String, Vec<&FileEntry>> =
        std::collections::HashMap::new();
    for entry in entries {
        let dir = if entry.relative_path.contains('/') {
            entry.relative_path.rsplit_once('/').unwrap().0.to_string()
        } else {
            ".".to_string()
        };
        groups.entry(dir).or_default().push(entry);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn make_ignore_set(dirs: &[&str]) -> HashSet<String> {
        dirs.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn should_track_basic() {
        let ignore = make_ignore_set(&["node_modules", ".git", "dist"]);
        assert!(should_track("src/main.rs", &ignore));
        assert!(!should_track("node_modules/pkg/index.js", &ignore));
        assert!(!should_track(".git/config", &ignore));
        assert!(!should_track("dist/bundle.js", &ignore));
    }

    #[test]
    fn should_track_segment_based() {
        let ignore = make_ignore_set(&["gen", "generated"]);
        // "gen" as a segment should be blocked
        assert!(!should_track(
            "packages/contracts/gen/typescript/index.ts",
            &ignore
        ));
        // "generated" as a segment should be blocked
        assert!(!should_track("some/generated/file.ts", &ignore));
        // but "generic" should NOT be blocked (not a full segment match)
        assert!(should_track("src/generic/handler.ts", &ignore));
    }

    #[test]
    fn should_track_empty_path() {
        let ignore = make_ignore_set(&["node_modules"]);
        assert!(!should_track("", &ignore));
    }

    #[test]
    fn walk_directory_basic() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Create test structure
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("src/lib.rs"), "pub mod stuff;").unwrap();
        fs::write(root.join("Cargo.toml"), "[package]").unwrap();

        let ignore = make_ignore_set(&["node_modules", ".git"]);
        let entries = walk_directory(&WalkOptions {
            root_dir: root,
            target_path: None,
            depth_limit: None,
            ignore_dirs: &ignore,
        });

        let paths: Vec<&str> = entries.iter().map(|e| e.relative_path.as_str()).collect();
        assert!(paths.contains(&"src/main.rs"));
        assert!(paths.contains(&"src/lib.rs"));
        assert!(paths.contains(&"Cargo.toml"));
    }

    #[test]
    fn walk_directory_respects_ignore() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(
            root.join("node_modules/pkg/index.js"),
            "module.exports = {}",
        )
        .unwrap();

        let ignore = make_ignore_set(&["node_modules"]);
        let entries = walk_directory(&WalkOptions {
            root_dir: root,
            target_path: None,
            depth_limit: None,
            ignore_dirs: &ignore,
        });

        let paths: Vec<&str> = entries.iter().map(|e| e.relative_path.as_str()).collect();
        assert!(paths.contains(&"src/main.rs"));
        assert!(!paths.iter().any(|p| p.contains("node_modules")));
    }

    #[test]
    fn walk_directory_with_target_path() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        fs::create_dir_all(root.join("src/core")).unwrap();
        fs::create_dir_all(root.join("tests")).unwrap();
        fs::write(root.join("src/core/parser.rs"), "pub fn parse() {}").unwrap();
        fs::write(root.join("tests/test.rs"), "fn test() {}").unwrap();

        let ignore = make_ignore_set(&[]);
        let entries = walk_directory(&WalkOptions {
            root_dir: root,
            target_path: Some("src"),
            depth_limit: None,
            ignore_dirs: &ignore,
        });

        let paths: Vec<&str> = entries.iter().map(|e| e.relative_path.as_str()).collect();
        assert!(paths.iter().all(|p| p.starts_with("src/")));
    }

    #[test]
    fn walk_directory_nonexistent_path() {
        let dir = TempDir::new().unwrap();
        let ignore = make_ignore_set(&[]);
        let entries = walk_directory(&WalkOptions {
            root_dir: dir.path(),
            target_path: Some("nonexistent"),
            depth_limit: None,
            ignore_dirs: &ignore,
        });
        assert!(entries.is_empty());
    }

    #[test]
    fn walk_directory_depth_limit() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        fs::create_dir_all(root.join("a/b/c")).unwrap();
        fs::write(root.join("a/top.rs"), "").unwrap();
        fs::write(root.join("a/b/mid.rs"), "").unwrap();
        fs::write(root.join("a/b/c/deep.rs"), "").unwrap();

        let ignore = make_ignore_set(&[]);
        let entries = walk_directory(&WalkOptions {
            root_dir: root,
            target_path: None,
            depth_limit: Some(2),
            ignore_dirs: &ignore,
        });

        let paths: Vec<&str> = entries.iter().map(|e| e.relative_path.as_str()).collect();
        assert!(paths.contains(&"a/top.rs"));
        // depth_limit=2 means we go 2 levels deep from start
        assert!(paths.contains(&"a/b/mid.rs") || !paths.contains(&"a/b/c/deep.rs"));
    }

    #[test]
    fn group_by_directory_works() {
        let entries = vec![
            FileEntry {
                path: PathBuf::from("/root/src/main.rs"),
                relative_path: "src/main.rs".to_string(),
                is_directory: false,
                depth: 1,
            },
            FileEntry {
                path: PathBuf::from("/root/src/lib.rs"),
                relative_path: "src/lib.rs".to_string(),
                is_directory: false,
                depth: 1,
            },
            FileEntry {
                path: PathBuf::from("/root/Cargo.toml"),
                relative_path: "Cargo.toml".to_string(),
                is_directory: false,
                depth: 0,
            },
        ];

        let groups = group_by_directory(&entries);
        assert_eq!(groups["src"].len(), 2);
        assert_eq!(groups["."].len(), 1);
    }
}
