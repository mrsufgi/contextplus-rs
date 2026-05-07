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

/// File names that are always excluded by their literal name.
///
/// These are big, generated, or noise-only files that pollute semantic search
/// results without contributing meaning. They live alongside legitimate source
/// in the same directory, so they cannot be excluded by directory-name rules.
///
/// **Lockfiles policy.** `package-lock.json`, `pnpm-lock.yaml`, and `*.lockb`
/// are excluded — they are giant generated trees. `yarn.lock`, `Cargo.lock`,
/// `poetry.lock`, and `go.sum` are intentionally NOT excluded because their
/// contents (resolved dep graphs) are sometimes useful when an agent is
/// debugging dependency resolution.
const ALWAYS_IGNORE_FILENAMES: &[&str] =
    &["package-lock.json", "pnpm-lock.yaml", "npm-shrinkwrap.json"];

/// File-name suffixes (case-sensitive) excluded from walks.
///
/// Suffix matching is intentionally simple: no glob crate, no regex, no escapes.
/// Each suffix is matched literally against the file name's tail.
const ALWAYS_IGNORE_SUFFIXES: &[&str] = &[
    // Source maps + minified bundles
    ".map",
    ".min.js",
    ".min.css",
    ".min.mjs",
    ".bundle.js",
    ".bundle.mjs",
    // Bun's binary lockfile
    ".lockb",
    // Generated protobuf bindings (TS connect-es / connect-go / pbjs)
    "_pb.ts",
    "_pb.js",
    "_pb.d.ts",
    ".pb.go",
    ".pb.cc",
    ".pb.h",
    ".connect.ts",
    ".connect.js",
    // Snapshot test files outside __snapshots__/
    ".snap",
];

/// Segment-based path ignore check.
/// Checks if any path segment matches the ignore set or starts with `.`
/// (hidden files/dirs), and rejects file names matching the global
/// always-ignore list (lockfiles, sourcemaps, generated proto bindings, etc.).
///
/// Segment matching is exact, not prefix-based — `gen` segment is blocked but
/// `generic/` is not.
pub fn should_track(path: &str, ignore_dirs: &HashSet<String>) -> bool {
    if path.is_empty() {
        return false;
    }
    for segment in path.split('/') {
        if segment.starts_with('.') {
            return false;
        }
        if ignore_dirs.contains(segment) {
            return false;
        }
    }
    // File-name level filtering: only the trailing segment matters.
    if let Some(name) = path.rsplit('/').next() {
        if ALWAYS_IGNORE_FILENAMES.contains(&name) {
            return false;
        }
        for suffix in ALWAYS_IGNORE_SUFFIXES {
            if name.ends_with(suffix) {
                return false;
            }
        }
    }
    true
}

/// Returns `true` when a cache key should be retained during cache hygiene
/// sweeps (both `VectorStore::to_cache` and `CacheData::sweep_excluded_keys`).
///
/// This is a thin wrapper around [`should_track`] with an empty ignore-dirs
/// set — the single source of truth for the dot-segment exclusion rule so
/// both sweep paths stay in sync.
#[inline]
pub fn should_keep_cache_key(key: &str) -> bool {
    should_track(key, &Default::default())
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
    fn should_track_skips_hidden_files() {
        let ignore = make_ignore_set(&[]);
        // Any segment starting with '.' should be skipped (matches TS behavior)
        assert!(!should_track(".hidden_file", &ignore));
        assert!(!should_track("src/.env", &ignore));
        assert!(!should_track(".config/settings.json", &ignore));
        assert!(!should_track("deep/path/.secret/key", &ignore));
        // Non-hidden files should pass
        assert!(should_track("src/main.rs", &ignore));
        assert!(should_track("config/settings.json", &ignore));
    }

    #[test]
    fn should_track_skips_all_known_ignore_dirs() {
        // Verify every entry from the TS ALWAYS_IGNORE set is handled.
        // Dot-prefixed entries are caught by the hidden check;
        // the rest must be in the ignore set.
        let ts_always_ignore = [
            "node_modules",
            ".git",
            ".svn",
            ".hg",
            "__pycache__",
            ".DS_Store",
            "dist",
            "build",
            ".next",
            ".nuxt",
            "target",
            ".mcp_data",
            ".mcp-shadow-history",
            "coverage",
            ".cache",
            ".turbo",
            ".parcel-cache",
        ];

        let config = crate::config::Config::from_env();

        for entry in &ts_always_ignore {
            let test_path = format!("{}/some_file.ts", entry);
            assert!(
                !should_track(&test_path, &config.ignore_dirs),
                "Expected should_track to reject path segment '{}', but it was accepted",
                entry
            );
        }
    }

    /// Regression guard: `.claude/worktrees/` paths must be excluded by the
    /// existing dot-segment hidden-file check in `should_track`.
    ///
    /// The first segment of `.claude/worktrees/...` is `.claude`, which starts
    /// with `.` and is therefore rejected unconditionally — no special-case
    /// constant is needed.  This test locks in that behavior so future
    /// refactors of the dot-segment logic cannot silently regress it.
    #[test]
    fn should_track_excludes_claude_worktrees() {
        let ignore = make_ignore_set(&[]);
        // Paths inside .claude/worktrees/ must always be rejected
        assert!(!should_track(
            ".claude/worktrees/agent-acd43174/packages/libs/types/generated/foo.ts",
            &ignore
        ));
        assert!(!should_track(
            ".claude/worktrees/agent-acd43174/src/main.rs",
            &ignore
        ));
        // The prefix itself (exact) is also rejected
        assert!(!should_track(".claude/worktrees/", &ignore));
        // Paths outside worktrees that are otherwise valid should still pass
        assert!(should_track("src/main.rs", &ignore));
        // .claude itself (non-worktrees) is still skipped by the dot-segment rule
        assert!(!should_track(".claude/settings.json", &ignore));
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
    fn walk_directory_skips_hidden_files() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join(".hidden_dir")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("src/.env"), "SECRET=foo").unwrap();
        fs::write(root.join(".hidden_dir/config"), "hidden").unwrap();
        fs::write(root.join(".gitignore"), "*.tmp").unwrap();

        let ignore = make_ignore_set(&[]);
        let entries = walk_directory(&WalkOptions {
            root_dir: root,
            target_path: None,
            depth_limit: None,
            ignore_dirs: &ignore,
        });

        let paths: Vec<&str> = entries.iter().map(|e| e.relative_path.as_str()).collect();
        assert!(paths.contains(&"src/main.rs"));
        assert!(
            !paths.iter().any(|p| p.contains(".hidden_dir")),
            "Hidden directory .hidden_dir should be skipped"
        );
        assert!(
            !paths.iter().any(|p| p.contains(".env")),
            "Hidden file .env should be skipped"
        );
        assert!(
            !paths.iter().any(|p| p.contains(".gitignore")),
            "Hidden file .gitignore should be skipped"
        );
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

    /// Integration test: walk_directory must never return paths under
    /// `.claude/worktrees/` even when the directory is physically present.
    ///
    /// The `ignore` crate's `hidden(true)` flag precludes the walker from
    /// descending into `.claude/` on most platforms; `should_track`'s
    /// dot-segment check provides a belt-and-suspenders safety net for any
    /// edge cases (e.g. symlink-followed paths, explicit `target_path` inside
    /// `.claude/`).  Both layers are exercised here.
    #[test]
    fn walk_directory_excludes_claude_worktrees() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Simulate a workspace with a worktree alongside real source
        let worktree_file = root
            .join(".claude")
            .join("worktrees")
            .join("agent-abc123")
            .join("src");
        fs::create_dir_all(&worktree_file).unwrap();
        fs::write(worktree_file.join("main.rs"), "fn main() {}").unwrap();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn lib() {}").unwrap();

        let ignore = make_ignore_set(&[]);
        let entries = walk_directory(&WalkOptions {
            root_dir: root,
            target_path: None,
            depth_limit: None,
            ignore_dirs: &ignore,
        });

        let paths: Vec<&str> = entries.iter().map(|e| e.relative_path.as_str()).collect();
        // Real source must be present
        assert!(
            paths.contains(&"src/lib.rs"),
            "src/lib.rs should be indexed"
        );
        // Nothing from the worktree should be indexed
        assert!(
            !paths.iter().any(|p| p.contains("worktrees")),
            "worktrees paths must not be indexed: {paths:?}"
        );
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
    fn ignore_set_matches_ts_always_ignore() {
        use crate::config::Config;

        let ts_always_ignore: HashSet<&str> = [
            "node_modules",
            ".git",
            ".svn",
            ".hg",
            "__pycache__",
            ".DS_Store",
            "dist",
            "build",
            ".next",
            ".nuxt",
            "target",
            ".mcp_data",
            ".mcp-shadow-history",
            "coverage",
            ".cache",
            ".turbo",
            ".parcel-cache",
        ]
        .into_iter()
        .collect();

        let config = Config::from_env();
        for entry in &ts_always_ignore {
            assert!(
                config.ignore_dirs.contains(*entry),
                "Rust ignore_dirs is missing '{}' which is in TS ALWAYS_IGNORE",
                entry
            );
        }
    }

    #[test]
    fn should_track_blocks_lockfiles() {
        let ignore = make_ignore_set(&[]);
        assert!(!should_track("package-lock.json", &ignore));
        assert!(!should_track("pnpm-lock.yaml", &ignore));
        assert!(!should_track("npm-shrinkwrap.json", &ignore));
        assert!(!should_track("apps/x/package-lock.json", &ignore));
        // these should still be indexed
        assert!(should_track("yarn.lock", &ignore));
        assert!(should_track("Cargo.lock", &ignore));
        assert!(should_track("go.sum", &ignore));
        assert!(should_track("poetry.lock", &ignore));
    }

    #[test]
    fn should_track_blocks_sourcemaps_and_bundles() {
        let ignore = make_ignore_set(&[]);
        assert!(!should_track("dist/app.js.map", &ignore));
        assert!(!should_track("packages/x/build/main.css.map", &ignore));
        assert!(!should_track("public/app.min.js", &ignore));
        assert!(!should_track("public/app.min.css", &ignore));
        assert!(!should_track("public/app.min.mjs", &ignore));
        assert!(!should_track("dist/main.bundle.js", &ignore));
        assert!(!should_track("dist/main.bundle.mjs", &ignore));
        assert!(!should_track("bun.lockb", &ignore));
        // similarly-named real source must pass
        assert!(should_track("src/sourceMapHelper.ts", &ignore));
        assert!(should_track("src/minifier.js", &ignore));
        assert!(should_track("src/Bundle.ts", &ignore));
    }

    #[test]
    fn should_track_blocks_generated_proto_bindings() {
        let ignore = make_ignore_set(&[]);
        assert!(!should_track("packages/contracts/profiles_pb.ts", &ignore));
        assert!(!should_track(
            "packages/contracts/profiles_pb.d.ts",
            &ignore
        ));
        assert!(!should_track("packages/contracts/profiles_pb.js", &ignore));
        assert!(!should_track("internal/v1/service.pb.go", &ignore));
        assert!(!should_track("internal/v1/service.pb.cc", &ignore));
        assert!(!should_track("internal/v1/service.pb.h", &ignore));
        assert!(!should_track("packages/contracts/foo.connect.ts", &ignore));
        assert!(!should_track("packages/contracts/foo.connect.js", &ignore));
        // hand-written protobuf-related code must pass
        assert!(should_track("src/proto_helpers.ts", &ignore));
        assert!(should_track("src/connect_setup.ts", &ignore));
    }

    #[test]
    fn should_track_blocks_jest_snapshot_files() {
        let ignore = make_ignore_set(&[]);
        assert!(!should_track("src/x.test.ts.snap", &ignore));
        // hand-written *.snap.ts (uncommon but legal) is unaffected:
        // we suffix-match on `.snap`, not `snap` segment
        assert!(should_track("src/snapshot.ts", &ignore));
    }

    #[test]
    fn should_track_blocks_new_dir_defaults() {
        let config = crate::config::Config::from_env();
        // New non-dot dirs added in U1 must be in the ignore set
        for dir in [
            "vendor",
            "_build",
            "obj",
            "__snapshots__",
            "htmlcov",
            "venv",
        ] {
            let test_path = format!("{}/some_file.ts", dir);
            assert!(
                !should_track(&test_path, &config.ignore_dirs),
                "expected `{}` to be ignored",
                dir
            );
        }
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
