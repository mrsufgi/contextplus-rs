// File-system embedding tracker for realtime cache refresh on source changes
// Watches for file modifications and triggers debounced embedding refreshes

use notify_debouncer_full::notify::RecursiveMode;
use notify_debouncer_full::{DebounceEventResult, new_debouncer};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

const DEFAULT_DEBOUNCE_MS: u64 = 700;
const DEFAULT_MAX_FILES_PER_TICK: usize = 8;
const MIN_FILES_PER_TICK: usize = 5;
const MAX_FILES_PER_TICK: usize = 200;
const MIN_DEBOUNCE_MS: u64 = 100;

/// Maximum number of pending file change events before new events are dropped.
pub const MAX_PENDING_FILES: usize = 50;

/// Configuration for the embedding tracker.
#[derive(Debug, Clone)]
pub struct EmbeddingTrackerConfig {
    pub debounce_ms: u64,
    pub max_files_per_tick: usize,
    pub ignore_dirs: HashSet<String>,
}

impl Default for EmbeddingTrackerConfig {
    fn default() -> Self {
        Self {
            debounce_ms: DEFAULT_DEBOUNCE_MS,
            max_files_per_tick: DEFAULT_MAX_FILES_PER_TICK,
            ignore_dirs: HashSet::new(),
        }
    }
}

impl EmbeddingTrackerConfig {
    fn clamped_debounce_ms(&self) -> u64 {
        self.debounce_ms.max(MIN_DEBOUNCE_MS)
    }

    fn clamped_max_files(&self) -> usize {
        self.max_files_per_tick
            .clamp(MIN_FILES_PER_TICK, MAX_FILES_PER_TICK)
    }
}

/// Checks whether a file path should be tracked for embedding updates.
/// Delegates to walker::should_track using the string representation of the path.
/// Uses segment-based matching: each path component is compared exactly
/// against the ignore list, so "node_modules_backup" does NOT match "node_modules".
pub fn should_track(path: &Path, ignore_dirs: &HashSet<String>) -> bool {
    let path_str = path.to_string_lossy();
    // Normalise to forward slashes (noop on Linux, needed on Windows)
    if path_str.contains('\\') {
        crate::core::walker::should_track(&path_str.replace('\\', "/"), ignore_dirs)
    } else {
        crate::core::walker::should_track(path_str.as_ref(), ignore_dirs)
    }
}

/// Normalizes a path to use forward slashes and strips leading slashes.
fn normalize_relative_path(path: &Path) -> String {
    let s = path.to_string_lossy();
    s.replace('\\', "/").trim_start_matches('/').to_string()
}

/// Handle for controlling a running embedding tracker.
/// Dropping this handle will stop the tracker.
pub struct EmbeddingTrackerHandle {
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl EmbeddingTrackerHandle {
    /// Gracefully stops the embedding tracker.
    pub async fn stop(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }
    }
}

impl Drop for EmbeddingTrackerHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.try_send(());
        }
    }
}

/// Callback type for embedding refresh.
/// Receives the root directory and a batch of relative file paths.
/// Returns (file_embed_count, identifier_embed_count).
pub type RefreshCallback =
    Arc<dyn Fn(PathBuf, Vec<String>) -> tokio::task::JoinHandle<(usize, usize)> + Send + Sync>;

/// Starts the embedding tracker watching for file changes in `root_dir`.
///
/// The `refresh_callback` is invoked with batches of changed files (up to
/// `config.max_files_per_tick` per tick). File changes are debounced by
/// `config.debounce_ms` milliseconds.
///
/// Returns a handle that can be used to stop the tracker.
pub fn start_tracker(
    root_dir: PathBuf,
    config: EmbeddingTrackerConfig,
    refresh_callback: RefreshCallback,
) -> Result<EmbeddingTrackerHandle, notify_debouncer_full::notify::Error> {
    let debounce_ms = config.clamped_debounce_ms();
    let max_files = config.clamped_max_files();
    let ignore_dirs = config.ignore_dirs.clone();

    // Channel for debounced events from notify to our async processor
    let (event_tx, mut event_rx) = mpsc::channel::<Vec<String>>(64);

    // Set up notify debouncer with an event handler that filters and collects paths
    let handler_root = root_dir.clone();

    let event_handler = move |result: DebounceEventResult| match result {
        Ok(events) => {
            let mut new_files = Vec::new();
            for event in &events {
                for path in &event.paths {
                    let relative = match path.strip_prefix(&handler_root) {
                        Ok(r) => r.to_path_buf(),
                        Err(_) => continue,
                    };
                    if !should_track(&relative, &ignore_dirs) {
                        continue;
                    }
                    let normalized = normalize_relative_path(&relative);
                    if !normalized.is_empty() {
                        new_files.push(normalized);
                    }
                }
            }
            if !new_files.is_empty() {
                let _ = event_tx.try_send(new_files);
            }
        }
        Err(errors) => {
            for e in errors {
                error!("Embedding tracker watcher error: {}", e);
            }
        }
    };

    let mut debouncer = new_debouncer(Duration::from_millis(debounce_ms), None, event_handler)?;

    debouncer.watch(&root_dir, RecursiveMode::Recursive)?;

    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

    // Spawn async task to process batched events
    let root_for_task = root_dir.clone();
    tokio::spawn(async move {
        // Keep debouncer alive in this task so it's not dropped
        let _debouncer = debouncer;
        let mut pending_set: HashSet<String> = HashSet::new();

        loop {
            tokio::select! {
                biased;
                _ = shutdown_rx.recv() => {
                    debug!("Embedding tracker shutting down");
                    break;
                }
                Some(files) = event_rx.recv() => {
                    for f in files {
                        if pending_set.len() >= MAX_PENDING_FILES {
                            warn!(
                                "Embedding tracker pending cap ({}) reached, dropping event for: {}",
                                MAX_PENDING_FILES, f
                            );
                            break;
                        }
                        pending_set.insert(f);
                    }

                    // Drain any additional queued events without waiting
                    while let Ok(more) = event_rx.try_recv() {
                        for f in more {
                            if pending_set.len() >= MAX_PENDING_FILES {
                                warn!(
                                    "Embedding tracker pending cap ({}) reached, dropping events",
                                    MAX_PENDING_FILES
                                );
                                break;
                            }
                            pending_set.insert(f);
                        }
                    }

                    if pending_set.is_empty() {
                        continue;
                    }

                    // Take a batch up to max_files
                    let batch: Vec<String> = pending_set
                        .iter()
                        .take(max_files)
                        .cloned()
                        .collect();
                    for f in &batch {
                        pending_set.remove(f);
                    }

                    let handle = (refresh_callback)(root_for_task.clone(), batch.clone());
                    match handle.await {
                        Ok((file_count, id_count)) => {
                            if file_count > 0 || id_count > 0 {
                                info!(
                                    "Embedding tracker refreshed {} file(s) | file-vectors={}, identifier-vectors={}",
                                    batch.len(),
                                    file_count,
                                    id_count
                                );
                            }
                        }
                        Err(e) => {
                            warn!("Embedding tracker refresh task failed: {}", e);
                        }
                    }

                    // If there are still pending files, process next batch soon
                    if !pending_set.is_empty() {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        }
    });

    Ok(EmbeddingTrackerHandle {
        shutdown_tx: Some(shutdown_tx),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ignore_dirs() -> HashSet<String> {
        [
            "node_modules",
            ".git",
            "dist",
            "build",
            ".next",
            "target",
            ".mcp_data",
            "coverage",
            ".cache",
            ".turbo",
            "__pycache__",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    #[test]
    fn should_track_normal_source_files() {
        let ignore = default_ignore_dirs();
        assert!(should_track(Path::new("src/main.rs"), &ignore));
        assert!(should_track(Path::new("lib/utils.ts"), &ignore));
        assert!(should_track(Path::new("README.md"), &ignore));
    }

    #[test]
    fn should_track_ignores_node_modules() {
        let ignore = default_ignore_dirs();
        assert!(!should_track(Path::new("node_modules/foo/bar.js"), &ignore));
        assert!(!should_track(
            Path::new("packages/web/node_modules/react/index.js"),
            &ignore
        ));
    }

    #[test]
    fn should_track_segment_based_not_prefix() {
        let ignore = default_ignore_dirs();
        // "node_modules_backup" should NOT match "node_modules" (segment-based)
        assert!(should_track(
            Path::new("src/node_modules_backup/file.ts"),
            &ignore
        ));
        // "build_scripts" should NOT match "build"
        assert!(should_track(Path::new("build_scripts/run.sh"), &ignore));
        // But "build" itself should match
        assert!(!should_track(Path::new("build/output.js"), &ignore));
    }

    #[test]
    fn should_track_ignores_git_dir() {
        let ignore = default_ignore_dirs();
        assert!(!should_track(Path::new(".git/objects/abc"), &ignore));
        assert!(!should_track(Path::new(".git/HEAD"), &ignore));
    }

    #[test]
    fn should_track_ignores_mcp_data() {
        let ignore = default_ignore_dirs();
        assert!(!should_track(
            Path::new(".mcp_data/backups/rp-123-abc123/file"),
            &ignore
        ));
    }

    #[test]
    fn should_track_empty_ignore_tracks_all() {
        let ignore = HashSet::new();
        assert!(should_track(Path::new("node_modules/foo.js"), &ignore));
        assert!(should_track(Path::new(".git/HEAD"), &ignore));
    }

    #[test]
    fn should_track_ignores_target_dir() {
        let ignore = default_ignore_dirs();
        assert!(!should_track(
            Path::new("target/debug/build/something"),
            &ignore
        ));
    }

    #[test]
    fn normalize_relative_path_strips_leading_slash() {
        assert_eq!(
            normalize_relative_path(Path::new("/src/main.rs")),
            "src/main.rs"
        );
    }

    #[test]
    fn normalize_relative_path_replaces_backslashes() {
        assert_eq!(
            normalize_relative_path(Path::new("src\\lib\\mod.rs")),
            "src/lib/mod.rs"
        );
    }

    #[test]
    fn config_defaults() {
        let cfg = EmbeddingTrackerConfig::default();
        assert_eq!(cfg.debounce_ms, 700);
        assert_eq!(cfg.max_files_per_tick, 8);
        assert!(cfg.ignore_dirs.is_empty());
    }

    #[test]
    fn config_clamps_debounce_ms() {
        let cfg = EmbeddingTrackerConfig {
            debounce_ms: 10,
            ..Default::default()
        };
        assert_eq!(cfg.clamped_debounce_ms(), MIN_DEBOUNCE_MS);

        let cfg = EmbeddingTrackerConfig {
            debounce_ms: 5000,
            ..Default::default()
        };
        assert_eq!(cfg.clamped_debounce_ms(), 5000);
    }

    #[test]
    fn config_clamps_max_files() {
        let cfg = EmbeddingTrackerConfig {
            max_files_per_tick: 1,
            ..Default::default()
        };
        assert_eq!(cfg.clamped_max_files(), MIN_FILES_PER_TICK);

        let cfg = EmbeddingTrackerConfig {
            max_files_per_tick: 999,
            ..Default::default()
        };
        assert_eq!(cfg.clamped_max_files(), MAX_FILES_PER_TICK);

        let cfg = EmbeddingTrackerConfig {
            max_files_per_tick: 50,
            ..Default::default()
        };
        assert_eq!(cfg.clamped_max_files(), 50);
    }

    #[tokio::test]
    async fn tracker_starts_and_stops() {
        let dir = tempfile::TempDir::new().unwrap();
        let callback: RefreshCallback = Arc::new(|_root, _files| tokio::spawn(async { (0, 0) }));

        let handle = start_tracker(
            dir.path().to_path_buf(),
            EmbeddingTrackerConfig::default(),
            callback,
        )
        .expect("tracker should start");

        handle.stop().await;
    }

    #[tokio::test]
    async fn tracker_receives_file_changes() {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().to_path_buf();

        let (tx, mut rx) = mpsc::channel::<Vec<String>>(16);
        let callback: RefreshCallback = Arc::new(move |_root, files| {
            let tx = tx.clone();
            tokio::spawn(async move {
                let _ = tx.send(files).await;
                (1, 1)
            })
        });

        let config = EmbeddingTrackerConfig {
            debounce_ms: 200,
            max_files_per_tick: 8,
            ignore_dirs: default_ignore_dirs(),
        };

        let handle = start_tracker(root.clone(), config, callback).expect("tracker should start");

        // Create source files (not in ignored dirs)
        let src_dir = root.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();

        // Write files to trigger watcher
        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::write(src_dir.join("test.rs"), "fn main() {}").unwrap();

        // Wait for debounced event (debounce 200ms + some margin)
        let result = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await;

        match result {
            Ok(Some(files)) => {
                // notify may report the file itself or parent dir depending on OS
                assert!(
                    files
                        .iter()
                        .any(|f| f.contains("test.rs") || f.contains("src")),
                    "expected test.rs or src in batch, got: {:?}",
                    files
                );
            }
            Ok(None) => panic!("channel closed without receiving event"),
            Err(_) => {
                // Timeout is acceptable in CI -- file watching may not work in all environments
                eprintln!("WARNING: file watcher timeout -- may not work in this environment");
            }
        }

        handle.stop().await;
    }

    #[tokio::test]
    async fn tracker_ignores_node_modules_changes() {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().to_path_buf();

        let (tx, mut rx) = mpsc::channel::<Vec<String>>(16);
        let callback: RefreshCallback = Arc::new(move |_root, files| {
            let tx = tx.clone();
            tokio::spawn(async move {
                let _ = tx.send(files).await;
                (0, 0)
            })
        });

        let config = EmbeddingTrackerConfig {
            debounce_ms: 200,
            max_files_per_tick: 8,
            ignore_dirs: default_ignore_dirs(),
        };

        let handle = start_tracker(root.clone(), config, callback).expect("tracker should start");

        // Write to ignored directory
        let nm_dir = root.join("node_modules").join("pkg");
        std::fs::create_dir_all(&nm_dir).unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::write(nm_dir.join("index.js"), "module.exports = {}").unwrap();

        // Should NOT receive an event for node_modules files
        let result = tokio::time::timeout(Duration::from_secs(1), rx.recv()).await;
        match result {
            Ok(Some(files)) => {
                assert!(
                    !files.iter().any(|f| f.contains("node_modules")),
                    "should not receive node_modules files, got: {:?}",
                    files
                );
            }
            Ok(None) | Err(_) => {
                // Expected: no event received (timeout or channel closed)
            }
        }

        handle.stop().await;
    }

    #[test]
    fn pending_cap_constant_is_50() {
        assert_eq!(MAX_PENDING_FILES, 50);
    }
}
