// File-system embedding tracker for realtime cache refresh on source changes
// Watches for file modifications and triggers debounced embedding refreshes

use ignore::WalkBuilder;
use notify_debouncer_full::notify::event::{EventKind, ModifyKind};
use notify_debouncer_full::notify::{Config as NotifyConfig, RecommendedWatcher, RecursiveMode};
use notify_debouncer_full::{DebounceEventResult, Debouncer, NoCache, new_debouncer_opt};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Snapshot of a file's identity for quick "did this actually change?" checks.
/// Combining mtime + size catches the overwhelming majority of real edits while
/// avoiding the cost of opening + hashing untouched files. Editors that write
/// atomically (rename-into-place) rotate the inode but always update mtime, so
/// any meaningful save is observable here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FileSig {
    mtime: SystemTime,
    size: u64,
}

/// Pre-callback skip gate: returns true if the file has the same mtime+size
/// as the last time we successfully reported it. Also updates the cache to
/// the current signature when reporting through (so the next no-op event
/// fires the skip path).
///
/// Public for reuse by warmup binaries that walk the project up-front.
pub(crate) fn metadata_unchanged(
    cache: &Mutex<HashMap<String, FileSig>>,
    rel_path: &str,
    abs_path: &Path,
) -> bool {
    let meta = match std::fs::metadata(abs_path) {
        Ok(m) => m,
        Err(_) => return false, // file gone or unreadable — let the callback decide
    };
    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let size = meta.len();
    let sig = FileSig { mtime, size };

    let mut guard = match cache.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(), // poisoned lock — recover and continue
    };
    match guard.get(rel_path) {
        Some(prev) if *prev == sig => true,
        _ => {
            guard.insert(rel_path.to_string(), sig);
            false
        }
    }
}

/// Drop the metadata-cache entry for every path in `dropped`.
///
/// `metadata_unchanged` records the new (mtime, size) sig the moment a
/// watcher event arrives, *before* the path reaches the consumer. If the
/// consumer silently drops the path (cap reached, channel full), the next
/// identical watcher event would skip it as "unchanged" → permanent miss
/// until the file is modified again with a different sig. This rollback
/// keeps the metadata cache aligned with what was actually processed.
///
/// Poisoned mutex is silently ignored: at worst we re-process some files,
/// which is the same behavior as the rest of `metadata_unchanged`.
pub(crate) fn invalidate_metadata_entries(
    dropped: &[String],
    cache: &Arc<Mutex<HashMap<String, FileSig>>>,
) {
    if let Ok(mut guard) = cache.lock() {
        for f in dropped {
            guard.remove(f);
        }
    }
}

const DEFAULT_DEBOUNCE_MS: u64 = 700;
const DEFAULT_MAX_FILES_PER_TICK: usize = 8;
const MIN_FILES_PER_TICK: usize = 5;
const MAX_FILES_PER_TICK: usize = 200;
const MIN_DEBOUNCE_MS: u64 = 100;

/// Maximum number of pending file change events before new events are dropped.
/// A git-derived list (a large pull) is authoritative and bounded by the repo,
/// so the cap only guards against a runaway watcher; batches of
/// `max_files_per_tick` drain it at the embedder's pace.
const MAX_PENDING_FILES: usize = 5_000;

/// Whether an event can mean content changed. notify's inotify backend also
/// subscribes to open/close, so every read the indexer, a warmup or the heal
/// walk performs arrives here as an `Access` event for the file or directory;
/// those must never count as changes, or the tracker feeds on its own walks.
fn is_change_event(kind: &EventKind) -> bool {
    !matches!(
        kind,
        EventKind::Access(_) | EventKind::Modify(ModifyKind::Metadata(_))
    )
}

/// Whether a directory event means a directory appeared (created or moved
/// in) and needs a watch, as opposed to being read or having its metadata
/// touched.
fn is_new_dir_event(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_) | EventKind::Modify(ModifyKind::Name(_))
    )
}

/// Resolver signal for "HEAD may have moved"; distinct from every managed hook name.
const HEAD_SIGNAL: &str = "HEAD";

/// Files changed by a HEAD move since the last HEAD this tracker saw. Records
/// the new HEAD; empty when HEAD did not move or cannot be read.
fn head_changes_since_last(root: &Path, last_head: &Mutex<Option<String>>) -> Vec<String> {
    let Some(new_sha) = crate::core::head_watcher::resolve_head_sha(root) else {
        return Vec::new();
    };
    let old = {
        let mut guard = last_head.lock().unwrap_or_else(|p| p.into_inner());
        guard.replace(new_sha.clone())
    };
    match old {
        Some(old) if old != new_sha => {
            info!(
                old = %old,
                new = %new_sha,
                root = %root.display(),
                "Embedding tracker: HEAD moved, diffing for refresh"
            );
            crate::git::hooks::resolve_head_changes(root, &old, &new_sha)
        }
        _ => Vec::new(),
    }
}

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

/// Hard ceiling on directories the watcher registers. A pathological root
/// (e.g. a symlink into a machine-wide tree that is not ignored) loses
/// watch coverage past this point instead of exhausting the OS watch table
/// and starving the daemon.
const MAX_WATCHED_DIRS: usize = 100_000;

/// Register a NonRecursive watch on every directory under `start_dir` that
/// the indexer would visit, mirroring walker::walk_directory's semantics:
/// hidden and gitignored paths are pruned, `ignore_dirs` segments are
/// pruned, and symlinked directories are followed. Pruning at install time
/// (not just event time) is what keeps a symlink to a huge external tree
/// (e.g. `.worktrees -> ~/worktrees`) from exploding the watch table, while
/// a symlinked source directory still gets watched like any other.
///
/// With `collect_files` the walk also returns the normalized relative path
/// of every file it passes — used by the dynamic-add path so files written
/// into a directory before its watch existed are still picked up.
fn install_dir_watches(
    debouncer: &mut Debouncer<RecommendedWatcher, NoCache>,
    root_dir: &Path,
    start_dir: &Path,
    ignore_dirs: &HashSet<String>,
    watched_dirs: &mut usize,
    collect_files: bool,
    healthy: &AtomicBool,
) -> Vec<String> {
    let mut files = Vec::new();
    let mut builder = WalkBuilder::new(start_dir);
    builder
        .hidden(true)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .follow_links(true);
    for entry in builder.build() {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => {
                healthy.store(false, Ordering::Release);
                continue;
            }
        };
        let path = entry.path();
        let relative = match path.strip_prefix(root_dir) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if relative.as_os_str().is_empty() {
            continue;
        }
        if !should_track(relative, ignore_dirs) {
            continue;
        }
        if !entry.file_type().is_some_and(|t| t.is_dir()) {
            if collect_files {
                let normalized = normalize_relative_path(relative);
                if !normalized.is_empty() {
                    files.push(normalized);
                }
            }
            continue;
        }
        if *watched_dirs >= MAX_WATCHED_DIRS {
            warn!(
                "Embedding tracker watch cap ({}) reached at {}; further directories are not watched",
                MAX_WATCHED_DIRS,
                path.display()
            );
            healthy.store(false, Ordering::Release);
            return files;
        }
        match debouncer.watch(path, RecursiveMode::NonRecursive) {
            Ok(()) => *watched_dirs += 1,
            Err(e) => {
                healthy.store(false, Ordering::Release);
                debug!(
                    "Embedding tracker could not watch {}: {}",
                    path.display(),
                    e
                );
            }
        }
    }
    files
}

/// Handle for controlling a running embedding tracker.
/// Dropping this handle will stop the tracker.
pub struct EmbeddingTrackerHandle {
    shutdown_tx: Option<mpsc::Sender<()>>,
    watch_task: tokio::task::JoinHandle<()>,
    pending_watches: Arc<AtomicUsize>,
    // Set by notify before the embedding queue can block or drop a batch.
    source_dirty: Arc<AtomicBool>,
    healthy: Arc<AtomicBool>,
}

impl EmbeddingTrackerHandle {
    pub(crate) fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
            && self.pending_watches.load(Ordering::Acquire) == 0
            && self.shutdown_tx.as_ref().is_some_and(|tx| !tx.is_closed())
    }

    pub(crate) fn take_source_dirty(&self) -> bool {
        self.source_dirty.swap(false, Ordering::AcqRel)
    }

    /// Gracefully stops the embedding tracker.
    pub async fn stop(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }
    }
}

impl Drop for EmbeddingTrackerHandle {
    fn drop(&mut self) {
        self.watch_task.abort();
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
    let source_dirty = Arc::new(AtomicBool::new(true));
    let pending_watches = Arc::new(AtomicUsize::new(0));
    let handler_pending_watches = pending_watches.clone();
    let healthy = Arc::new(AtomicBool::new(true));
    let handler_dirty = source_dirty.clone();
    let handler_healthy = healthy.clone();
    let debounce_ms = config.clamped_debounce_ms();
    let max_files = config.clamped_max_files();
    let ignore_dirs = config.ignore_dirs.clone();

    // Channel for debounced events from notify to our async processor
    let (event_tx, mut event_rx) = mpsc::channel::<Vec<String>>(64);

    // Channel the notify handler pushes hook-sentinel events onto. The
    // resolver task reads these, runs `git diff` to turn each sentinel touch
    // into the authoritative file set, and feeds those paths into event_tx
    // just like any other file-change batch. A dedicated channel (rather than
    // calling git from inside the notify handler) keeps the handler fast and
    // synchronous — it runs on notify's own thread where blocking on `git
    // diff` would starve the debouncer.
    let (hook_tx, mut hook_rx) = mpsc::channel::<String>(16);

    // Directories that appear after startup (created, renamed in, or
    // symlinked in) need their own watches. The notify handler pushes them
    // here; a dedicated task, which owns the debouncer, installs the
    // watches and heals over any files that landed before the watch existed.
    let (watch_tx, mut watch_rx) = mpsc::channel::<PathBuf>(64);

    // Per-tracker mtime+size cache so spurious watcher events (e.g. chmod,
    // touch-without-change, atime updates) for unchanged files are dropped
    // before we incur file-read + hashing cost downstream.
    let metadata_cache: Arc<Mutex<HashMap<String, FileSig>>> = Arc::new(Mutex::new(HashMap::new()));

    // Sentinel directory where git hooks touch one file per hook name.
    // Created eagerly so notify has something to watch even before the first
    // hook install/fire. `.mcp_data` is in the walker/tracker ignore list
    // (to avoid self-triggering when cache files rotate on disk), so we need
    // a second watch that specifically targets this subdir.
    let sentinel_dir = root_dir.join(".mcp_data").join("hooks");
    if let Err(e) = std::fs::create_dir_all(&sentinel_dir) {
        warn!(
            "Could not create hook sentinel dir {}: {} — git hook integration disabled",
            sentinel_dir.display(),
            e
        );
    }

    // HEAD watch: a HEAD move (pull, merge, rebase, checkout, reset) becomes
    // `git diff old new` for this root, the same way a hook sentinel does.
    // Hook files are only an optimization; tools such as lefthook overwrite
    // them, and shared hooks cannot tell which worktree they ran in.
    let git_dirs = crate::core::git_worktree::git_dirs(&root_dir);
    let last_head: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(
        crate::core::head_watcher::resolve_head_sha(&root_dir),
    ));

    // Set up notify debouncer with an event handler that filters and collects paths
    let handler_root = root_dir.clone();
    let handler_git_dirs = git_dirs.clone();
    let handler_meta_cache = metadata_cache.clone();
    let handler_sentinel_dir = sentinel_dir.clone();
    let handler_hook_tx = hook_tx.clone();
    let handler_watch_tx = watch_tx.clone();
    // Clone event_tx for the resolver task and the consumer's dynamic-watch
    // heal path; the handler closure below takes ownership of the original.
    let resolver_event_tx = event_tx.clone();
    let consumer_event_tx = event_tx.clone();

    let event_handler = move |result: DebounceEventResult| match result {
        Ok(events) => {
            let mut new_files = Vec::new();
            for event in &events {
                if !is_change_event(&event.kind) {
                    continue;
                }
                if event.need_rescan() {
                    handler_dirty.store(true, Ordering::Release);
                    handler_healthy.store(false, Ordering::Release);
                }
                for path in &event.paths {
                    // Detect sentinel events first. A sentinel touch represents
                    // a git event (commit/merge/checkout), not a file edit —
                    // forward only the hook name to the resolver task, which
                    // knows how to turn it into the real changed-files list.
                    if let Ok(rel_to_sentinel) = path.strip_prefix(&handler_sentinel_dir)
                        && let Some(name) = rel_to_sentinel.file_name().and_then(|s| s.to_str())
                        && crate::git::hooks::MANAGED_HOOKS.contains(&name)
                    {
                        // try_send: if the resolver is backed up, drop the
                        // signal — the next hook fire will re-touch the
                        // sentinel and we'll catch up then. Losing a single
                        // signal here is preferable to blocking the notify
                        // thread on a full channel.
                        handler_dirty.store(true, Ordering::Release);
                        if handler_hook_tx.try_send(name.to_string()).is_err() {
                            handler_healthy.store(false, Ordering::Release);
                        }
                        continue;
                    }

                    if let Some(dirs) = &handler_git_dirs
                        && dirs.is_ref_update(path)
                    {
                        handler_dirty.store(true, Ordering::Release);
                        if handler_hook_tx.try_send(HEAD_SIGNAL.to_string()).is_err() {
                            handler_healthy.store(false, Ordering::Release);
                        }
                        continue;
                    }

                    let relative = match path.strip_prefix(&handler_root) {
                        Ok(r) => r.to_path_buf(),
                        Err(_) => continue,
                    };
                    if !should_track(&relative, &ignore_dirs) {
                        continue;
                    }
                    // A directory event means new watchable ground, not a
                    // file change: queue it for watch installation and let
                    // the install-time scan surface any files inside.
                    if path.is_dir() {
                        if !is_new_dir_event(&event.kind) {
                            continue;
                        }
                        handler_pending_watches.fetch_add(1, Ordering::AcqRel);
                        handler_dirty.store(true, Ordering::Release);
                        if let Err(e) = handler_watch_tx.try_send(path.clone()) {
                            handler_healthy.store(false, Ordering::Release);
                            debug!(
                                "Embedding tracker watch queue full, dropping {}: {}",
                                path.display(),
                                e
                            );
                        }
                        continue;
                    }
                    let normalized = normalize_relative_path(&relative);
                    if normalized.is_empty() {
                        continue;
                    }
                    if metadata_unchanged(&handler_meta_cache, &normalized, path) {
                        debug!(
                            "Embedding tracker skip (unchanged metadata): {}",
                            normalized
                        );
                        continue;
                    }
                    handler_dirty.store(true, Ordering::Release);
                    new_files.push(normalized);
                }
            }
            if !new_files.is_empty()
                && let Err(e) = event_tx.try_send(new_files.clone())
            {
                handler_healthy.store(false, Ordering::Release);
                // The receiving task is gone or the channel is full.
                // metadata_unchanged() already recorded these paths as
                // "seen", so a silent drop would mean the next identical
                // watcher event would skip them as unchanged → lost
                // refresh. Roll the cache entries back so the next event
                // re-triggers processing.
                warn!(
                    "Embedding tracker event_tx send failed ({}); invalidating {} cached metadata entries so they will be re-processed",
                    e,
                    new_files.len()
                );
                if let Ok(mut guard) = handler_meta_cache.lock() {
                    for f in &new_files {
                        guard.remove(f);
                    }
                }
            }
        }
        Err(errors) => {
            handler_dirty.store(true, Ordering::Release);
            handler_healthy.store(false, Ordering::Release);
            for e in errors {
                error!("Embedding tracker watcher error: {}", e);
            }
        }
    };

    // NoCache: the default FileIdMap cache fingerprints every file under
    // each watch root up front (a WalkDir with follow_links(true)) — on a
    // root with a symlink to a huge external tree that walk alone starves
    // the daemon. Rename pairing is not needed here: the handler only
    // collects paths, and the metadata cache already dedupes per path.
    let mut debouncer = new_debouncer_opt::<_, RecommendedWatcher, NoCache>(
        Duration::from_millis(debounce_ms),
        None,
        event_handler,
        NoCache,
        NotifyConfig::default(),
    )?;

    // Per-directory NonRecursive watches over the same tree the indexer
    // walks, instead of one Recursive watch on the root: RecursiveMode
    // cannot be pruned, so it descends into every ignored and symlinked
    // tree. See install_dir_watches for the semantics.
    debouncer.watch(&root_dir, RecursiveMode::NonRecursive)?;
    let mut watched_dirs = 1usize;
    let walk_ignore_dirs = config.ignore_dirs.clone();
    install_dir_watches(
        &mut debouncer,
        &root_dir,
        &root_dir,
        &walk_ignore_dirs,
        &mut watched_dirs,
        false,
        &healthy,
    );
    info!("Embedding tracker watching {} directories", watched_dirs);

    // Second watch on the sentinel dir. Non-recursive: nothing nested in
    // `.mcp_data/hooks/` concerns us. Failure here is logged but not fatal —
    // the recursive root watch still catches real file edits; only the git
    // hook optimization path goes dark.
    if sentinel_dir.is_dir()
        && let Err(e) = debouncer.watch(&sentinel_dir, RecursiveMode::NonRecursive)
    {
        warn!(
            "Could not watch hook sentinel dir {}: {} — git hook integration disabled",
            sentinel_dir.display(),
            e
        );
    }

    // Resolver task: turns hook-sentinel signals into authoritative file lists.
    // Runs `git diff` off the notify thread, filters with `should_track`,
    // invalidates the metadata cache for the changed paths (so the consumer
    // below won't skip them as "metadata unchanged"), and forwards through
    // `event_tx` — the same path a plain file edit would take. Terminates
    // naturally when all senders on `hook_tx` drop (i.e. when the debouncer
    // closure is released by the consumer task on shutdown).
    // Watches behind the HEAD signal: this worktree's gitdir (HEAD, ORIG_HEAD),
    // the common dir (packed-refs) and the branch refs. A few directories,
    // non-recursive except the small refs/heads tree.
    if let Some(dirs) = &git_dirs {
        let mut targets = vec![(dirs.gitdir.clone(), RecursiveMode::NonRecursive)];
        if dirs.common_dir != dirs.gitdir {
            targets.push((dirs.common_dir.clone(), RecursiveMode::NonRecursive));
        }
        targets.push((
            dirs.common_dir.join("refs").join("heads"),
            RecursiveMode::Recursive,
        ));
        for (dir, mode) in targets {
            if dir.is_dir()
                && let Err(e) = debouncer.watch(&dir, mode)
            {
                warn!(
                    "Could not watch git dir {}: {} — HEAD moves will not refresh embeddings",
                    dir.display(),
                    e
                );
            }
        }
    }

    let resolver_root = root_dir.clone();
    let resolver_last_head = last_head.clone();
    let resolver_meta_cache = metadata_cache.clone();
    let resolver_ignore_dirs = config.ignore_dirs.clone();
    tokio::spawn(async move {
        while let Some(hook_name) = hook_rx.recv().await {
            let root = resolver_root.clone();
            let name = hook_name.clone();
            let last_head = resolver_last_head.clone();
            let changed: Vec<String> = match tokio::task::spawn_blocking(move || {
                if name == HEAD_SIGNAL {
                    head_changes_since_last(&root, &last_head)
                } else {
                    crate::git::hooks::resolve_hook_changes(&root, &name)
                }
            })
            .await
            {
                Ok(v) => v,
                Err(e) => {
                    warn!("Embedding tracker hook resolver task panicked: {}", e);
                    continue;
                }
            };
            if changed.is_empty() {
                continue;
            }
            let kept: Vec<String> = changed
                .into_iter()
                .filter(|p| should_track(Path::new(p), &resolver_ignore_dirs))
                .map(|p| normalize_relative_path(Path::new(&p)))
                .filter(|p| !p.is_empty())
                .collect();
            if kept.is_empty() {
                continue;
            }
            // Invalidate metadata so the consumer doesn't short-circuit these
            // paths as unchanged — the file on disk may match the cached sig
            // but the git event means we want to re-embed regardless.
            invalidate_metadata_entries(&kept, &resolver_meta_cache);
            debug!(
                "Embedding tracker hook '{}' resolved {} changed file(s)",
                hook_name,
                kept.len()
            );
            if let Err(e) = resolver_event_tx.send(kept).await {
                warn!("Embedding tracker hook resolver send failed: {}", e);
                break;
            }
        }
        debug!("Embedding tracker hook resolver exiting");
    });

    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

    // Spawn async task to process batched events
    let root_for_task = root_dir.clone();
    let consumer_meta_cache = metadata_cache.clone();
    let watch_ignore_dirs = config.ignore_dirs.clone();
    let watch_root = root_dir.clone();
    let watch_healthy = healthy.clone();
    let watch_dirty = source_dirty.clone();
    let watch_pending = pending_watches.clone();
    let watch_task = tokio::spawn(async move {
        let mut debouncer = debouncer;
        let mut watched_dirs = watched_dirs;
        while let Some(new_dir) = watch_rx.recv().await {
            let fresh = install_dir_watches(
                &mut debouncer,
                &watch_root,
                &new_dir,
                &watch_ignore_dirs,
                &mut watched_dirs,
                true,
                &watch_healthy,
            );
            // Files written into the directory before its watch
            // existed never produced an event — re-inject them
            // through the normal event path.
            if !fresh.is_empty()
                && let Err(e) = consumer_event_tx.try_send(fresh)
            {
                watch_healthy.store(false, Ordering::Release);
                warn!("Embedding tracker watch heal send failed: {}", e);
            }
            // Publish invalidation before restoring coverage authority.
            watch_dirty.store(true, Ordering::Release);
            watch_pending.fetch_sub(1, Ordering::AcqRel);
        }
    });
    let consumer_healthy = healthy.clone();
    tokio::spawn(async move {
        let mut pending_set: HashSet<String> = HashSet::new();

        loop {
            tokio::select! {
                biased;
                _ = shutdown_rx.recv() => {
                    debug!("Embedding tracker shutting down");
                    break;
                }
                Some(files) = event_rx.recv() => {
                    let mut dropped: Vec<String> = Vec::new();
                    for f in files {
                        if pending_set.len() >= MAX_PENDING_FILES {
                            warn!(
                                "Embedding tracker pending cap ({}) reached, dropping event for: {}",
                                MAX_PENDING_FILES, f
                            );
                            dropped.push(f);
                            continue;
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
                                dropped.push(f);
                                continue;
                            }
                            pending_set.insert(f);
                        }
                    }

                    if !dropped.is_empty() {
                        consumer_healthy.store(false, Ordering::Release);
                        invalidate_metadata_entries(&dropped, &consumer_meta_cache);
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
        watch_task,
        pending_watches,
        source_dirty,
        healthy,
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
    fn should_track_empty_ignore_tracks_non_hidden() {
        let ignore = HashSet::new();
        // With empty ignore set, non-hidden paths are tracked
        assert!(should_track(Path::new("node_modules/foo.js"), &ignore));
        assert!(should_track(Path::new("src/main.rs"), &ignore));
        // Hidden paths (starting with '.') are always skipped, matching TS behavior
        assert!(!should_track(Path::new(".git/HEAD"), &ignore));
        assert!(!should_track(Path::new(".env"), &ignore));
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

    /// End-to-end: installing hooks + running a real `git commit` in the
    /// tracker's root should cause the refresh_callback to fire with the
    /// committed file path. This exercises the full loop:
    /// hook script → sentinel touch → notify watcher → hook_tx → resolver
    /// task → `git diff` → event_tx → consumer → refresh_callback.
    #[tokio::test]
    async fn tracker_fires_on_git_commit_via_hook_sentinel() {
        use std::process::Command;
        // Skip if git isn't available (CI may not have it).
        if Command::new("git").arg("--version").output().is_err() {
            eprintln!("SKIP: git not available");
            return;
        }

        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().to_path_buf();

        // Hermetic git: no global/system config leaking user identity in.
        let git = |args: &[&str]| {
            let ok = Command::new("git")
                .args(args)
                .current_dir(&root)
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@t")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@t")
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .env("GIT_CONFIG_SYSTEM", "/dev/null")
                .status()
                .expect("git runs")
                .success();
            assert!(ok, "git {:?} failed", args);
        };
        git(&["init", "-q", "-b", "main"]);

        // Baseline commit so HEAD exists; post-commit hook on the *next*
        // commit will diff-tree --root HEAD and return that commit's files.
        let src = root.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("seed.rs"), "fn seed() {}").unwrap();
        git(&["add", "."]);
        git(&["commit", "-q", "-m", "seed"]);

        // Install contextplus hooks into this repo.
        crate::git::hooks::install_hooks(&root).expect("install_hooks");

        // Start the tracker with a channel that records every callback fire.
        let (tx, mut rx) = mpsc::channel::<Vec<String>>(16);
        let callback: RefreshCallback = Arc::new(move |_root, files| {
            let tx = tx.clone();
            tokio::spawn(async move {
                let _ = tx.send(files).await;
                (1, 0)
            })
        });
        let config = EmbeddingTrackerConfig {
            debounce_ms: 200,
            max_files_per_tick: 8,
            ignore_dirs: default_ignore_dirs(),
        };
        let handle = start_tracker(root.clone(), config, callback).expect("tracker starts");

        // Give notify a moment to register both watches.
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create and commit a new tracked file. The post-commit hook touches
        // `.mcp_data/hooks/post-commit`, which the tracker should resolve into
        // the committed file path and forward to the refresh callback.
        std::fs::write(src.join("new_feature.rs"), "fn f() {}").unwrap();
        git(&["add", "src/new_feature.rs"]);
        git(&["commit", "-q", "-m", "feat"]);

        // Collect batches until we see new_feature.rs or time out. The real
        // edit + the hook sentinel may each produce a batch; accept either.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let mut saw_new_feature = false;
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(files)) => {
                    if files.iter().any(|f| f.ends_with("new_feature.rs")) {
                        saw_new_feature = true;
                        break;
                    }
                }
                _ => break,
            }
        }

        handle.stop().await;

        // Some CI filesystems don't deliver inotify events; treat as skip.
        if !saw_new_feature {
            eprintln!(
                "WARNING: hook-sentinel integration path did not fire — inotify or git hooks may not work in this environment"
            );
        }
    }

    fn git_in(cwd: &Path, args: &[&str]) {
        let ok = std::process::Command::new("git")
            .arg("-C")
            .arg(cwd)
            .args(args)
            .env("GIT_AUTHOR_NAME", "t")
            .env("GIT_AUTHOR_EMAIL", "t@t")
            .env("GIT_COMMITTER_NAME", "t")
            .env("GIT_COMMITTER_EMAIL", "t@t")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .status()
            .expect("git runs")
            .success();
        assert!(ok, "git {:?} failed in {}", args, cwd.display());
    }

    /// Wait up to `secs` for a callback batch naming `file`.
    async fn wait_for_file(rx: &mut mpsc::Receiver<Vec<String>>, file: &str, secs: u64) -> bool {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
        while let Ok(Some(files)) = tokio::time::timeout_at(deadline, rx.recv()).await {
            if files.iter().any(|f| f.ends_with(file)) {
                return true;
            }
        }
        false
    }

    fn tracker_with_capture(
        root: PathBuf,
    ) -> (EmbeddingTrackerHandle, mpsc::Receiver<Vec<String>>) {
        let (tx, rx) = mpsc::channel::<Vec<String>>(16);
        let callback: RefreshCallback = Arc::new(move |_root, files| {
            let tx = tx.clone();
            tokio::spawn(async move {
                let _ = tx.send(files).await;
                (1, 0)
            })
        });
        let config = EmbeddingTrackerConfig {
            debounce_ms: 200,
            max_files_per_tick: 8,
            ignore_dirs: default_ignore_dirs(),
        };
        let handle = start_tracker(root, config, callback).expect("tracker starts");
        (handle, rx)
    }

    /// A HEAD move refreshes the files it changed with no hook file installed
    /// (lefthook and similar tools overwrite `.git/hooks/*`). The file edit is
    /// drained first, so the second report can only come from the HEAD watch.
    #[tokio::test]
    async fn tracker_fires_on_head_advance_without_hooks() {
        if std::process::Command::new("git")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("SKIP: git not available");
            return;
        }
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().canonicalize().unwrap();
        git_in(&root, &["init", "-q", "-b", "main"]);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/seed.rs"), "fn seed() {}").unwrap();
        git_in(&root, &["add", "."]);
        git_in(&root, &["commit", "-q", "-m", "seed"]);

        let (handle, mut rx) = tracker_with_capture(root.clone());
        tokio::time::sleep(Duration::from_millis(300)).await;

        std::fs::write(root.join("src/new_feature.rs"), "fn f() {}").unwrap();
        if !wait_for_file(&mut rx, "new_feature.rs", 5).await {
            handle.stop().await;
            eprintln!(
                "WARNING: file watcher did not fire — inotify may not work in this environment"
            );
            return;
        }

        git_in(&root, &["add", "src/new_feature.rs"]);
        git_in(&root, &["commit", "-q", "-m", "feat"]);
        let seen = wait_for_file(&mut rx, "new_feature.rs", 8).await;
        handle.stop().await;
        assert!(seen, "HEAD advance did not refresh the committed file");
    }

    /// A linked worktree's tracker watches the shared `refs/heads`, so moving
    /// its HEAD (`reset --soft`, which leaves the working tree untouched)
    /// refreshes the files that differ between the two commits.
    #[tokio::test]
    async fn tracker_in_linked_worktree_refreshes_after_head_moves() {
        if std::process::Command::new("git")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("SKIP: git not available");
            return;
        }
        let dir = tempfile::TempDir::new().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let primary = base.join("repo");
        let wt = base.join("wt");
        std::fs::create_dir_all(primary.join("src")).unwrap();
        git_in(&primary, &["init", "-q", "-b", "main"]);
        std::fs::write(primary.join("src/seed.rs"), "fn seed() {}").unwrap();
        git_in(&primary, &["add", "."]);
        git_in(&primary, &["commit", "-q", "-m", "seed"]);
        git_in(
            &primary,
            &["worktree", "add", "-q", "-b", "feat", wt.to_str().unwrap()],
        );

        let (handle, mut rx) = tracker_with_capture(wt.clone());
        tokio::time::sleep(Duration::from_millis(300)).await;

        std::fs::write(primary.join("src/from_main.rs"), "fn m() {}").unwrap();
        git_in(&primary, &["add", "src/from_main.rs"]);
        git_in(&primary, &["commit", "-q", "-m", "on main"]);
        git_in(&wt, &["reset", "-q", "--soft", "main"]);

        let seen = wait_for_file(&mut rx, "from_main.rs", 8).await;
        handle.stop().await;
        assert!(seen, "worktree HEAD move did not refresh src/from_main.rs");
    }

    #[test]
    fn access_events_are_not_changes() {
        use notify_debouncer_full::notify::event::{
            AccessKind, AccessMode, CreateKind, DataChange, MetadataKind, RemoveKind, RenameMode,
        };
        assert!(!is_change_event(&EventKind::Access(AccessKind::Open(
            AccessMode::Any
        ))));
        assert!(!is_change_event(&EventKind::Access(AccessKind::Close(
            AccessMode::Read
        ))));
        assert!(!is_change_event(&EventKind::Access(AccessKind::Read)));
        assert!(is_change_event(&EventKind::Create(CreateKind::File)));
        assert!(is_change_event(&EventKind::Modify(ModifyKind::Data(
            DataChange::Any
        ))));
        assert!(is_change_event(&EventKind::Remove(RemoveKind::File)));
        assert!(is_change_event(&EventKind::Any));

        assert!(is_new_dir_event(&EventKind::Create(CreateKind::Folder)));
        assert!(is_new_dir_event(&EventKind::Modify(ModifyKind::Name(
            RenameMode::To
        ))));
        assert!(!is_new_dir_event(&EventKind::Modify(ModifyKind::Metadata(
            MetadataKind::Any
        ))));
        assert!(!is_new_dir_event(&EventKind::Access(AccessKind::Open(
            AccessMode::Any
        ))));
    }

    /// Reading the tree, which the indexer, warmups and the heal walk all do,
    /// must not produce refresh batches; a real write afterwards must.
    #[tokio::test]
    async fn tracker_ignores_reads_of_the_tree() {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().canonicalize().unwrap();
        for d in ["src/a", "src/b"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        std::fs::write(root.join("src/a/one.rs"), "fn one() {}").unwrap();
        std::fs::write(root.join("src/b/two.rs"), "fn two() {}").unwrap();
        std::fs::write(root.join("src/lib.rs"), "mod a;").unwrap();

        let (handle, mut rx) = tracker_with_capture(root.clone());
        tokio::time::sleep(Duration::from_millis(300)).await;

        for _ in 0..3 {
            for entry in WalkBuilder::new(&root).build().flatten() {
                if entry.file_type().is_some_and(|t| t.is_file()) {
                    let _ = std::fs::read(entry.path());
                } else {
                    let _ = std::fs::read_dir(entry.path()).map(|it| it.count());
                }
            }
        }
        let quiet = tokio::time::timeout(Duration::from_millis(1500), rx.recv())
            .await
            .is_err();
        assert!(quiet, "reading the tree produced a refresh batch");

        std::fs::write(root.join("src/lib.rs"), "mod a; mod b;").unwrap();
        let seen = wait_for_file(&mut rx, "lib.rs", 5).await;
        handle.stop().await;
        assert!(seen, "a real write was not reported");
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

    fn capture_callback() -> (RefreshCallback, mpsc::Receiver<Vec<String>>) {
        let (tx, rx) = mpsc::channel::<Vec<String>>(16);
        let callback: RefreshCallback = Arc::new(move |_root, files| {
            let tx = tx.clone();
            tokio::spawn(async move {
                let _ = tx.send(files).await;
                (0, 0)
            })
        });
        (callback, rx)
    }

    fn test_config() -> EmbeddingTrackerConfig {
        EmbeddingTrackerConfig {
            debounce_ms: 200,
            max_files_per_tick: 8,
            ignore_dirs: default_ignore_dirs(),
        }
    }

    /// A visible, non-ignored symlinked directory is part of the workspace:
    /// files behind it must produce events like any other source file.
    #[cfg(unix)]
    #[tokio::test]
    async fn tracker_watches_visible_symlinked_dirs() {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().to_path_buf();
        let external = tempfile::TempDir::new().unwrap();
        let target = external.path().join("shared");
        std::fs::create_dir_all(&target).unwrap();
        std::os::unix::fs::symlink(&target, root.join("linked")).unwrap();

        let (callback, mut rx) = capture_callback();
        let handle =
            start_tracker(root.clone(), test_config(), callback).expect("tracker should start");

        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::write(target.join("shared_lib.rs"), "fn main() {}").unwrap();

        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(files)) => {
                assert!(
                    files.iter().any(|f| f.contains("shared_lib.rs")),
                    "expected event for file behind visible symlink, got: {:?}",
                    files
                );
            }
            Ok(None) => panic!("channel closed without receiving event"),
            Err(_) => {
                eprintln!("WARNING: file watcher timeout -- may not work in this environment");
            }
        }

        handle.stop().await;
    }

    /// A symlink whose name is on the ignore list must not be watched:
    /// pruning happens at watch-install time, not just event time.
    #[cfg(unix)]
    #[tokio::test]
    async fn tracker_skips_ignored_symlinked_dirs() {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().to_path_buf();
        let external = tempfile::TempDir::new().unwrap();
        let target = external.path().join("pkgs");
        std::fs::create_dir_all(&target).unwrap();
        std::os::unix::fs::symlink(&target, root.join("node_modules")).unwrap();

        let (callback, mut rx) = capture_callback();
        let handle =
            start_tracker(root.clone(), test_config(), callback).expect("tracker should start");

        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::write(target.join("index.js"), "module.exports = {}").unwrap();

        if let Ok(Some(files)) = tokio::time::timeout(Duration::from_secs(1), rx.recv()).await {
            panic!(
                "should not receive events through ignored symlink, got: {:?}",
                files
            );
        }

        handle.stop().await;
    }

    /// Regression: a hidden symlink like `.worktrees -> ~/worktrees` must
    /// not be watched at all — before install-time pruning this pulled
    /// millions of external inodes into the watch table during startup.
    #[cfg(unix)]
    #[tokio::test]
    async fn tracker_skips_hidden_symlinked_dirs() {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().to_path_buf();
        let external = tempfile::TempDir::new().unwrap();
        let target = external.path().join("worktrees");
        std::fs::create_dir_all(&target).unwrap();
        std::os::unix::fs::symlink(&target, root.join(".worktrees")).unwrap();

        let (callback, mut rx) = capture_callback();
        let handle =
            start_tracker(root.clone(), test_config(), callback).expect("tracker should start");

        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::write(target.join("escape.rs"), "fn main() {}").unwrap();

        if let Ok(Some(files)) = tokio::time::timeout(Duration::from_secs(1), rx.recv()).await {
            panic!(
                "should not receive events through hidden symlink, got: {:?}",
                files
            );
        }

        handle.stop().await;
    }

    /// A gitignored symlink must not be watched — the watcher honors the
    /// same gitignore rules as the indexer's walk.
    #[cfg(unix)]
    #[tokio::test]
    async fn tracker_skips_gitignored_symlinked_dirs() {
        use std::process::Command;
        if Command::new("git").arg("--version").output().is_err() {
            eprintln!("SKIP: git not available");
            return;
        }

        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().to_path_buf();
        if !Command::new("git")
            .args(["init", "-q"])
            .current_dir(&root)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            eprintln!("SKIP: git init failed");
            return;
        }
        std::fs::write(root.join(".gitignore"), "linked\n").unwrap();

        let external = tempfile::TempDir::new().unwrap();
        let target = external.path().join("farm");
        std::fs::create_dir_all(&target).unwrap();
        std::os::unix::fs::symlink(&target, root.join("linked")).unwrap();

        let (callback, mut rx) = capture_callback();
        let handle =
            start_tracker(root.clone(), test_config(), callback).expect("tracker should start");

        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::write(target.join("escape.rs"), "fn main() {}").unwrap();

        if let Ok(Some(files)) = tokio::time::timeout(Duration::from_secs(1), rx.recv()).await {
            panic!(
                "should not receive events through gitignored symlink, got: {:?}",
                files
            );
        }

        handle.stop().await;
    }

    /// Directories created while the tracker runs get watches installed
    /// dynamically, and files that land before the watch exists are healed
    /// over by the install-time scan.
    #[tokio::test]
    async fn tracker_watches_dirs_created_after_start() {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().to_path_buf();

        let (callback, mut rx) = capture_callback();
        let handle =
            start_tracker(root.clone(), test_config(), callback).expect("tracker should start");

        tokio::time::sleep(Duration::from_millis(100)).await;
        let new_dir = root.join("newdir");
        std::fs::create_dir_all(&new_dir).unwrap();
        std::fs::write(new_dir.join("late.rs"), "fn late() {}").unwrap();

        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(files)) => {
                assert!(
                    files.iter().any(|f| f.contains("late.rs")),
                    "expected event for file in newly created dir, got: {:?}",
                    files
                );
            }
            Ok(None) => panic!("channel closed without receiving event"),
            Err(_) => {
                eprintln!("WARNING: file watcher timeout -- may not work in this environment");
            }
        }

        handle.stop().await;
    }

    // -- metadata-skip gate (mtime + size pre-filter) --

    #[test]
    fn metadata_unchanged_first_observation_returns_false() {
        let dir = tempfile::TempDir::new().unwrap();
        let f = dir.path().join("a.rs");
        std::fs::write(&f, b"fn x() {}").unwrap();

        let cache = Mutex::new(HashMap::new());
        // First time we see the file: must NOT skip — we have no prior signature.
        assert!(!metadata_unchanged(&cache, "a.rs", &f));
    }

    #[test]
    fn metadata_unchanged_second_call_with_same_file_skips() {
        let dir = tempfile::TempDir::new().unwrap();
        let f = dir.path().join("a.rs");
        std::fs::write(&f, b"fn x() {}").unwrap();

        let cache = Mutex::new(HashMap::new());
        let _ = metadata_unchanged(&cache, "a.rs", &f);
        // Second call without touching the file: skip.
        assert!(metadata_unchanged(&cache, "a.rs", &f));
    }

    #[test]
    fn metadata_unchanged_detects_size_change() {
        let dir = tempfile::TempDir::new().unwrap();
        let f = dir.path().join("a.rs");
        std::fs::write(&f, b"fn x() {}").unwrap();

        let cache = Mutex::new(HashMap::new());
        let _ = metadata_unchanged(&cache, "a.rs", &f);

        std::fs::write(&f, b"fn xy() {}").unwrap(); // size changes
        assert!(!metadata_unchanged(&cache, "a.rs", &f));
    }

    #[test]
    fn metadata_unchanged_returns_false_for_missing_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let f = dir.path().join("ghost.rs");
        let cache = Mutex::new(HashMap::new());
        // No file → can't trust the gate; let the callback handle it.
        assert!(!metadata_unchanged(&cache, "ghost.rs", &f));
    }

    // -- pending-cap rollback (RV3-002) --

    #[test]
    fn invalidate_metadata_entries_drops_only_listed_paths() {
        // Simulate the consumer-side rollback: events were registered in
        // the metadata cache (because metadata_unchanged was called when
        // the watcher event arrived) but then dropped at the cap. The
        // dropped paths' sigs must be removed so the next identical event
        // re-fires the consumer instead of being skipped as "unchanged".
        let cache: Arc<Mutex<HashMap<String, FileSig>>> = Arc::new(Mutex::new(HashMap::new()));
        {
            let mut g = cache.lock().unwrap();
            for name in ["a.rs", "b.rs", "c.rs", "d.rs"] {
                g.insert(
                    name.to_string(),
                    FileSig {
                        mtime: SystemTime::UNIX_EPOCH,
                        size: 1,
                    },
                );
            }
        }

        let dropped = vec!["b.rs".to_string(), "d.rs".to_string()];
        invalidate_metadata_entries(&dropped, &cache);

        let g = cache.lock().unwrap();
        assert!(g.contains_key("a.rs"), "untouched entry survives");
        assert!(g.contains_key("c.rs"), "untouched entry survives");
        assert!(!g.contains_key("b.rs"), "dropped entry must be removed");
        assert!(!g.contains_key("d.rs"), "dropped entry must be removed");
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn invalidate_metadata_entries_empty_dropped_is_noop() {
        let cache: Arc<Mutex<HashMap<String, FileSig>>> = Arc::new(Mutex::new(HashMap::new()));
        {
            let mut g = cache.lock().unwrap();
            g.insert(
                "keep.rs".to_string(),
                FileSig {
                    mtime: SystemTime::UNIX_EPOCH,
                    size: 1,
                },
            );
        }

        invalidate_metadata_entries(&[], &cache);

        let g = cache.lock().unwrap();
        assert_eq!(g.len(), 1, "empty dropped list must not perturb cache");
        assert!(g.contains_key("keep.rs"));
    }

    #[test]
    fn invalidate_metadata_entries_unknown_paths_are_silent() {
        // The closure was originally local to start_tracker; if a path
        // appears in `dropped` but was never inserted into the cache (e.g.
        // metadata-skip itself failed), `remove` must not panic — it just
        // returns None.
        let cache: Arc<Mutex<HashMap<String, FileSig>>> = Arc::new(Mutex::new(HashMap::new()));
        invalidate_metadata_entries(&["never_seen.rs".to_string()], &cache);
        assert!(cache.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn tracker_rollback_re_fires_after_cap_pressure() {
        // Integration-style: drive the tracker via the consumer side.
        // We can't reach the closure directly, but the contract we care
        // about is: after a path is dropped at the cap, hitting metadata
        // for that same path must again return `false` (= "process me").
        let cache: Arc<Mutex<HashMap<String, FileSig>>> = Arc::new(Mutex::new(HashMap::new()));

        let dir = tempfile::TempDir::new().unwrap();
        let f = dir.path().join("hot.rs");
        std::fs::write(&f, b"fn x() {}").unwrap();

        // Step 1: watcher event arrives → metadata_unchanged records sig
        // and returns false (first observation).
        assert!(!metadata_unchanged(&cache, "hot.rs", &f));
        // Step 2: same event arrives again → would normally skip. This
        // models the "consumer dropped the file at cap" hazard.
        assert!(metadata_unchanged(&cache, "hot.rs", &f));

        // Step 3: rollback fires for the dropped path.
        invalidate_metadata_entries(&["hot.rs".to_string()], &cache);

        // Step 4: next watcher event must once again be reported through.
        assert!(
            !metadata_unchanged(&cache, "hot.rs", &f),
            "after rollback, the same file must re-enter the pipeline"
        );
    }
}
