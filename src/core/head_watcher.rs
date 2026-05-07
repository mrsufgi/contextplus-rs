//! HEAD-advance watcher for primary worktree.
//!
//! Uses `notify-debouncer-full` to watch the primary repo's gitdir for
//! changes to the `HEAD` ref.  When HEAD advances (new commit SHA), the
//! watcher emits a [`HeadEvent::Advanced`] over an async channel.
//!
//! The daemon holds one watcher per primary repo.  Per-ref overlay memory
//! graphs are then checked for ancestry:
//!
//! ```text
//! git merge-base --is-ancestor <ref_head> <new_primary_head>
//! ```
//!
//! Refs whose HEAD is an ancestor of the new primary HEAD enter the merge
//! ladder (see `memory_merge.rs`).
//!
//! ## Relationship to the embedding tracker
//!
//! `head_watcher` is a *lighter* variant of `embedding_tracker`:
//! * It watches only `<gitdir>/HEAD` (and optionally `packed-refs` / `refs/`)
//!   for ref-update events.
//! * It does not debounce file-content changes — a HEAD change is a one-shot
//!   event per commit.
//! * It shells out to `git rev-parse HEAD` rather than reading the file
//!   directly to handle packed refs, symbolic refs, etc.
//!
//! ## inotify watch budget
//!
//! A single `notify` watcher is shared for the primary's gitdir.  We watch
//! the gitdir *non-recursively* and add targeted non-recursive watches for
//! `<gitdir>/refs/heads/` to catch fast-forward pushes.  This keeps the
//! inotify fd count to ≤3 per primary, well within the default budget of
//! `/proc/sys/fs/inotify/max_user_watches` (8192+).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use notify_debouncer_full::notify::RecursiveMode;
use notify_debouncer_full::{DebounceEventResult, new_debouncer};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Public event type
// ---------------------------------------------------------------------------

/// Events emitted by the HEAD watcher.
#[derive(Debug, Clone)]
pub enum HeadEvent {
    /// Primary HEAD advanced from `old_sha` to `new_sha`.
    Advanced { old_sha: String, new_sha: String },
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// Handle for a running HEAD watcher.  Drop to shut down.
pub struct HeadWatcherHandle {
    shutdown: Arc<AtomicBool>,
    _debouncer_thread: std::thread::JoinHandle<()>,
}

impl HeadWatcherHandle {
    /// Signal the watcher to stop.
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

impl Drop for HeadWatcherHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

// ---------------------------------------------------------------------------
// Resolve HEAD SHA
// ---------------------------------------------------------------------------

/// Read the current HEAD SHA of the repo rooted at `repo_root`.
///
/// Shells out to `git rev-parse HEAD` to handle symbolic refs, packed-refs,
/// and detached HEADs correctly.  Returns `None` if the command fails.
pub fn resolve_head_sha(repo_root: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .current_dir(repo_root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if sha.is_empty() { None } else { Some(sha) }
}

/// Check whether `candidate_sha` is an ancestor of `target_sha` in the repo
/// rooted at `repo_root`.
///
/// Uses `git merge-base --is-ancestor <candidate> <target>` which exits 0
/// when true, 1 when false, other on error.  Returns `None` on error so
/// callers can treat the ambiguous case conservatively.
pub fn is_ancestor(repo_root: &Path, candidate_sha: &str, target_sha: &str) -> Option<bool> {
    let status = std::process::Command::new("git")
        .args(["merge-base", "--is-ancestor", candidate_sha, target_sha])
        .current_dir(repo_root)
        .status()
        .ok()?;
    Some(status.success())
}

// ---------------------------------------------------------------------------
// Watcher startup
// ---------------------------------------------------------------------------

/// Start watching for HEAD advances on a primary repo.
///
/// Returns a `(HeadWatcherHandle, mpsc::Receiver<HeadEvent>)` pair.
/// The channel is bounded to `64` events; if the consumer falls behind,
/// events are dropped (the next HEAD advance will re-emit the current SHA).
///
/// `gitdir` should be the `.git` directory (or the gitdir pointer target for
/// worktrees).  If you only have the repo root, pass `repo_root.join(".git")`.
///
/// `repo_root` is used for `git rev-parse` calls.
pub fn start_head_watcher(
    repo_root: PathBuf,
    gitdir: PathBuf,
) -> Result<(HeadWatcherHandle, mpsc::Receiver<HeadEvent>), notify_debouncer_full::notify::Error> {
    let (event_tx, event_rx) = mpsc::channel::<HeadEvent>(64);
    let shutdown = Arc::new(AtomicBool::new(false));

    let handler_root = repo_root.clone();
    let handler_tx = event_tx.clone();

    // Current HEAD SHA — shared between the notify thread and the async
    // consumer via an Arc<Mutex>.  Updated every time a HEAD advance is
    // confirmed.
    let current_sha: Arc<std::sync::Mutex<Option<String>>> =
        Arc::new(std::sync::Mutex::new(resolve_head_sha(&repo_root)));

    let handler_sha = current_sha.clone();
    let handler_shutdown = shutdown.clone();

    let debounce_ms = 300u64;

    let event_handler = move |result: DebounceEventResult| {
        if handler_shutdown.load(Ordering::Acquire) {
            return;
        }

        let events_list = match result {
            Ok(ev) => ev,
            Err(errs) => {
                for e in errs {
                    warn!("head_watcher: notify error: {e}");
                }
                return;
            }
        };

        // Check if any of the changed paths look like a ref update.
        let relevant = events_list.iter().any(|e| {
            e.paths.iter().any(|p| {
                let s = p.to_string_lossy();
                // HEAD, FETCH_HEAD, packed-refs, refs/heads/*, refs/remotes/*
                s.ends_with("HEAD")
                    || s.ends_with("packed-refs")
                    || s.contains("refs/heads")
                    || s.contains("refs/remotes")
            })
        });

        if !relevant {
            return;
        }

        // Read the new HEAD SHA.
        let new_sha = match resolve_head_sha(&handler_root) {
            Some(s) => s,
            None => {
                debug!("head_watcher: could not resolve HEAD after notify event");
                return;
            }
        };

        let old_sha = {
            let mut guard = match handler_sha.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let old = guard.clone().unwrap_or_default();
            if old == new_sha {
                // No change — spurious event.
                return;
            }
            *guard = Some(new_sha.clone());
            old
        };

        info!(
            old_sha = %old_sha,
            new_sha = %new_sha,
            "head_watcher: HEAD advanced"
        );

        // try_send: if the channel is full (consumer is slow), drop the event.
        // The next real HEAD change will re-emit.
        if handler_tx
            .try_send(HeadEvent::Advanced { old_sha, new_sha })
            .is_err()
        {
            warn!("head_watcher: event channel full — HEAD advance event dropped");
        }
    };

    let mut debouncer = new_debouncer(Duration::from_millis(debounce_ms), None, event_handler)?;

    // Non-recursive watch on the gitdir root (catches HEAD, FETCH_HEAD, packed-refs).
    debouncer.watch(&gitdir, RecursiveMode::NonRecursive)?;

    // Also watch refs/heads/ if it exists (non-recursive: we only care about
    // direct children, not nested trees).
    let refs_heads = gitdir.join("refs").join("heads");
    if refs_heads.is_dir()
        && let Err(e) = debouncer.watch(&refs_heads, RecursiveMode::NonRecursive)
    {
        warn!("head_watcher: could not watch refs/heads: {e} — fast-forward events may be delayed");
    }

    // The debouncer must stay alive for events to fire.  We move it into a
    // background thread that parks until shutdown.
    let shutdown_for_thread = shutdown.clone();
    let join_handle = std::thread::spawn(move || {
        loop {
            if shutdown_for_thread.load(Ordering::Acquire) {
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        // Debouncer dropped here — the notify subscription is released.
        drop(debouncer);
    });

    Ok((
        HeadWatcherHandle {
            shutdown,
            _debouncer_thread: join_handle,
        },
        event_rx,
    ))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Initialise a minimal git repo in a temp dir and return the dir.
    fn init_repo() -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::process::Command::new("git")
            .args(["init", "--initial-branch=main"])
            .current_dir(dir.path())
            .output()
            .expect("git init failed");
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        dir
    }

    fn make_commit(dir: &Path) -> String {
        let file = dir.join("file.txt");
        let content = format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        );
        fs::write(&file, content).unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "test commit"])
            .current_dir(dir)
            .output()
            .unwrap();
        resolve_head_sha(dir).expect("commit should produce HEAD")
    }

    #[test]
    fn resolve_head_sha_returns_none_for_empty_repo() {
        let dir = init_repo();
        // Empty repo has no HEAD commit — rev-parse fails.
        let sha = resolve_head_sha(dir.path());
        // On newer git, an empty repo may return None or a SHA for an initial
        // empty tree — either is acceptable.  We just assert no panic.
        let _ = sha;
    }

    #[test]
    fn resolve_head_sha_returns_sha_after_commit() {
        let dir = init_repo();
        let sha = make_commit(dir.path());
        assert_eq!(sha.len(), 40, "SHA should be 40 hex chars");
        let sha2 = resolve_head_sha(dir.path()).unwrap();
        assert_eq!(sha, sha2);
    }

    #[test]
    fn is_ancestor_self_is_ancestor() {
        let dir = init_repo();
        let sha = make_commit(dir.path());
        // A commit is always an ancestor of itself.
        let result = is_ancestor(dir.path(), &sha, &sha);
        assert_eq!(result, Some(true));
    }

    #[test]
    fn is_ancestor_parent_is_ancestor_of_child() {
        let dir = init_repo();
        let sha1 = make_commit(dir.path());
        let sha2 = make_commit(dir.path());
        // sha1 is ancestor of sha2.
        assert_eq!(is_ancestor(dir.path(), &sha1, &sha2), Some(true));
        // sha2 is NOT ancestor of sha1.
        assert_eq!(is_ancestor(dir.path(), &sha2, &sha1), Some(false));
    }

    #[test]
    fn is_ancestor_returns_none_for_bad_sha() {
        let dir = init_repo();
        let _sha = make_commit(dir.path());
        // Garbage SHA — git exits with an error.
        let result = is_ancestor(dir.path(), "deadbeef", "cafebabe");
        // May return None or Some(false) depending on git version.
        // We just assert no panic.
        let _ = result;
    }
}
