//! Multi-ref index scaffolding.
//!
//! A "ref" is the daemon's unit of isolation: typically one per active git
//! worktree of the primary repo. Today (U3) the daemon serves a single
//! default ref whose state lives directly on [`crate::server::SharedState`].
//! U4 will introduce real per-ref dispatch via `register_session`; U6 + U7
//! will migrate the per-ref heavy state (chunk index, memory graph) into
//! [`RefIndex`] proper.
//!
//! This module's job is to reserve the API surface (`RefId`, `RefIndex`,
//! `default_ref_id`, accessor methods) so future units can grow the
//! multi-ref behavior without churning every tool handler's import path.
//!
//! ## Key types
//!
//! * [`RefId`] — content-addressable identity for a ref. Derived from a
//!   canonical worktree path so the same physical worktree always hashes
//!   to the same id, regardless of which session initiated registration.
//! * [`RefIndex`] — per-ref identity metadata (root, parent, head) **plus
//!   per-ref heavy state** (embedding cache, identifier index, search index
//!   cache, embedding tracker, project cache).  Heavy state was migrated into
//!   `RefIndex` in U10; `SharedState` now holds `Arc` clones of the default
//!   ref's fields for backward compatibility during the U11 migration period.
//! * [`RefRegistry`] — the daemon's `HashMap<RefId, Arc<RefIndex>>`,
//!   wrapped in a `RwLock` so attach/detach can land without coordination
//!   with read-only tool dispatches.
//!
//! ## U6 additions
//!
//! `RefId` now has a stable hex representation (`to_hex`) used as the on-disk
//! directory name under `.mcp_data/refs/<hex>/`. The CAS store lookup path
//! calls [`RefIndex::fork_from`] to initialize the parent pointer on disk when
//! a new worktree ref is first attached.
//!
//! ## U10 additions
//!
//! `RefIndex` now carries per-ref caches:
//! - `embedding_cache` — content-addressed embedding vectors for this ref's files.
//! - `identifier_index` — parsed symbol embedding index for semantic identifier search.
//! - `search_index_cache` — HNSW search index cache with generation tracking.
//! - `tracker_handle` — per-ref notify-rs embedding tracker.
//! - `cache_generation` — monotonic counter bumped by the tracker on file events.
//! - `project_cache` — walked file entries and raw content for this ref.
//! - `memory_overlay` — CoW MemoryGraph overlay seam (was already present as a stub).
//!
//! For the **default ref**, these fields are initialized from disk (embedding
//! cache) or empty (everything else) at daemon start, matching the prior
//! `SharedState`-singleton behaviour.  For **non-default (worktree) refs**,
//! all fields start empty and populate on first use.
//!
//! **Seam for U4**: today `fork_from` is called explicitly by test code and by
//! `SharedState::new` for the single default ref. U4's `attach_ref` will call
//! it after resolving the fork base from `merge-base`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize};

use tokio::sync::RwLock;

use crate::core::embedding_tracker::EmbeddingTrackerHandle;
use crate::core::embeddings::CacheEntry;
use crate::core::memory_graph::MemoryGraph;
use crate::server::{IdentifierIndex, ProjectCache};
use crate::tools::semantic_search::CachedSearchIndex;

/// Stable identifier for a ref (worktree + HEAD).
///
/// Derived from the canonical path of the worktree root. Identical canonical
/// paths produce identical [`RefId`]s — which is the property that lets two
/// bridges connecting to the same worktree converge on one shared
/// [`RefIndex`] without explicit coordination.
///
/// **U6**: uses BLAKE3 of the canonical path bytes for cross-restart stability.
/// The first 8 bytes of the BLAKE3 hash are stored as a `u64` for in-process
/// routing; the full 32-byte hash is used as the on-disk directory name
/// (returned by [`RefId::to_hex`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RefId(pub u64);

impl RefId {
    /// Compute the [`RefId`] for a canonical worktree path.
    ///
    /// Caller is responsible for passing a canonicalized path — non-canonical
    /// inputs will produce non-canonical ids that may not match across calls.
    /// Use [`crate::core::git_worktree::resolve_primary_worktree`] +
    /// `canonicalize()` before constructing.
    ///
    /// **U6**: switched from `DefaultHasher` (process-local SipHash) to BLAKE3
    /// so the routing key is stable across daemon restarts and can serve as the
    /// on-disk directory name under `.mcp_data/refs/`.
    pub fn for_canonical_path(path: &Path) -> Self {
        let hash = blake3::hash(path.as_os_str().as_encoded_bytes());
        let bytes = hash.as_bytes();
        let id = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        RefId(id)
    }

    /// Return a 16-character lowercase hex string suitable for use as an
    /// on-disk directory name under `.mcp_data/refs/`.
    ///
    /// Using the full `u64` as hex (8 bytes = 16 hex chars) is sufficient for
    /// routing uniqueness in practice; collisions across simultaneously active
    /// refs would require `2^32` worktrees.
    pub fn to_hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

/// Per-ref identity metadata held by the daemon's registry.
///
/// U4 adds a session `refcount` (number of active bridges bound to this ref)
/// and a `head_sha` populated at registration.
///
/// **U6 additions**:
/// - `cas_ref_id_hex`: the stable on-disk ref-id hex (from BLAKE3 of canonical
///   path) used as the directory name under `.mcp_data/refs/`.
/// - `fork_from`: initializes the parent pointer on disk and is the seam that
///   U4's `attach_ref` will call after resolving the merge-base.
///
/// **U10 additions** — per-ref caches (migrated from `SharedState`):
/// - `embedding_cache` — content-addressed embedding vectors for this ref.
/// - `identifier_index` — parsed symbol index for semantic identifier search.
/// - `search_index_cache` — HNSW search index with generation tracking.
/// - `tracker_handle` — per-ref notify-rs embedding tracker (one per worktree).
/// - `cache_generation` — bumped on each tracker file-change event.
/// - `project_cache` — walked file entries and raw content for this ref.
/// - `memory_overlay` — CoW MemoryGraph overlay (populated at attach time).
///
/// Wrap in `Arc` for cheap cloning across handler dispatches.
pub struct RefIndex {
    /// Original `--root-dir` argument supplied by the bridge.
    pub root_dir: PathBuf,
    /// Canonicalized version (symlinks resolved). Used as the routing key.
    pub canonical_root: PathBuf,
    /// `RefId` of the parent ref this one CoW-forks from. `None` means
    /// "this is the primary ref" (no fork base).
    pub parent_ref_id: Option<RefId>,
    /// HEAD SHA at the time of registration. Used by U7's auto-merge.
    /// `None` until U4's `register_session` populates it.
    pub head_sha: Option<String>,
    /// Stable 16-character hex key for this ref's on-disk CAS directory.
    /// Derived from `BLAKE3(canonical_root)[:8]`. Stable across daemon
    /// restarts as long as the canonical path is the same.
    pub cas_ref_id_hex: String,
    /// Number of sessions currently attached to this ref. Incremented by
    /// `SharedState::attach_ref`, decremented by `detach_ref`. When it
    /// reaches zero the ref enters the TTL eviction queue.
    pub session_count: Arc<AtomicUsize>,

    // ── U10: per-ref caches ─────────────────────────────────────────────────
    /// Cached embedding vectors keyed by relative file path.
    /// `Arc`-wrapped so `SharedState` can hold a clone that shares the same
    /// lock as the default ref (backward-compat shim for U11 migration).
    pub embedding_cache: Arc<RwLock<HashMap<String, CacheEntry>>>,

    /// Cached identifier index (parsed symbols + embedding vectors).
    /// `Arc`-wrapped for the same backward-compat reason as `embedding_cache`.
    pub identifier_index: Arc<RwLock<Option<Arc<IdentifierIndex>>>>,

    /// Cached HNSW search index with generation counter.
    /// Already `Arc<RwLock<…>>` so background rebuild tasks can hold a clone.
    pub search_index_cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>>,

    /// Monotonic counter incremented by the embedding tracker on each file-change
    /// event batch. `semantic_code_search` compares this against
    /// `CachedSearchIndex::generation` to decide whether to re-walk.
    pub cache_generation: Arc<AtomicU64>,

    /// Per-ref embedding tracker handle (notify-rs watcher on `canonical_root`).
    /// `Arc<Mutex<…>>` so `SharedState` can hold a clone for backward compat.
    ///
    /// # Lifecycle
    /// Started lazily by `ensure_tracker_started`. Stopped on `Drop` of the
    /// inner `EmbeddingTrackerHandle` (notify-rs drops the watcher thread).
    /// When a ref is TTL-evicted from the registry the `Arc<RefIndex>` is
    /// dropped, decrementing the tracker handle's Arc; when the last clone is
    /// dropped the tracker stops automatically.
    ///
    /// TODO(U11): wire per-ref `ensure_tracker_started` so non-default refs
    /// start their own trackers on `canonical_root`, not the primary root.
    pub tracker_handle: Arc<std::sync::Mutex<Option<EmbeddingTrackerHandle>>>,

    /// Cached project state (walked file entries + raw content).
    /// `Arc`-wrapped for the backward-compat shim.
    pub project_cache: Arc<RwLock<Option<Arc<ProjectCache>>>>,

    /// Per-ref CoW memory-graph overlay.  `None` for the primary ref.
    /// Populated by `daemon::serve_connection` (U10) for non-default refs.
    ///
    /// The merge ladder in `daemon::spawn_head_watcher_task` reads this field;
    /// if it is `None` the ref is skipped with a debug log.
    pub memory_overlay: Option<Arc<RwLock<MemoryGraph>>>,
}

impl RefIndex {
    /// Build a new `RefIndex` with **empty** per-ref caches.
    ///
    /// Used for non-default (worktree) refs attached at runtime — they start
    /// with clean in-memory state that populates on first use.  The default
    /// ref is created via [`RefIndex::with_preloaded_cache`] so it can carry
    /// the disk-loaded embedding cache from the legacy path.
    pub fn new(root_dir: PathBuf, canonical_root: PathBuf, parent_ref_id: Option<RefId>) -> Self {
        let id = RefId::for_canonical_path(&canonical_root);
        Self {
            root_dir,
            canonical_root,
            parent_ref_id,
            head_sha: None,
            cas_ref_id_hex: id.to_hex(),
            session_count: Arc::new(AtomicUsize::new(0)),
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
            identifier_index: Arc::new(RwLock::new(None)),
            search_index_cache: Arc::new(RwLock::new(None)),
            cache_generation: Arc::new(AtomicU64::new(0)),
            tracker_handle: Arc::new(std::sync::Mutex::new(None)),
            project_cache: Arc::new(RwLock::new(None)),
            memory_overlay: None,
        }
    }

    /// Build a new `RefIndex` with an explicit head SHA and **empty** caches.
    ///
    /// Used by `daemon::serve_connection` when a non-default worktree ref is
    /// registered for the first time (the bridge announces its `head_sha`).
    ///
    /// `memory_overlay` is intentionally left `None` here so the caller
    /// (`daemon::serve_connection`) can decide whether to attach a CoW overlay
    /// based on runtime context (e.g. whether the ref is truly non-primary).
    /// The daemon attaches the overlay immediately after calling `attach_ref`.
    /// Tests that check "overlay is None → daemon skips merge" remain valid.
    pub fn new_with_head(
        root_dir: PathBuf,
        canonical_root: PathBuf,
        parent_ref_id: Option<RefId>,
        head_sha: String,
    ) -> Self {
        let id = RefId::for_canonical_path(&canonical_root);
        Self {
            root_dir,
            canonical_root,
            parent_ref_id,
            head_sha: Some(head_sha),
            cas_ref_id_hex: id.to_hex(),
            session_count: Arc::new(AtomicUsize::new(0)),
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
            identifier_index: Arc::new(RwLock::new(None)),
            search_index_cache: Arc::new(RwLock::new(None)),
            cache_generation: Arc::new(AtomicU64::new(0)),
            tracker_handle: Arc::new(std::sync::Mutex::new(None)),
            project_cache: Arc::new(RwLock::new(None)),
            memory_overlay: None,
        }
    }

    /// Build the **default ref's** `RefIndex` with a pre-loaded embedding cache.
    ///
    /// Called by `ContextPlusServer::new` so the primary ref carries the
    /// disk-persisted embedding cache from the prior session without an
    /// additional cold-start re-embedding pass.  All other caches start empty.
    ///
    /// The returned `Arc<RwLock<HashMap<...>>>` fields are then **Arc-cloned**
    /// into `SharedState` as backward-compat shims (U10 → U11 transition).
    pub fn with_preloaded_cache(
        root_dir: PathBuf,
        canonical_root: PathBuf,
        initial_embedding_cache: HashMap<String, CacheEntry>,
    ) -> Self {
        let id = RefId::for_canonical_path(&canonical_root);
        Self {
            root_dir,
            canonical_root,
            parent_ref_id: None,
            head_sha: None,
            cas_ref_id_hex: id.to_hex(),
            session_count: Arc::new(AtomicUsize::new(0)),
            embedding_cache: Arc::new(RwLock::new(initial_embedding_cache)),
            identifier_index: Arc::new(RwLock::new(None)),
            search_index_cache: Arc::new(RwLock::new(None)),
            cache_generation: Arc::new(AtomicU64::new(0)),
            tracker_handle: Arc::new(std::sync::Mutex::new(None)),
            project_cache: Arc::new(RwLock::new(None)),
            memory_overlay: None, // primary ref has no overlay
        }
    }

    /// Initialize the CAS on-disk layout for this ref.
    ///
    /// For **primary refs** (no parent): creates the ref directory under
    /// `.mcp_data/refs/<hex>/` with an empty manifest. The `parent` file is
    /// omitted (no parent chain).
    ///
    /// For **forked refs**: creates the ref directory and writes the parent
    /// pointer so chunk lookups chain through the parent's manifest.
    ///
    /// This is the U4 seam: today called at daemon construction for the single
    /// default ref. U4's `attach_ref` will call it after resolving fork base
    /// from `merge-base HEAD <upstream>`.
    ///
    /// `mcp_data_dir` — path to the primary worktree's `.mcp_data/` directory.
    /// `model` — embedding model name (used for CAS shard naming).
    pub fn fork_from(
        &self,
        mcp_data_dir: &Path,
        model: &str,
        parent: Option<&RefIndex>,
    ) -> crate::error::Result<()> {
        use crate::cache::cas::CasStore;
        let cas = CasStore::new(mcp_data_dir.to_path_buf(), model);
        match parent {
            None => {
                // Primary ref: just ensure the directory exists.
                let dir = mcp_data_dir.join("refs").join(&self.cas_ref_id_hex);
                std::fs::create_dir_all(&dir)?;
                // Write an empty manifest if none exists yet.
                if !dir.join("manifest.rkyv").exists() {
                    let empty = crate::cache::cas::Manifest::new();
                    cas.save_manifest(&self.cas_ref_id_hex, &empty)?;
                }
                Ok(())
            }
            Some(p) => {
                cas.fork_ref(&self.cas_ref_id_hex, &p.cas_ref_id_hex)?;
                Ok(())
            }
        }
    }
}

/// The daemon's ref registry. Wrapped in `tokio::sync::RwLock` so attach +
/// detach (writes) cannot block tool dispatches (reads).
pub type RefRegistry = RwLock<HashMap<RefId, Arc<RefIndex>>>;

/// Construct a registry pre-seeded with one default ref.
///
/// `SharedState::new` is called from synchronous initialization paths
/// (including `Test` blocks within an async runtime), so we cannot acquire
/// the lock at construction time — `blocking_write` would panic when called
/// from an async context. Instead, we build the inner map first and only
/// wrap it in the `RwLock` once it's already populated.
pub fn new_registry_with_default(default_ref_id: RefId, default_ref: Arc<RefIndex>) -> RefRegistry {
    let mut map = HashMap::new();
    map.insert(default_ref_id, default_ref);
    RwLock::new(map)
}

/// Construct an empty registry. Used by tests.
#[cfg(test)]
pub fn new_registry() -> RefRegistry {
    RwLock::new(HashMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ref_id_is_stable_for_same_path() {
        let p = PathBuf::from("/tmp/some/repo");
        let a = RefId::for_canonical_path(&p);
        let b = RefId::for_canonical_path(&p);
        assert_eq!(a, b);
    }

    #[test]
    fn ref_id_differs_for_different_paths() {
        let p1 = PathBuf::from("/tmp/repo-a");
        let p2 = PathBuf::from("/tmp/repo-b");
        assert_ne!(
            RefId::for_canonical_path(&p1),
            RefId::for_canonical_path(&p2)
        );
    }

    #[test]
    fn ref_index_constructor_initializes_head_sha_to_none() {
        let r = RefIndex::new(PathBuf::from("/tmp/r"), PathBuf::from("/tmp/r"), None);
        assert!(r.head_sha.is_none());
        assert!(r.parent_ref_id.is_none());
    }

    #[test]
    fn registry_starts_empty() {
        let reg = new_registry();
        let inner = reg.try_read().unwrap();
        assert!(inner.is_empty());
    }

    #[test]
    fn registry_with_default_holds_one_entry() {
        let p = PathBuf::from("/tmp/r");
        let id = RefId::for_canonical_path(&p);
        let r = Arc::new(RefIndex::new(p.clone(), p, None));
        let reg = new_registry_with_default(id, r.clone());
        let inner = reg.try_read().unwrap();
        assert_eq!(inner.len(), 1);
        assert!(inner.get(&id).is_some());
        // The Arc returned via the registry should be the same as the one
        // we inserted (so call sites can share, not copy).
        assert!(Arc::ptr_eq(inner.get(&id).unwrap(), &r));
    }

    // ---- U6: BLAKE3-based RefId stability ----

    #[test]
    fn ref_id_hex_is_16_chars() {
        let p = PathBuf::from("/tmp/repo");
        let id = RefId::for_canonical_path(&p);
        assert_eq!(id.to_hex().len(), 16);
    }

    #[test]
    fn ref_id_hex_is_stable_across_calls() {
        let p = PathBuf::from("/home/user/repos/my-project");
        let a = RefId::for_canonical_path(&p);
        let b = RefId::for_canonical_path(&p);
        assert_eq!(a.to_hex(), b.to_hex());
    }

    #[test]
    fn ref_index_has_cas_ref_id_hex() {
        let p = PathBuf::from("/tmp/myrepo");
        let r = RefIndex::new(p.clone(), p.clone(), None);
        let id = RefId::for_canonical_path(&p);
        assert_eq!(r.cas_ref_id_hex, id.to_hex());
        assert_eq!(r.cas_ref_id_hex.len(), 16);
    }

    // ---- U6: fork_from disk layout ----

    #[test]
    fn fork_from_primary_creates_manifest_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mcp_data = tmp.path().join(".mcp_data");
        std::fs::create_dir_all(&mcp_data).unwrap();

        let p = tmp.path().join("primary_repo");
        std::fs::create_dir_all(&p).unwrap();
        let primary = RefIndex::new(p.clone(), p, None);
        primary
            .fork_from(&mcp_data, "nomic-embed-text", None)
            .unwrap();

        let ref_dir = mcp_data.join("refs").join(&primary.cas_ref_id_hex);
        assert!(ref_dir.exists(), "ref dir should be created");
        assert!(
            ref_dir.join("manifest.rkyv").exists(),
            "manifest should be created"
        );
    }

    #[test]
    fn fork_from_child_writes_parent_pointer() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mcp_data = tmp.path().join(".mcp_data");
        std::fs::create_dir_all(&mcp_data).unwrap();

        let pp = tmp.path().join("primary");
        std::fs::create_dir_all(&pp).unwrap();
        let primary = RefIndex::new(pp.clone(), pp, None);
        primary.fork_from(&mcp_data, "model", None).unwrap();

        let cp = tmp.path().join("child");
        std::fs::create_dir_all(&cp).unwrap();
        let child = RefIndex::new(
            cp.clone(),
            cp,
            Some(RefId::for_canonical_path(&primary.canonical_root)),
        );
        child.fork_from(&mcp_data, "model", Some(&primary)).unwrap();

        let parent_file = mcp_data
            .join("refs")
            .join(&child.cas_ref_id_hex)
            .join("parent");
        assert!(parent_file.exists());
        let content = std::fs::read_to_string(&parent_file).unwrap();
        assert_eq!(content.trim(), primary.cas_ref_id_hex);
    }

    // ── U10: per-ref cache fields ─────────────────────────────────────────────

    /// `RefIndex::new` initializes all cache fields to empty/None.
    #[test]
    fn ref_index_new_has_empty_caches() {
        let p = PathBuf::from("/tmp/u10-test-empty");
        let r = RefIndex::new(p.clone(), p, None);

        // embedding_cache starts empty
        assert!(r.embedding_cache.try_read().unwrap().is_empty());
        // identifier_index starts None
        assert!(r.identifier_index.try_read().unwrap().is_none());
        // search_index_cache starts None
        assert!(r.search_index_cache.try_read().unwrap().is_none());
        // cache_generation starts at 0
        assert_eq!(
            r.cache_generation
                .load(std::sync::atomic::Ordering::Acquire),
            0
        );
        // tracker_handle starts None
        assert!(r.tracker_handle.lock().unwrap().is_none());
        // project_cache starts None
        assert!(r.project_cache.try_read().unwrap().is_none());
        // primary ref has no memory overlay
        assert!(r.memory_overlay.is_none());
    }

    /// `RefIndex::new_with_head` leaves `memory_overlay = None` — the daemon
    /// attaches the overlay after construction (see `daemon::serve_connection`).
    #[test]
    fn ref_index_new_with_head_has_no_overlay_by_default() {
        let p = PathBuf::from("/tmp/u10-test-overlay");
        let r = RefIndex::new_with_head(p.clone(), p, None, "abcdef".to_string());
        assert!(
            r.memory_overlay.is_none(),
            "new_with_head must not pre-attach an overlay — daemon does it post-construction"
        );
    }

    /// `RefIndex::with_preloaded_cache` carries the supplied embedding map.
    #[test]
    fn ref_index_with_preloaded_cache_holds_initial_embeddings() {
        let p = PathBuf::from("/tmp/u10-test-preloaded");
        let mut initial: HashMap<String, CacheEntry> = HashMap::new();
        initial.insert(
            "src/main.rs".to_string(),
            CacheEntry {
                hash: "deadbeef".to_string(),
                vector: vec![0.1, 0.2, 0.3],
            },
        );
        let r = RefIndex::with_preloaded_cache(p.clone(), p, initial);

        let cache = r.embedding_cache.try_read().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key("src/main.rs"));
        // primary ref built with this constructor has no overlay
        assert!(r.memory_overlay.is_none());
    }

    /// Writes to one ref's `embedding_cache` do NOT appear in another ref's cache.
    #[tokio::test]
    async fn ref_caches_are_isolated() {
        let p1 = PathBuf::from("/tmp/u10-iso-ref1");
        let p2 = PathBuf::from("/tmp/u10-iso-ref2");
        let ref1 = RefIndex::new(p1.clone(), p1, None);
        let ref2 = RefIndex::new(p2.clone(), p2, None);

        // Write into ref1's embedding cache.
        ref1.embedding_cache.write().await.insert(
            "a.rs".to_string(),
            CacheEntry {
                hash: "h1".to_string(),
                vector: vec![1.0],
            },
        );

        // ref2's cache remains empty.
        assert!(
            ref2.embedding_cache.read().await.is_empty(),
            "ref2 cache must not be affected by writes to ref1"
        );
    }
}
