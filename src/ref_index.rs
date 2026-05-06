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
//! * [`RefIndex`] — per-ref identity metadata (root, parent, head). Heavy
//!   per-ref state (caches, memory graph) is **not yet** carried here;
//!   it remains on `SharedState` until U4/U6 migrate it.
//! * [`RefRegistry`] — the daemon's `HashMap<RefId, Arc<RefIndex>>`,
//!   wrapped in a `RwLock` so attach/detach can land without coordination
//!   with read-only tool dispatches.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::RwLock;

/// Stable identifier for a ref (worktree + HEAD).
///
/// Derived from the canonical path of the worktree root. Identical canonical
/// paths produce identical [`RefId`]s — which is the property that lets two
/// bridges connecting to the same worktree converge on one shared
/// [`RefIndex`] without explicit coordination.
///
/// The current derivation uses the standard library's hasher for simplicity
/// and zero-cost. U6 may switch to BLAKE3 if cross-process stability across
/// daemon restarts becomes load-bearing for the on-disk CAS layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RefId(pub u64);

impl RefId {
    /// Compute the [`RefId`] for a canonical worktree path.
    ///
    /// Caller is responsible for passing a canonicalized path — non-canonical
    /// inputs will produce non-canonical ids that may not match across calls.
    /// Use [`crate::core::git_worktree::resolve_primary_worktree`] +
    /// `canonicalize()` before constructing.
    pub fn for_canonical_path(path: &Path) -> Self {
        use std::hash::{Hash, Hasher};
        // `DefaultHasher` is deterministic within a single process (it uses
        // SipHash 1-3 with fixed keys when constructed via `new`). That's
        // sufficient for routing within a daemon's lifetime. U6 will switch
        // to BLAKE3 when the on-disk CAS layout needs cross-restart-stable
        // keys.
        let mut hasher = std::hash::DefaultHasher::new();
        path.hash(&mut hasher);
        RefId(hasher.finish())
    }
}

/// Per-ref identity metadata held by the daemon's registry.
///
/// In U3 this struct only holds identification fields. U4 will add the
/// session refcount + parent linkage; U6 + U7 will migrate per-ref caches
/// and memory-graph overlay state into it.
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
}

impl RefIndex {
    /// Build a new `RefIndex` from a freshly-attached session.
    pub fn new(root_dir: PathBuf, canonical_root: PathBuf, parent_ref_id: Option<RefId>) -> Self {
        Self {
            root_dir,
            canonical_root,
            parent_ref_id,
            head_sha: None,
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
}
