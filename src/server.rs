// MCP server wiring — dispatches tool calls to underlying implementations.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use regex::Regex;
use rmcp::RoleServer;
use rmcp::handler::server::ServerHandler;
use rmcp::model::*;
use rmcp::service::RequestContext;
use serde_json::Value;
use tokio::sync::{OnceCell, RwLock, Semaphore};

use crate::cache::rkyv_store;
use crate::config::{Config, EmbedDocShape, RefWarmupMode, TrackerMode};
use crate::core::embedding_tracker::{
    EmbeddingTrackerConfig, EmbeddingTrackerHandle, RefreshCallback,
};
use crate::core::embeddings::{CacheEntry, OllamaClient};
use crate::core::structural_pool::STRUCTURAL_POOL;
use crate::core::tree_sitter::parse_with_tree_sitter;
use crate::core::walker::walk_with_config;
use crate::error::{ContextPlusError, Result};
use crate::server_adapters::{CachedWalkerIndexer, OllamaEmbedder};
pub use crate::server_definitions::{make_tool, tool_definitions};

pub(crate) mod snapshots;

/// Cached project state: walked file entries and their raw file contents.
/// Built lazily on first tool call. A running tracker invalidates it on file
/// changes; the TTL is a fallback only when no tracker is running for the ref.
/// Content is stored as `Arc<String>` so call-sites can clone the pointer
/// (cheap) and use `content.lines()` when line iteration is needed, avoiding
/// the `Vec<String>` split + `join("\n")` round-trip on every access.
#[derive(Clone)]
pub struct ProjectCache {
    pub file_entries: Vec<crate::core::walker::FileEntry>,
    /// Maps relative_path → raw file content. Clone the Arc (pointer-sized)
    /// at each call-site; do not reconstruct from lines.
    pub file_content: crate::core::walker::FileContents,
    /// The files git showed clean, with their blobs, both before and after
    /// the walk that read `file_content`. Recorded for a primary: a linked
    /// worktree skips reading its files that have the same clean blob.
    pub clean_blobs: Option<Arc<crate::core::git_worktree::CleanBlobs>>,
    pub last_refresh: Instant,
}

/// Cached lexical index and the document paths used to format its results.
#[derive(Clone)]
pub(crate) struct CachedLexicalIndex {
    /// The whole corpus, or with `base` only this ref's changed and added files.
    pub index: crate::tools::lexical_search::LexicalIndex,
    pub document_paths: Vec<String>,
    pub project_cache: Arc<ProjectCache>,
    pub generation: u64,
    pub base: Option<LexicalBase>,
}

/// The parent ref's shared index with the documents this ref changed or removed masked.
#[derive(Clone)]
pub(crate) struct LexicalBase {
    pub cached: Arc<CachedLexicalIndex>,
    pub mask: crate::tools::lexical_search::BaseMask,
}

impl CachedLexicalIndex {
    pub(crate) fn is_empty(&self) -> bool {
        self.document_paths.is_empty()
            && self
                .base
                .as_ref()
                .is_none_or(|base| base.cached.document_paths.is_empty())
    }

    /// Top `top_k` `(path, score)` hits, best first, equal scores by path.
    pub(crate) fn search(&self, query: &str, top_k: usize) -> Vec<(&str, f64)> {
        let path = |in_delta: bool, i: usize| {
            let paths = match &self.base {
                Some(base) if !in_delta => &base.cached.document_paths,
                _ => &self.document_paths,
            };
            Some(paths.get(i)?.as_str())
        };
        let mut hits: Vec<(&str, f64)> = match &self.base {
            Some(base) => self
                .index
                .search_over(&base.cached.index, &base.mask, query)
                .into_iter()
                .filter_map(|(in_delta, i, score)| Some((path(in_delta, i)?, score)))
                .collect(),
            None => self
                .index
                .search(query, usize::MAX)
                .into_iter()
                .filter_map(|(i, score)| Some((path(true, i)?, score)))
                .collect(),
        };
        hits.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        });
        hits.truncate(top_k);
        hits
    }

    /// Bytes this entry holds on its own, excluding a shared base.
    pub(crate) fn own_resident_bytes(&self) -> usize {
        self.index.estimated_resident_bytes()
            + self
                .base
                .as_ref()
                .map_or(0, |base| base.mask.estimated_resident_bytes())
    }
}

/// Cached identifier index: parsed symbols + their embedding vectors.
/// Rebuilt when file count changes. The 300-second TTL is a fallback only when
/// no tracker is running for the ref.
pub struct IdentifierIndex {
    pub docs: Segmented<crate::tools::semantic_identifiers::IdentifierDoc>,
    pub vectors: IdentifierVectorIndex,
    pub dims: usize,
    pub file_count: usize,
    pub built_at: Instant,
}

/// Identifier embeddings keyed by identifier text. An identifier index holds
/// the same allocations, so a vector is resident once however many indexes use it.
pub(crate) type IdentifierVectors = HashMap<String, Arc<[f32]>>;

/// One embedding per identifier, grouped by file like the index documents and
/// read as a flat buffer of `dims` floats per identifier.
#[derive(Clone)]
pub struct IdentifierVectorIndex {
    segments: Segmented<Arc<[f32]>>,
    dims: usize,
}

impl IdentifierVectorIndex {
    fn new(files: std::collections::BTreeMap<String, Arc<Vec<Arc<[f32]>>>>, dims: usize) -> Self {
        Self {
            segments: Segmented::from_files(files),
            dims,
        }
    }

    pub fn empty() -> Self {
        Self::new(std::collections::BTreeMap::new(), 0)
    }

    /// One vector per identifier; every vector must have `dims` floats.
    pub fn from_vectors(vectors: Vec<Vec<f32>>, dims: usize) -> Self {
        let vectors = vectors.into_iter().map(Arc::from).collect();
        Self::new(
            std::collections::BTreeMap::from([(String::new(), Arc::new(vectors))]),
            dims,
        )
    }

    /// Floats across all identifiers.
    pub fn len(&self) -> usize {
        self.segments.len() * self.dims
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The vector of identifier `index`.
    pub fn vector(&self, index: usize) -> &Arc<[f32]> {
        crate::tools::semantic_identifiers::IndexData::get(&self.segments, index)
    }

    fn file_segments(&self) -> &std::collections::BTreeMap<String, Arc<Vec<Arc<[f32]>>>> {
        &self.segments.files
    }
}

impl crate::tools::semantic_identifiers::IndexData<f32> for IdentifierVectorIndex {
    fn len(&self) -> usize {
        IdentifierVectorIndex::len(self)
    }
    fn get(&self, index: usize) -> &f32 {
        &self.vector(index / self.dims)[index % self.dims]
    }
    fn slice(&self, range: std::ops::Range<usize>) -> &[f32] {
        if range.is_empty() {
            return &[];
        }
        let local = range.start % self.dims;
        &self.vector(range.start / self.dims)[local..local + range.len()]
    }
}

#[derive(Clone)]
pub struct Segmented<T> {
    files: std::collections::BTreeMap<String, Arc<Vec<T>>>,
    offsets: Vec<(usize, String)>,
    len: usize,
}

impl<T> Segmented<T> {
    fn from_files(files: std::collections::BTreeMap<String, Arc<Vec<T>>>) -> Self {
        let mut len = 0;
        let offsets = files
            .iter()
            .filter(|(_, values)| !values.is_empty())
            .map(|(path, values)| {
                let offset = len;
                len += values.len();
                (offset, path.clone())
            })
            .collect();
        Self {
            files,
            offsets,
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.files.values().flat_map(|values| values.iter())
    }

    fn locate(&self, index: usize) -> (&[T], usize) {
        let position = self.offsets.partition_point(|(offset, _)| *offset <= index) - 1;
        let (offset, path) = &self.offsets[position];
        (&self.files[path], index - offset)
    }
}

impl<T> From<Vec<T>> for Segmented<T> {
    fn from(values: Vec<T>) -> Self {
        Self::from_files(std::collections::BTreeMap::from([(
            String::new(),
            Arc::new(values),
        )]))
    }
}

impl<T: Send + Sync> crate::tools::semantic_identifiers::IndexData<T> for Segmented<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, index: usize) -> &T {
        let (values, local) = self.locate(index);
        &values[local]
    }
    fn slice(&self, range: std::ops::Range<usize>) -> &[T] {
        let (values, local) = self.locate(range.start);
        &values[local..local + range.len()]
    }
}

struct RefreshGuard(Arc<std::sync::atomic::AtomicBool>);
impl Drop for RefreshGuard {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::Release);
    }
}

struct IncrementalReembedOutcome {
    updated: usize,
    skipped: usize,
    content_changed: bool,
}

const IDENTIFIER_INDEX_TTL_SECS: u64 = 300;

/// URL to fetch the instructions resource content from.
const INSTRUCTIONS_SOURCE_URL: &str = "https://contextplus.vercel.app/api/instructions";
/// MCP resource URI for the instructions resource.
const INSTRUCTIONS_RESOURCE_URI: &str = "contextplus://instructions";

/// Shared state accessible by all tool handlers.
///
/// ## U10 migration note
///
/// The per-ref cache fields (`embedding_cache`, `identifier_index`,
/// `search_index_cache`, `cache_generation`, `tracker_handle`, `project_cache`)
/// are now owned by [`crate::ref_index::RefIndex`].  During the U10→U11
/// transition they are kept here as **`Arc` clones** of the default ref's
/// fields.  This means all existing call sites (`state.embedding_cache.read()`,
/// etc.) continue to compile and operate on the correct data.
///
/// U11 will mechanically migrate the ~57 call sites to use
/// `state.ref_index(session.ref_id).await.embedding_cache` instead.
pub struct SharedState {
    pub config: Config,
    pub root_dir: PathBuf,
    /// Canonicalized version of root_dir — computed once at construction.
    /// Used by resolve_root() to validate caller-provided rootDir args without
    /// calling canonicalize() on every tool request.
    pub canonical_root: PathBuf,
    pub ollama: OllamaClient,
    /// **Backward-compat shim (U10).** Arc clone of the default ref's
    /// `project_cache`. Shares the same underlying `RwLock` as the default
    /// `RefIndex` entry. U11 will migrate call sites to use the per-ref field.
    pub project_cache: Arc<RwLock<Option<Arc<ProjectCache>>>>,
    /// **Backward-compat shim (U10).** Arc clone of the default ref's
    /// `embedding_cache`. All reads/writes go through the same lock as
    /// `state.default_ref().embedding_cache`. U11 migrates to per-ref.
    pub embedding_cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// **Backward-compat shim (U10).** Arc clone of the default ref's
    /// `identifier_index`. U11 migrates to per-ref.
    pub identifier_index: Arc<RwLock<Option<Arc<IdentifierIndex>>>>,
    /// Cached SearchIndex for semantic_code_search — reused when the walk fingerprint
    /// is unchanged, eliminating the per-request HNSW rebuild for large corpora.
    /// `Arc`-wrapped so background rebuild tasks can hold a clone.
    /// **U10**: also stored in the default ref's `search_index_cache` (same Arc).
    pub search_index_cache:
        Arc<RwLock<Option<Arc<crate::tools::semantic_search::CachedSearchIndex>>>>,
    /// Monotonic counter incremented by the embedding tracker whenever a file-change
    /// event is processed. `semantic_code_search` compares the counter at request time
    /// against `CachedSearchIndex::generation`; equality means the tracker has seen no
    /// changes since the last build and the filesystem walk can be skipped entirely.
    /// When the tracker is disabled the counter stays at 0 and the fingerprint-based
    /// fallback takes over.
    /// **U10**: same Arc stored in the default ref's `cache_generation`.
    pub cache_generation: Arc<std::sync::atomic::AtomicU64>,
    /// Cached instructions content — fetched once from remote, then served from memory.
    pub instructions_cache: OnceCell<String>,
    /// **Backward-compat shim (U10).** Arc clone of the default ref's
    /// `tracker_handle`. U11 migrates to per-ref.
    pub tracker_handle: Arc<std::sync::Mutex<Option<EmbeddingTrackerHandle>>>,
    /// Idle monitor handle — tool handlers touch this to reset the idle timer.
    pub idle_monitor: RwLock<Option<Arc<crate::core::process_lifecycle::IdleMonitor>>>,
    /// Drain flag — set when the parent process dies (or another graceful
    /// shutdown trigger fires). Once true, `dispatch` rejects new tool calls
    /// with a clean error so in-flight calls can finish before exit.
    pub draining: Arc<std::sync::atomic::AtomicBool>,
    /// Counter of currently in-flight tool dispatches. Incremented at the top
    /// of `dispatch_inner` via an `InflightGuard`; decremented on drop. The
    /// drain watcher exits the process once draining is true and this hits 0.
    pub inflight: Arc<std::sync::atomic::AtomicUsize>,
    /// Registry of `RefId → RefIndex` for all worktrees this daemon serves.
    /// Each `RefIndex` carries its own per-ref caches (U10+).
    /// The registry is `RwLock`-wrapped so attach/detach don't block tool calls.
    pub refs: crate::ref_index::RefRegistry,
    /// Permanent snapshot of the default ref. The primary ref is never evicted,
    /// so request-path fallback does not need to contend on the registry lock.
    default_ref: Arc<crate::ref_index::RefIndex>,
    /// `RefId` of the default (primary) ref. Tool dispatches that don't
    /// carry an explicit `RefId` use this. U4 / U11 will migrate to explicit
    /// per-session routing via `session_ref_id`.
    pub default_ref_id: crate::ref_index::RefId,
    /// Global Ollama embed semaphore. All callers that issue embed requests
    /// (on-demand tool calls, embedding tracker, ref warmup) must acquire a
    /// permit before calling Ollama so that N concurrent warmups cannot fan-out
    /// N×files requests and saturate a CPU-only Ollama instance.
    ///
    /// Capacity is set from `config.ollama_max_concurrent` at server construction
    /// and never changes for the lifetime of the daemon.
    pub ollama_semaphore: Arc<Semaphore>,
    /// Set of refs currently being warmed up (U18 idempotency guard).
    ///
    /// `spawn_ref_warmup` inserts the `RefId` before spawning the background
    /// task and removes it when the task completes via a `WarmupGuard` RAII
    /// drop. A second call for the same ref while the first is running is a
    /// no-op.
    pub warmup_in_flight:
        Arc<tokio::sync::Mutex<std::collections::HashSet<crate::ref_index::RefId>>>,
    access_clock: std::sync::atomic::AtomicU64,
    ref_access: std::sync::Mutex<HashMap<crate::ref_index::RefId, (u64, Instant)>>,
    budget_enforcement_running: std::sync::atomic::AtomicBool,
    last_budget_enforcement: std::sync::Mutex<Option<Instant>>,
    last_budget_trim: std::sync::Mutex<Option<Instant>>,
    /// When the under-budget path last asked the allocator how much it holds free.
    last_free_memory_check: std::sync::Mutex<Option<Instant>>,
    /// Set once the over-budget warning is logged, until memory is back under budget.
    budget_warned: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    pub(crate) measured_resident_override: std::sync::Mutex<Option<usize>>,
    #[cfg(test)]
    free_memory_checks: std::sync::atomic::AtomicUsize,
}

impl SharedState {
    /// Return a cloned `Arc` pointing at the global Ollama embed semaphore.
    ///
    /// Callers in U17 will hold this semaphore for the duration of each Ollama
    /// embed request. Callers in U18 will hold it across a full ref warmup pass.
    /// The permit count is fixed at `config.ollama_max_concurrent` for the
    /// lifetime of the daemon.
    pub fn ollama_semaphore(&self) -> Arc<Semaphore> {
        Arc::clone(&self.ollama_semaphore)
    }

    /// Look up the default ref's index. Today this is the only entry in the
    /// registry; U4 will introduce alternate refs.
    ///
    /// The `Option` return type is retained for compatibility; the permanent
    /// default snapshot always returns `Some`.
    pub fn default_ref(&self) -> Option<Arc<crate::ref_index::RefIndex>> {
        Some(Arc::clone(&self.default_ref))
    }

    /// Writes the primary checkout's snapshots that have unwritten changes.
    /// For a graceful shutdown.
    pub async fn flush_snapshots(&self) {
        let primary = Arc::clone(&self.default_ref);
        let config = self.config.clone();
        let _ = tokio::task::spawn_blocking(move || snapshots::flush(&config, &primary)).await;
    }

    /// Look up a ref by id. Reserved for U4's session-scoped dispatch.
    pub async fn ref_index(
        &self,
        id: crate::ref_index::RefId,
    ) -> Option<Arc<crate::ref_index::RefIndex>> {
        self.refs.read().await.get(&id).cloned()
    }

    /// Attach a session to an existing ref, or insert a new `RefIndex` and
    /// attach. Returns the `RefId` that the caller should use for subsequent
    /// tool dispatches.
    ///
    /// Concurrent calls from two bridges connecting to the same canonical root
    /// are safe: the write lock ensures only one `RefIndex` is inserted, and
    /// both callers will share the same `Arc<RefIndex>` after the lock is
    /// released.
    pub async fn attach_ref(
        &self,
        ref_id: crate::ref_index::RefId,
        make_ref: impl FnOnce() -> Arc<crate::ref_index::RefIndex>,
    ) -> Arc<crate::ref_index::RefIndex> {
        let existing = {
            let guard = self.refs.read().await;
            guard.get(&ref_id).cloned().inspect(|existing| {
                existing
                    .eviction_generation
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                existing
                    .session_count
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            })
        };
        if let Some(existing) = existing {
            self.touch_ref(ref_id);
            return existing;
        }

        let created = make_ref();
        if let Some(parent_id) = created.parent_ref_id {
            let parent = self.refs.read().await.get(&parent_id).cloned();
            if let Some(parent) = parent {
                if let Some(parent_vectors) = parent.identifier_vectors.get() {
                    let _ = created.identifier_vectors.set(Arc::clone(parent_vectors));
                }
                // Installs write the index and its source under the index write
                // lock, so reading both under one read guard cannot tear.
                let (identifier_index, identifier_source) = {
                    let index = parent.identifier_index.read().await;
                    let source = parent.identifier_source.read().await;
                    match (index.as_ref(), source.as_ref()) {
                        (Some(index), Some(source)) => {
                            (Some(Arc::clone(index)), Some(Arc::clone(source)))
                        }
                        _ => (None, None),
                    }
                };
                created.identifier_inherited.store(
                    identifier_index.is_some(),
                    std::sync::atomic::Ordering::Release,
                );
                *created.identifier_index.write().await = identifier_index;
                *created.identifier_source.write().await = identifier_source;
                let lexical = parent.lexical_search_cache.read().await.as_ref().cloned();
                created
                    .lexical_inherited
                    .store(lexical.is_some(), std::sync::atomic::Ordering::Release);
                *created.lexical_search_cache.write().await = lexical;
            }
        }

        let mut guard = self.refs.write().await;
        let entry = guard
            .entry(ref_id)
            .or_insert_with(|| Arc::clone(&created))
            .clone();
        entry
            .eviction_generation
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        entry
            .session_count
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        drop(guard);
        self.touch_ref(ref_id);
        entry
    }

    pub fn touch_ref(&self, ref_id: crate::ref_index::RefId) {
        let tick = self
            .access_clock
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .wrapping_add(1);
        self.ref_access
            .lock()
            .unwrap()
            .insert(ref_id, (tick, Instant::now()));
    }

    /// Detach a session from a ref. Decrements the session refcount. When the
    /// count reaches zero the ref enters the TTL eviction queue (spawned as a
    /// background task). The primary ref (matching `default_ref_id`) is never
    /// evicted from the registry regardless of refcount.
    ///
    /// `ttl` — how long after last-session-disconnect before the in-memory
    /// `RefIndex` is removed. `0` means immediate removal. The on-disk overlay
    /// (U6) is not touched here; only the in-memory registry entry is dropped.
    pub async fn detach_ref(
        self: &Arc<Self>,
        ref_id: crate::ref_index::RefId,
        ttl: std::time::Duration,
    ) {
        let count = {
            let guard = self.refs.read().await;
            guard
                .get(&ref_id)
                .map(|r| {
                    r.session_count
                        .fetch_sub(1, std::sync::atomic::Ordering::AcqRel)
                })
                .unwrap_or(0)
        };
        // count is the *pre-decrement* value; after this call it is count - 1.
        // Only schedule eviction when transitioning to 0.
        if count != 1 {
            return;
        }
        // Primary ref is never evicted — the daemon owns it for its lifetime.
        if ref_id == self.default_ref_id {
            return;
        }
        let epoch = {
            let guard = self.refs.read().await;
            let Some(owner) = guard.get(&ref_id) else {
                return;
            };
            owner
                .eviction_generation
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                + 1
        };
        let state = Arc::clone(self);
        tokio::spawn(async move {
            if !ttl.is_zero() {
                tokio::time::sleep(ttl).await;
            }
            // Re-check the refcount after the TTL — a new session may have
            // re-attached in the interim.
            let guard = state.refs.read().await;
            if let Some(r) = guard.get(&ref_id) {
                if r.session_count.load(std::sync::atomic::Ordering::Acquire) > 0
                    || r.eviction_generation
                        .load(std::sync::atomic::Ordering::Acquire)
                        != epoch
                {
                    tracing::debug!(
                        ref_id = ref_id.0,
                        "TTL eviction cancelled — new session attached"
                    );
                    return;
                }
            } else {
                return; // already removed
            }
            drop(guard);
            let mut guard = state.refs.write().await;
            // Double-check under write lock.
            if let Some(r) = guard.get(&ref_id)
                && r.session_count.load(std::sync::atomic::Ordering::Acquire) == 0
                && r.eviction_generation
                    .load(std::sync::atomic::Ordering::Acquire)
                    == epoch
            {
                r.cancel_background_tasks();
                r.tracker_handle
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take();
                guard.remove(&ref_id);
                state.ref_access.lock().unwrap().remove(&ref_id);
                tracing::info!(ref_id = ref_id.0, "ref evicted after TTL expiry");
            }
        });
    }

    /// Keeps the process within `resident_memory_budget_bytes` of RAM. The
    /// trigger is measured process memory; per-ref estimates only pick which
    /// idle worktrees to evict, least recently used first, down to 80% of the
    /// budget. The primary is never evicted: when it alone exceeds the budget
    /// one warning names its heaviest structures.
    pub async fn enforce_memory_budget(&self) {
        let refs: Vec<_> = self.refs.read().await.values().cloned().collect();
        let mut snapshots = Vec::with_capacity(refs.len());
        for owner in &refs {
            snapshots.push(ResidentSnapshot::capture(owner).await);
        }
        let Ok(components) = tokio::task::spawn_blocking(move || {
            snapshots
                .into_iter()
                .map(ResidentSnapshot::measure)
                .collect::<Vec<_>>()
        })
        .await
        else {
            return;
        };

        let mut holders: HashMap<usize, (usize, usize)> = HashMap::new();
        for &(ptr, bytes, _) in components.iter().flatten() {
            holders.entry(ptr).or_insert((bytes, 0)).1 += 1;
        }
        let estimated = holders
            .values()
            .fold(0usize, |total, (bytes, _)| total.saturating_add(*bytes));
        let budget = self.config.resident_memory_budget_bytes;
        let measured = self.measured_resident_bytes(estimated);
        if measured <= budget {
            self.budget_warned
                .store(false, std::sync::atomic::Ordering::Release);
            tracing::debug!(measured, estimated, budget, "Resident memory under budget");
            if !self.trim_due() || !interval_elapsed(&self.last_free_memory_check) {
                return;
            }
            *self.last_free_memory_check.lock().unwrap() = Some(Instant::now());
            #[cfg(test)]
            self.free_memory_checks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Index builds leave freed memory the allocator keeps for reuse.
            let threshold = MEMORY_RETAINED_FREE_TRIM_BYTES.max(budget / 8);
            let trimmed =
                tokio::task::spawn_blocking(move || trim_retained_free_memory(measured, threshold))
                    .await
                    .unwrap_or(false);
            if trimmed {
                *self.last_budget_trim.lock().unwrap() = Some(Instant::now());
            }
            return;
        }
        let low_watermark = budget / 5 * 4;
        let access = self.ref_access.lock().unwrap().clone();
        let mut candidates: Vec<_> = (0..refs.len())
            .filter_map(|i| {
                if Arc::ptr_eq(&refs[i], &self.default_ref) {
                    return None;
                }
                let id = crate::ref_index::RefId::for_canonical_path(&refs[i].canonical_root);
                let Some(&(tick, last_used)) = access.get(&id) else {
                    return Some((0, i));
                };
                (last_used.elapsed() >= MEMORY_BUDGET_MIN_IDLE).then_some((tick, i))
            })
            .collect();
        candidates.sort_unstable();
        let id_cache_name = cache_name("identifier-embeddings", &self.config);
        let mut expected = measured;
        let mut evicted = 0usize;
        for (_, i) in candidates {
            if expected <= low_watermark {
                break;
            }
            let owner = &refs[i];
            let holds_unique_bytes = components[i].iter().any(|(ptr, _, _)| holders[ptr].1 == 1);
            if !holds_unique_bytes
                || owner
                    .active_requests
                    .load(std::sync::atomic::Ordering::Acquire)
                    > 0
            {
                continue;
            }
            clear_ref_heavy_caches(owner, &id_cache_name).await;
            evicted += 1;
            for (ptr, _, _) in &components[i] {
                let holder = holders.get_mut(ptr).unwrap();
                holder.1 -= 1;
                if holder.1 == 0 {
                    expected = expected.saturating_sub(holder.0);
                }
            }
        }
        if evicted > 0 || self.trim_due() {
            self.trim_free_memory().await;
        }
        let estimated_after = holders
            .values()
            .filter(|(_, count)| *count > 0)
            .fold(0usize, |total, (bytes, _)| total.saturating_add(*bytes));
        let measured_after = self.measured_resident_bytes(estimated_after);
        if measured_after <= budget {
            self.budget_warned
                .store(false, std::sync::atomic::Ordering::Release);
            return;
        }
        if self
            .budget_warned
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            return;
        }
        let mib = |bytes: usize| bytes / (1024 * 1024);
        let primary = refs
            .iter()
            .position(|owner| Arc::ptr_eq(owner, &self.default_ref));
        let mut breakdown: BTreeMap<&'static str, usize> = BTreeMap::new();
        for (_, bytes, name) in primary.map_or(&[][..], |i| &components[i][..]) {
            *breakdown.entry(name).or_default() += bytes;
        }
        let breakdown = breakdown
            .iter()
            .map(|(name, bytes)| format!("{name}={}MiB", mib(*bytes)))
            .collect::<Vec<_>>()
            .join(" ");
        tracing::warn!(
            measured_mib = mib(measured_after),
            budget_mib = mib(budget),
            evicted_worktrees = evicted,
            "Resident memory is over CONTEXTPLUS_MEMORY_BUDGET_MB after evicting every idle \
             worktree it could; the primary checkout is never evicted. Estimated primary \
             structures: {breakdown}"
        );
    }

    fn trim_due(&self) -> bool {
        interval_elapsed(&self.last_budget_trim)
    }

    async fn trim_free_memory(&self) {
        release_free_memory().await;
        *self.last_budget_trim.lock().unwrap() = Some(Instant::now());
    }

    /// Bytes of RAM the process holds. Tests share one process, so there the
    /// estimate stands in unless a test sets a measurement.
    fn measured_resident_bytes(&self, estimated: usize) -> usize {
        #[cfg(test)]
        {
            self.measured_resident_override
                .lock()
                .unwrap()
                .unwrap_or(estimated)
        }
        #[cfg(not(test))]
        {
            process_resident_bytes().unwrap_or(estimated)
        }
    }

    pub fn schedule_memory_budget_enforcement(self: &Arc<Self>) {
        if self
            .budget_enforcement_running
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }
        let guard = BudgetEnforcementGuard(Arc::clone(self));
        tokio::spawn(async move {
            let state = &guard.0;
            for _ in 0..20 {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                if state.inflight.load(std::sync::atomic::Ordering::Acquire) == 0 {
                    break;
                }
            }
            let last = *state.last_budget_enforcement.lock().unwrap();
            if let Some(last) = last {
                tokio::time::sleep(MEMORY_BUDGET_MIN_INTERVAL.saturating_sub(last.elapsed())).await;
            }
            state.enforce_memory_budget().await;
            *state.last_budget_enforcement.lock().unwrap() = Some(Instant::now());
        });
    }
}

const MEMORY_BUDGET_MIN_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);
/// While memory stays over budget with nothing left to evict, freed memory is
/// returned to the OS at most this often.
const MEMORY_BUDGET_TRIM_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

/// Resident set size of this process, from `/proc/self/statm`.
#[cfg(any(not(test), target_os = "linux"))]
fn process_resident_bytes() -> Option<usize> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
    let page_size = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
    Some(pages.saturating_mul(page_size))
}

/// Freed memory the allocator may keep before it is returned to the OS while
/// the process is under budget.
const MEMORY_RETAINED_FREE_TRIM_BYTES: usize = 256 * 1024 * 1024;

/// Bytes the allocator has handed out and not had back.
fn allocator_in_use_bytes() -> Option<usize> {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        let info = unsafe { libc::mallinfo2() };
        Some(info.uordblks + info.hblkhd)
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    None
}

fn interval_elapsed(last: &std::sync::Mutex<Option<Instant>>) -> bool {
    last.lock()
        .unwrap()
        .is_none_or(|last| last.elapsed() >= MEMORY_BUDGET_TRIM_INTERVAL)
}

/// Returns freed memory to the OS when the allocator holds more than
/// `threshold` of `measured` free. Walks the allocator's bins, so it runs off
/// the async runtime.
fn trim_retained_free_memory(measured: usize, threshold: usize) -> bool {
    let retained_free =
        allocator_in_use_bytes().map_or(0, |in_use| measured.saturating_sub(in_use));
    tracing::debug!(measured, retained_free, threshold, "Allocator free memory");
    if retained_free <= threshold {
        return false;
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
    true
}

/// Returns memory the allocator holds free to the OS.
async fn release_free_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    let _ = tokio::task::spawn_blocking(|| unsafe { libc::malloc_trim(0) }).await;
}
/// A worktree used more recently than this is never evicted, so concurrently
/// active worktrees cannot evict each other into repeated cold rebuilds.
const MEMORY_BUDGET_MIN_IDLE: std::time::Duration = std::time::Duration::from_secs(60);

struct BudgetEnforcementGuard(Arc<SharedState>);

impl Drop for BudgetEnforcementGuard {
    fn drop(&mut self) {
        self.0
            .budget_enforcement_running
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Heavy per-ref structures captured under short read locks. Map sizes are
/// estimated in O(1); the `Arc` snapshots are walked off the async runtime.
/// A heavy structure: its address (shared structures count once), estimated
/// bytes and name.
type ResidentComponent = (usize, usize, &'static str);

struct ResidentSnapshot {
    components: Vec<ResidentComponent>,
    identifier_index: Option<Arc<IdentifierIndex>>,
    search_index: Option<Arc<crate::tools::semantic_search::CachedSearchIndex>>,
    project_cache: Option<Arc<ProjectCache>>,
    lexical: Option<Arc<CachedLexicalIndex>>,
}

impl ResidentSnapshot {
    async fn capture(owner: &crate::ref_index::RefIndex) -> Self {
        let mut components = Vec::new();
        components.push((
            Arc::as_ptr(&owner.embedding_cache) as usize,
            cache_entry_bytes(&*owner.embedding_cache.read().await),
            "file_vectors",
        ));
        if let Some(vectors) = owner.identifier_vectors.get() {
            components.push((
                Arc::as_ptr(vectors) as usize,
                identifier_vector_bytes(&*vectors.read().await),
                "identifier_vectors",
            ));
        }
        components.push((
            Arc::as_ptr(&owner.identifier_vector_overlay) as usize,
            identifier_vector_bytes(&*owner.identifier_vector_overlay.read().await),
            "identifier_overlay",
        ));
        // One statement per lock so no guard is held while the next is awaited.
        let identifier_index = owner.identifier_index.read().await.clone();
        let search_index = owner.search_index_cache.read().await.clone();
        let project_cache = owner.project_cache.read().await.clone();
        let lexical = owner.lexical_search_cache.read().await.clone();
        Self {
            components,
            identifier_index,
            search_index,
            project_cache,
            lexical,
        }
    }

    fn measure(self) -> Vec<ResidentComponent> {
        let mut components = self.components;
        if let Some(index) = &self.identifier_index {
            components.push((
                Arc::as_ptr(index) as usize,
                // The vectors themselves are charged to the maps they were looked up in.
                index.docs.len() * (128 + std::mem::size_of::<Arc<[f32]>>()),
                "identifier_index",
            ));
        }
        if let Some(index) = &self.search_index {
            components.push((
                Arc::as_ptr(index) as usize,
                index.own_resident_bytes(),
                "semantic_index",
            ));
            // Forks share the store, and keep an old one alive after the primary replaces it.
            if let Some(store) = index.index.vector_store() {
                components.push((
                    Arc::as_ptr(store) as usize,
                    store.estimated_resident_bytes(),
                    "vector_store",
                ));
            }
        }
        // Shared bases are keyed by the base's own pointer, so a base counts once
        // however many refs layer over it.
        // Keyword entries keep the file contents they were built from alive, which
        // can be older caches than the current one.
        let mut contents = |cache: &ProjectCache| {
            let files = &cache.file_content;
            components.push((
                Arc::as_ptr(files.own()) as usize,
                files.own_resident_bytes(),
                "file_contents",
            ));
            if let Some(base) = files.base() {
                components.push((
                    Arc::as_ptr(base) as usize,
                    crate::core::walker::content_map_bytes(base),
                    "file_contents",
                ));
            }
        };
        if let Some(cache) = &self.project_cache {
            contents(cache);
        }
        if let Some(cache) = &self.lexical {
            contents(&cache.project_cache);
            if let Some(base) = &cache.base {
                contents(&base.cached.project_cache);
            }
        }
        if let Some(cache) = &self.lexical {
            components.push((
                Arc::as_ptr(cache) as usize,
                cache.own_resident_bytes(),
                "keyword_index",
            ));
            if let Some(base) = &cache.base {
                components.push((
                    Arc::as_ptr(&base.cached) as usize,
                    base.cached.own_resident_bytes(),
                    "keyword_index",
                ));
            }
        }
        components.retain(|(_, bytes, _)| *bytes > 0);
        components
    }
}

/// Identifier vectors persisted under `root`, or none. Read in place from the
/// mapped archive, so loading holds no copy of the file. Keys are identifier
/// texts, not paths: the path hygiene sweep would drop any text that mentions
/// a dotted path segment and force it to be embedded again on every start.
fn load_identifier_vectors(root: &std::path::Path, name: &str) -> IdentifierVectors {
    let started = Instant::now();
    let mut vectors = IdentifierVectors::new();
    if let Err(error) = rkyv_store::visit_cache_entries(root, name, |key, vector| {
        vectors.insert(key.to_owned(), Arc::from(vector));
    }) {
        tracing::warn!(%error, cache = name, "Identifier vector cache unreadable");
        return IdentifierVectors::new();
    }
    tracing::info!(
        phase = "identifier_vectors_load",
        elapsed_ms = started.elapsed().as_millis(),
        vectors = vectors.len(),
        "cold-start phase"
    );
    vectors
}

/// The on-disk form of identifier vectors; the hash of an identifier is the
/// hash of its text.
fn identifier_cache_data(vectors: &IdentifierVectors) -> Option<rkyv_store::CacheData> {
    let dims = vectors
        .values()
        .map(|vector| vector.len())
        .find(|&len| len > 0)?;
    let mut data = rkyv_store::CacheData {
        dims: dims as u32,
        keys: Vec::with_capacity(vectors.len()),
        hashes: Vec::with_capacity(vectors.len()),
        vectors: Vec::with_capacity(vectors.len() * dims),
    };
    for (key, vector) in vectors.iter().filter(|(_, vector)| vector.len() == dims) {
        data.keys.push(key.clone());
        data.hashes.push(crate::core::parser::hash_content(key));
        data.vectors.extend_from_slice(vector);
    }
    Some(data)
}

/// O(1) estimate: vectors of one map share a width.
fn identifier_vector_bytes(vectors: &IdentifierVectors) -> usize {
    vectors.values().next().map_or(0, |vector| {
        vectors.len()
            * (std::mem::size_of::<(String, Arc<[f32]>)>()
                + 64
                + 2 * std::mem::size_of::<usize>()
                + std::mem::size_of_val::<[f32]>(vector))
    })
}

/// O(1) estimate: entries of one map share a vector width.
fn cache_entry_bytes(cache: &HashMap<String, CacheEntry>) -> usize {
    cache.values().next().map_or(0, |entry| {
        cache.len()
            * (std::mem::size_of::<(String, CacheEntry)>()
                + 64
                + entry.hash.capacity()
                + entry.vector.capacity() * std::mem::size_of::<f32>())
    })
}

use crate::tools::lexical_search::{lexical_term_counts, lexical_updates};

/// Documents parsed and tokenized together while a keyword index is built.
const LEXICAL_BUILD_BATCH: usize = 16;

/// Keyword index over `paths` of `files`, with the path of each document.
/// Parsing and tokenizing run in parallel a batch at a time.
fn build_lexical_index<'a>(
    paths: impl Iterator<Item = &'a str>,
    files: &crate::core::walker::FileContents,
) -> (crate::tools::lexical_search::LexicalIndex, Vec<String>) {
    use rayon::prelude::*;
    let built = build_lexical_index_batched(paths, |batch| {
        STRUCTURAL_POOL.install(|| {
            batch
                .par_iter()
                .map(|&path| lexical_term_counts(path, files))
                .collect()
        })
    });
    // Parsing on the pool threads left their allocator arenas holding the
    // freed parse state; hand it back rather than keep it resident.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
    built
}

/// Builds with `count` giving the term counts of each batch, so the counts
/// held at once stay small next to the index being built.
fn build_lexical_index_batched<'a>(
    paths: impl Iterator<Item = &'a str>,
    count: impl Fn(&[&'a str]) -> Vec<crate::tools::lexical_search::DocumentTermCounts>,
) -> (crate::tools::lexical_search::LexicalIndex, Vec<String>) {
    let started = Instant::now();
    let paths: Vec<&str> = paths.collect();
    let mut index = crate::tools::lexical_search::LexicalIndex::with_capacity(paths.len());
    for batch in paths.chunks(LEXICAL_BUILD_BATCH) {
        for (&path, counts) in batch.iter().zip(count(batch)) {
            index.push_counted(path, counts);
        }
    }
    index.finish_build();
    tracing::info!(
        phase = "lexical_build",
        elapsed_ms = started.elapsed().as_millis(),
        documents = paths.len(),
        "cold-start phase"
    );
    (index, paths.into_iter().map(str::to_owned).collect())
}

/// A cache layered over a base stays valid only while that base is still the
/// parent's current cache.
fn base_is_current(cache: &ProjectCache, base: &Option<Arc<ProjectCache>>) -> bool {
    match (cache.file_content.base(), base) {
        (None, _) => true,
        (Some(ours), Some(parent)) => Arc::ptr_eq(ours, parent.file_content.own()),
        (Some(_), None) => false,
    }
}

/// Walks `root` and reads its files. Given the parent's flat `base`, keeps only
/// the files that differ from it, layered over it; a file clean here with the
/// blob the base recorded in `clean_blobs`, and of the same size, is not read.
/// Past `FULL_REBUILD_CHANGE_FRACTION` of the base, builds flat. With
/// `record_clean_blobs` and no base, records `clean_blobs` for worktrees.
fn load_project_cache(
    root: &std::path::Path,
    config: &Config,
    base: Option<Arc<ProjectCache>>,
    record_clean_blobs: bool,
) -> ProjectCache {
    use crate::core::git_worktree::{clean_blobs, common_blobs};
    use crate::core::walker::{ContentMap, FileContents};
    use rayon::prelude::*;

    // git runs here, not on a structural worker.
    let base = base.map(|base| {
        let unchanged: std::collections::HashSet<String> =
            match (&base.clean_blobs, clean_blobs(root)) {
                (Some(snapshot), Some(current)) => {
                    common_blobs(snapshot, &current).into_keys().collect()
                }
                _ => Default::default(),
            };
        (Arc::clone(base.file_content.own()), unchanged)
    });
    let git_started = Instant::now();
    let clean_before = (base.is_none() && record_clean_blobs)
        .then(|| clean_blobs(root))
        .flatten();
    let git_before_ms = git_started.elapsed().as_millis();

    let started = Instant::now();
    let mut walk_ms = 0;
    let (file_entries, file_content) = STRUCTURAL_POOL.install(|| {
        let file_entries = walk_with_config(root, config);
        walk_ms = started.elapsed().as_millis();
        let read = |path: &String| std::fs::read_to_string(root.join(path)).ok().map(Arc::new);
        let layered = base.map(|(base, unchanged)| {
            // Same clean blob is not same bytes under a smudge filter or eol
            // conversion that differs between the trees; a size check catches most.
            let same_as_base = |path: &String| {
                unchanged.contains(path)
                    && base.get(path).is_some_and(|content| {
                        std::fs::metadata(root.join(path))
                            .is_ok_and(|meta| meta.len() == content.len() as u64)
                    })
            };
            let changed: Vec<(String, Option<Arc<String>>)> = file_entries
                .par_iter()
                .filter(|entry| !entry.is_directory && !same_as_base(&entry.relative_path))
                .filter_map(|entry| {
                    let content = read(&entry.relative_path);
                    (content.as_deref() != base.get(&entry.relative_path).map(|c| &**c))
                        .then(|| (entry.relative_path.clone(), content))
                })
                .collect();
            let present: std::collections::HashSet<&str> = file_entries
                .iter()
                .filter(|entry| !entry.is_directory)
                .map(|entry| entry.relative_path.as_str())
                .collect();
            let mut masked: std::collections::HashSet<String> = base
                .keys()
                .filter(|path| !present.contains(path.as_str()))
                .cloned()
                .collect();
            let mut own = ContentMap::new();
            for (path, content) in changed {
                match content {
                    Some(content) => {
                        own.insert(path, content);
                    }
                    None => {
                        masked.insert(path);
                    }
                }
            }
            let delta = own.len()
                + masked
                    .iter()
                    .filter(|path| !own.contains_key(*path))
                    .count();
            let within = delta as f64
                <= base.len() as f64 * crate::tools::semantic_search::FULL_REBUILD_CHANGE_FRACTION;
            let files = FileContents::layered(own, base, masked);
            if within {
                files
            } else {
                // Flat, sharing the base's content where it matched instead of reading it again.
                files
                    .iter()
                    .map(|(path, content)| (path.clone(), Arc::clone(content)))
                    .collect()
            }
        });
        let file_content = layered.unwrap_or_else(|| {
            file_entries
                .par_iter()
                .filter(|entry| !entry.is_directory)
                .filter_map(|entry| {
                    Some((entry.relative_path.clone(), read(&entry.relative_path)?))
                })
                .collect::<ContentMap>()
                .into()
        });
        (file_entries, file_content)
    });
    let read_ms = started.elapsed().as_millis() - walk_ms;
    let git_started = Instant::now();
    let clean_blobs = clean_before
        .and_then(|before| Some(common_blobs(&before, &clean_blobs(root)?)))
        .map(Arc::new);
    tracing::info!(
        phase = "project_cache",
        git_before_ms,
        walk_ms,
        read_ms,
        git_after_ms = git_started.elapsed().as_millis(),
        files = file_content.len(),
        "cold-start phase"
    );
    ProjectCache {
        file_entries,
        file_content,
        clean_blobs,
        last_refresh: Instant::now(),
    }
}

async fn clear_ref_heavy_caches(owner: &crate::ref_index::RefIndex, id_cache_name: &str) {
    // Background rebuilds and fills would refill what is cleared here.
    owner.cancel_background_tasks();
    owner
        .cache_generation
        .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    *owner.semantic_fill.lock().await = Default::default();
    owner.embedding_cache.write().await.clear();
    *owner.identifier_index.write().await = None;
    *owner.identifier_source.write().await = None;
    let save_lock = owner.identifier_save_lock.lock().await;
    let resident = match owner.identifier_vectors.get() {
        Some(vectors) if Arc::strong_count(vectors) == 1 => {
            std::mem::take(&mut *vectors.write().await)
        }
        _ => IdentifierVectors::default(),
    };
    let mut overlay = owner.identifier_vector_overlay.write().await;
    let mut pending = std::mem::take(&mut *overlay);
    for (key, vector) in std::mem::take(&mut *owner.identifier_unsaved.lock().unwrap()) {
        pending.entry(key).or_insert(vector);
    }
    if let Some(data) = identifier_cache_data(&pending) {
        let root = owner.root_dir.clone();
        let name = id_cache_name.to_string();
        let saved = tokio::task::spawn_blocking(move || {
            rkyv_store::save_cache_rebuilding(&root, &name, &data, || {
                let mut all = resident;
                all.extend(pending);
                identifier_cache_data(&all)
            })
        })
        .await;
        if let Ok(Err(error)) = saved {
            tracing::warn!(%error, "Identifier overlay flush failed");
        }
    }
    owner
        .identifier_overlay_loaded
        .store(false, std::sync::atomic::Ordering::Release);
    drop(overlay);
    drop(save_lock);
    *owner.search_index_cache.write().await = None;
    *owner.lexical_search_cache.write().await = None;
    *owner.project_cache.write().await = None;
}

/// The MCP server exposing context+ tools.
///
/// Each instance is associated with exactly one MCP transport. In daemon mode
/// the per-connection `serve_connection` function creates a session-scoped
/// copy via [`ContextPlusServer::with_session`] so that every tool call
/// dispatched on that connection resolves to the registered worktree's
/// [`crate::ref_index::RefIndex`] rather than the singleton default ref.
///
/// In stdio mode (`session_ref_id = None`) the server behaves identically to
/// the pre-U9 code path: all tool calls route through
/// `SharedState::default_ref`.
#[derive(Clone)]
pub struct ContextPlusServer {
    pub state: Arc<SharedState>,
    /// Set per-connection by `daemon::serve_connection` after
    /// `register_session` completes. `None` means "stdio mode or no
    /// handshake" — all tool calls fall back to `default_ref`.
    pub session_ref_id: Option<crate::ref_index::RefId>,
    session_config_warning: Option<Arc<SessionConfigWarning>>,
}

struct SessionConfigWarning {
    text: String,
    shown: std::sync::atomic::AtomicBool,
}

impl ContextPlusServer {
    /// Return a clone of this server with `session_ref_id` set to `ref_id`.
    ///
    /// The `Arc<SharedState>` is shared — only the routing key changes.
    /// Callers should prefer this over mutating `session_ref_id` directly so
    /// the original server (held by the daemon accept loop) remains unchanged.
    pub fn with_session(&self, ref_id: crate::ref_index::RefId) -> Self {
        Self {
            state: Arc::clone(&self.state),
            session_ref_id: Some(ref_id),
            session_config_warning: self.session_config_warning.clone(),
        }
    }

    pub(crate) fn with_session_config_warning(
        &self,
        ref_id: crate::ref_index::RefId,
        warning: Option<String>,
    ) -> Self {
        Self {
            state: Arc::clone(&self.state),
            session_ref_id: Some(ref_id),
            session_config_warning: warning.map(|text| {
                Arc::new(SessionConfigWarning {
                    text,
                    shown: std::sync::atomic::AtomicBool::new(false),
                })
            }),
        }
    }

    fn prepend_session_config_warning(&self, mut result: CallToolResult) -> CallToolResult {
        let Some(warning) = &self.session_config_warning else {
            return result;
        };
        if warning
            .shown
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            return result;
        }

        if let Some(first) = result.content.first_mut()
            && let RawContent::Text(text) = &mut first.raw
        {
            text.text = format!("{}\n{}", warning.text, text.text);
        } else {
            result.content.insert(0, Content::text(&warning.text));
        }
        result
    }

    /// Resolve the per-ref state for the current session.
    ///
    /// Resolution order:
    /// 1. `session_ref_id = Some(id)` and `id` is in the registry → that ref.
    /// 2. `session_ref_id = Some(id)` but `id` not found (registry tampered) →
    ///    fall back to `default_ref` and emit a debug log.
    /// 3. `session_ref_id = None` (stdio mode or no handshake) → `default_ref`.
    ///
    /// The `default_ref` always exists for the lifetime of the daemon, so an
    /// evicted session ref safely falls back to that permanent snapshot.
    ///
    /// **Single-ref behaviour is unchanged:** when `session_ref_id` is `None`
    /// (stdio) or equals `default_ref_id` (daemon, one worktree), this returns
    /// exactly the same `RefIndex` as the `SharedState` backward-compat shims.
    /// No observable difference for existing callers until U12 introduces
    /// per-ref file walkers.
    pub async fn current_ref(&self) -> Arc<crate::ref_index::RefIndex> {
        if let Some(id) = self.session_ref_id
            && let Some(ref_index) = self.state.ref_index(id).await
        {
            return ref_index;
        }
        Arc::clone(&self.state.default_ref)
    }
}

/// Sanitize a model name for use in cache filenames.
/// Replaces `/`, `:`, and other non-filename chars with `-`.
pub fn sanitize_model_name(model: &str) -> String {
    model
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Build a model- and document-settings-qualified cache name.
pub fn cache_name(base: &str, config: &Config) -> String {
    format!(
        "{}-{}",
        base,
        sanitize_model_name(&config.document_cache_identity())
    )
}

static TS_DECL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(\s*)(export\s+)?(default\s+)?(declare\s+)?(async\s+)?(abstract\s+)?(function\*?|class|interface|type|enum|const|let|var|namespace)\s+([A-Za-z_$][\w$]*)",
    )
    .expect("valid TypeScript declaration regex")
});
static TS_METHOD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(\s+)(async\s+)?((static|private|public|protected|readonly|get|set|override)\s+)*([A-Za-z_$][\w$]*)\??\s*(<[^>]*>)?\s*\(",
    )
    .expect("valid TypeScript method regex")
});
static TS_PROPFN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(\s+)(readonly\s+)?([A-Za-z_$][\w$]*)\??\s*:\s*(async\s*)?\(")
        .expect("valid TypeScript function-property regex")
});
static GO_DECL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(func|type)\s+").expect("valid Go declaration regex"));
static GO_IFACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s+([A-Z]\w*)\(").expect("valid Go interface regex"));
static SQL_DECL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\s*CREATE\s+(OR\s+REPLACE\s+)?(UNIQUE\s+)?(FUNCTION|TABLE|VIEW|MATERIALIZED VIEW|POLICY|TRIGGER|INDEX|TYPE)\b",
    )
    .expect("valid SQL declaration regex")
});

fn expanded_indent_width(indent: &str) -> usize {
    indent.chars().fold(0, |width, ch| {
        if ch == '\t' {
            width + (4 - width % 4)
        } else {
            width + 1
        }
    })
}

fn embedding_outline(relative_path: &str, content: &str) -> String {
    const TS_KEYWORDS: [&str; 20] = [
        "if", "for", "while", "switch", "return", "catch", "function", "await", "new", "throw",
        "else", "typeof", "super", "import", "export", "case", "do", "try", "void", "delete",
    ];

    let ext = relative_path
        .rsplit_once('.')
        .map_or(relative_path, |(_, ext)| ext);
    content
        .lines()
        .filter_map(|line| {
            let keep = match ext {
                "ts" | "tsx" => {
                    if let Some(captures) = TS_DECL.captures(line) {
                        if line.trim_end().ends_with(',') {
                            false
                        } else {
                            let indent = captures.get(1).map_or("", |m| m.as_str());
                            let kind = captures.get(7).map_or("", |m| m.as_str());
                            indent.is_empty()
                                || !matches!(kind, "const" | "let" | "var")
                                || line.contains("=>")
                                || line.contains("function")
                        }
                    } else {
                        let method = TS_METHOD
                            .captures(line)
                            .map(|captures| (captures, 5))
                            .or_else(|| TS_PROPFN.captures(line).map(|captures| (captures, 3)));
                        method.is_some_and(|(captures, name_group)| {
                            let indent = captures.get(1).map_or("", |m| m.as_str());
                            let name = captures.get(name_group).map_or("", |m| m.as_str());
                            !TS_KEYWORDS.contains(&name)
                                && expanded_indent_width(indent) <= 8
                                && !line.trim_end().ends_with(");")
                        })
                    }
                }
                "go" => GO_DECL.is_match(line) || GO_IFACE.is_match(line),
                "sql" => SQL_DECL.is_match(line),
                _ => false,
            };
            keep.then(|| line.trim().chars().take(160).collect::<String>())
        })
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .take(1500)
        .collect()
}

/// Build the text embedded for a code file.
pub fn build_embedding_document(
    relative_path: &str,
    content: &str,
    shape: EmbedDocShape,
) -> String {
    let head = crate::core::parser::truncate_to_char_boundary(content, 500);
    let header = crate::core::parser::extract_header(head);

    match shape {
        EmbedDocShape::Head => format!("{} {} {}", header, relative_path, head),
        EmbedDocShape::Outline => {
            let outline = embedding_outline(relative_path, content);
            format!("{}\n{}\n{}\n{}", relative_path, header, outline, head)
        }
    }
}

impl ContextPlusServer {
    pub fn new(root_dir: PathBuf, config: Config) -> Self {
        // Global Ollama embed semaphore (U16+U17). Capacity is sourced from
        // `config.ollama_max_concurrent`, which Config::from_env clamps into
        // [1, 64]. We wire the same semaphore into the OllamaClient so that
        // every outbound embed (warmup, tracker, on-demand) shares one budget.
        let ollama_semaphore = Arc::new(Semaphore::new(config.ollama_max_concurrent.max(1)));
        let ollama = OllamaClient::new_with_root(&config, Some(root_dir.clone()))
            .with_semaphore(Arc::clone(&ollama_semaphore));

        let embed_cache_name = cache_name("embeddings", &config);

        // Load embedding cache from disk if available (cross-restart persistence)
        let initial_cache = match rkyv_store::mmap_vector_store(&root_dir, &embed_cache_name) {
            Ok(Some(store)) => {
                let cache_map = store.to_cache();
                tracing::info!(
                    entries = cache_map.len(),
                    "Loaded embedding cache from disk"
                );
                cache_map
            }
            Ok(None) => {
                tracing::debug!("No embedding cache on disk, starting fresh");
                HashMap::new()
            }
            Err(e) => {
                tracing::warn!("Failed to load embedding cache from disk: {e}");
                HashMap::new()
            }
        };

        // Canonicalize once at construction so resolve_root() can skip per-request syscalls.
        let canonical_root = root_dir.canonicalize().unwrap_or_else(|_| root_dir.clone());

        // Build the default ref with the disk-loaded embedding cache (U10).
        // All other per-ref caches start empty.
        let default_ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_root);
        let default_ref = Arc::new(crate::ref_index::RefIndex::with_preloaded_cache(
            root_dir.clone(),
            canonical_root.clone(),
            initial_cache,
        ));

        // Clone the per-ref Arcs so SharedState can expose them as backward-compat
        // shims.  Both SharedState and the RefIndex entry point at the SAME
        // underlying RwLock / Mutex — writes through one are visible through the
        // other.  U11 will migrate call sites to go through RefIndex directly.
        let embedding_cache = Arc::clone(&default_ref.embedding_cache);
        let identifier_index = Arc::clone(&default_ref.identifier_index);
        let search_index_cache = Arc::clone(&default_ref.search_index_cache);
        let cache_generation = Arc::clone(&default_ref.cache_generation);
        let tracker_handle = Arc::clone(&default_ref.tracker_handle);
        let project_cache = Arc::clone(&default_ref.project_cache);

        let refs =
            crate::ref_index::new_registry_with_default(default_ref_id, Arc::clone(&default_ref));

        let state = Arc::new(SharedState {
            config,
            canonical_root,
            root_dir,
            ollama,
            project_cache,
            embedding_cache,
            identifier_index,
            search_index_cache,
            cache_generation,
            instructions_cache: OnceCell::new(),
            tracker_handle,
            idle_monitor: RwLock::new(None),
            draining: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            inflight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            refs,
            default_ref,
            default_ref_id,
            ollama_semaphore,
            warmup_in_flight: Arc::new(tokio::sync::Mutex::new(std::collections::HashSet::new())),
            access_clock: std::sync::atomic::AtomicU64::new(0),
            ref_access: std::sync::Mutex::new(HashMap::from([(
                default_ref_id,
                (0, Instant::now()),
            )])),
            budget_enforcement_running: std::sync::atomic::AtomicBool::new(false),
            last_budget_enforcement: std::sync::Mutex::new(None),
            last_budget_trim: std::sync::Mutex::new(None),
            last_free_memory_check: std::sync::Mutex::new(None),
            budget_warned: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            measured_resident_override: std::sync::Mutex::new(None),
            #[cfg(test)]
            free_memory_checks: std::sync::atomic::AtomicUsize::new(0),
        });
        Self {
            state,
            session_ref_id: None,
            session_config_warning: None,
        }
    }

    /// Build a refresh callback for the embedding tracker.
    pub async fn build_tracker_callback(&self) -> RefreshCallback {
        let root = self.current_ref().await.root_dir.clone();
        self.build_tracker_callback_for_root(root)
    }

    fn build_tracker_callback_for_root(&self, root: PathBuf) -> RefreshCallback {
        let server = self.clone();
        Arc::new(move |_root, files| {
            let srv = server.clone();
            let root = root.clone();
            let changed_files: Vec<PathBuf> = files.iter().map(|f| root.join(f)).collect();
            tokio::spawn(async move {
                let ref_index = srv.current_ref().await;
                tracing::debug!(
                    ref_id = %ref_index.cas_ref_id_hex,
                    paths = ?files,
                    "Embedding tracker refresh batch started"
                );
                let outcome = srv.incremental_reembed_detailed(&changed_files).await;
                let updated = outcome.updated;
                let skipped = outcome.skipped;
                tracing::debug!(
                    updated,
                    skipped,
                    "Incremental re-embedding for {} changed files",
                    changed_files.len()
                );
                // Watchers can emit batches for metadata/read activity. Only
                // invalidate when inspecting the source found a content change.
                if outcome.content_changed {
                    let new_gen = ref_index
                        .cache_generation
                        .fetch_add(1, std::sync::atomic::Ordering::Release)
                        + 1;
                    srv.refresh_search_paths(&files, new_gen).await;
                    tracing::debug!(
                        generation = new_gen,
                        updated,
                        skipped,
                        paths = ?files,
                        reason = "tracker batch contained changed content",
                        "cache_generation bumped after tracker event"
                    );
                } else {
                    tracing::debug!(
                        generation = ref_index
                            .cache_generation
                            .load(std::sync::atomic::Ordering::Acquire),
                        skipped,
                        paths = ?files,
                        reason = "all tracker paths were content-identical",
                        "cache_generation unchanged after tracker event"
                    );
                }
                (updated, skipped)
            })
        })
    }

    async fn refresh_search_paths(&self, paths: &[String], generation: u64) {
        use crate::tools::semantic_search::{
            CachedSearchIndex, SearchDocument, SymbolSearchEntry, extract_plain_text_header,
            is_text_index_candidate, semantic_embedding_content,
        };
        let owner = self.current_ref().await;
        if owner.search_index_cache.read().await.is_none() {
            return;
        }
        let mut docs = Vec::new();
        let mut vectors = Vec::new();
        let mut deleted = Vec::new();
        let eligible: std::collections::HashSet<_> =
            walk_with_config(&owner.root_dir, &self.state.config)
                .into_iter()
                .filter(|entry| !entry.is_directory)
                .map(|entry| entry.relative_path)
                .collect();
        for path in paths {
            if !eligible.contains(path) {
                deleted.push(path.clone());
                continue;
            }
            let Ok(content) = tokio::fs::read_to_string(owner.root_dir.join(path)).await else {
                deleted.push(path.clone());
                continue;
            };
            if content.len() > self.state.config.max_embed_file_size {
                deleted.push(path.clone());
                continue;
            }
            let text = semantic_embedding_content(path, &content);
            let (header, symbols, entries) = if is_text_index_candidate(path) {
                (extract_plain_text_header(&text), Vec::new(), Vec::new())
            } else {
                let symbols =
                    parse_with_tree_sitter(&content, path.rsplit('.').next().unwrap_or(""))
                        .unwrap_or_default();
                let names = symbols.iter().map(|s| s.name.clone()).collect();
                let entries = symbols
                    .into_iter()
                    .map(|s| SymbolSearchEntry {
                        name: s.name,
                        kind: Some(s.kind),
                        line: s.line,
                        end_line: Some(s.end_line),
                        signature: s.signature,
                    })
                    .collect();
                (
                    crate::core::parser::extract_header(&content),
                    names,
                    entries,
                )
            };
            let vector = owner
                .embedding_cache
                .read()
                .await
                .get(path)
                .filter(|entry| entry.hash == crate::core::parser::hash_content(&content))
                .map(|entry| entry.vector.clone());
            docs.push(SearchDocument::new(
                path.clone(),
                header,
                symbols,
                entries,
                text,
            ));
            docs.last_mut().unwrap().source_hash = crate::core::embeddings::content_hash(&content);
            vectors.push(vector);
        }
        let mut guard = owner.search_index_cache.write().await;
        if let Some(entry) = guard.as_mut() {
            CachedSearchIndex::refresh_ref_paths(
                entry,
                &owner.canonical_root,
                docs,
                vectors,
                &deleted,
                generation,
            );
        }
    }

    /// Start the embedding tracker for a specific ref if not already running
    /// and mode is not Off.
    ///
    /// Routes through `with_session(ref_id)` so the tracker's refresh callback
    /// and `incremental_reembed` invocations land on that ref's caches
    /// (`embedding_cache`, `search_index_cache`, `cache_generation`) rather
    /// than the default ref's.
    pub async fn ensure_tracker_started_for(&self, ref_id: crate::ref_index::RefId) {
        self.with_session(ref_id).ensure_tracker_started().await;
    }

    /// Start the embedding tracker if not already running and mode is not Off.
    pub async fn ensure_tracker_started(&self) {
        if self.state.config.embed_tracker_mode == TrackerMode::Off {
            return;
        }
        let ref_index = self.current_ref().await;
        let mut guard = ref_index.tracker_handle.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("tracker_handle mutex was poisoned; recovering inner value");
            poisoned.into_inner()
        });
        if guard.is_some() {
            return;
        }
        let tracker_config = EmbeddingTrackerConfig {
            debounce_ms: self.state.config.embed_tracker_debounce_ms,
            max_files_per_tick: self.state.config.embed_tracker_max_files,
            ignore_dirs: self.state.config.ignore_dirs.clone(),
        };
        let callback = self.build_tracker_callback_for_root(ref_index.root_dir.clone());
        match crate::core::embedding_tracker::start_tracker(
            ref_index.root_dir.clone(),
            tracker_config,
            callback,
        ) {
            Ok(handle) => {
                tracing::info!(
                    mode = %self.state.config.embed_tracker_mode,
                    "Embedding tracker started"
                );
                ref_index
                    .cache_generation
                    .fetch_add(1, std::sync::atomic::Ordering::Release);
                *guard = Some(handle);
            }
            Err(e) => {
                tracing::warn!("Failed to start embedding tracker: {e}");
            }
        }
    }

    /// Cancel all in-flight embedding requests (used during shutdown).
    pub fn cancel_all_embeddings(&self) {
        self.state.ollama.cancel_all_embeddings();
    }

    /// Spawn a background task that runs a trivial semantic search query to
    /// populate the in-memory `SearchIndex` cache and pre-build the HNSW index.
    /// After this task completes the first real user query is served from cache
    /// (warm path, ~1-2 s) instead of the cold path (~30 s).
    ///
    /// The task is fire-and-forget: errors are logged as warnings and never
    /// propagate to the caller.  Server startup is never delayed.
    ///
    /// Only a daemon, shared by every session, also preloads the keyword and
    /// identifier indexes (`preload_snapshots`); a private server keeps to the
    /// indexes its session asks for. Returns the preload task.
    pub fn spawn_warmup_task(
        &self,
        preload_snapshots: bool,
    ) -> Option<tokio::task::JoinHandle<()>> {
        let state = self.state.clone();
        tokio::spawn(async move {
            warmup_semantic_search_cache(&state).await;
        });
        if let Some(primary) = self.state.default_ref()
            && snapshots::enabled(&self.state.config, &primary)
        {
            tokio::task::spawn_blocking(move || {
                crate::cache::snapshot::remove_stale_temp_files(
                    &primary.root_dir,
                    crate::cache::snapshot::STALE_TEMP_AGE,
                )
            });
        }
        if preload_snapshots {
            self.spawn_snapshot_preload()
        } else {
            None
        }
    }

    /// Loads the keyword and identifier indexes of the primary checkout whose
    /// snapshots exist, alongside the semantic warmup, so the first query of
    /// each mode finds its index ready. A query arriving first waits only on
    /// the index it needs.
    fn spawn_snapshot_preload(&self) -> Option<tokio::task::JoinHandle<()>> {
        let primary = self.state.default_ref()?;
        if !snapshots::enabled(&self.state.config, &primary) {
            return None;
        }
        let keywords =
            crate::cache::snapshot::snapshot_path(&primary.root_dir, snapshots::KEYWORDS).exists();
        let identifiers =
            crate::cache::snapshot::snapshot_path(&primary.root_dir, snapshots::IDENTIFIERS)
                .exists();
        if !keywords && !identifiers {
            return None;
        }
        let server = self.clone();
        Some(tokio::spawn(async move {
            let Ok(cache) = server.ensure_project_cache_for(&primary).await else {
                return;
            };
            let keyword_index = async {
                if keywords
                    && let Err(error) = server.ensure_lexical_index_for(&primary, &cache).await
                {
                    tracing::warn!(%error, "keyword index preload failed");
                }
            };
            let identifier_index = async {
                if identifiers && let Err(error) = server.ensure_identifier_index(&cache).await {
                    tracing::warn!(%error, "identifier index preload failed");
                }
            };
            tokio::join!(keyword_index, identifier_index);
        }))
    }

    // -----------------------------------------------------------------------
    // U18: per-ref warmup on attach
    // -----------------------------------------------------------------------

    /// Trigger warmup for `ref_id` according to the configured
    /// [`RefWarmupMode`].
    ///
    /// The call is **idempotent**: if a warmup task for the same `ref_id` is
    /// already running, this is a no-op (logged at `DEBUG`).  Likewise, if the
    /// ref's `project_cache` is already populated and fresh, the spawned task
    /// exits early without re-walking.
    ///
    /// Warmup failure is non-fatal — errors are logged as `WARN` and never
    /// propagate to the connection handler.
    pub fn spawn_ref_warmup(&self, ref_id: crate::ref_index::RefId) {
        match self.state.config.ref_warmup_mode {
            RefWarmupMode::Off => {
                tracing::info!(
                    ref_id = ref_id.0,
                    mode = "off",
                    "ref_warmup mode=off — skipping"
                );
            }
            RefWarmupMode::Shallow => {
                self.spawn_shallow_warmup_task(ref_id);
            }
            RefWarmupMode::Full => {
                self.spawn_full_warmup_task(ref_id);
            }
        }
    }

    /// Spawn the shallow warmup background task for `ref_id`.
    ///
    /// Shallow warmup (U20 contract):
    /// 1. Walks the ref's `root_dir` via `walk_with_config`.
    /// 2. Reads all non-directory files into `ref.project_cache`.
    /// 3. Runs tree-sitter parsing and populates `identifier_index.docs`.
    /// 4. Calls `import_baseline_for_ref` — for every chunk, looks up the
    ///    BLAKE3 hash in the CAS parent chain.  Hits are loaded into
    ///    `embedding_cache` and used to build `search_index_cache` (HNSW).
    /// 5. Does NOT call `OllamaClient::embed*`.  Zero outbound Ollama calls.
    ///    Diff chunks (misses in step 4) remain unembedded; they fill lazily
    ///    on the first real tool call.
    ///
    /// Idempotent: skips if `project_cache` is already populated and fresh.
    fn spawn_shallow_warmup_task(&self, ref_id: crate::ref_index::RefId) {
        let state = Arc::clone(&self.state);
        let server = self.clone();
        tokio::spawn(async move {
            // --- Idempotency guard ---
            {
                let mut inflight = state.warmup_in_flight.lock().await;
                if inflight.contains(&ref_id) {
                    tracing::debug!(
                        ref_id = ref_id.0,
                        "ref_warmup shallow: already in-flight, skipping"
                    );
                    return;
                }
                inflight.insert(ref_id);
            }

            // Ensure we release the guard slot even on early return or panic.
            let _guard = WarmupGuard {
                state: Arc::clone(&state),
                ref_id,
            };

            let ref_index = match state.ref_index(ref_id).await {
                Some(r) => r,
                None => {
                    tracing::warn!(ref_id = ref_id.0, "ref_warmup shallow: ref not found");
                    return;
                }
            };

            // --- Skip if already warm ---
            {
                let guard = ref_index.project_cache.read().await;
                if let Some(ref cache) = *guard {
                    let ttl = state.config.cache_ttl_secs;
                    if ContextPlusServer::tracker_is_running(&ref_index)
                        || cache.last_refresh.elapsed().as_secs() < ttl
                    {
                        tracing::debug!(
                            ref_id = ref_id.0,
                            "ref_warmup shallow: project_cache already populated, skipping"
                        );
                        return;
                    }
                }
            }

            let root = ref_index.root_dir.clone();
            let config = state.config.clone();
            let t0 = std::time::Instant::now();
            tracing::info!(ref_id = ref_id.0, root = %root.display(), "ref_warmup shallow: starting walk");

            // --- Walk + read files (blocking I/O) ---
            let build_generation = ref_index
                .cache_generation
                .load(std::sync::atomic::Ordering::Acquire);
            let base = server.project_cache_base(&ref_index).await;
            let record = ref_index.parent_ref_id.is_none();
            let new_cache = tokio::task::spawn_blocking(move || {
                load_project_cache(&root, &config, base, record)
            })
            .await;

            let new_cache = match new_cache {
                Ok(c) => Arc::new(c),
                Err(e) => {
                    tracing::warn!(ref_id = ref_id.0, error = %e, "ref_warmup shallow: spawn_blocking failed");
                    return;
                }
            };

            // --- Populate project_cache ---
            let cache_installed = {
                let mut guard = ref_index.project_cache.write().await;
                // First-writer-wins: only update if still unpopulated (or stale).
                let needs_update = ref_index
                    .cache_generation
                    .load(std::sync::atomic::Ordering::Acquire)
                    == build_generation
                    && match &*guard {
                        None => true,
                        Some(c) => {
                            !ContextPlusServer::tracker_is_running(&ref_index)
                                && c.last_refresh.elapsed().as_secs() >= state.config.cache_ttl_secs
                        }
                    };
                if needs_update {
                    *guard = Some(Arc::clone(&new_cache));
                    tracing::debug!(
                        ref_id = ref_id.0,
                        reason = "shallow ref warmup",
                        "ProjectCache replaced"
                    );
                }
                needs_update
            };
            if !cache_installed {
                tracing::debug!(
                    ref_id = ref_id.0,
                    reason = "another authoritative cache won during shallow warmup",
                    "ref_warmup shallow: discarding build"
                );
                return;
            }

            // --- Parse tree-sitter symbols into identifier_index.docs ---
            // We run the full parse pipeline (flatten_symbols + token_set) but
            // skip the embedding step.  The identifier search handler will rebuild
            // with embeddings on the first real tool call.
            let cache_for_parse = Arc::clone(&new_cache);
            let doc_list: Vec<crate::tools::semantic_identifiers::IdentifierDoc> =
                tokio::task::spawn_blocking(move || {
                    use rayon::prelude::*;
                    cache_for_parse
                        .file_entries
                        .par_iter()
                        .filter(|e| !e.is_directory)
                        .filter_map(|entry| {
                            let content = cache_for_parse.file_content.get(&entry.relative_path)?;
                            crate::tools::semantic_identifiers::identifier_docs_for_file(
                                &entry.relative_path,
                                content,
                            )
                        })
                        .flatten()
                        .collect()
                })
                .await
                .unwrap_or_default();
            let file_count = new_cache
                .file_entries
                .iter()
                .filter(|e| !e.is_directory)
                .count();
            {
                let mut guard = ref_index.identifier_index.write().await;
                let needs_update = ref_index
                    .cache_generation
                    .load(std::sync::atomic::Ordering::Acquire)
                    == build_generation
                    && match &*guard {
                        None => true,
                        Some(idx) => idx.file_count != file_count,
                    };
                if needs_update {
                    *guard = Some(Arc::new(IdentifierIndex {
                        docs: doc_list.into(),
                        // vectors + dims left empty — shallow mode omits
                        // embedding calls.  Full mode (or first real tool call)
                        // will populate these fields.
                        vectors: IdentifierVectorIndex::empty(),
                        dims: 0,
                        file_count,
                        built_at: std::time::Instant::now(),
                    }));
                    tracing::debug!(
                        ref_id = ref_id.0,
                        reason = "shallow ref warmup",
                        "IdentifierIndex replaced"
                    );
                }
            }

            // --- U20: CAS baseline import — zero Ollama calls ---
            // Walk the CAS parent chain for every chunk in the corpus.  Hits are
            // loaded into `embedding_cache` and used to build `search_index_cache`.
            // Misses are left for lazy fill on first tool call (Shallow mode).
            let report = import_baseline_for_ref(&state, ref_id, Arc::clone(&new_cache)).await;
            tracing::info!(
                ref_id = ref_id.0,
                hits = report.hits,
                misses = report.misses.len(),
                elapsed_ms = t0.elapsed().as_millis(),
                files = file_count,
                "ref_warmup shallow: complete (no embed calls)"
            );
        });
    }

    /// Spawn the full warmup background task for `ref_id`.
    ///
    /// Full warmup (U20 contract):
    /// 1. Runs the same walk + parse + CAS baseline-import as Shallow — zero
    ///    code duplication.  The worktree is immediately searchable using the
    ///    primary's embedding corpus after this phase.
    /// 2. Embeds the diff chunks (CAS misses) via Ollama, gated by the
    ///    `ollama_semaphore`.  After this phase the worktree is fully indexed
    ///    including worktree-specific diffs.
    ///
    /// Idempotent: skips if `project_cache` is already warm.
    fn spawn_full_warmup_task(&self, ref_id: crate::ref_index::RefId) {
        let state = Arc::clone(&self.state);
        let server = self.clone();
        tokio::spawn(async move {
            // --- Idempotency guard ---
            {
                let mut inflight = state.warmup_in_flight.lock().await;
                if inflight.contains(&ref_id) {
                    tracing::debug!(
                        ref_id = ref_id.0,
                        "ref_warmup full: already in-flight, skipping"
                    );
                    return;
                }
                inflight.insert(ref_id);
            }

            let _guard = WarmupGuard {
                state: Arc::clone(&state),
                ref_id,
            };

            let ref_index = match state.ref_index(ref_id).await {
                Some(r) => r,
                None => {
                    tracing::warn!(ref_id = ref_id.0, "ref_warmup full: ref not found");
                    return;
                }
            };

            // --- Skip if already warm ---
            {
                let guard = ref_index.project_cache.read().await;
                if let Some(ref cache) = *guard
                    && (ContextPlusServer::tracker_is_running(&ref_index)
                        || cache.last_refresh.elapsed().as_secs() < state.config.cache_ttl_secs)
                {
                    tracing::debug!(
                        ref_id = ref_id.0,
                        "ref_warmup full: project_cache already populated, skipping"
                    );
                    return;
                }
            }

            let t0 = std::time::Instant::now();
            tracing::info!(ref_id = ref_id.0, "ref_warmup full: starting walk");

            // --- Phase 1: walk + read files ---
            let root = ref_index.root_dir.clone();
            let config = state.config.clone();
            let build_generation = ref_index
                .cache_generation
                .load(std::sync::atomic::Ordering::Acquire);
            let base = server.project_cache_base(&ref_index).await;
            let record = ref_index.parent_ref_id.is_none();
            let new_cache = tokio::task::spawn_blocking(move || {
                load_project_cache(&root, &config, base, record)
            })
            .await;

            let new_cache = match new_cache {
                Ok(c) => Arc::new(c),
                Err(e) => {
                    tracing::warn!(ref_id = ref_id.0, error = %e, "ref_warmup full: spawn_blocking failed");
                    return;
                }
            };

            // --- Populate project_cache ---
            let cache_installed = {
                let mut guard = ref_index.project_cache.write().await;
                let needs_update = ref_index
                    .cache_generation
                    .load(std::sync::atomic::Ordering::Acquire)
                    == build_generation
                    && match &*guard {
                        None => true,
                        Some(c) => {
                            !ContextPlusServer::tracker_is_running(&ref_index)
                                && c.last_refresh.elapsed().as_secs() >= state.config.cache_ttl_secs
                        }
                    };
                if needs_update {
                    *guard = Some(Arc::clone(&new_cache));
                    tracing::debug!(
                        ref_id = ref_id.0,
                        reason = "full ref warmup",
                        "ProjectCache replaced"
                    );
                }
                needs_update
            };
            if !cache_installed {
                tracing::debug!(
                    ref_id = ref_id.0,
                    reason = "another authoritative cache won during full warmup",
                    "ref_warmup full: discarding build"
                );
                return;
            }

            // --- Phase 2: tree-sitter parse ---
            let cache_for_parse = Arc::clone(&new_cache);
            let doc_list: Vec<crate::tools::semantic_identifiers::IdentifierDoc> =
                tokio::task::spawn_blocking(move || {
                    use rayon::prelude::*;
                    cache_for_parse
                        .file_entries
                        .par_iter()
                        .filter(|e| !e.is_directory)
                        .filter_map(|entry| {
                            let content = cache_for_parse.file_content.get(&entry.relative_path)?;
                            crate::tools::semantic_identifiers::identifier_docs_for_file(
                                &entry.relative_path,
                                content,
                            )
                        })
                        .flatten()
                        .collect()
                })
                .await
                .unwrap_or_default();
            let file_count = new_cache
                .file_entries
                .iter()
                .filter(|e| !e.is_directory)
                .count();
            {
                let mut guard = ref_index.identifier_index.write().await;
                let needs_update = ref_index
                    .cache_generation
                    .load(std::sync::atomic::Ordering::Acquire)
                    == build_generation
                    && match &*guard {
                        None => true,
                        Some(idx) => idx.file_count != file_count,
                    };
                if needs_update {
                    *guard = Some(Arc::new(IdentifierIndex {
                        docs: doc_list.into(),
                        vectors: IdentifierVectorIndex::empty(),
                        dims: 0,
                        file_count,
                        built_at: std::time::Instant::now(),
                    }));
                    tracing::debug!(
                        ref_id = ref_id.0,
                        reason = "full ref warmup",
                        "IdentifierIndex replaced"
                    );
                }
            }

            // --- Phase 3: CAS baseline import (same as Shallow, zero Ollama) ---
            let report = import_baseline_for_ref(&state, ref_id, Arc::clone(&new_cache)).await;
            tracing::info!(
                ref_id = ref_id.0,
                hits = report.hits,
                misses = report.misses.len(),
                "ref_warmup full: baseline import complete"
            );

            // --- Phase 4: embed diff chunks via Ollama (gated by semaphore) ---
            embed_diff_chunks(&state, ref_id, &report.misses).await;

            tracing::info!(
                ref_id = ref_id.0,
                elapsed_ms = t0.elapsed().as_millis(),
                files = file_count,
                "ref_warmup full: complete"
            );
        });
    }

    /// Fetch instructions content (cached after first successful fetch).
    async fn get_instructions(&self) -> String {
        self.state
            .instructions_cache
            .get_or_init(|| async {
                match reqwest::get(INSTRUCTIONS_SOURCE_URL).await {
                    Ok(resp) => match resp.text().await {
                        Ok(text) => text,
                        Err(e) => {
                            tracing::warn!("Failed to read instructions response body: {e}");
                            "Context+ instructions are temporarily unavailable.".to_string()
                        }
                    },
                    Err(e) => {
                        tracing::warn!("Failed to fetch instructions from remote: {e}");
                        "Context+ instructions are temporarily unavailable.".to_string()
                    }
                }
            })
            .await
            .clone()
    }

    // Arg-extraction helpers: try snake_case key first, fall back to camelCase.
    // MCP clients may send either form; schemas advertise snake_case names but
    // callers have historically sent camelCase equivalents (topK, semanticWeight, …).

    fn get_str(args: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
        let camel = crate::server_helpers::snake_to_camel(key);
        crate::server_helpers::get_str_compat(args, key, &camel).map(|s| s.to_string())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn get_str_or(args: &serde_json::Map<String, Value>, key: &str, default: &str) -> String {
        Self::get_str(args, key).unwrap_or_else(|| default.to_string())
    }

    fn get_usize(args: &serde_json::Map<String, Value>, key: &str) -> Option<usize> {
        let camel = crate::server_helpers::snake_to_camel(key);
        crate::server_helpers::get_usize_compat(args, key, &camel)
    }

    fn get_f64(args: &serde_json::Map<String, Value>, key: &str) -> Option<f64> {
        let camel = crate::server_helpers::snake_to_camel(key);
        crate::server_helpers::get_f64_compat(args, key, &camel)
    }

    fn get_bool(args: &serde_json::Map<String, Value>, key: &str) -> Option<bool> {
        let camel = crate::server_helpers::snake_to_camel(key);
        crate::server_helpers::get_bool_compat(args, key, &camel)
    }

    fn get_u32(args: &serde_json::Map<String, Value>, key: &str) -> Option<u32> {
        let camel = crate::server_helpers::snake_to_camel(key);
        crate::server_helpers::get_u32_compat(args, key, &camel)
    }

    fn get_string_array(args: &serde_json::Map<String, Value>, key: &str) -> Option<Vec<String>> {
        let camel = crate::server_helpers::snake_to_camel(key);
        crate::server_helpers::get_string_array_compat(args, key, &camel)
    }

    fn ok_text(text: String) -> CallToolResult {
        CallToolResult::success(vec![Content::text(text)])
    }

    fn err_text(text: String) -> CallToolResult {
        CallToolResult::error(vec![Content::text(text)])
    }

    // --- Walk + analyze helpers ---

    /// Returns a snapshot of the project cache for the current session's ref,
    /// lazily initializing or refreshing when the TTL has expired.
    ///
    /// When `session_ref_id` is set (daemon mode, per-connection), this walks
    /// the worktree's own file tree (`ref.root_dir`) and stores the result in
    /// the per-ref `project_cache`. When `session_ref_id` is `None` (stdio
    /// mode), it falls back to the primary ref — identical to pre-U12 behaviour.
    ///
    /// All filesystem I/O runs inside `spawn_blocking`.
    /// Uses Arc to avoid deep-cloning the entire cache on every tool call.
    async fn ensure_project_cache(&self) -> Result<Arc<ProjectCache>> {
        let ref_index = self.current_ref().await;
        self.ensure_project_cache_for(&ref_index).await
    }

    fn tracker_is_running(ref_index: &crate::ref_index::RefIndex) -> bool {
        ref_index
            .tracker_handle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .is_some_and(|handle| handle.is_healthy())
    }

    /// The parent ref's current, flat project cache, which a linked worktree's
    /// cache layers its own changes over.
    async fn project_cache_base(
        &self,
        ref_index: &crate::ref_index::RefIndex,
    ) -> Option<Arc<ProjectCache>> {
        let parent = self.state.ref_index(ref_index.parent_ref_id?).await?;
        if parent.parent_ref_id.is_some() || parent.canonical_root == ref_index.canonical_root {
            return None;
        }
        let cache = Box::pin(self.ensure_project_cache_for(&parent))
            .await
            .ok()?;
        cache.file_content.base().is_none().then_some(cache)
    }

    /// This ref's cache when it can be served without a rebuild. A flat cache
    /// never consults the parent. A layered one is served while the parent
    /// still holds its base and has no pending change. No lock is nested.
    async fn current_project_cache(
        &self,
        ref_index: &crate::ref_index::RefIndex,
    ) -> Option<Arc<ProjectCache>> {
        let source_dirty = |owner: &crate::ref_index::RefIndex| {
            owner
                .tracker_handle
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
                .is_some_and(|handle| handle.is_source_dirty())
        };
        if source_dirty(ref_index) {
            return None;
        }
        let cache = ref_index.project_cache.read().await.clone()?;
        let fresh = Self::tracker_is_running(ref_index)
            || cache.last_refresh.elapsed().as_secs() < self.state.config.cache_ttl_secs;
        if !fresh {
            return None;
        }
        let Some(base) = cache.file_content.base() else {
            return Some(cache);
        };
        let parent = self.state.ref_index(ref_index.parent_ref_id?).await?;
        if source_dirty(&parent) {
            return None;
        }
        let parent_cache = parent.project_cache.read().await.clone()?;
        Arc::ptr_eq(base, parent_cache.file_content.own()).then_some(cache)
    }

    /// Build-or-reuse the walked file cache for a **specific** ref, rather than
    /// the session's `current_ref()`. Used by tools that can be directed at an
    /// attached worktree (e.g. `get_blast_radius` with a `path` arg) so the scan
    /// runs against the worktree the caller asked for instead of silently
    /// answering from the connection's primary ref.
    async fn ensure_project_cache_for(
        &self,
        ref_index: &Arc<crate::ref_index::RefIndex>,
    ) -> Result<Arc<ProjectCache>> {
        if let Some(cache) = self.current_project_cache(ref_index).await {
            return Ok(cache);
        }
        // Only a rebuild, or a layered cache whose parent moved, needs the parent's cache.
        let base = self.project_cache_base(ref_index).await;
        // Serialize rebuilds with invalidation so an older walk cannot overwrite it.
        let mut project_guard = ref_index.project_cache.write().await;
        let dirty = ref_index
            .tracker_handle
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .is_some_and(|handle| handle.take_source_dirty());
        if dirty {
            *project_guard = None;

            ref_index
                .cache_generation
                .fetch_add(1, std::sync::atomic::Ordering::Release);
        }
        let ttl_secs = self.state.config.cache_ttl_secs;
        let tracker_running = Self::tracker_is_running(ref_index);

        // Fast path: a running tracker makes this cache authoritative until a
        // real file change invalidates it. Without a tracker, retain the TTL
        // fallback so external edits are eventually observed.
        {
            if let Some(ref cache) = *project_guard {
                let age_secs = cache.last_refresh.elapsed().as_secs();
                if (tracker_running || age_secs < ttl_secs) && base_is_current(cache, &base) {
                    tracing::debug!(
                        ref_id = %ref_index.cas_ref_id_hex,
                        tracker_running,
                        age_secs,
                        ttl_secs,
                        "ProjectCache hit"
                    );
                    return Ok(Arc::clone(cache));
                }
                tracing::debug!(
                    ref_id = %ref_index.cas_ref_id_hex,
                    age_secs,
                    ttl_secs,
                    "Rebuilding ProjectCache: tracker absent and TTL expired"
                );
            } else {
                tracing::debug!(
                    ref_id = %ref_index.cas_ref_id_hex,
                    tracker_running,
                    "Rebuilding ProjectCache: cache empty"
                );
            }
        }

        // Slow path: rebuild cache
        tracing::debug!(
            ref_id = %ref_index.cas_ref_id_hex,
            root = %ref_index.root_dir.display(),
            "walking project cache root"
        );
        let root = ref_index.root_dir.clone();
        let config = self.state.config.clone();

        let record = ref_index.parent_ref_id.is_none();
        let new_cache =
            tokio::task::spawn_blocking(move || load_project_cache(&root, &config, base, record))
                .await
                .map_err(|e| ContextPlusError::Other(format!("spawn_blocking failed: {e}")))?;

        let arc_cache = Arc::new(new_cache);

        // Store the Arc in per-ref state (cheap clone of the Arc pointer)
        *project_guard = Some(Arc::clone(&arc_cache));

        tracing::debug!(
            ref_id = %ref_index.cas_ref_id_hex,
            files = arc_cache.file_content.len(),
            root = %ref_index.root_dir.display(),
            "ProjectCache replaced after rebuild"
        );

        Ok(arc_cache)
    }

    /// Invalidate the project cache for the current session's ref.
    ///
    /// Called by the file watcher when files change. Clears the per-ref
    /// `project_cache` and `identifier_index` so they rebuild with fresh data
    /// on the next tool call.
    pub async fn invalidate_project_cache(&self) {
        self.invalidate_project_cache_with_reason("explicit invalidation")
            .await;
    }

    async fn invalidate_project_cache_with_reason(&self, reason: &'static str) {
        let ref_index = self.current_ref().await;
        let mut guard = ref_index.project_cache.write().await;
        let project_cache_was_populated = guard.is_some();
        *guard = None;
        drop(guard);
        let idx_guard = ref_index.identifier_index.read().await;
        let identifier_index_was_populated = idx_guard.is_some();

        drop(idx_guard);
        let lexical_guard = ref_index.lexical_search_cache.read().await;
        let lexical_index_was_populated = lexical_guard.is_some();

        tracing::debug!(
            ref_id = %ref_index.cas_ref_id_hex,
            project_cache_was_populated,
            identifier_index_was_populated,
            lexical_index_was_populated,
            reason,
            "ProjectCache, IdentifierIndex, and LexicalIndex invalidated"
        );
    }

    /// Incrementally re-embed specific changed files without invalidating the entire cache.
    ///
    /// Uses the current session's ref (`current_ref()`) so per-worktree caches
    /// stay isolated. Integrates CAS dedup: before sending a chunk to Ollama the
    /// content hash is checked against the per-ref manifest chain; on a hit the
    /// existing CAS blob is reused and only the manifest entry is updated —
    /// no Ollama call is made (U12 diff-only embedding).
    ///
    /// Returns (updated_count, skipped_count).
    pub async fn incremental_reembed(&self, files: &[std::path::PathBuf]) -> (usize, usize) {
        let outcome = self.incremental_reembed_detailed(files).await;
        (outcome.updated, outcome.skipped)
    }

    async fn incremental_reembed_detailed(
        &self,
        files: &[std::path::PathBuf],
    ) -> IncrementalReembedOutcome {
        use crate::cache::cas::{CasStore, ChunkHash, ChunkKey};

        let mut updated = 0usize;
        let mut skipped = 0usize;
        let mut content_changed = false;

        let max_file_size = self.state.config.max_embed_file_size as u64;
        let ref_index = self.current_ref().await;
        let project_cache = ref_index.project_cache.read().await.as_ref().cloned();

        // CAS setup for diff-only embedding via U6 content-addressed store.
        let mcp_data_dir = ref_index.root_dir.join(".mcp_data");
        let cas = CasStore::new(mcp_data_dir, self.state.config.document_cache_identity());
        let ref_id_hex = ref_index.cas_ref_id_hex.clone();

        let mut texts_to_embed: Vec<(String, String, String)> = Vec::new(); // (rel_path, hash, text)
        let mut cas_hit_entries: Vec<(String, String, Vec<f32>)> = Vec::new(); // (rel_path, hash, vector) from CAS
        let mut cas_manifest_updates: Vec<(ChunkKey, ChunkHash)> = Vec::new();
        // Keys evicted from the in-memory cache because the source file was deleted.
        // Must be plumbed through to `save_vector_store_merged_with_deletions` —
        // a plain merged save would re-populate them from disk on the next save.
        let mut deletions: Vec<String> = Vec::new();

        for file_path in files {
            let rel_path = match file_path.strip_prefix(&ref_index.root_dir) {
                Ok(r) => r.to_string_lossy().to_string(),
                Err(_) => match file_path.strip_prefix(&self.state.root_dir) {
                    Ok(r) => r.to_string_lossy().to_string(),
                    Err(_) => file_path.to_string_lossy().to_string(),
                },
            };

            let oversized = tokio::fs::metadata(file_path)
                .await
                .is_ok_and(|meta| meta.len() > max_file_size);

            let content = match tokio::fs::read_to_string(file_path).await {
                Ok(c) => c,
                Err(_) => {
                    // File deleted — remove from cache and record so the save
                    // path can evict the same key from disk.
                    let mut cache = ref_index.embedding_cache.write().await;
                    cache.remove(&rel_path);
                    deletions.push(rel_path.clone());
                    updated += 1;
                    content_changed = true;
                    continue;
                }
            };

            let hash = crate::core::parser::hash_content(&content);

            let project_content_matches = project_cache.as_ref().map(|cache| {
                cache
                    .file_content
                    .get(&rel_path)
                    .is_some_and(|cached| cached.as_str() == content)
            });

            // Check if content actually changed (in-memory cache hit)
            let embedding_cache_matches = {
                let cache = ref_index.embedding_cache.read().await;
                cache.get(&rel_path).is_some_and(|entry| entry.hash == hash)
            };
            if project_content_matches == Some(true)
                || (project_content_matches.is_none() && embedding_cache_matches)
            {
                skipped += 1;
                continue;
            }

            content_changed = true;
            if oversized || embedding_cache_matches {
                skipped += 1;
                continue;
            }

            let text =
                build_embedding_document(&rel_path, &content, self.state.config.embed_doc_shape);

            // CAS dedup: check if this chunk's BLAKE3 hash exists in the parent chain.
            // chunk_idx = 0 because we treat each file as a single chunk here.
            let chunk_hash = ChunkHash::of(&text);
            let key = ChunkKey::new(rel_path.clone(), 0);
            let cas_entry = match cas.lookup_chunk(&ref_id_hex, &key) {
                Ok(Some(h)) if h == chunk_hash => {
                    // Hash matches — read existing blob from CAS (no Ollama call).
                    match cas.read_blob(&h) {
                        Ok(Some(vec)) => {
                            tracing::debug!(
                                rel_path = %rel_path,
                                "CAS hit: skipping Ollama embed"
                            );
                            Some((chunk_hash.clone(), vec))
                        }
                        _ => None, // blob missing — fall through to embed
                    }
                }
                _ => None, // miss or error — fall through to embed
            };

            if let Some((ch, vec)) = cas_entry {
                // CAS hit: record the manifest entry and the in-memory cache update.
                cas_hit_entries.push((rel_path, hash, vec));
                // No new CAS write needed — blob already exists.
                cas_manifest_updates.push((key, ch));
            } else {
                texts_to_embed.push((rel_path, hash, text));
            }
        }

        // Source freshness must not wait for the embedding provider.
        if content_changed {
            self.invalidate_project_cache_with_reason("tracker inspected changed content")
                .await;
        }

        // Apply CAS hit entries to the in-memory embedding cache.
        if !cas_hit_entries.is_empty() {
            let mut cache = ref_index.embedding_cache.write().await;
            for (rel_path, hash, vec) in &cas_hit_entries {
                cache.insert(
                    rel_path.clone(),
                    CacheEntry {
                        hash: hash.clone(),
                        vector: vec.clone(),
                    },
                );
                updated += 1;
            }
        }

        // Nothing to embed and nothing to delete → skip the save entirely.
        if texts_to_embed.is_empty() && deletions.is_empty() {
            // Still need to update the CAS manifest for hits.
            if !cas_manifest_updates.is_empty()
                && let Err(e) = cas.update_manifest(&ref_id_hex, &cas_manifest_updates)
            {
                tracing::warn!("CAS manifest update failed (non-fatal): {e}");
            }
            if cas_hit_entries.is_empty() {
                return IncrementalReembedOutcome {
                    updated,
                    skipped,
                    content_changed,
                };
            }
            // Fall through to persist the updated in-memory cache to disk.
        }

        // Embed any pending texts (may be empty if this batch was deletion-only).
        let mut embed_failed = false;
        if !texts_to_embed.is_empty() {
            let embed_texts: Vec<String> =
                texts_to_embed.iter().map(|(_, _, t)| t.clone()).collect();
            match self.state.ollama.embed_documents(&embed_texts).await {
                Ok(vectors) => {
                    let mut cache = ref_index.embedding_cache.write().await;
                    for (i, (rel_path, hash, text)) in texts_to_embed.iter().enumerate() {
                        if i < vectors.len() {
                            let vec = &vectors[i];
                            cache.insert(
                                rel_path.clone(),
                                CacheEntry {
                                    hash: hash.clone(),
                                    vector: vec.clone(),
                                },
                            );
                            updated += 1;
                            // Write new blob + manifest entry to CAS.
                            let chunk_hash = ChunkHash::of(text);
                            let key = ChunkKey::new(rel_path.clone(), 0);
                            if let Err(e) = cas.write_blob(&chunk_hash, vec) {
                                tracing::warn!("CAS write_blob failed (non-fatal): {e}");
                            } else {
                                cas_manifest_updates.push((key, chunk_hash));
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Incremental re-embed failed: {e}");
                    embed_failed = true;
                }
            }
        }

        // Flush all CAS manifest updates in one batch.
        if !embed_failed
            && !cas_manifest_updates.is_empty()
            && let Err(e) = cas.update_manifest(&ref_id_hex, &cas_manifest_updates)
        {
            tracing::warn!("CAS manifest batch update failed (non-fatal): {e}");
        }

        // Persist on every successful pass — even pure-deletion ones — so the
        // on-disk cache evicts the deleted keys. A plain merged save would
        // silently re-populate them from disk and the cache would grow without
        // bound. Build the snapshot inside the write-guard scope, drop the
        // guard, then run the save off the Tokio worker via spawn_blocking.
        if !embed_failed {
            let store_to_save = {
                let cache = ref_index.embedding_cache.read().await;
                let store = crate::core::embeddings::VectorStore::from_cache(&cache);
                drop(cache);
                store
            };

            let embed_cache_name = cache_name("embeddings", &self.state.config);
            let root = ref_index.root_dir.clone();
            let deletions_owned = std::mem::take(&mut deletions);
            let result = tokio::task::spawn_blocking(move || {
                // Merge with disk under fd-lock: the runtime in-memory store
                // only covers keys this session has touched, so an overwrite
                // would silently drop entries written by a concurrent warmup
                // binary or a second MCP instance. Pass `deletions_owned` so
                // keys we evicted from memory also get evicted from disk.
                match store_to_save {
                    Some(s) => rkyv_store::save_vector_store_merged_with_deletions(
                        &root,
                        &embed_cache_name,
                        &s,
                        &deletions_owned,
                    ),
                    None => {
                        // No in-memory entries (e.g. only deletions, and the
                        // cache was emptied). Build an empty store so the
                        // deletions still get applied on disk.
                        let empty = crate::core::embeddings::VectorStore::new(
                            0,
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                        );
                        rkyv_store::save_vector_store_merged_with_deletions(
                            &root,
                            &embed_cache_name,
                            &empty,
                            &deletions_owned,
                        )
                    }
                }
            })
            .await;
            match result {
                Ok(Err(e)) => {
                    tracing::warn!("Failed to save incremental embedding cache: {e}")
                }
                Err(join_err) => {
                    tracing::warn!("save_vector_store spawn_blocking join failed: {join_err}")
                }
                Ok(Ok(())) => {}
            }
        }

        IncrementalReembedOutcome {
            updated,
            skipped,
            content_changed,
        }
    }

    /// Ensure the identifier index is built and cached.
    /// Returns cached index if TTL hasn't expired and file count is unchanged.
    /// Otherwise rebuilds: parses all symbols, embeds them, caches the result.
    async fn install_identifier_index_if_current(
        &self,
        source_cache: &Arc<ProjectCache>,
        build_generation: u64,
        index: &Arc<IdentifierIndex>,
    ) -> bool {
        use std::sync::atomic::Ordering;

        let ref_index = self.current_ref().await;
        let project_guard = ref_index.project_cache.read().await;
        let mut identifier_guard = ref_index.identifier_index.write().await;
        let source_is_current = project_guard
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, source_cache));
        let generation_is_current =
            ref_index.cache_generation.load(Ordering::Acquire) == build_generation;
        if !source_is_current || !generation_is_current {
            return false;
        }

        *ref_index.identifier_source.write().await = Some(Arc::clone(source_cache));
        *identifier_guard = Some(Arc::clone(index));
        ref_index
            .identifier_inherited
            .store(false, Ordering::Release);
        true
    }

    fn ensure_identifier_index<'a>(
        &'a self,
        cache: &'a Arc<ProjectCache>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<IdentifierIndex>>> + Send + 'a>,
    > {
        Box::pin(async move {
            let file_count = cache
                .file_entries
                .iter()
                .filter(|e| !e.is_directory)
                .count();
            let ref_index = self.current_ref().await;
            let tracker_running = Self::tracker_is_running(&ref_index);
            let source = ref_index.identifier_source.read().await.as_ref().cloned();
            let source_matches = source.as_ref().is_none_or(|old| Arc::ptr_eq(old, cache));

            // Fast path: index exists, file count is unchanged, and either the
            // tracker is authoritative or the tracker-off TTL remains valid.
            // A warmup-built index carries docs but no vectors (dims == 0); serving
            // it would score every identifier at zero similarity, so it does not
            // count as built.
            {
                let guard = ref_index.identifier_index.read().await;
                if let Some(ref idx) = *guard
                    && idx.file_count == file_count
                    && source_matches
                    && !ref_index
                        .identifier_inherited
                        .load(std::sync::atomic::Ordering::Acquire)
                    && (tracker_running
                        || idx.built_at.elapsed().as_secs() < IDENTIFIER_INDEX_TTL_SECS)
                    && (idx.dims > 0 || idx.docs.is_empty())
                {
                    tracing::debug!(
                        ref_id = %ref_index.cas_ref_id_hex,
                        tracker_running,
                        age_secs = idx.built_at.elapsed().as_secs(),
                        "IdentifierIndex hit"
                    );
                    return Ok(Arc::clone(idx));
                }

                let reason = match guard.as_ref() {
                    None => "cache empty",
                    Some(idx) if idx.file_count != file_count => "project file count changed",
                    Some(idx) if idx.dims == 0 && !idx.docs.is_empty() => {
                        "warmup index has no vectors"
                    }
                    Some(_) => "tracker absent and TTL expired",
                };
                tracing::debug!(
                    ref_id = %ref_index.cas_ref_id_hex,
                    tracker_running,
                    file_count,
                    reason,
                    "Rebuilding IdentifierIndex"
                );
            }

            // An index inherited from the parent ref answers for another tree,
            // so it is never served while this ref rebuilds.
            let previous = ref_index
                .identifier_index
                .read()
                .await
                .as_ref()
                .cloned()
                .filter(|_| {
                    !ref_index
                        .identifier_inherited
                        .load(std::sync::atomic::Ordering::Acquire)
                });
            if let (Some(source), Some(previous)) = (&source, previous) {
                let changed = cache
                    .file_content
                    .iter()
                    .filter(|(path, content)| source.file_content.get(path) != Some(*content))
                    .count()
                    + source
                        .file_content
                        .keys()
                        .filter(|path| !cache.file_content.contains_key(path))
                        .count();
                if changed as f64
                    > source.file_content.len() as f64
                        * crate::tools::semantic_search::FULL_REBUILD_CHANGE_FRACTION
                {
                    if ref_index
                        .identifier_rebuilding
                        .compare_exchange(
                            false,
                            true,
                            std::sync::atomic::Ordering::AcqRel,
                            std::sync::atomic::Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        let server = self.clone();
                        let cache = Arc::clone(cache);
                        let flag = Arc::clone(&ref_index.identifier_rebuilding);
                        let task = tokio::spawn(async move {
                            let _reset = RefreshGuard(flag);
                            if let Err(error) = server.build_identifier_index(&cache, true).await {
                                tracing::warn!(%error, "Background identifier rebuild failed");
                            }
                        });
                        ref_index.track_background_task(&task);
                    }
                    return Ok(previous);
                }
            }
            self.build_identifier_index(cache, false).await
        })
    }

    /// The identifier index of a linked worktree's parent, built first when
    /// missing, paired with the project cache it was built from.
    async fn parent_identifier_index(
        &self,
        ref_index: &crate::ref_index::RefIndex,
    ) -> Option<(Arc<IdentifierIndex>, Arc<ProjectCache>)> {
        let parent_id = ref_index.parent_ref_id?;
        let parent = self.state.ref_index(parent_id).await?;
        if parent.parent_ref_id.is_some() || parent.canonical_root == ref_index.canonical_root {
            return None;
        }
        let parent_server = self.with_session(parent_id);
        let parent_cache = parent_server.ensure_project_cache().await.ok()?;
        parent_server
            .ensure_identifier_index(&parent_cache)
            .await
            .ok()?;
        let index = parent.identifier_index.read().await;
        let source = parent.identifier_source.read().await;
        match (index.as_ref(), source.as_ref()) {
            (Some(index), Some(source)) if index.dims > 0 => {
                Some((Arc::clone(index), Arc::clone(source)))
            }
            _ => None,
        }
    }

    async fn build_identifier_index(
        &self,
        cache: &Arc<ProjectCache>,
        background: bool,
    ) -> Result<Arc<IdentifierIndex>> {
        self.build_identifier_index_seeded(cache, background, true)
            .await
    }

    /// Builds the identifier index of the current ref. A first build of a
    /// linked worktree, when `seed_from_parent`, starts from its parent's
    /// index and parses only the files whose content differs.
    async fn build_identifier_index_seeded(
        &self,
        cache: &Arc<ProjectCache>,
        background: bool,
        seed_from_parent: bool,
    ) -> Result<Arc<IdentifierIndex>> {
        let ref_index = self.current_ref().await;
        let update_guard = ref_index.identifier_update.lock().await;
        let source = ref_index.identifier_source.read().await.as_ref().cloned();
        let file_count = cache
            .file_entries
            .iter()
            .filter(|entry| !entry.is_directory)
            .count();
        if source.as_ref().is_some_and(|old| Arc::ptr_eq(old, cache))
            && let Some(index) = ref_index.identifier_index.read().await.as_ref()
            && index.dims > 0
        {
            return Ok(Arc::clone(index));
        }
        // Slow path: rebuild identifier index
        tracing::info!(
            file_count,
            "Building identifier index (parsing + embedding)"
        );
        let build_generation = ref_index
            .cache_generation
            .load(std::sync::atomic::Ordering::Acquire);
        let cache_clone = cache.clone();
        let previous = ref_index.identifier_index.read().await.as_ref().cloned();
        let mut incremental =
            previous.as_ref().is_some_and(|index| index.dims > 0) && source.is_some();
        let parent_seed = if incremental || !seed_from_parent {
            None
        } else {
            self.parent_identifier_index(&ref_index).await
        };
        let from_parent = parent_seed.is_some();
        let (previous, source) = match parent_seed {
            Some((parent_index, parent_source)) => {
                incremental = true;
                (Some(parent_index), Some(parent_source))
            }
            None => (previous, source),
        };
        let changed_paths: std::collections::HashSet<String> = if incremental {
            let old = source.as_ref().unwrap();
            cache
                .file_content
                .keys()
                .chain(old.file_content.keys())
                .filter(|path| cache.file_content.get(path) != old.file_content.get(path))
                .cloned()
                .collect()
        } else {
            cache.file_content.keys().cloned().collect()
        };
        let parse_paths = changed_paths.clone();
        let use_snapshot = !incremental && snapshots::enabled(&self.state.config, &ref_index);
        let snapshot_root = ref_index.root_dir.clone();
        let snapshot_config = self.state.config.clone();
        let started = Instant::now();

        // Step 1: Parse symbols (CPU-bound), while the vectors load below. A
        // first build takes the documents of unchanged files from the snapshot.
        let parse = tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            let seeded = use_snapshot
                .then(|| {
                    snapshots::load_identifiers(&snapshot_root, &snapshot_config, &cache_clone)
                })
                .flatten();
            let from_snapshot = seeded.is_some();
            let (mut docs, parse_paths) = seeded.unwrap_or((Vec::new(), parse_paths));
            let parsed_files = parse_paths.len();
            docs.par_extend(
                cache_clone
                    .file_entries
                    .par_iter()
                    .filter(|entry| !entry.is_directory)
                    .filter_map(|entry| {
                        let content = cache_clone.file_content.get(&entry.relative_path)?;
                        if !parse_paths.contains(&entry.relative_path) {
                            return None;
                        }
                        crate::tools::semantic_identifiers::identifier_docs_for_file(
                            &entry.relative_path,
                            content,
                        )
                    })
                    .flatten(),
            );
            (docs, parsed_files, from_snapshot)
        });

        let vectors_started = Instant::now();
        // Identifier texts are keyed by repo-relative path and signature, so a
        // worktree shares the primary's resident vectors and keeps only its own
        // misses in a per-ref overlay persisted under its root.
        let id_cache_name = cache_name("identifier-embeddings", &self.state.config);
        let is_worktree = ref_index.parent_ref_id.is_some();
        let base_owner = match ref_index.parent_ref_id {
            Some(parent_id) => self
                .state
                .ref_index(parent_id)
                .await
                .unwrap_or_else(|| Arc::clone(&ref_index)),
            None => Arc::clone(&ref_index),
        };
        let resident = base_owner
            .identifier_vectors
            .get_or_try_init(|| async {
                let root = base_owner.root_dir.clone();
                let name = id_cache_name.clone();
                tokio::task::spawn_blocking(move || {
                    Arc::new(RwLock::new(load_identifier_vectors(&root, &name)))
                })
                .await
                .map_err(|error| ContextPlusError::Other(error.to_string()))
            })
            .await?;
        let resident = ref_index
            .identifier_vectors
            .get_or_init(|| async { Arc::clone(resident) })
            .await;
        if is_worktree
            && !ref_index
                .identifier_overlay_loaded
                .load(std::sync::atomic::Ordering::Acquire)
        {
            let root = ref_index.root_dir.clone();
            let name = id_cache_name.clone();
            let loaded = tokio::task::spawn_blocking(move || load_identifier_vectors(&root, &name))
                .await
                .map_err(|error| ContextPlusError::Other(error.to_string()))?;
            let mut overlay = ref_index.identifier_vector_overlay.write().await;
            if !ref_index
                .identifier_overlay_loaded
                .swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                for (key, entry) in loaded {
                    overlay.entry(key).or_insert(entry);
                }
            }
        }
        let vectors_ms = vectors_started.elapsed().as_millis();
        let (identifier_docs, parsed_files, from_snapshot) = parse
            .await
            .map_err(|e| ContextPlusError::Other(format!("spawn_blocking failed: {e}")))?;
        let parse_ms = started.elapsed().as_millis();

        if identifier_docs.is_empty() && !incremental {
            let idx = Arc::new(IdentifierIndex {
                docs: Vec::new().into(),
                vectors: IdentifierVectorIndex::empty(),
                dims: 0,
                file_count,
                built_at: Instant::now(),
            });
            if !self
                .install_identifier_index_if_current(cache, build_generation, &idx)
                .await
            {
                tracing::debug!(
                    ref_id = %ref_index.cas_ref_id_hex,
                    build_generation,
                    reason = "project cache or generation changed during build",
                    "IdentifierIndex build discarded"
                );
                drop(update_guard);
                let fresh_cache = self.ensure_project_cache().await?;
                return Box::pin(self.ensure_identifier_index(&fresh_cache)).await;
            }
            tracing::debug!(
                ref_id = %ref_index.cas_ref_id_hex,
                reason = "parsed corpus contains no identifiers",
                "IdentifierIndex replaced after rebuild"
            );
            return Ok(idx);
        }

        // Step 2: Check identifier embedding cache on disk, embed only missing
        let n_identifiers = identifier_docs.len();
        tracing::info!(
            identifiers = n_identifiers,
            "Embedding identifiers (using disk cache for warm hits)"
        );

        let overlay = ref_index.identifier_vector_overlay.read().await;
        let id_caches = resident.read().await;

        // Partition: cached vs uncached identifiers. A cached vector is shared
        // with the map it was found in, never copied.
        let mut result_vectors: Vec<Option<Arc<[f32]>>> = Vec::with_capacity(n_identifiers);
        let mut uncached_indices: Vec<usize> = Vec::new();
        let mut uncached_texts: Vec<String> = Vec::new();

        for (i, doc) in identifier_docs.iter().enumerate() {
            if let Some(vector) = overlay.get(&doc.text).or_else(|| id_caches.get(&doc.text)) {
                result_vectors.push(Some(Arc::clone(vector)));
                continue;
            }
            result_vectors.push(None);
            uncached_indices.push(i);
            uncached_texts.push(doc.text.clone());
        }

        tracing::info!(
            cached = n_identifiers - uncached_indices.len(),
            uncached = uncached_indices.len(),
            "Identifier embedding cache hit/miss"
        );
        let lookup_done = started.elapsed().as_millis();

        drop(id_caches);
        drop(overlay);
        // Embed only uncached identifiers, in chunks to survive MCP connection timeouts.
        if !uncached_texts.is_empty() {
            let chunk_size = self.state.ollama.batch_size();
            for chunk_start in (0..uncached_indices.len()).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(uncached_indices.len());
                let chunk_texts = &uncached_texts[chunk_start..chunk_end];

                let chunk_vectors = self.state.ollama.embed_documents(chunk_texts).await?;
                for (&idx, vector) in uncached_indices[chunk_start..chunk_end]
                    .iter()
                    .zip(chunk_vectors)
                {
                    result_vectors[idx] = Some(Arc::from(vector));
                }
            }
        }

        if !uncached_indices.is_empty() {
            let target = if is_worktree {
                Arc::clone(&ref_index.identifier_vector_overlay)
            } else {
                Arc::clone(resident)
            };
            let mut target_guard = target.write().await;
            let mut unsaved = ref_index.identifier_unsaved.lock().unwrap();
            for &i in &uncached_indices {
                if let Some(vector) = &result_vectors[i] {
                    let key = identifier_docs[i].text.clone();
                    target_guard.insert(key.clone(), Arc::clone(vector));
                    unsaved.insert(key, Arc::clone(vector));
                }
            }
            drop(unsaved);
            drop(target_guard);
            let ticket = ref_index
                .identifier_persist_generation
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                + 1;
            let persist_generation = Arc::clone(&ref_index.identifier_persist_generation);
            let persist_root = ref_index.root_dir.clone();
            let save_lock = Arc::clone(&ref_index.identifier_save_lock);
            let unsaved = Arc::clone(&ref_index.identifier_unsaved);
            let resident_set = Arc::downgrade(&target);
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                // Serialized with the budget flush: both merge into the same file.
                let _save = save_lock.lock().await;
                if persist_generation.load(std::sync::atomic::Ordering::Acquire) != ticket {
                    return;
                }
                // Only the vectors embedded since the last save: the save merges
                // them into what is already on disk, or rebuilds a missing file
                // from the whole resident set.
                let pending = std::mem::take(&mut *unsaved.lock().unwrap());
                let Some(data) = identifier_cache_data(&pending) else {
                    return;
                };
                let result = tokio::task::spawn_blocking(move || {
                    rkyv_store::save_cache_rebuilding(&persist_root, &id_cache_name, &data, || {
                        let vectors = resident_set.upgrade()?;
                        identifier_cache_data(&vectors.blocking_read())
                    })
                })
                .await;
                if !matches!(result, Ok(Ok(()))) {
                    if let Ok(Err(error)) = &result {
                        tracing::warn!(%error, "Identifier cache persistence failed");
                    }
                    let mut unsaved = unsaved.lock().unwrap();
                    for (key, vector) in pending {
                        unsaved.entry(key).or_insert(vector);
                    }
                }
            });
        }

        let embed_ms = started.elapsed().as_millis() - lookup_done;
        let dims = result_vectors
            .first()
            .and_then(|v| v.as_ref())
            .map_or(0, |v| v.len());
        let zero: Arc<[f32]> = Arc::from(vec![0.0; dims]);

        let mut docs_by_file: std::collections::BTreeMap<String, Vec<_>> =
            std::collections::BTreeMap::new();
        let mut vectors_by_file: std::collections::BTreeMap<String, Vec<Arc<[f32]>>> =
            std::collections::BTreeMap::new();
        for (doc, vector) in identifier_docs.into_iter().zip(result_vectors) {
            let file_vectors = vectors_by_file.entry(doc.path.clone()).or_default();
            if dims != 0 {
                file_vectors.push(vector.unwrap_or_else(|| Arc::clone(&zero)));
            }
            docs_by_file.entry(doc.path.clone()).or_default().push(doc);
        }
        let mut docs = if incremental {
            previous.as_ref().unwrap().docs.files.clone()
        } else {
            std::collections::BTreeMap::new()
        };
        let mut vectors = if incremental {
            previous.as_ref().unwrap().vectors.file_segments().clone()
        } else {
            std::collections::BTreeMap::new()
        };
        for path in &changed_paths {
            docs.remove(path);
            vectors.remove(path);
        }
        docs.extend(
            docs_by_file
                .into_iter()
                .map(|(path, docs)| (path, Arc::new(docs))),
        );
        vectors.extend(
            vectors_by_file
                .into_iter()
                .map(|(path, vectors)| (path, Arc::new(vectors))),
        );
        let previous_dims = previous.as_ref().map_or(0, |index| index.dims);
        if incremental && dims != 0 && dims != previous_dims {
            *ref_index.identifier_source.write().await = None;
            drop(update_guard);
            return Box::pin(self.build_identifier_index_seeded(
                cache,
                background,
                seed_from_parent && !from_parent,
            ))
            .await;
        }
        let dims = if incremental { previous_dims } else { dims };
        let idx = Arc::new(IdentifierIndex {
            docs: Segmented::from_files(docs),
            vectors: IdentifierVectorIndex::new(vectors, dims),
            dims,
            file_count,
            built_at: Instant::now(),
        });
        tracing::info!(
            phase = "identifier_build",
            incremental,
            from_parent,
            from_snapshot,
            parsed_files,
            parse_ms,
            vectors_ms,
            embed_ms,
            uncached = uncached_indices.len(),
            assemble_ms = started.elapsed().as_millis() - lookup_done - embed_ms,
            "cold-start phase"
        );

        if !self
            .install_identifier_index_if_current(cache, build_generation, &idx)
            .await
        {
            tracing::debug!(
                ref_id = %ref_index.cas_ref_id_hex,
                build_generation,
                reason = "project cache or generation changed during build",
                "IdentifierIndex build discarded"
            );
            drop(update_guard);
            // A background rebuild must not re-walk a ref whose caches were
            // evicted or replaced; the next request rebuilds on demand.
            if background {
                return Ok(idx);
            }
            let fresh_cache = self.ensure_project_cache().await?;
            return Box::pin(self.ensure_identifier_index(&fresh_cache)).await;
        }
        tracing::debug!(
            ref_id = %ref_index.cas_ref_id_hex,
            identifiers = idx.docs.len(),
            dims = idx.dims,
            "IdentifierIndex replaced after rebuild"
        );
        let change = if incremental || from_snapshot {
            snapshots::Change::Files {
                changed: parsed_files,
                documents: file_count,
            }
        } else {
            snapshots::Change::Full
        };
        if matches!(change, snapshots::Change::Full) || parsed_files > 0 {
            snapshots::schedule_identifiers(&self.state.config, &ref_index, change);
        }

        Ok(idx)
    }

    /// Build or reuse the lexical index for the current ref.
    ///
    /// Project-cache identity catches TTL/manual replacements while generation
    /// catches tracker invalidations before another caller has rebuilt the
    /// project cache. The write lock makes concurrent cold callers single-flight.
    async fn ensure_lexical_index(
        &self,
        project_cache: &Arc<ProjectCache>,
    ) -> Result<Arc<CachedLexicalIndex>> {
        let ref_index = self.current_ref().await;
        if project_cache.file_content.base().is_some()
            && let Some(cached) = self
                .layered_lexical_index(&ref_index, project_cache)
                .await?
        {
            return Ok(cached);
        }
        self.ensure_lexical_index_for(&ref_index, project_cache)
            .await
    }

    /// Keyword index of a ref whose files layer over its parent's: the
    /// parent's shared index with this ref's changed and removed files masked,
    /// plus an index of only its changed and added files. `None` when the
    /// parent's index does not match the base this ref's files layer over.
    async fn layered_lexical_index(
        &self,
        ref_index: &Arc<crate::ref_index::RefIndex>,
        project_cache: &Arc<ProjectCache>,
    ) -> Result<Option<Arc<CachedLexicalIndex>>> {
        use crate::tools::lexical_search::BaseMask;
        use std::sync::atomic::Ordering;

        let Some(parent_id) = ref_index.parent_ref_id else {
            return Ok(None);
        };
        let Some(parent) = self.state.ref_index(parent_id).await else {
            return Ok(None);
        };
        let Some(parent_cache) = parent.project_cache.read().await.clone() else {
            return Ok(None);
        };
        if !project_cache
            .file_content
            .base()
            .is_some_and(|base| Arc::ptr_eq(base, parent_cache.file_content.own()))
        {
            return Ok(None);
        }
        let base = self
            .ensure_lexical_index_for(&parent, &parent_cache)
            .await?;
        let _update = ref_index.lexical_update.lock().await;
        let generation = ref_index.cache_generation.load(Ordering::Acquire);
        let previous = ref_index.lexical_search_cache.read().await.clone();
        if let Some(cached) = &previous
            && cached.generation == generation
            && Arc::ptr_eq(&cached.project_cache, project_cache)
            && cached
                .base
                .as_ref()
                .is_some_and(|layer| Arc::ptr_eq(&layer.cached, &base))
        {
            return Ok(previous);
        }
        if base.base.is_some() || !Arc::ptr_eq(&base.project_cache, &parent_cache) {
            // The parent is still rebuilding its index: a stale answer from this ref's
            // last entry, even the inherited one, beats copying the parent's index.
            return Ok(previous);
        }

        let source = Arc::clone(project_cache);
        let parent_index = Arc::clone(&base);
        let (index, document_paths, mask) = tokio::task::spawn_blocking(move || {
            let files = &source.file_content;
            let changed = |path: &str| files.shadows(path) || files.own().contains_key(path);
            let present: std::collections::HashSet<&str> = source
                .file_entries
                .iter()
                .filter(|entry| !entry.is_directory)
                .map(|entry| entry.relative_path.as_str())
                .collect();
            let masked = parent_index
                .document_paths
                .iter()
                .enumerate()
                .filter(|(_, path)| {
                    !path.is_empty() && (changed(path) || !present.contains(path.as_str()))
                })
                .map(|(i, _)| i)
                .collect();
            let in_base: std::collections::HashSet<&str> = parent_index
                .document_paths
                .iter()
                .map(String::as_str)
                .collect();
            let (index, document_paths) = build_lexical_index(
                source
                    .file_entries
                    .iter()
                    .filter(|entry| {
                        !entry.is_directory
                            && (changed(&entry.relative_path)
                                || !in_base.contains(entry.relative_path.as_str()))
                    })
                    .map(|entry| entry.relative_path.as_str()),
                files,
            );
            let mask = BaseMask::new(&parent_index.index, masked);
            (index, document_paths, mask)
        })
        .await
        .map_err(|e| {
            ContextPlusError::Other(format!("lexical delta spawn_blocking failed: {e}"))
        })?;

        let cached = Arc::new(CachedLexicalIndex {
            index,
            document_paths,
            project_cache: Arc::clone(project_cache),
            generation,
            base: Some(LexicalBase { cached: base, mask }),
        });
        let mut guard = ref_index.lexical_search_cache.write().await;
        if ref_index.cache_generation.load(Ordering::Acquire) == generation {
            *guard = Some(Arc::clone(&cached));
            ref_index.lexical_inherited.store(false, Ordering::Release);
        }
        tracing::debug!(
            ref_id = %ref_index.cas_ref_id_hex,
            generation,
            delta_documents = cached.document_paths.len(),
            "LexicalIndex delta replaced over the parent's index"
        );
        Ok(Some(cached))
    }

    async fn ensure_lexical_index_for(
        &self,
        ref_index: &Arc<crate::ref_index::RefIndex>,
        project_cache: &Arc<ProjectCache>,
    ) -> Result<Arc<CachedLexicalIndex>> {
        use std::sync::atomic::Ordering;

        let _update = ref_index.lexical_update.lock().await;
        let generation = ref_index.cache_generation.load(Ordering::Acquire);

        {
            let guard = ref_index.lexical_search_cache.read().await;
            if let Some(cached) = guard.as_ref()
                && cached.generation == generation
                && Arc::ptr_eq(&cached.project_cache, project_cache)
            {
                tracing::debug!(generation, "LexicalIndex cache hit");
                return Ok(Arc::clone(cached));
            }
        }

        let mut guard = ref_index.lexical_search_cache.write().await;
        if let Some(cached) = guard.as_ref()
            && cached.generation == generation
            && Arc::ptr_eq(&cached.project_cache, project_cache)
        {
            tracing::debug!(generation, "LexicalIndex cache hit after write lock");
            return Ok(Arc::clone(cached));
        }

        if let Some(previous) = guard.as_ref().filter(|previous| previous.base.is_none()) {
            let changed: Vec<_> = project_cache
                .file_content
                .iter()
                .filter(|(path, content)| {
                    previous.project_cache.file_content.get(path) != Some(*content)
                })
                .collect();
            let deleted: Vec<_> = previous
                .document_paths
                .iter()
                .enumerate()
                .filter(|(_, path)| {
                    !path.is_empty() && !project_cache.file_content.contains_key(path)
                })
                .map(|(i, _)| i)
                .collect();
            if (changed.len() + deleted.len()) as f64
                <= previous.index.document_count() as f64
                    * crate::tools::semantic_search::FULL_REBUILD_CHANGE_FRACTION
            {
                let mut entry = Arc::clone(previous);
                drop(guard);
                let cached = Arc::make_mut(&mut entry);
                let updates = lexical_updates(
                    &mut cached.document_paths,
                    changed
                        .into_iter()
                        .map(|(path, content)| (path.as_str(), content.as_str())),
                );
                for &i in &deleted {
                    cached.document_paths[i].clear();
                }
                cached.generation = generation;
                cached.project_cache = Arc::clone(project_cache);
                let change = snapshots::Change::Files {
                    changed: updates.len() + deleted.len(),
                    documents: cached.index.document_count(),
                };
                let entry = tokio::task::spawn_blocking(move || {
                    Arc::make_mut(&mut entry)
                        .index
                        .update_documents(updates, &deleted);
                    entry
                })
                .await
                .map_err(|error| ContextPlusError::Other(error.to_string()))?;
                let mut guard = ref_index.lexical_search_cache.write().await;
                if ref_index.cache_generation.load(Ordering::Acquire) == generation {
                    *guard = Some(Arc::clone(&entry));
                    ref_index.lexical_inherited.store(false, Ordering::Release);
                    drop(guard);
                    snapshots::schedule_keywords(&self.state.config, ref_index, change);
                }
                return Ok(entry);
            }
        }

        if guard.is_none() && snapshots::enabled(&self.state.config, ref_index) {
            let root = ref_index.root_dir.clone();
            let config = self.state.config.clone();
            let source = Arc::clone(project_cache);
            let loaded = tokio::task::spawn_blocking(move || {
                snapshots::load_keywords(&root, &config, &source, generation)
            })
            .await
            .ok()
            .flatten();
            if let Some((cached, changed)) = loaded {
                let documents = cached.index.document_count();
                let cached = Arc::new(cached);
                *guard = Some(Arc::clone(&cached));
                ref_index.lexical_inherited.store(false, Ordering::Release);
                drop(guard);
                if changed > 0 {
                    snapshots::schedule_keywords(
                        &self.state.config,
                        ref_index,
                        snapshots::Change::Files { changed, documents },
                    );
                }
                return Ok(cached);
            }
        }

        let reason = match guard.as_ref() {
            None => "empty",
            Some(cached) if cached.generation != generation => "generation changed",
            Some(_) => "project cache replaced",
        };
        tracing::debug!(generation, reason, "Rebuilding LexicalIndex");

        let cache_for_build = Arc::clone(project_cache);
        // An index inherited from the parent ref answers for another tree,
        // so it is never served while this ref rebuilds.
        let stale = guard
            .as_ref()
            .filter(|_| !ref_index.lexical_inherited.load(Ordering::Acquire))
            .cloned();
        if let Some(previous) = &stale
            && ref_index.lexical_rebuilding.load(Ordering::Acquire)
        {
            return Ok(Arc::clone(previous));
        }
        let build = tokio::task::spawn_blocking(move || {
            build_lexical_index(
                cache_for_build
                    .file_entries
                    .iter()
                    .filter(|e| !e.is_directory)
                    .map(|e| e.relative_path.as_str()),
                &cache_for_build.file_content,
            )
        });
        if let Some(previous) = stale {
            ref_index.lexical_rebuilding.store(true, Ordering::Release);
            let flag = Arc::clone(&ref_index.lexical_rebuilding);
            let lock = Arc::clone(&ref_index.lexical_search_cache);
            let source = Arc::clone(project_cache);
            let stale = Arc::clone(&previous);
            let owner = Arc::clone(ref_index);
            let config = self.state.config.clone();
            drop(guard);
            let task = tokio::spawn(async move {
                let _reset = RefreshGuard(flag);
                if let Ok((index, document_paths)) = build.await {
                    let mut guard = lock.write().await;
                    if guard
                        .as_ref()
                        .is_some_and(|current| Arc::ptr_eq(current, &stale))
                    {
                        *guard = Some(Arc::new(CachedLexicalIndex {
                            index,
                            document_paths,
                            project_cache: source,
                            generation,
                            base: None,
                        }));
                        drop(guard);
                        snapshots::schedule_keywords(&config, &owner, snapshots::Change::Full);
                    }
                }
            });
            ref_index.track_background_task(&task);
            return Ok(previous);
        }
        let (index, document_paths) = build.await.map_err(|e| {
            ContextPlusError::Other(format!("lexical index spawn_blocking failed: {e}"))
        })?;

        let cached = Arc::new(CachedLexicalIndex {
            index,
            document_paths,
            project_cache: Arc::clone(project_cache),
            generation,
            base: None,
        });
        *guard = Some(Arc::clone(&cached));
        ref_index.lexical_inherited.store(false, Ordering::Release);
        drop(guard);
        snapshots::schedule_keywords(&self.state.config, ref_index, snapshots::Change::Full);
        tracing::debug!(
            ref_id = %ref_index.cas_ref_id_hex,
            generation,
            documents = cached.document_paths.len(),
            "LexicalIndex replaced after rebuild"
        );
        Ok(cached)
    }

    // --- Tool dispatch ---

    pub async fn dispatch(
        &self,
        name: &str,
        args: serde_json::Map<String, Value>,
    ) -> CallToolResult {
        let ref_id = self.session_ref_id.unwrap_or(self.state.default_ref_id);
        self.state.touch_ref(ref_id);
        let serving = crate::core::process_lifecycle::InflightGuard::new(Arc::clone(
            &self.current_ref().await.active_requests,
        ));
        let result = match self.dispatch_inner(name, args).await {
            Ok(result) => result,
            Err(e) => Self::err_text(format!("Error: {}", e)),
        };
        drop(serving);
        self.state.schedule_memory_budget_enforcement();
        result
    }

    async fn dispatch_inner(
        &self,
        name: &str,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        use std::sync::atomic::Ordering;

        // Reject new calls once we're draining — let in-flight ones finish.
        if self.state.draining.load(Ordering::Acquire) {
            return Ok(Self::err_text(
                "server is shutting down — please retry".to_string(),
            ));
        }

        // Track in-flight dispatches via RAII so the drain watcher knows when
        // it's safe to exit. Decrement happens on drop, including panic paths.
        let _inflight =
            crate::core::process_lifecycle::InflightGuard::new(Arc::clone(&self.state.inflight));

        match name {
            "explore" => self.handle_explore(args).await,
            "outline" => self.handle_outline(args).await,
            "impact" => self.handle_impact(args).await,
            "check" => self.handle_check(args).await,
            "worktrees" => self.handle_worktrees(args).await,
            // Pre-facade names: still dispatch for one release, not listed.
            "get_context_tree" => self.handle_context_tree(args).await,
            "get_file_skeleton" => self.handle_file_skeleton(args).await,
            "get_blast_radius" => self.handle_blast_radius(args).await,
            "semantic_code_search" => self.handle_semantic_code_search(args).await,
            "semantic_identifier_search" => self.handle_semantic_identifier_search(args).await,
            "semantic_navigate" => self.handle_semantic_navigate(args).await,
            "run_static_analysis" => self.handle_static_analysis(args).await,
            "find_dead_code" => self.handle_find_dead_code(args).await,
            "review_pr_diff" => self.handle_review_pr_diff(args).await,
            "detect_dependency_loops" => self.handle_detect_dependency_loops(args).await,
            "check_embedding_quality" => self.handle_check_embedding_quality(args).await,
            "lexical_search" => self.handle_lexical_search(args).await,
            "attach_worktree" => self.handle_attach_worktree(args).await,
            "detach_worktree" => self.handle_detach_worktree(args).await,
            "list_worktrees" => self.handle_list_worktrees(args).await,
            _ => Ok(Self::err_text(format!("Unknown tool: {}", name))),
        }
    }

    // ----- Individual tool handlers -----

    async fn handle_context_tree(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        use crate::tools::context_tree as ct;

        let root = self.resolve_root(&args).await;
        let cache = self.ensure_project_cache().await?;

        // Build entries and analyses in spawn_blocking (tree-sitter parsing is CPU-bound)
        let (ct_entries, ct_analyses) = tokio::task::spawn_blocking(move || {
            let ct_entries: Vec<ct::FileEntry> = cache
                .file_entries
                .iter()
                .map(|e| ct::FileEntry {
                    relative_path: e.relative_path.clone(),
                    is_directory: e.is_directory,
                    depth: e.depth,
                })
                .collect();

            let mut ct_analyses = BTreeMap::new();
            for entry in &cache.file_entries {
                if entry.is_directory {
                    continue;
                }
                if let Some(content) = cache.file_content.get(&entry.relative_path) {
                    let content = Arc::clone(content);
                    let ext = entry.relative_path.rsplit('.').next().unwrap_or("");
                    if let Ok(symbols) = parse_with_tree_sitter(&content, ext) {
                        let header = crate::core::parser::extract_header(&content);
                        let tree_symbols: Vec<ct::TreeSymbol> =
                            symbols.iter().map(code_sym_to_tree_sym).collect();
                        ct_analyses.insert(
                            entry.relative_path.clone(),
                            ct::FileAnalysis {
                                header: if header.is_empty() {
                                    None
                                } else {
                                    Some(header)
                                },
                                symbols: tree_symbols,
                            },
                        );
                    }
                }
            }
            (ct_entries, ct_analyses)
        })
        .await
        .map_err(|e| ContextPlusError::Other(format!("spawn_blocking failed: {e}")))?;

        let options = ct::ContextTreeOptions {
            root_dir: root,
            target_path: Self::get_str(&args, "target_path"),
            depth_limit: Self::get_usize(&args, "depth_limit"),
            include_symbols: Self::get_bool(&args, "include_symbols"),
            max_tokens: Self::get_usize(&args, "max_tokens"),
        };

        let result = ct::get_context_tree(options, &ct_entries, &ct_analyses).await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_file_skeleton(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        use crate::tools::file_skeleton as fs;

        let file_path = Self::get_str(&args, "file_path")
            .or_else(|| Self::get_str(&args, "target_path"))
            .ok_or_else(|| ContextPlusError::Other("file_path is required".into()))?;

        let root = self.resolve_root(&args).await;
        let full_path = root.join(&file_path);

        // Check ProjectCache.file_content first to avoid a disk read on warm cache.
        let cached_content: Option<Arc<String>> = {
            let ref_index = self.current_ref().await;
            let cache_guard = ref_index.project_cache.read().await;
            if let Some(ref cache) = *cache_guard {
                cache.file_content.get(&file_path).map(Arc::clone)
            } else {
                None
            }
        };

        let disk_content: Option<String> = if cached_content.is_none() {
            tokio::fs::read_to_string(&full_path).await.ok()
        } else {
            None
        };

        // Prefer cached Arc content; fall back to freshly read string.
        // `cached_content` is `Option<Arc<String>>` — deref to `&str` via `as_str()`.
        // `disk_content` is `Option<String>` — deref to `&str` via `as_deref()`.
        let content_ref: Option<&str> = cached_content
            .as_deref()
            .map(String::as_str)
            .or(disk_content.as_deref());

        let analysis = content_ref.and_then(|c| {
            let ext = file_path.rsplit('.').next().unwrap_or("");
            let symbols = parse_with_tree_sitter(c, ext).ok()?;
            let header = crate::core::parser::extract_header(c);
            let skel_symbols: Vec<fs::SkeletonSymbol> =
                symbols.iter().map(code_sym_to_skel_sym).collect();
            Some(fs::SkeletonAnalysis {
                header: if header.is_empty() {
                    None
                } else {
                    Some(header)
                },
                symbols: skel_symbols,
                line_count: c.lines().count(),
            })
        });

        let options = fs::SkeletonOptions {
            file_path: file_path.clone(),
            root_dir: root,
        };

        let result = fs::get_file_skeleton(options, analysis.as_ref(), content_ref).await?;
        Ok(Self::ok_text(result))
    }

    /// Resolve which worktree a scope-limited read tool should scan.
    ///
    /// Default: the session's `current_ref()`. When `path` is provided it must
    /// be inside an **already-attached** worktree (registered via `attach_worktree`);
    /// otherwise we return an error result instead of silently scanning a
    /// different tree. That silent fallback is exactly what made `get_blast_radius`
    /// and `find_dead_code` report symbols defined on another branch as
    /// "used nowhere" / "dead". `Err` carries a ready-to-return error result.
    async fn resolve_scan_target(
        &self,
        args: &serde_json::Map<String, Value>,
        tool: &str,
    ) -> std::result::Result<Arc<crate::ref_index::RefIndex>, CallToolResult> {
        let Some(path) = Self::get_str(args, "path") else {
            return Ok(self.current_ref().await);
        };
        let current_ref = self.current_ref().await;
        // Dispatch rewrites routed absolute paths relative to the selected ref.
        let canonical = current_ref
            .root_dir
            .join(&path)
            .canonicalize()
            .map_err(|e| Self::err_text(format!("Cannot canonicalize path {path}: {e}")))?;
        let mut target = None;
        for root in canonical.ancestors() {
            let ref_id = crate::ref_index::RefId::for_canonical_path(root);
            if let Some(found) = self.state.ref_index(ref_id).await {
                target = Some(found);
                break;
            }
        }
        if let Some(ref target) = target {
            tracing::debug!(
                tool,
                path,
                resolved_path = %canonical.display(),
                ref_id = %target.cas_ref_id_hex,
                root = %target.root_dir.display(),
                "resolved scan target"
            );
        }
        target.ok_or_else(|| {
            Self::err_text(format!(
                "Worktree not attached: {}\n\
                 Call `attach_worktree` with this path first, then retry `{tool}` with the same \
                 `path`. Without an attached ref the scan would fall back to a different worktree \
                 and could wrongly report the symbol as unused/dead.",
                canonical.display()
            ))
        })
    }

    async fn handle_blast_radius(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let symbol_name = Self::get_str(&args, "symbol_name")
            .ok_or_else(|| ContextPlusError::Other("symbol_name is required".into()))?;
        let file_context = Self::get_str(&args, "file_context");

        // Optional `path` targets an attached worktree (e.g. reviewing a feature
        // branch from the primary). Defaults to the session's current ref.
        let ref_index = match self.resolve_scan_target(&args, "get_blast_radius").await {
            Ok(r) => r,
            Err(err) => return Ok(err),
        };

        let scanned_root = ref_index.root_dir.display().to_string();
        let cache = self.ensure_project_cache_for(&ref_index).await?;

        // find_symbol_usages scans all file content — CPU-bound, run in blocking thread pool.
        let formatted = tokio::task::spawn_blocking(move || {
            let files_scanned = cache.file_content.len();
            let result = crate::tools::blast_radius::find_symbol_usages(
                &symbol_name,
                file_context.as_deref(),
                &cache.file_content,
            );
            crate::tools::blast_radius::format_blast_radius(
                &symbol_name,
                &result,
                &scanned_root,
                files_scanned,
            )
        })
        .await
        .map_err(|e| ContextPlusError::Other(format!("blast_radius spawn_blocking failed: {e}")))?;

        Ok(Self::ok_text(formatted))
    }

    async fn handle_semantic_code_search(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        self.ensure_tracker_started().await;
        let query = Self::get_str(&args, "query")
            .ok_or_else(|| ContextPlusError::Other("query is required".into()))?;
        let root = self.resolve_root(&args).await;

        let options = crate::tools::semantic_search::SemanticSearchOptions {
            root_dir: root.clone(),
            query,
            top_k: Self::get_usize(&args, "top_k"),
            semantic_weight: Self::get_f64(&args, "semantic_weight"),
            keyword_weight: Self::get_f64(&args, "keyword_weight"),
            min_semantic_score: Self::get_f64(&args, "min_semantic_score"),
            min_keyword_score: Self::get_f64(&args, "min_keyword_score"),
            min_combined_score: Self::get_f64(&args, "min_combined_score"),
            require_keyword_match: Self::get_bool(&args, "require_keyword_match"),
            require_semantic_match: Self::get_bool(&args, "require_semantic_match"),
            include_globs: Self::get_string_array(&args, "include_globs"),
            exclude_globs: Self::get_string_array(&args, "exclude_globs"),
            recency_window_days: Self::get_u32(&args, "recency_window_days"),
            scope: match Self::get_str(&args, "scope").as_deref() {
                None | Some("all") => None,
                Some("code") => Some(crate::tools::semantic_search::SearchScope::Code),
                Some("docs") => Some(crate::tools::semantic_search::SearchScope::Docs),
                Some(_) => {
                    return Err(ContextPlusError::Other(
                        "scope must be code, docs, or all".into(),
                    ));
                }
            },
        };

        let embedder = OllamaEmbedder(self.state.ollama.clone());
        let walker = crate::server_adapters::RefWalkerIndexer {
            ref_index: self.current_ref().await,
            walker: CachedWalkerIndexer {
                config: self.state.config.clone(),
                ollama: self.state.ollama.clone(),
                state: self.state.clone(),
            },
        };

        // Pass the generation counter when the tracker is active so
        // `semantic_code_search` can skip the walk on a generation hit.
        // When the tracker is Off the counter stays at 0 and the
        // fingerprint-based fallback is used instead.
        walker.expire_stale_fork(&root).await;
        let ref_index = self.current_ref().await;
        let cache_gen = if self.state.config.embed_tracker_mode != crate::config::TrackerMode::Off {
            Some(&ref_index.cache_generation)
        } else {
            None
        };
        let result = crate::tools::semantic_search::semantic_code_search_owned(
            options,
            &embedder,
            Arc::new(walker),
            Some(Arc::clone(&ref_index.search_index_cache)),
            cache_gen.cloned(),
        )
        .await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_semantic_identifier_search(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        self.ensure_tracker_started().await;
        use crate::tools::semantic_identifiers::*;

        let query = Self::get_str(&args, "query")
            .ok_or_else(|| ContextPlusError::Other("query is required".into()))?;
        let root = self.resolve_root(&args).await;

        let cache = self.ensure_project_cache().await?;

        // Use cached identifier index (TTL=300s, rebuilds if file count changes)
        let idx = self.ensure_identifier_index(&cache).await?;

        if idx.docs.is_empty() {
            return Ok(Self::ok_text(
                "No supported identifiers found for semantic identifier search.".to_string(),
            ));
        }

        let options = SemanticIdentifierSearchOptions {
            root_dir: root.clone(),
            query,
            top_k: Self::get_usize(&args, "top_k"),
            top_calls_per_identifier: Self::get_usize(&args, "top_calls_per_identifier"),
            semantic_weight: Self::get_f64(&args, "semantic_weight"),
            keyword_weight: Self::get_f64(&args, "keyword_weight"),
            include_kinds: Self::get_string_array(&args, "include_kinds"),
        };

        let ref_index = self.current_ref().await;
        // Symlinked roots (macOS /var -> /private/var) must compare in canonical form.
        let scope = options
            .root_dir
            .canonicalize()
            .unwrap_or_else(|_| options.root_dir.clone());
        let candidates = (scope != ref_index.canonical_root).then(|| {
            idx.docs
                .iter()
                .enumerate()
                .filter_map(|(i, doc)| {
                    ref_index
                        .canonical_root
                        .join(&doc.path)
                        .starts_with(&scope)
                        .then_some(i)
                })
                .collect::<Vec<_>>()
        });

        let result = semantic_identifier_search(
            options,
            &OllamaEmbedder(self.state.ollama.clone()),
            &idx.docs,
            &idx.vectors,
            idx.dims,
            &cache.file_content,
            candidates.as_deref(),
        )
        .await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_semantic_navigate(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        self.ensure_tracker_started().await;
        let root = self.resolve_root(&args).await;

        let options = crate::tools::semantic_navigate::SemanticNavigateOptions {
            query: Self::get_str(&args, "query"),
            max_tokens: Self::get_usize(&args, "max_tokens"),
            root_dir: root.to_string_lossy().into(),
            max_depth: Self::get_usize(&args, "max_depth"),
            max_clusters: Self::get_usize(&args, "max_clusters"),
            min_clusters: Self::get_usize(&args, "min_clusters"),
            mode: Self::get_str(&args, "mode"),
        };

        let ref_index = self.current_ref().await;
        let indexer = crate::server_adapters::RefWalkerIndexer {
            ref_index: self.current_ref().await,
            walker: CachedWalkerIndexer {
                config: self.state.config.clone(),
                ollama: self.state.ollama.clone(),
                state: self.state.clone(),
            },
        };
        let result = crate::tools::semantic_navigate::semantic_navigate(
            options,
            &self.state.ollama,
            &self.state.config,
            &ref_index.embedding_cache,
            &ref_index.root_dir,
            Some(&indexer),
        )
        .await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_static_analysis(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let target_path = Self::get_str(&args, "target_path");
        let (root_dir, target_path) = self.route_static_analysis_target(&args, target_path).await;

        let options = crate::tools::static_analysis::StaticAnalysisOptions {
            executable_path: None,
            root_dir,
            target_path,
        };

        let result = crate::tools::static_analysis::run_static_analysis(options).await?;
        Ok(Self::ok_text(result))
    }

    /// Register an out-of-server-root worktree as a `RefIndex` in the registry,
    /// chaining its CAS manifest to the primary ref's so embedding lookups
    /// inherit the baseline. Mirrors `daemon::serve_connection`'s attach flow
    /// (CoW memory overlay, `fork_from`, per-ref warmup) — the difference is
    /// the trigger: an in-band MCP tool call instead of a network handshake.
    ///
    /// Idempotent: re-attaching the same canonical path returns the existing
    /// ref without bumping `session_count`, mirroring the daemon's behavior
    /// when a duplicate session connects (though there it bumps; here we
    /// keep the pin to "exactly one logical attach" so a single `detach_worktree`
    /// undoes it).
    ///
    /// CoW chain:
    /// 1. `RefIndex::new_with_head` registers the ref with the primary as parent.
    /// 2. `fork_from(mcp_data, model, primary_ref)` writes the parent pointer
    ///    in the CAS directory so chunk hashes chain through the primary's
    ///    manifest — this is what lets the worktree reuse the base cache.
    /// 3. `spawn_ref_warmup` triggers shallow warmup (per config) so the next
    ///    semantic call hits a warm HNSW built from CAS-imported vectors.
    async fn handle_attach_worktree(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let path = Self::get_str(&args, "path")
            .ok_or_else(|| ContextPlusError::Other("path is required".into()))?;
        let raw = PathBuf::from(&path);
        let canonical = match raw.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return Ok(Self::err_text(format!(
                    "Cannot canonicalize {}: {}",
                    path, e
                )));
            }
        };
        if !canonical.is_dir() {
            return Ok(Self::err_text(format!(
                "Not a directory: {}",
                canonical.display()
            )));
        }

        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical);

        // Fast path: already attached. Don't double-bump session_count.
        {
            let guard = self.state.refs.read().await;
            if let Some(existing) = guard.get(&ref_id) {
                return Ok(Self::ok_text(format!(
                    "Worktree already attached: {} (ref_id={}, sessions={})",
                    existing.canonical_root.display(),
                    ref_id.0,
                    existing
                        .session_count
                        .load(std::sync::atomic::Ordering::Acquire)
                )));
            }
        }

        let parent_ref_id = if ref_id != self.state.default_ref_id {
            Some(self.state.default_ref_id)
        } else {
            None
        };

        // Optional: read HEAD from the worktree's git index. Tolerates non-git
        // dirs (returns None → stored as None on the ref).
        let head_sha = crate::core::head_watcher::resolve_head_sha(&canonical);

        // Build the ref.
        // `attach_ref` is idempotent under concurrent calls: racing first
        // attaches may each build a candidate, but only one is inserted.
        let raw_for_closure = raw.clone();
        let canonical_for_closure = canonical.clone();
        let head_sha_for_closure = head_sha.clone();
        let ref_arc = self
            .state
            .attach_ref(ref_id, move || {
                std::sync::Arc::new(crate::ref_index::RefIndex::new_with_head(
                    raw_for_closure,
                    canonical_for_closure,
                    parent_ref_id,
                    head_sha_for_closure.unwrap_or_default(),
                ))
            })
            .await;

        // CAS init: chain the parent manifest so chunk lookups inherit the
        // baseline. Non-fatal: if disk is read-only or the CAS dir is unusable,
        // log and continue — the ref still functions for non-semantic tools
        // like `run_static_analysis`.
        {
            let mcp_data = self.state.root_dir.join(".mcp_data");
            let model = self.state.config.document_cache_identity();
            let parent_ref_opt = match parent_ref_id {
                Some(pid) => self.state.ref_index(pid).await,
                None => None,
            };
            if let Err(e) = ref_arc.fork_from(&mcp_data, &model, parent_ref_opt.as_deref()) {
                tracing::warn!(
                    ref_id = ref_id.0,
                    "attach_worktree CAS fork_from failed (non-fatal): {e}"
                );
            }
        }

        // Per-ref warmup (idempotent). Off / Shallow / Full per RefWarmupMode.
        self.spawn_ref_warmup(ref_id);

        // U11: Eager mode mirrors the daemon's startup behaviour for the
        // default ref — attached worktrees should also pick up live edits
        // without waiting for a tool call. Lazy mode lets the first tool
        // call start it on demand (via `ensure_tracker_started` through the
        // session-scoped server clone).
        if self.state.config.embed_tracker_mode == TrackerMode::Eager {
            self.ensure_tracker_started_for(ref_id).await;
        }

        let head_display = ref_arc
            .head_sha
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(|s| s[..s.len().min(8)].to_string())
            .unwrap_or_else(|| "<unknown>".to_string());

        Ok(Self::ok_text(format!(
            "Worktree attached: {}\n  ref_id  = {}\n  parent  = {}\n  head    = {}\n  sessions= {}\n  warmup  = {}",
            ref_arc.canonical_root.display(),
            ref_id.0,
            parent_ref_id
                .map(|p| p.0.to_string())
                .unwrap_or_else(|| "<none (this IS the primary)>".to_string()),
            head_display,
            ref_arc
                .session_count
                .load(std::sync::atomic::Ordering::Acquire),
            self.state.config.ref_warmup_mode,
        )))
    }

    /// Detach a previously-attached worktree. Decrements session_count; once it
    /// reaches zero the ref enters the TTL eviction queue (`cache_ttl_secs` from
    /// config). Refuses to detach the primary ref.
    async fn handle_detach_worktree(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let path = Self::get_str(&args, "path")
            .ok_or_else(|| ContextPlusError::Other("path is required".into()))?;
        let raw = PathBuf::from(&path);
        let canonical = match raw.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return Ok(Self::err_text(format!(
                    "Cannot canonicalize {}: {}",
                    path, e
                )));
            }
        };
        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical);

        if ref_id == self.state.default_ref_id {
            return Ok(Self::err_text(
                "Cannot detach the primary ref; the daemon owns its lifetime.".into(),
            ));
        }
        {
            let guard = self.state.refs.read().await;
            if !guard.contains_key(&ref_id) {
                return Ok(Self::err_text(format!(
                    "Not attached: {} (ref_id={})",
                    canonical.display(),
                    ref_id.0
                )));
            }
        }

        let ttl = self.state.config.cache_ttl_secs;
        self.state
            .detach_ref(ref_id, std::time::Duration::from_secs(ttl))
            .await;

        Ok(Self::ok_text(format!(
            "Worktree detach scheduled: {} (ref_id={}, ttl={}s)",
            canonical.display(),
            ref_id.0,
            ttl
        )))
    }

    /// List every ref currently in the registry (primary + attached worktrees).
    async fn handle_list_worktrees(
        &self,
        _args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let guard = self.state.refs.read().await;
        let mut rows: Vec<(u64, PathBuf, bool, usize, Option<String>)> = guard
            .iter()
            .map(|(id, r)| {
                (
                    id.0,
                    r.canonical_root.clone(),
                    *id == self.state.default_ref_id,
                    r.session_count.load(std::sync::atomic::Ordering::Acquire),
                    r.head_sha.clone(),
                )
            })
            .collect();
        drop(guard);
        // Primary first, then alphabetical by path — predictable output.
        rows.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.1.cmp(&b.1)));

        if rows.is_empty() {
            return Ok(Self::ok_text("No refs registered.".into()));
        }
        let mut out = format!(
            "Daemon search config: OLLAMA_EMBED_MODEL={}, OLLAMA_CHAT_MODEL={}, OLLAMA_HOST={}\nConfig source: {}\nRegistered refs:\n",
            self.state.config.ollama_embed_model,
            self.state.config.ollama_chat_model,
            self.state.config.ollama_host,
            self.state
                .config
                .config_source
                .as_deref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "process environment".into())
        );
        for (id, path, is_primary, sessions, head) in rows {
            let tag = if is_primary { " [primary]" } else { "" };
            let head_str = head
                .as_deref()
                .filter(|s| !s.is_empty())
                .map(|s| format!(" head={}", &s[..s.len().min(8)]))
                .unwrap_or_default();
            out.push_str(&format!(
                "- {}{} (ref_id={}, sessions={}{})\n",
                path.display(),
                tag,
                id,
                sessions,
                head_str
            ));
        }
        Ok(Self::ok_text(out))
    }

    /// If `target_path` is absolute and lives under a registered ref's
    /// canonical_root, route the linter to run from that ref's root with a
    /// relative target. This is what lets a session attached to the primary
    /// workspace lint a sibling worktree the daemon has already registered
    /// (e.g. `~/worktrees/<feature>`), without needing a per-worktree session.
    ///
    /// Falls back to `resolve_root(&args)` and the original `target_path`
    /// otherwise. The longest (most specific) matching ref wins, so nested
    /// worktrees route to the innermost one.
    async fn route_static_analysis_target(
        &self,
        args: &serde_json::Map<String, Value>,
        target_path: Option<String>,
    ) -> (PathBuf, Option<String>) {
        if let Some(target) = target_path.as_deref()
            && std::path::Path::new(target).is_absolute()
        {
            let target_pb = PathBuf::from(target);
            let canonical_target = target_pb
                .canonicalize()
                .unwrap_or_else(|_| target_pb.clone());

            let refs = self.state.refs.read().await;
            let mut best: Option<(PathBuf, PathBuf)> = None;
            for ref_idx in refs.values() {
                if canonical_target.starts_with(&ref_idx.canonical_root) {
                    let prev_len = best.as_ref().map(|(_, r)| r.as_os_str().len()).unwrap_or(0);
                    if ref_idx.canonical_root.as_os_str().len() >= prev_len {
                        best = Some((ref_idx.root_dir.clone(), ref_idx.canonical_root.clone()));
                    }
                }
            }
            drop(refs);

            if let Some((ref_root, canonical_root)) = best {
                let rel = canonical_target
                    .strip_prefix(&canonical_root)
                    .ok()
                    .map(|p| p.to_string_lossy().into_owned())
                    .filter(|s| !s.is_empty());
                return (ref_root, rel);
            }
        }
        (self.resolve_root(args).await, target_path)
    }

    // --- Facade: the six listed tools, mapped onto the handlers above ---

    fn move_arg(args: &mut serde_json::Map<String, Value>, from: &str, to: &str) {
        if let Some(v) = args.remove(from) {
            args.insert(to.to_string(), v);
        }
    }

    fn arg_or(args: &mut serde_json::Map<String, Value>, key: &str, default: &str) -> String {
        let v = Self::get_str(args, key).unwrap_or_else(|| default.to_string());
        args.remove(key);
        v
    }

    async fn handle_explore(
        &self,
        mut args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let kind = Self::arg_or(&mut args, "kind", "files");
        let matching = Self::arg_or(&mut args, "match", "meaning");
        Self::move_arg(&mut args, "path", "rootDir");
        match (kind.as_str(), matching.as_str()) {
            ("identifiers", "keywords") => {
                // Identifier search has no separate lexical path; rank by
                // keyword coverage only so `keywords` means what it says.
                args.insert("semantic_weight".into(), serde_json::json!(0.0));
                args.insert("keyword_weight".into(), serde_json::json!(1.0));
                self.handle_semantic_identifier_search(args).await
            }
            ("identifiers", _) => self.handle_semantic_identifier_search(args).await,
            ("clusters", _) => self.handle_semantic_navigate(args).await,
            (_, "keywords") => self.handle_lexical_search(args).await,
            _ => self.handle_semantic_code_search(args).await,
        }
    }

    async fn handle_outline(
        &self,
        mut args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let path = Self::get_str(&args, "path")
            .ok_or_else(|| ContextPlusError::Other("path is required".into()))?;
        let abs = if std::path::Path::new(&path).is_absolute() {
            PathBuf::from(&path)
        } else {
            self.current_ref().await.root_dir.join(&path)
        };
        if abs.is_dir() {
            Self::move_arg(&mut args, "path", "target_path");
            Self::move_arg(&mut args, "depth", "depth_limit");
            self.handle_context_tree(args).await
        } else {
            Self::move_arg(&mut args, "path", "file_path");
            args.remove("depth");
            args.remove("max_tokens");
            self.handle_file_skeleton(args).await
        }
    }

    async fn handle_impact(
        &self,
        mut args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        if Self::get_str(&args, "diff").is_some() {
            args.remove("what");
            return self.handle_review_pr_diff(args).await;
        }
        match Self::arg_or(&mut args, "what", "symbol").as_str() {
            "cycles" => self.handle_detect_dependency_loops(args).await,
            "dead" => self.handle_find_dead_code(args).await,
            _ => {
                if Self::get_str(&args, "symbol").is_none() {
                    return Err(ContextPlusError::Other(
                        "symbol or diff is required (or set what to cycles or dead)".into(),
                    ));
                }
                Self::move_arg(&mut args, "symbol", "symbol_name");
                Self::move_arg(&mut args, "file", "file_context");
                self.handle_blast_radius(args).await
            }
        }
    }

    async fn handle_check(
        &self,
        mut args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        match Self::arg_or(&mut args, "what", "lint").as_str() {
            "embeddings" => self.handle_check_embedding_quality(args).await,
            _ => {
                Self::move_arg(&mut args, "path", "target_path");
                self.handle_static_analysis(args).await
            }
        }
    }

    async fn handle_worktrees(
        &self,
        mut args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        match Self::arg_or(&mut args, "action", "list").as_str() {
            "attach" => self.handle_attach_worktree(args).await,
            "detach" => self.handle_detach_worktree(args).await,
            _ => self.handle_list_worktrees(args).await,
        }
    }

    async fn resolve_root(&self, args: &serde_json::Map<String, Value>) -> PathBuf {
        let ref_index = self.current_ref().await;
        if let Some(requested) = Self::get_str(args, "rootDir") {
            let requested_path = ref_index.root_dir.join(&requested);
            // Use pre-canonicalized root (computed once at construction, not per-request).
            if let Ok(canonical_requested) = requested_path.canonicalize()
                && canonical_requested.starts_with(&ref_index.canonical_root)
            {
                return canonical_requested;
            }
            tracing::warn!(
                requested = %requested,
                root = %ref_index.root_dir.display(),
                "Caller-provided rootDir is outside the server root; ignoring"
            );
        }
        ref_index.root_dir.clone()
    }

    // ----- find_dead_code -----

    async fn handle_find_dead_code(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        use crate::tools::dead_code_find::{
            DeadCodeOptions, find_dead_symbols, format_dead_symbols,
        };

        // Optional `path` targets an attached worktree. Defaults to the session's
        // current ref. Routing only via `current_ref()` would let a symbol used
        // on another branch be reported as dead from the wrong tree.
        let ref_index = match self.resolve_scan_target(&args, "find_dead_code").await {
            Ok(r) => r,
            Err(err) => return Ok(err),
        };
        let scanned_root = ref_index.root_dir.display().to_string();
        let cache = self.ensure_project_cache_for(&ref_index).await?;

        let ignore_kinds: Option<std::collections::HashSet<String>> =
            Self::get_string_array(&args, "ignore_kinds")
                .map(|v| v.into_iter().map(|s| s.to_lowercase()).collect());
        let ignore_names: Option<std::collections::HashSet<String>> =
            Self::get_string_array(&args, "ignore_names")
                .map(|v| v.into_iter().map(|s| s.to_lowercase()).collect());
        // Treat 0 as "use default" so callers cannot accidentally request a
        // truncated-to-zero result set that looks like "no dead code found".
        let max_results = Self::get_usize(&args, "max_results").filter(|&n| n > 0);

        let formatted = tokio::task::spawn_blocking(move || {
            let files_scanned = cache.file_content.len();
            let symbols_by_file: HashMap<PathBuf, Vec<crate::core::parser::CodeSymbol>> =
                build_symbols_by_file(&cache, |rel| PathBuf::from(rel));
            let mut tokens_by_file: HashMap<PathBuf, std::collections::HashSet<String>> =
                HashMap::new();

            for (rel_path, content) in &cache.file_content {
                let tokens: std::collections::HashSet<String> = content
                    .as_str()
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .filter(|t| !t.is_empty())
                    .map(|t| t.to_string())
                    .collect();
                tokens_by_file.insert(PathBuf::from(rel_path), tokens);
            }

            let mut opts = DeadCodeOptions::default();
            if let Some(kinds) = ignore_kinds {
                opts.ignore_kinds = kinds;
            }
            if let Some(names) = ignore_names {
                opts.ignore_names = names;
            }
            if let Some(max) = max_results {
                opts.max_results = max;
            }

            let dead = find_dead_symbols(&symbols_by_file, &tokens_by_file, &opts);
            format_dead_symbols(&dead, &scanned_root, files_scanned)
        })
        .await
        .map_err(|e| {
            ContextPlusError::Other(format!("find_dead_code spawn_blocking failed: {e}"))
        })?;

        Ok(Self::ok_text(formatted))
    }

    // ----- review_pr_diff -----

    async fn handle_review_pr_diff(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        use crate::core::dependent_expand::{ExpansionOptions, build_reverse_graph};
        use crate::tools::pr_review::{analyze, format_report};

        let diff = Self::get_str(&args, "diff")
            .ok_or_else(|| ContextPlusError::Other("diff is required".into()))?;
        // Clamp caller-supplied bounds so a missing/large value cannot
        // turn the BFS into an effectively unbounded walk.
        const MAX_HOPS_CAP: usize = 10;
        const MAX_FILES_CAP: usize = 2000;
        let max_hops = Self::get_usize(&args, "max_hops")
            .unwrap_or(2)
            .min(MAX_HOPS_CAP);
        let max_files = Self::get_usize(&args, "max_files")
            .unwrap_or(500)
            .min(MAX_FILES_CAP);

        let cache = self.ensure_project_cache().await?;
        let root = self.current_ref().await.root_dir.clone();

        let formatted = tokio::task::spawn_blocking(move || {
            let symbols_by_file: HashMap<String, Vec<crate::core::parser::CodeSymbol>> =
                build_symbols_by_file(&cache, |rel| rel.to_string());
            let all_abs_paths: Vec<PathBuf> = cache
                .file_content
                .keys()
                .map(|rel_path| root.join(rel_path))
                .collect();

            // build_reverse_graph requires absolute paths (it stat()s each file
            // through extract_imports), but analyze receives diff-derived seeds
            // which are RELATIVE (parsed from `+++ b/<rel>`). Re-key the graph
            // to relative paths so the BFS lookup matches; without this the
            // dependents half of every report is silently empty (RV3-001).
            let reverse_graph_abs = build_reverse_graph(&all_abs_paths);
            let reverse_graph: HashMap<PathBuf, std::collections::HashSet<PathBuf>> =
                reverse_graph_abs
                    .into_iter()
                    .filter_map(|(imported, importers)| {
                        let imported_rel = imported.strip_prefix(&root).ok()?.to_path_buf();
                        let importers_rel: std::collections::HashSet<PathBuf> = importers
                            .into_iter()
                            .filter_map(|p| {
                                p.strip_prefix(&root).ok().map(std::path::Path::to_path_buf)
                            })
                            .collect();
                        Some((imported_rel, importers_rel))
                    })
                    .collect();
            let expansion_opts = ExpansionOptions {
                max_hops,
                max_files,
            };

            let report = analyze(&diff, &symbols_by_file, &reverse_graph, expansion_opts);
            format_report(&report)
        })
        .await
        .map_err(|e| {
            ContextPlusError::Other(format!("review_pr_diff spawn_blocking failed: {e}"))
        })?;

        Ok(Self::ok_text(formatted))
    }

    // ----- detect_dependency_loops -----

    async fn handle_detect_dependency_loops(
        &self,
        _args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        use crate::core::dependent_expand::build_reverse_graph;
        use crate::tools::dependency_loop_detect::{find_cycles, format_cycles};

        let cache = self.ensure_project_cache().await?;
        let root = self.current_ref().await.root_dir.clone();

        let formatted = tokio::task::spawn_blocking(move || {
            let all_abs_paths: Vec<PathBuf> = cache
                .file_content
                .keys()
                .map(|rel| root.join(rel))
                .collect();

            // build_reverse_graph gives reverse edges (imported -> {importers}).
            // Invert to forward graph (importer -> {imported}) for cycle detection.
            let reverse = build_reverse_graph(&all_abs_paths);
            let mut forward: HashMap<PathBuf, std::collections::HashSet<PathBuf>> = HashMap::new();
            for (imported, importers) in &reverse {
                for importer in importers {
                    forward
                        .entry(importer.clone())
                        .or_default()
                        .insert(imported.clone());
                }
            }

            let cycles = find_cycles(&forward);
            format_cycles(&cycles)
        })
        .await
        .map_err(|e| {
            ContextPlusError::Other(format!(
                "detect_dependency_loops spawn_blocking failed: {e}"
            ))
        })?;

        Ok(Self::ok_text(formatted))
    }

    // ----- check_embedding_quality -----

    async fn handle_check_embedding_quality(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        use crate::tools::embedding_quality_check::{check_embeddings, format_report};

        // Reject explicit expected_dim=0 — without this filter, callers
        // passing 0 would have every non-empty vector flagged as a
        // dimension mismatch, producing a misleading "everything is broken"
        // report.
        let requested_dim = Self::get_usize(&args, "expected_dim").filter(|&d| d > 0);

        let vectors: Vec<(PathBuf, Vec<f32>)> = {
            let ref_index = self.current_ref().await;
            let guard = ref_index.embedding_cache.read().await;
            guard
                .iter()
                .map(|(path, entry)| (PathBuf::from(path), entry.vector.clone()))
                .collect()
        };

        // Pick a dim: caller override > first non-zero-length cached vector >
        // bail out. Distinguishing empty-cache from corrupt-cache (every
        // entry is zero-length) matters: the latter is a real diagnostic
        // signal that should not be silently swallowed.
        let inferred_dim = vectors.iter().map(|(_, v)| v.len()).find(|&n| n > 0);
        let expected_dim = match requested_dim.or(inferred_dim) {
            Some(d) if d > 0 => d,
            _ => {
                let msg = if vectors.is_empty() {
                    "Embedding quality report: 0 vector(s) cached and no `expected_dim` provided — nothing to check.".to_string()
                } else {
                    format!(
                        "Embedding quality report: cache appears corrupt — all {} vector(s) have zero length and no `expected_dim` was provided.",
                        vectors.len()
                    )
                };
                return Ok(Self::ok_text(msg));
            }
        };

        let formatted = tokio::task::spawn_blocking(move || {
            let report = check_embeddings(&vectors, expected_dim);
            format_report(&report)
        })
        .await
        .map_err(|e| {
            ContextPlusError::Other(format!(
                "check_embedding_quality spawn_blocking failed: {e}"
            ))
        })?;

        Ok(Self::ok_text(formatted))
    }

    // ----- lexical_search -----

    async fn handle_lexical_search(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        self.ensure_tracker_started().await;
        let query = Self::get_str(&args, "query")
            .ok_or_else(|| ContextPlusError::Other("query is required".into()))?;
        // top_k=0 would silently return zero hits (LexicalIndex::search short-
        // circuits on n=0) and the user would see "No matches" — masking the
        // bad input. Treat 0 as "use default" the same way find_dead_code does.
        let top_k = Self::get_usize(&args, "top_k")
            .filter(|&n| n > 0)
            .unwrap_or(10);

        let cache = self.ensure_project_cache().await?;
        let cached = self.ensure_lexical_index(&cache).await?;

        let formatted = tokio::task::spawn_blocking(move || {
            if cached.is_empty() {
                return "No files indexed. Ensure the project cache is populated.".to_string();
            }

            let hits = cached.search(&query, top_k);

            if hits.is_empty() {
                return format!("No lexical matches found for: {query}");
            }

            let mut lines = vec![format!(
                "Lexical search: {} result(s) for \"{query}\"",
                hits.len()
            )];
            lines.push(String::new());
            for (rank, (path, score)) in hits.iter().enumerate() {
                lines.push(format!("{}. {} (score: {:.3})", rank + 1, path, score));
            }
            lines.join("\n")
        })
        .await
        .map_err(|e| {
            ContextPlusError::Other(format!("lexical_search spawn_blocking failed: {e}"))
        })?;

        Ok(Self::ok_text(formatted))
    }
}

// --- ServerHandler implementation ---

impl ServerHandler for ContextPlusServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_server_info(Implementation::new(
            "contextplus",
            env!("CARGO_PKG_VERSION"),
        ))
        .with_instructions(
            "Code intelligence for this repository, five tools: explore (find code by \
             what it does; start here), outline (a file's signatures or a directory's \
             tree; call before reading a file), impact (who uses a symbol; call before \
             changing one; give it a diff to rank a whole change), check (the \
             project's linters, or the search index), worktrees (list, attach, detach). Calls run against \
             the git worktree your process is in, or the one an absolute path points \
             into, attached on first use; relative paths resolve from that worktree's \
             root.",
        )
    }

    fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = std::result::Result<ListResourcesResult, rmcp::ErrorData>>
    + Send
    + '_ {
        let resource = RawResource::new(INSTRUCTIONS_RESOURCE_URI, "contextplus_instructions")
            .with_description("Context+ usage instructions and best practices")
            .with_mime_type("text/markdown")
            .no_annotation();
        std::future::ready(Ok(ListResourcesResult {
            resources: vec![resource],
            meta: None,
            next_cursor: None,
        }))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<ReadResourceResult, rmcp::ErrorData> {
        if request.uri == INSTRUCTIONS_RESOURCE_URI {
            let text = self.get_instructions().await;
            Ok(ReadResourceResult::new(vec![
                ResourceContents::TextResourceContents {
                    uri: INSTRUCTIONS_RESOURCE_URI.to_string(),
                    mime_type: Some("text/markdown".to_string()),
                    text,
                    meta: None,
                },
            ]))
        } else {
            Err(rmcp::ErrorData::invalid_params(
                format!("Unknown resource URI: {}", request.uri),
                None,
            ))
        }
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = std::result::Result<ListToolsResult, rmcp::ErrorData>>
    + Send
    + '_ {
        // tool_definitions() returns &'static [Tool] — built once via LazyLock, zero allocation.
        std::future::ready(Ok(ListToolsResult {
            tools: tool_definitions().to_vec(),
            meta: None,
            next_cursor: None,
        }))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<CallToolResult, rmcp::ErrorData> {
        // Reset idle timer on every tool call.
        if let Some(monitor) = self.state.idle_monitor.read().await.as_ref() {
            monitor.touch();
        }
        let name = request.name.to_string();
        let mut args = request.arguments.unwrap_or_default();
        // Over stdio this process is the host's child, so its parent's cwd is
        // the agent's; the daemon bridge injects the same argument itself.
        if self.session_ref_id.is_none()
            && !args.contains_key(crate::core::client_cwd::CWD_ARG)
            && let Some(cwd) = crate::core::client_cwd::parent_process_cwd()
        {
            args.insert(
                crate::core::client_cwd::CWD_ARG.to_string(),
                Value::String(cwd.to_string_lossy().into_owned()),
            );
        }

        // Tool-call entry log — pairs with an exit log below. Lets daemon-log
        // operators correlate "Transport closed" / "completed without a
        // result" client-side errors against a specific tool dispatch and its
        // duration. Includes session_ref_id so multi-worktree races are
        // distinguishable.
        let _call_started = std::time::Instant::now();
        tracing::debug!(
            tool = %name,
            session_ref_id = self.session_ref_id.map(|r| r.0),
            "call_tool: entry"
        );

        let result = Ok(self.call_tool_routed(&name, args).await);

        let elapsed_ms = _call_started.elapsed().as_millis() as u64;
        let content_count = result
            .as_ref()
            .ok()
            .map(|r: &CallToolResult| r.content.len());
        tracing::info!(
            tool = %name,
            session_ref_id = self.session_ref_id.map(|r| r.0),
            elapsed_ms,
            content_count = ?content_count,
            "call_tool: exit"
        );
        result
    }
}

impl ContextPlusServer {
    /// Dispatch one tool call through the path-translation boundary. Entry
    /// point shared by the MCP `call_tool` handler and tests.
    pub async fn call_tool_routed(
        &self,
        name: &str,
        args: serde_json::Map<String, Value>,
    ) -> CallToolResult {
        // A path inside another worktree of this repo runs the whole call as
        // that worktree's session; routed paths come back relative, so the
        // re-entry below does not route again.
        let (routed, args) =
            crate::transport::dispatch::route_to_worktree_ref(self, name, args).await;
        if let Some(ref_id) = routed {
            let result = Box::pin(self.with_session(ref_id).call_tool_routed(name, args)).await;
            return self.prepend_session_config_warning(result);
        }

        // Route through U5's path-translation boundary.
        //
        // U9: `caller_root` is now resolved per-session:
        //   • If `session_ref_id` is Some(id), look up that ref's `root_dir`
        //     in the registry. This is set by `daemon::serve_connection` after
        //     `register_session` completes, so worktree-scoped connections get
        //     their own caller root automatically.
        //   • If `session_ref_id` is None (stdio mode / no handshake), fall
        //     back to `default_ref` — identical to pre-U9 behaviour.
        //   • If the id is Some but not found in the registry (registry
        //     tampered), warn and fall back to `default_ref`.
        //
        // The dispatch layer passes `foreign_roots = &[]` so output rewriting
        // is still identity in single-ref mode. Cross-ref leakage protection
        // (listing other refs' roots as `foreign_roots`) is reserved for U10+.
        let caller_root_opt =
            crate::transport::dispatch::caller_root_for_session(&self.state, self.session_ref_id)
                .await;
        let result = match caller_root_opt {
            Some(caller_root) => {
                crate::transport::dispatch::dispatch_with_translation(
                    self,
                    name,
                    args,
                    &caller_root,
                    // U9 + U14 wiring: `self.session_ref_id` is set per-connection by
                    // `daemon::serve_connection` after `register_session`. Passing it
                    // to `dispatch_with_translation` lets `foreign_roots_for_session`
                    // exclude the caller's own ref from the foreign-roots list and
                    // include only OTHER attached refs — activating cross-ref
                    // leakage protection for multi-ref daemons.
                    self.session_ref_id,
                )
                .await
            }
            None => {
                // Registry tampered with externally — fall back to direct
                // dispatch so we don't lose the request entirely.
                tracing::warn!(
                    "caller_root unavailable; bypassing path translation for tool {name}"
                );
                self.dispatch(name, args).await
            }
        };
        self.prepend_session_config_warning(result)
    }
}

// --- Type conversion helpers ---

fn code_sym_to_tree_sym(
    sym: &crate::core::parser::CodeSymbol,
) -> crate::tools::context_tree::TreeSymbol {
    crate::tools::context_tree::TreeSymbol {
        name: sym.name.clone(),
        kind: sym.kind.clone(),
        line: sym.line,
        end_line: sym.end_line,
        signature: sym.signature.clone().unwrap_or_default(),
        children: sym.children.iter().map(code_sym_to_tree_sym).collect(),
    }
}

fn code_sym_to_skel_sym(
    sym: &crate::core::parser::CodeSymbol,
) -> crate::tools::file_skeleton::SkeletonSymbol {
    crate::tools::file_skeleton::SkeletonSymbol {
        name: sym.name.clone(),
        kind: sym.kind.clone(),
        line: sym.line,
        end_line: sym.end_line,
        signature: sym.signature.clone().unwrap_or_default(),
        children: sym.children.iter().map(code_sym_to_skel_sym).collect(),
    }
}

// --- Symbol-index helper ---

/// Walk a [`ProjectCache`] once and parse every file via tree-sitter, keying
/// the resulting symbol map by whatever the caller wants. `key_fn` lets
/// callers pick `String` (review_pr_diff) or `PathBuf` (find_dead_code)
/// without copy-pasting the loop body.
fn build_symbols_by_file<K, F>(
    cache: &ProjectCache,
    key_fn: F,
) -> HashMap<K, Vec<crate::core::parser::CodeSymbol>>
where
    K: Eq + std::hash::Hash,
    F: Fn(&str) -> K,
{
    let mut symbols_by_file: HashMap<K, Vec<crate::core::parser::CodeSymbol>> = HashMap::new();
    for (rel_path, content) in &cache.file_content {
        let ext = rel_path.rsplit('.').next().unwrap_or("");
        if let Ok(syms) = parse_with_tree_sitter(content, ext) {
            symbols_by_file.insert(key_fn(rel_path), syms);
        }
    }
    symbols_by_file
}

// --- Metadata helper ---

// make_tool() is re-exported from server_definitions — see imports at top of this file.

// ---------------------------------------------------------------------------
// U18: warmup helpers
// ---------------------------------------------------------------------------

/// RAII guard that removes `ref_id` from `state.warmup_in_flight` on drop.
///
/// This ensures the in-flight slot is released even if the warmup task panics
/// or returns early, allowing a future `spawn_ref_warmup` call to retry.
struct WarmupGuard {
    state: Arc<SharedState>,
    ref_id: crate::ref_index::RefId,
}

impl Drop for WarmupGuard {
    fn drop(&mut self) {
        // We are inside a tokio task, so we can't call `.await` here.
        // Use `try_lock` — if the mutex is contended we skip (the warmup loop
        // will remove the entry shortly anyway on its own unlock path).
        if let Ok(mut inflight) = self.state.warmup_in_flight.try_lock() {
            inflight.remove(&self.ref_id);
        }
    }
}

// ---------------------------------------------------------------------------
// U20: baseline-import helpers
// ---------------------------------------------------------------------------

/// A chunk that had no CAS hit and needs an Ollama embed.
pub struct MissedChunk {
    /// Repo-relative file path.
    pub rel_path: String,
    /// FNV hash of the raw content (used as the in-memory `CacheEntry.hash`).
    pub content_hash: String,
    /// Text prepared for embedding using the configured document shape.
    pub embed_text: String,
}

/// Summary returned by the baseline-import pass.
pub struct BaselineImportReport {
    /// Number of CAS hits (chunks loaded from parent chain, zero Ollama).
    pub hits: usize,
    /// Chunks that had no CAS hit — available for Ollama embed in Full mode.
    pub misses: Vec<MissedChunk>,
}

/// Walk every file in `project_cache`, compute chunk hashes, look up the CAS
/// parent chain, and register any hits into the ref's in-memory
/// `embedding_cache` and `search_index_cache` — without any Ollama calls.
///
/// Called by **both** Shallow and Full warmup so neither duplicates logic:
/// - Shallow invokes this and discards the misses.
/// - Full invokes this then calls [`embed_diff_chunks`] on the misses.
///
/// The CAS is rooted at `state.root_dir/.mcp_data` (the primary worktree's
/// data directory) regardless of which ref is being warmed, matching the
/// behaviour of `incremental_reembed`.
async fn import_baseline_for_ref(
    state: &Arc<SharedState>,
    ref_id: crate::ref_index::RefId,
    project_cache: Arc<ProjectCache>,
) -> BaselineImportReport {
    use crate::cache::cas::{CasStore, ChunkHash, ChunkKey};
    use crate::tools::semantic_search::{CachedSearchIndex, IndexFingerprint, SearchDocument};

    let ref_index = match state.ref_index(ref_id).await {
        Some(r) => r,
        None => {
            tracing::warn!(ref_id = ref_id.0, "import_baseline_for_ref: ref not found");
            return BaselineImportReport {
                hits: 0,
                misses: Vec::new(),
            };
        }
    };

    // CAS lives at the primary worktree's .mcp_data directory.
    let mcp_data_dir = state.root_dir.join(".mcp_data");
    let cas = CasStore::new(mcp_data_dir, state.config.document_cache_identity());
    let ref_id_hex = ref_index.cas_ref_id_hex.clone();
    let max_file_size = state.config.max_embed_file_size;
    let embed_doc_shape = state.config.embed_doc_shape;

    // Collect per-file results on a blocking thread to avoid holding async locks
    // during synchronous I/O.
    type HitEntry = (String, String, Vec<f32>); // (rel_path, content_hash, vector)
    let project_cache_for_idx = Arc::clone(&project_cache);
    let (hits_raw, misses): (Vec<HitEntry>, Vec<MissedChunk>) =
        tokio::task::spawn_blocking(move || {
            let mut hits: Vec<HitEntry> = Vec::new();
            let mut misses: Vec<MissedChunk> = Vec::new();

            for (rel_path, content) in &project_cache.file_content {
                // Skip files that exceed the max embed size.
                if content.len() > max_file_size {
                    continue;
                }
                let content_hash = crate::core::parser::hash_content(content);
                let embed_text = build_embedding_document(rel_path, content, embed_doc_shape);

                let chunk_hash = ChunkHash::of(&embed_text);
                let key = ChunkKey::new(rel_path.clone(), 0);

                match cas.lookup_chunk(&ref_id_hex, &key) {
                    Ok(Some(h)) if h == chunk_hash => {
                        // Chunk hash matches manifest entry — try to load the blob.
                        match cas.read_blob(&h) {
                            Ok(Some(vec)) => {
                                hits.push((rel_path.clone(), content_hash, vec));
                            }
                            _ => {
                                // Blob missing despite manifest hit — treat as miss.
                                misses.push(MissedChunk {
                                    rel_path: rel_path.clone(),
                                    content_hash,
                                    embed_text,
                                });
                            }
                        }
                    }
                    _ => {
                        misses.push(MissedChunk {
                            rel_path: rel_path.clone(),
                            content_hash,
                            embed_text,
                        });
                    }
                }
            }
            (hits, misses)
        })
        .await
        .unwrap_or_default();

    let hit_count = hits_raw.len();

    // Register hits into the in-memory embedding_cache.
    if !hits_raw.is_empty() {
        let mut cache = ref_index.embedding_cache.write().await;
        for (rel_path, content_hash, vector) in &hits_raw {
            cache.insert(
                rel_path.clone(),
                CacheEntry {
                    hash: content_hash.clone(),
                    vector: vector.clone(),
                },
            );
        }
    }

    // Build search_index_cache from the inherited blobs.
    if !hits_raw.is_empty() {
        let project_snap = project_cache_for_idx;
        let hits_snap: Vec<(String, Vec<f32>)> = hits_raw
            .iter()
            .map(|(p, _, v)| (p.clone(), v.clone()))
            .collect();

        let (docs, vectors) = tokio::task::spawn_blocking(move || {
            let hit_map: HashMap<String, Vec<f32>> = hits_snap.into_iter().collect();
            let mut docs: Vec<SearchDocument> = Vec::new();
            let mut vectors: Vec<Option<Vec<f32>>> = Vec::new();
            for (rel_path, content) in &project_snap.file_content {
                let header = crate::core::parser::extract_header(content);
                let ext = rel_path.rsplit('.').next().unwrap_or("");
                let symbols: Vec<String> = if let Ok(syms) = parse_with_tree_sitter(content, ext) {
                    crate::core::parser::flatten_symbols(&syms, None)
                        .into_iter()
                        .map(|s| s.name)
                        .collect()
                } else {
                    Vec::new()
                };
                let doc = SearchDocument::new(
                    rel_path.clone(),
                    header,
                    symbols,
                    Vec::new(),
                    content.as_str().to_string(),
                );
                let vec = hit_map.get(rel_path).cloned();
                docs.push(doc);
                vectors.push(vec);
            }
            (docs, vectors)
        })
        .await
        .unwrap_or_else(|_| (Vec::new(), Vec::new()));

        if !docs.is_empty() {
            use crate::tools::semantic_search::SearchIndex;
            let current_gen = ref_index
                .cache_generation
                .load(std::sync::atomic::Ordering::Acquire);
            let fp = IndexFingerprint::from_docs(&docs);
            let mut idx = SearchIndex::new();
            idx.index_with_vectors_and_tuning(
                docs,
                vectors,
                crate::core::embeddings::HnswTuning::global(),
            );
            let cached = Arc::new(CachedSearchIndex::new(idx, fp, current_gen));
            let mut guard = ref_index.search_index_cache.write().await;
            *guard = Some(cached);
            tracing::debug!(
                ref_id = ref_id.0,
                hits = hit_count,
                "import_baseline_for_ref: search_index_cache built from inherited blobs"
            );
        }
    }

    BaselineImportReport {
        hits: hit_count,
        misses,
    }
}

/// Embed the diff chunks (CAS misses from [`import_baseline_for_ref`]) via
/// Ollama, gated by the `ollama_semaphore`.  Writes new blobs + manifest
/// entries to CAS and updates the ref's `embedding_cache`.
///
/// Called by `spawn_full_warmup_task` after the baseline-import phase.
/// After embeds land, refreshes `search_index_cache` to include the new
/// vectors.
async fn embed_diff_chunks(
    state: &Arc<SharedState>,
    ref_id: crate::ref_index::RefId,
    misses: &[MissedChunk],
) {
    use crate::cache::cas::{CasStore, ChunkHash, ChunkKey};
    use crate::tools::semantic_search::{CachedSearchIndex, IndexFingerprint, SearchDocument};

    if misses.is_empty() {
        return;
    }

    let ref_index = match state.ref_index(ref_id).await {
        Some(r) => r,
        None => {
            tracing::warn!(ref_id = ref_id.0, "embed_diff_chunks: ref not found");
            return;
        }
    };

    tracing::info!(
        ref_id = ref_id.0,
        count = misses.len(),
        "ref_warmup full: embedding diff chunks via Ollama"
    );

    let embed_texts: Vec<String> = misses.iter().map(|m| m.embed_text.clone()).collect();

    let vectors = match state.ollama.embed_documents(&embed_texts).await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                ref_id = ref_id.0,
                error = %e,
                "ref_warmup full: Ollama embed for diff chunks failed (non-fatal)"
            );
            return;
        }
    };

    // Write new blobs + update embedding_cache.
    let mcp_data_dir = state.root_dir.join(".mcp_data");
    let cas = CasStore::new(mcp_data_dir, state.config.document_cache_identity());
    let ref_id_hex = ref_index.cas_ref_id_hex.clone();

    let mut manifest_updates: Vec<(ChunkKey, ChunkHash)> = Vec::new();
    {
        let mut cache = ref_index.embedding_cache.write().await;
        for (i, miss) in misses.iter().enumerate() {
            if let Some(vec) = vectors.get(i) {
                cache.insert(
                    miss.rel_path.clone(),
                    CacheEntry {
                        hash: miss.content_hash.clone(),
                        vector: vec.clone(),
                    },
                );
                let chunk_hash = ChunkHash::of(&miss.embed_text);
                let key = ChunkKey::new(miss.rel_path.clone(), 0);
                if let Err(e) = cas.write_blob(&chunk_hash, vec) {
                    tracing::warn!(
                        rel_path = %miss.rel_path,
                        error = %e,
                        "embed_diff_chunks: CAS write_blob failed (non-fatal)"
                    );
                } else {
                    manifest_updates.push((key, chunk_hash));
                }
            }
        }
    }
    if let Err(e) = cas.update_manifest(&ref_id_hex, &manifest_updates) {
        tracing::warn!(
            ref_id = ref_id.0,
            error = %e,
            "embed_diff_chunks: manifest update failed (non-fatal)"
        );
    }

    // Refresh search_index_cache to include the newly embedded diff vectors.
    // Read the current embedding_cache snapshot and rebuild the index.
    let project_cache_snap = {
        let guard = ref_index.project_cache.read().await;
        guard.as_ref().map(Arc::clone)
    };
    if let Some(project_cache) = project_cache_snap {
        let embedding_snap: HashMap<String, Vec<f32>> = {
            let cache = ref_index.embedding_cache.read().await;
            cache
                .iter()
                .map(|(k, v)| (k.clone(), v.vector.clone()))
                .collect()
        };

        let (docs, vectors_for_idx) = tokio::task::spawn_blocking(move || {
            let mut docs: Vec<SearchDocument> = Vec::new();
            let mut vectors: Vec<Option<Vec<f32>>> = Vec::new();
            for (rel_path, content) in &project_cache.file_content {
                let header = crate::core::parser::extract_header(content);
                let ext = rel_path.rsplit('.').next().unwrap_or("");
                let symbols: Vec<String> = if let Ok(syms) = parse_with_tree_sitter(content, ext) {
                    crate::core::parser::flatten_symbols(&syms, None)
                        .into_iter()
                        .map(|s| s.name)
                        .collect()
                } else {
                    Vec::new()
                };
                let doc = SearchDocument::new(
                    rel_path.clone(),
                    header,
                    symbols,
                    Vec::new(),
                    (**content).clone(),
                );
                let vec = embedding_snap.get(rel_path).cloned();
                docs.push(doc);
                vectors.push(vec);
            }
            (docs, vectors)
        })
        .await
        .unwrap_or_else(|_| (Vec::new(), Vec::new()));

        if !docs.is_empty() {
            use crate::tools::semantic_search::SearchIndex;
            let current_gen = ref_index
                .cache_generation
                .load(std::sync::atomic::Ordering::Acquire);
            let fp = IndexFingerprint::from_docs(&docs);
            let mut idx = SearchIndex::new();
            idx.index_with_vectors_and_tuning(
                docs,
                vectors_for_idx,
                crate::core::embeddings::HnswTuning::global(),
            );
            let cached = Arc::new(CachedSearchIndex::new(idx, fp, current_gen));
            let mut guard = ref_index.search_index_cache.write().await;
            *guard = Some(cached);
            tracing::debug!(
                ref_id = ref_id.0,
                "embed_diff_chunks: search_index_cache refreshed with diff embeddings"
            );
        }
    }
}

/// Run the semantic-search warmup pipeline for a specific `ref_id`.
///
/// Mirrors `warmup_semantic_search_cache` but targets the per-ref
/// `search_index_cache` and `cache_generation` rather than the default ref's.
/// Retained for integration-test use; production Full warmup now delegates
/// to `import_baseline_for_ref` + `embed_diff_chunks` (U20).
#[allow(dead_code)]
async fn warmup_ref_search_cache(state: &Arc<SharedState>, ref_id: crate::ref_index::RefId) {
    use crate::server_adapters::{CachedWalkerIndexer, OllamaEmbedder};
    use crate::tools::semantic_search::{SemanticSearchOptions, semantic_code_search};

    let ref_index = match state.ref_index(ref_id).await {
        Some(r) => r,
        None => {
            tracing::warn!(ref_id = ref_id.0, "warmup_ref_search_cache: ref not found");
            return;
        }
    };

    let t0 = std::time::Instant::now();
    tracing::info!(
        ref_id = ref_id.0,
        "ref_warmup full: SearchIndex warmup starting"
    );

    let options = SemanticSearchOptions {
        root_dir: ref_index.root_dir.clone(),
        query: "warmup".to_string(),
        top_k: Some(1),
        semantic_weight: None,
        keyword_weight: None,
        min_semantic_score: None,
        min_keyword_score: None,
        min_combined_score: None,
        require_keyword_match: None,
        require_semantic_match: None,
        include_globs: None,
        exclude_globs: None,
        recency_window_days: None,
        scope: None,
    };

    let embedder = OllamaEmbedder(state.ollama.clone());
    let walker = CachedWalkerIndexer {
        config: state.config.clone(),
        ollama: state.ollama.clone(),
        state: Arc::clone(state),
    };

    match semantic_code_search(
        options,
        &embedder,
        &walker,
        Some(Arc::clone(&ref_index.search_index_cache)),
        Some(Arc::clone(&ref_index.cache_generation)),
    )
    .await
    {
        Ok(_) => tracing::info!(
            ref_id = ref_id.0,
            elapsed_ms = t0.elapsed().as_millis(),
            "ref_warmup full: SearchIndex warmup complete"
        ),
        Err(e) => tracing::warn!(
            ref_id = ref_id.0,
            elapsed_ms = t0.elapsed().as_millis(),
            error = %e,
            "ref_warmup full: SearchIndex warmup failed (non-fatal)"
        ),
    }
}

// ---------------------------------------------------------------------------
// Startup warmup
// ---------------------------------------------------------------------------

/// Run a trivial semantic search query through the full pipeline to populate
/// the in-memory `SearchIndex` cache and pre-build the HNSW index.
///
/// This is intentionally a module-level (non-`impl`) function so tests can
/// call it directly with a bare `Arc<SharedState>` without constructing a
/// full `ContextPlusServer`.
pub async fn warmup_semantic_search_cache(state: &Arc<SharedState>) {
    use crate::server_adapters::{CachedWalkerIndexer, OllamaEmbedder};
    use crate::tools::semantic_search::{SemanticSearchOptions, semantic_code_search};

    let t0 = std::time::Instant::now();
    tracing::info!("SearchIndex warmup starting");

    // Warmup always targets the default ref — it runs at daemon startup before
    // any per-session refs are registered.  `default_ref()` is guaranteed to be
    // present; the `expect` is safe under normal operating conditions.
    let default_ref = state
        .default_ref()
        .expect("default_ref always present during warmup");

    let options = SemanticSearchOptions {
        root_dir: default_ref.root_dir.clone(),
        query: "warmup".to_string(),
        top_k: Some(1),
        semantic_weight: None,
        keyword_weight: None,
        min_semantic_score: None,
        min_keyword_score: None,
        min_combined_score: None,
        require_keyword_match: None,
        require_semantic_match: None,
        include_globs: None,
        exclude_globs: None,
        recency_window_days: None,
        scope: None,
    };

    let embedder = OllamaEmbedder(state.ollama.clone());
    let walker = CachedWalkerIndexer {
        config: state.config.clone(),
        ollama: state.ollama.clone(),
        state: Arc::clone(state),
    };

    match semantic_code_search(
        options,
        &embedder,
        &walker,
        Some(Arc::clone(&default_ref.search_index_cache)),
        Some(Arc::clone(&default_ref.cache_generation)),
    )
    .await
    {
        Ok(_) => tracing::info!(
            elapsed_ms = t0.elapsed().as_millis(),
            "SearchIndex warmup complete"
        ),
        Err(e) => tracing::warn!(
            elapsed_ms = t0.elapsed().as_millis(),
            error = %e,
            "SearchIndex warmup failed (non-fatal)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::semantic_search::WalkAndIndexFn;
    use rmcp::model::RawContent;
    use serde_json::json;

    fn test_server() -> ContextPlusServer {
        let config = Config::from_env();
        let root = std::env::temp_dir().join("contextplus-test");
        let _ = std::fs::create_dir_all(&root);
        ContextPlusServer::new(root, config)
    }

    #[tokio::test]
    async fn project_cache_refresh_survives_global_rayon_saturation() {
        const CHILD: &str = "CONTEXTPLUS_RAYON_SATURATION_CHILD";
        if std::env::var_os(CHILD).is_none() {
            // Isolate saturation from other tests that legitimately use the global pool.
            let output = tokio::task::spawn_blocking(|| {
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "server::tests::project_cache_refresh_survives_global_rayon_saturation",
                        "--nocapture",
                    ])
                    .env(CHILD, "1")
                    .env("RAYON_NUM_THREADS", "2")
                    .output()
                    .unwrap()
            })
            .await
            .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let tree = tempfile::tempdir().unwrap();
        for name in ["one", "two", "three"] {
            std::fs::write(
                tree.path().join(format!("{name}.rs")),
                format!("fn {name}() {{}}"),
            )
            .unwrap();
        }
        let server = ContextPlusServer::new(tree.path().to_path_buf(), Config::from_env());
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let mut releases = Vec::new();
        for _ in 0..rayon::current_num_threads() {
            let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
            releases.push(release_tx);
            let started_tx = started_tx.clone();
            rayon::spawn(move || {
                started_tx.send(()).unwrap();
                let _ = release_rx.recv();
            });
        }
        for _ in &releases {
            started_rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
        }
        let refreshed = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            server.ensure_project_cache(),
        )
        .await;
        // Release even on timeout so Tokio can drain its blocked refresh task.
        drop(releases);
        let cache = refreshed
            .expect("project-cache refresh starved by global Rayon work")
            .unwrap();
        assert_eq!(cache.file_content.len(), 3);
    }

    #[test]
    fn embedding_cache_names_include_prefix_and_shape_identity() {
        let mut config = Config::from_env();
        config.ollama_embed_model = "nomic-embed-text".to_string();
        config.embed_query_prefix.clear();
        config.embed_doc_prefix.clear();
        config.embed_doc_shape = crate::config::EmbedDocShape::Head;

        assert_eq!(
            cache_name("embeddings", &config),
            "embeddings-nomic-embed-text"
        );
        assert_eq!(
            rkyv_store::query_cache_name(&config.query_cache_identity()),
            "query-embeddings-nomic-embed-text"
        );

        let baseline_doc = cache_name("embeddings", &config);
        config.embed_doc_prefix = "title: none | text: ".to_string();
        assert_ne!(cache_name("embeddings", &config), baseline_doc);

        config.embed_doc_prefix.clear();
        config.embed_doc_shape = crate::config::EmbedDocShape::Outline;
        assert_ne!(cache_name("embeddings", &config), baseline_doc);

        let baseline_query = rkyv_store::query_cache_name("nomic-embed-text");
        config.embed_query_prefix = "query: ".to_string();
        assert_ne!(
            rkyv_store::query_cache_name(&config.query_cache_identity()),
            baseline_query
        );

        config.ollama_embed_model = "a".repeat(70);
        config.embed_doc_shape = crate::config::EmbedDocShape::Outline;
        config.embed_doc_prefix = "first: ".to_string();
        let first = rkyv_store::model_slug(&config.document_cache_identity());
        config.embed_doc_prefix = "second: ".to_string();
        let second = rkyv_store::model_slug(&config.document_cache_identity());
        assert_ne!(first, second);
    }

    #[test]
    fn head_embedding_document_matches_legacy_format() {
        let content = "// Authentication helpers\npub fn authenticate(token: &str) -> bool { !token.is_empty() }\n";
        let legacy = "Authentication helpers | pub fn authenticate(token: &str) -> bool { !token.is_empty() } src/auth.rs // Authentication helpers\npub fn authenticate(token: &str) -> bool { !token.is_empty() }\n";

        assert_eq!(
            build_embedding_document("src/auth.rs", content, crate::config::EmbedDocShape::Head),
            legacy
        );
    }

    #[test]
    fn outline_embedding_document_matches_run2_d1_golden_files() {
        let fixtures = [
            include_str!("../tests/fixtures/embed_outline/synthetic-worker.json"),
            include_str!("../tests/fixtures/embed_outline/synthetic-worker.test.json"),
            include_str!("../tests/fixtures/embed_outline/synthetic_worker.go.json"),
        ];

        for fixture in fixtures {
            let fixture: serde_json::Value = serde_json::from_str(fixture).unwrap();
            let path = fixture["path"].as_str().unwrap();
            let content = fixture["content"].as_str().unwrap();
            let expected = fixture["expected"].as_str().unwrap();
            let actual =
                build_embedding_document(path, content, crate::config::EmbedDocShape::Outline);
            assert_eq!(actual.as_bytes(), expected.as_bytes(), "{path}");
        }
    }

    #[test]
    fn synthetic_outline_fixtures_cover_filters_and_caps() {
        let typescript: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/embed_outline/synthetic-worker.json"
        ))
        .unwrap();
        let outline = embedding_outline(
            typescript["path"].as_str().unwrap(),
            typescript["content"].as_str().unwrap(),
        );
        assert_eq!(outline.chars().count(), 1500);
        assert!(outline.lines().all(|line| line.chars().count() <= 160));
        assert!(outline.lines().any(|line| line.chars().count() == 160));
        assert!(outline.contains("export class SyntheticWorker"));
        assert!(outline.contains("async execute("));
        assert!(outline.contains("static create("));
        assert!(outline.contains("private reset("));
        assert!(outline.contains("transform<TValue>("));
        assert!(outline.contains("const arrowTask = (value: number) => value + x;"));
        assert!(!outline.contains("type ExternalShape,"));
        assert!(!outline.contains("const x = 1;"));

        let test_file: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/embed_outline/synthetic-worker.test.json"
        ))
        .unwrap();
        let test_outline = embedding_outline(
            test_file["path"].as_str().unwrap(),
            test_file["content"].as_str().unwrap(),
        );
        assert!(!test_outline.contains("describe('synthetic worker'"));
        assert!(test_outline.contains("beforeEach(() => {"));
        assert!(test_outline.contains("it('runs a task'"));
        assert!(!test_outline.contains("registerHook(createHook());"));

        let go: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/embed_outline/synthetic_worker.go.json"
        ))
        .unwrap();
        let go_outline = embedding_outline(
            go["path"].as_str().unwrap(),
            go["content"].as_str().unwrap(),
        );
        assert!(go_outline.contains("type Runner interface"));
        assert!(go_outline.contains("Start(context.Context) error"));
        assert!(go_outline.contains("func NewRunner(config Config) Runner"));
    }

    #[test]
    fn embedding_outline_matches_run2_sql_and_extension_rules() {
        let sql = "create table lower_case (id int);\nCREATE OR REPLACE FUNCTION do_work() RETURNS void;\nCREATE UNIQUE INDEX idx ON t (id);\n";
        assert_eq!(
            embedding_outline("schema.sql", sql),
            "create table lower_case (id int);\nCREATE OR REPLACE FUNCTION do_work() RETURNS void;\nCREATE UNIQUE INDEX idx ON t (id);"
        );
        assert_eq!(embedding_outline("src/lib.rs", "pub fn ignored() {}"), "");
    }

    #[test]
    fn tool_definitions_returns_the_five_facade_tools() {
        let defs = tool_definitions();
        assert_eq!(defs.len(), 5, "expected 5 tools, got {}", defs.len());
        for tool in defs {
            assert!(!tool.name.is_empty(), "tool name must not be empty");
            assert!(
                tool.description.is_some(),
                "tool '{}' must have a description",
                tool.name
            );
        }
    }

    #[test]
    fn tool_definitions_contain_expected_names() {
        let defs = tool_definitions();
        let names: Vec<&str> = defs.iter().map(|t| t.name.as_ref()).collect();
        for name in ["explore", "outline", "impact", "check", "worktrees"] {
            assert!(names.contains(&name), "missing tool: {}", name);
        }
        assert!(
            !names.contains(&"get_file_skeleton"),
            "pre-facade names must not be listed"
        );
    }

    fn facade_server() -> (tempfile::TempDir, ContextPlusServer) {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(
            tmp.path().join("src/auth.rs"),
            "pub fn verify_token(t: &str) -> bool { t.len() > 3 }\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("src/main.rs"),
            "mod auth;\nfn main() { auth::verify_token(\"x\"); }\n",
        )
        .unwrap();
        let mut config = Config::from_env();
        config.ollama_host = "http://127.0.0.1:1".to_string();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(tmp.path().to_path_buf(), config);
        (tmp, server)
    }

    async fn cluster_facade_server(
        files: impl IntoIterator<Item = (String, String)>,
    ) -> (tempfile::TempDir, wiremock::MockServer, ContextPlusServer) {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let provider = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = request
                    .body_json::<serde_json::Value>()
                    .ok()
                    .and_then(|body| body["input"].as_array().cloned())
                    .unwrap_or_default();
                let embeddings: Vec<Vec<f32>> = inputs
                    .iter()
                    .map(|input| {
                        let text = input.as_str().unwrap_or_default().to_lowercase();
                        if text.contains("projection") {
                            vec![1.0, 0.0]
                        } else if text.contains("event sourcing")
                            || text.contains("consumer checkpoint")
                            || text.contains("topic_alpha")
                        {
                            vec![0.8, 0.6]
                        } else {
                            vec![0.0, 1.0]
                        }
                    })
                    .collect();
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embeddings": embeddings }))
            })
            .mount(&provider)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&provider)
            .await;

        let tmp = tempfile::tempdir().unwrap();
        for (relative, content) in files {
            let absolute = tmp.path().join(relative);
            std::fs::create_dir_all(absolute.parent().unwrap()).unwrap();
            std::fs::write(absolute, content).unwrap();
        }
        let mut config = Config::from_env();
        config.ollama_host = provider.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(tmp.path().to_path_buf(), config);
        (tmp, provider, server)
    }

    fn text_of(result: &CallToolResult) -> String {
        match &result.content[0].raw {
            RawContent::Text(t) => t.text.clone(),
            _ => panic!("expected text content"),
        }
    }

    async fn identifier_server(
        files: &[(&str, &str)],
    ) -> (tempfile::TempDir, wiremock::MockServer, ContextPlusServer) {
        identifier_server_with_tracker_mode(files, TrackerMode::Off).await
    }

    async fn identifier_server_with_tracker_mode(
        files: &[(&str, &str)],
        tracker_mode: TrackerMode,
    ) -> (tempfile::TempDir, wiremock::MockServer, ContextPlusServer) {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let count = request
                    .body_json::<serde_json::Value>()
                    .ok()
                    .and_then(|body| body["input"].as_array().map(Vec::len))
                    .unwrap_or(1);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; count]
                }))
            })
            .mount(&ollama)
            .await;

        let repo = tempfile::tempdir().unwrap();
        for &(path, source) in files {
            let full_path = repo.path().join(path);
            std::fs::create_dir_all(full_path.parent().unwrap()).unwrap();
            std::fs::write(full_path, source).unwrap();
        }

        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = tracker_mode;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(repo.path().to_path_buf(), config);
        (repo, ollama, server)
    }

    async fn explore_scoped_identifier(server: &ContextPlusServer, matching: &str) -> String {
        let mut args = serde_json::Map::new();
        args.insert("query".into(), json!("idempotencyKey"));
        args.insert("kind".into(), json!("identifiers"));
        args.insert("match".into(), json!(matching));
        args.insert("path".into(), json!("packages/domains/payments"));
        args.insert("top_k".into(), json!(10));
        let result = server.dispatch("explore", args).await;
        assert_eq!(result.is_error, Some(false), "{}", text_of(&result));
        text_of(&result)
    }

    async fn explore_identifier(
        server: &ContextPlusServer,
        query: &str,
        path: Option<&str>,
    ) -> String {
        let mut args = serde_json::Map::new();
        args.insert("query".into(), json!(query));
        args.insert("kind".into(), json!("identifiers"));
        args.insert("match".into(), json!("keywords"));
        args.insert("top_k".into(), json!(10));
        args.insert("top_calls_per_identifier".into(), json!(10));
        if let Some(path) = path {
            args.insert("path".into(), json!(path));
        }
        let result = server.dispatch("explore", args).await;
        assert_eq!(result.is_error, Some(false), "{}", text_of(&result));
        text_of(&result)
    }

    fn identifier_block_for_path<'a>(output: &'a str, path: &str) -> &'a str {
        let definition_marker = format!(" - {path} (");
        output
            .split("\n\n")
            .find(|block| block.contains(&definition_marker))
            .unwrap_or_else(|| panic!("missing identifier result for {path}:\n{output}"))
    }

    #[tokio::test]
    async fn explore_identifier_private_top_level_const_stays_file_local() {
        let files = [
            (
                "src/a.ts",
                "const pending = begin();\nexport function start() {\n  return pending;\n}\n",
            ),
            (
                "src/b.ts",
                "import { start } from './a';\nconst pending = other();\nexport function run() {\n  start();\n  return pending;\n}\n",
            ),
        ];
        let (_repo, _ollama, server) = identifier_server(&files).await;
        let output = explore_identifier(&server, "pending", None).await;
        let a = identifier_block_for_path(&output, "src/a.ts");
        let b = identifier_block_for_path(&output, "src/b.ts");

        assert!(
            a.contains("Calls (1/1)"),
            "A's private pending leaked:\n{a}"
        );
        assert!(a.contains("src/a.ts:L3"), "A's local use is missing:\n{a}");
        assert!(
            b.contains("Calls (1/1)"),
            "B's private pending leaked:\n{b}"
        );
        assert!(b.contains("src/b.ts:L5"), "B's local use is missing:\n{b}");
    }

    #[tokio::test]
    async fn explore_identifier_counts_typescript_type_only_import_consumer() {
        let files = [
            (
                "src/a.ts",
                "export type ResolveActorGrantsFn = (input: string) => boolean;\n",
            ),
            (
                "src/b.ts",
                "import type { ResolveActorGrantsFn } from './a';\nexport const run = (resolve: ResolveActorGrantsFn) => resolve('x');\n",
            ),
        ];
        let (_repo, _ollama, server) = identifier_server(&files).await;
        let output = explore_identifier(&server, "ResolveActorGrantsFn", None).await;
        let result = identifier_block_for_path(&output, "src/a.ts");

        assert!(
            result.contains("Calls (1/1)"),
            "type-only consumer must count exactly once:\n{result}"
        );
        assert!(
            result.contains("src/b.ts:L2"),
            "type-only consumer location is missing:\n{result}"
        );
    }

    #[tokio::test]
    async fn explore_identifier_counts_local_rust_use_consumer() {
        let files = [
            (
                "src/account.rs",
                "pub fn load_account() -> usize { 1 }\npub fn reload() -> usize {\n    load_account()\n}\n",
            ),
            (
                "src/consumer.rs",
                "use crate::account::load_account;\npub fn run() -> usize {\n    load_account()\n}\n",
            ),
        ];
        let (_repo, _ollama, server) = identifier_server(&files).await;
        let output = explore_identifier(&server, "load_account", None).await;
        let result = identifier_block_for_path(&output, "src/account.rs");

        assert!(
            result.contains("Calls (2/2)"),
            "local Rust definition and imported consumer must total exactly two:\n{result}"
        );
        assert!(
            result.contains("src/account.rs:L3") && result.contains("src/consumer.rs:L3"),
            "expected Rust call locations are missing:\n{result}"
        );
    }

    #[tokio::test]
    async fn explore_identifier_keywords_honors_path_scope() {
        let files = [
            (
                "packages/domains/payments/idempotency.ts",
                "export const idempotencyKey = 'payments-key';\n",
            ),
            (
                "packages/platform/orchestrate/idempotency.ts",
                "export const idempotencyKey = 'orchestrate-key';\n",
            ),
        ];
        let (_repo, _ollama, server) = identifier_server(&files).await;
        let text = explore_scoped_identifier(&server, "keywords").await;

        assert!(
            text.contains("packages/domains/payments/idempotency.ts"),
            "scoped definition missing:\n{text}"
        );
        assert!(
            !text.contains("packages/platform/orchestrate/idempotency.ts"),
            "identifier outside path leaked into keyword results:\n{text}"
        );
    }

    #[tokio::test]
    async fn explore_identifier_meaning_honors_path_scope() {
        let files = [
            (
                "packages/domains/payments/idempotency.ts",
                "export const idempotencyKey = 'payments-key';\n",
            ),
            (
                "packages/platform/orchestrate/idempotency.ts",
                "export const idempotencyKey = 'orchestrate-key';\n",
            ),
        ];
        let (_repo, _ollama, server) = identifier_server(&files).await;
        let text = explore_scoped_identifier(&server, "meaning").await;

        assert!(
            text.contains("packages/domains/payments/idempotency.ts"),
            "scoped definition missing:\n{text}"
        );
        assert!(
            !text.contains("packages/platform/orchestrate/idempotency.ts"),
            "identifier outside path leaked into meaning results:\n{text}"
        );
    }

    #[tokio::test]
    async fn large_cached_identifier_index_preserves_scoped_and_unscoped_search_without_copying() {
        let mut inside = String::new();
        let mut outside = String::new();
        for i in 0..300 {
            inside.push_str(&format!("export const insideNoise{i} = {i};\n"));
            outside.push_str(&format!("export const outsideNoise{i} = {i};\n"));
        }
        inside.push_str("export const needleTarget = 'inside';\n");
        outside.push_str("export const needleTarget = 'outside';\n");
        let files = [
            ("src/inside/large.ts", inside.as_str()),
            ("src/outside/large.ts", outside.as_str()),
        ];
        let (_repo, _ollama, server) = identifier_server(&files).await;

        let unscoped = explore_identifier(&server, "needleTarget", None).await;
        assert!(
            unscoped.contains("src/inside/large.ts") && unscoped.contains("src/outside/large.ts"),
            "unscoped search lost a cached-index result:\n{unscoped}"
        );
        let index_before = server
            .current_ref()
            .await
            .identifier_index
            .read()
            .await
            .as_ref()
            .cloned()
            .expect("identifier index should be cached after the first search");
        assert!(
            index_before.docs.len() >= 600,
            "fixture was not large enough"
        );

        let scoped = explore_identifier(&server, "needleTarget", Some("src/inside")).await;
        assert!(
            scoped.contains("src/inside/large.ts"),
            "scoped search lost the in-scope result:\n{scoped}"
        );
        assert!(
            !scoped.contains("src/outside/large.ts"),
            "scoped search leaked the out-of-scope result:\n{scoped}"
        );
        let index_after = server
            .current_ref()
            .await
            .identifier_index
            .read()
            .await
            .as_ref()
            .cloned()
            .expect("identifier index should remain cached");
        assert!(
            Arc::ptr_eq(&index_before, &index_after),
            "scoped lookup unexpectedly replaced the cached index"
        );

        let source = include_str!("server.rs");
        let handler = source
            .split("async fn handle_semantic_identifier_search")
            .nth(1)
            .and_then(|tail| tail.split("async fn handle_semantic_navigate").next())
            .expect("identifier search handler source");
        assert!(
            !handler.contains("scoped_docs")
                && !handler.contains("scoped_vectors")
                && !handler.contains("doc.clone()")
                && !handler.contains("extend_from_slice"),
            "identifier search handler must borrow cached docs/vectors or select by index; it must not deep-copy the corpus"
        );
    }

    #[tokio::test]
    async fn deleted_file_disappears_from_semantic_lexical_and_identifier_indexes() {
        let mut owned_files = vec![(
            "000_deleted.rs".to_string(),
            "pub fn deleted_index_needle() { /* deleted semantic needle */ }\n".to_string(),
        )];
        for i in 0..24 {
            owned_files.push((
                format!("src/survivor_{i}.rs"),
                format!("pub fn survivor_{i}() {{}}\n"),
            ));
        }
        let borrowed_files: Vec<_> = owned_files
            .iter()
            .map(|(path, source)| (path.as_str(), source.as_str()))
            .collect();
        let (repo, _ollama, server) = identifier_server(&borrowed_files).await;

        let semantic_before = server
            .handle_semantic_code_search(semantic_args("deleted semantic needle"))
            .await
            .unwrap();
        assert!(
            text_of(&semantic_before).contains("000_deleted.rs"),
            "semantic fixture did not warm the deleted path: {}",
            text_of(&semantic_before)
        );
        let mut lexical_args = serde_json::Map::new();
        lexical_args.insert("query".into(), json!("deleted_index_needle"));
        let lexical_before = server
            .handle_lexical_search(lexical_args.clone())
            .await
            .unwrap();
        assert!(
            text_of(&lexical_before).contains("000_deleted.rs"),
            "lexical fixture did not warm the deleted path: {}",
            text_of(&lexical_before)
        );
        let identifier_before = explore_identifier(&server, "deleted_index_needle", None).await;
        assert!(
            identifier_before.contains("000_deleted.rs"),
            "identifier fixture did not warm the deleted path: {identifier_before}"
        );

        std::fs::remove_file(repo.path().join("000_deleted.rs")).unwrap();
        let callback = server.build_tracker_callback().await;
        callback(
            repo.path().to_path_buf(),
            vec!["000_deleted.rs".to_string()],
        )
        .await
        .unwrap();

        let semantic_after = server
            .handle_semantic_code_search(semantic_args("deleted semantic needle"))
            .await
            .unwrap();
        let lexical_after = server.handle_lexical_search(lexical_args).await.unwrap();
        let identifier_after = explore_identifier(&server, "deleted_index_needle", None).await;

        assert!(
            !text_of(&semantic_after).contains("000_deleted.rs"),
            "deleted path remained in the semantic index: {}",
            text_of(&semantic_after)
        );
        assert!(
            !text_of(&lexical_after).contains("000_deleted.rs"),
            "deleted path remained in the lexical index: {}",
            text_of(&lexical_after)
        );
        assert!(
            !identifier_after.contains("000_deleted.rs"),
            "deleted path remained in the identifier index: {identifier_after}"
        );
    }

    #[tokio::test]
    async fn single_file_change_does_not_reload_identifier_cache_from_disk() {
        let files = [(
            "src/account.rs",
            "pub fn load_account() { /* original account loader */ }\n",
        )];
        let (repo, _ollama, server) = identifier_server(&files).await;
        let cache_name = cache_name("identifier-embeddings", &server.state.config);
        let load_probe = rkyv_store::test_seams::LoadCacheProbe::new(repo.path(), &cache_name);

        let before = explore_identifier(&server, "load_account", None).await;
        assert!(before.contains("src/account.rs"), "{before}");
        assert_eq!(
            load_probe.count(),
            1,
            "the identifier embedding cache should be loaded once at first use"
        );

        std::fs::write(
            repo.path().join("src/account.rs"),
            "pub fn load_account_v2() { /* refreshed account loader */ }\n",
        )
        .unwrap();
        let callback = server.build_tracker_callback().await;
        callback(
            repo.path().to_path_buf(),
            vec!["src/account.rs".to_string()],
        )
        .await
        .unwrap();

        let after = explore_identifier(&server, "load_account_v2", None).await;
        assert!(after.contains("load_account_v2"), "{after}");
        assert_eq!(
            load_probe.count(),
            1,
            "a one-file refresh must reuse the resident identifier cache"
        );
    }

    async fn identifier_server_with_saved_cache() -> (
        tempfile::TempDir,
        wiremock::MockServer,
        ContextPlusServer,
        String,
        usize,
    ) {
        let files = [(
            "src/ledger.rs",
            "pub fn open_ledger() {}\npub fn close_ledger() {}\npub fn audit_ledger() {}\n",
        )];
        let (repo, ollama, server) = identifier_server(&files).await;
        let name = cache_name("identifier-embeddings", &server.state.config);
        explore_identifier(&server, "open_ledger", None).await;
        let saved = wait_for_identifier_cache(repo.path(), &name, 3).await;
        (repo, ollama, server, name, saved)
    }

    async fn wait_for_identifier_cache(
        root: &std::path::Path,
        name: &str,
        at_least: usize,
    ) -> usize {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if let Ok(Some(data)) = rkyv_store::load_cache(root, name)
                    && data.keys.len() >= at_least
                {
                    return data.keys.len();
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap_or(0)
    }

    async fn embed_one_more_identifier(repo: &std::path::Path, server: &ContextPlusServer) {
        std::fs::write(
            repo.join("src/ledger.rs"),
            "pub fn open_ledger() {}\npub fn close_ledger() {}\npub fn audit_ledger() {}\npub fn reopen_ledger() {}\n",
        )
        .unwrap();
        let callback = server.build_tracker_callback().await;
        callback(repo.to_path_buf(), vec!["src/ledger.rs".to_string()])
            .await
            .unwrap();
        let output = explore_identifier(server, "reopen_ledger", None).await;
        assert!(output.contains("reopen_ledger"), "{output}");
    }

    #[tokio::test]
    async fn identifier_save_rebuilds_an_unreadable_cache_file_from_memory() {
        let (repo, _ollama, server, name, saved) = identifier_server_with_saved_cache().await;
        assert!(saved >= 3, "the first identifier save never landed");
        std::fs::write(
            repo.path().join(".mcp_data").join(format!("{name}.rkyv")),
            b"not a cache",
        )
        .unwrap();

        embed_one_more_identifier(repo.path(), &server).await;

        let rebuilt = wait_for_identifier_cache(repo.path(), &name, 1).await;
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let rebuilt = rkyv_store::load_cache(repo.path(), &name)
            .unwrap()
            .map_or(rebuilt, |data| data.keys.len());
        assert!(
            rebuilt > saved,
            "the save after a corrupt cache file wrote {rebuilt} entries, not the {} in memory",
            saved + 1
        );
    }

    #[tokio::test]
    async fn eviction_flush_writes_unsaved_identifiers_and_rebuilds_a_missing_cache_file() {
        let (repo, _ollama, server, name, saved) = identifier_server_with_saved_cache().await;
        assert!(saved >= 3, "the first identifier save never landed");
        std::fs::remove_file(repo.path().join(".mcp_data").join(format!("{name}.rkyv"))).unwrap();

        let primary = server.state.default_ref().unwrap();
        let embedded: Arc<[f32]> = Arc::from(vec![0.0, 1.0]);
        primary
            .identifier_vectors
            .get()
            .unwrap()
            .write()
            .await
            .insert("reopen_ledger".into(), Arc::clone(&embedded));
        primary
            .identifier_unsaved
            .lock()
            .unwrap()
            .insert("reopen_ledger".into(), embedded);
        clear_ref_heavy_caches(&primary, &name).await;

        let flushed = rkyv_store::load_cache(repo.path(), &name)
            .unwrap()
            .map_or(0, |data| data.keys.len());
        assert!(
            flushed > saved,
            "eviction left {flushed} identifier entries on disk, not the {} in memory",
            saved + 1
        );
    }

    #[test]
    fn keyword_index_build_holds_one_batch_at_a_time() {
        let files: crate::core::walker::ContentMap = (0..300)
            .map(|i| {
                let body: String = (0..800)
                    .map(|j| format!("word{} ", (i * 7 + j) % 900))
                    .collect();
                (format!("src/file_{i}.rs"), Arc::new(body))
            })
            .collect();
        let files = crate::core::walker::FileContents::from(files);
        let mut paths: Vec<&str> = files.keys().map(String::as_str).collect();
        paths.sort_unstable();

        // Counted on this thread so the counting allocator sees every batch.
        let (((index, document_paths), retained), peak) = crate::alloc_probe::peak_bytes(|| {
            crate::alloc_probe::retained_bytes(|| {
                build_lexical_index_batched(paths.iter().copied(), |batch| {
                    batch
                        .iter()
                        .map(|path| lexical_term_counts(path, &files))
                        .collect()
                })
            })
        });

        assert_eq!(document_paths.len(), 300);
        assert_eq!(index.document_count(), 300);
        // Growing the index's own tables costs up to its size again; holding
        // every parsed document at once cost six times the index.
        assert!(
            peak - retained <= retained,
            "building held {} transient bytes over the {retained}-byte index",
            peak - retained
        );
    }

    #[test]
    fn loading_identifier_vectors_holds_at_most_one_extra_copy_at_peak() {
        let dir = tempfile::tempdir().unwrap();
        let dims = 256;
        let keys: Vec<String> = (0..2_000).map(|i| format!("identifier_{i}")).collect();
        let data = rkyv_store::CacheData {
            dims: dims as u32,
            hashes: keys.iter().map(|key| format!("hash-{key}")).collect(),
            vectors: vec![0.25; keys.len() * dims],
            keys,
        };
        rkyv_store::save_cache(dir.path(), "identifier-embeddings", &data).unwrap();
        drop(data);

        let ((vectors, retained), peak) = crate::alloc_probe::peak_bytes(|| {
            crate::alloc_probe::retained_bytes(|| {
                load_identifier_vectors(dir.path(), "identifier-embeddings")
            })
        });

        assert_eq!(vectors.len(), 2_000);
        assert!(
            peak <= retained * 5 / 2,
            "loading peaked at {peak} bytes to keep {retained}"
        );
    }

    #[tokio::test]
    async fn identifier_index_shares_resident_vectors_instead_of_copying_them() {
        use crate::tools::semantic_identifiers::IndexData;

        let files = [("src/a.rs", "pub fn alpha() {}\npub fn beta() {}\n")];
        let (_repo, _ollama, server) = identifier_server(&files).await;
        let cache = server.ensure_project_cache().await.unwrap();
        server.ensure_identifier_index(&cache).await.unwrap();
        let owner = server.current_ref().await;
        *owner.identifier_index.write().await = None;
        *owner.identifier_source.write().await = None;

        server.ensure_identifier_index(&cache).await.unwrap();

        let index = owner.identifier_index.read().await.clone().unwrap();
        let resident = owner.identifier_vectors.get().unwrap().read().await;
        assert_eq!(index.docs.len(), 2);
        for i in 0..index.docs.len() {
            assert!(
                Arc::ptr_eq(&resident[&index.docs.get(i).text], index.vectors.vector(i)),
                "the rebuilt index copied the resident vector of {}",
                index.docs.get(i).name
            );
        }
    }

    #[tokio::test]
    async fn one_file_identifier_refresh_does_not_copy_unchanged_records_or_vectors() {
        let mut stable = String::new();
        for i in 0..500 {
            stable.push_str(&format!("pub fn stable_identifier_{i}() {{}}\n"));
        }
        let files = [
            ("src/stable.rs", stable.as_str()),
            ("src/changed.rs", "pub fn changed_before() {}\n"),
        ];
        let (repo, _ollama, server) = identifier_server(&files).await;
        let before = explore_identifier(&server, "stable_identifier_499", None).await;
        assert!(before.contains("stable_identifier_499"), "{before}");

        let owner = server.current_ref().await;
        let segments_before = owner
            .identifier_index
            .read()
            .await
            .as_ref()
            .unwrap()
            .clone();
        std::fs::write(
            repo.path().join("src/changed.rs"),
            "pub fn changed_after() {}\n",
        )
        .unwrap();
        server.build_tracker_callback().await(
            repo.path().to_path_buf(),
            vec!["src/changed.rs".to_string()],
        )
        .await
        .unwrap();

        let after = explore_identifier(&server, "changed_after", None).await;
        assert!(after.contains("changed_after"), "{after}");
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            loop {
                let ready = owner
                    .identifier_index
                    .read()
                    .await
                    .as_ref()
                    .is_some_and(|index| {
                        index.docs.files["src/changed.rs"]
                            .iter()
                            .any(|doc| doc.name == "changed_after")
                    });
                if ready {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("identifier update never published changed_after");
        let segments_after = owner
            .identifier_index
            .read()
            .await
            .as_ref()
            .unwrap()
            .clone();
        assert!(
            Arc::ptr_eq(
                &segments_before.docs.files["src/stable.rs"],
                &segments_after.docs.files["src/stable.rs"]
            ),
            "unchanged document segment was deep-copied"
        );
        assert!(
            Arc::ptr_eq(
                &segments_before.vectors.file_segments()["src/stable.rs"],
                &segments_after.vectors.file_segments()["src/stable.rs"]
            ),
            "unchanged vector segment was deep-copied"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pending_mass_delta_survives_small_delta_during_blocked_rebuild() {
        use crate::tools::semantic_search::{
            CachedSearchIndex, EmbedFn, SearchDocument, SemanticSearchOptions,
            semantic_code_search_owned,
        };
        use std::sync::atomic::{AtomicU32, Ordering};

        struct BlockingWalker {
            calls: AtomicU32,
            started: Arc<tokio::sync::Notify>,
            release: Arc<tokio::sync::Notify>,
        }
        impl WalkAndIndexFn for BlockingWalker {
            fn walk_and_index(
                &self,
                _root: &std::path::Path,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)>,
                        > + Send
                        + '_,
                >,
            > {
                let call = self.calls.fetch_add(1, Ordering::Relaxed);
                let started = Arc::clone(&self.started);
                let release = Arc::clone(&self.release);
                Box::pin(async move {
                    if call > 0 {
                        started.notify_one();
                        release.notified().await;
                    }
                    let docs = (0..25)
                        .map(|i| {
                            let content = if call > 0 && i < 6 {
                                "blocked mass refresh needle"
                            } else {
                                "original content"
                            };
                            SearchDocument::new(
                                format!("src/file_{i}.rs"),
                                String::new(),
                                vec![],
                                vec![],
                                content.to_string(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let vectors = vec![Some(vec![1.0, 0.0]); docs.len()];
                    Ok((docs, vectors))
                })
            }
        }
        struct FixedEmbedder;
        impl EmbedFn for FixedEmbedder {
            fn embed(
                &self,
                _texts: &[String],
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>,
            > {
                Box::pin(async { Ok(vec![vec![1.0, 0.0]]) })
            }
        }

        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let walker: Arc<dyn WalkAndIndexFn> = Arc::new(BlockingWalker {
            calls: AtomicU32::new(0),
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        });
        let cache = Arc::new(RwLock::new(None));
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        // The index keys on the canonical root (macOS /tmp is a symlink to /private/tmp).
        let root = std::fs::canonicalize(std::env::temp_dir()).unwrap();
        let options = SemanticSearchOptions {
            root_dir: root.clone(),
            query: "blocked mass refresh needle".to_string(),
            top_k: Some(5),
            semantic_weight: Some(0.0),
            keyword_weight: Some(1.0),
            min_semantic_score: None,
            min_keyword_score: Some(0.01),
            min_combined_score: None,
            require_keyword_match: Some(true),
            require_semantic_match: Some(false),
            include_globs: None,
            exclude_globs: None,
            recency_window_days: None,
            scope: None,
        };
        semantic_code_search_owned(
            options.clone(),
            &FixedEmbedder,
            Arc::clone(&walker),
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
        )
        .await
        .unwrap();
        let previous = cache.read().await.as_ref().cloned().unwrap();
        let mass_docs = (0..6)
            .map(|i| {
                SearchDocument::new(
                    format!("src/file_{i}.rs"),
                    String::new(),
                    vec![],
                    vec![],
                    "blocked mass refresh needle".to_string(),
                )
            })
            .collect::<Vec<_>>();
        assert!(!CachedSearchIndex::refresh_paths(
            cache.write().await.as_mut().unwrap(),
            &root,
            mass_docs,
            vec![Some(vec![1.0, 0.0]); 6],
            &[],
            1,
        ));
        generation.store(1, Ordering::Release);

        semantic_code_search_owned(
            options.clone(),
            &FixedEmbedder,
            Arc::clone(&walker),
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
        )
        .await
        .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(10), started.notified())
            .await
            .expect("the pending mass delta did not start a background rebuild");
        assert!(CachedSearchIndex::refresh_paths(
            cache.write().await.as_mut().unwrap(),
            &root,
            vec![SearchDocument::new(
                "src/file_24.rs".to_string(),
                String::new(),
                vec![],
                vec![],
                "concurrent small delta".to_string(),
            )],
            vec![Some(vec![1.0, 0.0])],
            &[],
            2,
        ));
        generation.store(2, Ordering::Release);
        release.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while previous.rebuild_in_progress.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("background rebuild did not finish");

        let result = semantic_code_search_owned(
            options,
            &FixedEmbedder,
            walker,
            Some(cache),
            Some(generation),
        )
        .await
        .unwrap();
        assert!(
            result.contains("src/file_0.rs"),
            "the small delta replaced the rebuild base and discarded pending mass changes: {result}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lexical_one_file_update_keeps_previous_index_available_to_readers() {
        let owned: Vec<_> = (0..100)
            .map(|i| {
                (
                    format!("src/file_{i}.rs"),
                    format!("pub fn lexical_symbol_{i}() {{}}\n"),
                )
            })
            .collect();
        let borrowed: Vec<_> = owned
            .iter()
            .map(|(path, content)| (path.as_str(), content.as_str()))
            .collect();
        let (repo, _ollama, server) = identifier_server(&borrowed).await;
        let mut args = serde_json::Map::new();
        args.insert("query".into(), json!("lexical_symbol_0"));
        server.handle_lexical_search(args.clone()).await.unwrap();

        std::fs::write(
            repo.path().join("src/file_0.rs"),
            "pub fn lexical_replacement() {}\n",
        )
        .unwrap();
        server.build_tracker_callback().await(
            repo.path().to_path_buf(),
            vec!["src/file_0.rs".to_string()],
        )
        .await
        .unwrap();

        let pause = crate::tools::lexical_search::test_seams::pause_next_update();
        let updating_server = server.clone();
        let update = tokio::spawn(async move {
            updating_server.handle_lexical_search(args).await.unwrap();
        });
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            pause.wait_until_entered(),
        )
        .await
        .expect("lexical delta did not reach the update seam");

        let cache = Arc::clone(&server.current_ref().await.lexical_search_cache);
        let reader_available =
            tokio::time::timeout(std::time::Duration::from_millis(100), cache.read())
                .await
                .is_ok();
        pause.release();
        update.await.unwrap();

        assert!(
            reader_available,
            "a one-file lexical update held the cache write lock during posting maintenance"
        );
    }

    #[tokio::test]
    async fn explore_identifier_keywords_rejects_context_only_matches() {
        let files = [
            (
                "src/grants.ts",
                "export const cascadeGrants = () => true;\n",
            ),
            (
                "src/foreign.ts",
                "// cascade is discussed here\nexport const foreignOrgId = 'foreign';\n",
            ),
            (
                "src/notes.ts",
                "export function clearNoteTranscriptions() {\n  // cascade cleanup\n  return true;\n}\n",
            ),
            (
                "src/one-line.ts",
                "export function unrelated() { return cascade(); }\nexport const arrowBody = () => cascade();\nexport const arrowComment = () => true; // cascade cleanup\n",
            ),
        ];
        let (_repo, _ollama, server) = identifier_server(&files).await;
        let mut args = serde_json::Map::new();
        args.insert("query".into(), json!("cascade"));
        args.insert("kind".into(), json!("identifiers"));
        args.insert("match".into(), json!("keywords"));
        args.insert("top_k".into(), json!(10));
        let result = server.dispatch("explore", args).await;
        assert_eq!(result.is_error, Some(false), "{}", text_of(&result));
        let text = text_of(&result);

        assert!(
            text.contains("cascadeGrants"),
            "named match missing:\n{text}"
        );
        assert!(
            !text.contains("foreignOrgId")
                && !text.contains("clearNoteTranscriptions")
                && !text.contains("unrelated")
                && !text.contains("arrowBody")
                && !text.contains("arrowComment"),
            "comment/body-only identifiers leaked into parser-backed keyword results:\n{text}"
        );
    }

    #[tokio::test]
    async fn outline_routes_files_to_skeleton_and_directories_to_tree() {
        let (_tmp, server) = facade_server();
        let mut args = serde_json::Map::new();
        args.insert("path".into(), json!("src/auth.rs"));
        let file = server.dispatch("outline", args).await;
        assert_eq!(file.is_error, Some(false), "{}", text_of(&file));
        assert!(
            text_of(&file).contains("verify_token"),
            "{}",
            text_of(&file)
        );

        let mut args = serde_json::Map::new();
        args.insert("path".into(), json!("src"));
        args.insert("depth".into(), json!(1));
        let dir = server.dispatch("outline", args).await;
        assert_eq!(dir.is_error, Some(false), "{}", text_of(&dir));
        assert!(text_of(&dir).contains("auth.rs"), "{}", text_of(&dir));
    }

    #[tokio::test]
    async fn impact_routes_by_what_and_requires_symbol() {
        let (_tmp, server) = facade_server();
        let mut args = serde_json::Map::new();
        args.insert("symbol".into(), json!("verify_token"));
        let users = server.dispatch("impact", args).await;
        assert_eq!(users.is_error, Some(false), "{}", text_of(&users));
        assert!(text_of(&users).contains("main.rs"), "{}", text_of(&users));

        let mut args = serde_json::Map::new();
        args.insert("what".into(), json!("cycles"));
        let cycles = server.dispatch("impact", args).await;
        assert_eq!(cycles.is_error, Some(false), "{}", text_of(&cycles));

        let missing = server.dispatch("impact", serde_json::Map::new()).await;
        assert_eq!(missing.is_error, Some(true));
        assert!(text_of(&missing).contains("symbol or diff is required"));

        let mut args = serde_json::Map::new();
        args.insert(
            "diff".into(),
            json!("--- a/src/auth.rs\n+++ b/src/auth.rs\n@@ -1,1 +1,1 @@\n-pub fn verify_token(t: &str) -> bool { t.len() > 3 }\n+pub fn verify_token(t: &str) -> bool { t.len() > 4 }\n"),
        );
        let ranked = server.dispatch("impact", args).await;
        assert_eq!(ranked.is_error, Some(false), "{}", text_of(&ranked));
        assert!(text_of(&ranked).contains("auth.rs"), "{}", text_of(&ranked));
    }

    #[tokio::test]
    async fn explore_requires_query_and_keywords_mode_needs_no_embeddings() {
        let (_tmp, server) = facade_server();
        let missing = server.dispatch("explore", serde_json::Map::new()).await;
        assert_eq!(missing.is_error, Some(true));
        assert!(text_of(&missing).contains("query is required"));

        let mut args = serde_json::Map::new();
        args.insert("query".into(), json!("verify_token"));
        args.insert("match".into(), json!("keywords"));
        let hits = server.dispatch("explore", args).await;
        assert_eq!(hits.is_error, Some(false), "{}", text_of(&hits));
        assert!(text_of(&hits).contains("auth.rs"), "{}", text_of(&hits));
    }

    #[tokio::test]
    async fn explore_clusters_query_selects_and_orders_only_relevant_topic_files() {
        let projection_files = (0..24).map(|i| {
            (
                format!("packages/platform/projections/projection_rebuild_{i:02}.rs"),
                format!(
                    "pub fn topic_alpha_projection_{i:02}() {{ /* event sourcing consumer checkpoint rebuild */ }}"
                ),
            )
        });
        let consumer_files = (0..36).map(|i| {
            (
                format!("packages/platform/consumers/checkpoint_recovery_{i:02}.rs"),
                format!(
                    "pub fn topic_alpha_consumer_{i:02}() {{ /* event sourcing consumer checkpoint recovery */ }}"
                ),
            )
        });
        let payment_files = (0..64).map(|i| {
            (
                format!("packages/domains/payments/payment_settlement_{i:02}.rs"),
                format!("pub fn settle_invoice_{i:02}() {{ /* payment ledger */ }}"),
            )
        });
        let (_tmp, _provider, server) =
            cluster_facade_server(projection_files.chain(consumer_files).chain(payment_files))
                .await;
        let args = serde_json::Map::from_iter([
            (
                "query".to_string(),
                json!("event sourcing projection rebuild after consumer falls behind"),
            ),
            ("kind".to_string(), json!("clusters")),
            ("max_clusters".to_string(), json!(4)),
            ("min_clusters".to_string(), json!(1)),
        ]);

        let result = server.dispatch("explore", args).await;
        assert_eq!(result.is_error, Some(false), "{}", text_of(&result));
        let output = text_of(&result);
        assert!(
            output.contains("projection_rebuild_") && output.contains("checkpoint_recovery_"),
            "query-relevant topic-A files were absent:\n{output}"
        );
        assert!(
            !output.contains("payments") && !output.contains("payment_settlement_"),
            "query-driven clusters leaked unrelated payment files:\n{output}"
        );
        let first_group = output
            .lines()
            .filter(|line| line.trim_start().starts_with('['))
            .nth(1)
            .expect("first cluster group");
        assert!(
            first_group.contains("projection"),
            "the most relevant group must be first, got {first_group:?}\n{output}"
        );
    }

    #[tokio::test]
    async fn explore_clusters_scoped_generated_paths_are_excluded() {
        for scope in ["contracts/gen", "src/generated/subdir"] {
            for query in [None, Some("projection rebuild")] {
                let (_tmp, provider, server) = cluster_facade_server([(
                    format!("{scope}/client.rs"),
                    "pub fn projection_rebuild() {}".to_string(),
                )])
                .await;
                let mut args = serde_json::Map::from_iter([
                    ("kind".to_string(), json!("clusters")),
                    ("path".to_string(), json!(scope)),
                ]);
                if let Some(query) = query {
                    args.insert("query".to_string(), json!(query));
                }
                let result = server.dispatch("explore", args).await;
                assert_eq!(result.is_error, Some(false), "{}", text_of(&result));
                assert_eq!(
                    text_of(&result),
                    "No supported source files found in the project.",
                    "scope={scope}, query={query:?}"
                );
                assert!(
                    provider.received_requests().await.unwrap().is_empty(),
                    "excluded files must not reach embedding or labeling"
                );
            }
        }
    }

    #[tokio::test]
    async fn explore_clusters_respects_max_tokens_at_line_boundaries_with_marker() {
        let files = (0..48).map(|i| {
            (
                format!(
                    "packages/topic_{}/very_long_cluster_candidate_file_{i:02}.rs",
                    i % 3
                ),
                format!("pub fn cluster_candidate_{i:02}() {{}}"),
            )
        });
        let (_tmp, _provider, server) = cluster_facade_server(files).await;
        let max_tokens = 75usize;
        let args = serde_json::Map::from_iter([
            ("query".to_string(), json!("")),
            ("kind".to_string(), json!("clusters")),
            ("max_tokens".to_string(), json!(max_tokens)),
            ("max_clusters".to_string(), json!(2)),
            ("min_clusters".to_string(), json!(1)),
        ]);

        let result = server.dispatch("explore", args).await;
        assert_eq!(result.is_error, Some(false), "{}", text_of(&result));
        let output = text_of(&result);
        assert!(
            output.chars().count() <= max_tokens * 4,
            "cluster output exceeded the {max_tokens}-token/{}-character budget: {} chars\n{output}",
            max_tokens * 4,
            output.chars().count()
        );
        assert!(
            output.contains("more files"),
            "capped cluster output must report omitted files:\n{output}"
        );
        for line in output.lines().filter(|line| line.contains("packages/")) {
            assert!(
                line.trim_end().ends_with(".rs"),
                "cluster output truncated a file line: {line:?}\n{output}"
            );
        }
        let explore = tool_definitions()
            .iter()
            .find(|tool| tool.name.as_ref() == "explore")
            .expect("explore definition");
        assert!(
            explore.input_schema["properties"]
                .get("max_tokens")
                .is_some(),
            "explore schema must advertise max_tokens for cluster output"
        );
    }

    #[tokio::test]
    async fn explore_clusters_without_query_keeps_whole_repo_map_under_cap() {
        let alpha = (0..12).map(|i| {
            (
                format!("packages/topics/alpha/alpha_{i:02}.rs"),
                format!("pub fn alpha_{i:02}() {{}}"),
            )
        });
        let beta = (0..12).map(|i| {
            (
                format!("packages/topics/beta/beta_{i:02}.rs"),
                format!("pub fn beta_{i:02}() {{}}"),
            )
        });
        let (_tmp, _provider, server) = cluster_facade_server(alpha.chain(beta)).await;
        let max_tokens = 120usize;

        for query in [None, Some("   ")] {
            let mut args = serde_json::Map::from_iter([
                ("kind".to_string(), json!("clusters")),
                ("max_tokens".to_string(), json!(max_tokens)),
                ("max_clusters".to_string(), json!(2)),
                ("min_clusters".to_string(), json!(1)),
            ]);
            if let Some(query) = query {
                args.insert("query".to_string(), json!(query));
            }

            let result = server.dispatch("explore", args).await;
            assert_eq!(result.is_error, Some(false), "{}", text_of(&result));
            let output = text_of(&result);
            assert!(
                output.contains("alpha") && output.contains("beta"),
                "no-query clusters must retain the whole-repo map:\n{output}"
            );
            assert!(
                output.chars().count() <= max_tokens * 4,
                "no-query cluster map exceeded the cap: {} chars\n{output}",
                output.chars().count()
            );
        }
        let explore = tool_definitions()
            .iter()
            .find(|tool| tool.name.as_ref() == "explore")
            .expect("explore definition");
        let required = explore.input_schema["required"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert!(
            !required.iter().any(|name| name == "query"),
            "explore schema cannot require query when kind=clusters supports a whole-repo map"
        );
    }

    #[tokio::test]
    async fn routed_attached_worktree_clusters_read_their_healed_label_without_primary_leakage() {
        use crate::tools::semantic_navigate::{
            LabelQuality, cluster_cache_key, load_label_cache_full,
        };
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let provider = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let count = request
                    .body_json::<serde_json::Value>()
                    .ok()
                    .and_then(|body| body["input"].as_array().map(Vec::len))
                    .unwrap_or(1);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; count]
                }))
            })
            .mount(&provider)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": {
                    "content": "[\"Attached Payment Flow\", \"Attached Refund Flow\"]"
                }
            })))
            .mount(&provider)
            .await;

        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        let canonical_worktree = worktree.path().canonicalize().unwrap();
        let mut payment_paths = Vec::new();
        for i in 0..6 {
            let relative = format!("packages/payments/service/payment_{i}.rs");
            let path = canonical_worktree.join(&relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, format!("pub fn payment_{i}() {{}}\n")).unwrap();
            payment_paths.push(relative);
        }
        for i in 0..5 {
            let relative = format!("packages/refunds/service/refund_{i}.rs");
            let path = canonical_worktree.join(&relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, format!("pub fn refund_{i}() {{}}\n")).unwrap();
        }

        let mut config = Config::from_env();
        config.ollama_host = provider.uri();
        config.ollama_chat_model = "test-chat-model".to_string();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);
        let attach = server
            .handle_attach_worktree(serde_json::Map::from_iter([(
                "path".to_string(),
                json!(canonical_worktree.to_string_lossy().into_owned()),
            )]))
            .await
            .unwrap();
        assert_eq!(attach.is_error, Some(false), "{}", text_of(&attach));
        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_worktree);
        let session = server.with_session(ref_id);

        let request_args = || {
            serde_json::Map::from_iter([
                ("query".to_string(), json!("payments")),
                ("kind".to_string(), json!("clusters")),
                ("path".to_string(), json!(".")),
                ("max_clusters".to_string(), json!(4)),
                ("min_clusters".to_string(), json!(1)),
            ])
        };
        let first = session.dispatch("explore", request_args()).await;
        assert_eq!(first.is_error, Some(false), "{}", text_of(&first));
        assert!(
            text_of(&first).contains("[payments/service]"),
            "first routed clusters response did not use the worktree files: {}",
            text_of(&first)
        );

        let path_refs: Vec<&str> = payment_paths.iter().map(String::as_str).collect();
        let key = cluster_cache_key(&path_refs);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let cache = load_label_cache_full(&canonical_worktree);
                if cache.get(&key).is_some_and(|entry| {
                    entry.quality == LabelQuality::Llm && entry.label == "Attached Payment Flow"
                }) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("attached-ref clusters heal did not complete");

        let second = session.dispatch("explore", request_args()).await;
        let second_text = text_of(&second);
        assert_eq!(second.is_error, Some(false), "{second_text}");
        assert!(
            second_text.contains("[Attached Payment Flow] (6 files)"),
            "second routed clusters response did not read the healed worktree label: {second_text}"
        );
        assert!(
            second_text.contains("packages/payments/service/payment_0.rs")
                && second_text.contains("packages/payments/service/payment_5.rs"),
            "healed node did not preserve the flat worktree file group: {second_text}"
        );
        assert!(
            !load_label_cache_full(primary.path()).contains_key(&key),
            "attached-ref label cache leaked into the primary ref"
        );
    }

    /// `kind: identifiers` with `match: keywords` must rank by keyword coverage
    /// only: with a mock embedder that returns the same vector for everything,
    /// semantic similarity is 100% for every identifier, so only keyword-only
    /// weighting puts the exact name first with a score equal to its coverage.
    #[tokio::test]
    async fn explore_identifiers_with_keywords_ranks_by_keyword_only() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};
        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|req: &Request| {
                let body: serde_json::Value = req.body_json().unwrap_or_default();
                let n = body["input"].as_array().map_or(1, |a| a.len());
                let vecs: Vec<Vec<f32>> = (0..n).map(|_| vec![0.6, 0.8, 0.0]).collect();
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "embeddings": vecs }))
            })
            .mount(&ollama)
            .await;
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(
            tmp.path().join("src/auth.rs"),
            "pub fn verify_token(t: &str) -> bool { t.len() > 3 }\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("src/util.rs"),
            "pub fn unrelated_helper() {}\n",
        )
        .unwrap();
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(tmp.path().to_path_buf(), config);

        let mut args = serde_json::Map::new();
        args.insert("query".into(), json!("verify token"));
        args.insert("kind".into(), json!("identifiers"));
        args.insert("match".into(), json!("keywords"));
        let hits = server.dispatch("explore", args).await;
        assert_eq!(hits.is_error, Some(false), "{}", text_of(&hits));
        let text = text_of(&hits);
        let first = text
            .lines()
            .find(|l| l.trim_start().starts_with("1. "))
            .unwrap_or("");
        assert!(first.contains("verify_token"), "{text}");
        assert!(
            text.contains("Score: 100% | Semantic: 100% | Keyword: 100%"),
            "keyword-only weighting must make the score equal the keyword coverage:\n{text}"
        );
    }

    #[tokio::test]
    async fn check_and_worktrees_route_by_argument() {
        let (_tmp, server) = facade_server();
        let mut args = serde_json::Map::new();
        args.insert("what".into(), json!("embeddings"));
        let audit = server.dispatch("check", args).await;
        assert_eq!(audit.is_error, Some(false), "{}", text_of(&audit));
        assert!(
            text_of(&audit).contains("Embedding quality"),
            "{}",
            text_of(&audit)
        );

        let list = server.dispatch("worktrees", serde_json::Map::new()).await;
        assert_eq!(list.is_error, Some(false), "{}", text_of(&list));
        assert!(text_of(&list).contains("primary"), "{}", text_of(&list));
    }

    #[tokio::test]
    async fn pre_facade_names_still_dispatch() {
        let (_tmp, server) = facade_server();
        let mut args = serde_json::Map::new();
        args.insert("file_path".into(), json!("src/auth.rs"));
        let legacy = server.dispatch("get_file_skeleton", args).await;
        assert_eq!(legacy.is_error, Some(false), "{}", text_of(&legacy));
        assert!(text_of(&legacy).contains("verify_token"));
    }

    #[tokio::test]
    async fn dispatch_unknown_tool_returns_error() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("nonexistent_tool", args).await;
        assert_eq!(
            result.is_error,
            Some(true),
            "unknown tool should return is_error=true"
        );
        let text = result
            .content
            .first()
            .and_then(|c| match &c.raw {
                RawContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .unwrap_or("");
        assert!(
            text.contains("Unknown tool"),
            "expected 'Unknown tool' in error text, got: {}",
            text
        );
    }

    #[test]
    fn get_str_extracts_string() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), json!("value"));
        assert_eq!(
            ContextPlusServer::get_str(&args, "key"),
            Some("value".to_string())
        );
        assert_eq!(ContextPlusServer::get_str(&args, "missing"), None);
    }

    #[test]
    fn get_str_returns_none_for_non_string() {
        let mut args = serde_json::Map::new();
        args.insert("num".to_string(), json!(42));
        assert_eq!(ContextPlusServer::get_str(&args, "num"), None);
    }

    #[test]
    fn get_str_or_returns_default_when_missing() {
        let args = serde_json::Map::new();
        assert_eq!(
            ContextPlusServer::get_str_or(&args, "missing", "fallback"),
            "fallback"
        );
    }

    #[test]
    fn get_str_or_returns_value_when_present() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), json!("actual"));
        assert_eq!(
            ContextPlusServer::get_str_or(&args, "key", "fallback"),
            "actual"
        );
    }

    #[test]
    fn get_usize_extracts_number() {
        let mut args = serde_json::Map::new();
        args.insert("n".to_string(), json!(42));
        assert_eq!(ContextPlusServer::get_usize(&args, "n"), Some(42));
        assert_eq!(ContextPlusServer::get_usize(&args, "missing"), None);
    }

    #[test]
    fn get_f64_extracts_float() {
        let mut args = serde_json::Map::new();
        args.insert("f".to_string(), json!(2.78));
        let val = ContextPlusServer::get_f64(&args, "f").unwrap();
        assert!((val - 2.78).abs() < f64::EPSILON);
        assert_eq!(ContextPlusServer::get_f64(&args, "missing"), None);
    }

    #[test]
    fn get_bool_extracts_boolean() {
        let mut args = serde_json::Map::new();
        args.insert("b".to_string(), json!(true));
        assert_eq!(ContextPlusServer::get_bool(&args, "b"), Some(true));
        args.insert("b".to_string(), json!(false));
        assert_eq!(ContextPlusServer::get_bool(&args, "b"), Some(false));
        assert_eq!(ContextPlusServer::get_bool(&args, "missing"), None);
    }

    #[tokio::test]
    async fn resolve_root_uses_server_root_when_no_arg() {
        let server = test_server();
        let args = serde_json::Map::new();
        let root = server.resolve_root(&args).await;
        assert_eq!(root, server.state.root_dir);
    }

    #[tokio::test]
    async fn resolve_root_rejects_path_outside_server_root() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert("rootDir".to_string(), json!("/etc/passwd"));
        let root = server.resolve_root(&args).await;
        // Should fall back to server root since /etc/passwd is outside
        assert_eq!(root, server.state.root_dir);
    }

    #[test]
    fn ok_text_creates_success_result() {
        let result = ContextPlusServer::ok_text("hello".to_string());
        assert_eq!(result.is_error, Some(false));
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn err_text_creates_error_result() {
        let result = ContextPlusServer::err_text("oops".to_string());
        assert_eq!(result.is_error, Some(true));
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn make_tool_sets_required_params() {
        let tool = make_tool(
            "test_tool",
            "A test tool",
            &[
                ("required_param", "string", true, "A required param"),
                ("optional_param", "integer", false, "An optional param"),
            ],
        );
        assert_eq!(tool.name.as_ref(), "test_tool");
        assert_eq!(tool.description.as_deref(), Some("A test tool"));

        let schema = tool.input_schema.as_ref();
        let required = schema.get("required").and_then(|v| v.as_array()).unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].as_str(), Some("required_param"));
    }

    // --- ProjectCache tests ---

    /// Build a ContextPlusServer rooted at the given directory with a custom TTL.
    fn server_with_root_and_ttl(root: PathBuf, cache_ttl_secs: u64) -> ContextPlusServer {
        let mut config = Config::from_env();
        config.cache_ttl_secs = cache_ttl_secs;
        config.embed_tracker_mode = TrackerMode::Off;
        ContextPlusServer::new(root, config)
    }

    /// Create a temp dir with a few known files and return (TempDir, server).
    fn setup_cache_test(ttl_secs: u64) -> (tempfile::TempDir, ContextPlusServer) {
        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(tmp.path().join("hello.txt"), "line1\nline2\nline3\n").unwrap();
        std::fs::write(tmp.path().join("world.rs"), "fn main() {}\n").unwrap();
        let sub = tmp.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("nested.txt"), "nested content\n").unwrap();
        let server = server_with_root_and_ttl(tmp.path().to_path_buf(), ttl_secs);
        (tmp, server)
    }

    #[tokio::test]
    async fn ensure_project_cache_creates_cache_on_first_call() {
        let (_tmp, server) = setup_cache_test(300);

        // Cache starts as None
        {
            let guard = server.state.project_cache.read().await;
            assert!(guard.is_none(), "cache should be None before first call");
        }

        let cache = server.ensure_project_cache().await.unwrap();

        // Should have found our test files
        assert!(
            !cache.file_entries.is_empty(),
            "file_entries should not be empty"
        );
        assert!(
            !cache.file_content.is_empty(),
            "file_content should not be empty"
        );

        // Verify specific files are in file_content
        let has_hello = cache.file_content.contains_key("hello.txt");
        let has_world = cache.file_content.contains_key("world.rs");
        assert!(has_hello, "cache should contain hello.txt");
        assert!(has_world, "cache should contain world.rs");

        // Verify content: the raw string should contain all three lines
        let hello = cache.file_content["hello.txt"].as_str();
        let hello_lines: Vec<&str> = hello.lines().collect();
        assert_eq!(hello_lines, ["line1", "line2", "line3"]);

        // Arc refcount test: after 3 clones the strong count should be 4
        let arc = Arc::clone(&cache.file_content["hello.txt"]);
        let arc2 = Arc::clone(&arc);
        let arc3 = Arc::clone(&arc2);
        assert_eq!(
            Arc::strong_count(&arc3),
            4,
            "expected strong_count == 4 after 3 extra clones"
        );
        drop(arc);
        drop(arc2);
        drop(arc3);

        // Cache should now be populated in shared state
        {
            let guard = server.state.project_cache.read().await;
            assert!(
                guard.is_some(),
                "cache should be populated after first call"
            );
        }
    }

    #[tokio::test]
    async fn ensure_project_cache_returns_cached_data_on_second_call() {
        let (_tmp, server) = setup_cache_test(300);

        let cache1 = server.ensure_project_cache().await.unwrap();
        let refresh1 = cache1.last_refresh;

        let cache2 = server.ensure_project_cache().await.unwrap();
        let refresh2 = cache2.last_refresh;

        // Second call should return the same cached data (same refresh timestamp)
        assert_eq!(
            refresh1, refresh2,
            "second call should return cached data with same last_refresh"
        );
        assert_eq!(
            cache1.file_entries.len(),
            cache2.file_entries.len(),
            "cached file_entries count should be identical"
        );
    }

    #[tokio::test]
    async fn ensure_project_cache_respects_ttl_expiry() {
        // Use a TTL of 0 so the cache is always expired
        let (_tmp, server) = setup_cache_test(0);

        let cache1 = server.ensure_project_cache().await.unwrap();
        let refresh1 = cache1.last_refresh;

        // With TTL=0, the next call should rebuild (new Instant)
        // Small sleep to ensure Instant::now() differs
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let cache2 = server.ensure_project_cache().await.unwrap();
        let refresh2 = cache2.last_refresh;

        assert_ne!(
            refresh1, refresh2,
            "expired cache should be rebuilt with a new last_refresh"
        );
    }

    #[tokio::test]
    async fn ensure_project_cache_ignores_ttl_while_tracker_is_running() {
        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(tmp.path().join("hello.txt"), "hello\n").unwrap();
        let mut config = Config::from_env();
        config.cache_ttl_secs = 0;
        config.embed_tracker_mode = TrackerMode::Lazy;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(tmp.path().to_path_buf(), config);
        server.ensure_tracker_started().await;
        assert!(
            server
                .current_ref()
                .await
                .tracker_handle
                .lock()
                .unwrap()
                .is_some()
        );

        let first = server.ensure_project_cache().await.unwrap();
        let second = server.ensure_project_cache().await.unwrap();

        assert!(
            Arc::ptr_eq(&first, &second),
            "a running tracker should keep the project cache authoritative past its TTL"
        );
    }

    #[tokio::test]
    async fn invalidate_project_cache_sets_cache_to_none() {
        let (_tmp, server) = setup_cache_test(300);

        // Populate the cache
        server.ensure_project_cache().await.unwrap();
        {
            let guard = server.state.project_cache.read().await;
            assert!(
                guard.is_some(),
                "cache should be populated before invalidation"
            );
        }

        // Invalidate
        server.invalidate_project_cache().await;
        {
            let guard = server.state.project_cache.read().await;
            assert!(guard.is_none(), "cache should be None after invalidation");
        }
    }

    #[tokio::test]
    async fn ensure_project_cache_rebuilds_after_invalidation() {
        let (_tmp, server) = setup_cache_test(300);

        // Populate, invalidate, then rebuild
        let cache1 = server.ensure_project_cache().await.unwrap();
        let refresh1 = cache1.last_refresh;

        server.invalidate_project_cache().await;

        // Small sleep to ensure different Instant
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let cache2 = server.ensure_project_cache().await.unwrap();
        let refresh2 = cache2.last_refresh;

        assert_ne!(
            refresh1, refresh2,
            "cache should be rebuilt with new last_refresh after invalidation"
        );

        // Rebuilt cache should still find the same files
        assert!(
            cache2.file_content.contains_key("hello.txt"),
            "rebuilt cache should contain hello.txt"
        );
        assert!(
            cache2.file_content.contains_key("world.rs"),
            "rebuilt cache should contain world.rs"
        );
    }

    #[tokio::test]
    async fn lexical_search_reuses_cache_until_generation_or_project_cache_changes() {
        let (_tmp, server) = setup_cache_test(300);
        let mut args = serde_json::Map::new();
        args.insert("query".to_string(), json!("hello"));

        server.handle_lexical_search(args.clone()).await.unwrap();
        let ref_index = server.current_ref().await;
        let first = ref_index
            .lexical_search_cache
            .read()
            .await
            .as_ref()
            .cloned()
            .expect("first lexical search should populate the per-ref cache");

        server.handle_lexical_search(args.clone()).await.unwrap();
        let second = ref_index
            .lexical_search_cache
            .read()
            .await
            .as_ref()
            .cloned()
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        ref_index
            .cache_generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        server.handle_lexical_search(args.clone()).await.unwrap();
        let after_generation = ref_index
            .lexical_search_cache
            .read()
            .await
            .as_ref()
            .cloned()
            .unwrap();
        assert!(!Arc::ptr_eq(&second, &after_generation));
        server.handle_lexical_search(args.clone()).await.unwrap();
        let after_generation_reuse = ref_index
            .lexical_search_cache
            .read()
            .await
            .as_ref()
            .cloned()
            .unwrap();
        assert!(Arc::ptr_eq(&after_generation, &after_generation_reuse));

        server.invalidate_project_cache().await;
        server.handle_lexical_search(args.clone()).await.unwrap();
        let after_project_cache = ref_index
            .lexical_search_cache
            .read()
            .await
            .as_ref()
            .cloned()
            .unwrap();
        assert!(!Arc::ptr_eq(&after_generation_reuse, &after_project_cache));
        server.handle_lexical_search(args).await.unwrap();
        let after_project_cache_reuse = ref_index
            .lexical_search_cache
            .read()
            .await
            .as_ref()
            .cloned()
            .unwrap();
        assert!(Arc::ptr_eq(
            &after_project_cache,
            &after_project_cache_reuse
        ));
    }

    fn expired_empty_identifier_index(file_count: usize) -> Arc<IdentifierIndex> {
        Arc::new(IdentifierIndex {
            docs: Vec::new().into(),
            vectors: IdentifierVectorIndex::empty(),
            dims: 0,
            file_count,
            built_at: Instant::now()
                - std::time::Duration::from_secs(IDENTIFIER_INDEX_TTL_SECS + 1),
        })
    }

    #[tokio::test]
    async fn ensure_identifier_index_ignores_ttl_while_tracker_is_running() {
        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(tmp.path().join("notes.txt"), "hello\n").unwrap();
        let mut config = Config::from_env();
        config.embed_tracker_mode = TrackerMode::Lazy;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(tmp.path().to_path_buf(), config);
        server.ensure_tracker_started().await;
        let cache = server.ensure_project_cache().await.unwrap();
        let file_count = cache
            .file_entries
            .iter()
            .filter(|entry| !entry.is_directory)
            .count();
        let expired = expired_empty_identifier_index(file_count);
        *server.current_ref().await.identifier_index.write().await = Some(Arc::clone(&expired));

        let actual = server.ensure_identifier_index(&cache).await.unwrap();

        assert!(
            Arc::ptr_eq(&expired, &actual),
            "a running tracker should keep the identifier index authoritative past its TTL"
        );
    }

    #[tokio::test]
    async fn ensure_identifier_index_uses_ttl_when_tracker_is_off() {
        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        std::fs::write(tmp.path().join("notes.txt"), "hello\n").unwrap();
        let mut config = Config::from_env();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(tmp.path().to_path_buf(), config);
        let cache = server.ensure_project_cache().await.unwrap();
        let file_count = cache
            .file_entries
            .iter()
            .filter(|entry| !entry.is_directory)
            .count();
        let expired = expired_empty_identifier_index(file_count);
        *server.current_ref().await.identifier_index.write().await = Some(Arc::clone(&expired));

        let actual = server.ensure_identifier_index(&cache).await.unwrap();

        assert!(
            !Arc::ptr_eq(&expired, &actual),
            "tracker-off mode should retain TTL fallback invalidation"
        );
    }

    async fn attached_ref_with_slow_embedder(
        tracker_mode: TrackerMode,
    ) -> (
        tempfile::TempDir,
        tempfile::TempDir,
        wiremock::MockServer,
        ContextPlusServer,
        crate::ref_index::RefId,
    ) {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let body: serde_json::Value = request.body_json().unwrap_or_default();
                let count = body["input"].as_array().map_or(1, Vec::len);
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(2))
                    .set_body_json(serde_json::json!({
                        "embeddings": vec![vec![0.6, 0.8]; count]
                    }))
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().expect("failed to create primary temp dir");
        let attached = tempfile::tempdir().expect("failed to create attached temp dir");
        std::fs::write(attached.path().join("base.rs"), "fn baseline() {}\n").unwrap();

        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.cache_ttl_secs = 0;
        config.embed_tracker_mode = tracker_mode;
        config.embed_tracker_debounce_ms = 60_000;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);

        let canonical = attached.path().canonicalize().unwrap();
        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(canonical.to_string_lossy().to_string()),
        );
        let result = server.handle_attach_worktree(args).await.unwrap();
        assert_eq!(result.is_error, Some(false), "{}", text_of(&result));

        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical);
        (primary, attached, ollama, server, ref_id)
    }

    async fn attached_impact(
        server: &ContextPlusServer,
        attached: &std::path::Path,
        symbol: &str,
    ) -> String {
        let mut args = serde_json::Map::new();
        args.insert("symbol_name".to_string(), json!(symbol));
        args.insert(
            "path".to_string(),
            json!(attached.to_string_lossy().to_string()),
        );
        text_of(&server.handle_blast_radius(args).await.unwrap())
    }

    async fn wait_for_embed_request_count(ollama: &wiremock::MockServer, expected: usize) {
        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        loop {
            let actual = ollama.received_requests().await.unwrap_or_default().len();
            if actual >= expected {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "tracker callback never reached embedding request {expected}; observed {actual}"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    async fn held_embedding_endpoint() -> (
        String,
        tokio::sync::oneshot::Receiver<()>,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind held embedding endpoint");
        let address = listener.local_addr().expect("held endpoint address");
        let (request_started_tx, request_started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept embedding request");
            let mut request = Vec::new();
            let mut content_length = None;
            let mut header_end = None;
            let mut buffer = [0_u8; 4096];
            loop {
                let read = socket
                    .read(&mut buffer)
                    .await
                    .expect("read embedding request");
                assert!(read > 0, "embedding request closed before its body arrived");
                request.extend_from_slice(&buffer[..read]);

                if header_end.is_none()
                    && let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n")
                {
                    header_end = Some(end + 4);
                    let headers = String::from_utf8_lossy(&request[..end]);
                    content_length = headers.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    });
                }

                if let Some(end) = header_end
                    && request.len() >= end + content_length.unwrap_or(0)
                {
                    break;
                }
            }

            request_started_tx
                .send(())
                .expect("embedding request observer dropped");
            release_rx.await.expect("embedding release sender dropped");

            let body = r#"{"embeddings":[[0.6,0.8]]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write embedding response");
        });

        (
            format!("http://{address}"),
            request_started_rx,
            release_tx,
            task,
        )
    }

    #[tokio::test]
    async fn attached_tracker_keeps_impact_fresh_for_create_edit_rename_and_delete() {
        let (_primary, attached, ollama, server, ref_id) =
            attached_ref_with_slow_embedder(TrackerMode::Eager).await;
        let canonical = attached.path().canonicalize().unwrap();
        let ref_index = server
            .state
            .ref_index(ref_id)
            .await
            .expect("attached ref present");
        assert!(
            ref_index.tracker_handle.lock().unwrap().is_some(),
            "attached eager ref must have a running tracker"
        );

        let symbol = "trackerFreshUniqueSymbol42";
        let initial = attached_impact(&server, &canonical, symbol).await;
        assert!(initial.contains("has no references"), "{initial}");

        let created = canonical.join("created.rs");
        std::fs::write(&created, format!("fn {symbol}() {{}}\n")).unwrap();
        let callback = server.with_session(ref_id).build_tracker_callback().await;
        let create_refresh = callback(canonical.clone(), vec!["created.rs".to_string()]);
        wait_for_embed_request_count(&ollama, 1).await;
        let created_impact = attached_impact(&server, &canonical, symbol).await;
        create_refresh.abort();
        assert!(
            created_impact.contains("created.rs"),
            "impact stayed on the pre-create project cache while embedding was in flight:\n{created_impact}"
        );

        std::fs::write(
            &created,
            format!("fn {symbol}() {{}}\nfn caller() {{ {symbol}(); }} // edit-visible\n"),
        )
        .unwrap();
        let edit_refresh = callback(canonical.clone(), vec!["created.rs".to_string()]);
        wait_for_embed_request_count(&ollama, 2).await;
        let edited_impact = attached_impact(&server, &canonical, symbol).await;
        edit_refresh.abort();
        assert!(
            edited_impact.contains("edit-visible"),
            "impact stayed on the pre-edit project cache while embedding was in flight:\n{edited_impact}"
        );

        let renamed = canonical.join("renamed.rs");
        std::fs::rename(&created, &renamed).unwrap();
        let rename_refresh = callback(
            canonical.clone(),
            vec!["created.rs".to_string(), "renamed.rs".to_string()],
        );
        wait_for_embed_request_count(&ollama, 3).await;
        let renamed_impact = attached_impact(&server, &canonical, symbol).await;
        rename_refresh.abort();
        assert!(
            renamed_impact.contains("renamed.rs") && !renamed_impact.contains("  created.rs:"),
            "impact did not reflect the rename while embedding was in flight:\n{renamed_impact}"
        );

        std::fs::remove_file(&renamed).unwrap();
        callback(canonical.clone(), vec!["renamed.rs".to_string()])
            .await
            .unwrap();
        let deleted_impact = attached_impact(&server, &canonical, symbol).await;
        assert!(
            deleted_impact.contains("has no references"),
            "impact retained a deleted file:\n{deleted_impact}"
        );
    }

    #[tokio::test]
    async fn attached_tracker_bulk_replacement_is_visible_before_embedding_finishes() {
        let (_primary, attached, ollama, server, ref_id) =
            attached_ref_with_slow_embedder(TrackerMode::Eager).await;
        let canonical = attached.path().canonicalize().unwrap();
        let symbol = "bulkCheckoutUniqueSymbol73";

        let mut old_paths = Vec::new();
        for index in 0..32 {
            let relative = format!("old_{index:02}.rs");
            std::fs::write(
                canonical.join(&relative),
                format!("fn old_{index:02}() {{}}\n"),
            )
            .unwrap();
            old_paths.push(relative);
        }
        let initial = attached_impact(&server, &canonical, symbol).await;
        assert!(initial.contains("has no references"), "{initial}");

        for relative in &old_paths {
            std::fs::remove_file(canonical.join(relative)).unwrap();
        }
        let mut changed_paths = old_paths;
        for index in 0..32 {
            let relative = format!("new_{index:02}.rs");
            std::fs::write(
                canonical.join(&relative),
                format!("fn checkout_{index:02}() {{ {symbol}(); }}\n"),
            )
            .unwrap();
            changed_paths.push(relative);
        }

        let callback = server.with_session(ref_id).build_tracker_callback().await;
        let refresh = callback(canonical.clone(), changed_paths);
        wait_for_embed_request_count(&ollama, 1).await;
        let impact = attached_impact(&server, &canonical, symbol).await;
        refresh.abort();

        assert!(
            impact.contains("new_00.rs") && impact.contains("new_31.rs"),
            "impact served the pre-checkout project cache while bulk embedding was in flight:\n{impact}"
        );
        assert!(
            !impact.contains("  old_00.rs:"),
            "impact retained files removed by the bulk replacement:\n{impact}"
        );
    }

    #[tokio::test]
    async fn attached_ref_ttl_fallback_and_unchanged_reuse_survive_tracker_start() {
        let (_primary, attached, ollama, server, ref_id) =
            attached_ref_with_slow_embedder(TrackerMode::Lazy).await;
        let canonical = attached.path().canonicalize().unwrap();
        let ref_index = server
            .state
            .ref_index(ref_id)
            .await
            .expect("attached ref present");
        assert!(ref_index.tracker_handle.lock().unwrap().is_none());

        let initial = attached_impact(&server, &canonical, "ttlFallbackSymbol").await;
        assert!(initial.contains("has no references"), "{initial}");
        std::fs::write(
            canonical.join("base.rs"),
            "fn baseline() {}\nfn ttlFallbackSymbol() {}\n",
        )
        .unwrap();
        let refreshed = attached_impact(&server, &canonical, "ttlFallbackSymbol").await;
        assert!(
            refreshed.contains("base.rs"),
            "an attached ref without a tracker must refresh through its TTL fallback:\n{refreshed}"
        );

        server.ensure_tracker_started_for(ref_id).await;
        assert!(ref_index.tracker_handle.lock().unwrap().is_some());
        let first = server.ensure_project_cache_for(&ref_index).await.unwrap();
        let second = server.ensure_project_cache_for(&ref_index).await.unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "an unchanged tree with a healthy tracker must reuse its project cache"
        );

        std::fs::write(
            canonical.join("base.rs"),
            "fn baseline() {}\nfn ttlFallbackSymbol() {}\nfn afterTrackerStart() {}\n",
        )
        .unwrap();
        let callback = server.with_session(ref_id).build_tracker_callback().await;
        let refresh = callback(canonical.clone(), vec!["base.rs".to_string()]);
        wait_for_embed_request_count(&ollama, 1).await;
        let impact = attached_impact(&server, &canonical, "afterTrackerStart").await;
        refresh.abort();
        assert!(
            impact.contains("base.rs"),
            "tracker TTL bypass served stale content while embedding was in flight:\n{impact}"
        );
    }

    #[tokio::test]
    async fn attached_tracker_start_refreshes_cache_built_before_tracking() {
        let (_primary, attached, _ollama, server, ref_id) =
            attached_ref_with_slow_embedder(TrackerMode::Lazy).await;
        let canonical = attached.path().canonicalize().unwrap();
        let ref_index = server
            .state
            .ref_index(ref_id)
            .await
            .expect("attached ref present");
        assert!(ref_index.tracker_handle.lock().unwrap().is_none());

        let stale = server.ensure_project_cache_for(&ref_index).await.unwrap();
        assert_eq!(stale.file_content["base.rs"].as_str(), "fn baseline() {}\n");

        std::fs::write(
            canonical.join("base.rs"),
            "fn baseline() {}\n// pretrackeruniquetoken\n",
        )
        .unwrap();

        let mut args = serde_json::Map::new();
        args.insert("query".to_string(), json!("pretrackeruniquetoken"));
        let search = server
            .with_session(ref_id)
            .handle_lexical_search(args)
            .await
            .unwrap();
        let search_text = text_of(&search);
        assert!(
            search_text.contains("base.rs"),
            "the tool that started the attached tracker searched the pre-tracker snapshot:\n{search_text}"
        );

        let refreshed = ref_index
            .project_cache
            .read()
            .await
            .as_ref()
            .cloned()
            .expect("lexical search should populate the attached project cache");
        assert!(
            refreshed.file_content["base.rs"].contains("pretrackeruniquetoken"),
            "the first cache access after tracker startup reused the pre-tracker snapshot"
        );
        assert!(
            !Arc::ptr_eq(&stale, &refreshed),
            "tracker startup must replace a cache it did not observe being built"
        );

        let unchanged = server.ensure_project_cache_for(&ref_index).await.unwrap();
        assert!(
            Arc::ptr_eq(&refreshed, &unchanged),
            "the refreshed snapshot should remain authoritative when the tree is unchanged"
        );
    }

    #[tokio::test]
    async fn attached_real_watcher_covers_new_directory_while_embedding_is_in_flight() {
        let (ollama_host, request_started, release_embedding, embedding_server) =
            held_embedding_endpoint().await;
        let primary = tempfile::tempdir().expect("failed to create primary temp dir");
        let attached = tempfile::tempdir().expect("failed to create attached temp dir");
        std::fs::write(attached.path().join("base.rs"), "fn baseline() {}\n").unwrap();

        let mut config = Config::from_env();
        config.ollama_host = ollama_host;
        config.cache_ttl_secs = 0;
        config.embed_tracker_mode = TrackerMode::Eager;
        config.embed_tracker_debounce_ms = 100;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);
        let canonical = attached.path().canonicalize().unwrap();
        let mut attach_args = serde_json::Map::new();
        attach_args.insert(
            "path".to_string(),
            json!(canonical.to_string_lossy().to_string()),
        );
        let attached_result = server.handle_attach_worktree(attach_args).await.unwrap();
        assert_eq!(
            attached_result.is_error,
            Some(false),
            "{}",
            text_of(&attached_result)
        );

        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical);
        let ref_index = server
            .state
            .ref_index(ref_id)
            .await
            .expect("attached ref present");
        let initial = server.ensure_project_cache_for(&ref_index).await.unwrap();

        std::fs::write(
            canonical.join("base.rs"),
            "fn baseline() {}\nfn starts_slow_embedding() {}\n",
        )
        .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(3), request_started)
            .await
            .expect("real watcher did not start embedding within the debounce bound")
            .expect("held embedding endpoint stopped before receiving a request");

        let new_dir = canonical.join("new_module");
        std::fs::create_dir(&new_dir).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let directory_refresh = attached_impact(&server, &canonical, "directory_probe").await;
        assert!(
            directory_refresh.contains("has no references"),
            "unexpected probe result: {directory_refresh}"
        );
        let after_directory = ref_index
            .project_cache
            .read()
            .await
            .as_ref()
            .cloned()
            .expect("directory query should rebuild the project cache");
        assert!(
            !Arc::ptr_eq(&initial, &after_directory),
            "directory event was not consumed before writing inside the new directory"
        );

        let symbol = "unwatched_directory_symbol_91";
        std::fs::write(
            new_dir.join("late.rs"),
            format!("fn {symbol}() {{}}\nfn caller() {{ {symbol}(); }}\n"),
        )
        .unwrap();

        let observed = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let impact = attached_impact(&server, &canonical, symbol).await;
                if impact.contains("new_module/late.rs") {
                    break impact;
                }
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
        })
        .await;

        release_embedding
            .send(())
            .expect("embedding request finished before the test released it");
        tokio::time::timeout(std::time::Duration::from_secs(1), embedding_server)
            .await
            .expect("held embedding endpoint did not shut down")
            .expect("held embedding endpoint task panicked");

        let impact = observed.expect(
            "impact did not find the file in the new directory within the debounce bound while embedding remained in flight",
        );
        assert!(
            impact.contains(symbol),
            "unexpected impact output: {impact}"
        );
    }

    #[tokio::test]
    async fn tracker_callback_does_not_bump_generation_for_unchanged_content() {
        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        let content = "fn unchanged() {}\n";
        std::fs::write(tmp.path().join("unchanged.rs"), content).unwrap();
        let mut config = Config::from_env();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(tmp.path().to_path_buf(), config);
        server
            .current_ref()
            .await
            .embedding_cache
            .write()
            .await
            .insert(
                "unchanged.rs".to_string(),
                CacheEntry {
                    hash: crate::core::parser::hash_content(content),
                    vector: vec![1.0, 0.0],
                },
            );
        let generation = Arc::clone(&server.current_ref().await.cache_generation);
        let callback = server.build_tracker_callback().await;

        let result = callback(tmp.path().to_path_buf(), vec!["unchanged.rs".to_string()])
            .await
            .unwrap();

        assert_eq!(result, (0, 1));
        assert_eq!(
            generation.load(std::sync::atomic::Ordering::Acquire),
            0,
            "unchanged tracker events must not invalidate search caches"
        );
    }

    #[tokio::test]
    async fn review_regression_oversized_tracker_event_invalidates_source_caches() {
        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        let path = tmp.path().join("large.rs");
        std::fs::write(&path, "fn before() { let value = 12345; }\n").unwrap();
        let mut config = Config::from_env();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        config.max_embed_file_size = 8;
        let server = ContextPlusServer::new(tmp.path().to_path_buf(), config);
        server.ensure_project_cache().await.unwrap();

        std::fs::write(&path, "fn after() { let value = 67890; }\n").unwrap();
        let generation = Arc::clone(&server.current_ref().await.cache_generation);
        let result = server.build_tracker_callback().await(
            tmp.path().to_path_buf(),
            vec!["large.rs".to_string()],
        )
        .await
        .unwrap();

        assert_eq!(result, (0, 1));
        assert_eq!(
            generation.load(std::sync::atomic::Ordering::Acquire),
            1,
            "a changed oversized source file must invalidate search caches"
        );
        assert!(
            server
                .current_ref()
                .await
                .project_cache
                .read()
                .await
                .is_none(),
            "a changed oversized source file must invalidate the project cache"
        );
    }

    #[tokio::test]
    async fn review_regression_semantic_cache_hash_matches_unchanged_tracker_content() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|req: &Request| {
                let body: serde_json::Value = req.body_json().unwrap_or_default();
                let n = body["input"].as_array().map_or(1, |a| a.len());
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![0.6, 0.8]; n]
                }))
            })
            .mount(&ollama)
            .await;

        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        let content = "fn unchanged() { println!(\"still here\"); }\n";
        std::fs::write(tmp.path().join("unchanged.rs"), content).unwrap();
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(tmp.path().to_path_buf(), config);
        let walker = CachedWalkerIndexer {
            config: server.state.config.clone(),
            ollama: server.state.ollama.clone(),
            state: Arc::clone(&server.state),
        };
        walker.walk_and_index(tmp.path()).await.unwrap();
        let requests_before = ollama.received_requests().await.unwrap_or_default().len();

        let generation = Arc::clone(&server.current_ref().await.cache_generation);
        let result = server.build_tracker_callback().await(
            tmp.path().to_path_buf(),
            vec!["unchanged.rs".to_string()],
        )
        .await
        .unwrap();

        assert_eq!(result, (0, 1));
        assert_eq!(
            generation.load(std::sync::atomic::Ordering::Acquire),
            0,
            "semantic-produced cache entries must recognize unchanged source"
        );
        assert_eq!(
            ollama.received_requests().await.unwrap_or_default().len(),
            requests_before,
            "an unchanged tracker event must not embed again"
        );
    }

    #[tokio::test]
    async fn review_regression_identifier_build_cannot_restore_invalidated_snapshot() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|req: &Request| {
                let body: serde_json::Value = req.body_json().unwrap_or_default();
                let n = body["input"].as_array().map_or(1, |a| a.len());
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_millis(200))
                    .set_body_json(serde_json::json!({
                        "embeddings": vec![vec![0.6, 0.8]; n]
                    }))
            })
            .mount(&ollama)
            .await;

        let tmp = tempfile::tempdir().expect("failed to create temp dir");
        let path = tmp.path().join("changing.rs");
        std::fs::write(&path, "fn before_change() {}\n").unwrap();
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(tmp.path().to_path_buf(), config);
        let old_cache = server.ensure_project_cache().await.unwrap();

        let builder = {
            let server = server.clone();
            let old_cache = Arc::clone(&old_cache);
            tokio::spawn(async move { server.ensure_identifier_index(&old_cache).await.unwrap() })
        };
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while ollama
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
        {
            assert!(
                Instant::now() < deadline,
                "identifier build never reached embedding"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        std::fs::write(&path, "fn after_change() {}\n").unwrap();
        server
            .current_ref()
            .await
            .cache_generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        server
            .invalidate_project_cache_with_reason("test invalidation during identifier build")
            .await;

        let index = builder.await.unwrap();
        assert!(
            index.docs.iter().any(|doc| doc.name == "after_change"),
            "the request must retry against the fresh project cache"
        );
        assert!(
            index.docs.iter().all(|doc| doc.name != "before_change"),
            "the stale identifier snapshot must not be restored"
        );
    }

    #[tokio::test]
    async fn review_regression_warmup_keeps_expired_cache_with_live_tracker() {
        for mode in [RefWarmupMode::Shallow, RefWarmupMode::Full] {
            let tmp = tempfile::tempdir().expect("failed to create temp dir");
            std::fs::write(tmp.path().join("warm.rs"), "fn warm() {}\n").unwrap();
            let mut config = Config::from_env();
            config.embed_tracker_mode = TrackerMode::Lazy;
            config.ref_warmup_mode = mode;
            config.cache_ttl_secs = 1;
            let server = ContextPlusServer::new(tmp.path().to_path_buf(), config);
            let current = server.ensure_project_cache().await.unwrap();
            let expired = Arc::new(ProjectCache {
                file_entries: current.file_entries.clone(),
                file_content: current.file_content.clone(),
                clean_blobs: None,
                last_refresh: Instant::now() - std::time::Duration::from_secs(2),
            });
            *server.current_ref().await.project_cache.write().await = Some(Arc::clone(&expired));
            server.ensure_tracker_started().await;

            server.spawn_ref_warmup(server.state.default_ref_id);
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            let actual = server
                .current_ref()
                .await
                .project_cache
                .read()
                .await
                .as_ref()
                .cloned()
                .expect("warm cache should remain present");
            assert!(
                Arc::ptr_eq(&expired, &actual),
                "{mode:?} warmup must not replace a tracker-owned cache after TTL"
            );
        }
    }

    // ---------------------------------------------------------------
    // get_str edge cases
    // ---------------------------------------------------------------

    #[test]
    fn get_str_returns_empty_string_for_empty_string_value() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), json!(""));
        assert_eq!(
            ContextPlusServer::get_str(&args, "key"),
            Some("".to_string())
        );
    }

    #[test]
    fn get_str_returns_none_for_null_value() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), Value::Null);
        assert_eq!(ContextPlusServer::get_str(&args, "key"), None);
    }

    #[test]
    fn get_str_returns_none_for_array_value() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), json!(["a", "b"]));
        assert_eq!(ContextPlusServer::get_str(&args, "key"), None);
    }

    #[test]
    fn get_str_returns_none_for_boolean_value() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), json!(true));
        assert_eq!(ContextPlusServer::get_str(&args, "key"), None);
    }

    #[test]
    fn get_str_returns_none_for_object_value() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), json!({"nested": "obj"}));
        assert_eq!(ContextPlusServer::get_str(&args, "key"), None);
    }

    #[test]
    fn get_str_preserves_whitespace() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), json!("  hello  world  "));
        assert_eq!(
            ContextPlusServer::get_str(&args, "key"),
            Some("  hello  world  ".to_string())
        );
    }

    #[test]
    fn get_str_handles_unicode() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), json!("日本語テスト"));
        assert_eq!(
            ContextPlusServer::get_str(&args, "key"),
            Some("日本語テスト".to_string())
        );
    }

    // ---------------------------------------------------------------
    // get_str_or edge cases
    // ---------------------------------------------------------------

    #[test]
    fn get_str_or_returns_default_when_wrong_type() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), json!(42));
        assert_eq!(
            ContextPlusServer::get_str_or(&args, "key", "default"),
            "default"
        );
    }

    #[test]
    fn get_str_or_returns_empty_string_value_not_default() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), json!(""));
        // Empty string IS a valid string, should return it, not the default
        assert_eq!(ContextPlusServer::get_str_or(&args, "key", "default"), "");
    }

    #[test]
    fn get_str_or_returns_default_for_null() {
        let mut args = serde_json::Map::new();
        args.insert("key".to_string(), Value::Null);
        assert_eq!(
            ContextPlusServer::get_str_or(&args, "key", "fallback"),
            "fallback"
        );
    }

    // ---------------------------------------------------------------
    // get_usize edge cases
    // ---------------------------------------------------------------

    #[test]
    fn get_usize_returns_none_for_string() {
        let mut args = serde_json::Map::new();
        args.insert("n".to_string(), json!("42"));
        assert_eq!(ContextPlusServer::get_usize(&args, "n"), None);
    }

    #[test]
    fn get_usize_returns_none_for_negative_number() {
        let mut args = serde_json::Map::new();
        args.insert("n".to_string(), json!(-5));
        // as_u64() returns None for negative numbers
        assert_eq!(ContextPlusServer::get_usize(&args, "n"), None);
    }

    #[test]
    fn get_usize_handles_zero() {
        let mut args = serde_json::Map::new();
        args.insert("n".to_string(), json!(0));
        assert_eq!(ContextPlusServer::get_usize(&args, "n"), Some(0));
    }

    #[test]
    fn get_usize_returns_none_for_float() {
        let mut args = serde_json::Map::new();
        args.insert("n".to_string(), json!(3.75));
        // as_u64() returns None for non-integer values
        assert_eq!(ContextPlusServer::get_usize(&args, "n"), None);
    }

    #[test]
    fn get_usize_returns_none_for_bool() {
        let mut args = serde_json::Map::new();
        args.insert("n".to_string(), json!(true));
        assert_eq!(ContextPlusServer::get_usize(&args, "n"), None);
    }

    #[test]
    fn get_usize_returns_none_for_null() {
        let mut args = serde_json::Map::new();
        args.insert("n".to_string(), Value::Null);
        assert_eq!(ContextPlusServer::get_usize(&args, "n"), None);
    }

    #[test]
    fn get_usize_handles_large_number() {
        let mut args = serde_json::Map::new();
        args.insert("n".to_string(), json!(1_000_000u64));
        assert_eq!(ContextPlusServer::get_usize(&args, "n"), Some(1_000_000));
    }

    // ---------------------------------------------------------------
    // get_f64 edge cases
    // ---------------------------------------------------------------

    #[test]
    fn get_f64_returns_none_for_string() {
        let mut args = serde_json::Map::new();
        args.insert("f".to_string(), json!("3.14"));
        assert_eq!(ContextPlusServer::get_f64(&args, "f"), None);
    }

    #[test]
    fn get_f64_handles_integer_as_float() {
        let mut args = serde_json::Map::new();
        args.insert("f".to_string(), json!(42));
        // as_f64() should coerce integer to f64
        assert_eq!(ContextPlusServer::get_f64(&args, "f"), Some(42.0));
    }

    #[test]
    fn get_f64_handles_zero() {
        let mut args = serde_json::Map::new();
        args.insert("f".to_string(), json!(0.0));
        assert_eq!(ContextPlusServer::get_f64(&args, "f"), Some(0.0));
    }

    #[test]
    fn get_f64_handles_negative() {
        let mut args = serde_json::Map::new();
        args.insert("f".to_string(), json!(-1.5));
        let val = ContextPlusServer::get_f64(&args, "f").unwrap();
        assert!((val - (-1.5)).abs() < f64::EPSILON);
    }

    #[test]
    fn get_f64_returns_none_for_bool() {
        let mut args = serde_json::Map::new();
        args.insert("f".to_string(), json!(true));
        assert_eq!(ContextPlusServer::get_f64(&args, "f"), None);
    }

    #[test]
    fn get_f64_returns_none_for_null() {
        let mut args = serde_json::Map::new();
        args.insert("f".to_string(), Value::Null);
        assert_eq!(ContextPlusServer::get_f64(&args, "f"), None);
    }

    // ---------------------------------------------------------------
    // get_bool edge cases
    // ---------------------------------------------------------------

    #[test]
    fn get_bool_returns_none_for_string_true() {
        let mut args = serde_json::Map::new();
        args.insert("b".to_string(), json!("true"));
        // "true" as a string is NOT a boolean
        assert_eq!(ContextPlusServer::get_bool(&args, "b"), None);
    }

    #[test]
    fn get_bool_returns_none_for_number_one() {
        let mut args = serde_json::Map::new();
        args.insert("b".to_string(), json!(1));
        // Number 1 is NOT a boolean
        assert_eq!(ContextPlusServer::get_bool(&args, "b"), None);
    }

    #[test]
    fn get_bool_returns_none_for_number_zero() {
        let mut args = serde_json::Map::new();
        args.insert("b".to_string(), json!(0));
        assert_eq!(ContextPlusServer::get_bool(&args, "b"), None);
    }

    #[test]
    fn get_bool_returns_none_for_null() {
        let mut args = serde_json::Map::new();
        args.insert("b".to_string(), Value::Null);
        assert_eq!(ContextPlusServer::get_bool(&args, "b"), None);
    }

    // ---------------------------------------------------------------
    // resolve_root edge cases
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn resolve_root_accepts_subdirectory_inside_root() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let sub = tmp.path().join("subdir");
        std::fs::create_dir_all(&sub).unwrap();
        let server = server_with_root_and_ttl(tmp.path().to_path_buf(), 300);

        let mut args = serde_json::Map::new();
        args.insert(
            "rootDir".to_string(),
            json!(sub.to_string_lossy().to_string()),
        );

        let root = server.resolve_root(&args).await;
        // Should accept the subdirectory since it's inside the server root
        let canonical_sub = sub.canonicalize().unwrap();
        assert_eq!(root, canonical_sub);
    }

    #[tokio::test]
    async fn resolve_root_rejects_nonexistent_path() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert("rootDir".to_string(), json!("/nonexistent/path/xyz123"));
        let root = server.resolve_root(&args).await;
        // Should fall back to server root since path doesn't exist (canonicalize fails)
        assert_eq!(root, server.state.root_dir);
    }

    #[tokio::test]
    async fn resolve_root_rejects_empty_string() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert("rootDir".to_string(), json!(""));
        let root = server.resolve_root(&args).await;
        // Empty string can't be canonicalized to a path inside root
        assert_eq!(
            root.canonicalize().unwrap(),
            server.state.root_dir.canonicalize().unwrap()
        );
    }

    #[tokio::test]
    async fn resolve_root_rejects_relative_path_outside_root() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert("rootDir".to_string(), json!("../../etc"));
        let root = server.resolve_root(&args).await;
        assert_eq!(root, server.state.root_dir);
    }

    // ---------------------------------------------------------------
    // route_static_analysis_target — absolute target_path auto-routes
    // to a registered worktree's root (out-of-server-root paths).
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn route_static_analysis_target_falls_back_for_relative_target() {
        let server = test_server();
        let args = serde_json::Map::new();
        let (root, target) = server
            .route_static_analysis_target(&args, Some("src/lib.rs".to_string()))
            .await;
        // Relative target — no routing, falls back to resolve_root (= server root)
        // and preserves the relative target string verbatim.
        assert_eq!(root, server.state.root_dir);
        assert_eq!(target.as_deref(), Some("src/lib.rs"));
    }

    #[tokio::test]
    async fn route_static_analysis_target_routes_to_registered_worktree() {
        // Server lives at `server_root` (the "primary" /workspace analogue).
        // A separate, sibling worktree is registered at `wt_root` — outside
        // `server_root`. An absolute target_path under `wt_root` should route
        // the linter to run from `wt_root` with a rebased relative target.
        let server_root_tmp = tempfile::tempdir().unwrap();
        let wt_tmp = tempfile::tempdir().unwrap();
        let wt_root = wt_tmp.path().to_path_buf();
        let canonical_wt = wt_root.canonicalize().unwrap();
        // Create a file under the worktree so canonicalize succeeds on the target.
        let src_dir = canonical_wt.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let target_file = src_dir.join("lib.rs");
        std::fs::write(&target_file, "// stub\n").unwrap();

        let server = server_with_root_and_ttl(server_root_tmp.path().to_path_buf(), 300);

        // Register the worktree as a non-default ref.
        let wt_ref = crate::ref_index::RefIndex::new(wt_root.clone(), canonical_wt.clone(), None);
        let wt_ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_wt);
        {
            let mut guard = server.state.refs.write().await;
            guard.insert(wt_ref_id, std::sync::Arc::new(wt_ref));
        }

        let args = serde_json::Map::new();
        let (root, target) = server
            .route_static_analysis_target(&args, Some(target_file.to_string_lossy().into_owned()))
            .await;

        assert_eq!(root, wt_root, "linter should run from the worktree's root");
        assert_eq!(
            target.as_deref(),
            Some("src/lib.rs"),
            "target_path should be rebased relative to the routed worktree"
        );
    }

    #[tokio::test]
    async fn attach_worktree_registers_external_directory() {
        let server_root_tmp = tempfile::tempdir().unwrap();
        let wt_tmp = tempfile::tempdir().unwrap();
        let canonical_wt = wt_tmp.path().canonicalize().unwrap();

        let server = server_with_root_and_ttl(server_root_tmp.path().to_path_buf(), 300);

        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );

        let result = server.handle_attach_worktree(args).await.unwrap();
        assert_eq!(result.is_error, Some(false));

        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_wt);
        let registered = {
            let guard = server.state.refs.read().await;
            guard.get(&ref_id).cloned()
        };
        let registered = registered.expect("worktree should be in registry");
        assert_eq!(registered.canonical_root, canonical_wt);
        // Session_count bumped to 1 by attach_ref — pinned until matching detach.
        assert_eq!(
            registered
                .session_count
                .load(std::sync::atomic::Ordering::Acquire),
            1
        );
        // Parent pointer chains to the primary ref so CAS lookups inherit the baseline.
        assert_eq!(registered.parent_ref_id, Some(server.state.default_ref_id));
    }

    #[tokio::test]
    async fn attach_worktree_cas_chain_writes_parent_pointer() {
        // The CoW chain on disk: attach_worktree must write a `parent` pointer
        // inside the worktree's CAS dir so chunk lookups chain to the primary's
        // manifest. This is the on-disk half of the "build on top of base cache"
        // contract — verifying the in-memory `parent_ref_id` only catches half.
        let server_root_tmp = tempfile::tempdir().unwrap();
        let wt_tmp = tempfile::tempdir().unwrap();
        let canonical_wt = wt_tmp.path().canonicalize().unwrap();
        let server = server_with_root_and_ttl(server_root_tmp.path().to_path_buf(), 300);

        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );
        let _ = server.handle_attach_worktree(args).await.unwrap();

        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_wt);
        let ref_arc = server.state.ref_index(ref_id).await.expect("ref attached");
        let mcp_data = server.state.root_dir.join(".mcp_data");
        let ref_dir = mcp_data.join("refs").join(&ref_arc.cas_ref_id_hex);
        assert!(
            ref_dir.exists(),
            "CAS ref dir must be created by fork_from at {}",
            ref_dir.display()
        );
        // Manifest is written by fork_ref so the chain is fully initialized.
        assert!(
            ref_dir.join("manifest.rkyv").exists(),
            "fork_from must write an empty manifest at {}/manifest.rkyv",
            ref_dir.display()
        );
    }

    #[tokio::test]
    async fn detach_worktree_decrements_and_schedules_eviction() {
        let server_root_tmp = tempfile::tempdir().unwrap();
        let wt_tmp = tempfile::tempdir().unwrap();
        let canonical_wt = wt_tmp.path().canonicalize().unwrap();
        // ttl=0 so eviction is immediate-ish; tokio::spawn still runs async.
        let server = server_with_root_and_ttl(server_root_tmp.path().to_path_buf(), 0);

        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );
        let _ = server.handle_attach_worktree(args.clone()).await.unwrap();

        let result = server.handle_detach_worktree(args).await.unwrap();
        assert_eq!(result.is_error, Some(false));

        // Yield so the spawned eviction task can run (ttl=0 → no sleep).
        for _ in 0..20 {
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            let guard = server.state.refs.read().await;
            let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_wt);
            if !guard.contains_key(&ref_id) {
                return;
            }
        }
        panic!("worktree ref should have been evicted after detach with ttl=0");
    }

    #[tokio::test]
    async fn detach_worktree_refuses_primary() {
        let server = test_server();
        let primary_path = server.state.root_dir.clone();
        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(primary_path.to_string_lossy().to_string()),
        );
        let result = server.handle_detach_worktree(args).await.unwrap();
        assert_eq!(result.is_error, Some(true));
    }

    #[tokio::test]
    async fn detach_worktree_errors_on_unknown() {
        let server_root_tmp = tempfile::tempdir().unwrap();
        let wt_tmp = tempfile::tempdir().unwrap();
        let server = server_with_root_and_ttl(server_root_tmp.path().to_path_buf(), 300);

        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(wt_tmp.path().to_string_lossy().to_string()),
        );
        let result = server.handle_detach_worktree(args).await.unwrap();
        assert_eq!(result.is_error, Some(true));
    }

    #[tokio::test]
    async fn list_worktrees_marks_primary_and_includes_attached() {
        let server_root_tmp = tempfile::tempdir().unwrap();
        let wt_tmp = tempfile::tempdir().unwrap();
        let canonical_wt = wt_tmp.path().canonicalize().unwrap();
        let server = server_with_root_and_ttl(server_root_tmp.path().to_path_buf(), 300);

        let mut attach_args = serde_json::Map::new();
        attach_args.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );
        let _ = server.handle_attach_worktree(attach_args).await.unwrap();

        let result = server
            .handle_list_worktrees(serde_json::Map::new())
            .await
            .unwrap();
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(text.contains("[primary]"), "primary tag missing: {text}");
        assert!(
            text.contains(&canonical_wt.to_string_lossy().to_string()),
            "attached worktree path missing: {text}"
        );
        assert!(
            text.contains(&format!(
                "OLLAMA_EMBED_MODEL={}",
                server.state.config.ollama_embed_model
            )),
            "daemon embed model missing: {text}"
        );
        assert!(
            text.contains(&format!(
                "OLLAMA_CHAT_MODEL={}",
                server.state.config.ollama_chat_model
            )),
            "daemon chat model missing: {text}"
        );
        assert!(
            text.contains(&format!("OLLAMA_HOST={}", server.state.config.ollama_host)),
            "daemon Ollama host missing: {text}"
        );
    }

    #[tokio::test]
    async fn list_worktrees_names_primary_mcp_config_source() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".git")).unwrap();
        let config_path = root.path().join(".mcp.json");
        std::fs::write(
            &config_path,
            serde_json::json!({
                "mcpServers": {
                    "contextplus": {
                        "command": "/opt/contextplus-rs",
                        "env": {
                            "OLLAMA_EMBED_MODEL": "model-from-primary-file",
                            "CONTEXTPLUS_EMBED_TRACKER": "off",
                            "CONTEXTPLUS_WARMUP_ON_START": "false"
                        }
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        let inherited = std::collections::HashMap::from([(
            "OLLAMA_EMBED_MODEL".to_string(),
            "model-from-spawning-session".to_string(),
        )]);
        let resolved =
            crate::transport::daemon::resolve_daemon_startup_config(root.path(), &inherited);
        let server = ContextPlusServer::new(root.path().to_path_buf(), resolved.config);

        let result = server
            .handle_list_worktrees(serde_json::Map::new())
            .await
            .unwrap();
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text"),
        };

        assert!(
            text.contains("Config source:"),
            "source label missing: {text}"
        );
        assert!(
            text.contains(&config_path.display().to_string()),
            "primary .mcp.json path missing: {text}"
        );
        assert!(
            text.contains("OLLAMA_EMBED_MODEL=model-from-primary-file"),
            "resolved daemon model missing: {text}"
        );
    }

    #[tokio::test]
    async fn attach_worktree_is_idempotent() {
        let server_root_tmp = tempfile::tempdir().unwrap();
        let wt_tmp = tempfile::tempdir().unwrap();
        let canonical_wt = wt_tmp.path().canonicalize().unwrap();
        let server = server_with_root_and_ttl(server_root_tmp.path().to_path_buf(), 300);

        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );

        let _ = server.handle_attach_worktree(args.clone()).await.unwrap();
        let result2 = server.handle_attach_worktree(args).await.unwrap();
        let text = match &result2.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(
            text.contains("already attached"),
            "second attach should be idempotent; got: {text}"
        );
    }

    #[tokio::test]
    async fn attach_worktree_then_static_analysis_routes_to_it() {
        let server_root_tmp = tempfile::tempdir().unwrap();
        let wt_tmp = tempfile::tempdir().unwrap();
        let canonical_wt = wt_tmp.path().canonicalize().unwrap();
        let src_dir = canonical_wt.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let target_file = src_dir.join("lib.rs");
        std::fs::write(&target_file, "// stub\n").unwrap();

        let server = server_with_root_and_ttl(server_root_tmp.path().to_path_buf(), 300);

        // Attach via the new MCP tool.
        let mut attach_args = serde_json::Map::new();
        attach_args.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );
        let _ = server.handle_attach_worktree(attach_args).await.unwrap();

        // Now static-analysis with an absolute path under the worktree must route.
        let args = serde_json::Map::new();
        let (root, target) = server
            .route_static_analysis_target(&args, Some(target_file.to_string_lossy().into_owned()))
            .await;
        // Use canonical_wt for the comparison: on macOS, /var/folders/... is a
        // symlink to /private/var/folders/..., and attach_worktree canonicalizes
        // before storing root_dir. Comparing against wt_tmp.path() would fail
        // there.
        assert_eq!(root, canonical_wt);
        assert_eq!(target.as_deref(), Some("src/lib.rs"));
    }

    #[tokio::test]
    async fn attach_worktree_rejects_missing_path_arg() {
        let server = test_server();
        let result = server.handle_attach_worktree(serde_json::Map::new()).await;
        assert!(result.is_err(), "missing path should be a hard error");
    }

    #[tokio::test]
    async fn attach_worktree_rejects_nonexistent_path() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!("/nonexistent/contextplus-attach-test-xyz"),
        );
        let result = server.handle_attach_worktree(args).await.unwrap();
        assert_eq!(result.is_error, Some(true));
    }

    #[tokio::test]
    async fn attach_worktree_eager_mode_starts_per_ref_tracker() {
        // U11: attaching a worktree in Eager tracker mode must start an
        // embedding tracker scoped to the new ref, not the default ref. This
        // is the live-edit pickup path — without it, an attached worktree's
        // caches only refresh on full warmup.
        let server_root_tmp = tempfile::tempdir().unwrap();
        let wt_tmp = tempfile::tempdir().unwrap();
        let canonical_wt = wt_tmp.path().canonicalize().unwrap();

        let mut config = Config::from_env();
        config.embed_tracker_mode = TrackerMode::Eager;
        // Avoid background warmup interfering with the test — the tracker
        // start is independent of warmup mode.
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(server_root_tmp.path().to_path_buf(), config);

        // Default ref should NOT have a tracker yet — we never called
        // `ensure_tracker_started` for it in this test path.
        let default_ref = server.state.default_ref().expect("default ref present");
        assert!(
            default_ref.tracker_handle.lock().unwrap().is_none(),
            "default ref tracker should not start as a side-effect of attach"
        );

        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );
        let result = server.handle_attach_worktree(args).await.unwrap();
        assert_eq!(result.is_error, Some(false));

        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_wt);
        let wt_ref = server
            .state
            .ref_index(ref_id)
            .await
            .expect("worktree ref attached");

        assert!(
            wt_ref.tracker_handle.lock().unwrap().is_some(),
            "Eager mode must start a tracker on the attached worktree ref"
        );
        // Per-ref isolation: the default ref's handle must still be untouched.
        assert!(
            default_ref.tracker_handle.lock().unwrap().is_none(),
            "default ref tracker must not start when only a worktree was attached"
        );
    }

    #[tokio::test]
    async fn attach_worktree_lazy_mode_defers_tracker_start() {
        // Counterpoint to the eager test: Lazy mode must NOT start the
        // tracker at attach time. It starts on the first tool call against
        // the ref's session (handled by `ensure_tracker_started` via
        // `current_ref()`).
        let server_root_tmp = tempfile::tempdir().unwrap();
        let wt_tmp = tempfile::tempdir().unwrap();
        let canonical_wt = wt_tmp.path().canonicalize().unwrap();

        let mut config = Config::from_env();
        config.embed_tracker_mode = TrackerMode::Lazy;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(server_root_tmp.path().to_path_buf(), config);

        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );
        let _ = server.handle_attach_worktree(args).await.unwrap();

        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_wt);
        let wt_ref = server
            .state
            .ref_index(ref_id)
            .await
            .expect("worktree ref attached");
        assert!(
            wt_ref.tracker_handle.lock().unwrap().is_none(),
            "Lazy mode must not eagerly start the worktree tracker"
        );

        // First tool-style invocation through a session-scoped clone starts it.
        server.ensure_tracker_started_for(ref_id).await;
        assert!(
            wt_ref.tracker_handle.lock().unwrap().is_some(),
            "ensure_tracker_started_for(ref_id) must start the worktree's tracker"
        );
    }

    #[tokio::test]
    async fn ensure_tracker_started_for_off_mode_is_noop() {
        // Sanity: Off mode is a hard no-op regardless of call site.
        let server_root_tmp = tempfile::tempdir().unwrap();
        let wt_tmp = tempfile::tempdir().unwrap();
        let canonical_wt = wt_tmp.path().canonicalize().unwrap();

        let mut config = Config::from_env();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(server_root_tmp.path().to_path_buf(), config);

        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );
        let _ = server.handle_attach_worktree(args).await.unwrap();

        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_wt);
        let wt_ref = server
            .state
            .ref_index(ref_id)
            .await
            .expect("worktree ref attached");
        // Even an explicit call must be a no-op under Off mode.
        server.ensure_tracker_started_for(ref_id).await;
        assert!(
            wt_ref.tracker_handle.lock().unwrap().is_none(),
            "Off mode must never start a tracker"
        );
    }

    #[tokio::test]
    async fn route_static_analysis_target_falls_back_when_no_ref_matches() {
        let server = test_server();
        // Absolute path that lives nowhere near any registered ref.
        let unrelated = std::env::temp_dir().join("nowhere-cp-route-test-xyz/file.rs");
        let args = serde_json::Map::new();
        let (root, target) = server
            .route_static_analysis_target(&args, Some(unrelated.to_string_lossy().into_owned()))
            .await;
        // No matching ref → fall back to resolve_root, which returns server root,
        // and the absolute target stays as-is (caller's problem if it doesn't fit).
        assert_eq!(root, server.state.root_dir);
        assert_eq!(
            target.as_deref(),
            Some(unrelated.to_string_lossy().as_ref())
        );
    }

    // ---------------------------------------------------------------
    // ok_text / err_text content verification
    // ---------------------------------------------------------------

    #[test]
    fn ok_text_contains_correct_text() {
        let result = ContextPlusServer::ok_text("test message".to_string());
        assert_eq!(result.is_error, Some(false));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert_eq!(text, "test message");
    }

    #[test]
    fn err_text_contains_correct_text() {
        let result = ContextPlusServer::err_text("error message".to_string());
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert_eq!(text, "error message");
    }

    #[test]
    fn ok_text_handles_empty_string() {
        let result = ContextPlusServer::ok_text(String::new());
        assert_eq!(result.is_error, Some(false));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert_eq!(text, "");
    }

    #[test]
    fn err_text_handles_empty_string() {
        let result = ContextPlusServer::err_text(String::new());
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert_eq!(text, "");
    }

    #[test]
    fn ok_text_handles_multiline_text() {
        let result = ContextPlusServer::ok_text("line1\nline2\nline3".to_string());
        assert_eq!(result.is_error, Some(false));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert_eq!(text, "line1\nline2\nline3");
    }

    // ---------------------------------------------------------------
    // make_tool edge cases
    // ---------------------------------------------------------------

    #[test]
    fn make_tool_with_no_params() {
        let tool = make_tool("empty_tool", "No params", &[]);
        assert_eq!(tool.name.as_ref(), "empty_tool");
        let schema = tool.input_schema.as_ref();
        let props = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .unwrap();
        assert!(props.is_empty(), "should have no properties");
        // required key should not be present
        assert!(
            schema.get("required").is_none(),
            "should not have required key when no required params"
        );
    }

    #[test]
    fn make_tool_with_all_required_params() {
        let tool = make_tool(
            "all_required",
            "All required",
            &[
                ("a", "string", true, "Param a"),
                ("b", "integer", true, "Param b"),
            ],
        );
        let schema = tool.input_schema.as_ref();
        let required = schema.get("required").and_then(|v| v.as_array()).unwrap();
        assert_eq!(required.len(), 2);
        let req_strs: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(req_strs.contains(&"a"));
        assert!(req_strs.contains(&"b"));
    }

    #[test]
    fn make_tool_with_all_optional_params() {
        let tool = make_tool(
            "all_optional",
            "All optional",
            &[
                ("x", "string", false, "Param x"),
                ("y", "integer", false, "Param y"),
            ],
        );
        let schema = tool.input_schema.as_ref();
        // No required array since all are optional
        assert!(
            schema.get("required").is_none(),
            "should not have required key when all params are optional"
        );
        let props = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .unwrap();
        assert_eq!(props.len(), 2);
    }

    #[test]
    fn make_tool_schema_has_correct_type_field() {
        let tool = make_tool(
            "typed",
            "Typed params",
            &[("n", "integer", true, "A number")],
        );
        let schema = tool.input_schema.as_ref();
        assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("object"));
        let props = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .unwrap();
        let n_prop = props.get("n").and_then(|v| v.as_object()).unwrap();
        assert_eq!(n_prop.get("type").and_then(|v| v.as_str()), Some("integer"));
        assert_eq!(
            n_prop.get("description").and_then(|v| v.as_str()),
            Some("A number")
        );
    }

    // ---------------------------------------------------------------
    // ---------------------------------------------------------------

    // ---------------------------------------------------------------
    // code_sym_to_tree_sym / code_sym_to_skel_sym
    // ---------------------------------------------------------------

    #[test]
    fn code_sym_to_tree_sym_converts_basic_symbol() {
        let sym = crate::core::parser::CodeSymbol {
            name: "my_func".to_string(),
            kind: "function".to_string(),
            line: 10,
            end_line: 20,
            signature: Some("fn my_func(x: i32) -> bool".to_string()),
            children: vec![],
        };
        let tree_sym = code_sym_to_tree_sym(&sym);
        assert_eq!(tree_sym.name, "my_func");
        assert_eq!(tree_sym.kind, "function");
        assert_eq!(tree_sym.line, 10);
        assert_eq!(tree_sym.end_line, 20);
        assert_eq!(tree_sym.signature, "fn my_func(x: i32) -> bool");
        assert!(tree_sym.children.is_empty());
    }

    #[test]
    fn code_sym_to_tree_sym_converts_symbol_without_signature() {
        let sym = crate::core::parser::CodeSymbol {
            name: "MY_CONST".to_string(),
            kind: "constant".to_string(),
            line: 5,
            end_line: 5,
            signature: None,
            children: vec![],
        };
        let tree_sym = code_sym_to_tree_sym(&sym);
        assert_eq!(tree_sym.signature, ""); // None becomes empty string
    }

    #[test]
    fn code_sym_to_tree_sym_converts_nested_children() {
        let child = crate::core::parser::CodeSymbol {
            name: "inner".to_string(),
            kind: "method".to_string(),
            line: 15,
            end_line: 18,
            signature: Some("fn inner()".to_string()),
            children: vec![],
        };
        let parent = crate::core::parser::CodeSymbol {
            name: "MyClass".to_string(),
            kind: "class".to_string(),
            line: 10,
            end_line: 25,
            signature: Some("class MyClass".to_string()),
            children: vec![child],
        };
        let tree_sym = code_sym_to_tree_sym(&parent);
        assert_eq!(tree_sym.children.len(), 1);
        assert_eq!(tree_sym.children[0].name, "inner");
        assert_eq!(tree_sym.children[0].kind, "method");
    }

    #[test]
    fn code_sym_to_skel_sym_converts_basic_symbol() {
        let sym = crate::core::parser::CodeSymbol {
            name: "handler".to_string(),
            kind: "function".to_string(),
            line: 1,
            end_line: 50,
            signature: Some("async fn handler(req: Request) -> Response".to_string()),
            children: vec![],
        };
        let skel_sym = code_sym_to_skel_sym(&sym);
        assert_eq!(skel_sym.name, "handler");
        assert_eq!(skel_sym.kind, "function");
        assert_eq!(skel_sym.line, 1);
        assert_eq!(skel_sym.end_line, 50);
        assert_eq!(
            skel_sym.signature,
            "async fn handler(req: Request) -> Response"
        );
        assert!(skel_sym.children.is_empty());
    }

    #[test]
    fn code_sym_to_skel_sym_converts_symbol_without_signature() {
        let sym = crate::core::parser::CodeSymbol {
            name: "FOO".to_string(),
            kind: "variable".to_string(),
            line: 3,
            end_line: 3,
            signature: None,
            children: vec![],
        };
        let skel_sym = code_sym_to_skel_sym(&sym);
        assert_eq!(skel_sym.signature, "");
    }

    #[test]
    fn code_sym_to_skel_sym_converts_nested_children() {
        let child = crate::core::parser::CodeSymbol {
            name: "method_a".to_string(),
            kind: "method".to_string(),
            line: 12,
            end_line: 14,
            signature: None,
            children: vec![],
        };
        let parent = crate::core::parser::CodeSymbol {
            name: "Struct".to_string(),
            kind: "struct".to_string(),
            line: 10,
            end_line: 20,
            signature: Some("pub struct Struct".to_string()),
            children: vec![child],
        };
        let skel_sym = code_sym_to_skel_sym(&parent);
        assert_eq!(skel_sym.children.len(), 1);
        assert_eq!(skel_sym.children[0].name, "method_a");
    }

    // ---------------------------------------------------------------
    // dispatch with missing required arguments
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn dispatch_blast_radius_missing_symbol_name_returns_error() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("get_blast_radius", args).await;
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("symbol_name is required"),
            "expected symbol_name error, got: {}",
            text
        );
    }

    #[tokio::test]
    async fn blast_radius_path_to_unattached_worktree_returns_actionable_error() {
        // A `path` that is a real directory but was never `attach_worktree`d must
        // NOT silently fall back to the current ref (that fallback is what made
        // the tool scan the wrong tree). It must error and tell the caller to
        // attach first.
        let primary = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let server = server_with_root_and_ttl(primary.path().to_path_buf(), 300);

        let mut args = serde_json::Map::new();
        args.insert("symbol_name".to_string(), json!("doThing"));
        args.insert(
            "path".to_string(),
            json!(
                other
                    .path()
                    .canonicalize()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            ),
        );

        let result = server.dispatch("get_blast_radius", args).await;
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("not attached") && text.contains("attach_worktree"),
            "expected actionable not-attached error, got: {text}"
        );
    }

    #[tokio::test]
    async fn blast_radius_path_routes_to_attached_worktree_not_primary() {
        // Regression: reviewing a feature branch from the primary checkout.
        // The symbol exists ONLY in the attached worktree. Routing via the
        // session's primary ref (default, as in stdio mode) reported it as
        // "used nowhere"; passing the worktree `path` must find the real usages.
        let primary = tempfile::tempdir().unwrap();
        // Primary tree has the symbol's *name* nowhere.
        std::fs::write(primary.path().join("main.ts"), "export const z = 1;\n").unwrap();
        let server = server_with_root_and_ttl(primary.path().to_path_buf(), 300);

        // Build a separate worktree on disk that *does* use the symbol.
        let wt = tempfile::tempdir().unwrap();
        let canonical_wt = wt.path().canonicalize().unwrap();
        std::fs::create_dir_all(canonical_wt.join("src")).unwrap();
        std::fs::write(
            canonical_wt.join("src/app.ts"),
            "import { rlsAutoTxExtension } from './rls';\nprisma.$extends(rlsAutoTxExtension);\n",
        )
        .unwrap();

        // Register the worktree as an attached ref (what attach_worktree does).
        let wt_ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_wt);
        {
            let mut guard = server.state.refs.write().await;
            guard.insert(
                wt_ref_id,
                std::sync::Arc::new(crate::ref_index::RefIndex::new(
                    canonical_wt.clone(),
                    canonical_wt.clone(),
                    None,
                )),
            );
        }

        // Without `path`: scans the primary, finds nothing → scoped zero-result
        // that does NOT claim global absence.
        let mut bare = serde_json::Map::new();
        bare.insert("symbol_name".to_string(), json!("rlsAutoTxExtension"));
        let bare_text = match &server.dispatch("get_blast_radius", bare).await.content[0].raw {
            RawContent::Text(t) => t.text.clone(),
            _ => panic!("expected text"),
        };
        assert!(
            bare_text.contains("no references") && !bare_text.contains("anywhere in the codebase"),
            "primary scan should be a scoped miss, got: {bare_text}"
        );

        // With `path`: routes to the attached worktree and finds the usages.
        let mut targeted = serde_json::Map::new();
        targeted.insert("symbol_name".to_string(), json!("rlsAutoTxExtension"));
        targeted.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );
        let result = server.dispatch("get_blast_radius", targeted).await;
        assert_eq!(result.is_error, Some(false));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text"),
        };
        assert!(
            text.contains("src/app.ts") && text.contains("usages"),
            "targeted scan should find usages in the attached worktree, got: {text}"
        );
    }

    #[tokio::test]
    async fn find_dead_code_path_to_unattached_worktree_returns_actionable_error() {
        // Same routing guard as blast radius: a non-attached `path` must error,
        // and the error must name `find_dead_code` (verifies the helper threads
        // the tool name through).
        let primary = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let server = server_with_root_and_ttl(primary.path().to_path_buf(), 300);

        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(
                other
                    .path()
                    .canonicalize()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            ),
        );

        let result = server.dispatch("find_dead_code", args).await;
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("not attached") && text.contains("attach_worktree"),
            "expected actionable not-attached error, got: {text}"
        );
        assert!(
            text.contains("find_dead_code"),
            "error should name the calling tool, got: {text}"
        );
    }

    #[tokio::test]
    async fn find_dead_code_path_routes_to_attached_worktree() {
        // The dead-code scan must run against the attached worktree, not the
        // session's primary ref — otherwise it judges "dead" from the wrong tree.
        let primary = tempfile::tempdir().unwrap();
        std::fs::write(primary.path().join("p.ts"), "export const z = 1;\n").unwrap();
        let server = server_with_root_and_ttl(primary.path().to_path_buf(), 300);

        let wt = tempfile::tempdir().unwrap();
        let canonical_wt = wt.path().canonicalize().unwrap();
        std::fs::create_dir_all(canonical_wt.join("src")).unwrap();
        std::fs::write(
            canonical_wt.join("src/lib.ts"),
            "export function unusedThing() { return 1; }\n",
        )
        .unwrap();

        let wt_ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_wt);
        {
            let mut guard = server.state.refs.write().await;
            guard.insert(
                wt_ref_id,
                std::sync::Arc::new(crate::ref_index::RefIndex::new(
                    canonical_wt.clone(),
                    canonical_wt.clone(),
                    None,
                )),
            );
        }

        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );
        let result = server.dispatch("find_dead_code", args).await;
        assert_eq!(result.is_error, Some(false));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        // The scope line names the scanned root, proving routing hit the worktree.
        assert!(
            text.contains(&canonical_wt.display().to_string()),
            "find_dead_code should scope output to the attached worktree, got: {text}"
        );
    }

    #[tokio::test]
    async fn dispatch_file_skeleton_missing_file_path_returns_error() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("get_file_skeleton", args).await;
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("file_path is required"),
            "expected file_path error, got: {}",
            text
        );
    }

    #[tokio::test]
    async fn dispatch_semantic_code_search_missing_query_returns_error() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("semantic_code_search", args).await;
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("query is required"),
            "expected query error, got: {}",
            text
        );
    }

    #[tokio::test]
    async fn dispatch_semantic_navigate_without_query_does_not_error() {
        // query is optional in TS version — should not return a "query is required" error
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("semantic_navigate", args).await;
        // It may still error for other reasons (e.g., no files found), but not because query is missing
        if result.is_error == Some(true) {
            let text = match &result.content[0].raw {
                RawContent::Text(t) => t.text.as_str(),
                _ => "",
            };
            assert!(
                !text.contains("query is required"),
                "query should be optional, got: {}",
                text
            );
        }
    }

    // ---------------------------------------------------------------
    // tool_definitions schema validation
    // ---------------------------------------------------------------

    #[test]
    fn tool_definitions_outline_requires_path() {
        let defs = tool_definitions();
        let tool = defs.iter().find(|t| t.name.as_ref() == "outline").unwrap();
        let schema = tool.input_schema.as_ref();
        let required = schema.get("required").and_then(|v| v.as_array()).unwrap();
        let req_strs: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(req_strs.contains(&"path"));
    }

    #[test]
    fn tool_definitions_impact_has_no_required_params() {
        let defs = tool_definitions();
        let tool = defs.iter().find(|t| t.name.as_ref() == "impact").unwrap();
        let schema = tool.input_schema.as_ref();
        assert!(
            schema.get("required").is_none(),
            "impact should have no required params (symbol is needed only for what = symbol)"
        );
    }

    #[test]
    fn tool_definitions_all_tools_have_object_type_schema() {
        let defs = tool_definitions();
        for tool in defs {
            let schema = tool.input_schema.as_ref();
            assert_eq!(
                schema
                    .get("type")
                    .and_then(|v: &serde_json::Value| v.as_str()),
                Some("object"),
                "tool '{}' should have type: object in schema",
                tool.name
            );
        }
    }

    #[test]
    fn tool_definitions_all_tools_have_properties() {
        let defs = tool_definitions();
        for tool in defs {
            let schema = tool.input_schema.as_ref();
            assert!(
                schema.get("properties").is_some(),
                "tool '{}' should have properties in schema",
                tool.name
            );
        }
    }

    // ---------------------------------------------------------------
    // ContextPlusServer::new
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn server_new_initializes_with_correct_root() {
        let root = PathBuf::from("/tmp/test-root");
        let config = Config::from_env();
        let server = ContextPlusServer::new(root.clone(), config);
        assert_eq!(
            server.current_ref().await.root_dir,
            PathBuf::from("/tmp/test-root")
        );
        assert_eq!(server.state.root_dir, root);
    }

    // ---------------------------------------------------------------
    // Multiple args extraction patterns
    // ---------------------------------------------------------------

    #[test]
    fn get_str_handles_multiple_keys_independently() {
        let mut args = serde_json::Map::new();
        args.insert("a".to_string(), json!("alpha"));
        args.insert("b".to_string(), json!("beta"));
        args.insert("c".to_string(), json!(42));

        assert_eq!(
            ContextPlusServer::get_str(&args, "a"),
            Some("alpha".to_string())
        );
        assert_eq!(
            ContextPlusServer::get_str(&args, "b"),
            Some("beta".to_string())
        );
        assert_eq!(ContextPlusServer::get_str(&args, "c"), None);
        assert_eq!(ContextPlusServer::get_str(&args, "d"), None);
    }

    #[test]
    fn mixed_arg_extraction_from_same_map() {
        let mut args = serde_json::Map::new();
        args.insert("name".to_string(), json!("test"));
        args.insert("count".to_string(), json!(5));
        args.insert("weight".to_string(), json!(0.75));
        args.insert("enabled".to_string(), json!(true));

        assert_eq!(
            ContextPlusServer::get_str(&args, "name"),
            Some("test".to_string())
        );
        assert_eq!(ContextPlusServer::get_usize(&args, "count"), Some(5));
        let w = ContextPlusServer::get_f64(&args, "weight").unwrap();
        assert!((w - 0.75).abs() < f64::EPSILON);
        assert_eq!(ContextPlusServer::get_bool(&args, "enabled"), Some(true));
    }

    // ---------------------------------------------------------------
    // String array extraction (used in dispatch handlers)
    // ---------------------------------------------------------------

    #[test]
    fn string_array_extraction_pattern() {
        // This mirrors the pattern used for include_kinds and edge_filter
        let mut args = serde_json::Map::new();
        args.insert(
            "include_kinds".to_string(),
            json!(["function", "class", "method"]),
        );

        let result: Option<Vec<String>> =
            args.get("include_kinds")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                });

        let kinds = result.unwrap();
        assert_eq!(kinds, vec!["function", "class", "method"]);
    }

    #[test]
    fn string_array_extraction_returns_none_when_missing() {
        let args = serde_json::Map::new();
        let result: Option<Vec<String>> =
            args.get("include_kinds")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                });
        assert!(result.is_none());
    }

    #[test]
    fn string_array_extraction_filters_non_string_elements() {
        let mut args = serde_json::Map::new();
        args.insert(
            "kinds".to_string(),
            json!(["function", 42, "class", true, "method"]),
        );

        let result: Option<Vec<String>> = args.get("kinds").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        });

        let kinds = result.unwrap();
        // Non-string elements are filtered out
        assert_eq!(kinds, vec!["function", "class", "method"]);
    }

    #[test]
    fn string_array_extraction_handles_empty_array() {
        let mut args = serde_json::Map::new();
        args.insert("kinds".to_string(), json!([]));

        let result: Option<Vec<String>> = args.get("kinds").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        });

        let kinds = result.unwrap();
        assert!(kinds.is_empty());
    }

    #[test]
    fn string_array_extraction_returns_none_for_non_array() {
        let mut args = serde_json::Map::new();
        args.insert("kinds".to_string(), json!("not-an-array"));

        let result: Option<Vec<String>> = args.get("kinds").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        });

        assert!(result.is_none());
    }

    // ---------------------------------------------------------------
    // New tool handlers: dispatch tests
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn dispatch_find_dead_code_empty_project_returns_no_candidates() {
        // With an empty project cache (no files), the tool should succeed and
        // report that no dead-symbol candidates were found.
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("find_dead_code", args).await;
        assert_eq!(result.is_error, Some(false));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("No dead-symbol candidates"),
            "empty project should yield no candidates, got: {text}"
        );
    }

    #[tokio::test]
    async fn dispatch_review_pr_diff_missing_diff_returns_error() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("review_pr_diff", args).await;
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("diff is required"),
            "expected diff error, got: {text}"
        );
    }

    #[tokio::test]
    async fn dispatch_review_pr_diff_empty_diff_returns_empty_report() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert("diff".to_string(), json!(""));
        let result = server.dispatch("review_pr_diff", args).await;
        assert_eq!(result.is_error, Some(false));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("empty diff"),
            "empty diff should produce empty-report message, got: {text}"
        );
    }

    #[tokio::test]
    async fn dispatch_detect_dependency_loops_empty_project_returns_no_cycles() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("detect_dependency_loops", args).await;
        assert_eq!(result.is_error, Some(false));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("No import cycles"),
            "empty project should have no cycles, got: {text}"
        );
    }

    #[tokio::test]
    async fn dispatch_review_pr_diff_clamps_oversized_caps() {
        // Caller-supplied max_hops / max_files larger than the internal caps
        // (10 / 2000) must be silently clamped — the call must still succeed
        // rather than triggering an effectively unbounded BFS.
        let server = test_server();
        let mut args = serde_json::Map::new();
        // Minimal synthetic unified diff so the analyzer doesn't short-circuit
        // on the empty-diff path before the clamp matters.
        args.insert(
            "diff".to_string(),
            json!(
                "diff --git a/x.rs b/x.rs\n\
                 --- a/x.rs\n\
                 +++ b/x.rs\n\
                 @@ -1,1 +1,1 @@\n\
                 -fn old() {}\n\
                 +fn new() {}\n"
            ),
        );
        args.insert("max_hops".to_string(), json!(9_999_usize));
        args.insert("max_files".to_string(), json!(1_000_000_usize));
        let result = server.dispatch("review_pr_diff", args).await;
        assert_eq!(
            result.is_error,
            Some(false),
            "oversized caps must clamp (not error)"
        );
    }

    #[tokio::test]
    async fn dispatch_check_embedding_quality_empty_cache_reports_no_issues() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("check_embedding_quality", args).await;
        assert_eq!(result.is_error, Some(false));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("No issues found") || text.contains("0 vector"),
            "empty cache should yield no issues, got: {text}"
        );
    }

    #[tokio::test]
    async fn dispatch_lexical_search_missing_query_returns_error() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("lexical_search", args).await;
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("query is required"),
            "expected query error, got: {text}"
        );
    }

    #[tokio::test]
    async fn dispatch_lexical_search_empty_project_reports_no_files() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert("query".to_string(), json!("anything"));
        let result = server.dispatch("lexical_search", args).await;
        assert_eq!(result.is_error, Some(false));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        // Either "No files indexed" (no cache) or "No lexical matches" (empty result).
        assert!(
            text.contains("No files indexed") || text.contains("No lexical matches"),
            "empty project should yield no results, got: {text}"
        );
    }

    // ----- build_symbols_by_file (M-R3-04) -----
    //
    // The helper consolidates symbol-build loops in `find_dead_code` and
    // `review_pr_diff`. Direct unit tests guarantee the contract regardless
    // of which handler exercises it.

    fn cache_with_files(files: Vec<(&str, &str)>) -> ProjectCache {
        let entries: Vec<crate::core::walker::FileEntry> = files
            .iter()
            .map(|(path, _)| crate::core::walker::FileEntry {
                path: PathBuf::from(path),
                relative_path: path.to_string(),
                is_directory: false,
                depth: 0,
            })
            .collect();
        let file_content: crate::core::walker::FileContents = files
            .into_iter()
            .map(|(path, content)| (path.to_string(), Arc::new(content.to_string())))
            .collect();
        ProjectCache {
            file_entries: entries,
            file_content,
            clean_blobs: None,
            last_refresh: Instant::now(),
        }
    }

    #[test]
    fn build_symbols_by_file_string_keys_includes_parsed_files() {
        let cache = cache_with_files(vec![(
            "src/foo.rs",
            "pub fn alpha() {}\npub fn beta() {}\n",
        )]);

        let by_file: HashMap<String, Vec<crate::core::parser::CodeSymbol>> =
            build_symbols_by_file(&cache, |rel| rel.to_string());

        assert!(
            by_file.contains_key("src/foo.rs"),
            "expected helper to use String key directly from rel_path"
        );
        let names: Vec<&str> = by_file["src/foo.rs"]
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert!(names.contains(&"alpha"), "found {:?}", names);
        assert!(names.contains(&"beta"), "found {:?}", names);
    }

    #[test]
    fn build_symbols_by_file_pathbuf_keys_match_string_keys() {
        let cache = cache_with_files(vec![("src/lib.rs", "pub fn solo() {}\n")]);

        let by_str: HashMap<String, Vec<crate::core::parser::CodeSymbol>> =
            build_symbols_by_file(&cache, |rel| rel.to_string());
        let by_path: HashMap<PathBuf, Vec<crate::core::parser::CodeSymbol>> =
            build_symbols_by_file(&cache, |rel| PathBuf::from(rel));

        assert_eq!(by_str.len(), by_path.len());
        let str_names: Vec<String> = by_str["src/lib.rs"]
            .iter()
            .map(|s| s.name.clone())
            .collect();
        let path_names: Vec<String> = by_path[&PathBuf::from("src/lib.rs")]
            .iter()
            .map(|s| s.name.clone())
            .collect();
        assert_eq!(str_names, path_names, "key type must not affect contents");
    }

    #[test]
    fn build_symbols_by_file_skips_files_without_extension() {
        // tree-sitter dispatch is extension-driven; files with no extension
        // (Makefile, LICENSE) yield Err and must not appear in the map
        // rather than insert an empty Vec that callers would mistake for
        // "parsed but no symbols".
        let cache = cache_with_files(vec![
            ("README", "# project"),
            ("src/util.rs", "pub fn k() {}\n"),
        ]);

        let by_file: HashMap<String, Vec<crate::core::parser::CodeSymbol>> =
            build_symbols_by_file(&cache, |rel| rel.to_string());

        assert!(
            by_file.contains_key("src/util.rs"),
            "rust file must be present"
        );
        assert!(
            !by_file.contains_key("README"),
            "extensionless files must be omitted, not inserted as empty"
        );
    }

    #[test]
    fn build_symbols_by_file_empty_cache_returns_empty_map() {
        let cache = cache_with_files(vec![]);
        let by_file: HashMap<String, Vec<crate::core::parser::CodeSymbol>> =
            build_symbols_by_file(&cache, |rel| rel.to_string());
        assert!(by_file.is_empty());
    }

    // ---------------------------------------------------------------
    // camelCase arg-extraction regression tests (bug fix: MCP schema
    // advertises camelCase keys; handlers must accept them).
    // ---------------------------------------------------------------

    /// `get_str("top_k")` should find a key sent as `"topK"` (camelCase fallback).
    #[test]
    fn get_str_accepts_camel_case_fallback() {
        let mut args = serde_json::Map::new();
        args.insert("queryText".to_string(), json!("hello"));
        // snake key "query_text" → camel "queryText"
        assert_eq!(
            ContextPlusServer::get_str(&args, "query_text"),
            Some("hello".to_string()),
        );
    }

    /// `get_str` still works when only the snake_case key is present.
    #[test]
    fn get_str_snake_case_no_regression() {
        let mut args = serde_json::Map::new();
        args.insert("query_text".to_string(), json!("world"));
        assert_eq!(
            ContextPlusServer::get_str(&args, "query_text"),
            Some("world".to_string()),
        );
    }

    /// `get_str` prefers snake_case when both are present.
    #[test]
    fn get_str_prefers_snake_over_camel() {
        let mut args = serde_json::Map::new();
        args.insert("my_key".to_string(), json!("snake_value"));
        args.insert("myKey".to_string(), json!("camel_value"));
        assert_eq!(
            ContextPlusServer::get_str(&args, "my_key"),
            Some("snake_value".to_string()),
        );
    }

    /// `get_usize("top_k")` accepts `"topK": 3` sent by MCP client.
    #[test]
    fn get_usize_accepts_camel_case_top_k() {
        let mut args = serde_json::Map::new();
        args.insert("topK".to_string(), json!(3));
        assert_eq!(
            ContextPlusServer::get_usize(&args, "top_k"),
            Some(3),
            "topK (camelCase) must be resolved when top_k is absent",
        );
    }

    /// `get_usize("top_k")` still works with snake_case key — no regression.
    #[test]
    fn get_usize_snake_case_top_k_no_regression() {
        let mut args = serde_json::Map::new();
        args.insert("top_k".to_string(), json!(7));
        assert_eq!(ContextPlusServer::get_usize(&args, "top_k"), Some(7));
    }

    /// `get_f64("semantic_weight")` accepts `"semanticWeight": 0.9`.
    #[test]
    fn get_f64_accepts_camel_case_semantic_weight() {
        let mut args = serde_json::Map::new();
        args.insert("semanticWeight".to_string(), json!(0.9));
        let val = ContextPlusServer::get_f64(&args, "semantic_weight").unwrap();
        assert!((val - 0.9).abs() < f64::EPSILON);
    }

    /// `get_f64("min_combined_score")` accepts `"minCombinedScore": 0.4`.
    #[test]
    fn get_f64_accepts_camel_case_min_combined_score() {
        let mut args = serde_json::Map::new();
        args.insert("minCombinedScore".to_string(), json!(0.4));
        let val = ContextPlusServer::get_f64(&args, "min_combined_score").unwrap();
        assert!((val - 0.4).abs() < f64::EPSILON);
    }

    /// `get_bool("require_keyword_match")` accepts `"requireKeywordMatch": true`.
    #[test]
    fn get_bool_accepts_camel_case_require_keyword_match() {
        let mut args = serde_json::Map::new();
        args.insert("requireKeywordMatch".to_string(), json!(true));
        assert_eq!(
            ContextPlusServer::get_bool(&args, "require_keyword_match"),
            Some(true),
        );
    }

    /// `get_string_array("include_kinds")` accepts `"includeKinds": [...]`.
    #[test]
    fn get_string_array_accepts_camel_case_include_kinds() {
        let mut args = serde_json::Map::new();
        args.insert("includeKinds".to_string(), json!(["function", "method"]));
        let result = ContextPlusServer::get_string_array(&args, "include_kinds").unwrap();
        assert_eq!(result, vec!["function", "method"]);
    }

    /// `get_string_array("include_kinds")` still works with snake_case — no regression.
    #[test]
    fn get_string_array_snake_case_no_regression() {
        let mut args = serde_json::Map::new();
        args.insert("include_kinds".to_string(), json!(["class"]));
        let result = ContextPlusServer::get_string_array(&args, "include_kinds").unwrap();
        assert_eq!(result, vec!["class"]);
    }

    /// `get_u32("recency_window_days")` accepts `"recencyWindowDays": 30`.
    #[test]
    fn get_u32_accepts_camel_case_recency_window_days() {
        let mut args = serde_json::Map::new();
        args.insert("recencyWindowDays".to_string(), json!(30));
        assert_eq!(
            ContextPlusServer::get_u32(&args, "recency_window_days"),
            Some(30),
        );
    }

    // -----------------------------------------------------------------------
    // Warmup tests
    // -----------------------------------------------------------------------

    /// `warmup_semantic_search_cache` must not panic when Ollama is unreachable.
    /// The search will fail (embed call returns Err), which is caught and logged.
    #[tokio::test]
    async fn warmup_does_not_panic_on_ollama_error() {
        // Point at a guaranteed-dead host so the embed call fails immediately.
        unsafe {
            std::env::set_var("OLLAMA_HOST", "http://127.0.0.1:1");
        }
        let server = test_server();
        // Should return without panicking even when Ollama is unreachable.
        warmup_semantic_search_cache(&server.state).await;
        unsafe {
            std::env::remove_var("OLLAMA_HOST");
        }
    }

    /// `spawn_warmup_task` completes without panicking on a bare temp-dir state.
    #[tokio::test]
    async fn spawn_warmup_task_does_not_panic() {
        unsafe {
            std::env::set_var("OLLAMA_HOST", "http://127.0.0.1:1");
        }
        let server = test_server();
        server.spawn_warmup_task(false);
        // Brief yield so the spawned task has a chance to run.
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        unsafe {
            std::env::remove_var("OLLAMA_HOST");
        }
    }

    // Serialize env-var tests to prevent races between concurrent test threads.
    static WARMUP_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Config: `CONTEXTPLUS_WARMUP_ON_START` defaults to `true`.
    #[test]
    fn warmup_on_start_defaults_to_true() {
        let _guard = WARMUP_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::remove_var("CONTEXTPLUS_WARMUP_ON_START");
        }
        let cfg = Config::from_env();
        assert!(cfg.warmup_on_start);
    }

    /// Config: `CONTEXTPLUS_WARMUP_ON_START=false` disables warmup.
    #[test]
    fn warmup_on_start_can_be_disabled() {
        let _guard = WARMUP_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("CONTEXTPLUS_WARMUP_ON_START", "false");
        }
        let cfg = Config::from_env();
        assert!(!cfg.warmup_on_start);
        unsafe {
            std::env::remove_var("CONTEXTPLUS_WARMUP_ON_START");
        }
    }

    // -----------------------------------------------------------------------
    // Drain mode dispatch behavior
    // -----------------------------------------------------------------------

    fn extract_text(result: &CallToolResult) -> String {
        result
            .content
            .first()
            .and_then(|c| match &c.raw {
                RawContent::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn dispatch_rejects_new_calls_when_draining() {
        use std::sync::atomic::Ordering;

        let server = test_server();
        server.state.draining.store(true, Ordering::Release);

        let result = server
            .dispatch("get_context_tree", serde_json::Map::new())
            .await;

        assert_eq!(result.is_error, Some(true));
        let text = extract_text(&result);
        assert!(
            text.contains("shutting down"),
            "expected shutdown message, got: {text}"
        );
    }

    #[tokio::test]
    async fn dispatch_inflight_count_returns_to_zero_after_call() {
        use std::sync::atomic::Ordering;

        let server = test_server();
        let _ = server
            .dispatch("list_worktrees", serde_json::Map::new())
            .await;
        assert_eq!(server.state.inflight.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn dispatch_drain_rejects_unknown_tool_too() {
        use std::sync::atomic::Ordering;

        // Drain check must run before name dispatch so even unknown tools
        // get the consistent shutdown error during drain.
        let server = test_server();
        server.state.draining.store(true, Ordering::Release);
        let result = server
            .dispatch("nonexistent_tool", serde_json::Map::new())
            .await;
        let text = extract_text(&result);
        assert!(text.contains("shutting down"), "got: {text}");
    }

    // ── U9: with_session / session_ref_id ────────────────────────────────────

    /// `new()` produces a server with no session ref — stdio / no-handshake
    /// mode falls through to `default_ref` routing.
    #[test]
    fn new_server_has_no_session_ref_id() {
        let server = test_server();
        assert!(
            server.session_ref_id.is_none(),
            "freshly constructed server must have session_ref_id = None"
        );
    }

    /// `with_session` returns a clone that carries the given `RefId` while
    /// leaving the original unchanged.
    #[test]
    fn with_session_sets_ref_id_without_mutating_original() {
        use crate::ref_index::RefId;

        let server = test_server();
        let fake_id = RefId(0xdeadbeef_cafebabe);
        let session_server = server.with_session(fake_id);

        // Original is untouched.
        assert!(
            server.session_ref_id.is_none(),
            "original server must remain None after with_session"
        );
        // Clone carries the new id.
        assert_eq!(
            session_server.session_ref_id,
            Some(fake_id),
            "session copy must carry the provided RefId"
        );
        // Both share the same Arc<SharedState>.
        assert!(
            Arc::ptr_eq(&server.state, &session_server.state),
            "with_session must not clone SharedState — same Arc"
        );
    }

    /// `with_session` is idempotent: calling it again overwrites the ref_id.
    #[test]
    fn with_session_can_be_called_twice_overwriting_ref_id() {
        use crate::ref_index::RefId;

        let server = test_server();
        let id_a = RefId(1);
        let id_b = RefId(2);
        let s_a = server.with_session(id_a);
        let s_b = s_a.with_session(id_b);

        assert_eq!(s_a.session_ref_id, Some(id_a));
        assert_eq!(s_b.session_ref_id, Some(id_b));
    }

    // ── U10: per-ref state / backward-compat shims ───────────────────────────

    /// `state.embedding_cache` and `default_ref().embedding_cache` must share
    /// the same `Arc` — writes through one are visible through the other.
    #[tokio::test]
    async fn sharedstate_embedding_cache_shares_arc_with_default_ref() {
        let server = test_server();
        let default_ref = server
            .state
            .default_ref()
            .expect("default ref always present");

        assert!(
            Arc::ptr_eq(&server.state.embedding_cache, &default_ref.embedding_cache),
            "SharedState.embedding_cache must be the same Arc as default_ref().embedding_cache"
        );
    }

    /// `state.project_cache` and `default_ref().project_cache` share the same Arc.
    #[tokio::test]
    async fn sharedstate_project_cache_shares_arc_with_default_ref() {
        let server = test_server();
        let default_ref = server
            .state
            .default_ref()
            .expect("default ref always present");

        assert!(
            Arc::ptr_eq(&server.state.project_cache, &default_ref.project_cache),
            "SharedState.project_cache must share the Arc with default_ref().project_cache"
        );
    }

    /// `state.identifier_index` and `default_ref().identifier_index` share the same Arc.
    #[tokio::test]
    async fn sharedstate_identifier_index_shares_arc_with_default_ref() {
        let server = test_server();
        let default_ref = server
            .state
            .default_ref()
            .expect("default ref always present");

        assert!(
            Arc::ptr_eq(
                &server.state.identifier_index,
                &default_ref.identifier_index
            ),
            "SharedState.identifier_index must share the Arc with default_ref().identifier_index"
        );
    }

    /// `state.search_index_cache` and `default_ref().search_index_cache` share Arc.
    #[tokio::test]
    async fn sharedstate_search_index_cache_shares_arc_with_default_ref() {
        let server = test_server();
        let default_ref = server
            .state
            .default_ref()
            .expect("default ref always present");

        assert!(
            Arc::ptr_eq(
                &server.state.search_index_cache,
                &default_ref.search_index_cache
            ),
            "SharedState.search_index_cache must share the Arc with default_ref().search_index_cache"
        );
    }

    /// `state.cache_generation` and `default_ref().cache_generation` share Arc.
    #[tokio::test]
    async fn sharedstate_cache_generation_shares_arc_with_default_ref() {
        let server = test_server();
        let default_ref = server
            .state
            .default_ref()
            .expect("default ref always present");

        assert!(
            Arc::ptr_eq(
                &server.state.cache_generation,
                &default_ref.cache_generation
            ),
            "SharedState.cache_generation must share the Arc with default_ref().cache_generation"
        );
    }

    /// Writes through a non-default ref's `embedding_cache` do NOT appear in
    /// the default ref's (i.e. `SharedState`'s) cache.
    #[tokio::test]
    async fn non_default_ref_cache_is_isolated_from_default() {
        use crate::core::embeddings::CacheEntry;
        use crate::ref_index::{RefId, RefIndex};

        let server = test_server();

        // Attach a synthetic worktree ref.
        let worktree_path = std::path::PathBuf::from("/tmp/u10-compat-isolation-wt");
        let wt_ref_id = RefId::for_canonical_path(&worktree_path);
        let wt_ref_arc = server
            .state
            .attach_ref(wt_ref_id, || {
                Arc::new(RefIndex::new(
                    worktree_path.clone(),
                    worktree_path.clone(),
                    Some(server.state.default_ref_id),
                ))
            })
            .await;

        // Write into the worktree ref's cache.
        wt_ref_arc.embedding_cache.write().await.insert(
            "worktree_only.rs".to_string(),
            CacheEntry {
                hash: "wt_hash".to_string(),
                vector: vec![9.9],
            },
        );

        // The default ref (and SharedState shim) must not see this entry.
        assert!(
            !server
                .state
                .embedding_cache
                .read()
                .await
                .contains_key("worktree_only.rs"),
            "default ref cache must not contain worktree-only keys"
        );
    }

    #[tokio::test]
    async fn lane_m_attached_worktree_shares_primary_resident_identifier_vectors() {
        use crate::ref_index::{RefId, RefIndex};

        let server = test_server();
        let primary = server.state.default_ref().unwrap();
        let primary_vectors = Arc::new(RwLock::new(HashMap::from([(
            "fn shared_identifier()".to_string(),
            Arc::from(vec![0.25_f32; 768]),
        )])));
        primary
            .identifier_vectors
            .set(Arc::clone(&primary_vectors))
            .unwrap();

        let worktree_path = PathBuf::from("/tmp/lane-m-shared-identifier-vectors");
        let worktree_id = RefId::for_canonical_path(&worktree_path);
        let worktree = server
            .state
            .attach_ref(worktree_id, || {
                Arc::new(RefIndex::new(
                    worktree_path.clone(),
                    worktree_path,
                    Some(server.state.default_ref_id),
                ))
            })
            .await;
        let inherited = worktree
            .identifier_vectors
            .get()
            .expect("attach must install the primary's resident identifier vector base");

        assert!(
            Arc::ptr_eq(inherited, &primary_vectors),
            "an unchanged worktree must share, not clone, primary identifier vectors"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lane_m_attach_does_not_hold_registry_write_lock_while_inheriting_caches() {
        use crate::ref_index::{RefId, RefIndex};

        let server = test_server();
        let primary = server.state.default_ref().unwrap();
        let parent_cache = primary.identifier_index.write().await;
        let worktree_path = PathBuf::from("/tmp/lane-m-attach-lock-scope");
        let worktree_id = RefId::for_canonical_path(&worktree_path);
        let attaching = {
            let state = Arc::clone(&server.state);
            tokio::spawn(async move {
                state
                    .attach_ref(worktree_id, || {
                        Arc::new(RefIndex::new(
                            worktree_path.clone(),
                            worktree_path,
                            Some(state.default_ref_id),
                        ))
                    })
                    .await
            })
        };

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let registry_remained_readable = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            server.state.refs.read(),
        )
        .await
        .is_ok();
        drop(parent_cache);
        attaching.await.unwrap();

        assert!(
            registry_remained_readable,
            "attach held the registry write lock while awaiting a parent cache lock"
        );
    }

    #[tokio::test]
    async fn lane_m_worktree_identifier_misses_do_not_mutate_primary_resident_base() {
        use crate::config::{RefWarmupMode, TrackerMode};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let count = request
                    .body_json::<serde_json::Value>()
                    .ok()
                    .and_then(|body| body["input"].as_array().map(Vec::len))
                    .unwrap_or(1);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0, 0.0]; count]
                }))
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().unwrap();
        std::fs::write(primary.path().join("shared.rs"), "fn shared_symbol() {}\n").unwrap();
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);
        let primary_cache = server.ensure_project_cache().await.unwrap();
        server
            .ensure_identifier_index(&primary_cache)
            .await
            .unwrap();
        let primary_ref = server.state.default_ref().unwrap();
        let base = primary_ref.identifier_vectors.get().unwrap().clone();
        let keys_before = base
            .read()
            .await
            .keys()
            .cloned()
            .collect::<std::collections::HashSet<_>>();

        let worktree = tempfile::tempdir().unwrap();
        std::fs::write(
            worktree.path().join("shared.rs"),
            "fn shared_symbol() {}\nfn worktree_only_symbol() {}\n",
        )
        .unwrap();
        let canonical = worktree.path().canonicalize().unwrap();
        let mut attach = serde_json::Map::new();
        attach.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(attach).await.unwrap();
        let worktree_id = crate::ref_index::RefId::for_canonical_path(&canonical);
        let worktree_server = server.with_session(worktree_id);
        let worktree_ref = worktree_server.current_ref().await;
        *worktree_ref.identifier_index.write().await = None;
        *worktree_ref.identifier_source.write().await = None;
        let worktree_cache = worktree_server.ensure_project_cache().await.unwrap();
        worktree_server
            .ensure_identifier_index(&worktree_cache)
            .await
            .unwrap();

        let keys_after = base
            .read()
            .await
            .keys()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            keys_after, keys_before,
            "a worktree identifier miss was inserted into the primary's immutable base"
        );
    }

    #[tokio::test]
    async fn lane_m_worktree_first_identifier_load_is_shared_with_primary() {
        use crate::config::{RefWarmupMode, TrackerMode};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let count = request
                    .body_json::<serde_json::Value>()
                    .ok()
                    .and_then(|body| body["input"].as_array().map(Vec::len))
                    .unwrap_or(1);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0, 0.0]; count]
                }))
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        std::fs::write(worktree.path().join("only.rs"), "fn worktree_first() {}\n").unwrap();
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);
        let canonical = worktree.path().canonicalize().unwrap();
        let mut attach = serde_json::Map::new();
        attach.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(attach).await.unwrap();
        let worktree_server =
            server.with_session(crate::ref_index::RefId::for_canonical_path(&canonical));
        let cache = worktree_server.ensure_project_cache().await.unwrap();
        worktree_server
            .ensure_identifier_index(&cache)
            .await
            .unwrap();

        let worktree_base = worktree_server
            .current_ref()
            .await
            .identifier_vectors
            .get()
            .cloned()
            .unwrap();
        let primary_base = server
            .state
            .default_ref()
            .unwrap()
            .identifier_vectors
            .get()
            .cloned()
            .expect("a worktree-first load must populate the primary's shared base");
        assert!(
            Arc::ptr_eq(&worktree_base, &primary_base),
            "a worktree-first identifier load built a private copy of the primary base"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lane_m_evicted_worktree_overlay_reloads_from_disk_without_reembedding() {
        use crate::config::{RefWarmupMode, TrackerMode};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let count = request
                    .body_json::<serde_json::Value>()
                    .ok()
                    .and_then(|body| body["input"].as_array().map(Vec::len))
                    .unwrap_or(1);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0, 0.0]; count]
                }))
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().unwrap();
        std::fs::write(primary.path().join("shared.rs"), "fn shared_symbol() {}\n").unwrap();
        let worktree = tempfile::tempdir().unwrap();
        std::fs::write(
            worktree.path().join("shared.rs"),
            "fn shared_symbol() {}\nfn worktree_only_symbol() {}\n",
        )
        .unwrap();
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let id_cache_name = cache_name("identifier-embeddings", &config);
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);
        let canonical = worktree.path().canonicalize().unwrap();
        let mut attach = serde_json::Map::new();
        attach.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(attach).await.unwrap();
        let worktree_server =
            server.with_session(crate::ref_index::RefId::for_canonical_path(&canonical));
        let owner = worktree_server.current_ref().await;
        let cache = worktree_server.ensure_project_cache().await.unwrap();
        worktree_server
            .ensure_identifier_index(&cache)
            .await
            .unwrap();
        assert!(
            !owner.identifier_vector_overlay.read().await.is_empty(),
            "the worktree miss must land in its overlay"
        );
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while !matches!(
                rkyv_store::load_cache(&canonical, &id_cache_name),
                Ok(Some(_))
            ) {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the worktree overlay was not persisted");
        let embeds_before = ollama.received_requests().await.unwrap().len();

        clear_ref_heavy_caches(&owner, &id_cache_name).await;
        let cache = worktree_server.ensure_project_cache().await.unwrap();
        worktree_server
            .ensure_identifier_index(&cache)
            .await
            .unwrap();

        assert!(
            !owner.identifier_vector_overlay.read().await.is_empty(),
            "the evicted overlay was not reloaded"
        );
        assert_eq!(
            ollama.received_requests().await.unwrap().len(),
            embeds_before,
            "an evicted worktree re-embedded identifiers its persisted overlay already held"
        );
    }

    #[tokio::test]
    async fn lane_m_last_detach_cancels_identifier_persistence_and_drops_heavy_state() {
        use crate::config::{RefWarmupMode, TrackerMode};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let count = request
                    .body_json::<serde_json::Value>()
                    .ok()
                    .and_then(|body| body["input"].as_array().map(Vec::len))
                    .unwrap_or(1);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; count]
                }))
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        std::fs::write(
            worktree.path().join("persist.rs"),
            "fn persistence_keeps_owner_alive() {}\n",
        )
        .unwrap();
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);
        let canonical = worktree.path().canonicalize().unwrap();
        let mut attach = serde_json::Map::new();
        attach.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(attach).await.unwrap();
        let worktree_id = crate::ref_index::RefId::for_canonical_path(&canonical);
        let worktree_server = server.with_session(worktree_id);
        let cache = worktree_server.ensure_project_cache().await.unwrap();
        worktree_server
            .ensure_identifier_index(&cache)
            .await
            .unwrap();
        let owner = worktree_server.current_ref().await;
        let heavy_cache = Arc::downgrade(&owner.search_index_cache);
        drop(owner);
        drop(cache);

        server
            .state
            .detach_ref(worktree_id, std::time::Duration::ZERO)
            .await;
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if !server.state.refs.read().await.contains_key(&worktree_id) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("worktree ref was not evicted after its TTL");

        assert!(
            heavy_cache.upgrade().is_none(),
            "identifier persistence retained the evicted ref's heavy caches"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lane_m_last_detach_aborts_paused_semantic_rebuild_and_drops_heavy_state() {
        use crate::ref_index::{RefId, RefIndex};
        use crate::tools::semantic_search::{
            EmbedFn, SearchDocument, SemanticSearchOptions, semantic_code_search_owned,
        };
        use std::sync::atomic::{AtomicU32, Ordering};

        struct PausedRefWalker {
            ref_index: Arc<RefIndex>,
            calls: AtomicU32,
            started: Arc<tokio::sync::Notify>,
            release: Arc<tokio::sync::Notify>,
        }
        impl WalkAndIndexFn for PausedRefWalker {
            fn walk_and_index(
                &self,
                _root: &std::path::Path,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)>,
                        > + Send
                        + '_,
                >,
            > {
                let call = self.calls.fetch_add(1, Ordering::Relaxed);
                Box::pin(async move {
                    if call > 0 {
                        self.started.notify_one();
                        self.release.notified().await;
                    }
                    let docs = (0..25)
                        .map(|i| {
                            SearchDocument::new(
                                format!("src/file_{i}.rs"),
                                String::new(),
                                vec![],
                                vec![],
                                format!("paused rebuild content {call}"),
                            )
                        })
                        .collect::<Vec<_>>();
                    let vectors = vec![Some(vec![1.0, 0.0]); docs.len()];
                    Ok((docs, vectors))
                })
            }

            fn track_background_task(&self, task: &tokio::task::JoinHandle<()>) {
                self.ref_index.track_background_task(task);
            }
        }
        struct FixedEmbedder;
        impl EmbedFn for FixedEmbedder {
            fn embed(
                &self,
                _texts: &[String],
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>,
            > {
                Box::pin(async { Ok(vec![vec![1.0, 0.0]]) })
            }
        }

        let server = test_server();
        let worktree = tempfile::tempdir().unwrap();
        let root = worktree.path().canonicalize().unwrap();
        let worktree_id = RefId::for_canonical_path(&root);
        let owner = server
            .state
            .attach_ref(worktree_id, || {
                Arc::new(RefIndex::new(
                    root.clone(),
                    root.clone(),
                    Some(server.state.default_ref_id),
                ))
            })
            .await;
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let walker: Arc<dyn WalkAndIndexFn> = Arc::new(PausedRefWalker {
            ref_index: Arc::clone(&owner),
            calls: AtomicU32::new(0),
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        });
        let options = SemanticSearchOptions {
            root_dir: root.clone(),
            query: "paused rebuild content".to_string(),
            top_k: Some(5),
            semantic_weight: Some(0.0),
            keyword_weight: Some(1.0),
            min_semantic_score: None,
            min_keyword_score: Some(0.01),
            min_combined_score: None,
            require_keyword_match: Some(true),
            require_semantic_match: Some(false),
            include_globs: None,
            exclude_globs: None,
            recency_window_days: None,
            scope: None,
        };
        for generation in 0..2 {
            owner.cache_generation.store(generation, Ordering::Release);
            semantic_code_search_owned(
                options.clone(),
                &FixedEmbedder,
                Arc::clone(&walker),
                Some(Arc::clone(&owner.search_index_cache)),
                Some(Arc::clone(&owner.cache_generation)),
            )
            .await
            .unwrap();
        }
        tokio::time::timeout(std::time::Duration::from_secs(1), started.notified())
            .await
            .expect("the background semantic rebuild did not start");
        let stale_generation =
            Arc::downgrade(owner.search_index_cache.read().await.as_ref().unwrap());
        let heavy_cache = Arc::downgrade(&owner.search_index_cache);
        let evicted = Arc::downgrade(&owner);
        drop(walker);
        drop(owner);

        server
            .state
            .detach_ref(worktree_id, std::time::Duration::ZERO)
            .await;
        let released = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while evicted.upgrade().is_some()
                || heavy_cache.upgrade().is_some()
                || stale_generation.upgrade().is_some()
            {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await;
        drop(release);

        assert!(
            released.is_ok(),
            "a paused semantic rebuild kept the evicted ref's heavy state alive"
        );
    }

    #[tokio::test]
    async fn lane_m_detach_ttl_is_scoped_to_latest_zero_session_epoch() {
        use crate::ref_index::{RefId, RefIndex};

        let server = test_server();
        let path = PathBuf::from("/tmp/lane-m-detach-epoch");
        let ref_id = RefId::for_canonical_path(&path);
        server
            .state
            .attach_ref(ref_id, || {
                Arc::new(RefIndex::new(
                    path.clone(),
                    path.clone(),
                    Some(server.state.default_ref_id),
                ))
            })
            .await;
        let ttl = std::time::Duration::from_millis(120);
        server.state.detach_ref(ref_id, ttl).await;
        tokio::time::sleep(std::time::Duration::from_millis(90)).await;
        server
            .state
            .attach_ref(ref_id, || unreachable!("ref is still registered"))
            .await;
        server.state.detach_ref(ref_id, ttl).await;

        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
        assert!(
            server.state.refs.read().await.contains_key(&ref_id),
            "the first detach timer evicted the ref during the later detach epoch"
        );
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while server.state.refs.read().await.contains_key(&ref_id) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the latest detach epoch did not evict after its full TTL");
    }

    #[tokio::test]
    async fn lane_m_identifier_vectors_alone_are_memory_budget_evictable() {
        use crate::ref_index::{RefId, RefIndex};

        let mut config = Config::from_env();
        config.resident_memory_budget_bytes = 1024;
        let root = tempfile::tempdir().unwrap();
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        let path = PathBuf::from("/tmp/lane-m-identifier-only-budget");
        let ref_id = RefId::for_canonical_path(&path);
        let owner = Arc::new(RefIndex::new(path.clone(), path, None));
        owner
            .identifier_vectors
            .set(Arc::new(RwLock::new(HashMap::from([(
                "worktree-only-vector".to_string(),
                Arc::from(vec![0.5_f32; 4096]),
            )]))))
            .unwrap();
        let attached = server.state.attach_ref(ref_id, || Arc::clone(&owner)).await;
        mark_ref_idle(&server.state, ref_id);

        server.state.enforce_memory_budget().await;

        assert!(
            attached
                .identifier_vectors
                .get()
                .unwrap()
                .read()
                .await
                .is_empty(),
            "identifier-vector-only pressure was not reclaimed from the non-primary ref"
        );
        assert!(server.state.refs.read().await.contains_key(&ref_id));
    }

    #[tokio::test]
    async fn lane_m_dispatch_does_not_wait_for_synchronous_heap_scan() {
        let server = test_server();
        let primary = server.state.default_ref().unwrap();
        let cache_guard = primary.embedding_cache.write().await;
        let mut dispatch = tokio::spawn({
            let server = server.clone();
            async move {
                server
                    .dispatch("nonexistent_tool", serde_json::Map::new())
                    .await
            }
        });

        let returned_without_cache_lock =
            tokio::time::timeout(std::time::Duration::from_millis(100), &mut dispatch)
                .await
                .is_ok();
        drop(cache_guard);
        if !dispatch.is_finished() {
            dispatch.await.unwrap();
        }

        assert!(
            returned_without_cache_lock,
            "tool response waited for memory accounting to acquire a heavy-cache lock"
        );
    }

    #[tokio::test]
    async fn lane_m_memory_budget_evicts_lru_non_primary_caches_only() {
        use crate::core::embeddings::CacheEntry;
        use crate::ref_index::{RefId, RefIndex};

        let mut config = Config::from_env();
        config.resident_memory_budget_bytes = 11 * 1024;
        let root = tempfile::tempdir().unwrap();
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        let primary = server.state.default_ref().unwrap();

        let path_a = PathBuf::from("/tmp/lane-m-budget-a");
        let path_b = PathBuf::from("/tmp/lane-m-budget-b");
        let id_a = RefId::for_canonical_path(&path_a);
        let id_b = RefId::for_canonical_path(&path_b);
        let ref_a = server
            .state
            .attach_ref(id_a, || {
                Arc::new(RefIndex::new(
                    path_a.clone(),
                    path_a,
                    Some(server.state.default_ref_id),
                ))
            })
            .await;
        let ref_b = server
            .state
            .attach_ref(id_b, || {
                Arc::new(RefIndex::new(
                    path_b.clone(),
                    path_b,
                    Some(server.state.default_ref_id),
                ))
            })
            .await;

        for (owner, name) in [(&primary, "primary"), (&ref_a, "a"), (&ref_b, "b")] {
            owner.embedding_cache.write().await.insert(
                format!("{name}.rs"),
                CacheEntry {
                    hash: format!("{name}-hash"),
                    vector: vec![0.5; 1024],
                },
            );
        }
        server.state.touch_ref(id_a);
        server.state.touch_ref(id_b);
        mark_ref_idle(&server.state, id_a);
        mark_ref_idle(&server.state, id_b);

        server.state.enforce_memory_budget().await;

        assert!(
            !primary.embedding_cache.read().await.is_empty(),
            "the primary ref's heavy caches must never be evicted"
        );
        assert!(
            ref_a.embedding_cache.read().await.is_empty(),
            "the least recently used non-primary ref must be evicted first"
        );
        assert!(
            !ref_b.embedding_cache.read().await.is_empty(),
            "the most recently used worktree must remain when one eviction is sufficient"
        );
        assert!(server.state.ref_index(id_a).await.is_some());
    }

    #[tokio::test]
    async fn lane_m_memory_budget_never_evicts_a_ref_serving_a_request() {
        use crate::core::embeddings::CacheEntry;
        use crate::ref_index::{RefId, RefIndex};

        let mut config = Config::from_env();
        config.resident_memory_budget_bytes = 1024;
        let root = tempfile::tempdir().unwrap();
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        let path = PathBuf::from("/tmp/lane-m-budget-serving");
        let ref_id = RefId::for_canonical_path(&path);
        let owner = server
            .state
            .attach_ref(ref_id, || {
                Arc::new(RefIndex::new(
                    path.clone(),
                    path,
                    Some(server.state.default_ref_id),
                ))
            })
            .await;
        owner.embedding_cache.write().await.insert(
            "serving.rs".to_string(),
            CacheEntry {
                hash: "serving-hash".to_string(),
                vector: vec![0.5; 4096],
            },
        );

        mark_ref_idle(&server.state, ref_id);
        let serving =
            crate::core::process_lifecycle::InflightGuard::new(Arc::clone(&owner.active_requests));
        server.state.enforce_memory_budget().await;
        assert!(
            !owner.embedding_cache.read().await.is_empty(),
            "the budget evicted the caches of a ref that was serving a request"
        );

        drop(serving);
        server.state.enforce_memory_budget().await;
        assert!(
            owner.embedding_cache.read().await.is_empty(),
            "an idle over-budget ref must still be evicted"
        );
    }

    #[tokio::test]
    async fn lane_m_memory_budget_skips_worktrees_holding_only_shared_caches() {
        use crate::core::embeddings::CacheEntry;
        use crate::ref_index::{RefId, RefIndex};

        let mut config = Config::from_env();
        config.resident_memory_budget_bytes = 10 * 1024;
        let root = tempfile::tempdir().unwrap();
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        let primary = server.state.default_ref().unwrap();
        *primary.identifier_index.write().await = Some(Arc::new(IdentifierIndex {
            docs: Vec::new().into(),
            vectors: IdentifierVectorIndex::from_vectors(vec![vec![0.5_f32; 4]; 512], 4),
            dims: 4,
            file_count: 0,
            built_at: Instant::now(),
        }));
        *primary.identifier_source.write().await = Some(Arc::new(ProjectCache {
            file_entries: Vec::new(),
            file_content: Default::default(),
            clean_blobs: None,
            last_refresh: Instant::now(),
        }));

        let attach = |name: &str| {
            let path = PathBuf::from(format!("/tmp/lane-m-budget-shared-{name}"));
            let id = RefId::for_canonical_path(&path);
            let state = Arc::clone(&server.state);
            async move {
                let owner = state
                    .attach_ref(id, || {
                        Arc::new(RefIndex::new(
                            path.clone(),
                            path,
                            Some(state.default_ref_id),
                        ))
                    })
                    .await;
                (id, owner)
            }
        };
        let (shared_id, shared_only) = attach("only").await;
        let (unique_id, unique) = attach("unique").await;
        unique.embedding_cache.write().await.insert(
            "unique.rs".to_string(),
            CacheEntry {
                hash: "unique-hash".to_string(),
                vector: vec![0.5; 1024],
            },
        );
        server.state.touch_ref(shared_id);
        server.state.touch_ref(unique_id);
        mark_ref_idle(&server.state, shared_id);
        mark_ref_idle(&server.state, unique_id);
        *server.state.measured_resident_override.lock().unwrap() = Some(20 * 1024);

        server.state.enforce_memory_budget().await;

        assert!(
            shared_only.identifier_index.read().await.is_some(),
            "a worktree holding only the primary's shared caches was evicted"
        );
        assert!(
            unique.embedding_cache.read().await.is_empty(),
            "the worktree with unique resident bytes must be evicted"
        );
    }

    #[tokio::test]
    async fn lane_m_attached_worktree_does_not_inherit_primary_semantic_index() {
        use crate::ref_index::{RefId, RefIndex};
        use crate::tools::semantic_search::{CachedSearchIndex, IndexFingerprint, SearchIndex};

        let server = test_server();
        let primary = server.state.default_ref().unwrap();
        *primary.search_index_cache.write().await = Some(Arc::new(CachedSearchIndex::new(
            SearchIndex::new(),
            IndexFingerprint::from_docs(&[]),
            0,
        )));
        let path = PathBuf::from("/tmp/lane-m-no-semantic-inheritance");
        let id = RefId::for_canonical_path(&path);
        let worktree = server
            .state
            .attach_ref(id, || {
                Arc::new(RefIndex::new(
                    path.clone(),
                    path,
                    Some(server.state.default_ref_id),
                ))
            })
            .await;

        assert!(
            worktree.search_index_cache.read().await.is_none(),
            "a worktree inherited the primary's semantic index, which its fill could mutate"
        );
    }

    #[tokio::test]
    async fn lane_m_divergent_worktree_first_keywords_does_not_serve_primary_lexical_index() {
        use crate::config::{RefWarmupMode, TrackerMode};

        let primary = tempfile::tempdir().unwrap();
        for i in 0..10 {
            std::fs::write(
                primary.path().join(format!("primary_only_{i}.rs")),
                format!("fn primary_only_symbol_{i}() {{}}\n"),
            )
            .unwrap();
        }
        let mut config = Config::from_env();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);
        let primary_cache = server.ensure_project_cache().await.unwrap();
        server.ensure_lexical_index(&primary_cache).await.unwrap();

        let worktree = tempfile::tempdir().unwrap();
        for i in 0..10 {
            std::fs::write(
                worktree.path().join(format!("worktree_only_{i}.rs")),
                format!("fn worktree_only_symbol_{i}() {{}}\n"),
            )
            .unwrap();
        }
        let canonical = worktree.path().canonicalize().unwrap();
        let mut attach = serde_json::Map::new();
        attach.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(attach).await.unwrap();
        let worktree_server =
            server.with_session(crate::ref_index::RefId::for_canonical_path(&canonical));
        let cache = worktree_server.ensure_project_cache().await.unwrap();
        let lexical = worktree_server.ensure_lexical_index(&cache).await.unwrap();

        let paths = lexical.document_paths.to_vec();
        assert!(
            !paths.iter().any(|path| path.starts_with("primary_only_")),
            "worktree's first keywords index is the primary's: {paths:?}"
        );
        assert!(
            paths.iter().any(|path| path.starts_with("worktree_only_")),
            "worktree's first keywords index misses its own files: {paths:?}"
        );
    }

    #[tokio::test]
    async fn lane_m_divergent_worktree_first_identifiers_do_not_serve_primary_index() {
        use crate::config::{RefWarmupMode, TrackerMode};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let count = embed_request_inputs(request).len().max(1);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0, 0.0]; count]
                }))
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().unwrap();
        for i in 0..10 {
            std::fs::write(
                primary.path().join(format!("primary_only_{i}.rs")),
                format!("fn primary_only_symbol_{i}() {{}}\n"),
            )
            .unwrap();
        }
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);
        let primary_cache = server.ensure_project_cache().await.unwrap();
        server
            .ensure_identifier_index(&primary_cache)
            .await
            .unwrap();

        let worktree = tempfile::tempdir().unwrap();
        for i in 0..10 {
            std::fs::write(
                worktree.path().join(format!("worktree_only_{i}.rs")),
                format!("fn worktree_only_symbol_{i}() {{}}\n"),
            )
            .unwrap();
        }
        let canonical = worktree.path().canonicalize().unwrap();
        let mut attach = serde_json::Map::new();
        attach.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(attach).await.unwrap();
        let worktree_server =
            server.with_session(crate::ref_index::RefId::for_canonical_path(&canonical));
        let cache = worktree_server.ensure_project_cache().await.unwrap();
        let index = worktree_server
            .ensure_identifier_index(&cache)
            .await
            .unwrap();

        let paths: std::collections::BTreeSet<_> =
            index.docs.iter().map(|doc| doc.path.clone()).collect();
        assert!(
            !paths.iter().any(|path| path.starts_with("primary_only_")),
            "worktree's first identifiers index is the primary's: {paths:?}"
        );
        assert!(
            paths.iter().any(|path| path.starts_with("worktree_only_")),
            "worktree's first identifiers index misses its own files: {paths:?}"
        );
    }

    async fn identifier_test_server(
        ollama: &wiremock::MockServer,
        root: &std::path::Path,
    ) -> ContextPlusServer {
        use crate::config::{RefWarmupMode, TrackerMode};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, Request, ResponseTemplate};

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let vectors: Vec<Vec<f32>> = embed_request_inputs(request)
                    .iter()
                    .map(|input| vec![input.len() as f32, 1.0, 0.0])
                    .collect();
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embeddings": vectors }))
            })
            .mount(ollama)
            .await;
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        ContextPlusServer::new(root.to_path_buf(), config)
    }

    async fn attached_worktree(
        server: &ContextPlusServer,
        root: &std::path::Path,
    ) -> ContextPlusServer {
        let canonical = root.canonicalize().unwrap();
        let mut attach = serde_json::Map::new();
        attach.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(attach).await.unwrap();
        server.with_session(crate::ref_index::RefId::for_canonical_path(&canonical))
    }

    /// After a restart a worktree may query before the primary has built its
    /// identifier index; its build then shares the primary's documents of
    /// every identical file and parses only the files that differ.
    #[tokio::test]
    async fn worktree_identifier_build_reuses_primary_documents_of_identical_files() {
        let ollama = wiremock::MockServer::start().await;
        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        for dir in [primary.path(), worktree.path()] {
            std::fs::write(dir.join("same_a.rs"), "fn same_a() {}\n").unwrap();
            std::fs::write(dir.join("same_b.rs"), "fn same_b() {}\n").unwrap();
        }
        std::fs::write(primary.path().join("differs.rs"), "fn old_name() {}\n").unwrap();
        std::fs::write(worktree.path().join("differs.rs"), "fn new_name() {}\n").unwrap();
        let server = identifier_test_server(&ollama, primary.path()).await;
        let worktree_server = attached_worktree(&server, worktree.path()).await;

        let cache = worktree_server.ensure_project_cache().await.unwrap();
        let index = worktree_server
            .ensure_identifier_index(&cache)
            .await
            .unwrap();

        let primary_index = server
            .state
            .default_ref()
            .unwrap()
            .identifier_index
            .read()
            .await
            .clone()
            .expect("the worktree build seeds from the primary's identifier index");
        for path in ["same_a.rs", "same_b.rs"] {
            assert!(
                Arc::ptr_eq(&index.docs.files[path], &primary_index.docs.files[path]),
                "{path} is identical, so its documents are the primary's"
            );
            assert!(
                Arc::ptr_eq(
                    &index.vectors.file_segments()[path],
                    &primary_index.vectors.file_segments()[path]
                ),
                "{path} is identical, so its vectors are the primary's"
            );
        }
        assert!(!Arc::ptr_eq(
            &index.docs.files["differs.rs"],
            &primary_index.docs.files["differs.rs"]
        ));
        assert_eq!(index.docs.files["differs.rs"][0].name, "new_name");
    }

    /// A worktree build seeded from the primary's identifier index answers
    /// exactly as a full parse of the worktree does, whether its files are
    /// identical, changed, added or deleted.
    #[tokio::test]
    async fn worktree_identifier_build_from_primary_equals_full_parse() {
        fn documents(index: &IdentifierIndex) -> Vec<String> {
            let sorted = |tokens: &std::collections::HashSet<String>| {
                let mut tokens: Vec<_> = tokens.iter().cloned().collect();
                tokens.sort();
                tokens
            };
            index
                .docs
                .iter()
                .map(|doc| {
                    format!(
                        "{} {} {} {} {} {} {} {} {:?} {} {:?} {:?} {:?}",
                        doc.id,
                        doc.path,
                        doc.header,
                        doc.name,
                        doc.kind,
                        doc.line,
                        doc.end_line,
                        doc.signature,
                        doc.parent_name,
                        doc.text,
                        sorted(&doc.name_token_set),
                        sorted(&doc.signature_token_set),
                        sorted(&doc.parent_token_set),
                    )
                })
                .collect()
        }
        fn vectors(index: &IdentifierIndex) -> Vec<(String, Vec<Vec<f32>>)> {
            index
                .vectors
                .file_segments()
                .iter()
                .map(|(path, vectors)| {
                    (
                        path.clone(),
                        vectors.iter().map(|vector| vector.to_vec()).collect(),
                    )
                })
                .collect()
        }

        let ollama = wiremock::MockServer::start().await;
        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        let reference = tempfile::tempdir().unwrap();
        std::fs::write(
            primary.path().join("same.rs"),
            "struct Kept;\nimpl Kept {\n    fn kept(&self) {}\n}\n",
        )
        .unwrap();
        std::fs::write(primary.path().join("changed.rs"), "fn before() {}\n").unwrap();
        std::fs::write(primary.path().join("deleted.rs"), "fn deleted() {}\n").unwrap();
        for dir in [worktree.path(), reference.path()] {
            std::fs::copy(primary.path().join("same.rs"), dir.join("same.rs")).unwrap();
            std::fs::write(
                dir.join("changed.rs"),
                "fn after() {}\nfn after_too(x: u8) {}\n",
            )
            .unwrap();
            std::fs::write(dir.join("added.rs"), "fn added() {}\n").unwrap();
        }
        let server = identifier_test_server(&ollama, primary.path()).await;
        let worktree_server = attached_worktree(&server, worktree.path()).await;
        let cache = worktree_server.ensure_project_cache().await.unwrap();
        let seeded = worktree_server
            .ensure_identifier_index(&cache)
            .await
            .unwrap();

        let full_server = identifier_test_server(&ollama, reference.path()).await;
        let full_cache = full_server.ensure_project_cache().await.unwrap();
        let full = full_server
            .ensure_identifier_index(&full_cache)
            .await
            .unwrap();

        assert!(!seeded.docs.files.contains_key("deleted.rs"));
        assert_eq!(documents(&seeded), documents(&full));
        assert_eq!(vectors(&seeded), vectors(&full));
        assert_eq!(seeded.dims, full.dims);
        assert_eq!(seeded.file_count, full.file_count);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lane_m_last_detach_aborts_pending_fill_and_drops_evicted_ref() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                let response = ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; inputs.len().max(1)]
                }));
                if inputs
                    .iter()
                    .any(|input| input.contains("PENDING_FILL_MARKER"))
                {
                    response.set_delay(std::time::Duration::from_secs(30))
                } else {
                    response
                }
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        std::fs::write(
            worktree.path().join("fill.rs"),
            "fn PENDING_FILL_MARKER() {}\n",
        )
        .unwrap();
        let server = ContextPlusServer::new(
            primary.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 20, 60_000),
        );
        let canonical = worktree.path().canonicalize().unwrap();
        let mut attach = serde_json::Map::new();
        attach.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(attach).await.unwrap();
        let worktree_id = crate::ref_index::RefId::for_canonical_path(&canonical);
        let worktree_server = server.with_session(worktree_id);
        worktree_server
            .handle_semantic_code_search(semantic_args("pending fill"))
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while matching_embed_request_batches(&ollama, "PENDING_FILL_MARKER")
                .await
                .len()
                < 2
            {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the background fill never sent its embed request");
        let evicted = Arc::downgrade(&worktree_server.current_ref().await);
        drop(worktree_server);

        server
            .state
            .detach_ref(worktree_id, std::time::Duration::ZERO)
            .await;
        let released = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while evicted.upgrade().is_some() {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await;

        assert!(
            released.is_ok(),
            "a pending semantic fill kept the evicted ref alive"
        );
    }

    #[tokio::test]
    async fn lane_m_last_detach_keeps_pending_identifier_overlay_save() {
        use crate::config::{RefWarmupMode, TrackerMode};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let count = embed_request_inputs(request).len().max(1);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; count]
                }))
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        std::fs::write(
            worktree.path().join("overlay.rs"),
            "fn detached_overlay_symbol() {}\n",
        )
        .unwrap();
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let id_cache_name = cache_name("identifier-embeddings", &config);
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);
        let canonical = worktree.path().canonicalize().unwrap();
        let mut attach = serde_json::Map::new();
        attach.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(attach).await.unwrap();
        let worktree_id = crate::ref_index::RefId::for_canonical_path(&canonical);
        let worktree_server = server.with_session(worktree_id);
        let cache = worktree_server.ensure_project_cache().await.unwrap();
        worktree_server
            .ensure_identifier_index(&cache)
            .await
            .unwrap();
        drop(worktree_server);

        server
            .state
            .detach_ref(worktree_id, std::time::Duration::ZERO)
            .await;
        let saved = tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while !matches!(
                rkyv_store::load_cache(&canonical, &id_cache_name),
                Ok(Some(_))
            ) {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await;

        assert!(
            saved.is_ok(),
            "evicting the worktree dropped its pending identifier overlay save"
        );
    }

    fn mark_ref_idle(state: &SharedState, ref_id: crate::ref_index::RefId) {
        let idle_since = Instant::now()
            .checked_sub(MEMORY_BUDGET_MIN_IDLE * 2)
            .unwrap();
        if let Some(access) = state.ref_access.lock().unwrap().get_mut(&ref_id) {
            access.1 = idle_since;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn review_r3_budget_snapshot_does_not_deadlock_with_identifier_install() {
        let server = test_server();
        let owner = server.state.default_ref().unwrap();
        let project_read = owner.project_cache.read().await;
        let writer_owner = Arc::clone(&owner);
        let _writer = tokio::spawn(async move {
            *writer_owner.project_cache.write().await = None;
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let snapshot_owner = Arc::clone(&owner);
        let _snapshot = tokio::spawn(async move {
            let _ = ResidentSnapshot::capture(&snapshot_owner).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let acquired = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            owner.identifier_index.write(),
        )
        .await;
        assert!(
            acquired.is_ok(),
            "deadlock: the budget snapshot holds identifier_index.read while queued on project_cache.read"
        );
        drop(acquired);
        drop(project_read);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn review_r3_budget_eviction_stops_background_identifier_rebuild() {
        use crate::config::{RefWarmupMode, TrackerMode};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                let response = ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; inputs.len().max(1)]
                }));
                if inputs.iter().any(|input| input.contains("rebuild_marker")) {
                    response.set_delay(std::time::Duration::from_secs(1))
                } else {
                    response
                }
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        for i in 0..5 {
            std::fs::write(
                worktree.path().join(format!("base_{i}.rs")),
                format!("fn base_symbol_{i}() {{}}\n"),
            )
            .unwrap();
        }
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        config.resident_memory_budget_bytes = 1;
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);
        let canonical = worktree.path().canonicalize().unwrap();
        let mut attach = serde_json::Map::new();
        attach.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(attach).await.unwrap();
        let worktree_id = crate::ref_index::RefId::for_canonical_path(&canonical);
        let worktree_server = server.with_session(worktree_id);
        let cache = worktree_server.ensure_project_cache().await.unwrap();
        worktree_server
            .ensure_identifier_index(&cache)
            .await
            .unwrap();
        for i in 0..5 {
            std::fs::write(
                worktree.path().join(format!("rebuild_marker_{i}.rs")),
                format!("fn rebuild_marker_{i}() {{}}\n"),
            )
            .unwrap();
        }
        let owner = worktree_server.current_ref().await;
        *owner.project_cache.write().await = None;
        let fresh = worktree_server.ensure_project_cache().await.unwrap();
        worktree_server
            .ensure_identifier_index(&fresh)
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while matching_embed_request_batches(&ollama, "rebuild_marker")
                .await
                .is_empty()
            {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the background identifier rebuild never sent its embed request");

        mark_ref_idle(&server.state, worktree_id);
        server.state.enforce_memory_budget().await;
        assert!(owner.project_cache.read().await.is_none());
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;

        assert!(
            owner.project_cache.read().await.is_none()
                && owner.identifier_index.read().await.is_none(),
            "budget eviction was undone by the ref's background identifier rebuild"
        );
    }

    #[tokio::test]
    async fn review_r3_attach_does_not_inherit_identifier_index_without_its_source() {
        use crate::ref_index::{RefId, RefIndex};

        let server = test_server();
        let primary = server.state.default_ref().unwrap();
        *primary.identifier_index.write().await = Some(Arc::new(IdentifierIndex {
            docs: Vec::new().into(),
            vectors: IdentifierVectorIndex::from_vectors(vec![vec![0.5_f32; 4]; 2], 4),
            dims: 4,
            file_count: 0,
            built_at: Instant::now(),
        }));
        *primary.identifier_source.write().await = None;
        let path = PathBuf::from("/tmp/review-r3-sourceless-identifier-inheritance");
        let worktree = server
            .state
            .attach_ref(RefId::for_canonical_path(&path), || {
                Arc::new(RefIndex::new(
                    path.clone(),
                    path,
                    Some(server.state.default_ref_id),
                ))
            })
            .await;

        assert!(
            worktree.identifier_index.read().await.is_none(),
            "a worktree inherited an identifier index without the source it was built from"
        );
    }

    #[tokio::test]
    async fn review_r3_inherited_identifier_index_without_source_is_not_served() {
        use crate::config::{RefWarmupMode, TrackerMode};

        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        let mut config = Config::from_env();
        config.ollama_host = "http://127.0.0.1:1".to_string();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);
        let canonical = worktree.path().canonicalize().unwrap();
        let mut attach = serde_json::Map::new();
        attach.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(attach).await.unwrap();
        let worktree_server =
            server.with_session(crate::ref_index::RefId::for_canonical_path(&canonical));
        let owner = worktree_server.current_ref().await;
        let inherited = Arc::new(IdentifierIndex {
            docs: Vec::new().into(),
            vectors: IdentifierVectorIndex::from_vectors(vec![vec![0.5_f32; 4]; 2], 4),
            dims: 4,
            file_count: 0,
            built_at: Instant::now(),
        });
        *owner.identifier_index.write().await = Some(Arc::clone(&inherited));
        *owner.identifier_source.write().await = None;
        owner
            .identifier_inherited
            .store(true, std::sync::atomic::Ordering::Release);

        let cache = worktree_server.ensure_project_cache().await.unwrap();
        let index = worktree_server
            .ensure_identifier_index(&cache)
            .await
            .unwrap();

        assert!(
            !Arc::ptr_eq(&index, &inherited),
            "the fast path served an inherited identifier index"
        );
    }

    async fn attach_budget_worktree(
        server: &ContextPlusServer,
        name: &str,
    ) -> (crate::ref_index::RefId, Arc<crate::ref_index::RefIndex>) {
        use crate::core::embeddings::CacheEntry;
        use crate::ref_index::{RefId, RefIndex};

        let path = PathBuf::from(format!("/tmp/review-r3-budget-{name}"));
        let id = RefId::for_canonical_path(&path);
        let owner = server
            .state
            .attach_ref(id, || {
                Arc::new(RefIndex::new(
                    path.clone(),
                    path,
                    Some(server.state.default_ref_id),
                ))
            })
            .await;
        owner.embedding_cache.write().await.insert(
            format!("{name}.rs"),
            CacheEntry {
                hash: format!("{name}-hash"),
                vector: vec![0.5; 4096],
            },
        );
        (id, owner)
    }

    #[tokio::test]
    async fn review_r3_memory_budget_evicts_only_idle_worktrees() {
        let mut config = Config::from_env();
        config.resident_memory_budget_bytes = 1024;
        let root = tempfile::tempdir().unwrap();
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        let (id_a, ref_a) = attach_budget_worktree(&server, "a").await;
        let (_, ref_b) = attach_budget_worktree(&server, "b").await;

        server.state.enforce_memory_budget().await;
        assert!(
            !ref_a.embedding_cache.read().await.is_empty()
                && !ref_b.embedding_cache.read().await.is_empty(),
            "the budget evicted a worktree used within the idle window"
        );

        mark_ref_idle(&server.state, id_a);
        server.state.enforce_memory_budget().await;
        assert!(
            ref_a.embedding_cache.read().await.is_empty(),
            "an idle over-budget worktree must be evicted"
        );
        assert!(
            !ref_b.embedding_cache.read().await.is_empty(),
            "the budget evicted a worktree used within the idle window"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn process_resident_bytes_reads_this_process() {
        let touched = vec![1_u8; 64 * 1024 * 1024];
        let resident = process_resident_bytes().unwrap();
        assert!(resident >= touched.len(), "{resident}");
    }

    #[tokio::test]
    async fn memory_budget_triggers_on_measured_memory_when_estimates_are_under_it() {
        let mut config = Config::from_env();
        config.resident_memory_budget_bytes = 1024 * 1024;
        let root = tempfile::tempdir().unwrap();
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        let (id, worktree) = attach_budget_worktree(&server, "measured-over").await;
        mark_ref_idle(&server.state, id);
        *server.state.measured_resident_override.lock().unwrap() = Some(2 * 1024 * 1024);

        server.state.enforce_memory_budget().await;

        assert!(
            worktree.embedding_cache.read().await.is_empty(),
            "an idle worktree survived measured memory over the budget"
        );
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[tokio::test]
    async fn memory_under_budget_is_returned_to_the_os_when_mostly_freed() {
        let mut config = Config::from_env();
        config.resident_memory_budget_bytes = 256 * 1024 * 1024 * 1024;
        let root = tempfile::tempdir().unwrap();
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        *server.state.measured_resident_override.lock().unwrap() = Some(128 * 1024 * 1024 * 1024);

        server.state.enforce_memory_budget().await;

        assert!(
            server.state.last_budget_trim.lock().unwrap().is_some(),
            "memory the allocator holds free was not returned"
        );
    }

    #[tokio::test]
    async fn memory_under_budget_checks_freed_memory_at_most_once_per_trim_interval() {
        let mut config = Config::from_env();
        config.resident_memory_budget_bytes = 256 * 1024 * 1024 * 1024;
        let root = tempfile::tempdir().unwrap();
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        *server.state.measured_resident_override.lock().unwrap() = Some(128 * 1024 * 1024 * 1024);

        server.state.enforce_memory_budget().await;
        server.state.enforce_memory_budget().await;

        assert_eq!(
            server
                .state
                .free_memory_checks
                .load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the allocator was walked again within the trim interval"
        );
    }

    #[tokio::test]
    async fn memory_budget_ignores_estimates_while_measured_memory_is_under_it() {
        let mut config = Config::from_env();
        config.resident_memory_budget_bytes = 1024;
        let root = tempfile::tempdir().unwrap();
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        let (id, worktree) = attach_budget_worktree(&server, "measured-under").await;
        mark_ref_idle(&server.state, id);
        *server.state.measured_resident_override.lock().unwrap() = Some(512);

        server.state.enforce_memory_budget().await;

        assert!(
            !worktree.embedding_cache.read().await.is_empty(),
            "a worktree was evicted while measured memory was under the budget"
        );
    }

    #[tokio::test]
    async fn primary_alone_over_budget_warns_once_and_keeps_its_caches() {
        use crate::core::embeddings::CacheEntry;

        let mut config = Config::from_env();
        config.resident_memory_budget_bytes = 1024 * 1024;
        let root = tempfile::tempdir().unwrap();
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        let primary = server.state.default_ref().unwrap();
        primary.embedding_cache.write().await.insert(
            "primary.rs".to_string(),
            CacheEntry {
                hash: "primary-hash".to_string(),
                vector: vec![0.5; 4096],
            },
        );
        *server.state.measured_resident_override.lock().unwrap() = Some(2 * 1024 * 1024);
        let (logs, _capture) = crate::test_logs::captured_info_logs();

        server.state.enforce_memory_budget().await;
        server.state.enforce_memory_budget().await;

        let logs = crate::test_logs::logs_as_string(&logs);
        assert_eq!(
            logs.matches("over CONTEXTPLUS_MEMORY_BUDGET_MB").count(),
            1,
            "{logs}"
        );
        assert!(logs.contains("file_vectors="), "{logs}");
        assert!(!primary.embedding_cache.read().await.is_empty());
    }

    #[tokio::test]
    async fn review_r3_memory_budget_evicts_down_to_low_watermark() {
        let mut config = Config::from_env();
        config.resident_memory_budget_bytes = 40_000;
        let root = tempfile::tempdir().unwrap();
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        let mut refs = Vec::new();
        for name in ["a", "b", "c"] {
            refs.push(attach_budget_worktree(&server, name).await);
        }
        for (id, _) in &refs {
            mark_ref_idle(&server.state, *id);
        }

        server.state.enforce_memory_budget().await;

        assert!(
            refs[0].1.embedding_cache.read().await.is_empty()
                && refs[1].1.embedding_cache.read().await.is_empty(),
            "the budget stopped just under the limit instead of the low watermark"
        );
        assert!(!refs[2].1.embedding_cache.read().await.is_empty());
    }

    // ── U11: current_ref() routing ───────────────────────────────────────────

    /// `current_ref()` with `session_ref_id = None` returns the default ref.
    #[tokio::test]
    async fn current_ref_with_no_session_returns_default_ref() {
        let server = test_server();
        assert!(
            server.session_ref_id.is_none(),
            "precondition: freshly constructed server has no session ref"
        );

        let default_ref = server
            .state
            .default_ref()
            .expect("default ref always present");
        let current = server.current_ref().await;

        assert!(
            Arc::ptr_eq(&default_ref, &current),
            "current_ref() with None session must return the same Arc as default_ref()"
        );
    }

    /// `current_ref()` with a registered `session_ref_id` returns that ref.
    #[tokio::test]
    async fn current_ref_with_registered_session_returns_session_ref() {
        use crate::ref_index::{RefId, RefIndex};

        let server = test_server();

        // Register a synthetic worktree ref.
        let wt_path = std::path::PathBuf::from("/tmp/u11-current-ref-wt");
        let wt_ref_id = RefId::for_canonical_path(&wt_path);
        let wt_ref_arc = server
            .state
            .attach_ref(wt_ref_id, || {
                Arc::new(RefIndex::new(
                    wt_path.clone(),
                    wt_path.clone(),
                    Some(server.state.default_ref_id),
                ))
            })
            .await;

        // Build a session-scoped server clone and check current_ref().
        let session_server = server.with_session(wt_ref_id);
        let current = session_server.current_ref().await;

        assert!(
            Arc::ptr_eq(&wt_ref_arc, &current),
            "current_ref() for a registered session must return the registered RefIndex"
        );
    }

    /// `current_ref()` with an unregistered `session_ref_id` falls back to `default_ref`.
    #[tokio::test]
    async fn current_ref_with_unknown_session_falls_back_to_default_ref() {
        use crate::ref_index::RefId;

        let server = test_server();

        // Use a RefId that is NOT registered in the registry.
        let unknown_id = RefId(0xdeadbeef_12345678);
        let session_server = server.with_session(unknown_id);

        let default_ref = server
            .state
            .default_ref()
            .expect("default ref always present");
        let current = session_server.current_ref().await;

        assert!(
            Arc::ptr_eq(&default_ref, &current),
            "current_ref() with unknown session_ref_id must fall back to default_ref"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn default_ref_does_not_treat_registry_write_contention_as_absence() {
        let server = test_server();
        let state = Arc::clone(&server.state);
        let lookup_state = Arc::clone(&state);
        let expected_id = state.default_ref_id;
        let writer = state.refs.write().await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let lookup = tokio::task::spawn_blocking(move || {
            let _ = started_tx.send(());
            lookup_state.default_ref()
        });
        started_rx.await.unwrap();
        tokio::task::yield_now().await;
        drop(writer);

        let found = tokio::time::timeout(std::time::Duration::from_secs(1), lookup)
            .await
            .expect("default_ref did not complete after the registry writer released")
            .expect("default_ref panicked while the registry writer was held")
            .expect("default_ref treated registry contention as absence");
        assert_eq!(
            crate::ref_index::RefId::for_canonical_path(&found.canonical_root),
            expected_id
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn current_ref_does_not_panic_or_fall_back_during_registry_write_contention() {
        use crate::ref_index::{RefId, RefIndex};

        let server = test_server();
        let worktree = tempfile::tempdir().unwrap();
        let canonical = worktree.path().canonicalize().unwrap();
        let ref_id = RefId::for_canonical_path(&canonical);
        let expected = server
            .state
            .attach_ref(ref_id, || {
                Arc::new(RefIndex::new(
                    canonical.clone(),
                    canonical.clone(),
                    Some(server.state.default_ref_id),
                ))
            })
            .await;
        let session = server.with_session(ref_id);
        let writer = server.state.refs.write().await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let lookup = tokio::spawn(async move {
            let _ = started_tx.send(());
            session.current_ref().await
        });
        started_rx.await.unwrap();
        tokio::task::yield_now().await;
        drop(writer);

        let found = tokio::time::timeout(std::time::Duration::from_secs(1), lookup)
            .await
            .expect("current_ref did not complete after the registry writer released")
            .expect("current_ref panicked while the registry writer was held");
        assert!(
            Arc::ptr_eq(&found, &expected),
            "current_ref fell back to the default ref during registry contention"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn current_ref_waits_for_registry_writer_on_current_thread_runtime() {
        use crate::ref_index::{RefId, RefIndex};

        let server = test_server();
        let worktree = tempfile::tempdir().unwrap();
        let canonical = worktree.path().canonicalize().unwrap();
        let ref_id = RefId::for_canonical_path(&canonical);
        let expected = server
            .state
            .attach_ref(ref_id, || {
                Arc::new(RefIndex::new(
                    canonical.clone(),
                    canonical.clone(),
                    Some(server.state.default_ref_id),
                ))
            })
            .await;
        let session = server.with_session(ref_id);
        let writer = server.state.refs.write().await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let lookup = tokio::spawn(async move {
            let _ = started_tx.send(());
            session.current_ref().await
        });
        started_rx.await.unwrap();
        tokio::task::yield_now().await;
        drop(writer);

        let found = tokio::time::timeout(std::time::Duration::from_secs(1), lookup)
            .await
            .expect("current_ref did not complete after the registry writer released")
            .expect("current_ref panicked while the registry writer was held");
        assert!(
            Arc::ptr_eq(&found, &expected),
            "current_ref fell back to the default ref during registry contention"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_attach_and_tool_calls_do_not_panic_on_registry_contention() {
        use crate::ref_index::{RefId, RefIndex};

        const CALLS: usize = 32;
        let server = test_server();
        let writer = server.state.refs.write().await;

        let mut attaches = Vec::with_capacity(CALLS);
        let mut tools = Vec::with_capacity(CALLS);
        for i in 0..CALLS {
            let attaching = server.clone();
            attaches.push(tokio::spawn(async move {
                let root = std::env::temp_dir().join(format!("contextplus-contention-{i}"));
                let ref_id = RefId::for_canonical_path(&root);
                attaching
                    .state
                    .attach_ref(ref_id, || {
                        Arc::new(RefIndex::new(
                            root.clone(),
                            root,
                            Some(attaching.state.default_ref_id),
                        ))
                    })
                    .await;
            }));

            let calling = server.clone();
            tools.push(tokio::spawn(async move {
                let mut args = serde_json::Map::new();
                args.insert("path".into(), serde_json::Value::String(".".into()));
                calling.dispatch("outline", args).await
            }));
        }

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        drop(writer);

        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            for attach in attaches {
                attach.await.expect("attach task panicked");
            }
            for tool in tools {
                tool.await
                    .expect("tool call panicked during concurrent registry writes");
            }
        })
        .await
        .expect("concurrent attach/tool race exceeded its bounded budget");
    }

    /// Cross-ref isolation: writing to one ref's `embedding_cache` does NOT
    /// appear in a second ref's cache. This proves Arc-clone isolation per-ref.
    #[tokio::test]
    async fn cross_ref_embedding_cache_isolation() {
        use crate::core::embeddings::CacheEntry;
        use crate::ref_index::{RefId, RefIndex};

        let server = test_server();

        // Attach ref-A and ref-B as independent synthetic worktree refs.
        let path_a = std::path::PathBuf::from("/tmp/u11-isolation-ref-a");
        let path_b = std::path::PathBuf::from("/tmp/u11-isolation-ref-b");
        let id_a = RefId::for_canonical_path(&path_a);
        let id_b = RefId::for_canonical_path(&path_b);

        let ref_a = server
            .state
            .attach_ref(id_a, || {
                Arc::new(RefIndex::new(
                    path_a.clone(),
                    path_a.clone(),
                    Some(server.state.default_ref_id),
                ))
            })
            .await;

        let ref_b = server
            .state
            .attach_ref(id_b, || {
                Arc::new(RefIndex::new(
                    path_b.clone(),
                    path_b.clone(),
                    Some(server.state.default_ref_id),
                ))
            })
            .await;

        // Write a unique key into ref-A's embedding cache.
        ref_a.embedding_cache.write().await.insert(
            "ref_a_only.rs".to_string(),
            CacheEntry {
                hash: "hash_a".to_string(),
                vector: vec![1.0, 2.0, 3.0],
            },
        );

        // ref-B's cache must NOT contain ref-A's key.
        assert!(
            !ref_b
                .embedding_cache
                .read()
                .await
                .contains_key("ref_a_only.rs"),
            "ref-B's embedding_cache must not contain a key written to ref-A"
        );

        // default ref must also not contain it.
        assert!(
            !server
                .state
                .default_ref()
                .unwrap()
                .embedding_cache
                .read()
                .await
                .contains_key("ref_a_only.rs"),
            "default ref's embedding_cache must not contain a key written to ref-A"
        );
    }

    // ── U16: ollama_semaphore on SharedState ─────────────────────────────────

    /// `ollama_semaphore()` returns an `Arc<Semaphore>` whose permit count
    /// matches the configured `ollama_max_concurrent`.
    #[test]
    fn ollama_semaphore_permits_match_config() {
        let config = Config::from_env();
        let capacity = config.ollama_max_concurrent;
        let root = std::env::temp_dir().join("contextplus-u16-sem-test");
        let _ = std::fs::create_dir_all(&root);
        let server = ContextPlusServer::new(root, config);
        let sem = server.state.ollama_semaphore();
        assert_eq!(
            sem.available_permits(),
            capacity,
            "semaphore available_permits must equal configured ollama_max_concurrent"
        );
    }

    /// `ollama_semaphore()` returns a clone of the same underlying `Arc` —
    /// acquiring through one clone reduces permits visible through another.
    #[tokio::test]
    async fn ollama_semaphore_clones_share_state() {
        let root = std::env::temp_dir().join("contextplus-u16-sem-clone-test");
        let _ = std::fs::create_dir_all(&root);
        let server = ContextPlusServer::new(root, Config::from_env());

        let sem_a = server.state.ollama_semaphore();
        let sem_b = server.state.ollama_semaphore();

        // Both Arcs point at the same Semaphore.
        assert!(
            Arc::ptr_eq(&sem_a, &sem_b),
            "both Arcs must point to the same Semaphore"
        );
    }

    /// Integration: construct with `ollama_max_concurrent = 2`, acquire 2 permits,
    /// third `try_acquire` must return `Err`. This is the contract U17 relies on.
    #[tokio::test]
    async fn ollama_semaphore_blocks_at_capacity() {
        use crate::config::RefWarmupMode;

        let root = std::env::temp_dir().join("contextplus-u16-sem-block-test");
        let _ = std::fs::create_dir_all(&root);

        // Build a config with capacity=2 by overriding the env var.
        let config = {
            // SAFETY: test-only, single-threaded at this point in the test.
            unsafe { std::env::set_var("CONTEXTPLUS_OLLAMA_MAX_CONCURRENT", "2") };
            let c = Config::from_env();
            unsafe { std::env::remove_var("CONTEXTPLUS_OLLAMA_MAX_CONCURRENT") };
            c
        };

        assert_eq!(config.ollama_max_concurrent, 2);

        let server = ContextPlusServer::new(root, config);
        let sem = server.state.ollama_semaphore();

        // Acquire both permits.
        let _permit1 = sem.acquire().await.expect("first acquire must succeed");
        let _permit2 = sem.acquire().await.expect("second acquire must succeed");

        // Third try_acquire must fail (no permits left).
        assert!(
            sem.try_acquire().is_err(),
            "try_acquire on a fully-occupied semaphore must return Err"
        );

        // Dropping one permit frees a slot.
        drop(_permit1);
        assert!(
            sem.try_acquire().is_ok(),
            "try_acquire must succeed after a permit is released"
        );

        // Verify RefWarmupMode is accessible from the config stored in state.
        let _ = server.state.config.ref_warmup_mode;
        assert_eq!(server.state.config.ref_warmup_mode, RefWarmupMode::Shallow);
    }

    // -----------------------------------------------------------------------
    // U18: per-ref warmup tests
    // -----------------------------------------------------------------------

    fn test_server_with_warmup_mode(mode: crate::config::RefWarmupMode) -> ContextPlusServer {
        let mut config = Config::from_env();
        config.ref_warmup_mode = mode;
        let root = std::env::temp_dir().join(format!(
            "contextplus-u18-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        let _ = std::fs::create_dir_all(&root);
        ContextPlusServer::new(root, config)
    }

    /// mode=off: spawn_ref_warmup is a no-op; nothing in `warmup_in_flight`.
    #[tokio::test]
    async fn ref_warmup_mode_off_is_noop() {
        use crate::config::RefWarmupMode;
        let server = test_server_with_warmup_mode(RefWarmupMode::Off);
        let ref_id = server.state.default_ref_id;

        server.spawn_ref_warmup(ref_id);

        // Give a brief window for any (unexpected) spawned task to run.
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // warmup_in_flight should be empty — Off mode never inserts.
        let inflight = server.state.warmup_in_flight.lock().await;
        assert!(
            inflight.is_empty(),
            "mode=off must not add ref_id to warmup_in_flight"
        );

        // project_cache must still be None (no walk happened).
        let ref_index = server.state.default_ref().unwrap();
        let cache = ref_index.project_cache.read().await;
        assert!(cache.is_none(), "mode=off must not populate project_cache");
    }

    /// mode=shallow: task walks, populates project_cache; no Ollama calls.
    ///
    /// We verify Ollama is never called by pointing the host at a dead port.
    /// If `embed` were called it would try to connect and fail (which we'd see
    /// on the identifier_index). Since shallow mode doesn't touch embed, the
    /// task must succeed regardless.
    /// The ref warmup installs an identifier index with the parsed docs and no
    /// vectors (dims == 0) for the first real call to fill. Serving it from the
    /// fast path scores every identifier at 0% semantic, so it must not count as
    /// built even when file_count and TTL match.
    #[tokio::test]
    async fn identifier_search_embeds_when_index_has_no_vectors() {
        use crate::tools::semantic_identifiers::IdentifierDoc;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|req: &Request| {
                let body: serde_json::Value = req.body_json().unwrap_or_default();
                let n = body["input"].as_array().map_or(1, |a| a.len());
                let vecs: Vec<Vec<f32>> = (0..n).map(|_| vec![0.6, 0.8, 0.0]).collect();
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "embeddings": vecs }))
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            root.path().join("hello.rs"),
            "fn hello() {}\nfn world() {}\n",
        )
        .unwrap();
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);

        let cache = server.ensure_project_cache().await.unwrap();
        let file_count = cache
            .file_entries
            .iter()
            .filter(|e| !e.is_directory)
            .count();
        let doc = IdentifierDoc {
            id: "hello.rs:hello:1".into(),
            path: "hello.rs".into(),
            header: String::new(),
            name: "hello".into(),
            kind: "function".into(),
            kind_lower: "function".into(),
            line: 1,
            end_line: 1,
            signature: "fn hello() {}".into(),
            parent_name: None,
            text: "hello function fn hello() {} hello.rs".into(),
            name_token_set: crate::tools::semantic_identifiers::identifier_terms("hello"),
            signature_token_set: crate::tools::semantic_identifiers::identifier_terms(
                "fn hello() {}",
            ),
            parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
        };
        {
            let ref_index = server.current_ref().await;
            let mut guard = ref_index.identifier_index.write().await;
            *guard = Some(Arc::new(IdentifierIndex {
                docs: vec![doc].into(),
                vectors: IdentifierVectorIndex::empty(),
                dims: 0,
                file_count,
                built_at: Instant::now(),
            }));
        }

        let idx = server.ensure_identifier_index(&cache).await.unwrap();
        assert_eq!(
            idx.dims, 3,
            "a vector-less index must be rebuilt with vectors"
        );
        assert_eq!(idx.vectors.len(), idx.docs.len() * idx.dims);
        assert!(!idx.docs.is_empty());
        assert!(
            ollama
                .received_requests()
                .await
                .is_some_and(|r| !r.is_empty()),
            "Ollama must have been called"
        );
    }

    /// A worktree's identifier texts are keyed by repo-relative path, so the
    /// primary's identifier cache must serve it; otherwise the first identifier
    /// search in every worktree re-embeds the whole repo.
    #[tokio::test]
    async fn worktree_identifier_index_reuses_primary_cache() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|req: &Request| {
                let body: serde_json::Value = req.body_json().unwrap_or_default();
                let n = body["input"].as_array().map_or(1, |a| a.len());
                let vecs: Vec<Vec<f32>> = (0..n).map(|_| vec![0.6, 0.8, 0.0]).collect();
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "embeddings": vecs }))
            })
            .mount(&ollama)
            .await;
        let embed_calls = || async {
            ollama.received_requests().await.map_or(0, |r| {
                r.iter().filter(|q| q.url.path() == "/api/embed").count()
            })
        };

        let primary = tempfile::tempdir().expect("tempdir");
        let source = "fn hello() {}\nfn world() {}\n";
        std::fs::write(primary.path().join("hello.rs"), source).unwrap();
        let mut config = Config::from_env();
        config.ollama_host = ollama.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        let server = ContextPlusServer::new(primary.path().to_path_buf(), config);

        let cache = server.ensure_project_cache().await.unwrap();
        let idx = server.ensure_identifier_index(&cache).await.unwrap();
        assert_eq!(idx.dims, 3);
        let calls_after_primary = embed_calls().await;
        assert!(calls_after_primary >= 1, "primary build must embed");

        // A worktree with the same file at the same relative path.
        let wt = tempfile::tempdir().expect("tempdir");
        std::fs::write(wt.path().join("hello.rs"), source).unwrap();
        let canonical_wt = wt.path().canonicalize().unwrap();
        let mut args = serde_json::Map::new();
        args.insert(
            "path".to_string(),
            json!(canonical_wt.to_string_lossy().to_string()),
        );
        assert_eq!(
            server.handle_attach_worktree(args).await.unwrap().is_error,
            Some(false)
        );
        let wt_ref = crate::ref_index::RefId::for_canonical_path(&canonical_wt);
        let wt_server = server.with_session(wt_ref);

        let wt_cache = wt_server.ensure_project_cache().await.unwrap();
        let wt_idx = wt_server.ensure_identifier_index(&wt_cache).await.unwrap();
        assert_eq!(wt_idx.dims, 3, "worktree index must carry vectors");
        assert_eq!(wt_idx.docs.len(), idx.docs.len());
        assert_eq!(
            embed_calls().await,
            calls_after_primary,
            "worktree must reuse the primary's identifier cache, not re-embed"
        );
    }

    #[tokio::test]
    async fn ref_warmup_shallow_populates_project_cache_no_embed() {
        use crate::config::RefWarmupMode;
        use std::fs;

        // Build a tiny project with two files so walker finds something.
        let root = tempfile::tempdir().expect("tempdir");
        let root_path = root.path().to_path_buf();
        fs::write(root_path.join("hello.rs"), "fn hello() {}").unwrap();
        fs::write(root_path.join("world.txt"), "hello world").unwrap();

        let mut config = Config::from_env();
        // Point Ollama at a dead host — if embed is called it will error.
        config.ollama_host = "http://127.0.0.1:1".to_string();
        config.ref_warmup_mode = RefWarmupMode::Shallow;

        let server = ContextPlusServer::new(root_path.clone(), config);
        let ref_id = server.state.default_ref_id;

        server.spawn_ref_warmup(ref_id);

        // Wait up to 10 s for the background task to populate the cache.
        // 3 s was tight enough on slow CI runners (notably Ubuntu under load)
        // that the walk + tree-sitter parse occasionally missed the window;
        // 10 s gives generous headroom without slowing the happy path.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            {
                let ref_index = server.state.ref_index(ref_id).await.unwrap();
                let cache = ref_index.project_cache.read().await;
                if cache.is_some() {
                    break;
                }
            }
            if std::time::Instant::now() > deadline {
                panic!("ref_warmup shallow: project_cache never populated within 10s");
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        // Verify project_cache fields.
        {
            let ref_index = server.state.ref_index(ref_id).await.unwrap();
            let cache_guard = ref_index.project_cache.read().await;
            let cache = cache_guard.as_ref().unwrap();
            assert!(
                !cache.file_entries.is_empty(),
                "shallow warmup must produce non-empty file_entries"
            );
            assert!(
                cache.file_content.contains_key("hello.rs"),
                "shallow warmup must read hello.rs into file_content"
            );
        }

        // Wait for identifier_index to be set (parse runs after project_cache write).
        // CI runners are slower than local dev; 10s is the conservative ceiling.
        let deadline2 = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            {
                let ref_index = server.state.ref_index(ref_id).await.unwrap();
                let guard = ref_index.identifier_index.read().await;
                if guard.is_some() {
                    break;
                }
            }
            if std::time::Instant::now() > deadline2 {
                panic!("ref_warmup shallow: identifier_index never populated within 10s");
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        // Verify identifier_index.docs was populated (tree-sitter ran).
        {
            let ref_index = server.state.ref_index(ref_id).await.unwrap();
            let idx_guard = ref_index.identifier_index.read().await;
            let idx = idx_guard.as_ref().expect("identifier_index should be set");
            // hello.rs has one function; docs should be non-empty.
            assert!(
                !idx.docs.is_empty(),
                "shallow warmup must parse symbols into identifier_index.docs"
            );
            // dims == 0 confirms no embed call was made.
            assert_eq!(
                idx.dims, 0,
                "shallow warmup must NOT populate embedding dims (no Ollama call)"
            );
            assert!(
                idx.vectors.is_empty(),
                "shallow warmup must NOT populate vectors (no Ollama call)"
            );
        }
    }

    /// Idempotency: calling spawn_ref_warmup 5 times concurrently for the same
    /// ref_id must result in at most one warmup run (first-writer-wins).
    #[tokio::test]
    async fn ref_warmup_idempotent_concurrent_calls() {
        use crate::config::RefWarmupMode;
        use std::fs;

        let root = tempfile::tempdir().expect("tempdir");
        let root_path = root.path().to_path_buf();
        fs::write(root_path.join("a.rs"), "pub fn a() {}").unwrap();

        let mut config = Config::from_env();
        config.ollama_host = "http://127.0.0.1:1".to_string();
        config.ref_warmup_mode = RefWarmupMode::Shallow;

        let server = ContextPlusServer::new(root_path.clone(), config);
        let ref_id = server.state.default_ref_id;

        // Fire 5 concurrent spawn_ref_warmup calls.
        for _ in 0..5 {
            server.spawn_ref_warmup(ref_id);
        }

        // Wait for warmup to settle.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            {
                let ref_index = server.state.ref_index(ref_id).await.unwrap();
                let cache = ref_index.project_cache.read().await;
                if cache.is_some() {
                    break;
                }
            }
            if std::time::Instant::now() > deadline {
                panic!("ref_warmup idempotency: cache never populated within 5s");
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        // After completion, warmup_in_flight should be empty (guard released).
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        let inflight = server.state.warmup_in_flight.lock().await;
        assert!(
            inflight.is_empty(),
            "warmup_in_flight must be empty after all tasks complete"
        );
    }

    /// Per-ref isolation: warmup for ref-A must not touch ref-B's project_cache.
    #[tokio::test]
    async fn ref_warmup_per_ref_isolation() {
        use crate::config::RefWarmupMode;
        use crate::ref_index::{RefId, RefIndex};
        use std::fs;
        use std::sync::atomic::Ordering;

        // Build two separate roots.
        let root_a = tempfile::tempdir().expect("tempdir_a");
        let root_b = tempfile::tempdir().expect("tempdir_b");

        fs::write(root_a.path().join("a.rs"), "pub fn only_a() {}").unwrap();
        fs::write(root_b.path().join("b.rs"), "pub fn only_b() {}").unwrap();

        let canonical_b = root_b.path().canonicalize().unwrap();

        let mut config = Config::from_env();
        config.ollama_host = "http://127.0.0.1:1".to_string();
        config.ref_warmup_mode = RefWarmupMode::Shallow;

        // Build server rooted at root_a (default ref = ref_a).
        let server = ContextPlusServer::new(root_a.path().to_path_buf(), config.clone());
        let ref_id_a = server.state.default_ref_id;

        // Register ref_b manually.
        let ref_id_b = RefId::for_canonical_path(&canonical_b);
        server
            .state
            .attach_ref(ref_id_b, || {
                Arc::new(RefIndex::new(
                    root_b.path().to_path_buf(),
                    canonical_b.clone(),
                    None,
                ))
            })
            .await;
        // Undo the session-count side-effect from attach_ref.
        if let Some(r) = server.state.ref_index(ref_id_b).await {
            r.session_count.fetch_sub(1, Ordering::AcqRel);
        }

        // Warm ref_a only.
        server.spawn_ref_warmup(ref_id_a);

        // Wait for ref_a's cache.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            {
                let ref_a = server.state.ref_index(ref_id_a).await.unwrap();
                let cache = ref_a.project_cache.read().await;
                if cache.is_some() {
                    break;
                }
            }
            if std::time::Instant::now() > deadline {
                panic!("ref_warmup isolation: ref_a cache never populated");
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        // ref_b's project_cache must still be None.
        let ref_b = server.state.ref_index(ref_id_b).await.unwrap();
        let cache_b = ref_b.project_cache.read().await;
        assert!(
            cache_b.is_none(),
            "warming ref_a must not touch ref_b's project_cache"
        );

        // ref_a's cache must contain only its own files.
        let ref_a = server.state.ref_index(ref_id_a).await.unwrap();
        let cache_a = ref_a.project_cache.read().await;
        let cache_a = cache_a.as_ref().unwrap();
        assert!(
            cache_a.file_content.contains_key("a.rs"),
            "ref_a cache must contain a.rs"
        );
        assert!(
            !cache_a.file_content.contains_key("b.rs"),
            "ref_a cache must NOT contain b.rs"
        );
    }

    /// Config: default ref_warmup_mode is Shallow.
    #[test]
    fn ref_warmup_mode_defaults_to_shallow() {
        use crate::config::RefWarmupMode;
        // Ensure env var is absent.
        let _lock = WARMUP_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::remove_var("CONTEXTPLUS_REF_WARMUP_MODE");
        }
        let cfg = Config::from_env();
        assert_eq!(cfg.ref_warmup_mode, RefWarmupMode::Shallow);
    }

    /// Config: CONTEXTPLUS_REF_WARMUP_MODE=off disables warmup.
    #[test]
    fn ref_warmup_mode_can_be_disabled() {
        use crate::config::RefWarmupMode;
        let _lock = WARMUP_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("CONTEXTPLUS_REF_WARMUP_MODE", "off");
        }
        let cfg = Config::from_env();
        assert_eq!(cfg.ref_warmup_mode, RefWarmupMode::Off);
        unsafe {
            std::env::remove_var("CONTEXTPLUS_REF_WARMUP_MODE");
        }
    }

    /// Config: CONTEXTPLUS_REF_WARMUP_MODE=full enables full warmup.
    #[test]
    fn ref_warmup_mode_full() {
        use crate::config::RefWarmupMode;
        let _lock = WARMUP_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("CONTEXTPLUS_REF_WARMUP_MODE", "full");
        }
        let cfg = Config::from_env();
        assert_eq!(cfg.ref_warmup_mode, RefWarmupMode::Full);
        unsafe {
            std::env::remove_var("CONTEXTPLUS_REF_WARMUP_MODE");
        }
    }

    // -----------------------------------------------------------------------
    // U20: baseline-import tests
    // -----------------------------------------------------------------------

    /// Shallow warmup on a worktree ref that has a parent with one CAS blob
    /// inherits that blob into `embedding_cache` and `search_index_cache`.
    /// Zero Ollama calls — the dead-port assertion still holds.
    #[tokio::test]
    async fn ref_warmup_shallow_imports_parent_baseline() {
        use crate::cache::cas::{CasStore, ChunkHash, ChunkKey};
        use crate::config::RefWarmupMode;
        use crate::ref_index::{RefId, RefIndex};
        use std::fs;

        // ── Set up primary root with one Rust file. ──────────────────────────
        let primary_root = tempfile::tempdir().expect("primary_root tempdir");
        let primary_path = primary_root.path().to_path_buf();
        fs::write(primary_path.join("hello.rs"), "fn hello() {}").unwrap();

        // Build a server rooted at primary_path; Ollama pointed at dead port.
        let mut config = Config::from_env();
        config.ollama_host = "http://127.0.0.1:1".to_string();
        config.ref_warmup_mode = RefWarmupMode::Shallow;
        let server = ContextPlusServer::new(primary_path.clone(), config);

        let primary_ref_id = server.state.default_ref_id;
        let primary_ref = server.state.default_ref().unwrap();

        // ── Pre-seed the primary's CAS manifest + blob. ─────────────────────
        // Compute the same embed-text format that import_baseline_for_ref uses.
        let content = "fn hello() {}";
        let embed_text =
            build_embedding_document("hello.rs", content, server.state.config.embed_doc_shape);
        let chunk_hash = ChunkHash::of(&embed_text);
        let chunk_key = ChunkKey::new("hello.rs".to_string(), 0);

        // Write a synthetic vector (768-dim) to the CAS under the primary ref.
        let synthetic_vec: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();
        let mcp_data = primary_path.join(".mcp_data");
        fs::create_dir_all(&mcp_data).unwrap();
        let cas = CasStore::new(
            mcp_data.clone(),
            server.state.config.document_cache_identity(),
        );
        cas.write_blob(&chunk_hash, &synthetic_vec).unwrap();
        cas.update_manifest(
            &primary_ref.cas_ref_id_hex,
            &[(chunk_key.clone(), chunk_hash.clone())],
        )
        .unwrap();

        // ── Build a worktree ref forked from primary. ────────────────────────
        let wt_root = tempfile::tempdir().expect("wt_root tempdir");
        let wt_path = wt_root.path().to_path_buf();
        fs::write(wt_path.join("hello.rs"), "fn hello() {}").unwrap();
        let canonical_wt = wt_path.canonicalize().unwrap();
        let wt_ref_id = RefId::for_canonical_path(&canonical_wt);
        let wt_ref = server
            .state
            .attach_ref(wt_ref_id, || {
                Arc::new(RefIndex::new(
                    wt_path.clone(),
                    canonical_wt.clone(),
                    Some(primary_ref_id),
                ))
            })
            .await;

        // Write the parent pointer on disk so CAS lookup chains to primary.
        cas.write_parent(&wt_ref.cas_ref_id_hex, &primary_ref.cas_ref_id_hex)
            .unwrap();

        // ── Trigger shallow warmup on the worktree ref. ──────────────────────
        server.spawn_ref_warmup(wt_ref_id);

        // Wait up to 5 s for the embedding_cache to be populated.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            {
                let cache = wt_ref.embedding_cache.read().await;
                if cache.contains_key("hello.rs") {
                    break;
                }
            }
            if std::time::Instant::now() > deadline {
                panic!(
                    "ref_warmup_shallow_imports_parent_baseline: embedding_cache never populated"
                );
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        // ── Assertions. ──────────────────────────────────────────────────────
        {
            let cache = wt_ref.embedding_cache.read().await;
            let entry = cache
                .get("hello.rs")
                .expect("hello.rs must be in embedding_cache");
            assert_eq!(
                entry.vector.len(),
                768,
                "inherited vector must have the correct dimension"
            );
            // Spot-check the first element against our synthetic vector.
            assert!(
                (entry.vector[0] - 0.0_f32).abs() < 1e-5,
                "vector[0] must match the seeded synthetic vector"
            );
        }
        // search_index_cache must be populated (HNSW built from inherited blobs).
        {
            let idx = wt_ref.search_index_cache.read().await;
            assert!(
                idx.is_some(),
                "search_index_cache must be populated after baseline import"
            );
        }
        // identifier_index must have docs (tree-sitter ran).
        {
            let guard = wt_ref.identifier_index.read().await;
            let idx = guard.as_ref().expect("identifier_index must be set");
            assert!(!idx.docs.is_empty(), "tree-sitter must have parsed symbols");
        }
    }

    /// Full warmup imports the baseline (zero Ollama so far), then embeds diff
    /// chunks. With a dead Ollama host the diff-embed phase fails gracefully,
    /// but baseline hits must already be in `embedding_cache`.
    ///
    /// This test asserts the "extends Shallow" invariant: if the worktree has
    /// an identical file to the parent, that file appears in `embedding_cache`
    /// even when the Ollama host is unreachable.
    #[tokio::test]
    async fn ref_warmup_full_layers_ollama_on_baseline() {
        use crate::cache::cas::{CasStore, ChunkHash, ChunkKey};
        use crate::config::RefWarmupMode;
        use crate::ref_index::{RefId, RefIndex};
        use std::fs;

        // ── Primary root with one file whose CAS blob is pre-seeded. ─────────
        let primary_root = tempfile::tempdir().expect("primary_root tempdir");
        let primary_path = primary_root.path().to_path_buf();
        fs::write(primary_path.join("shared.rs"), "fn shared() {}").unwrap();

        let mut config = Config::from_env();
        // Dead Ollama host — baseline import succeeds (no Ollama), diff embed fails
        // gracefully.
        config.ollama_host = "http://127.0.0.1:1".to_string();
        config.ref_warmup_mode = RefWarmupMode::Full;
        let server = ContextPlusServer::new(primary_path.clone(), config);

        let primary_ref_id = server.state.default_ref_id;
        let primary_ref = server.state.default_ref().unwrap();

        // Pre-seed the CAS for the shared file.
        let content = "fn shared() {}";
        let embed_text =
            build_embedding_document("shared.rs", content, server.state.config.embed_doc_shape);
        let chunk_hash = ChunkHash::of(&embed_text);
        let chunk_key = ChunkKey::new("shared.rs".to_string(), 0);

        let seeded_vec: Vec<f32> = (0..768).map(|i| (i as f32) * 0.002).collect();
        let mcp_data = primary_path.join(".mcp_data");
        fs::create_dir_all(&mcp_data).unwrap();
        let cas = CasStore::new(
            mcp_data.clone(),
            server.state.config.document_cache_identity(),
        );
        cas.write_blob(&chunk_hash, &seeded_vec).unwrap();
        cas.update_manifest(&primary_ref.cas_ref_id_hex, &[(chunk_key, chunk_hash)])
            .unwrap();

        // ── Worktree ref: same shared file + a new diff file. ────────────────
        let wt_root = tempfile::tempdir().expect("wt_root tempdir");
        let wt_path = wt_root.path().to_path_buf();
        fs::write(wt_path.join("shared.rs"), "fn shared() {}").unwrap();
        fs::write(wt_path.join("new_in_wt.rs"), "fn new_fn() {}").unwrap();
        let canonical_wt = wt_path.canonicalize().unwrap();
        let wt_ref_id = RefId::for_canonical_path(&canonical_wt);
        let wt_ref = server
            .state
            .attach_ref(wt_ref_id, || {
                Arc::new(RefIndex::new(
                    wt_path.clone(),
                    canonical_wt.clone(),
                    Some(primary_ref_id),
                ))
            })
            .await;

        // Set parent pointer on disk.
        cas.write_parent(&wt_ref.cas_ref_id_hex, &primary_ref.cas_ref_id_hex)
            .unwrap();

        // ── Trigger full warmup. ──────────────────────────────────────────────
        server.spawn_ref_warmup(wt_ref_id);

        // Wait up to 5 s for project_cache to be populated (baseline import
        // phase runs after walk, before Ollama).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            {
                let cache = wt_ref.project_cache.read().await;
                if cache.is_some() {
                    break;
                }
            }
            if std::time::Instant::now() > deadline {
                panic!("ref_warmup_full_layers_ollama_on_baseline: project_cache never populated");
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        // Wait for the baseline import to install the search index, which it
        // does after caching the hit (shared.rs).
        let deadline2 = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if wt_ref.search_index_cache.read().await.is_some() {
                break;
            }
            if std::time::Instant::now() > deadline2 {
                panic!("ref_warmup_full_layers_ollama_on_baseline: search index never installed");
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        // ── Assertions. ──────────────────────────────────────────────────────
        // shared.rs must be in embedding_cache (inherited from parent — no Ollama).
        {
            let cache = wt_ref.embedding_cache.read().await;
            let entry = cache
                .get("shared.rs")
                .expect("shared.rs must be in embedding_cache from baseline import");
            assert_eq!(entry.vector.len(), 768, "inherited vector must be 768-dim");
        }
        // new_in_wt.rs is a diff file; with dead Ollama it won't be embedded, but
        // it must appear in project_cache (walk happened).
        {
            let guard = wt_ref.project_cache.read().await;
            let pc = guard.as_ref().unwrap();
            assert!(
                pc.file_content.contains_key("new_in_wt.rs"),
                "new_in_wt.rs must be in project_cache after walk"
            );
        }
        // search_index_cache must be populated from the baseline hits.
        {
            let guard = wt_ref.search_index_cache.read().await;
            assert!(
                guard.is_some(),
                "search_index_cache must be populated from baseline import"
            );
        }
    }

    fn semantic_fill_config(ollama_uri: &str, budget_ms: u64, fill_timeout_ms: u64) -> Config {
        let mut config = Config::from_env();
        config.ollama_host = ollama_uri.to_string();
        config.ollama_embed_model = "semantic-fill-test".to_string();
        config.embed_query_prefix.clear();
        config.embed_doc_prefix.clear();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        config.embed_batch_size = 32;
        config.ollama_max_concurrent = 1;
        config.embed_budget_ms = budget_ms;
        config.embed_fill_batch_timeout_ms = fill_timeout_ms;
        config
    }

    fn semantic_args(query: &str) -> serde_json::Map<String, serde_json::Value> {
        let mut args = serde_json::Map::new();
        args.insert("query".into(), json!(query));
        args.insert("top_k".into(), json!(5));
        args.insert("semantic_weight".into(), json!(1.0));
        args.insert("keyword_weight".into(), json!(0.0));
        args.insert("require_semantic_match".into(), json!(true));
        args
    }

    fn embed_request_inputs(request: &wiremock::Request) -> Vec<String> {
        request
            .body_json::<serde_json::Value>()
            .ok()
            .and_then(|body| body["input"].as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|input| input.as_str().map(str::to_string))
            .collect()
    }

    async fn matching_embed_request_batches(
        ollama: &wiremock::MockServer,
        marker: &str,
    ) -> Vec<usize> {
        ollama
            .received_requests()
            .await
            .unwrap_or_default()
            .iter()
            .map(embed_request_inputs)
            .filter(|inputs| inputs.iter().any(|input| input.contains(marker)))
            .map(|inputs| inputs.len())
            .collect()
    }

    async fn matching_embed_input_count(ollama: &wiremock::MockServer, marker: &str) -> usize {
        ollama
            .received_requests()
            .await
            .unwrap_or_default()
            .iter()
            .flat_map(embed_request_inputs)
            .filter(|input| input.contains(marker))
            .count()
    }

    fn run_git(root: &std::path::Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .env("GIT_AUTHOR_NAME", "ContextPlus Test")
            .env("GIT_AUTHOR_EMAIL", "contextplus@example.com")
            .env("GIT_COMMITTER_NAME", "ContextPlus Test")
            .env("GIT_COMMITTER_EMAIL", "contextplus@example.com")
            .output()
            .unwrap_or_else(|error| panic!("git {args:?} failed to start: {error}"));
        assert!(
            output.status.success(),
            "git {args:?} failed in {}: {}",
            root.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn add_linked_worktree(primary: &std::path::Path, worktree: &std::path::Path) {
        run_git(
            primary,
            &[
                "worktree",
                "add",
                worktree.to_str().unwrap(),
                "-b",
                "semantic-worktree-test",
            ],
        );
    }

    #[tokio::test]
    async fn attached_worktree_reuses_primary_vectors_and_embeds_only_changed_files() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let vectors: Vec<Vec<f32>> = embed_request_inputs(request)
                    .iter()
                    .map(|input| {
                        if input == "invoice payment status"
                            || input.contains("SHARED_STRIPE_RECONCILER")
                        {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        }
                    })
                    .collect();
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embeddings": vectors }))
            })
            .mount(&ollama)
            .await;

        let temp = tempfile::tempdir().unwrap();
        let primary = temp.path().join("primary");
        let worktree = temp.path().join("worktree");
        std::fs::create_dir_all(primary.join("src")).unwrap();
        run_git(&primary, &["init", "-b", "main"]);
        std::fs::write(
            primary.join("src/stripe_webhook.rs"),
            "fn SHARED_STRIPE_RECONCILER() { /* invoice payment status */ }\n",
        )
        .unwrap();
        std::fs::write(
            primary.join("src/changed.rs"),
            "fn PRIMARY_CHANGED_VERSION() {}\n",
        )
        .unwrap();
        run_git(&primary, &["add", "."]);
        run_git(&primary, &["commit", "-m", "baseline"]);

        let server = ContextPlusServer::new(
            primary.clone(),
            semantic_fill_config(&ollama.uri(), 1_000, 1_000),
        );
        let primary_result = server
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();
        assert!(
            text_of(&primary_result).contains("1. src/stripe_webhook.rs"),
            "primary ranking must establish the reusable vector: {}",
            text_of(&primary_result)
        );
        assert_eq!(
            server
                .current_ref()
                .await
                .embedding_cache
                .read()
                .await
                .len(),
            2
        );

        add_linked_worktree(&primary, &worktree);
        std::fs::write(
            worktree.join("src/changed.rs"),
            "fn WORKTREE_CHANGED_VERSION() {}\n",
        )
        .unwrap();
        run_git(&worktree, &["add", "."]);
        run_git(&worktree, &["commit", "-m", "change one file"]);

        let canonical_worktree = worktree.canonicalize().unwrap();
        let mut attach_args = serde_json::Map::new();
        attach_args.insert(
            "path".into(),
            json!(canonical_worktree.to_string_lossy().into_owned()),
        );
        let attached = server.handle_attach_worktree(attach_args).await.unwrap();
        assert_eq!(attached.is_error, Some(false), "{}", text_of(&attached));
        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_worktree);
        let worktree_ref = server.state.ref_index(ref_id).await.unwrap();
        assert!(
            worktree_ref.embedding_cache.read().await.is_empty(),
            "attach must begin with an empty per-ref cache in this regression setup"
        );
        let worktree_server = server.with_session(ref_id);

        let worktree_result = worktree_server
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();
        assert!(
            text_of(&worktree_result).contains("1. src/stripe_webhook.rs"),
            "the worktree must preserve the primary ranking for identical files: {}",
            text_of(&worktree_result)
        );
        assert_eq!(
            matching_embed_input_count(&ollama, "WORKTREE_CHANGED_VERSION").await,
            1,
            "the changed worktree file must be embedded exactly once"
        );

        let document_requests_before_warm_query = ollama
            .received_requests()
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|request| embed_request_inputs(&request))
            .filter(|inputs| inputs.iter().any(|input| input.contains("_VERSION")))
            .count();
        let warm_result = worktree_server
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();
        assert!(text_of(&warm_result).contains("1. src/stripe_webhook.rs"));
        let document_requests_after_warm_query = ollama
            .received_requests()
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|request| embed_request_inputs(&request))
            .filter(|inputs| inputs.iter().any(|input| input.contains("_VERSION")))
            .count();
        assert_eq!(
            document_requests_after_warm_query, document_requests_before_warm_query,
            "a warm worktree query must not re-embed documents"
        );
        assert_eq!(
            matching_embed_input_count(&ollama, "src/stripe_webhook.rs").await,
            1,
            "the identical file must reuse the primary vector instead of being re-embedded"
        );
    }

    /// A server started over a primary and a linked worktree whose changed
    /// file has a vector persisted by an earlier daemon, and the worktree's ref.
    async fn restarted_worktree_with_persisted_vector() -> (
        wiremock::MockServer,
        tempfile::TempDir,
        ContextPlusServer,
        crate::ref_index::RefId,
    ) {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let vectors: Vec<Vec<f32>> = embed_request_inputs(request)
                    .iter()
                    .map(|input| {
                        if input == "invoice payment status" {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        }
                    })
                    .collect();
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embeddings": vectors }))
            })
            .mount(&ollama)
            .await;

        let temp = tempfile::tempdir().unwrap();
        let primary = temp.path().join("primary");
        let worktree = temp.path().join("worktree");
        std::fs::create_dir_all(primary.join("src")).unwrap();
        run_git(&primary, &["init", "-b", "main"]);
        std::fs::write(primary.join("src/shared.rs"), "fn shared_helper() {}\n").unwrap();
        std::fs::write(
            primary.join("src/changed.rs"),
            "fn PRIMARY_CHANGED_VERSION() {}\n",
        )
        .unwrap();
        run_git(&primary, &["add", "."]);
        run_git(&primary, &["commit", "-m", "baseline"]);
        add_linked_worktree(&primary, &worktree);
        let changed = "fn WORKTREE_CHANGED_VERSION() { /* invoice payment status */ }\n";
        std::fs::write(worktree.join("src/changed.rs"), changed).unwrap();

        // What an earlier daemon's background fill persisted for this worktree.
        let config = semantic_fill_config(&ollama.uri(), 1_000, 1_000);
        // An older binary also persisted the vector the worktree inherited.
        let persisted = HashMap::from([
            (
                "src/changed.rs".to_string(),
                CacheEntry {
                    hash: crate::core::embeddings::content_hash(changed),
                    vector: vec![1.0, 0.0],
                },
            ),
            (
                "src/shared.rs".to_string(),
                CacheEntry {
                    hash: crate::core::embeddings::content_hash("fn shared_helper() {}\n"),
                    vector: vec![0.6, 0.8],
                },
            ),
        ]);
        rkyv_store::save_vector_store_merged(
            &worktree,
            &cache_name("embeddings", &config),
            &crate::core::embeddings::VectorStore::from_cache(&persisted).unwrap(),
        )
        .unwrap();

        let server = ContextPlusServer::new(primary.clone(), config);
        let canonical_worktree = worktree.canonicalize().unwrap();
        let mut attach_args = serde_json::Map::new();
        attach_args.insert(
            "path".into(),
            json!(canonical_worktree.to_string_lossy().into_owned()),
        );
        let attached = server.handle_attach_worktree(attach_args).await.unwrap();
        assert_eq!(attached.is_error, Some(false), "{}", text_of(&attached));
        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_worktree);
        (ollama, temp, server, ref_id)
    }

    #[tokio::test]
    async fn restarted_worktree_reuses_its_persisted_vectors_instead_of_reembedding() {
        let (ollama, _temp, server, ref_id) = restarted_worktree_with_persisted_vector().await;

        let result = server
            .with_session(ref_id)
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();
        assert_eq!(
            matching_embed_input_count(&ollama, "WORKTREE_CHANGED_VERSION").await,
            0,
            "a restarted worktree re-embedded a file whose vector it had persisted"
        );
        assert!(
            text_of(&result).contains("1. src/changed.rs"),
            "the persisted vector must rank the file: {}",
            text_of(&result)
        );
    }

    #[tokio::test]
    async fn worktree_walk_reuses_primary_documents_of_identical_files() {
        let (_ollama, _temp, server, ref_id) = restarted_worktree_with_persisted_vector().await;
        server
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();

        let (logs, _guard) = crate::test_logs::captured_info_logs();
        server
            .with_session(ref_id)
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();
        let logs = crate::test_logs::logs_as_string(&logs);
        let walk = logs
            .lines()
            .find(|line| line.contains("phase=\"semantic_walk\""))
            .unwrap_or_else(|| panic!("no semantic walk logged:\n{logs}"));
        assert!(
            walk.contains("documents=2 reused=1"),
            "the worktree parsed a file the primary had already parsed: {walk}"
        );
    }

    #[tokio::test]
    async fn restarted_worktree_reloads_only_vectors_its_parent_lacks() {
        let (_ollama, _temp, server, ref_id) = restarted_worktree_with_persisted_vector().await;
        server
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();

        server
            .with_session(ref_id)
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();

        let worktree_ref = server.state.ref_index(ref_id).await.unwrap();
        let cache = worktree_ref.embedding_cache.read().await;
        assert_eq!(
            cache.get("src/shared.rs").map(|entry| entry.vector.clone()),
            Some(vec![0.0, 1.0]),
            "the worktree reloaded its own copy of a vector its parent holds"
        );
        assert_eq!(cache.len(), 2);
    }

    #[tokio::test]
    async fn worktree_fill_persists_only_vectors_its_parent_lacks() {
        let (_ollama, temp, server, ref_id) = restarted_worktree_with_persisted_vector().await;
        let worktree = temp.path().join("worktree");
        std::fs::write(worktree.join("src/fresh.rs"), "fn WORKTREE_FRESH() {}\n").unwrap();
        server
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();

        server
            .with_session(ref_id)
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();

        let name = cache_name("embeddings", &server.state.config);
        let persisted = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                let persisted = rkyv_store::mmap_vector_store(&worktree, &name)
                    .unwrap()
                    .map(|store| store.to_cache())
                    .unwrap_or_default();
                if persisted.contains_key("src/fresh.rs") {
                    return persisted;
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the worktree fill never persisted its new vector");
        let mut keys: Vec<_> = persisted.keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            ["src/changed.rs", "src/fresh.rs"],
            "the worktree persisted vectors its parent holds"
        );
    }

    #[tokio::test]
    async fn evicted_worktree_reuses_its_persisted_vectors_instead_of_reembedding() {
        let (ollama, _temp, server, ref_id) = restarted_worktree_with_persisted_vector().await;
        let worktree_server = server.with_session(ref_id);
        worktree_server
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();
        let worktree_ref = server.state.ref_index(ref_id).await.unwrap();
        clear_ref_heavy_caches(
            &worktree_ref,
            &cache_name("identifier-embeddings", &server.state.config),
        )
        .await;

        let result = worktree_server
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();
        assert_eq!(
            matching_embed_input_count(&ollama, "WORKTREE_CHANGED_VERSION").await,
            0,
            "an evicted worktree re-embedded a file whose vector it had persisted"
        );
        assert!(
            text_of(&result).contains("1. src/changed.rs"),
            "the persisted vector must rank the file: {}",
            text_of(&result)
        );
    }

    #[tokio::test]
    async fn attached_worktree_filler_completes_missing_vectors_for_the_next_query() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                let vectors: Vec<Vec<f32>> = inputs
                    .iter()
                    .map(|input| {
                        if input == "invoice payment status"
                            || input.contains("WORKTREE_FILL_TARGET")
                        {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        }
                    })
                    .collect();
                let response = ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embeddings": vectors }));
                if inputs
                    .iter()
                    .any(|input| input.contains("WORKTREE_FILL_TARGET"))
                {
                    response.set_delay(std::time::Duration::from_millis(100))
                } else {
                    response
                }
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        std::fs::write(
            worktree.path().join("target.rs"),
            "fn WORKTREE_FILL_TARGET() { /* invoice payment status */ }\n",
        )
        .unwrap();
        let server = ContextPlusServer::new(
            primary.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 10, 500),
        );
        let canonical_worktree = worktree.path().canonicalize().unwrap();
        let mut attach_args = serde_json::Map::new();
        attach_args.insert(
            "path".into(),
            json!(canonical_worktree.to_string_lossy().into_owned()),
        );
        let attached = server.handle_attach_worktree(attach_args).await.unwrap();
        assert_eq!(attached.is_error, Some(false), "{}", text_of(&attached));
        let ref_id = crate::ref_index::RefId::for_canonical_path(&canonical_worktree);
        let worktree_ref = server.state.ref_index(ref_id).await.unwrap();
        let worktree_server = server.with_session(ref_id);

        let first = worktree_server
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();
        assert!(
            !text_of(&first).contains("target.rs"),
            "a vectorless document must not receive synthetic semantic credit: {}",
            text_of(&first)
        );

        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if worktree_ref
                    .embedding_cache
                    .read()
                    .await
                    .contains_key("target.rs")
                    && worktree_ref.search_index_cache.read().await.is_none()
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the attached ref filler must cache the vector and invalidate its index");
        assert!(
            !server
                .current_ref()
                .await
                .embedding_cache
                .read()
                .await
                .contains_key("target.rs"),
            "the worktree filler must not write into the primary cache"
        );
        let document_requests_before_next_query =
            matching_embed_input_count(&ollama, "WORKTREE_FILL_TARGET").await;

        let next = worktree_server
            .handle_semantic_code_search(semantic_args("invoice payment status"))
            .await
            .unwrap();
        assert!(
            text_of(&next).contains("1. target.rs"),
            "the next query must use the filled vector: {}",
            text_of(&next)
        );
        assert_eq!(
            matching_embed_input_count(&ollama, "WORKTREE_FILL_TARGET").await,
            document_requests_before_next_query,
            "the next query must use the cached fill instead of re-embedding"
        );
    }

    struct GatedOllama {
        uri: String,
        fill_started: Arc<tokio::sync::Semaphore>,
        release_fill: Arc<tokio::sync::Semaphore>,
        slow_query_started: Arc<tokio::sync::Semaphore>,
        task: tokio::task::JoinHandle<()>,
    }

    impl GatedOllama {
        async fn start() -> Self {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let fill_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let fill_started = Arc::new(tokio::sync::Semaphore::new(0));
            let release_fill = Arc::new(tokio::sync::Semaphore::new(0));
            let slow_query_started = Arc::new(tokio::sync::Semaphore::new(0));
            let task_fill_calls = Arc::clone(&fill_calls);
            let task_fill_started = Arc::clone(&fill_started);
            let task_release_fill = Arc::clone(&release_fill);
            let task_slow_started = Arc::clone(&slow_query_started);
            let task = tokio::spawn(async move {
                loop {
                    let Ok((mut stream, _)) = listener.accept().await else {
                        return;
                    };
                    let fill_calls = Arc::clone(&task_fill_calls);
                    let fill_started = Arc::clone(&task_fill_started);
                    let release_fill = Arc::clone(&task_release_fill);
                    let slow_started = Arc::clone(&task_slow_started);
                    tokio::spawn(async move {
                        let mut request = Vec::new();
                        let content_length = loop {
                            let mut chunk = [0_u8; 4096];
                            let Ok(read) = stream.read(&mut chunk).await else {
                                return;
                            };
                            if read == 0 {
                                return;
                            }
                            request.extend_from_slice(&chunk[..read]);
                            let Some(headers_end) =
                                request.windows(4).position(|window| window == b"\r\n\r\n")
                            else {
                                continue;
                            };
                            let headers = String::from_utf8_lossy(&request[..headers_end]);
                            let length = headers
                                .lines()
                                .find_map(|line| {
                                    line.to_ascii_lowercase()
                                        .strip_prefix("content-length:")
                                        .and_then(|value| value.trim().parse::<usize>().ok())
                                })
                                .unwrap_or(0);
                            if request.len() >= headers_end + 4 + length {
                                break length;
                            }
                        };
                        let headers_end = request
                            .windows(4)
                            .position(|window| window == b"\r\n\r\n")
                            .unwrap();
                        while request.len() < headers_end + 4 + content_length {
                            let mut chunk = [0_u8; 4096];
                            let Ok(read) = stream.read(&mut chunk).await else {
                                return;
                            };
                            if read == 0 {
                                return;
                            }
                            request.extend_from_slice(&chunk[..read]);
                        }
                        let body: serde_json::Value = serde_json::from_slice(
                            &request[headers_end + 4..headers_end + 4 + content_length],
                        )
                        .unwrap();
                        let inputs: Vec<_> = body["input"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter_map(|input| input.as_str())
                            .collect();
                        if inputs.iter().any(|input| input.contains("FILL_COMPLETES")) {
                            if fill_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                                std::future::pending::<()>().await;
                            }
                            fill_started.add_permits(1);
                            release_fill.acquire().await.unwrap().forget();
                        } else if inputs
                            .iter()
                            .any(|input| input.contains("SLOW_FRESH_DELTA"))
                        {
                            slow_started.add_permits(1);
                            std::future::pending::<()>().await;
                        }
                        let body = serde_json::json!({
                            "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                        })
                        .to_string();
                        let response = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(response.as_bytes()).await;
                    });
                }
            });
            Self {
                uri: format!("http://{address}"),
                fill_started,
                release_fill,
                slow_query_started,
                task,
            }
        }
    }

    impl Drop for GatedOllama {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    #[tokio::test]
    async fn semantic_background_fill_breaks_query_budget_livelock() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                let is_document = inputs.iter().any(|input| input.contains("FILL_DOC"));
                let vectors: Vec<Vec<f32>> = inputs
                    .iter()
                    .map(|input| {
                        if input.contains("target.rs") || input == "needle" {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        }
                    })
                    .collect();
                let response = ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embeddings": vectors }));
                if is_document {
                    response.set_delay(std::time::Duration::from_millis(120))
                } else {
                    response
                }
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("target.rs"), "fn FILL_DOC_target() {}\n").unwrap();
        std::fs::write(root.path().join("other.rs"), "fn FILL_DOC_other() {}\n").unwrap();
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 20, 500),
        );

        let first = server
            .handle_semantic_code_search(semantic_args("needle"))
            .await
            .unwrap();
        assert_eq!(first.is_error, Some(false), "{}", text_of(&first));

        while server
            .current_ref()
            .await
            .embedding_cache
            .read()
            .await
            .len()
            != 2
        {
            tokio::task::yield_now().await;
        }

        let document_requests_before = matching_embed_request_batches(&ollama, "FILL_DOC").await;
        let second = server
            .handle_semantic_code_search(semantic_args("needle"))
            .await
            .unwrap();
        assert_eq!(
            matching_embed_request_batches(&ollama, "FILL_DOC").await,
            document_requests_before,
            "a later query must not resubmit filled documents"
        );
        assert!(
            text_of(&second).contains("1. target.rs"),
            "the filled vector must rank the target: {}",
            text_of(&second)
        );
    }

    #[tokio::test]
    async fn semantic_queries_do_not_resubmit_documents_while_fill_is_running() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                let response = ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                }));
                if inputs.iter().any(|input| input.contains("QUEUED_DOC")) {
                    response.set_delay(std::time::Duration::from_millis(300))
                } else {
                    response
                }
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("queued.rs"), "fn QUEUED_DOC() {}\n").unwrap();
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 20, 1_000),
        );
        server
            .handle_semantic_code_search(semantic_args("needle"))
            .await
            .unwrap();

        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        loop {
            if matching_embed_request_batches(&ollama, "QUEUED_DOC")
                .await
                .len()
                >= 2
            {
                break;
            }
            assert!(Instant::now() < deadline, "background fill never started");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        let requests_while_running = matching_embed_request_batches(&ollama, "QUEUED_DOC").await;

        for _ in 0..3 {
            server
                .handle_semantic_code_search(semantic_args("needle"))
                .await
                .unwrap();
        }

        assert_eq!(
            matching_embed_request_batches(&ollama, "QUEUED_DOC").await,
            requests_while_running,
            "queries must answer from the current index while the queued document is filling"
        );
    }

    #[tokio::test]
    async fn semantic_background_fill_splits_timed_out_batches() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                let response = ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                }));
                if inputs.iter().any(|input| input.contains("SPLIT_SLOW")) {
                    response.set_delay(std::time::Duration::from_millis(150))
                } else if inputs.iter().any(|input| input.contains("SPLIT_DOC")) {
                    response.set_delay(std::time::Duration::from_millis(30))
                } else {
                    response
                }
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("a.rs"), "fn SPLIT_DOC_a() {}\n").unwrap();
        std::fs::write(root.path().join("b.rs"), "fn SPLIT_DOC_b() {}\n").unwrap();
        std::fs::write(root.path().join("z.rs"), "fn SPLIT_DOC_SPLIT_SLOW() {}\n").unwrap();
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 10, 60),
        );
        server
            .handle_semantic_code_search(semantic_args("needle"))
            .await
            .unwrap();

        let deadline = Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let current_ref = server.current_ref().await;
            let cache = current_ref.embedding_cache.read().await;
            if cache.contains_key("a.rs") && cache.contains_key("b.rs") {
                assert!(
                    !cache.contains_key("z.rs"),
                    "the timed-out singleton must not block or masquerade as a success"
                );
                break;
            }
            drop(cache);
            assert!(
                Instant::now() < deadline,
                "fast split halves were not cached"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        let batches = matching_embed_request_batches(&ollama, "SPLIT_DOC").await;
        assert!(
            batches.iter().filter(|&&size| size == 3).count() >= 2,
            "expected the query attempt and filler attempt for the full batch: {batches:?}"
        );
        assert!(
            batches.contains(&2) && batches.contains(&1),
            "a timed-out batch must be retried in halves down to a singleton: {batches:?}"
        );
    }

    #[tokio::test]
    async fn semantic_permanent_failure_is_suppressed_until_content_hash_changes() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                if inputs
                    .iter()
                    .any(|input| input.contains("PERMANENT_FAIL_v1"))
                {
                    ResponseTemplate::new(500)
                } else {
                    ResponseTemplate::new(200).set_body_json(serde_json::json!({
                        "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                    }))
                }
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("failure.rs");
        std::fs::write(&file, "fn PERMANENT_FAIL_v1() {}\n").unwrap();
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 30, 200),
        );
        server
            .handle_semantic_code_search(semantic_args("needle"))
            .await
            .unwrap();

        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        loop {
            let attempts = matching_embed_request_batches(&ollama, "PERMANENT_FAIL_v1")
                .await
                .len();
            if attempts == 3 {
                break;
            }
            assert!(attempts < 3, "failure threshold was exceeded: {attempts}");
            assert!(
                Instant::now() < deadline,
                "failure did not reach three attempts"
            );
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        for _ in 0..3 {
            server
                .handle_semantic_code_search(semantic_args("needle"))
                .await
                .unwrap();
        }
        assert_eq!(
            matching_embed_request_batches(&ollama, "PERMANENT_FAIL_v1")
                .await
                .len(),
            3,
            "the query path and filler must both suppress a permanently failed content hash"
        );

        let changed = "fn RECOVERED_v2_with_different_size() {}\n";
        std::fs::write(&file, changed).unwrap();
        server
            .handle_semantic_code_search(semantic_args("needle"))
            .await
            .unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        loop {
            let current_ref = server.current_ref().await;
            let cache = current_ref.embedding_cache.read().await;
            if cache
                .get("failure.rs")
                .is_some_and(|entry| entry.hash == crate::core::embeddings::content_hash(changed))
            {
                break;
            }
            drop(cache);
            assert!(
                Instant::now() < deadline,
                "changed content was not embedded"
            );
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert!(
            !matching_embed_request_batches(&ollama, "RECOVERED_v2")
                .await
                .is_empty(),
            "a new content hash must clear permanent-failure suppression"
        );
    }

    #[tokio::test]
    async fn r3_dimension_rebuild_reembeds_retained_documents() {
        use crate::tools::semantic_search::{CachedSearchIndex, SearchDocument, WalkAndIndexFn};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &wiremock::Request| {
                let vectors: Vec<Vec<f32>> = embed_request_inputs(request)
                    .iter()
                    .map(|text| {
                        if text == "needle" {
                            vec![1.0, 0.0]
                        } else {
                            vec![1.0, 0.0, 0.0]
                        }
                    })
                    .collect();
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embeddings": vectors }))
            })
            .mount(&ollama)
            .await;
        let root = tempfile::tempdir().unwrap();
        let old = "fn retained() {}\n";
        let changed = "fn changed() {}\n";
        std::fs::write(root.path().join("old.rs"), old).unwrap();
        std::fs::write(root.path().join("changed.rs"), changed).unwrap();
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 20, 500),
        );
        let owner = server.current_ref().await;
        for (path, content) in [("old.rs", old), ("changed.rs", changed)] {
            owner.embedding_cache.write().await.insert(
                path.into(),
                CacheEntry {
                    hash: crate::core::embeddings::content_hash(content),
                    vector: vec![1.0, 0.0],
                },
            );
        }
        server
            .handle_semantic_code_search(semantic_args("needle"))
            .await
            .unwrap();
        owner
            .embedding_cache
            .write()
            .await
            .get_mut("changed.rs")
            .unwrap()
            .vector = vec![1.0, 0.0, 0.0];
        {
            let mut cache = owner.search_index_cache.write().await;
            CachedSearchIndex::refresh_ref_paths(
                cache.as_mut().unwrap(),
                &root.path().canonicalize().unwrap(),
                vec![SearchDocument::new(
                    "changed.rs".into(),
                    String::new(),
                    vec![],
                    vec![],
                    changed.into(),
                )],
                vec![Some(vec![1.0, 0.0, 0.0])],
                &[],
                1,
            );
        }
        let walker = crate::server_adapters::RefWalkerIndexer {
            ref_index: owner.clone(),
            walker: CachedWalkerIndexer {
                config: server.state.config.clone(),
                ollama: server.state.ollama.clone(),
                state: server.state.clone(),
            },
        };
        let (docs, vectors) = walker.walk_and_index(root.path()).await.unwrap();
        assert_eq!(docs.len(), 2);
        assert!(
            vectors
                .iter()
                .all(|v| v.as_ref().is_some_and(|v| v.len() == 3)),
            "background shape rebuild must obtain replacements for retained vectors"
        );
    }

    #[tokio::test]
    async fn semantic_background_fill_updates_warm_index_without_full_rebuild() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                let vectors: Vec<Vec<f32>> = inputs
                    .iter()
                    .map(|input| {
                        if input == "needle" || input.contains("VECTOR_TARGET") {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        }
                    })
                    .collect();
                let response = ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embeddings": vectors }));
                if inputs.iter().any(|input| input.contains("VECTOR_TARGET")) {
                    response.set_delay(std::time::Duration::from_millis(100))
                } else {
                    response
                }
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().unwrap();
        let target_content = "fn VECTOR_TARGET() {}\n";
        let decoy_content = "fn unrelated_decoy() {}\n";
        std::fs::write(root.path().join("target.rs"), target_content).unwrap();
        std::fs::write(root.path().join("decoy.rs"), decoy_content).unwrap();
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 20, 500),
        );
        server
            .current_ref()
            .await
            .embedding_cache
            .write()
            .await
            .insert(
                "decoy.rs".to_string(),
                CacheEntry {
                    hash: crate::core::embeddings::content_hash(decoy_content),
                    vector: vec![0.0, 1.0],
                },
            );

        let before = server
            .handle_semantic_code_search(semantic_args("needle"))
            .await
            .unwrap();
        assert!(
            !text_of(&before).contains("target.rs"),
            "a vectorless target must not satisfy a semantic-only query"
        );
        let warm = server
            .current_ref()
            .await
            .search_index_cache
            .read()
            .await
            .as_ref()
            .cloned()
            .expect("the partial query must leave a warm SearchIndex");
        let rebuilds_before = warm.index.full_rebuild_count();

        let deadline = Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let vector_ready = server
                .current_ref()
                .await
                .embedding_cache
                .read()
                .await
                .contains_key("target.rs");
            let ref_index = server.current_ref().await;
            let fill_finished = !crate::server_adapters::test_seams::fill_running(&ref_index).await;
            if vector_ready && fill_finished {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "fill did not cache the vector and finish"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        let after_fill = server
            .current_ref()
            .await
            .search_index_cache
            .read()
            .await
            .as_ref()
            .cloned()
            .expect("a one-file fill must retain the warm SearchIndex");
        assert_eq!(
            after_fill.index.full_rebuild_count(),
            rebuilds_before,
            "a one-file fill must be routed through an incremental delta"
        );

        let walks_before = server
            .current_ref()
            .await
            .semantic_walks
            .load(std::sync::atomic::Ordering::Relaxed);
        let after = server
            .handle_semantic_code_search(semantic_args("needle"))
            .await
            .unwrap();
        assert_eq!(
            server
                .current_ref()
                .await
                .semantic_walks
                .load(std::sync::atomic::Ordering::Relaxed),
            walks_before,
            "next query reread and reparsed unchanged files after fill"
        );
        assert!(
            text_of(&after).contains("1. target.rs"),
            "the incrementally updated index must rank the newly filled vector: {}",
            text_of(&after)
        );
        assert_eq!(
            server
                .current_ref()
                .await
                .search_index_cache
                .read()
                .await
                .as_ref()
                .unwrap()
                .index
                .full_rebuild_count(),
            rebuilds_before,
            "the visibility query must not rebuild the full index"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn semantic_filler_rejects_non_regular_replacement() {
        let ollama = GatedOllama::start().await;
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("fill.rs");
        std::fs::write(&file, "fn FILL_COMPLETES_h1() {}\n").unwrap();
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri, 20, 5_000),
        );
        let walker = CachedWalkerIndexer {
            config: server.state.config.clone(),
            ollama: server.state.ollama.clone(),
            state: Arc::clone(&server.state),
        };
        walker.walk_and_index(root.path()).await.unwrap();
        ollama.fill_started.acquire().await.unwrap().forget();
        std::fs::remove_file(&file).unwrap();
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&file)
                .status()
                .unwrap()
                .success()
        );
        let unrelated = root.path().join("unrelated");
        std::fs::create_dir(&unrelated).unwrap();
        std::fs::write(unrelated.join("fresh.rs"), "fn fresh() {}\n").unwrap();
        ollama.release_fill.add_permits(1);
        let ref_index = server.current_ref().await;
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while crate::server_adapters::test_seams::fill_running(&ref_index).await {
                tokio::task::yield_now().await;
            }
            assert_eq!(
                ref_index
                    .semantic_vector_generation
                    .load(std::sync::atomic::Ordering::Acquire),
                0,
                "rejected fills must not advance the vector generation"
            );
            walker.walk_and_index(&unrelated).await.unwrap();
            let cache = ref_index.embedding_cache.read().await;
            assert!(
                !cache.contains_key("fill.rs"),
                "invalid filler vector was installed"
            );
        })
        .await
        .expect("filler validation must leave cache reads and admission available");
    }

    #[tokio::test]
    async fn semantic_filler_rejects_stale_completion() {
        let ollama = GatedOllama::start().await;
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("fill.rs");
        std::fs::write(&file, "fn FILL_COMPLETES_h1() {}\n").unwrap();
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri, 20, 5_000),
        );
        let walker = CachedWalkerIndexer {
            config: server.state.config.clone(),
            ollama: server.state.ollama.clone(),
            state: Arc::clone(&server.state),
        };
        walker.walk_and_index(root.path()).await.unwrap();
        ollama.fill_started.acquire().await.unwrap().forget();
        std::fs::write(&file, "fn changed_to_h2() {}\n").unwrap();
        ollama.release_fill.add_permits(1);
        let ref_index = server.current_ref().await;
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while crate::server_adapters::test_seams::fill_running(&ref_index).await {
                tokio::task::yield_now().await;
            }
            assert_eq!(
                ref_index
                    .semantic_vector_generation
                    .load(std::sync::atomic::Ordering::Acquire),
                0,
                "rejected fills must not advance the vector generation"
            );
            let _admission = ref_index.semantic_fill.lock().await;
            let cache = ref_index.embedding_cache.read().await;
            assert!(
                !cache.contains_key("fill.rs"),
                "invalid filler vector was installed"
            );
        })
        .await
        .expect("filler validation must leave cache reads and admission available");
    }

    #[tokio::test]
    async fn semantic_fill_installs_completed_batch_while_fresh_delta_query_is_slow() {
        let ollama = GatedOllama::start().await;

        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("fill.rs"),
            "fn FILL_COMPLETES_in_background() {}\n",
        )
        .unwrap();
        let mut config = semantic_fill_config(&ollama.uri, 1_200, 2_000);
        config.ollama_max_concurrent = 2;
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);

        let mut first_config = server.state.config.clone();
        first_config.embed_budget_ms = 20;
        let first_walker = CachedWalkerIndexer {
            config: first_config,
            ollama: server.state.ollama.clone(),
            state: Arc::clone(&server.state),
        };
        first_walker.walk_and_index(root.path()).await.unwrap();

        ollama.fill_started.acquire().await.unwrap().forget();

        std::fs::write(
            root.path().join("fresh.rs"),
            "fn SLOW_FRESH_DELTA_blocks_query() {}\n",
        )
        .unwrap();
        let slow_server = server.clone();
        let slow_query = tokio::spawn(async move {
            slow_server
                .handle_semantic_code_search(semantic_args("needle"))
                .await
        });

        ollama.slow_query_started.acquire().await.unwrap().forget();
        ollama.release_fill.add_permits(1);

        tokio::time::timeout(std::time::Duration::from_millis(150), async {
            loop {
                if server
                    .current_ref()
                    .await
                    .embedding_cache
                    .read()
                    .await
                    .contains_key("fill.rs")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("a released filler response must install while the fresh query is embedding");

        slow_query.abort();
    }

    #[tokio::test]
    async fn scoped_walker_preserves_documentation_scope_and_prior() {
        use crate::tools::semantic_search::{ResolvedSearchOptions, SearchIndex, SearchScope};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; embed_request_inputs(request).len()]
                }))
            })
            .mount(&ollama)
            .await;
        let root = tempfile::tempdir().unwrap();
        for (directory, filename) in [
            (
                "docs",
                "observability/betterstack-dashboards/14-phi-scrub-verification.json",
            ),
            ("packages/db/migrations", "202609260001_add_phi_fields.sql"),
            ("packages/db/queries", "phi_fields.sql"),
        ] {
            let file = root.path().join(directory).join(filename);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, "{}\n").unwrap();
        }
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 1_000, 3_000),
        );
        let walker = CachedWalkerIndexer {
            config: server.state.config.clone(),
            ollama: server.state.ollama.clone(),
            state: Arc::clone(&server.state),
        };
        for (directory, filename, documentation) in [
            (
                "docs",
                "observability/betterstack-dashboards/14-phi-scrub-verification.json",
                true,
            ),
            (
                "packages/db/migrations",
                "202609260001_add_phi_fields.sql",
                true,
            ),
            ("packages/db/queries", "phi_fields.sql", false),
        ] {
            let (docs, _) = walker
                .walk_and_index(&root.path().join(directory))
                .await
                .unwrap();
            assert_eq!(docs.len(), 1);
            assert_eq!(docs[0].path, filename);
            let mut index = SearchIndex::new();
            index.index_with_vectors(docs, vec![Some(vec![1.0, 0.0])]);
            let opts = ResolvedSearchOptions {
                semantic_weight: 1.0,
                keyword_weight: 0.0,
                min_semantic_score: 0.0,
                min_keyword_score: 0.0,
                min_combined_score: 0.0,
                ..Default::default()
            };
            let results = index.search("unrelated", &[1.0, 0.0], &opts);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].path, filename);
            let expected = if documentation { 80.0 } else { 100.0 };
            assert!(
                (results[0].score - expected).abs() < 1e-6,
                "{directory}: expected prior {expected}, got {:?}",
                results[0]
            );
            for scope in [SearchScope::Code, SearchScope::Docs] {
                let results = index.search(
                    "unrelated",
                    &[1.0, 0.0],
                    &ResolvedSearchOptions {
                        scope,
                        ..opts.clone()
                    },
                );
                assert_eq!(
                    results.len(),
                    usize::from((scope == SearchScope::Docs) == documentation),
                    "{directory}: scope {scope:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn semantic_aborted_query_releases_reservations_to_background_fill() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let document_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls = Arc::clone(&document_calls);
        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |request: &Request| {
                let inputs = embed_request_inputs(request);
                let response = ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                }));
                if inputs
                    .iter()
                    .any(|input| input.contains("ABORTED_RESERVATION"))
                    && calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0
                {
                    response.set_delay(std::time::Duration::from_millis(300))
                } else {
                    response
                }
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("reserved.rs"),
            "fn ABORTED_RESERVATION() {}\n",
        )
        .unwrap();
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 1_000, 3_000),
        );
        let query_server = server.clone();
        let query = tokio::spawn(async move {
            query_server
                .handle_semantic_code_search(semantic_args("needle"))
                .await
        });

        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        loop {
            if document_calls.load(std::sync::atomic::Ordering::SeqCst) == 1 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "query never acquired the document reservation"
            );
            tokio::task::yield_now().await;
        }
        query.abort();
        query.await.expect_err("query task must be cancelled");

        tokio::time::timeout(std::time::Duration::from_millis(150), async {
            loop {
                if server
                    .current_ref()
                    .await
                    .embedding_cache
                    .read()
                    .await
                    .contains_key("reserved.rs")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("background fill must claim and install an aborted query's reservation");
        assert!(
            document_calls.load(std::sync::atomic::Ordering::SeqCst) >= 2,
            "the background filler must make its own request after the query is aborted"
        );
    }

    #[tokio::test]
    async fn semantic_admission_rechecks_vector_filled_after_walk_snapshot() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                }))
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().unwrap();
        let content = "fn FILLED_BETWEEN_SNAPSHOT_AND_ADMISSION() {}\n";
        std::fs::write(root.path().join("filled.rs"), content).unwrap();
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 300, 1_000),
        );
        let ref_index = server.current_ref().await;
        let pause = crate::server_adapters::test_seams::pause_after_cache_snapshot(root.path());

        let query_server = server.clone();
        let query = tokio::spawn(async move {
            query_server
                .handle_semantic_code_search(semantic_args("needle"))
                .await
        });

        pause.wait_until_entered().await;
        let mut cache = ref_index.embedding_cache.write().await;
        cache.insert(
            "filled.rs".to_string(),
            CacheEntry {
                hash: crate::core::embeddings::content_hash(content),
                vector: vec![1.0, 0.0],
            },
        );
        drop(cache);
        pause.resume();

        query.await.unwrap().unwrap();
        assert!(
            matching_embed_request_batches(&ollama, "FILLED_BETWEEN_SNAPSHOT_AND_ADMISSION")
                .await
                .is_empty(),
            "admission must use the newly filled cache entry instead of embedding it again"
        );
    }

    #[tokio::test]
    async fn semantic_inheritance_preserves_newer_child_cache_and_pending_fill() {
        let ollama = wiremock::MockServer::start().await;
        let primary = tempfile::tempdir().unwrap();
        let child = tempfile::tempdir().unwrap();
        let path = "versioned.rs";
        let old_content = "fn INHERITED_OLDER_A() {}\n";
        let new_content = "fn CHILD_NEWER_B() {}\n";
        std::fs::write(primary.path().join(path), old_content).unwrap();
        std::fs::write(child.path().join(path), old_content).unwrap();

        let server = ContextPlusServer::new(
            primary.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 1_000, 1_000),
        );
        let old_hash = crate::core::embeddings::content_hash(old_content);
        server
            .current_ref()
            .await
            .embedding_cache
            .write()
            .await
            .insert(
                path.to_string(),
                CacheEntry {
                    hash: old_hash.clone(),
                    vector: vec![1.0, 0.0],
                },
            );

        let canonical_child = child.path().canonicalize().unwrap();
        let child_id = crate::ref_index::RefId::for_canonical_path(&canonical_child);
        let child_ref = server
            .state
            .attach_ref(child_id, || {
                Arc::new(crate::ref_index::RefIndex::new(
                    canonical_child.clone(),
                    canonical_child.clone(),
                    Some(server.state.default_ref_id),
                ))
            })
            .await;
        let pause =
            crate::server_adapters::test_seams::pause_after_file_snapshot(child.path(), old_hash);
        let walk_root = child.path().to_path_buf();
        let walk_state = Arc::clone(&server.state);
        let stale_walk = tokio::spawn(async move {
            CachedWalkerIndexer {
                config: walk_state.config.clone(),
                ollama: walk_state.ollama.clone(),
                state: walk_state,
            }
            .walk_and_index(&walk_root)
            .await
        });

        pause.wait_until_entered().await;
        std::fs::write(child.path().join(path), new_content).unwrap();
        let new_hash = crate::core::embeddings::content_hash(new_content);
        child_ref.embedding_cache.write().await.insert(
            path.to_string(),
            CacheEntry {
                hash: new_hash.clone(),
                vector: vec![0.0, 1.0],
            },
        );
        crate::server_adapters::test_seams::seed_pending(
            &child_ref,
            path,
            new_hash.clone(),
            new_content.to_string(),
        )
        .await;
        pause.resume();
        stale_walk.await.unwrap().unwrap();

        assert_eq!(
            child_ref
                .embedding_cache
                .read()
                .await
                .get(path)
                .map(|entry| entry.hash.as_str()),
            Some(new_hash.as_str()),
            "stale inherited content must not replace the child's newer cached hash"
        );
        assert_eq!(
            crate::server_adapters::test_seams::pending_hash(&child_ref, path).await,
            Some(new_hash),
            "stale inheritance must not cancel pending work for the child's newer hash"
        );
    }

    #[tokio::test]
    async fn semantic_ancestor_lookup_waits_for_registry_writer_and_reuses_vector() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                }))
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().unwrap();
        let child = tempfile::tempdir().unwrap();
        let path = "shared.rs";
        let content = "fn REGISTRY_CONTENTION_SHARED() {}\n";
        std::fs::write(primary.path().join(path), content).unwrap();
        std::fs::write(child.path().join(path), content).unwrap();
        let server = ContextPlusServer::new(
            primary.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 1_000, 1_000),
        );
        let hash = crate::core::embeddings::content_hash(content);
        server
            .current_ref()
            .await
            .embedding_cache
            .write()
            .await
            .insert(
                path.to_string(),
                CacheEntry {
                    hash: hash.clone(),
                    vector: vec![1.0, 0.0],
                },
            );

        let canonical_child = child.path().canonicalize().unwrap();
        let child_id = crate::ref_index::RefId::for_canonical_path(&canonical_child);
        server
            .state
            .attach_ref(child_id, || {
                Arc::new(crate::ref_index::RefIndex::new(
                    canonical_child.clone(),
                    canonical_child,
                    Some(server.state.default_ref_id),
                ))
            })
            .await;
        let child_server = server.with_session(child_id);
        let pause = crate::server_adapters::test_seams::pause_after_file_snapshot(
            &child.path().canonicalize().unwrap(),
            hash,
        );
        let query = tokio::spawn(async move {
            child_server
                .handle_semantic_code_search(semantic_args("needle"))
                .await
        });

        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            pause.wait_until_entered(),
        )
        .await
        .expect("query never reached the file snapshot");
        let registry_writer = server.state.refs.write().await;
        pause.resume();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(
            !query.is_finished(),
            "ancestor lookup treated registry contention as an absent parent"
        );
        drop(registry_writer);

        query.await.unwrap().unwrap();
        assert_eq!(
            matching_embed_input_count(&ollama, "REGISTRY_CONTENTION_SHARED").await,
            0,
            "the child must inherit the ancestor vector without embedding the document"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn semantic_older_observation_cannot_replace_newer_hash_in_flight() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                let response = ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                }));
                if inputs.iter().any(|input| input.contains("NEWER_HASH_v2")) {
                    response.set_delay(std::time::Duration::from_millis(150))
                } else {
                    response
                }
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("versioned.rs");
        let old_content = "fn OLDER_HASH_v1() {}\n";
        let new_content = "fn NEWER_HASH_v2() {}\n";
        std::fs::write(&source, old_content).unwrap();

        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 1_000, 1_000),
        );
        let pause = crate::server_adapters::test_seams::pause_after_file_snapshot(
            root.path(),
            crate::core::embeddings::content_hash(old_content),
        );
        let old_root = root.path().to_path_buf();
        let old_state = Arc::clone(&server.state);
        let old_walk = tokio::spawn(async move {
            CachedWalkerIndexer {
                config: old_state.config.clone(),
                ollama: old_state.ollama.clone(),
                state: old_state,
            }
            .walk_and_index(&old_root)
            .await
        });

        pause.wait_until_entered().await;
        std::fs::write(&source, new_content).unwrap();

        let new_root = root.path().to_path_buf();
        let new_state = Arc::clone(&server.state);
        let new_walk = tokio::spawn(async move {
            CachedWalkerIndexer {
                config: new_state.config.clone(),
                ollama: new_state.ollama.clone(),
                state: new_state,
            }
            .walk_and_index(&new_root)
            .await
        });

        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        loop {
            if !matching_embed_request_batches(&ollama, "NEWER_HASH_v2")
                .await
                .is_empty()
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "newer hash embedding never entered flight"
            );
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        pause.resume();

        new_walk.await.unwrap().unwrap();
        old_walk.await.unwrap().unwrap();

        let cached = server
            .current_ref()
            .await
            .embedding_cache
            .read()
            .await
            .get("versioned.rs")
            .cloned()
            .expect("versioned document must be cached");
        assert_eq!(
            cached.hash,
            crate::core::embeddings::content_hash(new_content),
            "an older observation admitted later must not replace the newer hash"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn semantic_inverse_read_order_installs_only_current_content_vector() {
        use std::io::Write;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                }))
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("versioned.rs");
        let older_content = "fn INVERSE_OLDER_H1() {}\n";
        let newer_content = "fn INVERSE_NEWER_H2() {}\n";
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&source)
                .status()
                .unwrap()
                .success()
        );
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 1_000, 1_000),
        );

        let first_root = root.path().to_path_buf();
        let first_state = Arc::clone(&server.state);
        let first_walk = tokio::spawn(async move {
            CachedWalkerIndexer {
                config: first_state.config.clone(),
                ollama: first_state.ollama.clone(),
                state: first_state,
            }
            .walk_and_index(&first_root)
            .await
        });
        let writer_path = source.clone();
        let mut first_writer = tokio::task::spawn_blocking(move || {
            std::fs::OpenOptions::new().write(true).open(writer_path)
        })
        .await
        .unwrap()
        .unwrap();

        std::fs::remove_file(&source).unwrap();
        std::fs::write(&source, older_content).unwrap();
        let older_pause = crate::server_adapters::test_seams::pause_after_file_snapshot(
            root.path(),
            crate::core::embeddings::content_hash(older_content),
        );
        let second_root = root.path().to_path_buf();
        let second_state = Arc::clone(&server.state);
        let second_walk = tokio::spawn(async move {
            CachedWalkerIndexer {
                config: second_state.config.clone(),
                ollama: second_state.ollama.clone(),
                state: second_state,
            }
            .walk_and_index(&second_root)
            .await
        });
        older_pause.wait_until_entered().await;

        std::fs::write(&source, newer_content).unwrap();
        first_writer.write_all(newer_content.as_bytes()).unwrap();
        drop(first_writer);
        first_walk.await.unwrap().unwrap();
        let current_hash = crate::core::embeddings::content_hash(newer_content);
        assert_eq!(
            server
                .current_ref()
                .await
                .embedding_cache
                .read()
                .await
                .get("versioned.rs")
                .map(|entry| entry.hash.as_str()),
            Some(current_hash.as_str()),
            "precondition: the first-started walk must install its newer observation"
        );

        older_pause.resume();
        second_walk.await.unwrap().unwrap();

        let cached = server
            .current_ref()
            .await
            .embedding_cache
            .read()
            .await
            .get("versioned.rs")
            .cloned()
            .expect("the current document must retain a vector");
        assert_eq!(
            cached.hash, current_hash,
            "a later admission of an older read must not install stale content"
        );
    }

    #[tokio::test]
    async fn semantic_failed_current_hash_supersedes_different_hash_in_flight() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                if inputs
                    .iter()
                    .any(|input| input.contains("FAILED_CURRENT_HBAD"))
                {
                    ResponseTemplate::new(500)
                } else {
                    let response = ResponseTemplate::new(200).set_body_json(serde_json::json!({
                        "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                    }));
                    if inputs.iter().any(|input| input.contains("IN_FLIGHT_HGOOD")) {
                        response.set_delay(std::time::Duration::from_millis(300))
                    } else {
                        response
                    }
                }
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("versioned.rs");
        let failed_content = "fn FAILED_CURRENT_HBAD() {}\n";
        let good_content = "fn IN_FLIGHT_HGOOD() {}\n";
        std::fs::write(&source, failed_content).unwrap();
        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 1_000, 1_000),
        );
        server
            .handle_semantic_code_search(semantic_args("needle"))
            .await
            .unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        loop {
            let attempts = matching_embed_request_batches(&ollama, "FAILED_CURRENT_HBAD")
                .await
                .len();
            if attempts == 3 {
                break;
            }
            assert!(attempts < 3, "failure threshold was exceeded: {attempts}");
            assert!(
                Instant::now() < deadline,
                "failed hash did not reach permanent suppression"
            );
            tokio::task::yield_now().await;
        }

        std::fs::write(&source, good_content).unwrap();
        let good_root = root.path().to_path_buf();
        let good_state = Arc::clone(&server.state);
        let good_walk = tokio::spawn(async move {
            CachedWalkerIndexer {
                config: good_state.config.clone(),
                ollama: good_state.ollama.clone(),
                state: good_state,
            }
            .walk_and_index(&good_root)
            .await
        });
        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        loop {
            if !matching_embed_request_batches(&ollama, "IN_FLIGHT_HGOOD")
                .await
                .is_empty()
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "changed hash never entered flight"
            );
            tokio::task::yield_now().await;
        }

        std::fs::write(&source, failed_content).unwrap();
        CachedWalkerIndexer {
            config: server.state.config.clone(),
            ollama: server.state.ollama.clone(),
            state: Arc::clone(&server.state),
        }
        .walk_and_index(root.path())
        .await
        .unwrap();
        good_walk.await.unwrap().unwrap();

        let stale_hash = crate::core::embeddings::content_hash(good_content);
        assert!(
            !server
                .current_ref()
                .await
                .embedding_cache
                .read()
                .await
                .get("versioned.rs")
                .is_some_and(|entry| entry.hash == stale_hash),
            "completion for superseded Hgood must not install after Hbad is current again"
        );
    }

    #[tokio::test]
    async fn semantic_subdirectory_search_fills_and_invalidates_non_default_ref() {
        use crate::ref_index::{RefId, RefIndex};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                let vectors: Vec<Vec<f32>> = inputs
                    .iter()
                    .map(|input| {
                        if input == "needle" || input.contains("SUBDIR_VECTOR_TARGET") {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        }
                    })
                    .collect();
                let response = ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "embeddings": vectors }));
                if inputs.iter().any(|input| input.contains("SUBDIR_")) {
                    response.set_delay(std::time::Duration::from_millis(100))
                } else {
                    response
                }
            })
            .mount(&ollama)
            .await;

        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        let canonical_worktree = worktree.path().canonicalize().unwrap();
        let scope = canonical_worktree.join("scope");
        std::fs::create_dir(&scope).unwrap();
        std::fs::write(scope.join("target.rs"), "fn SUBDIR_VECTOR_TARGET() {}\n").unwrap();
        std::fs::write(scope.join("decoy.rs"), "fn SUBDIR_VECTOR_DECOY() {}\n").unwrap();

        let server = ContextPlusServer::new(
            primary.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 20, 1_000),
        );
        let ref_id = RefId::for_canonical_path(&canonical_worktree);
        let worktree_ref = server
            .state
            .attach_ref(ref_id, || {
                Arc::new(RefIndex::new(
                    canonical_worktree.clone(),
                    canonical_worktree.clone(),
                    Some(server.state.default_ref_id),
                ))
            })
            .await;
        let session = server.with_session(ref_id);
        let mut args = semantic_args("needle");
        args.insert(
            "rootDir".into(),
            json!(scope.to_string_lossy().into_owned()),
        );

        session
            .handle_semantic_code_search(args.clone())
            .await
            .unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let worktree_count = worktree_ref.embedding_cache.read().await.len();
            let default_count = server
                .current_ref()
                .await
                .embedding_cache
                .read()
                .await
                .len();
            if worktree_count + default_count >= 2 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "subdirectory background fill did not finish"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        let after = session.handle_semantic_code_search(args).await.unwrap();
        let after_text = text_of(&after);
        let worktree_has_target = worktree_ref
            .embedding_cache
            .read()
            .await
            .contains_key("scope/target.rs");
        let default_has_target = server
            .current_ref()
            .await
            .embedding_cache
            .read()
            .await
            .contains_key("target.rs");
        assert!(
            worktree_has_target && !default_has_target && after_text.contains("1. target.rs"),
            "subdirectory fill must stay on the session ref and refresh its ranking: \
             worktree_has_target={worktree_has_target}, \
             default_has_target={default_has_target}, result={after_text}"
        );
    }

    #[tokio::test]
    async fn semantic_metadata_cache_is_scoped_to_canonical_search_root() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                }))
            })
            .mount(&ollama)
            .await;

        let root = tempfile::tempdir().unwrap();
        let scope_a = root.path().join("a");
        let scope_b = root.path().join("b");
        std::fs::create_dir_all(&scope_a).unwrap();
        std::fs::create_dir_all(&scope_b).unwrap();
        let item_a = scope_a.join("item.rs");
        let item_b = scope_b.join("item.rs");
        std::fs::write(&item_a, "fn SCOPE_A_ONLY() {}\n").unwrap();
        std::fs::write(&item_b, "fn SCOPE_B_ONLY() {}\n").unwrap();
        let modified = std::fs::metadata(&item_a).unwrap().modified().unwrap();
        std::fs::File::options()
            .write(true)
            .open(&item_b)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        let metadata_a = std::fs::metadata(&item_a).unwrap();
        let metadata_b = std::fs::metadata(&item_b).unwrap();
        assert_eq!(metadata_a.len(), metadata_b.len());
        assert_eq!(
            metadata_a.modified().unwrap(),
            metadata_b.modified().unwrap()
        );

        let server = ContextPlusServer::new(
            root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 1_000, 1_000),
        );
        let mut first_args = semantic_args("needle");
        first_args.insert(
            "rootDir".into(),
            json!(scope_a.to_string_lossy().into_owned()),
        );
        let first = server
            .handle_semantic_code_search(first_args)
            .await
            .unwrap();
        assert!(
            text_of(&first).contains("SCOPE_A_ONLY"),
            "precondition: first scope must build A's index: {}",
            text_of(&first)
        );

        let mut second_args = semantic_args("needle");
        second_args.insert(
            "rootDir".into(),
            json!(scope_b.to_string_lossy().into_owned()),
        );
        let second = server
            .handle_semantic_code_search(second_args)
            .await
            .unwrap();
        assert!(
            text_of(&second).contains("SCOPE_B_ONLY") && !text_of(&second).contains("SCOPE_A_ONLY"),
            "identical relative metadata in another root must not reuse A's index: {}",
            text_of(&second)
        );
    }

    #[tokio::test]
    async fn semantic_metadata_fingerprint_detects_edit_add_remove_and_rename() {
        let root = tempfile::tempdir().unwrap();
        let original = root.path().join("original.rs");
        std::fs::write(&original, "fn original() {}\n").unwrap();
        let config = semantic_fill_config("http://127.0.0.1:1", 20, 120_000);
        let server = ContextPlusServer::new(root.path().to_path_buf(), config);
        let walker = CachedWalkerIndexer {
            config: server.state.config.clone(),
            ollama: server.state.ollama.clone(),
            state: Arc::clone(&server.state),
        };

        let initial = walker
            .metadata_fingerprint(root.path())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            walker.metadata_fingerprint(root.path()).await.unwrap(),
            Some(initial.clone()),
            "an unchanged tree must have the same metadata fingerprint"
        );

        std::fs::write(
            &original,
            "fn original_with_a_larger_body() { let x = 1; }\n",
        )
        .unwrap();
        let edited = walker
            .metadata_fingerprint(root.path())
            .await
            .unwrap()
            .unwrap();
        assert_ne!(edited, initial, "a size-changing edit must be detected");

        let added_path = root.path().join("added.rs");
        std::fs::write(&added_path, "fn added() {}\n").unwrap();
        let added = walker
            .metadata_fingerprint(root.path())
            .await
            .unwrap()
            .unwrap();
        assert_ne!(added, edited, "an added path must be detected");

        std::fs::remove_file(&added_path).unwrap();
        let removed = walker
            .metadata_fingerprint(root.path())
            .await
            .unwrap()
            .unwrap();
        assert_ne!(removed, added, "a removed path must be detected");

        std::fs::rename(&original, root.path().join("renamed.rs")).unwrap();
        let renamed = walker
            .metadata_fingerprint(root.path())
            .await
            .unwrap()
            .unwrap();
        assert_ne!(renamed, removed, "a renamed path must be detected");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn semantic_metadata_fingerprint_falls_back_for_unavailable_entries() {
        let dangling_root = tempfile::tempdir().unwrap();
        std::fs::write(dangling_root.path().join("healthy.rs"), "fn healthy() {}\n").unwrap();
        std::os::unix::fs::symlink(
            dangling_root.path().join("missing-target.rs"),
            dangling_root.path().join("dangling.rs"),
        )
        .unwrap();
        let dangling_server = ContextPlusServer::new(
            dangling_root.path().to_path_buf(),
            semantic_fill_config("http://127.0.0.1:1", 20, 1_000),
        );
        let dangling_walker = CachedWalkerIndexer {
            config: dangling_server.state.config.clone(),
            ollama: dangling_server.state.ollama.clone(),
            state: Arc::clone(&dangling_server.state),
        };
        let dangling = dangling_walker
            .metadata_fingerprint(dangling_root.path())
            .await;
        assert!(
            matches!(dangling, Ok(None)),
            "a dangling symlink must disable the cheap fingerprint and fall back: {dangling:?}"
        );
    }

    #[tokio::test]
    async fn semantic_metadata_removal_between_enumeration_and_stat_falls_back_to_walk() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let ollama = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = embed_request_inputs(request);
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "embeddings": vec![vec![1.0, 0.0]; inputs.len()]
                }))
            })
            .mount(&ollama)
            .await;

        let removed_root = tempfile::tempdir().unwrap();
        let stable = removed_root.path().join("stable.rs");
        let ephemeral = removed_root.path().join("ephemeral.rs");
        std::fs::write(&stable, "fn STABLE_AFTER_REMOVAL() {}\n").unwrap();
        std::fs::write(&ephemeral, "fn REMOVE_AFTER_ENUMERATION() {}\n").unwrap();
        let server = ContextPlusServer::new(
            removed_root.path().to_path_buf(),
            semantic_fill_config(&ollama.uri(), 1_000, 1_000),
        );
        server
            .handle_semantic_code_search(semantic_args("needle"))
            .await
            .unwrap();

        let pause = crate::server_adapters::test_seams::pause_after_metadata_enumeration(
            removed_root.path(),
        );
        let query_server = server.clone();
        let query = tokio::spawn(async move {
            query_server
                .handle_semantic_code_search(semantic_args("needle"))
                .await
        });
        let wait_pause = Arc::clone(&pause);
        tokio::task::spawn_blocking(move || wait_pause.wait_until_enumerated())
            .await
            .unwrap();
        std::fs::remove_file(&ephemeral).unwrap();
        tokio::task::spawn_blocking(move || pause.resume())
            .await
            .unwrap();

        let result = query.await.unwrap().unwrap();
        assert!(
            result.is_error != Some(true) && text_of(&result).contains("STABLE_AFTER_REMOVAL"),
            "a regular-file removal race must fall back to the full walk: {}",
            text_of(&result)
        );
    }

    // --- linked worktrees layer their files and keyword index over the primary's ---

    fn lexdelta_config() -> Config {
        let mut config = Config::from_env();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        config
    }

    fn lexdelta_tag(i: usize) -> String {
        format!(
            "tag{}{}",
            (b'a' + (i % 26) as u8) as char,
            (b'a' + (i / 26) as u8) as char
        )
    }

    fn lexdelta_corpus(root: &std::path::Path, files: usize) {
        for i in 0..files {
            let dir = root.join(format!("src/area_{}", i % 4));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join(format!("file_{i}.rs")),
                format!(
                    "pub fn shared_symbol_{i}() -> usize {{ {i} }}\n// {} {}\n",
                    lexdelta_tag(i),
                    "common words vary ".repeat(i % 5 + 1)
                ),
            )
            .unwrap();
        }
    }

    /// Changes one file, adds one and deletes one.
    fn lexdelta_edit_worktree(root: &std::path::Path) {
        std::fs::write(
            root.join("src/area_1/file_1.rs"),
            "pub fn worktreechanged() -> usize { 1 }\n// common words\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/area_2/worktree_added.rs"),
            "pub fn worktreeadded() {}\n// shared symbol vary\n",
        )
        .unwrap();
        std::fs::remove_file(root.join("src/area_3/file_3.rs")).unwrap();
    }

    async fn lexdelta_attach(
        server: &ContextPlusServer,
        root: &std::path::Path,
    ) -> ContextPlusServer {
        let canonical = root.canonicalize().unwrap();
        let mut args = serde_json::Map::new();
        args.insert(
            "path".into(),
            json!(canonical.to_string_lossy().to_string()),
        );
        server.handle_attach_worktree(args).await.unwrap();
        server.with_session(crate::ref_index::RefId::for_canonical_path(&canonical))
    }

    async fn lexdelta_keywords(server: &ContextPlusServer, query: &str) -> String {
        let mut args = serde_json::Map::new();
        args.insert("query".into(), json!(query));
        args.insert("top_k".into(), json!(100));
        text_of(&server.handle_lexical_search(args).await.unwrap())
    }

    const LEXDELTA_QUERIES: &[&str] = &[
        "shared symbol",
        "common words vary",
        "usize",
        "worktreechanged",
        "worktreeadded",
        "tagba",
        "tagda",
        "tagfa",
        "shared_symbol_7 vary",
    ];

    async fn lexdelta_assert_matches_standalone(
        worktree: &ContextPlusServer,
        root: &std::path::Path,
    ) {
        let standalone = ContextPlusServer::new(root.to_path_buf(), lexdelta_config());
        for query in LEXDELTA_QUERIES {
            assert_eq!(
                lexdelta_keywords(worktree, query).await,
                lexdelta_keywords(&standalone, query).await,
                "worktree keywords for {query:?} differ from a standalone index of the worktree"
            );
        }
    }

    fn lexdelta_git(cwd: &std::path::Path, args: &[&str]) {
        let status = std::process::Command::new("git")
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
        assert!(status.status.success(), "git {args:?}: {status:?}");
    }

    #[tokio::test]
    async fn lexdelta_small_delta_worktree_layers_over_the_primary_caches() {
        let primary = tempfile::tempdir().unwrap();
        lexdelta_corpus(primary.path(), 40);
        let worktree = tempfile::tempdir().unwrap();
        lexdelta_corpus(worktree.path(), 40);
        lexdelta_edit_worktree(worktree.path());
        let server = ContextPlusServer::new(primary.path().to_path_buf(), lexdelta_config());
        let session = lexdelta_attach(&server, worktree.path()).await;

        lexdelta_assert_matches_standalone(&session, worktree.path()).await;
        assert!(
            lexdelta_keywords(&session, &lexdelta_tag(3))
                .await
                .contains("No lexical matches")
        );
        assert!(
            lexdelta_keywords(&session, &lexdelta_tag(1))
                .await
                .contains("No lexical matches")
        );
        assert!(
            lexdelta_keywords(&session, "worktreechanged")
                .await
                .contains("src/area_1/file_1.rs")
        );
        assert!(
            lexdelta_keywords(&session, "worktreeadded")
                .await
                .contains("src/area_2/worktree_added.rs")
        );

        let owner = session.current_ref().await;
        let primary_ref = server.state.default_ref().unwrap();
        let cache = owner.project_cache.read().await.clone().unwrap();
        let primary_cache = primary_ref.project_cache.read().await.clone().unwrap();
        assert!(
            cache
                .file_content
                .base()
                .is_some_and(|base| Arc::ptr_eq(base, primary_cache.file_content.own())),
            "the worktree's file contents do not layer over the primary's"
        );
        assert_eq!(
            cache.file_content.own().len(),
            2,
            "own = the changed and added file"
        );
        assert_eq!(
            cache.file_content["src/area_0/file_0.rs"].as_str(),
            primary_cache.file_content["src/area_0/file_0.rs"].as_str()
        );
        assert!(!cache.file_content.contains_key("src/area_3/file_3.rs"));
        let lexical = owner.lexical_search_cache.read().await.clone().unwrap();
        let primary_lexical = primary_ref
            .lexical_search_cache
            .read()
            .await
            .clone()
            .unwrap();
        assert!(
            lexical
                .base
                .as_ref()
                .is_some_and(|base| Arc::ptr_eq(&base.cached, &primary_lexical)),
            "the worktree's keyword index does not layer over the primary's"
        );
        assert_eq!(
            lexical.document_paths.len(),
            2,
            "delta = the changed and added file"
        );
    }

    #[tokio::test]
    async fn lexdelta_git_linked_worktree_layers_and_matches_standalone() {
        let primary = tempfile::tempdir().unwrap();
        lexdelta_git(primary.path(), &["init", "-q", "-b", "main"]);
        lexdelta_corpus(primary.path(), 40);
        lexdelta_git(primary.path(), &["add", "-A"]);
        lexdelta_git(primary.path(), &["commit", "-qm", "base"]);
        let holder = tempfile::tempdir().unwrap();
        let worktree = holder.path().join("feature");
        lexdelta_git(
            primary.path(),
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "feature",
                worktree.to_str().unwrap(),
            ],
        );
        std::fs::write(
            worktree.join("src/area_0/file_4.rs"),
            "pub fn committedchange() {}\n",
        )
        .unwrap();
        lexdelta_git(&worktree, &["commit", "-qam", "change"]);
        lexdelta_edit_worktree(&worktree);
        std::fs::write(
            primary.path().join("src/area_1/file_5.rs"),
            "pub fn primarydirty() {}\n",
        )
        .unwrap();

        let server = ContextPlusServer::new(primary.path().to_path_buf(), lexdelta_config());
        let session = lexdelta_attach(&server, &worktree).await;

        lexdelta_assert_matches_standalone(&session, &worktree).await;
        for (query, expected) in [
            ("committedchange", "src/area_0/file_4.rs"),
            (lexdelta_tag(5).as_str(), "src/area_1/file_5.rs"),
        ] {
            assert!(
                lexdelta_keywords(&session, query).await.contains(expected),
                "{query} does not find {expected}"
            );
        }
        assert!(
            lexdelta_keywords(&session, "primarydirty")
                .await
                .contains("No lexical matches")
        );
        let cache = session
            .current_ref()
            .await
            .project_cache
            .read()
            .await
            .clone()
            .unwrap();
        assert!(cache.file_content.base().is_some());
        let mut own: Vec<_> = cache.file_content.own().keys().cloned().collect();
        own.sort();
        assert_eq!(
            own,
            [
                "src/area_0/file_4.rs",
                "src/area_1/file_1.rs",
                "src/area_1/file_5.rs",
                "src/area_2/worktree_added.rs"
            ]
        );
    }

    #[tokio::test]
    async fn lexdelta_worktree_over_the_change_threshold_builds_a_standalone_index() {
        let primary = tempfile::tempdir().unwrap();
        lexdelta_corpus(primary.path(), 40);
        let worktree = tempfile::tempdir().unwrap();
        lexdelta_corpus(worktree.path(), 40);
        for i in 0..12 {
            std::fs::write(
                worktree
                    .path()
                    .join(format!("src/area_{}/file_{i}.rs", i % 4)),
                format!("pub fn rewritten_{i}() {{}}\n"),
            )
            .unwrap();
        }
        let server = ContextPlusServer::new(primary.path().to_path_buf(), lexdelta_config());
        let session = lexdelta_attach(&server, worktree.path()).await;

        lexdelta_assert_matches_standalone(&session, worktree.path()).await;
        let owner = session.current_ref().await;
        let cache = owner.project_cache.read().await.clone().unwrap();
        assert!(
            cache.file_content.base().is_none(),
            "a 30% delta must promote"
        );
        let lexical = owner.lexical_search_cache.read().await.clone().unwrap();
        assert!(lexical.base.is_none());
        assert_eq!(lexical.document_paths.len(), 40);
    }

    #[tokio::test]
    async fn lexdelta_primary_base_replacement_keeps_worktree_results_correct() {
        let primary = tempfile::tempdir().unwrap();
        lexdelta_corpus(primary.path(), 40);
        let worktree = tempfile::tempdir().unwrap();
        lexdelta_corpus(worktree.path(), 40);
        lexdelta_edit_worktree(worktree.path());
        let server = ContextPlusServer::new(primary.path().to_path_buf(), lexdelta_config());
        let session = lexdelta_attach(&server, worktree.path()).await;
        lexdelta_assert_matches_standalone(&session, worktree.path()).await;
        let old_base = server
            .state
            .default_ref()
            .unwrap()
            .project_cache
            .read()
            .await
            .clone()
            .unwrap();

        std::fs::write(
            primary.path().join("src/area_1/file_5.rs"),
            "pub fn primarynew() {}\n",
        )
        .unwrap();
        std::fs::write(
            primary.path().join("src/primary_added.rs"),
            "pub fn primaryadded() {}\n",
        )
        .unwrap();
        std::fs::remove_file(primary.path().join("src/area_2/file_6.rs")).unwrap();
        server.invalidate_project_cache().await;
        assert!(
            lexdelta_keywords(&server, "primarynew")
                .await
                .contains("file_5.rs")
        );

        lexdelta_assert_matches_standalone(&session, worktree.path()).await;
        for query in ["primarynew", "primaryadded"] {
            assert!(
                lexdelta_keywords(&session, query)
                    .await
                    .contains("No lexical matches"),
                "the worktree answered {query} from the primary's new files"
            );
        }
        assert!(
            lexdelta_keywords(&session, &lexdelta_tag(5))
                .await
                .contains("src/area_1/file_5.rs")
        );
        assert!(
            lexdelta_keywords(&session, &lexdelta_tag(6))
                .await
                .contains("src/area_2/file_6.rs")
        );
        let primary_cache = server
            .state
            .default_ref()
            .unwrap()
            .project_cache
            .read()
            .await
            .clone()
            .unwrap();
        assert!(!Arc::ptr_eq(&old_base, &primary_cache));
        let cache = session
            .current_ref()
            .await
            .project_cache
            .read()
            .await
            .clone()
            .unwrap();
        assert!(
            cache
                .file_content
                .base()
                .is_some_and(|base| Arc::ptr_eq(base, primary_cache.file_content.own())),
            "the worktree did not rebase onto the primary's new cache"
        );
    }

    #[tokio::test]
    async fn lexdelta_small_delta_worktree_resident_estimate_excludes_the_shared_base() {
        let primary = tempfile::tempdir().unwrap();
        lexdelta_corpus(primary.path(), 200);
        let worktree = tempfile::tempdir().unwrap();
        lexdelta_corpus(worktree.path(), 200);
        lexdelta_edit_worktree(worktree.path());
        let server = ContextPlusServer::new(primary.path().to_path_buf(), lexdelta_config());
        let session = lexdelta_attach(&server, worktree.path()).await;
        lexdelta_keywords(&server, "shared symbol").await;
        lexdelta_keywords(&session, "shared symbol").await;

        let primary_components = ResidentSnapshot::capture(&server.state.default_ref().unwrap())
            .await
            .measure();
        let worktree_components = ResidentSnapshot::capture(&*session.current_ref().await)
            .await
            .measure();
        let primary_bytes: usize = primary_components.iter().map(|(_, bytes, _)| bytes).sum();
        let unique_bytes: usize = worktree_components
            .iter()
            .filter(|(ptr, _, _)| {
                !primary_components
                    .iter()
                    .any(|(shared, _, _)| shared == ptr)
            })
            .map(|(_, bytes, _)| bytes)
            .sum();
        assert!(primary_bytes > 0);
        assert!(
            unique_bytes * 20 < primary_bytes,
            "a two-file delta holds {unique_bytes} unique bytes against a {primary_bytes}-byte base"
        );
    }

    fn lexdelta_git_primary(files: usize) -> (tempfile::TempDir, tempfile::TempDir, PathBuf) {
        let primary = tempfile::tempdir().unwrap();
        lexdelta_git(primary.path(), &["init", "-q", "-b", "main"]);
        lexdelta_corpus(primary.path(), files);
        lexdelta_git(primary.path(), &["add", "-A"]);
        lexdelta_git(primary.path(), &["commit", "-qm", "base"]);
        let holder = tempfile::tempdir().unwrap();
        let worktree = holder.path().join("feature");
        (primary, holder, worktree)
    }

    fn lexdelta_add_worktree(primary: &std::path::Path, worktree: &std::path::Path, start: &str) {
        lexdelta_git(
            primary,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "feature",
                worktree.to_str().unwrap(),
                start,
            ],
        );
    }

    #[test]
    fn lexdelta_worktree_reads_a_same_blob_file_whose_size_differs_from_the_base() {
        let (primary, _holder, worktree) = lexdelta_git_primary(10);
        lexdelta_add_worktree(primary.path(), &worktree, "main");
        let config = lexdelta_config();
        let path = "src/area_0/file_0.rs";
        let real = load_project_cache(primary.path(), &config, None, true);
        assert!(
            real.clean_blobs
                .as_ref()
                .is_some_and(|blobs| blobs.contains_key(path))
        );
        let mut smudged: crate::core::walker::ContentMap = real
            .file_content
            .iter()
            .map(|(path, content)| (path.clone(), Arc::clone(content)))
            .collect();
        smudged.insert(path.to_string(), Arc::new("smudged\r\n".to_string()));
        let base = Arc::new(ProjectCache {
            file_entries: real.file_entries.clone(),
            file_content: smudged.into(),
            clean_blobs: real.clean_blobs.clone(),
            last_refresh: Instant::now(),
        });

        let cache = load_project_cache(&worktree, &config, Some(base), false);
        assert!(cache.file_content.base().is_some());
        assert_eq!(
            cache.file_content[path].as_str(),
            std::fs::read_to_string(worktree.join(path)).unwrap()
        );
    }

    #[tokio::test]
    async fn lexdelta_worktree_reads_a_clean_file_the_primary_walk_skipped() {
        let (primary, _holder, worktree) = lexdelta_git_primary(40);
        std::fs::write(primary.path().join(".ignore"), "src/area_0/file_0.rs\n").unwrap();
        lexdelta_add_worktree(primary.path(), &worktree, "main");
        let server = ContextPlusServer::new(primary.path().to_path_buf(), lexdelta_config());
        let session = lexdelta_attach(&server, &worktree).await;

        let found = lexdelta_keywords(&session, &lexdelta_tag(0)).await;
        assert!(
            found.contains("src/area_0/file_0.rs"),
            "the worktree lost a file the primary's walk skipped: {found}"
        );
        lexdelta_assert_matches_standalone(&session, &worktree).await;
    }

    #[tokio::test]
    async fn lexdelta_worktree_ignores_a_primary_file_reverted_after_its_cache_was_built() {
        let (primary, _holder, worktree) = lexdelta_git_primary(40);
        lexdelta_add_worktree(primary.path(), &worktree, "main");
        std::fs::write(
            primary.path().join("src/area_1/file_5.rs"),
            "pub fn primarydirty() {}\n",
        )
        .unwrap();
        let server = ContextPlusServer::new(primary.path().to_path_buf(), lexdelta_config());
        assert!(
            lexdelta_keywords(&server, "primarydirty")
                .await
                .contains("file_5.rs")
        );
        lexdelta_git(primary.path(), &["checkout", "--", "src/area_1/file_5.rs"]);

        let session = lexdelta_attach(&server, &worktree).await;
        let stale = lexdelta_keywords(&session, "primarydirty").await;
        assert!(
            stale.contains("No lexical matches"),
            "the worktree served the primary's reverted content: {stale}"
        );
        assert!(
            lexdelta_keywords(&session, &lexdelta_tag(5))
                .await
                .contains("src/area_1/file_5.rs")
        );
        lexdelta_assert_matches_standalone(&session, &worktree).await;
    }

    #[tokio::test]
    async fn lexdelta_worktree_ignores_the_primary_branch_its_cache_was_built_on() {
        let (primary, _holder, worktree) = lexdelta_git_primary(40);
        lexdelta_git(primary.path(), &["checkout", "-q", "-b", "other"]);
        std::fs::write(
            primary.path().join("src/area_3/file_7.rs"),
            "pub fn otherbranch() {}\n",
        )
        .unwrap();
        lexdelta_git(primary.path(), &["commit", "-qam", "other"]);
        let server = ContextPlusServer::new(primary.path().to_path_buf(), lexdelta_config());
        assert!(
            lexdelta_keywords(&server, "otherbranch")
                .await
                .contains("file_7.rs")
        );
        lexdelta_add_worktree(primary.path(), &worktree, "main");
        lexdelta_git(primary.path(), &["checkout", "-q", "main"]);

        let session = lexdelta_attach(&server, &worktree).await;
        let stale = lexdelta_keywords(&session, "otherbranch").await;
        assert!(
            stale.contains("No lexical matches"),
            "the worktree served the primary's previous branch: {stale}"
        );
        assert!(
            lexdelta_keywords(&session, &lexdelta_tag(7))
                .await
                .contains("src/area_3/file_7.rs")
        );
        lexdelta_assert_matches_standalone(&session, &worktree).await;
    }

    #[tokio::test]
    async fn lexdelta_promoted_worktree_query_leaves_the_primary_cache_alone() {
        let primary = tempfile::tempdir().unwrap();
        lexdelta_corpus(primary.path(), 40);
        let worktree = tempfile::tempdir().unwrap();
        lexdelta_corpus(worktree.path(), 40);
        for i in 0..12 {
            std::fs::write(
                worktree
                    .path()
                    .join(format!("src/area_{}/file_{i}.rs", i % 4)),
                format!("pub fn rewritten_{i}() {{}}\n"),
            )
            .unwrap();
        }
        let server = ContextPlusServer::new(primary.path().to_path_buf(), lexdelta_config());
        let session = lexdelta_attach(&server, worktree.path()).await;
        lexdelta_keywords(&session, "shared symbol").await;
        let owner = session.current_ref().await;
        let cache = owner.project_cache.read().await.clone().unwrap();
        assert!(cache.file_content.base().is_none());

        let primary_ref = server.state.default_ref().unwrap();
        let mut primary_slot = primary_ref.project_cache.write().await;
        *primary_slot = None;
        let answer = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            lexdelta_keywords(&session, "rewritten_3"),
        )
        .await
        .expect("a promoted worktree's query waited on the primary's cache lock");
        assert!(answer.contains("src/area_3/file_3.rs"));
        drop(primary_slot);
        assert!(
            primary_ref.project_cache.read().await.is_none(),
            "a promoted worktree's query rebuilt the primary's cache"
        );
        assert!(Arc::ptr_eq(
            &cache,
            &owner.project_cache.read().await.clone().unwrap()
        ));
    }

    #[tokio::test]
    async fn lexdelta_resident_estimate_counts_an_old_primary_cache_pinned_by_the_keyword_entry() {
        let primary = tempfile::tempdir().unwrap();
        lexdelta_corpus(primary.path(), 40);
        let worktree = tempfile::tempdir().unwrap();
        lexdelta_corpus(worktree.path(), 40);
        lexdelta_edit_worktree(worktree.path());
        let server = ContextPlusServer::new(primary.path().to_path_buf(), lexdelta_config());
        let session = lexdelta_attach(&server, worktree.path()).await;
        lexdelta_keywords(&session, "shared symbol").await;
        let primary_ref = server.state.default_ref().unwrap();
        let old_base = primary_ref.project_cache.read().await.clone().unwrap();

        std::fs::write(
            primary.path().join("src/area_1/file_5.rs"),
            "pub fn primarynew() {}\n",
        )
        .unwrap();
        server.invalidate_project_cache().await;
        lexdelta_keywords(&server, "primarynew").await;
        let owner = session.current_ref().await;
        let rebased = session.ensure_project_cache_for(&owner).await.unwrap();
        let new_base = primary_ref.project_cache.read().await.clone().unwrap();
        assert!(!Arc::ptr_eq(&old_base, &new_base));
        assert!(
            rebased
                .file_content
                .base()
                .is_some_and(|base| Arc::ptr_eq(base, new_base.file_content.own()))
        );

        let old_contents = Arc::as_ptr(old_base.file_content.own()) as usize;
        drop(old_base);
        let components = ResidentSnapshot::capture(&owner).await.measure();
        assert!(
            components
                .iter()
                .any(|(ptr, bytes, _)| *ptr == old_contents && *bytes > 0),
            "the old primary content map pinned by the worktree's keyword entry is not measured"
        );
    }

    /// Above `ANN_THRESHOLD`, so the primary's index holds a vector store.
    const SEMANTIC_FORK_FILES: usize = 2_100;

    async fn semantic_fork_servers(
        edit: fn(&std::path::Path),
    ) -> (
        wiremock::MockServer,
        tempfile::TempDir,
        tempfile::TempDir,
        ContextPlusServer,
        ContextPlusServer,
    ) {
        let ollama = wiremock::MockServer::start().await;
        let primary = tempfile::tempdir().unwrap();
        let worktree = tempfile::tempdir().unwrap();
        lexdelta_corpus(primary.path(), SEMANTIC_FORK_FILES);
        lexdelta_corpus(worktree.path(), SEMANTIC_FORK_FILES);
        edit(worktree.path());
        let server = identifier_test_server(&ollama, primary.path()).await;
        let session = attached_worktree(&server, worktree.path()).await;
        (ollama, primary, worktree, server, session)
    }

    /// A scoped query takes the exact scan, so it starts no HNSW graph build
    /// that could take another test's `hnsw_test_seam` pause.
    async fn semantic_fork_query(server: &ContextPlusServer) -> String {
        let mut args = semantic_args("shared symbol");
        args.insert("scope".into(), json!("code"));
        text_of(&server.handle_semantic_code_search(args).await.unwrap())
    }

    async fn semantic_fork_index(
        server: &ContextPlusServer,
    ) -> Arc<crate::tools::semantic_search::CachedSearchIndex> {
        server
            .current_ref()
            .await
            .search_index_cache
            .read()
            .await
            .clone()
            .expect("a semantic index")
    }

    /// The only test here whose queries build a graph. `hnsw_test_seam` pauses
    /// are process-wide and taken by any test's build, so this reads graph
    /// state instead.
    #[tokio::test]
    async fn semantic_fork_worktree_shares_the_primary_vector_store_and_graph() {
        let (_ollama, _primary, worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        server
            .handle_semantic_code_search(semantic_args("shared symbol"))
            .await
            .unwrap();
        let primary = semantic_fork_index(&server).await;
        tokio::time::timeout(std::time::Duration::from_secs(60), async {
            while !primary.index.graph_is_built() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the primary's query builds its graph");

        let result = text_of(
            &session
                .handle_semantic_code_search(semantic_args("shared symbol"))
                .await
                .unwrap(),
        );
        let fork = semantic_fork_index(&session).await;
        assert!(
            fork.index.graph_is_built(),
            "the worktree's first query searched a graph it had yet to build"
        );
        assert!(
            !Arc::ptr_eq(&primary, &fork),
            "the worktree holds the primary's entry"
        );
        assert!(
            fork.index.shares_vector_store(&primary.index),
            "the worktree built its own vector store"
        );
        assert_eq!(fork.search_root(), worktree.path().canonicalize().unwrap());
        assert_eq!(fork.index.document_count(), SEMANTIC_FORK_FILES);
        assert!(
            !fork
                .index
                .documents()
                .iter()
                .any(|doc| doc.path == "src/area_3/file_3.rs"),
            "the worktree's deleted file is still indexed"
        );
        assert!(result.contains("1. src/"), "{result}");
    }

    fn semantic_fork_walker(
        server: &ContextPlusServer,
        ref_index: Arc<crate::ref_index::RefIndex>,
    ) -> crate::server_adapters::RefWalkerIndexer {
        crate::server_adapters::RefWalkerIndexer {
            ref_index,
            walker: CachedWalkerIndexer {
                config: server.state.config.clone(),
                ollama: server.state.ollama.clone(),
                state: server.state.clone(),
            },
        }
    }

    /// The query result of a server over `root` alone.
    async fn semantic_fork_standalone(
        server: &ContextPlusServer,
        root: &std::path::Path,
    ) -> String {
        let standalone = ContextPlusServer::new(root.to_path_buf(), server.state.config.clone());
        semantic_fork_query(&standalone).await
    }

    fn semantic_fork_rewrite_a_third(root: &std::path::Path) {
        for i in (0..SEMANTIC_FORK_FILES).step_by(3) {
            std::fs::write(
                root.join(format!("src/area_{}/file_{i}.rs", i % 4)),
                format!("pub fn rewritten_{i}() {{}}\n"),
            )
            .unwrap();
        }
    }

    /// After a restart the primary holds only its warmup index, walked from no
    /// root. The worktree's first query builds the primary's index once,
    /// installs it in the primary's slot and forks it.
    #[tokio::test]
    async fn semantic_fork_restarted_worktree_builds_the_primary_index_once_and_forks_it() {
        use crate::tools::semantic_search::{CachedSearchIndex, IndexFingerprint, SearchIndex};

        let (_ollama, primary_root, _worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        let primary = server.state.default_ref().unwrap();
        *primary.search_index_cache.write().await = Some(Arc::new(CachedSearchIndex::new(
            SearchIndex::new(),
            IndexFingerprint::from_docs(&[]),
            0,
        )));

        semantic_fork_query(&session).await;
        let primary_index = semantic_fork_index(&server).await;
        assert_eq!(
            primary_index.search_root(),
            primary_root.path().canonicalize().unwrap(),
            "the primary's index of its whole root is not installed"
        );
        assert!(
            semantic_fork_index(&session)
                .await
                .index
                .shares_vector_store(&primary_index.index),
            "the worktree did not fork the primary's index"
        );
        semantic_fork_query(&server).await;
        assert_eq!(
            primary
                .semantic_walks
                .load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the primary's index was built more than once"
        );
    }

    /// A walk over a worktree whose fork already shares the primary's store
    /// keeps its entry, and with it the batches queued there.
    #[tokio::test]
    async fn semantic_fork_walk_keeps_a_current_fork_and_its_queued_batches() {
        use crate::tools::semantic_search::{CachedSearchIndex, SearchDocument};

        let (_ollama, _primary, worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        semantic_fork_query(&server).await;
        semantic_fork_query(&session).await;
        let mut fork = semantic_fork_index(&session).await;
        fork.rebuild_in_progress
            .store(true, std::sync::atomic::Ordering::Release);
        assert!(CachedSearchIndex::refresh_paths(
            &mut fork,
            &worktree.path().canonicalize().unwrap(),
            vec![SearchDocument::new(
                "src/queued.rs".into(),
                String::new(),
                vec![],
                vec![],
                "queued".into(),
            )],
            vec![None],
            &[],
            1,
        ));
        fork.rebuild_in_progress
            .store(false, std::sync::atomic::Ordering::Release);
        assert!(Arc::ptr_eq(&fork, &semantic_fork_index(&session).await));

        semantic_fork_walker(&session, session.current_ref().await)
            .walk_and_index(worktree.path())
            .await
            .unwrap();
        assert!(
            Arc::ptr_eq(&fork, &semantic_fork_index(&session).await),
            "the walk replaced a fork of the primary's current store"
        );
    }

    #[tokio::test]
    async fn semantic_fork_is_dropped_when_the_slot_changed_during_its_walk() {
        use crate::tools::semantic_search::{CachedSearchIndex, IndexFingerprint, SearchIndex};

        let (_ollama, _primary, worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        semantic_fork_query(&server).await;
        let owner = session.current_ref().await;
        let root = worktree.path().to_path_buf();
        let pause = crate::server_adapters::test_seams::pause_after_cache_snapshot(&root);
        let walker = semantic_fork_walker(&session, Arc::clone(&owner));
        let walk = tokio::spawn(async move { walker.walk_and_index(&root).await.map(|_| ()) });
        pause.wait_until_entered().await;
        let racing = Arc::new(CachedSearchIndex::new(
            SearchIndex::new(),
            IndexFingerprint::from_docs(&[]),
            0,
        ));
        *owner.search_index_cache.write().await = Some(Arc::clone(&racing));
        pause.resume();
        walk.await.unwrap().unwrap();

        assert!(
            Arc::ptr_eq(&racing, &semantic_fork_index(&session).await),
            "a fork replaced the entry installed during its walk"
        );
    }

    #[tokio::test]
    async fn semantic_fork_skips_a_primary_with_queued_batches() {
        use crate::tools::semantic_search::{CachedSearchIndex, SearchDocument};

        let (_ollama, primary_root, worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        semantic_fork_query(&server).await;
        let primary = server.state.default_ref().unwrap();
        let mut entry = semantic_fork_index(&server).await;
        entry
            .rebuild_in_progress
            .store(true, std::sync::atomic::Ordering::Release);
        CachedSearchIndex::refresh_paths(
            &mut entry,
            &primary_root.path().canonicalize().unwrap(),
            vec![SearchDocument::new(
                "src/queued.rs".into(),
                String::new(),
                vec![],
                vec![],
                "queued".into(),
            )],
            vec![None],
            &[],
            1,
        );
        entry
            .rebuild_in_progress
            .store(false, std::sync::atomic::Ordering::Release);
        assert!(Arc::ptr_eq(
            &entry,
            primary.search_index_cache.read().await.as_ref().unwrap()
        ));

        let result = semantic_fork_query(&session).await;
        assert!(
            !semantic_fork_index(&session)
                .await
                .index
                .shares_vector_store(&entry.index),
            "the worktree forked a primary index with queued batches"
        );
        assert_eq!(
            result,
            semantic_fork_standalone(&server, worktree.path()).await
        );
    }

    #[tokio::test]
    async fn semantic_fork_skips_a_primary_mid_rebuild() {
        let (_ollama, _primary, worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        semantic_fork_query(&server).await;
        let entry = semantic_fork_index(&server).await;
        entry
            .rebuild_in_progress
            .store(true, std::sync::atomic::Ordering::Release);

        let result = semantic_fork_query(&session).await;
        entry
            .rebuild_in_progress
            .store(false, std::sync::atomic::Ordering::Release);
        assert!(
            !semantic_fork_index(&session)
                .await
                .index
                .shares_vector_store(&entry.index),
            "the worktree forked a primary index mid-rebuild"
        );
        assert_eq!(
            result,
            semantic_fork_standalone(&server, worktree.path()).await
        );
    }

    #[tokio::test]
    async fn semantic_fork_skips_a_scoped_worktree_search() {
        let (_ollama, _primary, worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        semantic_fork_query(&server).await;
        let mut args = semantic_args("shared symbol");
        args.insert("scope".into(), json!("code"));
        args.insert("rootDir".into(), json!("src/area_1"));
        let result = text_of(&session.handle_semantic_code_search(args).await.unwrap());

        let scoped = semantic_fork_index(&session).await;
        let sub_root = worktree.path().join("src/area_1").canonicalize().unwrap();
        assert_eq!(scoped.search_root(), sub_root);
        assert!(
            !scoped
                .index
                .shares_vector_store(&semantic_fork_index(&server).await.index)
        );
        assert_eq!(result, semantic_fork_standalone(&server, &sub_root).await);
    }

    #[tokio::test]
    async fn semantic_fork_worktree_without_its_parent_builds_standalone() {
        use crate::ref_index::{RefId, RefIndex};

        let (_ollama, _primary, worktree, server, _session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        semantic_fork_query(&server).await;
        let orphan = tempfile::tempdir().unwrap();
        lexdelta_corpus(orphan.path(), SEMANTIC_FORK_FILES);
        let root = orphan.path().canonicalize().unwrap();
        let id = RefId::for_canonical_path(&root);
        let missing = RefId::for_canonical_path(&worktree.path().join("gone"));
        server
            .state
            .attach_ref(id, || {
                Arc::new(RefIndex::new(root.clone(), root.clone(), Some(missing)))
            })
            .await;
        let session = server.with_session(id);

        let result = semantic_fork_query(&session).await;
        assert!(
            !semantic_fork_index(&session)
                .await
                .index
                .shares_vector_store(&semantic_fork_index(&server).await.index)
        );
        assert_eq!(result, semantic_fork_standalone(&server, &root).await);
    }

    #[tokio::test]
    async fn semantic_fork_worktree_over_the_change_threshold_builds_standalone() {
        let (_ollama, _primary, worktree, server, session) =
            semantic_fork_servers(semantic_fork_rewrite_a_third).await;
        semantic_fork_query(&server).await;

        let result = semantic_fork_query(&session).await;
        assert!(
            !semantic_fork_index(&session)
                .await
                .index
                .shares_vector_store(&semantic_fork_index(&server).await.index),
            "a worktree past the threshold forked the primary's index"
        );
        assert_eq!(
            result,
            semantic_fork_standalone(&server, worktree.path()).await
        );
    }

    /// Replaces the primary's vector store with a fresh build, as after an eviction.
    async fn semantic_fork_rebuild_primary(
        server: &ContextPlusServer,
    ) -> Arc<crate::tools::semantic_search::CachedSearchIndex> {
        let old = semantic_fork_index(server).await;
        *server.current_ref().await.search_index_cache.write().await = None;
        semantic_fork_query(server).await;
        let rebuilt = semantic_fork_index(server).await;
        assert!(!rebuilt.index.shares_vector_store(&old.index));
        rebuilt
    }

    /// Queries the primary until a background rebuild replaces the store of `old`.
    async fn semantic_fork_await_primary_rebuild(
        server: &ContextPlusServer,
        old: &crate::tools::semantic_search::CachedSearchIndex,
    ) -> Arc<crate::tools::semantic_search::CachedSearchIndex> {
        semantic_fork_query(server).await;
        tokio::time::timeout(std::time::Duration::from_secs(120), async {
            loop {
                let current = semantic_fork_index(server).await;
                if current.index.vector_store().is_some()
                    && !current.index.shares_vector_store(&old.index)
                    && !current
                        .rebuild_in_progress
                        .load(std::sync::atomic::Ordering::Acquire)
                {
                    return current;
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the primary rebuilds its store")
    }

    /// (path, score) of every code hit, sorted by score, then path.
    fn semantic_fork_hits(
        entry: &crate::tools::semantic_search::CachedSearchIndex,
    ) -> Vec<(String, f64)> {
        let opts = crate::tools::semantic_search::ResolvedSearchOptions {
            scope: crate::tools::semantic_search::SearchScope::Code,
            top_k: 10_000,
            min_combined_score: 0.0,
            ..Default::default()
        };
        let mut hits: Vec<_> = entry
            .index
            .search("shared symbol", &[13.0, 1.0, 0.0], &opts)
            .into_iter()
            .map(|hit| (hit.path, hit.score))
            .collect();
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        hits
    }

    /// [`semantic_fork_hits`] of a server over `root` alone.
    async fn semantic_fork_standalone_hits(
        server: &ContextPlusServer,
        root: &std::path::Path,
    ) -> Vec<(String, f64)> {
        let standalone = ContextPlusServer::new(root.to_path_buf(), server.state.config.clone());
        semantic_fork_query(&standalone).await;
        let entry = semantic_fork_index(&standalone).await;
        semantic_fork_hits(&entry)
    }

    #[tokio::test]
    async fn semantic_fork_primary_incremental_edits_leave_the_fork_in_place() {
        let (_ollama, primary_root, worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        semantic_fork_query(&server).await;
        semantic_fork_query(&session).await;
        let fork = semantic_fork_index(&session).await;
        std::fs::write(
            primary_root.path().join("src/area_1/file_5.rs"),
            "pub fn primaryedited() {}\n",
        )
        .unwrap();
        std::fs::write(
            primary_root.path().join("src/area_2/primary_added.rs"),
            "pub fn primaryadded() {}\n",
        )
        .unwrap();
        std::fs::remove_file(primary_root.path().join("src/area_3/file_7.rs")).unwrap();

        semantic_fork_query(&server).await;
        let primary = semantic_fork_index(&server).await;
        assert!(primary.index.document_count() > 0);
        assert!(
            primary.index.shares_vector_store(&fork.index),
            "the primary's incremental edits replaced its store"
        );
        semantic_fork_query(&session).await;
        assert!(
            Arc::ptr_eq(&fork, &semantic_fork_index(&session).await),
            "the primary's incremental edits re-forked the worktree"
        );
        assert_eq!(
            semantic_fork_hits(&fork),
            semantic_fork_standalone_hits(&server, worktree.path()).await
        );
    }

    /// An idle worktree moves onto the primary's new store at its first query
    /// after the primary replaces it, and walks no more after that.
    #[tokio::test]
    async fn semantic_fork_idle_worktree_reforks_onto_a_replaced_primary_store() {
        let (_ollama, _primary, worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        semantic_fork_query(&server).await;
        semantic_fork_query(&session).await;
        semantic_fork_query(&session).await;
        let primary = semantic_fork_rebuild_primary(&server).await;

        semantic_fork_query(&session).await;
        let fork = semantic_fork_index(&session).await;
        assert!(
            fork.index.shares_vector_store(&primary.index),
            "the idle worktree still searches the primary's old store"
        );
        assert_eq!(
            semantic_fork_hits(&fork),
            semantic_fork_standalone_hits(&server, worktree.path()).await
        );
        let owner = session.current_ref().await;
        let walks = owner
            .semantic_walks
            .load(std::sync::atomic::Ordering::Relaxed);
        semantic_fork_query(&session).await;
        assert_eq!(
            owner
                .semantic_walks
                .load(std::sync::atomic::Ordering::Relaxed),
            walks,
            "the re-forked worktree walked again"
        );
    }

    /// The budget's pointer-keyed components of the refs behind `servers`.
    async fn semantic_fork_resident(servers: &[&ContextPlusServer]) -> HashMap<usize, usize> {
        let mut holders = HashMap::new();
        for server in servers {
            let owner = server.current_ref().await;
            for (ptr, bytes, _) in ResidentSnapshot::capture(&owner).await.measure() {
                holders.insert(ptr, bytes);
            }
        }
        holders
    }

    #[tokio::test]
    async fn semantic_fork_resident_estimate_counts_a_store_shared_by_two_forks_once() {
        let (_ollama, _primary, _worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        let second = tempfile::tempdir().unwrap();
        lexdelta_corpus(second.path(), SEMANTIC_FORK_FILES);
        lexdelta_edit_worktree(second.path());
        let sibling = attached_worktree(&server, second.path()).await;
        semantic_fork_query(&server).await;
        semantic_fork_query(&session).await;
        semantic_fork_query(&sibling).await;
        let entries = [
            semantic_fork_index(&server).await,
            semantic_fork_index(&session).await,
            semantic_fork_index(&sibling).await,
        ];
        let store = Arc::clone(entries[0].index.vector_store().unwrap());
        assert!(
            entries
                .iter()
                .all(|entry| entry.index.shares_vector_store(&entries[0].index))
        );

        let holders = semantic_fork_resident(&[&server, &session, &sibling]).await;
        let charged: usize = entries
            .iter()
            .map(|entry| Arc::as_ptr(entry) as usize)
            .chain([Arc::as_ptr(&store) as usize])
            .filter_map(|ptr| holders.get(&ptr))
            .sum();
        let store_bytes = store.estimated_resident_bytes();
        assert!(store_bytes > 0);
        let own: usize = entries.iter().map(|entry| entry.own_resident_bytes()).sum();
        assert_eq!(
            charged,
            own + store_bytes,
            "the store shared by the primary and two forks is not charged once"
        );
    }

    #[tokio::test]
    async fn semantic_fork_resident_estimate_counts_an_old_store_pinned_by_a_fork() {
        let (_ollama, _primary, _worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        semantic_fork_query(&server).await;
        semantic_fork_query(&session).await;
        let (old_store, old_bytes) = {
            let store = Arc::clone(
                semantic_fork_index(&server)
                    .await
                    .index
                    .vector_store()
                    .unwrap(),
            );
            (
                Arc::as_ptr(&store) as usize,
                store.estimated_resident_bytes(),
            )
        };
        semantic_fork_rebuild_primary(&server).await;

        assert!(
            !semantic_fork_resident(&[&server])
                .await
                .contains_key(&old_store)
        );
        assert_eq!(
            semantic_fork_resident(&[&session]).await.get(&old_store),
            Some(&old_bytes),
            "the old primary store pinned by the worktree's fork is not charged to it"
        );
    }

    /// With the tracker on, the worktree's first query after the primary
    /// replaces its store answers from the old fork while a background refresh
    /// re-forks onto the new one.
    #[tokio::test]
    async fn semantic_fork_tracked_worktree_reforks_in_the_background() {
        let (_ollama, primary_root, worktree, untracked, _) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        let mut config = untracked.state.config.clone();
        config.embed_tracker_mode = crate::config::TrackerMode::Lazy;
        let server = ContextPlusServer::new(primary_root.path().to_path_buf(), config);
        let session = attached_worktree(&server, worktree.path()).await;
        semantic_fork_query(&server).await;
        semantic_fork_query(&session).await;
        semantic_fork_query(&session).await;
        let primary = semantic_fork_rebuild_primary(&server).await;

        semantic_fork_query(&session).await;
        let fork = tokio::time::timeout(std::time::Duration::from_secs(60), async {
            loop {
                let fork = semantic_fork_index(&session).await;
                if fork.index.shares_vector_store(&primary.index) {
                    return fork;
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the worktree re-forks onto the primary's new store");
        assert_eq!(
            semantic_fork_hits(&fork),
            semantic_fork_standalone_hits(&server, worktree.path()).await
        );
    }

    /// A worktree past the change threshold re-checks the fork once per new
    /// primary store, not on every query.
    #[tokio::test]
    async fn semantic_fork_promoted_worktree_walks_once_per_primary_store() {
        let (_ollama, _primary, _worktree, server, session) =
            semantic_fork_servers(semantic_fork_rewrite_a_third).await;
        semantic_fork_query(&server).await;
        semantic_fork_query(&session).await;
        semantic_fork_query(&session).await;
        let owner = session.current_ref().await;
        let walks = || {
            owner
                .semantic_walks
                .load(std::sync::atomic::Ordering::Relaxed)
        };
        let before = walks();
        semantic_fork_query(&session).await;
        assert_eq!(walks(), before, "an idle standalone worktree walked");

        let primary = semantic_fork_rebuild_primary(&server).await;
        for _ in 0..3 {
            semantic_fork_query(&session).await;
        }
        assert_eq!(
            walks(),
            before + 1,
            "the worktree did not re-check the new store exactly once"
        );
        assert!(
            !semantic_fork_index(&session)
                .await
                .index
                .shares_vector_store(&primary.index)
        );
    }

    /// A primary full rebuild leaves the worktree's old fork answering like the
    /// re-fork its next query makes over the new store.
    #[tokio::test]
    async fn semantic_fork_worktree_stays_correct_across_a_primary_full_rebuild() {
        let (_ollama, primary_root, worktree, server, session) =
            semantic_fork_servers(lexdelta_edit_worktree).await;
        semantic_fork_query(&server).await;
        semantic_fork_query(&session).await;
        semantic_fork_query(&session).await;
        let old_fork = semantic_fork_index(&session).await;
        // Past the change threshold and back: two full rebuilds.
        semantic_fork_rewrite_a_third(primary_root.path());
        let original = semantic_fork_index(&server).await;
        let rewritten = semantic_fork_await_primary_rebuild(&server, &original).await;
        lexdelta_corpus(primary_root.path(), SEMANTIC_FORK_FILES);
        let rebuilt = semantic_fork_await_primary_rebuild(&server, &rewritten).await;
        assert!(!rebuilt.index.shares_vector_store(&old_fork.index));

        let expected = semantic_fork_standalone_hits(&server, worktree.path()).await;
        assert_eq!(semantic_fork_hits(&old_fork), expected);

        semantic_fork_query(&session).await;
        let fork = semantic_fork_index(&session).await;
        assert!(
            fork.index.shares_vector_store(&rebuilt.index),
            "the worktree did not re-fork onto the rebuilt store"
        );
        assert_eq!(semantic_fork_hits(&fork), expected);
    }
}

#[cfg(test)]
mod cold_start_tests {
    use super::*;
    use serde_json::json;

    const KINDS: [&str; 3] = [
        snapshots::KEYWORDS,
        snapshots::IDENTIFIERS,
        snapshots::FILES,
    ];

    /// A small corpus of code and docs with overlapping vocabulary.
    fn corpus() -> Vec<(String, String)> {
        let topics = [
            "refund", "invoice", "payment", "ledger", "account", "session", "token", "order",
            "shipment", "report",
        ];
        let mut files = Vec::new();
        for (i, topic) in topics.iter().enumerate() {
            let other = topics[(i + 3) % topics.len()];
            files.push((
                format!("src/{topic}.rs"),
                format!(
                    "//! Handles the {topic} lifecycle and its {other} links.\n\
                     pub struct {Topic}Store {{ items: Vec<u32> }}\n\
                     pub fn record_{topic}(id: u32) -> u32 {{ id + {i} }}\n\
                     pub fn load_{other}_for_{topic}(id: u32) -> u32 {{ record_{topic}(id) }}\n\
                     impl {Topic}Store {{\n    pub fn total_{topic}s(&self) -> usize {{ self.items.len() }}\n}}\n",
                    Topic = topic[..1].to_uppercase() + &topic[1..],
                ),
            ));
            files.push((
                format!("web/{topic}.ts"),
                format!(
                    "// {topic} client for the {other} screen\n\
                     export function fetch{Topic}(id: string): Promise<string> {{ return load{Topic}(id); }}\n\
                     export const {topic}Limit = {i};\n\
                     export class {Topic}View {{ render(): string {{ return '{other}'; }} }}\n",
                    Topic = topic[..1].to_uppercase() + &topic[1..],
                ),
            ));
            files.push((
                format!("docs/{topic}.md"),
                format!(
                    "# {topic}\n\nHow a {topic} moves through the {other} pipeline, \
                     including {topic} retries.\n"
                ),
            ));
        }
        files
    }

    /// An embedding service whose vector for a text is a normalized bag of
    /// its letters, so distinct texts rank differently.
    async fn embedder() -> wiremock::MockServer {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(|request: &Request| {
                let inputs = request
                    .body_json::<serde_json::Value>()
                    .ok()
                    .and_then(|body| body["input"].as_array().cloned())
                    .unwrap_or_default();
                let embeddings: Vec<Vec<f32>> = inputs
                    .iter()
                    .map(|input| {
                        let mut vector = [0.01_f32; 16];
                        for byte in input.as_str().unwrap_or_default().bytes() {
                            if byte.is_ascii_alphabetic() {
                                vector[(byte.to_ascii_lowercase() - b'a') as usize % 16] += 1.0;
                            }
                        }
                        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                        vector.iter().map(|v| v / norm).collect()
                    })
                    .collect();
                ResponseTemplate::new(200).set_body_json(json!({ "embeddings": embeddings }))
            })
            .mount(&server)
            .await;
        server
    }

    fn write_files(root: &std::path::Path, files: &[(String, String)]) {
        for (path, content) in files {
            let full = root.join(path);
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(full, content).unwrap();
        }
    }

    fn server(
        root: &std::path::Path,
        embedder: &wiremock::MockServer,
        snapshots: bool,
    ) -> ContextPlusServer {
        let mut config = Config::from_env();
        config.ollama_host = embedder.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        config.embed_budget_ms = 60_000;
        config.snapshots = snapshots;
        ContextPlusServer::new(root.to_path_buf(), config)
    }

    fn text(result: &CallToolResult) -> String {
        match &result.content[0].raw {
            RawContent::Text(t) => t.text.clone(),
            _ => panic!("expected text content"),
        }
    }

    /// One answer per mode.
    async fn answers(server: &ContextPlusServer) -> Vec<String> {
        let queries = [
            json!({"query": "refund payment lifecycle", "top_k": 10}),
            json!({"query": "record_refund invoice", "match": "keywords", "top_k": 10}),
            json!({"query": "load the ledger account", "kind": "identifiers", "top_k": 10}),
            json!({"query": "fetchOrder", "kind": "identifiers", "match": "keywords", "top_k": 10}),
        ];
        let mut out = Vec::new();
        for query in queries {
            let result = server
                .dispatch("explore", query.as_object().unwrap().clone())
                .await;
            assert_eq!(result.is_error, Some(false), "{}", text(&result));
            out.push(text(&result));
        }
        out
    }

    /// Writes every snapshot of `server` now, and drops its pending writes.
    async fn flush(server: &ContextPlusServer) {
        let primary = server.state.default_ref().unwrap();
        for schedule in [
            &primary.snapshots.keywords,
            &primary.snapshots.identifiers,
            &primary.snapshots.files,
        ] {
            schedule.cancel();
        }
        let config = server.state.config.clone();
        let root = primary.root_dir.clone();
        let keywords = primary.lexical_search_cache.read().await.clone().unwrap();
        let identifiers = primary.identifier_index.read().await.clone().unwrap();
        let source = primary.identifier_source.read().await.clone().unwrap();
        let files = primary.search_index_cache.read().await.clone().unwrap();
        tokio::task::spawn_blocking(move || {
            assert!(snapshots::write_keywords(&root, &config, &keywords).unwrap());
            assert!(snapshots::write_identifiers(&root, &config, &identifiers, &source).unwrap());
            assert!(snapshots::write_files(&root, &config, &files).unwrap());
        })
        .await
        .unwrap();
    }

    fn loads(root: &std::path::Path) -> Vec<usize> {
        KINDS
            .iter()
            .map(|kind| snapshots::test_seams::load_count(root, kind))
            .collect()
    }

    #[tokio::test]
    async fn cold_start_snapshot_round_trip_answers_like_a_fresh_build() {
        let tmp = tempfile::tempdir().unwrap();
        write_files(tmp.path(), &corpus());
        let embedder = embedder().await;
        let first = server(tmp.path(), &embedder, true);
        let fresh = answers(&first).await;
        flush(&first).await;
        drop(first);

        let restarted = server(tmp.path(), &embedder, true);
        let before = loads(tmp.path());
        let loaded = answers(&restarted).await;

        let after = loads(tmp.path());
        for (kind, (before, after)) in KINDS.iter().zip(before.iter().zip(&after)) {
            assert!(after > before, "the {kind} snapshot was not used");
        }
        for (fresh, loaded) in fresh.iter().zip(&loaded) {
            assert_eq!(fresh, loaded);
        }
    }

    #[tokio::test]
    async fn cold_start_changes_since_the_snapshot_answer_like_a_full_build() {
        let tmp = tempfile::tempdir().unwrap();
        let mut files = corpus();
        write_files(tmp.path(), &files);
        let embedder = embedder().await;
        let first = server(tmp.path(), &embedder, true);
        answers(&first).await;
        flush(&first).await;
        drop(first);

        // One file changed, one added, one deleted since the snapshot.
        files[0]
            .1
            .push_str("pub fn refund_reversal_window() -> u32 { 30 }\n");
        files.push((
            "src/chargeback.rs".into(),
            "//! Chargeback disputes against a refund.\npub fn open_chargeback_for_refund(id: u32) -> u32 { id }\n".into(),
        ));
        write_files(tmp.path(), &files);
        std::fs::remove_file(tmp.path().join("web/invoice.ts")).unwrap();

        let before = loads(tmp.path());
        let restarted = server(tmp.path(), &embedder, true);
        let from_snapshot = answers(&restarted).await;
        let after = loads(tmp.path());
        drop(restarted);
        for (kind, (before, after)) in KINDS.iter().zip(before.iter().zip(&after)) {
            assert!(after > before, "the {kind} snapshot was not used");
        }

        let rebuilt = server(tmp.path(), &embedder, false);
        let full = answers(&rebuilt).await;
        for (full, from_snapshot) in full.iter().zip(&from_snapshot) {
            assert_eq!(full, from_snapshot);
        }
        let joined = from_snapshot.join("\n");
        assert!(
            joined.contains("chargeback"),
            "added file missing:\n{joined}"
        );
        assert!(
            !joined.contains("web/invoice.ts"),
            "deleted file served:\n{joined}"
        );
    }

    #[tokio::test]
    async fn cold_start_corrupt_snapshots_fall_back_to_a_full_build() {
        let tmp = tempfile::tempdir().unwrap();
        write_files(tmp.path(), &corpus());
        let embedder = embedder().await;
        let first = server(tmp.path(), &embedder, true);
        let fresh = answers(&first).await;
        flush(&first).await;
        drop(first);
        for kind in KINDS {
            let path = crate::cache::snapshot::snapshot_path(tmp.path(), kind);
            let mut bytes = std::fs::read(&path).unwrap();
            let middle = bytes.len() / 2;
            bytes[middle] ^= 0x5a;
            std::fs::write(&path, bytes).unwrap();
        }

        let before = loads(tmp.path());
        let restarted = server(tmp.path(), &embedder, true);
        let rebuilt = answers(&restarted).await;
        assert_eq!(loads(tmp.path()), before, "a corrupt snapshot was used");
        assert_eq!(fresh, rebuilt);
    }

    #[tokio::test]
    async fn cold_start_snapshots_from_another_embedding_config_are_not_used() {
        let tmp = tempfile::tempdir().unwrap();
        write_files(tmp.path(), &corpus());
        let embedder = embedder().await;
        let first = server(tmp.path(), &embedder, true);
        answers(&first).await;
        flush(&first).await;
        drop(first);

        let mut config = Config::from_env();
        config.ollama_host = embedder.uri();
        config.embed_tracker_mode = TrackerMode::Off;
        config.ref_warmup_mode = RefWarmupMode::Off;
        config.embed_budget_ms = 60_000;
        config.ollama_embed_model = "another-model".into();
        let other = ContextPlusServer::new(tmp.path().to_path_buf(), config);
        let before = loads(tmp.path());
        answers(&other).await;
        let after = loads(tmp.path());
        assert_eq!(
            after[1], before[1],
            "identifier documents of another model were used"
        );
        assert_eq!(
            after[2], before[2],
            "file documents of another model were used"
        );
        assert!(
            after[0] > before[0],
            "the keyword index does not depend on the model"
        );
    }

    #[test]
    fn cold_start_loaded_keyword_index_holds_no_more_than_a_built_one() {
        use crate::tools::lexical_search::{LexicalFields, LexicalIndex};

        let tmp = tempfile::tempdir().unwrap();
        let files: Vec<(String, String)> = (0..40)
            .flat_map(|round| {
                corpus().into_iter().map(move |(path, content)| {
                    (format!("r{round}/{path}"), content.repeat(1 + round % 3))
                })
            })
            .collect();
        write_files(tmp.path(), &files);
        let config = Config::from_env();
        let cache = Arc::new(load_project_cache(tmp.path(), &config, None, false));
        let paths: Vec<&str> = cache
            .file_entries
            .iter()
            .filter(|entry| !entry.is_directory)
            .map(|entry| entry.relative_path.as_str())
            .collect();

        // The same build as `build_lexical_index`, on this thread so that the
        // counting allocator sees all of it.
        let ((built, built_retained), built_peak) = crate::alloc_probe::peak_bytes(|| {
            crate::alloc_probe::retained_bytes(|| {
                let mut index = LexicalIndex::with_capacity(paths.len());
                for &path in &paths {
                    let content = cache.file_content.get(path).unwrap().as_str();
                    let ext = path.rsplit('.').next().unwrap_or("");
                    let symbols: Vec<String> = parse_with_tree_sitter(content, ext)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|symbol| symbol.name)
                        .collect();
                    let header = crate::core::parser::extract_header(content);
                    index.push_document(LexicalFields {
                        path,
                        symbols: &symbols,
                        header: &header,
                        content,
                    });
                }
                index.finish_build();
                let paths: Vec<String> = paths.iter().map(|path| path.to_string()).collect();
                (index, paths)
            })
        });
        let (built_index, built_paths) = built;
        let cached = CachedLexicalIndex {
            index: built_index,
            document_paths: built_paths,
            project_cache: Arc::clone(&cache),
            generation: 0,
            base: None,
        };
        assert!(snapshots::write_keywords(tmp.path(), &config, &cached).unwrap());

        let ((loaded, loaded_retained), loaded_peak) = crate::alloc_probe::peak_bytes(|| {
            crate::alloc_probe::retained_bytes(|| {
                let (index, paths, _digests) =
                    snapshots::read_keywords(tmp.path(), &config).unwrap();
                (index, paths)
            })
        });
        assert_eq!(loaded.0.document_count(), cached.index.document_count());
        assert_eq!(
            loaded.0.search("record refund invoice", 20),
            cached.index.search("record refund invoice", 20)
        );
        assert!(
            loaded_retained <= built_retained,
            "a loaded index holds {loaded_retained} bytes, a built one {built_retained}"
        );
        assert!(
            loaded_peak <= built_peak,
            "loading peaked at {loaded_peak} bytes, building at {built_peak}"
        );
    }

    #[test]
    fn cold_start_loaded_identifier_documents_hold_no_more_than_parsed_ones() {
        let tmp = tempfile::tempdir().unwrap();
        write_files(tmp.path(), &corpus());
        let config = Config::from_env();
        let cache = Arc::new(load_project_cache(tmp.path(), &config, None, false));
        let mut paths: Vec<&String> = cache.file_content.keys().collect();
        paths.sort();

        let (parsed, parsed_retained) = crate::alloc_probe::retained_bytes(|| {
            paths
                .iter()
                .filter_map(|path| {
                    crate::tools::semantic_identifiers::identifier_docs_for_file(
                        path,
                        cache.file_content.get(path).unwrap(),
                    )
                    .map(|docs| ((*path).clone(), Arc::new(docs)))
                })
                .collect::<BTreeMap<_, _>>()
        });
        let index = IdentifierIndex {
            docs: Segmented::from_files(parsed.clone()),
            vectors: IdentifierVectorIndex::empty(),
            dims: 2,
            file_count: paths.len(),
            built_at: Instant::now(),
        };
        assert!(snapshots::write_identifiers(tmp.path(), &config, &index, &cache).unwrap());
        let digests: HashMap<&str, crate::cache::snapshot::Digest> = paths
            .iter()
            .map(|path| {
                (
                    path.as_str(),
                    crate::cache::snapshot::digest(
                        cache.file_content.get(path).unwrap().as_bytes(),
                    ),
                )
            })
            .collect();

        let ((loaded, to_parse), loaded_retained) = crate::alloc_probe::retained_bytes(|| {
            snapshots::read_identifiers(tmp.path(), &config, &digests).unwrap()
        });
        assert!(to_parse.is_empty());
        let parsed: Vec<_> = parsed.values().flat_map(|docs| docs.iter()).collect();
        assert_eq!(loaded.len(), parsed.len());
        for (loaded, parsed) in loaded.iter().zip(&parsed) {
            assert_eq!(
                (
                    &loaded.id,
                    &loaded.text,
                    &loaded.kind_lower,
                    &loaded.signature
                ),
                (
                    &parsed.id,
                    &parsed.text,
                    &parsed.kind_lower,
                    &parsed.signature
                )
            );
            assert_eq!(loaded.name_token_set, parsed.name_token_set);
            assert_eq!(loaded.signature_token_set, parsed.signature_token_set);
            assert_eq!(loaded.parent_token_set, parsed.parent_token_set);
        }
        assert!(
            loaded_retained <= parsed_retained,
            "loaded documents hold {loaded_retained} bytes, parsed ones {parsed_retained}"
        );
    }

    #[test]
    fn cold_start_identifier_vectors_keep_texts_that_mention_dotted_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let name = "identifier-embeddings-test";
        let key = "load function () src/a.ts Reads apps/web/.storybook/main.ts ".to_string();
        let mut vectors = IdentifierVectors::new();
        vectors.insert(key.clone(), Arc::from(vec![0.5_f32, 0.25]));
        rkyv_store::save_cache(tmp.path(), name, &identifier_cache_data(&vectors).unwrap())
            .unwrap();

        let loaded = load_identifier_vectors(tmp.path(), name);

        assert_eq!(
            loaded.get(&key).map(|vector| vector.to_vec()),
            Some(vec![0.5, 0.25]),
            "an identifier text is not a path; path hygiene must not drop it"
        );
    }

    #[tokio::test]
    async fn cold_start_only_a_daemon_preloads_snapshots() {
        let tmp = tempfile::tempdir().unwrap();
        write_files(tmp.path(), &corpus());
        let embedder = embedder().await;
        let first = server(tmp.path(), &embedder, true);
        answers(&first).await;
        flush(&first).await;
        drop(first);

        let private = server(tmp.path(), &embedder, true);
        assert!(
            private.spawn_warmup_task(false).is_none(),
            "a private server preloaded the keyword and identifier indexes"
        );

        let before = loads(tmp.path());
        let daemon = server(tmp.path(), &embedder, true);
        daemon
            .spawn_warmup_task(true)
            .expect("a daemon preloads the indexes whose snapshots exist")
            .await
            .unwrap();
        let after = loads(tmp.path());
        assert!(
            after[0] > before[0],
            "the keyword snapshot was not preloaded"
        );
        assert!(
            after[1] > before[1],
            "the identifier snapshot was not preloaded"
        );
    }
}
