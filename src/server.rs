// MCP server wiring — dispatches tool calls to underlying implementations.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rmcp::RoleServer;
use rmcp::handler::server::ServerHandler;
use rmcp::model::*;
use rmcp::service::RequestContext;
use serde_json::Value;
use tokio::sync::{OnceCell, RwLock};

use crate::cache::rkyv_store;
use crate::config::{Config, TrackerMode};
use crate::core::embedding_tracker::{
    EmbeddingTrackerConfig, EmbeddingTrackerHandle, RefreshCallback,
};
use crate::core::embeddings::{CacheEntry, OllamaClient};
use crate::core::memory_graph::GraphStore;
use crate::core::tree_sitter::parse_with_tree_sitter;
use crate::core::walker::walk_with_config;
use crate::error::{ContextPlusError, Result};
use crate::server_adapters::{CachedWalkerIndexer, OllamaEmbedder};
pub use crate::server_definitions::{make_tool, tool_definitions};

/// Cached project state: walked file entries and their raw file contents.
/// Built lazily on first tool call, invalidated by file watcher or TTL expiry.
/// Content is stored as `Arc<String>` so call-sites can clone the pointer
/// (cheap) and use `content.lines()` when line iteration is needed, avoiding
/// the `Vec<String>` split + `join("\n")` round-trip on every access.
pub struct ProjectCache {
    pub file_entries: Vec<crate::core::walker::FileEntry>,
    /// Maps relative_path → raw file content. Clone the Arc (pointer-sized)
    /// at each call-site; do not reconstruct from lines.
    pub file_content: HashMap<String, Arc<String>>,
    pub last_refresh: Instant,
}

/// Cached identifier index: parsed symbols + their embedding vectors.
/// Rebuilt when file count changes or TTL (300s) expires.
pub struct IdentifierIndex {
    pub docs: Vec<crate::tools::semantic_identifiers::IdentifierDoc>,
    pub vector_buffer: Vec<f32>,
    pub dims: usize,
    pub file_count: usize,
    pub built_at: Instant,
}

const IDENTIFIER_INDEX_TTL_SECS: u64 = 300;

/// Format a Unix timestamp (seconds since epoch) as ISO 8601 string.
fn format_unix_timestamp(ts: u64) -> String {
    const SECS_PER_DAY: u64 = 86400;
    const SECS_PER_HOUR: u64 = 3600;
    const SECS_PER_MIN: u64 = 60;
    let days = ts / SECS_PER_DAY;
    let time_of_day = ts % SECS_PER_DAY;
    let hours = time_of_day / SECS_PER_HOUR;
    let minutes = (time_of_day % SECS_PER_HOUR) / SECS_PER_MIN;
    let seconds = time_of_day % SECS_PER_MIN;
    let z = days as i64 + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hours, minutes, seconds
    )
}

/// URL to fetch the instructions resource content from.
const INSTRUCTIONS_SOURCE_URL: &str = "https://contextplus.vercel.app/api/instructions";
/// MCP resource URI for the instructions resource.
const INSTRUCTIONS_RESOURCE_URI: &str = "contextplus://instructions";

/// Shared state accessible by all tool handlers.
pub struct SharedState {
    pub config: Config,
    pub root_dir: PathBuf,
    /// Canonicalized version of root_dir — computed once at construction.
    /// Used by resolve_root() to validate caller-provided rootDir args without
    /// calling canonicalize() on every tool request.
    pub canonical_root: PathBuf,
    pub ollama: OllamaClient,
    pub memory_graph: Arc<GraphStore>,
    pub project_cache: RwLock<Option<Arc<ProjectCache>>>,
    /// Cached embedding vectors keyed by relative file path.
    /// Uses CacheEntry (hash + vector) for content-hash invalidation.
    /// Persisted to disk via rkyv_store for cross-restart survival.
    pub embedding_cache: RwLock<HashMap<String, CacheEntry>>,
    /// Cached identifier index — avoids re-embedding all symbols on each call.
    pub identifier_index: RwLock<Option<Arc<IdentifierIndex>>>,
    /// Cached SearchIndex for semantic_code_search — reused when the walk fingerprint
    /// is unchanged, eliminating the per-request HNSW rebuild for large corpora.
    /// Wrapped in `Arc` so background rebuild tasks can hold a clone of the lock.
    pub search_index_cache:
        Arc<RwLock<Option<Arc<crate::tools::semantic_search::CachedSearchIndex>>>>,
    /// Monotonic counter incremented by the embedding tracker whenever a file-change
    /// event is processed. `semantic_code_search` compares the counter at request time
    /// against `CachedSearchIndex::generation`; equality means the tracker has seen no
    /// changes since the last build and the filesystem walk can be skipped entirely.
    /// When the tracker is disabled the counter stays at 0 and the fingerprint-based
    /// fallback takes over.
    pub cache_generation: Arc<std::sync::atomic::AtomicU64>,
    /// Cached instructions content — fetched once from remote, then served from memory.
    pub instructions_cache: OnceCell<String>,
    /// Tracker handle for lazy-start mode.
    pub tracker_handle: std::sync::Mutex<Option<EmbeddingTrackerHandle>>,
    /// Idle monitor handle — tool handlers touch this to reset the idle timer.
    pub idle_monitor: RwLock<Option<Arc<crate::core::process_lifecycle::IdleMonitor>>>,
}

/// The MCP server exposing context+ tools.
#[derive(Clone)]
pub struct ContextPlusServer {
    pub state: Arc<SharedState>,
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

/// Build a model-qualified cache name to prevent cross-model cache poisoning.
/// E.g., `cache_name("embeddings", "snowflake-arctic-embed2")` → `"embeddings-snowflake-arctic-embed2"`.
pub fn cache_name(base: &str, model: &str) -> String {
    format!("{}-{}", base, sanitize_model_name(model))
}

impl ContextPlusServer {
    pub fn new(root_dir: PathBuf, config: Config) -> Self {
        let ollama = OllamaClient::new_with_root(&config, Some(root_dir.clone()));
        let memory_graph = Arc::new(GraphStore::new());

        let embed_cache_name = cache_name("embeddings", &config.ollama_embed_model);

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

        let state = Arc::new(SharedState {
            config,
            canonical_root,
            root_dir,
            ollama,
            memory_graph,
            project_cache: RwLock::new(None),
            embedding_cache: RwLock::new(initial_cache),
            identifier_index: RwLock::new(None),
            search_index_cache: Arc::new(RwLock::new(None)),
            cache_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            instructions_cache: OnceCell::new(),
            tracker_handle: std::sync::Mutex::new(None),
            idle_monitor: RwLock::new(None),
        });
        Self { state }
    }

    /// Build a refresh callback for the embedding tracker.
    pub fn build_tracker_callback(&self) -> RefreshCallback {
        let server = self.clone();
        let root = self.state.root_dir.clone();
        Arc::new(move |_root, files| {
            let srv = server.clone();
            let root = root.clone();
            let changed_files: Vec<PathBuf> = files.iter().map(|f| root.join(f)).collect();
            tokio::spawn(async move {
                let (updated, skipped) = srv.incremental_reembed(&changed_files).await;
                tracing::debug!(
                    updated,
                    skipped,
                    "Incremental re-embedding for {} changed files",
                    changed_files.len()
                );
                // Invalidate the search-index cache by bumping the generation counter.
                // Any in-flight or subsequent `semantic_code_search` request will see a
                // generation mismatch and re-walk + rebuild instead of reusing the stale
                // cached index.
                let new_gen = srv
                    .state
                    .cache_generation
                    .fetch_add(1, std::sync::atomic::Ordering::Release)
                    + 1;
                tracing::debug!(
                    generation = new_gen,
                    "cache_generation bumped after tracker event"
                );
                (updated, skipped)
            })
        })
    }

    /// Start the embedding tracker if not already running and mode is not Off.
    pub fn ensure_tracker_started(&self) {
        if self.state.config.embed_tracker_mode == TrackerMode::Off {
            return;
        }
        let mut guard = self.state.tracker_handle.lock().unwrap_or_else(|poisoned| {
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
        let callback = self.build_tracker_callback();
        match crate::core::embedding_tracker::start_tracker(
            self.state.root_dir.clone(),
            tracker_config,
            callback,
        ) {
            Ok(handle) => {
                tracing::info!(
                    mode = %self.state.config.embed_tracker_mode,
                    "Embedding tracker started"
                );
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
    pub fn spawn_warmup_task(&self) {
        let state = self.state.clone();
        tokio::spawn(async move {
            warmup_semantic_search_cache(&state).await;
        });
    }

    fn root_dir(&self) -> &Path {
        &self.state.root_dir
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

    /// Returns a snapshot of the project cache, lazily initializing or refreshing
    /// when the TTL has expired. All filesystem I/O runs inside `spawn_blocking`.
    /// Uses Arc to avoid deep-cloning the entire cache on every tool call.
    async fn ensure_project_cache(&self) -> Result<Arc<ProjectCache>> {
        let ttl_secs = self.state.config.cache_ttl_secs;

        // Fast path: cache exists and is fresh — just clone the Arc (cheap)
        {
            let guard = self.state.project_cache.read().await;
            if let Some(ref cache) = *guard
                && cache.last_refresh.elapsed().as_secs() < ttl_secs
            {
                return Ok(Arc::clone(cache));
            }
        }

        // Slow path: rebuild cache
        let root = self.state.root_dir.clone();
        let config = self.state.config.clone();

        let new_cache = tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;

            let entries = walk_with_config(&root, &config);
            let file_content: HashMap<String, Arc<String>> = entries
                .par_iter()
                .filter(|entry| !entry.is_directory)
                .filter_map(|entry| {
                    let full_path = root.join(&entry.relative_path);
                    std::fs::read_to_string(&full_path)
                        .ok()
                        .map(|content| (entry.relative_path.clone(), Arc::new(content)))
                })
                .collect();
            ProjectCache {
                file_entries: entries,
                file_content,
                last_refresh: Instant::now(),
            }
        })
        .await
        .map_err(|e| ContextPlusError::Other(format!("spawn_blocking failed: {e}")))?;

        let arc_cache = Arc::new(new_cache);

        // Store the Arc in shared state (cheap clone of the Arc pointer)
        {
            let mut guard = self.state.project_cache.write().await;
            *guard = Some(Arc::clone(&arc_cache));
        }

        Ok(arc_cache)
    }

    /// Invalidate the project cache. Called by the file watcher when files change.
    /// Note: embedding_cache is NOT cleared — content hashes in CacheEntry
    /// handle staleness detection. Only clears the project file/line cache.
    /// Also invalidates the identifier index so it rebuilds with fresh data.
    pub async fn invalidate_project_cache(&self) {
        let mut guard = self.state.project_cache.write().await;
        *guard = None;
        drop(guard);
        let mut idx_guard = self.state.identifier_index.write().await;
        *idx_guard = None;
    }

    /// Incrementally re-embed specific changed files without invalidating the entire cache.
    /// Reads each file, computes content hash, re-embeds only if changed, updates cache.
    /// Returns (updated_count, skipped_count).
    pub async fn incremental_reembed(&self, files: &[std::path::PathBuf]) -> (usize, usize) {
        let mut updated = 0usize;
        let mut skipped = 0usize;

        let max_file_size = self.state.config.max_embed_file_size as u64;

        let mut texts_to_embed: Vec<(String, String, String)> = Vec::new(); // (rel_path, hash, text)

        for file_path in files {
            let rel_path = match file_path.strip_prefix(&self.state.root_dir) {
                Ok(r) => r.to_string_lossy().to_string(),
                Err(_) => file_path.to_string_lossy().to_string(),
            };

            if let Ok(meta) = tokio::fs::metadata(file_path).await
                && meta.len() > max_file_size
            {
                skipped += 1;
                continue;
            }

            let content = match tokio::fs::read_to_string(file_path).await {
                Ok(c) => c,
                Err(_) => {
                    // File deleted — remove from cache
                    let mut cache = self.state.embedding_cache.write().await;
                    cache.remove(&rel_path);
                    updated += 1;
                    continue;
                }
            };

            let hash = crate::core::parser::hash_content(&content);

            // Check if content actually changed
            {
                let cache = self.state.embedding_cache.read().await;
                if let Some(entry) = cache.get(&rel_path)
                    && entry.hash == hash
                {
                    skipped += 1;
                    continue;
                }
            }

            let truncated = if content.len() > 500 {
                crate::core::parser::truncate_to_char_boundary(&content, 500).to_string()
            } else {
                content
            };
            let header = crate::core::parser::extract_header(&truncated);
            let text = format!("{} {} {}", header, rel_path, truncated);
            texts_to_embed.push((rel_path, hash, text));
        }

        if texts_to_embed.is_empty() {
            return (updated, skipped);
        }

        let embed_texts: Vec<String> = texts_to_embed.iter().map(|(_, _, t)| t.clone()).collect();
        match self.state.ollama.embed(&embed_texts).await {
            Ok(vectors) => {
                let mut cache = self.state.embedding_cache.write().await;
                for (i, (rel_path, hash, _)) in texts_to_embed.iter().enumerate() {
                    if i < vectors.len() {
                        cache.insert(
                            rel_path.clone(),
                            CacheEntry {
                                hash: hash.clone(),
                                vector: vectors[i].clone(),
                            },
                        );
                        updated += 1;
                    }
                }

                // Persist to disk outside the write lock scope. The save path
                // takes a blocking fd-lock + does sync I/O, so move it off the
                // Tokio worker via spawn_blocking.
                let store = crate::core::embeddings::VectorStore::from_cache(&cache);
                drop(cache);
                let embed_cache_name =
                    cache_name("embeddings", &self.state.config.ollama_embed_model);
                if let Some(s) = store {
                    let root = self.state.root_dir.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        // Single-writer live path: overwrite without merge to avoid
                        // the ~146 MB read + clone overhead of save_vector_store.
                        rkyv_store::save_vector_store_overwrite(&root, &embed_cache_name, &s)
                    })
                    .await;
                    match result {
                        Ok(Err(e)) => {
                            tracing::warn!("Failed to save incremental embedding cache: {e}")
                        }
                        Err(join_err) => tracing::warn!(
                            "save_vector_store spawn_blocking join failed: {join_err}"
                        ),
                        Ok(Ok(())) => {}
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Incremental re-embed failed: {e}");
            }
        }

        // Invalidate project cache + identifier index so they rebuild with fresh data
        self.invalidate_project_cache().await;

        (updated, skipped)
    }

    /// Ensure the identifier index is built and cached.
    /// Returns cached index if TTL hasn't expired and file count is unchanged.
    /// Otherwise rebuilds: parses all symbols, embeds them, caches the result.
    async fn ensure_identifier_index(
        &self,
        cache: &Arc<ProjectCache>,
    ) -> Result<Arc<IdentifierIndex>> {
        let file_count = cache
            .file_entries
            .iter()
            .filter(|e| !e.is_directory)
            .count();

        // Fast path: index exists, TTL valid, file count unchanged — clone Arc (cheap)
        {
            let guard = self.state.identifier_index.read().await;
            if let Some(ref idx) = *guard
                && idx.file_count == file_count
                && idx.built_at.elapsed().as_secs() < IDENTIFIER_INDEX_TTL_SECS
            {
                return Ok(Arc::clone(idx));
            }
        }

        // Slow path: rebuild identifier index
        tracing::info!(
            file_count,
            "Building identifier index (parsing + embedding)"
        );
        let cache_clone = cache.clone();

        // Step 1: Parse symbols (CPU-bound)
        let identifier_docs = tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            cache_clone
                .file_entries
                .par_iter()
                .filter(|entry| !entry.is_directory)
                .filter_map(|entry| {
                    let content = cache_clone.file_content.get(&entry.relative_path)?;
                    let content = Arc::clone(content);
                    let ext = entry.relative_path.rsplit('.').next().unwrap_or("");
                    let symbols = parse_with_tree_sitter(&content, ext).ok()?;
                    let header = crate::core::parser::extract_header(&content);
                    let local_docs: Vec<crate::tools::semantic_identifiers::IdentifierDoc> =
                        crate::core::parser::flatten_symbols(&symbols, None)
                            .into_iter()
                            .map(|sym| {
                                let sig = sym.signature.clone().unwrap_or_default();
                                let parent = sym.parent_name.as_deref().unwrap_or("");
                                let text = format!(
                                    "{} {} {} {} {} {}",
                                    sym.name,
                                    sym.kind,
                                    sig,
                                    entry.relative_path,
                                    header,
                                    parent
                                );
                                let token_set =
                                    crate::tools::semantic_identifiers::IdentifierDoc::build_token_set(
                                        &sym.name,
                                        &sig,
                                        &entry.relative_path,
                                        &header,
                                    );
                                crate::tools::semantic_identifiers::IdentifierDoc {
                                    id: format!(
                                        "{}:{}:{}",
                                        entry.relative_path, sym.name, sym.line
                                    ),
                                    path: entry.relative_path.clone(),
                                    header: header.clone(),
                                    name: sym.name.clone(),
                                    kind_lower: sym.kind.to_lowercase(),
                                    kind: sym.kind.clone(),
                                    line: sym.line,
                                    end_line: sym.end_line,
                                    signature: sig,
                                    parent_name: sym.parent_name.clone(),
                                    text,
                                    token_set,
                                }
                            })
                            .collect();
                    Some(local_docs)
                })
                .flatten()
                .collect::<Vec<_>>()
        })
        .await
        .map_err(|e| ContextPlusError::Other(format!("spawn_blocking failed: {e}")))?;

        if identifier_docs.is_empty() {
            let idx = Arc::new(IdentifierIndex {
                docs: Vec::new(),
                vector_buffer: Vec::new(),
                dims: 0,
                file_count,
                built_at: Instant::now(),
            });
            let mut guard = self.state.identifier_index.write().await;
            *guard = Some(Arc::clone(&idx));
            return Ok(idx);
        }

        // Step 2: Check identifier embedding cache on disk, embed only missing
        let n_identifiers = identifier_docs.len();
        tracing::info!(
            identifiers = n_identifiers,
            "Embedding identifiers (using disk cache for warm hits)"
        );

        // Load identifier-specific embedding cache
        let id_cache_name = cache_name(
            "identifier-embeddings",
            &self.state.config.ollama_embed_model,
        );
        let id_cache = match rkyv_store::load_cache(&self.state.root_dir, &id_cache_name) {
            Ok(Some(data)) => {
                let store = data.to_store();
                tracing::info!(cached = store.count(), "Loaded identifier embedding cache");
                Some(store)
            }
            _ => None,
        };

        // Partition: cached vs uncached identifiers (use &str slices for cache lookup)
        let mut result_vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(n_identifiers);
        let mut uncached_indices: Vec<usize> = Vec::new();
        let mut uncached_texts: Vec<String> = Vec::new();

        for (i, doc) in identifier_docs.iter().enumerate() {
            if let Some(ref store) = id_cache
                && let Some(vec) = store.get_vector(&doc.text)
            {
                result_vectors.push(Some(vec.to_vec()));
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

        // Embed only uncached identifiers, in chunks to survive MCP connection timeouts.
        if !uncached_texts.is_empty() {
            let chunk_size = self.state.ollama.batch_size();
            for chunk_start in (0..uncached_indices.len()).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(uncached_indices.len());
                let chunk_texts = &uncached_texts[chunk_start..chunk_end];

                let chunk_vectors = self.state.ollama.embed(chunk_texts).await?;
                for (local_j, &idx) in uncached_indices[chunk_start..chunk_end].iter().enumerate() {
                    if local_j < chunk_vectors.len() {
                        result_vectors[idx] = Some(chunk_vectors[local_j].clone());
                    }
                }

                // Persist after each chunk so progress survives a timeout on the next batch.
                let all_vecs: Vec<Vec<f32>> =
                    result_vectors.iter().filter_map(|v| v.clone()).collect();
                if all_vecs.len() == n_identifiers {
                    let dims = all_vecs.first().map_or(0, |v| v.len()) as u32;
                    let keys: Vec<String> =
                        identifier_docs.iter().map(|d| d.text.clone()).collect();
                    let hashes: Vec<String> = keys
                        .iter()
                        .map(|k| crate::core::parser::hash_content(k))
                        .collect();
                    let flat: Vec<f32> = all_vecs.into_iter().flatten().collect();
                    let store = crate::core::embeddings::VectorStore::new(dims, keys, hashes, flat);
                    let root = self.state.root_dir.clone();
                    let cache_name_owned = id_cache_name.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        // Single-writer live path: overwrite without merge.
                        rkyv_store::save_vector_store_overwrite(&root, &cache_name_owned, &store)
                    })
                    .await;
                    match result {
                        Ok(Err(e)) => {
                            tracing::warn!("Failed to save identifier embedding cache: {e}")
                        }
                        Err(join_err) => tracing::warn!(
                            "save_vector_store spawn_blocking join failed: {join_err}"
                        ),
                        Ok(Ok(())) => {}
                    }
                }
            }
        }

        let dims = result_vectors
            .first()
            .and_then(|v| v.as_ref())
            .map_or(0, |v| v.len());
        let flat_buffer: Vec<f32> = result_vectors
            .into_iter()
            .flat_map(|v| v.unwrap_or_else(|| vec![0.0; dims]))
            .collect();

        let idx = Arc::new(IdentifierIndex {
            docs: identifier_docs,
            vector_buffer: flat_buffer,
            dims,
            file_count,
            built_at: Instant::now(),
        });

        // Store Arc in shared state (cheap pointer clone, no data copy)
        {
            let mut guard = self.state.identifier_index.write().await;
            *guard = Some(Arc::clone(&idx));
        }

        Ok(idx)
    }

    // --- Tool dispatch ---

    pub async fn dispatch(
        &self,
        name: &str,
        args: serde_json::Map<String, Value>,
    ) -> CallToolResult {
        match self.dispatch_inner(name, args).await {
            Ok(result) => result,
            Err(e) => Self::err_text(format!("Error: {}", e)),
        }
    }

    async fn dispatch_inner(
        &self,
        name: &str,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        match name {
            "get_context_tree" => self.handle_context_tree(args).await,
            "get_file_skeleton" => self.handle_file_skeleton(args).await,
            "get_blast_radius" => self.handle_blast_radius(args).await,
            "semantic_code_search" => self.handle_semantic_code_search(args).await,
            "semantic_identifier_search" => self.handle_semantic_identifier_search(args).await,
            "semantic_navigate" => self.handle_semantic_navigate(args).await,
            "get_feature_hub" => self.handle_feature_hub(args).await,
            "run_static_analysis" => self.handle_static_analysis(args).await,
            "propose_commit" => self.handle_propose_commit(args).await,
            "list_restore_points" => self.handle_list_restore_points(args).await,
            "undo_change" => self.handle_undo_change(args).await,
            "upsert_memory_node" => self.handle_upsert_memory_node(args).await,
            "create_relation" => self.handle_create_relation(args).await,
            "search_memory_graph" => self.handle_search_memory_graph(args).await,
            "prune_stale_links" => self.handle_prune_stale_links(args).await,
            "add_interlinked_context" => self.handle_add_interlinked_context(args).await,
            "retrieve_with_traversal" => self.handle_retrieve_with_traversal(args).await,
            "delete_memory_node" => self.handle_delete_memory_node(args).await,
            "find_dead_code" => self.handle_find_dead_code(args).await,
            "review_pr_diff" => self.handle_review_pr_diff(args).await,
            "detect_dependency_loops" => self.handle_detect_dependency_loops(args).await,
            "check_embedding_quality" => self.handle_check_embedding_quality(args).await,
            "lexical_search" => self.handle_lexical_search(args).await,
            _ => Ok(Self::err_text(format!("Unknown tool: {}", name))),
        }
    }

    // ----- Individual tool handlers -----

    async fn handle_context_tree(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        use crate::tools::context_tree as ct;

        let root = self.resolve_root(&args);
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

        let root = self.resolve_root(&args);
        let full_path = root.join(&file_path);

        // Check ProjectCache.file_content first to avoid a disk read on warm cache.
        let cached_content: Option<Arc<String>> = {
            let cache_guard = self.state.project_cache.read().await;
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

    async fn handle_blast_radius(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let symbol_name = Self::get_str(&args, "symbol_name")
            .ok_or_else(|| ContextPlusError::Other("symbol_name is required".into()))?;
        let file_context = Self::get_str(&args, "file_context");

        let cache = self.ensure_project_cache().await?;

        // find_symbol_usages scans all file content — CPU-bound, run in blocking thread pool.
        let formatted = tokio::task::spawn_blocking(move || {
            let result = crate::tools::blast_radius::find_symbol_usages(
                &symbol_name,
                file_context.as_deref(),
                &cache.file_content,
            );
            crate::tools::blast_radius::format_blast_radius(&symbol_name, &result)
        })
        .await
        .map_err(|e| ContextPlusError::Other(format!("blast_radius spawn_blocking failed: {e}")))?;

        Ok(Self::ok_text(formatted))
    }

    async fn handle_semantic_code_search(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        self.ensure_tracker_started();
        let query = Self::get_str(&args, "query")
            .ok_or_else(|| ContextPlusError::Other("query is required".into()))?;
        let root = self.resolve_root(&args);

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
        };

        let embedder = OllamaEmbedder(self.state.ollama.clone());
        let walker = CachedWalkerIndexer {
            config: self.state.config.clone(),
            ollama: self.state.ollama.clone(),
            state: self.state.clone(),
        };

        // Pass the generation counter when the tracker is active so
        // `semantic_code_search` can skip the walk on a generation hit.
        // When the tracker is Off the counter stays at 0 and the
        // fingerprint-based fallback is used instead.
        let cache_gen = if self.state.config.embed_tracker_mode != crate::config::TrackerMode::Off {
            Some(&self.state.cache_generation)
        } else {
            None
        };
        let result = crate::tools::semantic_search::semantic_code_search(
            options,
            &embedder,
            &walker,
            Some(Arc::clone(&self.state.search_index_cache)),
            cache_gen.cloned(),
        )
        .await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_semantic_identifier_search(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        self.ensure_tracker_started();
        use crate::tools::semantic_identifiers::*;

        let query = Self::get_str(&args, "query")
            .ok_or_else(|| ContextPlusError::Other("query is required".into()))?;
        let root = self.resolve_root(&args);

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

        let result = semantic_identifier_search(
            options,
            &OllamaEmbedder(self.state.ollama.clone()),
            &idx.docs,
            &idx.vector_buffer,
            idx.dims,
            &cache.file_content,
        )
        .await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_semantic_navigate(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        self.ensure_tracker_started();
        let root = self.resolve_root(&args);

        let options = crate::tools::semantic_navigate::SemanticNavigateOptions {
            root_dir: root.to_string_lossy().into(),
            max_depth: Self::get_usize(&args, "max_depth"),
            max_clusters: Self::get_usize(&args, "max_clusters"),
            min_clusters: Self::get_usize(&args, "min_clusters"),
            mode: Self::get_str(&args, "mode"),
        };

        let result = crate::tools::semantic_navigate::semantic_navigate(
            options,
            &self.state.ollama,
            &self.state.config,
            &self.state.embedding_cache,
            &self.state.root_dir,
        )
        .await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_feature_hub(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let root = self.resolve_root(&args);
        let hub_path = Self::get_str(&args, "hub_path");

        let options = crate::tools::feature_hub::FeatureHubOptions {
            root_dir: root.to_string_lossy().into(),
            hub_path,
            feature_name: Self::get_str(&args, "feature_name"),
            show_orphans: Self::get_bool(&args, "show_orphans"),
        };

        let result = crate::tools::feature_hub::get_feature_hub(options).await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_static_analysis(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let root = self.resolve_root(&args);
        let target_path = Self::get_str(&args, "target_path");

        let options = crate::tools::static_analysis::StaticAnalysisOptions {
            root_dir: root,
            target_path,
        };

        let result = crate::tools::static_analysis::run_static_analysis(options).await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_propose_commit(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let file_path = Self::get_str(&args, "file_path")
            .ok_or_else(|| ContextPlusError::Other("file_path is required".into()))?;
        let content = Self::get_str(&args, "new_content")
            .ok_or_else(|| ContextPlusError::Other("new_content is required".into()))?;
        let description = Self::get_str(&args, "description");
        let root = self.resolve_root(&args);

        let result = crate::tools::propose_commit::propose_commit(
            &root,
            &file_path,
            &content,
            description.as_deref(),
        )
        .await?;

        // Invalidate project cache after file write (matches TS invalidateSearchCache behavior)
        self.invalidate_project_cache().await;

        Ok(Self::ok_text(result))
    }

    async fn handle_list_restore_points(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let root = self.resolve_root(&args);
        let points = crate::git::shadow::list_restore_points(&root).await?;

        if points.is_empty() {
            return Ok(Self::ok_text("No restore points found.".to_string()));
        }

        let mut output = String::from("Restore Points:\n\n");
        for pt in &points {
            let file_names: Vec<&str> = pt.files.iter().map(|f| f.original_path.as_str()).collect();
            let iso_ts = format_unix_timestamp(pt.timestamp);
            output.push_str(&format!(
                "{} | {} | {} | {}\n",
                pt.id,
                iso_ts,
                file_names.join(", "),
                pt.description,
            ));
        }
        Ok(Self::ok_text(output))
    }

    async fn handle_undo_change(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let restore_point_id = Self::get_str(&args, "point_id")
            .ok_or_else(|| ContextPlusError::Other("point_id is required".into()))?;
        let root = self.resolve_root(&args);

        let restored = crate::git::shadow::restore_from_point(&root, &restore_point_id).await?;

        // Invalidate project cache after file restore (matches TS invalidateSearchCache behavior)
        self.invalidate_project_cache().await;

        let msg = format!(
            "Restored {} file(s) from restore point {}:\n  {}",
            restored.len(),
            restore_point_id,
            restored.join("\n  ")
        );
        Ok(Self::ok_text(msg))
    }

    async fn handle_upsert_memory_node(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let options = crate::tools::memory_tools::UpsertMemoryNodeOptions {
            root_dir: self.root_dir().to_string_lossy().into(),
            node_type: Self::get_str(&args, "type").unwrap_or_else(|| "concept".to_string()),
            label: Self::get_str(&args, "label")
                .ok_or_else(|| ContextPlusError::Other("label is required".into()))?,
            content: Self::get_str(&args, "content")
                .ok_or_else(|| ContextPlusError::Other("content is required".into()))?,
            metadata: parse_metadata(&args),
        };

        let store = &self.state.memory_graph;
        let result =
            crate::tools::memory_tools::tool_upsert_memory_node(store, &self.state.ollama, options)
                .await?;

        Ok(Self::ok_text(result))
    }

    async fn handle_create_relation(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let options = crate::tools::memory_tools::CreateRelationOptions {
            root_dir: self.root_dir().to_string_lossy().into(),
            // TS API: source_id / target_id (direct node IDs)
            source_id: Self::get_str(&args, "source_id"),
            source_label: Self::get_str(&args, "source_label"),
            source_type: Self::get_str(&args, "source_type")
                .unwrap_or_else(|| "concept".to_string()),
            target_id: Self::get_str(&args, "target_id"),
            target_label: Self::get_str(&args, "target_label"),
            target_type: Self::get_str(&args, "target_type")
                .unwrap_or_else(|| "concept".to_string()),
            relation: Self::get_str(&args, "relation").unwrap_or_else(|| "relates_to".to_string()),
            weight: Self::get_f64(&args, "weight").map(|w| w as f32),
            metadata: parse_metadata(&args),
        };

        let store = &self.state.memory_graph;
        let result = crate::tools::memory_tools::tool_create_relation(store, options).await?;

        Ok(Self::ok_text(result))
    }

    async fn handle_search_memory_graph(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let options = crate::tools::memory_tools::SearchMemoryGraphOptions {
            root_dir: self.root_dir().to_string_lossy().into(),
            query: Self::get_str(&args, "query")
                .ok_or_else(|| ContextPlusError::Other("query is required".into()))?,
            max_depth: Self::get_usize(&args, "max_depth"),
            top_k: Self::get_usize(&args, "top_k"),
            edge_filter: Self::get_string_array(&args, "edge_filter"),
        };

        let store = &self.state.memory_graph;
        let result = crate::tools::memory_tools::tool_search_memory_graph(
            store,
            &self.state.ollama,
            options,
        )
        .await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_prune_stale_links(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let options = crate::tools::memory_tools::PruneStaleLinksOptions {
            root_dir: self.root_dir().to_string_lossy().into(),
            threshold: Self::get_f64(&args, "threshold"),
        };

        let store = &self.state.memory_graph;
        let result = crate::tools::memory_tools::tool_prune_stale_links(store, options).await?;

        Ok(Self::ok_text(result))
    }

    async fn handle_add_interlinked_context(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let items = args
            .get("items")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        let obj = item.as_object()?;
                        Some(crate::tools::memory_tools::InterlinkedItem {
                            node_type: Self::get_str(obj, "type")
                                .unwrap_or_else(|| "concept".to_string()),
                            label: Self::get_str(obj, "label")?,
                            content: Self::get_str(obj, "content")?,
                            metadata: parse_metadata(obj),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let options = crate::tools::memory_tools::AddInterlinkedContextOptions {
            root_dir: self.root_dir().to_string_lossy().into(),
            items,
            auto_link: Self::get_bool(&args, "auto_link"),
        };

        let store = &self.state.memory_graph;
        let result = crate::tools::memory_tools::tool_add_interlinked_context(
            store,
            &self.state.ollama,
            options,
        )
        .await?;

        Ok(Self::ok_text(result))
    }

    async fn handle_retrieve_with_traversal(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let options = crate::tools::memory_tools::RetrieveWithTraversalOptions {
            root_dir: self.root_dir().to_string_lossy().into(),
            node_id: Self::get_str(&args, "start_node_id")
                .ok_or_else(|| ContextPlusError::Other("start_node_id is required".into()))?,
            max_depth: Self::get_usize(&args, "max_depth"),
            max_nodes: Self::get_usize(&args, "max_nodes"),
            edge_filter: Self::get_string_array(&args, "edge_filter"),
        };

        let store = &self.state.memory_graph;
        let result =
            crate::tools::memory_tools::tool_retrieve_with_traversal(store, options).await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_delete_memory_node(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let root_dir = self.root_dir().to_string_lossy().into_owned();
        let node_id = Self::get_str(&args, "node_id")
            .ok_or_else(|| ContextPlusError::Other("node_id is required".into()))?;

        let store = &self.state.memory_graph;
        let result =
            crate::tools::memory_tools::tool_delete_memory_node(store, &root_dir, &node_id).await?;

        Ok(Self::ok_text(result))
    }

    // --- Helpers ---

    fn resolve_root(&self, args: &serde_json::Map<String, Value>) -> PathBuf {
        if let Some(requested) = Self::get_str(args, "rootDir") {
            let requested_path = PathBuf::from(&requested);
            // Use pre-canonicalized root (computed once at construction, not per-request).
            if let Ok(canonical_requested) = requested_path.canonicalize()
                && canonical_requested.starts_with(&self.state.canonical_root)
            {
                return canonical_requested;
            }
            tracing::warn!(
                requested = %requested,
                root = %self.state.root_dir.display(),
                "Caller-provided rootDir is outside the server root; ignoring"
            );
        }
        self.state.root_dir.clone()
    }

    // ----- find_dead_code -----

    async fn handle_find_dead_code(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        use crate::tools::dead_code_find::{
            DeadCodeOptions, find_dead_symbols, format_dead_symbols,
        };

        let cache = self.ensure_project_cache().await?;

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
            format_dead_symbols(&dead)
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
        let root = self.state.root_dir.clone();

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
        let root = self.state.root_dir.clone();

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
            let guard = self.state.embedding_cache.read().await;
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
        use crate::tools::lexical_search::LexicalIndex;
        use crate::tools::semantic_search::SearchDocument;

        let query = Self::get_str(&args, "query")
            .ok_or_else(|| ContextPlusError::Other("query is required".into()))?;
        // top_k=0 would silently return zero hits (LexicalIndex::search short-
        // circuits on n=0) and the user would see "No matches" — masking the
        // bad input. Treat 0 as "use default" the same way find_dead_code does.
        let top_k = Self::get_usize(&args, "top_k")
            .filter(|&n| n > 0)
            .unwrap_or(10);

        let cache = self.ensure_project_cache().await?;

        let formatted = tokio::task::spawn_blocking(move || {
            let docs: Vec<SearchDocument> = cache
                .file_entries
                .iter()
                .filter(|e| !e.is_directory)
                .map(|e| {
                    let content: String = cache
                        .file_content
                        .get(&e.relative_path)
                        .map(|arc| arc.as_str().to_owned())
                        .unwrap_or_default();
                    let ext = e.relative_path.rsplit('.').next().unwrap_or("");
                    let symbols: Vec<String> = parse_with_tree_sitter(&content, ext)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|s| s.name)
                        .collect();
                    let header = crate::core::parser::extract_header(&content);
                    SearchDocument::new(e.relative_path.clone(), header, symbols, vec![], content)
                })
                .collect();

            if docs.is_empty() {
                return "No files indexed. Ensure the project cache is populated.".to_string();
            }

            let index = LexicalIndex::build(&docs);
            let hits = index.search(&query, top_k);

            if hits.is_empty() {
                return format!("No lexical matches found for: {query}");
            }

            let mut lines = vec![format!(
                "Lexical search: {} result(s) for \"{query}\"",
                hits.len()
            )];
            lines.push(String::new());
            for (rank, (doc_idx, score)) in hits.iter().enumerate() {
                if *doc_idx < docs.len() {
                    lines.push(format!(
                        "{}. {} (score: {:.3})",
                        rank + 1,
                        docs[*doc_idx].path,
                        score
                    ));
                }
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
            "Context+ semantic code analysis server. Provides semantic search, \
             blast radius analysis, context trees, file skeletons, navigation, \
             memory graph, and more.",
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
        let args = request.arguments.unwrap_or_default();
        Ok(self.dispatch(&name, args).await)
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

fn parse_metadata(
    args: &serde_json::Map<String, Value>,
) -> Option<std::collections::HashMap<String, String>> {
    args.get("metadata").and_then(|v| v.as_object()).map(|obj| {
        obj.iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
            .collect()
    })
}

// make_tool() is re-exported from server_definitions — see imports at top of this file.

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

    let options = SemanticSearchOptions {
        root_dir: state.root_dir.clone(),
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
        Some(Arc::clone(&state.search_index_cache)),
        Some(Arc::clone(&state.cache_generation)),
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
    use rmcp::model::RawContent;
    use serde_json::json;

    fn test_server() -> ContextPlusServer {
        let config = Config::from_env();
        let root = std::env::temp_dir().join("contextplus-test");
        let _ = std::fs::create_dir_all(&root);
        ContextPlusServer::new(root, config)
    }

    #[test]
    fn tool_definitions_returns_all_23_tools() {
        let defs = tool_definitions();
        assert_eq!(defs.len(), 23, "expected 23 tools, got {}", defs.len());
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
        let expected = [
            "get_context_tree",
            "get_file_skeleton",
            "get_blast_radius",
            "semantic_code_search",
            "semantic_identifier_search",
            "semantic_navigate",
            "get_feature_hub",
            "run_static_analysis",
            "propose_commit",
            "list_restore_points",
            "undo_change",
            "upsert_memory_node",
            "create_relation",
            "search_memory_graph",
            "prune_stale_links",
            "add_interlinked_context",
            "retrieve_with_traversal",
            "delete_memory_node",
            "find_dead_code",
            "review_pr_diff",
            "detect_dependency_loops",
            "check_embedding_quality",
            "lexical_search",
        ];
        for name in expected {
            assert!(names.contains(&name), "missing tool: {}", name);
        }
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

    #[tokio::test]
    async fn dispatch_list_restore_points_succeeds_with_empty_state() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("list_restore_points", args).await;
        assert_eq!(
            result.is_error,
            Some(false),
            "list_restore_points should not error on empty state"
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

    #[test]
    fn resolve_root_uses_server_root_when_no_arg() {
        let server = test_server();
        let args = serde_json::Map::new();
        let root = server.resolve_root(&args);
        assert_eq!(root, server.state.root_dir);
    }

    #[test]
    fn resolve_root_rejects_path_outside_server_root() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert("rootDir".to_string(), json!("/etc/passwd"));
        let root = server.resolve_root(&args);
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

    #[test]
    fn parse_metadata_extracts_map() {
        let mut args = serde_json::Map::new();
        let mut meta = serde_json::Map::new();
        meta.insert("source".to_string(), json!("test"));
        meta.insert("priority".to_string(), json!("high"));
        args.insert("metadata".to_string(), Value::Object(meta));

        let result = parse_metadata(&args).unwrap();
        assert_eq!(result.get("source"), Some(&"test".to_string()));
        assert_eq!(result.get("priority"), Some(&"high".to_string()));
    }

    #[test]
    fn parse_metadata_returns_none_when_missing() {
        let args = serde_json::Map::new();
        assert!(parse_metadata(&args).is_none());
    }

    // --- ProjectCache tests ---

    /// Build a ContextPlusServer rooted at the given directory with a custom TTL.
    fn server_with_root_and_ttl(root: PathBuf, cache_ttl_secs: u64) -> ContextPlusServer {
        let mut config = Config::from_env();
        config.cache_ttl_secs = cache_ttl_secs;
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

    #[test]
    fn resolve_root_accepts_subdirectory_inside_root() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let sub = tmp.path().join("subdir");
        std::fs::create_dir_all(&sub).unwrap();
        let server = server_with_root_and_ttl(tmp.path().to_path_buf(), 300);

        let mut args = serde_json::Map::new();
        args.insert(
            "rootDir".to_string(),
            json!(sub.to_string_lossy().to_string()),
        );

        let root = server.resolve_root(&args);
        // Should accept the subdirectory since it's inside the server root
        let canonical_sub = sub.canonicalize().unwrap();
        assert_eq!(root, canonical_sub);
    }

    #[test]
    fn resolve_root_rejects_nonexistent_path() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert("rootDir".to_string(), json!("/nonexistent/path/xyz123"));
        let root = server.resolve_root(&args);
        // Should fall back to server root since path doesn't exist (canonicalize fails)
        assert_eq!(root, server.state.root_dir);
    }

    #[test]
    fn resolve_root_rejects_empty_string() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert("rootDir".to_string(), json!(""));
        let root = server.resolve_root(&args);
        // Empty string can't be canonicalized to a path inside root
        assert_eq!(root, server.state.root_dir);
    }

    #[test]
    fn resolve_root_rejects_relative_path_outside_root() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert("rootDir".to_string(), json!("../../etc"));
        let root = server.resolve_root(&args);
        assert_eq!(root, server.state.root_dir);
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
    // parse_metadata edge cases
    // ---------------------------------------------------------------

    #[test]
    fn parse_metadata_returns_none_for_non_object_metadata() {
        let mut args = serde_json::Map::new();
        args.insert("metadata".to_string(), json!("not-an-object"));
        assert!(parse_metadata(&args).is_none());
    }

    #[test]
    fn parse_metadata_returns_none_for_array_metadata() {
        let mut args = serde_json::Map::new();
        args.insert("metadata".to_string(), json!(["a", "b"]));
        assert!(parse_metadata(&args).is_none());
    }

    #[test]
    fn parse_metadata_converts_non_string_values_to_empty_string() {
        let mut args = serde_json::Map::new();
        let mut meta = serde_json::Map::new();
        meta.insert("count".to_string(), json!(42));
        meta.insert("flag".to_string(), json!(true));
        meta.insert("valid".to_string(), json!("ok"));
        args.insert("metadata".to_string(), Value::Object(meta));

        let result = parse_metadata(&args).unwrap();
        // Non-string values get as_str() => None => unwrap_or("") => ""
        assert_eq!(result.get("count"), Some(&"".to_string()));
        assert_eq!(result.get("flag"), Some(&"".to_string()));
        assert_eq!(result.get("valid"), Some(&"ok".to_string()));
    }

    #[test]
    fn parse_metadata_handles_empty_object() {
        let mut args = serde_json::Map::new();
        args.insert("metadata".to_string(), json!({}));
        let result = parse_metadata(&args).unwrap();
        assert!(result.is_empty());
    }

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
    async fn dispatch_propose_commit_missing_file_path_returns_error() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("propose_commit", args).await;
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
    async fn dispatch_propose_commit_missing_content_returns_error() {
        let server = test_server();
        let mut args = serde_json::Map::new();
        args.insert("file_path".to_string(), json!("test.txt"));
        let result = server.dispatch("propose_commit", args).await;
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("new_content is required"),
            "expected new_content error, got: {}",
            text
        );
    }

    #[tokio::test]
    async fn dispatch_undo_change_missing_restore_point_id_returns_error() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("undo_change", args).await;
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("point_id is required"),
            "expected point_id error, got: {}",
            text
        );
    }

    #[tokio::test]
    async fn dispatch_upsert_memory_node_missing_label_returns_error() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("upsert_memory_node", args).await;
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("label is required"),
            "expected label error, got: {}",
            text
        );
    }

    #[tokio::test]
    async fn dispatch_create_relation_missing_source_returns_failure() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("create_relation", args).await;
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("source_id or source_label is required"),
            "expected source_id/source_label failure, got: {}",
            text
        );
    }

    #[tokio::test]
    async fn dispatch_search_memory_graph_missing_query_returns_error() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("search_memory_graph", args).await;
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
    async fn dispatch_retrieve_with_traversal_missing_node_id_returns_error() {
        let server = test_server();
        let args = serde_json::Map::new();
        let result = server.dispatch("retrieve_with_traversal", args).await;
        assert_eq!(result.is_error, Some(true));
        let text = match &result.content[0].raw {
            RawContent::Text(t) => t.text.as_str(),
            _ => panic!("expected text content"),
        };
        assert!(
            text.contains("start_node_id is required"),
            "expected start_node_id error, got: {}",
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
    fn tool_definitions_blast_radius_requires_symbol_name() {
        let defs = tool_definitions();
        let tool = defs
            .iter()
            .find(|t| t.name.as_ref() == "get_blast_radius")
            .unwrap();
        let schema = tool.input_schema.as_ref();
        let required = schema.get("required").and_then(|v| v.as_array()).unwrap();
        let req_strs: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(req_strs.contains(&"symbol_name"));
    }

    #[test]
    fn tool_definitions_propose_commit_requires_file_path_and_content() {
        let defs = tool_definitions();
        let tool = defs
            .iter()
            .find(|t| t.name.as_ref() == "propose_commit")
            .unwrap();
        let schema = tool.input_schema.as_ref();
        let required = schema.get("required").and_then(|v| v.as_array()).unwrap();
        let req_strs: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(req_strs.contains(&"file_path"));
        assert!(req_strs.contains(&"new_content"));
    }

    #[test]
    fn tool_definitions_context_tree_has_no_required_params() {
        let defs = tool_definitions();
        let tool = defs
            .iter()
            .find(|t| t.name.as_ref() == "get_context_tree")
            .unwrap();
        let schema = tool.input_schema.as_ref();
        // get_context_tree has no required params
        assert!(
            schema.get("required").is_none(),
            "get_context_tree should have no required params"
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

    #[test]
    fn server_new_initializes_with_correct_root() {
        let root = PathBuf::from("/tmp/test-root");
        let config = Config::from_env();
        let server = ContextPlusServer::new(root.clone(), config);
        assert_eq!(server.root_dir(), Path::new("/tmp/test-root"));
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

    #[test]
    fn format_unix_epoch() {
        assert_eq!(format_unix_timestamp(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn format_known_timestamp() {
        // 2024-01-15T12:30:45Z = 1705318245
        assert_eq!(format_unix_timestamp(1705318245), "2024-01-15T11:30:45Z");
    }

    #[test]
    fn format_y2k_timestamp() {
        // 2000-01-01T00:00:00Z = 946684800
        assert_eq!(format_unix_timestamp(946684800), "2000-01-01T00:00:00Z");
    }

    #[test]
    fn restore_point_pipe_format() {
        let iso_ts = format_unix_timestamp(1705318245);
        let line = format!(
            "{} | {} | {} | {}\n",
            "rp-123-abc456", iso_ts, "src/main.rs, src/lib.rs", "test restore",
        );
        assert_eq!(
            line,
            "rp-123-abc456 | 2024-01-15T11:30:45Z | src/main.rs, src/lib.rs | test restore\n"
        );
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
        let file_content: HashMap<String, Arc<String>> = files
            .into_iter()
            .map(|(path, content)| (path.to_string(), Arc::new(content.to_string())))
            .collect();
        ProjectCache {
            file_entries: entries,
            file_content,
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
        server.spawn_warmup_task();
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
}
