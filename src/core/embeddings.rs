use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use tokio_util::sync::CancellationToken;

use crate::config::Config;
use crate::error::{ContextPlusError, Result};

/// Type alias for the boxed future returned by embedding functions.
type EmbedFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MIN_EMBED_INPUT_CHARS: usize = 1;
const SINGLE_INPUT_SHRINK_FACTOR: f64 = 0.75;
const MAX_SINGLE_INPUT_RETRIES: usize = 15;

/// Hard wall-clock ceiling for a single embed HTTP call (send + body read).
///
/// Acts as a circuit breaker: reqwest's `Client::timeout` is best-effort and
/// has historically not always covered streamed body reads via `.json()`.
/// On CPU-only Ollama with large dense models (e.g. embeddinggemma at 300M+
/// params), a single 32-string batch can take > 60s, so this needs to be
/// generous — but bounded so a wedged connection cannot stall the warmup
/// binaries indefinitely.
const EMBED_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

// ---------------------------------------------------------------------------
// OllamaClient
// ---------------------------------------------------------------------------

/// HTTP client for Ollama embedding and chat APIs with adaptive batch/retry.
/// Runtime options for Ollama embed requests.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbedRuntimeOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_batch: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low_vram: Option<bool>,
}
impl EmbedRuntimeOptions {
    pub fn from_config(config: &Config) -> Option<Self> {
        let o = Self {
            num_gpu: config.embed_num_gpu,
            main_gpu: config.embed_main_gpu,
            num_thread: config.embed_num_thread,
            num_batch: config.embed_num_batch,
            num_ctx: config.embed_num_ctx,
            low_vram: config.embed_low_vram,
        };
        if o.num_gpu.is_none()
            && o.main_gpu.is_none()
            && o.num_thread.is_none()
            && o.num_batch.is_none()
            && o.num_ctx.is_none()
            && o.low_vram.is_none()
        {
            None
        } else {
            Some(o)
        }
    }
}

// ---------------------------------------------------------------------------
// BoundedLruCache
// ---------------------------------------------------------------------------

/// A bounded LRU cache for query embeddings.
///
/// Uses `IndexMap` for O(1) insertion-ordered storage. Promotion on `get` is
/// O(1): `shift_remove` swaps the target entry to the back by index, then
/// `insert` appends it at the tail. Eviction on `insert` is also O(1):
/// `shift_remove_index(0)` removes the front (oldest) entry in constant time.
/// This replaces the previous `VecDeque::retain` approach which was O(n) on
/// every cache hit and on every duplicate `insert`.
struct BoundedLruCache {
    cap: usize,
    map: indexmap::IndexMap<String, Vec<f32>>,
}

impl BoundedLruCache {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            map: indexmap::IndexMap::with_capacity(cap + 1),
        }
    }

    fn get(&mut self, key: &str) -> Option<&Vec<f32>> {
        // O(1): remove entry by key (swap with tail), re-insert at tail (MRU).
        let val = self.map.shift_remove(key)?;
        self.map.insert(key.to_string(), val);
        self.map.get(key)
    }

    fn insert(&mut self, key: String, val: Vec<f32>) {
        if self.map.contains_key(&key) {
            // O(1): promote existing key to MRU position.
            self.map.shift_remove(&key);
        } else if self.map.len() >= self.cap {
            // O(1): evict LRU (front of insertion-ordered map).
            self.map.shift_remove_index(0);
        }
        self.map.insert(key, val);
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

#[derive(Clone)]
pub struct OllamaClient {
    client: reqwest::Client,
    host: String,
    model: String,
    chat_model: String,
    batch_size: usize,
    query_batch_size: usize,
    embed_options: Option<EmbedRuntimeOptions>,
    embed_chunk_chars: usize,
    cancel_token: CancellationToken,
    request_timeout: std::time::Duration,
    query_cache: Arc<std::sync::Mutex<BoundedLruCache>>,
}

#[derive(serde::Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<&'a EmbedRuntimeOptions>,
    /// Keep model loaded in Ollama indefinitely after this request.
    /// Without this, Ollama evicts the model after 5 min of idle time,
    /// adding a cold-load penalty (~3-5s) to the first query after any gap.
    /// Ollama treats -1 as "keep loaded forever" (numeric seconds).
    keep_alive: i32,
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaClient {
    pub fn new(config: &Config) -> Self {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(4)
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("reqwest client build");
        Self {
            client,
            host: config.ollama_host.clone(),
            model: config.ollama_embed_model.clone(),
            chat_model: config.ollama_chat_model.clone(),
            batch_size: config.embed_batch_size,
            query_batch_size: config.query_batch_size,
            embed_options: EmbedRuntimeOptions::from_config(config),
            embed_chunk_chars: config.embed_chunk_chars,
            cancel_token: CancellationToken::new(),
            request_timeout: EMBED_REQUEST_TIMEOUT,
            query_cache: Arc::new(std::sync::Mutex::new(BoundedLruCache::new(256))),
        }
    }

    /// Return the number of entries in the query embedding LRU cache.
    pub fn query_cache_len(&self) -> usize {
        self.query_cache.lock().unwrap().len()
    }

    /// Override the per-request wall-clock deadline. Mostly useful in tests.
    pub fn with_request_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Cancel all in-flight embedding requests.
    /// The token is shared via Arc internally, so all clones see the cancellation.
    /// After cancellation, new requests will also fail until a fresh client is created.
    /// This is intentional for shutdown scenarios.
    pub fn cancel_all_embeddings(&self) {
        self.cancel_token.cancel();
    }

    /// Embed a slice of texts, returning one vector per text.
    /// Handles chunking of oversized inputs, batching, and adaptive retry.
    ///
    /// For single-text (query) calls the result is served from an in-process
    /// LRU cache (cap=256) on cache hit, skipping Ollama entirely.
    /// Multi-text batch calls (warmup/indexing) bypass the cache.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Cache check: single-text (query) path only.
        if texts.len() == 1 {
            let cached = {
                let mut cache = self.query_cache.lock().unwrap();
                cache.get(&texts[0]).cloned()
            };
            if let Some(vec) = cached {
                return Ok(vec![vec]);
            }
        }

        // Split each input into chunks; short inputs produce a single chunk.
        let chunked_inputs: Vec<Vec<&str>> = texts
            .iter()
            .map(|t| split_embedding_input(t, self.embed_chunk_chars))
            .collect();

        // Flatten all chunks into a single list for batched embedding.
        let flattened: Vec<String> = chunked_inputs
            .iter()
            .flat_map(|chunks| chunks.iter().map(|s| (*s).to_string()))
            .collect();

        // Embed all chunks in batches.
        let mut flat_embeddings = Vec::with_capacity(flattened.len());
        for batch in flattened.chunks(self.batch_size) {
            let batch_result = self.embed_batch_adaptive(batch).await?;
            flat_embeddings.extend(batch_result);
        }

        // Merge chunk embeddings back into one vector per original input.
        let mut results = Vec::with_capacity(texts.len());
        let mut offset = 0;
        for chunks in &chunked_inputs {
            let vectors = &flat_embeddings[offset..offset + chunks.len()];
            let weights: Vec<usize> = chunks.iter().map(|c| c.len()).collect();
            results.push(merge_embedding_vectors(vectors, &weights)?);
            offset += chunks.len();
        }

        // Cache population: single-text (query) path only.
        if texts.len() == 1
            && let Some(v) = results.first()
        {
            let mut cache = self.query_cache.lock().unwrap();
            cache.insert(texts[0].clone(), v.clone());
        }

        Ok(results)
    }

    /// Get the configured batch size (used for warmup/indexing).
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the configured query batch size (used for live search queries, default 1 for CPU-optimal throughput).
    pub fn query_batch_size(&self) -> usize {
        self.query_batch_size
    }

    /// Send a chat completion request to Ollama and return the response content.
    pub async fn chat(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/api/chat", self.host);
        let body = serde_json::json!({
            "model": self.chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": false,
            "think": false,
        });

        let resp = tokio::time::timeout(
            std::time::Duration::from_secs(90),
            self.client.post(&url).json(&body).send(),
        )
        .await
        .map_err(|_| ContextPlusError::Ollama("Chat request timed out after 90s".to_string()))?
        .map_err(|e| ContextPlusError::Ollama(format!("Chat request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ContextPlusError::Ollama(format!(
                "Ollama chat returned {}: {}",
                status, text
            )));
        }

        #[derive(serde::Deserialize)]
        struct ChatMessage {
            content: String,
        }
        #[derive(serde::Deserialize)]
        struct ChatResponse {
            message: ChatMessage,
        }

        let chat_resp: ChatResponse = resp.json().await.map_err(|e| {
            ContextPlusError::Ollama(format!("Failed to parse chat response: {}", e))
        })?;

        Ok(chat_resp.message.content)
    }

    fn embed_batch_adaptive<'a>(&'a self, batch: &'a [String]) -> EmbedFuture<'a> {
        Box::pin(async move {
            match self.call_embed_api(batch).await {
                Ok(embeddings) => {
                    if embeddings.len() != batch.len() {
                        return Err(ContextPlusError::Ollama(format!(
                            "embedding response size mismatch: expected {}, got {}",
                            batch.len(),
                            embeddings.len()
                        )));
                    }
                    Ok(embeddings)
                }
                Err(e) if is_context_length_error(&e) => {
                    if batch.len() == 1 {
                        let vec = self.embed_single_adaptive(&batch[0]).await?;
                        Ok(vec![vec])
                    } else {
                        // Binary split
                        let mid = batch.len().div_ceil(2);
                        let left = self.embed_batch_adaptive(&batch[..mid]).await?;
                        let right = self.embed_batch_adaptive(&batch[mid..]).await?;
                        Ok([left, right].concat())
                    }
                }
                Err(e) => Err(e),
            }
        })
    }

    async fn embed_single_adaptive(&self, input: &str) -> Result<Vec<f32>> {
        let mut candidate = input.to_string();

        for _attempt in 0..=MAX_SINGLE_INPUT_RETRIES {
            match self.call_embed_api(&[candidate.clone()]).await {
                Ok(mut vecs) => {
                    return vecs.pop().ok_or_else(|| {
                        ContextPlusError::Ollama("empty embedding response".into())
                    });
                }
                Err(e) if is_context_length_error(&e) => {
                    let next = shrink_input(&candidate);
                    if next.len() == candidate.len() {
                        return Err(e);
                    }
                    candidate = next;
                }
                Err(e) => return Err(e),
            }
        }
        Err(ContextPlusError::Ollama(
            "unable to embed oversized input after adaptive retries".into(),
        ))
    }

    async fn call_embed_api(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        // Check cancellation before starting the request
        if self.cancel_token.is_cancelled() {
            return Err(ContextPlusError::Cancelled);
        }

        let url = format!("{}/api/embed", self.host);
        let body = EmbedRequest {
            model: &self.model,
            input: inputs,
            options: self.embed_options.as_ref(),
            keep_alive: -1,
        };

        // Cover the WHOLE request lifecycle (send + status check + body read)
        // under a single deadline + cancel race. Previously only `.send()` was
        // raced against the cancel token, which left `response.json().await`
        // unsupervised — a slow/wedged Ollama body read could hang forever,
        // deadlocking the warmup binaries.
        let token = self.cancel_token.clone();
        let request = async {
            let response = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| ContextPlusError::Ollama(format!("request failed: {}", e)))?;

            if !response.status().is_success() {
                let status = response.status();
                let text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "unknown error".to_string());
                return Err(ContextPlusError::Ollama(format!(
                    "HTTP {}: {}",
                    status, text
                )));
            }

            let embed_response: EmbedResponse = response
                .json()
                .await
                .map_err(|e| ContextPlusError::Ollama(format!("response parse error: {}", e)))?;

            Ok(embed_response.embeddings)
        };

        let deadline = self.request_timeout;
        tokio::select! {
            biased;
            _ = token.cancelled() => Err(ContextPlusError::Cancelled),
            result = tokio::time::timeout(deadline, request) => match result {
                Ok(inner) => inner,
                Err(_) => Err(ContextPlusError::Ollama(format!(
                    "embedding request exceeded {}ms wall-clock deadline",
                    deadline.as_millis()
                ))),
            },
        }
    }
}

fn is_context_length_error(err: &ContextPlusError) -> bool {
    let msg = err.to_string().to_lowercase();
    msg.contains("input length exceeds context length")
        || (msg.contains("context") && msg.contains("exceed"))
}

fn shrink_input(input: &str) -> String {
    if input.len() <= MIN_EMBED_INPUT_CHARS {
        return input.to_string();
    }
    let next_len = (input.len() as f64 * SINGLE_INPUT_SHRINK_FACTOR) as usize;
    let next_len = next_len.max(MIN_EMBED_INPUT_CHARS);
    if next_len >= input.len() {
        return crate::core::parser::truncate_to_char_boundary(input, input.len() - 1).to_string();
    }
    crate::core::parser::truncate_to_char_boundary(input, next_len).to_string()
}

// ---------------------------------------------------------------------------
// Embedding chunk + merge
// ---------------------------------------------------------------------------

/// Split text into chunks of at most `chunk_chars` bytes, respecting char boundaries.
/// If the text fits in one chunk, returns a single-element vec (no copy).
pub fn split_embedding_input(text: &str, chunk_chars: usize) -> Vec<&str> {
    let chunk_chars = chunk_chars.max(1);
    if text.len() <= chunk_chars {
        return vec![text];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + chunk_chars).min(text.len());
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            end = start + 1;
            while end < text.len() && !text.is_char_boundary(end) {
                end += 1;
            }
        }
        chunks.push(&text[start..end]);
        start = end;
    }
    chunks
}

/// Weighted average of multiple embedding vectors.
/// `weights` are typically the character counts of each chunk.
/// Returns the merged vector with the same dimensionality as the inputs.
pub fn merge_embedding_vectors(vectors: &[Vec<f32>], weights: &[usize]) -> Result<Vec<f32>> {
    if vectors.is_empty() {
        return Err(ContextPlusError::Ollama(
            "Cannot merge empty embedding vectors".into(),
        ));
    }
    if vectors.len() == 1 {
        return Ok(vectors[0].clone());
    }
    let dim = vectors[0].len();
    let mut merged = vec![0.0f32; dim];
    let mut total_weight: f64 = 0.0;

    for (i, vector) in vectors.iter().enumerate() {
        if vector.len() != dim {
            return Err(ContextPlusError::Ollama(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                dim,
                vector.len()
            )));
        }
        let w = (*weights.get(i).unwrap_or(&1)).max(1) as f64;
        total_weight += w;
        for (d, val) in vector.iter().enumerate() {
            merged[d] += val * w as f32;
        }
    }

    if total_weight > 0.0 {
        let inv = 1.0 / total_weight as f32;
        for v in &mut merged {
            *v *= inv;
        }
    }

    Ok(merged)
}

// ---------------------------------------------------------------------------
// VectorData — owned vs mmap-backed vector storage
// ---------------------------------------------------------------------------

/// Backing storage for the flat f32 vector array.
///
/// `Owned` holds a regular `Vec<f32>` (heap-allocated copy).
/// `Mmap` points directly into an mmap'd rkyv cache file — true zero-copy.
/// The `Arc<memmap2::Mmap>` keeps the mapping alive while the pointer is live.
enum VectorData {
    Owned(Vec<f32>),
    Mmap {
        _mmap: Arc<memmap2::Mmap>,
        ptr: *const f32,
        len: usize,
    },
}

impl VectorData {
    fn as_slice(&self) -> &[f32] {
        match self {
            VectorData::Owned(v) => v,
            VectorData::Mmap { ptr, len, .. } => {
                // SAFETY: ptr was derived from a valid, aligned &[f32] inside a
                // live Mmap (kept alive by the Arc). The data is f32 on a
                // little-endian platform where rkyv's f32_le == f32.
                unsafe { std::slice::from_raw_parts(*ptr, *len) }
            }
        }
    }
}

// SAFETY: The Mmap is immutable (read-only mapping) and the Arc keeps it alive.
// The raw pointer is derived from the Mmap and only read through as_slice().
unsafe impl Send for VectorData {}
unsafe impl Sync for VectorData {}

// ---------------------------------------------------------------------------
// VectorStore
// ---------------------------------------------------------------------------

/// Threshold above which `find_nearest` dispatches to HNSW instead of brute force.
const HNSW_THRESHOLD: usize = 2000;

/// A heap-allocated f32 vector that satisfies the `instant_distance::Point` trait
/// using cosine *distance* (1.0 − cosine_similarity).
#[derive(Clone)]
struct HnswPoint(Vec<f32>);

impl instant_distance::Point for HnswPoint {
    fn distance(&self, other: &Self) -> f32 {
        // instant-distance minimises distance; cosine distance = 1 − cosine_similarity
        let sim = cosine_similarity_simsimd(&self.0, &other.0);
        // Clamp to [0, 2] — cosine similarity is in [-1, 1]
        (1.0 - sim).clamp(0.0, 2.0)
    }
}

/// Lazily-built HNSW map: maps HnswPoint → original key index (usize stored as String).
type HnswIndex = instant_distance::HnswMap<HnswPoint, usize>;

/// Runtime-tunable HNSW parameters.
///
/// Populated from [`crate::config::Config`] fields; defaults mirror the
/// `instant-distance` crate defaults so existing behaviour is preserved.
///
/// Note: the `M` parameter (bi-directional links per layer) in
/// `instant-distance 0.6.1` is a private compile-time `const M: usize = 32`
/// and is not exposed through the public `Builder` API.  The
/// `CONTEXTPLUS_HNSW_M` knob is therefore not implemented.
#[derive(Debug, Clone, Copy)]
pub struct HnswTuning {
    /// `efConstruction` — quality/speed trade-off at index build time.
    pub ef_construction: usize,
    /// `ef_search` — recall/latency trade-off at query time.
    pub ef_search: usize,
}

impl Default for HnswTuning {
    fn default() -> Self {
        Self {
            ef_construction: crate::config::DEFAULT_HNSW_EF_CONSTRUCTION,
            ef_search: crate::config::DEFAULT_HNSW_EF_SEARCH,
        }
    }
}

impl HnswTuning {
    /// Build a tuning snapshot from the current `Config`'s env-var-parsed values.
    /// This is the intended bridge from operator-configured env knobs
    /// (`CONTEXTPLUS_HNSW_EF_CONSTRUCTION`, `CONTEXTPLUS_HNSW_EF_SEARCH`) to
    /// runtime `VectorStore` / `SearchIndex` construction.
    pub fn from_config(config: &crate::config::Config) -> Self {
        Self {
            ef_construction: config.hnsw_ef_construction,
            ef_search: config.hnsw_ef_search,
        }
    }

    /// Return a process-wide tuning, parsed from env vars on first call.
    /// Runtime code that constructs a default `SearchIndex` / `VectorStore`
    /// should prefer this over `HnswTuning::default()` so operator env overrides
    /// actually take effect. Tests and isolated callers can still use
    /// `HnswTuning::default()` or pass an explicit tuning.
    pub fn global() -> Self {
        static GLOBAL: std::sync::OnceLock<HnswTuning> = std::sync::OnceLock::new();
        *GLOBAL.get_or_init(|| HnswTuning::from_config(&crate::config::Config::load()))
    }
}

/// In-memory flat vector store with cosine similarity search via simsimd.
/// Uses brute-force SIMD+rayon scan for stores ≤ `HNSW_THRESHOLD` vectors;
/// lazily builds an HNSW approximate nearest-neighbor index above that threshold.
pub struct VectorStore {
    dims: u32,
    count: u32,
    vectors: VectorData,
    keys: Vec<String>,
    hashes: Vec<String>,
    key_index: HashMap<String, usize>,
    /// Lazily-built HNSW index, constructed on first call to `find_nearest_hnsw`.
    hnsw_index: OnceLock<HnswIndex>,
    /// HNSW build/search tuning knobs (from env-var config).
    hnsw_tuning: HnswTuning,
}

impl VectorStore {
    /// Build a VectorStore from parallel arrays of keys, hashes, and vectors.
    pub fn new(dims: u32, keys: Vec<String>, hashes: Vec<String>, vectors: Vec<f32>) -> Self {
        Self::new_with_tuning(dims, keys, hashes, vectors, HnswTuning::default())
    }

    /// Like [`Self::new`] but with explicit HNSW tuning knobs.
    pub fn new_with_tuning(
        dims: u32,
        keys: Vec<String>,
        hashes: Vec<String>,
        vectors: Vec<f32>,
        hnsw_tuning: HnswTuning,
    ) -> Self {
        let count = keys.len() as u32;
        let mut key_index = HashMap::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            key_index.insert(key.clone(), i);
        }
        Self {
            dims,
            count,
            vectors: VectorData::Owned(vectors),
            keys,
            hashes,
            key_index,
            hnsw_index: OnceLock::new(),
            hnsw_tuning,
        }
    }

    /// Build a VectorStore with vector data backed by an mmap'd file (zero-copy).
    ///
    /// `mmap` must be kept alive for the lifetime of this VectorStore.
    /// `vectors_ptr` must point to a valid, aligned `&[f32]` region inside `mmap`.
    /// `vectors_len` is the number of f32 elements (not bytes).
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `vectors_ptr` points into the `mmap` region
    /// - The pointer is aligned to `align_of::<f32>()`
    /// - `vectors_len` f32 values are readable at that address
    /// - The platform is little-endian (so rkyv's f32_le == native f32)
    pub unsafe fn from_mmap(
        dims: u32,
        keys: Vec<String>,
        hashes: Vec<String>,
        vectors_ptr: *const f32,
        vectors_len: usize,
        mmap: Arc<memmap2::Mmap>,
    ) -> Self {
        // SAFETY: caller upholds the same invariants required by from_mmap_with_tuning.
        unsafe {
            Self::from_mmap_with_tuning(
                dims,
                keys,
                hashes,
                vectors_ptr,
                vectors_len,
                mmap,
                HnswTuning::default(),
            )
        }
    }

    /// Like [`Self::from_mmap`] but with explicit HNSW tuning knobs.
    ///
    /// # Safety
    /// Same requirements as [`Self::from_mmap`].
    pub unsafe fn from_mmap_with_tuning(
        dims: u32,
        keys: Vec<String>,
        hashes: Vec<String>,
        vectors_ptr: *const f32,
        vectors_len: usize,
        mmap: Arc<memmap2::Mmap>,
        hnsw_tuning: HnswTuning,
    ) -> Self {
        // Debug-mode bounds check: pointer must lie within the mmap region.
        debug_assert!(
            {
                let mmap_start = mmap.as_ptr() as usize;
                let mmap_end = mmap_start + mmap.len();
                let ptr_addr = vectors_ptr as usize;
                let ptr_end = ptr_addr + vectors_len * std::mem::size_of::<f32>();
                ptr_addr >= mmap_start && ptr_end <= mmap_end
            },
            "mmap vectors_ptr out of bounds"
        );
        let count = keys.len() as u32;
        let mut key_index = HashMap::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            key_index.insert(key.clone(), i);
        }
        Self {
            dims,
            count,
            vectors: VectorData::Mmap {
                _mmap: mmap,
                ptr: vectors_ptr,
                len: vectors_len,
            },
            keys,
            hashes,
            key_index,
            hnsw_index: OnceLock::new(),
            hnsw_tuning,
        }
    }

    /// Build from an EmbeddingCache (HashMap of path -> (hash, vector)).
    /// Single pass over the cache — no redundant HashMap lookups.
    pub fn from_cache(cache: &HashMap<String, CacheEntry>) -> Option<Self> {
        Self::from_cache_with_tuning(cache, HnswTuning::default())
    }

    /// Like [`Self::from_cache`] but with explicit HNSW tuning knobs.
    pub fn from_cache_with_tuning(
        cache: &HashMap<String, CacheEntry>,
        hnsw_tuning: HnswTuning,
    ) -> Option<Self> {
        if cache.is_empty() {
            return None;
        }
        let n = cache.len();
        let dims = cache.values().next()?.vector.len() as u32;
        let mut keys = Vec::with_capacity(n);
        let mut hashes = Vec::with_capacity(n);
        let mut vectors = Vec::with_capacity(n * dims as usize);
        for (key, entry) in cache {
            keys.push(key.clone());
            hashes.push(entry.hash.clone());
            vectors.extend_from_slice(&entry.vector);
        }
        Some(Self::new_with_tuning(
            dims,
            keys,
            hashes,
            vectors,
            hnsw_tuning,
        ))
    }

    /// Number of vectors stored.
    pub fn count(&self) -> usize {
        self.count as usize
    }

    /// Vector dimensions.
    pub fn dims(&self) -> usize {
        self.dims as usize
    }

    /// Get the content hash for a key.
    pub fn get_hash(&self, key: &str) -> Option<&str> {
        self.key_index
            .get(key)
            .map(|&idx| self.hashes[idx].as_str())
    }

    /// Check if a key exists.
    pub fn has_key(&self, key: &str) -> bool {
        self.key_index.contains_key(key)
    }

    /// Get a vector by key.
    pub fn get_vector(&self, key: &str) -> Option<&[f32]> {
        let vectors = self.vectors.as_slice();
        self.key_index.get(key).map(|&idx| {
            let offset = idx * self.dims as usize;
            &vectors[offset..offset + self.dims as usize]
        })
    }

    /// Get a key by index.
    pub fn key_at(&self, idx: usize) -> Option<&str> {
        self.keys.get(idx).map(|s| s.as_str())
    }

    /// Get all keys.
    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    /// Get all hashes.
    pub fn hashes(&self) -> &[String] {
        &self.hashes
    }

    /// Get raw vectors data.
    pub fn vectors_data(&self) -> &[f32] {
        self.vectors.as_slice()
    }

    /// Find the top-k nearest neighbors by cosine similarity.
    ///
    /// Dispatches to HNSW for stores > `HNSW_THRESHOLD` vectors (lazy index build on first call),
    /// or exact brute-force otherwise.
    /// Returns (key, similarity) pairs sorted by descending similarity.
    pub fn find_nearest(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        if self.count == 0 || query.len() != self.dims as usize {
            return Vec::new();
        }

        if self.count as usize > HNSW_THRESHOLD {
            self.find_nearest_hnsw(query, top_k)
        } else {
            self.find_nearest_brute_force(query, top_k)
        }
    }

    /// Approximate nearest-neighbor search using a lazily-built HNSW index.
    ///
    /// The index is constructed once on the first call and stored in `hnsw_index`
    /// via `OnceLock`. Subsequent calls reuse the same index with no locking overhead.
    ///
    /// Returns (key, cosine_similarity) pairs sorted by descending similarity.
    /// For stores smaller than a handful of vectors the results may not be perfectly
    /// ranked (ANN trade-off), but for large repos (>2K files) quality is high.
    pub fn find_nearest_hnsw(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        if self.count == 0 || query.len() != self.dims as usize || top_k == 0 {
            return Vec::new();
        }

        // Build HNSW index lazily — only once for the lifetime of this VectorStore.
        let tuning = self.hnsw_tuning;
        let index = self.hnsw_index.get_or_init(|| {
            let vectors = self.vectors.as_slice();
            let dims = self.dims as usize;
            let n = self.count as usize;

            let points: Vec<HnswPoint> = (0..n)
                .map(|i| {
                    let offset = i * dims;
                    HnswPoint(vectors[offset..offset + dims].to_vec())
                })
                .collect();
            // values[i] = original index i — lets us map PointId → key
            let values: Vec<usize> = (0..n).collect();

            instant_distance::Builder::default()
                .ef_construction(tuning.ef_construction)
                .ef_search(tuning.ef_search)
                .build(points, values)
        });

        let query_point = HnswPoint(query.to_vec());
        let mut search = instant_distance::Search::default();

        // Collect results; HnswMap::search returns items in distance order (nearest first).
        // We convert cosine distance back to cosine similarity.
        let mut results: Vec<(String, f32)> = index
            .search(&query_point, &mut search)
            .take(top_k)
            .map(|item| {
                let original_idx = *item.value;
                let cosine_sim = 1.0 - item.distance; // distance = 1 - similarity
                (self.keys[original_idx].clone(), cosine_sim)
            })
            .collect();

        // Ensure descending similarity order (HNSW returns ascending distance, i.e. descending sim)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Brute-force exact nearest neighbor search with SIMD cosine similarity.
    /// Uses sequential scan for <2K vectors (avoids rayon overhead), parallel for larger stores.
    pub fn find_nearest_brute_force(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let vectors = self.vectors.as_slice();
        let dims = self.dims as usize;
        let count = self.count as usize;

        const PARALLEL_THRESHOLD: usize = 2000;

        let mut scored: Vec<(usize, f32)> = if count >= PARALLEL_THRESHOLD {
            use rayon::prelude::*;
            (0..count)
                .into_par_iter()
                .map(|i| {
                    let offset = i * dims;
                    let stored = &vectors[offset..offset + dims];
                    (i, cosine_similarity_simsimd(query, stored))
                })
                .collect()
        } else {
            (0..count)
                .map(|i| {
                    let offset = i * dims;
                    let stored = &vectors[offset..offset + dims];
                    (i, cosine_similarity_simsimd(query, stored))
                })
                .collect()
        };

        // Partial sort: partition top_k elements in O(n) average, then sort only those.
        let top_k = top_k.min(scored.len());
        if top_k == 0 {
            return Vec::new();
        }
        scored.select_nth_unstable_by(top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .map(|(idx, sim)| (self.keys[idx].clone(), sim))
            .collect()
    }

    /// Cosine similarity for a specific key.
    pub fn cosine_by_key(&self, query: &[f32], key: &str) -> f32 {
        match self.key_index.get(key) {
            Some(&idx) => {
                let vectors = self.vectors.as_slice();
                let offset = idx * self.dims as usize;
                let stored = &vectors[offset..offset + self.dims as usize];
                cosine_similarity_simsimd(query, stored)
            }
            None => 0.0,
        }
    }

    /// Convert to a cache map.
    pub fn to_cache(&self) -> HashMap<String, CacheEntry> {
        let vectors = self.vectors.as_slice();
        let mut cache = HashMap::with_capacity(self.count as usize);
        for i in 0..self.count as usize {
            let offset = i * self.dims as usize;
            cache.insert(
                self.keys[i].clone(),
                CacheEntry {
                    hash: self.hashes[i].clone(),
                    vector: vectors[offset..offset + self.dims as usize].to_vec(),
                },
            );
        }
        cache
    }
}

/// A cache entry for a single embedding.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub hash: String,
    pub vector: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Cosine similarity via simsimd
// ---------------------------------------------------------------------------

/// Compute cosine similarity using simsimd SIMD acceleration.
/// simsimd returns cosine *distance* (0 = identical, 2 = opposite).
/// We convert to similarity: 1.0 - distance.
pub fn cosine_similarity_simsimd(a: &[f32], b: &[f32]) -> f32 {
    use simsimd::SpatialSimilarity;
    match f32::cosine(a, b) {
        Some(distance) => 1.0 - distance as f32,
        None => {
            tracing::warn!(
                a_len = a.len(),
                b_len = b.len(),
                "simsimd cosine returned None"
            );
            0.0
        }
    }
}

/// Fallback pure-Rust cosine similarity (for testing/comparison).
pub fn cosine_similarity_naive(a: &[f32], b: &[f32]) -> f32 {
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

// ---------------------------------------------------------------------------
// Hash content re-export for embedding cache invalidation
// ---------------------------------------------------------------------------

pub use crate::core::parser::hash_content as content_hash;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::parser::hash_content;

    // -- cosine similarity tests --

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity_simsimd(&a, &b);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "identical vectors should have similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity_simsimd(&a, &b);
        assert!(
            sim.abs() < 0.01,
            "orthogonal vectors should have similarity ~0.0, got {}",
            sim
        );
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity_simsimd(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 0.01,
            "opposite vectors should have similarity ~-1.0, got {}",
            sim
        );
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity_simsimd(&a, &b);
        // With zero vector, distance may be NaN → we return 0.0
        assert!(
            sim.abs() < 0.01 || sim.is_nan(),
            "zero vector similarity should be ~0.0 or NaN, got {}",
            sim
        );
    }

    #[test]
    fn cosine_simsimd_matches_naive() {
        let a = vec![0.5, 0.3, 0.8, 0.1];
        let b = vec![0.2, 0.9, 0.4, 0.6];
        let sim_simd = cosine_similarity_simsimd(&a, &b);
        let sim_naive = cosine_similarity_naive(&a, &b);
        assert!(
            (sim_simd - sim_naive).abs() < 0.01,
            "simsimd={} vs naive={}",
            sim_simd,
            sim_naive
        );
    }

    // -- VectorStore tests --

    fn make_store() -> VectorStore {
        let keys = vec![
            "src/auth.ts".to_string(),
            "src/db.ts".to_string(),
            "src/api.ts".to_string(),
        ];
        let hashes = vec!["h1".to_string(), "h2".to_string(), "h3".to_string()];
        // 3D vectors
        let vectors = vec![
            0.9, 0.1, 0.0, // auth: close to query [1,0,0]
            0.0, 0.9, 0.1, // db: orthogonal
            0.5, 0.5, 0.0, // api: middle ground
        ];
        VectorStore::new(3, keys, hashes, vectors)
    }

    #[test]
    fn vector_store_count_and_dims() {
        let store = make_store();
        assert_eq!(store.count(), 3);
        assert_eq!(store.dims(), 3);
    }

    #[test]
    fn vector_store_has_key() {
        let store = make_store();
        assert!(store.has_key("src/auth.ts"));
        assert!(!store.has_key("src/missing.ts"));
    }

    #[test]
    fn vector_store_get_hash() {
        let store = make_store();
        assert_eq!(store.get_hash("src/auth.ts"), Some("h1"));
        assert_eq!(store.get_hash("src/db.ts"), Some("h2"));
        assert_eq!(store.get_hash("nonexistent"), None);
    }

    #[test]
    fn vector_store_get_vector() {
        let store = make_store();
        let vec = store.get_vector("src/auth.ts").unwrap();
        assert_eq!(vec.len(), 3);
        assert!((vec[0] - 0.9).abs() < 0.001);
    }

    #[test]
    fn vector_store_find_nearest_ordering() {
        let store = make_store();
        let query = vec![1.0, 0.0, 0.0];
        let results = store.find_nearest(&query, 3);

        assert_eq!(results.len(), 3);
        // auth (0.9, 0.1, 0.0) should be closest to (1, 0, 0)
        assert_eq!(results[0].0, "src/auth.ts");
        // Similarity should be high
        assert!(results[0].1 > 0.9);
    }

    #[test]
    fn vector_store_find_nearest_top_k() {
        let store = make_store();
        let query = vec![1.0, 0.0, 0.0];
        let results = store.find_nearest(&query, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "src/auth.ts");
    }

    #[test]
    fn vector_store_find_nearest_empty() {
        let store = VectorStore::new(3, vec![], vec![], vec![]);
        let results = store.find_nearest(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn vector_store_find_nearest_wrong_dims() {
        let store = make_store();
        let results = store.find_nearest(&[1.0, 0.0], 5); // wrong dims
        assert!(results.is_empty());
    }

    #[test]
    fn vector_store_cosine_by_key() {
        let store = make_store();
        let query = vec![1.0, 0.0, 0.0];
        let sim = store.cosine_by_key(&query, "src/auth.ts");
        assert!(sim > 0.9);

        let sim_missing = store.cosine_by_key(&query, "nonexistent");
        assert_eq!(sim_missing, 0.0);
    }

    #[test]
    fn vector_store_round_trip_cache() {
        let store = make_store();
        let cache = store.to_cache();
        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key("src/auth.ts"));

        let rebuilt = VectorStore::from_cache(&cache).unwrap();
        assert_eq!(rebuilt.count(), 3);
        assert_eq!(rebuilt.dims(), 3);

        // Verify vectors match
        let vec = rebuilt.get_vector("src/auth.ts").unwrap();
        assert!((vec[0] - 0.9).abs() < 0.001);
    }

    #[test]
    fn vector_store_from_empty_cache() {
        let cache = HashMap::new();
        assert!(VectorStore::from_cache(&cache).is_none());
    }

    // -- shrink_input tests --

    #[test]
    fn shrink_input_reduces_length() {
        let input = "a".repeat(1000);
        let shrunk = shrink_input(&input);
        assert!(shrunk.len() < input.len());
        assert!(shrunk.len() >= MIN_EMBED_INPUT_CHARS);
    }

    #[test]
    fn shrink_input_minimum_floor() {
        let input = "a".repeat(MIN_EMBED_INPUT_CHARS);
        let shrunk = shrink_input(&input);
        assert_eq!(shrunk.len(), MIN_EMBED_INPUT_CHARS);
    }

    #[test]
    fn shrink_input_at_minimum_stays() {
        // With MIN_EMBED_INPUT_CHARS=1, a single char should not shrink further
        let input = "a";
        let shrunk = shrink_input(input);
        assert_eq!(shrunk, "a");
    }

    // -- hash_content for cache invalidation --

    #[test]
    fn content_hash_works() {
        let h1 = hash_content("hello world");
        let h2 = hash_content("hello world");
        assert_eq!(h1, h2);

        let h3 = hash_content("different");
        assert_ne!(h1, h3);
    }

    // -- OllamaClient construction --

    #[test]
    fn ollama_client_respects_config() {
        let mut config = Config::from_env();
        config.ollama_host = "http://test:1234".to_string();
        config.ollama_embed_model = "test-model".to_string();
        config.ollama_chat_model = "test-chat".to_string();
        config.embed_batch_size = 25;

        let client = OllamaClient::new(&config);
        assert_eq!(client.host, "http://test:1234");
        assert_eq!(client.model, "test-model");
        assert_eq!(client.chat_model, "test-chat");
        assert_eq!(client.batch_size, 25);
    }

    // -- OllamaClient::chat tests (wiremock) --

    fn config_with_host(host: &str) -> Config {
        let mut config = Config::from_env();
        config.ollama_host = host.to_string();
        config.ollama_chat_model = "test-chat-model".to_string();
        config
    }

    #[tokio::test]
    async fn chat_success_extracts_content() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": { "content": "Hello from LLM" }
            })))
            .mount(&server)
            .await;

        let client = OllamaClient::new(&config_with_host(&server.uri()));
        let result = client.chat("test prompt").await;
        assert_eq!(result.unwrap(), "Hello from LLM");
    }

    #[tokio::test]
    async fn chat_error_on_non_200() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let client = OllamaClient::new(&config_with_host(&server.uri()));
        let result = client.chat("test prompt").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("500"),
            "error should mention status code, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn chat_sends_configured_model() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": { "content": "ok" }
            })))
            .mount(&server)
            .await;

        let mut config = config_with_host(&server.uri());
        config.ollama_chat_model = "my-specific-model".to_string();
        let client = OllamaClient::new(&config);
        let _ = client.chat("hello").await;

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["model"], "my-specific-model");
    }

    // -- OllamaClient::embed tests --

    #[tokio::test]
    async fn embed_empty_input_returns_empty() {
        // No HTTP call should be made for empty input
        let mut config = Config::from_env();
        config.ollama_host = "http://localhost:0".to_string(); // unreachable port
        let client = OllamaClient::new(&config);
        let result = client.embed(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    // -- shrink_input UTF-8 safety --

    #[test]
    fn shrink_input_utf8_multibyte_safety() {
        // Build a string with multi-byte UTF-8 characters exceeding MIN_EMBED_INPUT_CHARS
        // Each emoji is 4 bytes; we need enough to exceed the minimum
        let emoji = "\u{1F600}"; // grinning face, 4 bytes
        let count = (MIN_EMBED_INPUT_CHARS / emoji.len()) + 50;
        let input: String = emoji.repeat(count);
        assert!(input.len() > MIN_EMBED_INPUT_CHARS);

        let shrunk = shrink_input(&input);
        assert!(shrunk.len() < input.len());
        // Must be valid UTF-8 (would panic on invalid)
        assert!(std::str::from_utf8(shrunk.as_bytes()).is_ok());
        // Must not split a multi-byte char: length should be multiple of 4
        assert_eq!(
            shrunk.len() % 4,
            0,
            "shrunk length {} should be a multiple of 4 (emoji bytes)",
            shrunk.len()
        );
    }

    #[test]
    fn shrink_input_mixed_utf8() {
        // Mix of ASCII and multi-byte: "a" (1 byte) + "\u{00E9}" (2 bytes) + "\u{1F600}" (4 bytes)
        let base = "a\u{00E9}\u{1F600}"; // 7 bytes per unit
        let count = (MIN_EMBED_INPUT_CHARS / base.len()) + 20;
        let input: String = base.repeat(count);
        assert!(input.len() > MIN_EMBED_INPUT_CHARS);

        let shrunk = shrink_input(&input);
        assert!(shrunk.len() < input.len());
        // Validate it's still valid UTF-8
        assert!(std::str::from_utf8(shrunk.as_bytes()).is_ok());
    }

    // -- is_context_length_error tests --

    #[test]
    fn detects_context_length_error_exact() {
        let err = ContextPlusError::Ollama("input length exceeds context length".into());
        assert!(is_context_length_error(&err));
    }

    #[test]
    fn detects_context_length_error_variant() {
        let err = ContextPlusError::Ollama("context window exceeded for model".into());
        assert!(is_context_length_error(&err));
    }

    #[test]
    fn non_context_error_not_detected() {
        let err = ContextPlusError::Ollama("connection refused".into());
        assert!(!is_context_length_error(&err));
    }

    #[test]
    fn io_error_not_context_length() {
        let err = ContextPlusError::Io(std::io::Error::other("context exceeded"));
        // The error message contains "context" and "exceed" but check it via to_string
        // IO error wraps differently: "IO error: context exceeded"
        // is_context_length_error checks to_string().to_lowercase()
        // This should still match because the string contains both words
        let result = is_context_length_error(&err);
        // The Display impl prepends "IO error: " so the lowercase string is
        // "io error: context exceeded" which contains "context" and "exceed"
        assert!(result);
    }

    // -- cosine_similarity_naive additional tests --

    #[test]
    fn naive_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity_naive(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "orthogonal vectors should have similarity ~0.0, got {}",
            sim
        );
    }

    #[test]
    fn naive_cosine_parallel() {
        let a = vec![3.0, 4.0];
        let b = vec![6.0, 8.0]; // same direction, different magnitude
        let sim = cosine_similarity_naive(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "parallel vectors should have similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn naive_cosine_anti_parallel() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity_naive(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "anti-parallel vectors should have similarity ~-1.0, got {}",
            sim
        );
    }

    #[test]
    fn naive_cosine_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let sim = cosine_similarity_naive(&a, &b);
        assert_eq!(sim, 0.0, "empty vectors should return 0.0");
    }

    #[test]
    fn naive_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity_naive(&a, &b);
        assert_eq!(sim, 0.0, "zero vector should return 0.0");
    }

    #[test]
    fn ero_none() {
        let mut c = Config::from_env();
        c.embed_num_gpu = None;
        c.embed_main_gpu = None;
        c.embed_num_thread = None;
        c.embed_num_batch = None;
        c.embed_num_ctx = None;
        c.embed_low_vram = None;
        assert!(EmbedRuntimeOptions::from_config(&c).is_none());
    }
    #[test]
    fn ero_some() {
        let mut c = Config::from_env();
        c.embed_num_gpu = Some(1);
        c.embed_main_gpu = None;
        c.embed_num_thread = None;
        c.embed_num_batch = None;
        c.embed_num_ctx = None;
        c.embed_low_vram = None;
        assert_eq!(
            EmbedRuntimeOptions::from_config(&c).unwrap().num_gpu,
            Some(1)
        );
    }
    #[test]
    fn ero_ser() {
        let o = EmbedRuntimeOptions {
            num_gpu: Some(1),
            main_gpu: None,
            num_thread: None,
            num_batch: None,
            num_ctx: Some(2048),
            low_vram: Some(true),
        };
        let j = serde_json::to_value(&o).unwrap();
        assert_eq!(j["num_gpu"], 1);
        assert!(j.get("main_gpu").is_none());
    }
    #[test]
    fn req_no_opts() {
        let r = EmbedRequest {
            model: "t",
            input: &[],
            options: None,
            keep_alive: -1,
        };
        let v = serde_json::to_value(&r).unwrap();
        assert!(v.get("options").is_none());
        assert_eq!(v["keep_alive"], -1);
    }
    #[test]
    fn req_with_opts() {
        let o = EmbedRuntimeOptions {
            num_gpu: Some(2),
            main_gpu: Some(0),
            num_thread: None,
            num_batch: None,
            num_ctx: None,
            low_vram: None,
        };
        let i = vec!["hi".into()];
        let r = EmbedRequest {
            model: "t",
            input: &i,
            options: Some(&o),
            keep_alive: -1,
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["options"]["num_gpu"], 2);
        assert_eq!(v["keep_alive"], -1);
    }

    // -- adaptive retry constant tests --

    #[test]
    fn constants_match_typescript_reference() {
        assert_eq!(MIN_EMBED_INPUT_CHARS, 1);
        assert_eq!(MAX_SINGLE_INPUT_RETRIES, 15);
        assert!((SINGLE_INPUT_SHRINK_FACTOR - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn shrink_input_iterates_toward_minimum() {
        // Start with a smaller input that converges within MAX_SINGLE_INPUT_RETRIES (15)
        // at 0.75x shrink factor. 100 chars needs ~16 iterations, so use 50 (needs ~12).
        let mut input = "x".repeat(50);
        let mut iterations = 0;
        while input.len() > MIN_EMBED_INPUT_CHARS {
            let next = shrink_input(&input);
            if next.len() == input.len() {
                break;
            }
            input = next;
            iterations += 1;
            assert!(iterations <= MAX_SINGLE_INPUT_RETRIES);
        }
        assert_eq!(input.len(), MIN_EMBED_INPUT_CHARS);
    }

    // -- adaptive embed_single_adaptive with wiremock --

    #[tokio::test]
    async fn embed_single_adaptive_shrinks_on_context_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let call_count = Arc::new(AtomicUsize::new(0));
        let counter = call_count.clone();

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |_req: &wiremock::Request| {
                let n = counter.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    ResponseTemplate::new(400)
                        .set_body_string("input length exceeds context length")
                } else {
                    ResponseTemplate::new(200)
                        .set_body_json(serde_json::json!({"embeddings": [[0.1, 0.2, 0.3]]}))
                }
            })
            .mount(&server)
            .await;

        let config = config_with_host(&server.uri());
        let client = OllamaClient::new(&config);
        let input = "a".repeat(500);
        let result = client.embed_single_adaptive(&input).await;

        assert!(result.is_ok(), "should succeed after adaptive shrinking");
        assert_eq!(result.unwrap(), vec![0.1, 0.2, 0.3]);
        assert!(call_count.load(Ordering::SeqCst) >= 3);
    }

    // -- adaptive embed_batch_adaptive with wiremock --

    #[tokio::test]
    async fn embed_batch_adaptive_splits_on_context_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let call_count = Arc::new(AtomicUsize::new(0));
        let counter = call_count.clone();

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |req: &wiremock::Request| {
                let n = counter.fetch_add(1, Ordering::SeqCst);
                let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
                let input_count = body["input"].as_array().unwrap().len();
                if n == 0 && input_count > 2 {
                    ResponseTemplate::new(400)
                        .set_body_string("input length exceeds context length")
                } else {
                    let embeddings: Vec<Vec<f32>> =
                        (0..input_count).map(|i| vec![i as f32, 0.0, 0.0]).collect();
                    ResponseTemplate::new(200)
                        .set_body_json(serde_json::json!({"embeddings": embeddings}))
                }
            })
            .mount(&server)
            .await;

        let config = config_with_host(&server.uri());
        let client = OllamaClient::new(&config);
        let texts: Vec<String> = (0..4).map(|i| format!("text_{}", i)).collect();
        let result = client.embed_batch_adaptive(&texts).await;

        assert!(result.is_ok(), "batch should succeed after splitting");
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 4);
    }

    // -- cancellation tests --

    #[tokio::test]
    async fn cancellation_stops_embed_requests() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings": [[1.0, 2.0, 3.0]]}))
                    .set_delay(std::time::Duration::from_secs(30)),
            )
            .mount(&server)
            .await;

        let config = config_with_host(&server.uri());
        let client = OllamaClient::new(&config);
        let texts = vec!["hello world".to_string()];

        let client_clone = client.clone();
        let handle = tokio::spawn(async move { client_clone.embed(&texts).await });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        client.cancel_all_embeddings();

        let result = handle.await.unwrap();
        assert!(result.is_err());
        assert!(
            matches!(result, Err(ContextPlusError::Cancelled)),
            "error should be Cancelled variant"
        );
    }

    #[tokio::test]
    async fn embed_request_timeout_returns_ollama_error_within_bound() {
        // Regression for the warmup deadlock: prior to the fix, only `.send()`
        // was raced against the cancel token, leaving `response.json().await`
        // unsupervised. A slow/wedged Ollama body read could hang forever.
        // After the fix, the whole send+body lifecycle is bounded by
        // `with_request_timeout`.
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings": [[1.0, 2.0, 3.0]]}))
                    // Simulate a wedged Ollama: response would arrive far past the
                    // configured request_timeout. With the deadline race in place
                    // we should bail with an Ollama error in ~200ms, not hang for
                    // the full delay.
                    .set_delay(std::time::Duration::from_secs(30)),
            )
            .mount(&server)
            .await;

        let config = config_with_host(&server.uri());
        let client =
            OllamaClient::new(&config).with_request_timeout(std::time::Duration::from_millis(200));

        let start = std::time::Instant::now();
        let result = client.embed(&["hello".to_string()]).await;
        let elapsed = start.elapsed();

        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "should bail near the deadline, took {:?}",
            elapsed
        );
        match result {
            Err(ContextPlusError::Ollama(msg)) => {
                assert!(
                    msg.contains("deadline"),
                    "expected deadline error, got: {}",
                    msg
                );
            }
            other => panic!("expected Ollama deadline error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn cancellation_before_request_returns_cancelled() {
        let mut config = Config::from_env();
        config.ollama_host = "http://localhost:0".to_string();
        let client = OllamaClient::new(&config);

        client.cancel_all_embeddings();

        let result = client.embed(&["test".to_string()]).await;
        assert!(result.is_err());
        assert!(matches!(result, Err(ContextPlusError::Cancelled)));
    }

    // -- LRU cache tests --

    #[tokio::test]
    async fn query_lru_cache_hit_skips_ollama() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings": [[0.1, 0.2, 0.3]]})),
            )
            .mount(&server)
            .await;

        let client = OllamaClient::new(&config_with_host(&server.uri()));

        // Call embed with the same text twice
        let result1 = client.embed(&["stripe webhook".to_string()]).await.unwrap();
        let result2 = client.embed(&["stripe webhook".to_string()]).await.unwrap();

        assert_eq!(result1, result2);

        // Only 1 POST should have been made — second hit from cache
        let requests = server.received_requests().await.unwrap();
        assert_eq!(
            requests.len(),
            1,
            "expected 1 Ollama call, got {}",
            requests.len()
        );
    }

    #[test]
    fn query_lru_cache_evicts_oldest_at_capacity() {
        let mut cache = BoundedLruCache::new(3);
        cache.insert("a".to_string(), vec![1.0]);
        cache.insert("b".to_string(), vec![2.0]);
        cache.insert("c".to_string(), vec![3.0]);
        cache.insert("d".to_string(), vec![4.0]); // should evict "a"
        assert!(
            cache.get("a").is_none(),
            "oldest entry 'a' should have been evicted"
        );
        assert!(
            cache.get("d").is_some(),
            "newest entry 'd' should be present"
        );
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn query_lru_cache_get_promotes_to_mru() {
        // cap=2: insert a, b → access a (promotes a to MRU) → insert c → b evicted, a survives.
        let mut cache = BoundedLruCache::new(2);
        cache.insert("a".to_string(), vec![1.0]);
        cache.insert("b".to_string(), vec![2.0]);
        cache.get("a"); // promote "a" — now order is [b, a], LRU is b
        cache.insert("c".to_string(), vec![3.0]); // should evict "b"
        assert!(cache.get("b").is_none(), "'b' should have been evicted");
        assert!(
            cache.get("a").is_some(),
            "'a' should survive after promotion"
        );
        assert!(cache.get("c").is_some(), "'c' should be present");
    }

    #[tokio::test]
    async fn query_lru_cache_miss_on_different_text() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings": [[0.5, 0.6, 0.7]]})),
            )
            .mount(&server)
            .await;

        let client = OllamaClient::new(&config_with_host(&server.uri()));

        client.embed(&["query one".to_string()]).await.unwrap();
        client.embed(&["query two".to_string()]).await.unwrap();

        // Both queries are different — 2 Ollama calls expected
        let requests = server.received_requests().await.unwrap();
        assert_eq!(
            requests.len(),
            2,
            "expected 2 Ollama calls, got {}",
            requests.len()
        );
    }

    #[tokio::test]
    async fn multi_text_embed_skips_cache() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})),
            )
            .mount(&server)
            .await;

        let client = OllamaClient::new(&config_with_host(&server.uri()));
        let texts = vec!["a".to_string(), "b".to_string()];

        // Call embed twice with the same multi-text slice
        client.embed(&texts).await.unwrap();
        client.embed(&texts).await.unwrap();

        // Multi-text calls bypass cache — 2 Ollama calls expected
        let requests = server.received_requests().await.unwrap();
        assert_eq!(
            requests.len(),
            2,
            "expected 2 Ollama calls for multi-text, got {}",
            requests.len()
        );

        // Cache should be empty since multi-text calls do not populate it
        assert_eq!(
            client.query_cache_len(),
            0,
            "cache should be empty after multi-text calls"
        );
    }

    // -- HNSW index tests --

    /// Build a VectorStore with `n` 4-dimensional vectors spread across the unit sphere.
    fn make_store_n(n: usize) -> VectorStore {
        let dims = 4usize;
        let mut keys = Vec::with_capacity(n);
        let mut hashes = Vec::with_capacity(n);
        let mut vectors: Vec<f32> = Vec::with_capacity(n * dims);
        for i in 0..n {
            keys.push(format!("file_{i}.ts"));
            hashes.push(format!("h{i}"));
            // Spread vectors across the unit sphere deterministically
            let angle = (i as f32) * 0.31415;
            vectors.push(angle.cos());
            vectors.push(angle.sin());
            vectors.push(((i as f32) * 0.1).cos() * 0.5);
            vectors.push(((i as f32) * 0.07).sin() * 0.5);
        }
        VectorStore::new(dims as u32, keys, hashes, vectors)
    }

    /// Build two large (>2000) identical stores: one via `new`, one via `new_with_tuning`
    /// with default knobs — so tests can compare their outputs.
    fn make_large_identical_stores() -> (VectorStore, VectorStore) {
        let n = 2001_usize;
        let dims = 3_u32;
        let keys: Vec<String> = (0..n).map(|i| format!("k{i}")).collect();
        let hashes: Vec<String> = keys.clone();
        let vectors: Vec<f32> = (0..n)
            .flat_map(|i| {
                let v = (i as f32) * 0.001;
                vec![v, v + 0.5, 1.0 - v]
            })
            .collect();
        let store_default = VectorStore::new(dims, keys.clone(), hashes.clone(), vectors.clone());
        let store_tuned =
            VectorStore::new_with_tuning(dims, keys, hashes, vectors, HnswTuning::default());
        (store_default, store_tuned)
    }

    #[test]
    fn hnsw_find_nearest_returns_correct_top_k() {
        // 20 known 4-dim vectors; query close to index 0 (angle=0 → (1,0,…))
        let store = make_store_n(20);
        let query = vec![1.0f32, 0.0, 0.0, 0.0];

        let brute = store.find_nearest_brute_force(&query, 3);
        let hnsw = store.find_nearest_hnsw(&query, 3);

        assert!(!hnsw.is_empty(), "HNSW should return results");

        // ANN may not be perfect on tiny sets; require ≥2 of top-3 to match brute force
        let brute_keys: std::collections::HashSet<_> =
            brute.iter().map(|(k, _)| k.as_str()).collect();
        let overlap = hnsw
            .iter()
            .filter(|(k, _)| brute_keys.contains(k.as_str()))
            .count();
        assert!(
            overlap >= 2,
            "HNSW top-3 should overlap ≥2 with brute-force top-3, got overlap={overlap}\n  hnsw={hnsw:?}\n  brute={brute:?}"
        );
    }

    #[test]
    fn find_nearest_uses_brute_force_below_threshold() {
        // 50 entries — well below 2000 threshold
        let store = make_store_n(50);
        let query = vec![1.0f32, 0.0, 0.0, 0.0];

        let brute = store.find_nearest_brute_force(&query, 5);
        let dispatched = store.find_nearest(&query, 5);

        // Below threshold, find_nearest MUST use brute force — results are identical
        assert_eq!(
            brute, dispatched,
            "Below HNSW threshold, find_nearest must match brute_force exactly"
        );
    }

    #[test]
    fn hnsw_index_is_built_lazily() {
        let store = make_store_n(20);
        let query = vec![1.0f32, 0.0, 0.0, 0.0];

        // First call builds the index, second reuses it — results must be identical
        let r1 = store.find_nearest_hnsw(&query, 3);
        let r2 = store.find_nearest_hnsw(&query, 3);

        assert_eq!(
            r1, r2,
            "Repeated HNSW calls must return consistent results (OnceLock)"
        );
        assert!(!r1.is_empty());
    }

    /// Verify that `HnswTuning::default()` matches the expected defaults.
    #[test]
    fn hnsw_tuning_default_values() {
        let t = HnswTuning::default();
        assert_eq!(
            t.ef_construction,
            crate::config::DEFAULT_HNSW_EF_CONSTRUCTION
        );
        assert_eq!(t.ef_search, crate::config::DEFAULT_HNSW_EF_SEARCH);
    }

    /// Verify that `new_with_tuning` produces a store that answers HNSW queries
    /// identically to `new` when using default knob values.
    #[test]
    fn hnsw_tuning_default_matches_new() {
        let (store_default, store_tuned) = make_large_identical_stores();
        let query = vec![1.0_f32; 3];
        let r1 = store_default.find_nearest_hnsw(&query, 5);
        let r2 = store_tuned.find_nearest_hnsw(&query, 5);
        // Both should find results; key sets must match (order may vary for HNSW ties).
        let keys1: std::collections::HashSet<_> = r1.iter().map(|(k, _)| k.clone()).collect();
        let keys2: std::collections::HashSet<_> = r2.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(keys1, keys2, "default tuning should match new() results");
    }

    /// Verify that a custom ef_search setting is accepted without panic.
    #[test]
    fn hnsw_tuning_custom_ef_search_no_panic() {
        let tuning = HnswTuning {
            ef_construction: 50,
            ef_search: 256,
        };
        let (store, _) = {
            let n = 2001_usize;
            let dims = 3_u32;
            let keys: Vec<String> = (0..n).map(|i| format!("k{i}")).collect();
            let hashes: Vec<String> = keys.to_vec();
            let vectors: Vec<f32> = (0..n)
                .flat_map(|i| {
                    let v = i as f32;
                    vec![v, v + 1.0, v + 2.0]
                })
                .collect();
            (
                VectorStore::new_with_tuning(dims, keys, hashes, vectors, tuning),
                (),
            )
        };
        let query = vec![1.0_f32, 2.0, 3.0];
        let results = store.find_nearest_hnsw(&query, 5);
        assert!(
            !results.is_empty(),
            "custom ef_search=256 should still return results"
        );
    }
}

#[cfg(test)]
mod chunk_merge_tests {
    use crate::core::embeddings::{merge_embedding_vectors, split_embedding_input};

    #[test]
    fn split_short_input_returns_single_chunk() {
        let chunks = split_embedding_input("hello world", 2000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn split_exact_boundary() {
        let chunks = split_embedding_input("abcde", 5);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn split_long_input() {
        let chunks = split_embedding_input("abcdefghij", 3);
        assert_eq!(chunks, vec!["abc", "def", "ghi", "j"]);
    }

    #[test]
    fn split_preserves_content() {
        let text = "the quick brown fox jumps over the lazy dog";
        let chunks = split_embedding_input(text, 10);
        let reassembled: String = chunks.into_iter().collect();
        assert_eq!(reassembled, text);
    }

    #[test]
    fn split_multibyte() {
        let text = "😀😁😂";
        let chunks = split_embedding_input(text, 5);
        let reassembled: String = chunks.into_iter().collect();
        assert_eq!(reassembled, text);
    }

    #[test]
    fn split_empty() {
        let chunks = split_embedding_input("", 100);
        assert_eq!(chunks, vec![""]);
    }

    #[test]
    fn merge_single() {
        let merged = merge_embedding_vectors(&[vec![1.0, 2.0, 3.0]], &[10]).unwrap();
        assert_eq!(merged, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn merge_equal_weights() {
        let merged = merge_embedding_vectors(&[vec![2.0, 4.0], vec![4.0, 6.0]], &[1, 1]).unwrap();
        assert!((merged[0] - 3.0).abs() < 1e-5);
        assert!((merged[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn merge_weighted() {
        let merged = merge_embedding_vectors(&[vec![10.0, 0.0], vec![0.0, 10.0]], &[3, 1]).unwrap();
        assert!((merged[0] - 7.5).abs() < 1e-5);
        assert!((merged[1] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn merge_empty_errors() {
        assert!(merge_embedding_vectors(&[], &[]).is_err());
    }

    #[test]
    fn merge_dim_mismatch_errors() {
        assert!(merge_embedding_vectors(&[vec![1.0, 2.0], vec![3.0]], &[1, 1]).is_err());
    }

    #[test]
    fn merge_preserves_dim() {
        let dim = 384;
        let v1: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (dim - i) as f32).collect();
        let merged = merge_embedding_vectors(&[v1, v2], &[100, 200]).unwrap();
        assert_eq!(merged.len(), dim);
    }
}
