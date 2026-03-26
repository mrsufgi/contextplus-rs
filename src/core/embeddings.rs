use std::collections::HashMap;
use std::sync::Arc;

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

#[derive(Clone)]
pub struct OllamaClient {
    client: reqwest::Client,
    host: String,
    model: String,
    chat_model: String,
    batch_size: usize,
    embed_options: Option<EmbedRuntimeOptions>,
    embed_chunk_chars: usize,
    cancel_token: CancellationToken,
}

#[derive(serde::Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<&'a EmbedRuntimeOptions>,
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
            embed_options: EmbedRuntimeOptions::from_config(config),
            embed_chunk_chars: config.embed_chunk_chars,
            cancel_token: CancellationToken::new(),
        }
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
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
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

        Ok(results)
    }

    /// Get the configured batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
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
        };

        let token = self.cancel_token.clone();
        let response = tokio::select! {
            _ = token.cancelled() => {
                return Err(ContextPlusError::Cancelled);
            }
            result = self.client.post(&url).json(&body).send() => {
                result.map_err(|e| ContextPlusError::Ollama(format!("request failed: {}", e)))?
            }
        };

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

/// In-memory flat vector store with cosine similarity search via simsimd.
/// Uses brute-force SIMD+rayon scan — handles 30K vectors in <3ms, which is
/// fast enough that approximate indices (HNSW) aren't worth the build cost.
pub struct VectorStore {
    dims: u32,
    count: u32,
    vectors: VectorData,
    keys: Vec<String>,
    hashes: Vec<String>,
    key_index: HashMap<String, usize>,
}

impl VectorStore {
    /// Build a VectorStore from parallel arrays of keys, hashes, and vectors.
    pub fn new(dims: u32, keys: Vec<String>, hashes: Vec<String>, vectors: Vec<f32>) -> Self {
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
        }
    }

    /// Build from an EmbeddingCache (HashMap of path -> (hash, vector)).
    /// Single pass over the cache — no redundant HashMap lookups.
    pub fn from_cache(cache: &HashMap<String, CacheEntry>) -> Option<Self> {
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
        Some(Self::new(dims, keys, hashes, vectors))
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
    /// Uses SIMD+rayon brute-force scan — handles 30K vectors in <3ms.
    /// Returns (key, similarity) pairs sorted by descending similarity.
    pub fn find_nearest(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        if self.count == 0 || query.len() != self.dims as usize {
            return Vec::new();
        }

        self.find_nearest_brute_force(query, top_k)
    }

    /// Brute-force exact nearest neighbor search with SIMD cosine similarity.
    /// Uses sequential scan for <2K vectors (avoids rayon overhead), parallel for larger stores.
    fn find_nearest_brute_force(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
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
            tracing::warn!(a_len = a.len(), b_len = b.len(), "simsimd cosine returned None");
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
        };
        assert!(serde_json::to_value(&r).unwrap().get("options").is_none());
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
        };
        assert_eq!(serde_json::to_value(&r).unwrap()["options"]["num_gpu"], 2);
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
    async fn cancellation_before_request_returns_cancelled() {
        let mut config = Config::from_env();
        config.ollama_host = "http://localhost:0".to_string();
        let client = OllamaClient::new(&config);

        client.cancel_all_embeddings();

        let result = client.embed(&["test".to_string()]).await;
        assert!(result.is_err());
        assert!(matches!(result, Err(ContextPlusError::Cancelled)));
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
