use std::collections::HashMap;

use crate::config::Config;
use crate::error::{ContextPlusError, Result};

/// Type alias for the boxed future returned by embedding functions.
type EmbedFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MIN_EMBED_INPUT_CHARS: usize = 256;
const SINGLE_INPUT_SHRINK_FACTOR: f64 = 0.75;
const MAX_SINGLE_INPUT_RETRIES: usize = 8;

// ---------------------------------------------------------------------------
// OllamaClient
// ---------------------------------------------------------------------------

/// HTTP client for Ollama embedding and chat APIs with adaptive batch/retry.
#[derive(Clone)]
pub struct OllamaClient {
    client: reqwest::Client,
    host: String,
    model: String,
    chat_model: String,
    batch_size: usize,
}

#[derive(serde::Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaClient {
    pub fn new(config: &Config) -> Self {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(4)
            .build()
            .expect("reqwest client build");
        Self {
            client,
            host: config.ollama_host.clone(),
            model: config.ollama_embed_model.clone(),
            chat_model: config.ollama_chat_model.clone(),
            batch_size: config.embed_batch_size,
        }
    }

    /// Embed a slice of texts, returning one vector per text.
    /// Handles batching and adaptive retry on context length errors.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let batch_result = self.embed_batch_adaptive(chunk).await?;
            all_embeddings.extend(batch_result);
        }
        Ok(all_embeddings)
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
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
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
        let url = format!("{}/api/embed", self.host);
        let body = EmbedRequest {
            model: &self.model,
            input: inputs,
        };

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
// VectorStore
// ---------------------------------------------------------------------------

/// In-memory flat vector store with cosine similarity search via simsimd.
pub struct VectorStore {
    dims: u32,
    count: u32,
    vectors: Vec<f32>,
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
            vectors,
            keys,
            hashes,
            key_index,
        }
    }

    /// Build from an EmbeddingCache (HashMap of path -> (hash, vector)).
    pub fn from_cache(cache: &HashMap<String, CacheEntry>) -> Option<Self> {
        if cache.is_empty() {
            return None;
        }
        let keys: Vec<String> = cache.keys().cloned().collect();
        let dims = cache[&keys[0]].vector.len() as u32;
        let hashes: Vec<String> = keys.iter().map(|k| cache[k].hash.clone()).collect();
        let mut vectors = Vec::with_capacity(keys.len() * dims as usize);
        for key in &keys {
            vectors.extend_from_slice(&cache[key].vector);
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
        self.key_index.get(key).map(|&idx| {
            let offset = idx * self.dims as usize;
            &self.vectors[offset..offset + self.dims as usize]
        })
    }

    /// Get a key by index.
    pub fn key_at(&self, idx: usize) -> &str {
        &self.keys[idx]
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
        &self.vectors
    }

    /// Find the top-k nearest neighbors by cosine similarity using simsimd.
    /// Returns (key, similarity) pairs sorted by descending similarity.
    pub fn find_nearest(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        if self.count == 0 || query.len() != self.dims as usize {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.count as usize);

        for i in 0..self.count as usize {
            let offset = i * self.dims as usize;
            let stored = &self.vectors[offset..offset + self.dims as usize];

            let similarity = cosine_similarity_simsimd(query, stored);
            scored.push((i, similarity));
        }

        // Sort descending by similarity
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(idx, sim)| (self.keys[idx].clone(), sim))
            .collect()
    }

    /// Cosine similarity for a specific key.
    pub fn cosine_by_key(&self, query: &[f32], key: &str) -> f32 {
        match self.key_index.get(key) {
            Some(&idx) => {
                let offset = idx * self.dims as usize;
                let stored = &self.vectors[offset..offset + self.dims as usize];
                cosine_similarity_simsimd(query, stored)
            }
            None => 0.0,
        }
    }

    /// Convert to a cache map.
    pub fn to_cache(&self) -> HashMap<String, CacheEntry> {
        let mut cache = HashMap::with_capacity(self.count as usize);
        for i in 0..self.count as usize {
            let offset = i * self.dims as usize;
            cache.insert(
                self.keys[i].clone(),
                CacheEntry {
                    hash: self.hashes[i].clone(),
                    vector: self.vectors[offset..offset + self.dims as usize].to_vec(),
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
        None => 0.0,
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
    fn shrink_input_short_stays() {
        let input = "short";
        let shrunk = shrink_input(input);
        assert_eq!(shrunk, "short");
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
}
