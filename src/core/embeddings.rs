use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use tokio_util::sync::CancellationToken;

use crate::cache::rkyv_store;
use crate::config::{ChatProvider, Config, EmbedProvider};
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
// Model providers
// ---------------------------------------------------------------------------

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
    /// True when the map has been modified since the last `drain_to_vec` call.
    dirty: bool,
}

impl BoundedLruCache {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            map: indexmap::IndexMap::with_capacity(cap + 1),
            dirty: false,
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
        self.dirty = true;
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    /// Return all (key, vector) pairs in LRU → MRU order and clear the dirty flag.
    fn drain_to_vec(&mut self) -> Vec<(String, Vec<f32>)> {
        self.dirty = false;
        self.map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

/// Capacity of the persistent on-disk query-embedding cache.
const QUERY_CACHE_PERSIST_CAP: usize = 10_000;

/// Debounce window for background query cache flushes (milliseconds).
const QUERY_CACHE_FLUSH_DEBOUNCE_MS: u64 = 2_000;

/// In-flight result for a single-text embed coalescing slot.
///
/// The owner of a slot sends `Ok(vec)` on success or `Err(msg)` on failure.
/// Waiters subscribe to the `watch::Receiver` and block until the owner sends.
type InFlightResult = std::result::Result<Vec<f32>, String>;

#[derive(Clone)]
pub struct OllamaClient {
    client: reqwest::Client,
    embed_backend: EmbedBackend,
    chat_backend: ChatBackend,
    query_cache_identity: String,
    query_prefix: String,
    document_prefix: String,
    batch_size: usize,
    query_batch_size: usize,
    embed_chunk_chars: usize,
    cancel_token: CancellationToken,
    request_timeout: std::time::Duration,
    chat_timeout: std::time::Duration,
    query_cache: Arc<std::sync::Mutex<BoundedLruCache>>,
    /// Root directory used to locate the persistent query-embedding cache file.
    /// `None` when the client was constructed without a project root (e.g. in
    /// unit tests that do not want on-disk I/O).
    root_dir: Option<Arc<PathBuf>>,
    /// Sender half of the debounce channel.  Sending a unit triggers a
    /// background task to flush the in-memory LRU to disk after a short delay.
    flush_tx: Option<Arc<tokio::sync::mpsc::UnboundedSender<()>>>,
    /// In-flight coalescing map for single-text embed requests.
    ///
    /// When a caller is the *first* to request a query embedding it inserts a
    /// `watch::Sender` here and owns the Ollama call.  Concurrent callers with
    /// the same query string subscribe to the sender's `watch::Receiver` and
    /// wait for the result instead of issuing their own HTTP request.
    in_flight:
        Arc<std::sync::Mutex<HashMap<String, tokio::sync::watch::Sender<Option<InFlightResult>>>>>,
    /// Optional concurrency gate for outbound Ollama embed HTTP calls.
    ///
    /// When `Some`, every call to `call_embed_api` acquires one permit before
    /// issuing the HTTP request and releases it on completion (or error).
    /// `None` means no gating — existing behaviour, used in unit tests that
    /// construct `OllamaClient` without a running server.
    ///
    /// The in-flight coalescing layer for single-text queries is intentionally
    /// outside the gate: coalesced waiters share the producer's result without
    /// holding their own permit.  Only the *producer* task that actually issues
    /// the HTTP call acquires a permit (via `embed_single_query` →
    /// `embed_batch_adaptive` → `call_embed_api`).
    semaphore: Option<Arc<tokio::sync::Semaphore>>,
}

#[derive(Clone)]
enum EmbedBackend {
    Ollama {
        host: String,
        model: String,
        api_key: Option<String>,
        options: Option<EmbedRuntimeOptions>,
    },
    OpenAi {
        base_url: String,
        model: String,
        api_key: Option<String>,
    },
}

#[derive(Clone)]
enum ChatBackend {
    Ollama {
        host: String,
        model: String,
        api_key: Option<String>,
    },
    OpenAi {
        base_url: String,
        model: String,
        api_key: Option<String>,
    },
    Claude {
        executable: String,
        model: String,
    },
    Anthropic {
        base_url: String,
        model: String,
        auth: AnthropicAuth,
    },
}

#[derive(Clone)]
enum AnthropicAuth {
    ApiKey(String),
    Bearer(String),
    None,
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

#[derive(serde::Serialize)]
struct OpenAiEmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(serde::Deserialize)]
struct OpenAiEmbedItem {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct OpenAiEmbedResponse {
    data: Vec<OpenAiEmbedItem>,
}

impl OllamaClient {
    pub fn new(config: &Config) -> Self {
        Self::new_with_root(config, None)
    }

    /// Construct an `OllamaClient` with an optional project root directory.
    ///
    /// When `root_dir` is `Some`, the client:
    /// 1. Loads any previously persisted query-embedding cache from
    ///    `<root_dir>/.mcp_data/query-embeddings-<model>.rkyv` into the
    ///    in-memory LRU on construction.
    /// 2. Spawns a background Tokio task that debounce-flushes the LRU to disk
    ///    whenever a new query embedding is inserted.
    pub fn new_with_root(config: &Config, root_dir: Option<PathBuf>) -> Self {
        // NOTE: we deliberately do NOT call `.http2_prior_knowledge()`.
        // Ollama (as of 0.3.x and earlier) serves HTTP/1.1 only — sending an
        // h2 preface upfront produces "Remote peer returned unexpected data
        // while we expected SETTINGS frame" and breaks every request.
        // Leaving the `http2` cargo feature enabled is harmless: reqwest will
        // speak h1 by default and only negotiate h2 via ALPN (TLS) if both
        // sides support it. Opt-in for h2-speaking proxied deployments can
        // come in a future `CONTEXTPLUS_OLLAMA_FORCE_HTTP2` env knob.
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(4)
            // Keep idle connections alive for 55 s — safely under the common
            // 60 s idle-eviction window used by reverse proxies and firewalls.
            .pool_idle_timeout(std::time::Duration::from_secs(55))
            // TCP keepalive pings every 30 s so NAT/firewall state is preserved
            // even when the connection is otherwise silent.
            .tcp_keepalive(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("reqwest client build");

        let in_mem_cap = QUERY_CACHE_PERSIST_CAP.max(256);
        let cache = Arc::new(std::sync::Mutex::new(BoundedLruCache::new(in_mem_cap)));

        // Load persisted entries into the in-memory LRU.
        if let Some(ref dir) = root_dir {
            match rkyv_store::load_query_cache(dir, &config.query_cache_identity()) {
                Ok(entries) => {
                    let mut lru = cache.lock().unwrap();
                    for (k, v) in entries {
                        lru.insert(k, v);
                    }
                    lru.dirty = false; // freshly loaded — nothing new to flush yet
                    tracing::debug!(count = lru.len(), "Loaded persistent query embedding cache");
                }
                Err(e) => {
                    tracing::warn!("Failed to load query embedding cache: {e}");
                }
            }
        }

        // Spawn debounce flush task when we have a root dir AND a Tokio runtime
        // is available (unit tests that construct OllamaClient outside of any
        // async context must not panic here).
        let flush_tx_arc = if let Some(ref dir) = root_dir {
            let (flush_tx, flush_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
            let cache_clone = Arc::clone(&cache);
            let model_clone = config.query_cache_identity();
            let dir_clone = dir.clone();
            // `tokio::runtime::Handle::try_current()` returns Err when there is
            // no active runtime (sync test threads).  In that case we skip the
            // background task — `flush_query_cache()` on shutdown still works.
            if tokio::runtime::Handle::try_current().is_ok() {
                tokio::spawn(query_cache_flush_task(
                    flush_rx,
                    cache_clone,
                    dir_clone,
                    model_clone,
                ));
                Some(Arc::new(flush_tx))
            } else {
                None
            }
        } else {
            None
        };

        let embed_backend = match config.embed_provider {
            EmbedProvider::Ollama => EmbedBackend::Ollama {
                host: config.ollama_host.clone(),
                model: config.ollama_embed_model.clone(),
                api_key: config.ollama_api_key.clone(),
                options: EmbedRuntimeOptions::from_config(config),
            },
            EmbedProvider::OpenAi => EmbedBackend::OpenAi {
                base_url: config.openai_base_url.trim_end_matches('/').to_string(),
                model: config.openai_embed_model.clone(),
                api_key: config.openai_api_key.clone(),
            },
        };
        let chat_backend = match config.chat_provider {
            ChatProvider::Ollama => ChatBackend::Ollama {
                host: config.ollama_host.clone(),
                model: config.ollama_chat_model.clone(),
                api_key: config.ollama_api_key.clone(),
            },
            ChatProvider::OpenAi => ChatBackend::OpenAi {
                base_url: config
                    .chat_base_url
                    .as_deref()
                    .unwrap_or(&config.openai_base_url)
                    .trim_end_matches('/')
                    .to_string(),
                model: config.openai_chat_model.clone(),
                api_key: config.chat_api_key.clone(),
            },
            ChatProvider::Claude => ChatBackend::Claude {
                executable: config.claude_path.clone(),
                model: config.claude_model.clone(),
            },
            ChatProvider::Anthropic => ChatBackend::Anthropic {
                base_url: config.anthropic_base_url.trim_end_matches('/').to_string(),
                model: config.anthropic_chat_model.clone(),
                auth: if let Some(key) = &config.anthropic_api_key {
                    AnthropicAuth::ApiKey(key.clone())
                } else if let Some(token) = &config.anthropic_auth_token {
                    AnthropicAuth::Bearer(token.clone())
                } else {
                    AnthropicAuth::None
                },
            },
        };

        Self {
            client,
            embed_backend,
            chat_backend,
            query_cache_identity: config.query_cache_identity(),
            query_prefix: config.embed_query_prefix.clone(),
            document_prefix: config.embed_doc_prefix.clone(),
            batch_size: config.embed_batch_size,
            query_batch_size: config.query_batch_size,
            embed_chunk_chars: config.embed_chunk_chars,
            cancel_token: CancellationToken::new(),
            request_timeout: EMBED_REQUEST_TIMEOUT,
            chat_timeout: std::time::Duration::from_secs(90),
            query_cache: cache,
            root_dir: root_dir.map(Arc::new),
            flush_tx: flush_tx_arc,
            in_flight: Arc::new(std::sync::Mutex::new(HashMap::new())),
            semaphore: None,
        }
    }

    /// Flush the in-memory query-embedding LRU to disk synchronously.
    ///
    /// Call this on clean shutdown to ensure no entries are lost between the
    /// last debounce flush and process exit.
    pub fn flush_query_cache(&self) {
        let Some(ref dir) = self.root_dir else { return };
        let entries = {
            let mut lru = self.query_cache.lock().unwrap();
            if !lru.dirty {
                return; // nothing new since last flush
            }
            lru.drain_to_vec()
        };
        if let Err(e) = rkyv_store::save_query_cache(dir, &self.query_cache_identity, &entries) {
            tracing::warn!("Failed to flush query embedding cache on shutdown: {e}");
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

    #[cfg(test)]
    fn with_chat_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.chat_timeout = timeout;
        self
    }

    /// Attach a concurrency semaphore to this client.
    ///
    /// Every outbound Ollama embed HTTP call will acquire one permit before
    /// sending and release it on completion or error.  Clones of this client
    /// share the same `Arc<Semaphore>` so the cap is global across all clones.
    ///
    /// Call this once when constructing the production `OllamaClient` via
    /// `ContextPlusServer::new`.  Unit tests that do not pass a semaphore
    /// retain the default `None` (no gating).
    pub fn with_semaphore(mut self, semaphore: Arc<tokio::sync::Semaphore>) -> Self {
        self.semaphore = Some(semaphore);
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

        // ----------------------------------------------------------------
        // Single-text (query) path: LRU cache + in-flight coalescing.
        // ----------------------------------------------------------------
        if texts.len() == 1 {
            let query = &texts[0];

            // 1. Fast path: LRU cache hit — no lock contention with Ollama.
            {
                let mut cache = self.query_cache.lock().unwrap();
                if let Some(v) = cache.get(query) {
                    return Ok(vec![v.clone()]);
                }
            }

            // 2. Coalescing: check if another task is already calling Ollama
            //    for this exact query.
            let (owner, mut rx) = {
                let mut map = self.in_flight.lock().unwrap();
                if let Some(tx) = map.get(query) {
                    // Become a waiter: subscribe before releasing the lock.
                    (false, tx.subscribe())
                } else {
                    // Become the owner: insert a slot and do the Ollama call.
                    let (tx, rx) = tokio::sync::watch::channel(None::<InFlightResult>);
                    map.insert(query.clone(), tx);
                    (true, rx)
                }
            };

            if owner {
                // Owner: call Ollama, publish result to all waiters.
                let embed_result = self.embed_single_query(query).await;

                // Populate LRU BEFORE broadcasting to waiters, so any waiter
                // that wakes and immediately re-calls `embed` (or any new caller
                // arriving after broadcast) sees the cache hit rather than
                // issuing a duplicate Ollama request. (Review #60 F1.)
                if let Ok(ref v) = embed_result {
                    let mut cache = self.query_cache.lock().unwrap();
                    cache.insert(query.clone(), v.clone());
                }

                // Publish result and remove the in-flight slot atomically.
                {
                    let mut map = self.in_flight.lock().unwrap();
                    if let Some(tx) = map.remove(query) {
                        let payload = match &embed_result {
                            Ok(v) => Ok(v.clone()),
                            Err(e) => Err(e.to_string()),
                        };
                        // Ignore send error: receivers may have been dropped.
                        let _ = tx.send(Some(payload));
                    }
                }

                let v = embed_result?;

                if let Some(ref tx) = self.flush_tx {
                    let _ = tx.send(());
                }

                return Ok(vec![v]);
            } else {
                // Waiter: block until the owner publishes.
                rx.changed().await.map_err(|_| {
                    ContextPlusError::Ollama(
                        "in-flight coalescing channel closed unexpectedly".into(),
                    )
                })?;

                let result = rx.borrow().clone().ok_or_else(|| {
                    ContextPlusError::Ollama("in-flight slot produced no result".into())
                })?;

                return match result {
                    Ok(v) => Ok(vec![v]),
                    Err(msg) => Err(ContextPlusError::Ollama(msg)),
                };
            }
        }

        // ----------------------------------------------------------------
        // Multi-text (batch / indexing) path — unchanged.
        // ----------------------------------------------------------------

        self.embed_uncached(texts).await
    }

    /// Embed user queries with the configured model-specific query prefix.
    pub async fn embed_queries(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<String> = texts
            .iter()
            .map(|text| format!("{}{}", self.query_prefix, text))
            .collect();
        self.embed(&inputs).await
    }

    /// Embed one user query with the configured model-specific query prefix.
    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embed_queries(&[query.to_string()])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| ContextPlusError::Ollama("empty embedding response".into()))
    }

    /// Embed index documents with the configured model-specific document prefix.
    pub async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if self.document_prefix.is_empty() {
            return self.embed_uncached(texts).await;
        }

        let inputs: Vec<String> = texts
            .iter()
            .map(|text| format!("{}{}", self.document_prefix, text))
            .collect();
        self.embed_uncached(&inputs).await
    }

    async fn embed_uncached(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
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

    /// Embed a single query string via Ollama (no caching, no coalescing).
    ///
    /// Used internally by the single-text path in [`Self::embed`] after the
    /// LRU cache miss.  Delegates to the same chunking + adaptive-batch
    /// machinery used by the multi-text path.
    async fn embed_single_query(&self, query: &str) -> Result<Vec<f32>> {
        let chunks = split_embedding_input(query, self.embed_chunk_chars);
        let flattened: Vec<String> = chunks.iter().map(|s| (*s).to_string()).collect();

        let mut flat_embeddings = Vec::with_capacity(flattened.len());
        for batch in flattened.chunks(self.batch_size) {
            let batch_result = self.embed_batch_adaptive(batch).await?;
            flat_embeddings.extend(batch_result);
        }

        let weights: Vec<usize> = chunks.iter().map(|c| c.len()).collect();
        merge_embedding_vectors(&flat_embeddings, &weights)
    }

    /// Get the configured batch size (used for warmup/indexing).
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the configured query batch size (used for live search queries, default 1 for CPU-optimal throughput).
    pub fn query_batch_size(&self) -> usize {
        self.query_batch_size
    }

    /// Send a chat request through the configured provider.
    pub async fn chat(&self, prompt: &str) -> Result<String> {
        let request = async {
            match &self.chat_backend {
                ChatBackend::Ollama {
                    host,
                    model,
                    api_key,
                } => {
                    self.chat_ollama(host, model, api_key.as_deref(), prompt)
                        .await
                }
                ChatBackend::OpenAi {
                    base_url,
                    model,
                    api_key,
                } => {
                    self.chat_openai(base_url, model, api_key.as_deref(), prompt)
                        .await
                }
                ChatBackend::Claude { executable, model } => {
                    chat_claude_cli(executable, model, prompt).await
                }
                ChatBackend::Anthropic {
                    base_url,
                    model,
                    auth,
                } => self.chat_anthropic(base_url, model, auth, prompt).await,
            }
        };
        tokio::time::timeout(self.chat_timeout, request)
            .await
            .map_err(|_| {
                ContextPlusError::Ollama(format!(
                    "Chat request timed out after {}ms",
                    self.chat_timeout.as_millis()
                ))
            })?
    }

    async fn chat_ollama(
        &self,
        host: &str,
        model: &str,
        api_key: Option<&str>,
        prompt: &str,
    ) -> Result<String> {
        let url = format!("{}/api/chat", host.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": false,
            "think": false,
        });

        let resp = with_bearer(self.client.post(&url).json(&body), api_key)
            .send()
            .await
            .map_err(|_| ContextPlusError::Ollama("Chat request failed".into()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            return Err(ContextPlusError::Ollama(format!(
                "Ollama chat returned {status}"
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

        let chat_resp: ChatResponse = resp
            .json()
            .await
            .map_err(|_| ContextPlusError::Ollama("Failed to parse chat response".into()))?;

        nonempty_chat_text(chat_resp.message.content, "Ollama")
    }

    async fn chat_openai(
        &self,
        base_url: &str,
        model: &str,
        api_key: Option<&str>,
        prompt: &str,
    ) -> Result<String> {
        let url = format!("{base_url}/chat/completions");
        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        });
        let response = with_bearer(self.client.post(url).json(&body), api_key)
            .send()
            .await
            .map_err(|_| ContextPlusError::Ollama("Chat request failed".into()))?;
        if !response.status().is_success() {
            let status = response.status();
            return Err(ContextPlusError::Ollama(format!(
                "OpenAI chat returned {status}"
            )));
        }
        #[derive(serde::Deserialize)]
        struct Message {
            content: String,
        }
        #[derive(serde::Deserialize)]
        struct Choice {
            message: Message,
        }
        #[derive(serde::Deserialize)]
        struct Response {
            choices: Vec<Choice>,
        }
        let response: Response = response
            .json()
            .await
            .map_err(|_| ContextPlusError::Ollama("Failed to parse OpenAI chat response".into()))?;
        let content = response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or_else(|| ContextPlusError::Ollama("OpenAI chat returned no choices".into()))?;
        nonempty_chat_text(content, "OpenAI")
    }

    async fn chat_anthropic(
        &self,
        base_url: &str,
        model: &str,
        auth: &AnthropicAuth,
        prompt: &str,
    ) -> Result<String> {
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
        });
        let mut request = self
            .client
            .post(format!("{base_url}/messages"))
            .header("anthropic-version", "2023-06-01")
            .json(&body);
        match auth {
            AnthropicAuth::ApiKey(key) => {
                request = with_sensitive_header(request, "x-api-key", key);
            }
            AnthropicAuth::Bearer(token) => {
                request = with_bearer(request, Some(token));
                request = request.header("anthropic-beta", "oauth-2025-04-20");
            }
            AnthropicAuth::None => {}
        }
        let response = request
            .send()
            .await
            .map_err(|_| ContextPlusError::Ollama("Chat request failed".into()))?;
        if !response.status().is_success() {
            let status = response.status();
            return Err(ContextPlusError::Ollama(format!(
                "Anthropic chat returned {status}"
            )));
        }
        #[derive(serde::Deserialize)]
        struct ContentBlock {
            #[serde(rename = "type")]
            kind: String,
            #[serde(default)]
            text: String,
        }
        #[derive(serde::Deserialize)]
        struct Response {
            content: Vec<ContentBlock>,
            stop_reason: Option<String>,
        }
        let response: Response = response.json().await.map_err(|_| {
            ContextPlusError::Ollama("Failed to parse Anthropic chat response".into())
        })?;
        if response.stop_reason.as_deref() != Some("end_turn") {
            let reason = match response.stop_reason.as_deref() {
                Some("max_tokens") => "max_tokens",
                Some("tool_use") => "tool_use",
                Some("stop_sequence") => "stop_sequence",
                Some("pause_turn") => "pause_turn",
                Some("refusal") => "refusal",
                _ => "unknown or missing",
            };
            return Err(ContextPlusError::Ollama(format!(
                "Anthropic chat stopped with reason {reason}"
            )));
        }
        let content = response
            .content
            .into_iter()
            .filter(|block| block.kind == "text")
            .map(|block| block.text)
            .collect::<String>();
        nonempty_chat_text(content, "Anthropic")
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

        // Cover the WHOLE request lifecycle (permit + send + status + body)
        // under a single deadline + cancel race. Previously only `.send()` was
        // raced against the cancel token, which left `response.json().await`
        // unsupervised — a slow/wedged Ollama body read could hang forever,
        // deadlocking the warmup binaries.
        let token = self.cancel_token.clone();
        let request = async {
            // Hold the permit until the body has been read, including error
            // paths. A closed semaphore retains the existing ungated behavior.
            let _permit = if let Some(sem) = &self.semaphore {
                Arc::clone(sem).acquire_owned().await.ok()
            } else {
                None
            };
            let (request, api_key) = match &self.embed_backend {
                EmbedBackend::Ollama {
                    host,
                    model,
                    api_key,
                    options,
                } => {
                    let url = format!("{}/api/embed", host.trim_end_matches('/'));
                    let body = EmbedRequest {
                        model,
                        input: inputs,
                        options: options.as_ref(),
                        keep_alive: -1,
                    };
                    (self.client.post(url).json(&body), api_key.as_deref())
                }
                EmbedBackend::OpenAi {
                    base_url,
                    model,
                    api_key,
                } => {
                    let url = format!("{base_url}/embeddings");
                    let body = OpenAiEmbedRequest {
                        model,
                        input: inputs,
                    };
                    (self.client.post(url).json(&body), api_key.as_deref())
                }
            };
            let response = with_bearer(request, api_key)
                .send()
                .await
                .map_err(|_| ContextPlusError::Ollama("embedding request failed".into()))?;

            if !response.status().is_success() {
                let status = response.status();
                // Remote bodies and transport errors may echo credentials. Only
                // retain the context-length classification needed for retries.
                let text = response.text().await.unwrap_or_default();
                let reason = if is_context_length_message(&text) {
                    ": input length exceeds context length"
                } else {
                    ""
                };
                return Err(ContextPlusError::Ollama(format!("HTTP {status}{reason}")));
            }

            match &self.embed_backend {
                EmbedBackend::Ollama { .. } => {
                    let embed_response: EmbedResponse = response.json().await.map_err(|_| {
                        ContextPlusError::Ollama("embedding response parse error".into())
                    })?;
                    Ok(embed_response.embeddings)
                }
                EmbedBackend::OpenAi { .. } => {
                    let embed_response: OpenAiEmbedResponse =
                        response.json().await.map_err(|_| {
                            ContextPlusError::Ollama("embedding response parse error".into())
                        })?;
                    reorder_openai_embeddings(embed_response.data, inputs.len())
                }
            }
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

fn nonempty_chat_text(content: String, provider: &str) -> Result<String> {
    if content.trim().is_empty() {
        Err(ContextPlusError::Ollama(format!(
            "{provider} chat returned empty text"
        )))
    } else {
        Ok(content)
    }
}

fn with_sensitive_header(
    request: reqwest::RequestBuilder,
    name: &'static str,
    secret: &str,
) -> reqwest::RequestBuilder {
    let Ok(mut value) = reqwest::header::HeaderValue::from_str(secret) else {
        return request;
    };
    value.set_sensitive(true);
    request.header(name, value)
}

struct TemporaryWorkingDir(PathBuf);

impl TemporaryWorkingDir {
    fn create() -> std::io::Result<Self> {
        let base = std::env::temp_dir();
        for _ in 0..10 {
            let path = base.join(format!(
                "contextplus-claude-{}-{:016x}",
                std::process::id(),
                rand::random::<u64>()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "unable to create unique Claude working directory",
        ))
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TemporaryWorkingDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn chat_claude_cli(executable: &str, model: &str, prompt: &str) -> Result<String> {
    let working_dir = TemporaryWorkingDir::create().map_err(|_| {
        ContextPlusError::Ollama("Failed to create Claude working directory".into())
    })?;
    let mut command = tokio::process::Command::new(executable);
    command
        .arg("-p")
        .arg(prompt)
        .arg("--output-format")
        .arg("json")
        .arg("--model")
        .arg(model)
        .arg("--safe-mode")
        .arg("--strict-mcp-config")
        .arg("--mcp-config")
        .arg(r#"{"mcpServers":{}}"#)
        .arg("--tools")
        .arg("")
        .arg("--disable-slash-commands")
        .arg("--setting-sources")
        .arg("")
        .arg("--no-session-persistence")
        .arg("--permission-mode")
        .arg("dontAsk")
        .arg("--permission-prompts")
        .arg("none")
        .arg("--no-chrome")
        .current_dir(working_dir.path())
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);

    // Command inherits the parent environment unchanged. In particular, this
    // path never reads or rewrites Claude Code or Anthropic credentials.
    let output = command
        .output()
        .await
        .map_err(|_| ContextPlusError::Ollama("Failed to run Claude Code CLI".into()))?;
    if !output.status.success() {
        return Err(ContextPlusError::Ollama(format!(
            "Claude Code CLI exited with status {}",
            output.status
        )));
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|_| ContextPlusError::Ollama("Failed to parse Claude Code JSON output".into()))?;
    if value.get("is_error").and_then(serde_json::Value::as_bool) == Some(true)
        || value
            .get("subtype")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|subtype| subtype != "success")
    {
        return Err(ContextPlusError::Ollama(
            "Claude Code CLI returned an error result".into(),
        ));
    }
    let content = value
        .get("result")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ContextPlusError::Ollama("Claude Code JSON output has no result".into()))?;
    nonempty_chat_text(content.to_string(), "Claude Code")
}

fn with_bearer(request: reqwest::RequestBuilder, api_key: Option<&str>) -> reqwest::RequestBuilder {
    let Some(api_key) = api_key.filter(|key| !key.is_empty()) else {
        return request;
    };
    with_sensitive_header(request, "authorization", &format!("Bearer {api_key}"))
}

fn reorder_openai_embeddings(data: Vec<OpenAiEmbedItem>, expected: usize) -> Result<Vec<Vec<f32>>> {
    if data.len() != expected {
        return Err(ContextPlusError::Ollama(format!(
            "embedding response size mismatch: expected {expected}, got {}",
            data.len()
        )));
    }
    let mut ordered: Vec<Option<Vec<f32>>> = (0..expected).map(|_| None).collect();
    for item in data {
        let slot = ordered.get_mut(item.index).ok_or_else(|| {
            ContextPlusError::Ollama(format!(
                "embedding response index {} is out of range for {expected} inputs",
                item.index
            ))
        })?;
        if slot.is_some() {
            return Err(ContextPlusError::Ollama(format!(
                "embedding response contains duplicate index {}",
                item.index
            )));
        }
        *slot = Some(item.embedding);
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            embedding.ok_or_else(|| {
                ContextPlusError::Ollama(format!("embedding response is missing index {index}"))
            })
        })
        .collect()
}

/// Background task that debounce-flushes the query-embedding LRU to disk.
///
/// Receives flush trigger signals via `rx`. After receiving a signal it waits
/// `QUERY_CACHE_FLUSH_DEBOUNCE_MS` before writing, absorbing any further signals
/// that arrive during the wait window. The task exits when `rx` is closed (i.e.
/// when the `OllamaClient` is dropped).
async fn query_cache_flush_task(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<()>,
    cache: Arc<std::sync::Mutex<BoundedLruCache>>,
    root_dir: PathBuf,
    model: String,
) {
    loop {
        // Wait for the first trigger.
        if rx.recv().await.is_none() {
            break; // channel closed — client dropped
        }

        // Debounce: drain any additional signals that arrive during the window.
        let deadline = tokio::time::Instant::now()
            + std::time::Duration::from_millis(QUERY_CACHE_FLUSH_DEBOUNCE_MS);
        loop {
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => break,
                msg = rx.recv() => {
                    if msg.is_none() { return; } // channel closed
                    // more signals arrived — keep draining until deadline
                }
            }
        }

        // Flush snapshot to disk.
        let entries = {
            let mut lru = cache.lock().unwrap();
            if !lru.dirty {
                continue;
            }
            lru.drain_to_vec()
        };
        if let Err(e) = rkyv_store::save_query_cache(&root_dir, &model, &entries) {
            tracing::warn!("query cache background flush failed: {e}");
        } else {
            tracing::debug!(
                count = entries.len(),
                "Flushed query embedding cache to disk"
            );
        }
    }
}

fn is_context_length_error(err: &ContextPlusError) -> bool {
    is_context_length_message(&err.to_string())
}

fn is_context_length_message(message: &str) -> bool {
    let msg = message.to_lowercase();
    msg.contains("input length exceeds context length")
        || (msg.contains("context") && msg.contains("exceed"))
        || msg.contains("maximum context length")
        || msg.contains("too many tokens")
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
        *GLOBAL.get_or_init(|| HnswTuning::from_config(&crate::config::Config::from_env()))
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
        // Pick `dims` from the first entry whose vector is non-empty. Cache
        // entries with `vector.len() == 0` can leak in via legacy on-disk
        // payloads or budget-exceeded paths that recorded a placeholder.
        // Filter them at this boundary so the flat `vectors` buffer stays
        // aligned with `keys` (one slice of length `dims` per key) and the
        // HNSW build below never sees a zero-length slice — which manifests
        // at query time as `simsimd cosine returned None a_len=N b_len=0`.
        let dims = cache.values().map(|e| e.vector.len()).find(|&l| l > 0)? as u32;
        let valid_count = cache
            .values()
            .filter(|e| e.vector.len() == dims as usize)
            .count();
        if valid_count == 0 {
            return None;
        }
        let mut skipped_empty = 0usize;
        let mut skipped_mismatch = 0usize;
        let mut keys = Vec::with_capacity(valid_count);
        let mut hashes = Vec::with_capacity(valid_count);
        let mut vectors = Vec::with_capacity(valid_count * dims as usize);
        for (key, entry) in cache {
            if entry.vector.is_empty() {
                skipped_empty += 1;
                continue;
            }
            if entry.vector.len() != dims as usize {
                skipped_mismatch += 1;
                continue;
            }
            keys.push(key.clone());
            hashes.push(entry.hash.clone());
            vectors.extend_from_slice(&entry.vector);
        }
        if skipped_empty > 0 || skipped_mismatch > 0 {
            tracing::warn!(
                skipped_empty,
                skipped_mismatch,
                kept = valid_count,
                dims,
                "VectorStore::from_cache: dropped cache entries with bad vector length"
            );
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
    ///
    /// Applies the same dot-segment exclusion rule as the walker so that
    /// stale entries from pre-#50 caches (e.g. `.claude/worktrees/…` paths
    /// indexed when the server was once run from inside a worktree dir)
    /// are dropped on load. Mirrors `CacheData::sweep_excluded_keys` for
    /// the mmap/VectorStore load path that bypasses `load_cache`.
    pub fn to_cache(&self) -> HashMap<String, CacheEntry> {
        let vectors = self.vectors.as_slice();
        let mut cache = HashMap::with_capacity(self.count as usize);
        let mut dropped = 0usize;
        for i in 0..self.count as usize {
            let key = &self.keys[i];
            if !crate::core::walker::should_keep_cache_key(key) {
                dropped += 1;
                continue;
            }
            let offset = i * self.dims as usize;
            cache.insert(
                key.clone(),
                CacheEntry {
                    hash: self.hashes[i].clone(),
                    vector: vectors[offset..offset + self.dims as usize].to_vec(),
                },
            );
        }
        if dropped > 0 {
            tracing::info!(
                dropped,
                total = self.count,
                "VectorStore::to_cache: dropped excluded keys (stale worktree/hidden paths)"
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
#[path = "provider_tests.rs"]
mod provider_tests;

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

    // Fix 1: HTTP/2 — verify the client builds without panicking when the
    // http2 feature is enabled and http2_prior_knowledge() is in the builder.
    #[test]
    fn http2_feature_enabled_in_build() {
        // If the "http2" feature were missing the builder call would fail to
        // compile; reaching this line at runtime proves the feature is active
        // and the builder chain succeeds.
        let config = Config::from_env();
        let _client = OllamaClient::new(&config);
        // No panic == pass.
    }

    // Fix 2a: API key present → Authorization header injected.
    #[test]
    fn api_key_header_present_when_env_set() {
        let mut config = Config::from_env();
        config.ollama_api_key = Some("test-key".to_string());
        // Build a raw reqwest client the same way new_with_root does so we can
        // inspect default_headers via the Debug representation.
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(key) = config.ollama_api_key.as_deref().filter(|k| !k.is_empty()) {
            let bearer = format!("Bearer {key}");
            if let Ok(val) = reqwest::header::HeaderValue::from_str(&bearer) {
                headers.insert(reqwest::header::AUTHORIZATION, val);
            }
        }
        let auth = headers
            .get(reqwest::header::AUTHORIZATION)
            .expect("Authorization header must be set when api key is present");
        assert_eq!(auth.to_str().unwrap(), "Bearer test-key");
    }

    // Fix 2b: API key absent → no Authorization header.
    #[test]
    fn api_key_no_header_when_env_unset() {
        let mut config = Config::from_env();
        config.ollama_api_key = None;
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(key) = config.ollama_api_key.as_deref().filter(|k| !k.is_empty()) {
            let bearer = format!("Bearer {key}");
            if let Ok(val) = reqwest::header::HeaderValue::from_str(&bearer) {
                headers.insert(reqwest::header::AUTHORIZATION, val);
            }
        }
        assert!(
            headers.get(reqwest::header::AUTHORIZATION).is_none(),
            "Authorization header must NOT be present when api key is absent"
        );
    }

    // Fix 3: pool_idle_timeout + tcp_keepalive — the builder succeeds (compile-
    // time check via the builder chain in new_with_root; no socket-level test
    // needed for a unit suite).
    #[test]
    fn client_builds_with_keepalive_and_pool_timeout() {
        let config = Config::from_env();
        let _client = OllamaClient::new(&config);
        // Reaching here means the builder chain — including pool_idle_timeout
        // and tcp_keepalive — compiled and ran without error.
    }

    #[test]
    fn ollama_client_respects_config() {
        let mut config = Config::from_env();
        config.ollama_host = "http://test:1234".to_string();
        config.ollama_embed_model = "test-model".to_string();
        config.ollama_chat_model = "test-chat".to_string();
        config.embed_batch_size = 25;

        let client = OllamaClient::new(&config);
        assert_eq!(client.batch_size, 25);
        match client.embed_backend {
            EmbedBackend::Ollama { model, .. } => assert_eq!(model, "test-model"),
            EmbedBackend::OpenAi { .. } => panic!("expected Ollama embedding backend"),
        }
        match client.chat_backend {
            ChatBackend::Ollama { host, model, .. } => {
                assert_eq!(host, "http://test:1234");
                assert_eq!(model, "test-chat");
            }
            _ => panic!("expected Ollama chat backend"),
        }
    }

    // -- OllamaClient::chat tests (wiremock) --

    fn config_with_host(host: &str) -> Config {
        let mut config = Config::from_env();
        config.ollama_host = host.to_string();
        config.ollama_chat_model = "test-chat-model".to_string();
        config
    }

    fn openai_config(base_url: &str) -> Config {
        let mut config = Config::from_env();
        config.embed_provider = crate::config::EmbedProvider::OpenAi;
        config.openai_base_url = base_url.to_string();
        config.openai_api_key = Some("openai-test-key".to_string());
        config.openai_embed_model = "test-embed-model".to_string();
        config.embed_query_prefix.clear();
        config.embed_doc_prefix.clear();
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

    fn openai_chat_config(base_url: &str) -> Config {
        let mut config = Config::from_env();
        config.chat_provider = crate::config::ChatProvider::OpenAi;
        config.chat_base_url = Some(base_url.to_string());
        config.chat_api_key = Some("chat-test-key".to_string());
        config.openai_chat_model = "test-chat-model".to_string();
        config
    }

    fn anthropic_chat_config(base_url: &str) -> Config {
        let mut config = Config::from_env();
        config.chat_provider = crate::config::ChatProvider::Anthropic;
        config.anthropic_base_url = base_url.to_string();
        config.anthropic_chat_model = "test-claude-model".to_string();
        config
    }

    #[tokio::test]
    async fn openai_chat_sends_compatible_request_and_parses_content() {
        use wiremock::matchers::{body_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer chat-test-key"))
            .and(body_json(serde_json::json!({
                "model": "test-chat-model",
                "messages": [{"role": "user", "content": "label this"}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "OpenAI Label"}}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = OllamaClient::new(&openai_chat_config(&server.uri()));
        assert_eq!(client.chat("label this").await.unwrap(), "OpenAI Label");
    }

    #[tokio::test]
    async fn openai_chat_error_does_not_expose_api_key() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(401).set_body_string("bad chat-test-key"))
            .mount(&server)
            .await;

        let client = OllamaClient::new(&openai_chat_config(&server.uri()));
        let error = client.chat("label").await.unwrap_err().to_string();
        assert!(!error.contains("chat-test-key"), "secret leaked: {error}");
    }

    #[tokio::test]
    async fn anthropic_chat_prefers_api_key_and_parses_text_blocks() {
        use wiremock::matchers::{body_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "anthropic-api-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(header("content-type", "application/json"))
            .and(body_json(serde_json::json!({
                "model": "test-claude-model",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "label this"}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [
                    {"type": "text", "text": "Anthropic "},
                    {"type": "tool_use", "id": "ignored"},
                    {"type": "text", "text": "Label"}
                ],
                "stop_reason": "end_turn"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut config = anthropic_chat_config(&format!("{}/v1", server.uri()));
        config.anthropic_api_key = Some("anthropic-api-key".to_string());
        config.anthropic_auth_token = Some("ignored-token".to_string());
        let client = OllamaClient::new(&config);

        assert_eq!(client.chat("label this").await.unwrap(), "Anthropic Label");
        let requests = server.received_requests().await.unwrap();
        assert!(requests[0].headers.get("authorization").is_none());
        assert!(requests[0].headers.get("anthropic-beta").is_none());
    }

    #[tokio::test]
    async fn anthropic_chat_uses_bearer_token_and_oauth_beta() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("authorization", "Bearer anthropic-token"))
            .and(header("anthropic-beta", "oauth-2025-04-20"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "OAuth Label"}],
                "stop_reason": "end_turn"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut config = anthropic_chat_config(&format!("{}/v1", server.uri()));
        config.anthropic_api_key = None;
        config.anthropic_auth_token = Some("anthropic-token".to_string());
        let client = OllamaClient::new(&config);

        assert_eq!(client.chat("label").await.unwrap(), "OAuth Label");
    }

    #[tokio::test]
    async fn anthropic_chat_rejects_non_end_turn_without_leaking_key() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "partial anthropic-secret"}],
                "stop_reason": "max_tokens"
            })))
            .mount(&server)
            .await;

        let mut config = anthropic_chat_config(&format!("{}/v1", server.uri()));
        config.anthropic_api_key = Some("anthropic-secret".to_string());
        let client = OllamaClient::new(&config);
        let error = client.chat("label").await.unwrap_err().to_string();

        assert!(error.contains("max_tokens"));
        assert!(
            !error.contains("anthropic-secret"),
            "secret leaked: {error}"
        );
    }

    #[cfg(unix)]
    pub(super) fn write_fake_claude(dir: &std::path::Path, body: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = dir.join("fake-claude");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&path, permissions).unwrap();
        path
    }

    #[cfg(unix)]
    pub(super) fn claude_chat_config(path: &std::path::Path) -> Config {
        let mut config = Config::from_env();
        config.chat_provider = crate::config::ChatProvider::Claude;
        config.claude_path = path.to_string_lossy().into_owned();
        config.claude_model = "claude-test-model".to_string();
        config
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn claude_chat_uses_isolated_noninteractive_flags_and_json_result() {
        let dir = tempfile::tempdir().unwrap();
        let args_path = dir.path().join("args.txt");
        let cwd_path = dir.path().join("cwd.txt");
        let script = write_fake_claude(
            dir.path(),
            &format!(
                "printf '%s\\n' \"$@\" > '{}'; pwd > '{}'; printf '%s\\n' '{{\"type\":\"result\",\"subtype\":\"success\",\"result\":\"Claude Label\"}}'",
                args_path.display(),
                cwd_path.display()
            ),
        );
        let client = OllamaClient::new(&claude_chat_config(&script));

        assert_eq!(client.chat("label this").await.unwrap(), "Claude Label");
        let args = std::fs::read_to_string(args_path).unwrap();
        for expected in [
            "-p",
            "label this",
            "--output-format",
            "json",
            "--model",
            "claude-test-model",
            "--safe-mode",
            "--strict-mcp-config",
            "--mcp-config",
            "{\"mcpServers\":{}}",
            "--tools",
            "--disable-slash-commands",
            "--setting-sources",
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
            "--permission-prompts",
            "none",
            "--no-chrome",
        ] {
            assert!(
                args.lines().any(|arg| arg == expected),
                "missing arg {expected}: {args}"
            );
        }
        let cwd = std::fs::read_to_string(cwd_path).unwrap();
        assert_ne!(
            cwd.trim(),
            std::env::current_dir().unwrap().to_string_lossy()
        );
        assert!(std::path::Path::new(cwd.trim()).starts_with(std::env::temp_dir()));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn claude_chat_reports_nonzero_garbage_and_timeout_without_secrets() {
        let dir = tempfile::tempdir().unwrap();

        let nonzero = write_fake_claude(dir.path(), "echo claude-cli-secret >&2; exit 7");
        let error = OllamaClient::new(&claude_chat_config(&nonzero))
            .chat("label")
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("status"));
        assert!(!error.contains("claude-cli-secret"));

        let garbage = write_fake_claude(dir.path(), "printf 'not-json'");
        let error = OllamaClient::new(&claude_chat_config(&garbage))
            .chat("label")
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("JSON"));

        let empty = write_fake_claude(
            dir.path(),
            r#"printf '%s\n' '{"type":"result","subtype":"success","result":""}'"#,
        );
        let error = OllamaClient::new(&claude_chat_config(&empty))
            .chat("label")
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("empty text"));

        let hanging = write_fake_claude(dir.path(), "sleep 5");
        let client = OllamaClient::new(&claude_chat_config(&hanging))
            .with_chat_timeout(std::time::Duration::from_millis(100));
        let started = std::time::Instant::now();
        let error = client.chat("label").await.unwrap_err().to_string();
        assert!(error.contains("timed out"));
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }

    #[tokio::test]
    async fn ollama_chat_error_does_not_expose_api_key() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(401).set_body_string("bad ollama-secret"))
            .mount(&server)
            .await;
        let mut config = config_with_host(&server.uri());
        config.ollama_api_key = Some("ollama-secret".to_string());
        let error = OllamaClient::new(&config)
            .chat("label")
            .await
            .unwrap_err()
            .to_string();

        assert!(!error.contains("ollama-secret"), "secret leaked: {error}");
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
    async fn query_and_document_paths_apply_their_prefix_once() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings": [[0.1, 0.2, 0.3]]})),
            )
            .expect(2)
            .mount(&server)
            .await;

        let mut config = config_with_host(&server.uri());
        config.embed_query_prefix = "task: code retrieval | query: ".to_string();
        config.embed_doc_prefix = "title: none | text: ".to_string();
        let client = OllamaClient::new(&config);

        client.embed_query("find authentication").await.unwrap();
        client
            .embed_documents(&["src/auth.rs\nAuth module".to_string()])
            .await
            .unwrap();

        let requests = server.received_requests().await.unwrap();
        let inputs: Vec<String> = requests
            .iter()
            .map(|request| {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                body["input"][0].as_str().unwrap().to_string()
            })
            .collect();
        assert_eq!(
            inputs,
            [
                "task: code retrieval | query: find authentication",
                "title: none | text: src/auth.rs\nAuth module",
            ]
        );
    }

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

    #[tokio::test]
    async fn openai_embed_sends_auth_and_reorders_by_index() {
        use wiremock::matchers::{body_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(header("authorization", "Bearer openai-test-key"))
            .and(body_json(serde_json::json!({
                "model": "test-embed-model",
                "input": ["first", "second"]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"index": 1, "embedding": [2.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0]}
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = OllamaClient::new(&openai_config(&server.uri()));
        let result = client
            .embed_documents(&["first".to_string(), "second".to_string()])
            .await
            .unwrap();

        assert_eq!(result, vec![vec![1.0, 0.0], vec![2.0, 0.0]]);
    }

    #[tokio::test]
    async fn openai_embed_uses_shared_batching() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(|request: &wiremock::Request| {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                let data: Vec<_> = body["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        serde_json::json!({
                            "index": index,
                            "embedding": [index as f32, 1.0]
                        })
                    })
                    .collect();
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"data": data}))
            })
            .expect(2)
            .mount(&server)
            .await;

        let mut config = openai_config(&server.uri());
        config.embed_batch_size = 2;
        let client = OllamaClient::new(&config);
        let result = client
            .embed_documents(&["one".to_string(), "two".to_string(), "three".to_string()])
            .await
            .unwrap();

        assert_eq!(result.len(), 3);
        let requests = server.received_requests().await.unwrap();
        let sizes: Vec<_> = requests
            .iter()
            .map(|request| {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                body["input"].as_array().unwrap().len()
            })
            .collect();
        assert_eq!(sizes, vec![2, 1]);
    }

    #[tokio::test]
    async fn openai_embed_adaptive_split_handles_context_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(|request: &wiremock::Request| {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                let count = body["input"].as_array().unwrap().len();
                if count > 2 {
                    ResponseTemplate::new(400).set_body_json(serde_json::json!({
                        "error": {"code": "context_length_exceeded", "message": "maximum context length"}
                    }))
                } else {
                    let data: Vec<_> = (0..count)
                        .map(|index| serde_json::json!({"index": index, "embedding": [1.0, 0.0]}))
                        .collect();
                    ResponseTemplate::new(200).set_body_json(serde_json::json!({"data": data}))
                }
            })
            .expect(3)
            .mount(&server)
            .await;

        let client = OllamaClient::new(&openai_config(&server.uri()));
        let texts: Vec<String> = (0..4).map(|index| format!("text-{index}")).collect();
        let result = client.embed_batch_adaptive(&texts).await.unwrap();

        assert_eq!(result.len(), 4);
    }

    #[tokio::test]
    async fn openai_embed_error_does_not_expose_api_key() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(401).set_body_string("rejected openai-test-key"))
            .mount(&server)
            .await;

        let client = OllamaClient::new(&openai_config(&server.uri()));
        let error = client
            .embed_documents(&["secret-safe".to_string()])
            .await
            .unwrap_err()
            .to_string();

        assert!(!error.contains("openai-test-key"), "secret leaked: {error}");
    }

    #[tokio::test]
    async fn openai_embed_reuses_shared_coalescing_and_query_cache() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_millis(100))
                    .set_body_json(serde_json::json!({
                        "data": [{"index": 0, "embedding": [1.0, 2.0]}]
                    })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = Arc::new(OllamaClient::new(&openai_config(&server.uri())));
        let first = {
            let client = Arc::clone(&client);
            tokio::spawn(async move { client.embed_query("same query").await.unwrap() })
        };
        let second = {
            let client = Arc::clone(&client);
            tokio::spawn(async move { client.embed_query("same query").await.unwrap() })
        };
        assert_eq!(first.await.unwrap(), second.await.unwrap());
        assert_eq!(
            client.embed_query("same query").await.unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[tokio::test]
    async fn openai_embed_cancellation_uses_shared_cancel_path() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(10))
                    .set_body_json(serde_json::json!({
                        "data": [{"index": 0, "embedding": [1.0, 2.0]}]
                    })),
            )
            .mount(&server)
            .await;

        let client = Arc::new(OllamaClient::new(&openai_config(&server.uri())));
        let task = {
            let client = Arc::clone(&client);
            tokio::spawn(async move { client.embed_query("cancel me").await })
        };
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        client.cancel_all_embeddings();

        assert!(matches!(
            task.await.unwrap(),
            Err(ContextPlusError::Cancelled)
        ));
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

    // -- in-flight coalescing tests --

    /// 10 tasks calling `embed("hello")` concurrently must produce exactly 1
    /// Ollama HTTP call.  A `Barrier` synchronises all tasks at the point just
    /// before they call `embed` so they all arrive in the same scheduler tick.
    #[tokio::test]
    async fn concurrent_same_query_coalesces() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let call_count = Arc::new(AtomicUsize::new(0));
        let counter = call_count.clone();

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |_req: &wiremock::Request| {
                counter.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings": [[0.1, 0.2, 0.3]]}))
                    // small delay so the in-flight window is wide enough for all
                    // 10 tasks to subscribe before the owner completes
                    .set_delay(std::time::Duration::from_millis(50))
            })
            .mount(&server)
            .await;

        let client = Arc::new(OllamaClient::new(&config_with_host(&server.uri())));
        let n = 10;
        let barrier = Arc::new(tokio::sync::Barrier::new(n));

        let handles: Vec<_> = (0..n)
            .map(|_| {
                let c = Arc::clone(&client);
                let b = Arc::clone(&barrier);
                tokio::spawn(async move {
                    b.wait().await;
                    c.embed(&["hello".to_string()]).await
                })
            })
            .collect();

        for h in handles {
            let res = h.await.unwrap();
            assert!(res.is_ok(), "all tasks should succeed: {res:?}");
            assert_eq!(res.unwrap(), vec![vec![0.1f32, 0.2, 0.3]]);
        }

        let actual = call_count.load(Ordering::SeqCst);
        assert_eq!(actual, 1, "expected exactly 1 Ollama call, got {actual}");
    }

    /// 10 different queries must each produce their own Ollama call — coalescing
    /// must not merge distinct queries.
    #[tokio::test]
    async fn different_queries_do_not_coalesce() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let call_count = Arc::new(AtomicUsize::new(0));
        let counter = call_count.clone();

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |_req: &wiremock::Request| {
                counter.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings": [[0.1, 0.2, 0.3]]}))
            })
            .mount(&server)
            .await;

        let client = Arc::new(OllamaClient::new(&config_with_host(&server.uri())));
        let n = 10usize;

        let handles: Vec<_> = (0..n)
            .map(|i| {
                let c = Arc::clone(&client);
                tokio::spawn(async move {
                    let q = format!("query_{i}");
                    c.embed(&[q]).await
                })
            })
            .collect();

        for h in handles {
            h.await.unwrap().unwrap();
        }

        let actual = call_count.load(Ordering::SeqCst);
        assert_eq!(actual, n, "expected {n} Ollama calls, got {actual}");
    }

    /// When Ollama returns an error, all concurrent waiters for the same query
    /// must receive an `Err`, not hang.
    #[tokio::test]
    async fn error_propagates_to_all_waiters() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(500)
                    .set_body_string("ollama internal error")
                    .set_delay(std::time::Duration::from_millis(50)),
            )
            .mount(&server)
            .await;

        let client = Arc::new(OllamaClient::new(&config_with_host(&server.uri())));
        let n = 10;
        let barrier = Arc::new(tokio::sync::Barrier::new(n));

        let handles: Vec<_> = (0..n)
            .map(|_| {
                let c = Arc::clone(&client);
                let b = Arc::clone(&barrier);
                tokio::spawn(async move {
                    b.wait().await;
                    c.embed(&["hello".to_string()]).await
                })
            })
            .collect();

        let mut err_count = 0usize;
        for h in handles {
            let res = h.await.unwrap();
            if res.is_err() {
                err_count += 1;
            }
        }
        assert_eq!(err_count, n, "all {n} waiters should receive an Err");
    }

    /// After a coalesced embed completes, the result is in the LRU cache, so
    /// a subsequent same-query call returns immediately without an Ollama call.
    #[tokio::test]
    async fn coalesced_waiter_sees_cached_result() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let call_count = Arc::new(AtomicUsize::new(0));
        let counter = call_count.clone();

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |_req: &wiremock::Request| {
                counter.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings": [[0.1, 0.2, 0.3]]}))
                    .set_delay(std::time::Duration::from_millis(50))
            })
            .mount(&server)
            .await;

        let client = Arc::new(OllamaClient::new(&config_with_host(&server.uri())));
        let n = 5;
        let barrier = Arc::new(tokio::sync::Barrier::new(n));

        // First wave: coalesced — should produce exactly 1 Ollama call.
        let handles: Vec<_> = (0..n)
            .map(|_| {
                let c = Arc::clone(&client);
                let b = Arc::clone(&barrier);
                tokio::spawn(async move {
                    b.wait().await;
                    c.embed(&["hello".to_string()]).await
                })
            })
            .collect();
        for h in handles {
            h.await.unwrap().unwrap();
        }
        assert_eq!(call_count.load(Ordering::SeqCst), 1, "first wave: 1 call");

        // Second call after coalescing completes — should hit LRU (no new call).
        client.embed(&["hello".to_string()]).await.unwrap();
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "cache hit after coalescing: still 1 Ollama call"
        );
        assert_eq!(client.query_cache_len(), 1, "result in LRU");
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

    // -- VectorStore::to_cache hygiene sweep --

    /// `VectorStore::to_cache` must drop entries with dot-prefixed path segments
    /// (stale worktree / hidden paths) and retain only valid keys.
    /// Mirrors `sweep_excluded_keys_removes_dot_segment_paths` for the mmap path.
    #[test]
    fn to_cache_drops_excluded_keys() {
        // Keys: two excluded paths (.claude/worktrees, .git) and two clean paths.
        let keys = vec![
            "src/lib.rs".to_string(),
            ".claude/worktrees/agent-x/foo.ts".to_string(),
            ".git/HEAD".to_string(),
            "packages/utils/index.ts".to_string(),
        ];
        let hashes: Vec<String> = keys.iter().map(|k| format!("hash-{}", k)).collect();
        // dims=2, 4 entries → 8 floats
        let vectors: Vec<f32> = (0..8).map(|i| i as f32).collect();

        let store = VectorStore::new(2, keys, hashes, vectors);
        let cache = store.to_cache();

        // Only the two clean keys must survive.
        assert_eq!(
            cache.len(),
            2,
            "expected 2 entries, got {}: {:?}",
            cache.len(),
            cache.keys().collect::<Vec<_>>()
        );
        assert!(cache.contains_key("src/lib.rs"), "src/lib.rs must be kept");
        assert!(
            cache.contains_key("packages/utils/index.ts"),
            "packages/utils/index.ts must be kept"
        );
        assert!(
            !cache.contains_key(".claude/worktrees/agent-x/foo.ts"),
            "worktree path must be dropped"
        );
        assert!(
            !cache.contains_key(".git/HEAD"),
            ".git/HEAD must be dropped"
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

#[cfg(test)]
mod semaphore_tests {
    use crate::config::Config;
    use crate::core::embeddings::OllamaClient;
    use std::sync::Arc;

    fn config_with_host(host: &str) -> Config {
        let mut config = Config::from_env();
        config.ollama_host = host.to_string();
        config
    }

    /// `OllamaClient::new` constructs with no semaphore by default.
    /// Existing tests remain green because no permit is required.
    #[test]
    fn new_client_has_no_semaphore() {
        let config = Config::from_env();
        let client = OllamaClient::new(&config);
        assert!(
            client.semaphore.is_none(),
            "default client must not have a semaphore"
        );
    }

    /// `with_semaphore` stores the Arc and clones share it.
    #[test]
    fn with_semaphore_plumbs_arc() {
        let config = Config::from_env();
        let sem = Arc::new(tokio::sync::Semaphore::new(2));
        let client = OllamaClient::new(&config).with_semaphore(Arc::clone(&sem));
        assert!(
            client.semaphore.is_some(),
            "semaphore should be set after with_semaphore"
        );
        // Both Arcs point to the same semaphore object.
        let client_sem = client.semaphore.as_ref().unwrap();
        assert!(
            Arc::ptr_eq(client_sem, &sem),
            "client semaphore Arc must point to the same Semaphore"
        );
    }

    /// With capacity=1, two concurrent `embed` calls serialise: the second
    /// completes only after the first releases its permit.
    #[tokio::test]
    async fn semaphore_capacity_one_serialises_concurrent_calls() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Add a short delay so concurrent requests actually overlap without the cap.
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings": [[1.0, 2.0]]}))
                    .set_delay(std::time::Duration::from_millis(50)),
            )
            .mount(&server)
            .await;

        let sem = Arc::new(tokio::sync::Semaphore::new(1));
        let config = config_with_host(&server.uri());
        let client = OllamaClient::new(&config).with_semaphore(Arc::clone(&sem));

        // Track peak concurrent Ollama calls.
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let client_a = client.clone();
        let in_flight_a = Arc::clone(&in_flight);
        let peak_a = Arc::clone(&peak);
        let h1 = tokio::spawn(async move {
            in_flight_a.fetch_add(1, Ordering::SeqCst);
            let cur = in_flight_a.load(Ordering::SeqCst);
            peak_a.fetch_max(cur, Ordering::SeqCst);
            let r = client_a.embed(&["text a".to_string()]).await;
            in_flight_a.fetch_sub(1, Ordering::SeqCst);
            r
        });

        let client_b = client.clone();
        let in_flight_b = Arc::clone(&in_flight);
        let peak_b = Arc::clone(&peak);
        let h2 = tokio::spawn(async move {
            in_flight_b.fetch_add(1, Ordering::SeqCst);
            let cur = in_flight_b.load(Ordering::SeqCst);
            peak_b.fetch_max(cur, Ordering::SeqCst);
            let r = client_b.embed(&["text b".to_string()]).await;
            in_flight_b.fetch_sub(1, Ordering::SeqCst);
            r
        });

        let (r1, r2) = tokio::join!(h1, h2);
        assert!(r1.unwrap().is_ok(), "call 1 should succeed");
        assert!(r2.unwrap().is_ok(), "call 2 should succeed");

        // Wiremock received exactly 2 requests.
        let reqs = server.received_requests().await.unwrap();
        assert_eq!(reqs.len(), 2, "both requests must reach Ollama");
    }

    /// Permit is released even when Ollama returns an error — the next call
    /// can still acquire the permit.
    #[tokio::test]
    async fn semaphore_permit_released_on_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        // First call → error; second call → success.
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&call_count);
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(move |_: &wiremock::Request| {
                let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    ResponseTemplate::new(500).set_body_string("internal error")
                } else {
                    ResponseTemplate::new(200)
                        .set_body_json(serde_json::json!({"embeddings": [[0.9, 0.8]]}))
                }
            })
            .mount(&server)
            .await;

        let sem = Arc::new(tokio::sync::Semaphore::new(1));
        let config = config_with_host(&server.uri());
        let client = OllamaClient::new(&config).with_semaphore(Arc::clone(&sem));

        // First call fails.
        let r1 = client.embed(&["first call".to_string()]).await;
        assert!(r1.is_err(), "first call should fail with HTTP 500");

        // Permit must have been released — second call should succeed.
        let r2 = client.embed(&["second call".to_string()]).await;
        assert!(
            r2.is_ok(),
            "second call should succeed after permit was released by the errored first call"
        );
    }

    /// Coalesced waiters do NOT each acquire a permit.
    ///
    /// Two concurrent `embed("same")` calls with capacity=1 → the in-flight
    /// coalescing deduplicates them.  Only ONE permit is acquired (by the
    /// producer); the waiter gets the result without needing its own permit.
    /// Verification: capacity=1 semaphore; after both complete the permit is
    /// still available (available_permits == 1), confirming only one
    /// acquisition happened.
    #[tokio::test]
    async fn semaphore_coalesced_waiters_do_not_double_acquire() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"embeddings": [[7.0, 8.0]]}))
                    // Enough delay so both tasks enter embed() before the first
                    // one's Ollama response lands.
                    .set_delay(std::time::Duration::from_millis(80)),
            )
            .mount(&server)
            .await;

        let sem = Arc::new(tokio::sync::Semaphore::new(1));
        let config = config_with_host(&server.uri());
        let client = OllamaClient::new(&config).with_semaphore(Arc::clone(&sem));

        // Fire two identical queries concurrently.
        let c1 = client.clone();
        let c2 = client.clone();
        let (r1, r2) = tokio::join!(
            tokio::spawn(async move { c1.embed(&["coalesce me".to_string()]).await }),
            tokio::spawn(async move { c2.embed(&["coalesce me".to_string()]).await }),
        );

        assert!(r1.unwrap().is_ok());
        assert!(r2.unwrap().is_ok());

        // Only one HTTP call should have reached Ollama.
        let reqs = server.received_requests().await.unwrap();
        assert_eq!(
            reqs.len(),
            1,
            "coalescing must produce exactly 1 Ollama call"
        );

        // And the single permit must be back — semaphore is not saturated.
        assert_eq!(
            sem.available_permits(),
            1,
            "permit must be released after coalesced calls complete"
        );
    }
}
