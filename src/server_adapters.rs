// Adapter structs that bridge shared server state to tool function traits.
// Extracted from server.rs (Round 11D) to reduce server.rs line count.

use std::path::Path;
use std::sync::Arc;

use crate::cache::rkyv_store;
use crate::config::Config;
use crate::core::embeddings::{CacheEntry, OllamaClient, content_hash};
use crate::core::parser::detect_language;
use crate::core::tree_sitter::parse_with_tree_sitter;
use crate::core::walker::walk_with_config;
use crate::error::Result;
use crate::server::{SharedState, cache_name};
use crate::tools::semantic_search::{
    EmbedFn, MAX_TEXT_DOC_CHARS, SearchDocument, SymbolSearchEntry, WalkAndIndexFn,
    extract_plain_text_header, is_text_index_candidate,
};

// --- OllamaEmbedder ---

/// Thin adapter: wraps OllamaClient to implement the EmbedFn trait used by semantic search.
pub struct OllamaEmbedder(pub OllamaClient);

impl EmbedFn for OllamaEmbedder {
    fn embed(
        &self,
        texts: &[String],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>>
    {
        let texts = texts.to_vec();
        Box::pin(async move { self.0.embed(&texts).await })
    }
}

// --- CachedWalkerIndexer ---

/// Walk the project, read file contents, and return SearchDocuments with embedding vectors.
/// Uses the embedding cache in SharedState for warm hits — only embeds new/changed files.
pub struct CachedWalkerIndexer {
    pub config: Config,
    pub ollama: OllamaClient,
    pub state: Arc<SharedState>,
}

impl WalkAndIndexFn for CachedWalkerIndexer {
    fn walk_and_index(
        &self,
        root_dir: &Path,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)>>
                + Send
                + '_,
        >,
    > {
        let root = root_dir.to_path_buf();
        let config = self.config.clone();
        let ollama = self.ollama.clone();
        let state = self.state.clone();
        Box::pin(async move {
            // `CachedWalkerIndexer` is constructed with an `Arc<SharedState>` and has
            // no per-session `RefId`.  It always operates on the default ref.  This is
            // correct for the warmup path and for single-ref stdio mode.  Per-session
            // dispatch via `ContextPlusServer::handle_semantic_code_search` obtains the
            // ref-appropriate `search_index_cache` before constructing this walker and
            // passes it directly — so the embedding cache here only needs to be the
            // default ref's cache for the file-embedding step.
            let default_ref = state
                .default_ref()
                .expect("default_ref always present in CachedWalkerIndexer");
            // Clone the Arcs so we can drop `default_ref` and avoid lifetime issues
            // with borrowing into the async block.
            let embedding_cache = Arc::clone(&default_ref.embedding_cache);
            let store_root = default_ref.root_dir.clone();
            drop(default_ref);
            let entries = walk_with_config(&root, &config);

            let max_file_size = config.max_embed_file_size as u64;
            // Read all files concurrently (up to 32 at a time)
            let mut join_set = tokio::task::JoinSet::new();
            for (i, entry) in entries.iter().enumerate() {
                let full_path = root.join(&entry.relative_path);
                let rel_path = entry.relative_path.clone();
                join_set.spawn(async move {
                    if let Ok(meta) = tokio::fs::metadata(&full_path).await
                        && meta.len() > max_file_size
                    {
                        return (i, rel_path, None);
                    }
                    let content = tokio::fs::read_to_string(&full_path).await.ok();
                    (i, rel_path, content)
                });
            }

            // Collect results in order
            let mut file_contents: Vec<(usize, String, Option<String>)> =
                Vec::with_capacity(entries.len());
            while let Some(result) = join_set.join_next().await {
                if let Ok(item) = result {
                    file_contents.push(item);
                }
            }
            file_contents.sort_unstable_by_key(|(i, _, _)| *i);

            let mut docs = Vec::new();
            let mut content_hashes = Vec::new();

            for (_, rel_path, maybe_content) in &file_contents {
                let content = match maybe_content {
                    Some(c) => c,
                    None => continue,
                };

                // Text/data file path: index raw content for semantic search
                if is_text_index_candidate(rel_path) {
                    let truncated: String = content.chars().take(MAX_TEXT_DOC_CHARS).collect();
                    let header = extract_plain_text_header(&truncated);
                    content_hashes.push((rel_path.clone(), content_hash(&truncated)));
                    docs.push(SearchDocument::new(
                        rel_path.clone(),
                        header,
                        vec![],
                        vec![],
                        truncated,
                    ));
                    continue;
                }

                // Code file path: parse with tree-sitter
                let ext = rel_path.rsplit('.').next().unwrap_or("");
                let symbols = parse_with_tree_sitter(content, ext).unwrap_or_default();
                let header = crate::core::parser::extract_header(content);

                let symbol_names: Vec<String> = symbols.iter().map(|s| s.name.clone()).collect();
                let symbol_entries: Vec<SymbolSearchEntry> = symbols
                    .iter()
                    .map(|s| SymbolSearchEntry {
                        name: s.name.clone(),
                        kind: Some(s.kind.clone()),
                        line: s.line,
                        end_line: Some(s.end_line),
                        signature: s.signature.clone(),
                    })
                    .collect();

                let doc_content = format!(
                    "{} {}",
                    detect_language(rel_path).unwrap_or("unknown"),
                    content.chars().take(500).collect::<String>()
                );
                content_hashes.push((rel_path.clone(), content_hash(&doc_content)));

                docs.push(SearchDocument::new(
                    rel_path.clone(),
                    header,
                    symbol_names,
                    symbol_entries,
                    doc_content,
                ));
            }

            if docs.is_empty() {
                return Ok((docs, Vec::new()));
            }

            // Resolve vectors from cache, embed only uncached/stale files
            let cache_read = embedding_cache.read().await;
            let mut vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(docs.len());
            let mut uncached_indices: Vec<usize> = Vec::new();
            let mut uncached_texts: Vec<String> = Vec::new();

            for (i, (rel_path, hash)) in content_hashes.iter().enumerate() {
                if let Some(entry) = cache_read.get(rel_path)
                    && entry.hash == *hash
                {
                    vectors.push(Some(entry.vector.clone()));
                    continue;
                }
                vectors.push(None);
                uncached_indices.push(i);
                uncached_texts.push(docs[i].content.clone());
            }
            drop(cache_read);

            tracing::info!(
                cached = docs.len() - uncached_indices.len(),
                uncached = uncached_indices.len(),
                "semantic_code_search embedding cache hit/miss"
            );

            // Embed only uncached files — TIME-BOXED, PER-BATCH TIMEOUT,
            // and CONCURRENT up to `ollama_max_concurrent`.
            //
            // Three failure modes this defends against:
            //
            // 1. Many uncached batches: embedding 3000+ files sequentially at
            //    ~7s/batch on CPU Ollama can take 12-18 min. Codex's bridge
            //    times out and closes the connection.
            //
            // 2. A single batch hangs: on a slow CPU Ollama, one batch can
            //    take 60+ s. Without a per-batch timeout, a global "elapsed"
            //    check between batches never fires while one batch is stuck.
            //
            // 3. Sequential under-utilization: with `ollama_max_concurrent=4`
            //    the server can sustain 4 in-flight HTTP requests, but a
            //    one-at-a-time loop only ever uses one. Within the same 20s
            //    budget, parallel issue gets ~Nx more vectors filled before
            //    we bail.
            //
            // The pipeline:
            //   - Pre-build per-batch chunks.
            //   - Issue up to `ollama_max_concurrent` in flight at once via
            //     `JoinSet`, each wrapped in `tokio::time::timeout(remaining)`.
            //   - On every completion: queue the next pending chunk if budget
            //     remains; otherwise drain in-flight tasks (each bounded by
            //     its own remaining budget) and exit.
            //   - Reassemble results in order so `new_vectors` aligns with
            //     `uncached_indices` exactly as before.
            //
            // OllamaClient already gates HTTP concurrency via its own
            // semaphore (U17), so even if we somehow over-spawn, only
            // `ollama_max_concurrent` requests reach the wire.
            //
            // Budget is deliberately tight (20 s).
            const EMBED_BUDGET_MS: u128 = 20_000;
            if !uncached_texts.is_empty() {
                let embed_start = std::time::Instant::now();
                let batch_size = config.embed_batch_size.max(1);
                let max_concurrent = config.ollama_max_concurrent.max(1);

                let chunks: Vec<Vec<String>> = uncached_texts
                    .chunks(batch_size)
                    .map(|c| c.to_vec())
                    .collect();
                let n_chunks = chunks.len();
                let chunk_sizes: Vec<usize> = chunks.iter().map(|c| c.len()).collect();
                let mut per_chunk_results: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_chunks];

                let mut budget_exceeded = false;
                // (chunk_idx, outer = timeout result, inner = embed result)
                type EmbedBatchOutcome = (
                    usize,
                    std::result::Result<Result<Vec<Vec<f32>>>, tokio::time::error::Elapsed>,
                );
                let mut join_set: tokio::task::JoinSet<EmbedBatchOutcome> =
                    tokio::task::JoinSet::new();
                let mut next_idx = 0usize;

                // Spawn helper closure to keep the seed and drain loops in sync.
                let spawn_one = |join_set: &mut tokio::task::JoinSet<_>,
                                 idx: usize,
                                 chunk: Vec<String>,
                                 ollama: OllamaClient,
                                 remaining: std::time::Duration| {
                    join_set.spawn(async move {
                        (
                            idx,
                            tokio::time::timeout(
                                remaining,
                                async move { ollama.embed(&chunk).await },
                            )
                            .await,
                        )
                    });
                };

                // Seed: spawn up to max_concurrent batches.
                while next_idx < n_chunks && join_set.len() < max_concurrent {
                    let elapsed_ms = embed_start.elapsed().as_millis();
                    if elapsed_ms >= EMBED_BUDGET_MS {
                        budget_exceeded = true;
                        break;
                    }
                    let remaining =
                        std::time::Duration::from_millis((EMBED_BUDGET_MS - elapsed_ms) as u64);
                    let chunk = chunks[next_idx].clone();
                    spawn_one(&mut join_set, next_idx, chunk, ollama.clone(), remaining);
                    next_idx += 1;
                }

                // Drain: collect each completion, queue the next chunk while
                // budget remains, and let in-flight tasks finish under their
                // own per-batch deadline once budget is exceeded.
                while let Some(joined) = join_set.join_next().await {
                    match joined {
                        Ok((idx, Ok(Ok(embeddings)))) => {
                            per_chunk_results[idx] = embeddings;
                        }
                        Ok((idx, Ok(Err(e)))) => {
                            tracing::warn!(chunk_idx = idx, "Embedding batch failed: {e}");
                        }
                        Ok((idx, Err(_elapsed))) => {
                            tracing::warn!(
                                chunk_idx = idx,
                                "Embedding batch timed out within budget"
                            );
                            budget_exceeded = true;
                        }
                        Err(join_err) => {
                            tracing::warn!("Embedding task join failed: {join_err}");
                        }
                    }

                    let elapsed_ms = embed_start.elapsed().as_millis();
                    if elapsed_ms >= EMBED_BUDGET_MS {
                        budget_exceeded = true;
                    }
                    if !budget_exceeded && next_idx < n_chunks {
                        let remaining =
                            std::time::Duration::from_millis((EMBED_BUDGET_MS - elapsed_ms) as u64);
                        let chunk = chunks[next_idx].clone();
                        spawn_one(&mut join_set, next_idx, chunk, ollama.clone(), remaining);
                        next_idx += 1;
                    }
                }

                // Flatten per-chunk results back to a flat `new_vectors`
                // aligned with `uncached_indices`. Missing slots (failed
                // batches, never scheduled, timed out) become empty vectors;
                // the cache writer below skips them.
                let mut new_vectors: Vec<Vec<f32>> = Vec::with_capacity(uncached_texts.len());
                for (i, results) in per_chunk_results.iter_mut().enumerate() {
                    let expected = chunk_sizes[i];
                    if results.len() == expected {
                        new_vectors.append(results);
                    } else {
                        for _ in 0..expected {
                            new_vectors.push(Vec::new());
                        }
                    }
                }

                let chunks_skipped = next_idx < n_chunks;
                if budget_exceeded || chunks_skipped {
                    let still_uncached = new_vectors.iter().filter(|v| v.is_empty()).count();
                    tracing::warn!(
                        budget_ms = EMBED_BUDGET_MS as u64,
                        elapsed_ms = embed_start.elapsed().as_millis() as u64,
                        embedded = uncached_texts.len() - still_uncached,
                        skipped = still_uncached,
                        chunks_scheduled = next_idx,
                        chunks_total = n_chunks,
                        concurrency = max_concurrent,
                        "semantic_code_search embed budget exceeded — returning partial results; tracker will fill remaining in background"
                    );
                }

                // Store new vectors in cache and build the snapshot to persist.
                // Critical: build the VectorStore inside the write-guard scope,
                // then drop the guard BEFORE the disk save. The merging save
                // takes a blocking fd-lock and does ~146 MB of sync I/O — if we
                // held the in-memory write lock across that, every concurrent
                // request waiting on `embedding_cache.read().await` would stall.
                let store_to_save = {
                    let mut cache_write = embedding_cache.write().await;
                    for (j, &idx) in uncached_indices.iter().enumerate() {
                        if j < new_vectors.len() && !new_vectors[j].is_empty() {
                            let (rel_path, hash) = &content_hashes[idx];
                            cache_write.insert(
                                rel_path.clone(),
                                CacheEntry {
                                    hash: hash.clone(),
                                    vector: new_vectors[j].clone(),
                                },
                            );
                            vectors[idx] = Some(new_vectors[j].clone());
                        }
                    }
                    let store = crate::core::embeddings::VectorStore::from_cache(&cache_write);
                    drop(cache_write);
                    store
                };

                // Persist to disk off the Tokio worker via spawn_blocking — the
                // merging save acquires a blocking fd-lock and does sync I/O.
                // Merge with disk under fd-lock: this adapter's in-memory cache
                // only covers keys touched in this session, so an overwrite would
                // silently drop entries written by warmup_embeddings or a second
                // MCP instance racing on the same cache file.
                if let Some(store) = store_to_save {
                    let code_cache_name = cache_name("embeddings", &config.ollama_embed_model);
                    let root = store_root.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        rkyv_store::save_vector_store_merged(&root, &code_cache_name, &store)
                    })
                    .await;
                    match result {
                        Ok(Err(e)) => tracing::warn!("Failed to save embedding cache: {e}"),
                        Err(join_err) => tracing::warn!(
                            "save_vector_store spawn_blocking join failed: {join_err}"
                        ),
                        Ok(Ok(())) => {}
                    }
                }
            }

            Ok((docs, vectors))
        })
    }
}
