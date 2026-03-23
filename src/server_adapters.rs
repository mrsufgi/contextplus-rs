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
use crate::tools::semantic_search::{EmbedFn, SearchDocument, SymbolSearchEntry, WalkAndIndexFn};

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
            let embedding_cache = &state.embedding_cache;
            let store_root = &state.root_dir;
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

            // Embed only uncached files
            if !uncached_texts.is_empty() {
                let batch_size = config.embed_batch_size.max(1);
                let mut new_vectors = Vec::with_capacity(uncached_texts.len());
                for chunk in uncached_texts.chunks(batch_size) {
                    match ollama.embed(chunk).await {
                        Ok(embeddings) => new_vectors.extend(embeddings),
                        Err(e) => {
                            tracing::warn!("Embedding batch failed: {e}, filling with None");
                            for _ in chunk {
                                new_vectors.push(Vec::new());
                            }
                        }
                    }
                }

                // Store new vectors in cache and fill result
                {
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

                    // Persist to disk with model-qualified name (lock dropped after block)
                    let code_cache_name = cache_name("embeddings", &config.ollama_embed_model);
                    if let Some(store) =
                        crate::core::embeddings::VectorStore::from_cache(&cache_write)
                        && let Err(e) =
                            rkyv_store::save_vector_store(store_root, &code_cache_name, &store)
                    {
                        tracing::warn!("Failed to save embedding cache: {e}");
                    }
                }
            }

            Ok((docs, vectors))
        })
    }
}
