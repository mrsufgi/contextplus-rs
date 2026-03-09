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
use tokio::sync::RwLock;

use crate::cache::rkyv_store;
use crate::config::Config;
use crate::core::embeddings::{CacheEntry, OllamaClient};
use crate::core::memory_graph::GraphStore;
use crate::core::parser::detect_language;
use crate::core::tree_sitter::parse_with_tree_sitter;
use crate::core::walker::walk_with_config;
use crate::error::{ContextPlusError, Result};
use crate::tools::semantic_search::EmbedFn;

/// Cached project state: walked file entries and their line contents.
/// Built lazily on first tool call, invalidated by file watcher or TTL expiry.
pub struct ProjectCache {
    pub file_entries: Vec<crate::core::walker::FileEntry>,
    pub file_lines: HashMap<String, Vec<String>>,
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

/// Shared state accessible by all tool handlers.
pub struct SharedState {
    pub config: Config,
    pub root_dir: PathBuf,
    pub ollama: OllamaClient,
    pub memory_graph: GraphStore,
    pub project_cache: RwLock<Option<Arc<ProjectCache>>>,
    /// Cached embedding vectors keyed by relative file path.
    /// Uses CacheEntry (hash + vector) for content-hash invalidation.
    /// Persisted to disk via rkyv_store for cross-restart survival.
    pub embedding_cache: RwLock<HashMap<String, CacheEntry>>,
    /// Cached identifier index — avoids re-embedding all symbols on each call.
    pub identifier_index: RwLock<Option<Arc<IdentifierIndex>>>,
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
        let ollama = OllamaClient::new(&config);
        let memory_graph = GraphStore::new();

        let embed_cache_name = cache_name("embeddings", &config.ollama_embed_model);

        // Load embedding cache from disk if available (cross-restart persistence)
        let initial_cache = match rkyv_store::load_vector_store(&root_dir, &embed_cache_name) {
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

        let state = Arc::new(SharedState {
            config,
            root_dir,
            ollama,
            memory_graph,
            project_cache: RwLock::new(None),
            embedding_cache: RwLock::new(initial_cache),
            identifier_index: RwLock::new(None),
        });
        Self { state }
    }

    fn root_dir(&self) -> &Path {
        &self.state.root_dir
    }

    fn get_str(args: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
        args.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn get_str_or(args: &serde_json::Map<String, Value>, key: &str, default: &str) -> String {
        Self::get_str(args, key).unwrap_or_else(|| default.to_string())
    }

    fn get_usize(args: &serde_json::Map<String, Value>, key: &str) -> Option<usize> {
        args.get(key).and_then(|v| v.as_u64()).map(|n| n as usize)
    }

    fn get_f64(args: &serde_json::Map<String, Value>, key: &str) -> Option<f64> {
        args.get(key).and_then(|v| v.as_f64())
    }

    fn get_bool(args: &serde_json::Map<String, Value>, key: &str) -> Option<bool> {
        args.get(key).and_then(|v| v.as_bool())
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
            let file_lines: HashMap<String, Vec<String>> = entries
                .par_iter()
                .filter(|entry| !entry.is_directory)
                .filter_map(|entry| {
                    let full_path = root.join(&entry.relative_path);
                    std::fs::read_to_string(&full_path).ok().map(|content| {
                        let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
                        (entry.relative_path.clone(), lines)
                    })
                })
                .collect();
            ProjectCache {
                file_entries: entries,
                file_lines,
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

        let mut texts_to_embed: Vec<(String, String, String)> = Vec::new(); // (rel_path, hash, text)

        for file_path in files {
            let rel_path = match file_path.strip_prefix(&self.state.root_dir) {
                Ok(r) => r.to_string_lossy().to_string(),
                Err(_) => file_path.to_string_lossy().to_string(),
            };

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
            let header =
                crate::core::parser::extract_header(&truncated.lines().collect::<Vec<_>>());
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

                // Persist to disk outside the write lock scope
                let store = crate::core::embeddings::VectorStore::from_cache(&cache);
                drop(cache);
                let embed_cache_name =
                    cache_name("embeddings", &self.state.config.ollama_embed_model);
                if let Some(s) = store
                    && let Err(e) =
                        rkyv_store::save_vector_store(&self.state.root_dir, &embed_cache_name, &s)
                {
                    tracing::warn!("Failed to save incremental embedding cache: {e}");
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
            let mut docs = Vec::new();
            for entry in &cache_clone.file_entries {
                if entry.is_directory {
                    continue;
                }
                if let Some(lines) = cache_clone.file_lines.get(&entry.relative_path) {
                    let content = lines.join("\n");
                    let ext = entry.relative_path.rsplit('.').next().unwrap_or("");
                    if let Ok(symbols) = parse_with_tree_sitter(&content, ext) {
                        let line_refs: Vec<&str> = lines.iter().map(|l| l.as_str()).collect();
                        let header = crate::core::parser::extract_header(&line_refs);
                        for sym in crate::core::parser::flatten_symbols(&symbols, None) {
                            let sig = sym.signature.clone().unwrap_or_default();
                            let text = format!(
                                "{} {} {} {}",
                                sym.name, sym.kind, entry.relative_path, sig
                            );
                            docs.push(crate::tools::semantic_identifiers::IdentifierDoc {
                                id: format!("{}:{}:{}", entry.relative_path, sym.name, sym.line),
                                path: entry.relative_path.clone(),
                                header: header.clone(),
                                name: sym.name.clone(),
                                kind: sym.kind.clone(),
                                line: sym.line,
                                end_line: sym.end_line,
                                signature: sig,
                                parent_name: sym.parent_name.clone(),
                                text,
                            });
                        }
                    }
                }
            }
            docs
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

        // Embed only uncached identifiers
        if !uncached_texts.is_empty() {
            let new_vectors = self.state.ollama.embed(&uncached_texts).await?;
            for (j, &idx) in uncached_indices.iter().enumerate() {
                if j < new_vectors.len() {
                    result_vectors[idx] = Some(new_vectors[j].clone());
                }
            }

            // Persist updated identifier embedding cache to disk
            let all_vecs: Vec<Vec<f32>> = result_vectors.iter().filter_map(|v| v.clone()).collect();
            if all_vecs.len() == n_identifiers {
                let dims = all_vecs.first().map_or(0, |v| v.len()) as u32;
                let keys: Vec<String> = identifier_docs.iter().map(|d| d.text.clone()).collect();
                let hashes: Vec<String> = keys
                    .iter()
                    .map(|k| crate::core::parser::hash_content(k))
                    .collect();
                let flat: Vec<f32> = all_vecs.into_iter().flatten().collect();
                let store = crate::core::embeddings::VectorStore::new(dims, keys, hashes, flat);
                if let Err(e) =
                    rkyv_store::save_vector_store(&self.state.root_dir, &id_cache_name, &store)
                {
                    tracing::warn!("Failed to save identifier embedding cache: {e}");
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
        let root_clone = root.clone();
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
                if let Some(lines) = cache.file_lines.get(&entry.relative_path) {
                    let content = lines.join("\n");
                    let ext = entry.relative_path.rsplit('.').next().unwrap_or("");
                    if let Ok(symbols) = parse_with_tree_sitter(&content, ext) {
                        let line_refs: Vec<&str> = lines.iter().map(|l| l.as_str()).collect();
                        let header = crate::core::parser::extract_header(&line_refs);
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
            let _ = &root_clone; // ensure root_clone lives through the closure
            (ct_entries, ct_analyses)
        })
        .await
        .map_err(|e| ContextPlusError::Other(format!("spawn_blocking failed: {e}")))?;

        let options = ct::ContextTreeOptions {
            root_dir: root,
            target_path: Self::get_str(&args, "targetPath"),
            depth_limit: Self::get_usize(&args, "depthLimit"),
            include_symbols: Self::get_bool(&args, "includeSymbols"),
            max_tokens: Self::get_usize(&args, "maxTokens"),
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
            .or_else(|| Self::get_str(&args, "filePath"))
            .or_else(|| Self::get_str(&args, "target_path"))
            .or_else(|| Self::get_str(&args, "targetPath"))
            .ok_or_else(|| ContextPlusError::Other("file_path is required".into()))?;

        let root = self.resolve_root(&args);
        let full_path = root.join(&file_path);

        let content = tokio::fs::read_to_string(&full_path).await.ok();
        let content_ref = content.as_deref();

        let analysis = content_ref.and_then(|c| {
            let ext = file_path.rsplit('.').next().unwrap_or("");
            let symbols = parse_with_tree_sitter(c, ext).ok()?;
            let header = crate::core::parser::extract_header(&c.lines().collect::<Vec<_>>());
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
            .or_else(|| Self::get_str(&args, "symbolName"))
            .ok_or_else(|| ContextPlusError::Other("symbol_name is required".into()))?;
        let file_context =
            Self::get_str(&args, "file_context").or_else(|| Self::get_str(&args, "fileContext"));

        let cache = self.ensure_project_cache().await?;

        let result = crate::tools::blast_radius::find_symbol_usages(
            &symbol_name,
            file_context.as_deref(),
            &cache.file_lines,
        );
        let formatted = crate::tools::blast_radius::format_blast_radius(&symbol_name, &result);
        Ok(Self::ok_text(formatted))
    }

    async fn handle_semantic_code_search(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let query = Self::get_str(&args, "query")
            .ok_or_else(|| ContextPlusError::Other("query is required".into()))?;
        let root = self.resolve_root(&args);

        let options = crate::tools::semantic_search::SemanticSearchOptions {
            root_dir: root.clone(),
            query,
            top_k: Self::get_usize(&args, "top_k").or_else(|| Self::get_usize(&args, "topK")),
            semantic_weight: Self::get_f64(&args, "semantic_weight")
                .or_else(|| Self::get_f64(&args, "semanticWeight")),
            keyword_weight: Self::get_f64(&args, "keyword_weight")
                .or_else(|| Self::get_f64(&args, "keywordWeight")),
            min_semantic_score: Self::get_f64(&args, "min_semantic_score")
                .or_else(|| Self::get_f64(&args, "minSemanticScore")),
            min_keyword_score: Self::get_f64(&args, "min_keyword_score")
                .or_else(|| Self::get_f64(&args, "minKeywordScore")),
            min_combined_score: Self::get_f64(&args, "min_combined_score")
                .or_else(|| Self::get_f64(&args, "minCombinedScore")),
            require_keyword_match: Self::get_bool(&args, "require_keyword_match")
                .or_else(|| Self::get_bool(&args, "requireKeywordMatch")),
            require_semantic_match: Self::get_bool(&args, "require_semantic_match")
                .or_else(|| Self::get_bool(&args, "requireSemanticMatch")),
        };

        let embedder = OllamaEmbedder(self.state.ollama.clone());
        let walker = CachedWalkerIndexer {
            config: self.state.config.clone(),
            ollama: self.state.ollama.clone(),
            state: self.state.clone(),
        };

        let result =
            crate::tools::semantic_search::semantic_code_search(options, &embedder, &walker)
                .await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_semantic_identifier_search(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
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
            top_k: Self::get_usize(&args, "top_k").or_else(|| Self::get_usize(&args, "topK")),
            top_calls_per_identifier: Self::get_usize(&args, "top_calls_per_identifier")
                .or_else(|| Self::get_usize(&args, "topCallsPerIdentifier")),
            semantic_weight: Self::get_f64(&args, "semantic_weight")
                .or_else(|| Self::get_f64(&args, "semanticWeight")),
            keyword_weight: Self::get_f64(&args, "keyword_weight")
                .or_else(|| Self::get_f64(&args, "keywordWeight")),
            include_kinds: args
                .get("include_kinds")
                .or_else(|| args.get("includeKinds"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                }),
        };

        let result = semantic_identifier_search(
            options,
            &OllamaEmbedder(self.state.ollama.clone()),
            &idx.docs,
            &idx.vector_buffer,
            idx.dims,
            &cache.file_lines,
        )
        .await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_semantic_navigate(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        // query is optional in TS version — accepted but unused for clustering
        let _query = Self::get_str(&args, "query");
        let root = self.resolve_root(&args);

        let options = crate::tools::semantic_navigate::SemanticNavigateOptions {
            root_dir: root.to_string_lossy().into(),
            max_depth: Self::get_usize(&args, "max_depth")
                .or_else(|| Self::get_usize(&args, "maxDepth")),
            max_clusters: Self::get_usize(&args, "max_clusters")
                .or_else(|| Self::get_usize(&args, "maxClusters")),
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
        let hub_path = Self::get_str(&args, "hubPath");

        let options = crate::tools::feature_hub::FeatureHubOptions {
            root_dir: root.to_string_lossy().into(),
            hub_path,
            feature_name: Self::get_str(&args, "featureName"),
            show_orphans: Self::get_bool(&args, "showOrphans"),
        };

        let result = crate::tools::feature_hub::get_feature_hub(options).await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_static_analysis(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let root = self.resolve_root(&args);
        let target_path = Self::get_str(&args, "targetPath");

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
            .or_else(|| Self::get_str(&args, "filePath"))
            .ok_or_else(|| ContextPlusError::Other("file_path is required".into()))?;
        let content = Self::get_str(&args, "new_content")
            .or_else(|| Self::get_str(&args, "content"))
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
            output.push_str(&format!(
                "  {} (ts: {}, {})\n    Files: {}\n\n",
                pt.id,
                pt.timestamp,
                pt.description,
                file_names.join(", ")
            ));
        }
        Ok(Self::ok_text(output))
    }

    async fn handle_undo_change(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let restore_point_id = Self::get_str(&args, "point_id")
            .or_else(|| Self::get_str(&args, "restorePointId"))
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
            node_type: Self::get_str(&args, "type")
                .or_else(|| Self::get_str(&args, "nodeType"))
                .unwrap_or_else(|| "concept".to_string()),
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
            source_id: Self::get_str(&args, "source_id")
                .or_else(|| Self::get_str(&args, "sourceId")),
            source_label: Self::get_str(&args, "source_label")
                .or_else(|| Self::get_str(&args, "sourceLabel")),
            source_type: Self::get_str(&args, "source_type")
                .or_else(|| Self::get_str(&args, "sourceType"))
                .unwrap_or_else(|| "concept".to_string()),
            target_id: Self::get_str(&args, "target_id")
                .or_else(|| Self::get_str(&args, "targetId")),
            target_label: Self::get_str(&args, "target_label")
                .or_else(|| Self::get_str(&args, "targetLabel")),
            target_type: Self::get_str(&args, "target_type")
                .or_else(|| Self::get_str(&args, "targetType"))
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
            max_depth: Self::get_usize(&args, "max_depth")
                .or_else(|| Self::get_usize(&args, "maxDepth")),
            top_k: Self::get_usize(&args, "top_k").or_else(|| Self::get_usize(&args, "topK")),
            edge_filter: args
                .get("edge_filter")
                .or_else(|| args.get("edgeFilter"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                }),
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
                                .or_else(|| Self::get_str(obj, "nodeType"))
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
            auto_link: Self::get_bool(&args, "auto_link")
                .or_else(|| Self::get_bool(&args, "autoLink")),
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
                .or_else(|| Self::get_str(&args, "nodeId"))
                .ok_or_else(|| ContextPlusError::Other("start_node_id is required".into()))?,
            max_depth: Self::get_usize(&args, "max_depth")
                .or_else(|| Self::get_usize(&args, "maxDepth")),
            edge_filter: args
                .get("edge_filter")
                .or_else(|| args.get("edgeFilter"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                }),
        };

        let store = &self.state.memory_graph;
        let result =
            crate::tools::memory_tools::tool_retrieve_with_traversal(store, options).await?;
        Ok(Self::ok_text(result))
    }

    // --- Helpers ---

    fn resolve_root(&self, args: &serde_json::Map<String, Value>) -> PathBuf {
        if let Some(requested) = Self::get_str(args, "rootDir") {
            let requested_path = PathBuf::from(&requested);
            // Normalize both paths to handle `.`, `..`, symlinks, etc.
            let canonical_root = self
                .state
                .root_dir
                .canonicalize()
                .unwrap_or_else(|_| self.state.root_dir.clone());
            if let Ok(canonical_requested) = requested_path.canonicalize()
                && canonical_requested.starts_with(&canonical_root)
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
}

// --- ServerHandler implementation ---

impl ServerHandler for ContextPlusServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
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

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = std::result::Result<ListToolsResult, rmcp::ErrorData>>
    + Send
    + '_ {
        std::future::ready(Ok(ListToolsResult {
            tools: tool_definitions(),
            meta: None,
            next_cursor: None,
        }))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<CallToolResult, rmcp::ErrorData> {
        let name = request.name.to_string();
        let args = request.arguments.unwrap_or_default();
        Ok(self.dispatch(&name, args).await)
    }
}

// --- EmbedFn adapter for OllamaClient ---

struct OllamaEmbedder(OllamaClient);

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

// --- WalkAndIndexFn adapter (uses embedding cache for warm hits) ---

use crate::core::embeddings::content_hash;
use crate::tools::semantic_search::{SearchDocument, SymbolSearchEntry, WalkAndIndexFn};

struct CachedWalkerIndexer {
    config: Config,
    ollama: OllamaClient,
    state: Arc<SharedState>,
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

            // Read all files concurrently (up to 32 at a time)
            let mut join_set = tokio::task::JoinSet::new();
            for (i, entry) in entries.iter().enumerate() {
                let full_path = root.join(&entry.relative_path);
                let rel_path = entry.relative_path.clone();
                join_set.spawn(async move {
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
                let header =
                    crate::core::parser::extract_header(&content.lines().collect::<Vec<_>>());

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

                    // Persist to disk with model-qualified name
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

// --- Tool definitions ---

fn tool_definitions() -> Vec<Tool> {
    vec![
        make_tool(
            "get_context_tree",
            "Build a token-aware context tree showing file structure and symbols. Prunes detail levels based on max_tokens budget.",
            &[
                (
                    "target_path",
                    "string",
                    false,
                    "Specific directory or file to analyze (relative to project root)",
                ),
                (
                    "depth_limit",
                    "integer",
                    false,
                    "How many folder levels deep to scan. Use 1-2 for large projects.",
                ),
                (
                    "include_symbols",
                    "boolean",
                    false,
                    "Include function/class/enum names in the tree (default true)",
                ),
                (
                    "max_tokens",
                    "integer",
                    false,
                    "Maximum tokens for output. Auto-prunes if exceeded (default 20000)",
                ),
            ],
        ),
        make_tool(
            "get_file_skeleton",
            "Get function signatures, class definitions, and line ranges for a file without reading full content.",
            &[(
                "file_path",
                "string",
                true,
                "Path to the file to inspect (relative to project root)",
            )],
        ),
        make_tool(
            "get_blast_radius",
            "Find every file that imports or references a symbol. Maps the full impact of changing it.",
            &[
                (
                    "symbol_name",
                    "string",
                    true,
                    "The function, class, or variable name to trace across the codebase",
                ),
                (
                    "file_context",
                    "string",
                    false,
                    "The file where the symbol is defined. Excludes the definition line from results.",
                ),
            ],
        ),
        make_tool(
            "semantic_code_search",
            "Search code files semantically using natural language queries. Combines embedding similarity with keyword matching for hybrid ranking.",
            &[
                (
                    "query",
                    "string",
                    true,
                    "Natural language description of what you're looking for",
                ),
                (
                    "top_k",
                    "integer",
                    false,
                    "Number of matches to return (default 5, max 50)",
                ),
                (
                    "semantic_weight",
                    "number",
                    false,
                    "Weight for embedding similarity in hybrid ranking (default 0.72)",
                ),
                (
                    "keyword_weight",
                    "number",
                    false,
                    "Weight for keyword overlap in hybrid ranking (default 0.28)",
                ),
                (
                    "min_semantic_score",
                    "number",
                    false,
                    "Minimum semantic score filter (0-1 or 0-100)",
                ),
                (
                    "min_keyword_score",
                    "number",
                    false,
                    "Minimum keyword score filter (0-1 or 0-100)",
                ),
                (
                    "min_combined_score",
                    "number",
                    false,
                    "Minimum final score filter (0-1 or 0-100)",
                ),
                (
                    "require_keyword_match",
                    "boolean",
                    false,
                    "When true, only return files with keyword overlap",
                ),
                (
                    "require_semantic_match",
                    "boolean",
                    false,
                    "When true, only return files with positive semantic similarity",
                ),
            ],
        ),
        make_tool(
            "semantic_identifier_search",
            "Search for functions, classes, and variables by semantic meaning. Returns identifiers with call-site rankings.",
            &[
                (
                    "query",
                    "string",
                    true,
                    "Natural language intent to match identifiers and usages",
                ),
                (
                    "top_k",
                    "integer",
                    false,
                    "How many identifiers to return (default 5)",
                ),
                (
                    "top_calls_per_identifier",
                    "integer",
                    false,
                    "How many ranked call sites per identifier (default 10)",
                ),
                (
                    "include_kinds",
                    "array",
                    false,
                    "Optional kinds filter, e.g. [\"function\", \"method\", \"variable\"]",
                ),
                (
                    "semantic_weight",
                    "number",
                    false,
                    "Weight for semantic similarity score (default 0.78)",
                ),
                (
                    "keyword_weight",
                    "number",
                    false,
                    "Weight for keyword overlap score (default 0.22)",
                ),
            ],
        ),
        make_tool(
            "semantic_navigate",
            "Cluster files by semantic similarity using spectral clustering. Returns labeled groups for codebase navigation.",
            &[
                (
                    "max_depth",
                    "integer",
                    false,
                    "Maximum nesting depth of clusters (default 3)",
                ),
                (
                    "max_clusters",
                    "integer",
                    false,
                    "Maximum sub-clusters per level (default 20)",
                ),
            ],
        ),
        make_tool(
            "get_feature_hub",
            "Navigate Obsidian-style wikilinks to discover feature hubs and their connections.",
            &[
                (
                    "hub_path",
                    "string",
                    false,
                    "Path to a specific hub .md file (relative to root)",
                ),
                (
                    "feature_name",
                    "string",
                    false,
                    "Feature name to search for. Finds matching hub file automatically.",
                ),
                (
                    "show_orphans",
                    "boolean",
                    false,
                    "If true, lists all source files not linked to any hub.",
                ),
            ],
        ),
        make_tool(
            "run_static_analysis",
            "Run available linters (tsc, eslint, cargo check, ruff) on the project or a specific file.",
            &[(
                "target_path",
                "string",
                false,
                "Specific file or folder to lint (relative to root). Omit for full project.",
            )],
        ),
        make_tool(
            "propose_commit",
            "Write a file with validation (header, comments, nesting, line count) and create a shadow restore point for undo.",
            &[
                (
                    "file_path",
                    "string",
                    true,
                    "Where to save the file (relative to project root)",
                ),
                (
                    "new_content",
                    "string",
                    true,
                    "The complete file content to save",
                ),
                ("description", "string", false, "Description of the change"),
            ],
        ),
        make_tool(
            "list_restore_points",
            "List all shadow restore points created by propose_commit.",
            &[],
        ),
        make_tool(
            "undo_change",
            "Restore files from a shadow restore point created by propose_commit.",
            &[(
                "point_id",
                "string",
                true,
                "The restore point ID (format: rp-timestamp-hash). Get from list_restore_points.",
            )],
        ),
        make_tool(
            "upsert_memory_node",
            "Create or update a memory graph node. Nodes are uniquely identified by (label, type).",
            &[
                (
                    "type",
                    "string",
                    true,
                    "Node type: concept, file, symbol, note",
                ),
                (
                    "label",
                    "string",
                    true,
                    "Short identifier for the node. Used for deduplication with type.",
                ),
                (
                    "content",
                    "string",
                    true,
                    "Detailed content for the node. Used for embedding generation.",
                ),
                (
                    "metadata",
                    "object",
                    false,
                    "Optional key-value metadata pairs",
                ),
            ],
        ),
        make_tool(
            "create_relation",
            "Create or update a relation between two memory graph nodes.",
            &[
                ("source_id", "string", true, "ID of the source memory node"),
                ("target_id", "string", true, "ID of the target memory node"),
                (
                    "relation",
                    "string",
                    true,
                    "Relationship type: relates_to, depends_on, implements, references, similar_to, contains",
                ),
                (
                    "weight",
                    "number",
                    false,
                    "Edge weight 0-1. Higher = stronger relationship (default 1.0)",
                ),
                (
                    "metadata",
                    "object",
                    false,
                    "Optional key-value metadata for the edge",
                ),
            ],
        ),
        make_tool(
            "search_memory_graph",
            "Search the memory graph by semantic similarity and BFS traversal.",
            &[
                (
                    "query",
                    "string",
                    true,
                    "Natural language query to search the memory graph",
                ),
                (
                    "max_depth",
                    "integer",
                    false,
                    "How many hops to traverse from direct matches (default 1)",
                ),
                (
                    "top_k",
                    "integer",
                    false,
                    "Number of direct matches to return (default 5)",
                ),
                (
                    "edge_filter",
                    "array",
                    false,
                    "Only traverse edges of these types. Omit for all types.",
                ),
            ],
        ),
        make_tool(
            "prune_stale_links",
            "Remove memory graph edges with decayed weight below threshold, and orphan nodes.",
            &[(
                "threshold",
                "number",
                false,
                "Minimum decayed weight to keep an edge (default 0.15). Lower = keep more edges.",
            )],
        ),
        make_tool(
            "add_interlinked_context",
            "Add multiple memory nodes at once with optional auto-linking by semantic similarity.",
            &[
                (
                    "items",
                    "array",
                    true,
                    "Array of nodes to add. Each needs type, label, and content.",
                ),
                (
                    "auto_link",
                    "boolean",
                    false,
                    "Whether to auto-create similarity edges (default true)",
                ),
            ],
        ),
        make_tool(
            "retrieve_with_traversal",
            "Retrieve a memory node and its neighborhood via BFS traversal with depth penalty.",
            &[
                (
                    "start_node_id",
                    "string",
                    true,
                    "ID of the memory node to start traversal from",
                ),
                (
                    "max_depth",
                    "integer",
                    false,
                    "Maximum traversal depth from start node (default 2)",
                ),
                (
                    "edge_filter",
                    "array",
                    false,
                    "Only traverse edges of these types. Omit for all.",
                ),
            ],
        ),
    ]
}

fn make_tool(name: &str, description: &str, params: &[(&str, &str, bool, &str)]) -> Tool {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    for (pname, ptype, is_required, pdesc) in params {
        let mut prop = serde_json::Map::new();
        prop.insert("type".into(), Value::String(ptype.to_string()));
        prop.insert("description".into(), Value::String(pdesc.to_string()));
        properties.insert(pname.to_string(), Value::Object(prop));
        if *is_required {
            required.push(Value::String(pname.to_string()));
        }
    }

    let mut schema = serde_json::Map::new();
    schema.insert("type".into(), Value::String("object".into()));
    schema.insert("properties".into(), Value::Object(properties));
    if !required.is_empty() {
        schema.insert("required".into(), Value::Array(required));
    }

    Tool::new_with_raw(
        name.to_string(),
        Some(description.to_string().into()),
        Arc::new(schema),
    )
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
    fn tool_definitions_returns_all_17_tools() {
        let defs = tool_definitions();
        assert_eq!(defs.len(), 17, "expected 17 tools, got {}", defs.len());
        for tool in &defs {
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
            !cache.file_lines.is_empty(),
            "file_lines should not be empty"
        );

        // Verify specific files are in file_lines
        let has_hello = cache.file_lines.contains_key("hello.txt");
        let has_world = cache.file_lines.contains_key("world.rs");
        assert!(has_hello, "cache should contain hello.txt");
        assert!(has_world, "cache should contain world.rs");

        // Verify content
        let hello_lines = &cache.file_lines["hello.txt"];
        assert_eq!(hello_lines, &["line1", "line2", "line3"]);

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
            cache2.file_lines.contains_key("hello.txt"),
            "rebuilt cache should contain hello.txt"
        );
        assert!(
            cache2.file_lines.contains_key("world.rs"),
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
        args.insert("filePath".to_string(), json!("test.txt"));
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
        for tool in &defs {
            let schema = tool.input_schema.as_ref();
            assert_eq!(
                schema.get("type").and_then(|v| v.as_str()),
                Some("object"),
                "tool '{}' should have type: object in schema",
                tool.name
            );
        }
    }

    #[test]
    fn tool_definitions_all_tools_have_properties() {
        let defs = tool_definitions();
        for tool in &defs {
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
        // This mirrors the pattern used for includeKinds and edgeFilter
        let mut args = serde_json::Map::new();
        args.insert(
            "includeKinds".to_string(),
            json!(["function", "class", "method"]),
        );

        let result: Option<Vec<String>> =
            args.get("includeKinds")
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
            args.get("includeKinds")
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
}
