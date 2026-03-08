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

use crate::config::Config;
use crate::core::embeddings::OllamaClient;
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

/// Shared state accessible by all tool handlers.
pub struct SharedState {
    pub config: Config,
    pub root_dir: PathBuf,
    pub ollama: OllamaClient,
    pub memory_graph: GraphStore,
    pub project_cache: RwLock<Option<ProjectCache>>,
}

/// The MCP server exposing context+ tools.
#[derive(Clone)]
pub struct ContextPlusServer {
    pub state: Arc<SharedState>,
}

impl ContextPlusServer {
    pub fn new(root_dir: PathBuf, config: Config) -> Self {
        let ollama = OllamaClient::new(&config);
        let memory_graph = GraphStore::new();
        let state = Arc::new(SharedState {
            config,
            root_dir,
            ollama,
            memory_graph,
            project_cache: RwLock::new(None),
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
    async fn ensure_project_cache(&self) -> Result<Arc<ProjectCache>> {
        let ttl_secs = self.state.config.cache_ttl_secs;

        // Fast path: cache exists and is fresh
        {
            let guard = self.state.project_cache.read().await;
            if let Some(ref cache) = *guard
                && cache.last_refresh.elapsed().as_secs() < ttl_secs
            {
                let file_entries = cache.file_entries.clone();
                let file_lines = cache.file_lines.clone();
                let last_refresh = cache.last_refresh;
                return Ok(Arc::new(ProjectCache {
                    file_entries,
                    file_lines,
                    last_refresh,
                }));
            }
        }

        // Slow path: rebuild cache
        let root = self.state.root_dir.clone();
        let config = self.state.config.clone();

        let new_cache = tokio::task::spawn_blocking(move || {
            let entries = walk_with_config(&root, &config);
            let mut file_lines = HashMap::new();
            for entry in &entries {
                if entry.is_directory {
                    continue;
                }
                let full_path = root.join(&entry.relative_path);
                if let Ok(content) = std::fs::read_to_string(&full_path) {
                    let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
                    file_lines.insert(entry.relative_path.clone(), lines);
                }
            }
            ProjectCache {
                file_entries: entries,
                file_lines,
                last_refresh: Instant::now(),
            }
        })
        .await
        .map_err(|e| ContextPlusError::Other(format!("spawn_blocking failed: {e}")))?;

        let arc_cache = Arc::new(new_cache);

        // Store a clone in the shared state
        {
            let mut guard = self.state.project_cache.write().await;
            *guard = Some(ProjectCache {
                file_entries: arc_cache.file_entries.clone(),
                file_lines: arc_cache.file_lines.clone(),
                last_refresh: arc_cache.last_refresh,
            });
        }

        Ok(arc_cache)
    }

    /// Invalidate the project cache. Called by the file watcher when files change.
    pub async fn invalidate_project_cache(&self) {
        let mut guard = self.state.project_cache.write().await;
        *guard = None;
    }

    /// Helper: convert cached file_lines HashMap into the Vec<(String, Vec<String>)> format
    /// expected by blast_radius and semantic_identifier_search.
    fn file_lines_as_vec(cache: &ProjectCache) -> Vec<(String, Vec<String>)> {
        cache
            .file_lines
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    // --- Tool dispatch ---

    async fn dispatch(&self, name: &str, args: serde_json::Map<String, Value>) -> CallToolResult {
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

        let file_path = Self::get_str(&args, "filePath")
            .or_else(|| Self::get_str(&args, "targetPath"))
            .ok_or_else(|| ContextPlusError::Other("filePath is required".into()))?;

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
        let symbol_name = Self::get_str(&args, "symbolName")
            .ok_or_else(|| ContextPlusError::Other("symbolName is required".into()))?;
        let file_context = Self::get_str(&args, "fileContext");

        let cache = self.ensure_project_cache().await?;
        let file_lines = Self::file_lines_as_vec(&cache);

        let result = crate::tools::blast_radius::find_symbol_usages(
            &symbol_name,
            file_context.as_deref(),
            &file_lines,
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
            top_k: Self::get_usize(&args, "topK"),
            semantic_weight: Self::get_f64(&args, "semanticWeight"),
            keyword_weight: Self::get_f64(&args, "keywordWeight"),
            min_semantic_score: Self::get_f64(&args, "minSemanticScore"),
            min_keyword_score: Self::get_f64(&args, "minKeywordScore"),
            min_combined_score: Self::get_f64(&args, "minCombinedScore"),
            require_keyword_match: Self::get_bool(&args, "requireKeywordMatch"),
            require_semantic_match: Self::get_bool(&args, "requireSemanticMatch"),
        };

        let embedder = OllamaEmbedder(self.state.ollama.clone());
        let walker = WalkerIndexer {
            config: self.state.config.clone(),
            ollama: self.state.ollama.clone(),
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
        let file_lines = Self::file_lines_as_vec(&cache);

        // Build identifier docs by parsing each file (CPU-bound, use spawn_blocking)
        let identifier_docs = tokio::task::spawn_blocking(move || {
            let mut docs = Vec::new();
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
                        for sym in crate::core::parser::flatten_symbols(&symbols, None) {
                            let sig = sym.signature.clone().unwrap_or_default();
                            let text = format!(
                                "{} {} {} {}",
                                sym.name, sym.kind, entry.relative_path, sig
                            );
                            docs.push(IdentifierDoc {
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
            return Ok(Self::ok_text(
                "No supported identifiers found for semantic identifier search.".to_string(),
            ));
        }

        // Embed identifier docs
        let texts: Vec<String> = identifier_docs.iter().map(|d| d.text.clone()).collect();
        let vectors = self.state.ollama.embed(&texts).await?;
        let dims = vectors.first().map_or(0, |v| v.len());
        let flat_buffer: Vec<f32> = vectors.into_iter().flatten().collect();

        let options = SemanticIdentifierSearchOptions {
            root_dir: root.clone(),
            query,
            top_k: Self::get_usize(&args, "topK"),
            top_calls_per_identifier: Self::get_usize(&args, "topCallsPerIdentifier"),
            semantic_weight: Self::get_f64(&args, "semanticWeight"),
            keyword_weight: Self::get_f64(&args, "keywordWeight"),
            include_kinds: args
                .get("includeKinds")
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
            &identifier_docs,
            &flat_buffer,
            dims,
            &file_lines,
        )
        .await?;
        Ok(Self::ok_text(result))
    }

    async fn handle_semantic_navigate(
        &self,
        args: serde_json::Map<String, Value>,
    ) -> Result<CallToolResult> {
        let _query = Self::get_str(&args, "query")
            .ok_or_else(|| ContextPlusError::Other("query is required".into()))?;
        let root = self.resolve_root(&args);

        let options = crate::tools::semantic_navigate::SemanticNavigateOptions {
            root_dir: root.to_string_lossy().into(),
            max_depth: Self::get_usize(&args, "maxDepth"),
            max_clusters: Self::get_usize(&args, "maxClusters"),
        };

        let result = crate::tools::semantic_navigate::semantic_navigate(
            options,
            &self.state.ollama,
            &self.state.config,
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
        let file_path = Self::get_str(&args, "filePath")
            .ok_or_else(|| ContextPlusError::Other("filePath is required".into()))?;
        let content = Self::get_str(&args, "content")
            .ok_or_else(|| ContextPlusError::Other("content is required".into()))?;
        let description = Self::get_str(&args, "description");
        let root = self.resolve_root(&args);

        let result = crate::tools::propose_commit::propose_commit(
            &root,
            &file_path,
            &content,
            description.as_deref(),
        )
        .await?;
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
        let restore_point_id = Self::get_str(&args, "restorePointId")
            .ok_or_else(|| ContextPlusError::Other("restorePointId is required".into()))?;
        let root = self.resolve_root(&args);

        let restored = crate::git::shadow::restore_from_point(&root, &restore_point_id).await?;
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
            node_type: Self::get_str_or(&args, "nodeType", "concept"),
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
            source_label: Self::get_str(&args, "sourceLabel")
                .ok_or_else(|| ContextPlusError::Other("sourceLabel is required".into()))?,
            source_type: Self::get_str_or(&args, "sourceType", "concept"),
            target_label: Self::get_str(&args, "targetLabel")
                .ok_or_else(|| ContextPlusError::Other("targetLabel is required".into()))?,
            target_type: Self::get_str_or(&args, "targetType", "concept"),
            relation: Self::get_str_or(&args, "relation", "relates_to"),
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
            max_depth: Self::get_usize(&args, "maxDepth"),
            top_k: Self::get_usize(&args, "topK"),
            edge_filter: args
                .get("edgeFilter")
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
                            node_type: Self::get_str(obj, "nodeType")
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
            auto_link: Self::get_bool(&args, "autoLink"),
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
            node_id: Self::get_str(&args, "nodeId")
                .ok_or_else(|| ContextPlusError::Other("nodeId is required".into()))?,
            max_depth: Self::get_usize(&args, "maxDepth"),
            edge_filter: args
                .get("edgeFilter")
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

// --- WalkAndIndexFn adapter ---

use crate::tools::semantic_search::{SearchDocument, SymbolSearchEntry, WalkAndIndexFn};

struct WalkerIndexer {
    config: Config,
    ollama: OllamaClient,
}

impl WalkAndIndexFn for WalkerIndexer {
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
        Box::pin(async move {
            let entries = walk_with_config(&root, &config);
            let mut docs = Vec::new();

            for entry in &entries {
                let full_path = root.join(&entry.relative_path);
                let content = match tokio::fs::read_to_string(&full_path).await {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let ext = entry.relative_path.rsplit('.').next().unwrap_or("");
                let symbols = parse_with_tree_sitter(&content, ext).unwrap_or_default();
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

                docs.push(SearchDocument {
                    path: entry.relative_path.clone(),
                    header,
                    symbols: symbol_names,
                    symbol_entries,
                    content: format!(
                        "{} {}",
                        detect_language(&entry.relative_path).unwrap_or("unknown"),
                        content.chars().take(500).collect::<String>()
                    ),
                });
            }

            // Embed all document content via Ollama
            let texts: Vec<String> = docs.iter().map(|d| d.content.clone()).collect();
            let mut vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(docs.len());

            if texts.is_empty() {
                return Ok((docs, vectors));
            }

            // Embed in batches
            let batch_size = config.embed_batch_size.max(1);
            for chunk in texts.chunks(batch_size) {
                match ollama.embed(chunk).await {
                    Ok(embeddings) => {
                        for emb in embeddings {
                            vectors.push(Some(emb));
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Embedding batch failed: {e}, filling with None");
                        for _ in chunk {
                            vectors.push(None);
                        }
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
            "Build a token-aware context tree showing file structure and symbols. Prunes detail levels based on maxTokens budget.",
            &[
                ("rootDir", "string", false, "Root directory to analyze"),
                (
                    "includeSymbols",
                    "boolean",
                    false,
                    "Include symbols in output (default true)",
                ),
                ("maxTokens", "integer", false, "Token budget for output"),
            ],
        ),
        make_tool(
            "get_file_skeleton",
            "Get function signatures, class definitions, and line ranges for a file without reading full content.",
            &[
                ("filePath", "string", true, "Relative path to the file"),
                ("rootDir", "string", false, "Root directory"),
            ],
        ),
        make_tool(
            "get_blast_radius",
            "Find every file that imports or references a symbol. Maps the full impact of changing it.",
            &[
                ("symbolName", "string", true, "Symbol to search for"),
                (
                    "fileContext",
                    "string",
                    false,
                    "File where symbol is defined (to exclude definition)",
                ),
                ("rootDir", "string", false, "Root directory"),
            ],
        ),
        make_tool(
            "semantic_code_search",
            "Search code files semantically using natural language queries. Combines embedding similarity with keyword matching for hybrid ranking.",
            &[
                ("query", "string", true, "Natural language search query"),
                (
                    "topK",
                    "integer",
                    false,
                    "Number of results (default 5, max 50)",
                ),
                ("rootDir", "string", false, "Root directory"),
                (
                    "semanticWeight",
                    "number",
                    false,
                    "Weight for semantic score (default 0.72)",
                ),
                (
                    "keywordWeight",
                    "number",
                    false,
                    "Weight for keyword score (default 0.28)",
                ),
            ],
        ),
        make_tool(
            "semantic_identifier_search",
            "Search for functions, classes, and variables by semantic meaning. Returns identifiers with call-site rankings.",
            &[
                ("query", "string", true, "Natural language search query"),
                ("topK", "integer", false, "Number of results (default 5)"),
                ("rootDir", "string", false, "Root directory"),
                ("includeKinds", "array", false, "Filter by symbol kinds"),
            ],
        ),
        make_tool(
            "semantic_navigate",
            "Cluster files by semantic similarity using spectral clustering. Returns labeled groups for codebase navigation.",
            &[
                ("query", "string", true, "Navigation query"),
                ("rootDir", "string", false, "Root directory"),
                (
                    "maxClusters",
                    "integer",
                    false,
                    "Maximum number of clusters",
                ),
                ("maxFiles", "integer", false, "Maximum files to analyze"),
            ],
        ),
        make_tool(
            "get_feature_hub",
            "Navigate Obsidian-style wikilinks to discover feature hubs and their connections.",
            &[
                ("rootDir", "string", false, "Root directory"),
                ("hubPath", "string", false, "Specific hub file to analyze"),
            ],
        ),
        make_tool(
            "run_static_analysis",
            "Run available linters (tsc, eslint, cargo check, ruff) on the project or a specific file.",
            &[
                ("rootDir", "string", false, "Root directory"),
                ("targetPath", "string", false, "Specific file to analyze"),
            ],
        ),
        make_tool(
            "propose_commit",
            "Write a file with validation (header, comments, nesting, line count) and create a shadow restore point for undo.",
            &[
                ("filePath", "string", true, "Relative path for the file"),
                ("content", "string", true, "File content to write"),
                ("description", "string", false, "Description of the change"),
                ("rootDir", "string", false, "Root directory"),
            ],
        ),
        make_tool(
            "list_restore_points",
            "List all shadow restore points created by propose_commit.",
            &[("rootDir", "string", false, "Root directory")],
        ),
        make_tool(
            "undo_change",
            "Restore files from a shadow restore point created by propose_commit.",
            &[
                (
                    "restorePointId",
                    "string",
                    true,
                    "Restore point ID (format: rp-{timestamp}-{random})",
                ),
                ("rootDir", "string", false, "Root directory"),
            ],
        ),
        make_tool(
            "upsert_memory_node",
            "Create or update a memory graph node. Nodes are uniquely identified by (label, type).",
            &[
                ("label", "string", true, "Node label"),
                ("content", "string", true, "Node content"),
                (
                    "nodeType",
                    "string",
                    false,
                    "Node type: concept, file, symbol, note (default: concept)",
                ),
            ],
        ),
        make_tool(
            "create_relation",
            "Create or update a relation between two memory graph nodes.",
            &[
                ("sourceLabel", "string", true, "Source node label"),
                (
                    "sourceType",
                    "string",
                    false,
                    "Source node type (default: concept)",
                ),
                ("targetLabel", "string", true, "Target node label"),
                (
                    "targetType",
                    "string",
                    false,
                    "Target node type (default: concept)",
                ),
                (
                    "relation",
                    "string",
                    false,
                    "Relation type: relates_to, depends_on, implements, references, similar_to, contains",
                ),
                ("weight", "number", false, "Relation weight (0.0-1.0)"),
            ],
        ),
        make_tool(
            "search_memory_graph",
            "Search the memory graph by semantic similarity and BFS traversal.",
            &[
                ("query", "string", true, "Search query"),
                ("topK", "integer", false, "Number of results"),
                ("maxDepth", "integer", false, "Maximum traversal depth"),
            ],
        ),
        make_tool(
            "prune_stale_links",
            "Remove memory graph edges with decayed weight below threshold, and orphan nodes.",
            &[(
                "threshold",
                "number",
                false,
                "Weight threshold (default 0.1)",
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
                    "Array of {nodeType, label, content} objects",
                ),
                (
                    "autoLink",
                    "boolean",
                    false,
                    "Auto-link similar nodes (cosine > 0.72)",
                ),
            ],
        ),
        make_tool(
            "retrieve_with_traversal",
            "Retrieve a memory node and its neighborhood via BFS traversal with depth penalty.",
            &[
                ("nodeId", "string", true, "Node ID to start traversal from"),
                ("maxDepth", "integer", false, "Maximum traversal depth"),
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

    #[test]
    fn file_lines_as_vec_converts_hashmap_correctly() {
        let mut file_lines = HashMap::new();
        file_lines.insert(
            "a.txt".to_string(),
            vec!["alpha".to_string(), "beta".to_string()],
        );
        file_lines.insert("b.txt".to_string(), vec!["gamma".to_string()]);

        let cache = ProjectCache {
            file_entries: vec![],
            file_lines,
            last_refresh: Instant::now(),
        };

        let vec_result = ContextPlusServer::file_lines_as_vec(&cache);
        assert_eq!(vec_result.len(), 2, "should have 2 entries");

        // Collect into a map for order-independent comparison
        let result_map: HashMap<String, Vec<String>> = vec_result.into_iter().collect();
        assert_eq!(
            result_map["a.txt"],
            vec!["alpha".to_string(), "beta".to_string()]
        );
        assert_eq!(result_map["b.txt"], vec!["gamma".to_string()]);
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
}
