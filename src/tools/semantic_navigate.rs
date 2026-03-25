// Semantic project navigator using spectral clustering and Ollama labeling.
// Browse codebase by meaning: embeds files, clusters vectors, generates labels.

use crate::cache::rkyv_store;
use crate::config::Config;
use crate::core::clustering::{find_path_pattern, spectral_cluster_with_min};
use crate::core::embeddings::VectorStore;
use crate::core::embeddings::{CacheEntry, OllamaClient};
use crate::core::walker;
use crate::error::Result;
use futures::stream::{self, StreamExt};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tokio::io::AsyncReadExt;
use tokio::sync::RwLock;

use super::navigate_constants::*;

/// Options for the semantic navigation tool.
#[derive(Debug, Clone)]
pub struct SemanticNavigateOptions {
    pub root_dir: String,
    pub max_depth: Option<usize>,
    pub max_clusters: Option<usize>,
}

/// Information about a source file for clustering.
#[derive(Debug, Clone, Default)]
struct FileInfo {
    relative_path: String,
    header: String,
    content: String,
    symbol_preview: Vec<String>,
}

/// A hierarchical cluster node.
#[derive(Debug, Clone)]
struct ClusterNode {
    label: String,
    files: Vec<FileInfo>,
    children: Vec<ClusterNode>,
}

/// Perform semantic navigation: embed files, cluster, label, return tree.
///
/// Uses `embedding_cache` to avoid re-embedding files that haven't changed.
/// On warm runs (cache populated + content unchanged), this skips all Ollama embed calls.
/// New embeddings are persisted to disk via rkyv_store for cross-restart survival.
pub async fn semantic_navigate(
    options: SemanticNavigateOptions,
    ollama: &OllamaClient,
    config: &Config,
    embedding_cache: &RwLock<HashMap<String, CacheEntry>>,
    root_dir: &Path,
) -> Result<String> {
    let max_clusters = options.max_clusters.unwrap_or(20);
    let max_depth = options.max_depth.unwrap_or(3);
    let root = PathBuf::from(&options.root_dir);

    // Walk directory for source files using shared walker infrastructure
    let mut files = collect_source_files_via_walker(&root, config).await?;
    if files.is_empty() {
        return Ok("No supported source files found in the project.".to_string());
    }

    // Cap file count to keep spectral clustering tractable.
    // Sample evenly across the sorted file list to preserve directory diversity.
    let sampled = files.len() > MAX_NAVIGATE_FILES;
    if sampled {
        let total = files.len();
        let step = total as f64 / MAX_NAVIGATE_FILES as f64;
        let sampled_files: Vec<FileInfo> = (0..MAX_NAVIGATE_FILES)
            .map(|i| {
                let idx = (i as f64 * step).floor() as usize;
                std::mem::take(&mut files[idx.min(total - 1)])
            })
            .collect();
        files = sampled_files;
    }

    // Navigate uses its OWN embedding cache (not the shared one from semantic_code_search).
    // Navigate embeds with path-weighted text ("{path} {path} {path} {header} {content}")
    // which produces different vectors than search ("{lang} {content}").
    let nav_cache_name = nav_cache_name(&config.ollama_embed_model);
    let mut nav_cache: HashMap<String, CacheEntry> = HashMap::new();
    if let Ok(Some(store)) = rkyv_store::load_vector_store(root_dir, &nav_cache_name) {
        let dims = store.dims() as usize;
        let flat = store.vectors_data();
        let keys = store.keys();
        let hashes = store.hashes();
        for (i, key) in keys.iter().enumerate() {
            nav_cache.insert(key.clone(), CacheEntry {
                hash: hashes[i].clone(),
                vector: flat[i * dims..(i + 1) * dims].to_vec(),
            });
        }
    }
    let nav_cache_lock = RwLock::new(nav_cache);

    let vectors = match resolve_embeddings(
        &files,
        ollama,
        &nav_cache_lock,
        root_dir,
        &nav_cache_name,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            return Ok(format!(
                "Ollama not available for embeddings: {}\nMake sure Ollama is running with an embedding model.",
                e
            ));
        }
    };

    if files.len() <= MAX_FILES_PER_LEAF {
        // Small project: just list files with labels
        let file_labels = label_files(&files).await;
        let mut lines = vec![format!("Semantic Navigator: {} files\n", files.len())];
        for (i, file) in files.iter().enumerate() {
            let symbols = if file.symbol_preview.is_empty() {
                String::new()
            } else {
                format!(" | symbols: {}", file.symbol_preview.join(", "))
            };
            let label = file_labels
                .get(i)
                .cloned()
                .unwrap_or_else(|| file.header.clone());
            lines.push(format!("  {} - {}{}", file.relative_path, label, symbols));
        }
        return Ok(lines.join("\n"));
    }

    // Depth 0: group by directory structure (domain boundaries).
    // Spectral clustering can't separate domains in uniform-architecture monorepos
    // because embeddings capture code patterns, not domain identity.
    // Directory structure IS the domain boundary — use it as the top level,
    // then spectral clustering sub-clusters within each group at depth 1+.
    let dir_groups = group_by_directory(&files);

    let mut children: Vec<ClusterNode> = Vec::new();
    let mut large_groups: Vec<(String, Vec<usize>)> = Vec::new();

    for (dir_label, group_indices) in &dir_groups {
        if group_indices.len() <= MAX_FILES_PER_LEAF {
            children.push(ClusterNode {
                label: dir_label.clone(),
                files: group_indices.iter().map(|&i| files[i].clone()).collect(),
                children: Vec::new(),
            });
        } else {
            large_groups.push((dir_label.clone(), group_indices.clone()));
        }
    }

    // Spectral-cluster large groups in parallel (depth 1+)
    let hierarchy_futures: Vec<_> = large_groups
        .iter()
        .map(|(_, indices)| {
            Box::pin(build_hierarchy(
                &files,
                &vectors,
                indices,
                max_clusters,
                1,
                max_depth,
                ollama,
            ))
        })
        .collect();

    let hierarchy_results = futures::future::join_all(hierarchy_futures).await;
    for ((label, _), mut node) in large_groups.into_iter().zip(hierarchy_results) {
        node.label = label;
        children.push(node);
    }

    // Sort by file count descending
    children.sort_by(|a, b| count_files_in_node(b).cmp(&count_files_in_node(a)));

    let root_node = ClusterNode {
        label: "Project".to_string(),
        files: Vec::new(),
        children,
    };

    let tree_text = render_cluster_tree(&root_node, 0);
    let sampled_note = if sampled {
        format!(" (sampled {} of total)", MAX_NAVIGATE_FILES)
    } else {
        String::new()
    };
    Ok(format!(
        "Semantic Navigator: {} files{} organized by meaning\n\n{}",
        files.len(),
        sampled_note,
        tree_text
    ))
}

/// Walk the directory using shared walker infrastructure and collect source file information.
async fn collect_source_files_via_walker(root: &Path, config: &Config) -> Result<Vec<FileInfo>> {
    let allowed_extensions: HashSet<&str> = NAVIGATE_EXTENSIONS.iter().copied().collect();

    // walk_with_config is synchronous (uses the `ignore` crate), run on blocking thread
    let root_clone = root.to_path_buf();
    let config_clone = config.clone();
    let entries =
        tokio::task::spawn_blocking(move || walker::walk_with_config(&root_clone, &config_clone))
            .await
            .unwrap_or_default();

    // Filter to supported extensions and non-directories
    let max_file_size = config.max_embed_file_size as u64;
    let filtered: Vec<_> = entries
        .into_iter()
        .filter(|entry| {
            if entry.is_directory {
                return false;
            }
            let ext = entry
                .path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            allowed_extensions.contains(ext)
        })
        .collect();

    // Read files concurrently with buffer_unordered(64)
    let file_futures = filtered.into_iter().map(|entry| async move {
        let meta = tokio::fs::metadata(&entry.path).await.ok()?;
        if meta.len() > max_file_size {
            return None;
        }

        let content = if meta.len() > 2048 {
            // Large file: read only first 2KB to avoid wasting memory
            let mut f = tokio::fs::File::open(&entry.path).await.ok()?;
            let mut buf = vec![0u8; 2048];
            let n = f.read(&mut buf).await.unwrap_or(0);
            buf.truncate(n);
            String::from_utf8_lossy(&buf).to_string()
        } else {
            // Small file: read entirely
            tokio::fs::read_to_string(&entry.path).await.ok()?
        };

        let header = extract_header(&content);
        let truncated_content = if content.len() > MAX_CONTENT_CHARS {
            crate::core::parser::truncate_to_char_boundary(&content, MAX_CONTENT_CHARS).to_string()
        } else {
            content
        };

        Some(FileInfo {
            relative_path: entry.relative_path,
            header,
            content: truncated_content,
            symbol_preview: Vec::new(),
        })
    });

    let files: Vec<FileInfo> = stream::iter(file_futures)
        .buffer_unordered(64)
        .filter_map(|opt| async { opt })
        .collect()
        .await;

    Ok(files)
}

/// Resolve embedding vectors for files, using cache where possible.
///
/// Returns one Vec<f32> per file (same order as `files`).
/// Reads from `embedding_cache` first; checks content hash to detect stale entries.
/// Only calls Ollama for files not in cache or whose content has changed.
/// Stores newly embedded vectors back into the cache and persists to disk.
async fn resolve_embeddings(
    files: &[FileInfo],
    ollama: &OllamaClient,
    embedding_cache: &RwLock<HashMap<String, CacheEntry>>,
    root_dir: &Path,
    embed_model: &str,
) -> std::result::Result<Vec<Vec<f32>>, crate::error::ContextPlusError> {
    let cache_read = embedding_cache.read().await;

    // Partition: which files have valid cached vectors, which need (re-)embedding
    let mut result_vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(files.len());
    let mut uncached_indices: Vec<usize> = Vec::new();
    let mut uncached_texts: Vec<String> = Vec::new();
    let mut uncached_hashes: Vec<String> = Vec::new();

    for (i, file) in files.iter().enumerate() {
        // Hash includes "nav3:" prefix to:
        // 1. Separate from semantic_code_search vectors (different embed format)
        // 2. Invalidate old navigate vectors (pre-path-weighting)
        // Bump the version number when the embed text format changes.
        let file_hash = nav_content_hash(&file.relative_path, &file.content);
        if let Some(entry) = cache_read.get(&file.relative_path) {
            if entry.hash == file_hash {
                // Cache hit: content unchanged
                result_vectors.push(Some(entry.vector.clone()));
            } else {
                // Cache stale: content changed, need re-embed
                result_vectors.push(None);
                uncached_indices.push(i);
                // Path-weighted embed text: repeat path 3x to boost domain signal.
                // Code embeddings capture structural patterns (imports, patterns),
                // path captures domain identity (billing vs scheduling vs IAM).
                uncached_texts.push(nav_embed_text(&file.relative_path, &file.header, &file.content));
                uncached_hashes.push(file_hash);
            }
        } else {
            // Cache miss: never seen this file
            result_vectors.push(None);
            uncached_indices.push(i);
            uncached_texts.push(nav_embed_text(&file.relative_path, &file.header, &file.content));
            uncached_hashes.push(file_hash);
        }
    }
    drop(cache_read);

    // If everything was cached and fresh, skip Ollama entirely
    if uncached_indices.is_empty() {
        return Ok(result_vectors.into_iter().map(|v| v.unwrap()).collect());
    }

    // Embed uncached/stale files in chunks, saving progress after each batch.
    // This prevents losing all work if the MCP connection times out mid-run.
    let embed_cache_name = crate::server::cache_name("embeddings", embed_model);
    let chunk_size = ollama.batch_size();

    for chunk_start in (0..uncached_indices.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(uncached_indices.len());
        let chunk_texts = &uncached_texts[chunk_start..chunk_end];

        let mut chunk_vectors = ollama.embed(chunk_texts).await?;

        // Store this chunk's vectors in cache, then drop lock BEFORE disk I/O.
        let store_to_save = {
            let mut cache_write = embedding_cache.write().await;
            for (local_j, &file_idx) in uncached_indices[chunk_start..chunk_end].iter().enumerate() {
                if local_j < chunk_vectors.len() {
                    let vec = std::mem::take(&mut chunk_vectors[local_j]);
                    cache_write.insert(
                        files[file_idx].relative_path.clone(),
                        CacheEntry {
                            hash: uncached_hashes[chunk_start + local_j].clone(),
                            vector: vec.clone(),
                        },
                    );
                    result_vectors[file_idx] = Some(vec);
                }
            }
            VectorStore::from_cache(&cache_write)
        };

        // Persist after each chunk so progress survives a timeout on the next batch.
        if let Some(store) = store_to_save
            && let Err(e) = rkyv_store::save_vector_store(root_dir, &embed_cache_name, &store)
        {
            tracing::warn!("Failed to save embedding cache to disk: {e}");
        }
    }

    Ok(result_vectors
        .into_iter()
        .map(|v| v.unwrap_or_default())
        .collect())
}

/// Extract a header comment from the first few lines of a file.
pub fn extract_header(content: &str) -> String {
    let mut header_lines = Vec::new();
    for line in content.lines().take(MAX_HEADER_LINES) {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with("--") {
            let cleaned = trimmed
                .trim_start_matches("//")
                .trim_start_matches('#')
                .trim_start_matches("--")
                .trim();
            header_lines.push(cleaned.to_string());
        } else if !trimmed.is_empty() {
            break;
        }
    }
    let joined = header_lines.join(" ");
    if joined.len() > MAX_HEADER_LEN {
        crate::core::parser::truncate_to_char_boundary(&joined, MAX_HEADER_LEN).to_string()
    } else {
        joined
    }
}

/// Label files using Ollama chat for small sets.
async fn label_files(files: &[FileInfo]) -> Vec<String> {
    // For small file sets (≤ MAX_FILES_PER_LEAF), headers are descriptive enough.
    // Skip LLM call entirely — saves ~6s per invocation.
    files.iter().map(|f| {
        if f.header.is_empty() {
            f.relative_path.split('/').next_back().unwrap_or(&f.relative_path).to_string()
        } else {
            f.header.clone()
        }
    }).collect()
}

/// Label sibling clusters using path patterns first, Ollama chat only for unlabeled clusters.
///
/// Optimization: if every cluster has a path_pattern, skip LLM entirely (~6s saved per call).
/// For mixed cases, only send unlabeled clusters to the LLM in a single batched call.
async fn label_sibling_clusters(
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    ollama: &OllamaClient,
) -> Vec<String> {
    if clusters.is_empty() {
        return Vec::new();
    }

    // Build initial labels from path patterns or file names
    let mut labels: Vec<Option<String>> = clusters
        .iter()
        .map(|(files, pattern)| {
            if let Some(pp) = pattern {
                Some(pp.clone())
            } else if files.len() <= 3 {
                // Small cluster: just use file names
                let names: Vec<&str> = files
                    .iter()
                    .filter_map(|f| f.relative_path.split('/').next_back())
                    .collect();
                let joined = names.join(", ");
                Some(if joined.len() > 40 {
                    crate::core::parser::truncate_to_char_boundary(&joined, 40).to_string()
                } else {
                    joined
                })
            } else {
                None
            }
        })
        .collect();

    // If all clusters are already labeled, skip LLM entirely
    if labels.iter().all(|l| l.is_some()) {
        return labels.into_iter().map(|l| l.unwrap()).collect();
    }

    // Collect indices of unlabeled clusters for batched LLM call
    let unlabeled: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(_, l)| l.is_none())
        .map(|(i, _)| i)
        .collect();

    const MAX_FILES_PER_LABEL: usize = 10;

    let descriptions: Vec<String> = unlabeled
        .iter()
        .enumerate()
        .map(|(desc_idx, &cluster_idx)| {
            let (files, _) = &clusters[cluster_idx];
            let sample_files = if files.len() > MAX_FILES_PER_LABEL {
                &files[..MAX_FILES_PER_LABEL]
            } else {
                files
            };
            let file_list = sample_files
                .iter()
                .map(|f| {
                    let desc = if f.header.is_empty() {
                        "no description"
                    } else {
                        &f.header
                    };
                    format!("{}: {}", f.relative_path, desc)
                })
                .collect::<Vec<_>>()
                .join("\n  ");
            format!(
                "Cluster {} ({} files):\n  {}",
                desc_idx + 1,
                files.len(),
                file_list
            )
        })
        .collect();

    let prompt = format!(
        "Label each cluster with EXACTLY 2 words. Return ONLY a JSON array of strings, one per cluster.\n\n{}\n\nJSON array of {} strings:",
        descriptions.join("\n\n"),
        unlabeled.len()
    );

    if let Ok(response) = ollama.chat(&prompt).await {
        if let Some(json_str) = extract_json_array(&response) {
            if let Ok(llm_labels) = serde_json::from_str::<Vec<String>>(&json_str) {
                for (j, &cluster_idx) in unlabeled.iter().enumerate() {
                    if let Some(label) = llm_labels.get(j) {
                        labels[cluster_idx] = Some(label.clone());
                    }
                }
            }
        }
    }

    // Fill any remaining unlabeled with a smart fallback:
    // Find the deepest directory segment that distinguishes this cluster.
    // In monorepos, parts[0] is always "packages" — useless. We want
    // "domains/billing" or "platform/auth" or "frontend/hooks".
    labels
        .into_iter()
        .enumerate()
        .map(|(i, l)| {
            l.unwrap_or_else(|| {
                let (files, _) = &clusters[i];
                derive_cluster_label(files)
                    .unwrap_or_else(|| format!("Cluster {}", i + 1))
            })
        })
        .collect()
}

/// Derive a human-readable label for a cluster from its file paths.
///
/// Finds the deepest common path prefix, then picks the last 1-2 meaningful
/// segments. In a monorepo where everything is under `packages/`, this yields
/// labels like "domains/billing" or "platform/auth" instead of "packages".
fn derive_cluster_label(files: &[&FileInfo]) -> Option<String> {
    if files.is_empty() {
        return None;
    }

    let paths: Vec<Vec<&str>> = files
        .iter()
        .map(|f| f.relative_path.split('/').collect::<Vec<_>>())
        .collect();

    let min_depth = paths.iter().map(|p| p.len()).min().unwrap_or(0);
    if min_depth < 2 {
        return None;
    }

    // Find how deep the common prefix goes (excluding the filename)
    let mut common_depth = 0;
    for d in 0..min_depth.saturating_sub(1) {
        if paths.iter().all(|p| p[d] == paths[0][d]) {
            common_depth = d + 1;
        } else {
            break;
        }
    }

    if common_depth == 0 {
        // No common prefix — count the most frequent segment at depth 1
        // (skip depth 0 which is often "packages" or "apps" in monorepos)
        let mut seg_counts: HashMap<&str, usize> = HashMap::new();
        for p in &paths {
            let depth = if p.len() > 2 { 1 } else { 0 };
            *seg_counts.entry(p[depth]).or_default() += 1;
        }
        return seg_counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .filter(|(_, c)| *c > files.len() / 3)
            .map(|(seg, _)| seg.to_string());
    }

    // Use last 1-2 segments of the common prefix for the label.
    // E.g., common prefix ["packages", "domains", "billing"] → "domains/billing"
    let start = if common_depth >= 2 { common_depth - 2 } else { 0 };
    let label_parts: Vec<&str> = (start..common_depth).map(|d| paths[0][d]).collect();
    let label = label_parts.join("/");

    // Skip labels that are just generic top-level dirs
    if label == "packages" || label == "apps" || label == "src" || label == "lib" {
        // Try one level deeper: find the most common next segment
        if common_depth < min_depth.saturating_sub(1) {
            let mut next_counts: HashMap<&str, usize> = HashMap::new();
            for p in &paths {
                *next_counts.entry(p[common_depth]).or_default() += 1;
            }
            return next_counts
                .into_iter()
                .max_by_key(|(_, c)| *c)
                .filter(|(_, c)| *c > files.len() / 3)
                .map(|(seg, _)| format!("{}/{}", label, seg));
        }
        return None;
    }

    Some(label)
}

/// Group files by meaningful directory structure for top-level clustering.
///
/// In a monorepo like `packages/domains/{billing,scheduling,...}`, groups by the
/// deepest "interesting" directory level. Generic prefixes like `packages/` are
/// skipped to find the actual domain boundary.
///
/// Returns `(label, indices)` pairs sorted by directory name.
fn group_by_directory(files: &[FileInfo]) -> Vec<(String, Vec<usize>)> {
    let generic_dirs: HashSet<&str> = ["packages", "src", "lib", "apps", "internal", "cmd"]
        .iter()
        .copied()
        .collect();

    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, file) in files.iter().enumerate() {
        let parts: Vec<&str> = file.relative_path.split('/').collect();

        // Find the first "interesting" directory (skip generic prefixes)
        // For "packages/domains/billing/service/index.ts" → "domains/billing"
        // For "apps/emr-api/src/app.ts" → "emr-api"
        // For "scripts/migration/run.ts" → "scripts"
        let label = if parts.len() >= 3 {
            // Try to find 2-level label skipping generics
            let mut start = 0;
            while start < parts.len().saturating_sub(2) && generic_dirs.contains(parts[start]) {
                start += 1;
            }
            if start + 1 < parts.len().saturating_sub(1) {
                format!("{}/{}", parts[start], parts[start + 1])
            } else if start < parts.len().saturating_sub(1) {
                parts[start].to_string()
            } else {
                parts[0].to_string()
            }
        } else if parts.len() == 2 {
            parts[0].to_string()
        } else {
            "root".to_string()
        };

        groups.entry(label).or_default().push(i);
    }

    // Merge tiny groups (< 15 files) into an "other" bucket
    let mut result: Vec<(String, Vec<usize>)> = Vec::new();
    let mut other: Vec<usize> = Vec::new();

    for (label, indices) in groups {
        if indices.len() < MIN_DIR_GROUP_SIZE {
            other.extend(indices);
        } else {
            result.push((label, indices));
        }
    }

    if !other.is_empty() {
        result.push(("other".to_string(), other));
    }

    result.sort_by(|a, b| a.0.cmp(&b.0));
    result
}

/// Count total files in a cluster node (including all children recursively).
fn count_files_in_node(node: &ClusterNode) -> usize {
    if node.children.is_empty() {
        node.files.len()
    } else {
        node.children.iter().map(count_files_in_node).sum()
    }
}

/// Extract a JSON array string from LLM response text.
fn extract_json_array(text: &str) -> Option<String> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end >= start {
        Some(text[start..=end].to_string())
    } else {
        None
    }
}

/// Recursively build hierarchical cluster tree.
/// Uses index slices into the top-level `all_files`/`all_vectors` to avoid cloning.
async fn build_hierarchy(
    all_files: &[FileInfo],
    all_vectors: &[Vec<f32>],
    indices: &[usize],
    max_clusters: usize,
    depth: usize,
    max_depth: usize,
    ollama: &OllamaClient,
) -> ClusterNode {
    if indices.len() <= MAX_FILES_PER_LEAF || depth >= max_depth {
        return ClusterNode {
            label: String::new(),
            files: indices.iter().map(|&i| all_files[i].clone()).collect(),
            children: Vec::new(),
        };
    }

    // Build local vectors slice for clustering — moved into spawn_blocking
    // to avoid blocking the tokio executor during O(n^3) eigendecomposition.
    let local_vectors: Vec<Vec<f32>> = indices.iter().map(|&i| all_vectors[i].clone()).collect();
    let mc = max_clusters;
    let cluster_results = tokio::task::spawn_blocking(move || {
        spectral_cluster_with_min(&local_vectors, mc, 2)
    })
    .await
    .unwrap_or_else(|_| vec![]);

    if cluster_results.len() <= 1 {
        return ClusterNode {
            label: String::new(),
            files: indices.iter().map(|&i| all_files[i].clone()).collect(),
            children: Vec::new(),
        };
    }

    // Find the parent's common prefix to suppress redundant path_patterns.
    // Within a directory group, all files share e.g. "packages/domains/billing/",
    // so find_path_pattern always returns "packages/domains/billing/*" — useless.
    // Only keep path_pattern if it's MORE specific than the parent prefix.
    let parent_prefix = {
        let parts: Vec<Vec<&str>> = indices.iter().map(|&i| all_files[i].relative_path.split('/').collect::<Vec<_>>()).collect();
        let min_depth = parts.iter().map(|p| p.len()).min().unwrap_or(0);
        let mut prefix_depth = 0;
        for d in 0..min_depth.saturating_sub(1) {
            if parts.iter().all(|p| p[d] == parts[0][d]) {
                prefix_depth = d + 1;
            } else {
                break;
            }
        }
        prefix_depth
    };

    // Map cluster result indices back to global indices
    let child_index_groups: Vec<(Vec<usize>, Option<String>)> = cluster_results
        .iter()
        .map(|cluster| {
            let global_indices: Vec<usize> =
                cluster.indices.iter().map(|&li| indices[li]).collect();
            let child_paths: Vec<String> = global_indices
                .iter()
                .map(|&gi| all_files[gi].relative_path.clone())
                .collect();
            let pattern = find_path_pattern(&child_paths);
            // Suppress pattern if it's just the parent prefix + "*"
            // (e.g., "packages/domains/billing/*" when parent is "packages/domains/billing/")
            let useful_pattern = pattern.filter(|p| {
                let p_depth = p.split('/').count();
                p_depth > parent_prefix + 1
            });
            (global_indices, useful_pattern)
        })
        .collect();

    // Get labels for sibling clusters — pass &FileInfo slices without cloning vecs
    let label_input: Vec<(Vec<&FileInfo>, Option<String>)> = child_index_groups
        .iter()
        .map(|(idxs, pattern)| {
            let file_refs: Vec<&FileInfo> = idxs.iter().map(|&i| &all_files[i]).collect();
            (file_refs, pattern.clone())
        })
        .collect();

    // Run labeling and child recursion concurrently.
    // Labeling sends one batched LLM call while children do clustering (CPU-bound).
    // Safe because Ollama serializes requests and each call is small (one group's siblings).
    let label_future = label_sibling_clusters(&label_input, ollama);

    let child_futures: Vec<_> = child_index_groups
        .iter()
        .map(|(child_indices, _pattern)| {
            Box::pin(build_hierarchy(
                all_files,
                all_vectors,
                child_indices,
                max_clusters,
                depth + 1,
                max_depth,
                ollama,
            ))
        })
        .collect();

    let (labels, child_results) = futures::future::join(
        label_future,
        futures::future::join_all(child_futures),
    )
    .await;

    let children: Vec<ClusterNode> = child_results
        .into_iter()
        .enumerate()
        .map(|(i, mut child)| {
            child.label = labels
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("Cluster {}", i + 1));
            child
        })
        .collect();

    ClusterNode {
        label: String::new(),
        files: Vec::new(),
        children,
    }
}

/// Render a cluster tree as indented text.
fn render_cluster_tree(node: &ClusterNode, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    let mut result = String::new();

    if !node.label.is_empty() {
        let count = count_files_in_node(node);
        result.push_str(&format!("{}[{}] ({} files)\n", pad, node.label, count));
    }

    if !node.children.is_empty() {
        for child in &node.children {
            result.push_str(&render_cluster_tree(child, indent + 1));
        }
    } else {
        let show_count = node.files.len().min(MAX_FILES_PER_LEAF_DISPLAY);
        for file in node.files.iter().take(show_count) {
            let label = if file.header.is_empty() {
                String::new()
            } else {
                format!(" - {}", file.header)
            };
            let symbols = if file.symbol_preview.is_empty() {
                String::new()
            } else {
                format!(" | symbols: {}", file.symbol_preview.join(", "))
            };
            result.push_str(&format!(
                "{}  {}{}{}\n",
                pad, file.relative_path, label, symbols
            ));
        }
        if node.files.len() > MAX_FILES_PER_LEAF_DISPLAY {
            result.push_str(&format!(
                "{}  (+{} more files)\n",
                pad,
                node.files.len() - MAX_FILES_PER_LEAF_DISPLAY
            ));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::embeddings::content_hash;

    #[test]
    fn extract_header_comment_slashes() {
        let content = "// This is the main module\n// It handles routing\nfn main() {}";
        let header = extract_header(content);
        assert_eq!(header, "This is the main module It handles routing");
    }

    #[test]
    fn extract_header_hash_comments() {
        let content = "# Config loader\nimport os";
        let header = extract_header(content);
        assert_eq!(header, "Config loader");
    }

    #[test]
    fn extract_header_no_comments() {
        let content = "fn main() {\n    println!(\"hello\");\n}";
        let header = extract_header(content);
        assert_eq!(header, "");
    }

    #[test]
    fn extract_header_truncates() {
        let long_comment = format!("// {}", "x".repeat(300));
        let header = extract_header(&long_comment);
        assert_eq!(header.len(), 200);
    }

    #[test]
    fn extract_json_array_basic() {
        let text = r#"Here is the result: ["a", "b", "c"] done."#;
        let arr = extract_json_array(text);
        assert_eq!(arr, Some(r#"["a", "b", "c"]"#.to_string()));
    }

    #[test]
    fn extract_json_array_none() {
        assert_eq!(extract_json_array("no json here"), None);
    }

    #[test]
    fn render_cluster_tree_leaf() {
        let node = ClusterNode {
            label: "Test".to_string(),
            files: vec![FileInfo {
                relative_path: "src/main.rs".to_string(),
                header: "entry point".to_string(),
                content: String::new(),
                symbol_preview: vec!["main@L1".to_string()],
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(rendered.contains("[Test]"));
        assert!(rendered.contains("src/main.rs - entry point | symbols: main@L1"));
    }

    #[test]
    fn render_cluster_tree_nested() {
        let child = ClusterNode {
            label: "Child".to_string(),
            files: vec![FileInfo {
                relative_path: "a.rs".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let parent = ClusterNode {
            label: "Parent".to_string(),
            files: Vec::new(),
            children: vec![child],
        };
        let rendered = render_cluster_tree(&parent, 0);
        assert!(rendered.contains("[Parent]"));
        assert!(rendered.contains("  [Child]"));
        assert!(rendered.contains("    a.rs"));
    }

    #[test]
    fn extract_header_sql_dash_comments() {
        let content = "-- Migration script\n-- Adds users table\nCREATE TABLE users();";
        let header = extract_header(content);
        assert_eq!(header, "Migration script Adds users table");
    }

    #[test]
    fn extract_header_empty_leading_lines() {
        let content = "\n\n// After blanks\nfn main() {}";
        let header = extract_header(content);
        assert_eq!(header, "After blanks");
    }

    #[test]
    fn extract_header_empty_content() {
        let header = extract_header("");
        assert_eq!(header, "");
    }

    #[test]
    fn extract_json_array_nested_brackets() {
        let text = r#"Sure: [["a", "b"], ["c"]] done"#;
        let arr = extract_json_array(text);
        assert_eq!(arr, Some(r#"[["a", "b"], ["c"]]"#.to_string()));
    }

    #[test]
    fn extract_json_array_garbage_text() {
        assert_eq!(extract_json_array("just some random text"), None);
        assert_eq!(extract_json_array("has ] but no ["), None);
    }

    #[test]
    fn extract_json_array_only_brackets() {
        let arr = extract_json_array("[]");
        assert_eq!(arr, Some("[]".to_string()));
    }

    #[test]
    fn navigate_extensions_contains_expected() {
        assert!(NAVIGATE_EXTENSIONS.contains(&"rs"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"ts"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"tsx"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"py"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"go"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"sql"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"proto"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"json"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"yaml"));
    }

    #[test]
    fn navigate_extensions_excludes_binary_formats() {
        assert!(!NAVIGATE_EXTENSIONS.contains(&"png"));
        assert!(!NAVIGATE_EXTENSIONS.contains(&"jpg"));
        assert!(!NAVIGATE_EXTENSIONS.contains(&"exe"));
        assert!(!NAVIGATE_EXTENSIONS.contains(&"wasm"));
    }

    #[tokio::test]
    async fn collect_source_files_filters_by_extension() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        tokio::fs::write(root.join("code.rs"), "fn main() {}")
            .await
            .expect("write");
        tokio::fs::write(root.join("image.png"), "fake png")
            .await
            .expect("write");
        tokio::fs::write(root.join("readme.md"), "# Hello")
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 1);
        assert!(files[0].relative_path.contains("code.rs"));
    }

    #[tokio::test]
    async fn collect_source_files_from_temp() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        // Create some source files
        let src = root.join("src");
        tokio::fs::create_dir_all(&src).await.expect("mkdir");
        tokio::fs::write(src.join("main.rs"), "// Entry point\nfn main() {}")
            .await
            .expect("write");
        tokio::fs::write(src.join("lib.rs"), "pub mod core;")
            .await
            .expect("write");
        // Create an ignored directory
        let nm = root.join("node_modules");
        tokio::fs::create_dir_all(&nm).await.expect("mkdir");
        tokio::fs::write(nm.join("pkg.js"), "module.exports = {}")
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|f| f.relative_path.contains("main.rs")));
        assert!(files.iter().any(|f| f.relative_path.contains("lib.rs")));
    }

    // --- extract_header additional tests ---

    #[test]
    fn extract_header_mixed_comment_styles() {
        // Only lines starting with //, #, or -- are collected; stops at first non-comment, non-empty line
        let content = "// Rust comment\n# Python comment\ncode_here()";
        let header = extract_header(content);
        assert_eq!(header, "Rust comment Python comment");
    }

    #[test]
    fn extract_header_stops_at_five_lines() {
        // extract_header only inspects the first 5 lines
        let content = "// line 1\n// line 2\n// line 3\n// line 4\n// line 5\n// line 6\ncode";
        let header = extract_header(content);
        assert_eq!(header, "line 1 line 2 line 3 line 4 line 5");
        assert!(!header.contains("line 6"));
    }

    #[test]
    fn extract_header_only_whitespace_lines() {
        let content = "   \n   \n   \n   \n   \n";
        let header = extract_header(content);
        // whitespace-only lines are not empty ("" after trim), so the first non-empty non-comment stops it
        assert_eq!(header, "");
    }

    #[test]
    fn extract_header_comment_with_extra_slashes() {
        // trim_start_matches("//") strips "//" but leaves the third "/"
        let content = "/// Documentation comment\nfn foo() {}";
        let header = extract_header(content);
        assert_eq!(header, "/ Documentation comment");
    }

    #[test]
    fn extract_header_multiple_hashes() {
        let content = "### Section heading\nsome code";
        let header = extract_header(content);
        assert_eq!(header, "Section heading");
    }

    #[test]
    fn extract_header_dash_comment_single_dash_not_matched() {
        // Single dash doesn't match "--" prefix, so it stops header extraction
        let content = "- list item\ncode";
        let header = extract_header(content);
        assert_eq!(header, "");
    }

    // --- extract_json_array additional tests ---

    #[test]
    fn extract_json_array_with_leading_text_and_trailing_text() {
        let text = "```json\n[{\"label\": \"Auth\"}]\n```";
        let arr = extract_json_array(text);
        assert_eq!(arr, Some("[{\"label\": \"Auth\"}]".to_string()));
    }

    #[test]
    fn extract_json_array_multiple_arrays_picks_outer() {
        // find('[') gives first bracket, rfind(']') gives last bracket
        let text = "[\"a\"] and then [\"b\"]";
        let arr = extract_json_array(text);
        // Should span from first [ to last ]
        assert_eq!(arr, Some("[\"a\"] and then [\"b\"]".to_string()));
    }

    #[test]
    fn extract_json_array_bracket_only_close_before_open() {
        // ] appears before [ in the string; find('[') returns position after rfind(']')
        let text = "] some text [";
        let arr = extract_json_array(text);
        // find('[') = 12, rfind(']') = 0 => end < start => None
        assert_eq!(arr, None);
    }

    #[test]
    fn extract_json_array_objects_array() {
        let text =
            r#"[{"overarchingTheme":"auth","distinguishingFeature":"login","label":"User Auth"}]"#;
        let arr = extract_json_array(text);
        assert_eq!(arr, Some(text.to_string()));
    }

    // --- render_cluster_tree additional tests ---

    #[test]
    fn render_cluster_tree_empty_label_no_bracket_line() {
        let node = ClusterNode {
            label: String::new(),
            files: vec![FileInfo {
                relative_path: "foo.rs".to_string(),
                header: "a file".to_string(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        // No label line should appear
        assert!(!rendered.contains('['));
        assert!(rendered.contains("foo.rs - a file"));
    }

    #[test]
    fn render_cluster_tree_file_no_header_no_symbols() {
        let node = ClusterNode {
            label: "Leaf".to_string(),
            files: vec![FileInfo {
                relative_path: "bare.ts".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(rendered.contains("[Leaf]"));
        // No " - " label and no " | symbols:" suffix
        assert!(rendered.contains("  bare.ts\n"));
        assert!(!rendered.contains(" - "));
        assert!(!rendered.contains("symbols:"));
    }

    #[test]
    fn render_cluster_tree_multiple_files_in_leaf() {
        let node = ClusterNode {
            label: "Group".to_string(),
            files: vec![
                FileInfo {
                    relative_path: "a.rs".to_string(),
                    header: "first".to_string(),
                    content: String::new(),
                    symbol_preview: Vec::new(),
                },
                FileInfo {
                    relative_path: "b.rs".to_string(),
                    header: "second".to_string(),
                    content: String::new(),
                    symbol_preview: vec!["fn_b@L1".to_string()],
                },
                FileInfo {
                    relative_path: "c.rs".to_string(),
                    header: String::new(),
                    content: String::new(),
                    symbol_preview: vec!["fn_c@L1".to_string(), "fn_d@L5".to_string()],
                },
            ],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(rendered.contains("a.rs - first\n"));
        assert!(rendered.contains("b.rs - second | symbols: fn_b@L1\n"));
        assert!(rendered.contains("c.rs | symbols: fn_c@L1, fn_d@L5\n"));
    }

    #[test]
    fn render_cluster_tree_with_initial_indent() {
        let node = ClusterNode {
            label: "Deep".to_string(),
            files: vec![FileInfo {
                relative_path: "x.py".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 3);
        // 3 levels of indent = 6 spaces before [Deep]
        assert!(rendered.starts_with("      [Deep]"));
        // File should be at indent 3 + 2 more spaces = 8 spaces
        assert!(rendered.contains("        x.py"));
    }

    #[test]
    fn render_cluster_tree_deeply_nested_three_levels() {
        let grandchild = ClusterNode {
            label: "Grandchild".to_string(),
            files: vec![FileInfo {
                relative_path: "gc.rs".to_string(),
                header: "deep file".to_string(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let child = ClusterNode {
            label: "Child".to_string(),
            files: Vec::new(),
            children: vec![grandchild],
        };
        let root = ClusterNode {
            label: "Root".to_string(),
            files: Vec::new(),
            children: vec![child],
        };
        let rendered = render_cluster_tree(&root, 0);
        assert!(rendered.contains("[Root]"));
        assert!(rendered.contains("  [Child]"));
        assert!(rendered.contains("    [Grandchild]"));
        assert!(rendered.contains("      gc.rs - deep file"));
    }

    #[test]
    fn render_cluster_tree_multiple_children() {
        let child_a = ClusterNode {
            label: "A".to_string(),
            files: vec![FileInfo {
                relative_path: "a.rs".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let child_b = ClusterNode {
            label: "B".to_string(),
            files: vec![FileInfo {
                relative_path: "b.rs".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let parent = ClusterNode {
            label: "Parent".to_string(),
            files: Vec::new(),
            children: vec![child_a, child_b],
        };
        let rendered = render_cluster_tree(&parent, 0);
        assert!(rendered.contains("[Parent]"));
        assert!(rendered.contains("  [A]"));
        assert!(rendered.contains("    a.rs"));
        assert!(rendered.contains("  [B]"));
        assert!(rendered.contains("    b.rs"));
    }

    #[test]
    fn render_cluster_tree_empty_node() {
        // Node with no label, no files, no children produces empty string
        let node = ClusterNode {
            label: String::new(),
            files: Vec::new(),
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert_eq!(rendered, "");
    }

    #[test]
    fn render_cluster_tree_children_take_priority_over_files() {
        // When a node has both children and files, only children are rendered
        let child = ClusterNode {
            label: "Child".to_string(),
            files: vec![FileInfo {
                relative_path: "child.rs".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let parent = ClusterNode {
            label: "Parent".to_string(),
            files: vec![FileInfo {
                relative_path: "parent_file.rs".to_string(),
                header: "should not appear".to_string(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: vec![child],
        };
        let rendered = render_cluster_tree(&parent, 0);
        // The parent's files are not rendered because children exist
        assert!(!rendered.contains("parent_file.rs"));
        assert!(rendered.contains("child.rs"));
    }

    // --- collect_source_files_via_walker additional tests ---

    #[tokio::test]
    async fn collect_source_files_truncates_long_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        let long_content = "a".repeat(1000);
        tokio::fs::write(root.join("long.rs"), &long_content)
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 1);
        // Content should be truncated to ~500 bytes
        assert!(
            files[0].content.len() <= 500,
            "content len {} should be <= 500",
            files[0].content.len()
        );
    }

    #[tokio::test]
    async fn collect_source_files_short_content_not_truncated() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        let short_content = "fn main() {}";
        tokio::fs::write(root.join("short.rs"), short_content)
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].content, short_content);
    }

    #[tokio::test]
    async fn collect_source_files_extracts_header() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        tokio::fs::write(
            root.join("documented.py"),
            "# Data processor\nimport pandas",
        )
        .await
        .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].header, "Data processor");
    }

    #[tokio::test]
    async fn collect_source_files_multiple_extensions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        tokio::fs::write(root.join("main.rs"), "fn main() {}")
            .await
            .expect("write");
        tokio::fs::write(root.join("app.ts"), "console.log('hello')")
            .await
            .expect("write");
        tokio::fs::write(root.join("script.py"), "print('hi')")
            .await
            .expect("write");
        tokio::fs::write(root.join("config.yaml"), "key: value")
            .await
            .expect("write");
        tokio::fs::write(root.join("query.sql"), "SELECT 1")
            .await
            .expect("write");
        tokio::fs::write(root.join("schema.proto"), "syntax = \"proto3\";")
            .await
            .expect("write");
        // Should be excluded:
        tokio::fs::write(root.join("styles.css"), "body {}")
            .await
            .expect("write");
        tokio::fs::write(root.join("page.html"), "<html></html>")
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        // 6 accepted extensions, 2 excluded
        assert_eq!(files.len(), 6);
        let paths: Vec<&str> = files.iter().map(|f| f.relative_path.as_str()).collect();
        assert!(!paths.iter().any(|p| p.contains(".css")));
        assert!(!paths.iter().any(|p| p.contains(".html")));
    }

    #[tokio::test]
    async fn collect_source_files_empty_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert!(files.is_empty());
    }

    #[tokio::test]
    async fn collect_source_files_nested_directories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        let deep = root.join("a").join("b").join("c");
        tokio::fs::create_dir_all(&deep).await.expect("mkdir");
        tokio::fs::write(deep.join("deep.go"), "package main")
            .await
            .expect("write");
        tokio::fs::write(root.join("top.js"), "export default {}")
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|f| f.relative_path.contains("deep.go")));
        assert!(files.iter().any(|f| f.relative_path.contains("top.js")));
    }

    #[tokio::test]
    async fn collect_source_files_symbol_preview_is_empty() {
        // symbol_preview is always empty in collect_source_files_via_walker
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        tokio::fs::write(root.join("mod.rs"), "pub fn foo() {}\npub fn bar() {}")
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 1);
        assert!(files[0].symbol_preview.is_empty());
    }

    // --- SemanticNavigateOptions tests ---

    #[test]
    fn semantic_navigate_options_defaults() {
        let opts = SemanticNavigateOptions {
            root_dir: "/tmp".to_string(),
            max_depth: None,
            max_clusters: None,
        };
        assert_eq!(opts.root_dir, "/tmp");
        assert!(opts.max_depth.is_none());
        assert!(opts.max_clusters.is_none());
    }

    #[test]
    fn semantic_navigate_options_with_values() {
        let opts = SemanticNavigateOptions {
            root_dir: "/project".to_string(),
            max_depth: Some(5),
            max_clusters: Some(10),
        };
        assert_eq!(opts.max_depth, Some(5));
        assert_eq!(opts.max_clusters, Some(10));
    }

    // --- MAX_FILES_PER_LEAF constant test ---

    #[test]
    fn max_files_per_leaf_is_20() {
        assert_eq!(MAX_FILES_PER_LEAF, 20);
    }

    // --- FileInfo struct tests ---

    #[test]
    fn file_info_clone() {
        let fi = FileInfo {
            relative_path: "src/lib.rs".to_string(),
            header: "Library root".to_string(),
            content: "pub mod foo;".to_string(),
            symbol_preview: vec!["foo@L1".to_string()],
        };
        let cloned = fi.clone();
        assert_eq!(cloned.relative_path, fi.relative_path);
        assert_eq!(cloned.header, fi.header);
        assert_eq!(cloned.content, fi.content);
        assert_eq!(cloned.symbol_preview, fi.symbol_preview);
    }

    #[test]
    fn file_info_debug() {
        let fi = FileInfo {
            relative_path: "test.rs".to_string(),
            header: String::new(),
            content: String::new(),
            symbol_preview: Vec::new(),
        };
        let debug = format!("{:?}", fi);
        assert!(debug.contains("FileInfo"));
        assert!(debug.contains("test.rs"));
    }

    // --- ClusterNode struct tests ---

    #[test]
    fn cluster_node_clone() {
        let node = ClusterNode {
            label: "Test".to_string(),
            files: vec![FileInfo {
                relative_path: "src/a.rs".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let cloned = node.clone();
        assert_eq!(cloned.label, node.label);
        assert_eq!(cloned.files.len(), 1);
    }

    #[test]
    fn cluster_node_debug() {
        let node = ClusterNode {
            label: "Debug test".to_string(),
            files: Vec::new(),
            children: Vec::new(),
        };
        let debug = format!("{:?}", node);
        assert!(debug.contains("ClusterNode"));
        assert!(debug.contains("Debug test"));
    }

    // --- NAVIGATE_EXTENSIONS completeness tests ---

    #[test]
    fn navigate_extensions_covers_all_listed() {
        let expected = vec![
            "rs", "ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "go", "java", "c", "cpp", "h",
            "hpp", "cc", "rb", "sh", "bash", "zsh", "sql", "graphql", "proto", "yaml", "yml",
            "toml", "json",
        ];
        for ext in &expected {
            assert!(
                NAVIGATE_EXTENSIONS.contains(ext),
                "expected extension '{}' not in NAVIGATE_EXTENSIONS",
                ext
            );
        }
        assert_eq!(NAVIGATE_EXTENSIONS.len(), expected.len());
    }

    // --- extract_header boundary / edge case tests ---

    #[test]
    fn extract_header_exactly_200_bytes() {
        // Create a comment that is exactly 200 chars after stripping the "//"
        let comment_body = "a".repeat(200);
        let content = format!("// {}", comment_body);
        let header = extract_header(&content);
        // 200 chars exactly => should be kept as-is (len <= 200)
        assert_eq!(header.len(), 200);
    }

    #[test]
    fn extract_header_201_bytes_truncated() {
        let comment_body = "b".repeat(201);
        let content = format!("// {}", comment_body);
        let header = extract_header(&content);
        assert_eq!(header.len(), 200);
    }

    #[test]
    fn extract_header_blank_comment_lines() {
        // Comment markers with no text after them
        let content = "//\n//\ncode()";
        let header = extract_header(content);
        // Each line after stripping "//" and trimming gives "", joined with " " => " "
        assert_eq!(header, " ");
    }

    #[test]
    fn extract_header_non_comment_first_line() {
        let content = "use std::io;\n// This is a comment";
        let header = extract_header(content);
        // First non-empty line is not a comment, so extraction stops immediately
        assert_eq!(header, "");
    }

    // --- render_cluster_tree additional edge cases ---

    #[test]
    fn render_cluster_tree_file_with_symbols_no_header() {
        let node = ClusterNode {
            label: "Syms".to_string(),
            files: vec![FileInfo {
                relative_path: "lib.rs".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec!["Foo@L1".to_string(), "Bar@L10".to_string()],
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        // Should have symbols but no header dash
        assert!(rendered.contains("lib.rs | symbols: Foo@L1, Bar@L10"));
        assert!(!rendered.contains(" - "));
    }

    #[test]
    fn render_cluster_tree_file_with_header_and_symbols() {
        let node = ClusterNode {
            label: "Full".to_string(),
            files: vec![FileInfo {
                relative_path: "main.go".to_string(),
                header: "entry".to_string(),
                content: String::new(),
                symbol_preview: vec!["main@L1".to_string()],
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(rendered.contains("main.go - entry | symbols: main@L1"));
    }

    // --- extract_json_array edge cases ---

    #[test]
    fn extract_json_array_with_newlines() {
        let text = "Here:\n[\n  \"a\",\n  \"b\"\n]\nDone.";
        let arr = extract_json_array(text);
        assert!(arr.is_some());
        let val = arr.unwrap();
        assert!(val.starts_with('['));
        assert!(val.ends_with(']'));
        assert!(val.contains("\"a\""));
    }

    #[test]
    fn extract_json_array_empty_string() {
        assert_eq!(extract_json_array(""), None);
    }

    #[test]
    fn extract_json_array_only_open_bracket() {
        // Has '[' but no ']'
        assert_eq!(extract_json_array("["), None);
    }

    #[test]
    fn extract_json_array_only_close_bracket() {
        // Has ']' but no '['
        assert_eq!(extract_json_array("]"), None);
    }

    // ── resolve_embeddings cache tests ────────────────────────────────

    fn make_test_file(path: &str) -> FileInfo {
        FileInfo {
            relative_path: path.to_string(),
            header: format!("header for {path}"),
            content: format!("content of {path}"),
            symbol_preview: vec![],
        }
    }

    /// Build a CacheEntry with the correct content hash for a test file.
    fn make_cache_entry(path: &str, vector: Vec<f32>) -> CacheEntry {
        let file = make_test_file(path);
        CacheEntry {
            hash: nav_content_hash(&file.relative_path, &file.content),
            vector,
        }
    }

    #[tokio::test]
    async fn resolve_embeddings_all_cached_skips_ollama() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Pre-populate cache with known vectors (matching content hashes)
        let cache: RwLock<HashMap<String, CacheEntry>> = RwLock::new(HashMap::new());
        {
            let mut w = cache.write().await;
            w.insert(
                "src/a.rs".to_string(),
                make_cache_entry("src/a.rs", vec![1.0, 2.0, 3.0]),
            );
            w.insert(
                "src/b.rs".to_string(),
                make_cache_entry("src/b.rs", vec![4.0, 5.0, 6.0]),
            );
        }

        let files = vec![make_test_file("src/a.rs"), make_test_file("src/b.rs")];

        // OllamaClient pointing to a non-existent server — if it tries to call
        // Ollama it will fail. Success proves cache was used.
        let config = crate::config::Config::from_env();
        let mut bad_config = config.clone();
        bad_config.ollama_host = "http://127.0.0.1:1".to_string(); // unreachable
        let ollama = OllamaClient::new(&bad_config);

        let result = resolve_embeddings(&files, &ollama, &cache, tmp.path(), "test-model").await;
        assert!(result.is_ok(), "should succeed from cache without Ollama");
        let vecs = result.unwrap();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(vecs[1], vec![4.0, 5.0, 6.0]);
    }

    #[tokio::test]
    async fn resolve_embeddings_partial_cache_only_embeds_missing() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let tmp = tempfile::TempDir::new().unwrap();
        let mock_server = MockServer::start().await;

        // Return a single 3-dim vector for each embed call
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [[7.0, 8.0, 9.0]]
            })))
            .expect(1) // Should only be called ONCE (for the uncached file)
            .mount(&mock_server)
            .await;

        let cache: RwLock<HashMap<String, CacheEntry>> = RwLock::new(HashMap::new());
        {
            let mut w = cache.write().await;
            w.insert(
                "src/cached.rs".to_string(),
                make_cache_entry("src/cached.rs", vec![1.0, 2.0, 3.0]),
            );
        }

        let files = vec![
            make_test_file("src/cached.rs"),
            make_test_file("src/uncached.rs"),
        ];

        let mut config = crate::config::Config::from_env();
        config.ollama_host = mock_server.uri();
        let ollama = OllamaClient::new(&config);

        let result = resolve_embeddings(&files, &ollama, &cache, tmp.path(), "test-model").await;
        assert!(result.is_ok());
        let vecs = result.unwrap();
        assert_eq!(vecs[0], vec![1.0, 2.0, 3.0]); // from cache
        assert_eq!(vecs[1], vec![7.0, 8.0, 9.0]); // from Ollama

        // Verify the newly embedded vector was stored in cache with correct hash
        let r = cache.read().await;
        let entry = r.get("src/uncached.rs").unwrap();
        assert_eq!(entry.vector, vec![7.0, 8.0, 9.0]);
        assert_eq!(entry.hash, nav_content_hash("src/uncached.rs", "content of src/uncached.rs"));
    }

    #[tokio::test]
    async fn resolve_embeddings_stores_all_new_vectors_in_cache() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let tmp = tempfile::TempDir::new().unwrap();
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [[1.0, 0.0], [0.0, 1.0]]
            })))
            .mount(&mock_server)
            .await;

        let cache: RwLock<HashMap<String, CacheEntry>> = RwLock::new(HashMap::new());
        let files = vec![make_test_file("a.ts"), make_test_file("b.ts")];

        let mut config = crate::config::Config::from_env();
        config.ollama_host = mock_server.uri();
        let ollama = OllamaClient::new(&config);

        let result = resolve_embeddings(&files, &ollama, &cache, tmp.path(), "test-model")
            .await
            .unwrap();
        assert_eq!(result.len(), 2);

        // Both should now be in cache
        let r = cache.read().await;
        assert_eq!(r.len(), 2);
        assert!(r.contains_key("a.ts"));
        assert!(r.contains_key("b.ts"));
    }

    #[tokio::test]
    async fn resolve_embeddings_empty_files_returns_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache: RwLock<HashMap<String, CacheEntry>> = RwLock::new(HashMap::new());
        let files: Vec<FileInfo> = vec![];

        let config = crate::config::Config::from_env();
        let ollama = OllamaClient::new(&config);

        let result = resolve_embeddings(&files, &ollama, &cache, tmp.path(), "test-model")
            .await
            .unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn resolve_embeddings_stale_hash_triggers_reembed() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let tmp = tempfile::TempDir::new().unwrap();
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [[9.0, 9.0, 9.0]]
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        // Cache entry has a stale hash (doesn't match current file content)
        let cache: RwLock<HashMap<String, CacheEntry>> = RwLock::new(HashMap::new());
        {
            let mut w = cache.write().await;
            w.insert(
                "src/changed.rs".to_string(),
                CacheEntry {
                    hash: "stale_hash_that_wont_match".to_string(),
                    vector: vec![1.0, 1.0, 1.0],
                },
            );
        }

        let files = vec![make_test_file("src/changed.rs")];

        let mut config = crate::config::Config::from_env();
        config.ollama_host = mock_server.uri();
        let ollama = OllamaClient::new(&config);

        let result = resolve_embeddings(&files, &ollama, &cache, tmp.path(), "test-model")
            .await
            .unwrap();
        // Should have re-embedded with new vector from Ollama
        assert_eq!(result[0], vec![9.0, 9.0, 9.0]);

        // Cache should now have the new vector and correct hash
        let r = cache.read().await;
        let entry = r.get("src/changed.rs").unwrap();
        assert_eq!(entry.vector, vec![9.0, 9.0, 9.0]);
        assert_eq!(entry.hash, content_hash("nav3:src/changed.rscontent of src/changed.rs"));
    }

    // ── group_by_directory tests ─────────────────────────────────────

    #[test]
    fn group_by_directory_monorepo_domains() {
        let mut files = Vec::new();
        // Need >= MIN_DIR_GROUP_SIZE (15) files per group to avoid merging into "other"
        for i in 0..16 {
            files.push(make_test_file(&format!("packages/domains/billing/file{}.ts", i)));
        }
        for i in 0..16 {
            files.push(make_test_file(&format!("packages/domains/scheduling/file{}.ts", i)));
        }
        let groups = group_by_directory(&files);
        let labels: Vec<&str> = groups.iter().map(|g| g.0.as_str()).collect();
        assert!(labels.contains(&"domains/billing"), "Expected 'domains/billing', got {:?}", labels);
        assert!(labels.contains(&"domains/scheduling"), "Expected 'domains/scheduling', got {:?}", labels);
    }

    #[test]
    fn group_by_directory_root_files() {
        let files = vec![
            make_test_file("main.rs"),
            make_test_file("Cargo.toml"),
        ];
        let groups = group_by_directory(&files);
        // Both are single-component paths → "root" label, but < MIN_DIR_GROUP_SIZE → merged to "other"
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "other");
        assert_eq!(groups[0].1.len(), 2);
    }

    #[test]
    fn group_by_directory_empty() {
        let groups = group_by_directory(&[]);
        assert!(groups.is_empty());
    }

    #[test]
    fn group_by_directory_merges_tiny_groups() {
        // Create two groups each with < MIN_DIR_GROUP_SIZE files
        let mut files = Vec::new();
        for i in 0..3 {
            files.push(make_test_file(&format!("alpha/deep/file{}.ts", i)));
        }
        for i in 0..3 {
            files.push(make_test_file(&format!("beta/deep/file{}.ts", i)));
        }
        let groups = group_by_directory(&files);
        // Both groups have 3 files (< 15), so all should merge into "other"
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "other");
        assert_eq!(groups[0].1.len(), 6);
    }

    // ── derive_cluster_label tests ───────────────────────────────────

    #[test]
    fn derive_cluster_label_deep_common_prefix() {
        let files: Vec<FileInfo> = (0..5)
            .map(|i| make_test_file(&format!("packages/domains/billing/file{}.ts", i)))
            .collect();
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert!(label.is_some());
        let l = label.unwrap();
        assert!(l.contains("billing"), "Expected 'billing' in label, got '{}'", l);
    }

    #[test]
    fn derive_cluster_label_generic_prefix_fallback() {
        let files: Vec<FileInfo> = vec![
            make_test_file("src/controllers/auth.ts"),
            make_test_file("src/controllers/user.ts"),
            make_test_file("src/controllers/billing.ts"),
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        // "src" is generic → should look deeper → "src/controllers"
        assert!(label.is_some());
        let l = label.unwrap();
        assert!(l.contains("controllers"), "Expected 'controllers' in label, got '{}'", l);
    }

    #[test]
    fn derive_cluster_label_empty() {
        let label = derive_cluster_label(&[]);
        assert_eq!(label, None);
    }

    #[test]
    fn derive_cluster_label_no_common_prefix() {
        let files: Vec<FileInfo> = vec![
            make_test_file("alpha/foo.ts"),
            make_test_file("beta/bar.ts"),
            make_test_file("gamma/baz.ts"),
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        // No common prefix, no dominant segment → None
        assert_eq!(label, None);
    }

    // ── count_files_in_node tests ────────────────────────────────────

    #[test]
    fn count_files_leaf_node() {
        let node = ClusterNode {
            label: "test".to_string(),
            files: vec![make_test_file("a.rs"), make_test_file("b.rs")],
            children: Vec::new(),
        };
        assert_eq!(count_files_in_node(&node), 2);
    }

    #[test]
    fn count_files_nested_children() {
        let child1 = ClusterNode {
            label: "c1".to_string(),
            files: vec![make_test_file("a.rs"), make_test_file("b.rs"), make_test_file("c.rs")],
            children: Vec::new(),
        };
        let child2 = ClusterNode {
            label: "c2".to_string(),
            files: vec![make_test_file("d.rs"), make_test_file("e.rs")],
            children: Vec::new(),
        };
        let parent = ClusterNode {
            label: "parent".to_string(),
            files: Vec::new(),
            children: vec![child1, child2],
        };
        assert_eq!(count_files_in_node(&parent), 5);
    }

    #[tokio::test]
    async fn label_files_returns_headers() {
        let files = vec![
            make_test_file("src/auth.rs"), // has header "header for src/auth.rs"
            FileInfo {
                relative_path: "src/empty.rs".to_string(),
                header: String::new(),
                content: "fn main() {}".to_string(),
                symbol_preview: vec![],
            },
        ];
        let labels = label_files(&files).await;
        assert_eq!(labels[0], "header for src/auth.rs");
        assert_eq!(labels[1], "empty.rs"); // filename fallback
    }

    #[test]
    fn render_cluster_tree_truncates_large_leaves() {
        let files: Vec<FileInfo> = (0..15)
            .map(|i| make_test_file(&format!("src/file{}.rs", i)))
            .collect();
        let node = ClusterNode {
            label: "big leaf".to_string(),
            files,
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(rendered.contains("[big leaf]"));
        assert!(rendered.contains("(+5 more files)"));
        // Should show exactly MAX_FILES_PER_LEAF_DISPLAY file lines
        let file_lines = rendered
            .lines()
            .filter(|l| l.contains("src/file"))
            .count();
        assert_eq!(file_lines, 10); // MAX_FILES_PER_LEAF_DISPLAY
    }

    #[test]
    fn render_cluster_tree_shows_file_count() {
        let node = ClusterNode {
            label: "test".to_string(),
            files: vec![make_test_file("a.rs"), make_test_file("b.rs")],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(
            rendered.contains("[test] (2 files)"),
            "got: {}",
            rendered
        );
    }
}
