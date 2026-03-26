// Semantic project navigator using spectral clustering and Ollama labeling.
// Browse codebase by meaning: embeds files, clusters vectors, generates labels.

use crate::cache::rkyv_store;
use crate::config::Config;
use crate::core::clustering::spectral_cluster_with_min;
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
    _embedding_cache: &RwLock<HashMap<String, CacheEntry>>,
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
    if let Ok(Some(store)) = rkyv_store::mmap_vector_store(root_dir, &nav_cache_name) {
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

    // Phase 1: Cluster all groups concurrently (CPU-bound, no LLM).
    // Phase 2: Label all groups' sub-clusters in ONE batched LLM call.
    // This avoids both sequential slowness and concurrent LLM queue explosion.

    // First, cluster all groups to find their sub-clusters
    let cluster_futures: Vec<_> = large_groups
        .iter()
        .map(|(_, indices)| {
            let local_vecs: Vec<Vec<f32>> = indices.iter().map(|&i| vectors[i].clone()).collect();
            let mc = max_clusters;
            async move {
                tokio::task::spawn_blocking(move || {
                    spectral_cluster_with_min(&local_vecs, mc, 2)
                })
                .await
                .unwrap_or_else(|_| vec![])
            }
        })
        .collect();

    let all_cluster_results = futures::future::join_all(cluster_futures).await;

    // Now collect ALL unlabeled sub-clusters from ALL groups into one batch
    // and make a single LLM call to label them all
    struct PendingGroup {
        label: String,
        indices: Vec<usize>,
        cluster_results: Vec<crate::core::clustering::ClusterResult>,
    }

    let pending_groups: Vec<PendingGroup> = large_groups
        .into_iter()
        .zip(all_cluster_results)
        .map(|((label, indices), clusters)| PendingGroup {
            label,
            indices,
            cluster_results: clusters,
        })
        .collect();

    // Collect all sub-cluster file lists for one batched LLM call
    let mut all_sublabels: Vec<(usize, usize, Vec<&FileInfo>)> = Vec::new(); // (group_idx, cluster_idx, files)
    for (gi, group) in pending_groups.iter().enumerate() {
        if group.cluster_results.len() <= 1 {
            continue;
        }
        for (ci, cluster) in group.cluster_results.iter().enumerate() {
            let file_refs: Vec<&FileInfo> = cluster.indices
                .iter()
                .map(|&li| &files[group.indices[li]])
                .collect();
            if file_refs.len() > 3 {
                all_sublabels.push((gi, ci, file_refs));
            }
        }
    }

    // Batch LLM call for sub-clusters. Cap at 10 clusters per call to keep
    // prompt size manageable (~11s for 4 clusters, ~30s for 10).
    // Clusters beyond the cap use path-based fallbacks.
    const MAX_LLM_CLUSTERS: usize = 10;
    let mut llm_label_map: HashMap<(usize, usize), String> = HashMap::new();

    // Sort by file count descending — label the biggest clusters with LLM first
    all_sublabels.sort_by(|a, b| b.2.len().cmp(&a.2.len()));
    let llm_batch = if all_sublabels.len() > MAX_LLM_CLUSTERS {
        &all_sublabels[..MAX_LLM_CLUSTERS]
    } else {
        &all_sublabels
    };

    if !llm_batch.is_empty() {
        const MAX_FILES_PER_LABEL: usize = 5;
        let descriptions: Vec<String> = llm_batch
            .iter()
            .enumerate()
            .map(|(desc_idx, (gi, _, file_refs))| {
                let parent_label = &pending_groups[*gi].label;

                // Group files by subdirectory for representative sampling
                let mut subdir_files: HashMap<String, Vec<&FileInfo>> = HashMap::new();
                for f in file_refs.iter() {
                    let subdir = Path::new(&f.relative_path)
                        .parent()
                        .and_then(|p| {
                            let components: Vec<_> = p.components().collect();
                            if components.is_empty() {
                                None
                            } else {
                                let depth = components.len().min(2);
                                let sub: PathBuf = components[..depth].iter().collect();
                                Some(sub.to_string_lossy().to_string())
                            }
                        })
                        .unwrap_or_else(|| ".".to_string());
                    subdir_files.entry(subdir).or_default().push(f);
                }

                // Subdirectory summary sorted by count descending
                let mut subdir_counts: Vec<(String, usize)> = subdir_files
                    .iter()
                    .map(|(dir, files)| (dir.clone(), files.len()))
                    .collect();
                subdir_counts.sort_by(|a, b| b.1.cmp(&a.1));
                let subdir_summary = subdir_counts
                    .iter()
                    .map(|(dir, count)| format!("{} ({})", dir, count))
                    .collect::<Vec<_>>()
                    .join(", ");

                // Round-robin across subdirectories for representative sample
                let sample: Vec<&FileInfo> = if file_refs.len() <= MAX_FILES_PER_LABEL {
                    file_refs.iter().copied().collect()
                } else {
                    let mut sorted_dirs: Vec<(&String, &Vec<&FileInfo>)> =
                        subdir_files.iter().collect();
                    sorted_dirs.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

                    let mut picked: Vec<&FileInfo> = Vec::with_capacity(MAX_FILES_PER_LABEL);
                    let mut dir_indices: Vec<usize> = vec![0; sorted_dirs.len()];
                    while picked.len() < MAX_FILES_PER_LABEL {
                        let mut added_this_round = false;
                        for (di, (_, files)) in sorted_dirs.iter().enumerate() {
                            if picked.len() >= MAX_FILES_PER_LABEL {
                                break;
                            }
                            let idx = dir_indices[di];
                            if idx < files.len() {
                                picked.push(files[idx]);
                                dir_indices[di] += 1;
                                added_this_round = true;
                            }
                        }
                        if !added_this_round {
                            break;
                        }
                    }
                    picked
                };

                let file_list = sample
                    .iter()
                    .map(|f| {
                        let desc = if f.header.is_empty() { "no description" } else { &f.header };
                        format!("{}: {}", f.relative_path, desc)
                    })
                    .collect::<Vec<_>>()
                    .join("\n  ");
                format!(
                    "Cluster {} ({} files, within \"{}\"):\n  Subdirectories: {}\n  Sample files:\n  {}",
                    desc_idx + 1, file_refs.len(), parent_label, subdir_summary, file_list
                )
            })
            .collect();

        let prompt = format!(
            "Label each cluster based on its OVERALL structure and subdirectory distribution, not individual filenames. Return EXACTLY 2 words per label. Return ONLY a JSON array of strings, one per cluster.\n\n{}\n\nJSON array of {} strings:",
            descriptions.join("\n\n"),
            llm_batch.len()
        );

        if let Ok(response) = ollama.chat(&prompt).await {
            if let Some(json_str) = extract_json_array(&response) {
                if let Ok(labels) = serde_json::from_str::<Vec<String>>(&json_str) {
                    for (j, (gi, ci, _)) in llm_batch.iter().enumerate() {
                        if let Some(label) = labels.get(j) {
                            llm_label_map.insert((*gi, *ci), label.clone());
                        }
                    }
                }
            }
        }
    }

    // Build the final tree using LLM labels where available, fallbacks otherwise
    for (gi, group) in pending_groups.into_iter().enumerate() {
        if group.cluster_results.len() <= 1 {
            // Single cluster or no split — leaf node
            children.push(ClusterNode {
                label: group.label,
                files: group.indices.iter().map(|&i| files[i].clone()).collect(),
                children: Vec::new(),
            });
            continue;
        }

        let group_label = group.label.clone();
        let mut sub_children: Vec<ClusterNode> = Vec::new();
        for (ci, cluster) in group.cluster_results.iter().enumerate() {
            let global_indices: Vec<usize> = cluster.indices.iter().map(|&li| group.indices[li]).collect();
            let refs: Vec<&FileInfo> = global_indices.iter().map(|&i| &files[i]).collect();

            // Try multiple label sources in order of quality
            let raw_label = llm_label_map.get(&(gi, ci)).cloned()
                .or_else(|| derive_cluster_label(&refs))
                .or_else(|| find_label_disambiguator(&refs));

            // Check if the label duplicates the parent group name
            let label = match raw_label {
                Some(l) if !label_matches_parent(&l, &group.label) => l,
                _ => {
                    // Label matches parent or is None — use disambiguator
                    find_label_disambiguator(&refs)
                        .filter(|d| !label_matches_parent(d, &group.label))
                        .unwrap_or_else(|| {
                            // Last resort: smart fallback using directory + file type heuristics
                            let refs2: Vec<&FileInfo> = global_indices.iter().map(|&i| &files[i]).collect();
                            describe_file_group(&refs2)
                        })
                }
            };

            // Depth 2: for large sub-clusters, do one more round of spectral clustering
            if global_indices.len() > MAX_FILES_PER_LEAF && max_depth > 2 {
                let sub_vecs: Vec<Vec<f32>> = global_indices.iter().map(|&i| vectors[i].clone()).collect();
                let mc = max_clusters;
                let sub_results = tokio::task::spawn_blocking(move || {
                    spectral_cluster_with_min(&sub_vecs, mc, 2)
                })
                .await
                .unwrap_or_else(|_| vec![]);

                if sub_results.len() > 1 {
                    let mut depth2_children: Vec<ClusterNode> = Vec::new();
                    for sub_cluster in &sub_results {
                        let d2_indices: Vec<usize> = sub_cluster.indices.iter().map(|&li| global_indices[li]).collect();
                        let refs: Vec<&FileInfo> = d2_indices.iter().map(|&i| &files[i]).collect();
                        let raw_d2 = derive_cluster_label(&refs)
                            .or_else(|| find_label_disambiguator(&refs));
                        let d2_label = match raw_d2 {
                            Some(l) if !label_matches_parent(&l, &label) && !label_matches_parent(&l, &group_label) => l,
                            _ => find_label_disambiguator(&refs)
                                .filter(|d| !label_matches_parent(d, &label) && !label_matches_parent(d, &group_label))
                                .unwrap_or_else(|| {
                                    // Last resort: smart fallback using directory + file type heuristics
                                    let refs2: Vec<&FileInfo> = d2_indices.iter().map(|&i| &files[i]).collect();
                                    describe_file_group(&refs2)
                                }),
                        };
                        depth2_children.push(ClusterNode {
                            label: d2_label,
                            files: d2_indices.iter().map(|&i| files[i].clone()).collect(),
                            children: Vec::new(),
                        });
                    }
                    // Deduplicate depth-2 labels
                    let mut d2_labels: Vec<String> = depth2_children.iter().map(|c| c.label.clone()).collect();
                    let d2_input: Vec<(Vec<&FileInfo>, Option<String>)> = depth2_children.iter()
                        .map(|c| (c.files.iter().collect::<Vec<&FileInfo>>(), None))
                        .collect();
                    deduplicate_sibling_labels(&mut d2_labels, &d2_input);
                    for (i, child) in depth2_children.iter_mut().enumerate() {
                        child.label = d2_labels[i].clone();
                    }

                    sub_children.push(ClusterNode {
                        label,
                        files: Vec::new(),
                        children: depth2_children,
                    });
                } else {
                    sub_children.push(ClusterNode {
                        label,
                        files: global_indices.iter().map(|&i| files[i].clone()).collect(),
                        children: Vec::new(),
                    });
                }
            } else {
                sub_children.push(ClusterNode {
                    label,
                    files: global_indices.iter().map(|&i| files[i].clone()).collect(),
                    children: Vec::new(),
                });
            }
        }

        // Deduplicate depth-1 sibling labels
        let mut d1_labels: Vec<String> = sub_children.iter().map(|c| c.label.clone()).collect();
        let d1_input: Vec<(Vec<&FileInfo>, Option<String>)> = sub_children.iter()
            .map(|c| {
                let refs: Vec<&FileInfo> = if c.children.is_empty() {
                    c.files.iter().collect()
                } else {
                    // For nodes with children, collect all descendant files
                    fn collect_files(node: &ClusterNode) -> Vec<&FileInfo> {
                        if node.children.is_empty() {
                            node.files.iter().collect()
                        } else {
                            node.children.iter().flat_map(collect_files).collect()
                        }
                    }
                    collect_files(c)
                };
                (refs, None)
            })
            .collect();
        deduplicate_sibling_labels(&mut d1_labels, &d1_input);
        for (i, child) in sub_children.iter_mut().enumerate() {
            child.label = d1_labels[i].clone();
        }

        children.push(ClusterNode {
            label: group.label,
            files: Vec::new(),
            children: sub_children,
        });
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
        // Hash includes version prefix (NAV_HASH_VERSION) to:
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
    // embed_model here is actually the full cache name (e.g., "navigate-nomic-embed-text")
    // passed by the caller — don't wrap it again with cache_name()
    let embed_cache_name = embed_model.to_string();
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

/// Check if a label is essentially the same as its parent group label.
fn label_matches_parent(label: &str, parent: &str) -> bool {
    let l = label.to_lowercase();
    let p = parent.to_lowercase();
    l == p || p.contains(&l) || l.contains(&p)
}

/// Deduplicate sibling cluster labels by REPLACING duplicates with disambiguators.
/// E.g., "scheduling" ×3 → "tests", "service", "repository" (not "scheduling (tests)")
fn deduplicate_sibling_labels(labels: &mut Vec<String>, clusters: &[(Vec<&FileInfo>, Option<String>)]) {
    let mut seen: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        seen.entry(label.to_lowercase()).or_default().push(i);
    }

    for (_key, indices) in &seen {
        if indices.len() <= 1 {
            continue;
        }
        // Collect disambiguators for all duplicates first
        let disambigs: Vec<(usize, Option<String>)> = indices
            .iter()
            .map(|&idx| {
                if idx < clusters.len() {
                    let (files, _) = &clusters[idx];
                    (idx, find_label_disambiguator(files))
                } else {
                    (idx, None)
                }
            })
            .collect();

        // Check if all disambiguators are the same (e.g., all "tests")
        let unique_disambigs: HashSet<String> = disambigs.iter()
            .filter_map(|(_, d)| d.clone())
            .collect();

        if unique_disambigs.len() >= 2 {
            // Good — disambiguators are different. REPLACE the label entirely.
            for (idx, disambig) in &disambigs {
                if let Some(d) = disambig {
                    labels[*idx] = d.clone();
                }
                // Leave None ones unchanged (they'll get #N in second pass)
            }
        }
        // If all disambiguators are identical or None, skip — second pass handles it
    }

    // Second pass: guarantee uniqueness with #N for any remaining duplicates
    let mut final_seen: HashMap<String, usize> = HashMap::new();
    for label in labels.iter_mut() {
        let key = label.to_lowercase();
        let count = final_seen.entry(key).or_insert(0);
        *count += 1;
        if *count > 1 {
            *label = format!("{} #{}", label, count);
        }
    }
}

/// Find a disambiguator for a cluster — checks for test files, architecture layers, etc.
fn find_label_disambiguator(files: &[&FileInfo]) -> Option<String> {
    // Test files
    let test_count = files.iter().filter(|f| {
        f.relative_path.contains(".test.") || f.relative_path.contains(".spec.")
            || f.relative_path.contains("/test/") || f.relative_path.contains("/tests/")
    }).count();
    if test_count > files.len() / 2 {
        return Some("tests".to_string());
    }

    // Architecture layers
    let layers = [
        ("service", "service"), ("repository", "repository"), ("repo", "repository"),
        ("delivery", "delivery"), ("domain", "domain"), ("http", "http"),
        ("temporal", "temporal"),
    ];
    let mut layer_counts: HashMap<&str, usize> = HashMap::new();
    for f in files {
        for (pattern, label) in &layers {
            if f.relative_path.contains(&format!("/{}/", pattern)) {
                *layer_counts.entry(label).or_default() += 1;
            }
        }
    }
    if let Some((layer, count)) = layer_counts.iter().max_by_key(|(_, c)| **c) {
        if *count > files.len() / 3 {
            return Some(layer.to_string());
        }
    }

    None
}

/// Produce a descriptive label for a group of files when all other heuristics fail.
///
/// Tries three strategies in order:
/// 1. Most common subdirectory after the shared prefix (e.g. "RecordSession")
/// 2. Dominant file-type description (e.g. "React components", "test files")
/// 3. File count with content type (e.g. "14 React components", "8 TypeScript modules")
fn describe_file_group(refs: &[&FileInfo]) -> String {
    if refs.is_empty() {
        return "empty cluster".to_string();
    }

    // Strategy 1: most common parent directory after common prefix
    let paths: Vec<Vec<&str>> = refs
        .iter()
        .map(|f| f.relative_path.split('/').collect::<Vec<_>>())
        .collect();
    let min_depth = paths.iter().map(|p| p.len()).min().unwrap_or(0);
    let mut common = 0;
    for d in 0..min_depth.saturating_sub(1) {
        if paths.iter().all(|p| p[d] == paths[0][d]) {
            common = d + 1;
        } else {
            break;
        }
    }
    if common < min_depth.saturating_sub(1) {
        let mut seg_counts: HashMap<&str, usize> = HashMap::new();
        for p in &paths {
            if common < p.len().saturating_sub(1) {
                *seg_counts.entry(p[common]).or_default() += 1;
            }
        }
        if let Some((seg, _)) = seg_counts.into_iter().max_by_key(|(_, c)| *c) {
            return seg.to_string();
        }
    }

    // Strategy 2 & 3: describe by dominant file type
    file_type_label(refs)
}

/// Classify a group of files by their dominant extension/suffix pattern.
///
/// Returns a descriptive string like "React components" or "8 test files".
fn file_type_label(refs: &[&FileInfo]) -> String {
    let n = refs.len();

    // Count files by type category
    let mut category_counts: HashMap<&str, usize> = HashMap::new();
    for f in refs {
        let path = &f.relative_path;
        let cat = if path.ends_with(".test.ts")
            || path.ends_with(".test.tsx")
            || path.ends_with(".spec.ts")
            || path.ends_with(".spec.tsx")
            || path.ends_with("_test.go")
            || path.ends_with("_test.rs")
        {
            "test files"
        } else if path.ends_with(".tsx") {
            "React components"
        } else if path.ends_with(".schema.ts") || path.ends_with(".schema.js") {
            "schemas"
        } else if path.ends_with(".proto") {
            "proto definitions"
        } else if path.ends_with(".sql") {
            "SQL migrations"
        } else if path.ends_with(".go") {
            "Go source"
        } else if path.ends_with(".rs") {
            "Rust source"
        } else if path.ends_with(".ts") || path.ends_with(".js") {
            "TypeScript modules"
        } else if path.ends_with(".json") {
            "JSON configs"
        } else if path.ends_with(".yml") || path.ends_with(".yaml") {
            "YAML configs"
        } else if path.ends_with(".css") || path.ends_with(".scss") {
            "stylesheets"
        } else {
            "files"
        };
        *category_counts.entry(cat).or_default() += 1;
    }

    // Find dominant category (>50% of files)
    if let Some((cat, count)) = category_counts.iter().max_by_key(|(_, c)| **c) {
        if *count > n / 2 {
            return format!("{} {}", n, cat);
        }
    }

    // No dominant type — just count with "files"
    format!("{} files", n)
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


/// Derive a human-readable label for a cluster from its file paths.
///
/// Maps path-based labels like "delivery/http" to more semantic descriptions
/// like "HTTP routes". Falls back to the original label if no mapping exists.
fn map_path_to_description(path_label: &str) -> &str {
    const LAYER_DESCRIPTIONS: &[(&[&str], &str)] = &[
        (&["delivery", "http"], "HTTP routes"),
        (&["delivery", "temporal"], "Temporal workflows"),
        (&["delivery", "nats"], "NATS consumers"),
        (&["repository", "pg"], "database queries"),
        (&["repository"], "data access"),
        (&["service"], "business logic"),
        (&["domain"], "domain models"),
        (&["delivery"], "API delivery"),
        (&["test", "integration"], "integration tests"),
        (&["tests", "integration"], "integration tests"),
    ];

    let segments: Vec<&str> = path_label.split('/').collect();

    // Try multi-segment matches first (longer patterns), then single-segment
    for (pattern, description) in LAYER_DESCRIPTIONS {
        if pattern.len() > segments.len() {
            continue;
        }
        if **pattern == segments[..pattern.len()] {
            return description;
        }
    }

    // No mapping found — return the original path label as-is
    path_label
}

/// Looks for the most common DISTINGUISHING directory segment — the first
/// segment after the common prefix where files diverge. This prevents
/// returning the parent directory name when all files share it.
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

    // ── Content-pattern heuristics for homogeneous domain clusters ──
    // These fire BEFORE the segment-based heuristic so that clusters where
    // all files share a deep common prefix still get meaningful labels.
    // We scan ALL directory segments (including common prefix) because the
    // architecture layer name may be part of the shared prefix itself.

    // 1. Test file detection: if >50% of files are test files, label as "tests"
    {
        let test_count = files
            .iter()
            .filter(|f| {
                let p = &f.relative_path;
                p.ends_with(".test.ts")
                    || p.ends_with(".test.tsx")
                    || p.ends_with(".spec.ts")
                    || p.ends_with(".spec.tsx")
                    || p.ends_with("_test.go")
                    || p.ends_with("_test.rs")
            })
            .count();
        if test_count > files.len() / 2 {
            return Some("tests".to_string());
        }
    }

    // 2. Architecture layer detection: check ALL path segments for well-known
    //    layer names, pick the most frequent one.
    {
        let layer_map: &[(&[&str], &str)] = &[
            (&["service"], "business logic"),
            (&["repository", "repo"], "data access"),
            (&["domain"], "domain models"),
            (&["test", "tests", "__tests__"], "tests"),
            (&["delivery"], "API delivery"),
        ];
        // Two-segment layers (checked first for specificity)
        let two_seg_map: &[(&str, &str, &str)] = &[
            ("delivery", "http", "HTTP routes"),
            ("delivery", "temporal", "Temporal workflows"),
            ("delivery", "nats", "NATS consumers"),
            ("repository", "pg", "database queries"),
            ("test", "integration", "integration tests"),
            ("tests", "integration", "integration tests"),
        ];

        let mut layer_counts: HashMap<&str, usize> = HashMap::new();

        for p in &paths {
            let dir_segs = &p[..p.len().saturating_sub(1)]; // all dir segments, exclude filename
            // Check two-segment layers first
            let mut matched = false;
            for (seg1, seg2, label) in two_seg_map {
                if dir_segs.len() >= 2
                    && dir_segs.iter().any(|s| s.eq_ignore_ascii_case(seg1))
                    && dir_segs.iter().any(|s| s.eq_ignore_ascii_case(seg2))
                {
                    *layer_counts.entry(label).or_default() += 1;
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }
            // Check single-segment layers
            for (keywords, label) in layer_map {
                if dir_segs
                    .iter()
                    .any(|s| keywords.iter().any(|k| s.eq_ignore_ascii_case(k)))
                {
                    *layer_counts.entry(label).or_default() += 1;
                    break;
                }
            }
        }

        if let Some((label, count)) = layer_counts.iter().max_by_key(|(_, c)| **c) {
            if *count > files.len() / 2 {
                return Some(label.to_string());
            }
        }
    }

    // 3. Common file extension grouping: if >50% share a distinctive extension
    {
        let mut ext_counts: HashMap<&str, usize> = HashMap::new();
        let mut index_count: usize = 0;
        for f in files {
            let fname = f.relative_path.split('/').next_back().unwrap_or("");
            if fname == "index.ts" || fname == "index.js" || fname == "index.tsx" || fname == "mod.rs" {
                index_count += 1;
            }
            if let Some(ext) = fname.rsplit('.').next() {
                *ext_counts.entry(ext).or_default() += 1;
            }
        }
        if index_count > files.len() / 2 {
            return Some("barrel exports".to_string());
        }
        if let Some((ext, count)) = ext_counts.iter().max_by_key(|(_, c)| **c) {
            if *count > files.len() / 2 {
                match *ext {
                    "proto" => return Some("proto definitions".to_string()),
                    "sql" => return Some("migrations".to_string()),
                    _ => {}
                }
            }
        }
    }

    // Look at the FIRST segment after the common prefix — that's where
    // this cluster differs from siblings. Find the most common value.
    if common_depth < min_depth.saturating_sub(1) {
        const GENERIC_SEGMENTS: &[&str] = &[
            "src", "lib", "dist", "build", "utils", "helpers", "common", "shared",
            "core", "types", "config", "internal", "cmd", "pkg",
        ];
        let mut seg_counts: HashMap<&str, usize> = HashMap::new();
        for p in &paths {
            if common_depth < p.len().saturating_sub(1) {
                *seg_counts.entry(p[common_depth]).or_default() += 1;
            }
        }
        if let Some((seg, count)) = seg_counts.iter().max_by_key(|(_, c)| **c) {
            if *count > files.len() / 3 && !GENERIC_SEGMENTS.contains(seg) {
                // If there's a second-level distinguisher too, use it
                let mut sub_counts: HashMap<&str, usize> = HashMap::new();
                for p in &paths {
                    if common_depth + 1 < p.len().saturating_sub(1) && p[common_depth] == *seg {
                        *sub_counts.entry(p[common_depth + 1]).or_default() += 1;
                    }
                }
                if let Some((sub_seg, sub_count)) = sub_counts.iter().max_by_key(|(_, c)| **c) {
                    if *sub_count > files.len() / 3 && !GENERIC_SEGMENTS.contains(sub_seg) {
                        let raw = format!("{}/{}", seg, sub_seg);
                        let descriptive = map_path_to_description(&raw);
                        return Some(descriptive.to_string());
                    }
                }
                return Some(map_path_to_description(seg).to_string());
            }
        }
    }

    // No distinguishing segment — try to find a common keyword from file names
    // E.g., files like "SoapTranscription.tsx", "SoapDiagnosis.tsx" → "Soap"
    // or "useSideBarActions.ts", "useSideBarBulk.ts" → "SideBar"
    {
        let filenames: Vec<&str> = files
            .iter()
            .filter_map(|f| f.relative_path.split('/').next_back())
            .filter_map(|name| name.split('.').next()) // strip extension
            .collect();
        if filenames.len() >= 2 {
            // Find the longest common prefix of filenames
            let first = filenames[0].as_bytes();
            let mut prefix_len = first.len();
            for name in &filenames[1..] {
                let bytes = name.as_bytes();
                prefix_len = prefix_len.min(bytes.len());
                for i in 0..prefix_len {
                    if first[i] != bytes[i] {
                        prefix_len = i;
                        break;
                    }
                }
            }
            // Use prefix if it's a meaningful word (3+ chars, not just "use" or "get")
            if prefix_len >= 4 {
                let prefix = &filenames[0][..prefix_len];
                // Trim to last uppercase boundary for camelCase
                if let Some(last_upper) = prefix.rfind(|c: char| c.is_uppercase()) {
                    if last_upper > 0 {
                        let trimmed = &prefix[..last_upper];
                        if trimmed.len() >= 4 {
                            return Some(trimmed.to_string());
                        }
                    }
                }
                // Only return full prefix if it's still >= 4 chars
                return Some(prefix.to_string());
            }
        }
    }

    // Last resort: use the most common header keyword
    // Only accept words that look like real English/code identifiers:
    // alphabetic, 4+ chars, no punctuation/special chars, not a stopword.
    {
        fn is_valid_label_word(word: &str) -> bool {
            word.len() >= 4
                && word.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_')
                && word.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)
                && !matches!(
                    word.to_lowercase().as_str(),
                    "from" | "with" | "this" | "that" | "have" | "been" | "will"
                    | "each" | "then" | "than" | "some" | "only" | "also" | "into"
                    | "more" | "most" | "such" | "used" | "uses" | "using"
                    | "file" | "files" | "type" | "types" | "data" | "code"
                    | "package" | "generated" | "module" | "index" | "export"
                    | "header" | "description" | "param" | "parameter" | "syntax"
                    | "import" | "const" | "function" | "interface" | "class"
                    | "async" | "await" | "return" | "string" | "number" | "boolean"
                    | "undefined" | "null" | "void" | "true" | "false"
                    | "todo" | "fixme" | "note" | "hack"
                )
        }

        let mut word_counts: HashMap<&str, usize> = HashMap::new();
        for f in files {
            for word in f.header.split_whitespace() {
                let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
                if is_valid_label_word(clean) {
                    *word_counts.entry(clean).or_default() += 1;
                }
            }
        }
        if let Some((word, count)) = word_counts.iter().max_by_key(|(_, c)| **c) {
            if *count > files.len() / 3 {
                return Some(word.to_string());
            }
        }
    }

    // Fall back to last 1-2 segments of common prefix
    if common_depth >= 2 {
        let label = format!("{}/{}", paths[0][common_depth - 2], paths[0][common_depth - 1]);
        let generic = ["packages", "apps", "src", "lib", "internal"];
        if !generic.contains(&label.as_str()) {
            return Some(label);
        }
    }

    None
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

    // Merge tiny groups (< 5 files) into an "other" bucket
    let mut result: Vec<(String, Vec<usize>)> = Vec::new();
    let mut other: Vec<usize> = Vec::new();

    for (label, indices) in groups {
        if indices.len() < MIN_DIR_GROUP_SIZE {
            other.extend(indices);
        } else {
            result.push((label, indices));
        }
    }

    // If "other" is too large, re-split it with a lower threshold
    if other.len() > 100 {
        // Re-group the "other" files by top-level directory with a lower threshold
        let mut sub_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for &idx in &other {
            let parts: Vec<&str> = files[idx].relative_path.split('/').collect();
            let label = if parts.len() >= 2 { parts[0].to_string() } else { "misc".to_string() };
            sub_groups.entry(label).or_default().push(idx);
        }
        let mut remaining_other: Vec<usize> = Vec::new();
        for (label, indices) in sub_groups {
            if indices.len() >= 3 {
                result.push((label, indices));
            } else {
                remaining_other.extend(indices);
            }
        }
        if !remaining_other.is_empty() {
            result.push(("other".to_string(), remaining_other));
        }
    } else if !other.is_empty() {
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
        assert_eq!(entry.hash, nav_content_hash("src/changed.rs", "content of src/changed.rs"));
    }

    // ── group_by_directory tests ─────────────────────────────────────

    #[test]
    fn group_by_directory_monorepo_domains() {
        let mut files = Vec::new();
        // Need >= MIN_DIR_GROUP_SIZE (5) files per group to avoid merging into "other"
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
        // Both are single-component paths → "root" label, but < MIN_DIR_GROUP_SIZE (5) → merged to "other"
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
        // Both groups have 3 files (< 5), so all should merge into "other"
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "other");
        assert_eq!(groups[0].1.len(), 6);
    }

    // ── derive_cluster_label tests ───────────────────────────────────

    #[test]
    fn derive_cluster_label_deep_common_prefix() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "packages/domains/billing/service.ts".into(), header: "Billing service".into(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/billing/repo.ts".into(), header: "Billing repository".into(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/billing/handler.ts".into(), header: "Billing handler".into(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/billing/types.ts".into(), header: "Billing types".into(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/billing/index.ts".into(), header: "Billing exports".into(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert!(label.is_some(), "Expected Some label for billing files");
        // Should contain "Billing" (from header keyword) or "billing" (from path)
        let l = label.unwrap();
        assert!(l.to_lowercase().contains("billing"), "Expected 'billing' in label, got '{}'", l);
    }

    #[test]
    fn derive_cluster_label_generic_prefix_fallback() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "src/controllers/auth.ts".into(), header: "Auth controller".into(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "src/controllers/user.ts".into(), header: "User controller".into(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "src/controllers/billing.ts".into(), header: "Billing controller".into(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert!(label.is_some());
        let l = label.unwrap();
        // Should find "controllers" from path or "controller" from header keyword
        assert!(l.to_lowercase().contains("controller"), "Expected 'controller' in label, got '{}'", l);
    }

    #[test]
    fn derive_cluster_label_empty() {
        let label = derive_cluster_label(&[]);
        assert_eq!(label, None);
    }

    #[test]
    fn derive_cluster_label_no_common_prefix() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "alpha/foo.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "beta/bar.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "gamma/baz.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        // No common prefix, no dominant segment, no common filename prefix, no headers → None
        assert_eq!(label, None);
    }


    #[test]
    fn derive_cluster_label_test_files() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "packages/domains/scheduling/service/appointment.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/scheduling/service/availability.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/scheduling/domain/date-range.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("tests".to_string()));
    }

    #[test]
    fn derive_cluster_label_repository_layer() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "packages/domains/scheduling/repository/pg/appointment.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/scheduling/repository/pg/service.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/scheduling/repository/pg/availability.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("database queries".to_string()));
    }

    #[test]
    fn derive_cluster_label_service_layer() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "packages/domains/billing/service/invoice.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/billing/service/payment.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/billing/service/refund.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("business logic".to_string()));
    }

    #[test]
    fn derive_cluster_label_http_routes() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "packages/domains/billing/delivery/http/invoice-handler.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/billing/delivery/http/payment-handler.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/billing/delivery/http/routes.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("HTTP routes".to_string()));
    }

    #[test]
    fn derive_cluster_label_proto_files() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "packages/proto/billing/invoice.proto".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/proto/billing/payment.proto".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/proto/billing/refund.proto".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("proto definitions".to_string()));
    }

    #[test]
    fn derive_cluster_label_barrel_exports() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "packages/domains/billing/service/index.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/billing/repository/index.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/domains/billing/domain/index.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("barrel exports".to_string()));
    }

    #[test]
    fn derive_cluster_label_sql_migrations() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "packages/db/migrations/001_create_users.sql".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/db/migrations/002_create_orgs.sql".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/db/migrations/003_add_billing.sql".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("migrations".to_string()));
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

    #[test]
    fn deduplicate_labels_with_test_disambiguator() {
        let test_files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "pkg/service/auth.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "pkg/service/user.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "pkg/service/billing.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let impl_files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "pkg/service/auth.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "pkg/service/user.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let test_refs: Vec<&FileInfo> = test_files.iter().collect();
        let impl_refs: Vec<&FileInfo> = impl_files.iter().collect();
        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> = vec![
            (test_refs, None),
            (impl_refs, None),
        ];
        let mut labels = vec!["service".to_string(), "service".to_string()];
        deduplicate_sibling_labels(&mut labels, &clusters);
        assert_ne!(labels[0], labels[1], "Labels should be disambiguated");
        assert!(labels[0].contains("test") || labels[1].contains("test"), "One should mention tests: {:?}", labels);
    }

    #[test]
    fn deduplicate_labels_no_dups_unchanged() {
        let f1: Vec<FileInfo> = vec![make_test_file("a.ts")];
        let f2: Vec<FileInfo> = vec![make_test_file("b.ts")];
        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> = vec![
            (f1.iter().collect(), None),
            (f2.iter().collect(), None),
        ];
        let mut labels = vec!["alpha".to_string(), "beta".to_string()];
        deduplicate_sibling_labels(&mut labels, &clusters);
        assert_eq!(labels, vec!["alpha", "beta"]);
    }

    #[test]
    fn find_disambiguator_detects_tests() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "pkg/auth.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "pkg/user.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "pkg/billing.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = find_label_disambiguator(&refs);
        assert_eq!(result, Some("tests".to_string()));
    }

    #[test]
    fn find_disambiguator_returns_none_for_mixed() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "alpha/foo.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "beta/bar.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(find_label_disambiguator(&refs), None);
    }

    #[test]
    fn derive_cluster_label_filename_prefix_camelcase() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "app/components/SoapTranscription.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "app/components/SoapDiagnosis.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "app/components/SoapHistory.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert!(label.is_some());
        assert_eq!(label.unwrap(), "Soap");
    }

    #[test]
    fn derive_cluster_label_filename_prefix_too_short_after_trim() {
        // "useCallback" + "useContext" → common prefix "useC" → camelCase trim → "use" (3 chars, too short)
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "hooks/useCallback.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "hooks/useContext.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        // Should NOT return "use" (too short after camelCase trim)
        // Should either return "useC" (full prefix) or fall through to another heuristic
        if let Some(l) = &label {
            assert_ne!(l, "use", "Should not return 'use' (too short after camelCase trim)");
        }
    }

    #[test]
    fn deduplicate_labels_same_disambiguator_gets_numbered() {
        // Both clusters are test files, both get disambiguator "tests"
        // Should end up as different labels, not "tests (tests)" x2
        let test_files1: Vec<FileInfo> = vec![
            FileInfo { relative_path: "pkg/a.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "pkg/b.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let test_files2: Vec<FileInfo> = vec![
            FileInfo { relative_path: "pkg/c.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "pkg/d.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> = vec![
            (test_files1.iter().collect(), None),
            (test_files2.iter().collect(), None),
        ];
        let mut labels = vec!["tests".to_string(), "tests".to_string()];
        deduplicate_sibling_labels(&mut labels, &clusters);
        assert_ne!(labels[0], labels[1], "Labels should differ after dedup: {:?}", labels);
    }

    #[test]
    fn derive_cluster_label_header_blocklist() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "gen/a.ts".into(), header: "@generated by protobuf-ts".into(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "gen/b.ts".into(), header: "@generated by protobuf-ts".into(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "gen/c.ts".into(), header: "@generated by protobuf-ts".into(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        if let Some(l) = &label {
            assert!(!l.to_lowercase().contains("@generated"), "Should not use @generated as label: {}", l);
            assert!(l != "generated", "Should not use 'generated' as label");
        }
    }

    // --- group_by_directory tests ---

    #[test]
    fn group_by_directory_skips_apps_prefix() {
        // "apps/emr-api/src/app.ts" should group as "emr-api", not "apps"
        let mut files = Vec::new();
        for i in 0..20 {
            files.push(FileInfo {
                relative_path: format!("apps/emr-api/src/file{}.ts", i),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            });
        }
        let groups = group_by_directory(&files);
        let labels: Vec<&str> = groups.iter().map(|(l, _)| l.as_str()).collect();
        assert!(
            !labels.contains(&"apps"),
            "Should skip 'apps' prefix, got {:?}",
            labels
        );
    }

    // --- derive_cluster_label additional tests ---

    #[test]
    fn derive_cluster_label_temporal_workflows() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "pkg/delivery/temporal/workflows/appointment.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/delivery/temporal/workflows/membership.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/delivery/temporal/activities/node/scheduling.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert!(label.is_some());
        let l = label.unwrap().to_lowercase();
        assert!(
            l.contains("temporal") || l.contains("workflow"),
            "Expected temporal-related label, got '{}'",
            l
        );
    }

    #[test]
    fn derive_cluster_label_nats_consumers() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "pkg/delivery/nats/consumer.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/delivery/nats/event-schemas.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/delivery/nats/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert!(label.is_some());
        let l = label.unwrap().to_lowercase();
        assert!(
            l.contains("nats") || l.contains("consumer") || l.contains("event"),
            "Expected nats-related label, got '{}'",
            l
        );
    }

    // --- deduplicate_sibling_labels additional tests ---

    #[test]
    fn deduplicate_replaces_not_appends() {
        let svc_files: Vec<FileInfo> = (0..5)
            .map(|i| FileInfo {
                relative_path: format!("pkg/service/svc{}.ts", i),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            })
            .collect();
        let test_files: Vec<FileInfo> = (0..5)
            .map(|i| FileInfo {
                relative_path: format!("pkg/service/svc{}.test.ts", i),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            })
            .collect();
        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> = vec![
            (svc_files.iter().collect(), None),
            (test_files.iter().collect(), None),
        ];
        let mut labels = vec!["service".to_string(), "service".to_string()];
        deduplicate_sibling_labels(&mut labels, &clusters);
        // Should REPLACE with disambiguators, not append "(service)" or "(tests)"
        assert_ne!(labels[0], labels[1]);
        // One should be "tests" label (not "service (service)" or "service (tests)")
        assert!(
            labels.iter().any(|l| l == "tests" || l.contains("test")),
            "Expected 'tests' label: {:?}",
            labels
        );
    }
}
