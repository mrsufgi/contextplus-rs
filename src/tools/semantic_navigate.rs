// Semantic project navigator using spectral clustering and Ollama labeling.
// Browse codebase by meaning: embeds files, clusters vectors, generates labels.

use crate::cache::rkyv_store;
use crate::config::Config;
use crate::core::clustering::{spectral_cluster_with_min, ClusterResult};
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

/// A directory group with its spectral clustering results, pending LLM labeling.
struct PendingGroup {
    label: String,
    indices: Vec<usize>,
    cluster_results: Vec<ClusterResult>,
}

/// Check if a file path looks like a test file.
fn is_test_file(path: &str) -> bool {
    path.ends_with(".test.ts")
        || path.ends_with(".test.tsx")
        || path.ends_with(".spec.ts")
        || path.ends_with(".spec.tsx")
        || path.ends_with("_test.go")
        || path.ends_with("_test.rs")
        || path.contains("/test/")
        || path.contains("/tests/")
        || path.contains("/__tests__/")
}

/// Classify a file path into a type category string.
fn classify_file_type(path: &str) -> &'static str {
    if is_test_file(path) {
        return "tests";
    }
    if path.ends_with(".tsx") || path.ends_with(".jsx") {
        return "components";
    }
    if path.ends_with(".proto") {
        return "proto";
    }
    if path.ends_with(".sql") {
        return "sql";
    }
    if path.ends_with(".go") {
        return "go";
    }
    if path.ends_with(".rs") {
        return "rust";
    }
    if path.ends_with(".json") {
        return "json";
    }
    if path.ends_with(".yml") || path.ends_with(".yaml") {
        return "yaml";
    }
    if path.ends_with(".css") || path.ends_with(".scss") {
        return "css";
    }
    if path.ends_with(".schema.ts") || path.ends_with(".schema.js") {
        return "schemas";
    }
    "source" // generic for .ts/.js and others
}

/// Map a file type category to a human-readable display label.
/// Returns `None` for categories too generic to be useful.
fn file_type_display(category: &str) -> Option<&'static str> {
    match category {
        "tests" => Some("test files"),
        "components" => Some("React components"),
        "proto" => Some("proto definitions"),
        "sql" => Some("SQL migrations"),
        "go" => Some("Go source"),
        "rust" => Some("Rust source"),
        "json" => Some("JSON configs"),
        "yaml" => Some("YAML configs"),
        "css" => Some("stylesheets"),
        "schemas" => Some("schemas"),
        "source" => None, // too generic for TS-dominant projects
        _ => None,
    }
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

    // Cluster directory groups, label sub-clusters with LLM, and build the tree.
    let mut children = build_labeled_tree(
        &files,
        &vectors,
        dir_groups,
        max_clusters,
        max_depth,
        ollama,
    )
    .await;

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

/// Cluster directory groups, label sub-clusters with LLM, and build the tree.
async fn build_labeled_tree(
    files: &[FileInfo],
    vectors: &[Vec<f32>],
    dir_groups: Vec<(String, Vec<usize>)>,
    max_clusters: usize,
    max_depth: usize,
    ollama: &OllamaClient,
) -> Vec<ClusterNode> {
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
    let pending_groups: Vec<PendingGroup> = large_groups
        .into_iter()
        .zip(all_cluster_results)
        .map(|((label, indices), clusters)| PendingGroup {
            label,
            indices,
            cluster_results: clusters,
        })
        .collect();

    // Get LLM labels for sub-clusters
    let llm_label_map = label_subclusters_with_llm(&pending_groups, files, ollama, 10).await;

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

    children
}

/// Build LLM prompt and get labels for sub-clusters.
async fn label_subclusters_with_llm(
    pending_groups: &[PendingGroup],
    files: &[FileInfo],
    ollama: &OllamaClient,
    max_llm_clusters: usize,
) -> HashMap<(usize, usize), String> {
    let mut llm_label_map: HashMap<(usize, usize), String> = HashMap::new();

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

    // Sort by file count descending — label the biggest clusters with LLM first
    all_sublabels.sort_by(|a, b| b.2.len().cmp(&a.2.len()));
    let llm_batch = if all_sublabels.len() > max_llm_clusters {
        &all_sublabels[..max_llm_clusters]
    } else {
        &all_sublabels
    };

    tracing::info!(
        total_sublabels = all_sublabels.len(),
        llm_batch_size = llm_batch.len(),
        max_llm_clusters = max_llm_clusters,
        "semantic_navigate: sub-cluster LLM batching — {} of {} clusters will get LLM labels",
        llm_batch.len(),
        all_sublabels.len()
    );
    for (idx, (gi, ci, file_refs)) in all_sublabels.iter().enumerate() {
        let in_batch = idx < llm_batch.len();
        tracing::info!(
            group_idx = gi,
            cluster_idx = ci,
            file_count = file_refs.len(),
            in_llm_batch = in_batch,
            "semantic_navigate: sub-cluster [{}, {}] has {} files, in_llm_batch={}",
            gi, ci, file_refs.len(), in_batch
        );
    }

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
                    "Cluster {} (TOTAL: {} files, within \"{}\"):\n  Subdirectory distribution (label should reflect the largest groups): {}\n  Sample files:\n  {}",
                    desc_idx + 1, file_refs.len(), parent_label, subdir_summary, file_list
                )
            })
            .collect();

        let prompt = format!(
            "Label each cluster based on the MAJORITY of its files, using the subdirectory distribution as the primary signal.\n\
            Do NOT name a cluster after a single file or minority feature — name it after what MOST files do.\n\
            A cluster with 80 component files and 1 validation file should be \"UI Components\", not \"Validation\".\n\
            A cluster with 15 scheduling pages and 1 microphone util should be \"Scheduling Pages\", not \"Microphone Utils\".\n\
            Look at the file counts per subdirectory — the subdirectory with the MOST files determines the label.\n\
            Return EXACTLY 2-5 words per label. Return ONLY a JSON array of strings, one per cluster.\n\n{}\n\nJSON array of {} strings:",
            descriptions.join("\n\n"),
            llm_batch.len()
        );

        if let Ok(response) = ollama.chat(&prompt).await {
            if let Some(json_str) = extract_json_array(&response) {
                if let Ok(labels) = serde_json::from_str::<Vec<String>>(&json_str) {
                    tracing::info!(
                        label_count = labels.len(),
                        batch_size = llm_batch.len(),
                        "semantic_navigate: LLM returned {} labels for {} clusters",
                        labels.len(), llm_batch.len()
                    );
                    for (j, (gi, ci, file_refs)) in llm_batch.iter().enumerate() {
                        if let Some(label) = labels.get(j) {
                            let clean_label = label.trim();
                            // Post-processing: reject garbage labels
                            if clean_label.is_empty()
                                || clean_label.len() > 50
                                || clean_label.contains('.')
                                || clean_label.contains('/')
                            {
                                tracing::info!(
                                    group_idx = gi,
                                    cluster_idx = ci,
                                    label = clean_label,
                                    "semantic_navigate: rejected LLM label (garbage format) for [{}, {}]: {:?}",
                                    gi, ci, clean_label
                                );
                                continue;
                            }
                            // Post-processing: reject labels that name a minority feature.
                            // If the label words match a specific subdirectory that holds <20%
                            // of the cluster's files, the LLM was misled by a prominent filename.
                            if !validate_label_against_cluster(clean_label, file_refs) {
                                tracing::info!(
                                    group_idx = gi,
                                    cluster_idx = ci,
                                    label = clean_label,
                                    file_count = file_refs.len(),
                                    "semantic_navigate: rejected LLM label (validation failed) for [{}, {}]: {:?} ({} files)",
                                    gi, ci, clean_label, file_refs.len()
                                );
                                continue;
                            }
                            tracing::info!(
                                group_idx = gi,
                                cluster_idx = ci,
                                label = clean_label,
                                file_count = file_refs.len(),
                                "semantic_navigate: accepted LLM label for [{}, {}]: {:?} ({} files)",
                                gi, ci, clean_label, file_refs.len()
                            );
                            llm_label_map.insert((*gi, *ci), clean_label.to_string());
                        }
                    }
                }
            }
        } else {
            tracing::info!("semantic_navigate: LLM chat call failed for sub-cluster labeling");
        }
    }

    llm_label_map
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
    if l == p { return true; }
    // Parent contains label as a complete segment
    if p.split('/').any(|seg| seg == l) { return true; }
    // Label equals a segment of the parent path
    if l.split('/').any(|seg| p.split('/').any(|ps| ps == seg)) { return true; }
    false
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

    // Second pass: for remaining duplicates, find what makes each unique
    let mut final_seen: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        final_seen.entry(label.to_lowercase()).or_default().push(i);
    }
    for (_key, indices) in &final_seen {
        if indices.len() <= 1 {
            continue;
        }

        // Compute a distinguishing suffix for each duplicate cluster
        let suffixes: Vec<(usize, String)> = indices
            .iter()
            .filter_map(|&idx| {
                if idx >= clusters.len() {
                    return None;
                }
                let (files, _) = &clusters[idx];
                let suffix = describe_cluster_uniqueness(files, clusters, indices, idx);
                Some((idx, suffix))
            })
            .collect();

        // Check if the computed suffixes are actually unique across siblings
        let unique_suffixes: HashSet<&str> = suffixes.iter().map(|(_, s)| s.as_str()).collect();
        if unique_suffixes.len() == suffixes.len() {
            // All suffixes are distinct — use them
            for (idx, suffix) in &suffixes {
                let base = labels[*idx].split(" #").next().unwrap_or(&labels[*idx]).to_string();
                // Skip redundant "(N files)" if label already starts with a count
                let label_has_count = base.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false);
                if label_has_count && suffix.contains("files") {
                    continue;
                }
                labels[*idx] = format!("{} ({})", base, suffix);
            }
        } else {
            // Suffixes collided — fall back to numbering
            let mut counter = 0usize;
            for &idx in indices {
                counter += 1;
                if counter > 1 {
                    labels[idx] = format!("{} #{}", labels[idx], counter);
                }
            }
        }
    }
}

/// Describe what makes THIS cluster unique compared to its duplicate siblings.
///
/// Tries in order:
/// 1. Most common directory segment unique to this cluster (not shared by siblings)
/// 2. Dominant file extension/type if different from siblings
/// 3. Dominant API domain from generated filenames (e.g., "scheduling API types")
/// 4. File count as last resort
fn describe_cluster_uniqueness(
    files: &[&FileInfo],
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    sibling_indices: &[usize],
    my_idx: usize,
) -> String {
    if let Some(dir) = find_unique_dominant_directory(files, clusters, sibling_indices, my_idx) {
        return dir;
    }
    if let Some(ftype) = find_unique_file_type(files, clusters, sibling_indices, my_idx) {
        return ftype;
    }
    if let Some(domain) = find_unique_api_domain(files, clusters, sibling_indices, my_idx) {
        return domain;
    }
    format!("{} files", files.len())
}

/// Find the most common directory segment in this cluster that other sibling clusters don't share.
fn find_unique_dominant_directory(
    files: &[&FileInfo],
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    sibling_indices: &[usize],
    my_idx: usize,
) -> Option<String> {
    let my_segments = count_directory_segments(files);

    let mut other_segments: HashMap<String, usize> = HashMap::new();
    for &idx in sibling_indices {
        if idx == my_idx || idx >= clusters.len() {
            continue;
        }
        let (other_files, _) = &clusters[idx];
        for (seg, count) in count_directory_segments(other_files) {
            *other_segments.entry(seg).or_default() += count;
        }
    }

    let mut best: Option<(String, usize)> = None;
    for (seg, count) in &my_segments {
        let other_count = other_segments.get(seg).copied().unwrap_or(0);
        if *count > files.len() / 3 && other_count == 0 {
            if best.as_ref().map_or(true, |(_, bc)| count > bc) {
                best = Some((seg.clone(), *count));
            }
        }
    }

    best.map(|(seg, _)| seg)
}

/// Count occurrences of each meaningful directory segment in file paths.
fn count_directory_segments(files: &[&FileInfo]) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for f in files {
        for segment in f.relative_path.split('/') {
            if segment.contains('.') || segment.len() < 2 {
                continue;
            }
            *counts.entry(segment.to_string()).or_default() += 1;
        }
    }
    counts
}

/// Find the dominant file type of this cluster if it differs from all siblings.
fn find_unique_file_type(
    files: &[&FileInfo],
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    sibling_indices: &[usize],
    my_idx: usize,
) -> Option<String> {
    let my_type = dominant_file_type(files)?;

    for &idx in sibling_indices {
        if idx == my_idx || idx >= clusters.len() {
            continue;
        }
        let (other_files, _) = &clusters[idx];
        if let Some(other_type) = dominant_file_type(other_files) {
            if other_type == my_type {
                return None;
            }
        }
    }

    Some(my_type.to_string())
}

/// Return the dominant file type category for a cluster, if >50% of files share it.
fn dominant_file_type(files: &[&FileInfo]) -> Option<&'static str> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for f in files {
        *counts
            .entry(classify_file_type(&f.relative_path))
            .or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .filter(|(_, c)| *c > files.len() / 2)
        .map(|(cat, _)| cat)
}

/// Extract the API domain from a generated type filename.
///
/// Matches patterns like `getApiV1SchedulingServices200.ts` → `"scheduling"`,
/// `postApiV1AuthLogin200.ts` → `"auth"`, `patchApiV1OrganizationsMembers.ts` → `"organizations"`.
fn extract_api_domain_from_filename(filename: &str) -> Option<String> {
    let lower = filename.to_lowercase();
    let pos = lower.find("apiv1")?;
    let after = &filename[pos + 5..];
    if after.is_empty() {
        return None;
    }
    // The domain name starts right after "ApiV1" in PascalCase.
    // Find where the domain word ends: the next uppercase letter after the
    // initial run. E.g., "SchedulingServices200.ts" → "Scheduling".
    let bytes = after.as_bytes();
    let mut end = 1; // skip first char (always uppercase start of domain)
    while end < bytes.len() {
        let c = bytes[end] as char;
        if c.is_uppercase() || c.is_ascii_digit() || c == '.' {
            break;
        }
        end += 1;
    }
    let domain = &after[..end];
    if domain.len() >= 3 {
        Some(domain.to_lowercase())
    } else {
        None
    }
}

/// Find the dominant API domain in a cluster's filenames (>50% must share it).
fn dominant_api_domain(files: &[&FileInfo]) -> Option<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for f in files {
        let fname = f.relative_path.split('/').next_back().unwrap_or("");
        if let Some(domain) = extract_api_domain_from_filename(fname) {
            *counts.entry(domain).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .filter(|(_, c)| *c > files.len() / 2)
        .map(|(domain, _)| format!("{} API types", domain))
}

/// Find a unique API domain label for this cluster that siblings don't share.
fn find_unique_api_domain(
    files: &[&FileInfo],
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    sibling_indices: &[usize],
    my_idx: usize,
) -> Option<String> {
    let my_domain = dominant_api_domain(files)?;

    for &idx in sibling_indices {
        if idx == my_idx || idx >= clusters.len() {
            continue;
        }
        let (other_files, _) = &clusters[idx];
        if let Some(other_domain) = dominant_api_domain(other_files) {
            if other_domain == my_domain {
                return None;
            }
        }
    }

    Some(my_domain)
}

/// Find a disambiguator for a cluster — checks for test files, architecture layers, etc.
fn find_label_disambiguator(files: &[&FileInfo]) -> Option<String> {
    // Test files
    let test_count = files
        .iter()
        .filter(|f| is_test_file(&f.relative_path))
        .count();
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

    // Strategy 1: most common parent directory after common prefix (skip generic segments)
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
            if !GENERIC_SEGMENTS.contains(&seg) && !is_nextjs_route_param(seg) {
                return seg.to_string();
            }
        }
    }

    // Strategy 1.5: If files span multiple subdirectories, list the top 2-3
    {
        let mut subdir_counts: HashMap<&str, usize> = HashMap::new();
        for p in &paths {
            if common < p.len().saturating_sub(1) {
                *subdir_counts.entry(p[common]).or_default() += 1;
            }
        }
        if subdir_counts.len() >= 2 {
            let mut sorted: Vec<_> = subdir_counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<&str> = sorted.iter().take(3).map(|(name, _)| *name).filter(|n| !is_nextjs_route_param(n)).collect();
            if !top.is_empty() {
                return top.join(" + ");
            }
        }
    }

    // Strategy 2: describe by dominant file type (skips overly generic labels)
    if let Some(label) = file_type_label(refs) {
        return label;
    }

    // Strategy 3: fallback — use directory name (even generic) with count, never bare "N files"
    let n = refs.len();
    if common < min_depth.saturating_sub(1) {
        let mut seg_counts: HashMap<&str, usize> = HashMap::new();
        for p in &paths {
            if common < p.len().saturating_sub(1) {
                *seg_counts.entry(p[common]).or_default() += 1;
            }
        }
        if let Some((seg, _)) = seg_counts.into_iter().max_by_key(|(_, c)| *c) {
            if !is_nextjs_route_param(seg) {
                return format!("{} {} modules", n, seg);
            }
        }
    }

    format!("{} source files", n)
}

/// Classify a group of files by their dominant extension/suffix pattern.
///
/// Returns `None` for overly generic labels (e.g. "TypeScript modules" in a TS project).
fn file_type_label(refs: &[&FileInfo]) -> Option<String> {
    dominant_file_type(refs)
        .and_then(file_type_display)
        .map(|display| format!("{} {}", refs.len(), display))
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

/// Validate that an LLM-generated label actually represents the cluster content.
///
/// Returns `false` if the label appears to name a minority feature — e.g. when
/// one prominent file skews the LLM into labeling an 81-file cluster after a
/// single file's concern. We check by tokenizing the label into words and seeing
/// if those words appear predominantly in only a small fraction of the cluster's
/// file paths. If the label words match a specific subdirectory holding <20% of
/// files (and don't match the majority), the label is rejected.
fn validate_label_against_cluster(label: &str, file_refs: &[&FileInfo]) -> bool {
    if file_refs.len() < 5 {
        // Too small to have a mislabeling problem
        return true;
    }

    // Generic label terms describe code nature, not specific subdirectories.
    // These words commonly appear in LLM labels but rarely in file paths, causing
    // false rejections when a few paths happen to contain "components/" or "api/".
    const LABEL_ALLOWLIST: &[&str] = &[
        "react", "vue", "angular", "svelte", "next", "nuxt",
        "node", "express", "fastify", "nest",
        "components", "hooks", "pages", "layouts", "views", "widgets",
        "api", "rest", "graphql", "grpc", "proto",
        "template", "templates", "config", "configs", "configuration",
        "server", "client", "frontend", "backend",
        "typescript", "javascript", "python", "golang", "rust",
        "source", "modules", "packages", "library",
        "form", "forms", "modal", "modals", "dialog", "dialogs",
        "page", "route", "routes", "routing", "navigation",
        "auth", "authentication",
        "state", "store", "redux", "zustand", "context",
        "style", "styles", "styled", "css", "scss",
        "test", "tests", "spec", "specs", "testing",
        "util", "utils", "utility", "utilities", "helper", "helpers",
        "service", "services", "handler", "handlers",
        "model", "models", "entity", "entities", "schema", "schemas",
        "feature", "features", "domain", "domains",
        "shared", "common", "core", "base", "internal",
        "dashboard", "admin", "portal", "management",
        "workflow", "workflows",
    ];

    let label_lower = label.to_lowercase();
    let label_words: Vec<&str> = label_lower
        .split_whitespace()
        .filter(|w| w.len() > 2) // skip short words like "of", "and"
        .collect();

    if label_words.is_empty() {
        return true;
    }

    // Filter out allowlisted terms — only use path-specific words for validation
    let path_specific_words: Vec<&&str> = label_words
        .iter()
        .filter(|w| !LABEL_ALLOWLIST.contains(w))
        .collect();

    // If ALL label words are allowlisted terms (e.g. "React Form Components"),
    // the label is conceptual, not path-derived — always accept it.
    if path_specific_words.is_empty() {
        tracing::info!(
            label = label,
            "semantic_navigate: validate_label — all words are allowlisted terms, accepting: {:?}",
            label
        );
        return true;
    }

    // Count how many files have paths matching any PATH-SPECIFIC label word
    let matching_files = file_refs
        .iter()
        .filter(|f| {
            let path_lower = f.relative_path.to_lowercase();
            path_specific_words.iter().any(|w| path_lower.contains(**w))
        })
        .count();

    let match_ratio = matching_files as f64 / file_refs.len() as f64;

    // If fewer than 20% of files match the path-specific label words,
    // the LLM likely named the cluster after a minority feature.
    // Exception: labels with zero matches are conceptual (not path-derived) — accept them.
    if matching_files > 0 && match_ratio < 0.20 {
        tracing::info!(
            label = label,
            matching_files = matching_files,
            total_files = file_refs.len(),
            match_ratio = format!("{:.2}", match_ratio).as_str(),
            path_specific_words = format!("{:?}", path_specific_words).as_str(),
            "semantic_navigate: validate_label — rejecting {:?} (match_ratio={:.2}, path_words={:?})",
            label, match_ratio, path_specific_words
        );
        return false;
    }

    tracing::info!(
        label = label,
        matching_files = matching_files,
        total_files = file_refs.len(),
        match_ratio = format!("{:.2}", match_ratio).as_str(),
        "semantic_navigate: validate_label — accepting {:?} (match_ratio={:.2}, {} of {} files)",
        label, match_ratio, matching_files, file_refs.len()
    );
    true
}

/// Returns true if a directory segment is a Next.js dynamic route parameter
/// (e.g., `[templateId]`, `[...nextauth]`, `[patientId]`, `[[...slug]]`).
fn is_nextjs_route_param(seg: &str) -> bool {
    seg.starts_with('[') || seg.ends_with(']')
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
            .filter(|f| is_test_file(&f.relative_path))
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
        let mut seg_counts: HashMap<&str, usize> = HashMap::new();
        for p in &paths {
            if common_depth < p.len().saturating_sub(1) {
                *seg_counts.entry(p[common_depth]).or_default() += 1;
            }
        }
        if let Some((seg, count)) = seg_counts.iter().max_by_key(|(_, c)| **c) {
            if *count > files.len() / 3 && !GENERIC_SEGMENTS.contains(seg) && !is_nextjs_route_param(seg) {
                // If there's a second-level distinguisher too, use it
                let mut sub_counts: HashMap<&str, usize> = HashMap::new();
                for p in &paths {
                    if common_depth + 1 < p.len().saturating_sub(1) && p[common_depth] == *seg {
                        *sub_counts.entry(p[common_depth + 1]).or_default() += 1;
                    }
                }
                if let Some((sub_seg, sub_count)) = sub_counts.iter().max_by_key(|(_, c)| **c) {
                    if *sub_count > files.len() / 3 && !GENERIC_SEGMENTS.contains(sub_seg) && !is_nextjs_route_param(sub_seg) {
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
                    // Generic code constants/markers that appear in file headers
                    | "section" | "default" | "config" | "options" | "settings"
                    | "props" | "state" | "action" | "reducer" | "store"
                    | "error" | "event" | "handler" | "callback" | "listener"
                    | "item" | "items" | "list" | "array" | "object"
                    | "value" | "result" | "response" | "request"
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
        let segs_clean = !is_nextjs_route_param(paths[0][common_depth - 2]) && !is_nextjs_route_param(paths[0][common_depth - 1]);
        if segs_clean && !generic.contains(&label.as_str()) {
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
        // Don't add "(N files)" if label already starts with a number (e.g., "6 source files")
        let label_starts_with_count = node.label.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false);
        if label_starts_with_count {
            result.push_str(&format!("{}[{}]\n", pad, node.label));
        } else {
            result.push_str(&format!("{}[{}] ({} files)\n", pad, node.label, count));
        }
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

    // --- describe_file_group tests ---

    #[test]
    fn describe_file_group_spanning_three_generic_subdirs() {
        // When all divergent segments are generic (src, lib, utils),
        // strategy 1 skips them and strategy 1.5 joins the top dirs with " + "
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "project/src/foo.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "project/lib/bar.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "project/utils/baz.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = describe_file_group(&refs);
        assert!(result.contains(" + "), "Expected 'A + B + C' format, got: {}", result);
    }

    #[test]
    fn describe_file_group_all_in_one_subdir_tsx() {
        // All files in same subdir, all .tsx -> strategy 2 (file_type_label) fires
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "app/components/Header.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "app/components/Footer.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "app/components/Sidebar.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = describe_file_group(&refs);
        assert!(result.contains("React components"), "Expected React components label, got: {}", result);
    }

    #[test]
    fn describe_file_group_single_non_generic_subdir() {
        // Files diverge at a non-generic segment -> strategy 1 returns that segment
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "app/billing/invoice.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "app/billing/payment.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "app/scheduling/calendar.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = describe_file_group(&refs);
        // Strategy 1 picks the most common non-generic segment ("billing" with 2 files)
        assert_eq!(result, "billing");
    }

    // --- file_type_label tests ---

    #[test]
    fn file_type_label_all_tsx_returns_react_components() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "app/Header.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "app/Footer.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "app/Nav.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = file_type_label(&refs);
        assert_eq!(result, Some("3 React components".to_string()));
    }

    #[test]
    fn file_type_label_all_test_ts_returns_test_files() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "src/auth.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "src/billing.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = file_type_label(&refs);
        assert_eq!(result, Some("2 test files".to_string()));
    }

    #[test]
    fn file_type_label_all_ts_returns_none() {
        // Plain .ts is "too_generic" -> should return None
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "src/auth.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "src/billing.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(file_type_label(&refs), None);
    }

    #[test]
    fn file_type_label_mixed_returns_none() {
        // Mixed types, no >50% dominant category
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "a.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "b.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "c.proto".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "d.sql".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(file_type_label(&refs), None);
    }

    // --- validate_label_against_cluster tests ---

    #[test]
    fn validate_label_all_technology_terms_accepted() {
        // All words are technology terms -> conceptual label, always accepted
        let files: Vec<FileInfo> = (0..10)
            .map(|i| FileInfo {
                relative_path: format!("pkg/module{}.ts", i),
                header: String::new(), content: String::new(), symbol_preview: vec![],
            })
            .collect();
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert!(validate_label_against_cluster("React Form Components", &refs));
    }

    #[test]
    fn validate_label_path_specific_word_below_20_pct_rejected() {
        // "zebra" is path-specific, matches only 1 of 10 files (<20%) -> rejected
        let mut files: Vec<FileInfo> = (0..9)
            .map(|i| FileInfo {
                relative_path: format!("pkg/module{}.ts", i),
                header: String::new(), content: String::new(), symbol_preview: vec![],
            })
            .collect();
        files.push(FileInfo {
            relative_path: "pkg/zebra/special.ts".into(),
            header: String::new(), content: String::new(), symbol_preview: vec![],
        });
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert!(!validate_label_against_cluster("Zebra Handler", &refs));
    }

    #[test]
    fn validate_label_conceptual_zero_matches_accepted() {
        // Path-specific word "analytics" matches 0 files -> conceptual, accepted
        let files: Vec<FileInfo> = (0..10)
            .map(|i| FileInfo {
                relative_path: format!("pkg/module{}.ts", i),
                header: String::new(), content: String::new(), symbol_preview: vec![],
            })
            .collect();
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert!(validate_label_against_cluster("Analytics Pipeline", &refs));
    }

    #[test]
    fn validate_label_small_cluster_always_true() {
        // Fewer than 5 files -> always accepted regardless of content
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "a.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "b.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert!(validate_label_against_cluster("Completely Wrong Label XYZ", &refs));
    }

    // --- is_nextjs_route_param tests ---

    #[test]
    fn is_nextjs_route_param_dynamic_segment() {
        assert!(is_nextjs_route_param("[templateId]"));
    }

    #[test]
    fn is_nextjs_route_param_catch_all() {
        assert!(is_nextjs_route_param("[...nextauth]"));
    }

    #[test]
    fn is_nextjs_route_param_regular_dir() {
        assert!(!is_nextjs_route_param("components"));
    }

    #[test]
    fn is_nextjs_route_param_filename() {
        assert!(!is_nextjs_route_param("index.ts"));
    }

    // --- label_matches_parent tests ---

    #[test]
    fn label_matches_parent_exact_match() {
        assert!(label_matches_parent("scheduling", "scheduling"));
    }

    #[test]
    fn label_matches_parent_parent_contains_label() {
        assert!(label_matches_parent("scheduling", "domains/scheduling"));
    }

    #[test]
    fn label_matches_parent_label_segment_matches_parent_segment() {
        // label "auth" matches parent "domains/auth" because label segment == parent segment
        assert!(label_matches_parent("auth", "domains/auth"));
    }

    #[test]
    fn label_matches_parent_no_match() {
        assert!(!label_matches_parent("billing", "scheduling"));
    }

    #[test]
    fn label_matches_parent_substring_not_segment_no_match() {
        // "auth" is a substring of "authentication" but NOT a segment match
        assert!(!label_matches_parent("Authentication Service", "auth"));
    }

    // --- map_path_to_description tests ---

    #[test]
    fn map_path_to_description_delivery_http() {
        assert_eq!(map_path_to_description("delivery/http"), "HTTP routes");
    }

    #[test]
    fn map_path_to_description_repository_pg() {
        assert_eq!(map_path_to_description("repository/pg"), "database queries");
    }

    #[test]
    fn map_path_to_description_service() {
        assert_eq!(map_path_to_description("service"), "business logic");
    }

    #[test]
    fn map_path_to_description_unknown_returns_as_is() {
        assert_eq!(map_path_to_description("unknown"), "unknown");
    }

    // --- dominant_file_type tests ---

    #[test]
    fn dominant_file_type_all_test_files() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "src/a.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "src/b.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "src/c.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(dominant_file_type(&refs).as_deref(), Some("tests"));
    }

    #[test]
    fn dominant_file_type_mixed_no_majority() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "a.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "b.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "c.go".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "d.rs".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(dominant_file_type(&refs), None);
    }

    #[test]
    fn dominant_file_type_all_tsx() {
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "Header.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "Footer.tsx".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(dominant_file_type(&refs).as_deref(), Some("components"));
    }

    // --- derive_cluster_label tests (exercises is_valid_label_word indirectly) ---

    #[test]
    fn derive_cluster_label_picks_architecture_layer() {
        // Files all in delivery/http -> should label as "HTTP routes"
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "pkg/scheduling/delivery/http/handler.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "pkg/scheduling/delivery/http/routes.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "pkg/scheduling/delivery/http/middleware.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = derive_cluster_label(&refs);
        assert_eq!(result, Some("HTTP routes".to_string()));
    }

    #[test]
    fn derive_cluster_label_picks_test_files() {
        // Majority test files -> should label as "tests"
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "pkg/auth/service/auth.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "pkg/auth/service/login.test.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "pkg/auth/service/token.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = derive_cluster_label(&refs);
        assert_eq!(result, Some("tests".to_string()));
    }

    #[test]
    fn derive_cluster_label_returns_none_for_shallow_paths() {
        // Files at depth 1 -> min_depth < 2, returns None
        let files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "main.rs".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(derive_cluster_label(&refs), None);
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

    // --- derive_cluster_label Next.js route param tests ---

    #[test]
    fn derive_cluster_label_skips_nextjs_route_params() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "app/templates/[templateId]/page.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/templates/[templateId]/layout.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        if let Some(l) = &label {
            assert!(
                !l.contains("templateId"),
                "Should not use Next.js route param as label: {}",
                l
            );
        }
    }

    #[test]
    fn derive_cluster_label_skips_patient_route_param() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "app/people/[patientId]/overview.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/people/[patientId]/history.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        if let Some(l) = &label {
            assert!(
                !l.contains("patientId"),
                "Should not use Next.js route param as label: {}",
                l
            );
        }
    }

    #[test]
    fn derive_cluster_label_skips_catch_all_route_param() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "app/api/auth/[...nextauth]/route.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/api/auth/[...nextauth]/config.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        if let Some(l) = &label {
            assert!(
                !l.contains("nextauth") && !l.contains("..."),
                "Should not use Next.js catch-all route param as label: {}",
                l
            );
        }
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

    // --- extract_api_domain_from_filename tests ---

    #[test]
    fn extract_api_domain_scheduling() {
        assert_eq!(
            extract_api_domain_from_filename("getApiV1SchedulingServices200.ts"),
            Some("scheduling".to_string())
        );
    }

    #[test]
    fn extract_api_domain_auth() {
        assert_eq!(
            extract_api_domain_from_filename("postApiV1AuthLogin200.ts"),
            Some("auth".to_string())
        );
    }

    #[test]
    fn extract_api_domain_organizations() {
        assert_eq!(
            extract_api_domain_from_filename("patchApiV1OrganizationsMembers.ts"),
            Some("organizations".to_string())
        );
    }

    #[test]
    fn extract_api_domain_no_match() {
        assert_eq!(extract_api_domain_from_filename("index.ts"), None);
        assert_eq!(extract_api_domain_from_filename("utils.ts"), None);
    }

    #[test]
    fn extract_api_domain_short_domain() {
        assert_eq!(extract_api_domain_from_filename("getApiV1AbTest.ts"), None);
    }

    // --- find_unique_api_domain / describe_cluster_uniqueness tests ---

    #[test]
    fn describe_cluster_uniqueness_api_domains() {
        let scheduling_files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "packages/libs/types/generated/getApiV1SchedulingServices200.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/libs/types/generated/getApiV1SchedulingAppointments200.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/libs/types/generated/postApiV1SchedulingSlots200.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let auth_files: Vec<FileInfo> = vec![
            FileInfo { relative_path: "packages/libs/types/generated/postApiV1AuthLogin200.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/libs/types/generated/postApiV1AuthRegister200.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "packages/libs/types/generated/getApiV1AuthSession200.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];

        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> = vec![
            (scheduling_files.iter().collect(), None),
            (auth_files.iter().collect(), None),
        ];
        let sibling_indices = vec![0usize, 1usize];

        let suffix0 = describe_cluster_uniqueness(
            &clusters[0].0, &clusters, &sibling_indices, 0,
        );
        let suffix1 = describe_cluster_uniqueness(
            &clusters[1].0, &clusters, &sibling_indices, 1,
        );

        assert_eq!(suffix0, "scheduling API types");
        assert_eq!(suffix1, "auth API types");
    }

    #[test]
    fn describe_cluster_uniqueness_same_api_domain_falls_back() {
        let files_a: Vec<FileInfo> = vec![
            FileInfo { relative_path: "gen/getApiV1SchedulingA.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "gen/getApiV1SchedulingB.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];
        let files_b: Vec<FileInfo> = vec![
            FileInfo { relative_path: "gen/postApiV1SchedulingC.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "gen/postApiV1SchedulingD.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
            FileInfo { relative_path: "gen/postApiV1SchedulingE.ts".into(), header: String::new(), content: String::new(), symbol_preview: vec![] },
        ];

        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> = vec![
            (files_a.iter().collect(), None),
            (files_b.iter().collect(), None),
        ];
        let sibling_indices = vec![0usize, 1usize];

        let suffix0 = describe_cluster_uniqueness(
            &clusters[0].0, &clusters, &sibling_indices, 0,
        );
        assert_eq!(suffix0, "2 files");
    }
}
