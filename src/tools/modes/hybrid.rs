use crate::core::clustering::{ClusterResult, spectral_cluster_with_min};
use crate::core::embeddings::OllamaClient;
use std::collections::HashMap;
use std::path::Path;

use super::super::labels::*;
use super::super::navigate_constants::*;
use super::super::semantic_navigate::{ClusterNode, FileInfo};
use super::ClusterParams;

/// A directory group with its spectral clustering results, pending LLM labeling.
pub(crate) struct PendingGroup {
    pub(crate) label: String,
    pub(crate) indices: Vec<usize>,
    pub(crate) cluster_results: Vec<ClusterResult>,
}

/// Cluster directory groups, label sub-clusters with LLM, and build the tree.
pub(crate) async fn build_labeled_tree(
    files: &[FileInfo],
    vectors: &[Vec<f32>],
    dir_groups: Vec<(String, Vec<usize>)>,
    params: &ClusterParams,
    ollama: &OllamaClient,
    root_dir: &Path,
) -> Vec<ClusterNode> {
    let mut children: Vec<ClusterNode> = Vec::new();
    let mut large_groups: Vec<(String, Vec<usize>)> = Vec::new();

    for (dir_label, group_indices) in &dir_groups {
        if group_indices.len() <= MAX_FILES_PER_LEAF {
            // Small group — flat leaf node. Spectral clustering on <10 files
            // produces noisy singletons with #2/#3 suffixes.
            children.push(ClusterNode {
                label: dir_label.clone(),
                files: group_indices.iter().map(|&i| files[i].clone()).collect(),
                children: Vec::new(),
            });
        } else {
            // 10+ files — spectral clustering produces meaningful sub-clusters
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
            let mc = params.max_clusters;
            let mn = params.min_clusters;
            async move {
                tokio::task::spawn_blocking(move || spectral_cluster_with_min(&local_vecs, mc, mn))
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
    let llm_label_map = label_subclusters_with_llm(&pending_groups, files, ollama, root_dir).await;

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
            let global_indices: Vec<usize> = cluster
                .indices
                .iter()
                .map(|&li| group.indices[li])
                .collect();
            let refs: Vec<&FileInfo> = global_indices.iter().map(|&i| &files[i]).collect();

            // Try multiple label sources in order of quality
            let raw_label = llm_label_map
                .get(&(gi, ci))
                .cloned()
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
                            let refs2: Vec<&FileInfo> =
                                global_indices.iter().map(|&i| &files[i]).collect();
                            describe_file_group(&refs2)
                        })
                }
            };

            // Depth 2: for large sub-clusters, do one more round of spectral clustering
            if global_indices.len() > MAX_FILES_PER_LEAF && params.max_depth > 2 {
                let sub_vecs: Vec<Vec<f32>> =
                    global_indices.iter().map(|&i| vectors[i].clone()).collect();
                let mc = params.max_clusters;
                let mn = params.min_clusters;
                let sub_results = tokio::task::spawn_blocking(move || {
                    spectral_cluster_with_min(&sub_vecs, mc, mn)
                })
                .await
                .unwrap_or_else(|_| vec![]);

                if sub_results.len() > 1 {
                    let mut depth2_children: Vec<ClusterNode> = Vec::new();
                    for sub_cluster in &sub_results {
                        let d2_indices: Vec<usize> = sub_cluster
                            .indices
                            .iter()
                            .map(|&li| global_indices[li])
                            .collect();
                        let refs: Vec<&FileInfo> = d2_indices.iter().map(|&i| &files[i]).collect();
                        let raw_d2 =
                            derive_cluster_label(&refs).or_else(|| find_label_disambiguator(&refs));
                        let d2_label = match raw_d2 {
                            Some(l)
                                if !label_matches_parent(&l, &label)
                                    && !label_matches_parent(&l, &group_label) =>
                            {
                                l
                            }
                            _ => find_label_disambiguator(&refs)
                                .filter(|d| {
                                    !label_matches_parent(d, &label)
                                        && !label_matches_parent(d, &group_label)
                                })
                                .unwrap_or_else(|| {
                                    // Last resort: smart fallback using directory + file type heuristics
                                    let refs2: Vec<&FileInfo> =
                                        d2_indices.iter().map(|&i| &files[i]).collect();
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
                    let mut d2_labels: Vec<String> =
                        depth2_children.iter().map(|c| c.label.clone()).collect();
                    let d2_input: Vec<(Vec<&FileInfo>, Option<String>)> = depth2_children
                        .iter()
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
        let d1_input: Vec<(Vec<&FileInfo>, Option<String>)> = sub_children
            .iter()
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
pub(crate) async fn label_subclusters_with_llm(
    pending_groups: &[PendingGroup],
    files: &[FileInfo],
    ollama: &OllamaClient,
    root_dir: &Path,
) -> HashMap<(usize, usize), String> {
    let mut llm_label_map: HashMap<(usize, usize), String> = HashMap::new();

    // Load existing label cache
    let label_cache = super::super::semantic_navigate::load_label_cache(root_dir);

    // Collect all sub-cluster file lists for one batched LLM call
    let mut all_sublabels: Vec<(usize, usize, Vec<&FileInfo>)> = Vec::new(); // (group_idx, cluster_idx, files)
    for (gi, group) in pending_groups.iter().enumerate() {
        if group.cluster_results.len() <= 1 {
            continue;
        }
        for (ci, cluster) in group.cluster_results.iter().enumerate() {
            let file_refs: Vec<&FileInfo> = cluster
                .indices
                .iter()
                .map(|&li| &files[group.indices[li]])
                .collect();
            all_sublabels.push((gi, ci, file_refs));
        }
    }

    // Check which clusters already have cached labels
    let mut cached_keys: HashMap<(usize, usize), String> = HashMap::new(); // (gi,ci) -> cache_key
    for (gi, ci, file_refs) in &all_sublabels {
        let paths: Vec<&str> = file_refs.iter().map(|f| f.relative_path.as_str()).collect();
        let key = super::super::semantic_navigate::cluster_cache_key(&paths);
        if let Some(label) = label_cache.get(&key) {
            tracing::info!(
                group_idx = gi,
                cluster_idx = ci,
                label = label.as_str(),
                "semantic_navigate: using cached label for [{}, {}]: {:?}",
                gi,
                ci,
                label
            );
            llm_label_map.insert((*gi, *ci), label.clone());
        }
        cached_keys.insert((*gi, *ci), key);
    }

    // Filter out cached sublabels — only send uncached to LLM
    let uncached_sublabels: Vec<&(usize, usize, Vec<&FileInfo>)> = all_sublabels
        .iter()
        .filter(|(gi, ci, _)| !llm_label_map.contains_key(&(*gi, *ci)))
        .collect();

    tracing::info!(
        total_sublabels = all_sublabels.len(),
        uncached_count = uncached_sublabels.len(),
        "semantic_navigate: sub-cluster LLM batching — {} uncached of {} total clusters",
        uncached_sublabels.len(),
        all_sublabels.len()
    );

    // Send ALL uncached clusters to LLM in batches to avoid timeout
    if !uncached_sublabels.is_empty() {
        for batch in uncached_sublabels.chunks(LLM_BATCH_SIZE) {
            tracing::info!(
                batch_size = batch.len(),
                "semantic_navigate: sending batch of {} clusters to LLM",
                batch.len()
            );

            let descriptions: Vec<String> = batch
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
                                    let sub: std::path::PathBuf = components[..depth].iter().collect();
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
                        file_refs.to_vec()
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
                    {
                        let letter = (b'A' + (desc_idx as u8 % 26)) as char;
                        format!(
                            "Group {} (TOTAL: {} files, within \"{}\"):\n  Subdirectory distribution (label should reflect the largest groups): {}\n  Sample files:\n  {}",
                            letter, file_refs.len(), parent_label, subdir_summary, file_list
                        )
                    }
                })
                .collect();

            let prompt = format!(
                "You are labeling clusters of source code files in a software project.\n\
                For each cluster, think about:\n\
                - What is the overarching THEME of these files? (not the directory name)\n\
                - What DISTINGUISHES this cluster from its siblings?\n\
                Give each cluster a descriptive label of 2-5 words that captures its PURPOSE.\n\n\
                Good labels: \"Appointment Core Logic\", \"Auth Middleware\", \"Patient Data Access\", \"Webhook Event Handlers\"\n\
                Bad labels: \"service\", \"delivery/http\", \"repository/pg\", \"files\", \"source\"\n\n\
                Do NOT echo directory names as labels — describe what the code DOES.\n\
                Do NOT name after a single file — name after the MAJORITY.\n\n\
                {}\n\n\
                Return ONLY a JSON array of {} strings, one per cluster.",
                descriptions.join("\n\n"),
                batch.len()
            );

            if let Ok(response) = ollama.chat(&prompt).await {
                if let Some(json_str) =
                    super::super::semantic_navigate::extract_json_array(&response)
                    && let Ok(labels) = serde_json::from_str::<Vec<String>>(&json_str)
                {
                    tracing::info!(
                        label_count = labels.len(),
                        batch_size = batch.len(),
                        "semantic_navigate: LLM returned {} labels for {} clusters",
                        labels.len(),
                        batch.len()
                    );
                    for (j, (gi, ci, file_refs)) in batch.iter().enumerate() {
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
                                    gi,
                                    ci,
                                    clean_label
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
                                    gi,
                                    ci,
                                    clean_label,
                                    file_refs.len()
                                );
                                continue;
                            }
                            tracing::info!(
                                group_idx = gi,
                                cluster_idx = ci,
                                label = clean_label,
                                file_count = file_refs.len(),
                                "semantic_navigate: accepted LLM label for [{}, {}]: {:?} ({} files)",
                                gi,
                                ci,
                                clean_label,
                                file_refs.len()
                            );
                            llm_label_map.insert((*gi, *ci), clean_label.to_string());
                        }
                    }
                }
            } else {
                tracing::info!(
                    "semantic_navigate: LLM chat call failed for sub-cluster labeling batch"
                );
            }
        }

        // Save newly generated LLM labels to the cache
        let mut updated_cache = label_cache;
        for (gi, ci, file_refs) in all_sublabels.iter() {
            if let Some(label) = llm_label_map.get(&(*gi, *ci)) {
                let paths: Vec<&str> = file_refs.iter().map(|f| f.relative_path.as_str()).collect();
                let key = super::super::semantic_navigate::cluster_cache_key(&paths);
                updated_cache.insert(key, label.clone());
            }
        }
        super::super::semantic_navigate::save_label_cache(root_dir, &updated_cache);
    }

    llm_label_map
}

/// Group files by meaningful directory structure for top-level clustering.
///
/// In a monorepo like `packages/domains/{billing,scheduling,...}`, groups by the
/// deepest "interesting" directory level. Generic prefixes like `packages/` are
/// skipped to find the actual domain boundary.
///
/// Returns `(label, indices)` pairs sorted by directory name.
pub(crate) fn group_by_directory(files: &[FileInfo]) -> Vec<(String, Vec<usize>)> {
    let generic_dirs: std::collections::HashSet<&str> =
        ["packages", "src", "lib", "apps", "internal", "cmd"]
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
            let label = if parts.len() >= 2 {
                parts[0].to_string()
            } else {
                "misc".to_string()
            };
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
            // Describe the "other" bucket by its most common subdirectories
            let mut subdir_counts: HashMap<&str, usize> = HashMap::new();
            for &idx in &remaining_other {
                let parts: Vec<&str> = files[idx].relative_path.split('/').collect();
                if parts.len() >= 2 {
                    *subdir_counts.entry(parts[0]).or_default() += 1;
                }
            }
            let mut sorted: Vec<_> = subdir_counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<&str> = sorted.iter().take(3).map(|(name, _)| *name).collect();
            let label = if top.is_empty() {
                "other".to_string()
            } else {
                top.join(" + ")
            };
            result.push((label, remaining_other));
        }
    } else if !other.is_empty() {
        // Describe the "other" bucket by its most common subdirectories
        let mut subdir_counts: HashMap<&str, usize> = HashMap::new();
        for &idx in &other {
            let parts: Vec<&str> = files[idx].relative_path.split('/').collect();
            if parts.len() >= 2 {
                *subdir_counts.entry(parts[0]).or_default() += 1;
            }
        }
        let mut sorted: Vec<_> = subdir_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let top: Vec<&str> = sorted.iter().take(3).map(|(name, _)| *name).collect();
        let label = if top.is_empty() {
            "other".to_string()
        } else {
            top.join(" + ")
        };
        result.push((label, other));
    }

    result.sort_by(|a, b| a.0.cmp(&b.0));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::core::clustering::ClusterResult;
    use crate::tools::semantic_navigate;

    fn config_with_host(host: &str) -> Config {
        let mut config = Config::from_env();
        config.ollama_host = host.to_string();
        config.ollama_chat_model = "test-chat-model".to_string();
        config
    }

    fn make_file(path: &str) -> FileInfo {
        FileInfo {
            relative_path: path.to_string(),
            header: format!("header for {path}"),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn label_subclusters_with_llm_uses_cached_labels_without_network() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let files = vec![
            make_file("src/auth/login.rs"),
            make_file("src/auth/session.rs"),
        ];
        let pending_groups = vec![PendingGroup {
            label: "auth".to_string(),
            indices: vec![0, 1],
            cluster_results: vec![
                ClusterResult { indices: vec![0] },
                ClusterResult { indices: vec![1] },
            ],
        }];

        let mut cache = HashMap::new();
        cache.insert(
            semantic_navigate::cluster_cache_key(&["src/auth/login.rs"]),
            "Login Flow".to_string(),
        );
        cache.insert(
            semantic_navigate::cluster_cache_key(&["src/auth/session.rs"]),
            "Session Management".to_string(),
        );
        semantic_navigate::save_label_cache(tempdir.path(), &cache);

        let client = OllamaClient::new(&config_with_host("http://127.0.0.1:9"));

        let labels =
            label_subclusters_with_llm(&pending_groups, &files, &client, tempdir.path()).await;

        assert_eq!(labels.get(&(0, 0)).map(String::as_str), Some("Login Flow"));
        assert_eq!(
            labels.get(&(0, 1)).map(String::as_str),
            Some("Session Management")
        );
    }

    #[tokio::test]
    async fn build_labeled_tree_keeps_small_directory_groups_flat() {
        let files = vec![
            make_file("src/auth/login.rs"),
            make_file("src/auth/session.rs"),
        ];
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let dir_groups = vec![("auth".to_string(), vec![0, 1])];
        let params = ClusterParams {
            max_clusters: 4,
            min_clusters: 2,
            max_depth: 3,
        };
        let client = OllamaClient::new(&config_with_host("http://127.0.0.1:9"));

        let tree = build_labeled_tree(
            &files,
            &vectors,
            dir_groups,
            &params,
            &client,
            std::path::Path::new("."),
        )
        .await;

        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].label, "auth");
        assert!(tree[0].children.is_empty());
        assert_eq!(tree[0].files.len(), 2);
    }
}
