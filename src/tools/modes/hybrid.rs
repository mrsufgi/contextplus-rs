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
    let mut pending_groups = Vec::new();

    for (dir_label, group_indices) in &dir_groups {
        if group_indices.len() <= MAX_FILES_PER_LEAF {
            // Small group — flat leaf node. Spectral clustering on <10 files
            // produces noisy singletons with #2/#3 suffixes.
            pending_groups.push(PendingGroup {
                label: dir_label.clone(),
                indices: group_indices.clone(),
                cluster_results: vec![ClusterResult {
                    indices: (0..group_indices.len()).collect(),
                }],
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
    pending_groups.extend(large_groups.into_iter().zip(all_cluster_results).map(
        |((label, indices), mut clusters)| {
            if clusters.is_empty() {
                clusters.push(ClusterResult {
                    indices: (0..indices.len()).collect(),
                });
            }
            PendingGroup {
                label,
                indices,
                cluster_results: clusters,
            }
        },
    ));

    // Get LLM labels for sub-clusters
    let llm_label_map = label_subclusters_with_llm(&pending_groups, files, ollama, root_dir).await;

    // Build the final tree using LLM labels where available, fallbacks otherwise
    for (gi, group) in pending_groups.into_iter().enumerate() {
        if group.cluster_results.len() <= 1 {
            // Single cluster or no split — leaf node
            children.push(ClusterNode {
                label: llm_label_map.get(&(gi, 0)).cloned().unwrap_or(group.label),
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

/// Owned snapshot of a sub-cluster, suitable for moving across `tokio::spawn`.
struct OwnedSubcluster {
    cache_key: String,
    parent_label: String,
    files: Vec<FileInfo>,
}

/// Resolve labels for sub-clusters with stale-while-revalidate semantics.
///
/// - cache hit: return cached label; requeue heuristic-quality entries.
/// - cache miss: return a heuristic now, persist it, queue an LLM heal.
///
/// Returns the (gi, ci) -> label_to_use_now map. The background task fires
/// off via `tokio::spawn` and survives the foreground response.
pub(crate) async fn label_subclusters_with_llm(
    pending_groups: &[PendingGroup],
    files: &[FileInfo],
    ollama: &OllamaClient,
    root_dir: &Path,
) -> HashMap<(usize, usize), String> {
    use super::super::labels::heuristic_label;
    use super::super::semantic_navigate::{
        CachedLabel, LabelQuality, cluster_cache_key, load_label_cache_async,
        save_label_cache_async,
    };

    let mut label_map: HashMap<(usize, usize), String> = HashMap::new();

    // Load existing label cache (with quality)
    let label_cache = load_label_cache_async(root_dir).await;

    // Collect all sub-cluster file lists
    let mut all_sublabels: Vec<(usize, usize, Vec<&FileInfo>)> = Vec::new(); // (group_idx, cluster_idx, files)
    for (gi, group) in pending_groups.iter().enumerate() {
        for (ci, cluster) in group.cluster_results.iter().enumerate() {
            let file_refs: Vec<&FileInfo> = cluster
                .indices
                .iter()
                .map(|&li| &files[group.indices[li]])
                .collect();
            all_sublabels.push((gi, ci, file_refs));
        }
    }

    // For every sub-cluster decide: cache hit, or (heuristic + heal queue).
    let mut heal_queue: Vec<OwnedSubcluster> = Vec::new();
    let mut heuristic_writes: HashMap<String, CachedLabel> = HashMap::new();

    for (gi, ci, file_refs) in &all_sublabels {
        let paths: Vec<&str> = file_refs.iter().map(|f| f.relative_path.as_str()).collect();
        let key = cluster_cache_key(&paths);
        if let Some(cached) = label_cache.get(&key) {
            tracing::info!(
                group_idx = gi,
                cluster_idx = ci,
                label = cached.label.as_str(),
                quality = ?cached.quality,
                "semantic_navigate: using cached label for [{}, {}]: {:?}",
                gi,
                ci,
                cached.label
            );
            label_map.insert((*gi, *ci), cached.label.clone());
            if cached.quality == LabelQuality::Llm {
                continue;
            }
        } else {
            let h = if pending_groups[*gi].cluster_results.len() == 1 {
                pending_groups[*gi].label.clone()
            } else {
                heuristic_label(file_refs)
            };
            label_map.insert((*gi, *ci), h.clone());
            heuristic_writes.insert(key.clone(), CachedLabel::heuristic(h));
        }
        heal_queue.push(OwnedSubcluster {
            cache_key: key,
            parent_label: pending_groups[*gi].label.clone(),
            files: file_refs.iter().map(|f| (*f).clone()).collect(),
        });
    }

    // Persist heuristics now so a crash leaves usable labels on disk.
    if !heuristic_writes.is_empty()
        && let Err(error) = save_label_cache_async(root_dir, &heuristic_writes).await
    {
        tracing::info!(reason = %error, "label cache save failed");
    }

    // Spawn the background LLM heal — caller does NOT await this.
    if !heal_queue.is_empty() {
        spawn_subcluster_heal(heal_queue, ollama.clone(), root_dir.to_path_buf());
    } else {
        tracing::info!("semantic_navigate: heal spawn skipped: no heuristic sub-clusters");
    }

    label_map
}

/// Run the LLM batch for a queue of sub-clusters and merge the upgraded
/// labels into the on-disk cache. Same prompt/parsing/validation as the
/// previous foreground path — moved here so the foreground returns instantly.
async fn run_subcluster_llm_heal(
    queue: &[OwnedSubcluster],
    ollama: &OllamaClient,
    root_dir: &Path,
) {
    use super::super::semantic_navigate::{CachedLabel, save_label_cache_async};
    if queue.is_empty() {
        return;
    }

    // Build (gi, ci, file_refs) tuples on the owned data. We synthesize indices
    // for batch logging; they don't have to match the foreground gi/ci.
    let all_sublabels: Vec<(usize, usize, Vec<&FileInfo>)> = queue
        .iter()
        .enumerate()
        .map(|(i, snap)| (i, 0usize, snap.files.iter().collect::<Vec<&FileInfo>>()))
        .collect();
    let parent_labels: Vec<&str> = queue.iter().map(|s| s.parent_label.as_str()).collect();

    let mut llm_label_map: HashMap<usize, String> = HashMap::new();

    for batch in all_sublabels.chunks(LLM_BATCH_SIZE) {
        tracing::info!(
            batch_size = batch.len(),
            "semantic_navigate: heal — sending batch of {} clusters to LLM",
            batch.len()
        );

        let descriptions: Vec<String> = batch
            .iter()
            .enumerate()
            .map(|(desc_idx, (queue_idx, _, file_refs))| {
                let parent_label = parent_labels[*queue_idx];

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
                                let sub: std::path::PathBuf =
                                    components[..depth].iter().collect();
                                Some(sub.to_string_lossy().to_string())
                            }
                        })
                        .unwrap_or_else(|| ".".to_string());
                    subdir_files.entry(subdir).or_default().push(f);
                }

                let mut subdir_counts: Vec<(String, usize)> = subdir_files
                    .iter()
                    .map(|(dir, files)| (dir.clone(), files.len()))
                    .collect();
                subdir_counts.sort_by_key(|b| std::cmp::Reverse(b.1));
                let subdir_summary = subdir_counts
                    .iter()
                    .map(|(dir, count)| format!("{} ({})", dir, count))
                    .collect::<Vec<_>>()
                    .join(", ");

                let sample: Vec<&FileInfo> = if file_refs.len() <= MAX_FILES_PER_LABEL {
                    file_refs.clone()
                } else {
                    let mut sorted_dirs: Vec<(&String, &Vec<&FileInfo>)> =
                        subdir_files.iter().collect();
                    sorted_dirs.sort_by_key(|b| std::cmp::Reverse(b.1.len()));

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
                        let desc = if f.header.is_empty() {
                            "no description"
                        } else {
                            &f.header
                        };
                        format!("{}: {}", f.relative_path, desc)
                    })
                    .collect::<Vec<_>>()
                    .join("\n  ");
                let letter = (b'A' + (desc_idx as u8 % 26)) as char;
                format!(
                    "Group {} (TOTAL: {} files, within \"{}\"):\n  Subdirectory distribution (label should reflect the largest groups): {}\n  Sample files:\n  {}",
                    letter, file_refs.len(), parent_label, subdir_summary, file_list
                )
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

        let response = match ollama.chat(&prompt).await {
            Ok(response) => response,
            Err(error) => {
                tracing::info!(
                    reason = %super::super::semantic_navigate::heal_chat_error_reason(&error),
                    "semantic_navigate: heal — LLM chat call failed for sub-cluster batch"
                );
                continue;
            }
        };
        tracing::debug!(
            bytes = response.len(),
            "semantic_navigate: heal chat result"
        );
        let labels =
            super::super::semantic_navigate::extract_label_array(&response, batch.len(), false);
        let Some(labels) = labels else {
            let response_shape = super::super::semantic_navigate::response_shape(&response);
            tracing::info!(
                response_shape,
                "semantic_navigate: heal parse failure: expected one label per cluster"
            );
            continue;
        };
        let mut rejections = Vec::new();
        for ((queue_idx, _, file_refs), label) in batch.iter().zip(&labels) {
            let label = label.trim();
            if label.is_empty() || label.len() > 50 || label.contains('.') || label.contains('/') {
                rejections.push(format!(
                    "invalid label format; total_files={}",
                    file_refs.len()
                ));
                continue;
            }
            if !validate_label_against_cluster(label, file_refs) {
                let (matching_files, total_files) =
                    super::super::labels::label_validation_rejection(label, file_refs)
                        .expect("rejected label has a validation reason");
                rejections.push(format!("minority path match: matching_files={matching_files}, total_files={total_files}"));
                continue;
            }
            llm_label_map.insert(*queue_idx, label.to_string());
        }
        if !rejections.is_empty() {
            tracing::info!(reasons = ?rejections, "semantic_navigate: heal validation rejected labels");
        }
    }

    // Merge upgraded labels into on-disk cache.
    let mut upgrades: HashMap<String, CachedLabel> = HashMap::new();
    for (queue_idx, snap) in queue.iter().enumerate() {
        if let Some(label) = llm_label_map.get(&queue_idx) {
            upgrades.insert(snap.cache_key.clone(), CachedLabel::llm(label.clone()));
        }
    }
    if !upgrades.is_empty() {
        if let Err(error) = save_label_cache_async(root_dir, &upgrades).await {
            tracing::info!(reason = %error, "semantic_navigate: heal cache publication failed (heuristics retained)");
            return;
        }
        tracing::info!(
            upgraded = upgrades.len(),
            queued = queue.len(),
            "semantic_navigate: heal — upgraded {} of {} sub-cluster labels",
            upgrades.len(),
            queue.len()
        );
    }
}

/// Spawn a `tokio::spawn` task that runs the LLM heal for sub-clusters.
/// Process-wide dedup ensures the same `cache_key` never has two heals in
/// flight; if every key in `queue` is already being healed, this is a no-op.
fn spawn_subcluster_heal(
    queue: Vec<OwnedSubcluster>,
    ollama: OllamaClient,
    root_dir: std::path::PathBuf,
) {
    use super::super::semantic_navigate::claim_heal_for_root;

    let keys: Vec<String> = queue.iter().map(|s| s.cache_key.clone()).collect();
    let (claimed, claims) = claim_heal_for_root(&keys, &root_dir);
    if claimed.is_empty() {
        tracing::info!("semantic_navigate: heal spawn skipped: all keys already in-flight");
        return;
    }
    let to_heal: Vec<OwnedSubcluster> = queue
        .into_iter()
        .filter(|s| claimed.contains(&s.cache_key))
        .collect();
    let claimed_keys: Vec<String> = to_heal.iter().map(|s| s.cache_key.clone()).collect();

    tracing::info!(claimed = claimed_keys.len(), root = %root_dir.display(), "semantic_navigate: heal spawn");
    tokio::spawn(async move {
        let _claims = claims;
        run_subcluster_llm_heal(&to_heal, &ollama, &root_dir).await;
        tracing::info!("semantic_navigate: heal completed");
    });
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
            sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
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
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
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
    use std::io::Write;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[derive(Clone)]
    struct CapturedWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for CapturedWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

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

    fn two_subclusters(files: &[FileInfo], parent: &str) -> Vec<PendingGroup> {
        assert_eq!(files.len(), 2);
        vec![PendingGroup {
            label: parent.to_string(),
            indices: vec![0, 1],
            cluster_results: vec![
                ClusterResult { indices: vec![0] },
                ClusterResult { indices: vec![1] },
            ],
        }]
    }

    async fn mock_chat(content: &str) -> (MockServer, OllamaClient) {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": { "content": content }
            })))
            .mount(&server)
            .await;
        let client = OllamaClient::new(&config_with_host(&server.uri()));
        (server, client)
    }

    fn seed_heuristics(root: &Path, files: &[FileInfo], labels: &[&str]) -> Vec<String> {
        use semantic_navigate::{CachedLabel, cluster_cache_key, save_label_cache_full};

        let keyed_labels: Vec<(String, CachedLabel)> = files
            .iter()
            .zip(labels)
            .map(|(file, label)| {
                (
                    cluster_cache_key(&[file.relative_path.as_str()]),
                    CachedLabel::heuristic((*label).to_string()),
                )
            })
            .collect();
        let keys = keyed_labels.iter().map(|(key, _)| key.clone()).collect();
        let entries = keyed_labels.into_iter().collect();
        save_label_cache_full(root, &entries);
        keys
    }

    async fn wait_for_llm_labels(root: &Path, expected: &[(String, &str)]) -> bool {
        tokio::time::timeout(Duration::from_millis(250), async {
            loop {
                let cache = semantic_navigate::load_label_cache_full(root);
                if expected.iter().all(|(key, label)| {
                    cache.get(key).is_some_and(|entry| {
                        entry.quality == semantic_navigate::LabelQuality::Llm
                            && entry.label == *label
                    })
                }) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .is_ok()
    }

    fn captured_info_logs() -> (Arc<Mutex<Vec<u8>>>, impl tracing::Subscriber) {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let writer_logs = Arc::clone(&logs);
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .without_time()
            .with_target(false)
            .with_writer(move || CapturedWriter(Arc::clone(&writer_logs)))
            .finish();
        (logs, subscriber)
    }

    fn logs_as_string(logs: &Arc<Mutex<Vec<u8>>>) -> String {
        String::from_utf8(logs.lock().unwrap().clone()).unwrap()
    }

    #[tokio::test]
    async fn cached_heuristics_are_healed_and_returned_by_the_next_clusters_call() {
        let tempdir = tempfile::tempdir().unwrap();
        let files = vec![
            make_file("packages/payments/repository/stripe.rs"),
            make_file("packages/payments/service/charge.rs"),
        ];
        let pending = two_subclusters(&files, "payments");
        let keys = seed_heuristics(tempdir.path(), &files, &["repository/stripe", "service"]);
        let (_server, client) =
            mock_chat(r#"["Stripe Payment Gateway", "Payment Orchestration"]"#).await;

        let first = label_subclusters_with_llm(&pending, &files, &client, tempdir.path()).await;
        assert_eq!(
            first.get(&(0, 0)).map(String::as_str),
            Some("repository/stripe")
        );
        assert_eq!(first.get(&(0, 1)).map(String::as_str), Some("service"));

        let healed = wait_for_llm_labels(
            tempdir.path(),
            &[
                (keys[0].clone(), "Stripe Payment Gateway"),
                (keys[1].clone(), "Payment Orchestration"),
            ],
        )
        .await;
        assert!(
            healed,
            "background heal did not replace cached heuristic labels before timeout"
        );

        let second = label_subclusters_with_llm(&pending, &files, &client, tempdir.path()).await;
        let second_labels: std::collections::HashSet<&str> =
            second.values().map(String::as_str).collect();
        assert_eq!(
            second_labels,
            std::collections::HashSet::from(["Stripe Payment Gateway", "Payment Orchestration"])
        );
    }

    #[tokio::test]
    async fn prose_wrapped_chat_response_uses_the_valid_label_array() {
        use semantic_navigate::{CachedLabel, LabelQuality, load_label_cache_full};

        let tempdir = tempfile::tempdir().unwrap();
        let file = make_file("packages/payments/service/authorize.rs");
        let key = semantic_navigate::cluster_cache_key(&[file.relative_path.as_str()]);
        semantic_navigate::save_label_cache_full(
            tempdir.path(),
            &HashMap::from([(key.clone(), CachedLabel::heuristic("service".to_string()))]),
        );
        let response = "I considered calling tools [] but no tool was needed.\n\
                        The requested labels are [\"Payment Authorization Flow\"].\n\
                        I hope that helps.";
        let (_server, client) = mock_chat(response).await;
        let queue = vec![OwnedSubcluster {
            cache_key: key.clone(),
            parent_label: "payments".to_string(),
            files: vec![file],
        }];

        run_subcluster_llm_heal(&queue, &client, tempdir.path()).await;

        let cache = load_label_cache_full(tempdir.path());
        assert_eq!(
            cache[&key].quality,
            LabelQuality::Llm,
            "prose-wrapped response did not produce an LLM cache upgrade"
        );
        assert_eq!(cache[&key].label, "Payment Authorization Flow");
    }

    #[tokio::test]
    async fn heal_scans_past_non_label_arrays_for_the_expected_string_array() {
        use semantic_navigate::{CachedLabel, LabelQuality, load_label_cache_full};

        let cases = [
            (
                "numeric-prefix",
                "I considered tool arguments [1, 2]. The labels are [\"Payment Authorization Flow\"].",
            ),
            (
                "object-prefix",
                "Tool metadata [{\"name\":\"search\"}]. The labels are [\"Payment Authorization Flow\"].",
            ),
            (
                "wrong-length-prefix",
                "Draft labels [\"scratch\", \"notes\"]. Final labels [\"Payment Authorization Flow\"].",
            ),
        ];
        let mut failures = Vec::new();

        for (case, response) in cases {
            let tempdir = tempfile::tempdir().unwrap();
            let file = make_file(&format!("packages/payments/service/{case}.rs"));
            let key = semantic_navigate::cluster_cache_key(&[file.relative_path.as_str()]);
            semantic_navigate::save_label_cache_full(
                tempdir.path(),
                &HashMap::from([(key.clone(), CachedLabel::heuristic("service".to_string()))]),
            );
            let (_server, client) = mock_chat(response).await;
            let queue = vec![OwnedSubcluster {
                cache_key: key.clone(),
                parent_label: "payments".to_string(),
                files: vec![file],
            }];

            run_subcluster_llm_heal(&queue, &client, tempdir.path()).await;

            let cache = load_label_cache_full(tempdir.path());
            if cache.get(&key).is_none_or(|entry| {
                entry.quality != LabelQuality::Llm || entry.label != "Payment Authorization Flow"
            }) {
                failures.push(case);
            }
        }

        assert!(
            failures.is_empty(),
            "heal stopped at unrelated arrays instead of scanning for the expected label array: {failures:?}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn validation_rejections_keep_heuristics_and_log_one_batch_reason() {
        use semantic_navigate::{CachedLabel, LabelQuality, load_label_cache_full};

        let tempdir = tempfile::tempdir().unwrap();
        let mut first_files: Vec<FileInfo> = (0..9)
            .map(|i| make_file(&format!("packages/payments/service/charge_{i}.rs")))
            .collect();
        first_files.push(make_file("packages/payments/zebra/handler.rs"));
        let mut second_files: Vec<FileInfo> = (0..9)
            .map(|i| make_file(&format!("packages/payments/repository/payment_{i}.rs")))
            .collect();
        second_files.push(make_file("packages/payments/falcon/handler.rs"));
        let first_refs: Vec<&str> = first_files
            .iter()
            .map(|file| file.relative_path.as_str())
            .collect();
        let second_refs: Vec<&str> = second_files
            .iter()
            .map(|file| file.relative_path.as_str())
            .collect();
        let first_key = semantic_navigate::cluster_cache_key(&first_refs);
        let second_key = semantic_navigate::cluster_cache_key(&second_refs);
        semantic_navigate::save_label_cache_full(
            tempdir.path(),
            &HashMap::from([
                (
                    first_key.clone(),
                    CachedLabel::heuristic("service".to_string()),
                ),
                (
                    second_key.clone(),
                    CachedLabel::heuristic("repository".to_string()),
                ),
            ]),
        );
        let queue = vec![
            OwnedSubcluster {
                cache_key: first_key.clone(),
                parent_label: "payments".to_string(),
                files: first_files,
            },
            OwnedSubcluster {
                cache_key: second_key.clone(),
                parent_label: "payments".to_string(),
                files: second_files,
            },
        ];
        let (_server, client) = mock_chat(r#"["Zebra Handler", "Falcon Handler"]"#).await;
        let (logs, subscriber) = captured_info_logs();
        let _guard = tracing::subscriber::set_default(subscriber);

        run_subcluster_llm_heal(&queue, &client, tempdir.path()).await;

        let cache = load_label_cache_full(tempdir.path());
        assert_eq!(cache[&first_key].quality, LabelQuality::Heuristic);
        assert_eq!(cache[&first_key].label, "service");
        assert_eq!(cache[&second_key].quality, LabelQuality::Heuristic);
        assert_eq!(cache[&second_key].label, "repository");
        let logs = logs_as_string(&logs);
        let rejection_lines = logs.lines().filter(|line| line.contains("reject")).count();
        assert_eq!(
            rejection_lines, 1,
            "validation rejection must be logged once per batch with a reason:\n{logs}"
        );
        assert!(
            logs.contains("matching_files") && logs.contains("total_files"),
            "validation rejection log omitted its reason:\n{logs}"
        );
    }

    #[tokio::test]
    async fn attached_worktree_ref_heal_is_read_by_the_next_call() {
        let tempdir = tempfile::tempdir().unwrap();
        let primary = tempdir.path().join("primary");
        let worktree = tempdir.path().join("feature-payments");
        let worktree_gitdir = primary.join(".git/worktrees/feature-payments");
        std::fs::create_dir_all(&worktree_gitdir).unwrap();
        std::fs::create_dir_all(&worktree).unwrap();
        std::fs::write(worktree_gitdir.join("commondir"), "../..").unwrap();
        std::fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", worktree_gitdir.display()),
        )
        .unwrap();
        let files = vec![
            make_file("packages/payments/repository/stripe.rs"),
            make_file("packages/payments/service/refund.rs"),
        ];
        let pending = two_subclusters(&files, "payments");
        let keys = seed_heuristics(&worktree, &files, &["repository/stripe", "service"]);
        let (_server, client) = mock_chat(r#"["Stripe Data Access", "Refund Processing"]"#).await;

        let first = label_subclusters_with_llm(&pending, &files, &client, &worktree).await;
        assert!(first.values().any(|label| label == "repository/stripe"));
        let healed = wait_for_llm_labels(
            &worktree,
            &[
                (keys[0].clone(), "Stripe Data Access"),
                (keys[1].clone(), "Refund Processing"),
            ],
        )
        .await;
        assert!(
            healed,
            "attached worktree heal was not readable from the worktree label cache"
        );

        let second = label_subclusters_with_llm(&pending, &files, &client, &worktree).await;
        let labels: std::collections::HashSet<&str> = second.values().map(String::as_str).collect();
        assert_eq!(
            labels,
            std::collections::HashSet::from(["Stripe Data Access", "Refund Processing"])
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn chat_error_is_logged_once_with_reason_and_without_secrets() {
        use semantic_navigate::{CachedLabel, load_label_cache_full};

        let tempdir = tempfile::tempdir().unwrap();
        let file = make_file("packages/payments/service/capture.rs");
        let key = semantic_navigate::cluster_cache_key(&[file.relative_path.as_str()]);
        semantic_navigate::save_label_cache_full(
            tempdir.path(),
            &HashMap::from([(key.clone(), CachedLabel::heuristic("service".to_string()))]),
        );
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(
                ResponseTemplate::new(503).set_body_string("provider-secret-must-not-be-logged"),
            )
            .mount(&server)
            .await;
        let client = OllamaClient::new(&config_with_host(&server.uri()));
        let queue = vec![OwnedSubcluster {
            cache_key: key.clone(),
            parent_label: "payments".to_string(),
            files: vec![file],
        }];
        let (logs, subscriber) = captured_info_logs();
        let _guard = tracing::subscriber::set_default(subscriber);

        run_subcluster_llm_heal(&queue, &client, tempdir.path()).await;

        assert_eq!(load_label_cache_full(tempdir.path())[&key].label, "service");
        let logs = logs_as_string(&logs);
        let failure_lines = logs
            .lines()
            .filter(|line| line.contains("chat") && line.contains("fail"))
            .count();
        assert_eq!(
            failure_lines, 1,
            "chat failure must be logged once per batch:\n{logs}"
        );
        assert!(
            logs.contains("503 Service Unavailable"),
            "chat failure log omitted the provider error reason:\n{logs}"
        );
        assert!(
            !logs.contains("provider-secret-must-not-be-logged"),
            "chat failure log exposed provider response contents:\n{logs}"
        );
        assert!(
            !logs.contains("packages/payments/service/capture.rs"),
            "chat failure log exposed prompt contents:\n{logs}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn r3_cache_write_failure_retains_heuristics_without_upgrade_log() {
        use semantic_navigate::CachedLabel;
        let tempdir = tempfile::tempdir().unwrap();
        let file = make_file("packages/payments/service/settle.rs");
        let key = semantic_navigate::cluster_cache_key(&[file.relative_path.as_str()]);
        semantic_navigate::save_label_cache_full(
            tempdir.path(),
            &HashMap::from([(key.clone(), CachedLabel::heuristic("service".to_string()))]),
        );
        std::fs::create_dir(tempdir.path().join(".mcp_data/navigate-labels.json.tmp")).unwrap();
        let (_server, client) = mock_chat("[\"Payment Processing Core\"]").await;
        let queue = vec![OwnedSubcluster {
            cache_key: key.clone(),
            parent_label: "payments".to_string(),
            files: vec![file],
        }];
        let (logs, subscriber) = captured_info_logs();
        let _guard = tracing::subscriber::set_default(subscriber);
        run_subcluster_llm_heal(&queue, &client, tempdir.path()).await;
        assert_eq!(
            semantic_navigate::load_label_cache_full(tempdir.path())[&key].quality,
            semantic_navigate::LabelQuality::Heuristic
        );
        let logs = logs_as_string(&logs);
        assert_eq!(
            logs.lines().filter(|l| l.contains("failed")).count(),
            1,
            "{logs}"
        );
        assert!(!logs.contains("upgraded"), "{logs}");
        assert!(logs.contains("cache publication failed"), "{logs}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn parse_failure_logs_one_truncated_response_snippet() {
        use semantic_navigate::CachedLabel;

        let tempdir = tempfile::tempdir().unwrap();
        let file = make_file("packages/payments/service/settle.rs");
        let key = semantic_navigate::cluster_cache_key(&[file.relative_path.as_str()]);
        semantic_navigate::save_label_cache_full(
            tempdir.path(),
            &HashMap::from([(key.clone(), CachedLabel::heuristic("service".to_string()))]),
        );
        let response = format!(
            "Authorization: Bearer PARSE_SECRET_NEAR_PREFIX; model returned prose instead of labels: {}NEVER_LOG_THIS_RESPONSE_TAIL",
            "x".repeat(512)
        );
        let (_server, client) = mock_chat(&response).await;
        let queue = vec![OwnedSubcluster {
            cache_key: key,
            parent_label: "payments".to_string(),
            files: vec![file],
        }];
        let (logs, subscriber) = captured_info_logs();
        let _guard = tracing::subscriber::set_default(subscriber);

        run_subcluster_llm_heal(&queue, &client, tempdir.path()).await;

        let logs = logs_as_string(&logs);
        let parse_lines = logs.lines().filter(|line| line.contains("parse")).count();
        assert_eq!(
            parse_lines, 1,
            "parse failure must be logged once per batch:\n{logs}"
        );
        assert!(
            logs.contains("kind=prose") && logs.contains("brackets=0"),
            "parse failure log omitted the response shape:\n{logs}"
        );
        assert!(
            !logs.contains("PARSE_SECRET_NEAR_PREFIX"),
            "parse failure logged credential text from the response prefix:\n{logs}"
        );
        assert!(
            !logs.contains("NEVER_LOG_THIS_RESPONSE_TAIL"),
            "parse failure logged the untruncated response:\n{logs}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn in_flight_batch_skip_is_logged_once_with_reason() {
        use semantic_navigate::{claim_in_flight_keys, release_in_flight_keys};

        let tempdir = tempfile::tempdir().unwrap();
        let files = vec![
            make_file("packages/payments/repository/inflight_stripe.rs"),
            make_file("packages/payments/service/inflight_charge.rs"),
        ];
        let pending = two_subclusters(&files, "payments");
        let keys = seed_heuristics(tempdir.path(), &files, &["repository", "service"]);
        let claimed = claim_in_flight_keys(&keys);
        assert_eq!(claimed.len(), 2);
        let client = OllamaClient::new(&config_with_host("http://127.0.0.1:9"));
        let (logs, subscriber) = captured_info_logs();
        let _guard = tracing::subscriber::set_default(subscriber);

        let labels = label_subclusters_with_llm(&pending, &files, &client, tempdir.path()).await;
        release_in_flight_keys(&keys);

        assert_eq!(labels.get(&(0, 0)).map(String::as_str), Some("repository"));
        assert_eq!(labels.get(&(0, 1)).map(String::as_str), Some("service"));
        let logs = logs_as_string(&logs);
        let skipped_lines = logs
            .lines()
            .filter(|line| line.contains("spawn") && line.contains("skip"))
            .count();
        assert_eq!(
            skipped_lines, 1,
            "heal spawn skip must be logged once per batch:\n{logs}"
        );
        assert!(
            logs.contains("in-flight") || logs.contains("in_flight"),
            "heal spawn skip log omitted the in-flight reason:\n{logs}"
        );
    }

    #[tokio::test]
    async fn every_chat_provider_error_keeps_hybrid_heuristic_cache() {
        use crate::config::ChatProvider;
        use semantic_navigate::{
            CachedLabel, LabelQuality, load_label_cache_full, save_label_cache_full,
        };
        use wiremock::{Mock, MockServer, ResponseTemplate};
        for provider in [
            ChatProvider::Ollama,
            ChatProvider::OpenAi,
            ChatProvider::Anthropic,
            ChatProvider::Claude,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let server = MockServer::start().await;
            Mock::given(wiremock::matchers::method("POST"))
                .respond_with(ResponseTemplate::new(503))
                .expect(if provider == ChatProvider::Claude {
                    0
                } else {
                    1
                })
                .mount(&server)
                .await;
            let mut config = config_with_host(&server.uri());
            config.chat_provider = provider;
            config.openai_base_url = server.uri();
            config.chat_base_url = None;
            config.anthropic_base_url = server.uri();
            config.claude_path = dir
                .path()
                .join("missing-claude")
                .to_string_lossy()
                .into_owned();
            let file = make_file("src/auth/session.rs");
            let key = semantic_navigate::cluster_cache_key(&[&file.relative_path]);
            let cache =
                HashMap::from([(key.clone(), CachedLabel::heuristic("Session Flow".into()))]);
            save_label_cache_full(dir.path(), &cache);
            let queue = vec![OwnedSubcluster {
                cache_key: key.clone(),
                parent_label: "auth".into(),
                files: vec![file],
            }];
            run_subcluster_llm_heal(&queue, &OllamaClient::new(&config), dir.path()).await;
            let after = load_label_cache_full(dir.path());
            assert_eq!(after[&key].label, "Session Flow");
            assert_eq!(after[&key].quality, LabelQuality::Heuristic);
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

    #[tokio::test]
    async fn build_labeled_tree_uses_healed_label_for_a_flat_small_group() {
        let tempdir = tempfile::tempdir().unwrap();
        let files = vec![
            make_file("packages/payments/service/authorize.rs"),
            make_file("packages/payments/service/capture.rs"),
        ];
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let groups = vec![("payments/service".to_string(), vec![0, 1])];
        let params = ClusterParams {
            max_clusters: 4,
            min_clusters: 2,
            max_depth: 3,
        };
        let (_server, client) = mock_chat(r#"["Payment Authorization Flow"]"#).await;
        let paths: Vec<&str> = files
            .iter()
            .map(|file| file.relative_path.as_str())
            .collect();
        let key = semantic_navigate::cluster_cache_key(&paths);

        let first = build_labeled_tree(
            &files,
            &vectors,
            groups.clone(),
            &params,
            &client,
            tempdir.path(),
        )
        .await;
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].label, "payments/service");
        assert!(first[0].children.is_empty());
        assert_eq!(first[0].files.len(), files.len());

        assert!(
            wait_for_llm_labels(
                tempdir.path(),
                &[(key.clone(), "Payment Authorization Flow")]
            )
            .await,
            "flat small-group heal did not complete before timeout"
        );
        let second =
            build_labeled_tree(&files, &vectors, groups, &params, &client, tempdir.path()).await;

        assert_eq!(second.len(), 1);
        assert_eq!(second[0].label, "Payment Authorization Flow");
        assert!(second[0].children.is_empty(), "flat group became nested");
        assert_eq!(
            second[0]
                .files
                .iter()
                .map(|file| file.relative_path.as_str())
                .collect::<Vec<_>>(),
            paths
        );
    }

    #[tokio::test]
    async fn build_labeled_tree_uses_healed_label_for_an_unsplit_large_group() {
        let tempdir = tempfile::tempdir().unwrap();
        let files: Vec<FileInfo> = (0..128)
            .map(|i| make_file(&format!("packages/payments/service/payment_{i}.rs")))
            .collect();
        let vectors = vec![vec![f32::NAN, 0.0]; files.len()];
        let groups = vec![("payments/service".to_string(), (0..files.len()).collect())];
        let params = ClusterParams {
            max_clusters: 4,
            min_clusters: 1,
            max_depth: 3,
        };
        let (_server, client) = mock_chat(r#"["Payment Processing Core"]"#).await;
        let paths: Vec<&str> = files
            .iter()
            .map(|file| file.relative_path.as_str())
            .collect();
        let key = semantic_navigate::cluster_cache_key(&paths);

        let first = build_labeled_tree(
            &files,
            &vectors,
            groups.clone(),
            &params,
            &client,
            tempdir.path(),
        )
        .await;
        assert_eq!(first.len(), 1, "large group unexpectedly split");
        assert_eq!(first[0].label, "payments/service");
        assert!(first[0].children.is_empty());
        assert_eq!(first[0].files.len(), files.len());

        assert!(
            wait_for_llm_labels(tempdir.path(), &[(key.clone(), "Payment Processing Core")]).await,
            "unsplit large-group heal did not complete before timeout"
        );
        let second =
            build_labeled_tree(&files, &vectors, groups, &params, &client, tempdir.path()).await;

        assert_eq!(second.len(), 1);
        assert_eq!(second[0].label, "Payment Processing Core");
        assert!(second[0].children.is_empty(), "unsplit group became nested");
        assert_eq!(second[0].files.len(), files.len());
    }
}
