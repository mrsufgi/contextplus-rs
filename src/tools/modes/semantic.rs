use crate::core::clustering::spectral_cluster_with_min;
use crate::core::embeddings::OllamaClient;
use std::path::Path;

use super::super::navigate_constants::*;
use super::super::semantic_navigate::{ClusterNode, FileInfo, fallback_label};
use super::ClusterParams;

/// SEMANTIC MODE: Recursively build cluster hierarchy using pure spectral clustering.
/// Like the original contextplus — no directory-based grouping, LLM labels at every level.
pub(crate) async fn build_semantic_hierarchy(
    all_files: &[FileInfo],
    all_vectors: &[Vec<f32>],
    indices: &[usize],
    depth: usize,
    params: &ClusterParams,
    ollama: &OllamaClient,
    root_dir: &Path,
) -> ClusterNode {
    if indices.len() <= MAX_FILES_PER_LEAF || depth >= params.max_depth {
        return ClusterNode {
            label: String::new(),
            files: indices.iter().map(|&i| all_files[i].clone()).collect(),
            children: Vec::new(),
        };
    }

    // Spectral clustering on this group's vectors
    let local_vectors: Vec<Vec<f32>> = indices.iter().map(|&i| all_vectors[i].clone()).collect();
    let mc = params.max_clusters;
    let mn = params.min_clusters;
    let cluster_results =
        tokio::task::spawn_blocking(move || spectral_cluster_with_min(&local_vectors, mc, mn))
            .await
            .unwrap_or_else(|_| vec![]);

    if cluster_results.len() <= 1 {
        return ClusterNode {
            label: String::new(),
            files: indices.iter().map(|&i| all_files[i].clone()).collect(),
            children: Vec::new(),
        };
    }

    // Map back to global indices
    let child_groups: Vec<Vec<usize>> = cluster_results
        .iter()
        .map(|c| c.indices.iter().map(|&li| indices[li]).collect())
        .collect();

    // Label sibling clusters with LLM (single batched call)
    let label_input: Vec<(Vec<&FileInfo>, Option<String>)> = child_groups
        .iter()
        .map(|idxs| {
            let refs: Vec<&FileInfo> = idxs.iter().map(|&i| &all_files[i]).collect();
            (refs, None)
        })
        .collect();

    let labels =
        super::super::semantic_navigate::label_clusters_with_cache(&label_input, ollama, root_dir)
            .await;

    // Recurse into children sequentially (to avoid Ollama contention)
    let mut children: Vec<ClusterNode> = Vec::new();
    for (i, child_indices) in child_groups.iter().enumerate() {
        let mut child = Box::pin(build_semantic_hierarchy(
            all_files,
            all_vectors,
            child_indices,
            depth + 1,
            params,
            ollama,
            root_dir,
        ))
        .await;
        child.label = labels.get(i).cloned().unwrap_or_else(|| {
            let refs: Vec<&FileInfo> = child_groups[i].iter().map(|&idx| &all_files[idx]).collect();
            fallback_label(&refs)
        });
        children.push(child);
    }

    ClusterNode {
        label: String::new(),
        files: Vec::new(),
        children,
    }
}

/// Label clusters for semantic mode using LLM with themed prompt.
pub(crate) async fn label_clusters_for_semantic_mode(
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    ollama: &OllamaClient,
) -> Vec<String> {
    if clusters.is_empty() {
        return Vec::new();
    }

    // Build LLM prompt — original contextplus asked for overarchingTheme + distinguishingFeature + label
    let descriptions: Vec<String> = clusters
        .iter()
        .enumerate()
        .map(|(i, (files, _))| {
            let sample: Vec<String> = files
                .iter()
                .take(10)
                .map(|f| {
                    let desc = if f.header.is_empty() {
                        "no description"
                    } else {
                        &f.header
                    };
                    format!("{}: {}", f.relative_path, desc)
                })
                .collect();
            // Use letters (A, B, C) instead of "Cluster N" to prevent LLM echoing
            let letter = (b'A' + (i as u8 % 26)) as char;
            format!(
                "Group {} ({} files):\n  {}",
                letter,
                files.len(),
                sample.join("\n  ")
            )
        })
        .collect();

    let prompt = format!(
        "You are labeling clusters of source code files.\n\
         For each group below, give a SHORT descriptive label (2-4 words) describing the PURPOSE of the code.\n\
         Look at the file paths and descriptions to understand what each group does.\n\n\
         Example input:\n\
         Group A (5 files): service/auth.ts, service/session.ts, service/user.ts...\n\
         Group B (3 files): repository/pg/user.ts, repository/pg/session.ts...\n\
         Example output: [\"Authentication Services\", \"User Data Access\"]\n\n\
         {}\n\n\
         Return ONLY a JSON array of {} strings:",
        descriptions.join("\n\n"),
        clusters.len()
    );

    match ollama.chat(&prompt).await {
        Ok(response) => {
            // Try to parse the rich response format
            if let Some(json_str) = super::super::semantic_navigate::extract_json_array(&response) {
                // Try rich format first: [{overarchingTheme, distinguishingFeature, label}]
                if let Ok(rich) = serde_json::from_str::<Vec<serde_json::Value>>(&json_str) {
                    let labels: Vec<String> = rich
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            let raw = v
                                .get("label")
                                .and_then(|l| l.as_str())
                                .or_else(|| v.as_str())
                                .map(|s| s.to_string());
                            match raw {
                                Some(s) if !s.is_empty() => s,
                                _ => {
                                    let (files, _) = &clusters[i.min(clusters.len() - 1)];
                                    fallback_label(files)
                                }
                            }
                        })
                        .collect();
                    if labels.len() == clusters.len() {
                        return labels;
                    }
                }
            }
            // LLM response unparseable — use path-based fallbacks
            clusters
                .iter()
                .map(|(files, _)| fallback_label(files))
                .collect()
        }
        Err(_) => {
            // LLM failed — use path-based fallbacks
            clusters
                .iter()
                .map(|(files, _)| fallback_label(files))
                .collect()
        }
    }
}
