// Semantic project navigator using spectral clustering and Ollama labeling.
// Browse codebase by meaning: embeds files, clusters vectors, generates labels.

use crate::config::Config;
use crate::core::clustering::{find_path_pattern, spectral_cluster};
use crate::core::embeddings::OllamaClient;
use crate::core::walker;
use crate::error::Result;
use serde::Deserialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const MAX_FILES_PER_LEAF: usize = 20;

/// Extensions accepted for semantic navigation (without leading dot).
/// Superset of tree_sitter::get_supported_extensions — includes data formats
/// (sql, graphql, proto, yaml, yml, toml, json) that tree-sitter doesn't parse
/// but are useful for clustering.
const NAVIGATE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "go", "java", "c", "cpp", "h", "hpp", "cc",
    "rb", "sh", "bash", "zsh", "sql", "graphql", "proto", "yaml", "yml", "toml", "json",
];

/// Options for the semantic navigation tool.
#[derive(Debug, Clone)]
pub struct SemanticNavigateOptions {
    pub root_dir: String,
    pub max_depth: Option<usize>,
    pub max_clusters: Option<usize>,
}

/// Information about a source file for clustering.
#[derive(Debug, Clone)]
struct FileInfo {
    relative_path: String,
    header: String,
    content: String,
    symbol_preview: Vec<String>,
}

/// A hierarchical cluster node.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ClusterNode {
    label: String,
    path_pattern: Option<String>,
    files: Vec<FileInfo>,
    children: Vec<ClusterNode>,
}

/// Perform semantic navigation: embed files, cluster, label, return tree.
pub async fn semantic_navigate(
    options: SemanticNavigateOptions,
    ollama: &OllamaClient,
    config: &Config,
) -> Result<String> {
    let max_clusters = options.max_clusters.unwrap_or(20);
    let max_depth = options.max_depth.unwrap_or(3);
    let root = PathBuf::from(&options.root_dir);

    // Walk directory for source files using shared walker infrastructure
    let files = collect_source_files_via_walker(&root, config).await?;
    if files.is_empty() {
        return Ok("No supported source files found in the project.".to_string());
    }

    // Build embedding texts
    let embed_texts: Vec<String> = files
        .iter()
        .map(|f| format!("{} {} {}", f.header, f.relative_path, f.content))
        .collect();

    // Fetch embeddings via OllamaClient
    let vectors = match ollama.embed(&embed_texts).await {
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
        let file_labels = label_files(&files, ollama).await;
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

    // Build hierarchical cluster tree
    let tree = build_hierarchy(&files, &vectors, max_clusters, 0, max_depth, ollama).await;
    let mut root_node = tree;
    root_node.label = "Project".to_string();

    Ok(format!(
        "Semantic Navigator: {} files organized by meaning\n\n{}",
        files.len(),
        render_cluster_tree(&root_node, 0)
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

    // Filter to supported extensions and non-directories, then read content
    let mut files = Vec::new();
    for entry in entries {
        if entry.is_directory {
            continue;
        }

        let ext = entry
            .path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        if !allowed_extensions.contains(ext) {
            continue;
        }

        if let Ok(content) = tokio::fs::read_to_string(&entry.path).await {
            let header = extract_header(&content);
            let truncated_content = if content.len() > 500 {
                crate::core::parser::truncate_to_char_boundary(&content, 500).to_string()
            } else {
                content
            };

            files.push(FileInfo {
                relative_path: entry.relative_path,
                header,
                content: truncated_content,
                symbol_preview: Vec::new(),
            });
        }
    }

    Ok(files)
}

/// Extract a header comment from the first few lines of a file.
fn extract_header(content: &str) -> String {
    let mut header_lines = Vec::new();
    for line in content.lines().take(5) {
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
    if joined.len() > 200 {
        crate::core::parser::truncate_to_char_boundary(&joined, 200).to_string()
    } else {
        joined
    }
}

/// Label files using Ollama chat for small sets.
async fn label_files(files: &[FileInfo], ollama: &OllamaClient) -> Vec<String> {
    let file_list = files
        .iter()
        .map(|f| format!("{}: {}", f.relative_path, f.header))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "For each file below, produce a 3-7 word description. Return ONLY a JSON array of strings.\n\n{}",
        file_list
    );

    match ollama.chat(&prompt).await {
        Ok(response) => {
            if let Some(json_match) = extract_json_array(&response)
                && let Ok(labels) = serde_json::from_str::<Vec<String>>(&json_match)
            {
                return labels;
            }
            files.iter().map(|f| f.header.clone()).collect()
        }
        Err(_) => files.iter().map(|f| f.header.clone()).collect(),
    }
}

/// Label sibling clusters using Ollama chat.
async fn label_sibling_clusters(
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    ollama: &OllamaClient,
) -> Vec<String> {
    if clusters.is_empty() {
        return Vec::new();
    }
    if clusters.len() == 1 {
        if let Some(ref pp) = clusters[0].1 {
            return vec![pp.clone()];
        }
        let names: Vec<&str> = clusters[0]
            .0
            .iter()
            .filter_map(|f| f.relative_path.split('/').next_back())
            .collect();
        let joined = names.join(", ");
        return vec![if joined.len() > 40 {
            crate::core::parser::truncate_to_char_boundary(&joined, 40).to_string()
        } else {
            joined
        }];
    }

    let descriptions: Vec<String> = clusters
        .iter()
        .enumerate()
        .map(|(i, (files, pattern))| {
            let file_list = files
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
            let pp = pattern
                .as_ref()
                .map(|p| format!(" (pattern: {})", p))
                .unwrap_or_default();
            format!("Cluster {}{}:\n  {}", i + 1, pp, file_list)
        })
        .collect();

    let prompt = format!(
        r#"You are labeling clusters of code files. For each cluster below, produce EXACTLY one JSON array of objects, each with:
- "overarchingTheme": a sentence about the cluster's theme
- "distinguishingFeature": what makes this cluster unique vs siblings
- "label": EXACTLY 2 words describing the cluster

{}

Respond with ONLY a JSON array of {} objects. No other text."#,
        descriptions.join("\n\n"),
        clusters.len()
    );

    match ollama.chat(&prompt).await {
        Ok(response) => {
            if let Some(json_str) = extract_json_array(&response) {
                #[derive(Deserialize)]
                struct ClusterLabel {
                    label: Option<String>,
                }
                if let Ok(labels) = serde_json::from_str::<Vec<ClusterLabel>>(&json_str) {
                    return labels
                        .into_iter()
                        .enumerate()
                        .map(|(i, l)| {
                            let base = l.label.unwrap_or_else(|| format!("Cluster {}", i + 1));
                            if let Some(ref pp) = clusters[i].1 {
                                format!("{} ({})", base, pp)
                            } else {
                                base
                            }
                        })
                        .collect();
                }
            }
            clusters
                .iter()
                .enumerate()
                .map(|(i, (_, pp))| pp.clone().unwrap_or_else(|| format!("Cluster {}", i + 1)))
                .collect()
        }
        Err(_) => clusters
            .iter()
            .enumerate()
            .map(|(i, (_, pp))| pp.clone().unwrap_or_else(|| format!("Cluster {}", i + 1)))
            .collect(),
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
async fn build_hierarchy(
    files: &[FileInfo],
    vectors: &[Vec<f32>],
    max_clusters: usize,
    depth: usize,
    max_depth: usize,
    ollama: &OllamaClient,
) -> ClusterNode {
    if files.len() <= MAX_FILES_PER_LEAF || depth >= max_depth {
        return ClusterNode {
            label: String::new(),
            path_pattern: find_path_pattern(
                &files
                    .iter()
                    .map(|f| f.relative_path.clone())
                    .collect::<Vec<_>>(),
            ),
            files: files.to_vec(),
            children: Vec::new(),
        };
    }

    let cluster_results = spectral_cluster(vectors, max_clusters);

    if cluster_results.len() <= 1 {
        return ClusterNode {
            label: String::new(),
            path_pattern: find_path_pattern(
                &files
                    .iter()
                    .map(|f| f.relative_path.clone())
                    .collect::<Vec<_>>(),
            ),
            files: files.to_vec(),
            children: Vec::new(),
        };
    }

    // Build child metadata
    #[allow(clippy::type_complexity)]
    let child_metas: Vec<(Vec<&FileInfo>, Vec<Vec<f32>>, Option<String>)> = cluster_results
        .iter()
        .map(|cluster| {
            let cluster_files: Vec<&FileInfo> =
                cluster.indices.iter().map(|&i| &files[i]).collect();
            let cluster_vectors: Vec<Vec<f32>> = cluster
                .indices
                .iter()
                .map(|&i| vectors[i].clone())
                .collect();
            let paths: Vec<String> = cluster_files
                .iter()
                .map(|f| f.relative_path.clone())
                .collect();
            let pattern = find_path_pattern(&paths);
            (cluster_files, cluster_vectors, pattern)
        })
        .collect();

    // Get labels for sibling clusters
    let label_input: Vec<(Vec<&FileInfo>, Option<String>)> = child_metas
        .iter()
        .map(|(files, _, pattern)| (files.clone(), pattern.clone()))
        .collect();
    let labels = label_sibling_clusters(&label_input, ollama).await;

    // Recurse into children
    let mut children = Vec::new();
    for (i, (cluster_files, cluster_vectors, _pattern)) in child_metas.into_iter().enumerate() {
        let owned_files: Vec<FileInfo> = cluster_files.iter().map(|f| (*f).clone()).collect();
        let mut child = Box::pin(build_hierarchy(
            &owned_files,
            &cluster_vectors,
            max_clusters,
            depth + 1,
            max_depth,
            ollama,
        ))
        .await;
        child.label = labels
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("Cluster {}", i + 1));
        children.push(child);
    }

    ClusterNode {
        label: String::new(),
        path_pattern: find_path_pattern(
            &files
                .iter()
                .map(|f| f.relative_path.clone())
                .collect::<Vec<_>>(),
        ),
        files: Vec::new(),
        children,
    }
}

/// Render a cluster tree as indented text.
fn render_cluster_tree(node: &ClusterNode, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    let mut result = String::new();

    if !node.label.is_empty() {
        result.push_str(&format!("{}[{}]\n", pad, node.label));
    }

    if !node.children.is_empty() {
        for child in &node.children {
            result.push_str(&render_cluster_tree(child, indent + 1));
        }
    } else {
        for file in &node.files {
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
            path_pattern: None,
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
            path_pattern: None,
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
            path_pattern: None,
            files: Vec::new(),
            children: vec![child],
        };
        let rendered = render_cluster_tree(&parent, 0);
        assert!(rendered.contains("[Parent]"));
        assert!(rendered.contains("  [Child]"));
        assert!(rendered.contains("    a.rs"));
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
}
