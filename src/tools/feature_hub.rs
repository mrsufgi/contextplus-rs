// Bundled skeleton view of all files linked from a hub map-of-content.
// Obsidian-style wikilink navigator for feature-oriented codebase exploration.

use crate::core::hub::{discover_hubs, find_orphaned_files, parse_hub_file, resolve_link};
use crate::core::safe_path::resolve_safe_path;
use crate::error::Result;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Options for the feature hub tool.
#[derive(Debug, Clone)]
pub struct FeatureHubOptions {
    pub root_dir: String,
    pub hub_path: Option<String>,
    pub feature_name: Option<String>,
    pub show_orphans: Option<bool>,
}

/// Get feature hub information: list hubs, show hub details, or find orphaned files.
pub async fn get_feature_hub(options: FeatureHubOptions) -> Result<String> {
    let root = PathBuf::from(&options.root_dir);
    let show_orphans = options.show_orphans.unwrap_or(false);

    // Case 1: No specific hub requested and no orphans -> list all hubs
    if options.hub_path.is_none() && options.feature_name.is_none() && !show_orphans {
        let hubs = discover_hubs(&root).await?;
        if hubs.is_empty() {
            return Ok(
                "No hub files found. Create a .md file with [[path/to/file]] links to establish a feature hub."
                    .to_string(),
            );
        }

        let mut out = vec![format!("Feature Hubs ({}):", hubs.len()), String::new()];
        for h in &hubs {
            let hub_full = root.join(h);
            if let Ok(content) = tokio::fs::read_to_string(&hub_full).await {
                let info = parse_hub_file(h, &content);
                out.push(format!(
                    "  {} | {} | {} links",
                    h,
                    info.title,
                    info.links.len()
                ));
            }
        }
        return Ok(out.join("\n"));
    }

    // Case 2: Show orphaned files
    if show_orphans {
        let all_files = collect_all_file_paths(&root).await?;
        let orphans = find_orphaned_files(&root, &all_files).await?;
        if orphans.is_empty() {
            return Ok("No orphaned files. All source files are linked to a hub.".to_string());
        }

        let mut out = vec![
            format!("Orphaned Files ({}):", orphans.len()),
            "These files are not linked to any feature hub:".to_string(),
            String::new(),
        ];
        for o in &orphans {
            out.push(format!("  ! {}", o));
        }
        out.push(String::new());
        out.push(format!(
            "Fix: Add [[{}]] to the appropriate hub .md file.",
            orphans[0]
        ));
        return Ok(out.join("\n"));
    }

    // Case 3: Find hub by path or name
    let hub_rel_path = if let Some(ref path) = options.hub_path {
        path.clone()
    } else if let Some(ref name) = options.feature_name {
        match find_hub_by_name(&root, name).await? {
            Some(p) => p,
            None => {
                let hubs = discover_hubs(&root).await?;
                let hub_list = if hubs.is_empty() {
                    "  (none)".to_string()
                } else {
                    hubs.iter()
                        .map(|h| format!("  - {}", h))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                return Ok(format!(
                    "No hub found for feature \"{}\". Available hubs:\n{}",
                    name, hub_list
                ));
            }
        }
    } else {
        return Ok("Provide hub_path, feature_name, or set show_orphans=true.".to_string());
    };

    // Case 4: Show specific hub details — validate path before use.
    resolve_safe_path(&root, &hub_rel_path)?;
    let hub_full = root.join(&hub_rel_path);
    if tokio::fs::metadata(&hub_full).await.is_err() {
        return Ok(format!("Hub file not found: {}", hub_rel_path));
    }

    let content = tokio::fs::read_to_string(&hub_full).await?;
    let hub = parse_hub_file(&hub_rel_path, &content);

    let mut out = vec![
        format!("Hub: {}", hub.title),
        format!("Path: {}", hub_rel_path),
        format!("Links: {}", hub.links.len()),
    ];

    if !hub.cross_links.is_empty() {
        let cross_link_names: Vec<String> =
            hub.cross_links.iter().map(|c| c.hub_name.clone()).collect();
        out.push(format!("Cross-links: {}", cross_link_names.join(", ")));
    }

    out.push(String::new());
    out.push("---".to_string());
    out.push(String::new());

    let mut resolved = Vec::new();
    let mut missing = Vec::new();

    for link in &hub.links {
        if resolve_link(&root, &link.target).await.is_some() {
            resolved.push(link.target.clone());
        } else {
            missing.push(link.target.clone());
        }
    }

    // Show resolved files with content preview
    for file_path in &resolved {
        let desc = hub
            .links
            .iter()
            .find(|l| l.target == *file_path)
            .and_then(|l| l.description.as_ref());

        if let Some(d) = desc {
            out.push(format!("## {} - {}", file_path, d));
        } else {
            out.push(format!("## {}", file_path));
        }

        // Show first 20 lines as preview
        let full = root.join(file_path);
        if let Ok(file_content) = tokio::fs::read_to_string(&full).await {
            let preview: String = file_content.lines().take(20).collect::<Vec<_>>().join("\n");
            out.push(preview);
        }
        out.push(String::new());
    }

    if !missing.is_empty() {
        out.push("---".to_string());
        out.push(format!("Missing Links ({}):", missing.len()));
        for m in &missing {
            out.push(format!("  x {}", m));
        }
    }

    Ok(out.join("\n"))
}

/// Find a hub file by feature name (partial match).
async fn find_hub_by_name(root: &Path, name: &str) -> Result<Option<String>> {
    let hubs = discover_hubs(root).await?;
    let lower = name.to_lowercase();

    // Exact match first
    let exact = hubs.iter().find(|h| {
        let h_lower = h.to_lowercase();
        h_lower == format!("{}.md", lower) || h_lower.ends_with(&format!("/{}.md", lower))
    });
    if let Some(h) = exact {
        return Ok(Some(h.clone()));
    }

    // Partial match
    let partial = hubs.iter().find(|h| h.to_lowercase().contains(&lower));
    Ok(partial.cloned())
}

/// Collect all file paths relative to root (non-directory).
async fn collect_all_file_paths(root: &Path) -> Result<Vec<String>> {
    let skip: HashSet<&str> = [
        "node_modules",
        ".git",
        "build",
        "dist",
        ".mcp_data",
        "target",
    ]
    .into_iter()
    .collect();

    let mut paths = Vec::new();
    let mut dirs = vec![root.to_path_buf()];

    while let Some(dir) = dirs.pop() {
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(e) => e,
            Err(_) => continue,
        };

        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if skip.contains(name_str.as_ref()) {
                continue;
            }

            let file_type = match entry.file_type().await {
                Ok(ft) => ft,
                Err(_) => continue,
            };

            let full_path = entry.path();
            if file_type.is_dir() {
                dirs.push(full_path);
            } else {
                if let Ok(rel) = full_path.strip_prefix(root) {
                    paths.push(rel.to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }

    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn list_hubs_empty_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let options = FeatureHubOptions {
            root_dir: dir.path().to_string_lossy().to_string(),
            hub_path: None,
            feature_name: None,
            show_orphans: None,
        };
        let result = get_feature_hub(options).await.expect("ok");
        assert!(result.contains("No hub files found"));
    }

    #[tokio::test]
    async fn list_hubs_finds_hub() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        tokio::fs::write(
            root.join("feature.md"),
            "# Auth Feature\n\n- [[src/auth.rs]]",
        )
        .await
        .expect("write");

        let options = FeatureHubOptions {
            root_dir: root.to_string_lossy().to_string(),
            hub_path: None,
            feature_name: None,
            show_orphans: None,
        };
        let result = get_feature_hub(options).await.expect("ok");
        assert!(result.contains("Feature Hubs (1)"));
        assert!(result.contains("feature.md"));
        assert!(result.contains("Auth Feature"));
    }

    #[tokio::test]
    async fn show_specific_hub() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        tokio::fs::write(
            root.join("hub.md"),
            "# My Hub\n\n- [[src/main.rs|entry]]\n- [[missing.rs]]",
        )
        .await
        .expect("write");

        let src = root.join("src");
        tokio::fs::create_dir_all(&src).await.expect("mkdir");
        tokio::fs::write(src.join("main.rs"), "fn main() {\n    println!(\"hi\");\n}")
            .await
            .expect("write");

        let options = FeatureHubOptions {
            root_dir: root.to_string_lossy().to_string(),
            hub_path: Some("hub.md".to_string()),
            feature_name: None,
            show_orphans: None,
        };
        let result = get_feature_hub(options).await.expect("ok");
        assert!(result.contains("Hub: My Hub"));
        assert!(result.contains("Links: 2"));
        assert!(result.contains("## src/main.rs - entry"));
        assert!(result.contains("Missing Links (1)"));
        assert!(result.contains("missing.rs"));
    }

    #[tokio::test]
    async fn show_orphaned_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        tokio::fs::write(root.join("hub.md"), "# Hub\n- [[linked.rs]]")
            .await
            .expect("write");
        tokio::fs::write(root.join("linked.rs"), "fn linked() {}")
            .await
            .expect("write");
        tokio::fs::write(root.join("orphan.rs"), "fn orphan() {}")
            .await
            .expect("write");

        let options = FeatureHubOptions {
            root_dir: root.to_string_lossy().to_string(),
            hub_path: None,
            feature_name: None,
            show_orphans: Some(true),
        };
        let result = get_feature_hub(options).await.expect("ok");
        assert!(result.contains("Orphaned Files"));
        assert!(result.contains("orphan.rs"));
    }

    #[tokio::test]
    async fn find_hub_by_feature_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        tokio::fs::write(
            root.join("auth.md"),
            "# Authentication\n\n- [[src/auth.rs]]",
        )
        .await
        .expect("write");

        let options = FeatureHubOptions {
            root_dir: root.to_string_lossy().to_string(),
            hub_path: None,
            feature_name: Some("auth".to_string()),
            show_orphans: None,
        };
        let result = get_feature_hub(options).await.expect("ok");
        assert!(result.contains("Hub: Authentication"));
    }

    #[tokio::test]
    async fn feature_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let options = FeatureHubOptions {
            root_dir: dir.path().to_string_lossy().to_string(),
            hub_path: None,
            feature_name: Some("nonexistent".to_string()),
            show_orphans: None,
        };
        let result = get_feature_hub(options).await.expect("ok");
        assert!(result.contains("No hub found for feature"));
    }

    #[tokio::test]
    async fn hub_file_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let options = FeatureHubOptions {
            root_dir: dir.path().to_string_lossy().to_string(),
            hub_path: Some("nonexistent.md".to_string()),
            feature_name: None,
            show_orphans: None,
        };
        let result = get_feature_hub(options).await.expect("ok");
        assert!(result.contains("Hub file not found"));
    }

    #[tokio::test]
    async fn no_orphans_message() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        // No source files at all
        tokio::fs::write(root.join("hub.md"), "# Hub\n- [[readme.md]]")
            .await
            .expect("write");

        let options = FeatureHubOptions {
            root_dir: root.to_string_lossy().to_string(),
            hub_path: None,
            feature_name: None,
            show_orphans: Some(true),
        };
        let result = get_feature_hub(options).await.expect("ok");
        assert!(result.contains("No orphaned files"));
    }
}
