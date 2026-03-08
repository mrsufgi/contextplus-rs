// Obsidian-style hub parser extracting wikilinks and cross-link tags.
// Parses [[path/to/file]] and [[path|alias]] from markdown files.

use regex::Regex;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// A parsed wikilink target with optional description alias.
#[derive(Debug, Clone, PartialEq)]
pub struct HubLink {
    pub target: String,
    pub description: Option<String>,
}

/// A cross-link reference found via `@linked-to [[hub]]` syntax.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossLink {
    pub hub_name: String,
    pub source_file: String,
}

/// Full parsed information about a hub markdown file.
#[derive(Debug, Clone)]
pub struct HubInfo {
    pub hub_path: String,
    pub title: String,
    pub links: Vec<HubLink>,
    pub cross_links: Vec<CrossLink>,
    pub raw: String,
}

/// Parse wikilinks from markdown content: `[[target]]` or `[[target|description]]`
/// Excludes cross-link references (`@linked-to [[...]]`).
pub fn parse_wiki_links(content: &str) -> Vec<HubLink> {
    let cross_link_re = Regex::new(r"@linked-to\s+\[\[[^\]]+\]\]").expect("valid regex");
    let cleaned = cross_link_re.replace_all(content, "");

    let wikilink_re = Regex::new(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]").expect("valid regex");
    let mut links = Vec::new();
    let mut seen = HashSet::new();

    for cap in wikilink_re.captures_iter(&cleaned) {
        let target = cap[1].trim().to_string();
        if seen.insert(target.clone()) {
            let description = cap.get(2).map(|m| m.as_str().trim().to_string());
            links.push(HubLink {
                target,
                description,
            });
        }
    }
    links
}

/// Parse `@linked-to [[hubName]]` cross-link references.
pub fn parse_cross_links(content: &str, source_file: &str) -> Vec<CrossLink> {
    let re = Regex::new(r"@linked-to\s+\[\[([^\]]+)\]\]").expect("valid regex");
    re.captures_iter(content)
        .map(|cap| CrossLink {
            hub_name: cap[1].trim().to_string(),
            source_file: source_file.to_string(),
        })
        .collect()
}

/// Extract a `// FEATURE: ...` or `# FEATURE: ...` or `-- FEATURE: ...` tag from file header.
pub fn extract_feature_tag(content: &str) -> Option<String> {
    let re = Regex::new(r"(?m)^(?://|#|--)\s*FEATURE:\s*(.+)$").expect("valid regex");
    re.captures(content).map(|cap| cap[1].trim().to_string())
}

/// Parse a hub markdown file into structured information.
pub fn parse_hub_file(hub_path: &str, content: &str) -> HubInfo {
    let path = Path::new(hub_path);
    let mut title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled")
        .to_string();

    // Extract first heading as title
    let heading_re = Regex::new(r"(?m)^#\s+(.+)$").expect("valid regex");
    if let Some(cap) = heading_re.captures(content) {
        title = cap[1].trim().to_string();
    }

    HubInfo {
        hub_path: hub_path.to_string(),
        title,
        links: parse_wiki_links(content),
        cross_links: parse_cross_links(content, hub_path),
        raw: content.to_string(),
    }
}

/// Check if a markdown file contains wikilinks (is a hub).
pub fn has_wikilinks(content: &str) -> bool {
    let re = Regex::new(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]").expect("valid regex");
    re.is_match(content)
}

/// Walk a directory tree and discover hub files (markdown files with wikilinks).
/// Returns paths relative to root_dir.
pub async fn discover_hubs(root_dir: &Path) -> crate::error::Result<Vec<String>> {
    let skip: HashSet<&str> = ["node_modules", ".git", "build", "dist", ".mcp_data"]
        .iter()
        .copied()
        .collect();

    let mut hubs = Vec::new();
    let mut dirs = vec![root_dir.to_path_buf()];

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
            } else if name_str.ends_with(".md")
                && let Ok(content) = tokio::fs::read_to_string(&full_path).await
                    && has_wikilinks(&content)
                        && let Ok(rel) = full_path.strip_prefix(root_dir) {
                            let rel_str = rel.to_string_lossy().replace('\\', "/");
                            hubs.push(rel_str);
                        }
        }
    }

    hubs.sort();
    Ok(hubs)
}

/// Find files that are not linked from any hub.
pub async fn find_orphaned_files(
    root_dir: &Path,
    all_file_paths: &[String],
) -> crate::error::Result<Vec<String>> {
    let hubs = discover_hubs(root_dir).await?;
    let mut linked_files = HashSet::new();

    for hub_rel_path in &hubs {
        let hub_full = root_dir.join(hub_rel_path);
        if let Ok(content) = tokio::fs::read_to_string(&hub_full).await {
            let info = parse_hub_file(hub_rel_path, &content);
            for link in &info.links {
                linked_files.insert(link.target.replace('\\', "/"));
            }
        }
        linked_files.insert(hub_rel_path.replace('\\', "/"));
    }

    Ok(all_file_paths
        .iter()
        .filter(|f| !f.ends_with(".md"))
        .filter(|f| !linked_files.contains(&f.replace('\\', "/")))
        .cloned()
        .collect())
}

/// Format a hub link for markdown output.
pub fn format_hub_link(target: &str, description: &str) -> String {
    if description.is_empty() {
        format!("- [[{}]]", target)
    } else {
        format!("- [[{}|{}]]", target, description)
    }
}

/// Resolve a wikilink target to an actual file path relative to root_dir.
pub async fn resolve_link(root_dir: &Path, target: &str) -> Option<PathBuf> {
    let full = root_dir.join(target);
    if tokio::fs::metadata(&full).await.is_ok() {
        Some(full)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_wikilink() {
        let content = "See [[src/main.rs]] for details.";
        let links = parse_wiki_links(content);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "src/main.rs");
        assert_eq!(links[0].description, None);
    }

    #[test]
    fn parse_wikilink_with_alias() {
        let content = "Check [[src/lib.rs|library root]].";
        let links = parse_wiki_links(content);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "src/lib.rs");
        assert_eq!(links[0].description, Some("library root".to_string()));
    }

    #[test]
    fn parse_multiple_wikilinks() {
        let content = "Files: [[a.rs]] and [[b.rs]] and [[c.rs|the C file]]";
        let links = parse_wiki_links(content);
        assert_eq!(links.len(), 3);
        assert_eq!(links[0].target, "a.rs");
        assert_eq!(links[1].target, "b.rs");
        assert_eq!(links[2].target, "c.rs");
        assert_eq!(links[2].description, Some("the C file".to_string()));
    }

    #[test]
    fn parse_deduplicates_wikilinks() {
        let content = "See [[a.rs]] and again [[a.rs]]";
        let links = parse_wiki_links(content);
        assert_eq!(links.len(), 1);
    }

    #[test]
    fn parse_excludes_cross_links() {
        let content = "@linked-to [[auth-hub]]\nSee [[src/main.rs]]";
        let links = parse_wiki_links(content);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].target, "src/main.rs");
    }

    #[test]
    fn parse_cross_links_basic() {
        let content = "@linked-to [[auth-hub]]\n@linked-to [[forms-hub]]";
        let cross = parse_cross_links(content, "test.md");
        assert_eq!(cross.len(), 2);
        assert_eq!(cross[0].hub_name, "auth-hub");
        assert_eq!(cross[0].source_file, "test.md");
        assert_eq!(cross[1].hub_name, "forms-hub");
    }

    #[test]
    fn extract_feature_tag_comment_style() {
        assert_eq!(
            extract_feature_tag("// FEATURE: Memory Graph"),
            Some("Memory Graph".to_string())
        );
        assert_eq!(
            extract_feature_tag("# FEATURE: Auth Flow"),
            Some("Auth Flow".to_string())
        );
        assert_eq!(
            extract_feature_tag("-- FEATURE: SQL Queries"),
            Some("SQL Queries".to_string())
        );
    }

    #[test]
    fn extract_feature_tag_none() {
        assert_eq!(extract_feature_tag("no feature tag here"), None);
    }

    #[test]
    fn parse_hub_file_basic() {
        let content = "# My Feature\n\n- [[src/a.rs]]\n- [[src/b.rs|B file]]\n";
        let info = parse_hub_file("docs/feature.md", content);
        assert_eq!(info.title, "My Feature");
        assert_eq!(info.links.len(), 2);
        assert_eq!(info.links[0].target, "src/a.rs");
        assert_eq!(info.links[1].description, Some("B file".to_string()));
    }

    #[test]
    fn parse_hub_file_no_heading() {
        let content = "Some content with [[link.rs]]";
        let info = parse_hub_file("notes/todo.md", content);
        assert_eq!(info.title, "todo");
        assert_eq!(info.links.len(), 1);
    }

    #[test]
    fn has_wikilinks_true() {
        assert!(has_wikilinks("See [[file.rs]] here"));
    }

    #[test]
    fn has_wikilinks_false() {
        assert!(!has_wikilinks("No links here"));
    }

    #[test]
    fn format_hub_link_with_description() {
        assert_eq!(
            format_hub_link("src/main.rs", "entry point"),
            "- [[src/main.rs|entry point]]"
        );
    }

    #[test]
    fn format_hub_link_without_description() {
        assert_eq!(format_hub_link("src/main.rs", ""), "- [[src/main.rs]]");
    }

    #[tokio::test]
    async fn discover_hubs_in_temp_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        // Create a hub file with wikilinks
        tokio::fs::write(root.join("hub.md"), "# Hub\n- [[src/a.rs]]")
            .await
            .expect("write");
        // Create a non-hub markdown file
        tokio::fs::write(root.join("readme.md"), "# Readme\nNo links here.")
            .await
            .expect("write");

        let hubs = discover_hubs(root).await.expect("discover");
        assert_eq!(hubs, vec!["hub.md"]);
    }

    #[tokio::test]
    async fn find_orphaned_files_basic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        tokio::fs::write(root.join("hub.md"), "# Hub\n- [[linked.rs]]")
            .await
            .expect("write");

        let all_files = vec![
            "linked.rs".to_string(),
            "orphan.rs".to_string(),
            "hub.md".to_string(),
        ];

        let orphans = find_orphaned_files(root, &all_files).await.expect("orphans");
        assert_eq!(orphans, vec!["orphan.rs"]);
    }

    #[tokio::test]
    async fn resolve_link_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        tokio::fs::write(root.join("test.rs"), "fn main() {}")
            .await
            .expect("write");

        let result = resolve_link(root, "test.rs").await;
        assert!(result.is_some());
    }

    #[tokio::test]
    async fn resolve_link_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let result = resolve_link(dir.path(), "nonexistent.rs").await;
        assert!(result.is_none());
    }
}
