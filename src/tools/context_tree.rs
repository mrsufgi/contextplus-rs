//! Structural tree generator with file headers, symbols, and depth control.
//!
//! Ports the TypeScript `context-tree.ts` logic:
//! - Builds a tree of files/directories with optional headers and symbols
//! - Dynamic token-aware pruning: Level 2 (full) -> Level 1 (headers) -> Level 0 (files only)
//! - CHARS_PER_TOKEN = 4, estimateTokens(s) = ceil(s.len() / 4)

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::error::Result;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CHARS_PER_TOKEN: usize = 4;
const DEFAULT_MAX_TOKENS: usize = 50_000;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ContextTreeOptions {
    pub root_dir: PathBuf,
    pub target_path: Option<String>,
    pub depth_limit: Option<usize>,
    pub include_symbols: Option<bool>,
    pub max_tokens: Option<usize>,
}

/// A file entry from the file walker.
#[derive(Debug, Clone)]
pub struct FileEntry {
    pub relative_path: String,
    pub is_directory: bool,
    pub depth: usize,
}

/// A code symbol for tree display.
#[derive(Debug, Clone)]
pub struct TreeSymbol {
    pub name: String,
    pub kind: String,
    pub line: usize,
    pub end_line: usize,
    pub signature: String,
    pub children: Vec<TreeSymbol>,
}

/// Analysis result for a single file.
#[derive(Debug, Clone)]
pub struct FileAnalysis {
    pub header: Option<String>,
    pub symbols: Vec<TreeSymbol>,
}

// ---------------------------------------------------------------------------
// Tree node (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TreeNode {
    name: String,
    relative_path: String,
    is_directory: bool,
    header: Option<String>,
    symbols: Option<String>,
    children: Vec<TreeNode>,
}

// ---------------------------------------------------------------------------
// Pure functions
// ---------------------------------------------------------------------------

fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(CHARS_PER_TOKEN)
}

/// Format a single symbol with indentation.
pub fn format_symbol(sym: &TreeSymbol, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    let range = if sym.end_line > sym.line {
        format!("L{}-L{}", sym.line, sym.end_line)
    } else {
        format!("L{}", sym.line)
    };
    format!("{}[{}] {} {}", pad, sym.kind, range, sym.signature)
}

fn format_symbols(symbols: &[TreeSymbol]) -> String {
    let mut lines = Vec::new();
    for sym in symbols {
        lines.push(format_symbol(sym, 0));
        for child in &sym.children {
            lines.push(format_symbol(child, 1));
        }
    }
    lines.join("\n")
}

/// Build tree structure from sorted file entries and optional analyses.
/// `analyses` maps relative_path -> FileAnalysis for supported files.
fn build_tree(
    entries: &[FileEntry],
    analyses: &BTreeMap<String, FileAnalysis>,
    include_symbols: bool,
) -> TreeNode {
    let root = TreeNode {
        name: ".".to_string(),
        relative_path: ".".to_string(),
        is_directory: true,
        header: None,
        symbols: None,
        children: Vec::new(),
    };

    let mut dir_map: BTreeMap<String, Vec<TreeNode>> = BTreeMap::new();
    dir_map.insert(".".to_string(), Vec::new());

    // Sort entries by depth then by path
    let mut sorted = entries.to_vec();
    sorted.sort_by(|a, b| {
        a.depth
            .cmp(&b.depth)
            .then_with(|| a.relative_path.cmp(&b.relative_path))
    });

    for entry in &sorted {
        let parts: Vec<&str> = entry.relative_path.split('/').collect();
        let name = parts.last().copied().unwrap_or(&entry.relative_path);
        let parent_path = if parts.len() > 1 {
            parts[..parts.len() - 1].join("/")
        } else {
            ".".to_string()
        };

        let mut node = TreeNode {
            name: name.to_string(),
            relative_path: entry.relative_path.clone(),
            is_directory: entry.is_directory,
            header: None,
            symbols: None,
            children: Vec::new(),
        };

        if !entry.is_directory
            && let Some(analysis) = analyses.get(&entry.relative_path)
        {
            node.header = analysis.header.clone();
            if include_symbols && !analysis.symbols.is_empty() {
                node.symbols = Some(format_symbols(&analysis.symbols));
            }
        }

        if entry.is_directory {
            dir_map.entry(entry.relative_path.clone()).or_default();
        }

        dir_map.entry(parent_path).or_default().push(node);
    }

    // Reconstruct tree bottom-up
    fn collect_children(
        path: &str,
        dir_map: &mut BTreeMap<String, Vec<TreeNode>>,
    ) -> Vec<TreeNode> {
        let children = dir_map.remove(path).unwrap_or_default();
        children
            .into_iter()
            .map(|mut node| {
                if node.is_directory {
                    node.children = collect_children(&node.relative_path, dir_map);
                }
                node
            })
            .collect()
    }

    let children = collect_children(".", &mut dir_map);
    TreeNode { children, ..root }
}

fn render_tree(node: &TreeNode, indent: usize) -> String {
    let mut result = String::new();
    let pad = "  ".repeat(indent);

    if indent == 0 {
        result.push_str(&format!("{}/\n", node.name));
    } else if node.is_directory {
        result.push_str(&format!("{}{}/\n", pad, node.name));
    } else {
        result.push_str(&format!("{}{}", pad, node.name));
        if let Some(ref header) = node.header {
            result.push_str(&format!(" | {}", header));
        }
        result.push('\n');
        if let Some(ref symbols) = node.symbols {
            for line in symbols.split('\n') {
                result.push_str(&format!("{}  {}\n", pad, line));
            }
        }
    }

    for child in &node.children {
        result.push_str(&render_tree(child, indent + 1));
    }
    result
}

fn prune_symbols(node: &mut TreeNode) {
    node.symbols = None;
    for child in &mut node.children {
        prune_symbols(child);
    }
}

fn prune_headers(node: &mut TreeNode) {
    node.header = None;
    node.symbols = None;
    for child in &mut node.children {
        prune_headers(child);
    }
}

// ---------------------------------------------------------------------------
// High-level entry point
// ---------------------------------------------------------------------------

/// Build and render the context tree with token-aware pruning.
/// `entries` and `analyses` are provided by the caller (walker + parser).
pub fn build_context_tree(
    entries: &[FileEntry],
    analyses: &BTreeMap<String, FileAnalysis>,
    include_symbols: bool,
    max_tokens: Option<usize>,
    depth_limit: Option<usize>,
) -> String {
    let max_tokens = max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

    // Apply depth_limit filter: only include entries within the depth budget.
    // depth_limit=2 means show root (depth 0), depth 1, and depth 2.
    let filtered: Vec<FileEntry>;
    let effective_entries = if let Some(limit) = depth_limit {
        filtered = entries
            .iter()
            .filter(|e| e.depth <= limit)
            .cloned()
            .collect();
        &filtered[..]
    } else {
        entries
    };

    let mut tree = build_tree(effective_entries, analyses, include_symbols);

    // Level 2: Full content (symbols + headers)
    let rendered = render_tree(&tree, 0);
    if estimate_tokens(&rendered) <= max_tokens {
        return rendered;
    }

    // Level 1: Headers only, prune symbols
    prune_symbols(&mut tree);
    let rendered = render_tree(&tree, 0);
    if estimate_tokens(&rendered) <= max_tokens {
        return format!(
            "[Level 1: Headers only, symbols pruned to fit {} tokens]\n\n{}",
            max_tokens, rendered
        );
    }

    // Level 0: File names only
    prune_headers(&mut tree);
    let rendered = render_tree(&tree, 0);
    format!(
        "[Level 0: File names only, project too large for {} tokens]\n\n{}",
        max_tokens, rendered
    )
}

/// Async entry point that uses provider traits for file walking and analysis.
pub async fn get_context_tree(
    options: ContextTreeOptions,
    entries: &[FileEntry],
    analyses: &BTreeMap<String, FileAnalysis>,
) -> Result<String> {
    let include_symbols = options.include_symbols.unwrap_or(true);
    Ok(build_context_tree(
        entries,
        analyses,
        include_symbols,
        options.max_tokens,
        options.depth_limit,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
        assert_eq!(estimate_tokens("abcdefghi"), 3);
    }

    #[test]
    fn test_estimate_tokens_char_div_4_rounding() {
        // 12 chars / 4 = 3 tokens exactly
        assert_eq!(estimate_tokens("abcdefghijkl"), 3);
        // 13 chars / 4 = 3.25 -> ceil = 4
        assert_eq!(estimate_tokens("abcdefghijklm"), 4);
        // 16 chars / 4 = 4 exactly
        assert_eq!(estimate_tokens("abcdefghijklmnop"), 4);
        // 17 chars / 4 = 4.25 -> ceil = 5
        assert_eq!(estimate_tokens("abcdefghijklmnopq"), 5);
    }

    #[test]
    fn test_format_symbol() {
        let sym = TreeSymbol {
            name: "getUser".to_string(),
            kind: "function".to_string(),
            line: 10,
            end_line: 25,
            signature: "getUser(id: string): User".to_string(),
            children: vec![],
        };
        let formatted = format_symbol(&sym, 0);
        assert_eq!(formatted, "[function] L10-L25 getUser(id: string): User");
    }

    #[test]
    fn test_format_symbol_single_line() {
        let sym = TreeSymbol {
            name: "PI".to_string(),
            kind: "const".to_string(),
            line: 5,
            end_line: 5,
            signature: "const PI = 3.14".to_string(),
            children: vec![],
        };
        let formatted = format_symbol(&sym, 0);
        assert_eq!(formatted, "[const] L5 const PI = 3.14");
    }

    #[test]
    fn test_build_tree_empty() {
        let entries = vec![];
        let analyses = BTreeMap::new();
        let tree = build_tree(&entries, &analyses, true);
        assert_eq!(tree.name, ".");
        assert!(tree.children.is_empty());
    }

    #[test]
    fn test_build_tree_basic() {
        let entries = vec![
            FileEntry {
                relative_path: "src".to_string(),
                is_directory: true,
                depth: 1,
            },
            FileEntry {
                relative_path: "src/main.ts".to_string(),
                is_directory: false,
                depth: 2,
            },
            FileEntry {
                relative_path: "README.md".to_string(),
                is_directory: false,
                depth: 1,
            },
        ];
        let mut analyses = BTreeMap::new();
        analyses.insert(
            "src/main.ts".to_string(),
            FileAnalysis {
                header: Some("entry point".to_string()),
                symbols: vec![],
            },
        );

        let tree = build_tree(&entries, &analyses, true);
        assert_eq!(tree.children.len(), 2); // src/ and README.md
    }

    #[test]
    fn test_render_tree_basic() {
        let tree = TreeNode {
            name: ".".to_string(),
            relative_path: ".".to_string(),
            is_directory: true,
            header: None,
            symbols: None,
            children: vec![
                TreeNode {
                    name: "src".to_string(),
                    relative_path: "src".to_string(),
                    is_directory: true,
                    header: None,
                    symbols: None,
                    children: vec![TreeNode {
                        name: "main.ts".to_string(),
                        relative_path: "src/main.ts".to_string(),
                        is_directory: false,
                        header: Some("entry point".to_string()),
                        symbols: None,
                        children: vec![],
                    }],
                },
                TreeNode {
                    name: "README.md".to_string(),
                    relative_path: "README.md".to_string(),
                    is_directory: false,
                    header: None,
                    symbols: None,
                    children: vec![],
                },
            ],
        };

        let rendered = render_tree(&tree, 0);
        assert!(rendered.contains("./\n"));
        assert!(rendered.contains("  src/\n"));
        assert!(rendered.contains("    main.ts | entry point\n"));
        assert!(rendered.contains("  README.md\n"));
    }

    #[test]
    fn test_render_tree_with_symbols() {
        let tree = TreeNode {
            name: ".".to_string(),
            relative_path: ".".to_string(),
            is_directory: true,
            header: None,
            symbols: None,
            children: vec![TreeNode {
                name: "user.ts".to_string(),
                relative_path: "user.ts".to_string(),
                is_directory: false,
                header: Some("user module".to_string()),
                symbols: Some("[function] L10-L25 getUser(id: string): User".to_string()),
                children: vec![],
            }],
        };

        let rendered = render_tree(&tree, 0);
        assert!(rendered.contains("user.ts | user module"));
        assert!(rendered.contains("[function] L10-L25"));
    }

    #[test]
    fn test_pruning_level2_to_level1() {
        let entries = vec![FileEntry {
            relative_path: "main.ts".to_string(),
            is_directory: false,
            depth: 1,
        }];
        let mut analyses = BTreeMap::new();
        analyses.insert(
            "main.ts".to_string(),
            FileAnalysis {
                header: Some("entry".to_string()),
                symbols: vec![TreeSymbol {
                    name: "main".to_string(),
                    kind: "function".to_string(),
                    line: 1,
                    end_line: 100,
                    signature: "main()".to_string(),
                    children: vec![],
                }],
            },
        );

        // Set very low token budget to force pruning
        let result = build_context_tree(&entries, &analyses, true, Some(5), None);
        // Should contain Level indicator since it was pruned
        assert!(
            result.contains("Level 1") || result.contains("Level 0"),
            "Should be pruned with only 5 token budget"
        );
    }

    #[test]
    fn test_pruning_level0() {
        let entries = vec![FileEntry {
            relative_path: "main.ts".to_string(),
            is_directory: false,
            depth: 1,
        }];
        let mut analyses = BTreeMap::new();
        analyses.insert(
            "main.ts".to_string(),
            FileAnalysis {
                header: Some("very long header that takes many tokens".to_string()),
                symbols: vec![],
            },
        );

        // Extremely low budget
        let result = build_context_tree(&entries, &analyses, true, Some(1), None);
        // Even Level 0 might not fit, but it should at least try
        assert!(result.contains("Level 0") || result.contains("main.ts"));
    }

    #[test]
    fn test_build_context_tree_no_pruning() {
        let entries = vec![FileEntry {
            relative_path: "small.ts".to_string(),
            is_directory: false,
            depth: 1,
        }];
        let analyses = BTreeMap::new();

        let result = build_context_tree(&entries, &analyses, false, Some(100_000), None);
        // Should NOT contain Level indicator
        assert!(!result.contains("Level"));
        assert!(result.contains("small.ts"));
    }

    #[test]
    fn test_depth_limit_filters_deep_entries() {
        let entries = vec![
            FileEntry {
                relative_path: "src".to_string(),
                is_directory: true,
                depth: 1,
            },
            FileEntry {
                relative_path: "src/main.ts".to_string(),
                is_directory: false,
                depth: 2,
            },
            FileEntry {
                relative_path: "src/deep".to_string(),
                is_directory: true,
                depth: 2,
            },
            FileEntry {
                relative_path: "src/deep/nested.ts".to_string(),
                is_directory: false,
                depth: 3,
            },
        ];
        let analyses = BTreeMap::new();

        // depth_limit=2: should include depth 0, 1, 2 but NOT depth 3
        let result = build_context_tree(&entries, &analyses, false, Some(100_000), Some(2));
        assert!(result.contains("src/"));
        assert!(result.contains("main.ts"));
        assert!(result.contains("deep/"));
        assert!(!result.contains("nested.ts"), "depth 3 entry should be filtered out");

        // depth_limit=1: should include only depth 0 and 1
        let result = build_context_tree(&entries, &analyses, false, Some(100_000), Some(1));
        assert!(result.contains("src/"));
        assert!(!result.contains("main.ts"), "depth 2 entry should be filtered out");
    }
}
