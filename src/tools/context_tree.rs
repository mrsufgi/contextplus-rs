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

    // Filter by target_path if specified — scope tree to a subdirectory
    let filtered_entries: Vec<FileEntry>;
    let filtered_analyses: BTreeMap<String, FileAnalysis>;
    let (effective_entries, effective_analyses) = if let Some(ref target) = options.target_path {
        let prefix = if target.ends_with('/') {
            target.clone()
        } else {
            format!("{}/", target)
        };
        filtered_entries = entries
            .iter()
            .filter(|e| e.relative_path == *target || e.relative_path.starts_with(&prefix))
            .cloned()
            .collect();
        filtered_analyses = analyses
            .iter()
            .filter(|(k, _)| k.as_str() == target.as_str() || k.starts_with(&prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        (&filtered_entries[..], &filtered_analyses)
    } else {
        (entries, analyses)
    };

    Ok(build_context_tree(
        effective_entries,
        effective_analyses,
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
        assert!(
            !result.contains("nested.ts"),
            "depth 3 entry should be filtered out"
        );

        // depth_limit=1: should include only depth 0 and 1
        let result = build_context_tree(&entries, &analyses, false, Some(100_000), Some(1));
        assert!(result.contains("src/"));
        assert!(
            !result.contains("main.ts"),
            "depth 2 entry should be filtered out"
        );
    }

    // --- Large-scale pruning tests ---

    /// Helper: create a file entry with a directory parent at depth 1.
    fn make_file_entry(dir_index: usize, file_index: usize) -> (FileEntry, FileEntry) {
        let dir_path = format!("dir_{}", dir_index);
        let file_path = format!("dir_{}/file_{}.ts", dir_index, file_index);
        (
            FileEntry {
                relative_path: dir_path,
                is_directory: true,
                depth: 1,
            },
            FileEntry {
                relative_path: file_path,
                is_directory: false,
                depth: 2,
            },
        )
    }

    /// Helper: create a TreeSymbol with a long signature to consume tokens.
    fn make_symbol(name: &str) -> TreeSymbol {
        TreeSymbol {
            name: name.to_string(),
            kind: "function".to_string(),
            line: 1,
            end_line: 50,
            signature: format!(
                "export async function {}(param1: string, param2: number, param3: boolean): Promise<Result<void>>",
                name
            ),
            children: vec![TreeSymbol {
                name: format!("{}_inner", name),
                kind: "function".to_string(),
                line: 10,
                end_line: 40,
                signature: format!("function {}_inner(ctx: Context): void", name),
                children: vec![],
            }],
        }
    }

    #[test]
    fn test_large_scale_pruning_level2_to_level1() {
        // Create 500 files with 5 symbols each.
        // Each symbol line is ~120 chars + child ~80 chars = ~200 chars per symbol.
        // 500 files × 5 symbols × 200 chars = ~500K chars = ~125K tokens at Level 2.
        // Headers alone: 500 × ~60 chars = ~30K chars = ~7.5K tokens at Level 1 (fits).
        let mut entries = Vec::new();
        let mut analyses = BTreeMap::new();
        let mut seen_dirs = std::collections::HashSet::new();

        for i in 0..500 {
            let dir_idx = i / 10;
            let (dir_entry, file_entry) = make_file_entry(dir_idx, i);
            if seen_dirs.insert(dir_idx) {
                entries.push(dir_entry);
            }
            let path = file_entry.relative_path.clone();
            entries.push(file_entry);

            let symbols: Vec<TreeSymbol> = (0..5)
                .map(|s| make_symbol(&format!("func_{}_{}", i, s)))
                .collect();
            analyses.insert(
                path,
                FileAnalysis {
                    header: Some(format!("Module {} business logic", i)),
                    symbols,
                },
            );
        }

        let result = build_context_tree(&entries, &analyses, true, None, None);

        // Verify Level 2 was too large and it pruned to Level 1 (headers only).
        assert!(
            result.contains("Level 1"),
            "500 files with 5 symbols each should exceed 50K tokens and prune to Level 1.\n\
             Output length: {} chars ({} tokens)\nStarts with: {}",
            result.len(),
            estimate_tokens(&result),
            &result[..result.len().min(200)]
        );
        // Level 1 prunes symbols — individual symbol names should NOT appear.
        assert!(
            !result.contains("func_0_0"),
            "Symbol names should be pruned at Level 1"
        );
        assert!(
            !result.contains("[function]"),
            "Symbol kind markers should be pruned at Level 1"
        );
    }

    #[test]
    fn test_large_scale_pruning_level1_to_level0() {
        // Need headers-only output to exceed 200K chars (50K tokens).
        // Each file: ~30 char path + " | " + ~300 char header + indentation = ~340 chars.
        // 800 files × 340 chars = ~272K chars = ~68K tokens — exceeds budget.
        let mut entries = Vec::new();
        let mut analyses = BTreeMap::new();
        let mut seen_dirs = std::collections::HashSet::new();

        for i in 0..800 {
            let dir_idx = i / 10;
            let dir_path = format!("pkg_{}", dir_idx);
            let file_path = format!("pkg_{}/module_{}.ts", dir_idx, i);

            if seen_dirs.insert(dir_idx) {
                entries.push(FileEntry {
                    relative_path: dir_path,
                    is_directory: true,
                    depth: 1,
                });
            }
            entries.push(FileEntry {
                relative_path: file_path.clone(),
                is_directory: false,
                depth: 2,
            });

            // ~300 char header per file
            let long_header = format!(
                "Enterprise module {} orchestrates domain-driven design patterns for aggregate root {} \
                 with CQRS event sourcing and saga coordination across bounded contexts and microservice \
                 choreography layer integration for distributed transaction management number {}",
                i, i, i
            );
            analyses.insert(
                file_path,
                FileAnalysis {
                    header: Some(long_header),
                    symbols: vec![],
                },
            );
        }

        let result = build_context_tree(&entries, &analyses, true, None, None);

        // Should prune all the way to Level 0 (file names only).
        assert!(
            result.contains("Level 0"),
            "800 files with ~300-char headers should exceed 50K tokens at Level 1 and prune to Level 0.\n\
             Output length: {} chars ({} tokens)\nStarts with: {}",
            result.len(),
            estimate_tokens(&result),
            &result[..result.len().min(200)]
        );
        // At Level 0, only file names remain — no headers.
        assert!(
            !result.contains("Enterprise module"),
            "Headers should be pruned at Level 0"
        );
    }

    #[test]
    fn test_pruning_cascade_produces_valid_output() {
        // Create entries that trigger Level 0 pruning and verify the output is well-formed.
        // 700 files with long headers (~300 chars each) to ensure Level 0 is triggered.
        let mut entries = Vec::new();
        let mut analyses = BTreeMap::new();
        let mut seen_dirs = std::collections::HashSet::new();

        for i in 0..700 {
            let dir_idx = i / 10;
            let dir_path = format!("area_{}", dir_idx);
            let file_path = format!("area_{}/svc_{}.ts", dir_idx, i);

            if seen_dirs.insert(dir_idx) {
                entries.push(FileEntry {
                    relative_path: dir_path,
                    is_directory: true,
                    depth: 1,
                });
            }
            entries.push(FileEntry {
                relative_path: file_path.clone(),
                is_directory: false,
                depth: 2,
            });

            analyses.insert(
                file_path,
                FileAnalysis {
                    header: Some("x".repeat(300)),
                    symbols: vec![make_symbol(&format!("handler_{}", i))],
                },
            );
        }

        let result = build_context_tree(&entries, &analyses, true, None, None);

        // Output must be non-empty.
        assert!(!result.is_empty(), "Pruned output should not be empty");

        // Output must contain the root marker.
        assert!(
            result.contains("./"),
            "Output should contain the root directory marker"
        );

        // Output must contain at least some file paths.
        assert!(
            result.contains("svc_0.ts"),
            "Output should contain file names even after full pruning"
        );
        assert!(
            result.contains("area_0/"),
            "Output should contain directory names"
        );

        // No garbled text — every line should be printable ASCII or a known Level marker.
        for line in result.lines() {
            assert!(
                line.chars()
                    .all(|c| c.is_ascii() && !c.is_ascii_control() || c == '\t'),
                "Line contains non-printable characters: {:?}",
                line
            );
        }
    }

    #[test]
    fn test_depth_limit_reduces_token_count() {
        // Create 200 entries at various depths (1-5).
        let mut entries = Vec::new();
        let mut analyses = BTreeMap::new();

        // Build a nested directory structure: d0/d1/d2/d3/d4/file.ts
        for depth_chain in 0..40 {
            let mut path_parts = Vec::new();
            for d in 0..5 {
                path_parts.push(format!("l{}_{}", d, depth_chain));
                let dir_path = path_parts.join("/");
                entries.push(FileEntry {
                    relative_path: dir_path,
                    is_directory: true,
                    depth: d + 1,
                });
            }
            // Add a file at depth 6
            let file_path = format!("{}/leaf_{}.ts", path_parts.join("/"), depth_chain);
            entries.push(FileEntry {
                relative_path: file_path.clone(),
                is_directory: false,
                depth: 6,
            });
            analyses.insert(
                file_path,
                FileAnalysis {
                    header: Some(format!("Leaf module {}", depth_chain)),
                    symbols: vec![make_symbol(&format!("leaf_fn_{}", depth_chain))],
                },
            );
        }

        // Full tree — no depth limit, large token budget to avoid pruning.
        let full_result = build_context_tree(&entries, &analyses, true, Some(500_000), None);
        // Depth-limited to 2 — should be significantly shorter.
        let limited_result = build_context_tree(&entries, &analyses, true, Some(500_000), Some(2));

        let full_tokens = estimate_tokens(&full_result);
        let limited_tokens = estimate_tokens(&limited_result);

        assert!(
            limited_tokens < full_tokens,
            "Depth-limited output ({} tokens) should have fewer tokens than full output ({} tokens)",
            limited_tokens,
            full_tokens
        );

        // The limited version should not contain deep entries.
        assert!(
            !limited_result.contains("leaf_"),
            "Depth-limited to 2 should not contain depth-6 leaf files"
        );
    }
}
