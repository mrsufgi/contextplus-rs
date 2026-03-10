//! Detailed function signature extractor without reading full file bodies.
//!
//! Ports the TypeScript `file-skeleton.ts` logic:
//! - Returns structural skeleton: signatures, params, return types only
//! - Format: [kind] L{line}-L{endLine} signature;
//! - Falls back to first N lines for unsupported or symbol-less files

use std::path::PathBuf;

use crate::core::safe_path::resolve_safe_path;
use crate::error::Result;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SkeletonOptions {
    pub file_path: String,
    pub root_dir: PathBuf,
}

/// A code symbol with signature and line range info.
#[derive(Debug, Clone)]
pub struct SkeletonSymbol {
    pub name: String,
    pub kind: String,
    pub line: usize,
    pub end_line: usize,
    pub signature: String,
    pub children: Vec<SkeletonSymbol>,
}

/// File analysis result for skeleton rendering.
#[derive(Debug, Clone)]
pub struct SkeletonAnalysis {
    pub header: Option<String>,
    pub symbols: Vec<SkeletonSymbol>,
    pub line_count: usize,
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

fn format_line_range(line: usize, end_line: usize) -> String {
    if end_line > line {
        format!("L{}-L{}", line, end_line)
    } else {
        format!("L{}", line)
    }
}

/// Format the skeleton output from a file analysis.
fn format_signature_block(analysis: &SkeletonAnalysis) -> String {
    let mut lines = Vec::new();

    if let Some(ref header) = analysis.header {
        lines.push(format!("// {}", header));
        lines.push(String::new());
    }

    for sym in &analysis.symbols {
        lines.push(format!(
            "[{}] {} {};",
            sym.kind,
            format_line_range(sym.line, sym.end_line),
            sym.signature
        ));
        for child in &sym.children {
            lines.push(format!(
                "  [{}] {} {};",
                child.kind,
                format_line_range(child.line, child.end_line),
                child.signature
            ));
        }
        if !sym.children.is_empty() {
            lines.push(String::new());
        }
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// High-level entry points
// ---------------------------------------------------------------------------

/// Render file skeleton from a pre-computed analysis.
/// If `analysis` is None, the file is unsupported.
/// If `analysis` has no symbols, show preview lines.
pub fn render_skeleton(
    file_path: &str,
    analysis: Option<&SkeletonAnalysis>,
    file_content: Option<&str>,
) -> String {
    match analysis {
        None => {
            // Unsupported file type
            let preview = file_content
                .map(|content| content.lines().take(20).collect::<Vec<_>>().join("\n"))
                .unwrap_or_default();
            format!(
                "[Unsupported language, showing first 20 lines]\n\n{}",
                preview
            )
        }
        Some(analysis) if analysis.symbols.is_empty() => {
            // No symbols detected
            let preview = file_content
                .map(|content| content.lines().take(30).collect::<Vec<_>>().join("\n"))
                .unwrap_or_default();
            format!(
                "[No symbols detected, showing first 30 lines]\n\n{}",
                preview
            )
        }
        Some(analysis) => {
            let parts = [
                format!("File: {} ({} lines)", file_path, analysis.line_count),
                format!("Symbols: {} top-level definitions", analysis.symbols.len()),
                String::new(),
                format_signature_block(analysis),
            ];
            parts.join("\n")
        }
    }
}

/// Async entry point using provider traits.
/// Caller provides the analysis and file content.
/// Validates `file_path` against `root_dir` to prevent path traversal.
pub async fn get_file_skeleton(
    options: SkeletonOptions,
    analysis: Option<&SkeletonAnalysis>,
    file_content: Option<&str>,
) -> Result<String> {
    // Validate the user-supplied path — reject traversal attempts.
    resolve_safe_path(&options.root_dir, &options.file_path)?;
    Ok(render_skeleton(&options.file_path, analysis, file_content))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_line_range_single() {
        assert_eq!(format_line_range(10, 10), "L10");
    }

    #[test]
    fn test_format_line_range_multi() {
        assert_eq!(format_line_range(10, 25), "L10-L25");
    }

    #[test]
    fn test_render_skeleton_unsupported() {
        let content = "line1\nline2\nline3";
        let output = render_skeleton("data.csv", None, Some(content));
        assert!(output.contains("Unsupported language"));
        assert!(output.contains("line1"));
    }

    #[test]
    fn test_render_skeleton_no_symbols() {
        let analysis = SkeletonAnalysis {
            header: None,
            symbols: vec![],
            line_count: 50,
        };
        let content = "const x = 1;\nconst y = 2;";
        let output = render_skeleton("config.ts", Some(&analysis), Some(content));
        assert!(output.contains("No symbols detected"));
        assert!(output.contains("const x = 1;"));
    }

    #[test]
    fn test_render_skeleton_with_symbols() {
        let analysis = SkeletonAnalysis {
            header: Some("user service module".to_string()),
            symbols: vec![
                SkeletonSymbol {
                    name: "UserService".to_string(),
                    kind: "class".to_string(),
                    line: 5,
                    end_line: 50,
                    signature: "class UserService".to_string(),
                    children: vec![
                        SkeletonSymbol {
                            name: "getUser".to_string(),
                            kind: "method".to_string(),
                            line: 10,
                            end_line: 25,
                            signature: "getUser(id: string): User".to_string(),
                            children: vec![],
                        },
                        SkeletonSymbol {
                            name: "createUser".to_string(),
                            kind: "method".to_string(),
                            line: 27,
                            end_line: 45,
                            signature: "createUser(data: UserInput): User".to_string(),
                            children: vec![],
                        },
                    ],
                },
                SkeletonSymbol {
                    name: "validateEmail".to_string(),
                    kind: "function".to_string(),
                    line: 55,
                    end_line: 60,
                    signature: "validateEmail(email: string): boolean".to_string(),
                    children: vec![],
                },
            ],
            line_count: 65,
        };

        let output = render_skeleton("src/user.ts", Some(&analysis), None);
        assert!(output.contains("File: src/user.ts (65 lines)"));
        assert!(output.contains("Symbols: 2 top-level definitions"));
        assert!(output.contains("// user service module"));
        assert!(output.contains("[class] L5-L50 class UserService;"));
        assert!(output.contains("  [method] L10-L25 getUser(id: string): User;"));
        assert!(output.contains("  [method] L27-L45 createUser(data: UserInput): User;"));
        assert!(output.contains("[function] L55-L60 validateEmail(email: string): boolean;"));
    }

    #[test]
    fn test_render_skeleton_with_header_no_children() {
        let analysis = SkeletonAnalysis {
            header: Some("utilities".to_string()),
            symbols: vec![SkeletonSymbol {
                name: "add".to_string(),
                kind: "function".to_string(),
                line: 3,
                end_line: 5,
                signature: "add(a: number, b: number): number".to_string(),
                children: vec![],
            }],
            line_count: 10,
        };

        let output = render_skeleton("utils.ts", Some(&analysis), None);
        assert!(output.contains("// utilities"));
        assert!(output.contains("[function] L3-L5 add(a: number, b: number): number;"));
        assert!(output.contains("1 top-level definitions"));
    }

    #[test]
    fn test_render_skeleton_no_content_fallback() {
        let output = render_skeleton("missing.bin", None, None);
        assert!(output.contains("Unsupported language"));
    }

    #[test]
    fn test_render_skeleton_preview_truncation() {
        let lines: Vec<String> = (1..=50).map(|i| format!("line {}", i)).collect();
        let content = lines.join("\n");
        let output = render_skeleton("big.csv", None, Some(&content));
        // Should only show first 20 lines
        assert!(output.contains("line 1"));
        assert!(output.contains("line 20"));
        assert!(!output.contains("line 21"));
    }
}
