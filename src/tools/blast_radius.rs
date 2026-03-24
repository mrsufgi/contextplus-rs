//! Dependency graph analyzer to trace symbol usage across the codebase.
//!
//! Ports the TypeScript `blast-radius.ts` logic:
//! - Finds every file and line where a symbol is referenced
//! - Groups results by file with line context
//! - Excludes definition lines when fileContext matches
//! - Uses regex with escaped special chars (prevents ReDoS)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use regex::Regex;

use crate::error::Result;
use crate::tools::semantic_identifiers::{escape_regex, is_definition_line};

/// Type alias for the boxed future returned by file-line providers.
type FileLinesFuture<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<HashMap<String, Vec<String>>>> + Send + 'a>,
>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_CONTEXT_LEN: usize = 120;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BlastRadiusOptions {
    pub root_dir: PathBuf,
    pub symbol_name: String,
    pub file_context: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SymbolUsage {
    pub file: String,
    pub line: usize,
    pub context: String,
}

#[derive(Debug)]
pub struct BlastRadiusResult {
    pub by_file: HashMap<String, Vec<SymbolUsage>>,
}

impl BlastRadiusResult {
    /// Total number of usages across all files.
    pub fn total_usages(&self) -> usize {
        self.by_file.values().map(|v| v.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

/// Search for all usages of a symbol across provided file lines.
/// `file_lines` maps relative_path -> lines of the file.
/// Returns usages grouped by file.
pub fn find_symbol_usages(
    symbol_name: &str,
    file_context: Option<&str>,
    file_lines: &HashMap<String, Vec<String>>,
) -> BlastRadiusResult {
    let escaped = escape_regex(symbol_name);
    // \b only works at word-character boundaries. For symbols with non-word
    // chars (e.g. $transaction), skip the boundary on that side.
    let first_is_word = symbol_name.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_');
    let last_is_word = symbol_name.ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_');
    let prefix = if first_is_word { r"\b" } else { "" };
    let suffix = if last_is_word { r"\b" } else { "" };
    let pattern_str = format!("{}{}{}", prefix, escaped, suffix);
    let symbol_pattern = match Regex::new(&pattern_str) {
        Ok(re) => re,
        Err(_) => {
            return BlastRadiusResult {
                by_file: HashMap::new(),
            };
        }
    };

    let mut by_file: HashMap<String, Vec<SymbolUsage>> = HashMap::new();

    for (relative_path, lines) in file_lines {
        for (i, line) in lines.iter().enumerate() {
            if !symbol_pattern.is_match(line) {
                continue;
            }

            // Skip definition line if this file matches fileContext
            if let Some(ctx) = file_context
                && relative_path == ctx
                && is_definition_line(line, symbol_name)
            {
                continue;
            }

            let context = line.trim();
            let context = if context.len() > MAX_CONTEXT_LEN {
                crate::core::parser::truncate_to_char_boundary(context, MAX_CONTEXT_LEN)
            } else {
                context
            };
            by_file
                .entry(relative_path.clone())
                .or_default()
                .push(SymbolUsage {
                    file: relative_path.clone(),
                    line: i + 1,
                    context: context.to_string(),
                });
        }
    }

    BlastRadiusResult { by_file }
}

/// Format blast radius results as text output (matching TS format).
pub fn format_blast_radius(symbol_name: &str, result: &BlastRadiusResult) -> String {
    let total = result.total_usages();
    if total == 0 {
        return format!(
            "Symbol \"{}\" is not used anywhere in the codebase.",
            symbol_name
        );
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "Blast radius for \"{}\": {} usages in {} files\n",
        symbol_name,
        total,
        result.by_file.len()
    ));

    // Sort files for deterministic output
    let mut sorted_files: Vec<&String> = result.by_file.keys().collect();
    sorted_files.sort();
    for file in sorted_files {
        let file_usages = &result.by_file[file];
        lines.push(format!("  {}:", file));
        for u in file_usages {
            lines.push(format!("    L{}: {}", u.line, u.context));
        }
    }

    if total <= 1 {
        lines.push(format!(
            "\n\u{26A0} LOW USAGE: This symbol is used only {} time(s). Consider inlining if it's under 20 lines.",
            total
        ));
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// High-level entry point
// ---------------------------------------------------------------------------

/// Trait for providing file lines. Implementations can use cached file lines
/// from the identifier index or read files from disk.
pub trait FileLineProvider: Send + Sync {
    fn get_file_lines(&self, root_dir: &Path) -> FileLinesFuture<'_>;
}

/// Run blast radius analysis.
pub async fn get_blast_radius(
    options: BlastRadiusOptions,
    file_line_provider: &dyn FileLineProvider,
) -> Result<String> {
    let symbol_name = options.symbol_name.trim();
    if symbol_name.is_empty() {
        return Ok("Symbol name is empty.".to_string());
    }

    let file_lines = file_line_provider.get_file_lines(&options.root_dir).await?;

    let result = find_symbol_usages(symbol_name, options.file_context.as_deref(), &file_lines);

    Ok(format_blast_radius(symbol_name, &result))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_file_lines(data: Vec<(&str, Vec<&str>)>) -> HashMap<String, Vec<String>> {
        data.into_iter()
            .map(|(path, lines)| {
                (
                    path.to_string(),
                    lines.into_iter().map(|l| l.to_string()).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn test_find_usages_basic() {
        let file_lines = make_file_lines(vec![
            (
                "src/user.ts",
                vec![
                    "export function getUserById(id: string) {",
                    "  return db.find(id);",
                    "}",
                ],
            ),
            (
                "src/handler.ts",
                vec![
                    "import { getUserById } from './user';",
                    "const user = getUserById(req.params.id);",
                    "console.log(user);",
                ],
            ),
        ]);

        let result = find_symbol_usages("getUserById", Some("src/user.ts"), &file_lines);

        // Definition in user.ts should be excluded
        // Usage in handler.ts should be included (import + call)
        assert!(result.total_usages() > 0);
        assert!(result.by_file.contains_key("src/handler.ts"));
        // The definition line in user.ts should be excluded
        let user_usages = result.by_file.get("src/user.ts");
        assert!(
            user_usages.is_none() || user_usages.unwrap().is_empty(),
            "Definition line should be excluded from same-file usages"
        );
    }

    #[test]
    fn test_find_usages_no_file_context() {
        let file_lines = make_file_lines(vec![(
            "src/user.ts",
            vec!["export function myFunc() {", "  myFunc();"],
        )]);

        // Without file_context, definition is not excluded
        let result = find_symbol_usages("myFunc", None, &file_lines);
        assert_eq!(result.total_usages(), 2);
    }

    #[test]
    fn test_find_usages_no_matches() {
        let file_lines = make_file_lines(vec![(
            "src/app.ts",
            vec!["const x = 42;", "console.log(x);"],
        )]);

        let result = find_symbol_usages("nonExistent", None, &file_lines);
        assert_eq!(result.total_usages(), 0);
    }

    #[test]
    fn test_find_usages_word_boundary() {
        let file_lines = make_file_lines(vec![(
            "src/app.ts",
            vec!["getUserById(1);", "getUser(2);", "getUserByIdAndName(3);"],
        )]);

        // "getUser" should match "getUser(2);" but NOT "getUserById" or "getUserByIdAndName"
        let result = find_symbol_usages("getUser", None, &file_lines);
        assert_eq!(result.total_usages(), 1);
        assert_eq!(result.by_file.values().next().unwrap()[0].line, 2);
    }

    #[test]
    fn test_find_usages_special_chars() {
        let file_lines = make_file_lines(vec![(
            "src/app.ts",
            vec!["$transaction.commit();", "const $transaction = db.begin();"],
        )]);

        // Dollar sign should be escaped properly
        let result = find_symbol_usages("$transaction", None, &file_lines);
        assert_eq!(result.total_usages(), 2);
    }

    #[test]
    fn test_format_no_usages() {
        let result = BlastRadiusResult {
            by_file: HashMap::new(),
        };
        let output = format_blast_radius("myFunc", &result);
        assert_eq!(
            output,
            "Symbol \"myFunc\" is not used anywhere in the codebase."
        );
    }

    #[test]
    fn test_format_single_usage_low_warning() {
        let usage = SymbolUsage {
            file: "src/handler.ts".to_string(),
            line: 5,
            context: "const x = myFunc();".to_string(),
        };
        let mut by_file = HashMap::new();
        by_file.insert("src/handler.ts".to_string(), vec![usage]);
        let result = BlastRadiusResult { by_file };
        let output = format_blast_radius("myFunc", &result);
        assert!(output.contains("LOW USAGE"));
        assert!(output.contains("1 time(s)"));
    }

    #[test]
    fn test_format_multiple_files() {
        let mut by_file = HashMap::new();
        by_file.insert(
            "src/a.ts".to_string(),
            vec![
                SymbolUsage {
                    file: "src/a.ts".to_string(),
                    line: 1,
                    context: "use myFunc".to_string(),
                },
                SymbolUsage {
                    file: "src/a.ts".to_string(),
                    line: 5,
                    context: "call myFunc".to_string(),
                },
            ],
        );
        by_file.insert(
            "src/b.ts".to_string(),
            vec![SymbolUsage {
                file: "src/b.ts".to_string(),
                line: 3,
                context: "import myFunc".to_string(),
            }],
        );
        let result = BlastRadiusResult { by_file };
        let output = format_blast_radius("myFunc", &result);
        assert!(output.contains("3 usages in 2 files"));
        assert!(output.contains("src/a.ts:"));
        assert!(output.contains("src/b.ts:"));
        // Should NOT have low usage warning
        assert!(!output.contains("LOW USAGE"));
    }

    #[test]
    fn test_context_truncation() {
        let long_line = "a".repeat(200);
        let file_lines: HashMap<String, Vec<String>> = [(
            "src/app.ts".to_string(),
            vec![format!("myFunc {}", long_line)],
        )]
        .into_iter()
        .collect();
        let result = find_symbol_usages("myFunc", None, &file_lines);
        assert_eq!(result.total_usages(), 1);
        assert!(result.by_file.values().next().unwrap()[0].context.len() <= MAX_CONTEXT_LEN);
    }

    #[test]
    fn low_usage_warning_uses_warning_emoji_not_ascii() {
        let usage = SymbolUsage {
            file: "src/handler.ts".to_string(),
            line: 5,
            context: "const x = myFunc();".to_string(),
        };
        let mut by_file = HashMap::new();
        by_file.insert("src/handler.ts".to_string(), vec![usage]);
        let result = BlastRadiusResult { by_file };
        let output = format_blast_radius("myFunc", &result);
        assert!(
            output.contains("\u{26A0} LOW USAGE"),
            "Low usage warning should use warning emoji, got: {}",
            output
        );
        assert!(
            !output.contains("!! LOW USAGE"),
            "Should not contain ASCII !! prefix"
        );
    }
}
