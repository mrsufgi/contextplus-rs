// Code commit gatekeeper enforcing 2-line headers and abstraction limits
// Validates file content and creates shadow restore points before writing

use crate::error::{ContextPlusError, Result};
use crate::git::shadow::create_restore_point;
use std::path::Path;
use tokio::fs;

/// Validates that a user-provided path stays within the root directory.
fn validate_path(root: &Path, user_path: &str) -> Result<std::path::PathBuf> {
    let joined = root.join(user_path);
    // Normalize without requiring the file to exist (no canonicalize)
    let mut normalized = std::path::PathBuf::new();
    for component in joined.components() {
        match component {
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return Err(ContextPlusError::Other(format!(
                        "Path traversal detected: {user_path}"
                    )));
                }
            }
            std::path::Component::Normal(c) => normalized.push(c),
            std::path::Component::RootDir => normalized.push("/"),
            std::path::Component::Prefix(p) => normalized.push(p.as_os_str()),
            std::path::Component::CurDir => {}
        }
    }
    // Verify it's still under root
    let root_canonical = if root.is_absolute() {
        root.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(root)
    };
    if !normalized.starts_with(&root_canonical) {
        return Err(ContextPlusError::Other(format!(
            "Path traversal detected: {user_path}"
        )));
    }
    Ok(normalized)
}

const MAX_NESTING_DEPTH: usize = 6;
const MAX_LINE_COUNT: usize = 1000;

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
struct ValidationError {
    rule: String,
    message: String,
}

/// Maps file extensions to their single-line comment prefix.
fn comment_prefix_for_ext(ext: &str) -> Option<&'static str> {
    match ext {
        "ts" | "tsx" | "js" | "jsx" | "rs" | "go" | "c" | "cpp" | "java" | "cs" | "swift"
        | "kt" | "zig" => Some("//"),
        "py" | "rb" => Some("#"),
        "lua" => Some("--"),
        _ => None,
    }
}

/// Returns the file extension without the leading dot, lowercase.
fn file_ext(path: &str) -> String {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
}

/// Checks whether the file type supports header/comment validation.
fn is_supported_file(path: &str) -> bool {
    comment_prefix_for_ext(&file_ext(path)).is_some()
}

/// Validates that the first 2 non-empty lines within the first 5 lines are comments.
fn validate_header(lines: &[&str], ext: &str) -> Vec<ValidationError> {
    let prefix = match comment_prefix_for_ext(ext) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let header_lines: Vec<&&str> = lines
        .iter()
        .take(5)
        .filter(|l| l.starts_with(prefix))
        .collect();

    let mut errors = Vec::new();
    if header_lines.len() < 2 {
        errors.push(ValidationError {
            rule: "header".to_string(),
            message: format!(
                "Missing 2-line file header. First 2 lines must be {} comments explaining the file.",
                prefix
            ),
        });
    }
    errors
}

/// Validates that no standalone comment lines appear after the header (line 3+).
/// Only full-line comments are flagged, not inline comments in code.
fn validate_no_inline_comments(lines: &[&str], ext: &str) -> Vec<ValidationError> {
    let is_script_lang = matches!(ext, "py" | "rb");
    let prefix = if is_script_lang { "#" } else { "//" };

    let mut errors = Vec::new();
    for (i, line) in lines.iter().enumerate().skip(2) {
        let trimmed = line.trim();
        if trimmed.starts_with(prefix)
            && !trimmed.starts_with("#!")
            && !trimmed.starts_with("#include")
        {
            errors.push(ValidationError {
                rule: "no-comments".to_string(),
                message: format!(
                    "Unauthorized comment found on line {}: {}",
                    i + 1,
                    crate::core::parser::truncate_to_char_boundary(trimmed, 80)
                ),
            });
        }
    }
    errors
}

/// Validates nesting depth (max 6) and file length (max 1000 lines).
fn validate_abstraction(lines: &[&str]) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let mut depth: i32 = 0;
    let mut max_depth: i32 = 0;

    for line in lines {
        depth += line.matches('{').count() as i32;
        depth -= line.matches('}').count() as i32;
        if depth > max_depth {
            max_depth = depth;
        }
    }

    if max_depth > MAX_NESTING_DEPTH as i32 {
        errors.push(ValidationError {
            rule: "nesting".to_string(),
            message: format!(
                "Nesting depth of {} detected. Maximum recommended is 3-4 levels. Flatten the structure.",
                max_depth
            ),
        });
    }

    if lines.len() > MAX_LINE_COUNT {
        errors.push(ValidationError {
            rule: "file-length".to_string(),
            message: format!(
                "File is {} lines. Maximum recommended is 500-1000. Consider splitting.",
                lines.len()
            ),
        });
    }

    errors
}

/// Validates file content, creates a restore point, and writes the file.
///
/// Returns a summary string with validation results and restore point info.
pub async fn propose_commit(
    root_dir: &Path,
    file_path: &str,
    content: &str,
    description: Option<&str>,
) -> Result<String> {
    let ext = file_ext(file_path);
    let lines: Vec<&str> = content.lines().collect();
    let mut all_errors: Vec<ValidationError> = Vec::new();

    if is_supported_file(file_path) {
        all_errors.extend(validate_header(&lines, &ext));
        all_errors.extend(validate_no_inline_comments(&lines, &ext));
    }
    all_errors.extend(validate_abstraction(&lines));

    // If too many comment violations, reject outright
    let comment_errors: Vec<&ValidationError> = all_errors
        .iter()
        .filter(|e| e.rule == "no-comments")
        .collect();

    if comment_errors.len() > 5 {
        let mut result = vec![format!(
            "REJECTED: {} violations found.\n",
            all_errors.len()
        )];
        for e in all_errors.iter().take(10) {
            result.push(format!("  [{}] {}", e.rule, e.message));
        }
        if all_errors.len() > 10 {
            result.push(format!(
                "  ... and {} more violations",
                all_errors.len() - 10
            ));
        }
        result.push("\nFix all violations and resubmit.".to_string());
        return Ok(result.join("\n"));
    }

    // Warnings = everything that isn't a blocking comment error
    let warnings: Vec<&ValidationError> = all_errors
        .iter()
        .filter(|e| e.rule != "no-comments" || comment_errors.len() <= 5)
        .collect();

    // Create restore point before writing
    let default_desc = format!("Pre-commit: {}", file_path);
    let desc = description.unwrap_or(&default_desc);
    let rp = create_restore_point(root_dir, &[file_path], desc).await?;

    // Validate path stays within root
    let full_path = validate_path(root_dir, file_path)?;
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::write(&full_path, content).await?;

    // Build result message
    let mut result = vec![format!("File saved: {}", file_path)];

    if !warnings.is_empty() {
        result.push(format!("\n{} warning(s):", warnings.len()));
        for w in &warnings {
            result.push(format!("  [{}] {}", w.rule, w.message));
        }
    }

    result.push(format!(
        "\nRestore point created: {}. Use undo tools if needed.",
        rp.id
    ));

    Ok(result.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn validate_header_valid_two_lines() {
        let lines = vec!["// First line", "// Second line", "code here"];
        let errors = validate_header(&lines, "ts");
        assert!(errors.is_empty());
    }

    #[test]
    fn validate_header_missing_returns_error() {
        let lines = vec!["let x = 1;", "let y = 2;"];
        let errors = validate_header(&lines, "ts");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rule, "header");
        assert!(errors[0].message.contains("Missing 2-line"));
    }

    #[test]
    fn validate_header_one_comment_line_insufficient() {
        let lines = vec!["// Only one", "code here"];
        let errors = validate_header(&lines, "rs");
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn validate_no_inline_comments_clean_code() {
        let lines = vec!["// header 1", "// header 2", "fn main() {}", "let x = 1;"];
        let errors = validate_no_inline_comments(&lines, "rs");
        assert!(errors.is_empty());
    }

    #[test]
    fn validate_no_inline_comments_flags_comment_after_header() {
        let lines = vec![
            "// header 1",
            "// header 2",
            "fn main() {}",
            "  // rogue comment",
            "let x = 1;",
        ];
        let errors = validate_no_inline_comments(&lines, "rs");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rule, "no-comments");
        assert!(errors[0].message.contains("line 4"));
    }

    #[test]
    fn nesting_depth_flat_code() {
        let lines = vec!["let a = 1;", "let b = 2;", "let c = 3;"];
        let errors = validate_abstraction(&lines);
        assert!(errors.is_empty());
    }

    #[test]
    fn nesting_depth_deeply_nested() {
        let lines = vec![
            "{", "{", "{", "{", "{", "{", "{", "}", "}", "}", "}", "}", "}", "}",
        ];
        let errors = validate_abstraction(&lines);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rule, "nesting");
        assert!(errors[0].message.contains("7"));
    }

    #[test]
    fn nesting_depth_exactly_at_limit() {
        // Depth of exactly 6 should pass (MAX_NESTING_DEPTH = 6)
        let lines = vec!["{", "{", "{", "{", "{", "{", "}", "}", "}", "}", "}", "}"];
        let errors = validate_abstraction(&lines);
        assert!(errors.is_empty());
    }

    #[test]
    fn header_valid_when_two_comment_lines() {
        let lines = vec!["// Line one", "// Line two", "fn main() {}"];
        let errors = validate_header(&lines, "rs");
        assert!(errors.is_empty());
    }

    #[test]
    fn header_invalid_when_missing() {
        let lines = vec!["fn main() {}", "  println!(\"hello\");", "}"];
        let errors = validate_header(&lines, "rs");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rule, "header");
    }

    #[test]
    fn header_skipped_for_unknown_ext() {
        let lines = vec!["no comment here"];
        let errors = validate_header(&lines, "xyz");
        assert!(errors.is_empty());
    }

    #[test]
    fn header_python_uses_hash() {
        let lines = vec!["# line one", "# line two", "def main():"];
        let errors = validate_header(&lines, "py");
        assert!(errors.is_empty());
    }

    #[test]
    fn inline_comments_detected() {
        let lines = vec![
            "// header 1",
            "// header 2",
            "  // this is bad",
            "code here",
        ];
        let errors = validate_no_inline_comments(&lines, "rs");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rule, "no-comments");
    }

    #[test]
    fn inline_comments_ignore_shebang() {
        let lines = vec!["# header 1", "# header 2", "#!/usr/bin/env python", "code"];
        let errors = validate_no_inline_comments(&lines, "py");
        assert!(errors.is_empty());
    }

    #[test]
    fn nesting_depth_within_limit() {
        let lines = vec![
            "fn a() {",
            "  if x {",
            "    if y {",
            "      if z {",
            "      }",
            "    }",
            "  }",
            "}",
        ];
        let errors = validate_abstraction(&lines);
        assert!(errors.is_empty(), "depth 4 should be fine");
    }

    #[test]
    fn nesting_depth_exceeds_limit() {
        // Create nesting of 7
        let lines = vec![
            "{", // 1
            "{", // 2
            "{", // 3
            "{", // 4
            "{", // 5
            "{", // 6
            "{", // 7
            "}", "}", "}", "}", "}", "}", "}",
        ];
        let errors = validate_abstraction(&lines);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rule, "nesting");
    }

    #[test]
    fn file_length_over_1000_flagged() {
        let lines: Vec<&str> = (0..1001).map(|_| "line").collect();
        let errors = validate_abstraction(&lines);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rule, "file-length");
    }

    #[test]
    fn file_length_at_1000_ok() {
        let lines: Vec<&str> = (0..1000).map(|_| "line").collect();
        let errors = validate_abstraction(&lines);
        assert!(errors.is_empty());
    }

    #[test]
    fn is_supported_file_recognizes_extensions() {
        assert!(is_supported_file("main.rs"));
        assert!(is_supported_file("app.ts"));
        assert!(is_supported_file("script.py"));
        assert!(is_supported_file("main.go"));
        assert!(!is_supported_file("data.json"));
        assert!(!is_supported_file("readme.md"));
    }

    #[test]
    fn file_ext_handles_edge_cases() {
        assert_eq!(file_ext("main.RS"), "rs");
        assert_eq!(file_ext("no_ext"), "");
        assert_eq!(file_ext("path/to/file.tsx"), "tsx");
    }

    #[tokio::test]
    async fn propose_commit_creates_file_and_restore_point() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        let content = "// File description here\n// FEATURE: Test\nfn main() {}\n";
        let result = propose_commit(root, "src/main.rs", content, Some("test write"))
            .await
            .unwrap();

        assert!(result.contains("File saved: src/main.rs"));
        assert!(result.contains("Restore point created:"));

        // Verify file was written
        let written = tokio::fs::read_to_string(root.join("src/main.rs"))
            .await
            .unwrap();
        assert_eq!(written, content);
    }

    #[tokio::test]
    async fn propose_commit_warns_on_missing_header() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        let content = "fn main() {}\n";
        let result = propose_commit(root, "app.rs", content, None).await.unwrap();

        assert!(result.contains("File saved:"));
        assert!(result.contains("warning(s):"));
        assert!(result.contains("[header]"));
    }

    #[tokio::test]
    async fn propose_commit_rejects_too_many_comments() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        let mut lines = vec!["// header 1", "// header 2"];
        for _ in 0..10 {
            lines.push("// unauthorized comment");
        }
        lines.push("fn main() {}");
        let content = lines.join("\n");

        let result = propose_commit(root, "bad.rs", &content, None)
            .await
            .unwrap();

        assert!(result.contains("REJECTED"));
    }

    #[tokio::test]
    async fn propose_commit_backs_up_existing_file() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Create original file
        tokio::fs::write(root.join("existing.rs"), "// old\n// file\nold code")
            .await
            .unwrap();

        let new_content = "// new\n// file\nnew code";
        let result = propose_commit(root, "existing.rs", new_content, None)
            .await
            .unwrap();

        assert!(result.contains("File saved:"));

        // File should now have new content
        let content = tokio::fs::read_to_string(root.join("existing.rs"))
            .await
            .unwrap();
        assert_eq!(content, new_content);

        // Restore point should exist
        let points = crate::git::shadow::list_restore_points(root).await.unwrap();
        assert_eq!(points.len(), 1);
    }

    #[tokio::test]
    async fn propose_commit_non_supported_file_skips_header_validation() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        let content = "{ \"key\": \"value\" }\n";
        let result = propose_commit(root, "data.json", content, None)
            .await
            .unwrap();

        // Should succeed without header warnings for JSON
        assert!(result.contains("File saved:"));
        assert!(!result.contains("[header]"));
    }

    #[test]
    fn validate_path_blocks_traversal() {
        let root = Path::new("/tmp/project");
        assert!(validate_path(root, "../../etc/passwd").is_err());
        assert!(validate_path(root, "../../../root/.ssh/id_rsa").is_err());
    }

    #[test]
    fn validate_path_allows_normal_paths() {
        let root = Path::new("/tmp/project");
        assert!(validate_path(root, "src/main.rs").is_ok());
        assert!(validate_path(root, "nested/deep/file.ts").is_ok());
    }

    #[tokio::test]
    async fn propose_commit_rejects_path_traversal() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let content = "// header\n// line2\nmalicious content";
        let result = propose_commit(root, "../../etc/evil", content, None).await;
        assert!(result.is_err() || result.unwrap().contains("traversal"));
    }
}
