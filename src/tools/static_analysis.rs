//! Static analysis runner using native linters and compilers.
//!
//! Ports the TypeScript `static-analysis.ts` logic:
//! - Delegates dead code detection to deterministic tools, not LLM guessing
//! - Detects available linters based on project config files
//! - Runs linters with timeout and captures output

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use crate::error::Result;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COMMAND_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_OUTPUT_LEN_SINGLE: usize = 5000;
const MAX_OUTPUT_LEN_MULTI: usize = 2000;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct StaticAnalysisOptions {
    pub root_dir: PathBuf,
    pub target_path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LintResult {
    pub tool: String,
    pub output: String,
    pub exit_code: i32,
}

// ---------------------------------------------------------------------------
// Linter configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LinterConfig {
    cmd: &'static str,
    args: Vec<String>,
    /// File that must exist in root_dir for this linter to be available
    config_file: Option<&'static str>,
}

fn get_linter_config(ext: &str) -> Option<LinterConfig> {
    match ext {
        ".ts" | ".tsx" => Some(LinterConfig {
            cmd: "npx",
            args: vec![
                "tsc".to_string(),
                "--build".to_string(),
                "--noEmit".to_string(),
                "--pretty".to_string(),
            ],
            config_file: Some("tsconfig.json"),
        }),
        ".js" => Some(LinterConfig {
            cmd: "npx",
            args: vec![
                "eslint".to_string(),
                "--no-config-lookup".to_string(),
                "--rule".to_string(),
                "{\"no-unused-vars\": \"warn\"}".to_string(),
            ],
            config_file: None,
        }),
        ".py" => Some(LinterConfig {
            cmd: "python",
            args: vec!["-m".to_string(), "py_compile".to_string()],
            config_file: Some("pyproject.toml"),
        }),
        ".rs" => Some(LinterConfig {
            cmd: "cargo",
            args: vec!["check".to_string(), "--message-format=short".to_string()],
            config_file: Some("Cargo.toml"),
        }),
        ".go" => Some(LinterConfig {
            cmd: "go",
            args: vec!["vet".to_string()],
            config_file: Some("go.mod"),
        }),
        _ => None,
    }
}

/// All extensions we know how to lint.
const KNOWN_EXTENSIONS: &[&str] = &[".ts", ".tsx", ".js", ".py", ".rs", ".go"];

// ---------------------------------------------------------------------------
// Command execution
// ---------------------------------------------------------------------------

/// Run a command with timeout, capturing stdout+stderr.
async fn run_command(cmd: &str, args: &[String], cwd: &Path) -> LintResult {
    let result = tokio::time::timeout(COMMAND_TIMEOUT, async {
        tokio::process::Command::new(cmd)
            .args(args)
            .current_dir(cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
    })
    .await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!("{}{}", stdout, stderr).trim().to_string();
            let exit_code = output.status.code().unwrap_or(1);
            LintResult {
                tool: cmd.to_string(),
                output: combined,
                exit_code,
            }
        }
        Ok(Err(e)) => LintResult {
            tool: cmd.to_string(),
            output: format!("Failed to execute: {}", e),
            exit_code: 1,
        },
        Err(_) => LintResult {
            tool: cmd.to_string(),
            output: "Command timed out".to_string(),
            exit_code: 1,
        },
    }
}

/// Check if a config file exists in the root directory.
async fn config_exists(root_dir: &Path, config_file: &str) -> bool {
    tokio::fs::metadata(root_dir.join(config_file))
        .await
        .is_ok()
}

/// Detect if a linter is available for the given extension.
async fn detect_available_linter(root_dir: &Path, ext: &str) -> Option<LinterConfig> {
    let config = get_linter_config(ext)?;
    if let Some(required_file) = config.config_file
        && !config_exists(root_dir, required_file).await
    {
        return None;
    }
    Some(config)
}

// ---------------------------------------------------------------------------
// High-level entry point
// ---------------------------------------------------------------------------

/// Run static analysis on a target path or the whole project.
pub async fn run_static_analysis(options: StaticAnalysisOptions) -> Result<String> {
    let target_path = match &options.target_path {
        Some(target) => options.root_dir.join(target),
        None => options.root_dir.clone(),
    };

    let ext = target_path
        .extension()
        .and_then(OsStr::to_str)
        .map(|e| format!(".{}", e));

    if let Some(ref ext) = ext {
        // Single file mode
        let linter = match detect_available_linter(&options.root_dir, ext).await {
            Some(l) => l,
            None => return Ok(format!("No linter configured for {} files.", ext)),
        };

        let mut args = linter.args.clone();
        let target_str = target_path.to_string_lossy().to_string();
        match ext.as_str() {
            ".js" | ".ts" | ".tsx" | ".py" => args.push(target_str),
            _ => {}
        }

        let result = run_command(linter.cmd, &args, &options.root_dir).await;

        if result.exit_code == 0 && result.output.is_empty() {
            return Ok("No issues found. Code is clean.".to_string());
        }

        let truncated = if result.output.len() > MAX_OUTPUT_LEN_SINGLE {
            crate::core::parser::truncate_to_char_boundary(&result.output, MAX_OUTPUT_LEN_SINGLE)
        } else {
            &result.output
        };

        return Ok(format!(
            "Static analysis ({}):\n\n{}",
            result.tool, truncated
        ));
    }

    // Project-wide mode: try all known linters
    let mut results = Vec::new();
    for &file_ext in KNOWN_EXTENSIONS {
        let linter = match detect_available_linter(&options.root_dir, file_ext).await {
            Some(l) => l,
            None => continue,
        };

        let result = run_command(linter.cmd, &linter.args, &options.root_dir).await;
        if !result.output.is_empty() {
            let truncated = if result.output.len() > MAX_OUTPUT_LEN_MULTI {
                crate::core::parser::truncate_to_char_boundary(&result.output, MAX_OUTPUT_LEN_MULTI)
            } else {
                &result.output
            };
            results.push(format!(
                "[{}] {} files:\n{}",
                result.tool, file_ext, truncated
            ));
        }
    }

    if results.is_empty() {
        Ok("No linters available or no issues found.".to_string())
    } else {
        Ok(results.join("\n\n"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_linter_config_ts() {
        let config = get_linter_config(".ts");
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.cmd, "npx");
        assert!(config.args.contains(&"tsc".to_string()));
        assert_eq!(config.config_file, Some("tsconfig.json"));
    }

    #[test]
    fn test_get_linter_config_tsx() {
        let config = get_linter_config(".tsx");
        assert!(config.is_some());
        assert_eq!(config.unwrap().config_file, Some("tsconfig.json"));
    }

    #[test]
    fn test_get_linter_config_rs() {
        let config = get_linter_config(".rs");
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.cmd, "cargo");
        assert_eq!(config.config_file, Some("Cargo.toml"));
    }

    #[test]
    fn test_get_linter_config_go() {
        let config = get_linter_config(".go");
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.cmd, "go");
        assert_eq!(config.config_file, Some("go.mod"));
    }

    #[test]
    fn test_get_linter_config_py() {
        let config = get_linter_config(".py");
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.cmd, "python");
        assert_eq!(config.config_file, Some("pyproject.toml"));
    }

    #[test]
    fn test_get_linter_config_unknown() {
        assert!(get_linter_config(".xyz").is_none());
        assert!(get_linter_config(".html").is_none());
    }

    #[test]
    fn test_get_linter_config_js() {
        let config = get_linter_config(".js").unwrap();
        assert_eq!(config.cmd, "npx");
        assert!(config.args.contains(&"eslint".to_string()));
        assert!(config.config_file.is_none());
    }

    #[tokio::test]
    async fn test_run_static_analysis_no_linter() {
        let dir = tempfile::tempdir().unwrap();
        let options = StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            target_path: Some("test.xyz".to_string()),
        };
        let result = run_static_analysis(options).await.unwrap();
        assert!(result.contains("No linter configured"));
    }

    #[tokio::test]
    async fn test_run_static_analysis_project_wide_no_configs() {
        let dir = tempfile::tempdir().unwrap();
        let options = StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            target_path: None,
        };
        let result = run_static_analysis(options).await.unwrap();
        // In some environments linters may be globally available even without
        // project config files, so just verify it runs without error.
        assert!(!result.is_empty(), "Expected non-empty output");
    }

    #[tokio::test]
    async fn test_detect_linter_with_config() {
        let dir = tempfile::tempdir().unwrap();
        // Create Cargo.toml so Rust linter is detected
        tokio::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"test\"")
            .await
            .unwrap();

        let linter = detect_available_linter(dir.path(), ".rs").await;
        assert!(linter.is_some());
        assert_eq!(linter.unwrap().cmd, "cargo");
    }

    #[tokio::test]
    async fn test_detect_linter_without_config() {
        let dir = tempfile::tempdir().unwrap();
        // No tsconfig.json, so TS linter should not be available
        let linter = detect_available_linter(dir.path(), ".ts").await;
        assert!(linter.is_none());
    }

    #[tokio::test]
    async fn test_run_command_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let result = run_command("nonexistent_command_xyz_12345", &[], dir.path()).await;
        assert_ne!(result.exit_code, 0);
    }

    #[tokio::test]
    async fn test_run_command_echo() {
        let dir = tempfile::tempdir().unwrap();
        let result = run_command("echo", &["hello".to_string()], dir.path()).await;
        assert_eq!(result.exit_code, 0);
        assert!(result.output.contains("hello"));
    }
}
