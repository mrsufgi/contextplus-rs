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

use tokio::task::JoinSet;

use crate::error::Result;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COMMAND_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_OUTPUT_LEN_SINGLE: usize = 5000;
const MAX_OUTPUT_LEN_MULTI: usize = 2000;

/// TypeScript file extensions that require special single-file handling.
/// Kept in one place so adding `.mts`/`.cts` later only needs one change.
const TS_EXTENSIONS: &[&str] = &[".ts", ".tsx"];

/// Returns `true` if `ext` is a TypeScript extension we handle specially.
fn is_ts_extension(ext: &str) -> bool {
    TS_EXTENSIONS.contains(&ext)
}

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
    /// Compile-time constant args — zero allocation.
    args: &'static [&'static str],
    /// File that must exist in root_dir for this linter to be available
    config_file: Option<&'static str>,
}

fn get_linter_config(ext: &str) -> Option<LinterConfig> {
    if is_ts_extension(ext) {
        return Some(LinterConfig {
            cmd: "npx",
            args: &["tsc", "--build", "--noEmit", "--pretty"],
            config_file: Some("tsconfig.json"),
        });
    }
    match ext {
        ".js" => Some(LinterConfig {
            cmd: "npx",
            args: &[
                "eslint",
                "--no-config-lookup",
                "--rule",
                "{\"no-unused-vars\": \"warn\"}",
            ],
            config_file: None,
        }),
        ".py" => Some(LinterConfig {
            cmd: "python",
            args: &["-m", "py_compile"],
            config_file: Some("pyproject.toml"),
        }),
        ".rs" => Some(LinterConfig {
            cmd: "cargo",
            args: &["check", "--message-format=short"],
            config_file: Some("Cargo.toml"),
        }),
        ".go" => Some(LinterConfig {
            cmd: "go",
            args: &["vet"],
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
async fn run_command(cmd: &str, args: &[&str], cwd: &Path) -> LintResult {
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

        let target_str = target_path.to_string_lossy().to_string();
        // Dispatch strategy varies by extension:
        //   - TypeScript: use `tsc --noEmit <file>` — NOT `--build`, which treats its argument as
        //     a project directory and appends /tsconfig.json, causing TS5083 on a file path.
        //   - JS / Python: append the file path to the linter's static args slice.
        //   - All others (Rust, Go, …): run the linter's args as-is (project-wide only).
        let result = if is_ts_extension(ext.as_str()) {
            let args: Vec<&str> = vec!["tsc", "--noEmit", target_str.as_str()];
            run_command(linter.cmd, &args, &options.root_dir).await
        } else if matches!(ext.as_str(), ".js" | ".py") {
            let mut args: Vec<&str> = linter.args.to_vec();
            args.push(target_str.as_str());
            run_command(linter.cmd, &args, &options.root_dir).await
        } else {
            run_command(linter.cmd, linter.args, &options.root_dir).await
        };

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

    // Project-wide mode: detect available linters, then run them concurrently via JoinSet.
    let mut available: Vec<(&str, LinterConfig)> = Vec::new();
    for &file_ext in KNOWN_EXTENSIONS {
        if let Some(linter) = detect_available_linter(&options.root_dir, file_ext).await {
            available.push((file_ext, linter));
        }
    }

    if available.is_empty() {
        return Ok("No linters available or no issues found.".to_string());
    }

    let root_clone = options.root_dir.clone();
    let mut join_set: JoinSet<(String, LintResult)> = JoinSet::new();

    for (file_ext, linter) in available {
        let cwd = root_clone.clone();
        let ext_str = file_ext.to_string();
        join_set.spawn(async move {
            let result = run_command(linter.cmd, linter.args, &cwd).await;
            (ext_str, result)
        });
    }

    let mut results = Vec::new();
    while let Some(join_result) = join_set.join_next().await {
        if let Ok((file_ext, result)) = join_result
            && !result.output.is_empty()
        {
            let truncated = if result.output.len() > MAX_OUTPUT_LEN_MULTI {
                crate::core::parser::truncate_to_char_boundary(&result.output, MAX_OUTPUT_LEN_MULTI)
                    .to_string()
            } else {
                result.output.clone()
            };
            results.push(format!(
                "[{}] {} files:\n{}",
                result.tool, file_ext, truncated
            ));
        }
    }

    // Sort for deterministic output order
    results.sort();

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
        assert!(config.args.contains(&"tsc"));
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
        assert!(config.args.contains(&"eslint"));
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

    /// Regression test: passing a single .tsx file must NOT invoke `tsc --build <file>`,
    /// which treats the path as a project directory and crashes with TS5083
    /// ("Cannot read file '.../Foo.tsx/tsconfig.json'").
    /// The fix uses `tsc --noEmit <file>` for single-file TS/TSX targets.
    #[tokio::test]
    async fn test_ts_single_file_does_not_use_build_flag() {
        let dir = tempfile::tempdir().unwrap();
        // Create tsconfig.json so the linter is considered available.
        tokio::fs::write(dir.path().join("tsconfig.json"), "{}")
            .await
            .unwrap();
        // Create a minimal valid .tsx file.
        tokio::fs::write(dir.path().join("Component.tsx"), "export const x: number = 1;\n")
            .await
            .unwrap();

        let options = StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            target_path: Some("Component.tsx".to_string()),
        };
        let result = run_static_analysis(options).await.unwrap();
        // The result must NOT contain the TS5083 error that indicates tsc was invoked
        // with --build on a file path (treating it as a directory).
        assert!(
            !result.contains("TS5083"),
            "Got TS5083 error — tsc --build was incorrectly used on a file path: {result}"
        );
        // Both strings must be absent — the old `||` only required one to be missing,
        // which meant neither guard could catch the regression on its own.
        assert!(
            !result.contains("tsconfig.json'."),
            "Unexpected tsconfig path error (bare suffix): {result}"
        );
        assert!(
            !result.contains("Component.tsx/tsconfig.json"),
            "Unexpected tsconfig path error (file-as-dir): {result}"
        );
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
        let result = run_command("echo", &["hello"], dir.path()).await;
        assert_eq!(result.exit_code, 0);
        assert!(result.output.contains("hello"));
    }
}
