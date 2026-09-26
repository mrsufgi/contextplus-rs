//! Static analysis runner using native linters and compilers.
//!
//! Ports the TypeScript `static-analysis.ts` logic:
//! - Delegates dead code detection to deterministic tools, not LLM guessing
//! - Detects available linters based on project config files
//! - Runs linters with timeout and captures output

use std::ffi::{OsStr, OsString};
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
    /// Executable search path; None uses the process PATH.
    pub executable_path: Option<OsString>,
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
            args: &[
                "tsc",
                "-p",
                "tsconfig.json",
                "--noEmit",
                "--pretty",
                "false",
            ],
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
const KNOWN_EXTENSIONS: &[&str] = &[".ts", ".js", ".py", ".rs", ".go"];

// ---------------------------------------------------------------------------
// Command execution
// ---------------------------------------------------------------------------

/// Run a command with timeout, capturing stdout+stderr.
#[cfg(test)]
async fn run_command(cmd: &str, args: &[&str], cwd: &Path) -> LintResult {
    run_command_with_timeout(cmd, args, cwd, COMMAND_TIMEOUT, OsStr::new("/usr/bin:/bin")).await
}

async fn run_command_with_timeout(
    cmd: &str,
    args: &[&str],
    cwd: &Path,
    command_timeout: Duration,
    executable_path: &OsStr,
) -> LintResult {
    let result = tokio::time::timeout(command_timeout, async {
        let mut command = tokio::process::Command::new(cmd);
        command
            .args(args)
            .current_dir(cwd)
            .env("PATH", executable_path)
            .env("NO_COLOR", "1")
            .env("npm_config_offline", "true")
            .kill_on_drop(true)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        command.process_group(0);
        let child = command.spawn()?;
        #[cfg(unix)]
        let mut group = ProcessGroup(Some(
            child.id().expect("newly spawned child has a PID") as i32
        ));
        let output = child.wait_with_output().await;
        #[cfg(unix)]
        {
            group.0 = None;
        }
        output
    })
    .await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = strip_ansi(&format!("{}{}", stdout, stderr))
                .trim()
                .to_string();
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
            output: format!(
                "{} {} timed out after {} ms",
                cmd,
                args.join(" "),
                command_timeout.as_millis()
            ),
            exit_code: 1,
        },
    }
}

#[cfg(unix)]
struct ProcessGroup(Option<i32>);

#[cfg(unix)]
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        // Negative PID addresses the isolated group, including wrapper descendants.
        // Tokio's child drop handles reaping the direct child on cancellation.
        if let Some(pid) = self.0 {
            unsafe { libc::kill(-pid, libc::SIGKILL) };
        }
    }
}

fn strip_ansi(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('[') => {
                for ch in chars.by_ref() {
                    if ('@'..='~').contains(&ch) {
                        break;
                    }
                }
            }
            Some(']') => {
                while let Some(ch) = chars.next() {
                    if ch == '\u{7}' || (ch == '\u{1b}' && chars.next() == Some('\\')) {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

async fn project_dir(root: &Path, target: &Path) -> PathBuf {
    let mut dir = if target.is_dir() {
        target
    } else {
        target.parent().unwrap_or(root)
    };
    while dir.starts_with(root) {
        if config_exists(dir, "tsconfig.json").await || config_exists(dir, "package.json").await {
            return dir.to_path_buf();
        }
        if dir == root {
            break;
        }
        let Some(parent) = dir.parent() else {
            break;
        };
        dir = parent;
    }
    root.to_path_buf()
}

fn local_node_tool(cwd: &Path, root: &Path, name: &str) -> Option<PathBuf> {
    for dir in cwd.ancestors().take_while(|dir| dir.starts_with(root)) {
        let path = dir.join("node_modules/.bin").join(name);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn node_tool(cwd: &Path, root: &Path, name: &str, executable_path: &OsStr) -> Option<PathBuf> {
    local_node_tool(cwd, root, name).or_else(|| path_tool(name, executable_path))
}

fn path_tool(name: &str, executable_path: &OsStr) -> Option<PathBuf> {
    std::env::split_paths(executable_path)
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
}

fn has_project_references(config: &str) -> bool {
    let mut json = config.trim_start_matches('\u{feff}').as_bytes().to_vec();
    let mut i = 0;
    let mut quoted = false;
    while i < json.len() {
        match json[i] {
            b'\\' if quoted => i += 1,
            b'"' => quoted = !quoted,
            b'/' if !quoted && json.get(i + 1) == Some(&b'/') => {
                while i < json.len() && json[i] != b'\n' {
                    json[i] = b' ';
                    i += 1;
                }
                continue;
            }
            b'/' if !quoted && json.get(i + 1) == Some(&b'*') => {
                json[i..i + 2].fill(b' ');
                i += 2;
                while i + 1 < json.len() && &json[i..i + 2] != b"*/" {
                    json[i] = b' ';
                    i += 1;
                }
                if i + 1 < json.len() {
                    json[i..i + 2].fill(b' ');
                    i += 2;
                }
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    // JSONC permits trailing commas; preserve commas inside strings.
    i = 0;
    quoted = false;
    while i < json.len() {
        match json[i] {
            b'\\' if quoted => i += 1,
            b'"' => quoted = !quoted,
            b',' if !quoted => {
                if matches!(
                    json[i + 1..].iter().find(|b| !b.is_ascii_whitespace()),
                    Some(b'}' | b']')
                ) {
                    json[i] = b' ';
                }
            }
            _ => {}
        }
        i += 1;
    }
    serde_json::from_slice::<serde_json::Value>(&json)
        .ok()
        .and_then(|value| value.get("references").map(serde_json::Value::is_array))
        .unwrap_or(false)
}

async fn typescript_dir(root: &Path, cwd: &Path) -> PathBuf {
    for dir in cwd.ancestors().take_while(|dir| dir.starts_with(root)) {
        if config_exists(dir, "tsconfig.json").await {
            return dir.to_path_buf();
        }
    }
    cwd.to_path_buf()
}

async fn run_linter(
    linter: &LinterConfig,
    ext: &str,
    cwd: &Path,
    root: &Path,
    target: Option<&Path>,
    command_timeout: Duration,
    executable_path: &OsStr,
) -> LintResult {
    let mut cmd = linter.cmd.to_string();
    let mut args: Vec<String> = linter.args.iter().map(|arg| arg.to_string()).collect();
    if is_ts_extension(ext) {
        let config = tokio::fs::read_to_string(cwd.join("tsconfig.json"))
            .await
            .unwrap_or_default();
        if has_project_references(&config) {
            args = ["tsc", "--build", "--pretty", "false"]
                .map(str::to_string)
                .to_vec();
        }
    }
    if is_ts_extension(ext) || ext == ".js" {
        let tool = if is_ts_extension(ext) {
            "tsc"
        } else {
            "eslint"
        };
        let selected = if ext == ".js" {
            local_node_tool(cwd, root, "oxlint")
                .map(|path| (path, true))
                .or_else(|| local_node_tool(cwd, root, "eslint").map(|path| (path, false)))
                .or_else(|| path_tool("oxlint", executable_path).map(|path| (path, true)))
                .or_else(|| path_tool("eslint", executable_path).map(|path| (path, false)))
        } else {
            node_tool(cwd, root, tool, executable_path).map(|path| (path, false))
        };
        if let Some((path, oxlint)) = selected {
            cmd = path.to_string_lossy().into_owned();
            if oxlint {
                args.clear();
            } else {
                args.remove(0);
            }
        } else {
            args.insert(0, "--no-install".to_string());
        }
        if ext == ".js" {
            args.push(
                target
                    .unwrap_or(Path::new("."))
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    } else if ext == ".py"
        && let Some(target) = target
    {
        args.push(target.to_string_lossy().into_owned());
    }
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let mut result =
        run_command_with_timeout(&cmd, &args, cwd, command_timeout, executable_path).await;
    if is_ts_extension(ext) {
        result.output = strip_node_modules_diagnostics(&result.output);
    }
    result
}

/// Strip diagnostics whose source location lives in `node_modules/`.
///
/// Even with `--skipLibCheck`, tsc occasionally surfaces dep-internal
/// errors when a downstream package imports a type with a real bug.
/// Those errors are noise from the user's perspective: they can't fix
/// dep code from their workspace. This pass drops any line that begins
/// with a path containing `node_modules/`. Multi-line diagnostics
/// (header line + indented detail/code-frame lines) are kept or
/// dropped together: when a header line is dropped, all following
/// indented lines until the next non-indented line are dropped too.
fn strip_node_modules_diagnostics(raw: &str) -> String {
    let mut out = Vec::with_capacity(raw.len());
    let mut dropping = false;
    for line in raw.lines() {
        let starts_indented = line.starts_with(' ') || line.starts_with('\t');
        if !starts_indented {
            // New header line — decide drop/keep.
            dropping = line.contains("node_modules/") && !line.contains("timed out after");
        }
        if !dropping {
            out.push(line);
        }
    }
    out.join("\n").trim_end().to_string()
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
    run_static_analysis_with_timeout(options, COMMAND_TIMEOUT).await
}

async fn run_static_analysis_with_timeout(
    options: StaticAnalysisOptions,
    command_timeout: Duration,
) -> Result<String> {
    let executable_path = options
        .executable_path
        .clone()
        .unwrap_or_else(|| std::env::var_os("PATH").unwrap_or_default());
    let target_path = match &options.target_path {
        Some(target) => options.root_dir.join(target),
        None => options.root_dir.clone(),
    };

    let cwd = if options.target_path.is_some() {
        project_dir(&options.root_dir, &target_path).await
    } else {
        options.root_dir.clone()
    };

    let ts_cwd = typescript_dir(&options.root_dir, &cwd).await;

    let ext = target_path
        .extension()
        .and_then(OsStr::to_str)
        .map(|e| format!(".{}", e));

    if let Some(ref ext) = ext.filter(|_| !target_path.is_dir()) {
        let cwd = if is_ts_extension(ext) { &ts_cwd } else { &cwd };
        // Single file mode
        let linter = match detect_available_linter(cwd, ext).await {
            Some(l) => l,
            None => return Ok(format!("No linter configured for {} files.", ext)),
        };

        let result = run_linter(
            &linter,
            ext,
            cwd,
            &options.root_dir,
            Some(&target_path),
            command_timeout,
            &executable_path,
        )
        .await;

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
        let config_dir = if is_ts_extension(file_ext) {
            &ts_cwd
        } else {
            &cwd
        };
        if let Some(linter) = detect_available_linter(config_dir, file_ext).await {
            available.push((file_ext, linter));
        }
    }

    if available.is_empty() {
        return Ok("No linters available or no issues found.".to_string());
    }

    let root_clone = cwd;
    let mut join_set: JoinSet<(String, LintResult)> = JoinSet::new();

    for (file_ext, linter) in available {
        let cwd = if is_ts_extension(file_ext) {
            ts_cwd.clone()
        } else {
            root_clone.clone()
        };
        let root = options.root_dir.clone();
        let ext_str = if is_ts_extension(file_ext) {
            ".ts/.tsx"
        } else {
            file_ext
        }
        .to_string();
        let executable_path = executable_path.clone();
        join_set.spawn(async move {
            let result = run_linter(
                &linter,
                file_ext,
                &cwd,
                &root,
                None,
                command_timeout,
                &executable_path,
            )
            .await;
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
            if file_ext == ".go" && result.output.starts_with("no Go files") {
                results.push(format!("[{}] {}", result.tool, truncated));
                continue;
            }
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
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::time::Instant;

    struct ProcessGuard(Option<i32>);

    impl ProcessGuard {
        fn new(pid: i32) -> Self {
            Self(Some(pid))
        }

        fn disarm(&mut self) {
            self.0 = None;
        }
    }

    impl Drop for ProcessGuard {
        fn drop(&mut self) {
            if let Some(pid) = self.0 {
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
            }
        }
    }

    fn install_executable(path: &Path, script: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, script).unwrap();
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).unwrap();
    }

    fn install_fake_node_tools(dir: &Path, behavior: &str) -> PathBuf {
        let bin_dir = dir.join("fake-bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let invocations = dir.join("invocations.log");
        let script = format!(
            "#!/bin/sh\n\
             tool=$(/usr/bin/basename \"$0\")\n\
             printf '%s|%s|%s\\n' \"$tool\" \"$PWD\" \"$*\" >> '{}'\n\
             if [ \"$tool\" = npx ]; then requested=\"$1\"; else requested=\"$tool\"; fi\n\
             {}\n",
            invocations.display(),
            behavior
        );
        for tool in ["npx", "tsc", "eslint"] {
            let path = bin_dir.join(tool);
            install_executable(&path, &script);
        }
        invocations
    }

    fn invocation_lines(path: &Path) -> Vec<String> {
        fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn typescript_invocations(path: &Path) -> Vec<String> {
        invocation_lines(path)
            .into_iter()
            .filter(|line| {
                let mut fields = line.splitn(3, '|');
                let executable = fields.next().unwrap_or_default();
                let _cwd = fields.next();
                let args = fields.next().unwrap_or_default();
                executable == "tsc"
                    || (executable == "npx" && args.split_whitespace().next() == Some("tsc"))
            })
            .collect()
    }

    fn invocation_args(invocation: &str) -> &str {
        invocation.splitn(3, '|').nth(2).unwrap_or_default()
    }

    fn invocation_uses_build_mode(invocation: &str) -> bool {
        invocation_args(invocation)
            .split_whitespace()
            .any(|arg| matches!(arg, "--build" | "-b"))
    }

    fn invocation_project_config(invocation: &str) -> Option<PathBuf> {
        let mut fields = invocation.splitn(3, '|');
        let _executable = fields.next()?;
        let cwd = Path::new(fields.next()?);
        let args: Vec<_> = fields.next()?.split_whitespace().collect();
        let config = args
            .windows(2)
            .find_map(|pair| matches!(pair[0], "-p" | "--project").then_some(Path::new(pair[1])))?;
        Some(if config.is_absolute() {
            config.to_path_buf()
        } else {
            cwd.join(config)
        })
    }

    async fn read_recorded_pid(path: &Path) -> i32 {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Ok(raw) = tokio::fs::read_to_string(path).await
                    && let Ok(pid) = raw.trim().parse()
                {
                    break pid;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("fake command did not record its descendant PID")
    }

    fn process_is_running(pid: i32) -> bool {
        let Ok(stat) = fs::read_to_string(Path::new("/proc").join(pid.to_string()).join("stat"))
        else {
            return false;
        };
        stat.rsplit_once(") ")
            .and_then(|(_, fields)| fields.chars().next())
            != Some('Z')
    }

    async fn wait_for_process_exit(pid: i32, timeout: Duration) -> bool {
        tokio::time::timeout(timeout, async {
            while process_is_running(pid) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .is_ok()
    }

    async fn write_package_covered_by_root_tsconfig(root: &Path) {
        tokio::fs::create_dir_all(root.join("packages/app/src"))
            .await
            .unwrap();
        tokio::fs::write(
            root.join("tsconfig.json"),
            r#"{"include":["packages/app/src"]}"#,
        )
        .await
        .unwrap();
        tokio::fs::write(root.join("packages/app/package.json"), "{}")
            .await
            .unwrap();
        tokio::fs::write(
            root.join("packages/app/src/index.ts"),
            "export const x = 1;\n",
        )
        .await
        .unwrap();
    }

    async fn write_referenced_ts_project(root: &Path) {
        tokio::fs::create_dir_all(root.join("packages/app/src"))
            .await
            .unwrap();
        tokio::fs::write(
            root.join("tsconfig.json"),
            r#"{"files":[],"references":[{"path":"./packages/app"}]}"#,
        )
        .await
        .unwrap();
        tokio::fs::write(root.join("packages/app/tsconfig.json"), "{}")
            .await
            .unwrap();
        tokio::fs::write(
            root.join("packages/app/src/index.ts"),
            "export const x = 1;\n",
        )
        .await
        .unwrap();
        tokio::fs::write(
            root.join("packages/app/src/component.tsx"),
            "export const Component = () => null;\n",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn lane_c_ts_build_never_combines_skip_lib_check() {
        let dir = tempfile::tempdir().unwrap();
        write_referenced_ts_project(dir.path()).await;
        let invocations = install_fake_node_tools(
            dir.path(),
            r#"case "$requested" in
                 tsc) printf 'typescript checked\n' ;;
               esac"#,
        );

        run_static_analysis(StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(dir.path().join("fake-bin").into_os_string()),
            target_path: None,
        })
        .await
        .unwrap();

        let ts_runs = typescript_invocations(&invocations);
        assert!(!ts_runs.is_empty(), "expected a TypeScript invocation");
        for invocation in ts_runs {
            assert!(
                !(invocation.contains("--build") && invocation.contains("--skipLibCheck")),
                "TypeScript invocation combined --build with --skipLibCheck: {invocation}"
            );
        }
    }

    #[tokio::test]
    async fn review_jsonc_active_references_with_comment_uses_build_mode() {
        let dir = tempfile::tempdir().unwrap();
        write_referenced_ts_project(dir.path()).await;
        tokio::fs::write(
            dir.path().join("tsconfig.json"),
            r#"{"files":[],"references": /* projects */ [{"path":"./packages/app"}]}"#,
        )
        .await
        .unwrap();
        let invocations = install_fake_node_tools(dir.path(), "");

        run_static_analysis(StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(dir.path().join("fake-bin").into_os_string()),
            target_path: None,
        })
        .await
        .unwrap();

        let ts_runs = typescript_invocations(&invocations);
        assert_eq!(ts_runs.len(), 1, "expected one TypeScript run: {ts_runs:?}");
        assert!(
            invocation_uses_build_mode(&ts_runs[0]),
            "an active top-level JSONC references property must select build mode: {ts_runs:?}"
        );
        assert!(
            !invocation_args(&ts_runs[0]).contains("--noEmit"),
            "build mode must not include --noEmit: {ts_runs:?}"
        );
    }

    #[tokio::test]
    async fn review_jsonc_commented_out_references_uses_project_no_emit_mode() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(
            dir.path().join("tsconfig.json"),
            "{\n  \"files\": [],\n  // \"references\": [{\"path\": \"./ignored\"}]\n}\n",
        )
        .await
        .unwrap();
        let invocations = install_fake_node_tools(dir.path(), "");

        run_static_analysis(StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(dir.path().join("fake-bin").into_os_string()),
            target_path: None,
        })
        .await
        .unwrap();

        let ts_runs = typescript_invocations(&invocations);
        assert_eq!(ts_runs.len(), 1, "expected one TypeScript run: {ts_runs:?}");
        assert!(
            !invocation_uses_build_mode(&ts_runs[0]),
            "a commented-out references property must not select build mode: {ts_runs:?}"
        );
        assert!(
            invocation_project_config(&ts_runs[0]).is_some()
                && invocation_args(&ts_runs[0]).contains("--noEmit"),
            "a config without active references must use project/noEmit mode: {ts_runs:?}"
        );
    }

    #[tokio::test]
    async fn lane_c_ts_and_tsx_share_one_typescript_run() {
        let dir = tempfile::tempdir().unwrap();
        write_referenced_ts_project(dir.path()).await;
        let invocations = install_fake_node_tools(
            dir.path(),
            r#"case "$requested" in
                 tsc) printf 'typescript checked\n' ;;
               esac"#,
        );

        run_static_analysis(StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(dir.path().join("fake-bin").into_os_string()),
            target_path: None,
        })
        .await
        .unwrap();

        let ts_runs = typescript_invocations(&invocations);
        assert_eq!(
            ts_runs.len(),
            1,
            ".ts and .tsx must be checked by one TypeScript command; got {ts_runs:?}"
        );
    }

    #[tokio::test]
    async fn lane_c_directory_path_uses_nearest_package_tsconfig() {
        let dir = tempfile::tempdir().unwrap();
        write_referenced_ts_project(dir.path()).await;
        let invocations = install_fake_node_tools(
            dir.path(),
            r#"case "$requested" in
                 tsc) printf 'typescript checked\n' ;;
               esac"#,
        );

        run_static_analysis(StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(dir.path().join("fake-bin").into_os_string()),
            target_path: Some("packages/app/src".to_string()),
        })
        .await
        .unwrap();

        let package_dir = dir.path().join("packages/app");
        let ts_runs = typescript_invocations(&invocations);
        assert!(!ts_runs.is_empty(), "expected a scoped TS run");
        for invocation in &ts_runs {
            let fields: Vec<_> = invocation.splitn(3, '|').collect();
            assert_eq!(
                fields.get(1).copied(),
                Some(package_dir.to_string_lossy().as_ref()),
                "TypeScript must run from the nearest package containing the requested path: {ts_runs:?}"
            );
        }
    }

    #[tokio::test]
    async fn review_file_in_package_without_tsconfig_uses_ancestor_tsconfig() {
        let dir = tempfile::tempdir().unwrap();
        write_package_covered_by_root_tsconfig(dir.path()).await;
        let invocations = install_fake_node_tools(dir.path(), "");

        run_static_analysis(StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(dir.path().join("fake-bin").into_os_string()),
            target_path: Some("packages/app/src/index.ts".to_string()),
        })
        .await
        .unwrap();

        let ts_runs = typescript_invocations(&invocations);
        assert_eq!(
            ts_runs.len(),
            1,
            "a package.json must not hide the ancestor tsconfig for a file check: {ts_runs:?}"
        );
        assert_eq!(
            invocation_project_config(&ts_runs[0]),
            Some(dir.path().join("tsconfig.json")),
            "the file check must use the applicable ancestor config: {ts_runs:?}"
        );
    }

    #[tokio::test]
    async fn review_directory_in_package_without_tsconfig_uses_ancestor_tsconfig() {
        let dir = tempfile::tempdir().unwrap();
        write_package_covered_by_root_tsconfig(dir.path()).await;
        let invocations = install_fake_node_tools(dir.path(), "");

        run_static_analysis(StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(dir.path().join("fake-bin").into_os_string()),
            target_path: Some("packages/app/src".to_string()),
        })
        .await
        .unwrap();

        let ts_runs = typescript_invocations(&invocations);
        assert_eq!(
            ts_runs.len(),
            1,
            "a package.json must not hide the ancestor tsconfig for a directory check: {ts_runs:?}"
        );
        assert_eq!(
            invocation_project_config(&ts_runs[0]),
            Some(dir.path().join("tsconfig.json")),
            "the directory check must use the applicable ancestor config: {ts_runs:?}"
        );
    }

    #[tokio::test]
    async fn lane_c_static_analysis_strips_ansi_codes() {
        let dir = tempfile::tempdir().unwrap();
        write_referenced_ts_project(dir.path()).await;
        let _invocations = install_fake_node_tools(
            dir.path(),
            r#"case "$requested" in
                 tsc) printf '\033[31merror TS2322: canned failure\033[0m\n' ;;
               esac"#,
        );

        let output = run_static_analysis(StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(dir.path().join("fake-bin").into_os_string()),
            target_path: None,
        })
        .await
        .unwrap();

        assert!(output.contains("error TS2322: canned failure"), "{output}");
        assert!(
            !output.contains('\u{1b}'),
            "static-analysis output leaked ANSI escape codes: {output:?}"
        );
    }

    #[tokio::test]
    async fn review_local_eslint_precedes_path_oxlint() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("package.json"), "{}")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("index.js"), "const value = 1;\n")
            .await
            .unwrap();
        let invocations = dir.path().join("invocations.log");
        install_executable(
            &dir.path().join("node_modules/.bin/eslint"),
            &format!(
                "#!/bin/sh\nprintf 'local-eslint|%s|%s\\n' \"$PWD\" \"$*\" >> '{}'\n",
                invocations.display()
            ),
        );
        let path_dir = dir.path().join("path-bin");
        install_executable(
            &path_dir.join("oxlint"),
            &format!(
                "#!/bin/sh\nprintf 'path-oxlint|%s|%s\\n' \"$PWD\" \"$*\" >> '{}'\n",
                invocations.display()
            ),
        );

        run_static_analysis(StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(path_dir.into_os_string()),
            target_path: Some("index.js".to_string()),
        })
        .await
        .unwrap();

        let lines = invocation_lines(&invocations);
        assert!(
            lines.iter().any(|line| line.starts_with("local-eslint|")),
            "the project's eslint must run before an unrelated PATH oxlint: {lines:?}"
        );
        assert!(
            !lines.iter().any(|line| line.starts_with("path-oxlint|")),
            "PATH oxlint must not override the project's eslint: {lines:?}"
        );
    }

    #[tokio::test]
    async fn lane_c_timeout_names_command_and_keeps_other_results() {
        let dir = tempfile::tempdir().unwrap();
        write_referenced_ts_project(dir.path()).await;
        let descendant_pid_path = dir.path().join("timeout-descendant.pid");
        let behavior = format!(
            r#"case "$requested" in
                 tsc) printf 'typescript sibling completed\n' ;;
                 eslint|oxlint) /bin/sleep 30 & descendant=$!
                                printf '%s\n' "$descendant" > '{}'
                                wait "$descendant" ;;
               esac"#,
            descendant_pid_path.display()
        );
        let _invocations = install_fake_node_tools(dir.path(), &behavior);
        fs::copy(
            dir.path().join("fake-bin/eslint"),
            dir.path().join("fake-bin/oxlint"),
        )
        .unwrap();

        let started = Instant::now();
        let output = run_static_analysis_with_timeout(
            StaticAnalysisOptions {
                root_dir: dir.path().to_path_buf(),
                executable_path: Some(dir.path().join("fake-bin").into_os_string()),
                target_path: None,
            },
            Duration::from_millis(100),
        )
        .await
        .unwrap();
        let elapsed = started.elapsed();
        let descendant_pid = read_recorded_pid(&descendant_pid_path).await;
        let mut descendant = ProcessGuard::new(descendant_pid);
        let descendant_exited =
            wait_for_process_exit(descendant_pid, Duration::from_millis(750)).await;

        assert!(
            output.contains("typescript sibling completed"),
            "a timed-out lint command must not suppress other check results: {output}"
        );
        assert!(
            output.contains("eslint") || output.contains("oxlint"),
            "timeout must name the command that timed out: {output}"
        );
        assert!(
            output.contains("timed out after 100 ms"),
            "timeout must report its configured duration: {output}"
        );
        assert!(
            elapsed < Duration::from_secs(1),
            "the timed-out check must return promptly; elapsed {elapsed:?}"
        );
        assert!(
            descendant_exited,
            "timing out the lint wrapper left descendant PID {descendant_pid} running"
        );
        descendant.disarm();
    }

    #[tokio::test]
    async fn review_cancelling_command_kills_and_reaps_descendants() {
        let dir = tempfile::tempdir().unwrap();
        let descendant_pid_path = dir.path().join("cancel-descendant.pid");
        let command = dir.path().join("cancel-wrapper");
        install_executable(
            &command,
            &format!(
                "#!/bin/sh\n/bin/sleep 30 &\ndescendant=$!\nprintf '%s\\n' \"$descendant\" > '{}'\nwait \"$descendant\"\n",
                descendant_pid_path.display()
            ),
        );
        let cwd = dir.path().to_path_buf();
        let task = tokio::spawn(async move {
            run_command_with_timeout(
                command.to_string_lossy().as_ref(),
                &[],
                &cwd,
                Duration::from_secs(30),
                OsStr::new("/usr/bin:/bin"),
            )
            .await
        });
        let descendant_pid = read_recorded_pid(&descendant_pid_path).await;
        let mut descendant = ProcessGuard::new(descendant_pid);

        let cancelled_at = Instant::now();
        task.abort();
        let join_result = task.await;
        let cancel_elapsed = cancelled_at.elapsed();
        let descendant_exited =
            wait_for_process_exit(descendant_pid, Duration::from_millis(750)).await;

        assert!(join_result.is_err() && join_result.unwrap_err().is_cancelled());
        assert!(
            cancel_elapsed < Duration::from_secs(1),
            "cancelling the command task must return promptly; elapsed {cancel_elapsed:?}"
        );
        assert!(
            descendant_exited,
            "cancelling the command left descendant PID {descendant_pid} running"
        );
        descendant.disarm();
    }

    #[tokio::test]
    async fn concurrent_checks_use_their_own_executable_paths() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let mut logs = Vec::new();
        for dir in [&first, &second] {
            write_package_covered_by_root_tsconfig(dir.path()).await;
            logs.push(install_fake_node_tools(dir.path(), "helper"));
            install_executable(
                &dir.path().join("fake-bin/helper"),
                &format!("#!/bin/sh\nprintf '%s' '{}'\n", dir.path().display()),
            );
        }
        let check = |dir: &Path| {
            run_static_analysis(StaticAnalysisOptions {
                root_dir: dir.to_path_buf(),
                target_path: Some("packages/app/src".to_string()),
                executable_path: Some(dir.join("fake-bin").into_os_string()),
            })
        };
        let (a, b) = tokio::join!(check(first.path()), check(second.path()));
        for (dir, output, log) in [
            (&first, a.unwrap(), &logs[0]),
            (&second, b.unwrap(), &logs[1]),
        ] {
            assert!(output.contains(dir.path().to_str().unwrap()), "{output}");
            let runs = typescript_invocations(log);
            assert_eq!(runs.len(), 1, "{runs:?}");
            assert_eq!(
                invocation_project_config(&runs[0]),
                Some(dir.path().join("tsconfig.json"))
            );
        }
    }

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
        install_fake_node_tools(dir.path(), "");
        let options = StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(dir.path().join("fake-bin").into_os_string()),
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
        tokio::fs::write(
            dir.path().join("Component.tsx"),
            "export const x: number = 1;\n",
        )
        .await
        .unwrap();

        install_fake_node_tools(dir.path(), "");
        let options = StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(dir.path().join("fake-bin").into_os_string()),
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
        install_fake_node_tools(dir.path(), "");
        let options = StaticAnalysisOptions {
            root_dir: dir.path().to_path_buf(),
            executable_path: Some(dir.path().join("fake-bin").into_os_string()),
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

    #[test]
    fn test_strip_node_modules_drops_dep_diagnostics() {
        // Realistic tsc output: a dep diagnostic with a code-frame, then
        // a user-code diagnostic. Only the user-code diagnostic should
        // survive.
        let raw = "\
node_modules/some-pkg/dist/types.d.ts(42,3): error TS2304: Cannot find name 'Foo'.\n\
\n  42   declare const Foo: unknown;\n     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\
src/main.ts(10,5): error TS7006: Parameter 'x' implicitly has an 'any' type.\n\
\n  10   function f(x) { return x }\n         ~\n";
        let filtered = strip_node_modules_diagnostics(raw);
        assert!(
            !filtered.contains("node_modules/"),
            "filter must drop node_modules header AND its indented detail lines; got:\n{filtered}"
        );
        assert!(
            filtered.contains("src/main.ts(10,5)"),
            "user-code diagnostic must survive; got:\n{filtered}"
        );
        assert!(
            filtered.contains("implicitly has an 'any' type"),
            "user-code diagnostic detail must survive"
        );
    }

    #[test]
    fn test_strip_node_modules_keeps_clean_output() {
        let raw = "src/main.ts(1,1): error TS2304: Cannot find name 'X'.\n";
        let filtered = strip_node_modules_diagnostics(raw);
        assert!(filtered.contains("src/main.ts"));
    }

    #[test]
    fn test_strip_node_modules_handles_empty() {
        assert_eq!(strip_node_modules_diagnostics(""), "");
    }
}
