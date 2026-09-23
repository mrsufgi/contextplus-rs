//! Working directory of the process that spawned this one.
//!
//! MCP hosts (Claude Code, Codex) spawn the server or bridge as a direct
//! child and change their own cwd when the agent moves into a git worktree.
//! Reading the parent's cwd is what lets a session follow the agent into
//! that worktree without the agent passing absolute paths.

use std::path::PathBuf;

/// Tool argument the bridge and the stdio server add to every `tools/call`
/// carrying the host's current working directory. Stripped before dispatch.
pub const CWD_ARG: &str = "_cwd";

/// Current working directory of the parent process, when the platform
/// exposes it (Linux `/proc`).
pub fn parent_process_cwd() -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        let ppid = unsafe { libc::getppid() };
        std::fs::read_link(format!("/proc/{ppid}/cwd")).ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Add [`CWD_ARG`] to a `tools/call` line; leave every other line untouched.
///
/// Invalid JSON and messages without an arguments object pass through
/// byte-for-byte so the bridge never corrupts traffic it does not understand.
pub fn inject_cwd(line: &[u8], cwd: Option<&std::path::Path>) -> Vec<u8> {
    let Some(cwd) = cwd else {
        return line.to_vec();
    };
    let Ok(mut msg) = serde_json::from_slice::<serde_json::Value>(line) else {
        return line.to_vec();
    };
    if msg.get("method").and_then(|m| m.as_str()) != Some("tools/call") {
        return line.to_vec();
    }
    let Some(params) = msg.get_mut("params").and_then(|p| p.as_object_mut()) else {
        return line.to_vec();
    };
    let arguments = params
        .entry("arguments")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    let Some(arguments) = arguments.as_object_mut() else {
        return line.to_vec();
    };
    arguments.insert(
        CWD_ARG.to_string(),
        serde_json::Value::String(cwd.to_string_lossy().into_owned()),
    );
    let mut out = serde_json::to_vec(&msg).unwrap_or_else(|_| line.to_vec());
    out.push(b'\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[cfg(target_os = "linux")]
    #[test]
    fn parent_process_cwd_is_an_existing_directory() {
        let cwd = parent_process_cwd().expect("linux exposes /proc/<ppid>/cwd");
        assert!(cwd.is_dir(), "{}", cwd.display());
    }

    #[test]
    fn inject_cwd_adds_arg_to_tools_call() {
        let line = br#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_file_skeleton","arguments":{"file_path":"src/a.rs"}}}"#;
        let out = inject_cwd(line, Some(Path::new("/wt/feat")));
        let msg: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(msg["params"]["arguments"][CWD_ARG], "/wt/feat");
        assert_eq!(msg["params"]["arguments"]["file_path"], "src/a.rs");
        assert_eq!(out.last(), Some(&b'\n'));
    }

    #[test]
    fn inject_cwd_creates_arguments_when_missing() {
        let line =
            br#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_worktrees"}}"#;
        let out = inject_cwd(line, Some(Path::new("/wt/feat")));
        let msg: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(msg["params"]["arguments"][CWD_ARG], "/wt/feat");
    }

    #[test]
    fn inject_cwd_leaves_other_messages_untouched() {
        let init = br#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        assert_eq!(inject_cwd(init, Some(Path::new("/wt"))), init.to_vec());
        let garbage = b"not json\n";
        assert_eq!(
            inject_cwd(garbage, Some(Path::new("/wt"))),
            garbage.to_vec()
        );
        let call = br#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":{}}}"#;
        assert_eq!(inject_cwd(call, None), call.to_vec());
    }
}
