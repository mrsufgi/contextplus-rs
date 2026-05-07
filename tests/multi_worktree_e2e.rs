//! End-to-end integration test: bridge spawned from a linked worktree finds
//! worktree-only content.
//!
//! # What works today (single-ref, primary walk only)
//!
//! - The daemon starts correctly from a real git-init'd primary repo.
//! - A bridge connecting from the primary root completes the `register_session`
//!   handshake and receives `SessionReady::Ready`.
//! - A bridge connecting from the linked worktree path also handshakes
//!   successfully (ref_id differs from primary's ref_id).
//! - Only ONE daemon process is running at a time (both bridges share the same
//!   socket, because `paths::daemon_dir` follows `.git` pointers back to the
//!   primary).
//! - `lexical_search "primary_only_symbol"` from the primary bridge returns a
//!   hit (positive control — single-ref, primary walk is already wired).
//!
//! # What needs U9/U10/U11/U12 wiring to fully pass
//!
//! - **`worktree_finds_worktree_only_symbol`** (`#[ignore = "U10 dependency: …"]`):
//!   the per-ref embed pipeline (U10) must walk the worktree's own file tree and
//!   index `src/feat_only.rs` into a separate embedding shard. Until that lands,
//!   the daemon serves all lexical_search requests from the primary ref's file
//!   cache, which never includes files added exclusively in the linked worktree.
//!
//! - **`worktree_does_not_find_primary_only_symbol_via_path_leak`**
//!   (`#[ignore = "U14 dependency: …"]`):
//!   the cross-ref path-leak guard (U14) asserts that the worktree bridge's
//!   response paths never contain the primary's directory prefix. This is only
//!   meaningful once U10 routes tool calls through the worktree ref.
//!
//! - **`primary_does_not_find_worktree_only_symbol`** (`#[ignore = "U10 dependency: …"]`):
//!   negative control that requires per-ref scoping; currently lexical_search on
//!   primary walks primary's cache only (correct by accident — primary cache never
//!   contains the worktree-only file), but the assertion only becomes load-bearing
//!   once U10 land and could theoretically cross-contaminate caches.
//!
//! # `#[ignore]` activation
//!
//! Remove the `#[ignore]` attribute from each test when the corresponding unit
//! (U10, U11, U14) is merged. No other changes should be needed — the test
//! scaffolding, git setup, and daemon helpers are all fully wired.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use contextplus_rs::config::Config;
use contextplus_rs::server::ContextPlusServer;
use contextplus_rs::transport::client::{RegisterSession, SessionReady, read_frame, write_frame};
use contextplus_rs::transport::daemon::{self, AcquireOutcome, LockGuard};
use contextplus_rs::transport::paths;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

// ---------------------------------------------------------------------------
// Git setup helpers
// ---------------------------------------------------------------------------

mod helpers {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    /// Run a git command inside `dir`, panic on failure.
    pub fn git(dir: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(dir)
            .env("GIT_AUTHOR_NAME", "Test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "Test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com")
            .status()
            .unwrap_or_else(|e| panic!("git {args:?}: spawn failed: {e}"));
        assert!(
            status.success(),
            "git {args:?} exited with {status} in {}",
            dir.display()
        );
    }

    /// Set up a primary git repo with one source file, return the primary root.
    ///
    /// Creates:
    ///   `<primary>/src/lib.rs` — contains `pub fn primary_only_symbol() {}`
    ///
    /// Commits everything on branch `main`.
    pub fn make_primary_repo(primary: &Path) {
        git(primary, &["init", "-b", "main"]);
        // Needed for some git versions that require user config.
        git(primary, &["config", "user.email", "test@example.com"]);
        git(primary, &["config", "user.name", "Test"]);

        let src = primary.join("src");
        std::fs::create_dir_all(&src).expect("create src/");
        std::fs::write(src.join("lib.rs"), "pub fn primary_only_symbol() {}\n")
            .expect("write lib.rs");

        git(primary, &["add", "."]);
        git(primary, &["commit", "-m", "init: add primary_only_symbol"]);
    }

    /// Add a linked git worktree at `<parent_of_primary>/wt-feat` on branch
    /// `feat/worktree-only`. Within the worktree, creates and commits a new
    /// source file containing `pub fn worktree_only_symbol_xyz_unique() {}`.
    ///
    /// Returns the path to the worktree root.
    pub fn add_linked_worktree(primary: &Path) -> PathBuf {
        let wt_path = primary.parent().unwrap().join("wt-feat");

        // git worktree add <path> -b <branch>
        git(
            primary,
            &[
                "worktree",
                "add",
                wt_path.to_str().unwrap(),
                "-b",
                "feat/worktree-only",
            ],
        );

        // Add a new file that only exists in the worktree branch.
        let wt_src = wt_path.join("src");
        std::fs::create_dir_all(&wt_src).expect("create wt src/");
        std::fs::write(
            wt_src.join("feat_only.rs"),
            "pub fn worktree_only_symbol_xyz_unique() {}\n",
        )
        .expect("write feat_only.rs");

        git(&wt_path, &["add", "."]);
        git(
            &wt_path,
            &["commit", "-m", "feat: add worktree_only_symbol_xyz_unique"],
        );

        wt_path
    }
}

// ---------------------------------------------------------------------------
// Daemon spawn helper (mirrors daemon_transport.rs pattern)
// ---------------------------------------------------------------------------

/// Wrap `daemon::run` so the spawned task can own the lock guard.
/// idle_secs=0 disables the idle-shutdown timer so tests don't race.
async fn run_daemon_task(
    server: ContextPlusServer,
    listener: tokio::net::UnixListener,
    socket_path: PathBuf,
    pid_path: PathBuf,
    lock: LockGuard,
) {
    let _ = daemon::run(server, listener, socket_path, pid_path, 0, lock).await;
}

/// Bring up an in-process daemon rooted at `primary`. Returns `(handle, socket_path)`.
async fn spawn_daemon_at(primary: &Path) -> (tokio::task::JoinHandle<()>, PathBuf) {
    let lock = match daemon::acquire_lock(primary).expect("acquire_lock") {
        AcquireOutcome::Acquired(l) => l,
        AcquireOutcome::AlreadyRunning => panic!("fresh primary dir should never be contended"),
    };
    let listener = daemon::bind_listener(primary).expect("bind_listener");
    daemon::write_pid_file(primary);

    let socket_path = paths::daemon_socket_path(primary);
    let pid_path = paths::daemon_pid_path(primary);
    let server = ContextPlusServer::new(primary.to_path_buf(), Config::from_env());

    let handle = tokio::spawn(run_daemon_task(
        server,
        listener,
        socket_path.clone(),
        pid_path,
        lock,
    ));

    // Wait for the socket to appear.
    for _ in 0..50 {
        if socket_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(socket_path.exists(), "daemon never bound its socket");

    (handle, socket_path)
}

// ---------------------------------------------------------------------------
// Bridge call helper
// ---------------------------------------------------------------------------

/// Connect to `socket`, register a session with `client_root`, then issue a
/// single MCP `tools/call` for `tool` with `args`. Returns the raw JSON-RPC
/// response `Value`.
///
/// This is the workhorse that every test uses: it handles the full handshake
/// (register_session → initialize → notifications/initialized → tools/call).
async fn bridge_call(
    socket: &Path,
    client_root: &Path,
    tool: &str,
    args: Value,
) -> (SessionReady, Value) {
    let mut stream = UnixStream::connect(socket)
        .await
        .expect("connect to daemon socket");

    // Step 0: register_session handshake.
    let reg = RegisterSession {
        client_root: client_root.to_path_buf(),
        head_sha: "deadbeef".to_owned(),
        client_pid: std::process::id(),
    };
    write_frame(&mut stream, &reg)
        .await
        .expect("write register_session");
    let session_ready: SessionReady = read_frame(&mut stream).await.expect("read session_ready");

    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    // Step 1: MCP initialize.
    let init = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-multi-wt", "version": "0.0.0"}
        }
    });
    let mut init_bytes = serde_json::to_vec(&init).unwrap();
    init_bytes.push(b'\n');
    write_half.write_all(&init_bytes).await.unwrap();
    write_half.flush().await.unwrap();

    let mut _init_resp = String::new();
    reader.read_line(&mut _init_resp).await.unwrap();

    // Step 2: notifications/initialized.
    let notif_bytes = b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
    write_half.write_all(notif_bytes).await.unwrap();
    write_half.flush().await.unwrap();

    // Step 3: tools/call.
    let call = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": tool,
            "arguments": args
        }
    });
    let mut call_bytes = serde_json::to_vec(&call).unwrap();
    call_bytes.push(b'\n');
    write_half.write_all(&call_bytes).await.unwrap();
    write_half.flush().await.unwrap();

    let mut resp_line = String::new();
    reader
        .read_line(&mut resp_line)
        .await
        .expect("read tool response");
    assert!(!resp_line.is_empty(), "daemon closed before tool response");

    let resp_json: Value =
        serde_json::from_str(resp_line.trim()).expect("tool response is valid JSON-RPC");

    (session_ready, resp_json)
}

/// Extract the text content from a `tools/call` JSON-RPC response.
///
/// Returns the concatenated text from `result.content[*].text`, or the full
/// response serialised as a fallback for debugging.
fn extract_text(resp: &Value) -> String {
    if let Some(Some(arr)) = resp.pointer("/result/content").map(|c| c.as_array()) {
        let joined: String = arr
            .iter()
            .filter_map(|chunk| chunk.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n");
        if !joined.is_empty() {
            return joined;
        }
    }
    // Fallback: serialise the whole response so test assertions have something
    // to show in their failure messages.
    serde_json::to_string_pretty(resp).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Test 1: positive control — primary bridge finds primary-only symbol
// ---------------------------------------------------------------------------
//
// This test exercises the existing single-ref, primary-walk path.  It must
// pass today without any U10/U11 dependency.

#[tokio::test(flavor = "multi_thread")]
async fn primary_finds_primary_only_symbol() {
    let td = TempDir::new().unwrap();
    let primary = td.path().join("repo");
    std::fs::create_dir_all(&primary).unwrap();

    helpers::make_primary_repo(&primary);
    let _wt = helpers::add_linked_worktree(&primary);

    let (handle, socket) = spawn_daemon_at(&primary).await;

    let (session_ready, resp) = bridge_call(
        &socket,
        &primary,
        "lexical_search",
        json!({"query": "primary_only_symbol"}),
    )
    .await;

    assert!(
        matches!(
            session_ready,
            SessionReady::Ready { .. } | SessionReady::Warming { .. }
        ),
        "primary bridge must get Ready/Warming, got {session_ready:?}"
    );

    let text = extract_text(&resp);
    assert!(
        text.contains("primary_only_symbol") || text.contains("lib.rs"),
        "primary bridge should find primary_only_symbol in its own tree;\nresponse: {text}"
    );

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 2: single daemon serves both worktrees (process-count assertion)
// ---------------------------------------------------------------------------
//
// Both bridges connect to the same socket. The socket's inode does not change
// between the first and second connection — proving one daemon, not two.

#[tokio::test(flavor = "multi_thread")]
async fn single_daemon_serves_both_worktrees() {
    let td = TempDir::new().unwrap();
    let primary = td.path().join("repo");
    std::fs::create_dir_all(&primary).unwrap();

    helpers::make_primary_repo(&primary);
    let wt = helpers::add_linked_worktree(&primary);

    let (handle, socket) = spawn_daemon_at(&primary).await;

    // Verify inode is stable across both connections.
    use std::os::unix::fs::MetadataExt;
    let inode_before = std::fs::metadata(&socket).unwrap().ino();

    // Bridge A: primary
    let mut stream_a = UnixStream::connect(&socket).await.unwrap();
    let reg_a = RegisterSession {
        client_root: primary.clone(),
        head_sha: "deadbeef".to_owned(),
        client_pid: std::process::id(),
    };
    write_frame(&mut stream_a, &reg_a).await.unwrap();
    let ready_a: SessionReady = read_frame(&mut stream_a).await.unwrap();

    // Bridge B: linked worktree (must resolve to the same daemon socket).
    let wt_socket = paths::daemon_socket_path(&wt);
    // The worktree should resolve to the same socket file as the primary.
    assert_eq!(
        socket.canonicalize().unwrap_or_else(|_| socket.clone()),
        wt_socket
            .canonicalize()
            .unwrap_or_else(|_| wt_socket.clone()),
        "linked worktree socket path must equal primary socket path"
    );

    let mut stream_b = UnixStream::connect(&socket).await.unwrap();
    let reg_b = RegisterSession {
        client_root: wt.clone(),
        head_sha: "deadbeef".to_owned(),
        client_pid: std::process::id(),
    };
    write_frame(&mut stream_b, &reg_b).await.unwrap();
    let ready_b: SessionReady = read_frame(&mut stream_b).await.unwrap();

    let inode_after = std::fs::metadata(&socket).unwrap().ino();
    assert_eq!(
        inode_before, inode_after,
        "socket inode changed — a second daemon must have spawned"
    );

    // Both sessions must be Ready/Warming.
    assert!(
        matches!(
            ready_a,
            SessionReady::Ready { .. } | SessionReady::Warming { .. }
        ),
        "primary session must be Ready/Warming, got {ready_a:?}"
    );
    assert!(
        matches!(
            ready_b,
            SessionReady::Ready { .. } | SessionReady::Warming { .. }
        ),
        "worktree session must be Ready/Warming, got {ready_b:?}"
    );

    // The two sessions should have different ref_ids (different roots → different refs).
    let ref_a = match &ready_a {
        SessionReady::Ready { ref_id, .. } | SessionReady::Warming { ref_id, .. } => *ref_id,
        _ => panic!(),
    };
    let ref_b = match &ready_b {
        SessionReady::Ready { ref_id, .. } | SessionReady::Warming { ref_id, .. } => *ref_id,
        _ => panic!(),
    };
    assert_ne!(
        ref_a, ref_b,
        "primary and worktree bridges must have different ref_ids"
    );

    drop(stream_a);
    drop(stream_b);
    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 3: worktree bridge finds worktree-only symbol
// ---------------------------------------------------------------------------
//
// GOAL: the load-bearing assertion — a bridge connected from the linked
// worktree's root should find `worktree_only_symbol_xyz_unique` (which only
// exists in `wt-feat/src/feat_only.rs`, not in `main`).
//
// BLOCKED ON: U10 (per-ref embed pipeline). Until U10 lands and wires per-ref
// file-tree walking + indexing, the daemon serves lexical_search from the
// primary ref's file cache, which never includes worktree-only files.
//
// ACTIVATION: remove `#[ignore]` when U10 is merged.

#[tokio::test(flavor = "multi_thread")]
async fn worktree_finds_worktree_only_symbol() {
    let td = TempDir::new().unwrap();
    let primary = td.path().join("repo");
    std::fs::create_dir_all(&primary).unwrap();

    helpers::make_primary_repo(&primary);
    let wt = helpers::add_linked_worktree(&primary);

    let (handle, socket) = spawn_daemon_at(&primary).await;

    let (session_ready, resp) = bridge_call(
        &socket,
        &wt,
        "lexical_search",
        json!({"query": "worktree_only_symbol_xyz_unique"}),
    )
    .await;

    assert!(
        matches!(
            session_ready,
            SessionReady::Ready { .. } | SessionReady::Warming { .. }
        ),
        "worktree bridge must get Ready/Warming, got {session_ready:?}"
    );

    let text = extract_text(&resp);
    assert!(
        text.contains("worktree_only_symbol_xyz_unique") || text.contains("feat_only.rs"),
        "worktree bridge must find the worktree-only symbol;\nresponse: {text}"
    );

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 4: primary bridge does NOT find worktree-only symbol
// ---------------------------------------------------------------------------
//
// Negative control: `primary_only_symbol` must NOT bleed into results
// when a worktree-scoped search is done.
//
// This currently passes because the primary ref's cache only indexes primary's
// file tree. However the assertion only becomes load-bearing (and non-trivial)
// once U10 lands and per-ref scoping is enforced; before that it passes for
// the wrong reason. We mark it ignored so it acts as a proper regression guard
// when activated alongside U10.
//
// ACTIVATION: remove `#[ignore]` when U10 + U11 tool-dispatch routing land.

#[ignore = "U10 dependency: per-ref scoping not yet enforced; passes today by accident (primary cache never sees worktree files)"]
#[tokio::test(flavor = "multi_thread")]
async fn primary_does_not_find_worktree_only_symbol() {
    let td = TempDir::new().unwrap();
    let primary = td.path().join("repo");
    std::fs::create_dir_all(&primary).unwrap();

    helpers::make_primary_repo(&primary);
    let wt = helpers::add_linked_worktree(&primary);

    let (handle, socket) = spawn_daemon_at(&primary).await;

    // Bridge from primary — must NOT see worktree-only file.
    let (_session_ready, resp) = bridge_call(
        &socket,
        &primary,
        "lexical_search",
        json!({"query": "worktree_only_symbol_xyz_unique"}),
    )
    .await;

    let text = extract_text(&resp);
    assert!(
        !text.contains("worktree_only_symbol_xyz_unique"),
        "primary bridge must NOT find worktree-only symbol;\nresponse: {text}"
    );

    // Suppress unused-variable warning while worktree isn't directly needed.
    let _ = wt;

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 5: worktree bridge response paths must NOT contain primary prefix
// ---------------------------------------------------------------------------
//
// Cross-ref leakage guard: even if U10 wires per-ref indexing, output paths
// from a worktree session must never contain the primary root's filesystem
// prefix. Uses `path_translation::json_contains_string` to probe the raw
// response JSON.
//
// BLOCKED ON: U14 (per-ref output path rewriting). Without U14's output
// translation the tool result paths are relative (repo-relative), so the
// check is vacuous. Once U14 rewrites absolute output paths to the caller's
// root, this assertion gains meaning.
//
// ACTIVATION: remove `#[ignore]` when U14 (cross-ref path rewriting) is merged.

#[ignore = "U14 dependency: per-ref output path rewriting not yet wired; leakage check is vacuous without absolute path output"]
#[tokio::test(flavor = "multi_thread")]
async fn worktree_does_not_find_primary_only_symbol_via_path_leak() {
    use contextplus_rs::core::path_translation::json_contains_string;

    let td = TempDir::new().unwrap();
    let primary = td.path().join("repo");
    std::fs::create_dir_all(&primary).unwrap();

    helpers::make_primary_repo(&primary);
    let wt = helpers::add_linked_worktree(&primary);

    let (handle, socket) = spawn_daemon_at(&primary).await;

    // Bridge B: worktree. Run lexical_search for any result.
    let (_session_ready, resp) =
        bridge_call(&socket, &wt, "lexical_search", json!({"query": "fn"})).await;

    // The response must NOT contain the primary root's filesystem path as a
    // string embedded anywhere in the JSON tree.
    let primary_str = primary.to_str().unwrap();
    assert!(
        !json_contains_string(&resp, primary_str),
        "worktree bridge response must not contain primary root path '{primary_str}';\
         \nresponse: {resp:#}"
    );

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 6: worktree bridge handshake — session_id and ref_id are populated
// ---------------------------------------------------------------------------
//
// Lightweight smoke test that does NOT depend on any U10+ feature: simply
// confirms that the `register_session` handshake with a linked-worktree root
// produces a valid `Ready` reply with non-zero ref_id.  Exercises the path
// through `daemon::serve_connection` → `RefId::for_canonical_path` →
// `attach_ref`.

#[tokio::test(flavor = "multi_thread")]
async fn worktree_bridge_gets_valid_session_ready() {
    let td = TempDir::new().unwrap();
    let primary = td.path().join("repo");
    std::fs::create_dir_all(&primary).unwrap();

    helpers::make_primary_repo(&primary);
    let wt = helpers::add_linked_worktree(&primary);

    let (handle, socket) = spawn_daemon_at(&primary).await;

    let mut stream = UnixStream::connect(&socket).await.unwrap();
    let reg = RegisterSession {
        client_root: wt.clone(),
        head_sha: "deadbeef".to_owned(),
        client_pid: std::process::id(),
    };
    write_frame(&mut stream, &reg).await.unwrap();
    let session_ready: SessionReady = read_frame(&mut stream).await.unwrap();

    match &session_ready {
        SessionReady::Ready { session_id, ref_id } => {
            assert!(!session_id.is_empty(), "session_id must be non-empty");
            assert_ne!(*ref_id, 0, "ref_id must be non-zero for worktree");
        }
        SessionReady::Warming {
            session_id, ref_id, ..
        } => {
            assert!(!session_id.is_empty(), "session_id must be non-empty");
            assert_ne!(*ref_id, 0, "ref_id must be non-zero for worktree");
        }
        SessionReady::RejectedDraining => {
            panic!("worktree bridge was rejected (draining) — daemon not ready?");
        }
    }

    drop(stream);
    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 7: worktree attach triggers shallow warmup (project_cache pre-populated)
// ---------------------------------------------------------------------------
//
// GOAL: When a bridge connects from a linked worktree and
// `CONTEXTPLUS_REF_WARMUP_MODE=shallow` (the default post-U16), the
// `spawn_ref_warmup` task runs the file walker so the ref's `project_cache`
// is populated *before* the first tool call — removing the cold-build latency
// that previously hit on `lexical_search`.
//
// The assertion strategy: call `lexical_search` immediately after connect
// (no sleep). With eager-warmup the cache is already built; without it the
// daemon builds the cache inline in the tool call (still works, just slower).
// The presence of results is the correctness assertion; the absence of a
// first-call rebuild *latency spike* is the performance assertion (not tested
// here).
//
// BLOCKED ON: U16 (`RefWarmupMode::Shallow` default) and U18
// (`spawn_ref_warmup` wired from `serve_connection` before `SessionReady` is
// sent, so the cache is ready by the time the client issues its first call).
//
// ACTIVATION: remove `#[ignore]` when U16 + U18 are merged.

#[ignore = "U18 dependency: spawn_ref_warmup not yet wired from serve_connection; project_cache eager population requires U16+U18"]
#[tokio::test(flavor = "multi_thread")]
async fn worktree_attach_triggers_shallow_warmup() {
    let td = TempDir::new().unwrap();
    let primary = td.path().join("repo");
    std::fs::create_dir_all(&primary).unwrap();

    helpers::make_primary_repo(&primary);
    let wt = helpers::add_linked_worktree(&primary);

    // Shallow mode: walker runs but no Ollama calls.
    unsafe {
        std::env::set_var("CONTEXTPLUS_REF_WARMUP_MODE", "shallow");
    }

    let (handle, socket) = spawn_daemon_at(&primary).await;

    // Issue a tool call from the worktree bridge immediately — no sleep.
    // The pre-populated project_cache should serve this without a cold build.
    let (session_ready, resp) = bridge_call(
        &socket,
        &wt,
        "lexical_search",
        json!({"query": "worktree_only_symbol_xyz_unique"}),
    )
    .await;

    assert!(
        matches!(
            session_ready,
            SessionReady::Ready { .. } | SessionReady::Warming { .. }
        ),
        "worktree bridge must get Ready/Warming, got {session_ready:?}"
    );

    // The response should indicate the project_cache was already populated
    // (lexical_search returns results, not an empty response or an error
    // about a missing cache).
    let resp_str = serde_json::to_string_pretty(&resp).unwrap_or_default();
    assert!(
        resp.pointer("/result").is_some(),
        "lexical_search should return a result, not an error;\nresponse: {resp_str}"
    );

    // Clean-up env.
    unsafe {
        std::env::remove_var("CONTEXTPLUS_REF_WARMUP_MODE");
    }

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 8: warmup failure is non-fatal — daemon still serves tool calls
// ---------------------------------------------------------------------------
//
// GOAL: If `spawn_ref_warmup` encounters an error (e.g. the walker cannot
// read the ref's root because `.mcp_data` was made read-only mid-attach),
// the daemon logs a warning but does NOT reject the connection. Subsequent
// tool calls on the same session still succeed.
//
// Implementation: we make the `.mcp_data` directory read-only *after* the
// daemon starts but *before* the worktree attaches. The CAS `fork_from` call
// in `serve_connection` will fail writing the parent pointer; `spawn_ref_warmup`
// (U18) would similarly fail. The daemon must continue serving.
//
// BLOCKED ON: U18 (`spawn_ref_warmup` error path — it must not propagate the
// error to the connection handler). The non-fatal behaviour for `fork_from`
// failure already exists (`tracing::warn!` + continue), but U18 must apply
// the same pattern for warmup failures.
//
// ACTIVATION: remove `#[ignore]` when U18 is merged (the fork_from failure
// path is already non-fatal; the warmup error path needs the same guard).

#[ignore = "U18 dependency: spawn_ref_warmup error path not yet implemented; non-fatal warmup requires U18"]
#[tokio::test(flavor = "multi_thread")]
async fn attach_warmup_failure_is_nonfatal() {
    let td = TempDir::new().unwrap();
    let primary = td.path().join("repo");
    std::fs::create_dir_all(&primary).unwrap();

    helpers::make_primary_repo(&primary);
    let wt = helpers::add_linked_worktree(&primary);

    let (handle, socket) = spawn_daemon_at(&primary).await;

    use std::os::unix::fs::PermissionsExt;

    // Make `.mcp_data` read-only so CAS fork_from and any warmup disk write fail.
    let mcp_data = primary.join(".mcp_data");
    if mcp_data.exists() {
        let ro_perms = std::fs::Permissions::from_mode(0o555);
        let _ = std::fs::set_permissions(&mcp_data, ro_perms);
    }

    // Attempt to register a worktree session — fork_from will fail (non-fatal).
    let session_ready: SessionReady = {
        let mut stream = UnixStream::connect(&socket)
            .await
            .expect("connect to daemon socket");
        let reg = RegisterSession {
            client_root: wt.clone(),
            head_sha: "deadbeef".to_owned(),
            client_pid: std::process::id(),
        };
        write_frame(&mut stream, &reg)
            .await
            .expect("write register_session");
        read_frame(&mut stream).await.expect("read session_ready")
    };

    // LOAD-BEARING: the daemon must not reject the connection even when warmup fails.
    assert!(
        matches!(
            session_ready,
            SessionReady::Ready { .. } | SessionReady::Warming { .. }
        ),
        "daemon must accept the session even when warmup fails; got {session_ready:?}"
    );

    // Now restore permissions and verify the daemon still serves tool calls from primary.
    let rw_perms = std::fs::Permissions::from_mode(0o755);
    let _ = std::fs::set_permissions(&mcp_data, rw_perms);

    let (_sr, resp) = bridge_call(&socket, &primary, "get_context_tree", json!({})).await;

    assert!(
        resp.pointer("/result").is_some(),
        "daemon must still serve tool calls after warmup failure;\nresponse: {resp:#}"
    );

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 7: daemon socket path for linked worktree equals primary socket path
// ---------------------------------------------------------------------------
//
// White-box check: `paths::daemon_socket_path(&wt)` must resolve to the same
// file as `paths::daemon_socket_path(&primary)`. This is the path-resolution
// invariant that makes all worktree-sharing work. No daemon is started — this
// is a pure filesystem + git-pointer resolution test.

#[tokio::test]
async fn daemon_socket_path_for_worktree_equals_primary() {
    let td = TempDir::new().unwrap();
    let primary = td.path().join("repo");
    std::fs::create_dir_all(&primary).unwrap();

    helpers::make_primary_repo(&primary);
    let wt = helpers::add_linked_worktree(&primary);

    let primary_socket = paths::daemon_socket_path(&primary);
    let wt_socket = paths::daemon_socket_path(&wt);

    // The paths need not be identical strings (symlinks, canonicalization), but
    // they must point at the same location — compare via canonicalize of the
    // parent directory (the socket file itself might not exist yet).
    let primary_dir = primary_socket
        .parent()
        .unwrap()
        .canonicalize()
        .unwrap_or_else(|_| primary_socket.parent().unwrap().to_path_buf());
    let wt_dir = wt_socket
        .parent()
        .unwrap()
        .canonicalize()
        .unwrap_or_else(|_| wt_socket.parent().unwrap().to_path_buf());

    assert_eq!(
        primary_dir,
        wt_dir,
        "daemon_dir for worktree ({}) must equal daemon_dir for primary ({})",
        wt_dir.display(),
        primary_dir.display()
    );

    assert_eq!(
        primary_socket.file_name(),
        wt_socket.file_name(),
        "socket filename must be the same"
    );
}
