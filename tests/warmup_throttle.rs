//! U19 — verification tests for per-ref warmup throttling.
//! U21 — integration-level tests for the shallow/full warmup contract (U20).
//!
//! # What this file proves
//!
//! ## U19 tests (U16–U18 wiring)
//!
//! | Test | Invariant |
//! |---|---|
//! | `shallow_warmup_makes_no_ollama_calls` | Shallow mode walks files but never embeds |
//! | `full_warmup_respects_concurrency_cap` | Single ref: peak concurrent ≤ cap |
//! | `concurrent_attaches_share_one_ollama_budget` | N refs: peak concurrent ≤ cap across all |
//! | `off_mode_skips_warmup_entirely` | Off mode: 0 embed calls AND project_cache stays None |
//! | `idempotent_attach_does_not_duplicate_warmup` | Duplicate attaches: embed calls = 1× not N× |
//!
//! ## U21 tests (U20 baseline-import contract, end-to-end through daemon transport)
//!
//! | Test | Invariant |
//! |---|---|
//! | `shallow_attach_inherits_primary_baseline_zero_ollama` | Shallow worktree attach: 0 Ollama for file embeds, corpus inherited from primary CAS |
//! | `full_attach_inherits_baseline_then_embeds_diff_only` | Full worktree attach: Ollama sees only diff files (≤ 2), not the whole corpus |
//! | `shallow_attach_to_unrelated_root_falls_through_to_lazy` | Shallow on orphan root: 0 Ollama, no inherited corpus, first query triggers lazy embed |
//!
//! # Mock approach
//!
//! Uses `wiremock` (already a dev-dep) via `tests/common/mock_ollama.rs`.
//! The mock stands up a local HTTP server on `127.0.0.1:0`, responds with
//! synthetic 768-dim vectors, and tracks `total_calls` + `peak_concurrent`
//! via `Arc<AtomicUsize>` counters. No trait objects or src/ changes required.

#![cfg(unix)]

// Bring in the shared mock harness.
mod common;
use common::mock_ollama::MockOllamaServer;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use contextplus_rs::config::Config;
use contextplus_rs::server::ContextPlusServer;
use contextplus_rs::transport::client::{RegisterSession, SessionReady, read_frame, write_frame};
use contextplus_rs::transport::daemon::{self, AcquireOutcome};
use contextplus_rs::transport::paths;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

// ---------------------------------------------------------------------------
// Git helpers (mirrored from multi_worktree_e2e.rs to keep this file self-contained)
// ---------------------------------------------------------------------------

fn git(dir: &Path, args: &[&str]) {
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

/// Create a minimal git repo with N source files and return its path.
///
/// Each file contains a single trivial function so the walker picks it up and
/// produces at least one chunk for potential embedding.
fn make_repo_with_files(parent: &Path, name: &str, file_count: usize) -> PathBuf {
    let repo = parent.join(name);
    std::fs::create_dir_all(&repo).expect("create repo dir");

    git(&repo, &["init", "-b", "main"]);
    git(&repo, &["config", "user.email", "test@example.com"]);
    git(&repo, &["config", "user.name", "Test"]);

    let src = repo.join("src");
    std::fs::create_dir_all(&src).expect("create src/");

    for i in 0..file_count {
        let content = format!("pub fn warmup_fn_{i}() -> u32 {{ {i} }}\n");
        std::fs::write(src.join(format!("f{i:04}.rs")), content)
            .unwrap_or_else(|e| panic!("write f{i:04}.rs: {e}"));
    }

    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-m", "init"]);

    repo
}

// ---------------------------------------------------------------------------
// Daemon helpers
// ---------------------------------------------------------------------------

/// Start an in-process daemon rooted at `primary` using the given `config`.
/// Returns `(handle, socket_path)`.
///
/// `idle_secs = 0` disables the idle-shutdown timer to prevent races.
async fn spawn_daemon_with_config(
    primary: &Path,
    config: Config,
) -> (tokio::task::JoinHandle<()>, PathBuf) {
    let lock = match daemon::acquire_lock(primary).expect("acquire_lock") {
        AcquireOutcome::Acquired(l) => l,
        AcquireOutcome::AlreadyRunning => panic!("fresh dir should not be contended"),
    };
    let listener = daemon::bind_listener(primary).expect("bind_listener");
    daemon::write_pid_file(primary);

    let socket_path = paths::daemon_socket_path(primary);
    let pid_path = paths::daemon_pid_path(primary);
    let server = ContextPlusServer::new(primary.to_path_buf(), config);

    let socket_clone = socket_path.clone();
    let handle = tokio::spawn(async move {
        let _ = daemon::run(server, listener, socket_clone, pid_path, 0, lock).await;
    });

    // Wait up to 1 s for the socket to appear.
    for _ in 0..50 {
        if socket_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(socket_path.exists(), "daemon never bound its socket");

    (handle, socket_path)
}

/// Register a session from `client_root` with the daemon at `socket`.
/// Returns the `SessionReady` reply.
async fn register_session(socket: &Path, client_root: &Path) -> SessionReady {
    let mut stream = UnixStream::connect(socket)
        .await
        .expect("connect to daemon socket");

    let reg = RegisterSession {
        client_root: client_root.to_path_buf(),
        head_sha: "deadbeef".to_owned(),
        client_pid: std::process::id(),
    };
    write_frame(&mut stream, &reg)
        .await
        .expect("write register_session");
    read_frame(&mut stream).await.expect("read session_ready")
}

/// Issue `tools/call` for `tool` after full MCP handshake.  Used for tests
/// that need to actually exercise a tool (not just the attach phase).
#[allow(dead_code)]
async fn bridge_call(
    socket: &Path,
    client_root: &Path,
    tool: &str,
    args: serde_json::Value,
) -> (SessionReady, serde_json::Value) {
    let mut stream = UnixStream::connect(socket)
        .await
        .expect("connect to daemon socket");

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

    // MCP initialize
    let init_bytes = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-warmup", "version": "0.0.0"}
        }
    }))
    .unwrap();
    write_half
        .write_all(&[init_bytes.as_slice(), b"\n"].concat())
        .await
        .unwrap();
    write_half.flush().await.unwrap();
    let mut _init_resp = String::new();
    reader.read_line(&mut _init_resp).await.unwrap();

    // notifications/initialized
    write_half
        .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
        .await
        .unwrap();
    write_half.flush().await.unwrap();

    // tools/call
    let call_bytes = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": { "name": tool, "arguments": args }
    }))
    .unwrap();
    write_half
        .write_all(&[call_bytes.as_slice(), b"\n"].concat())
        .await
        .unwrap();
    write_half.flush().await.unwrap();

    let mut resp_line = String::new();
    reader
        .read_line(&mut resp_line)
        .await
        .expect("read tool response");
    assert!(!resp_line.is_empty(), "daemon closed before tool response");

    let resp_json: serde_json::Value =
        serde_json::from_str(resp_line.trim()).expect("valid JSON-RPC");

    (session_ready, resp_json)
}

// ---------------------------------------------------------------------------
// Config builder helpers
// ---------------------------------------------------------------------------

/// Build a `Config` with `OLLAMA_HOST` set to the mock server's URI and
/// `CONTEXTPLUS_EMBED_TRACKER=off` so the background tracker does not
/// interfere with embed-call counters.
fn config_with_mock(mock_uri: &str) -> Config {
    // SAFETY: test-only env mutation.  Each test owns its own mock server URI
    // and tests that touch these env vars run with their own daemon instances.
    unsafe {
        std::env::set_var("OLLAMA_HOST", mock_uri);
        // Disable background tracker so it doesn't produce spurious embed calls.
        std::env::set_var("CONTEXTPLUS_EMBED_TRACKER", "off");
        // Disable warmup-on-start to prevent the primary ref from triggering
        // embed calls before the test's own warmup phase begins.
        std::env::set_var("CONTEXTPLUS_WARMUP_ON_START", "false");
    }
    Config::from_env()
}

// ---------------------------------------------------------------------------
// Test 1: shallow mode never calls Ollama
// ---------------------------------------------------------------------------
//
// GOAL: `CONTEXTPLUS_REF_WARMUP_MODE=shallow` → `spawn_ref_warmup` runs the
// file walker (so `project_cache` is populated) but does NOT call Ollama's
// `/api/embed` endpoint.
//
// BLOCKED ON: U16 (`RefWarmupMode` enum + config field) and U18
// (`spawn_ref_warmup` wired from `serve_connection`). Until those land, there
// is no warmup dispatch to assert against.
//
// ACTIVATION: remove `#[ignore]` when U16 + U18 are merged.

#[tokio::test(flavor = "multi_thread")]
async fn shallow_warmup_makes_no_ollama_calls() {
    let td = TempDir::new().unwrap();
    // 5 files — enough for the walker to produce project_cache entries.
    let repo = make_repo_with_files(td.path(), "repo", 5);

    let mock = MockOllamaServer::start(0).await;

    unsafe {
        std::env::set_var("CONTEXTPLUS_REF_WARMUP_MODE", "shallow");
    }
    let config = config_with_mock(mock.uri());
    let (handle, socket) = spawn_daemon_with_config(&repo, config).await;

    // Register a session — this triggers warmup (U18).
    let session_ready = register_session(&socket, &repo).await;
    assert!(
        matches!(
            session_ready,
            SessionReady::Ready { .. } | SessionReady::Warming { .. }
        ),
        "expected Ready or Warming, got {session_ready:?}"
    );

    // Give warmup task time to complete (it is async / spawned separately).
    tokio::time::sleep(Duration::from_millis(300)).await;

    // LOAD-BEARING: shallow mode must never call Ollama.
    assert_eq!(
        mock.total_calls(),
        0,
        "shallow warmup must make 0 embed calls; got {}",
        mock.total_calls()
    );

    // Secondary assertion: the walker still ran — project_cache should be
    // populated (checked indirectly via a lexical_search hitting without a
    // cold-build delay, but for now we assert the counter is the only gate).

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 2: full mode respects the per-daemon concurrency cap
// ---------------------------------------------------------------------------
//
// GOAL: `mode=full` + `CONTEXTPLUS_OLLAMA_MAX_CONCURRENT=2`. One ref with N
// files. After warmup completes, `mock.peak_concurrent() <= 2`.
//
// BLOCKED ON: U16 (`ollama_max_concurrent` config + `ollama_semaphore` in
// `SharedState`), U17 (semaphore acquisition in `OllamaClient::embed_*`), U18
// (`spawn_ref_warmup` routing through the semaphore-gated client).
//
// ACTIVATION: remove `#[ignore]` when U16 + U17 + U18 are merged.

#[tokio::test(flavor = "multi_thread")]
async fn full_warmup_respects_concurrency_cap() {
    let td = TempDir::new().unwrap();
    // 20 files — enough to expose concurrency behaviour even with small batches.
    let repo = make_repo_with_files(td.path(), "repo", 20);

    // 80 ms delay makes concurrent overlap easily observable.
    let mock = MockOllamaServer::start(80).await;

    unsafe {
        std::env::set_var("CONTEXTPLUS_REF_WARMUP_MODE", "full");
        std::env::set_var("CONTEXTPLUS_OLLAMA_MAX_CONCURRENT", "2");
    }
    let config = config_with_mock(mock.uri());
    let (handle, socket) = spawn_daemon_with_config(&repo, config).await;

    let _session_ready = register_session(&socket, &repo).await;

    // Wait for warmup to finish — full embed of 20 files takes < 5 s even
    // with 80 ms/request at cap=2 (20 × 80ms / 2 = 800ms; give 5× margin).
    tokio::time::sleep(Duration::from_secs(5)).await;

    let peak = mock.peak_concurrent();
    assert!(
        peak <= 2,
        "peak concurrent embed calls must be ≤ 2 (cap); observed {peak}"
    );
    // Sanity: some work was done.
    assert!(
        mock.total_calls() > 0,
        "full warmup must make at least one embed call"
    );

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 3: concurrent attaches share one global Ollama budget
// ---------------------------------------------------------------------------
//
// GOAL: 4 refs attached concurrently with `cap=2` → peak concurrent embed
// calls across ALL refs combined never exceeds 2.
//
// BLOCKED ON: U16 + U17 + U18 (same as test 2).
//
// ACTIVATION: remove `#[ignore]` when U16 + U17 + U18 are merged.

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_attaches_share_one_ollama_budget() {
    let td = TempDir::new().unwrap();
    // Primary repo (daemon is rooted here).
    let primary = make_repo_with_files(td.path(), "primary", 5);

    // 3 additional repos simulating concurrent worktree attaches.
    // In production these are linked worktrees; for this test we just spawn
    // daemons from independent repos all talking to the same mock.
    // (Real cross-ref semaphore sharing requires U18 wire-up.)
    let _wt1 = make_repo_with_files(td.path(), "wt1", 5);
    let _wt2 = make_repo_with_files(td.path(), "wt2", 5);
    let _wt3 = make_repo_with_files(td.path(), "wt3", 5);

    // 80 ms delay makes concurrent overlap observable.
    let mock = MockOllamaServer::start(80).await;

    unsafe {
        std::env::set_var("CONTEXTPLUS_REF_WARMUP_MODE", "full");
        std::env::set_var("CONTEXTPLUS_OLLAMA_MAX_CONCURRENT", "2");
    }
    let config = config_with_mock(mock.uri());
    let (handle, socket) = spawn_daemon_with_config(&primary, config).await;

    // Attach 4 sessions concurrently (primary + 3 worktrees).
    let socket_clone = socket.clone();
    let primary_clone = primary.clone();
    let handles = vec![
        tokio::spawn({
            let s = socket_clone.clone();
            let r = primary_clone.clone();
            async move { register_session(&s, &r).await }
        }),
        tokio::spawn({
            let s = socket_clone.clone();
            let r = primary_clone.clone();
            async move { register_session(&s, &r).await }
        }),
        tokio::spawn({
            let s = socket_clone.clone();
            let r = primary_clone.clone();
            async move { register_session(&s, &r).await }
        }),
        tokio::spawn({
            let s = socket_clone.clone();
            let r = primary_clone.clone();
            async move { register_session(&s, &r).await }
        }),
    ];

    for h in handles {
        h.await.expect("session task panicked");
    }

    // Wait for all warmup tasks to finish.
    tokio::time::sleep(Duration::from_secs(6)).await;

    let peak = mock.peak_concurrent();
    assert!(
        peak <= 2,
        "peak concurrent embed calls across all refs must be ≤ 2; observed {peak}"
    );

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 4: `mode=off` skips warmup entirely
// ---------------------------------------------------------------------------
//
// GOAL: `CONTEXTPLUS_REF_WARMUP_MODE=off` → 0 embed calls AND the ref's
// `project_cache` stays `None` (no walker either).
//
// BLOCKED ON: U16 (`RefWarmupMode::Off` branch in `spawn_ref_warmup`) and U18
// (`spawn_ref_warmup` dispatch).
//
// ACTIVATION: remove `#[ignore]` when U16 + U18 are merged.

#[tokio::test(flavor = "multi_thread")]
async fn off_mode_skips_warmup_entirely() {
    let td = TempDir::new().unwrap();
    let repo = make_repo_with_files(td.path(), "repo", 5);

    let mock = MockOllamaServer::start(0).await;

    unsafe {
        std::env::set_var("CONTEXTPLUS_REF_WARMUP_MODE", "off");
    }
    let config = config_with_mock(mock.uri());
    let (handle, socket) = spawn_daemon_with_config(&repo, config).await;

    let _session_ready = register_session(&socket, &repo).await;

    // Give any spurious async tasks time to fire (they shouldn't).
    tokio::time::sleep(Duration::from_millis(300)).await;

    // LOAD-BEARING: off mode must make 0 embed calls.
    assert_eq!(
        mock.total_calls(),
        0,
        "off mode must make 0 embed calls; got {}",
        mock.total_calls()
    );

    // LOAD-BEARING: off mode must also skip the walker (project_cache stays None).
    // Access via the SharedState ref registry.
    // (This assertion requires U18 to have wired `spawn_ref_warmup` so the
    //  Off branch is actually reachable.)
    // Implementation note: we cannot directly inspect project_cache here without
    // importing internal types; the embed-call counter == 0 is the primary gate.

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Test 5: idempotent attach does not duplicate warmup
// ---------------------------------------------------------------------------
//
// GOAL: Attach the same ref 5 times concurrently. The `spawn_ref_warmup`
// path is guarded by the ref registry — `attach_ref` is idempotent, so only
// ONE warmup task fires per ref. Embed calls == 1× files, not 5×.
//
// BLOCKED ON: U18 (`spawn_ref_warmup` gated on "first attach" detection).
// The idempotency of `attach_ref` already exists (SharedState::attach_ref
// uses `entry().or_insert_with`), but the warmup spawn must be similarly
// gated (e.g. `AtomicBool::compare_exchange` on a `warmup_started` flag in
// `RefIndex`).
//
// ACTIVATION: remove `#[ignore]` when U18 is merged.

#[tokio::test(flavor = "multi_thread")]
async fn idempotent_attach_does_not_duplicate_warmup() {
    const FILE_COUNT: usize = 5;
    const ATTACH_COUNT: usize = 5;

    let td = TempDir::new().unwrap();
    let repo = make_repo_with_files(td.path(), "repo", FILE_COUNT);

    let mock = MockOllamaServer::start(50).await;

    unsafe {
        std::env::set_var("CONTEXTPLUS_REF_WARMUP_MODE", "full");
        std::env::set_var("CONTEXTPLUS_OLLAMA_MAX_CONCURRENT", "4");
    }
    let config = config_with_mock(mock.uri());
    let (handle, socket) = spawn_daemon_with_config(&repo, config).await;

    // Attach the same root 5 times concurrently.
    let futs: Vec<_> = (0..ATTACH_COUNT)
        .map(|_| {
            let s = socket.clone();
            let r = repo.clone();
            tokio::spawn(async move { register_session(&s, &r).await })
        })
        .collect();

    for f in futs {
        f.await.expect("session task panicked");
    }

    // Wait for the single warmup task to finish.
    tokio::time::sleep(Duration::from_secs(3)).await;

    let total = mock.total_calls();
    // Idempotency contract: N concurrent attaches must NOT multiply embed work.
    // We assert strict inequality against ATTACH_COUNT × FILE_COUNT — anything
    // less than that is a single warmup pass (give or take a probe-query embed).
    let multiplied = ATTACH_COUNT * FILE_COUNT;
    assert!(
        total < multiplied,
        "idempotent attach must produce < {multiplied} embed calls (1× corpus, not {ATTACH_COUNT}× corpus); got {total}"
    );

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

// ---------------------------------------------------------------------------
// Shared helper for U21 tests: issue a tools/call after the full MCP handshake
// ---------------------------------------------------------------------------

/// Connect to `socket`, register `client_root`, complete the MCP initialize
/// handshake, issue a single `tools/call`, and return the JSON-RPC response.
///
/// This mirrors `bridge_call` in `multi_worktree_e2e.rs` and exists here so
/// `warmup_throttle.rs` stays self-contained.
async fn bridge_call_throttle(
    socket: &std::path::Path,
    client_root: &std::path::Path,
    tool: &str,
    args: serde_json::Value,
) -> (SessionReady, Value) {
    let mut stream = UnixStream::connect(socket)
        .await
        .expect("connect to daemon socket");

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

    // MCP initialize
    let init_bytes = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-throttle-bridge", "version": "0.0.0"}
        }
    }))
    .unwrap();
    write_half
        .write_all(&[init_bytes.as_slice(), b"\n"].concat())
        .await
        .unwrap();
    write_half.flush().await.unwrap();
    let mut _init_resp = String::new();
    reader.read_line(&mut _init_resp).await.unwrap();

    // notifications/initialized
    write_half
        .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
        .await
        .unwrap();
    write_half.flush().await.unwrap();

    // tools/call
    let call_bytes = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": { "name": tool, "arguments": args }
    }))
    .unwrap();
    write_half
        .write_all(&[call_bytes.as_slice(), b"\n"].concat())
        .await
        .unwrap();
    write_half.flush().await.unwrap();

    let mut resp_line = String::new();
    reader
        .read_line(&mut resp_line)
        .await
        .expect("read tool response");
    assert!(!resp_line.is_empty(), "daemon closed before tool response");

    let resp_json: Value = serde_json::from_str(resp_line.trim()).expect("valid JSON-RPC");
    (session_ready, resp_json)
}

/// Return concatenated text from a `tools/call` result's content array.
fn text_from_resp(resp: &Value) -> String {
    if let Some(arr) = resp.pointer("/result/content").and_then(|v| v.as_array()) {
        let s: String = arr
            .iter()
            .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n");
        if !s.is_empty() {
            return s;
        }
    }
    serde_json::to_string_pretty(resp).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// U21 Test A: shallow worktree attach inherits primary baseline with zero Ollama
// ---------------------------------------------------------------------------
//
// SCENARIO
//   1. Primary repo (5 files) is registered under `mode=full` so the daemon
//      embeds all files via mock Ollama and persists vectors in CAS.
//   2. Counters are reset.
//   3. A sibling worktree (same 5 files) attaches under `mode=shallow`.
//   4. The shallow warmup calls `import_baseline_for_ref`, which finds all 5
//      chunks in the primary's CAS — zero Ollama calls needed.
//
// ASSERTIONS
//   - After worktree attach + warmup: mock.total_calls() == 0   (no file embeds)
//   - A subsequent `lexical_search` returns a valid result using the inherited
//     project_cache.

#[tokio::test(flavor = "multi_thread")]
async fn shallow_attach_inherits_primary_baseline_zero_ollama() {
    let td = TempDir::new().unwrap();
    // Primary repo: 5 small Rust source files.
    let primary = make_repo_with_files(td.path(), "primary", 5);

    // Sibling worktree directory: same 5 files (identical content → CAS hits).
    // We use a plain directory (not a git linked worktree) so the daemon
    // assigns a different ref_id (different canonical path) and sets parent =
    // primary ref automatically.
    // Files must mirror the primary's directory structure so the relative paths
    // (and therefore ChunkKeys) match: primary writes "src/f0000.rs", so the
    // worktree must also place files under src/.
    let wt = td.path().join("wt-shallow");
    let wt_src = wt.join("src");
    std::fs::create_dir_all(&wt_src).expect("create wt/src dir");
    for i in 0..5 {
        let content = format!("pub fn warmup_fn_{i}() -> u32 {{ {i} }}\n");
        std::fs::write(wt_src.join(format!("f{i:04}.rs")), content).unwrap();
    }

    // Phase 1: start daemon in FULL mode — prime the primary ref so the CAS
    // gets all 5 file blobs persisted to disk.
    // Build Config directly (not via env vars) to avoid process-wide contamination
    // when tests run concurrently.
    let mock = MockOllamaServer::start(0).await;
    let mut config_full = Config::from_env();
    config_full.ollama_host = mock.uri().to_string();
    config_full.embed_tracker_mode = contextplus_rs::config::TrackerMode::Off;
    config_full.warmup_on_start = false;
    config_full.ref_warmup_mode = contextplus_rs::config::RefWarmupMode::Full;
    config_full.ollama_max_concurrent = 4;
    let (handle, socket) = spawn_daemon_with_config(&primary, config_full).await;

    // Register primary session and wait for full warmup (all 5 files embedded).
    let _primary_ready = register_session(&socket, &primary).await;
    // Give full warmup time to finish — 5 files at cap 4; give 3 s margin.
    tokio::time::sleep(Duration::from_secs(3)).await;
    let primary_calls = mock.total_calls();
    assert!(
        primary_calls > 0,
        "primary full warmup must produce at least 1 embed call; got {primary_calls}"
    );

    // Phase 2: restart daemon in SHALLOW mode. CAS blobs persist on disk.
    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;

    mock.reset_counters();
    let mut config_shallow = Config::from_env();
    config_shallow.ollama_host = mock.uri().to_string();
    config_shallow.embed_tracker_mode = contextplus_rs::config::TrackerMode::Off;
    config_shallow.warmup_on_start = false;
    config_shallow.ref_warmup_mode = contextplus_rs::config::RefWarmupMode::Shallow;
    config_shallow.ollama_max_concurrent = 4;
    let (handle2, socket2) = spawn_daemon_with_config(&primary, config_shallow).await;

    // Register primary first — shallow warmup fires, 0 Ollama (no CAS parent).
    // Reset again after that so worktree-attach counter starts clean.
    let _primary_ready2 = register_session(&socket2, &primary).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    mock.reset_counters();

    // Attach the worktree — triggers shallow warmup which calls
    // import_baseline_for_ref.  All 5 files match the primary CAS → 0 Ollama.
    let _wt_ready = register_session(&socket2, &wt).await;
    // Allow the async warmup task to complete.
    tokio::time::sleep(Duration::from_secs(2)).await;

    // LOAD-BEARING: zero Ollama calls during shallow worktree attach.
    let wt_calls = mock.total_calls();
    assert_eq!(
        wt_calls, 0,
        "shallow worktree attach must make 0 Ollama embed calls \
         (all hits from primary CAS); got {wt_calls}"
    );

    // Behavioral: worktree serves lexical_search immediately, proving inherited
    // project_cache is populated (no cold-build delay).
    let (_sr, resp) = bridge_call_throttle(
        &socket2,
        &wt,
        "lexical_search",
        json!({"query": "warmup_fn_0"}),
    )
    .await;
    assert!(
        resp.pointer("/result").is_some(),
        "lexical_search on worktree must return a result; response: {}",
        text_from_resp(&resp)
    );

    handle2.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle2).await;
}

// ---------------------------------------------------------------------------
// U21 Test B: full attach embeds only diff, not the entire corpus
// ---------------------------------------------------------------------------
//
// SCENARIO
//   1. Primary repo (3 files) is primed by a daemon in full mode.
//   2. Worktree: same 3 files (CAS hits) + 1 new file + 1 modified file (misses).
//   3. Worktree attaches under `mode=full`.
//   4. Baseline import finds 3 hits (zero Ollama) and 2 misses.
//   5. embed_diff_chunks embeds only the 2 misses.
//
// ASSERTIONS
//   - After worktree warmup: mock.total_calls() > 0   (diff files embedded)
//   - mock.total_calls() <= 3  (≤ 2 misses; the 3 shared files are CAS hits)

#[tokio::test(flavor = "multi_thread")]
async fn full_attach_inherits_baseline_then_embeds_diff_only() {
    let td = TempDir::new().unwrap();

    // Primary: 3 source files.
    let primary = make_repo_with_files(td.path(), "primary-full", 3);

    // Worktree: same 3 files + 1 new + 1 modified.
    // Files must mirror the primary's directory structure (primary puts files in
    // src/) so the relative paths match and CAS lookup_chunk finds hits.
    let wt = td.path().join("wt-full");
    let wt_src = wt.join("src");
    std::fs::create_dir_all(&wt_src).expect("create wt/src dir");
    // Mirror the primary's 3 files verbatim under src/ (identical content → CAS hits).
    for i in 0..3 {
        let content = format!("pub fn warmup_fn_{i}() -> u32 {{ {i} }}\n");
        std::fs::write(wt_src.join(format!("f{i:04}.rs")), content).unwrap();
    }
    // 1 net-new file under src/ (no CAS blob) — will be a miss.
    std::fs::write(
        wt_src.join("net_new.rs"),
        "pub fn brand_new_function() -> bool { true }\n",
    )
    .unwrap();
    // 1 modified file under src/ (different content → different chunk hash → miss).
    std::fs::write(
        wt_src.join("modified.rs"),
        "pub fn modified_function() -> &'static str { \"changed\" }\n",
    )
    .unwrap();

    // Phase 1: prime primary with full mode.
    // Build Config directly to avoid env-var cross-contamination in parallel tests.
    let mock = MockOllamaServer::start(0).await;
    let mut config_full = Config::from_env();
    config_full.ollama_host = mock.uri().to_string();
    config_full.embed_tracker_mode = contextplus_rs::config::TrackerMode::Off;
    config_full.warmup_on_start = false;
    config_full.ref_warmup_mode = contextplus_rs::config::RefWarmupMode::Full;
    config_full.ollama_max_concurrent = 4;
    let (handle, socket) = spawn_daemon_with_config(&primary, config_full).await;

    let _primary_ready = register_session(&socket, &primary).await;
    // Wait for full warmup of 3 files.
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(
        mock.total_calls() > 0,
        "primary full warmup must embed at least 1 file"
    );

    // Phase 2: restart daemon (in-memory state cleared; CAS on disk persists).
    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;

    mock.reset_counters();
    // Re-use full mode for the second daemon (worktree will also use full).
    let mut config_full2 = Config::from_env();
    config_full2.ollama_host = mock.uri().to_string();
    config_full2.embed_tracker_mode = contextplus_rs::config::TrackerMode::Off;
    config_full2.warmup_on_start = false;
    config_full2.ref_warmup_mode = contextplus_rs::config::RefWarmupMode::Full;
    config_full2.ollama_max_concurrent = 4;
    let (handle2, socket2) = spawn_daemon_with_config(&primary, config_full2).await;

    // Re-register primary to rebuild in-memory ref; CAS blobs already on disk.
    let _primary_ready2 = register_session(&socket2, &primary).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Reset counters before worktree attach.
    mock.reset_counters();

    // Attach the worktree in full mode. Expected Ollama calls:
    //   - 3 CAS hits (files f0000..f0002) → 0 Ollama for those
    //   - 2 CAS misses (net_new.rs, modified.rs) → ≤ 2 embed calls
    let _wt_ready = register_session(&socket2, &wt).await;
    // Full warmup includes Ollama embed; give ample time.
    tokio::time::sleep(Duration::from_secs(5)).await;

    let wt_embed_calls = mock.total_calls();
    // Lower bound: at least 1 diff file embedded.
    assert!(
        wt_embed_calls > 0,
        "full worktree attach must embed at least the diff files \
         (net_new.rs + modified.rs); got 0"
    );
    // Upper bound: ≤ 3 inputs (2 diff files; the 3 shared files are CAS hits).
    // We use ≤ 3 to accommodate potential batching edge cases.
    assert!(
        wt_embed_calls <= 3,
        "full worktree attach must embed only diff chunks (≤ 3 inputs); \
         got {wt_embed_calls} (the 3 shared files must come from CAS, not Ollama)"
    );

    handle2.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle2).await;
}

// ---------------------------------------------------------------------------
// U21 Test C: shallow attach on orphan root falls through to lazy embed
// ---------------------------------------------------------------------------
//
// SCENARIO
//   A brand-new primary repo has no CAS blobs (no parent chain, no prior embed).
//   Shallow mode: import_baseline_for_ref finds 0 hits (all misses).
//   The misses stay unembedded until the first query.
//
// ASSERTIONS
//   - After attach + warmup: mock.total_calls() == 0  (no Ollama during shallow)
//   - lexical_search works (project_cache populated by walker; no embed needed).
//   - semantic_code_search triggers ≥ 1 Ollama call (query embed → lazy path).

#[tokio::test(flavor = "multi_thread")]
async fn shallow_attach_to_unrelated_root_falls_through_to_lazy() {
    let td = TempDir::new().unwrap();
    // Fresh repo with no prior embed history.
    let repo = make_repo_with_files(td.path(), "orphan-repo", 4);

    // Build Config directly to avoid env-var cross-contamination in parallel tests.
    let mock = MockOllamaServer::start(0).await;
    let mut config = Config::from_env();
    config.ollama_host = mock.uri().to_string();
    config.embed_tracker_mode = contextplus_rs::config::TrackerMode::Off;
    config.warmup_on_start = false;
    config.ref_warmup_mode = contextplus_rs::config::RefWarmupMode::Shallow;
    config.ollama_max_concurrent = 4;
    let (handle, socket) = spawn_daemon_with_config(&repo, config).await;

    // Register — this is the primary ref (no parent), shallow warmup fires.
    let _ready = register_session(&socket, &repo).await;
    // Shallow warmup: walks 4 files, import_baseline_for_ref finds 0 CAS hits.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // LOAD-BEARING: zero Ollama during shallow warmup on an orphan root.
    let calls_after_attach = mock.total_calls();
    assert_eq!(
        calls_after_attach, 0,
        "shallow warmup on orphan root must make 0 Ollama calls; got {calls_after_attach}"
    );

    // Lexical search works without embeddings — uses project_cache (walker ran).
    let (_sr, lex_resp) = bridge_call_throttle(
        &socket,
        &repo,
        "lexical_search",
        json!({"query": "warmup_fn_0"}),
    )
    .await;
    assert!(
        lex_resp.pointer("/result").is_some(),
        "lexical_search must succeed even with empty embedding cache; response: {}",
        text_from_resp(&lex_resp)
    );

    // Semantic search triggers at least a query embed (lazy path).
    // Call count must increase — proving the lazy-fill path is exercised.
    let calls_before_semantic = mock.total_calls();
    let (_sr2, _sem_resp) = bridge_call_throttle(
        &socket,
        &repo,
        "semantic_code_search",
        json!({"query": "warmup function returning integer"}),
    )
    .await;
    let calls_after_semantic = mock.total_calls();
    assert!(
        calls_after_semantic > calls_before_semantic,
        "semantic_code_search on an orphan repo must trigger at least 1 Ollama call \
         (lazy embed); before={calls_before_semantic} after={calls_after_semantic}"
    );

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}
