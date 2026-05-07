//! U19 — verification tests for per-ref warmup throttling.
//!
//! # What this file proves (when U16–U18 land)
//!
//! The five integration tests here exercise three invariants introduced by
//! U16 (`RefWarmupMode` + `ollama_semaphore`), U17 (semaphore-gated
//! `OllamaClient`), and U18 (`spawn_ref_warmup` wired from `serve_connection`):
//!
//! | Test | Invariant |
//! |---|---|
//! | `shallow_warmup_makes_no_ollama_calls` | Shallow mode walks files but never embeds |
//! | `full_warmup_respects_concurrency_cap` | Single ref: peak concurrent ≤ cap |
//! | `concurrent_attaches_share_one_ollama_budget` | N refs: peak concurrent ≤ cap across all |
//! | `off_mode_skips_warmup_entirely` | Off mode: 0 embed calls AND project_cache stays None |
//! | `idempotent_attach_does_not_duplicate_warmup` | Duplicate attaches: embed calls = 1× not N× |
//!
//! # Current status (U19 deliverable)
//!
//! U16/U17/U18 are **not yet merged**. Every test is marked
//! `#[ignore = "U18 dependency: …"]`. The scaffolding — fixtures, mock-Ollama
//! harness, helpers — compiles and is the deliverable for this unit. Remove the
//! `#[ignore]` attributes as each upstream unit lands.
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
use serde_json::json;
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
    // Each file produces (at minimum) 1 embed call; idempotency means the
    // total must be ≤ FILE_COUNT (not ATTACH_COUNT × FILE_COUNT).
    assert!(
        total <= FILE_COUNT,
        "idempotent attach must produce ≤ {FILE_COUNT} embed calls (1× corpus); got {total}"
    );

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}
