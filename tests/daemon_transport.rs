//! Integration tests for the per-workspace daemon transport.
//!
//! These tests bring up the daemon in-process (no fork — we don't want test
//! flakes around child-process orchestration) and connect raw `UnixStream`
//! clients to the bound socket. The MCP wire protocol is exercised by hand
//! via newline-delimited JSON-RPC, which is the framing rmcp's
//! `AsyncRwTransport` uses on stdio + sockets.
//!
//! Smoke coverage:
//! 1. End-to-end MCP round trip over a Unix socket — initialize, then
//!    `tools/call get_context_tree` returns a non-empty body.
//! 2. Two concurrent clients on the same daemon get correct, independent
//!    responses (single daemon serving multiple sessions).
//! 3. Lock contention: two `acquire_lock` attempts in the same process
//!    serialize correctly via `flock`.
//! 4. Stale-socket recovery: a leftover socket file is replaced cleanly.
//! 5. Client `connect_or_spawn` against a live daemon does not double-spawn.

#![cfg(unix)]

use std::path::Path;
use std::time::Duration;

use contextplus_rs::config::Config;
use contextplus_rs::server::ContextPlusServer;
use contextplus_rs::transport::daemon::{self, AcquireOutcome, LockGuard};
use contextplus_rs::transport::paths;
use serde_json::json;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// Bring up a daemon bound to a fresh tempdir socket. Returns the tempdir,
/// the spawned task handle, and the absolute socket path.
async fn spawn_daemon_for_test() -> (TempDir, tokio::task::JoinHandle<()>, std::path::PathBuf) {
    let dir = TempDir::new().unwrap();
    let root = dir.path().to_path_buf();

    let lock = match daemon::acquire_lock(&root).expect("acquire_lock") {
        AcquireOutcome::Acquired(l) => l,
        AcquireOutcome::AlreadyRunning => panic!("fresh tempdir should never be contended"),
    };
    let listener = daemon::bind_listener(&root).expect("bind_listener");
    daemon::write_pid_file(&root);

    let socket_path = paths::daemon_socket_path(&root);
    let pid_path = paths::daemon_pid_path(&root);

    let server = ContextPlusServer::new(root.clone(), Config::from_env());

    let handle = tokio::spawn(run_daemon_task(
        server,
        listener,
        socket_path.clone(),
        pid_path,
        lock,
    ));

    // Wait for the socket file to appear.
    for _ in 0..50 {
        if socket_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(socket_path.exists(), "daemon never bound its socket");
    (dir, handle, socket_path)
}

/// Wrap `daemon::run` so the spawned task can own the lock guard. We disable
/// the idle timer (`0`) so tests don't race against shutdown.
async fn run_daemon_task(
    server: ContextPlusServer,
    listener: tokio::net::UnixListener,
    socket_path: std::path::PathBuf,
    pid_path: std::path::PathBuf,
    lock: LockGuard,
) {
    let _ = daemon::run(server, listener, socket_path, pid_path, 0, lock).await;
}

/// Speak MCP over an accepted Unix stream by hand (newline-delimited JSON).
/// Drives initialize + `tools/call get_context_tree` and returns the response
/// body's first text chunk.
async fn round_trip_get_context_tree(stream: UnixStream) -> String {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    // Step 1: initialize.
    let init = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.0.0"}
        }
    });
    write_line(&mut write_half, &init).await;
    let _init_resp = read_line(&mut reader).await;

    // Required notification post-handshake.
    let notif = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });
    write_line(&mut write_half, &notif).await;

    // Step 2: tools/call get_context_tree.
    let call = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "get_context_tree",
            "arguments": {}
        }
    });
    write_line(&mut write_half, &call).await;
    let resp = read_line(&mut reader).await;

    // Extract text from result.content[0].text.
    let parsed: serde_json::Value =
        serde_json::from_str(&resp).expect("server sent valid JSON-RPC");
    parsed
        .pointer("/result/content/0/text")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            panic!("expected /result/content/0/text in response, got: {resp}");
        })
}

async fn write_line<W: tokio::io::AsyncWrite + Unpin>(w: &mut W, value: &serde_json::Value) {
    let mut bytes = serde_json::to_vec(value).unwrap();
    bytes.push(b'\n');
    w.write_all(&bytes).await.expect("write_all");
    w.flush().await.expect("flush");
}

async fn read_line<R: tokio::io::AsyncBufRead + Unpin>(r: &mut R) -> String {
    let mut buf = String::new();
    let n = r
        .read_line(&mut buf)
        .await
        .expect("read_line on Unix socket");
    assert!(n > 0, "EOF before reply");
    buf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn daemon_serves_get_context_tree_over_unix_socket() {
    let (dir, handle, socket_path) = spawn_daemon_for_test().await;
    std::fs::write(dir.path().join("hello.txt"), "hello daemon").unwrap();

    let stream = UnixStream::connect(&socket_path)
        .await
        .expect("connect to daemon");
    let text = round_trip_get_context_tree(stream).await;
    assert!(!text.is_empty(), "tree response should be non-empty");

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_serves_two_concurrent_clients() {
    let (dir, handle, socket_path) = spawn_daemon_for_test().await;
    std::fs::write(dir.path().join("a.txt"), "alpha").unwrap();

    let s1 = UnixStream::connect(&socket_path).await.unwrap();
    let s2 = UnixStream::connect(&socket_path).await.unwrap();

    let (t1, t2) = tokio::join!(
        round_trip_get_context_tree(s1),
        round_trip_get_context_tree(s2)
    );
    assert!(!t1.is_empty(), "client 1 got empty tree");
    assert!(!t2.is_empty(), "client 2 got empty tree");

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
}

#[tokio::test]
async fn lock_serializes_two_daemon_attempts() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();

    let g1 = match daemon::acquire_lock(root).unwrap() {
        AcquireOutcome::Acquired(g) => g,
        AcquireOutcome::AlreadyRunning => panic!("first attempt must win"),
    };
    match daemon::acquire_lock(root).unwrap() {
        AcquireOutcome::AlreadyRunning => {}
        AcquireOutcome::Acquired(_) => panic!("second attempt must be contended"),
    }
    drop(g1);
    match daemon::acquire_lock(root).unwrap() {
        AcquireOutcome::Acquired(_) => {}
        AcquireOutcome::AlreadyRunning => {
            panic!("lock not released after first guard dropped")
        }
    }
}

#[tokio::test]
async fn bind_recovers_from_stale_socket_file() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let mcp_data = root.join(paths::MCP_DATA_DIR);
    std::fs::create_dir_all(&mcp_data).unwrap();

    let stale = paths::daemon_socket_path(root);
    std::fs::write(&stale, b"stale junk").unwrap();
    let listener = daemon::bind_listener(root).expect("rebind over stale socket");
    drop(listener);
}

/// `connect_or_spawn` should connect to an already-running daemon without
/// spawning a second one. We assert by counting that the socket is unchanged
/// before and after the call.
#[tokio::test(flavor = "multi_thread")]
async fn client_connect_to_existing_daemon_skips_spawn() {
    use contextplus_rs::transport::client;

    let (dir, handle, socket_path) = spawn_daemon_for_test().await;
    let root = dir.path().to_path_buf();

    // Pre-call mtime gives us a coarse fingerprint for "did the socket get
    // recreated?". If a second daemon spawned, we'd see the file replaced.
    let before = std::fs::metadata(&socket_path).unwrap();

    let stream = client::connect_or_spawn(&root)
        .await
        .expect("connect_or_spawn against live daemon");
    drop(stream);

    let after = std::fs::metadata(&socket_path).unwrap();
    // inode equality is a stronger check than mtime — a re-bind would create
    // a brand-new inode.
    use std::os::unix::fs::MetadataExt;
    assert_eq!(
        before.ino(),
        after.ino(),
        "client::connect_or_spawn should reuse the existing daemon socket",
    );

    handle.abort();
    let _ = tokio::time::timeout(Duration::from_secs(1), handle).await;
    let _ = Path::new(&root);
}

#[test]
fn path_helpers_consistent() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let s = paths::daemon_socket_path(root);
    let l = paths::daemon_lock_path(root);
    let p = paths::daemon_pid_path(root);
    assert_eq!(s.parent(), l.parent());
    assert_eq!(l.parent(), p.parent());
}
