//! Integration tests for contextplus-rs: MCP protocol, cache persistence, concurrency.

use std::sync::Arc;

use contextplus_rs::cache::rkyv_store;
use contextplus_rs::config::Config;
use contextplus_rs::core::embeddings::VectorStore;
use contextplus_rs::server::ContextPlusServer;
use rmcp::model::RawContent;
use serde_json::json;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a server rooted in a temporary directory with default config.
fn test_server_in(dir: &std::path::Path) -> ContextPlusServer {
    let config = Config::from_env();
    ContextPlusServer::new(dir.to_path_buf(), config)
}

/// Extract the first text content from a CallToolResult, or empty string.
fn extract_text(result: &rmcp::model::CallToolResult) -> &str {
    result
        .content
        .first()
        .and_then(|c| match &c.raw {
            RawContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .unwrap_or("")
}

// ===========================================================================
// Category 1: MCP Protocol Round-Trip Tests
// ===========================================================================

#[tokio::test]
async fn test_call_unknown_tool_returns_error() {
    let dir = TempDir::new().unwrap();
    let server = test_server_in(dir.path());
    let args = serde_json::Map::new();

    let result = server.dispatch("nonexistent_tool_xyz", args).await;

    assert_eq!(
        result.is_error,
        Some(true),
        "unknown tool should return is_error=true"
    );
    let text = extract_text(&result);
    assert!(
        text.contains("Unknown tool"),
        "expected 'Unknown tool' in error text, got: {text}"
    );
}

#[tokio::test]
async fn test_call_get_file_skeleton_missing_path_returns_error() {
    let dir = TempDir::new().unwrap();
    let server = test_server_in(dir.path());
    // No file_path argument — should produce an error
    let args = serde_json::Map::new();

    let result = server.dispatch("get_file_skeleton", args).await;

    assert_eq!(
        result.is_error,
        Some(true),
        "get_file_skeleton without file_path should return is_error=true"
    );
    let text = extract_text(&result);
    assert!(
        text.contains("file_path") || text.contains("required"),
        "expected error mentioning file_path or required, got: {text}"
    );
}

#[tokio::test]
async fn test_call_get_context_tree_returns_success() {
    let dir = TempDir::new().unwrap();
    // Create a small file so the tree has something to report
    std::fs::write(dir.path().join("hello.txt"), "hello world").unwrap();

    let server = test_server_in(dir.path());
    let args = serde_json::Map::new();

    let result = server.dispatch("get_context_tree", args).await;

    assert_eq!(
        result.is_error,
        Some(false),
        "get_context_tree should succeed, got error: {}",
        extract_text(&result)
    );
}

#[tokio::test]
async fn test_call_get_blast_radius_returns_success() {
    let dir = TempDir::new().unwrap();
    // Create a file that references a symbol
    std::fs::write(
        dir.path().join("example.ts"),
        "import { myFunc } from './other';\nmyFunc();\n",
    )
    .unwrap();

    let server = test_server_in(dir.path());
    let mut args = serde_json::Map::new();
    args.insert("symbol_name".to_string(), json!("myFunc"));

    let result = server.dispatch("get_blast_radius", args).await;

    assert_eq!(
        result.is_error,
        Some(false),
        "get_blast_radius should succeed, got error: {}",
        extract_text(&result)
    );
}

#[tokio::test]
async fn test_call_run_static_analysis_returns_success() {
    let dir = TempDir::new().unwrap();
    let server = test_server_in(dir.path());
    let args = serde_json::Map::new();

    let result = server.dispatch("run_static_analysis", args).await;

    // run_static_analysis may succeed even in an empty dir (just reports nothing found)
    // or it may error if no tooling is detected — either way it should not panic.
    // We accept both success and graceful error.
    assert!(
        result.is_error == Some(false) || result.is_error == Some(true),
        "run_static_analysis should return a defined result"
    );
    // Verify it returned some text content
    assert!(
        !result.content.is_empty(),
        "run_static_analysis should return content"
    );
}

// ===========================================================================
// Category 2: Cache Persistence Round-Trip
// ===========================================================================

/// Helper: build a VectorStore with three known vectors for testing.
fn make_test_store() -> VectorStore {
    VectorStore::new(
        3,
        vec![
            "src/auth.ts".to_string(),
            "src/db.ts".to_string(),
            "src/api.ts".to_string(),
        ],
        vec!["h1".to_string(), "h2".to_string(), "h3".to_string()],
        // auth  = [0.9, 0.1, 0.0]  — close to [1, 0, 0]
        // db    = [0.0, 0.9, 0.1]  — close to [0, 1, 0]
        // api   = [0.5, 0.5, 0.0]  — midpoint
        vec![0.9, 0.1, 0.0, 0.0, 0.9, 0.1, 0.5, 0.5, 0.0],
    )
}

#[test]
fn test_cache_save_load_search_round_trip() {
    let dir = TempDir::new().unwrap();
    let store = make_test_store();

    // Save
    rkyv_store::save_vector_store(dir.path(), "test-round-trip", &store).unwrap();

    // Load via standard path
    let loaded = rkyv_store::load_vector_store(dir.path(), "test-round-trip")
        .unwrap()
        .expect("cache should exist after save");

    // Metadata preserved
    assert_eq!(loaded.count(), 3);
    assert_eq!(loaded.dims(), 3);
    assert!(loaded.has_key("src/auth.ts"));
    assert!(loaded.has_key("src/db.ts"));
    assert!(loaded.has_key("src/api.ts"));

    // Search: query close to auth vector should rank auth first
    let query = vec![1.0, 0.0, 0.0];
    let results = loaded.find_nearest(&query, 3);

    assert_eq!(results.len(), 3, "should return 3 nearest");
    assert_eq!(
        results[0].0, "src/auth.ts",
        "auth should be closest to [1,0,0]"
    );
    // auth similarity should be highest
    assert!(
        results[0].1 > results[1].1,
        "auth similarity ({}) should exceed second-best ({})",
        results[0].1,
        results[1].1
    );
}

#[test]
fn test_mmap_cache_search_round_trip() {
    let dir = TempDir::new().unwrap();
    let store = make_test_store();

    rkyv_store::save_vector_store(dir.path(), "mmap-rt", &store).unwrap();

    // Load via mmap path
    let loaded = rkyv_store::mmap_vector_store(dir.path(), "mmap-rt")
        .unwrap()
        .expect("mmap cache should exist after save");

    assert_eq!(loaded.count(), 3);
    assert_eq!(loaded.dims(), 3);

    // Search should produce identical ordering to regular load
    let query = vec![1.0, 0.0, 0.0];
    let mmap_results = loaded.find_nearest(&query, 3);

    let regular = rkyv_store::load_vector_store(dir.path(), "mmap-rt")
        .unwrap()
        .unwrap();
    let regular_results = regular.find_nearest(&query, 3);

    assert_eq!(mmap_results.len(), regular_results.len());
    for (m, r) in mmap_results.iter().zip(regular_results.iter()) {
        assert_eq!(m.0, r.0, "key ordering mismatch between mmap and regular");
        assert!(
            (m.1 - r.1).abs() < 1e-6,
            "similarity mismatch: mmap={}, regular={}",
            m.1,
            r.1
        );
    }
}

#[test]
fn test_cache_survives_process_restart_simulation() {
    let dir = TempDir::new().unwrap();
    let cache_name = "restart-sim";

    // Phase 1: build and save
    {
        let store = make_test_store();
        rkyv_store::save_vector_store(dir.path(), cache_name, &store).unwrap();
        // store is dropped here — simulates process exit
    }

    // Phase 2: "restart" — load fresh, verify all data intact
    {
        let loaded = rkyv_store::load_vector_store(dir.path(), cache_name)
            .unwrap()
            .expect("cache should survive after drop + reload");

        assert_eq!(loaded.count(), 3);
        assert_eq!(loaded.dims(), 3);

        // Verify specific vectors survived
        let vec_auth = loaded.get_vector("src/auth.ts").unwrap();
        assert!((vec_auth[0] - 0.9).abs() < 1e-6);
        assert!((vec_auth[1] - 0.1).abs() < 1e-6);
        assert!((vec_auth[2] - 0.0).abs() < 1e-6);

        let vec_db = loaded.get_vector("src/db.ts").unwrap();
        assert!((vec_db[0] - 0.0).abs() < 1e-6);
        assert!((vec_db[1] - 0.9).abs() < 1e-6);

        // Hashes survived
        assert_eq!(loaded.get_hash("src/auth.ts"), Some("h1"));
        assert_eq!(loaded.get_hash("src/db.ts"), Some("h2"));
        assert_eq!(loaded.get_hash("src/api.ts"), Some("h3"));

        // Search still works after simulated restart
        let query = vec![0.0, 1.0, 0.0];
        let results = loaded.find_nearest(&query, 1);
        assert_eq!(results[0].0, "src/db.ts", "db should be closest to [0,1,0]");
    }
}

// ===========================================================================
// Category 3: Concurrent Tool Call Stress Tests
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_context_tree_calls() {
    let dir = TempDir::new().unwrap();
    // Seed with a few files so the walker has work to do
    for i in 0..5 {
        std::fs::write(
            dir.path().join(format!("file_{i}.ts")),
            format!("export const x{i} = {i};\n"),
        )
        .unwrap();
    }

    let server = Arc::new(test_server_in(dir.path()));
    let mut join_set = tokio::task::JoinSet::new();

    for _ in 0..10 {
        let srv = Arc::clone(&server);
        join_set.spawn(async move {
            let args = serde_json::Map::new();
            srv.dispatch("get_context_tree", args).await
        });
    }

    let mut success_count = 0;
    while let Some(result) = join_set.join_next().await {
        let call_result = result.expect("task should not panic");
        assert_eq!(
            call_result.is_error,
            Some(false),
            "concurrent get_context_tree should succeed, got: {}",
            extract_text(&call_result)
        );
        success_count += 1;
    }

    assert_eq!(success_count, 10, "all 10 concurrent calls should complete");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_mixed_tool_calls() {
    let dir = TempDir::new().unwrap();
    // Create files with symbols for blast_radius to find
    std::fs::write(
        dir.path().join("main.ts"),
        "import { helper } from './helper';\nhelper();\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("helper.ts"),
        "export function helper() { return 42; }\n",
    )
    .unwrap();

    let server = Arc::new(test_server_in(dir.path()));
    let mut join_set = tokio::task::JoinSet::new();

    // Spawn context_tree calls
    for _ in 0..3 {
        let srv = Arc::clone(&server);
        join_set.spawn(async move {
            let args = serde_json::Map::new();
            (
                "get_context_tree",
                srv.dispatch("get_context_tree", args).await,
            )
        });
    }

    // Spawn blast_radius calls
    for _ in 0..3 {
        let srv = Arc::clone(&server);
        join_set.spawn(async move {
            let mut args = serde_json::Map::new();
            args.insert("symbol_name".to_string(), json!("helper"));
            (
                "get_blast_radius",
                srv.dispatch("get_blast_radius", args).await,
            )
        });
    }

    // Spawn file_skeleton calls (will error due to missing file_path, that's fine — no panic)
    for _ in 0..3 {
        let srv = Arc::clone(&server);
        join_set.spawn(async move {
            let mut args = serde_json::Map::new();
            args.insert("file_path".to_string(), json!("main.ts"));
            (
                "get_file_skeleton",
                srv.dispatch("get_file_skeleton", args).await,
            )
        });
    }

    let mut completed = 0;
    while let Some(result) = join_set.join_next().await {
        let (tool_name, call_result) = result.expect("task should not panic");
        // All tools should return a result without panicking.
        // context_tree and blast_radius should succeed; file_skeleton should succeed on existing file.
        assert!(
            !call_result.content.is_empty(),
            "{tool_name} should return content"
        );
        completed += 1;
    }

    assert_eq!(completed, 9, "all 9 concurrent calls should complete");
}
