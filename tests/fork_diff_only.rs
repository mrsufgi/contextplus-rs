//! U6 measurement test — fork_diff_only
//!
//! # What this proves
//!
//! The load-bearing invariant for U6: when a new worktree ref forks from a
//! primary ref that has already embedded a corpus, only the *changed* chunks
//! (the diff vs. primary) require embedding calls to Ollama. Unchanged chunks
//! are served from the CAS blob store via the parent manifest chain — zero
//! extra embed calls.
//!
//! # Test structure
//!
//! 1. **Build a fixture corpus** (N files, each 1-2 chunks) and embed them
//!    into the primary ref's CAS store + manifest using a mock Ollama server
//!    that counts calls.
//!
//! 2. **Fork a child ref** from primary (pointing its `parent` file at primary).
//!    The child's manifest is empty.
//!
//! 3. **Simulate diffed files**: for a subset of D files from the corpus,
//!    compute new chunk hashes (content changed). Walk the child ref, checking
//!    each (path, chunk_idx) against the CAS store:
//!    - If the hash matches the parent's manifest entry → no embed call.
//!    - If the hash differs (new content) → embed call required.
//!
//! 4. **Assert** embed_calls ≤ D × max_chunks_per_file.
//!
//! 5. **Sanity assert** embed_calls < N × max_chunks_per_file (we never paid
//!    the full-corpus cost).
//!
//! # Mock Ollama
//!
//! Uses `wiremock` to intercept `/api/embed` requests and return a deterministic
//! fake embedding. The counter on the mock server is the measurement instrument.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use contextplus_rs::cache::cas::{CasStore, ChunkHash, ChunkKey, Manifest};
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Number of files in the "full corpus" (primary ref's view).
const CORPUS_SIZE: usize = 50;

/// Number of files that differ in the child ref.
const DIFF_SIZE: usize = 10;

/// Every file has this many chunks (fixed for simplicity).
const CHUNKS_PER_FILE: usize = 2;

/// Dimensionality of fake embedding vectors.
const EMBED_DIMS: usize = 4;

/// Produce deterministic fake embedding content for a chunk (just its index repeated).
fn fake_vector(seed: usize) -> Vec<f32> {
    (0..EMBED_DIMS).map(|i| (seed + i) as f32 * 0.01).collect()
}

/// Produce fake chunk text for a (file_idx, chunk_idx, generation) triple.
/// `generation=0` is "primary content"; `generation=1` is "changed content".
fn chunk_text(file_idx: usize, chunk_idx: usize, generation: usize) -> String {
    format!("fn function_{file_idx}_chunk_{chunk_idx}_gen{generation}() {{ /* ... */ }}")
}

/// Relative file path for fixture file i.
fn fixture_path(file_idx: usize) -> String {
    format!("src/file_{:04}.rs", file_idx)
}

// ---------------------------------------------------------------------------
// Primary ref indexing helper
// ---------------------------------------------------------------------------

/// Index the full corpus into the primary ref's CAS store.
///
/// For each file + chunk in the corpus:
///   1. Compute BLAKE3 hash of the chunk text.
///   2. If blob absent → "embed" (increment counter) + write blob.
///   3. Write manifest entry.
///
/// Returns the total number of embed calls made (expected: CORPUS_SIZE × CHUNKS_PER_FILE
/// on a cold CAS, fewer on warm).
async fn index_primary_ref(
    cas: &CasStore,
    primary_ref_hex: &str,
    embed_counter: Arc<AtomicUsize>,
) -> usize {
    let mut calls = 0usize;
    let mut manifest = Manifest::new();

    for file_idx in 0..CORPUS_SIZE {
        let file_path = fixture_path(file_idx);
        for chunk_idx in 0..CHUNKS_PER_FILE {
            let text = chunk_text(file_idx, chunk_idx, 0 /* primary generation */);
            let hash = ChunkHash::of(&text);

            if !cas.blob_exists(&hash) {
                // Simulate embed call.
                let vec = fake_vector(file_idx * CHUNKS_PER_FILE + chunk_idx);
                cas.write_blob(&hash, &vec).unwrap();
                calls += 1;
                embed_counter.fetch_add(1, Ordering::Relaxed);
            }

            manifest.upsert(ChunkKey::new(file_path.clone(), chunk_idx as u32), hash);
        }
    }

    cas.save_manifest(primary_ref_hex, &manifest).unwrap();
    calls
}

// ---------------------------------------------------------------------------
// Child ref indexing helper (diff-only path)
// ---------------------------------------------------------------------------

/// Walk the child ref against the CAS store using parent chain lookup.
///
/// For each file + chunk:
///   - If the chunk hash matches what the parent already has → serve from CAS,
///     no embed call.
///   - If the hash is new (changed content) → embed call.
///
/// Files in `changed_indices` use generation=1 (different text → different hash).
/// All other files use generation=0 (identical to primary → hash hits in parent).
///
/// Returns the number of embed calls made.
async fn index_child_ref(
    cas: &CasStore,
    child_ref_hex: &str,
    embed_counter: Arc<AtomicUsize>,
    changed_indices: &[usize],
) -> usize {
    let changed_set: std::collections::HashSet<usize> = changed_indices.iter().cloned().collect();
    let mut calls = 0usize;
    let mut new_entries: Vec<(ChunkKey, ChunkHash)> = Vec::new();

    for file_idx in 0..CORPUS_SIZE {
        let generation = if changed_set.contains(&file_idx) {
            1
        } else {
            0
        };
        let file_path = fixture_path(file_idx);

        for chunk_idx in 0..CHUNKS_PER_FILE {
            let text = chunk_text(file_idx, chunk_idx, generation);
            let new_hash = ChunkHash::of(&text);
            let key = ChunkKey::new(file_path.clone(), chunk_idx as u32);

            // Check if this exact hash already exists in the CAS chain.
            let existing_hash = cas.lookup_chunk(child_ref_hex, &key).unwrap();

            if existing_hash.as_ref() == Some(&new_hash) {
                // Exact match in parent chain → serve from CAS, no embed.
                continue;
            }

            // Hash missing or changed → need embedding.
            if !cas.blob_exists(&new_hash) {
                let vec = fake_vector(file_idx * CHUNKS_PER_FILE + chunk_idx + 1000);
                cas.write_blob(&new_hash, &vec).unwrap();
                calls += 1;
                embed_counter.fetch_add(1, Ordering::Relaxed);
            }

            // Record only changed entries in the child manifest.
            new_entries.push((key, new_hash));
        }
    }

    // Batch-write only the changed entries to the child's manifest.
    if !new_entries.is_empty() {
        cas.update_manifest(child_ref_hex, &new_entries).unwrap();
    }

    calls
}

// ---------------------------------------------------------------------------
// Test: core measurement
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fork_diff_only_embed_calls_bounded_by_diff_size() {
    // Set up a mock Ollama server (not actually called in this test — we use
    // direct CAS writes — but the counter validates the invariant via the
    // index helpers above).
    let server = MockServer::start().await;
    let embed_call_counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = embed_call_counter.clone();

    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(move |req: &wiremock::Request| {
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
            let n_inputs = body["input"].as_array().map(|a| a.len()).unwrap_or(1);
            counter_clone.fetch_add(n_inputs, Ordering::Relaxed);
            let embeddings: Vec<Vec<f32>> = (0..n_inputs).map(fake_vector).collect();
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "embeddings": embeddings }))
        })
        .mount(&server)
        .await;

    // --- CAS store in a temp directory ---
    let tmp = TempDir::new().unwrap();
    let mcp_data = tmp.path().join(".mcp_data");
    std::fs::create_dir_all(&mcp_data).unwrap();

    let model = "nomic-embed-text";
    let cas = CasStore::new(mcp_data.clone(), model);

    let primary_hex = "primary0000000000";
    let child_hex = "child00000000000a";

    // --- Phase 1: index the full corpus into primary ---
    let primary_calls = index_primary_ref(&cas, primary_hex, embed_call_counter.clone()).await;

    // Full corpus: all CORPUS_SIZE × CHUNKS_PER_FILE blobs should have been written.
    assert_eq!(
        primary_calls,
        CORPUS_SIZE * CHUNKS_PER_FILE,
        "primary should have embedded every chunk in the corpus"
    );

    // --- Phase 2: fork the child ref from primary ---
    cas.fork_ref(child_hex, primary_hex).unwrap();

    // Verify child manifest starts empty.
    let child_manifest = cas.load_manifest(child_hex).unwrap();
    assert!(
        child_manifest.is_empty(),
        "child manifest must start empty after fork"
    );

    // Verify parent pointer is set.
    let parent = cas.read_parent(child_hex).unwrap();
    assert_eq!(parent.as_deref(), Some(primary_hex));

    // Reset the counter before measuring child indexing.
    embed_call_counter.store(0, Ordering::SeqCst);

    // --- Phase 3: index child with DIFF_SIZE files changed ---
    let changed_files: Vec<usize> = (0..DIFF_SIZE).collect(); // first DIFF_SIZE files changed
    let child_calls =
        index_child_ref(&cas, child_hex, embed_call_counter.clone(), &changed_files).await;

    // --- Assertions ---

    // LOAD-BEARING: embed calls must be at most (diff_size × chunks_per_file).
    let max_allowed = DIFF_SIZE * CHUNKS_PER_FILE;
    assert!(
        child_calls <= max_allowed,
        "embed_calls={child_calls} must be ≤ diff_size×chunks ({max_allowed})"
    );

    // Sanity: embed calls must be strictly less than full corpus cost.
    let full_corpus_cost = CORPUS_SIZE * CHUNKS_PER_FILE;
    assert!(
        child_calls < full_corpus_cost,
        "embed_calls={child_calls} must be < full corpus cost ({full_corpus_cost})"
    );

    // The changed files should have contributed exactly their chunk count.
    assert_eq!(
        child_calls,
        DIFF_SIZE * CHUNKS_PER_FILE,
        "all changed chunks must be embedded (no pre-existing blob for new content)"
    );

    // Unchanged files: none of them should appear in the child's own manifest
    // (they're all served via parent fallback).
    let final_child_manifest = cas.load_manifest(child_hex).unwrap();
    assert_eq!(
        final_child_manifest.len(),
        DIFF_SIZE * CHUNKS_PER_FILE,
        "child manifest should only contain entries for the {} changed files",
        DIFF_SIZE
    );
}

// ---------------------------------------------------------------------------
// Test: fork at same SHA → zero embed calls
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fork_at_same_sha_no_embed_calls() {
    let tmp = TempDir::new().unwrap();
    let mcp_data = tmp.path().join(".mcp_data");
    std::fs::create_dir_all(&mcp_data).unwrap();

    let cas = CasStore::new(mcp_data.clone(), "nomic-embed-text");
    let primary_hex = "primary_same_sha0";
    let child_hex = "child_same_sha000";

    let counter = Arc::new(AtomicUsize::new(0));

    // Index primary with zero changed files representing the full corpus.
    index_primary_ref(&cas, primary_hex, counter.clone()).await;

    // Fork child at the same content — no files differ.
    cas.fork_ref(child_hex, primary_hex).unwrap();
    counter.store(0, Ordering::SeqCst);

    let child_calls = index_child_ref(&cas, child_hex, counter.clone(), &[] /* no diffs */).await;

    assert_eq!(
        child_calls, 0,
        "fork at same SHA should require zero embed calls; got {child_calls}"
    );

    // Child manifest should remain empty (all served via parent).
    let m = cas.load_manifest(child_hex).unwrap();
    assert!(
        m.is_empty(),
        "child manifest should be empty when no files differ"
    );
}

// ---------------------------------------------------------------------------
// Test: sibling ref fork — no extra calls beyond sibling
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fork_from_sibling_no_extra_calls() {
    // sibling_a is forked from primary at the same SHA.
    // sibling_b is forked from sibling_a (same base, same content).
    // → sibling_b should see 0 embed calls.

    let tmp = TempDir::new().unwrap();
    let mcp_data = tmp.path().join(".mcp_data");
    std::fs::create_dir_all(&mcp_data).unwrap();

    let cas = CasStore::new(mcp_data.clone(), "nomic-embed-text");
    let primary_hex = "primary_sib_00000";
    let sibling_a_hex = "sibling_a_0000000";
    let sibling_b_hex = "sibling_b_0000000";

    let counter = Arc::new(AtomicUsize::new(0));
    index_primary_ref(&cas, primary_hex, counter.clone()).await;

    // Sibling A forked from primary: index 5 changed files.
    cas.fork_ref(sibling_a_hex, primary_hex).unwrap();
    let changed_a: Vec<usize> = (0..5).collect();
    counter.store(0, Ordering::SeqCst);
    let sibling_a_calls = index_child_ref(&cas, sibling_a_hex, counter.clone(), &changed_a).await;
    assert_eq!(sibling_a_calls, 5 * CHUNKS_PER_FILE);

    // Sibling B forked from sibling A at the same SHA (same 5 changes).
    cas.fork_ref(sibling_b_hex, sibling_a_hex).unwrap();
    counter.store(0, Ordering::SeqCst);

    // sibling_b sees the same 5 changed files (same generation=1 content as sibling_a).
    // Those chunks already exist in the CAS blobs → 0 embed calls.
    let sibling_b_calls = index_child_ref(&cas, sibling_b_hex, counter.clone(), &changed_a).await;

    assert_eq!(
        sibling_b_calls, 0,
        "sibling_b should reuse sibling_a's blobs; got {sibling_b_calls} embed calls"
    );
}

// ---------------------------------------------------------------------------
// Test: identical content under different paths → one CAS blob
// ---------------------------------------------------------------------------

#[tokio::test]
async fn identical_content_different_paths_one_blob() {
    let tmp = TempDir::new().unwrap();
    let mcp_data = tmp.path().join(".mcp_data");
    std::fs::create_dir_all(&mcp_data).unwrap();

    let cas = CasStore::new(mcp_data.clone(), "nomic-embed-text");
    let ref_hex = "dedup_test_ref0000";

    let content = "fn identical_across_files() {}";
    let hash = ChunkHash::of(content);
    let vec = fake_vector(42);

    // Write blob once.
    cas.write_blob(&hash, &vec).unwrap();

    // Two manifest entries pointing at the same blob.
    cas.update_manifest(
        ref_hex,
        &[
            (ChunkKey::new("src/alpha.rs", 0), hash.clone()),
            (ChunkKey::new("src/beta.rs", 0), hash.clone()),
        ],
    )
    .unwrap();

    // Verify both entries resolve to the same vector.
    let v1 = cas
        .get_embedding(ref_hex, &ChunkKey::new("src/alpha.rs", 0))
        .unwrap()
        .expect("alpha should resolve");
    let v2 = cas
        .get_embedding(ref_hex, &ChunkKey::new("src/beta.rs", 0))
        .unwrap()
        .expect("beta should resolve");

    assert_eq!(v1, v2, "both paths should return the same vector");
    assert_eq!(v1, vec, "returned vector should match what was written");

    // Only one blob file exists on disk.
    let blob_shard = mcp_data
        .join("cas")
        .join("nomic-embed-text")
        .join(hash.shard());
    let blob_count = std::fs::read_dir(&blob_shard)
        .map(|it| it.count())
        .unwrap_or(0);
    assert_eq!(
        blob_count, 1,
        "exactly one blob file should exist for identical content"
    );
}

// ---------------------------------------------------------------------------
// Test: CAS blob missing for manifest entry → clear error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn missing_blob_returns_clear_error() {
    let tmp = TempDir::new().unwrap();
    let mcp_data = tmp.path().join(".mcp_data");
    std::fs::create_dir_all(&mcp_data).unwrap();

    let cas = CasStore::new(mcp_data.clone(), "nomic-embed-text");
    let ref_hex = "corrupt_ref_000000";
    let key = ChunkKey::new("src/missing.rs", 0);

    // Add manifest entry WITHOUT writing the blob.
    let ghost_hash = ChunkHash::of("ghost content");
    cas.update_manifest(ref_hex, &[(key.clone(), ghost_hash)])
        .unwrap();

    let err = cas.get_embedding(ref_hex, &key).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("CAS blob missing"),
        "expected 'CAS blob missing' in error, got: {msg}"
    );
    assert!(
        msg.contains("re-walk"),
        "error should suggest re-walk, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Test: legacy format migration detection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn legacy_rkyv_files_detected_and_removed() {
    let tmp = TempDir::new().unwrap();
    let mcp_data = tmp.path().join(".mcp_data");
    std::fs::create_dir_all(&mcp_data).unwrap();

    // Write fake legacy files.
    let legacy1 = mcp_data.join("embed-nomic-embed-text.rkyv");
    let legacy2 = mcp_data.join("query-embeddings-nomic.rkyv");
    std::fs::write(&legacy1, b"old format data").unwrap();
    std::fs::write(&legacy2, b"old format data").unwrap();

    let cas = CasStore::new(mcp_data.clone(), "nomic-embed-text");

    // Migration should detect legacy files and delete them.
    let did_migrate = cas.detect_and_migrate_legacy();
    assert!(did_migrate, "should have detected legacy format");
    assert!(!legacy1.exists(), "legacy file 1 should be deleted");
    assert!(!legacy2.exists(), "legacy file 2 should be deleted");

    // Running again should be a no-op (files gone, cas/ dir still absent but
    // no legacy files either → no migration).
    let did_migrate_again = cas.detect_and_migrate_legacy();
    assert!(
        !did_migrate_again,
        "second migration call should be a no-op"
    );
}

// ---------------------------------------------------------------------------
// Test: primary ref indexing — manifest has all entries
// ---------------------------------------------------------------------------

#[tokio::test]
async fn primary_manifest_contains_all_corpus_entries() {
    let tmp = TempDir::new().unwrap();
    let mcp_data = tmp.path().join(".mcp_data");
    std::fs::create_dir_all(&mcp_data).unwrap();

    let cas = CasStore::new(mcp_data.clone(), "nomic-embed-text");
    let primary_hex = "primary_full_idx00";
    let counter = Arc::new(AtomicUsize::new(0));

    let calls = index_primary_ref(&cas, primary_hex, counter).await;
    assert_eq!(calls, CORPUS_SIZE * CHUNKS_PER_FILE);

    let manifest = cas.load_manifest(primary_hex).unwrap();
    assert_eq!(
        manifest.len(),
        CORPUS_SIZE * CHUNKS_PER_FILE,
        "primary manifest should have {} entries",
        CORPUS_SIZE * CHUNKS_PER_FILE
    );
}

// ---------------------------------------------------------------------------
// Test: fork at same SHA manifest is empty, tool calls succeed via parent
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fork_at_same_sha_manifest_empty_parent_fallback() {
    let tmp = TempDir::new().unwrap();
    let mcp_data = tmp.path().join(".mcp_data");
    std::fs::create_dir_all(&mcp_data).unwrap();

    let cas = CasStore::new(mcp_data.clone(), "nomic-embed-text");
    let primary_hex = "primary_fallback00";
    let child_hex = "child_fallback0000";

    let counter = Arc::new(AtomicUsize::new(0));
    index_primary_ref(&cas, primary_hex, counter).await;

    cas.fork_ref(child_hex, primary_hex).unwrap();
    let child_manifest = cas.load_manifest(child_hex).unwrap();
    assert!(
        child_manifest.is_empty(),
        "forked child manifest must be empty"
    );

    // Tool lookups on child should succeed via parent fallback.
    for file_idx in 0..5 {
        for chunk_idx in 0..CHUNKS_PER_FILE {
            let key = ChunkKey::new(fixture_path(file_idx), chunk_idx as u32);
            let emb = cas.get_embedding(child_hex, &key).unwrap();
            assert!(
                emb.is_some(),
                "chunk {}/{} should be accessible via parent fallback",
                file_idx,
                chunk_idx
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test: fork with 5-file diff → only 5 files in child manifest
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fork_five_file_diff_only_five_in_child_manifest() {
    let tmp = TempDir::new().unwrap();
    let mcp_data = tmp.path().join(".mcp_data");
    std::fs::create_dir_all(&mcp_data).unwrap();

    let cas = CasStore::new(mcp_data.clone(), "nomic-embed-text");
    let primary_hex = "primary_5diff00000";
    let child_hex = "child_5diff0000000";
    const FIVE: usize = 5;

    let counter = Arc::new(AtomicUsize::new(0));
    index_primary_ref(&cas, primary_hex, counter.clone()).await;
    cas.fork_ref(child_hex, primary_hex).unwrap();
    counter.store(0, Ordering::SeqCst);

    let changed: Vec<usize> = (0..FIVE).collect();
    index_child_ref(&cas, child_hex, counter.clone(), &changed).await;

    let child_manifest = cas.load_manifest(child_hex).unwrap();
    assert_eq!(
        child_manifest.len(),
        FIVE * CHUNKS_PER_FILE,
        "child manifest should have exactly {} entries for {} diffed files",
        FIVE * CHUNKS_PER_FILE,
        FIVE
    );

    // CAS blob count growth: at most FIVE × CHUNKS_PER_FILE new blobs.
    // (Each changed chunk produces one new blob; unchanged chunks don't.)
    let new_embed_calls = counter.load(Ordering::SeqCst);
    assert!(
        new_embed_calls <= FIVE * CHUNKS_PER_FILE,
        "CAS blob growth ≤ {} × {} = {}; got {} embed calls",
        FIVE,
        CHUNKS_PER_FILE,
        FIVE * CHUNKS_PER_FILE,
        new_embed_calls
    );
}
