//! Benchmark: warm search query using the production scoring path.
//!
//! Uses `SearchIndex::search` (hybrid semantic + keyword, brute-force for
//! corpora under `HNSW_MIN_VECTORS = 50 000`) — the exact path production
//! takes for repositories of typical size.  The previous bench called
//! `VectorStore::find_nearest` directly, which switches to HNSW above
//! `HNSW_THRESHOLD = 2 000`; that path is never taken by production below
//! 50 K embedded files.
//!
//! Suites:
//!   `warm_search_pipeline/warm_search_only/<N>_files`   — index in memory
//!   `warm_search_pipeline/full_pipeline/<N>_files`      — mmap load + index build + search
//!   `hash_check_staleness/*`                            — no-op refresh (VectorStore layer)

use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tempfile::TempDir;

use contextplus_rs::cache::rkyv_store::{CacheData, load_cache_mmap, save_cache};
use contextplus_rs::core::embeddings::{CacheEntry, VectorStore};
use contextplus_rs::tools::semantic_search::{ResolvedSearchOptions, SearchDocument, SearchIndex};

const DIMS: usize = 1024;
const QUERY_STR: &str = "handler service feature";

/// Synthetic query vector, deterministic.
fn generate_query() -> Vec<f32> {
    (0..DIMS)
        .map(|d| ((d * 31 + 17) % 1000) as f32 / 1000.0)
        .collect()
}

/// `n` `SearchDocument`s with deterministic embedding vectors.
/// Used for `SearchIndex::index_with_vectors`.
fn generate_docs(n: usize) -> (Vec<SearchDocument>, Vec<Option<Vec<f32>>>) {
    let mut docs = Vec::with_capacity(n);
    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        let path = format!(
            "packages/domains/feature_{}/service/handler_{}.ts",
            i / 50,
            i
        );
        let doc = SearchDocument::new(path, String::new(), vec![], vec![], String::new());
        let v: Vec<f32> = (0..DIMS)
            .map(|d| ((i * 7 + d * 13 + 42) % 1000) as f32 / 1000.0)
            .collect();
        docs.push(doc);
        vectors.push(Some(v));
    }
    (docs, vectors)
}

/// Build a `CacheData` for `n` documents — used to pre-populate the on-disk
/// cache for the full-pipeline bench.
fn generate_cache_data(n: usize) -> CacheData {
    let keys: Vec<String> = (0..n)
        .map(|i| {
            format!(
                "packages/domains/feature_{}/service/handler_{}.ts",
                i / 50,
                i
            )
        })
        .collect();
    let hashes: Vec<String> = (0..n).map(|i| format!("hash_{}", i)).collect();
    let vectors: Vec<f32> = (0..n)
        .flat_map(|i| (0..DIMS).map(move |d| ((i * 7 + d * 13 + 42) % 1000) as f32 / 1000.0))
        .collect();
    CacheData {
        dims: DIMS as u32,
        keys,
        hashes,
        vectors,
    }
}

/// `VectorStore` keyed cache used only by the hash-check staleness bench.
fn generate_cache_entries(n: usize) -> HashMap<String, CacheEntry> {
    let mut cache = HashMap::with_capacity(n);
    for i in 0..n {
        let key = format!(
            "packages/domains/feature_{}/service/handler_{}.ts",
            i / 50,
            i
        );
        let hash = format!("hash_{}", i);
        let vector: Vec<f32> = (0..DIMS)
            .map(|d| ((i * 7 + d * 13 + 42) % 1000) as f32 / 1000.0)
            .collect();
        cache.insert(key, CacheEntry { hash, vector });
    }
    cache
}

/// Build a `SearchIndex` from `n` synthetic documents.
fn build_index(n: usize) -> SearchIndex {
    let (docs, vectors) = generate_docs(n);
    let mut index = SearchIndex::new();
    index.index_with_vectors(docs, vectors);
    index
}

/// Warm search: `SearchIndex` already in memory → `search()`.
///
/// This is the production path for corpora under 50 K embedded files.
/// Scoring is hybrid (semantic + keyword) over every document — no HNSW graph
/// is consulted here.  The old bench called `VectorStore::find_nearest`, which
/// activated HNSW for N > 2 000; that over-reported speed for 5 K and 30 K.
fn bench_warm_search_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("warm_search_pipeline");
    group.sample_size(20);

    let query = generate_query();
    let opts = ResolvedSearchOptions::default();

    for &count in &[1_000, 5_000, 30_000] {
        // Build index once; timed loop does only the search.
        let index = build_index(count);

        group.bench_with_input(
            BenchmarkId::new("warm_search_only", format!("{}_files", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    black_box(index.search(black_box(QUERY_STR), black_box(&query), &opts))
                });
            },
        );

        // Full pipeline: mmap load → SearchIndex build → search.
        let data = generate_cache_data(count);
        let dir = TempDir::new().unwrap();
        save_cache(dir.path(), "search-bench", &data).unwrap();

        group.bench_with_input(
            BenchmarkId::new("full_pipeline", format!("{}_files", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    let loaded = load_cache_mmap(dir.path(), "search-bench")
                        .unwrap()
                        .unwrap();
                    let n = loaded.keys.len();
                    let d = loaded.dims as usize;
                    let docs: Vec<SearchDocument> = loaded
                        .keys
                        .iter()
                        .map(|k| {
                            SearchDocument::new(
                                k.clone(),
                                String::new(),
                                vec![],
                                vec![],
                                String::new(),
                            )
                        })
                        .collect();
                    let vecs: Vec<Option<Vec<f32>>> = (0..n)
                        .map(|i| Some(loaded.vectors[i * d..(i + 1) * d].to_vec()))
                        .collect();
                    let mut idx = SearchIndex::new();
                    idx.index_with_vectors(docs, vecs);
                    black_box(idx.search(black_box(QUERY_STR), black_box(&query), &opts))
                });
            },
        );
    }

    group.finish();
}

/// Hash-check staleness detection (the no-op refresh path).
///
/// Tests `VectorStore::get_hash` — the staleness layer, not the search path.
fn bench_hash_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_check_staleness");

    for &count in &[1_000, 5_000, 30_000] {
        let cache = generate_cache_entries(count);
        let store = VectorStore::from_cache(&cache).unwrap();

        let file_hashes: Vec<(String, String)> = cache
            .iter()
            .map(|(k, v)| (k.clone(), v.hash.clone()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("all_fresh", format!("{}_files", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    let mut stale_count = 0u32;
                    for (key, hash) in &file_hashes {
                        if store.get_hash(key) != Some(hash.as_str()) {
                            stale_count += 1;
                        }
                    }
                    black_box(stale_count)
                });
            },
        );

        let mut modified_hashes = file_hashes.clone();
        for item in modified_hashes.iter_mut().step_by(10) {
            item.1 = "modified_hash".to_string();
        }

        group.bench_with_input(
            BenchmarkId::new("10pct_stale", format!("{}_files", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    let mut stale_count = 0u32;
                    for (key, hash) in &modified_hashes {
                        if store.get_hash(key) != Some(hash.as_str()) {
                            stale_count += 1;
                        }
                    }
                    black_box(stale_count)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_warm_search_pipeline, bench_hash_check);
criterion_main!(benches);
