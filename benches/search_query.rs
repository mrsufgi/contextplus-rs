//! Benchmark: End-to-end warm search query simulation.
//!
//! Simulates the full search pipeline: build VectorStore from cache entries,
//! run find_nearest, then format results — without Ollama (uses pre-computed vectors).
//! This measures everything except the embedding HTTP call.
//! TS baseline: ~1.5s warm. Rust target: <100ms.

use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tempfile::TempDir;

use contextplus_rs::cache::rkyv_store::{CacheData, load_cache_mmap, save_cache};
use contextplus_rs::core::embeddings::{CacheEntry, VectorStore};

/// Generate a realistic embedding cache with `n` file entries.
fn generate_cache(n: usize, dims: usize) -> HashMap<String, CacheEntry> {
    let mut cache = HashMap::with_capacity(n);
    for i in 0..n {
        let key = format!(
            "packages/domains/feature_{}/service/handler_{}.ts",
            i / 50,
            i
        );
        let hash = format!("hash_{}", i);
        let vector: Vec<f32> = (0..dims)
            .map(|d| ((i * 7 + d * 13 + 42) % 1000) as f32 / 1000.0)
            .collect();
        cache.insert(key, CacheEntry { hash, vector });
    }
    cache
}

fn generate_query(dims: usize) -> Vec<f32> {
    (0..dims)
        .map(|d| ((d * 31 + 17) % 1000) as f32 / 1000.0)
        .collect()
}

/// Simulates the warm search path: mmap load → VectorStore → find_nearest → format.
fn bench_warm_search_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("warm_search_pipeline");
    group.sample_size(20);

    for &count in &[1000, 5000, 30000] {
        let dims = 1024;
        let cache = generate_cache(count, dims);
        let store = VectorStore::from_cache(&cache).unwrap();
        let data = CacheData::from_store(&store);
        let dir = TempDir::new().unwrap();
        save_cache(dir.path(), "search-bench", &data).unwrap();

        let query = generate_query(dims);

        // Full pipeline: load from disk (mmap) → build store → search → format
        group.bench_with_input(
            BenchmarkId::new("full_pipeline", format!("{}_files", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    // 1. Load cache from disk via mmap
                    let loaded = load_cache_mmap(dir.path(), "search-bench")
                        .unwrap()
                        .unwrap();

                    // 2. Build VectorStore
                    let store = loaded.to_store();

                    // 3. Find nearest
                    let results = store.find_nearest(black_box(&query), 5);

                    // 4. Format results (simulates response building)
                    let mut output = String::with_capacity(2048);
                    for (key, score) in &results {
                        output.push_str(&format!("{} (score: {:.4})\n", key, score));
                    }

                    black_box(output)
                });
            },
        );

        // Warm path: store already in memory → search → format
        let warm_store = VectorStore::from_cache(&cache).unwrap();
        group.bench_with_input(
            BenchmarkId::new("warm_search_only", format!("{}_files", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    let results = warm_store.find_nearest(black_box(&query), 5);
                    let mut output = String::with_capacity(2048);
                    for (key, score) in &results {
                        output.push_str(&format!("{} (score: {:.4})\n", key, score));
                    }
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark hash-check staleness detection (the no-op refresh path).
fn bench_hash_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_check_staleness");

    for &count in &[1000, 5000, 30000] {
        let dims = 1024;
        let cache = generate_cache(count, dims);
        let store = VectorStore::from_cache(&cache).unwrap();

        // Simulate checking all files for staleness (the no-op refresh path)
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

        // Simulate 10% stale files
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
