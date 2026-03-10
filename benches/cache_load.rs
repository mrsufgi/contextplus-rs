//! Benchmark: rkyv cache save/load/mmap performance.
//!
//! Measures the critical path: how fast can we persist and reload the VectorStore
//! from disk? This is the #1 bottleneck in the TS version (115ms VectorStore, 1109ms raw).
//! Rust target: <10ms for 30K vectors × 1024 dims.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tempfile::TempDir;

use contextplus_rs::cache::rkyv_store::{CacheData, load_cache, load_cache_mmap, save_cache};
use contextplus_rs::core::embeddings::VectorStore;

/// Generate a realistic-sized VectorStore with `n` vectors of `dims` dimensions.
fn generate_store(n: usize, dims: usize) -> VectorStore {
    let keys: Vec<String> = (0..n).map(|i| format!("src/file_{}.ts", i)).collect();
    let hashes: Vec<String> = (0..n).map(|i| format!("hash_{}", i)).collect();
    let mut vectors = Vec::with_capacity(n * dims);
    for i in 0..n {
        for d in 0..dims {
            // Deterministic pseudo-random values
            vectors.push(((i * 7 + d * 13) % 1000) as f32 / 1000.0);
        }
    }
    VectorStore::new(dims as u32, keys, hashes, vectors)
}

fn bench_cache_save(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_save");

    for &count in &[1000, 5000, 30000] {
        let store = generate_store(count, 1024);
        let data = CacheData::from_store(&store);
        let dir = TempDir::new().unwrap();

        group.bench_with_input(
            BenchmarkId::new("rkyv_save", format!("{}x1024", count)),
            &count,
            |b, _| {
                b.iter(|| {
                    save_cache(dir.path(), "bench", &data).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_cache_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_load");

    for &count in &[1000, 5000, 30000] {
        let store = generate_store(count, 1024);
        let data = CacheData::from_store(&store);
        let dir = TempDir::new().unwrap();
        save_cache(dir.path(), "bench", &data).unwrap();

        group.bench_with_input(
            BenchmarkId::new("rkyv_read", format!("{}x1024", count)),
            &count,
            |b, _| {
                b.iter(|| black_box(load_cache(dir.path(), "bench").unwrap().unwrap()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rkyv_mmap", format!("{}x1024", count)),
            &count,
            |b, _| {
                b.iter(|| black_box(load_cache_mmap(dir.path(), "bench").unwrap().unwrap()));
            },
        );

        // Measure just the VectorStore construction from loaded data
        let loaded_data = load_cache(dir.path(), "bench").unwrap().unwrap();
        group.bench_with_input(
            BenchmarkId::new("to_store", format!("{}x1024", count)),
            &count,
            |b, _| {
                b.iter(|| black_box(loaded_data.to_store()));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cache_save, bench_cache_load);
criterion_main!(benches);
