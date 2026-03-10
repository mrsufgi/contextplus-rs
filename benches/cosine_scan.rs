//! Benchmark: SIMD cosine similarity scan performance.
//!
//! Measures simsimd (AVX-512/AVX2 auto-dispatch) vs naive scalar loop
//! for scanning 30K vectors × 1024 dimensions — the core of every search query.
//! TS baseline: ~50ms (JS loop). Rust target: <20ms (SIMD).

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use contextplus_rs::core::embeddings::{
    VectorStore, cosine_similarity_naive, cosine_similarity_simsimd,
};

/// Generate deterministic pseudo-random vectors.
fn generate_vectors(n: usize, dims: usize) -> Vec<f32> {
    let mut vectors = Vec::with_capacity(n * dims);
    for i in 0..n {
        for d in 0..dims {
            vectors.push(((i * 7 + d * 13 + 42) % 1000) as f32 / 1000.0);
        }
    }
    vectors
}

fn generate_query(dims: usize) -> Vec<f32> {
    (0..dims)
        .map(|d| ((d * 31 + 17) % 1000) as f32 / 1000.0)
        .collect()
}

fn bench_cosine_single_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_single_pair");

    for &dims in &[128, 512, 1024] {
        let a: Vec<f32> = (0..dims).map(|d| (d as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..dims).map(|d| (d as f32 * 0.002).cos()).collect();

        group.bench_with_input(BenchmarkId::new("simsimd", dims), &dims, |bench, _| {
            bench.iter(|| cosine_similarity_simsimd(&a, &b));
        });

        group.bench_with_input(BenchmarkId::new("naive", dims), &dims, |bench, _| {
            bench.iter(|| cosine_similarity_naive(&a, &b));
        });
    }

    group.finish();
}

fn bench_cosine_full_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_full_scan");
    // Use realistic measurement time for the large scans
    group.sample_size(20);

    for &(count, dims) in &[(1000, 1024), (5000, 1024), (30000, 1024)] {
        let vectors = generate_vectors(count, dims);
        let query = generate_query(dims);

        // Raw loop with simsimd
        group.bench_with_input(
            BenchmarkId::new("simsimd_scan", format!("{}x{}", count, dims)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    let mut best = f32::NEG_INFINITY;
                    for i in 0..count {
                        let offset = i * dims;
                        let stored = &vectors[offset..offset + dims];
                        let sim = cosine_similarity_simsimd(&query, stored);
                        if sim > best {
                            best = sim;
                        }
                    }
                    best
                });
            },
        );

        // Raw loop with naive
        group.bench_with_input(
            BenchmarkId::new("naive_scan", format!("{}x{}", count, dims)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    let mut best = f32::NEG_INFINITY;
                    for i in 0..count {
                        let offset = i * dims;
                        let stored = &vectors[offset..offset + dims];
                        let sim = cosine_similarity_naive(&query, stored);
                        if sim > best {
                            best = sim;
                        }
                    }
                    best
                });
            },
        );
    }

    group.finish();
}

fn bench_vectorstore_find_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("vectorstore_find_nearest");
    group.sample_size(20);

    for &count in &[1000, 5000, 30000] {
        let dims = 1024usize;
        let keys: Vec<String> = (0..count).map(|i| format!("src/file_{}.ts", i)).collect();
        let hashes: Vec<String> = (0..count).map(|i| format!("hash_{}", i)).collect();
        let vectors = generate_vectors(count, dims);
        let store = VectorStore::new(dims as u32, keys, hashes, vectors);
        let query = generate_query(dims);

        group.bench_with_input(
            BenchmarkId::new("top_5", format!("{}x{}", count, dims)),
            &count,
            |bench, _| {
                bench.iter(|| black_box(store.find_nearest(black_box(&query), 5)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("top_20", format!("{}x{}", count, dims)),
            &count,
            |bench, _| {
                bench.iter(|| black_box(store.find_nearest(black_box(&query), 20)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_single_pair,
    bench_cosine_full_scan,
    bench_vectorstore_find_nearest,
);
criterion_main!(benches);
