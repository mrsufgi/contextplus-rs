//! Benchmark: spectral clustering performance at various scales.
//!
//! Measures `spectral_cluster` with synthetic embedding vectors at 50, 200, 500 items.
//! The affinity matrix is n*n, so 500 items = 250K entries.
//! Uses vectors with clear cluster structure to exercise the full pipeline:
//! affinity matrix -> Laplacian -> eigendecomposition -> k-means.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use contextplus_rs::core::clustering::spectral_cluster;

/// Generate synthetic vectors with `k` clear clusters of roughly equal size.
/// Each cluster is centered around a different axis in `dims`-dimensional space.
fn generate_clustered_vectors(n: usize, dims: usize, k: usize) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        let cluster = i % k;
        let mut vec = vec![0.0_f32; dims];
        // Base direction for this cluster
        let base_dim = cluster % dims;
        vec[base_dim] = 1.0;
        // Add small noise to make vectors non-identical within a cluster
        for (d, v) in vec.iter_mut().enumerate() {
            *v += ((i * 7 + d * 13 + 42) % 100) as f32 / 1000.0;
        }
        // Normalize
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vectors.push(vec);
    }
    vectors
}

fn bench_spectral_cluster(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_clustering");
    group.sample_size(10);

    let dims = 64;
    let max_clusters = 10;

    for &n in &[50, 200, 500] {
        let k = 3.min(n / 4).max(2); // 2-3 real clusters
        let vectors = generate_clustered_vectors(n, dims, k);

        group.bench_with_input(
            BenchmarkId::new("cluster", format!("{}_items", n)),
            &n,
            |bench, _| {
                bench.iter(|| {
                    let result = spectral_cluster(&vectors, max_clusters);
                    assert!(
                        result.len() >= 2,
                        "expected at least 2 clusters for {} items",
                        n
                    );
                    result
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_spectral_cluster);
criterion_main!(benches);
