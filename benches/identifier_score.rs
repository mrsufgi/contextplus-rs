//! Benchmark: identifier scoring with hybrid semantic + keyword ranking.
//!
//! Measures `score_identifiers` with synthetic IdentifierDoc vectors
//! at 1000, 5000, 10000 identifiers. This is the hot path for
//! semantic identifier search (excludes embedding HTTP call).

use std::collections::HashSet;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use contextplus_rs::tools::semantic_identifiers::{IdentifierDoc, score_identifiers};

/// Generate synthetic identifier docs.
fn generate_docs(n: usize) -> Vec<IdentifierDoc> {
    let kinds = ["function", "class", "interface", "type", "method", "const"];
    (0..n)
        .map(|i| {
            let kind = kinds[i % kinds.len()];
            let name = format!("symbol_{}", i);
            IdentifierDoc {
                id: format!("src/mod_{}/file_{}.ts:{}:{}", i / 50, i, name, i * 10 + 1),
                path: format!("src/mod_{}/file_{}.ts", i / 50, i),
                header: format!("module {}", i / 50),
                name: name.clone(),
                kind: kind.to_string(),
                line: i * 10 + 1,
                end_line: i * 10 + 20,
                signature: format!("{}(arg: string): Result", name),
                parent_name: if i % 3 == 0 {
                    Some(format!("Service_{}", i / 10))
                } else {
                    None
                },
                text: format!(
                    "{} {} {}(arg: string): Result src/mod_{}/file_{}.ts module {}",
                    kind,
                    name,
                    name,
                    i / 50,
                    i,
                    i / 50
                ),
            }
        })
        .collect()
}

/// Generate deterministic pseudo-random vectors for identifiers.
fn generate_vector_buffer(n: usize, dims: usize) -> Vec<f32> {
    let mut buf = Vec::with_capacity(n * dims);
    for i in 0..n {
        for d in 0..dims {
            buf.push(((i * 7 + d * 13 + 42) % 1000) as f32 / 1000.0);
        }
    }
    buf
}

/// Generate a query vector.
fn generate_query(dims: usize) -> Vec<f32> {
    (0..dims)
        .map(|d| ((d * 31 + 17) % 1000) as f32 / 1000.0)
        .collect()
}

fn bench_score_identifiers(c: &mut Criterion) {
    let mut group = c.benchmark_group("identifier_score");
    group.sample_size(20);

    let dims = 384; // Common embedding dimension for small models

    for &count in &[1000, 5000, 10000] {
        let docs = generate_docs(count);
        let vector_buffer = generate_vector_buffer(count, dims);
        let query_vec = generate_query(dims);
        let query_terms: HashSet<String> = ["symbol", "service", "result"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Score all identifiers, top 5
        group.bench_with_input(
            BenchmarkId::new("top_5", format!("{}_ids", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    black_box(score_identifiers(
                        black_box(&docs),
                        black_box(&query_vec),
                        black_box(&query_terms),
                        black_box(&vector_buffer),
                        dims,
                        &None,
                        0.78,
                        0.22,
                        5,
                    ))
                });
            },
        );

        // Score all identifiers, top 20
        group.bench_with_input(
            BenchmarkId::new("top_20", format!("{}_ids", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    black_box(score_identifiers(
                        black_box(&docs),
                        black_box(&query_vec),
                        black_box(&query_terms),
                        black_box(&vector_buffer),
                        dims,
                        &None,
                        0.78,
                        0.22,
                        20,
                    ))
                });
            },
        );

        // With kind filter
        let kind_filter: Option<HashSet<String>> = Some(
            ["function", "method"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        group.bench_with_input(
            BenchmarkId::new("filtered_kinds", format!("{}_ids", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    black_box(score_identifiers(
                        black_box(&docs),
                        black_box(&query_vec),
                        black_box(&query_terms),
                        black_box(&vector_buffer),
                        dims,
                        black_box(&kind_filter),
                        0.78,
                        0.22,
                        5,
                    ))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_score_identifiers);
criterion_main!(benches);
