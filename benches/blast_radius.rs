//! Benchmark: blast radius symbol search across file lines.
//!
//! Measures `find_symbol_usages` with synthetic file data at 500, 2000 files,
//! each with 50-200 lines. The target symbol appears in ~5% of files.

use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use contextplus_rs::tools::blast_radius::find_symbol_usages;

/// Generate synthetic file lines. The symbol `targetSymbol` appears
/// in approximately `hit_pct`% of files, on 1-2 lines per file.
fn generate_file_lines(
    file_count: usize,
    lines_per_file: usize,
    symbol: &str,
    hit_pct: usize,
) -> HashMap<String, Vec<String>> {
    let mut map = HashMap::with_capacity(file_count);

    for i in 0..file_count {
        let mut lines = Vec::with_capacity(lines_per_file);
        let has_symbol = (i % 100) < hit_pct;

        for j in 0..lines_per_file {
            if has_symbol && j == lines_per_file / 3 {
                // Usage line (import-like)
                lines.push(format!("import {{ {} }} from './module_{}';\n", symbol, i));
            } else if has_symbol && j == lines_per_file * 2 / 3 {
                // Usage line (call-like)
                lines.push(format!("  const result = {}(params);\n", symbol));
            } else {
                // Filler line
                lines.push(format!(
                    "  const var_{} = someOtherFunction({});\n",
                    j,
                    j * 2 + 1
                ));
            }
        }

        map.insert(
            format!("packages/domain/mod_{}/handler_{}.ts", i / 50, i),
            lines,
        );
    }
    map
}

fn bench_find_symbol_usages(c: &mut Criterion) {
    let mut group = c.benchmark_group("blast_radius");
    group.sample_size(20);

    let symbol = "getUserById";

    for &(file_count, lines) in &[(500, 100), (2000, 100), (2000, 200)] {
        let file_lines = generate_file_lines(file_count, lines, symbol, 5);

        // Without file_context (no definition exclusion)
        group.bench_with_input(
            BenchmarkId::new("no_context", format!("{}_files_{}l", file_count, lines)),
            &file_count,
            |bench, _| {
                bench.iter(|| {
                    black_box(find_symbol_usages(
                        black_box(symbol),
                        None,
                        black_box(&file_lines),
                    ))
                });
            },
        );

        // With file_context (definition exclusion active)
        let first_file = file_lines.keys().next().unwrap().clone();
        group.bench_with_input(
            BenchmarkId::new("with_context", format!("{}_files_{}l", file_count, lines)),
            &file_count,
            |bench, _| {
                bench.iter(|| {
                    black_box(find_symbol_usages(
                        black_box(symbol),
                        Some(black_box(&first_file)),
                        black_box(&file_lines),
                    ))
                });
            },
        );
    }

    // Benchmark with special regex characters in symbol name
    let special_symbol = "$transaction";
    let special_lines = generate_file_lines(500, 100, special_symbol, 5);
    group.bench_with_input(
        BenchmarkId::new("special_chars", "500_files_100l"),
        &500,
        |bench, _| {
            bench.iter(|| {
                black_box(find_symbol_usages(
                    black_box(special_symbol),
                    None,
                    black_box(&special_lines),
                ))
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_find_symbol_usages);
criterion_main!(benches);
