//! Benchmark: context tree building and rendering at scale.
//!
//! Measures `build_context_tree` with synthetic file entries at 500, 2000, 5000 files.
//! Tests both with and without symbols to measure the impact of symbol formatting.

use std::collections::BTreeMap;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use contextplus_rs::tools::context_tree::{
    FileAnalysis, FileEntry, TreeSymbol, build_context_tree,
};

/// Generate realistic nested file entries at various depths.
fn generate_entries(count: usize) -> Vec<FileEntry> {
    let dirs = [
        "src",
        "src/core",
        "src/core/utils",
        "src/tools",
        "src/tools/handlers",
        "src/delivery",
        "src/delivery/http",
        "src/delivery/nats",
        "tests",
        "tests/unit",
        "tests/integration",
        "packages",
        "packages/domain",
        "packages/domain/auth",
        "packages/domain/billing",
        "packages/domain/scheduling",
    ];

    let extensions = ["ts", "rs", "go", "py", "js", "tsx"];

    let mut entries = Vec::with_capacity(count + dirs.len());

    // Add directory entries
    for dir in &dirs {
        let depth = dir.matches('/').count() + 1;
        entries.push(FileEntry {
            relative_path: dir.to_string(),
            is_directory: true,
            depth,
        });
    }

    // Add file entries distributed across directories
    for i in 0..count {
        let dir = dirs[i % dirs.len()];
        let ext = extensions[i % extensions.len()];
        let depth = dir.matches('/').count() + 2;
        entries.push(FileEntry {
            relative_path: format!("{}/file_{}.{}", dir, i, ext),
            is_directory: false,
            depth,
        });
    }

    entries
}

/// Generate analyses with headers and symbols for a subset of file entries.
fn generate_analyses(entries: &[FileEntry], with_symbols: bool) -> BTreeMap<String, FileAnalysis> {
    let mut analyses = BTreeMap::new();
    for entry in entries.iter().filter(|e| !e.is_directory) {
        let symbols = if with_symbols {
            vec![
                TreeSymbol {
                    name: "handleRequest".to_string(),
                    kind: "function".to_string(),
                    line: 10,
                    end_line: 45,
                    signature: "handleRequest(req: Request): Response".to_string(),
                    children: vec![TreeSymbol {
                        name: "validate".to_string(),
                        kind: "function".to_string(),
                        line: 15,
                        end_line: 25,
                        signature: "validate(input: unknown): boolean".to_string(),
                        children: vec![],
                    }],
                },
                TreeSymbol {
                    name: "Config".to_string(),
                    kind: "interface".to_string(),
                    line: 1,
                    end_line: 8,
                    signature: "interface Config".to_string(),
                    children: vec![],
                },
            ]
        } else {
            vec![]
        };
        analyses.insert(
            entry.relative_path.clone(),
            FileAnalysis {
                header: Some(format!("module for {}", entry.relative_path)),
                symbols,
            },
        );
    }
    analyses
}

fn bench_context_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("context_tree");
    group.sample_size(20);

    for &count in &[500, 2000, 5000] {
        let entries = generate_entries(count);

        // Without symbols
        let analyses_no_sym = generate_analyses(&entries, false);
        group.bench_with_input(
            BenchmarkId::new("no_symbols", format!("{}_files", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    let result = build_context_tree(&entries, &analyses_no_sym, false, None, None);
                    assert!(!result.is_empty());
                    result
                });
            },
        );

        // With symbols
        let analyses_sym = generate_analyses(&entries, true);
        group.bench_with_input(
            BenchmarkId::new("with_symbols", format!("{}_files", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    let result = build_context_tree(&entries, &analyses_sym, true, None, None);
                    assert!(!result.is_empty());
                    result
                });
            },
        );

        // With depth limit
        group.bench_with_input(
            BenchmarkId::new("depth_limit_2", format!("{}_files", count)),
            &count,
            |bench, _| {
                bench.iter(|| {
                    let result = build_context_tree(&entries, &analyses_sym, true, None, Some(2));
                    assert!(!result.is_empty());
                    result
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_context_tree);
criterion_main!(benches);
