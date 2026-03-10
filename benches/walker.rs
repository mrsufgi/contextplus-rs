//! Benchmark: gitignore-aware directory walking at small and medium scale.
//!
//! Creates a real tempdir with synthetic source files, then measures
//! `walk_directory` end-to-end including gitignore parsing and path filtering.
//! Benchmark groups: walker/walk/small_100 and walker/walk/medium_500.

use std::collections::HashSet;
use std::fs;


use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tempfile::TempDir;

use contextplus_rs::core::walker::{WalkOptions, walk_directory};

/// Create a tempdir with `n` synthetic source files spread across a few directories.
fn create_file_tree(n: usize) -> TempDir {
    let dir = TempDir::new().expect("failed to create tempdir");
    let root = dir.path();

    let subdirs = ["src", "src/core", "src/tools", "tests", "docs"];
    for subdir in &subdirs {
        fs::create_dir_all(root.join(subdir)).expect("mkdir");
    }

    // Write a minimal .gitignore so the walker exercises the gitignore code path
    fs::write(root.join(".gitignore"), "target/\n*.log\n").expect("write .gitignore");

    let extensions = ["ts", "rs", "go", "py"];
    for i in 0..n {
        let subdir = subdirs[i % subdirs.len()];
        let ext = extensions[i % extensions.len()];
        let file_path = root.join(subdir).join(format!("file_{}.{}", i, ext));
        fs::write(&file_path, format!("// file {}\nfn main() {{}}\n", i)).expect("write file");
    }

    dir
}

/// Shared ignore dirs set (empty — bench the common case).
fn empty_ignore() -> HashSet<String> {
    HashSet::new()
}

fn bench_walker(c: &mut Criterion) {
    let mut group = c.benchmark_group("walker/walk");
    group.sample_size(20);

    // Create trees once outside the benchmark loop — we're measuring walking, not fs setup
    let small_dir = create_file_tree(100);
    let medium_dir = create_file_tree(500);

    let ignore_dirs = empty_ignore();

    group.bench_with_input(BenchmarkId::new("small", "100"), &100usize, |bench, _| {
        let opts = WalkOptions {
            root_dir: small_dir.path(),
            target_path: None,
            depth_limit: None,
            ignore_dirs: &ignore_dirs,
        };
        bench.iter(|| black_box(walk_directory(black_box(&opts))));
    });

    group.bench_with_input(BenchmarkId::new("medium", "500"), &500usize, |bench, _| {
        let opts = WalkOptions {
            root_dir: medium_dir.path(),
            target_path: None,
            depth_limit: None,
            ignore_dirs: &ignore_dirs,
        };
        bench.iter(|| black_box(walk_directory(black_box(&opts))));
    });

    // With depth_limit — exercises early-exit path
    group.bench_with_input(
        BenchmarkId::new("medium_depth2", "500"),
        &500usize,
        |bench, _| {
            let opts = WalkOptions {
                root_dir: medium_dir.path(),
                target_path: None,
                depth_limit: Some(2),
                ignore_dirs: &ignore_dirs,
            };
            bench.iter(|| black_box(walk_directory(black_box(&opts))));
        },
    );

    group.finish();
}

criterion_group!(benches, bench_walker);
criterion_main!(benches);
