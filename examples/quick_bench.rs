use contextplus_rs::core::embeddings::{
    CacheEntry, VectorStore, cosine_similarity_naive, cosine_similarity_simsimd,
};
use contextplus_rs::tools::semantic_search::{ResolvedSearchOptions, SearchDocument, SearchIndex};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

fn generate_vectors(n: usize, dims: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n * dims);
    for i in 0..n {
        for d in 0..dims {
            v.push(((i * 7 + d * 13 + 42) % 1000) as f32 / 1000.0);
        }
    }
    v
}

fn generate_query(dims: usize) -> Vec<f32> {
    (0..dims)
        .map(|d| ((d * 31 + 17) % 1000) as f32 / 1000.0)
        .collect()
}

fn main() {
    let dims = 1024usize;
    println!("=== contextplus-rs Benchmark ===\n");

    // 1. Cosine similarity: SIMD vs naive
    let a: Vec<f32> = (0..dims).map(|d| (d as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..dims).map(|d| (d as f32 * 0.002).cos()).collect();

    let iters = 100_000;
    let start = Instant::now();
    for _ in 0..iters {
        black_box(cosine_similarity_naive(black_box(&a), black_box(&b)));
    }
    let naive_us = start.elapsed().as_micros() as f64 / iters as f64;

    let start = Instant::now();
    for _ in 0..iters {
        black_box(cosine_similarity_simsimd(black_box(&a), black_box(&b)));
    }
    let simd_us = start.elapsed().as_micros() as f64 / iters as f64;

    println!("Cosine similarity (1024 dims, 100K iters):");
    println!("  naive:  {:.3} µs/op", naive_us);
    println!("  simsimd: {:.3} µs/op", simd_us);
    println!("  speedup: {:.1}x\n", naive_us / simd_us);

    // 2. VectorStore find_nearest at various sizes (SIMD+rayon brute-force)
    println!("--- find_nearest (SIMD+rayon, exact) ---");
    for &count in &[1000, 5000, 30000] {
        let keys: Vec<String> = (0..count).map(|i| format!("src/file_{}.ts", i)).collect();
        let hashes: Vec<String> = (0..count).map(|i| format!("hash_{}", i)).collect();
        let vectors = generate_vectors(count, dims);
        let store = VectorStore::new(dims as u32, keys, hashes, vectors);
        let query = generate_query(dims);

        let runs = if count >= 30000 { 10 } else { 50 };
        // Warm up
        let _ = store.find_nearest(&query, 5);
        let start = Instant::now();
        for _ in 0..runs {
            black_box(store.find_nearest(black_box(&query), 5));
        }
        let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        println!("  {} vectors: {:.2} ms/query", count, avg_ms);
    }

    // 3. Cache load (rkyv save + load + mmap)
    println!();
    let count = 5000;
    let mut cache: HashMap<String, CacheEntry> = HashMap::new();
    for i in 0..count {
        cache.insert(
            format!("src/file_{}.ts", i),
            CacheEntry {
                hash: format!("h{}", i),
                vector: generate_query(dims),
            },
        );
    }
    let store = VectorStore::from_cache(&cache).unwrap();
    let data = contextplus_rs::cache::rkyv_store::CacheData::from_store(&store);
    let dir = tempfile::TempDir::new().unwrap();
    contextplus_rs::cache::rkyv_store::save_cache(dir.path(), "bench", &data).unwrap();

    let runs = 100;
    let start = Instant::now();
    for _ in 0..runs {
        let _ = contextplus_rs::cache::rkyv_store::load_cache(dir.path(), "bench").unwrap();
    }
    let load_ms = start.elapsed().as_millis() as f64 / runs as f64;

    let start = Instant::now();
    for _ in 0..runs {
        let _ = contextplus_rs::cache::rkyv_store::load_cache_mmap(dir.path(), "bench").unwrap();
    }
    let mmap_ms = start.elapsed().as_millis() as f64 / runs as f64;

    println!("Cache load (5000 vectors × 1024 dims, 100 iters):");
    println!("  rkyv deserialize: {:.2} ms", load_ms);
    println!("  rkyv mmap:        {:.2} ms", mmap_ms);
    println!("  speedup:          {:.1}x\n", load_ms / mmap_ms);

    // 4. Tree-sitter parse
    let ts_code = r#"
export interface ProfileServiceDependencies {
    prisma: PrismaClient;
    logger: Logger;
}
export function createProfileService(deps: ProfileServiceDependencies) {
    async function findById(id: string): Promise<Profile | null> {
        return prisma.profile.findUnique({ where: { id } });
    }
    async function create(data: CreateProfileInput): Promise<Profile> {
        return prisma.profile.create({ data });
    }
    return { findById, create };
}
export type ProfileService = ReturnType<typeof createProfileService>;
"#;
    let runs = 10_000;
    let start = Instant::now();
    for _ in 0..runs {
        let _ = contextplus_rs::core::tree_sitter::parse_with_tree_sitter(ts_code, "ts");
    }
    let parse_us = start.elapsed().as_micros() as f64 / runs as f64;
    println!(
        "Tree-sitter parse (TypeScript, 10K iters): {:.1} µs/op ({:.2} ms)",
        parse_us,
        parse_us / 1000.0
    );

    // 5. Hash check (no-op refresh path)
    let count = 30000;
    let keys: Vec<String> = (0..count).map(|i| format!("src/file_{}.ts", i)).collect();
    let hashes: Vec<String> = (0..count).map(|i| format!("hash_{}", i)).collect();
    let vectors = generate_vectors(count, dims);
    let store = VectorStore::new(dims as u32, keys, hashes, vectors);
    let file_hashes: Vec<(String, String)> = (0..count)
        .map(|i| (format!("src/file_{}.ts", i), format!("hash_{}", i)))
        .collect();

    let runs = 100;
    let start = Instant::now();
    for _ in 0..runs {
        let mut stale = 0;
        for (k, h) in &file_hashes {
            if store.get_hash(k) != Some(h.as_str()) {
                stale += 1;
            }
        }
        assert_eq!(stale, 0);
    }
    let hash_ms = start.elapsed().as_millis() as f64 / runs as f64;
    println!(
        "\nHash check staleness (30K files, 100 iters): {:.2} ms",
        hash_ms
    );

    // 6. SearchIndex hybrid search (flat buffer + simsimd + rayon)
    println!("\n--- SearchIndex hybrid search ---");
    for &count in &[1000, 3000] {
        let docs: Vec<SearchDocument> = (0..count)
            .map(|i| {
                SearchDocument::new(
                    format!("packages/domains/feature_{}/handler_{}.ts", i / 50, i),
                    format!("handler for feature {}", i),
                    vec![format!("handleFeature{}", i)],
                    vec![],
                    format!("processes feature {} data with validation", i),
                )
            })
            .collect();
        let vectors: Vec<Option<Vec<f32>>> = (0..count)
            .map(|i| {
                Some(
                    (0..dims)
                        .map(|d| ((i * 7 + d * 13 + 42) % 1000) as f32 / 1000.0)
                        .collect(),
                )
            })
            .collect();
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);
        let query_vec = generate_query(dims);
        let opts = ResolvedSearchOptions {
            top_k: 5,
            semantic_weight: 0.72,
            keyword_weight: 0.28,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.1,
            require_keyword_match: false,
            require_semantic_match: false,
        };
        // Warm up
        let _ = index.search("feature validation", &query_vec, &opts);
        let runs = 50;
        let start = Instant::now();
        for _ in 0..runs {
            black_box(index.search("feature validation", black_box(&query_vec), &opts));
        }
        let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        println!("  {} docs: {:.2} ms/query", count, avg_ms);
    }

    println!("\n=== vs TypeScript Baselines ===");
    println!("| Operation                  | TS (optimized) | Rust       | Speedup |");
    println!("|----------------------------|----------------|------------|---------|");
    println!(
        "| Cosine 1024-dim            | ~5 µs (JS)     | {:.3} µs   | {:.0}x     |",
        simd_us,
        5.0 / simd_us
    );
    println!(
        "| Cache load 5K vectors      | ~115 ms        | {:.2} ms   | {:.0}x     |",
        mmap_ms,
        115.0 / mmap_ms
    );
    println!(
        "| Tree-sitter parse (TS)     | ~10 ms (WASM)  | {:.2} ms   | {:.0}x     |",
        parse_us / 1000.0,
        10.0 / (parse_us / 1000.0)
    );
    println!(
        "| Hash check 30K (no-op)     | ~870 ms        | {:.2} ms   | {:.0}x     |",
        hash_ms,
        870.0 / hash_ms
    );
}
