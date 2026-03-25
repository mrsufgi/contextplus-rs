//! Debug tool: load cached embeddings, run spectral clustering, print eigenvalue spectrum.

use contextplus_rs::cache::rkyv_store;
use contextplus_rs::core::clustering::spectral_cluster_with_min;
use contextplus_rs::tools::navigate_constants::{MAX_NAVIGATE_FILES, nav_cache_name};
use std::path::Path;

fn main() {
    let root = Path::new("/workspace");
    let model = std::env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
    let cache_name = nav_cache_name(&model);

    let store = rkyv_store::load_vector_store(root, &cache_name)
        .expect("Failed to load embedding cache")
        .expect("No cache file found");

    let dims = store.dims() as usize;
    let count = store.count() as usize;
    let flat = store.vectors_data();

    println!("Loaded {} vectors, dim={}", count, dims);

    // Reconstruct Vec<Vec<f32>> from flat data
    let all_vectors: Vec<Vec<f32>> = (0..count)
        .map(|i| flat[i * dims..(i + 1) * dims].to_vec())
        .collect();

    // Sample to MAX_NAVIGATE_FILES to match semantic_navigate behavior
    let vectors: Vec<Vec<f32>> = if all_vectors.len() > MAX_NAVIGATE_FILES {
        let step = all_vectors.len() as f64 / MAX_NAVIGATE_FILES as f64;
        (0..MAX_NAVIGATE_FILES)
            .map(|i| {
                let idx = (i as f64 * step).floor() as usize;
                all_vectors[idx.min(all_vectors.len() - 1)].clone()
            })
            .collect()
    } else {
        all_vectors
    };

    println!("Clustering {} vectors with max_clusters=10...", vectors.len());

    // Enable tracing to see our debug logs
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let results = spectral_cluster_with_min(&vectors, 10, 2);
    println!("\nResult: {} clusters", results.len());
    for (i, c) in results.iter().enumerate() {
        println!("  Cluster {}: {} files", i, c.indices.len());
    }
}
