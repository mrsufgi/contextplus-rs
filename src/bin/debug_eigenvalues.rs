//! Debug tool: load cached embeddings, run spectral clustering, print eigenvalue spectrum.

use contextplus_rs::cache::rkyv_store;
use contextplus_rs::core::clustering::{
    build_affinity_matrix, full_eigen, find_optimal_k, normalized_laplacian,
    spectral_cluster_with_min,
};
use contextplus_rs::tools::navigate_constants::{MAX_NAVIGATE_FILES, nav_cache_name};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    // Init tracing BEFORE anything else so early logs aren't lost
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let root: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("Failed to get current directory"));
    let root = root.as_path();

    let model = std::env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
    let cache_name = nav_cache_name(&model);

    println!("Root path: {}", root.display());
    println!("Cache name: {cache_name}");

    // --- Cache load ---
    let t0 = Instant::now();
    let store = rkyv_store::load_vector_store(root, &cache_name)
        .unwrap_or_else(|e| panic!("Failed to load embedding cache at {}: {e}", root.display()))
        .unwrap_or_else(|| panic!("No cache file found at {}/{cache_name}", root.display()));
    let cache_load_ms = t0.elapsed().as_millis();
    println!("Cache load: {cache_load_ms}ms");

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

    let n = vectors.len();
    println!("\nClustering {n} vectors with max_clusters=10...\n");

    // --- Affinity matrix ---
    let t1 = Instant::now();
    let affinity = build_affinity_matrix(&vectors);
    let affinity_ms = t1.elapsed().as_millis();
    println!("Affinity matrix build: {affinity_ms}ms");

    // --- Laplacian + eigendecomposition ---
    let t2 = Instant::now();
    let laplacian = normalized_laplacian(&affinity);
    let laplacian_ms = t2.elapsed().as_millis();
    println!("Normalized Laplacian: {laplacian_ms}ms");

    let t3 = Instant::now();
    let (eigenvalues, _eigenvectors) = full_eigen(laplacian);
    let eigen_ms = t3.elapsed().as_millis();
    println!("Eigendecomposition: {eigen_ms}ms");

    // --- Print eigenvalues ---
    let display_count = eigenvalues.len().min(10);
    println!("\nFirst {display_count} eigenvalues:");
    for (i, &val) in eigenvalues[..display_count].iter().enumerate() {
        let gap = if i > 0 {
            format!("  gap={:.6}", val - eigenvalues[i - 1])
        } else {
            String::new()
        };
        println!("  [{i}] {val:.6}{gap}");
    }

    let optimal_k = find_optimal_k(&eigenvalues, 10);
    println!("\nOptimal k (eigengap heuristic): {optimal_k}");

    // --- Full clustering (for comparison) ---
    let t4 = Instant::now();
    let results = spectral_cluster_with_min(&vectors, 10, 2);
    let cluster_ms = t4.elapsed().as_millis();
    println!("\nFull clustering: {cluster_ms}ms");

    println!("Result: {} clusters", results.len());
    for (i, c) in results.iter().enumerate() {
        println!("  Cluster {i}: {} files", c.indices.len());
    }

    let total_ms = t0.elapsed().as_millis();
    println!("\nTotal time: {total_ms}ms");
}
