//! Warmup tool: pre-embed files with path-weighted format for semantic_navigate.
use contextplus_rs::cache::rkyv_store;
use contextplus_rs::config::Config;
use contextplus_rs::core::embeddings::{CacheEntry, OllamaClient, VectorStore, content_hash};
use contextplus_rs::core::walker;
use std::collections::{HashMap, HashSet};
use std::path::Path;

const NAVIGATE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "go", "java",
    "rb", "sh", "sql", "graphql", "proto", "yaml", "yml", "toml", "json",
];
const MAX_NAVIGATE_FILES: usize = 1500;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let config = Config::from_env();
    let ollama = OllamaClient::new(&config);
    let root = Path::new("/workspace");
    let allowed: HashSet<&str> = NAVIGATE_EXTENSIONS.iter().copied().collect();

    // Walk files
    let entries = walker::walk_with_config(root, &config);
    let max_size = config.max_embed_file_size as u64;
    let mut files: Vec<(String, String)> = Vec::new(); // (path, content)

    for entry in entries {
        if entry.is_directory { continue; }
        let ext = entry.path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !allowed.contains(ext) { continue; }
        if let Ok(meta) = std::fs::metadata(&entry.path) {
            if meta.len() > max_size { continue; }
        }
        if let Ok(content) = std::fs::read_to_string(&entry.path) {
            let truncated = if content.len() > 500 {
                contextplus_rs::core::parser::truncate_to_char_boundary(&content, 500).to_string()
            } else { content };
            files.push((entry.relative_path, truncated));
        }
    }

    println!("Found {} files", files.len());

    // Sample if needed
    if files.len() > MAX_NAVIGATE_FILES {
        let total = files.len();
        let step = total as f64 / MAX_NAVIGATE_FILES as f64;
        files = (0..MAX_NAVIGATE_FILES)
            .map(|i| {
                let idx = (i as f64 * step).floor() as usize;
                files[idx.min(total - 1)].clone()
            })
            .collect();
    }

    println!("Processing {} files with path-weighted embedding", files.len());

    // Load existing cache
    let cache_name = format!("navigate-{}", config.ollama_embed_model);
    let mut cache: HashMap<String, CacheEntry> = HashMap::new();
    if let Ok(Some(store)) = rkyv_store::load_vector_store(root, &cache_name) {
        let dims = store.dims() as usize;
        let flat = store.vectors_data();
        let keys = store.keys();
        let hashes = store.hashes();
        for (i, key) in keys.iter().enumerate() {
            let vec = flat[i * dims..(i + 1) * dims].to_vec();
            cache.insert(key.clone(), CacheEntry {
                hash: hashes[i].clone(),
                vector: vec,
            });
        }
        println!("Loaded {} existing cache entries", cache.len());
    }

    // Find uncached files (with nav: prefix hash)
    let mut uncached: Vec<(usize, String)> = Vec::new(); // (index, embed_text)
    let mut nav_hashes: Vec<String> = Vec::new();

    for (i, (path, content)) in files.iter().enumerate() {
        let nav_hash = content_hash(&format!("nav3:{}{}", path, content));
        nav_hashes.push(nav_hash.clone());

        if let Some(entry) = cache.get(path) {
            if entry.hash == nav_hash { continue; }
        }
        let embed_text = format!("{p} {p} {p} {content}", p = path);
        uncached.push((i, embed_text));
    }

    println!("{} files need re-embedding, {} cached", uncached.len(), files.len() - uncached.len());

    if uncached.is_empty() {
        println!("Cache is fully warm!");
        return;
    }

    // Embed in batches
    let batch_size = config.embed_batch_size;
    let total_batches = (uncached.len() + batch_size - 1) / batch_size;

    for (batch_idx, chunk) in uncached.chunks(batch_size).enumerate() {
        let texts: Vec<String> = chunk.iter().map(|(_, t)| t.clone()).collect();
        print!("Batch {}/{} ({} files)... ", batch_idx + 1, total_batches, texts.len());

        match ollama.embed(&texts).await {
            Ok(vectors) => {
                for (j, (file_idx, _)) in chunk.iter().enumerate() {
                    if j < vectors.len() {
                        let (path, _) = &files[*file_idx];
                        cache.insert(path.clone(), CacheEntry {
                            hash: nav_hashes[*file_idx].clone(),
                            vector: vectors[j].clone(),
                        });
                    }
                }
                println!("ok");
            }
            Err(e) => {
                println!("FAILED: {}", e);
            }
        }

        // Save after each batch
        let store = VectorStore::from_cache(&cache);
        if let Some(s) = store {
            if let Err(e) = rkyv_store::save_vector_store(root, &cache_name, &s) {
                eprintln!("Failed to save cache: {}", e);
            }
        }
    }

    println!("\nDone! Cache now has {} entries", cache.len());
}
