//! Warmup tool: pre-builds the file-content embedding cache used by
//! semantic_code_search. Mirrors warmup_identifiers but populates the
//! `embeddings-{model}.rkyv` store keyed by relative file path.
//!
//! The MCP server's tracker only embeds files in response to filesystem
//! changes or query traffic — it does NOT do a bulk pass at startup, even
//! in `eager` mode. On a fresh checkout this leaves `semantic_code_search`
//! falling back to lazy embed-on-demand for nearly every result, which is
//! painfully slow on CPU-only Ollama. Run this once to seed the cache; the
//! tracker keeps it fresh from then on.
//!
//! Usage: `warmup_embeddings [root_dir]`
//!   `root_dir`: workspace root (default: current directory)

use contextplus_rs::cache::rkyv_store;
use contextplus_rs::config::Config;
use contextplus_rs::core::embeddings::{CacheEntry, OllamaClient, VectorStore};
use contextplus_rs::core::parser::hash_content;
use contextplus_rs::core::tree_sitter::get_supported_extensions;
use contextplus_rs::core::walker;
use contextplus_rs::server::cache_name;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let config = Config::from_env();
    let ollama = OllamaClient::new(&config);

    let root_arg = std::env::args().nth(1);
    let root = Path::new(root_arg.as_deref().unwrap_or("."));
    let root = if root.is_relative() {
        std::env::current_dir().unwrap().join(root)
    } else {
        root.to_path_buf()
    };
    let root = root.as_path();

    let supported: HashSet<&str> = get_supported_extensions().iter().copied().collect();
    let max_size = config.max_embed_file_size as u64;

    // Step 1: enumerate candidate files
    println!("Scanning workspace: {}", root.display());
    let entries = walker::walk_with_config(root, &config);

    // (rel_path, content_hash, content)
    let mut docs: Vec<(String, String, String)> = Vec::new();
    let mut skipped_size = 0usize;
    let mut skipped_unsupported = 0usize;

    for entry in &entries {
        if entry.is_directory {
            continue;
        }
        let ext_with_dot = entry
            .path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{}", e))
            .unwrap_or_default();
        if !supported.contains(ext_with_dot.as_str()) {
            skipped_unsupported += 1;
            continue;
        }
        if let Ok(meta) = std::fs::metadata(&entry.path)
            && meta.len() > max_size
        {
            skipped_size += 1;
            continue;
        }
        let content = match std::fs::read_to_string(&entry.path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if content.trim().is_empty() {
            continue;
        }
        let hash = hash_content(&content);
        docs.push((entry.relative_path.clone(), hash, content));
    }

    println!(
        "Found {} embeddable files (skipped: {} oversized, {} unsupported)",
        docs.len(),
        skipped_size,
        skipped_unsupported
    );

    if docs.is_empty() {
        println!("Nothing to embed.");
        return;
    }

    // Step 2: load existing cache; partition into hits / misses
    let cache_name_str = cache_name("embeddings", &config.ollama_embed_model);

    // Load into a HashMap so we can merge new entries later without losing
    // anything previously persisted by another writer (e.g. the live MCP).
    let mut cache_map: HashMap<String, CacheEntry> =
        match rkyv_store::load_cache(root, &cache_name_str) {
            Ok(Some(data)) => {
                let dim = data.dims as usize;
                let mut map = HashMap::with_capacity(data.keys.len());
                for (i, key) in data.keys.iter().enumerate() {
                    let off = i * dim;
                    if off + dim > data.vectors.len() {
                        continue;
                    }
                    let hash = data.hashes.get(i).cloned().unwrap_or_default();
                    map.insert(
                        key.clone(),
                        CacheEntry {
                            hash,
                            vector: data.vectors[off..off + dim].to_vec(),
                        },
                    );
                }
                println!("Loaded {} existing cache entries", map.len());
                map
            }
            _ => HashMap::new(),
        };

    // Files needing embed = those whose hash differs from cache (or absent)
    let needs_embed: Vec<usize> = docs
        .iter()
        .enumerate()
        .filter_map(|(i, (rel, hash, _))| match cache_map.get(rel) {
            Some(entry) if &entry.hash == hash => None, // already cached, content unchanged
            _ => Some(i),
        })
        .collect();

    println!(
        "{} files need embedding, {} already cached and current",
        needs_embed.len(),
        docs.len() - needs_embed.len()
    );

    if needs_embed.is_empty() {
        println!("Cache is fully warm!");
        return;
    }

    // Step 3: embed in batches; persist after each batch so a crash mid-run
    // doesn't lose hours of work.
    let batch_size = config.embed_batch_size.max(1);
    let total_batches = needs_embed.len().div_ceil(batch_size);
    let mut completed = 0usize;
    let mut failed_batches = 0usize;

    for (batch_idx, chunk_start) in (0..needs_embed.len()).step_by(batch_size).enumerate() {
        let chunk_end = (chunk_start + batch_size).min(needs_embed.len());
        let indices = &needs_embed[chunk_start..chunk_end];
        let texts: Vec<String> = indices.iter().map(|&i| docs[i].2.clone()).collect();

        print!(
            "Embedding batch {}/{} ({} files)... ",
            batch_idx + 1,
            total_batches,
            texts.len()
        );

        match ollama.embed(&texts).await {
            Ok(vectors) => {
                let mut added = 0usize;
                for (j, &doc_idx) in indices.iter().enumerate() {
                    if let Some(vec) = vectors.get(j) {
                        let (rel, hash, _) = &docs[doc_idx];
                        cache_map.insert(
                            rel.clone(),
                            CacheEntry {
                                hash: hash.clone(),
                                vector: vec.clone(),
                            },
                        );
                        added += 1;
                    }
                }
                completed += added;
                println!("ok ({} added)", added);
            }
            Err(e) => {
                failed_batches += 1;
                println!("FAILED: {}", e);
                continue;
            }
        }

        // Persist progress after each batch.
        let store = VectorStore::from_cache(&cache_map);
        if let Some(s) = store
            && let Err(e) = rkyv_store::save_vector_store(root, &cache_name_str, &s)
        {
            eprintln!(
                "WARN: failed to flush cache after batch {}: {}",
                batch_idx + 1,
                e
            );
        }
    }

    println!(
        "\nDone! {} new file embeddings; {} batches failed; {} total cache entries on disk.",
        completed,
        failed_batches,
        cache_map.len()
    );
}
