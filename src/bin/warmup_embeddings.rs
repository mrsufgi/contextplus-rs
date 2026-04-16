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
                let map = cache_data_to_map(&data);
                println!("Loaded {} existing cache entries", map.len());
                map
            }
            _ => HashMap::new(),
        };

    let needs_embed = partition_needs_embed(&docs, &cache_map);

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

/// Decide which docs need re-embedding: every doc whose `(rel_path, hash)`
/// pair is not already current in `cache_map`.
///
/// A doc is skipped only when the cache holds the **same key AND same hash**.
/// Different hash for the same key (file content changed since last embed)
/// returns the index for re-embedding. Missing key likewise.
fn partition_needs_embed(
    docs: &[(String, String, String)],
    cache_map: &HashMap<String, CacheEntry>,
) -> Vec<usize> {
    docs.iter()
        .enumerate()
        .filter_map(|(i, (rel, hash, _))| match cache_map.get(rel) {
            Some(entry) if &entry.hash == hash => None,
            _ => Some(i),
        })
        .collect()
}

/// Convert a flat `CacheData` (parallel keys/hashes/vectors arrays) back into
/// a `key → CacheEntry` map suitable for in-memory updates before the next
/// `save_vector_store` call.
///
/// Defensive: skips any entry whose vector range would walk past the end of
/// the flat `vectors` slice, guarding against truncated cache files. Misses
/// in `hashes` fall back to the empty string so a malformed entry doesn't
/// kill the load.
fn cache_data_to_map(data: &rkyv_store::CacheData) -> HashMap<String, CacheEntry> {
    let dim = data.dims as usize;
    let mut map = HashMap::with_capacity(data.keys.len());
    if dim == 0 {
        return map;
    }
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
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(rel: &str, hash: &str, body: &str) -> (String, String, String) {
        (rel.to_string(), hash.to_string(), body.to_string())
    }

    fn entry(hash: &str, dim: usize) -> CacheEntry {
        CacheEntry {
            hash: hash.to_string(),
            vector: vec![0.0; dim],
        }
    }

    #[test]
    fn partition_empty_cache_returns_all_docs() {
        let docs = vec![doc("a.ts", "h1", "x"), doc("b.ts", "h2", "y")];
        let cache: HashMap<String, CacheEntry> = HashMap::new();

        let result = partition_needs_embed(&docs, &cache);

        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn partition_all_cached_with_matching_hash_returns_empty() {
        let docs = vec![doc("a.ts", "h1", "x"), doc("b.ts", "h2", "y")];
        let mut cache = HashMap::new();
        cache.insert("a.ts".into(), entry("h1", 4));
        cache.insert("b.ts".into(), entry("h2", 4));

        let result = partition_needs_embed(&docs, &cache);

        assert!(result.is_empty(), "fully-current cache should need no work");
    }

    #[test]
    fn partition_stale_hash_re_embeds() {
        // Same key, different hash => re-embed (content changed since last cache)
        let docs = vec![doc("a.ts", "h1-NEW", "x")];
        let mut cache = HashMap::new();
        cache.insert("a.ts".into(), entry("h1-OLD", 4));

        let result = partition_needs_embed(&docs, &cache);

        assert_eq!(result, vec![0]);
    }

    #[test]
    fn partition_mixed_only_returns_changed_or_missing() {
        let docs = vec![
            doc("kept.ts", "h-keep", "k"),    // index 0: cached + current → skip
            doc("changed.ts", "h-NEW", "c"),  // index 1: cached but stale → embed
            doc("brandnew.ts", "h-new", "n"), // index 2: not in cache → embed
        ];
        let mut cache = HashMap::new();
        cache.insert("kept.ts".into(), entry("h-keep", 4));
        cache.insert("changed.ts".into(), entry("h-OLD", 4));

        let result = partition_needs_embed(&docs, &cache);

        assert_eq!(result, vec![1, 2]);
    }

    // -- cache_data_to_map --

    #[test]
    fn cache_data_to_map_round_trip_basic() {
        let data = rkyv_store::CacheData {
            dims: 2,
            keys: vec!["a.ts".into(), "b.ts".into()],
            hashes: vec!["ha".into(), "hb".into()],
            vectors: vec![0.1, 0.2, 0.3, 0.4],
        };
        let map = cache_data_to_map(&data);
        assert_eq!(map.len(), 2);
        let a = &map["a.ts"];
        assert_eq!(a.hash, "ha");
        assert_eq!(a.vector, vec![0.1, 0.2]);
        let b = &map["b.ts"];
        assert_eq!(b.hash, "hb");
        assert_eq!(b.vector, vec![0.3, 0.4]);
    }

    #[test]
    fn cache_data_to_map_zero_dims_returns_empty() {
        // A degenerate cache file with dims=0 must not panic on the slice math.
        let data = rkyv_store::CacheData {
            dims: 0,
            keys: vec!["a.ts".into()],
            hashes: vec!["ha".into()],
            vectors: vec![],
        };
        assert!(cache_data_to_map(&data).is_empty());
    }

    #[test]
    fn cache_data_to_map_skips_truncated_tail_entries() {
        // Truncated cache: 3 keys but only 2 entries' worth of vector data.
        // The third entry must be skipped, not panic on out-of-bounds slicing.
        let data = rkyv_store::CacheData {
            dims: 2,
            keys: vec!["a.ts".into(), "b.ts".into(), "c.ts".into()],
            hashes: vec!["ha".into(), "hb".into(), "hc".into()],
            vectors: vec![0.1, 0.2, 0.3, 0.4], // only 4 floats = 2 entries
        };
        let map = cache_data_to_map(&data);
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("a.ts"));
        assert!(map.contains_key("b.ts"));
        assert!(!map.contains_key("c.ts"));
    }

    #[test]
    fn cache_data_to_map_missing_hash_falls_back_to_empty_string() {
        // If hashes vec is shorter than keys, fall back to "" rather than panic.
        let data = rkyv_store::CacheData {
            dims: 1,
            keys: vec!["a.ts".into(), "b.ts".into()],
            hashes: vec!["ha".into()], // missing hash for b.ts
            vectors: vec![0.1, 0.2],
        };
        let map = cache_data_to_map(&data);
        assert_eq!(map["a.ts"].hash, "ha");
        assert_eq!(map["b.ts"].hash, "");
    }

    #[test]
    fn partition_extra_cache_entries_are_ignored() {
        // The cache may contain entries for files no longer in the workspace.
        // partition_needs_embed only considers what's in `docs`; orphans are
        // left alone (the merging save_cache will preserve them on disk).
        let docs = vec![doc("present.ts", "h-p", "p")];
        let mut cache = HashMap::new();
        cache.insert("present.ts".into(), entry("h-p", 4));
        cache.insert("deleted-since.ts".into(), entry("h-x", 4));

        let result = partition_needs_embed(&docs, &cache);

        assert!(result.is_empty());
    }
}
