//! Warmup tool: pre-builds the identifier embedding cache for semantic_identifier_search.
//! Run before first use to avoid MCP timeout on cold start.
//!
//! Usage: `warmup_identifiers [root_dir]`
//!   `root_dir`: workspace root (default: current directory)

use contextplus_rs::cache::rkyv_store;
use contextplus_rs::config::Config;
use contextplus_rs::core::embeddings::{OllamaClient, VectorStore};
use contextplus_rs::core::parser::{extract_header, flatten_symbols, hash_content};
use contextplus_rs::core::tree_sitter::{get_supported_extensions, parse_with_tree_sitter};
use contextplus_rs::core::walker;
use contextplus_rs::server::cache_name;
use contextplus_rs::tools::semantic_identifiers::IdentifierDoc;
use std::collections::HashSet;
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

    // Step 1: Walk workspace and parse symbols with tree-sitter
    println!("Scanning workspace: {}", root.display());
    let entries = walker::walk_with_config(root, &config);

    let mut identifier_docs: Vec<IdentifierDoc> = Vec::new();
    let mut files_parsed = 0usize;

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
            continue;
        }
        if let Ok(meta) = std::fs::metadata(&entry.path)
            && meta.len() > max_size
        {
            continue;
        }
        let content = match std::fs::read_to_string(&entry.path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        if let Ok(symbols) = parse_with_tree_sitter(&content, &ext_with_dot) {
            let header = extract_header(&content);
            for sym in flatten_symbols(&symbols, None) {
                let sig = sym.signature.clone().unwrap_or_default();
                let parent = sym.parent_name.as_deref().unwrap_or("");
                let text = format!(
                    "{} {} {} {} {} {}",
                    sym.name, sym.kind, sig, entry.relative_path, header, parent
                );
                let token_set =
                    IdentifierDoc::build_token_set(&sym.name, &sig, &entry.relative_path, &header);
                identifier_docs.push(IdentifierDoc {
                    id: format!("{}:{}:{}", entry.relative_path, sym.name, sym.line),
                    path: entry.relative_path.clone(),
                    header: header.clone(),
                    name: sym.name.clone(),
                    kind_lower: sym.kind.to_lowercase(),
                    kind: sym.kind.clone(),
                    line: sym.line,
                    end_line: sym.end_line,
                    signature: sig,
                    parent_name: sym.parent_name.clone(),
                    text,
                    token_set,
                });
            }
            files_parsed += 1;
        }
    }

    println!(
        "Parsed {} files, found {} identifiers",
        files_parsed,
        identifier_docs.len()
    );

    if identifier_docs.is_empty() {
        println!("No identifiers found — nothing to embed.");
        return;
    }

    // Step 2: Load existing identifier embedding cache
    let id_cache_name = cache_name("identifier-embeddings", &config.ollama_embed_model);
    let id_cache = match rkyv_store::load_cache(root, &id_cache_name) {
        Ok(Some(data)) => {
            let store = data.to_store();
            println!("Loaded {} existing cache entries", store.count());
            Some(store)
        }
        _ => None,
    };

    // Partition: cached vs uncached
    let n_identifiers = identifier_docs.len();
    let mut result_vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(n_identifiers);
    let mut uncached_indices: Vec<usize> = Vec::new();
    let mut uncached_texts: Vec<String> = Vec::new();

    for (i, doc) in identifier_docs.iter().enumerate() {
        if let Some(ref store) = id_cache
            && let Some(vec) = store.get_vector(&doc.text)
        {
            result_vectors.push(Some(vec.to_vec()));
            continue;
        }
        result_vectors.push(None);
        uncached_indices.push(i);
        uncached_texts.push(doc.text.clone());
    }

    println!(
        "{} identifiers need embedding, {} already cached",
        uncached_indices.len(),
        n_identifiers - uncached_indices.len()
    );

    if uncached_indices.is_empty() {
        println!("Cache is fully warm!");
        return;
    }

    // Step 3: Embed in batches, saving progress after each batch
    let batch_size = config.embed_batch_size;
    let total_batches = uncached_indices.len().div_ceil(batch_size);

    for (batch_idx, chunk_start) in (0..uncached_indices.len()).step_by(batch_size).enumerate() {
        let chunk_end = (chunk_start + batch_size).min(uncached_indices.len());
        let chunk_texts = &uncached_texts[chunk_start..chunk_end];

        print!(
            "Embedding batch {}/{} ({} identifiers)... ",
            batch_idx + 1,
            total_batches,
            chunk_texts.len()
        );

        match ollama.embed(chunk_texts).await {
            Ok(vectors) => {
                for (local_j, &idx) in uncached_indices[chunk_start..chunk_end].iter().enumerate() {
                    if local_j < vectors.len() {
                        result_vectors[idx] = Some(vectors[local_j].clone());
                    }
                }
                println!("ok");
            }
            Err(e) => {
                println!("FAILED: {}", e);
                continue;
            }
        }

        // Save progress after each batch
        let all_vecs: Vec<Vec<f32>> = result_vectors.iter().filter_map(|v| v.clone()).collect();
        if all_vecs.len() == n_identifiers {
            let dims = all_vecs.first().map_or(0, |v| v.len()) as u32;
            let keys: Vec<String> = identifier_docs.iter().map(|d| d.text.clone()).collect();
            let hashes: Vec<String> = keys.iter().map(|k| hash_content(k)).collect();
            let flat: Vec<f32> = all_vecs.into_iter().flatten().collect();
            let store = VectorStore::new(dims, keys, hashes, flat);
            if let Err(e) = rkyv_store::save_vector_store(root, &id_cache_name, &store) {
                eprintln!("Failed to save cache: {}", e);
            }
        }
    }

    // Final save for partial caches (when some batches failed)
    let cached_count = result_vectors.iter().filter(|v| v.is_some()).count();
    println!(
        "\nDone! {} / {} identifiers embedded and cached",
        cached_count, n_identifiers
    );
}
