//! Pre-generate and cache LLM labels for semantic_navigate.
//! Run once to warm the label cache, then all subsequent calls are instant.
//!
//! Strategy: runs semantic_navigate in semantic mode on the workspace root
//! AND on each major subdirectory. This generates LLM labels at every level
//! of the hierarchy, populating the cache aggressively.

use contextplus_rs::config::Config;
use contextplus_rs::core::embeddings::OllamaClient;
use contextplus_rs::core::walker;
use contextplus_rs::tools::semantic_navigate::{SemanticNavigateOptions, semantic_navigate};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let config = Config::from_env();
    let ollama = OllamaClient::new(&config);
    let root_arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/workspace".to_string());
    let root = Path::new(&root_arg);

    println!("Warming up LLM label cache for semantic_navigate...");
    println!("Root: {}", root.display());

    // Phase 1: Run on workspace root (hybrid mode for top-level structure)
    println!("\n=== Phase 1: Workspace root (hybrid mode) ===");
    run_warmup(&ollama, &config, root, "hybrid").await;

    // Phase 2: Discover major subdirectories and run semantic mode on each
    println!("\n=== Phase 2: Scoped directories (semantic mode) ===");
    let subdirs = discover_major_subdirs(root, &config);
    println!("Found {} directories to warm", subdirs.len());

    for (i, dir) in subdirs.iter().enumerate() {
        let rel = dir.strip_prefix(root).unwrap_or(dir);
        println!("\n[{}/{}] Warming: {}", i + 1, subdirs.len(), rel.display());
        run_warmup(&ollama, &config, dir, "semantic").await;
    }

    // Report
    let cache_path = root.join(".mcp_data").join("navigate-labels.json");
    if let Ok(data) = std::fs::read_to_string(&cache_path)
        && let Ok(cache) = serde_json::from_str::<HashMap<String, String>>(&data)
    {
        println!("\n=== Done! {} LLM labels cached ===", cache.len());
    }
}

async fn run_warmup(ollama: &OllamaClient, config: &Config, dir: &Path, mode: &str) {
    let options = SemanticNavigateOptions {
        root_dir: dir.to_string_lossy().to_string(),
        max_depth: Some(3),
        max_clusters: Some(10),
        min_clusters: Some(3),
        mode: Some(mode.to_string()),
    };

    let embedding_cache = RwLock::new(HashMap::new());
    match semantic_navigate(options, ollama, config, &embedding_cache, dir).await {
        Ok(_) => println!("  ✓ Labels cached"),
        Err(e) => eprintln!("  ✗ Error: {}", e),
    }
}

/// Discover major subdirectories worth warming (domains, platform, apps, etc.)
fn discover_major_subdirs(root: &Path, config: &Config) -> Vec<std::path::PathBuf> {
    let entries = walker::walk_with_config(root, config);
    let mut dirs: HashSet<std::path::PathBuf> = HashSet::new();

    for entry in &entries {
        if entry.is_directory {
            continue;
        }
        let parts: Vec<&str> = entry.relative_path.split('/').collect();
        // Find directories 2-3 levels deep with enough files
        if parts.len() >= 3 {
            let dir2 = root.join(parts[0]).join(parts[1]);
            dirs.insert(dir2);
            if parts.len() >= 4 {
                let dir3 = root.join(parts[0]).join(parts[1]).join(parts[2]);
                dirs.insert(dir3);
            }
        }
    }

    // Filter to directories that exist and have 10+ files
    let mut result: Vec<std::path::PathBuf> = dirs
        .into_iter()
        .filter(|d| d.is_dir())
        .filter(|d| {
            entries
                .iter()
                .filter(|e| !e.is_directory && e.path.starts_with(d))
                .count()
                >= 10
        })
        .collect();

    result.sort();
    // Deduplicate: remove parent if child is present
    let mut final_dirs: Vec<std::path::PathBuf> = Vec::new();
    for dir in &result {
        let dominated = result.iter().any(|other| {
            other != dir
                && other.starts_with(dir)
                && other.components().count() > dir.components().count()
        });
        if !dominated {
            final_dirs.push(dir.clone());
        }
    }
    final_dirs
}
