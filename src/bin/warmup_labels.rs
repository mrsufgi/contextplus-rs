//! Pre-generate and cache LLM labels for semantic_navigate.
//! Run once to warm the label cache, then all subsequent calls are instant.
//!
//! This binary runs semantic_navigate in hybrid mode, triggering LLM calls
//! for all clusters. The labels are cached to disk via the label cache mechanism,
//! so subsequent semantic_navigate calls skip the LLM entirely.

use contextplus_rs::config::Config;
use contextplus_rs::core::embeddings::OllamaClient;
use contextplus_rs::tools::semantic_navigate::{SemanticNavigateOptions, semantic_navigate};
use std::collections::HashMap;
use std::path::Path;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let config = Config::from_env();
    let ollama = OllamaClient::new(&config);
    let root_arg = std::env::args().nth(1).unwrap_or_else(|| "/workspace".to_string());
    let root = Path::new(&root_arg);

    println!("Warming up LLM label cache for semantic_navigate...");
    println!("Root: {}", root.display());

    // Use semantic mode for warmup — it generates LLM labels at EVERY level,
    // not just for large clusters. This populates the cache more aggressively.
    let options = SemanticNavigateOptions {
        root_dir: root.to_string_lossy().to_string(),
        max_depth: Some(3),
        max_clusters: Some(10),
        min_clusters: Some(3),
        mode: Some("semantic".to_string()),
    };

    let embedding_cache = RwLock::new(HashMap::new());

    match semantic_navigate(options, &ollama, &config, &embedding_cache, root).await {
        Ok(output) => {
            println!("\nNavigation complete. Labels are now cached.\n");
            println!("{}", output);
        }
        Err(e) => {
            eprintln!("Error during warmup: {}", e);
            std::process::exit(1);
        }
    }

    println!("\nLabel cache is warm! Subsequent semantic_navigate calls will be instant.");
}
