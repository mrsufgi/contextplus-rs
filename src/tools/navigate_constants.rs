//! Shared constants and helpers for semantic_navigate and its companion binaries.
//! Single source of truth — warmup_navigate and debug_eigenvalues import from here.

use std::collections::HashSet;
use std::sync::LazyLock;

use crate::core::embeddings::content_hash;

/// Default cap on files fed to the top-level navigate clustering step.
/// Even with Lanczos top-k eigendecomposition (O(n^2*k)), affinity-matrix
/// construction is O(n^2) at ~8 bytes/entry — n=2000 is ~32MB, still cheap;
/// n=3000+ pushes wall-clock past acceptable interactive latency on cold
/// caches. Override at runtime via `CONTEXTPLUS_NAVIGATE_MAX_FILES`.
pub const MAX_NAVIGATE_FILES_DEFAULT: usize = 2000;

/// Resolve the effective navigate file cap, honoring the
/// `CONTEXTPLUS_NAVIGATE_MAX_FILES` env override. Falls back to
/// `MAX_NAVIGATE_FILES_DEFAULT` when the var is unset or unparsable.
pub fn max_navigate_files() -> usize {
    std::env::var("CONTEXTPLUS_NAVIGATE_MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_NAVIGATE_FILES_DEFAULT)
}

/// Maximum files in a leaf cluster before it gets sub-clustered.
/// Lowered from 20 to 10 to force semantic splitting on medium groups
/// (e.g., a 15-file service/ directory gets split into business logic vs tests).
pub const MAX_FILES_PER_LEAF: usize = 10;

/// Maximum files to display per leaf cluster in rendered output.
pub const MAX_FILES_PER_LEAF_DISPLAY: usize = 10;

/// Minimum files in a directory group before it gets merged into "other".
pub const MIN_DIR_GROUP_SIZE: usize = 5;

/// Maximum characters of file content used for embedding.
pub const MAX_CONTENT_CHARS: usize = 500;

/// Maximum header lines to scan for file description.
pub const MAX_HEADER_LINES: usize = 5;

/// Maximum header length in characters.
pub const MAX_HEADER_LEN: usize = 200;

/// Cache hash version prefix. Bump when embed text format changes
/// to invalidate old cached vectors.
/// Bumped to nav4 for FNV-1a 64-bit hash (was djb2 32-bit in nav3).
const NAV_HASH_VERSION: &str = "nav4:";

/// Directory segments too generic to use as cluster labels.
pub static GENERIC_SEGMENTS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "src", "lib", "dist", "build", "utils", "helpers", "common", "shared", "core", "types",
        "config", "internal", "cmd", "pkg",
    ]
    .into_iter()
    .collect()
});

/// Extensions accepted for semantic navigation (without leading dot).
pub const NAVIGATE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "go", "java", "c", "cpp", "h", "hpp", "cc",
    "rb", "sh", "bash", "zsh", "sql", "graphql", "proto", "yaml", "yml", "toml", "json",
];

/// Build the path-weighted embed text for a file.
/// Path is repeated 3x to boost domain signal over code pattern similarity.
pub fn nav_embed_text(path: &str, header: &str, content: &str) -> String {
    format!("{p} {p} {p} {h} {c}", p = path, h = header, c = content)
}

/// Compute the navigate-specific content hash for cache invalidation.
/// Includes version prefix + path + content so format changes invalidate cache.
pub fn nav_content_hash(path: &str, content: &str) -> String {
    content_hash(&format!("{}{}{}", NAV_HASH_VERSION, path, content))
}

/// File name for cached cluster labels (LLM-generated).
pub const LABEL_CACHE_FILE: &str = "navigate-labels.json";

/// Maximum clusters to send to LLM in a single batch (avoids timeout).
pub const LLM_BATCH_SIZE: usize = 10;

/// Maximum files to sample per cluster when building LLM label prompts.
pub const MAX_FILES_PER_LABEL: usize = 5;

/// Blend ratio for embedding vs import-graph affinity (1.0 = pure embedding, 0.0 = pure imports).
/// 0.9 means 90% embedding + 10% import adjacency for gentle structural nudging.
pub const IMPORT_BLEND_ALPHA: f64 = 0.9;

/// Build the navigate cache file name for a given embedding model.
/// Uses server::sanitize_model_name for filesystem safety
/// (e.g., "unclemusclez/jina-embeddings-v2-base-code" → "navigate-unclemusclez-jina-embeddings-v2-base-code")
pub fn nav_cache_name(model: &str) -> String {
    format!("navigate-{}", crate::server::sanitize_model_name(model))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_segments_contains_known_segment() {
        assert!(
            GENERIC_SEGMENTS.contains("src"),
            "expected 'src' to be in GENERIC_SEGMENTS"
        );
        assert!(
            GENERIC_SEGMENTS.contains("utils"),
            "expected 'utils' to be in GENERIC_SEGMENTS"
        );
    }

    #[test]
    fn generic_segments_does_not_contain_domain_segment() {
        assert!(
            !GENERIC_SEGMENTS.contains("billing"),
            "expected 'billing' to not be in GENERIC_SEGMENTS"
        );
        assert!(
            !GENERIC_SEGMENTS.contains("scheduling"),
            "expected 'scheduling' to not be in GENERIC_SEGMENTS"
        );
    }
}
