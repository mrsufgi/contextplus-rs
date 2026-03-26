//! Shared constants and helpers for semantic_navigate and its companion binaries.
//! Single source of truth — warmup_navigate and debug_eigenvalues import from here.

use crate::core::embeddings::content_hash;

/// Maximum files to cluster. Set to usize::MAX to disable sampling.
/// With directory-based grouping at depth 0, spectral clustering only
/// runs on individual groups (50-400 files each), not the full file set.
/// The O(n³) eigen is bounded by group size, not total files.
pub const MAX_NAVIGATE_FILES: usize = usize::MAX;

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
pub const GENERIC_SEGMENTS: &[&str] = &[
    "src", "lib", "dist", "build", "utils", "helpers", "common", "shared",
    "core", "types", "config", "internal", "cmd", "pkg",
];

/// Extensions accepted for semantic navigation (without leading dot).
pub const NAVIGATE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "go", "java",
    "c", "cpp", "h", "hpp", "cc",
    "rb", "sh", "bash", "zsh",
    "sql", "graphql", "proto", "yaml", "yml", "toml", "json",
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

/// Build the navigate cache file name for a given embedding model.
pub fn nav_cache_name(model: &str) -> String {
    format!("navigate-{}", model)
}
