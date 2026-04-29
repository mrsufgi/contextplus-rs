// Semantic project navigator using spectral clustering and Ollama labeling.
// Browse codebase by meaning: embeds files, clusters vectors, generates labels.

use crate::cache::rkyv_store;
use crate::config::Config;
use crate::core::clustering::{
    blend_affinity_matrices, build_affinity_matrix, build_import_adjacency,
    spectral_cluster_with_affinity,
};
use crate::core::embeddings::VectorStore;
use crate::core::embeddings::{CacheEntry, OllamaClient};
use crate::core::walker;
use crate::error::Result;
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use tokio::io::AsyncReadExt;
use tokio::sync::RwLock;

use super::labels::*;
use super::modes::ClusterParams;
use super::modes::hybrid::{build_labeled_tree, group_by_directory};
use super::modes::imports::extract_all_import_edges;
use super::modes::semantic::build_semantic_hierarchy;
use super::navigate_constants::*;

/// Quality marker for a cached cluster label. Heuristic labels are produced
/// instantly from path/symbol signal; LLM labels are produced by the chat model
/// and considered higher quality. The on-disk cache progressively heals from
/// `Heuristic` to `Llm` as the background labeler runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum LabelQuality {
    Heuristic,
    Llm,
}

/// On-disk representation of a cluster label, with its quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CachedLabel {
    pub label: String,
    pub quality: LabelQuality,
}

impl CachedLabel {
    pub(crate) fn heuristic(label: String) -> Self {
        Self {
            label,
            quality: LabelQuality::Heuristic,
        }
    }

    pub(crate) fn llm(label: String) -> Self {
        Self {
            label,
            quality: LabelQuality::Llm,
        }
    }
}

/// Backward-compatible deserializer: accepts either the legacy `String` form
/// (treated as `quality: "llm"`) or the new `{ "label": ..., "quality": ... }`
/// form. Existing `navigate-labels.json` files in the wild MUST keep loading.
fn deserialize_label_map<'de, D>(
    deserializer: D,
) -> std::result::Result<HashMap<String, CachedLabel>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{MapAccess, Visitor};
    use std::fmt;

    struct EntryVisitor;
    impl<'de> Visitor<'de> for EntryVisitor {
        type Value = HashMap<String, CachedLabel>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a map from cache key to label string or {label, quality} object")
        }
        fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            #[derive(Deserialize)]
            #[serde(untagged)]
            enum Either {
                Legacy(String),
                Modern(CachedLabel),
            }
            let mut out: HashMap<String, CachedLabel> = HashMap::new();
            while let Some((k, v)) = access.next_entry::<String, Either>()? {
                let entry = match v {
                    Either::Legacy(label) => CachedLabel::llm(label),
                    Either::Modern(c) => c,
                };
                out.insert(k, entry);
            }
            Ok(out)
        }
    }

    deserializer.deserialize_map(EntryVisitor)
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
struct LabelCacheFile {
    #[serde(deserialize_with = "deserialize_label_map")]
    entries: HashMap<String, CachedLabel>,
}

/// Process-wide set of cluster cache keys with an in-flight LLM heal in
/// progress. A second `semantic_navigate` call that hits the same cluster
/// before the background task completes will skip enqueuing it.
static IN_FLIGHT_HEAL_KEYS: LazyLock<Mutex<HashSet<String>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Mark these cache keys as in-flight. Returns the subset that was NOT already
/// in-flight — the caller should heal exactly those keys.
pub(crate) fn claim_in_flight_keys(keys: &[String]) -> Vec<String> {
    let mut guard = IN_FLIGHT_HEAL_KEYS.lock().expect("heal lock poisoned");
    let mut claimed = Vec::with_capacity(keys.len());
    for k in keys {
        if guard.insert(k.clone()) {
            claimed.push(k.clone());
        }
    }
    claimed
}

/// Release a set of cache keys after the background heal completes (or fails).
pub(crate) fn release_in_flight_keys(keys: &[String]) {
    let mut guard = IN_FLIGHT_HEAL_KEYS.lock().expect("heal lock poisoned");
    for k in keys {
        guard.remove(k);
    }
}

/// Load cached LLM labels from disk. Returns a map of cluster_hash -> label.
/// Find the .mcp_data directory by walking up from the given path.
/// Only returns an EXISTING .mcp_data directory — never creates one.
/// Falls back to None if no .mcp_data is found in any ancestor.
fn find_mcp_data_dir(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        let candidate = dir.join(".mcp_data");
        if candidate.is_dir() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Get the .mcp_data directory for label caching.
/// Walks up to find an existing one, or creates in `start` as fallback.
fn get_label_cache_dir(start: &Path) -> PathBuf {
    find_mcp_data_dir(start).unwrap_or_else(|| {
        let fallback = start.join(".mcp_data");
        let _ = std::fs::create_dir_all(&fallback);
        fallback
    })
}

/// Load the quality-aware label cache from disk. Backward-compatible with the
/// legacy `{ "<hash>": "<label>" }` form (legacy entries are treated as
/// `quality: Llm`).
pub(crate) fn load_label_cache_full(root_dir: &Path) -> HashMap<String, CachedLabel> {
    let cache_dir = get_label_cache_dir(root_dir);
    let cache_path = cache_dir.join(LABEL_CACHE_FILE);
    if let Ok(data) = std::fs::read_to_string(&cache_path) {
        match serde_json::from_str::<LabelCacheFile>(&data) {
            Ok(file) => file.entries,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "navigate-labels.json: failed to deserialize, starting empty"
                );
                HashMap::new()
            }
        }
    } else {
        HashMap::new()
    }
}

/// Load label cache from disk as a flat `key -> label` map (label string only).
/// Used by call sites that only consume the label text. Uses the SERVER root
/// (not scoped rootDir) so cached labels are shared across all scoped navigate
/// calls.
#[allow(dead_code)] // legacy helper retained for tests/external warmup tools
pub(crate) fn load_label_cache(root_dir: &Path) -> HashMap<String, String> {
    load_label_cache_full(root_dir)
        .into_iter()
        .map(|(k, v)| (k, v.label))
        .collect()
}

/// Save the (full quality-aware) label cache to disk. Uses the workspace root's
/// `.mcp_data`.
pub(crate) fn save_label_cache_full(root_dir: &Path, cache: &HashMap<String, CachedLabel>) {
    let cache_dir = get_label_cache_dir(root_dir);
    let cache_path = cache_dir.join(LABEL_CACHE_FILE);
    let file = LabelCacheFile {
        entries: cache.clone(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&file) {
        let _ = std::fs::write(&cache_path, json);
    }
}

/// Merge-write helper: load existing cache, overlay `incoming` entries (with
/// the rule that an incoming `Llm` entry replaces a `Heuristic` one, but a
/// `Heuristic` entry never overwrites an existing `Llm` entry), and save.
///
/// This is critical for the stale-while-revalidate flow: a foreground call
/// writes a heuristic, then a background task writes the LLM upgrade. We must
/// not clobber the LLM upgrade with a parallel heuristic write for a different
/// key, and we must not regress an LLM label back to a heuristic.
pub(crate) fn save_label_cache_merged(root_dir: &Path, incoming: &HashMap<String, CachedLabel>) {
    let mut existing = load_label_cache_full(root_dir);
    for (k, new_entry) in incoming {
        match existing.get(k) {
            Some(prev)
                if prev.quality == LabelQuality::Llm
                    && new_entry.quality == LabelQuality::Heuristic =>
            {
                // Don't regress an LLM label back to heuristic.
            }
            _ => {
                existing.insert(k.clone(), new_entry.clone());
            }
        }
    }
    save_label_cache_full(root_dir, &existing);
}

/// Save the legacy `key -> label_string` map. Preserves quality of existing
/// entries on disk: if the entry was already `Llm`, stays `Llm`; if not, the
/// incoming label is recorded as `Llm` (call sites that use this API are
/// uniformly the LLM-result writers — see `label_clusters_with_cache`).
#[allow(dead_code)] // legacy helper retained for hybrid tests + external warmup tools
pub(crate) fn save_label_cache(root_dir: &Path, cache: &HashMap<String, String>) {
    let upgraded: HashMap<String, CachedLabel> = cache
        .iter()
        .map(|(k, v)| (k.clone(), CachedLabel::llm(v.clone())))
        .collect();
    save_label_cache_merged(root_dir, &upgraded);
}

/// Hash a cluster's file paths to create a stable cache key.
pub(crate) fn cluster_cache_key(file_paths: &[&str]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    let mut sorted: Vec<&str> = file_paths.to_vec();
    sorted.sort();
    for p in &sorted {
        p.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

/// Options for the semantic navigation tool.
#[derive(Debug, Clone)]
pub struct SemanticNavigateOptions {
    pub root_dir: String,
    pub max_depth: Option<usize>,
    pub max_clusters: Option<usize>,
    pub min_clusters: Option<usize>,
    /// Clustering mode:
    /// - "hybrid" (default): directory-based top-level grouping + spectral sub-clustering.
    ///   Best for CPU (fewer LLM calls, directory boundaries as domain boundaries).
    /// - "semantic": pure spectral clustering from the top, like the original contextplus.
    ///   Best with GPU (fast LLM labels at every level, true semantic grouping).
    /// - "imports": blends embedding similarity with import-graph adjacency for
    ///   structure-aware semantic clustering. Uses 70% embedding + 30% import graph.
    pub mode: Option<String>,
}

/// Information about a source file for clustering.
#[derive(Debug, Clone, Default)]
pub struct FileInfo {
    pub relative_path: String,
    pub header: String,
    pub content: String,
    pub symbol_preview: Vec<String>,
}

/// A hierarchical cluster node.
#[derive(Debug, Clone)]
pub struct ClusterNode {
    pub label: String,
    pub files: Vec<FileInfo>,
    pub children: Vec<ClusterNode>,
}

/// Perform semantic navigation: embed files, cluster, label, return tree.
///
/// Uses `embedding_cache` to avoid re-embedding files that haven't changed.
/// On warm runs (cache populated + content unchanged), this skips all Ollama embed calls.
/// New embeddings are persisted to disk via rkyv_store for cross-restart survival.
pub async fn semantic_navigate(
    options: SemanticNavigateOptions,
    ollama: &OllamaClient,
    config: &Config,
    _embedding_cache: &RwLock<HashMap<String, CacheEntry>>,
    root_dir: &Path,
) -> Result<String> {
    // max_clusters controls spectral clustering at depth 1+.
    // Depth 0 uses directory-based grouping which creates one group per
    // meaningful directory (not limited by max_clusters). This is intentional
    // because directory boundaries are the natural domain boundaries.
    let cluster_params = ClusterParams {
        max_clusters: options.max_clusters.unwrap_or(20),
        min_clusters: options.min_clusters.unwrap_or(2),
        max_depth: options.max_depth.unwrap_or(3),
    };
    let root = PathBuf::from(&options.root_dir);

    // Walk directory for source files using shared walker infrastructure
    let mut files = collect_source_files_via_walker(&root, config).await?;
    if files.is_empty() {
        return Ok("No supported source files found in the project.".to_string());
    }

    // Cap file count to keep spectral clustering tractable.
    // Sample evenly across the sorted file list to preserve directory diversity.
    // Cap is env-overridable via `CONTEXTPLUS_NAVIGATE_MAX_FILES`.
    let max_files = max_navigate_files();
    let sampled = files.len() > max_files;
    if sampled {
        let total = files.len();
        tracing::warn!(
            total_files = total,
            sampled_files = max_files,
            "semantic_navigate: file count exceeds cap, using even-spaced sampling \
             (override via CONTEXTPLUS_NAVIGATE_MAX_FILES)"
        );
        let step = total as f64 / max_files as f64;
        let sampled_files: Vec<FileInfo> = (0..max_files)
            .map(|i| {
                let idx = (i as f64 * step).floor() as usize;
                std::mem::take(&mut files[idx.min(total - 1)])
            })
            .collect();
        files = sampled_files;
    }

    // Navigate uses its OWN embedding cache (not the shared one from semantic_code_search).
    // Navigate embeds with path-weighted text ("{path} {path} {path} {header} {content}")
    // which produces different vectors than search ("{lang} {content}").
    let nav_cache_name = nav_cache_name(&config.ollama_embed_model);
    let mut nav_cache: HashMap<String, CacheEntry> = HashMap::new();
    if let Ok(Some(store)) = rkyv_store::mmap_vector_store(root_dir, &nav_cache_name) {
        let dims = store.dims();
        let flat = store.vectors_data();
        let keys = store.keys();
        let hashes = store.hashes();
        for (i, key) in keys.iter().enumerate() {
            nav_cache.insert(
                key.clone(),
                CacheEntry {
                    hash: hashes[i].clone(),
                    vector: flat[i * dims..(i + 1) * dims].to_vec(),
                },
            );
        }
    }
    let nav_cache_lock = RwLock::new(nav_cache);

    let vectors = match resolve_embeddings(
        &files,
        ollama,
        &nav_cache_lock,
        root_dir,
        &nav_cache_name,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            return Ok(format!(
                "Ollama not available for embeddings: {}\nMake sure Ollama is running with an embedding model.",
                e
            ));
        }
    };

    if files.len() <= MAX_FILES_PER_LEAF {
        // Small project: just list files with labels
        let file_labels = label_files(&files).await;
        let mut lines = vec![format!("Semantic Navigator: {} files\n", files.len())];
        for (i, file) in files.iter().enumerate() {
            let symbols = if file.symbol_preview.is_empty() {
                String::new()
            } else {
                format!(" | symbols: {}", file.symbol_preview.join(", "))
            };
            let label = file_labels
                .get(i)
                .cloned()
                .unwrap_or_else(|| file.header.clone());
            lines.push(format!("  {} - {}{}", file.relative_path, label, symbols));
        }
        return Ok(lines.join("\n"));
    }

    let use_semantic_mode = options.mode.as_deref() == Some("semantic");
    let use_imports_mode = options.mode.as_deref() == Some("imports");

    let root_node = if use_imports_mode {
        // IMPORTS MODE: Blend embedding similarity with import graph
        // for structure-aware semantic clustering.
        let all_indices: Vec<usize> = (0..files.len()).collect();

        // Step 1: Extract imports from all files
        let import_edges = extract_all_import_edges(&files, &root);

        // Step 2: Build blended affinity matrix (embedding similarity + import adjacency)
        let local_vectors: Vec<Vec<f32>> =
            all_indices.iter().map(|&i| vectors[i].clone()).collect();
        let n = files.len();
        let mc = cluster_params.max_clusters;
        let mn = cluster_params.min_clusters;
        let cluster_results = tokio::task::spawn_blocking(move || {
            let embed_affinity = build_affinity_matrix(&local_vectors);
            let raw_import_adj = build_import_adjacency(n, &import_edges);

            // Normalize import adjacency to [0,1] range to match embedding cosine similarity.
            // Raw adjacency is binary (0/1) while embeddings are continuous (0.0-1.0).
            // Also apply decay: direct imports get full weight, but we don't want
            // transitive chains (A→B→C) to collapse everything into one cluster.
            // Use 90% embedding + 10% import for gentle structural nudging.
            let blended =
                blend_affinity_matrices(&embed_affinity, &raw_import_adj, IMPORT_BLEND_ALPHA);
            // Step 3: Cluster using blended affinity
            spectral_cluster_with_affinity(blended, mc, mn)
        })
        .await
        .unwrap_or_else(|_| vec![]);

        if cluster_results.len() <= 1 {
            // Single cluster or failure — return flat list
            ClusterNode {
                label: "Project".to_string(),
                files: files.to_vec(),
                children: Vec::new(),
            }
        } else {
            // Step 4: Label clusters and build tree (reuse existing labeling infrastructure)
            let child_groups: Vec<Vec<usize>> =
                cluster_results.iter().map(|c| c.indices.clone()).collect();

            let label_input: Vec<(Vec<&FileInfo>, Option<String>)> = child_groups
                .iter()
                .map(|idxs| {
                    let refs: Vec<&FileInfo> = idxs.iter().map(|&i| &files[i]).collect();
                    (refs, None)
                })
                .collect();

            let labels = label_clusters_with_cache(&label_input, ollama, root_dir).await;

            let children: Vec<ClusterNode> = child_groups
                .iter()
                .enumerate()
                .map(|(i, idxs)| {
                    let label = labels.get(i).cloned().unwrap_or_else(|| {
                        let refs: Vec<&FileInfo> = idxs.iter().map(|&idx| &files[idx]).collect();
                        fallback_label(&refs)
                    });
                    ClusterNode {
                        label,
                        files: idxs.iter().map(|&idx| files[idx].clone()).collect(),
                        children: Vec::new(),
                    }
                })
                .collect();

            ClusterNode {
                label: "Project".to_string(),
                files: Vec::new(),
                children,
            }
        }
    } else if use_semantic_mode {
        // SEMANTIC MODE: Pure spectral clustering from the top, like the original contextplus.
        // Best with GPU (fast LLM labels at every level). Recursively splits by embedding
        // similarity — no directory-based grouping.
        let all_indices: Vec<usize> = (0..files.len()).collect();
        let mut tree = build_semantic_hierarchy(
            &files,
            &vectors,
            &all_indices,
            0,
            &cluster_params,
            ollama,
            root_dir,
        )
        .await;
        tree.label = "Project".to_string();
        tree
    } else {
        // HYBRID MODE (default): Directory-based top-level grouping + spectral sub-clustering.
        // Best for CPU (fewer LLM calls, directory boundaries as domain boundaries).
        let dir_groups = group_by_directory(&files);
        let mut children = build_labeled_tree(
            &files,
            &vectors,
            dir_groups,
            &cluster_params,
            ollama,
            root_dir,
        )
        .await;

        children.sort_by_key(|b| std::cmp::Reverse(count_files_in_node(b)));

        ClusterNode {
            label: "Project".to_string(),
            files: Vec::new(),
            children,
        }
    };

    let tree_text = render_cluster_tree(&root_node, 0);
    let sampled_note = if sampled {
        format!(" (sampled {} of total)", max_files)
    } else {
        String::new()
    };
    Ok(format!(
        "Semantic Navigator: {} files{} organized by meaning\n\n{}",
        files.len(),
        sampled_note,
        tree_text
    ))
}

/// Owned snapshot of a cluster's labelable data, suitable for moving across
/// task boundaries (cannot use `Vec<&FileInfo>` since that lifetime won't
/// outlive a `tokio::spawn`).
#[derive(Clone)]
pub(crate) struct OwnedClusterSnapshot {
    pub(crate) cache_key: String,
    pub(crate) files: Vec<FileInfo>,
    pub(crate) parent: Option<String>,
}

/// Label clusters with stale-while-revalidate semantics:
///   - Cache hit (heuristic OR llm) → return cached label.
///   - Cache miss → write a heuristic immediately, return it, queue an LLM heal.
///
/// After collecting heals, spawn a single background task that runs the LLM
/// batch and merge-writes the upgraded labels. Process-wide in-flight dedup
/// prevents a second concurrent call from re-enqueuing the same cluster.
pub(crate) async fn label_clusters_with_cache(
    label_input: &[(Vec<&FileInfo>, Option<String>)],
    ollama: &OllamaClient,
    root_dir: &Path,
) -> Vec<String> {
    let label_cache = load_label_cache_full(root_dir);
    let mut labels: Vec<String> = Vec::with_capacity(label_input.len());
    let mut cache_keys: Vec<String> = Vec::with_capacity(label_input.len());
    let mut heal_queue: Vec<OwnedClusterSnapshot> = Vec::new();
    let mut heuristic_writes: HashMap<String, CachedLabel> = HashMap::new();

    for (cluster_files, parent) in label_input.iter() {
        let paths: Vec<&str> = cluster_files
            .iter()
            .map(|f| f.relative_path.as_str())
            .collect();
        let key = cluster_cache_key(&paths);
        cache_keys.push(key.clone());

        if let Some(cached) = label_cache.get(&key) {
            labels.push(cached.label.clone());
        } else {
            let heuristic = heuristic_label(cluster_files);
            labels.push(heuristic.clone());
            heuristic_writes.insert(key.clone(), CachedLabel::heuristic(heuristic));
            heal_queue.push(OwnedClusterSnapshot {
                cache_key: key,
                files: cluster_files.iter().map(|f| (*f).clone()).collect(),
                parent: parent.clone(),
            });
        }
    }

    // Persist heuristics immediately so a crash before the LLM heal still leaves
    // a usable label on disk.
    if !heuristic_writes.is_empty() {
        save_label_cache_merged(root_dir, &heuristic_writes);
    }

    // Fire-and-forget background heal. Caller does not await this future.
    if !heal_queue.is_empty() {
        spawn_background_heal(heal_queue, ollama.clone(), root_dir.to_path_buf());
    }

    deduplicate_sibling_labels(&mut labels, label_input);
    labels
}

/// Spawn a `tokio::spawn` background task that LLM-labels the uncached
/// clusters and merge-writes upgrades to the on-disk cache. Survives the
/// originating handler's response. Process-wide dedup ensures the same
/// cache_key never has two heals in flight.
pub(crate) fn spawn_background_heal(
    queue: Vec<OwnedClusterSnapshot>,
    ollama: OllamaClient,
    root_dir: PathBuf,
) {
    let keys: Vec<String> = queue.iter().map(|s| s.cache_key.clone()).collect();
    let claimed = claim_in_flight_keys(&keys);
    if claimed.is_empty() {
        return;
    }
    let to_heal: Vec<OwnedClusterSnapshot> = queue
        .into_iter()
        .filter(|s| claimed.contains(&s.cache_key))
        .collect();

    tokio::spawn(async move {
        let result = run_llm_heal(&to_heal, &ollama, &root_dir).await;
        if let Err(e) = result {
            tracing::warn!(
                error = %e,
                count = to_heal.len(),
                "semantic_navigate: background LLM heal failed (heuristics retained)"
            );
        }
        let claimed_keys: Vec<String> = to_heal.iter().map(|s| s.cache_key.clone()).collect();
        release_in_flight_keys(&claimed_keys);
    });
}

async fn run_llm_heal(
    queue: &[OwnedClusterSnapshot],
    ollama: &OllamaClient,
    root_dir: &Path,
) -> Result<()> {
    if queue.is_empty() {
        return Ok(());
    }
    // Build the borrowed input shape that the existing batched-LLM helper expects.
    let mut input: Vec<(Vec<&FileInfo>, Option<String>)> = Vec::with_capacity(queue.len());
    for snap in queue {
        let refs: Vec<&FileInfo> = snap.files.iter().collect();
        input.push((refs, snap.parent.clone()));
    }
    let llm_labels = super::modes::semantic::label_clusters_for_semantic_mode(&input, ollama).await;

    let mut upgrades: HashMap<String, CachedLabel> = HashMap::new();
    for (i, snap) in queue.iter().enumerate() {
        if let Some(label) = llm_labels.get(i)
            && !label.is_empty()
        {
            upgrades.insert(snap.cache_key.clone(), CachedLabel::llm(label.clone()));
        }
    }
    if !upgrades.is_empty() {
        save_label_cache_merged(root_dir, &upgrades);
    }
    Ok(())
}

/// Walk the directory using shared walker infrastructure and collect source file information.
async fn collect_source_files_via_walker(root: &Path, config: &Config) -> Result<Vec<FileInfo>> {
    let allowed_extensions: HashSet<&str> = NAVIGATE_EXTENSIONS.iter().copied().collect();

    // walk_with_config is synchronous (uses the `ignore` crate), run on blocking thread
    let root_clone = root.to_path_buf();
    let config_clone = config.clone();
    let entries =
        tokio::task::spawn_blocking(move || walker::walk_with_config(&root_clone, &config_clone))
            .await
            .unwrap_or_default();

    // Filter to supported extensions and non-directories
    let max_file_size = config.max_embed_file_size as u64;
    let filtered: Vec<_> = entries
        .into_iter()
        .filter(|entry| {
            if entry.is_directory {
                return false;
            }
            let ext = entry
                .path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            allowed_extensions.contains(ext)
        })
        .collect();

    // Read files concurrently with buffer_unordered(64)
    let file_futures = filtered.into_iter().map(|entry| async move {
        let meta = tokio::fs::metadata(&entry.path).await.ok()?;
        if meta.len() > max_file_size {
            return None;
        }

        let content = if meta.len() > 2048 {
            // Large file: read only first 2KB to avoid wasting memory
            let mut f = tokio::fs::File::open(&entry.path).await.ok()?;
            let mut buf = vec![0u8; 2048];
            let n = f.read(&mut buf).await.unwrap_or(0);
            buf.truncate(n);
            String::from_utf8_lossy(&buf).to_string()
        } else {
            // Small file: read entirely
            tokio::fs::read_to_string(&entry.path).await.ok()?
        };

        let header = extract_header(&content);
        let truncated_content = if content.len() > MAX_CONTENT_CHARS {
            crate::core::parser::truncate_to_char_boundary(&content, MAX_CONTENT_CHARS).to_string()
        } else {
            content
        };

        Some(FileInfo {
            relative_path: entry.relative_path,
            header,
            content: truncated_content,
            symbol_preview: Vec::new(),
        })
    });

    let files: Vec<FileInfo> = stream::iter(file_futures)
        .buffer_unordered(64)
        .filter_map(|opt| async { opt })
        .collect()
        .await;

    Ok(files)
}

/// Resolve embedding vectors for files, using cache where possible.
///
/// Returns one Vec<f32> per file (same order as `files`).
/// Reads from `embedding_cache` first; checks content hash to detect stale entries.
/// Only calls Ollama for files not in cache or whose content has changed.
/// Stores newly embedded vectors back into the cache and persists to disk.
async fn resolve_embeddings(
    files: &[FileInfo],
    ollama: &OllamaClient,
    embedding_cache: &RwLock<HashMap<String, CacheEntry>>,
    root_dir: &Path,
    embed_model: &str,
) -> std::result::Result<Vec<Vec<f32>>, crate::error::ContextPlusError> {
    let cache_read = embedding_cache.read().await;

    // Partition: which files have valid cached vectors, which need (re-)embedding
    let mut result_vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(files.len());
    let mut uncached_indices: Vec<usize> = Vec::new();
    let mut uncached_texts: Vec<String> = Vec::new();
    let mut uncached_hashes: Vec<String> = Vec::new();

    for (i, file) in files.iter().enumerate() {
        // Hash includes version prefix (NAV_HASH_VERSION) to:
        // 1. Separate from semantic_code_search vectors (different embed format)
        // 2. Invalidate old navigate vectors (pre-path-weighting)
        // Bump the version number when the embed text format changes.
        let file_hash = nav_content_hash(&file.relative_path, &file.content);
        if let Some(entry) = cache_read.get(&file.relative_path) {
            if entry.hash == file_hash {
                // Cache hit: content unchanged
                result_vectors.push(Some(entry.vector.clone()));
            } else {
                // Cache stale: content changed, need re-embed
                result_vectors.push(None);
                uncached_indices.push(i);
                // Path-weighted embed text: repeat path 3x to boost domain signal.
                // Code embeddings capture structural patterns (imports, patterns),
                // path captures domain identity (billing vs scheduling vs IAM).
                uncached_texts.push(nav_embed_text(
                    &file.relative_path,
                    &file.header,
                    &file.content,
                ));
                uncached_hashes.push(file_hash);
            }
        } else {
            // Cache miss: never seen this file
            result_vectors.push(None);
            uncached_indices.push(i);
            uncached_texts.push(nav_embed_text(
                &file.relative_path,
                &file.header,
                &file.content,
            ));
            uncached_hashes.push(file_hash);
        }
    }
    drop(cache_read);

    // If everything was cached and fresh, skip Ollama entirely
    if uncached_indices.is_empty() {
        return Ok(result_vectors.into_iter().map(|v| v.unwrap()).collect());
    }

    // Embed uncached/stale files in chunks, saving progress after each batch.
    // This prevents losing all work if the MCP connection times out mid-run.
    // embed_model here is actually the full cache name (e.g., "navigate-nomic-embed-text")
    // passed by the caller — don't wrap it again with cache_name()
    let embed_cache_name = embed_model.to_string();
    let chunk_size = ollama.batch_size();

    for chunk_start in (0..uncached_indices.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(uncached_indices.len());
        let chunk_texts = &uncached_texts[chunk_start..chunk_end];

        let mut chunk_vectors = ollama.embed(chunk_texts).await?;

        // Store this chunk's vectors in cache, then drop lock BEFORE disk I/O.
        let store_to_save = {
            let mut cache_write = embedding_cache.write().await;
            for (local_j, &file_idx) in uncached_indices[chunk_start..chunk_end].iter().enumerate()
            {
                if local_j < chunk_vectors.len() {
                    let vec = std::mem::take(&mut chunk_vectors[local_j]);
                    cache_write.insert(
                        files[file_idx].relative_path.clone(),
                        CacheEntry {
                            hash: uncached_hashes[chunk_start + local_j].clone(),
                            vector: vec.clone(),
                        },
                    );
                    result_vectors[file_idx] = Some(vec);
                }
            }
            VectorStore::from_cache(&cache_write)
        };

        // Persist after each chunk so progress survives a timeout on the next batch.
        if let Some(store) = store_to_save
            && let Err(e) =
                rkyv_store::save_vector_store_merged(root_dir, &embed_cache_name, &store)
        {
            tracing::warn!("Failed to save embedding cache to disk: {e}");
        }
    }

    Ok(result_vectors
        .into_iter()
        .map(|v| v.unwrap_or_default())
        .collect())
}

/// Derive a human-readable label for a cluster using the best available heuristic.
/// Tries (in order): path-based label, disambiguator, file-group description.
pub(crate) fn fallback_label(files: &[&FileInfo]) -> String {
    derive_cluster_label(files)
        .or_else(|| find_label_disambiguator(files))
        .unwrap_or_else(|| describe_file_group(files))
}

/// Extract a header comment from the first few lines of a file.
pub fn extract_header(content: &str) -> String {
    let mut header_lines = Vec::new();
    for line in content.lines().take(MAX_HEADER_LINES) {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with("--") {
            let cleaned = trimmed
                .trim_start_matches("//")
                .trim_start_matches('#')
                .trim_start_matches("--")
                .trim();
            header_lines.push(cleaned.to_string());
        } else if !trimmed.is_empty() {
            break;
        }
    }
    let joined = header_lines.join(" ");
    if joined.len() > MAX_HEADER_LEN {
        crate::core::parser::truncate_to_char_boundary(&joined, MAX_HEADER_LEN).to_string()
    } else {
        joined
    }
}

/// Count total files in a cluster node (including all children recursively).
fn count_files_in_node(node: &ClusterNode) -> usize {
    if node.children.is_empty() {
        node.files.len()
    } else {
        node.children.iter().map(count_files_in_node).sum()
    }
}

/// Extract a JSON array string from LLM response text.
pub(crate) fn extract_json_array(text: &str) -> Option<String> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end >= start {
        Some(text[start..=end].to_string())
    } else {
        None
    }
}

/// Render a cluster tree as indented text.
fn render_cluster_tree(node: &ClusterNode, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    let mut result = String::new();

    if !node.label.is_empty() {
        let count = count_files_in_node(node);
        // Don't add "(N files)" if label already starts with a number (e.g., "6 source files")
        let label_starts_with_count = node
            .label
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false);
        if label_starts_with_count {
            result.push_str(&format!("{}[{}]\n", pad, node.label));
        } else {
            result.push_str(&format!("{}[{}] ({} files)\n", pad, node.label, count));
        }
    }

    if !node.children.is_empty() {
        for child in &node.children {
            result.push_str(&render_cluster_tree(child, indent + 1));
        }
    } else {
        let show_count = node.files.len().min(MAX_FILES_PER_LEAF_DISPLAY);
        for file in node.files.iter().take(show_count) {
            let label = if file.header.is_empty() {
                String::new()
            } else {
                format!(" - {}", file.header)
            };
            let symbols = if file.symbol_preview.is_empty() {
                String::new()
            } else {
                format!(" | symbols: {}", file.symbol_preview.join(", "))
            };
            result.push_str(&format!(
                "{}  {}{}{}\n",
                pad, file.relative_path, label, symbols
            ));
        }
        if node.files.len() > MAX_FILES_PER_LEAF_DISPLAY {
            result.push_str(&format!(
                "{}  (+{} more files)\n",
                pad,
                node.files.len() - MAX_FILES_PER_LEAF_DISPLAY
            ));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_header_comment_slashes() {
        let content = "// This is the main module\n// It handles routing\nfn main() {}";
        let header = extract_header(content);
        assert_eq!(header, "This is the main module It handles routing");
    }

    #[test]
    fn extract_header_hash_comments() {
        let content = "# Config loader\nimport os";
        let header = extract_header(content);
        assert_eq!(header, "Config loader");
    }

    #[test]
    fn extract_header_no_comments() {
        let content = "fn main() {\n    println!(\"hello\");\n}";
        let header = extract_header(content);
        assert_eq!(header, "");
    }

    #[test]
    fn extract_header_truncates() {
        let long_comment = format!("// {}", "x".repeat(300));
        let header = extract_header(&long_comment);
        assert_eq!(header.len(), 200);
    }

    #[test]
    fn extract_json_array_basic() {
        let text = r#"Here is the result: ["a", "b", "c"] done."#;
        let arr = extract_json_array(text);
        assert_eq!(arr, Some(r#"["a", "b", "c"]"#.to_string()));
    }

    #[test]
    fn extract_json_array_none() {
        assert_eq!(extract_json_array("no json here"), None);
    }

    #[test]
    fn render_cluster_tree_leaf() {
        let node = ClusterNode {
            label: "Test".to_string(),
            files: vec![FileInfo {
                relative_path: "src/main.rs".to_string(),
                header: "entry point".to_string(),
                content: String::new(),
                symbol_preview: vec!["main@L1".to_string()],
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(rendered.contains("[Test]"));
        assert!(rendered.contains("src/main.rs - entry point | symbols: main@L1"));
    }

    #[test]
    fn render_cluster_tree_nested() {
        let child = ClusterNode {
            label: "Child".to_string(),
            files: vec![FileInfo {
                relative_path: "a.rs".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let parent = ClusterNode {
            label: "Parent".to_string(),
            files: Vec::new(),
            children: vec![child],
        };
        let rendered = render_cluster_tree(&parent, 0);
        assert!(rendered.contains("[Parent]"));
        assert!(rendered.contains("  [Child]"));
        assert!(rendered.contains("    a.rs"));
    }

    #[test]
    fn extract_header_sql_dash_comments() {
        let content = "-- Migration script\n-- Adds users table\nCREATE TABLE users();";
        let header = extract_header(content);
        assert_eq!(header, "Migration script Adds users table");
    }

    #[test]
    fn extract_header_empty_leading_lines() {
        let content = "\n\n// After blanks\nfn main() {}";
        let header = extract_header(content);
        assert_eq!(header, "After blanks");
    }

    #[test]
    fn extract_header_empty_content() {
        let header = extract_header("");
        assert_eq!(header, "");
    }

    #[test]
    fn extract_json_array_nested_brackets() {
        let text = r#"Sure: [["a", "b"], ["c"]] done"#;
        let arr = extract_json_array(text);
        assert_eq!(arr, Some(r#"[["a", "b"], ["c"]]"#.to_string()));
    }

    #[test]
    fn extract_json_array_garbage_text() {
        assert_eq!(extract_json_array("just some random text"), None);
        assert_eq!(extract_json_array("has ] but no ["), None);
    }

    #[test]
    fn extract_json_array_only_brackets() {
        let arr = extract_json_array("[]");
        assert_eq!(arr, Some("[]".to_string()));
    }

    #[test]
    fn navigate_extensions_contains_expected() {
        assert!(NAVIGATE_EXTENSIONS.contains(&"rs"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"ts"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"tsx"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"py"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"go"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"sql"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"proto"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"json"));
        assert!(NAVIGATE_EXTENSIONS.contains(&"yaml"));
    }

    #[test]
    fn navigate_extensions_excludes_binary_formats() {
        assert!(!NAVIGATE_EXTENSIONS.contains(&"png"));
        assert!(!NAVIGATE_EXTENSIONS.contains(&"jpg"));
        assert!(!NAVIGATE_EXTENSIONS.contains(&"exe"));
        assert!(!NAVIGATE_EXTENSIONS.contains(&"wasm"));
    }

    #[tokio::test]
    async fn collect_source_files_filters_by_extension() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        tokio::fs::write(root.join("code.rs"), "fn main() {}")
            .await
            .expect("write");
        tokio::fs::write(root.join("image.png"), "fake png")
            .await
            .expect("write");
        tokio::fs::write(root.join("readme.md"), "# Hello")
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 1);
        assert!(files[0].relative_path.contains("code.rs"));
    }

    #[tokio::test]
    async fn collect_source_files_from_temp() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        // Create some source files
        let src = root.join("src");
        tokio::fs::create_dir_all(&src).await.expect("mkdir");
        tokio::fs::write(src.join("main.rs"), "// Entry point\nfn main() {}")
            .await
            .expect("write");
        tokio::fs::write(src.join("lib.rs"), "pub mod core;")
            .await
            .expect("write");
        // Create an ignored directory
        let nm = root.join("node_modules");
        tokio::fs::create_dir_all(&nm).await.expect("mkdir");
        tokio::fs::write(nm.join("pkg.js"), "module.exports = {}")
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|f| f.relative_path.contains("main.rs")));
        assert!(files.iter().any(|f| f.relative_path.contains("lib.rs")));
    }

    // --- extract_header additional tests ---

    #[test]
    fn extract_header_mixed_comment_styles() {
        // Only lines starting with //, #, or -- are collected; stops at first non-comment, non-empty line
        let content = "// Rust comment\n# Python comment\ncode_here()";
        let header = extract_header(content);
        assert_eq!(header, "Rust comment Python comment");
    }

    #[test]
    fn extract_header_stops_at_five_lines() {
        // extract_header only inspects the first 5 lines
        let content = "// line 1\n// line 2\n// line 3\n// line 4\n// line 5\n// line 6\ncode";
        let header = extract_header(content);
        assert_eq!(header, "line 1 line 2 line 3 line 4 line 5");
        assert!(!header.contains("line 6"));
    }

    #[test]
    fn extract_header_only_whitespace_lines() {
        let content = "   \n   \n   \n   \n   \n";
        let header = extract_header(content);
        // whitespace-only lines are not empty ("" after trim), so the first non-empty non-comment stops it
        assert_eq!(header, "");
    }

    #[test]
    fn extract_header_comment_with_extra_slashes() {
        // trim_start_matches("//") strips "//" but leaves the third "/"
        let content = "/// Documentation comment\nfn foo() {}";
        let header = extract_header(content);
        assert_eq!(header, "/ Documentation comment");
    }

    #[test]
    fn extract_header_multiple_hashes() {
        let content = "### Section heading\nsome code";
        let header = extract_header(content);
        assert_eq!(header, "Section heading");
    }

    #[test]
    fn extract_header_dash_comment_single_dash_not_matched() {
        // Single dash doesn't match "--" prefix, so it stops header extraction
        let content = "- list item\ncode";
        let header = extract_header(content);
        assert_eq!(header, "");
    }

    // --- extract_json_array additional tests ---

    #[test]
    fn extract_json_array_with_leading_text_and_trailing_text() {
        let text = "```json\n[{\"label\": \"Auth\"}]\n```";
        let arr = extract_json_array(text);
        assert_eq!(arr, Some("[{\"label\": \"Auth\"}]".to_string()));
    }

    #[test]
    fn extract_json_array_multiple_arrays_picks_outer() {
        // find('[') gives first bracket, rfind(']') gives last bracket
        let text = "[\"a\"] and then [\"b\"]";
        let arr = extract_json_array(text);
        // Should span from first [ to last ]
        assert_eq!(arr, Some("[\"a\"] and then [\"b\"]".to_string()));
    }

    #[test]
    fn extract_json_array_bracket_only_close_before_open() {
        // ] appears before [ in the string; find('[') returns position after rfind(']')
        let text = "] some text [";
        let arr = extract_json_array(text);
        // find('[') = 12, rfind(']') = 0 => end < start => None
        assert_eq!(arr, None);
    }

    #[test]
    fn extract_json_array_objects_array() {
        let text =
            r#"[{"overarchingTheme":"auth","distinguishingFeature":"login","label":"User Auth"}]"#;
        let arr = extract_json_array(text);
        assert_eq!(arr, Some(text.to_string()));
    }

    // --- render_cluster_tree additional tests ---

    #[test]
    fn render_cluster_tree_empty_label_no_bracket_line() {
        let node = ClusterNode {
            label: String::new(),
            files: vec![FileInfo {
                relative_path: "foo.rs".to_string(),
                header: "a file".to_string(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        // No label line should appear
        assert!(!rendered.contains('['));
        assert!(rendered.contains("foo.rs - a file"));
    }

    #[test]
    fn render_cluster_tree_file_no_header_no_symbols() {
        let node = ClusterNode {
            label: "Leaf".to_string(),
            files: vec![FileInfo {
                relative_path: "bare.ts".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(rendered.contains("[Leaf]"));
        // No " - " label and no " | symbols:" suffix
        assert!(rendered.contains("  bare.ts\n"));
        assert!(!rendered.contains(" - "));
        assert!(!rendered.contains("symbols:"));
    }

    #[test]
    fn render_cluster_tree_multiple_files_in_leaf() {
        let node = ClusterNode {
            label: "Group".to_string(),
            files: vec![
                FileInfo {
                    relative_path: "a.rs".to_string(),
                    header: "first".to_string(),
                    content: String::new(),
                    symbol_preview: Vec::new(),
                },
                FileInfo {
                    relative_path: "b.rs".to_string(),
                    header: "second".to_string(),
                    content: String::new(),
                    symbol_preview: vec!["fn_b@L1".to_string()],
                },
                FileInfo {
                    relative_path: "c.rs".to_string(),
                    header: String::new(),
                    content: String::new(),
                    symbol_preview: vec!["fn_c@L1".to_string(), "fn_d@L5".to_string()],
                },
            ],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(rendered.contains("a.rs - first\n"));
        assert!(rendered.contains("b.rs - second | symbols: fn_b@L1\n"));
        assert!(rendered.contains("c.rs | symbols: fn_c@L1, fn_d@L5\n"));
    }

    #[test]
    fn render_cluster_tree_with_initial_indent() {
        let node = ClusterNode {
            label: "Deep".to_string(),
            files: vec![FileInfo {
                relative_path: "x.py".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 3);
        // 3 levels of indent = 6 spaces before [Deep]
        assert!(rendered.starts_with("      [Deep]"));
        // File should be at indent 3 + 2 more spaces = 8 spaces
        assert!(rendered.contains("        x.py"));
    }

    #[test]
    fn render_cluster_tree_deeply_nested_three_levels() {
        let grandchild = ClusterNode {
            label: "Grandchild".to_string(),
            files: vec![FileInfo {
                relative_path: "gc.rs".to_string(),
                header: "deep file".to_string(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let child = ClusterNode {
            label: "Child".to_string(),
            files: Vec::new(),
            children: vec![grandchild],
        };
        let root = ClusterNode {
            label: "Root".to_string(),
            files: Vec::new(),
            children: vec![child],
        };
        let rendered = render_cluster_tree(&root, 0);
        assert!(rendered.contains("[Root]"));
        assert!(rendered.contains("  [Child]"));
        assert!(rendered.contains("    [Grandchild]"));
        assert!(rendered.contains("      gc.rs - deep file"));
    }

    #[test]
    fn render_cluster_tree_multiple_children() {
        let child_a = ClusterNode {
            label: "A".to_string(),
            files: vec![FileInfo {
                relative_path: "a.rs".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let child_b = ClusterNode {
            label: "B".to_string(),
            files: vec![FileInfo {
                relative_path: "b.rs".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let parent = ClusterNode {
            label: "Parent".to_string(),
            files: Vec::new(),
            children: vec![child_a, child_b],
        };
        let rendered = render_cluster_tree(&parent, 0);
        assert!(rendered.contains("[Parent]"));
        assert!(rendered.contains("  [A]"));
        assert!(rendered.contains("    a.rs"));
        assert!(rendered.contains("  [B]"));
        assert!(rendered.contains("    b.rs"));
    }

    #[test]
    fn render_cluster_tree_empty_node() {
        // Node with no label, no files, no children produces empty string
        let node = ClusterNode {
            label: String::new(),
            files: Vec::new(),
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert_eq!(rendered, "");
    }

    #[test]
    fn render_cluster_tree_children_take_priority_over_files() {
        // When a node has both children and files, only children are rendered
        let child = ClusterNode {
            label: "Child".to_string(),
            files: vec![FileInfo {
                relative_path: "child.rs".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: Vec::new(),
        };
        let parent = ClusterNode {
            label: "Parent".to_string(),
            files: vec![FileInfo {
                relative_path: "parent_file.rs".to_string(),
                header: "should not appear".to_string(),
                content: String::new(),
                symbol_preview: Vec::new(),
            }],
            children: vec![child],
        };
        let rendered = render_cluster_tree(&parent, 0);
        // The parent's files are not rendered because children exist
        assert!(!rendered.contains("parent_file.rs"));
        assert!(rendered.contains("child.rs"));
    }

    // --- collect_source_files_via_walker additional tests ---

    #[tokio::test]
    async fn collect_source_files_truncates_long_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        let long_content = "a".repeat(1000);
        tokio::fs::write(root.join("long.rs"), &long_content)
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 1);
        // Content should be truncated to ~500 bytes
        assert!(
            files[0].content.len() <= 500,
            "content len {} should be <= 500",
            files[0].content.len()
        );
    }

    #[tokio::test]
    async fn collect_source_files_short_content_not_truncated() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        let short_content = "fn main() {}";
        tokio::fs::write(root.join("short.rs"), short_content)
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].content, short_content);
    }

    #[tokio::test]
    async fn collect_source_files_extracts_header() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        tokio::fs::write(
            root.join("documented.py"),
            "# Data processor\nimport pandas",
        )
        .await
        .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].header, "Data processor");
    }

    #[tokio::test]
    async fn collect_source_files_multiple_extensions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        tokio::fs::write(root.join("main.rs"), "fn main() {}")
            .await
            .expect("write");
        tokio::fs::write(root.join("app.ts"), "console.log('hello')")
            .await
            .expect("write");
        tokio::fs::write(root.join("script.py"), "print('hi')")
            .await
            .expect("write");
        tokio::fs::write(root.join("config.yaml"), "key: value")
            .await
            .expect("write");
        tokio::fs::write(root.join("query.sql"), "SELECT 1")
            .await
            .expect("write");
        tokio::fs::write(root.join("schema.proto"), "syntax = \"proto3\";")
            .await
            .expect("write");
        // Should be excluded:
        tokio::fs::write(root.join("styles.css"), "body {}")
            .await
            .expect("write");
        tokio::fs::write(root.join("page.html"), "<html></html>")
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        // 6 accepted extensions, 2 excluded
        assert_eq!(files.len(), 6);
        let paths: Vec<&str> = files.iter().map(|f| f.relative_path.as_str()).collect();
        assert!(!paths.iter().any(|p| p.contains(".css")));
        assert!(!paths.iter().any(|p| p.contains(".html")));
    }

    #[tokio::test]
    async fn collect_source_files_empty_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert!(files.is_empty());
    }

    #[tokio::test]
    async fn collect_source_files_nested_directories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        let deep = root.join("a").join("b").join("c");
        tokio::fs::create_dir_all(&deep).await.expect("mkdir");
        tokio::fs::write(deep.join("deep.go"), "package main")
            .await
            .expect("write");
        tokio::fs::write(root.join("top.js"), "export default {}")
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|f| f.relative_path.contains("deep.go")));
        assert!(files.iter().any(|f| f.relative_path.contains("top.js")));
    }

    #[tokio::test]
    async fn collect_source_files_symbol_preview_is_empty() {
        // symbol_preview is always empty in collect_source_files_via_walker
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();

        tokio::fs::write(root.join("mod.rs"), "pub fn foo() {}\npub fn bar() {}")
            .await
            .expect("write");

        let config = Config::from_env();
        let files = collect_source_files_via_walker(&root, &config)
            .await
            .expect("collect");
        assert_eq!(files.len(), 1);
        assert!(files[0].symbol_preview.is_empty());
    }

    // --- SemanticNavigateOptions tests ---

    #[test]
    fn semantic_navigate_options_defaults() {
        let opts = SemanticNavigateOptions {
            root_dir: "/tmp".to_string(),
            max_depth: None,
            max_clusters: None,
            min_clusters: None,
            mode: None,
        };
        assert_eq!(opts.root_dir, "/tmp");
        assert!(opts.max_depth.is_none());
        assert!(opts.max_clusters.is_none());
        assert!(opts.min_clusters.is_none());
        assert!(opts.mode.is_none());
    }

    #[test]
    fn semantic_navigate_options_with_values() {
        let opts = SemanticNavigateOptions {
            root_dir: "/project".to_string(),
            max_depth: Some(5),
            max_clusters: Some(10),
            min_clusters: Some(3),
            mode: Some("semantic".to_string()),
        };
        assert_eq!(opts.max_depth, Some(5));
        assert_eq!(opts.max_clusters, Some(10));
        assert_eq!(opts.min_clusters, Some(3));
    }

    // --- MAX_FILES_PER_LEAF constant test ---

    #[test]
    fn max_files_per_leaf_is_10() {
        assert_eq!(MAX_FILES_PER_LEAF, 10);
    }

    // --- NAVIGATE_EXTENSIONS completeness tests ---

    #[test]
    fn navigate_extensions_covers_all_listed() {
        let expected = vec![
            "rs", "ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "go", "java", "c", "cpp", "h",
            "hpp", "cc", "rb", "sh", "bash", "zsh", "sql", "graphql", "proto", "yaml", "yml",
            "toml", "json",
        ];
        for ext in &expected {
            assert!(
                NAVIGATE_EXTENSIONS.contains(ext),
                "expected extension '{}' not in NAVIGATE_EXTENSIONS",
                ext
            );
        }
        assert_eq!(NAVIGATE_EXTENSIONS.len(), expected.len());
    }

    // --- extract_header boundary / edge case tests ---

    #[test]
    fn extract_header_exactly_200_bytes() {
        // Create a comment that is exactly 200 chars after stripping the "//"
        let comment_body = "a".repeat(200);
        let content = format!("// {}", comment_body);
        let header = extract_header(&content);
        // 200 chars exactly => should be kept as-is (len <= 200)
        assert_eq!(header.len(), 200);
    }

    #[test]
    fn extract_header_201_bytes_truncated() {
        let comment_body = "b".repeat(201);
        let content = format!("// {}", comment_body);
        let header = extract_header(&content);
        assert_eq!(header.len(), 200);
    }

    #[test]
    fn extract_header_blank_comment_lines() {
        // Comment markers with no text after them
        let content = "//\n//\ncode()";
        let header = extract_header(content);
        // Each line after stripping "//" and trimming gives "", joined with " " => " "
        assert_eq!(header, " ");
    }

    #[test]
    fn extract_header_non_comment_first_line() {
        let content = "use std::io;\n// This is a comment";
        let header = extract_header(content);
        // First non-empty line is not a comment, so extraction stops immediately
        assert_eq!(header, "");
    }

    // --- render_cluster_tree additional edge cases ---

    #[test]
    fn render_cluster_tree_file_with_symbols_no_header() {
        let node = ClusterNode {
            label: "Syms".to_string(),
            files: vec![FileInfo {
                relative_path: "lib.rs".to_string(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec!["Foo@L1".to_string(), "Bar@L10".to_string()],
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        // Should have symbols but no header dash
        assert!(rendered.contains("lib.rs | symbols: Foo@L1, Bar@L10"));
        assert!(!rendered.contains(" - "));
    }

    #[test]
    fn render_cluster_tree_file_with_header_and_symbols() {
        let node = ClusterNode {
            label: "Full".to_string(),
            files: vec![FileInfo {
                relative_path: "main.go".to_string(),
                header: "entry".to_string(),
                content: String::new(),
                symbol_preview: vec!["main@L1".to_string()],
            }],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(rendered.contains("main.go - entry | symbols: main@L1"));
    }

    // --- extract_json_array edge cases ---

    #[test]
    fn extract_json_array_with_newlines() {
        let text = "Here:\n[\n  \"a\",\n  \"b\"\n]\nDone.";
        let arr = extract_json_array(text);
        assert!(arr.is_some());
        let val = arr.unwrap();
        assert!(val.starts_with('['));
        assert!(val.ends_with(']'));
        assert!(val.contains("\"a\""));
    }

    #[test]
    fn extract_json_array_empty_string() {
        assert_eq!(extract_json_array(""), None);
    }

    #[test]
    fn extract_json_array_only_open_bracket() {
        // Has '[' but no ']'
        assert_eq!(extract_json_array("["), None);
    }

    #[test]
    fn extract_json_array_only_close_bracket() {
        // Has ']' but no '['
        assert_eq!(extract_json_array("]"), None);
    }

    // ── resolve_embeddings cache tests ────────────────────────────────

    fn make_test_file(path: &str) -> FileInfo {
        FileInfo {
            relative_path: path.to_string(),
            header: format!("header for {path}"),
            content: format!("content of {path}"),
            symbol_preview: vec![],
        }
    }

    /// Build a CacheEntry with the correct content hash for a test file.
    fn make_cache_entry(path: &str, vector: Vec<f32>) -> CacheEntry {
        let file = make_test_file(path);
        CacheEntry {
            hash: nav_content_hash(&file.relative_path, &file.content),
            vector,
        }
    }

    #[tokio::test]
    async fn resolve_embeddings_all_cached_skips_ollama() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Pre-populate cache with known vectors (matching content hashes)
        let cache: RwLock<HashMap<String, CacheEntry>> = RwLock::new(HashMap::new());
        {
            let mut w = cache.write().await;
            w.insert(
                "src/a.rs".to_string(),
                make_cache_entry("src/a.rs", vec![1.0, 2.0, 3.0]),
            );
            w.insert(
                "src/b.rs".to_string(),
                make_cache_entry("src/b.rs", vec![4.0, 5.0, 6.0]),
            );
        }

        let files = vec![make_test_file("src/a.rs"), make_test_file("src/b.rs")];

        // OllamaClient pointing to a non-existent server — if it tries to call
        // Ollama it will fail. Success proves cache was used.
        let config = crate::config::Config::from_env();
        let mut bad_config = config.clone();
        bad_config.ollama_host = "http://127.0.0.1:1".to_string(); // unreachable
        let ollama = OllamaClient::new(&bad_config);

        let result = resolve_embeddings(&files, &ollama, &cache, tmp.path(), "test-model").await;
        assert!(result.is_ok(), "should succeed from cache without Ollama");
        let vecs = result.unwrap();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(vecs[1], vec![4.0, 5.0, 6.0]);
    }

    #[tokio::test]
    async fn resolve_embeddings_partial_cache_only_embeds_missing() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let tmp = tempfile::TempDir::new().unwrap();
        let mock_server = MockServer::start().await;

        // Return a single 3-dim vector for each embed call
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [[7.0, 8.0, 9.0]]
            })))
            .expect(1) // Should only be called ONCE (for the uncached file)
            .mount(&mock_server)
            .await;

        let cache: RwLock<HashMap<String, CacheEntry>> = RwLock::new(HashMap::new());
        {
            let mut w = cache.write().await;
            w.insert(
                "src/cached.rs".to_string(),
                make_cache_entry("src/cached.rs", vec![1.0, 2.0, 3.0]),
            );
        }

        let files = vec![
            make_test_file("src/cached.rs"),
            make_test_file("src/uncached.rs"),
        ];

        let mut config = crate::config::Config::from_env();
        config.ollama_host = mock_server.uri();
        let ollama = OllamaClient::new(&config);

        let result = resolve_embeddings(&files, &ollama, &cache, tmp.path(), "test-model").await;
        assert!(result.is_ok());
        let vecs = result.unwrap();
        assert_eq!(vecs[0], vec![1.0, 2.0, 3.0]); // from cache
        assert_eq!(vecs[1], vec![7.0, 8.0, 9.0]); // from Ollama

        // Verify the newly embedded vector was stored in cache with correct hash
        let r = cache.read().await;
        let entry = r.get("src/uncached.rs").unwrap();
        assert_eq!(entry.vector, vec![7.0, 8.0, 9.0]);
        assert_eq!(
            entry.hash,
            nav_content_hash("src/uncached.rs", "content of src/uncached.rs")
        );
    }

    #[tokio::test]
    async fn resolve_embeddings_stores_all_new_vectors_in_cache() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let tmp = tempfile::TempDir::new().unwrap();
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [[1.0, 0.0], [0.0, 1.0]]
            })))
            .mount(&mock_server)
            .await;

        let cache: RwLock<HashMap<String, CacheEntry>> = RwLock::new(HashMap::new());
        let files = vec![make_test_file("a.ts"), make_test_file("b.ts")];

        let mut config = crate::config::Config::from_env();
        config.ollama_host = mock_server.uri();
        let ollama = OllamaClient::new(&config);

        let result = resolve_embeddings(&files, &ollama, &cache, tmp.path(), "test-model")
            .await
            .unwrap();
        assert_eq!(result.len(), 2);

        // Both should now be in cache
        let r = cache.read().await;
        assert_eq!(r.len(), 2);
        assert!(r.contains_key("a.ts"));
        assert!(r.contains_key("b.ts"));
    }

    #[tokio::test]
    async fn resolve_embeddings_empty_files_returns_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache: RwLock<HashMap<String, CacheEntry>> = RwLock::new(HashMap::new());
        let files: Vec<FileInfo> = vec![];

        let config = crate::config::Config::from_env();
        let ollama = OllamaClient::new(&config);

        let result = resolve_embeddings(&files, &ollama, &cache, tmp.path(), "test-model")
            .await
            .unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn resolve_embeddings_stale_hash_triggers_reembed() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let tmp = tempfile::TempDir::new().unwrap();
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [[9.0, 9.0, 9.0]]
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        // Cache entry has a stale hash (doesn't match current file content)
        let cache: RwLock<HashMap<String, CacheEntry>> = RwLock::new(HashMap::new());
        {
            let mut w = cache.write().await;
            w.insert(
                "src/changed.rs".to_string(),
                CacheEntry {
                    hash: "stale_hash_that_wont_match".to_string(),
                    vector: vec![1.0, 1.0, 1.0],
                },
            );
        }

        let files = vec![make_test_file("src/changed.rs")];

        let mut config = crate::config::Config::from_env();
        config.ollama_host = mock_server.uri();
        let ollama = OllamaClient::new(&config);

        let result = resolve_embeddings(&files, &ollama, &cache, tmp.path(), "test-model")
            .await
            .unwrap();
        // Should have re-embedded with new vector from Ollama
        assert_eq!(result[0], vec![9.0, 9.0, 9.0]);

        // Cache should now have the new vector and correct hash
        let r = cache.read().await;
        let entry = r.get("src/changed.rs").unwrap();
        assert_eq!(entry.vector, vec![9.0, 9.0, 9.0]);
        assert_eq!(
            entry.hash,
            nav_content_hash("src/changed.rs", "content of src/changed.rs")
        );
    }

    // ── group_by_directory tests ─────────────────────────────────────

    #[test]
    fn group_by_directory_monorepo_domains() {
        let mut files = Vec::new();
        // Need >= MIN_DIR_GROUP_SIZE (5) files per group to avoid merging into "other"
        for i in 0..16 {
            files.push(make_test_file(&format!(
                "packages/domains/billing/file{}.ts",
                i
            )));
        }
        for i in 0..16 {
            files.push(make_test_file(&format!(
                "packages/domains/scheduling/file{}.ts",
                i
            )));
        }
        let groups = group_by_directory(&files);
        let labels: Vec<&str> = groups.iter().map(|g| g.0.as_str()).collect();
        assert!(
            labels.contains(&"domains/billing"),
            "Expected 'domains/billing', got {:?}",
            labels
        );
        assert!(
            labels.contains(&"domains/scheduling"),
            "Expected 'domains/scheduling', got {:?}",
            labels
        );
    }

    #[test]
    fn group_by_directory_root_files() {
        let files = vec![make_test_file("main.rs"), make_test_file("Cargo.toml")];
        let groups = group_by_directory(&files);
        // Both are single-component paths → "root" label, but < MIN_DIR_GROUP_SIZE (5) → merged to "other"
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "other");
        assert_eq!(groups[0].1.len(), 2);
    }

    #[test]
    fn group_by_directory_empty() {
        let groups = group_by_directory(&[]);
        assert!(groups.is_empty());
    }

    #[test]
    fn group_by_directory_merges_tiny_groups() {
        // Create two groups each with < MIN_DIR_GROUP_SIZE files
        let mut files = Vec::new();
        for i in 0..3 {
            files.push(make_test_file(&format!("alpha/deep/file{}.ts", i)));
        }
        for i in 0..3 {
            files.push(make_test_file(&format!("beta/deep/file{}.ts", i)));
        }
        let groups = group_by_directory(&files);
        // Both groups have 3 files (< 5), so all should merge into one bucket
        // Label is derived from top-level subdirectories instead of bare "other"
        assert_eq!(groups.len(), 1);
        assert!(
            groups[0].0 == "alpha + beta" || groups[0].0 == "beta + alpha",
            "Expected descriptive label for other bucket, got '{}'",
            groups[0].0
        );
        assert_eq!(groups[0].1.len(), 6);
    }

    // ── count_files_in_node tests ────────────────────────────────────

    #[test]
    fn count_files_leaf_node() {
        let node = ClusterNode {
            label: "test".to_string(),
            files: vec![make_test_file("a.rs"), make_test_file("b.rs")],
            children: Vec::new(),
        };
        assert_eq!(count_files_in_node(&node), 2);
    }

    #[test]
    fn count_files_nested_children() {
        let child1 = ClusterNode {
            label: "c1".to_string(),
            files: vec![
                make_test_file("a.rs"),
                make_test_file("b.rs"),
                make_test_file("c.rs"),
            ],
            children: Vec::new(),
        };
        let child2 = ClusterNode {
            label: "c2".to_string(),
            files: vec![make_test_file("d.rs"), make_test_file("e.rs")],
            children: Vec::new(),
        };
        let parent = ClusterNode {
            label: "parent".to_string(),
            files: Vec::new(),
            children: vec![child1, child2],
        };
        assert_eq!(count_files_in_node(&parent), 5);
    }

    #[test]
    fn render_cluster_tree_truncates_large_leaves() {
        let files: Vec<FileInfo> = (0..15)
            .map(|i| make_test_file(&format!("src/file{}.rs", i)))
            .collect();
        let node = ClusterNode {
            label: "big leaf".to_string(),
            files,
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(rendered.contains("[big leaf]"));
        assert!(rendered.contains("(+5 more files)"));
        // Should show exactly MAX_FILES_PER_LEAF_DISPLAY file lines
        let file_lines = rendered.lines().filter(|l| l.contains("src/file")).count();
        assert_eq!(file_lines, 10); // MAX_FILES_PER_LEAF_DISPLAY
    }

    #[test]
    fn render_cluster_tree_shows_file_count() {
        let node = ClusterNode {
            label: "test".to_string(),
            files: vec![make_test_file("a.rs"), make_test_file("b.rs")],
            children: Vec::new(),
        };
        let rendered = render_cluster_tree(&node, 0);
        assert!(rendered.contains("[test] (2 files)"), "got: {}", rendered);
    }

    // --- group_by_directory tests ---

    #[test]
    fn group_by_directory_skips_apps_prefix() {
        // "apps/emr-api/src/app.ts" should group as "emr-api", not "apps"
        let mut files = Vec::new();
        for i in 0..20 {
            files.push(FileInfo {
                relative_path: format!("apps/emr-api/src/file{}.ts", i),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            });
        }
        let groups = group_by_directory(&files);
        let labels: Vec<&str> = groups.iter().map(|(l, _)| l.as_str()).collect();
        assert!(
            !labels.contains(&"apps"),
            "Should skip 'apps' prefix, got {:?}",
            labels
        );
    }

    // ── Stale-while-revalidate label cache schema tests ──────────────────

    /// Existing `navigate-labels.json` files in the wild were `{ "<hash>": "<label>" }`.
    /// They MUST keep loading after the schema upgrade.
    #[test]
    fn cache_load_accepts_legacy_string_format() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let mcp_data = tempdir.path().join(".mcp_data");
        std::fs::create_dir_all(&mcp_data).expect("mkdir");
        let cache_path = mcp_data.join(LABEL_CACHE_FILE);
        std::fs::write(
            &cache_path,
            r#"{"abc123": "old-llm-label", "def456": "another"}"#,
        )
        .expect("write legacy cache");

        let cache = load_label_cache_full(tempdir.path());
        assert_eq!(cache.len(), 2);
        let entry = cache.get("abc123").expect("abc123 present");
        assert_eq!(entry.label, "old-llm-label");
        assert_eq!(entry.quality, LabelQuality::Llm);
    }

    /// New schema: `{ "<hash>": { "label": "...", "quality": "heuristic" | "llm" } }`.
    #[test]
    fn cache_load_accepts_quality_format() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let mcp_data = tempdir.path().join(".mcp_data");
        std::fs::create_dir_all(&mcp_data).expect("mkdir");
        let cache_path = mcp_data.join(LABEL_CACHE_FILE);
        std::fs::write(
            &cache_path,
            r#"{
                "h1": {"label": "fast-label", "quality": "heuristic"},
                "h2": {"label": "rich-label", "quality": "llm"}
            }"#,
        )
        .expect("write modern cache");

        let cache = load_label_cache_full(tempdir.path());
        assert_eq!(cache.len(), 2);
        assert_eq!(cache["h1"].label, "fast-label");
        assert_eq!(cache["h1"].quality, LabelQuality::Heuristic);
        assert_eq!(cache["h2"].label, "rich-label");
        assert_eq!(cache["h2"].quality, LabelQuality::Llm);
    }

    /// A cache with mixed legacy + modern entries must round-trip through one
    /// load + save cycle, ending up uniformly in modern form on disk.
    #[test]
    fn cache_load_save_roundtrip_normalizes_mixed_schema() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let mcp_data = tempdir.path().join(".mcp_data");
        std::fs::create_dir_all(&mcp_data).expect("mkdir");
        let cache_path = mcp_data.join(LABEL_CACHE_FILE);
        std::fs::write(
            &cache_path,
            r#"{
                "legacy_key": "old-style",
                "modern_key": {"label": "new-style", "quality": "heuristic"}
            }"#,
        )
        .expect("write mixed cache");

        let loaded = load_label_cache_full(tempdir.path());
        save_label_cache_full(tempdir.path(), &loaded);

        let on_disk = std::fs::read_to_string(&cache_path).expect("read");
        // Must now be valid modern JSON (not raw strings) for both keys.
        let parsed: HashMap<String, CachedLabel> = serde_json::from_str::<LabelCacheFile>(&on_disk)
            .expect("modern parse")
            .entries;
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed["legacy_key"].quality, LabelQuality::Llm);
        assert_eq!(parsed["modern_key"].quality, LabelQuality::Heuristic);
    }

    /// Heuristic-then-LLM merge must end on disk with `quality=Llm` and the
    /// LLM-supplied label, even when the heuristic save came first.
    #[test]
    fn heuristic_then_llm_overwrite_persists() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let key = "k1".to_string();

        let mut h_writes: HashMap<String, CachedLabel> = HashMap::new();
        h_writes.insert(key.clone(), CachedLabel::heuristic("auth/* ".to_string()));
        save_label_cache_merged(tempdir.path(), &h_writes);

        // Verify heuristic landed.
        let after_h = load_label_cache_full(tempdir.path());
        assert_eq!(after_h[&key].quality, LabelQuality::Heuristic);
        assert_eq!(after_h[&key].label, "auth/* ");

        // Now upgrade.
        let mut l_writes: HashMap<String, CachedLabel> = HashMap::new();
        l_writes.insert(
            key.clone(),
            CachedLabel::llm("Authentication Flow".to_string()),
        );
        save_label_cache_merged(tempdir.path(), &l_writes);

        let after_l = load_label_cache_full(tempdir.path());
        assert_eq!(after_l[&key].quality, LabelQuality::Llm);
        assert_eq!(after_l[&key].label, "Authentication Flow");
    }

    /// A heuristic write MUST NOT regress an existing LLM label back down.
    /// This protects against e.g. a freshly-evicted cluster being relabeled
    /// while a stale background heal also lands.
    #[test]
    fn heuristic_does_not_regress_existing_llm_entry() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let key = "k1".to_string();

        let mut llm: HashMap<String, CachedLabel> = HashMap::new();
        llm.insert(key.clone(), CachedLabel::llm("Good Label".to_string()));
        save_label_cache_merged(tempdir.path(), &llm);

        let mut heur: HashMap<String, CachedLabel> = HashMap::new();
        heur.insert(key.clone(), CachedLabel::heuristic("auth/* ".to_string()));
        save_label_cache_merged(tempdir.path(), &heur);

        let after = load_label_cache_full(tempdir.path());
        assert_eq!(after[&key].quality, LabelQuality::Llm);
        assert_eq!(after[&key].label, "Good Label");
    }

    /// In-flight dedup: claiming the same key twice gives it once.
    #[test]
    fn claim_in_flight_keys_dedups_within_one_call() {
        let k = format!("dedup-test-{}", uuid_like_suffix());
        let claimed1 = claim_in_flight_keys(&[k.clone(), k.clone()]);
        assert_eq!(claimed1.len(), 1);
        // Second claim while the first is still in flight returns nothing.
        let claimed2 = claim_in_flight_keys(std::slice::from_ref(&k));
        assert!(claimed2.is_empty());
        // After release, claim succeeds again.
        release_in_flight_keys(std::slice::from_ref(&k));
        let claimed3 = claim_in_flight_keys(std::slice::from_ref(&k));
        assert_eq!(claimed3, vec![k.clone()]);
        release_in_flight_keys(&[k]);
    }

    fn uuid_like_suffix() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("{:x}", nanos)
    }

    /// Loading a real-world legacy `navigate-labels.json` (captured from
    /// /workspace/.mcp_data) must succeed without error and treat every
    /// entry as `quality=Llm`.
    #[test]
    fn cache_load_accepts_real_world_legacy_fixture() {
        let fixture = include_str!("../../tests/fixtures/legacy-navigate-labels.json");
        let parsed: LabelCacheFile =
            serde_json::from_str(fixture).expect("legacy fixture must parse");
        assert!(!parsed.entries.is_empty(), "fixture should have entries");
        for entry in parsed.entries.values() {
            assert_eq!(
                entry.quality,
                LabelQuality::Llm,
                "legacy entries must deserialize as Llm quality"
            );
        }
    }

    /// Stale-while-revalidate end-to-end: first call returns heuristic
    /// labels and persists them with `quality=Heuristic`; the background
    /// task then upgrades to `quality=Llm` once the stub LLM responds.
    #[tokio::test]
    async fn label_clusters_with_cache_stale_while_revalidate() {
        use crate::config::Config;
        use crate::core::embeddings::OllamaClient;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let tempdir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(tempdir.path().join(".mcp_data")).unwrap();

        // Stub Ollama: the chat endpoint returns deterministic labels.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": {
                    "content": r#"["Authentication Services"]"#
                }
            })))
            .mount(&server)
            .await;

        let mut config = Config::from_env();
        config.ollama_host = server.uri();
        config.ollama_chat_model = "test-model".to_string();
        let client = OllamaClient::new(&config);

        let f1 = FileInfo {
            relative_path: "src/auth/login.rs".to_string(),
            header: "login flow".to_string(),
            ..Default::default()
        };
        let f2 = FileInfo {
            relative_path: "src/auth/session.rs".to_string(),
            header: "session lifecycle".to_string(),
            ..Default::default()
        };
        let cluster: Vec<&FileInfo> = vec![&f1, &f2];
        let input: Vec<(Vec<&FileInfo>, Option<String>)> = vec![(cluster, None)];

        // First call: should return heuristic immediately.
        let labels = label_clusters_with_cache(&input, &client, tempdir.path()).await;
        assert_eq!(labels.len(), 1);

        // On-disk cache should now hold a Heuristic entry for this cluster.
        let key = cluster_cache_key(&["src/auth/login.rs", "src/auth/session.rs"]);
        let after_first = load_label_cache_full(tempdir.path());
        let entry = after_first
            .get(&key)
            .expect("heuristic entry persisted before background heal");
        assert_eq!(entry.quality, LabelQuality::Heuristic);
        assert_eq!(labels[0], entry.label);

        // Wait for the background heal to land. Poll up to ~5 seconds.
        let mut upgraded = false;
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let cache = load_label_cache_full(tempdir.path());
            if let Some(e) = cache.get(&key)
                && e.quality == LabelQuality::Llm
            {
                assert_eq!(e.label, "Authentication Services");
                upgraded = true;
                break;
            }
        }
        assert!(upgraded, "background heal did not upgrade label to Llm");
    }
}
