use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fd_lock::RwLock as FdRwLock;
use rkyv::{Archive, Deserialize, Serialize};

use crate::core::embeddings::{CacheEntry, VectorStore};
use crate::error::{ContextPlusError, Result};

// Compile-time assertion: this crate only runs correctly on little-endian platforms.
// rkyv serializes f32 as f32_le; on big-endian the zero-copy pointer cast would produce
// garbage values. (x86_64, aarch64, and RISC-V LE are all little-endian.)
const _: () = assert!(
    cfg!(target_endian = "little"),
    "contextplus-rs rkyv cache requires a little-endian platform"
);

// ---------------------------------------------------------------------------
// Cache directory
// ---------------------------------------------------------------------------

const CACHE_DIR: &str = ".mcp_data";
const WORKTREE_SUBDIR: &str = "worktrees";
const CACHE_VERSION: u8 = 1;
/// Header size padded to 16-byte alignment so rkyv data starts aligned.
const HEADER_SIZE: usize = 16;

/// Detect whether `root_dir` is a git worktree (as opposed to a main checkout
/// or non-git directory). Returns the worktree name if so.
///
/// Git worktrees are identified by `.git` being a *file* (not a directory)
/// containing `gitdir: /path/to/main/.git/worktrees/<name>`. The main checkout
/// has `.git` as a directory; non-git directories have neither.
///
/// Without this distinction, two checkouts of the same repo (main + worktree)
/// would silently share `.mcp_data/*.rkyv` — embeddings keyed by relative paths
/// would collide and corrupt each other (see memory: worktree_corrupts_git_config).
fn worktree_name(root_dir: &Path) -> Option<String> {
    let git_path = root_dir.join(".git");
    let metadata = fs::metadata(&git_path).ok()?;
    if metadata.is_dir() {
        return None; // main checkout
    }
    let contents = fs::read_to_string(&git_path).ok()?;
    let line = contents.lines().find(|l| l.starts_with("gitdir:"))?;
    let gitdir = line.trim_start_matches("gitdir:").trim();
    // Expect: /path/to/main/.git/worktrees/<name>
    let path = Path::new(gitdir);
    let parent = path.parent()?;
    if parent.file_name()?.to_str()? != "worktrees" {
        return None;
    }
    let name = path.file_name()?.to_str()?;
    // Reject names that would resolve to a parent directory or contain a path
    // separator: a malicious or buggy `gitdir:` pointer ending in `..` would
    // otherwise collapse the worktree's cache into the main checkout's
    // directory and silently defeat the per-worktree isolation guarantee.
    if name.is_empty() || name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return None;
    }
    Some(name.to_string())
}

fn cache_dir(root_dir: &Path) -> PathBuf {
    let base = root_dir.join(CACHE_DIR);
    match worktree_name(root_dir) {
        Some(name) => base.join(WORKTREE_SUBDIR).join(name),
        None => base,
    }
}

fn cache_path(root_dir: &Path, name: &str) -> PathBuf {
    cache_dir(root_dir).join(format!("{}.rkyv", name))
}

fn write_lock_path(root_dir: &Path, name: &str) -> PathBuf {
    // Per-cache lock so that concurrent writes to *different* caches in the
    // same directory don't serialize against each other.
    cache_dir(root_dir).join(format!(".{}.write.lock", name))
}

pub fn ensure_cache_dir(root_dir: &Path) -> Result<()> {
    fs::create_dir_all(cache_dir(root_dir))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Serializable cache format
// ---------------------------------------------------------------------------

#[derive(Archive, Serialize, Deserialize, Debug)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct CacheData {
    pub dims: u32,
    pub keys: Vec<String>,
    pub hashes: Vec<String>,
    pub vectors: Vec<f32>,
}

impl CacheData {
    /// Build from a VectorStore.
    pub fn from_store(store: &VectorStore) -> Self {
        Self {
            dims: store.dims() as u32,
            keys: store.keys().to_vec(),
            hashes: store.hashes().to_vec(),
            vectors: store.vectors_data().to_vec(),
        }
    }

    /// Convert to a VectorStore.
    pub fn to_store(&self) -> VectorStore {
        VectorStore::new(
            self.dims,
            self.keys.clone(),
            self.hashes.clone(),
            self.vectors.clone(),
        )
    }

    /// Remove cache entries whose keys would be excluded by the walker's path
    /// filter (dot-segment hidden-path rule).
    ///
    /// This is the hygiene sweep that clears stale entries accumulated before
    /// the dot-segment exclusion was enforced — e.g. `.claude/worktrees/…`
    /// paths written when the MCP server was mistakenly run from inside a
    /// worktree directory, or leftovers from an older code version that did not
    /// apply `should_track` during initial indexing.
    ///
    /// The sweep is cheap (single linear pass, no allocation when nothing is
    /// removed) and is called unconditionally on every `load_cache` so the
    /// on-disk file self-heals on the next `save_cache` write.
    ///
    /// Returns the number of entries removed (for logging).
    pub fn sweep_excluded_keys(&mut self) -> usize {
        let dim = self.dims as usize;
        if dim == 0 || self.keys.is_empty() {
            return 0;
        }

        // A well-formed CacheData has keys.len() == hashes.len() and
        // vectors.len() == keys.len() * dim. If these invariants are violated
        // (partially-written file, older version with different schema, etc.),
        // treat it as corruption: skip the entire entry rather than silently
        // producing misaligned keys/hashes/vectors. Downstream `to_store()`
        // would otherwise panic on misalignment.
        let mut keep_new_keys: Vec<String> = Vec::with_capacity(self.keys.len());
        let mut keep_new_hashes: Vec<String> = Vec::with_capacity(self.hashes.len());
        let mut keep_new_vectors: Vec<f32> = Vec::with_capacity(self.vectors.len());
        let mut removed = 0usize;

        for (i, key) in self.keys.iter().enumerate() {
            let off = i * dim;
            let hash_ok = self.hashes.get(i).is_some();
            let vector_ok = off + dim <= self.vectors.len();

            if !hash_ok || !vector_ok {
                // Corrupt or truncated entry — drop it entirely.
                tracing::warn!(
                    key = %key,
                    hash_present = hash_ok,
                    vector_in_bounds = vector_ok,
                    "sweep_excluded_keys: misaligned cache entry — dropping"
                );
                removed += 1;
                continue;
            }

            if crate::core::walker::should_track(key, &Default::default()) {
                keep_new_keys.push(key.clone());
                keep_new_hashes.push(self.hashes[i].clone());
                keep_new_vectors.extend_from_slice(&self.vectors[off..off + dim]);
            } else {
                removed += 1;
            }
        }

        if removed > 0 {
            self.keys = keep_new_keys;
            self.hashes = keep_new_hashes;
            self.vectors = keep_new_vectors;
        }

        removed
    }

    /// Build from a cache map.
    pub fn from_cache_map(cache: &HashMap<String, CacheEntry>) -> Option<Self> {
        if cache.is_empty() {
            return None;
        }
        let n = cache.len();
        let dims = cache.values().next()?.vector.len() as u32;
        debug_assert!(
            cache.values().all(|e| e.vector.len() == dims as usize),
            "from_cache_map: mixed vector dimensions in cache"
        );
        let mut keys = Vec::with_capacity(n);
        let mut hashes = Vec::with_capacity(n);
        let mut vectors = Vec::with_capacity(n * dims as usize);
        for (key, entry) in cache {
            // Skip mismatched-dim entries rather than panicking — corrupts
            // would otherwise propagate into the on-disk archive and poison
            // every future load.
            if entry.vector.len() != dims as usize {
                continue;
            }
            keys.push(key.clone());
            hashes.push(entry.hash.clone());
            vectors.extend_from_slice(&entry.vector);
        }
        Some(Self {
            dims,
            keys,
            hashes,
            vectors,
        })
    }
}

// ---------------------------------------------------------------------------
// Save / Load with rkyv
// ---------------------------------------------------------------------------

/// Per-process monotonic counter to ensure unique temp file names across threads.
static WRITE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Save a CacheData to disk using rkyv serialization.
///
/// Multi-process safe: before writing, reloads the on-disk state and merges it
/// with `data`. Without this, two processes sharing the same cache file (e.g.
/// two Claude Code sessions each spawning their own MCP, or an offline warmup
/// binary running alongside a live MCP server) would clobber each other —
/// each process snapshots its own in-memory state and the last writer wins.
///
/// The merge is "incoming overrides disk by key": entries present in both keep
/// the incoming hash + vector, entries unique to either side are preserved.
/// This reflects the typical caller intent — a fresh embed for a re-hashed
/// file should win over a stale cache entry, and embeds neither process knows
/// about should be retained.
///
/// Atomic on-disk swap is preserved (write to unique temp file then rename).
pub fn save_cache(root_dir: &Path, name: &str, data: &CacheData) -> Result<()> {
    save_cache_with_deletions(root_dir, name, data, &[])
}

/// Same as [`save_cache`] but additionally removes any keys present in
/// `deletions` from the merged result before persisting.
///
/// Deletion semantics: load disk, overlay incoming on top (incoming wins for
/// shared keys), then strip every key in `deletions`. This lets the live MCP
/// runtime persist file deletions across save cycles — without it, a key
/// removed from the in-memory cache would be silently re-merged from disk on
/// every subsequent save and the cache would grow monotonically.
///
/// All work happens under the same fd-lock as `save_cache`.
pub fn save_cache_with_deletions(
    root_dir: &Path,
    name: &str,
    data: &CacheData,
    deletions: &[String],
) -> Result<()> {
    ensure_cache_dir(root_dir)?;

    // Acquire an exclusive advisory lock on the sentinel file before the
    // load-merge-write sequence to prevent TOCTOU lost-update races between
    // concurrent writers (multiple threads or processes sharing the same cache
    // directory). The lock is blocking: a waiting writer will park until the
    // current writer releases it, which is preferable to silently losing entries.
    let lock_path = write_lock_path(root_dir, name);
    let lock_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)?;
    let mut fd_lock = FdRwLock::new(lock_file);
    // `write()` blocks until all other holders release the lock.
    let _guard = fd_lock.write()?;

    // --- critical section: load → merge → write ---

    // Read-modify-write: merge with whatever is currently on disk so we don't
    // lose entries written by a concurrent process between our load and save.
    //
    // A corrupt or version-mismatched on-disk file (truncated payload, bumped
    // CACHE_VERSION, partial write from a prior crash) returns Err here.
    // Propagating it would brick every subsequent save_cache call indefinitely
    // until the cache file is manually removed. Rotate the broken file aside
    // (preserving operator-recoverable data) and proceed with the incoming
    // snapshot — refusing to write forever is strictly worse than rotating.
    let mut merged = match load_cache(root_dir, name) {
        Ok(Some(disk)) => merge_cache_data(disk, data),
        Ok(None) => clone_cache_data(data),
        Err(e) => {
            let rotated = rotate_corrupt_cache(root_dir, name);
            match rotated {
                Ok(Some(rotated_path)) => tracing::warn!(
                    cache = name,
                    error = %e,
                    rotated = %rotated_path.display(),
                    "save_cache: on-disk cache unreadable; rotated to {} and overwriting with incoming snapshot",
                    rotated_path.display()
                ),
                Ok(None) => tracing::warn!(
                    cache = name,
                    error = %e,
                    "save_cache: on-disk cache unreadable and not present after rotation attempt; overwriting with incoming snapshot"
                ),
                Err(rotate_err) => tracing::warn!(
                    cache = name,
                    error = %e,
                    rotate_error = %rotate_err,
                    "save_cache: on-disk cache unreadable and rotation failed; overwriting with incoming snapshot"
                ),
            }
            clone_cache_data(data)
        }
    };

    if !deletions.is_empty() {
        apply_deletions(&mut merged, deletions);
    }

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&merged)
        .map_err(|e| ContextPlusError::Serialization(format!("rkyv serialize: {}", e)))?;

    let path = cache_path(root_dir, name);
    let tmp_path = cache_dir(root_dir).join(format!(
        "{}.rkyv.{}.{}.tmp",
        name,
        std::process::id(),
        WRITE_COUNTER.fetch_add(1, Ordering::Relaxed),
    ));

    // Write: 16-byte aligned header (version + padding) + rkyv data
    let mut buf = Vec::with_capacity(HEADER_SIZE + bytes.len());
    buf.resize(HEADER_SIZE, 0);
    buf[0] = CACHE_VERSION;
    buf.extend_from_slice(&bytes);

    if let Err(e) = fs::write(&tmp_path, &buf) {
        let _ = fs::remove_file(&tmp_path);
        return Err(e.into());
    }
    if let Err(e) = fs::rename(&tmp_path, &path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(e.into());
    }

    // _guard dropped here, releasing the exclusive lock
    Ok(())
}

fn clone_cache_data(data: &CacheData) -> CacheData {
    CacheData {
        dims: data.dims,
        keys: data.keys.clone(),
        hashes: data.hashes.clone(),
        vectors: data.vectors.clone(),
    }
}

/// Remove every key in `deletions` from `data` in-place, keeping the parallel
/// arrays (keys, hashes, vectors) consistent. No-op for keys not present.
fn apply_deletions(data: &mut CacheData, deletions: &[String]) {
    if deletions.is_empty() || data.keys.is_empty() {
        return;
    }
    let dim = data.dims as usize;
    let to_remove: std::collections::HashSet<&str> = deletions.iter().map(|s| s.as_str()).collect();

    let mut new_keys = Vec::with_capacity(data.keys.len());
    let mut new_hashes = Vec::with_capacity(data.keys.len());
    let mut new_vectors = if dim == 0 {
        Vec::new()
    } else {
        Vec::with_capacity(data.keys.len() * dim)
    };

    for (i, key) in data.keys.iter().enumerate() {
        if to_remove.contains(key.as_str()) {
            continue;
        }
        new_keys.push(key.clone());
        new_hashes.push(data.hashes.get(i).cloned().unwrap_or_default());
        if dim > 0 {
            let off = i * dim;
            if off + dim <= data.vectors.len() {
                new_vectors.extend_from_slice(&data.vectors[off..off + dim]);
            } else {
                // Misaligned entry — drop it (parallel arrays must stay sized).
                new_keys.pop();
                new_hashes.pop();
            }
        }
    }

    data.keys = new_keys;
    data.hashes = new_hashes;
    data.vectors = new_vectors;
}

/// Move a corrupt cache file aside as `<path>.corrupt.<unix_ts>` so the
/// operator can recover data from it. Returns the new path on success, or
/// `Ok(None)` if the cache file was already absent. Errors are propagated.
fn rotate_corrupt_cache(root_dir: &Path, name: &str) -> Result<Option<PathBuf>> {
    let path = cache_path(root_dir, name);
    if !path.exists() {
        return Ok(None);
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut rotated = path.clone();
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("cache");
    rotated.set_file_name(format!("{}.corrupt.{}", file_name, ts));
    fs::rename(&path, &rotated)?;
    Ok(Some(rotated))
}

/// Merge two `CacheData` snapshots, with `incoming` overriding `disk` for
/// entries that share a key. Returns a freshly-allocated `CacheData`.
///
/// Dim mismatches with both sides non-empty (e.g. caches from different
/// embedding models pointed at the same path) are unrecoverable; in that
/// case `incoming` wins outright and `disk` is dropped. This shouldn't
/// happen in normal operation because cache file names include the model.
fn merge_cache_data(disk: CacheData, incoming: &CacheData) -> CacheData {
    if !disk.keys.is_empty() && !incoming.keys.is_empty() && disk.dims != incoming.dims {
        return clone_cache_data(incoming);
    }

    let dim = if disk.keys.is_empty() {
        incoming.dims as usize
    } else {
        disk.dims as usize
    };

    if dim == 0 {
        return clone_cache_data(incoming);
    }

    let mut map: HashMap<String, (String, Vec<f32>)> =
        HashMap::with_capacity(disk.keys.len() + incoming.keys.len());

    let mut insert = |keys: Vec<String>, hashes: Vec<String>, vectors: Vec<f32>| {
        for (i, key) in keys.into_iter().enumerate() {
            let off = i * dim;
            if off + dim > vectors.len() {
                continue;
            }
            let hash = hashes.get(i).cloned().unwrap_or_default();
            let vec = vectors[off..off + dim].to_vec();
            map.insert(key, (hash, vec));
        }
    };

    insert(disk.keys, disk.hashes, disk.vectors);
    // Incoming overwrites by key (newer embed wins for shared keys)
    insert(
        incoming.keys.clone(),
        incoming.hashes.clone(),
        incoming.vectors.clone(),
    );

    let n = map.len();
    let mut keys = Vec::with_capacity(n);
    let mut hashes = Vec::with_capacity(n);
    let mut vectors = Vec::with_capacity(n * dim);
    for (key, (hash, vec)) in map {
        keys.push(key);
        hashes.push(hash);
        vectors.extend_from_slice(&vec);
    }

    CacheData {
        dims: dim as u32,
        keys,
        hashes,
        vectors,
    }
}

/// Load a CacheData from disk.
/// Validates version byte and uses rkyv bytecheck for safe deserialization.
pub fn load_cache(root_dir: &Path, name: &str) -> Result<Option<CacheData>> {
    let path = cache_path(root_dir, name);
    if !path.exists() {
        return Ok(None);
    }

    let bytes = fs::read(&path)?;
    if bytes.len() < HEADER_SIZE {
        return Ok(None);
    }

    // Check version
    if bytes[0] != CACHE_VERSION {
        return Err(ContextPlusError::Cache(format!(
            "unsupported cache version: {} (expected {})",
            bytes[0], CACHE_VERSION
        )));
    }

    let data_bytes = &bytes[HEADER_SIZE..];

    let mut data = rkyv::from_bytes::<CacheData, rkyv::rancor::Error>(data_bytes)
        .map_err(|e| ContextPlusError::Cache(format!("rkyv deserialize: {}", e)))?;

    // Hygiene sweep: drop any entries whose key would be excluded by the
    // walker's dot-segment rule (e.g. stale `.claude/worktrees/…` paths that
    // were indexed before the exclusion was enforced).
    let removed = data.sweep_excluded_keys();
    if removed > 0 {
        tracing::info!(
            removed,
            cache = name,
            "load_cache: swept {} excluded-path entries (stale worktree/hidden paths)",
            removed
        );
    }

    Ok(Some(data))
}

/// Load directly into a VectorStore.
pub fn load_vector_store(root_dir: &Path, name: &str) -> Result<Option<VectorStore>> {
    match load_cache(root_dir, name)? {
        Some(data) => Ok(Some(data.to_store())),
        None => Ok(None),
    }
}

/// Merge an in-memory `VectorStore` into the on-disk cache, preserving any
/// keys the on-disk file has that this caller doesn't know about.
///
/// This is the single canonical save path. Use on any caller — live MCP
/// runtime, warmup binaries, tests — because no caller actually owns the
/// entire keyspace. The MCP server's in-memory snapshot only covers the
/// subset of keys it has touched this session; an offline `warmup_*` binary
/// or a second MCP instance may have written entries the current process
/// never loaded. A naive overwrite would silently drop them.
///
/// Atomic: acquires the cache's fd-lock, loads the on-disk `CacheData`,
/// merges incoming entries on top (incoming wins for shared keys), and writes
/// the union back via tmp-file + rename. All under one lock — no TOCTOU
/// window between the load and the save.
///
/// Cost on a ~150 MB cache: one extra read + one merge pass + one write.
pub fn save_vector_store_merged(root_dir: &Path, name: &str, store: &VectorStore) -> Result<()> {
    save_vector_store_merged_with_deletions(root_dir, name, store, &[])
}

/// Same as [`save_vector_store_merged`] but additionally evicts any keys in
/// `deletions` from the on-disk cache.
///
/// Required on the live MCP runtime when source files have been deleted
/// since the last save: the in-memory cache evicts the entry on file
/// deletion, but a plain merged save would re-populate it from the on-disk
/// snapshot and the cache would grow monotonically. Pass the set of keys
/// that were `cache.remove`'d since the last save here and the on-disk
/// state will track the deletions.
pub fn save_vector_store_merged_with_deletions(
    root_dir: &Path,
    name: &str,
    store: &VectorStore,
    deletions: &[String],
) -> Result<()> {
    let data = CacheData::from_store(store);
    save_cache_with_deletions(root_dir, name, &data, deletions)
}

// ---------------------------------------------------------------------------
// Query embedding cache (persistent, keyed by query text)
// ---------------------------------------------------------------------------

/// Maximum number of query-embedding entries to persist on disk.
pub const QUERY_CACHE_MAX: usize = 10_000;

/// Serializable format for the persistent query-embedding cache.
///
/// Unlike `CacheData` (which stores file embeddings keyed by path + content
/// hash), `QueryCacheData` stores embeddings keyed by raw query text. No hash
/// column is needed — the key itself is the identity.
///
/// Layout: parallel arrays (`keys[i]` → `vectors[i*dims..(i+1)*dims]`).
#[derive(Archive, Serialize, Deserialize, Debug)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct QueryCacheData {
    pub dims: u32,
    pub keys: Vec<String>,
    pub vectors: Vec<f32>,
}

/// Build a sanitized model-name slug suitable for use in a filename.
///
/// Replaces any character that is not alphanumeric, `-`, or `_` with `-` and
/// truncates to 64 bytes so the resulting filename stays filesystem-safe.
pub fn model_slug(model: &str) -> String {
    let slug: String = model
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    slug.chars().take(64).collect()
}

/// Return the cache name (without `.rkyv` extension) for a query cache file.
///
/// Example: `model = "snowflake-arctic-embed2"` → `"query-embeddings-snowflake-arctic-embed2"`.
pub fn query_cache_name(model: &str) -> String {
    format!("query-embeddings-{}", model_slug(model))
}

/// Save the in-memory query LRU snapshot to disk.
///
/// `entries` is a slice of `(query_text, embedding_vector)` pairs representing
/// the current LRU state (oldest-first, MRU last). At most `QUERY_CACHE_MAX`
/// entries are written; if the slice is longer, the MRU tail is kept (we slice
/// from the end).
///
/// Uses an atomic write (tmp → rename). No merge with the existing file —
/// the caller's in-memory LRU is the single source of truth for the query
/// cache (only one MCP process writes it).
pub fn save_query_cache(
    root_dir: &Path,
    model: &str,
    entries: &[(String, Vec<f32>)],
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }

    ensure_cache_dir(root_dir)?;

    let entries = if entries.len() > QUERY_CACHE_MAX {
        &entries[entries.len() - QUERY_CACHE_MAX..]
    } else {
        entries
    };

    let dims = entries[0].1.len() as u32;
    let mut keys = Vec::with_capacity(entries.len());
    let mut vectors = Vec::with_capacity(entries.len() * dims as usize);
    for (k, v) in entries {
        if v.len() as u32 != dims {
            continue; // skip dim-mismatched entries
        }
        keys.push(k.clone());
        vectors.extend_from_slice(v);
    }

    let data = QueryCacheData {
        dims,
        keys,
        vectors,
    };

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&data)
        .map_err(|e| ContextPlusError::Serialization(format!("rkyv serialize query cache: {e}")))?;

    let name = query_cache_name(model);
    let path = cache_path(root_dir, &name);
    let tmp_path = cache_dir(root_dir).join(format!(
        "{}.rkyv.{}.{}.tmp",
        name,
        std::process::id(),
        WRITE_COUNTER.fetch_add(1, Ordering::Relaxed),
    ));

    let mut buf = Vec::with_capacity(HEADER_SIZE + bytes.len());
    buf.resize(HEADER_SIZE, 0);
    buf[0] = CACHE_VERSION;
    buf.extend_from_slice(&bytes);

    if let Err(e) = fs::write(&tmp_path, &buf) {
        let _ = fs::remove_file(&tmp_path);
        return Err(e.into());
    }
    if let Err(e) = fs::rename(&tmp_path, &path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(e.into());
    }

    Ok(())
}

/// Load the persisted query-embedding cache from disk.
///
/// Returns `Ok(vec![])` if the file does not exist or is empty.
/// Returns `Err` only on format/version errors.
pub fn load_query_cache(root_dir: &Path, model: &str) -> Result<Vec<(String, Vec<f32>)>> {
    let name = query_cache_name(model);
    let path = cache_path(root_dir, &name);
    if !path.exists() {
        return Ok(vec![]);
    }

    let bytes = fs::read(&path)?;
    if bytes.len() < HEADER_SIZE {
        return Ok(vec![]);
    }
    if bytes[0] != CACHE_VERSION {
        return Err(ContextPlusError::Cache(format!(
            "query cache: unsupported version {} (expected {})",
            bytes[0], CACHE_VERSION
        )));
    }

    let data_bytes = &bytes[HEADER_SIZE..];
    let data = rkyv::from_bytes::<QueryCacheData, rkyv::rancor::Error>(data_bytes)
        .map_err(|e| ContextPlusError::Cache(format!("rkyv deserialize query cache: {e}")))?;

    let dim = data.dims as usize;
    if dim == 0 || data.keys.is_empty() {
        return Ok(vec![]);
    }

    let mut result = Vec::with_capacity(data.keys.len());
    for (i, key) in data.keys.iter().enumerate() {
        let off = i * dim;
        if off + dim > data.vectors.len() {
            tracing::warn!(key = %key, "load_query_cache: truncated vector entry — skipping");
            continue;
        }
        result.push((key.clone(), data.vectors[off..off + dim].to_vec()));
    }

    Ok(result)
}

/// Load cache with mmap for zero-copy read (uses memmap2).
/// Falls back to regular load on mmap failure.
pub fn load_cache_mmap(root_dir: &Path, name: &str) -> Result<Option<CacheData>> {
    let path = cache_path(root_dir, name);
    if !path.exists() {
        return Ok(None);
    }

    let file = fs::File::open(&path)?;
    let metadata = file.metadata()?;
    if (metadata.len() as usize) < HEADER_SIZE + 1 {
        return Ok(None);
    }

    // Try mmap
    let mmap = unsafe { memmap2::Mmap::map(&file) };
    match mmap {
        Ok(mmap) => {
            if mmap[0] != CACHE_VERSION {
                return Err(ContextPlusError::Cache(format!(
                    "unsupported cache version: {} (expected {})",
                    mmap[0], CACHE_VERSION
                )));
            }
            let data_bytes = &mmap[HEADER_SIZE..];
            let data = rkyv::from_bytes::<CacheData, rkyv::rancor::Error>(data_bytes)
                .map_err(|e| ContextPlusError::Cache(format!("rkyv mmap deserialize: {}", e)))?;
            Ok(Some(data))
        }
        Err(_) => {
            // Fallback to regular load
            load_cache(root_dir, name)
        }
    }
}

/// Load a VectorStore with zero-copy mmap for the vector data.
///
/// Keys and hashes are copied out (they're small), but the bulk vector data
/// (which can be 100MB+) stays in the mmap'd region — no heap allocation.
///
/// Uses `rkyv::access` to get a reference to the archived data without
/// deserializing, then reinterprets the archived `Vec<f32_le>` data pointer
/// as `*const f32` (safe on little-endian platforms where f32_le == f32).
///
/// Falls back to the regular (copying) `load_vector_store` on mmap failure.
pub fn mmap_vector_store(root_dir: &Path, name: &str) -> Result<Option<VectorStore>> {
    let path = cache_path(root_dir, name);
    if !path.exists() {
        return Ok(None);
    }

    let file = fs::File::open(&path)?;
    let metadata = file.metadata()?;
    if (metadata.len() as usize) < HEADER_SIZE + 1 {
        return Ok(None);
    }

    // Try mmap; fall back to regular load on failure
    let mmap = match unsafe { memmap2::Mmap::map(&file) } {
        Ok(m) => m,
        Err(_) => return load_vector_store(root_dir, name),
    };

    // Validate version header
    if mmap[0] != CACHE_VERSION {
        return Err(ContextPlusError::Cache(format!(
            "unsupported cache version: {} (expected {})",
            mmap[0], CACHE_VERSION
        )));
    }

    let data_bytes = &mmap[HEADER_SIZE..];

    // Zero-copy access: returns &ArchivedCacheData pointing into the mmap
    let archived = rkyv::access::<ArchivedCacheData, rkyv::rancor::Error>(data_bytes)
        .map_err(|e| ContextPlusError::Cache(format!("rkyv access: {}", e)))?;

    // Copy small metadata (keys + hashes are typically <1% of total size)
    let dims = archived.dims.to_native();
    let keys: Vec<String> = archived
        .keys
        .iter()
        .map(|s| s.as_str().to_owned())
        .collect();
    let hashes: Vec<String> = archived
        .hashes
        .iter()
        .map(|s| s.as_str().to_owned())
        .collect();

    // Zero-copy vector data: reinterpret &[f32_le] as *const f32.
    // SAFETY: On little-endian platforms (x86_64, aarch64), rkyv's f32_le has
    // identical memory layout to native f32. The archived vectors slice lives
    // inside the mmap region which we keep alive via Arc.
    let archived_vectors = archived.vectors.as_slice();
    let vectors_ptr = archived_vectors.as_ptr() as *const f32;
    let vectors_len = archived_vectors.len();

    // Runtime alignment check: rkyv guarantees alignment for archived data,
    // but verify in debug builds that the cast is valid.
    debug_assert_eq!(
        vectors_ptr as usize % std::mem::align_of::<f32>(),
        0,
        "mmap f32 pointer is not 4-byte aligned"
    );

    let mmap = Arc::new(mmap);

    // SAFETY: vectors_ptr points into the mmap, is aligned (rkyv guarantees
    // alignment), and the Arc<Mmap> keeps the mapping alive.
    let store =
        unsafe { VectorStore::from_mmap(dims, keys, hashes, vectors_ptr, vectors_len, mmap) };

    Ok(Some(store))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_test_data() -> CacheData {
        CacheData {
            dims: 3,
            keys: vec!["src/auth.ts".to_string(), "src/db.ts".to_string()],
            hashes: vec!["hash_a".to_string(), "hash_b".to_string()],
            vectors: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    }

    #[test]
    fn rkyv_round_trip() {
        let dir = TempDir::new().unwrap();
        let data = make_test_data();

        save_cache(dir.path(), "test-cache", &data).unwrap();
        let loaded = load_cache(dir.path(), "test-cache").unwrap().unwrap();

        assert_eq!(loaded.dims, data.dims);
        assert_eq!(loaded.keys, data.keys);
        assert_eq!(loaded.hashes, data.hashes);
        assert_eq!(loaded.vectors.len(), data.vectors.len());
        for (a, b) in loaded.vectors.iter().zip(data.vectors.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let dir = TempDir::new().unwrap();
        let result = load_cache(dir.path(), "nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn save_creates_cache_dir() {
        let dir = TempDir::new().unwrap();
        let data = make_test_data();

        // .mcp_data should not exist yet
        assert!(!cache_dir(dir.path()).exists());

        save_cache(dir.path(), "test", &data).unwrap();

        assert!(cache_dir(dir.path()).exists());
        assert!(cache_path(dir.path(), "test").exists());
    }

    #[test]
    fn wrong_version_returns_error() {
        let dir = TempDir::new().unwrap();
        ensure_cache_dir(dir.path()).unwrap();

        // Write a file with wrong version (must be >= HEADER_SIZE + 1 to not be treated as empty)
        let path = cache_path(dir.path(), "bad");
        let mut bad_data = vec![99u8; HEADER_SIZE + 4];
        bad_data[0] = 99; // wrong version
        fs::write(&path, &bad_data).unwrap();

        let result = load_cache(dir.path(), "bad");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("unsupported cache version"));
    }

    #[test]
    fn empty_file_returns_none() {
        let dir = TempDir::new().unwrap();
        ensure_cache_dir(dir.path()).unwrap();

        let path = cache_path(dir.path(), "empty");
        fs::write(&path, []).unwrap();

        let result = load_cache(dir.path(), "empty").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn vector_store_round_trip() {
        let dir = TempDir::new().unwrap();

        let store = VectorStore::new(
            3,
            vec!["a.ts".to_string(), "b.ts".to_string()],
            vec!["h1".to_string(), "h2".to_string()],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        save_vector_store_merged(dir.path(), "vectors", &store).unwrap();
        let loaded = load_vector_store(dir.path(), "vectors").unwrap().unwrap();

        assert_eq!(loaded.count(), 2);
        assert_eq!(loaded.dims(), 3);
        assert!(loaded.has_key("a.ts"));
        assert!(loaded.has_key("b.ts"));

        let vec_a = loaded.get_vector("a.ts").unwrap();
        assert!((vec_a[0] - 1.0).abs() < 1e-6);
        assert!((vec_a[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn cache_data_from_store() {
        let store = VectorStore::new(
            2,
            vec!["x".to_string()],
            vec!["hx".to_string()],
            vec![0.5, 0.5],
        );
        let data = CacheData::from_store(&store);
        assert_eq!(data.dims, 2);
        assert_eq!(data.keys.len(), 1);
        assert_eq!(data.vectors.len(), 2);
    }

    #[test]
    fn cache_data_from_cache_map() {
        let mut cache = HashMap::new();
        cache.insert(
            "file.rs".to_string(),
            CacheEntry {
                hash: "abc".to_string(),
                vector: vec![1.0, 2.0],
            },
        );
        let data = CacheData::from_cache_map(&cache).unwrap();
        assert_eq!(data.dims, 2);
        assert_eq!(data.keys.len(), 1);
    }

    #[test]
    fn cache_data_from_empty_map() {
        let cache = HashMap::new();
        assert!(CacheData::from_cache_map(&cache).is_none());
    }

    #[test]
    fn mmap_load_works() {
        let dir = TempDir::new().unwrap();
        let data = make_test_data();

        save_cache(dir.path(), "mmap-test", &data).unwrap();
        let loaded = load_cache_mmap(dir.path(), "mmap-test").unwrap().unwrap();

        assert_eq!(loaded.dims, data.dims);
        assert_eq!(loaded.keys, data.keys);
    }

    #[test]
    fn mmap_nonexistent_returns_none() {
        let dir = TempDir::new().unwrap();
        let result = load_cache_mmap(dir.path(), "no-such").unwrap();
        assert!(result.is_none());
    }

    // -- Concurrent-writer merge regression tests --

    #[test]
    fn save_preserves_disk_entries_not_in_incoming() {
        // Simulates: process A loaded N entries, process B saved N+M entries,
        // process A then saves only its (smaller) snapshot. Without the merge,
        // process B's M new entries would be wiped.
        let dir = TempDir::new().unwrap();

        // Process B writes a richer state to disk first
        let disk = CacheData {
            dims: 2,
            keys: vec!["a".into(), "b".into(), "c".into()],
            hashes: vec!["ha".into(), "hb".into(), "hc".into()],
            vectors: vec![1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        };
        save_cache(dir.path(), "race", &disk).unwrap();

        // Process A snapshots only the original 1 entry it knew about
        let stale = CacheData {
            dims: 2,
            keys: vec!["a".into()],
            hashes: vec!["ha".into()],
            vectors: vec![1.0, 1.1],
        };
        save_cache(dir.path(), "race", &stale).unwrap();

        let loaded = load_cache(dir.path(), "race").unwrap().unwrap();
        let key_set: std::collections::HashSet<_> = loaded.keys.iter().cloned().collect();
        assert!(
            key_set.contains("b") && key_set.contains("c"),
            "merge must preserve b/c written by the concurrent writer; got keys={:?}",
            loaded.keys
        );
        assert_eq!(loaded.keys.len(), 3);
    }

    #[test]
    fn save_incoming_overrides_disk_for_shared_keys() {
        // When the same key is in both, incoming wins (re-embed of a changed file
        // should replace the stale entry rather than be silently discarded).
        let dir = TempDir::new().unwrap();

        let disk = CacheData {
            dims: 2,
            keys: vec!["a".into()],
            hashes: vec!["old-hash".into()],
            vectors: vec![0.1, 0.2],
        };
        save_cache(dir.path(), "override", &disk).unwrap();

        let incoming = CacheData {
            dims: 2,
            keys: vec!["a".into()],
            hashes: vec!["new-hash".into()],
            vectors: vec![0.9, 0.8],
        };
        save_cache(dir.path(), "override", &incoming).unwrap();

        let loaded = load_cache(dir.path(), "override").unwrap().unwrap();
        let idx = loaded.keys.iter().position(|k| k == "a").unwrap();
        assert_eq!(loaded.hashes[idx], "new-hash");
        assert!((loaded.vectors[idx * 2] - 0.9).abs() < 1e-6);
        assert!((loaded.vectors[idx * 2 + 1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn save_fresh_path_no_merge_needed() {
        // When the cache file doesn't exist, save the incoming data verbatim.
        let dir = TempDir::new().unwrap();
        let data = CacheData {
            dims: 2,
            keys: vec!["x".into(), "y".into()],
            hashes: vec!["hx".into(), "hy".into()],
            vectors: vec![0.1, 0.2, 0.3, 0.4],
        };
        save_cache(dir.path(), "fresh", &data).unwrap();

        let loaded = load_cache(dir.path(), "fresh").unwrap().unwrap();
        assert_eq!(loaded.keys.len(), 2);
        let key_set: std::collections::HashSet<_> = loaded.keys.iter().cloned().collect();
        assert!(key_set.contains("x") && key_set.contains("y"));
    }

    #[test]
    fn save_preserves_disk_entries_when_incoming_is_empty() {
        // Edge case: a writer with no in-memory entries shouldn't wipe the disk.
        let dir = TempDir::new().unwrap();

        let disk = CacheData {
            dims: 2,
            keys: vec!["keep-me".into()],
            hashes: vec!["h1".into()],
            vectors: vec![0.5, 0.5],
        };
        save_cache(dir.path(), "noop", &disk).unwrap();

        let empty = CacheData {
            dims: 2,
            keys: vec![],
            hashes: vec![],
            vectors: vec![],
        };
        save_cache(dir.path(), "noop", &empty).unwrap();

        let loaded = load_cache(dir.path(), "noop").unwrap().unwrap();
        assert!(
            loaded.keys.iter().any(|k| k == "keep-me"),
            "an empty incoming snapshot must not wipe pre-existing entries"
        );
    }

    #[test]
    fn atomic_write_survives() {
        let dir = TempDir::new().unwrap();
        let data = make_test_data();

        // Save twice — second should overwrite cleanly
        save_cache(dir.path(), "atomic", &data).unwrap();
        save_cache(dir.path(), "atomic", &data).unwrap();

        let loaded = load_cache(dir.path(), "atomic").unwrap().unwrap();
        // Order is not preserved through the merge round-trip, but the key set must match.
        let loaded_keys: std::collections::HashSet<_> = loaded.keys.iter().cloned().collect();
        let data_keys: std::collections::HashSet<_> = data.keys.iter().cloned().collect();
        assert_eq!(loaded_keys, data_keys);

        // No temp files should linger in the cache directory
        let cache = cache_dir(dir.path());
        let lingering: Vec<_> = fs::read_dir(&cache)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "tmp"))
            .collect();
        assert!(
            lingering.is_empty(),
            "temp files should not linger: {:?}",
            lingering
        );
    }

    /// ADV-004 regression: N concurrent threads each writing disjoint keys must
    /// produce a final cache that contains ALL entries (union), not just the last
    /// writer's batch.  Without the file lock this test fails intermittently
    /// (last writer silently drops earlier writers' entries); with the lock it
    /// must pass every time.
    #[test]
    fn concurrent_writers_no_lost_updates() {
        use std::collections::HashSet;
        use std::sync::Arc as StdArc;
        use std::sync::Barrier;

        const N: usize = 4;
        let dir = TempDir::new().unwrap();
        let root = dir.path().to_path_buf();

        // Ensure the cache dir and lock file exist before spawning threads so
        // all threads race against the same pre-created sentinel.
        ensure_cache_dir(&root).unwrap();

        let barrier = StdArc::new(Barrier::new(N));

        std::thread::scope(|s| {
            for i in 0..N {
                let root = root.clone();
                let barrier = StdArc::clone(&barrier);
                s.spawn(move || {
                    // Each thread owns exactly one unique key.
                    let key = format!("file_{}.ts", i);
                    let data = CacheData {
                        dims: 2,
                        keys: vec![key],
                        hashes: vec![format!("hash_{}", i)],
                        vectors: vec![i as f32, i as f32 + 0.1],
                    };
                    // All threads synchronize here to maximise the chance of
                    // concurrent entry into save_cache.
                    barrier.wait();
                    save_cache(&root, "race-test", &data).unwrap();
                });
            }
        });

        let loaded = load_cache(&root, "race-test").unwrap().unwrap();
        let key_set: HashSet<_> = loaded.keys.iter().cloned().collect();

        for i in 0..N {
            let expected = format!("file_{}.ts", i);
            assert!(
                key_set.contains(&expected),
                "key '{}' was lost in concurrent write — TOCTOU not fixed; got keys={:?}",
                expected,
                loaded.keys
            );
        }
        assert_eq!(
            loaded.keys.len(),
            N,
            "expected exactly {} keys, got {} — duplicate or missing entries",
            N,
            loaded.keys.len()
        );

        // TEST-004: assert per-thread CONTENT (hash + vector bytes) survives,
        // not just the key. A regression where merge silently dropped
        // values but kept keys (e.g. by re-using a stale disk vector under
        // a key the incoming writer was overwriting) would still pass the
        // key-set check above. This pins the "incoming-overrides-disk"
        // contract end-to-end.
        let dim = loaded.dims as usize;
        assert_eq!(dim, 2);
        assert_eq!(
            loaded.vectors.len(),
            N * dim,
            "vectors buffer length mismatch: {} ≠ {} keys × {} dims",
            loaded.vectors.len(),
            N,
            dim
        );
        for (idx, key) in loaded.keys.iter().enumerate() {
            // Recover writer index from the key; each thread writes exactly one
            // disjoint key so reconstruction is deterministic.
            let writer = key
                .strip_prefix("file_")
                .and_then(|s| s.strip_suffix(".ts"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or_else(|| panic!("unexpected key shape: {}", key));
            assert_eq!(
                loaded.hashes[idx],
                format!("hash_{}", writer),
                "hash for key {} doesn't match writer {}",
                key,
                writer
            );
            let off = idx * dim;
            assert!(
                (loaded.vectors[off] - writer as f32).abs() < 1e-6
                    && (loaded.vectors[off + 1] - (writer as f32 + 0.1)).abs() < 1e-6,
                "vector bytes for key {} corrupted; expected [{}, {}], got [{}, {}]",
                key,
                writer,
                writer as f32 + 0.1,
                loaded.vectors[off],
                loaded.vectors[off + 1]
            );
        }
    }

    // -- mmap_vector_store tests (zero-copy) --

    #[test]
    fn mmap_vector_store_nonexistent_returns_none() {
        let dir = TempDir::new().unwrap();
        let result = mmap_vector_store(dir.path(), "no-such").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn mmap_vector_store_round_trip() {
        let dir = TempDir::new().unwrap();

        let store = VectorStore::new(
            3,
            vec!["a.ts".to_string(), "b.ts".to_string()],
            vec!["h1".to_string(), "h2".to_string()],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );

        save_vector_store_merged(dir.path(), "zc", &store).unwrap();
        let loaded = mmap_vector_store(dir.path(), "zc").unwrap().unwrap();

        assert_eq!(loaded.count(), 2);
        assert_eq!(loaded.dims(), 3);
        assert!(loaded.has_key("a.ts"));
        assert!(loaded.has_key("b.ts"));

        let vec_a = loaded.get_vector("a.ts").unwrap();
        assert!((vec_a[0] - 1.0).abs() < 1e-6);
        assert!((vec_a[1] - 2.0).abs() < 1e-6);
        assert!((vec_a[2] - 3.0).abs() < 1e-6);

        let vec_b = loaded.get_vector("b.ts").unwrap();
        assert!((vec_b[0] - 4.0).abs() < 1e-6);
        assert!((vec_b[1] - 5.0).abs() < 1e-6);
        assert!((vec_b[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn mmap_vector_store_matches_regular_load() {
        let dir = TempDir::new().unwrap();

        let store = VectorStore::new(
            3,
            vec![
                "src/auth.ts".to_string(),
                "src/db.ts".to_string(),
                "src/api.ts".to_string(),
            ],
            vec!["h1".to_string(), "h2".to_string(), "h3".to_string()],
            vec![0.9, 0.1, 0.0, 0.0, 0.9, 0.1, 0.5, 0.5, 0.0],
        );

        save_vector_store_merged(dir.path(), "cmp", &store).unwrap();

        let regular = load_vector_store(dir.path(), "cmp").unwrap().unwrap();
        let mmap = mmap_vector_store(dir.path(), "cmp").unwrap().unwrap();

        // Same metadata
        assert_eq!(mmap.count(), regular.count());
        assert_eq!(mmap.dims(), regular.dims());

        // Same keys, hashes, and vectors
        for key in regular.keys() {
            assert!(mmap.has_key(key));
            assert_eq!(mmap.get_hash(key), regular.get_hash(key));

            let rv = regular.get_vector(key).unwrap();
            let mv = mmap.get_vector(key).unwrap();
            assert_eq!(rv.len(), mv.len());
            for (a, b) in rv.iter().zip(mv.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "vector mismatch for key '{}': regular={}, mmap={}",
                    key,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn mmap_vector_store_find_nearest_matches() {
        let dir = TempDir::new().unwrap();

        let store = VectorStore::new(
            3,
            vec![
                "src/auth.ts".to_string(),
                "src/db.ts".to_string(),
                "src/api.ts".to_string(),
            ],
            vec!["h1".to_string(), "h2".to_string(), "h3".to_string()],
            vec![0.9, 0.1, 0.0, 0.0, 0.9, 0.1, 0.5, 0.5, 0.0],
        );

        save_vector_store_merged(dir.path(), "nn", &store).unwrap();

        let regular = load_vector_store(dir.path(), "nn").unwrap().unwrap();
        let mmap = mmap_vector_store(dir.path(), "nn").unwrap().unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let regular_results = regular.find_nearest(&query, 3);
        let mmap_results = mmap.find_nearest(&query, 3);

        assert_eq!(regular_results.len(), mmap_results.len());
        for (r, m) in regular_results.iter().zip(mmap_results.iter()) {
            assert_eq!(r.0, m.0, "ordering mismatch");
            assert!(
                (r.1 - m.1).abs() < 1e-6,
                "similarity mismatch: regular={}, mmap={}",
                r.1,
                m.1
            );
        }
    }

    // -- Worktree-aware namespacing --

    #[test]
    fn cache_dir_main_checkout_unchanged() {
        // No .git → treated as main checkout (backward-compatible path).
        let dir = TempDir::new().unwrap();
        assert_eq!(cache_dir(dir.path()), dir.path().join(".mcp_data"));
    }

    #[test]
    fn cache_dir_with_git_directory_is_main_checkout() {
        // .git as directory → main checkout.
        let dir = TempDir::new().unwrap();
        fs::create_dir(dir.path().join(".git")).unwrap();
        assert_eq!(cache_dir(dir.path()), dir.path().join(".mcp_data"));
    }

    #[test]
    fn cache_dir_with_git_file_is_worktree() {
        // .git as file with `gitdir: …/worktrees/<name>` → namespaced.
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join(".git"),
            "gitdir: /tmp/main/.git/worktrees/feature-x\n",
        )
        .unwrap();
        let expected = dir
            .path()
            .join(".mcp_data")
            .join("worktrees")
            .join("feature-x");
        assert_eq!(cache_dir(dir.path()), expected);
    }

    #[test]
    fn worktree_caches_dont_collide_with_main() {
        // Critical: the same key written to a worktree must NOT clobber the
        // main checkout's cache entry of the same name.
        let main = TempDir::new().unwrap();
        let wt = TempDir::new().unwrap();

        // Mark `wt` as a worktree
        fs::write(
            wt.path().join(".git"),
            "gitdir: /tmp/main/.git/worktrees/wt-a\n",
        )
        .unwrap();

        let main_data = CacheData {
            dims: 1,
            keys: vec!["k".into()],
            hashes: vec!["main".into()],
            vectors: vec![1.0],
        };
        let wt_data = CacheData {
            dims: 1,
            keys: vec!["k".into()],
            hashes: vec!["worktree".into()],
            vectors: vec![9.0],
        };

        save_cache(main.path(), "shared", &main_data).unwrap();
        save_cache(wt.path(), "shared", &wt_data).unwrap();

        let loaded_main = load_cache(main.path(), "shared").unwrap().unwrap();
        let loaded_wt = load_cache(wt.path(), "shared").unwrap().unwrap();

        assert_eq!(loaded_main.hashes[0], "main");
        assert_eq!(loaded_wt.hashes[0], "worktree");
    }

    #[test]
    fn malformed_git_file_falls_through_to_main_checkout() {
        // A .git file without `gitdir:` or with malformed contents shouldn't
        // crash — fall back to the unprefixed cache path.
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join(".git"), "garbage\n").unwrap();
        assert_eq!(cache_dir(dir.path()), dir.path().join(".mcp_data"));
    }

    #[test]
    fn git_file_pointing_outside_worktrees_falls_through() {
        // gitdir pointing somewhere other than .git/worktrees/<name> (e.g.
        // submodule) shouldn't be treated as a worktree.
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join(".git"),
            "gitdir: /tmp/repo/.git/modules/sub\n",
        )
        .unwrap();
        assert_eq!(cache_dir(dir.path()), dir.path().join(".mcp_data"));
    }

    #[test]
    fn write_lock_paths_are_per_cache_name() {
        // Two different cache names in the same directory must use distinct
        // lock files; otherwise concurrent writes serialize unnecessarily.
        let dir = TempDir::new().unwrap();
        let lock_a = write_lock_path(dir.path(), "cache_a");
        let lock_b = write_lock_path(dir.path(), "cache_b");
        assert_ne!(lock_a, lock_b);
        assert!(
            lock_a
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("cache_a"),
            "lock filename must include the cache name, got {:?}",
            lock_a
        );
    }

    #[test]
    fn save_cache_merge_still_preserves_disk_keys() {
        // Regression guard: the merging save_cache must still preserve disk-side
        // keys absent from the incoming snapshot (concurrent-writer safety).
        let dir = TempDir::new().unwrap();

        let disk = CacheData {
            dims: 2,
            keys: vec!["disk-only".into(), "shared".into()],
            hashes: vec!["hd".into(), "hs".into()],
            vectors: vec![1.0, 1.1, 2.0, 2.1],
        };
        save_cache(dir.path(), "merge-guard", &disk).unwrap();

        let incoming = CacheData {
            dims: 2,
            keys: vec!["shared".into()],
            hashes: vec!["hs-new".into()],
            vectors: vec![3.0, 3.1],
        };
        save_cache(dir.path(), "merge-guard", &incoming).unwrap();

        let loaded = load_cache(dir.path(), "merge-guard").unwrap().unwrap();
        assert_eq!(
            loaded.keys.len(),
            2,
            "merging save_cache must preserve disk-only keys; got {:?}",
            loaded.keys
        );
        assert!(
            loaded.keys.iter().any(|k| k == "disk-only"),
            "disk-only key must survive a merging save"
        );
    }

    #[test]
    fn worktree_name_rejects_path_traversal_pointers() {
        // A malicious gitdir pointer ending in `..` or `.` would otherwise
        // collapse the worktree's cache into the main checkout's directory.
        let dir = TempDir::new().unwrap();

        // gitdir ending in ".." → reject
        fs::write(
            dir.path().join(".git"),
            "gitdir: /tmp/repo/.git/worktrees/..\n",
        )
        .unwrap();
        assert_eq!(cache_dir(dir.path()), dir.path().join(".mcp_data"));

        // gitdir ending in "." → reject
        fs::write(
            dir.path().join(".git"),
            "gitdir: /tmp/repo/.git/worktrees/.\n",
        )
        .unwrap();
        assert_eq!(cache_dir(dir.path()), dir.path().join(".mcp_data"));
    }

    #[test]
    fn mmap_vector_store_to_cache_works() {
        let dir = TempDir::new().unwrap();

        let store = VectorStore::new(
            2,
            vec!["x.ts".to_string()],
            vec!["hx".to_string()],
            vec![0.5, 0.7],
        );

        save_vector_store_merged(dir.path(), "cache-test", &store).unwrap();
        let loaded = mmap_vector_store(dir.path(), "cache-test")
            .unwrap()
            .unwrap();

        let cache = loaded.to_cache();
        assert_eq!(cache.len(), 1);
        let entry = &cache["x.ts"];
        assert_eq!(entry.hash, "hx");
        assert!((entry.vector[0] - 0.5).abs() < 1e-6);
        assert!((entry.vector[1] - 0.7).abs() < 1e-6);
    }

    // TEST-001..003: protect merge_cache_data + load_cache contracts that
    // were silently breaking before the Round 3 fixes (dim mismatch dropped
    // disk; truncated vectors panicked; corrupted payloads bricked the
    // cache instead of surfacing as a load error).

    #[test]
    fn merge_drops_disk_when_dims_disagree() {
        // Disk and incoming come from different embedding models — the
        // vector spaces are incompatible, so the only safe merge is to
        // discard the disk side and keep `incoming` whole.
        let disk = CacheData {
            dims: 4,
            keys: vec!["src/a.ts".to_string()],
            hashes: vec!["ha".to_string()],
            vectors: vec![1.0, 2.0, 3.0, 4.0],
        };
        let incoming = CacheData {
            dims: 3,
            keys: vec!["src/b.ts".to_string()],
            hashes: vec!["hb".to_string()],
            vectors: vec![0.1, 0.2, 0.3],
        };

        let merged = merge_cache_data(disk, &incoming);

        assert_eq!(merged.dims, 3);
        assert_eq!(merged.keys, vec!["src/b.ts".to_string()]);
        assert_eq!(merged.hashes, vec!["hb".to_string()]);
        assert_eq!(merged.vectors, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn merge_skips_entries_with_truncated_vector_buffer() {
        // disk advertises two keys at dims=3 but only carries one vector's
        // worth of floats. The bounds-check inside merge must skip the
        // truncated entry instead of panicking on the slice.
        let disk = CacheData {
            dims: 3,
            keys: vec!["src/keep.ts".to_string(), "src/truncated.ts".to_string()],
            hashes: vec!["hk".to_string(), "ht".to_string()],
            vectors: vec![0.1, 0.2, 0.3], // only enough for the first key
        };
        let incoming = CacheData {
            dims: 3,
            keys: vec!["src/new.ts".to_string()],
            hashes: vec!["hn".to_string()],
            vectors: vec![0.4, 0.5, 0.6],
        };

        let merged = merge_cache_data(disk, &incoming);

        assert_eq!(merged.dims, 3);
        let mut by_key: HashMap<&str, (&str, &[f32])> = HashMap::new();
        for (i, k) in merged.keys.iter().enumerate() {
            let off = i * merged.dims as usize;
            by_key.insert(
                k.as_str(),
                (
                    merged.hashes[i].as_str(),
                    &merged.vectors[off..off + merged.dims as usize],
                ),
            );
        }
        assert!(by_key.contains_key("src/keep.ts"));
        assert!(by_key.contains_key("src/new.ts"));
        assert!(
            !by_key.contains_key("src/truncated.ts"),
            "truncated entry must be dropped, not retained with garbage data"
        );
        let (kh, kv) = by_key["src/keep.ts"];
        assert_eq!(kh, "hk");
        assert_eq!(kv, &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn load_returns_cache_error_on_truncated_payload() {
        // Header says version OK, but the rkyv body has been truncated.
        // load_cache must return Err(Cache(...)) so save_cache's Round 3
        // fallback path can decide to overwrite instead of bricking — the
        // pre-fix behavior bubbled the error up and killed the writer.
        let dir = TempDir::new().unwrap();
        ensure_cache_dir(dir.path()).unwrap();

        // Write a valid header + a single byte of "rkyv body" — definitely
        // not a valid serialized CacheData.
        let path = cache_path(dir.path(), "truncated");
        let mut bad = vec![0u8; HEADER_SIZE + 1];
        bad[0] = CACHE_VERSION;
        fs::write(&path, &bad).unwrap();

        let result = load_cache(dir.path(), "truncated");
        match result {
            Err(ContextPlusError::Cache(msg)) => {
                assert!(
                    msg.contains("rkyv") || msg.to_lowercase().contains("deserial"),
                    "expected rkyv deserialize error, got: {msg}"
                );
            }
            other => panic!("expected Err(Cache(_)), got {:?}", other),
        }
    }

    // -- sweep_excluded_keys / load_cache hygiene sweep --

    /// `sweep_excluded_keys` must remove entries whose keys contain a dot-prefixed
    /// path segment (the walker's hidden-file rule), while leaving valid paths
    /// untouched.
    #[test]
    fn sweep_excluded_keys_removes_dot_segment_paths() {
        let mut data = CacheData {
            dims: 2,
            keys: vec![
                "src/main.rs".into(),
                ".claude/worktrees/agent-abc/src/main.rs".into(),
                ".claude/worktrees/agent-xyz/packages/foo.ts".into(),
                "packages/lib/index.ts".into(),
            ],
            hashes: vec!["h1".into(), "h2".into(), "h3".into(), "h4".into()],
            vectors: vec![
                1.0, 1.1, // src/main.rs
                2.0, 2.1, // .claude/worktrees/agent-abc/...
                3.0, 3.1, // .claude/worktrees/agent-xyz/...
                4.0, 4.1, // packages/lib/index.ts
            ],
        };

        let removed = data.sweep_excluded_keys();

        assert_eq!(removed, 2, "two worktree entries should be removed");
        assert_eq!(data.keys.len(), 2);
        assert!(
            data.keys.iter().any(|k| k == "src/main.rs"),
            "src/main.rs must survive"
        );
        assert!(
            data.keys.iter().any(|k| k == "packages/lib/index.ts"),
            "packages/lib/index.ts must survive"
        );
        assert!(
            !data.keys.iter().any(|k| k.contains(".claude")),
            ".claude/… paths must be removed"
        );
        // Vector data must be consistent with remaining keys
        assert_eq!(
            data.vectors.len(),
            data.keys.len() * data.dims as usize,
            "vectors must stay aligned after sweep"
        );
    }

    /// `sweep_excluded_keys` must be a no-op when all paths are valid.
    #[test]
    fn sweep_excluded_keys_noop_on_clean_data() {
        let mut data = CacheData {
            dims: 2,
            keys: vec!["src/a.rs".into(), "lib/b.ts".into()],
            hashes: vec!["ha".into(), "hb".into()],
            vectors: vec![1.0, 1.1, 2.0, 2.1],
        };

        let removed = data.sweep_excluded_keys();

        assert_eq!(removed, 0);
        assert_eq!(data.keys.len(), 2);
        assert_eq!(data.vectors.len(), 4);
    }

    /// `sweep_excluded_keys` on empty CacheData must not panic.
    #[test]
    fn sweep_excluded_keys_empty_data_is_noop() {
        let mut data = CacheData {
            dims: 0,
            keys: vec![],
            hashes: vec![],
            vectors: vec![],
        };
        assert_eq!(data.sweep_excluded_keys(), 0);
    }

    /// Integration: `load_cache` must automatically drop `.claude/worktrees/…`
    /// entries that were present in the on-disk file from a previous (buggy)
    /// indexing run — the caller must never see them in the returned CacheData.
    #[test]
    fn load_cache_sweeps_worktree_entries_from_disk() {
        let dir = TempDir::new().unwrap();

        // Write a cache file that contains a mix of valid and worktree paths,
        // simulating a cache produced by an older code version.
        let dirty = CacheData {
            dims: 2,
            keys: vec![
                "src/server.rs".into(),
                ".claude/worktrees/agent-a30a5dbd/packages/libs/types/generated/foo.ts".into(),
                ".claude/worktrees/fix-121-lint/src/lib.rs".into(),
                "packages/api/index.ts".into(),
            ],
            hashes: vec!["h1".into(), "h2".into(), "h3".into(), "h4".into()],
            vectors: vec![
                0.1, 0.2, // src/server.rs
                0.3, 0.4, // .claude/worktrees/agent-a30a5dbd/…
                0.5, 0.6, // .claude/worktrees/fix-121-lint/…
                0.7, 0.8, // packages/api/index.ts
            ],
        };
        // The hygiene sweep only fires on `load_cache`; `save_cache` writes the
        // payload as-is into a fresh dir (empty disk → no merge).
        save_cache(dir.path(), "test-dirty", &dirty).unwrap();

        // Now load through the public API — sweep must have fired.
        let loaded = load_cache(dir.path(), "test-dirty").unwrap().unwrap();

        assert_eq!(
            loaded.keys.len(),
            2,
            "only the two valid keys must survive; got {:?}",
            loaded.keys
        );
        assert!(
            loaded.keys.iter().any(|k| k == "src/server.rs"),
            "src/server.rs must be present"
        );
        assert!(
            loaded.keys.iter().any(|k| k == "packages/api/index.ts"),
            "packages/api/index.ts must be present"
        );
        assert!(
            !loaded.keys.iter().any(|k| k.contains(".claude")),
            ".claude/worktrees/… entries must be swept on load; remaining keys={:?}",
            loaded.keys
        );
        // Vector buffer must be consistent with the remaining key count.
        assert_eq!(
            loaded.vectors.len(),
            loaded.keys.len() * loaded.dims as usize,
            "vector buffer must stay aligned after sweep"
        );
    }

    // -- Query cache persistence tests --

    /// Round-trip: queries saved to disk survive a reload.
    #[test]
    fn query_cache_round_trip() {
        let dir = TempDir::new().unwrap();
        let model = "test-model";
        let entries = vec![
            ("what is auth".to_string(), vec![0.1f32, 0.2, 0.3]),
            ("database connection".to_string(), vec![0.4, 0.5, 0.6]),
        ];

        save_query_cache(dir.path(), model, &entries).unwrap();
        let loaded = load_query_cache(dir.path(), model).unwrap();

        assert_eq!(loaded.len(), 2);
        let keys: Vec<&str> = loaded.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"what is auth"));
        assert!(keys.contains(&"database connection"));

        let v = loaded
            .iter()
            .find(|(k, _)| k == "what is auth")
            .map(|(_, v)| v)
            .unwrap();
        assert!((v[0] - 0.1).abs() < 1e-6);
        assert!((v[2] - 0.3).abs() < 1e-6);
    }

    /// Missing file → empty result (no error).
    #[test]
    fn query_cache_missing_file_returns_empty() {
        let dir = TempDir::new().unwrap();
        let result = load_query_cache(dir.path(), "no-model").unwrap();
        assert!(result.is_empty());
    }

    /// Cap enforcement: saving more than QUERY_CACHE_MAX entries keeps only the
    /// MRU tail (size == QUERY_CACHE_MAX after the write).
    #[test]
    fn query_cache_cap_enforcement() {
        let dir = TempDir::new().unwrap();
        let model = "cap-model";
        let n = QUERY_CACHE_MAX + 50;
        let entries: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| (format!("query_{i}"), vec![i as f32, i as f32 + 0.1]))
            .collect();

        save_query_cache(dir.path(), model, &entries).unwrap();
        let loaded = load_query_cache(dir.path(), model).unwrap();

        assert_eq!(
            loaded.len(),
            QUERY_CACHE_MAX,
            "persisted entries must be capped at QUERY_CACHE_MAX"
        );
        // The MRU tail (last QUERY_CACHE_MAX entries) must survive.
        let last_key = format!("query_{}", n - 1);
        assert!(
            loaded.iter().any(|(k, _)| k == &last_key),
            "MRU entry '{}' must be present after cap enforcement",
            last_key
        );
    }

    /// Different model → different file on disk (no cross-contamination).
    #[test]
    fn query_cache_model_in_filename() {
        let dir = TempDir::new().unwrap();
        let entries_a = vec![("qa".to_string(), vec![1.0f32, 0.0])];
        let entries_b = vec![("qb".to_string(), vec![0.0f32, 1.0])];

        save_query_cache(dir.path(), "model-a", &entries_a).unwrap();
        save_query_cache(dir.path(), "model-b", &entries_b).unwrap();

        let loaded_a = load_query_cache(dir.path(), "model-a").unwrap();
        let loaded_b = load_query_cache(dir.path(), "model-b").unwrap();

        assert_eq!(loaded_a.len(), 1);
        assert_eq!(loaded_a[0].0, "qa");

        assert_eq!(loaded_b.len(), 1);
        assert_eq!(loaded_b[0].0, "qb");
    }

    // -- save_vector_store_merged tests (live MCP runtime path) --

    #[test]
    fn merged_save_persists_all_keys_when_disk_is_empty() {
        // Empty disk + a fresh in-memory snapshot: every key must end up on disk.
        let dir = TempDir::new().unwrap();
        let store = VectorStore::new(
            2,
            vec!["src/a.rs".into(), "src/b.rs".into()],
            vec!["ha".into(), "hb".into()],
            vec![1.0, 2.0, 3.0, 4.0],
        );

        save_vector_store_merged(dir.path(), "merged-empty", &store).unwrap();

        let loaded = load_vector_store(dir.path(), "merged-empty")
            .unwrap()
            .unwrap();
        assert_eq!(loaded.count(), 2);
        assert!(loaded.has_key("src/a.rs"));
        assert!(loaded.has_key("src/b.rs"));
    }

    #[test]
    fn merged_save_preserves_disk_keys_unknown_to_caller() {
        // Reproduces the regression: warmup writes A,B,C to disk; the live MCP
        // then does a partial save with only B,D. After the merged save the
        // file must contain {A,B,C,D}, with B's vector taken from the in-memory
        // (incoming) snapshot.
        let dir = TempDir::new().unwrap();

        let warmup_disk = CacheData {
            dims: 2,
            keys: vec!["A".into(), "B".into(), "C".into()],
            hashes: vec!["hA-old".into(), "hB-old".into(), "hC-old".into()],
            vectors: vec![1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        };
        save_cache(dir.path(), "merged-runtime", &warmup_disk).unwrap();

        // Live MCP only knows B and D — its in-memory store is missing A and C.
        let runtime_store = VectorStore::new(
            2,
            vec!["B".into(), "D".into()],
            vec!["hB-new".into(), "hD-new".into()],
            vec![20.0, 21.0, 4.0, 4.1],
        );
        save_vector_store_merged(dir.path(), "merged-runtime", &runtime_store).unwrap();

        let loaded = load_cache(dir.path(), "merged-runtime").unwrap().unwrap();
        let key_set: std::collections::HashSet<_> = loaded.keys.iter().cloned().collect();
        assert_eq!(
            key_set,
            ["A", "B", "C", "D"]
                .iter()
                .map(|s| (*s).to_string())
                .collect::<std::collections::HashSet<_>>(),
            "merged save dropped disk-only keys; got {:?}",
            loaded.keys
        );

        // B's vector must be the incoming one (20.0, 21.0), not the stale disk
        // value (2.0, 2.1).
        let b_idx = loaded.keys.iter().position(|k| k == "B").unwrap();
        assert_eq!(loaded.hashes[b_idx], "hB-new");
        assert!((loaded.vectors[b_idx * 2] - 20.0).abs() < 1e-6);
        assert!((loaded.vectors[b_idx * 2 + 1] - 21.0).abs() < 1e-6);

        // A and C must keep their original disk values.
        let a_idx = loaded.keys.iter().position(|k| k == "A").unwrap();
        assert_eq!(loaded.hashes[a_idx], "hA-old");
        assert!((loaded.vectors[a_idx * 2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn merged_save_concurrent_writers_union_all_keys() {
        // N threads each call save_vector_store_merged with a unique key.
        // The fd-lock + load-merge-save sequence must produce a final cache
        // containing every thread's key, with no losses.
        use std::collections::HashSet;
        use std::sync::Arc as StdArc;
        use std::sync::Barrier;

        const N: usize = 4;
        let dir = TempDir::new().unwrap();
        let root = dir.path().to_path_buf();
        ensure_cache_dir(&root).unwrap();

        let barrier = StdArc::new(Barrier::new(N));

        std::thread::scope(|s| {
            for i in 0..N {
                let root = root.clone();
                let barrier = StdArc::clone(&barrier);
                s.spawn(move || {
                    let key = format!("vs_file_{}.rs", i);
                    let store = VectorStore::new(
                        2,
                        vec![key],
                        vec![format!("hash_{}", i)],
                        vec![i as f32, i as f32 + 0.5],
                    );
                    barrier.wait();
                    save_vector_store_merged(&root, "merged-race", &store).unwrap();
                });
            }
        });

        let loaded = load_vector_store(&root, "merged-race").unwrap().unwrap();
        let key_set: HashSet<_> = loaded.keys().iter().cloned().collect();
        for i in 0..N {
            let expected = format!("vs_file_{}.rs", i);
            assert!(
                key_set.contains(&expected),
                "save_vector_store_merged lost concurrent writer's key '{}'; got keys={:?}",
                expected,
                loaded.keys()
            );
        }
        assert_eq!(loaded.count(), N);
    }

    /// Disk has {A,B,C}; in-memory has {B}; deletions=[C].
    /// Post-state must be {A,B} — A survives (disk-only key preserved), B is
    /// updated from incoming, C is evicted.
    #[test]
    fn merged_save_honors_deletion_list() {
        let dir = TempDir::new().unwrap();

        let disk = CacheData {
            dims: 2,
            keys: vec!["A".into(), "B".into(), "C".into()],
            hashes: vec!["hA".into(), "hB-old".into(), "hC".into()],
            vectors: vec![1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        };
        save_cache(dir.path(), "del", &disk).unwrap();

        let in_mem = VectorStore::new(2, vec!["B".into()], vec!["hB-new".into()], vec![20.0, 21.0]);
        save_vector_store_merged_with_deletions(dir.path(), "del", &in_mem, &["C".to_string()])
            .unwrap();

        let loaded = load_cache(dir.path(), "del").unwrap().unwrap();
        let key_set: std::collections::HashSet<_> = loaded.keys.iter().cloned().collect();
        assert_eq!(
            key_set,
            ["A", "B"]
                .iter()
                .map(|s| (*s).to_string())
                .collect::<std::collections::HashSet<_>>(),
            "deletion list must remove C while preserving A and B; got {:?}",
            loaded.keys
        );

        // B must have the incoming hash (incoming wins on shared keys).
        let b_idx = loaded.keys.iter().position(|k| k == "B").unwrap();
        assert_eq!(loaded.hashes[b_idx], "hB-new");
        assert!((loaded.vectors[b_idx * 2] - 20.0).abs() < 1e-6);

        // A retains disk values.
        let a_idx = loaded.keys.iter().position(|k| k == "A").unwrap();
        assert_eq!(loaded.hashes[a_idx], "hA");

        // Vector buffer must stay aligned with the surviving key count.
        assert_eq!(
            loaded.vectors.len(),
            loaded.keys.len() * loaded.dims as usize
        );
    }

    /// Garbage on disk → save must rotate the corrupt file aside as
    /// `<path>.corrupt.<unix_ts>` and write the incoming snapshot fresh.
    #[test]
    fn merged_save_rotates_corrupt_file_aside() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let dir = TempDir::new().unwrap();
        ensure_cache_dir(dir.path()).unwrap();

        // Plant garbage at the cache path.
        let cache_file = cache_path(dir.path(), "corrupt-test");
        fs::write(&cache_file, b"\x00not a valid rkyv payload\x00\x01\x02\x03").unwrap();

        let before_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let in_mem = VectorStore::new(
            2,
            vec!["new-key".into()],
            vec!["h-new".into()],
            vec![9.0, 9.5],
        );
        save_vector_store_merged(dir.path(), "corrupt-test", &in_mem).unwrap();

        // The new cache file must contain the in-memory entries — clean state.
        let loaded = load_cache(dir.path(), "corrupt-test").unwrap().unwrap();
        assert_eq!(loaded.keys, vec!["new-key".to_string()]);
        assert_eq!(loaded.hashes, vec!["h-new".to_string()]);
        assert!((loaded.vectors[0] - 9.0).abs() < 1e-6);

        // A rotated `<file>.corrupt.<ts>` sibling must exist.
        let parent = cache_file.parent().unwrap();
        let mut found_rotated = None;
        for entry in fs::read_dir(parent).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("corrupt-test.rkyv.corrupt.") {
                found_rotated = Some((entry.path(), name));
                break;
            }
        }
        let (rotated_path, rotated_name) =
            found_rotated.expect("corrupt file must be rotated aside as <path>.corrupt.<ts>");

        // The trailing component after `.corrupt.` must be a unix timestamp
        // close to "now" (sanity check — within a generous window).
        let ts_str = rotated_name
            .rsplit('.')
            .next()
            .expect("rotated name must have a timestamp suffix");
        let ts: u64 = ts_str
            .parse()
            .expect("rotated suffix must be a unix timestamp");
        let after_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        assert!(
            ts >= before_ts.saturating_sub(2) && ts <= after_ts + 2,
            "rotation timestamp {ts} out of expected window [{before_ts}, {after_ts}]"
        );

        // The rotated bytes must equal the original garbage we planted —
        // the operator can recover from this file.
        let rotated_bytes = fs::read(&rotated_path).unwrap();
        assert_eq!(
            rotated_bytes, b"\x00not a valid rkyv payload\x00\x01\x02\x03",
            "rotated file must preserve the original (corrupt) bytes"
        );
    }

    /// Production-grade concurrency: K=200 disk keys + N=8 threads each holding
    /// a small subset (10 disk keys + 50 thread-unique keys). Every disk key
    /// must survive and every thread's unique keys must survive — no key losses.
    /// Last-writer-wins on shared keys is acceptable; we only assert the union.
    #[test]
    fn merged_save_large_disk_small_memory_subsets_under_concurrency() {
        use std::collections::HashSet;
        use std::sync::Arc as StdArc;
        use std::sync::Barrier;

        const K: usize = 200; // disk keys
        const N: usize = 8; // threads
        const SHARED_PER_THREAD: usize = 10; // each thread overlaps 10 disk keys
        const UNIQUE_PER_THREAD: usize = 50; // and adds 50 unique keys

        let dir = TempDir::new().unwrap();
        let root = dir.path().to_path_buf();
        ensure_cache_dir(&root).unwrap();

        // Pre-seed disk with K random keys via a single merged save.
        let disk_keys: Vec<String> = (0..K).map(|i| format!("disk_key_{:04}", i)).collect();
        let disk_hashes: Vec<String> = (0..K).map(|i| format!("disk_hash_{:04}", i)).collect();
        let mut disk_vectors: Vec<f32> = Vec::with_capacity(K * 2);
        for i in 0..K {
            disk_vectors.push(i as f32);
            disk_vectors.push(i as f32 + 0.5);
        }
        let disk_store = VectorStore::new(2, disk_keys.clone(), disk_hashes.clone(), disk_vectors);
        save_vector_store_merged(&root, "big-race", &disk_store).unwrap();

        let barrier = StdArc::new(Barrier::new(N));

        std::thread::scope(|s| {
            for t in 0..N {
                let root = root.clone();
                let barrier = StdArc::clone(&barrier);
                let disk_keys = disk_keys.clone();
                s.spawn(move || {
                    // Deterministic per-thread subset: take SHARED_PER_THREAD
                    // disk keys at offset t*SHARED_PER_THREAD (wrapping), and
                    // UNIQUE_PER_THREAD thread-unique keys.
                    let mut keys: Vec<String> =
                        Vec::with_capacity(SHARED_PER_THREAD + UNIQUE_PER_THREAD);
                    let mut hashes: Vec<String> = Vec::with_capacity(keys.capacity());
                    let mut vectors: Vec<f32> = Vec::with_capacity(keys.capacity() * 2);
                    for i in 0..SHARED_PER_THREAD {
                        let idx = (t * SHARED_PER_THREAD + i) % disk_keys.len();
                        keys.push(disk_keys[idx].clone());
                        hashes.push(format!("thread_{}_overwrite_{}", t, i));
                        vectors.push(1000.0 + t as f32);
                        vectors.push(1000.0 + t as f32 + 0.5);
                    }
                    for u in 0..UNIQUE_PER_THREAD {
                        keys.push(format!("thread_{}_unique_{:04}", t, u));
                        hashes.push(format!("thread_{}_unique_hash_{:04}", t, u));
                        vectors.push(t as f32 * 100.0 + u as f32);
                        vectors.push(t as f32 * 100.0 + u as f32 + 0.5);
                    }
                    let store = VectorStore::new(2, keys, hashes, vectors);
                    barrier.wait();
                    save_vector_store_merged(&root, "big-race", &store).unwrap();
                });
            }
        });

        let loaded = load_vector_store(&root, "big-race").unwrap().unwrap();
        let final_keys: HashSet<String> = loaded.keys().iter().cloned().collect();

        // Every disk key must survive.
        for dk in &disk_keys {
            assert!(
                final_keys.contains(dk),
                "disk key '{}' was lost under concurrent merged saves",
                dk
            );
        }

        // Every thread's unique keys must survive.
        for t in 0..N {
            for u in 0..UNIQUE_PER_THREAD {
                let key = format!("thread_{}_unique_{:04}", t, u);
                assert!(
                    final_keys.contains(&key),
                    "thread {}'s unique key '{}' was lost under concurrent merged saves",
                    t,
                    key
                );
            }
        }

        // Total count must be at least disk + thread-unique = K + N*UNIQUE_PER_THREAD.
        assert!(
            loaded.count() >= K + N * UNIQUE_PER_THREAD,
            "expected at least {} keys, got {}",
            K + N * UNIQUE_PER_THREAD,
            loaded.count()
        );
    }
}
