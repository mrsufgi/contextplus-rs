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
    let merged = match load_cache(root_dir, name)? {
        Some(disk) => merge_cache_data(disk, data),
        None => clone_cache_data(data),
    };

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

    let data = rkyv::from_bytes::<CacheData, rkyv::rancor::Error>(data_bytes)
        .map_err(|e| ContextPlusError::Cache(format!("rkyv deserialize: {}", e)))?;

    Ok(Some(data))
}

/// Load directly into a VectorStore.
pub fn load_vector_store(root_dir: &Path, name: &str) -> Result<Option<VectorStore>> {
    match load_cache(root_dir, name)? {
        Some(data) => Ok(Some(data.to_store())),
        None => Ok(None),
    }
}

/// Save a VectorStore to disk.
pub fn save_vector_store(root_dir: &Path, name: &str, store: &VectorStore) -> Result<()> {
    let data = CacheData::from_store(store);
    save_cache(root_dir, name, &data)
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

        save_vector_store(dir.path(), "vectors", &store).unwrap();
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

        save_vector_store(dir.path(), "zc", &store).unwrap();
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

        save_vector_store(dir.path(), "cmp", &store).unwrap();

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

        save_vector_store(dir.path(), "nn", &store).unwrap();

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

        save_vector_store(dir.path(), "cache-test", &store).unwrap();
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
}
