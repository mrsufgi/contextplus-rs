use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
const CACHE_VERSION: u8 = 1;
/// Header size padded to 16-byte alignment so rkyv data starts aligned.
const HEADER_SIZE: usize = 16;

fn cache_dir(root_dir: &Path) -> PathBuf {
    root_dir.join(CACHE_DIR)
}

fn cache_path(root_dir: &Path, name: &str) -> PathBuf {
    cache_dir(root_dir).join(format!("{}.rkyv", name))
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
        let mut keys = Vec::with_capacity(n);
        let mut hashes = Vec::with_capacity(n);
        let mut vectors = Vec::with_capacity(n * dims as usize);
        for (key, entry) in cache {
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
/// Uses atomic write: write to a unique temp file then rename.
pub fn save_cache(root_dir: &Path, name: &str, data: &CacheData) -> Result<()> {
    ensure_cache_dir(root_dir)?;

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(data)
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

    Ok(())
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

    #[test]
    fn atomic_write_survives() {
        let dir = TempDir::new().unwrap();
        let data = make_test_data();

        // Save twice — second should overwrite cleanly
        save_cache(dir.path(), "atomic", &data).unwrap();
        save_cache(dir.path(), "atomic", &data).unwrap();

        let loaded = load_cache(dir.path(), "atomic").unwrap().unwrap();
        assert_eq!(loaded.keys, data.keys);

        // No temp files should linger in the cache directory
        let cache = cache_dir(dir.path());
        let lingering: Vec<_> = fs::read_dir(&cache)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "tmp"))
            .collect();
        assert!(lingering.is_empty(), "temp files should not linger: {:?}", lingering);
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
