//! Content-addressed blob store for chunk embeddings.
//!
//! # Layout
//!
//! ```text
//! .mcp_data/
//!   cas/<model_slug>/<ab>/<cdef...>.emb   // BLAKE3-keyed embeddings
//!   refs/<ref_id>/manifest.rkyv           // (file_path, chunk_idx) → chunk_hash
//!   refs/<ref_id>/parent                  // parent's ref_id as hex, or empty
//!   migration_in_progress                 // marker written before drop-rebuild
//! ```
//!
//! # Design decisions
//!
//! - **BLAKE3** is used instead of SHA-256: ~3× faster on the same hardware with
//!   the same collision-resistance guarantees for this use-case. Keys are 32-byte
//!   hex strings (64 chars) split into `<ab>/<remaining>` directory shards to
//!   keep directory entry counts manageable on large corpora.
//!
//! - **CAS blobs are append-only**. A blob written for a given (model, hash) is
//!   never mutated. Two workers writing the same blob simultaneously produce the
//!   same bytes; the last `rename` wins harmlessly.
//!
//! - **Per-ref manifests** carry only the delta vs. parent. A forked ref starts
//!   with an empty manifest; on first walk, only files whose BLAKE3 chunk hashes
//!   differ from the parent's manifest entries are embedded and recorded here.
//!
//! - **Parent chain lookups** chain through `parent` files on disk until a
//!   manifest hit is found or we reach the root (no parent).
//!
//! # Migration from old path-keyed `.rkyv` format
//!
//! On daemon start, [`CasStore::detect_and_migrate_legacy`] checks for old-format `.rkyv` files
//! in `.mcp_data/` with the `cas/` directory absent. When detected, all legacy
//! files are deleted, a `migration_in_progress` marker is written, and embeddings
//! rebuild fresh. The marker makes the operation idempotent on crash.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use rkyv::{Archive, Deserialize, Serialize};

use crate::cache::rkyv_store::model_slug;
use crate::error::{ContextPlusError, Result};

// ---------------------------------------------------------------------------
// ChunkHash
// ---------------------------------------------------------------------------

/// 32-byte BLAKE3 content hash of a text chunk.
///
/// Serialized as a 64-character lowercase hex string on disk and in manifests
/// so that it survives any future schema changes without binary corruption.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct ChunkHash(pub [u8; 32]);

impl ChunkHash {
    /// Compute the BLAKE3 hash of `text`.
    pub fn of(text: &str) -> Self {
        let hash = blake3::hash(text.as_bytes());
        ChunkHash(*hash.as_bytes())
    }

    /// Return the two-byte directory shard prefix (first byte, hex).
    pub fn shard(&self) -> String {
        format!("{:02x}", self.0[0])
    }

    /// Return the remaining 31-byte suffix as a hex string (62 chars).
    pub fn suffix(&self) -> String {
        hex_encode(&self.0[1..])
    }

    /// Full 64-char hex representation.
    pub fn to_hex(&self) -> String {
        hex_encode(&self.0)
    }

    /// Parse from a 64-char hex string.
    pub fn from_hex(s: &str) -> Option<Self> {
        let bytes = hex_decode(s)?;
        if bytes.len() != 32 {
            return None;
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Some(ChunkHash(arr))
    }
}

impl fmt::Display for ChunkHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/// Key into the per-ref manifest: (repo-relative file path, zero-based chunk index).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct ChunkKey {
    /// Repo-relative path with forward slashes (e.g. `src/main.rs`).
    pub path: String,
    /// Zero-based chunk index within the file.
    pub chunk_idx: u32,
}

impl ChunkKey {
    pub fn new(path: impl Into<String>, chunk_idx: u32) -> Self {
        ChunkKey {
            path: path.into(),
            chunk_idx,
        }
    }
}

/// Serializable manifest: maps `(path, chunk_idx)` → `ChunkHash`.
///
/// This is the *delta* vs. the parent ref. Unchanged files (same BLAKE3 hashes)
/// are not recorded here — they are found by walking up the parent chain.
#[derive(Debug, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug))]
pub struct Manifest {
    /// Parallel arrays: `keys[i]` → `hashes[i]`.
    pub keys: Vec<ChunkKey>,
    pub hashes: Vec<ChunkHash>,
}

impl Manifest {
    pub fn new() -> Self {
        Manifest {
            keys: Vec::new(),
            hashes: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Build an in-memory lookup map from the manifest's parallel arrays.
    pub fn to_map(&self) -> HashMap<ChunkKey, ChunkHash> {
        self.keys
            .iter()
            .zip(self.hashes.iter())
            .map(|(k, h)| (k.clone(), h.clone()))
            .collect()
    }

    /// Insert or overwrite a (key → hash) entry.
    pub fn upsert(&mut self, key: ChunkKey, hash: ChunkHash) {
        if let Some(pos) = self.keys.iter().position(|k| k == &key) {
            self.hashes[pos] = hash;
        } else {
            self.keys.push(key);
            self.hashes.push(hash);
        }
    }
}

impl Default for Manifest {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// On-disk layout helpers
// ---------------------------------------------------------------------------

const CAS_DIR: &str = "cas";
const REFS_DIR: &str = "refs";
const MANIFEST_FILE: &str = "manifest.rkyv";
const PARENT_FILE: &str = "parent";
const MIGRATION_MARKER: &str = "migration_in_progress";

/// Manifest header size (same layout as rkyv_store's HEADER_SIZE).
const MANIFEST_HEADER_SIZE: usize = 16;
const MANIFEST_VERSION: u8 = 1;

fn cas_root(mcp_data: &Path, model: &str) -> PathBuf {
    mcp_data.join(CAS_DIR).join(model_slug(model))
}

fn blob_path(mcp_data: &Path, model: &str, hash: &ChunkHash) -> PathBuf {
    cas_root(mcp_data, model)
        .join(hash.shard())
        .join(format!("{}.emb", hash.suffix()))
}

fn ref_dir(mcp_data: &Path, ref_id_hex: &str) -> PathBuf {
    mcp_data.join(REFS_DIR).join(ref_id_hex)
}

fn manifest_path(mcp_data: &Path, ref_id_hex: &str) -> PathBuf {
    ref_dir(mcp_data, ref_id_hex).join(MANIFEST_FILE)
}

fn parent_path(mcp_data: &Path, ref_id_hex: &str) -> PathBuf {
    ref_dir(mcp_data, ref_id_hex).join(PARENT_FILE)
}

// ---------------------------------------------------------------------------
// CasStore
// ---------------------------------------------------------------------------

/// The content-addressed blob store for a daemon instance.
///
/// Lives under `<primary>/.mcp_data/` and is shared across all refs.
/// Thread-safe: all methods take `&self` and perform atomic disk operations.
pub struct CasStore {
    /// Path to the `.mcp_data` directory (primary worktree's).
    mcp_data: PathBuf,
    /// Embedding model name (used for shard directory naming).
    model: String,
}

impl CasStore {
    pub fn new(mcp_data: PathBuf, model: impl Into<String>) -> Self {
        CasStore {
            mcp_data,
            model: model.into(),
        }
    }

    /// Check whether a CAS blob exists for the given chunk hash.
    pub fn blob_exists(&self, hash: &ChunkHash) -> bool {
        blob_path(&self.mcp_data, &self.model, hash).exists()
    }

    /// Read the embedding vector from the CAS blob.
    ///
    /// Returns `None` if the blob does not exist.
    /// Returns `Err` on I/O or parse failure.
    pub fn read_blob(&self, hash: &ChunkHash) -> Result<Option<Vec<f32>>> {
        let path = blob_path(&self.mcp_data, &self.model, hash);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path)?;
        if bytes.len() % 4 != 0 {
            return Err(ContextPlusError::Cache(format!(
                "CAS blob {} has non-f32-aligned length {}",
                hash,
                bytes.len()
            )));
        }
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(Some(floats))
    }

    /// Write an embedding vector to the CAS blob for `hash`.
    ///
    /// Uses an atomic `tmp → rename` pattern so a concurrent write of the
    /// same blob (identical bytes) is harmless.
    pub fn write_blob(&self, hash: &ChunkHash, vector: &[f32]) -> Result<()> {
        let path = blob_path(&self.mcp_data, &self.model, hash);

        // Fast path: blob already on disk — skip write (CAS = immutable).
        if path.exists() {
            return Ok(());
        }

        // Ensure shard directory exists.
        let shard_dir = path.parent().expect("blob path has parent");
        std::fs::create_dir_all(shard_dir)?;

        // Encode as little-endian f32 bytes.
        let mut bytes = Vec::with_capacity(vector.len() * 4);
        for f in vector {
            bytes.extend_from_slice(&f.to_le_bytes());
        }

        // Atomic write via tmp file.
        let tmp_path = path.with_extension(format!("emb.{}.tmp", std::process::id()));

        if let Err(e) = std::fs::write(&tmp_path, &bytes) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e.into());
        }
        if let Err(e) = std::fs::rename(&tmp_path, &path) {
            let _ = std::fs::remove_file(&tmp_path);
            // If rename fails because the target already exists (another writer
            // won the race), that's fine — both wrote identical bytes.
            if !path.exists() {
                return Err(e.into());
            }
        }

        Ok(())
    }

    /// Load a ref's manifest from disk.
    ///
    /// Returns an empty manifest if the file does not exist.
    pub fn load_manifest(&self, ref_id_hex: &str) -> Result<Manifest> {
        let path = manifest_path(&self.mcp_data, ref_id_hex);
        if !path.exists() {
            return Ok(Manifest::new());
        }
        let bytes = std::fs::read(&path)?;
        if bytes.len() < MANIFEST_HEADER_SIZE {
            return Ok(Manifest::new());
        }
        if bytes[0] != MANIFEST_VERSION {
            return Err(ContextPlusError::Cache(format!(
                "manifest {}: unsupported version {} (expected {})",
                ref_id_hex, bytes[0], MANIFEST_VERSION
            )));
        }
        let data = &bytes[MANIFEST_HEADER_SIZE..];
        let manifest = rkyv::from_bytes::<Manifest, rkyv::rancor::Error>(data)
            .map_err(|e| ContextPlusError::Cache(format!("manifest deserialize: {e}")))?;
        Ok(manifest)
    }

    /// Save a ref's manifest to disk atomically.
    pub fn save_manifest(&self, ref_id_hex: &str, manifest: &Manifest) -> Result<()> {
        let dir = ref_dir(&self.mcp_data, ref_id_hex);
        std::fs::create_dir_all(&dir)?;

        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(manifest)
            .map_err(|e| ContextPlusError::Serialization(format!("manifest serialize: {e}")))?;

        let path = manifest_path(&self.mcp_data, ref_id_hex);
        let tmp_path = dir.join(format!("manifest.rkyv.{}.tmp", std::process::id()));

        let mut buf = Vec::with_capacity(MANIFEST_HEADER_SIZE + bytes.len());
        buf.resize(MANIFEST_HEADER_SIZE, 0);
        buf[0] = MANIFEST_VERSION;
        buf.extend_from_slice(&bytes);

        if let Err(e) = std::fs::write(&tmp_path, &buf) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e.into());
        }
        if let Err(e) = std::fs::rename(&tmp_path, &path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e.into());
        }
        Ok(())
    }

    /// Read the parent ref-id hex for a given ref. Returns `None` if not set.
    pub fn read_parent(&self, ref_id_hex: &str) -> Result<Option<String>> {
        let path = parent_path(&self.mcp_data, ref_id_hex);
        if !path.exists() {
            return Ok(None);
        }
        let s = std::fs::read_to_string(&path)?;
        let s = s.trim().to_string();
        if s.is_empty() { Ok(None) } else { Ok(Some(s)) }
    }

    /// Write the parent ref-id hex for a given ref.
    ///
    /// Called when a new worktree ref forks from a base ref.
    pub fn write_parent(&self, ref_id_hex: &str, parent_hex: &str) -> Result<()> {
        let dir = ref_dir(&self.mcp_data, ref_id_hex);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(parent_path(&self.mcp_data, ref_id_hex), parent_hex)?;
        Ok(())
    }

    /// Look up a chunk hash in a ref's manifest, chaining through parent refs
    /// until a hit is found or the chain is exhausted.
    ///
    /// Returns `Ok(None)` on a miss (embedding not yet computed).
    pub fn lookup_chunk(&self, ref_id_hex: &str, key: &ChunkKey) -> Result<Option<ChunkHash>> {
        let mut current = ref_id_hex.to_string();
        let mut depth = 0usize;
        const MAX_CHAIN_DEPTH: usize = 128; // guard against cycles

        loop {
            if depth >= MAX_CHAIN_DEPTH {
                tracing::warn!(
                    ref_id = ref_id_hex,
                    depth,
                    "CAS parent chain depth limit exceeded — treating as miss"
                );
                return Ok(None);
            }
            let manifest = self.load_manifest(&current)?;
            if let Some(hash) = manifest
                .keys
                .iter()
                .zip(manifest.hashes.iter())
                .find(|(k, _)| *k == key)
                .map(|(_, h)| h.clone())
            {
                return Ok(Some(hash));
            }
            // Walk up the parent chain.
            match self.read_parent(&current)? {
                Some(parent) => {
                    current = parent;
                    depth += 1;
                }
                None => return Ok(None), // root ref, no hit
            }
        }
    }

    /// Full lookup: find the chunk hash via parent chain, then fetch the blob.
    ///
    /// Returns `Ok(None)` on a miss (either no manifest entry or blob missing).
    /// Returns a descriptive error if the manifest has an entry but the blob
    /// is absent — this indicates CAS corruption and the caller should
    /// re-queue the chunk for embedding.
    pub fn get_embedding(&self, ref_id_hex: &str, key: &ChunkKey) -> Result<Option<Vec<f32>>> {
        match self.lookup_chunk(ref_id_hex, key)? {
            None => Ok(None),
            Some(hash) => {
                match self.read_blob(&hash)? {
                    Some(vec) => Ok(Some(vec)),
                    None => {
                        // Manifest entry present but blob missing — CAS corruption.
                        Err(ContextPlusError::Cache(format!(
                            "CAS blob missing for manifest entry {}/{} (hash {}): ref needs re-walk",
                            key.path, key.chunk_idx, hash
                        )))
                    }
                }
            }
        }
    }

    /// Write a chunk's hash to the ref's manifest and write the embedding blob
    /// to CAS. Safe to call from multiple threads; the manifest write is not
    /// under a global lock — callers should batch updates and call
    /// `save_manifest` once per file rather than per-chunk.
    ///
    /// To batch-write, prefer:
    /// 1. Compute all `(key, hash)` pairs for the file.
    /// 2. Call `write_blob` for each new hash.
    /// 3. Call `update_manifest` once with the collected pairs.
    pub fn write_blob_only(&self, hash: &ChunkHash, vector: &[f32]) -> Result<()> {
        self.write_blob(hash, vector)
    }

    /// Atomically update the ref's manifest with new `(key, hash)` pairs.
    ///
    /// Loads the existing manifest, merges the new pairs (new entries win),
    /// and saves atomically. Thread-safe only if no other writer is
    /// concurrently modifying the same ref's manifest — callers must serialize
    /// manifest writes per ref.
    pub fn update_manifest(
        &self,
        ref_id_hex: &str,
        updates: &[(ChunkKey, ChunkHash)],
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let mut manifest = self.load_manifest(ref_id_hex)?;
        for (key, hash) in updates {
            manifest.upsert(key.clone(), hash.clone());
        }
        self.save_manifest(ref_id_hex, &manifest)
    }

    /// Initialize a forked ref on disk.
    ///
    /// Creates the ref directory and writes the parent pointer. The manifest
    /// starts empty (on first walk, only changed files contribute entries).
    pub fn fork_ref(&self, ref_id_hex: &str, parent_hex: &str) -> Result<()> {
        let dir = ref_dir(&self.mcp_data, ref_id_hex);
        std::fs::create_dir_all(&dir)?;
        self.write_parent(ref_id_hex, parent_hex)?;
        // Write an empty manifest so the ref directory is fully initialized.
        let empty = Manifest::new();
        self.save_manifest(ref_id_hex, &empty)?;
        Ok(())
    }

    /// Return the path of the legacy-format detection sentinel.
    fn migration_marker_path(&self) -> PathBuf {
        self.mcp_data.join(MIGRATION_MARKER)
    }

    /// Return `true` if the new CAS directory exists.
    fn cas_dir_exists(&self) -> bool {
        self.mcp_data.join(CAS_DIR).exists()
    }

    /// Detect and handle the legacy path-keyed `.rkyv` format.
    ///
    /// Called at daemon start. If `.mcp_data/cas/` is absent but `.rkyv` files
    /// exist, deletes all legacy `.rkyv` files, writes the migration marker,
    /// and returns `true` to signal that a fresh rebuild is needed.
    ///
    /// If the migration marker already exists (crash during prior migration),
    /// the marker is removed and the function returns `true` so the caller
    /// triggers a fresh rebuild.
    ///
    /// Returns `false` when no action is needed (CAS dir already present).
    pub fn detect_and_migrate_legacy(&self) -> bool {
        let marker = self.migration_marker_path();

        // Crash recovery: marker left from a prior failed migration.
        if marker.exists() {
            tracing::info!(
                "CAS migration marker found at {} — prior migration incomplete; clearing for fresh rebuild",
                marker.display()
            );
            let _ = std::fs::remove_file(&marker);
            return true;
        }

        if self.cas_dir_exists() {
            return false; // already on new format
        }

        // Check whether any legacy `.rkyv` files exist.
        let legacy_files: Vec<PathBuf> = match list_legacy_rkyv_files(&self.mcp_data) {
            Ok(files) => files,
            Err(e) => {
                tracing::warn!("CAS migration: could not list legacy files: {e}");
                return false;
            }
        };

        if legacy_files.is_empty() {
            return false; // new daemon with empty data dir — no migration needed
        }

        // Write marker before deleting so a crash here is recoverable.
        if let Err(e) = std::fs::write(&marker, b"migration in progress\n") {
            tracing::warn!("CAS migration: could not write marker: {e}; aborting migration");
            return false;
        }

        let mut deleted = 0usize;
        for path in &legacy_files {
            match std::fs::remove_file(path) {
                Ok(()) => {
                    tracing::info!(
                        "CAS migration: deleted legacy cache file {}",
                        path.display()
                    );
                    deleted += 1;
                }
                Err(e) => {
                    tracing::warn!("CAS migration: failed to delete {}: {}", path.display(), e);
                }
            }
        }

        tracing::info!(
            deleted,
            "CAS migration: removed legacy path-keyed .rkyv files; daemon will rebuild fresh"
        );

        // Remove the marker — migration complete.
        let _ = std::fs::remove_file(&marker);
        true
    }

    /// Return a reference to the `.mcp_data` base path.
    pub fn mcp_data(&self) -> &Path {
        &self.mcp_data
    }
}

// ---------------------------------------------------------------------------
// Legacy file detection
// ---------------------------------------------------------------------------

/// List all `.rkyv` files in `mcp_data` (non-recursive, top-level only).
///
/// These are the old path-keyed embedding caches written before the CAS layout.
fn list_legacy_rkyv_files(mcp_data: &Path) -> io::Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    if !mcp_data.exists() {
        return Ok(result);
    }
    for entry in std::fs::read_dir(mcp_data)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("rkyv") {
            result.push(path);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Hex helpers (no external dep — std only)
// ---------------------------------------------------------------------------

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_store(tmp: &TempDir) -> CasStore {
        let mcp_data = tmp.path().join(".mcp_data");
        std::fs::create_dir_all(&mcp_data).unwrap();
        CasStore::new(mcp_data, "nomic-embed-text")
    }

    fn dummy_vector(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 * 0.1).collect()
    }

    // ---- ChunkHash ----

    #[test]
    fn chunk_hash_of_is_deterministic() {
        let h1 = ChunkHash::of("hello world");
        let h2 = ChunkHash::of("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn chunk_hash_different_text_differs() {
        let h1 = ChunkHash::of("abc");
        let h2 = ChunkHash::of("xyz");
        assert_ne!(h1, h2);
    }

    #[test]
    fn chunk_hash_hex_roundtrip() {
        let h = ChunkHash::of("roundtrip test");
        let hex = h.to_hex();
        let parsed = ChunkHash::from_hex(&hex).expect("parse should succeed");
        assert_eq!(h, parsed);
    }

    #[test]
    fn chunk_hash_shard_and_suffix_cover_all_bytes() {
        let h = ChunkHash::of("shard test");
        let shard = h.shard();
        let suffix = h.suffix();
        assert_eq!(shard.len(), 2); // 1 byte = 2 hex chars
        assert_eq!(suffix.len(), 62); // 31 bytes = 62 hex chars
    }

    // ---- Manifest ----

    #[test]
    fn manifest_upsert_adds_new_entry() {
        let mut m = Manifest::new();
        let key = ChunkKey::new("src/main.rs", 0);
        let hash = ChunkHash::of("fn main() {}");
        m.upsert(key.clone(), hash.clone());
        assert_eq!(m.len(), 1);
        let map = m.to_map();
        assert_eq!(map.get(&key), Some(&hash));
    }

    #[test]
    fn manifest_upsert_overwrites_existing() {
        let mut m = Manifest::new();
        let key = ChunkKey::new("src/main.rs", 0);
        let hash1 = ChunkHash::of("v1");
        let hash2 = ChunkHash::of("v2");
        m.upsert(key.clone(), hash1);
        m.upsert(key.clone(), hash2.clone());
        assert_eq!(m.len(), 1);
        let map = m.to_map();
        assert_eq!(map.get(&key), Some(&hash2));
    }

    #[test]
    fn manifest_identical_paths_different_chunks() {
        let mut m = Manifest::new();
        let hash = ChunkHash::of("chunk content");
        m.upsert(ChunkKey::new("src/lib.rs", 0), hash.clone());
        m.upsert(ChunkKey::new("src/lib.rs", 1), hash.clone());
        assert_eq!(m.len(), 2);
    }

    // ---- CasStore blob I/O ----

    #[test]
    fn write_and_read_blob() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let hash = ChunkHash::of("fn hello() {}");
        let vec = dummy_vector(4);
        store.write_blob(&hash, &vec).unwrap();
        let loaded = store.read_blob(&hash).unwrap().expect("blob should exist");
        assert_eq!(loaded, vec);
    }

    #[test]
    fn blob_exists_false_before_write() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let hash = ChunkHash::of("not written yet");
        assert!(!store.blob_exists(&hash));
    }

    #[test]
    fn blob_exists_true_after_write() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let hash = ChunkHash::of("written");
        store.write_blob(&hash, &dummy_vector(3)).unwrap();
        assert!(store.blob_exists(&hash));
    }

    #[test]
    fn write_blob_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let hash = ChunkHash::of("idempotent");
        let vec = dummy_vector(4);
        store.write_blob(&hash, &vec).unwrap();
        store.write_blob(&hash, &vec).unwrap(); // second write is a no-op
        let loaded = store.read_blob(&hash).unwrap().unwrap();
        assert_eq!(loaded, vec);
    }

    #[test]
    fn identical_content_under_different_paths_shares_one_blob() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let content = "identical content";
        let hash = ChunkHash::of(content);
        let vec = dummy_vector(8);
        store.write_blob(&hash, &vec).unwrap();

        // Two manifest entries, same hash.
        let ref_id = "aabbccdd";
        store
            .update_manifest(
                ref_id,
                &[
                    (ChunkKey::new("src/a.rs", 0), hash.clone()),
                    (ChunkKey::new("src/b.rs", 0), hash.clone()),
                ],
            )
            .unwrap();

        // Exactly one blob file on disk.
        let blob_path_a = blob_path(&store.mcp_data, &store.model, &hash);
        assert!(blob_path_a.exists());
        // Both keys resolve to the same hash.
        let map = store.load_manifest(ref_id).unwrap().to_map();
        let h1 = map.get(&ChunkKey::new("src/a.rs", 0)).unwrap();
        let h2 = map.get(&ChunkKey::new("src/b.rs", 0)).unwrap();
        assert_eq!(h1, h2);
    }

    // ---- Manifest save / load ----

    #[test]
    fn save_and_load_manifest_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let ref_id = "deafbeef";
        let key = ChunkKey::new("src/server.rs", 2);
        let hash = ChunkHash::of("server chunk 2");
        let mut manifest = Manifest::new();
        manifest.upsert(key.clone(), hash.clone());
        store.save_manifest(ref_id, &manifest).unwrap();

        let loaded = store.load_manifest(ref_id).unwrap();
        let map = loaded.to_map();
        assert_eq!(map.get(&key), Some(&hash));
    }

    #[test]
    fn load_manifest_returns_empty_when_missing() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let m = store.load_manifest("nonexistent_ref").unwrap();
        assert!(m.is_empty());
    }

    // ---- Parent chain lookup ----

    #[test]
    fn lookup_chunk_chains_through_parent() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let parent_id = "parent001";
        let child_id = "child002";
        let key = ChunkKey::new("src/lib.rs", 0);
        let hash = ChunkHash::of("parent content");
        let vec = dummy_vector(4);

        // Write the blob and add to parent's manifest.
        store.write_blob(&hash, &vec).unwrap();
        store
            .update_manifest(parent_id, &[(key.clone(), hash.clone())])
            .unwrap();

        // Fork child from parent (empty child manifest).
        store.fork_ref(child_id, parent_id).unwrap();

        // Child has no entry in its own manifest but should find it via parent.
        let found = store.lookup_chunk(child_id, &key).unwrap();
        assert_eq!(found, Some(hash.clone()));

        // get_embedding should return the vector.
        let emb = store.get_embedding(child_id, &key).unwrap();
        assert_eq!(emb, Some(vec));
    }

    #[test]
    fn lookup_chunk_returns_none_on_miss() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let result = store
            .lookup_chunk("noref", &ChunkKey::new("not/there.rs", 0))
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn get_embedding_errors_on_blob_missing_for_manifest_entry() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let ref_id = "corrupt001";
        let key = ChunkKey::new("src/foo.rs", 0);
        // Fabricate a hash and add it to the manifest WITHOUT writing the blob.
        let hash = ChunkHash::of("ghost chunk");
        store
            .update_manifest(ref_id, &[(key.clone(), hash)])
            .unwrap();
        let err = store.get_embedding(ref_id, &key).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("CAS blob missing"), "unexpected error: {msg}");
    }

    // ---- Fork ----

    #[test]
    fn fork_ref_creates_parent_pointer() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        store.fork_ref("child_x", "parent_y").unwrap();
        let parent = store.read_parent("child_x").unwrap();
        assert_eq!(parent, Some("parent_y".to_string()));
    }

    #[test]
    fn fork_ref_manifest_starts_empty() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        store.fork_ref("child_z", "parent_w").unwrap();
        let m = store.load_manifest("child_z").unwrap();
        assert!(m.is_empty());
    }

    // ---- Legacy migration ----

    #[test]
    fn detect_and_migrate_legacy_no_legacy_files_no_action() {
        let tmp = TempDir::new().unwrap();
        let mcp_data = tmp.path().join(".mcp_data");
        std::fs::create_dir_all(&mcp_data).unwrap();
        let store = CasStore::new(mcp_data, "model");
        // No .rkyv files, no cas dir — brand new daemon, no migration needed.
        assert!(!store.detect_and_migrate_legacy());
    }

    #[test]
    fn detect_and_migrate_legacy_deletes_rkyv_files() {
        let tmp = TempDir::new().unwrap();
        let mcp_data = tmp.path().join(".mcp_data");
        std::fs::create_dir_all(&mcp_data).unwrap();
        // Simulate legacy files.
        std::fs::write(mcp_data.join("embed-nomic-embed-text.rkyv"), b"fake").unwrap();
        std::fs::write(mcp_data.join("query-embeddings-nomic.rkyv"), b"fake").unwrap();

        let store = CasStore::new(mcp_data.clone(), "model");
        let migrated = store.detect_and_migrate_legacy();
        assert!(migrated, "should have detected legacy files");
        // Legacy files should be gone.
        assert!(!mcp_data.join("embed-nomic-embed-text.rkyv").exists());
        assert!(!mcp_data.join("query-embeddings-nomic.rkyv").exists());
    }

    #[test]
    fn detect_and_migrate_legacy_cas_dir_exists_skips() {
        let tmp = TempDir::new().unwrap();
        let mcp_data = tmp.path().join(".mcp_data");
        std::fs::create_dir_all(mcp_data.join("cas")).unwrap();
        // Add a fake .rkyv file — should NOT be deleted because cas/ already exists.
        std::fs::write(mcp_data.join("some.rkyv"), b"data").unwrap();

        let store = CasStore::new(mcp_data.clone(), "model");
        assert!(!store.detect_and_migrate_legacy());
        assert!(mcp_data.join("some.rkyv").exists()); // not deleted
    }

    #[test]
    fn detect_and_migrate_legacy_marker_triggers_rebuild() {
        let tmp = TempDir::new().unwrap();
        let mcp_data = tmp.path().join(".mcp_data");
        std::fs::create_dir_all(&mcp_data).unwrap();
        // Simulate a crash during prior migration.
        std::fs::write(mcp_data.join(MIGRATION_MARKER), b"in progress").unwrap();

        let store = CasStore::new(mcp_data.clone(), "model");
        let migrated = store.detect_and_migrate_legacy();
        assert!(migrated, "marker should trigger rebuild");
        // Marker should be cleaned up.
        assert!(!mcp_data.join(MIGRATION_MARKER).exists());
    }

    // ---- update_manifest ----

    #[test]
    fn update_manifest_merges_with_existing() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let ref_id = "merge_test";
        let k1 = ChunkKey::new("a.rs", 0);
        let k2 = ChunkKey::new("b.rs", 0);
        let h1 = ChunkHash::of("a");
        let h2 = ChunkHash::of("b");

        store
            .update_manifest(ref_id, &[(k1.clone(), h1.clone())])
            .unwrap();
        store
            .update_manifest(ref_id, &[(k2.clone(), h2.clone())])
            .unwrap();

        let m = store.load_manifest(ref_id).unwrap();
        let map = m.to_map();
        assert_eq!(map.get(&k1), Some(&h1));
        assert_eq!(map.get(&k2), Some(&h2));
    }
}
