// Adapter structs that bridge shared server state to tool function traits.
// Extracted from server.rs (Round 11D) to reduce server.rs line count.

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use crate::cache::rkyv_store;
use crate::config::Config;
use crate::core::embeddings::{CacheEntry, OllamaClient, content_hash};
use crate::core::walker::walk_with_config;
use crate::error::Result;
use crate::server::{SharedState, build_embedding_document, cache_name};
use crate::tools::semantic_search::{
    CachedSearchIndex, EmbedFn, SearchDocument, WalkAndIndexFn, semantic_embedding_content,
};

#[cfg(test)]
pub(crate) mod test_seams {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Barrier, Mutex, OnceLock};

    pub(crate) struct AsyncPause {
        entered: tokio::sync::Semaphore,
        resume: tokio::sync::Semaphore,
    }

    impl AsyncPause {
        fn new() -> Self {
            Self {
                entered: tokio::sync::Semaphore::new(0),
                resume: tokio::sync::Semaphore::new(0),
            }
        }

        pub(crate) async fn wait_until_entered(&self) {
            self.entered.acquire().await.unwrap().forget();
        }

        pub(crate) fn resume(&self) {
            self.resume.add_permits(1);
        }
    }

    struct FileSnapshotPause {
        hash: String,
        pause: Arc<AsyncPause>,
    }

    fn file_snapshot_slots() -> &'static Mutex<BTreeMap<PathBuf, FileSnapshotPause>> {
        static SLOTS: OnceLock<Mutex<BTreeMap<PathBuf, FileSnapshotPause>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    fn cache_snapshot_slots() -> &'static Mutex<BTreeMap<PathBuf, Arc<AsyncPause>>> {
        static SLOTS: OnceLock<Mutex<BTreeMap<PathBuf, Arc<AsyncPause>>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    pub(crate) fn pause_after_file_snapshot(root: &Path, hash: String) -> Arc<AsyncPause> {
        let pause = Arc::new(AsyncPause::new());
        file_snapshot_slots().lock().unwrap().insert(
            root.to_path_buf(),
            FileSnapshotPause {
                hash,
                pause: Arc::clone(&pause),
            },
        );
        pause
    }

    pub(crate) fn pause_after_cache_snapshot(root: &Path) -> Arc<AsyncPause> {
        let pause = Arc::new(AsyncPause::new());
        cache_snapshot_slots()
            .lock()
            .unwrap()
            .insert(root.to_path_buf(), Arc::clone(&pause));
        pause
    }

    pub(crate) async fn after_file_snapshot(root: &Path, hashes: &[(String, String)]) {
        let pause = {
            let mut slots = file_snapshot_slots().lock().unwrap();
            let matches = slots
                .get(root)
                .is_some_and(|candidate| hashes.iter().any(|(_, hash)| hash == &candidate.hash));
            matches.then(|| slots.remove(root).unwrap().pause)
        };
        if let Some(pause) = pause {
            pause.entered.add_permits(1);
            pause.resume.acquire().await.unwrap().forget();
        }
    }

    pub(crate) async fn after_cache_snapshot(root: &Path) {
        let pause = cache_snapshot_slots().lock().unwrap().remove(root);
        if let Some(pause) = pause {
            pause.entered.add_permits(1);
            pause.resume.acquire().await.unwrap().forget();
        }
    }

    pub(crate) async fn seed_pending(
        ref_index: &crate::ref_index::RefIndex,
        path: &str,
        hash: String,
        text: String,
    ) {
        ref_index.semantic_fill.lock().await.pending.insert(
            path.to_string(),
            super::FillDocument {
                path: path.to_string(),
                hash,
                text,
                owner: None,
            },
        );
    }

    pub(crate) async fn pending_hash(
        ref_index: &crate::ref_index::RefIndex,
        path: &str,
    ) -> Option<String> {
        ref_index
            .semantic_fill
            .lock()
            .await
            .pending
            .get(path)
            .map(|document| document.hash.clone())
    }

    pub(crate) async fn fill_running(ref_index: &crate::ref_index::RefIndex) -> bool {
        ref_index.semantic_fill.lock().await.running
    }

    pub(crate) struct MetadataPause {
        enumerated: Barrier,
        resume: Barrier,
    }

    impl MetadataPause {
        fn new() -> Self {
            Self {
                enumerated: Barrier::new(2),
                resume: Barrier::new(2),
            }
        }

        pub(crate) fn wait_until_enumerated(&self) {
            self.enumerated.wait();
        }

        pub(crate) fn resume(&self) {
            self.resume.wait();
        }
    }

    fn metadata_slots() -> &'static Mutex<BTreeMap<PathBuf, Arc<MetadataPause>>> {
        static SLOTS: OnceLock<Mutex<BTreeMap<PathBuf, Arc<MetadataPause>>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    pub(crate) fn pause_after_metadata_enumeration(root: &Path) -> Arc<MetadataPause> {
        let pause = Arc::new(MetadataPause::new());
        metadata_slots()
            .lock()
            .unwrap()
            .insert(root.to_path_buf(), Arc::clone(&pause));
        pause
    }

    pub(crate) fn after_metadata_enumeration(root: &Path) {
        let pause = metadata_slots().lock().unwrap().remove(root);
        if let Some(pause) = pause {
            pause.enumerated.wait();
            pause.resume.wait();
        }
    }
}

// --- OllamaEmbedder ---

/// Thin adapter: wraps OllamaClient to implement the EmbedFn trait used by semantic search.
pub struct OllamaEmbedder(pub OllamaClient);

impl EmbedFn for OllamaEmbedder {
    fn embed(
        &self,
        texts: &[String],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>>
    {
        let texts = texts.to_vec();
        Box::pin(async move { self.0.embed_queries(&texts).await })
    }
}

// --- CachedWalkerIndexer ---

/// Walk the project, read file contents, and return SearchDocuments with embedding vectors.
/// Uses the embedding cache in SharedState for warm hits — only embeds new/changed files.
pub struct CachedWalkerIndexer {
    pub config: Config,
    pub ollama: OllamaClient,
    pub state: Arc<SharedState>,
}

impl WalkAndIndexFn for CachedWalkerIndexer {
    fn vector_generation(
        &self,
        root: &Path,
    ) -> crate::tools::semantic_search::VectorGenerationFuture<'_> {
        let canonical = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        Box::pin(async move {
            self.state
                .refs
                .read()
                .await
                .values()
                .find(|r| r.canonical_root == canonical)
                .map(|r| {
                    r.semantic_vector_generation
                        .load(std::sync::atomic::Ordering::Acquire)
                })
                .unwrap_or(0)
        })
    }

    fn metadata_fingerprint(
        &self,
        root_dir: &Path,
    ) -> crate::tools::semantic_search::MetadataFingerprintFuture<'_> {
        let root = root_dir.to_path_buf();
        let config = self.config.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                let mut entries = walk_with_config(&root, &config);
                entries.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
                #[cfg(test)]
                test_seams::after_metadata_enumeration(&root);
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                std::fs::canonicalize(&root)
                    .unwrap_or_else(|_| root.clone())
                    .hash(&mut hash);
                for entry in &entries {
                    entry.relative_path.hash(&mut hash);
                    let Ok(meta) = std::fs::metadata(root.join(&entry.relative_path)) else {
                        return Ok(None);
                    };
                    let Ok(modified) = meta.modified() else {
                        return Ok(None);
                    };
                    if meta.is_dir() != entry.is_directory || (!meta.is_file() && !meta.is_dir()) {
                        return Ok(None);
                    }
                    meta.len().hash(&mut hash);
                    modified.hash(&mut hash);
                }
                Ok(Some(crate::tools::semantic_search::MetadataFingerprint {
                    n_entries: entries.len(),
                    metadata_hash: hash.finish(),
                }))
            })
            .await
            .map_err(|e| crate::error::ContextPlusError::Other(e.to_string()))?
        })
    }

    fn walk_and_index(
        &self,
        root_dir: &Path,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)>>
                + Send
                + '_,
        >,
    > {
        let root = root_dir.to_path_buf();
        Box::pin(async move {
            let canonical = std::fs::canonicalize(&root).unwrap_or_else(|_| root.clone());
            let ref_index = self
                .state
                .refs
                .read()
                .await
                .values()
                .filter(|r| canonical.starts_with(&r.canonical_root))
                .max_by_key(|r| r.canonical_root.components().count())
                .cloned()
                .or_else(|| self.state.default_ref())
                .expect("registered ref");
            self.walk_for_ref(&root, ref_index).await
        })
    }
}

pub struct RefWalkerIndexer {
    pub walker: CachedWalkerIndexer,
    pub ref_index: Arc<crate::ref_index::RefIndex>,
}

impl RefWalkerIndexer {
    pub(crate) fn walk_candidates(
        &self,
        root: &Path,
        candidates: std::collections::HashSet<String>,
    ) -> crate::tools::semantic_search::WalkAndIndexFuture<'_> {
        self.walker
            .walk_for_ref_candidates(root, self.ref_index.clone(), Some(candidates))
    }

    /// Expires a worktree's semantic entry when its parent holds a forkable
    /// vector store the worktree has not forked or been refused, so a query of
    /// its whole `root` walks and re-forks.
    pub(crate) async fn expire_stale_fork(&self, root: &Path) {
        let ref_index = &self.ref_index;
        let Some(parent_id) = ref_index.parent_ref_id else {
            return;
        };
        if tokio::fs::canonicalize(root).await.ok().as_deref() != Some(&ref_index.canonical_root) {
            return;
        }
        let Some(parent) = self.walker.state.ref_index(parent_id).await else {
            return;
        };
        let base = parent.search_index_cache.read().await.clone();
        let Some(base) = base.filter(|base| base.forkable_at(&parent.canonical_root)) else {
            return;
        };
        if base.index.vector_store().is_some_and(|store| {
            std::sync::Weak::ptr_eq(&ref_index.fork_base.lock().unwrap(), &Arc::downgrade(store))
        }) {
            return;
        }
        let entry = ref_index.search_index_cache.read().await.clone();
        let Some(entry) = entry.filter(|entry| !entry.index.shares_vector_store(&base.index))
        else {
            return;
        };
        ref_index
            .cache_generation
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        *entry.metadata.write().unwrap() = None;
    }
}

impl WalkAndIndexFn for RefWalkerIndexer {
    fn vector_generation(
        &self,
        _root: &Path,
    ) -> crate::tools::semantic_search::VectorGenerationFuture<'_> {
        Box::pin(async {
            self.ref_index
                .semantic_vector_generation
                .load(std::sync::atomic::Ordering::Acquire)
        })
    }

    fn metadata_fingerprint(
        &self,
        root: &Path,
    ) -> crate::tools::semantic_search::MetadataFingerprintFuture<'_> {
        self.walker.metadata_fingerprint(root)
    }

    fn walk_and_index(
        &self,
        root_dir: &Path,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)>>
                + Send
                + '_,
        >,
    > {
        self.walker.walk_for_ref(root_dir, self.ref_index.clone())
    }

    fn track_background_task(&self, task: &tokio::task::JoinHandle<()>) {
        self.ref_index.track_background_task(task);
    }
}

impl CachedWalkerIndexer {
    fn walk_for_ref(
        &self,
        root_dir: &Path,
        ref_index: Arc<crate::ref_index::RefIndex>,
    ) -> crate::tools::semantic_search::WalkAndIndexFuture<'_> {
        self.walk_for_ref_candidates(root_dir, ref_index, None)
    }

    fn walk_for_ref_candidates(
        &self,
        root_dir: &Path,
        ref_index: Arc<crate::ref_index::RefIndex>,
        candidates: Option<std::collections::HashSet<String>>,
    ) -> crate::tools::semantic_search::WalkAndIndexFuture<'_> {
        let root = root_dir.to_path_buf();
        let config = self.config.clone();
        let ollama = self.ollama.clone();
        Box::pin(async move {
            let canonical = std::fs::canonicalize(&root).unwrap_or_else(|_| root.clone());
            let prefix = canonical
                .strip_prefix(&ref_index.canonical_root)
                .unwrap_or(Path::new(""));
            let embedding_cache = Arc::clone(&ref_index.embedding_cache);
            let full_walk = candidates.is_none() && prefix.as_os_str().is_empty();
            let fork_parent = match ref_index.parent_ref_id {
                Some(parent_id) if full_walk => self.state.ref_index(parent_id).await,
                _ => None,
            };
            if let Some(parent) = &fork_parent {
                self.build_parent_index(parent, &ref_index).await;
            }
            let walk_start = WalkStart {
                generation: ref_index
                    .cache_generation
                    .load(std::sync::atomic::Ordering::Acquire),
                vector_generation: ref_index
                    .semantic_vector_generation
                    .load(std::sync::atomic::Ordering::Acquire),
                seen: ref_index
                    .search_index_cache
                    .read()
                    .await
                    .as_ref()
                    .map(Arc::downgrade),
            };
            #[cfg(test)]
            ref_index
                .semantic_walks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let started = std::time::Instant::now();
            let entries = walk_with_config(&root, &config);
            let walk_ms = started.elapsed().as_millis();

            let max_file_size = config.max_embed_file_size as u64;
            // Read all files concurrently (up to 32 at a time)
            let mut join_set = tokio::task::JoinSet::new();
            for (i, entry) in entries.iter().enumerate() {
                if candidates
                    .as_ref()
                    .is_some_and(|paths| !paths.contains(&entry.relative_path))
                {
                    continue;
                }
                let full_path = root.join(&entry.relative_path);
                let rel_path = prefix
                    .join(&entry.relative_path)
                    .to_string_lossy()
                    .into_owned();
                join_set.spawn(async move {
                    if let Ok(meta) = tokio::fs::metadata(&full_path).await
                        && meta.len() > max_file_size
                    {
                        return (i, rel_path, None);
                    }
                    let content = tokio::fs::read_to_string(&full_path).await.ok();
                    (i, rel_path, content)
                });
            }

            let mut file_contents: Vec<(usize, String, Option<String>)> =
                Vec::with_capacity(entries.len());
            while let Some(result) = join_set.join_next().await {
                if let Ok(item) = result {
                    file_contents.push(item);
                }
            }
            file_contents.sort_unstable_by_key(|(i, _, _)| *i);
            let read_ms = started.elapsed().as_millis() - walk_ms;

            // Parsing runs in parallel; a file whose content matches the
            // snapshot's document keeps that document's parsed fields.
            let doc_shape = config.embed_doc_shape;
            let seed_config = config.clone();
            let seed_ref = Arc::clone(&ref_index);
            // A worktree takes the documents of files identical to its parent's.
            let parent_index = match ref_index.parent_ref_id {
                Some(parent_id) => match self.state.ref_index(parent_id).await {
                    Some(parent) => parent
                        .search_index_cache
                        .read()
                        .await
                        .clone()
                        .filter(|cached| cached.search_root() == parent.canonical_root),
                    None => None,
                },
                None => None,
            };
            let (mut docs, content_hashes, embedding_texts, reused) =
                tokio::task::spawn_blocking(move || {
                    use rayon::prelude::*;
                    let seeds = crate::server::snapshots::file_seed(&seed_config, &seed_ref);
                    let parent_documents = documents_by_path(parent_index.as_deref());
                    let built: Vec<(SearchDocument, (String, String), String, bool)> =
                        file_contents
                            .into_par_iter()
                            .filter_map(|(_, rel_path, content)| {
                                let content = content?;
                                let hash = content_hash(&content);
                                let embedding_text =
                                    build_embedding_document(&rel_path, &content, doc_shape);
                                let (doc, reused) = walk_document(
                                    &rel_path,
                                    &content,
                                    &hash,
                                    seeds.as_deref(),
                                    &parent_documents,
                                );
                                Some((doc, (rel_path, hash), embedding_text, reused))
                            })
                            .collect();
                    let reused = built.iter().filter(|(.., reused)| *reused).count();
                    let mut docs = Vec::with_capacity(built.len());
                    let mut content_hashes = Vec::with_capacity(built.len());
                    let mut embedding_texts = Vec::with_capacity(built.len());
                    for (doc, hash, text, _) in built {
                        docs.push(doc);
                        content_hashes.push(hash);
                        embedding_texts.push(text);
                    }
                    (docs, content_hashes, embedding_texts, reused)
                })
                .await
                .map_err(|e| crate::error::ContextPlusError::Other(e.to_string()))?;
            if full_walk {
                crate::server::snapshots::file_seed_used(&ref_index);
                if reused < docs.len() {
                    let change = match reused {
                        0 => crate::server::snapshots::Change::Full,
                        _ => crate::server::snapshots::Change::Files {
                            changed: docs.len() - reused,
                            documents: docs.len(),
                        },
                    };
                    crate::server::snapshots::schedule_files(&config, &ref_index, change);
                }
            }

            for (doc, (_, hash)) in docs.iter_mut().zip(&content_hashes) {
                doc.source_hash = hash.clone();
                doc.path = Path::new(&doc.path)
                    .strip_prefix(prefix)
                    .unwrap_or(Path::new(&doc.path))
                    .to_string_lossy()
                    .into_owned();
            }

            tracing::info!(
                phase = "semantic_walk",
                walk_ms,
                read_ms,
                documents_ms = started.elapsed().as_millis() - walk_ms - read_ms,
                documents = docs.len(),
                reused,
                "cold-start phase"
            );

            #[cfg(test)]
            test_seams::after_file_snapshot(&root, &content_hashes).await;

            if docs.is_empty() {
                return Ok((docs, Vec::new()));
            }

            let parent_vectors = match ref_index.parent_ref_id {
                Some(parent_id) => self
                    .state
                    .ref_index(parent_id)
                    .await
                    .map(|parent| Arc::clone(&parent.embedding_cache)),
                None => None,
            };
            // A worktree starts, and restarts after an eviction, from the
            // vectors it persisted that its parent lacks.
            if let Some(parent_vectors) = &parent_vectors
                && embedding_cache.read().await.is_empty()
            {
                let root = ref_index.root_dir.clone();
                let name = cache_name("embeddings", &config);
                if let Ok(Ok(Some(store))) =
                    tokio::task::spawn_blocking(move || rkyv_store::mmap_vector_store(&root, &name))
                        .await
                {
                    let own: HashMap<String, CacheEntry> = {
                        let parent = parent_vectors.read().await;
                        store
                            .to_cache()
                            .into_iter()
                            .filter(|(path, entry)| !inherits(&parent, path, entry))
                            .collect()
                    };
                    let mut cache = embedding_cache.write().await;
                    if cache.is_empty() {
                        *cache = own;
                    }
                }
            }

            let fill_snapshot = ref_index.semantic_fill.lock().await;
            let cache_read = embedding_cache.read().await;
            let observed: Vec<_> = content_hashes
                .iter()
                .map(|(path, _)| {
                    (
                        cache_read.get(path).map(|entry| entry.hash.clone()),
                        fill_snapshot.pending.get(path).map(|doc| doc.hash.clone()),
                    )
                })
                .collect();
            let mut vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(docs.len());
            let mut uncached_indices: Vec<usize> = Vec::new();
            let mut uncached_texts: Vec<String> = Vec::new();

            for (i, (rel_path, hash)) in content_hashes.iter().enumerate() {
                if let Some(entry) = cache_read.get(rel_path)
                    && entry.hash == *hash
                {
                    vectors.push(Some(entry.vector.clone()));
                    continue;
                }
                vectors.push(None);
                uncached_indices.push(i);
                uncached_texts.push(embedding_texts[i].clone());
            }
            drop(cache_read);
            drop(fill_snapshot);

            // Query and filler vectors live in memory even when no CAS manifest exists.
            let mut ancestor_id = ref_index.parent_ref_id;
            let mut visited = std::collections::HashSet::new();
            let mut inherited = Vec::new();
            while let Some(id) = ancestor_id {
                if !visited.insert(id) {
                    break;
                }
                let ancestor = self.state.refs.read().await.get(&id).cloned();
                let Some(ancestor) = ancestor else {
                    break;
                };
                let cache = ancestor.embedding_cache.read().await;
                for &idx in &uncached_indices {
                    let (path, hash) = &content_hashes[idx];
                    if vectors[idx].is_none()
                        && let Some(entry) = cache.get(path).filter(|entry| entry.hash == *hash)
                    {
                        vectors[idx] = Some(entry.vector.clone());
                        inherited.push((idx, entry.clone()));
                    }
                }
                tracing::info!(
                    ref_id = %ref_index.cas_ref_id_hex,
                    ancestor_ref_id = %ancestor.cas_ref_id_hex,
                    ancestor_entries = cache.len(),
                    inherited = inherited.len(),
                    "semantic ancestor cache lookup"
                );
                ancestor_id = ancestor.parent_ref_id;
            }
            let mut current = vec![true; content_hashes.len()];
            for &(idx, _) in &inherited {
                let (path, hash) = &content_hashes[idx];
                current[idx] = FillDocument {
                    path: path.clone(),
                    hash: hash.clone(),
                    text: String::new(),
                    owner: None,
                }
                .is_current(&ref_index.canonical_root, config.max_embed_file_size)
                .await;
            }
            if !inherited.is_empty() {
                let fill = ref_index.semantic_fill.lock().await;
                let mut cache = embedding_cache.write().await;
                for (idx, entry) in inherited {
                    let (path, _) = &content_hashes[idx];
                    // File validation runs without locks; reject intervening cache/fill changes.
                    if current[idx]
                        && cache.get(path).map(|entry| &entry.hash) == observed[idx].0.as_ref()
                        && fill.pending.get(path).map(|doc| &doc.hash) == observed[idx].1.as_ref()
                    {
                        cache.insert(path.clone(), entry);
                    } else {
                        current[idx] = false;
                        vectors[idx] = None;
                    }
                }
            }
            uncached_indices.retain(|&idx| vectors[idx].is_none());

            #[cfg(test)]
            test_seams::after_cache_snapshot(&root).await;

            let cache_entries = embedding_cache.read().await.len();
            tracing::info!(
                ref_id = %ref_index.cas_ref_id_hex,
                cache_entries,
                cached = docs.len() - uncached_indices.len(),
                uncached = uncached_indices.len(),
                "semantic_code_search embedding cache hit/miss"
            );

            for (idx, (path, hash)) in content_hashes.iter().enumerate() {
                if current[idx] && vectors[idx].is_none() {
                    current[idx] = FillDocument {
                        path: path.clone(),
                        hash: hash.clone(),
                        text: String::new(),
                        owner: None,
                    }
                    .is_current(&ref_index.canonical_root, config.max_embed_file_size)
                    .await;
                }
            }
            let owner = Arc::new(());
            let mut fill = ref_index.semantic_fill.lock().await;
            let cache = embedding_cache.read().await;
            let mut pending = Vec::new();
            for (idx, (path, hash)) in content_hashes.iter().enumerate() {
                if !current[idx] {
                    continue;
                }
                if let Some(entry) = cache.get(path).filter(|entry| entry.hash == *hash) {
                    vectors[idx] = Some(entry.vector.clone());
                    if fill.pending.get(path).is_some_and(|doc| doc.hash == *hash) {
                        fill.pending.remove(path);
                    }
                    continue;
                }
                let mut doc = FillDocument {
                    path: path.clone(),
                    hash: hash.clone(),
                    text: embedding_texts[idx].clone(),
                    owner: Some(Arc::downgrade(&owner)),
                };
                if fill.failed(&doc) {
                    fill.pending.remove(path);
                    continue;
                }
                if let Some(current) = fill.pending.get(path) {
                    if current.hash == *hash {
                        continue;
                    }
                    doc.owner = None;
                } else {
                    pending.push((idx, doc.clone()));
                }
                fill.pending.insert(path.clone(), doc);
            }
            drop(cache);
            if !fill.running && !fill.pending.is_empty() {
                fill.running = true;
                let owner = ref_index.clone();
                let ollama = ollama.clone();
                let config = config.clone();
                let parent_vectors = parent_vectors.clone();
                let task = tokio::spawn(async move {
                    run_fill(owner, ollama, config, parent_vectors).await;
                });
                ref_index.track_background_task(&task);
            }
            drop(fill);
            let deadline = tokio::time::Instant::now()
                + std::time::Duration::from_millis(config.embed_budget_ms);
            for chunk in pending.chunks(config.embed_batch_size.max(1)) {
                if tokio::time::Instant::now() >= deadline {
                    break;
                }
                let texts: Vec<_> = chunk.iter().map(|(_, d)| d.text.clone()).collect();
                match tokio::time::timeout_at(deadline, ollama.embed_documents(&texts)).await {
                    Ok(Ok(result)) if result.len() == chunk.len() => {
                        let mut current = Vec::with_capacity(chunk.len());
                        for (_, doc) in chunk {
                            current.push(
                                doc.is_current(
                                    &ref_index.canonical_root,
                                    config.max_embed_file_size,
                                )
                                .await,
                            );
                        }
                        let mut fill = ref_index.semantic_fill.lock().await;
                        let mut cache = embedding_cache.write().await;
                        for (((idx, doc), vector), current) in chunk.iter().zip(result).zip(current)
                        {
                            if !current {
                                if fill
                                    .pending
                                    .get(&doc.path)
                                    .is_some_and(|cur| cur.hash == doc.hash)
                                {
                                    fill.pending.remove(&doc.path);
                                }
                                continue;
                            }
                            if vector.is_empty()
                                || !fill
                                    .pending
                                    .get(&doc.path)
                                    .is_some_and(|cur| cur.hash == doc.hash)
                            {
                                continue;
                            }
                            cache.insert(
                                doc.path.clone(),
                                CacheEntry {
                                    hash: doc.hash.clone(),
                                    vector: vector.clone(),
                                },
                            );
                            vectors[*idx] = Some(vector);
                            if fill
                                .pending
                                .get(&doc.path)
                                .is_some_and(|cur| cur.hash == doc.hash)
                            {
                                fill.pending.remove(&doc.path);
                            }
                        }
                    }
                    Ok(Err(_)) => {
                        let mut fill = ref_index.semantic_fill.lock().await;
                        for (_, doc) in chunk {
                            if fill
                                .pending
                                .get(&doc.path)
                                .is_some_and(|cur| cur.hash == doc.hash)
                            {
                                fill.record_error(doc);
                            }
                        }
                    }
                    _ => {}
                }
            }
            let mut fill = ref_index.semantic_fill.lock().await;
            for (_, doc) in &pending {
                if fill.failed(doc)
                    && fill
                        .pending
                        .get(&doc.path)
                        .is_some_and(|cur| cur.hash == doc.hash)
                {
                    fill.pending.remove(&doc.path);
                }
            }
            let queued = pending
                .iter()
                .filter(|(idx, _)| vectors[*idx].is_none())
                .count();
            drop(owner);
            drop(fill);
            if queued > 0 {
                tracing::warn!(
                    queued,
                    "semantic_code_search returning partial results; leftovers queued for background fill"
                );
            }

            let replacement_dims = ref_index
                .search_index_cache
                .read()
                .await
                .as_ref()
                .and_then(|entry| entry.pending_vector_dimensions());
            if let Some(dims) = replacement_dims {
                let missing: Vec<_> = vectors
                    .iter()
                    .enumerate()
                    .filter(|(_, vector)| vector.as_ref().is_none_or(|v| v.len() != dims))
                    .map(|(i, _)| i)
                    .collect();
                if !missing.is_empty() {
                    let texts: Vec<_> = missing
                        .iter()
                        .map(|&i| embedding_texts[i].clone())
                        .collect();
                    let replacements = ollama.embed_documents(&texts).await?;
                    if replacements.len() != missing.len()
                        || replacements.iter().any(|v| v.len() != dims)
                    {
                        return Err(crate::error::ContextPlusError::Other(
                            "incompatible replacement embedding shape".into(),
                        ));
                    }
                    let mut cache = embedding_cache.write().await;
                    for (i, vector) in missing.into_iter().zip(replacements) {
                        let (path, hash) = &content_hashes[i];
                        if cache.get(path).is_none_or(|entry| entry.hash == *hash) {
                            cache.insert(
                                path.clone(),
                                CacheEntry {
                                    hash: hash.clone(),
                                    vector: vector.clone(),
                                },
                            );
                        }
                        vectors[i] = Some(vector);
                    }
                }
            }
            match fork_parent {
                Some(parent) => self
                    .seed_fork(&ref_index, &parent, canonical, docs, vectors, walk_start)
                    .await
                    .map(|(docs, vectors, _)| (docs, vectors)),
                None => Ok((docs, vectors)),
            }
        })
    }

    /// Builds the parent's index of its whole root when the parent holds none,
    /// as after a restart, and installs it so a worktree can fork it.
    async fn build_parent_index(
        &self,
        parent: &Arc<crate::ref_index::RefIndex>,
        ref_index: &crate::ref_index::RefIndex,
    ) {
        if parent.parent_ref_id.is_some() || parent.canonical_root == ref_index.canonical_root {
            return;
        }
        let seen = {
            let current = parent.search_index_cache.read().await;
            if current
                .as_ref()
                .is_some_and(|entry| entry.search_root() == parent.canonical_root)
            {
                return;
            }
            current.as_ref().map(Arc::downgrade)
        };
        let generation = parent
            .cache_generation
            .load(std::sync::atomic::Ordering::Acquire);
        let vector_generation = parent
            .semantic_vector_generation
            .load(std::sync::atomic::Ordering::Acquire);
        let started = std::time::Instant::now();
        let metadata = self
            .metadata_fingerprint(&parent.canonical_root)
            .await
            .ok()
            .flatten();
        let (docs, vectors) = match self
            .walk_for_ref(&parent.canonical_root, Arc::clone(parent))
            .await
        {
            Ok(walked) if !walked.0.is_empty() => walked,
            Ok(_) => return,
            Err(error) => {
                tracing::warn!(%error, "parent semantic index build failed");
                return;
            }
        };
        let root = parent.canonical_root.clone();
        let Ok(entry) = tokio::task::spawn_blocking(move || {
            CachedSearchIndex::build(
                &root,
                docs,
                vectors,
                generation,
                vector_generation,
                metadata,
            )
        })
        .await
        else {
            return;
        };
        let installed = entry.install(&mut *parent.search_index_cache.write().await, seen.as_ref());
        tracing::info!(
            phase = "semantic_parent_index",
            ref_id = %parent.cas_ref_id_hex,
            installed,
            elapsed_ms = started.elapsed().as_millis(),
            "cold-start phase"
        );
    }

    /// Installs a fork of the parent's semantic index as this worktree's,
    /// unless the worktree's entry already shares the parent's vector store.
    async fn seed_fork(
        &self,
        ref_index: &crate::ref_index::RefIndex,
        parent: &crate::ref_index::RefIndex,
        root: std::path::PathBuf,
        docs: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
        start: WalkStart,
    ) -> Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>, bool)> {
        let base = parent.search_index_cache.read().await.clone();
        let Some(base) = base.filter(|base| base.forkable_at(&parent.canonical_root)) else {
            return Ok((docs, vectors, false));
        };
        if let Some(store) = base.index.vector_store() {
            *ref_index.fork_base.lock().unwrap() = Arc::downgrade(store);
        }
        if ref_index
            .search_index_cache
            .read()
            .await
            .as_ref()
            .is_some_and(|current| current.index.shares_vector_store(&base.index))
        {
            return Ok((docs, vectors, false));
        }
        let started = std::time::Instant::now();
        let (generation, vector_generation) = (start.generation, start.vector_generation);
        let (docs, vectors, fork) = tokio::task::spawn_blocking(move || {
            let fork = base.fork(&root, &docs, &vectors, generation, vector_generation);
            (docs, vectors, fork)
        })
        .await
        .map_err(|e| crate::error::ContextPlusError::Other(e.to_string()))?;
        let Some(fork) = fork else {
            return Ok((docs, vectors, false));
        };
        let installed = fork.install(
            &mut *ref_index.search_index_cache.write().await,
            start.seen.as_ref(),
        );
        tracing::info!(
            phase = "semantic_fork",
            ref_id = %ref_index.cas_ref_id_hex,
            parent_ref_id = %parent.cas_ref_id_hex,
            installed,
            elapsed_ms = started.elapsed().as_millis(),
            "cold-start phase"
        );
        Ok((docs, vectors, installed))
    }

    /// Forks the parent's semantic index over a worktree's warmup `files`, one
    /// generation behind so its first query serves the fork while a background
    /// walk queues the changed files for fill. `false` when the worktree holds
    /// no fork of the parent's store.
    pub(crate) async fn fork_warmup(
        &self,
        ref_index: &Arc<crate::ref_index::RefIndex>,
        files: &crate::server::ProjectCache,
    ) -> bool {
        let Some(parent_id) = ref_index.parent_ref_id else {
            return false;
        };
        let Some(parent) = self.state.ref_index(parent_id).await else {
            return false;
        };
        self.build_parent_index(&parent, ref_index).await;
        let base = parent.search_index_cache.read().await.clone();
        let Some(base) = base.filter(|base| base.forkable_at(&parent.canonical_root)) else {
            return false;
        };
        let start = WalkStart {
            generation: ref_index
                .cache_generation
                .load(std::sync::atomic::Ordering::Acquire),
            vector_generation: ref_index
                .semantic_vector_generation
                .load(std::sync::atomic::Ordering::Acquire),
            seen: ref_index
                .search_index_cache
                .read()
                .await
                .as_ref()
                .map(Arc::downgrade),
        };
        let Some((docs, vectors)) = self
            .warmup_documents(ref_index, files, Some(Arc::clone(&base)))
            .await
        else {
            return false;
        };
        let root = ref_index.canonical_root.clone();
        if let Ok((.., true)) = self
            .seed_fork(ref_index, &parent, root, docs, vectors, start)
            .await
        {
            ref_index
                .cache_generation
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        ref_index
            .search_index_cache
            .read()
            .await
            .as_ref()
            .is_some_and(|entry| entry.index.shares_vector_store(&base.index))
    }

    /// Installs a primary's index of its whole root from its warmup `files`,
    /// one generation behind so its first query walks and updates it, unless
    /// the primary already holds one.
    pub(crate) async fn primary_warmup(
        &self,
        ref_index: &Arc<crate::ref_index::RefIndex>,
        files: &crate::server::ProjectCache,
    ) {
        let seen = {
            let current = ref_index.search_index_cache.read().await;
            if current
                .as_ref()
                .is_some_and(|entry| entry.search_root() == ref_index.canonical_root)
            {
                return;
            }
            current.as_ref().map(Arc::downgrade)
        };
        let generation = ref_index
            .cache_generation
            .load(std::sync::atomic::Ordering::Acquire);
        let vector_generation = ref_index
            .semantic_vector_generation
            .load(std::sync::atomic::Ordering::Acquire);
        let Some((docs, vectors)) = self.warmup_documents(ref_index, files, None).await else {
            return;
        };
        if vectors.iter().all(Option::is_none) {
            return;
        }
        let root = ref_index.canonical_root.clone();
        let Ok(entry) = tokio::task::spawn_blocking(move || {
            CachedSearchIndex::build(&root, docs, vectors, generation, vector_generation, None)
        })
        .await
        else {
            return;
        };
        if entry.install(
            &mut *ref_index.search_index_cache.write().await,
            seen.as_ref(),
        ) {
            ref_index
                .cache_generation
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
    }

    /// The walk's documents of `files`, reusing `parent`'s, with vectors from
    /// this ref's or its ancestors' caches and no Ollama call.
    async fn warmup_documents(
        &self,
        ref_index: &Arc<crate::ref_index::RefIndex>,
        files: &crate::server::ProjectCache,
        parent: Option<Arc<CachedSearchIndex>>,
    ) -> Option<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)> {
        let max_size = self.config.max_embed_file_size;
        let files: Vec<(String, Arc<String>)> = files
            .file_entries
            .iter()
            .filter(|entry| !entry.is_directory)
            .filter_map(|entry| {
                let content = files.file_content.get(&entry.relative_path)?;
                (content.len() <= max_size)
                    .then(|| (entry.relative_path.clone(), Arc::clone(content)))
            })
            .collect();
        let config = self.config.clone();
        let seed_ref = Arc::clone(ref_index);
        let docs = tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            let seeds = crate::server::snapshots::file_seed(&config, &seed_ref);
            let parent_documents = documents_by_path(parent.as_deref());
            files
                .par_iter()
                .map(|(path, content)| {
                    let hash = content_hash(content);
                    let (mut doc, _) =
                        walk_document(path, content, &hash, seeds.as_deref(), &parent_documents);
                    doc.source_hash = hash;
                    doc
                })
                .collect::<Vec<_>>()
        })
        .await
        .ok()?;
        let mut vectors = vec![None; docs.len()];
        let mut caches = vec![Arc::clone(&ref_index.embedding_cache)];
        let mut ancestor_id = ref_index.parent_ref_id;
        let mut visited = std::collections::HashSet::new();
        while let Some(id) = ancestor_id.filter(|id| visited.insert(*id)) {
            let Some(ancestor) = self.state.ref_index(id).await else {
                break;
            };
            caches.push(Arc::clone(&ancestor.embedding_cache));
            ancestor_id = ancestor.parent_ref_id;
        }
        for cache in caches {
            let cache = cache.read().await;
            for (vector, doc) in vectors.iter_mut().zip(&docs) {
                if vector.is_none()
                    && let Some(entry) = cache
                        .get(&doc.path)
                        .filter(|entry| entry.hash == doc.source_hash)
                {
                    *vector = Some(entry.vector.clone());
                }
            }
        }
        Some((docs, vectors))
    }
}

/// Documents of `index` by path.
fn documents_by_path(index: Option<&CachedSearchIndex>) -> HashMap<&str, &SearchDocument> {
    index
        .map(|cached| {
            cached
                .index
                .documents()
                .iter()
                .map(|doc| (doc.path.as_str(), doc))
                .collect()
        })
        .unwrap_or_default()
}

/// A walked file's document: its snapshot seed's or its parent's when the
/// content matches, else a fresh parse; `true` when reused.
fn walk_document(
    rel_path: &str,
    content: &str,
    hash: &str,
    seeds: Option<&crate::tools::semantic_search::DocumentSeeds>,
    parent_documents: &HashMap<&str, &SearchDocument>,
) -> (SearchDocument, bool) {
    let doc_content = semantic_embedding_content(rel_path, content);
    let seeded = match seeds.and_then(|s| s.get(rel_path)) {
        Some(seed) => seed.document(rel_path.to_string(), hash, doc_content),
        None => Err(doc_content),
    };
    let seeded = seeded.or_else(|doc_content| match parent_documents.get(rel_path) {
        Some(doc) if doc.source_hash == hash && doc.content == doc_content => Ok((*doc).clone()),
        _ => Err(doc_content),
    });
    match seeded {
        Ok(doc) => (doc, true),
        Err(doc_content) => (
            crate::tools::semantic_search::file_document(
                rel_path.to_string(),
                content,
                doc_content,
            ),
            false,
        ),
    }
}

/// A worktree's semantic slot and generations when its walk began.
struct WalkStart {
    generation: u64,
    vector_generation: u64,
    seen: Option<std::sync::Weak<CachedSearchIndex>>,
}

#[derive(Clone)]
struct FillDocument {
    path: String,
    hash: String,
    text: String,
    owner: Option<std::sync::Weak<()>>,
}

impl FillDocument {
    async fn is_current(&self, root: &Path, max_size: usize) -> bool {
        let path = root.join(&self.path);
        let hash = self.hash.clone();
        tokio::task::spawn_blocking(move || {
            use std::io::Read;
            let mut options = std::fs::OpenOptions::new();
            options.read(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                // Opening a replacement FIFO must not wait for a writer.
                options.custom_flags(libc::O_NONBLOCK);
            }
            let Ok(file) = options.open(path) else {
                return false;
            };
            let Ok(metadata) = file.metadata() else {
                return false;
            };
            if !metadata.is_file() || metadata.len() > max_size as u64 {
                return false;
            }
            let mut content = String::new();
            (&file)
                .take(max_size as u64)
                .read_to_string(&mut content)
                .is_ok()
                && file
                    .metadata()
                    .is_ok_and(|meta| meta.len() == content.len() as u64)
                && content_hash(&content) == hash
        })
        .await
        .unwrap_or(false)
    }
}

#[derive(Default)]
pub(crate) struct SemanticFill {
    running: bool,
    pending: BTreeMap<String, FillDocument>,
    failures: BTreeMap<String, (String, u8)>,
}

impl SemanticFill {
    fn failed(&self, doc: &FillDocument) -> bool {
        self.failures
            .get(&doc.path)
            .is_some_and(|(hash, n)| hash == &doc.hash && *n >= 3)
    }

    fn record_error(&mut self, doc: &FillDocument) {
        let entry = self
            .failures
            .entry(doc.path.clone())
            .or_insert((doc.hash.clone(), 0));
        if entry.0 != doc.hash {
            *entry = (doc.hash.clone(), 0);
        }
        entry.1 += 1;
        if entry.1 == 3 {
            tracing::warn!(
                path = doc.path,
                hash = doc.hash,
                "Embedding permanently failed after three errors; skipping until content changes"
            );
        }
    }
}

type FileVectors = tokio::sync::RwLock<HashMap<String, CacheEntry>>;

/// Whether `parent` holds the vector of `entry`, the content of `path`.
fn inherits(parent: &HashMap<String, CacheEntry>, path: &str, entry: &CacheEntry) -> bool {
    parent.get(path).is_some_and(|held| held.hash == entry.hash)
}

/// Persists the vectors of `ref_index`; a worktree's only those its parent
/// lacks, dropping the rest from disk.
async fn persist_fill(
    ref_index: &crate::ref_index::RefIndex,
    config: &Config,
    parent_vectors: Option<&FileVectors>,
) {
    let (store, inherited) = {
        let cache = ref_index.embedding_cache.read().await;
        match parent_vectors {
            Some(parent) => {
                let parent = parent.read().await;
                let mut own = HashMap::new();
                let mut inherited = Vec::new();
                for (path, entry) in cache.iter() {
                    if inherits(&parent, path, entry) {
                        inherited.push(path.clone());
                    } else {
                        own.insert(path.clone(), entry.clone());
                    }
                }
                (
                    crate::core::embeddings::VectorStore::from_cache(&own),
                    inherited,
                )
            }
            None => (
                crate::core::embeddings::VectorStore::from_cache(&cache),
                Vec::new(),
            ),
        }
    };
    if let Some(store) = store {
        let root = ref_index.root_dir.clone();
        let name = cache_name("embeddings", config);
        match tokio::task::spawn_blocking(move || {
            rkyv_store::save_vector_store_merged_with_deletions(&root, &name, &store, &inherited)
        })
        .await
        {
            Ok(Ok(())) => {}
            result => tracing::warn!(?result, "Failed to persist background embeddings"),
        }
    }
}

async fn run_fill(
    ref_index: Arc<crate::ref_index::RefIndex>,
    ollama: OllamaClient,
    config: Config,
    parent_vectors: Option<Arc<FileVectors>>,
) {
    let mut completed = 0usize;
    loop {
        let batch: Vec<_> = {
            let fill = ref_index.semantic_fill.lock().await;
            fill.pending
                .values()
                .filter(|doc| {
                    doc.owner
                        .as_ref()
                        .is_none_or(|owner| owner.upgrade().is_none())
                })
                .take(8)
                .cloned()
                .collect()
        };
        if batch.is_empty() {
            if !ref_index.semantic_fill.lock().await.pending.is_empty() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                continue;
            }
            persist_fill(&ref_index, &config, parent_vectors.as_deref()).await;
            let mut fill = ref_index.semantic_fill.lock().await;
            if fill.pending.is_empty() {
                fill.running = false;
                return;
            }
            continue;
        }
        let mut batches = vec![batch];
        while let Some(batch) = batches.pop() {
            let texts: Vec<_> = batch.iter().map(|d| d.text.clone()).collect();
            let outcome = tokio::time::timeout(
                std::time::Duration::from_millis(config.embed_fill_batch_timeout_ms),
                ollama.embed_documents(&texts),
            )
            .await;
            if outcome.is_err() && batch.len() > 1 {
                let mid = batch.len().div_ceil(2);
                batches.push(batch[..mid].to_vec());
                batches.push(batch[mid..].to_vec());
                continue;
            }
            let mut current = Vec::with_capacity(batch.len());
            if matches!(&outcome, Ok(Ok(vectors)) if vectors.len() == batch.len() && vectors.iter().all(|v| !v.is_empty()))
            {
                for doc in &batch {
                    current.push(
                        doc.is_current(&ref_index.canonical_root, config.max_embed_file_size)
                            .await,
                    );
                }
            }
            let mut fill = ref_index.semantic_fill.lock().await;
            let mut ready = Vec::new();
            match outcome {
                Ok(Ok(vectors))
                    if vectors.len() == batch.len() && vectors.iter().all(|v| !v.is_empty()) =>
                {
                    let mut cache = ref_index.embedding_cache.write().await;
                    for ((doc, vector), current) in batch.iter().zip(vectors).zip(current) {
                        let pending_matches = fill
                            .pending
                            .get(&doc.path)
                            .is_some_and(|cur| cur.hash == doc.hash);
                        if pending_matches {
                            fill.pending.remove(&doc.path);
                        }
                        if current && pending_matches {
                            ready.push((doc.path.clone(), doc.hash.clone(), vector.clone()));
                            cache.insert(
                                doc.path.clone(),
                                CacheEntry {
                                    hash: doc.hash.clone(),
                                    vector,
                                },
                            );
                            completed += 1;
                        }
                    }
                    if ready.is_empty() {
                        continue;
                    }
                    let vector_generation = ref_index
                        .semantic_vector_generation
                        .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                        + 1;
                    let mut index = ref_index.search_index_cache.write().await;
                    if index
                        .as_ref()
                        .is_some_and(|entry| !entry.has_vector_shape())
                    {
                        // A vectorless bootstrap has no searchable generation to preserve.
                        *index = None;
                    } else if let Some(entry) = index.as_mut() {
                        crate::tools::semantic_search::CachedSearchIndex::refresh_vectors(
                            entry,
                            &ref_index.canonical_root,
                            ready,
                            vector_generation,
                        );
                    }
                }
                result => {
                    for doc in &batch {
                        if !fill
                            .pending
                            .get(&doc.path)
                            .is_some_and(|cur| cur.hash == doc.hash)
                        {
                            continue;
                        }
                        if result.is_ok() {
                            fill.record_error(doc);
                        }
                        if (result.is_err() || fill.failed(doc))
                            && fill
                                .pending
                                .get(&doc.path)
                                .is_some_and(|cur| cur.hash == doc.hash)
                        {
                            fill.pending.remove(&doc.path);
                        }
                    }
                }
            }
            drop(fill);
            if completed >= 64 {
                persist_fill(&ref_index, &config, parent_vectors.as_deref()).await;
                completed = 0;
            }
        }
    }
}
