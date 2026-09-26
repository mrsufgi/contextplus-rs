// Adapter structs that bridge shared server state to tool function traits.
// Extracted from server.rs (Round 11D) to reduce server.rs line count.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use crate::cache::rkyv_store;
use crate::config::Config;
use crate::core::embeddings::{CacheEntry, OllamaClient, content_hash};
use crate::core::tree_sitter::parse_with_tree_sitter;
use crate::core::walker::walk_with_config;
use crate::error::Result;
use crate::server::{SharedState, build_embedding_document, cache_name};
use crate::tools::semantic_search::{
    EmbedFn, SearchDocument, SymbolSearchEntry, WalkAndIndexFn, extract_plain_text_header,
    is_text_index_candidate, semantic_embedding_content,
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
    fn vector_generation(&self, root: &Path) -> u64 {
        let canonical = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        self.state
            .refs
            .try_read()
            .ok()
            .and_then(|refs| {
                refs.values()
                    .find(|r| r.canonical_root == canonical)
                    .map(|r| {
                        r.semantic_vector_generation
                            .load(std::sync::atomic::Ordering::Acquire)
                    })
            })
            .unwrap_or(0)
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

impl WalkAndIndexFn for RefWalkerIndexer {
    fn vector_generation(&self, _root: &Path) -> u64 {
        self.ref_index
            .semantic_vector_generation
            .load(std::sync::atomic::Ordering::Acquire)
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
}

impl CachedWalkerIndexer {
    fn walk_for_ref(
        &self,
        root_dir: &Path,
        ref_index: Arc<crate::ref_index::RefIndex>,
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
            let entries = walk_with_config(&root, &config);

            let max_file_size = config.max_embed_file_size as u64;
            // Read all files concurrently (up to 32 at a time)
            let mut join_set = tokio::task::JoinSet::new();
            for (i, entry) in entries.iter().enumerate() {
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

            let mut docs = Vec::new();
            let mut content_hashes = Vec::new();
            let mut embedding_texts = Vec::new();

            for (_, rel_path, maybe_content) in &file_contents {
                let content = match maybe_content {
                    Some(c) => c,
                    None => continue,
                };

                if is_text_index_candidate(rel_path) {
                    let truncated = semantic_embedding_content(rel_path, content);
                    let header = extract_plain_text_header(&truncated);
                    content_hashes.push((rel_path.clone(), content_hash(content)));
                    embedding_texts.push(build_embedding_document(
                        rel_path,
                        content,
                        config.embed_doc_shape,
                    ));
                    docs.push(SearchDocument::new(
                        rel_path.clone(),
                        header,
                        vec![],
                        vec![],
                        truncated,
                    ));
                    continue;
                }

                let ext = rel_path.rsplit('.').next().unwrap_or("");
                let symbols = parse_with_tree_sitter(content, ext).unwrap_or_default();
                let header = crate::core::parser::extract_header(content);

                let symbol_names: Vec<String> = symbols.iter().map(|s| s.name.clone()).collect();
                let symbol_entries: Vec<SymbolSearchEntry> = symbols
                    .iter()
                    .map(|s| SymbolSearchEntry {
                        name: s.name.clone(),
                        kind: Some(s.kind.clone()),
                        line: s.line,
                        end_line: Some(s.end_line),
                        signature: s.signature.clone(),
                    })
                    .collect();

                let doc_content = semantic_embedding_content(rel_path, content);
                let embedding_text =
                    build_embedding_document(rel_path, content, config.embed_doc_shape);
                content_hashes.push((rel_path.clone(), content_hash(content)));
                embedding_texts.push(embedding_text);

                docs.push(SearchDocument::new(
                    rel_path.clone(),
                    header,
                    symbol_names,
                    symbol_entries,
                    doc_content,
                ));
            }

            for doc in &mut docs {
                doc.path = Path::new(&doc.path)
                    .strip_prefix(prefix)
                    .unwrap_or(Path::new(&doc.path))
                    .to_string_lossy()
                    .into_owned();
            }

            #[cfg(test)]
            test_seams::after_file_snapshot(&root, &content_hashes).await;

            if docs.is_empty() {
                return Ok((docs, Vec::new()));
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
                let ref_index = ref_index.clone();
                let ollama = ollama.clone();
                let config = config.clone();
                tokio::spawn(async move {
                    run_fill(ref_index, ollama, config).await;
                });
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

            Ok((docs, vectors))
        })
    }
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

async fn persist_fill(ref_index: &crate::ref_index::RefIndex, config: &Config) {
    let store =
        crate::core::embeddings::VectorStore::from_cache(&*ref_index.embedding_cache.read().await);
    if let Some(store) = store {
        let root = ref_index.root_dir.clone();
        let name = cache_name("embeddings", config);
        match tokio::task::spawn_blocking(move || {
            rkyv_store::save_vector_store_merged(&root, &name, &store)
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
            persist_fill(&ref_index, &config).await;
            ref_index
                .semantic_vector_generation
                .fetch_add(1, std::sync::atomic::Ordering::Release);
            *ref_index.search_index_cache.write().await = None;
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
                    ref_index
                        .semantic_vector_generation
                        .fetch_add(1, std::sync::atomic::Ordering::Release);
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
                ref_index
                    .semantic_vector_generation
                    .fetch_add(1, std::sync::atomic::Ordering::Release);
                *ref_index.search_index_cache.write().await = None;
                persist_fill(&ref_index, &config).await;
                completed = 0;
            }
        }
    }
}
