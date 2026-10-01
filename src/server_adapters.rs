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
    CachedSearchIndex, EmbedFn, IndexFingerprint, SearchDocument, SearchIndex, WalkAndIndexFn,
    WalkOutcome, semantic_embedding_content,
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

    fn child_lookup_slots() -> &'static Mutex<BTreeMap<PathBuf, Arc<AsyncPause>>> {
        static SLOTS: OnceLock<Mutex<BTreeMap<PathBuf, Arc<AsyncPause>>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    /// Pauses the next worktree vector lookup of the ref rooted at `root`
    /// after it reads the worktrees' caches.
    pub(crate) fn pause_after_child_lookup(root: &Path) -> Arc<AsyncPause> {
        let pause = Arc::new(AsyncPause::new());
        child_lookup_slots()
            .lock()
            .unwrap()
            .insert(root.to_path_buf(), Arc::clone(&pause));
        pause
    }

    pub(crate) async fn after_child_lookup(root: &Path) {
        let pause = child_lookup_slots().lock().unwrap().remove(root);
        if let Some(pause) = pause {
            pause.entered.add_permits(1);
            pause.resume.acquire().await.unwrap().forget();
        }
    }

    fn store_read_slots() -> &'static Mutex<BTreeMap<PathBuf, usize>> {
        static SLOTS: OnceLock<Mutex<BTreeMap<PathBuf, usize>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    /// Counts the persisted-store reads of the worktree at `root` from now on.
    pub(crate) fn record_store_reads(root: &Path) {
        store_read_slots()
            .lock()
            .unwrap()
            .insert(root.to_path_buf(), 0);
    }

    pub(crate) fn store_read(root: &Path) {
        if let Some(reads) = store_read_slots().lock().unwrap().get_mut(root) {
            *reads += 1;
        }
    }

    pub(crate) fn store_reads(root: &Path) -> usize {
        store_read_slots()
            .lock()
            .unwrap()
            .remove(root)
            .unwrap_or_default()
    }

    fn read_slots() -> &'static Mutex<BTreeMap<PathBuf, Vec<String>>> {
        static SLOTS: OnceLock<Mutex<BTreeMap<PathBuf, Vec<String>>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    /// Records the source files read under `root` from now on.
    pub(crate) fn record_reads(root: &Path) {
        read_slots()
            .lock()
            .unwrap()
            .insert(root.canonicalize().unwrap(), Vec::new());
    }

    pub(crate) fn file_read(root: &Path, path: &str) {
        let mut slots = read_slots().lock().unwrap();
        if slots.is_empty() {
            return;
        }
        if let Some(paths) = root
            .canonicalize()
            .ok()
            .and_then(|root| slots.get_mut(&root))
        {
            paths.push(path.to_string());
        }
    }

    pub(crate) fn reads(root: &Path) -> Vec<String> {
        read_slots()
            .lock()
            .unwrap()
            .remove(&root.canonicalize().unwrap())
            .unwrap_or_default()
    }

    fn revalidation_slots() -> &'static Mutex<BTreeMap<PathBuf, Vec<String>>> {
        static SLOTS: OnceLock<Mutex<BTreeMap<PathBuf, Vec<String>>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    /// Records the paths whose content is checked on disk under `root` from now on.
    pub(crate) fn record_revalidations(root: &Path) {
        revalidation_slots()
            .lock()
            .unwrap()
            .insert(root.to_path_buf(), Vec::new());
    }

    pub(crate) fn revalidated(root: &Path, path: &str) {
        if let Some(paths) = revalidation_slots().lock().unwrap().get_mut(root) {
            paths.push(path.to_string());
        }
    }

    pub(crate) fn revalidations(root: &Path) -> Vec<String> {
        revalidation_slots()
            .lock()
            .unwrap()
            .remove(root)
            .unwrap_or_default()
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

    pub(crate) struct BlockingPause {
        entered: Barrier,
        resume: Barrier,
    }

    impl BlockingPause {
        fn new() -> Self {
            Self {
                entered: Barrier::new(2),
                resume: Barrier::new(2),
            }
        }

        pub(crate) fn wait_until_entered(&self) {
            self.entered.wait();
        }

        pub(crate) fn resume(&self) {
            self.resume.wait();
        }

        fn enter(&self) {
            self.entered.wait();
            self.resume.wait();
        }
    }

    fn metadata_slots() -> &'static Mutex<BTreeMap<PathBuf, Arc<BlockingPause>>> {
        static SLOTS: OnceLock<Mutex<BTreeMap<PathBuf, Arc<BlockingPause>>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    pub(crate) fn pause_after_metadata_enumeration(root: &Path) -> Arc<BlockingPause> {
        let pause = Arc::new(BlockingPause::new());
        metadata_slots()
            .lock()
            .unwrap()
            .insert(root.to_path_buf(), Arc::clone(&pause));
        pause
    }

    pub(crate) fn after_metadata_enumeration(root: &Path) {
        let pause = metadata_slots().lock().unwrap().remove(root);
        if let Some(pause) = pause {
            pause.enter();
        }
    }

    fn outline_parse_slots() -> &'static Mutex<BTreeMap<PathBuf, Arc<BlockingPause>>> {
        static SLOTS: OnceLock<Mutex<BTreeMap<PathBuf, Arc<BlockingPause>>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    /// Pauses the next file a directory outline of the ref rooted at `root`
    /// parses, on the structural pool thread parsing it.
    pub(crate) fn pause_outline_parse(root: &Path) -> Arc<BlockingPause> {
        let pause = Arc::new(BlockingPause::new());
        outline_parse_slots()
            .lock()
            .unwrap()
            .insert(root.to_path_buf(), Arc::clone(&pause));
        pause
    }

    pub(crate) fn outline_parse(root: &Path) {
        let pause = {
            let mut slots = outline_parse_slots().lock().unwrap();
            if slots.is_empty() {
                return;
            }
            slots.remove(root)
        };
        if let Some(pause) = pause {
            pause.enter();
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

    /// The ref's semantic entry when it is current for `root`, with `root`'s
    /// path inside it: walked from `root` or an ancestor, settled, and
    /// unchanged since by the tracker's generation or, with the tracker off,
    /// by its files' metadata.
    pub(crate) async fn current_index(
        &self,
        root: &Path,
    ) -> Option<(Arc<CachedSearchIndex>, std::path::PathBuf)> {
        use std::sync::atomic::Ordering;
        let ref_index = &self.ref_index;
        let entry = ref_index.search_index_cache.read().await.clone()?;
        let canonical = tokio::fs::canonicalize(root).await.ok()?;
        let search_root = entry.search_root();
        if !search_root.is_absolute() {
            return None;
        }
        let prefix = canonical.strip_prefix(search_root).ok()?.to_path_buf();
        if !entry.is_settled(ref_index.semantic_vector_generation.load(Ordering::Acquire)) {
            return None;
        }
        let current = if self.walker.config.embed_tracker_mode != crate::config::TrackerMode::Off {
            entry.generation.load(Ordering::Acquire)
                == ref_index.cache_generation.load(Ordering::Acquire)
        } else {
            let metadata = self.walker.metadata_fingerprint(search_root).await.ok()?;
            metadata.is_some() && *entry.metadata.read().unwrap() == metadata
        };
        current.then_some((entry, prefix))
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
        let Some(base) = forkable_base(&parent).await else {
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

    fn walk_or_install(
        &self,
        root_dir: &Path,
    ) -> crate::tools::semantic_search::WalkOrInstallFuture<'_> {
        self.walker
            .walk_ref(root_dir, self.ref_index.clone(), None, false)
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
        let walk = self.walk_ref(root_dir, ref_index, candidates, true);
        Box::pin(async move {
            match walk.await? {
                WalkOutcome::Documents(docs, vectors) => Ok((docs, vectors)),
                WalkOutcome::Installed(_) => unreachable!("a walk for documents returns them"),
            }
        })
    }

    /// Walks `root_dir` of `ref_index`, or its `candidates`, into documents and
    /// vectors. A full walk of a worktree forks its parent's index when it
    /// can, and returns the fork it installed when no `documents` are wanted.
    fn walk_ref(
        &self,
        root_dir: &Path,
        ref_index: Arc<crate::ref_index::RefIndex>,
        candidates: Option<std::collections::HashSet<String>>,
        documents: bool,
    ) -> crate::tools::semantic_search::WalkOrInstallFuture<'_> {
        let root = root_dir.to_path_buf();
        Box::pin(async move {
            let canonical = std::fs::canonicalize(&root).unwrap_or_else(|_| root.clone());
            let prefix = canonical
                .strip_prefix(&ref_index.canonical_root)
                .unwrap_or(Path::new(""));
            let full_walk = candidates.is_none() && prefix.as_os_str().is_empty();
            let fork_parent = match ref_index.parent_ref_id {
                Some(parent_id) if full_walk => self.state.ref_index(parent_id).await,
                _ => None,
            };
            if let Some(parent) = &fork_parent {
                self.build_parent_index(parent, &ref_index).await;
            }
            let walk_start = WalkStart::capture(&ref_index).await;
            #[cfg(test)]
            ref_index
                .semantic_walks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // A worktree forks its parent's index when it can.
            let base = match &fork_parent {
                Some(parent) => forkable_base(parent).await,
                None => None,
            };
            let replacement_dims = ref_index
                .search_index_cache
                .read()
                .await
                .as_ref()
                .and_then(|entry| entry.pending_vector_dimensions());
            let base =
                base.filter(|base| replacement_dims.is_none_or(|dims| dims == base.index.dims()));
            let mut walked = None;
            if let (Some(parent), Some(base)) = (&fork_parent, &base) {
                match self.fork_documents(&root, parent, base).await {
                    Ok(forked) => {
                        return self
                            .fork_walk(
                                &root, &ref_index, parent, base, forked, walk_start, documents,
                            )
                            .await;
                    }
                    Err(files) => walked = files,
                }
            }

            let (docs, content_hashes, embedding_texts) = self
                .read_documents(
                    &root,
                    prefix,
                    &ref_index,
                    candidates.as_ref(),
                    full_walk,
                    base.clone(),
                    walked,
                )
                .await?;
            #[cfg(test)]
            test_seams::after_file_snapshot(&root, &content_hashes).await;
            if docs.is_empty() {
                return Ok(WalkOutcome::Documents(docs, Vec::new()));
            }
            if let (Some(parent), Some(base)) = (&fork_parent, &base)
                && let Some(shared) = Shared::of(&base.index, &docs)
            {
                let forked = shared.forked(docs, content_hashes, embedding_texts);
                return self
                    .fork_walk(
                        &root, &ref_index, parent, base, forked, walk_start, documents,
                    )
                    .await;
            }
            let vectors = self
                .walk_vectors(&root, &ref_index, &content_hashes, &embedding_texts, true)
                .await?;
            match fork_parent {
                Some(parent) => self
                    .seed_fork(&ref_index, &parent, canonical, docs, vectors, walk_start)
                    .await
                    .map(|(docs, vectors, _)| WalkOutcome::Documents(docs, vectors)),
                None => Ok(WalkOutcome::Documents(docs, vectors)),
            }
        })
    }

    /// Forks `base` over a worktree walk's `forked` documents: vectors for the
    /// worktree's own documents alone, then the fork installed unless the
    /// worktree's entry already shares `base`'s store. The fork it installed
    /// when no `documents` are wanted, else the walk's documents and vectors.
    #[allow(clippy::too_many_arguments)]
    async fn fork_walk(
        &self,
        root: &Path,
        ref_index: &Arc<crate::ref_index::RefIndex>,
        parent: &crate::ref_index::RefIndex,
        base: &Arc<CachedSearchIndex>,
        forked: Forked,
        start: WalkStart,
        documents: bool,
    ) -> Result<WalkOutcome> {
        let Forked {
            documents: walked,
            deleted,
        } = forked;
        let mut order = Vec::with_capacity(walked.len());
        let mut built = Vec::new();
        let mut content_hashes = Vec::new();
        let mut embedding_texts = Vec::new();
        for document in walked {
            order.push(match document {
                ForkDocument::Shared(at) => Ok(at),
                ForkDocument::Own(own) => {
                    let (doc, hash, text) = *own;
                    content_hashes.push((doc.path.clone(), hash));
                    embedding_texts.push(text);
                    built.push(doc);
                    Err(built.len() - 1)
                }
            });
        }
        let vectors = self
            .walk_vectors(root, ref_index, &content_hashes, &embedding_texts, false)
            .await?;
        let fingerprint = {
            let (base, order, built) = (Arc::clone(base), order.clone(), built.clone());
            tokio::task::spawn_blocking(move || {
                IndexFingerprint::of(order.iter().map(|at| match at {
                    Ok(at) => &base.index.documents()[*at],
                    Err(i) => &built[*i],
                }))
            })
            .await
            .map_err(|e| crate::error::ContextPlusError::Other(e.to_string()))?
        };
        let installed = self
            .install_fork(
                ref_index,
                parent,
                Arc::clone(base),
                built.clone(),
                vectors.clone(),
                deleted,
                fingerprint,
                start,
            )
            .await;
        if let Some(installed) = installed.filter(|_| !documents) {
            return Ok(WalkOutcome::Installed(installed));
        }
        let base = Arc::clone(base);
        let walked = tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            order
                .into_par_iter()
                .map(|at| match at {
                    Ok(at) => (
                        SearchDocument::clone(&base.index.documents()[at]),
                        base.index.vector_at(at).map(<[f32]>::to_vec),
                    ),
                    Err(i) => (built[i].clone(), vectors[i].clone()),
                })
                .unzip()
        })
        .await
        .map_err(|e| crate::error::ContextPlusError::Other(e.to_string()))?;
        let (docs, vectors) = walked;
        Ok(WalkOutcome::Documents(docs, vectors))
    }

    /// The vectors of a walk's documents, by path and content hash: this ref's
    /// cached ones, its ancestors', its attached worktrees', and embeddings
    /// within the budget, the rest queued for the background fill. With
    /// `inherit`, ancestors' vectors are copied into this ref's cache once
    /// their files are rechecked; a fork uses them without a copy.
    async fn walk_vectors(
        &self,
        #[cfg_attr(not(test), allow(unused_variables))] root: &Path,
        ref_index: &Arc<crate::ref_index::RefIndex>,
        content_hashes: &[(String, String)],
        embedding_texts: &[String],
        inherit: bool,
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let config = &self.config;
        let ollama = &self.ollama;
        let embedding_cache = Arc::clone(&ref_index.embedding_cache);
        let mut borrowed = vec![false; content_hashes.len()];
        let parent_vectors = match ref_index.parent_ref_id {
            Some(parent_id) => self
                .state
                .ref_index(parent_id)
                .await
                .map(|parent| Arc::clone(&parent.embedding_cache)),
            None => None,
        };
        if let Some(parent_vectors) = &parent_vectors {
            reload_own_vectors(ref_index, parent_vectors, config, false).await;
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
        let mut vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(content_hashes.len());
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
                    // A fork keeps only its own vectors in its cache.
                    if inherit {
                        inherited.push((idx, entry.clone()));
                    } else {
                        borrowed[idx] = true;
                    }
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
        // A fork's entry holds the vectors it took from a parent that has since
        // moved off their content; they become this ref's own.
        let unfound: Vec<usize> = uncached_indices
            .iter()
            .copied()
            .filter(|&idx| vectors[idx].is_none())
            .collect();
        if !unfound.is_empty() {
            let wanted: Vec<(&str, &str)> = unfound
                .iter()
                .map(|&idx| {
                    let (path, hash) = &content_hashes[idx];
                    (path.as_str(), hash.as_str())
                })
                .collect();
            for (idx, vector) in unfound.iter().zip(held_vectors(ref_index, &wanted).await) {
                if let Some((_, vector)) = vector {
                    vectors[*idx] = Some(vector.clone());
                    let hash = content_hashes[*idx].1.clone();
                    inherited.push((*idx, CacheEntry { hash, vector }));
                }
            }
        }
        let mut current = vec![true; content_hashes.len()];
        let started = std::time::Instant::now();
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
            let validate_ms = started.elapsed().as_millis();
            let fill = ref_index.semantic_fill.lock().await;
            let mut cache = embedding_cache.write().await;
            let lock_ms = started.elapsed().as_millis() - validate_ms;
            tracing::info!(
                phase = "semantic_inherit",
                ref_id = %ref_index.cas_ref_id_hex,
                inherited = inherited.len(),
                validate_ms,
                lock_ms,
                "cold-start phase"
            );
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
        let (miss_indices, misses): (Vec<usize>, Vec<VectorMiss>) = uncached_indices
            .iter()
            .filter(|&&idx| current[idx] && vectors[idx].is_none())
            .map(|&idx| {
                let (path, hash) = &content_hashes[idx];
                let miss = VectorMiss {
                    path: path.clone(),
                    hash: hash.clone(),
                    cached_hash: observed[idx].0.clone(),
                    pending_hash: observed[idx].1.clone(),
                };
                (idx, miss)
            })
            .unzip();
        let adopted =
            adopt_worktree_vectors(&self.state, ref_index, &misses, config.max_embed_file_size)
                .await;
        for (idx, vector) in miss_indices.into_iter().zip(adopted) {
            if vector.is_some() {
                vectors[idx] = vector;
            }
        }
        uncached_indices.retain(|&idx| vectors[idx].is_none());

        #[cfg(test)]
        test_seams::after_cache_snapshot(root).await;

        let cache_entries = embedding_cache.read().await.len();
        tracing::info!(
            ref_id = %ref_index.cas_ref_id_hex,
            cache_entries,
            cached = content_hashes.len() - uncached_indices.len(),
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
            if !current[idx] || borrowed[idx] {
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
            let owner = Arc::clone(ref_index);
            let ollama = ollama.clone();
            let config = config.clone();
            let parent_vectors = parent_vectors.clone();
            let state = Arc::clone(&self.state);
            let task = tokio::spawn(async move {
                run_fill(state, owner, ollama, config, parent_vectors).await;
            });
            ref_index.track_background_task(&task);
        }
        drop(fill);
        let deadline =
            tokio::time::Instant::now() + std::time::Duration::from_millis(config.embed_budget_ms);
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
                            doc.is_current(&ref_index.canonical_root, config.max_embed_file_size)
                                .await,
                        );
                    }
                    let mut fill = ref_index.semantic_fill.lock().await;
                    let mut cache = embedding_cache.write().await;
                    for (((idx, doc), vector), current) in chunk.iter().zip(result).zip(current) {
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
                .filter(|(i, vector)| {
                    !borrowed[*i] && vector.as_ref().is_none_or(|v| v.len() != dims)
                })
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
        Ok(vectors)
    }

    /// Walks `root` and reads its files, or the `candidates` among them, into
    /// documents with their content hashes and embedding texts. A file whose
    /// content matches the snapshot's document, or its parent's (`base` when
    /// forkable), keeps that document's parsed fields. A full walk takes the
    /// files already `walked` instead of walking and reading them.
    #[allow(clippy::too_many_arguments)]
    async fn read_documents(
        &self,
        root: &Path,
        prefix: &Path,
        ref_index: &Arc<crate::ref_index::RefIndex>,
        candidates: Option<&std::collections::HashSet<String>>,
        full_walk: bool,
        base: Option<Arc<CachedSearchIndex>>,
        walked: Option<WalkedFiles>,
    ) -> Result<(Vec<SearchDocument>, Vec<(String, String)>, Vec<String>)> {
        let config = &self.config;
        let started = std::time::Instant::now();
        let max_file_size = config.max_embed_file_size as u64;
        let mut walk_ms = 0;
        let file_contents: Vec<(usize, String, Option<Arc<String>>)> = match walked {
            Some(walked) => walked
                .into_iter()
                .enumerate()
                .map(|(i, (path, content))| {
                    let rel_path = prefix.join(path).to_string_lossy().into_owned();
                    let content = content.filter(|c| c.len() as u64 <= max_file_size);
                    (i, rel_path, content)
                })
                .collect(),
            None => {
                let entries = walk_with_config(root, config);
                walk_ms = started.elapsed().as_millis();
                // Read all files concurrently (up to 32 at a time)
                let mut join_set = tokio::task::JoinSet::new();
                for (i, entry) in entries.iter().enumerate() {
                    if candidates.is_some_and(|paths| !paths.contains(&entry.relative_path)) {
                        continue;
                    }
                    let full_path = root.join(&entry.relative_path);
                    let rel_path = prefix
                        .join(&entry.relative_path)
                        .to_string_lossy()
                        .into_owned();
                    #[cfg(test)]
                    test_seams::file_read(root, &entry.relative_path);
                    join_set.spawn(async move {
                        if let Ok(meta) = tokio::fs::metadata(&full_path).await
                            && meta.len() > max_file_size
                        {
                            return (i, rel_path, None);
                        }
                        let content = tokio::fs::read_to_string(&full_path).await.ok();
                        (i, rel_path, content.map(Arc::new))
                    });
                }
                let mut file_contents = Vec::with_capacity(entries.len());
                while let Some(result) = join_set.join_next().await {
                    if let Ok(item) = result {
                        file_contents.push(item);
                    }
                }
                file_contents.sort_unstable_by_key(|(i, _, _)| *i);
                file_contents
            }
        };
        let read_ms = started.elapsed().as_millis() - walk_ms;

        // Parsing runs in parallel; a file whose content matches the
        // snapshot's document keeps that document's parsed fields.
        let doc_shape = config.embed_doc_shape;
        let seed_config = config.clone();
        let seed_ref = Arc::clone(ref_index);
        // A worktree takes the documents of files identical to its parent's.
        let parent_index = match (base, ref_index.parent_ref_id) {
            (Some(base), _) => Some(base),
            (None, Some(parent_id)) => match self.state.ref_index(parent_id).await {
                Some(parent) => parent
                    .search_index_cache
                    .read()
                    .await
                    .clone()
                    .filter(|cached| cached.search_root() == parent.canonical_root),
                None => None,
            },
            (None, None) => None,
        };
        let (mut docs, content_hashes, embedding_texts, reused) =
            tokio::task::spawn_blocking(move || {
                use rayon::prelude::*;
                let seeds = crate::server::snapshots::file_seed(&seed_config, &seed_ref);
                let parent_documents = documents_by_path(parent_index.as_deref());
                let built: Vec<(SearchDocument, (String, String), String, bool)> = file_contents
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
            crate::server::snapshots::file_seed_used(ref_index);
            if reused < docs.len() {
                let change = match reused {
                    0 => crate::server::snapshots::Change::Full,
                    _ => crate::server::snapshots::Change::Files {
                        changed: docs.len() - reused,
                        documents: docs.len(),
                    },
                };
                crate::server::snapshots::schedule_files(config, ref_index, change);
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
        Ok((docs, content_hashes, embedding_texts))
    }

    /// The documents of a full walk of the worktree at `root` whose parent
    /// holds the forkable `base`. A file the parent's cached contents show
    /// unchanged keeps `base`'s document unread when `base` indexed those
    /// contents with a vector; the other files are read. Past the promotion
    /// threshold, the walk's files as far as it has them; `Err(None)` when
    /// the parent's contents are unavailable.
    async fn fork_documents(
        &self,
        root: &Path,
        parent: &Arc<crate::ref_index::RefIndex>,
        base: &Arc<CachedSearchIndex>,
    ) -> std::result::Result<Forked, Option<WalkedFiles>> {
        // The flat file cache the primary holds, however old: its clean blobs
        // and contents were recorded together. A walk never builds one.
        let files = parent
            .project_cache
            .read()
            .await
            .clone()
            .filter(|cache| cache.file_content.base().is_none())
            .ok_or(None)?;
        let started = std::time::Instant::now();
        let config = self.config.clone();
        let walk_root = root.to_path_buf();
        let files = Arc::new(
            tokio::task::spawn_blocking(move || {
                crate::server::load_project_cache(&walk_root, &config, Some(files), false)
            })
            .await
            .map_err(|_| None)?,
        );
        let max_size = self.config.max_embed_file_size;
        if files.file_content.base().is_none() {
            return Err(Some(
                files
                    .file_entries
                    .iter()
                    .filter(|entry| !entry.is_directory)
                    .map(|entry| {
                        let path = entry.relative_path.clone();
                        let content = files.file_content.get(&path).cloned();
                        (path, content)
                    })
                    .collect(),
            ));
        }
        let walk_ms = started.elapsed().as_millis();
        let changed = files.file_content.own().len();

        let (classify_files, classify_base) = (Arc::clone(&files), Arc::clone(base));
        let mut walked = tokio::task::spawn_blocking(move || {
            classify(&classify_files, &classify_base.index, max_size)
        })
        .await
        .map_err(|_| None)?;

        // Files the parent's contents do not vouch for are read as a walk reads them.
        let read_started = std::time::Instant::now();
        let mut join_set = tokio::task::JoinSet::new();
        for (i, (path, walked)) in walked.iter().enumerate() {
            if !matches!(walked, Walked::Read(None)) {
                continue;
            }
            let full_path = root.join(path);
            #[cfg(test)]
            test_seams::file_read(root, path);
            join_set.spawn(async move {
                if let Ok(meta) = tokio::fs::metadata(&full_path).await
                    && meta.len() > max_size as u64
                {
                    return (i, None);
                }
                let content = tokio::fs::read_to_string(&full_path).await.ok();
                (i, content.map(Arc::new))
            });
        }
        let mut read = 0usize;
        while let Some(result) = join_set.join_next().await {
            if let Ok((i, content)) = result {
                read += 1;
                walked[i].1 =
                    content.map_or(Walked::Skipped, |content| Walked::Read(Some(content)));
            }
        }
        let read_ms = read_started.elapsed().as_millis();

        let doc_shape = self.config.embed_doc_shape;
        let base = Arc::clone(base);
        let (walked, forked) = tokio::task::spawn_blocking(move || {
            let forked = Forked::of(&walked, &base.index, doc_shape);
            (walked, forked)
        })
        .await
        .map_err(|_| None)?;
        let Some(forked) = forked else {
            // Refused: the standalone walk takes the contents at hand.
            return Err(walked
                .into_iter()
                .map(|(path, walked)| match walked {
                    Walked::Shared(_) => {
                        let content = files.file_content.get(&path).cloned();
                        Some((path, content))
                    }
                    Walked::Read(Some(content)) => Some((path, Some(content))),
                    Walked::Skipped => Some((path, None)),
                    Walked::Read(None) => None,
                })
                .collect());
        };
        tracing::info!(
            phase = "semantic_walk",
            walk_ms,
            read_ms,
            documents_ms = started.elapsed().as_millis() - walk_ms - read_ms,
            documents = forked.documents.len(),
            reused = forked
                .documents
                .iter()
                .filter(|doc| matches!(doc, ForkDocument::Shared(_)))
                .count(),
            changed,
            read,
            "cold-start phase"
        );
        Ok(forked)
    }

    /// Installs `base` forked with a worktree's `changed` documents, their
    /// `vectors` and its `deleted` paths as the worktree's index, unless the
    /// worktree's entry already shares `base`'s store. `fingerprint` is that
    /// of all the worktree's documents. The fork, when installed.
    #[allow(clippy::too_many_arguments)]
    async fn install_fork(
        &self,
        ref_index: &crate::ref_index::RefIndex,
        parent: &crate::ref_index::RefIndex,
        base: Arc<CachedSearchIndex>,
        changed: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
        deleted: Vec<String>,
        fingerprint: IndexFingerprint,
        start: WalkStart,
    ) -> Option<Arc<CachedSearchIndex>> {
        // Recorded once forked or refused, so a fork dropped by a changed slot is retried.
        let store = base.index.vector_store().map(Arc::downgrade);
        let record = || {
            if let Some(store) = &store {
                *ref_index.fork_base.lock().unwrap() = store.clone();
            }
        };
        if ref_index
            .search_index_cache
            .read()
            .await
            .as_ref()
            .is_some_and(|current| current.index.shares_vector_store(&base.index))
        {
            record();
            return None;
        }
        let started = std::time::Instant::now();
        let root = ref_index.canonical_root.clone();
        let (generation, vector_generation) = (start.generation, start.vector_generation);
        let (changed_count, deleted_count) = (changed.len(), deleted.len());
        let fork = tokio::task::spawn_blocking(move || {
            base.fork_delta(
                &root,
                changed,
                vectors,
                &deleted,
                fingerprint,
                generation,
                vector_generation,
            )
        })
        .await
        .ok()
        .flatten();
        let Some(fork) = fork else {
            record();
            return None;
        };
        let installed = {
            let mut slot = ref_index.search_index_cache.write().await;
            fork.install(&mut slot, start.seen.as_ref())
                .then(|| slot.clone())
                .flatten()
        };
        if installed.is_some() {
            record();
        }
        tracing::info!(
            phase = "semantic_fork",
            ref_id = %ref_index.cas_ref_id_hex,
            parent_ref_id = %parent.cas_ref_id_hex,
            installed = installed.is_some(),
            changed = changed_count,
            deleted = deleted_count,
            elapsed_ms = started.elapsed().as_millis(),
            "cold-start phase"
        );
        installed
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
        if parent.search_index_cache.read().await.is_some() {
            return;
        }
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
        let installed = entry.install(&mut *parent.search_index_cache.write().await, None);
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
        let Some(base) = forkable_base(parent).await else {
            return Ok((docs, vectors, false));
        };
        // Recorded once forked or refused, so a fork dropped by a changed slot is retried.
        let store = base.index.vector_store().map(Arc::downgrade);
        let record = || {
            if let Some(store) = &store {
                *ref_index.fork_base.lock().unwrap() = store.clone();
            }
        };
        if ref_index
            .search_index_cache
            .read()
            .await
            .as_ref()
            .is_some_and(|current| current.index.shares_vector_store(&base.index))
        {
            record();
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
            record();
            return Ok((docs, vectors, false));
        };
        let installed = fork.install(
            &mut *ref_index.search_index_cache.write().await,
            start.seen.as_ref(),
        );
        if installed {
            record();
        }
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
    /// walk queues the changed files for fill. The paths of `files` the fork
    /// it installed shares, which need no vector of the worktree's own; `None`
    /// when the worktree holds no fork of the parent's store.
    pub(crate) async fn fork_warmup(
        &self,
        ref_index: &Arc<crate::ref_index::RefIndex>,
        files: &Arc<crate::server::ProjectCache>,
    ) -> Option<std::collections::HashSet<String>> {
        let parent = self.warmup_parent(ref_index).await?;
        let mut base = forkable_base(&parent).await?;
        let start = WalkStart::capture(ref_index).await;
        let (installed, shared) = if files.file_content.base().is_some() {
            let mut forked = self.warmup_delta(ref_index, files, &base).await;
            // Forks the store the parent holds once the documents are ready.
            if let Some(current) = forkable_base(&parent).await
                && !Arc::ptr_eq(&current, &base)
            {
                base = current;
                forked = self.warmup_delta(ref_index, files, &base).await;
            }
            match forked {
                Some(fork) => {
                    let installed = self
                        .install_fork(
                            ref_index,
                            &parent,
                            Arc::clone(&base),
                            fork.changed,
                            fork.vectors,
                            fork.deleted,
                            fork.fingerprint,
                            start,
                        )
                        .await
                        .is_some();
                    (installed, fork.shared)
                }
                None => {
                    if let Some(store) = base.index.vector_store() {
                        *ref_index.fork_base.lock().unwrap() = Arc::downgrade(store);
                    }
                    (false, Default::default())
                }
            }
        } else {
            let (docs, vectors) = self
                .warmup_documents(ref_index, files, Some(Arc::clone(&base)))
                .await?;
            let root = ref_index.canonical_root.clone();
            let installed = matches!(
                self.seed_fork(ref_index, &parent, root, docs, vectors, start)
                    .await,
                Ok((.., true))
            );
            (installed, Default::default())
        };
        if installed {
            ref_index
                .cache_generation
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            return Some(shared);
        }
        // A concurrent walk may have installed a fork of `base`, sharing what it may.
        ref_index
            .search_index_cache
            .read()
            .await
            .as_ref()
            .is_some_and(|entry| entry.index.shares_vector_store(&base.index))
            .then(Default::default)
    }

    /// The parent a worktree's warmup forks, its index built first from its
    /// cached vectors when it holds none: the warmup makes no Ollama call.
    async fn warmup_parent(
        &self,
        ref_index: &crate::ref_index::RefIndex,
    ) -> Option<Arc<crate::ref_index::RefIndex>> {
        let parent = self.state.ref_index(ref_index.parent_ref_id?).await?;
        if parent.parent_ref_id.is_none()
            && parent.canonical_root != ref_index.canonical_root
            && parent.search_index_cache.read().await.is_none()
            && let Some(files) = parent.project_cache.read().await.clone()
        {
            self.primary_warmup(&parent, &files).await;
        }
        Some(parent)
    }

    /// The paths of a worktree's warmup `files` that a fork of its parent's
    /// index would share, which need no vector of the worktree's own.
    pub(crate) async fn fork_shared_paths(
        &self,
        ref_index: &crate::ref_index::RefIndex,
        files: &Arc<crate::server::ProjectCache>,
    ) -> std::collections::HashSet<String> {
        let Some(parent) = self.warmup_parent(ref_index).await else {
            return Default::default();
        };
        let Some(base) = forkable_base(&parent).await else {
            return Default::default();
        };
        if files.file_content.base().is_none() {
            return Default::default();
        }
        let files = Arc::clone(files);
        let max_size = self.config.max_embed_file_size;
        tokio::task::spawn_blocking(move || {
            classify(&files, &base.index, max_size)
                .into_iter()
                .filter(|(_, walked)| matches!(walked, Walked::Shared(_)))
                .map(|(path, _)| path)
                .collect()
        })
        .await
        .unwrap_or_default()
    }

    /// The fork of `base` over a worktree's layered warmup `files`: its own
    /// documents with vectors from this ref's or its ancestors' caches, its
    /// deleted paths, the paths it shares and the fingerprint of all its
    /// documents. Files the parent's index shares are neither parsed nor
    /// looked up. `None` past the promotion threshold.
    async fn warmup_delta(
        &self,
        ref_index: &Arc<crate::ref_index::RefIndex>,
        files: &Arc<crate::server::ProjectCache>,
        base: &Arc<CachedSearchIndex>,
    ) -> Option<WarmupFork> {
        let max_size = self.config.max_embed_file_size;
        let doc_shape = self.config.embed_doc_shape;
        let (files, classify_base) = (Arc::clone(files), Arc::clone(base));
        let (changed, deleted, shared, fingerprint) = tokio::task::spawn_blocking(move || {
            // The warmup takes every content from its files, as the walk it precedes.
            let walked: Vec<_> = classify(&files, &classify_base.index, max_size)
                .into_iter()
                .map(|(path, walked)| match walked {
                    Walked::Read(None) => {
                        let content = files.file_content.get(&path).cloned();
                        (
                            path,
                            content.map_or(Walked::Skipped, |c| Walked::Read(Some(c))),
                        )
                    }
                    walked => (path, walked),
                })
                .collect();
            let Forked { documents, deleted } =
                Forked::of(&walked, &classify_base.index, doc_shape)?;
            let fingerprint =
                IndexFingerprint::of(documents.iter().map(|document| match document {
                    ForkDocument::Shared(at) => &classify_base.index.documents()[*at],
                    ForkDocument::Own(own) => &own.0,
                }));
            let mut shared = std::collections::HashSet::new();
            let mut changed = Vec::new();
            for document in documents {
                match document {
                    ForkDocument::Own(own) => changed.push(own.0),
                    ForkDocument::Shared(at) => {
                        shared.insert(classify_base.index.documents()[at].path.clone());
                    }
                }
            }
            Some((changed, deleted, shared, fingerprint))
        })
        .await
        .ok()??;
        let vectors = self.cached_vectors(ref_index, &changed).await;
        #[cfg(test)]
        test_seams::after_cache_snapshot(&ref_index.root_dir).await;
        Some(WarmupFork {
            changed,
            vectors,
            deleted,
            shared,
            fingerprint,
        })
    }

    /// Vectors of `docs` from this ref's or its ancestors' caches, by path and
    /// content hash.
    async fn cached_vectors(
        &self,
        ref_index: &crate::ref_index::RefIndex,
        docs: &[SearchDocument],
    ) -> Vec<Option<Vec<f32>>> {
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
            for (vector, doc) in vectors.iter_mut().zip(docs) {
                if vector.is_none()
                    && let Some(entry) = cache
                        .get(&doc.path)
                        .filter(|entry| entry.hash == doc.source_hash)
                {
                    *vector = Some(entry.vector.clone());
                }
            }
        }
        vectors
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
        let vectors = self.cached_vectors(ref_index, &docs).await;
        #[cfg(test)]
        test_seams::after_cache_snapshot(&ref_index.root_dir).await;
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
                .map(|doc| (doc.path.as_str(), &**doc))
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

/// A file whose vector a ref lacks: its path, content hash, and the ref's
/// cached and pending hashes for the path when the miss was seen.
pub(crate) struct VectorMiss {
    pub(crate) path: String,
    pub(crate) hash: String,
    pub(crate) cached_hash: Option<String>,
    pub(crate) pending_hash: Option<String>,
}

impl VectorMiss {
    pub(crate) async fn observe(
        ref_index: &crate::ref_index::RefIndex,
        path: String,
        hash: String,
    ) -> Self {
        let fill = ref_index.semantic_fill.lock().await;
        let cache = ref_index.embedding_cache.read().await;
        let cached_hash = cache.get(&path).map(|entry| entry.hash.clone());
        let pending_hash = fill.pending.get(&path).map(|doc| doc.hash.clone());
        Self {
            path,
            hash,
            cached_hash,
            pending_hash,
        }
    }
}

/// Where a ref holds a vector.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Held {
    Cached,
    Indexed,
    /// In a worktree's persisted store, which saves replace by rename, so a
    /// mapped vector never changes after it is read.
    Persisted,
}

/// The linked git worktrees of the checkout at `root`, read on each call
/// from the `gitdir` record git keeps for each, which `git worktree move`
/// rewrites in place and which may hold a path relative to its own directory.
fn linked_worktree_roots(root: &Path) -> Vec<std::path::PathBuf> {
    let Some(dirs) = crate::core::git_worktree::git_dirs(root) else {
        return Vec::new();
    };
    std::fs::read_dir(dirs.common_dir.join("worktrees"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let gitdir = std::fs::read_to_string(entry.path().join("gitdir")).ok()?;
            entry
                .path()
                .join(gitdir.trim())
                .parent()?
                .canonicalize()
                .ok()
        })
        .collect()
}

/// Vectors the persisted stores of the primary `ref_index`'s linked git
/// worktrees hold at each `(path, hash)` of `wanted`, read only from stores
/// of its embedding config and of the dimensions of its cache, or else of
/// the first such store.
async fn persisted_worktree_vectors(
    state: &SharedState,
    ref_index: &crate::ref_index::RefIndex,
    wanted: Vec<(String, String)>,
) -> Vec<Option<Vec<f32>>> {
    let mut vectors = vec![None; wanted.len()];
    if ref_index.parent_ref_id.is_some() || wanted.is_empty() {
        return vectors;
    }
    let mut dims = {
        let cache = ref_index.embedding_cache.read().await;
        cache
            .values()
            .map(|entry| entry.vector.len())
            .find(|&len| len > 0)
    };
    let root = ref_index.canonical_root.clone();
    let name = cache_name("embeddings", &state.config);
    let read = tokio::task::spawn_blocking(move || {
        for root in linked_worktree_roots(&root) {
            if vectors.iter().all(Option::is_some) {
                break;
            }
            // A removed worktree, or a directory no longer one, has no store of its own.
            if !root.join(".git").is_file() {
                continue;
            }
            #[cfg(test)]
            test_seams::store_read(&root);
            let Ok(Some(store)) = rkyv_store::mmap_vector_store(&root, &name) else {
                continue;
            };
            if store.dims() == 0 || *dims.get_or_insert(store.dims()) != store.dims() {
                continue;
            }
            for (vector, (path, hash)) in vectors.iter_mut().zip(&wanted) {
                if vector.is_none() && store.get_hash(path) == Some(hash.as_str()) {
                    *vector = store.get_vector(path).map(<[f32]>::to_vec);
                }
            }
        }
        vectors
    })
    .await;
    read.unwrap_or_default()
}

/// The vector `ref_index` holds of each `(path, hash)`: its cached one, else
/// its semantic entry's of its whole root, which a fork shares with its parent.
async fn held_vectors(
    ref_index: &crate::ref_index::RefIndex,
    wanted: &[(&str, &str)],
) -> Vec<Option<(Held, Vec<f32>)>> {
    let mut vectors: Vec<_> = {
        let cache = ref_index.embedding_cache.read().await;
        wanted
            .iter()
            .map(|(path, hash)| {
                cache
                    .get(*path)
                    .filter(|entry| entry.hash == *hash)
                    .map(|entry| (Held::Cached, entry.vector.clone()))
            })
            .collect()
    };
    if vectors.iter().all(Option::is_some) {
        return vectors;
    }
    let entry = ref_index.search_index_cache.read().await.clone();
    let Some(entry) = entry.filter(|entry| entry.search_root() == ref_index.canonical_root) else {
        return vectors;
    };
    let held = positions_by_path(&entry.index);
    for (vector, (path, hash)) in vectors.iter_mut().zip(wanted) {
        if vector.is_none()
            && let Some(&at) = held.get(*path)
            && entry.index.documents()[at].source_hash == *hash
        {
            *vector = entry
                .index
                .vector_at(at)
                .map(|vector| (Held::Indexed, vector.to_vec()));
        }
    }
    vectors
}

/// Vectors that `ref_index`'s attached worktrees hold for `misses` at the same
/// path and content hash, copied into its cache. A vector is taken only while
/// the file still has that hash, the worktree still holds it, and the ref's
/// cached and pending hashes for the path are as observed.
pub(crate) async fn adopt_worktree_vectors(
    state: &SharedState,
    ref_index: &crate::ref_index::RefIndex,
    misses: &[VectorMiss],
    max_size: usize,
) -> Vec<Option<Vec<f32>>> {
    let mut vectors = vec![None; misses.len()];
    if misses.is_empty() {
        return vectors;
    }
    let children = state.attached_children(ref_index).await;
    // The attached worktree it was found in; `None` for a persisted store.
    let mut found: Vec<Option<(Option<usize>, Held, CacheEntry)>> = vec![None; misses.len()];
    for (child_idx, child) in children.iter().enumerate() {
        let (unfound, wanted): (Vec<usize>, Vec<(&str, &str)>) = found
            .iter()
            .zip(misses)
            .enumerate()
            .filter(|(_, (slot, _))| slot.is_none())
            .map(|(i, (_, miss))| (i, (miss.path.as_str(), miss.hash.as_str())))
            .unzip();
        if unfound.is_empty() {
            break;
        }
        for (i, vector) in unfound.into_iter().zip(held_vectors(child, &wanted).await) {
            found[i] = vector.map(|(held, vector)| {
                let hash = misses[i].hash.clone();
                (Some(child_idx), held, CacheEntry { hash, vector })
            });
        }
    }
    if found.iter().any(Option::is_none) {
        let (unfound, wanted): (Vec<usize>, Vec<(String, String)>) = found
            .iter()
            .zip(misses)
            .enumerate()
            .filter(|(_, (slot, _))| slot.is_none())
            .map(|(i, (_, miss))| (i, (miss.path.clone(), miss.hash.clone())))
            .unzip();
        let persisted = persisted_worktree_vectors(state, ref_index, wanted).await;
        for (i, vector) in unfound.into_iter().zip(persisted) {
            found[i] = vector.map(|vector| {
                let hash = misses[i].hash.clone();
                (None, Held::Persisted, CacheEntry { hash, vector })
            });
        }
    }
    #[cfg(test)]
    test_seams::after_child_lookup(&ref_index.canonical_root).await;
    let mut current = vec![false; misses.len()];
    for ((current, slot), miss) in current.iter_mut().zip(&found).zip(misses) {
        if slot.is_some() {
            *current = FillDocument {
                path: miss.path.clone(),
                hash: miss.hash.clone(),
                text: String::new(),
                owner: None,
            }
            .is_current(&ref_index.canonical_root, max_size)
            .await;
        }
    }
    // Taken only while the worktree still holds it where it was found.
    for (child_idx, child) in children.iter().enumerate() {
        let (still, wanted): (Vec<usize>, Vec<(&str, &str)>) = found
            .iter()
            .zip(misses)
            .enumerate()
            .filter(|(i, (slot, _))| {
                current[*i]
                    && slot
                        .as_ref()
                        .is_some_and(|(held_by, ..)| *held_by == Some(child_idx))
            })
            .map(|(i, (_, miss))| (i, (miss.path.as_str(), miss.hash.as_str())))
            .unzip();
        if still.is_empty() {
            continue;
        }
        for (i, vector) in still.into_iter().zip(held_vectors(child, &wanted).await) {
            current[i] = vector.is_some_and(|(held, _)| {
                found[i]
                    .as_ref()
                    .is_some_and(|(_, found_in, _)| *found_in == held)
            });
        }
    }
    if current.contains(&true) {
        let fill = ref_index.semantic_fill.lock().await;
        let mut cache = ref_index.embedding_cache.write().await;
        for (i, (slot, miss)) in found.into_iter().zip(misses).enumerate() {
            if let Some((.., entry)) = slot.filter(|_| {
                current[i]
                    && cache.get(&miss.path).map(|entry| &entry.hash) == miss.cached_hash.as_ref()
                    && fill.pending.get(&miss.path).map(|doc| &doc.hash)
                        == miss.pending_hash.as_ref()
            }) {
                vectors[i] = Some(entry.vector.clone());
                cache.insert(miss.path.clone(), entry);
            }
        }
    }
    if !children.is_empty() {
        tracing::info!(
            ref_id = %ref_index.cas_ref_id_hex,
            worktrees = children.len(),
            misses = misses.len(),
            adopted = vectors.iter().flatten().count(),
            "semantic worktree cache lookup"
        );
    }
    vectors
}

/// A warmup's fork of its parent's index: its own documents and their
/// vectors, the parent's paths it lacks and those it shares, and the
/// fingerprint of all its documents.
struct WarmupFork {
    changed: Vec<SearchDocument>,
    vectors: Vec<Option<Vec<f32>>>,
    deleted: Vec<String>,
    shared: std::collections::HashSet<String>,
    fingerprint: IndexFingerprint,
}

/// A worktree's semantic slot and generations when its walk began.
struct WalkStart {
    generation: u64,
    vector_generation: u64,
    seen: Option<std::sync::Weak<CachedSearchIndex>>,
}

impl WalkStart {
    async fn capture(ref_index: &crate::ref_index::RefIndex) -> Self {
        Self {
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
        }
    }
}

/// A file of a worktree's walk over its parent's forkable index.
enum Walked {
    /// Its content is the one the parent's index holds at this position.
    Shared(usize),
    /// Its content, or `None` until it is read.
    Read(Option<Arc<String>>),
    /// Not indexed: past the size limit or unreadable.
    Skipped,
}

/// A walk's files in walk order with their contents: `None` past the size
/// limit or unreadable.
type WalkedFiles = Vec<(String, Option<Arc<String>>)>;

/// Positions of `index`'s documents by path.
fn positions_by_path(index: &SearchIndex) -> HashMap<&str, usize> {
    index
        .documents()
        .iter()
        .enumerate()
        .map(|(at, doc)| (doc.path.as_str(), at))
        .collect()
}

/// Whether `index`'s document at `at` holds `content`, hashed `hash`, of the
/// file at `path`, with a vector.
fn holds(index: &SearchIndex, at: usize, path: &str, content: &str, hash: &str) -> bool {
    let doc = &index.documents()[at];
    doc.source_hash == hash
        && index.vector_at(at).is_some()
        && doc.content == semantic_embedding_content(path, content)
}

/// Each file of `files`, a worktree's contents layered over its parent's,
/// against the parent's forkable `base`: shared when those contents show it
/// unchanged and `base` indexed them with a vector, else its own content, or
/// `None` while that is unknown.
fn classify(
    files: &crate::server::ProjectCache,
    base: &SearchIndex,
    max_size: usize,
) -> Vec<(String, Walked)> {
    use rayon::prelude::*;
    let held = positions_by_path(base);
    let contents = &files.file_content;
    files
        .file_entries
        .par_iter()
        .filter(|entry| !entry.is_directory)
        .map(|entry| {
            let path = &entry.relative_path;
            let walked = match contents.own().get(path) {
                Some(content) if content.len() > max_size => Walked::Skipped,
                Some(content) => Walked::Read(Some(Arc::clone(content))),
                None if contents.shadows(path) => Walked::Read(None),
                None => match contents.get(path) {
                    Some(content) if content.len() > max_size => Walked::Skipped,
                    Some(content) => {
                        let hash = content_hash(content);
                        held.get(path.as_str())
                            .copied()
                            .filter(|&at| holds(base, at, path, content, &hash))
                            .map_or(Walked::Read(None), Walked::Shared)
                    }
                    None => Walked::Read(None),
                },
            };
            (path.clone(), walked)
        })
        .collect()
}

/// A document of a worktree's walk over its parent's forkable index.
enum ForkDocument {
    /// The parent's document at this position.
    Shared(usize),
    /// A document of the worktree's own content, with its content hash and
    /// embedding text.
    Own(Box<(SearchDocument, String, String)>),
}

/// A worktree's documents over its parent's forkable index.
struct Forked {
    /// In walk order.
    documents: Vec<ForkDocument>,
    /// The parent's documents the worktree does not have.
    deleted: Vec<String>,
}

impl Forked {
    /// The documents of the classified files `walked` over `base`; `None`
    /// when the worktree's own changes pass the promotion threshold.
    fn of(
        walked: &[(String, Walked)],
        base: &SearchIndex,
        doc_shape: crate::config::EmbedDocShape,
    ) -> Option<Self> {
        use rayon::prelude::*;
        let held = positions_by_path(base);
        let parent_documents: HashMap<&str, &SearchDocument> = base
            .documents()
            .iter()
            .map(|doc| (doc.path.as_str(), &**doc))
            .collect();
        let documents: Vec<_> = walked
            .par_iter()
            .filter_map(|(path, walked)| match walked {
                Walked::Shared(at) => Some(ForkDocument::Shared(*at)),
                Walked::Read(Some(content)) => {
                    let hash = content_hash(content);
                    if let Some(&at) = held
                        .get(path.as_str())
                        .filter(|&&at| holds(base, at, path, content, &hash))
                    {
                        return Some(ForkDocument::Shared(at));
                    }
                    let text = build_embedding_document(path, content, doc_shape);
                    let (mut doc, _) = walk_document(path, content, &hash, None, &parent_documents);
                    doc.source_hash = hash.clone();
                    Some(ForkDocument::Own(Box::new((doc, hash, text))))
                }
                Walked::Read(None) | Walked::Skipped => None,
            })
            .collect();
        let walked: std::collections::HashSet<&str> = documents
            .iter()
            .map(|document| match document {
                ForkDocument::Shared(at) => base.documents()[*at].path.as_str(),
                ForkDocument::Own(own) => own.0.path.as_str(),
            })
            .collect();
        let deleted: Vec<String> = base
            .documents()
            .iter()
            .filter(|doc| !walked.contains(doc.path.as_str()))
            .map(|doc| doc.path.clone())
            .collect();
        let changed = documents
            .iter()
            .filter(|document| matches!(document, ForkDocument::Own(_)))
            .count();
        ((changed + deleted.len()) as f64
            <= base.documents().len() as f64
                * crate::tools::semantic_search::FULL_REBUILD_CHANGE_FRACTION)
            .then_some(Self { documents, deleted })
    }
}

/// A worktree walk's documents identical to its parent's, which a fork of the
/// parent's index keeps with their vectors, and the parent's documents the
/// walk no longer has.
struct Shared {
    /// For each walked document, the position of the parent's identical one.
    positions: Vec<Option<usize>>,
    deleted: Vec<String>,
}

impl Shared {
    /// `None` when the walk's own changes pass the promotion threshold.
    fn of(base: &SearchIndex, docs: &[SearchDocument]) -> Option<Self> {
        let held: HashMap<&str, usize> = base
            .documents()
            .iter()
            .enumerate()
            .map(|(i, doc)| (doc.path.as_str(), i))
            .collect();
        let positions: Vec<Option<usize>> = docs
            .iter()
            .map(|doc| {
                held.get(doc.path.as_str()).copied().filter(|&i| {
                    let old = &base.documents()[i];
                    old.source_hash == doc.source_hash
                        && old.content == doc.content
                        && old.search_text == doc.search_text
                        && base.vector_at(i).is_some()
                })
            })
            .collect();
        let walked: std::collections::HashSet<&str> =
            docs.iter().map(|doc| doc.path.as_str()).collect();
        let deleted: Vec<String> = base
            .documents()
            .iter()
            .filter(|doc| !walked.contains(doc.path.as_str()))
            .map(|doc| doc.path.clone())
            .collect();
        let changed = positions.iter().filter(|at| at.is_none()).count();
        ((changed + deleted.len()) as f64
            <= base.documents().len() as f64
                * crate::tools::semantic_search::FULL_REBUILD_CHANGE_FRACTION)
            .then_some(Self { positions, deleted })
    }

    /// The walked `docs`, with their content hashes and embedding texts, over
    /// the parent's index.
    fn forked(
        self,
        docs: Vec<SearchDocument>,
        content_hashes: Vec<(String, String)>,
        embedding_texts: Vec<String>,
    ) -> Forked {
        let documents = docs
            .into_iter()
            .zip(content_hashes)
            .zip(embedding_texts)
            .zip(self.positions)
            .map(|(((doc, (_, hash)), text), at)| match at {
                Some(at) => ForkDocument::Shared(at),
                None => ForkDocument::Own(Box::new((doc, hash, text))),
            })
            .collect();
        Forked {
            documents,
            deleted: self.deleted,
        }
    }
}

/// The parent's semantic index when a worktree can fork it.
async fn forkable_base(parent: &crate::ref_index::RefIndex) -> Option<Arc<CachedSearchIndex>> {
    let base = parent.search_index_cache.read().await.clone();
    base.filter(|base| base.forkable_at(&parent.canonical_root))
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
        #[cfg(test)]
        test_seams::revalidated(root, &self.path);
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

/// A worktree starts, and restarts after an eviction, from the vectors it
/// persisted that its parent lacks.
async fn reload_own_vectors(
    ref_index: &crate::ref_index::RefIndex,
    parent_vectors: &FileVectors,
    config: &Config,
    indexed_only: bool,
) {
    let embedding_cache = &ref_index.embedding_cache;
    if !embedding_cache.read().await.is_empty() {
        return;
    }
    let started = std::time::Instant::now();
    let root = ref_index.root_dir.clone();
    let name = cache_name("embeddings", config);
    if let Ok(Ok(Some(store))) =
        tokio::task::spawn_blocking(move || rkyv_store::mmap_vector_store(&root, &name)).await
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
        if cache.is_empty() && !(indexed_only && evicted(ref_index)) {
            *cache = own;
        }
    }
    let entries = embedding_cache.read().await.len();
    tracing::info!(
        phase = "semantic_vector_reload",
        ref_id = %ref_index.cas_ref_id_hex,
        entries,
        elapsed_ms = started.elapsed().as_millis(),
        "cold-start phase"
    );
}

/// Whether an eviction dropped `ref_index`'s semantic entry. It does so
/// before clearing the vectors, so a caller holding the vector lock that sees
/// the entry, or its eviction mid-write, is cleared after.
fn evicted(ref_index: &crate::ref_index::RefIndex) -> bool {
    ref_index
        .search_index_cache
        .try_read()
        .is_ok_and(|entry| entry.is_none())
}

/// Copies into each worktree attached to `ref_index` the vectors its semantic
/// entry holds at content `ref_index` has since replaced or `deleted`, and
/// persists them and any vector it copied from `ref_index` that `ref_index`
/// no longer holds, so a restart keeps them. Vectors `ref_index` still holds
/// stay shared.
pub(crate) async fn keep_moved_off_vectors(
    state: &SharedState,
    ref_index: &crate::ref_index::RefIndex,
    config: &Config,
    deleted: &[String],
) {
    let deleted: std::collections::HashSet<&str> = deleted.iter().map(String::as_str).collect();
    for child in state.attached_children(ref_index).await {
        let entry = child.search_index_cache.read().await.clone();
        let Some(entry) = entry.filter(|entry| entry.search_root() == child.canonical_root) else {
            continue;
        };
        reload_own_vectors(&child, &ref_index.embedding_cache, config, true).await;
        let (moved, own) = {
            let own = child.embedding_cache.read().await;
            let parent = ref_index.embedding_cache.read().await;
            let moved: Vec<(String, Option<String>, CacheEntry)> = entry
                .index
                .documents()
                .iter()
                .enumerate()
                .filter(|(_, doc)| {
                    own.get(&doc.path)
                        .is_none_or(|held| held.hash != doc.source_hash)
                        && match parent.get(&doc.path) {
                            Some(held) => held.hash != doc.source_hash,
                            None => deleted.contains(doc.path.as_str()),
                        }
                })
                .filter_map(|(at, doc)| {
                    let vector = entry.index.vector_at(at)?.to_vec();
                    let hash = doc.source_hash.clone();
                    let seen = own.get(&doc.path).map(|held| held.hash.clone());
                    Some((doc.path.clone(), seen, CacheEntry { hash, vector }))
                })
                .collect();
            let own: Vec<(String, String)> = own
                .iter()
                .filter(|(path, held)| !inherits(&parent, path, held))
                .map(|(path, held)| (path.clone(), held.hash.clone()))
                .collect();
            (moved, own)
        };
        let mut dirty = false;
        if !moved.is_empty() {
            let mut cache = child.embedding_cache.write().await;
            if evicted(&child) {
                continue;
            }
            for (path, seen, entry) in moved {
                // A stale own vector gives way; a newer one the fill wrote since stays.
                if cache.get(&path).map(|held| &held.hash) == seen.as_ref() {
                    cache.insert(path, entry);
                    dirty = true;
                }
            }
        }
        // Vectors the parent held at the worktree's last save are not on its disk.
        if !dirty && !own.is_empty() {
            let root = child.root_dir.clone();
            let name = cache_name("embeddings", config);
            let saved =
                tokio::task::spawn_blocking(move || rkyv_store::mmap_vector_store(&root, &name))
                    .await;
            dirty = match saved {
                Ok(Ok(Some(store))) => own
                    .iter()
                    .any(|(path, hash)| store.get_hash(path) != Some(hash.as_str())),
                _ => true,
            };
        }
        if dirty {
            save_vectors(&child, config, Some(&ref_index.embedding_cache)).await;
        }
    }
}

/// Keeps the vectors attached worktrees share that `ref_index` moved off,
/// then persists `ref_index`'s.
async fn persist_fill(
    state: &SharedState,
    ref_index: &crate::ref_index::RefIndex,
    config: &Config,
    parent_vectors: Option<&FileVectors>,
) {
    keep_moved_off_vectors(state, ref_index, config, &[]).await;
    save_vectors(ref_index, config, parent_vectors).await;
}

/// Persists the vectors of `ref_index`; a worktree's only those its parent
/// lacks, dropping the rest from disk.
async fn save_vectors(
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
    state: Arc<SharedState>,
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
            persist_fill(&state, &ref_index, &config, parent_vectors.as_deref()).await;
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
                persist_fill(&state, &ref_index, &config, parent_vectors.as_deref()).await;
                completed = 0;
            }
        }
    }
}
