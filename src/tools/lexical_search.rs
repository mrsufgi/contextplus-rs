// TODO: For very large corpora (>~10k files) callers may swap this in-process
// index for a tantivy-backed index in a follow-up. The public API of
// `LexicalIndex` (build / search / document_count) and `rrf_merge` are
// designed to be drop-in replaceable.

//! In-process lexical inverted index with BM25F scoring and Reciprocal Rank
//! Fusion (RRF) merge.
//!
//! No external dependencies — pure Rust, no I/O, no async.

use std::collections::HashMap;

use crate::tools::semantic_search::{SearchDocument, split_camel_case};

// ---------------------------------------------------------------------------
// LexicalIndex
// ---------------------------------------------------------------------------

// Tunable BM25F parameters, ordered as path, symbols, header, body.
const K1: f64 = 1.2;
const FIELD_WEIGHTS: [f64; 4] = [3.0, 3.0, 1.5, 1.0];
const FIELD_B: [f64; 4] = [0.3, 0.3, 0.5, 0.75];
const TEST_PRIOR: f64 = 0.6;
const GENERATED_PRIOR: f64 = 0.4;
const PROSE_PRIOR: f64 = 0.6;
const LOCK_PRIOR: f64 = 0.2;
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "are", "how", "what",
    "where", "when", "why", "which", "does", "do", "with", "by", "from", "that", "this", "it",
    "be",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PathPriorClassification {
    pub(crate) is_test_like: bool,
    pub(crate) is_generated: bool,
    pub(crate) is_planning_prose: bool,
    pub(crate) is_lockfile: bool,
    pub(crate) is_documentation: bool,
}

pub(crate) fn classify_path_prior(path: &str) -> PathPriorClassification {
    // A leading slash also recognizes directory priors at the repository root.
    let path = format!("/{}", path.replace('\\', "/").to_lowercase());
    PathPriorClassification {
        is_test_like: [
            ".test.",
            ".spec.",
            "/test/",
            "/tests/",
            "__tests__/",
            "/__fixtures__/",
            "/fixtures/",
            "/__mocks__/",
            "/testdata/",
            "/test-data/",
            ".fixture.",
        ]
        .iter()
        .any(|pattern| path.contains(pattern))
            || path.ends_with(".snap"),
        is_generated: ["/generated/", "/gen/", ".generated.", "_pb.", ".pb."]
            .iter()
            .any(|pattern| path.contains(pattern)),
        is_planning_prose: [
            "/agent-os/specs/",
            "/docs/plans/",
            "/docs/brainstorms/",
            "/todos/",
        ]
        .iter()
        .any(|pattern| path.contains(pattern)),
        is_documentation: path.ends_with(".md")
            || path.ends_with(".mdx")
            || path.contains("/docs/")
            || path.contains("/agent-os/")
            || (path.contains("/migrations/") && path.ends_with(".sql")),
        is_lockfile: path.ends_with(".lock") || path.ends_with("/package-lock.json"),
    }
}

impl PathPriorClassification {
    fn non_test_multiplier(self) -> f64 {
        let mut prior = 1.0;
        if self.is_generated {
            prior *= GENERATED_PRIOR;
        }
        if self.is_planning_prose {
            prior *= PROSE_PRIOR;
        }
        if self.is_lockfile {
            prior *= LOCK_PRIOR;
        }
        prior
    }

    pub(crate) fn meaning_multiplier(self, wants_tests: bool) -> f64 {
        let test_prior = if self.is_test_like && !wants_tests {
            TEST_PRIOR
        } else {
            1.0
        };
        // Square-root strength keeps semantic priors milder than lexical priors,
        // allowing strong test matches to surface while breaking near ties for source.
        (self.non_test_multiplier() * test_prior).sqrt()
            * if self.is_documentation { 0.8 } else { 1.0 }
    }
}

pub(crate) fn is_test_intent_token(token: &str) -> bool {
    matches!(token, "test" | "spec" | "fixture")
}

type Posting = (u32, [u32; 4]);

/// Documents containing one term with its per-field counts, sorted by
/// document. Most terms of a large corpus occur in one document only, so a
/// single posting is stored inline instead of in its own allocation.
#[derive(Clone)]
enum PostingList {
    One(Posting),
    Many(Vec<Posting>),
}

impl Default for PostingList {
    fn default() -> Self {
        Self::Many(Vec::new())
    }
}

impl PostingList {
    fn as_slice(&self) -> &[Posting] {
        match self {
            Self::One(posting) => std::slice::from_ref(posting),
            Self::Many(postings) => postings,
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn contains(&self, doc: usize) -> bool {
        self.as_slice()
            .binary_search_by_key(&doc, |posting| posting.0 as usize)
            .is_ok()
    }

    fn insert(&mut self, doc: u32, counts: [u32; 4]) {
        match self {
            Self::Many(postings) if postings.is_empty() => *self = Self::One((doc, counts)),
            Self::One(posting) if posting.0 == doc => posting.1 = counts,
            Self::One(posting) => {
                let mut postings = vec![*posting, (doc, counts)];
                postings.sort_unstable_by_key(|posting| posting.0);
                *self = Self::Many(postings);
            }
            Self::Many(postings) => {
                match postings.binary_search_by_key(&doc, |posting| posting.0) {
                    Ok(i) => postings[i].1 = counts,
                    Err(i) => postings.insert(i, (doc, counts)),
                }
            }
        }
    }

    fn remove(&mut self, doc: u32) {
        match self {
            Self::One(posting) if posting.0 == doc => *self = Self::default(),
            Self::One(_) => {}
            Self::Many(postings) => {
                if let Ok(i) = postings.binary_search_by_key(&doc, |posting| posting.0) {
                    postings.remove(i);
                }
                if let [only] = postings.as_slice() {
                    *self = Self::One(*only);
                }
            }
        }
    }

    fn shrink_to_fit(&mut self) {
        if let Self::Many(postings) = self {
            postings.shrink_to_fit();
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::One(_) => 0,
            Self::Many(postings) => postings.capacity() * std::mem::size_of::<Posting>(),
        }
    }
}

/// Every term of the index, stored once: the texts back to back in one
/// buffer and a hash table of term ids keyed by text.
#[derive(Clone, Default)]
struct TermDictionary {
    text: String,
    ends: Vec<u32>,
    ids: hashbrown::HashTable<u32>,
    hasher: std::hash::RandomState,
}

fn term_at<'a>(text: &'a str, ends: &[u32], id: u32) -> &'a str {
    let id = id as usize;
    let start = if id == 0 { 0 } else { ends[id - 1] as usize };
    &text[start..ends[id] as usize]
}

impl TermDictionary {
    fn get(&self, term: &str) -> Option<u32> {
        let hash = std::hash::BuildHasher::hash_one(&self.hasher, term);
        self.ids
            .find(hash, |&id| term_at(&self.text, &self.ends, id) == term)
            .copied()
    }

    fn intern(&mut self, term: &str) -> u32 {
        if let Some(id) = self.get(term) {
            return id;
        }
        let Self {
            text,
            ends,
            ids,
            hasher,
        } = self;
        let id = u32::try_from(ends.len()).expect("lexical index holds at most u32::MAX terms");
        text.push_str(term);
        ends.push(u32::try_from(text.len()).expect("lexical term text exceeds 4 GiB"));
        let hash = std::hash::BuildHasher::hash_one(&*hasher, term);
        ids.insert_unique(hash, id, |&other| {
            std::hash::BuildHasher::hash_one(&*hasher, term_at(text, ends, other))
        });
        id
    }

    #[cfg(feature = "memory-profile")]
    fn len(&self) -> usize {
        self.ends.len()
    }

    fn shrink_to_fit(&mut self) {
        let Self {
            text,
            ends,
            ids,
            hasher,
        } = self;
        ids.shrink_to_fit(|&id| {
            std::hash::BuildHasher::hash_one(&*hasher, term_at(text, ends, id))
        });
        text.shrink_to_fit();
        ends.shrink_to_fit();
    }

    fn heap_bytes(&self) -> usize {
        // A table bucket is one control byte and a u32, at 7/8 load at most.
        self.text.capacity()
            + self.ends.capacity() * std::mem::size_of::<u32>()
            + self.ids.capacity() * 8 / 7 * (1 + std::mem::size_of::<u32>())
    }
}

/// In-process inverted index over a [`SearchDocument`] slice.
#[derive(Clone)]
pub struct LexicalIndex {
    terms: TermDictionary,
    /// Indexed by term id.
    posting: Vec<PostingList>,
    documents: Vec<DocumentFields>,
    average_lengths: [f64; 4],
    total_lengths: [f64; 4],
    /// Term ids of each document.
    document_terms: Vec<Box<[u32]>>,
    doc_count: usize,
    #[cfg(test)]
    last_update_work: UpdateWork,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UpdateWork {
    pub(crate) postings_visited: usize,
    pub(crate) documents_visited: usize,
}

#[cfg(test)]
pub(crate) mod test_seams {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};

    fn slot() -> &'static Mutex<Option<Arc<Pause>>> {
        static SLOT: OnceLock<Mutex<Option<Arc<Pause>>>> = OnceLock::new();
        SLOT.get_or_init(|| Mutex::new(None))
    }

    pub(crate) struct Pause {
        entered: AtomicBool,
        released: AtomicBool,
    }

    impl Pause {
        pub(crate) async fn wait_until_entered(&self) {
            while !self.entered.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        }

        pub(crate) fn release(&self) {
            self.released.store(true, Ordering::Release);
        }
    }

    impl Drop for Pause {
        fn drop(&mut self) {
            self.release();
        }
    }

    pub(crate) fn pause_next_update() -> Arc<Pause> {
        let pause = Arc::new(Pause {
            entered: AtomicBool::new(false),
            released: AtomicBool::new(false),
        });
        *slot().lock().unwrap() = Some(Arc::clone(&pause));
        pause
    }

    pub(crate) fn before_update() {
        let pause = slot().lock().unwrap().take();
        if let Some(pause) = pause {
            pause.entered.store(true, Ordering::Release);
            while !pause.released.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        }
    }
}

#[derive(Clone)]
struct DocumentFields {
    lengths: [u32; 4],
    prior: f64,
    is_test: bool,
}

/// Lowercased tokens of `text` with their occurrence counts: the camelCase /
/// snake_case parts every scorer uses, plus each whole identifier, so that
/// `resolveScopeMode` is a token of its own and not only `resolve`, `scope`
/// and `mode`, which almost every file contains.
fn token_counts(text: &str) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    for part in split_camel_case(text) {
        *counts.entry(part).or_insert(0) += 1;
    }
    for word in text
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| w.len() > 1)
    {
        let whole = word.to_lowercase();
        let parts = split_camel_case(word);
        if parts.len() != 1 || parts[0] != whole {
            *counts.entry(whole).or_insert(0) += 1;
        }
    }
    counts
}

/// The fields of one document the index reads.
pub(crate) struct LexicalFields<'a> {
    pub(crate) path: &'a str,
    pub(crate) symbols: &'a [String],
    pub(crate) header: &'a str,
    pub(crate) content: &'a str,
}

impl<'a> From<&'a SearchDocument> for LexicalFields<'a> {
    fn from(doc: &'a SearchDocument) -> Self {
        Self {
            path: &doc.path,
            symbols: &doc.symbols,
            header: &doc.header,
            content: &doc.content,
        }
    }
}

/// Per-field term counts of one document and its per-field lengths.
pub(crate) type DocumentTermCounts = (HashMap<String, [u32; 4]>, [u32; 4]);

/// Per-field term counts of one document and its per-field lengths.
pub(crate) fn document_term_counts(doc: &LexicalFields<'_>) -> DocumentTermCounts {
    let symbols = doc.symbols.join(" ");
    let mut terms: HashMap<String, [u32; 4]> = HashMap::new();
    let mut lengths = [0_u32; 4];
    for (field, text) in [doc.path, &symbols, doc.header, doc.content]
        .into_iter()
        .enumerate()
    {
        for (token, count) in token_counts(text) {
            lengths[field] = lengths[field].saturating_add(count);
            terms.entry(token).or_default()[field] = count;
        }
    }
    (terms, lengths)
}

/// Term counts of the document of `path`.
pub(crate) fn lexical_term_counts(
    path: &str,
    files: &crate::core::walker::FileContents,
) -> DocumentTermCounts {
    let content = files.get(path).map_or("", |content| content.as_str());
    let ext = path.rsplit('.').next().unwrap_or("");
    let symbols: Vec<String> = crate::core::tree_sitter::parse_with_tree_sitter(content, ext)
        .unwrap_or_default()
        .into_iter()
        .map(|symbol| symbol.name)
        .collect();
    let header = crate::core::parser::extract_header(content);
    document_term_counts(&LexicalFields {
        path,
        symbols: &symbols,
        header: &header,
        content,
    })
}

/// Keyword-index updates for `changed` files, each at its document's slot or
/// at a slot appended to `document_paths` for a new file.
pub(crate) fn lexical_updates<'a>(
    document_paths: &mut Vec<String>,
    changed: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Vec<(usize, SearchDocument)> {
    let mut slots: HashMap<&str, usize> = HashMap::new();
    let changed: Vec<_> = changed.into_iter().collect();
    for (i, path) in document_paths.iter().enumerate() {
        slots.entry(path.as_str()).or_insert(i);
    }
    let mut placed: Vec<Option<usize>> = changed
        .iter()
        .map(|(path, _)| slots.get(path).copied())
        .collect();
    drop(slots);
    for (slot, (path, _)) in placed.iter_mut().zip(&changed) {
        if slot.is_none() {
            document_paths.push((*path).to_owned());
            *slot = Some(document_paths.len() - 1);
        }
    }
    changed
        .into_iter()
        .zip(placed)
        .map(|((path, content), slot)| {
            let ext = path.rsplit('.').next().unwrap_or("");
            let symbols = crate::core::tree_sitter::parse_with_tree_sitter(content, ext)
                .unwrap_or_default()
                .into_iter()
                .map(|s| s.name)
                .collect();
            (
                slot.expect("every changed file has a slot"),
                SearchDocument::new(
                    path.to_owned(),
                    crate::core::parser::extract_header(content),
                    symbols,
                    vec![],
                    content.to_owned(),
                ),
            )
        })
        .collect()
}

fn document_fields(path: &str, lengths: [u32; 4]) -> DocumentFields {
    let classification = classify_path_prior(path);
    DocumentFields {
        lengths,
        prior: classification.non_test_multiplier(),
        is_test: classification.is_test_like,
    }
}

impl LexicalIndex {
    /// Document slots, live or deleted.
    pub(crate) fn slot_count(&self) -> usize {
        self.documents.len()
    }

    /// Terms in the dictionary, used or not.
    pub(crate) fn term_count(&self) -> usize {
        self.posting.len()
    }

    /// Terms no document contains any more.
    pub(crate) fn dead_term_count(&self) -> usize {
        self.posting.iter().filter(|list| list.len() == 0).count()
    }

    #[cfg(test)]
    fn posting_count(&self) -> usize {
        self.posting.iter().map(PostingList::len).sum()
    }

    #[cfg(feature = "memory-profile")]
    pub(crate) fn profile_stats(&self) -> String {
        let postings: usize = self.posting.iter().map(PostingList::len).sum();
        let singletons = self.posting.iter().filter(|list| list.len() == 1).count();
        let document_terms: usize = self.document_terms.iter().map(|terms| terms.len()).sum();
        format!(
            "terms={} singleton_terms={singletons} postings={postings} documents={} document_terms={document_terms} estimated_bytes={}",
            self.terms.len(),
            self.documents.len(),
            self.estimated_resident_bytes(),
        )
    }

    pub(crate) fn estimated_resident_bytes(&self) -> usize {
        self.terms.heap_bytes()
            + self.posting.capacity() * std::mem::size_of::<PostingList>()
            + self
                .posting
                .iter()
                .map(PostingList::heap_bytes)
                .sum::<usize>()
            + self.documents.capacity() * std::mem::size_of::<DocumentFields>()
            + self.document_terms.capacity() * std::mem::size_of::<Box<[u32]>>()
            + self
                .document_terms
                .iter()
                .map(|terms| terms.len() * std::mem::size_of::<u32>())
                .sum::<usize>()
    }

    fn postings(&self, term: &str) -> Option<&PostingList> {
        self.terms.get(term).map(|id| &self.posting[id as usize])
    }

    /// Records `terms` as the terms of document `doc` and returns their ids.
    fn add_postings(&mut self, doc: usize, terms: HashMap<String, [u32; 4]>) -> Box<[u32]> {
        let doc = u32::try_from(doc).expect("lexical index holds at most u32::MAX documents");
        terms
            .into_iter()
            .map(|(term, counts)| {
                let id = self.terms.intern(&term);
                if id as usize == self.posting.len() {
                    self.posting.push(PostingList::default());
                }
                self.posting[id as usize].insert(doc, counts);
                id
            })
            .collect()
    }

    /// Build an index from a slice of [`SearchDocument`]s.
    ///
    /// Records separate path, definition-name, header and body frequencies.
    pub fn build(docs: &[SearchDocument]) -> Self {
        let mut index = Self::with_capacity(docs.len());
        for doc in docs {
            index.push_document(doc.into());
        }
        index.finish_build();
        index
    }

    /// An empty index to [`push_document`](Self::push_document) into, then
    /// [`finish_build`](Self::finish_build).
    pub(crate) fn with_capacity(documents: usize) -> Self {
        Self {
            terms: TermDictionary::default(),
            posting: Vec::new(),
            documents: Vec::with_capacity(documents),
            average_lengths: [1.0; 4],
            total_lengths: [0.0; 4],
            document_terms: Vec::with_capacity(documents),
            doc_count: 0,
            #[cfg(test)]
            last_update_work: UpdateWork::default(),
        }
    }

    /// Adds the next document of a build.
    pub(crate) fn push_document(&mut self, doc: LexicalFields<'_>) {
        self.push_counted(doc.path, document_term_counts(&doc));
    }

    /// Adds the next document of a build from its
    /// [`document_term_counts`], which may be computed on another thread.
    pub(crate) fn push_counted(&mut self, path: &str, counts: DocumentTermCounts) {
        let (terms, lengths) = counts;
        for (total, length) in self.total_lengths.iter_mut().zip(lengths) {
            *total += f64::from(length);
        }
        let ids = self.add_postings(self.documents.len(), terms);
        self.document_terms.push(ids);
        self.documents.push(document_fields(path, lengths));
        self.doc_count += 1;
    }

    /// Releases build slack and sets the average field lengths.
    pub(crate) fn finish_build(&mut self) {
        self.terms.shrink_to_fit();
        self.posting.shrink_to_fit();
        self.posting.iter_mut().for_each(PostingList::shrink_to_fit);
        self.documents.shrink_to_fit();
        self.document_terms.shrink_to_fit();
        self.average_lengths = if self.doc_count == 0 {
            [1.0; 4]
        } else {
            self.total_lengths
                .map(|total| (total / self.doc_count as f64).max(f64::EPSILON))
        };
    }

    pub(crate) fn update_documents(
        &mut self,
        updates: Vec<(usize, SearchDocument)>,
        deleted: &[usize],
    ) {
        #[cfg(test)]
        {
            self.last_update_work = UpdateWork::default();
            test_seams::before_update();
        }
        let affected: std::collections::HashSet<usize> = deleted
            .iter()
            .copied()
            .chain(updates.iter().map(|(i, _)| *i))
            .collect();
        #[cfg(test)]
        let mut postings_visited = 0;
        for &i in &affected {
            if i < self.documents.len() {
                #[cfg(test)]
                {
                    self.last_update_work.documents_visited += 1;
                }
                for &id in std::mem::take(&mut self.document_terms[i]).iter() {
                    #[cfg(test)]
                    {
                        postings_visited += 1;
                    }
                    self.posting[id as usize].remove(i as u32);
                }
                for (total, length) in self.total_lengths.iter_mut().zip(self.documents[i].lengths)
                {
                    *total -= f64::from(length);
                }
                self.documents[i].lengths = [0; 4];
            }
        }
        self.doc_count -= deleted.len();
        for (i, doc) in updates {
            let (terms, lengths) = document_term_counts(&(&doc).into());
            for (total, length) in self.total_lengths.iter_mut().zip(lengths) {
                *total += f64::from(length);
            }
            let ids = self.add_postings(i, terms);
            let fields = document_fields(&doc.path, lengths);
            if i >= self.documents.len() {
                self.document_terms.push(ids);
                self.doc_count += 1;
                self.documents.push(fields);
            } else {
                self.document_terms[i] = ids;
                self.documents[i] = fields;
            }
        }
        self.average_lengths = self.total_lengths;
        for total in &mut self.average_lengths {
            *total = (*total / self.doc_count.max(1) as f64).max(f64::EPSILON);
        }
        #[cfg(test)]
        {
            self.last_update_work.postings_visited = postings_visited;
        }
    }

    /// A copy that holds only the slots marked in `live`, renumbered in
    /// order, and only the terms those slots contain. It ranks every query
    /// as this index does.
    pub(crate) fn compacted(&self, live: &[bool]) -> Self {
        let mut slots = vec![u32::MAX; self.documents.len()];
        let kept = slots
            .iter_mut()
            .zip(live)
            .filter(|(_, live)| **live)
            .map(|(slot, _)| slot);
        for (next, slot) in (0_u32..).zip(kept) {
            *slot = next;
        }
        let mut terms = TermDictionary::default();
        let mut term_ids = vec![u32::MAX; self.posting.len()];
        let mut posting = Vec::new();
        for (id, list) in self.posting.iter().enumerate() {
            let postings: Vec<Posting> = list
                .as_slice()
                .iter()
                .filter_map(|(doc, counts)| {
                    let slot = slots[*doc as usize];
                    (slot != u32::MAX).then_some((slot, *counts))
                })
                .collect();
            if postings.is_empty() {
                continue;
            }
            term_ids[id] = terms.intern(term_at(&self.terms.text, &self.terms.ends, id as u32));
            posting.push(match postings.as_slice() {
                [only] => PostingList::One(*only),
                _ => PostingList::Many(postings),
            });
        }
        let (documents, document_terms): (Vec<_>, Vec<_>) = self
            .documents
            .iter()
            .zip(&self.document_terms)
            .zip(live)
            .filter(|(_, live)| **live)
            .map(|((fields, ids), _)| {
                let ids: Box<[u32]> = ids
                    .iter()
                    .map(|&id| term_ids[id as usize])
                    .filter(|&id| id != u32::MAX)
                    .collect();
                (fields.clone(), ids)
            })
            .unzip();
        let mut index = Self {
            terms,
            posting,
            documents,
            average_lengths: self.average_lengths,
            total_lengths: self.total_lengths,
            document_terms,
            doc_count: self.doc_count,
            #[cfg(test)]
            last_update_work: UpdateWork::default(),
        };
        index.terms.shrink_to_fit();
        index.posting.shrink_to_fit();
        index
    }

    /// Writes the whole index, read back by [`read_snapshot`](Self::read_snapshot).
    pub(crate) fn write_snapshot(
        &self,
        out: &mut crate::cache::snapshot::SnapshotWriter,
    ) -> std::io::Result<()> {
        out.str(&self.terms.text)?;
        out.u32s(&self.terms.ends)?;
        out.usize(self.posting.len())?;
        let posting_values = self
            .posting
            .iter()
            .map(|list| 1 + 5 * list.len())
            .sum::<usize>();
        out.u32_seq(
            posting_values,
            self.posting.iter().flat_map(|list| {
                let postings = list.as_slice();
                std::iter::once(postings.len() as u32).chain(
                    postings
                        .iter()
                        .flat_map(|(doc, counts)| std::iter::once(*doc).chain(*counts)),
                )
            }),
        )?;
        out.usize(self.documents.len())?;
        for fields in &self.documents {
            for length in fields.lengths {
                out.u32(length)?;
            }
            out.f64(fields.prior)?;
            out.bool(fields.is_test)?;
        }
        for value in self.average_lengths.iter().chain(&self.total_lengths) {
            out.f64(*value)?;
        }
        let term_values = self
            .document_terms
            .iter()
            .map(|terms| 1 + terms.len())
            .sum::<usize>();
        out.u32_seq(
            term_values,
            self.document_terms
                .iter()
                .flat_map(|terms| std::iter::once(terms.len() as u32).chain(terms.iter().copied())),
        )?;
        out.usize(self.doc_count)
    }

    /// An index written by [`write_snapshot`](Self::write_snapshot), or `None`
    /// when the payload is inconsistent.
    pub(crate) fn read_snapshot(
        input: &mut crate::cache::snapshot::SnapshotReader<'_>,
    ) -> Option<Self> {
        let text = input.str()?.to_owned();
        let ends = input.u32s()?;
        let mut start = 0;
        for &end in &ends {
            let end = end as usize;
            if end < start || !text.is_char_boundary(end) {
                return None;
            }
            start = end;
        }
        if start != text.len() {
            return None;
        }
        let terms_len = ends.len();
        let mut terms = TermDictionary {
            text,
            ends,
            ids: hashbrown::HashTable::with_capacity(terms_len),
            hasher: std::hash::RandomState::new(),
        };
        {
            let TermDictionary {
                text,
                ends,
                ids,
                hasher,
            } = &mut terms;
            for id in 0..terms_len as u32 {
                let hash = std::hash::BuildHasher::hash_one(&*hasher, term_at(text, ends, id));
                ids.insert_unique(hash, id, |&other| {
                    std::hash::BuildHasher::hash_one(&*hasher, term_at(text, ends, other))
                });
            }
        }

        let posting_len = input.usize()?;
        if posting_len != terms_len {
            return None;
        }
        let mut values = input.u32_seq()?;
        let mut posting = Vec::with_capacity(posting_len);
        let mut max_doc = None::<u32>;
        for _ in 0..posting_len {
            let len = values.next_value()? as usize;
            if len > values.len() / 5 {
                return None;
            }
            let mut read = || -> Option<Posting> {
                let doc = values.next_value()?;
                max_doc = max_doc.max(Some(doc));
                Some((
                    doc,
                    [
                        values.next_value()?,
                        values.next_value()?,
                        values.next_value()?,
                        values.next_value()?,
                    ],
                ))
            };
            posting.push(match len {
                0 => PostingList::default(),
                1 => PostingList::One(read()?),
                _ => {
                    let mut postings = Vec::with_capacity(len);
                    for _ in 0..len {
                        postings.push(read()?);
                    }
                    PostingList::Many(postings)
                }
            });
        }
        if !values.is_empty() {
            return None;
        }

        let documents_len = input.usize()?;
        let mut documents = Vec::with_capacity(documents_len);
        for _ in 0..documents_len {
            let lengths = [input.u32()?, input.u32()?, input.u32()?, input.u32()?];
            documents.push(DocumentFields {
                lengths,
                prior: input.f64()?,
                is_test: input.bool()?,
            });
        }
        if max_doc.is_some_and(|doc| doc as usize >= documents_len) {
            return None;
        }
        let mut lengths = [0.0; 8];
        for value in &mut lengths {
            *value = input.f64()?;
        }
        let mut values = input.u32_seq()?;
        let mut document_terms = Vec::with_capacity(documents_len);
        for _ in 0..documents_len {
            let len = values.next_value()? as usize;
            if len > values.len() {
                return None;
            }
            let mut ids = Vec::with_capacity(len);
            for _ in 0..len {
                ids.push(
                    values
                        .next_value()
                        .filter(|&id| (id as usize) < terms_len)?,
                );
            }
            document_terms.push(ids.into_boxed_slice());
        }
        if !values.is_empty() {
            return None;
        }
        let doc_count = input.usize_value()?;
        if doc_count > documents_len {
            return None;
        }
        Some(Self {
            terms,
            posting,
            documents,
            average_lengths: lengths[..4].try_into().unwrap(),
            total_lengths: lengths[4..].try_into().unwrap(),
            document_terms,
            doc_count,
            #[cfg(test)]
            last_update_work: UpdateWork::default(),
        })
    }

    #[cfg(test)]
    pub(crate) fn last_update_work(&self) -> UpdateWork {
        self.last_update_work
    }

    /// Number of documents in the index.
    pub fn document_count(&self) -> usize {
        self.doc_count
    }

    /// Search the index for `query`, returning up to `top_k` `(doc_idx,
    /// score)` pairs sorted by score descending (ties broken by smaller
    /// `doc_idx` first).
    ///
    /// BM25F term scores, multiplied by squared query coverage and path priors.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(usize, f64)> {
        if top_k == 0 || self.doc_count == 0 {
            return Vec::new();
        }
        let tokens = query_tokens(query);
        let df: Vec<f64> = tokens
            .iter()
            .map(|token| self.postings(token).map_or(0.0, |p| p.len() as f64))
            .collect();
        let mut ranked = self.scored(
            &tokens,
            &df,
            self.doc_count as f64,
            &self.average_lengths,
            |_| false,
        );
        // Sort descending by score; ties → ascending by doc_idx.
        ranked.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(top_k);
        ranked
    }

    /// BM25F score of every document matching `tokens`, given the corpus
    /// statistics to score against. `df` is aligned with `tokens`.
    fn scored(
        &self,
        tokens: &[String],
        df: &[f64],
        doc_count: f64,
        average_lengths: &[f64; 4],
        skip: impl Fn(usize) -> bool,
    ) -> Vec<(usize, f64)> {
        let wants_tests = tokens.iter().any(|token| is_test_intent_token(token));
        let mut scores: HashMap<usize, (f64, usize)> = HashMap::new();
        for (token, &df) in tokens.iter().zip(df) {
            if let Some(postings) = self.postings(token) {
                let idf = (1.0 + (doc_count - df + 0.5) / (df + 0.5)).ln();
                for (doc_idx, counts) in postings.as_slice() {
                    let doc_idx = *doc_idx as usize;
                    if skip(doc_idx) {
                        continue;
                    }
                    let doc = &self.documents[doc_idx];
                    let tf: f64 = (0..4)
                        .map(|field| {
                            FIELD_WEIGHTS[field] * f64::from(counts[field])
                                / (1.0 - FIELD_B[field]
                                    + FIELD_B[field] * f64::from(doc.lengths[field])
                                        / average_lengths[field])
                        })
                        .sum();
                    let score = scores.entry(doc_idx).or_default();
                    score.0 += idf * tf / (K1 + tf);
                    score.1 += 1;
                }
            }
        }
        scores
            .into_iter()
            .map(|(idx, (score, matched))| {
                let doc = &self.documents[idx];
                let coverage = matched as f64 / tokens.len() as f64;
                let test_prior = if doc.is_test && !wants_tests {
                    TEST_PRIOR
                } else {
                    1.0
                };
                (idx, score * coverage.powi(2) * doc.prior * test_prior)
            })
            .collect()
    }

    /// Searches this index as the delta over `base` without its `mask`ed
    /// documents, scoring with the statistics of that combined corpus: its
    /// document count, average field lengths and per-term document frequency.
    /// Hits are unsorted `(in_delta, doc_idx, score)`.
    pub(crate) fn search_over(
        &self,
        base: &LexicalIndex,
        mask: &BaseMask,
        query: &str,
    ) -> Vec<(bool, usize, f64)> {
        let doc_count = base.doc_count + self.doc_count - mask.masked.len();
        if doc_count == 0 {
            return Vec::new();
        }
        let mut average_lengths = [0.0; 4];
        for (field, average) in average_lengths.iter_mut().enumerate() {
            let total = base.total_lengths[field] - mask.lengths[field] + self.total_lengths[field];
            *average = (total / doc_count as f64).max(f64::EPSILON);
        }
        let tokens = query_tokens(query);
        let df: Vec<f64> = tokens
            .iter()
            .map(|token| {
                let in_base = base.postings(token).map_or(0, |postings| {
                    postings.len()
                        - mask
                            .masked
                            .iter()
                            .filter(|&&doc| postings.contains(doc))
                            .count()
                });
                let in_delta = self.postings(token).map_or(0, |p| p.len());
                (in_base + in_delta) as f64
            })
            .collect();
        let doc_count = doc_count as f64;
        let mut hits: Vec<(bool, usize, f64)> = base
            .scored(&tokens, &df, doc_count, &average_lengths, |doc| {
                mask.masked.contains(&doc)
            })
            .into_iter()
            .map(|(doc, score)| (false, doc, score))
            .collect();
        hits.extend(
            self.scored(&tokens, &df, doc_count, &average_lengths, |_| false)
                .into_iter()
                .map(|(doc, score)| (true, doc, score)),
        );
        hits
    }
}

fn query_tokens(query: &str) -> Vec<String> {
    let mut tokens: Vec<String> = token_counts(query)
        .into_keys()
        .filter(|token| !STOPWORDS.contains(&token.as_str()))
        .collect();
    tokens.sort_unstable();
    tokens
}

/// Live documents of a base index hidden from a delta built over it, with
/// their summed field lengths.
#[derive(Clone, Default)]
pub(crate) struct BaseMask {
    masked: std::collections::HashSet<usize>,
    lengths: [f64; 4],
}

impl BaseMask {
    /// `masked` must name live documents of `base`.
    pub(crate) fn new(base: &LexicalIndex, mut masked: std::collections::HashSet<usize>) -> Self {
        masked.retain(|&doc| doc < base.documents.len());
        let mut lengths = [0.0; 4];
        for &doc in &masked {
            for (total, length) in lengths.iter_mut().zip(base.documents[doc].lengths) {
                *total += f64::from(length);
            }
        }
        Self { masked, lengths }
    }

    pub(crate) fn estimated_resident_bytes(&self) -> usize {
        self.masked.capacity() * std::mem::size_of::<usize>()
    }
}

// ---------------------------------------------------------------------------
// Reciprocal Rank Fusion
// ---------------------------------------------------------------------------

/// Combine multiple ranked lists of doc indices into a single merged ranking
/// using Reciprocal Rank Fusion (RRF).
///
/// For each ranking `r` and 0-based rank `i`, each document accumulates
/// `1.0 / (k + i + 1)`. `k = 60` is the standard default that smooths
/// contributions from deep positions. Returns the top `top_k` results sorted
/// by merged score descending; ties are broken by smaller `doc_idx` first.
pub fn rrf_merge(rankings: &[Vec<usize>], k: f64, top_k: usize) -> Vec<(usize, f64)> {
    if top_k == 0 || rankings.is_empty() {
        return Vec::new();
    }

    let mut scores: HashMap<usize, f64> = HashMap::new();

    for ranking in rankings {
        for (i, &doc_idx) in ranking.iter().enumerate() {
            *scores.entry(doc_idx).or_insert(0.0) += 1.0 / (k + i as f64 + 1.0);
        }
    }

    let mut ranked: Vec<(usize, f64)> = scores.into_iter().collect();
    ranked.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(top_k);
    ranked
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::semantic_search::SearchDocument;

    fn make_doc(path: &str, header: &str, symbols: &[&str], content: &str) -> SearchDocument {
        SearchDocument::new(
            path.to_string(),
            header.to_string(),
            symbols.iter().map(|s| s.to_string()).collect(),
            vec![],
            content.to_string(),
        )
    }

    // -----------------------------------------------------------------------
    // LexicalIndex tests
    // -----------------------------------------------------------------------

    /// A file that defines `resolveScopeMode` must outrank files that merely
    /// contain the words resolve, scope and mode.
    #[test]
    fn exact_identifier_outranks_files_that_only_share_its_parts() {
        let docs = vec![
            make_doc(
                "docs/a.md",
                "",
                &[],
                "resolve the scope and the mode of a request",
            ),
            make_doc("docs/b.md", "", &[], "scope mode resolve resolve"),
            make_doc(
                "src/plugin.ts",
                "",
                &["resolveScopeMode"],
                "function resolveScopeMode(request) { return scope; }",
            ),
        ];
        let idx = LexicalIndex::build(&docs);
        let results = idx.search("resolveScopeMode", 3);
        assert_eq!(results[0].0, 2, "{results:?}");
        assert!(results[0].1 > results[1].1, "{results:?}");
    }

    #[test]
    fn source_file_outranks_ten_times_larger_test_file() {
        let docs = vec![
            make_doc(
                "src/profile-enums.ts",
                "",
                &["numericProfileStatusToString"],
                "",
            ),
            make_doc(
                "packages/domains/profiles/test/profiles.db.test.ts",
                "",
                &[],
                &"numeric profile status ".repeat(10),
            ),
        ];
        let idx = LexicalIndex::build(&docs);
        let results = idx.search("numeric profile status", 2);

        assert_eq!(
            results[0].0, 0,
            "source file should rank first: {results:?}"
        );
    }

    #[test]
    fn generated_file_is_demoted_below_equally_matching_handwritten_file() {
        let docs = vec![
            make_doc(
                "src/generated/account_client.rs",
                "",
                &[],
                "hydrate account record",
            ),
            make_doc(
                "src/handwritten/account_client.rs",
                "",
                &[],
                "hydrate account record",
            ),
        ];
        let idx = LexicalIndex::build(&docs);
        let results = idx.search("hydrate account record", 2);

        assert_eq!(
            results[0].0, 1,
            "hand-written file should rank first: {results:?}"
        );
        let generated_score = results.iter().find(|(idx, _)| *idx == 0).unwrap().1;
        let handwritten_score = results.iter().find(|(idx, _)| *idx == 1).unwrap().1;
        assert!(
            (generated_score / handwritten_score - 0.4).abs() < 1e-12,
            "generated score should be 0.4 of handwritten score: {results:?}"
        );
    }

    #[test]
    fn matching_all_query_terms_beats_repeating_one_term_fifty_times() {
        let docs = vec![
            make_doc("src/repeater.rs", "", &[], &"alpha ".repeat(50)),
            make_doc("src/complete.rs", "", &[], "alpha beta gamma"),
        ];
        let idx = LexicalIndex::build(&docs);
        let results = idx.search("alpha beta gamma", 2);

        assert_eq!(
            results[0].0, 1,
            "full query coverage should rank first: {results:?}"
        );
        let repeater_score = results.iter().find(|(idx, _)| *idx == 0).unwrap().1;
        let alpha_results = idx.search("alpha", 2);
        let alpha_score = alpha_results.iter().find(|(idx, _)| *idx == 0).unwrap().1;
        assert!(
            (repeater_score - alpha_score / 9.0).abs() < 1e-12,
            "single-term coverage should reduce the repeater score by 1/9: query={results:?}, alpha={alpha_results:?}"
        );
    }

    #[test]
    fn shorter_document_wins_when_term_frequency_is_equal() {
        let docs = vec![
            make_doc(
                "src/long.rs",
                "",
                &[],
                &format!("needle {}", "padding ".repeat(200)),
            ),
            make_doc("src/short.rs", "", &[], "needle"),
        ];
        let idx = LexicalIndex::build(&docs);
        let results = idx.search("needle", 2);

        assert_eq!(
            results[0].0, 1,
            "shorter document should rank first: {results:?}"
        );
    }

    #[test]
    fn path_match_outranks_body_only_match() {
        let docs = vec![
            make_doc("src/worker.rs", "", &[], "payment retry"),
            make_doc("src/payment/retry.rs", "", &[], ""),
        ];
        let idx = LexicalIndex::build(&docs);
        let results = idx.search("payment retry", 2);

        assert_eq!(results[0].0, 1, "path match should rank first: {results:?}");
    }

    #[test]
    fn query_stopwords_do_not_change_ranking() {
        let docs = vec![
            make_doc("src/webhook.rs", "", &[], "configure webhook"),
            make_doc(
                "docs/configuration.md",
                "",
                &[],
                &format!("configure {}", "how to the ".repeat(20)),
            ),
        ];
        let idx = LexicalIndex::build(&docs);
        let without_stopwords: Vec<usize> = idx
            .search("configure webhook", 2)
            .into_iter()
            .map(|(doc_idx, _)| doc_idx)
            .collect();
        let with_stopwords: Vec<usize> = idx
            .search("how to configure the webhook", 2)
            .into_iter()
            .map(|(doc_idx, _)| doc_idx)
            .collect();

        assert_eq!(with_stopwords, without_stopwords);
    }

    #[test]
    fn test_path_prior_is_skipped_when_query_contains_test() {
        let docs = vec![
            make_doc(
                "src/parser.rs",
                "",
                &[],
                &format!("test {}", "parser ".repeat(50)),
            ),
            make_doc("src/parser.test.rs", "", &[], "parser test"),
        ];
        let idx = LexicalIndex::build(&docs);
        let ordinary_results = idx.search("parser", 2);
        let test_results = idx.search("parser test", 2);

        assert_eq!(
            ordinary_results[0].0, 0,
            "test prior should apply to an ordinary query: {ordinary_results:?}"
        );
        assert_eq!(
            test_results[0].0, 1,
            "test prior should be skipped for a test query: {test_results:?}"
        );
    }

    #[test]
    fn fixture_paths_are_test_paths_and_fixture_query_skips_prior() {
        for fixture_path in [
            "src/__fixtures__/event.json",
            "src/fixtures/event.json",
            "src/__mocks__/event.ts",
            "src/testdata/event.json",
            "src/test-data/event.json",
            "src/event.fixture.json",
            "src/event.snap",
        ] {
            let docs = vec![
                make_doc(fixture_path, "", &[], "hydrate fixture payload"),
                make_doc("src/event.rs", "", &[], "hydrate fixture payload"),
            ];
            let idx = LexicalIndex::build(&docs);

            let ordinary_results = idx.search("hydrate payload", 2);
            assert_eq!(
                ordinary_results[0].0, 1,
                "fixture path should be demoted for an ordinary query: path={fixture_path}, results={ordinary_results:?}"
            );

            for query in ["hydrate fixture payload", "hydrate test payload"] {
                let intent_results = idx.search(query, 2);
                assert_eq!(
                    intent_results[0].0, 0,
                    "fixture prior should be skipped for explicit test intent: path={fixture_path}, query={query}, results={intent_results:?}"
                );
            }
        }
    }

    #[test]
    fn repeated_occurrences_rank_higher() {
        let docs = vec![
            make_doc("a.rs", "", &[], "token"),
            make_doc("b.rs", "", &[], "token token token token"),
        ];
        let idx = LexicalIndex::build(&docs);
        let results = idx.search("token", 2);
        assert_eq!(results[0].0, 1, "{results:?}");
    }

    #[test]
    fn snake_case_identifier_is_a_token_of_its_own() {
        let docs = vec![
            make_doc("a.rs", "", &[], "verify the token"),
            make_doc("b.rs", "", &["verify_token"], "fn verify_token() {}"),
        ];
        let idx = LexicalIndex::build(&docs);
        let results = idx.search("verify_token", 2);
        assert_eq!(results[0].0, 1, "{results:?}");
    }

    #[test]
    fn empty_index_returns_no_results() {
        let idx = LexicalIndex::build(&[]);
        let results = idx.search("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn single_doc_query_matches_returns_it() {
        let docs = vec![make_doc("foo/bar.rs", "Bar module", &[], "hello world")];
        let idx = LexicalIndex::build(&docs);
        let results = idx.search("hello", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn two_docs_query_in_only_one_returns_only_that_one() {
        let docs = vec![
            make_doc("a.rs", "Alpha", &[], "unique alpha content"),
            make_doc("b.rs", "Beta", &[], "different beta content"),
        ];
        let idx = LexicalIndex::build(&docs);
        let results = idx.search("unique", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn idf_rare_token_outranks_common_token() {
        // "rare" appears in only doc 0; "common" appears in all three.
        // A query for both tokens should give doc 0 the highest score.
        let docs = vec![
            make_doc("a.rs", "A", &[], "rare common"),
            make_doc("b.rs", "B", &[], "common"),
            make_doc("c.rs", "C", &[], "common"),
        ];
        let idx = LexicalIndex::build(&docs);
        let results = idx.search("rare common", 3);
        // doc 0 has both tokens and the rare token boost — it must be first.
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn search_is_case_insensitive() {
        let docs = vec![make_doc("x.rs", "X", &[], "Hello World")];
        let idx = LexicalIndex::build(&docs);
        // Upper-case query should still match.
        let r1 = idx.search("HELLO", 5);
        let r2 = idx.search("hello", 5);
        assert_eq!(r1.len(), 1);
        assert_eq!(r2.len(), 1);
        assert_eq!(r1[0].0, r2[0].0);
    }

    #[test]
    fn camel_case_token_split_indexed_and_searchable() {
        // "verifyToken" should be split into "verify" + "token" at index time.
        let docs = vec![make_doc("auth.rs", "Auth", &["verifyToken"], "")];
        let idx = LexicalIndex::build(&docs);
        // Querying the sub-token "verify" should find doc 0.
        let results = idx.search("verify", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn document_count_reflects_build() {
        let docs = vec![
            make_doc("a.rs", "", &[], ""),
            make_doc("b.rs", "", &[], ""),
            make_doc("c.rs", "", &[], ""),
        ];
        let idx = LexicalIndex::build(&docs);
        assert_eq!(idx.document_count(), 3);
    }

    fn corpus_of_rare_terms() -> Vec<SearchDocument> {
        (0..400)
            .map(|i| {
                let body: Vec<String> = (0..250).map(|j| format!("rare{i}term{j}")).collect();
                make_doc(&format!("src/file_{i}.rs"), "", &[], &body.join(" "))
            })
            .collect()
    }

    #[test]
    fn index_holds_each_rare_term_in_a_few_dozen_bytes() {
        let docs = corpus_of_rare_terms();
        let (index, bytes) = crate::alloc_probe::retained_bytes(|| LexicalIndex::build(&docs));
        let postings = index.posting_count();
        assert!(postings >= 100_000, "corpus too small: {postings}");
        assert!(
            bytes / postings <= 64,
            "{} bytes per posting ({bytes} bytes for {postings} postings)",
            bytes / postings
        );
    }

    #[test]
    fn one_document_update_only_visits_affected_postings_and_lengths() {
        let docs: Vec<_> = (0..2_000)
            .map(|i| {
                make_doc(
                    &format!("src/file_{i}.rs"),
                    &format!("header_{i}"),
                    &[&format!("symbol_{i}")],
                    &format!("unique_body_token_{i} shared_token"),
                )
            })
            .collect();
        let mut index = LexicalIndex::build(&docs);

        index.update_documents(
            vec![(
                777,
                make_doc(
                    "src/file_777.rs",
                    "replacement_header",
                    &["replacement_symbol"],
                    "replacement_body_token shared_token",
                ),
            )],
            &[],
        );

        let work = index.last_update_work();
        assert!(
            work.postings_visited <= 16,
            "one-file refresh traversed unrelated posting entries: {work:?}"
        );
        assert!(
            work.documents_visited <= 1,
            "one-file refresh recomputed lengths across the corpus: {work:?}"
        );
        assert_eq!(index.search("replacement_body_token", 1)[0].0, 777);
    }

    // -----------------------------------------------------------------------
    // rrf_merge tests
    // -----------------------------------------------------------------------

    #[test]
    fn rrf_merge_single_ranking_returns_it() {
        let ranking = vec![0usize, 1, 2];
        let merged = rrf_merge(std::slice::from_ref(&ranking), 60.0, 10);
        // Order must be preserved (rank 0 has the highest RRF score).
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].0, 0);
        assert_eq!(merged[1].0, 1);
        assert_eq!(merged[2].0, 2);
    }

    #[test]
    fn rrf_merge_two_rankings_doc_in_both_ranks_higher() {
        // ranking A: [0, 1]   ranking B: [1, 2]
        // doc 1 appears in both → higher merged score than doc 0 or doc 2.
        let r_a = vec![0usize, 1];
        let r_b = vec![1usize, 2];
        let merged = rrf_merge(&[r_a, r_b], 60.0, 10);
        assert_eq!(merged[0].0, 1, "doc 1 should rank first");
    }

    #[test]
    fn rrf_merge_larger_k_flattens_contributions() {
        // With larger k the difference between rank-0 and rank-1 scores shrinks.
        let ranking = vec![0usize, 1];
        let merged_small_k = rrf_merge(std::slice::from_ref(&ranking), 1.0, 2);
        let merged_large_k = rrf_merge(std::slice::from_ref(&ranking), 1000.0, 2);
        let diff_small = merged_small_k[0].1 - merged_small_k[1].1;
        let diff_large = merged_large_k[0].1 - merged_large_k[1].1;
        assert!(
            diff_small > diff_large,
            "larger k should produce flatter score differences"
        );
    }

    #[test]
    fn rrf_merge_respects_top_k_cap() {
        let ranking: Vec<usize> = (0..20).collect();
        let merged = rrf_merge(&[ranking], 60.0, 5);
        assert_eq!(merged.len(), 5);
    }

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    const WORDS: &[&str] = &[
        "alpha", "beta", "scope", "mode", "resolve", "account", "record", "hydrate", "profile",
        "status", "numeric", "invoice", "payment", "worker", "queue", "token", "parser", "cache",
        "test", "fixture",
    ];

    fn random_words(rng: &mut Rng, count: usize) -> Vec<String> {
        (0..count)
            .map(|_| {
                let word = WORDS[rng.below(WORDS.len())];
                if rng.below(4) == 0 {
                    let next = WORDS[rng.below(WORDS.len())];
                    format!("{word}{}{}", next[..1].to_uppercase(), &next[1..])
                } else {
                    word.to_string()
                }
            })
            .collect()
    }

    fn random_doc(rng: &mut Rng, path: String) -> SearchDocument {
        let symbol_count = rng.below(3);
        let symbols = random_words(rng, symbol_count);
        let header_len = rng.below(5);
        let header = random_words(rng, header_len).join(" ");
        let body_len = 1 + rng.below(40);
        let body = random_words(rng, body_len).join(" ");
        SearchDocument::new(path, header, symbols, vec![], body)
    }

    fn random_path(rng: &mut Rng, i: usize) -> String {
        match rng.below(5) {
            0 => format!("tests/case_{i}.test.ts"),
            1 => format!("docs/note_{i}.md"),
            2 => format!("src/generated/gen_{i}.rs"),
            _ => format!("src/module_{i}.rs"),
        }
    }

    fn by_score_then_path(hits: &mut [(String, f64)]) {
        hits.sort_by(|a, b| {
            let key = |score: f64| (score * 1e9).round() as i64;
            key(b.1).cmp(&key(a.1)).then_with(|| a.0.cmp(&b.0))
        });
    }

    /// A delta index over a masked base answers like one index built over the
    /// worktree's whole corpus: same files, same order, same scores.
    #[test]
    fn delta_over_masked_base_matches_standalone_index_of_random_edits() {
        for seed in 1..=60_u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let base_len = 8 + rng.below(30);
            let mut base_docs: Vec<SearchDocument> = (0..base_len)
                .map(|i| {
                    let path = random_path(&mut rng, i);
                    random_doc(&mut rng, path)
                })
                .collect();
            let (base, mut base_paths) = if seed % 2 == 0 {
                (LexicalIndex::build(&base_docs), Vec::new())
            } else {
                // A base kept current incrementally carries a tombstone.
                base_docs.push(random_doc(&mut rng, "src/removed.rs".into()));
                let mut index = LexicalIndex::build(&base_docs);
                index.update_documents(vec![], &[base_docs.len() - 1]);
                base_docs.pop();
                (index, vec![String::new()])
            };
            base_paths.splice(0..0, base_docs.iter().map(|doc| doc.path.clone()));

            let mut worktree: Vec<Option<SearchDocument>> =
                base_docs.iter().cloned().map(Some).collect();
            let mut masked = std::collections::HashSet::new();
            let mut added = Vec::new();
            let op_count = 1 + rng.below(8);
            for op in 0..op_count {
                let target = rng.below(base_docs.len());
                match rng.below(3) {
                    0 => {
                        worktree[target] = None;
                        masked.insert(target);
                    }
                    1 => {
                        let path = base_docs[target].path.clone();
                        worktree[target] = Some(random_doc(&mut rng, path));
                        masked.insert(target);
                    }
                    _ => added.push(random_doc(&mut rng, format!("src/added_{op}.rs"))),
                }
            }
            let mut delta_docs = added.clone();
            delta_docs.extend(masked.iter().filter_map(|&i| worktree[i].clone()));
            let mut standalone_docs: Vec<SearchDocument> =
                worktree.iter().flatten().cloned().collect();
            standalone_docs.extend(added);
            let standalone = LexicalIndex::build(&standalone_docs);
            let delta = LexicalIndex::build(&delta_docs);
            let mask = BaseMask::new(&base, masked.iter().copied().collect());
            let deleted: Vec<&String> = masked
                .iter()
                .filter(|&&i| worktree[i].is_none())
                .map(|&i| &base_docs[i].path)
                .collect();

            for _ in 0..6 {
                let query_len = 1 + rng.below(3);
                let query = random_words(&mut rng, query_len).join(" ");
                let mut layered: Vec<(String, f64)> = delta
                    .search_over(&base, &mask, &query)
                    .into_iter()
                    .map(|(in_delta, i, score)| {
                        let path = if in_delta {
                            delta_docs[i].path.clone()
                        } else {
                            base_paths[i].clone()
                        };
                        (path, score)
                    })
                    .collect();
                let mut expected: Vec<(String, f64)> = standalone
                    .search(&query, usize::MAX)
                    .into_iter()
                    .map(|(i, score)| (standalone_docs[i].path.clone(), score))
                    .collect();
                by_score_then_path(&mut layered);
                by_score_then_path(&mut expected);
                assert!(!expected.is_empty() || layered.is_empty());
                assert_eq!(
                    layered.iter().map(|hit| &hit.0).collect::<Vec<_>>(),
                    expected.iter().map(|hit| &hit.0).collect::<Vec<_>>(),
                    "seed {seed} query {query:?}"
                );
                for (got, want) in layered.iter().zip(&expected) {
                    assert!(
                        (got.1 - want.1).abs() <= 1e-9 * want.1.abs().max(1.0),
                        "seed {seed} query {query:?}: {got:?} vs {want:?}"
                    );
                }
                assert!(
                    !layered.iter().any(|(path, _)| deleted.contains(&path)),
                    "seed {seed}: a deleted file was returned"
                );
            }
        }
    }
}
