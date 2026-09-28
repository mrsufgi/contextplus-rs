// TODO: For very large corpora (>~10k files) callers may swap this in-process
// index for a tantivy-backed index in a follow-up. The public API of
// `LexicalIndex` (build / search / document_count) and `rrf_merge` are
// designed to be drop-in replaceable.

//! In-process lexical inverted index with BM25F scoring and Reciprocal Rank
//! Fusion (RRF) merge.
//!
//! No external dependencies — pure Rust, no I/O, no async.

use std::collections::HashMap;
use std::sync::Arc;

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

type PostingList = HashMap<usize, [u32; 4]>;

/// In-process inverted index over a [`SearchDocument`] slice.
#[derive(Clone)]
pub struct LexicalIndex {
    posting: HashMap<String, Arc<PostingList>>,
    documents: Vec<DocumentFields>,
    average_lengths: [f64; 4],
    total_lengths: [f64; 4],
    document_terms: Vec<Arc<Vec<String>>>,
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

impl LexicalIndex {
    pub(crate) fn estimated_resident_bytes(&self) -> usize {
        self.posting
            .iter()
            .map(|(term, list)| {
                term.capacity()
                    + list.capacity()
                        * (std::mem::size_of::<usize>() + 4 * std::mem::size_of::<u32>())
            })
            .sum::<usize>()
            + self.documents.capacity() * std::mem::size_of::<DocumentFields>()
            + self
                .document_terms
                .iter()
                .flat_map(|terms| terms.iter())
                .map(String::capacity)
                .sum::<usize>()
    }

    /// Build an index from a slice of [`SearchDocument`]s.
    ///
    /// Records separate path, definition-name, header and body frequencies.
    pub fn build(docs: &[SearchDocument]) -> Self {
        let mut posting: HashMap<String, Arc<PostingList>> = HashMap::new();
        let mut documents = Vec::with_capacity(docs.len());
        let mut average_lengths = [0.0; 4];
        let mut document_terms = Vec::with_capacity(docs.len());

        for (idx, doc) in docs.iter().enumerate() {
            let symbols = doc.symbols.join(" ");
            let mut terms: HashMap<String, [u32; 4]> = HashMap::new();
            let mut lengths = [0_u32; 4];
            for (field, text) in [doc.path.as_str(), &symbols, &doc.header, &doc.content]
                .into_iter()
                .enumerate()
            {
                for (token, count) in token_counts(text) {
                    lengths[field] = lengths[field].saturating_add(count);
                    terms.entry(token).or_default()[field] = count;
                }
                average_lengths[field] += f64::from(lengths[field]);
            }
            document_terms.push(Arc::new(terms.keys().cloned().collect()));
            for (token, counts) in terms {
                Arc::make_mut(posting.entry(token).or_default()).insert(idx, counts);
            }
            let classification = classify_path_prior(&doc.path);
            let is_test = classification.is_test_like;
            let prior = classification.non_test_multiplier();
            documents.push(DocumentFields {
                lengths,
                prior,
                is_test,
            });
        }
        let total_lengths = average_lengths;
        for length in &mut average_lengths {
            *length = if docs.is_empty() {
                1.0
            } else {
                (*length / docs.len() as f64).max(f64::EPSILON)
            };
        }
        Self {
            posting,
            documents,
            average_lengths,
            total_lengths,
            document_terms,
            doc_count: docs.len(),
            #[cfg(test)]
            last_update_work: UpdateWork::default(),
        }
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
                for term in Arc::make_mut(&mut self.document_terms[i]).drain(..) {
                    if let Some(entries) = self.posting.get_mut(&term) {
                        #[cfg(test)]
                        {
                            postings_visited += 1;
                        }
                        Arc::make_mut(entries).remove(&i);
                        if entries.is_empty() {
                            self.posting.remove(&term);
                        }
                    }
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
            let mut single = Self::build(&[doc]);
            for (total, length) in self.total_lengths.iter_mut().zip(single.total_lengths) {
                *total += length;
            }
            let terms = single.document_terms.remove(0);
            if i >= self.documents.len() {
                self.document_terms.push(terms);
                self.doc_count += 1;
                self.documents.push(single.documents.remove(0));
            } else {
                self.document_terms[i] = terms;
                self.documents[i] = single.documents.remove(0);
            }
            for (term, entries) in single.posting {
                Arc::make_mut(self.posting.entry(term).or_default()).insert(i, entries[&0]);
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
            .map(|token| self.posting.get(token).map_or(0.0, |p| p.len() as f64))
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
            if let Some(postings) = self.posting.get(token) {
                let idf = (1.0 + (doc_count - df + 0.5) / (df + 0.5)).ln();
                for (&doc_idx, counts) in postings.iter() {
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
                let in_base = base.posting.get(token).map_or(0, |postings| {
                    postings.len()
                        - mask
                            .masked
                            .iter()
                            .filter(|doc| postings.contains_key(doc))
                            .count()
                });
                let in_delta = self.posting.get(token).map_or(0, |p| p.len());
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
