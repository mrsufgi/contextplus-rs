// TODO: For very large corpora (>~10k files) callers may swap this in-process
// index for a tantivy-backed index in a follow-up. The public API of
// `LexicalIndex` (build / search / document_count) and `rrf_merge` are
// designed to be drop-in replaceable.

//! In-process lexical inverted index with TF-IDF scoring and Reciprocal Rank
//! Fusion (RRF) merge.
//!
//! No external dependencies — pure Rust, no I/O, no async.

use std::collections::HashMap;

use crate::tools::semantic_search::{SearchDocument, split_camel_case};

// ---------------------------------------------------------------------------
// LexicalIndex
// ---------------------------------------------------------------------------

/// In-process inverted index over a [`SearchDocument`] slice.
///
/// Construct via [`LexicalIndex::build`]. Each posting maps a lowercase token
/// → sorted `Vec` of doc indices that contain it (binary presence — multiple
/// occurrences do **not** inflate the posting list).
pub struct LexicalIndex {
    /// token → sorted list of doc indices that contain the token.
    posting: HashMap<String, Vec<usize>>,
    doc_count: usize,
}

impl LexicalIndex {
    /// Build an index from a slice of [`SearchDocument`]s.
    ///
    /// Tokenises `path + header + symbols.join(' ') + content` via
    /// [`split_camel_case`], lowercases each token, and records the unique
    /// `(token, doc_idx)` edges.
    pub fn build(docs: &[SearchDocument]) -> Self {
        let mut posting: HashMap<String, Vec<usize>> = HashMap::new();

        for (idx, doc) in docs.iter().enumerate() {
            let raw = format!(
                "{} {} {} {}",
                doc.path,
                doc.header,
                doc.symbols.join(" "),
                doc.content
            );
            // Collect the unique lowercased tokens for this document.
            let mut tokens: Vec<String> = split_camel_case(&raw)
                .into_iter()
                .map(|t| t.to_lowercase())
                .collect();
            tokens.sort_unstable();
            tokens.dedup();

            for token in tokens {
                posting.entry(token).or_default().push(idx);
            }
        }

        // Posting lists are already in ascending order because we iterate
        // doc indices 0..N in order, but sort them explicitly for safety.
        for list in posting.values_mut() {
            list.sort_unstable();
            list.dedup();
        }

        Self {
            posting,
            doc_count: docs.len(),
        }
    }

    /// Number of documents in the index.
    pub fn document_count(&self) -> usize {
        self.doc_count
    }

    /// Search the index for `query`, returning up to `top_k` `(doc_idx,
    /// score)` pairs sorted by score descending (ties broken by smaller
    /// `doc_idx` first).
    ///
    /// Score = Σ over query tokens of `idf(token) * tf_present(token, doc)`
    ///
    /// * `idf(token) = ln((1 + doc_count) / (1 + df)) + 1`  (smoothed IDF)
    /// * `tf_present` is binary: 1.0 if the doc contains the token, else 0.0
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(usize, f64)> {
        if top_k == 0 || self.doc_count == 0 {
            return Vec::new();
        }

        let query_tokens: Vec<String> = split_camel_case(query)
            .into_iter()
            .map(|t| t.to_lowercase())
            .collect();

        // Accumulate per-doc scores.
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for token in &query_tokens {
            if let Some(postings) = self.posting.get(token.as_str()) {
                let df = postings.len() as f64;
                let idf = ((1.0 + self.doc_count as f64) / (1.0 + df)).ln() + 1.0;
                for &doc_idx in postings {
                    // tf_present = 1.0 (binary)
                    *scores.entry(doc_idx).or_insert(0.0) += idf;
                }
            }
        }

        let mut ranked: Vec<(usize, f64)> = scores.into_iter().collect();
        // Sort descending by score; ties → ascending by doc_idx.
        ranked.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(top_k);
        ranked
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

    // -----------------------------------------------------------------------
    // rrf_merge tests
    // -----------------------------------------------------------------------

    #[test]
    fn rrf_merge_single_ranking_returns_it() {
        let ranking = vec![0usize, 1, 2];
        let merged = rrf_merge(&[ranking.clone()], 60.0, 10);
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
        let merged_small_k = rrf_merge(&[ranking.clone()], 1.0, 2);
        let merged_large_k = rrf_merge(&[ranking.clone()], 1000.0, 2);
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
}
