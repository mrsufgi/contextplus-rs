//! File-level semantic search with hybrid scoring (semantic + keyword).
//!
//! Ports the TypeScript `semantic-search.ts` logic:
//! - Builds a SearchIndex from file headers + symbols + content
//! - Uses Ollama embeddings for semantic similarity
//! - Combines semantic score with keyword coverage for hybrid ranking

use std::borrow::Cow;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::{ContextPlusError, Result};

/// Type alias for the boxed future returned by embedding functions.
type EmbedFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>;

/// Type alias for the boxed future returned by walk-and-index functions.
type WalkAndIndexFuture<'a> = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)>>
            + Send
            + 'a,
    >,
>;

// ---------------------------------------------------------------------------
// Constants (matching TS reference)
// ---------------------------------------------------------------------------

const DEFAULT_SEMANTIC_WEIGHT: f64 = 0.72;
const DEFAULT_KEYWORD_WEIGHT: f64 = 0.28;
const DEFAULT_TOP_K: usize = 5;
const MAX_TOP_K: usize = 50;
const MAX_QUERY_LEN: usize = 2000;
const DEFAULT_MIN_COMBINED_SCORE: f64 = 0.1;
const PHRASE_BOOST: f64 = 0.15;
const TERM_COVERAGE_WEIGHT: f64 = 0.65;
const SYMBOL_COVERAGE_WEIGHT: f64 = 0.20;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SemanticSearchOptions {
    pub root_dir: PathBuf,
    pub query: String,
    pub top_k: Option<usize>,
    pub semantic_weight: Option<f64>,
    pub keyword_weight: Option<f64>,
    pub min_semantic_score: Option<f64>,
    pub min_keyword_score: Option<f64>,
    pub min_combined_score: Option<f64>,
    pub require_keyword_match: Option<bool>,
    pub require_semantic_match: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub path: String,
    pub score: f64,
    pub semantic_score: f64,
    pub keyword_score: f64,
    pub header: String,
    pub matched_symbols: Vec<String>,
    pub matched_symbol_locations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SearchDocument {
    pub path: String,
    pub header: String,
    pub symbols: Vec<String>,
    pub symbol_entries: Vec<SymbolSearchEntry>,
    pub content: String,
    /// Pre-computed lowercase searchable text for keyword scoring.
    /// Built once at index time to avoid `format!()` + `to_lowercase()` per query.
    pub search_text: String,
    /// Pre-computed lowercase terms from all fields for term coverage.
    pub search_terms: HashSet<String>,
    /// Pre-computed camelCase-split token sets per symbol (parallel to `symbols`).
    pub symbol_tokens: Vec<HashSet<String>>,
    /// Pre-computed camelCase-split token sets per symbol entry (parallel to `symbol_entries`).
    pub symbol_entry_tokens: Vec<HashSet<String>>,
}

impl SearchDocument {
    /// Create a SearchDocument with pre-computed search fields.
    pub fn new(
        path: String,
        header: String,
        symbols: Vec<String>,
        symbol_entries: Vec<SymbolSearchEntry>,
        content: String,
    ) -> Self {
        let raw_text = format!("{} {} {} {}", path, header, symbols.join(" "), content);
        let search_text = raw_text.to_lowercase();
        let search_terms = split_camel_case(&raw_text).into_iter().collect();
        let symbol_tokens = symbols
            .iter()
            .map(|s| split_camel_case(s).into_iter().collect())
            .collect();
        let symbol_entry_tokens = symbol_entries
            .iter()
            .map(|e| split_camel_case(&e.name).into_iter().collect())
            .collect();
        Self {
            path,
            header,
            symbols,
            symbol_entries,
            content,
            search_text,
            search_terms,
            symbol_tokens,
            symbol_entry_tokens,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SymbolSearchEntry {
    pub name: String,
    pub kind: Option<String>,
    pub line: usize,
    pub end_line: Option<usize>,
    pub signature: Option<String>,
}

// ---------------------------------------------------------------------------
// Resolved options (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ResolvedSearchOptions {
    pub top_k: usize,
    pub semantic_weight: f64,
    pub keyword_weight: f64,
    pub min_semantic_score: f64,
    pub min_keyword_score: f64,
    pub min_combined_score: f64,
    pub require_keyword_match: bool,
    pub require_semantic_match: bool,
}

// ---------------------------------------------------------------------------
// Pure helper functions
// ---------------------------------------------------------------------------

/// Split camelCase, snake_case, kebab-case text into lowercase tokens.
/// Filters tokens with length > 1.
/// "getUserById" -> ["get", "user", "by", "id"]
pub fn split_camel_case(text: &str) -> Vec<String> {
    let mut result = String::with_capacity(text.len() + 16);
    let mut prev: Option<char> = None;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if let Some(p) = prev {
            // camelCase boundary: lowercase followed by uppercase
            if p.is_ascii_lowercase() && c.is_ascii_uppercase() {
                result.push(' ');
            }
            // ACRONYMWord boundary: uppercase followed by uppercase+lowercase
            if p.is_ascii_uppercase()
                && c.is_ascii_uppercase()
                && chars.peek().is_some_and(|next| next.is_ascii_lowercase())
            {
                result.push(' ');
            }
        }
        result.push(c);
        prev = Some(c);
    }

    result
        .to_lowercase()
        .split(|c: char| c == ' ' || c == '_' || c == '-' || !c.is_ascii_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|s| s.to_string())
        .collect()
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn normalize_weight(value: Option<f64>, fallback: f64) -> f64 {
    match value {
        Some(v) if v.is_finite() && v >= 0.0 => v,
        _ => fallback,
    }
}

fn normalize_threshold(value: Option<f64>, fallback: f64) -> f64 {
    match value {
        Some(v) if v.is_finite() => {
            if v > 1.0 {
                clamp01(v / 100.0)
            } else {
                clamp01(v)
            }
        }
        _ => fallback,
    }
}

fn normalize_top_k(value: Option<usize>, fallback: usize) -> usize {
    match value {
        Some(k) if k >= 1 => k.min(MAX_TOP_K),
        _ => fallback,
    }
}

fn resolve_search_options(opts: &SemanticSearchOptions) -> ResolvedSearchOptions {
    ResolvedSearchOptions {
        top_k: normalize_top_k(opts.top_k, DEFAULT_TOP_K),
        semantic_weight: normalize_weight(opts.semantic_weight, DEFAULT_SEMANTIC_WEIGHT),
        keyword_weight: normalize_weight(opts.keyword_weight, DEFAULT_KEYWORD_WEIGHT),
        min_semantic_score: normalize_threshold(opts.min_semantic_score, 0.0),
        min_keyword_score: normalize_threshold(opts.min_keyword_score, 0.0),
        min_combined_score: normalize_threshold(
            opts.min_combined_score,
            DEFAULT_MIN_COMBINED_SCORE,
        ),
        require_keyword_match: opts.require_keyword_match.unwrap_or(false),
        require_semantic_match: opts.require_semantic_match.unwrap_or(false),
    }
}

/// Cosine similarity between two f32 vectors.
/// Delegates to simsimd for SIMD-accelerated computation.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    if a.iter().all(|&v| v == 0.0) {
        return 0.0;
    }
    crate::core::embeddings::cosine_similarity_simsimd(a, b) as f64
}

/// Get term coverage: fraction of query terms that appear in doc terms.
fn get_term_coverage(query_terms: &HashSet<String>, doc_terms: &HashSet<String>) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let matched = query_terms
        .iter()
        .filter(|t| doc_terms.contains(*t))
        .count();
    matched as f64 / query_terms.len() as f64
}

/// Find symbols whose pre-computed tokens overlap with query terms.
fn get_matched_symbols(
    symbols: &[String],
    symbol_tokens: &[HashSet<String>],
    query_terms: &HashSet<String>,
) -> Vec<String> {
    if query_terms.is_empty() {
        return Vec::new();
    }
    symbols
        .iter()
        .zip(symbol_tokens.iter())
        .filter(|(_, tokens)| tokens.iter().any(|t| query_terms.contains(t)))
        .map(|(s, _)| s.clone())
        .collect()
}

/// Find symbol entries whose pre-computed tokens overlap with query terms.
fn get_matched_symbol_entries<'a>(
    entries: &'a [SymbolSearchEntry],
    entry_tokens: &[HashSet<String>],
    query_terms: &HashSet<String>,
) -> Vec<&'a SymbolSearchEntry> {
    if query_terms.is_empty() {
        return Vec::new();
    }
    entries
        .iter()
        .zip(entry_tokens.iter())
        .filter(|(_, tokens)| tokens.iter().any(|t| query_terms.contains(t)))
        .map(|(e, _)| e)
        .collect()
}

fn format_line_range(line: usize, end_line: Option<usize>) -> String {
    match end_line {
        Some(el) if el > line => format!("L{}-L{}", line, el),
        _ => format!("L{}", line),
    }
}

/// Compute keyword score from term coverage, symbol coverage, and phrase boost.
/// Uses pre-computed `doc.search_text`, `doc.search_terms`, and `doc.symbol_tokens`
/// to avoid per-query allocations.
fn compute_keyword_score(
    query_lower: &str,
    query_terms: &HashSet<String>,
    doc: &SearchDocument,
    matched_symbols: &[String],
) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let phrase_boost = if !query_lower.is_empty() && doc.search_text.contains(query_lower) {
        PHRASE_BOOST
    } else {
        0.0
    };
    // Build symbol_terms from pre-computed token sets (no split_camel_case at query time)
    let symbol_terms: HashSet<&String> = doc
        .symbol_tokens
        .iter()
        .zip(doc.symbols.iter())
        .filter(|(_, sym)| matched_symbols.contains(sym))
        .flat_map(|(tokens, _)| tokens.iter())
        .collect();
    let term_coverage = get_term_coverage(query_terms, &doc.search_terms);
    let symbol_coverage = if query_terms.is_empty() {
        0.0
    } else {
        let matched = query_terms
            .iter()
            .filter(|t| symbol_terms.contains(t))
            .count();
        matched as f64 / query_terms.len() as f64
    };
    clamp01(
        term_coverage * TERM_COVERAGE_WEIGHT
            + symbol_coverage * SYMBOL_COVERAGE_WEIGHT
            + phrase_boost,
    )
}

/// Combine semantic and keyword scores with configured weights.
fn compute_combined_score(
    semantic_score: f64,
    keyword_score: f64,
    opts: &ResolvedSearchOptions,
) -> f64 {
    let semantic_component = semantic_score.max(0.0);
    let total_weight = opts.semantic_weight + opts.keyword_weight;
    if total_weight <= 0.0 {
        return semantic_component;
    }
    clamp01(
        (opts.semantic_weight * semantic_component + opts.keyword_weight * keyword_score)
            / total_weight,
    )
}

/// Truncate query to MAX_QUERY_LEN characters.
/// Returns `Cow::Borrowed` when no modification is needed (zero allocation).
pub fn sanitize_query(query: &str) -> Cow<'_, str> {
    let q = query.trim();
    if q.len() > MAX_QUERY_LEN {
        Cow::Owned(crate::core::parser::truncate_to_char_boundary(q, MAX_QUERY_LEN).to_string())
    } else {
        Cow::Borrowed(q)
    }
}

// ---------------------------------------------------------------------------
// SearchIndex -- indexes documents and runs hybrid search
// ---------------------------------------------------------------------------

/// In-memory search index holding documents and their embedding vectors.
/// Vectors are stored in a flat contiguous buffer for cache-friendly SIMD access.
pub struct SearchIndex {
    documents: Vec<SearchDocument>,
    /// Flat buffer: `vector_buffer[i * dims .. (i+1) * dims]` is the vector for doc `i`.
    /// Docs without vectors have `has_vector[i] == false` and zeros in the buffer.
    vector_buffer: Vec<f32>,
    /// Which docs have valid embedding vectors.
    has_vector: Vec<bool>,
    /// Dimensions per vector (0 if no vectors indexed yet).
    dims: usize,
}

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchIndex {
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
            vector_buffer: Vec::new(),
            has_vector: Vec::new(),
            dims: 0,
        }
    }

    /// Index documents using pre-computed vectors.
    /// `vectors` must be the same length as `docs`.
    pub fn index_with_vectors(
        &mut self,
        docs: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
    ) {
        debug_assert_eq!(docs.len(), vectors.len());
        // Determine dims from first non-None vector
        let dims = vectors
            .iter()
            .find_map(|v| v.as_ref().map(|v| v.len()))
            .unwrap_or(0);
        let n = docs.len();
        let mut buffer = vec![0.0f32; n * dims];
        let mut has_vec = Vec::with_capacity(n);
        for (i, v) in vectors.iter().enumerate() {
            if let Some(vec) = v {
                let offset = i * dims;
                buffer[offset..offset + dims].copy_from_slice(vec);
                has_vec.push(true);
            } else {
                has_vec.push(false);
            }
        }
        self.documents = docs;
        self.vector_buffer = buffer;
        self.has_vector = has_vec;
        self.dims = dims;
    }

    /// Perform hybrid search against the indexed documents.
    pub fn search(
        &self,
        query: &str,
        query_vec: &[f32],
        opts: &ResolvedSearchOptions,
    ) -> Vec<SearchResult> {
        let query_terms: HashSet<String> = split_camel_case(query).into_iter().collect();
        let query_lower = query.trim().to_lowercase();

        #[allow(clippy::type_complexity)]
        let mut scored: Vec<(usize, f64, f64, f64, Vec<String>, Vec<String>)> = self
            .documents
            .par_iter()
            .enumerate()
            .filter_map(|(i, doc)| {
                if !self.has_vector[i] {
                    return None;
                }
                let offset = i * self.dims;
                let vec_slice = &self.vector_buffer[offset..offset + self.dims];
                let semantic_score = cosine(query_vec, vec_slice);

                let matched_entries = get_matched_symbol_entries(
                    &doc.symbol_entries,
                    &doc.symbol_entry_tokens,
                    &query_terms,
                );
                let matched_symbols = if !matched_entries.is_empty() {
                    matched_entries.iter().map(|e| e.name.clone()).collect()
                } else {
                    get_matched_symbols(&doc.symbols, &doc.symbol_tokens, &query_terms)
                };
                let matched_symbol_locations: Vec<String> = matched_entries
                    .iter()
                    .map(|e| format!("{}@{}", e.name, format_line_range(e.line, e.end_line)))
                    .collect();

                let keyword_score =
                    compute_keyword_score(&query_lower, &query_terms, doc, &matched_symbols);
                let combined_score = compute_combined_score(semantic_score, keyword_score, opts);

                if opts.require_semantic_match && semantic_score <= 0.0 {
                    return None;
                }
                if opts.require_keyword_match && keyword_score <= 0.0 {
                    return None;
                }
                if semantic_score.max(0.0) < opts.min_semantic_score {
                    return None;
                }
                if keyword_score < opts.min_keyword_score {
                    return None;
                }
                if combined_score < opts.min_combined_score {
                    return None;
                }

                Some((
                    i,
                    combined_score,
                    semantic_score,
                    keyword_score,
                    matched_symbols,
                    matched_symbol_locations,
                ))
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        scored
            .into_iter()
            .take(opts.top_k)
            .map(
                |(
                    idx,
                    score,
                    semantic_score,
                    keyword_score,
                    matched_symbols,
                    matched_symbol_locations,
                )| {
                    let doc = &self.documents[idx];
                    SearchResult {
                        path: doc.path.clone(),
                        score: (score * 1000.0).round() / 10.0,
                        semantic_score: (semantic_score.max(0.0) * 1000.0).round() / 10.0,
                        keyword_score: (keyword_score * 1000.0).round() / 10.0,
                        header: doc.header.clone(),
                        matched_symbols,
                        matched_symbol_locations,
                    }
                },
            )
            .collect()
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }
}

// ---------------------------------------------------------------------------
// Format output
// ---------------------------------------------------------------------------

/// Format search results as text output (matching TS format).
pub fn format_search_results(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No matching files found for the given query.".to_string();
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "Top {} hybrid matches for: \"{}\"\n",
        results.len(),
        query
    ));

    for (i, r) in results.iter().enumerate() {
        lines.push(format!("{}. {} ({}% total)", i + 1, r.path, r.score));
        lines.push(format!(
            "   Semantic: {}% | Keyword: {}%",
            r.semantic_score, r.keyword_score
        ));
        if !r.header.is_empty() {
            lines.push(format!("   Header: {}", r.header));
        }
        if !r.matched_symbols.is_empty() {
            lines.push(format!(
                "   Matched symbols: {}",
                r.matched_symbols.join(", ")
            ));
        }
        if !r.matched_symbol_locations.is_empty() {
            lines.push(format!(
                "   Definition lines: {}",
                r.matched_symbol_locations.join(", ")
            ));
        }
        lines.push(String::new());
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// High-level entry point (to be wired with SharedState)
// ---------------------------------------------------------------------------

/// Run semantic code search. Caller provides the embedding function and file walker.
/// This is the main entry point that tool handlers should call.
pub async fn semantic_code_search(
    options: SemanticSearchOptions,
    embed_fn: &dyn EmbedFn,
    walk_and_index_fn: &dyn WalkAndIndexFn,
) -> Result<String> {
    let query = sanitize_query(&options.query);
    if query.is_empty() {
        return Ok("No matching files found for the given query.".to_string());
    }

    let resolved = resolve_search_options(&options);

    // Get query embedding — embed takes &[String], so convert Cow<str> to String only once.
    let query_string = query.as_ref().to_string();
    let query_vecs = embed_fn.embed(std::slice::from_ref(&query_string)).await?;
    let query_vec = query_vecs
        .into_iter()
        .next()
        .ok_or_else(|| ContextPlusError::Ollama("Empty embedding response".into()))?;

    // Build or retrieve index
    let (docs, vectors) = walk_and_index_fn.walk_and_index(&options.root_dir).await?;

    let mut index = SearchIndex::new();
    index.index_with_vectors(docs, vectors);

    let results = index.search(query.as_ref(), &query_vec, &resolved);
    Ok(format_search_results(query.as_ref(), &results))
}

// ---------------------------------------------------------------------------
// Trait bounds (for dependency injection)
// ---------------------------------------------------------------------------

/// Trait for embedding text into vectors.
pub trait EmbedFn: Send + Sync {
    fn embed(&self, texts: &[String]) -> EmbedFuture<'_>;
}

/// Trait for walking files and producing indexed documents with vectors.
pub trait WalkAndIndexFn: Send + Sync {
    fn walk_and_index(&self, root_dir: &Path) -> WalkAndIndexFuture<'_>;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::parser::hash_content;

    // -- hash_content tests --

    #[test]
    fn test_hash_content_empty() {
        assert_eq!(hash_content(""), "0");
    }

    #[test]
    fn test_hash_content_deterministic() {
        let input = "export function getUserById(id: string): User {}";
        assert_eq!(hash_content(input), hash_content(input));
    }

    #[test]
    fn test_hash_content_different_inputs() {
        assert_ne!(hash_content("foo"), hash_content("bar"));
    }

    #[test]
    fn test_hash_content_hello() {
        let result = hash_content("hello");
        assert!(!result.is_empty());
        // Verify consistency
        assert_eq!(result, hash_content("hello"));
    }

    // -- split_camel_case tests --

    #[test]
    fn test_split_camel_case_basic() {
        let result = split_camel_case("getUserById");
        assert_eq!(result, vec!["get", "user", "by", "id"]);
    }

    #[test]
    fn test_split_camel_case_snake() {
        let result = split_camel_case("get_user_by_id");
        assert_eq!(result, vec!["get", "user", "by", "id"]);
    }

    #[test]
    fn test_split_camel_case_kebab() {
        let result = split_camel_case("get-user-by-id");
        assert_eq!(result, vec!["get", "user", "by", "id"]);
    }

    #[test]
    fn test_split_camel_case_acronym() {
        let result = split_camel_case("parseHTMLDocument");
        assert_eq!(result, vec!["parse", "html", "document"]);
    }

    #[test]
    fn test_split_camel_case_filters_short() {
        let result = split_camel_case("aB");
        assert!(result.is_empty());
    }

    #[test]
    fn test_split_camel_case_path() {
        let result = split_camel_case("src/tools/semantic-search.ts");
        assert!(result.contains(&"src".to_string()));
        assert!(result.contains(&"tools".to_string()));
        assert!(result.contains(&"semantic".to_string()));
        assert!(result.contains(&"search".to_string()));
        assert!(result.contains(&"ts".to_string()));
    }

    // -- cosine tests --

    #[test]
    fn test_cosine_identical() {
        // cosine() only normalizes `a` — `b` is assumed pre-normalized (Ollama vectors)
        let norm = (1.0_f32 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt();
        let a = vec![1.0 / norm, 2.0 / norm, 3.0 / norm];
        let b = vec![1.0 / norm, 2.0 / norm, 3.0 / norm];
        let sim = cosine(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    // -- clamp01 tests --

    #[test]
    fn test_clamp01() {
        assert_eq!(clamp01(-0.5), 0.0);
        assert_eq!(clamp01(0.5), 0.5);
        assert_eq!(clamp01(1.5), 1.0);
    }

    // -- term_coverage tests --

    #[test]
    fn test_term_coverage_full() {
        let query: HashSet<String> = ["get", "user"].iter().map(|s| s.to_string()).collect();
        let doc: HashSet<String> = ["get", "user", "by", "id"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!((get_term_coverage(&query, &doc) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_term_coverage_partial() {
        let query: HashSet<String> = ["get", "user", "profile"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let doc: HashSet<String> = ["get", "user"].iter().map(|s| s.to_string()).collect();
        let coverage = get_term_coverage(&query, &doc);
        assert!((coverage - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_term_coverage_empty_query() {
        let query: HashSet<String> = HashSet::new();
        let doc: HashSet<String> = ["get"].iter().map(|s| s.to_string()).collect();
        assert_eq!(get_term_coverage(&query, &doc), 0.0);
    }

    // -- matched_symbols tests --

    #[test]
    fn test_matched_symbols() {
        let symbols = vec![
            "getUserById".to_string(),
            "deletePost".to_string(),
            "createUser".to_string(),
        ];
        let symbol_tokens: Vec<HashSet<String>> = symbols
            .iter()
            .map(|s| split_camel_case(s).into_iter().collect())
            .collect();
        let query_terms: HashSet<String> = ["user"].iter().map(|s| s.to_string()).collect();
        let matched = get_matched_symbols(&symbols, &symbol_tokens, &query_terms);
        assert!(matched.contains(&"getUserById".to_string()));
        assert!(matched.contains(&"createUser".to_string()));
        assert!(!matched.contains(&"deletePost".to_string()));
    }

    // -- keyword score tests --

    #[test]
    fn test_keyword_score_with_phrase_match() {
        let doc = SearchDocument::new(
            "src/auth.ts".to_string(),
            "auth handler".to_string(),
            vec!["authenticate".to_string()],
            vec![],
            "how does auth work".to_string(),
        );
        let query = "auth";
        let query_terms: HashSet<String> = split_camel_case(query).into_iter().collect();
        let matched = get_matched_symbols(&doc.symbols, &doc.symbol_tokens, &query_terms);
        let query_lower = query.trim().to_lowercase();
        let score = compute_keyword_score(&query_lower, &query_terms, &doc, &matched);
        assert!(score > 0.0);
    }

    // -- combined score tests --

    #[test]
    fn test_combined_score() {
        let opts = ResolvedSearchOptions {
            top_k: 5,
            semantic_weight: 0.72,
            keyword_weight: 0.28,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.1,
            require_keyword_match: false,
            require_semantic_match: false,
        };
        let combined = compute_combined_score(0.8, 0.6, &opts);
        let expected = 0.72 * 0.8 + 0.28 * 0.6;
        assert!((combined - expected).abs() < 1e-6);
    }

    // -- SearchIndex tests --

    #[test]
    fn test_search_index_empty() {
        let index = SearchIndex::new();
        assert_eq!(index.document_count(), 0);
    }

    #[test]
    fn test_search_index_basic() {
        let docs = vec![
            SearchDocument::new(
                "src/auth.ts".to_string(),
                "authentication module".to_string(),
                vec!["verifyToken".to_string()],
                vec![],
                "JWT verification".to_string(),
            ),
            SearchDocument::new(
                "src/db.ts".to_string(),
                "database connection".to_string(),
                vec!["connect".to_string()],
                vec![],
                "PostgreSQL driver".to_string(),
            ),
        ];

        let query_vec = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            Some(vec![0.9, 0.1, 0.0]), // auth: high similarity
            Some(vec![0.1, 0.9, 0.0]), // db: low similarity
        ];

        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let opts = ResolvedSearchOptions {
            top_k: 5,
            semantic_weight: 0.72,
            keyword_weight: 0.28,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
        };

        let results = index.search("auth", &query_vec, &opts);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].path, "src/auth.ts");
    }

    #[test]
    fn test_search_index_with_symbol_entries() {
        let docs = vec![SearchDocument::new(
            "src/user.ts".to_string(),
            "user service".to_string(),
            vec!["getUserById".to_string()],
            vec![SymbolSearchEntry {
                name: "getUserById".to_string(),
                kind: Some("function".to_string()),
                line: 10,
                end_line: Some(25),
                signature: Some("getUserById(id: string): User".to_string()),
            }],
            "user management".to_string(),
        )];

        let query_vec = vec![1.0, 0.0, 0.0];
        let vectors = vec![Some(vec![0.8, 0.2, 0.0])];

        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let opts = ResolvedSearchOptions {
            top_k: 5,
            semantic_weight: 0.72,
            keyword_weight: 0.28,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
        };

        let results = index.search("getUserById", &query_vec, &opts);
        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .matched_symbol_locations
                .contains(&"getUserById@L10-L25".to_string())
        );
    }

    #[test]
    fn test_search_index_filters() {
        let docs = vec![SearchDocument::new(
            "src/low.ts".to_string(),
            "low relevance".to_string(),
            vec![],
            vec![],
            "nothing useful".to_string(),
        )];

        let query_vec = vec![1.0, 0.0, 0.0];
        let vectors = vec![Some(vec![0.01, 0.99, 0.0])]; // very low similarity

        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let opts = ResolvedSearchOptions {
            top_k: 5,
            semantic_weight: 0.72,
            keyword_weight: 0.28,
            min_semantic_score: 0.5, // high threshold
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
        };

        let results = index.search("something", &query_vec, &opts);
        assert!(results.is_empty());
    }

    // -- format output tests --

    #[test]
    fn test_format_empty_results() {
        let output = format_search_results("test query", &[]);
        assert_eq!(output, "No matching files found for the given query.");
    }

    #[test]
    fn test_format_results() {
        let results = vec![SearchResult {
            path: "src/auth.ts".to_string(),
            score: 85.5,
            semantic_score: 90.0,
            keyword_score: 70.0,
            header: "auth module".to_string(),
            matched_symbols: vec!["verifyToken".to_string()],
            matched_symbol_locations: vec!["verifyToken@L10-L25".to_string()],
        }];
        let output = format_search_results("auth", &results);
        assert!(output.contains("src/auth.ts"));
        assert!(output.contains("85.5% total"));
        assert!(output.contains("Semantic: 90%"));
        assert!(output.contains("Keyword: 70%"));
        assert!(output.contains("verifyToken"));
    }

    // -- sanitize_query tests --

    #[test]
    fn test_sanitize_query_short() {
        assert_eq!(sanitize_query("hello world"), "hello world");
    }

    #[test]
    fn test_sanitize_query_trimmed() {
        assert_eq!(sanitize_query("  hello  "), "hello");
    }

    #[test]
    fn test_sanitize_query_long() {
        let long = "a".repeat(3000);
        let result = sanitize_query(&long);
        assert_eq!(result.len(), MAX_QUERY_LEN);
    }

    #[test]
    fn test_normalize_threshold_over_1() {
        assert!((normalize_threshold(Some(50.0), 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_top_k_capped() {
        assert_eq!(normalize_top_k(Some(100), 5), MAX_TOP_K);
        assert_eq!(normalize_top_k(Some(0), 5), 5); // fallback for 0
        assert_eq!(normalize_top_k(None, 5), 5);
    }
}
