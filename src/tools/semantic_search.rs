//! File-level semantic search with hybrid scoring (semantic + keyword).
//!
//! Ports the TypeScript `semantic-search.ts` logic:
//! - Builds a SearchIndex from file headers + symbols + content
//! - Uses Ollama embeddings for semantic similarity
//! - Combines semantic score with keyword coverage for hybrid ranking

use std::borrow::Cow;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use regex::Regex;
use tokio::sync::RwLock;

use crate::core::embeddings::VectorStore;
use crate::error::{ContextPlusError, Result};
use crate::tools::scoring::{
    DEFAULT_KEYWORD_WEIGHT, DEFAULT_SEMANTIC_WEIGHT, DEFAULT_TOP_K, clamp01, keyword_coverage,
    normalize_weight,
};

/// Maximum additive bonus from recency (kept small so it nudges ties, not
/// dominates relevance).
const MAX_RECENCY_BOOST: f64 = 0.05;

// ---------------------------------------------------------------------------
// ANN pre-filter constants
// ---------------------------------------------------------------------------

/// Corpus size at which `SearchIndex::search` switches from O(N) brute-force
/// cosine scan to an HNSW approximate-nearest-neighbor pre-filter. Must match
/// `HNSW_THRESHOLD` in `core/embeddings.rs` (both guard the same boundary).
const ANN_THRESHOLD: usize = 2_000;

/// How many ANN candidates to fetch per top-k result. The candidate pool is
/// `top_k * ANN_CANDIDATE_MULTIPLIER`, capped at the corpus size. Larger
/// values improve recall at the cost of more scoring work. Override at
/// runtime with `CONTEXTPLUS_ANN_CANDIDATE_MULTIPLIER`.
const ANN_CANDIDATE_MULTIPLIER: usize = 10;

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

// DEFAULT_SEMANTIC_WEIGHT, DEFAULT_KEYWORD_WEIGHT, DEFAULT_TOP_K re-exported
// from crate::tools::scoring (canonical source of truth).
const MAX_TOP_K: usize = 50;
const MAX_QUERY_LEN: usize = 2000;
const DEFAULT_MIN_COMBINED_SCORE: f64 = 0.1;
const PHRASE_BOOST: f64 = 0.15;
const TERM_COVERAGE_WEIGHT: f64 = 0.65;
const SYMBOL_COVERAGE_WEIGHT: f64 = 0.20;

/// Maximum characters of raw content to include for text file documents.
pub const MAX_TEXT_DOC_CHARS: usize = 4000;

/// Default maximum file size (bytes) for text files to be indexed.
pub const DEFAULT_MAX_EMBED_FILE_SIZE: u64 = 50 * 1024;

/// Extensions that identify text/data files eligible for semantic search indexing.
const TEXT_INDEX_EXTENSIONS: &[&str] = &[
    ".md", ".txt", ".json", ".jsonc", ".geojson", ".csv", ".tsv", ".ndjson", ".yaml", ".yml",
    ".toml", ".lock", ".env",
];

// ---------------------------------------------------------------------------
// Text file indexing helpers
// ---------------------------------------------------------------------------

/// Check if a file path has a text/data extension eligible for indexing.
pub fn is_text_index_candidate(file_path: &str) -> bool {
    let lower = file_path.to_lowercase();
    TEXT_INDEX_EXTENSIONS.iter().any(|ext| lower.ends_with(ext))
}

/// Extract a plain-text header from content: up to 2 non-empty lines, each capped at 120 chars.
pub fn extract_plain_text_header(content: &str) -> String {
    let mut header_lines: Vec<&str> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let capped = if trimmed.len() > 120 {
            &trimmed[..120]
        } else {
            trimmed
        };
        header_lines.push(capped);
        if header_lines.len() >= 2 {
            break;
        }
    }
    header_lines.join(" | ")
}

/// Read the `CONTEXTPLUS_MAX_EMBED_FILE_SIZE` env var, falling back to the default.
pub fn get_max_embed_file_size() -> u64 {
    std::env::var("CONTEXTPLUS_MAX_EMBED_FILE_SIZE")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|v| v.max(1024))
        .unwrap_or(DEFAULT_MAX_EMBED_FILE_SIZE)
}

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
    /// Globs (`src/**/*.ts`, `*.rs`) that a result path must match. Empty = no
    /// restriction. Patterns are OR'd — a path matching *any* glob passes.
    pub include_globs: Option<Vec<String>>,
    /// Globs that exclude a result path (applied after include_globs). Useful
    /// for filtering out tests, generated code, vendored deps, etc.
    pub exclude_globs: Option<Vec<String>>,
    /// Optional recency window in days. Results within the window receive a
    /// small score boost, decaying linearly with age. None = no recency tilt.
    pub recency_window_days: Option<u32>,
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
    /// Short content excerpt anchored at the first matched symbol (or the
    /// document header). Bounded to a few lines so callers can render it
    /// inline without bloating context. `None` when no useful snippet can
    /// be extracted (empty content, etc.).
    pub snippet: Option<String>,
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
    /// Compiled include globs — a result must match at least one (empty = pass-all).
    pub include_globs: Vec<Regex>,
    /// Compiled exclude globs — a result matching any of these is dropped.
    pub exclude_globs: Vec<Regex>,
    /// Recency window applied as a small additive score boost (0..MAX_RECENCY_BOOST).
    pub recency_window_days: Option<u32>,
    /// Root directory used to resolve relative `doc.path` to absolute paths
    /// (needed by recency_boost to stat the file). Empty when unknown — the
    /// recency boost simply degrades to 0 in that case.
    pub root_dir: PathBuf,
}

impl Default for ResolvedSearchOptions {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K,
            semantic_weight: DEFAULT_SEMANTIC_WEIGHT,
            keyword_weight: DEFAULT_KEYWORD_WEIGHT,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: DEFAULT_MIN_COMBINED_SCORE,
            require_keyword_match: false,
            require_semantic_match: false,
            include_globs: Vec::new(),
            exclude_globs: Vec::new(),
            recency_window_days: None,
            root_dir: PathBuf::new(),
        }
    }
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
        include_globs: compile_globs(opts.include_globs.as_deref()),
        exclude_globs: compile_globs(opts.exclude_globs.as_deref()),
        recency_window_days: opts.recency_window_days,
        root_dir: opts.root_dir.clone(),
    }
}

/// Compile a list of user-supplied globs into anchored regexes.
/// Invalid patterns are silently dropped — better to over-match than to fail
/// a search outright on a bad glob.
fn compile_globs(globs: Option<&[String]>) -> Vec<Regex> {
    let Some(globs) = globs else {
        return Vec::new();
    };
    globs
        .iter()
        .filter_map(|g| Regex::new(&glob_to_regex(g)).ok())
        .collect()
}

/// Translate a path glob (`src/**/*.ts`, `*.rs`, `tests/foo_?.rs`) into an
/// anchored regex. Recognized syntax:
///   `**`  → match any number of path components (including zero)
///   `*`   → match any character except `/`
///   `?`   → match a single non-`/` character
///   anything else is matched literally
pub fn glob_to_regex(glob: &str) -> String {
    // Iterate by char (not byte) so multi-byte UTF-8 in paths survives intact.
    let mut out = String::with_capacity(glob.len() * 2 + 4);
    out.push('^');
    let chars: Vec<char> = glob.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '*' if i + 1 < chars.len() && chars[i + 1] == '*' => {
                // ** → match zero or more path segments
                out.push_str(".*");
                i += 2;
                // swallow a trailing slash so `src/**/foo` accepts `src/foo`
                if i < chars.len() && chars[i] == '/' {
                    i += 1;
                }
            }
            '*' => {
                out.push_str("[^/]*");
                i += 1;
            }
            '?' => {
                out.push_str("[^/]");
                i += 1;
            }
            // All Rust-regex metacharacters that need escaping when matched
            // literally. Source: regex crate "Syntax" docs.
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | '[' | ']' | '\\' => {
                out.push('\\');
                out.push(c);
                i += 1;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out.push('$');
    out
}

fn path_passes_filters(path: &str, opts: &ResolvedSearchOptions) -> bool {
    if !opts.include_globs.is_empty() && !opts.include_globs.iter().any(|r| r.is_match(path)) {
        return false;
    }
    if opts.exclude_globs.iter().any(|r| r.is_match(path)) {
        return false;
    }
    true
}

/// Compute a small additive recency boost for a file based on its mtime.
/// Returns 0.0 when no boost should apply (no window configured, file
/// missing, etc.). Boost decays linearly: a file modified today gets the
/// full MAX_RECENCY_BOOST, a file at the edge of the window gets ~0.
pub fn recency_boost(path: &Path, window_days: Option<u32>) -> f64 {
    let Some(window) = window_days else {
        return 0.0;
    };
    if window == 0 {
        return 0.0;
    }
    let Ok(meta) = std::fs::metadata(path) else {
        return 0.0;
    };
    let Ok(mtime) = meta.modified() else {
        return 0.0;
    };
    let Ok(age) = mtime.elapsed() else {
        return 0.0; // mtime in the future — treat as no boost
    };
    let age_days = age.as_secs() as f64 / 86_400.0;
    let window_days = window as f64;
    if age_days >= window_days {
        return 0.0;
    }
    MAX_RECENCY_BOOST * (1.0 - age_days / window_days)
}

/// Maximum number of lines emitted in a result snippet (excluding the
/// optional `…` truncation marker). Keeps result payloads small enough
/// that callers can render many results without context blow-up.
pub const SNIPPET_MAX_LINES: usize = 6;

/// Pull a short excerpt out of `content` anchored at `line` (1-indexed).
/// Skips blank-only excerpts. When `end_line` is supplied the snippet
/// won't exceed it. Returns `None` when there's nothing meaningful to
/// surface (empty content or a line that's out of bounds).
pub fn extract_snippet(content: &str, line: u32, end_line: Option<u32>) -> Option<String> {
    if content.is_empty() {
        return None;
    }
    if line == 0 {
        return None;
    }
    let start_idx = (line as usize).saturating_sub(1);
    let lines: Vec<&str> = content.lines().collect();
    if start_idx >= lines.len() {
        return None;
    }
    let hard_end = end_line
        .map(|e| (e as usize).min(lines.len()))
        .unwrap_or(lines.len());
    let soft_end = (start_idx + SNIPPET_MAX_LINES).min(hard_end);
    let take_end = soft_end.max(start_idx + 1);
    let slice = &lines[start_idx..take_end];
    let joined: String = slice.join("\n");
    if joined.trim().is_empty() {
        return None;
    }
    if take_end < hard_end {
        Some(format!("{joined}\n…"))
    } else {
        Some(joined)
    }
}

/// Parse a `name@L<start>` or `name@L<start>-L<end>` location string back
/// into `(start, Option<end>)`. Returns `None` for malformed inputs.
pub(crate) fn parse_location_string(loc: &str) -> Option<(u32, Option<u32>)> {
    let (_name, range) = loc.rsplit_once('@')?;
    let range = range.strip_prefix('L')?;
    if let Some((s, e)) = range.split_once("-L") {
        let start: u32 = s.parse().ok()?;
        let end: u32 = e.parse().ok()?;
        Some((start, Some(end)))
    } else {
        let start: u32 = range.parse().ok()?;
        Some((start, None))
    }
}

/// Pick a snippet for `doc` using its first matched-symbol location, or
/// fall back to the document header / first content line when no symbol
/// matched.
pub(crate) fn snippet_for_doc(
    doc: &SearchDocument,
    matched_symbol_locations: &[String],
) -> Option<String> {
    if let Some(loc) = matched_symbol_locations.first()
        && let Some((start, end)) = parse_location_string(loc)
        && let Some(snippet) = extract_snippet(&doc.content, start, end)
    {
        return Some(snippet);
    }
    // Fall back to the first non-empty content line if no symbol info.
    let first_line = doc.content.lines().find(|l| !l.trim().is_empty())?;
    let trimmed: String = first_line.chars().take(160).collect();
    if trimmed.trim().is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// Cosine similarity between two f32 vectors.
/// Delegates to simsimd for SIMD-accelerated computation.
///
/// Returns 0.0 if dimensions mismatch — simsimd would otherwise read past
/// the shorter slice's end (UB in release builds).
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        debug_assert!(
            false,
            "vector dimension mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        return 0.0;
    }
    if a.iter().all(|&v| v == 0.0) {
        return 0.0;
    }
    crate::core::embeddings::cosine_similarity_simsimd(a, b) as f64
}

/// Get term coverage: fraction of query terms that appear in doc terms.
/// Delegates to `scoring::keyword_coverage` — canonical implementation lives there.
fn get_term_coverage(query_terms: &HashSet<String>, doc_terms: &HashSet<String>) -> f64 {
    keyword_coverage(query_terms, doc_terms)
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

/// Heuristic classification of a search query, used to boost matches against
/// the symbol kinds the user is most likely looking for.
///
/// The mapping is intentionally loose — the boost is small and stacks with
/// (rather than overrides) the semantic + keyword scores. Mismatches stay
/// rankable on those signals alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKind {
    /// PascalCase (`MemberRepo`, `StripeError`) → likely class/interface/type.
    Class,
    /// snake_case or camelCase identifier (`get_user`, `getUserById`) → likely function/method.
    Function,
    /// Dotted (`foo.bar.baz`), double-colon (`module::Type`), or contains '/' →
    /// likely a qualified module path.
    Path,
    /// Anything else (free-form English question, single lowercase word, etc.).
    Generic,
}

/// Multipliers applied to the keyword component when the query kind matches the
/// kind of the best-matched symbol. Small enough to nudge ties, not overrule.
const QUERY_KIND_BOOST_CLASS: f64 = 1.5;
const QUERY_KIND_BOOST_FUNCTION: f64 = 1.3;
const QUERY_KIND_BOOST_PATH: f64 = 2.0;

/// Detect the lexical shape of a query.
pub fn detect_query_kind(query: &str) -> QueryKind {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return QueryKind::Generic;
    }
    // Multi-word natural language — no shape to lean on. Anything with
    // whitespace inside it is a phrase, not an identifier.
    if trimmed.chars().any(|c| c.is_whitespace()) {
        return QueryKind::Generic;
    }
    if trimmed.contains("::") || trimmed.contains('/') || trimmed.contains('.') {
        return QueryKind::Path;
    }
    let mut chars = trimmed.chars();
    let first = chars.next().unwrap();
    let has_underscore = trimmed.contains('_');
    let has_upper_after_first = chars.clone().any(|c| c.is_ascii_uppercase());

    if first.is_ascii_uppercase() && !has_underscore {
        // Pure PascalCase or single capital letter → class-shape.
        return QueryKind::Class;
    }
    if has_underscore || has_upper_after_first {
        // snake_case or camelCase → function/method-shape.
        return QueryKind::Function;
    }
    QueryKind::Generic
}

/// Multiplier to apply to the keyword score given a query kind and the
/// kind of the best-matched symbol entry. Returns 1.0 when no boost applies.
pub fn query_kind_boost(query_kind: QueryKind, matched_kind: Option<&str>) -> f64 {
    let Some(kind) = matched_kind else {
        return 1.0;
    };
    let kind = kind.to_ascii_lowercase();
    match query_kind {
        QueryKind::Class
            if matches!(
                kind.as_str(),
                "class" | "interface" | "type" | "struct" | "enum" | "trait"
            ) =>
        {
            QUERY_KIND_BOOST_CLASS
        }
        QueryKind::Function
            if matches!(
                kind.as_str(),
                "function" | "method" | "fn" | "const" | "let" | "var"
            ) =>
        {
            QUERY_KIND_BOOST_FUNCTION
        }
        // Paths boost any symbol — the *file path* is what matters.
        QueryKind::Path => QUERY_KIND_BOOST_PATH,
        _ => 1.0,
    }
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
// ---------------------------------------------------------------------------
// CachedSearchIndex — reuse across requests
// ---------------------------------------------------------------------------

/// Cheap fingerprint used to decide whether the `SearchIndex` is still valid.
/// Computed from the walk result without any extra I/O.
///
/// Uses `std::hash::DefaultHasher` (SipHash with a fixed zero seed — deterministic
/// across processes) over each document's path + content bytes. This closes the
/// collision window that a plain `content.len()` sum left open: swap-balanced
/// edits (one file grows by N bytes while another shrinks by N) would previously
/// collide; the path+content hash will not.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IndexFingerprint {
    /// Number of documents returned by the walker.
    pub n_docs: usize,
    /// SipHash over `(path, content)` for each document, order-dependent.
    pub content_hash: u64,
}

impl IndexFingerprint {
    /// Compute a fingerprint from a slice of `SearchDocument`s.
    pub fn from_docs(docs: &[SearchDocument]) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::hash::DefaultHasher::new();
        docs.len().hash(&mut hasher);
        for d in docs {
            d.path.hash(&mut hasher);
            d.content.hash(&mut hasher);
        }
        Self {
            n_docs: docs.len(),
            content_hash: hasher.finish(),
        }
    }
}

/// A `SearchIndex` paired with the fingerprint that was current when it was built.
/// Stored in `SharedState` and reused across MCP requests when the fingerprint is unchanged.
pub struct CachedSearchIndex {
    pub index: SearchIndex,
    pub fingerprint: IndexFingerprint,
    /// Tracker generation counter value at the time this entry was built.
    ///
    /// Atomic so it can be updated on the fingerprint-hit fast path (which only
    /// holds a read-lock). Without this, a tracker-bump followed by an
    /// unchanged-content fingerprint-hit would leave `generation` stale,
    /// permanently locking the fast path out — every future request would
    /// walk + fingerprint until a full rebuild happened.
    pub generation: std::sync::atomic::AtomicU64,
    /// Number of times this cached entry has been reused (observability / tests).
    /// Wraps silently on `u64` overflow — only meaningful for monitoring deltas,
    /// not as an absolute counter.
    pub(crate) reuse_count: std::sync::atomic::AtomicU64,
}

impl CachedSearchIndex {
    fn new(index: SearchIndex, fingerprint: IndexFingerprint, generation: u64) -> Self {
        Self {
            index,
            fingerprint,
            generation: std::sync::atomic::AtomicU64::new(generation),
            reuse_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Increment and return the reuse counter.
    pub fn record_reuse(&self) -> u64 {
        self.reuse_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }
}

// ---------------------------------------------------------------------------
// SearchIndex -- indexes documents and runs hybrid search
// ---------------------------------------------------------------------------

/// In-memory search index holding documents and their embedding vectors.
/// Vectors are stored in a flat contiguous buffer for cache-friendly SIMD access.
/// For corpora larger than `ANN_THRESHOLD`, a `VectorStore` is also built so
/// `search()` can use HNSW ANN pre-filtering instead of a full cosine scan.
///
/// Memory optimisation: when `ann_store` is `Some` (corpus ≥ `ANN_THRESHOLD`),
/// `vector_buffer` is released (`capacity() == 0`) because the `VectorStore`
/// already owns an identical copy. The cosine-scoring path then retrieves
/// per-document vectors from `ann_store` via `get_vector`. Below threshold the
/// buffer is kept for the brute-force O(N) scan.
pub struct SearchIndex {
    documents: Vec<SearchDocument>,
    /// Flat buffer: `vector_buffer[i * dims .. (i+1) * dims]` is the vector for doc `i`.
    /// Docs without vectors have `has_vector[i] == false` and zeros in the buffer.
    /// **Dropped (capacity 0) when `ann_store.is_some()`** — vectors live in the
    /// store instead, avoiding the double-allocation.
    vector_buffer: Vec<f32>,
    /// Which docs have valid embedding vectors.
    has_vector: Vec<bool>,
    /// Dimensions per vector (0 if no vectors indexed yet).
    dims: usize,
    /// HNSW-backed VectorStore built when the number of embedded docs exceeds
    /// `ANN_THRESHOLD`. Maps relative file path → embedding vector.
    /// `None` when the corpus is below threshold or no vectors are available.
    ann_store: Option<VectorStore>,
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
            ann_store: None,
        }
    }

    /// Index documents using pre-computed vectors.
    /// `vectors` must be the same length as `docs`.
    /// Uses default HNSW tuning; callers needing custom tuning should use
    /// [`Self::index_with_vectors_and_tuning`].
    pub fn index_with_vectors(
        &mut self,
        docs: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
    ) {
        self.index_with_vectors_and_tuning(
            docs,
            vectors,
            crate::core::embeddings::HnswTuning::default(),
        );
    }

    /// Index documents using pre-computed vectors with explicit HNSW tuning.
    /// Use `HnswTuning::from_config(&config)` to respect env-var overrides.
    pub fn index_with_vectors_and_tuning(
        &mut self,
        docs: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
        hnsw_tuning: crate::core::embeddings::HnswTuning,
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
            match v {
                Some(vec) if vec.len() == dims => {
                    let offset = i * dims;
                    buffer[offset..offset + dims].copy_from_slice(vec);
                    has_vec.push(true);
                }
                Some(_) => {
                    // Mixed-dim vector — treat as missing rather than panicking.
                    has_vec.push(false);
                }
                None => has_vec.push(false),
            }
        }
        // Count embedded docs to decide whether to build the ANN store.
        let embedded_count = has_vec.iter().filter(|&&v| v).count();

        // Build a VectorStore (and therefore the HNSW index) when the corpus
        // is large enough for ANN pre-filtering to be worthwhile.
        let ann_store = if dims > 0 && embedded_count >= ANN_THRESHOLD {
            let mut keys = Vec::with_capacity(embedded_count);
            let mut hashes: Vec<String> = Vec::with_capacity(embedded_count);
            let mut flat: Vec<f32> = Vec::with_capacity(embedded_count * dims);
            for (i, d) in docs.iter().enumerate() {
                if has_vec[i] {
                    keys.push(d.path.clone());
                    hashes.push(String::new()); // hash unused by SearchIndex
                    let offset = i * dims;
                    flat.extend_from_slice(&buffer[offset..offset + dims]);
                }
            }
            Some(VectorStore::new_with_tuning(
                dims as u32,
                keys,
                hashes,
                flat,
                hnsw_tuning,
            ))
        } else {
            None
        };

        // When the ANN store owns the vectors, release the flat buffer to avoid
        // carrying two identical copies of the embedding data in memory.
        // The brute-force scoring path (corpus < ANN_THRESHOLD) still needs the
        // buffer, so it is only dropped when ann_store is Some.
        if ann_store.is_some() {
            buffer = Vec::new();
        }

        self.documents = docs;
        self.vector_buffer = buffer;
        self.has_vector = has_vec;
        self.dims = dims;
        self.ann_store = ann_store;
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
        let query_kind = detect_query_kind(query);

        // Precompute recency boost for every document ONCE before the parallel
        // scoring loop.  For a 5k-document index this reduces fs::metadata()
        // calls from O(N) inside rayon work-units (expensive per-thread
        // syscall) to a single sequential O(N) pass whose results are read via
        // O(1) HashMap look-ups inside par_iter.
        // Skip building the map entirely when the window is disabled.
        let recency_by_path: std::collections::HashMap<&str, f64> =
            if opts.recency_window_days.is_none_or(|w| w == 0) {
                std::collections::HashMap::new()
            } else {
                self.documents
                    .iter()
                    .map(|d| {
                        let boost =
                            recency_boost(&opts.root_dir.join(&d.path), opts.recency_window_days);
                        (d.path.as_str(), boost)
                    })
                    .collect()
            };

        // ANN pre-filter: when the corpus is large and we have an HNSW index,
        // restrict the cosine scan to a small candidate set instead of O(N).
        // Docs without embeddings bypass this filter and go through keyword scoring only.
        let ann_candidate_set: Option<std::collections::HashSet<usize>> =
            self.ann_store.as_ref().and_then(|store| {
                // Read optional runtime multiplier override.
                let multiplier = std::env::var("CONTEXTPLUS_ANN_CANDIDATE_MULTIPLIER")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(ANN_CANDIDATE_MULTIPLIER)
                    .max(1);
                let candidate_count = (opts.top_k * multiplier).min(store.count());
                if candidate_count == 0 {
                    return None;
                }
                let hits = store.find_nearest(query_vec, candidate_count);
                // Build a set of *document* indices from the path→doc lookup.
                let path_to_idx: std::collections::HashMap<&str, usize> = self
                    .documents
                    .iter()
                    .enumerate()
                    .map(|(i, d)| (d.path.as_str(), i))
                    .collect();
                let indices: std::collections::HashSet<usize> = hits
                    .iter()
                    .filter_map(|(path, _)| path_to_idx.get(path.as_str()).copied())
                    .collect();
                Some(indices)
            });

        #[allow(clippy::type_complexity)]
        let mut scored: Vec<(usize, f64, f64, f64, Vec<String>, Vec<String>)> = self
            .documents
            .par_iter()
            .enumerate()
            .filter_map(|(i, doc)| {
                if !self.has_vector[i] {
                    // No embedding: skip semantic scoring entirely, but still
                    // let docs pass to keyword scoring below (they will receive
                    // semantic_score = 0.0 which is handled by existing filters).
                    // We return None here to keep the ANN path clean; keyword-only
                    // docs are collected separately after this iterator.
                    return None;
                }
                // ANN pre-filter: skip docs not in the candidate set when ANN is active.
                if ann_candidate_set.as_ref().is_some_and(|c| !c.contains(&i)) {
                    return None;
                }
                if !path_passes_filters(&doc.path, opts) {
                    return None;
                }
                // When ann_store owns the vectors (corpus ≥ ANN_THRESHOLD) the
                // flat buffer has been dropped to save memory; retrieve via the
                // store's key→index map.  Below threshold the buffer is used
                // directly for cache-friendly sequential access.
                let vec_slice: &[f32] = if let Some(store) = self.ann_store.as_ref() {
                    match store.get_vector(&doc.path) {
                        Some(v) => v,
                        None => {
                            // Should be unreachable for well-formed input: every
                            // has_vector=true doc was pushed into ann_store by path
                            // at index_with_vectors time. The only way this fires
                            // in practice is a duplicate path that got overwritten
                            // during VectorStore construction — a latent bug at a
                            // different layer. Log so the silent-drop is visible.
                            tracing::warn!(
                                path = %doc.path,
                                "ann_store missing vector for has_vector=true doc — excluding from results"
                            );
                            return None;
                        }
                    }
                } else {
                    let offset = i * self.dims;
                    &self.vector_buffer[offset..offset + self.dims]
                };
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

                let raw_keyword_score =
                    compute_keyword_score(&query_lower, &query_terms, doc, &matched_symbols);
                // Apply query-kind boost using the kind of the best-matched symbol entry
                // (first matched_entries hit; falls back to no boost when nothing matched).
                let best_kind = matched_entries.first().and_then(|e| e.kind.as_deref());
                let keyword_score =
                    clamp01(raw_keyword_score * query_kind_boost(query_kind, best_kind));
                let base_combined = compute_combined_score(semantic_score, keyword_score, opts);
                // Recency: small additive nudge so freshly-touched files break ties upward.
                // Value was precomputed into `recency_by_path` before par_iter — O(1) lookup.
                let recency = recency_by_path
                    .get(doc.path.as_str())
                    .copied()
                    .unwrap_or(0.0);
                let combined_score = clamp01(base_combined + recency);

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

        // Keyword-only pass for docs that have NO embedding vector.
        // These are never reached by the ANN path (which only covers embedded docs),
        // so we score them separately and merge into `scored` before the final sort.
        // This preserves the fallback guarantee: a doc with no embedding but a strong
        // keyword match can still surface via `require_keyword_match`-compatible queries.
        #[allow(clippy::type_complexity)]
        let keyword_only: Vec<(usize, f64, f64, f64, Vec<String>, Vec<String>)> = self
            .documents
            .par_iter()
            .enumerate()
            .filter_map(|(i, doc)| {
                if self.has_vector[i] {
                    return None; // already handled above
                }
                if !path_passes_filters(&doc.path, opts) {
                    return None;
                }
                // Semantic score is 0 for docs with no embedding.
                let semantic_score: f64 = 0.0;

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

                let raw_keyword_score =
                    compute_keyword_score(&query_lower, &query_terms, doc, &matched_symbols);
                let best_kind = matched_entries.first().and_then(|e| e.kind.as_deref());
                let keyword_score =
                    clamp01(raw_keyword_score * query_kind_boost(query_kind, best_kind));
                // Zero-score pollution guard: a doc with no embedding and no keyword
                // match contributes nothing — don't surface it just because the
                // default `min_combined_score = 0.0` lets zero through.
                if keyword_score <= 0.0 {
                    return None;
                }
                let base_combined = compute_combined_score(semantic_score, keyword_score, opts);
                let recency = recency_by_path
                    .get(doc.path.as_str())
                    .copied()
                    .unwrap_or(0.0);
                let combined_score = clamp01(base_combined + recency);

                if opts.require_semantic_match {
                    return None; // no embedding → semantic_score == 0
                }
                if opts.require_keyword_match && keyword_score <= 0.0 {
                    return None;
                }
                if opts.min_semantic_score > 0.0 {
                    return None; // semantic_score == 0.0 < any positive threshold
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

        scored.extend(keyword_only);

        // Partial sort: O(N) partition to top_k, then sort only the small slice.
        let k = opts.top_k.min(scored.len());
        if k == 0 {
            return vec![];
        }
        // Partition and sort use the SAME 3-key comparator. Using only
        // `combined_score` for the partition is unsound: when combined_score
        // ties exist at position k, `select_nth_unstable` can evict the
        // tiebreaker-winner, silently changing the final top-k.
        if k < scored.len() {
            scored.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal))
                    .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
            });
            scored.truncate(k);
        }
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
                    let snippet = snippet_for_doc(doc, &matched_symbol_locations);
                    SearchResult {
                        path: doc.path.clone(),
                        score: (score * 1000.0).round() / 10.0,
                        semantic_score: (semantic_score.max(0.0) * 1000.0).round() / 10.0,
                        keyword_score: (keyword_score * 1000.0).round() / 10.0,
                        header: doc.header.clone(),
                        matched_symbols,
                        matched_symbol_locations,
                        snippet,
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
    format_search_results_with_freshness(query, results, None)
}

/// Format search results with an optional cache-freshness banner. The
/// banner reports total indexed documents so the caller can sanity-check
/// that the search ran against a populated index.
pub fn format_search_results_with_freshness(
    query: &str,
    results: &[SearchResult],
    indexed_documents: Option<usize>,
) -> String {
    if results.is_empty() {
        return "No matching files found for the given query.".to_string();
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "Top {} hybrid matches for: \"{}\"\n",
        results.len(),
        query
    ));
    if let Some(count) = indexed_documents {
        lines.push(format!("Index: {count} document(s)\n"));
    }

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
        if let Some(snippet) = &r.snippet {
            lines.push("   Snippet:".to_string());
            for snippet_line in snippet.lines() {
                lines.push(format!("     {snippet_line}"));
            }
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
///
/// `index_cache` is optional: when `Some`, the assembled `SearchIndex` is cached across
/// requests and only rebuilt when needed. The cache uses two validity checks in order:
///
/// 1. **Generation-based (fast path):** if `cache_generation` is `Some` and the current
///    generation counter matches `CachedSearchIndex::generation`, the tracker has reported
///    no file changes since the last build — the filesystem walk is **skipped entirely**.
/// 2. **Fingerprint-based (fallback):** if the generation differs (tracker fired) *or*
///    the tracker is disabled (`cache_generation` is `None`), we walk + compute a content
///    fingerprint. If that matches the cached fingerprint the index is still reused; only
///    a true content mismatch triggers a full rebuild.
///
/// Concurrent callers that race on a stale cache use double-check locking so at most one
/// rebuild occurs.
pub async fn semantic_code_search(
    options: SemanticSearchOptions,
    embed_fn: &dyn EmbedFn,
    walk_and_index_fn: &dyn WalkAndIndexFn,
    index_cache: Option<&RwLock<Option<Arc<CachedSearchIndex>>>>,
    cache_generation: Option<&Arc<std::sync::atomic::AtomicU64>>,
) -> Result<String> {
    let query = sanitize_query(&options.query);
    if query.is_empty() {
        return Ok("No matching files found for the given query.".to_string());
    }

    let resolved = resolve_search_options(&options);

    // Embed query first (independent of the index).
    let query_string = query.as_ref().to_string();
    let query_vecs = embed_fn.embed(std::slice::from_ref(&query_string)).await?;
    let query_vec = query_vecs
        .into_iter()
        .next()
        .ok_or_else(|| ContextPlusError::Ollama("Empty embedding response".into()))?;

    // Obtain the SearchIndex — from cache if available and still fresh, otherwise rebuild.
    let cached_arc: Arc<CachedSearchIndex> = match index_cache {
        None => {
            // No cache slot provided — always rebuild (unit-test / legacy path).
            let current_gen = cache_generation
                .map(|g| g.load(std::sync::atomic::Ordering::Acquire))
                .unwrap_or(0);
            let (docs, vectors) = walk_and_index_fn.walk_and_index(&options.root_dir).await?;
            let fp = IndexFingerprint::from_docs(&docs);
            let mut idx = SearchIndex::new();
            idx.index_with_vectors_and_tuning(
                docs,
                vectors,
                crate::core::embeddings::HnswTuning::global(),
            );
            Arc::new(CachedSearchIndex::new(idx, fp, current_gen))
        }
        Some(lock) => {
            // Snapshot the current tracker generation before any locking.
            let current_gen = cache_generation
                .map(|g| g.load(std::sync::atomic::Ordering::Acquire))
                .unwrap_or(0);

            // Fast path: read-lock only. If the tracker generation matches we can
            // skip the filesystem walk entirely — the tracker guarantees no file
            // changes have occurred since the cache was last built.
            if cache_generation.is_some() {
                let guard = lock.read().await;
                if let Some(ref cached) = *guard
                    && cached.generation.load(std::sync::atomic::Ordering::Acquire) == current_gen
                {
                    let reuses = cached.record_reuse();
                    tracing::debug!(
                        reuses,
                        generation = current_gen,
                        "SearchIndex generation hit — skipping walk"
                    );
                    let results = cached.index.search(query.as_ref(), &query_vec, &resolved);
                    return Ok(format_search_results_with_freshness(
                        query.as_ref(),
                        &results,
                        Some(cached.index.document_count()),
                    ));
                }
                // Generation mismatch (or no cache yet) — fall through to walk + fingerprint.
            }

            // Walk the filesystem and compute fingerprint (tracker-off fallback or
            // generation mismatch meaning the tracker saw a change).
            let (docs, vectors) = walk_and_index_fn.walk_and_index(&options.root_dir).await?;
            let fp = IndexFingerprint::from_docs(&docs);

            {
                let guard = lock.read().await;
                if let Some(ref cached) = *guard
                    && cached.fingerprint == fp
                {
                    let reuses = cached.record_reuse();
                    // Content unchanged even though tracker fired — update the cached
                    // generation so the next request takes the fast path without
                    // walking. `AtomicU64::store` is safe under a read-lock because
                    // it's an atomic operation on a `&self` receiver. Without this,
                    // a tracker-bump → fingerprint-hit cycle would permanently lock
                    // the fast path out.
                    cached
                        .generation
                        .store(current_gen, std::sync::atomic::Ordering::Release);
                    tracing::debug!(
                        reuses,
                        generation = current_gen,
                        "SearchIndex fingerprint hit — reusing index, updated generation"
                    );
                    let results = cached.index.search(query.as_ref(), &query_vec, &resolved);
                    return Ok(format_search_results_with_freshness(
                        query.as_ref(),
                        &results,
                        Some(cached.index.document_count()),
                    ));
                }
            }

            // Slow path: acquire write-lock, double-check, then rebuild.
            let mut guard = lock.write().await;
            // Another task may have rebuilt while we waited for the write-lock.
            if let Some(ref cached) = *guard
                && cached.fingerprint == fp
            {
                let reuses = cached.record_reuse();
                // Same fix as the read-lock path: bring the cached generation
                // forward so the next request takes the fast path.
                cached
                    .generation
                    .store(current_gen, std::sync::atomic::Ordering::Release);
                tracing::debug!(
                    reuses,
                    generation = current_gen,
                    "SearchIndex cache hit (post-write-lock) — reusing index, updated generation"
                );
                let results = cached.index.search(query.as_ref(), &query_vec, &resolved);
                return Ok(format_search_results_with_freshness(
                    query.as_ref(),
                    &results,
                    Some(cached.index.document_count()),
                ));
            }

            tracing::info!(
                n_docs = fp.n_docs,
                "SearchIndex cache miss — rebuilding index"
            );
            let mut idx = SearchIndex::new();
            idx.index_with_vectors_and_tuning(
                docs,
                vectors,
                crate::core::embeddings::HnswTuning::global(),
            );
            let arc = Arc::new(CachedSearchIndex::new(idx, fp, current_gen));
            *guard = Some(Arc::clone(&arc));
            arc
        }
    };

    let results = cached_arc
        .index
        .search(query.as_ref(), &query_vec, &resolved);
    Ok(format_search_results_with_freshness(
        query.as_ref(),
        &results,
        Some(cached_arc.index.document_count()),
    ))
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
        assert_eq!(hash_content(""), "cbf29ce484222325");
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
        };

        let results = index.search("something", &query_vec, &opts);
        assert!(results.is_empty());
    }

    // -- recency precompute path (200-doc smoke test) --

    #[test]
    fn test_search_recency_precompute_200_docs() {
        // Build 200 documents so that the precomputed-recency code path is
        // exercised at a non-trivial scale.  Correctness is covered by the
        // targeted recency_boost_* tests; this test verifies the loop
        // completes without panicking and returns sensible results.
        let n = 200usize;
        let docs: Vec<SearchDocument> = (0..n)
            .map(|i| {
                SearchDocument::new(
                    format!("src/module_{i}.ts"),
                    format!("module {i}"),
                    vec![format!("fn_{i}")],
                    vec![],
                    format!("content for module {i}"),
                )
            })
            .collect();

        // Vectors diverge slightly so cosine similarity spreads the scores.
        let query_vec = vec![1.0_f32, 0.0, 0.0];
        let vectors: Vec<Option<Vec<f32>>> = (0..n)
            .map(|i| {
                let x = 1.0 - i as f32 * 0.004; // 1.0 → 0.204
                let y = (1.0_f32 - x * x).max(0.0).sqrt();
                Some(vec![x, y, 0.0])
            })
            .collect();

        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let opts = ResolvedSearchOptions {
            top_k: 10,
            semantic_weight: 0.72,
            keyword_weight: 0.28,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
            recency_window_days: Some(7), // exercise the precompute branch
            ..Default::default()
        };

        let results = index.search("module", &query_vec, &opts);
        // We asked for top_k=10 and have 200 docs — must get exactly 10 back.
        assert_eq!(results.len(), 10);
        // Scores must be in descending order.
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_search_partial_sort_top_k_ordering_100_docs() {
        // Verify that with 100+ candidates the partial-sort path returns exactly
        // top_k results in strictly descending combined-score order, and that the
        // scores match a naïve full-sort of all candidates.
        let n = 100usize;
        let query_vec = vec![1.0_f32, 0.0, 0.0];

        let docs: Vec<SearchDocument> = (0..n)
            .map(|i| {
                SearchDocument::new(
                    format!("src/file_{i}.ts"),
                    format!("file {i}"),
                    vec![format!("sym_{i}")],
                    vec![],
                    format!("content {i}"),
                )
            })
            .collect();

        // Each document gets a unique cosine similarity that decreases with index,
        // so the expected rank order is doc 0 > doc 1 > … > doc 99.
        let vectors: Vec<Option<Vec<f32>>> = (0..n)
            .map(|i| {
                let x = 1.0 - i as f32 * 0.009; // 1.0 → 0.109
                let y = (1.0_f32 - x * x).max(0.0).sqrt();
                Some(vec![x, y, 0.0])
            })
            .collect();

        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let top_k = 10;
        let opts = ResolvedSearchOptions {
            top_k,
            semantic_weight: 1.0,
            keyword_weight: 0.0,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
            ..Default::default()
        };

        let results = index.search("file", &query_vec, &opts);

        // Must return exactly top_k results.
        assert_eq!(
            results.len(),
            top_k,
            "expected exactly top_k={top_k} results"
        );

        // Results must be in non-increasing score order.
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "out-of-order: {} >= {} violated",
                w[0].score,
                w[1].score
            );
        }

        // The top result must be file_0 (highest cosine similarity).
        assert_eq!(results[0].path, "src/file_0.ts");

        // The 10th result must be file_9 (10th highest similarity).
        assert_eq!(results[top_k - 1].path, "src/file_9.ts");
    }

    #[test]
    fn test_search_partial_sort_respects_tiebreakers() {
        // Regression guard for the partition-vs-sort comparator mismatch bug:
        // when two docs tie on combined_score, the 3-key comparator must break
        // the tie consistently in both `select_nth_unstable_by` and `sort_by`
        // so the keyword-score winner is retained in top-k.
        //
        // Setup: N+1 docs all with the same cosine similarity (so combined_score
        // ties). One doc contains the query keyword; the others don't. With
        // `top_k = 1`, the keyword-matching doc must win.
        let n = 20usize;
        let query_vec = vec![1.0_f32, 0.0, 0.0];

        let mut docs: Vec<SearchDocument> = (0..n)
            .map(|i| {
                SearchDocument::new(
                    format!("src/noise_{i}.ts"),
                    format!("noise {i}"),
                    vec![format!("noise_sym_{i}")],
                    vec![],
                    format!("unrelated content {i}"),
                )
            })
            .collect();
        // The one doc whose symbol matches the query keyword.
        docs.push(SearchDocument::new(
            "src/target.ts".to_string(),
            "target file".to_string(),
            vec!["findTarget".to_string()],
            vec![],
            "findTarget implementation".to_string(),
        ));

        // All docs get identical cosine similarity — forces combined_score ties.
        let vectors: Vec<Option<Vec<f32>>> = (0..=n).map(|_| Some(vec![1.0, 0.0, 0.0])).collect();

        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let opts = ResolvedSearchOptions {
            top_k: 1,
            semantic_weight: 0.5,
            keyword_weight: 0.5,
            ..Default::default()
        };

        let results = index.search("findTarget", &query_vec, &opts);

        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].path, "src/target.ts",
            "keyword-score tiebreaker must retain the matching doc across partition + sort"
        );
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
            snippet: None,
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

    // -- query_kind detection --

    #[test]
    fn detect_pascal_case_as_class() {
        assert_eq!(detect_query_kind("MemberRepo"), QueryKind::Class);
        assert_eq!(detect_query_kind("StripeError"), QueryKind::Class);
        assert_eq!(detect_query_kind("X"), QueryKind::Class);
    }

    #[test]
    fn detect_camel_or_snake_as_function() {
        assert_eq!(detect_query_kind("getUserById"), QueryKind::Function);
        assert_eq!(detect_query_kind("get_user_by_id"), QueryKind::Function);
        assert_eq!(detect_query_kind("doStuff"), QueryKind::Function);
    }

    #[test]
    fn detect_dotted_or_path_as_path() {
        assert_eq!(detect_query_kind("foo.bar.baz"), QueryKind::Path);
        assert_eq!(detect_query_kind("module::Type"), QueryKind::Path);
        assert_eq!(detect_query_kind("src/auth/login.ts"), QueryKind::Path);
    }

    #[test]
    fn detect_natural_language_as_generic() {
        assert_eq!(detect_query_kind("how does login work"), QueryKind::Generic);
        assert_eq!(detect_query_kind("user"), QueryKind::Generic);
        assert_eq!(detect_query_kind(""), QueryKind::Generic);
    }

    #[test]
    fn detect_two_word_phrases_as_generic() {
        // Pre-fix these were classified as Function/Class because the
        // whitespace-count guard required >=2 spaces (3+ words).
        assert_eq!(detect_query_kind("doStuff more"), QueryKind::Generic);
        assert_eq!(detect_query_kind("parseHTML config"), QueryKind::Generic);
        assert_eq!(detect_query_kind("MemberRepo find"), QueryKind::Generic);
    }

    // -- query_kind_boost mapping --

    #[test]
    fn boost_class_query_against_class_symbol() {
        let boost = query_kind_boost(QueryKind::Class, Some("class"));
        assert!((boost - QUERY_KIND_BOOST_CLASS).abs() < 1e-9);
    }

    #[test]
    fn boost_function_query_against_method_symbol() {
        let boost = query_kind_boost(QueryKind::Function, Some("method"));
        assert!((boost - QUERY_KIND_BOOST_FUNCTION).abs() < 1e-9);
    }

    #[test]
    fn boost_path_query_always_applies() {
        // Paths describe file location — any matched symbol is fine.
        assert!(
            (query_kind_boost(QueryKind::Path, Some("function")) - QUERY_KIND_BOOST_PATH).abs()
                < 1e-9
        );
        assert!(
            (query_kind_boost(QueryKind::Path, Some("class")) - QUERY_KIND_BOOST_PATH).abs() < 1e-9
        );
    }

    #[test]
    fn boost_returns_one_when_kinds_misalign() {
        assert_eq!(query_kind_boost(QueryKind::Class, Some("function")), 1.0);
        assert_eq!(query_kind_boost(QueryKind::Function, Some("class")), 1.0);
        assert_eq!(query_kind_boost(QueryKind::Generic, Some("class")), 1.0);
        assert_eq!(query_kind_boost(QueryKind::Class, None), 1.0);
    }

    // -- is_text_index_candidate tests --

    #[test]
    fn test_text_index_candidate_markdown() {
        assert!(is_text_index_candidate("README.md"));
        assert!(is_text_index_candidate("docs/guide.MD"));
    }

    #[test]
    fn test_text_index_candidate_json_yaml_toml() {
        assert!(is_text_index_candidate("package.json"));
        assert!(is_text_index_candidate("config.yaml"));
        assert!(is_text_index_candidate("settings.yml"));
        assert!(is_text_index_candidate("Cargo.toml"));
    }

    #[test]
    fn test_text_index_candidate_data_formats() {
        assert!(is_text_index_candidate("data.csv"));
        assert!(is_text_index_candidate("data.tsv"));
        assert!(is_text_index_candidate("stream.ndjson"));
        assert!(is_text_index_candidate("config.jsonc"));
        assert!(is_text_index_candidate("map.geojson"));
    }

    #[test]
    fn test_text_index_candidate_special() {
        assert!(is_text_index_candidate("yarn.lock"));
        assert!(is_text_index_candidate(".env"));
        assert!(is_text_index_candidate("notes.txt"));
    }

    #[test]
    fn test_text_index_candidate_code_files_rejected() {
        assert!(!is_text_index_candidate("main.rs"));
        assert!(!is_text_index_candidate("app.ts"));
        assert!(!is_text_index_candidate("index.js"));
        assert!(!is_text_index_candidate("lib.py"));
        assert!(!is_text_index_candidate("main.go"));
    }

    // -- extract_plain_text_header tests --

    #[test]
    fn test_extract_plain_text_header_basic() {
        let content = "# My Title\nSome description\n\nMore content";
        let header = extract_plain_text_header(content);
        assert_eq!(header, "# My Title | Some description");
    }

    #[test]
    fn test_extract_plain_text_header_skips_empty_lines() {
        let content = "\n\n  \nFirst line\n\nSecond line";
        let header = extract_plain_text_header(content);
        assert_eq!(header, "First line | Second line");
    }

    #[test]
    fn test_extract_plain_text_header_caps_line_length() {
        let long_line = "x".repeat(200);
        let content = format!("{}\nshort", long_line);
        let header = extract_plain_text_header(&content);
        assert!(header.starts_with(&"x".repeat(120)));
        assert!(header.contains(" | short"));
    }

    #[test]
    fn test_extract_plain_text_header_single_line() {
        let content = "Only one meaningful line";
        let header = extract_plain_text_header(content);
        assert_eq!(header, "Only one meaningful line");
    }

    #[test]
    fn test_extract_plain_text_header_empty() {
        let header = extract_plain_text_header("");
        assert_eq!(header, "");
    }

    // -- MAX_TEXT_DOC_CHARS content cap test --

    #[test]
    fn test_text_content_cap() {
        let content: String = "a".repeat(5000);
        let truncated: String = content.chars().take(MAX_TEXT_DOC_CHARS).collect();
        assert_eq!(truncated.len(), MAX_TEXT_DOC_CHARS);
    }

    // -- glob_to_regex tests --

    fn glob_matches(glob: &str, path: &str) -> bool {
        Regex::new(&glob_to_regex(glob)).unwrap().is_match(path)
    }

    #[test]
    fn glob_double_star_crosses_components() {
        assert!(glob_matches("src/**/*.ts", "src/auth/login.ts"));
        assert!(glob_matches("src/**/*.ts", "src/auth/handlers/login.ts"));
        assert!(glob_matches("src/**/foo.ts", "src/foo.ts"));
        assert!(!glob_matches("src/**/*.ts", "lib/auth.ts"));
    }

    #[test]
    fn glob_single_star_does_not_cross_separator() {
        assert!(glob_matches("*.rs", "main.rs"));
        assert!(!glob_matches("*.rs", "src/main.rs"));
    }

    #[test]
    fn glob_question_matches_single_char() {
        assert!(glob_matches("foo_?.rs", "foo_a.rs"));
        assert!(!glob_matches("foo_?.rs", "foo_ab.rs"));
        assert!(!glob_matches("foo_?.rs", "foo_/b.rs"));
    }

    #[test]
    fn glob_escapes_regex_metacharacters() {
        assert!(glob_matches("file.rs", "file.rs"));
        assert!(!glob_matches("file.rs", "fileXrs"));
    }

    #[test]
    fn glob_handles_multibyte_utf8_paths() {
        // Pre-fix this iterated bytes and split UTF-8 sequences mid-codepoint.
        assert!(glob_matches("**/café/**", "src/café/menu.ts"));
        assert!(glob_matches("docs/中文.md", "docs/中文.md"));
        assert!(!glob_matches("docs/中文.md", "docs/eng.md"));
    }

    #[test]
    fn glob_escapes_brackets_and_braces_literally() {
        assert!(glob_matches("file[1].rs", "file[1].rs"));
        assert!(!glob_matches("file[1].rs", "file1.rs"));
        assert!(glob_matches("a{b}c", "a{b}c"));
    }

    // -- path_passes_filters tests --

    fn opts_with_globs(include: &[&str], exclude: &[&str]) -> ResolvedSearchOptions {
        ResolvedSearchOptions {
            include_globs: include
                .iter()
                .map(|g| Regex::new(&glob_to_regex(g)).unwrap())
                .collect(),
            exclude_globs: exclude
                .iter()
                .map(|g| Regex::new(&glob_to_regex(g)).unwrap())
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn path_filters_empty_globs_passes_all() {
        let opts = opts_with_globs(&[], &[]);
        assert!(path_passes_filters("anywhere/foo.rs", &opts));
        assert!(path_passes_filters("vendor/dep.ts", &opts));
    }

    #[test]
    fn path_filters_include_only_keeps_matches() {
        let opts = opts_with_globs(&["src/**/*.rs"], &[]);
        assert!(path_passes_filters("src/foo.rs", &opts));
        assert!(path_passes_filters("src/sub/bar.rs", &opts));
        assert!(!path_passes_filters("tests/main.rs", &opts));
    }

    #[test]
    fn path_filters_exclude_overrides_include() {
        let opts = opts_with_globs(&["src/**/*.rs"], &["**/generated/*.rs"]);
        assert!(path_passes_filters("src/foo.rs", &opts));
        assert!(!path_passes_filters("src/generated/proto.rs", &opts));
    }

    #[test]
    fn path_filters_exclude_only_drops_matches() {
        let opts = opts_with_globs(&[], &["target/**"]);
        assert!(path_passes_filters("src/main.rs", &opts));
        assert!(!path_passes_filters("target/debug/foo", &opts));
    }

    // -- recency_boost tests --

    #[test]
    fn recency_boost_no_window_returns_zero() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        assert_eq!(recency_boost(tmp.path(), None), 0.0);
    }

    #[test]
    fn recency_boost_zero_window_returns_zero() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        assert_eq!(recency_boost(tmp.path(), Some(0)), 0.0);
    }

    #[test]
    fn recency_boost_missing_file_returns_zero() {
        let path = std::path::Path::new("/nonexistent/path/that/should/not/exist.rs");
        assert_eq!(recency_boost(path, Some(7)), 0.0);
    }

    #[test]
    fn recency_boost_fresh_file_near_max() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let boost = recency_boost(tmp.path(), Some(7));
        assert!(boost > 0.0);
        assert!(boost <= MAX_RECENCY_BOOST);
        // a freshly-touched file should be very close to MAX (within a few seconds)
        assert!(boost > MAX_RECENCY_BOOST * 0.99);
    }

    // -- parse_location_string tests --

    #[test]
    fn parse_location_string_with_range() {
        assert_eq!(
            parse_location_string("getUserById@L10-L25"),
            Some((10, Some(25)))
        );
    }

    #[test]
    fn parse_location_string_without_range() {
        assert_eq!(parse_location_string("foo@L42"), Some((42, None)));
    }

    #[test]
    fn parse_location_string_handles_at_in_name() {
        // names containing '@' use the LAST '@' as the separator
        assert_eq!(parse_location_string("ns@inner@L1-L2"), Some((1, Some(2))));
    }

    #[test]
    fn parse_location_string_rejects_garbage() {
        assert_eq!(parse_location_string(""), None);
        assert_eq!(parse_location_string("nothing"), None);
        assert_eq!(parse_location_string("foo@notaline"), None);
        assert_eq!(parse_location_string("foo@L"), None);
    }

    // -- extract_snippet tests --

    #[test]
    fn extract_snippet_uses_line_range() {
        let content = "line1\nline2\nline3\nline4\nline5\n";
        let s = extract_snippet(content, 2, Some(4)).unwrap();
        assert!(s.starts_with("line2\n"));
        assert!(s.contains("line3"));
        assert!(s.contains("line4"));
        // start..end was 3 lines so SNIPPET_MAX_LINES isn't the cap
        assert!(!s.contains('…'));
    }

    #[test]
    fn extract_snippet_truncates_at_max_lines() {
        let content: String = (1..=20).map(|n| format!("line{n}\n")).collect();
        let s = extract_snippet(&content, 1, Some(20)).unwrap();
        assert!(s.contains("line1"));
        assert!(s.contains("line6"));
        assert!(s.ends_with("…"));
    }

    #[test]
    fn extract_snippet_returns_none_for_empty_content() {
        assert!(extract_snippet("", 1, None).is_none());
    }

    #[test]
    fn extract_snippet_returns_none_for_out_of_bounds_line() {
        assert!(extract_snippet("only\none\nline\n", 99, None).is_none());
    }

    #[test]
    fn extract_snippet_returns_none_for_line_zero() {
        assert!(extract_snippet("foo\nbar\n", 0, None).is_none());
    }

    // -- snippet_for_doc tests --

    fn doc_with(content: &str, symbols: Vec<SymbolSearchEntry>) -> SearchDocument {
        SearchDocument::new(
            "src/foo.ts".to_string(),
            "header".to_string(),
            symbols.iter().map(|s| s.name.clone()).collect(),
            symbols,
            content.to_string(),
        )
    }

    #[test]
    fn snippet_for_doc_uses_matched_location() {
        let doc = doc_with("a\nb\nc\nd\ne\n", vec![]);
        let snippet = snippet_for_doc(&doc, &["foo@L2-L4".to_string()]).unwrap();
        assert!(snippet.contains('b'));
        assert!(snippet.contains('d'));
    }

    #[test]
    fn snippet_for_doc_falls_back_to_first_line_when_no_locations() {
        let doc = doc_with("\n\nfirst real line\nmore\n", vec![]);
        let snippet = snippet_for_doc(&doc, &[]).unwrap();
        assert_eq!(snippet, "first real line");
    }

    #[test]
    fn snippet_for_doc_returns_none_for_empty_content() {
        let doc = doc_with("", vec![]);
        assert!(snippet_for_doc(&doc, &[]).is_none());
    }

    // -- format_search_results_with_freshness tests --

    #[test]
    fn format_results_includes_freshness_banner() {
        let results = vec![SearchResult {
            path: "src/x.ts".to_string(),
            score: 50.0,
            semantic_score: 60.0,
            keyword_score: 40.0,
            header: String::new(),
            matched_symbols: vec![],
            matched_symbol_locations: vec![],
            snippet: None,
        }];
        let out = format_search_results_with_freshness("q", &results, Some(123));
        assert!(out.contains("Index: 123 document(s)"));
    }

    #[test]
    fn format_results_renders_snippet_indented() {
        let results = vec![SearchResult {
            path: "src/x.ts".to_string(),
            score: 50.0,
            semantic_score: 60.0,
            keyword_score: 40.0,
            header: String::new(),
            matched_symbols: vec![],
            matched_symbol_locations: vec![],
            snippet: Some("fn foo() {\n  bar()\n}".to_string()),
        }];
        let out = format_search_results_with_freshness("q", &results, None);
        assert!(out.contains("Snippet:"));
        assert!(out.contains("     fn foo() {"));
        assert!(out.contains("       bar()"));
    }

    #[test]
    fn format_results_omits_freshness_when_none() {
        let results = vec![SearchResult {
            path: "src/x.ts".to_string(),
            score: 50.0,
            semantic_score: 60.0,
            keyword_score: 40.0,
            header: String::new(),
            matched_symbols: vec![],
            matched_symbol_locations: vec![],
            snippet: None,
        }];
        let out = format_search_results_with_freshness("q", &results, None);
        assert!(!out.contains("Index:"));
    }

    // -- ANN pre-filter tests --

    /// Helper: build a unit vector in R^4 with x≈1 and small perturbation.
    fn unit_vec(x: f32, y: f32) -> Vec<f32> {
        let norm = (x * x + y * y).sqrt();
        vec![x / norm, y / norm, 0.0, 0.0]
    }

    /// Build `n` SearchDocuments with synthetic embeddings in R^4.
    fn make_ann_corpus(n: usize) -> (Vec<SearchDocument>, Vec<Option<Vec<f32>>>) {
        let docs: Vec<SearchDocument> = (0..n)
            .map(|i| {
                SearchDocument::new(
                    format!("src/file_{i}.ts"),
                    format!("file {i}"),
                    vec![format!("sym_{i}")],
                    vec![],
                    format!("content {i}"),
                )
            })
            .collect();
        // Each doc gets a unique direction: higher index → slightly more orthogonal.
        let vectors: Vec<Option<Vec<f32>>> = (0..n)
            .map(|i| {
                let x = 1.0_f32 - i as f32 * (0.9 / n as f32);
                let y = (1.0_f32 - x * x).max(0.0).sqrt();
                Some(unit_vec(x, y))
            })
            .collect();
        (docs, vectors)
    }

    #[test]
    fn test_ann_path_triggers_above_threshold() {
        // Build a corpus just above the ANN threshold.
        let n = ANN_THRESHOLD + 50;
        let (docs, vectors) = make_ann_corpus(n);
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        // The ANN store must have been built.
        assert!(
            index.ann_store.is_some(),
            "VectorStore should be Some for corpus size {n} >= ANN_THRESHOLD {ANN_THRESHOLD}"
        );

        // Query pointing at doc 0's direction: top result must be file_0.
        let query_vec = unit_vec(1.0, 0.001);
        let opts = ResolvedSearchOptions {
            top_k: 5,
            semantic_weight: 1.0,
            keyword_weight: 0.0,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
            ..Default::default()
        };

        let results = index.search("file", &query_vec, &opts);
        assert_eq!(results.len(), 5, "should return top_k=5 results");
        assert_eq!(
            results[0].path, "src/file_0.ts",
            "ANN path must return doc_0 as the nearest neighbour"
        );
    }

    #[test]
    fn test_brute_force_path_below_threshold() {
        // Corpus below ANN_THRESHOLD → ann_store must be None.
        let n = ANN_THRESHOLD - 1;
        let (docs, vectors) = make_ann_corpus(n);
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        assert!(
            index.ann_store.is_none(),
            "VectorStore should be None for corpus size {n} < ANN_THRESHOLD {ANN_THRESHOLD}"
        );

        // Brute-force path must still return correct top-k.
        let query_vec = unit_vec(1.0, 0.001);
        let opts = ResolvedSearchOptions {
            top_k: 3,
            semantic_weight: 1.0,
            keyword_weight: 0.0,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
            ..Default::default()
        };

        let results = index.search("file", &query_vec, &opts);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].path, "src/file_0.ts");
    }

    #[test]
    fn test_no_embedding_doc_surfaces_via_keyword() {
        // Mix: 3 docs with vectors, 1 doc without. The no-vector doc has a
        // strong keyword match. It must appear in results even when semantic
        // scoring would filter it out.
        let docs = vec![
            SearchDocument::new(
                "src/alpha.ts".to_string(),
                "alpha module".to_string(),
                vec!["alpha".to_string()],
                vec![],
                "alpha content".to_string(),
            ),
            SearchDocument::new(
                "src/beta.ts".to_string(),
                "beta module".to_string(),
                vec!["beta".to_string()],
                vec![],
                "beta content".to_string(),
            ),
            SearchDocument::new(
                "src/gamma.ts".to_string(),
                "gamma module".to_string(),
                vec!["gamma".to_string()],
                vec![],
                "gamma content".to_string(),
            ),
            // No embedding — pure keyword match.
            SearchDocument::new(
                "src/special.ts".to_string(),
                "special module".to_string(),
                vec!["findSpecial".to_string()],
                vec![],
                "findSpecial implementation".to_string(),
            ),
        ];

        let vectors: Vec<Option<Vec<f32>>> = vec![
            Some(vec![0.9, 0.1, 0.0, 0.0]),
            Some(vec![0.5, 0.5, 0.0, 0.0]),
            Some(vec![0.1, 0.9, 0.0, 0.0]),
            None, // no embedding for special.ts
        ];

        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let query_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        let opts = ResolvedSearchOptions {
            top_k: 5,
            semantic_weight: 0.5,
            keyword_weight: 0.5,
            min_semantic_score: 0.0, // allow zero semantic score
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
            ..Default::default()
        };

        let results = index.search("findSpecial", &query_vec, &opts);
        let paths: Vec<&str> = results.iter().map(|r| r.path.as_str()).collect();
        assert!(
            paths.contains(&"src/special.ts"),
            "no-embedding doc with keyword match must appear in results; got: {paths:?}"
        );
    }

    #[test]
    fn test_no_embedding_doc_without_keyword_does_not_pollute_results() {
        // Regression guard for the zero-score pollution gap: a doc with NO
        // embedding AND NO keyword match must NOT surface just because
        // `min_combined_score = 0.0` (default) lets a zero-score through.
        let docs = vec![
            SearchDocument::new(
                "src/alpha.ts".to_string(),
                "alpha module".to_string(),
                vec!["alpha".to_string()],
                vec![],
                "alpha content".to_string(),
            ),
            // No embedding, no symbol or content term that matches `findTarget`.
            SearchDocument::new(
                "src/noise.ts".to_string(),
                "unrelated".to_string(),
                vec!["nothingHere".to_string()],
                vec![],
                "unrelated content".to_string(),
            ),
        ];
        let vectors: Vec<Option<Vec<f32>>> = vec![Some(vec![0.9, 0.1, 0.0, 0.0]), None];

        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let query_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        let opts = ResolvedSearchOptions {
            top_k: 5,
            semantic_weight: 1.0, // fully semantic — keyword contributes 0
            keyword_weight: 0.0,
            ..Default::default() // min_* all 0.0 by default
        };

        let results = index.search("findTarget", &query_vec, &opts);
        let paths: Vec<&str> = results.iter().map(|r| r.path.as_str()).collect();
        assert!(
            !paths.contains(&"src/noise.ts"),
            "no-embedding doc with zero keyword match must not pollute results; got: {paths:?}"
        );
    }

    // -- Memory-efficiency tests --

    /// Helper: build a corpus of `n` docs with `dims`-dimensional unit vectors.
    fn make_small_dim_corpus(
        n: usize,
        dims: usize,
    ) -> (Vec<SearchDocument>, Vec<Option<Vec<f32>>>) {
        let docs: Vec<SearchDocument> = (0..n)
            .map(|i| {
                SearchDocument::new(
                    format!("src/file_{i}.rs"),
                    format!("file {i}"),
                    vec![format!("sym_{i}")],
                    vec![],
                    format!("content {i}"),
                )
            })
            .collect();
        let mut vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = vec![0.0f32; dims];
            v[i % dims] = 1.0;
            vectors.push(Some(v));
        }
        (docs, vectors)
    }

    /// When the corpus exceeds ANN_THRESHOLD the flat `vector_buffer` must be
    /// released (capacity == 0) because the `VectorStore` already owns the data.
    /// This guards the fix for the double-allocation identified in PR #48 review F2.
    #[test]
    fn test_vector_buffer_dropped_when_ann_store_present() {
        // Use small dims (16) to keep the test fast even at 2050 docs.
        let n = ANN_THRESHOLD + 50;
        let dims = 16;
        let (docs, vectors) = make_small_dim_corpus(n, dims);
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        assert!(
            index.ann_store.is_some(),
            "ann_store must be Some for n={n} >= ANN_THRESHOLD"
        );
        assert_eq!(
            index.vector_buffer.capacity(),
            0,
            "vector_buffer must be released (capacity 0) when ann_store owns the vectors; \
             double-allocation detected"
        );
    }

    /// Below ANN_THRESHOLD the buffer is retained for the brute-force path.
    #[test]
    fn test_vector_buffer_retained_below_threshold() {
        let n = ANN_THRESHOLD - 1;
        let dims = 16;
        let (docs, vectors) = make_small_dim_corpus(n, dims);
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        assert!(index.ann_store.is_none());
        assert!(
            index.vector_buffer.capacity() > 0,
            "vector_buffer must be retained below ANN_THRESHOLD"
        );
    }

    /// Regression guard for PR #51 review F2: verify search still returns
    /// correct results via the brute-force + vector_buffer path below ANN_THRESHOLD.
    #[test]
    fn test_search_correct_below_threshold_via_buffer() {
        // Use make_ann_corpus's deterministic R^4 layout but sized below threshold
        // so the brute-force path runs.
        let n = ANN_THRESHOLD - 50;
        let (docs, vectors) = make_ann_corpus(n);
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        assert!(index.ann_store.is_none(), "below threshold: ann_store=None");
        assert!(
            index.vector_buffer.capacity() > 0,
            "below threshold: vector_buffer must be retained"
        );

        let query_vec = unit_vec(1.0, 0.001);
        let opts = ResolvedSearchOptions {
            top_k: 3,
            semantic_weight: 1.0,
            keyword_weight: 0.0,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
            ..Default::default()
        };
        let results = index.search("anything", &query_vec, &opts);
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].path, "src/file_0.ts",
            "brute-force path must still rank file_0 (unit vec 1,0,0,0) top"
        );
    }

    /// Correctness: search still returns results after vector_buffer is dropped (ANN path).
    /// Uses make_ann_corpus (unique R^4 directions) to guarantee a clear nearest neighbour.
    #[test]
    fn test_search_correct_after_buffer_drop() {
        // Reuse make_ann_corpus which assigns unique monotonically-rotating unit
        // vectors in R^4. File_0 is the closest match to query (1,0,0,0).
        let n = ANN_THRESHOLD + 50;
        let (docs, vectors) = make_ann_corpus(n);
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        // Buffer must be gone.
        assert_eq!(index.vector_buffer.capacity(), 0);

        let query_vec = unit_vec(1.0, 0.001);
        let opts = ResolvedSearchOptions {
            top_k: 5,
            semantic_weight: 1.0,
            keyword_weight: 0.0,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
            ..Default::default()
        };

        let results = index.search("file", &query_vec, &opts);
        assert_eq!(results.len(), 5, "should return top_k results");
        // File_0 is the most aligned with query (1,~0,0,0).
        assert_eq!(
            results[0].path, "src/file_0.ts",
            "nearest doc must be file_0.ts after buffer drop"
        );
    }

    // -- CachedSearchIndex / IndexFingerprint tests --

    fn make_doc(path: &str, content: &str) -> SearchDocument {
        SearchDocument::new(
            path.to_string(),
            content[..content.len().min(80)].to_string(),
            vec![],
            vec![],
            content.to_string(),
        )
    }

    #[test]
    fn test_index_fingerprint_matches_same_docs() {
        let docs = vec![make_doc("a.rs", "hello"), make_doc("b.rs", "world")];
        let fp1 = IndexFingerprint::from_docs(&docs);
        let fp2 = IndexFingerprint::from_docs(&docs);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_index_fingerprint_differs_on_content_change() {
        let docs1 = vec![make_doc("a.rs", "hello")];
        let docs2 = vec![make_doc("a.rs", "hello CHANGED")];
        assert_ne!(
            IndexFingerprint::from_docs(&docs1),
            IndexFingerprint::from_docs(&docs2)
        );
    }

    #[test]
    fn test_index_fingerprint_differs_on_doc_count_change() {
        let docs1 = vec![make_doc("a.rs", "x")];
        let docs2 = vec![make_doc("a.rs", "x"), make_doc("b.rs", "y")];
        assert_ne!(
            IndexFingerprint::from_docs(&docs1),
            IndexFingerprint::from_docs(&docs2)
        );
    }

    #[test]
    fn test_cached_search_index_reuse_counter() {
        let docs = vec![make_doc("a.rs", "foo")];
        let fp = IndexFingerprint::from_docs(&docs);
        let mut idx = SearchIndex::new();
        idx.index_with_vectors(docs, vec![None]);
        let cached = CachedSearchIndex::new(idx, fp, 0);

        assert_eq!(cached.record_reuse(), 1);
        assert_eq!(cached.record_reuse(), 2);
        assert_eq!(cached.record_reuse(), 3);
    }

    // Verifies second identical query reuses the cache (reuse_count increments)
    // and that a changed walk result triggers a rebuild (reuse_count stays at 0).
    #[tokio::test]
    async fn test_semantic_code_search_cache_hit_and_miss() {
        use std::sync::atomic::Ordering;
        use std::sync::{Arc as StdArc, Mutex};

        // Stub embedder: always returns a fixed vector.
        struct FixedEmbedder;
        impl EmbedFn for FixedEmbedder {
            fn embed(
                &self,
                _texts: &[String],
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>,
            > {
                Box::pin(async { Ok(vec![vec![1.0_f32, 0.0]]) })
            }
        }

        // Stub walker: returns docs controlled by a shared counter.
        struct CountingWalker {
            rebuild_count: StdArc<Mutex<u32>>,
            // When true, the second call returns a doc with different content.
            change_on_second: bool,
        }
        impl WalkAndIndexFn for CountingWalker {
            fn walk_and_index(
                &self,
                _root: &Path,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)>,
                        > + Send
                        + '_,
                >,
            > {
                let mut count = self.rebuild_count.lock().unwrap();
                *count += 1;
                let call_n = *count;
                drop(count);
                let change = self.change_on_second && call_n > 1;
                Box::pin(async move {
                    let content = if change {
                        "changed content"
                    } else {
                        "stable content"
                    };
                    let docs = vec![make_doc("a.rs", content)];
                    let vecs = vec![Some(vec![1.0_f32, 0.0])];
                    Ok((docs, vecs))
                })
            }
        }

        let cache: RwLock<Option<Arc<CachedSearchIndex>>> = RwLock::new(None);
        let walk_count = StdArc::new(Mutex::new(0u32));

        let walker = CountingWalker {
            rebuild_count: StdArc::clone(&walk_count),
            change_on_second: false,
        };

        let opts = SemanticSearchOptions {
            root_dir: std::path::PathBuf::from("/tmp"),
            query: "stable".to_string(),
            top_k: None,
            semantic_weight: None,
            keyword_weight: None,
            min_semantic_score: None,
            min_keyword_score: None,
            min_combined_score: None,
            require_keyword_match: None,
            require_semantic_match: None,
            include_globs: None,
            exclude_globs: None,
            recency_window_days: None,
        };

        // First call — cache miss, index built.
        let _ = semantic_code_search(opts.clone(), &FixedEmbedder, &walker, Some(&cache), None)
            .await
            .unwrap();

        // Second call — cache hit, reuse_count should be 1.
        let _ = semantic_code_search(opts.clone(), &FixedEmbedder, &walker, Some(&cache), None)
            .await
            .unwrap();

        let reuses = {
            let guard = cache.read().await;
            guard.as_ref().unwrap().reuse_count.load(Ordering::Relaxed)
        };
        assert_eq!(
            reuses, 1,
            "Expected exactly one cache reuse on second identical query"
        );
    }

    // Verifies that a changed walk result clears the cached index.
    #[tokio::test]
    async fn test_semantic_code_search_cache_invalidated_on_change() {
        use std::sync::atomic::Ordering;
        use std::sync::{Arc as StdArc, Mutex};

        struct FixedEmbedder;
        impl EmbedFn for FixedEmbedder {
            fn embed(
                &self,
                _texts: &[String],
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>,
            > {
                Box::pin(async { Ok(vec![vec![1.0_f32, 0.0]]) })
            }
        }

        struct ChangingWalker {
            call: StdArc<Mutex<u32>>,
        }
        impl WalkAndIndexFn for ChangingWalker {
            fn walk_and_index(
                &self,
                _root: &Path,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)>,
                        > + Send
                        + '_,
                >,
            > {
                let mut n = self.call.lock().unwrap();
                *n += 1;
                let call_n = *n;
                drop(n);
                Box::pin(async move {
                    // Second walk returns different content — triggers fingerprint mismatch.
                    let content = if call_n == 1 {
                        "first"
                    } else {
                        "second-and-different"
                    };
                    let docs = vec![make_doc("a.rs", content)];
                    let vecs = vec![Some(vec![1.0_f32, 0.0])];
                    Ok((docs, vecs))
                })
            }
        }

        let cache: RwLock<Option<Arc<CachedSearchIndex>>> = RwLock::new(None);
        let call = StdArc::new(Mutex::new(0u32));
        let walker = ChangingWalker { call };

        let opts = SemanticSearchOptions {
            root_dir: std::path::PathBuf::from("/tmp"),
            query: "content".to_string(),
            top_k: None,
            semantic_weight: None,
            keyword_weight: None,
            min_semantic_score: None,
            min_keyword_score: None,
            min_combined_score: None,
            require_keyword_match: None,
            require_semantic_match: None,
            include_globs: None,
            exclude_globs: None,
            recency_window_days: None,
        };

        let _ = semantic_code_search(opts.clone(), &FixedEmbedder, &walker, Some(&cache), None)
            .await
            .unwrap();
        let _ = semantic_code_search(opts.clone(), &FixedEmbedder, &walker, Some(&cache), None)
            .await
            .unwrap();

        // reuse_count == 0: the second call replaced the cache rather than reusing it.
        let reuses = {
            let guard = cache.read().await;
            guard.as_ref().unwrap().reuse_count.load(Ordering::Relaxed)
        };
        assert_eq!(reuses, 0, "Changed walk should trigger rebuild, not reuse");
    }

    // Verifies concurrent requests with a stale cache rebuild at most once.
    #[tokio::test]
    async fn test_semantic_code_search_concurrent_no_double_rebuild() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicU32, Ordering};

        struct FixedEmbedder;
        impl EmbedFn for FixedEmbedder {
            fn embed(
                &self,
                _texts: &[String],
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>,
            > {
                Box::pin(async { Ok(vec![vec![1.0_f32, 0.0]]) })
            }
        }

        // Walker that counts how many times a fresh index is _requested_ to be built.
        // All walks return identical docs so fingerprints always match after the first build.
        struct StableWalker {
            walk_count: StdArc<AtomicU32>,
        }
        impl WalkAndIndexFn for StableWalker {
            fn walk_and_index(
                &self,
                _root: &Path,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)>,
                        > + Send
                        + '_,
                >,
            > {
                self.walk_count.fetch_add(1, Ordering::Relaxed);
                Box::pin(async {
                    let docs = vec![make_doc("a.rs", "stable")];
                    let vecs = vec![Some(vec![1.0_f32, 0.0])];
                    Ok((docs, vecs))
                })
            }
        }

        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let walk_count = StdArc::new(AtomicU32::new(0));

        let opts = SemanticSearchOptions {
            root_dir: std::path::PathBuf::from("/tmp"),
            query: "stable".to_string(),
            top_k: None,
            semantic_weight: None,
            keyword_weight: None,
            min_semantic_score: None,
            min_keyword_score: None,
            min_combined_score: None,
            require_keyword_match: None,
            require_semantic_match: None,
            include_globs: None,
            exclude_globs: None,
            recency_window_days: None,
        };

        // Spawn 8 concurrent queries against the same empty cache.
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache_ref = Arc::clone(&cache);
                let wc = StdArc::clone(&walk_count);
                let opts = opts.clone();
                tokio::spawn(async move {
                    let walker = StableWalker { walk_count: wc };
                    semantic_code_search(opts, &FixedEmbedder, &walker, Some(&*cache_ref), None)
                        .await
                        .unwrap();
                })
            })
            .collect();

        for h in handles {
            h.await.unwrap();
        }

        // All 8 tasks walk (to compute fingerprint), but the index build happens at most
        // a small number of times (bounded by write-lock contention, not 8).
        // With double-check locking, only the first writer actually rebuilds;
        // subsequent writers find the cache already matches and return early.
        // Walk count == 8 (every call walks), but we verify reuse_count shows cache hits.
        let reuses = {
            let guard = cache.read().await;
            guard
                .as_ref()
                .unwrap()
                .reuse_count
                .load(std::sync::atomic::Ordering::Relaxed)
        };
        // At least 7 out of 8 requests must have been cache hits.
        assert!(
            reuses >= 7,
            "Expected >= 7 cache reuses from 8 concurrent requests, got {reuses}"
        );
    }

    // ---------------------------------------------------------------------------
    // Generation-based cache tests
    // ---------------------------------------------------------------------------

    /// Helper: build a minimal SemanticSearchOptions for tests.
    fn gen_test_opts() -> SemanticSearchOptions {
        SemanticSearchOptions {
            root_dir: std::path::PathBuf::from("/tmp"),
            query: "anything".to_string(),
            top_k: None,
            semantic_weight: None,
            keyword_weight: None,
            min_semantic_score: None,
            min_keyword_score: None,
            min_combined_score: None,
            require_keyword_match: None,
            require_semantic_match: None,
            include_globs: None,
            exclude_globs: None,
            recency_window_days: None,
        }
    }

    struct FixedEmbedder2;
    impl EmbedFn for FixedEmbedder2 {
        fn embed(
            &self,
            _texts: &[String],
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>>
        {
            Box::pin(async { Ok(vec![vec![1.0_f32, 0.0]]) })
        }
    }

    /// Walker that counts walk_and_index invocations.
    struct CountWalker(Arc<std::sync::atomic::AtomicU32>);
    impl WalkAndIndexFn for CountWalker {
        fn walk_and_index(
            &self,
            _root: &Path,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)>,
                    > + Send
                    + '_,
            >,
        > {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Box::pin(async {
                let docs = vec![make_doc("a.rs", "stable content")];
                let vecs = vec![Some(vec![1.0_f32, 0.0])];
                Ok((docs, vecs))
            })
        }
    }

    /// Cache hit without a file change: when the generation matches, the walk
    /// should be skipped entirely on the second request.
    #[tokio::test]
    async fn test_generation_hit_skips_walk() {
        let cache: RwLock<Option<Arc<CachedSearchIndex>>> = RwLock::new(None);
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let walk_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let walker = CountWalker(Arc::clone(&walk_count));
        let opts = gen_test_opts();

        // First request — cold cache, must walk.
        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder2,
            &walker,
            Some(&cache),
            Some(&generation),
        )
        .await
        .unwrap();
        assert_eq!(
            walk_count.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "first request must walk"
        );

        // Second request — same generation, cache still valid → walk must be skipped.
        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder2,
            &walker,
            Some(&cache),
            Some(&generation),
        )
        .await
        .unwrap();
        assert_eq!(
            walk_count.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "second request with same generation must NOT walk"
        );

        // Verify the cache was actually reused (reuse_count == 1).
        let reuses = {
            let g = cache.read().await;
            g.as_ref()
                .unwrap()
                .reuse_count
                .load(std::sync::atomic::Ordering::Relaxed)
        };
        assert_eq!(reuses, 1, "cache must have been reused exactly once");
    }

    /// File-change event (generation bump) forces a walk + rebuild on next request.
    #[tokio::test]
    async fn test_generation_bump_triggers_walk() {
        let cache: RwLock<Option<Arc<CachedSearchIndex>>> = RwLock::new(None);
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let walk_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let walker = CountWalker(Arc::clone(&walk_count));
        let opts = gen_test_opts();

        // Prime the cache.
        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder2,
            &walker,
            Some(&cache),
            Some(&generation),
        )
        .await
        .unwrap();
        assert_eq!(
            walk_count.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "first request must walk"
        );

        // Simulate a tracker file-change event: bump the generation.
        generation.fetch_add(1, std::sync::atomic::Ordering::Release);

        // Next request — generation mismatch → must walk.
        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder2,
            &walker,
            Some(&cache),
            Some(&generation),
        )
        .await
        .unwrap();
        assert_eq!(
            walk_count.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "request after generation bump must re-walk"
        );
    }

    /// When tracker is disabled (generation=None), fingerprint path still works:
    /// the walk happens every time but the index is reused when content is unchanged.
    #[tokio::test]
    async fn test_tracker_off_fingerprint_backstop() {
        let cache: RwLock<Option<Arc<CachedSearchIndex>>> = RwLock::new(None);
        let walk_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let walker = CountWalker(Arc::clone(&walk_count));
        let opts = gen_test_opts();

        // No generation counter — tracker-off mode.
        let _ = semantic_code_search(opts.clone(), &FixedEmbedder2, &walker, Some(&cache), None)
            .await
            .unwrap();
        let _ = semantic_code_search(opts.clone(), &FixedEmbedder2, &walker, Some(&cache), None)
            .await
            .unwrap();

        // Walk happens on every request (no generation shortcut).
        assert_eq!(
            walk_count.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "tracker-off: walk on every request"
        );
        // But the index is reused (fingerprint matched on second call).
        let reuses = {
            let g = cache.read().await;
            g.as_ref()
                .unwrap()
                .reuse_count
                .load(std::sync::atomic::Ordering::Relaxed)
        };
        assert_eq!(
            reuses, 1,
            "fingerprint backstop: index reused on second call"
        );
    }

    /// Concurrent requests with matching generation: only the first request walks;
    /// the rest take the generation-hit fast path and skip the walk.
    #[tokio::test]
    async fn test_generation_concurrent_no_double_walk() {
        use std::sync::Arc as StdArc;

        let cache: StdArc<RwLock<Option<Arc<CachedSearchIndex>>>> = StdArc::new(RwLock::new(None));
        let generation: StdArc<std::sync::atomic::AtomicU64> =
            StdArc::new(std::sync::atomic::AtomicU64::new(0));
        let walk_count: StdArc<std::sync::atomic::AtomicU32> =
            StdArc::new(std::sync::atomic::AtomicU32::new(0));

        // Prime cache with a single synchronous call first.
        {
            let walker = CountWalker(StdArc::clone(&walk_count));
            let opts = gen_test_opts();
            let _ = semantic_code_search(
                opts,
                &FixedEmbedder2,
                &walker,
                Some(&*cache),
                Some(&generation),
            )
            .await
            .unwrap();
        }
        // Reset walk counter — we care about post-prime behaviour.
        walk_count.store(0, std::sync::atomic::Ordering::Relaxed);

        // Now spawn 8 concurrent requests — all should hit the generation cache.
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache_ref = StdArc::clone(&cache);
                let gen_ref = StdArc::clone(&generation);
                let wc = StdArc::clone(&walk_count);
                tokio::spawn(async move {
                    let walker = CountWalker(wc);
                    let opts = gen_test_opts();
                    semantic_code_search(
                        opts,
                        &FixedEmbedder2,
                        &walker,
                        Some(&*cache_ref),
                        Some(&gen_ref),
                    )
                    .await
                    .unwrap();
                })
            })
            .collect();
        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(
            walk_count.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "all 8 concurrent requests must skip the walk on a generation hit"
        );
    }
}
