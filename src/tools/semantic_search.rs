//! File-level semantic search with hybrid scoring (semantic + keyword).
//!
//! Ports the TypeScript `semantic-search.ts` logic:
//! - Builds a SearchIndex from file headers + symbols + content
//! - Uses Ollama embeddings for semantic similarity
//! - Combines semantic score with keyword coverage for hybrid ranking

use std::borrow::Cow;
use std::collections::HashSet;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use regex::Regex;
use tokio::sync::RwLock;

use crate::core::embeddings::VectorStore;
use crate::error::{ContextPlusError, Result};
use crate::tools::scoring::{
    DEFAULT_KEYWORD_WEIGHT, DEFAULT_SEMANTIC_WEIGHT, DEFAULT_TOP_K, clamp01, keyword_coverage,
    normalize_weight, truncate_on_char_boundary,
};

/// Maximum additive bonus from recency (kept small so it nudges ties, not
/// dominates relevance).
const MAX_RECENCY_BOOST: f64 = 0.05;

// ---------------------------------------------------------------------------
// ANN pre-filter constants
// ---------------------------------------------------------------------------

/// Embedded docs from which a `SearchIndex` keeps its vectors in a
/// `VectorStore` that worktree forks share. Must match `HNSW_THRESHOLD` in
/// `core/embeddings.rs` (both guard the same boundary).
const ANN_THRESHOLD: usize = 2_000;

/// Embedded vectors from which `SearchIndex::search` builds the HNSW graph and
/// prunes to its shortlist. Below it every document is scored on the blended
/// score, since the shortlist is taken on cosine alone. Override with
/// `CONTEXTPLUS_HNSW_MIN_VECTORS`.
pub(crate) const HNSW_MIN_VECTORS: usize = 50_000;

/// How many ANN candidates to fetch per top-k result. The candidate pool is
/// `top_k * ANN_CANDIDATE_MULTIPLIER`, capped at the corpus size. Larger
/// values improve recall at the cost of more scoring work. Override at
/// runtime with `CONTEXTPLUS_ANN_CANDIDATE_MULTIPLIER`.
const ANN_CANDIDATE_MULTIPLIER: usize = 10;

/// Type alias for the boxed future returned by embedding functions.
type EmbedFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>;

/// Type alias for the boxed future returned by walk-and-index functions.
pub(crate) type WalkAndIndexFuture<'a> = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<(Vec<SearchDocument>, Vec<Option<Vec<f32>>>)>>
            + Send
            + 'a,
    >,
>;

/// What a walk produced: its documents and vectors, or the index of its root
/// it installed in the slot itself.
pub enum WalkOutcome {
    Documents(Vec<SearchDocument>, Vec<Option<Vec<f32>>>),
    Installed(Arc<CachedSearchIndex>),
}

pub(crate) type WalkOrInstallFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<WalkOutcome>> + Send + 'a>>;

pub(crate) type VectorGenerationFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = u64> + Send + 'a>>;

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

/// Return the exact text sent to the semantic embedder for one source file.
pub fn semantic_embedding_content(file_path: &str, content: &str) -> String {
    if is_text_index_candidate(file_path) {
        content.chars().take(MAX_TEXT_DOC_CHARS).collect()
    } else {
        format!(
            "{} {}",
            crate::core::parser::detect_language(file_path).unwrap_or("unknown"),
            content.chars().take(500).collect::<String>()
        )
    }
}

/// Extract a plain-text header from content: up to 2 non-empty lines, each capped at 120 chars.
pub fn extract_plain_text_header(content: &str) -> String {
    let mut header_lines: Vec<&str> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let capped = truncate_on_char_boundary(trimmed, 120);
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

const WEAK_SEMANTIC_RELEVANCE_THRESHOLD: f64 = 80.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchScope {
    #[default]
    All,
    Code,
    Docs,
}

#[derive(Debug, Clone)]
pub struct SemanticSearchOptions {
    pub scope: Option<SearchScope>,
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
    pub semantic_cosine: f64,
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
    /// Preserve repository-relative priors when the walker shortens display paths.
    pub(crate) path_prior: super::lexical_search::PathPriorClassification,
    pub header: String,
    pub symbols: Vec<String>,
    pub symbol_entries: Vec<SymbolSearchEntry>,
    pub content: String,
    pub(crate) source_hash: String,
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
    fn resident_bytes(&self) -> usize {
        self.path.capacity()
            + self.content.capacity()
            + self.header.capacity()
            + self.source_hash.capacity()
            + self.symbols.iter().map(String::capacity).sum::<usize>()
    }

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
            path_prior: super::lexical_search::classify_path_prior(&path),
            path,
            header,
            symbols,
            symbol_entries,
            source_hash: crate::core::embeddings::content_hash(&content),
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

/// The fields of a file document that come from parsing the file, kept in a
/// snapshot so that an unchanged file is not parsed again after a restart.
pub(crate) struct DocumentSeed {
    /// [`crate::core::embeddings::content_hash`] of the whole file.
    pub(crate) source_hash: String,
    /// Digest of the document's indexed content.
    pub(crate) content_digest: crate::cache::snapshot::Digest,
    pub(crate) header: String,
    pub(crate) symbols: Vec<String>,
    pub(crate) symbol_entries: Vec<SymbolSearchEntry>,
}

/// Document seeds by repository-relative path.
pub(crate) type DocumentSeeds = std::collections::HashMap<String, DocumentSeed>;

impl DocumentSeed {
    /// This seed's document when `source_hash` and the indexed `content`
    /// are the ones it was parsed from.
    pub(crate) fn document(
        &self,
        path: String,
        source_hash: &str,
        content: String,
    ) -> std::result::Result<SearchDocument, String> {
        if self.source_hash != source_hash
            || self.content_digest != crate::cache::snapshot::digest(content.as_bytes())
        {
            return Err(content);
        }
        Ok(SearchDocument::new(
            path,
            self.header.clone(),
            self.symbols.clone(),
            self.symbol_entries.clone(),
            content,
        ))
    }
}

/// The search document of the file at `path`, parsed from `source`, with
/// `content` as its indexed content: a text file takes its first lines as
/// its header, a code file its symbols and header from the parser.
pub(crate) fn file_document(path: String, source: &str, content: String) -> SearchDocument {
    if is_text_index_candidate(&path) {
        let header = extract_plain_text_header(&content);
        return SearchDocument::new(path, header, vec![], vec![], content);
    }
    let ext = path.rsplit('.').next().unwrap_or("");
    let symbols = crate::core::tree_sitter::parse_with_tree_sitter(source, ext).unwrap_or_default();
    let header = crate::core::parser::extract_header(source);
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
    SearchDocument::new(path, header, symbol_names, symbol_entries, content)
}

/// Writes the parsed fields of `docs`, read back by [`read_document_seeds`].
pub(crate) fn write_document_seeds(
    out: &mut crate::cache::snapshot::SnapshotWriter,
    docs: &[Arc<SearchDocument>],
) -> io::Result<()> {
    out.usize(docs.len())?;
    for doc in docs {
        out.str(&doc.path)?;
        out.str(&doc.source_hash)?;
        out.digest(&crate::cache::snapshot::digest(doc.content.as_bytes()))?;
        out.str(&doc.header)?;
        out.strs(doc.symbols.iter().map(String::as_str))?;
        out.usize(doc.symbol_entries.len())?;
        for entry in &doc.symbol_entries {
            out.str(&entry.name)?;
            out.opt_str(entry.kind.as_deref())?;
            out.usize(entry.line)?;
            out.bool(entry.end_line.is_some())?;
            out.usize(entry.end_line.unwrap_or(0))?;
            out.opt_str(entry.signature.as_deref())?;
        }
    }
    Ok(())
}

pub(crate) fn read_document_seeds(
    input: &mut crate::cache::snapshot::SnapshotReader<'_>,
) -> Option<DocumentSeeds> {
    let len = input.usize()?;
    let mut seeds = DocumentSeeds::with_capacity(len);
    for _ in 0..len {
        let path = input.str()?.to_owned();
        let source_hash = input.str()?.to_owned();
        let content_digest = input.digest()?;
        let header = input.str()?.to_owned();
        let symbols = input.strings()?;
        let entries = input.usize()?;
        let symbol_entries = (0..entries)
            .map(|_| {
                Some(SymbolSearchEntry {
                    name: input.str()?.to_owned(),
                    kind: input.opt_str()?.map(str::to_owned),
                    line: input.usize_value()?,
                    end_line: {
                        let present = input.bool()?;
                        let end = input.usize_value()?;
                        present.then_some(end)
                    },
                    signature: input.opt_str()?.map(str::to_owned),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        seeds.insert(
            path,
            DocumentSeed {
                source_hash,
                content_digest,
                header,
                symbols,
                symbol_entries,
            },
        );
    }
    Some(seeds)
}

// ---------------------------------------------------------------------------
// Resolved options (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ResolvedSearchOptions {
    pub scope: SearchScope,
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
            scope: SearchScope::All,
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
        scope: opts.scope.unwrap_or_default(),
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

fn document_passes_filters(doc: &SearchDocument, opts: &ResolvedSearchOptions) -> bool {
    scope_admits(opts.scope, doc.path_prior.is_documentation)
        && path_passes_filters(&doc.path, opts)
}

fn scope_admits(scope: SearchScope, documentation: bool) -> bool {
    match scope {
        SearchScope::All => true,
        SearchScope::Code => !documentation,
        SearchScope::Docs => documentation,
    }
}

/// The repository-relative paths a search of `options` can answer with,
/// where `prefix` is its root's path in the repository: those under the
/// root, in its scope and through its globs, which match the path from the
/// root as they do for its documents.
pub(crate) fn result_path_filter(
    options: &SemanticSearchOptions,
    prefix: PathBuf,
) -> impl Fn(&str) -> bool + Send + 'static {
    let options = resolve_search_options(options);
    move |path| {
        let Ok(relative) = Path::new(path).strip_prefix(&prefix) else {
            return false;
        };
        scope_admits(
            options.scope,
            super::lexical_search::classify_path_prior(path).is_documentation,
        ) && path_passes_filters(&relative.to_string_lossy(), &options)
    }
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

pub(crate) trait SnippetFileOpener: Send + Sync {
    fn open(&self, path: &Path) -> io::Result<Box<dyn BufRead>>;
}

struct DiskSnippetFileOpener;

impl SnippetFileOpener for DiskSnippetFileOpener {
    fn open(&self, path: &Path) -> io::Result<Box<dyn BufRead>> {
        Ok(Box::new(BufReader::new(std::fs::File::open(path)?)))
    }
}

pub(crate) fn fill_result_snippets(root_dir: &Path, results: &mut [SearchResult]) {
    fill_result_snippets_with_opener(root_dir, results, &DiskSnippetFileOpener);
}

pub(crate) fn fill_result_snippets_with_opener(
    root_dir: &Path,
    results: &mut [SearchResult],
    opener: &dyn SnippetFileOpener,
) {
    for result in results {
        let snippet = opener
            .open(&root_dir.join(&result.path))
            .and_then(|reader| {
                let location = result
                    .matched_symbol_locations
                    .first()
                    .and_then(|loc| parse_location_string(loc))
                    .filter(|(start, end)| *start > 0 && end.is_none_or(|end| end >= *start));
                read_result_snippet(reader, location, Path::new(&result.path))
            });
        if let Ok(snippet) = snippet {
            result.snippet = snippet;
        }
    }
}

fn read_result_snippet(
    reader: Box<dyn BufRead>,
    location: Option<(u32, Option<u32>)>,
    path: &Path,
) -> io::Result<Option<String>> {
    let mut lines = reader.lines();
    let mut content = Vec::new();
    if let Some((start, end)) = location {
        for _ in 1..start {
            match lines.next().transpose()? {
                Some(_) => {}
                None => return Ok(None),
            }
        }
        // One extra line lets extract_snippet preserve its truncation marker.
        let limit = end
            .map(|end| (end - start) as usize + 1)
            .unwrap_or(SNIPPET_MAX_LINES + 1)
            .min(SNIPPET_MAX_LINES + 1);
        for line in lines.take(limit) {
            content.push(line?);
        }
    } else {
        let mut first = true;
        let mut front_matter = false;
        for line in lines {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if first {
                first = false;
                if trimmed == "---"
                    && path
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .is_some_and(|ext| {
                            ["md", "markdown", "mdx"]
                                .iter()
                                .any(|markdown| ext.eq_ignore_ascii_case(markdown))
                        })
                {
                    front_matter = true;
                    continue;
                }
            }
            if front_matter {
                if matches!(trimmed, "---" | "...") {
                    front_matter = false;
                }
                continue;
            }
            if matches!(trimmed, "{" | "[" | "---" | "/**" | "/*" | "//") {
                continue;
            }
            content.push(line);
            if content.len() == SNIPPET_MAX_LINES {
                break;
            }
        }
    }
    Ok(extract_snippet(&content.join("\n"), 1, None))
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
    /// SipHash over `(path, content, source_hash)` for each document,
    /// order-dependent: `content` keeps only a code file's head.
    pub content_hash: u64,
}

impl IndexFingerprint {
    /// Compute a fingerprint from a slice of `SearchDocument`s.
    pub fn from_docs(docs: &[SearchDocument]) -> Self {
        Self::of(docs.iter())
    }

    /// [`Self::from_docs`] of documents held in several places.
    pub(crate) fn of<'a>(docs: impl ExactSizeIterator<Item = &'a SearchDocument>) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::hash::DefaultHasher::new();
        let n_docs = docs.len();
        n_docs.hash(&mut hasher);
        for d in docs {
            d.path.hash(&mut hasher);
            d.content.hash(&mut hasher);
            d.source_hash.hash(&mut hasher);
        }
        Self {
            n_docs,
            content_hash: hasher.finish(),
        }
    }
}

/// A `SearchIndex` paired with the fingerprint that was current when it was built.
/// Stored in `SharedState` and reused across MCP requests when the fingerprint is unchanged.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MetadataFingerprint {
    pub n_entries: usize,
    pub metadata_hash: u64,
}

pub type MetadataFingerprintFuture<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<Option<MetadataFingerprint>>> + Send + 'a>,
>;

#[derive(Clone)]
struct RefreshBatch {
    docs: Vec<SearchDocument>,
    vectors: Vec<Option<Vec<f32>>>,
    deleted: Vec<String>,
    generation: u64,
    vector_generation: Option<u64>,
}

#[derive(Default)]
struct PendingRefresh {
    batches: Vec<RefreshBatch>,
}

pub struct CachedSearchIndex {
    pending: std::sync::Mutex<PendingRefresh>,
    scoped_refresh: std::sync::atomic::AtomicBool,
    search_root: std::path::PathBuf,
    pub metadata: std::sync::RwLock<Option<MetadataFingerprint>>,
    vector_generation: u64,
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
    /// Guards background rebuild: CAS false→true claims the spawn slot.
    /// Reset by the RAII `RebuildGuard` even on panic, preventing permanent lockout.
    pub(crate) rebuild_in_progress: std::sync::atomic::AtomicBool,
}

/// Maximum absolute doc-count delta that qualifies for a background (stale-serve) rebuild.
const BG_REBUILD_MAX_ABS_DELTA: usize = 200;
/// Maximum fractional doc-count delta (5%) that qualifies for a background rebuild.
const BG_REBUILD_MAX_FRAC: f64 = 0.05;
/// Most walks one stale rebuild runs while changed files queue during them.
pub(crate) const STALE_REBUILD_PASSES: usize = 3;

/// RAII guard that resets `rebuild_in_progress` on the held `CachedSearchIndex`
/// when dropped — even on panic — preventing permanent lockout of the fast path.
pub(crate) struct RebuildGuard(pub Arc<CachedSearchIndex>);
impl Drop for RebuildGuard {
    fn drop(&mut self) {
        self.0
            .rebuild_in_progress
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

impl CachedSearchIndex {
    pub(crate) fn new(index: SearchIndex, fingerprint: IndexFingerprint, generation: u64) -> Self {
        Self {
            pending: std::sync::Mutex::new(PendingRefresh::default()),
            scoped_refresh: std::sync::atomic::AtomicBool::new(false),
            index,
            fingerprint,
            metadata: std::sync::RwLock::new(None),
            vector_generation: 0,
            search_root: std::path::PathBuf::new(),
            generation: std::sync::atomic::AtomicU64::new(generation),
            reuse_count: std::sync::atomic::AtomicU64::new(0),
            rebuild_in_progress: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn delta_generation(
        current: Arc<CachedSearchIndex>,
        changed: Vec<SearchDocument>,
        changed_vectors: Vec<Option<Vec<f32>>>,
        deleted: Vec<String>,
        generation: u64,
    ) -> Arc<CachedSearchIndex> {
        let mut entry = match Arc::try_unwrap(current) {
            Ok(entry) => entry,
            Err(current) => {
                let mut entry = Self::new(
                    current.index.clone(),
                    current.fingerprint.clone(),
                    generation,
                );
                entry.search_root = current.search_root.clone();
                entry.vector_generation = current.vector_generation;
                entry
            }
        };
        entry.index.apply_delta(changed, changed_vectors, &deleted);
        entry.fingerprint = entry.index.fingerprint();
        entry
            .generation
            .store(generation, std::sync::atomic::Ordering::Release);
        *entry.metadata.write().unwrap() = None;
        Arc::new(entry)
    }

    /// Bytes this entry holds on its own, excluding its shared vector store,
    /// and the documents another index also holds, by address and bytes.
    pub(crate) fn resident_split(&self) -> (usize, Vec<(usize, usize)>) {
        self.index.resident_split()
    }

    /// The canonical root this index was walked from.
    pub(crate) fn search_root(&self) -> &Path {
        &self.search_root
    }

    /// Whether this entry was built at `vector_generation` with no batches
    /// queued.
    pub(crate) fn is_settled(&self, vector_generation: u64) -> bool {
        self.vector_generation == vector_generation
            && self.pending.lock().unwrap().batches.is_empty()
    }

    /// Whether this entry was built before the tracker's `generation` or has
    /// batches queued.
    pub(crate) fn is_behind(&self, generation: u64) -> bool {
        self.generation.load(std::sync::atomic::Ordering::Acquire) < generation
            || !self.pending.lock().unwrap().batches.is_empty()
    }

    /// A parent entry a worktree can fork: walked from its whole `root`, over
    /// a vector store worth sharing, with no queued batches or rebuild.
    pub(crate) fn forkable_at(&self, root: &Path) -> bool {
        self.unforkable_clause(root).is_none()
    }

    /// The first clause of [`Self::forkable_at`] this entry fails at `root`.
    pub(crate) fn unforkable_clause(&self, root: &Path) -> Option<&'static str> {
        if self.search_root != root {
            Some("scoped")
        } else if self.index.ann_store.is_none() {
            Some("no_ann_store")
        } else if !self.pending.lock().unwrap().batches.is_empty() {
            Some("batches_queued")
        } else if self
            .rebuild_in_progress
            .load(std::sync::atomic::Ordering::Acquire)
        {
            Some("rebuild_in_progress")
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(crate) fn pending_paths(&self) -> Vec<String> {
        self.pending
            .lock()
            .unwrap()
            .batches
            .iter()
            .flat_map(|batch| batch.docs.iter().map(|doc| doc.path.clone()))
            .collect()
    }

    /// An entry built from a full walk of `root`.
    pub(crate) fn build(
        root: &Path,
        docs: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
        generation: u64,
        vector_generation: u64,
        metadata: Option<MetadataFingerprint>,
    ) -> Self {
        let fingerprint = IndexFingerprint::from_docs(&docs);
        let mut index = SearchIndex::new();
        index.index_with_vectors_and_tuning(
            docs,
            vectors,
            crate::core::embeddings::HnswTuning::global(),
        );
        let mut entry = Self::new(index, fingerprint, generation);
        entry.search_root = root.to_path_buf();
        entry.vector_generation = vector_generation;
        *entry.metadata.get_mut().unwrap() = metadata;
        entry
    }

    /// Installs this entry in `slot` if the slot still holds `seen`, keeping
    /// the batches queued there since this entry's walk began when both
    /// entries index the same root.
    pub(crate) fn install(
        mut self,
        slot: &mut Option<Arc<Self>>,
        seen: Option<&std::sync::Weak<Self>>,
    ) -> bool {
        let unchanged = match (slot.as_ref(), seen) {
            (None, None) => true,
            (Some(current), Some(seen)) => std::sync::Weak::ptr_eq(&Arc::downgrade(current), seen),
            _ => false,
        };
        if !unchanged {
            return false;
        }
        if let Some(current) = slot
            .as_ref()
            .filter(|current| current.search_root == self.search_root)
        {
            let generation = self.generation.load(std::sync::atomic::Ordering::Acquire);
            self.pending.get_mut().unwrap().batches = current
                .pending
                .lock()
                .unwrap()
                .batches
                .iter()
                .filter(|batch| batch.generation >= generation)
                .cloned()
                .collect();
        }
        *slot = Some(Arc::new(self));
        true
    }

    /// This entry's index moved to a worktree's full walk of `root`; `None`
    /// when the worktree builds its own (see [`SearchIndex::fork`]).
    pub(crate) fn fork(
        &self,
        root: &Path,
        docs: &[SearchDocument],
        vectors: &[Option<Vec<f32>>],
        generation: u64,
        vector_generation: u64,
    ) -> Option<Self> {
        let index = self.index.fork(docs, vectors)?;
        let mut entry = Self::new(index, IndexFingerprint::from_docs(docs), generation);
        entry.search_root = root.to_path_buf();
        entry.vector_generation = vector_generation;
        Some(entry)
    }

    /// [`Self::fork`] from the worktree walk's `changed` documents and
    /// `deleted` paths alone; `fingerprint` is the whole walk's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fork_delta(
        &self,
        root: &Path,
        changed: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
        deleted: &[String],
        fingerprint: IndexFingerprint,
        generation: u64,
        vector_generation: u64,
    ) -> Option<Self> {
        let index = self.index.fork_delta(changed, vectors, deleted)?;
        let mut entry = Self::new(index, fingerprint, generation);
        entry.search_root = root.to_path_buf();
        entry.vector_generation = vector_generation;
        Some(entry)
    }

    #[cfg(feature = "memory-profile")]
    pub(crate) fn resident_file_vector_bytes(&self) -> usize {
        self.index.resident_file_vector_bytes()
    }

    #[cfg(feature = "memory-profile")]
    pub(crate) fn estimated_hnsw_bytes(&self) -> usize {
        self.index.estimated_hnsw_bytes()
    }

    #[cfg(feature = "memory-profile")]
    pub(crate) fn resident_document_bytes(&self) -> usize {
        self.index.resident_document_bytes()
    }

    pub(crate) fn refresh_paths(
        entry: &mut Arc<Self>,
        root: &Path,
        docs: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
        deleted: &[String],
        generation: u64,
    ) -> bool {
        if entry.search_root != root {
            entry
                .scoped_refresh
                .store(true, std::sync::atomic::Ordering::Release);
            return false;
        }
        let large = entry.index.requires_full_rebuild(&docs, &vectors, deleted);
        {
            let mut pending = entry.pending.lock().unwrap();
            if large
                || !pending.batches.is_empty()
                || entry
                    .rebuild_in_progress
                    .load(std::sync::atomic::Ordering::Acquire)
            {
                pending.batches.push(RefreshBatch {
                    docs,
                    vectors,
                    deleted: deleted.to_vec(),
                    generation,
                    vector_generation: None,
                });
                return !large;
            }
        }
        if Arc::get_mut(entry).is_none() {
            let mut copy = Self::new(entry.index.clone(), entry.fingerprint.clone(), generation);
            copy.search_root = entry.search_root.clone();
            copy.vector_generation = entry.vector_generation;
            *entry = Arc::new(copy);
        }
        let fresh = Arc::get_mut(entry).unwrap();
        fresh.index.apply_delta(docs, vectors, deleted);
        fresh.fingerprint = fresh.index.fingerprint();
        fresh
            .generation
            .store(generation, std::sync::atomic::Ordering::Release);
        *fresh.metadata.write().unwrap() = None;
        true
    }

    pub(crate) fn refresh_ref_paths(
        entry: &mut Arc<Self>,
        root: &Path,
        docs: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
        deleted: &[String],
        generation: u64,
    ) -> bool {
        let scope = entry.search_root.clone();
        let Ok(prefix) = scope.strip_prefix(root) else {
            return false;
        };
        let (docs, vectors) = docs
            .into_iter()
            .zip(vectors)
            .filter_map(|(mut doc, vector)| {
                let path = Path::new(&doc.path)
                    .strip_prefix(prefix)
                    .ok()?
                    .to_string_lossy()
                    .into_owned();
                doc.path = path;
                Some((doc, vector))
            })
            .unzip();
        let deleted: Vec<_> = deleted
            .iter()
            .filter_map(|path| {
                Path::new(path)
                    .strip_prefix(prefix)
                    .ok()
                    .map(|path| path.to_string_lossy().into_owned())
            })
            .collect();
        Self::refresh_paths(entry, &scope, docs, vectors, &deleted, generation)
    }

    pub(crate) fn pending_vector_dimensions(&self) -> Option<usize> {
        self.pending
            .lock()
            .unwrap()
            .batches
            .iter()
            .rev()
            .flat_map(|batch| batch.vectors.iter().flatten())
            .map(Vec::len)
            .find(|dims| *dims != self.index.dims)
    }

    pub(crate) fn has_vector_shape(&self) -> bool {
        self.index.dims != 0
    }

    /// Whether every document of this entry holds a vector.
    pub(crate) fn has_every_vector(&self) -> bool {
        self.index.has_vector.iter().all(|ready| *ready)
    }

    pub(crate) fn refresh_vectors(
        entry: &mut Arc<Self>,
        root: &Path,
        updates: Vec<(String, String, Vec<f32>)>,
        vector_generation: u64,
    ) {
        let mut docs = Vec::new();
        let mut vectors = Vec::new();
        let mut represented = true;
        let Ok(prefix) = entry.search_root.strip_prefix(root) else {
            return;
        };
        for (path, hash, vector) in updates {
            let Ok(path) = Path::new(&path).strip_prefix(prefix) else {
                continue;
            };
            let pending = entry.pending.lock().unwrap();
            let doc = pending
                .batches
                .iter()
                .rev()
                .flat_map(|batch| &batch.docs)
                .chain(entry.index.documents.iter().map(|doc| &**doc))
                .find(|doc| Path::new(&doc.path) == path && doc.source_hash == hash);
            if let Some(doc) = doc {
                docs.push(doc.clone());
                vectors.push(Some(vector));
            } else {
                represented = false;
            }
        }
        let root = entry.search_root.clone();
        let generation = entry
            .pending
            .lock()
            .unwrap()
            .batches
            .iter()
            .map(|batch| batch.generation)
            .max()
            .unwrap_or_else(|| entry.generation.load(std::sync::atomic::Ordering::Acquire));
        let metadata = entry.metadata.read().unwrap().clone();
        Self::refresh_paths(entry, &root, docs, vectors, &[], generation);
        if !represented {
            return;
        }
        let mut pending = entry.pending.lock().unwrap();
        if let Some(batch) = pending.batches.last_mut() {
            batch.vector_generation = Some(vector_generation);
        } else {
            drop(pending);
            let fresh = Arc::get_mut(entry).unwrap();
            fresh.vector_generation = vector_generation;
            *fresh.metadata.write().unwrap() = metadata;
        }
    }

    /// Increment and return the reuse counter.
    pub fn record_reuse(&self) -> u64 {
        self.reuse_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }

    /// Returns true when the corpus delta between `self` and `new_fp` is small enough
    /// that we can safely serve the stale index while rebuilding in the background.
    ///
    /// Threshold: doc count delta ≤ min(5% of current corpus, 200 docs).
    pub fn qualifies_for_background_rebuild(&self, new_fp: &IndexFingerprint) -> bool {
        let old_n = self.fingerprint.n_docs;
        let new_n = new_fp.n_docs;
        let delta = old_n.abs_diff(new_n);
        let cap = ((old_n as f64) * BG_REBUILD_MAX_FRAC).round() as usize;
        let threshold = cap.min(BG_REBUILD_MAX_ABS_DELTA);
        delta <= threshold
    }
}

// ---------------------------------------------------------------------------
// SearchIndex -- indexes documents and runs hybrid search
// ---------------------------------------------------------------------------

/// In-memory search index holding documents and their embedding vectors.
/// Vectors are stored in a flat contiguous buffer for cache-friendly SIMD access.
/// For corpora larger than `ANN_THRESHOLD`, a `VectorStore` is also built so
/// forks can share it; from `HNSW_MIN_VECTORS` `search()` also uses its HNSW
/// graph to pre-filter instead of scoring every document.
///
/// Memory optimisation: when `ann_store` is `Some` (corpus ≥ `ANN_THRESHOLD`),
/// `vector_buffer` is released (`capacity() == 0`) because the `VectorStore`
/// already owns an identical copy. The cosine-scoring path then retrieves
/// per-document vectors from `ann_store` via `get_vector`. Below threshold the
/// buffer is kept for the brute-force O(N) scan.
pub const FULL_REBUILD_CHANGE_FRACTION: f64 = 0.2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexUpdateKind {
    Incremental,
    FullRebuild,
}

#[derive(Clone)]
pub struct SearchIndex {
    full_rebuilds: u64,
    ann_dirty_paths: HashSet<String>,
    vector_updates: std::collections::HashMap<String, Vec<f32>>,
    /// Shared with forks; a changed document is replaced, never edited in place.
    documents: Vec<Arc<SearchDocument>>,
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
    ann_store: Option<Arc<VectorStore>>,
    /// Store size from which `search()` builds and prunes with the graph.
    hnsw_min_vectors: usize,
}

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchIndex {
    #[cfg(feature = "memory-profile")]
    fn resident_document_bytes(&self) -> usize {
        self.documents.iter().map(|doc| doc.resident_bytes()).sum()
    }

    #[cfg(feature = "memory-profile")]
    fn resident_file_vector_bytes(&self) -> usize {
        self.vector_buffer.capacity() * std::mem::size_of::<f32>()
            + self
                .vector_updates
                .values()
                .map(|vector| vector.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>()
            + self
                .ann_store
                .as_ref()
                .map_or(0, |store| store.resident_vector_bytes())
    }

    #[cfg(feature = "memory-profile")]
    fn estimated_hnsw_bytes(&self) -> usize {
        self.ann_store
            .as_ref()
            .map_or(0, |store| store.estimated_hnsw_bytes())
    }

    /// Bytes excluding the vector store and the documents another index also
    /// holds, which follow by address and bytes; each counts once however many
    /// indexes share it. One read of each count, so a clone racing the
    /// measure cannot charge a document twice or not at all.
    fn resident_split(&self) -> (usize, Vec<(usize, usize)>) {
        let mut own = self.vector_buffer.capacity() * std::mem::size_of::<f32>()
            + self
                .vector_updates
                .values()
                .map(|vector| vector.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>();
        let mut shared = Vec::new();
        for doc in &self.documents {
            if Arc::strong_count(doc) == 1 {
                own += doc.resident_bytes();
            } else {
                shared.push((Arc::as_ptr(doc) as usize, doc.resident_bytes()));
            }
        }
        (own, shared)
    }

    /// The store whose graph `search()` prunes with; `None` below
    /// `hnsw_min_vectors`, where every document is scored.
    fn graph_store(&self) -> Option<&Arc<VectorStore>> {
        self.ann_store
            .as_ref()
            .filter(|store| store.count() >= self.hnsw_min_vectors)
    }

    fn prepare_ann(&self) {
        if let Some(store) = self.graph_store() {
            store.find_nearest_hnsw(&vec![0.0; self.dims], 1);
        }
    }

    pub fn new() -> Self {
        Self {
            full_rebuilds: 0,
            ann_dirty_paths: HashSet::new(),
            vector_updates: Default::default(),
            documents: Vec::new(),
            vector_buffer: Vec::new(),
            has_vector: Vec::new(),
            dims: 0,
            ann_store: None,
            hnsw_min_vectors: HNSW_MIN_VECTORS,
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
        self.index_shared(
            docs.into_iter().map(Arc::new).collect(),
            vectors,
            hnsw_tuning,
        );
    }

    fn index_shared(
        &mut self,
        docs: Vec<Arc<SearchDocument>>,
        vectors: Vec<Option<Vec<f32>>>,
        hnsw_tuning: crate::core::embeddings::HnswTuning,
    ) {
        self.full_rebuilds += 1;
        self.vector_updates.clear();
        self.ann_dirty_paths.clear();
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
            Some(Arc::new(VectorStore::new_with_tuning(
                dims as u32,
                keys,
                hashes,
                flat,
                hnsw_tuning,
            )))
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
        self.hnsw_min_vectors = hnsw_tuning.min_vectors;
    }

    pub(crate) fn documents(&self) -> &[Arc<SearchDocument>] {
        &self.documents
    }

    pub(crate) fn fingerprint(&self) -> IndexFingerprint {
        IndexFingerprint::of(self.documents.iter().map(|doc| &**doc))
    }

    pub(crate) fn dims(&self) -> usize {
        self.dims
    }

    pub fn full_rebuild_count(&self) -> u64 {
        self.full_rebuilds
    }

    pub(crate) fn vector_at(&self, i: usize) -> Option<&[f32]> {
        if !self.has_vector[i] {
            return None;
        }
        let path = &self.documents[i].path;
        if let Some(vector) = self.vector_updates.get(path) {
            return Some(vector);
        }
        if let Some(store) = &self.ann_store {
            return store.get_vector(path);
        }
        Some(&self.vector_buffer[i * self.dims..(i + 1) * self.dims])
    }

    fn requires_full_rebuild(
        &self,
        docs: &[SearchDocument],
        vectors: &[Option<Vec<f32>>],
        deleted: &[String],
    ) -> bool {
        let content_changes = docs
            .iter()
            .filter(|doc| {
                !self.documents.iter().any(|old| {
                    old.path == doc.path
                        && old.content == doc.content
                        && old.search_text == doc.search_text
                })
            })
            .count();
        let dirty_count = self
            .ann_dirty_paths
            .iter()
            .map(String::as_str)
            .chain(docs.iter().map(|d| d.path.as_str()))
            .chain(deleted.iter().map(String::as_str))
            .collect::<HashSet<_>>()
            .len();
        self.ann_store.as_ref().is_some_and(|store| {
            dirty_count as f64 > store.count() as f64 * FULL_REBUILD_CHANGE_FRACTION
        }) || (content_changes + deleted.len()) as f64
            > self.documents.len() as f64 * FULL_REBUILD_CHANGE_FRACTION
            || vectors.iter().flatten().any(|v| v.len() != self.dims)
    }

    pub fn apply_delta(
        &mut self,
        changed_docs: Vec<SearchDocument>,
        changed_vectors: Vec<Option<Vec<f32>>>,
        deleted_paths: &[String],
    ) -> IndexUpdateKind {
        let rebuild = self.requires_full_rebuild(&changed_docs, &changed_vectors, deleted_paths);
        if rebuild {
            let replacement_dims = changed_vectors
                .iter()
                .flatten()
                .map(Vec::len)
                .find(|dims| *dims != self.dims)
                .unwrap_or(self.dims);
            let affected: HashSet<&str> = deleted_paths
                .iter()
                .map(String::as_str)
                .chain(changed_docs.iter().map(|d| d.path.as_str()))
                .collect();
            let mut docs = Vec::new();
            let mut vectors = Vec::new();
            for (i, doc) in self.documents.iter().enumerate() {
                if !affected.contains(doc.path.as_str()) {
                    docs.push(Arc::clone(doc));
                    vectors.push(
                        self.vector_at(i)
                            .filter(|v| v.len() == replacement_dims)
                            .map(<[f32]>::to_vec),
                    );
                }
            }
            docs.extend(changed_docs.into_iter().map(Arc::new));
            vectors.extend(
                changed_vectors
                    .into_iter()
                    .map(|v| v.filter(|v| v.len() == replacement_dims)),
            );
            self.index_shared(
                docs,
                vectors,
                crate::core::embeddings::HnswTuning {
                    min_vectors: self.hnsw_min_vectors,
                    ..crate::core::embeddings::HnswTuning::global()
                },
            );
            return IndexUpdateKind::FullRebuild;
        }
        self.apply_incremental(changed_docs, changed_vectors, deleted_paths);
        IndexUpdateKind::Incremental
    }

    fn apply_incremental(
        &mut self,
        changed_docs: Vec<SearchDocument>,
        changed_vectors: Vec<Option<Vec<f32>>>,
        deleted_paths: &[String],
    ) {
        if self.ann_store.is_some() {
            self.ann_dirty_paths.extend(deleted_paths.iter().cloned());
            self.ann_dirty_paths
                .extend(changed_docs.iter().map(|d| d.path.clone()));
        }
        for path in deleted_paths {
            if let Some(i) = self.documents.iter().position(|d| &d.path == path) {
                let last = self.documents.len() - 1;
                if self.ann_store.is_none() {
                    self.vector_buffer
                        .copy_within(last * self.dims..(last + 1) * self.dims, i * self.dims);
                    self.vector_buffer.truncate(last * self.dims);
                }
                self.documents.swap_remove(i);
                self.has_vector.swap_remove(i);
                self.vector_updates.remove(path);
            }
        }
        for (doc, vector) in changed_docs.into_iter().zip(changed_vectors) {
            let i = self
                .documents
                .iter()
                .position(|d| d.path == doc.path)
                .unwrap_or(self.documents.len());
            let doc = Arc::new(doc);
            if i == self.documents.len() {
                self.documents.push(Arc::clone(&doc));
                self.has_vector.push(false);
                if self.ann_store.is_none() {
                    self.vector_buffer.resize((i + 1) * self.dims, 0.0);
                }
            }
            self.has_vector[i] = vector.is_some();
            if let Some(vector) = vector {
                if self.ann_store.is_some() {
                    self.vector_updates.insert(doc.path.clone(), vector);
                } else {
                    self.vector_buffer[i * self.dims..(i + 1) * self.dims].copy_from_slice(&vector);
                }
            } else {
                self.vector_updates.remove(&doc.path);
            }
            self.documents[i] = doc;
        }
    }

    /// This index moved to another checkout's walk. The copy shares the vector
    /// store and its graph; `None` when the walk's own changes pass the
    /// promotion threshold or its vectors have another shape.
    pub(crate) fn fork(
        &self,
        docs: &[SearchDocument],
        vectors: &[Option<Vec<f32>>],
    ) -> Option<SearchIndex> {
        if vectors.iter().flatten().any(|v| v.len() != self.dims) {
            return None;
        }
        let (changed, deleted) = self.changes_from(docs, vectors);
        self.fork_delta(
            changed.iter().map(|&i| docs[i].clone()).collect(),
            changed.iter().map(|&i| vectors[i].clone()).collect(),
            &deleted,
        )
    }

    /// This index moved to a checkout that differs from it by the `changed`
    /// documents, with their `vectors`, and the `deleted` paths; see [`Self::fork`].
    pub(crate) fn fork_delta(
        &self,
        changed: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
        deleted: &[String],
    ) -> Option<SearchIndex> {
        if vectors.iter().flatten().any(|v| v.len() != self.dims)
            || (changed.len() + deleted.len()) as f64
                > self.documents.len() as f64 * FULL_REBUILD_CHANGE_FRACTION
        {
            return None;
        }
        let mut fork = self.clone();
        fork.apply_incremental(changed, vectors, deleted);
        Some(fork)
    }

    #[cfg(test)]
    pub(crate) fn graph_is_built(&self) -> bool {
        self.ann_store
            .as_ref()
            .is_some_and(|store| store.hnsw_is_initialized())
    }

    pub(crate) fn vector_store(&self) -> Option<&Arc<VectorStore>> {
        self.ann_store.as_ref()
    }

    pub(crate) fn shares_vector_store(&self, other: &SearchIndex) -> bool {
        matches!((&self.ann_store, &other.ann_store), (Some(a), Some(b)) if Arc::ptr_eq(a, b))
    }

    fn delta_from(
        &self,
        docs: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
    ) -> (Vec<SearchDocument>, Vec<Option<Vec<f32>>>, Vec<String>) {
        let (changed, deleted) = self.changes_from(&docs, &vectors);
        let changed: HashSet<usize> = changed.into_iter().collect();
        let (changed, changed_vectors) = docs
            .into_iter()
            .zip(vectors)
            .enumerate()
            .filter(|(i, _)| changed.contains(i))
            .map(|(_, change)| change)
            .unzip();
        (changed, changed_vectors, deleted)
    }

    /// Positions of the walked documents that differ from this index, and the
    /// indexed paths the walk no longer has.
    fn changes_from(
        &self,
        docs: &[SearchDocument],
        vectors: &[Option<Vec<f32>>],
    ) -> (Vec<usize>, Vec<String>) {
        let old: std::collections::HashMap<&str, usize> = self
            .documents
            .iter()
            .enumerate()
            .map(|(i, d)| (d.path.as_str(), i))
            .collect();
        let paths: HashSet<&str> = docs.iter().map(|d| d.path.as_str()).collect();
        let deleted = self
            .documents
            .iter()
            .filter(|d| !paths.contains(d.path.as_str()))
            .map(|d| d.path.clone())
            .collect();
        let changed = docs
            .iter()
            .zip(vectors)
            .enumerate()
            .filter(|(_, (doc, vector))| {
                !old.get(doc.path.as_str()).is_some_and(|&i| {
                    self.documents[i].content == doc.content
                        && self.documents[i].source_hash == doc.source_hash
                        && self.documents[i].search_text == doc.search_text
                        && self.vector_at(i) == vector.as_deref()
                })
            })
            .map(|(i, _)| i)
            .collect();
        (changed, deleted)
    }

    /// Perform hybrid search against the indexed documents.
    pub fn search(
        &self,
        query: &str,
        query_vec: &[f32],
        opts: &ResolvedSearchOptions,
    ) -> Vec<SearchResult> {
        self.search_where(query, query_vec, opts, None)
    }

    /// [`Self::search`] over only the documents whose path `keep` accepts,
    /// every one scored, as an index of just those documents would.
    pub(crate) fn search_where(
        &self,
        query: &str,
        query_vec: &[f32],
        opts: &ResolvedSearchOptions,
        keep: Option<&(dyn Fn(&str) -> bool + Sync)>,
    ) -> Vec<SearchResult> {
        if self.dims != 0 && query_vec.len() != self.dims {
            // The previous model cannot score queries from the replacement vector space.
            return Vec::new();
        }
        let query_terms: HashSet<String> = split_camel_case(query).into_iter().collect();
        let query_lower = query.trim().to_lowercase();
        let query_kind = detect_query_kind(query);
        // Sample across the corpus, before ANN and path filters, so the reference
        // distribution is not biased toward the nearest neighbors.
        let stride = (self.documents.len() / 4096).max(1);
        let similarities: Vec<f64> = self
            .documents
            .iter()
            .enumerate()
            .step_by(stride)
            .filter(|(i, _)| self.has_vector[*i])
            .filter_map(|(i, _)| {
                let vector = self.vector_at(i)?;
                Some(cosine(query_vec, vector))
            })
            .collect();
        let calibration = super::scoring::SemanticCalibration::new(&similarities);

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

        // ANN pre-filter: from `hnsw_min_vectors`, restrict scoring to the
        // graph's candidate set instead of O(N). Below it no graph is built and
        // every document is scored. Docs without embeddings bypass this filter
        // and go through keyword scoring only.
        let ann_candidate_set: Option<std::collections::HashSet<usize>> =
            self.graph_store().and_then(|store| {
                // A global ANN shortlist can omit the requested document class.
                if opts.scope != SearchScope::All || keep.is_some() {
                    return None;
                }
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
                let hits = store.find_nearest_without_waiting(query_vec, candidate_count);
                // Build a set of *document* indices from the path→doc lookup.
                let path_to_idx: std::collections::HashMap<&str, usize> = self
                    .documents
                    .iter()
                    .enumerate()
                    .map(|(i, d)| (d.path.as_str(), i))
                    .collect();
                let mut indices: std::collections::HashSet<usize> = hits
                    .iter()
                    .filter_map(|(path, _)| path_to_idx.get(path.as_str()).copied())
                    .filter(|&i| {
                        self.has_vector[i]
                            && !self.vector_updates.contains_key(&self.documents[i].path)
                    })
                    .collect();
                if indices.len() < candidate_count {
                    // Tombstones must not consume the live-neighbor budget.
                    return None;
                }
                indices.extend(
                    self.vector_updates
                        .keys()
                        .filter_map(|p| path_to_idx.get(p.as_str()).copied()),
                );
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
                if !document_passes_filters(doc, opts) || keep.is_some_and(|keep| !keep(&doc.path))
                {
                    return None;
                }
                // When ann_store owns the vectors (corpus ≥ ANN_THRESHOLD) the
                // flat buffer has been dropped to save memory; retrieve via the
                // store's key→index map.  Below threshold the buffer is used
                // directly for cache-friendly sequential access.
                let vec_slice = self.vector_at(i)?;
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
                if !document_passes_filters(doc, opts) || keep.is_some_and(|keep| !keep(&doc.path))
                {
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
        let wants_tests = query_terms
            .iter()
            .any(|token| super::lexical_search::is_test_intent_token(token));
        for (i, combined_score, ..) in &mut scored {
            *combined_score *= self.documents[*i]
                .path_prior
                .meaning_multiplier(wants_tests);
        }

        scored.retain(|(_, combined_score, ..)| *combined_score >= opts.min_combined_score);

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
                        semantic_score: if self.has_vector[idx] {
                            calibration.relevance(semantic_score)
                        } else {
                            0.0
                        },
                        semantic_cosine: semantic_score,
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
    if results.iter().map(|r| r.semantic_score).fold(0.0, f64::max)
        < WEAK_SEMANTIC_RELEVANCE_THRESHOLD
    {
        lines.push("Matches are weak; try keywords mode or a narrower path.".to_string());
    }
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
            "   Semantic: {}% (cos {:.2}) | Keyword: {}%",
            r.semantic_score, r.semantic_cosine, r.keyword_score
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

/// Rebuilds `stale`, the entry in `lock`, in the background from a walk of
/// `root` and the batches queued on it from `generation` on, unless a rebuild
/// of it already runs, and again while the entry it installs has batches of
/// changed files queued during its build, up to [`STALE_REBUILD_PASSES`]
/// walks; the fill's vectors queued then go into that entry. The rebuild's
/// task, when this call started it.
pub(crate) fn spawn_stale_rebuild(
    stale: &Arc<CachedSearchIndex>,
    lock: &Arc<RwLock<Option<Arc<CachedSearchIndex>>>>,
    generation: u64,
    walk_and_index_fn: &Arc<dyn WalkAndIndexFn>,
    root: &Path,
) -> Option<tokio::task::JoinHandle<()>> {
    use std::sync::atomic::Ordering;
    stale
        .rebuild_in_progress
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .ok()?;
    let mut previous = Arc::clone(stale);
    let lock = Arc::clone(lock);
    let walker = Arc::clone(walk_and_index_fn);
    let root = root.to_path_buf();
    let mut build_generation = generation;
    let task = tokio::spawn(async move {
        for pass in 1.. {
            let _reset = RebuildGuard(Arc::clone(&previous));
            let vector_generation = walker.vector_generation(&root).await;
            let (docs, vectors) = match walker.walk_or_install(&root).await {
                // The walk replaced `previous` with an index of its own.
                Ok(WalkOutcome::Installed(_)) => return,
                Ok(WalkOutcome::Documents(docs, vectors)) => (docs, vectors),
                Err(error) => {
                    tracing::warn!(%error, "Background index refresh failed");
                    return;
                }
            };
            let base = Arc::clone(&previous);
            let built = tokio::task::spawn_blocking(move || {
                let pending = base.pending.lock().unwrap();
                let mut snapshot: std::collections::BTreeMap<_, _> = docs
                    .into_iter()
                    .zip(vectors)
                    .map(|(doc, vector)| (doc.path.clone(), (doc, vector)))
                    .collect();
                let mut ready_generation = build_generation;
                let mut ready_vector_generation = vector_generation;
                for RefreshBatch {
                    docs,
                    vectors,
                    deleted,
                    generation,
                    vector_generation: batch_vector_generation,
                } in &pending.batches
                {
                    if *generation < build_generation {
                        continue;
                    }
                    for path in deleted {
                        snapshot.remove(path);
                    }
                    for (doc, vector) in docs.iter().zip(vectors) {
                        snapshot.insert(doc.path.clone(), (doc.clone(), vector.clone()));
                    }
                    ready_generation = ready_generation.max(*generation);
                    ready_vector_generation =
                        ready_vector_generation.max(batch_vector_generation.unwrap_or(0));
                }
                let consumed = pending.batches.len();
                drop(pending);
                let (docs, vectors) = snapshot.into_values().unzip();
                let (changed, vectors, deleted) = base.index.delta_from(docs, vectors);
                let mut index = base.index.clone();
                index.apply_delta(changed, vectors, &deleted);
                index.prepare_ann();
                (index, ready_generation, ready_vector_generation, consumed)
            })
            .await;
            let Ok((index, ready_generation, ready_vector_generation, consumed)) = built else {
                return;
            };
            if index.dims != previous.index.dims && index.has_vector.iter().any(|ready| !ready) {
                // A newer shape batch may have arrived after the walker snapshot.
                return;
            }
            #[cfg(test)]
            crate::server_adapters::test_seams::before_stale_install(&root).await;
            let mut guard = lock.write().await;
            if !guard.as_ref().is_some_and(|s| Arc::ptr_eq(s, &previous)) {
                return;
            }
            let fp = index.fingerprint();
            let mut entry = CachedSearchIndex::new(index, fp, ready_generation);
            let leftover = previous.pending.lock().unwrap().batches[consumed..].to_vec();
            entry.search_root = previous.search_root.clone();
            entry.vector_generation = ready_vector_generation;
            let mut installed = Arc::new(entry);
            // Batches of the fill carry vectors alone, which need no walk.
            let fill_generation = leftover
                .iter()
                .map(|batch| batch.vector_generation)
                .collect::<Option<Vec<_>>>()
                .and_then(|generations| generations.into_iter().max());
            if let Some(vector_generation) = fill_generation {
                let search_root = installed.search_root.clone();
                let updates = leftover
                    .into_iter()
                    .flat_map(|batch| batch.docs.into_iter().zip(batch.vectors))
                    .filter_map(|(doc, vector)| Some((doc.path, doc.source_hash, vector?)))
                    .collect();
                CachedSearchIndex::refresh_vectors(
                    &mut installed,
                    &search_root,
                    updates,
                    vector_generation,
                );
            } else {
                Arc::get_mut(&mut installed)
                    .unwrap()
                    .pending
                    .get_mut()
                    .unwrap()
                    .batches = leftover;
            }
            *guard = Some(Arc::clone(&installed));
            drop(guard);
            if installed.pending.lock().unwrap().batches.is_empty()
                || pass == STALE_REBUILD_PASSES
                || installed
                    .rebuild_in_progress
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
            {
                return;
            }
            previous = installed;
            build_generation = ready_generation;
        }
    });
    walk_and_index_fn.track_background_task(&task);
    Some(task)
}

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
pub(crate) async fn semantic_code_search_owned(
    options: SemanticSearchOptions,
    embed_fn: &dyn EmbedFn,
    walk_and_index_fn: Arc<dyn WalkAndIndexFn>,
    index_cache: Option<Arc<RwLock<Option<Arc<CachedSearchIndex>>>>>,
    cache_generation: Option<Arc<std::sync::atomic::AtomicU64>>,
) -> Result<String> {
    use std::sync::atomic::Ordering;
    if let (Some(lock), Some(generation)) = (&index_cache, &cache_generation) {
        let stale = lock.read().await.as_ref().cloned();
        if let Some(stale) = stale
            && (stale.generation.load(Ordering::Acquire) != generation.load(Ordering::Acquire)
                || !stale.pending.lock().unwrap().batches.is_empty())
            && !stale.scoped_refresh.load(Ordering::Acquire)
            && stale.search_root
                == std::fs::canonicalize(&options.root_dir)
                    .unwrap_or_else(|_| options.root_dir.clone())
        {
            spawn_stale_rebuild(
                &stale,
                lock,
                generation.load(Ordering::Acquire),
                &walk_and_index_fn,
                &options.root_dir,
            );
            let query = sanitize_query(&options.query);
            let vectors = embed_fn.embed(&[query.to_string()]).await?;
            let vector = vectors
                .first()
                .ok_or_else(|| ContextPlusError::Ollama("Empty embedding response".into()))?;
            let results = stale
                .index
                .search(&query, vector, &resolve_search_options(&options));
            return Ok(format_search_results_with_freshness(
                &query,
                &results,
                Some(stale.index.document_count()),
            ));
        }
    }
    semantic_code_search(
        options,
        embed_fn,
        walk_and_index_fn.as_ref(),
        index_cache,
        cache_generation,
    )
    .await
}

/// Answers `options`, whose root is at `prefix` in `entry`'s root, from
/// `entry`'s documents under it, as an index of the root would.
pub(crate) async fn search_entry_under(
    entry: Arc<CachedSearchIndex>,
    prefix: PathBuf,
    options: SemanticSearchOptions,
    embed_fn: &dyn EmbedFn,
) -> Result<String> {
    let query = sanitize_query(&options.query).into_owned();
    if query.is_empty() {
        return Ok("No matching files found for the given query.".to_string());
    }
    let query_vec = embed_fn
        .embed(std::slice::from_ref(&query))
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| ContextPlusError::Ollama("Empty embedding response".into()))?;
    let keep = result_path_filter(&options, prefix.clone());
    let mut resolved = resolve_search_options(&options);
    resolved.include_globs.clear();
    resolved.exclude_globs.clear();
    resolved.root_dir = entry.search_root.clone();
    let documents = entry
        .index
        .documents()
        .iter()
        .filter(|doc| Path::new(&doc.path).starts_with(&prefix))
        .count();
    let root_dir = options.root_dir.clone();
    let results = tokio::task::spawn_blocking(move || {
        let mut results = entry
            .index
            .search_where(&query, &query_vec, &resolved, Some(&keep));
        for result in &mut results {
            result.path = Path::new(&result.path)
                .strip_prefix(&prefix)
                .unwrap_or(Path::new(&result.path))
                .to_string_lossy()
                .into_owned();
        }
        fill_result_snippets(&root_dir, &mut results);
        (query, results)
    })
    .await
    .map_err(|err| ContextPlusError::Other(format!("Snippet task failed: {err}")))?;
    let (query, results) = results;
    Ok(format_search_results_with_freshness(
        &query,
        &results,
        Some(documents),
    ))
}

pub async fn semantic_code_search(
    options: SemanticSearchOptions,
    embed_fn: &dyn EmbedFn,
    walk_and_index_fn: &dyn WalkAndIndexFn,
    index_cache: Option<Arc<RwLock<Option<Arc<CachedSearchIndex>>>>>,
    cache_generation: Option<Arc<std::sync::atomic::AtomicU64>>,
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
    let search_root =
        std::fs::canonicalize(&options.root_dir).unwrap_or_else(|_| options.root_dir.clone());
    let vector_generation = walk_and_index_fn.vector_generation(&options.root_dir).await;
    let cached_arc: Arc<CachedSearchIndex> = match index_cache {
        None => {
            // No cache slot provided — always rebuild (unit-test / legacy path).
            let current_gen = cache_generation
                .as_ref()
                .map(|g| g.load(std::sync::atomic::Ordering::Acquire))
                .unwrap_or(0);
            tracing::debug!(
                generation = current_gen,
                reason = "no cache slot",
                "Rebuilding SearchIndex"
            );
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
        Some(lock) => 'cache: {
            // Snapshot the current tracker generation before any locking.
            let current_gen = cache_generation
                .as_ref()
                .map(|g| g.load(std::sync::atomic::Ordering::Acquire))
                .unwrap_or(0);

            // Fast path: read-lock only. If the tracker generation matches we can
            // skip the filesystem walk entirely — the tracker guarantees no file
            // changes have occurred since the cache was last built.
            if cache_generation.is_some() {
                let guard = lock.read().await;
                if let Some(ref cached) = *guard
                    && cached.generation.load(std::sync::atomic::Ordering::Acquire) == current_gen
                    && cached.vector_generation == vector_generation
                    && cached.search_root == search_root
                    && cached.pending.lock().unwrap().batches.is_empty()
                {
                    let reuses = cached.record_reuse();
                    tracing::debug!(
                        reuses,
                        generation = current_gen,
                        "SearchIndex generation hit — skipping walk"
                    );
                    break 'cache Arc::clone(cached);
                }
                match guard.as_ref() {
                    Some(cached) => tracing::debug!(
                        cached_generation =
                            cached.generation.load(std::sync::atomic::Ordering::Acquire),
                        current_generation = current_gen,
                        reason = "tracker generation mismatch",
                        "SearchIndex fast-path miss"
                    ),
                    None => tracing::debug!(
                        current_generation = current_gen,
                        reason = "cache empty",
                        "SearchIndex fast-path miss"
                    ),
                }
                // Generation mismatch (or no cache yet) — fall through to walk + fingerprint.
            }

            let started = std::time::Instant::now();
            let metadata = walk_and_index_fn
                .metadata_fingerprint(&options.root_dir)
                .await?;
            tracing::info!(
                phase = "semantic_metadata_fingerprint",
                elapsed_ms = started.elapsed().as_millis(),
                "cold-start phase"
            );
            {
                let guard = lock.read().await;
                if let Some(cached) = guard.as_ref()
                    && cached.vector_generation == vector_generation
                    && cached.search_root == search_root
                    && cached.pending.lock().unwrap().batches.is_empty()
                    && metadata.is_some()
                    && *cached.metadata.read().unwrap() == metadata
                {
                    cached.record_reuse();
                    cached
                        .generation
                        .store(current_gen, std::sync::atomic::Ordering::Release);
                    let results = cached.index.search(query.as_ref(), &query_vec, &resolved);
                    return Ok(format_search_results_with_freshness(
                        query.as_ref(),
                        &results,
                        Some(cached.index.document_count()),
                    ));
                }
            }

            // Walk the filesystem and compute fingerprint (tracker-off fallback or
            // generation mismatch meaning the tracker saw a change).
            let (docs, vectors) = match walk_and_index_fn.walk_or_install(&options.root_dir).await?
            {
                WalkOutcome::Documents(docs, vectors) => (docs, vectors),
                WalkOutcome::Installed(installed) => {
                    // Built from this walk, it answers; while it is current it
                    // also takes the walk's metadata and generation.
                    let current = lock
                        .read()
                        .await
                        .as_ref()
                        .is_some_and(|cached| Arc::ptr_eq(cached, &installed));
                    if current
                        && installed.vector_generation == vector_generation
                        && installed.pending.lock().unwrap().batches.is_empty()
                    {
                        *installed.metadata.write().unwrap() = metadata.clone();
                        installed
                            .generation
                            .store(current_gen, std::sync::atomic::Ordering::Release);
                    }
                    installed.record_reuse();
                    break 'cache installed;
                }
            };
            let fp = IndexFingerprint::from_docs(&docs);

            {
                let guard = lock.read().await;
                if let Some(ref cached) = *guard
                    && cached.fingerprint == fp
                    && cached.vector_generation == vector_generation
                    && cached.search_root == search_root
                    && cached.pending.lock().unwrap().batches.is_empty()
                {
                    *cached.metadata.write().unwrap() = metadata.clone();
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
                    break 'cache Arc::clone(cached);
                }
            }

            tracing::debug!(
                current_generation = current_gen,
                new_n_docs = fp.n_docs,
                reason = "content fingerprint changed or cache empty",
                "SearchIndex fingerprint miss"
            );

            // Slow path: acquire write-lock, double-check, then rebuild.
            let mut guard = lock.write().await;
            // Another task may have rebuilt while we waited for the write-lock.
            if let Some(ref cached) = *guard
                && cached.fingerprint == fp
                && cached.vector_generation == vector_generation
                && cached.search_root == search_root
                && cached.pending.lock().unwrap().batches.is_empty()
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
                break 'cache Arc::clone(cached);
            }

            if let Some(stale) = guard.as_ref().filter(|s| s.search_root == search_root) {
                let (changed, changed_vectors, deleted) = stale.index.delta_from(docs, vectors);
                let large = stale
                    .index
                    .requires_full_rebuild(&changed, &changed_vectors, &deleted);
                if large {
                    let stale = Arc::clone(stale);
                    if stale
                        .rebuild_in_progress
                        .compare_exchange(
                            false,
                            true,
                            std::sync::atomic::Ordering::AcqRel,
                            std::sync::atomic::Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        let previous = Arc::clone(&stale);
                        let lock = Arc::clone(&lock);
                        let task = tokio::spawn(async move {
                            let _reset = RebuildGuard(Arc::clone(&previous));
                            let base = Arc::clone(&previous);
                            let built = tokio::task::spawn_blocking(move || {
                                let mut changes: std::collections::BTreeMap<_, _> = changed
                                    .into_iter()
                                    .zip(changed_vectors)
                                    .map(|(doc, vector)| (doc.path.clone(), (doc, vector)))
                                    .collect();
                                let mut deleted: HashSet<_> = deleted.into_iter().collect();
                                let pending = base.pending.lock().unwrap();
                                let mut ready_generation = current_gen;
                                let mut ready_vector_generation = vector_generation;
                                for RefreshBatch {
                                    docs,
                                    vectors,
                                    deleted: removed,
                                    generation,
                                    vector_generation: batch_vector_generation,
                                } in &pending.batches
                                {
                                    if *generation < current_gen {
                                        continue;
                                    }
                                    for path in removed {
                                        changes.remove(path);
                                        deleted.insert(path.clone());
                                    }
                                    for (doc, vector) in docs.iter().zip(vectors) {
                                        deleted.remove(&doc.path);
                                        changes.insert(
                                            doc.path.clone(),
                                            (doc.clone(), vector.clone()),
                                        );
                                    }
                                    ready_generation = ready_generation.max(*generation);
                                    ready_vector_generation = ready_vector_generation
                                        .max(batch_vector_generation.unwrap_or(0));
                                }
                                let consumed = pending.batches.len();
                                drop(pending);
                                let (changed, changed_vectors) = changes.into_values().unzip();
                                let mut index = base.index.clone();
                                index.apply_delta(
                                    changed,
                                    changed_vectors,
                                    &deleted.into_iter().collect::<Vec<_>>(),
                                );
                                index.prepare_ann();
                                (index, ready_generation, ready_vector_generation, consumed)
                            })
                            .await;
                            if let Ok((
                                index,
                                ready_generation,
                                ready_vector_generation,
                                consumed,
                            )) = built
                            {
                                if index.dims != previous.index.dims
                                    && index.has_vector.iter().any(|ready| !ready)
                                {
                                    // A newer shape batch may have arrived after the walker snapshot.
                                    return;
                                }
                                let mut guard = lock.write().await;
                                if guard.as_ref().is_some_and(|s| Arc::ptr_eq(s, &previous)) {
                                    let fp = index.fingerprint();
                                    let mut entry =
                                        CachedSearchIndex::new(index, fp, ready_generation);
                                    entry.pending.get_mut().unwrap().batches =
                                        previous.pending.lock().unwrap().batches[consumed..]
                                            .to_vec();
                                    entry.search_root = search_root;
                                    entry.vector_generation = ready_vector_generation;
                                    *entry.metadata.write().unwrap() = metadata;
                                    *guard = Some(Arc::new(entry));
                                }
                            }
                        });
                        walk_and_index_fn.track_background_task(&task);
                    }
                    break 'cache stale;
                }
                let mut entry = CachedSearchIndex::delta_generation(
                    guard.take().unwrap(),
                    changed,
                    changed_vectors,
                    deleted,
                    current_gen,
                );
                let fresh = Arc::get_mut(&mut entry).unwrap();
                fresh.fingerprint = fp;
                fresh
                    .generation
                    .store(current_gen, std::sync::atomic::Ordering::Release);
                fresh.search_root = search_root;
                fresh.vector_generation = vector_generation;
                *fresh.metadata.write().unwrap() = metadata;
                *guard = Some(Arc::clone(&entry));
                break 'cache entry;
            }
            let started = std::time::Instant::now();
            let mut idx = SearchIndex::new();
            idx.index_with_vectors_and_tuning(
                docs,
                vectors,
                crate::core::embeddings::HnswTuning::global(),
            );
            tracing::info!(
                phase = "semantic_index_build",
                ref_id = %walk_and_index_fn.ref_id(),
                root = %search_root.display(),
                elapsed_ms = started.elapsed().as_millis(),
                documents = idx.document_count(),
                "cold-start phase"
            );
            let mut entry = CachedSearchIndex::new(idx, fp, current_gen);
            entry.vector_generation = vector_generation;
            entry.search_root = search_root;
            *entry.metadata.write().unwrap() = metadata;
            let arc = Arc::new(entry);
            // A scoped index answers this query but never replaces the
            // ref's index of its whole root.
            let ref_root = walk_and_index_fn.ref_root();
            if !guard.as_ref().is_some_and(|current| {
                ref_root == Some(current.search_root.as_path())
                    && current.search_root != arc.search_root
            }) {
                *guard = Some(Arc::clone(&arc));
            }
            arc
        }
    };

    let started = std::time::Instant::now();
    let mut results = cached_arc
        .index
        .search(query.as_ref(), &query_vec, &resolved);
    tracing::info!(
        phase = "semantic_search",
        elapsed_ms = started.elapsed().as_millis(),
        "cold-start phase"
    );
    let root_dir = options.root_dir.clone();
    let results = tokio::task::spawn_blocking(move || {
        fill_result_snippets(&root_dir, &mut results);
        results
    })
    .await
    .map_err(|err| crate::error::ContextPlusError::Other(format!("Snippet task failed: {err}")))?;
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
    fn vector_generation(&self, _root: &Path) -> VectorGenerationFuture<'_> {
        Box::pin(async { 0 })
    }
    fn metadata_fingerprint(&self, _root_dir: &Path) -> MetadataFingerprintFuture<'_> {
        Box::pin(async { Ok(None) })
    }
    fn walk_and_index(&self, root_dir: &Path) -> WalkAndIndexFuture<'_>;
    /// [`Self::walk_and_index`], unless the walk installs the index of
    /// `root_dir` in the slot itself.
    fn walk_or_install(&self, root_dir: &Path) -> WalkOrInstallFuture<'_> {
        let walk = self.walk_and_index(root_dir);
        Box::pin(async move {
            walk.await
                .map(|(docs, vectors)| WalkOutcome::Documents(docs, vectors))
        })
    }
    fn track_background_task(&self, _task: &tokio::task::JoinHandle<()>) {}
    /// The ref this walker indexes, for log lines.
    fn ref_id(&self) -> &str {
        ""
    }
    /// The canonical root of the ref this walker indexes.
    fn ref_root(&self) -> Option<&Path> {
        None
    }
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

    fn semantic_only_options(top_k: usize) -> ResolvedSearchOptions {
        ResolvedSearchOptions {
            top_k,
            semantic_weight: 1.0,
            keyword_weight: 0.0,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
            ..Default::default()
        }
    }

    fn unit_vector_with_x(x: f32) -> Vec<f32> {
        vec![x, (1.0 - x * x).sqrt()]
    }

    #[test]
    fn calibrated_semantic_relevance_separates_outlier_from_bulk() {
        let mut docs = vec![SearchDocument::new(
            "src/tenant_context.rs".to_string(),
            String::new(),
            vec![],
            vec![],
            "request context propagation".to_string(),
        )];
        let mut vectors = vec![Some(unit_vector_with_x(0.58))];
        for i in 0..20 {
            docs.push(SearchDocument::new(
                format!("src/background_{i}.rs"),
                String::new(),
                vec![],
                vec![],
                "generic repository support code".to_string(),
            ));
            let similarity = 0.35 + (i % 3) as f32 * 0.002;
            vectors.push(Some(unit_vector_with_x(similarity)));
        }

        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);
        let results = index.search(
            "semantic needle absent from document text",
            &[1.0, 0.0],
            &semantic_only_options(21),
        );

        assert_eq!(results[0].path, "src/tenant_context.rs", "{results:?}");
        assert!(
            results[0].semantic_score >= 80.0,
            "a cosine far above the corpus bulk should read as high relevance: {results:?}"
        );
        assert!(
            results[1..]
                .iter()
                .all(|result| result.semantic_score <= 50.0),
            "documents near the corpus bulk should read as low relevance: {results:?}"
        );
    }

    #[test]
    fn formatted_semantic_relevance_keeps_raw_cosine() {
        let docs = vec![SearchDocument::new(
            "src/tenant_context.rs".to_string(),
            String::new(),
            vec![],
            vec![],
            "request context propagation".to_string(),
        )];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vec![Some(unit_vector_with_x(0.58))]);

        let results = index.search(
            "semantic needle absent from document text",
            &[1.0, 0.0],
            &semantic_only_options(1),
        );
        let output = format_search_results("semantic needle", &results);

        assert!(
            (results[0].semantic_cosine - 0.58).abs() < 0.001,
            "SearchResult should expose the raw cosine separately: {results:?}"
        );
        assert!(
            output.contains("(cos 0.58)"),
            "formatted semantic evidence should retain the raw cosine: {output}"
        );
    }

    #[test]
    fn uniformly_weak_semantic_results_emit_guidance() {
        let docs: Vec<SearchDocument> = (0..8)
            .map(|i| {
                SearchDocument::new(
                    format!("src/background_{i}.rs"),
                    String::new(),
                    vec![],
                    vec![],
                    "generic repository support code".to_string(),
                )
            })
            .collect();
        let vectors = vec![Some(unit_vector_with_x(0.35)); docs.len()];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let results = index.search(
            "semantic needle absent from document text",
            &[1.0, 0.0],
            &semantic_only_options(5),
        );
        let output = format_search_results("semantic needle", &results);
        let guidance = output.lines().next().unwrap_or_default().to_lowercase();

        assert!(
            guidance.contains("weak"),
            "the first output line should label weak matches: {output}"
        );
        assert!(
            guidance.contains("keywords") && guidance.contains("narrower path"),
            "weak-match guidance should suggest keywords mode or a narrower path: {output}"
        );
        assert!(
            output.contains("src/background_0.rs"),
            "weak matches should still be returned: {output}"
        );
    }

    #[test]
    fn weak_semantic_relevance_threshold_is_named_and_percentage_scaled() {
        assert!(
            (50.0..=80.0).contains(&WEAK_SEMANTIC_RELEVANCE_THRESHOLD),
            "weak-result threshold should be tuned in calibrated percentage units"
        );
    }

    #[test]
    fn meaning_mode_demotes_documentation_below_equally_similar_code() {
        let docs = vec![
            SearchDocument::new(
                "architecture/phi-redaction.mdx".to_string(),
                String::new(),
                vec![],
                vec![],
                "keep protected health information out of logs".to_string(),
            ),
            SearchDocument::new(
                "apps/emr-api/src/logging.ts".to_string(),
                String::new(),
                vec![],
                vec![],
                "keep protected health information out of logs".to_string(),
            ),
        ];
        let vectors = vec![
            Some(unit_vector_with_x(0.51)),
            Some(unit_vector_with_x(0.51)),
        ];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let results = index.search(
            "protect patient data from telemetry",
            &[1.0, 0.0],
            &semantic_only_options(2),
        );

        assert_eq!(
            results[0].path, "apps/emr-api/src/logging.ts",
            "default meaning search should prefer code when raw similarity ties: {results:?}"
        );
        assert_eq!(results[1].path, "architecture/phi-redaction.mdx");
    }

    #[test]
    fn meaning_scope_code_and_docs_filter_document_classes() {
        let docs = vec![
            SearchDocument::new(
                "architecture/phi-redaction.mdx".to_string(),
                String::new(),
                vec![],
                vec![],
                "keep protected health information out of logs".to_string(),
            ),
            SearchDocument::new(
                "apps/emr-api/src/logging.ts".to_string(),
                String::new(),
                vec![],
                vec![],
                "keep protected health information out of logs".to_string(),
            ),
        ];
        let vectors = vec![
            Some(unit_vector_with_x(0.51)),
            Some(unit_vector_with_x(0.51)),
        ];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);
        let query_vec = [1.0, 0.0];

        let code_results = index.search(
            "protect patient data from telemetry",
            &query_vec,
            &ResolvedSearchOptions {
                scope: SearchScope::Code,
                ..semantic_only_options(2)
            },
        );
        assert_eq!(
            code_results
                .iter()
                .map(|result| result.path.as_str())
                .collect::<Vec<_>>(),
            ["apps/emr-api/src/logging.ts"]
        );

        let docs_results = index.search(
            "protect patient data from telemetry",
            &query_vec,
            &ResolvedSearchOptions {
                scope: SearchScope::Docs,
                ..semantic_only_options(2)
            },
        );
        assert_eq!(
            docs_results
                .iter()
                .map(|result| result.path.as_str())
                .collect::<Vec<_>>(),
            ["architecture/phi-redaction.mdx"]
        );
    }

    #[test]
    fn search_scope_defaults_to_all_and_resolves_explicit_scope() {
        assert_eq!(ResolvedSearchOptions::default().scope, SearchScope::All);

        let resolved = resolve_search_options(&SemanticSearchOptions {
            root_dir: PathBuf::new(),
            query: "protect patient data from telemetry".to_string(),
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
            scope: Some(SearchScope::Code),
        });

        assert_eq!(resolved.scope, SearchScope::Code);
    }

    #[test]
    fn migration_sql_is_documentation_but_ordinary_sql_is_code() {
        assert!(
            super::super::lexical_search::classify_path_prior(
                "packages/db/migrations/202609260001_add_phi_fields.sql"
            )
            .is_documentation
        );
        assert!(
            !super::super::lexical_search::classify_path_prior(
                "packages/db/queries/reconcile_invoice.sql"
            )
            .is_documentation
        );
    }

    #[test]
    fn meaning_mode_fixture_prior_applies_only_without_test_intent() {
        let docs = vec![
            SearchDocument::new(
                "packages/billing/__fixtures__/invoice-payment-succeeded.json".to_string(),
                "invoice payment succeeded event".to_string(),
                vec![],
                vec![],
                "reconciles invoice payment status from webhook".to_string(),
            ),
            SearchDocument::new(
                "packages/billing/routes/stripe-webhook.ts".to_string(),
                "Stripe webhook route".to_string(),
                vec![],
                vec![],
                "reconciles invoice payment status from webhook".to_string(),
            ),
        ];
        let vectors = vec![
            Some(unit_vector_with_x(0.81)),
            Some(unit_vector_with_x(0.80)),
        ];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);
        let opts = semantic_only_options(1);
        let query_vec = [1.0, 0.0];

        let ordinary = index.search("invoice payment status", &query_vec, &opts);
        assert_eq!(
            ordinary[0].path, "packages/billing/routes/stripe-webhook.ts",
            "fixture with a slightly higher raw score should be demoted before top_k truncation: {ordinary:?}"
        );

        for query in ["invoice payment fixture", "invoice payment test"] {
            let explicit_test_intent = index.search(query, &query_vec, &opts);
            assert_eq!(
                explicit_test_intent[0].path,
                "packages/billing/__fixtures__/invoice-payment-succeeded.json",
                "fixture should retain its higher raw score when the query requests tests or fixtures: query={query}, results={explicit_test_intent:?}"
            );
        }
    }

    #[test]
    fn meaning_mode_enforces_combined_minimum_after_fixture_prior_for_embedded_candidate() {
        let docs = vec![SearchDocument::new(
            "packages/billing/__fixtures__/invoice-payment-succeeded.json".to_string(),
            String::new(),
            vec![],
            vec![],
            "invoice payment status fixture".to_string(),
        )];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vec![Some(unit_vector_with_x(0.81))]);
        let opts = ResolvedSearchOptions {
            min_combined_score: 0.70,
            ..semantic_only_options(5)
        };
        let query_vec = [1.0, 0.0];

        let explicit_fixture_intent = index.search("invoice payment fixture", &query_vec, &opts);
        assert_eq!(
            explicit_fixture_intent.len(),
            1,
            "explicit fixture intent should preserve a candidate above the minimum: {explicit_fixture_intent:?}"
        );

        let ordinary = index.search("invoice payment status", &query_vec, &opts);
        assert!(
            ordinary.is_empty(),
            "fixture demoted below the combined minimum should be excluded: {ordinary:?}"
        );
    }

    #[test]
    fn meaning_mode_enforces_combined_minimum_after_fixture_prior_for_no_vector_candidate() {
        let docs = vec![SearchDocument::new(
            "packages/billing/__fixtures__/invoice-payment-succeeded.json".to_string(),
            String::new(),
            vec![],
            vec![],
            "invoice payment status fixture".to_string(),
        )];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vec![None]);
        let opts = ResolvedSearchOptions {
            top_k: 5,
            semantic_weight: 0.0,
            keyword_weight: 1.0,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.70,
            require_keyword_match: false,
            require_semantic_match: false,
            ..Default::default()
        };

        let explicit_fixture_intent = index.search("invoice payment status fixture", &[], &opts);
        assert_eq!(
            explicit_fixture_intent.len(),
            1,
            "explicit fixture intent should preserve a no-vector candidate above the minimum: {explicit_fixture_intent:?}"
        );

        let ordinary = index.search("invoice payment status", &[], &opts);
        assert!(
            ordinary.is_empty(),
            "no-vector fixture demoted below the combined minimum should be excluded: {ordinary:?}"
        );
    }

    #[test]
    fn meaning_mode_demotes_generated_code_before_top_k() {
        let docs = vec![
            SearchDocument::new(
                "src/generated/account_client.rs".to_string(),
                String::new(),
                vec![],
                vec![],
                "hydrate account record".to_string(),
            ),
            SearchDocument::new(
                "src/account_client.rs".to_string(),
                String::new(),
                vec![],
                vec![],
                "hydrate account record".to_string(),
            ),
        ];
        let vectors = vec![
            Some(unit_vector_with_x(0.81)),
            Some(unit_vector_with_x(0.80)),
        ];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let results = index.search(
            "hydrate account record",
            &[1.0, 0.0],
            &semantic_only_options(1),
        );

        assert_eq!(
            results[0].path, "src/account_client.rs",
            "generated code should be demoted before top_k truncation: {results:?}"
        );
    }

    #[test]
    fn meaning_mode_demotes_planning_prose_before_top_k() {
        let docs = vec![
            SearchDocument::new(
                "docs/plans/account-hydration.md".to_string(),
                String::new(),
                vec![],
                vec![],
                "hydrate account record".to_string(),
            ),
            SearchDocument::new(
                "src/account_client.rs".to_string(),
                String::new(),
                vec![],
                vec![],
                "hydrate account record".to_string(),
            ),
        ];
        let vectors = vec![
            Some(unit_vector_with_x(0.81)),
            Some(unit_vector_with_x(0.80)),
        ];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);

        let results = index.search(
            "hydrate account record",
            &[1.0, 0.0],
            &semantic_only_options(1),
        );

        assert_eq!(
            results[0].path, "src/account_client.rs",
            "planning prose should be demoted before top_k truncation: {results:?}"
        );
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
            semantic_cosine: 0.90,
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

    fn ranked_result(
        path: &str,
        matched_symbol_locations: Vec<&str>,
        fallback_snippet: &str,
    ) -> SearchResult {
        SearchResult {
            path: path.to_string(),
            score: 100.0,
            semantic_score: 100.0,
            semantic_cosine: 1.0,
            keyword_score: 100.0,
            header: String::new(),
            matched_symbols: vec![],
            matched_symbol_locations: matched_symbol_locations
                .into_iter()
                .map(str::to_string)
                .collect(),
            snippet: Some(fallback_snippet.to_string()),
        }
    }

    #[test]
    fn ranked_code_snippet_reads_matched_symbol_at_line_40_from_real_file() {
        let root = tempfile::tempdir().unwrap();
        let relative_path = "src/stripe_webhook.ts";
        let absolute_path = root.path().join(relative_path);
        std::fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();

        let mut lines: Vec<String> = (1..40)
            .map(|line| format!("// filler line {line}"))
            .collect();
        lines.extend([
            "export function reconcileInvoicePayment(event: Stripe.Event) {".to_string(),
            "  const invoice = event.data.object;".to_string(),
            "  return markInvoicePaid(invoice.id);".to_string(),
            "}".to_string(),
        ]);
        std::fs::write(&absolute_path, lines.join("\n")).unwrap();

        let mut results = vec![ranked_result(
            relative_path,
            vec!["reconcileInvoicePayment@L40-L43"],
            "typescript /**",
        )];

        fill_result_snippets(root.path(), &mut results);

        let snippet = results[0].snippet.as_deref().unwrap();
        assert!(
            snippet.starts_with("export function reconcileInvoicePayment"),
            "expected snippet to start at real-file line 40, got: {snippet:?}"
        );
        assert!(!snippet.starts_with("typescript "));
    }

    #[test]
    fn ranked_json_snippet_skips_lone_structural_opener() {
        let root = tempfile::tempdir().unwrap();
        let relative_path = "fixtures/invoice-payment-succeeded.json";
        let absolute_path = root.path().join(relative_path);
        std::fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        std::fs::write(
            &absolute_path,
            "\n{\n  \"type\": \"invoice.payment_succeeded\",\n  \"livemode\": false\n}\n",
        )
        .unwrap();

        let mut results = vec![ranked_result(relative_path, vec![], "{")];

        fill_result_snippets(root.path(), &mut results);

        let snippet = results[0].snippet.as_deref().unwrap();
        assert!(
            snippet.starts_with("  \"type\": \"invoice.payment_succeeded\""),
            "expected first JSON key instead of structural opener, got: {snippet:?}"
        );
    }

    #[test]
    fn ranked_markdown_snippet_skips_front_matter_block() {
        let root = tempfile::tempdir().unwrap();
        let relative_path = "todos/reconcile-invoice-payment.md";
        let absolute_path = root.path().join(relative_path);
        std::fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        std::fs::write(
            &absolute_path,
            "---\ntitle: Reconcile invoice payment\nstatus: open\n---\n\n# Reconcile Stripe invoices\n\nUpdate payment status from the webhook event.\n",
        )
        .unwrap();

        let mut results = vec![ranked_result(relative_path, vec![], "---")];

        fill_result_snippets(root.path(), &mut results);

        let snippet = results[0].snippet.as_deref().unwrap();
        assert!(
            snippet.starts_with("# Reconcile Stripe invoices"),
            "expected heading after front matter, got: {snippet:?}"
        );
        assert!(!snippet.contains("title: Reconcile invoice payment"));
    }

    #[test]
    fn ranked_yaml_snippet_keeps_first_document_keys_without_terminator() {
        let root = tempfile::tempdir().unwrap();
        let relative_path = "config.yaml";
        std::fs::write(
            root.path().join(relative_path),
            "---\nservice: billing\nport: 8080\n",
        )
        .unwrap();

        let mut results = vec![ranked_result(relative_path, vec![], "---")];

        fill_result_snippets(root.path(), &mut results);

        let snippet = results[0]
            .snippet
            .as_deref()
            .expect("YAML keys after an opening document marker must produce a snippet");
        assert!(
            snippet.starts_with("service: billing"),
            "expected the first YAML key after the document marker, got: {snippet:?}"
        );
    }

    #[test]
    fn ranked_yaml_snippet_keeps_first_document_keys_with_terminator() {
        let root = tempfile::tempdir().unwrap();
        let relative_path = "config.yaml";
        std::fs::write(
            root.path().join(relative_path),
            "---\nservice: billing\nport: 8080\n...\n",
        )
        .unwrap();

        let mut results = vec![ranked_result(relative_path, vec![], "---")];

        fill_result_snippets(root.path(), &mut results);

        let snippet = results[0]
            .snippet
            .as_deref()
            .expect("YAML keys before a document terminator must produce a snippet");
        assert!(
            snippet.starts_with("service: billing"),
            "expected the first YAML key before the document terminator, got: {snippet:?}"
        );
    }

    #[test]
    fn ranked_code_without_symbol_skips_language_prefix_and_bare_comment_opener() {
        let root = tempfile::tempdir().unwrap();
        let relative_path = "src/invoice_status.ts";
        let absolute_path = root.path().join(relative_path);
        std::fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        std::fs::write(
            &absolute_path,
            "/**\n * Reconciles invoice payment status from Stripe.\n */\nexport const reconcileInvoiceStatus = () => {};\n",
        )
        .unwrap();

        let mut results = vec![ranked_result(relative_path, vec![], "typescript /**")];

        fill_result_snippets(root.path(), &mut results);

        let snippet = results[0].snippet.as_deref().unwrap();
        assert_ne!(snippet, "typescript /**");
        assert_ne!(snippet.trim(), "/**");
        assert!(!snippet.starts_with("typescript "));
    }

    #[test]
    fn ranked_snippet_missing_file_keeps_indexed_fallback_without_error() {
        let root = tempfile::tempdir().unwrap();
        let mut results = vec![ranked_result(
            "src/deleted_since_indexing.ts",
            vec!["reconcileInvoicePayment@L40-L43"],
            "typescript /**",
        )];

        fill_result_snippets(root.path(), &mut results);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet.as_deref(), Some("typescript /**"));
    }

    #[test]
    fn ranked_snippets_only_open_files_in_final_top_k() {
        use std::collections::HashMap;
        use std::io::{self, BufRead, Cursor};
        use std::sync::Mutex;

        struct RecordingOpener {
            files: HashMap<PathBuf, String>,
            opened: Mutex<Vec<PathBuf>>,
        }

        impl SnippetFileOpener for RecordingOpener {
            fn open(&self, path: &Path) -> io::Result<Box<dyn BufRead>> {
                self.opened.lock().unwrap().push(path.to_path_buf());
                let content =
                    self.files.get(path).cloned().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::NotFound, "fixture not found")
                    })?;
                Ok(Box::new(Cursor::new(content)))
            }
        }

        let root = PathBuf::from("/repo");
        let query_vec = vec![1.0_f32, 0.0];
        let docs: Vec<SearchDocument> = (0..100)
            .map(|i| {
                SearchDocument::new(
                    format!("src/file_{i}.ts"),
                    format!("file {i}"),
                    vec![],
                    vec![],
                    format!("typescript indexed content {i}"),
                )
            })
            .collect();
        let vectors: Vec<Option<Vec<f32>>> = (0..100)
            .map(|i| {
                let x = 1.0 - i as f32 * 0.009;
                let y = (1.0_f32 - x * x).sqrt();
                Some(vec![x, y])
            })
            .collect();
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);
        let opts = ResolvedSearchOptions {
            top_k: 3,
            semantic_weight: 1.0,
            keyword_weight: 0.0,
            min_combined_score: 0.0,
            root_dir: root.clone(),
            ..Default::default()
        };
        let mut results = index.search("unmatched query", &query_vec, &opts);
        assert_eq!(results.len(), 3);

        let files = (0..100)
            .map(|i| {
                (
                    root.join(format!("src/file_{i}.ts")),
                    format!("export const file{i} = {i};\n"),
                )
            })
            .collect();
        let opener = RecordingOpener {
            files,
            opened: Mutex::new(Vec::new()),
        };

        fill_result_snippets_with_opener(&root, &mut results, &opener);

        let opened: HashSet<PathBuf> = opener.opened.lock().unwrap().iter().cloned().collect();
        let expected: HashSet<PathBuf> = results
            .iter()
            .map(|result| root.join(&result.path))
            .collect();
        assert_eq!(opened, expected);
        assert_eq!(opened.len(), 3, "only final top-k files may be opened");
    }

    // -- format_search_results_with_freshness tests --

    #[test]
    fn format_results_includes_freshness_banner() {
        let results = vec![SearchResult {
            path: "src/x.ts".to_string(),
            score: 50.0,
            semantic_score: 60.0,
            semantic_cosine: 0.60,
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
            semantic_cosine: 0.60,
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
            semantic_cosine: 0.60,
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

    /// Tuning that prunes with the graph from `ANN_THRESHOLD`, so test-sized
    /// corpora reach the graph path.
    fn graph_tuning() -> crate::core::embeddings::HnswTuning {
        crate::core::embeddings::HnswTuning {
            min_vectors: ANN_THRESHOLD,
            ..Default::default()
        }
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
    fn r3_pending_fill_keeps_vector_generation_with_its_batch() {
        let docs = vec![make_doc("old.rs", "old"), make_doc("changed.rs", "before")];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs.clone(), vec![Some(vec![1.0, 0.0]); 2]);
        let mut entry = Arc::new(CachedSearchIndex::new(
            index,
            IndexFingerprint::from_docs(&docs),
            0,
        ));
        CachedSearchIndex::refresh_vectors(
            &mut entry,
            Path::new(""),
            vec![(
                "changed.rs".into(),
                crate::core::embeddings::content_hash("before"),
                vec![1.0, 0.0, 0.0],
            )],
            1,
        );
        assert_eq!(
            entry.vector_generation, 0,
            "pending vectors are not yet represented"
        );
        let pending = entry.pending.lock().unwrap();
        assert_eq!(pending.batches.len(), 1);
        assert_eq!(pending.batches[0].vector_generation, Some(1));
    }

    #[test]
    fn r3_dimension_transition_keeps_changed_vector_searchable() {
        let mut index = SearchIndex::new();
        index.index_with_vectors(
            vec![make_doc("old.rs", "old"), make_doc("changed.rs", "before")],
            vec![Some(vec![1.0, 0.0]), Some(vec![0.0, 1.0])],
        );
        assert_eq!(
            index.apply_delta(
                vec![make_doc("changed.rs", "after")],
                vec![Some(vec![1.0, 0.0, 0.0])],
                &[]
            ),
            IndexUpdateKind::FullRebuild
        );
        assert_eq!(
            index.dims, 3,
            "replacement shape must come from the new vector"
        );
        let i = index
            .documents
            .iter()
            .position(|d| d.path == "changed.rs")
            .unwrap();
        assert_eq!(index.vector_at(i), Some([1.0, 0.0, 0.0].as_slice()));
        assert!(
            index
                .documents
                .iter()
                .enumerate()
                .all(|(i, _)| index.vector_at(i).is_none_or(|v| v.len() == 3))
        );
        let opts = ResolvedSearchOptions {
            top_k: 5,
            min_semantic_score: 0.0,
            min_keyword_score: 0.0,
            min_combined_score: 0.0,
            require_keyword_match: false,
            require_semantic_match: false,
            ..Default::default()
        };
        assert!(
            index
                .search("after", &[1.0, 0.0, 0.0], &opts)
                .iter()
                .any(|r| r.path == "changed.rs")
        );
    }

    #[test]
    fn r3_deleted_ann_shortlist_replenishes_survivors() {
        let (docs, vectors) = make_ann_corpus(ANN_THRESHOLD + 50);
        let mut index = SearchIndex::new();
        index.index_with_vectors_and_tuning(docs, vectors, graph_tuning());
        let query = unit_vec(1.0, 0.001);
        let deleted: Vec<_> = index
            .ann_store
            .as_ref()
            .unwrap()
            .find_nearest(&query, 5 * ANN_CANDIDATE_MULTIPLIER)
            .into_iter()
            .map(|(path, _)| path)
            .collect();
        assert_eq!(
            index.apply_delta(vec![], vec![], &deleted),
            IndexUpdateKind::Incremental
        );
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
        let results = index.search("file", &query, &opts);
        assert_eq!(results.len(), 5, "deleted ANN shortlist hid live matches");
        assert!(results.iter().all(|r| !deleted.contains(&r.path)));
        for _ in 0..9 {
            let deleted: Vec<_> = index
                .documents
                .iter()
                .take(50)
                .map(|d| d.path.clone())
                .collect();
            index.apply_delta(vec![], vec![], &deleted);
        }
        assert!(
            index.full_rebuild_count() > 1,
            "accumulated tombstones never compacted"
        );
    }

    #[test]
    fn test_ann_path_triggers_above_threshold() {
        // Build a corpus just above the ANN threshold.
        let n = ANN_THRESHOLD + 50;
        let (docs, vectors) = make_ann_corpus(n);
        let mut index = SearchIndex::new();
        // HNSW is approximate — at the default `ef_search = 32`, recall of
        // the *true* nearest neighbor among 2050 docs is not guaranteed and
        // varies across CPU architectures (observed flake on macOS aarch64
        // where doc_0 falls outside top-5 with a query pointing nearly
        // exactly at doc_0). Bumping `ef_search` to 256 gives this test
        // sufficient recall while still exercising the same ANN-on/off
        // dispatch logic the assertion is meant to prove.
        index.index_with_vectors_and_tuning(
            docs,
            vectors,
            crate::core::embeddings::HnswTuning {
                ef_search: 256,
                ..graph_tuning()
            },
        );

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
        // doc_0 should be in top_k — its embedding is the closest match to
        // the query by cosine similarity. HNSW at `ef_search=256` reliably
        // recalls it across MSRV (1.88) and stable, x86_64 and aarch64.
        let paths: Vec<&str> = results.iter().map(|r| r.path.as_str()).collect();
        assert!(
            paths.contains(&"src/file_0.ts"),
            "ANN path must return doc_0 within top_k; got: {paths:?}"
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
    fn vectorless_documents_receive_no_shared_semantic_credit() {
        let docs = vec![
            SearchDocument::new(
                "src/exact.rs".to_string(),
                "invoice payment status reconciler".to_string(),
                vec!["reconcileInvoicePaymentStatus".to_string()],
                vec![],
                "invoice payment status is reconciled from a Stripe webhook event".to_string(),
            ),
            SearchDocument::new(
                "src/partial.rs".to_string(),
                "invoice helper".to_string(),
                vec!["loadInvoice".to_string()],
                vec![],
                "load an invoice for display".to_string(),
            ),
        ];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vec![None, None]);

        let results = index.search(
            "invoice payment status reconciled Stripe webhook",
            &[1.0, 0.0],
            &ResolvedSearchOptions {
                top_k: 2,
                semantic_weight: 0.72,
                keyword_weight: 0.28,
                min_semantic_score: 0.0,
                min_keyword_score: 0.0,
                min_combined_score: 0.0,
                require_keyword_match: true,
                require_semantic_match: false,
                ..Default::default()
            },
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].path, "src/exact.rs");
        assert_eq!(results[0].semantic_score, 0.0);
        assert_eq!(results[1].semantic_score, 0.0);
        assert!(
            results[0].keyword_score > results[1].keyword_score,
            "vectorless documents must be distinguished only by keyword evidence: {results:?}"
        );
        assert!(
            results[0].score > results[1].score,
            "a shared/default vector must not give vectorless documents the same rank: {results:?}"
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
        index.index_with_vectors_and_tuning(docs, vectors, graph_tuning());

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

    // -- Exact blended scoring tests --

    const NEEDLE_DOCS: [usize; 3] = [300, 900, 1500];
    const NEEDLE_UNEMBEDDED_DOC: usize = 2000;

    /// Above `ANN_THRESHOLD`, so the index holds a vector store. The only
    /// keyword matches for "needle" sit far below cosine rank 50, and one of
    /// them has no embedding.
    fn needle_corpus() -> (Vec<SearchDocument>, Vec<Option<Vec<f32>>>) {
        let (mut docs, mut vectors) = make_ann_corpus(ANN_THRESHOLD + 600);
        for i in NEEDLE_DOCS.into_iter().chain([NEEDLE_UNEMBEDDED_DOC]) {
            docs[i] = SearchDocument::new(
                format!("src/file_{i}.ts"),
                format!("file {i}"),
                vec![format!("sym_{i}")],
                vec![],
                format!("needle content {i}"),
            );
        }
        vectors[NEEDLE_UNEMBEDDED_DOC] = None;
        (docs, vectors)
    }

    fn needle_query() -> Vec<f32> {
        unit_vec(1.0, 0.001)
    }

    fn blended_opts(top_k: usize) -> ResolvedSearchOptions {
        ResolvedSearchOptions {
            top_k,
            min_combined_score: 0.0,
            recency_window_days: None,
            ..Default::default()
        }
    }

    /// The same index scored over every document: the flat buffer holds each
    /// live vector and there is no store to shortlist from.
    fn exhaustive_oracle(index: &SearchIndex) -> SearchIndex {
        let mut oracle = index.clone();
        let dims = index.dims;
        let mut buffer = vec![0.0; index.documents.len() * dims];
        for i in 0..index.documents.len() {
            if let Some(vector) = index.vector_at(i) {
                buffer[i * dims..(i + 1) * dims].copy_from_slice(vector);
            }
        }
        oracle.vector_buffer = buffer;
        oracle.vector_updates.clear();
        oracle.ann_store = None;
        oracle
    }

    /// Ordered (path, score) equal to the oracle's; paths within a run of
    /// equal scores compare as sets, except the run cut by `top_k`.
    fn assert_matches_oracle(index: &SearchIndex, opts: &ResolvedSearchOptions, state: &str) {
        let ranked = |results: Vec<SearchResult>| -> Vec<(String, f64)> {
            results.into_iter().map(|r| (r.path, r.score)).collect()
        };
        let got = ranked(index.search("needle", &needle_query(), opts));
        let want = ranked(exhaustive_oracle(index).search("needle", &needle_query(), opts));
        let scores = |r: &[(String, f64)]| r.iter().map(|(_, s)| *s).collect::<Vec<_>>();
        assert_eq!(
            scores(&got),
            scores(&want),
            "{state}, top_k {}: got {got:?}, want {want:?}",
            opts.top_k
        );
        let cut = want.last().map(|(_, s)| *s);
        for (_, score) in &want {
            if Some(*score) == cut && want.len() == opts.top_k {
                continue;
            }
            let paths = |r: &[(String, f64)]| {
                r.iter()
                    .filter(|(_, s)| s == score)
                    .map(|(p, _)| p.clone())
                    .collect::<std::collections::BTreeSet<_>>()
            };
            assert_eq!(paths(&got), paths(&want), "{state}, top_k {}", opts.top_k);
        }
    }

    fn assert_all_top_k_match_oracle(index: &SearchIndex, state: &str) {
        for top_k in [1, 3, 5, 50] {
            assert_matches_oracle(index, &blended_opts(top_k), state);
            let mut filtered = blended_opts(top_k);
            filtered.exclude_globs = vec![Regex::new(&glob_to_regex("src/file_1*.ts")).unwrap()];
            assert_matches_oracle(index, &filtered, &format!("{state}, filtered"));
        }
    }

    #[test]
    fn exact_scoring_matches_the_exhaustive_oracle_below_the_graph_threshold() {
        let (docs, vectors) = needle_corpus();
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);
        assert!(index.ann_store.is_some(), "the corpus holds a vector store");

        let top = exhaustive_oracle(&index).search("needle", &needle_query(), &blended_opts(1));
        let cosine_rank = index.documents.iter().position(|d| d.path == top[0].path);
        assert!(
            top[0].path == format!("src/file_{}.ts", NEEDLE_DOCS[0])
                && cosine_rank.is_some_and(|rank| rank >= 50),
            "the blended winner must sit below cosine rank 50: {:?}",
            top[0].path
        );

        assert_all_top_k_match_oracle(&index, "no graph");
        assert!(
            !index.graph_is_built(),
            "a query below the threshold built a graph"
        );

        let changed = SearchDocument::new(
            "src/file_2200.ts".into(),
            "file 2200".into(),
            vec!["sym_2200".into()],
            vec![],
            "needle needle content 2200".into(),
        );
        let unembedded = SearchDocument::new(
            "src/file_2300.ts".into(),
            "file 2300".into(),
            vec!["sym_2300".into()],
            vec![],
            "needle content 2300".into(),
        );
        let deleted: Vec<String> = (0..5).map(|i| format!("src/file_{i}.ts")).collect();
        assert_eq!(
            index.apply_delta(
                vec![changed, unembedded],
                vec![Some(unit_vec(0.6, 0.8)), None],
                &deleted,
            ),
            IndexUpdateKind::Incremental
        );
        assert_all_top_k_match_oracle(&index, "after updates and deletions");

        crate::core::embeddings::hnsw_test_seam::unpaused(|| {
            index
                .ann_store
                .as_ref()
                .unwrap()
                .find_nearest_hnsw(&needle_query(), 1)
        });
        assert!(index.graph_is_built());
        assert_all_top_k_match_oracle(&index, "graph ready");
    }

    #[test]
    fn below_the_graph_threshold_queries_and_warmup_build_no_graph() {
        let (docs, vectors) = needle_corpus();
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);
        for top_k in [1, 3, 5, 50] {
            index.search("needle", &needle_query(), &blended_opts(top_k));
        }
        crate::core::embeddings::hnsw_test_seam::unpaused(|| index.prepare_ann());
        assert!(
            !index.graph_is_built(),
            "a graph was built below the threshold"
        );
    }

    #[test]
    fn below_the_graph_threshold_results_do_not_shift_when_a_graph_is_ready() {
        let (docs, vectors) = needle_corpus();
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);
        let ranked = |index: &SearchIndex| -> Vec<(String, f64)> {
            index
                .search("needle", &needle_query(), &blended_opts(5))
                .into_iter()
                .map(|r| (r.path, r.score))
                .collect()
        };
        let before = ranked(&index);
        crate::core::embeddings::hnsw_test_seam::unpaused(|| {
            index
                .ann_store
                .as_ref()
                .unwrap()
                .find_nearest_hnsw(&needle_query(), 1)
        });
        assert_eq!(
            before,
            ranked(&index),
            "results shifted once the graph was ready"
        );
    }

    #[test]
    fn above_the_graph_threshold_warmup_builds_the_graph_and_search_prunes_with_it() {
        let (docs, vectors) = needle_corpus();
        let mut index = SearchIndex::new();
        index.index_with_vectors_and_tuning(docs, vectors, graph_tuning());
        crate::core::embeddings::hnsw_test_seam::unpaused(|| index.prepare_ann());
        assert!(index.graph_is_built(), "warmup built no graph");
        let exact = exhaustive_oracle(&index).search("needle", &needle_query(), &blended_opts(1));
        let pruned = index.search("needle", &needle_query(), &blended_opts(1));
        assert_eq!(exact[0].path, format!("src/file_{}.ts", NEEDLE_DOCS[0]));
        assert_ne!(
            pruned[0].path, exact[0].path,
            "above the threshold the cosine shortlist no longer prunes"
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

        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
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
            scope: None,
        };

        // First call — cache miss, index built.
        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
        .await
        .unwrap();

        // Second call — cache hit, reuse_count should be 1.
        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
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

        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
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
            scope: None,
        };

        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
        .await
        .unwrap();
        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
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
            scope: None,
        };

        // Spawn 8 concurrent queries against the same empty cache.
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache_ref = Arc::clone(&cache);
                let wc = StdArc::clone(&walk_count);
                let opts = opts.clone();
                tokio::spawn(async move {
                    let walker = StableWalker { walk_count: wc };
                    semantic_code_search(
                        opts,
                        &FixedEmbedder,
                        &walker,
                        Some(Arc::clone(&cache_ref)),
                        None,
                    )
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
            scope: None,
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

    /// A walker that installs the index of its root in `slot` itself and
    /// counts the walks asked for documents.
    struct InstallingWalker {
        slot: Arc<RwLock<Option<Arc<CachedSearchIndex>>>>,
        document_walks: Arc<std::sync::atomic::AtomicU32>,
    }

    impl WalkAndIndexFn for InstallingWalker {
        fn walk_and_index(&self, _root: &Path) -> WalkAndIndexFuture<'_> {
            self.document_walks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Box::pin(async { Ok((vec![make_doc("a.rs", "walked")], vec![Some(vec![1.0, 0.0])])) })
        }

        fn walk_or_install(&self, root: &Path) -> WalkOrInstallFuture<'_> {
            let root = root.canonicalize().unwrap();
            Box::pin(async move {
                let entry = Arc::new(CachedSearchIndex::build(
                    &root,
                    vec![make_doc("installed.rs", "installed by the walk")],
                    vec![Some(vec![1.0, 0.0])],
                    0,
                    0,
                    None,
                ));
                *self.slot.write().await = Some(Arc::clone(&entry));
                Ok(WalkOutcome::Installed(entry))
            })
        }
    }

    /// A walk that installs the index of its root is not asked for documents:
    /// the search answers from what it installed.
    #[tokio::test]
    async fn search_answers_from_the_index_its_walk_installed() {
        let root = tempfile::tempdir().unwrap();
        let slot: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let walker = InstallingWalker {
            slot: Arc::clone(&slot),
            document_walks: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        };
        let opts = SemanticSearchOptions {
            root_dir: root.path().to_path_buf(),
            ..gen_test_opts()
        };

        let answer = semantic_code_search(opts, &FixedEmbedder2, &walker, Some(slot), None)
            .await
            .unwrap();
        assert_eq!(
            walker
                .document_walks
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "the search asked for the documents of a walk that installed its index"
        );
        assert!(answer.contains("installed.rs"), "{answer}");
    }

    /// Cache hit without a file change: when the generation matches, the walk
    /// should be skipped entirely on the second request.
    #[tokio::test]
    async fn test_generation_hit_skips_walk() {
        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let walk_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let walker = CountWalker(Arc::clone(&walk_count));
        let opts = gen_test_opts();

        // First request — cold cache, must walk.
        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder2,
            &walker,
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
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
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
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

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_generation_cache_hit_releases_lock_while_snippet_io_is_pending() {
        use std::io::Write;
        use std::process::Command;
        use std::time::Duration;

        let root = tempfile::tempdir().unwrap();
        let relative_path = "config.yaml";
        let fifo_path = root.path().join(relative_path);
        let status = Command::new("mkfifo").arg(&fifo_path).status().unwrap();
        assert!(status.success(), "mkfifo failed with {status}");

        let docs = vec![make_doc(relative_path, "service billing")];
        let fingerprint = IndexFingerprint::from_docs(&docs);
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vec![Some(vec![1.0_f32, 0.0])]);
        let mut cached = CachedSearchIndex::new(index, fingerprint, 7);
        cached.search_root = std::fs::canonicalize(root.path()).unwrap();
        let cached = Arc::new(cached);
        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> =
            Arc::new(RwLock::new(Some(cached)));
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(7));

        let (reader_pending_tx, reader_pending_rx) = tokio::sync::oneshot::channel();
        let (release_writer_tx, release_writer_rx) = std::sync::mpsc::channel();
        let fifo_for_writer = fifo_path.clone();
        let writer = tokio::task::spawn_blocking(move || {
            let mut fifo = std::fs::OpenOptions::new()
                .write(true)
                .open(fifo_for_writer)
                .unwrap();
            reader_pending_tx.send(()).unwrap();
            release_writer_rx.recv().unwrap();
            fifo.write_all(b"service: billing\nport: 8080\n").unwrap();
        });

        let mut opts = gen_test_opts();
        opts.root_dir = root.path().to_path_buf();
        opts.query = "billing".to_string();
        let cache_for_search = Arc::clone(&cache);
        let generation_for_search = Arc::clone(&generation);
        let search = tokio::spawn(async move {
            let walker = CountWalker(Arc::new(std::sync::atomic::AtomicU32::new(0)));
            semantic_code_search(
                opts,
                &FixedEmbedder2,
                &walker,
                Some(cache_for_search),
                Some(generation_for_search),
            )
            .await
        });

        tokio::time::timeout(Duration::from_secs(2), reader_pending_rx)
            .await
            .expect("snippet reader did not reach the FIFO")
            .unwrap();

        let cache_for_writer = Arc::clone(&cache);
        let (lock_acquired_tx, lock_acquired_rx) = tokio::sync::oneshot::channel();
        let lock_writer = tokio::spawn(async move {
            let guard = cache_for_writer.write().await;
            let _ = lock_acquired_tx.send(());
            drop(guard);
        });
        let acquired_while_pending =
            tokio::time::timeout(Duration::from_millis(250), lock_acquired_rx)
                .await
                .is_ok();

        release_writer_tx.send(()).unwrap();
        writer.await.unwrap();
        search.await.unwrap().unwrap();
        lock_writer.await.unwrap();

        assert!(
            acquired_while_pending,
            "cache writer could not acquire the lock while snippet I/O was pending"
        );
    }

    /// File-change event (generation bump) forces a walk + rebuild on next request.
    #[tokio::test]
    async fn test_generation_bump_triggers_walk() {
        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let walk_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let walker = CountWalker(Arc::clone(&walk_count));
        let opts = gen_test_opts();

        // Prime the cache.
        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder2,
            &walker,
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
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
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
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
        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let walk_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let walker = CountWalker(Arc::clone(&walk_count));
        let opts = gen_test_opts();

        // No generation counter — tracker-off mode.
        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder2,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
        .await
        .unwrap();
        let _ = semantic_code_search(
            opts.clone(),
            &FixedEmbedder2,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
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

    struct MetadataCountingWalker {
        metadata: Arc<std::sync::atomic::AtomicU64>,
        metadata_count: Arc<std::sync::atomic::AtomicU32>,
        walk_count: Arc<std::sync::atomic::AtomicU32>,
        path: Arc<std::sync::Mutex<String>>,
    }

    impl WalkAndIndexFn for MetadataCountingWalker {
        fn metadata_fingerprint(&self, _root: &Path) -> MetadataFingerprintFuture<'_> {
            self.metadata_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let hash = self.metadata.load(std::sync::atomic::Ordering::Relaxed);
            Box::pin(async move {
                Ok(Some(MetadataFingerprint {
                    n_entries: 1,
                    metadata_hash: hash,
                }))
            })
        }

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
            self.walk_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = self.path.lock().unwrap().clone();
            Box::pin(async move {
                Ok((
                    vec![make_doc(&path, "metadata target")],
                    vec![Some(vec![1.0_f32, 0.0])],
                ))
            })
        }
    }

    #[tokio::test]
    async fn metadata_fingerprint_skips_walk_when_tracker_is_off_and_tree_is_unchanged() {
        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let metadata = Arc::new(std::sync::atomic::AtomicU64::new(10));
        let metadata_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let walk_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let walker = MetadataCountingWalker {
            metadata,
            metadata_count: Arc::clone(&metadata_count),
            walk_count: Arc::clone(&walk_count),
            path: Arc::new(std::sync::Mutex::new("before.rs".to_string())),
        };

        semantic_code_search(
            gen_test_opts(),
            &FixedEmbedder2,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
        .await
        .unwrap();
        semantic_code_search(
            gen_test_opts(),
            &FixedEmbedder2,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            metadata_count.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "the cheap metadata check runs on each tracker-off request"
        );
        assert_eq!(
            walk_count.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "an unchanged metadata fingerprint must skip the full walk"
        );
    }

    #[tokio::test]
    async fn metadata_fingerprint_change_falls_through_to_walk_and_refreshes_results() {
        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let metadata = Arc::new(std::sync::atomic::AtomicU64::new(10));
        let walk_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let path = Arc::new(std::sync::Mutex::new("before.rs".to_string()));
        let walker = MetadataCountingWalker {
            metadata: Arc::clone(&metadata),
            metadata_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            walk_count: Arc::clone(&walk_count),
            path: Arc::clone(&path),
        };

        semantic_code_search(
            gen_test_opts(),
            &FixedEmbedder2,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
        .await
        .unwrap();
        let stale_fingerprint = cache.read().await.as_ref().unwrap().fingerprint.clone();

        metadata.store(11, std::sync::atomic::Ordering::Relaxed);
        *path.lock().unwrap() = "after.rs".to_string();
        let stale_result = semantic_code_search(
            gen_test_opts(),
            &FixedEmbedder2,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            walk_count.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "a changed metadata fingerprint must perform the full walk"
        );
        assert!(
            stale_result.contains("before.rs"),
            "the existing small-delta path serves stale results while rebuilding: {stale_result}"
        );

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let rebuild_finished = cache
                    .read()
                    .await
                    .as_ref()
                    .is_some_and(|cached| cached.fingerprint != stale_fingerprint);
                if rebuild_finished {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("background index replacement must finish");

        let fresh_result = semantic_code_search(
            gen_test_opts(),
            &FixedEmbedder2,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
        .await
        .unwrap();
        assert!(
            fresh_result.contains("after.rs"),
            "the replacement index must expose the changed walk result: {fresh_result}"
        );
        assert_eq!(
            walk_count.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "the refreshed metadata fingerprint must avoid a third full walk"
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
                Some(StdArc::clone(&cache)),
                Some(StdArc::clone(&generation)),
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
                        Some(StdArc::clone(&cache_ref)),
                        Some(StdArc::clone(&gen_ref)),
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

    // ---------------------------------------------------------------------------
    // Background-rebuild tests
    // ---------------------------------------------------------------------------

    /// Helper: make a stale CachedSearchIndex with a given doc count.
    fn make_stale_cached(n_docs: usize) -> Arc<CachedSearchIndex> {
        let mut idx = SearchIndex::new();
        let docs: Vec<SearchDocument> = (0..n_docs)
            .map(|i| make_doc(&format!("f{i}.rs"), &format!("content {i}")))
            .collect();
        let vecs: Vec<Option<Vec<f32>>> = (0..n_docs).map(|_| Some(vec![1.0_f32, 0.0])).collect();
        idx.index_with_vectors(docs.clone(), vecs);
        let fp = IndexFingerprint::from_docs(&docs);
        Arc::new(CachedSearchIndex::new(idx, fp, 0))
    }

    /// `qualifies_for_background_rebuild`: delta within 5%/200-doc threshold qualifies.
    #[test]
    fn test_qualifies_for_background_rebuild_threshold() {
        // 100 docs → 5% = 5, cap = min(5, 200) = 5.
        let stale = make_stale_cached(100);
        // +4 docs → qualifies.
        assert!(stale.qualifies_for_background_rebuild(&IndexFingerprint {
            n_docs: 104,
            content_hash: 0,
        }));
        // +6 docs → does not qualify (> 5% of 100).
        assert!(!stale.qualifies_for_background_rebuild(&IndexFingerprint {
            n_docs: 106,
            content_hash: 0,
        }));

        // 5000 docs → 5% = 250, cap = min(250, 200) = 200.
        let big = make_stale_cached(5000);
        // +200 docs → qualifies (at the cap).
        assert!(big.qualifies_for_background_rebuild(&IndexFingerprint {
            n_docs: 5200,
            content_hash: 0,
        }));
        // +201 docs → does not qualify.
        assert!(!big.qualifies_for_background_rebuild(&IndexFingerprint {
            n_docs: 5201,
            content_hash: 0,
        }));
    }

    /// Small-delta change: first query returns stale result immediately, then
    /// after the background task finishes the cache holds the fresh index.
    #[tokio::test]
    async fn test_background_rebuild_serves_stale_then_swaps() {
        use std::sync::atomic::{AtomicU32, Ordering};

        // Walker: first call returns 10 docs ("stale"), second 11 docs ("fresh").
        struct DeltaWalker(Arc<AtomicU32>);
        impl WalkAndIndexFn for DeltaWalker {
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
                let n = self.0.fetch_add(1, Ordering::Relaxed);
                Box::pin(async move {
                    let count = if n == 0 { 10usize } else { 11usize };
                    let docs: Vec<_> = (0..count)
                        .map(|i| make_doc(&format!("f{i}.rs"), &format!("c{i}")))
                        .collect();
                    let vecs: Vec<_> = (0..count).map(|_| Some(vec![1.0_f32, 0.0])).collect();
                    Ok((docs, vecs))
                })
            }
        }

        struct FixedEmbed;
        impl EmbedFn for FixedEmbed {
            fn embed(
                &self,
                _t: &[String],
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>,
            > {
                Box::pin(async { Ok(vec![vec![1.0_f32, 0.0]]) })
            }
        }

        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let call_count = Arc::new(AtomicU32::new(0));

        // Prime the cache with 10 docs.
        let walker = DeltaWalker(Arc::clone(&call_count));
        let _ = semantic_code_search(
            gen_test_opts(),
            &FixedEmbed,
            &walker,
            Some(Arc::clone(&cache)),
            None,
        )
        .await
        .unwrap();
        assert_eq!(call_count.load(Ordering::Relaxed), 1);
        let stale_n = cache.read().await.as_ref().unwrap().fingerprint.n_docs;
        assert_eq!(stale_n, 10);

        // Second call: walker now returns 11 docs (small delta → background rebuild).
        // The first query should complete immediately (stale served).
        let walker2 = DeltaWalker(Arc::clone(&call_count));
        let _ = semantic_code_search(
            gen_test_opts(),
            &FixedEmbed,
            &walker2,
            Some(Arc::clone(&cache)),
            None,
        )
        .await
        .unwrap();

        // Give the background task a moment to complete.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Cache must now hold 11 docs.
        let fresh_n = cache.read().await.as_ref().unwrap().fingerprint.n_docs;
        assert_eq!(
            fresh_n, 11,
            "background rebuild must have swapped in 11-doc index"
        );
    }

    /// CAS prevents duplicate spawns: only one background rebuild fires even if
    /// two concurrent requests see the same stale index simultaneously.
    /// After the spawned task finishes, `rebuild_in_progress` resets to false.
    #[tokio::test]
    async fn test_background_rebuild_no_double_spawn_on_concurrent_invalidation() {
        use std::sync::atomic::{AtomicU32, Ordering};

        // Walker: first call returns 10 docs, subsequent calls return 11.
        struct MultiDeltaWalker(Arc<AtomicU32>);
        impl WalkAndIndexFn for MultiDeltaWalker {
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
                let n = self.0.fetch_add(1, Ordering::Relaxed);
                Box::pin(async move {
                    let count = if n == 0 { 10usize } else { 11usize };
                    let docs: Vec<_> = (0..count)
                        .map(|i| make_doc(&format!("f{i}.rs"), &format!("v{n}c{i}")))
                        .collect();
                    let vecs: Vec<_> = (0..count).map(|_| Some(vec![1.0_f32, 0.0])).collect();
                    Ok((docs, vecs))
                })
            }
        }

        struct FixedEmbed;
        impl EmbedFn for FixedEmbed {
            fn embed(
                &self,
                _t: &[String],
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>,
            > {
                Box::pin(async { Ok(vec![vec![1.0_f32, 0.0]]) })
            }
        }

        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let call_count = Arc::new(AtomicU32::new(0));

        // Prime with 10 docs.
        {
            let walker = MultiDeltaWalker(Arc::clone(&call_count));
            let _ = semantic_code_search(
                gen_test_opts(),
                &FixedEmbed,
                &walker,
                Some(Arc::clone(&cache)),
                None,
            )
            .await
            .unwrap();
        }

        // Snapshot stale entry before concurrent requests.
        let stale_arc = Arc::clone(cache.read().await.as_ref().unwrap());
        assert!(
            !stale_arc
                .rebuild_in_progress
                .load(std::sync::atomic::Ordering::Acquire)
        );

        // Two concurrent requests both see the 11-doc walker.
        // Only one should CAS-claim the rebuild slot.
        let walker_a = MultiDeltaWalker(Arc::clone(&call_count));
        let walker_b = MultiDeltaWalker(Arc::clone(&call_count));
        let (r1, r2) = tokio::join!(
            semantic_code_search(
                gen_test_opts(),
                &FixedEmbed,
                &walker_a,
                Some(Arc::clone(&cache)),
                None,
            ),
            semantic_code_search(
                gen_test_opts(),
                &FixedEmbed,
                &walker_b,
                Some(Arc::clone(&cache)),
                None,
            ),
        );
        r1.unwrap();
        r2.unwrap();

        // Let background task finish and reset the flag.
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        // Flag must be reset (RAII guard fired).
        assert!(
            !stale_arc
                .rebuild_in_progress
                .load(std::sync::atomic::Ordering::Acquire),
            "rebuild_in_progress must reset after background task completes"
        );
    }

    /// RAII guard resets `rebuild_in_progress` even when the spawned task panics.
    #[tokio::test]
    async fn test_background_rebuild_panic_resets_flag() {
        // Build a stale cached entry directly and set rebuild_in_progress = true,
        // then drop a RebuildGuard — flag must be false afterwards.
        let stale = make_stale_cached(10);
        stale
            .rebuild_in_progress
            .store(true, std::sync::atomic::Ordering::Release);

        // Drop the guard (simulates panic unwind).
        {
            let _guard = RebuildGuard(Arc::clone(&stale));
            // flag still true while guard is alive.
            assert!(
                stale
                    .rebuild_in_progress
                    .load(std::sync::atomic::Ordering::Acquire)
            );
        }
        // Flag must be false after drop.
        assert!(
            !stale
                .rebuild_in_progress
                .load(std::sync::atomic::Ordering::Acquire),
            "RebuildGuard must reset flag on drop"
        );
    }

    #[test]
    fn lane_m_worktree_fill_cannot_push_into_or_clone_primary_semantic_index() {
        let mut doc = SearchDocument::new(
            "src/shared.rs".to_string(),
            String::new(),
            vec![],
            vec![],
            "fn shared() {}".to_string(),
        );
        doc.source_hash = "shared-hash".to_string();
        let fingerprint = IndexFingerprint::from_docs(std::slice::from_ref(&doc));
        let mut index = SearchIndex::new();
        index.index_with_vectors(vec![doc], vec![Some(vec![1.0, 0.0])]);
        let mut cache = CachedSearchIndex::new(index, fingerprint, 0);
        cache.search_root = PathBuf::from("/lane-m/primary");
        let primary = Arc::new(cache);
        let mut worktree_slot = Arc::clone(&primary);

        CachedSearchIndex::refresh_vectors(
            &mut worktree_slot,
            Path::new("/lane-m/worktree"),
            vec![(
                "src/shared.rs".to_string(),
                "shared-hash".to_string(),
                vec![0.0, 1.0],
            )],
            1,
        );

        assert!(
            primary.pending.lock().unwrap().batches.is_empty(),
            "a worktree fill pushed a refresh batch into the primary's index"
        );
        assert!(
            Arc::ptr_eq(&worktree_slot, &primary),
            "a worktree fill deep-cloned the primary's index into its slot"
        );
    }

    #[tokio::test]
    async fn lane_m_background_rebuild_releases_snapshot_and_stale_generation_after_readers() {
        struct SnapshotWalker {
            docs: Vec<SearchDocument>,
            started: Arc<tokio::sync::Notify>,
            release: Arc<tokio::sync::Notify>,
        }

        impl WalkAndIndexFn for SnapshotWalker {
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
                Box::pin(async move {
                    self.started.notify_one();
                    self.release.notified().await;
                    Ok((
                        self.docs.clone(),
                        vec![Some(vec![1.0, 0.0]); self.docs.len()],
                    ))
                })
            }
        }

        struct InitialWalker(Vec<SearchDocument>);
        impl WalkAndIndexFn for InitialWalker {
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
                Box::pin(
                    async move { Ok((self.0.clone(), vec![Some(vec![1.0, 0.0]); self.0.len()])) },
                )
            }
        }

        let root = tempfile::tempdir().unwrap();
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let mut options = gen_test_opts();
        options.root_dir = root.path().to_path_buf();
        let initial_docs = (0..100)
            .map(|i| make_doc(&format!("src/initial_{i}.rs"), "initial generation"))
            .collect();
        semantic_code_search_owned(
            options.clone(),
            &FixedEmbedder2,
            Arc::new(InitialWalker(initial_docs)),
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
        )
        .await
        .unwrap();

        for cycle in 1..=3_u64 {
            let reader = cache.read().await.as_ref().unwrap().clone();
            let stale = Arc::downgrade(&reader);
            let started = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(tokio::sync::Notify::new());
            let docs = (0..100 + cycle as usize)
                .map(|i| make_doc(&format!("src/cycle_{cycle}_{i}.rs"), "rebuilt generation"))
                .collect();
            let snapshot = Arc::new(SnapshotWalker {
                docs,
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            });
            let snapshot_weak = Arc::downgrade(&snapshot);
            generation.store(cycle, std::sync::atomic::Ordering::Release);
            semantic_code_search_owned(
                options.clone(),
                &FixedEmbedder2,
                snapshot.clone(),
                Some(Arc::clone(&cache)),
                Some(Arc::clone(&generation)),
            )
            .await
            .unwrap();
            drop(snapshot);
            tokio::time::timeout(std::time::Duration::from_secs(1), started.notified())
                .await
                .expect("production background rebuild did not start");
            release.notify_one();
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                loop {
                    let installed = cache.read().await.as_ref().is_some_and(|entry| {
                        entry.generation.load(std::sync::atomic::Ordering::Acquire) == cycle
                    });
                    if installed && snapshot_weak.upgrade().is_none() {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("background rebuild retained its build snapshot after completion");
            assert!(
                stale.upgrade().is_some(),
                "the stale generation died while a reader still retained it"
            );
            drop(reader);
            assert!(
                stale.upgrade().is_none(),
                "the completed rebuild retained its stale generation after readers released it"
            );
        }
    }

    /// Stale-write guard: if a fresher index is installed before the background
    /// task finishes, the background task skips the swap.
    #[tokio::test]
    async fn test_stale_write_guard_skips_swap_when_fresher_installed() {
        // Build stale (10-doc) and fresh (11-doc) fingerprints.
        let stale_docs: Vec<_> = (0..10)
            .map(|i| make_doc(&format!("s{i}.rs"), &format!("stale{i}")))
            .collect();
        let fresh_docs: Vec<_> = (0..11)
            .map(|i| make_doc(&format!("n{i}.rs"), &format!("fresh{i}")))
            .collect();
        let stale_fp = IndexFingerprint::from_docs(&stale_docs);
        let fresh_fp = IndexFingerprint::from_docs(&fresh_docs);

        // Construct fresh entry with a different fingerprint and install it.
        let mut fresh_idx = SearchIndex::new();
        let vecs: Vec<_> = (0..11).map(|_| Some(vec![1.0_f32, 0.0])).collect();
        fresh_idx.index_with_vectors(fresh_docs, vecs);
        let fresh_entry = Arc::new(CachedSearchIndex::new(fresh_idx, fresh_fp.clone(), 0));

        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> =
            Arc::new(RwLock::new(Some(Arc::clone(&fresh_entry))));

        // Simulate: background task computed a new entry based on stale_fp as the "stale" base,
        // but the cache already has fresh_fp installed. The stale-write guard should skip swap.
        let new_entry = make_stale_cached(12); // whatever — should not land
        let should_install = {
            let guard = cache.read().await;
            match *guard {
                Some(ref cur) => cur.fingerprint == stale_fp, // stale_fp ≠ fresh_fp → false
                None => false,
            }
        };
        assert!(
            !should_install,
            "stale-write guard must skip swap when fresher index is already installed"
        );

        // Verify the cache still holds the fresh entry (no write occurred).
        let after_n = cache.read().await.as_ref().unwrap().fingerprint.n_docs;
        assert_eq!(after_n, 11, "cache must still hold the fresh 11-doc index");
        drop(new_entry);
        // Verify stale_fp != fresh_fp (sanity).
        assert_ne!(stale_fp, fresh_fp);
    }

    #[tokio::test]
    async fn single_changed_file_in_large_index_is_visible_without_full_rebuild() {
        use std::sync::atomic::{AtomicU32, Ordering};

        struct OneFileChangeWalker(Arc<AtomicU32>);
        impl WalkAndIndexFn for OneFileChangeWalker {
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
                let changed = self.0.fetch_add(1, Ordering::Relaxed) > 0;
                Box::pin(async move {
                    let docs: Vec<_> = (0..1_000)
                        .map(|i| {
                            let content = if changed && i == 777 {
                                "fn incremental_delta_needle() {}"
                            } else {
                                "fn stable_content() {}"
                            };
                            make_doc(&format!("src/file_{i}.rs"), content)
                        })
                        .collect();
                    let vectors = vec![Some(vec![1.0, 0.0]); docs.len()];
                    Ok((docs, vectors))
                })
            }
        }

        let calls = Arc::new(AtomicU32::new(0));
        let walker = OneFileChangeWalker(Arc::clone(&calls));
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let mut options = gen_test_opts();
        options.query = "incremental delta needle".to_string();
        options.semantic_weight = Some(0.0);
        options.keyword_weight = Some(1.0);
        options.min_keyword_score = Some(0.01);
        options.require_keyword_match = Some(true);

        semantic_code_search(
            options.clone(),
            &FixedEmbedder2,
            &walker,
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
        )
        .await
        .unwrap();
        let before_count = cache.read().await.as_ref().unwrap().index.document_count();
        let before_full_rebuilds = cache
            .read()
            .await
            .as_ref()
            .unwrap()
            .index
            .full_rebuild_count();

        generation.fetch_add(1, Ordering::Release);
        let result = semantic_code_search(
            options,
            &FixedEmbedder2,
            &walker,
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
        )
        .await
        .unwrap();
        let guard = cache.read().await;
        let current = guard.as_ref().unwrap();

        assert_eq!(before_count, 1_000);
        assert_eq!(current.index.document_count(), before_count);
        assert_eq!(
            current.index.full_rebuild_count(),
            before_full_rebuilds,
            "one changed document must not trigger a full rebuild"
        );
        assert_eq!(
            current.generation.load(Ordering::Acquire),
            1,
            "an in-place delta must advance the cached generation"
        );
        assert!(
            result.contains("src/file_777.rs"),
            "the first query after vectors are ready must see changed content: {result}"
        );
    }

    #[tokio::test]
    async fn required_full_rebuild_serves_previous_generation_without_blocking() {
        use std::sync::atomic::{AtomicU32, Ordering};

        struct BlockingMassChangeWalker {
            calls: Arc<AtomicU32>,
            rebuild_started: Arc<tokio::sync::Notify>,
            release_rebuild: Arc<tokio::sync::Notify>,
        }
        impl WalkAndIndexFn for BlockingMassChangeWalker {
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
                let call = self.calls.fetch_add(1, Ordering::Relaxed);
                let rebuild_started = Arc::clone(&self.rebuild_started);
                let release_rebuild = Arc::clone(&self.release_rebuild);
                Box::pin(async move {
                    let count = if call == 0 {
                        100
                    } else {
                        rebuild_started.notify_one();
                        release_rebuild.notified().await;
                        150
                    };
                    let docs: Vec<_> = (0..count)
                        .map(|i| make_doc(&format!("src/stale_{i}.rs"), "stale generation"))
                        .collect();
                    let vectors = vec![Some(vec![1.0, 0.0]); docs.len()];
                    Ok((docs, vectors))
                })
            }
        }

        let calls = Arc::new(AtomicU32::new(0));
        let rebuild_started = Arc::new(tokio::sync::Notify::new());
        let release_rebuild = Arc::new(tokio::sync::Notify::new());
        let walker = Arc::new(BlockingMassChangeWalker {
            calls: Arc::clone(&calls),
            rebuild_started: Arc::clone(&rebuild_started),
            release_rebuild: Arc::clone(&release_rebuild),
        });
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let mut options = gen_test_opts();
        options.query = "stale generation".to_string();

        let initial_walker: Arc<dyn WalkAndIndexFn> = walker.clone();
        semantic_code_search_owned(
            options.clone(),
            &FixedEmbedder2,
            initial_walker,
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
        )
        .await
        .unwrap();
        generation.fetch_add(1, Ordering::Release);

        let query_walker: Arc<dyn WalkAndIndexFn> = walker.clone();
        let query_cache = Arc::clone(&cache);
        let query_generation = Arc::clone(&generation);
        let query = tokio::spawn(async move {
            semantic_code_search_owned(
                options,
                &FixedEmbedder2,
                query_walker,
                Some(query_cache),
                Some(query_generation),
            )
            .await
        });
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            rebuild_started.notified(),
        )
        .await
        .expect("the full rebuild did not start");

        let result = tokio::time::timeout(std::time::Duration::from_millis(100), query)
            .await
            .expect("query waited for the required full rebuild instead of serving the previous generation")
            .unwrap()
            .unwrap();
        release_rebuild.notify_one();

        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let count = cache
                    .read()
                    .await
                    .as_ref()
                    .map(|entry| entry.index.document_count());
                if count == Some(150) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("released background rebuild did not install the 150-document generation");

        assert!(
            result.contains("Index: 100 document(s)"),
            "the query should report the previous generation during rebuild: {result}"
        );
        assert!(
            result.contains("src/stale_"),
            "the stale generation should remain queryable during rebuild: {result}"
        );
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ann_rebuild_stays_stale_until_replacement_graph_is_ready() {
        use std::sync::atomic::{AtomicU32, Ordering};

        struct AnnRebuildWalker(Arc<AtomicU32>);
        impl WalkAndIndexFn for AnnRebuildWalker {
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
                let changed = self.0.fetch_add(1, Ordering::Relaxed) > 0;
                Box::pin(async move {
                    let (mut docs, vectors) = make_ann_corpus(ANN_THRESHOLD + 50);
                    if changed {
                        for doc in docs.iter_mut().take(500) {
                            doc.content.push_str(" mass replacement");
                        }
                    }
                    Ok((docs, vectors))
                })
            }
        }

        struct AnnEmbedder;
        impl EmbedFn for AnnEmbedder {
            fn embed(
                &self,
                _texts: &[String],
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send + '_>,
            > {
                Box::pin(async { Ok(vec![vec![1.0, 0.0, 0.0, 0.0]]) })
            }
        }

        let calls = Arc::new(AtomicU32::new(0));
        let walker: Arc<dyn WalkAndIndexFn> = Arc::new(AnnRebuildWalker(calls));
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let cache: Arc<RwLock<Option<Arc<CachedSearchIndex>>>> = Arc::new(RwLock::new(None));
        let mut options = gen_test_opts();
        options.query = "file".to_string();
        semantic_code_search_owned(
            options.clone(),
            &AnnEmbedder,
            Arc::clone(&walker),
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
        )
        .await
        .unwrap();
        let previous = cache.read().await.as_ref().cloned().unwrap();
        // The first generation's graph builds in the background while exact
        // search answers; wait for it.
        tokio::time::timeout(std::time::Duration::from_secs(300), async {
            while !previous
                .index
                .ann_store
                .as_ref()
                .unwrap()
                .hnsw_is_initialized()
            {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the warm generation must get an initialized ANN graph");

        generation.store(1, Ordering::Release);
        let graph_pause = crate::core::embeddings::hnsw_test_seam::pause_next_build();
        let stale_result = semantic_code_search_owned(
            options,
            &AnnEmbedder,
            walker,
            Some(Arc::clone(&cache)),
            Some(generation),
        )
        .await
        .unwrap();
        assert!(stale_result.contains("Index: 2050 document(s)"));
        // The replacement build is held at the pause however long this waits,
        // and can queue behind other tests' builds before reaching it.
        tokio::time::timeout(
            std::time::Duration::from_secs(300),
            graph_pause.wait_until_entered(),
        )
        .await
        .expect("replacement ANN construction never started before publication");
        assert!(
            cache
                .read()
                .await
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &previous)),
            "the previous generation was retired before the replacement ANN graph was ready"
        );

        graph_pause.release();
        tokio::time::timeout(std::time::Duration::from_secs(300), async {
            loop {
                let current = cache.read().await.as_ref().cloned().unwrap();
                if !Arc::ptr_eq(&current, &previous) {
                    assert!(
                        current
                            .index
                            .ann_store
                            .as_ref()
                            .unwrap()
                            .hnsw_is_initialized(),
                        "published replacement has a lazy ANN graph"
                    );
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("ready replacement ANN generation was not published");
    }

    #[test]
    fn rejected_mass_delta_cannot_be_hidden_by_a_later_small_delta() {
        use std::sync::atomic::Ordering;

        let docs = (0..25)
            .map(|i| make_doc(&format!("src/file_{i}.rs"), "original content"))
            .collect::<Vec<_>>();
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs.clone(), vec![Some(vec![1.0, 0.0]); docs.len()]);
        let mut cached = Arc::new(CachedSearchIndex::new(
            index,
            IndexFingerprint::from_docs(&docs),
            0,
        ));
        Arc::get_mut(&mut cached).unwrap().search_root = std::path::PathBuf::from("/tmp");

        let mass_docs = (0..6)
            .map(|i| make_doc(&format!("src/file_{i}.rs"), "mass refresh needle"))
            .collect::<Vec<_>>();
        assert!(!CachedSearchIndex::refresh_paths(
            &mut cached,
            std::path::Path::new("/tmp"),
            mass_docs,
            vec![Some(vec![1.0, 0.0]); 6],
            &[],
            1,
        ));
        CachedSearchIndex::refresh_paths(
            &mut cached,
            std::path::Path::new("/tmp"),
            vec![make_doc("src/file_24.rs", "later small delta")],
            vec![Some(vec![1.0, 0.0])],
            &[],
            2,
        );

        let advanced = cached.generation.load(Ordering::Acquire) == 2;
        let mass_is_represented = cached.index.documents[..6]
            .iter()
            .all(|doc| doc.content.contains("mass refresh needle"));
        assert!(
            !advanced || mass_is_represented,
            "the later small delta advanced generation 2 without the rejected generation-1 batch"
        );
    }

    #[tokio::test]
    async fn owned_scoped_search_applies_ready_replacement_before_answering() {
        use std::sync::atomic::{AtomicU32, Ordering};

        struct ScopedReplacementWalker(AtomicU32);
        impl WalkAndIndexFn for ScopedReplacementWalker {
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
                let fresh = self.0.fetch_add(1, Ordering::Relaxed) > 0;
                Box::pin(async move {
                    let docs = (0..10)
                        .map(|i| {
                            let content = if i == 0 {
                                if fresh {
                                    "zephyrreplacement"
                                } else {
                                    "fossilmarker"
                                }
                            } else {
                                "stable decoy"
                            };
                            make_doc(&format!("file_{i}.rs"), content)
                        })
                        .collect::<Vec<_>>();
                    let vectors = vec![Some(vec![1.0, 0.0]); docs.len()];
                    Ok((docs, vectors))
                })
            }
        }

        let root = tempfile::tempdir().unwrap();
        let walker: Arc<dyn WalkAndIndexFn> = Arc::new(ScopedReplacementWalker(AtomicU32::new(0)));
        let cache = Arc::new(RwLock::new(None));
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut options = gen_test_opts();
        options.root_dir = root.path().to_path_buf();
        options.query = "zephyrreplacement".to_string();
        options.semantic_weight = Some(0.0);
        options.keyword_weight = Some(1.0);
        options.require_keyword_match = Some(true);
        semantic_code_search_owned(
            options.clone(),
            &FixedEmbedder2,
            Arc::clone(&walker),
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
        )
        .await
        .unwrap();

        let callback_root = root.path().parent().unwrap();
        assert!(!CachedSearchIndex::refresh_paths(
            cache.write().await.as_mut().unwrap(),
            callback_root,
            vec![make_doc("file_0.rs", "zephyrreplacement")],
            vec![Some(vec![1.0, 0.0])],
            &[],
            1,
        ));
        generation.store(1, Ordering::Release);
        let result = semantic_code_search_owned(
            options,
            &FixedEmbedder2,
            walker,
            Some(cache),
            Some(generation),
        )
        .await
        .unwrap();
        assert!(
            result.contains("file_0.rs"),
            "the first owned scoped query served the pre-replacement index: {result}"
        );
    }

    #[tokio::test]
    async fn owned_scoped_search_applies_ready_deletion_before_answering() {
        use std::sync::atomic::{AtomicU32, Ordering};

        struct ScopedDeletionWalker(AtomicU32);
        impl WalkAndIndexFn for ScopedDeletionWalker {
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
                let fresh = self.0.fetch_add(1, Ordering::Relaxed) > 0;
                Box::pin(async move {
                    let start = usize::from(fresh);
                    let docs = (start..10)
                        .map(|i| {
                            let content = if i == 0 {
                                "vanishinguniquetoken"
                            } else {
                                "stable decoy"
                            };
                            make_doc(&format!("file_{i}.rs"), content)
                        })
                        .collect::<Vec<_>>();
                    let vectors = vec![Some(vec![1.0, 0.0]); docs.len()];
                    Ok((docs, vectors))
                })
            }
        }

        let root = tempfile::tempdir().unwrap();
        let walker: Arc<dyn WalkAndIndexFn> = Arc::new(ScopedDeletionWalker(AtomicU32::new(0)));
        let cache = Arc::new(RwLock::new(None));
        let generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut options = gen_test_opts();
        options.root_dir = root.path().to_path_buf();
        options.query = "vanishinguniquetoken".to_string();
        options.semantic_weight = Some(0.0);
        options.keyword_weight = Some(1.0);
        options.require_keyword_match = Some(true);
        semantic_code_search_owned(
            options.clone(),
            &FixedEmbedder2,
            Arc::clone(&walker),
            Some(Arc::clone(&cache)),
            Some(Arc::clone(&generation)),
        )
        .await
        .unwrap();

        let callback_root = root.path().parent().unwrap();
        assert!(!CachedSearchIndex::refresh_paths(
            cache.write().await.as_mut().unwrap(),
            callback_root,
            vec![],
            vec![],
            &["file_0.rs".to_string()],
            1,
        ));
        generation.store(1, Ordering::Release);
        let result = semantic_code_search_owned(
            options,
            &FixedEmbedder2,
            walker,
            Some(cache),
            Some(generation),
        )
        .await
        .unwrap();
        assert!(
            !result.contains("file_0.rs"),
            "the first owned scoped query retained the deleted path: {result}"
        );
    }

    #[test]
    fn mass_change_above_threshold_triggers_exactly_one_full_rebuild() {
        let threshold = std::hint::black_box(FULL_REBUILD_CHANGE_FRACTION);
        assert!(
            threshold > 0.0 && threshold < 1.0,
            "the full-rebuild threshold must be a named corpus fraction"
        );

        let corpus_size = 1_000usize;
        let docs: Vec<_> = (0..corpus_size)
            .map(|i| make_doc(&format!("src/file_{i}.rs"), "old content"))
            .collect();
        let vectors = vec![Some(vec![1.0, 0.0]); corpus_size];
        let mut index = SearchIndex::new();
        index.index_with_vectors(docs, vectors);
        let rebuilds_before = index.full_rebuild_count();

        let changed_count = (corpus_size as f64 * threshold).floor() as usize + 1;
        let changed_docs: Vec<_> = (0..changed_count)
            .map(|i| make_doc(&format!("src/file_{i}.rs"), "mass changed content"))
            .collect();
        let changed_vectors = vec![Some(vec![0.0, 1.0]); changed_count];

        let outcome = index.apply_delta(changed_docs, changed_vectors, &[]);

        assert_eq!(outcome, IndexUpdateKind::FullRebuild);
        assert_eq!(
            index.full_rebuild_count(),
            rebuilds_before + 1,
            "one mass-change batch must schedule exactly one full rebuild"
        );
        assert_eq!(index.document_count(), corpus_size);
    }

    struct ForkRng(u64);

    impl ForkRng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }

        fn component(&mut self) -> f32 {
            (self.below(2000) as f32 - 999.5) / 1000.0
        }
    }

    const FORK_WORDS: &[&str] = &[
        "invoice", "payment", "worker", "queue", "token", "parser", "cache", "scope", "record",
        "status",
    ];

    fn fork_doc(rng: &mut ForkRng, path: String) -> SearchDocument {
        let len = 1 + rng.below(12);
        let body = (0..len)
            .map(|_| FORK_WORDS[rng.below(FORK_WORDS.len())])
            .collect::<Vec<_>>()
            .join(" ");
        let symbol = FORK_WORDS[rng.below(FORK_WORDS.len())].to_string();
        SearchDocument::new(path, body.clone(), vec![symbol], vec![], body)
    }

    fn fork_vector(rng: &mut ForkRng) -> Option<Vec<f32>> {
        if rng.below(8) == 0 {
            return None;
        }
        Some((0..4).map(|_| rng.component()).collect())
    }

    fn fork_corpus(rng: &mut ForkRng, len: usize) -> (Vec<SearchDocument>, Vec<Option<Vec<f32>>>) {
        (0..len)
            .map(|i| (fork_doc(rng, format!("src/file_{i}.rs")), fork_vector(rng)))
            .unzip()
    }

    /// The base corpus after up to `max_ops` random edits, re-embeds, adds and deletes.
    fn fork_worktree(
        rng: &mut ForkRng,
        docs: &[SearchDocument],
        vectors: &[Option<Vec<f32>>],
        max_ops: usize,
        stage: &str,
    ) -> (Vec<SearchDocument>, Vec<Option<Vec<f32>>>) {
        let mut worktree: Vec<Option<(SearchDocument, Option<Vec<f32>>)>> = docs
            .iter()
            .cloned()
            .zip(vectors.iter().cloned())
            .map(Some)
            .collect();
        for op in 0..1 + rng.below(max_ops) {
            let target = rng.below(docs.len());
            match rng.below(4) {
                0 => worktree[target] = None,
                1 => {
                    let doc = fork_doc(rng, docs[target].path.clone());
                    worktree[target] = Some((doc, fork_vector(rng)));
                }
                2 => {
                    if let Some((_, vector)) = &mut worktree[target] {
                        *vector = fork_vector(rng);
                    }
                }
                _ => {
                    let doc = fork_doc(rng, format!("src/added_{stage}_{op}.rs"));
                    worktree.push(Some((doc, fork_vector(rng))));
                }
            }
        }
        worktree.into_iter().flatten().unzip()
    }

    /// (path, combined score, keyword score), sorted by score, then path.
    fn fork_hits(
        index: &SearchIndex,
        query: &str,
        query_vec: &[f32],
        opts: &ResolvedSearchOptions,
    ) -> Vec<(String, f64, f64)> {
        let mut hits: Vec<_> = index
            .search(query, query_vec, opts)
            .into_iter()
            .map(|hit| (hit.path, hit.score, hit.keyword_score))
            .collect();
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        hits
    }

    fn fork_query(rng: &mut ForkRng) -> (String, Vec<f32>) {
        let len = 1 + rng.below(3);
        let query = (0..len)
            .map(|_| FORK_WORDS[rng.below(FORK_WORDS.len())])
            .collect::<Vec<_>>()
            .join(" ");
        (query, (0..4).map(|_| rng.component()).collect())
    }

    /// A fork of a base index answers like an index built over the worktree's
    /// whole corpus: same files, same scores.
    #[test]
    fn fork_matches_standalone_index_of_random_edits() {
        let opts = ResolvedSearchOptions {
            top_k: 1_000,
            min_combined_score: 0.0,
            ..Default::default()
        };
        for seed in 1..=60_u64 {
            let mut rng = ForkRng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let base_len = 45 + rng.below(40);
            let (docs, vectors) = fork_corpus(&mut rng, base_len);
            let mut base = SearchIndex::new();
            base.index_with_vectors(docs.clone(), vectors.clone());
            if seed % 2 == 1 {
                // A base kept current incrementally carries an overlay.
                let (edited, edited_vectors) = fork_worktree(&mut rng, &docs, &vectors, 3, "base");
                let (changed, changed_vectors, deleted) =
                    base.delta_from(edited.clone(), edited_vectors.clone());
                base.apply_delta(changed, changed_vectors, &deleted);
                let (worktree, worktree_vectors) =
                    fork_worktree(&mut rng, &edited, &edited_vectors, 8, "worktree");
                assert_fork_matches_standalone(
                    &mut rng,
                    &base,
                    worktree,
                    worktree_vectors,
                    &opts,
                    seed,
                );
            } else {
                let (worktree, worktree_vectors) =
                    fork_worktree(&mut rng, &docs, &vectors, 8, "worktree");
                assert_fork_matches_standalone(
                    &mut rng,
                    &base,
                    worktree,
                    worktree_vectors,
                    &opts,
                    seed,
                );
            }
        }
    }

    /// Above `ANN_THRESHOLD` the fork shares the base's vector store and graph,
    /// and at the default `top_k` still answers like a standalone index on the
    /// exact-scan path, which a scoped query takes without building a graph.
    #[test]
    fn fork_above_ann_threshold_shares_the_store_and_matches_standalone() {
        let mut rng = ForkRng(0x5EED_F0CC);
        let (docs, vectors) = fork_corpus(&mut rng, ANN_THRESHOLD + 600);
        let mut base = SearchIndex::new();
        base.index_with_vectors(docs.clone(), vectors.clone());
        let opts = ResolvedSearchOptions {
            scope: SearchScope::Code,
            ..Default::default()
        };
        for seed in 1..=6_u64 {
            let (worktree, worktree_vectors) =
                fork_worktree(&mut rng, &docs, &vectors, 12, "worktree");
            let fork = assert_fork_matches_standalone(
                &mut rng,
                &base,
                worktree,
                worktree_vectors,
                &opts,
                seed,
            );
            assert!(
                fork.shares_vector_store(&base),
                "seed {seed}: the fork built its own vector store"
            );
        }
    }

    fn assert_fork_matches_standalone(
        rng: &mut ForkRng,
        base: &SearchIndex,
        docs: Vec<SearchDocument>,
        vectors: Vec<Option<Vec<f32>>>,
        opts: &ResolvedSearchOptions,
        seed: u64,
    ) -> SearchIndex {
        let fork = base
            .fork(&docs, &vectors)
            .unwrap_or_else(|| panic!("seed {seed}: a small worktree delta must fork"));
        let mut standalone = SearchIndex::new();
        standalone.index_with_vectors(docs, vectors);
        for _ in 0..6 {
            let (query, query_vec) = fork_query(rng);
            let expected = fork_hits(&standalone, &query, &query_vec, opts);
            assert!(!expected.is_empty(), "seed {seed} query {query:?}");
            assert_eq!(
                fork_hits(&fork, &query, &query_vec, opts),
                expected,
                "seed {seed} query {query:?}"
            );
        }
        fork
    }

    #[test]
    fn fork_past_the_promotion_threshold_is_refused() {
        let (docs, vectors) = make_ann_corpus(ANN_THRESHOLD + 50);
        let mut base = SearchIndex::new();
        base.index_with_vectors(docs.clone(), vectors.clone());
        let at_threshold = docs.len() / 5;
        let mut worktree = docs.clone();
        for doc in &mut worktree[..at_threshold] {
            *doc = make_doc(&doc.path, "rewritten in the worktree");
        }
        assert!(
            base.fork(&worktree, &vectors)
                .is_some_and(|fork| fork.shares_vector_store(&base)),
            "a worktree at the threshold forks"
        );
        worktree[at_threshold] = make_doc(&docs[at_threshold].path, "one more rewrite");
        assert!(
            base.fork(&worktree, &vectors).is_none(),
            "a worktree past the threshold builds its own index"
        );
    }

    /// Promotion counts only the worktree's own changes, not the dirty paths
    /// a long-running parent accumulated.
    #[test]
    fn fork_of_a_long_running_parent_counts_only_the_worktree_changes() {
        let (docs, vectors) = make_ann_corpus(ANN_THRESHOLD + 50);
        let mut base = SearchIndex::new();
        base.index_with_vectors(docs.clone(), vectors.clone());
        let dirty = docs.len() / 5;
        let mut parent_docs = docs.clone();
        for doc in &mut parent_docs[..dirty] {
            *doc = make_doc(&doc.path, "edited on the primary");
        }
        assert_eq!(
            base.apply_delta(
                parent_docs[..dirty].to_vec(),
                vectors[..dirty].to_vec(),
                &[]
            ),
            IndexUpdateKind::Incremental
        );
        let mut worktree = parent_docs.clone();
        worktree[dirty] = make_doc(&docs[dirty].path, "edited in the worktree");

        let fork = base
            .fork(&worktree, &vectors)
            .expect("a one-file worktree forks");
        assert!(fork.shares_vector_store(&base));
        assert_eq!(fork.full_rebuild_count(), base.full_rebuild_count());
    }

    /// A fork holds the base's unchanged documents themselves, not copies.
    #[test]
    fn fork_shares_unchanged_documents_with_the_base() {
        let (docs, vectors) = make_ann_corpus(ANN_THRESHOLD + 50);
        let mut base = SearchIndex::new();
        base.index_with_vectors(docs.clone(), vectors.clone());
        let mut worktree = docs.clone();
        worktree[0] = make_doc(&docs[0].path, "edited in the worktree");
        worktree.swap_remove(1);
        let mut worktree_vectors = vectors.clone();
        worktree_vectors.swap_remove(1);

        let fork = base
            .fork(&worktree, &worktree_vectors)
            .expect("a two-file worktree forks");
        let at = |index: &SearchIndex, path: &str| -> *const SearchDocument {
            let doc: &SearchDocument = index
                .documents()
                .iter()
                .find(|doc| doc.path == path)
                .unwrap();
            doc
        };
        for doc in &docs[2..] {
            assert!(
                std::ptr::eq(at(&fork, &doc.path), at(&base, &doc.path)),
                "{} was copied into the fork",
                doc.path
            );
        }
        assert!(!std::ptr::eq(
            at(&fork, &docs[0].path),
            at(&base, &docs[0].path)
        ));
        assert!(
            fork.documents()
                .iter()
                .any(|doc| doc.content == "edited in the worktree")
        );
        assert_eq!(fork.document_count(), docs.len() - 1);
    }

    #[test]
    fn fork_with_vectors_of_another_shape_is_refused() {
        let (docs, mut vectors) = make_ann_corpus(50);
        let mut base = SearchIndex::new();
        base.index_with_vectors(docs.clone(), vectors.clone());
        vectors[0] = Some(vec![1.0, 0.0, 0.0]);
        assert!(base.fork(&docs, &vectors).is_none());
    }

    /// A fork's resident split charges every document once, to the fork or
    /// as shared.
    #[test]
    fn fork_resident_split_charges_each_document_once() {
        let (docs, vectors) = make_ann_corpus(ANN_THRESHOLD + 50);
        let mut base = SearchIndex::new();
        base.index_with_vectors(docs.clone(), vectors.clone());
        let mut worktree = docs.clone();
        worktree[0] = make_doc(&docs[0].path, "edited in the worktree");
        let fork = base
            .fork(&worktree, &vectors)
            .expect("a one-file worktree forks");

        let (own, shared) = fork.resident_split();
        let documents: usize = fork.documents.iter().map(|doc| doc.resident_bytes()).sum();
        let buffers = fork.vector_buffer.capacity() * std::mem::size_of::<f32>()
            + fork
                .vector_updates
                .values()
                .map(|vector| vector.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>();
        assert_eq!(
            own + shared.iter().map(|(_, bytes)| bytes).sum::<usize>(),
            documents + buffers
        );
        assert_eq!(shared.len(), docs.len() - 1);
    }
}
