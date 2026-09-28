//! Identifier-level semantic search with call-site ranking and line metadata.
//!
//! Ports the TypeScript `semantic-identifiers.ts` logic:
//! - Indexes all code symbols (functions, classes, types, etc.) with embeddings
//! - Ranks identifiers by hybrid semantic + keyword score
//! - Finds and ranks call-sites for each top identifier

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use regex::Regex;

use crate::core::walker::FileContents;
use crate::error::{ContextPlusError, Result};
use crate::tools::scoring::{
    DEFAULT_TOP_K, clamp01, keyword_coverage, normalize_weight, truncate_on_char_boundary,
};
use crate::tools::semantic_search::{cosine, sanitize_query, split_camel_case};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Identifier search uses a higher semantic weight than the file-level default
/// (0.72) because code symbol names are denser and keyword overlap alone is
/// insufficient to distinguish semantically similar identifiers.
const IDENTIFIER_SEMANTIC_WEIGHT: f64 = 0.78;
const IDENTIFIER_KEYWORD_WEIGHT: f64 = 0.22;

// Aliases so call sites that reference DEFAULT_SEMANTIC_WEIGHT / DEFAULT_KEYWORD_WEIGHT
// continue to resolve to the identifier-tuned values.
const DEFAULT_SEMANTIC_WEIGHT: f64 = IDENTIFIER_SEMANTIC_WEIGHT;
const DEFAULT_KEYWORD_WEIGHT: f64 = IDENTIFIER_KEYWORD_WEIGHT;

const DEFAULT_TOP_CALLS: usize = 10;
const MAX_TOP_K: usize = 50;

/// Call-site ranking uses an even higher semantic weight (0.82) because the
/// surrounding context snippet is short and keyword presence is noisy.
const CALLSITE_SEMANTIC_WEIGHT: f64 = 0.82;
const CALLSITE_KEYWORD_WEIGHT: f64 = 0.18;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SemanticIdentifierSearchOptions {
    pub root_dir: PathBuf,
    pub query: String,
    pub top_k: Option<usize>,
    pub top_calls_per_identifier: Option<usize>,
    pub include_kinds: Option<Vec<String>>,
    pub semantic_weight: Option<f64>,
    pub keyword_weight: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct IdentifierDoc {
    pub id: String,
    pub path: String,
    pub header: String,
    pub name: String,
    pub kind: String,
    /// Lowercased `kind` — pre-computed at index time to avoid per-query
    /// `.to_lowercase()` allocations inside `score_identifiers`.
    pub kind_lower: String,
    pub line: usize,
    pub end_line: usize,
    pub signature: String,
    pub parent_name: Option<String>,
    pub text: String,
    pub name_token_set: HashSet<String>,
    pub signature_token_set: HashSet<String>,
    pub parent_token_set: HashSet<String>,
}

impl IdentifierDoc {
    /// Build one field's keyword evidence once, while constructing the index.
    pub fn evidence_tokens(text: &str) -> HashSet<String> {
        identifier_terms(text)
    }
}

#[derive(Debug, Clone)]
pub struct RankedIdentifier {
    pub doc: IdentifierDoc,
    pub semantic_score: f64,
    pub keyword_score: f64,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct CallSite {
    pub file: String,
    pub line: usize,
    pub context: String,
    pub semantic_score: f64,
    pub keyword_score: f64,
    pub score: f64,
}

#[derive(Debug)]
pub struct CallSiteResult {
    pub sites: Vec<CallSite>,
    pub total: usize,
}

// ---------------------------------------------------------------------------
// Definition line detection (shared with blast_radius)
// ---------------------------------------------------------------------------

/// Check if a line defines (rather than uses) a symbol.
/// Matches: function, class, enum, interface, struct, type, trait, fn, def, func,
/// const, let, var, pub, export (with optional async function).
pub fn is_definition_line(line: &str, symbol_name: &str) -> bool {
    let trimmed = line.trim_start();

    // Pattern 1: function/class/enum/interface/struct/type/trait/fn/def/func <name>
    if let Some(rest) = strip_definition_keyword_1(trimmed) {
        // The symbol name should appear as the next identifier
        let rest = rest.trim_start();
        if let Some(after) = rest.strip_prefix(symbol_name)
            && (after.is_empty()
                || after.starts_with('(')
                || after.starts_with('<')
                || after.starts_with(' ')
                || after.starts_with(':')
                || after.starts_with('{'))
        {
            return true;
        }
    }

    // Pattern 2: const/let/var/pub/export [async] [function] <name>
    // The symbol must be the name being declared, not a call expression
    if let Some(rest) = strip_definition_keyword_2(trimmed) {
        let rest = rest.trim_start();
        // Skip optional `async function` after export
        let rest = rest.strip_prefix("async ").unwrap_or(rest);
        let rest = rest
            .strip_prefix("function ")
            .or_else(|| rest.strip_prefix("fn "))
            .or_else(|| rest.strip_prefix("class "))
            .or_else(|| rest.strip_prefix("enum "))
            .or_else(|| rest.strip_prefix("interface "))
            .or_else(|| rest.strip_prefix("type "))
            .unwrap_or(rest)
            .trim_start();
        if let Some(after) = rest.strip_prefix(symbol_name)
            && (after.is_empty()
                || after.starts_with('(')
                || after.starts_with('<')
                || after.starts_with(' ')
                || after.starts_with(':')
                || after.starts_with('{')
                || after.starts_with('='))
        {
            return true;
        }
    }

    false
}

fn strip_definition_keyword_1(line: &str) -> Option<&str> {
    let keywords = [
        "function ",
        "class ",
        "enum ",
        "interface ",
        "struct ",
        "type ",
        "trait ",
        "fn ",
        "def ",
        "func ",
    ];
    for kw in &keywords {
        if let Some(rest) = line.strip_prefix(kw) {
            return Some(rest);
        }
        // Also handle `export function`, `async function`, `pub fn`, etc.
        if let Some(rest) = line
            .strip_prefix("export ")
            .or_else(|| line.strip_prefix("pub "))
            .or_else(|| line.strip_prefix("pub(crate) "))
            .or_else(|| line.strip_prefix("async "))
        {
            if let Some(rest2) = rest.strip_prefix(kw) {
                return Some(rest2);
            }
            // Handle `export async function`
            if let Some(rest2) = rest.strip_prefix("async ")
                && let Some(rest3) = rest2.strip_prefix(kw)
            {
                return Some(rest3);
            }
        }
    }
    None
}

fn strip_definition_keyword_2(line: &str) -> Option<&str> {
    let keywords = ["const ", "let ", "var "];
    for kw in &keywords {
        if let Some(rest) = line.strip_prefix(kw) {
            return Some(rest);
        }
        // Handle `export const`, `pub const`, etc.
        if let Some(rest) = line
            .strip_prefix("export ")
            .or_else(|| line.strip_prefix("pub "))
            && let Some(rest2) = rest.strip_prefix(kw)
        {
            return Some(rest2);
        }
    }
    None
}

/// Escape special regex characters in a string (prevents ReDoS).
pub fn escape_regex(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '.' | '*' | '+' | '?' | '^' | '$' | '{' | '}' | '(' | ')' | '|' | '[' | ']' | '\\' => {
                escaped.push('\\');
                escaped.push(c);
            }
            _ => escaped.push(c),
        }
    }
    escaped
}

// ---------------------------------------------------------------------------
// Keyword coverage
// ---------------------------------------------------------------------------

/// Tokenize `input` with `identifier_terms` then delegate to
/// `scoring::keyword_coverage` for uncached call-site snippets.
fn get_keyword_coverage(query_terms: &HashSet<String>, input: &str) -> f64 {
    keyword_coverage(query_terms, &identifier_terms(input))
}

/// Whole lowercased identifiers in `text` (`selectableScopes` ->
/// `selectablescopes`), the unsplit counterpart of `split_camel_case`.
fn whole_identifiers(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| w.len() > 1)
        .map(str::to_lowercase)
}

/// camelCase / snake_case parts plus each whole identifier, so a query naming
/// `selectableScopes` only fully matches an identifier with that exact name.
pub(crate) fn identifier_terms(text: &str) -> HashSet<String> {
    let mut terms: HashSet<String> = split_camel_case(text).into_iter().collect();
    terms.extend(whole_identifiers(text));
    terms
}

fn format_line_range(line: usize, end_line: usize) -> String {
    if end_line > line {
        format!("L{}-L{}", line, end_line)
    } else {
        format!("L{}", line)
    }
}

fn normalize_kinds(kinds: &Option<Vec<String>>) -> Option<HashSet<String>> {
    let kinds = kinds.as_ref()?;
    let normalized: HashSet<String> = kinds
        .iter()
        .map(|k| k.trim().to_lowercase())
        .filter(|k| !k.is_empty())
        .collect();
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

// ---------------------------------------------------------------------------
// Call-site ranking
// ---------------------------------------------------------------------------

/// Find and rank call-sites for a given symbol across all file content.
/// `file_content` maps relative_path -> raw file content (`Arc<String>`).
/// `query_vec` and `query_terms` are from the user query.
/// Returns the top `limit` call-sites ranked by hybrid score.
pub fn rank_call_sites(
    query_terms: &HashSet<String>,
    query_vec: &[f32],
    symbol: &IdentifierDoc,
    file_content: &FileContents,
    limit: usize,
    // Optional: pre-computed vectors for call-site text. If None, only keyword score is used.
    callsite_vectors: Option<&dyn CallSiteVectorProvider>,
) -> CallSiteResult {
    let escaped_name = escape_regex(&symbol.name);
    let pattern_str = if symbol.kind == "function" || symbol.kind == "method" {
        format!(r"\b{}\s*\(", escaped_name)
    } else {
        format!(r"\b{}\b", escaped_name)
    };
    let call_pattern = match Regex::new(&pattern_str) {
        Ok(re) => re,
        Err(_) => {
            return CallSiteResult {
                sites: vec![],
                total: 0,
            };
        }
    };

    let embed_budget = (limit * 4).max(30);
    let externally_visible = file_content.get(&symbol.path).is_some_and(|content| {
        let ext = Path::new(&symbol.path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        crate::core::tree_sitter::identifier_is_exported(content, ext, &symbol.name, symbol.line)
    });

    // Lazy per-file line-split: only files that pass the cheap substring
    // pre-filter on the full content get split. This avoids splitting every
    // file in the corpus just to skip files that don't mention the symbol.
    // (Review #59 F1.)
    let mut file_entries: Vec<(&String, Vec<&str>)> = Vec::new();

    // Collect candidates as (file_index, line_number, line_offset, keyword_score)
    // to avoid cloning file Strings and context Strings in the inner loop.
    let mut candidates: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut keyword_buf = String::with_capacity(512);

    for (file, content) in file_content.iter() {
        // Fast pre-filter on the full file content (no allocation, no split):
        // skip files that don't contain the symbol name at all.
        if !content.contains(symbol.name.as_str()) {
            continue;
        }
        let ext = Path::new(file)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        if !crate::core::tree_sitter::get_supported_extensions()
            .iter()
            .any(|supported| supported.trim_start_matches('.') == ext)
        {
            continue;
        }
        if *file != symbol.path
            && (!externally_visible
                || !imports_definition(file, content, ext, symbol, file_content))
        {
            continue;
        }

        // Matching file — split into lines once and record it for the
        // ranked-output phase.
        let lines: Vec<&str> = content.lines().collect();
        let fi = file_entries.len();
        file_entries.push((file, lines));
        // Use index access (vec doesn't reallocate across this inner loop
        // because we don't push again inside it).
        let lines = &file_entries[fi].1;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("import ")
                || trimmed.starts_with("use ")
                || trimmed.starts_with("//")
                || trimmed.starts_with("/*")
                || trimmed.starts_with('*')
                || trimmed.starts_with('#')
                || !call_pattern.is_match(line)
            {
                continue;
            }
            // Skip the symbol's own definition line
            if *file == symbol.path && i + 1 == symbol.line {
                continue;
            }
            if is_definition_line(line, &symbol.name) {
                continue;
            }

            let context = line.trim();
            let context = truncate_on_char_boundary(context, 220);
            // Reuse buffer instead of format! allocation per iteration
            keyword_buf.clear();
            keyword_buf.push_str(file);
            keyword_buf.push(' ');
            keyword_buf.push_str(context);
            let keyword_score = get_keyword_coverage(query_terms, &keyword_buf);
            candidates.push((fi, i + 1, i, keyword_score));
        }
    }

    if candidates.is_empty() {
        return CallSiteResult {
            sites: vec![],
            total: 0,
        };
    }

    let total = candidates.len();

    // Sample top candidates by keyword score for embedding
    candidates.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(embed_budget);

    let mut ranked: Vec<CallSite> = candidates
        .iter()
        .map(|(fi, line_num, line_idx, keyword_score)| {
            let (file, lines) = &file_entries[*fi];
            let raw_line = lines[*line_idx];
            let context = raw_line.trim();
            let context = truncate_on_char_boundary(context, 220);

            let semantic_score = callsite_vectors
                .and_then(|provider| {
                    let text = format!("{} {}", file, context);
                    provider.get_vector(&text).map(|vec| {
                        let sim = cosine(query_vec, &vec);
                        sim.max(0.0)
                    })
                })
                .unwrap_or(0.0);

            let score = clamp01(
                semantic_score * CALLSITE_SEMANTIC_WEIGHT + keyword_score * CALLSITE_KEYWORD_WEIGHT,
            );
            CallSite {
                file: (*file).clone(),
                line: *line_num,
                context: context.to_string(),
                semantic_score,
                keyword_score: *keyword_score,
                score,
            }
        })
        .collect();

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(limit.max(1));

    CallSiteResult {
        sites: ranked,
        total,
    }
}

fn imports_definition(
    file: &str,
    content: &str,
    ext: &str,
    symbol: &IdentifierDoc,
    files: &FileContents,
) -> bool {
    crate::core::tree_sitter::extract_identifier_imports(content, ext)
        .iter()
        .filter_map(|import| {
            if ext == "rs" {
                return resolve_rust_identifier_import(import, file, files);
            }
            crate::core::import_resolver::resolve_import_with(import, Path::new(file), |path| {
                files.contains_key(&path.to_string_lossy())
            })
        })
        .any(|path| path == Path::new(&symbol.path))
}

fn resolve_rust_identifier_import(
    import: &str,
    file: &str,
    files: &FileContents,
) -> Option<PathBuf> {
    let mut parts = import.split("::").peekable();
    let mut base = Path::new(file).parent()?.to_path_buf();
    if matches!(parts.peek().copied(), Some("self" | "super")) {
        let stem = Path::new(file).file_stem()?.to_str()?;
        if !matches!(stem, "lib" | "main" | "mod") {
            base.push(stem);
        }
    }
    match parts.peek().copied()? {
        "crate" => {
            parts.next();
            while base.file_name().is_some_and(|name| name != "src") {
                if !base.pop() {
                    return None;
                }
            }
        }
        "self" => {
            parts.next();
        }
        "super" => {
            while parts.peek() == Some(&"super") {
                parts.next();
                base.pop();
            }
        }
        _ => return None,
    }
    let mut resolved = None;
    for part in parts {
        if part.starts_with('{') || part == "*" {
            break;
        }
        base.push(part.split_whitespace().next()?);
        for candidate in [base.with_extension("rs"), base.join("mod.rs")] {
            if files.contains_key(&candidate.to_string_lossy()) {
                resolved = Some(candidate);
            }
        }
    }
    resolved
}

/// Trait for providing pre-computed vectors for call-site text.
pub trait CallSiteVectorProvider {
    fn get_vector(&self, text: &str) -> Option<Vec<f32>>;
}

// ---------------------------------------------------------------------------
// Identifier scoring
// ---------------------------------------------------------------------------

/// Score identifiers against a query using hybrid semantic + keyword ranking.
/// Uses index-based scoring to avoid cloning IdentifierDoc per candidate — only
/// the final top_k docs are cloned.
#[allow(clippy::too_many_arguments)]
pub fn score_identifiers(
    docs: &[IdentifierDoc],
    query_vec: &[f32],
    query_terms: &HashSet<String>,
    vector_buffer: &[f32],
    vector_dims: usize,
    include_kinds: &Option<HashSet<String>>,
    semantic_weight: f64,
    keyword_weight: f64,
    top_k: usize,
) -> Vec<RankedIdentifier> {
    score_identifier_candidates(
        docs,
        query_vec,
        query_terms,
        vector_buffer,
        vector_dims,
        include_kinds,
        semantic_weight,
        keyword_weight,
        top_k,
        None,
    )
}

pub trait IndexData<T>: Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> &T;
    fn slice(&self, range: std::ops::Range<usize>) -> &[T];
}

impl<T: Sync> IndexData<T> for [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    fn get(&self, index: usize) -> &T {
        &self[index]
    }
    fn slice(&self, range: std::ops::Range<usize>) -> &[T] {
        &self[range]
    }
}

impl<T: Sync> IndexData<T> for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn get(&self, index: usize) -> &T {
        &self[index]
    }
    fn slice(&self, range: std::ops::Range<usize>) -> &[T] {
        &self[range]
    }
}

#[allow(clippy::too_many_arguments)]
fn score_identifier_candidates(
    docs: &(impl IndexData<IdentifierDoc> + ?Sized),
    query_vec: &[f32],
    query_terms: &HashSet<String>,
    vector_buffer: &(impl IndexData<f32> + ?Sized),
    vector_dims: usize,
    include_kinds: &Option<HashSet<String>>,
    semantic_weight: f64,
    keyword_weight: f64,
    top_k: usize,
    candidates: Option<&[usize]>,
) -> Vec<RankedIdentifier> {
    // Phase 1: Score all docs, collecting only indices + scores (no clone).
    let mut scored: Vec<(usize, f64, f64, f64)> = (0..candidates
        .map_or(docs.len(), <[usize]>::len))
        .into_par_iter()
        .filter_map(|position| {
            let i = candidates.map_or(position, |indices| indices[position]);
            let doc = docs.get(i);
            if let Some(kinds) = include_kinds
                && !kinds.contains(&doc.kind_lower)
            {
                return None;
            }

            let offset = i * vector_dims;
            if offset + vector_dims > vector_buffer.len() {
                return None;
            }
            let vec_slice = vector_buffer.slice(offset..offset + vector_dims);
            let semantic_score =
                crate::core::embeddings::cosine_similarity_simsimd(query_vec, vec_slice).max(0.0)
                    as f64;

            let name_score = keyword_coverage(query_terms, &doc.name_token_set);
            let signature_score = keyword_coverage(query_terms, &doc.signature_token_set);
            let parent_score = keyword_coverage(query_terms, &doc.parent_token_set);
            let keyword_score = if name_score > 0.0 || signature_score > 0.0 {
                let evidence = name_score.max(signature_score * 0.6);
                evidence + (1.0 - evidence) * parent_score * 0.3
            } else {
                0.0
            };
            if semantic_weight == 0.0 && keyword_weight > 0.0 && keyword_score == 0.0 {
                return None;
            }

            let total_weight = semantic_weight + keyword_weight;
            let score = if total_weight > 0.0 {
                clamp01(
                    (semantic_weight * semantic_score + keyword_weight * keyword_score)
                        / total_weight,
                )
            } else {
                semantic_score
            };

            Some((i, score, semantic_score, keyword_score))
        })
        .collect();

    // Phase 2: Partial sort + truncate to top_k.
    let top_k = top_k.min(scored.len());
    if top_k == 0 {
        return Vec::new();
    }
    scored.select_nth_unstable_by(top_k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(top_k);
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Phase 3: Clone only the top_k docs.
    scored
        .into_iter()
        .map(
            |(idx, score, semantic_score, keyword_score)| RankedIdentifier {
                doc: docs.get(idx).clone(),
                semantic_score,
                keyword_score,
                score,
            },
        )
        .collect()
}

// ---------------------------------------------------------------------------
// Format output
// ---------------------------------------------------------------------------

/// Format identifier search results with call-sites as text output.
pub fn format_identifier_results(
    query: &str,
    ranked: &[RankedIdentifier],
    call_results: &[CallSiteResult],
) -> String {
    if ranked.is_empty() {
        return "No supported identifiers found for semantic identifier search.".to_string();
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "Top {} identifier matches for: \"{}\"",
        ranked.len(),
        query
    ));
    lines.push(String::new());

    for (i, item) in ranked.iter().enumerate() {
        let range = format_line_range(item.doc.line, item.doc.end_line);
        lines.push(format!(
            "{}. {} {} - {} ({})",
            i + 1,
            item.doc.kind,
            item.doc.name,
            item.doc.path,
            range
        ));
        lines.push(format!(
            "   Score: {}% | Semantic: {}% | Keyword: {}%",
            (item.score * 1000.0).round() / 10.0,
            (item.semantic_score * 1000.0).round() / 10.0,
            (item.keyword_score * 1000.0).round() / 10.0,
        ));
        lines.push(format!("   Signature: {}", item.doc.signature));
        if let Some(ref parent) = item.doc.parent_name {
            lines.push(format!("   Parent: {}", parent));
        }

        if let Some(calls) = call_results.get(i) {
            if calls.sites.is_empty() {
                lines.push("   Calls: none found".to_string());
            } else {
                lines.push(format!("   Calls ({}/{}):", calls.sites.len(), calls.total));
                for (j, site) in calls.sites.iter().enumerate() {
                    lines.push(format!(
                        "     {}. {}:L{} ({}%) {}",
                        j + 1,
                        site.file,
                        site.line,
                        (site.score * 1000.0).round() / 10.0,
                        site.context
                    ));
                }
            }
        } else {
            lines.push("   Calls: none found".to_string());
        }
        lines.push(String::new());
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// High-level entry point
// ---------------------------------------------------------------------------

/// Run semantic identifier search.
/// Caller provides pre-built identifier index data and embedding functions.
#[allow(clippy::too_many_arguments)]
pub async fn semantic_identifier_search(
    options: SemanticIdentifierSearchOptions,
    embed_fn: &dyn crate::tools::semantic_search::EmbedFn,
    identifier_docs: &(impl IndexData<IdentifierDoc> + ?Sized),
    vector_buffer: &(impl IndexData<f32> + ?Sized),
    vector_dims: usize,
    file_content: &FileContents,
    candidates: Option<&[usize]>,
) -> Result<String> {
    let query = sanitize_query(&options.query);
    if query.is_empty() {
        return Ok("No supported identifiers found for semantic identifier search.".to_string());
    }

    let top_k = options.top_k.unwrap_or(DEFAULT_TOP_K).clamp(1, MAX_TOP_K);
    let top_calls = options
        .top_calls_per_identifier
        .unwrap_or(DEFAULT_TOP_CALLS)
        .max(1);
    let semantic_weight = normalize_weight(options.semantic_weight, DEFAULT_SEMANTIC_WEIGHT);
    let keyword_weight = normalize_weight(options.keyword_weight, DEFAULT_KEYWORD_WEIGHT);
    let include_kinds = normalize_kinds(&options.include_kinds);

    if identifier_docs.is_empty() {
        return Ok("No supported identifiers found for semantic identifier search.".to_string());
    }

    // Get query embedding — embed takes &[String], so convert Cow<str> to String only once.
    let query_string = query.as_ref().to_string();
    let query_vecs = embed_fn.embed(std::slice::from_ref(&query_string)).await?;
    let query_vec = query_vecs
        .into_iter()
        .next()
        .ok_or_else(|| ContextPlusError::Ollama("Empty embedding response".into()))?;
    let query_terms = identifier_terms(query.as_ref());

    // Score identifiers
    let top = score_identifier_candidates(
        identifier_docs,
        &query_vec,
        &query_terms,
        vector_buffer,
        vector_dims,
        &include_kinds,
        semantic_weight,
        keyword_weight,
        top_k,
        candidates,
    );

    if top.is_empty() {
        return Ok("No identifiers matched the requested kind filters.".to_string());
    }

    // Rank call-sites for each top identifier — parallel across identifiers.
    // Each call scans the full file_content corpus; doing them in parallel via rayon
    // reduces wall time from O(top_k × corpus_lines) to O(corpus_lines) on
    // machines with ≥ top_k cores.
    let call_results: Vec<CallSiteResult> = top
        .par_iter()
        .map(|item| {
            rank_call_sites(
                &query_terms,
                &query_vec,
                &item.doc,
                file_content,
                top_calls,
                None,
            )
        })
        .collect();

    Ok(format_identifier_results(
        query.as_ref(),
        &top,
        &call_results,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Compute the vector norm (L2) of a vector.
    /// Only used in tests — moved here to eliminate dead code warning.
    fn vector_norm(vec: &[f32]) -> f64 {
        let mut sum: f64 = 0.0;
        for &v in vec {
            sum += (v as f64) * (v as f64);
        }
        sum.sqrt()
    }

    // -- is_definition_line tests --

    #[test]
    fn test_is_definition_line_function() {
        assert!(is_definition_line(
            "export function getUserById(id: string) {",
            "getUserById"
        ));
    }

    #[test]
    fn test_is_definition_line_class() {
        assert!(is_definition_line("class UserService {", "UserService"));
    }

    #[test]
    fn test_is_definition_line_const() {
        assert!(is_definition_line(
            "const getUserById = async () => {",
            "getUserById"
        ));
    }

    #[test]
    fn test_is_definition_line_rust_fn() {
        assert!(is_definition_line(
            "pub fn get_user_by_id(id: &str) -> User {",
            "get_user_by_id"
        ));
    }

    #[test]
    fn test_is_definition_line_usage() {
        assert!(!is_definition_line(
            "  const result = getUserById(id);",
            "getUserById"
        ));
    }

    #[test]
    fn test_is_definition_line_import() {
        // import has neither function/class nor const/let keywords
        assert!(!is_definition_line(
            "import { getUserById } from './user';",
            "getUserById"
        ));
    }

    // -- escape_regex tests --

    #[test]
    fn test_escape_regex_special_chars() {
        assert_eq!(escape_regex("foo.bar"), r"foo\.bar");
        assert_eq!(escape_regex("a*b+c?"), r"a\*b\+c\?");
        assert_eq!(escape_regex("no_specials"), "no_specials");
    }

    #[test]
    fn test_escape_regex_all_special() {
        let input = ".*+?^${}()|[]\\";
        let escaped = escape_regex(input);
        // Every char should be prefixed with backslash
        assert_eq!(escaped, r"\.\*\+\?\^\$\{\}\(\)\|\[\]\\");
    }

    // -- get_keyword_coverage tests --

    #[test]
    fn test_keyword_coverage_full() {
        let query: HashSet<String> = ["user", "get"].iter().map(|s| s.to_string()).collect();
        let coverage = get_keyword_coverage(&query, "getUserById returns user data");
        assert!((coverage - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_keyword_coverage_partial() {
        let query: HashSet<String> = ["user", "delete"].iter().map(|s| s.to_string()).collect();
        let coverage = get_keyword_coverage(&query, "getUserById returns user data");
        assert!((coverage - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_keyword_coverage_empty() {
        let query: HashSet<String> = HashSet::new();
        assert_eq!(get_keyword_coverage(&query, "anything"), 0.0);
    }

    #[test]
    fn exact_identifier_name_outranks_shared_parts_under_keyword_weights() {
        let path = "packages/platform/context/src/scope-authz-plugin.ts";
        let make_doc = |name: &str, kind: &str, sig: &str, line: usize| IdentifierDoc {
            id: format!("{path}:{name}:{line}"),
            path: path.to_string(),
            header: String::new(),
            name: name.to_string(),
            kind: kind.to_string(),
            kind_lower: kind.to_lowercase(),
            line,
            end_line: line,
            signature: sig.to_string(),
            parent_name: None,
            text: format!("{name} {kind} {sig} {path}"),
            name_token_set: crate::tools::semantic_identifiers::identifier_terms(name),
            signature_token_set: crate::tools::semantic_identifiers::identifier_terms(sig),
            parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
        };
        let docs = vec![
            make_doc(
                "mode",
                "const",
                "const mode = resolveScopeMode(request);",
                670,
            ),
            make_doc(
                "selectableScopes",
                "function",
                "function selectableScopes(raw: OrgScope | readonly OrgScope[]): Set<OrgScope>",
                92,
            ),
        ];
        let query_terms = identifier_terms("resolveScopeMode selectableScopes");
        let results = score_identifiers(
            &docs,
            &[1.0, 0.0],
            &query_terms,
            &[1.0, 0.0, 1.0, 0.0],
            2,
            &None,
            0.0,
            1.0,
            2,
        );
        assert_eq!(results[0].doc.name, "selectableScopes");
        assert!(results[0].keyword_score > results[1].keyword_score);
    }

    fn keyword_test_doc(
        name: &str,
        signature: &str,
        parent_name: Option<&str>,
        header: &str,
        text: &str,
        line: usize,
    ) -> IdentifierDoc {
        IdentifierDoc {
            id: format!("src/grants.ts:{name}:{line}"),
            path: "src/grants.ts".to_string(),
            header: header.to_string(),
            name: name.to_string(),
            kind: "const".to_string(),
            kind_lower: "const".to_string(),
            line,
            end_line: line,
            signature: signature.to_string(),
            parent_name: parent_name.map(str::to_string),
            text: text.to_string(),
            name_token_set: crate::tools::semantic_identifiers::identifier_terms(name),
            signature_token_set: crate::tools::semantic_identifiers::identifier_terms(signature),
            parent_token_set: crate::tools::semantic_identifiers::identifier_terms(
                (parent_name.map(str::to_string)).as_deref().unwrap_or(""),
            ),
        }
    }

    #[test]
    fn keyword_score_prefers_name_over_signature_and_partial_name() {
        let docs = vec![
            keyword_test_doc(
                "cascadeGrants",
                "const cascadeGrants = resolve();",
                None,
                "",
                "const cascadeGrants = resolve();",
                1,
            ),
            keyword_test_doc(
                "applyPolicy",
                "const applyPolicy = (input: CascadeGrants) => input;",
                None,
                "",
                "const applyPolicy = (input: CascadeGrants) => input;",
                2,
            ),
            keyword_test_doc(
                "cascadeWorker",
                "const cascadeWorker = resolve();",
                None,
                "",
                "const cascadeWorker = resolve();",
                3,
            ),
        ];
        let results = score_identifiers(
            &docs,
            &[1.0],
            &identifier_terms("cascade grants"),
            &[1.0, 1.0, 1.0],
            1,
            &None,
            0.0,
            1.0,
            docs.len(),
        );
        let score = |name: &str| {
            results
                .iter()
                .find(|result| result.doc.name == name)
                .unwrap()
                .keyword_score
        };

        assert!(
            score("cascadeGrants") > score("applyPolicy"),
            "an exact name match must outrank signature-only evidence: {results:?}"
        );
        assert!(
            score("cascadeGrants") > score("cascadeWorker"),
            "a full name match must outrank a single-part name match: {results:?}"
        );
    }

    #[test]
    fn keyword_score_requires_name_or_signature_evidence_before_parent_context() {
        let docs = vec![
            keyword_test_doc(
                "execute",
                "const execute = () => {};",
                Some("CascadeCoordinator"),
                "",
                "const execute = () => {};",
                1,
            ),
            keyword_test_doc(
                "foreignOrgId",
                "const foreignOrgId: string;",
                None,
                "cascade cleanup helpers",
                "const foreignOrgId: string;",
                2,
            ),
            keyword_test_doc(
                "patientId",
                "const patientId: string;",
                None,
                "",
                "const patientId: string; // cascade cleanup",
                3,
            ),
            keyword_test_doc(
                "cascadeWorker",
                "const cascadeWorker = () => {};",
                Some("GrantCoordinator"),
                "",
                "const cascadeWorker = () => {};",
                4,
            ),
            keyword_test_doc(
                "cascadeTask",
                "const cascadeTask = () => {};",
                None,
                "",
                "const cascadeTask = () => {};",
                5,
            ),
        ];
        let results = score_identifiers(
            &docs,
            &[1.0],
            &identifier_terms("cascade grant"),
            &[1.0; 5],
            1,
            &None,
            1.0,
            1.0,
            docs.len(),
        );
        let score = |line: usize| {
            results
                .iter()
                .find(|result| result.doc.line == line)
                .unwrap()
                .keyword_score
        };

        assert_eq!(
            score(1),
            0.0,
            "parent-only evidence must not grant execute keyword credit: {results:?}"
        );
        assert_eq!(
            score(2),
            0.0,
            "module comments/header text must not contribute keyword evidence"
        );
        assert_eq!(
            score(3),
            0.0,
            "identifier body text must not contribute keyword evidence"
        );
        assert!(
            score(4) > score(5),
            "parent context should improve rank only after name/signature evidence passes the gate: {results:?}"
        );
    }

    #[test]
    fn keyword_only_scoring_excludes_parent_only_candidate() {
        let docs = vec![
            keyword_test_doc(
                "execute",
                "const execute = () => {};",
                Some("CascadeCoordinator"),
                "",
                "const execute = () => {};",
                1,
            ),
            keyword_test_doc(
                "cascadeGrants",
                "const cascadeGrants = () => {};",
                None,
                "",
                "const cascadeGrants = () => {};",
                2,
            ),
        ];
        let results = score_identifiers(
            &docs,
            &[1.0],
            &identifier_terms("cascade"),
            &[1.0, 1.0],
            1,
            &None,
            0.0,
            1.0,
            docs.len(),
        );

        assert!(
            results
                .iter()
                .any(|result| result.doc.name == "cascadeGrants"),
            "name evidence should remain eligible: {results:?}"
        );
        assert!(
            results.iter().all(|result| result.doc.name != "execute"),
            "parent-only execute must be absent from keyword-only results: {results:?}"
        );
    }

    #[test]
    fn keyword_only_scoring_drops_identifiers_without_identifier_evidence() {
        let docs = vec![
            keyword_test_doc(
                "cascadeGrants",
                "const cascadeGrants = resolve();",
                None,
                "",
                "const cascadeGrants = resolve();",
                1,
            ),
            keyword_test_doc(
                "userId",
                "const userId: string;",
                None,
                "",
                "const userId: string; // cascade grants are recalculated here",
                2,
            ),
        ];
        let results = score_identifiers(
            &docs,
            &[1.0],
            &identifier_terms("cascade grants"),
            &[1.0, 1.0],
            1,
            &None,
            0.0,
            1.0,
            docs.len(),
        );

        assert_eq!(
            results.len(),
            1,
            "keyword-only search must omit zero-evidence identifiers: {results:?}"
        );
        assert_eq!(results[0].doc.name, "cascadeGrants");
    }

    // -- vector_norm tests --

    #[test]
    fn test_vector_norm() {
        let v = vec![3.0, 4.0];
        assert!((vector_norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_norm_zero() {
        let v = vec![0.0, 0.0, 0.0];
        assert_eq!(vector_norm(&v), 0.0);
    }

    // -- format_line_range tests --

    #[test]
    fn test_format_line_range_single() {
        assert_eq!(format_line_range(10, 10), "L10");
    }

    #[test]
    fn test_format_line_range_multi() {
        assert_eq!(format_line_range(10, 25), "L10-L25");
    }

    // -- score_identifiers tests --

    #[test]
    fn test_score_identifiers_basic() {
        let docs = vec![
            IdentifierDoc {
                id: "src/user.ts:getUserById:10".to_string(),
                path: "src/user.ts".to_string(),
                header: "user service".to_string(),
                name: "getUserById".to_string(),
                kind: "function".to_string(),
                kind_lower: "function".to_string(),
                line: 10,
                end_line: 25,
                signature: "getUserById(id: string): User".to_string(),
                parent_name: None,
                text: "getUserById function getUserById(id: string): User src/user.ts user service"
                    .to_string(),
                name_token_set: crate::tools::semantic_identifiers::identifier_terms("getUserById"),
                signature_token_set: crate::tools::semantic_identifiers::identifier_terms(
                    "getUserById(id: string): User",
                ),
                parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
            },
            IdentifierDoc {
                id: "src/db.ts:connect:5".to_string(),
                path: "src/db.ts".to_string(),
                header: "database".to_string(),
                name: "connect".to_string(),
                kind: "function".to_string(),
                kind_lower: "function".to_string(),
                line: 5,
                end_line: 15,
                signature: "connect(): Connection".to_string(),
                parent_name: None,
                text: "connect function connect(): Connection src/db.ts database".to_string(),
                name_token_set: crate::tools::semantic_identifiers::identifier_terms("connect"),
                signature_token_set: crate::tools::semantic_identifiers::identifier_terms(
                    "connect(): Connection",
                ),
                parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
            },
        ];

        // Fake vectors: getUserById closer to query
        let query_vec = vec![1.0, 0.0, 0.0];
        let vector_buffer = vec![
            0.9, 0.1, 0.0, // getUserById
            0.1, 0.9, 0.0, // connect
        ];
        let query_terms: HashSet<String> = ["user", "get"].iter().map(|s| s.to_string()).collect();

        let results = score_identifiers(
            &docs,
            &query_vec,
            &query_terms,
            &vector_buffer,
            3,
            &None,
            0.78,
            0.22,
            5,
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc.name, "getUserById");
    }

    #[test]
    fn test_score_identifiers_kind_filter() {
        let docs = vec![
            IdentifierDoc {
                id: "src/user.ts:User:1".to_string(),
                path: "src/user.ts".to_string(),
                header: "types".to_string(),
                name: "User".to_string(),
                kind: "class".to_string(),
                kind_lower: "class".to_string(),
                line: 1,
                end_line: 20,
                signature: "class User".to_string(),
                parent_name: None,
                text: "User class".to_string(),
                name_token_set: crate::tools::semantic_identifiers::identifier_terms("User"),
                signature_token_set: crate::tools::semantic_identifiers::identifier_terms(
                    "class User",
                ),
                parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
            },
            IdentifierDoc {
                id: "src/user.ts:getUser:25".to_string(),
                path: "src/user.ts".to_string(),
                header: "types".to_string(),
                name: "getUser".to_string(),
                kind: "function".to_string(),
                kind_lower: "function".to_string(),
                line: 25,
                end_line: 30,
                signature: "getUser(): User".to_string(),
                parent_name: None,
                text: "getUser function".to_string(),
                name_token_set: crate::tools::semantic_identifiers::identifier_terms("getUser"),
                signature_token_set: crate::tools::semantic_identifiers::identifier_terms(
                    "getUser(): User",
                ),
                parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
            },
        ];

        let query_vec = vec![1.0, 0.0];
        let vector_buffer = vec![0.5, 0.5, 0.5, 0.5];
        let query_terms = HashSet::new();
        let kinds = Some(["function"].iter().map(|s| s.to_string()).collect());

        let results = score_identifiers(
            &docs,
            &query_vec,
            &query_terms,
            &vector_buffer,
            2,
            &kinds,
            0.78,
            0.22,
            5,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc.kind, "function");
    }

    // -- rank_call_sites tests --

    #[test]
    fn test_rank_call_sites_basic() {
        let symbol = IdentifierDoc {
            id: "src/user.ts:getUserById:3".to_string(),
            path: "src/user.ts".to_string(),
            header: "user service".to_string(),
            name: "getUserById".to_string(),
            kind: "function".to_string(),
            kind_lower: "function".to_string(),
            line: 3,
            end_line: 5,
            signature: "getUserById(id: string): User".to_string(),
            parent_name: None,
            text: "getUserById function".to_string(),
            name_token_set: crate::tools::semantic_identifiers::identifier_terms("getUserById"),
            signature_token_set: crate::tools::semantic_identifiers::identifier_terms(
                "getUserById(id: string): User",
            ),
            parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
        };

        let file_content: FileContents = [
            (
                "src/user.ts".to_string(),
                Arc::new(
                    "import something from './something';\n// user service\nexport function getUserById(id: string): User {\n  return db.query(id);\n}".to_string(),
                ),
            ),
            (
                "src/handler.ts".to_string(),
                Arc::new(
                    "import { getUserById } from './user';\nconst user = getUserById(req.params.id);\nconsole.log(user);".to_string(),
                ),
            ),
        ]
        .into_iter()
        .collect();

        let query_terms: HashSet<String> = ["user", "get"].iter().map(|s| s.to_string()).collect();
        let query_vec = vec![1.0, 0.0, 0.0];

        let result = rank_call_sites(&query_terms, &query_vec, &symbol, &file_content, 10, None);

        // Should find the call in handler.ts (not the definition in user.ts)
        assert!(result.total > 0);
        // The call site at "const user = getUserById(req.params.id);" should be found
        assert!(!result.sites.is_empty());
    }

    #[test]
    fn test_rank_call_sites_skips_definition() {
        let symbol = IdentifierDoc {
            id: "src/user.ts:myFunc:1".to_string(),
            path: "src/user.ts".to_string(),
            header: "".to_string(),
            name: "myFunc".to_string(),
            kind: "function".to_string(),
            kind_lower: "function".to_string(),
            line: 1,
            end_line: 5,
            signature: "myFunc()".to_string(),
            parent_name: None,
            text: "myFunc function".to_string(),
            name_token_set: crate::tools::semantic_identifiers::identifier_terms("myFunc"),
            signature_token_set: crate::tools::semantic_identifiers::identifier_terms("myFunc()"),
            parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
        };

        let file_content: FileContents = [(
            "src/user.ts".to_string(),
            Arc::new("function myFunc() {\n  return 42;\n}".to_string()),
        )]
        .into_iter()
        .collect();

        let query_terms = HashSet::new();
        let query_vec = vec![1.0];

        let result = rank_call_sites(&query_terms, &query_vec, &symbol, &file_content, 10, None);
        assert_eq!(result.total, 0);
    }

    #[test]
    fn test_rank_call_sites_empty() {
        let symbol = IdentifierDoc {
            id: "test:noMatch:1".to_string(),
            path: "test".to_string(),
            header: "".to_string(),
            name: "noMatch".to_string(),
            kind: "function".to_string(),
            kind_lower: "function".to_string(),
            line: 1,
            end_line: 5,
            signature: "noMatch()".to_string(),
            parent_name: None,
            text: "noMatch".to_string(),
            name_token_set: crate::tools::semantic_identifiers::identifier_terms("noMatch"),
            signature_token_set: crate::tools::semantic_identifiers::identifier_terms("noMatch()"),
            parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
        };
        let file_content: FileContents = [(
            "other.ts".to_string(),
            Arc::new("const x = 42;".to_string()),
        )]
        .into_iter()
        .collect();
        let query_terms = HashSet::new();
        let query_vec = vec![1.0];

        let result = rank_call_sites(&query_terms, &query_vec, &symbol, &file_content, 10, None);
        assert_eq!(result.total, 0);
        assert!(result.sites.is_empty());
    }

    fn temp_repo_content(files: &[(&str, &str)]) -> (tempfile::TempDir, FileContents) {
        let repo = tempfile::tempdir().unwrap();
        let mut content = HashMap::new();
        for (path, source) in files {
            let full_path = repo.path().join(path);
            std::fs::create_dir_all(full_path.parent().unwrap()).unwrap();
            std::fs::write(&full_path, source).unwrap();
            content.insert((*path).to_string(), Arc::new((*source).to_string()));
        }
        (repo, content.into())
    }

    #[test]
    fn local_identifier_calls_stay_within_the_defining_file() {
        let (_repo, file_content) = temp_repo_content(&[
            (
                "src/a.ts",
                "function run() {\n  const pending = begin();\n  if (pending) consume(pending);\n}\n",
            ),
            (
                "src/b.ts",
                "// pending is discussed here but is unrelated\n",
            ),
            (
                "src/c.ts",
                "function other() {\n  const pending = otherWork();\n  return pending;\n}\n",
            ),
            ("README.md", "The pending value is documented here.\n"),
        ]);
        let symbol = IdentifierDoc {
            id: "src/a.ts:pending:2".to_string(),
            path: "src/a.ts".to_string(),
            header: String::new(),
            name: "pending".to_string(),
            kind: "const".to_string(),
            kind_lower: "const".to_string(),
            line: 2,
            end_line: 2,
            signature: "const pending = begin();".to_string(),
            parent_name: Some("run".to_string()),
            text: "pending const const pending = begin(); run".to_string(),
            name_token_set: crate::tools::semantic_identifiers::identifier_terms("pending"),
            signature_token_set: crate::tools::semantic_identifiers::identifier_terms(
                "const pending = begin();",
            ),
            parent_token_set: crate::tools::semantic_identifiers::identifier_terms(
                (Some("run".to_string())).as_deref().unwrap_or(""),
            ),
        };

        let result = rank_call_sites(
            &identifier_terms("pending"),
            &[1.0],
            &symbol,
            &file_content,
            10,
            None,
        );
        let files: HashSet<&str> = result.sites.iter().map(|site| site.file.as_str()).collect();

        assert_eq!(result.total, 1, "resolved calls were {:#?}", result.sites);
        assert_eq!(files, HashSet::from(["src/a.ts"]));
    }

    #[test]
    fn exported_identifier_calls_require_an_import_from_the_defining_module() {
        let (_repo, file_content) = temp_repo_content(&[
            (
                "src/account.ts",
                "export function loadAccount() { return {}; }\nexport function reload() { return loadAccount(); }\n",
            ),
            (
                "src/consumer.ts",
                "import { loadAccount } from './account';\nconst account = loadAccount();\n",
            ),
            (
                "src/unrelated.ts",
                "// loadAccount() is only mentioned here; it is not imported\n",
            ),
            ("README.md", "Call loadAccount() to load an account.\n"),
        ]);
        let symbol = IdentifierDoc {
            id: "src/account.ts:loadAccount:1".to_string(),
            path: "src/account.ts".to_string(),
            header: String::new(),
            name: "loadAccount".to_string(),
            kind: "function".to_string(),
            kind_lower: "function".to_string(),
            line: 1,
            end_line: 1,
            signature: "export function loadAccount()".to_string(),
            parent_name: None,
            text: "loadAccount function export function loadAccount()".to_string(),
            name_token_set: crate::tools::semantic_identifiers::identifier_terms("loadAccount"),
            signature_token_set: crate::tools::semantic_identifiers::identifier_terms(
                "export function loadAccount()",
            ),
            parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
        };

        let result = rank_call_sites(
            &identifier_terms("load account"),
            &[1.0],
            &symbol,
            &file_content,
            10,
            None,
        );
        let files: HashSet<&str> = result.sites.iter().map(|site| site.file.as_str()).collect();

        assert_eq!(result.total, 2, "resolved calls were {:#?}", result.sites);
        assert_eq!(files, HashSet::from(["src/account.ts", "src/consumer.ts"]));
    }

    // -- format output tests --

    #[test]
    fn test_format_identifier_results_empty() {
        let output = format_identifier_results("test", &[], &[]);
        assert_eq!(
            output,
            "No supported identifiers found for semantic identifier search."
        );
    }

    #[test]
    fn test_format_identifier_results() {
        let ranked = vec![RankedIdentifier {
            doc: IdentifierDoc {
                id: "test:fn:1".to_string(),
                path: "src/user.ts".to_string(),
                header: "user service".to_string(),
                name: "getUser".to_string(),
                kind: "function".to_string(),
                kind_lower: "function".to_string(),
                line: 10,
                end_line: 25,
                signature: "getUser(id: string): User".to_string(),
                parent_name: Some("UserService".to_string()),
                text: "getUser function".to_string(),
                name_token_set: crate::tools::semantic_identifiers::identifier_terms("getUser"),
                signature_token_set: crate::tools::semantic_identifiers::identifier_terms(
                    "getUser(id: string): User",
                ),
                parent_token_set: crate::tools::semantic_identifiers::identifier_terms(
                    (Some("UserService".to_string())).as_deref().unwrap_or(""),
                ),
            },
            semantic_score: 0.85,
            keyword_score: 0.65,
            score: 0.80,
        }];
        let calls = vec![CallSiteResult {
            sites: vec![CallSite {
                file: "src/handler.ts".to_string(),
                line: 42,
                context: "const user = getUser(id);".to_string(),
                semantic_score: 0.7,
                keyword_score: 0.5,
                score: 0.65,
            }],
            total: 3,
        }];

        let output = format_identifier_results("getUser", &ranked, &calls);
        assert!(output.contains("function getUser"));
        assert!(output.contains("src/user.ts"));
        assert!(output.contains("L10-L25"));
        assert!(output.contains("Parent: UserService"));
        assert!(output.contains("Calls (1/3)"));
        assert!(output.contains("src/handler.ts:L42"));
    }

    // -- identifier text quality tests --

    #[test]
    fn test_identifier_text_includes_header_and_parent() {
        // Verify the identifier text format matches TS:
        // "{name} {kind} {signature} {path} {header} {parentName}"
        let doc = IdentifierDoc {
            id: "src/user.ts:getUserById:10".to_string(),
            path: "src/user.ts".to_string(),
            header: "user service module".to_string(),
            name: "getUserById".to_string(),
            kind: "function".to_string(),
            kind_lower: "function".to_string(),
            line: 10,
            end_line: 25,
            signature: "getUserById(id: string): User".to_string(),
            parent_name: Some("UserService".to_string()),
            text: "getUserById function getUserById(id: string): User src/user.ts user service module UserService".to_string(),
            name_token_set: crate::tools::semantic_identifiers::identifier_terms("getUserById"),
            signature_token_set: crate::tools::semantic_identifiers::identifier_terms("getUserById(id: string): User"),
            parent_token_set: crate::tools::semantic_identifiers::identifier_terms((Some("UserService".to_string())).as_deref().unwrap_or("")),
        };
        assert!(doc.text.contains("getUserById"));
        assert!(doc.text.contains("function"));
        assert!(doc.text.contains("getUserById(id: string): User"));
        assert!(doc.text.contains("src/user.ts"));
        assert!(doc.text.contains("user service module"));
        assert!(doc.text.contains("UserService"));
    }

    #[test]
    fn test_identifier_text_without_parent() {
        let doc = IdentifierDoc {
            id: "src/db.ts:connect:5".to_string(),
            path: "src/db.ts".to_string(),
            header: "database module".to_string(),
            name: "connect".to_string(),
            kind: "function".to_string(),
            kind_lower: "function".to_string(),
            line: 5,
            end_line: 15,
            signature: "connect(): Connection".to_string(),
            parent_name: None,
            text: "connect function connect(): Connection src/db.ts database module ".to_string(),
            name_token_set: crate::tools::semantic_identifiers::identifier_terms("connect"),
            signature_token_set: crate::tools::semantic_identifiers::identifier_terms(
                "connect(): Connection",
            ),
            parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
        };
        assert!(doc.text.contains("connect"));
        assert!(doc.text.contains("database module"));
        assert!(doc.text.contains("src/db.ts"));
    }

    // -- normalize_kinds tests --

    #[test]
    fn test_normalize_kinds_none() {
        assert!(normalize_kinds(&None).is_none());
    }

    #[test]
    fn test_normalize_kinds_empty() {
        assert!(normalize_kinds(&Some(vec![])).is_none());
    }

    #[test]
    fn test_normalize_kinds_normalizes() {
        let kinds = Some(vec!["Function".to_string(), " CLASS ".to_string()]);
        let result = normalize_kinds(&kinds).unwrap();
        assert!(result.contains("function"));
        assert!(result.contains("class"));
    }

    // -- performance regression test --

    /// Synthetic corpus: 500 identifier docs + 200 files × 100 lines.
    /// The target symbol name appears on ~10% of lines to exercise the hot path.
    /// Asserts that `rank_call_sites` for 5 identifiers completes in under 2 s,
    /// which guards against O(N × top_k) regressions on the call-site scan loop.
    #[test]
    fn test_rank_call_sites_perf_large_corpus() {
        use std::time::Instant;

        // Build a fake file corpus: 200 files, 100 lines each.
        // Every 10th line calls "syncStripeQuantity(" so there are ~2000 matches.
        let num_files = 200usize;
        let lines_per_file = 100usize;
        let mut file_content: HashMap<String, Arc<String>> = HashMap::new();
        file_content.insert(
            "src/mod_0.ts".to_string(),
            Arc::new(
                "export function syncStripeQuantity(orgId: string, seats: number): void {}"
                    .to_string(),
            ),
        );
        for fi in 0..num_files {
            let mut lines = Vec::with_capacity(lines_per_file);
            lines.push("import { syncStripeQuantity } from './mod_0';".to_string());
            for li in 1..lines_per_file {
                if (li - 1) % 10 == 0 {
                    lines.push(format!(
                        "  const result = syncStripeQuantity(orgId_{fi}_{li}, seats);"
                    ));
                } else {
                    lines.push(format!(
                        "  const x_{fi}_{li} = doSomethingElse(param_{li});"
                    ));
                }
            }
            file_content.insert(format!("src/module_{fi}.ts"), Arc::new(lines.join("\n")));
        }
        let file_contents: FileContents = file_content.into();

        // Build 500 fake identifier docs (only a handful will be top-ranked,
        // but we need enough to simulate a real corpus size for score_identifiers).
        let num_docs = 500usize;
        let dims = 4usize;
        let mut docs: Vec<IdentifierDoc> = Vec::with_capacity(num_docs);
        let mut vector_buffer: Vec<f32> = Vec::with_capacity(num_docs * dims);
        for i in 0..num_docs {
            let name = if i == 0 {
                "syncStripeQuantity".to_string()
            } else {
                format!("identifier_{i}")
            };
            let doc = IdentifierDoc {
                id: format!("src/mod_{i}.ts:{name}:{}", i * 3),
                path: format!("src/mod_{i}.ts"),
                header: "stripe billing".to_string(),
                name: name.clone(),
                kind: "function".to_string(),
                kind_lower: "function".to_string(),
                line: i * 3 + 1,
                end_line: i * 3 + 5,
                signature: format!("{name}(orgId: string, seats: number): void"),
                parent_name: None,
                text: format!(
                    "{name} function {name}(orgId: string, seats: number): void \
                     src/mod_{i}.ts stripe billing"
                ),
                name_token_set: crate::tools::semantic_identifiers::identifier_terms(&name),
                signature_token_set: crate::tools::semantic_identifiers::identifier_terms(
                    &(format!("{name}(orgId: string, seats: number): void")),
                ),
                parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
            };
            docs.push(doc);
            // Give syncStripeQuantity (index 0) a high-similarity vector; rest low.
            if i == 0 {
                vector_buffer.extend_from_slice(&[0.9, 0.1, 0.0, 0.0]);
            } else {
                // Spread other docs so they score lower
                let v = (i as f32 % 10.0) / 10.0;
                vector_buffer.extend_from_slice(&[0.1, v, 0.0, 0.0]);
            }
        }

        let query_vec: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let query_terms: HashSet<String> = ["sync", "stripe", "quantity"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Score identifiers to get top 5 (simulates the full search path).
        let top = score_identifiers(
            &docs,
            &query_vec,
            &query_terms,
            &vector_buffer,
            dims,
            &None,
            DEFAULT_SEMANTIC_WEIGHT,
            DEFAULT_KEYWORD_WEIGHT,
            5,
        );
        assert!(!top.is_empty(), "expected at least one top identifier");

        // Time the call-site ranking for all top identifiers.
        // Before the fix this was sequential O(top_k × corpus_lines); after the
        // fix it runs in parallel and with per-file substring pre-filtering.
        let start = Instant::now();
        let call_results: Vec<CallSiteResult> = top
            .par_iter()
            .map(|item| {
                rank_call_sites(
                    &query_terms,
                    &query_vec,
                    &item.doc,
                    &file_contents,
                    10,
                    None,
                )
            })
            .collect();
        let elapsed = start.elapsed();

        // Sanity-check: syncStripeQuantity should find call sites.
        let stripe_result = call_results
            .iter()
            .find(|_| true) // first result corresponds to best-ranked identifier
            .expect("should have at least one call-site result");
        // 200 files × 10 matching lines per file = 2000 total; capped by candidate_cap.
        assert!(
            stripe_result.total > 0 || top[0].doc.name != "syncStripeQuantity",
            "syncStripeQuantity should have call sites"
        );

        // 2s was tight enough to flake on a loaded shared macOS runner (observed
        // 2.06s, then 3.31s on a retry, while this stayed green on every recent
        // main run). 10s still catches a real O(n) → O(n²) regression by a wide
        // margin — the fast path finishes in well under 1s locally — while
        // giving CI-load noise room to breathe.
        assert!(
            elapsed.as_secs_f64() < 10.0,
            "rank_call_sites for 5 identifiers over 200-file corpus took {:.2}s (limit: 10s)",
            elapsed.as_secs_f64()
        );
    }

    // -- precomputed token-set tests --

    #[test]
    fn warm_scoring_uses_separate_precomputed_evidence_without_live_tokenization() {
        let mut docs = Vec::with_capacity(2_000);
        let mut vectors = Vec::with_capacity(2_000);
        for i in 0..2_000 {
            let (name, signature, parent) = if i == 1_337 {
                (
                    "cascadeGrants".to_string(),
                    "const cascadeGrants = resolve();".to_string(),
                    Some("GrantCoordinator"),
                )
            } else {
                (
                    format!("unrelatedIdentifier{i}"),
                    format!("const unrelatedIdentifier{i} = resolve();"),
                    None,
                )
            };
            docs.push(keyword_test_doc(
                &name,
                &signature,
                parent,
                "",
                &signature,
                i + 1,
            ));
            vectors.push(1.0);
        }

        // This exercises only the already-built index scoring stage: there is
        // no parser, import resolver, embedder, or startup path in this loop.
        for _ in 0..3 {
            let results = score_identifiers(
                &docs,
                &[1.0],
                &identifier_terms("cascade grants"),
                &vectors,
                1,
                &None,
                0.0,
                1.0,
                1,
            );
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc.name, "cascadeGrants");
            assert_eq!(results[0].keyword_score, 1.0);
        }

        let source = include_str!("semantic_identifiers.rs");
        let doc_definition = source
            .split("pub struct IdentifierDoc")
            .nth(1)
            .and_then(|tail| tail.split("impl IdentifierDoc").next())
            .expect("IdentifierDoc source");
        for field in ["name_token_set", "signature_token_set", "parent_token_set"] {
            assert!(
                doc_definition.contains(field),
                "IdentifierDoc must precompute separate {field} evidence"
            );
        }
        assert!(
            !doc_definition.contains("pub token_set"),
            "IdentifierDoc must not retain or clone the obsolete combined token_set"
        );

        let scoring = source
            .split("pub fn score_identifiers")
            .nth(1)
            .and_then(|tail| tail.split("pub fn format_identifier_results").next())
            .expect("score_identifiers source");
        assert!(
            !scoring.contains("get_keyword_coverage")
                && !scoring.contains("identifier_terms")
                && !scoring.contains("split_camel_case"),
            "warm score_identifiers must consume precomputed evidence without live tokenization"
        );
    }

    #[test]
    fn identifier_doc_does_not_retain_legacy_combined_token_set() {
        let source = include_str!("semantic_identifiers.rs");
        let doc_definition = source
            .split("pub struct IdentifierDoc")
            .nth(1)
            .and_then(|tail| tail.split("impl IdentifierDoc").next())
            .expect("IdentifierDoc source");

        assert!(
            !doc_definition.contains("pub token_set"),
            "the legacy combined token_set wastes memory and permits hot-loop scoring regressions"
        );
    }

    /// Regression: `score_identifiers` must return the same ranked order when
    /// using the pre-computed `token_set` as it did with inline tokenization.
    #[test]
    fn score_identifiers_results_unchanged() {
        // Two docs: one is a strong keyword + semantic match; the other is weak.
        let make_doc =
            |name: &str, sig: &str, path: &str, header: &str, kind: &str| IdentifierDoc {
                id: format!("{path}:{name}:1"),
                path: path.to_string(),
                header: header.to_string(),
                name: name.to_string(),
                kind: kind.to_string(),
                kind_lower: kind.to_lowercase(),
                line: 1,
                end_line: 5,
                signature: sig.to_string(),
                parent_name: None,
                text: format!("{name} {kind} {sig} {path} {header}"),
                name_token_set: crate::tools::semantic_identifiers::identifier_terms(name),
                signature_token_set: crate::tools::semantic_identifiers::identifier_terms(sig),
                parent_token_set: crate::tools::semantic_identifiers::identifier_terms(""),
            };

        let docs = vec![
            make_doc(
                "getUserById",
                "getUserById(id: string): User",
                "src/user.ts",
                "user service",
                "function",
            ),
            make_doc(
                "connectDatabase",
                "connectDatabase(): Connection",
                "src/db.ts",
                "database layer",
                "function",
            ),
        ];

        // Vector favours getUserById (index 0).
        let query_vec = vec![1.0f32, 0.0, 0.0];
        let vector_buffer = vec![
            0.9f32, 0.1, 0.0, // getUserById
            0.1, 0.9, 0.0, // connectDatabase
        ];
        let query_terms: HashSet<String> = ["user", "get"].iter().map(|s| s.to_string()).collect();

        let results = score_identifiers(
            &docs,
            &query_vec,
            &query_terms,
            &vector_buffer,
            3,
            &None,
            0.78,
            0.22,
            5,
        );

        assert_eq!(results.len(), 2);
        // getUserById must rank first (higher semantic + keyword coverage).
        assert_eq!(
            results[0].doc.name, "getUserById",
            "ranking changed after precompute refactor"
        );
        assert_eq!(results[1].doc.name, "connectDatabase");
    }
}
