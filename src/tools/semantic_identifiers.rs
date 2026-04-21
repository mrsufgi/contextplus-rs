//! Identifier-level semantic search with call-site ranking and line metadata.
//!
//! Ports the TypeScript `semantic-identifiers.ts` logic:
//! - Indexes all code symbols (functions, classes, types, etc.) with embeddings
//! - Ranks identifiers by hybrid semantic + keyword score
//! - Finds and ranks call-sites for each top identifier

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use rayon::prelude::*;
use regex::Regex;

use crate::error::{ContextPlusError, Result};
use crate::tools::scoring::{DEFAULT_TOP_K, clamp01, keyword_coverage, normalize_weight};
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
    /// Pre-computed token set over `name + signature + path + header`.
    ///
    /// Built once at index time via `split_camel_case`; eliminates one
    /// `format!` allocation and one re-tokenization per doc per query in
    /// the hot `score_identifiers` loop.
    pub token_set: HashSet<String>,
}

impl IdentifierDoc {
    /// Build the pre-computed `token_set` from the four fields used in keyword
    /// scoring.  Call this once at index time; never call inside a query loop.
    pub fn build_token_set(
        name: &str,
        signature: &str,
        path: &str,
        header: &str,
    ) -> HashSet<String> {
        let combined = format!("{name} {signature} {path} {header}");
        split_camel_case(&combined).into_iter().collect()
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

/// Tokenize `input` with `split_camel_case` then delegate to
/// `scoring::keyword_coverage`.  Identifier search always needs to tokenize
/// the document string on the fly (unlike file-level search, which
/// pre-computes token sets at index time).
fn get_keyword_coverage(query_terms: &HashSet<String>, input: &str) -> f64 {
    let doc_tokens: HashSet<String> = split_camel_case(input).into_iter().collect();
    keyword_coverage(query_terms, &doc_tokens)
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
    file_content: &HashMap<String, Arc<String>>,
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

    // Cap: only gather enough candidates to fill the embed budget.
    // Using a hard ceiling avoids unbounded Vec growth when a common symbol
    // name matches thousands of lines — we only need the top `embed_budget`
    // by keyword score, so stop collecting once we have far more than that.
    let embed_budget = (limit * 4).max(30);
    // Collect 4× the embed_budget before applying the top-k filter, giving a
    // good statistical sample without scanning the entire tail of matches.
    let candidate_cap = embed_budget * 4;

    // Lazy per-file line-split: only files that pass the cheap substring
    // pre-filter on the full content get split. This avoids splitting every
    // file in the corpus just to skip files that don't mention the symbol.
    // (Review #59 F1.)
    let mut file_entries: Vec<(&String, Vec<&str>)> = Vec::new();

    // Collect candidates as (file_index, line_number, line_offset, keyword_score)
    // to avoid cloning file Strings and context Strings in the inner loop.
    let mut candidates: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut keyword_buf = String::with_capacity(512);

    'outer: for (file, content) in file_content.iter() {
        // Fast pre-filter on the full file content (no allocation, no split):
        // skip files that don't contain the symbol name at all.
        if !content.contains(symbol.name.as_str()) {
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
            if !call_pattern.is_match(line) {
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
            let context = if context.len() > 220 {
                &context[..220]
            } else {
                context
            };
            // Reuse buffer instead of format! allocation per iteration
            keyword_buf.clear();
            keyword_buf.push_str(file);
            keyword_buf.push(' ');
            keyword_buf.push_str(context);
            let keyword_score = get_keyword_coverage(query_terms, &keyword_buf);
            candidates.push((fi, i + 1, i, keyword_score));

            // Hard cap: once we have enough candidates to fill a good sample
            // for the embed budget, stop collecting. This prevents O(N) growth
            // for common symbol names with thousands of matches.
            if candidates.len() >= candidate_cap {
                break 'outer;
            }
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
            let context = if context.len() > 220 {
                &context[..220]
            } else {
                context
            };

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
    // Phase 1: Score all docs, collecting only indices + scores (no clone).
    let mut scored: Vec<(usize, f64, f64, f64)> = docs
        .par_iter()
        .enumerate()
        .filter_map(|(i, doc)| {
            if let Some(kinds) = include_kinds
                && !kinds.contains(&doc.kind_lower)
            {
                return None;
            }

            let offset = i * vector_dims;
            if offset + vector_dims > vector_buffer.len() {
                return None;
            }
            let vec_slice = &vector_buffer[offset..offset + vector_dims];
            let semantic_score =
                crate::core::embeddings::cosine_similarity_simsimd(query_vec, vec_slice).max(0.0)
                    as f64;

            let keyword_score = keyword_coverage(query_terms, &doc.token_set);

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
                doc: docs[idx].clone(),
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
pub async fn semantic_identifier_search(
    options: SemanticIdentifierSearchOptions,
    embed_fn: &dyn crate::tools::semantic_search::EmbedFn,
    identifier_docs: &[IdentifierDoc],
    vector_buffer: &[f32],
    vector_dims: usize,
    file_content: &HashMap<String, Arc<String>>,
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
    let query_terms: HashSet<String> = split_camel_case(query.as_ref()).into_iter().collect();

    // Score identifiers
    let top = score_identifiers(
        identifier_docs,
        &query_vec,
        &query_terms,
        vector_buffer,
        vector_dims,
        &include_kinds,
        semantic_weight,
        keyword_weight,
        top_k,
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
                token_set: HashSet::new(),
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
                token_set: HashSet::new(),
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
                token_set: HashSet::new(),
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
                token_set: HashSet::new(),
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
            text: "getUserById function".to_string(),
            token_set: HashSet::new(),
        };

        let file_content: HashMap<String, Arc<String>> = [
            (
                "src/user.ts".to_string(),
                Arc::new(
                    "import something\n// user service\nexport function getUserById(id: string): User {\n  return db.query(id);\n}".to_string(),
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
            token_set: HashSet::new(),
        };

        let file_content: HashMap<String, Arc<String>> = [(
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
            token_set: HashSet::new(),
        };
        let file_content: HashMap<String, Arc<String>> = [(
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
                token_set: HashSet::new(),
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
            token_set: HashSet::new(),
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
            token_set: HashSet::new(),
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
        for fi in 0..num_files {
            let mut lines = Vec::with_capacity(lines_per_file);
            for li in 0..lines_per_file {
                if li % 10 == 0 {
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
                token_set: IdentifierDoc::build_token_set(
                    &name,
                    &format!("{name}(orgId: string, seats: number): void"),
                    &format!("src/mod_{i}.ts"),
                    "stripe billing",
                ),
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
                rank_call_sites(&query_terms, &query_vec, &item.doc, &file_content, 10, None)
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

        assert!(
            elapsed.as_secs_f64() < 2.0,
            "rank_call_sites for 5 identifiers over 200-file corpus took {:.2}s (limit: 2s)",
            elapsed.as_secs_f64()
        );
    }

    // -- precomputed token-set tests --

    /// The `token_set` stored in an `IdentifierDoc` must be identical to what
    /// `split_camel_case(format!(...))` would have produced at query time.
    /// This guards against drift between the build-time helper and the old
    /// inline tokenization path.
    #[test]
    fn precomputed_token_set_matches_live_tokenization() {
        let name = "getUserById";
        let signature = "getUserById(id: string): User";
        let path = "src/user.ts";
        let header = "user service module";

        // Pre-computed path (new).
        let precomputed = IdentifierDoc::build_token_set(name, signature, path, header);

        // Live path (old — what score_identifiers used to do on every query).
        let keyword_input = format!("{name} {signature} {path} {header}");
        let live: HashSet<String> = split_camel_case(&keyword_input).into_iter().collect();

        assert_eq!(
            precomputed, live,
            "precomputed token_set diverges from live tokenization"
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
                token_set: IdentifierDoc::build_token_set(name, sig, path, header),
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
