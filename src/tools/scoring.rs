//! Centralized hybrid-scoring constants and helpers shared by all semantic
//! search tools (`semantic_search`, `semantic_identifiers`).
//!
//! ## Weight rationale
//!
//! `DEFAULT_SEMANTIC_WEIGHT = 0.72 / DEFAULT_KEYWORD_WEIGHT = 0.28` are taken
//! from the original TypeScript `semantic-search.ts` port — the first and
//! primary calibration point — and are the values advertised in the MCP tool
//! schemas.  `semantic_identifiers` was tuned later for code-symbol search and
//! overrides these defaults locally (see `IDENTIFIER_SEMANTIC_WEIGHT`).

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Canonical defaults
// ---------------------------------------------------------------------------

/// Semantic component weight for the primary file-level hybrid score.
/// Matches the TypeScript `semantic-search.ts` baseline (0.72) and the values
/// documented in the MCP tool schema.
pub const DEFAULT_SEMANTIC_WEIGHT: f64 = 0.72;

/// Keyword component weight complementing `DEFAULT_SEMANTIC_WEIGHT`.
pub const DEFAULT_KEYWORD_WEIGHT: f64 = 0.28;

/// Default number of results to return when the caller does not specify `top_k`.
pub const DEFAULT_TOP_K: usize = 5;

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// Clamp `value` to `[0.0, 1.0]`.
#[inline]
pub fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

/// Return `value` if it is finite and non-negative, otherwise return
/// `fallback`.  Used to sanitize user-supplied weight overrides.
#[inline]
pub fn normalize_weight(value: Option<f64>, fallback: f64) -> f64 {
    match value {
        Some(v) if v.is_finite() && v >= 0.0 => v,
        _ => fallback,
    }
}

/// Fraction of `query_terms` that are present in `doc_tokens`.
///
/// Returns `0.0` when `query_terms` is empty to avoid division by zero.
/// Both sets must already be lowercased / split — callers are responsible for
/// tokenization (e.g. via `split_camel_case`).
#[inline]
pub fn keyword_coverage(query_terms: &HashSet<String>, doc_tokens: &HashSet<String>) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let matched = query_terms
        .iter()
        .filter(|t| doc_tokens.contains(*t))
        .count();
    matched as f64 / query_terms.len() as f64
}

/// Truncate `s` to at most `max_bytes` bytes **without splitting a UTF-8
/// character**.
///
/// Plain `&s[..max_bytes]` panics when `max_bytes` lands inside a multi-byte
/// char (e.g. the box-drawing `═`, U+2550, is 3 bytes) — this is exactly the
/// crash that took down the semantic-search daemon on markdown banner lines
/// like `# ═══…`.  This helper floors the cut to the nearest preceding char
/// boundary instead, and returns `s` unchanged when it already fits.
#[inline]
pub fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp01_bounds() {
        assert_eq!(clamp01(-0.5), 0.0);
        assert_eq!(clamp01(0.5), 0.5);
        assert_eq!(clamp01(1.5), 1.0);
    }

    #[test]
    fn normalize_weight_valid() {
        assert!((normalize_weight(Some(0.6), 0.72) - 0.6).abs() < 1e-10);
    }

    #[test]
    fn normalize_weight_fallback_on_none() {
        assert!((normalize_weight(None, 0.72) - 0.72).abs() < 1e-10);
    }

    #[test]
    fn normalize_weight_fallback_on_nan() {
        assert!((normalize_weight(Some(f64::NAN), 0.72) - 0.72).abs() < 1e-10);
    }

    #[test]
    fn normalize_weight_fallback_on_negative() {
        assert!((normalize_weight(Some(-1.0), 0.72) - 0.72).abs() < 1e-10);
    }

    #[test]
    fn keyword_coverage_full_match() {
        let query: HashSet<String> = ["user", "get"].iter().map(|s| s.to_string()).collect();
        let doc: HashSet<String> = ["get", "user", "by", "id"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!((keyword_coverage(&query, &doc) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn keyword_coverage_partial_match() {
        let query: HashSet<String> = ["user", "delete", "by"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let doc: HashSet<String> = ["get", "user", "by", "id"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!((keyword_coverage(&query, &doc) - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn keyword_coverage_empty_query() {
        let query: HashSet<String> = HashSet::new();
        let doc: HashSet<String> = ["get"].iter().map(|s| s.to_string()).collect();
        assert_eq!(keyword_coverage(&query, &doc), 0.0);
    }

    #[test]
    fn truncate_on_char_boundary_caps_ascii_exactly() {
        assert_eq!(truncate_on_char_boundary("abcdef", 3), "abc");
    }

    #[test]
    fn truncate_on_char_boundary_short_string_unchanged() {
        assert_eq!(truncate_on_char_boundary("abc", 10), "abc");
    }

    #[test]
    fn truncate_on_char_boundary_does_not_split_multibyte() {
        // Regression: box-drawing '═' (U+2550) is 3 bytes; a banner line that
        // exceeds the cap must not panic and must not split the char. This is
        // the exact input that crashed the daemon on '# ═══…' markdown banners.
        let banner = format!("# {}", "═".repeat(60)); // 2 + 60*3 = 182 bytes
        let out = truncate_on_char_boundary(&banner, 120);
        assert!(out.len() <= 120);
        assert!(banner.starts_with(out));
        assert!(out.chars().all(|c| c == '#' || c == ' ' || c == '═'));
    }
}
