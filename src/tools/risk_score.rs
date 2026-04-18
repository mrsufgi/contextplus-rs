//! Composite risk score for blast-radius ordering.
//!
//! Given a set of changed files (with line counts and changed symbols) and
//! the dependent files reached by reverse-import expansion, produce a
//! ranked list of (path, score, reasons) tuples so callers can surface
//! the riskiest files first in PR review or impact tooling.
//!
//! The score is a small linear combination of three signals:
//!   • changed lines  — direct churn
//!   • changed symbols — semantic surface
//!   • dependent count — downstream blast radius
//!
//! Weights are intentionally modest constants; nothing here calls out to
//! a model. Tune by editing the constants below if a calibration dataset
//! shows the ordering is wrong.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const WEIGHT_CHANGED_LINES: f64 = 1.0;
pub const WEIGHT_CHANGED_SYMBOLS: f64 = 5.0;
pub const WEIGHT_DEPENDENTS: f64 = 3.0;

/// Per-file inputs to the scorer. All counts are u32 — large diffs get
/// clamped at u32::MAX which is safely above any realistic input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRiskInput {
    pub path: PathBuf,
    pub changed_lines: u32,
    pub changed_symbol_count: u32,
    pub dependent_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FileRisk {
    pub path: PathBuf,
    pub score: f64,
    /// Short human-readable bullets explaining the score's components.
    pub reasons: Vec<String>,
}

/// Compute risk for a single file.
pub fn score_file(input: &FileRiskInput) -> FileRisk {
    let lines_part = WEIGHT_CHANGED_LINES * input.changed_lines as f64;
    let symbol_part = WEIGHT_CHANGED_SYMBOLS * input.changed_symbol_count as f64;
    let dep_part = WEIGHT_DEPENDENTS * input.dependent_count as f64;
    let score = lines_part + symbol_part + dep_part;

    let mut reasons = Vec::new();
    if input.changed_lines > 0 {
        reasons.push(format!("{} changed lines", input.changed_lines));
    }
    if input.changed_symbol_count > 0 {
        reasons.push(format!(
            "{} changed symbol(s)",
            input.changed_symbol_count
        ));
    }
    if input.dependent_count > 0 {
        reasons.push(format!(
            "{} dependent file(s) within 2 hops",
            input.dependent_count
        ));
    }

    FileRisk {
        path: input.path.clone(),
        score,
        reasons,
    }
}

/// Score every input and return them sorted highest-risk first. Stable
/// secondary sort on path so the result is deterministic when scores tie.
pub fn rank(inputs: &[FileRiskInput]) -> Vec<FileRisk> {
    let mut out: Vec<FileRisk> = inputs.iter().map(score_file).collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    out
}

/// Convenience: build risk inputs from raw count maps. Files that appear
/// in any of the three maps are scored; missing entries default to zero.
pub fn rank_from_maps(
    changed_lines: &HashMap<PathBuf, u32>,
    changed_symbols: &HashMap<PathBuf, u32>,
    dependents: &HashMap<PathBuf, u32>,
) -> Vec<FileRisk> {
    let mut all: std::collections::BTreeSet<&Path> = std::collections::BTreeSet::new();
    all.extend(changed_lines.keys().map(|p| p.as_path()));
    all.extend(changed_symbols.keys().map(|p| p.as_path()));
    all.extend(dependents.keys().map(|p| p.as_path()));

    let inputs: Vec<FileRiskInput> = all
        .into_iter()
        .map(|p| FileRiskInput {
            path: p.to_path_buf(),
            changed_lines: changed_lines.get(p).copied().unwrap_or(0),
            changed_symbol_count: changed_symbols.get(p).copied().unwrap_or(0),
            dependent_count: dependents.get(p).copied().unwrap_or(0),
        })
        .collect();

    rank(&inputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(path: &str, lines: u32, symbols: u32, deps: u32) -> FileRiskInput {
        FileRiskInput {
            path: PathBuf::from(path),
            changed_lines: lines,
            changed_symbol_count: symbols,
            dependent_count: deps,
        }
    }

    #[test]
    fn score_file_combines_all_three_components() {
        let r = score_file(&input("a", 10, 2, 3));
        let expected = WEIGHT_CHANGED_LINES * 10.0
            + WEIGHT_CHANGED_SYMBOLS * 2.0
            + WEIGHT_DEPENDENTS * 3.0;
        assert!((r.score - expected).abs() < 1e-9);
        assert_eq!(r.reasons.len(), 3);
    }

    #[test]
    fn score_file_drops_zero_components_from_reasons() {
        let r = score_file(&input("a", 5, 0, 0));
        assert_eq!(r.reasons.len(), 1);
        assert!(r.reasons[0].contains("changed lines"));
    }

    #[test]
    fn rank_sorts_high_score_first() {
        let inputs = vec![
            input("low.rs", 1, 0, 0),
            input("high.rs", 100, 5, 10),
            input("mid.rs", 20, 1, 2),
        ];
        let ranked = rank(&inputs);
        assert_eq!(ranked[0].path, PathBuf::from("high.rs"));
        assert_eq!(ranked[1].path, PathBuf::from("mid.rs"));
        assert_eq!(ranked[2].path, PathBuf::from("low.rs"));
    }

    #[test]
    fn rank_breaks_ties_by_path_alphabetically() {
        let inputs = vec![input("zzz.rs", 5, 0, 0), input("aaa.rs", 5, 0, 0)];
        let ranked = rank(&inputs);
        assert_eq!(ranked[0].path, PathBuf::from("aaa.rs"));
        assert_eq!(ranked[1].path, PathBuf::from("zzz.rs"));
    }

    #[test]
    fn rank_empty_input_returns_empty() {
        assert!(rank(&[]).is_empty());
    }

    #[test]
    fn rank_from_maps_unions_keys() {
        let mut lines = HashMap::new();
        lines.insert(PathBuf::from("a.rs"), 10);
        let mut symbols = HashMap::new();
        symbols.insert(PathBuf::from("b.rs"), 1);
        let mut deps = HashMap::new();
        deps.insert(PathBuf::from("c.rs"), 5);

        let ranked = rank_from_maps(&lines, &symbols, &deps);
        let paths: Vec<&PathBuf> = ranked.iter().map(|r| &r.path).collect();
        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&&PathBuf::from("a.rs")));
        assert!(paths.contains(&&PathBuf::from("b.rs")));
        assert!(paths.contains(&&PathBuf::from("c.rs")));
    }

    #[test]
    fn rank_from_maps_default_missing_to_zero() {
        let mut lines = HashMap::new();
        lines.insert(PathBuf::from("a.rs"), 5);
        let symbols = HashMap::new();
        let deps = HashMap::new();
        let ranked = rank_from_maps(&lines, &symbols, &deps);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].score, WEIGHT_CHANGED_LINES * 5.0);
    }

    #[test]
    fn dependent_count_outweighs_pure_line_churn_at_scale() {
        // 5 deps × WEIGHT_DEPENDENTS (3.0) = 15  vs  10 lines × 1.0 = 10
        let dep_heavy = score_file(&input("dep.rs", 0, 0, 5));
        let line_heavy = score_file(&input("line.rs", 10, 0, 0));
        assert!(dep_heavy.score > line_heavy.score);
    }
}
