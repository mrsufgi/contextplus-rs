//! `pr_review` — analyse a unified diff and produce a risk-ranked
//! impact report.
//!
//! Composition of three primitives:
//!
//! 1. [`crate::git::diff::parse_unified_diff`] — diff text → file/line ranges.
//! 2. [`crate::git::diff::changed_symbols`] — line ranges → overlapping symbols.
//! 3. [`crate::core::dependent_expand::expand_dependents`] — 2-hop reverse-import
//!    BFS from changed files.
//!
//! Then [`crate::tools::risk_score::rank`] orders the union of (changed,
//! dependent) files by composite risk score.
//!
//! The function is pure: callers supply the parsed symbols-by-file and
//! reverse import graph. The MCP wiring is responsible for building those
//! from disk.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::core::dependent_expand::{ExpansionHit, ExpansionOptions, expand_dependents};
use crate::core::parser::CodeSymbol;
use crate::git::diff::{FileChange, changed_symbols, parse_unified_diff};
use crate::tools::risk_score::{FileRisk, FileRiskInput, rank};

/// One file in the PR review report.
#[derive(Debug, Clone)]
pub struct PrFileEntry {
    pub path: PathBuf,
    /// `true` when the file appears in the diff itself (not just touched by
    /// dependent expansion).
    pub is_changed: bool,
    /// Hop distance from the nearest changed file (0 for changed files).
    pub hop: usize,
    pub changed_lines: u32,
    pub changed_symbols: Vec<String>,
    pub risk: FileRisk,
}

/// Full review report for one diff.
#[derive(Debug, Clone)]
pub struct PrReviewReport {
    pub entries: Vec<PrFileEntry>,
    pub total_changed_files: usize,
    pub total_dependent_files: usize,
}

/// Top-level entry point. Pure function — see module docs.
pub fn analyze(
    diff: &str,
    symbols_by_file: &HashMap<String, Vec<CodeSymbol>>,
    reverse_graph: &HashMap<PathBuf, HashSet<PathBuf>>,
    expansion_opts: ExpansionOptions,
) -> PrReviewReport {
    let changes = parse_unified_diff(diff);
    let changed_paths: Vec<PathBuf> = changes.iter().map(|c| PathBuf::from(&c.path)).collect();
    let dependents: Vec<ExpansionHit> =
        expand_dependents(reverse_graph, &changed_paths, expansion_opts);

    let lines_per_file = lines_per_file(&changes);
    let symbols_per_file_set: HashMap<String, Vec<&CodeSymbol>> =
        changed_symbols_grouped(&changes, symbols_by_file);
    let changed_symbol_count_map: HashMap<PathBuf, u32> = symbols_per_file_set
        .iter()
        .map(|(p, syms)| {
            (
                PathBuf::from(p),
                u32::try_from(syms.len()).unwrap_or(u32::MAX),
            )
        })
        .collect();

    let dependent_count_map = dependent_count_per_seed(&changed_paths, &dependents);

    // Build risk inputs for every file we want to surface (changed ∪ dependents).
    let mut all_paths: Vec<PathBuf> = dependents.iter().map(|h| h.path.clone()).collect();
    for c in &changed_paths {
        if !all_paths.iter().any(|p| p == c) {
            all_paths.push(c.clone());
        }
    }

    let inputs: Vec<FileRiskInput> = all_paths
        .iter()
        .map(|path| {
            let changed_lines = lines_per_file.get(path).copied().unwrap_or(0);
            let changed_symbol_count = changed_symbol_count_map.get(path).copied().unwrap_or(0);
            // For changed files, use the count of *their* dependent files.
            // For dependent files, use 0 (we don't recurse another level).
            let dependent_count = dependent_count_map.get(path).copied().unwrap_or(0);
            FileRiskInput {
                path: path.clone(),
                changed_lines,
                changed_symbol_count,
                dependent_count,
            }
        })
        .collect();

    let risk_ranked = rank(&inputs);

    // Index by path so we can splice in symbol names + hop info.
    let hop_by_path: HashMap<&Path, usize> = dependents
        .iter()
        .map(|h| (h.path.as_path(), h.hop))
        .collect();
    let changed_set: HashSet<&Path> = changed_paths.iter().map(|p| p.as_path()).collect();

    let entries: Vec<PrFileEntry> = risk_ranked
        .into_iter()
        .map(|fr| {
            let path = fr.path.clone();
            let is_changed = changed_set.contains(path.as_path());
            let hop = hop_by_path.get(path.as_path()).copied().unwrap_or(0);
            let changed_lines = lines_per_file.get(&path).copied().unwrap_or(0);
            let changed_symbols = symbols_per_file_set
                .get(path.to_string_lossy().as_ref())
                .map(|v| v.iter().map(|s| s.name.clone()).collect())
                .unwrap_or_default();
            PrFileEntry {
                path,
                is_changed,
                hop,
                changed_lines,
                changed_symbols,
                risk: fr,
            }
        })
        .collect();

    let total_changed_files = changed_paths.len();
    let total_dependent_files = dependents.iter().filter(|h| h.hop > 0).count();

    PrReviewReport {
        entries,
        total_changed_files,
        total_dependent_files,
    }
}

/// Format a [`PrReviewReport`] as a short text summary for tool output.
pub fn format_report(report: &PrReviewReport) -> String {
    if report.entries.is_empty() {
        return "PR review: empty diff (no files changed).".to_string();
    }
    let mut lines = vec![format!(
        "PR review: {} changed file(s), {} dependent file(s) within 2 hops",
        report.total_changed_files, report.total_dependent_files
    )];
    lines.push(String::new());
    for (i, e) in report.entries.iter().enumerate() {
        let tag = if e.is_changed {
            "changed".to_string()
        } else {
            format!("dependent (hop {})", e.hop)
        };
        lines.push(format!(
            "{}. {} [{}] — risk {:.1}",
            i + 1,
            e.path.display(),
            tag,
            e.risk.score
        ));
        if !e.risk.reasons.is_empty() {
            lines.push(format!("   {}", e.risk.reasons.join("; ")));
        }
        if !e.changed_symbols.is_empty() {
            lines.push(format!(
                "   Touched symbols: {}",
                e.changed_symbols.join(", ")
            ));
        }
    }
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn lines_per_file(changes: &[FileChange]) -> HashMap<PathBuf, u32> {
    let mut out = HashMap::new();
    for c in changes {
        let total: u32 = c
            .ranges
            .iter()
            .map(|r| {
                let len = r.end.saturating_sub(r.start).saturating_add(1);
                u32::try_from(len).unwrap_or(u32::MAX)
            })
            .sum();
        out.insert(PathBuf::from(&c.path), total);
    }
    out
}

fn changed_symbols_grouped<'a>(
    changes: &[FileChange],
    symbols_by_file: &'a HashMap<String, Vec<CodeSymbol>>,
) -> HashMap<String, Vec<&'a CodeSymbol>> {
    let mut out: HashMap<String, Vec<&CodeSymbol>> = HashMap::new();
    let touched = changed_symbols(changes, symbols_by_file);
    for cs in touched {
        out.entry(cs.path).or_default().push(cs.symbol);
    }
    out
}

/// For each changed seed file, count how many strict-dependent (hop > 0)
/// hits trace back through the reverse graph reachable from it. Cheap
/// approximation: every dependent counts towards every seed it could have
/// been reached from. For a single-seed diff this is exact; for a
/// multi-seed diff we attribute the dependent's hop count to each seed
/// (slight over-counting is fine for risk ranking).
fn dependent_count_per_seed(
    seeds: &[PathBuf],
    expansion: &[ExpansionHit],
) -> HashMap<PathBuf, u32> {
    let mut out: HashMap<PathBuf, u32> = HashMap::new();
    let n_dependents = expansion.iter().filter(|h| h.hop > 0).count();
    if n_dependents == 0 || seeds.is_empty() {
        return out;
    }
    // Spread credit evenly across seeds. With one seed this gives the full
    // count; with N seeds each gets ceil(n_dependents / N).
    let n = seeds.len() as u32;
    let per_seed = ((n_dependents as u32).div_ceil(n)).max(1);
    for s in seeds {
        out.insert(s.clone(), per_seed);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::parser::CodeSymbol;

    fn sym(name: &str, line: usize, end_line: usize) -> CodeSymbol {
        CodeSymbol {
            name: name.to_string(),
            kind: "function".to_string(),
            line,
            end_line,
            signature: None,
            children: Vec::new(),
        }
    }

    fn diff_one_file() -> &'static str {
        "diff --git a/src/auth.rs b/src/auth.rs\n\
         --- a/src/auth.rs\n\
         +++ b/src/auth.rs\n\
         @@ -10,3 +10,3 @@\n\
         -old line\n\
         -old line\n\
         -old line\n\
         +new line\n\
         +new line\n\
         +new line\n"
    }

    #[test]
    fn analyze_returns_changed_file_entry() {
        let symbols =
            HashMap::from([("src/auth.rs".to_string(), vec![sym("verify_token", 8, 20)])]);
        let reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();

        let report = analyze(
            diff_one_file(),
            &symbols,
            &reverse,
            ExpansionOptions::default(),
        );
        assert_eq!(report.total_changed_files, 1);
        assert_eq!(report.entries.len(), 1);
        let e = &report.entries[0];
        assert!(e.is_changed);
        assert_eq!(e.hop, 0);
        assert!(e.changed_lines >= 1);
        assert_eq!(e.changed_symbols, vec!["verify_token"]);
    }

    #[test]
    fn analyze_includes_dependent_files() {
        let symbols =
            HashMap::from([("src/auth.rs".to_string(), vec![sym("verify_token", 8, 20)])]);
        let mut reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        reverse.insert(
            PathBuf::from("src/auth.rs"),
            HashSet::from([PathBuf::from("src/api/login.rs")]),
        );

        let report = analyze(
            diff_one_file(),
            &symbols,
            &reverse,
            ExpansionOptions::default(),
        );
        assert!(report.total_dependent_files >= 1);
        assert!(
            report
                .entries
                .iter()
                .any(|e| e.path == Path::new("src/api/login.rs") && !e.is_changed)
        );
    }

    #[test]
    fn analyze_empty_diff_returns_empty_report() {
        let symbols: HashMap<String, Vec<CodeSymbol>> = HashMap::new();
        let reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        let report = analyze("", &symbols, &reverse, ExpansionOptions::default());
        assert_eq!(report.total_changed_files, 0);
        assert!(report.entries.is_empty());
    }

    #[test]
    fn analyze_orders_higher_risk_first() {
        // Two changed files: heavy.rs has 100 lines + 3 symbols, light.rs has 1 line + 0.
        let diff = "diff --git a/heavy.rs b/heavy.rs\n\
                    --- a/heavy.rs\n\
                    +++ b/heavy.rs\n\
                    @@ -1,100 +1,100 @@\n\
                    diff --git a/light.rs b/light.rs\n\
                    --- a/light.rs\n\
                    +++ b/light.rs\n\
                    @@ -5,1 +5,1 @@\n";
        let symbols = HashMap::from([
            (
                "heavy.rs".to_string(),
                vec![sym("a", 1, 30), sym("b", 31, 60), sym("c", 61, 100)],
            ),
            ("light.rs".to_string(), vec![]),
        ]);
        let reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        let report = analyze(diff, &symbols, &reverse, ExpansionOptions::default());
        assert_eq!(report.entries[0].path, PathBuf::from("heavy.rs"));
    }

    #[test]
    fn format_report_handles_empty() {
        let report = PrReviewReport {
            entries: Vec::new(),
            total_changed_files: 0,
            total_dependent_files: 0,
        };
        assert_eq!(
            format_report(&report),
            "PR review: empty diff (no files changed)."
        );
    }

    #[test]
    fn format_report_renders_summary_and_entries() {
        let symbols =
            HashMap::from([("src/auth.rs".to_string(), vec![sym("verify_token", 8, 20)])]);
        let reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        let report = analyze(
            diff_one_file(),
            &symbols,
            &reverse,
            ExpansionOptions::default(),
        );
        let out = format_report(&report);
        assert!(out.contains("PR review: 1 changed file"));
        assert!(out.contains("src/auth.rs"));
        assert!(out.contains("[changed]"));
        assert!(out.contains("verify_token"));
    }

    #[test]
    fn lines_per_file_sums_multi_hunk_changes() {
        let diff = "diff --git a/x.rs b/x.rs\n\
                    --- a/x.rs\n\
                    +++ b/x.rs\n\
                    @@ -1,5 +1,5 @@\n\
                    @@ -20,3 +20,3 @@\n";
        let report = analyze(
            diff,
            &HashMap::new(),
            &HashMap::new(),
            ExpansionOptions::default(),
        );
        let entry = &report.entries[0];
        // 5 lines + 3 lines = 8
        assert_eq!(entry.changed_lines, 8);
    }
}
