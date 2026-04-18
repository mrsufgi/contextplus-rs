//! `dead_code_find` — heuristic detector for symbols that appear unused.
//!
//! ## Heuristic
//!
//! A symbol is reported as **potentially dead** when its name does not
//! appear as a token in *any other* indexed file. The detector is
//! deliberately conservative: it produces candidates, not deletions.
//!
//! Known limitations (callers MUST treat output as advisory):
//!   - public API exposed across crate boundaries can look unused
//!   - trait method implementations called via dynamic dispatch
//!   - symbols invoked through FFI / reflection / macros
//!   - re-exports — the re-exporting file might just re-export by glob
//!
//! `ignore_kinds` lets callers filter out symbol kinds that almost
//! always produce false positives (e.g. `mod`, `impl`).
//!
//! ## Inputs
//!
//! Pure: callers supply the parsed symbol map and a per-file set of
//! identifiers (the *token map*). The MCP tool wiring is responsible for
//! building these from disk.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::core::parser::CodeSymbol;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeadSymbol {
    pub path: PathBuf,
    pub name: String,
    pub kind: String,
    pub line: usize,
}

/// Tunables. Defaults skip kinds whose unused-detection signal is unreliable.
#[derive(Debug, Clone)]
pub struct DeadCodeOptions {
    /// Symbol kinds to skip entirely (case-insensitive). Defaults skip
    /// `mod`, `impl`, `trait`, `test` — common false-positive sources.
    pub ignore_kinds: HashSet<String>,
    /// Symbol names to skip entirely (case-insensitive). Defaults skip
    /// well-known framework entry points (`main`, `default`, etc.).
    pub ignore_names: HashSet<String>,
    /// Cap the result list — first N symbols (sorted by path, then line).
    pub max_results: usize,
}

impl Default for DeadCodeOptions {
    fn default() -> Self {
        let ignore_kinds = ["mod", "impl", "trait", "test"]
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        let ignore_names = [
            "main", "default", "new", "drop", "clone", "fmt", "from", "into",
        ]
        .iter()
        .map(|s| s.to_lowercase())
        .collect();
        Self {
            ignore_kinds,
            ignore_names,
            max_results: 200,
        }
    }
}

/// Find potentially dead symbols. See module docs for the heuristic.
pub fn find_dead_symbols(
    symbols_by_file: &HashMap<PathBuf, Vec<CodeSymbol>>,
    tokens_by_file: &HashMap<PathBuf, HashSet<String>>,
    opts: &DeadCodeOptions,
) -> Vec<DeadSymbol> {
    // Build a (token → defining-file) reverse map so we can quickly check
    // whether an identifier is used outside its defining file.
    let mut name_use_count: HashMap<&str, HashSet<&Path>> = HashMap::new();
    for (path, tokens) in tokens_by_file {
        for tok in tokens {
            name_use_count
                .entry(tok.as_str())
                .or_default()
                .insert(path.as_path());
        }
    }

    let mut out = Vec::new();
    for (path, symbols) in symbols_by_file {
        for sym in symbols {
            collect_dead(path, sym, &name_use_count, opts, &mut out);
        }
    }

    // Deterministic order: path then line.
    out.sort_by(|a, b| a.path.cmp(&b.path).then(a.line.cmp(&b.line)));
    out.truncate(opts.max_results);
    out
}

fn collect_dead(
    path: &Path,
    sym: &CodeSymbol,
    name_use: &HashMap<&str, HashSet<&Path>>,
    opts: &DeadCodeOptions,
    out: &mut Vec<DeadSymbol>,
) {
    let kind_lc = sym.kind.to_lowercase();
    let name_lc = sym.name.to_lowercase();
    let skip = opts.ignore_kinds.contains(&kind_lc) || opts.ignore_names.contains(&name_lc);

    if !skip && !sym.name.is_empty() {
        let users = name_use.get(sym.name.as_str());
        let used_elsewhere = users
            .map(|set| set.iter().any(|p| *p != path))
            .unwrap_or(false);
        if !used_elsewhere {
            out.push(DeadSymbol {
                path: path.to_path_buf(),
                name: sym.name.clone(),
                kind: sym.kind.clone(),
                line: sym.line,
            });
        }
    }

    for child in &sym.children {
        collect_dead(path, child, name_use, opts, out);
    }
}

/// Format a list of dead symbols as text for tool output.
pub fn format_dead_symbols(items: &[DeadSymbol]) -> String {
    if items.is_empty() {
        return "No dead-symbol candidates found.".to_string();
    }
    let mut lines = vec![format!("Found {} dead-symbol candidate(s):", items.len())];
    lines.push(
        "(Heuristic — verify before deletion. Public APIs, trait methods, \
         and reflection-invoked symbols may show up here.)"
            .to_string(),
    );
    lines.push(String::new());
    for d in items {
        lines.push(format!(
            "{}:{}  {} {}",
            d.path.display(),
            d.line,
            d.kind,
            d.name
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(name: &str, kind: &str, line: usize) -> CodeSymbol {
        CodeSymbol {
            name: name.to_string(),
            kind: kind.to_string(),
            line,
            end_line: line + 5,
            signature: None,
            children: Vec::new(),
        }
    }

    fn sym_with_children(
        name: &str,
        kind: &str,
        line: usize,
        children: Vec<CodeSymbol>,
    ) -> CodeSymbol {
        CodeSymbol {
            name: name.to_string(),
            kind: kind.to_string(),
            line,
            end_line: line + 10,
            signature: None,
            children,
        }
    }

    fn options_no_filters() -> DeadCodeOptions {
        DeadCodeOptions {
            ignore_kinds: HashSet::new(),
            ignore_names: HashSet::new(),
            max_results: 100,
        }
    }

    #[test]
    fn flags_symbol_with_no_external_users() {
        let path_a = PathBuf::from("a.rs");
        let path_b = PathBuf::from("b.rs");
        let symbols = HashMap::from([(path_a.clone(), vec![sym("orphan", "function", 10)])]);
        let tokens = HashMap::from([
            (path_a.clone(), HashSet::from(["orphan".to_string()])),
            (path_b.clone(), HashSet::from(["unrelated".to_string()])),
        ]);
        let dead = find_dead_symbols(&symbols, &tokens, &options_no_filters());
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].name, "orphan");
    }

    #[test]
    fn does_not_flag_symbol_used_in_other_file() {
        let path_a = PathBuf::from("a.rs");
        let path_b = PathBuf::from("b.rs");
        let symbols = HashMap::from([(path_a.clone(), vec![sym("used", "function", 10)])]);
        let tokens = HashMap::from([
            (path_a.clone(), HashSet::from(["used".to_string()])),
            (path_b.clone(), HashSet::from(["used".to_string()])),
        ]);
        let dead = find_dead_symbols(&symbols, &tokens, &options_no_filters());
        assert!(dead.is_empty());
    }

    #[test]
    fn skips_ignored_kinds() {
        let path_a = PathBuf::from("a.rs");
        let symbols = HashMap::from([(
            path_a.clone(),
            vec![sym("MyTrait", "trait", 5), sym("foo", "function", 20)],
        )]);
        let tokens = HashMap::from([(
            path_a.clone(),
            HashSet::from(["MyTrait".to_string(), "foo".to_string()]),
        )]);
        let dead = find_dead_symbols(&symbols, &tokens, &DeadCodeOptions::default());
        assert_eq!(dead.len(), 1, "trait should be skipped, foo flagged");
        assert_eq!(dead[0].name, "foo");
    }

    #[test]
    fn skips_ignored_names() {
        let path_a = PathBuf::from("a.rs");
        let symbols = HashMap::from([(path_a.clone(), vec![sym("main", "function", 1)])]);
        let tokens = HashMap::from([(path_a.clone(), HashSet::from(["main".to_string()]))]);
        let dead = find_dead_symbols(&symbols, &tokens, &DeadCodeOptions::default());
        assert!(dead.is_empty());
    }

    #[test]
    fn descends_into_children() {
        let path_a = PathBuf::from("a.rs");
        let parent = sym_with_children(
            "Parent",
            "struct",
            5,
            vec![sym("dead_method", "function", 10)],
        );
        let path_b = PathBuf::from("b.rs");
        let symbols = HashMap::from([(path_a.clone(), vec![parent])]);
        let tokens = HashMap::from([
            (
                path_a.clone(),
                HashSet::from(["dead_method".to_string(), "Parent".to_string()]),
            ),
            (path_b.clone(), HashSet::from(["Parent".to_string()])),
        ]);
        let dead = find_dead_symbols(&symbols, &tokens, &options_no_filters());
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].name, "dead_method");
    }

    #[test]
    fn empty_inputs_return_empty() {
        let symbols: HashMap<PathBuf, Vec<CodeSymbol>> = HashMap::new();
        let tokens: HashMap<PathBuf, HashSet<String>> = HashMap::new();
        assert!(find_dead_symbols(&symbols, &tokens, &options_no_filters()).is_empty());
    }

    #[test]
    fn respects_max_results_cap() {
        let path_a = PathBuf::from("a.rs");
        let symbols = HashMap::from([(
            path_a.clone(),
            (0..10)
                .map(|i| sym(&format!("dead_{i}"), "function", 10 + i))
                .collect(),
        )]);
        let tokens = HashMap::from([(
            path_a.clone(),
            (0..10).map(|i| format!("dead_{i}")).collect::<HashSet<_>>(),
        )]);
        let mut opts = options_no_filters();
        opts.max_results = 3;
        let dead = find_dead_symbols(&symbols, &tokens, &opts);
        assert_eq!(dead.len(), 3);
    }

    #[test]
    fn output_sorted_by_path_then_line() {
        let path_a = PathBuf::from("a.rs");
        let path_b = PathBuf::from("b.rs");
        let symbols = HashMap::from([
            (path_b.clone(), vec![sym("z1", "function", 50)]),
            (
                path_a.clone(),
                vec![sym("a2", "function", 20), sym("a1", "function", 5)],
            ),
        ]);
        let tokens = HashMap::from([
            (
                path_a.clone(),
                HashSet::from(["a1".to_string(), "a2".to_string()]),
            ),
            (path_b.clone(), HashSet::from(["z1".to_string()])),
        ]);
        let dead = find_dead_symbols(&symbols, &tokens, &options_no_filters());
        assert_eq!(dead[0].path, path_a);
        assert_eq!(dead[0].line, 5);
        assert_eq!(dead[1].path, path_a);
        assert_eq!(dead[1].line, 20);
        assert_eq!(dead[2].path, path_b);
    }

    #[test]
    fn format_output_handles_empty() {
        assert_eq!(format_dead_symbols(&[]), "No dead-symbol candidates found.");
    }

    #[test]
    fn format_output_renders_entries() {
        let items = vec![DeadSymbol {
            path: PathBuf::from("src/foo.rs"),
            name: "bar".to_string(),
            kind: "function".to_string(),
            line: 42,
        }];
        let out = format_dead_symbols(&items);
        assert!(out.contains("src/foo.rs:42"));
        assert!(out.contains("function bar"));
        assert!(out.contains("1 dead-symbol candidate"));
    }
}
