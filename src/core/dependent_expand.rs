//! 2-hop reverse-dependency expansion.
//!
//! Given a seed set of files (e.g. files changed in a PR), returns the files
//! that import them transitively up to N hops. This is the foundation of
//! "review-aware" tools — surfacing the immediate blast radius of an edit
//! without paying for a full project-wide blast scan.
//!
//! Builds a reverse import graph lazily by parsing each file's imports via
//! [`crate::core::tree_sitter::extract_imports`] and resolving them through
//! [`crate::core::import_resolver`]. The forward edges become reverse edges:
//! `imports("a.ts") = ["./b"]` ⇒ reverse(`b.ts`) ⊇ {`a.ts`}.
//!
//! Capped at `max_files` per call (default 500) so a hub module doesn't drag
//! the entire project into a result set.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::core::import_resolver;
use crate::core::tree_sitter as ts;

const DEFAULT_MAX_HOPS: usize = 2;
const DEFAULT_MAX_FILES: usize = 500;

/// Tunables for [`expand_dependents`].
#[derive(Debug, Clone, Copy)]
pub struct ExpansionOptions {
    pub max_hops: usize,
    pub max_files: usize,
}

impl Default for ExpansionOptions {
    fn default() -> Self {
        Self {
            max_hops: DEFAULT_MAX_HOPS,
            max_files: DEFAULT_MAX_FILES,
        }
    }
}

/// One entry in the expansion result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpansionHit {
    pub path: PathBuf,
    /// 0 for seeds, 1 for direct importers, 2 for grand-importers.
    pub hop: usize,
}

/// Build a reverse import graph from a list of source files. Each entry maps a
/// file to the set of files that import it.
///
/// `all_files` should be the complete set of files to consider (typically the
/// project's tracked source files). Files outside this set won't appear in the
/// graph even if they're imported.
pub fn build_reverse_graph(all_files: &[PathBuf]) -> HashMap<PathBuf, HashSet<PathBuf>> {
    let in_set: HashSet<&Path> = all_files.iter().map(|p| p.as_path()).collect();

    // Parse imports in parallel (each file is independent CPU + blocking I/O).
    // Collect per-file edge lists, then merge sequentially into the final map.
    let edge_lists: Vec<Vec<(PathBuf, PathBuf)>> = all_files
        .par_iter()
        .map(|file| {
            let imports = ts::extract_imports(file);
            import_resolver::resolve_file_imports(file, &imports)
                .into_iter()
                .filter(|(_, imported)| in_set.contains(imported.as_path()))
                .collect()
        })
        .collect();

    let mut reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
    for edges in edge_lists {
        for (importing, imported) in edges {
            // Only record edges where both endpoints are in the project set;
            // an external dependency or a path outside the project would only
            // pollute the reverse graph.
            reverse.entry(imported).or_default().insert(importing);
        }
    }
    reverse
}

/// BFS from `seeds` over the reverse graph, returning seeds + their dependents
/// up to `opts.max_hops` away. Cap result at `opts.max_files` (BFS order, so
/// closer files are preferred).
///
/// Each result carries its hop distance from the nearest seed (0 for seeds).
pub fn expand_dependents(
    reverse: &HashMap<PathBuf, HashSet<PathBuf>>,
    seeds: &[PathBuf],
    opts: ExpansionOptions,
) -> Vec<ExpansionHit> {
    let mut visited: HashMap<PathBuf, usize> = HashMap::new();
    let mut queue: VecDeque<(PathBuf, usize)> = VecDeque::new();
    let mut order: Vec<PathBuf> = Vec::new();

    for s in seeds {
        if visited.insert(s.clone(), 0).is_none() {
            queue.push_back((s.clone(), 0));
            order.push(s.clone());
        }
    }

    while let Some((node, hop)) = queue.pop_front() {
        if order.len() >= opts.max_files {
            break;
        }
        if hop >= opts.max_hops {
            continue;
        }
        let Some(parents) = reverse.get(&node) else {
            continue;
        };
        // Sort for deterministic output across runs.
        let mut sorted: Vec<&PathBuf> = parents.iter().collect();
        sorted.sort();
        for p in sorted {
            if order.len() >= opts.max_files {
                break;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(p.clone()) {
                e.insert(hop + 1);
                queue.push_back((p.clone(), hop + 1));
                order.push(p.clone());
            }
        }
    }

    order
        .into_iter()
        .map(|p| {
            let hop = *visited.get(&p).unwrap_or(&0);
            ExpansionHit { path: p, hop }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn expand_returns_seeds_at_hop_zero_when_no_graph() {
        let reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        let seeds = vec![PathBuf::from("a.ts")];
        let hits = expand_dependents(&reverse, &seeds, ExpansionOptions::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hop, 0);
    }

    #[test]
    fn expand_finds_direct_importers_at_hop_one() {
        let mut reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        // b.ts and c.ts both import a.ts
        reverse.insert(
            PathBuf::from("a.ts"),
            HashSet::from([PathBuf::from("b.ts"), PathBuf::from("c.ts")]),
        );
        let hits = expand_dependents(
            &reverse,
            &[PathBuf::from("a.ts")],
            ExpansionOptions::default(),
        );
        // seed + 2 importers
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].hop, 0);
        assert!(
            hits.iter()
                .any(|h| h.path == Path::new("b.ts") && h.hop == 1)
        );
        assert!(
            hits.iter()
                .any(|h| h.path == Path::new("c.ts") && h.hop == 1)
        );
    }

    #[test]
    fn expand_finds_grand_importers_at_hop_two() {
        // d.ts → c.ts → a.ts. Seed = a.ts. Expect d.ts at hop 2.
        let mut reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        reverse.insert(
            PathBuf::from("a.ts"),
            HashSet::from([PathBuf::from("c.ts")]),
        );
        reverse.insert(
            PathBuf::from("c.ts"),
            HashSet::from([PathBuf::from("d.ts")]),
        );

        let hits = expand_dependents(
            &reverse,
            &[PathBuf::from("a.ts")],
            ExpansionOptions::default(),
        );
        assert!(
            hits.iter()
                .any(|h| h.path == Path::new("c.ts") && h.hop == 1)
        );
        assert!(
            hits.iter()
                .any(|h| h.path == Path::new("d.ts") && h.hop == 2)
        );
    }

    #[test]
    fn expand_respects_max_hops() {
        let mut reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        reverse.insert(
            PathBuf::from("a.ts"),
            HashSet::from([PathBuf::from("c.ts")]),
        );
        reverse.insert(
            PathBuf::from("c.ts"),
            HashSet::from([PathBuf::from("d.ts")]),
        );
        reverse.insert(
            PathBuf::from("d.ts"),
            HashSet::from([PathBuf::from("e.ts")]),
        );

        let opts = ExpansionOptions {
            max_hops: 1,
            max_files: 100,
        };
        let hits = expand_dependents(&reverse, &[PathBuf::from("a.ts")], opts);
        assert!(hits.iter().any(|h| h.path == Path::new("c.ts")));
        assert!(!hits.iter().any(|h| h.path == Path::new("d.ts")));
        assert!(!hits.iter().any(|h| h.path == Path::new("e.ts")));
    }

    #[test]
    fn expand_caps_at_max_files() {
        let mut reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        let many: HashSet<PathBuf> = (0..50)
            .map(|i| PathBuf::from(format!("dep{}.ts", i)))
            .collect();
        reverse.insert(PathBuf::from("hub.ts"), many);

        let opts = ExpansionOptions {
            max_hops: 2,
            max_files: 10,
        };
        let hits = expand_dependents(&reverse, &[PathBuf::from("hub.ts")], opts);
        assert_eq!(hits.len(), 10);
    }

    #[test]
    fn expand_dedupes_diamond_imports() {
        // Diamond: x → a, x → b, both a and b → seed
        let mut reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        reverse.insert(
            PathBuf::from("seed.ts"),
            HashSet::from([PathBuf::from("a.ts"), PathBuf::from("b.ts")]),
        );
        reverse.insert(
            PathBuf::from("a.ts"),
            HashSet::from([PathBuf::from("x.ts")]),
        );
        reverse.insert(
            PathBuf::from("b.ts"),
            HashSet::from([PathBuf::from("x.ts")]),
        );

        let hits = expand_dependents(
            &reverse,
            &[PathBuf::from("seed.ts")],
            ExpansionOptions::default(),
        );
        let x_count = hits.iter().filter(|h| h.path == Path::new("x.ts")).count();
        assert_eq!(x_count, 1, "diamond importer must appear exactly once");
    }

    #[test]
    fn expand_handles_cycle_without_infinite_loop() {
        // a ↔ b: both import each other
        let mut reverse: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        reverse.insert(
            PathBuf::from("a.ts"),
            HashSet::from([PathBuf::from("b.ts")]),
        );
        reverse.insert(
            PathBuf::from("b.ts"),
            HashSet::from([PathBuf::from("a.ts")]),
        );

        let hits = expand_dependents(
            &reverse,
            &[PathBuf::from("a.ts")],
            ExpansionOptions::default(),
        );
        // Just verifies termination + no duplication of seed.
        let a_count = hits.iter().filter(|h| h.path == Path::new("a.ts")).count();
        assert_eq!(a_count, 1);
    }

    #[test]
    fn build_reverse_graph_from_real_files() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();
        fs::write(dir.join("a.ts"), "export const a = 1;").unwrap();
        fs::write(
            dir.join("b.ts"),
            "import { a } from './a';\nexport const b = a;",
        )
        .unwrap();
        fs::write(
            dir.join("c.ts"),
            "import { b } from './b';\nexport const c = b;",
        )
        .unwrap();

        let files = vec![dir.join("a.ts"), dir.join("b.ts"), dir.join("c.ts")];
        let reverse = build_reverse_graph(&files);

        // a.ts is imported by b.ts
        let importers_of_a = reverse.get(&dir.join("a.ts")).expect("a.ts has importers");
        assert!(importers_of_a.contains(&dir.join("b.ts")));

        // b.ts is imported by c.ts
        let importers_of_b = reverse.get(&dir.join("b.ts")).expect("b.ts has importers");
        assert!(importers_of_b.contains(&dir.join("c.ts")));
    }

    #[test]
    fn build_reverse_graph_skips_external_imports() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();
        fs::write(
            dir.join("app.ts"),
            "import { Stripe } from 'stripe';\nimport { local } from './local';",
        )
        .unwrap();
        fs::write(dir.join("local.ts"), "export const local = 1;").unwrap();

        let files = vec![dir.join("app.ts"), dir.join("local.ts")];
        let reverse = build_reverse_graph(&files);

        // No entry for "stripe" (external)
        assert!(!reverse.contains_key(&PathBuf::from("stripe")));
        // local is correctly inverted
        assert!(
            reverse
                .get(&dir.join("local.ts"))
                .map(|s| s.contains(&dir.join("app.ts")))
                .unwrap_or(false)
        );
    }

    #[test]
    fn end_to_end_expansion_from_real_files() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();
        fs::write(dir.join("util.ts"), "export const x = 1;").unwrap();
        fs::write(dir.join("svc.ts"), "import { x } from './util';").unwrap();
        fs::write(dir.join("api.ts"), "import './svc';").unwrap();

        let files = vec![dir.join("util.ts"), dir.join("svc.ts"), dir.join("api.ts")];
        let reverse = build_reverse_graph(&files);

        let hits = expand_dependents(
            &reverse,
            &[dir.join("util.ts")],
            ExpansionOptions::default(),
        );
        let names: Vec<String> = hits
            .iter()
            .map(|h| h.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();

        assert!(names.contains(&"util.ts".to_string())); // seed
        assert!(names.contains(&"svc.ts".to_string())); // hop 1
        assert!(names.contains(&"api.ts".to_string())); // hop 2
    }

    /// Parallel parity: build_reverse_graph on ≥100 files must produce the
    /// same edge set as a sequential reference built from the same inputs.
    #[test]
    fn build_reverse_graph_parallel_matches_sequential_100_files() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // 100 leaf files (no imports)
        for i in 0..100usize {
            fs::write(dir.join(format!("leaf{i}.ts")), "export const x = 1;").unwrap();
        }
        // 10 hub files each importing 10 leaves
        for h in 0..10usize {
            let imports: String = (0..10)
                .map(|i| format!("import {{ x }} from './leaf{}';", h * 10 + i))
                .collect::<Vec<_>>()
                .join("\n");
            fs::write(dir.join(format!("hub{h}.ts")), imports).unwrap();
        }

        let mut all_files: Vec<PathBuf> = (0..100)
            .map(|i| dir.join(format!("leaf{i}.ts")))
            .chain((0..10).map(|h| dir.join(format!("hub{h}.ts"))))
            .collect();
        all_files.sort(); // deterministic order for both runs

        // Parallel (new implementation)
        let parallel_result = build_reverse_graph(&all_files);

        // Sequential reference
        let mut seq_result: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        let in_set: HashSet<&std::path::Path> = all_files.iter().map(|p| p.as_path()).collect();
        for file in &all_files {
            let imports = crate::core::tree_sitter::extract_imports(file);
            for (importing, imported) in
                crate::core::import_resolver::resolve_file_imports(file, &imports)
            {
                if in_set.contains(imported.as_path()) {
                    seq_result.entry(imported).or_default().insert(importing);
                }
            }
        }

        // Both graphs must have the same keys
        let mut par_keys: Vec<&PathBuf> = parallel_result.keys().collect();
        let mut seq_keys: Vec<&PathBuf> = seq_result.keys().collect();
        par_keys.sort();
        seq_keys.sort();
        assert_eq!(par_keys, seq_keys, "reverse graph keys must match");

        // Each key must have the same importer set
        for key in par_keys {
            assert_eq!(
                parallel_result[key], seq_result[key],
                "importer set for {key:?} must match"
            );
        }
    }
}
