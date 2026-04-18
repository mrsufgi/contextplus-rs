//! Detect import cycles in a dependency graph using Tarjan's SCC algorithm.
//!
//! Given a graph produced by [`crate::core::dependent_expand::build_reverse_graph`]
//! (or any forward import graph — cycles are symmetric), find all strongly-connected
//! components with ≥2 nodes, plus self-loops.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One import cycle: an ordered ring where `files[i]` imports `files[(i+1) % n]`
/// and the last file imports the first. The ring is normalised so the
/// lexicographically smallest file appears first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyCycle {
    pub files: Vec<PathBuf>,
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Find all import cycles in `graph`.
///
/// `graph` maps a file to the set of files it imports (forward edges).
/// Singleton SCCs without a self-edge are **not** reported.
/// Returned cycles are sorted by length (shortest first); ties are broken
/// lexicographically on the normalised file list.
pub fn find_cycles(graph: &HashMap<PathBuf, HashSet<PathBuf>>) -> Vec<DependencyCycle> {
    let mut cycles = tarjan_sccs(graph);

    // Normalise each cycle: rotate so the lex-smallest file is first.
    for cycle in &mut cycles {
        normalise(&mut cycle.files);
    }

    // Sort: primary = length, secondary = lex order of the files list.
    cycles.sort_by(|a, b| {
        a.files
            .len()
            .cmp(&b.files.len())
            .then_with(|| a.files.cmp(&b.files))
    });

    cycles
}

/// Format cycles as a human-readable string.
pub fn format_cycles(cycles: &[DependencyCycle]) -> String {
    if cycles.is_empty() {
        return "No import cycles detected.".to_string();
    }

    let mut out = String::new();
    for (i, cycle) in cycles.iter().enumerate() {
        out.push_str(&format!("Cycle {} ({} files):\n", i + 1, cycle.files.len()));
        for file in &cycle.files {
            out.push_str(&format!("  {}\n", file.display()));
        }
        // Show the closing edge explicitly.
        if !cycle.files.is_empty() {
            out.push_str(&format!("  → {}\n", cycle.files[0].display()));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tarjan SCC implementation (iterative to avoid stack overflows on large graphs)
// ---------------------------------------------------------------------------

struct TarjanState {
    index: usize,
    stack: Vec<usize>,
    on_stack: Vec<bool>,
    indices: Vec<Option<usize>>,
    lowlinks: Vec<usize>,
    sccs: Vec<Vec<usize>>,
}

fn tarjan_sccs(graph: &HashMap<PathBuf, HashSet<PathBuf>>) -> Vec<DependencyCycle> {
    // Assign stable integer IDs to every node that appears in the graph.
    let mut nodes: Vec<PathBuf> = graph.keys().cloned().collect();
    // Also include nodes that appear only as targets (may have no out-edges).
    for targets in graph.values() {
        for t in targets {
            if !graph.contains_key(t) {
                nodes.push(t.clone());
            }
        }
    }
    nodes.sort();
    nodes.dedup();

    let n = nodes.len();
    let node_id: HashMap<&PathBuf, usize> = nodes.iter().enumerate().map(|(i, p)| (p, i)).collect();

    // Build adjacency list using integer IDs.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (src, dsts) in graph {
        if let Some(&sid) = node_id.get(src) {
            for dst in dsts {
                if let Some(&did) = node_id.get(dst) {
                    adj[sid].push(did);
                }
            }
        }
    }

    let mut state = TarjanState {
        index: 0,
        stack: Vec::new(),
        on_stack: vec![false; n],
        indices: vec![None; n],
        lowlinks: vec![0; n],
        sccs: Vec::new(),
    };

    for v in 0..n {
        if state.indices[v].is_none() {
            strongconnect_iterative(v, &adj, &mut state);
        }
    }

    // Convert SCCs to DependencyCycle, filtering singletons without self-loops.
    let mut cycles = Vec::new();
    for scc in &state.sccs {
        if scc.len() > 1 {
            let files: Vec<PathBuf> = scc.iter().map(|&i| nodes[i].clone()).collect();
            cycles.push(DependencyCycle { files });
        } else if scc.len() == 1 {
            let id = scc[0];
            // Self-loop: node points to itself.
            if adj[id].contains(&id) {
                cycles.push(DependencyCycle {
                    files: vec![nodes[id].clone()],
                });
            }
        }
    }

    cycles
}

/// Iterative version of Tarjan's strongconnect to avoid recursion-depth issues.
fn strongconnect_iterative(start: usize, adj: &[Vec<usize>], state: &mut TarjanState) {
    // Work stack: (node, iterator index into adj[node])
    let mut work: Vec<(usize, usize)> = vec![(start, 0)];

    state.indices[start] = Some(state.index);
    state.lowlinks[start] = state.index;
    state.index += 1;
    state.stack.push(start);
    state.on_stack[start] = true;

    while let Some((v, ei)) = work.last_mut() {
        let v = *v;
        if *ei < adj[v].len() {
            let w = adj[v][*ei];
            *ei += 1;
            if state.indices[w].is_none() {
                // Tree edge: recurse.
                state.indices[w] = Some(state.index);
                state.lowlinks[w] = state.index;
                state.index += 1;
                state.stack.push(w);
                state.on_stack[w] = true;
                work.push((w, 0));
            } else if state.on_stack[w] {
                // Back edge.
                let w_idx = state.indices[w].unwrap();
                if w_idx < state.lowlinks[v] {
                    state.lowlinks[v] = w_idx;
                }
            }
        } else {
            // All edges from v processed — pop.
            work.pop();
            if let Some(&(parent, _)) = work.last() {
                if state.lowlinks[v] < state.lowlinks[parent] {
                    state.lowlinks[parent] = state.lowlinks[v];
                }
            }
            // If v is a root of an SCC, pop the stack.
            if state.lowlinks[v] == state.indices[v].unwrap() {
                let mut scc = Vec::new();
                loop {
                    let w = state.stack.pop().unwrap();
                    state.on_stack[w] = false;
                    scc.push(w);
                    if w == v {
                        break;
                    }
                }
                state.sccs.push(scc);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Normalisation helpers
// ---------------------------------------------------------------------------

/// Rotate `ring` so the lexicographically smallest element is first.
fn normalise(ring: &mut Vec<PathBuf>) {
    if ring.is_empty() {
        return;
    }
    let min_pos = ring
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    ring.rotate_left(min_pos);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    fn make_graph(edges: &[(&str, &str)]) -> HashMap<PathBuf, HashSet<PathBuf>> {
        let mut g: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        for (src, dst) in edges {
            g.entry(p(src)).or_default().insert(p(dst));
        }
        g
    }

    // 1. Empty graph → no cycles.
    #[test]
    fn empty_graph() {
        let g: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
        assert!(find_cycles(&g).is_empty());
    }

    // 2. DAG (no cycles) → no cycles.
    #[test]
    fn dag_no_cycles() {
        let g = make_graph(&[("a", "b"), ("b", "c"), ("a", "c")]);
        assert!(find_cycles(&g).is_empty());
    }

    // 3. Simple 2-cycle A→B→A.
    #[test]
    fn two_cycle() {
        let g = make_graph(&[("a", "b"), ("b", "a")]);
        let cycles = find_cycles(&g);
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].files.len(), 2);
        // Normalised: lex-first is "a"
        assert_eq!(cycles[0].files[0], p("a"));
    }

    // 4. 3-cycle A→B→C→A.
    #[test]
    fn three_cycle() {
        let g = make_graph(&[("a", "b"), ("b", "c"), ("c", "a")]);
        let cycles = find_cycles(&g);
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].files.len(), 3);
        assert_eq!(cycles[0].files[0], p("a"));
    }

    // 5. Self-loop A→A counts as a 1-cycle.
    #[test]
    fn self_loop() {
        let g = make_graph(&[("a", "a")]);
        let cycles = find_cycles(&g);
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].files, vec![p("a")]);
    }

    // 6. DAG node connected to a separate cycle → only the cycle reported.
    #[test]
    fn dag_node_plus_cycle() {
        // x→a→b→a (x is a DAG entry point, a↔b form the cycle)
        let g = make_graph(&[("x", "a"), ("a", "b"), ("b", "a")]);
        let cycles = find_cycles(&g);
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].files.len(), 2);
        let names: Vec<&str> = cycles[0]
            .files
            .iter()
            .map(|p| p.to_str().unwrap())
            .collect();
        assert!(names.contains(&"a") && names.contains(&"b"));
    }

    // 7. Two independent cycles → both reported, sorted by length then lex.
    #[test]
    fn two_independent_cycles() {
        // Cycle 1: a↔b (length 2)
        // Cycle 2: c→d→e→c (length 3)
        let g = make_graph(&[("a", "b"), ("b", "a"), ("c", "d"), ("d", "e"), ("e", "c")]);
        let cycles = find_cycles(&g);
        assert_eq!(cycles.len(), 2);
        // Shortest first
        assert_eq!(cycles[0].files.len(), 2);
        assert_eq!(cycles[1].files.len(), 3);
        assert_eq!(cycles[0].files[0], p("a"));
        assert_eq!(cycles[1].files[0], p("c"));
    }

    // 8. Normalisation: A→B→A and B→A→B should yield the same canonical ring.
    #[test]
    fn normalisation_canonical() {
        let g1 = make_graph(&[("a", "b"), ("b", "a")]);
        let g2 = make_graph(&[("b", "a"), ("a", "b")]);
        let c1 = find_cycles(&g1);
        let c2 = find_cycles(&g2);
        assert_eq!(c1.len(), 1);
        assert_eq!(c2.len(), 1);
        // Both should start with "a" (lex-first), then "b".
        assert_eq!(c1[0].files, c2[0].files);
        assert_eq!(c1[0].files[0], p("a"));
        assert_eq!(c1[0].files[1], p("b"));
    }

    // 9. format_cycles on empty → "No import cycles detected."
    #[test]
    fn format_empty() {
        assert_eq!(format_cycles(&[]), "No import cycles detected.");
    }

    // 10. format_cycles on a 2-cycle produces expected lines.
    #[test]
    fn format_two_cycle() {
        let cycle = DependencyCycle {
            files: vec![p("a"), p("b")],
        };
        let out = format_cycles(&[cycle]);
        assert!(out.contains("Cycle 1 (2 files)"));
        assert!(out.contains("a"));
        assert!(out.contains("b"));
    }
}
