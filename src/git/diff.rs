//! Function-level git-diff → symbol mapping.
//!
//! Parses unified diff output (`git diff --unified=0 <ref>...HEAD`) and maps
//! each changed line range to the source-level symbols that contain it. This
//! lets reviewers (and review tools) reason about *which functions changed*
//! rather than *which lines changed* — far more meaningful for code review.
//!
//! The diff format we accept:
//! ```text
//! diff --git a/src/foo.rs b/src/foo.rs
//! @@ -10,3 +10,5 @@
//! @@ -42 +44,2 @@
//! ```
//!
//! Each `@@ -A,B +C,D @@` hunk header gives a "new" range starting at line C
//! covering D lines (D defaults to 1 when omitted, e.g. `+44`). For deletions
//! (D == 0) we record a zero-length range at line C — the deletion site.

use std::collections::HashMap;

use crate::core::parser::CodeSymbol;

/// A contiguous block of lines in the *post-change* version of a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRange {
    pub start: usize, // 1-indexed, inclusive
    pub end: usize,   // 1-indexed, inclusive (== start for single-line)
}

impl LineRange {
    pub fn contains(&self, line: usize) -> bool {
        line >= self.start && line <= self.end
    }

    pub fn overlaps_symbol(&self, symbol: &CodeSymbol) -> bool {
        // CodeSymbol.line/end_line are 1-indexed inclusive.
        // Standard interval overlap: max(starts) <= min(ends).
        self.start.max(symbol.line) <= self.end.min(symbol.end_line)
    }
}

/// All changed line ranges in the new version of one file.
#[derive(Debug, Clone, Default)]
pub struct FileChange {
    pub path: String,
    pub ranges: Vec<LineRange>,
}

/// Parse `git diff --unified=0` output into per-file changed line ranges.
///
/// Renames produce two entries (old + new path) — callers should resolve via
/// the symbol map they pass to [`changed_symbols`].
///
/// Pure deletions produce a zero-length range at the deletion site (the line
/// where content used to be). This still flags the surrounding symbol.
pub fn parse_unified_diff(diff: &str) -> Vec<FileChange> {
    let mut out: Vec<FileChange> = Vec::new();
    let mut current: Option<FileChange> = None;

    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            if let Some(c) = current.take() {
                out.push(c);
            }
            current = Some(FileChange {
                path: rest.trim().to_string(),
                ranges: Vec::new(),
            });
        } else if let Some(rest) = line.strip_prefix("+++ ") {
            // `+++ /dev/null` (deletion) — still record so callers see the file
            if let Some(c) = current.take() {
                out.push(c);
            }
            if rest.trim() != "/dev/null" {
                current = Some(FileChange {
                    path: rest.trim().to_string(),
                    ranges: Vec::new(),
                });
            }
        } else if line.starts_with("@@") {
            if let Some(range) = parse_hunk_header(line)
                && let Some(c) = current.as_mut()
            {
                c.ranges.push(range);
            }
        }
    }
    if let Some(c) = current.take() {
        out.push(c);
    }
    out
}

/// Parse a `@@ -A[,B] +C[,D] @@ ...` hunk header. Returns the new-side range.
/// Supports the elided-count form (`+44` == `+44,1`) and zero-count deletions
/// (`+44,0` collapses to a one-line range at the surrounding line).
fn parse_hunk_header(line: &str) -> Option<LineRange> {
    // Strip the leading `@@` and trailing `@@ ...`
    let inner = line.strip_prefix("@@")?.trim_start();
    let plus_idx = inner.find('+')?;
    let after_plus = &inner[plus_idx + 1..];
    let end = after_plus
        .find(' ')
        .or_else(|| after_plus.find('@'))
        .unwrap_or(after_plus.len());
    let spec = &after_plus[..end];
    let (start_str, count_str) = match spec.split_once(',') {
        Some((s, c)) => (s, c),
        None => (spec, "1"),
    };
    let start: usize = start_str.parse().ok()?;
    let count: usize = count_str.parse().ok()?;
    let end_line = if count == 0 {
        // pure deletion — flag the surrounding line (start may be 0 if file new; clamp to 1)
        start.max(1)
    } else {
        start + count - 1
    };
    Some(LineRange {
        start: start.max(1),
        end: end_line.max(start.max(1)),
    })
}

/// Map each changed file to the symbols whose source range intersects any
/// changed line. Symbols are matched by walking the (already-parsed)
/// `symbols_by_file` map.
///
/// Symbols with no name overlap (e.g. a top-level `import` block touched by an
/// edit) are still returned — the caller decides whether to filter them.
pub fn changed_symbols<'a>(
    changes: &[FileChange],
    symbols_by_file: &'a HashMap<String, Vec<CodeSymbol>>,
) -> Vec<ChangedSymbol<'a>> {
    let mut out = Vec::new();
    for change in changes {
        let Some(symbols) = symbols_by_file.get(&change.path) else {
            continue;
        };
        for sym in symbols {
            collect_changed(sym, &change.ranges, &change.path, &mut out);
        }
    }
    out
}

fn collect_changed<'a>(
    symbol: &'a CodeSymbol,
    ranges: &[LineRange],
    path: &str,
    out: &mut Vec<ChangedSymbol<'a>>,
) {
    if ranges.iter().any(|r| r.overlaps_symbol(symbol)) {
        out.push(ChangedSymbol {
            path: path.to_string(),
            symbol,
        });
    }
    for child in &symbol.children {
        collect_changed(child, ranges, path, out);
    }
}

/// A symbol whose source range overlaps a diff hunk. Holds a borrowed
/// reference into the caller's symbol map so we don't clone large trees.
#[derive(Debug)]
pub struct ChangedSymbol<'a> {
    pub path: String,
    pub symbol: &'a CodeSymbol,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(name: &str, kind: &str, start: usize, end: usize) -> CodeSymbol {
        CodeSymbol {
            name: name.to_string(),
            kind: kind.to_string(),
            line: start,
            end_line: end,
            signature: None,
            children: Vec::new(),
        }
    }

    #[test]
    fn parses_simple_added_range() {
        let diff = "\
diff --git a/src/foo.rs b/src/foo.rs
--- a/src/foo.rs
+++ b/src/foo.rs
@@ -10,3 +10,5 @@
 unchanged
+new
+new
";
        let changes = parse_unified_diff(diff);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, "src/foo.rs");
        assert_eq!(changes[0].ranges, vec![LineRange { start: 10, end: 14 }]);
    }

    #[test]
    fn parses_elided_count_as_one_line() {
        let diff = "\
+++ b/src/bar.rs
@@ -42 +44 @@
-old
+new
";
        let ranges = parse_unified_diff(diff)[0].ranges.clone();
        assert_eq!(ranges, vec![LineRange { start: 44, end: 44 }]);
    }

    #[test]
    fn parses_pure_deletion_as_one_line_anchor() {
        let diff = "\
+++ b/src/baz.rs
@@ -10,3 +9,0 @@
-gone
-gone
-gone
";
        // count=0 — collapses to a one-line range at line 9.
        let ranges = parse_unified_diff(diff)[0].ranges.clone();
        assert_eq!(ranges, vec![LineRange { start: 9, end: 9 }]);
    }

    #[test]
    fn parses_multiple_hunks_in_one_file() {
        let diff = "\
+++ b/src/multi.rs
@@ -1,2 +1,3 @@
@@ -50 +51,5 @@
@@ -100,1 +106 @@
";
        let ranges = parse_unified_diff(diff)[0].ranges.clone();
        assert_eq!(
            ranges,
            vec![
                LineRange { start: 1, end: 3 },
                LineRange { start: 51, end: 55 },
                LineRange { start: 106, end: 106 },
            ]
        );
    }

    #[test]
    fn parses_multiple_files() {
        let diff = "\
+++ b/a.rs
@@ -1 +1 @@
+++ b/b.rs
@@ -5,2 +5,3 @@
";
        let changes = parse_unified_diff(diff);
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].path, "a.rs");
        assert_eq!(changes[1].path, "b.rs");
    }

    #[test]
    fn skips_dev_null_for_pure_delete() {
        let diff = "\
+++ /dev/null
@@ -1,3 +0,0 @@
-old
-old
-old
";
        let changes = parse_unified_diff(diff);
        assert!(changes.is_empty(), "deletion-only diff should produce no FileChange");
    }

    #[test]
    fn changed_symbols_finds_overlapping_function() {
        let diff = "\
+++ b/src/x.rs
@@ -10,2 +10,2 @@
";
        let changes = parse_unified_diff(diff);

        let mut map = HashMap::new();
        map.insert(
            "src/x.rs".to_string(),
            vec![
                sym("untouched", "function", 1, 5),
                sym("touched", "function", 8, 20),
                sym("also_untouched", "function", 25, 30),
            ],
        );

        let hits = changed_symbols(&changes, &map);
        let names: Vec<_> = hits.iter().map(|h| h.symbol.name.as_str()).collect();
        assert_eq!(names, vec!["touched"]);
    }

    #[test]
    fn changed_symbols_descends_into_children() {
        let diff = "\
+++ b/src/x.rs
@@ -15 +15 @@
";
        let changes = parse_unified_diff(diff);

        let mut parent = sym("Outer", "class", 10, 30);
        parent.children.push(sym("inner_method", "method", 14, 18));
        let mut map = HashMap::new();
        map.insert("src/x.rs".to_string(), vec![parent]);

        let hits = changed_symbols(&changes, &map);
        // Both Outer (overlaps) and inner_method (overlaps) should be returned.
        let names: Vec<_> = hits.iter().map(|h| h.symbol.name.as_str()).collect();
        assert_eq!(names, vec!["Outer", "inner_method"]);
    }

    #[test]
    fn changed_symbols_ignores_files_with_no_parsed_symbols() {
        let diff = "\
+++ b/unknown.rs
@@ -1 +1 @@
";
        let changes = parse_unified_diff(diff);
        let map: HashMap<String, Vec<CodeSymbol>> = HashMap::new();
        assert!(changed_symbols(&changes, &map).is_empty());
    }

    #[test]
    fn line_range_contains_inclusive_bounds() {
        let r = LineRange { start: 5, end: 10 };
        assert!(r.contains(5));
        assert!(r.contains(10));
        assert!(!r.contains(4));
        assert!(!r.contains(11));
    }

    #[test]
    fn line_range_overlap_edge_cases() {
        let r = LineRange { start: 10, end: 20 };
        assert!(r.overlaps_symbol(&sym("touch_start", "fn", 5, 10)));
        assert!(r.overlaps_symbol(&sym("touch_end", "fn", 20, 30)));
        assert!(r.overlaps_symbol(&sym("contains", "fn", 1, 100)));
        assert!(!r.overlaps_symbol(&sym("before", "fn", 1, 9)));
        assert!(!r.overlaps_symbol(&sym("after", "fn", 21, 25)));
    }
}
