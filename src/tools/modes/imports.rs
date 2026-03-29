use crate::core::import_resolver::resolve_import;
use crate::core::tree_sitter::extract_imports;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::super::semantic_navigate::FileInfo;

/// Extract import edges for all files. Returns (source_idx, target_idx) pairs
/// where source imports target. Used by the "imports" clustering mode.
pub(crate) fn extract_all_import_edges(files: &[FileInfo], root: &Path) -> Vec<(usize, usize)> {
    // Build a path → index lookup for resolving imports to file indices
    let path_to_idx: HashMap<PathBuf, usize> = files
        .iter()
        .enumerate()
        .map(|(i, f)| (root.join(&f.relative_path), i))
        .collect();

    let mut edges = Vec::new();
    for (i, file) in files.iter().enumerate() {
        let file_path = root.join(&file.relative_path);
        let raw_imports = extract_imports(&file_path);
        for raw in &raw_imports {
            if let Some(resolved) = resolve_import(raw, &file_path) {
                // Try canonicalized path to match the lookup
                if let Ok(canonical) = resolved.canonicalize()
                    && let Some(&j) = path_to_idx.get(&canonical)
                {
                    edges.push((i, j));
                    continue; // Found via canonical — skip fallback
                }
                // Also try the resolved path directly (for relative paths that
                // may not exist on disk yet or match without canonicalization)
                if let Some(&j) = path_to_idx.get(&resolved) {
                    edges.push((i, j));
                }
            }
        }
    }
    edges
}
