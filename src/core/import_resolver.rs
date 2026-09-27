//! Import path resolver — maps raw import specifiers to file paths.
//! Used for import-graph clustering in semantic_navigate.

use std::path::{Path, PathBuf};

/// Resolve a raw import specifier to an absolute file path.
/// Returns None for external imports (npm packages, node builtins).
pub fn resolve_import(import_path: &str, importing_file: &Path) -> Option<PathBuf> {
    resolve_import_with(import_path, importing_file, Path::is_file)
}

pub(crate) fn resolve_import_with(
    import_path: &str,
    importing_file: &Path,
    exists: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    // Skip non-relative imports (external packages)
    if !import_path.starts_with('.') {
        return None;
    }

    let dir = importing_file.parent()?;
    let mut base = PathBuf::new();
    for component in dir.join(import_path).components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                base.pop();
            }
            other => base.push(other.as_os_str()),
        }
    }

    // Try exact path first
    if exists(&base) {
        return Some(base);
    }

    // Try with extensions
    for ext in &[".ts", ".tsx", ".js", ".jsx", ".go", ".rs"] {
        let with_ext = base.with_extension(&ext[1..]);
        if exists(&with_ext) {
            return Some(with_ext);
        }
    }

    // Try as directory with index file
    for index in &["index.ts", "index.tsx", "index.js", "index.jsx"] {
        let index_path = base.join(index);
        if exists(&index_path) {
            return Some(index_path);
        }
    }

    None
}

/// Resolve all imports for a file, returning (importing_file, imported_file) pairs.
pub fn resolve_file_imports(file_path: &Path, raw_imports: &[String]) -> Vec<(PathBuf, PathBuf)> {
    raw_imports
        .iter()
        .filter_map(|import| {
            resolve_import(import, file_path).map(|resolved| (file_path.to_path_buf(), resolved))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn resolve_relative_import_with_extension() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();
        fs::write(dir.join("foo.ts"), "export const x = 1;").unwrap();
        fs::write(dir.join("bar.ts"), "import { x } from './foo';").unwrap();

        let result = resolve_import("./foo", &dir.join("bar.ts"));
        assert!(result.is_some());
        assert!(result.unwrap().ends_with("foo.ts"));
    }

    #[test]
    fn resolve_index_file() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();
        fs::create_dir_all(dir.join("utils")).unwrap();
        fs::write(dir.join("utils/index.ts"), "export const x = 1;").unwrap();
        fs::write(dir.join("main.ts"), "import { x } from './utils';").unwrap();

        let result = resolve_import("./utils", &dir.join("main.ts"));
        assert!(result.is_some());
    }

    #[test]
    fn skip_external_import() {
        let result = resolve_import("stripe", Path::new("/workspace/src/app.ts"));
        assert!(result.is_none());
    }

    #[test]
    fn skip_scoped_package_import() {
        let result = resolve_import("@acme/lib-context", Path::new("/workspace/src/app.ts"));
        assert!(result.is_none());
    }

    #[test]
    fn skip_node_builtin_import() {
        let result = resolve_import("fs/promises", Path::new("/workspace/src/app.ts"));
        assert!(result.is_none());
    }

    #[test]
    fn resolve_file_imports_filters_external() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();
        fs::write(dir.join("foo.ts"), "export const x = 1;").unwrap();
        let main_file = dir.join("main.ts");
        fs::write(&main_file, "").unwrap();

        let imports = vec![
            "./foo".to_string(),
            "stripe".to_string(),
            "@acme/lib-context".to_string(),
        ];

        let pairs = resolve_file_imports(&main_file, &imports);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].1.ends_with("foo.ts"));
    }
}
