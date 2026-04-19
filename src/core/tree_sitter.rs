use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use regex::Regex;
use tree_sitter::{Language, Node, Parser};

use crate::core::parser::CodeSymbol;
use crate::error::{ContextPlusError, Result};

thread_local! {
    static PARSER_CACHE: RefCell<HashMap<&'static str, Parser>> = RefCell::new(HashMap::new());
}

/// Extension-to-grammar mapping for the 16 supported native grammars.
fn grammar_for_ext(ext: &str) -> Option<(&'static str, Language)> {
    match ext {
        ".ts" | "ts" => Some((
            "typescript",
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        )),
        ".tsx" | "tsx" => Some(("tsx", tree_sitter_typescript::LANGUAGE_TSX.into())),
        ".js" | "js" | ".jsx" | "jsx" | ".mjs" | "mjs" | ".cjs" | "cjs" => {
            Some(("javascript", tree_sitter_javascript::LANGUAGE.into()))
        }
        ".py" | "py" => Some(("python", tree_sitter_python::LANGUAGE.into())),
        ".rs" | "rs" => Some(("rust", tree_sitter_rust::LANGUAGE.into())),
        ".go" | "go" => Some(("go", tree_sitter_go::LANGUAGE.into())),
        ".java" | "java" => Some(("java", tree_sitter_java::LANGUAGE.into())),
        ".c" | "c" | ".h" | "h" => Some(("c", tree_sitter_c::LANGUAGE.into())),
        ".cpp" | "cpp" | ".hpp" | "hpp" | ".cc" | "cc" => {
            Some(("cpp", tree_sitter_cpp::LANGUAGE.into()))
        }
        ".sh" | "sh" | ".bash" | "bash" | ".zsh" | "zsh" => {
            Some(("bash", tree_sitter_bash::LANGUAGE.into()))
        }
        ".rb" | "rb" => Some(("ruby", tree_sitter_ruby::LANGUAGE.into())),
        ".php" | "php" => Some(("php", tree_sitter_php::LANGUAGE_PHP.into())),
        ".cs" | "cs" => Some(("c_sharp", tree_sitter_c_sharp::LANGUAGE.into())),
        ".kt" | "kt" | ".kts" | "kts" => Some(("kotlin", tree_sitter_kotlin_ng::LANGUAGE.into())),
        ".html" | "html" | ".htm" | "htm" => Some(("html", tree_sitter_html::LANGUAGE.into())),
        ".css" | "css" => Some(("css", tree_sitter_css::LANGUAGE.into())),
        _ => None,
    }
}

/// AST node types that represent definitions, mapped to our symbol kinds.
fn definition_types(grammar_name: &str) -> HashMap<&'static str, &'static str> {
    match grammar_name {
        "typescript" | "tsx" => HashMap::from([
            ("function_declaration", "function"),
            ("method_definition", "method"),
            ("class_declaration", "class"),
            ("interface_declaration", "interface"),
            ("enum_declaration", "enum"),
            ("type_alias_declaration", "type"),
            ("lexical_declaration", "const"),
        ]),
        "javascript" => HashMap::from([
            ("function_declaration", "function"),
            ("method_definition", "method"),
            ("class_declaration", "class"),
            ("variable_declaration", "const"),
        ]),
        "python" => HashMap::from([
            ("function_definition", "function"),
            ("class_definition", "class"),
        ]),
        "rust" => HashMap::from([
            ("function_item", "function"),
            ("struct_item", "struct"),
            ("enum_item", "enum"),
            ("trait_item", "trait"),
            ("impl_item", "impl"),
            ("mod_item", "module"),
            ("macro_definition", "macro"),
        ]),
        "go" => HashMap::from([
            ("function_declaration", "function"),
            ("method_declaration", "method"),
            ("type_spec", "type"),
        ]),
        "java" => HashMap::from([
            ("method_declaration", "method"),
            ("class_declaration", "class"),
            ("interface_declaration", "interface"),
            ("enum_declaration", "enum"),
        ]),
        "c" => HashMap::from([
            ("function_definition", "function"),
            ("struct_specifier", "struct"),
            ("enum_specifier", "enum"),
        ]),
        "cpp" => HashMap::from([
            ("function_definition", "function"),
            ("class_specifier", "class"),
            ("struct_specifier", "struct"),
            ("enum_specifier", "enum"),
        ]),
        "bash" => HashMap::from([("function_definition", "function")]),
        "ruby" => HashMap::from([
            ("method", "function"),
            ("singleton_method", "function"),
            ("class", "class"),
            ("module", "module"),
        ]),
        "php" => HashMap::from([
            ("function_definition", "function"),
            ("method_declaration", "method"),
            ("class_declaration", "class"),
            ("interface_declaration", "interface"),
            ("enum_declaration", "enum"),
        ]),
        "c_sharp" => HashMap::from([
            ("method_declaration", "method"),
            ("class_declaration", "class"),
            ("interface_declaration", "interface"),
            ("enum_declaration", "enum"),
            ("struct_declaration", "struct"),
        ]),
        "kotlin" => HashMap::from([
            ("function_declaration", "function"),
            ("class_declaration", "class"),
            ("object_declaration", "class"),
        ]),
        "html" => HashMap::from([
            ("element", "element"),
            ("script_element", "script"),
            ("style_element", "style"),
        ]),
        "css" => HashMap::from([
            ("rule_set", "rule"),
            ("media_statement", "media"),
            ("keyframes_statement", "keyframes"),
        ]),
        _ => HashMap::new(),
    }
}

/// Extract the name of a symbol from an AST node.
fn extract_name<'a>(node: &Node<'a>, source: &'a [u8]) -> String {
    // Try field-based name extraction first
    if let Some(name_node) = node
        .child_by_field_name("name")
        .or_else(|| node.child_by_field_name("declarator"))
    {
        // Handle nested declarators (C/C++ function pointers)
        if (name_node.kind() == "function_declarator" || name_node.kind() == "pointer_declarator")
            && let Some(inner) = name_node
                .child_by_field_name("declarator")
                .or_else(|| name_node.named_child(0))
        {
            return inner.utf8_text(source).unwrap_or("anonymous").to_string();
        }
        return name_node
            .utf8_text(source)
            .unwrap_or("anonymous")
            .to_string();
    }

    // Try to find an identifier child
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i) {
            let kind = child.kind();
            if kind == "identifier"
                || kind == "type_identifier"
                || kind == "property_identifier"
                || kind == "simple_identifier"
            {
                return child.utf8_text(source).unwrap_or("anonymous").to_string();
            }
            // Handle variable_declarator / const_declaration
            if (kind == "variable_declarator" || kind == "const_declaration")
                && let Some(inner) = child.child_by_field_name("name")
            {
                return inner.utf8_text(source).unwrap_or("anonymous").to_string();
            }
        }
    }

    // Fallback: first word of the node text
    let text = node.utf8_text(source).unwrap_or("anonymous");
    text.split(|c: char| c.is_whitespace() || c == '(' || c == '{')
        .next()
        .unwrap_or("anonymous")
        .to_string()
}

/// Extract the first-line signature of a node, capped at 150 chars.
fn extract_signature<'a>(node: &Node<'a>, source: &'a [u8]) -> String {
    let text = node.utf8_text(source).unwrap_or("");
    let first_line = text.lines().next().unwrap_or("").trim();
    if first_line.len() > 150 {
        format!(
            "{}...",
            crate::core::parser::truncate_to_char_boundary(first_line, 150)
        )
    } else {
        first_line.to_string()
    }
}

/// Return `true` if `node` is a variable/lexical declaration whose binding
/// target is a destructure pattern (`object_pattern` or `array_pattern`).
///
/// Tree-sitter structure for `const { a, b } = rhs()`:
///   lexical_declaration
///     variable_declarator
///       name: object_pattern    ← destructure
///       value: call_expression
///
/// vs. `const foo = rhs()`:
///   lexical_declaration
///     variable_declarator
///       name: identifier        ← plain binding — keep emitting
///       value: call_expression
///
/// Skipping destructure-binding declarations prevents dead-code false
/// positives: the bound names appear later in the same file as *references*
/// (not definitions), and the cross-file heuristic should never flag them.
fn is_destructure_binding(node: &Node) -> bool {
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i)
            && (child.kind() == "variable_declarator" || child.kind() == "const_declaration")
            && let Some(name_node) = child.child_by_field_name("name")
        {
            let k = name_node.kind();
            if k == "object_pattern" || k == "array_pattern" {
                return true;
            }
        }
    }
    false
}

/// Walk the AST and collect symbols, returning them as a Vec.
fn collect_symbols(
    node: &Node,
    source: &[u8],
    def_types: &HashMap<&str, &str>,
    depth: usize,
    max_depth: usize,
) -> Vec<CodeSymbol> {
    if depth > max_depth {
        return Vec::new();
    }

    let mut results = Vec::new();

    if let Some(&kind) = def_types.get(node.kind()) {
        // Skip destructure-binding declarations — they are not true symbol
        // definitions and would produce false-positive dead-code candidates.
        if is_destructure_binding(node) {
            return results;
        }

        // This node is a definition — collect its children as nested symbols
        let mut children = Vec::new();
        for i in 0..node.named_child_count() {
            if let Some(child) = node.named_child(i) {
                children.extend(collect_symbols(
                    &child,
                    source,
                    def_types,
                    depth + 1,
                    max_depth,
                ));
            }
        }

        results.push(CodeSymbol {
            name: extract_name(node, source),
            kind: kind.to_string(),
            line: node.start_position().row + 1,
            end_line: node.end_position().row + 1,
            signature: Some(extract_signature(node, source)),
            children,
        });
    } else {
        // Not a definition node — recurse into children at same depth
        for i in 0..node.named_child_count() {
            if let Some(child) = node.named_child(i) {
                results.extend(collect_symbols(&child, source, def_types, depth, max_depth));
            }
        }
    }

    results
}

/// Parse source code with tree-sitter and extract symbols.
/// Uses `thread_local!` parser pool since Parser is `!Send` in v0.25.
pub fn parse_with_tree_sitter(content: &str, ext: &str) -> Result<Vec<CodeSymbol>> {
    let (grammar_name, language) = grammar_for_ext(ext)
        .ok_or_else(|| ContextPlusError::TreeSitter(format!("unsupported extension: {}", ext)))?;

    let def_types = definition_types(grammar_name);
    if def_types.is_empty() {
        return Ok(Vec::new());
    }

    PARSER_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        let parser = cache.entry(grammar_name).or_insert_with(|| {
            let mut p = Parser::new();
            // set_language on first creation; grammar_name is stable so this is always correct
            let _ = p.set_language(&language);
            p
        });
        parser.reset();

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| ContextPlusError::TreeSitter("parse returned None".to_string()))?;

        let root = tree.root_node();
        let source = content.as_bytes();
        let symbols = collect_symbols(&root, source, &def_types, 0, 3);
        Ok(symbols)
    })
}

/// Get the list of supported file extensions.
pub fn get_supported_extensions() -> &'static [&'static str] {
    &[
        ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".rs", ".go", ".java", ".c", ".h",
        ".cpp", ".hpp", ".cc", ".sh", ".bash", ".zsh", ".rb", ".php", ".cs", ".kt", ".kts",
        ".html", ".htm", ".css",
    ]
}

/// Return `true` when a TypeScript/TSX `import_statement` or `export_statement`
/// node represents a *type-only* declaration that is fully erased at compile
/// time and therefore creates no runtime dependency.
///
/// Covers the top-level form only:
/// - `import type { Foo } from './bar'`
/// - `export type { Foo } from './bar'`
///
/// Per-specifier type modifiers (`import { type Foo, Bar }`) are NOT
/// considered type-only at the statement level because the statement still
/// carries at least one runtime import (`Bar`). Those edges are correctly
/// retained.
fn is_type_only_ts_import(node: &Node<'_>) -> bool {
    // The `type` keyword, when present at the statement level, appears as a
    // non-named child immediately after the `import`/`export` keyword token.
    // Iterate all children (both named and anonymous) and look for a child
    // whose kind is the literal string "type" before the first named import
    // clause or namespace import.
    let child_count = node.child_count();
    for i in 0..child_count {
        if let Some(child) = node.child(i) {
            match child.kind() {
                // The `type` keyword signals a type-only import/export.
                "type" => return true,
                // Stop scanning once we hit the import clause — the `type`
                // keyword would have appeared before it.
                "import_clause" | "namespace_import" | "named_imports" | "export_clause"
                | "namespace_export" => return false,
                _ => {}
            }
        }
    }
    false
}

/// Recursively collect import paths from AST nodes.
///
/// Per-language match arms each guard on `node.kind()` via a nested `if`
/// rather than a match-arm guard. clippy::collapsible_match would suggest
/// folding the inner `if` into the arm — keep it explicit here so the per-
/// language blocks share the same shape and stay easy to extend.
#[allow(clippy::collapsible_match)]
fn collect_imports_from_node(
    node: Node,
    source: &[u8],
    grammar_name: &str,
    imports: &mut Vec<String>,
) {
    match grammar_name {
        "typescript" | "tsx" | "javascript" => {
            // ES import/export: import ... from 'path'; export ... from 'path'
            if node.kind() == "import_statement" || node.kind() == "export_statement" {
                // Skip type-only declarations: `import type { … }` and
                // `export type { … }` are erased at compile time and never
                // create a runtime dependency. Keeping them in the graph
                // produces false-positive cycles (e.g. the berries
                // subscription-domain 3-file SCC).
                if is_type_only_ts_import(&node) {
                    return;
                }
                if let Some(source_node) = node.child_by_field_name("source")
                    && let Ok(text) = source_node.utf8_text(source)
                {
                    let path = text.trim_matches(|c| c == '"' || c == '\'');
                    if !path.is_empty() {
                        imports.push(path.to_string());
                    }
                }
                return; // Don't recurse into import/export children
            }
            // CJS require('path')
            if node.kind() == "call_expression"
                && let Some(func) = node.child_by_field_name("function")
                && func.kind() == "identifier"
                && func.utf8_text(source).unwrap_or("") == "require"
                && let Some(args) = node.child_by_field_name("arguments")
                && let Some(first_arg) = args.named_child(0)
                && first_arg.kind() == "string"
                && let Ok(text) = first_arg.utf8_text(source)
            {
                let path = text.trim_matches(|c| c == '"' || c == '\'');
                if !path.is_empty() {
                    imports.push(path.to_string());
                }
            }
        }
        "python" => {
            // import foo; import foo.bar
            if node.kind() == "import_statement" {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(text) = name_node.utf8_text(source) {
                        imports.push(text.to_string());
                    }
                } else {
                    // Fallback: find dotted_name children
                    for i in 0..node.named_child_count() {
                        if let Some(child) = node.named_child(i)
                            && child.kind() == "dotted_name"
                            && let Ok(text) = child.utf8_text(source)
                        {
                            imports.push(text.to_string());
                        }
                    }
                }
                return;
            }
            // from foo import bar
            if node.kind() == "import_from_statement" {
                if let Some(module_node) = node.child_by_field_name("module_name")
                    && let Ok(text) = module_node.utf8_text(source)
                {
                    imports.push(text.to_string());
                }
                return;
            }
        }
        "go" => {
            // import "path" or import ( "path1"; "path2" )
            if node.kind() == "import_declaration" {
                for i in 0..node.named_child_count() {
                    if let Some(child) = node.named_child(i) {
                        if child.kind() == "import_spec" {
                            if let Some(path_node) = child.child_by_field_name("path")
                                && let Ok(text) = path_node.utf8_text(source)
                            {
                                let path = text.trim_matches('"');
                                if !path.is_empty() {
                                    imports.push(path.to_string());
                                }
                            }
                        } else if child.kind() == "import_spec_list" {
                            for j in 0..child.named_child_count() {
                                if let Some(spec) = child.named_child(j)
                                    && spec.kind() == "import_spec"
                                    && let Some(path_node) = spec.child_by_field_name("path")
                                    && let Ok(text) = path_node.utf8_text(source)
                                {
                                    let path = text.trim_matches('"');
                                    if !path.is_empty() {
                                        imports.push(path.to_string());
                                    }
                                }
                            }
                        } else if child.kind() == "interpreted_string_literal" {
                            // Single import without spec wrapper
                            if let Ok(text) = child.utf8_text(source) {
                                let path = text.trim_matches('"');
                                if !path.is_empty() {
                                    imports.push(path.to_string());
                                }
                            }
                        }
                    }
                }
                return;
            }
        }
        "rust" => {
            // use foo::bar::baz;
            if node.kind() == "use_declaration" {
                // Extract the full path text after "use " and before ";"
                if let Ok(text) = node.utf8_text(source) {
                    let trimmed = text.trim();
                    let path = trimmed
                        .strip_prefix("use ")
                        .unwrap_or(trimmed)
                        .trim_end_matches(';')
                        .trim();
                    if !path.is_empty() {
                        imports.push(path.to_string());
                    }
                }
                return;
            }
        }
        _ => {}
    }

    // Recurse into children
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i) {
            collect_imports_from_node(child, source, grammar_name, imports);
        }
    }
}

/// Regex fallback for extracting imports when tree-sitter parsing fails.
fn extract_imports_regex(content: &str) -> Vec<String> {
    let mut imports = Vec::new();

    // ES imports/exports: import/export ... from 'path'
    if let Ok(re) = Regex::new(r#"(?:import|export)\s+.*?from\s+['"]([^'"]+)['"]"#) {
        for cap in re.captures_iter(content) {
            imports.push(cap[1].to_string());
        }
    }

    // CJS: require('path')
    if let Ok(re) = Regex::new(r#"require\s*\(\s*['"]([^'"]+)['"]\s*\)"#) {
        for cap in re.captures_iter(content) {
            imports.push(cap[1].to_string());
        }
    }

    // Python: import foo / from foo import bar
    if let Ok(re) = Regex::new(r#"^\s*import\s+([\w.]+)"#) {
        for cap in re.captures_iter(content) {
            imports.push(cap[1].to_string());
        }
    }
    if let Ok(re) = Regex::new(r#"^\s*from\s+([\w.]+)\s+import"#) {
        for cap in re.captures_iter(content) {
            imports.push(cap[1].to_string());
        }
    }

    // Go: import "path" — skipped in regex fallback due to false positive risk
    // (quoted strings appear everywhere in Go, not just imports)

    // Rust: use foo::bar;
    if let Ok(re) = Regex::new(r#"^\s*use\s+([\w:]+(?:::\{[^}]+\})?)\s*;"#) {
        for cap in re.captures_iter(content) {
            imports.push(cap[1].to_string());
        }
    }

    imports
}

/// Extract import paths from a source file using tree-sitter.
/// Returns a list of raw import specifiers (e.g., "./billing-service", "@berries/lib-context", "fs/promises").
pub fn extract_imports(path: &Path) -> Vec<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e))
        .unwrap_or_default();

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let (grammar_name, language) = match grammar_for_ext(&ext) {
        Some(g) => g,
        None => return extract_imports_regex(&content),
    };

    let result = PARSER_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        let parser = cache.entry(grammar_name).or_insert_with(|| {
            let mut p = Parser::new();
            let _ = p.set_language(&language);
            p
        });
        parser.reset();

        parser.parse(&content, None).map(|tree| {
            let root = tree.root_node();
            let source = content.as_bytes();
            let mut imports = Vec::new();
            collect_imports_from_node(root, source, grammar_name, &mut imports);
            imports
        })
    });

    // Fall back to regex ONLY when tree-sitter failed to produce a parse tree
    // (result is None). An empty import list from a successful parse is
    // authoritative — falling back on empty would re-introduce type-only
    // imports that the tree-sitter pass correctly suppressed.
    match result {
        Some(imports) => imports,
        None => extract_imports_regex(&content),
    }
}

/// Extract import paths from source code string (for testing or when content is already loaded).
pub fn extract_imports_from_str(content: &str, ext: &str) -> Vec<String> {
    let (grammar_name, language) = match grammar_for_ext(ext) {
        Some(g) => g,
        None => return extract_imports_regex(content),
    };

    let result = PARSER_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        let parser = cache.entry(grammar_name).or_insert_with(|| {
            let mut p = Parser::new();
            let _ = p.set_language(&language);
            p
        });
        parser.reset();

        parser.parse(content, None).map(|tree| {
            let root = tree.root_node();
            let source = content.as_bytes();
            let mut imports = Vec::new();
            collect_imports_from_node(root, source, grammar_name, &mut imports);
            imports
        })
    });

    // Fall back to regex ONLY when tree-sitter failed to produce a parse tree
    // (result is None). An empty import list from a successful parse is
    // authoritative — falling back on empty would re-introduce type-only
    // imports that the tree-sitter pass correctly suppressed.
    match result {
        Some(imports) => imports,
        None => extract_imports_regex(content),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_typescript_function() {
        let code = r#"
export function greet(name: string): string {
    return `Hello, ${name}!`;
}
"#;
        let symbols = parse_with_tree_sitter(code, ".ts").unwrap();
        assert!(!symbols.is_empty());
        let func = &symbols[0];
        assert_eq!(func.name, "greet");
        assert_eq!(func.kind, "function");
        assert!(func.line > 0);
    }

    #[test]
    fn parse_typescript_class_with_methods() {
        let code = r#"
class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }

    subtract(a: number, b: number): number {
        return a - b;
    }
}
"#;
        let symbols = parse_with_tree_sitter(code, ".ts").unwrap();
        assert!(!symbols.is_empty());
        let class = &symbols[0];
        assert_eq!(class.name, "Calculator");
        assert_eq!(class.kind, "class");
        assert!(class.children.len() >= 2);
        assert_eq!(class.children[0].name, "add");
        assert_eq!(class.children[0].kind, "method");
    }

    #[test]
    fn parse_python() {
        let code = r#"
def hello(name):
    print(f"Hello, {name}!")

class MyClass:
    def my_method(self):
        pass
"#;
        let symbols = parse_with_tree_sitter(code, ".py").unwrap();
        assert!(symbols.len() >= 2);
        assert_eq!(symbols[0].name, "hello");
        assert_eq!(symbols[0].kind, "function");
        assert_eq!(symbols[1].name, "MyClass");
        assert_eq!(symbols[1].kind, "class");
    }

    #[test]
    fn parse_rust_code() {
        let code = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub struct Point {
    x: f64,
    y: f64,
}

pub enum Color {
    Red,
    Green,
    Blue,
}
"#;
        let symbols = parse_with_tree_sitter(code, ".rs").unwrap();
        assert!(symbols.len() >= 3);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"add"));
        assert!(names.contains(&"Point"));
        assert!(names.contains(&"Color"));
    }

    #[test]
    fn parse_go_code() {
        let code = r#"
package main

func Hello(name string) string {
    return "Hello, " + name
}

type Server struct {
    Port int
    Host string
}
"#;
        let symbols = parse_with_tree_sitter(code, ".go").unwrap();
        assert!(symbols.len() >= 2);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Hello"));
        // Go type_spec should capture Server
        assert!(names.contains(&"Server"));
    }

    #[test]
    fn parse_javascript() {
        let code = r#"
function processData(data) {
    return data.map(item => item * 2);
}

class DataStore {
    constructor() {
        this.items = [];
    }
}
"#;
        let symbols = parse_with_tree_sitter(code, ".js").unwrap();
        assert!(!symbols.is_empty());
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"processData"));
    }

    #[test]
    fn parse_c_code() {
        let code = r#"
int add(int a, int b) {
    return a + b;
}

struct Point {
    int x;
    int y;
};
"#;
        let symbols = parse_with_tree_sitter(code, ".c").unwrap();
        assert!(!symbols.is_empty());
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"add"));
    }

    #[test]
    fn parse_java_code() {
        let code = r#"
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"#;
        let symbols = parse_with_tree_sitter(code, ".java").unwrap();
        assert!(!symbols.is_empty());
        assert_eq!(symbols[0].name, "Calculator");
        assert_eq!(symbols[0].kind, "class");
    }

    #[test]
    fn parse_bash_code() {
        let code = r#"
greet() {
    echo "Hello, $1!"
}
"#;
        let symbols = parse_with_tree_sitter(code, ".sh").unwrap();
        assert!(!symbols.is_empty());
        assert_eq!(symbols[0].name, "greet");
        assert_eq!(symbols[0].kind, "function");
    }

    #[test]
    fn unsupported_extension_returns_error() {
        let result = parse_with_tree_sitter("some content", ".xyz");
        assert!(result.is_err());
    }

    #[test]
    fn supported_extensions_list() {
        let exts = get_supported_extensions();
        assert!(exts.contains(&".ts"));
        assert!(exts.contains(&".py"));
        assert!(exts.contains(&".rs"));
        assert!(exts.contains(&".go"));
        assert!(exts.contains(&".sh"));
    }

    #[test]
    fn signature_extraction() {
        let code = "export function veryLongFunctionName(param1: string, param2: number, param3: boolean): Promise<Result> {\n    return doStuff();\n}\n";
        let symbols = parse_with_tree_sitter(code, ".ts").unwrap();
        assert!(!symbols.is_empty());
        let sig = symbols[0].signature.as_ref().unwrap();
        assert!(sig.contains("veryLongFunctionName"));
    }

    #[test]
    fn parse_cpp_code() {
        let code = r#"
class Shape {
public:
    virtual double area() const = 0;
};

struct Rectangle {
    double width;
    double height;
};
"#;
        let symbols = parse_with_tree_sitter(code, ".cpp").unwrap();
        assert!(!symbols.is_empty());
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Shape"));
    }

    #[test]
    fn parse_ruby_code() {
        let code = "class Calculator\n  def add(a, b)\n    a + b\n  end\nend\n\nmodule Helpers\n  def greet(name)\n    name\n  end\nend\n";
        let symbols = parse_with_tree_sitter(code, ".rb").unwrap();
        assert!(!symbols.is_empty());
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Calculator"));
        assert!(names.contains(&"Helpers"));
    }

    #[test]
    fn parse_php_code() {
        let code = "<?php\nclass UserService {\n    public function getUser() {\n        return 1;\n    }\n}\n\nfunction helper() {\n    return 1;\n}\n";
        let symbols = parse_with_tree_sitter(code, ".php").unwrap();
        assert!(!symbols.is_empty());
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"UserService"));
        assert!(names.contains(&"helper"));
    }

    #[test]
    fn parse_csharp_code() {
        let code = "public class Calculator {\n    public int Add(int a, int b) {\n        return a + b;\n    }\n}\n\npublic interface IService {\n    void Execute();\n}\n\npublic enum Color {\n    Red,\n    Green,\n    Blue\n}\n";
        let symbols = parse_with_tree_sitter(code, ".cs").unwrap();
        assert!(!symbols.is_empty());
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Calculator"));
        assert!(names.contains(&"IService"));
        assert!(names.contains(&"Color"));
    }

    #[test]
    fn parse_kotlin_code() {
        let code = "class Calculator {\n    fun add(a: Int, b: Int): Int {\n        return a + b\n    }\n}\n\nfun topLevel(): String {\n    return \\\"hello\\\"\n}\n";
        let symbols = parse_with_tree_sitter(code, ".kt").unwrap();
        assert!(!symbols.is_empty());
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Calculator"));
        // top-level functions may use a different node type in kotlin-ng grammar
    }

    #[test]
    fn parse_html_code() {
        let code = "<html><head><title>Test</title></head><body><p>Hello</p></body></html>";
        let result = parse_with_tree_sitter(code, ".html");
        assert!(result.is_ok());
    }

    #[test]
    fn parse_css_code() {
        let code = ".container {\n    display: flex;\n}\n";
        let symbols = parse_with_tree_sitter(code, ".css").unwrap();
        assert!(!symbols.is_empty());
    }

    #[test]
    fn supported_extensions_includes_new_grammars() {
        let exts = get_supported_extensions();
        assert!(exts.contains(&".rb"));
        assert!(exts.contains(&".php"));
        assert!(exts.contains(&".cs"));
        assert!(exts.contains(&".kt"));
        assert!(exts.contains(&".kts"));
        assert!(exts.contains(&".html"));
        assert!(exts.contains(&".css"));
    }

    #[test]
    fn extension_mapping_completeness() {
        for ext in get_supported_extensions() {
            assert!(
                grammar_for_ext(ext).is_some(),
                "Extension {} is in supported list but has no grammar mapping",
                ext
            );
        }
    }

    // --- Import extraction tests ---

    #[test]
    fn extract_imports_typescript_es_imports() {
        let code = r#"
import { BillingService } from './billing-service';
import * as context from '@berries/lib-context';
import fs from 'fs/promises';
import type { Config } from '../config';

export function doStuff() {}
"#;
        let imports = extract_imports_from_str(code, ".ts");
        assert!(imports.contains(&"./billing-service".to_string()));
        assert!(imports.contains(&"@berries/lib-context".to_string()));
        assert!(imports.contains(&"fs/promises".to_string()));
        // `import type { … }` is erased at compile time — it must NOT appear
        // in the runtime import list so it cannot form false dependency cycles.
        assert!(
            !imports.contains(&"../config".to_string()),
            "type-only import must be excluded from runtime imports"
        );
    }

    #[test]
    fn extract_imports_typescript_type_only_excluded() {
        // All three forms of type-only imports must be dropped.
        let code = r#"
import type { Foo } from './foo';
import type * as bar from './bar';
export type { Baz } from './baz';
"#;
        let imports = extract_imports_from_str(code, ".ts");
        assert!(
            imports.is_empty(),
            "all type-only imports/re-exports must be excluded; got: {:?}",
            imports
        );
    }

    #[test]
    fn extract_imports_typescript_mixed_keeps_runtime_edge() {
        // `import { type Foo, Bar }` — only Foo is type-only, Bar is runtime.
        // The whole statement still carries a runtime edge and must be retained.
        let code = "import { type Foo, Bar } from './mixed';\n";
        let imports = extract_imports_from_str(code, ".ts");
        assert!(
            imports.contains(&"./mixed".to_string()),
            "mixed import (type + runtime specifiers) must retain the edge; got: {:?}",
            imports
        );
    }

    #[test]
    fn extract_imports_typescript_require() {
        let code = r#"
const path = require('path');
const { readFile } = require('fs/promises');
"#;
        let imports = extract_imports_from_str(code, ".js");
        assert!(imports.contains(&"path".to_string()));
        assert!(imports.contains(&"fs/promises".to_string()));
    }

    #[test]
    fn extract_imports_typescript_reexports() {
        let code = r#"
export { default } from './component';
export * from './types';
"#;
        let imports = extract_imports_from_str(code, ".ts");
        assert!(imports.contains(&"./component".to_string()));
        assert!(imports.contains(&"./types".to_string()));
    }

    #[test]
    fn extract_imports_python() {
        let code = r#"
import os
import json
from pathlib import Path
from collections.abc import Mapping

def main():
    pass
"#;
        let imports = extract_imports_from_str(code, ".py");
        assert!(imports.contains(&"os".to_string()));
        assert!(imports.contains(&"json".to_string()));
        assert!(imports.contains(&"pathlib".to_string()));
        assert!(imports.contains(&"collections.abc".to_string()));
    }

    #[test]
    fn extract_imports_go() {
        let code = r#"
package main

import (
    "fmt"
    "net/http"
    "github.com/gofiber/fiber/v2"
)

func main() {}
"#;
        let imports = extract_imports_from_str(code, ".go");
        assert!(imports.contains(&"fmt".to_string()));
        assert!(imports.contains(&"net/http".to_string()));
        assert!(imports.contains(&"github.com/gofiber/fiber/v2".to_string()));
    }

    #[test]
    fn extract_imports_rust() {
        let code = r#"
use std::collections::HashMap;
use std::path::Path;
use crate::core::parser::CodeSymbol;

fn main() {}
"#;
        let imports = extract_imports_from_str(code, ".rs");
        assert!(imports.contains(&"std::collections::HashMap".to_string()));
        assert!(imports.contains(&"std::path::Path".to_string()));
        assert!(imports.contains(&"crate::core::parser::CodeSymbol".to_string()));
    }

    #[test]
    fn extract_imports_file_based() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("contextplus_import_test");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("test_imports.ts");
        let mut f = std::fs::File::create(&file_path).unwrap();
        write!(
            f,
            "import {{ foo }} from './foo';\nimport bar from 'bar-pkg';\n"
        )
        .unwrap();
        drop(f);

        let imports = extract_imports(&file_path);
        assert!(imports.contains(&"./foo".to_string()));
        assert!(imports.contains(&"bar-pkg".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_imports_nonexistent_file_returns_empty() {
        let imports = extract_imports(Path::new("/nonexistent/file.ts"));
        assert!(imports.is_empty());
    }

    #[test]
    fn extract_imports_unsupported_ext_uses_regex_fallback() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("contextplus_import_fallback_test");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("test.svelte");
        let mut f = std::fs::File::create(&file_path).unwrap();
        writeln!(f, "import {{ onMount }} from 'svelte';").unwrap();
        drop(f);

        let imports = extract_imports(&file_path);
        assert!(imports.contains(&"svelte".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_imports_empty_file_returns_empty() {
        let imports = extract_imports_from_str("", ".ts");
        assert!(imports.is_empty());
    }

    #[test]
    fn extract_imports_no_imports_returns_empty() {
        let code = "export function hello() { return 'world'; }\n";
        let imports = extract_imports_from_str(code, ".ts");
        assert!(imports.is_empty());
    }

    // --- Destructure-binding LHS fix ---

    /// Top-level `const foo = bar()` MUST appear in the symbol list.
    /// Destructure-binding LHS identifiers (`const { x, y } = z()`) MUST NOT.
    ///
    /// This guards against the false-positive dead-code reports described in
    /// https://github.com/mrsufgi/contextplus-rs/issues — where names like
    /// `redis` and `sessionStore` from `const { redis, sessionStore } = makeStores()`
    /// were flagged as dead symbols even though they were used inline.
    #[test]
    fn const_plain_binding_emitted_destructure_lhs_skipped() {
        let code = r#"
const foo = makeFoo();
const { x, y } = makePoint();
const [a, b] = makePair();
"#;
        let symbols = parse_with_tree_sitter(code, ".ts").unwrap();
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();

        // Plain binding should still be collected
        assert!(
            names.contains(&"foo"),
            "expected 'foo' in symbols, got: {names:?}"
        );

        // Destructure LHS names must NOT be emitted as top-level symbols —
        // neither as bare identifiers nor as raw object-pattern text like "{ x, y }".
        // The whole lexical_declaration for a destructure should be skipped entirely.
        assert!(
            !names.contains(&"x"),
            "destructure LHS 'x' must not appear in symbols, got: {names:?}"
        );
        assert!(
            !names.contains(&"y"),
            "destructure LHS 'y' must not appear in symbols, got: {names:?}"
        );
        assert!(
            !names.contains(&"a"),
            "array destructure LHS 'a' must not appear in symbols, got: {names:?}"
        );
        assert!(
            !names.contains(&"b"),
            "array destructure LHS 'b' must not appear in symbols, got: {names:?}"
        );

        // Also verify no symbol with a name starting with '{' or '[' sneaks through
        // (which would be the raw object_pattern / array_pattern text)
        for name in &names {
            assert!(
                !name.starts_with('{') && !name.starts_with('['),
                "raw destructure pattern text leaked as symbol name: {name:?}"
            );
        }
    }
}
