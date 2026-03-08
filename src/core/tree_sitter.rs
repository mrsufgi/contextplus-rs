use std::collections::HashMap;

use tree_sitter::{Language, Node, Parser};

use crate::core::parser::CodeSymbol;
use crate::error::{ContextPlusError, Result};

/// Extension-to-grammar mapping for the 10 supported native grammars.
fn grammar_for_ext(ext: &str) -> Option<(&'static str, Language)> {
    match ext {
        ".ts" => Some(("typescript", tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())),
        ".tsx" => Some(("tsx", tree_sitter_typescript::LANGUAGE_TSX.into())),
        ".js" | ".jsx" | ".mjs" | ".cjs" => {
            Some(("javascript", tree_sitter_javascript::LANGUAGE.into()))
        }
        ".py" => Some(("python", tree_sitter_python::LANGUAGE.into())),
        ".rs" => Some(("rust", tree_sitter_rust::LANGUAGE.into())),
        ".go" => Some(("go", tree_sitter_go::LANGUAGE.into())),
        ".java" => Some(("java", tree_sitter_java::LANGUAGE.into())),
        ".c" | ".h" => Some(("c", tree_sitter_c::LANGUAGE.into())),
        ".cpp" | ".hpp" | ".cc" => Some(("cpp", tree_sitter_cpp::LANGUAGE.into())),
        ".sh" | ".bash" | ".zsh" => Some(("bash", tree_sitter_bash::LANGUAGE.into())),
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
            ("impl_item", "class"),
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
                && let Some(inner) = child.child_by_field_name("name") {
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
        format!("{}...", crate::core::parser::truncate_to_char_boundary(first_line, 150))
    } else {
        first_line.to_string()
    }
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
        // This node is a definition — collect its children as nested symbols
        let mut children = Vec::new();
        for i in 0..node.named_child_count() {
            if let Some(child) = node.named_child(i) {
                children.extend(collect_symbols(&child, source, def_types, depth + 1, max_depth));
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

    thread_local! {
        static PARSER: std::cell::RefCell<Parser> = std::cell::RefCell::new(Parser::new());
    }

    PARSER.with(|parser_cell| {
        let mut parser = parser_cell.borrow_mut();
        parser.reset();
        parser
            .set_language(&language)
            .map_err(|e| ContextPlusError::TreeSitter(format!("set_language failed: {}", e)))?;

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
        ".cpp", ".hpp", ".cc", ".sh", ".bash", ".zsh",
    ]
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
}
