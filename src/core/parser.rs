use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

/// FNV-1a 64-bit hash for cache invalidation.
/// Zero dependencies, ~2 ops per byte, 50% collision probability at ~4 billion entries.
/// Replaces the original 32-bit JS djb2 hash which had collision risk at ~77K entries.
/// Output is hex-encoded for compact, URL-safe cache keys.
pub fn hash_content(text: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;

    let mut hash = FNV_OFFSET;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{:016x}", hash)
}

/// Truncate a string to at most `max_bytes` bytes, but never split a UTF-8 codepoint.
pub fn truncate_to_char_boundary(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    // Find the last char boundary at or before max_bytes
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[derive(Debug, Clone)]
pub struct CodeSymbol {
    pub name: String,
    pub kind: String,
    pub line: usize,
    pub end_line: usize,
    pub signature: Option<String>,
    pub children: Vec<CodeSymbol>,
}

#[derive(Debug, Clone)]
pub struct SymbolLocation {
    pub name: String,
    pub kind: String,
    pub line: usize,
    pub end_line: usize,
    pub signature: Option<String>,
    pub parent_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FileAnalysis {
    pub path: String,
    pub symbols: Vec<CodeSymbol>,
    pub header: String,
    pub language: String,
    pub line_count: usize,
}

/// Extract a human-readable header from the first few lines of a file.
/// Strips comment markers and skips imports/use statements.
/// Accepts a `&str` directly — callers no longer need to `.lines().collect()`.
pub fn extract_header(content: &str) -> String {
    let mut header_lines: Vec<String> = Vec::new();
    for line in content.lines().take(10) {
        let stripped = line
            .trim_start_matches("//")
            .trim_start_matches('#')
            .trim_start_matches("--")
            .trim_start_matches('*')
            .trim_start_matches("/**")
            .trim_end_matches("*/")
            .trim();
        if stripped.is_empty()
            || stripped.starts_with('!')
            || stripped.starts_with("use ")
            || stripped.starts_with("import ")
        {
            continue;
        }
        header_lines.push(stripped.to_string());
        if header_lines.len() >= 2 {
            break;
        }
    }
    header_lines.join(" | ")
}

/// Detect language from file extension.
/// Uses `eq_ignore_ascii_case` to avoid heap-allocating a lowercase String per call.
pub fn detect_language(file_path: &str) -> Option<&'static str> {
    let ext = Path::new(file_path).extension().and_then(|e| e.to_str())?;
    if ext.eq_ignore_ascii_case("ts") || ext.eq_ignore_ascii_case("tsx") {
        Some("typescript")
    } else if ext.eq_ignore_ascii_case("js")
        || ext.eq_ignore_ascii_case("jsx")
        || ext.eq_ignore_ascii_case("mjs")
    {
        Some("javascript")
    } else if ext.eq_ignore_ascii_case("py") {
        Some("python")
    } else if ext.eq_ignore_ascii_case("rs") {
        Some("rust")
    } else if ext.eq_ignore_ascii_case("go") {
        Some("go")
    } else if ext.eq_ignore_ascii_case("java") {
        Some("java")
    } else if ext.eq_ignore_ascii_case("cs") {
        Some("csharp")
    } else if ext.eq_ignore_ascii_case("c") || ext.eq_ignore_ascii_case("h") {
        Some("c")
    } else if ext.eq_ignore_ascii_case("cpp")
        || ext.eq_ignore_ascii_case("hpp")
        || ext.eq_ignore_ascii_case("cc")
    {
        Some("cpp")
    } else if ext.eq_ignore_ascii_case("rb") {
        Some("ruby")
    } else if ext.eq_ignore_ascii_case("swift") {
        Some("swift")
    } else if ext.eq_ignore_ascii_case("kt") || ext.eq_ignore_ascii_case("kts") {
        Some("kotlin")
    } else if ext.eq_ignore_ascii_case("lua") {
        Some("lua")
    } else if ext.eq_ignore_ascii_case("zig") {
        Some("zig")
    } else if ext.eq_ignore_ascii_case("sh")
        || ext.eq_ignore_ascii_case("bash")
        || ext.eq_ignore_ascii_case("zsh")
    {
        Some("bash")
    } else if ext.eq_ignore_ascii_case("php") {
        Some("php")
    } else if ext.eq_ignore_ascii_case("html") || ext.eq_ignore_ascii_case("htm") {
        Some("html")
    } else if ext.eq_ignore_ascii_case("css") {
        Some("css")
    } else if ext.eq_ignore_ascii_case("toml") {
        Some("toml")
    } else if ext.eq_ignore_ascii_case("yaml") || ext.eq_ignore_ascii_case("yml") {
        Some("yaml")
    } else {
        None
    }
}

/// Check if a file path is a supported source code file.
pub fn is_supported_file(file_path: &str) -> bool {
    detect_language(file_path).is_some()
}

/// Flatten nested symbols into a flat list with parent info.
/// Uses an accumulator parameter to avoid intermediate Vec allocations per recursion level.
pub fn flatten_symbols(symbols: &[CodeSymbol], parent_name: Option<&str>) -> Vec<SymbolLocation> {
    let mut result = Vec::new();
    flatten_symbols_into(symbols, parent_name, &mut result);
    result
}

fn flatten_symbols_into(
    symbols: &[CodeSymbol],
    parent_name: Option<&str>,
    out: &mut Vec<SymbolLocation>,
) {
    for sym in symbols {
        out.push(SymbolLocation {
            name: sym.name.clone(),
            kind: sym.kind.clone(),
            line: sym.line,
            end_line: sym.end_line,
            signature: sym.signature.clone(),
            parent_name: parent_name.map(|s| s.to_string()),
        });
        if !sym.children.is_empty() {
            flatten_symbols_into(&sym.children, Some(&sym.name), out);
        }
    }
}

/// Format a symbol for display.
pub fn format_symbol(sym: &CodeSymbol, indent: usize) -> String {
    let prefix = "  ".repeat(indent);
    let kind_label = if sym.kind == "method" {
        "method"
    } else {
        &sym.kind
    };
    let line_label = if sym.end_line > sym.line {
        format!("L{}-L{}", sym.line, sym.end_line)
    } else {
        format!("L{}", sym.line)
    };

    let mut result = if sym.kind == "function" || sym.kind == "method" {
        format!(
            "{}{}: {} ({})",
            prefix,
            kind_label,
            sym.signature.as_deref().unwrap_or(&sym.name),
            line_label
        )
    } else {
        format!("{}{}: {} ({})", prefix, kind_label, sym.name, line_label)
    };

    for child in &sym.children {
        result.push('\n');
        result.push_str(&format_symbol(child, indent + 1));
    }
    result
}

// --- Regex fallback parser ---
// When tree-sitter has no grammar for a language (or parsing fails), we fall
// back to line-by-line regex matching to extract top-level symbols.

/// Generic regex patterns that match common definition keywords across many languages.
static GENERIC_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        (
            Regex::new(r"^(?:pub\s+)?(?:export\s+)?(?:async\s+)?(?:fn|func|function|def)\s+(\w+)")
                .unwrap(),
            "function",
        ),
        (
            Regex::new(r"^(?:pub\s+)?(?:export\s+)?(?:abstract\s+)?(?:class|struct)\s+(\w+)")
                .unwrap(),
            "class",
        ),
        (
            Regex::new(r"^(?:pub\s+)?(?:export\s+)?(?:enum|interface|type|trait|protocol)\s+(\w+)")
                .unwrap(),
            "enum",
        ),
    ]
});

/// Find the end of a brace-delimited block starting from `start_idx`.
fn find_brace_block_end(lines: &[&str], start_idx: usize) -> usize {
    let mut depth: i32 = 0;
    let mut seen_opening = false;

    for (i, line) in lines.iter().enumerate().skip(start_idx) {
        for ch in line.chars() {
            if ch == '{' {
                depth += 1;
                seen_opening = true;
            } else if ch == '}' && seen_opening {
                depth -= 1;
                if depth <= 0 {
                    return i + 1; // 1-indexed end line
                }
            }
        }
    }
    start_idx + 1
}

/// Parse source code with regex fallback, extracting top-level symbols.
/// This is used when tree-sitter does not support the language or fails.
pub fn parse_with_regex_fallback(content: &str) -> Vec<CodeSymbol> {
    let lines: Vec<&str> = content.lines().collect();
    let mut symbols = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        for (pattern, kind) in GENERIC_PATTERNS.iter() {
            if let Some(caps) = pattern.captures(trimmed)
                && let Some(name_match) = caps.get(1)
            {
                let name = name_match.as_str().to_string();
                let end_line = find_brace_block_end(&lines, i);
                let signature = trimmed
                    .trim_end_matches(|c: char| c == '{' || c.is_whitespace())
                    .to_string();
                symbols.push(CodeSymbol {
                    name,
                    kind: kind.to_string(),
                    line: i + 1,
                    end_line,
                    signature: Some(if signature.len() > 150 {
                        format!("{}...", truncate_to_char_boundary(&signature, 150))
                    } else {
                        signature
                    }),
                    children: Vec::new(),
                });
                break; // first match wins for this line
            }
        }
    }
    symbols
}

#[cfg(test)]
mod tests {
    use super::*;

    // FNV-1a 64-bit hash tests — property-based (not cross-language reference values)
    #[test]
    fn hash_content_deterministic() {
        assert_eq!(hash_content("hello"), hash_content("hello"));
        assert_eq!(hash_content(""), hash_content(""));
    }

    #[test]
    fn hash_content_different_inputs_differ() {
        assert_ne!(hash_content("hello"), hash_content("world"));
        assert_ne!(hash_content("abc"), hash_content("abd"));
        assert_ne!(hash_content(""), hash_content("a"));
    }

    #[test]
    fn hash_content_is_16_hex_chars() {
        let h = hash_content("test");
        assert_eq!(
            h.len(),
            16,
            "FNV-1a 64-bit should produce 16 hex chars, got: {}",
            h
        );
        assert!(
            h.chars().all(|c| c.is_ascii_hexdigit()),
            "non-hex char in: {}",
            h
        );
    }

    #[test]
    fn hash_content_empty_is_fnv_offset() {
        // FNV-1a of empty input is the offset basis: 0xcbf29ce484222325
        assert_eq!(hash_content(""), "cbf29ce484222325");
    }

    #[test]
    fn extract_header_basic() {
        let content = "// Multi-language symbol extraction with tree-sitter AST\n// Supports 36 languages via WASM grammars\nimport { readFile } from 'fs/promises';";
        let header = extract_header(content);
        assert_eq!(
            header,
            "Multi-language symbol extraction with tree-sitter AST | Supports 36 languages via WASM grammars"
        );
    }

    #[test]
    fn extract_header_skips_imports() {
        let content = "import something from 'module';\nuse std::io;\n// Actual header line";
        let header = extract_header(content);
        assert_eq!(header, "Actual header line");
    }

    #[test]
    fn extract_header_empty_input() {
        assert_eq!(extract_header(""), "");
    }

    #[test]
    fn detect_language_works() {
        assert_eq!(detect_language("foo.ts"), Some("typescript"));
        assert_eq!(detect_language("foo.tsx"), Some("typescript"));
        assert_eq!(detect_language("foo.js"), Some("javascript"));
        assert_eq!(detect_language("foo.py"), Some("python"));
        assert_eq!(detect_language("foo.rs"), Some("rust"));
        assert_eq!(detect_language("foo.go"), Some("go"));
        assert_eq!(detect_language("foo.java"), Some("java"));
        assert_eq!(detect_language("foo.c"), Some("c"));
        assert_eq!(detect_language("foo.cpp"), Some("cpp"));
        assert_eq!(detect_language("foo.sh"), Some("bash"));
        assert_eq!(detect_language("foo.rb"), Some("ruby"));
        assert_eq!(detect_language("foo.php"), Some("php"));
        assert_eq!(detect_language("foo.cs"), Some("csharp"));
        assert_eq!(detect_language("foo.kt"), Some("kotlin"));
        assert_eq!(detect_language("foo.kts"), Some("kotlin"));
        assert_eq!(detect_language("foo.html"), Some("html"));
        assert_eq!(detect_language("foo.css"), Some("css"));
        assert_eq!(detect_language("foo.toml"), Some("toml"));
        assert_eq!(detect_language("foo.yaml"), Some("yaml"));
        assert_eq!(detect_language("foo.yml"), Some("yaml"));
        assert_eq!(detect_language("foo.unknown"), None);
        assert_eq!(detect_language("noext"), None);
    }

    #[test]
    fn is_supported_file_works() {
        assert!(is_supported_file("main.rs"));
        assert!(is_supported_file("app.ts"));
        assert!(is_supported_file("style.css"));
        assert!(is_supported_file("config.toml"));
        assert!(!is_supported_file("readme.md"));
        assert!(!is_supported_file("data.json"));
    }

    #[test]
    fn flatten_symbols_works() {
        let syms = vec![CodeSymbol {
            name: "MyClass".to_string(),
            kind: "class".to_string(),
            line: 1,
            end_line: 20,
            signature: None,
            children: vec![CodeSymbol {
                name: "myMethod".to_string(),
                kind: "method".to_string(),
                line: 5,
                end_line: 10,
                signature: Some("myMethod(a: number)".to_string()),
                children: vec![],
            }],
        }];
        let flat = flatten_symbols(&syms, None);
        assert_eq!(flat.len(), 2);
        assert_eq!(flat[0].name, "MyClass");
        assert!(flat[0].parent_name.is_none());
        assert_eq!(flat[1].name, "myMethod");
        assert_eq!(flat[1].parent_name.as_deref(), Some("MyClass"));
    }

    #[test]
    fn format_symbol_basic() {
        let sym = CodeSymbol {
            name: "doStuff".to_string(),
            kind: "function".to_string(),
            line: 10,
            end_line: 20,
            signature: Some("fn doStuff(x: i32) -> i32".to_string()),
            children: vec![],
        };
        let output = format_symbol(&sym, 0);
        assert!(output.contains("function: fn doStuff(x: i32) -> i32 (L10-L20)"));
    }

    // --- Regex fallback tests ---

    #[test]
    fn regex_fallback_extracts_functions() {
        let code = "function greet(name) {\n    console.log(name);\n}\n\ndef process(data):\n    return data\n\nfn compute(x: i32) -> i32 {\n    x * 2\n}\n";
        let symbols = parse_with_regex_fallback(code);
        assert!(symbols.len() >= 3);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"greet"));
        assert!(names.contains(&"process"));
        assert!(names.contains(&"compute"));
        assert!(symbols.iter().all(|s| s.kind == "function"));
    }

    #[test]
    fn regex_fallback_extracts_classes_and_structs() {
        let code = "class MyClass {\n    constructor() {}\n}\n\nstruct Point {\n    x: f64,\n    y: f64,\n}\n";
        let symbols = parse_with_regex_fallback(code);
        assert!(symbols.len() >= 2);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"MyClass"));
        assert!(names.contains(&"Point"));
        assert!(symbols.iter().all(|s| s.kind == "class"));
    }

    #[test]
    fn regex_fallback_extracts_enums_interfaces_traits() {
        let code = "enum Color {\n    Red,\n    Green,\n}\n\ninterface Drawable {\n    draw(): void;\n}\n\ntrait Display {\n    fn fmt(&self) -> String;\n}\n";
        let symbols = parse_with_regex_fallback(code);
        assert!(symbols.len() >= 3);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Color"));
        assert!(names.contains(&"Drawable"));
        assert!(names.contains(&"Display"));
    }

    #[test]
    fn regex_fallback_handles_empty_content() {
        let symbols = parse_with_regex_fallback("");
        assert!(symbols.is_empty());
    }

    #[test]
    fn regex_fallback_handles_no_symbols() {
        let code = "// just a comment\nlet x = 5;\nprint('hello');\n";
        let symbols = parse_with_regex_fallback(code);
        assert!(symbols.is_empty());
    }

    #[test]
    fn regex_fallback_swift_like_code() {
        let code = "class ViewController {\n    func viewDidLoad() {\n        super.viewDidLoad()\n    }\n}\n\nstruct Point {\n    var x: Double\n    var y: Double\n}\n";
        let symbols = parse_with_regex_fallback(code);
        assert!(symbols.len() >= 2);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"ViewController"));
        assert!(names.contains(&"Point"));
    }

    #[test]
    fn regex_fallback_sets_line_numbers() {
        let code = "function hello() {\n    return 1;\n}\n";
        let symbols = parse_with_regex_fallback(code);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].line, 1);
        assert_eq!(symbols[0].end_line, 3);
    }

    #[test]
    fn regex_fallback_extracts_signature() {
        let code = "export async function fetchData(url: string) {\n    return fetch(url);\n}\n";
        let symbols = parse_with_regex_fallback(code);
        assert_eq!(symbols.len(), 1);
        let sig = symbols[0].signature.as_ref().unwrap();
        assert!(sig.contains("fetchData"));
        assert!(!sig.ends_with('{'));
    }
}
