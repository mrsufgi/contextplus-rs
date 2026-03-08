use std::path::Path;

/// Java-style string hash, ported exactly from TypeScript.
/// `hash = ((hash << 5) - hash) + charCode | 0` then base-36 encode.
pub fn hash_content(text: &str) -> String {
    let mut h: i32 = 0;
    for byte in text.encode_utf16() {
        h = h.wrapping_shl(5).wrapping_sub(h).wrapping_add(byte as i32);
        // `| 0` in JS forces 32-bit signed integer — wrapping arithmetic does this in Rust
    }
    if h == 0 {
        return "0".to_string();
    }
    // JS Number.prototype.toString(36) for negative numbers produces "-" + abs.toString(36)
    if h < 0 {
        format!("-{}", radix36(h.unsigned_abs()))
    } else {
        radix36(h as u32)
    }
}

fn radix36(mut n: u32) -> String {
    if n == 0 {
        return "0".to_string();
    }
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut buf = Vec::with_capacity(8);
    while n > 0 {
        buf.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    buf.reverse();
    String::from_utf8(buf).unwrap()
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
pub fn extract_header(lines: &[&str]) -> String {
    let mut header_lines: Vec<String> = Vec::new();
    for line in lines.iter().take(10) {
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
pub fn detect_language(file_path: &str) -> Option<&'static str> {
    let ext = Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    match ext.as_deref() {
        Some("ts") | Some("tsx") => Some("typescript"),
        Some("js") | Some("jsx") | Some("mjs") => Some("javascript"),
        Some("py") => Some("python"),
        Some("rs") => Some("rust"),
        Some("go") => Some("go"),
        Some("java") => Some("java"),
        Some("cs") => Some("csharp"),
        Some("c") | Some("h") => Some("c"),
        Some("cpp") | Some("hpp") | Some("cc") => Some("cpp"),
        Some("rb") => Some("ruby"),
        Some("swift") => Some("swift"),
        Some("kt") => Some("kotlin"),
        Some("lua") => Some("lua"),
        Some("zig") => Some("zig"),
        Some("sh") | Some("bash") | Some("zsh") => Some("bash"),
        _ => None,
    }
}

/// Check if a file path is a supported source code file.
pub fn is_supported_file(file_path: &str) -> bool {
    detect_language(file_path).is_some()
}

/// Flatten nested symbols into a flat list with parent info.
pub fn flatten_symbols(
    symbols: &[CodeSymbol],
    parent_name: Option<&str>,
) -> Vec<SymbolLocation> {
    let mut result = Vec::new();
    for sym in symbols {
        result.push(SymbolLocation {
            name: sym.name.clone(),
            kind: sym.kind.clone(),
            line: sym.line,
            end_line: sym.end_line,
            signature: sym.signature.clone(),
            parent_name: parent_name.map(|s| s.to_string()),
        });
        if !sym.children.is_empty() {
            result.extend(flatten_symbols(&sym.children, Some(&sym.name)));
        }
    }
    result
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

#[cfg(test)]
mod tests {
    use super::*;

    // Cross-language hash verification: values computed from the TypeScript reference
    #[test]
    fn hash_content_hello() {
        assert_eq!(hash_content("hello"), "1n1e4y");
    }

    #[test]
    fn hash_content_empty() {
        assert_eq!(hash_content(""), "0");
    }

    #[test]
    fn hash_content_abc() {
        assert_eq!(hash_content("abc"), "22ci");
    }

    #[test]
    fn hash_content_unicode() {
        assert_eq!(hash_content("Hello World!"), "-g0z65f");
    }

    #[test]
    fn hash_content_long_string() {
        assert_eq!(
            hash_content("the quick brown fox jumps over the lazy dog"),
            "-yg2171"
        );
    }

    #[test]
    fn extract_header_basic() {
        let lines = vec![
            "// Multi-language symbol extraction with tree-sitter AST",
            "// Supports 36 languages via WASM grammars",
            "import { readFile } from 'fs/promises';",
        ];
        let header = extract_header(&lines);
        assert_eq!(
            header,
            "Multi-language symbol extraction with tree-sitter AST | Supports 36 languages via WASM grammars"
        );
    }

    #[test]
    fn extract_header_skips_imports() {
        let lines = vec![
            "import something from 'module';",
            "use std::io;",
            "// Actual header line",
        ];
        let header = extract_header(&lines);
        assert_eq!(header, "Actual header line");
    }

    #[test]
    fn extract_header_empty_input() {
        let lines: Vec<&str> = vec![];
        assert_eq!(extract_header(&lines), "");
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
        assert_eq!(detect_language("foo.unknown"), None);
        assert_eq!(detect_language("noext"), None);
    }

    #[test]
    fn is_supported_file_works() {
        assert!(is_supported_file("main.rs"));
        assert!(is_supported_file("app.ts"));
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
}
