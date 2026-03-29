// Labeling, disambiguation, and file classification functions for semantic navigation.
// Extracted from semantic_navigate.rs for modularity.

use std::collections::{HashMap, HashSet};

use super::navigate_constants::*;
use super::semantic_navigate::FileInfo;

// ── File classification ─────────────────────────────────────────────

/// Check if a file path looks like a test file.
pub(crate) fn is_test_file(path: &str) -> bool {
    path.ends_with(".test.ts")
        || path.ends_with(".test.tsx")
        || path.ends_with(".spec.ts")
        || path.ends_with(".spec.tsx")
        || path.ends_with("_test.go")
        || path.ends_with("_test.rs")
        || path.contains("/test/")
        || path.contains("/tests/")
        || path.contains("/__tests__/")
}

/// Classify a file path into a type category string.
pub(crate) fn classify_file_type(path: &str) -> &'static str {
    if is_test_file(path) {
        return "tests";
    }
    if path.ends_with(".tsx") || path.ends_with(".jsx") {
        return "components";
    }
    if path.ends_with(".proto") {
        return "proto";
    }
    if path.ends_with(".sql") {
        return "sql";
    }
    if path.ends_with(".go") {
        return "go";
    }
    if path.ends_with(".rs") {
        return "rust";
    }
    if path.ends_with(".json") {
        return "json";
    }
    if path.ends_with(".yml") || path.ends_with(".yaml") {
        return "yaml";
    }
    if path.ends_with(".css") || path.ends_with(".scss") {
        return "css";
    }
    if path.ends_with(".schema.ts") || path.ends_with(".schema.js") {
        return "schemas";
    }
    "source" // generic for .ts/.js and others
}

/// Map a file type category to a human-readable display label.
/// Returns `None` for categories too generic to be useful.
pub(crate) fn file_type_display(category: &str) -> Option<&'static str> {
    match category {
        "tests" => Some("test files"),
        "components" => Some("React components"),
        "proto" => Some("proto definitions"),
        "sql" => Some("SQL migrations"),
        "go" => Some("Go source"),
        "rust" => Some("Rust source"),
        "json" => Some("JSON configs"),
        "yaml" => Some("YAML configs"),
        "css" => Some("stylesheets"),
        "schemas" => Some("schemas"),
        "source" => None, // too generic for TS-dominant projects
        _ => None,
    }
}

/// Returns true if a directory segment is a Next.js dynamic route parameter
/// (e.g., `[templateId]`, `[...nextauth]`, `[patientId]`, `[[...slug]]`).
pub(crate) fn is_nextjs_route_param(seg: &str) -> bool {
    seg.starts_with('[') || seg.ends_with(']')
}

/// Derive a human-readable label for a cluster from its file paths.
///
/// Maps path-based labels like "delivery/http" to more semantic descriptions
/// like "HTTP routes". Falls back to the original label if no mapping exists.
pub(crate) fn map_path_to_description(path_label: &str) -> &str {
    const LAYER_DESCRIPTIONS: &[(&[&str], &str)] = &[
        (&["delivery", "http"], "HTTP routes"),
        (&["delivery", "temporal"], "Temporal workflows"),
        (&["delivery", "nats"], "NATS consumers"),
        (&["repository", "pg"], "database queries"),
        (&["repository"], "data access"),
        (&["service"], "business logic"),
        (&["domain"], "domain models"),
        (&["delivery"], "API delivery"),
        (&["test", "integration"], "integration tests"),
        (&["tests", "integration"], "integration tests"),
    ];

    let segments: Vec<&str> = path_label.split('/').collect();

    // Try multi-segment matches first (longer patterns), then single-segment
    for (pattern, description) in LAYER_DESCRIPTIONS {
        if pattern.len() > segments.len() {
            continue;
        }
        if **pattern == segments[..pattern.len()] {
            return description;
        }
    }

    // No mapping found — return the original path label as-is
    path_label
}

/// Check if a word is a valid label word (not a stopword, not too short, etc.)
pub(crate) fn is_valid_label_word(word: &str) -> bool {
    word.len() >= 4
        && word
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        && word
            .chars()
            .next()
            .map(|c| c.is_alphabetic())
            .unwrap_or(false)
        && !matches!(
            word.to_lowercase().as_str(),
            "from" | "with" | "this" | "that" | "have" | "been" | "will"
            | "each" | "then" | "than" | "some" | "only" | "also" | "into"
            | "more" | "most" | "such" | "used" | "uses" | "using"
            | "file" | "files" | "type" | "types" | "data" | "code"
            | "package" | "generated" | "module" | "index" | "export"
            | "header" | "description" | "param" | "parameter" | "syntax"
            | "import" | "const" | "function" | "interface" | "class"
            | "async" | "await" | "return" | "string" | "number" | "boolean"
            | "undefined" | "null" | "void" | "true" | "false"
            | "todo" | "fixme" | "note" | "hack"
            // Generic code constants/markers that appear in file headers
            | "section" | "default" | "config" | "options" | "settings"
            | "props" | "state" | "action" | "reducer" | "store"
            | "error" | "event" | "handler" | "callback" | "listener"
            | "item" | "items" | "list" | "array" | "object"
            | "value" | "result" | "response" | "request"
        )
}

// ── Labeling ────────────────────────────────────────────────────────

/// Looks for the most common DISTINGUISHING directory segment — the first
/// segment after the common prefix where files diverge. This prevents
/// returning the parent directory name when all files share it.
pub(crate) fn derive_cluster_label(files: &[&FileInfo]) -> Option<String> {
    if files.is_empty() {
        return None;
    }

    let paths: Vec<Vec<&str>> = files
        .iter()
        .map(|f| f.relative_path.split('/').collect::<Vec<_>>())
        .collect();

    let min_depth = paths.iter().map(|p| p.len()).min().unwrap_or(0);
    if min_depth < 2 {
        return None;
    }

    // Find how deep the common prefix goes (excluding the filename)
    let mut common_depth = 0;
    for d in 0..min_depth.saturating_sub(1) {
        if paths.iter().all(|p| p[d] == paths[0][d]) {
            common_depth = d + 1;
        } else {
            break;
        }
    }

    // ── Content-pattern heuristics for homogeneous domain clusters ──
    // These fire BEFORE the segment-based heuristic so that clusters where
    // all files share a deep common prefix still get meaningful labels.
    // We scan ALL directory segments (including common prefix) because the
    // architecture layer name may be part of the shared prefix itself.

    // 1. Test file detection: if >50% of files are test files, label with tested modules
    {
        let test_count = files
            .iter()
            .filter(|f| is_test_file(&f.relative_path))
            .count();
        if test_count > files.len() / 2 {
            // Find what's being tested — look at filenames without test suffixes
            let tested_modules: Vec<&str> = files
                .iter()
                .filter_map(|f| f.relative_path.split('/').next_back())
                .filter_map(|name| {
                    name.strip_suffix(".test.ts")
                        .or_else(|| name.strip_suffix(".test.tsx"))
                        .or_else(|| name.strip_suffix(".spec.ts"))
                        .or_else(|| name.strip_suffix(".spec.tsx"))
                        .or_else(|| name.strip_suffix("_test.go"))
                        .or_else(|| name.strip_suffix("_test.rs"))
                })
                .collect();
            if !tested_modules.is_empty() {
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for m in &tested_modules {
                    *counts.entry(m).or_default() += 1;
                }
                let mut sorted: Vec<_> = counts.into_iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
                let top: Vec<&str> = sorted.iter().take(3).map(|(n, _)| *n).collect();
                return Some(format!("{} tests", top.join(" + ")));
            }
            return Some("tests".to_string());
        }
    }

    // 2. Architecture layer detection: check ALL path segments for well-known
    //    layer names, pick the most frequent one.
    {
        let layer_map: &[(&[&str], &str)] = &[
            (&["service"], "business logic"),
            (&["repository", "repo"], "data access"),
            (&["domain"], "domain models"),
            (&["test", "tests", "__tests__"], "tests"),
            (&["delivery"], "API delivery"),
        ];
        // Two-segment layers (checked first for specificity)
        let two_seg_map: &[(&str, &str, &str)] = &[
            ("delivery", "http", "HTTP routes"),
            ("delivery", "temporal", "Temporal workflows"),
            ("delivery", "nats", "NATS consumers"),
            ("repository", "pg", "database queries"),
            ("test", "integration", "integration tests"),
            ("tests", "integration", "integration tests"),
        ];

        let mut layer_counts: HashMap<&str, usize> = HashMap::new();

        for p in &paths {
            let dir_segs = &p[..p.len().saturating_sub(1)]; // all dir segments, exclude filename
            // Check two-segment layers first
            let mut matched = false;
            for (seg1, seg2, label) in two_seg_map {
                if dir_segs.len() >= 2
                    && dir_segs.iter().any(|s| s.eq_ignore_ascii_case(seg1))
                    && dir_segs.iter().any(|s| s.eq_ignore_ascii_case(seg2))
                {
                    *layer_counts.entry(label).or_default() += 1;
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }
            // Check single-segment layers
            for (keywords, label) in layer_map {
                if dir_segs
                    .iter()
                    .any(|s| keywords.iter().any(|k| s.eq_ignore_ascii_case(k)))
                {
                    *layer_counts.entry(label).or_default() += 1;
                    break;
                }
            }
        }

        if let Some((label, count)) = layer_counts.iter().max_by_key(|(_, c)| **c)
            && *count > files.len() / 2
        {
            // For generic labels like "business logic", try to describe WHICH logic
            if *label == "business logic" || *label == "service" {
                let names: Vec<&str> = files
                    .iter()
                    .filter_map(|f| f.relative_path.split('/').next_back())
                    .filter_map(|n| {
                        n.strip_suffix(".ts")
                            .or_else(|| n.strip_suffix(".tsx"))
                            .or_else(|| n.strip_suffix(".go"))
                            .or_else(|| n.strip_suffix(".rs"))
                    })
                    .filter(|n| {
                        !n.contains("test")
                            && !n.contains("spec")
                            && *n != "index"
                            && *n != "mod"
                            && !n.contains("error")
                    })
                    .collect();
                if !names.is_empty() {
                    // Deduplicate and take top names
                    let mut name_counts: HashMap<&str, usize> = HashMap::new();
                    for n in &names {
                        *name_counts.entry(n).or_default() += 1;
                    }
                    let mut sorted: Vec<_> = name_counts.into_iter().collect();
                    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
                    let top: Vec<&str> = sorted.iter().take(3).map(|(n, _)| *n).collect();
                    return Some(format!("{} logic", top.join(" + ")));
                }
            }
            // For "domain models" / "domain entities", enhance with file names
            if label.starts_with("domain") {
                let names: Vec<&str> = files
                    .iter()
                    .filter_map(|f| f.relative_path.split('/').next_back())
                    .filter_map(|n| n.strip_suffix(".ts"))
                    .filter(|n| *n != "index" && !n.contains("test"))
                    .take(3)
                    .collect();
                if !names.is_empty() {
                    return Some(format!("{} models", names.join(" + ")));
                }
            }
            return Some(label.to_string());
        }
    }

    // 3. Common file extension grouping: if >50% share a distinctive extension
    {
        let mut ext_counts: HashMap<&str, usize> = HashMap::new();
        let mut index_count: usize = 0;
        for f in files {
            let fname = f.relative_path.split('/').next_back().unwrap_or("");
            if fname == "index.ts"
                || fname == "index.js"
                || fname == "index.tsx"
                || fname == "mod.rs"
            {
                index_count += 1;
            }
            if let Some(ext) = fname.rsplit('.').next() {
                *ext_counts.entry(ext).or_default() += 1;
            }
        }
        if index_count > files.len() / 2 {
            // Find the most common domain directory (2-3 levels up from index files)
            // e.g., "packages/domains/scheduling/service/index.ts" → "scheduling"
            let domain_name = {
                let mut dir_counts: HashMap<&str, usize> = HashMap::new();
                for f in files {
                    let segs: Vec<&str> = f.relative_path.split('/').collect();
                    // Walk segments backwards (skip filename), pick the first
                    // non-generic segment that isn't the immediate parent
                    // (immediate parent is often "service", "repository", etc.)
                    let len = segs.len();
                    if len >= 3 {
                        // Check segments from 2 levels up to 3 levels up
                        for i in (0..len.saturating_sub(2)).rev().take(3) {
                            let seg = segs[i];
                            if !GENERIC_SEGMENTS.contains(&seg)
                                && !matches!(
                                    seg,
                                    "packages"
                                        | "domains"
                                        | "apps"
                                        | "src"
                                        | "lib"
                                        | "internal"
                                        | "cmd"
                                        | "pkg"
                                )
                                && !is_nextjs_route_param(seg)
                            {
                                *dir_counts.entry(seg).or_default() += 1;
                                break;
                            }
                        }
                    }
                }
                dir_counts
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(name, _)| name.to_string())
            };
            return Some(match domain_name {
                Some(name) => format!("{} exports", name),
                None => "barrel exports".to_string(),
            });
        }
        if let Some((ext, count)) = ext_counts.iter().max_by_key(|(_, c)| **c)
            && *count > files.len() / 2
        {
            match *ext {
                "proto" => {
                    // Extract domain from proto paths: "contracts/proto/billing/v1/..." → "billing"
                    let mut domain_counts: HashMap<&str, usize> = HashMap::new();
                    for f in files {
                        let parts: Vec<&str> = f.relative_path.split('/').collect();
                        for (i, part) in parts.iter().enumerate() {
                            if *part == "proto" && i + 1 < parts.len() {
                                *domain_counts.entry(parts[i + 1]).or_default() += 1;
                                break;
                            }
                        }
                    }
                    if let Some((domain, _)) = domain_counts.iter().max_by_key(|(_, c)| **c)
                        && !domain.contains('.')
                    {
                        return Some(format!("{} proto", domain));
                    }
                    return Some("proto definitions".to_string());
                }
                "sql" => return Some("migrations".to_string()),
                _ => {}
            }
        }
    }

    // Look at the FIRST segment after the common prefix — that's where
    // this cluster differs from siblings. Find the most common value.
    if common_depth < min_depth.saturating_sub(1) {
        let mut seg_counts: HashMap<&str, usize> = HashMap::new();
        for p in &paths {
            if common_depth < p.len().saturating_sub(1) {
                *seg_counts.entry(p[common_depth]).or_default() += 1;
            }
        }
        if let Some((seg, count)) = seg_counts.iter().max_by_key(|(_, c)| **c)
            && *count > files.len() / 3
            && !GENERIC_SEGMENTS.contains(seg)
            && !is_nextjs_route_param(seg)
        {
            // If there's a second-level distinguisher too, use it
            let mut sub_counts: HashMap<&str, usize> = HashMap::new();
            for p in &paths {
                if common_depth + 1 < p.len().saturating_sub(1) && p[common_depth] == *seg {
                    *sub_counts.entry(p[common_depth + 1]).or_default() += 1;
                }
            }
            if let Some((sub_seg, sub_count)) = sub_counts.iter().max_by_key(|(_, c)| **c)
                && *sub_count > files.len() / 3
                && !GENERIC_SEGMENTS.contains(sub_seg)
                && !is_nextjs_route_param(sub_seg)
            {
                let raw = format!("{}/{}", seg, sub_seg);
                let descriptive = map_path_to_description(&raw);
                return Some(descriptive.to_string());
            }
            return Some(map_path_to_description(seg).to_string());
        }
    }

    // No distinguishing segment — try to find a common keyword from file names
    // E.g., files like "SoapTranscription.tsx", "SoapDiagnosis.tsx" → "Soap"
    // or "useSideBarActions.ts", "useSideBarBulk.ts" → "SideBar"
    {
        let filenames: Vec<&str> = files
            .iter()
            .filter_map(|f| f.relative_path.split('/').next_back())
            .filter_map(|name| name.split('.').next()) // strip extension
            .collect();
        if filenames.len() >= 2 {
            // Find the longest common prefix of filenames
            let first = filenames[0].as_bytes();
            let mut prefix_len = first.len();
            for name in &filenames[1..] {
                let bytes = name.as_bytes();
                prefix_len = prefix_len.min(bytes.len());
                for i in 0..prefix_len {
                    if first[i] != bytes[i] {
                        prefix_len = i;
                        break;
                    }
                }
            }
            // Use prefix if it's a meaningful word (3+ chars, not just "use" or "get")
            if prefix_len >= 4 {
                let prefix = &filenames[0][..prefix_len];
                // Trim to last uppercase boundary for camelCase
                if let Some(last_upper) = prefix.rfind(|c: char| c.is_uppercase())
                    && last_upper > 0
                {
                    let trimmed = &prefix[..last_upper];
                    if trimmed.len() >= 4 {
                        return Some(trimmed.to_string());
                    }
                }
                // Only return full prefix if it's still >= 4 chars
                return Some(prefix.to_string());
            }
        }
    }

    // Last resort: use the most common header keyword
    // Only accept words that look like real English/code identifiers:
    // alphabetic, 4+ chars, no punctuation/special chars, not a stopword.
    {
        let mut word_counts: HashMap<&str, usize> = HashMap::new();
        for f in files {
            for word in f.header.split_whitespace() {
                let clean =
                    word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
                if is_valid_label_word(clean) {
                    *word_counts.entry(clean).or_default() += 1;
                }
            }
        }
        if let Some((word, count)) = word_counts.iter().max_by_key(|(_, c)| **c)
            && *count > files.len() / 3
        {
            return Some(word.to_string());
        }
    }

    // Fall back to last 1-2 segments of common prefix
    if common_depth >= 2 {
        let label = format!(
            "{}/{}",
            paths[0][common_depth - 2],
            paths[0][common_depth - 1]
        );
        let generic = ["packages", "apps", "src", "lib", "internal"];
        let segs_clean = !is_nextjs_route_param(paths[0][common_depth - 2])
            && !is_nextjs_route_param(paths[0][common_depth - 1]);
        if segs_clean && !generic.contains(&label.as_str()) {
            return Some(label);
        }
    }

    None
}

/// Find a disambiguator for a cluster — checks for test files, architecture layers, etc.
pub(crate) fn find_label_disambiguator(files: &[&FileInfo]) -> Option<String> {
    // Test files
    let test_count = files
        .iter()
        .filter(|f| is_test_file(&f.relative_path))
        .count();
    if test_count > files.len() / 2 {
        // Try to name what's being tested
        let tested_modules: Vec<&str> = files
            .iter()
            .filter_map(|f| f.relative_path.split('/').next_back())
            .filter_map(|name| {
                name.strip_suffix(".test.ts")
                    .or_else(|| name.strip_suffix(".test.tsx"))
                    .or_else(|| name.strip_suffix(".spec.ts"))
                    .or_else(|| name.strip_suffix(".spec.tsx"))
                    .or_else(|| name.strip_suffix("_test.go"))
                    .or_else(|| name.strip_suffix("_test.rs"))
            })
            .collect();
        if !tested_modules.is_empty() {
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for m in &tested_modules {
                *counts.entry(m).or_default() += 1;
            }
            let mut sorted: Vec<_> = counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
            let top: Vec<&str> = sorted.iter().take(3).map(|(n, _)| *n).collect();
            return Some(format!("{} tests", top.join(" + ")));
        }
        return Some("tests".to_string());
    }

    // Architecture layers
    let layers = [
        ("service", "service"),
        ("repository", "repository"),
        ("repo", "repository"),
        ("delivery", "delivery"),
        ("domain", "domain"),
        ("http", "http"),
        ("temporal", "temporal"),
    ];
    let mut layer_counts: HashMap<&str, usize> = HashMap::new();
    for f in files {
        for (pattern, label) in &layers {
            if f.relative_path.contains(&format!("/{}/", pattern)) {
                *layer_counts.entry(label).or_default() += 1;
            }
        }
    }
    if let Some((layer, count)) = layer_counts.iter().max_by_key(|(_, c)| **c)
        && *count > files.len() / 3
    {
        return Some(layer.to_string());
    }

    None
}

/// Produce a descriptive label for a group of files when all other heuristics fail.
///
/// Tries three strategies in order:
/// 1. Most common subdirectory after the shared prefix (e.g. "RecordSession")
/// 2. Dominant file-type description (e.g. "React components", "test files")
/// 3. File count with content type (e.g. "14 React components", "8 TypeScript modules")
pub(crate) fn describe_file_group(refs: &[&FileInfo]) -> String {
    if refs.is_empty() {
        return "empty cluster".to_string();
    }

    // Strategy 1: most common parent directory after common prefix (skip generic segments)
    let paths: Vec<Vec<&str>> = refs
        .iter()
        .map(|f| f.relative_path.split('/').collect::<Vec<_>>())
        .collect();
    let min_depth = paths.iter().map(|p| p.len()).min().unwrap_or(0);
    let mut common = 0;
    for d in 0..min_depth.saturating_sub(1) {
        if paths.iter().all(|p| p[d] == paths[0][d]) {
            common = d + 1;
        } else {
            break;
        }
    }
    if common < min_depth.saturating_sub(1) {
        let mut seg_counts: HashMap<&str, usize> = HashMap::new();
        for p in &paths {
            if common < p.len().saturating_sub(1) {
                *seg_counts.entry(p[common]).or_default() += 1;
            }
        }
        if let Some((seg, _)) = seg_counts.into_iter().max_by_key(|(_, c)| *c)
            && !GENERIC_SEGMENTS.contains(&seg)
            && !is_nextjs_route_param(seg)
        {
            return seg.to_string();
        }
    }

    // Strategy 1.5: If files span multiple subdirectories, list the top 2-3
    {
        let mut subdir_counts: HashMap<&str, usize> = HashMap::new();
        for p in &paths {
            if common < p.len().saturating_sub(1) {
                *subdir_counts.entry(p[common]).or_default() += 1;
            }
        }
        if subdir_counts.len() >= 2 {
            let mut sorted: Vec<_> = subdir_counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
            let top: Vec<&str> = sorted
                .iter()
                .take(3)
                .map(|(name, _)| *name)
                .filter(|n| !is_nextjs_route_param(n))
                .collect();
            if !top.is_empty() {
                return top.join(" + ");
            }
        }
    }

    // Strategy 2: describe by dominant file type (skips overly generic labels)
    if let Some(label) = file_type_label(refs) {
        return label;
    }

    // Strategy 3: fallback — use directory name (even generic) with count, never bare "N files"
    let n = refs.len();
    if common < min_depth.saturating_sub(1) {
        let mut seg_counts: HashMap<&str, usize> = HashMap::new();
        for p in &paths {
            if common < p.len().saturating_sub(1) {
                *seg_counts.entry(p[common]).or_default() += 1;
            }
        }
        if let Some((seg, _)) = seg_counts.into_iter().max_by_key(|(_, c)| *c)
            && !is_nextjs_route_param(seg)
        {
            return format!("{} {} modules", n, seg);
        }
    }

    // Use file names instead of bare count
    let names: Vec<&str> = refs
        .iter()
        .filter_map(|f| f.relative_path.split('/').next_back())
        .filter_map(|n| {
            n.strip_suffix(".ts")
                .or_else(|| n.strip_suffix(".tsx"))
                .or_else(|| n.strip_suffix(".go"))
        })
        .filter(|n| *n != "index" && *n != "mod")
        .collect();
    let deduped: Vec<&str> = names.iter().take(3).cloned().collect();
    if !deduped.is_empty() {
        return deduped.join(" + ");
    }
    // Fall back to count only as absolute last resort
    format!("{} files", refs.len())
}

/// Classify a group of files by their dominant extension/suffix pattern.
///
/// Returns `None` for overly generic labels (e.g. "TypeScript modules" in a TS project).
pub(crate) fn file_type_label(refs: &[&FileInfo]) -> Option<String> {
    dominant_file_type(refs)
        .and_then(file_type_display)
        .map(|display| format!("{} {}", refs.len(), display))
}

/// Label files using Ollama chat for small sets.
pub(crate) async fn label_files(files: &[FileInfo]) -> Vec<String> {
    // For small file sets (≤ MAX_FILES_PER_LEAF), headers are descriptive enough.
    // Skip LLM call entirely — saves ~6s per invocation.
    files
        .iter()
        .map(|f| {
            if f.header.is_empty() {
                f.relative_path
                    .split('/')
                    .next_back()
                    .unwrap_or(&f.relative_path)
                    .to_string()
            } else {
                f.header.clone()
            }
        })
        .collect()
}

/// Deduplicate sibling cluster labels by REPLACING duplicates with disambiguators.
/// E.g., "scheduling" ×3 → "tests", "service", "repository" (not "scheduling (tests)")
pub(crate) fn deduplicate_sibling_labels(
    labels: &mut [String],
    clusters: &[(Vec<&FileInfo>, Option<String>)],
) {
    let mut seen: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        seen.entry(label.to_lowercase()).or_default().push(i);
    }

    for indices in seen.values() {
        if indices.len() <= 1 {
            continue;
        }
        // Collect disambiguators for all duplicates first
        let disambigs: Vec<(usize, Option<String>)> = indices
            .iter()
            .map(|&idx| {
                if idx < clusters.len() {
                    let (files, _) = &clusters[idx];
                    (idx, find_label_disambiguator(files))
                } else {
                    (idx, None)
                }
            })
            .collect();

        // Check if all disambiguators are the same (e.g., all "tests")
        let unique_disambigs: HashSet<String> =
            disambigs.iter().filter_map(|(_, d)| d.clone()).collect();

        if unique_disambigs.len() >= 2 {
            // Good — disambiguators are different. REPLACE the label entirely.
            for (idx, disambig) in &disambigs {
                if let Some(d) = disambig {
                    labels[*idx] = d.clone();
                }
                // Leave None ones unchanged (they'll get #N in second pass)
            }
        }
        // If all disambiguators are identical or None, skip — second pass handles it
    }

    // Second pass: for remaining duplicates, find what makes each unique
    let mut final_seen: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        final_seen.entry(label.to_lowercase()).or_default().push(i);
    }
    for indices in final_seen.values() {
        if indices.len() <= 1 {
            continue;
        }

        // Compute a distinguishing suffix for each duplicate cluster
        let suffixes: Vec<(usize, String)> = indices
            .iter()
            .filter_map(|&idx| {
                if idx >= clusters.len() {
                    return None;
                }
                let (files, _) = &clusters[idx];
                let suffix = describe_cluster_uniqueness(files, clusters, indices, idx);
                Some((idx, suffix))
            })
            .collect();

        // Check if the computed suffixes are actually unique across siblings
        let unique_suffixes: HashSet<&str> = suffixes.iter().map(|(_, s)| s.as_str()).collect();
        if unique_suffixes.len() == suffixes.len() {
            // All suffixes are distinct — use them
            for (idx, suffix) in &suffixes {
                let base = labels[*idx]
                    .split(" #")
                    .next()
                    .unwrap_or(&labels[*idx])
                    .to_string();
                // Skip redundant "(N files)" if label already starts with a count
                let label_has_count = base
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false);
                if label_has_count && suffix.contains("files") {
                    continue;
                }
                // Use file names instead of suffix when:
                // 1. Suffix is already in the base label (redundant)
                // 2. Suffix is a generic/technical term (node, src, lib, etc.)
                // 3. Suffix is just a file count ("N files")
                let suffix_is_weak = base.to_lowercase().contains(&suffix.to_lowercase())
                    || GENERIC_SEGMENTS.contains(&suffix.to_lowercase().as_str())
                    || suffix.contains("files")
                    || suffix == "node"
                    || suffix == "src"
                    || suffix == "lib";

                if suffix_is_weak && *idx < clusters.len() {
                    let (cluster_files, _) = &clusters[*idx];
                    let names: Vec<&str> = cluster_files
                        .iter()
                        .filter_map(|f| f.relative_path.split('/').next_back())
                        .filter_map(|n| {
                            n.strip_suffix(".ts")
                                .or_else(|| n.strip_suffix(".tsx"))
                                .or_else(|| n.strip_suffix(".go"))
                                .or_else(|| n.strip_suffix(".rs"))
                        })
                        .filter(|n| *n != "index" && *n != "mod")
                        .take(3)
                        .collect();
                    if !names.is_empty() {
                        labels[*idx] = format!("{} ({})", base, names.join(" + "));
                        continue;
                    }
                }
                labels[*idx] = format!("{} ({})", base, suffix);
            }
        } else {
            // Suffixes collided — fall back to numbering
            let mut counter = 0usize;
            for &idx in indices {
                counter += 1;
                if counter > 1 {
                    labels[idx] = format!("{} #{}", labels[idx], counter);
                }
            }
        }
    }
}

/// Check if a label is essentially the same as its parent group label.
pub(crate) fn label_matches_parent(label: &str, parent: &str) -> bool {
    let l = label.to_lowercase();
    let p = parent.to_lowercase();
    if l == p {
        return true;
    }
    // Parent contains label as a complete segment
    if p.split('/').any(|seg| seg == l) {
        return true;
    }
    // Label equals a segment of the parent path
    if l.split('/').any(|seg| p.split('/').any(|ps| ps == seg)) {
        return true;
    }
    false
}

// ── Disambiguation ──────────────────────────────────────────────────

/// Describe what makes THIS cluster unique compared to its duplicate siblings.
///
/// Tries in order:
/// 1. Most common directory segment unique to this cluster (not shared by siblings)
/// 2. Dominant file extension/type if different from siblings
/// 3. Dominant API domain from generated filenames (e.g., "scheduling API types")
/// 4. File count as last resort
pub(crate) fn describe_cluster_uniqueness(
    files: &[&FileInfo],
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    sibling_indices: &[usize],
    my_idx: usize,
) -> String {
    if let Some(dir) = find_unique_dominant_directory(files, clusters, sibling_indices, my_idx) {
        return dir;
    }
    if let Some(ftype) = find_unique_file_type(files, clusters, sibling_indices, my_idx) {
        return ftype;
    }
    if let Some(domain) = find_unique_api_domain(files, clusters, sibling_indices, my_idx) {
        return domain;
    }
    if let Some(subdir) = find_distinctive_subdirectory(files, clusters, sibling_indices, my_idx) {
        return subdir;
    }
    format!("{} files", files.len())
}

/// Find the most common directory segment in this cluster that other sibling clusters don't share.
pub(crate) fn find_unique_dominant_directory(
    files: &[&FileInfo],
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    sibling_indices: &[usize],
    my_idx: usize,
) -> Option<String> {
    let my_segments = count_directory_segments(files);

    let mut other_segments: HashMap<String, usize> = HashMap::new();
    for &idx in sibling_indices {
        if idx == my_idx || idx >= clusters.len() {
            continue;
        }
        let (other_files, _) = &clusters[idx];
        for (seg, count) in count_directory_segments(other_files) {
            *other_segments.entry(seg).or_default() += count;
        }
    }

    let mut best: Option<(String, usize)> = None;
    for (seg, count) in &my_segments {
        let other_count = other_segments.get(seg).copied().unwrap_or(0);
        if *count > files.len() / 3
            && other_count == 0
            && best.as_ref().is_none_or(|(_, bc)| count > bc)
        {
            best = Some((seg.clone(), *count));
        }
    }

    best.map(|(seg, _)| seg)
}

/// Find the dominant file type of this cluster if it differs from all siblings.
pub(crate) fn find_unique_file_type(
    files: &[&FileInfo],
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    sibling_indices: &[usize],
    my_idx: usize,
) -> Option<String> {
    let my_type = dominant_file_type(files)?;

    for &idx in sibling_indices {
        if idx == my_idx || idx >= clusters.len() {
            continue;
        }
        let (other_files, _) = &clusters[idx];
        if let Some(other_type) = dominant_file_type(other_files)
            && other_type == my_type
        {
            return None;
        }
    }

    Some(my_type.to_string())
}

/// Find a unique API domain label for this cluster that siblings don't share.
pub(crate) fn find_unique_api_domain(
    files: &[&FileInfo],
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    sibling_indices: &[usize],
    my_idx: usize,
) -> Option<String> {
    let my_domain = dominant_api_domain(files)?;

    for &idx in sibling_indices {
        if idx == my_idx || idx >= clusters.len() {
            continue;
        }
        let (other_files, _) = &clusters[idx];
        if let Some(other_domain) = dominant_api_domain(other_files)
            && other_domain == my_domain
        {
            return None;
        }
    }

    Some(my_domain)
}

/// Find the most distinctive subdirectory (parent dir of files) for this cluster
/// compared to its siblings. Useful when clusters share the same grandparent directory
/// but contain files from different subdirectories within it.
pub(crate) fn find_distinctive_subdirectory(
    files: &[&FileInfo],
    clusters: &[(Vec<&FileInfo>, Option<String>)],
    sibling_indices: &[usize],
    my_idx: usize,
) -> Option<String> {
    let my_subdirs = count_subdirectories(files);

    // Count subdirectories across all sibling clusters
    let mut sibling_subdirs: HashMap<String, usize> = HashMap::new();
    for &idx in sibling_indices {
        if idx == my_idx || idx >= clusters.len() {
            continue;
        }
        let (sib_files, _) = &clusters[idx];
        for (dir, count) in count_subdirectories(sib_files) {
            *sibling_subdirs.entry(dir).or_default() += count;
        }
    }

    // Find subdirectory that's most common in THIS cluster but rare in siblings
    my_subdirs
        .into_iter()
        .filter(|(_, count)| *count >= 3)
        .max_by_key(|(dir, my_count)| {
            let sib_count = sibling_subdirs.get(dir).copied().unwrap_or(0);
            *my_count as i64 - sib_count as i64
        })
        .map(|(dir, _)| dir)
}

/// Count occurrences of each meaningful directory segment in file paths.
pub(crate) fn count_directory_segments(files: &[&FileInfo]) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for f in files {
        for segment in f.relative_path.split('/') {
            if segment.contains('.') || segment.len() < 2 {
                continue;
            }
            *counts.entry(segment.to_string()).or_default() += 1;
        }
    }
    counts
}

/// Count occurrences of the parent directory (2nd-to-last path segment) for each file.
pub(crate) fn count_subdirectories(files: &[&FileInfo]) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for f in files {
        let parts: Vec<&str> = f.relative_path.split('/').collect();
        if parts.len() >= 2 {
            let parent = parts[parts.len() - 2];
            if !is_nextjs_route_param(parent) {
                *counts.entry(parent.to_string()).or_default() += 1;
            }
        }
    }
    counts
}

/// Return the dominant file type category for a cluster, if >50% of files share it.
pub(crate) fn dominant_file_type(files: &[&FileInfo]) -> Option<&'static str> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for f in files {
        *counts
            .entry(classify_file_type(&f.relative_path))
            .or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .filter(|(_, c)| *c > files.len() / 2)
        .map(|(cat, _)| cat)
}

/// Find the dominant API domain in a cluster's filenames (>50% must share it).
pub(crate) fn dominant_api_domain(files: &[&FileInfo]) -> Option<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for f in files {
        let fname = f.relative_path.split('/').next_back().unwrap_or("");
        if let Some(domain) = extract_api_domain_from_filename(fname) {
            *counts.entry(domain).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .filter(|(_, c)| *c > files.len() / 2)
        .map(|(domain, _)| format!("{} API types", domain))
}

/// Extract the API domain from a generated type filename.
///
/// Matches patterns like `getApiV1SchedulingServices200.ts` → `"scheduling"`,
/// `postApiV1AuthLogin200.ts` → `"auth"`, `patchApiV1OrganizationsMembers.ts` → `"organizations"`.
pub(crate) fn extract_api_domain_from_filename(filename: &str) -> Option<String> {
    let lower = filename.to_lowercase();
    let pos = lower.find("apiv1")?;
    let after = &filename[pos + 5..];
    if after.is_empty() {
        return None;
    }
    // The domain name starts right after "ApiV1" in PascalCase.
    // Find where the domain word ends: the next uppercase letter after the
    // initial run. E.g., "SchedulingServices200.ts" → "Scheduling".
    let bytes = after.as_bytes();
    let mut end = 1; // skip first char (always uppercase start of domain)
    while end < bytes.len() {
        let c = bytes[end] as char;
        if c.is_uppercase() || c.is_ascii_digit() || c == '.' {
            break;
        }
        end += 1;
    }
    let domain = &after[..end];
    if domain.len() >= 3 {
        Some(domain.to_lowercase())
    } else {
        None
    }
}

// ── Validation ──────────────────────────────────────────────────────

/// Generic label terms describe code nature, not specific subdirectories.
/// These words commonly appear in LLM labels but rarely in file paths, causing
/// false rejections when a few paths happen to contain "components/" or "api/".
pub(crate) const LABEL_ALLOWLIST: &[&str] = &[
    "react",
    "vue",
    "angular",
    "svelte",
    "next",
    "nuxt",
    "node",
    "express",
    "fastify",
    "nest",
    "components",
    "hooks",
    "pages",
    "layouts",
    "views",
    "widgets",
    "api",
    "rest",
    "graphql",
    "grpc",
    "proto",
    "template",
    "templates",
    "config",
    "configs",
    "configuration",
    "server",
    "client",
    "frontend",
    "backend",
    "typescript",
    "javascript",
    "python",
    "golang",
    "rust",
    "source",
    "modules",
    "packages",
    "library",
    "form",
    "forms",
    "modal",
    "modals",
    "dialog",
    "dialogs",
    "page",
    "route",
    "routes",
    "routing",
    "navigation",
    "auth",
    "authentication",
    "state",
    "store",
    "redux",
    "zustand",
    "context",
    "style",
    "styles",
    "styled",
    "css",
    "scss",
    "test",
    "tests",
    "spec",
    "specs",
    "testing",
    "util",
    "utils",
    "utility",
    "utilities",
    "helper",
    "helpers",
    "service",
    "services",
    "handler",
    "handlers",
    "model",
    "models",
    "entity",
    "entities",
    "schema",
    "schemas",
    "feature",
    "features",
    "domain",
    "domains",
    "shared",
    "common",
    "core",
    "base",
    "internal",
    "dashboard",
    "admin",
    "portal",
    "management",
    "workflow",
    "workflows",
];

/// Validate that an LLM-generated label actually represents the cluster content.
///
/// Returns `false` if the label appears to name a minority feature — e.g. when
/// one prominent file skews the LLM into labeling an 81-file cluster after a
/// single file's concern. We check by tokenizing the label into words and seeing
/// if those words appear predominantly in only a small fraction of the cluster's
/// file paths. If the label words match a specific subdirectory holding <20% of
/// files (and don't match the majority), the label is rejected.
pub(crate) fn validate_label_against_cluster(label: &str, file_refs: &[&FileInfo]) -> bool {
    if file_refs.len() < 5 {
        // Too small to have a mislabeling problem
        return true;
    }

    let label_lower = label.to_lowercase();
    let label_words: Vec<&str> = label_lower
        .split_whitespace()
        .filter(|w| w.len() > 2) // skip short words like "of", "and"
        .collect();

    if label_words.is_empty() {
        return true;
    }

    // Filter out allowlisted terms — only use path-specific words for validation
    let path_specific_words: Vec<&&str> = label_words
        .iter()
        .filter(|w| !LABEL_ALLOWLIST.contains(w))
        .collect();

    // If ALL label words are allowlisted terms (e.g. "React Form Components"),
    // the label is conceptual, not path-derived — always accept it.
    if path_specific_words.is_empty() {
        tracing::info!(
            label = label,
            "semantic_navigate: validate_label — all words are allowlisted terms, accepting: {:?}",
            label
        );
        return true;
    }

    // Count how many files have paths matching any PATH-SPECIFIC label word
    let matching_files = file_refs
        .iter()
        .filter(|f| {
            let path_lower = f.relative_path.to_lowercase();
            path_specific_words.iter().any(|w| path_lower.contains(**w))
        })
        .count();

    let match_ratio = matching_files as f64 / file_refs.len() as f64;

    // If fewer than 20% of files match the path-specific label words,
    // the LLM likely named the cluster after a minority feature.
    // Exception: labels with zero matches are conceptual (not path-derived) — accept them.
    if matching_files > 0 && match_ratio < 0.20 {
        tracing::info!(
            label = label,
            matching_files = matching_files,
            total_files = file_refs.len(),
            match_ratio = format!("{:.2}", match_ratio).as_str(),
            path_specific_words = format!("{:?}", path_specific_words).as_str(),
            "semantic_navigate: validate_label — rejecting {:?} (match_ratio={:.2}, path_words={:?})",
            label,
            match_ratio,
            path_specific_words
        );
        return false;
    }

    tracing::info!(
        label = label,
        matching_files = matching_files,
        total_files = file_refs.len(),
        match_ratio = format!("{:.2}", match_ratio).as_str(),
        "semantic_navigate: validate_label — accepting {:?} (match_ratio={:.2}, {} of {} files)",
        label,
        match_ratio,
        matching_files,
        file_refs.len()
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- file classification tests ---

    #[test]
    fn is_nextjs_route_param_dynamic_segment() {
        assert!(is_nextjs_route_param("[templateId]"));
    }

    #[test]
    fn is_nextjs_route_param_catch_all() {
        assert!(is_nextjs_route_param("[...nextauth]"));
    }

    #[test]
    fn is_nextjs_route_param_regular_dir() {
        assert!(!is_nextjs_route_param("components"));
    }

    #[test]
    fn is_nextjs_route_param_filename() {
        assert!(!is_nextjs_route_param("index.ts"));
    }

    // --- map_path_to_description tests ---

    #[test]
    fn map_path_to_description_delivery_http() {
        assert_eq!(map_path_to_description("delivery/http"), "HTTP routes");
    }

    #[test]
    fn map_path_to_description_repository_pg() {
        assert_eq!(map_path_to_description("repository/pg"), "database queries");
    }

    #[test]
    fn map_path_to_description_service() {
        assert_eq!(map_path_to_description("service"), "business logic");
    }

    #[test]
    fn map_path_to_description_unknown_returns_as_is() {
        assert_eq!(map_path_to_description("unknown"), "unknown");
    }

    // --- label_matches_parent tests ---

    #[test]
    fn label_matches_parent_exact_match() {
        assert!(label_matches_parent("scheduling", "scheduling"));
    }

    #[test]
    fn label_matches_parent_parent_contains_label() {
        assert!(label_matches_parent("scheduling", "domains/scheduling"));
    }

    #[test]
    fn label_matches_parent_label_segment_matches_parent_segment() {
        assert!(label_matches_parent("auth", "domains/auth"));
    }

    #[test]
    fn label_matches_parent_no_match() {
        assert!(!label_matches_parent("billing", "scheduling"));
    }

    #[test]
    fn label_matches_parent_substring_not_segment_no_match() {
        assert!(!label_matches_parent("Authentication Service", "auth"));
    }

    // --- describe_file_group tests ---

    #[test]
    fn describe_file_group_spanning_three_generic_subdirs() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "project/src/foo.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "project/lib/bar.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "project/utils/baz.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = describe_file_group(&refs);
        assert!(
            result.contains(" + "),
            "Expected 'A + B + C' format, got: {}",
            result
        );
    }

    #[test]
    fn describe_file_group_all_in_one_subdir_tsx() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "app/components/Header.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/components/Footer.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/components/Sidebar.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = describe_file_group(&refs);
        assert!(
            result.contains("React components"),
            "Expected React components label, got: {}",
            result
        );
    }

    #[test]
    fn describe_file_group_single_non_generic_subdir() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "app/billing/invoice.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/billing/payment.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/scheduling/calendar.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = describe_file_group(&refs);
        assert_eq!(result, "billing");
    }

    // --- file_type_label tests ---

    #[test]
    fn file_type_label_all_tsx_returns_react_components() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "app/Header.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/Footer.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/Nav.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = file_type_label(&refs);
        assert_eq!(result, Some("3 React components".to_string()));
    }

    #[test]
    fn file_type_label_all_test_ts_returns_test_files() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "src/auth.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "src/billing.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = file_type_label(&refs);
        assert_eq!(result, Some("2 test files".to_string()));
    }

    #[test]
    fn file_type_label_all_ts_returns_none() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "src/auth.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "src/billing.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(file_type_label(&refs), None);
    }

    #[test]
    fn file_type_label_mixed_returns_none() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "a.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "b.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "c.proto".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "d.sql".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(file_type_label(&refs), None);
    }

    // --- validate_label_against_cluster tests ---

    #[test]
    fn validate_label_all_technology_terms_accepted() {
        let files: Vec<FileInfo> = (0..10)
            .map(|i| FileInfo {
                relative_path: format!("pkg/module{}.ts", i),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            })
            .collect();
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert!(validate_label_against_cluster(
            "React Form Components",
            &refs
        ));
    }

    #[test]
    fn validate_label_path_specific_word_below_20_pct_rejected() {
        let mut files: Vec<FileInfo> = (0..9)
            .map(|i| FileInfo {
                relative_path: format!("pkg/module{}.ts", i),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            })
            .collect();
        files.push(FileInfo {
            relative_path: "pkg/zebra/special.ts".into(),
            header: String::new(),
            content: String::new(),
            symbol_preview: vec![],
        });
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert!(!validate_label_against_cluster("Zebra Handler", &refs));
    }

    #[test]
    fn validate_label_conceptual_zero_matches_accepted() {
        let files: Vec<FileInfo> = (0..10)
            .map(|i| FileInfo {
                relative_path: format!("pkg/module{}.ts", i),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            })
            .collect();
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert!(validate_label_against_cluster("Analytics Pipeline", &refs));
    }

    #[test]
    fn validate_label_small_cluster_always_true() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "a.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "b.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert!(validate_label_against_cluster(
            "Completely Wrong Label XYZ",
            &refs
        ));
    }

    // --- dominant_file_type tests ---

    #[test]
    fn dominant_file_type_all_test_files() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "src/a.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "src/b.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "src/c.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(dominant_file_type(&refs), Some("tests"));
    }

    #[test]
    fn dominant_file_type_mixed_no_majority() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "a.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "b.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "c.go".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "d.rs".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(dominant_file_type(&refs), None);
    }

    #[test]
    fn dominant_file_type_all_tsx() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "Header.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "Footer.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(dominant_file_type(&refs), Some("components"));
    }

    // --- derive_cluster_label tests ---

    #[test]
    fn derive_cluster_label_picks_architecture_layer() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "pkg/scheduling/delivery/http/handler.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/scheduling/delivery/http/routes.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/scheduling/delivery/http/middleware.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = derive_cluster_label(&refs);
        assert_eq!(result, Some("HTTP routes".to_string()));
    }

    #[test]
    fn derive_cluster_label_picks_test_files() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "pkg/auth/service/auth.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/auth/service/login.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/auth/service/token.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = derive_cluster_label(&refs);
        assert_eq!(result, Some("auth + login tests".to_string()));
    }

    #[test]
    fn derive_cluster_label_returns_none_for_shallow_paths() {
        let files: Vec<FileInfo> = vec![FileInfo {
            relative_path: "main.rs".into(),
            header: String::new(),
            content: String::new(),
            symbol_preview: vec![],
        }];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(derive_cluster_label(&refs), None);
    }

    #[test]
    fn derive_cluster_label_deep_common_prefix() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/domains/billing/service.ts".into(),
                header: "Billing service".into(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/billing/repo.ts".into(),
                header: "Billing repository".into(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/billing/handler.ts".into(),
                header: "Billing handler".into(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/billing/types.ts".into(),
                header: "Billing types".into(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/billing/index.ts".into(),
                header: "Billing exports".into(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert!(label.is_some(), "Expected Some label for billing files");
        let l = label.unwrap();
        assert!(
            l.to_lowercase().contains("billing"),
            "Expected 'billing' in label, got '{}'",
            l
        );
    }

    #[test]
    fn derive_cluster_label_generic_prefix_fallback() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "src/controllers/auth.ts".into(),
                header: "Auth controller".into(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "src/controllers/user.ts".into(),
                header: "User controller".into(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "src/controllers/billing.ts".into(),
                header: "Billing controller".into(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert!(label.is_some());
        let l = label.unwrap();
        assert!(
            l.to_lowercase().contains("controller"),
            "Expected 'controller' in label, got '{}'",
            l
        );
    }

    #[test]
    fn derive_cluster_label_empty() {
        let label = derive_cluster_label(&[]);
        assert_eq!(label, None);
    }

    #[test]
    fn derive_cluster_label_no_common_prefix() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "alpha/foo.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "beta/bar.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "gamma/baz.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, None);
    }

    #[test]
    fn derive_cluster_label_test_files() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/domains/scheduling/service/appointment.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/scheduling/service/availability.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/scheduling/domain/date-range.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(
            label,
            Some("appointment + availability + date-range tests".to_string())
        );
    }

    #[test]
    fn derive_cluster_label_repository_layer() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/domains/scheduling/repository/pg/appointment.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/scheduling/repository/pg/service.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/scheduling/repository/pg/availability.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("database queries".to_string()));
    }

    #[test]
    fn derive_cluster_label_service_layer() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/domains/billing/service/invoice.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/billing/service/payment.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/billing/service/refund.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("invoice + payment + refund logic".to_string()));
    }

    #[test]
    fn derive_cluster_label_http_routes() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/domains/billing/delivery/http/invoice-handler.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/billing/delivery/http/payment-handler.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/billing/delivery/http/routes.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("HTTP routes".to_string()));
    }

    #[test]
    fn derive_cluster_label_proto_files() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/proto/billing/invoice.proto".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/proto/billing/payment.proto".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/proto/billing/refund.proto".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("billing proto".to_string()));
    }

    #[test]
    fn derive_cluster_label_proto_files_different_domains() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "contracts/proto/iam/v1/auth.proto".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "contracts/proto/iam/v1/roles.proto".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "contracts/proto/billing/v1/invoice.proto".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("iam proto".to_string()));
    }

    #[test]
    fn derive_cluster_label_proto_files_flat_structure() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "proto/service.proto".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "proto/messages.proto".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("proto definitions".to_string()));
    }

    #[test]
    fn derive_cluster_label_barrel_exports() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/domains/billing/service/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/billing/repository/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/billing/domain/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("billing exports".to_string()));
    }

    #[test]
    fn derive_cluster_label_barrel_exports_no_domain() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "src/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "lib/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("barrel exports".to_string()));
    }

    #[test]
    fn derive_cluster_label_barrel_exports_different_domains() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/domains/scheduling/service/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/scheduling/repository/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/scheduling/domain/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("scheduling exports".to_string()));
    }

    #[test]
    fn derive_cluster_label_barrel_exports_iam_domain() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/domains/iam/delivery/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/domains/iam/service/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("iam exports".to_string()));
    }

    #[test]
    fn derive_cluster_label_sql_migrations() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/db/migrations/001_create_users.sql".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/db/migrations/002_create_orgs.sql".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/db/migrations/003_add_billing.sql".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert_eq!(label, Some("migrations".to_string()));
    }

    #[test]
    fn derive_cluster_label_filename_prefix_camelcase() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "app/components/SoapTranscription.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/components/SoapDiagnosis.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/components/SoapHistory.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert!(label.is_some());
        assert_eq!(label.unwrap(), "Soap");
    }

    #[test]
    fn derive_cluster_label_filename_prefix_too_short_after_trim() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "hooks/useCallback.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "hooks/useContext.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        if let Some(l) = &label {
            assert_ne!(
                l, "use",
                "Should not return 'use' (too short after camelCase trim)"
            );
        }
    }

    #[test]
    fn derive_cluster_label_header_blocklist() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "gen/a.ts".into(),
                header: "@generated by protobuf-ts".into(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "gen/b.ts".into(),
                header: "@generated by protobuf-ts".into(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "gen/c.ts".into(),
                header: "@generated by protobuf-ts".into(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        if let Some(l) = &label {
            assert!(
                !l.to_lowercase().contains("@generated"),
                "Should not use @generated as label: {}",
                l
            );
            assert!(l != "generated", "Should not use 'generated' as label");
        }
    }

    #[test]
    fn derive_cluster_label_temporal_workflows() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "pkg/delivery/temporal/workflows/appointment.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/delivery/temporal/workflows/membership.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/delivery/temporal/activities/node/scheduling.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert!(label.is_some());
        let l = label.unwrap().to_lowercase();
        assert!(
            l.contains("temporal") || l.contains("workflow"),
            "Expected temporal-related label, got '{}'",
            l
        );
    }

    #[test]
    fn derive_cluster_label_nats_consumers() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "pkg/delivery/nats/consumer.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/delivery/nats/event-schemas.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/delivery/nats/index.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        assert!(label.is_some());
        let l = label.unwrap().to_lowercase();
        assert!(
            l.contains("nats") || l.contains("consumer") || l.contains("event"),
            "Expected nats-related label, got '{}'",
            l
        );
    }

    #[test]
    fn derive_cluster_label_skips_nextjs_route_params() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "app/templates/[templateId]/page.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/templates/[templateId]/layout.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        if let Some(l) = &label {
            assert!(
                !l.contains("templateId"),
                "Should not use Next.js route param as label: {}",
                l
            );
        }
    }

    #[test]
    fn derive_cluster_label_skips_patient_route_param() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "app/people/[patientId]/overview.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/people/[patientId]/history.tsx".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        if let Some(l) = &label {
            assert!(
                !l.contains("patientId"),
                "Should not use Next.js route param as label: {}",
                l
            );
        }
    }

    #[test]
    fn derive_cluster_label_skips_catch_all_route_param() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "app/api/auth/[...nextauth]/route.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "app/api/auth/[...nextauth]/config.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let label = derive_cluster_label(&refs);
        if let Some(l) = &label {
            assert!(
                !l.contains("nextauth") && !l.contains("..."),
                "Should not use Next.js catch-all route param as label: {}",
                l
            );
        }
    }

    // --- deduplicate_sibling_labels tests ---

    #[test]
    fn deduplicate_labels_with_test_disambiguator() {
        let test_files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "pkg/service/auth.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/service/user.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/service/billing.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let impl_files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "pkg/service/auth.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/service/user.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let test_refs: Vec<&FileInfo> = test_files.iter().collect();
        let impl_refs: Vec<&FileInfo> = impl_files.iter().collect();
        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> =
            vec![(test_refs, None), (impl_refs, None)];
        let mut labels = vec!["service".to_string(), "service".to_string()];
        deduplicate_sibling_labels(&mut labels, &clusters);
        assert_ne!(labels[0], labels[1], "Labels should be disambiguated");
        assert!(
            labels[0].contains("test") || labels[1].contains("test"),
            "One should mention tests: {:?}",
            labels
        );
    }

    #[test]
    fn deduplicate_labels_no_dups_unchanged() {
        let f1: Vec<FileInfo> = vec![FileInfo {
            relative_path: "a.ts".into(),
            header: String::new(),
            content: String::new(),
            symbol_preview: vec![],
        }];
        let f2: Vec<FileInfo> = vec![FileInfo {
            relative_path: "b.ts".into(),
            header: String::new(),
            content: String::new(),
            symbol_preview: vec![],
        }];
        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> =
            vec![(f1.iter().collect(), None), (f2.iter().collect(), None)];
        let mut labels = vec!["alpha".to_string(), "beta".to_string()];
        deduplicate_sibling_labels(&mut labels, &clusters);
        assert_eq!(labels, vec!["alpha", "beta"]);
    }

    #[test]
    fn deduplicate_labels_same_disambiguator_gets_numbered() {
        let test_files1: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "pkg/a.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/b.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let test_files2: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "pkg/c.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/d.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> = vec![
            (test_files1.iter().collect(), None),
            (test_files2.iter().collect(), None),
        ];
        let mut labels = vec!["tests".to_string(), "tests".to_string()];
        deduplicate_sibling_labels(&mut labels, &clusters);
        assert_ne!(
            labels[0], labels[1],
            "Labels should differ after dedup: {:?}",
            labels
        );
    }

    #[test]
    fn deduplicate_replaces_not_appends() {
        let svc_files: Vec<FileInfo> = (0..5)
            .map(|i| FileInfo {
                relative_path: format!("pkg/service/svc{}.ts", i),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            })
            .collect();
        let test_files: Vec<FileInfo> = (0..5)
            .map(|i| FileInfo {
                relative_path: format!("pkg/service/svc{}.test.ts", i),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            })
            .collect();
        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> = vec![
            (svc_files.iter().collect(), None),
            (test_files.iter().collect(), None),
        ];
        let mut labels = vec!["service".to_string(), "service".to_string()];
        deduplicate_sibling_labels(&mut labels, &clusters);
        assert_ne!(labels[0], labels[1]);
        assert!(
            labels.iter().any(|l| l == "tests" || l.contains("test")),
            "Expected 'tests' label: {:?}",
            labels
        );
    }

    // --- find_label_disambiguator tests ---

    #[test]
    fn find_disambiguator_detects_tests() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "pkg/auth.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/user.test.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "pkg/billing.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        let result = find_label_disambiguator(&refs);
        assert_eq!(result, Some("auth + user tests".to_string()));
    }

    #[test]
    fn find_disambiguator_returns_none_for_mixed() {
        let files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "alpha/foo.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "beta/bar.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let refs: Vec<&FileInfo> = files.iter().collect();
        assert_eq!(find_label_disambiguator(&refs), None);
    }

    // --- extract_api_domain_from_filename tests ---

    #[test]
    fn extract_api_domain_scheduling() {
        assert_eq!(
            extract_api_domain_from_filename("getApiV1SchedulingServices200.ts"),
            Some("scheduling".to_string())
        );
    }

    #[test]
    fn extract_api_domain_auth() {
        assert_eq!(
            extract_api_domain_from_filename("postApiV1AuthLogin200.ts"),
            Some("auth".to_string())
        );
    }

    #[test]
    fn extract_api_domain_organizations() {
        assert_eq!(
            extract_api_domain_from_filename("patchApiV1OrganizationsMembers.ts"),
            Some("organizations".to_string())
        );
    }

    #[test]
    fn extract_api_domain_no_match() {
        assert_eq!(extract_api_domain_from_filename("index.ts"), None);
        assert_eq!(extract_api_domain_from_filename("utils.ts"), None);
    }

    #[test]
    fn extract_api_domain_short_domain() {
        assert_eq!(extract_api_domain_from_filename("getApiV1AbTest.ts"), None);
    }

    // --- describe_cluster_uniqueness tests ---

    #[test]
    fn describe_cluster_uniqueness_api_domains() {
        let scheduling_files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/libs/types/generated/getApiV1SchedulingServices200.ts"
                    .into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/libs/types/generated/getApiV1SchedulingAppointments200.ts"
                    .into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/libs/types/generated/postApiV1SchedulingSlots200.ts"
                    .into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let auth_files: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "packages/libs/types/generated/postApiV1AuthLogin200.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/libs/types/generated/postApiV1AuthRegister200.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "packages/libs/types/generated/getApiV1AuthSession200.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];

        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> = vec![
            (scheduling_files.iter().collect(), None),
            (auth_files.iter().collect(), None),
        ];
        let sibling_indices = vec![0usize, 1usize];

        let suffix0 = describe_cluster_uniqueness(&clusters[0].0, &clusters, &sibling_indices, 0);
        let suffix1 = describe_cluster_uniqueness(&clusters[1].0, &clusters, &sibling_indices, 1);

        assert_eq!(suffix0, "scheduling API types");
        assert_eq!(suffix1, "auth API types");
    }

    #[test]
    fn describe_cluster_uniqueness_same_api_domain_falls_back() {
        let files_a: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "gen/getApiV1SchedulingA.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "gen/getApiV1SchedulingB.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];
        let files_b: Vec<FileInfo> = vec![
            FileInfo {
                relative_path: "gen/postApiV1SchedulingC.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "gen/postApiV1SchedulingD.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "gen/postApiV1SchedulingE.ts".into(),
                header: String::new(),
                content: String::new(),
                symbol_preview: vec![],
            },
        ];

        let clusters: Vec<(Vec<&FileInfo>, Option<String>)> = vec![
            (files_a.iter().collect(), None),
            (files_b.iter().collect(), None),
        ];
        let sibling_indices = vec![0usize, 1usize];

        let suffix0 = describe_cluster_uniqueness(&clusters[0].0, &clusters, &sibling_indices, 0);
        assert_eq!(suffix0, "2 files");
    }

    // --- label_files tests ---

    #[tokio::test]
    async fn label_files_returns_headers() {
        let files = vec![
            FileInfo {
                relative_path: "src/auth.rs".to_string(),
                header: "header for src/auth.rs".to_string(),
                content: "content of src/auth.rs".to_string(),
                symbol_preview: vec![],
            },
            FileInfo {
                relative_path: "src/empty.rs".to_string(),
                header: String::new(),
                content: "fn main() {}".to_string(),
                symbol_preview: vec![],
            },
        ];
        let labels = label_files(&files).await;
        assert_eq!(labels[0], "header for src/auth.rs");
        assert_eq!(labels[1], "empty.rs"); // filename fallback
    }
}
