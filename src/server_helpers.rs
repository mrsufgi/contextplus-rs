// Argument extraction helpers — try snake_case first, fall back to camelCase.
// Used across all tool handlers to accept both naming conventions.

use serde_json::Value;

/// Extract a `&str` from args, trying `snake_key` first then `camel_key`.
pub fn get_str_compat<'a>(
    args: &'a serde_json::Map<String, Value>,
    snake_key: &str,
    camel_key: &str,
) -> Option<&'a str> {
    args.get(snake_key)
        .or_else(|| args.get(camel_key))
        .and_then(|v| v.as_str())
}

/// Extract a `usize` from args, trying `snake_key` first then `camel_key`.
pub fn get_usize_compat(
    args: &serde_json::Map<String, Value>,
    snake_key: &str,
    camel_key: &str,
) -> Option<usize> {
    args.get(snake_key)
        .or_else(|| args.get(camel_key))
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
}

/// Extract a `bool` from args, trying `snake_key` first then `camel_key`.
pub fn get_bool_compat(
    args: &serde_json::Map<String, Value>,
    snake_key: &str,
    camel_key: &str,
) -> Option<bool> {
    args.get(snake_key)
        .or_else(|| args.get(camel_key))
        .and_then(|v| v.as_bool())
}

/// Extract a `f64` from args, trying `snake_key` first then `camel_key`.
pub fn get_f64_compat(
    args: &serde_json::Map<String, Value>,
    snake_key: &str,
    camel_key: &str,
) -> Option<f64> {
    args.get(snake_key)
        .or_else(|| args.get(camel_key))
        .and_then(|v| v.as_f64())
}

/// Extract a `Vec<String>` from an array arg, trying `snake_key` first then `camel_key`.
pub fn get_string_array_compat(
    args: &serde_json::Map<String, Value>,
    snake_key: &str,
    camel_key: &str,
) -> Option<Vec<String>> {
    args.get(snake_key)
        .or_else(|| args.get(camel_key))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(pairs: &[(&str, serde_json::Value)]) -> serde_json::Map<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn get_str_compat_prefers_snake() {
        let args = map(&[("my_key", json!("snake")), ("myKey", json!("camel"))]);
        assert_eq!(get_str_compat(&args, "my_key", "myKey"), Some("snake"));
    }

    #[test]
    fn get_str_compat_falls_back_to_camel() {
        let args = map(&[("myKey", json!("camel"))]);
        assert_eq!(get_str_compat(&args, "my_key", "myKey"), Some("camel"));
    }

    #[test]
    fn get_str_compat_returns_none_when_missing() {
        let args = map(&[]);
        assert_eq!(get_str_compat(&args, "my_key", "myKey"), None);
    }

    #[test]
    fn get_usize_compat_works() {
        let args = map(&[("top_k", json!(5))]);
        assert_eq!(get_usize_compat(&args, "top_k", "topK"), Some(5));
        let args = map(&[("topK", json!(10))]);
        assert_eq!(get_usize_compat(&args, "top_k", "topK"), Some(10));
    }

    #[test]
    fn get_bool_compat_works() {
        let args = map(&[("require_match", json!(true))]);
        assert_eq!(
            get_bool_compat(&args, "require_match", "requireMatch"),
            Some(true)
        );
        let args = map(&[("requireMatch", json!(false))]);
        assert_eq!(
            get_bool_compat(&args, "require_match", "requireMatch"),
            Some(false)
        );
    }

    #[test]
    fn get_f64_compat_works() {
        let args = map(&[("semantic_weight", json!(0.75))]);
        let val = get_f64_compat(&args, "semantic_weight", "semanticWeight").unwrap();
        assert!((val - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn get_string_array_compat_works() {
        let args = map(&[("include_kinds", json!(["function", "method"]))]);
        let result = get_string_array_compat(&args, "include_kinds", "includeKinds").unwrap();
        assert_eq!(result, vec!["function", "method"]);
    }

    #[test]
    fn get_string_array_compat_camel_fallback() {
        let args = map(&[("includeKinds", json!(["class"]))]);
        let result = get_string_array_compat(&args, "include_kinds", "includeKinds").unwrap();
        assert_eq!(result, vec!["class"]);
    }
}
