// Shared utility functions used across ContextPlus core modules.

use std::borrow::Cow;

/// Clamp a value to [0.0, 1.0].
#[inline]
pub fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// Normalize a weight value to [0.0, 1.0] given min/max bounds.
/// Returns 0.0 if max == min to avoid division by zero.
#[inline]
pub fn normalize_weight(value: f64, min: f64, max: f64) -> f64 {
    if (max - min).abs() < f64::EPSILON {
        return 0.0;
    }
    clamp01((value - min) / (max - min))
}

/// Format a line range as "L{start}-L{end}" or "L{line}" if start == end.
pub fn format_line_range(start: usize, end: usize) -> String {
    if end > start {
        format!("L{}-L{}", start, end)
    } else {
        format!("L{}", start)
    }
}

/// Return the string unchanged if it fits within `max` bytes, otherwise
/// truncate to the nearest UTF-8 character boundary and return an owned copy.
///
/// This avoids a heap allocation in the common case where no truncation is needed.
pub fn truncate_if_over(s: &str, max: usize) -> Cow<'_, str> {
    if s.len() <= max {
        Cow::Borrowed(s)
    } else {
        // Find last valid char boundary at or before `max`
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        Cow::Owned(s[..end].to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp01_clamps() {
        assert_eq!(clamp01(-1.0), 0.0);
        assert_eq!(clamp01(2.0), 1.0);
        assert!((clamp01(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn normalize_weight_basic() {
        assert!((normalize_weight(5.0, 0.0, 10.0) - 0.5).abs() < 1e-10);
        assert_eq!(normalize_weight(0.0, 0.0, 10.0), 0.0);
        assert_eq!(normalize_weight(10.0, 0.0, 10.0), 1.0);
    }

    #[test]
    fn normalize_weight_same_min_max() {
        assert_eq!(normalize_weight(5.0, 5.0, 5.0), 0.0);
    }

    #[test]
    fn format_line_range_range() {
        assert_eq!(format_line_range(1, 10), "L1-L10");
    }

    #[test]
    fn format_line_range_single() {
        assert_eq!(format_line_range(5, 5), "L5");
        assert_eq!(format_line_range(5, 3), "L5"); // end < start → single
    }

    #[test]
    fn truncate_if_over_short_returns_borrowed() {
        let s = "hello";
        let result = truncate_if_over(s, 10);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(&*result, "hello");
    }

    #[test]
    fn truncate_if_over_exact_length_returns_borrowed() {
        let s = "hello";
        let result = truncate_if_over(s, 5);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn truncate_if_over_long_returns_owned() {
        let s = "hello world";
        let result = truncate_if_over(s, 5);
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(&*result, "hello");
    }

    #[test]
    fn truncate_if_over_respects_utf8_boundary() {
        // "café" is 5 bytes (c=1, a=1, f=1, é=2)
        let s = "café";
        assert_eq!(s.len(), 5);
        // max=4 should not split the 'é' (bytes 3-4), so result is "caf"
        let result = truncate_if_over(s, 4);
        assert_eq!(&*result, "caf");
    }
}
