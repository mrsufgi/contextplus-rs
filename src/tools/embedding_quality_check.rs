use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IssueKind {
    ZeroVector,
    NonFinite,
    DimensionMismatch,
    DuplicateVector,
}

#[derive(Debug, Clone)]
pub struct VectorIssue {
    pub path: PathBuf,
    pub kind: IssueKind,
    pub detail: String,
}

#[derive(Debug)]
pub struct QualityReport {
    pub total_vectors: usize,
    pub expected_dim: usize,
    pub issues: Vec<VectorIssue>,
    pub duplicate_groups: usize,
}

/// Normalize a vector to unit length for duplicate detection.
/// Returns None if the norm is too small (zero/near-zero vector).
fn normalize(v: &[f32]) -> Option<Vec<u64>> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-6 {
        return None;
    }
    // Represent as quantized bits for hashing equality. Normalize -0.0 →
    // +0.0 first so vectors that differ only in zero sign hash identically.
    let normalized: Vec<u64> = v
        .iter()
        .map(|x| {
            let scaled = x / norm;
            let canonical = if scaled == 0.0 { 0.0f32 } else { scaled };
            canonical.to_bits() as u64
        })
        .collect();
    Some(normalized)
}

/// Check the health of a set of embeddings.
///
/// # Arguments
/// * `vectors` - slice of (path, embedding) pairs
/// * `expected_dim` - the expected dimensionality of each vector
pub fn check_embeddings(vectors: &[(PathBuf, Vec<f32>)], expected_dim: usize) -> QualityReport {
    let mut issues: Vec<VectorIssue> = Vec::new();

    // Map from normalized vector representation → list of paths with that vector
    let mut normalized_map: HashMap<Vec<u64>, Vec<PathBuf>> = HashMap::new();

    for (path, vec) in vectors {
        // Check dimension mismatch
        if vec.len() != expected_dim {
            issues.push(VectorIssue {
                path: path.clone(),
                kind: IssueKind::DimensionMismatch,
                detail: format!("expected dim {}, got {}", expected_dim, vec.len()),
            });
            // Still continue to check other issues if possible, but skip norm/dup checks
            // since dimensionality is wrong — we can still check for non-finite values.
        }

        // Check for NaN / Inf
        let has_nonfinite = vec.iter().any(|x| !x.is_finite());
        if has_nonfinite {
            issues.push(VectorIssue {
                path: path.clone(),
                kind: IssueKind::NonFinite,
                detail: "vector contains NaN or Inf values".to_string(),
            });
            // Can't meaningfully compute norm or normalize; skip further checks
            continue;
        }

        // Check zero/near-zero norm (only for correctly-dimensioned vectors)
        if vec.len() == expected_dim {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-6 {
                issues.push(VectorIssue {
                    path: path.clone(),
                    kind: IssueKind::ZeroVector,
                    detail: format!("L2 norm {:.2e} < 1e-6", norm),
                });
                // Zero vectors cannot be meaningfully normalized; skip duplicate check
                continue;
            }

            // Build normalized fingerprint for duplicate detection
            if let Some(fingerprint) = normalize(vec) {
                normalized_map
                    .entry(fingerprint)
                    .or_default()
                    .push(path.clone());
            }
        }
    }

    // Find duplicate groups (groups with >= 2 members)
    let mut duplicate_groups = 0usize;
    for paths in normalized_map.values() {
        if paths.len() >= 2 {
            duplicate_groups += 1;
            for path in paths {
                issues.push(VectorIssue {
                    path: path.clone(),
                    kind: IssueKind::DuplicateVector,
                    detail: format!(
                        "vector is identical (after normalization) to {} other(s) in this group",
                        paths.len() - 1
                    ),
                });
            }
        }
    }

    QualityReport {
        total_vectors: vectors.len(),
        expected_dim,
        issues,
        duplicate_groups,
    }
}

/// Format a `QualityReport` into a short human-readable summary.
///
/// Lists up to the first 50 issues; appends `(+N more)` when truncated.
pub fn format_report(report: &QualityReport) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "Embedding quality report: {} vector(s), expected dim {}\n",
        report.total_vectors, report.expected_dim
    ));

    let total_issues = report.issues.len();
    if total_issues == 0 {
        out.push_str("  No issues found.\n");
    } else {
        out.push_str(&format!(
            "  Issues: {} total, {} duplicate group(s)\n",
            total_issues, report.duplicate_groups
        ));

        const MAX_LISTED: usize = 50;
        let shown = total_issues.min(MAX_LISTED);
        for issue in &report.issues[..shown] {
            out.push_str(&format!(
                "  [{:?}] {}: {}\n",
                issue.kind,
                issue.path.display(),
                issue.detail
            ));
        }
        if total_issues > MAX_LISTED {
            out.push_str(&format!("  (+{} more)\n", total_issues - MAX_LISTED));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    fn good_vec(dim: usize) -> Vec<f32> {
        // Simple unit vector along first axis
        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;
        v
    }

    // ── empty input ──────────────────────────────────────────────────────────

    #[test]
    fn empty_input_no_issues() {
        let report = check_embeddings(&[], 128);
        assert_eq!(report.total_vectors, 0);
        assert_eq!(report.issues.len(), 0);
        assert_eq!(report.duplicate_groups, 0);
    }

    // ── all-good vectors ─────────────────────────────────────────────────────

    #[test]
    fn all_good_no_issues() {
        let mut v1 = vec![0.0f32; 4];
        v1[0] = 1.0;
        let mut v2 = vec![0.0f32; 4];
        v2[1] = 1.0;
        let mut v3 = vec![0.0f32; 4];
        v3[2] = 1.0;

        let vecs = vec![(p("a"), v1), (p("b"), v2), (p("c"), v3)];
        let report = check_embeddings(&vecs, 4);
        assert_eq!(report.total_vectors, 3);
        assert_eq!(report.issues.len(), 0);
        assert_eq!(report.duplicate_groups, 0);
    }

    // ── zero vector ──────────────────────────────────────────────────────────

    #[test]
    fn zero_vector_flagged() {
        let zero = vec![0.0f32; 4];
        let vecs = vec![(p("zero.bin"), zero)];
        let report = check_embeddings(&vecs, 4);
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].kind, IssueKind::ZeroVector);
        assert_eq!(report.issues[0].path, p("zero.bin"));
    }

    // ── non-finite vector ────────────────────────────────────────────────────

    #[test]
    fn nan_vector_flagged() {
        let nan_vec = vec![f32::NAN, 0.0, 0.0, 0.0];
        let vecs = vec![(p("nan.bin"), nan_vec)];
        let report = check_embeddings(&vecs, 4);
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].kind, IssueKind::NonFinite);
    }

    #[test]
    fn inf_vector_flagged() {
        let inf_vec = vec![f32::INFINITY, 0.0, 0.0, 0.0];
        let vecs = vec![(p("inf.bin"), inf_vec)];
        let report = check_embeddings(&vecs, 4);
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].kind, IssueKind::NonFinite);
    }

    // ── dimension mismatch ───────────────────────────────────────────────────

    #[test]
    fn dimension_mismatch_flagged() {
        let short = vec![1.0f32, 0.0]; // dim 2, expected 4
        let vecs = vec![(p("short.bin"), short)];
        let report = check_embeddings(&vecs, 4);
        let dim_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.kind == IssueKind::DimensionMismatch)
            .collect();
        assert_eq!(dim_issues.len(), 1);
        assert!(dim_issues[0].detail.contains("expected dim 4"));
        assert!(dim_issues[0].detail.contains("got 2"));
    }

    // ── duplicate vectors ────────────────────────────────────────────────────

    #[test]
    fn two_identical_vectors_are_duplicates() {
        let v = good_vec(4);
        let vecs = vec![(p("a.bin"), v.clone()), (p("b.bin"), v)];
        let report = check_embeddings(&vecs, 4);
        assert_eq!(report.duplicate_groups, 1);
        let dup_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.kind == IssueKind::DuplicateVector)
            .collect();
        assert_eq!(dup_issues.len(), 2);
    }

    #[test]
    fn three_identical_plus_one_unique_one_group() {
        let v = good_vec(4);
        let mut unique = vec![0.0f32; 4];
        unique[3] = 1.0;
        let vecs = vec![
            (p("a.bin"), v.clone()),
            (p("b.bin"), v.clone()),
            (p("c.bin"), v),
            (p("d.bin"), unique),
        ];
        let report = check_embeddings(&vecs, 4);
        assert_eq!(report.duplicate_groups, 1);
        let dup_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.kind == IssueKind::DuplicateVector)
            .collect();
        assert_eq!(dup_issues.len(), 3);
    }

    // ── format_report ────────────────────────────────────────────────────────

    #[test]
    fn format_report_no_issues() {
        let report = check_embeddings(&[], 128);
        let text = format_report(&report);
        assert!(text.contains("No issues found"));
        assert!(text.contains("0 vector(s)"));
    }

    #[test]
    fn format_report_lists_issues_and_counts() {
        let zero = vec![0.0f32; 4];
        let vecs = vec![(p("zero.bin"), zero)];
        let report = check_embeddings(&vecs, 4);
        let text = format_report(&report);
        assert!(text.contains("Issues:"));
        assert!(text.contains("ZeroVector"));
        assert!(text.contains("zero.bin"));
    }

    #[test]
    fn format_report_truncates_at_50_and_shows_footer() {
        // Create 60 zero vectors to trigger truncation
        let vecs: Vec<(PathBuf, Vec<f32>)> = (0..60)
            .map(|i| (p(&format!("v{}.bin", i)), vec![0.0f32; 4]))
            .collect();
        let report = check_embeddings(&vecs, 4);
        let text = format_report(&report);
        assert!(
            text.contains("(+10 more)"),
            "expected truncation footer, got:\n{}",
            text
        );
    }
}
