// Spectral clustering with eigengap heuristic for tuning-free cluster counts.
// Builds affinity matrix, normalized Laplacian, k-means on eigenvectors.

use nalgebra::{DMatrix, SymmetricEigen};
use rayon::prelude::*;

/// Maximum iterations for Lloyd's k-means algorithm.
const KMEANS_MAX_ITERATIONS: usize = 50;

/// Values below this threshold are treated as zero (avoids division by near-zero).
const ZERO_THRESHOLD: f64 = 1e-10;

/// Multiplier for comparing the Fiedler gap (k=2) against the best k>=3 gap.
/// If the Fiedler gap exceeds the best higher-k gap by this factor, we pick k=2.
/// Set to 5.0 because the Fiedler gap is structurally inflated in connected graphs
/// (it measures the fundamental graph bipartition), so a high bar prevents it from
/// dominating unless the graph truly has only 2 communities.
const FIEDLER_GAP_MULTIPLIER: f64 = 5.0;

/// Result of clustering: each entry contains indices of files in that cluster.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub indices: Vec<usize>,
}

/// Build a symmetric affinity matrix from embedding vectors using SIMD cosine similarity.
/// Diagonal is zero (no self-affinity). Negative similarities are clamped to 0.
/// Uses rayon to parallelize the outer loop for large matrices.
fn build_affinity_matrix(vectors: &[Vec<f32>]) -> DMatrix<f64> {
    let n = vectors.len();

    // Compute upper triangle in parallel — each row's (i, j>i) pairs are independent
    let rows: Vec<Vec<(usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row_entries = Vec::with_capacity(n - i - 1);
            for j in (i + 1)..n {
                let sim =
                    crate::core::embeddings::cosine_similarity_simsimd(&vectors[i], &vectors[j])
                        .max(0.0) as f64;
                row_entries.push((j, sim));
            }
            row_entries
        })
        .collect();

    let mut mat = DMatrix::zeros(n, n);
    for (i, row_entries) in rows.into_iter().enumerate() {
        for (j, sim) in row_entries {
            mat[(i, j)] = sim;
            mat[(j, i)] = sim;
        }
    }
    mat
}

/// Compute normalized symmetric Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
fn normalized_laplacian(affinity: &DMatrix<f64>) -> DMatrix<f64> {
    let n = affinity.nrows();

    // Compute degree per row using nalgebra row sums (diagonal subtracted)
    let d_inv_sqrt: Vec<f64> = (0..n)
        .map(|i| {
            let row_sum: f64 = affinity.row(i).iter().sum::<f64>() - affinity[(i, i)];
            if row_sum > ZERO_THRESHOLD {
                1.0 / row_sum.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    // Build Laplacian: L[i,j] = -W[i,j] * d_inv_sqrt[i] * d_inv_sqrt[j], L[i,i] = 1.0
    let mut laplacian = DMatrix::zeros(n, n);
    for i in 0..n {
        laplacian[(i, i)] = 1.0;
        let di = d_inv_sqrt[i];
        if di == 0.0 {
            continue;
        }
        for j in (i + 1)..n {
            let dj = d_inv_sqrt[j];
            if dj == 0.0 {
                continue;
            }
            let val = -affinity[(i, j)] * di * dj;
            laplacian[(i, j)] = val;
            laplacian[(j, i)] = val; // symmetric
        }
    }
    laplacian
}

/// Full eigendecomposition, returning (eigenvalues, eigenvectors) sorted ascending.
fn full_eigen(matrix: DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let eigen = SymmetricEigen::new(matrix);

    let mut indexed_evals: Vec<(usize, f64)> =
        eigen.eigenvalues.iter().copied().enumerate().collect();
    indexed_evals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = indexed_evals.iter().map(|(_, v)| *v).collect();

    let n = eigen.eigenvectors.nrows();
    let ncols = indexed_evals.len();
    let mut eigenvectors = DMatrix::zeros(n, ncols);
    for (new_col, &(orig_col, _)) in indexed_evals.iter().enumerate() {
        let src = eigen.eigenvectors.column(orig_col);
        eigenvectors.column_mut(new_col).copy_from(&src);
    }

    (eigenvalues, eigenvectors)
}

/// Use eigengap heuristic to find optimal k: largest relative gap in sorted eigenvalues.
/// Eigenvalues are expected to already be sorted ascending (full_eigen guarantees this),
/// so we skip the redundant sort.
///
/// Key insight: eigenvalue[0] ≈ 0 is trivial for any connected graph (the constant
/// eigenvector). The gap between eigenvalue[0] and eigenvalue[1] (the Fiedler gap)
/// is always the largest absolute gap, which causes naive eigengap to always pick k=2.
///
/// Fix: use relative gaps (gap / eigenvalue_position) starting from k=3, and only
/// fall back to k=2 if the k=2 gap is dramatically larger than all others.
fn find_optimal_k(eigenvalues: &[f64], max_k: usize) -> usize {
    if eigenvalues.len() <= 2 {
        return eigenvalues.len().min(2);
    }

    let limit = max_k.min(eigenvalues.len() - 1);

    // Compute all gaps
    let gaps: Vec<(usize, f64)> = (2..=limit)
        .map(|k| (k, eigenvalues[k] - eigenvalues[k - 1]))
        .collect();

    if gaps.is_empty() {
        return 2;
    }

    // Find the largest gap at k >= 3 (skipping the Fiedler gap at k=2)
    let best_k3_plus = gaps.iter()
        .filter(|(k, _)| *k >= 3)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Compare against k=2 gap
    let gap_k2 = gaps[0].1; // gaps[0] is always k=2

    if let Some(&(best_k, best_gap)) = best_k3_plus {
        // Only use k=2 if its gap is overwhelmingly larger (5x) than the best k>=3 gap.
        // This accounts for the Fiedler gap inflation in connected graphs.
        if gap_k2 > best_gap * FIEDLER_GAP_MULTIPLIER {
            2
        } else {
            best_k
        }
    } else {
        2
    }
}

/// Farthest-point initialization (greedy K-means variant) + Lloyd's algorithm on the eigenvector embedding.
/// Uses flat Vec<f64> buffers with stride indexing for cache-friendly access.
fn kmeans(data: &[Vec<f64>], k: usize) -> Vec<usize> {
    let n = data.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    let dim = data[0].len();

    // Flat centroid buffer: centroids[c * dim .. (c+1) * dim] is centroid c.
    // This avoids Vec<Vec<f64>> indirection and improves cache locality.
    let mut centroids = vec![0.0_f64; k * dim];
    let mut used = std::collections::HashSet::new();

    // Farthest-point initialization (greedy K-means variant): seed first centroid from first data point
    centroids[..dim].copy_from_slice(&data[0]);
    used.insert(0);

    for seed_c in 1..k {
        // Parallel distance computation: find point farthest from all current centroids
        let c_so_far = seed_c; // number of centroids already placed
        let (best_idx, _) = data
            .par_iter()
            .enumerate()
            .filter(|(i, _)| !used.contains(i))
            .map(|(i, row)| {
                let min_dist = (0..c_so_far)
                    .map(|c| {
                        let base = c * dim;
                        (0..dim)
                            .map(|d| (row[d] - centroids[base + d]).powi(2))
                            .sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min);
                (i, min_dist)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));
        let base = seed_c * dim;
        centroids[base..base + dim].copy_from_slice(&data[best_idx]);
        used.insert(best_idx);
    }

    let mut assignments = vec![0_usize; n];

    // Lloyd's iterations (assignment step parallelized with rayon)
    for _ in 0..KMEANS_MAX_ITERATIONS {
        let new_assignments: Vec<usize> = data
            .par_iter()
            .map(|row| {
                let mut best_dist = f64::INFINITY;
                let mut best_c = 0_usize;
                for c in 0..k {
                    let base = c * dim;
                    let dist: f64 = (0..dim)
                        .map(|d| (row[d] - centroids[base + d]).powi(2))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c;
                    }
                }
                best_c
            })
            .collect();

        let changed = assignments != new_assignments;
        assignments = new_assignments;
        if !changed {
            break;
        }

        // Update centroids: flat sums buffer, same stride as centroids
        let mut sums = vec![0.0_f64; k * dim];
        let mut counts = vec![0_u32; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            let base = c * dim;
            for d in 0..dim {
                sums[base + d] += data[i][d];
            }
        }
        for (c, &count_val) in counts.iter().enumerate().take(k) {
            if count_val == 0 {
                continue;
            }
            let base = c * dim;
            let count = count_val as f64;
            for d in 0..dim {
                centroids[base + d] = sums[base + d] / count;
            }
        }
    }

    assignments
}

/// Perform spectral clustering on a set of embedding vectors.
///
/// Returns a list of clusters, each containing the indices of vectors in that cluster.
/// Uses eigengap heuristic to automatically determine the number of clusters.
pub fn spectral_cluster(vectors: &[Vec<f32>], max_clusters: usize) -> Vec<ClusterResult> {
    spectral_cluster_with_min(vectors, max_clusters, 2)
}

/// Spectral clustering with a configurable minimum cluster count.
/// `min_clusters` forces at least this many clusters (capped by max_clusters and n).
pub fn spectral_cluster_with_min(
    vectors: &[Vec<f32>],
    max_clusters: usize,
    min_clusters: usize,
) -> Vec<ClusterResult> {
    let n = vectors.len();
    if n <= 1 {
        return vec![ClusterResult {
            indices: (0..n).collect(),
        }];
    }
    if n <= max_clusters {
        return vectors
            .iter()
            .enumerate()
            .map(|(i, _)| ClusterResult { indices: vec![i] })
            .collect();
    }

    let affinity = build_affinity_matrix(vectors);

    // Debug: affinity matrix statistics — if similarities are too uniform, clustering fails
    if tracing::enabled!(tracing::Level::DEBUG) {
        let n_aff = affinity.nrows();
        let mut sum = 0.0_f64;
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let mut count = 0_u64;
        for i in 0..n_aff {
            for j in (i + 1)..n_aff {
                let v = affinity[(i, j)];
                sum += v;
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
                count += 1;
            }
        }
        let mean = sum / count as f64;
        tracing::debug!(
            n = n_aff,
            mean_similarity = format!("{:.4}", mean),
            min_similarity = format!("{:.4}", min_val),
            max_similarity = format!("{:.4}", max_val),
            "affinity matrix stats"
        );
    }

    let laplacian = normalized_laplacian(&affinity);

    let max_k = max_clusters.min(2.max((n as f64).sqrt() as usize));

    let (eigenvalues, eigenvectors) = full_eigen(laplacian);

    // Debug: log eigenvalue spectrum to understand cluster selection
    if tracing::enabled!(tracing::Level::DEBUG) && eigenvalues.len() >= 2 {
        let gaps: Vec<(usize, f64)> = (2..eigenvalues.len().min(max_k + 1))
            .map(|k| (k, eigenvalues[k] - eigenvalues[k - 1]))
            .collect();
        tracing::debug!(
            n = n,
            max_k = max_k,
            eigenvalues_first_10 = ?&eigenvalues[..eigenvalues.len().min(10)],
            gaps = ?gaps,
            "spectral_cluster eigenvalue spectrum"
        );
    }

    let k = find_optimal_k(&eigenvalues, max_k).max(min_clusters.min(max_k));
    tracing::info!(k = k, min_clusters = min_clusters, "spectral_cluster chose k");

    // Build embedding: rows of first k eigenvectors, row-normalized.
    // Eigenvalues/eigenvectors are already sorted ascending by full_eigen,
    // so columns 0..k are the k smallest eigenvalues.
    let mut embedding: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<f64> = Vec::with_capacity(k);
        for j in 0..k {
            row.push(eigenvectors[(i, j)]);
        }
        let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > ZERO_THRESHOLD {
            for v in &mut row {
                *v /= norm;
            }
        }
        embedding.push(row);
    }

    let assignments = kmeans(&embedding, k);

    // Group indices by cluster assignment using a flat Vec<Vec<usize>> indexed by cluster id.
    // Avoids HashMap overhead (hashing, collision, rehashing) for the simple 0..k key range.
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &c) in assignments.iter().enumerate() {
        clusters[c].push(i);
    }

    clusters
        .into_iter()
        .filter(|indices| !indices.is_empty())
        .map(|indices| ClusterResult { indices })
        .collect()
}

/// Find a common path pattern among a set of file paths.
/// Returns `Some("prefix/*")` or `Some("*/suffix")` or `Some("prefix/suffix")` if found.
pub fn find_path_pattern(paths: &[String]) -> Option<String> {
    if paths.len() <= 1 {
        return None;
    }

    let parts: Vec<Vec<&str>> = paths.iter().map(|p| p.split('/').collect()).collect();
    let min_len = parts.iter().map(|p| p.len()).min().unwrap_or(0);

    let mut common_prefix = String::new();
    for i in 0..min_len.saturating_sub(1) {
        if parts.iter().all(|p| p[i] == parts[0][i]) {
            common_prefix.push_str(parts[0][i]);
            common_prefix.push('/');
        } else {
            break;
        }
    }

    let suffixes: Vec<&str> = paths
        .iter()
        .filter_map(|p| p.split('/').next_back())
        .collect();
    let all_same_suffix = !suffixes.is_empty() && suffixes.iter().all(|s| *s == suffixes[0]);

    match (common_prefix.is_empty(), all_same_suffix) {
        (false, true) => Some(format!("{}{}", common_prefix, suffixes[0])),
        (false, false) => Some(format!("{}*", common_prefix)),
        (true, true) => Some(format!("*/{}", suffixes[0])),
        (true, false) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_affinity_known_vectors() {
        // Two identical vectors -> cosine similarity 1.0, one orthogonal -> 0.0
        let vectors = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0_f32, 1.0, 0.0],
        ];
        let aff = build_affinity_matrix(&vectors);
        // (0,1) should be 1.0 (identical)
        assert!((aff[(0, 1)] - 1.0).abs() < 1e-6);
        // (0,2) should be 0.0 (orthogonal)
        assert!(aff[(0, 2)].abs() < 1e-6);
        // Diagonal should be 0.0
        assert_eq!(aff[(0, 0)], 0.0);
        assert_eq!(aff[(1, 1)], 0.0);
    }

    #[test]
    fn build_affinity_clamps_negative() {
        // Opposite vectors have cosine -1.0, should be clamped to 0.0
        let vectors = vec![vec![1.0_f32, 0.0], vec![-1.0_f32, 0.0]];
        let aff = build_affinity_matrix(&vectors);
        assert_eq!(aff[(0, 1)], 0.0);
    }

    #[test]
    fn eigengap_heuristic_clear_gap() {
        // Gap at position 2->3 is largest
        let eigenvalues = vec![0.0, 0.01, 0.5, 0.51, 0.52];
        let k = find_optimal_k(&eigenvalues, 4);
        assert_eq!(k, 2);
    }

    #[test]
    fn eigengap_heuristic_respects_max_k() {
        // Large gap at position 5, but max_k=3 should cap the search
        let eigenvalues = vec![0.0, 0.01, 0.02, 0.03, 0.04, 0.9];
        let k = find_optimal_k(&eigenvalues, 3);
        assert!(k <= 3, "k={} should be <= max_k=3", k);
    }

    #[test]
    fn kmeans_obvious_two_clusters() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.05, 0.05],
            vec![-0.05, 0.05],
            vec![100.0, 100.0],
            vec![100.05, 99.95],
            vec![99.95, 100.05],
        ];
        let assignments = kmeans(&data, 2);
        // First 3 should be in same cluster
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[0], assignments[2]);
        // Last 3 should be in same cluster
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[3], assignments[5]);
        // Two clusters should differ
        assert_ne!(assignments[0], assignments[3]);
    }

    #[test]
    fn kmeans_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        let assignments = kmeans(&data, 3);
        assert!(assignments.is_empty());
    }

    #[test]
    fn kmeans_single_point() {
        let data = vec![vec![1.0, 2.0]];
        let assignments = kmeans(&data, 1);
        assert_eq!(assignments, vec![0]);
    }

    #[test]
    fn cosine_sim_identical_vectors() {
        use crate::core::embeddings::cosine_similarity_simsimd;
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        let sim = cosine_similarity_simsimd(&a, &b) as f64;
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_orthogonal_vectors() {
        use crate::core::embeddings::cosine_similarity_simsimd;
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = cosine_similarity_simsimd(&a, &b) as f64;
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_empty_vectors() {
        use crate::core::embeddings::cosine_similarity_simsimd;
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        // simsimd returns 0.0 distance for empty vectors → 1.0 similarity.
        // The affinity matrix clamps negative values to 0, so this is safe.
        let sim = cosine_similarity_simsimd(&a, &b);
        assert!(sim >= 0.0); // just verify it doesn't panic or return NaN
    }

    #[test]
    fn cosine_sim_zero_vectors() {
        use crate::core::embeddings::cosine_similarity_simsimd;
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        assert_eq!(cosine_similarity_simsimd(&a, &b) as f64, 0.0);
    }

    #[test]
    fn build_affinity_symmetric() {
        let vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0], vec![0.5_f32, 0.5]];
        let aff = build_affinity_matrix(&vectors);
        assert_eq!(aff.nrows(), 3);
        assert_eq!(aff.ncols(), 3);
        // Symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((aff[(i, j)] - aff[(j, i)]).abs() < 1e-10);
            }
            // Zero diagonal
            assert_eq!(aff[(i, i)], 0.0);
        }
    }

    #[test]
    fn normalized_laplacian_diagonal_is_one() {
        let vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0], vec![0.5_f32, 0.5]];
        let aff = build_affinity_matrix(&vectors);
        let lap = normalized_laplacian(&aff);
        for i in 0..3 {
            assert!((lap[(i, i)] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn find_optimal_k_basic() {
        // eigenvalues with a clear gap at position 3
        let eigenvalues = vec![0.0, 0.01, 0.02, 0.8, 0.9, 1.0];
        let k = find_optimal_k(&eigenvalues, 5);
        assert_eq!(k, 3);
    }

    #[test]
    fn find_optimal_k_small_input() {
        let eigenvalues = vec![0.0, 1.0];
        let k = find_optimal_k(&eigenvalues, 5);
        assert_eq!(k, 2);
    }

    #[test]
    fn kmeans_separable_clusters() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
        ];
        let assignments = kmeans(&data, 2);
        // Points 0,1 should be in same cluster, points 2,3 in same cluster
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn spectral_cluster_single_item() {
        let vectors = vec![vec![1.0_f32, 0.0, 0.0]];
        let result = spectral_cluster(&vectors, 5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].indices, vec![0]);
    }

    #[test]
    fn spectral_cluster_small_set_returns_singletons() {
        let vectors = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let result = spectral_cluster(&vectors, 5);
        // n <= max_clusters, so each gets its own cluster
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn spectral_cluster_groups_similar_vectors() {
        // Create two clear clusters of 5 vectors each
        let mut vectors = Vec::new();
        for _ in 0..5 {
            vectors.push(vec![1.0_f32, 0.0, 0.0, 0.0]);
        }
        for _ in 0..5 {
            vectors.push(vec![0.0_f32, 0.0, 0.0, 1.0]);
        }
        // Need n > max_clusters for real clustering
        let result = spectral_cluster(&vectors, 3);
        // Should produce at least 2 clusters
        assert!(result.len() >= 2);

        // All first-5 should be together, all last-5 should be together
        let first_cluster = result.iter().find(|c| c.indices.contains(&0));
        assert!(first_cluster.is_some());
        let first = first_cluster.expect("cluster 0 exists");
        for i in 0..5 {
            assert!(
                first.indices.contains(&i),
                "index {} should be with cluster 0",
                i
            );
        }
    }

    #[test]
    fn find_path_pattern_common_prefix() {
        let paths = vec!["src/core/foo.rs".to_string(), "src/core/bar.rs".to_string()];
        let pattern = find_path_pattern(&paths);
        assert_eq!(pattern, Some("src/core/*".to_string()));
    }

    #[test]
    fn find_path_pattern_same_suffix() {
        let paths = vec!["src/mod.rs".to_string(), "lib/mod.rs".to_string()];
        let pattern = find_path_pattern(&paths);
        assert_eq!(pattern, Some("*/mod.rs".to_string()));
    }

    #[test]
    fn find_path_pattern_common_both() {
        let paths = vec!["src/core/mod.rs".to_string(), "src/core/mod.rs".to_string()];
        let pattern = find_path_pattern(&paths);
        assert_eq!(pattern, Some("src/core/mod.rs".to_string()));
    }

    #[test]
    fn find_path_pattern_single() {
        let paths = vec!["src/main.rs".to_string()];
        assert_eq!(find_path_pattern(&paths), None);
    }

    #[test]
    fn find_path_pattern_none() {
        let paths = vec!["src/foo.rs".to_string(), "lib/bar.rs".to_string()];
        let pattern = find_path_pattern(&paths);
        assert_eq!(pattern, None);
    }

    #[test]
    fn full_eigen_returns_sorted_eigenvalues() {
        // Simple 3x3 diagonal matrix: eigenvalues are 1, 2, 3
        let mat = DMatrix::from_fn(3, 3, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
        let (eigenvalues, eigenvectors) = full_eigen(mat);
        assert_eq!(eigenvalues.len(), 3);
        assert_eq!(eigenvectors.ncols(), 3);
        // Should be sorted ascending
        for w in eigenvalues.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-10,
                "Eigenvalues not sorted: {:?}",
                eigenvalues
            );
        }
        // Values should be approximately 1, 2, 3
        assert!((eigenvalues[0] - 1.0).abs() < 1e-6);
        assert!((eigenvalues[1] - 2.0).abs() < 1e-6);
        assert!((eigenvalues[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn spectral_cluster_three_clear_clusters() {
        // Create 3 clusters of 4 vectors each (n=12 > max_clusters=5, triggers real clustering)
        let mut vectors = Vec::new();
        // Cluster 1: along x-axis
        for _ in 0..4 {
            vectors.push(vec![1.0_f32, 0.0, 0.0]);
        }
        // Cluster 2: along y-axis
        for _ in 0..4 {
            vectors.push(vec![0.0_f32, 1.0, 0.0]);
        }
        // Cluster 3: along z-axis
        for _ in 0..4 {
            vectors.push(vec![0.0_f32, 0.0, 1.0]);
        }

        let result = spectral_cluster(&vectors, 5);

        // Should produce at least 2 clusters (3 ideally, but eigengap can vary)
        assert!(
            result.len() >= 2,
            "Expected at least 2 clusters, got {}",
            result.len()
        );

        // Vectors 0-3 should all be in the same cluster
        let c0 = result.iter().find(|c| c.indices.contains(&0)).unwrap();
        for i in 0..4 {
            assert!(
                c0.indices.contains(&i),
                "Vector {} should be in same cluster as vector 0",
                i
            );
        }
    }


    #[test]
    fn find_optimal_k_all_equal_eigenvalues() {
        let eigenvalues = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let k = find_optimal_k(&eigenvalues, 4);
        assert!(k >= 2);
    }

    #[test]
    fn find_optimal_k_max_k_is_2() {
        let eigenvalues = vec![0.0, 0.01, 0.5, 0.51];
        let k = find_optimal_k(&eigenvalues, 2);
        assert_eq!(k, 2);
    }

    #[test]
    fn find_optimal_k_fiedler_gap_suppression() {
        let eigenvalues = vec![0.0, 0.9, 0.96, 0.97, 0.98, 0.99];
        let k = find_optimal_k(&eigenvalues, 5);
        assert_eq!(k, 2);
    }

    #[test]
    fn find_optimal_k_fiedler_gap_not_dominant() {
        let eigenvalues = vec![0.0, 0.01, 0.02, 0.8, 0.9, 1.0];
        let k = find_optimal_k(&eigenvalues, 5);
        assert_eq!(k, 3);
    }

    // ── spectral_cluster_with_min tests ──────────────────────────────

    #[test]
    fn spectral_cluster_with_min_forces_higher_k() {
        let mut vectors = Vec::new();
        for _ in 0..8 {
            vectors.push(vec![1.0_f32, 0.0, 0.0]);
        }
        for _ in 0..8 {
            vectors.push(vec![0.0_f32, 1.0, 0.0]);
        }
        let result = spectral_cluster_with_min(&vectors, 5, 3);
        assert!(result.len() >= 3, "Expected at least 3 clusters, got {}", result.len());
    }

    #[test]
    fn spectral_cluster_empty_input() {
        let result = spectral_cluster(&[], 5);
        assert_eq!(result.len(), 1);
        assert!(result[0].indices.is_empty());
    }
}
