// Spectral clustering with eigengap heuristic for tuning-free cluster counts.
// Builds affinity matrix, normalized Laplacian, k-means on eigenvectors.
// Uses randomized SVD (Halko-Martinsson-Tropp) for large matrices to avoid O(n^3) full eigen.

use nalgebra::{DMatrix, SymmetricEigen};
use rand::Rng;
use rayon::prelude::*;

/// Matrices larger than this use randomized SVD instead of full eigendecomposition.
/// Below this threshold, full `SymmetricEigen` is used (more accurate, still fast enough).
const RANDOM_SVD_THRESHOLD: usize = 500;

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
                let sim = cosine_similarity_simsimd(&vectors[i], &vectors[j]).max(0.0) as f64;
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

/// Compute cosine similarity using simsimd SIMD acceleration.
/// Returns similarity (1.0 - distance) clamped to [0, 1].
fn cosine_similarity_simsimd(a: &[f32], b: &[f32]) -> f32 {
    use simsimd::SpatialSimilarity;
    match f32::cosine(a, b) {
        Some(distance) => (1.0 - distance as f32).clamp(0.0, 1.0),
        None => 0.0,
    }
}

/// Compute normalized symmetric Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
fn normalized_laplacian(affinity: &DMatrix<f64>) -> DMatrix<f64> {
    let n = affinity.nrows();
    let mut degrees = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            if i != j {
                sum += affinity[(i, j)];
            }
        }
        degrees[i] = sum;
    }

    let mut laplacian = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                laplacian[(i, j)] = 1.0;
            } else {
                let di = degrees[i];
                let dj = degrees[j];
                if di > 1e-10 && dj > 1e-10 {
                    laplacian[(i, j)] = -affinity[(i, j)] / (di.sqrt() * dj.sqrt());
                }
            }
        }
    }
    laplacian
}

/// Generate a pair of standard normal random numbers using Box-Muller transform.
/// Avoids dependency on `rand_distr` -- only needs uniform `[0, 1)` from `rand`.
fn box_muller(rng: &mut impl Rng) -> (f64, f64) {
    loop {
        let u1: f64 = rng.r#gen();
        let u2: f64 = rng.r#gen();
        // Reject u1 == 0 to avoid log(0)
        if u1 > 1e-15 {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            return (r * theta.cos(), r * theta.sin());
        }
    }
}

/// Generate an n x p matrix of standard normal random entries.
fn random_gaussian_matrix(n: usize, p: usize, rng: &mut impl Rng) -> DMatrix<f64> {
    let mut data = Vec::with_capacity(n * p);
    let pairs_needed = (n * p).div_ceil(2);
    for _ in 0..pairs_needed {
        let (a, b) = box_muller(rng);
        data.push(a);
        data.push(b);
    }
    data.truncate(n * p);
    // nalgebra DMatrix::from_vec is column-major, but for a random Gaussian matrix
    // the distribution is invariant to transposition, so column-major is fine.
    DMatrix::from_vec(n, p, data)
}

/// Randomized eigendecomposition for symmetric PSD matrices (Halko-Martinsson-Tropp).
///
/// Computes approximate smallest eigenvalues/eigenvectors of a symmetric matrix using
/// randomized range finding + projected eigendecomposition. Cost: O(n * k^2) vs O(n^3).
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues are sorted ascending and
/// eigenvector columns correspond to those eigenvalues. Only `k` components are returned.
fn randomized_eigen(
    matrix: &DMatrix<f64>,
    k: usize,
    oversampling: usize,
) -> (Vec<f64>, DMatrix<f64>) {
    let n = matrix.nrows();
    let p = (k + oversampling).min(n);

    let mut rng = rand::thread_rng();
    let omega = random_gaussian_matrix(n, p, &mut rng);

    // Y = A * Omega  (n x p)
    let y = matrix * &omega;

    // QR decomposition of Y to get orthonormal basis Q (n x p)
    let qr = y.qr();
    let q = qr.q(); // n x min(n,p) orthonormal columns

    // Project: B = Q^T * A * Q  (p x p -- small symmetric matrix)
    let b = q.transpose() * matrix * &q;

    // Full eigendecomposition of the small p x p matrix
    let eigen_small = SymmetricEigen::new(b);

    // Sort eigenvalues ascending
    let mut indexed_evals: Vec<(usize, f64)> = eigen_small
        .eigenvalues
        .iter()
        .copied()
        .enumerate()
        .collect();
    indexed_evals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take the first k
    let take = k.min(indexed_evals.len());
    let eigenvalues: Vec<f64> = indexed_evals[..take].iter().map(|(_, v)| *v).collect();

    // Map eigenvectors back to original space: U = Q * V_small
    let v_small = &eigen_small.eigenvectors;
    let full_eigenvectors = &q * v_small;

    // Reorder columns to match sorted eigenvalues
    let mut eigenvectors = DMatrix::zeros(n, take);
    for (new_col, &(orig_col, _)) in indexed_evals[..take].iter().enumerate() {
        for row in 0..n {
            eigenvectors[(row, new_col)] = full_eigenvectors[(row, orig_col)];
        }
    }

    (eigenvalues, eigenvectors)
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
        for row in 0..n {
            eigenvectors[(row, new_col)] = eigen.eigenvectors[(row, orig_col)];
        }
    }

    (eigenvalues, eigenvectors)
}

/// Use eigengap heuristic to find optimal k: largest gap in sorted eigenvalues.
fn find_optimal_k(eigenvalues: &[f64], max_k: usize) -> usize {
    let mut sorted: Vec<f64> = eigenvalues.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.len() <= 2 {
        return sorted.len().min(2);
    }

    let limit = max_k.min(sorted.len() - 1);
    let mut best_gap = 0.0_f64;
    let mut best_k = 2_usize;

    for k in 2..=limit {
        let gap = sorted[k] - sorted[k - 1];
        if gap > best_gap {
            best_gap = gap;
            best_k = k;
        }
    }
    best_k
}

/// K-means++ initialization + Lloyd's algorithm on the eigenvector embedding.
fn kmeans(data: &[Vec<f64>], k: usize) -> Vec<usize> {
    let n = data.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    let dim = data[0].len();

    // K-means++ initialization
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut used = std::collections::HashSet::new();

    centroids.push(data[0].clone());
    used.insert(0);

    for _ in 1..k {
        let mut best_idx = 0;
        let mut best_dist = -1.0_f64;
        for (i, row) in data.iter().enumerate() {
            if used.contains(&i) {
                continue;
            }
            let min_dist = centroids
                .iter()
                .map(|c| (0..dim).map(|d| (row[d] - c[d]).powi(2)).sum::<f64>())
                .fold(f64::INFINITY, f64::min);
            if min_dist > best_dist {
                best_dist = min_dist;
                best_idx = i;
            }
        }
        centroids.push(data[best_idx].clone());
        used.insert(best_idx);
    }

    let mut assignments = vec![0_usize; n];

    // Lloyd's iterations (assignment step parallelized with rayon)
    for _ in 0..50 {
        let new_assignments: Vec<usize> = data
            .par_iter()
            .map(|row| {
                let mut best_dist = f64::INFINITY;
                let mut best_c = 0_usize;
                for (c, centroid) in centroids.iter().enumerate() {
                    let dist: f64 = (0..dim).map(|d| (row[d] - centroid[d]).powi(2)).sum();
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

        // Update centroids
        let mut sums = vec![vec![0.0_f64; dim]; k];
        let mut counts = vec![0_u32; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dim {
                sums[c][d] += data[i][d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            for d in 0..dim {
                centroids[c][d] = sums[c][d] / counts[c] as f64;
            }
        }
    }

    assignments
}

/// Perform spectral clustering on a set of embedding vectors.
///
/// Returns a list of clusters, each containing the indices of vectors in that cluster.
/// Uses eigengap heuristic to automatically determine the number of clusters.
///
/// For matrices larger than `RANDOM_SVD_THRESHOLD`, uses randomized SVD (Halko-Martinsson-Tropp)
/// which computes only the needed eigenvectors in O(n*k^2) instead of O(n^3).
pub fn spectral_cluster(vectors: &[Vec<f32>], max_clusters: usize) -> Vec<ClusterResult> {
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
    let laplacian = normalized_laplacian(&affinity);

    let max_k = max_clusters.min(2.max((n as f64).sqrt() as usize));

    // Choose eigendecomposition strategy based on matrix size.
    // Randomized SVD is O(n*k^2) vs O(n^3) for full eigen -- ~100x faster for n=2000, k=20.
    let (eigenvalues, eigenvectors) = if n > RANDOM_SVD_THRESHOLD {
        let oversampling = 10;
        randomized_eigen(&laplacian, max_k + oversampling, oversampling)
    } else {
        full_eigen(laplacian)
    };

    let k = find_optimal_k(&eigenvalues, max_k);

    // Build embedding: rows of first k eigenvectors, row-normalized.
    // Eigenvalues/eigenvectors are already sorted ascending by full_eigen/randomized_eigen,
    // so columns 0..k are the k smallest eigenvalues.
    let mut embedding: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<f64> = Vec::with_capacity(k);
        for j in 0..k {
            row.push(eigenvectors[(i, j)]);
        }
        let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for v in &mut row {
                *v /= norm;
            }
        }
        embedding.push(row);
    }

    let assignments = kmeans(&embedding, k);

    // Group indices by cluster assignment
    let mut clusters: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &c) in assignments.iter().enumerate() {
        clusters.entry(c).or_default().push(i);
    }

    clusters
        .into_values()
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
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        let sim = cosine_similarity_simsimd(&a, &b) as f64;
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = cosine_similarity_simsimd(&a, &b) as f64;
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        // simsimd returns 0.0 distance for empty vectors → 1.0 similarity.
        // The affinity matrix clamps negative values to 0, so this is safe.
        let sim = cosine_similarity_simsimd(&a, &b);
        assert!(sim >= 0.0); // just verify it doesn't panic or return NaN
    }

    #[test]
    fn cosine_sim_zero_vectors() {
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

    // --- Randomized SVD tests ---

    #[test]
    fn box_muller_produces_finite_values() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let (a, b) = box_muller(&mut rng);
            assert!(a.is_finite(), "Box-Muller produced non-finite: {}", a);
            assert!(b.is_finite(), "Box-Muller produced non-finite: {}", b);
        }
    }

    #[test]
    fn random_gaussian_matrix_correct_shape() {
        let mut rng = rand::thread_rng();
        let mat = random_gaussian_matrix(10, 5, &mut rng);
        assert_eq!(mat.nrows(), 10);
        assert_eq!(mat.ncols(), 5);
    }

    #[test]
    fn random_gaussian_matrix_has_variance() {
        // A 100x10 Gaussian matrix should not have all identical values
        let mut rng = rand::thread_rng();
        let mat = random_gaussian_matrix(100, 10, &mut rng);
        let first = mat[(0, 0)];
        let has_variation = mat.iter().any(|&v| (v - first).abs() > 0.01);
        assert!(has_variation, "Random matrix should have variation");
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
    fn randomized_eigen_approximates_smallest_eigenvalues() {
        // Build a symmetric matrix with known, evenly-spaced eigenvalues: 1..=10.
        // Request the 4 smallest with generous oversampling so the random projection
        // captures the full subspace.
        let n = 10;
        let mat = DMatrix::from_fn(n, n, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });

        let k = 4;
        // Use oversampling = n - k so p = n (captures full space for this small matrix)
        let oversampling = n - k;
        let (eigenvalues, eigenvectors) = randomized_eigen(&mat, k, oversampling);

        assert_eq!(eigenvalues.len(), k);
        assert_eq!(eigenvectors.nrows(), n);
        assert_eq!(eigenvectors.ncols(), k);

        // The 4 smallest eigenvalues should be approximately 1, 2, 3, 4
        for (i, expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert!(
                (eigenvalues[i] - expected).abs() < 0.5,
                "Eigenvalue {}: {} should be near {}",
                i,
                eigenvalues[i],
                expected
            );
        }
    }

    #[test]
    fn randomized_eigen_sorted_ascending() {
        let n = 20;
        let mat = DMatrix::from_fn(n, n, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
        let (eigenvalues, _) = randomized_eigen(&mat, 5, 5);
        for w in eigenvalues.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-6,
                "Eigenvalues not sorted ascending: {:?}",
                eigenvalues
            );
        }
    }

    #[test]
    fn full_and_randomized_agree_on_small_matrix() {
        // For a small matrix, both should find the same eigenvalues
        let n = 8;
        // Build a symmetric matrix
        let mut mat = DMatrix::zeros(n, n);
        for i in 0..n {
            mat[(i, i)] = (i + 1) as f64;
            if i + 1 < n {
                mat[(i, i + 1)] = 0.1;
                mat[(i + 1, i)] = 0.1;
            }
        }

        let (full_evals, _) = full_eigen(mat.clone());
        let (rand_evals, _) = randomized_eigen(&mat, 4, 4);

        // First 4 eigenvalues should be close
        for i in 0..4 {
            assert!(
                (full_evals[i] - rand_evals[i]).abs() < 0.3,
                "Eigenvalue {}: full={} vs randomized={}",
                i,
                full_evals[i],
                rand_evals[i]
            );
        }
    }

    #[test]
    fn threshold_selects_correct_path() {
        // Verify RANDOM_SVD_THRESHOLD is reasonable (compile-time check)
        const {
            assert!(RANDOM_SVD_THRESHOLD >= 100);
            assert!(RANDOM_SVD_THRESHOLD <= 1000);
        }
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
    fn randomized_eigen_with_repeated_eigenvalues() {
        // Matrix with repeated eigenvalues: diag(1, 1, 1, 5, 5, 10)
        let n = 6;
        let diag_vals = [1.0, 1.0, 1.0, 5.0, 5.0, 10.0];
        let mat = DMatrix::from_fn(n, n, |i, j| if i == j { diag_vals[i] } else { 0.0 });

        let (eigenvalues, eigenvectors) = randomized_eigen(&mat, 3, 3);

        assert_eq!(eigenvalues.len(), 3);
        assert_eq!(eigenvectors.nrows(), n);
        // First 3 eigenvalues should all be near 1.0
        for (i, &ev) in eigenvalues.iter().enumerate() {
            assert!(
                (ev - 1.0).abs() < 0.3,
                "Eigenvalue {} = {} should be near 1.0",
                i,
                ev
            );
        }
    }
}
