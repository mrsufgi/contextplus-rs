// Spectral clustering with eigengap heuristic for tuning-free cluster counts.
// Builds affinity matrix, normalized Laplacian, k-means on eigenvectors.

use nalgebra::{DMatrix, SymmetricEigen};

/// Result of clustering: each entry contains indices of files in that cluster.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub indices: Vec<usize>,
}

/// Build a symmetric affinity matrix from embedding vectors using cosine similarity.
/// Diagonal is zero (no self-affinity). Negative similarities are clamped to 0.
fn build_affinity_matrix(vectors: &[Vec<f32>]) -> DMatrix<f64> {
    let n = vectors.len();
    let mut mat = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_sim(&vectors[i], &vectors[j]).max(0.0);
            mat[(i, j)] = sim;
            mat[(j, i)] = sim;
        }
    }
    mat
}

/// Compute cosine similarity between two f32 vectors, returning f64.
fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for i in 0..len {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-15 { 0.0 } else { dot / denom }
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

    // Lloyd's iterations
    for _ in 0..50 {
        let mut changed = false;
        for i in 0..n {
            let mut best_dist = f64::INFINITY;
            let mut best_c = 0_usize;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist: f64 = (0..dim).map(|d| (data[i][d] - centroid[d]).powi(2)).sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }
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
    let eigen = SymmetricEigen::new(laplacian);

    let eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    let eigenvectors = &eigen.eigenvectors;

    let max_k = max_clusters.min(2.max((n as f64).sqrt() as usize));
    let k = find_optimal_k(&eigenvalues, max_k);

    // Sort eigenvalue indices by value (ascending) and take first k eigenvectors
    let mut sorted_indices: Vec<usize> = (0..eigenvalues.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build embedding: rows of first k eigenvectors, row-normalized
    let mut embedding: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<f64> = Vec::with_capacity(k);
        for j in 0..k {
            row.push(eigenvectors[(i, sorted_indices[j])]);
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
    fn cosine_sim_identical_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        let sim = cosine_sim(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = cosine_sim(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_sim(&a, &b), 0.0);
    }

    #[test]
    fn cosine_sim_zero_vectors() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        assert_eq!(cosine_sim(&a, &b), 0.0);
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
}
