pub mod hybrid;
pub mod imports;
pub mod semantic;

/// Shared clustering parameters to avoid too-many-arguments lint.
pub(crate) struct ClusterParams {
    pub(crate) max_clusters: usize,
    pub(crate) min_clusters: usize,
    pub(crate) max_depth: usize,
}
