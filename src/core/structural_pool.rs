use std::sync::LazyLock;

// Structural reads and parsing must not queue behind semantic indexing.
pub(crate) static STRUCTURAL_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .num_threads(
            std::thread::available_parallelism()
                .map_or(1, std::num::NonZeroUsize::get)
                .min(4),
        )
        .thread_name(|index| format!("structural-{index}"))
        .build()
        .expect("failed to create structural thread pool")
});
