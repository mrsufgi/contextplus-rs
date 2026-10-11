use contextplus_rs as _;
use tikv_jemalloc_ctl::{epoch, stats};

#[test]
fn jemalloc_stats_rise_and_fall_with_large_allocation() {
    let e = epoch::mib().unwrap();
    e.advance().unwrap();
    let before = stats::allocated::mib().unwrap().read().unwrap();

    let large: Vec<u8> = vec![1u8; 16 * 1024 * 1024];
    e.advance().unwrap();
    let during = stats::allocated::mib().unwrap().read().unwrap();

    assert!(
        during >= before + 15 * 1024 * 1024,
        "jemalloc stats.allocated must rise by >= 15 MiB on a 16 MiB alloc: before={before} during={during}"
    );

    drop(large);
    e.advance().unwrap();
    let after = stats::allocated::mib().unwrap().read().unwrap();

    assert!(
        after < during,
        "jemalloc stats.allocated must fall after drop: during={during} after={after}"
    );
}
