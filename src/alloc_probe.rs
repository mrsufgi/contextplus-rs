//! Test-only allocator that counts the heap bytes each thread holds, charged at
//! glibc chunk granularity so many small allocations cost what they really do.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

pub(crate) struct CountingAllocator;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

fn chunk(size: usize) -> isize {
    (size + 8).div_ceil(16).max(2) as isize * 16
}

fn charge(bytes: isize) {
    let _ = LIVE.try_with(|live| {
        let now = live.get() + bytes;
        live.set(now);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        charge(chunk(layout.size()));
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        charge(chunk(layout.size()));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        charge(-chunk(layout.size()));
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        charge(chunk(new_size) - chunk(layout.size()));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Runs `f` and returns its result with the heap bytes this thread gained
/// while it ran, which is what the result holds when `f` allocates only here.
pub(crate) fn retained_bytes<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = LIVE.with(Cell::get);
    let result = f();
    let after = LIVE.with(Cell::get);
    (result, after.saturating_sub(before).max(0) as usize)
}

/// Runs `f` and returns its result with the most heap bytes this thread held
/// above its starting point while it ran.
pub(crate) fn peak_bytes<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = LIVE.with(Cell::get);
    let outer_peak = PEAK.with(|peak| peak.replace(before));
    let result = f();
    let peak = PEAK.with(|peak| peak.replace(outer_peak.max(peak.get())));
    (result, peak.saturating_sub(before).max(0) as usize)
}
