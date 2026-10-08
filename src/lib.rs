pub mod cache;
pub mod config;
pub mod core;
pub mod error;
pub mod git;
#[cfg(feature = "memory-profile")]
pub mod memory_profile;
pub mod ref_index;
pub mod server;
pub mod server_adapters;
pub mod server_definitions;
pub mod server_helpers;
pub mod tools;
pub mod transport;

#[cfg(all(not(test), not(target_env = "msvc")))]
use tikv_jemallocator::Jemalloc;
#[cfg(all(not(test), not(target_env = "msvc")))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
#[cfg(all(not(test), not(target_env = "msvc")))]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static _RJEM_MALLOC_CONF: &[u8] =
    b"background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0\0";

#[cfg(test)]
mod alloc_probe;
#[cfg(test)]
#[global_allocator]
static ALLOCATOR: alloc_probe::CountingAllocator = alloc_probe::CountingAllocator;
#[cfg(test)]
mod test_logs;
