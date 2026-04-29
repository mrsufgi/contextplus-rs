// When the `lapack` feature is enabled, force-link the system OpenBLAS via
// openblas-src so its `cargo:rustc-link-lib=openblas` directive is emitted.
// Required for `lapack-custom` mode in nalgebra-lapack.
#[cfg(feature = "lapack")]
extern crate openblas_src;

pub mod cache;
pub mod config;
pub mod core;
pub mod error;
pub mod git;
pub mod server;
pub mod server_adapters;
pub mod server_definitions;
pub mod server_helpers;
pub mod tools;
