//! Parallel sparse and dense vector operations (rayon back-end).
//!
//! All functions here are feature-gated behind `cfg(feature = "rayon")`.
//! When the feature is disabled, every function falls back to the scalar
//! implementation in `sparse::ops`.

pub mod rayon_ops;

pub use rayon_ops::{
    parallel_spmv, parallel_spmv_add,
    parallel_axpy, parallel_axpby,
    parallel_dot, parallel_norm2,
};
