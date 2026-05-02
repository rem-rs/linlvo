//! BoomerAMG-compatible interface built on top of `linger`'s own AMG.
//!
//! Enable with the `hypre-rs` Cargo feature.
//!
//! # Current implementation status
//!
//! | Capability | Status |
//! |-----------|--------|
//! | `BoomerAmgPrecond` / `BoomerAmgConfig` | ✓ Implemented (wraps `AmgPrecond`) |
//! | AMS H(curl) auxiliary-space path | ✓ Available via `linger::AmsPrecond` |
//! | ADS H(div) auxiliary-space path | ✓ Available via `linger::AdsPrecond` |
//! | AIR nonsymmetric restriction | ✓ Available via `linger::amg::air` |
//! | ParCSR partitioned matrix bridge | ○ Planned (roadmap M1) |
//! | Device-policy hooks (GPU passthrough) | ○ Planned (roadmap M2) |
//!
//! This module exposes a naming convention that mirrors the HYPRE C library
//! (`BoomerAmgPrecond`, `BoomerAmgConfig`) so that code originally written
//! against HYPRE can be ported to this pure-Rust implementation with
//! minimal changes.
//!
//! # Example
//! ```rust,no_run
//! # #[cfg(feature = "hypre-rs")] {
//! use linger::hypre_rs::{BoomerAmgConfig, BoomerAmgPrecond};
//! use linger::sparse::CsrMatrix;
//! use linger::core::preconditioner::Preconditioner;
//!
//! let a: CsrMatrix<f64> = CsrMatrix::identity(64);
//! let config = BoomerAmgConfig::default();
//! let prec = BoomerAmgPrecond::new(&a, config);
//! # }
//! ```

use crate::amg::{AmgConfig, AmgHierarchy, AmgPrecond};
use crate::core::scalar::Scalar;
use crate::sparse::CsrMatrix;

// ─── Re-exports with HYPRE-compatible names ──────────────────────────────────

/// AMG preconditioner with BoomerAMG-compatible constructor API.
pub type BoomerAmgPrecond<T> = AmgPrecond<T>;

/// Configuration for [`BoomerAmgPrecond`].
pub type BoomerAmgConfig = AmgConfig;

/// Convenience factory — mirrors `HYPRE_BoomerAMGCreate` + `SetOperator`.
///
/// Constructs the AMG hierarchy from `a` using `config`, then wraps it in a
/// [`BoomerAmgPrecond`] ready for use with any [`KrylovSolver`].
pub fn create_boomer_amg<T: Scalar>(
    a: CsrMatrix<T>,
    config: BoomerAmgConfig,
) -> BoomerAmgPrecond<T> {
    let hier = AmgHierarchy::build(a, config);
    AmgPrecond::new(hier)
}
