//! `linger` — pure-Rust sparse linear system solver library.
//!
//! Provides Krylov iterative methods, algebraic multigrid, and a rich
//! preconditioner library targeting large-scale FEA problems.
//! The core solver layer is `wasm32` compatible.

#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

pub mod core;
pub mod sparse;
pub mod precond;
pub mod iterative;
pub mod amg;
pub mod parallel;

#[cfg(feature = "wasm")]
pub mod wasm;

// ─── Re-exports ──────────────────────────────────────────────────────────────

pub use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
};

pub use crate::precond::{
    JacobiPrecond, SorPrecond, SsorPrecond, Ilu0Precond,
    IlukPrecond, IlutPrecond, Icc0Precond, SpaiPrecond,
    AdditivePrecond, MultiplicativePrecond,
};
pub use crate::iterative::{ConjugateGradient, Gmres, BiCgStab, Minres, Fgmres, Lgmres};
pub use crate::amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, CycleType, SmootherType};
pub use crate::sparse::{BsrMatrix, BsrBuilder};
pub use crate::parallel::{
    parallel_spmv, parallel_spmv_add,
    parallel_axpy, parallel_axpby,
    parallel_dot, parallel_norm2,
};
