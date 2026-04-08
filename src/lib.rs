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
pub mod eigen;
pub mod direct;
pub mod builder;

#[cfg(feature = "wasm")]
pub mod wasm;

// ─── Re-exports ──────────────────────────────────────────────────────────────

pub use crate::core::{
    error::SolverError,
    operator::{LinearOperator, TransposeOperator},
    preconditioner::Preconditioner,
    scalar::{Scalar, ComplexScalar},
    solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
};

pub use num_complex::Complex;

pub use crate::precond::{
    JacobiPrecond, SorPrecond, SsorPrecond, Ilu0Precond,
    IlukPrecond, IlutPrecond, Icc0Precond, SpaiPrecond,
    AdditivePrecond, MultiplicativePrecond,
};
pub use crate::iterative::{ConjugateGradient, Gmres, BiCgStab, Minres, Fgmres, Lgmres};
pub use crate::amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, CycleType, SmootherType};
pub use crate::sparse::{BsrMatrix, BsrBuilder, read_matrix_market, read_matrix_market_coo, read_matrix_market_str, read_matrix_market_coo_str, write_matrix_market, write_matrix_market_str, MmioError};
pub use crate::parallel::{
    parallel_spmv, parallel_spmv_add,
    parallel_axpy, parallel_axpby,
    parallel_dot, parallel_norm2,
};

pub use crate::eigen::{
    EigenParams, EigenResult, EigenSolver, EigenWhich,
    PowerIter, SubspaceIter, InverseIter, RayleighQuotientIter,
    LanczosIter, ArnoldiIter,
    GeneralizedEigen, ShiftInvertLanczos,
    KrylovSchur, Lobpcg,
    LanczosSvd, SvdResult,
    QuadraticEigen,
    NonlinearOperator, NepNewton,
};

pub use crate::direct::{
    DirectSolver, DirectOptions, DirectSolverPrecond,
    SparseLu, SparseCholesky, MultifrontalLu, MultifrontalOptions,
    ordering::{OrderingMethod, rcm, colamd, nd},
};

pub use crate::builder::{
    SolverBuilder, SolveMethod, DirectBackend, PrecondChoice, Ordering as SolverOrdering,
    solve_auto,
};
