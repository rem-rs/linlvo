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
pub mod parallel_dist;
pub mod eigen;
pub mod direct;
pub mod builder;
pub mod simd;

#[cfg(feature = "blas")]
pub mod blas_backend;

#[cfg(feature = "blas-oxiblas")]
pub mod oxiblas_backend;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "hypre-rs")]
pub mod hypre_rs;

#[cfg(feature = "petsc-rs")]
pub mod petsc_rs;

// ─── Re-exports ──────────────────────────────────────────────────────────────

pub use crate::core::{
    error::SolverError,
    operator::{LinearOperator, TransposeOperator},
    preconditioner::Preconditioner,
    scalar::{Scalar, ComplexScalar},
    solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
    dense::DenseMatrix,
};

pub use num_complex::Complex;

pub use crate::precond::{
    JacobiPrecond, SorPrecond, SsorPrecond, Ilu0Precond,
    IlukPrecond, IlutPrecond, Icc0Precond, IldltPrecond, SpaiPrecond,
    AdditivePrecond, MultiplicativePrecond, BlockJacobiPrecond,
    AmsPrecond, AmsConfig, AmsProfile, AuxSpaceSolver, AuxSolverProfile, AuxAmgProfile,
    AdsPrecond, AdsConfig, AdsProfile,
};
pub use crate::iterative::{ConjugateGradient, Gmres, BiCgStab, Minres, Fgmres, Lgmres, Idrs, Tfqmr, PipeCg};
pub use crate::iterative::complex_gmres::{ComplexGmres, ComplexGmresWorkspace, ComplexGmresResult};
pub use crate::amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, CycleType, SmootherType, LevelInfo};
pub use crate::sparse::{
    BsrMatrix, BsrBuilder,
    DiaMatrix, EllMatrix,
    CsrMatrix, CooMatrix,
    read_matrix_market, read_matrix_market_coo,
    read_matrix_market_str, read_matrix_market_coo_str,
    write_matrix_market, write_matrix_market_str,
    MmioError,
};
pub use crate::parallel::{
    parallel_spmv, parallel_spmv_add,
    parallel_axpy, parallel_axpby,
    parallel_dot, parallel_norm2,
};
pub use crate::parallel_dist::{
    PartitionLayout, block_partition,
    HaloExchange, HaloError, LocalHaloExchange,
    HaloPlan, NeighborHaloPlan,
    DistCsrMatrix,
    dist_cg, DistCgParams, DistCgResult,
    GlobalReduce, LocalReduce,
};

#[cfg(feature = "mpi")]
pub use crate::parallel_dist::{MpiHaloExchange, MpiReduce};

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
    SparseLu, SupernodalSparseLu, SparseCholesky, SupernodalSparseCholesky, SparseLdlt, MultifrontalLu, MultifrontalOptions, MumpsSolver, MklSolver,
    ordering::{OrderingMethod, rcm, colamd, nd},
};

pub use crate::builder::{
    SolverBuilder, SolveMethod, DirectBackend, PrecondChoice, Ordering as SolverOrdering,
    BuilderPrecondReport, BuilderSolveReport,
    ExternalBackend, BackendCapabilities, EffectiveBackend, BackendSelectionReport,
    solve_auto,
};
