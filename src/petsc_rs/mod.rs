//! PETSc KSP-compatible interface built on top of `linger`'s solver builder.
//!
//! Enable with the `petsc-rs` Cargo feature.
//!
//! # Current implementation status
//!
//! | Capability | Status |
//! |-----------|--------|
//! | `KspSolver` create/set-operators/solve lifecycle | ✓ Implemented |
//! | All Krylov methods (CG/GMRES/BiCGSTAB/MINRES/FGMRES/LGMRES/IDR/TFQMR) | ✓ Available |
//! | Preconditioner passthrough (AMG/ILU/AMS/ADS) | ✓ Available via `SolveMethod` |
//! | SNES-style nonlinear path hooks | ○ Planned |
//! | SLEPc eigensolver handoff | ○ Planned |
//! | Distributed-memory KSP (requires `mpi` feature) | ○ Planned |
//!
//! This module exposes a lifecycle API that mirrors the PETSc C library
//! (`KspSolver::create`, `set_operators`, `solve`) so that code originally
//! written against PETSc can be ported to this pure-Rust implementation
//! with minimal changes.
//!
//! # Example
//! ```rust,no_run
//! # #[cfg(feature = "petsc-rs")] {
//! use linger::petsc_rs::KspSolver;
//! use linger::sparse::CsrMatrix;
//! use linger::core::vector::DenseVec;
//!
//! let a: CsrMatrix<f64> = CsrMatrix::identity(64);
//! let b = DenseVec::zeros(64);
//!
//! let (x, report) = KspSolver::create()
//!     .rtol(1e-10)
//!     .max_iter(500)
//!     .set_operators(a)
//!     .solve(b)
//!     .expect("KSP solve failed");
//!
//! println!("converged in {} iterations", report.iters);
//! # }
//! ```

use crate::builder::{BuilderSolveReport, SolveMethod, SolverBuilder};
use crate::core::error::SolverError;
use crate::core::vector::DenseVec;
use crate::sparse::CsrMatrix;

// ─── KspSolver ───────────────────────────────────────────────────────────────

/// PETSc-inspired Krylov solver with a create / set-operators / solve
/// lifecycle.
pub struct KspSolver {
    builder: SolverBuilder,
}

impl KspSolver {
    /// Create a KSP context with default settings (GMRES, ILU(0), rtol=1e-8).
    pub fn create() -> Self {
        KspSolver {
            builder: SolverBuilder::new(),
        }
    }

    /// Set relative residual tolerance (mirrors `KSPSetTolerances`).
    pub fn rtol(mut self, v: f64) -> Self {
        self.builder = self.builder.rtol(v);
        self
    }

    /// Set absolute residual tolerance (mirrors `KSPSetTolerances`).
    pub fn atol(mut self, v: f64) -> Self {
        self.builder = self.builder.atol(v);
        self
    }

    /// Set maximum number of iterations (mirrors `KSPSetTolerances`).
    pub fn max_iter(mut self, n: usize) -> Self {
        self.builder = self.builder.max_iter(n);
        self
    }

    /// Select the Krylov method (mirrors `KSPSetType`).
    pub fn method(mut self, m: SolveMethod) -> Self {
        self.builder = self.builder.method(m);
        self
    }

    /// Bind the system matrix (mirrors `KSPSetOperators`).
    ///
    /// Returns a [`BoundKspSolver`] ready to call [`BoundKspSolver::solve`].
    pub fn set_operators(self, a: CsrMatrix<f64>) -> BoundKspSolver {
        BoundKspSolver {
            builder: self.builder,
            a,
        }
    }
}

// ─── BoundKspSolver ──────────────────────────────────────────────────────────

/// A KSP context with an operator bound — ready to call `solve`.
pub struct BoundKspSolver {
    builder: SolverBuilder,
    a: CsrMatrix<f64>,
}

impl BoundKspSolver {
    /// Solve `A x = b` (mirrors `KSPSolve`).
    ///
    /// Returns the solution vector and solver diagnostics.
    pub fn solve(
        self,
        b: DenseVec<f64>,
    ) -> Result<(DenseVec<f64>, BuilderSolveReport), SolverError> {
        self.builder.solve_with_report(&self.a, &b)
    }
}
