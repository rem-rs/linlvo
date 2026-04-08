//! High-level `SolverBuilder` — unified entry point for all linger solvers.
//!
//! `SolverBuilder` provides a fluent, self-documenting API that unifies Krylov
//! iterative and direct solvers under a single interface.  It handles the
//! `analyze → factorize → solve` lifecycle automatically and wires up
//! preconditioners transparently.
//!
//! # Quick examples
//!
//! ```text
//! use linger::builder::{SolverBuilder, SolveMethod, DirectBackend};
//!
//! // Simple direct solve (multifrontal LU, RCM ordering)
//! let x = SolverBuilder::new()
//!     .method(SolveMethod::Direct(DirectBackend::Multifrontal))
//!     .solve(&a, &b)?;
//!
//! // GMRES with ILU(0) preconditioner
//! let x = SolverBuilder::new()
//!     .method(SolveMethod::Gmres { restart: 30 })
//!     .precond(PrecondChoice::Ilu0)
//!     .rtol(1e-10)
//!     .max_iter(200)
//!     .solve(&a, &b)?;
//!
//! // GMRES with exact MultifrontalLu preconditioner
//! let x = SolverBuilder::new()
//!     .method(SolveMethod::Gmres { restart: 20 })
//!     .precond(PrecondChoice::DirectLu(DirectBackend::Multifrontal))
//!     .solve(&a, &b)?;
//!
//! // CG for SPD (default settings)
//! let x = SolverBuilder::new()
//!     .method(SolveMethod::Cg)
//!     .precond(PrecondChoice::Icc0)
//!     .solve(&a, &b)?;
//! ```

use crate::{
    core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}},
    sparse::CsrMatrix,
    iterative::{ConjugateGradient, Gmres, BiCgStab},
    KrylovSolver, SolverParams, VerboseLevel,
    direct::{DirectOptions, DirectSolver, DirectSolverPrecond, ordering::OrderingMethod},
    core::preconditioner::Preconditioner,
};

// ─── Public enums ─────────────────────────────────────────────────────────────

/// Which backend to use for direct solvers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectBackend {
    /// Gilbert-Peierls sparse LU (general square, dense working storage).
    Lu,
    /// Left-looking sparse Cholesky (symmetric positive definite only).
    Cholesky,
    /// Multifrontal LU with optional BLR compression (general square).
    Multifrontal,
}

/// Top-level solver selection.
#[derive(Debug, Clone)]
pub enum SolveMethod {
    /// Conjugate Gradient — symmetric positive definite systems.
    Cg,
    /// GMRES with specified restart parameter.
    Gmres {
        /// Krylov subspace restart dimension.  Typical values: 20–50.
        restart: usize,
    },
    /// BiCGSTAB — general non-symmetric, less memory than GMRES.
    BiCgStab,
    /// Exact direct solve (no Krylov outer iteration).
    Direct(DirectBackend),
}

/// Preconditioner choice for Krylov methods.
#[derive(Debug, Clone)]
pub enum PrecondChoice {
    /// No preconditioner.
    None,
    /// Jacobi (diagonal scaling).
    Jacobi,
    /// ILU(0) — zero-fill incomplete LU.
    Ilu0,
    /// ICC(0) — zero-fill incomplete Cholesky (SPD only).
    Icc0,
    /// Exact direct-solver preconditioner.
    ///
    /// Uses the given direct backend to factor `A` and applies the exact
    /// triangular solve at each preconditioner application.  One Krylov
    /// iteration is usually sufficient (exact precond = direct solve).
    DirectLu(DirectBackend),
}

/// Fill-reducing permutation exposed at builder level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ordering {
    /// No permutation.
    Natural,
    /// Reverse Cuthill-McKee (bandwidth reduction, good for SPD).
    Rcm,
    /// Approximate Minimum Degree via COLAMD (fill reduction, non-symmetric).
    Colamd,
    /// Multilevel Nested Dissection — best fill reduction for large unstructured meshes.
    NodeNd,
}

// ─── Builder ─────────────────────────────────────────────────────────────────

/// Fluent builder for configuring and running linear solvers.
///
/// All fields have sensible defaults:
/// - method: `Gmres { restart: 30 }`
/// - precond: `None`
/// - ordering: `Rcm`
/// - rtol: `1e-8`
/// - max_iter: `1000`
/// - verbose: silent
#[derive(Debug, Clone)]
pub struct SolverBuilder {
    method:   SolveMethod,
    precond:  PrecondChoice,
    ordering: Ordering,
    rtol:     f64,
    atol:     f64,
    max_iter: usize,
    verbose:  bool,
}

impl Default for SolverBuilder {
    fn default() -> Self {
        SolverBuilder {
            method:   SolveMethod::Gmres { restart: 30 },
            precond:  PrecondChoice::None,
            ordering: Ordering::Rcm,
            rtol:     1e-8,
            atol:     0.0,
            max_iter: 1000,
            verbose:  false,
        }
    }
}

impl SolverBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self { Self::default() }

    /// Set the solution method.
    pub fn method(mut self, m: SolveMethod) -> Self { self.method = m; self }

    /// Set the preconditioner (for Krylov methods).
    pub fn precond(mut self, p: PrecondChoice) -> Self { self.precond = p; self }

    /// Set the fill-reducing ordering for direct solvers / direct preconditioners.
    pub fn ordering(mut self, o: Ordering) -> Self { self.ordering = o; self }

    /// Set the relative residual tolerance for Krylov convergence.
    pub fn rtol(mut self, v: f64) -> Self { self.rtol = v; self }

    /// Set the absolute residual tolerance for Krylov convergence.
    pub fn atol(mut self, v: f64) -> Self { self.atol = v; self }

    /// Set the maximum number of Krylov iterations.
    pub fn max_iter(mut self, v: usize) -> Self { self.max_iter = v; self }

    /// Enable iteration-by-iteration convergence logging.
    pub fn verbose(mut self) -> Self { self.verbose = true; self }

    // ─── solve ────────────────────────────────────────────────────────────────

    /// Solve `A x = b`.
    ///
    /// Returns `x` on success, or a [`SolverError`] describing the failure.
    pub fn solve<T: Scalar>(&self, a: &CsrMatrix<T>, b: &DenseVec<T>)
        -> Result<DenseVec<T>, SolverError>
    {
        let n = a.nrows();
        if b.len() != n {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: a.ncols(), rhs_len: b.len(),
            });
        }
        let mut x = DenseVec::zeros(n);
        self.solve_into(a, b, &mut x)?;
        Ok(x)
    }

    /// Solve `A x = b`, storing the result in the pre-allocated `x`.
    pub fn solve_into<T: Scalar>(
        &self,
        a:  &CsrMatrix<T>,
        b:  &DenseVec<T>,
        x:  &mut DenseVec<T>,
    ) -> Result<(), SolverError> {
        match &self.method {
            SolveMethod::Direct(backend) => self.run_direct(backend, a, b, x),
            _ => self.run_krylov(a, b, x),
        }
    }

    // ─── internal: direct ─────────────────────────────────────────────────────

    fn direct_opts(&self) -> DirectOptions {
        DirectOptions {
            ordering: match self.ordering {
                Ordering::Natural => OrderingMethod::Natural,
                Ordering::Rcm     => OrderingMethod::Rcm,
                Ordering::Colamd  => OrderingMethod::Colamd,
                Ordering::NodeNd  => OrderingMethod::NodeNd,
            },
            ..Default::default()
        }
    }

    fn krylov_params(&self) -> SolverParams {
        SolverParams {
            rtol:     self.rtol,
            atol:     self.atol,
            max_iter: self.max_iter,
            verbose:  if self.verbose { VerboseLevel::Iterations } else { VerboseLevel::Silent },
            ..Default::default()
        }
    }

    fn run_direct<T: Scalar>(
        &self,
        backend: &DirectBackend,
        a: &CsrMatrix<T>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
    ) -> Result<(), SolverError> {
        use crate::direct::{SparseLu, SparseCholesky, MultifrontalLu, MultifrontalOptions};
        match backend {
            DirectBackend::Lu => {
                let mut s = SparseLu::<T>::new(self.direct_opts());
                s.factor(a)?;
                s.solve(b, x)
            }
            DirectBackend::Cholesky => {
                let mut s = SparseCholesky::<T>::new(self.direct_opts());
                s.factor(a)?;
                s.solve(b, x)
            }
            DirectBackend::Multifrontal => {
                let mut s = MultifrontalLu::<T>::with_options(MultifrontalOptions {
                    base: self.direct_opts(),
                    ..Default::default()
                });
                s.factor(a)?;
                s.solve(b, x)
            }
        }
    }

    // ─── internal: Krylov ─────────────────────────────────────────────────────

    fn run_krylov<T: Scalar>(
        &self,
        a: &CsrMatrix<T>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
    ) -> Result<(), SolverError> {
        let params = self.krylov_params();
        match &self.precond {
            PrecondChoice::None => {
                self.dispatch_krylov::<T>(a, None, b, x, &params)
            }
            PrecondChoice::Jacobi => {
                let p = crate::JacobiPrecond::from_csr(a)?;
                self.dispatch_krylov(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, &params)
            }
            PrecondChoice::Ilu0 => {
                let p = crate::Ilu0Precond::from_csr(a)?;
                self.dispatch_krylov(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, &params)
            }
            PrecondChoice::Icc0 => {
                let p = crate::Icc0Precond::from_csr(a)?;
                self.dispatch_krylov(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, &params)
            }
            PrecondChoice::DirectLu(backend) => {
                self.run_krylov_direct_precond(backend, a, b, x, &params)
            }
        }
    }

    fn dispatch_krylov<T: Scalar>(
        &self,
        a:       &CsrMatrix<T>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b:       &DenseVec<T>,
        x:       &mut DenseVec<T>,
        params:  &SolverParams,
    ) -> Result<(), SolverError> {
        match &self.method {
            SolveMethod::Cg => {
                ConjugateGradient::<T>::default().solve(a, precond, b, x, params)?;
            }
            SolveMethod::Gmres { restart } => {
                Gmres::<T>::new(*restart).solve(a, precond, b, x, params)?;
            }
            SolveMethod::BiCgStab => {
                BiCgStab::<T>::default().solve(a, precond, b, x, params)?;
            }
            SolveMethod::Direct(_) => unreachable!(),
        }
        Ok(())
    }

    fn run_krylov_direct_precond<T: Scalar>(
        &self,
        backend: &DirectBackend,
        a:       &CsrMatrix<T>,
        b:       &DenseVec<T>,
        x:       &mut DenseVec<T>,
        params:  &SolverParams,
    ) -> Result<(), SolverError> {
        use crate::direct::{SparseLu, SparseCholesky, MultifrontalLu, MultifrontalOptions};
        match backend {
            DirectBackend::Lu => {
                let s = SparseLu::<T>::new(self.direct_opts());
                let p = DirectSolverPrecond::new(s, a)?;
                self.dispatch_krylov(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, params)
            }
            DirectBackend::Cholesky => {
                let s = SparseCholesky::<T>::new(self.direct_opts());
                let p = DirectSolverPrecond::new(s, a)?;
                self.dispatch_krylov(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, params)
            }
            DirectBackend::Multifrontal => {
                let s = MultifrontalLu::<T>::with_options(MultifrontalOptions {
                    base: self.direct_opts(),
                    ..Default::default()
                });
                let p = DirectSolverPrecond::new(s, a)?;
                self.dispatch_krylov(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, params)
            }
        }
    }
}

// ─── Convenience functions ────────────────────────────────────────────────────

/// Solve `A x = b` with automatic method selection.
///
/// Uses Cholesky if `spd_hint` is true, otherwise MultifrontalLu.  Intended as
/// the simplest possible entry point for users who don't want to configure a
/// full `SolverBuilder`.
///
/// For more control use [`SolverBuilder`] directly.
pub fn solve_auto<T: Scalar>(
    a:        &CsrMatrix<T>,
    b:        &DenseVec<T>,
    spd_hint: bool,
) -> Result<DenseVec<T>, SolverError> {
    let backend = if spd_hint { DirectBackend::Cholesky } else { DirectBackend::Multifrontal };
    SolverBuilder::new()
        .method(SolveMethod::Direct(backend))
        .solve(a, b)
}
