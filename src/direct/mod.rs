//! Sparse direct solver module (Sprint 13).
//!
//! Provides exact sparse direct solvers as a complement to linger's iterative
//! methods.  All implementations are pure-Rust with zero external dependencies,
//! making the module fully compatible with `wasm32-unknown-unknown`.
//!
//! # Solvers
//!
//! | Struct | Algorithm | Suitable for |
//! |--------|-----------|--------------|
//! | [`SparseLu`] | Gilbert-Peierls supernodal-free LU + partial pivoting | General square matrices |
//! | [`SparseCholesky`] | Left-looking incomplete Cholesky | SPD matrices |
//! | [`SparseLdlt`] | Left-looking sparse LDLᵀ | Symmetric (indefinite) matrices |
//!
//! # Reordering
//!
//! Before factorising, apply a fill-reducing permutation to dramatically
//! reduce memory and time:
//!
//! ```text
//! use linger::direct::ordering::{rcm, OrderingMethod};
//! let perm = rcm(&a);   // Reverse Cuthill-McKee
//! ```
//!
//! # Direct solver as preconditioner
//!
//! Any [`DirectSolver`] can be wrapped in [`DirectSolverPrecond`] to plug it
//! into any [`KrylovSolver`](crate::KrylovSolver) as a preconditioner:
//!
//! ```text
//! use linger::direct::{SparseLu, DirectSolverPrecond};
//! let lu = SparseLu::<f64>::default();
//! let precond = DirectSolverPrecond::new(lu, &a)?;
//! let result = ConjugateGradient::default()
//!     .solve(&a, Some(&precond), &b, &mut x, &params)?;
//! ```

pub mod ordering;
pub mod etree;
pub mod symbolic;
pub mod blr;
mod triangular;
mod lu;
mod lu_sn;
mod cholesky;
mod cholesky_sn;
mod ldlt;
mod multifrontal;

pub use lu::SparseLu;
pub use lu_sn::{SupernodalSparseLu, SNode as SupernodalSNode};
pub use cholesky::SparseCholesky;
pub use cholesky_sn::{SupernodalSparseCholesky, SNode as CholeskySNode};
pub use ldlt::SparseLdlt;
pub use multifrontal::{MultifrontalLu, MultifrontalOptions};
pub use blr::{BlrBlock, BlrMatrix, compress_block, compress_block_adaptive};
pub use triangular::{forward_solve, backward_solve};
pub use symbolic::{SymbolicCholesky, SymbolicLu, symbolic_cholesky, symbolic_lu};

use crate::core::{error::SolverError, preconditioner::Preconditioner, scalar::Scalar, vector::DenseVec};
use crate::sparse::CsrMatrix;

// ─── Core trait ──────────────────────────────────────────────────────────────

/// Three-phase interface for sparse direct solvers.
///
/// The three phases mirror the standard direct solver workflow:
///
/// 1. **`analyze`** — structural analysis: fill-reducing reordering + symbolic
///    factorisation.  Only depends on the *sparsity pattern* of `a`; can be
///    reused across multiple matrices that share the same structure.
///
/// 2. **`factorize`** — numerical factorisation: compute L, U (or L, Lᵀ)
///    factors from the actual values of `a`.
///
/// 3. **`solve`** — triangular solve: forward + backward substitution with the
///    stored factors to compute `x = A⁻¹ b`.
///
/// A typical use-case with a single matrix:
/// ```text
/// solver.analyze(&a)?;
/// solver.factorize(&a)?;
/// solver.solve(&b, &mut x)?;
/// // same factors — solve for another RHS cheaply
/// solver.solve(&b2, &mut x2)?;
/// ```
pub trait DirectSolver<T: Scalar>: Send + Sync {
    /// Structural analysis: reordering + symbolic factorisation.
    ///
    /// Must be called once before `factorize`.  Can be skipped on subsequent
    /// calls if the sparsity pattern of `a` has not changed
    /// (`reuse_symbolic = true` on the concrete solver).
    fn analyze(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError>;

    /// Numerical factorisation: compute L and U (or L and Lᵀ) from `a`.
    ///
    /// `analyze` must have been called first.
    fn factorize(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError>;

    /// Triangular solve: given stored factors, compute `x ← A⁻¹ b`.
    ///
    /// `factorize` must have been called first.
    fn solve(&self, b: &DenseVec<T>, x: &mut DenseVec<T>) -> Result<(), SolverError>;

    /// Convenience: analyze + factorize in one call.
    fn factor(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        self.analyze(a)?;
        self.factorize(a)
    }

    /// Solve for multiple right-hand sides reusing the stored factors.
    ///
    /// Default implementation calls `solve` in a loop; concrete solvers may
    /// override this for efficiency (e.g., blocked triangular solves).
    fn solve_multi(&self, bs: &[DenseVec<T>], xs: &mut [DenseVec<T>]) -> Result<(), SolverError> {
        if bs.len() != xs.len() {
            return Err(SolverError::DimensionMismatch {
                op_rows: bs.len(),
                op_cols: 1,
                rhs_len: xs.len(),
            });
        }
        for (b, x) in bs.iter().zip(xs.iter_mut()) {
            self.solve(b, x)?;
        }
        Ok(())
    }

    /// Drop the stored numerical factors (but keep symbolic analysis).
    ///
    /// After calling this, `factorize` can be called again with updated values.
    fn reset_factors(&mut self);
}

// ─── DirectSolverPrecond ─────────────────────────────────────────────────────

/// Wraps any [`DirectSolver`] as a [`Preconditioner`].
///
/// The preconditioner application is a single exact triangular solve: `M⁻¹ x`,
/// where M is the factored matrix.  This turns a direct solver into a
/// one-iteration "perfect" preconditioner — useful when the direct solve is
/// too expensive on its own but can be amortised over many Krylov iterations
/// when the matrix is ill-conditioned.
///
/// # Example
/// ```text
/// let precond = DirectSolverPrecond::new(SparseLu::<f64>::default(), &a)?;
/// Gmres::new(30).solve(&a, Some(&precond), &b, &mut x, &params)?;
/// ```
pub struct DirectSolverPrecond<T: Scalar, S: DirectSolver<T>> {
    solver: S,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar, S: DirectSolver<T>> DirectSolverPrecond<T, S> {
    /// Build the preconditioner: runs analyze + factorize on `a`.
    pub fn new(mut solver: S, a: &CsrMatrix<T>) -> Result<Self, SolverError> {
        solver.factor(a)?;
        Ok(Self { solver, _marker: std::marker::PhantomData })
    }

    /// Access the underlying solver (e.g. to reuse for solve_multi).
    pub fn solver(&self) -> &S { &self.solver }
}

impl<T: Scalar, S: DirectSolver<T>> crate::core::operator::LinearOperator
    for DirectSolverPrecond<T, S>
{
    type Vector = DenseVec<T>;

    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        // Silently ignore errors in apply — preconditioner trait is infallible.
        // In practice, a well-factored matrix will never error here.
        let _ = self.solver.solve(x, y);
    }

    fn nrows(&self) -> usize { 0 } // not used by Preconditioner path
    fn ncols(&self) -> usize { 0 }
}

impl<T: Scalar, S: DirectSolver<T>> Preconditioner for DirectSolverPrecond<T, S> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let _ = self.solver.solve(x, y);
    }
}

// ─── Options ─────────────────────────────────────────────────────────────────

/// Configuration shared across direct solvers.
#[derive(Debug, Clone)]
pub struct DirectOptions {
    /// Fill-reducing reordering applied before factorisation.
    pub ordering: ordering::OrderingMethod,
    /// Diagonal pivot threshold in (0, 1].
    ///
    /// 1.0 = full partial pivoting (maximally stable).
    /// Values closer to 0 prefer diagonal pivots (preserves sparsity but less
    /// stable).  Default: 1.0.
    pub pivot_threshold: f64,
    /// When `true`, skip `analyze` on subsequent `factor` calls if the
    /// sparsity pattern is unchanged.  Default: `false`.
    pub reuse_symbolic: bool,
    /// Number of iterative refinement steps after the triangular solve.
    /// 0 = no refinement.  Default: 0.
    pub refine_steps: usize,
}

impl Default for DirectOptions {
    fn default() -> Self {
        Self {
            ordering: ordering::OrderingMethod::Rcm,
            pivot_threshold: 1.0,
            reuse_symbolic: false,
            refine_steps: 0,
        }
    }
}
