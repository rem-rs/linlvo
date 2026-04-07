//! Inverse Iteration and Rayleigh Quotient Iteration.
//!
//! **Inverse Iteration** — finds the eigenpair closest to a shift `σ`:
//! ```text
//! solve (A − σI) y = xₖ  →  x_{k+1} = y / ‖y‖
//! λ ≈ xᵀ A x  (Rayleigh quotient)
//! ```
//! Convergence rate: |(λ_target − σ) / (λ_next − σ)|.
//! With σ = 0 this finds the smallest-magnitude eigenpair.
//!
//! **Rayleigh Quotient Iteration** — shifts adaptively: σₖ = xₖᵀ A xₖ.
//! Converges cubically near the eigenvector (for symmetric operators).
//!
//! The inner linear system `(A − σI)y = x` is solved with a matrix-free
//! GMRES helper, so no direct factorisation is required.

use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    scalar::Scalar,
    vector::{DenseVec, Vector},
};
use super::{
    EigenParams, EigenResult, EigenSolver,
    fill_random, normalise, rayleigh_quotient, residual_norm,
    matfree_gmres,
};

// ─── ShiftedOp helper ────────────────────────────────────────────────────────

/// Wraps an operator `A` and a diagonal shift `σ`, presenting `(A − σI)`.
struct ShiftedOp<'a, T: Scalar, Op: LinearOperator<Vector = DenseVec<T>>> {
    op:    &'a Op,
    shift: T,
}

impl<'a, T: Scalar, Op: LinearOperator<Vector = DenseVec<T>>> LinearOperator
    for ShiftedOp<'a, T, Op>
{
    type Vector = DenseVec<T>;

    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        self.op.apply(x, y);
        let ys = y.as_mut_slice();
        let xs = x.as_slice();
        for i in 0..ys.len() { ys[i] -= self.shift * xs[i]; }
    }

    fn nrows(&self) -> usize { self.op.nrows() }
    fn ncols(&self) -> usize { self.op.ncols() }
}

unsafe impl<'a, T: Scalar, Op: LinearOperator<Vector = DenseVec<T>> + Send> Send
    for ShiftedOp<'a, T, Op> {}
unsafe impl<'a, T: Scalar, Op: LinearOperator<Vector = DenseVec<T>> + Sync> Sync
    for ShiftedOp<'a, T, Op> {}

// ─── InverseIter ─────────────────────────────────────────────────────────────

/// Inverse iteration with a fixed shift `σ`.
///
/// Finds the eigenpair of `A` whose eigenvalue is closest to `σ`.
/// With `shift = 0.0` this targets the smallest-magnitude eigenpair.
pub struct InverseIter<T: Scalar> {
    /// Spectral shift σ.
    pub shift: T,
    /// Relative tolerance for the inner linear solve.
    pub inner_rtol: f64,
    /// Maximum iterations for the inner linear solve.
    pub inner_max_iter: usize,
    /// Random seed for the start vector.
    pub seed: u64,
}

impl<T: Scalar> InverseIter<T> {
    pub fn new(shift: T) -> Self {
        InverseIter { shift, inner_rtol: 1e-10, inner_max_iter: 500, seed: 42 }
    }
}

impl<T: Scalar> Default for InverseIter<T> {
    fn default() -> Self { Self::new(T::zero()) }
}

impl<T: Scalar> EigenSolver<T> for InverseIter<T> {
    fn solve<Op>(
        &self,
        op: &Op,
        params: &EigenParams<T>,
    ) -> Result<EigenResult<T>, SolverError>
    where
        Op: LinearOperator<Vector = DenseVec<T>>,
    {
        let n = op.nrows();
        assert_eq!(n, op.ncols(), "InverseIter: operator must be square");
        assert_eq!(
            params.n_eigenvalues, 1,
            "InverseIter computes a single eigenpair; use SubspaceIter for multiple"
        );

        let mut x = DenseVec::zeros(n);
        fill_random(&mut x, self.seed);
        normalise(&mut x);

        let shifted = ShiftedOp { op, shift: self.shift };
        let inner_rtol = T::from_f64(self.inner_rtol);

        let mut ax = DenseVec::zeros(n);
        let mut lambda = T::zero();
        let mut res_norm = T::zero();

        for iter in 0..params.max_iter {
            let mut y = DenseVec::zeros(n);
            matfree_gmres(&shifted, &x, &mut y, inner_rtol, self.inner_max_iter, 30)
                .ok(); // best-effort; outer iteration detects convergence via residual

            x.copy_from(&y);
            normalise(&mut x);

            op.apply(&x, &mut ax);
            lambda = rayleigh_quotient(&ax, &x);
            res_norm = residual_norm(&ax, &x, lambda);

            let lam_abs = if lambda.abs() > T::from_f64(1e-14) { lambda.abs() } else { T::one() };
            let rel = res_norm / lam_abs;

            if params.verbose {
                let r = num_traits::ToPrimitive::to_f64(&rel).unwrap_or(f64::NAN);
                let l = num_traits::ToPrimitive::to_f64(&lambda).unwrap_or(f64::NAN);
                println!("  InverseIter iter {:4}  λ = {l:+.8e}  ‖r‖/|λ| = {r:.3e}", iter + 1);
            }

            if rel < params.tol {
                return Ok(EigenResult {
                    eigenvalues:  vec![lambda],
                    eigenvectors: vec![x],
                    converged:    1,
                    iterations:   iter + 1,
                    residuals:    vec![res_norm],
                });
            }
        }

        Err(SolverError::ConvergenceFailed {
            max_iter: params.max_iter,
            residual: num_traits::ToPrimitive::to_f64(&res_norm).unwrap_or(f64::INFINITY),
        })
    }
}

// ─── RayleighQuotientIter ────────────────────────────────────────────────────

/// Rayleigh Quotient Iteration — adaptive-shift inverse iteration.
///
/// Updates the shift each step as `σₖ = xₖᵀ A xₖ`, giving **cubic**
/// convergence near an eigenvector for symmetric operators.
pub struct RayleighQuotientIter<T: Scalar> {
    /// Initial shift (0 ⇒ smallest magnitude).
    pub initial_shift: T,
    pub inner_rtol: f64,
    pub inner_max_iter: usize,
    pub seed: u64,
}

impl<T: Scalar> RayleighQuotientIter<T> {
    pub fn new(initial_shift: T) -> Self {
        RayleighQuotientIter {
            initial_shift,
            inner_rtol: 1e-10,
            inner_max_iter: 500,
            seed: 42,
        }
    }
}

impl<T: Scalar> Default for RayleighQuotientIter<T> {
    fn default() -> Self { Self::new(T::zero()) }
}

impl<T: Scalar> EigenSolver<T> for RayleighQuotientIter<T> {
    fn solve<Op>(
        &self,
        op: &Op,
        params: &EigenParams<T>,
    ) -> Result<EigenResult<T>, SolverError>
    where
        Op: LinearOperator<Vector = DenseVec<T>>,
    {
        let n = op.nrows();
        assert_eq!(n, op.ncols(), "RayleighQuotientIter: operator must be square");
        assert_eq!(params.n_eigenvalues, 1, "RayleighQuotientIter computes a single eigenpair");

        let mut x = DenseVec::zeros(n);
        fill_random(&mut x, self.seed);
        normalise(&mut x);

        let mut ax = DenseVec::zeros(n);
        op.apply(&x, &mut ax);
        // Use the supplied initial_shift for the first step, then switch to
        // the adaptive Rayleigh quotient once the vector has been refined.
        let mut sigma = self.initial_shift;
        let inner_rtol = T::from_f64(self.inner_rtol);

        let mut lambda = sigma;
        let mut res_norm = T::zero();

        for iter in 0..params.max_iter {
            let shifted = ShiftedOp { op, shift: sigma };
            let mut y = DenseVec::zeros(n);
            matfree_gmres(&shifted, &x, &mut y, inner_rtol, self.inner_max_iter, 30)
                .ok(); // near-singular near convergence is expected; proceed with best y

            x.copy_from(&y);
            normalise(&mut x);

            op.apply(&x, &mut ax);
            lambda = rayleigh_quotient(&ax, &x);
            res_norm = residual_norm(&ax, &x, lambda);
            sigma = lambda;

            let lam_abs = if lambda.abs() > T::from_f64(1e-14) { lambda.abs() } else { T::one() };
            let rel = res_norm / lam_abs;

            if params.verbose {
                let r = num_traits::ToPrimitive::to_f64(&rel).unwrap_or(f64::NAN);
                let l = num_traits::ToPrimitive::to_f64(&lambda).unwrap_or(f64::NAN);
                println!("  RQI iter {:4}  λ = {l:+.8e}  ‖r‖/|λ| = {r:.3e}", iter + 1);
            }

            if rel < params.tol {
                return Ok(EigenResult {
                    eigenvalues:  vec![lambda],
                    eigenvectors: vec![x],
                    converged:    1,
                    iterations:   iter + 1,
                    residuals:    vec![res_norm],
                });
            }
        }

        Err(SolverError::ConvergenceFailed {
            max_iter: params.max_iter,
            residual: num_traits::ToPrimitive::to_f64(&res_norm).unwrap_or(f64::INFINITY),
        })
    }
}
