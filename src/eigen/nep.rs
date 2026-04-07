//! Nonlinear Eigenvalue Problem (NEP) — Sprint 12.
//!
//! Provides a basic Newton/inverse-iteration solver for finding a single
//! eigenpair `(λ, x)` of a nonlinear eigenvalue problem `T(λ) x = 0`.
//!
//! **Algorithm:**
//! Alternates between:
//! 1. **λ update** (Rayleigh-functional step):
//!    `δλ = -(xᵀ T(λ) x) / (xᵀ T'(λ) x)`
//! 2. **x update** (inverse-iteration-like step):
//!    Solve `T(λ + ε) w = x`  with a small regularisation shift `ε`,
//!    then set `x ← w / ‖w‖`.
//!
//! For the standard problem `T(λ) = A − λI` the λ-update becomes the
//! Rayleigh quotient update, giving cubic convergence near an eigenvalue.
//! The x-update is inverse iteration, which converges geometrically.

use crate::core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}};
use super::{fill_random, dot, normalise, matfree_gmres};
use crate::core::operator::LinearOperator;

// ─── NonlinearOperator trait ──────────────────────────────────────────────────

/// Trait for operators `T(λ)` arising in nonlinear eigenvalue problems.
///
/// The problem is `T(λ) x = 0`.  The user must provide `apply_t`; the
/// derivative `apply_dt` defaults to a finite-difference approximation.
pub trait NonlinearOperator<T: Scalar>: Send + Sync {
    /// Dimension of the square operator `T(λ)`.
    fn nrows(&self) -> usize;

    /// Compute `out = T(λ) v`.
    fn apply_t(&self, lam: T, v: &DenseVec<T>, out: &mut DenseVec<T>);

    /// Compute `out = T'(λ) v` (derivative with respect to λ).
    ///
    /// Default: central finite difference with step `h = √ε max(|λ|, 1e-8)`.
    fn apply_dt(&self, lam: T, v: &DenseVec<T>, out: &mut DenseVec<T>) {
        let h = T::machine_epsilon().sqrt() * (lam.abs().max(T::from_f64(1e-8)));
        let lam_plus  = lam + h;
        let lam_minus = lam - h;
        let n = self.nrows();
        let mut tp = DenseVec::zeros(n);
        let mut tm = DenseVec::zeros(n);
        self.apply_t(lam_plus,  v, &mut tp);
        self.apply_t(lam_minus, v, &mut tm);
        let two_h = h + h;
        let os = out.as_mut_slice();
        let tps = tp.as_slice();
        let tms = tm.as_slice();
        for i in 0..n { os[i] = (tps[i] - tms[i]) / two_h; }
    }
}

// ─── NepNewton ────────────────────────────────────────────────────────────────

/// Newton / inverse-iteration for a single eigenpair of a NEP.
///
/// Combines:
/// - Rayleigh-functional update for λ (quadratic convergence)
/// - Inverse iteration for x (linear convergence, dominant direction selection)
pub struct NepNewton<T: Scalar> {
    pub max_iter: usize,
    pub tol: T,
    /// Initial eigenvalue estimate (shift).
    pub initial_shift: T,
    pub seed: u64,
}

impl<T: Scalar> NepNewton<T> {
    pub fn new(initial_shift: T, tol: T, max_iter: usize) -> Self {
        NepNewton { max_iter, tol, initial_shift, seed: 42 }
    }

    /// Compute one eigenpair `(λ, x)` near `self.initial_shift`.
    pub fn solve<Op: NonlinearOperator<T>>(
        &self,
        op: &Op,
    ) -> Result<(T, DenseVec<T>), SolverError> {
        let n = op.nrows();

        // Initial vector: random, normalised
        let mut x = DenseVec::zeros(n);
        fill_random(&mut x, self.seed);
        normalise(&mut x);

        let mut lam = self.initial_shift;

        for _iter in 0..self.max_iter {
            // ── x update: solve T(λ + ε) w = x ──────────────────────────────
            // Add regularisation ε to avoid exact singularity at the eigenvalue.
            // Use a shifted operator: T(λ + shift_eps) where shift_eps → 0 as
            // we converge.  We just compute T(λ) x, then pick a small reg shift.
            let mut r = DenseVec::zeros(n);
            op.apply_t(lam, &x, &mut r);
            let res_norm = r.norm2();
            if res_norm < self.tol { return Ok((lam, x)); }

            // Inverse iteration: solve T(λ + ε) w = x (w ≈ x/0 = dominant eigvec)
            // We use a regularised shift to make T non-singular.
            let eps = T::from_f64(1e-6) * (T::one() + lam.abs());
            let reg_lam = lam + eps;
            let reg_op = TAtLambda { op, lam: reg_lam, n };
            let mut w = x.clone();
            let _ = matfree_gmres(&reg_op, &x, &mut w, T::from_f64(1e-6), 100, 30);
            let wnrm = w.norm2();
            if wnrm > T::from_f64(1e-15) {
                w.scale(T::one() / wnrm);
                x = w;
            }

            // ── λ update: Rayleigh-functional step ───────────────────────────
            // r = T(λ) x (recompute with updated x)
            op.apply_t(lam, &x, &mut r);

            // T'(λ) x
            let mut dt_x = DenseVec::zeros(n);
            op.apply_dt(lam, &x, &mut dt_x);

            let xtr   = dot(x.as_slice(), r.as_slice());
            let xtdtx = dot(x.as_slice(), dt_x.as_slice());

            if xtdtx.abs() > T::from_f64(1e-15) {
                let delta_lam = -xtr / xtdtx;
                // Dampen the step to avoid overshooting
                let damped = if delta_lam.abs() > T::one() {
                    delta_lam.signum()
                } else {
                    delta_lam
                };
                lam = lam + damped;
            }
        }

        // Final residual check
        let mut r = DenseVec::zeros(n);
        op.apply_t(lam, &x, &mut r);
        let res = r.norm2();
        if res < self.tol { return Ok((lam, x)); }

        Err(SolverError::ConvergenceFailed {
            max_iter: self.max_iter,
            residual: num_traits::ToPrimitive::to_f64(&res).unwrap_or(f64::INFINITY),
        })
    }
}

// ─── Helper: LinearOperator wrapper for T(λ) at a fixed λ ────────────────────

struct TAtLambda<'a, T: Scalar, Op: NonlinearOperator<T>> {
    op: &'a Op,
    lam: T,
    n: usize,
}

impl<'a, T: Scalar, Op: NonlinearOperator<T>> LinearOperator for TAtLambda<'a, T, Op> {
    type Vector = DenseVec<T>;
    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        self.op.apply_t(self.lam, x, y);
    }
    fn nrows(&self) -> usize { self.n }
    fn ncols(&self) -> usize { self.n }
}

unsafe impl<'a, T: Scalar, Op: NonlinearOperator<T>> Send for TAtLambda<'a, T, Op> {}
unsafe impl<'a, T: Scalar, Op: NonlinearOperator<T>> Sync for TAtLambda<'a, T, Op> {}
