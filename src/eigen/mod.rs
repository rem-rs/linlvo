//! Eigenvalue solvers — Sprint 7 / 8 / 9 / 10 / 11 / 12.
//!
//! | Sprint | Algorithms |
//! |--------|-----------|
//! | 7 | [`PowerIter`], [`SubspaceIter`], [`InverseIter`], [`RayleighQuotientIter`] |
//! | 8 | [`LanczosIter`] (IRLM), [`ArnoldiIter`] (IRAM) |
//! | 9 | [`GeneralizedEigen`], [`ShiftInvertLanczos`] |
//! | 10 | [`KrylovSchur`], [`Lobpcg`] |
//! | 11 | [`LanczosSvd`], [`SvdResult`] |
//! | 12 | [`QuadraticEigen`], [`NonlinearOperator`], [`NepNewton`] |
//!
//! All algorithms operate through the [`LinearOperator`] trait so they work
//! with any CSR, nalgebra, or matrix-free operator.

pub mod power;
pub mod inverse;
pub mod subspace;
pub mod lanczos;
pub mod arnoldi;
pub mod generalized;
pub mod krylov_schur;
pub mod lobpcg;
pub mod svd;
pub mod qep;
pub mod nep;
pub mod ame;

pub use power::PowerIter;
pub use inverse::{InverseIter, RayleighQuotientIter};
pub use subspace::SubspaceIter;
pub use lanczos::LanczosIter;
pub use arnoldi::ArnoldiIter;
pub use generalized::{GeneralizedEigen, ShiftInvertLanczos};
pub use krylov_schur::KrylovSchur;
pub use lobpcg::Lobpcg;
pub use svd::{LanczosSvd, SvdResult};
pub use qep::QuadraticEigen;
pub use nep::{NonlinearOperator, NepNewton};
pub use ame::{AmeSolver, AmeConfig, AmeResult};

use crate::core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}};

// ─── matrix-free GMRES for inverse iteration ─────────────────────────────────

/// Minimal GMRES(restart) that works against any [`LinearOperator`].
///
/// Used internally by [`InverseIter`] and [`RayleighQuotientIter`] to solve
/// `(A − σI) y = x` without needing to materialise the shifted matrix as a
/// `CsrMatrix`.
pub(crate) fn matfree_gmres<T: Scalar, Op: crate::core::operator::LinearOperator<Vector = DenseVec<T>>>(
    op: &Op,
    b: &DenseVec<T>,
    x: &mut DenseVec<T>,
    rtol: T,
    max_iter: usize,
    restart: usize,
) -> Result<(), SolverError> {
    let n = b.len();
    let norm_b = b.norm2();
    let norm_b_f = if norm_b == T::zero() { T::one() } else { norm_b };
    let mut total = 0usize;

    'outer: loop {
        let mut r = DenseVec::zeros(n);
        {
            let mut ax = DenseVec::zeros(n);
            op.apply(x, &mut ax);
            let rs = r.as_mut_slice();
            for i in 0..n { rs[i] = b[i] - ax[i]; }
        }
        let beta = r.norm2();
        if beta / norm_b_f < rtol { break; }
        if total >= max_iter { break; }

        let m = restart.min(max_iter - total);
        let mut v: Vec<DenseVec<T>> = Vec::with_capacity(m + 1);
        let mut v0 = r.clone();
        v0.scale(T::one() / beta);
        v.push(v0);

        let mut h: Vec<Vec<T>> = Vec::new();
        let mut cs: Vec<T> = Vec::new();
        let mut sn: Vec<T> = Vec::new();
        let mut g = vec![T::zero(); m + 1];
        g[0] = beta;

        let mut j_final = 0;
        let mut inner_ok = false;

        for j in 0..m {
            let mut w = DenseVec::zeros(n);
            op.apply(&v[j], &mut w);

            let mut hcol: Vec<T> = Vec::with_capacity(j + 2);
            for vi in v.iter().take(j + 1) {
                let hij = dot(vi.as_slice(), w.as_slice());
                hcol.push(hij);
                let ws = w.as_mut_slice();
                let vis = vi.as_slice();
                for i in 0..n { ws[i] -= hij * vis[i]; }
            }
            let h_next = w.norm2();
            hcol.push(h_next);
            h.push(hcol);
            if h_next > T::machine_epsilon() { w.scale(T::one() / h_next); }
            v.push(w);

            let hj = h.last_mut().unwrap();
            for i in 0..j {
                let tmp = cs[i] * hj[i] + sn[i] * hj[i + 1];
                hj[i + 1] = -sn[i] * hj[i] + cs[i] * hj[i + 1];
                hj[i] = tmp;
            }
            let denom = (hj[j] * hj[j] + hj[j + 1] * hj[j + 1]).sqrt();
            let (c, s) = if denom > T::zero() {
                (hj[j] / denom, hj[j + 1] / denom)
            } else {
                (T::one(), T::zero())
            };
            cs.push(c); sn.push(s);
            hj[j]     = c * hj[j] + s * hj[j + 1];
            hj[j + 1] = T::zero();
            g[j + 1] = -s * g[j];
            g[j]     =  c * g[j];

            total += 1;
            j_final = j + 1;
            if g[j + 1].abs() / norm_b_f < rtol { inner_ok = true; break; }
        }

        // Back-substitution
        let jf = j_final;
        let mut y = vec![T::zero(); jf];
        for i in (0..jf).rev() {
            let mut s = g[i];
            for k in (i + 1)..jf { s -= h[k][i] * y[k]; }
            if h[i][i].abs() > T::zero() { y[i] = s / h[i][i]; }
        }
        for j in 0..jf { x.axpy(y[j], &v[j]); }

        if inner_ok || total >= max_iter { break 'outer; }
    }

    // Final residual check — always Ok; caller decides if result is good enough.
    Ok(())
}

// ─── EigenWhich ───────────────────────────────────────────────────────────────

/// Which eigenvalues to target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigenWhich {
    /// Eigenvalue(s) with largest absolute value (default).
    LargestMagnitude,
    /// Eigenvalue(s) with smallest absolute value (requires shift-invert or
    /// a non-singular operator).
    SmallestMagnitude,
    /// Algebraically largest eigenvalue(s).
    LargestAlgebraic,
    /// Algebraically smallest eigenvalue(s).
    SmallestAlgebraic,
    /// Both extremes (symmetric problems only; returns 2·`n_eigenvalues` values).
    BothEnds,
}

// ─── EigenParams ─────────────────────────────────────────────────────────────

/// Parameters controlling an eigenvalue solve.
#[derive(Debug, Clone)]
pub struct EigenParams<T: Scalar> {
    /// Number of eigenvalue/vector pairs to compute.
    pub n_eigenvalues: usize,
    /// Which part of the spectrum to target.
    pub which: EigenWhich,
    /// Convergence tolerance: ‖Ax − λx‖₂ / |λ| < tol.
    pub tol: T,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Print convergence info each iteration when `true`.
    pub verbose: bool,
}

impl<T: Scalar> EigenParams<T> {
    /// Construct with sensible defaults (`tol = 1e-10`, `max_iter = 1000`).
    pub fn new(n_eigenvalues: usize, which: EigenWhich) -> Self {
        EigenParams {
            n_eigenvalues,
            which,
            tol: T::from_f64(1e-10),
            max_iter: 1_000,
            verbose: false,
        }
    }
}

// ─── EigenResult ─────────────────────────────────────────────────────────────

/// Output of an eigenvalue solve.
#[derive(Debug, Clone)]
pub struct EigenResult<T: Scalar> {
    /// Computed eigenvalues, ordered by convergence / magnitude.
    pub eigenvalues: Vec<T>,
    /// Corresponding eigenvectors (unit-normalised).
    /// `eigenvectors[k]` pairs with `eigenvalues[k]`.
    pub eigenvectors: Vec<DenseVec<T>>,
    /// Number of eigenvalues that met the tolerance.
    pub converged: usize,
    /// Total matrix-vector products consumed.
    pub iterations: usize,
    /// Final residual norms ‖Ax − λx‖₂ for each pair.
    pub residuals: Vec<T>,
}

// ─── EigenSolver trait ───────────────────────────────────────────────────────

/// Common interface for all eigenvalue algorithms.
pub trait EigenSolver<T: Scalar> {
    /// Compute eigenvalue/vector pairs of the operator `op`.
    ///
    /// # Errors
    /// Returns [`SolverError::ConvergenceFailed`] if the algorithm does not
    /// meet `params.tol` within `params.max_iter` iterations.
    fn solve<Op>(
        &self,
        op: &Op,
        params: &EigenParams<T>,
    ) -> Result<EigenResult<T>, SolverError>
    where
        Op: crate::core::operator::LinearOperator<Vector = DenseVec<T>>;
}

// ─── Shared helpers (crate-private) ──────────────────────────────────────────

/// Rayleigh quotient  λ ≈ xᵀAx / xᵀx.
pub(crate) fn rayleigh_quotient<T: Scalar>(
    ax: &DenseVec<T>,
    x: &DenseVec<T>,
) -> T {
    let xax = dot(ax.as_slice(), x.as_slice());
    let xx  = dot(x.as_slice(),  x.as_slice());
    if xx == T::zero() { T::zero() } else { xax / xx }
}

/// Residual norm  ‖Ax − λx‖₂.
pub(crate) fn residual_norm<T: Scalar>(
    ax: &DenseVec<T>,
    x: &DenseVec<T>,
    lambda: T,
) -> T {
    let n = ax.len();
    let mut s = T::zero();
    for i in 0..n {
        let d = ax[i] - lambda * x[i];
        s += d * d;
    }
    s.sqrt()
}

/// Normalise `v` in-place; returns the old norm (‖v‖₂ before scaling).
pub(crate) fn normalise<T: Scalar>(v: &mut DenseVec<T>) -> T {
    let nrm = v.norm2();
    if nrm > T::zero() {
        v.scale(T::one() / nrm);
    }
    nrm
}

/// Plain dot product over slices.
pub(crate) fn dot<T: Scalar>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b.iter()).fold(T::zero(), |s, (&ai, &bi)| s + ai * bi)
}

/// Fill a vector with a pseudo-random sequence (simple LCG — deterministic,
/// no external dependency).
pub(crate) fn fill_random<T: Scalar>(v: &mut DenseVec<T>, seed: u64) {
    let mut state = seed.wrapping_add(1);
    for x in v.as_mut_slice().iter_mut() {
        state = state.wrapping_mul(6_364_136_223_846_793_005)
                     .wrapping_add(1_442_695_040_888_963_407);
        // Map to (-1, 1)
        let frac = (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
        *x = T::from_f64(frac);
    }
}

/// Gram-Schmidt orthogonalise `v` against the columns in `basis` (in place).
pub(crate) fn orthogonalise_against<T: Scalar>(
    v: &mut DenseVec<T>,
    basis: &[DenseVec<T>],
) {
    for b in basis {
        let proj = dot(v.as_slice(), b.as_slice());
        let vs = v.as_mut_slice();
        let bs = b.as_slice();
        for i in 0..vs.len() {
            vs[i] -= proj * bs[i];
        }
    }
}
