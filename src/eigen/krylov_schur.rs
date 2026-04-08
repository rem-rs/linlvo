//! Krylov-Schur restart — Sprint 10.
//!
//! Stewart (2001) showed that the Krylov-Schur decomposition
//! `A Vₘ = Vₘ Rₘ + fₘ eₘᵀ` (Rₘ upper quasi-triangular / Schur form)
//! allows the wanted Schur vectors to be locked in a numerically clean way
//! by simple column selection — no bulge-chasing required.
//!
//! This implementation uses the Hessenberg → Schur reduction from Sprint 8
//! (ArnoldiIter's Francis-QR step) and performs deflation by sorting the
//! Schur form so that wanted eigenvalues appear first.
//!
//! Works for **any** square operator; for symmetric operators the Schur form
//! is diagonal and this reduces to thick-restart Lanczos.

#![allow(clippy::needless_range_loop)]
use crate::core::{error::SolverError, operator::LinearOperator, scalar::Scalar, vector::{DenseVec, Vector}};
use super::{EigenParams, EigenResult, EigenSolver, fill_random, residual_norm, normalise, orthogonalise_against};
use super::arnoldi::{arnoldi_extend, hessenberg_eig};
use super::lanczos::sort_ritz;

// ─── KrylovSchur ─────────────────────────────────────────────────────────────

/// Krylov-Schur method (Stewart 2001) — robust restart for any square operator.
///
/// Compared to IRAM:
/// - No bulge-chasing: restart by simple Schur-vector selection.
/// - Naturally handles deflation (locking converged pairs).
/// - Works well for both symmetric and non-symmetric problems.
///
/// # Parameters
/// - `ncv`: Krylov space size (default `min(2k+1, n)`)
/// - `seed`: random seed for the initial vector
pub struct KrylovSchur {
    pub ncv:  Option<usize>,
    pub seed: u64,
}

impl Default for KrylovSchur {
    fn default() -> Self { KrylovSchur { ncv: None, seed: 42 } }
}

impl KrylovSchur {
    pub fn new(ncv: usize) -> Self { KrylovSchur { ncv: Some(ncv), seed: 42 } }
}

impl<T: Scalar> EigenSolver<T> for KrylovSchur {
    fn solve<Op>(&self, op: &Op, params: &EigenParams<T>) -> Result<EigenResult<T>, SolverError>
    where Op: LinearOperator<Vector = DenseVec<T>>
    {
        let n   = op.nrows();
        let nev = params.n_eigenvalues;
        assert_eq!(n, op.ncols(), "KrylovSchur: operator must be square");
        assert!(nev >= 1 && nev < n, "nev must be in 1..n");

        let ncv = self.ncv.unwrap_or_else(|| n.min(nev + nev.max(20)));
        assert!(ncv > nev && ncv <= n, "ncv must satisfy nev < ncv <= n");

        // ── Initial Arnoldi factorisation of length ncv ────────────────────
        let mut v0 = DenseVec::zeros(n);
        fill_random(&mut v0, self.seed);
        normalise(&mut v0);

        let (mut v_cols, mut h_mat, mut f) = arnoldi_extend(op, vec![v0], n, ncv);
        let mut n_locked = 0usize; // number of converged (locked) pairs

        for outer in 0..params.max_iter {
            // ── Compute Schur decomposition of H ───────────────────────────
            let (schur_vals, schur_vecs) = hessenberg_eig(&h_mat, ncv);

            // Sort: wanted first, unwanted last
            let mut order: Vec<usize> = (0..ncv).collect();
            sort_ritz(&mut order, &schur_vals, params.which);

            // ── Convergence check ──────────────────────────────────────────
            let beta_m = f.norm2();
            let mut conv_flags = vec![false; nev];
            let mut residuals  = vec![T::zero(); nev];

            for (ki, &wi) in order[..nev].iter().enumerate() {
                if ki < n_locked { conv_flags[ki] = true; continue; }
                let theta   = schur_vals[wi];
                let s_last  = schur_vecs[wi][ncv - 1].abs();
                let lam_abs = if theta.abs() > T::from_f64(1e-14) { theta.abs() } else { T::one() };
                residuals[ki] = beta_m * s_last / lam_abs;
                if residuals[ki] < params.tol {
                    conv_flags[ki] = true;
                    if ki == n_locked { n_locked += 1; }
                }
            }

            let n_conv = conv_flags.iter().filter(|&&c| c).count();

            if params.verbose {
                let max_r = residuals.iter().cloned()
                    .map(|r| num_traits::ToPrimitive::to_f64(&r).unwrap_or(f64::NAN))
                    .fold(0f64, f64::max);
                println!("  KrylovSchur outer {:3}  n_conv={}/{}  max‖r‖={:.3e}", outer, n_conv, nev, max_r);
            }

            if n_conv >= nev {
                return extract_result(op, &v_cols, &schur_vals, &schur_vecs, &order[..nev], ncv, n, outer + 1);
            }

            // ── Krylov-Schur truncation ────────────────────────────────────
            // Reorder Schur form: wanted pairs go to leading positions.
            // Build Q = reordered Schur vectors as columns.
            let mut q = vec![T::zero(); ncv * ncv];
            for (j, &wi) in order.iter().enumerate() {
                for i in 0..ncv { q[i * ncv + j] = schur_vecs[wi][i]; }
            }

            // Rotate V: V_new[:,j] = V Q[:,j]  for j=0..nev+1
            let keep = nev + 1; // keep nev+1 vectors (one new search direction)
            let mut v_new: Vec<DenseVec<T>> = Vec::with_capacity(keep);
            for j in 0..keep {
                let mut col = DenseVec::zeros(n);
                for i in 0..ncv { col.axpy(q[i * ncv + j], &v_cols[i]); }
                v_new.push(col);
            }

            // New residual: f_new = f * q[ncv-1, nev] (the (nev+1)-th Schur vector)
            let fac = q[(ncv - 1) * ncv + nev]; // Q[ncv-1, nev]
            {
                let fs = f.as_mut_slice();
                for fi in fs.iter_mut() { *fi *= fac; }
            }
            let beta_f = f.norm2();
            let mut v_next = f.clone();
            if beta_f > T::from_f64(1e-14) {
                v_next.scale(T::one() / beta_f);
            } else {
                fill_random(&mut v_next, self.seed + outer as u64);
                orthogonalise_against(&mut v_next, &v_new[..keep - 1]);
                normalise(&mut v_next);
            }
            v_new[keep - 1] = v_next;

            // Extend from keep columns to ncv
            let (vc2, mut hm2, f2) = arnoldi_extend(op, v_new, n, ncv);
            // Patch the leading keep×keep block of H with the reordered Schur form.
            // H_reordered = Q^T H Q restricted to the leading keep×keep block.
            for row in 0..keep {
                for col in 0..keep {
                    let mut val = T::zero();
                    for i in 0..ncv {
                        for j in 0..ncv {
                            val += q[i * ncv + row] * h_mat[i * ncv + j] * q[j * ncv + col];
                        }
                    }
                    hm2[row * ncv + col] = val;
                }
            }
            v_cols = vc2; h_mat = hm2; f = f2;
        }

        Err(SolverError::ConvergenceFailed { max_iter: params.max_iter, residual: f64::INFINITY })
    }
}

/// Extract final `EigenResult` from Schur data.
#[allow(clippy::too_many_arguments)]
fn extract_result<T: Scalar, Op: LinearOperator<Vector = DenseVec<T>>>(
    op:          &Op,
    v_cols:      &[DenseVec<T>],
    schur_vals:  &[T],
    schur_vecs:  &[Vec<T>],
    wanted_idx:  &[usize],
    ncv:         usize,
    n:           usize,
    iterations:  usize,
) -> Result<EigenResult<T>, SolverError> {
    let nev = wanted_idx.len();
    let mut eigenvalues  = Vec::with_capacity(nev);
    let mut eigenvectors = Vec::with_capacity(nev);
    let mut final_res    = Vec::with_capacity(nev);

    for &wi in wanted_idx {
        let lam = schur_vals[wi];
        let mut x = DenseVec::zeros(n);
        for j in 0..ncv { x.axpy(schur_vecs[wi][j], &v_cols[j]); }
        normalise(&mut x);
        let mut ax = DenseVec::zeros(n);
        op.apply(&x, &mut ax);
        final_res.push(residual_norm(&ax, &x, lam));
        eigenvalues.push(lam);
        eigenvectors.push(x);
    }

    Ok(EigenResult { eigenvalues, eigenvectors, converged: nev, iterations, residuals: final_res })
}
