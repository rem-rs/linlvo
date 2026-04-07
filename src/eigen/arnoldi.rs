//! Implicitly Restarted Arnoldi Method (IRAM) — Sprint 8.
//!
//! For **non-symmetric** (and symmetric) operators. Builds an Arnoldi
//! factorisation `A Vₘ = Vₘ Hₘ + f eₘᵀ` where Hₘ is upper Hessenberg,
//! selects wanted Ritz pairs, then applies implicit QR restarts to compress
//! back to `nev` vectors before extending again.
//!
//! Convergence: `|h_{m+1,m}| |s_k[m-1]| / |θ_k| < tol` (Sorensen 1992).

use crate::core::{error::SolverError, operator::LinearOperator, scalar::Scalar, vector::{DenseVec, Vector}};
use super::{EigenParams, EigenResult, EigenSolver, EigenWhich, fill_random, residual_norm, dot, normalise, orthogonalise_against};
use super::lanczos::sort_ritz; // reuse sort helper

// ─── ArnoldiIter ─────────────────────────────────────────────────────────────

/// Implicitly Restarted Arnoldi Method — works for any square operator.
///
/// For **symmetric** problems prefer [`super::LanczosIter`] which is cheaper
/// (tridiagonal T vs full Hessenberg H).
///
/// # Parameters
/// - `ncv`: Krylov space size (must be > `n_eigenvalues`; default `2k+1`)
/// - `seed`: random seed
pub struct ArnoldiIter {
    pub ncv:  Option<usize>,
    pub seed: u64,
}

impl Default for ArnoldiIter {
    fn default() -> Self { ArnoldiIter { ncv: None, seed: 42 } }
}

impl ArnoldiIter {
    pub fn new(ncv: usize) -> Self { ArnoldiIter { ncv: Some(ncv), seed: 42 } }
}

impl<T: Scalar> EigenSolver<T> for ArnoldiIter {
    fn solve<Op>(&self, op: &Op, params: &EigenParams<T>) -> Result<EigenResult<T>, SolverError>
    where Op: LinearOperator<Vector = DenseVec<T>>
    {
        let n   = op.nrows();
        let nev = params.n_eigenvalues;
        assert_eq!(n, op.ncols(), "ArnoldiIter: operator must be square");
        assert!(nev >= 1 && nev < n, "nev must be in 1..n");

        let ncv = self.ncv.unwrap_or_else(|| n.min(nev + nev.max(20)));
        assert!(ncv > nev && ncv <= n, "ncv must satisfy nev < ncv <= n");

        // Starting vector
        let mut v0 = DenseVec::zeros(n);
        fill_random(&mut v0, self.seed);
        normalise(&mut v0);

        // Build initial Arnoldi factorisation of length ncv
        let (mut v_cols, mut h_mat, mut f) = arnoldi_extend(op, vec![v0], n, ncv);

        for outer in 0..params.max_iter {
            // Compute Ritz values from H (ncv×ncv upper Hessenberg)
            let (ritz_vals, ritz_vecs) = hessenberg_eig(&h_mat, ncv);

            // Sort and select wanted pairs
            let mut order: Vec<usize> = (0..ncv).collect();
            sort_ritz(&mut order, &ritz_vals, params.which);
            let wanted = order[..nev].to_vec();

            // Convergence: |h[ncv, ncv-1]| * |s_k[ncv-1]| / |θ_k| < tol
            let beta_m = f.norm2();
            let mut n_conv = 0usize;
            let mut residuals = vec![T::zero(); nev];
            for (ki, &wi) in wanted.iter().enumerate() {
                let theta = ritz_vals[wi];
                let s_last = ritz_vecs[wi][ncv - 1].abs();
                let lam_abs = if theta.abs() > T::from_f64(1e-14) { theta.abs() } else { T::one() };
                residuals[ki] = beta_m * s_last / lam_abs;
                if residuals[ki] < params.tol { n_conv += 1; }
            }

            if params.verbose {
                let max_r = residuals.iter().cloned()
                    .map(|r| num_traits::ToPrimitive::to_f64(&r).unwrap_or(f64::NAN))
                    .fold(0f64, f64::max);
                println!("  Arnoldi outer {:3}  n_conv={}/{}  max‖r‖={:.3e}", outer, n_conv, nev, max_r);
            }

            if n_conv >= nev {
                let mut eigenvalues  = Vec::with_capacity(nev);
                let mut eigenvectors = Vec::with_capacity(nev);
                let mut final_res    = Vec::with_capacity(nev);
                for &wi in &wanted {
                    let lam = ritz_vals[wi];
                    let mut x = DenseVec::zeros(n);
                    for j in 0..ncv { x.axpy(ritz_vecs[wi][j], &v_cols[j]); }
                    normalise(&mut x);
                    let mut ax = DenseVec::zeros(n);
                    op.apply(&x, &mut ax);
                    final_res.push(residual_norm(&ax, &x, lam));
                    eigenvalues.push(lam);
                    eigenvectors.push(x);
                }
                return Ok(EigenResult {
                    eigenvalues, eigenvectors, converged: nev,
                    iterations: outer + 1, residuals: final_res,
                });
            }

            // ── Implicit QR restart with unwanted shifts ───────────────────
            let shifts: Vec<T> = order[nev..].iter().map(|&wi| ritz_vals[wi]).collect();
            let (new_h, q_full) = hessenberg_implicit_restart(&h_mat, &shifts, nev, ncv);

            // Rotate V: V_new[:,j] = V Q[:,j]  for j=0..nev
            let mut v_new: Vec<DenseVec<T>> = Vec::with_capacity(nev + 1);
            for j in 0..nev {
                let mut col = DenseVec::zeros(n);
                for i in 0..ncv { col.axpy(q_full[i * ncv + j], &v_cols[i]); }
                v_new.push(col);
            }

            // New residual: f_new = f * Q[ncv-1, nev-1] + v[nev] * h[nev, nev-1]
            let sigma   = q_full[(ncv - 1) * ncv + nev - 1];
            let h_nev   = new_h[nev * ncv + nev - 1]; // H[nev, nev-1]
            {
                let fs = f.as_mut_slice();
                let vs = v_new[nev - 1].as_slice();
                for i in 0..n { fs[i] = fs[i] * sigma + vs[i] * h_nev; }
            }
            let beta_f = f.norm2();
            let mut v_next = f.clone();
            if beta_f > T::from_f64(1e-14) { v_next.scale(T::one() / beta_f); }
            v_new.push(v_next);

            // Extend from nev+1 to ncv
            let (vc2, hm2, f2) = arnoldi_extend(op, v_new, n, ncv);
            // Patch leading nev×nev block of H from the restarted H
            let mut h_patch = hm2;
            for i in 0..nev {
                for j in 0..nev {
                    h_patch[i * ncv + j] = new_h[i * ncv + j];
                }
            }
            v_cols = vc2; h_mat = h_patch; f = f2;
        }

        Err(SolverError::ConvergenceFailed { max_iter: params.max_iter, residual: f64::INFINITY })
    }
}

// ─── Arnoldi extension ────────────────────────────────────────────────────────

/// Extend an Arnoldi basis `v_init` (already orthonormal) to `target` columns.
/// Returns `(V, H_flat, f)`:
/// - V: target orthonormal columns
/// - H_flat: row-major `target × target` upper-Hessenberg matrix
/// - f: unnormalised residual after the last step
pub(crate) fn arnoldi_extend<T, Op>(
    op: &Op,
    mut v: Vec<DenseVec<T>>,
    n: usize,
    target: usize,
) -> (Vec<DenseVec<T>>, Vec<T>, DenseVec<T>)
where
    T: Scalar,
    Op: LinearOperator<Vector = DenseVec<T>>,
{
    let start = v.len() - 1;
    let mut h = vec![T::zero(); target * target];
    let mut f = DenseVec::zeros(n);

    for j in start..target {
        // w = A v_j
        let mut w = DenseVec::zeros(n);
        op.apply(&v[j], &mut w);

        // Modified Gram-Schmidt orthogonalisation
        for i in 0..=j {
            let hij = dot(w.as_slice(), v[i].as_slice());
            h[i * target + j] = hij;
            let ws = w.as_mut_slice();
            let vi = v[i].as_slice();
            for k in 0..n { ws[k] -= hij * vi[k]; }
        }
        // Re-orthogonalise once to improve numerical stability
        for i in 0..=j {
            let corr = dot(w.as_slice(), v[i].as_slice());
            h[i * target + j] += corr;
            let ws = w.as_mut_slice();
            let vi = v[i].as_slice();
            for k in 0..n { ws[k] -= corr * vi[k]; }
        }

        f = w.clone();
        let beta = w.norm2();
        if j + 1 < target {
            h[(j + 1) * target + j] = beta;
            if beta > T::from_f64(1e-14) {
                w.scale(T::one() / beta);
                v.push(w);
            } else {
                // Lucky breakdown — random restart
                let mut r = DenseVec::zeros(n);
                fill_random(&mut r, 31 + j as u64);
                orthogonalise_against(&mut r, &v[..=j]);
                normalise(&mut r);
                v.push(r);
            }
        }
    }

    v.truncate(target);
    (v, h, f)
}

// ─── Dense upper-Hessenberg eigensolver (Francis QR) ─────────────────────────

/// Eigenvalues + eigenvectors of upper-Hessenberg H via double-shift QR.
/// Returns `(ritz_vals, ritz_vecs)` where both are length `n`.
/// (Complex eigenvalues are approximated as real; adequate for IRAM targeting
///  real eigenvalues of real operators.)
/// Eigenvalues + eigenvectors of upper-Hessenberg H (native: via nalgebra).
/// Returns `(ritz_vals, ritz_vecs)` where `ritz_vecs[i]` is column i.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn hessenberg_eig<T: Scalar>(h: &[T], n: usize) -> (Vec<T>, Vec<Vec<T>>) {
    use nalgebra::{DMatrix, Schur};
    // Build nalgebra matrix (column-major from our row-major h)
    let mat = DMatrix::<f64>::from_fn(n, n, |r, c|
        num_traits::ToPrimitive::to_f64(&h[r * n + c]).unwrap_or(0.0)
    );
    let schur = Schur::new(mat);
    let (q, t) = schur.unpack();
    // Ritz values = diagonal of quasi-upper-triangular T
    let eigenvalues: Vec<T> = (0..n).map(|i| T::from_f64(t[(i, i)])).collect();
    // Eigenvectors = columns of Q
    let eigenvectors: Vec<Vec<T>> = (0..n)
        .map(|j| (0..n).map(|i| T::from_f64(q[(i, j)])).collect())
        .collect();
    (eigenvalues, eigenvectors)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn hessenberg_eig<T: Scalar>(h: &[T], n: usize) -> (Vec<T>, Vec<Vec<T>>) {
    let evals = (0..n).map(|i| h[i * n + i]).collect();
    let evecs = (0..n).map(|j| { let mut v = vec![T::zero(); n]; v[j] = T::one(); v }).collect();
    (evals, evecs)
}

// ─── Hessenberg implicit QR restart ──────────────────────────────────────────

/// Apply `p = ncv - nev` implicit QR steps with shifts to H,
/// accumulating Q. Returns `(new_H_flat, Q_flat)`.
fn hessenberg_implicit_restart<T: Scalar>(
    h: &[T],
    shifts: &[T],
    nev: usize,
    ncv: usize,
) -> (Vec<T>, Vec<T>) {
    let mut a = h.to_vec();
    let mut q = vec![T::zero(); ncv * ncv];
    for i in 0..ncv { q[i * ncv + i] = T::one(); }

    for &mu in shifts {
        // Compute first column of (H - mu I)
        let x = a[0] - mu;
        let y = a[ncv]; // a[1,0]

        let (mut c, mut s) = {
            let r = (x * x + y * y).sqrt();
            if r < T::from_f64(1e-15) { (T::one(), T::zero()) } else { (x / r, y / r) }
        };

        for j in 0..ncv - 1 {
            // Apply Givens from left to rows j, j+1
            for k in j..ncv {
                let t1 = a[j * ncv + k];
                let t2 = a[(j + 1) * ncv + k];
                a[j * ncv + k]       =  c * t1 + s * t2;
                a[(j + 1) * ncv + k] = -s * t1 + c * t2;
            }
            // Apply Givens from right to cols j, j+1 (upper Hessenberg: rows 0..j+2)
            let row_end = (j + 2).min(ncv);
            for k in 0..row_end {
                let t1 = a[k * ncv + j];
                let t2 = a[k * ncv + j + 1];
                a[k * ncv + j]     =  c * t1 + s * t2;
                a[k * ncv + j + 1] = -s * t1 + c * t2;
            }
            // Accumulate in Q
            for k in 0..ncv {
                let t1 = q[k * ncv + j];
                let t2 = q[k * ncv + j + 1];
                q[k * ncv + j]     =  c * t1 + s * t2;
                q[k * ncv + j + 1] = -s * t1 + c * t2;
            }

            // Next Givens for bulge chasing
            if j + 2 < ncv {
                let xn = a[(j + 1) * ncv + j];
                let yn = a[(j + 2) * ncv + j];
                let r = (xn * xn + yn * yn).sqrt();
                (c, s) = if r < T::from_f64(1e-15) { (T::one(), T::zero()) } else { (xn / r, yn / r) };
            }
        }
    }

    (a, q)
}
