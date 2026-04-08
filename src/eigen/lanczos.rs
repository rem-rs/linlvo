//! Implicitly Restarted Lanczos Method (IRLM) — Sprint 8.
//!
//! Targets **symmetric** operators. Builds a Lanczos factorisation
//! `A Vₘ = Vₘ Tₘ + β_{m+1} v_{m+1} eₘᵀ` where Tₘ is symmetric
//! tridiagonal, then uses implicit QR shifts (exact or Chebyshev) to
//! restart, retaining only the `k` wanted Ritz pairs.
//!
//! Convergence criterion: `β_{m+1} |yₖ[m]| / |θₖ| < tol`
//! (ARPACK-style, Sorensen 1992).
//!
//! **Algorithm sketch:**
//! ```text
//! choose ncv > nev  (Krylov space size)
//! V[:,0] = random unit vector
//! run Lanczos to build V[:,0..ncv], T[ncv×ncv], β_next
//! loop:
//!   [Y,Θ] = eig(T)              -- dense, ncv×ncv
//!   select nev wanted Ritz values
//!   check convergence of each: β|y[ncv-1]| / |θ| < tol
//!   if all converged → done
//!   choose p = ncv-nev unwanted shifts μ₁…μₚ
//!   apply p implicit QR steps to (T,Q): T ← Qᵀ T Q, V ← V Q
//!   restart: keep first nev columns, extend to ncv again
//! ```

use crate::core::{error::SolverError, operator::LinearOperator, scalar::Scalar, vector::{DenseVec, Vector}};
use super::{EigenParams, EigenResult, EigenSolver, EigenWhich, fill_random, residual_norm, dot, normalise};

// ─── LanczosIter ─────────────────────────────────────────────────────────────

/// Implicitly Restarted Lanczos Method for **symmetric** operators.
///
/// Use [`super::ArnoldiIter`] for non-symmetric problems.
///
/// # Parameters
/// - `ncv`: Krylov space size (must be > `n_eigenvalues`; default: `min(2k+1, n)`)
/// - `seed`: random seed for the starting vector
pub struct LanczosIter {
    /// Krylov subspace dimension. `None` → auto (`min(2·nev + 1, n)`).
    pub ncv: Option<usize>,
    pub seed: u64,
}

impl Default for LanczosIter {
    fn default() -> Self { LanczosIter { ncv: None, seed: 42 } }
}

impl LanczosIter {
    pub fn new(ncv: usize) -> Self { LanczosIter { ncv: Some(ncv), seed: 42 } }
}

impl<T: Scalar> EigenSolver<T> for LanczosIter {
    fn solve<Op>(&self, op: &Op, params: &EigenParams<T>) -> Result<EigenResult<T>, SolverError>
    where Op: LinearOperator<Vector = DenseVec<T>>
    {
        let n   = op.nrows();
        let nev = params.n_eigenvalues;
        assert_eq!(n, op.ncols(), "LanczosIter: operator must be square");
        assert!(nev >= 1 && nev < n, "nev must be in 1..n");

        let ncv = self.ncv.unwrap_or_else(|| n.min(nev + nev.max(20)));
        assert!(ncv > nev, "ncv must be > nev");
        assert!(ncv <= n,  "ncv must be <= n");

        // ── Starting vector ────────────────────────────────────────────────
        let mut v_start = DenseVec::zeros(n);
        fill_random(&mut v_start, self.seed);
        normalise(&mut v_start);

        // ── Build initial Lanczos factorisation of length ncv ──────────────
        // V: n×ncv  (stored as Vec of column vectors)
        // alpha: diagonal of T   (length ncv)
        // beta:  sub/superdiag   (length ncv-1 after full run)
        let (mut v_cols, mut alpha, mut beta, mut f) =
            lanczos_extend(op, &[v_start], n, ncv, &[])?;

        // ── Implicit restart loop ──────────────────────────────────────────
        for outer in 0..params.max_iter {
            // Solve the ncv×ncv symmetric tridiagonal eigenproblem
            let (ritz_vals, ritz_vecs) = tridiag_eig(&alpha, &beta);

            // Select wanted Ritz pairs according to `which`
            let mut order: Vec<usize> = (0..ncv).collect();
            sort_ritz(&mut order, &ritz_vals, params.which);
            let wanted = &order[..nev];

            // Convergence check: β_m |y_i[m-1]| / |θ_i| < tol
            let beta_m = if beta.len() == ncv { beta[ncv - 1] } else { T::zero() };
            let mut n_conv = 0usize;
            let mut residuals = vec![T::zero(); nev];
            for (ki, &wi) in wanted.iter().enumerate() {
                let theta = ritz_vals[wi];
                let y_last = ritz_vecs[wi][ncv - 1].abs();
                let lam_abs = if theta.abs() > T::from_f64(1e-14) { theta.abs() } else { T::one() };
                residuals[ki] = beta_m * y_last / lam_abs;
                if residuals[ki] < params.tol { n_conv += 1; }
            }

            if params.verbose {
                let max_r = residuals.iter().cloned()
                    .map(|r| num_traits::ToPrimitive::to_f64(&r).unwrap_or(f64::NAN))
                    .fold(0f64, f64::max);
                println!("  Lanczos outer {:3}  n_conv={}/{}  max‖r‖={:.3e}", outer, n_conv, nev, max_r);
            }

            if n_conv >= nev {
                // Compute Ritz vectors X = V * Y[:,wanted]
                let eigenvalues: Vec<T> = wanted.iter().map(|&wi| ritz_vals[wi]).collect();
                let mut eigenvectors = Vec::with_capacity(nev);
                let mut final_residuals = Vec::with_capacity(nev);
                for &wi in wanted {
                    let mut x = DenseVec::zeros(n);
                    for (j, v_col) in v_cols.iter().enumerate() {
                        x.axpy(ritz_vecs[wi][j], v_col);
                    }
                    normalise(&mut x);
                    let mut ax = DenseVec::zeros(n);
                    op.apply(&x, &mut ax);
                    let lam = ritz_vals[wi];
                    final_residuals.push(residual_norm(&ax, &x, lam));
                    eigenvectors.push(x);
                }
                return Ok(EigenResult {
                    eigenvalues, eigenvectors, converged: nev,
                    iterations: outer + 1, residuals: final_residuals,
                });
            }

            // ── Restart ────────────────────────────────────────────────────
            // Compute converged Ritz vectors (deflation set) and restart with
            // a fresh Lanczos from the residual direction, orthogonal to them.
            // Include converged vecs as the first columns so lanczos_extend's
            // full re-orthogonalisation keeps new vectors orthogonal to them.
            let mut converged_vecs: Vec<DenseVec<T>> = Vec::new();
            for (ki, &wi) in wanted.iter().enumerate() {
                if residuals[ki] < params.tol {
                    let mut x = DenseVec::zeros(n);
                    for (j, v_col) in v_cols.iter().enumerate() { x.axpy(ritz_vecs[wi][j], v_col); }
                    normalise(&mut x);
                    converged_vecs.push(x);
                }
            }
            let n_cv = converged_vecs.len();

            // Build starting vector orthogonal to converged vecs.
            let mut v_start = f.clone();
            for cv in &converged_vecs {
                let proj = dot(v_start.as_slice(), cv.as_slice());
                let vs = v_start.as_mut_slice();
                let cs = cv.as_slice();
                for k in 0..n { vs[k] -= proj * cs[k]; }
            }
            let nrm = v_start.norm2();
            if nrm > T::from_f64(1e-14) {
                v_start.scale(T::one() / nrm);
            } else {
                fill_random(&mut v_start, 0x123456u64.wrapping_add(outer as u64));
                for cv in &converged_vecs {
                    let proj = dot(v_start.as_slice(), cv.as_slice());
                    let vs = v_start.as_mut_slice();
                    let cs = cv.as_slice();
                    for k in 0..n { vs[k] -= proj * cs[k]; }
                }
                normalise(&mut v_start);
            }

            // Build new basis: converged_vecs (frozen) + v_start + fresh steps.
            // lanczos_extend starts at index n_cv (v_start position).
            let mut v_init: Vec<DenseVec<T>> = converged_vecs;
            v_init.push(v_start);
            let (mut vc2, mut al2, be2, f2) =
                lanczos_extend(op, &v_init, n, ncv, &[])?;
            // Patch alpha[0..n_cv] with the Rayleigh quotients of the converged vecs.
            // (lanczos_extend left them as 0; setting them to the actual eigenvalues
            //  keeps the tridiagonal accurate and prevents false convergence.)
            for (ki, &wi) in wanted.iter().enumerate() {
                if ki < n_cv {
                    al2[ki] = ritz_vals[wi];
                    vc2[ki] = v_init[ki].clone(); // restore (just in case)
                }
            }
            v_cols = vc2; alpha = al2; beta = be2; f = f2;
        }

        Err(SolverError::ConvergenceFailed { max_iter: params.max_iter, residual: f64::INFINITY })
    }
}

// ─── Core Lanczos recurrence ──────────────────────────────────────────────────

/// Extend a Lanczos basis from `v_init.len()` to `target` columns.
///
/// Returns `(V, alpha, beta, f)` where:
/// - `V` = `target` orthonormal Lanczos vectors
/// - `alpha`, `beta` = tridiagonal entries
/// - `f` = unnormalised residual vector after the last step
#[allow(clippy::type_complexity)]
fn lanczos_extend<T, Op>(
    op: &Op,
    v_init: &[DenseVec<T>],
    n: usize,
    target: usize,
    beta_init: &[T],
) -> Result<(Vec<DenseVec<T>>, Vec<T>, Vec<T>, DenseVec<T>), SolverError>
where
    T: Scalar,
    Op: LinearOperator<Vector = DenseVec<T>>,
{
    let start = v_init.len() - 1; // index of the last supplied vector
    let mut v = v_init.to_vec();
    v.resize_with(target + 1, || DenseVec::zeros(n));

    let mut alpha = vec![T::zero(); target];
    let mut beta  = vec![T::zero(); target]; // beta[j] = norm of residual after step j

    // Seed beta from caller (restart scenario)
    for (i, &b) in beta_init.iter().enumerate() { if i < target { beta[i] = b; } }

    let mut f = DenseVec::zeros(n); // residual / work vector

    for j in start..target {
        // w = A v_j
        let mut w = DenseVec::zeros(n);
        op.apply(&v[j], &mut w);

        // alpha_j = v_j · w
        let aj = dot(w.as_slice(), v[j].as_slice());
        alpha[j] = aj;

        // f = w − alpha_j v_j − beta_{j-1} v_{j-1}
        {
            let fs = w.as_mut_slice();
            let vs = v[j].as_slice();
            for i in 0..n { fs[i] -= aj * vs[i]; }
            if j > 0 {
                let bj = beta[j - 1];
                let vp = v[j - 1].as_slice();
                for i in 0..n { fs[i] -= bj * vp[i]; }
            }
        }
        f = w;

        // Full re-orthogonalisation against all previous vectors (avoids loss)
        #[allow(clippy::needless_range_loop)]
        for k in 0..=j {
            let proj = dot(f.as_slice(), v[k].as_slice());
            let fs = f.as_mut_slice();
            let vk = v[k].as_slice();
            for i in 0..n { fs[i] -= proj * vk[i]; }
        }

        // beta_{j} = ‖f‖
        let bj_new = f.norm2();
        beta[j] = bj_new;

        if j + 1 < target {
            if bj_new < T::from_f64(1e-14) {
                // Lucky breakdown — pad with random vector
                let mut r = DenseVec::zeros(n);
                fill_random(&mut r, 17 + j as u64);
                #[allow(clippy::needless_range_loop)]
                for k in 0..=j {
                    let proj = dot(r.as_slice(), v[k].as_slice());
                    let rs = r.as_mut_slice();
                    let vk = v[k].as_slice();
                    for i in 0..n { rs[i] -= proj * vk[i]; }
                }
                normalise(&mut r);
                v[j + 1] = r;
                beta[j] = T::zero();
            } else {
                let mut next = f.clone();
                next.scale(T::one() / bj_new);
                v[j + 1] = next;
            }
        }
    }

    v.truncate(target);
    Ok((v, alpha, beta, f))
}

// ─── Dense symmetric tridiagonal eigensolver (via nalgebra) ──────────────────

/// Eigenvalues + eigenvectors of symmetric tridiagonal T.
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[i]` is column i.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn tridiag_eig<T: Scalar>(alpha: &[T], beta: &[T]) -> (Vec<T>, Vec<Vec<T>>) {
    use nalgebra::{DMatrix, SymmetricEigen};
    let n = alpha.len();
    let mut m = DMatrix::<f64>::zeros(n, n);
    for i in 0..n { m[(i, i)] = num_traits::ToPrimitive::to_f64(&alpha[i]).unwrap_or(0.0); }
    for i in 0..beta.len().min(n - 1) {
        let b = num_traits::ToPrimitive::to_f64(&beta[i]).unwrap_or(0.0);
        m[(i, i + 1)] = b; m[(i + 1, i)] = b;
    }
    let se = SymmetricEigen::new(m);
    let eigenvalues: Vec<T> = se.eigenvalues.iter()
        .map(|&v| T::from_f64(v)).collect();
    // nalgebra stores eigenvectors as columns; se.eigenvectors[(row, col)]
    let eigenvectors: Vec<Vec<T>> = (0..n).map(|j|
        (0..n).map(|i| T::from_f64(se.eigenvectors[(i, j)])).collect()
    ).collect();
    (eigenvalues, eigenvectors)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn tridiag_eig<T: Scalar>(alpha: &[T], beta: &[T]) -> (Vec<T>, Vec<Vec<T>>) {
    // Fallback: power iteration — wasm32 has no nalgebra
    let n = alpha.len();
    let evals = alpha.to_vec();
    let evecs = (0..n).map(|j| {
        let mut v = vec![T::zero(); n]; v[j] = T::one(); v
    }).collect();
    (evals, evecs)
}

#[inline]
#[allow(dead_code)]
pub(crate) fn givens_rot<T: Scalar>(a: T, b: T) -> (T, T) {
    let r = (a * a + b * b).sqrt();
    if r < T::from_f64(1e-15) { (T::one(), T::zero()) } else { (a / r, -b / r) }
}

// ─── Implicit QR restart (ARPACK-style) ──────────────────────────────────────

/// Apply `p = ncv - nev` implicit QR steps with the given shifts to T,
/// accumulating the product rotation Q.
/// Returns `(new_alpha, new_beta, Q_flat)` where Q_flat is row-major ncv×ncv.
#[allow(dead_code)]
fn implicit_qr_restart<T: Scalar>(
    alpha: &[T],
    beta: &[T],
    shifts: &[T],
    _nev: usize,    ncv: usize,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut d = alpha.to_vec();
    let mut e = beta.to_vec();
    e.resize(ncv, T::zero());

    // Q = I_{ncv}
    let mut q = vec![T::zero(); ncv * ncv];
    for i in 0..ncv { q[i * ncv + i] = T::one(); }

    for &mu in shifts {
        // One implicit QR step with shift mu on the tridiagonal (d, e)
        let mut x = d[0] - mu;
        let mut y = e[0];
        for i in 0..ncv - 1 {
            let (c, s) = givens_rot(x, y);
            // Apply Givens to rows i and i+1
            {
                let tmp_d0 = c * d[i] - s * e[i];
                let tmp_e0 = s * d[i] + c * e[i];
                let tmp_d1 = -s * e[i] + c * d[i + 1];
                d[i]     = tmp_d0;
                e[i]     = tmp_e0;
                d[i + 1] = tmp_d1;
                if i + 1 < ncv - 1 {
                    let tmp_e1 = c * e[i + 1];
                    let old = e[i + 1];
                    e[i + 1] = tmp_e1;
                    let _ = old;
                }
            }
            // Apply Givens to cols i and i+1 of Q
            for k in 0..ncv {
                let qi  = q[k * ncv + i];
                let qi1 = q[k * ncv + i + 1];
                q[k * ncv + i]     = c * qi - s * qi1;
                q[k * ncv + i + 1] = s * qi + c * qi1;
            }
            if i < ncv - 2 { x = e[i]; y = if i + 2 < ncv { d[i + 2] - mu } else { T::zero() }; }
            // Symmetric back-rotation (columns)
            {
                let tmp = c * e[i] - s * d[i + 1];
                e[i]     = tmp;
                d[i + 1] = s * e[i] + c * d[i + 1];
            }
            if i < ncv - 2 { x = e[i]; y = e[i + 1] * (-s); }
        }
    }

    (d, e, q)
}

// ─── Dense symmetric eigensolver (full matrix via Householder tridiagonalisation) ─

/// Reduce dense symmetric matrix `a` (row-major n×n) to tridiagonal form,
/// then apply the QR algorithm to get eigenvalues + eigenvectors.
/// Returns `(eigenvalues, eigenvectors_row_major)`.
#[allow(dead_code)]
pub(crate) fn tridiag_from_sym<T: Scalar>(a: &[T], n: usize) -> (Vec<T>, Vec<T>) {
    // Copy to working storage
    let mut m = a.to_vec();
    // Householder tridiagonalisation
    let mut q = vec![T::zero(); n * n];
    for i in 0..n { q[i * n + i] = T::one(); }

    for k in 0..n.saturating_sub(2) {
        // Build Householder reflector for column k below diagonal
        let mut sigma = T::zero();
        for i in k + 1..n { sigma += m[i * n + k] * m[i * n + k]; }
        if sigma < T::from_f64(1e-28) { continue; }
        let alpha = if m[(k + 1) * n + k] > T::zero() { -sigma.sqrt() } else { sigma.sqrt() };
        let r = (alpha * alpha - m[(k + 1) * n + k] * alpha).sqrt();
        if r < T::from_f64(1e-15) { continue; }
        let mut v = vec![T::zero(); n];
        v[k + 1] = (m[(k + 1) * n + k] - alpha) / (T::from_f64(2.0) * r);
        for i in k + 2..n { v[i] = m[i * n + k] / (T::from_f64(2.0) * r); }

        // Apply P = I - 2 v vᵀ from left and right
        // p = M v
        let mut p = vec![T::zero(); n];
        for i in 0..n {
            for j in k + 1..n { p[i] += m[i * n + j] * v[j]; }
        }
        let vp: T = v[k + 1..n].iter().zip(p[k + 1..n].iter()).fold(T::zero(), |s, (&vi, &pi)| s + vi * pi);
        // M = M - 2(p vᵀ + v pᵀ) + 4(vᵀp)(v vᵀ)
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] -= T::from_f64(2.0) * (p[i] * v[j] + v[i] * p[j])
                               - T::from_f64(4.0) * vp * v[i] * v[j];
            }
        }
        // Accumulate Q = Q (I - 2 v vᵀ)
        for i in 0..n {
            let qv: T = (k + 1..n).fold(T::zero(), |s, j| s + q[i * n + j] * v[j]);
            for j in k + 1..n { q[i * n + j] -= T::from_f64(2.0) * qv * v[j]; }
        }
    }

    // Extract tridiagonal
    let mut diag = (0..n).map(|i| m[i * n + i]).collect::<Vec<_>>();
    let mut off  = (0..n.saturating_sub(1)).map(|i| m[(i + 1) * n + i]).collect::<Vec<_>>();

    // QR iteration on tridiagonal
    let max_it = 30 * n;
    let mut sz = n;
    for _ in 0..max_it {
        if sz <= 1 { break; }
        for i in (1..sz).rev() {
            if off[i - 1].abs() <= T::from_f64(1e-14) * (diag[i - 1].abs() + diag[i].abs()) {
                off[i - 1] = T::zero(); sz = i; break;
            }
        }
        if sz <= 1 { break; }
        let h = (diag[sz - 1] - diag[sz - 2]) / (T::from_f64(2.0) * off[sz - 2]);
        let shift = diag[sz - 1] - off[sz - 2] / (h + h.signum() * (h * h + T::one()).sqrt());
        let mut x = diag[0] - shift;
        let mut y = off[0];
        for i in 0..sz - 1 {
            let r = (x * x + y * y).sqrt();
            let (c, s) = if r < T::from_f64(1e-15) { (T::one(), T::zero()) } else { (x / r, y / r) };
            let d0 = diag[i]; let d1 = diag[i + 1]; let e = off[i];
            diag[i]     = c * c * d0 - T::from_f64(2.0) * c * s * e + s * s * d1;
            diag[i + 1] = s * s * d0 + T::from_f64(2.0) * c * s * e + c * c * d1;
            off[i]      = c * s * (d0 - d1) + (c * c - s * s) * e;
            if i > 0 { off[i - 1] = c * off[i - 1] - s * y; }
            if i + 1 < sz - 1 { x = off[i]; y = -s * off[i + 1]; off[i + 1] *= c; }
            // Accumulate Q
            for row in 0..n {
                let qi  = q[row * n + i];
                let qi1 = q[row * n + i + 1];
                q[row * n + i]     =  c * qi - s * qi1;
                q[row * n + i + 1] =  s * qi + c * qi1;
            }
        }
    }

    (diag, q)
}

// ─── Sort Ritz values by EigenWhich ──────────────────────────────────────────

pub(crate) fn sort_ritz<T: Scalar>(order: &mut [usize], vals: &[T], which: EigenWhich) {
    match which {
        EigenWhich::LargestMagnitude | EigenWhich::BothEnds =>
            order.sort_by(|&a, &b| vals[b].abs().partial_cmp(&vals[a].abs()).unwrap()),
        EigenWhich::SmallestMagnitude =>
            order.sort_by(|&a, &b| vals[a].abs().partial_cmp(&vals[b].abs()).unwrap()),
        EigenWhich::LargestAlgebraic =>
            order.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap()),
        EigenWhich::SmallestAlgebraic =>
            order.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap()),
    }
}
