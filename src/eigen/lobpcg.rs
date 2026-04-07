//! LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient — Sprint 10.
//!
//! Knyazev (2001) algorithm for the **symmetric** standard eigenvalue problem
//! `A x = λ x` (or generalized `A x = λ B x` with B = mass matrix).
//!
//! Finds the `k` algebraically **smallest** (or largest) eigenvalues without
//! forming a Krylov basis; uses a block CG recurrence over an augmented
//! subspace spanned by `{X, W, P}` (`W = preconditioned residual, P = previous
//! search direction`).
//!
//! **Why LOBPCG for FEA?**
//! - Natural block method: computes `k` modes simultaneously.
//! - Exploits `AMG` as preconditioner → mesh-independent convergence.
//! - Memory: `O(n·k)` — no Krylov history.
//! - Competitive with Lanczos for `k ≪ n`.
//!
//! **Algorithm (Knyazev 2001, Algorithm 4.1):**
//! ```text
//! X₀ = random n×k orthonormal block
//! R₀ = A X₀ − X₀ Λ₀             (block residual)
//! for iter = 0, 1, …:
//!     W = T⁻¹ R  (preconditioned residual; T⁻¹ ≈ A⁻¹)
//!     S = [X, W, P]              (search space, 3k columns; P=0 first iter)
//!     solve Rayleigh-Ritz on S:  [A_S, B_S] C = C Θ
//!     X_{k+1} = S C[:,0:k]
//!     P_{k+1} = W C_W + P C_P
//!     check ‖R‖ < tol
//! ```

use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    vector::{DenseVec, Vector},
};
use super::{EigenParams, EigenResult, EigenSolver, EigenWhich, fill_random, residual_norm, dot};

// ─── LOBPCG ──────────────────────────────────────────────────────────────────

/// LOBPCG eigensolver for symmetric positive definite operators.
///
/// Best combined with an AMG preconditioner (`linger::AmgPrecond`) for FEA
/// structural modal analysis.
///
/// Set `which = EigenWhich::SmallestAlgebraic` for structural modes (default),
/// or `LargestAlgebraic` for the top modes.
pub struct Lobpcg<'p, T: Scalar> {
    /// Optional preconditioner T⁻¹ ≈ A⁻¹.
    pub precond: Option<&'p dyn Preconditioner<Vector = DenseVec<T>>>,
    pub seed: u64,
}

impl<'p, T: Scalar> Default for Lobpcg<'p, T> {
    fn default() -> Self { Lobpcg { precond: None, seed: 42 } }
}

impl<'p, T: Scalar> Lobpcg<'p, T> {
    pub fn new(precond: Option<&'p dyn Preconditioner<Vector = DenseVec<T>>>) -> Self {
        Lobpcg { precond, seed: 42 }
    }
}

impl<'p, T: Scalar> EigenSolver<T> for Lobpcg<'p, T> {
    fn solve<Op>(&self, op: &Op, params: &EigenParams<T>) -> Result<EigenResult<T>, SolverError>
    where Op: LinearOperator<Vector = DenseVec<T>>
    {
        let n   = op.nrows();
        let k   = params.n_eigenvalues;
        assert_eq!(n, op.ncols(), "LOBPCG: operator must be square");
        assert!(k >= 1 && k < n, "nev must be in 1..n");

        // ── Initialise X: n×k random orthonormal block ─────────────────────
        let mut x_cols: Vec<DenseVec<T>> = Vec::with_capacity(k);
        for j in 0..k {
            let mut col = DenseVec::zeros(n);
            fill_random(&mut col, self.seed.wrapping_add(j as u64 * 0xdeadbeef));
            // Orthogonalise against previous X columns
            for prev in &x_cols {
                let proj = dot(col.as_slice(), prev.as_slice());
                let cs = col.as_mut_slice();
                let ps = prev.as_slice();
                for i in 0..n { cs[i] -= proj * ps[i]; }
            }
            let nrm = col.norm2();
            if nrm > T::zero() { col.scale(T::one() / nrm); }
            x_cols.push(col);
        }

        // AX = A * X columns
        let mut ax_cols: Vec<DenseVec<T>> = x_cols.iter().map(|x| {
            let mut ax = DenseVec::zeros(n); op.apply(x, &mut ax); ax
        }).collect();

        // Λ = diag(xᵢᵀ A xᵢ)
        let mut lambdas: Vec<T> = x_cols.iter().zip(ax_cols.iter())
            .map(|(x, ax)| dot(ax.as_slice(), x.as_slice())).collect();

        // P columns (previous search directions): start zero
        let mut p_cols: Vec<Option<DenseVec<T>>> = vec![None; k];

        for iter in 0..params.max_iter {
            // ── Compute residuals R = AX − X Λ ─────────────────────────────
            let mut r_cols: Vec<DenseVec<T>> = (0..k).map(|j| {
                let mut r = DenseVec::zeros(n);
                let axs = ax_cols[j].as_slice();
                let xs  = x_cols[j].as_slice();
                let rs  = r.as_mut_slice();
                for i in 0..n { rs[i] = axs[i] - lambdas[j] * xs[i]; }
                r
            }).collect();

            // ── Convergence check ───────────────────────────────────────────
            let res_norms: Vec<T> = r_cols.iter().map(|r| r.norm2()).collect();
            let lam_max = lambdas.iter().cloned().map(|l| if l < T::zero() { -l } else { l })
                .fold(T::one(), |a, b| if b > a { b } else { a });
            let max_rel = res_norms.iter().cloned()
                .map(|r| r / lam_max)
                .fold(T::zero(), |a, b| if b > a { b } else { a });

            if params.verbose {
                let mr = num_traits::ToPrimitive::to_f64(&max_rel).unwrap_or(f64::NAN);
                println!("  LOBPCG iter {:4}  max‖r‖/|λ| = {mr:.3e}", iter + 1);
            }

            if max_rel < params.tol {
                let eigenvalues = lambdas.clone();
                let eigenvectors = x_cols.clone();
                let residuals = res_norms;
                return Ok(EigenResult { eigenvalues, eigenvectors, converged: k,
                    iterations: iter + 1, residuals });
            }

            // ── Precondition residuals: W = T⁻¹ R ─────────────────────────
            let mut w_cols: Vec<DenseVec<T>> = r_cols.iter().map(|r| {
                if let Some(m) = self.precond {
                    let mut w = DenseVec::zeros(n);
                    m.apply_precond(r, &mut w);
                    w
                } else {
                    r.clone()
                }
            }).collect();

            // Orthogonalise W against X
            for w in w_cols.iter_mut() {
                for x in x_cols.iter() {
                    let proj = dot(w.as_slice(), x.as_slice());
                    let ws = w.as_mut_slice();
                    let xs = x.as_slice();
                    for i in 0..n { ws[i] -= proj * xs[i]; }
                }
                let nrm = w.norm2();
                if nrm > T::from_f64(1e-14) { w.scale(T::one() / nrm); }
            }

            // ── Rayleigh-Ritz on S = [X, W, P] ─────────────────────────────
            // Build the block subspace and solve the small dense eigenproblem.
            // Cap s_width at n to avoid rank-deficient Gram matrix.
            let has_p = p_cols[0].is_some() && (3 * k <= n);
            let s_width = if has_p { 3 * k } else { (2 * k).min(n) };
            let mut s: Vec<DenseVec<T>> = Vec::with_capacity(s_width);
            s.extend_from_slice(&x_cols[..s_width.min(k)]);
            let w_cnt = (s_width - s.len()).min(k);
            s.extend_from_slice(&w_cols[..w_cnt]);
            if has_p && s.len() < s_width {
                let p_cnt = s_width - s.len();
                for p in p_cols.iter().take(p_cnt) { s.push(p.as_ref().unwrap().clone()); }
            }

            // A_S[i,j] = sᵢᵀ A sⱼ
            let mut as_cols: Vec<DenseVec<T>> = s.iter().map(|sv| {
                let mut asv = DenseVec::zeros(n); op.apply(sv, &mut asv); asv
            }).collect();

            let m = s_width;
            let mut a_s = vec![T::zero(); m * m];
            let mut b_s = vec![T::zero(); m * m]; // B = I for standard problem
            for i in 0..m {
                for j in 0..m {
                    a_s[i * m + j] = dot(s[i].as_slice(), as_cols[j].as_slice());
                    b_s[i * m + j] = dot(s[i].as_slice(), s[j].as_slice());
                }
            }

            // Solve the small dense symmetric generalised problem
            // A_S c = θ B_S c  via Cholesky B_S = L Lᵀ, then standard eig of L⁻¹ A_S L⁻ᵀ
            let (theta, c_vecs) = dense_symm_eig_gen(&a_s, &b_s, m);

            // Sort by algebraic order matching `which`
            let mut order: Vec<usize> = (0..m).collect();
            match params.which {
                EigenWhich::SmallestAlgebraic | EigenWhich::SmallestMagnitude =>
                    order.sort_by(|&a, &b| theta[a].partial_cmp(&theta[b]).unwrap()),
                _ =>
                    order.sort_by(|&a, &b| theta[b].partial_cmp(&theta[a]).unwrap()),
            }

            // ── Update X, P, AX ────────────────────────────────────────────
            let mut x_new:  Vec<DenseVec<T>> = Vec::with_capacity(k);
            let mut ax_new: Vec<DenseVec<T>> = Vec::with_capacity(k);
            let mut p_new:  Vec<DenseVec<T>> = Vec::with_capacity(k);

            for ki in 0..k {
                let col_idx = order[ki];
                // X_new[:,ki] = S * c[col_idx]
                let mut xn  = DenseVec::zeros(n);
                let mut axn = DenseVec::zeros(n);
                let mut pn  = DenseVec::zeros(n);
                for j in 0..m {
                    let cj = c_vecs[col_idx * m + j];
                    xn.axpy(cj,  &s[j]);
                    axn.axpy(cj, &as_cols[j]);
                    // P_new = W part + P part of S * c
                    if j >= k { // W and P columns
                        pn.axpy(cj, &s[j]);
                    }
                }
                // Normalise X_new
                let nrm = xn.norm2();
                if nrm > T::from_f64(1e-14) {
                    let inv = T::one() / nrm;
                    xn.scale(inv); axn.scale(inv); pn.scale(inv);
                }
                x_new.push(xn);
                ax_new.push(axn);
                p_new.push(pn);
                lambdas[ki] = theta[col_idx];
            }

            // B-orthogonalise X_new (Gram-Schmidt, B = I)
            for j in 1..k {
                for i in 0..j {
                    let proj = dot(x_new[j].as_slice(), x_new[i].as_slice());
                    let (left, right) = x_new.split_at_mut(j);
                    let xj = &mut right[0];
                    let xi = &left[i];
                    let xjs = xj.as_mut_slice();
                    let xis = xi.as_slice();
                    for idx in 0..n { xjs[idx] -= proj * xis[idx]; }
                    // same for ax_new
                    let proj2 = proj;
                    let (al, ar) = ax_new.split_at_mut(j);
                    let axj = &mut ar[0];
                    let axi = &al[i];
                    let axjs = axj.as_mut_slice();
                    let axis = axi.as_slice();
                    for idx in 0..n { axjs[idx] -= proj2 * axis[idx]; }
                }
                let nrm = x_new[j].norm2();
                if nrm > T::from_f64(1e-14) {
                    let inv = T::one() / nrm;
                    x_new[j].scale(inv); ax_new[j].scale(inv);
                }
            }

            x_cols  = x_new;
            ax_cols = ax_new;
            p_cols  = p_new.into_iter().map(|p| Some(p)).collect();
        }

        let eigenvalues  = lambdas;
        let residuals: Vec<T> = x_cols.iter().zip(ax_cols.iter()).zip(eigenvalues.iter())
            .map(|((x, ax), &lam)| residual_norm(ax, x, lam))
            .collect();
        let max_res = residuals.iter().cloned().fold(T::zero(), |a, b| if b > a { b } else { a });
        Err(SolverError::ConvergenceFailed {
            max_iter: params.max_iter,
            residual: num_traits::ToPrimitive::to_f64(&max_res).unwrap_or(f64::INFINITY),
        })
    }
}

// ─── Dense symmetric generalised eigensolver ─────────────────────────────────

/// Solve `A c = θ B c` for small dense symmetric matrices.
/// Returns `(eigenvalues, eigenvectors_flat_col_major)` — column j of evecs
/// is the j-th eigenvector (length n), stored consecutively.
#[cfg(not(target_arch = "wasm32"))]
fn dense_symm_eig_gen<T: Scalar>(a: &[T], b: &[T], n: usize) -> (Vec<T>, Vec<T>) {
    use nalgebra::{DMatrix, SymmetricEigen};

    let na = DMatrix::<f64>::from_fn(n, n, |r, c|
        num_traits::ToPrimitive::to_f64(&a[r * n + c]).unwrap_or(0.0));
    let nb = DMatrix::<f64>::from_fn(n, n, |r, c|
        num_traits::ToPrimitive::to_f64(&b[r * n + c]).unwrap_or(0.0));

    // Cholesky B = L Lᵀ
    let chol = match nb.clone().cholesky() {
        Some(c) => c,
        None => {
            // B not positive definite — fall back to standard problem
            let se = SymmetricEigen::new(na);
            let evals: Vec<T> = se.eigenvalues.iter().map(|&v| T::from_f64(v)).collect();
            let mut evecs = vec![T::zero(); n * n];
            for j in 0..n { for i in 0..n { evecs[j * n + i] = T::from_f64(se.eigenvectors[(i, j)]); } }
            return (evals, evecs);
        }
    };
    let l = chol.l();
    let li = match l.clone().try_inverse() {
        Some(m) => m,
        None => DMatrix::identity(n, n),
    };

    // C = L⁻¹ A (L⁻¹)ᵀ  — symmetric
    let c = &li * &na * li.transpose();
    let se = SymmetricEigen::new(c);

    let evals: Vec<T> = se.eigenvalues.iter().map(|&v| T::from_f64(v)).collect();
    // Recover original eigenvectors: v = (L⁻¹)ᵀ z
    let vecs = li.transpose() * &se.eigenvectors;
    let mut evecs = vec![T::zero(); n * n];
    for j in 0..n { for i in 0..n { evecs[j * n + i] = T::from_f64(vecs[(i, j)]); } }
    (evals, evecs)
}

#[cfg(target_arch = "wasm32")]
fn dense_symm_eig_gen<T: Scalar>(a: &[T], _b: &[T], n: usize) -> (Vec<T>, Vec<T>) {
    // Fallback: return diagonal elements as eigenvalues, identity eigenvectors
    let evals = (0..n).map(|i| a[i * n + i]).collect();
    let mut evecs = vec![T::zero(); n * n];
    for i in 0..n { evecs[i * n + i] = T::one(); }
    (evals, evecs)
}
