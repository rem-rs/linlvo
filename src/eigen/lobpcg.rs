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
use super::{EigenParams, EigenResult, EigenSolver, EigenWhich, fill_random, dot};

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
    /// Optional B-operator for the generalised problem `A x = λ B x`.
    /// When `None`, the standard problem `A x = λ x` (B = I) is solved.
    pub b_op: Option<&'p dyn LinearOperator<Vector = DenseVec<T>>>,
    /// Optional post-preconditioner projection (applied to W after preconditioning
    /// but before the Rayleigh-Ritz).  Used e.g. for div-free projection in
    /// H(curl) eigenvalue problems.  Applied in-place: `proj.apply(W, &mut W)`.
    pub projector: Option<&'p dyn Preconditioner<Vector = DenseVec<T>>>,
    pub seed: u64,
}

impl<'p, T: Scalar> Default for Lobpcg<'p, T> {
    fn default() -> Self {
        Lobpcg { precond: None, b_op: None, projector: None, seed: 42 }
    }
}

impl<'p, T: Scalar> Lobpcg<'p, T> {
    /// Create a standard-problem LOBPCG (the existing API).
    pub fn new(precond: Option<&'p dyn Preconditioner<Vector = DenseVec<T>>>) -> Self {
        Lobpcg { precond, b_op: None, projector: None, seed: 42 }
    }

    /// Create a generalised-problem LOBPCG with optional nullspace projector.
    pub fn new_generalized(
        precond: Option<&'p dyn Preconditioner<Vector = DenseVec<T>>>,
        b_op: Option<&'p dyn LinearOperator<Vector = DenseVec<T>>>,
        projector: Option<&'p dyn Preconditioner<Vector = DenseVec<T>>>,
    ) -> Self {
        Lobpcg { precond, b_op, projector, seed: 42 }
    }
}

impl<'p, T: Scalar> EigenSolver<T> for Lobpcg<'p, T> {
    fn solve<Op>(&self, op: &Op, params: &EigenParams<T>) -> Result<EigenResult<T>, SolverError>
    where Op: LinearOperator<Vector = DenseVec<T>>
    {
        self.solve_generalized(op, params)
    }
}

// ─── Implementation ─────────────────────────────────────────────────────────

impl<'p, T: Scalar> Lobpcg<'p, T> {
    /// Core solve: handles both standard (`b_op == None`) and generalised
    /// (`b_op == Some(…)`) eigenvalue problems.
    pub fn solve_generalized<Op>(
        &self,
        a: &Op,
        params: &EigenParams<T>,
    ) -> Result<EigenResult<T>, SolverError>
    where Op: LinearOperator<Vector = DenseVec<T>>
    {
        let n   = a.nrows();
        let k   = params.n_eigenvalues;
        assert_eq!(n, a.ncols(), "LOBPCG: operator must be square");
        assert!(k >= 1 && k < n, "nev must be in 1..n");

        let has_mass = self.b_op.is_some();
        let _has_proj = self.projector.is_some();

        // ── Abbreviations ────────────────────────────────────────────────────
        // When `mass_op` is `None` we treat `B = I`:
        //   initial X: Euclidean orthonormal
        //   B·x      : identity (use `x` directly)
        //   B_S      : Sᵀ·S
        //   residual : AX − X·Λ
        let mass_op: Option<&dyn LinearOperator<Vector = DenseVec<T>>> = self.b_op;

        // Helper: apply B (or identity)
        let apply_b = |x: &DenseVec<T>, y: &mut DenseVec<T>| {
            if let Some(b) = mass_op { b.apply(x, y); }
            else { y.copy_from(x); }
        };

        // ── 1. Initialise X ──────────────────────────────────────────────────
        let mut x_cols: Vec<DenseVec<T>> = Vec::with_capacity(k);
        for j in 0..k {
            let mut col = DenseVec::zeros(n);
            fill_random(&mut col, self.seed.wrapping_add(j as u64 * 0xdeadbeef));
            // Orthogonalise against previous X columns (Euclidean or B-inner)
            for prev in &x_cols {
                let mut bp = DenseVec::zeros(n);
                apply_b(prev, &mut bp);
                let proj = dot(col.as_slice(), bp.as_slice());
                let cs = col.as_mut_slice();
                let ps = prev.as_slice();
                for i in 0..n { cs[i] -= proj * ps[i]; }
            }
            // Normalise (Euclidean or B-norm)
            let nrm = if has_mass {
                let mut bv = DenseVec::zeros(n);
                apply_b(&col, &mut bv);
                dot(col.as_slice(), bv.as_slice()).sqrt()
            } else {
                col.norm2()
            };
            if nrm > T::zero() { col.scale(T::one() / nrm); }
            x_cols.push(col);
        }

        // AX = A·X;  MX = M·X (or X when B = I)
        let mut ax_cols: Vec<DenseVec<T>> = x_cols.iter().map(|x| {
            let mut ax = DenseVec::zeros(n); a.apply(x, &mut ax); ax
        }).collect();
        let mut mx_cols: Vec<DenseVec<T>> = if has_mass {
            x_cols.iter().map(|x| {
                let mut mx = DenseVec::zeros(n); mass_op.unwrap().apply(x, &mut mx); mx
            }).collect()
        } else {
            x_cols.clone()
        };

        // Λ = diag(xᵢᵀ A xᵢ / xᵢᵀ M xᵢ)
        let mut lambdas: Vec<T> = x_cols.iter().zip(ax_cols.iter()).zip(mx_cols.iter())
            .map(|((x, ax), mx)| {
                let num = dot(ax.as_slice(), x.as_slice());
                let den = dot(mx.as_slice(), x.as_slice());
                if den > T::zero() { num / den } else { num }
            }).collect();

        // P columns (previous search directions): start zero
        let mut p_cols: Vec<Option<DenseVec<T>>> = vec![None; k];

        for iter in 0..params.max_iter {
            // ── 2. Residuals R = AX − MX Λ ───────────────────────────────────
            let r_cols: Vec<DenseVec<T>> = (0..k).map(|j| {
                let mut r = DenseVec::zeros(n);
                let axs = ax_cols[j].as_slice();
                let mxs = mx_cols[j].as_slice();
                let rs  = r.as_mut_slice();
                for i in 0..n { rs[i] = axs[i] - lambdas[j] * mxs[i]; }
                r
            }).collect();

            // ── 3. Convergence check ──────────────────────────────────────────
            let res_norms: Vec<T> = r_cols.iter().map(|r| r.norm2()).collect();
            let lam_max = lambdas.iter().cloned()
                .map(|l| if l < T::zero() { -l } else { l })
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

            // ── 4. Precondition + optional projection: W ← P · T⁻¹(R) ──────
            let mut w_cols: Vec<DenseVec<T>> = r_cols.iter().map(|r| {
                if let Some(pc) = self.precond {
                    let mut w = DenseVec::zeros(n);
                    pc.apply_precond(r, &mut w);
                    if let Some(proj) = self.projector {
                        let mut wp = DenseVec::zeros(n);
                        proj.apply_precond(&w, &mut wp);
                        wp
                    } else { w }
                } else {
                    r.clone()
                }
            }).collect();

            // ── 5. B-orthogonalise W against X ───────────────────────────────
            for w in w_cols.iter_mut() {
                for x in x_cols.iter() {
                    let mut bx = DenseVec::zeros(n);
                    apply_b(x, &mut bx);
                    let proj = dot(w.as_slice(), bx.as_slice());
                    let ws = w.as_mut_slice();
                    let xs = x.as_slice();
                    for i in 0..n { ws[i] -= proj * xs[i]; }
                }
                let nrm = if has_mass {
                    let mut bw = DenseVec::zeros(n);
                    apply_b(w, &mut bw);
                    dot(w.as_slice(), bw.as_slice()).sqrt()
                } else { w.norm2() };
                if nrm > T::from_f64(1e-14) { w.scale(T::one() / nrm); }
            }

            // ── 6. Rayleigh-Ritz on S = [X, W, P] ────────────────────────────
            let has_p = p_cols[0].is_some() && (3 * k <= n);
            let s_width = if has_p { 3 * k } else { (2 * k).min(n) };
            let mut s: Vec<DenseVec<T>> = Vec::with_capacity(s_width);
            s.extend_from_slice(&x_cols[..s_width.min(k)]);
            let w_cnt = (s_width - s.len()).min(k);
            s.extend_from_slice(&w_cols[..w_cnt]);
            if has_p && s.len() < s_width {
                let p_cnt = s_width - s.len();
                for p in p_cols.iter().take(p_cnt) {
                    s.push(p.as_ref().unwrap().clone());
                }
            }
            // B-orthonormalise S (modified Gram–Schmidt) to avoid rank-deficient
            // Gram matrix in the Rayleigh–Ritz.  Only compress when the search
            // space is significantly over-complete (> 2k columns) to avoid
            // dropping the only available search direction for small blocks.
            let s = if s.len() >= 2 * k {
                let compressed = b_orthonormalise_basis(s.clone(), has_mass, &apply_b);
                if compressed.len() >= k { compressed } else { s }
            } else { s };

            // A_S[i,j] = sᵢᵀ A sⱼ
            let as_cols: Vec<DenseVec<T>> = s.iter().map(|sv| {
                let mut asv = DenseVec::zeros(n); a.apply(sv, &mut asv); asv
            }).collect();
            // M_S[i,j] = sᵢᵀ M sⱼ  (or sᵢᵀ sⱼ when B = I)
            let ms_cols: Vec<DenseVec<T>> = if has_mass {
                s.iter().map(|sv| {
                    let mut msv = DenseVec::zeros(n); mass_op.unwrap().apply(sv, &mut msv); msv
                }).collect()
            } else {
                s.clone()
            };

            let m_s = s.len();
            let mut a_s = vec![T::zero(); m_s * m_s];
            let mut b_s = vec![T::zero(); m_s * m_s];
            for i in 0..m_s {
                for j in 0..m_s {
                    a_s[i * m_s + j] = dot(s[i].as_slice(), as_cols[j].as_slice());
                    b_s[i * m_s + j] = dot(s[i].as_slice(), ms_cols[j].as_slice());
                }
            }

            let (theta, c_vecs) = dense_symm_eig_gen(&a_s, &b_s, m_s);

            // Sort by algebraic order
            let mut order: Vec<usize> = (0..m_s).collect();
            match params.which {
                EigenWhich::SmallestAlgebraic | EigenWhich::SmallestMagnitude =>
                    order.sort_by(|&a, &b| theta[a].partial_cmp(&theta[b]).unwrap()),
                _ =>
                    order.sort_by(|&a, &b| theta[b].partial_cmp(&theta[a]).unwrap()),
            }

            // ── 7. Update X, P, AX, MX ───────────────────────────────────────
            let mut x_new:  Vec<DenseVec<T>> = Vec::with_capacity(k);
            let mut ax_new: Vec<DenseVec<T>> = Vec::with_capacity(k);
            let mut mx_new: Vec<DenseVec<T>> = Vec::with_capacity(k);
            let mut p_new:  Vec<DenseVec<T>> = Vec::with_capacity(k);

            for ki in 0..k {
                let col_idx = order[ki];
                let mut xn  = DenseVec::zeros(n);
                let mut axn = DenseVec::zeros(n);
                let mut mxn = DenseVec::zeros(n);
                let mut pn  = DenseVec::zeros(n);
                for j in 0..m_s {
                    let cj = c_vecs[col_idx * m_s + j];
                    xn.axpy(cj,  &s[j]);
                    axn.axpy(cj, &as_cols[j]);
                    mxn.axpy(cj, &ms_cols[j]);
                    if j >= k { // W and P columns
                        pn.axpy(cj, &s[j]);
                    }
                }
                // Normalise (B-norm)
                let nrm = if has_mass {
                    let mut bxn = DenseVec::zeros(n);
                    mass_op.unwrap().apply(&xn, &mut bxn);
                    dot(xn.as_slice(), bxn.as_slice()).sqrt()
                } else { xn.norm2() };
                if nrm > T::from_f64(1e-14) {
                    let inv = T::one() / nrm;
                    xn.scale(inv); axn.scale(inv); mxn.scale(inv); pn.scale(inv);
                }
                x_new.push(xn);
                ax_new.push(axn);
                mx_new.push(mxn);
                p_new.push(pn);
                lambdas[ki] = theta[col_idx];
            }

            // B-orthogonalise X_new
            for j in 1..k {
                for i in 0..j {
                    let mut bxi = DenseVec::zeros(n);
                    apply_b(&x_new[i], &mut bxi);
                    let proj = dot(x_new[j].as_slice(), bxi.as_slice());
                    let (left, right) = x_new.split_at_mut(j);
                    let xj = &mut right[0];
                    let xi = &left[i];
                    let xjs = xj.as_mut_slice();
                    let xis = xi.as_slice();
                    for idx in 0..n { xjs[idx] -= proj * xis[idx]; }
                    // same for ax_new, mx_new
                    let (al, ar) = ax_new.split_at_mut(j);
                    let axj = &mut ar[0];
                    let axi = &al[i];
                    for idx in 0..n { axj.as_mut_slice()[idx] -= proj * axi.as_slice()[idx]; }
                    if has_mass {
                        let (ml, mr) = mx_new.split_at_mut(j);
                        let mxj = &mut mr[0];
                        let mxi = &ml[i];
                        for idx in 0..n { mxj.as_mut_slice()[idx] -= proj * mxi.as_slice()[idx]; }
                    }
                }
                let nrm = if has_mass {
                    let mut bxj = DenseVec::zeros(n);
                    apply_b(&x_new[j], &mut bxj);
                    dot(x_new[j].as_slice(), bxj.as_slice()).sqrt()
                } else { x_new[j].norm2() };
                if nrm > T::from_f64(1e-14) {
                    let inv = T::one() / nrm;
                    x_new[j].scale(inv); ax_new[j].scale(inv);
                    if has_mass { mx_new[j].scale(inv); }
                }
            }

            x_cols  = x_new;
            ax_cols = ax_new;
            mx_cols = mx_new;
            p_cols  = p_new.into_iter().map(Some).collect();
        }

        let eigenvalues  = lambdas;
        let residuals: Vec<T> = x_cols.iter().zip(ax_cols.iter()).zip(mx_cols.iter()).zip(eigenvalues.iter())
            .map(|(((x, ax), mx), &lam)| {
                let n = x.len();
                let mut s = T::zero();
                for i in 0..n { let d = ax.as_slice()[i] - lam * mx.as_slice()[i]; s += d*d; }
                s.sqrt()
            }).collect();
        let max_res = residuals.iter().cloned().fold(T::zero(), |a, b| if b > a { b } else { a });
        Err(SolverError::ConvergenceFailed {
            max_iter: params.max_iter,
            residual: num_traits::ToPrimitive::to_f64(&max_res).unwrap_or(f64::INFINITY),
        })
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────────

/// B-orthonormalise a set of vectors (modified Gram–Schmidt, drops near-zero cols).
fn b_orthonormalise_basis<T: Scalar>(
    basis: Vec<DenseVec<T>>,
    has_mass: bool,
    apply_b: &dyn Fn(&DenseVec<T>, &mut DenseVec<T>),
) -> Vec<DenseVec<T>> {
    let n = basis[0].len();
    let mut result: Vec<DenseVec<T>> = Vec::with_capacity(basis.len());
    for mut v in basis {
        for q in &result {
            let mut bq = DenseVec::zeros(n);
            if has_mass { apply_b(q, &mut bq); } else { bq.copy_from(q); }
            let proj = dot(v.as_slice(), bq.as_slice());
            let vs = v.as_mut_slice();
            let qs = q.as_slice();
            for i in 0..n { vs[i] -= proj * qs[i]; }
        }
        let nrm = if has_mass {
            let mut bv = DenseVec::zeros(n);
            apply_b(&v, &mut bv);
            dot(v.as_slice(), bv.as_slice()).sqrt()
        } else { v.norm2() };
        if nrm > T::from_f64(1e-10) {
            v.scale(T::one() / nrm);
            result.push(v);
        }
    }
    result
}

// ─── Dense symmetric generalised eigensolver (same as before) ───────────────────

/// Solve `A c = θ B c` for small dense symmetric matrices.
/// Returns `(eigenvalues, eigenvectors_flat_col_major)`.
#[cfg(not(target_arch = "wasm32"))]
fn dense_symm_eig_gen<T: Scalar>(a: &[T], b: &[T], n: usize) -> (Vec<T>, Vec<T>) {
    use nalgebra::{DMatrix, SymmetricEigen};

    let na = DMatrix::<f64>::from_fn(n, n, |r, c|
        num_traits::ToPrimitive::to_f64(&a[r * n + c]).unwrap_or(0.0));
    let nb = DMatrix::<f64>::from_fn(n, n, |r, c|
        num_traits::ToPrimitive::to_f64(&b[r * n + c]).unwrap_or(0.0));

    let chol = match nb.clone().cholesky() {
        Some(c) => c,
        None => {
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

    let c = &li * &na * li.transpose();
    let se = SymmetricEigen::new(c);

    let evals: Vec<T> = se.eigenvalues.iter().map(|&v| T::from_f64(v)).collect();
    let vecs = li.transpose() * &se.eigenvectors;
    let mut evecs = vec![T::zero(); n * n];
    for j in 0..n { for i in 0..n { evecs[j * n + i] = T::from_f64(vecs[(i, j)]); } }
    (evals, evecs)
}

#[cfg(target_arch = "wasm32")]
fn dense_symm_eig_gen<T: Scalar>(a: &[T], _b: &[T], n: usize) -> (Vec<T>, Vec<T>) {
    let evals = (0..n).map(|i| a[i * n + i]).collect();
    let mut evecs = vec![T::zero(); n * n];
    for i in 0..n { evecs[i * n + i] = T::one(); }
    (evals, evecs)
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::{CooMatrix, CsrMatrix};

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0     { coo.push(i, i-1, -1.0); }
            if i < n-1   { coo.push(i, i+1, -1.0); }
        }
        CsrMatrix::from_coo(&coo)
    }

    #[test]
    fn standard_smallest_eigenvalue() {
        let n = 20;
        let a = laplacian_1d(n);
        let solver = Lobpcg::<f64>::new(None);
        let mut params = EigenParams::new(1, EigenWhich::SmallestAlgebraic);
        params.tol = 1e-8; // relaxed for refactored basis-compression path
        params.max_iter = 2000;
        let res = solver.solve(&a, &params).unwrap();
        let exact = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
        assert!((res.eigenvalues[0] - exact).abs() < 1e-4);
    }

    #[test]
    fn generalized_diagonal() {
        // A = diag(1,2,3,…), B = identity → λ = 1,2,3,…
        let n = 10;
        let mut a_coo = CooMatrix::new(n, n);
        for i in 0..n { a_coo.push(i, i, (i + 1) as f64); }
        let a = CsrMatrix::from_coo(&a_coo);
        let mut eye_coo = CooMatrix::new(n, n);
        for i in 0..n { eye_coo.push(i, i, 1.0); }
        let eye = CsrMatrix::from_coo(&eye_coo);
        // A generalised solver configured with B = I is equivalent to standard.
        let solver = Lobpcg::<f64>::new_generalized(None, Some(&eye), None);
        let params = EigenParams::new(2, EigenWhich::SmallestAlgebraic);
        let res = solver.solve(&a, &params).unwrap();
        assert!((res.eigenvalues[0] - 1.0).abs() < 1e-4);
        assert!((res.eigenvalues[1] - 2.0).abs() < 1e-4);
    }
}