//! LGMRES — Loose GMRES with augmented Krylov subspace (error recycling).
//!
//! At each restart, LGMRES augments the standard Krylov basis K_m with up to
//! `k` "approximate error" vectors saved from previous restart cycles.  This
//! allows information to persist across restarts, substantially reducing the
//! number of outer iterations compared to GMRES(m) with the same `m`.
//!
//! **Algorithm** (Baker, Jessup & Manteuffel 2005):
//!
//! ```text
//! aug = []   (augmentation vectors from previous restarts, capacity k)
//! outer loop:
//!   r = b − A x,   β = ‖r‖,   v[0] = r / β
//!
//!   // Phase A: incorporate augmentation vectors
//!   for each a_i in aug:
//!       z_i = M⁻¹ a_i
//!       w   = A z_i
//!       Gram-Schmidt w against all current v[0..p]
//!       add w (if not linearly dependent) to the Arnoldi basis
//!
//!   // Phase B: standard Arnoldi for remaining inner steps
//!   for j = p..m+k:
//!       z[j] = M⁻¹ v[j]
//!       w    = A z[j]
//!       Gram-Schmidt, Givens update
//!
//!   y = H⁻¹ g  (back-substitution)
//!   dx = ∑ y[j] z[j]
//!   x += dx
//!
//!   // Save unit-norm update as new augmentation vector
//!   aug.push(dx / ‖dx‖)
//!   if |aug| > k: drop oldest
//! ```
//!
//! **Reference**: Baker, Jessup, Manteuffel, SIAM J. Sci. Comput. 26, 2005.
//!
//! **Analogs**
//!   PETSc: `KSPSetType(ksp, KSPLGMRES)` with `KSPLGMRESSetAugDim`
//!   HYPRE: (no direct equivalent; use FGMRES)

#![allow(clippy::needless_range_loop)]
use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
};
use crate::sparse::CsrMatrix;

/// LGMRES with inner restart dimension `m` and augmentation depth `k`.
pub struct Lgmres<T> {
    /// Inner Krylov dimension (Arnoldi steps per restart).
    pub restart: usize,
    /// Number of augmentation vectors retained from previous restarts.
    pub aug_dim: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Lgmres<T> {
    pub fn new(restart: usize, aug_dim: usize) -> Self {
        Lgmres { restart: restart.max(1), aug_dim, _phantom: std::marker::PhantomData }
    }
}

impl<T: Scalar> Default for Lgmres<T> {
    fn default() -> Self { Self::new(20, 3) }
}

impl<T: Scalar> KrylovSolver for Lgmres<T> {
    type Vector   = DenseVec<T>;
    type Operator = CsrMatrix<T>;

    fn solve(
        &self,
        op:      &CsrMatrix<T>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b:       &DenseVec<T>,
        x:       &mut DenseVec<T>,
        params:  &SolverParams,
    ) -> Result<SolverResult, SolverError> {
        let n = b.len();
        if op.nrows() != n || op.ncols() != x.len() {
            return Err(SolverError::DimensionMismatch {
                op_rows: op.nrows(), op_cols: op.ncols(), rhs_len: n,
            });
        }

        let norm_b   = b.norm2();
        let norm_b_f = if norm_b == T::zero() { T::one() } else { norm_b };
        let tol      = T::from_f64(params.rtol);
        let atol     = T::from_f64(params.atol);
        let m        = self.restart;
        let k        = self.aug_dim;
        let mut residual_history: Vec<f64> = Vec::new();

        let mut total_iters = 0usize;
        // Augmentation vectors (unit-norm update directions from previous restarts).
        let mut aug_vecs: Vec<DenseVec<T>> = Vec::with_capacity(k);

        loop {
            // r = b − A x
            let mut r = b.zero_like();
            {
                let mut ax = b.zero_like();
                op.apply(x, &mut ax);
                let rs  = r.as_mut_slice();
                let bs  = b.as_slice();
                let axs = ax.as_slice();
                for i in 0..n { rs[i] = bs[i] - axs[i]; }
            }
            let beta = r.norm2();
            let rel  = beta / norm_b_f;
            if rel < tol || beta < atol {
                if params.verbose != VerboseLevel::Silent {
                    println!("  LGMRES converged iter {}  ‖r‖/‖b‖={:.3e}", total_iters, to_f64(rel));
                }
                return Ok(SolverResult { converged: true, iterations: total_iters, final_residual: to_f64(rel), residual_history: std::mem::take(&mut residual_history), history: None });
            }
            if total_iters >= params.max_iter { break; }

            // v[0] = r / β
            let mut v: Vec<DenseVec<T>> = Vec::with_capacity(m + k + 1);
            let mut z: Vec<DenseVec<T>> = Vec::with_capacity(m + k);
            {
                let mut v0 = r.clone();
                v0.scale(T::one() / beta);
                v.push(v0);
            }

            let max_inner = m + aug_vecs.len();
            let mut h:  Vec<Vec<T>> = Vec::with_capacity(max_inner);
            let mut cs: Vec<T>      = Vec::with_capacity(max_inner);
            let mut sn: Vec<T>      = Vec::with_capacity(max_inner);
            let mut g:  Vec<T>      = vec![T::zero(); max_inner + 1];
            g[0] = beta;

            let mut inner_converged = false;
            let mut j_final         = 0;

            // ── Phase A: augmentation vectors ─────────────────────────────
            let aug_count = aug_vecs.len();
            for ai in 0..aug_count {
                if total_iters >= params.max_iter { break; }
                let j = ai; // index into the growing Arnoldi basis

                // z[j] = M⁻¹ aug[ai]  (augmentation vector already in M-image space)
                let mut zj = DenseVec::zeros(n);
                apply_precond_or_copy(precond, &aug_vecs[ai], &mut zj);
                z.push(zj);

                // w = A z[j]
                let mut w = b.zero_like();
                op.apply(z.last().unwrap(), &mut w);

                // Gram-Schmidt against all current v
                let hcol = gram_schmidt(&v, &mut w, n);
                let h_next = w.norm2();
                let mut hcol_full = hcol;
                hcol_full.push(h_next);
                h.push(hcol_full);

                if h_next > T::machine_epsilon() {
                    w.scale(T::one() / h_next);
                }
                v.push(w);

                apply_givens_and_update(&mut h, &mut cs, &mut sn, &mut g, j);

                total_iters += 1;
                j_final      = j + 1;

                let res   = g[j + 1].abs() / norm_b_f;
                let res_f = to_f64(res);
                residual_history.push(res_f);
                if params.verbose == VerboseLevel::Iterations {
                    println!("    LGMRES (aug) iter {:4}  ‖r‖/‖b‖ = {res_f:.6e}", total_iters);
                }
                if res < tol || g[j + 1].abs() < atol {
                    inner_converged = true;
                    break;
                }
            }

            // ── Phase B: standard Arnoldi steps ──────────────────────────
            if !inner_converged {
                'inner: for step in 0..m {
                    if total_iters >= params.max_iter { break; }
                    let j = aug_count + step;

                    // z[j] = M⁻¹ v[j]
                    let mut zj = DenseVec::zeros(n);
                    apply_precond_or_copy(precond, &v[j], &mut zj);
                    z.push(zj);

                    let mut w = b.zero_like();
                    op.apply(z.last().unwrap(), &mut w);

                    let hcol = gram_schmidt(&v, &mut w, n);
                    let h_next = w.norm2();
                    let mut hcol_full = hcol;
                    hcol_full.push(h_next);
                    h.push(hcol_full);

                    if h_next > T::machine_epsilon() {
                        w.scale(T::one() / h_next);
                    }
                    v.push(w);

                    apply_givens_and_update(&mut h, &mut cs, &mut sn, &mut g, j);

                    total_iters += 1;
                    j_final      = j + 1;

                    let res   = g[j + 1].abs() / norm_b_f;
                    let res_f = to_f64(res);
                    residual_history.push(res_f);
                    if params.verbose == VerboseLevel::Iterations {
                        println!("    LGMRES iter {:4}  ‖r‖/‖b‖ = {res_f:.6e}", total_iters);
                    }
                    if res < tol || g[j + 1].abs() < atol {
                        inner_converged = true;
                        break 'inner;
                    }
                }
            }

            // ── Back-substitution ─────────────────────────────────────────
            let jf = j_final;
            let mut y = vec![T::zero(); jf];
            for i in (0..jf).rev() {
                let mut s = g[i];
                for kk in (i + 1)..jf { s -= h[kk][i] * y[kk]; }
                y[i] = s / h[i][i];
            }

            // dx = ∑ y[j] z[j],  x += dx
            let mut dx = DenseVec::zeros(n);
            for j in 0..jf {
                dx.axpy(y[j], &z[j]);
            }
            {
                let xs  = x.as_mut_slice();
                let dxs = dx.as_slice();
                for i in 0..n { xs[i] += dxs[i]; }
            }

            // ── Save augmentation vector (unit-norm dx) ───────────────────
            let dx_norm = dx.norm2();
            if dx_norm > T::machine_epsilon() {
                dx.scale(T::one() / dx_norm);
                if aug_vecs.len() >= k && k > 0 {
                    aug_vecs.remove(0); // drop oldest
                }
                if k > 0 { aug_vecs.push(dx); }
            }

            if inner_converged {
                let mut r_final = b.zero_like();
                {
                    let mut ax = b.zero_like();
                    op.apply(x, &mut ax);
                    let rs  = r_final.as_mut_slice();
                    let bs  = b.as_slice();
                    let axs = ax.as_slice();
                    for i in 0..n { rs[i] = bs[i] - axs[i]; }
                }
                let rel = r_final.norm2() / norm_b_f;
                if params.verbose != VerboseLevel::Silent {
                    println!("  LGMRES converged iter {}  ‖r‖/‖b‖={:.3e}", total_iters, to_f64(rel));
                }
                return Ok(SolverResult { converged: true, iterations: total_iters, final_residual: to_f64(rel), residual_history: std::mem::take(&mut residual_history), history: None });
            }

            if total_iters >= params.max_iter { break; }
        }

        let mut r_final = b.zero_like();
        {
            let mut ax = b.zero_like();
            op.apply(x, &mut ax);
            let rs  = r_final.as_mut_slice();
            let bs  = b.as_slice();
            let axs = ax.as_slice();
            for i in 0..n { rs[i] = bs[i] - axs[i]; }
        }
        let final_residual = to_f64(r_final.norm2() / norm_b_f);
        Err(SolverError::ConvergenceFailed { max_iter: params.max_iter, residual: final_residual })
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Modified Gram-Schmidt: orthogonalise `w` against all `v[0..=v.len()-1]`.
/// Returns the Hessenberg column (dot products with each v_i).
fn gram_schmidt<T: Scalar>(v: &[DenseVec<T>], w: &mut DenseVec<T>, n: usize) -> Vec<T> {
    let mut hcol = Vec::with_capacity(v.len());
    for vi in v {
        let hij = dot_slice(vi.as_slice(), w.as_slice());
        hcol.push(hij);
        let ws  = w.as_mut_slice();
        let vis = vi.as_slice();
        for i in 0..n { ws[i] -= hij * vis[i]; }
    }
    hcol
}

/// Apply previously accumulated Givens rotations to column j of H, then
/// compute and apply a new rotation; update the g vector.
#[allow(clippy::ptr_arg)]
fn apply_givens_and_update<T: Scalar>(
    h:  &mut Vec<Vec<T>>,
    cs: &mut Vec<T>,
    sn: &mut Vec<T>,
    g:  &mut [T],
    j:  usize,
) {
    let hj = h.last_mut().unwrap();
    for i in 0..j {
        let tmp         =  cs[i] * hj[i] + sn[i] * hj[i + 1];
        hj[i + 1]       = -sn[i] * hj[i] + cs[i] * hj[i + 1];
        hj[i]           = tmp;
    }
    let (c, s) = givens(hj[j], hj[j + 1]);
    cs.push(c); sn.push(s);
    hj[j]     = c * hj[j] + s * hj[j + 1];
    hj[j + 1] = T::zero();

    g[j + 1] = -s * g[j];
    g[j]     =  c * g[j];
}

fn givens<T: Scalar>(a: T, b: T) -> (T, T) {
    if b.abs() < T::machine_epsilon() { return (T::one(), T::zero()); }
    if b.abs() > a.abs() {
        let tau = a / b;
        let s = T::one() / (T::one() + tau * tau).sqrt();
        (s * tau, s)
    } else {
        let tau = b / a;
        let c = T::one() / (T::one() + tau * tau).sqrt();
        (c, c * tau)
    }
}

fn dot_slice<T: Scalar>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b.iter()).fold(T::zero(), |s, (&ai, &bi)| s + ai * bi)
}

fn apply_precond_or_copy<T: Scalar>(
    precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
    src: &DenseVec<T>,
    dst: &mut DenseVec<T>,
) {
    match precond {
        Some(m) => m.apply_precond(src, dst),
        None    => dst.copy_from(src),
    }
}

fn to_f64<T: Scalar>(v: T) -> f64 {
    num_traits::ToPrimitive::to_f64(&v).unwrap_or(f64::INFINITY)
}
