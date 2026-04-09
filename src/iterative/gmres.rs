//! GMRES(m) — Generalized Minimal Residual with restart.
//!
//! Builds an orthonormal Krylov basis V_m via Arnoldi/modified Gram-Schmidt,
//! maintains an upper-Hessenberg system H_m, applies Givens rotations to keep
//! it in upper-triangular form, then back-substitutes to get the update.
//!
//! After `restart` inner iterations (or convergence), the outer loop restarts
//! with the current x as the new initial guess.
//!
//! **Algorithm** (Saad §6.5 / "GMRES revisited"):
//! ```text
//! outer loop (restart):
//!   r = b − A x,  β = ‖r‖,  v₁ = r/β
//!   for j = 1…m:
//!       w = A M⁻¹ vⱼ   [right preconditioned]   (or M⁻¹A vⱼ for left)
//!       modified Gram-Schmidt: h_{ij} = vᵢ·w,  w -= h_{ij} vᵢ
//!       h_{j+1,j} = ‖w‖,  v_{j+1} = w / h_{j+1,j}
//!       apply Givens rotations to column j of H
//!       if |g_{j+1}| / β < tol → converge
//!   y = H^{-1} g  (back-substitution)
//!   x += V_m y
//! ```
//!
//! **Analogs**
//!   PETSc: `KSPSetType(ksp, KSPGMRES)` with `KSPGMRESSetRestart`
//!   HYPRE: `HYPRE_GMRESCreate` with `HYPRE_GMRESSetMaxIter`

use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
};
use crate::sparse::CsrMatrix;

/// GMRES(m) solver with restart.
pub struct Gmres<T> {
    /// Number of Krylov vectors before restart.
    pub restart: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Gmres<T> {
    pub fn new(restart: usize) -> Self {
        Gmres { restart: restart.max(1), _phantom: std::marker::PhantomData }
    }
}

impl<T: Scalar> Default for Gmres<T> {
    fn default() -> Self { Self::new(30) }
}

impl<T: Scalar> KrylovSolver for Gmres<T> {
    type Vector = DenseVec<T>;
    type Operator = CsrMatrix<T>;

    fn solve(
        &self,
        op: &CsrMatrix<T>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        params: &SolverParams,
    ) -> Result<SolverResult, SolverError> {
        let n = b.len();
        if op.nrows() != n || op.ncols() != x.len() {
            return Err(SolverError::DimensionMismatch {
                op_rows: op.nrows(),
                op_cols: op.ncols(),
                rhs_len: n,
            });
        }

        let norm_b = b.norm2();
        let norm_b_f = if norm_b == T::zero() { T::one() } else { norm_b };
        let tol = T::from_f64(params.rtol);
        let atol = T::from_f64(params.atol);
        let m = self.restart;
        let mut residual_history: Vec<f64> = Vec::new();

        let mut total_iters = 0usize;

        // Outer restart loop
        loop {
            // r = b - A x
            let mut r = b.zero_like();
            {
                let mut ax = b.zero_like();
                op.apply(x, &mut ax);
                let rs = r.as_mut_slice();
                let bs = b.as_slice();
                let axs = ax.as_slice();
                for i in 0..n { rs[i] = bs[i] - axs[i]; }
            }
            let beta = r.norm2();

            let rel = beta / norm_b_f;
            if rel < tol || beta < atol {
                if params.verbose != VerboseLevel::Silent {
                    println!("  GMRES converged (restart check) iter {}  ‖r‖/‖b‖={:.3e}", total_iters, to_f64(rel));
                }
                return Ok(SolverResult {
                    converged: true,
                    iterations: total_iters,
                    final_residual: to_f64(rel),
                    residual_history: residual_history.clone(),
                    history: None,
                });
            }
            if total_iters >= params.max_iter {
                break;
            }

            // V[0] = r / β
            let mut v: Vec<DenseVec<T>> = Vec::with_capacity(m + 1);
            {
                let mut v0 = r.clone();
                v0.scale(T::one() / beta);
                v.push(v0);
            }

            // Upper Hessenberg H (stored column-major: h[j] = column j, length j+2)
            let mut h: Vec<Vec<T>> = Vec::with_capacity(m);
            // Givens rotations: (cs, sn)
            let mut cs: Vec<T> = Vec::with_capacity(m);
            let mut sn: Vec<T> = Vec::with_capacity(m);
            // g = [β, 0, 0, …]
            let mut g: Vec<T> = vec![T::zero(); m + 1];
            g[0] = beta;

            let mut inner_converged = false;
            let mut j_final = 0;

            'inner: for j in 0..m {
                if total_iters >= params.max_iter { break; }

                // w = A M⁻¹ vⱼ
                let mut z = DenseVec::zeros(n);
                apply_precond_or_copy(precond, &v[j], &mut z);
                let mut w = b.zero_like();
                op.apply(&z, &mut w);

                // Modified Gram-Schmidt orthogonalisation
                let mut hcol: Vec<T> = Vec::with_capacity(j + 2);
                for vi in &v[..=j] {
                    let hij = dot_slice(vi.as_slice(), w.as_slice());
                    hcol.push(hij);
                    // w -= hij * vi
                    let ws = w.as_mut_slice();
                    let vis = vi.as_slice();
                    for i in 0..n { ws[i] -= hij * vis[i]; }
                }
                let h_next = w.norm2();
                hcol.push(h_next);
                h.push(hcol);

                // Normalise w → v_{j+1}
                if h_next > T::machine_epsilon() {
                    w.scale(T::one() / h_next);
                }
                v.push(w);

                // Apply previous Givens rotations to column j
                let hj = h.last_mut().unwrap();
                for i in 0..j {
                    let tmp = cs[i] * hj[i] + sn[i] * hj[i + 1];
                    hj[i + 1] = -sn[i] * hj[i] + cs[i] * hj[i + 1];
                    hj[i] = tmp;
                }

                // Compute new Givens rotation for entry (j, j+1)
                let (c, s) = givens(hj[j], hj[j + 1]);
                cs.push(c);
                sn.push(s);

                hj[j]     = c * hj[j] + s * hj[j + 1];
                hj[j + 1] = T::zero();

                // Update g
                g[j + 1] = -s * g[j];
                g[j]     =  c * g[j];

                total_iters += 1;
                j_final = j + 1;

                let res = g[j + 1].abs() / norm_b_f;
                let res_f = to_f64(res);
                residual_history.push(res_f);
                if params.verbose == VerboseLevel::Iterations {
                    println!("    GMRES iter {:4}  ‖r‖/‖b‖ = {res_f:.6e}", total_iters);
                }

                if res < tol || g[j + 1].abs() < atol {
                    inner_converged = true;
                    break 'inner;
                }
            }

            // Back-substitution: y = H^{-1} g  (j_final × j_final upper triangular)
            let jf = j_final;
            let mut y = vec![T::zero(); jf];
            for i in (0..jf).rev() {
                let mut s = g[i];
                for k in (i + 1)..jf {
                    s -= h[k][i] * y[k];
                }
                y[i] = s / h[i][i];
            }

            // x += V[:jf] · y   (using right-preconditioned vectors)
            // Actually we need to apply M⁻¹ to each basis vector before accumulating.
            // Since we stored v (preconditioned by M⁻¹ before multiply), we need
            // to undo: x += sum y_j * M⁻¹ v_j
            for j in 0..jf {
                let mut mz = DenseVec::zeros(n);
                apply_precond_or_copy(precond, &v[j], &mut mz);
                x.axpy(y[j], &mz);
            }

            if inner_converged {
                // Compute true residual for final result
                let mut r_final = b.zero_like();
                {
                    let mut ax = b.zero_like();
                    op.apply(x, &mut ax);
                    let rs = r_final.as_mut_slice();
                    let bs = b.as_slice();
                    let axs = ax.as_slice();
                    for i in 0..n { rs[i] = bs[i] - axs[i]; }
                }
                let rel = r_final.norm2() / norm_b_f;
                if params.verbose != VerboseLevel::Silent {
                    println!("  GMRES converged iter {}  ‖r‖/‖b‖={:.3e}", total_iters, to_f64(rel));
                }
                return Ok(SolverResult {
                    converged: true,
                    iterations: total_iters,
                    final_residual: to_f64(rel),
                    residual_history: residual_history.clone(),
                    history: None,
                });
            }

            if total_iters >= params.max_iter { break; }
            // continue restart
        }

        // Compute final residual
        let mut r_final = b.zero_like();
        {
            let mut ax = b.zero_like();
            op.apply(x, &mut ax);
            let rs = r_final.as_mut_slice();
            let bs = b.as_slice();
            let axs = ax.as_slice();
            for i in 0..n { rs[i] = bs[i] - axs[i]; }
        }
        let final_residual = to_f64(r_final.norm2() / norm_b_f);
        Err(SolverError::ConvergenceFailed { max_iter: params.max_iter, residual: final_residual })
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Compute Givens rotation (c, s) such that [c s; -s c] [a; b] = [sqrt(a²+b²); 0].
/// c = a/r,  s = b/r,  r = sqrt(a²+b²).
fn givens<T: Scalar>(a: T, b: T) -> (T, T) {
    if b.abs() < T::machine_epsilon() {
        return (T::one(), T::zero());
    }
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
        None => dst.copy_from(src),
    }
}

fn to_f64<T: Scalar>(v: T) -> f64 {
    num_traits::ToPrimitive::to_f64(&v).unwrap_or(f64::INFINITY)
}
