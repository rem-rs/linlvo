//! FGMRES — Flexible Generalized Minimal Residual.
//!
//! Identical to GMRES(m) except that the preconditioner is allowed to **vary**
//! at each inner iteration (e.g. an inner Krylov solve, a V-cycle, or any
//! nonlinear operator).  This is achieved by storing the preconditioned
//! vectors z_j separately from the Krylov basis v_j:
//!
//! ```text
//! outer loop (restart):
//!   r = b − A x,   β = ‖r‖,   v[0] = r / β
//!   for j = 0..m:
//!       z[j] = M_j⁻¹ v[j]           ← preconditioned vector (stored)
//!       w    = A z[j]
//!       modified Gram-Schmidt on v[0..j]
//!       apply/build Givens for column j
//!   y = H⁻¹ g   (back-substitution)
//!   x += ∑_j y[j] z[j]              ← use z, not v
//! ```
//!
//! With a **fixed** preconditioner FGMRES produces the same iterates as
//! standard right-preconditioned GMRES.
//!
//! **Reference**: Saad, "A flexible inner-outer preconditioned GMRES algorithm",
//!   SIAM J. Sci. Comput. 14 (1993).
//!
//! **Analogs**
//!   PETSc: `KSPSetType(ksp, KSPFGMRES)` with `KSPFGMRESSetRestart`
//!   HYPRE: `HYPRE_FlexGMRESCreate`

use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
};

/// Flexible GMRES(m) with restart.
pub struct Fgmres<T> {
    /// Number of inner Arnoldi steps before restart.
    pub restart: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Fgmres<T> {
    pub fn new(restart: usize) -> Self {
        Fgmres { restart: restart.max(1), _phantom: std::marker::PhantomData }
    }
}

impl<T: Scalar> Default for Fgmres<T> {
    fn default() -> Self { Self::new(30) }
}

impl<T: Scalar> KrylovSolver for Fgmres<T> {
    type Vector   = DenseVec<T>;

    fn solve(
        &self,
        op:      &dyn LinearOperator<Vector = DenseVec<T>>,
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
        let mut residual_history: Vec<f64> = Vec::new();

        let mut total_iters = 0usize;

        loop {
            // r = b − A x
            let mut r = b.zero_like();
            {
                let mut ax = b.zero_like();
                op.apply(x, &mut ax);
                crate::simd::dense_ops::simd_sub(b.as_slice(), ax.as_slice(), r.as_mut_slice());
            }
            let beta = r.norm2();
            let rel = beta / norm_b_f;
            if rel < tol || beta < atol {
                if params.verbose != VerboseLevel::Silent {
                    println!("  FGMRES converged iter {}  ‖r‖/‖b‖={:.3e}", total_iters, to_f64(rel));
                }
                return Ok(SolverResult { converged: true, iterations: total_iters, final_residual: to_f64(rel), residual_history: std::mem::take(&mut residual_history), history: None });
            }
            if total_iters >= params.max_iter { break; }

            // v[0] = r / β
            let mut v: Vec<DenseVec<T>> = Vec::with_capacity(m + 1);
            let mut z: Vec<DenseVec<T>> = Vec::with_capacity(m);
            {
                let mut v0 = r.clone();
                v0.scale(T::one() / beta);
                v.push(v0);
            }

            let mut h: Vec<Vec<T>> = Vec::with_capacity(m);
            let mut cs: Vec<T>     = Vec::with_capacity(m);
            let mut sn: Vec<T>     = Vec::with_capacity(m);
            let mut g: Vec<T>      = vec![T::zero(); m + 1];
            g[0] = beta;

            let mut inner_converged = false;
            let mut j_final         = 0;

            'inner: for j in 0..m {
                if total_iters >= params.max_iter { break; }

                // z[j] = M⁻¹ v[j]  (flexible: precond may vary per step)
                let mut zj = DenseVec::zeros(n);
                apply_precond_or_copy(precond, &v[j], &mut zj);
                z.push(zj);

                // w = A z[j]
                let mut w = b.zero_like();
                op.apply(z.last().unwrap(), &mut w);

                // Modified Gram-Schmidt
                let mut hcol: Vec<T> = Vec::with_capacity(j + 2);
                for vi in &v[..=j] {
                    let hij = dot_slice(vi.as_slice(), w.as_slice());
                    hcol.push(hij);
                    crate::simd::dense_ops::simd_axpy(-hij, vi.as_slice(), w.as_mut_slice());
                }
                let h_next = w.norm2();
                hcol.push(h_next);
                h.push(hcol);

                if h_next > T::machine_epsilon() {
                    w.scale(T::one() / h_next);
                }
                v.push(w);

                // Apply previous Givens rotations
                let hj = h.last_mut().unwrap();
                for i in 0..j {
                    let tmp         =  cs[i] * hj[i] + sn[i] * hj[i + 1];
                    hj[i + 1]       = -sn[i] * hj[i] + cs[i] * hj[i + 1];
                    hj[i]           = tmp;
                }

                // New Givens rotation
                let (c, s) = givens(hj[j], hj[j + 1]);
                cs.push(c); sn.push(s);
                hj[j]     = c * hj[j] + s * hj[j + 1];
                hj[j + 1] = T::zero();

                g[j + 1] = -s * g[j];
                g[j]     =  c * g[j];

                total_iters += 1;
                j_final      = j + 1;

                let res   = g[j + 1].abs() / norm_b_f;
                let res_f = to_f64(res);
                residual_history.push(res_f);
                if params.verbose == VerboseLevel::Iterations {
                    println!("    FGMRES iter {:4}  ‖r‖/‖b‖ = {res_f:.6e}", total_iters);
                }

                if res < tol || g[j + 1].abs() < atol {
                    inner_converged = true;
                    break 'inner;
                }
            }

            // Back-substitution: y = H⁻¹ g
            let jf = j_final;
            let mut y = vec![T::zero(); jf];
            for i in (0..jf).rev() {
                let mut s = g[i];
                for k in (i + 1)..jf { s -= h[k][i] * y[k]; }
                y[i] = s / h[i][i];
            }

            // x += ∑ y[j] z[j]   (use stored preconditioned vectors)
            for j in 0..jf {
                x.axpy(y[j], &z[j]);
            }

            if inner_converged {
                let mut r_final = b.zero_like();
                {
                    let mut ax = b.zero_like();
                    op.apply(x, &mut ax);
                    crate::simd::dense_ops::simd_sub(b.as_slice(), ax.as_slice(), r_final.as_mut_slice());
                }
                let rel = r_final.norm2() / norm_b_f;
                if params.verbose != VerboseLevel::Silent {
                    println!("  FGMRES converged iter {}  ‖r‖/‖b‖={:.3e}", total_iters, to_f64(rel));
                }
                return Ok(SolverResult { converged: true, iterations: total_iters, final_residual: to_f64(rel), residual_history: std::mem::take(&mut residual_history), history: None });
            }

            if total_iters >= params.max_iter { break; }
        }

        let mut r_final = b.zero_like();
        {
            let mut ax = b.zero_like();
            op.apply(x, &mut ax);
            crate::simd::dense_ops::simd_sub(b.as_slice(), ax.as_slice(), r_final.as_mut_slice());
        }
        let final_residual = to_f64(r_final.norm2() / norm_b_f);
        Err(SolverError::ConvergenceFailed { max_iter: params.max_iter, residual: final_residual })
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

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
    crate::simd::dense_ops::simd_dot(a, b)
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
