//! MINRES — Minimum Residual method for symmetric (possibly indefinite) systems.
//!
//! Uses the Lanczos three-term recurrence and the Paige–Saunders QR update to
//! minimise the residual norm over successive Krylov subspaces without storing
//! the entire basis.
//!
//! Suitable for **symmetric** (not necessarily positive definite) systems.
//! For SPD systems CG is preferred (cheaper per iteration); MINRES is needed
//! when A may have negative eigenvalues.
//!
//! **Reference**: Paige & Saunders, SIAM J. Numer. Anal. 12, 1975.
//! Implementation follows the Stanford SOL group Matlab reference code.
//!
//! **Analogs**
//!   PETSc: `KSPSetType(ksp, KSPMINRES)`
//!   HYPRE: not directly exposed; use GMRES for indefinite systems.

#![allow(clippy::needless_range_loop)]
use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
};

/// MINRES solver for symmetric (possibly indefinite) systems.
pub struct Minres<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Minres<T> {
    pub fn new() -> Self { Minres { _phantom: std::marker::PhantomData } }
}

impl<T: Scalar> Default for Minres<T> {
    fn default() -> Self { Self::new() }
}

impl<T: Scalar> KrylovSolver for Minres<T> {
    type Vector = DenseVec<T>;

    fn solve(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
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
        let mut residual_history: Vec<f64> = Vec::new();

        // r = b - A x₀
        let mut r = b.zero_like();
        {
            let mut ax = b.zero_like();
            op.apply(x, &mut ax);
            let rs = r.as_mut_slice();
            for i in 0..n { rs[i] = b.as_slice()[i] - ax.as_slice()[i]; }
        }

        // y = M⁻¹ r
        let mut y = DenseVec::zeros(n);
        apply_precond_or_copy(precond, &r, &mut y);

        // beta1 = sqrt(r · y)  (M-inner-product norm of r)
        let beta1_sq = dot_slice(r.as_slice(), y.as_slice());
        if beta1_sq < T::zero() {
            return Err(SolverError::NumericalBreakdown {
                detail: "MINRES: preconditioner is not positive definite (r·Mr < 0)".into(),
            });
        }
        let beta1 = beta1_sq.sqrt();

        if beta1 < T::machine_epsilon() {
            return Ok(SolverResult {
                converged: true, iterations: 0, final_residual: 0.0, residual_history: vec![], history: None,
            });
        }

        // ── Initialise Lanczos (following Matlab SOL naming) ─────────────────
        //   r2  = current "unnormalized" Lanczos vector  (= beta * v_k)
        //   r1  = previous one                           (= beta_prev * v_{k-1})
        //   y   = M⁻¹ * r2
        let mut oldb  = T::zero();
        let mut beta  = beta1;
        let mut r1    = r.clone();    // r1 = b - Ax₀
        let mut r2    = r.clone();    // r2 starts as r1

        // Paige–Saunders QR state
        let mut dbar   = T::zero();
        let mut epsln  = T::zero();
        let mut phibar = beta1;
        let mut cs     = -T::one();
        let mut sn     = T::zero();

        // Direction vectors (three-term recurrence for solution update)
        let mut w  = DenseVec::zeros(n);
        let mut w2 = DenseVec::zeros(n);

        for k in 0..params.max_iter {
            // ── Lanczos step ──────────────────────────────────────────────────
            let s = T::one() / beta;

            // v = r2 / beta  (current normalized Lanczos vector)
            let mut v = r2.clone();
            v.scale(s);

            // Compute A * v
            let mut av = b.zero_like();
            op.apply(&v, &mut av);

            // Subtract previous Lanczos component: av -= (beta / oldb) * r1
            // (Only from iter 2 onwards; when k=0, oldb=0 and r1=r2 so skip)
            if k > 0 {
                let scale = beta / oldb;
                let avs = av.as_mut_slice();
                let r1s = r1.as_slice();
                for i in 0..n { avs[i] -= scale * r1s[i]; }
            }

            // alpha = v' * av
            let alpha = dot_slice(v.as_slice(), av.as_slice());

            // Subtract current component: av -= (alpha / beta) * r2
            {
                let avs = av.as_mut_slice();
                let r2s = r2.as_slice();
                for i in 0..n { avs[i] -= (alpha / beta) * r2s[i]; }
            }

            // Advance Lanczos storage
            r1.copy_from(&r2);

            // r2 = M⁻¹ * av
            apply_precond_or_copy(precond, &av, &mut r2);

            oldb = beta;
            let beta_sq = dot_slice(av.as_slice(), r2.as_slice());
            beta = if beta_sq > T::zero() { beta_sq.sqrt() } else { T::zero() };

            // ── Paige–Saunders QR update ──────────────────────────────────────
            //
            // Apply previous Givens rotation Qk-1 to get:
            //   delta_k  = cs_{k-1} * dbar + sn_{k-1} * alpha
            //   gbar_k   = sn_{k-1} * dbar - cs_{k-1} * alpha
            //   epsln_{k+1} = sn_{k-1} * beta
            //   dbar_{k+1} = -cs_{k-1} * beta
            let oldeps = epsln;
            let delta  = cs * dbar + sn * alpha;
            let gbar   = sn * dbar - cs * alpha;
            epsln = sn * beta;
            dbar  = -(cs) * beta;

            // Compute new Givens rotation to eliminate beta from [gbar; beta]
            let gamma = (gbar * gbar + beta * beta).sqrt();
            let gamma = if gamma < T::machine_epsilon() * T::from_f64(1e6) {
                T::machine_epsilon() * T::from_f64(1e6)
            } else {
                gamma
            };
            cs = gbar / gamma;
            sn = beta  / gamma;

            let phi    = cs * phibar;
            phibar     = sn * phibar;   // tracks ‖r_k‖ estimate

            // ── Update direction vector and solution ──────────────────────────
            //   w_k = (v_k - oldeps * w_{k-2} - delta * w_{k-1}) / gamma
            let denom = T::one() / gamma;
            let w1 = w2.clone();  // w_{k-2}
            w2.copy_from(&w);     // w_{k-1}
            {
                let ws  = w.as_mut_slice();
                let w1s = w1.as_slice();
                let w2s = w2.as_slice();
                let vs  = v.as_slice();
                for i in 0..n {
                    ws[i] = (vs[i] - oldeps * w1s[i] - delta * w2s[i]) * denom;
                }
            }
            x.axpy(phi, &w);

            // ── Convergence check ─────────────────────────────────────────────
            let res   = phibar.abs() / norm_b_f;
            let res_f = to_f64(res);
            residual_history.push(res_f);
            if params.verbose == VerboseLevel::Iterations {
                println!("    MINRES iter {:4}  ‖r‖/‖b‖ = {res_f:.6e}", k + 1);
            }
            if res < tol || phibar.abs() < atol {
                if params.verbose != VerboseLevel::Silent {
                    println!("  MINRES converged iter {}  ‖r‖/‖b‖={res_f:.3e}", k + 1);
                }
                return Ok(SolverResult {
                    converged: true, iterations: k + 1, final_residual: res_f, residual_history: std::mem::take(&mut residual_history), history: None,
                });
            }
        }

        let final_residual = to_f64(phibar.abs() / norm_b_f);
        Err(SolverError::ConvergenceFailed { max_iter: params.max_iter, residual: final_residual })
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

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
        None => dst.copy_from(src),
    }
}

fn to_f64<T: Scalar>(v: T) -> f64 {
    num_traits::ToPrimitive::to_f64(&v).unwrap_or(f64::INFINITY)
}
