//! TFQMR — Transpose-Free Quasi-Minimal Residual solver.
//!
//! A short-recurrence Krylov method for non-symmetric systems.  Uses 2 matvecs
//! per outer step and avoids the omega division of BiCGSTAB.
//!
//! ## Algorithm
//!
//! Freund 1993, *A Transpose-Free Quasi-Minimal Residual Algorithm*,
//! SIAM J. Sci. Comput. 14(2), pp. 470–482.  Right-preconditioned variant.
//!
//! Key variables (matching Barrett et al. *Templates for the Solution of
//! Linear Systems*, 1994):
//! ```text
//! y, u = A·M⁻¹·y:  CGS direction vectors
//! v:                CGS accumulation (v_1 = u_1, then recurrence)
//! w:                quasi-residual
//! d:                search direction for x update
//! tau, theta, eta:  quasi-minimal residual scalars
//! ```
//!
//! Per outer step k (2 matvecs):
//! 1. sigma = (r̃, v),  alpha = rho / sigma
//! 2. y_half = y - alpha * v
//! 3. ay_half = A * M⁻¹ * y_half            ← matvec 1
//! 4. Half-step a: w -= alpha*u,  d = y + c*d,  update x
//! 5. Half-step b: w -= alpha*ay_half, d = y_half + c*d, update x
//! 6. y = w + beta*y_half;  u = A*M⁻¹*y    ← matvec 2
//! 7. v = u + beta*(ay_half + beta*v)
//!
//! ## Analogs
//!
//! PETSc: `KSPSetType(ksp, KSPTFQMR)`

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

pub struct Tfqmr<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Tfqmr<T> {
    pub fn new() -> Self { Tfqmr { _phantom: std::marker::PhantomData } }
}

impl<T: Scalar> Default for Tfqmr<T> {
    fn default() -> Self { Self::new() }
}

impl<T: Scalar> KrylovSolver for Tfqmr<T> {
    type Vector   = DenseVec<T>;
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
                op_rows: op.nrows(), op_cols: op.ncols(), rhs_len: n,
            });
        }

        let tol      = T::from_f64(params.rtol);
        let atol     = T::from_f64(params.atol);
        let max_iter = params.max_iter;
        let norm_b   = b.norm2();
        let norm_b_f = if norm_b == T::zero() { T::one() } else { norm_b };
        let mut residual_history: Vec<f64> = Vec::new();

        // Pre-allocated scratch for preconditioner.
        let mut pc_in  = DenseVec::zeros(n);
        let mut pc_out = DenseVec::zeros(n);

        // M^{-1} * src into dst slice; returns dst as vec for caller convenience.
        let inv = |src: &[T], pi: &mut DenseVec<T>, po: &mut DenseVec<T>| -> Vec<T> {
            pi.as_mut_slice().copy_from_slice(src);
            if let Some(pc) = precond { pc.apply_precond(pi, po); }
            else                      { po.copy_from(pi); }
            po.as_slice().to_vec()
        };

        // A * src → vec.
        let av = |src: &[T]| -> Vec<T> {
            let sd = DenseVec::from_vec(src.to_vec());
            let mut out = DenseVec::zeros(n);
            op.apply(&sd, &mut out);
            out.as_slice().to_vec()
        };

        // A * M^{-1} * src → vec.
        let _am = |src: &[T], pi: &mut DenseVec<T>, po: &mut DenseVec<T>| -> Vec<T> {
            av(&inv(src, pi, po))
        };

        // ── Initial residual r = b − A x ──────────────────────────────────────
        let r0 = {
            let mut ax = DenseVec::zeros(n);
            op.apply(x, &mut ax);
            let mut r = vec![T::zero(); n];
            let bs = b.as_slice(); let axs = ax.as_slice();
            for i in 0..n { r[i] = bs[i] - axs[i]; }
            r
        };

        let norm_r0 = norm2(&r0);
        if norm_r0 <= atol || norm_r0 <= tol * norm_b_f {
            return Ok(SolverResult {
                converged: true, iterations: 0,
                final_residual: to_f64(norm_r0 / norm_b_f),
                residual_history, history: None,
            });
        }

        // ── Initialisation ─────────────────────────────────────────────────────
        let r_shadow = r0.clone();
        let mut y = r0.clone();
        // z = M^{-1} * y,  u = A * z  (u_1 = A * M^{-1} * r0)
        let z0 = inv(&y, &mut pc_in, &mut pc_out);
        let mut u = av(&z0);
        let mut v = u.clone();          // v_1 = u_1
        let mut w = r0.clone();
        let mut d = vec![T::zero(); n];

        let mut tau   = norm_r0;
        let mut theta = T::zero();
        let mut eta   = T::zero();
        let mut rho   = dot(&r_shadow, &r0);

        let mut iters    = 0usize;
        let mut converged = false;

        'outer: for k in 0..max_iter {
            let sigma = dot(&r_shadow, &v);
            if sigma.abs() < T::machine_epsilon() * T::from_f64(1e6) { break 'outer; }
            let alpha = rho / sigma;

            // y_half = y - alpha * v
            let y_half: Vec<T> = (0..n).map(|l| y[l] - alpha * v[l]).collect();
            // z = M^{-1} * y,  z_half = M^{-1} * y_half  (for x update direction)
            let z      = inv(&y,      &mut pc_in, &mut pc_out);
            let z_half = inv(&y_half, &mut pc_in, &mut pc_out);
            // ay_half = A * z_half  (matvec 1 of this step)
            let ay_half = av(&z_half);

            // ── Half-step a (odd m = 2k+1): uses u = A*z (from prev step/init) ─
            for l in 0..n { w[l] = w[l] - alpha * u[l]; }
            let coeff_a = if alpha.abs() < T::machine_epsilon() { T::zero() }
                          else { theta * theta * eta / alpha };
            // d uses z (= M^{-1}*y) for the right-preconditioned x update
            for l in 0..n { d[l] = z[l] + coeff_a * d[l]; }
            theta = norm2(&w) / tau;
            let ca = T::one() / (T::one() + theta * theta).sqrt();
            tau   = tau * theta * ca;
            eta   = ca * ca * alpha;
            { let xs = x.as_mut_slice(); for l in 0..n { xs[l] = xs[l] + eta * d[l]; } }

            iters += 1;
            let est_a = tau * T::from_f64(((2 * k + 1) as f64).sqrt());
            let rel_a = est_a / norm_b_f;
            residual_history.push(to_f64(rel_a));
            if params.verbose == VerboseLevel::Iterations {
                println!("    TFQMR iter {:4}a  τ√m/‖b‖={:.6e}", k + 1, to_f64(rel_a));
            }
            if rel_a <= tol || est_a <= atol { converged = true; break 'outer; }
            if iters >= max_iter { break 'outer; }

            // ── Half-step b (even m = 2k+2): uses ay_half ─────────────────────
            for l in 0..n { w[l] = w[l] - alpha * ay_half[l]; }
            let coeff_b = if alpha.abs() < T::machine_epsilon() { T::zero() }
                          else { theta * theta * eta / alpha };
            // d uses z_half (= M^{-1}*y_half)
            for l in 0..n { d[l] = z_half[l] + coeff_b * d[l]; }
            theta = norm2(&w) / tau;
            let cb = T::one() / (T::one() + theta * theta).sqrt();
            tau   = tau * theta * cb;
            eta   = cb * cb * alpha;
            { let xs = x.as_mut_slice(); for l in 0..n { xs[l] = xs[l] + eta * d[l]; } }

            iters += 1;
            let est_b = tau * T::from_f64(((2 * k + 2) as f64).sqrt());
            let rel_b = est_b / norm_b_f;
            residual_history.push(to_f64(rel_b));
            if params.verbose == VerboseLevel::Iterations {
                println!("    TFQMR iter {:4}b  τ√m/‖b‖={:.6e}", k + 1, to_f64(rel_b));
            }
            if rel_b <= tol || est_b <= atol { converged = true; break 'outer; }
            if iters >= max_iter { break 'outer; }

            // ── Update for next outer step ─────────────────────────────────────
            let rho_new = dot(&r_shadow, &w);
            if rho.abs() < T::machine_epsilon() * T::from_f64(1e6) { break 'outer; }
            let beta = rho_new / rho;
            rho = rho_new;

            // y = w + beta * y_half
            for l in 0..n { y[l] = w[l] + beta * y_half[l]; }
            // z_new = M^{-1} * y;  u = A * z_new  (matvec 2 of this step)
            let z_new = inv(&y, &mut pc_in, &mut pc_out);
            u = av(&z_new);
            // v = u + beta * (ay_half + beta * v)
            for l in 0..n { v[l] = u[l] + beta * (ay_half[l] + beta * v[l]); }
        }

        // ── Exact final residual ───────────────────────────────────────────────
        let mut ax = DenseVec::zeros(n);
        op.apply(x, &mut ax);
        let final_rn: T = {
            let mut s = T::zero();
            let bs = b.as_slice(); let axs = ax.as_slice();
            for i in 0..n { let di = bs[i] - axs[i]; s = s + di * di; }
            s.sqrt()
        };
        let final_res = to_f64(final_rn / norm_b_f);
        if !converged && (final_rn <= atol || final_rn <= tol * norm_b_f) {
            converged = true;
        }

        if params.verbose != VerboseLevel::Silent {
            if converged {
                println!("TFQMR: converged in {iters} iters, rel_res={final_res:.3e}");
            } else {
                println!("TFQMR: NOT converged after {iters} iters, rel_res={final_res:.3e}");
            }
        }

        Ok(SolverResult { converged, iterations: iters, final_residual: final_res, residual_history, history: None })
    }
}

fn dot<T: Scalar>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b.iter()).fold(T::zero(), |s, (&ai, &bi)| s + ai * bi)
}

fn norm2<T: Scalar>(a: &[T]) -> T {
    a.iter().fold(T::zero(), |s, &v| s + v * v).sqrt()
}

fn to_f64<T: Scalar>(v: T) -> f64 {
    num_traits::ToPrimitive::to_f64(&v).unwrap_or(f64::INFINITY)
}
