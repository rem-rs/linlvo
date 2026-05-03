//! PIPECG — Pipelined Conjugate Gradient.
//!
//! A CG variant that reduces the number of global reductions per iteration
//! from 2 (standard PCG) to 1, improving weak scalability in distributed
//! settings where global all-reduce is the dominant communication bottleneck.
//!
//! ## Algorithm (Ghysels & Vanroose 2014)
//!
//! The pipelined variant reshuffles the PCG recurrence to expose an extra
//! SpMV that can overlap with the global dot-product:
//!
//! ```text
//! r₀ = b − A x₀,  u₀ = M⁻¹ r₀,  w₀ = A u₀
//! γ₀ = <r₀, u₀>,  δ₀ = <w₀, u₀>
//! z₀ = M⁻¹ w₀,    q₀ = A z₀
//! p₀ = u₀, s₀ = w₀, ŝ₀ = z₀
//!
//! for k = 0, 1, …:
//!   α_k  = γ_k / δ_k
//!   x_{k+1} = x_k + α_k p_k
//!   r_{k+1} = r_k − α_k s_k
//!   u_{k+1} = u_k − α_k ŝ_k              (= M⁻¹ r_{k+1} without extra precond call)
//!   w_{k+1} = w_k − α_k q_k
//!   γ_{k+1} = <r_{k+1}, u_{k+1}>
//!   δ_{k+1} = <w_{k+1}, u_{k+1}>
//!   β_k = γ_{k+1} / γ_k
//!   z_{k+1}  = M⁻¹ w_{k+1}               (one precond call per iter)
//!   q_{k+1}  = A z_{k+1}                  (one SpMV per iter — can overlap reduce)
//!   p_{k+1}  = u_{k+1} + β_k p_k
//!   s_{k+1}  = w_{k+1} + β_k s_k
//!   ŝ_{k+1}  = z_{k+1} + β_k ŝ_k
//! ```
//!
//! In a distributed setting `γ_{k+1}` and `δ_{k+1}` can be computed while
//! `q_{k+1}` = A z_{k+1} is being applied, hiding the all-reduce latency.
//! This implementation is single-node but maintains the same recurrence so it
//! serves as a correct drop-in and can be swapped with an MPI-aware version.
//!
//! ## Preconditioner requirement
//!
//! PIPECG calls the preconditioner **twice** per setup step (iterations 0)
//! and **once** per subsequent iteration.  The preconditioner must be
//! **fixed** (non-varying).  For varying preconditioners use FGMRES.
//!
//! ## Reference
//!
//! Ghysels, P., & Vanroose, W. (2014). *Hiding global communication latency
//! in the GMRES algorithm on massively parallel machines.*
//! SIAM J. Sci. Comput., 36(1), C48–C71.
//!
//! ## Analogs
//!
//! PETSc: `KSPSetType(ksp, KSPPIPECG)` (added in PETSc 3.5)

#![allow(clippy::needless_range_loop)]

use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
};

// ─── Public struct ────────────────────────────────────────────────────────────

/// Pipelined Conjugate Gradient solver.
///
/// Drop-in replacement for [`ConjugateGradient`](super::ConjugateGradient) on
/// SPD systems.  Requires **one fewer global dot-product per iteration** —
/// the communication-hiding benefit is only realised in a distributed runtime,
/// but correctness is guaranteed on all targets.
pub struct PipeCg<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> PipeCg<T> {
    pub fn new() -> Self {
        PipeCg { _phantom: std::marker::PhantomData }
    }
}

impl<T: Scalar> Default for PipeCg<T> {
    fn default() -> Self { Self::new() }
}

// ─── KrylovSolver impl ────────────────────────────────────────────────────────

impl<T: Scalar> KrylovSolver for PipeCg<T> {
    type Vector = DenseVec<T>;

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
                op_rows: op.nrows(),
                op_cols: op.ncols(),
                rhs_len: n,
            });
        }

        // ── Scratch buffers ───────────────────────────────────────────────────
        let mut r  = DenseVec::zeros(n); // residual
        let mut u  = DenseVec::zeros(n); // M⁻¹ r
        let mut w  = DenseVec::zeros(n); // A u
        let mut z  = DenseVec::zeros(n); // M⁻¹ w
        let mut q  = DenseVec::zeros(n); // A z
        let mut p  = DenseVec::zeros(n); // search dir for x
        let mut s  = DenseVec::zeros(n); // search dir for r
        let mut sh = DenseVec::zeros(n); // search dir for u (ŝ)

        // ── Initialisation ────────────────────────────────────────────────────
        // r₀ = b − A x₀
        let mut ax = DenseVec::zeros(n);
        op.apply(x, &mut ax);
        for i in 0..n {
            r.as_mut_slice()[i] = b.as_slice()[i] - ax.as_slice()[i];
        }

        // u₀ = M⁻¹ r₀
        apply_pc_or_copy(precond, &r, &mut u);

        let norm_b = b.norm2();
        let norm_b_f = if norm_b == T::zero() { T::one() } else { norm_b };
        let mut residual_history: Vec<f64> = Vec::new();
        let verbose_history = params.verbose == VerboseLevel::Iterations;
        let mut history: Option<Vec<f64>> = if verbose_history { Some(Vec::new()) } else { None };

        // ── Pre-loop convergence check ────────────────────────────────────────
        {
            let r0_norm = r.norm2();
            if r0_norm <= T::from_f64(params.atol) || r0_norm / norm_b_f <= T::from_f64(params.rtol) {
                if params.verbose != VerboseLevel::Silent {
                    println!("  PIPECG: initial residual already below tolerance");
                }
                return Ok(SolverResult {
                    converged: true,
                    iterations: 0,
                    final_residual: num_traits::ToPrimitive::to_f64(&(r0_norm / norm_b_f)).unwrap_or(0.0),
                    residual_history,
                    history,
                });
            }
        }

        // w₀ = A u₀
        op.apply(&u, &mut w);
        // z₀ = M⁻¹ w₀
        apply_pc_or_copy(precond, &w, &mut z);
        // q₀ = A z₀
        op.apply(&z, &mut q);

        let mut gamma = dot(&r, &u);
        let mut delta = dot(&w, &u);

        if !gamma.is_finite() {
            return Err(SolverError::NumericalBreakdown {
                detail: "PIPECG: non-finite <r,u> at init; check matrix/RHS".into(),
            });
        }

        p.copy_from(&u);
        s.copy_from(&w);
        sh.copy_from(&z);

        let check_every = if params.check_interval > 0 { params.check_interval } else { usize::MAX };

        // ── Main loop ─────────────────────────────────────────────────────────
        for k in 0..params.max_iter {
            // Convergence check on pipelined residual.
            let r_norm = r.norm2();
            let res_now = r_norm / norm_b_f;
            let res_f = num_traits::ToPrimitive::to_f64(&res_now).unwrap_or(f64::INFINITY);
            residual_history.push(res_f);
            if let Some(h) = history.as_mut() { h.push(res_f); }
            if params.verbose == VerboseLevel::Iterations {
                println!("  PIPECG iter {:4}  ‖r‖/‖b‖ = {:.3e}", k + 1, res_f);
            }

            if res_now < T::from_f64(params.rtol) || r_norm < T::from_f64(params.atol) {
                if params.verbose != VerboseLevel::Silent {
                    println!("  PIPECG converged at iter {}  ‖r‖/‖b‖ = {:.3e}", k + 1, res_f);
                }
                return Ok(SolverResult {
                    converged: true,
                    iterations: k + 1,
                    final_residual: res_f,
                    residual_history,
                    history,
                });
            }

            // ── δ health check ────────────────────────────────────────────────
            // A bad delta (non-finite, ≤ 0, or negligible relative to gamma)
            // indicates floating-point drift or a non-SPD matrix.  Recompute
            // the true residual and, if not yet converged, perform a full
            // restart so the iteration can recover.
            if !delta.is_finite() || delta <= T::zero()
                || delta.abs() < T::machine_epsilon() * T::from_f64(1e3) * gamma.abs()
            {
                op.apply(x, &mut ax);
                crate::simd::dense_ops::simd_sub(b.as_slice(), ax.as_slice(), r.as_mut_slice());
                let true_r_norm = r.norm2();
                let true_res = true_r_norm / norm_b_f;
                if true_r_norm <= T::from_f64(params.atol) || true_res < T::from_f64(params.rtol) {
                    let final_f = num_traits::ToPrimitive::to_f64(&true_res).unwrap_or(f64::INFINITY);
                    if params.verbose != VerboseLevel::Silent {
                        println!("  PIPECG converged (true-res check) iter {}  ‖r‖/‖b‖ = {:.3e}", k + 1, final_f);
                    }
                    return Ok(SolverResult {
                        converged: true,
                        iterations: k + 1,
                        final_residual: final_f,
                        residual_history,
                        history,
                    });
                }
                // Full restart: rebuild pipelined quantities from true residual.
                apply_pc_or_copy(precond, &r, &mut u);
                op.apply(&u, &mut w);
                apply_pc_or_copy(precond, &w, &mut z);
                op.apply(&z, &mut q);
                gamma = dot(&r, &u);
                delta = dot(&w, &u);
                p.copy_from(&u);
                s.copy_from(&w);
                sh.copy_from(&z);
                // If delta is STILL non-positive after restart the matrix is not SPD.
                if !delta.is_finite() || delta <= T::zero() {
                    return Err(SolverError::NumericalBreakdown {
                        detail: format!(
                            "PIPECG: δ ≤ 0 after restart at iter {} (δ={:.3e}); matrix may be indefinite",
                            k + 1,
                            num_traits::ToPrimitive::to_f64(&delta).unwrap_or(f64::NAN),
                        ),
                    });
                }
                continue; // skip normal update; resume from top with restarted state
            }

            let alpha = gamma / delta;

            // Update iterates.
            let xs = x.as_mut_slice();
            let rs = r.as_mut_slice();
            let us = u.as_mut_slice();
            let ws = w.as_mut_slice();
            let ps = p.as_slice();
            let ss = s.as_slice();
            let shs = sh.as_slice();
            let qs = q.as_slice();
            for i in 0..n {
                xs[i] += alpha * ps[i];
                rs[i] -= alpha * ss[i];
                us[i] -= alpha * shs[i];
                ws[i] -= alpha * qs[i];
            }

            let gamma_new = dot_slices(rs, us);
            let delta_new = dot_slices(ws, us);

            // z_{k+1} = M⁻¹ w_{k+1}
            apply_pc_or_copy(precond, &w, &mut z);
            // q_{k+1} = A z_{k+1}  (could overlap with the global reduce above)
            op.apply(&z, &mut q);

            let beta = gamma_new / gamma;
            gamma = gamma_new;
            delta = delta_new;

            // Periodic true-residual recompute to suppress floating-point drift.
            // Also resets search directions to prevent accumulated-error runaway.
            if (k + 1) % check_every == 0 {
                op.apply(x, &mut ax);
                crate::simd::dense_ops::simd_sub(b.as_slice(), ax.as_slice(), r.as_mut_slice());
                // Check convergence on the true residual.
                let tr_norm = r.norm2();
                if tr_norm <= T::from_f64(params.atol) || tr_norm / norm_b_f < T::from_f64(params.rtol) {
                    let fr = num_traits::ToPrimitive::to_f64(&(tr_norm / norm_b_f)).unwrap_or(0.0);
                    if params.verbose != VerboseLevel::Silent {
                        println!("  PIPECG converged (check_interval) iter {}  ‖r‖/‖b‖ = {:.3e}", k + 1, fr);
                    }
                    return Ok(SolverResult {
                        converged: true, iterations: k + 1, final_residual: fr,
                        residual_history, history,
                    });
                }
                // Recompute u, w, z, q from the corrected r.
                apply_pc_or_copy(precond, &r, &mut u);
                op.apply(&u, &mut w);
                apply_pc_or_copy(precond, &w, &mut z);
                op.apply(&z, &mut q);
                gamma = dot(&r, &u);
                delta = dot(&w, &u);
                // Reset search directions (prevents directional drift).
                p.copy_from(&u);
                s.copy_from(&w);
                sh.copy_from(&z);
            }

            // Update search directions.
            let ps = p.as_mut_slice();
            let ss = s.as_mut_slice();
            let shs = sh.as_mut_slice();
            let us_new = u.as_slice();
            let ws_new = w.as_slice();
            let zs = z.as_slice();
            for i in 0..n {
                ps[i] = us_new[i] + beta * ps[i];
                ss[i] = ws_new[i] + beta * ss[i];
                shs[i] = zs[i]    + beta * shs[i];
            }
        }

        let final_res = {
            let r_norm = r.norm2();
            let res = r_norm / norm_b_f;
            num_traits::ToPrimitive::to_f64(&res).unwrap_or(f64::INFINITY)
        };
        Err(SolverError::ConvergenceFailed {
            max_iter: params.max_iter,
            residual: final_res,
        })
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

fn apply_pc_or_copy<T: Scalar>(
    precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
    x: &DenseVec<T>,
    y: &mut DenseVec<T>,
) {
    match precond {
        Some(p) => p.apply_precond(x, y),
        None    => y.copy_from(x),
    }
}

#[inline]
fn dot<T: Scalar>(a: &DenseVec<T>, b: &DenseVec<T>) -> T {
    dot_slices(a.as_slice(), b.as_slice())
}

#[inline]
fn dot_slices<T: Scalar>(a: &[T], b: &[T]) -> T {
    crate::simd::dense_ops::simd_dot(a, b)
}
