//! Preconditioned Conjugate Gradient (PCG) solver.
//!
//! Implements the standard PCG algorithm for symmetric positive definite (SPD)
//! systems  A x = b.  When no preconditioner is provided the method reduces
//! to classical CG.
//!
//! **Algorithm** (from Saad §6.7 / Trefethen & Bau, Lecture 38):
//! ```text
//! r₀ = b − A x₀
//! z₀ = M⁻¹ r₀
//! p₀ = z₀
//! for k = 0, 1, …:
//!     α_k  = (rᵢ·zᵢ) / (pᵢ·A pᵢ)
//!     x_{k+1} = xᵢ + α_k pᵢ
//!     r_{k+1} = rᵢ − α_k A pᵢ
//!     z_{k+1} = M⁻¹ r_{k+1}
//!     β_k  = (r_{k+1}·z_{k+1}) / (rᵢ·zᵢ)
//!     p_{k+1} = z_{k+1} + β_k pᵢ
//! ```
//!
//! Every `check_interval` iterations the residual is *recomputed* from scratch
//! (`r = b − A x`) to prevent floating-point drift.
//!
//! **Analogs**
//!   PETSc: `KSPSetType(ksp, KSPCG)` with optional `PCJACOBI` or `PCILU`
//!   HYPRE: `HYPRE_PCGCreate` with `HYPRE_PCGSetPrecond`

use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
};

/// Reusable scratch buffers for repeated CG solves with the same vector length.
pub struct CgWorkspace<T: Scalar> {
    r: DenseVec<T>,
    z: DenseVec<T>,
    p: DenseVec<T>,
    ap: DenseVec<T>,
    ax: DenseVec<T>,
}

impl<T: Scalar> CgWorkspace<T> {
    pub fn new(n: usize) -> Self {
        Self {
            r: DenseVec::zeros(n),
            z: DenseVec::zeros(n),
            p: DenseVec::zeros(n),
            ap: DenseVec::zeros(n),
            ax: DenseVec::zeros(n),
        }
    }

    fn ensure_len(&mut self, n: usize) {
        if self.r.len() != n {
            *self = Self::new(n);
        }
    }
}

/// Preconditioned Conjugate Gradient solver.
///
/// Suitable for **symmetric positive definite** systems only.
/// For non-symmetric or indefinite systems use [`super::BiCgStab`] or
/// [`super::Gmres`].
pub struct ConjugateGradient<T> {
    /// How often to recompute the residual from scratch (prevents drift).
    pub check_interval: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> ConjugateGradient<T> {
    /// Create a new CG solver.
    ///
    /// `check_interval`: recompute residual every N iterations (default 50).
    pub fn new(check_interval: usize) -> Self {
        ConjugateGradient { check_interval, _phantom: std::marker::PhantomData }
    }

    /// Solve `A x = b` using caller-owned scratch buffers to amortize allocations.
    pub fn solve_with_workspace(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        params: &SolverParams,
        workspace: &mut CgWorkspace<T>,
    ) -> Result<SolverResult, SolverError> {
        self.solve_with_workspace_impl(op, precond, b, x, params, workspace, None)
    }

    /// Run exactly `iterations` CG steps using caller-owned scratch buffers.
    ///
    /// This is intended for deterministic cost measurement rather than normal
    /// solve-to-tolerance usage. The method suppresses tolerance-based early
    /// exit, but still returns breakdown errors if the iteration cannot proceed.
    pub fn solve_fixed_iters_with_workspace(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        iterations: usize,
        workspace: &mut CgWorkspace<T>,
    ) -> Result<SolverResult, SolverError> {
        self.solve_with_workspace_impl(op, precond, b, x, &SolverParams::default(), workspace, Some(iterations))
    }

    /// Run exactly `iterations` CG steps with an internal workspace.
    pub fn solve_fixed_iters(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        iterations: usize,
    ) -> Result<SolverResult, SolverError> {
        let mut workspace = CgWorkspace::new(b.len());
        self.solve_fixed_iters_with_workspace(op, precond, b, x, iterations, &mut workspace)
    }

    fn solve_with_workspace_impl(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        params: &SolverParams,
        workspace: &mut CgWorkspace<T>,
        fixed_iterations: Option<usize>,
    ) -> Result<SolverResult, SolverError> {
        let n = b.len();
        if op.nrows() != n || op.ncols() != x.len() {
            return Err(SolverError::DimensionMismatch {
                op_rows: op.nrows(),
                op_cols: op.ncols(),
                rhs_len: n,
            });
        }

        workspace.ensure_len(n);

        let norm_b = b.norm2();
        let norm_b_f = if norm_b == T::zero() { T::one() } else { norm_b };
        let mut residual_history: Vec<f64> = Vec::new();
        let verbose_history = params.verbose == VerboseLevel::Iterations;
        let mut history: Option<Vec<f64>> = if verbose_history { Some(Vec::new()) } else { None };
    let target_iterations = fixed_iterations.unwrap_or(params.max_iter);
    let allow_early_exit = fixed_iterations.is_none();

        // r = b − A x₀
        op.apply(x, &mut workspace.ax);
        crate::simd::dense_ops::simd_sub(b.as_slice(), workspace.ax.as_slice(), workspace.r.as_mut_slice());

        // Compute z = M⁻¹ r (preconditioner) or z = r (no preconditioner).
        apply_precond_or_copy(precond, &workspace.r, &mut workspace.z);

        // MFEM `CGSolver::Mult` convergence semantics (linalg/solvers.cpp:869-1053,
        // MFEM 4.10): the threshold is computed ONCE from the initial residual
        // energy `nom0 = (B r₀, r₀)` (or `‖r₀‖²` without a preconditioner),
        //
        //     r0 = max(nom0·rel_tol², abs_tol²)              (solvers.cpp:915)
        //
        // and every step tests the new energy `betanom = (B r, r)` against that
        // fixed threshold (solvers.cpp:977).  The preconditioned-energy *ratio*
        // must therefore reach rel_tol² — not rel_tol: callers wanting the
        // legacy `PCG()` helper's sqrt convention (`SetRelTol(sqrt(RTOL))`)
        // must pass sqrt(RTOL) themselves (see fem-solver's `solve_pcg`, which
        // wraps exactly that helper).  Stopping on the un-squared ratio made
        // rtol 1e-12 stop at true residual ~1e-6 (D976).
        let nom0 = if precond.is_some() {
            dot_slice(workspace.r.as_slice(), workspace.z.as_slice())
        } else {
            dot_slice(workspace.r.as_slice(), workspace.r.as_slice())
        };
        if !nom0.is_finite() {
            return Err(SolverError::NumericalBreakdown {
                detail: "CG: non-finite <r,z> at initialization; check matrix/RHS values and preconditioner output".into(),
            });
        }

        // MFEM solvers.cpp:916-928 — indefinite preconditioner at iteration 0.
        if nom0 < T::zero() {
            if params.verbose != VerboseLevel::Silent {
                println!(
                    "  PCG: The preconditioner is not positive definite. (Br, r) = {:.6e}",
                    to_f64(nom0)
                );
            }
            return Err(SolverError::NumericalBreakdown {
                detail: format!(
                    "CG: preconditioner not positive definite at iteration 0 ((B r₀, r₀) = {:.3e})",
                    to_f64(nom0)
                ),
            });
        }

        let mut rz = nom0;
        let r0 = (nom0 * T::from_f64(params.rtol * params.rtol))
            .max(T::from_f64(params.atol * params.atol));

        // MFEM solvers.cpp:918-928 — iteration-0 test against the fixed
        // threshold (covers b = 0 and an exact initial guess; final_iter = 0).
        // Gated off for fixed-iteration benchmark runs (they always execute
        // exactly the requested number of steps).
        if allow_early_exit && nom0 <= r0 {
            let res_f = to_f64(workspace.r.norm2() / norm_b_f);
            if params.verbose != VerboseLevel::Silent {
                println!("  CG converged at iter 0  ‖r‖/‖b‖ = {res_f:.3e}");
            }
            residual_history.push(res_f);
            return Ok(SolverResult {
                converged: true,
                iterations: 0,
                final_residual: res_f,
                residual_history: std::mem::take(&mut residual_history),
                history: history.take(),
            });
        }

        workspace.p.copy_from(&workspace.z);

        for k in 0..target_iterations {
            op.apply(&workspace.p, &mut workspace.ap);
            let pap = dot_slice(workspace.p.as_slice(), workspace.ap.as_slice());
            if !pap.is_finite() || !rz.is_finite() {
                return Err(SolverError::NumericalBreakdown {
                    detail: format!(
                        "CG: non-finite scalar at iter {} (pAp={:.3e}, rz={:.3e}); try scaling matrix/RHS or a more robust preconditioner",
                        k + 1,
                        to_f64(pap),
                        to_f64(rz),
                    ),
                });
            }

            let r_norm = workspace.r.norm2();
            let res_now = r_norm / norm_b_f;

            if pap.abs() < T::machine_epsilon() * T::from_f64(1e3) * rz.abs() {
                if !allow_early_exit || (res_now > T::from_f64(params.rtol) && r_norm > T::from_f64(params.atol)) {
                    return Err(SolverError::NumericalBreakdown {
                        detail: format!(
                            "CG: pAp≈0 before reaching tolerance at iter {} (rel_res={:.3e}); matrix may be indefinite/singular, try GMRES/MINRES or stronger preconditioner",
                            k + 1,
                            to_f64(res_now),
                        ),
                    });
                }
                let res_f = to_f64(res_now);
                if params.verbose != VerboseLevel::Silent {
                    println!("  CG converged (p·Ap≈0) iter {}  ‖r‖/‖b‖ = {res_f:.3e}", k + 1);
                }
                residual_history.push(res_f);
                return Ok(SolverResult {
                    converged: true,
                    iterations: k + 1,
                    final_residual: res_f,
                    residual_history: std::mem::take(&mut residual_history),
                    history: history.take(),
                });
            }

            let alpha = rz / pap;
            x.axpy(alpha, &workspace.p);

            {
                let rs = workspace.r.as_mut_slice();
                let aps = workspace.ap.as_slice();
                for i in 0..n {
                    rs[i] -= alpha * aps[i];
                }
            }

            if (k + 1) % self.check_interval == 0 {
                op.apply(x, &mut workspace.ax);
                crate::simd::dense_ops::simd_sub(b.as_slice(), workspace.ax.as_slice(), workspace.r.as_mut_slice());
            }

            apply_precond_or_copy(precond, &workspace.r, &mut workspace.z);
            let rz_new = dot_slice(workspace.r.as_slice(), workspace.z.as_slice());
            if !rz_new.is_finite() {
                return Err(SolverError::NumericalBreakdown {
                    detail: format!(
                        "CG: non-finite <r,z> at iter {}; preconditioner or operator produced invalid values",
                        k + 1,
                    ),
                });
            }

            // MFEM solvers.cpp:938-946 — indefinite preconditioner inside the
            // loop, checked before the iteration print.
            if rz_new < T::zero() {
                if params.verbose != VerboseLevel::Silent {
                    println!(
                        "  PCG: The preconditioner is not positive definite. (Br, r) = {:.6e}",
                        to_f64(rz_new)
                    );
                }
                return Err(SolverError::NumericalBreakdown {
                    detail: format!(
                        "CG: preconditioner not positive definite at iter {} ((B r, r) = {:.3e})",
                        k + 1,
                        to_f64(rz_new)
                    ),
                });
            }

            if params.verbose == VerboseLevel::Iterations {
                println!("    CG iter {:4}  (B r, r) = {:.9e}", k + 1, to_f64(rz_new));
            }

            let res = workspace.r.norm2() / norm_b_f;
            let res_f = to_f64(res);
            residual_history.push(res_f);
            if let Some(ref mut h) = history { h.push(res_f); }
            if params.verbose == VerboseLevel::Iterations {
                println!("    CG iter {:4}  ‖r‖/‖b‖ = {res_f:.6e}", k + 1);
            }
            // MFEM solvers.cpp:977 — fixed-threshold test: the new energy
            // `betanom = (B r, r)` against `r0 = max(nom0·rel_tol², abs_tol²)`
            // computed once before the loop.  (D976: the previous test
            // `|rz_new|/|rz₀| < rtol` used the un-squared tolerance and stopped
            // six orders of magnitude in √ratio too early.)
            if allow_early_exit && rz_new <= r0 {
                if params.verbose == VerboseLevel::Iterations {
                    println!("  CG converged at iter {}  ‖r‖/‖b‖ = {res_f:.3e}", k + 1);
                }
                return Ok(SolverResult {
                    converged: true,
                    iterations: k + 1,
                    final_residual: res_f,
                    residual_history: std::mem::take(&mut residual_history),
                    history: history.take(),
                });
            }

            let beta = rz_new / rz;
            {
                let ps = workspace.p.as_mut_slice();
                let zs = workspace.z.as_slice();
                for i in 0..n {
                    ps[i] = zs[i] + beta * ps[i];
                }
            }
            rz = rz_new;
        }

        let final_residual = to_f64(workspace.r.norm2() / norm_b_f);
        if fixed_iterations.is_some() {
            Ok(SolverResult {
                converged: false,
                iterations: target_iterations,
                final_residual,
                residual_history,
                history,
            })
        } else {
            Err(SolverError::ConvergenceFailed { max_iter: params.max_iter, residual: final_residual })
        }
    }
}

impl<T: Scalar> Default for ConjugateGradient<T> {
    fn default() -> Self { Self::new(50) }
}

impl<T: Scalar> KrylovSolver for ConjugateGradient<T> {
    type Vector = DenseVec<T>;

    fn solve(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        params: &SolverParams,
    ) -> Result<SolverResult, SolverError> {
        let mut workspace = CgWorkspace::new(b.len());
        self.solve_with_workspace(op, precond, b, x, params, &mut workspace)
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
        None    => dst.copy_from(src),
    }
}

fn to_f64<T: Scalar>(v: T) -> f64 {
    num_traits::ToPrimitive::to_f64(&v).unwrap_or(f64::INFINITY)
}
