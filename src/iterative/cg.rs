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

        // r = b − A x₀
        op.apply(x, &mut workspace.ax);
        {
            let rs = workspace.r.as_mut_slice();
            let bs = b.as_slice();
            let axs = workspace.ax.as_slice();
            for i in 0..n { rs[i] = bs[i] - axs[i]; }
        }

        apply_precond_or_copy(precond, &workspace.r, &mut workspace.z);
        workspace.p.copy_from(&workspace.z);

        let mut rz = dot_slice(workspace.r.as_slice(), workspace.z.as_slice());
        if !rz.is_finite() {
            return Err(SolverError::NumericalBreakdown {
                detail: "CG: non-finite <r,z> at initialization; check matrix/RHS values and preconditioner output".into(),
            });
        }

        for k in 0..params.max_iter {
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
            if res_now < T::from_f64(params.rtol) || r_norm < T::from_f64(params.atol) {
                let res_f = to_f64(res_now);
                if params.verbose != VerboseLevel::Silent {
                    println!("  CG converged at iter {}  ‖r‖/‖b‖ = {res_f:.3e}", k + 1);
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

            if pap.abs() < T::machine_epsilon() * T::from_f64(1e3) * rz.abs() {
                if res_now > T::from_f64(params.rtol) && r_norm > T::from_f64(params.atol) {
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
                let rs = workspace.r.as_mut_slice();
                let bs = b.as_slice();
                let axs = workspace.ax.as_slice();
                for i in 0..n {
                    rs[i] = bs[i] - axs[i];
                }
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

            let res = workspace.r.norm2() / norm_b_f;
            let res_f = to_f64(res);
            residual_history.push(res_f);
            if let Some(ref mut h) = history { h.push(res_f); }
            if params.verbose == VerboseLevel::Iterations {
                println!("    CG iter {:4}  ‖r‖/‖b‖ = {res_f:.6e}", k + 1);
            }
            if res < T::from_f64(params.rtol) || workspace.r.norm2() < T::from_f64(params.atol) {
                if params.verbose != VerboseLevel::Silent {
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
        Err(SolverError::ConvergenceFailed { max_iter: params.max_iter, residual: final_residual })
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
