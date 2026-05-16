#![allow(clippy::needless_range_loop)]
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

/// Reusable scratch buffers for repeated GMRES solves with fixed dimension/restart.
pub struct GmresWorkspace<T: Scalar> {
    restart: usize,
    r: DenseVec<T>,
    v: Vec<DenseVec<T>>,
    z_scratch: DenseVec<T>,
    w_scratch: DenseVec<T>,
    mz_scratch: DenseVec<T>,
    ax_scratch: DenseVec<T>,
    h: Vec<Vec<T>>,
    cs: Vec<T>,
    sn: Vec<T>,
    g: Vec<T>,
}

impl<T: Scalar> GmresWorkspace<T> {
    pub fn new(n: usize, restart: usize) -> Self {
        let restart = restart.max(1);
        Self {
            restart,
            r: DenseVec::zeros(n),
            v: (0..=restart).map(|_| DenseVec::zeros(n)).collect(),
            z_scratch: DenseVec::zeros(n),
            w_scratch: DenseVec::zeros(n),
            mz_scratch: DenseVec::zeros(n),
            ax_scratch: DenseVec::zeros(n),
            h: (0..restart).map(|_| Vec::with_capacity(restart + 1)).collect(),
            cs: Vec::with_capacity(restart),
            sn: Vec::with_capacity(restart),
            g: vec![T::zero(); restart + 1],
        }
    }

    fn ensure_shape(&mut self, n: usize, restart: usize) {
        let restart = restart.max(1);
        if self.restart != restart || self.r.len() != n {
            *self = Self::new(n, restart);
        }
    }
}

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

    /// Solve `A x = b` using caller-owned scratch buffers to amortize allocations.
    pub fn solve_with_workspace(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        params: &SolverParams,
        workspace: &mut GmresWorkspace<T>,
    ) -> Result<SolverResult, SolverError> {
        self.solve_with_workspace_impl(op, precond, b, x, params, workspace, None)
    }

    /// Run exactly `iterations` GMRES steps using caller-owned scratch buffers.
    ///
    /// This is intended for deterministic cost measurement rather than normal
    /// solve-to-tolerance usage. The method suppresses tolerance-based early
    /// exit, but still returns breakdown errors if Arnoldi/backsolve fails.
    pub fn solve_fixed_iters_with_workspace(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        iterations: usize,
        workspace: &mut GmresWorkspace<T>,
    ) -> Result<SolverResult, SolverError> {
        self.solve_with_workspace_impl(op, precond, b, x, &SolverParams::default(), workspace, Some(iterations))
    }

    /// Run exactly `iterations` GMRES steps with an internal workspace.
    pub fn solve_fixed_iters(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        iterations: usize,
    ) -> Result<SolverResult, SolverError> {
        let mut workspace = GmresWorkspace::new(b.len(), self.restart);
        self.solve_fixed_iters_with_workspace(op, precond, b, x, iterations, &mut workspace)
    }

    fn solve_with_workspace_impl(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        params: &SolverParams,
        workspace: &mut GmresWorkspace<T>,
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

        let norm_b = b.norm2();
        let norm_b_f = if norm_b == T::zero() { T::one() } else { norm_b };
        let tol = T::from_f64(params.rtol);
        let atol = T::from_f64(params.atol);
        let m = self.restart;
        workspace.ensure_shape(n, m);
        let mut residual_history: Vec<f64> = Vec::new();
        let mut total_iters = 0usize;
        let target_iterations = fixed_iterations.unwrap_or(params.max_iter);
        let allow_early_exit = fixed_iterations.is_none();

        loop {
            op.apply(x, &mut workspace.ax_scratch);
            crate::simd::dense_ops::simd_sub(b.as_slice(), workspace.ax_scratch.as_slice(), workspace.r.as_mut_slice());
            let beta = workspace.r.norm2();
            if !beta.is_finite() {
                return Err(SolverError::NumericalBreakdown {
                    detail: format!(
                        "GMRES: non-finite residual norm at restart begin (iter {}); check matrix/RHS values",
                        total_iters,
                    ),
                });
            }

            let rel = beta / norm_b_f;
            if allow_early_exit && (rel < tol || beta < atol) {
                if params.verbose != VerboseLevel::Silent {
                    println!("  GMRES converged (restart check) iter {}  ‖r‖/‖b‖={:.3e}", total_iters, to_f64(rel));
                }
                return Ok(SolverResult {
                    converged: true,
                    iterations: total_iters,
                    final_residual: to_f64(rel),
                    residual_history: std::mem::take(&mut residual_history),
                    history: None,
                });
            }
            if total_iters >= target_iterations {
                break;
            }

            {
                let v0s = workspace.v[0].as_mut_slice();
                let rs = workspace.r.as_slice();
                let inv_beta = T::one() / beta;
                for i in 0..n { v0s[i] = rs[i] * inv_beta; }
            }

            for col in workspace.h.iter_mut() { col.clear(); }
            workspace.cs.clear();
            workspace.sn.clear();
            for gi in workspace.g.iter_mut() { *gi = T::zero(); }
            workspace.g[0] = beta;

            let mut inner_converged = false;
            let mut j_final = 0;
            let local_restart = if allow_early_exit {
                m
            } else {
                m.min(target_iterations - total_iters)
            };

            'inner: for j in 0..local_restart {
                if total_iters >= target_iterations { break; }

                apply_precond_or_copy(precond, &workspace.v[j], &mut workspace.z_scratch);
                op.apply(&workspace.z_scratch, &mut workspace.w_scratch);

                let hj = &mut workspace.h[j];
                hj.clear();
                for vi in &workspace.v[..=j] {
                    let hij = dot_slice(vi.as_slice(), workspace.w_scratch.as_slice());
                    hj.push(hij);
                    crate::simd::dense_ops::simd_axpy(-hij, vi.as_slice(), workspace.w_scratch.as_mut_slice());
                }
                let h_next = workspace.w_scratch.norm2();
                if !h_next.is_finite() {
                    return Err(SolverError::NumericalBreakdown {
                        detail: format!(
                            "GMRES: non-finite Arnoldi norm h[{},{}]; try scaling matrix/RHS or different preconditioner",
                            j + 1,
                            j,
                        ),
                    });
                }
                hj.push(h_next);

                {
                    let vj1 = workspace.v[j + 1].as_mut_slice();
                    let ws = workspace.w_scratch.as_slice();
                    if h_next > T::machine_epsilon() {
                        let inv = T::one() / h_next;
                        for i in 0..n { vj1[i] = ws[i] * inv; }
                    } else {
                        vj1.copy_from_slice(ws);
                    }
                }

                for i in 0..j {
                    let tmp = workspace.cs[i] * hj[i] + workspace.sn[i] * hj[i + 1];
                    hj[i + 1] = -workspace.sn[i] * hj[i] + workspace.cs[i] * hj[i + 1];
                    hj[i] = tmp;
                }

                let (c, s) = givens(hj[j], hj[j + 1]);
                workspace.cs.push(c);
                workspace.sn.push(s);
                hj[j] = c * hj[j] + s * hj[j + 1];
                hj[j + 1] = T::zero();

                workspace.g[j + 1] = -s * workspace.g[j];
                workspace.g[j] = c * workspace.g[j];

                total_iters += 1;
                j_final = j + 1;

                let res = workspace.g[j + 1].abs() / norm_b_f;
                if !res.is_finite() {
                    return Err(SolverError::NumericalBreakdown {
                        detail: format!(
                            "GMRES: non-finite relative residual at iter {}; Krylov basis likely corrupted",
                            total_iters,
                        ),
                    });
                }
                let res_f = to_f64(res);
                residual_history.push(res_f);
                if params.verbose == VerboseLevel::Iterations {
                    println!("    GMRES iter {:4}  ‖r‖/‖b‖ = {res_f:.6e}", total_iters);
                }

                if allow_early_exit && (res < tol || workspace.g[j + 1].abs() < atol) {
                    inner_converged = true;
                    break 'inner;
                }
            }

            let jf = j_final;
            let mut y = vec![T::zero(); jf];
            for i in (0..jf).rev() {
                if workspace.h[i][i].abs() <= T::machine_epsilon() {
                    return Err(SolverError::NumericalBreakdown {
                        detail: format!(
                            "GMRES: near-zero Hessenberg diagonal at backsolve i={}; try larger restart or stronger preconditioner",
                            i,
                        ),
                    });
                }
                let mut s = workspace.g[i];
                for k in (i + 1)..jf {
                    s -= workspace.h[k][i] * y[k];
                }
                y[i] = s / workspace.h[i][i];
            }

            for (j, &yj) in y.iter().enumerate() {
                apply_precond_or_copy(precond, &workspace.v[j], &mut workspace.mz_scratch);
                x.axpy(yj, &workspace.mz_scratch);
            }

            if inner_converged {
                op.apply(x, &mut workspace.ax_scratch);
                let rfnorm = {
                    let bs = b.as_slice();
                    let axs = workspace.ax_scratch.as_slice();
                    (0..n).map(|i| { let d = bs[i] - axs[i]; d * d }).fold(T::zero(), |a, v| a + v).sqrt()
                };
                let rel = rfnorm / norm_b_f;
                if params.verbose != VerboseLevel::Silent {
                    println!("  GMRES converged iter {}  ‖r‖/‖b‖={:.3e}", total_iters, to_f64(rel));
                }
                return Ok(SolverResult {
                    converged: true,
                    iterations: total_iters,
                    final_residual: to_f64(rel),
                    residual_history: std::mem::take(&mut residual_history),
                    history: None,
                });
            }

            if total_iters >= target_iterations { break; }
        }

        op.apply(x, &mut workspace.ax_scratch);
        let rfnorm = {
            let bs = b.as_slice();
            let axs = workspace.ax_scratch.as_slice();
            (0..n).map(|i| { let d = bs[i] - axs[i]; d * d }).fold(T::zero(), |a, v| a + v).sqrt()
        };
        let final_residual = to_f64(rfnorm / norm_b_f);
        if fixed_iterations.is_some() {
            Ok(SolverResult {
                converged: false,
                iterations: total_iters,
                final_residual,
                residual_history,
                history: None,
            })
        } else {
            Err(SolverError::ConvergenceFailed { max_iter: params.max_iter, residual: final_residual })
        }
    }
}

impl<T: Scalar> Default for Gmres<T> {
    fn default() -> Self { Self::new(30) }
}

impl<T: Scalar> KrylovSolver for Gmres<T> {
    type Vector = DenseVec<T>;

    fn solve(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        params: &SolverParams,
    ) -> Result<SolverResult, SolverError> {
        let mut workspace = GmresWorkspace::new(b.len(), self.restart);
        self.solve_with_workspace(op, precond, b, x, params, &mut workspace)
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
