//! BiConjugate Gradient Stabilized (BiCGSTAB) solver.
//!
//! A short-recurrence method for **non-symmetric** (and symmetric) linear
//! systems.  More numerically stable than BiCG because it replaces the
//! transpose product with a local minimisation step.
//!
//! **Algorithm** (van der Vorst 1992, Saad §7.4.2):
//! ```text
//! r₀ = b − A x₀,   r̃ = r₀  (shadow residual, fixed)
//! ρ₋₁ = α = ω = 1,  v = p = 0
//! for k = 1, 2, …:
//!     ρ = r̃ · r
//!     β = (ρ/ρ₋₁)(α/ω)
//!     p = r + β(p − ω v)
//!     v = A M⁻¹ p
//!     α = ρ / (r̃ · v)
//!     s = r − α v
//!     if ‖s‖ < tol → x += α M⁻¹ p; done
//!     t = A M⁻¹ s
//!     ω = (t · s) / (t · t)
//!     x += α M⁻¹ p + ω M⁻¹ s
//!     r = s − ω t
//! ```
//!
//! **Analogs**
//!   PETSc: `KSPSetType(ksp, KSPBCGS)`
//!   HYPRE: `HYPRE_BiCGSTABCreate`

use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
};
use crate::sparse::CsrMatrix;

/// BiCGSTAB solver for general (non-symmetric) systems.
pub struct BiCgStab<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> BiCgStab<T> {
    pub fn new() -> Self { BiCgStab { _phantom: std::marker::PhantomData } }
}

impl<T: Scalar> Default for BiCgStab<T> {
    fn default() -> Self { Self::new() }
}

impl<T: Scalar> KrylovSolver for BiCgStab<T> {
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
        let mut residual_history: Vec<f64> = Vec::new();

        let mut history = if params.verbose == VerboseLevel::Iterations {
            Some(Vec::new())
        } else {
            None
        };

        // r = b - A x₀
        let mut r = b.zero_like();
        {
            let mut ax = b.zero_like();
            op.apply(x, &mut ax);
            let rs = r.as_mut_slice();
            let bs = b.as_slice();
            let axs = ax.as_slice();
            for i in 0..n { rs[i] = bs[i] - axs[i]; }
        }

        // r̃ = r₀  (shadow residual — kept fixed)
        let r_shadow = r.clone();

        let mut p = DenseVec::zeros(n);
        let mut v = DenseVec::zeros(n);

        let mut rho_prev = T::one();
        let mut alpha = T::one();
        let mut omega = T::one();

        for k in 0..params.max_iter {
            // Check convergence before breakdown test: a near-zero residual also makes rho≈0.
            let r_norm = r.norm2();
            let res_early = r_norm / norm_b_f;
            if res_early < tol || r_norm < atol {
                let res_f = to_f64(res_early);
                residual_history.push(res_f);
                if let Some(ref mut h) = history { h.push(res_f); }
                if params.verbose != VerboseLevel::Silent {
                    println!("  BiCGSTAB converged iter {}  ‖r‖/‖b‖={res_f:.3e}", k + 1);
                }
                return Ok(SolverResult {
                    converged: true, iterations: k + 1, final_residual: res_f, residual_history: residual_history.clone(), history,
                });
            }

            let rho = dot_slice(r_shadow.as_slice(), r.as_slice());

            if rho.abs() < T::machine_epsilon() {
                return Err(SolverError::NumericalBreakdown {
                    detail: format!("BiCGSTAB: rho ≈ 0 at iter {k}"),
                });
            }

            let beta = (rho / rho_prev) * (alpha / omega);

            // p = r + β (p − ω v)
            {
                let ps = p.as_mut_slice();
                let rs = r.as_slice();
                let vs = v.as_slice();
                for i in 0..n {
                    ps[i] = rs[i] + beta * (ps[i] - omega * vs[i]);
                }
            }

            // v = A M⁻¹ p
            let mut ph = DenseVec::zeros(n);
            apply_precond_or_copy(precond, &p, &mut ph);
            op.apply(&ph, &mut v);

            let rv = dot_slice(r_shadow.as_slice(), v.as_slice());
            if rv.abs() < T::machine_epsilon() {
                return Err(SolverError::NumericalBreakdown {
                    detail: format!("BiCGSTAB: r̃·v ≈ 0 at iter {k}"),
                });
            }
            alpha = rho / rv;

            // s = r - α v
            let mut s = r.clone();
            {
                let ss = s.as_mut_slice();
                let vs = v.as_slice();
                for i in 0..n { ss[i] -= alpha * vs[i]; }
            }

            let s_norm = s.norm2();
            // Early exit: x += α M⁻¹ p
            if s_norm / norm_b_f < tol || s_norm < atol {
                x.axpy(alpha, &ph);
                let res = s_norm / norm_b_f;
                let res_f = to_f64(res);
                residual_history.push(res_f);
                if let Some(ref mut h) = history { h.push(res_f); }
                if params.verbose != VerboseLevel::Silent {
                    println!("  BiCGSTAB converged (early) iter {}  ‖r‖/‖b‖={:.3e}", k + 1, res_f);
                }
                return Ok(SolverResult {
                    converged: true,
                    iterations: k + 1,
                    final_residual: res_f,
                    residual_history: residual_history.clone(),
                    history,
                });
            }

            // t = A M⁻¹ s
            let mut sh = DenseVec::zeros(n);
            apply_precond_or_copy(precond, &s, &mut sh);
            let mut t = b.zero_like();
            op.apply(&sh, &mut t);

            let tt = dot_slice(t.as_slice(), t.as_slice());
            // If t ≈ 0 we're essentially converged (s ≈ 0 too in practice).
            if tt.abs() < T::machine_epsilon() * T::from_f64(1e-6) {
                x.axpy(alpha, &ph);
                let res = s.norm2() / norm_b_f;
                let res_f = to_f64(res);
                if res_f < params.rtol {
                    if params.verbose != VerboseLevel::Silent {
                        println!("  BiCGSTAB converged (t≈0) iter {}  ‖r‖/‖b‖={res_f:.3e}", k + 1);
                    }
                    return Ok(SolverResult {
                        converged: true, iterations: k + 1, final_residual: res_f, residual_history: residual_history.clone(), history,
                    });
                }
                return Err(SolverError::NumericalBreakdown {
                    detail: format!("BiCGSTAB: t·t ≈ 0 at iter {k}"),
                });
            }
            omega = dot_slice(t.as_slice(), s.as_slice()) / tt;

            // x += α M⁻¹ p + ω M⁻¹ s
            x.axpy(alpha, &ph);
            x.axpy(omega, &sh);

            // r = s - ω t
            {
                let rs = r.as_mut_slice();
                let ss = s.as_slice();
                let ts = t.as_slice();
                for i in 0..n { rs[i] = ss[i] - omega * ts[i]; }
            }

            rho_prev = rho;

            let res = r.norm2() / norm_b_f;
            let res_f = to_f64(res);
            residual_history.push(res_f);
            if let Some(ref mut h) = history { h.push(res_f); }
            if params.verbose == VerboseLevel::Iterations {
                println!("    BiCGSTAB iter {:4}  ‖r‖/‖b‖ = {res_f:.6e}", k + 1);
            }

            if res < tol || r.norm2() < atol {
                if params.verbose != VerboseLevel::Silent {
                    println!("  BiCGSTAB converged iter {}  ‖r‖/‖b‖={res_f:.3e}", k + 1);
                }
                return Ok(SolverResult {
                    converged: true,
                    iterations: k + 1,
                    final_residual: to_f64(res),
                    residual_history: residual_history.clone(),
                    history,
                });
            }

            if omega.abs() < T::machine_epsilon() {
                return Err(SolverError::NumericalBreakdown {
                    detail: format!("BiCGSTAB: omega ≈ 0 at iter {k}"),
                });
            }
        }

        let final_residual = to_f64(r.norm2() / norm_b_f);
        Err(SolverError::ConvergenceFailed { max_iter: params.max_iter, residual: final_residual })
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
        None => dst.copy_from(src),
    }
}

fn to_f64<T: Scalar>(v: T) -> f64 {
    num_traits::ToPrimitive::to_f64(&v).unwrap_or(f64::INFINITY)
}
