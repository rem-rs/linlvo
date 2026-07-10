//! Conjugate Gradient solver for complex-symmetric systems.
//!
//! Solves `A x = b` where `A` is a `LinearOperator` over `Complex<T>`.
//! For EFIE matrices that are symmetric (Z_ij = Z_ji), standard CG with
//! the Hermitian inner product ⟨u,v⟩ = Σ conj(uᵢ)·vᵢ converges robustly.
//!
//! Storage: O(N) — only 5 vectors (r, z, p, Ap, Ax).
//!
//! # Algorithm
//! ```text
//! r₀ = b − A x₀,  z₀ = M⁻¹ r₀,  p₀ = z₀
//! for k = 0, 1, …:
//!     α_k  = ⟨r_k, z_k⟩ / ⟨p_k, A p_k⟩
//!     x_{k+1} = x_k + α_k p_k
//!     r_{k+1} = r_k − α_k A p_k
//!     z_{k+1} = M⁻¹ r_{k+1}
//!     β_k  = ⟨r_{k+1}, z_{k+1}⟩ / ⟨r_k, z_k⟩
//!     p_{k+1} = z_{k+1} + β_k p_k
//! ```
//!
//! References: Hestenes & Stiefel (1952), Saad §6.7.

use crate::core::{
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    vector::{DenseVec, Vector},
};
use crate::core::error::SolverError;
use num_complex::Complex;
use num_traits::NumCast;

#[inline]
fn to_f64<T: Scalar>(v: T) -> f64 {
    <f64 as NumCast>::from(v).unwrap_or(f64::INFINITY)
}

/// Result returned by [`ComplexCg`].
#[derive(Debug, Clone)]
pub struct ComplexCgResult {
    /// Number of iterations performed.
    pub iters: usize,
    /// Final relative residual ‖r‖ / ‖b‖.
    pub final_residual: f64,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Residual history (one entry per check_interval steps).
    pub residual_history: Vec<f64>,
}

/// Reusable scratch buffers for repeated CG solves on the same problem size.
pub struct ComplexCgWorkspace<T: Scalar> {
    r: DenseVec<Complex<T>>,
    z: DenseVec<Complex<T>>,
    p: DenseVec<Complex<T>>,
    ap: DenseVec<Complex<T>>,
    ax: DenseVec<Complex<T>>,
}

impl<T: Scalar> ComplexCgWorkspace<T> {
    pub fn new(n: usize) -> Self {
        let zero = Complex::new(T::zero(), T::zero());
        Self {
            r: DenseVec::from(vec![zero; n]),
            z: DenseVec::from(vec![zero; n]),
            p: DenseVec::from(vec![zero; n]),
            ap: DenseVec::from(vec![zero; n]),
            ax: DenseVec::from(vec![zero; n]),
        }
    }

    fn ensure_len(&mut self, n: usize) {
        if self.r.len() != n {
            *self = Self::new(n);
        }
    }
}

/// Conjugate Gradient solver for complex-symmetric linear systems.
///
/// Suitable when the matrix `A` is symmetric (EFIE-type).  Storage O(N).
/// For non-symmetric systems use [`ComplexGmres`][super::ComplexGmres].
pub struct ComplexCg<T> {
    /// How often to recompute the residual from scratch (default 50).
    pub check_interval: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> ComplexCg<T> {
    /// Create a new CG solver.
    ///
    /// `check_interval`: recompute residual every N iterations (default 50).
    pub fn new(check_interval: usize) -> Self {
        ComplexCg { check_interval, _phantom: std::marker::PhantomData }
    }

    /// Solve `A x = b` with internal workspace allocation.
    pub fn solve(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<Complex<T>>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<Complex<T>>>>,
        b: &DenseVec<Complex<T>>,
        x: &mut DenseVec<Complex<T>>,
        rtol: f64,
        atol: f64,
        max_iter: usize,
    ) -> Result<ComplexCgResult, SolverError> {
        let mut ws = ComplexCgWorkspace::new(b.len());
        self.solve_with_workspace(op, precond, b, x, rtol, atol, max_iter, &mut ws)
    }

    /// Solve `A x = b` using caller-owned scratch buffers.
    pub fn solve_with_workspace(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<Complex<T>>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<Complex<T>>>>,
        b: &DenseVec<Complex<T>>,
        x: &mut DenseVec<Complex<T>>,
        rtol: f64,
        atol: f64,
        max_iter: usize,
        workspace: &mut ComplexCgWorkspace<T>,
    ) -> Result<ComplexCgResult, SolverError> {
        let n = b.len();
        workspace.ensure_len(n);

        let norm_b = b.norm2();  // T (real, L2 norm)
        let norm_b_f = if norm_b == T::zero() { T::one() } else { norm_b };
        let mut residual_history: Vec<f64> = Vec::new();

        // r = b - A·x₀
        op.apply(x, &mut workspace.ax);
        for i in 0..n {
            workspace.r.as_mut_slice()[i] = b.as_slice()[i] - workspace.ax.as_slice()[i];
        }

        // z = M⁻¹·r (or just r if no preconditioner)
        match precond {
            Some(m) => m.apply_precond(&workspace.r, &mut workspace.z),
            None => workspace.z.copy_from(&workspace.r),
        }
        workspace.p.copy_from(&workspace.z);

        let mut rz = hermitian_dot(workspace.r.as_slice(), workspace.z.as_slice());

        for k in 0..max_iter {
            // A·p
            op.apply(&workspace.p, &mut workspace.ap);

            // α = ⟨r,z⟩ / ⟨p,Ap⟩
            let pap = hermitian_dot(workspace.p.as_slice(), workspace.ap.as_slice());

            let r_norm = workspace.r.norm2();  // T — L2 norm of current residual
            let res_now = to_f64(r_norm / norm_b_f);
            residual_history.push(res_now);

            if res_now < rtol || to_f64(r_norm) < atol {
                return Ok(ComplexCgResult {
                    converged: true,
                    iters: k + 1,
                    final_residual: res_now,
                    residual_history,
                });
            }

            if pap == Complex::new(T::zero(), T::zero()) {
                return Err(SolverError::NumericalBreakdown {
                    detail: format!("ComplexCG: ⟨p,Ap⟩≈0 at iter {} (res={:.3e}); matrix may be indefinite", k + 1, res_now),
                });
            }

            let alpha = rz / pap;
            // x += α·p
            for i in 0..n {
                workspace.ax.as_mut_slice()[i] = x.as_slice()[i] + alpha * workspace.p.as_slice()[i];
            }
            x.copy_from(&workspace.ax);

            // r -= α·Ap
            for i in 0..n {
                workspace.r.as_mut_slice()[i] -= alpha * workspace.ap.as_slice()[i];
            }

            // Periodic full-residual recomputation to prevent drift
            if (k + 1) % self.check_interval == 0 {
                op.apply(x, &mut workspace.ax);
                for i in 0..n {
                    workspace.r.as_mut_slice()[i] = b.as_slice()[i] - workspace.ax.as_slice()[i];
                }
            }

            // z = M⁻¹·r
            match precond {
                Some(m) => m.apply_precond(&workspace.r, &mut workspace.z),
                None => workspace.z.copy_from(&workspace.r),
            }

            let rz_new = hermitian_dot(workspace.r.as_slice(), workspace.z.as_slice());

            let beta = rz_new / rz;
            // p = z + β·p
            for i in 0..n {
                workspace.p.as_mut_slice()[i] = workspace.z.as_slice()[i] + beta * workspace.p.as_slice()[i];
            }
            rz = rz_new;
        }

        let final_res = to_f64(workspace.r.norm2() / norm_b_f);
        Err(SolverError::ConvergenceFailed { max_iter, residual: final_res })
    }
}

/// Hermitian inner product: ⟨u,v⟩ = Σ conj(uᵢ)·vᵢ
fn hermitian_dot<T: Scalar>(u: &[Complex<T>], v: &[Complex<T>]) -> Complex<T> {
    u.iter().zip(v.iter()).fold(Complex::new(T::zero(), T::zero()), |acc, (u_i, v_i)| {
        acc + u_i.conj() * v_i
    })
}

impl<T: Scalar> Default for ComplexCg<T> {
    fn default() -> Self { Self::new(50) }
}
