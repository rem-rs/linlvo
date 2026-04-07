//! Power Iteration — finds the dominant (largest-magnitude) eigenpair.
//!
//! **Algorithm:**
//! ```text
//! x₀ = random unit vector
//! for k = 0, 1, …:
//!     y     = A xₖ
//!     λₖ    = xₖᵀ y          (Rayleigh quotient, equals ‖y‖ for exact eigenvector)
//!     x_{k+1} = y / ‖y‖
//!     if ‖Axₖ − λₖ xₖ‖ / |λₖ| < tol  →  converge
//! ```
//!
//! Convergence rate is |λ₁/λ₂|; slow when the two leading eigenvalues are
//! close in magnitude.  Use [`SubspaceIter`] for multiple eigenvalues or when
//! better convergence is needed.

use crate::core::{error::SolverError, operator::LinearOperator, scalar::Scalar, vector::{DenseVec, Vector}};
use super::{
    EigenParams, EigenResult, EigenSolver,
    fill_random, normalise, rayleigh_quotient, residual_norm,
};

// ─── PowerIter ───────────────────────────────────────────────────────────────

/// Power iteration — targets the single eigenvalue of largest absolute value.
///
/// If `params.n_eigenvalues > 1`, the call is forwarded to the multi-vector
/// [`SubspaceIter`] algorithm automatically.
///
/// # Limitations
/// - Real arithmetic only (complex eigenvalues not detected).
/// - Converges to the dominant real eigenpair; may converge slowly when two
///   eigenvalues have nearly equal magnitude.
pub struct PowerIter {
    /// Random seed for the initial vector.
    pub seed: u64,
}

impl Default for PowerIter {
    fn default() -> Self { PowerIter { seed: 42 } }
}

impl PowerIter {
    pub fn new(seed: u64) -> Self { PowerIter { seed } }
}

impl<T: Scalar> EigenSolver<T> for PowerIter {
    fn solve<Op>(
        &self,
        op: &Op,
        params: &EigenParams<T>,
    ) -> Result<EigenResult<T>, SolverError>
    where
        Op: LinearOperator<Vector = DenseVec<T>>,
    {
        // Validate
        let n = op.nrows();
        assert_eq!(n, op.ncols(), "PowerIter: operator must be square");
        assert!(params.n_eigenvalues >= 1, "n_eigenvalues must be >= 1");

        // For multiple eigenvalues delegate to SubspaceIter.
        if params.n_eigenvalues > 1 {
            return super::subspace::SubspaceIter::new(self.seed)
                .solve(op, params);
        }

        // ── Single dominant eigenpair ──────────────────────────────────────
        let mut x = DenseVec::zeros(n);
        fill_random(&mut x, self.seed);
        normalise(&mut x);

        let mut ax = DenseVec::zeros(n);
        let mut lambda = T::zero();
        let mut res_norm = T::zero();

        for iter in 0..params.max_iter {
            // y = A x
            op.apply(&x, &mut ax);

            // λ = xᵀ(Ax) / xᵀx  (x is unit so denominator = 1)
            lambda = rayleigh_quotient(&ax, &x);

            // residual  ‖Ax − λx‖
            res_norm = residual_norm(&ax, &x, lambda);

            // relative tolerance guard (avoid division by zero)
            let lam_abs = if lambda.abs() > T::from_f64(1e-14) {
                lambda.abs()
            } else {
                T::one()
            };
            let rel = res_norm / lam_abs;

            if params.verbose {
                let r = num_traits::ToPrimitive::to_f64(&rel).unwrap_or(f64::NAN);
                let l = num_traits::ToPrimitive::to_f64(&lambda).unwrap_or(f64::NAN);
                println!("  PowerIter iter {:4}  λ = {l:+.8e}  ‖r‖/|λ| = {r:.3e}", iter + 1);
            }

            if rel < params.tol {
                // Normalise final vector
                normalise(&mut ax); // ax now holds y/‖y‖ ≈ eigenvector
                return Ok(EigenResult {
                    eigenvalues:  vec![lambda],
                    eigenvectors: vec![ax],
                    converged:    1,
                    iterations:   iter + 1,
                    residuals:    vec![res_norm],
                });
            }

            // x_{k+1} = y / ‖y‖
            x.copy_from(&ax);
            normalise(&mut x);
        }

        // Return best estimate even on non-convergence (plus error).
        let rel_f = {
            let lam_abs = if lambda.abs() > T::from_f64(1e-14) { lambda.abs() } else { T::one() };
            num_traits::ToPrimitive::to_f64(&(res_norm / lam_abs)).unwrap_or(f64::INFINITY)
        };
        Err(SolverError::ConvergenceFailed {
            max_iter: params.max_iter,
            residual: rel_f,
        })
    }
}
