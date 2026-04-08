//! Orthogonal (Subspace) Iteration — computes `k` dominant eigenpairs.
//!
//! **Algorithm (block power iteration with QR orthogonalisation):**
//! ```text
//! X₀ = random n×k matrix with orthonormal columns
//! for iter = 0, 1, …:
//!     Z       = A Xᵢ           (k matrix-vector products)
//!     Xᵢ₊₁, R = QR(Z)          (modified Gram-Schmidt)
//!     Λᵢ      = diag(Xᵢᵀ A Xᵢ) (Rayleigh quotients)
//!     if all ‖A xⱼ − λⱼ xⱼ‖ / |λⱼ| < tol  →  converge
//! ```
//!
//! Convergence rate of the j-th pair is |λₖ₊₁/λⱼ|.

use crate::core::{error::SolverError, operator::LinearOperator, scalar::Scalar, vector::{DenseVec, Vector}};
use super::{
    EigenParams, EigenResult, EigenSolver, EigenWhich,
    fill_random, normalise, rayleigh_quotient, residual_norm, dot, orthogonalise_against,
};

/// Orthogonal (subspace) iteration.
///
/// Works for **symmetric** operators; for non-symmetric operators eigenvalues
/// are still real Rayleigh quotients but eigenvectors may not be accurate —
/// use Arnoldi (Sprint 8) in that case.
pub struct SubspaceIter {
    pub seed: u64,
}

impl Default for SubspaceIter {
    fn default() -> Self { SubspaceIter { seed: 42 } }
}

impl SubspaceIter {
    pub fn new(seed: u64) -> Self { SubspaceIter { seed } }
}

impl<T: Scalar> EigenSolver<T> for SubspaceIter {
    fn solve<Op>(
        &self,
        op: &Op,
        params: &EigenParams<T>,
    ) -> Result<EigenResult<T>, SolverError>
    where
        Op: LinearOperator<Vector = DenseVec<T>>,
    {
        let n = op.nrows();
        let k = params.n_eigenvalues;
        assert_eq!(n, op.ncols(), "SubspaceIter: operator must be square");
        assert!(k >= 1 && k <= n, "n_eigenvalues must be in 1..=n");

        // ── Initialise k random orthonormal columns ────────────────────────
        let mut basis: Vec<DenseVec<T>> = Vec::with_capacity(k);
        for j in 0..k {
            let mut v = DenseVec::zeros(n);
            fill_random(&mut v, self.seed.wrapping_add(j as u64 * 0x9e37));
            orthogonalise_against(&mut v, &basis);
            normalise(&mut v);
            basis.push(v);
        }

        let mut lambdas  = vec![T::zero(); k];
        let mut ax_vecs: Vec<DenseVec<T>> = (0..k).map(|_| DenseVec::zeros(n)).collect();
        let mut res_norms = vec![T::zero(); k];

        for iter in 0..params.max_iter {
            // ── Z = A X (k SpMVs) ─────────────────────────────────────────
            for j in 0..k {
                op.apply(&basis[j], &mut ax_vecs[j]);
            }

            // ── Rayleigh quotients ─────────────────────────────────────────
            for j in 0..k {
                lambdas[j] = rayleigh_quotient(&ax_vecs[j], &basis[j]);
                res_norms[j] = residual_norm(&ax_vecs[j], &basis[j], lambdas[j]);
            }

            // ── Convergence check ──────────────────────────────────────────
            let all_converged = (0..k).all(|j| {
                let lam_abs = if lambdas[j].abs() > T::from_f64(1e-14) {
                    lambdas[j].abs()
                } else {
                    T::one()
                };
                res_norms[j] / lam_abs < params.tol
            });

            if params.verbose {
                let max_rel = (0..k).map(|j| {
                    let lam_abs = if lambdas[j].abs() > T::from_f64(1e-14) { lambdas[j].abs() } else { T::one() };
                    num_traits::ToPrimitive::to_f64(&(res_norms[j] / lam_abs)).unwrap_or(f64::NAN)
                }).fold(0f64, f64::max);
                println!("  SubspaceIter iter {:4}  max ‖r‖/|λ| = {max_rel:.3e}", iter + 1);
            }

            if all_converged {
                // Sort by descending |λ|
                let mut pairs: Vec<(T, DenseVec<T>, T)> = lambdas.iter()
                    .zip(ax_vecs.iter())
                    .zip(res_norms.iter())
                    .map(|((&lam, axv), &rn)| (lam, axv.clone(), rn))
                    .collect();
                sort_by_which(&mut pairs, params.which);

                return Ok(EigenResult {
                    eigenvalues:  pairs.iter().map(|(l, _, _)| *l).collect(),
                    eigenvectors: pairs.iter().map(|(_, v, _)| {
                        let mut ev = v.clone(); normalise(&mut ev); ev
                    }).collect(),
                    converged:    k,
                    iterations:   iter + 1,
                    residuals:    pairs.iter().map(|(_, _, r)| *r).collect(),
                });
            }

            // ── QR via modified Gram-Schmidt on Z columns ──────────────────
            // Replace basis with orthonormal columns of Z
            for j in 0..k {
                basis[j].copy_from(&ax_vecs[j]);
                // Orthogonalise against previous new basis columns
                for i in 0..j {
                    let proj = dot(basis[j].as_slice(), basis[i].as_slice());
                    let (left, right) = basis.split_at_mut(j);
                    let bj = &mut right[0];
                    let bi = &left[i];
                    let bis = bi.as_slice();
                    let bjs = bj.as_mut_slice();
                    for idx in 0..n { bjs[idx] -= proj * bis[idx]; }
                }
                normalise(&mut basis[j]);
            }
        }

        // Non-convergence — return best estimates + error
        let _converged_count = (0..k).filter(|&j| {
            let lam_abs = if lambdas[j].abs() > T::from_f64(1e-14) { lambdas[j].abs() } else { T::one() };
            res_norms[j] / lam_abs < params.tol
        }).count();

        let worst_rel = (0..k).map(|j| {
            let lam_abs = if lambdas[j].abs() > T::from_f64(1e-14) { lambdas[j].abs() } else { T::one() };
            num_traits::ToPrimitive::to_f64(&(res_norms[j] / lam_abs)).unwrap_or(f64::INFINITY)
        }).fold(0f64, f64::max);

        Err(SolverError::ConvergenceFailed {
            max_iter: params.max_iter,
            residual: worst_rel,
        })
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn sort_by_which<T: Scalar>(pairs: &mut [(T, DenseVec<T>, T)], which: EigenWhich) {
    match which {
        EigenWhich::LargestMagnitude | EigenWhich::BothEnds =>
            pairs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap()),
        EigenWhich::SmallestMagnitude =>
            pairs.sort_by(|a, b| a.0.abs().partial_cmp(&b.0.abs()).unwrap()),
        EigenWhich::LargestAlgebraic =>
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()),
        EigenWhich::SmallestAlgebraic =>
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()),
    }
}
