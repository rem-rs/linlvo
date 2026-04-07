//! Singular Value Decomposition via Lanczos bidiagonalisation — Sprint 11.
//!
//! Computes the **k largest** singular values (and optionally left/right
//! singular vectors) of a rectangular operator `A : Rⁿ → Rᵐ`.
//!
//! **Algorithm** (Lanczos on AᵀA):
//! ```text
//! 1. Run LanczosIter on the symmetric PSD operator (AᵀA) to get k eigenvalues
//!    λᵢ and right singular vectors vᵢ.
//! 2. Singular values:  σᵢ = √max(λᵢ, 0)
//! 3. Left singular vectors: uᵢ = A vᵢ / σᵢ   (if σᵢ > tol)
//! ```
//!
//! For best accuracy on rectangular matrices use `ncv ≥ 2k + 1`.

use crate::core::{
    error::SolverError,
    operator::{LinearOperator, TransposeOperator},
    scalar::Scalar,
    vector::{DenseVec, Vector},
};
use super::{EigenParams, EigenWhich, fill_random, dot, normalise};
use super::lanczos::{LanczosIter, sort_ritz};
use super::EigenSolver;

// ─── SvdResult ────────────────────────────────────────────────────────────────

/// Result of a partial SVD computation.
#[derive(Debug, Clone)]
pub struct SvdResult<T: Scalar> {
    /// Singular values `σ₁ ≥ σ₂ ≥ … ≥ σₖ`.
    pub singular_values: Vec<T>,
    /// Left singular vectors `U[:,i]` (length m). `None` if not requested.
    pub left_vectors: Option<Vec<DenseVec<T>>>,
    /// Right singular vectors `V[:,i]` (length n). `None` if not requested.
    pub right_vectors: Option<Vec<DenseVec<T>>>,
    /// Number of singular values that converged.
    pub converged: usize,
    /// Residual norms `‖A vᵢ − σᵢ uᵢ‖₂` (empty if vectors not computed).
    pub residuals: Vec<T>,
}

// ─── LanczosSvd ──────────────────────────────────────────────────────────────

/// Partial SVD via Lanczos on AᵀA.
///
/// Requires the operator to implement [`TransposeOperator`] so that both
/// `A x` and `Aᵀ y` products can be computed.
///
/// # Example
/// ```rust,ignore
/// let svd = LanczosSvd::default();
/// let res = svd.solve(&a, 3, 1e-10, 200, true).unwrap();
/// println!("σ = {:?}", res.singular_values);
/// ```
pub struct LanczosSvd {
    /// Krylov space size. `None` → auto (`min(2k + 1, n)`).
    pub ncv: Option<usize>,
    pub seed: u64,
}

impl Default for LanczosSvd {
    fn default() -> Self { LanczosSvd { ncv: None, seed: 42 } }
}

impl LanczosSvd {
    pub fn new(ncv: usize) -> Self { LanczosSvd { ncv: Some(ncv), seed: 42 } }

    /// Compute the `k` largest singular values of `op`.
    ///
    /// Set `compute_vectors = true` to also obtain U and V.
    pub fn solve<T, Op>(
        &self,
        op: &Op,
        k: usize,
        tol: T,
        max_iter: usize,
        compute_vectors: bool,
    ) -> Result<SvdResult<T>, SolverError>
    where
        T: Scalar,
        Op: LinearOperator<Vector = DenseVec<T>> + TransposeOperator,
    {
        let m = op.nrows();
        let n = op.ncols();
        assert!(k >= 1, "k must be ≥ 1");
        assert!(k <= n.min(m), "k must be ≤ min(nrows, ncols)");
        // LanczosIter requires nev < n; for k == n we compute n-1 and hope
        // the smallest is negligible (or just cap).
        let nev = k.min(n - 1).max(1);

        // Wrap A as the operator B = AᵀA acting on Rⁿ
        let b_op = AtAOperator { op, n };

        let ncv = self.ncv.unwrap_or_else(|| n.min(nev + nev.max(20)));
        let lanczos = LanczosIter { ncv: Some(ncv), seed: self.seed };

        let mut params: EigenParams<T> = EigenParams {
            n_eigenvalues: nev,
            which: EigenWhich::LargestAlgebraic,
            tol,
            max_iter,
            verbose: false,
        };

        let eigen = lanczos.solve(&b_op, &params)?;

        // Convert λᵢ → σᵢ; sort descending
        let mut pairs: Vec<(T, DenseVec<T>)> = eigen.eigenvalues
            .into_iter()
            .zip(eigen.eigenvectors.into_iter())
            .map(|(lam, v)| {
                let sigma = if lam > T::zero() { lam.sqrt() } else { T::zero() };
                (sigma, v)
            })
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let singular_values: Vec<T> = pairs.iter().map(|(s, _)| *s).collect();
        let right_vecs: Vec<DenseVec<T>> = pairs.into_iter().map(|(_, v)| v).collect();

        if !compute_vectors {
            return Ok(SvdResult {
                singular_values,
                left_vectors: None,
                right_vectors: Some(right_vecs),
                converged: eigen.converged,
                residuals: Vec::new(),
            });
        }

        // Compute left vectors: uᵢ = A vᵢ / σᵢ
        let mut left_vecs: Vec<DenseVec<T>> = Vec::with_capacity(k);
        let mut residuals: Vec<T> = Vec::with_capacity(k);

        for (i, v) in right_vecs.iter().enumerate() {
            let sigma = singular_values[i];
            let mut av = DenseVec::zeros(m);
            op.apply(v, &mut av);
            if sigma > T::from_f64(1e-14) {
                av.scale(T::one() / sigma);
                // Residual: ‖A v − σ u‖ (after normalising u = A v / σ)
                // ≡ ‖A v − σ * (A v / σ)‖ = 0 by definition, so check ‖Aᵀ u − σ v‖
                let mut atu = DenseVec::zeros(n);
                op.apply_transpose(&av, &mut atu);
                let mut diff = T::zero();
                for j in 0..n {
                    let d = atu[j] - sigma * v[j];
                    diff += d * d;
                }
                residuals.push(diff.sqrt());
            } else {
                residuals.push(T::zero());
            }
            left_vecs.push(av);
        }

        Ok(SvdResult {
            singular_values,
            left_vectors: Some(left_vecs),
            right_vectors: Some(right_vecs),
            converged: eigen.converged,
            residuals,
        })
    }
}

// ─── AᵀA operator wrapper ────────────────────────────────────────────────────

struct AtAOperator<'a, Op> {
    op: &'a Op,
    n:  usize,
}

impl<'a, T, Op> LinearOperator for AtAOperator<'a, Op>
where
    T: Scalar,
    Op: LinearOperator<Vector = DenseVec<T>> + TransposeOperator + Send + Sync,
{
    type Vector = DenseVec<T>;

    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        // y = AᵀA x: first w = A x, then y = Aᵀ w
        let m = self.op.nrows();
        let mut w = DenseVec::zeros(m);
        self.op.apply(x, &mut w);
        self.op.apply_transpose(&w, y);
    }

    fn nrows(&self) -> usize { self.n }
    fn ncols(&self) -> usize { self.n }
}

// Safety: AtAOperator only holds a shared reference — it is Send+Sync if Op is.
unsafe impl<'a, Op: Send + Sync> Send for AtAOperator<'a, Op> {}
unsafe impl<'a, Op: Send + Sync> Sync for AtAOperator<'a, Op> {}
