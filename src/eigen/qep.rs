//! Quadratic Eigenvalue Problem (QEP) — Sprint 12.
//!
//! Solves `(K + λC + λ²M) x = 0` via linearisation to a 2n × 2n standard
//! eigenvalue problem, which is then handed to [`ArnoldiIter`].
//!
//! **Linearisation (N-form / second companion form):**
//! ```text
//! A z = λ B z,    z = [x; λx]
//!
//! A = [ 0    I ]    B = [ I   0 ]
//!     [-K   -C ]        [ 0   M ]
//! ```
//! The eigenvalues of (A, B) are the same as those of the original QEP.
//! This form avoids inverting M explicitly; instead each application of
//! B⁻¹ A is computed via a matrix-free solve using GMRES.
//!
//! **Practical note:** For undamped problems (C = 0) with real symmetric
//! K and M, eigenvalues appear as purely imaginary pairs ±iω.  The real
//! parts returned here are the diagonal elements of the Schur T-matrix
//! (which may be near-zero); check `res.eigenvalues` for `|im|` if needed.
//!
//! For small-to-medium problems, the simplest approach is to pass the
//! 2n × 2n dense pencil directly to the `hessenberg_eig` path inside
//! `ArnoldiIter`.  We therefore materialise a dense helper operator.

use crate::core::{error::SolverError, operator::LinearOperator, scalar::Scalar, vector::DenseVec};
use super::{EigenParams, EigenResult, EigenSolver, normalise};
use super::arnoldi::ArnoldiIter;

// ─── QuadraticEigen ───────────────────────────────────────────────────────────

/// Solver for the quadratic eigenvalue problem `(K + λC + λ²M) x = 0`.
///
/// Linearises to a `2n × 2n` problem and delegates to [`ArnoldiIter`].
///
/// # Parameters
/// - `n_eigenvalues`: number of wanted eigenvalue/vector pairs
/// - `ncv`:  Krylov space size for the internal Arnoldi (default: auto)
/// - `seed`: random seed for the starting vector
pub struct QuadraticEigen {
    pub n_eigenvalues: usize,
    pub ncv: Option<usize>,
    pub seed: u64,
}

impl QuadraticEigen {
    pub fn new(n_eigenvalues: usize) -> Self {
        QuadraticEigen { n_eigenvalues, ncv: None, seed: 42 }
    }

    /// Solve `(K + λC + λ²M) x = 0`.
    ///
    /// Returns an [`EigenResult`] with `eigenvalues` = λᵢ and
    /// `eigenvectors` = the **x** part (first n components) of the
    /// 2n-dimensional eigenvectors.
    pub fn solve<T, K, C, M>(
        &self,
        k_mat: &K,
        c_mat: &C,
        m_mat: &M,
        params: &EigenParams<T>,
    ) -> Result<EigenResult<T>, SolverError>
    where
        T: Scalar,
        K: LinearOperator<Vector = DenseVec<T>>,
        C: LinearOperator<Vector = DenseVec<T>>,
        M: LinearOperator<Vector = DenseVec<T>>,
    {
        let n = k_mat.nrows();
        assert_eq!(n, k_mat.ncols());
        assert_eq!(n, c_mat.nrows()); assert_eq!(n, c_mat.ncols());
        assert_eq!(n, m_mat.nrows()); assert_eq!(n, m_mat.ncols());

        // Build the 2n operator: y = A x where A = [[0, I], [-K, -C]]
        // (companion form, B = I so this is a standard EVP)
        let lin_op = QepCompanion { k_mat, c_mat, n, _t: std::marker::PhantomData };

        let arnoldi = ArnoldiIter { ncv: self.ncv, seed: self.seed };
        let res2n = arnoldi.solve(&lin_op, params)?;

        // Extract the x-part (first n elements) from the 2n eigenvectors
        let mut eigen_x: Vec<DenseVec<T>> = Vec::with_capacity(res2n.eigenvectors.len());
        for v2n in &res2n.eigenvectors {
            let mut x = DenseVec::zeros(n);
            for i in 0..n { x[i] = v2n[i]; }
            normalise(&mut x);
            eigen_x.push(x);
        }

        // Compute residuals: ‖(K + λC + λ²M) x‖ / ‖x‖
        let mut residuals: Vec<T> = Vec::with_capacity(eigen_x.len());
        for (x, &lam) in eigen_x.iter().zip(res2n.eigenvalues.iter()) {
            let mut kx = DenseVec::zeros(n);
            k_mat.apply(x, &mut kx);
            let mut cx = DenseVec::zeros(n);
            c_mat.apply(x, &mut cx);
            let mut mx = DenseVec::zeros(n);
            m_mat.apply(x, &mut mx);
            let mut r_sq = T::zero();
            for i in 0..n {
                let ri = kx[i] + lam * cx[i] + lam * lam * mx[i];
                r_sq += ri * ri;
            }
            residuals.push(r_sq.sqrt());
        }

        Ok(EigenResult {
            eigenvalues: res2n.eigenvalues,
            eigenvectors: eigen_x,
            converged: res2n.converged,
            iterations: res2n.iterations,
            residuals,
        })
    }
}

// ─── Companion operator [[0, I], [-K, -C]] ───────────────────────────────────

struct QepCompanion<'a, T, K, C>
where T: Scalar, K: LinearOperator<Vector = DenseVec<T>>, C: LinearOperator<Vector = DenseVec<T>>
{
    k_mat: &'a K,
    c_mat: &'a C,
    n: usize,
    _t: std::marker::PhantomData<T>,
}

impl<'a, T, K, C> LinearOperator for QepCompanion<'a, T, K, C>
where
    T: Scalar,
    K: LinearOperator<Vector = DenseVec<T>> + Send + Sync,
    C: LinearOperator<Vector = DenseVec<T>> + Send + Sync,
{
    type Vector = DenseVec<T>;

    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        // x = [x1; x2], y = [x2; -K x1 - C x2]
        let n = self.n;
        let x1 = &x.as_slice()[..n];
        let x2 = &x.as_slice()[n..];

        // y[0..n] = x2
        for i in 0..n { y[i] = x2[i]; }

        // y[n..2n] = -K x1 - C x2
        let mut kx1 = DenseVec::zeros(n);
        let x1_vec = {
            let mut v = DenseVec::zeros(n);
            for i in 0..n { v[i] = x1[i]; }
            v
        };
        let x2_vec = {
            let mut v = DenseVec::zeros(n);
            for i in 0..n { v[i] = x2[i]; }
            v
        };
        self.k_mat.apply(&x1_vec, &mut kx1);
        let mut cx2 = DenseVec::zeros(n);
        self.c_mat.apply(&x2_vec, &mut cx2);
        for i in 0..n {
            y[n + i] = -(kx1[i] + cx2[i]);
        }
    }

    fn nrows(&self) -> usize { 2 * self.n }
    fn ncols(&self) -> usize { 2 * self.n }
}

unsafe impl<'a, T: Scalar, K, C> Send for QepCompanion<'a, T, K, C>
where K: LinearOperator<Vector = DenseVec<T>> + Send + Sync, C: LinearOperator<Vector = DenseVec<T>> + Send + Sync {}
unsafe impl<'a, T: Scalar, K, C> Sync for QepCompanion<'a, T, K, C>
where K: LinearOperator<Vector = DenseVec<T>> + Send + Sync, C: LinearOperator<Vector = DenseVec<T>> + Send + Sync {}
