//! Adapter wrapping `faer::sparse::SparseColMat` as a linger [`LinearOperator`].
//!
//! Targets faer ≥ 0.21.
//!
//! # Usage
//! ```ignore
//! use faer::sparse::SparseColMat;
//! use linger::sparse::adapt_faer::FaerSparseOp;
//!
//! let faer_mat: SparseColMat<usize, f64> = /* ... */;
//! let op = FaerSparseOp::new(faer_mat);
//! ```

use crate::core::{operator::LinearOperator, scalar::Scalar, vector::DenseVec};
use faer::sparse::SparseColMat;

/// Newtype wrapping a `faer::sparse::SparseColMat<usize, T>` as a linger operator.
///
/// The inner matrix is column-major (CSC); SpMV is implemented by a
/// column-scatter loop compatible with wasm32 (no rayon).
pub struct FaerSparseOp<T>(SparseColMat<usize, T>)
where
    SparseColMat<usize, T>: Send + Sync;

impl<T: Scalar> FaerSparseOp<T>
where
    SparseColMat<usize, T>: Send + Sync,
{
    /// Wrap an existing faer sparse matrix.
    pub fn new(mat: SparseColMat<usize, T>) -> Self {
        FaerSparseOp(mat)
    }

    /// Unwrap and return the inner matrix.
    pub fn into_inner(self) -> SparseColMat<usize, T> {
        self.0
    }
}

impl<T: Scalar> LinearOperator for FaerSparseOp<T>
where
    SparseColMat<usize, T>: Send + Sync,
{
    type Vector = DenseVec<T>;

    /// Compute `y = A · x` via a CSC column-scatter SpMV.
    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let mat = self.0.as_ref();
        let x_s = x.as_slice();
        let y_s = y.as_mut_slice();

        for v in y_s.iter_mut() {
            *v = T::zero();
        }

        let col_ptr  = mat.col_ptr();   // &[usize], length ncols+1
        let row_idx  = mat.row_idx();   // &[usize], length nnz
        let val      = mat.val();       // &[T],     length nnz

        for j in 0..mat.ncols() {
            let start = col_ptr[j];
            let end   = col_ptr[j + 1];
            let xj    = x_s[j];
            for k in start..end {
                let i = row_idx[k];
                y_s[i] += val[k] * xj;
            }
        }
    }

    fn nrows(&self) -> usize { self.0.nrows() }
    fn ncols(&self) -> usize { self.0.ncols() }
}
