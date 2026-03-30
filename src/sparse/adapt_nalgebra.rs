//! Adapter wrapping `nalgebra_sparse::CsrMatrix` as a linger [`LinearOperator`].
//!
//! Requires the `nalgebra-sparse` crate (v0.4).
//!
//! # Usage
//! ```ignore
//! use nalgebra_sparse::CsrMatrix as NaCsr;
//! use linger::sparse::adapt_nalgebra::NalgebraCsrOp;
//!
//! let na_csr: NaCsr<f64> = /* ... */;
//! let op = NalgebraCsrOp::new(na_csr);
//! ```

use crate::core::{operator::LinearOperator, scalar::Scalar, vector::DenseVec};
use nalgebra::RealField;
use nalgebra_sparse::CsrMatrix as NaCsrMatrix;

/// Newtype wrapping a `nalgebra_sparse::CsrMatrix<T>` as a linger operator.
///
/// The vector type is [`DenseVec<T>`]; the SpMV is performed via manual
/// row-iteration over nalgebra's internal CSR structure.
pub struct NalgebraCsrOp<T>(NaCsrMatrix<T>);

impl<T: Scalar + RealField> NalgebraCsrOp<T> {
    /// Wrap an existing nalgebra CSR matrix.
    pub fn new(mat: NaCsrMatrix<T>) -> Self {
        NalgebraCsrOp(mat)
    }

    /// Unwrap and return the inner matrix.
    pub fn into_inner(self) -> NaCsrMatrix<T> {
        self.0
    }

    /// Borrow the inner matrix.
    pub fn inner(&self) -> &NaCsrMatrix<T> {
        &self.0
    }
}

impl<T: Scalar + RealField> LinearOperator for NalgebraCsrOp<T> {
    type Vector = DenseVec<T>;

    /// Compute `y = A · x` by iterating over nalgebra's row views.
    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let x_s = x.as_slice();
        let y_s = y.as_mut_slice();
        for (i, row) in self.0.row_iter().enumerate() {
            let mut sum = T::zero();
            for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                sum += val * x_s[col];
            }
            y_s[i] = sum;
        }
    }

    fn nrows(&self) -> usize { self.0.nrows() }
    fn ncols(&self) -> usize { self.0.ncols() }
}

// ─── SAFETY ──────────────────────────────────────────────────────────────────
// NaCsrMatrix<T: Send+Sync> is Send+Sync because its internal storage
// (row_offsets, col_indices, values) are plain Vecs.
unsafe impl<T: Scalar + RealField + Send> Send for NalgebraCsrOp<T> {}
unsafe impl<T: Scalar + RealField + Sync> Sync for NalgebraCsrOp<T> {}
