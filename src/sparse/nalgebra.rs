//! Native-target integration for `nalgebra_sparse::CsrMatrix`.
//!
//! `linger::core::LinearOperator` is a local trait, so it can be implemented
//! directly for nalgebra's CSR matrix without a wrapper newtype.

use crate::core::{operator::LinearOperator, scalar::Scalar, vector::DenseVec};
use nalgebra::RealField;
use nalgebra_sparse::CsrMatrix as NaCsrMatrix;

impl<T: Scalar + RealField> LinearOperator for NaCsrMatrix<T> {
    type Vector = DenseVec<T>;

    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let x_s = x.as_slice();
        let y_s = y.as_mut_slice();

        for (i, row) in self.row_iter().enumerate() {
            let mut sum = T::zero();
            for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
                sum += val * x_s[col];
            }
            y_s[i] = sum;
        }
    }

    fn nrows(&self) -> usize { self.nrows() }
    fn ncols(&self) -> usize { self.ncols() }
}