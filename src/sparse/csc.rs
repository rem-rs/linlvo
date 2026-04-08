#![allow(clippy::needless_range_loop)]
use crate::core::{operator::LinearOperator, scalar::Scalar, vector::DenseVec};
use crate::sparse::csr::CsrMatrix;

/// Compressed Sparse Column (CSC) matrix.
///
/// Stored as the transpose image of CSR: `col_ptr[j]..col_ptr[j+1]` indexes
/// all entries in column `j`, with their row indices in `row_idx`.
///
/// The primary way to obtain a CSC matrix is via [`CsrMatrix::transpose`].
#[derive(Debug, Clone)]
pub struct CscMatrix<T> {
    nrows:   usize,
    ncols:   usize,
    col_ptr: Vec<usize>, // length ncols + 1
    row_idx: Vec<usize>, // length nnz
    values:  Vec<T>,     // length nnz
}

impl<T: Scalar> CscMatrix<T> {
    // ─── Constructors ────────────────────────────────────────────────────────

    /// Construct from raw CSC arrays.
    ///
    /// # Panics
    /// Panics if `col_ptr.len() != ncols + 1`.
    pub fn from_raw(
        nrows:   usize,
        ncols:   usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        values:  Vec<T>,
    ) -> Self {
        assert_eq!(col_ptr.len(), ncols + 1, "col_ptr must have ncols+1 entries");
        assert_eq!(row_idx.len(), values.len(), "row_idx and values must have equal length");
        Self { nrows, ncols, col_ptr, row_idx, values }
    }

    // ─── Dimensions / accessors ──────────────────────────────────────────────

    /// Number of rows.
    pub fn nrows(&self) -> usize { self.nrows }
    /// Number of columns.
    pub fn ncols(&self) -> usize { self.ncols }
    /// Number of stored non-zero entries.
    pub fn nnz(&self) -> usize { self.values.len() }

    /// Raw column-pointer array (length `ncols + 1`).
    pub fn col_ptr(&self) -> &[usize] { &self.col_ptr }
    /// Raw row-index array (length `nnz`).
    pub fn row_idx(&self) -> &[usize] { &self.row_idx }
    /// Raw value array (length `nnz`).
    pub fn values(&self) -> &[T] { &self.values }

    // ─── Sparse matrix–vector product ────────────────────────────────────────

    /// Compute  `y ← A · x`  (overwrites `y`).
    ///
    /// Iterates column-by-column which is cache-friendly for CSC.
    ///
    /// # Panics
    /// Panics if `x.len() != ncols` or `y.len() != nrows`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols, "CscMatrix::spmv: x length mismatch");
        assert_eq!(y.len(), self.nrows, "CscMatrix::spmv: y length mismatch");
        // Zero y first.
        for v in y.iter_mut() { *v = T::zero(); }
        for j in 0..self.ncols {
            let xj = x[j];
            if xj == T::zero() { continue; }
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                y[self.row_idx[k]] += self.values[k] * xj;
            }
        }
    }

    // ─── Conversion back to CSR ───────────────────────────────────────────────

    /// Convert to CSR (= transpose of `self` as CSC).
    pub fn to_csr(&self) -> CsrMatrix<T> {
        CsrMatrix::from_raw(
            self.ncols,
            self.nrows,
            self.col_ptr.clone(),
            self.row_idx.clone(),
            self.values.clone(),
        )
    }

    /// Transpose: return the CSR representation of `Aᵀ`.
    ///
    /// For CSC this is the same data re-interpreted as CSR.
    pub fn transpose(&self) -> CsrMatrix<T> {
        self.to_csr()
    }
}

// ─── LinearOperator impl ──────────────────────────────────────────────────────

impl<T: Scalar> LinearOperator for CscMatrix<T> {
    type Vector = DenseVec<T>;

    #[inline]
    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        self.spmv(x.as_slice(), y.as_mut_slice());
    }

    fn nrows(&self) -> usize { self.nrows }
    fn ncols(&self) -> usize { self.ncols }
}
