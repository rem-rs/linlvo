//! ELLPACK (ELL) sparse matrix format.
//!
//! ELL stores a sparse matrix by padding each row to the same number of
//! nonzeros (`max_nnz_per_row`).  This gives a rectangular `nrows × max_nnz`
//! column-index array and a corresponding value array — both stored
//! **column-major** (Fortran order) to enable coalesced GPU memory access.
//!
//! Layout:
//! - `col_idx[i + j * nrows]`: column index of the j-th stored entry in row i
//!   (padded with `PADDING_IDX = usize::MAX` for unused slots).
//! - `values[i + j * nrows]`: corresponding value (padding entries are `T::zero()`).
//!
//! **When to use:**
//! - Matrices where all rows have roughly the same number of nonzeros.
//! - GPU SpMV kernels (each thread handles one row; coalesced access pattern).
//! - Finite-difference / finite-element stencils with uniform connectivity.
//!
//! **When NOT to use:**
//! - Highly irregular sparsity (one very dense row wastes memory for all others).
//!   Use CSR or BSR instead.
//!
//! **Analogs:**
//!   cuSPARSE: `CUSPARSE_FORMAT_ELL`
//!   ELLPACK-R: extension with per-row nnz for reduced padding

use crate::core::{operator::LinearOperator, scalar::Scalar, vector::DenseVec};
use crate::sparse::{CooMatrix, CsrMatrix};

/// Sentinel column index used for padding slots.
const PADDING_IDX: usize = usize::MAX;

/// ELLPACK (ELL) sparse matrix with uniform-width storage.
///
/// # Examples
/// ```
/// use linger::sparse::{CooMatrix, ell::EllMatrix};
///
/// let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
/// coo.push(0, 0, 2.0); coo.push(0, 1, -1.0);
/// coo.push(1, 0, -1.0); coo.push(1, 1, 2.0); coo.push(1, 2, -1.0);
/// coo.push(2, 1, -1.0); coo.push(2, 2, 2.0);
/// let ell = EllMatrix::from_csr(&linger::sparse::CsrMatrix::from_coo(&coo));
/// assert_eq!(ell.nrows(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct EllMatrix<T> {
    nrows:          usize,
    ncols:          usize,
    /// Maximum number of nonzeros per row (= width of the padded arrays).
    max_nnz_per_row: usize,
    /// Column indices, column-major: `col_idx[i + j * nrows]`.
    /// Padding slots contain `PADDING_IDX`.
    col_idx: Vec<usize>,
    /// Values, column-major: `values[i + j * nrows]`.
    /// Padding slots are `T::zero()`.
    values:  Vec<T>,
}

impl<T: Scalar> EllMatrix<T> {
    // ─── Constructors ────────────────────────────────────────────────────────

    /// Build an ELL matrix from a CSR matrix.
    ///
    /// All rows are padded to `max_nnz_per_row` (the widest row in the CSR).
    /// If the CSR has zero rows or is empty, an empty ELL is returned.
    pub fn from_csr(csr: &CsrMatrix<T>) -> Self {
        let nrows = csr.nrows();
        let ncols = csr.ncols();

        if nrows == 0 {
            return Self { nrows, ncols, max_nnz_per_row: 0, col_idx: vec![], values: vec![] };
        }

        // Find the widest row.
        let max_nnz = (0..nrows)
            .map(|i| csr.row_ptr()[i + 1] - csr.row_ptr()[i])
            .max()
            .unwrap_or(0);

        let col_idx = vec![PADDING_IDX; nrows * max_nnz];
        let values  = vec![T::zero();   nrows * max_nnz];
        let mut ell = Self { nrows, ncols, max_nnz_per_row: max_nnz, col_idx, values };

        let csr_col = csr.col_idx();
        let csr_val = csr.values();
        let rp      = csr.row_ptr();

        for i in 0..nrows {
            for (j, k) in (rp[i]..rp[i + 1]).enumerate() {
                ell.col_idx[i + j * nrows] = csr_col[k];
                ell.values[i + j * nrows]  = csr_val[k];
            }
        }
        ell
    }

    /// Build an ELL matrix from COO format (convenience wrapper).
    pub fn from_coo(coo: &CooMatrix<T>) -> Self {
        Self::from_csr(&CsrMatrix::from_coo(coo))
    }

    // ─── Dimensions ──────────────────────────────────────────────────────────

    /// Number of rows.
    pub fn nrows(&self) -> usize { self.nrows }
    /// Number of columns.
    pub fn ncols(&self) -> usize { self.ncols }
    /// Maximum nonzeros per row (= padded width).
    pub fn max_nnz_per_row(&self) -> usize { self.max_nnz_per_row }
    /// Number of structurally stored slots (including padding).
    pub fn storage_slots(&self) -> usize { self.nrows * self.max_nnz_per_row }
    /// Actual nonzero count (excluding padding).
    pub fn nnz(&self) -> usize {
        self.col_idx.iter().filter(|&&c| c != PADDING_IDX).count()
    }

    // ─── Raw access ──────────────────────────────────────────────────────────

    /// Column-index array (column-major, `nrows × max_nnz_per_row`).
    pub fn col_idx(&self) -> &[usize] { &self.col_idx }
    /// Value array (column-major, `nrows × max_nnz_per_row`).
    pub fn values(&self) -> &[T] { &self.values }

    // ─── SpMV ────────────────────────────────────────────────────────────────

    /// Compute `y = A * x` (in-place, overwrites y).
    ///
    /// # Panics
    /// Panics if `x.len() != ncols` or `y.len() != nrows`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols, "EllMatrix::spmv: x length mismatch");
        assert_eq!(y.len(), self.nrows, "EllMatrix::spmv: y length mismatch");

        for yi in y.iter_mut() { *yi = T::zero(); }

        let nrows = self.nrows;
        for j in 0..self.max_nnz_per_row {
            for i in 0..nrows {
                let c = self.col_idx[i + j * nrows];
                if c == PADDING_IDX { continue; }
                y[i] += self.values[i + j * nrows] * x[c];
            }
        }
    }

    // ─── Conversions ─────────────────────────────────────────────────────────

    /// Convert back to CSR format (drops padding, preserves order).
    pub fn to_csr(&self) -> CsrMatrix<T> {
        let mut coo = CooMatrix::with_capacity(self.nrows, self.ncols, self.nnz());
        let nrows = self.nrows;
        for j in 0..self.max_nnz_per_row {
            for i in 0..nrows {
                let c = self.col_idx[i + j * nrows];
                if c == PADDING_IDX { continue; }
                coo.push(i, c, self.values[i + j * nrows]);
            }
        }
        CsrMatrix::from_coo(&coo)
    }
}

// ─── LinearOperator impl ────────────────────────────────────────────────────

impl<T: Scalar> LinearOperator for EllMatrix<T> {
    type Vector = DenseVec<T>;

    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        use crate::core::vector::Vector;
        if y.len() != self.nrows {
            *y = DenseVec::zeros(self.nrows);
        }
        self.spmv(x.as_slice(), y.as_mut_slice());
    }
    fn nrows(&self) -> usize { self.nrows }
    fn ncols(&self) -> usize { self.ncols }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0     { coo.push(i, i - 1, -1.0); }
            if i < n - 1 { coo.push(i, i + 1, -1.0); }
        }
        CsrMatrix::from_coo(&coo)
    }

    #[test]
    fn from_csr_dimensions() {
        let csr = laplacian_1d(5);
        let ell = EllMatrix::<f64>::from_csr(&csr);
        assert_eq!(ell.nrows(), 5);
        assert_eq!(ell.ncols(), 5);
        // Interior rows have 3 entries; that's the maximum.
        assert_eq!(ell.max_nnz_per_row(), 3);
    }

    #[test]
    fn nnz_matches_csr() {
        let csr = laplacian_1d(6);
        let ell = EllMatrix::<f64>::from_csr(&csr);
        assert_eq!(ell.nnz(), csr.nnz());
    }

    #[test]
    fn spmv_identity() {
        let n = 5;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n { coo.push(i, i, 1.0); }
        let ell = EllMatrix::from_coo(&coo);
        let x = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![0.0f64; n];
        ell.spmv(&x, &mut y);
        assert_eq!(y, x);
    }

    #[test]
    fn spmv_matches_csr() {
        let csr = laplacian_1d(8);
        let ell = EllMatrix::<f64>::from_csr(&csr);
        let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let mut y_csr = vec![0.0f64; 8];
        let mut y_ell = vec![0.0f64; 8];
        csr.spmv(&x, &mut y_csr);
        ell.spmv(&x, &mut y_ell);
        for (a, b) in y_csr.iter().zip(y_ell.iter()) {
            assert!((a - b).abs() < 1e-14, "CSR={a}, ELL={b}");
        }
    }

    #[test]
    fn to_csr_roundtrip() {
        let csr = laplacian_1d(7);
        let ell = EllMatrix::<f64>::from_csr(&csr);
        let csr2 = ell.to_csr();
        assert_eq!(csr.nrows(), csr2.nrows());
        assert_eq!(csr.nnz(), csr2.nnz());
    }

    #[test]
    fn empty_matrix() {
        let coo: CooMatrix<f64> = CooMatrix::new(0, 0);
        let ell = EllMatrix::from_coo(&coo);
        assert_eq!(ell.nrows(), 0);
        assert_eq!(ell.nnz(), 0);
    }

    #[test]
    fn linear_operator_trait() {
        let csr = laplacian_1d(4);
        let ell = EllMatrix::<f64>::from_csr(&csr);
        let x = DenseVec::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let mut y = DenseVec::zeros(4);
        ell.apply(&x, &mut y);
        // Boundary rows: 2*1 - 1 = 1; interior rows: -1+2-1 = 0.
        assert!((y.as_slice()[0] - 1.0).abs() < 1e-14);
        assert!((y.as_slice()[1]).abs() < 1e-14);
    }
}
