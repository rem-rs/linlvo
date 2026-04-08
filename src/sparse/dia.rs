//! Diagonal (DIA) sparse matrix format.
//!
//! DIA stores a sparse matrix as a set of diagonals.  Each diagonal is
//! identified by its **offset** from the main diagonal:
//! - `offset = 0`:  main diagonal
//! - `offset > 0`:  super-diagonal (above main)
//! - `offset < 0`:  sub-diagonal (below main)
//!
//! Layout:
//! - `offsets`: sorted list of stored diagonal offsets (length `num_diags`).
//! - `data`: 2-D array of shape `[nrows × num_diags]` stored **column-major**
//!   (i.e., `data[i + d * nrows]` is the value of diagonal `d` at row `i`).
//!   Entries that fall outside the matrix bounds are ignored during SpMV.
//!
//! **When to use:**
//! - Structured finite-difference / finite-element stencils (Poisson, heat,
//!   wave equation) where all nonzeros lie on a fixed set of diagonals.
//! - Banded matrices (e.g., tridiagonal, pentadiagonal).
//!
//! **When NOT to use:**
//! - General unstructured sparsity — use CSR instead.
//! - Very wide diagonals relative to matrix size (poor fill ratio).
//!
//! **Analogs:**
//!   SciPy: `scipy.sparse.dia_matrix`
//!   cuSPARSE: `CUSPARSE_FORMAT_BSR` (banded via DIA-like layouts)

#![allow(clippy::needless_range_loop)]
use crate::core::{operator::LinearOperator, scalar::Scalar, vector::DenseVec};
use crate::sparse::{CooMatrix, CsrMatrix};

/// Diagonal (DIA) sparse matrix.
///
/// # Examples
/// ```
/// use linger::sparse::{CooMatrix, dia::DiaMatrix};
///
/// // Tridiagonal [-1, 2, -1]
/// let n = 5usize;
/// let mut coo: CooMatrix<f64> = CooMatrix::new(n, n);
/// for i in 0..n {
///     coo.push(i, i, 2.0);
///     if i > 0     { coo.push(i, i - 1, -1.0); }
///     if i < n - 1 { coo.push(i, i + 1, -1.0); }
/// }
/// let dia = DiaMatrix::from_csr(&linger::sparse::CsrMatrix::from_coo(&coo));
/// assert_eq!(dia.num_diags(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct DiaMatrix<T> {
    nrows:     usize,
    ncols:     usize,
    /// Diagonal offsets (sorted ascending).
    offsets:   Vec<isize>,
    /// Values, column-major over diagonals: `data[i + d * nrows]` for row `i`,
    /// diagonal index `d`.  Out-of-bounds positions store `T::zero()`.
    data:      Vec<T>,
}

impl<T: Scalar> DiaMatrix<T> {
    // ─── Constructors ────────────────────────────────────────────────────────

    /// Build a DIA matrix from a CSR matrix.
    ///
    /// Every distinct diagonal that has at least one nonzero is stored.
    /// The resulting `offsets` array is sorted ascending.
    pub fn from_csr(csr: &CsrMatrix<T>) -> Self {
        let nrows = csr.nrows();
        let ncols = csr.ncols();

        if nrows == 0 || csr.nnz() == 0 {
            return Self { nrows, ncols, offsets: vec![], data: vec![] };
        }

        // Collect unique diagonals.
        let mut diag_set: std::collections::BTreeSet<isize> = Default::default();
        let rp  = csr.row_ptr();
        let ci  = csr.col_idx();
        for i in 0..nrows {
            for k in rp[i]..rp[i + 1] {
                let offset = ci[k] as isize - i as isize;
                diag_set.insert(offset);
            }
        }

        let offsets: Vec<isize> = diag_set.into_iter().collect(); // already sorted
        let num_diags = offsets.len();
        let mut data = vec![T::zero(); nrows * num_diags];

        // Build a fast offset → diagonal-index map.
        let offset_to_d: std::collections::HashMap<isize, usize> =
            offsets.iter().enumerate().map(|(d, &off)| (off, d)).collect();

        let vals = csr.values();
        for i in 0..nrows {
            for k in rp[i]..rp[i + 1] {
                let offset = ci[k] as isize - i as isize;
                let d = offset_to_d[&offset];
                data[i + d * nrows] = vals[k];
            }
        }

        Self { nrows, ncols, offsets, data }
    }

    /// Build a DIA matrix from COO format (convenience wrapper).
    pub fn from_coo(coo: &CooMatrix<T>) -> Self {
        Self::from_csr(&CsrMatrix::from_coo(coo))
    }

    // ─── Dimensions ──────────────────────────────────────────────────────────

    /// Number of rows.
    pub fn nrows(&self) -> usize { self.nrows }
    /// Number of columns.
    pub fn ncols(&self) -> usize { self.ncols }
    /// Number of stored diagonals.
    pub fn num_diags(&self) -> usize { self.offsets.len() }
    /// Diagonal offsets (sorted ascending).
    pub fn offsets(&self) -> &[isize] { &self.offsets }
    /// Raw value array (column-major, `nrows × num_diags`).
    pub fn data(&self) -> &[T] { &self.data }
    /// Actual nonzero count (entries whose column falls within `[0, ncols)`).
    pub fn nnz(&self) -> usize {
        let nrows = self.nrows as isize;
        let ncols = self.ncols as isize;
        let mut count = 0usize;
        for (d, &off) in self.offsets.iter().enumerate() {
            for i in 0..self.nrows {
                let col = i as isize + off;
                if col >= 0 && col < ncols {
                    let v = self.data[i + d * self.nrows];
                    if v != T::zero() { count += 1; }
                }
                let _ = nrows;
            }
        }
        count
    }

    // ─── SpMV ────────────────────────────────────────────────────────────────

    /// Compute `y = A * x` (in-place, overwrites y).
    ///
    /// Iterates over stored diagonals; entries outside `[0, ncols)` are skipped.
    ///
    /// # Panics
    /// Panics if `x.len() != ncols` or `y.len() != nrows`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols, "DiaMatrix::spmv: x length mismatch");
        assert_eq!(y.len(), self.nrows, "DiaMatrix::spmv: y length mismatch");

        for yi in y.iter_mut() { *yi = T::zero(); }

        let ncols = self.ncols as isize;
        let nrows  = self.nrows;

        for (d, &off) in self.offsets.iter().enumerate() {
            for i in 0..nrows {
                let col = i as isize + off;
                if col < 0 || col >= ncols { continue; }
                y[i] += self.data[i + d * nrows] * x[col as usize];
            }
        }
    }

    // ─── Conversions ─────────────────────────────────────────────────────────

    /// Convert back to CSR format.
    pub fn to_csr(&self) -> CsrMatrix<T> {
        let ncols = self.ncols as isize;
        let nrows  = self.nrows;
        let mut coo = CooMatrix::with_capacity(nrows, self.ncols, self.nnz());

        for (d, &off) in self.offsets.iter().enumerate() {
            for i in 0..nrows {
                let col = i as isize + off;
                if col < 0 || col >= ncols { continue; }
                let v = self.data[i + d * nrows];
                if v != T::zero() {
                    coo.push(i, col as usize, v);
                }
            }
        }
        CsrMatrix::from_coo(&coo)
    }
}

// ─── LinearOperator impl ─────────────────────────────────────────────────────

impl<T: Scalar> LinearOperator for DiaMatrix<T> {
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
    fn tridiagonal_has_three_diags() {
        let csr = laplacian_1d(6);
        let dia = DiaMatrix::<f64>::from_csr(&csr);
        assert_eq!(dia.num_diags(), 3);
        // offsets should be -1, 0, +1 (sorted)
        assert_eq!(dia.offsets(), &[-1isize, 0, 1]);
    }

    #[test]
    fn nnz_matches_csr() {
        let csr = laplacian_1d(7);
        let dia = DiaMatrix::<f64>::from_csr(&csr);
        assert_eq!(dia.nnz(), csr.nnz());
    }

    #[test]
    fn spmv_identity() {
        let n = 5;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n { coo.push(i, i, 1.0); }
        let dia = DiaMatrix::from_coo(&coo);
        let x = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![0.0f64; n];
        dia.spmv(&x, &mut y);
        assert_eq!(y, x);
    }

    #[test]
    fn spmv_matches_csr() {
        let csr = laplacian_1d(8);
        let dia = DiaMatrix::<f64>::from_csr(&csr);
        let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let mut y_csr = vec![0.0f64; 8];
        let mut y_dia = vec![0.0f64; 8];
        csr.spmv(&x, &mut y_csr);
        dia.spmv(&x, &mut y_dia);
        for (a, b) in y_csr.iter().zip(y_dia.iter()) {
            assert!((a - b).abs() < 1e-14, "CSR={a}, DIA={b}");
        }
    }

    #[test]
    fn to_csr_roundtrip() {
        let csr = laplacian_1d(7);
        let dia = DiaMatrix::<f64>::from_csr(&csr);
        let csr2 = dia.to_csr();
        assert_eq!(csr.nrows(), csr2.nrows());
        assert_eq!(csr.nnz(), csr2.nnz());
    }

    #[test]
    fn superdiagonal_only() {
        // Matrix with only one super-diagonal (offset = +2).
        let mut coo = CooMatrix::new(4, 4);
        coo.push(0, 2, 3.0);
        coo.push(1, 3, 5.0);
        let dia = DiaMatrix::from_coo(&coo);
        assert_eq!(dia.num_diags(), 1);
        assert_eq!(dia.offsets(), &[2isize]);
        let x = vec![0.0f64, 0.0, 1.0, 2.0];
        let mut y = vec![0.0f64; 4];
        dia.spmv(&x, &mut y);
        assert!((y[0] - 3.0).abs() < 1e-14);
        assert!((y[1] - 10.0).abs() < 1e-14);
        assert!((y[2]).abs() < 1e-14);
        assert!((y[3]).abs() < 1e-14);
    }

    #[test]
    fn empty_matrix() {
        let coo: CooMatrix<f64> = CooMatrix::new(0, 0);
        let dia = DiaMatrix::from_coo(&coo);
        assert_eq!(dia.nrows(), 0);
        assert_eq!(dia.num_diags(), 0);
    }

    #[test]
    fn rectangular_matrix() {
        // 3×5 matrix with one entry on the super-diagonal (offset = +2)
        let mut coo = CooMatrix::new(3, 5);
        coo.push(0, 2, 7.0);
        coo.push(1, 3, 8.0);
        coo.push(2, 4, 9.0);
        let dia = DiaMatrix::from_coo(&coo);
        let x = vec![0.0f64, 0.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0f64; 3];
        dia.spmv(&x, &mut y);
        assert!((y[0] - 7.0).abs() < 1e-14);
        assert!((y[1] - 8.0).abs() < 1e-14);
        assert!((y[2] - 9.0).abs() < 1e-14);
    }

    #[test]
    fn linear_operator_trait() {
        let csr = laplacian_1d(4);
        let dia = DiaMatrix::<f64>::from_csr(&csr);
        let x = DenseVec::from_vec(vec![1.0f64; 4]);
        let mut y = DenseVec::zeros(4);
        dia.apply(&x, &mut y);
        // boundary: 2-1=1, interior: -1+2-1=0
        assert!((y.as_slice()[0] - 1.0).abs() < 1e-14);
        assert!((y.as_slice()[1]).abs() < 1e-14);
    }
}
