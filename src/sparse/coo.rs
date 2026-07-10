use crate::core::scalar::ComplexScalar;

/// Coordinate (COO) sparse matrix — the preferred format for assembly.
///
/// Entries may be added in any order, and duplicate `(row, col)` pairs are
/// summed when converting to CSR/CSC.
///
/// # Examples
/// ```
/// use linger::sparse::CooMatrix;
///
/// let mut m: CooMatrix<f64> = CooMatrix::new(3, 3);
/// m.push(0, 0, 4.0);
/// m.push(0, 1, -1.0);
/// m.push(1, 0, -1.0);
/// m.push(1, 1, 4.0);
/// ```
#[derive(Debug, Clone)]
pub struct CooMatrix<T> {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) rows:   Vec<usize>,
    pub(crate) cols:   Vec<usize>,
    pub(crate) values: Vec<T>,
}

impl<T: ComplexScalar> CooMatrix<T> {
    /// Create an empty `nrows × ncols` COO matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        CooMatrix {
            nrows,
            ncols,
            rows:   Vec::new(),
            cols:   Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create an empty matrix with pre-allocated capacity for `nnz` entries.
    pub fn with_capacity(nrows: usize, ncols: usize, nnz: usize) -> Self {
        CooMatrix {
            nrows,
            ncols,
            rows:   Vec::with_capacity(nnz),
            cols:   Vec::with_capacity(nnz),
            values: Vec::with_capacity(nnz),
        }
    }

    /// Append the entry `(row, col, value)`.
    ///
    /// Duplicate indices are allowed and will be summed during conversion.
    ///
    /// # Panics
    /// Panics in debug mode if `row >= nrows` or `col >= ncols`.
    pub fn push(&mut self, row: usize, col: usize, value: T) {
        debug_assert!(row < self.nrows, "row {row} out of bounds (nrows={})", self.nrows);
        debug_assert!(col < self.ncols, "col {col} out of bounds (ncols={})", self.ncols);
        self.rows.push(row);
        self.cols.push(col);
        self.values.push(value);
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize { self.nrows }
    /// Number of columns.
    pub fn ncols(&self) -> usize { self.ncols }
    /// Number of stored (possibly duplicate) entries.
    pub fn nnz(&self) -> usize { self.rows.len() }

    /// Row indices of all stored entries.
    pub fn row_indices(&self) -> &[usize] { &self.rows }
    /// Column indices of all stored entries.
    pub fn col_indices(&self) -> &[usize] { &self.cols }
    /// Values of all stored entries.
    pub fn values(&self) -> &[T] { &self.values }

    // ── Convenience aliases for complex types ────────────────────────────────
    // These exist so that callers using `CooMatrix<Complex<T>>` can call
    // `new_complex` / `push_complex` without ambiguity.  After the
    // ComplexScalar unification, `new` and `push` also work for complex types.

    /// Create an empty matrix (alias for `new`, kept for complex call-sites).
    pub fn new_complex(nrows: usize, ncols: usize) -> Self {
        Self::new(nrows, ncols)
    }

    /// Append an entry (alias for `push`, kept for complex call-sites).
    pub fn push_complex(&mut self, row: usize, col: usize, value: T) {
        self.push(row, col, value);
    }

    /// Number of rows (complex alias).
    pub fn nrows_c(&self) -> usize { self.nrows }
    /// Number of columns (complex alias).
    pub fn ncols_c(&self) -> usize { self.ncols }
    /// Number of stored entries (complex alias).
    pub fn nnz_c(&self) -> usize { self.rows.len() }
}
