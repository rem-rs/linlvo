#![allow(clippy::needless_range_loop)]
use crate::core::{operator::LinearOperator, scalar::{ComplexScalar, Scalar}, vector::DenseVec};
use crate::sparse::{coo::CooMatrix, csc::CscMatrix};

/// Compressed Sparse Row (CSR) matrix.
///
/// This is the primary sparse format used by all iterative solvers and
/// preconditioners in linger.
///
/// Layout:
/// - `row_ptr[i]..row_ptr[i+1]` indexes the entries belonging to row `i`.
/// - `col_idx[k]` and `values[k]` are the column index and value of entry `k`.
///
/// # Examples
/// ```
/// use linger::sparse::{CooMatrix, CsrMatrix};
///
/// let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
/// coo.push(0, 0, 2.0); coo.push(0, 1, -1.0);
/// coo.push(1, 0, -1.0); coo.push(1, 1, 2.0); coo.push(1, 2, -1.0);
/// coo.push(2, 1, -1.0); coo.push(2, 2, 2.0);
/// let csr = CsrMatrix::from_coo(&coo);
/// assert_eq!(csr.nnz(), 7);
/// ```
#[derive(Debug, Clone)]
pub struct CsrMatrix<T> {
    nrows:   usize,
    ncols:   usize,
    row_ptr: Vec<usize>, // length nrows + 1
    col_idx: Vec<usize>, // length nnz
    values:  Vec<T>,     // length nnz
}

impl<T: ComplexScalar> CsrMatrix<T> {
    // ─── Constructors ────────────────────────────────────────────────────────

    /// Build a CSR matrix from COO format.
    ///
    /// Entries are sorted by `(row, col)` and duplicate pairs are summed.
    pub fn from_coo(coo: &CooMatrix<T>) -> Self {
        let nrows = coo.nrows;
        let ncols = coo.ncols;
        let nnz_input = coo.rows.len();

        if nnz_input == 0 {
            return Self {
                nrows,
                ncols,
                row_ptr: vec![0; nrows + 1],
                col_idx: vec![],
                values:  vec![],
            };
        }

        // Sort entries by (row, col).
        let mut order: Vec<usize> = (0..nnz_input).collect();
        order.sort_unstable_by_key(|&i| (coo.rows[i], coo.cols[i]));

        // Merge duplicate (row, col) entries by summing their values.
        let mut merged: Vec<(usize, usize, T)> = Vec::with_capacity(nnz_input);
        for &i in &order {
            let r = coo.rows[i];
            let c = coo.cols[i];
            let v = coo.values[i];
            if let Some(last) = merged.last_mut() {
                if last.0 == r && last.1 == c {
                    last.2 += v;
                    continue;
                }
            }
            merged.push((r, c, v));
        }

        // Build row_ptr via per-row counts then prefix sum.
        let nnz = merged.len();
        let mut counts = vec![0usize; nrows];
        for &(r, _, _) in &merged {
            counts[r] += 1;
        }
        let mut row_ptr = vec![0usize; nrows + 1];
        for r in 0..nrows {
            row_ptr[r + 1] = row_ptr[r] + counts[r];
        }

        let mut col_idx = Vec::with_capacity(nnz);
        let mut values  = Vec::with_capacity(nnz);
        for (_, c, v) in merged {
            col_idx.push(c);
            values.push(v);
        }

        Self { nrows, ncols, row_ptr, col_idx, values }
    }

    /// Construct directly from raw CSR arrays.
    ///
    /// # Panics
    /// Panics if `row_ptr.len() != nrows + 1` or the arrays are inconsistent.
    pub fn from_raw(
        nrows:   usize,
        ncols:   usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values:  Vec<T>,
    ) -> Self {
        assert_eq!(row_ptr.len(), nrows + 1, "row_ptr must have nrows+1 entries");
        assert_eq!(col_idx.len(), values.len(), "col_idx and values must have equal length");
        assert_eq!(*row_ptr.last().unwrap(), col_idx.len(), "row_ptr.last() must equal nnz");
        // Bounds-check col_idx in debug builds so that spmv's get_unchecked is safe.
        #[cfg(debug_assertions)]
        for (k, &c) in col_idx.iter().enumerate() {
            assert!(c < ncols, "from_raw: col_idx[{k}] = {c} out of bounds (ncols = {ncols})");
        }
        Self { nrows, ncols, row_ptr, col_idx, values }
    }

    // ─── Dimensions / accessors ──────────────────────────────────────────────

    /// Number of rows.
    pub fn nrows(&self) -> usize { self.nrows }
    /// Number of columns.
    pub fn ncols(&self) -> usize { self.ncols }
    /// Number of stored non-zero entries.
    pub fn nnz(&self) -> usize { self.values.len() }

    /// Validate the CSR structure.
    ///
    /// Returns `Ok(())` if:
    /// - `row_ptr` starts at 0, is non-decreasing, and ends at `nnz`
    /// - every `col_idx` value is in `[0, ncols)`
    /// - within each row, `col_idx` values are **strictly increasing** (no duplicates, sorted)
    ///
    /// Returns `Err(String)` describing the first violation found.
    pub fn validate(&self) -> Result<(), String> {
        let nnz = self.values.len();
        // row_ptr checks
        if self.row_ptr.len() != self.nrows + 1 {
            return Err(format!(
                "row_ptr length {} ≠ nrows+1 = {}",
                self.row_ptr.len(), self.nrows + 1
            ));
        }
        if self.row_ptr[0] != 0 {
            return Err(format!("row_ptr[0] = {} ≠ 0", self.row_ptr[0]));
        }
        if self.row_ptr[self.nrows] != nnz {
            return Err(format!(
                "row_ptr[nrows] = {} ≠ nnz = {}",
                self.row_ptr[self.nrows], nnz
            ));
        }
        for i in 0..self.nrows {
            if self.row_ptr[i] > self.row_ptr[i + 1] {
                return Err(format!(
                    "row_ptr non-monotone at row {i}: row_ptr[{i}]={} > row_ptr[{}]={}",
                    self.row_ptr[i], i + 1, self.row_ptr[i + 1]
                ));
            }
        }
        // col_idx checks
        if self.col_idx.len() != nnz {
            return Err(format!("col_idx length {} ≠ nnz = {}", self.col_idx.len(), nnz));
        }
        for row in 0..self.nrows {
            let start = self.row_ptr[row];
            let end   = self.row_ptr[row + 1];
            let mut prev: Option<usize> = None;
            for idx in start..end {
                let col = self.col_idx[idx];
                if col >= self.ncols {
                    return Err(format!(
                        "col_idx[{idx}] = {col} out of bounds (ncols = {})",
                        self.ncols
                    ));
                }
                if let Some(p) = prev {
                    if col <= p {
                        return Err(format!(
                            "row {row}: col_idx not strictly increasing at positions {}, {} ({} ≥ {})",
                            idx - 1, idx, p, col
                        ));
                    }
                }
                prev = Some(col);
            }
        }
        Ok(())
    }

    /// Raw row-pointer array (length `nrows + 1`).
    pub fn row_ptr(&self) -> &[usize] { &self.row_ptr }
    /// Raw column-index array (length `nnz`).
    pub fn col_idx(&self) -> &[usize] { &self.col_idx }
    /// Raw value array (length `nnz`).
    pub fn values(&self) -> &[T] { &self.values }

    /// Iterate over all `(row, col, value)` triplets in row-major order.
    pub fn triplets(&self) -> impl Iterator<Item = (usize, usize, T)> + '_ {
        (0..self.nrows).flat_map(move |r| {
            let start = self.row_ptr[r];
            let end   = self.row_ptr[r + 1];
            self.col_idx[start..end]
                .iter()
                .zip(self.values[start..end].iter())
                .map(move |(&c, &v)| (r, c, v))
        })
    }

    // ─── Sparse matrix–vector products ───────────────────────────────────────

    /// Compute  `y ← A · x`  (overwrites `y`).
    ///
    /// # Panics
    /// Panics if `x.len() != ncols` or `y.len() != nrows`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols, "spmv: x length mismatch");
        assert_eq!(y.len(), self.nrows, "spmv: y length mismatch");
        let row_ptr = &self.row_ptr;
        let col_idx = &self.col_idx;
        let values = &self.values;
        for (i, yi) in y.iter_mut().enumerate() {
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            *yi = unsafe { csr_row_dot_unchecked(col_idx, values, x, start, end) };
        }
    }

    /// Compute  `y ← α·A·x + β·y`  (generalised SpMV, overwrites `y`).
    pub fn spmv_add(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) {
        assert_eq!(x.len(), self.ncols, "spmv_add: x length mismatch");
        assert_eq!(y.len(), self.nrows, "spmv_add: y length mismatch");
        let row_ptr = &self.row_ptr;
        let col_idx = &self.col_idx;
        let values = &self.values;
        for (i, yi) in y.iter_mut().enumerate() {
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            let sum = unsafe { csr_row_dot_unchecked(col_idx, values, x, start, end) };
            *yi = alpha * sum + beta * *yi;
        }
    }

    // ─── Structure operations ─────────────────────────────────────────────────

    /// Transpose: return the CSC representation of `Aᵀ`.
    ///
    /// For a CSR matrix A, the transposed CSC is obtained by swapping the roles
    /// of row_ptr↔col_ptr and row_idx↔col_idx — an O(nnz) operation.
    pub fn transpose(&self) -> CscMatrix<T> {
        // A^T as CSC: column j of A^T = row j of A.
        // So col_ptr = row_ptr, row_idx = col_idx (from A's CSR).
        CscMatrix::from_raw(
            self.ncols,             // nrows of A^T
            self.nrows,             // ncols of A^T
            self.row_ptr.clone(),   // col_ptr of CSC(A^T)
            self.col_idx.clone(),   // row_idx of CSC(A^T)
            self.values.clone(),
        )
    }

    /// Extract the main diagonal.
    ///
    /// Returns a vector of length `min(nrows, ncols)`.
    /// Missing diagonal entries are represented as zero.
    pub fn diag(&self) -> Vec<T> {
        let n = self.nrows.min(self.ncols);
        let mut d = vec![T::zero(); n];
        for i in 0..n {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                if self.col_idx[k] == i {
                    d[i] = self.values[k];
                    break;
                }
            }
        }
        d
    }

    /// Return `true` if the matrix is structurally symmetric
    /// (i.e. every `(i,j)` entry has a corresponding `(j,i)` entry).
    pub fn is_structurally_symmetric(&self) -> bool {
        if self.nrows != self.ncols {
            return false;
        }
        for (r, c, _) in self.triplets() {
            // Check that (c, r) exists.
            let row_start = self.row_ptr[c];
            let row_end   = self.row_ptr[c + 1];
            if !self.col_idx[row_start..row_end].contains(&r) {
                return false;
            }
        }
        true
    }

    /// Sparse matrix–matrix product: `C = self · B`.
    ///
    /// Uses a hash-map accumulator per row (correct but not cache-optimal).
    /// Suitable for setup-phase use where performance is secondary to correctness.
    pub fn matmat(&self, b: &CsrMatrix<T>) -> CsrMatrix<T> {
        assert_eq!(self.ncols, b.nrows, "matmat: inner dimensions must match");
        let m = self.nrows;
        let n = b.ncols;

        let mut c_rows: Vec<Vec<(usize, T)>> = vec![Vec::new(); m];
        let mut marks = vec![usize::MAX; n];
        let mut accum = vec![T::zero(); n];
        let mut touched_cols: Vec<usize> = Vec::new();

        for i in 0..m {
            touched_cols.clear();
            for ka in self.row_ptr[i]..self.row_ptr[i + 1] {
                let k   = self.col_idx[ka];
                let a_ik = self.values[ka];
                for kb in b.row_ptr[k]..b.row_ptr[k + 1] {
                    let j   = b.col_idx[kb];
                    let b_kj = b.values[kb];
                    if marks[j] != i {
                        marks[j] = i;
                        accum[j] = a_ik * b_kj;
                        touched_cols.push(j);
                    } else {
                        accum[j] += a_ik * b_kj;
                    }
                }
            }

            let mut row: Vec<(usize, T)> = Vec::with_capacity(touched_cols.len());
            for &j in &touched_cols {
                let value = accum[j];
                if value != T::zero() {
                    row.push((j, value));
                }
            }
            row.sort_unstable_by_key(|&(j, _)| j);
            c_rows[i] = row;
        }

        // Pack into CSR.
        let nnz: usize = c_rows.iter().map(|r| r.len()).sum();
        let mut row_ptr = vec![0usize; m + 1];
        let mut col_idx = Vec::with_capacity(nnz);
        let mut values  = Vec::with_capacity(nnz);
        for (i, row) in c_rows.iter().enumerate() {
            row_ptr[i + 1] = row_ptr[i] + row.len();
            for &(j, v) in row {
                col_idx.push(j);
                values.push(v);
            }
        }
        CsrMatrix { nrows: m, ncols: n, row_ptr, col_idx, values }
    }

    /// Compute `Aᵀ` as a `CsrMatrix` (rather than `CscMatrix`).
    ///
    /// Needed for Galerkin projection `Pᵀ A P`.
    pub fn transpose_csr(&self) -> CsrMatrix<T> {
        let m = self.nrows;
        let n = self.ncols;
        let nnz = self.values.len();

        // Count entries per column (= row of Aᵀ).
        let mut counts = vec![0usize; n];
        for &c in &self.col_idx {
            counts[c] += 1;
        }
        let mut row_ptr = vec![0usize; n + 1];
        for j in 0..n {
            row_ptr[j + 1] = row_ptr[j] + counts[j];
        }

        // Fill col_idx and values of Aᵀ.
        let mut col_idx = vec![0usize; nnz];
        let mut values  = vec![T::zero(); nnz];
        let mut cursor  = row_ptr[..n].to_vec(); // write positions per row

        for i in 0..m {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[k];
                let pos = cursor[j];
                col_idx[pos] = i;
                values[pos]  = self.values[k];
                cursor[j] += 1;
            }
        }

        CsrMatrix { nrows: n, ncols: m, row_ptr, col_idx, values }
    }
}

#[inline(always)]
unsafe fn csr_row_dot_unchecked<T: ComplexScalar>(
    col_idx: &[usize],
    values: &[T],
    x: &[T],
    start: usize,
    end: usize,
) -> T {
    match end - start {
        0 => T::zero(),
        1 => *values.get_unchecked(start) * *x.get_unchecked(*col_idx.get_unchecked(start)),
        2 => {
            let c0 = *col_idx.get_unchecked(start);
            let c1 = *col_idx.get_unchecked(start + 1);
            *values.get_unchecked(start) * *x.get_unchecked(c0)
                + *values.get_unchecked(start + 1) * *x.get_unchecked(c1)
        }
        3 => {
            let c0 = *col_idx.get_unchecked(start);
            let c1 = *col_idx.get_unchecked(start + 1);
            let c2 = *col_idx.get_unchecked(start + 2);
            *values.get_unchecked(start) * *x.get_unchecked(c0)
                + *values.get_unchecked(start + 1) * *x.get_unchecked(c1)
                + *values.get_unchecked(start + 2) * *x.get_unchecked(c2)
        }
        4 => {
            let c0 = *col_idx.get_unchecked(start);
            let c1 = *col_idx.get_unchecked(start + 1);
            let c2 = *col_idx.get_unchecked(start + 2);
            let c3 = *col_idx.get_unchecked(start + 3);
            *values.get_unchecked(start) * *x.get_unchecked(c0)
                + *values.get_unchecked(start + 1) * *x.get_unchecked(c1)
                + *values.get_unchecked(start + 2) * *x.get_unchecked(c2)
                + *values.get_unchecked(start + 3) * *x.get_unchecked(c3)
        }
        _ => {
            // For rows with ≥5 NNZ use SIMD-accelerated gather-dot if available.
            unsafe { crate::simd::simd_row_dot(col_idx, values, x, start, end) }
        }
    }
}

// ─── LinearOperator impl ──────────────────────────────────────────────────────

impl<T: ComplexScalar> LinearOperator for CsrMatrix<T> {
    type Vector = DenseVec<T>;

    #[inline]
    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        self.spmv(x.as_slice(), y.as_mut_slice());
    }

    fn nrows(&self) -> usize { self.nrows }
    fn ncols(&self) -> usize { self.ncols }
}

impl<T: ComplexScalar> crate::core::operator::TransposeOperator for CsrMatrix<T> {
    fn apply_transpose(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        // y = Aᵀ x : scatter each row contribution
        let ys = y.as_mut_slice();
        for v in ys.iter_mut() { *v = T::zero(); }
        let xs = x.as_slice();
        for i in 0..self.nrows {
            let xi = xs[i];
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[k];
                ys[j] += self.values[k] * xi;
            }
        }
    }
}

// `from_complex_coo` is now available as `from_coo` for all ComplexScalar types.
// Kept as a convenience alias for backward compatibility with call sites that
// use `CsrMatrix::from_complex_coo(&coo)` on `CooMatrix<Complex<T>>`.
//
// Note: `from_coo` in `impl<T: ComplexScalar> CsrMatrix<T>` already handles
// complex element types identically.
