//! Block Sparse Row (BSR) matrix format.
//!
//! BSR stores a sparse matrix as a grid of dense `r×c` blocks, where all
//! blocks in a row-block share the same block-row index.  This improves cache
//! utilisation for block-structured systems (e.g., multi-DOF FEA nodes).
//!
//! Layout (block size r × c):
//! - `block_row_ptr[I]..block_row_ptr[I+1]`: range of stored blocks in block-row I.
//! - `block_col_idx[K]`: block-column index of the K-th stored block.
//! - `block_vals[K]`: the r×c block stored **row-major** in a flat `Vec<T>` of
//!   length `r * c`.
//!
//! The scalar dimension is `nblock_rows * r` × `nblock_cols * c`.
//!
//! **Analogs**
//!   PETSc: `MATBAIJ` (block AIJ = block CSR)
//!   HYPRE: `HYPRE_ParCSRMatrix` with block extensions

use crate::core::scalar::Scalar;

/// Block Sparse Row matrix with uniform block size `r × c`.
#[derive(Debug, Clone)]
pub struct BsrMatrix<T> {
    /// Number of block-rows.
    pub nblock_rows: usize,
    /// Number of block-columns.
    pub nblock_cols: usize,
    /// Block height.
    pub block_rows: usize,
    /// Block width.
    pub block_cols: usize,
    /// Block-row pointer (length `nblock_rows + 1`).
    pub block_row_ptr: Vec<usize>,
    /// Block-column indices (length `n_blocks`).
    pub block_col_idx: Vec<usize>,
    /// Dense block values, row-major, each block has `block_rows * block_cols` elements.
    /// Total length: `n_blocks * block_rows * block_cols`.
    pub block_vals: Vec<T>,
}

impl<T: Scalar> BsrMatrix<T> {
    /// Construct a BSR matrix from raw arrays.
    ///
    /// # Panics
    /// Panics if array lengths are inconsistent.
    pub fn from_raw(
        nblock_rows:   usize,
        nblock_cols:   usize,
        block_rows:    usize,
        block_cols:    usize,
        block_row_ptr: Vec<usize>,
        block_col_idx: Vec<usize>,
        block_vals:    Vec<T>,
    ) -> Self {
        let n_blocks = block_col_idx.len();
        assert_eq!(block_row_ptr.len(), nblock_rows + 1);
        assert_eq!(*block_row_ptr.last().unwrap(), n_blocks);
        assert_eq!(block_vals.len(), n_blocks * block_rows * block_cols);
        BsrMatrix { nblock_rows, nblock_cols, block_rows, block_cols,
                    block_row_ptr, block_col_idx, block_vals }
    }

    /// Total number of scalar rows.
    pub fn nrows(&self) -> usize { self.nblock_rows * self.block_rows }
    /// Total number of scalar columns.
    pub fn ncols(&self) -> usize { self.nblock_cols * self.block_cols }
    /// Number of stored blocks.
    pub fn n_blocks(&self) -> usize { self.block_col_idx.len() }
    /// Number of stored non-zeros (scalar entries, counting zero padding in blocks).
    pub fn nnz_stored(&self) -> usize { self.n_blocks() * self.block_rows * self.block_cols }

    // ─── SpMV ────────────────────────────────────────────────────────────────

    /// Compute `y ← A · x` (block SpMV).
    ///
    /// # Panics
    /// Panics if `x.len() != ncols` or `y.len() != nrows`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        let nr = self.nrows();
        let nc = self.ncols();
        assert_eq!(x.len(), nc, "BsrMatrix::spmv: x length mismatch");
        assert_eq!(y.len(), nr, "BsrMatrix::spmv: y length mismatch");

        let r  = self.block_rows;
        let c  = self.block_cols;
        let bs = r * c; // block size in scalars

        for y in y.iter_mut() { *y = T::zero(); }

        for I in 0..self.nblock_rows {
            let row_start = I * r; // first scalar row of block-row I
            for k in self.block_row_ptr[I]..self.block_row_ptr[I + 1] {
                let J         = self.block_col_idx[k];
                let col_start = J * c; // first scalar col of block-col J
                let blk       = &self.block_vals[k * bs..(k + 1) * bs];

                // Dense r×c multiply: y[row_start..+r] += blk * x[col_start..+c]
                for bi in 0..r {
                    let mut sum = T::zero();
                    for bj in 0..c {
                        sum += blk[bi * c + bj] * x[col_start + bj];
                    }
                    y[row_start + bi] += sum;
                }
            }
        }
    }

    /// Parallel block SpMV using Rayon (falls back to serial without feature).
    pub fn spmv_parallel(&self, x: &[T], y: &mut [T])
    where
        T: Send + Sync,
    {
        let nr = self.nrows();
        let nc = self.ncols();
        assert_eq!(x.len(), nc);
        assert_eq!(y.len(), nr);

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let r  = self.block_rows;
            let c  = self.block_cols;
            let bs = r * c;
            let rp = &self.block_row_ptr;
            let ci = &self.block_col_idx;
            let bv = &self.block_vals;

            // Each block-row is independent.
            let results: Vec<Vec<T>> = (0..self.nblock_rows)
                .into_par_iter()
                .map(|I| {
                    let mut row_out = vec![T::zero(); r];
                    for k in rp[I]..rp[I + 1] {
                        let J         = ci[k];
                        let col_start = J * c;
                        let blk       = &bv[k * bs..(k + 1) * bs];
                        for bi in 0..r {
                            let mut sum = T::zero();
                            for bj in 0..c {
                                sum += blk[bi * c + bj] * x[col_start + bj];
                            }
                            row_out[bi] += sum;
                        }
                    }
                    row_out
                })
                .collect();

            for (I, row_out) in results.iter().enumerate() {
                y[I * r..(I + 1) * r].copy_from_slice(row_out);
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            self.spmv(x, y);
        }
    }

    // ─── Conversions ─────────────────────────────────────────────────────────

    /// Convert to CSR format (for interop with solvers).
    pub fn to_csr(&self) -> crate::sparse::CsrMatrix<T> {
        use crate::sparse::CooMatrix;
        let r  = self.block_rows;
        let c  = self.block_cols;
        let bs = r * c;
        let mut coo = CooMatrix::with_capacity(self.nrows(), self.ncols(), self.nnz_stored());
        for I in 0..self.nblock_rows {
            for k in self.block_row_ptr[I]..self.block_row_ptr[I + 1] {
                let J   = self.block_col_idx[k];
                let blk = &self.block_vals[k * bs..(k + 1) * bs];
                for bi in 0..r {
                    for bj in 0..c {
                        let v = blk[bi * c + bj];
                        if v != T::zero() {
                            coo.push(I * r + bi, J * c + bj, v);
                        }
                    }
                }
            }
        }
        crate::sparse::CsrMatrix::from_coo(&coo)
    }
}

// ─── Assembly helper ──────────────────────────────────────────────────────────

/// Builder for BSR matrices: accumulate block entries then finalize.
pub struct BsrBuilder<T> {
    nblock_rows: usize,
    nblock_cols: usize,
    block_rows:  usize,
    block_cols:  usize,
    /// (block_row, block_col, flat_block_values[r*c])
    entries: Vec<(usize, usize, Vec<T>)>,
}

impl<T: Scalar> BsrBuilder<T> {
    pub fn new(nblock_rows: usize, nblock_cols: usize, block_rows: usize, block_cols: usize) -> Self {
        BsrBuilder { nblock_rows, nblock_cols, block_rows, block_cols, entries: Vec::new() }
    }

    /// Add a dense `block_rows × block_cols` block at position (I, J).
    /// `vals` must be row-major with length `block_rows * block_cols`.
    pub fn push_block(&mut self, I: usize, J: usize, vals: Vec<T>) {
        assert_eq!(vals.len(), self.block_rows * self.block_cols);
        self.entries.push((I, J, vals));
    }

    /// Finalise into a `BsrMatrix`.  Duplicate (I, J) entries are summed.
    pub fn build(mut self) -> BsrMatrix<T> {
        let r  = self.block_rows;
        let c  = self.block_cols;
        let bs = r * c;

        // Sort by (block_row, block_col).
        self.entries.sort_unstable_by_key(|&(I, J, _)| (I, J));

        // Merge duplicate blocks by summing.
        let mut merged: Vec<(usize, usize, Vec<T>)> = Vec::new();
        for (I, J, vals) in self.entries {
            if let Some(last) = merged.last_mut() {
                if last.0 == I && last.1 == J {
                    for (a, b) in last.2.iter_mut().zip(vals.iter()) {
                        *a += *b;
                    }
                    continue;
                }
            }
            merged.push((I, J, vals));
        }

        // Pack into BSR arrays.
        let mut block_row_ptr = vec![0usize; self.nblock_rows + 1];
        let mut block_col_idx = Vec::with_capacity(merged.len());
        let mut block_vals    = Vec::with_capacity(merged.len() * bs);

        for &(I, J, ref v) in &merged {
            block_row_ptr[I + 1] += 1; // count
            block_col_idx.push(J);
            block_vals.extend_from_slice(v);
        }
        // Prefix sum for row_ptr.
        for i in 0..self.nblock_rows {
            block_row_ptr[i + 1] += block_row_ptr[i];
        }

        BsrMatrix { nblock_rows: self.nblock_rows, nblock_cols: self.nblock_cols,
                    block_rows: r, block_cols: c,
                    block_row_ptr, block_col_idx, block_vals }
    }
}
