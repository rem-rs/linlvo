//! AIR (Approximate Ideal Restriction) baseline utilities.
//!
//! This module provides a lightweight AIR-compatible restriction builder that
//! can be used for nonsymmetric AMG hierarchies.
//!
//! Baseline formula:
//! - C/F splitting from RS coarsening.
//! - Restriction rows are indexed by coarse points.
//! - For coarse point `c`:
//!   - `R[c, c] = 1`
//!   - for fine neighbours `f`: `R[c, f] = -A[c, f] / A[f, f]`
//!
//! This corresponds to a diagonal approximation of `A_ff^{-1}`.

use crate::amg::coarsen_rs::{coarse_index_map, NodeType};
use crate::core::scalar::Scalar;
use crate::sparse::CsrMatrix;

/// Build a baseline AIR restriction operator with diagonal `A_ff^{-1}`.
///
/// Returns a matrix `R` of shape `(n_coarse, n_fine)`.
pub fn air_restriction_diag<T: Scalar>(
    a: &CsrMatrix<T>,
    status: &[NodeType],
) -> CsrMatrix<T> {
    let n = a.nrows();
    let (nc, c_map) = coarse_index_map(status);
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vs = a.values();
    let diag = a.diag();

    let mut rows: Vec<Vec<(usize, T)>> = vec![Vec::new(); nc];

    for c in 0..n {
        if status[c] != NodeType::Coarse {
            continue;
        }

        let rc = c_map[c];
        let mut row: Vec<(usize, T)> = Vec::new();
        row.push((c, T::one()));

        for k in rp[c]..rp[c + 1] {
            let j = ci[k];
            if j == c {
                continue;
            }
            if status[j] == NodeType::Fine || status[j] == NodeType::Undecided {
                let dff = diag[j];
                if dff.abs() > T::machine_epsilon() {
                    row.push((j, -vs[k] / dff));
                }
            }
        }

        row.sort_unstable_by_key(|&(col, _)| col);
        rows[rc] = row;
    }

    pack_csr(nc, n, rows)
}

fn pack_csr<T: Scalar>(
    nrows: usize,
    ncols: usize,
    rows: Vec<Vec<(usize, T)>>,
) -> CsrMatrix<T> {
    let nnz: usize = rows.iter().map(|r| r.len()).sum();
    let mut row_ptr = vec![0usize; nrows + 1];
    let mut col_idx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    for (i, row) in rows.iter().enumerate() {
        row_ptr[i + 1] = row_ptr[i] + row.len();
        for &(j, v) in row {
            col_idx.push(j);
            values.push(v);
        }
    }
    CsrMatrix::from_raw(nrows, ncols, row_ptr, col_idx, values)
}
