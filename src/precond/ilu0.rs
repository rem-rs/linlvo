//! Incomplete LU factorisation with zero fill-in — ILU(0).
//!
//! Stores the L and U factors in-place in a single CSR structure where:
//! - entries with col < row belong to L (unit lower triangular; diagonal is
//!   implicitly 1 and not stored)
//! - entries with col >= row belong to U (including the diagonal)
//!
//! The sparsity pattern is fixed to that of the original matrix.
//!
//! **Analogs**
//!   PETSc: `PCILU` with `PCFactorSetLevels(pc, 0)`
//!   HYPRE: `HYPRE_EuclidCreate` with level 0

#![allow(clippy::needless_range_loop)]
use crate::core::{
    error::SolverError, preconditioner::Preconditioner, scalar::Scalar, vector::DenseVec,
};
use crate::sparse::CsrMatrix;

/// ILU(0) preconditioner.
///
/// After `from_csr`, the stored values hold the LU factors in-place.
/// `apply_precond` solves the two triangular systems:
///   1. Forward:  L z = x   (unit lower triangular, diagonal = 1)
///   2. Backward: U y = z   (upper triangular)
pub struct Ilu0Precond<T> {
    nrows: usize,
    /// Combined LU values stored in the original CSR sparsity pattern.
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    lu_val: Vec<T>,
    /// Position of the diagonal entry in each row (index into lu_val / col_idx).
    diag_pos: Vec<usize>,
}

impl<T: Scalar> Ilu0Precond<T> {
    /// Compute the ILU(0) factorisation of `mat`.
    ///
    /// Returns `Err(SolverError::SingularMatrix)` if a zero pivot is encountered.
    pub fn from_csr(mat: &CsrMatrix<T>) -> Result<Self, SolverError> {
        let n = mat.nrows();
        if mat.ncols() != n {
            return Err(SolverError::PrecondSetupFailed {
                reason: "ILU(0) requires a square matrix".into(),
            });
        }

        let row_ptr = mat.row_ptr().to_vec();
        let col_idx = mat.col_idx().to_vec();
        let mut lu_val = mat.values().to_vec();

        // Locate diagonal position for each row.
        let mut diag_pos = vec![0usize; n];
        for i in 0..n {
            let mut found = false;
            for k in row_ptr[i]..row_ptr[i + 1] {
                if col_idx[k] == i {
                    diag_pos[i] = k;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(SolverError::PrecondSetupFailed {
                    reason: format!("ILU(0): missing diagonal entry at row {i}"),
                });
            }
        }

        // ILU(0) factorisation in-place (Saad, "Iterative Methods", Alg. 10.4).
        let tol = T::machine_epsilon() * T::from_f64(1e6);
        for i in 1..n {
            // For each entry (i, k) in row i where k < i (lower-triangle):
            let i_start = row_ptr[i];
            let i_end = row_ptr[i + 1];

            for pos_ik in i_start..i_end {
                let k = col_idx[pos_ik];
                if k >= i {
                    break; // entries are sorted; no more lower-triangular entries
                }
                let u_kk = lu_val[diag_pos[k]];
                if u_kk.abs() < tol {
                    return Err(SolverError::SingularMatrix { row: k });
                }
                let factor = lu_val[pos_ik] / u_kk;
                lu_val[pos_ik] = factor; // store multiplier in L part

                // Update row i entries (j > k) that exist in sparsity pattern.
                // We need to walk row k's upper part and row i simultaneously.
                let k_diag = diag_pos[k];
                let k_end = row_ptr[k + 1];

                for pos_kj in (k_diag + 1)..k_end {
                    let j = col_idx[pos_kj];
                    // Find (i, j) in row i's sparsity pattern.
                    if let Some(pos_ij) = find_entry(&col_idx, row_ptr[i], i_end, j) {
                        let v_kj = lu_val[pos_kj];
                        lu_val[pos_ij] -= factor * v_kj;
                    }
                    // If (i,j) not in pattern, ILU(0) drops this fill-in.
                }
            }
        }

        Ok(Ilu0Precond { nrows: n, row_ptr, col_idx, lu_val, diag_pos })
    }
}

/// Binary search for column `j` within the slice `col_idx[start..end]`.
/// Returns `Some(absolute_index)` if found, `None` otherwise.
#[inline]
fn find_entry(col_idx: &[usize], start: usize, end: usize, j: usize) -> Option<usize> {
    // Linear scan — pattern is usually small per row for sparse FE matrices.
    for pos in start..end {
        match col_idx[pos].cmp(&j) {
            std::cmp::Ordering::Equal => return Some(pos),
            std::cmp::Ordering::Greater => return None,
            std::cmp::Ordering::Less => {}
        }
    }
    None
}

impl<T: Scalar> Preconditioner for Ilu0Precond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n = self.nrows;
        let xs = x.as_slice();
        let ys = y.as_mut_slice();

        // Phase 1 — forward solve  L z = x  (z stored in ys temporarily)
        // L is unit lower triangular: l_ii = 1, multipliers stored below diagonal.
        for i in 0..n {
            let mut sum = xs[i];
            for k in self.row_ptr[i]..self.diag_pos[i] {
                // col_idx[k] < i, so ys[col_idx[k]] is already computed
                sum -= self.lu_val[k] * ys[self.col_idx[k]];
            }
            ys[i] = sum; // no division because l_ii = 1
        }

        // Phase 2 — backward solve  U y = z  (z in ys, overwrite in-place)
        for i in (0..n).rev() {
            let mut sum = ys[i];
            for k in (self.diag_pos[i] + 1)..self.row_ptr[i + 1] {
                sum -= self.lu_val[k] * ys[self.col_idx[k]];
            }
            ys[i] = sum / self.lu_val[self.diag_pos[i]];
        }
    }
}
