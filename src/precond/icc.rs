//! ICC(0) — Incomplete Cholesky Factorisation with zero fill-in.
//!
//! Computes a lower-triangular factor L such that A ≈ L Lᵀ, retaining only
//! the nonzero pattern of the **lower triangular** part of A.
//!
//! Suitable for **symmetric positive definite** systems only.  The diagonal
//! entries of L are √(a_{ii} − ∑_{k<i} l²_{ik}), so a non-positive value
//! indicates the matrix is not SPD and is reported as an error.
//!
//! **Algorithm** (row-by-row left-looking): Saad §10.3.2.
//!
//! Apply:
//!   1. Forward:   L z = x   (lower triangular, non-unit diagonal)
//!   2. Backward:  Lᵀ y = z  (row-descending accumulation trick)
//!
//! **Analogs**
//!   PETSc: `PCICC` with `PCFactorSetLevels(pc, 0)`
//!   HYPRE: `HYPRE_BoomerAMGSetRelaxType` (Cholesky smoother)

use crate::core::{
    error::SolverError, preconditioner::Preconditioner, scalar::Scalar, vector::DenseVec,
};
use crate::sparse::CsrMatrix;

/// ICC(0) preconditioner for symmetric positive-definite matrices.
pub struct Icc0Precond<T> {
    nrows:    usize,
    /// Lower-triangular part of L stored in CSR (col ≤ row, including diagonal).
    row_ptr:  Vec<usize>,
    col_idx:  Vec<usize>,
    val:      Vec<T>,
    diag_pos: Vec<usize>,
}

impl<T: Scalar> Icc0Precond<T> {
    /// Compute ICC(0) factorisation of `mat`.
    ///
    /// Only the lower-triangular part of `mat` (including diagonal) is used.
    pub fn from_csr(mat: &CsrMatrix<T>) -> Result<Self, SolverError> {
        let n = mat.nrows();
        if mat.ncols() != n {
            return Err(SolverError::PrecondSetupFailed {
                reason: "ICC(0) requires a square matrix".into(),
            });
        }

        let rp_orig = mat.row_ptr();
        let ci_orig = mat.col_idx();
        let vs_orig = mat.values();

        // ── Extract lower-triangular part into working CSR ────────────────
        let mut rp   = vec![0usize; n + 1];
        let mut ci   = Vec::new();
        let mut val  = Vec::new();
        let mut diag_pos = vec![0usize; n];

        for i in 0..n {
            let mut found = false;
            for k in rp_orig[i]..rp_orig[i + 1] {
                let j = ci_orig[k];
                if j > i {
                    break; // CSR columns are sorted; no more lower entries
                }
                if j == i {
                    diag_pos[i] = ci.len();
                    found = true;
                }
                ci.push(j);
                val.push(vs_orig[k]);
            }
            if !found {
                return Err(SolverError::PrecondSetupFailed {
                    reason: format!("ICC(0): missing diagonal entry at row {i}"),
                });
            }
            rp[i + 1] = ci.len();
        }

        // ── Left-looking ICC(0) factorisation (row-by-row) ───────────────
        // For each row i and each k < i with val[row_i, k] ≠ 0:
        //   val[i, k] -= sum_{p < k, shared} val[i, p] * val[k, p]
        //   val[i, k] /= val[k, k]
        // Then:
        //   val[i, i] = sqrt(val[i, i] - sum_{k < i} val[i, k]²)
        let eps = T::machine_epsilon() * T::from_f64(1e6);

        for i in 0..n {
            // Process each lower-triangle entry (i, k) with k < i
            let i_start = rp[i];
            let i_diag  = diag_pos[i];

            for pos_ik in i_start..i_diag {
                let k = ci[pos_ik];

                // Subtract contributions from columns p < k shared by row i and row k
                {
                    let mut pos_i = i_start;
                    let mut pos_k = rp[k];
                    let k_diag   = diag_pos[k];
                    while pos_i < pos_ik && pos_k < k_diag {
                        let ci_i = ci[pos_i];
                        let ci_k = ci[pos_k];
                        match ci_i.cmp(&ci_k) {
                            std::cmp::Ordering::Less    => { pos_i += 1; }
                            std::cmp::Ordering::Greater => { pos_k += 1; }
                            std::cmp::Ordering::Equal   => {
                                let tmp = val[pos_i] * val[pos_k];
                                val[pos_ik] -= tmp;
                                pos_i += 1; pos_k += 1;
                            }
                        }
                    }
                }

                // Divide by diagonal l[k, k]
                let dkk = val[diag_pos[k]];
                if dkk.abs() < eps {
                    return Err(SolverError::SingularMatrix { row: k });
                }
                val[pos_ik] /= dkk;
            }

            // Update and sqrt-ify diagonal
            let mut d = val[i_diag];
            for pos_ik in i_start..i_diag {
                let l_ik = val[pos_ik];
                d -= l_ik * l_ik;
            }
            if d <= T::zero() {
                return Err(SolverError::NumericalBreakdown {
                    detail: format!("ICC(0): non-positive definite pivot at row {i} (d={d:?})"),
                });
            }
            val[i_diag] = d.sqrt();
        }

        Ok(Icc0Precond { nrows: n, row_ptr: rp, col_idx: ci, val, diag_pos })
    }
}

impl<T: Scalar> Preconditioner for Icc0Precond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n = self.nrows;
        let xs = x.as_slice();
        let ys = y.as_mut_slice();

        // Phase 1 — forward solve  L z = x  (z stored in ys)
        // z[i] = (x[i] - ∑_{k<i} l[i,k] z[k]) / l[i,i]
        for i in 0..n {
            let mut s = xs[i];
            for pos in self.row_ptr[i]..self.diag_pos[i] {
                s -= self.val[pos] * ys[self.col_idx[pos]];
            }
            ys[i] = s / self.val[self.diag_pos[i]];
        }

        // Phase 2 — backward solve  Lᵀ y = z  (row-descending accumulation)
        // Process row k from n−1 down to 0:
        //   ys[k] /= l[k,k]                      → y[k] numerator divided
        //   ys[j] -= l[k,j] * ys[k]  for j < k  → propagate contribution
        for k in (0..n).rev() {
            ys[k] /= self.val[self.diag_pos[k]];
            for pos in self.row_ptr[k]..self.diag_pos[k] {
                let j = self.col_idx[pos];
                ys[j] -= self.val[pos] * ys[k];
            }
        }
    }
}
