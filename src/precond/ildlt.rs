//! Incomplete LDLᵀ factorisation with zero fill-in — ILDLᵀ(0).
//!
//! Computes an approximate factorisation `A ≈ L D Lᵀ` where:
//! - `L` is unit lower-triangular with the same sparsity pattern as the lower
//!   triangle of `A` (no fill-in beyond what is in `A`)
//! - `D` is a diagonal matrix (may have negative entries — handles indefinite A)
//!
//! Unlike ICC(0), ILDLᵀ(0) works for **symmetric indefinite** matrices (as long
//! as no zero pivots appear in the incomplete factorisation).  Common use cases:
//! saddle-point problems, symmetric discretisations with negative eigenvalues.
//!
//! ## Algorithm (row-by-row left-looking)
//!
//! For each row i (0-indexed):
//!
//! 1. Copy row i of A's lower-triangle into a working vector.
//! 2. For each k < i with L[i,k] != 0:
//!    - Form the multiplier:  L[i,k] /= D[k]
//!    - Subtract from remaining lower entries j > k in same row:
//!      val[i,j] -= L[i,k] * D[k] * L[j,k]   (only for j in sparsity pattern)
//! 3. Set D[i] = working_val[i,i].
//!
//! The symmetry means L[j,k] = A[j,k] after processing row j < i.
//!
//! ## Solve
//!
//! Given `A x ≈ L D Lᵀ x = b`:
//! 1. Forward:   `L z = b`    (unit lower-triangular, diagonal = 1)
//! 2. Diagonal:  `D w = z`    (w[i] = z[i] / d[i])
//! 3. Backward:  `Lᵀ x = w`  (upper-triangular transpose of L)
//!
//! ## Reference
//!
//! Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
//! SIAM. §10.3.3 (Incomplete LDLᵀ factorisation).
//!
//! Benzi, M., & Wathen, A. J. (2008). Some preconditioning techniques for
//! saddle point problems. *Model Order Reduction: Theory, Research Aspects and
//! Applications*, 195-211.

#![allow(clippy::needless_range_loop)]

use crate::core::{
    error::SolverError, preconditioner::Preconditioner, scalar::{ComplexScalar, Scalar}, vector::DenseVec,
};
use crate::sparse::CsrMatrix;

// ─── Public struct ────────────────────────────────────────────────────────────

/// ILDLᵀ(0) preconditioner for symmetric (possibly indefinite) matrices.
///
/// The sparsity pattern is fixed to the lower-triangular part of the input
/// matrix (zero fill-in). Compute with [`IldltPrecond::from_csr`].
///
/// # Example
/// ```text
/// use linger::precond::IldltPrecond;
/// let precond = IldltPrecond::<f64>::from_csr(&a)?;
/// cg.solve(&a, Some(&precond), &b, &mut x, &params)?;
/// ```
pub struct IldltPrecond<T> {
    nrows: usize,
    /// CSR storage for the lower-triangular L factor (unit diagonal not stored).
    /// The pattern is exactly the lower-triangular pattern of the input matrix.
    row_ptr:  Vec<usize>,
    col_idx:  Vec<usize>,
    l_val:    Vec<T>,
    /// Diagonal factor D[i].
    d:        Vec<T>,
}

impl<T: ComplexScalar> IldltPrecond<T> {
    /// Compute the ILDLᵀ(0) factorisation of `mat`.
    ///
    /// Only the lower-triangular part of `mat` (col ≤ row, including diagonal)
    /// is accessed.  The matrix must be symmetric — only the lower triangle is
    /// read.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::SingularMatrix`] if a zero (or near-zero) pivot
    /// D[k] is encountered during factorisation, and
    /// [`SolverError::PrecondSetupFailed`] if the matrix is not square or
    /// missing a diagonal entry.
    pub fn from_csr(mat: &CsrMatrix<T>) -> Result<Self, SolverError> {
        let n = mat.nrows();
        if mat.ncols() != n {
            return Err(SolverError::PrecondSetupFailed {
                reason: "ILDLt(0) requires a square matrix".into(),
            });
        }

        let rp_orig = mat.row_ptr();
        let ci_orig = mat.col_idx();
        let vs_orig = mat.values();

        // ── Extract lower-triangular pattern and values ───────────────────────
        // rp / ci / l_val will hold L (unit lower-tri, diagonal not stored).
        // We keep the diagonal separately in `d`.
        let mut rp       = vec![0usize; n + 1];
        let mut ci       = Vec::<usize>::new();
        let mut l_val    = Vec::<T>::new();
        let mut d        = vec![T::zero(); n];

        for i in 0..n {
            let mut found_diag = false;
            for k in rp_orig[i]..rp_orig[i + 1] {
                let j = ci_orig[k];
                if j > i { break; } // CSR col_idx are sorted; past lower triangle
                if j == i {
                    d[i] = vs_orig[k];      // D[i] ← A[i,i] (updated below)
                    found_diag = true;
                } else {
                    // Off-diagonal lower entry.
                    ci.push(j);
                    l_val.push(vs_orig[k]);
                }
            }
            if !found_diag {
                return Err(SolverError::PrecondSetupFailed {
                    reason: format!("ILDLt(0): missing diagonal at row {i}"),
                });
            }
            rp[i + 1] = ci.len();
        }

        // ── ILDLᵀ(0) factorisation (left-looking, row-by-row) ────────────────
        //
        // We need fast random access: for entry (j, k) in row j, find it quickly.
        // Since we only need within-row lookup, build a per-row hash: row_map[i]
        // maps column k → position in l_val.
        // For small matrices this is fine; for large sparse, the linear scan is
        // acceptable since |row| ≤ nnz/n on average.
        //
        // Algorithm (Saad §10.3.3):
        //   For i = 0..n-1:
        //     for k in non-zeros of L[i, 0..i-1] (i.e., rp[i]..rp[i+1]):
        //       let l_ik_old = l_val at position (i,k)  (from A before update)
        //       l_ik = l_ik_old / D[k]
        //       update l_val at (i,k): l_val[pos] = l_ik
        //       D[i] -= l_ik * D[k] * l_ik_old
        //       for each (i, j) with j > k and j != i (off-diag in row i):
        //         if (j, k) exists in L (as l_jk):
        //           l_val[pos(i,j)] -= l_ik * D[k] * l_jk

        let tol = T::machine_epsilon() * <T::Real as Scalar>::from_f64(1e6);

        for i in 0..n {
            // Scan lower-triangular entries of row i: columns k < i.
            let i_start = rp[i];
            let i_end   = rp[i + 1];

            for pos_ik in i_start..i_end {
                let k = ci[pos_ik];
                // k < i always (lower-triangle storage).

                let d_k = d[k];
                if d_k.abs() < tol {
                    return Err(SolverError::SingularMatrix { row: k });
                }

                // Multiplier: l_ik_orig is the current value (starts as A[i,k]).
                let l_ik_orig = l_val[pos_ik];
                let l_ik      = l_ik_orig / d_k;
                l_val[pos_ik] = l_ik; // store the multiplier in-place

                // Update D[i]: D[i] -= l_ik * D[k] * l_ik_orig.
                d[i] -= l_ik * d_k * l_ik_orig;

                // Update remaining off-diagonal entries in row i (columns j > k).
                // For each position pos_ij in row i after pos_ik:
                for pos_ij in (pos_ik + 1)..i_end {
                    let j = ci[pos_ij];
                    // j > k (since col_idx are sorted within a row).
                    // We need L[j, k] = l_val at position (j, k) in row j.
                    // Use linear scan within row j.
                    if let Some(l_jk) = find_l_entry(&ci, &l_val, rp[j], rp[j + 1], k) {
                        l_val[pos_ij] -= l_ik * d_k * l_jk;
                    }
                }
            }

            // At this point D[i] holds the pivot; check for breakdown.
            if d[i].abs() < tol {
                return Err(SolverError::SingularMatrix { row: i });
            }
        }

        Ok(Self { nrows: n, row_ptr: rp, col_idx: ci, l_val, d })
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize { self.nrows }

    /// Diagonal D factors.
    pub fn d(&self) -> &[T] { &self.d }
}

// ─── Preconditioner impl ─────────────────────────────────────────────────────

impl<T: ComplexScalar> Preconditioner for IldltPrecond<T> {
    type Vector = DenseVec<T>;

    /// Apply M⁻¹ x = (L D Lᵀ)⁻¹ x:
    /// 1. Forward solve  L z = x   (unit lower triangular)
    /// 2. Diagonal solve w = D⁻¹ z
    /// 3. Backward solve Lᵀ y = w  (upper-triangular transpose of L)
    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n = self.nrows;
        let xs = x.as_slice();
        let ys = y.as_mut_slice();

        // ── Step 1: Forward solve L z = x (unit lower triangular) ────────────
        // z[i] = x[i] - sum_{j < i, L[i,j] != 0} L[i,j] * z[j]
        ys.copy_from_slice(xs);
        for i in 0..n {
            for pos in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[pos]; // j < i (lower triangle)
                let lij = self.l_val[pos];
                let yj  = ys[j]; // j < i, already solved
                ys[i] -= lij * yj;
            }
        }

        // ── Step 2: Diagonal solve w[i] = z[i] / D[i] ────────────────────────
        for i in 0..n {
            if self.d[i] != T::zero() {
                ys[i] /= self.d[i];
            }
        }

        // ── Step 3: Backward solve Lᵀ x = w ─────────────────────────────────
        // Lᵀ is upper triangular.  Processing from bottom to top:
        // for each row i (from n-1 downto 0), for each L[i,j] with j < i:
        //   w[j] -= L[i,j] * w[i]
        // This updates earlier (smaller) indices using already-solved later ones.
        for i in (0..n).rev() {
            let wi = ys[i];
            for pos in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[pos]; // j < i
                ys[j] -= self.l_val[pos] * wi;
            }
        }
    }
}

// ─── Helper ──────────────────────────────────────────────────────────────────

/// Find `l_val` at position `(row, col=target)` in a CSR row.
/// Returns `None` if the column does not exist in the row.
#[inline]
fn find_l_entry<T: Copy>(
    col_idx: &[usize],
    l_val:   &[T],
    row_start: usize,
    row_end:   usize,
    target:    usize,
) -> Option<T> {
    // Binary search since col_idx within a row are sorted.
    let slice = &col_idx[row_start..row_end];
    match slice.binary_search(&target) {
        Ok(local_pos) => Some(l_val[row_start + local_pos]),
        Err(_)        => None,
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        sparse::{CooMatrix, CsrMatrix},
        DenseVec,
    };

    fn tridiag_spd(n: usize, d: f64, off: f64) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, d);
            if i > 0     { coo.push(i, i - 1, off); }
            if i < n - 1 { coo.push(i, i + 1, off); }
        }
        CsrMatrix::from_coo(&coo)
    }

    fn solve_ildlt(a: &CsrMatrix<f64>, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let precond = IldltPrecond::<f64>::from_csr(a).unwrap();
        let x_in = DenseVec::from_vec(b.to_vec());
        let mut y = DenseVec::zeros(n);
        precond.apply_precond(&x_in, &mut y);
        y.as_slice().to_vec()
    }

    // ── 1. Diagonal matrix: LDLᵀ = D itself; M⁻¹ x = D⁻¹ x ────────────────

    #[test]
    fn ildlt_diagonal_exact() {
        let n = 5;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n { coo.push(i, i, i as f64 + 2.0); }
        let a = CsrMatrix::from_coo(&coo);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let y = solve_ildlt(&a, &b);
        for i in 0..n {
            let expected = b[i] / (i as f64 + 2.0);
            assert!((y[i] - expected).abs() < 1e-12,
                "diagonal ILDLt at i={i}: got {}, expected {}", y[i], expected);
        }
    }

    // ── 2. Tridiagonal SPD: should be invertible and give finite output ──────

    #[test]
    fn ildlt_tridiag_spd_finite() {
        let n = 10;
        let a = tridiag_spd(n, 4.0, -1.0);
        let b = vec![1.0f64; n];
        let y = solve_ildlt(&a, &b);
        assert!(y.iter().all(|v| v.is_finite()),
            "ILDLt output contains non-finite");
    }

    // ── 3. D entries should match A diagonal after factorisation (tridiag) ───

    #[test]
    fn ildlt_d_positive_spd() {
        let n = 8;
        let a = tridiag_spd(n, 4.0, -1.0);
        let precond = IldltPrecond::<f64>::from_csr(&a).unwrap();
        assert!(precond.d().iter().all(|&v| v > 0.0),
            "ILDLt on SPD matrix should have positive D");
    }

    // ── 4. Identity matrix: M⁻¹ x = x ──────────────────────────────────────

    #[test]
    fn ildlt_identity_is_identity_precond() {
        let n = 4;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.push(i, i, 1.0); }
        let a = CsrMatrix::from_coo(&coo);
        let b = vec![1.0, 2.0, 3.0, 4.0f64];
        let y = solve_ildlt(&a, &b);
        for i in 0..n {
            assert!((y[i] - b[i]).abs() < 1e-13,
                "ILDLt identity precond: y[{i}]={} != b[{i}]={}", y[i], b[i]);
        }
    }

    // ── 5. Homogeneity: M⁻¹(alpha * x) = alpha * M⁻¹(x) ───────────────────

    #[test]
    fn ildlt_homogeneous() {
        let n = 6;
        let a = tridiag_spd(n, 3.0, -0.5);
        let precond = IldltPrecond::<f64>::from_csr(&a).unwrap();
        let x: Vec<f64> = vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5];
        let alpha = 3.7f64;
        let bx  = DenseVec::from_vec(x.clone());
        let bax = DenseVec::from_vec(x.iter().map(|&v| alpha * v).collect());
        let mut y   = DenseVec::zeros(n);
        let mut yax = DenseVec::zeros(n);
        precond.apply_precond(&bx, &mut y);
        precond.apply_precond(&bax, &mut yax);
        let err = y.as_slice().iter().zip(yax.as_slice())
            .map(|(yi, yaxi)| (alpha * yi - yaxi).abs())
            .fold(0.0f64, f64::max);
        assert!(err < 1e-12, "ILDLt not homogeneous: max err = {err}");
    }

    // ── 6. Error on non-square matrix ────────────────────────────────────────

    #[test]
    fn ildlt_error_non_square() {
        use crate::sparse::CooMatrix;
        let mut coo = CooMatrix::<f64>::new(3, 4);
        for i in 0..3 { coo.push(i, i, 1.0); }
        let a = CsrMatrix::from_coo(&coo);
        assert!(IldltPrecond::<f64>::from_csr(&a).is_err(),
            "expected error for non-square matrix");
    }
}
