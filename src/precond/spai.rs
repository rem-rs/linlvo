//! SPAI — Sparse Approximate Inverse.
//!
//! Computes M ≈ A⁻¹ by minimising ‖A M − I‖_F column by column:
//!
//! ```text
//! min_{m_j}  ‖A m_j − e_j‖₂   subject to  support(m_j) ⊆ J_j
//! ```
//!
//! The sparsity pattern J_j of each column is chosen as the nonzero pattern of
//! column j of A (for symmetric A, this equals the nonzero pattern of row j).
//! The row set I_j = ⋃_{k ∈ J_j} { nonzero rows of column k of A } forms the
//! "active" row block.  The local least-squares problem
//!
//! ```text
//! min  ‖Â m̂_j − ê_j‖₂
//! ```
//!
//! where Â = A[I_j, J_j] and ê_j = (e_j)[I_j], is solved with a dense QR
//! factorisation (modified Gram-Schmidt).
//!
//! **Reference**: Grote & Huckle, SIAM J. Sci. Comput. 18, 1997.
//!
//! **Analogs**
//!   PETSc: `PCSPAI`
//!   HYPRE: (no direct equivalent; use `EUCLID` or `Parasails`)

#![allow(clippy::needless_range_loop)]
use crate::core::{
    error::SolverError, preconditioner::Preconditioner, scalar::Scalar, vector::DenseVec,
};
use crate::sparse::CsrMatrix;

/// Static-pattern SPAI preconditioner.
///
/// The sparsity pattern of M matches the sparsity pattern of A (symmetric
/// assumption: the column pattern of A = the row pattern of A).
pub struct SpaiPrecond<T> {
    nrows: usize,
    /// M stored column-by-column (col j → sorted list of (row, value)).
    cols:  Vec<Vec<(usize, T)>>,
}

impl<T: Scalar> SpaiPrecond<T> {
    /// Compute the static-pattern SPAI for `mat`.
    ///
    /// Assumes `mat` is (approximately) symmetric; uses row patterns as column
    /// patterns for M.
    pub fn from_csr(mat: &CsrMatrix<T>) -> Result<Self, SolverError> {
        let n = mat.nrows();
        if mat.ncols() != n {
            return Err(SolverError::PrecondSetupFailed {
                reason: "SPAI requires a square matrix".into(),
            });
        }

        let rp = mat.row_ptr();
        let ci = mat.col_idx();
        let vs = mat.values();

        let mut cols: Vec<Vec<(usize, T)>> = Vec::with_capacity(n);

        for j in 0..n {
            // J = nonzero columns of row j (= column j's pattern for symmetric A)
            let j_set: Vec<usize> = ci[rp[j]..rp[j + 1]].to_vec();

            // I = union of row patterns for each k ∈ J
            let mut i_set: Vec<usize> = Vec::new();
            for &k in &j_set {
                for pos in rp[k]..rp[k + 1] {
                    i_set.push(ci[pos]);
                }
            }
            i_set.sort_unstable();
            i_set.dedup();

            let ni = i_set.len();
            let nj = j_set.len();

            // Build dense submatrix Â[ni × nj] = A[I, J]
            let mut a_hat = vec![T::zero(); ni * nj];
            for (jj, &k) in j_set.iter().enumerate() {
                for pos in rp[k]..rp[k + 1] {
                    let row = ci[pos];
                    if let Ok(ii) = i_set.binary_search(&row) {
                        a_hat[ii + jj * ni] = vs[pos]; // column-major
                    }
                }
            }

            // Build ê_j: 1 if j ∈ I, else 0
            let mut e_hat = vec![T::zero(); ni];
            if let Ok(pos_j) = i_set.binary_search(&j) {
                e_hat[pos_j] = T::one();
            }

            // Solve min ‖Â m̂ − ê‖₂ via QR (modified Gram-Schmidt, column-major)
            let m_hat = qr_lstsq(&a_hat, &e_hat, ni, nj);

            // Store column j of M
            let col_j: Vec<(usize, T)> = j_set
                .iter()
                .enumerate()
                .map(|(jj, &row)| (row, m_hat[jj]))
                .collect();
            cols.push(col_j);
        }

        Ok(SpaiPrecond { nrows: n, cols })
    }
}

impl<T: Scalar> Preconditioner for SpaiPrecond<T> {
    type Vector = DenseVec<T>;

    /// Compute y = M x  (SpMV with M stored column-by-column).
    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n = self.nrows;
        let xs = x.as_slice();
        let ys = y.as_mut_slice();
        for yi in ys.iter_mut() { *yi = T::zero(); }

        // y += M x = sum_j x[j] * (column j of M)
        for j in 0..n {
            let xj = xs[j];
            if xj == T::zero() { continue; }
            for &(row, val) in &self.cols[j] {
                ys[row] += val * xj;
            }
        }
    }
}

// ─── Dense QR least-squares solver ───────────────────────────────────────────

/// Solve min ‖A m − b‖₂ via modified Gram-Schmidt QR.
///
/// `a` is stored **column-major**: a[i + j*m] is row i, col j.
/// Returns the `n`-vector solution m̂ (or zeros if the system is rank-deficient).
fn qr_lstsq<T: Scalar>(a: &[T], b: &[T], m: usize, n: usize) -> Vec<T> {
    if m == 0 || n == 0 {
        return vec![T::zero(); n];
    }

    // Work on copies; Q stored implicitly via modified GS on columns.
    let mut q = a.to_vec();   // column-major, m × n
    let mut r  = vec![T::zero(); n * n]; // upper triangular, n × n, column-major

    let eps = T::machine_epsilon() * T::from_f64(1e3);

    // Modified Gram-Schmidt
    for j in 0..n {
        // ‖q[:, j]‖
        let norm_sq: T = (0..m).fold(T::zero(), |s, i| s + q[i + j * m] * q[i + j * m]);
        let norm = norm_sq.sqrt();

        if norm < eps {
            // Rank-deficient column: leave as zero, set r[j,j] = 0
            r[j + j * n] = T::zero();
            continue;
        }
        r[j + j * n] = norm;
        // Normalise column j
        for i in 0..m {
            q[i + j * m] /= norm;
        }
        // Orthogonalise subsequent columns against column j
        for kk in (j + 1)..n {
            let dot: T = (0..m).fold(T::zero(), |s, i| s + q[i + j * m] * q[i + kk * m]);
            r[j + kk * n] = dot;
            // Copy column j to temp to avoid simultaneous mutable+immutable borrow
            let col_j: Vec<T> = (0..m).map(|i| q[i + j * m]).collect();
            for i in 0..m {
                q[i + kk * m] -= dot * col_j[i];
            }
        }
    }

    // Form Qᵀ b
    let mut qtb = vec![T::zero(); n];
    for j in 0..n {
        let dot: T = (0..m).fold(T::zero(), |s, i| s + q[i + j * m] * b[i]);
        qtb[j] = dot;
    }

    // Back-substitute R m̂ = Qᵀ b
    let mut m_hat = vec![T::zero(); n];
    for j in (0..n).rev() {
        let rjj = r[j + j * n];
        if rjj.abs() < eps {
            m_hat[j] = T::zero();
            continue;
        }
        let mut s = qtb[j];
        for k in (j + 1)..n {
            s -= r[j + k * n] * m_hat[k];
        }
        m_hat[j] = s / rjj;
    }

    m_hat
}
