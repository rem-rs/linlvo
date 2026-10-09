//! MFEM-compatible Gauss-Seidel smoother (symmetric GS sweeps).
//!
//! Unlike the SSOR preconditioner in [`super::sor`] which uses the strict
//! factorization `M = (D+ωL)·D⁻¹·(D+ωU)`, this smoother performs full
//! forward/backward Gauss-Seidel sweeps over ALL off-diagonal entries —
//! exactly matching MFEM's `GSSmoother::Mult` with `type=0` (symmetric).
//!
//! ```text
//! Forward:  for i = 0 … n-1:
//!     sum = Σ_{j≠i} A[i,j] · y[j]
//!     y[i] = (r[i] - sum) / A[i,i]
//!
//! Backward: for i = n-1 … 0:
//!     sum = Σ_{j≠i} A[i,j] · y[j]
//!     y[i] = (r[i] - sum) / A[i,i]
//! ```
//!
//! Reference: MFEM `sparsesmoothers.cpp` — `GSSmoother::Mult`.

use crate::core::{
    error::SolverError,
    preconditioner::Preconditioner,
    scalar::{ComplexScalar, Scalar},
    vector::DenseVec,
};
use crate::sparse::CsrMatrix;

/// MFEM-compatible symmetric Gauss-Seidel smoother.
///
/// Performs one forward sweep + one backward sweep per `apply_precond` call,
/// using ALL off-diagonal entries of each row (matching MFEM's GSSmoother).
pub struct GaussSeidelSmoother<T> {
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
    diag: Vec<T>,
}

impl<T: ComplexScalar> GaussSeidelSmoother<T> {
    /// Build from a CSR matrix.
    ///
    /// The matrix must be square with non-zero diagonal entries.
    pub fn from_csr(mat: &CsrMatrix<T>) -> Result<Self, SolverError> {
        let n = mat.nrows();
        if mat.ncols() != n {
            return Err(SolverError::PrecondSetupFailed {
                reason: "GaussSeidelSmoother requires a square matrix".into(),
            });
        }

        let tol = T::machine_epsilon() * <T::Real as Scalar>::from_f64(1e6);
        let diag = mat.diag();
        for (i, &d) in diag.iter().enumerate() {
            if d.abs() < tol {
                return Err(SolverError::PrecondSetupFailed {
                    reason: format!("near-zero diagonal at row {i}: {d:?}"),
                });
            }
        }

        // Store the full CSR (row_ptr, col_idx, values) directly.
        let row_ptr = mat.row_ptr().to_vec();
        let col_idx = mat.col_idx().to_vec();
        let values = mat.values().to_vec();

        Ok(GaussSeidelSmoother { n, row_ptr, col_idx, values, diag })
    }

    /// Forward Gauss-Seidel sweep: for i = 0..n, update y[i] using all y[j].
    fn sweep_forward(&self, r: &[T], y: &mut [T]) {
        for i in 0..self.n {
            let mut sum = T::zero();
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                let c = self.col_idx[k];
                if c != i {
                    sum = sum + self.values[k] * y[c];
                }
            }
            y[i] = (r[i] - sum) / self.diag[i];
        }
    }

    /// Backward Gauss-Seidel sweep: for i = n-1..0, update y[i] using all y[j].
    ///
    /// Within a row the nonzeros are scanned in **descending** column order,
    /// mirroring MFEM `SparseMatrix::Gauss_Seidel_back`'s finalized (CSR) path
    /// (`linalg/sparsemat.cpp`: `for (j = Ip[s]-1; j >= Ip[i]; j--)`).  The
    /// scan direction fixes the floating-point summation order of the
    /// off-diagonal accumulation, so this sweep is bitwise-compatible with
    /// MFEM rather than merely algebraically equal (D945).
    fn sweep_backward(&self, r: &[T], y: &mut [T]) {
        for ii in 0..self.n {
            let i = self.n - 1 - ii;
            let mut sum = T::zero();
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in (start..end).rev() {
                let c = self.col_idx[k];
                if c != i {
                    sum = sum + self.values[k] * y[c];
                }
            }
            y[i] = (r[i] - sum) / self.diag[i];
        }
    }
}

impl<T: ComplexScalar> Preconditioner for GaussSeidelSmoother<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let xs = x.as_slice();
        let ys = y.as_mut_slice();
        // y starts as 0 (MFEM: if !iterative_mode, y = 0.0)
        // CG calls apply_precond with the preconditioner initialized from
        // the current residual, so y entering as the residual x is the
        // right-hand side and y should be zeroed.
        for i in 0..self.n {
            ys[i] = T::zero();
        }

        // Forward sweep
        self.sweep_forward(xs, ys);

        // Backward sweep (uses ys updated by forward sweep)
        self.sweep_backward(xs, ys);
    }
}
