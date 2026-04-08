//! SOR and SSOR preconditioners.
//!
//! **SOR** (Successive Over-Relaxation) — one forward sweep of
//!   (D + ωL) z = ω r
//!
//! **SSOR** (Symmetric SOR) — forward sweep, diagonal scale, backward sweep:
//!   (D + ωL) z₁ = ω r
//!   z₂ᵢ = (ω/(2−ω)) · aᵢᵢ · z₁ᵢ   [diagonal scaling]
//!   (D + ωU) z  = z₂
//!
//! Both require ω ∈ (0, 2) for convergence with SPD matrices.
//!
//! **Analogs**
//!   PETSc: `PCSOR` with `PCSORSetSymmetric` / `PCSORSetOmega`
//!   HYPRE: `HYPRE_BoomerAMGSetRelaxType` (SOR=6, SSOR=9)

#![allow(clippy::needless_range_loop)]
use crate::core::{
    error::SolverError, preconditioner::Preconditioner, scalar::Scalar, vector::DenseVec,
};
use crate::sparse::CsrMatrix;

// ─── shared internals ────────────────────────────────────────────────────────

/// Compact CSR-like storage for lower and upper triangular parts plus diagonal.
struct SplitCsr<T> {
    nrows: usize,
    /// Diagonal values (length n).
    diag: Vec<T>,
    /// Lower-triangular part: (row_ptr, col_idx, values) with col < row.
    l_row_ptr: Vec<usize>,
    l_col_idx: Vec<usize>,
    l_values: Vec<T>,
    /// Upper-triangular part: (row_ptr, col_idx, values) with col > row.
    u_row_ptr: Vec<usize>,
    u_col_idx: Vec<usize>,
    u_values: Vec<T>,
}

impl<T: Scalar> SplitCsr<T> {
    fn from_csr(mat: &CsrMatrix<T>) -> Result<Self, SolverError> {
        let n = mat.nrows();
        if mat.ncols() != n {
            return Err(SolverError::PrecondSetupFailed {
                reason: "SOR/SSOR requires a square matrix".into(),
            });
        }

        let tol = T::machine_epsilon() * T::from_f64(1e6);
        let diag_vec = mat.diag();
        for (i, &d) in diag_vec.iter().enumerate() {
            if d.abs() < tol {
                return Err(SolverError::PrecondSetupFailed {
                    reason: format!("near-zero diagonal at row {i}: {d:?}"),
                });
            }
        }

        let mut l_row_ptr = vec![0usize; n + 1];
        let mut u_row_ptr = vec![0usize; n + 1];
        let mut l_col_idx = Vec::new();
        let mut l_values = Vec::new();
        let mut u_col_idx = Vec::new();
        let mut u_values = Vec::new();

        for (r, c, v) in mat.triplets() {
            if c < r {
                l_col_idx.push(c);
                l_values.push(v);
                l_row_ptr[r + 1] += 1;
            } else if c > r {
                u_col_idx.push(c);
                u_values.push(v);
                u_row_ptr[r + 1] += 1;
            }
        }
        // prefix-sum
        for i in 0..n {
            l_row_ptr[i + 1] += l_row_ptr[i];
            u_row_ptr[i + 1] += u_row_ptr[i];
        }

        Ok(SplitCsr {
            nrows: n,
            diag: diag_vec,
            l_row_ptr,
            l_col_idx,
            l_values,
            u_row_ptr,
            u_col_idx,
            u_values,
        })
    }

    /// Forward (L+D) solve:  (D + ωL) z = ω r
    fn forward_solve(&self, omega: T, r: &[T], z: &mut [T]) {
        let n = self.nrows;
        for i in 0..n {
            let mut sum = T::zero();
            let start = self.l_row_ptr[i];
            let end = self.l_row_ptr[i + 1];
            for k in start..end {
                sum += self.l_values[k] * z[self.l_col_idx[k]];
            }
            z[i] = (omega * r[i] - omega * sum) / self.diag[i];
        }
    }

    /// Backward (D+U) solve:  (D + ωU) z = rhs
    fn backward_solve(&self, omega: T, rhs: &[T], z: &mut [T]) {
        let n = self.nrows;
        for i in (0..n).rev() {
            let mut sum = T::zero();
            let start = self.u_row_ptr[i];
            let end = self.u_row_ptr[i + 1];
            for k in start..end {
                sum += self.u_values[k] * z[self.u_col_idx[k]];
            }
            z[i] = (rhs[i] - omega * sum) / self.diag[i];
        }
    }
}

// ─── SOR preconditioner ───────────────────────────────────────────────────────

/// Forward SOR preconditioner:  M⁻¹ r = (D + ωL)⁻¹ · ω r
pub struct SorPrecond<T> {
    split: SplitCsr<T>,
    omega: T,
}

impl<T: Scalar> SorPrecond<T> {
    /// Build from a CSR matrix with relaxation parameter ω ∈ (0, 2).
    pub fn from_csr(mat: &CsrMatrix<T>, omega: T) -> Result<Self, SolverError> {
        let two = T::from_f64(2.0);
        if omega <= T::zero() || omega >= two {
            return Err(SolverError::PrecondSetupFailed {
                reason: format!("omega must be in (0,2), got {omega:?}"),
            });
        }
        Ok(SorPrecond {
            split: SplitCsr::from_csr(mat)?,
            omega,
        })
    }
}

impl<T: Scalar> Preconditioner for SorPrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        self.split.forward_solve(self.omega, x.as_slice(), y.as_mut_slice());
    }
}

// ─── SSOR preconditioner ──────────────────────────────────────────────────────

/// Symmetric SOR preconditioner.
///
/// Three-phase solve:
/// 1. Forward:   (D + ωL) z₁ = ω r
/// 2. Diagonal:  z₂ᵢ = (ω/(2−ω)) · dᵢ · z₁ᵢ
/// 3. Backward:  (D + ωU) z  = z₂
pub struct SsorPrecond<T> {
    split: SplitCsr<T>,
    omega: T,
}

impl<T: Scalar> SsorPrecond<T> {
    /// Build from a CSR matrix with relaxation parameter ω ∈ (0, 2).
    pub fn from_csr(mat: &CsrMatrix<T>, omega: T) -> Result<Self, SolverError> {
        let two = T::from_f64(2.0);
        if omega <= T::zero() || omega >= two {
            return Err(SolverError::PrecondSetupFailed {
                reason: format!("omega must be in (0,2), got {omega:?}"),
            });
        }
        Ok(SsorPrecond {
            split: SplitCsr::from_csr(mat)?,
            omega,
        })
    }
}

impl<T: Scalar> Preconditioner for SsorPrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n = self.split.nrows;
        let omega = self.omega;
        let two = T::from_f64(2.0);

        // Phase 1: forward solve into y (temporarily used as z₁)
        let mut z1 = vec![T::zero(); n];
        self.split.forward_solve(omega, x.as_slice(), &mut z1);

        // Phase 2: diagonal scaling → z₂ stored in z1 in-place
        // z₂ᵢ = (ω/(2−ω)) · dᵢ · z₁ᵢ
        let scale = omega / (two - omega);
        for i in 0..n {
            z1[i] = scale * self.split.diag[i] * z1[i];
        }

        // Phase 3: backward solve  (D + ωU) y = z₂
        let rhs = z1;
        self.split.backward_solve(omega, &rhs, y.as_mut_slice());
    }
}
