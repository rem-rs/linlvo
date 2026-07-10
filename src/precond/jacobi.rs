//! Jacobi (diagonal) preconditioner.
//!
//! M⁻¹ = diag(A)⁻¹ — scales each component by the reciprocal diagonal.
//!
//! **Analogs**
//!   PETSc: `PCJACOBI`
//!   HYPRE: `HYPRE_BoomerAMGSetRelaxType(precond, 7)` (Jacobi relaxation)

use crate::core::{
    error::SolverError, preconditioner::Preconditioner, scalar::{ComplexScalar, Scalar}, vector::DenseVec,
};
use crate::sparse::CsrMatrix;

/// Jacobi (diagonal) preconditioner.
///
/// Stores the reciprocal of each diagonal entry.  Setup is O(n); each
/// application is a simple element-wise multiply — O(n) with no memory
/// traffic beyond the vector itself.
pub struct JacobiPrecond<T> {
    inv_diag: Vec<T>,
}

impl<T: ComplexScalar> JacobiPrecond<T> {
    /// Build from a CSR matrix.
    ///
    /// Returns `Err(SolverError::PrecondSetupFailed)` if any diagonal entry
    /// has absolute value below `1e6 * ε_machine` (near-zero pivot).
    pub fn from_csr(mat: &CsrMatrix<T>) -> Result<Self, SolverError> {
        let diag = mat.diag();
        let tol = T::machine_epsilon() * <T::Real as Scalar>::from_f64(1e6);
        for (i, &d) in diag.iter().enumerate() {
            if d.abs() < tol {
                return Err(SolverError::PrecondSetupFailed {
                    reason: format!("near-zero diagonal at row {i}: {d:?}"),
                });
            }
        }
        let inv_diag = diag.iter().map(|&d| T::one() / d).collect();
        Ok(JacobiPrecond { inv_diag })
    }
}

impl<T: ComplexScalar> Preconditioner for JacobiPrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        for (yi, (&xi, &di)) in y
            .as_mut_slice()
            .iter_mut()
            .zip(x.as_slice().iter().zip(self.inv_diag.iter()))
        {
            *yi = xi * di;
        }
    }
}
