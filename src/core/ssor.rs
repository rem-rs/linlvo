//! Symmetric Successive Over-Relaxation (SSOR) preconditioner.
//!
//! For a symmetric matrix `A = D + L + Lᵀ` where `D` is diagonal and `L` is the
//! strictly lower-triangular part, the SSOR preconditioner is:
//!
//! ```text
//! M = (D + ωL) · D⁻¹ · (D + ωL)ᵀ
//! ```
//!
//! SSOR is **symmetric** (preserves the symmetry of the preconditioned system),
//! making it suitable as a preconditioner for CG, MINRES, or TFQMR applied to
//! symmetric matrices.
//!
//! For ω = 1 it reduces to symmetric Gauss-Seidel.
//! For ω > 1 it over-relaxes (may accelerate convergence for diagonally-dominant
//! matrices but risks instability).
//! For ω < 1 it under-relaxes (more robust for indefinite systems).
//!
//! Reference: Saad, "Iterative Methods for Sparse Linear Systems" §12.3.2.

use crate::core::{
    preconditioner::Preconditioner,
    scalar::Scalar,
    vector::{DenseVec, Vector},
};
use num_complex::Complex;

/// CSR-style sparse matrix entry for building the SSOR preconditioner.
/// Stores the lower-triangular structure of `A = D + L + Lᵀ`.
pub struct SsorSparse<T: Scalar> {
    /// Number of rows = columns (full system size).
    pub n: usize,
    /// Diagonal entries D[i].
    pub diag: Vec<Complex<T>>,
    /// Row pointers for lower-triangular sparse matrix L (CSR format).
    pub l_row_ptr: Vec<usize>,
    /// Column indices for L.
    pub l_col_idx: Vec<usize>,
    /// Values for L (only stores entries BELOW the diagonal).
    pub l_vals: Vec<Complex<T>>,
}

/// SSOR preconditioner for complex-symmetric systems.
///
/// Apply step: `x = M⁻¹ · r` via forward/backward substitution:
///   1. (D + ωL) · y = r       (forward, row 0…n-1)
///   2. z = D · y               (scaling)
///   3. (D + ωL)ᵀ · x = z      (backward, row n-1…0)
pub struct SsorPrecond<T: Scalar> {
    n: usize,
    diag: Vec<Complex<T>>,
    l_row_ptr: Vec<usize>,
    l_col_idx: Vec<usize>,
    l_vals: Vec<Complex<T>>,
    omega: f64,
}

impl<T: Scalar> SsorPrecond<T> {
    /// Build the SSOR preconditioner from the sparse matrix structure.
    ///
    /// * `diag` — diagonal entries D[i] (length n).
    /// * `l_row_ptr`, `l_col_idx`, `l_vals` — CSR representation of the
    ///   **strictly lower-triangular** part L.
    /// * `omega` — relaxation parameter (0 < ω < 2; default 1.0).
    ///   ω = 1 → symmetric Gauss-Seidel.
    ///   ω > 1 → over-relaxation (use cautiously with indefinite matrices).
    pub fn new(
        diag: Vec<Complex<T>>,
        l_row_ptr: Vec<usize>,
        l_col_idx: Vec<usize>,
        l_vals: Vec<Complex<T>>,
        omega: f64,
    ) -> Self {
        let n = diag.len();
        Self { n, diag, l_row_ptr, l_col_idx, l_vals, omega }
    }

    /// Forward substitution: solve (D + ωL) · y = r
    fn solve_forward(&self, r: &[Complex<T>]) -> Vec<Complex<T>> {
        let mut y = vec![Complex::new(T::zero(), T::zero()); self.n];
        for i in 0..self.n {
            let mut sum = Complex::new(T::zero(), T::zero());
            let start = self.l_row_ptr[i];
            let end = self.l_row_ptr[i + 1];
            for k in start..end {
                let j = self.l_col_idx[k];
                if j < i {
                    sum += self.l_vals[k] * y[j];
                }
            }
            y[i] = (r[i] - Complex::new(T::from_f64(self.omega), T::zero()) * sum)
                / self.diag[i];
        }
        y
    }

    /// Backward substitution: solve (D + ωL)ᵀ · x = z
    fn solve_backward(&self, z: &[Complex<T>]) -> Vec<Complex<T>> {
        let mut x = vec![Complex::new(T::zero(), T::zero()); self.n];
        for ii in 0..self.n {
            let i = self.n - 1 - ii;
            let mut sum = Complex::new(T::zero(), T::zero());
            let start = self.l_row_ptr[i];
            let end = self.l_row_ptr[i + 1];
            for k in start..end {
                let j = self.l_col_idx[k];
                if j < i {
                    // (D + ωL)ᵀ has ω·A[j,i] at position (i,j), i > j
                    // Since A is symmetric: A[j,i] = A[i,j] = L[i,j]
                    sum += self.l_vals[k] * x[j];
                }
            }
            x[i] = (z[i] - Complex::new(T::from_f64(self.omega), T::zero()) * sum)
                / self.diag[i];
        }
        x
    }
}

impl<T: Scalar> Preconditioner for SsorPrecond<T> {
    type Vector = DenseVec<Complex<T>>;

    fn apply_precond(&self, src: &Self::Vector, dst: &mut Self::Vector) {
        let r = src.as_slice();
        // Step 1: (D + ωL) · y = r
        let y = self.solve_forward(r);
        // Step 2: z = D · y
        let z: Vec<Complex<T>> = y.iter().zip(self.diag.iter())
            .map(|(&yi, &di)| yi * di)
            .collect();
        // Step 3: (D + ωL)ᵀ · x = z
        let x = self.solve_backward(&z);
        // Copy to output
        let ys = dst.as_mut_slice();
        for (i, &xi) in x.iter().enumerate() {
            ys[i] = xi;
        }
    }
}
