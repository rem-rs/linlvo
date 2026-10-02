//! AMG smoothers: weighted Jacobi, Gauss-Seidel, and Chebyshev.
//!
//! All hot paths are now delegated to the SIMD-accelerated implementations in
//! [`crate::simd::smoother`].  The private scalar helpers (jacobi_sweep,
//! gs_forward, gs_backward, chebyshev_sweep) have been removed.

use crate::core::scalar::{ComplexScalar, Scalar};
use crate::core::vector::DenseVec;
use crate::sparse::CsrMatrix;
use crate::simd::smoother::{
    jacobi_smooth, gs_smooth, chebyshev_smooth, l1_sgs_smooth,
    estimate_spectral_radius,
};
use num_traits::{One, Zero};

/// Gershgorin upper bound of the spectrum of the symmetric scaling
/// `D^{-1/2} A D^{-1/2}` (same spectrum as `D⁻¹A`):
/// `max_i (|a_ii| + Σ_{j≠i} |a_ij|) / |a_ii|`.
///
/// Guaranteed upper bound, O(nnz) sequential — unlike the power-iteration
/// [`estimate_spectral_radius`], whose uniform start vector is dominated by
/// the *lowest* mode and can undershoot ρ, which flips the sign of the
/// Chebyshev inverse polynomial above λmax and makes the cycle preconditioner
/// indefinite (CG then aborts on `(B r, r) < 0`, MFEM solvers.cpp:938).
pub fn gershgorin_upper_scaled<T: ComplexScalar>(a: &CsrMatrix<T>) -> T::Real {
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vs = a.values();
    let mut bound = T::Real::zero();
    for i in 0..a.nrows() {
        let mut diag = T::Real::zero();
        let mut off = T::Real::zero();
        for k in rp[i]..rp[i + 1] {
            let m = vs[k].abs();
            if ci[k] == i { diag = m; } else { off += m; }
        }
        if diag > T::Real::zero() {
            let r = <T::Real as One>::one() + off / diag;
            if r > bound { bound = r; }
        }
    }
    bound
}

/// Smoother variant.
#[derive(Clone, Debug)]
pub enum SmootherType {
    /// Weighted Jacobi with relaxation ω (typically 2/3 for AMG).
    WeightedJacobi { omega: f64 },
    /// Forward Gauss-Seidel (one sweep).
    GaussSeidel,
    /// Symmetric Gauss-Seidel (forward + backward).
    SymmetricGaussSeidel,
    /// L1-scaled symmetric Gauss-Seidel (hypre BoomerAMG relax type 8, the
    /// AMS B_Pi/B_G smoother): the sweep divides by the l1 row norm instead
    /// of the diagonal, so rows of a singular coarse operator whose diagonal
    /// underflows stay stable.
    L1SymmetricGaussSeidel,
    /// Chebyshev polynomial smoother (degree iterations, eigenvalue ratio).
    ///
    /// `degree` is the polynomial degree (number of iterations, typically 2–5).
    /// `ratio` controls `λ_min = λ_max / ratio` (typically 3–10 for AMG smoothing).
    Chebyshev { degree: usize, ratio: f64 },
}

/// Apply `n_sweeps` pre-smoothing iterations: `x ← smooth(A, x, b)`.
pub fn smooth<T: ComplexScalar>(
    a:       &CsrMatrix<T>,
    x:       &mut DenseVec<T>,
    b:       &DenseVec<T>,
    smoother: &SmootherType,
    n_sweeps: usize,
) {
    smooth_with_hint(a, x, b, smoother, n_sweeps, None);
}

/// Like [`smooth`] but accepts an optional cached spectral radius ρ(D⁻¹A).
///
/// When the smoother is `Chebyshev` and `spectral_radius` is `Some`, the
/// expensive power-iteration estimate is skipped.
pub fn smooth_with_hint<T: ComplexScalar>(
    a:       &CsrMatrix<T>,
    x:       &mut DenseVec<T>,
    b:       &DenseVec<T>,
    smoother: &SmootherType,
    n_sweeps: usize,
    spectral_radius: Option<T>,
) {
    match smoother {
        SmootherType::WeightedJacobi { omega } => {
            let omega = T::from_f64(*omega);
            jacobi_smooth(a, x, b, omega, n_sweeps);
        }
        SmootherType::GaussSeidel => {
            gs_smooth(a, x, b, false, n_sweeps);
        }
        SmootherType::SymmetricGaussSeidel => {
            gs_smooth(a, x, b, true, n_sweeps);
        }
        SmootherType::L1SymmetricGaussSeidel => {
            l1_sgs_smooth(a, x, b, n_sweeps);
        }
        SmootherType::Chebyshev { degree, ratio } => {
            // Use cached ρ(D⁻¹A) or estimate via power iterations, then clamp
            // with the guaranteed Gershgorin bound: an under-estimated ρ puts
            // λmax below the true spectrum top and the Chebyshev inverse
            // polynomial turns NEGATIVE there — the cycle preconditioner
            // becomes indefinite and CG aborts on (B r, r) < 0 (MFEM
            // solvers.cpp:938 semantics, D976).
            let rho = spectral_radius.unwrap_or_else(|| estimate_spectral_radius(a, 10));
            let gersh = gershgorin_upper_scaled(a);
            let rho = if gersh > rho.abs() { T::from_real(gersh) } else { rho };
            let lambda_max = rho * T::from_f64(1.1);
            let lambda_min = lambda_max / T::from_f64(*ratio);
            // Guard: if estimate is zero/tiny, fall back to Jacobi.
            if lambda_max.abs() < <T::Real as Scalar>::from_f64(1e-14) {
                let omega = T::from_real(<T::Real as Scalar>::from_f64(0.667));
                jacobi_smooth(a, x, b, omega, n_sweeps);
            } else {
                for _ in 0..n_sweeps {
                    chebyshev_smooth(a, x, b, lambda_min, lambda_max, *degree);
                }
            }
        }
    }
}
