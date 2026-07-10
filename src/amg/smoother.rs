//! AMG smoothers: weighted Jacobi, Gauss-Seidel, and Chebyshev.
//!
//! All hot paths are now delegated to the SIMD-accelerated implementations in
//! [`crate::simd::smoother`].  The private scalar helpers (jacobi_sweep,
//! gs_forward, gs_backward, chebyshev_sweep) have been removed.

use crate::core::scalar::{ComplexScalar, Scalar};
use crate::core::vector::DenseVec;
use crate::sparse::CsrMatrix;
use crate::simd::smoother::{
    jacobi_smooth, gs_smooth, chebyshev_smooth,
    estimate_spectral_radius,
};

/// Smoother variant.
#[derive(Clone, Debug)]
pub enum SmootherType {
    /// Weighted Jacobi with relaxation ω (typically 2/3 for AMG).
    WeightedJacobi { omega: f64 },
    /// Forward Gauss-Seidel (one sweep).
    GaussSeidel,
    /// Symmetric Gauss-Seidel (forward + backward).
    SymmetricGaussSeidel,
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
        SmootherType::Chebyshev { degree, ratio } => {
            // Use cached ρ(D⁻¹A) or estimate via power iterations.
            let rho = spectral_radius.unwrap_or_else(|| estimate_spectral_radius(a, 10));
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
