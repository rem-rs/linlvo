//! AMG smoothers: weighted Jacobi, Gauss-Seidel, and Chebyshev.

use crate::core::{operator::LinearOperator, scalar::Scalar, vector::{DenseVec, Vector}};
use crate::sparse::CsrMatrix;

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
pub fn smooth<T: Scalar>(
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
pub fn smooth_with_hint<T: Scalar>(
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
            jacobi_sweep(a, x, b, omega, n_sweeps);
        }
        SmootherType::GaussSeidel => {
            for _ in 0..n_sweeps { gs_forward(a, x, b); }
        }
        SmootherType::SymmetricGaussSeidel => {
            for _ in 0..n_sweeps {
                gs_forward(a, x, b);
                gs_backward(a, x, b);
            }
        }
        SmootherType::Chebyshev { degree, ratio } => {
            // Use cached ρ(D⁻¹A) or estimate via power iterations.
            let rho = spectral_radius.unwrap_or_else(|| estimate_spectral_radius_dinv_a(a, 10));
            let lambda_max = rho * T::from_f64(1.1);
            let lambda_min = lambda_max / T::from_f64(*ratio);
            // Guard: if estimate is zero/tiny, fall back to Jacobi.
            if lambda_max.abs() < T::from_f64(1e-14) {
                let omega = T::from_f64(0.667);
                jacobi_sweep(a, x, b, omega, n_sweeps);
            } else {
                for _ in 0..n_sweeps {
                    chebyshev_sweep(a, x, b, lambda_min, lambda_max, *degree);
                }
            }
        }
    }
}

// ─── Weighted Jacobi ─────────────────────────────────────────────────────────

fn jacobi_sweep<T: Scalar>(
    a:      &CsrMatrix<T>,
    x:      &mut DenseVec<T>,
    b:      &DenseVec<T>,
    omega:  T,
    sweeps: usize,
) {
    let n  = b.len();
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vs = a.values();
    let bs = b.as_slice();

    let mut ax = DenseVec::zeros(n);

    for _ in 0..sweeps {
        a.apply(x, &mut ax);
        let xs  = x.as_mut_slice();
        let axs = ax.as_slice();

        // Locate diagonal per row.
        for i in 0..n {
            let mut d = T::zero();
            for k in rp[i]..rp[i + 1] {
                if ci[k] == i { d = vs[k]; break; }
            }
            if d.abs() > T::machine_epsilon() {
                xs[i] += omega * (bs[i] - axs[i]) / d;
            }
        }
    }
}

// ─── Gauss-Seidel ────────────────────────────────────────────────────────────

fn gs_forward<T: Scalar>(a: &CsrMatrix<T>, x: &mut DenseVec<T>, b: &DenseVec<T>) {
    let n  = b.len();
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vs = a.values();
    let bs = b.as_slice();
    let xs = x.as_mut_slice();

    for i in 0..n {
        let mut d   = T::zero();
        let mut sum = bs[i];
        for k in rp[i]..rp[i + 1] {
            let j = ci[k];
            if j == i { d = vs[k]; } else { sum -= vs[k] * xs[j]; }
        }
        if d.abs() > T::machine_epsilon() {
            xs[i] = sum / d;
        }
    }
}

fn gs_backward<T: Scalar>(a: &CsrMatrix<T>, x: &mut DenseVec<T>, b: &DenseVec<T>) {
    let n  = b.len();
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vs = a.values();
    let bs = b.as_slice();
    let xs = x.as_mut_slice();

    for i in (0..n).rev() {
        let mut d   = T::zero();
        let mut sum = bs[i];
        for k in rp[i]..rp[i + 1] {
            let j = ci[k];
            if j == i { d = vs[k]; } else { sum -= vs[k] * xs[j]; }
        }
        if d.abs() > T::machine_epsilon() {
            xs[i] = sum / d;
        }
    }
}

// ─── Chebyshev polynomial smoother ──────────────────────────────────────────

/// Estimate spectral radius of `D^{-1}A` using power iterations, where `D = diag(A)`.
///
/// This is the correct eigenvalue bound for diagonally-preconditioned Chebyshev.
pub fn estimate_spectral_radius_dinv_a<T: Scalar>(a: &CsrMatrix<T>, n_iter: usize) -> T {
    let n = a.nrows();
    if n == 0 { return T::zero(); }

    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vs = a.values();

    // Extract diagonal inverse.
    let mut diag_inv = vec![T::one(); n];
    for i in 0..n {
        for k in rp[i]..rp[i + 1] {
            if ci[k] == i {
                let d = vs[k];
                if d.abs() > T::machine_epsilon() {
                    diag_inv[i] = T::one() / d;
                }
                break;
            }
        }
    }

    // Power iteration on D^{-1}A: w = D^{-1} * (A * v).
    let inv_sqrt_n = T::one() / T::from_f64((n as f64).sqrt());
    let mut v = DenseVec::from_vec(vec![inv_sqrt_n; n]);
    let mut w = DenseVec::zeros(n);
    let mut rho = T::one();

    for _ in 0..n_iter {
        a.apply(&v, &mut w);
        let ws = w.as_mut_slice();
        for i in 0..n { ws[i] *= diag_inv[i]; }
        rho = T::from_f64(ws.iter().map(|&x| {
            let f = num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0);
            f * f
        }).sum::<f64>().sqrt());
        if rho > T::from_f64(1e-14) {
            let inv = T::one() / rho;
            let vs2 = v.as_mut_slice();
            for i in 0..n { vs2[i] = ws[i] * inv; }
        }
    }
    rho
}

/// Chebyshev polynomial smoother iteration with diagonal preconditioning.
///
/// The eigenvalue interval `[lambda_min, lambda_max]` must be for `D^{-1}A`
/// where `D = diag(A)`.  The iteration uses `M^{-1} = D^{-1}` as preconditioner.
fn chebyshev_sweep<T: Scalar>(
    a:          &CsrMatrix<T>,
    x:          &mut DenseVec<T>,
    b:          &DenseVec<T>,
    lambda_min: T,
    lambda_max: T,
    n_iter:     usize,
) {
    let n = b.len();
    if n == 0 || n_iter == 0 { return; }

    // Chebyshev parameters for the interval [lambda_min, lambda_max].
    let sigma = (lambda_max + lambda_min) / T::from_f64(2.0);  // center
    let delta = (lambda_max - lambda_min) / T::from_f64(2.0);  // half-width

    if sigma.abs() < T::machine_epsilon() { return; }

    // theta = 2 / (lambda_max + lambda_min) = 1/sigma
    let theta = T::one() / sigma;

    // Recurrence parameter: (delta/(2*sigma))^2
    let half_delta_over_sigma = delta / (T::from_f64(2.0) * sigma);
    let qdsa = half_delta_over_sigma * half_delta_over_sigma;

    // Diagonal inverse M^{-1} = diag(A)^{-1}.
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vs = a.values();
    let mut diag_inv = vec![T::one(); n];
    for i in 0..n {
        for k in rp[i]..rp[i + 1] {
            if ci[k] == i {
                let d = vs[k];
                if d.abs() > T::machine_epsilon() {
                    diag_inv[i] = T::one() / d;
                }
                break;
            }
        }
    }

    let two_theta = T::from_f64(2.0) * theta;

    let mut ax = DenseVec::zeros(n);
    let mut d_vec = vec![T::zero(); n];
    let mut rho_prev = T::zero();

    let bs = b.as_slice();

    for k in 0..n_iter {
        // r = b - A x
        a.apply(x, &mut ax);
        let axs = ax.as_slice();

        if k == 0 {
            // d₀ = θ · M⁻¹ · r₀
            for i in 0..n {
                d_vec[i] = theta * diag_inv[i] * (bs[i] - axs[i]);
            }
            rho_prev = T::one();
        } else {
            let rho = T::one() / (T::one() - qdsa * rho_prev);
            for i in 0..n {
                d_vec[i] = rho * (two_theta * diag_inv[i] * (bs[i] - axs[i])
                           + (rho - T::one()) * d_vec[i]);
            }
            rho_prev = rho;
        }

        let xs = x.as_mut_slice();
        for i in 0..n { xs[i] += d_vec[i]; }
    }
}
