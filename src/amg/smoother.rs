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
}

/// Apply `n_sweeps` pre-smoothing iterations: `x ← smooth(A, x, b)`.
pub fn smooth<T: Scalar>(
    a:       &CsrMatrix<T>,
    x:       &mut DenseVec<T>,
    b:       &DenseVec<T>,
    smoother: &SmootherType,
    n_sweeps: usize,
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
