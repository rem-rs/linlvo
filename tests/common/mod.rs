//! Shared test utilities based on the Method of Manufactured Solutions (MMS).
//!
//! Each helper constructs a linear system `A · x_exact = b` with a known
//! analytic solution so that solver convergence can be verified precisely.

#![allow(dead_code)]

use linger::{
    sparse::{CooMatrix, CsrMatrix},
    Scalar,
};
use num_traits::Float;

// ─── 1-D Poisson ─────────────────────────────────────────────────────────────

/// Construct the 1-D Poisson system on `n` interior points.
///
/// Matrix: tridiagonal `[−1, 2, −1]` (n × n).
/// Exact solution: `x_exact[i] = sin(π · (i+1) / (n+1))`.
/// RHS: `b = A · x_exact`.
///
/// Returns `(A, x_exact, b)`.
pub fn make_poisson_1d<T: Scalar>(n: usize) -> (CsrMatrix<T>, Vec<T>, Vec<T>) {
    assert!(n >= 2, "make_poisson_1d: need at least 2 interior points");

    let mut coo = CooMatrix::with_capacity(n, n, 3 * n - 2);
    for i in 0..n {
        coo.push(i, i, T::from_f64(2.0));
        if i > 0     { coo.push(i, i - 1, T::from_f64(-1.0)); }
        if i < n - 1 { coo.push(i, i + 1, T::from_f64(-1.0)); }
    }
    let a = CsrMatrix::from_coo(&coo);

    let pi = T::from_f64(std::f64::consts::PI);
    let n1 = T::from_f64((n + 1) as f64);
    let x_exact: Vec<T> = (0..n)
        .map(|i| (pi * T::from_f64((i + 1) as f64) / n1).sin())
        .collect();

    let mut b = vec![T::zero(); n];
    a.spmv(&x_exact, &mut b);

    (a, x_exact, b)
}

// ─── 2-D Poisson ─────────────────────────────────────────────────────────────

/// Construct the 2-D Poisson system on an `nx × ny` interior grid.
///
/// Uses the standard 5-point stencil with natural (row-major) ordering.
/// The global DOF index for grid point `(i, j)` is `i * ny + j`.
///
/// Exact solution: `x_exact[i,j] = sin(π·(i+1)/(nx+1)) · sin(π·(j+1)/(ny+1))`.
/// RHS: `b = A · x_exact`.
///
/// Returns `(A, x_exact, b)`.
pub fn make_poisson_2d<T: Scalar>(
    nx: usize,
    ny: usize,
) -> (CsrMatrix<T>, Vec<T>, Vec<T>) {
    assert!(nx >= 2 && ny >= 2, "make_poisson_2d: need at least 2 points per dimension");

    let n = nx * ny;
    let mut coo = CooMatrix::with_capacity(n, n, 5 * n);

    let dof = |i: usize, j: usize| i * ny + j;

    for i in 0..nx {
        for j in 0..ny {
            let row = dof(i, j);
            coo.push(row, row, T::from_f64(4.0)); // self
            if i > 0     { coo.push(row, dof(i - 1, j), T::from_f64(-1.0)); }
            if i < nx-1  { coo.push(row, dof(i + 1, j), T::from_f64(-1.0)); }
            if j > 0     { coo.push(row, dof(i, j - 1), T::from_f64(-1.0)); }
            if j < ny-1  { coo.push(row, dof(i, j + 1), T::from_f64(-1.0)); }
        }
    }
    let a = CsrMatrix::from_coo(&coo);

    let pi  = T::from_f64(std::f64::consts::PI);
    let nx1 = T::from_f64((nx + 1) as f64);
    let ny1 = T::from_f64((ny + 1) as f64);
    let x_exact: Vec<T> = (0..nx)
        .flat_map(|i| {
            let si = (pi * T::from_f64((i + 1) as f64) / nx1).sin();
            (0..ny).map(move |j| {
                let sj = (pi * T::from_f64((j + 1) as f64) / ny1).sin();
                si * sj
            })
        })
        .collect();

    let mut b = vec![T::zero(); n];
    a.spmv(&x_exact, &mut b);

    (a, x_exact, b)
}

// ─── Non-symmetric convection-diffusion ──────────────────────────────────────

/// Construct a 1-D convection-diffusion system with Péclet number `peclet`.
///
/// Upwind discretisation on `n` interior points (uniform mesh `h = 1/(n+1)`):
///   `−u'' + Pe·u' = f`
///
/// Stencil (upwind for `Pe > 0`):
///   `a[i,i]   =  2 + Pe·h`
///   `a[i,i-1] = −1 − Pe·h`
///   `a[i,i+1] = −1`
///
/// Exact solution: `x_exact[i] = sin(π·(i+1)/(n+1))`.
///
/// Returns `(A, x_exact, b)`.
pub fn make_nonsymmetric_convdiff<T: Scalar>(
    n: usize,
    peclet: T,
) -> (CsrMatrix<T>, Vec<T>, Vec<T>) {
    assert!(n >= 2, "make_nonsymmetric_convdiff: need at least 2 interior points");

    let h   = T::one() / T::from_f64((n + 1) as f64);
    let peh = peclet * h;

    let mut coo = CooMatrix::with_capacity(n, n, 3 * n - 2);
    for i in 0..n {
        coo.push(i, i, T::from_f64(2.0) + peh);         // diagonal
        if i > 0 {
            coo.push(i, i - 1, -(T::one() + peh));       // lower
        }
        if i < n - 1 {
            coo.push(i, i + 1, -T::one());               // upper
        }
    }
    let a = CsrMatrix::from_coo(&coo);

    let pi = T::from_f64(std::f64::consts::PI);
    let n1 = T::from_f64((n + 1) as f64);
    let x_exact: Vec<T> = (0..n)
        .map(|i| (pi * T::from_f64((i + 1) as f64) / n1).sin())
        .collect();

    let mut b = vec![T::zero(); n];
    a.spmv(&x_exact, &mut b);

    (a, x_exact, b)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Compute the relative residual  `‖b − A·x‖₂ / ‖b‖₂`.
pub fn relative_residual<T: Scalar>(
    a:  &CsrMatrix<T>,
    x:  &[T],
    b:  &[T],
) -> T {
    assert_eq!(a.nrows(), b.len());
    assert_eq!(a.ncols(), x.len());

    let mut r = vec![T::zero(); b.len()];
    a.spmv(x, &mut r);
    let norm_b = b.iter().fold(T::zero(), |acc, &v| acc + v * v).sqrt();
    let norm_r = r
        .iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (&ri, &bi)| {
            let d = ri - bi;
            acc + d * d
        })
        .sqrt();

    if norm_b == T::zero() { norm_r } else { norm_r / norm_b }
}
