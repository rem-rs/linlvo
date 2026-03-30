//! Standalone sparse BLAS-level operations.
//!
//! These thin wrappers over [`CsrMatrix`] / [`CscMatrix`] methods provide a
//! functional interface that is easy to parallelize in Sprint 5.

use crate::core::scalar::Scalar;
use crate::sparse::{csc::CscMatrix, csr::CsrMatrix};

// ─── SpMV ────────────────────────────────────────────────────────────────────

/// Compute  `y ← A · x`  using a CSR matrix.
///
/// # Panics
/// Panics if dimensions are incompatible.
pub fn spmv_csr<T: Scalar>(mat: &CsrMatrix<T>, x: &[T], y: &mut [T]) {
    mat.spmv(x, y);
}

/// Compute  `y ← α·A·x + β·y`  using a CSR matrix.
pub fn spmv_csr_add<T: Scalar>(mat: &CsrMatrix<T>, alpha: T, x: &[T], beta: T, y: &mut [T]) {
    mat.spmv_add(alpha, x, beta, y);
}

/// Compute  `y ← A · x`  using a CSC matrix.
pub fn spmv_csc<T: Scalar>(mat: &CscMatrix<T>, x: &[T], y: &mut [T]) {
    mat.spmv(x, y);
}

// ─── Dense-vector AXPY ───────────────────────────────────────────────────────

/// `y += alpha * x`  for dense slice vectors.
///
/// This is a fallback scalar implementation; the parallel version
/// (`rayon_ops::axpy`) will be added in Sprint 5.
#[inline]
pub fn axpy<T: Scalar>(alpha: T, x: &[T], y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len(), "axpy: length mismatch");
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// `y = alpha * x + beta * y`  for dense slice vectors.
#[inline]
pub fn axpby<T: Scalar>(alpha: T, x: &[T], beta: T, y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len(), "axpby: length mismatch");
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = alpha * xi + beta * *yi;
    }
}

// ─── Diagonal extraction ─────────────────────────────────────────────────────

/// Extract the main diagonal of a CSR matrix.
///
/// Returns a `Vec<T>` of length `min(nrows, ncols)`.
pub fn extract_diagonal<T: Scalar>(mat: &CsrMatrix<T>) -> Vec<T> {
    mat.diag()
}

// ─── Dot product / norm ───────────────────────────────────────────────────────

/// Euclidean inner product `<x, y>`.
#[inline]
pub fn dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    debug_assert_eq!(x.len(), y.len(), "dot: length mismatch");
    x.iter().zip(y.iter()).fold(T::zero(), |acc, (&a, &b)| acc + a * b)
}

/// Euclidean 2-norm  `√(Σ xᵢ²)`.
#[inline]
pub fn norm2<T: Scalar>(x: &[T]) -> T {
    x.iter().fold(T::zero(), |acc, &v| acc + v * v).sqrt()
}
