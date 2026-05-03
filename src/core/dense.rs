//! Dense matrix type supporting real and complex scalars.
//!
//! [`DenseMatrix<T>`] is a heap-allocated, row-major rectangular matrix.
//! Supported element types: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
//!
//! * **Real matrices** (`T: Scalar`) — GEMV uses SIMD AVX2, or CBLAS when the
//!   `blas` feature is enabled.
//! * **Complex matrices** (`T = Complex<U>`) — GEMV uses an optimised scalar
//!   loop, or CBLAS `zgemv`/`cgemv` with `blas`.
//!
//! Both types implement [`LinearOperator`] and [`TransposeOperator`], so they
//! can be passed directly into any Krylov solver.

use num_complex::Complex;
use num_traits::Zero;
use super::scalar::{Scalar, ComplexScalar};
use super::vector::{DenseVec, Vector};
use super::operator::{LinearOperator, TransposeOperator};

// ─── DenseMatrix<T> ──────────────────────────────────────────────────────────

/// Heap-allocated dense matrix stored in row-major (C) order.
///
/// `T` can be `f32`, `f64`, `Complex<f32>`, or `Complex<f64>`.
///
/// # Examples
/// ```
/// use linger::DenseMatrix;
/// use num_complex::Complex;
///
/// // Real 2×3 matrix
/// let a = DenseMatrix::<f64>::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
/// let x = linger::DenseVec::from_vec(vec![1.0_f64, 1.0, 1.0]);
/// let mut y = linger::DenseVec::zeros(2);
/// a.apply_real(&x, &mut y);  // y = A*x
/// assert!((y[0] - 6.0).abs() < 1e-12);
///
/// // Complex 2×2 impedance matrix
/// let z = DenseMatrix::<Complex<f64>>::from_fn(2, 2, |i, j| {
///     Complex::new((i + j) as f64, (i as f64) - (j as f64))
/// });
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix<T> {
    nrows: usize,
    ncols: usize,
    /// Row-major storage: element `(i, j)` is at `data[i * ncols + j]`.
    data: Vec<T>,
}

// ─── Constructors + base ops (all ComplexScalar types) ───────────────────────

impl<T: ComplexScalar> DenseMatrix<T> {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create a zero matrix of size `nrows × ncols`.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols, data: vec![T::zero(); nrows * ncols] }
    }

    /// Create a matrix from a flat row-major buffer.
    ///
    /// # Panics
    /// Panics if `data.len() != nrows * ncols`.
    pub fn from_vec(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), nrows * ncols, "DenseMatrix::from_vec: length mismatch");
        Self { nrows, ncols, data }
    }

    /// Create a matrix from a generating function `f(row, col) -> T`.
    pub fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                data.push(f(i, j));
            }
        }
        Self { nrows, ncols, data }
    }

    // ── Dimensions & access ──────────────────────────────────────────────────

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize { self.nrows }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize { self.ncols }

    /// Read-only view of the flat row-major buffer.
    #[inline]
    pub fn as_slice(&self) -> &[T] { &self.data }

    /// Mutable view of the flat row-major buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] { &mut self.data }

    /// Read-only access to element `(i, j)`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[i * self.ncols + j]
    }

    /// Mutable reference to element `(i, j)`.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.ncols + j]
    }

    // ── Frobenius norm ────────────────────────────────────────────────────────

    /// Frobenius norm `‖A‖_F = √(Σ |aᵢⱼ|²)`.
    ///
    /// Returns a **real** value for both real and complex matrices.
    pub fn frobenius_norm(&self) -> T::Real {
        let ss = self.data.iter().fold(T::Real::zero(), |acc, v| {
            let a = v.abs();
            acc + a * a
        });
        ss.sqrt()
    }

    // ── Transpose ────────────────────────────────────────────────────────────

    /// Returns `Aᵀ` (conjugate-transpose / Hermitian adjoint for complex).
    ///
    /// For real matrices this is the ordinary transpose.
    /// For complex matrices each element is conjugated: `Aᴴ[j,i] = conj(A[i,j])`.
    pub fn adjoint(&self) -> DenseMatrix<T> {
        let mut out = vec![T::zero(); self.nrows * self.ncols];
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                out[j * self.nrows + i] = self.data[i * self.ncols + j].conj();
            }
        }
        DenseMatrix { nrows: self.ncols, ncols: self.nrows, data: out }
    }

    /// Returns `Aᵀ` without conjugation (plain transpose, even for complex).
    pub fn transpose(&self) -> DenseMatrix<T> {
        let mut out = vec![T::zero(); self.nrows * self.ncols];
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                out[j * self.nrows + i] = self.data[i * self.ncols + j];
            }
        }
        DenseMatrix { nrows: self.ncols, ncols: self.nrows, data: out }
    }

    // ── GEMM (scalar, works for both real and complex) ────────────────────────

    /// Dense matrix-matrix product: returns `alpha * A * B` as a new matrix.
    pub fn gemm(&self, alpha: T, b: &DenseMatrix<T>) -> DenseMatrix<T> {
        assert_eq!(self.ncols, b.nrows, "DenseMatrix::gemm: inner dimension mismatch");
        let m = self.nrows;
        let k = self.ncols;
        let n = b.ncols;
        let mut c = vec![T::zero(); m * n];
        for i in 0..m {
            for p in 0..k {
                let a_ip = alpha * self.data[i * k + p];
                for j in 0..n {
                    c[i * n + j] += a_ip * b.data[p * n + j];
                }
            }
        }
        DenseMatrix { nrows: m, ncols: n, data: c }
    }

}

// ── Column access for real matrices ──────────────────────────────────────────

impl<T: Scalar> DenseMatrix<T> {
    /// Copy column `j` into a new `DenseVec`.
    pub fn col(&self, j: usize) -> DenseVec<T> {
        assert!(j < self.ncols, "DenseMatrix::col: column index out of range");
        let mut v = vec![T::zero(); self.nrows];
        for i in 0..self.nrows {
            v[i] = self.data[i * self.ncols + j];
        }
        DenseVec::<T>::from_vec(v)
    }

    /// Write `DenseVec` `v` into column `j`.
    pub fn set_col(&mut self, j: usize, v: &DenseVec<T>) {
        assert!(j < self.ncols, "DenseMatrix::set_col: column index out of range");
        assert_eq!(v.len(), self.nrows, "DenseMatrix::set_col: length mismatch");
        for i in 0..self.nrows {
            self.data[i * self.ncols + j] = v.as_slice()[i];
        }
    }
}

// ─── Index ───────────────────────────────────────────────────────────────────

impl<T: Copy> std::ops::Index<(usize, usize)> for DenseMatrix<T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i * self.ncols + j]
    }
}

impl<T: Copy> std::ops::IndexMut<(usize, usize)> for DenseMatrix<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.data[i * self.ncols + j]
    }
}

// ─── Real SIMD / BLAS GEMV ───────────────────────────────────────────────────

impl<T: Scalar> DenseMatrix<T> {
    /// `y += alpha * A * x` — SIMD-accelerated for real scalars.
    ///
    /// On x86_64 with AVX2: vectorised over the row dot-products.
    /// With `blas` feature: delegates to `cblas_dgemv` / `cblas_sgemv`.
    #[inline]
    pub fn gemv_add(&self, alpha: T, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        assert_eq!(x.len(), self.ncols, "DenseMatrix::gemv_add: x length mismatch");
        assert_eq!(y.len(), self.nrows, "DenseMatrix::gemv_add: y length mismatch");
        #[cfg(feature = "blas")]
        {
            crate::blas_backend::real_gemv_add(alpha, &self.data, self.nrows, self.ncols,
                                               x.as_slice(), y.as_mut_slice());
            return;
        }
        #[cfg(not(feature = "blas"))]
        crate::simd::dense_ops::simd_gemv(alpha, &self.data, self.nrows, self.ncols,
                                          x.as_slice(), y.as_mut_slice());
    }

    /// `y += alpha * Aᵀ * x` — SIMD-accelerated for real scalars.
    #[inline]
    pub fn gemv_t_add(&self, alpha: T, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        assert_eq!(x.len(), self.nrows, "DenseMatrix::gemv_t_add: x length mismatch");
        assert_eq!(y.len(), self.ncols, "DenseMatrix::gemv_t_add: y length mismatch");
        #[cfg(feature = "blas")]
        {
            crate::blas_backend::real_gemv_t_add(alpha, &self.data, self.nrows, self.ncols,
                                                 x.as_slice(), y.as_mut_slice());
            return;
        }
        #[cfg(not(feature = "blas"))]
        crate::simd::dense_ops::simd_gemv_t(alpha, &self.data, self.nrows, self.ncols,
                                            x.as_slice(), y.as_mut_slice());
    }

    /// Convenience: overwrites `y = A * x`.
    #[inline]
    pub fn apply_real(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        assert_eq!(x.len(), self.ncols, "DenseMatrix::apply_real: x length mismatch");
        if y.len() != self.nrows { *y = DenseVec::<T>::zeros(self.nrows); }
        y.fill(T::zero());
        self.gemv_add(T::one(), x, y);
    }
}

// ─── Complex GEMV ────────────────────────────────────────────────────────────

impl<T: Scalar> DenseMatrix<Complex<T>> {
    /// `y += alpha * A * x` for complex matrices.
    ///
    /// With `blas` feature: delegates to `cblas_zgemv` / `cblas_cgemv`.
    /// Otherwise: optimised scalar loop.
    #[inline]
    pub fn gemv_add(&self, alpha: Complex<T>, x: &DenseVec<Complex<T>>,
                    y: &mut DenseVec<Complex<T>>) {
        assert_eq!(x.len(), self.ncols, "DenseMatrix::gemv_add (complex): x length mismatch");
        assert_eq!(y.len(), self.nrows, "DenseMatrix::gemv_add (complex): y length mismatch");
        #[cfg(feature = "blas")]
        {
            crate::blas_backend::complex_gemv_add(alpha, &self.data, self.nrows, self.ncols,
                                                  x.as_slice(), y.as_mut_slice());
            return;
        }
        #[cfg(not(feature = "blas"))]
        {
            let xd = x.as_slice();
            let yd = y.as_mut_slice();
            for i in 0..self.nrows {
                let row = &self.data[i * self.ncols .. (i + 1) * self.ncols];
                let mut s = Complex::zero();
                for j in 0..self.ncols { s += row[j] * xd[j]; }
                yd[i] += alpha * s;
            }
        }
    }

    /// `y += alpha * Aᵀ * x` (no conjugation) for complex matrices.
    #[inline]
    pub fn gemv_t_add(&self, alpha: Complex<T>, x: &DenseVec<Complex<T>>,
                      y: &mut DenseVec<Complex<T>>) {
        assert_eq!(x.len(), self.nrows, "DenseMatrix::gemv_t_add (complex): x length mismatch");
        assert_eq!(y.len(), self.ncols, "DenseMatrix::gemv_t_add (complex): y length mismatch");
        let xd = x.as_slice();
        let yd = y.as_mut_slice();
        for i in 0..self.nrows {
            let ax = alpha * xd[i];
            let row = &self.data[i * self.ncols .. (i + 1) * self.ncols];
            for j in 0..self.ncols { yd[j] += row[j] * ax; }
        }
    }

    /// `y += alpha * Aᴴ * x` (conjugate-transpose / Hermitian adjoint).
    #[inline]
    pub fn gemv_h_add(&self, alpha: Complex<T>, x: &DenseVec<Complex<T>>,
                      y: &mut DenseVec<Complex<T>>) {
        assert_eq!(x.len(), self.nrows, "DenseMatrix::gemv_h_add: x length mismatch");
        assert_eq!(y.len(), self.ncols, "DenseMatrix::gemv_h_add: y length mismatch");
        let xd = x.as_slice();
        let yd = y.as_mut_slice();
        for i in 0..self.nrows {
            let ax = alpha * xd[i];
            let row = &self.data[i * self.ncols .. (i + 1) * self.ncols];
            for j in 0..self.ncols { yd[j] += row[j].conj() * ax; }
        }
    }

    /// Convenience: overwrites `y = A * x`.
    #[inline]
    pub fn apply_complex(&self, x: &DenseVec<Complex<T>>, y: &mut DenseVec<Complex<T>>) {
        assert_eq!(x.len(), self.ncols, "DenseMatrix::apply_complex: x length mismatch");
        if y.len() != self.nrows { *y = vec![Complex::new(T::zero(), T::zero()); self.nrows].into(); }
        for v in y.as_mut_slice().iter_mut() { *v = Complex::new(T::zero(), T::zero()); }
        self.gemv_add(Complex::new(T::one(), T::zero()), x, y);
    }

    /// Copy column `j` into a new `DenseVec<Complex<T>>`.
    pub fn col(&self, j: usize) -> DenseVec<Complex<T>> {
        assert!(j < self.ncols, "DenseMatrix::col (complex): column index out of range");
        let mut v = vec![Complex::new(T::zero(), T::zero()); self.nrows];
        for i in 0..self.nrows {
            v[i] = self.data[i * self.ncols + j];
        }
        v.into()
    }

    /// Write `DenseVec<Complex<T>>` into column `j`.
    pub fn set_col(&mut self, j: usize, v: &DenseVec<Complex<T>>) {
        assert!(j < self.ncols, "DenseMatrix::set_col (complex): column index out of range");
        assert_eq!(v.len(), self.nrows, "DenseMatrix::set_col (complex): length mismatch");
        for i in 0..self.nrows {
            self.data[i * self.ncols + j] = v.as_slice()[i];
        }
    }
}

// ─── LinearOperator (real) ───────────────────────────────────────────────────

impl<T: Scalar> LinearOperator for DenseMatrix<T> {
    type Vector = DenseVec<T>;

    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        self.apply_real(x, y);
    }
    fn nrows(&self) -> usize { self.nrows }
    fn ncols(&self) -> usize { self.ncols }
}

impl<T: Scalar> TransposeOperator for DenseMatrix<T> {
    fn apply_transpose(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        assert_eq!(x.len(), self.nrows, "DenseMatrix::apply_transpose: x length mismatch");
        if y.len() != self.ncols { *y = DenseVec::<T>::zeros(self.ncols); }
        y.fill(T::zero());
        self.gemv_t_add(T::one(), x, y);
    }
}

// ─── LinearOperator (complex) ────────────────────────────────────────────────

impl<T: Scalar> LinearOperator for DenseMatrix<Complex<T>> {
    type Vector = DenseVec<Complex<T>>;

    fn apply(&self, x: &DenseVec<Complex<T>>, y: &mut DenseVec<Complex<T>>) {
        self.apply_complex(x, y);
    }
    fn nrows(&self) -> usize { self.nrows }
    fn ncols(&self) -> usize { self.ncols }
}

impl<T: Scalar> TransposeOperator for DenseMatrix<Complex<T>> {
    fn apply_transpose(&self, x: &DenseVec<Complex<T>>, y: &mut DenseVec<Complex<T>>) {
        assert_eq!(x.len(), self.nrows);
        if y.len() != self.ncols {
            *y = vec![Complex::new(T::zero(), T::zero()); self.ncols].into();
        }
        for v in y.as_mut_slice().iter_mut() { *v = Complex::new(T::zero(), T::zero()); }
        self.gemv_h_add(Complex::new(T::one(), T::zero()), x, y);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    // ── Real matrix tests ─────────────────────────────────────────────────────

    #[test]
    fn identity_apply() {
        let eye: DenseMatrix<f64> = DenseMatrix::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let x = DenseVec::from_vec(vec![1.0, 2.0, 3.0]);
        let mut y = DenseVec::zeros(3);
        eye.apply(&x, &mut y);
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 2.0).abs() < 1e-12);
        assert!((y[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn gemv_add_accumulates() {
        let a: DenseMatrix<f64> = DenseMatrix::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
        let x = DenseVec::from_vec(vec![1.0, 1.0, 1.0]);
        let mut y = DenseVec::from_vec(vec![10.0, 10.0]);
        a.gemv_add(2.0, &x, &mut y);
        assert!((y[0] - 22.0).abs() < 1e-12, "y[0]={}", y[0]);
        assert!((y[1] - 40.0).abs() < 1e-12, "y[1]={}", y[1]);
    }

    #[test]
    fn gemv_t_add_correct() {
        let a: DenseMatrix<f64> = DenseMatrix::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f64);
        let x = DenseVec::from_vec(vec![1.0, 1.0, 1.0]);
        let mut y = DenseVec::zeros(2);
        a.gemv_t_add(1.0, &x, &mut y);
        assert!((y[0] - 9.0).abs() < 1e-12, "y[0]={}", y[0]);
        assert!((y[1] - 12.0).abs() < 1e-12, "y[1]={}", y[1]);
    }

    #[test]
    fn transpose_correct() {
        let a: DenseMatrix<f64> = DenseMatrix::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
        let at = a.transpose();
        assert_eq!(at.nrows(), 3);
        assert_eq!(at.ncols(), 2);
        for i in 0..2 {
            for j in 0..3 {
                assert!((a[(i, j)] - at[(j, i)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn frobenius_norm_real_correct() {
        let a: DenseMatrix<f64> = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        let n: f64 = a.frobenius_norm();
        assert!((n - 10.0_f64.sqrt()).abs() < 1e-12, "norm={n}");
    }

    #[test]
    fn gemm_correct() {
        let a: DenseMatrix<f64> = DenseMatrix::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
        let b: DenseMatrix<f64> = DenseMatrix::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f64);
        let c = a.gemm(1.0, &b);
        assert!((c[(0, 0)] - 22.0).abs() < 1e-12);
        assert!((c[(0, 1)] - 28.0).abs() < 1e-12);
        assert!((c[(1, 0)] - 49.0).abs() < 1e-12);
        assert!((c[(1, 1)] - 64.0).abs() < 1e-12);
    }

    #[test]
    fn apply_transpose_real_correct() {
        let a: DenseMatrix<f64> = DenseMatrix::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f64);
        let x = DenseVec::from_vec(vec![1.0, 1.0, 1.0]);
        let mut y = DenseVec::zeros(2);
        a.apply_transpose(&x, &mut y);
        assert!((y[0] - 9.0).abs() < 1e-12, "y[0]={}", y[0]);
        assert!((y[1] - 12.0).abs() < 1e-12, "y[1]={}", y[1]);
    }

    #[test]
    fn col_set_col() {
        let mut a: DenseMatrix<f64> = DenseMatrix::zeros(3, 2);
        let c = DenseVec::from_vec(vec![1.0, 2.0, 3.0]);
        a.set_col(1, &c);
        let out = a.col(1);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 2.0).abs() < 1e-12);
        assert!((out[2] - 3.0).abs() < 1e-12);
    }

    // ── Complex matrix tests ──────────────────────────────────────────────────

    #[test]
    fn complex_zeros_and_from_fn() {
        let a: DenseMatrix<Complex<f64>> = DenseMatrix::from_fn(2, 2, |i, j| {
            Complex::new(i as f64, j as f64)
        });
        assert_eq!(a[(0, 1)], Complex::new(0.0, 1.0));
        assert_eq!(a[(1, 0)], Complex::new(1.0, 0.0));
    }

    #[test]
    fn complex_gemv_add() {
        // Z = [[1+0i, 0+1i], [0-1i, 2+0i]], x = [1+0i, 0+1i]
        // Z*x = [(1)(1)+(0+i)(i), (0-i)(1)+(2)(i)] = [1+i²,  -i+2i] = [1-1, i] = [0, i]
        let z: DenseMatrix<Complex<f64>> = DenseMatrix::from_vec(2, 2, vec![
            Complex::new(1.0,  0.0), Complex::new(0.0,  1.0),
            Complex::new(0.0, -1.0), Complex::new(2.0,  0.0),
        ]);
        let x: DenseVec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)].into();
        let mut y: DenseVec<Complex<f64>> = vec![Complex::new(0.0_f64, 0.0_f64); 2].into();
        z.apply(&x, &mut y);
        assert!((y[0] - Complex::new(0.0, 0.0)).norm() < 1e-12, "y[0]={:?}", y[0]);
        assert!((y[1] - Complex::new(0.0, 1.0)).norm() < 1e-12, "y[1]={:?}", y[1]);
    }

    #[test]
    fn complex_adjoint() {
        // A = [[1+2i, 3+4i]], A* = [[1-2i], [3-4i]]
        let a: DenseMatrix<Complex<f64>> = DenseMatrix::from_vec(1, 2, vec![
            Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
        ]);
        let ah = a.adjoint();
        assert_eq!(ah.nrows(), 2);
        assert_eq!(ah.ncols(), 1);
        assert_eq!(ah[(0, 0)], Complex::new(1.0, -2.0));
        assert_eq!(ah[(1, 0)], Complex::new(3.0, -4.0));
    }

    #[test]
    fn complex_frobenius_norm() {
        // A = [[3+4i]], |a|² = 9+16=25, ‖A‖_F = 5
        let a: DenseMatrix<Complex<f64>> = DenseMatrix::from_vec(1, 1, vec![
            Complex::new(3.0, 4.0),
        ]);
        let n: f64 = a.frobenius_norm();
        assert!((n - 5.0).abs() < 1e-12, "norm={n}");
    }

    #[test]
    fn complex_apply_transpose_is_hermitian() {
        // For a Hermitian matrix A = A*, apply_transpose should give A* x = A* x
        let a: DenseMatrix<Complex<f64>> = DenseMatrix::from_vec(2, 2, vec![
            Complex::new(2.0, 0.0), Complex::new(1.0,  1.0),
            Complex::new(1.0, -1.0), Complex::new(3.0, 0.0),
        ]);
        let x: DenseVec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)].into();
        let mut y: DenseVec<Complex<f64>> = vec![Complex::new(0.0_f64, 0.0_f64); 2].into();
        a.apply_transpose(&x, &mut y); // y = A^H x
        // A^H = A (it's Hermitian), A*x = [[2,1+i],[1-i,3]] * [1,i]
        //   y[0] = 2*1 + (1+i)*i = 2 + i + i² = 2+i-1 = 1+i
        //   y[1] = (1-i)*1 + 3*i = 1-i+3i = 1+2i
        assert!((y[0] - Complex::new(1.0, 1.0)).norm() < 1e-12, "y[0]={:?}", y[0]);
        assert!((y[1] - Complex::new(1.0, 2.0)).norm() < 1e-12, "y[1]={:?}", y[1]);
    }

    #[test]
    fn complex_gemm() {
        // A = [[1+i, 0], [0, 1-i]]  (2×2 diagonal complex)
        // A * A = [[2i, 0], [0, -2i]]
        let a: DenseMatrix<Complex<f64>> = DenseMatrix::from_vec(2, 2, vec![
            Complex::new(1.0, 1.0), Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new(1.0, -1.0),
        ]);
        let c = a.gemm(Complex::new(1.0, 0.0), &a);
        assert!((c[(0, 0)] - Complex::new(0.0, 2.0)).norm() < 1e-12);
        assert!((c[(1, 1)] - Complex::new(0.0, -2.0)).norm() < 1e-12);
        assert!(c[(0, 1)].norm() < 1e-12);
        assert!(c[(1, 0)].norm() < 1e-12);
    }
}
