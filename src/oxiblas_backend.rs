//! OxiBLAS backend for `DenseMatrix` GEMV operations.
//!
//! Pure Rust alternative to the CBLAS `cblas-sys` path.
//! Activated by `blas-oxiblas` feature — no C library deps.

#![cfg(feature = "blas-oxiblas")]

use oxiblas::prelude::*;
use oxiblas::MatRef;
use num_complex::Complex;
use crate::core::scalar::Scalar;

// ── Real GEMV (non-transpose) ───────────────────────────────────────────

pub fn real_gemv_add<T: Scalar>(
    alpha: T, a: &[T], nrows: usize, ncols: usize, x: &[T], y: &mut [T],
) {
    if std::mem::size_of::<T>() == 8 {
        let a_f = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
        let x_f = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let y_f = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
        let alpha_f = unsafe { *(&alpha as *const T as *const f64) };
        let mat = unsafe { MatRef::new(a_f.as_ptr() as *const f64, nrows, ncols, ncols) };
        gemv(GemvTrans::NoTrans, alpha_f, mat, x_f, 1.0, y_f);
    } else {
        crate::simd::dense_ops::simd_gemv(alpha, a, nrows, ncols, x, y);
    }
}

// ── Real GEMV (transpose) ───────────────────────────────────────────────

pub fn real_gemv_t_add<T: Scalar>(
    alpha: T, a: &[T], nrows: usize, ncols: usize, x: &[T], y: &mut [T],
) {
    if std::mem::size_of::<T>() == 8 {
        let a_f = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
        let x_f = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let y_f = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
        let alpha_f = unsafe { *(&alpha as *const T as *const f64) };
        let mat = unsafe { MatRef::new(a_f.as_ptr() as *const f64, nrows, ncols, ncols) };
        gemv(GemvTrans::Trans, alpha_f, mat, x_f, 1.0, y_f);
    } else {
        crate::simd::dense_ops::simd_gemv_t(alpha, a, nrows, ncols, x, y);
    }
}

// ── Complex GEMV (non-transpose) ───────────────────────────────────────

pub fn complex_gemv_add<T: Scalar>(
    alpha: Complex<T>,
    a: &[Complex<T>],
    nrows: usize,
    ncols: usize,
    x: &[Complex<T>],
    y: &mut [Complex<T>],
) {
    if std::mem::size_of::<T>() == 8 {
        let a_f = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const Complex<f64>, a.len()) };
        let x_f = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const Complex<f64>, x.len()) };
        let y_f = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut Complex<f64>, y.len()) };
        let alpha_f = Complex::new(alpha.re.to_f64().unwrap_or(0.0), alpha.im.to_f64().unwrap_or(0.0));
        let mat = unsafe { MatRef::new(a_f.as_ptr() as *const Complex<f64>, nrows, ncols, ncols) };
        gemv(GemvTrans::NoTrans, alpha_f, mat, x_f, Complex::new(1.0, 0.0), y_f);
    } else {
        let xd = x;
        let yd = y;
        for i in 0..nrows {
            let row = &a[i * ncols .. (i + 1) * ncols];
            let mut s = Complex::new(T::zero(), T::zero());
            for j in 0..ncols { s += row[j] * xd[j]; }
            yd[i] += alpha * s;
        }
    }
}
