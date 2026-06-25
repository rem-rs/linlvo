//! CBLAS backend for `DenseMatrix` GEMV and GEMM operations.
//!
//! Activated by the `blas` feature flag.  Users select the BLAS implementation
//! via sub-features:
//! - `blas-openblas`  — OpenBLAS (Linux / macOS)
//! - `blas-accelerate` — macOS Accelerate framework
//! - `blas-netlib`    — reference BLAS
//!
//! ## Safety
//! All functions here call raw C BLAS routines via `cblas-sys`.  Slice lengths
//! are checked by the callers in `dense.rs` before reaching these functions.

#![cfg(feature = "blas")]

use cblas_sys::{
    cblas_dgemv, cblas_sgemv,
    cblas_zgemv, cblas_cgemv,
    CblasRowMajor, CblasNoTrans, CblasTrans,
    CBLAS_LAYOUT, CBLAS_TRANSPOSE,
};
use num_complex::Complex;
use crate::core::scalar::Scalar;

// ─── Real GEMV ───────────────────────────────────────────────────────────────

/// Dispatches `y += alpha * A * x` to `cblas_dgemv` or `cblas_sgemv`.
pub fn real_gemv_add<T: Scalar>(
    alpha: T, a: &[T], nrows: usize, ncols: usize, x: &[T], y: &mut [T],
) {
    let m = nrows as i32;
    let n = ncols as i32;
    let layout = CBLAS_LAYOUT::CblasRowMajor;
    let trans = CBLAS_TRANSPOSE::CblasNoTrans;
    if std::mem::size_of::<T>() == 8 {
        let alpha_f = unsafe { *(&alpha as *const T as *const f64) };
        let a_f = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
        let x_f = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let y_f = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
        unsafe {
            cblas_dgemv(layout, trans, m, n,
                        alpha_f, a_f.as_ptr(), n,
                        x_f.as_ptr(), 1,
                        1.0_f64, y_f.as_mut_ptr(), 1);
        }
    } else if std::mem::size_of::<T>() == 4 {
        // f32 path
        let alpha_f = unsafe { *(&alpha as *const T as *const f32) };
        let a_f = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
        let x_f = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let y_f = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, y.len()) };
        unsafe {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n,
                        alpha_f, a_f.as_ptr(), n,
                        x_f.as_ptr(), 1,
                        1.0_f32, y_f.as_mut_ptr(), 1);
        }
    } else {
        // Scalar fallback (should not happen for f32/f64)
        crate::simd::dense_ops::simd_gemv(alpha, a, nrows, ncols, x, y);
    }
}

/// Dispatches `y += alpha * Aᵀ * x` to `cblas_dgemv` or `cblas_sgemv`.
pub fn real_gemv_t_add<T: Scalar>(
    alpha: T, a: &[T], nrows: usize, ncols: usize, x: &[T], y: &mut [T],
) {
    let m = nrows as i32;
    let n = ncols as i32;
    let layout = CBLAS_LAYOUT::CblasRowMajor;
    let trans = CBLAS_TRANSPOSE::CblasTrans;
    if std::mem::size_of::<T>() == 8 {
        let alpha_f = unsafe { *(&alpha as *const T as *const f64) };
        let a_f = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
        let x_f = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let y_f = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
        unsafe {
            cblas_dgemv(CblasRowMajor, CblasTrans, m, n,
                        alpha_f, a_f.as_ptr(), n,
                        x_f.as_ptr(), 1,
                        1.0_f64, y_f.as_mut_ptr(), 1);
        }
    } else if std::mem::size_of::<T>() == 4 {
        let alpha_f = unsafe { *(&alpha as *const T as *const f32) };
        let a_f = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
        let x_f = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let y_f = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, y.len()) };
        unsafe {
            cblas_sgemv(CblasRowMajor, CblasTrans, m, n,
                        alpha_f, a_f.as_ptr(), n,
                        x_f.as_ptr(), 1,
                        1.0_f32, y_f.as_mut_ptr(), 1);
        }
    } else {
        crate::simd::dense_ops::simd_gemv_t(alpha, a, nrows, ncols, x, y);
    }
}

// ─── Complex GEMV ────────────────────────────────────────────────────────────

/// Dispatches `y += alpha * A * x` to `cblas_zgemv` (f64) or `cblas_cgemv` (f32).
///
/// CBLAS `zgemv` takes `alpha` and `beta` as void pointers to two-element arrays
/// [re, im].
pub fn complex_gemv_add<T: Scalar>(
    alpha: Complex<T>,
    a: &[Complex<T>],
    nrows: usize,
    ncols: usize,
    x: &[Complex<T>],
    y: &mut [Complex<T>],
) {
    let m = nrows as i32;
    let n = ncols as i32;
    let layout = CBLAS_LAYOUT::CblasRowMajor;
    let trans = CBLAS_TRANSPOSE::CblasNoTrans;
    if std::mem::size_of::<T>() == 8 {
        // Complex<f64> path — cblas_zgemv
        let alpha64 = [alpha.re.to_f64().unwrap_or(0.0), alpha.im.to_f64().unwrap_or(0.0)];
        let beta64  = [1.0_f64, 0.0_f64]; // beta = 1 (accumulate)
        // SAFETY: Complex<f64> has the same layout as two consecutive f64s
        let a_ptr = a.as_ptr() as *const num_complex::Complex<f64>;
        let x_ptr = x.as_ptr() as *const num_complex::Complex<f64>;
        let y_ptr = y.as_mut_ptr() as *mut num_complex::Complex<f64>;
        unsafe {
            cblas_zgemv(layout, trans, m, n,
                        alpha64.as_ptr() as *const cblas_sys::c_double_complex,
                        a_ptr as *const cblas_sys::c_double_complex, n,
                        x_ptr as *const cblas_sys::c_double_complex, 1,
                        beta64.as_ptr() as *const cblas_sys::c_double_complex,
                        y_ptr as *mut cblas_sys::c_double_complex, 1);
        }
    } else if std::mem::size_of::<T>() == 4 {
        // Complex<f32> path — cblas_cgemv
        let alpha32 = [alpha.re.to_f32().unwrap_or(0.0), alpha.im.to_f32().unwrap_or(0.0)];
        let beta32  = [1.0_f32, 0.0_f32];
        let a_ptr = a.as_ptr() as *const num_complex::Complex<f32>;
        let x_ptr = x.as_ptr() as *const num_complex::Complex<f32>;
        let y_ptr = y.as_mut_ptr() as *mut num_complex::Complex<f32>;
        unsafe {
            cblas_cgemv(layout, trans, m, n,
                        alpha32.as_ptr() as *const cblas_sys::c_float_complex,
                        a_ptr as *const cblas_sys::c_float_complex, n,
                        x_ptr as *const cblas_sys::c_float_complex, 1,
                        beta32.as_ptr() as *const cblas_sys::c_float_complex,
                        y_ptr as *mut cblas_sys::c_float_complex, 1);
        }
    } else {
        // Scalar fallback
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

// Helper: f64 to f32 conversion used in BLAS dispatch
use num_traits::ToPrimitive;
