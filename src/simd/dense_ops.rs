//! SIMD-accelerated dense vector operations (AXPY, AXPBY, DOT, NORM2).
//!
//! These operations are used frequently in Krylov solvers:
//! - AXPY:  y ← α·x + y
//! - AXPBY: y ← α·x + β·y
//! - DOT:   s = xᵀy
//! - NORM2: n = ‖x‖₂
//!
//! SIMD acceleration provides 2-4x speedup for large vectors.
//! Dispatches to:
//! - x86_64: AVX2 (f64: 4-lane, f32: 8-lane) — runtime detection
//! - AArch64: NEON (f64: 2-lane, f32: 4-lane) — always available on Apple Silicon
//! - Scalar fallback otherwise.

use crate::core::scalar::Scalar;

#[cfg(target_arch = "aarch64")]
use super::aarch64;

/// Dispatch helper: reinterpret generic T slices as f64 slices and call `f`.
///
/// # Safety
/// Caller must ensure `std::mem::size_of::<T>() == 8` (i.e., T is f64-sized).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn with_f64<T, R>(
    alpha: T,
    x: &[T],
    y: &mut [T],
    f: unsafe fn(f64, &[f64], &mut [f64]) -> R,
) -> R {
    let a = *(&alpha as *const T as *const f64);
    let xf = std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len());
    let yf = std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len());
    f(a, xf, yf)
}

/// Dispatch helper for f32-sized T.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn with_f32<T, R>(
    alpha: T,
    x: &[T],
    y: &mut [T],
    f: unsafe fn(f32, &[f32], &mut [f32]) -> R,
) -> R {
    let a = *(&alpha as *const T as *const f32);
    let xf = std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len());
    let yf = std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, y.len());
    f(a, xf, yf)
}

// ─── AXPY ────────────────────────────────────────────────────────────────────

/// SIMD-accelerated `y ← α·x + y` for dense vectors.
///
/// Falls back to scalar implementation if not on x86_64 or SIMD not available.
#[inline]
pub fn simd_axpy<T: Scalar>(alpha: T, x: &[T], y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len(), "axpy: length mismatch");

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        if std::mem::size_of::<T>() == 8 {
            unsafe { with_f64(alpha, x, y, |a, xf, yf| x86_64::avx2_axpy_f64(a, xf, yf)); }
            return;
        } else if std::mem::size_of::<T>() == 4 {
            unsafe { with_f32(alpha, x, y, |a, xf, yf| x86_64::avx2_axpy_f32(a, xf, yf)); }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::mem::size_of::<T>() == 8 {
            let a  = unsafe { *(&alpha as *const T as *const f64) };
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
            let yf = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
            unsafe { aarch64::neon_axpy_f64(a, xf, yf); }
            return;
        } else if std::mem::size_of::<T>() == 4 {
            let a  = unsafe { *(&alpha as *const T as *const f32) };
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
            let yf = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, y.len()) };
            unsafe { aarch64::neon_axpy_f32(a, xf, yf); }
            return;
        }
    }

    scalar_axpy(alpha, x, y);
}

// ─── AXPBY ───────────────────────────────────────────────────────────────────

/// SIMD-accelerated `y ← α·x + β·y` for dense vectors.
#[inline]
pub fn simd_axpby<T: Scalar>(alpha: T, x: &[T], beta: T, y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len(), "axpby: length mismatch");

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        if std::mem::size_of::<T>() == 8 {
            unsafe {
                let a = *(&alpha as *const T as *const f64);
                let b = *(&beta  as *const T as *const f64);
                let xf = std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len());
                let yf = std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len());
                x86_64::avx2_axpby_f64(a, xf, b, yf);
            }
            return;
        } else if std::mem::size_of::<T>() == 4 {
            unsafe {
                let a = *(&alpha as *const T as *const f32);
                let b = *(&beta  as *const T as *const f32);
                let xf = std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len());
                let yf = std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, y.len());
                x86_64::avx2_axpby_f32(a, xf, b, yf);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::mem::size_of::<T>() == 8 {
            let a  = unsafe { *(&alpha as *const T as *const f64) };
            let b  = unsafe { *(&beta  as *const T as *const f64) };
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
            let yf = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
            unsafe { aarch64::neon_axpby_f64(a, xf, b, yf); }
            return;
        } else if std::mem::size_of::<T>() == 4 {
            let a  = unsafe { *(&alpha as *const T as *const f32) };
            let b  = unsafe { *(&beta  as *const T as *const f32) };
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
            let yf = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, y.len()) };
            unsafe { aarch64::neon_axpby_f32(a, xf, b, yf); }
            return;
        }
    }

    scalar_axpby(alpha, x, beta, y);
}

// ─── DOT ─────────────────────────────────────────────────────────────────────

/// SIMD-accelerated dot product `s = xᵀy` for dense vectors.
///
/// For real scalars only (f32/f64). Provides 2-4x speedup vs scalar loop.
#[inline]
pub fn simd_dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    debug_assert_eq!(x.len(), y.len(), "dot: length mismatch");

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        if std::mem::size_of::<T>() == 8 {
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
            let yf = unsafe { std::slice::from_raw_parts(y.as_ptr() as *const f64, y.len()) };
            let r = x86_64::avx2_dot_f64(xf, yf);
            return unsafe { *(&r as *const f64 as *const T) };
        } else if std::mem::size_of::<T>() == 4 {
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
            let yf = unsafe { std::slice::from_raw_parts(y.as_ptr() as *const f32, y.len()) };
            let r = x86_64::avx2_dot_f32(xf, yf);
            return unsafe { *(&r as *const f32 as *const T) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::mem::size_of::<T>() == 8 {
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
            let yf = unsafe { std::slice::from_raw_parts(y.as_ptr() as *const f64, y.len()) };
            let r = unsafe { aarch64::neon_dot_f64(xf, yf) };
            return unsafe { *(&r as *const f64 as *const T) };
        } else if std::mem::size_of::<T>() == 4 {
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
            let yf = unsafe { std::slice::from_raw_parts(y.as_ptr() as *const f32, y.len()) };
            let r = unsafe { aarch64::neon_dot_f32(xf, yf) };
            return unsafe { *(&r as *const f32 as *const T) };
        }
    }

    scalar_dot(x, y)
}

// ─── NORM2 ───────────────────────────────────────────────────────────────────

/// SIMD-accelerated Euclidean norm `‖x‖₂` for dense vectors.
#[inline]
pub fn simd_norm2<T: Scalar>(x: &[T]) -> T {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        if std::mem::size_of::<T>() == 8 {
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
            let r = x86_64::avx2_norm2_f64(xf);
            return unsafe { *(&r as *const f64 as *const T) };
        } else if std::mem::size_of::<T>() == 4 {
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
            let r = x86_64::avx2_norm2_f32(xf);
            return unsafe { *(&r as *const f32 as *const T) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Use NEON only for longer vectors to avoid floating-point ordering
        // differences vs scalar that can cause numerical instability in
        // sensitive eigensolvers (e.g. LOBPCG for tiny systems).
        if std::mem::size_of::<T>() == 8 && x.len() >= 8 {
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
            let r = unsafe { aarch64::neon_norm2_f64(xf) };
            return unsafe { *(&r as *const f64 as *const T) };
        } else if std::mem::size_of::<T>() == 4 && x.len() >= 8 {
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
            let r = unsafe { aarch64::neon_norm2_f32(xf) };
            return unsafe { *(&r as *const f32 as *const T) };
        }
    }

    scalar_norm2(x)
}

// ─── SUB ─────────────────────────────────────────────────────────────────────

/// SIMD-accelerated element-wise subtraction `out[i] = a[i] − b[i]`.
///
/// Single-pass operation; more efficient than copy + axpy(-1, …).
#[inline]
pub fn simd_sub<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
    debug_assert_eq!(a.len(), b.len(), "sub: length mismatch a/b");
    debug_assert_eq!(a.len(), out.len(), "sub: length mismatch a/out");

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        if std::mem::size_of::<T>() == 8 {
            let af  = unsafe { std::slice::from_raw_parts(a.as_ptr()   as *const f64, a.len()) };
            let bf  = unsafe { std::slice::from_raw_parts(b.as_ptr()   as *const f64, b.len()) };
            let of_ = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f64, out.len()) };
            x86_64::avx2_sub_f64(af, bf, of_);
            return;
        } else if std::mem::size_of::<T>() == 4 {
            let af  = unsafe { std::slice::from_raw_parts(a.as_ptr()   as *const f32, a.len()) };
            let bf  = unsafe { std::slice::from_raw_parts(b.as_ptr()   as *const f32, b.len()) };
            let of_ = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, out.len()) };
            x86_64::avx2_sub_f32(af, bf, of_);
            return;
        }
    }

    scalar_sub(a, b, out);
}

// aarch64 sub not needed — scalar is fine; only add if profiling shows it's hot

// ─── SCALE ───────────────────────────────────────────────────────────────────

/// SIMD-accelerated in-place scale `y[i] *= alpha`.
#[inline]
pub fn simd_scale<T: Scalar>(alpha: T, y: &mut [T]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        if std::mem::size_of::<T>() == 8 {
            let a = unsafe { *(&alpha as *const T as *const f64) };
            let yf = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
            x86_64::avx2_scale_f64(a, yf);
            return;
        } else if std::mem::size_of::<T>() == 4 {
            let a = unsafe { *(&alpha as *const T as *const f32) };
            let yf = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, y.len()) };
            x86_64::avx2_scale_f32(a, yf);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::mem::size_of::<T>() == 8 {
            let a  = unsafe { *(&alpha as *const T as *const f64) };
            let yf = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
            unsafe { aarch64::neon_scale_f64(a, yf); }
            return;
        } else if std::mem::size_of::<T>() == 4 {
            let a  = unsafe { *(&alpha as *const T as *const f32) };
            let yf = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, y.len()) };
            unsafe { aarch64::neon_scale_f32(a, yf); }
            return;
        }
    }
    for v in y.iter_mut() { *v *= alpha; }
}

// ─── GEMV ────────────────────────────────────────────────────────────────────

/// SIMD-accelerated dense GEMV: `y[i] += alpha * (A_row_i · x)`.
///
/// `a` is a row-major `nrows × ncols` matrix stored as a flat slice.
/// `x` has length `ncols`, `y` has length `nrows`.
#[inline]
pub fn simd_gemv<T: Scalar>(alpha: T, a: &[T], nrows: usize, ncols: usize, x: &[T], y: &mut [T]) {
    debug_assert_eq!(a.len(), nrows * ncols, "simd_gemv: a length mismatch");
    debug_assert_eq!(x.len(), ncols, "simd_gemv: x length mismatch");
    debug_assert_eq!(y.len(), nrows, "simd_gemv: y length mismatch");

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        if std::mem::size_of::<T>() == 8 {
            let a_f = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
            let x_f = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
            let y_f = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
            let al  = unsafe { *(&alpha as *const T as *const f64) };
            x86_64::avx2_gemv_f64(al, a_f, nrows, ncols, x_f, y_f);
            return;
        } else if std::mem::size_of::<T>() == 4 {
            let a_f = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
            let x_f = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
            let y_f = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, y.len()) };
            let al  = unsafe { *(&alpha as *const T as *const f32) };
            x86_64::avx2_gemv_f32(al, a_f, nrows, ncols, x_f, y_f);
            return;
        }
    }
    scalar_gemv(alpha, a, nrows, ncols, x, y);
}

/// SIMD-accelerated dense GEMV-transpose: `y[j] += alpha * (A_col_j · x)`.
///
/// Equivalent to `y += alpha * Aᵀ x` where `A` is `nrows × ncols`.
/// `x` has length `nrows`, `y` has length `ncols`.
///
/// Implemented as `nrows` AXPY calls, each of which is SIMD-accelerated.
#[inline]
pub fn simd_gemv_t<T: Scalar>(alpha: T, a: &[T], nrows: usize, ncols: usize, x: &[T], y: &mut [T]) {
    debug_assert_eq!(a.len(), nrows * ncols, "simd_gemv_t: a length mismatch");
    debug_assert_eq!(x.len(), nrows, "simd_gemv_t: x length mismatch");
    debug_assert_eq!(y.len(), ncols, "simd_gemv_t: y length mismatch");
    for i in 0..nrows {
        let row = &a[i * ncols .. (i + 1) * ncols];
        simd_axpy(alpha * x[i], row, y);
    }
}

// ─── HADAMARD ────────────────────────────────────────────────────────────────

/// SIMD-accelerated element-wise product `out[i] = a[i] * b[i]`.
#[inline]
pub fn simd_hadamard<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
    debug_assert_eq!(a.len(), b.len(), "hadamard: length mismatch a/b");
    debug_assert_eq!(a.len(), out.len(), "hadamard: length mismatch a/out");

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        if std::mem::size_of::<T>() == 8 {
            let af  = unsafe { std::slice::from_raw_parts(a.as_ptr()   as *const f64, a.len()) };
            let bf  = unsafe { std::slice::from_raw_parts(b.as_ptr()   as *const f64, b.len()) };
            let of_ = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f64, out.len()) };
            x86_64::avx2_hadamard_f64(af, bf, of_);
            return;
        } else if std::mem::size_of::<T>() == 4 {
            let af  = unsafe { std::slice::from_raw_parts(a.as_ptr()   as *const f32, a.len()) };
            let bf  = unsafe { std::slice::from_raw_parts(b.as_ptr()   as *const f32, b.len()) };
            let of_ = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, out.len()) };
            x86_64::avx2_hadamard_f32(af, bf, of_);
            return;
        }
    }
    for ((oi, &ai), &bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) { *oi = ai * bi; }
}

// ─── Scalar fallbacks ────────────────────────────────────────────────────────

#[inline]
fn scalar_axpy<T: Scalar>(alpha: T, x: &[T], y: &mut [T]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

#[inline]
fn scalar_axpby<T: Scalar>(alpha: T, x: &[T], beta: T, y: &mut [T]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = alpha * xi + beta * *yi;
    }
}

#[inline]
fn scalar_dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    x.iter().zip(y.iter()).fold(T::zero(), |acc, (&a, &b)| acc + a * b)
}

#[inline]
fn scalar_norm2<T: Scalar>(x: &[T]) -> T {
    let ss = x.iter().fold(T::zero(), |acc, &v| acc + v * v);
    ss.sqrt()
}

#[inline]
fn scalar_sub<T: Scalar>(a: &[T], b: &[T], out: &mut [T]) {
    for ((oi, &ai), &bi) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *oi = ai - bi;
    }
}

#[inline]
fn scalar_gemv<T: Scalar>(alpha: T, a: &[T], nrows: usize, ncols: usize, x: &[T], y: &mut [T]) {
    for i in 0..nrows {
        let mut s = T::zero();
        for j in 0..ncols { s += a[i * ncols + j] * x[j]; }
        y[i] += alpha * s;
    }
}

// ─── x86_64 AVX2 implementations ─────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn hsum_pd(v: __m256d) -> f64 {
        let v = _mm256_hadd_pd(v, v);
        let upper = _mm256_extractf128_pd(v, 1);
        let lower = _mm256_castpd256_pd128(v);
        _mm_cvtsd_f64(_mm_add_pd(lower, upper))
    }

    #[inline]
    unsafe fn hsum_ps(v: __m256) -> f32 {
        let v = _mm256_hadd_ps(v, v);
        let v = _mm256_hadd_ps(v, v);
        let upper = _mm256_extractf128_ps(v, 1);
        let lower = _mm256_castps256_ps128(v);
        _mm_cvtss_f32(_mm_add_ps(lower, upper))
    }

    /// y ← α·x + y  (f64, 4-lane AVX2)
    pub fn avx2_axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
        let alpha_vec = unsafe { _mm256_set1_pd(alpha) };
        let len = x.len();
        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                let xv = _mm256_loadu_pd(x.as_ptr().add(i));
                let yv = _mm256_loadu_pd(y.as_ptr().add(i));
                let r  = _mm256_add_pd(_mm256_mul_pd(alpha_vec, xv), yv);
                _mm256_storeu_pd(y.as_mut_ptr().add(i), r);
            }
            i += 4;
        }
        while i < len { y[i] += alpha * x[i]; i += 1; }
    }

    /// y ← α·x + y  (f32, 8-lane AVX2)
    pub fn avx2_axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
        let alpha_vec = unsafe { _mm256_set1_ps(alpha) };
        let len = x.len();
        let mut i = 0;
        while i + 8 <= len {
            unsafe {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i));
                let yv = _mm256_loadu_ps(y.as_ptr().add(i));
                let r  = _mm256_add_ps(_mm256_mul_ps(alpha_vec, xv), yv);
                _mm256_storeu_ps(y.as_mut_ptr().add(i), r);
            }
            i += 8;
        }
        while i < len { y[i] += alpha * x[i]; i += 1; }
    }

    /// y ← α·x + β·y  (f64, 4-lane AVX2)
    pub fn avx2_axpby_f64(alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
        let (av, bv) = unsafe { (_mm256_set1_pd(alpha), _mm256_set1_pd(beta)) };
        let len = x.len();
        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                let xv = _mm256_loadu_pd(x.as_ptr().add(i));
                let yv = _mm256_loadu_pd(y.as_ptr().add(i));
                let r  = _mm256_add_pd(_mm256_mul_pd(av, xv), _mm256_mul_pd(bv, yv));
                _mm256_storeu_pd(y.as_mut_ptr().add(i), r);
            }
            i += 4;
        }
        while i < len { y[i] = alpha * x[i] + beta * y[i]; i += 1; }
    }

    /// y ← α·x + β·y  (f32, 8-lane AVX2)
    pub fn avx2_axpby_f32(alpha: f32, x: &[f32], beta: f32, y: &mut [f32]) {
        let (av, bv) = unsafe { (_mm256_set1_ps(alpha), _mm256_set1_ps(beta)) };
        let len = x.len();
        let mut i = 0;
        while i + 8 <= len {
            unsafe {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i));
                let yv = _mm256_loadu_ps(y.as_ptr().add(i));
                let r  = _mm256_add_ps(_mm256_mul_ps(av, xv), _mm256_mul_ps(bv, yv));
                _mm256_storeu_ps(y.as_mut_ptr().add(i), r);
            }
            i += 8;
        }
        while i < len { y[i] = alpha * x[i] + beta * y[i]; i += 1; }
    }

    /// s = xᵀy  (f64, 4-lane AVX2)
    pub fn avx2_dot_f64(x: &[f64], y: &[f64]) -> f64 {
        let mut acc = unsafe { _mm256_setzero_pd() };
        let len = x.len();
        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                let xv = _mm256_loadu_pd(x.as_ptr().add(i));
                let yv = _mm256_loadu_pd(y.as_ptr().add(i));
                acc = _mm256_add_pd(acc, _mm256_mul_pd(xv, yv));
            }
            i += 4;
        }
        let mut s = unsafe { hsum_pd(acc) };
        while i < len { s += x[i] * y[i]; i += 1; }
        s
    }

    /// s = xᵀy  (f32, 8-lane AVX2)
    pub fn avx2_dot_f32(x: &[f32], y: &[f32]) -> f32 {
        let mut acc = unsafe { _mm256_setzero_ps() };
        let len = x.len();
        let mut i = 0;
        while i + 8 <= len {
            unsafe {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i));
                let yv = _mm256_loadu_ps(y.as_ptr().add(i));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(xv, yv));
            }
            i += 8;
        }
        let mut s = unsafe { hsum_ps(acc) };
        while i < len { s += x[i] * y[i]; i += 1; }
        s
    }

    /// ‖x‖₂  (f64, 4-lane AVX2)
    pub fn avx2_norm2_f64(x: &[f64]) -> f64 {
        let mut acc = unsafe { _mm256_setzero_pd() };
        let len = x.len();
        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                let xv = _mm256_loadu_pd(x.as_ptr().add(i));
                acc = _mm256_add_pd(acc, _mm256_mul_pd(xv, xv));
            }
            i += 4;
        }
        let mut ss = unsafe { hsum_pd(acc) };
        while i < len { ss += x[i] * x[i]; i += 1; }
        ss.sqrt()
    }

    /// ‖x‖₂  (f32, 8-lane AVX2)
    pub fn avx2_norm2_f32(x: &[f32]) -> f32 {
        let mut acc = unsafe { _mm256_setzero_ps() };
        let len = x.len();
        let mut i = 0;
        while i + 8 <= len {
            unsafe {
                let xv = _mm256_loadu_ps(x.as_ptr().add(i));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(xv, xv));
            }
            i += 8;
        }
        let mut ss = unsafe { hsum_ps(acc) };
        while i < len { ss += x[i] * x[i]; i += 1; }
        ss.sqrt()
    }

    /// out[i] = a[i] − b[i]  (f64, 4-lane AVX2)
    pub fn avx2_sub_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                let av = _mm256_loadu_pd(a.as_ptr().add(i));
                let bv = _mm256_loadu_pd(b.as_ptr().add(i));
                _mm256_storeu_pd(out.as_mut_ptr().add(i), _mm256_sub_pd(av, bv));
            }
            i += 4;
        }
        while i < len { out[i] = a[i] - b[i]; i += 1; }
    }

    /// out[i] = a[i] − b[i]  (f32, 8-lane AVX2)
    pub fn avx2_sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + 8 <= len {
            unsafe {
                let av = _mm256_loadu_ps(a.as_ptr().add(i));
                let bv = _mm256_loadu_ps(b.as_ptr().add(i));
                _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_sub_ps(av, bv));
            }
            i += 8;
        }
        while i < len { out[i] = a[i] - b[i]; i += 1; }
    }

    /// y[i] *= alpha  (f64, 4-lane AVX2)
    pub fn avx2_scale_f64(alpha: f64, y: &mut [f64]) {
        let av = unsafe { _mm256_set1_pd(alpha) };
        let len = y.len();
        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                let yv = _mm256_loadu_pd(y.as_ptr().add(i));
                _mm256_storeu_pd(y.as_mut_ptr().add(i), _mm256_mul_pd(av, yv));
            }
            i += 4;
        }
        while i < len { y[i] *= alpha; i += 1; }
    }

    /// y[i] *= alpha  (f32, 8-lane AVX2)
    pub fn avx2_scale_f32(alpha: f32, y: &mut [f32]) {
        let av = unsafe { _mm256_set1_ps(alpha) };
        let len = y.len();
        let mut i = 0;
        while i + 8 <= len {
            unsafe {
                let yv = _mm256_loadu_ps(y.as_ptr().add(i));
                _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_mul_ps(av, yv));
            }
            i += 8;
        }
        while i < len { y[i] *= alpha; i += 1; }
    }

    /// y[i] += alpha * dot(A_row_i, x)  (f64, row-major nrows×ncols, 4-lane AVX2)
    pub fn avx2_gemv_f64(alpha: f64, a: &[f64], nrows: usize, ncols: usize, x: &[f64], y: &mut [f64]) {
        for i in 0..nrows {
            let row = &a[i * ncols .. (i + 1) * ncols];
            let mut acc = unsafe { _mm256_setzero_pd() };
            let mut j = 0;
            while j + 4 <= ncols {
                unsafe {
                    let av = _mm256_loadu_pd(row.as_ptr().add(j));
                    let xv = _mm256_loadu_pd(x.as_ptr().add(j));
                    acc = _mm256_add_pd(acc, _mm256_mul_pd(av, xv));
                }
                j += 4;
            }
            let mut s = unsafe { hsum_pd(acc) };
            while j < ncols { s += row[j] * x[j]; j += 1; }
            y[i] += alpha * s;
        }
    }

    /// y[i] += alpha * dot(A_row_i, x)  (f32, row-major nrows×ncols, 8-lane AVX2)
    pub fn avx2_gemv_f32(alpha: f32, a: &[f32], nrows: usize, ncols: usize, x: &[f32], y: &mut [f32]) {
        for i in 0..nrows {
            let row = &a[i * ncols .. (i + 1) * ncols];
            let mut acc = unsafe { _mm256_setzero_ps() };
            let mut j = 0;
            while j + 8 <= ncols {
                unsafe {
                    let av = _mm256_loadu_ps(row.as_ptr().add(j));
                    let xv = _mm256_loadu_ps(x.as_ptr().add(j));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(av, xv));
                }
                j += 8;
            }
            let mut s = unsafe { hsum_ps(acc) };
            while j < ncols { s += row[j] * x[j]; j += 1; }
            y[i] += alpha * s;
        }
    }

    /// out[i] = a[i] * b[i]  (f64, 4-lane AVX2)
    pub fn avx2_hadamard_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                let av = _mm256_loadu_pd(a.as_ptr().add(i));
                let bv = _mm256_loadu_pd(b.as_ptr().add(i));
                _mm256_storeu_pd(out.as_mut_ptr().add(i), _mm256_mul_pd(av, bv));
            }
            i += 4;
        }
        while i < len { out[i] = a[i] * b[i]; i += 1; }
    }

    /// out[i] = a[i] * b[i]  (f32, 8-lane AVX2)
    pub fn avx2_hadamard_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + 8 <= len {
            unsafe {
                let av = _mm256_loadu_ps(a.as_ptr().add(i));
                let bv = _mm256_loadu_ps(b.as_ptr().add(i));
                _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(av, bv));
            }
            i += 8;
        }
        while i < len { out[i] = a[i] * b[i]; i += 1; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_axpy() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![1.0_f64, 1.0, 1.0, 1.0, 1.0];
        simd_axpy(2.0, &x, &mut y);
        let expected = [3.0, 5.0, 7.0, 9.0, 11.0];
        for (got, exp) in y.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_simd_axpby() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![2.0_f64, 2.0, 2.0, 2.0, 2.0];
        simd_axpby(2.0, &x, 0.5, &mut y);
        // y = 2*x + 0.5*y → [3, 5, 7, 9, 11]
        let expected = [3.0, 5.0, 7.0, 9.0, 11.0];
        for (got, exp) in y.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_simd_dot() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0_f64, 1.0, 1.0, 1.0, 1.0];
        let s = simd_dot(&x, &y);
        assert!((s - 15.0).abs() < 1e-12, "got {s}");

        // Dot with itself
        let s2 = simd_dot(&x, &x);
        assert!((s2 - 55.0).abs() < 1e-12, "got {s2}");
    }

    #[test]
    fn test_simd_sub() {
        let a = vec![3.0_f64, 5.0, 7.0, 9.0, 11.0];
        let b = vec![1.0_f64, 2.0, 3.0, 4.0,  5.0];
        let mut out = vec![0.0_f64; 5];
        simd_sub(&a, &b, &mut out);
        let expected = [2.0, 3.0, 4.0, 5.0, 6.0];
        for (got, exp) in out.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_simd_norm2() {
        let x = vec![3.0_f64, 4.0];
        let n = simd_norm2(&x);
        assert!((n - 5.0).abs() < 1e-12, "got {n}");

        let x9 = vec![1.0_f64; 9];
        let n9 = simd_norm2(&x9);
        assert!((n9 - 3.0).abs() < 1e-12, "got {n9}");
    }

    #[test]
    fn test_simd_lobpcg_sequence_n4() {
        // Reproduce the sequence that LOBPCG does for a 4x4 diagonal matrix
        let a_diag = [0.3_f64, 1.2, 2.5, 4.0];
        // Deterministic "random" init (matches fill_random seed 42 + 0 * deadbeef = 42+1=43)
        let mut x = vec![0.1_f64, -0.3, 0.7, -0.5];  // arbitrary non-zero
        let nrm = simd_norm2(&x);
        assert!(nrm > 1e-10, "norm is zero?! {nrm}");
        simd_scale(1.0 / nrm, &mut x);
        let nrm2 = simd_norm2(&x);
        assert!((nrm2 - 1.0).abs() < 1e-12, "after normalize: nrm={nrm2}");
        // Ax
        let ax: Vec<f64> = x.iter().zip(a_diag.iter()).map(|(xi, ai)| ai * xi).collect();
        let lambda = simd_dot(&x, &ax);
        assert!(lambda > 0.0, "lambda should be positive, got {lambda}");
        assert!(lambda >= 0.29 && lambda <= 4.01, "lambda should be in [0.3, 4.0], got {lambda}");
    }

    #[test]
    fn test_simd_scale() {
        let mut y = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        simd_scale(2.0, &mut y);
        let expected = [2.0_f64, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        for (g, e) in y.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-12, "got {g}, expected {e}");
        }
    }

    #[test]
    fn test_simd_gemv() {
        // 3×3 identity * [1,2,3] = [1,2,3]
        let a = vec![
            1.0_f64, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let x = vec![1.0_f64, 2.0, 3.0];
        let mut y = vec![0.0_f64; 3];
        simd_gemv(1.0, &a, 3, 3, &x, &mut y);
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 2.0).abs() < 1e-12);
        assert!((y[2] - 3.0).abs() < 1e-12);

        // 2×3 matrix
        let a2 = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x2 = vec![1.0_f64, 1.0, 1.0];
        let mut y2 = vec![0.0_f64; 2];
        simd_gemv(1.0, &a2, 2, 3, &x2, &mut y2);
        assert!((y2[0] - 6.0).abs() < 1e-12);
        assert!((y2[1] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_simd_gemv_t() {
        // 3×2 matrix A, compute A^T * x where x: length 3, y: length 2
        // A = [[1,2],[3,4],[5,6]], A^T = [[1,3,5],[2,4,6]]
        // x = [1,1,1] → A^T*x = [9, 12]
        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0_f64, 1.0, 1.0];
        let mut y = vec![0.0_f64; 2];
        simd_gemv_t(1.0, &a, 3, 2, &x, &mut y);
        assert!((y[0] - 9.0).abs() < 1e-12, "y[0]={}", y[0]);
        assert!((y[1] - 12.0).abs() < 1e-12, "y[1]={}", y[1]);
    }

    #[test]
    fn test_simd_hadamard() {
        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0_f64, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0_f64; 5];
        simd_hadamard(&a, &b, &mut out);
        let expected = [2.0_f64, 6.0, 12.0, 20.0, 30.0];
        for (g, e) in out.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-12, "got {g}, expected {e}");
        }
    }
}
