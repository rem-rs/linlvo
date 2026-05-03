//! SIMD-accelerated sparse and dense matrix operations.
//!
//! This module provides high-performance implementations using SIMD instructions
//! (AVX2, SSE4.2, scalar fallback).
//!
//! CPU feature detection is performed at runtime via `std::is_x86_feature_detected!`.
//! For targets without SIMD support, scalar implementations are used.

use crate::core::scalar::Scalar;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

pub mod dense_ops;
pub mod smoother;

pub use dense_ops::{simd_axpy, simd_axpby, simd_dot, simd_norm2, simd_sub,
                    simd_scale, simd_gemv, simd_gemv_t, simd_hadamard};
pub use smoother::{jacobi_smooth_simd, gs_smooth_simd, chebyshev_smooth_simd,
                   estimate_spectral_radius};

/// Perform a dot product of a sparse row with a dense vector using SIMD if available.
///
/// # Arguments
/// - `col_idx`: column indices (CSR format)
/// - `values`: non-zero values (CSR format)
/// - `x`: dense input vector
/// - `start`: start index in values/col_idx arrays
/// - `end`: end index in values/col_idx arrays
///
/// # Safety
/// - All indices in `col_idx[start..end]` must be < `x.len()`
/// - Indices `start` and `end` must be valid for both `values` and `col_idx`
pub unsafe fn simd_row_dot<T: Scalar>(
    col_idx: &[usize],
    values: &[T],
    x: &[T],
    start: usize,
    end: usize,
) -> T {
    #[cfg(target_arch = "x86_64")]
    {
        // Dispatch based on SIMD capabilities
        if is_x86_feature_detected!("avx2") {
            return x86_64::avx2_row_dot(col_idx, values, x, start, end);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on AArch64.
        if std::mem::size_of::<T>() == 8 {
            let vf = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const f64, values.len()) };
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr()      as *const f64, x.len()) };
            let r  = unsafe { aarch64::neon_row_dot_f64(col_idx, vf, xf, start, end) };
            return unsafe { *(&r as *const f64 as *const T) };
        } else if std::mem::size_of::<T>() == 4 {
            let vf = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const f32, values.len()) };
            let xf = unsafe { std::slice::from_raw_parts(x.as_ptr()      as *const f32, x.len()) };
            let r  = unsafe { aarch64::neon_row_dot_f32(col_idx, vf, xf, start, end) };
            return unsafe { *(&r as *const f32 as *const T) };
        }
    }

    // Fallback to scalar implementation
    scalar_row_dot(col_idx, values, x, start, end)
}

/// Scalar fallback implementation for sparse row dot product.
#[inline]
unsafe fn scalar_row_dot<T: Scalar>(
    col_idx: &[usize],
    values: &[T],
    x: &[T],
    start: usize,
    end: usize,
) -> T {
    match end - start {
        0 => T::zero(),
        1 => *values.get_unchecked(start) * *x.get_unchecked(*col_idx.get_unchecked(start)),
        2 => {
            let c0 = *col_idx.get_unchecked(start);
            let c1 = *col_idx.get_unchecked(start + 1);
            *values.get_unchecked(start) * *x.get_unchecked(c0)
                + *values.get_unchecked(start + 1) * *x.get_unchecked(c1)
        }
        3 => {
            let c0 = *col_idx.get_unchecked(start);
            let c1 = *col_idx.get_unchecked(start + 1);
            let c2 = *col_idx.get_unchecked(start + 2);
            *values.get_unchecked(start) * *x.get_unchecked(c0)
                + *values.get_unchecked(start + 1) * *x.get_unchecked(c1)
                + *values.get_unchecked(start + 2) * *x.get_unchecked(c2)
        }
        4 => {
            let c0 = *col_idx.get_unchecked(start);
            let c1 = *col_idx.get_unchecked(start + 1);
            let c2 = *col_idx.get_unchecked(start + 2);
            let c3 = *col_idx.get_unchecked(start + 3);
            *values.get_unchecked(start) * *x.get_unchecked(c0)
                + *values.get_unchecked(start + 1) * *x.get_unchecked(c1)
                + *values.get_unchecked(start + 2) * *x.get_unchecked(c2)
                + *values.get_unchecked(start + 3) * *x.get_unchecked(c3)
        }
        _ => {
            let mut sum = T::zero();
            let mut k = start;
            while k < end {
                sum += *values.get_unchecked(k) * *x.get_unchecked(*col_idx.get_unchecked(k));
                k += 1;
            }
            sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::{CooMatrix, CsrMatrix};

    #[test]
    fn test_simd_row_dot_single() {
        let col_idx = vec![0, 2];
        let values = vec![2.0_f64, 3.0_f64];
        let x = vec![1.0_f64, 0.0_f64, 4.0_f64];
        let result = unsafe { simd_row_dot(&col_idx, &values, &x, 0, 2) };
        assert!((result - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_row_dot_poisson_1d() {
        // 1D Poisson: tridiagonal matrix
        let n = 10;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0 { coo.push(i, i - 1, -1.0); }
            if i < n - 1 { coo.push(i, i + 1, -1.0); }
        }
        let csr = CsrMatrix::from_coo(&coo);
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut y = vec![0.0; n];

        csr.spmv(&x, &mut y);

        // Verify: y[0] = 2*1 - 2 = 0, y[1] = 2*2 - 1 - 3 = 0, etc.
        for i in 0..n {
            let expected = if i == 0 {
                2.0 * x[0] - x[1]
            } else if i == n - 1 {
                2.0 * x[n - 1] - x[n - 2]
            } else {
                2.0 * x[i] - x[i - 1] - x[i + 1]
            };
            assert!((y[i] - expected).abs() < 1e-10, "y[{i}] mismatch");
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
