//! x86_64-specific SIMD implementations using AVX2 and scalar fallback.
//!
//! AVX2 provides 256-bit vectors (4 × f64 or 8 × f32 lanes).
//! This is widely available on modern x86_64 CPUs (post-2013).

use crate::core::scalar::Scalar;

/// Efficient horizontal sum for 256-bit f64 vector (4 lanes).
/// 
/// Uses SIMD shuffle and addition instead of store-to-memory approach.
/// Reduces memory traffic and instruction latency.
#[inline]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum_f64(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    
    // v = [a, b, c, d]
    // After hadd_pd: [a+b, c+d, a+b, c+d]
    let v = _mm256_hadd_pd(v, v);
    
    // Extract lower 128-bit and add to upper 128-bit
    let upper = _mm256_extractf128_pd(v, 1);
    let lower = _mm256_castpd256_pd128(v);
    let sum_vec = _mm_add_pd(lower, upper);
    
    // Extract scalar from the 128-bit result
    // The result is [sum, ?, ?, ?] where sum = a+b+c+d
    _mm_cvtsd_f64(sum_vec)
}

/// Efficient horizontal sum for 256-bit f32 vector (8 lanes).
///
/// Uses SIMD shuffle and addition instead of store-to-memory approach.
#[inline]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum_f32(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    
    // v = [a, b, c, d, e, f, g, h]
    // After hadd_ps: [a+b, c+d, e+f, g+h, a+b, c+d, e+f, g+h]
    let v = _mm256_hadd_ps(v, v);
    
    // After second hadd_ps: [a+b+c+d, e+f+g+h, ...]
    let v = _mm256_hadd_ps(v, v);
    
    // Extract lower 128-bit and add to upper 128-bit
    let upper = _mm256_extractf128_ps(v, 1);
    let lower = _mm256_castps256_ps128(v);
    let sum_vec = _mm_add_ps(lower, upper);
    
    // Extract scalar
    _mm_cvtss_f32(sum_vec)
}

/// AVX2-accelerated sparse row dot product (dispatch based on type).
pub unsafe fn avx2_row_dot<T: Scalar>(
    col_idx: &[usize],
    values: &[T],
    x: &[T],
    start: usize,
    end: usize,
) -> T {
    // For now, we dispatch based on size hints
    // Production code would use type specialization or trait dispatch
    if end - start < 4 {
        // Too small for SIMD, use scalar path
        return super::scalar_row_dot(col_idx, values, x, start, end);
    }

    // The actual SIMD implementation is type-specific and would be called
    // via specialization in production. For this sketch, we use scalar.
    super::scalar_row_dot(col_idx, values, x, start, end)
}

/// AVX2 specialization for f64 (4 lanes of 64-bit floats).
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn avx2_row_dot_f64(
    col_idx: &[usize],
    values: &[f64],
    x: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    use std::arch::x86_64::*;

    let len = end - start;
    if len == 0 {
        return 0.0;
    }

    // Initialize accumulator to zero (4 lanes of 0.0)
    let mut sum_vec = _mm256_setzero_pd();

    // Process 4 elements at a time
    let mut i = start;
    while i + 3 < end {
        // Load 4 column indices and gather x values
        let c0 = *col_idx.get_unchecked(i);
        let c1 = *col_idx.get_unchecked(i + 1);
        let c2 = *col_idx.get_unchecked(i + 2);
        let c3 = *col_idx.get_unchecked(i + 3);

        let x0 = *x.get_unchecked(c0);
        let x1 = *x.get_unchecked(c1);
        let x2 = *x.get_unchecked(c2);
        let x3 = *x.get_unchecked(c3);

        let vals = _mm256_loadu_pd(values.as_ptr().add(i) as *const f64);
        let xs = _mm256_setr_pd(x0, x1, x2, x3);

        let prod = _mm256_mul_pd(vals, xs);
        sum_vec = _mm256_add_pd(sum_vec, prod);

        i += 4;
    }

    // Horizontal sum of the 4 lanes using efficient SIMD operations
    let mut sum = super::scalar_row_dot(col_idx, values, x, i, end);
    sum += hsum_f64(sum_vec);

    sum
}

/// AVX2 specialization for f32 (8 lanes of 32-bit floats).
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn avx2_row_dot_f32(
    col_idx: &[usize],
    values: &[f32],
    x: &[f32],
    start: usize,
    end: usize,
) -> f32 {
    use std::arch::x86_64::*;

    let len = end - start;
    if len == 0 {
        return 0.0;
    }

    // Initialize accumulator to zero (8 lanes of 0.0f)
    let mut sum_vec = _mm256_setzero_ps();

    // Process 8 elements at a time
    let mut i = start;
    while i + 7 < end {
        let c0 = *col_idx.get_unchecked(i);
        let c1 = *col_idx.get_unchecked(i + 1);
        let c2 = *col_idx.get_unchecked(i + 2);
        let c3 = *col_idx.get_unchecked(i + 3);
        let c4 = *col_idx.get_unchecked(i + 4);
        let c5 = *col_idx.get_unchecked(i + 5);
        let c6 = *col_idx.get_unchecked(i + 6);
        let c7 = *col_idx.get_unchecked(i + 7);

        let x0 = *x.get_unchecked(c0);
        let x1 = *x.get_unchecked(c1);
        let x2 = *x.get_unchecked(c2);
        let x3 = *x.get_unchecked(c3);
        let x4 = *x.get_unchecked(c4);
        let x5 = *x.get_unchecked(c5);
        let x6 = *x.get_unchecked(c6);
        let x7 = *x.get_unchecked(c7);

        let vals = _mm256_loadu_ps(values.as_ptr().add(i) as *const f32);
        let xs = _mm256_setr_ps(x0, x1, x2, x3, x4, x5, x6, x7);

        let prod = _mm256_mul_ps(vals, xs);
        sum_vec = _mm256_add_ps(sum_vec, prod);

        i += 8;
    }

    // Horizontal sum of the 8 lanes using efficient SIMD operations
    let mut sum = super::scalar_row_dot(col_idx, values, x, i, end);
    sum += hsum_f32(sum_vec);

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_row_dot_f64() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available, skipping test");
            return;
        }

        let col_idx = vec![0, 1, 2, 3];
        let values = vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64];
        let x = vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64];

        let result = unsafe { avx2_row_dot_f64(&col_idx, &values, &x, 0, 4) };
        let expected = 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0; // 30.0
        assert!((result - expected).abs() < 1e-10, "expected {}, got {}", expected, result);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_row_dot_f32() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available, skipping test");
            return;
        }

        let col_idx = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let values: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let x: Vec<f32> = (1..=8).map(|i| i as f32).collect();

        let result = unsafe { avx2_row_dot_f32(&col_idx, &values, &x, 0, 8) };
        let expected: f32 = (1..=8).map(|i| (i as f32).powi(2)).sum();
        assert!((result - expected).abs() < 1e-5, "expected {}, got {}", expected, result);
    }
}
