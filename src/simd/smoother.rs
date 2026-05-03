//! SIMD-accelerated smoother implementations for AMG.
//!
//! Currently provides SIMD-optimized Jacobi smoother.
//! Other smoothers (Gauss-Seidel, Chebyshev) follow.

use crate::core::scalar::Scalar;
use crate::core::vector::{DenseVec, Vector};
use crate::sparse::CsrMatrix;

/// SIMD-accelerated Jacobi smoother: `x ← x + ω·D⁻¹·(b - A·x)`.
///
/// Jacobi smoothing is the simplest smoother used in AMG:
/// - Extract diagonal D of A
/// - Apply: `x_{k+1} = x_k + ω·D⁻¹·(b - A·x_k)`
///
/// SIMD acceleration targets the scaling by D⁻¹ and vector operations.
pub fn jacobi_smooth_simd<T: Scalar>(
    a: &CsrMatrix<T>,
    x: &mut DenseVec<T>,
    b: &DenseVec<T>,
    omega: T,
    iterations: usize,
) {
    let n = x.len();
    debug_assert_eq!(a.nrows(), n);
    debug_assert_eq!(a.ncols(), n);
    debug_assert_eq!(b.len(), n);

    // Extract diagonal (1x cost)
    let diag = a.diag();
    
    // Workspace: r = b - A*x
    let mut r = b.clone();
    let mut ax = DenseVec::zeros(n);

    for _ in 0..iterations {
        // r = b - A*x
        a.spmv(x.as_slice(), &mut ax.as_mut_slice());
        for i in 0..n {
            r.as_mut_slice()[i] = b.as_slice()[i] - ax.as_slice()[i];
        }

        // x ← x + ω·D⁻¹·r
        // Optimized with SIMD-accelerated scaling
        jacobi_scale_simd(x.as_mut_slice(), &diag, &r.as_slice(), omega);
    }
}

/// SIMD-accelerated scaling: `x ← x + α·D⁻¹·v`.
///
/// This is the critical loop in Jacobi smoothing.
/// SIMD can provide 2-4x speedup for large vectors.
#[inline]
fn jacobi_scale_simd<T: Scalar>(
    x: &mut [T],
    diag: &[T],
    v: &[T],
    alpha: T,
) {
    debug_assert_eq!(x.len(), diag.len());
    debug_assert_eq!(x.len(), v.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Dispatch based on element type size (f64 vs f32)
            if std::mem::size_of::<T>() == 8 {
                // Treat as f64
                let x_f64 = unsafe {
                    std::slice::from_raw_parts_mut(
                        x.as_mut_ptr() as *mut f64,
                        x.len(),
                    )
                };
                let diag_f64 = unsafe {
                    std::slice::from_raw_parts(
                        diag.as_ptr() as *const f64,
                        diag.len(),
                    )
                };
                let v_f64 = unsafe {
                    std::slice::from_raw_parts(
                        v.as_ptr() as *const f64,
                        v.len(),
                    )
                };
                let alpha_f64 = unsafe {
                    *((&alpha as *const T) as *const f64)
                };

                return x86_64::jacobi_scale_simd_f64(x_f64, diag_f64, v_f64, alpha_f64);
            }
        }
    }

    // Scalar fallback
    jacobi_scale_scalar(x, diag, v, alpha);
}

/// Scalar fallback for Jacobi scaling.
#[inline]
fn jacobi_scale_scalar<T: Scalar>(
    x: &mut [T],
    diag: &[T],
    v: &[T],
    alpha: T,
) {
    for i in 0..x.len() {
        if diag[i] != T::zero() {
            x[i] += alpha * v[i] / diag[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    /// SIMD Jacobi scaling for f64: `x ← x + α·D⁻¹·v`.
    pub fn jacobi_scale_simd_f64(
        x: &mut [f64],
        diag: &[f64],
        v: &[f64],
        alpha: f64,
    ) {
        let alpha_vec = _mm256_set1_pd(alpha);
        let mut i = 0;
        let len = x.len();

        // Process 4 elements at a time
        while i + 3 < len {
            unsafe {
                let diag_vec = _mm256_loadu_pd(diag.as_ptr().add(i));
                let v_vec = _mm256_loadu_pd(v.as_ptr().add(i));
                let x_vec = _mm256_loadu_pd(x.as_ptr().add(i));

                // Compute: x + α·v / diag
                // = x + α·v·(1/diag)
                let inv_diag = _mm256_div_pd(_mm256_set1_pd(1.0), diag_vec);
                let scaled = _mm256_mul_pd(alpha_vec, v_vec);
                let term = _mm256_mul_pd(scaled, inv_diag);
                let result = _mm256_add_pd(x_vec, term);

                _mm256_storeu_pd(x.as_mut_ptr().add(i), result);
            }
            i += 4;
        }

        // Scalar fallback for remainder
        for j in i..len {
            if diag[j] != 0.0 {
                x[j] += alpha * v[j] / diag[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CooMatrix;

    #[test]
    fn test_jacobi_smooth_simple() {
        let n = 10;
        let mut coo = CooMatrix::new(n, n);
        
        // Create a simple diagonal dominant matrix
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0 { coo.push(i, i - 1, -0.5); }
            if i < n - 1 { coo.push(i, i + 1, -0.5); }
        }
        
        let a = CsrMatrix::from_coo(&coo);
        let b = DenseVec::from_vec(vec![1.0; n]);
        let mut x = DenseVec::zeros(n);
        
        // One iteration of Jacobi smoothing
        jacobi_smooth_simd(&a, &mut x, &b, 1.0, 1);
        
        // After one iteration, x should be non-zero
        let norm: f64 = x.as_slice().iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        assert!(norm > 0.0, "Jacobi smoothing should change x");
    }
}
