//! AArch64 NEON SIMD implementations.
//!
//! NEON is always available on AArch64 (including Apple Silicon M1/M2/M3).
//! All functions are annotated with `#[target_feature(enable = "neon")]` so
//! the compiler can emit NEON instructions without a blanket `-C target-feature`
//! flag.  Call-sites gate on `#[cfg(target_arch = "aarch64")]`; no runtime
//! detection is needed because NEON is mandatory on AArch64.
//!
//! Register widths used:
//! - `float64x2_t` — 128-bit, 2 × f64
//! - `float32x4_t` — 128-bit, 4 × f32

#[allow(unused_imports)]
use std::arch::aarch64::*;

// ─── Horizontal sums ─────────────────────────────────────────────────────────

/// Sum all 2 f64 lanes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hsum_f64x2(v: float64x2_t) -> f64 {
    vaddvq_f64(v)
}

/// Sum all 4 f32 lanes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    vaddvq_f32(v)
}

// ─── SpMV row dot ─────────────────────────────────────────────────────────────

/// NEON gather-style row dot for f64: `sum = Σ values[k] * x[col_idx[k]]`.
///
/// Processes 2 non-zeros per iteration (one `float64x2_t`).
/// Remainder handled scalar.
#[target_feature(enable = "neon")]
pub unsafe fn neon_row_dot_f64(
    col_idx: &[usize],
    values:  &[f64],
    x:       &[f64],
    start:   usize,
    end:     usize,
) -> f64 {
    let len = end - start;
    let mut acc = vdupq_n_f64(0.0);
    let mut i = 0;

    while i + 1 < len {
        let k0 = start + i;
        let k1 = k0 + 1;
        // Gather: load two x values addressed by col_idx
        let x0 = *x.get_unchecked(col_idx[k0]);
        let x1 = *x.get_unchecked(col_idx[k1]);
        let xv = vcombine_f64(vcreate_f64(x0.to_bits()), vcreate_f64(x1.to_bits()));
        let vv = vld1q_f64(values.as_ptr().add(k0));
        acc = vfmaq_f64(acc, vv, xv);
        i += 2;
    }

    let mut sum = hsum_f64x2(acc);
    // Scalar tail
    if i < len {
        let k = start + i;
        sum += *values.get_unchecked(k) * *x.get_unchecked(col_idx[k]);
    }
    sum
}

/// NEON gather-style row dot for f32: `sum = Σ values[k] * x[col_idx[k]]`.
///
/// Processes 4 non-zeros per iteration (one `float32x4_t`).
#[target_feature(enable = "neon")]
pub unsafe fn neon_row_dot_f32(
    col_idx: &[usize],
    values:  &[f32],
    x:       &[f32],
    start:   usize,
    end:     usize,
) -> f32 {
    let len = end - start;
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 3 < len {
        let k0 = start + i;
        let x0 = *x.get_unchecked(col_idx[k0]);
        let x1 = *x.get_unchecked(col_idx[k0 + 1]);
        let x2 = *x.get_unchecked(col_idx[k0 + 2]);
        let x3 = *x.get_unchecked(col_idx[k0 + 3]);
        let xv = {
            let lo = vcombine_f32(
                vdup_n_f32(x0), // [x0, x0]
                vdup_n_f32(x1), // [x1, x1]
            );
            // We want [x0, x1, x2, x3]; build via lane insertion.
            let mut tmp = vdupq_n_f32(0.0);
            tmp = vsetq_lane_f32(x0, tmp, 0);
            tmp = vsetq_lane_f32(x1, tmp, 1);
            tmp = vsetq_lane_f32(x2, tmp, 2);
            tmp = vsetq_lane_f32(x3, tmp, 3);
            let _ = lo; // suppress unused
            tmp
        };
        let vv = vld1q_f32(values.as_ptr().add(k0));
        acc = vfmaq_f32(acc, vv, xv);
        i += 4;
    }

    let mut sum = hsum_f32x4(acc);
    while i < len {
        let k = start + i;
        sum += *values.get_unchecked(k) * *x.get_unchecked(col_idx[k]);
        i += 1;
    }
    sum
}

// ─── AXPY  y ← α·x + y ──────────────────────────────────────────────────────

/// NEON AXPY for f64: `y ← α·x + y`.  Processes 2 elements per iteration.
#[target_feature(enable = "neon")]
pub unsafe fn neon_axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let av = vdupq_n_f64(alpha);
    let mut i = 0;
    while i + 1 < n {
        let xv = vld1q_f64(x.as_ptr().add(i));
        let yv = vld1q_f64(y.as_ptr().add(i));
        // Use separate mul+add (not FMA) to match scalar floating-point semantics.
        let rv = vaddq_f64(yv, vmulq_f64(av, xv));
        vst1q_f64(y.as_mut_ptr().add(i), rv);
        i += 2;
    }
    // Scalar tail
    while i < n {
        *y.get_unchecked_mut(i) += alpha * *x.get_unchecked(i);
        i += 1;
    }
}

/// NEON AXPY for f32: `y ← α·x + y`.  Processes 4 elements per iteration.
#[target_feature(enable = "neon")]
pub unsafe fn neon_axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let av = vdupq_n_f32(alpha);
    let mut i = 0;
    while i + 3 < n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let yv = vld1q_f32(y.as_ptr().add(i));
        let rv = vaddq_f32(yv, vmulq_f32(av, xv));
        vst1q_f32(y.as_mut_ptr().add(i), rv);
        i += 4;
    }
    while i < n {
        *y.get_unchecked_mut(i) += alpha * *x.get_unchecked(i);
        i += 1;
    }
}

// ─── AXPBY  y ← α·x + β·y ───────────────────────────────────────────────────

/// NEON AXPBY for f64: `y ← α·x + β·y`.
#[target_feature(enable = "neon")]
pub unsafe fn neon_axpby_f64(alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let av = vdupq_n_f64(alpha);
    let bv = vdupq_n_f64(beta);
    let mut i = 0;
    while i + 1 < n {
        let xv = vld1q_f64(x.as_ptr().add(i));
        let yv = vld1q_f64(y.as_ptr().add(i));
        // y = beta*y + alpha*x  (separate mul+add to match scalar semantics)
        let rv = vaddq_f64(vmulq_f64(bv, yv), vmulq_f64(av, xv));
        vst1q_f64(y.as_mut_ptr().add(i), rv);
        i += 2;
    }
    while i < n {
        *y.get_unchecked_mut(i) = alpha * *x.get_unchecked(i) + beta * *y.get_unchecked(i);
        i += 1;
    }
}

/// NEON AXPBY for f32: `y ← α·x + β·y`.
#[target_feature(enable = "neon")]
pub unsafe fn neon_axpby_f32(alpha: f32, x: &[f32], beta: f32, y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let av = vdupq_n_f32(alpha);
    let bv = vdupq_n_f32(beta);
    let mut i = 0;
    while i + 3 < n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let yv = vld1q_f32(y.as_ptr().add(i));
        let rv = vaddq_f32(vmulq_f32(bv, yv), vmulq_f32(av, xv));
        vst1q_f32(y.as_mut_ptr().add(i), rv);
        i += 4;
    }
    while i < n {
        *y.get_unchecked_mut(i) = alpha * *x.get_unchecked(i) + beta * *y.get_unchecked(i);
        i += 1;
    }
}

// ─── DOT  s = xᵀy ───────────────────────────────────────────────────────────

/// NEON dot product for f64.
#[target_feature(enable = "neon")]
pub unsafe fn neon_dot_f64(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut acc = vdupq_n_f64(0.0);
    let mut i = 0;
    while i + 1 < n {
        let xv = vld1q_f64(x.as_ptr().add(i));
        let yv = vld1q_f64(y.as_ptr().add(i));
        acc = vaddq_f64(acc, vmulq_f64(xv, yv));
        i += 2;
    }
    let mut s = hsum_f64x2(acc);
    while i < n { s += *x.get_unchecked(i) * *y.get_unchecked(i); i += 1; }
    s
}

/// NEON dot product for f32.
#[target_feature(enable = "neon")]
pub unsafe fn neon_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 3 < n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let yv = vld1q_f32(y.as_ptr().add(i));
        acc = vaddq_f32(acc, vmulq_f32(xv, yv));
        i += 4;
    }
    let mut s = hsum_f32x4(acc);
    while i < n { s += *x.get_unchecked(i) * *y.get_unchecked(i); i += 1; }
    s
}

// ─── NORM2  ‖x‖₂ ─────────────────────────────────────────────────────────────

/// NEON squared-norm accumulation for f64 (return √).
#[target_feature(enable = "neon")]
pub unsafe fn neon_norm2_f64(x: &[f64]) -> f64 {
    let n = x.len();
    let mut acc = vdupq_n_f64(0.0);
    let mut i = 0;
    while i + 1 < n {
        let xv = vld1q_f64(x.as_ptr().add(i));
        // Use separate mul+add (not FMA) to match scalar floating-point semantics.
        acc = vaddq_f64(acc, vmulq_f64(xv, xv));
        i += 2;
    }
    let mut s = hsum_f64x2(acc);
    while i < n { s += x[i] * x[i]; i += 1; }
    s.sqrt()
}

/// NEON squared-norm accumulation for f32 (return √).
#[target_feature(enable = "neon")]
pub unsafe fn neon_norm2_f32(x: &[f32]) -> f32 {
    let n = x.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 3 < n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        // Use separate mul+add to match scalar floating-point semantics.
        acc = vaddq_f32(acc, vmulq_f32(xv, xv));
        i += 4;
    }
    let mut s = hsum_f32x4(acc);
    while i < n { s += x[i] * x[i]; i += 1; }
    s.sqrt()
}

// ─── Jacobi scale  x ← x + α·D⁻¹·v ─────────────────────────────────────────

/// NEON Jacobi scale for f64: `x ← x + α·(v / diag)`.
#[target_feature(enable = "neon")]
pub unsafe fn neon_jacobi_scale_f64(x: &mut [f64], diag: &[f64], v: &[f64], alpha: f64) {
    let n = x.len();
    let av = vdupq_n_f64(alpha);
    let one = vdupq_n_f64(1.0);
    let mut i = 0;
    while i + 1 < n {
        let dv = vld1q_f64(diag.as_ptr().add(i));
        let vv = vld1q_f64(v.as_ptr().add(i));
        let xv = vld1q_f64(x.as_ptr().add(i));
        let inv_d = vdivq_f64(one, dv);
        // x += alpha * v * inv_d
        let rv = vfmaq_f64(xv, av, vmulq_f64(vv, inv_d));
        vst1q_f64(x.as_mut_ptr().add(i), rv);
        i += 2;
    }
    while i < n {
        if diag[i] != 0.0 {
            x[i] += alpha * v[i] / diag[i];
        }
        i += 1;
    }
}

// ─── SUB  z = x - y ──────────────────────────────────────────────────────────

/// NEON element-wise subtraction for f64: `z ← x - y`.
#[target_feature(enable = "neon")]
pub unsafe fn neon_sub_f64(x: &[f64], y: &[f64], z: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    let n = x.len();
    let mut i = 0;
    while i + 1 < n {
        let xv = vld1q_f64(x.as_ptr().add(i));
        let yv = vld1q_f64(y.as_ptr().add(i));
        vst1q_f64(z.as_mut_ptr().add(i), vsubq_f64(xv, yv));
        i += 2;
    }
    while i < n { z[i] = x[i] - y[i]; i += 1; }
}

// ─── SCALE  y ← α·y ──────────────────────────────────────────────────────────

/// NEON in-place scale for f64: `y ← α·y`.
#[target_feature(enable = "neon")]
pub unsafe fn neon_scale_f64(alpha: f64, y: &mut [f64]) {
    let n = y.len();
    let av = vdupq_n_f64(alpha);
    let mut i = 0;
    while i + 1 < n {
        let yv = vld1q_f64(y.as_ptr().add(i));
        vst1q_f64(y.as_mut_ptr().add(i), vmulq_f64(av, yv));
        i += 2;
    }
    while i < n { y[i] *= alpha; i += 1; }
}

/// NEON in-place scale for f32: `y ← α·y`.
#[target_feature(enable = "neon")]
pub unsafe fn neon_scale_f32(alpha: f32, y: &mut [f32]) {
    let n = y.len();
    let av = vdupq_n_f32(alpha);
    let mut i = 0;
    while i + 3 < n {
        let yv = vld1q_f32(y.as_ptr().add(i));
        vst1q_f32(y.as_mut_ptr().add(i), vmulq_f32(av, yv));
        i += 4;
    }
    while i < n { y[i] *= alpha; i += 1; }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vecs(n: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (n - i) as f64 / n as f64).collect();
        (x, y)
    }

    #[test]
    fn neon_axpy_f64_correctness() {
        let n = 13; // odd — tests scalar tail
        let alpha = 2.5_f64;
        let (x, mut y_neon) = make_vecs(n);
        let mut y_ref = y_neon.clone();

        for i in 0..n { y_ref[i] += alpha * x[i]; }
        unsafe { neon_axpy_f64(alpha, &x, &mut y_neon); }

        for i in 0..n {
            assert!((y_neon[i] - y_ref[i]).abs() < 1e-13,
                "i={i}: {} vs {}", y_neon[i], y_ref[i]);
        }
    }

    #[test]
    fn neon_axpby_f64_correctness() {
        let n = 17;
        let alpha = 3.0_f64;
        let beta  = -0.5_f64;
        let (x, mut y_neon) = make_vecs(n);
        let mut y_ref = y_neon.clone();

        for i in 0..n { y_ref[i] = alpha * x[i] + beta * y_ref[i]; }
        unsafe { neon_axpby_f64(alpha, &x, beta, &mut y_neon); }

        for i in 0..n {
            assert!((y_neon[i] - y_ref[i]).abs() < 1e-13,
                "i={i}: {} vs {}", y_neon[i], y_ref[i]);
        }
    }

    #[test]
    fn neon_dot_f64_correctness() {
        let n = 20;
        let (x, y) = make_vecs(n);
        let ref_dot: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        let neon_dot = unsafe { neon_dot_f64(&x, &y) };
        assert!((neon_dot - ref_dot).abs() < 1e-12,
            "dot: {} vs {}", neon_dot, ref_dot);
    }

    #[test]
    fn neon_norm2_f64_correctness() {
        let n = 15;
        let (x, _) = make_vecs(n);
        let ref_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let neon_norm = unsafe { neon_norm2_f64(&x) };
        assert!((neon_norm - ref_norm).abs() < 1e-12,
            "norm2: {} vs {}", neon_norm, ref_norm);
    }

    #[test]
    fn neon_row_dot_f64_correctness() {
        // Simulate a sparse row: values = [1.0, 2.0, 3.0], col_idx = [0, 2, 4]
        let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let col_idx = vec![0usize, 2, 4, 1, 3];
        let x = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0];
        // row 0: uses first 3 entries → 1*10 + 2*30 + 3*50 = 10+60+150 = 220
        let ref_val = 1.0 * x[col_idx[0]] + 2.0 * x[col_idx[1]] + 3.0 * x[col_idx[2]];
        let neon_val = unsafe { neon_row_dot_f64(&col_idx, &values, &x, 0, 3) };
        assert!((neon_val - ref_val).abs() < 1e-12,
            "row_dot: {} vs {}", neon_val, ref_val);
    }

    #[test]
    fn neon_jacobi_scale_f64_correctness() {
        let n = 11;
        let alpha = 0.5_f64;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let v: Vec<f64>    = (0..n).map(|i| i as f64 * 2.0).collect();
        let mut x_neon = vec![1.0_f64; n];
        let mut x_ref  = x_neon.clone();

        for i in 0..n { x_ref[i] += alpha * v[i] / diag[i]; }
        unsafe { neon_jacobi_scale_f64(&mut x_neon, &diag, &v, alpha); }

        for i in 0..n {
            assert!((x_neon[i] - x_ref[i]).abs() < 1e-13,
                "i={i}: {} vs {}", x_neon[i], x_ref[i]);
        }
    }

    #[test]
    fn neon_axpy_f32_correctness() {
        let n = 19;
        let alpha = 1.5_f32;
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let mut y_neon: Vec<f32> = (0..n).map(|i| (n - i) as f32 / n as f32).collect();
        let mut y_ref = y_neon.clone();

        for i in 0..n { y_ref[i] += alpha * x[i]; }
        unsafe { neon_axpy_f32(alpha, &x, &mut y_neon); }

        for i in 0..n {
            assert!((y_neon[i] - y_ref[i]).abs() < 1e-6,
                "i={i}: {} vs {}", y_neon[i], y_ref[i]);
        }
    }
}
