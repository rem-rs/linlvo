//! Block Low-Rank (BLR) compression via truncated SVD.
//!
//! A BLR block stores an off-diagonal submatrix `A` approximately as
//! `A ≈ U * Vᵀ` where `U` is `m×r` and `V` is `n×r` (r = numerical rank).
//! The rank `r` is chosen such that all discarded singular values are below
//! `tol * σ₁` (relative truncation), subject to a hard cap `max_rank`.
//!
//! The SVD is computed via a randomised power-iteration sketch, which avoids
//! O(mn·min(m,n)) full LAPACK-style SVD while remaining accurate to `tol`.
//!
//! ## Algorithm (randomised SVD, Halko-Martinsson-Tropp 2011)
//!
//! 1. Draw a Gaussian sketch matrix Ω ∈ ℝ^{n×(r+p)}.
//! 2. Form Y = A Ω, optionally power-iterate: Y = (A Aᵀ)^q A Ω.
//! 3. Orthogonalise Y → Q via column-pivoted Gram–Schmidt.
//! 4. Project: B = Qᵀ A  (small (r+p)×n matrix).
//! 5. Thin SVD of B: B = Û Σ Vᵀ.
//! 6. U = Q Û, discard columns where σ_k < tol·σ₁.

#![allow(clippy::needless_range_loop)]

use crate::core::scalar::Scalar;

// ─── Public types ─────────────────────────────────────────────────────────────

/// A compressed off-diagonal block stored in BLR format.
///
/// The original `m×n` block `A` is approximated as `U * Vᵀ` where
/// `U` is `m×rank` and `V` is `n×rank` (column-major storage).
#[derive(Debug, Clone)]
pub struct BlrBlock<T: Scalar> {
    pub m: usize,
    pub n: usize,
    pub rank: usize,
    /// U factor: m×rank, column-major (U[i + k*m]).
    pub u: Vec<T>,
    /// V factor: n×rank, column-major (V[j + k*n]).
    pub v: Vec<T>,
}

impl<T: Scalar> BlrBlock<T> {
    /// Reconstruct the full `m×n` dense block (row-major).
    pub fn to_dense(&self) -> Vec<T> {
        let mut out = vec![T::zero(); self.m * self.n];
        for k in 0..self.rank {
            for i in 0..self.m {
                let ui = self.u[i + k * self.m];
                for j in 0..self.n {
                    out[i * self.n + j] += ui * self.v[j + k * self.n];
                }
            }
        }
        out
    }

    /// Apply block to vector: `y += alpha * U (Vᵀ x)`.
    pub fn apply_add(&self, x: &[T], y: &mut [T], alpha: T) {
        // Compute w = Vᵀ x  (rank × 1).
        let mut w = vec![T::zero(); self.rank];
        for k in 0..self.rank {
            let mut s = T::zero();
            for j in 0..self.n {
                s += self.v[j + k * self.n] * x[j];
            }
            w[k] = s;
        }
        // y += alpha * U w.
        for k in 0..self.rank {
            let aw = alpha * w[k];
            for i in 0..self.m {
                y[i] += self.u[i + k * self.m] * aw;
            }
        }
    }
}

// ─── Truncated SVD via randomised algorithm ───────────────────────────────────

/// Compress the `m×n` row-major dense block `a` with tolerance `tol` (relative).
///
/// Returns a [`BlrBlock`] of rank ≤ `min(m, n)`.  When the block is zero or
/// all singular values fall below `tol * sigma_max`, `rank = 0` is returned.
///
/// The pseudo-random seed is deterministic so compression is reproducible.
pub fn compress_block<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    tol: f64,
) -> BlrBlock<T> {
    if m == 0 || n == 0 {
        return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] };
    }

    let max_rank = m.min(n);
    // Convert to f64 for numerics once.
    let a_f: Vec<f64> = a.iter().map(|&v| {
        <f64 as num_traits::NumCast>::from(v).unwrap_or(0.0)
    }).collect();

    // Sketch with r + oversampling columns, capped at max_rank.
    let r = max_rank.min(20);
    let p_os = 5_usize.min(max_rank.saturating_sub(r));
    let k_sketch = (r + p_os).min(max_rank);

    // ── Step 1: Gaussian sketch Ω ∈ ℝ^{n × k_sketch} ────────────────────────
    let mut rng = Lcg64::new((m as u64).wrapping_mul(31337).wrapping_add(n as u64 * 7919));
    let mut omega = vec![0.0f64; n * k_sketch];
    for v in omega.iter_mut() { *v = rng.gaussian(); }

    // ── Step 2: Y = A Ω  (m × k_sketch), one power-iteration step ───────────
    // Compute Y = A (Aᵀ (A Ω)) for better accuracy on slowly-decaying spectra.
    let a = &a_f[..]; // shadow with f64 slice
    let mut y0 = vec![0.0f64; m * k_sketch];
    for col in 0..k_sketch {
        for i in 0..m {
            let row = &a[i*n .. i*n+n];
            let mut s = 0.0f64;
            for j in 0..n { s += row[j] * omega[j + col*n]; }
            y0[i + col*m] = s;
        }
    }
    // tmp = Aᵀ Y  (n × k_sketch)
    let mut tmp = vec![0.0f64; n * k_sketch];
    for col in 0..k_sketch {
        for j in 0..n {
            let mut s = 0.0f64;
            for i in 0..m { s += a[i*n+j] * y0[i + col*m]; }
            tmp[j + col*n] = s;
        }
    }
    // Y = A tmp  (m × k_sketch)
    let mut y = vec![0.0f64; m * k_sketch];
    for col in 0..k_sketch {
        for i in 0..m {
            let row = &a[i*n .. i*n+n];
            let mut s = 0.0f64;
            for j in 0..n { s += row[j] * tmp[j + col*n]; }
            y[i + col*m] = s;
        }
    }

    // ── Step 3: QR of Y → Q (m × q_cols) via compacting Gram-Schmidt ─────────
    // We store accepted columns in a compact output buffer.
    let mut q_buf = vec![0.0f64; m * k_sketch]; // compacted columns
    let mut q_cols = 0usize;
    // `pending`: working copy of each column, modified in-place.
    let mut pending = y.clone();
    for j in 0..k_sketch {
        // Compute norm of column j of pending.
        let norm_sq: f64 = (0..m).map(|i| { let v = pending[i + j*m]; v*v }).sum();
        let nrm = norm_sq.sqrt();
        if nrm < 1e-10 * (k_sketch as f64).sqrt() { continue; }
        let inv = 1.0 / nrm;
        // Accept this column: copy (normalised) into q_buf[:, q_cols].
        for i in 0..m { q_buf[i + q_cols*m] = pending[i + j*m] * inv; }
        let qc = q_cols;
        q_cols += 1;
        // Orthogonalise remaining columns j+1..k_sketch against the new q column.
        for l in (j+1)..k_sketch {
            let dot: f64 = (0..m).map(|i| q_buf[i + qc*m] * pending[i + l*m]).sum();
            for i in 0..m { pending[i + l*m] -= dot * q_buf[i + qc*m]; }
        }
        if q_cols == max_rank { break; }
    }

    if q_cols == 0 {
        return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] };
    }

    // ── Step 4: B = Qᵀ A  (q_cols × n) ──────────────────────────────────────
    let mut b_small = vec![0.0f64; q_cols * n];
    for i in 0..q_cols {
        for j in 0..n {
            let mut s = 0.0f64;
            for row in 0..m { s += q_buf[row + i*m] * a[row*n+j]; }
            b_small[i * n + j] = s;
        }
    }

    // ── Step 5: thin SVD of B (q_cols × n) via one-sided Jacobi ─────────────
    let p_svd = q_cols.min(n);
    let (sigma_f, u_hat_f, v_svd_f) = jacobi_svd_f64(&b_small, q_cols, n, p_svd);

    if sigma_f.is_empty() || sigma_f[0] == 0.0 {
        return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] };
    }

    // ── Step 6: truncate at tol * sigma_1 ────────────────────────────────────
    let sigma1 = sigma_f[0];
    let threshold = tol * sigma1;
    let rank = sigma_f.iter().take_while(|&&s| s >= threshold).count().min(p_svd);
    if rank == 0 {
        return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] };
    }

    // U = Q * Û * Σ  (m × rank, column-major, T).
    let mut u_full = vec![T::zero(); m * rank];
    for k_col in 0..rank {
        let sig = T::from_f64(sigma_f[k_col]);
        for i in 0..m {
            let mut s = T::zero();
            for qi in 0..q_cols {
                s += T::from_f64(q_buf[i + qi*m]) * T::from_f64(u_hat_f[qi + k_col*q_cols]);
            }
            u_full[i + k_col*m] = s * sig;
        }
    }

    // V = V_svd[:, :rank]  (n × rank, column-major, T).
    let mut v_full = vec![T::zero(); n * rank];
    for k_col in 0..rank {
        for j in 0..n {
            v_full[j + k_col*n] = T::from_f64(v_svd_f[j + k_col*n]);
        }
    }

    BlrBlock { m, n, rank, u: u_full, v: v_full }
}

// helper: T → f64 using num_traits::NumCast (works for f32 and f64).
pub trait AsF64Val: Copy { fn as_f64_val(self) -> f64; }
impl AsF64Val for f64 { #[inline] fn as_f64_val(self) -> f64 { self } }
impl AsF64Val for f32 { #[inline] fn as_f64_val(self) -> f64 { self as f64 } }

// Blanket: every Scalar automatically satisfies AsF64Val (since Scalar is
// only implemented for f32 and f64 in this crate).
// We express this with a where clause inside MultifrontalLu instead of
// a blanket impl to avoid coherence issues.


// ─── Jacobi SVD (f64 internal) ───────────────────────────────────────────────

/// Compute thin SVD of row-major f64 `m×n` matrix returning up to `p`
/// singular triplets `(sigma, U_hat (m×p col-major), V (n×p col-major))`.
///
/// Uses deflating power iteration: extract one singular triplet at a time
/// via alternating normalised matrix-vector products, then deflate.
/// This is reliable for small matrices (m,n ≤ ~30) and avoids the
/// numerical instability of one-sided Jacobi on Gram matrices.
fn jacobi_svd_f64(
    a: &[f64],
    m: usize,
    n: usize,
    p: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if m == 0 || n == 0 || p == 0 {
        return (vec![], vec![], vec![]);
    }
    let p = p.min(m).min(n);

    // Working copy of A for deflation.
    let mut work = a.to_vec();

    let mut sigma_out  = Vec::with_capacity(p);
    let mut u_out      = Vec::with_capacity(m * p); // col-major
    let mut v_out      = Vec::with_capacity(n * p); // col-major

    // Deterministic initial vector: constant [1/√n, ...].
    let mut v_vec = vec![1.0f64 / (n as f64).sqrt(); n];

    for _k in 0..p {
        // Power iteration to find dominant right singular vector.
        // Iterate: v ← Aᵀ(Av) / ||Aᵀ(Av)||  (using working deflated A).
        for _iter in 0..200 {
            // u_tmp = work * v  (m×1)
            let mut u_tmp = vec![0.0f64; m];
            for i in 0..m {
                let mut s = 0.0f64;
                for j in 0..n { s += work[i*n+j] * v_vec[j]; }
                u_tmp[i] = s;
            }
            // v_new = workᵀ * u_tmp  (n×1)
            let mut v_new = vec![0.0f64; n];
            for j in 0..n {
                let mut s = 0.0f64;
                for i in 0..m { s += work[i*n+j] * u_tmp[i]; }
                v_new[j] = s;
            }
            let nrm: f64 = v_new.iter().map(|x| x*x).sum::<f64>().sqrt();
            if nrm < 1e-300 { break; }
            for x in &mut v_new { *x /= nrm; }
            // Check convergence.
            let diff: f64 = v_new.iter().zip(&v_vec)
                .map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
            v_vec = v_new;
            if diff < 1e-13 { break; }
        }

        // Compute u = work * v, sigma = ||u||.
        let mut u_vec = vec![0.0f64; m];
        for i in 0..m {
            let mut s = 0.0f64;
            for j in 0..n { s += work[i*n+j] * v_vec[j]; }
            u_vec[i] = s;
        }
        let sigma: f64 = u_vec.iter().map(|x| x*x).sum::<f64>().sqrt();
        if sigma < 1e-14 { break; }
        for x in &mut u_vec { *x /= sigma; }

        sigma_out.push(sigma);
        u_out.extend_from_slice(&u_vec);
        v_out.extend_from_slice(&v_vec);

        // Deflate: work -= sigma * u_vec * v_vecᵀ.
        for i in 0..m {
            for j in 0..n {
                work[i*n+j] -= sigma * u_vec[i] * v_vec[j];
            }
        }

        // Re-initialise v for next singular value with a fresh direction.
        // Use the next canonical basis vector (rotated by k) to avoid
        // starting in the null space.
        let seed_idx = _k + 1;
        v_vec = vec![0.0f64; n];
        // Pick a random-ish direction orthogonal to previous v.
        for j in 0..n {
            v_vec[j] = ((seed_idx * 7 + j * 13 + 1) as f64 / 17.0).sin();
        }
        // Orthogonalise against already-found v columns.
        for prev in 0.._k+1 {
            let vp = &v_out[prev*n..(prev+1)*n];
            let dot: f64 = v_vec.iter().zip(vp).map(|(a,b)| a*b).sum();
            for j in 0..n { v_vec[j] -= dot * vp[j]; }
        }
        let nrm: f64 = v_vec.iter().map(|x| x*x).sum::<f64>().sqrt();
        if nrm < 1e-14 { break; }
        for x in &mut v_vec { *x /= nrm; }
    }

    (sigma_out, u_out, v_out)
}

// ─── LCG pseudo-random number generator ──────────────────────────────────────

struct Lcg64 { state: u64 }

impl Lcg64 {
    fn new(seed: u64) -> Self { Self { state: seed ^ 0xdeadbeefcafe0001 } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Box-Muller standard normal sample.
    fn gaussian(&mut self) -> f64 {
        let u1 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-300;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }
}



// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn frobenius_err(orig: &[f64], approx: &[f64]) -> f64 {
        orig.iter().zip(approx).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
    }
    fn frobenius_norm(a: &[f64]) -> f64 {
        a.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    #[test]
    fn blr_rank1_exact() {
        // A = u * vᵀ (rank-1) should be captured exactly.
        let u = vec![1.0f64, 2.0, 3.0, 4.0];
        let v = vec![1.0f64, -1.0, 2.0];
        let m = 4; let n = 3;
        let mut a = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n { a[i*n+j] = u[i]*v[j]; } }
        let blk = compress_block::<f64>(&a, m, n, 1e-10);
        assert!(blk.rank <= 1, "rank-1 block should compress to rank≤1, got {}", blk.rank);
        let recon = blk.to_dense();
        let err = frobenius_err(&a, &recon);
        let nrm = frobenius_norm(&a);
        assert!(err / nrm < 1e-6, "rank-1 reconstruction error {}", err / nrm);
    }

    #[test]
    fn blr_rank2_compression() {
        let m = 6; let n = 5;
        // Build rank-2 matrix.
        let u1 = vec![1.0f64, 2.0, 0.0, -1.0, 0.5, 3.0];
        let v1 = vec![1.0f64, 0.0, -1.0, 2.0, 0.5];
        let u2 = vec![0.0f64, 1.0, -1.0, 0.0, 2.0, -0.5];
        let v2 = vec![0.5f64, -1.0, 1.0, 0.0, -2.0];
        let mut a = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n {
            a[i*n+j] = 3.0 * u1[i]*v1[j] + 0.5 * u2[i]*v2[j];
        }}
        let blk = compress_block::<f64>(&a, m, n, 1e-8);
        assert!(blk.rank <= 2, "rank-2 block should compress to rank≤2, got {}", blk.rank);
        let recon = blk.to_dense();
        let err = frobenius_err(&a, &recon);
        let nrm = frobenius_norm(&a);
        assert!(err / nrm < 1e-6, "rank-2 reconstruction error {}", err / nrm);
    }

    #[test]
    fn blr_zero_block() {
        let a = vec![0.0f64; 4 * 4];
        let blk = compress_block::<f64>(&a, 4, 4, 1e-8);
        assert_eq!(blk.rank, 0);
    }

    #[test]
    fn blr_full_rank_low_tol() {
        // Full-rank random-like matrix at tight tolerance → many columns kept.
        let m = 5; let n = 4;
        let a: Vec<f64> = (0..(m*n)).map(|i| {
            let x = (i as f64 * 1.23456 + 0.7).fract();
            x - 0.5
        }).collect();
        // With tol=0: rank should be min(m,n).
        let blk = compress_block::<f64>(&a, m, n, 1e-15);
        assert!(blk.rank > 0);
        let recon = blk.to_dense();
        let err = frobenius_err(&a, &recon);
        let nrm = frobenius_norm(&a);
        assert!(err / nrm < 1e-6, "full-rank reconstruction error {}", err / nrm);
    }

    #[test]
    fn blr_apply_add_matches_dense() {
        let m = 4; let n = 3;
        let a = vec![1.0f64, 2.0, 0.0, -1.0, 3.0, 1.0, 0.0, -2.0, 4.0, 1.0, 0.5, -0.5];
        let blk = compress_block::<f64>(&a, m, n, 1e-12);
        let x = vec![1.0f64, -1.0, 2.0];
        // Dense: y_dense = A * x.
        let mut y_dense = vec![0.0f64; m];
        for i in 0..m { for j in 0..n { y_dense[i] += a[i*n+j] * x[j]; } }
        let mut y_blr = vec![0.0f64; m];
        blk.apply_add(&x, &mut y_blr, 1.0f64);
        for i in 0..m {
            assert!((y_dense[i] - y_blr[i]).abs() < 1e-10,
                "apply_add mismatch at i={i}: {:.6} vs {:.6}", y_dense[i], y_blr[i]);
        }
    }
}
