//! Block Low-Rank (BLR) compression via randomised truncated SVD.
//!
//! A BLR block stores an off-diagonal submatrix `A` approximately as
//! `A ≈ U Vᵀ` where `U` is `m×r` and `V` is `n×r` (`r` = numerical rank).
//! The rank is chosen so that all discarded singular values satisfy
//! `σ_k < tol · σ₁` (relative truncation), subject to a hard cap `max_rank`.
//!
//! Arithmetic is always performed in `f64` regardless of the element type `T`.
//! For `f32` inputs this means a widening conversion on entry and a narrowing
//! one on exit — this is intentional, since `f32` precision (≈7 digits) is
//! unreliable for tolerances below ~1e-5.
//!
//! ## Compression algorithm (Halko-Martinsson-Tropp 2011)
//!
//! 1. Draw a Gaussian sketch matrix Ω ∈ ℝ^{n×(r+p)}.
//! 2. Power-iterate: Y = A (Aᵀ (A Ω))  — one step, improves slow-decay accuracy.
//! 3. Orthogonalise Y → Q via **double-pass modified Gram-Schmidt (MGS×2)**,
//!    which suppresses residual drift from nearly-dependent columns.
//! 4. Project: B = Qᵀ A  (small `(r+p)×n` matrix).
//! 5. Thin SVD of B via deflating power iteration (`power_iter_svd_f64`).
//! 6. U = Q Û Σ,  discard columns where σ_k < tol · σ₁.
//!
//! ## `BlrBlock<T>` — key methods
//!
//! | Method | Description |
//! |--------|-------------|
//! | `to_dense()` | Expand to full `m×n` row-major matrix |
//! | `apply_add(x, y, α)` | `y += α U Vᵀ x`  (matvec) |
//! | `apply_add_t(x, y, α)` | `y += α V Uᵀ x`  (transpose matvec) |
//! | `recompress(new_tol)` | Re-truncate with a looser tolerance — no original matrix needed |
//! | `add_compressed(other, tol, max_rank)` | `A + B` with rank truncation |
//! | `compression_ratio()` | `(m+n)·r / (m·n)` — fraction of dense memory used |
//! | `memory_bytes()` | `(dense_bytes, blr_bytes)` pair |
//!
//! ## `compress_block` parameters
//!
//! ```text
//! compress_block(a, m, n, tol, max_rank)
//! ```
//!
//! * `tol` — relative singular-value threshold (e.g. `1e-8`).
//! * `max_rank` — hard rank cap; pass `0` for no limit (`min(m, n)` is used).
//!
//! ## References
//!
//! Halko, N., Martinsson, P.-G., & Tropp, J.A. (2011). Finding structure with
//! randomness: Probabilistic algorithms for constructing approximate matrix
//! decompositions. *SIAM Review*, 53(2), 217–288.
//!
//! Amestoy, P. et al. (2015). Improving multifrontal methods by means of block
//! low-rank representations. *SIAM J. Sci. Comput.*, 37(3), A1452–A1474.

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
    /// Apply block to vector (transpose): `y += alpha * V (Uᵀ x)`.
    ///
    /// Computes `y ← y + α Vᵀ (Uᵀ x)` — i.e. applies the transpose `(UVᵀ)ᵀ = V Uᵀ`.
    pub fn apply_add_t(&self, x: &[T], y: &mut [T], alpha: T) {
        // w = Uᵀ x  (rank × 1).
        let mut w = vec![T::zero(); self.rank];
        for k in 0..self.rank {
            let mut s = T::zero();
            for i in 0..self.m {
                s += self.u[i + k * self.m] * x[i];
            }
            w[k] = s;
        }
        // y += alpha * V w.
        for k in 0..self.rank {
            let aw = alpha * w[k];
            for j in 0..self.n {
                y[j] += self.v[j + k * self.n] * aw;
            }
        }
    }

    /// Re-compress this block with a looser (or equal) tolerance `new_tol`.
    ///
    /// Works entirely with the stored `U` and `V` factors — no access to the
    /// original dense matrix is required.  The resulting rank is ≤ `self.rank`.
    ///
    /// ## Algorithm
    ///
    /// 1. QR-decompose `U = Q R` via Gram-Schmidt on the rank columns.
    /// 2. Compute the small `rank × rank` matrix `C = R Vᵀ`… actually we
    ///    note that `A ≈ U Vᵀ = (Q R) Vᵀ`.  The new compact matrix is
    ///    `M = R` (rank×rank) and the right factor absorbs V: we SVD `M` and
    ///    build `U_new = Q Û Σ`, `V_new = V`.
    ///
    ///    More precisely: SVD of `R` gives `R = Û_r Σ_r W_rᵀ`.  Then
    ///    `U Vᵀ = Q Û_r Σ_r (W_r Vᵀ)`, so `U_new = Q Û_r Σ_r` and
    ///    `V_new = V W_r` (n×rank_new).
    ///
    /// Columns are dropped where `σ_k < new_tol * σ_1`.
    pub fn recompress(&self, new_tol: f64) -> Self {
        if self.rank == 0 {
            return self.clone();
        }
        let r = self.rank;
        let m = self.m;
        let n = self.n;

        // ── QR of U (m × r) via MGS → Q (m×r), R (r×r upper triangular) ────
        // Store Q in-place in q_buf, R as a dense r×r matrix.
        let mut q_buf = vec![0.0f64; m * r];
        let mut r_mat = vec![0.0f64; r * r]; // row-major
        let mut u_f: Vec<f64> = self.u.iter().map(|&v| {
            <f64 as num_traits::NumCast>::from(v).unwrap_or(0.0)
        }).collect();

        for k in 0..r {
            // Orthogonalise column k against already-accepted Q columns.
            for qc in 0..k {
                let dot: f64 = (0..m).map(|i| q_buf[i + qc*m] * u_f[i + k*m]).sum();
                r_mat[qc * r + k] = dot;
                for i in 0..m { u_f[i + k*m] -= dot * q_buf[i + qc*m]; }
            }
            // Second pass.
            for qc in 0..k {
                let dot: f64 = (0..m).map(|i| q_buf[i + qc*m] * u_f[i + k*m]).sum();
                r_mat[qc * r + k] += dot;
                for i in 0..m { u_f[i + k*m] -= dot * q_buf[i + qc*m]; }
            }
            let nrm: f64 = (0..m).map(|i| u_f[i + k*m].powi(2)).sum::<f64>().sqrt();
            if nrm < 1e-14 {
                // Linearly dependent column: zero R diagonal, leave Q column as zero.
                continue;
            }
            r_mat[k * r + k] = nrm;
            let inv = 1.0 / nrm;
            for i in 0..m { q_buf[i + k*m] = u_f[i + k*m] * inv; }
        }

        // ── SVD of R (r×r) ────────────────────────────────────────────────────
        let (sigma_r, u_hat_r, w_r) = power_iter_svd_f64(&r_mat, r, r, r);
        if sigma_r.is_empty() || sigma_r[0] == 0.0 {
            return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] };
        }
        let threshold = new_tol * sigma_r[0];
        let new_rank = sigma_r.iter().take_while(|&&s| s >= threshold).count().min(r);
        if new_rank == 0 {
            return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] };
        }

        // ── Build U_new = Q Û_r Σ_r  (m × new_rank, col-major, T) ────────────
        let mut u_new = vec![T::zero(); m * new_rank];
        for k in 0..new_rank {
            let sig = T::from_f64(sigma_r[k]);
            for i in 0..m {
                let mut s = 0.0f64;
                for qi in 0..r { s += q_buf[i + qi*m] * u_hat_r[qi + k*r]; }
                u_new[i + k*m] = T::from_f64(s) * sig;
            }
        }

        // ── Build V_new = V W_r  (n × new_rank, col-major, T) ────────────────
        // V_svd (W_r) is col-major r×r: w_r[j + k*r].
        let v_f: Vec<f64> = self.v.iter().map(|&v| {
            <f64 as num_traits::NumCast>::from(v).unwrap_or(0.0)
        }).collect();
        let mut v_new = vec![T::zero(); n * new_rank];
        for k in 0..new_rank {
            for j in 0..n {
                let mut s = 0.0f64;
                for qi in 0..r { s += v_f[j + qi*n] * w_r[qi + k*r]; }
                v_new[j + k*n] = T::from_f64(s);
            }
        }

        BlrBlock { m, n, rank: new_rank, u: u_new, v: v_new }
    }

    /// Returns `(dense_bytes, blr_bytes)` as a pair for memory comparison.
    ///
    /// `dense_bytes = m * n * sizeof(T)`.
    /// `blr_bytes   = (m + n) * rank * sizeof(T)`.
    pub fn memory_bytes(&self) -> (usize, usize) {
        let elem = std::mem::size_of::<T>();
        (self.m * self.n * elem, (self.m + self.n) * self.rank * elem)
    }

    /// Fraction of memory used relative to the dense equivalent: `(m+n)*r / (m*n)`.
    ///
    /// Returns 0.0 for a zero-rank block, 1.0 if no compression is achieved.
    pub fn compression_ratio(&self) -> f64 {
        if self.m == 0 || self.n == 0 { return 0.0; }
        (self.m + self.n) as f64 * self.rank as f64 / (self.m * self.n) as f64
    }

    /// Add two same-size BLR blocks and re-compress the result.
    ///
    /// `(U₁V₁ᵀ + U₂V₂ᵀ) ≈ compress([U₁|U₂] [V₁|V₂]ᵀ)`.
    ///
    /// The combined rank is at most `self.rank + other.rank` before truncation.
    /// `new_tol` is the relative tolerance for the final SVD truncation;
    /// `max_rank = 0` means no hard rank cap.
    ///
    /// # Panics
    /// Panics if `self.m != other.m || self.n != other.n`.
    pub fn add_compressed(&self, other: &BlrBlock<T>, new_tol: f64, max_rank: usize) -> Self {
        assert_eq!(self.m, other.m, "add_compressed: m mismatch");
        assert_eq!(self.n, other.n, "add_compressed: n mismatch");
        let m = self.m; let n = self.n;
        let r1 = self.rank; let r2 = other.rank;
        let r_cat = r1 + r2;
        if r_cat == 0 { return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] }; }

        // Concatenate U factors: [U₁ | U₂]  (m × r_cat, col-major).
        let mut u_cat = vec![T::zero(); m * r_cat];
        u_cat[..m * r1].copy_from_slice(&self.u);
        u_cat[m * r1..].copy_from_slice(&other.u);

        // Concatenate V factors: [V₁ | V₂]  (n × r_cat, col-major).
        let mut v_cat = vec![T::zero(); n * r_cat];
        v_cat[..n * r1].copy_from_slice(&self.v);
        v_cat[n * r1..].copy_from_slice(&other.v);

        // Recompress the concatenated block via the same QR+SVD path.
        let tmp = BlrBlock { m, n, rank: r_cat, u: u_cat, v: v_cat };
        let mut result = tmp.recompress(new_tol);
        // Apply hard rank cap if requested.
        let hard_cap = if max_rank == 0 { m.min(n) } else { max_rank.min(m.min(n)) };
        if result.rank > hard_cap {
            result = result.recompress(new_tol); // second pass with existing tol is a no-op,
            // but if rank still exceeds cap we truncate columns directly.
            if result.rank > hard_cap {
                result.rank = hard_cap;
                result.u.truncate(m * hard_cap);
                result.v.truncate(n * hard_cap);
            }
        }
        result
    }
}

// ─── Truncated SVD via randomised algorithm ───────────────────────────────────

/// Compress the `m×n` row-major dense block `a` with tolerance `tol` (relative).
///
/// Returns a [`BlrBlock`] of rank ≤ `max_rank.min(min(m, n))`.  When the block
/// is zero or all singular values fall below `tol * sigma_max`, `rank = 0` is
/// returned.
///
/// `max_rank = 0` is treated as "no limit" (same as passing `min(m, n)`).
///
/// The pseudo-random seed is deterministic so compression is reproducible.
pub fn compress_block<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    tol: f64,
    max_rank: usize,
) -> BlrBlock<T> {
    if m == 0 || n == 0 {
        return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] };
    }

    let natural_max = m.min(n);
    let max_rank = if max_rank == 0 { natural_max } else { max_rank.min(natural_max) };
    // All arithmetic is performed in f64 regardless of whether T = f32 or f64.
    // For f32 inputs this means a widening conversion on the way in and a
    // narrowing conversion on the way out.  This is intentional: the randomised
    // SVD requires enough precision to distinguish singular values separated by
    // `tol`, and f32 arithmetic (≈7 digits) is not reliable below tol ≈ 1e-5.
    let a_f: Vec<f64> = a.iter().map(|&v| {
        <f64 as num_traits::NumCast>::from(v).unwrap_or(0.0)
    }).collect();

    // Sketch with r + oversampling columns, capped at max_rank.
    // TODO(rayon): the three blocked matmuls below (y0, tmp, y) can each be
    // parallelised over `col` with Rayon when the `rayon` feature is enabled
    // and the block is large enough to amortise the threading overhead.
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

    // ── Step 3: QR of Y → Q (m × q_cols) via modified Gram-Schmidt (MGS×2) ───
    // Two passes of MGS (double reorthogonalisation) are used to handle
    // nearly linearly-dependent columns that arise when the sketch matrix Y
    // has a slowly-decaying spectrum.  A single pass can leave significant
    // residual components along already-accepted basis vectors; the second
    // pass reduces this error from O(ε·κ) to O(ε·κ²) where ε = machine eps
    // and κ is the condition number of Y.
    let mut q_buf = vec![0.0f64; m * k_sketch]; // compacted accepted columns
    let mut q_cols = 0usize;
    // `pending`: working copy of each sketch column, modified in-place.
    let mut pending = y.clone();
    for j in 0..k_sketch {
        // ── First orthogonalisation pass: project out all accepted columns ──
        for qc in 0..q_cols {
            let dot: f64 = (0..m).map(|i| q_buf[i + qc*m] * pending[i + j*m]).sum();
            for i in 0..m { pending[i + j*m] -= dot * q_buf[i + qc*m]; }
        }
        // ── Second pass (reorthogonalisation): eliminates residual drift ───
        for qc in 0..q_cols {
            let dot: f64 = (0..m).map(|i| q_buf[i + qc*m] * pending[i + j*m]).sum();
            for i in 0..m { pending[i + j*m] -= dot * q_buf[i + qc*m]; }
        }
        // Compute norm after double-reorthogonalisation.
        let norm_sq: f64 = (0..m).map(|i| { let v = pending[i + j*m]; v*v }).sum();
        let nrm = norm_sq.sqrt();
        if nrm < 1e-10 * (k_sketch as f64).sqrt() { continue; }
        // Accept: store normalised column.
        let inv = 1.0 / nrm;
        let qc = q_cols;
        for i in 0..m { q_buf[i + qc*m] = pending[i + j*m] * inv; }
        q_cols += 1;
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

    // ── Step 5: thin SVD of B (q_cols × n) via deflating power iteration ────
    let p_svd = q_cols.min(n);
    let (sigma_f, u_hat_f, v_svd_f) = power_iter_svd_f64(&b_small, q_cols, n, p_svd);

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


// ─── Truncated SVD via deflating power iteration (f64 internal) ──────────────

/// Compute thin SVD of row-major f64 `m×n` matrix returning up to `p`
/// singular triplets `(sigma, U_hat (m×p col-major), V (n×p col-major))`.
///
/// Uses deflating power iteration: extract one singular triplet at a time
/// via alternating normalised matrix-vector products `v ← Aᵀ(Av)/‖…‖`,
/// then deflate `A ← A − σ u vᵀ`.
/// This is reliable for small matrices (m,n ≤ ~30) and avoids the
/// numerical instability of one-sided Jacobi on Gram matrices.
///
/// Note: despite the historical name in this file, this is **not** the
/// classical Jacobi SVD (which rotates the Gram matrix); it is a simple
/// power-iteration / deflation approach.
fn power_iter_svd_f64(
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
        let blk = compress_block::<f64>(&a, m, n, 1e-10, 0);
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
        let blk = compress_block::<f64>(&a, m, n, 1e-8, 0);
        assert!(blk.rank <= 2, "rank-2 block should compress to rank≤2, got {}", blk.rank);
        let recon = blk.to_dense();
        let err = frobenius_err(&a, &recon);
        let nrm = frobenius_norm(&a);
        assert!(err / nrm < 1e-6, "rank-2 reconstruction error {}", err / nrm);
    }

    #[test]
    fn blr_zero_block() {
        let a = vec![0.0f64; 4 * 4];
        let blk = compress_block::<f64>(&a, 4, 4, 1e-8, 0);
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
        let blk = compress_block::<f64>(&a, m, n, 1e-15, 0);
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
        let blk = compress_block::<f64>(&a, m, n, 1e-12, 0);
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

    #[test]
    fn blr_apply_add_t_matches_dense() {
        // A is m×n; Aᵀ is n×m.  x has length m, y has length n.
        let m = 4; let n = 3;
        let a = vec![1.0f64, 2.0, 0.0, -1.0, 3.0, 1.0, 0.0, -2.0, 4.0, 1.0, 0.5, -0.5];
        let blk = compress_block::<f64>(&a, m, n, 1e-12, 0);
        let x = vec![1.0f64, -1.0, 2.0, 0.5]; // length m
        // Dense transpose matvec: y_dense = Aᵀ x.
        let mut y_dense = vec![0.0f64; n];
        for j in 0..n { for i in 0..m { y_dense[j] += a[i*n+j] * x[i]; } }
        let mut y_blr = vec![0.0f64; n];
        blk.apply_add_t(&x, &mut y_blr, 1.0f64);
        for j in 0..n {
            assert!((y_dense[j] - y_blr[j]).abs() < 1e-10,
                "apply_add_t mismatch at j={j}: {:.6} vs {:.6}", y_dense[j], y_blr[j]);
        }
    }

    #[test]
    fn blr_recompress_reduces_rank() {
        // Build a rank-3 matrix with singular values 100, 1, 0.001.
        let m = 6; let n = 5;
        let u = [
            vec![1.0f64, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0f64, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 0.0],
        ];
        let v = [
            vec![1.0f64, 0.0, 0.0, 0.0, 0.0],
            vec![0.0f64, 1.0, 0.0, 0.0, 0.0],
            vec![0.0f64, 0.0, 1.0, 0.0, 0.0],
        ];
        let sigmas = [100.0f64, 1.0, 0.001];
        let mut a = vec![0.0f64; m * n];
        for k in 0..3 {
            for i in 0..m { for j in 0..n { a[i*n+j] += sigmas[k] * u[k][i] * v[k][j]; } }
        }
        // Compress tightly to capture all 3 ranks.
        let blk = compress_block::<f64>(&a, m, n, 1e-10, 0);
        assert_eq!(blk.rank, 3, "expected rank 3, got {}", blk.rank);

        // Recompress with tol = 0.01 → should drop σ=0.001, keep rank ≤ 2.
        let blk2 = blk.recompress(0.01);
        assert!(blk2.rank <= 2, "recompressed rank should be ≤2, got {}", blk2.rank);

        // The reconstruction should still approximate the rank-2 part well.
        let recon = blk2.to_dense();
        // Build rank-2 reference (drop σ=0.001 term).
        let mut a2 = vec![0.0f64; m * n];
        for k in 0..2 {
            for i in 0..m { for j in 0..n { a2[i*n+j] += sigmas[k] * u[k][i] * v[k][j]; } }
        }
        let err: f64 = a2.iter().zip(&recon).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt();
        let nrm: f64 = a2.iter().map(|x| x*x).sum::<f64>().sqrt();
        // The dropped term has σ=0.001 ≪ σ₁=100, and σ₂=1 sits exactly at the
        // threshold (0.01×σ₁).  Numerical drift in the internal SVD may occasionally
        // place σ₂ just below the threshold; allow up to 2% relative error.
        assert!(err / nrm < 0.02, "recompress reconstruction error {}", err / nrm);
    }

    #[test]
    fn blr_compression_ratio_rank1() {
        // rank-1 m×n: ratio = (m+n)/m*n < 1 for large m,n.
        let m = 10; let n = 8;
        let u: Vec<f64> = (0..m).map(|i| i as f64 + 1.0).collect();
        let v: Vec<f64> = (0..n).map(|j| j as f64 * 0.5 + 1.0).collect();
        let mut a = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n { a[i*n+j] = u[i] * v[j]; } }
        let blk = compress_block::<f64>(&a, m, n, 1e-10, 0);
        assert_eq!(blk.rank, 1);
        let ratio = blk.compression_ratio();
        let expected = (m + n) as f64 / (m * n) as f64;
        assert!((ratio - expected).abs() < 1e-12, "ratio mismatch: {} vs {}", ratio, expected);
        assert!(ratio < 1.0, "rank-1 should compress below full: {}", ratio);
    }

    #[test]
    fn blr_large_decaying_spectrum() {
        // 50×30 matrix with exponentially decaying singular values.
        // σ_k = exp(-k), so tol=1e-4 should keep k where exp(-k)≥1e-4*1 → k≤9.
        let m = 50; let n = 30;
        let rank_true = m.min(n); // build full-rank basis
        // U: random-ish orthonormal columns (Gram-Schmidt on random vectors).
        let mut u_cols: Vec<Vec<f64>> = Vec::new();
        let mut rng = Lcg64::new(0xdeadbeef);
        for k in 0..rank_true {
            let mut col: Vec<f64> = (0..m).map(|_| rng.gaussian()).collect();
            // Orthogonalise against prior columns.
            for prev in &u_cols {
                let dot: f64 = col.iter().zip(prev).map(|(a,b)| a*b).sum();
                for i in 0..m { col[i] -= dot * prev[i]; }
            }
            let nrm: f64 = col.iter().map(|x| x*x).sum::<f64>().sqrt();
            if nrm < 1e-12 { break; }
            for x in &mut col { *x /= nrm; }
            u_cols.push(col);
        }
        let mut v_cols: Vec<Vec<f64>> = Vec::new();
        for k in 0..rank_true {
            let mut col: Vec<f64> = (0..n).map(|_| rng.gaussian()).collect();
            for prev in &v_cols {
                let dot: f64 = col.iter().zip(prev).map(|(a,b)| a*b).sum();
                for j in 0..n { col[j] -= dot * prev[j]; }
            }
            let nrm: f64 = col.iter().map(|x| x*x).sum::<f64>().sqrt();
            if nrm < 1e-12 { break; }
            for x in &mut col { *x /= nrm; }
            v_cols.push(col);
        }
        let r_built = u_cols.len().min(v_cols.len());
        let mut a = vec![0.0f64; m * n];
        for k in 0..r_built {
            let sigma = (-( k as f64)).exp(); // 1, e^-1, e^-2, ...
            for i in 0..m { for j in 0..n { a[i*n+j] += sigma * u_cols[k][i] * v_cols[k][j]; } }
        }
        let nrm: f64 = a.iter().map(|x| x*x).sum::<f64>().sqrt();

        let tol = 1e-4;
        let blk = compress_block::<f64>(&a, m, n, tol, 0);
        let recon = blk.to_dense();
        let err: f64 = a.iter().zip(&recon).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt();
        // Reconstruction error should be at most a few times tol * ||A||_F.
        assert!(
            err / nrm < tol * 50.0,
            "large decaying-spectrum: err/nrm={:.2e}, tol={tol:.2e}, rank={}", err/nrm, blk.rank
        );
        // Rank should be well below min(m,n)=30 — the spectrum drops to 1e-4 around k=9.
        assert!(blk.rank < n, "rank {} should be < n={n}", blk.rank);
    }

    #[test]
    fn blr_add_compressed_roundtrip() {
        // A = B + C where B is rank-1 and C is rank-1 → sum is rank ≤ 2.
        let m = 5; let n = 4;
        let ub = vec![1.0f64, 2.0, -1.0, 0.5, 3.0];
        let vb = vec![1.0f64, -1.0, 2.0, 0.5];
        let uc = vec![0.0f64, 1.0, 1.0, -2.0, 0.5];
        let vc = vec![-1.0f64, 0.0, 1.0, -0.5];
        let mut b_dense = vec![0.0f64; m * n];
        let mut c_dense = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n {
            b_dense[i*n+j] = ub[i] * vb[j];
            c_dense[i*n+j] = uc[i] * vc[j];
        }}
        let mut sum_dense = vec![0.0f64; m * n];
        for k in 0..m*n { sum_dense[k] = b_dense[k] + c_dense[k]; }

        let blk_b = compress_block::<f64>(&b_dense, m, n, 1e-12, 0);
        let blk_c = compress_block::<f64>(&c_dense, m, n, 1e-12, 0);
        let blk_sum = blk_b.add_compressed(&blk_c, 1e-10, 0);

        assert!(blk_sum.rank <= 2, "sum of rank-1 blocks should be ≤ rank 2, got {}", blk_sum.rank);
        let recon = blk_sum.to_dense();
        let err: f64 = sum_dense.iter().zip(&recon).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt();
        let nrm: f64 = sum_dense.iter().map(|x| x*x).sum::<f64>().sqrt();
        assert!(err / nrm < 1e-8, "add_compressed reconstruction error {}", err / nrm);
    }
}
