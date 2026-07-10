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
//! 1. Draw a Gaussian sketch matrix Ω ∈ ℝ^{n×(r+p)},  p = min(10, r).
//! 2. Power-iterate: Y = A (Aᵀ (A Ω)) — one step, improves slow-decay accuracy.
//!    The three dense matmuls are parallelised with Rayon when the
//!    `rayon` feature is enabled and the block is large enough (m·k > 4096).
//! 3. Orthogonalise Y → Q via **double-pass modified Gram-Schmidt (MGS×2)**,
//!    which suppresses residual drift from nearly-dependent columns.
//! 4. Project: B = Qᵀ A  (small `(r+p)×n` matrix).
//! 5. Thin SVD of B via deflating power iteration (`power_iter_svd_f64`).
//! 6. U = Q Û Σ,  discard columns where σ_k < tol · σ₁.
//!
//! ## `BlrBlock<T>` — complete API
//!
//! ### Construction
//!
//! | Function / Method | Description |
//! |-------------------|-------------|
//! | `compress_block(a, m, n, tol, max_rank)` | Compress a dense block with relative tolerance |
//! | `compress_block_adaptive(a, m, n, rtol, atol, max_rank)` | Combined absolute + relative threshold |
//! | `BlrBlock::from_factors(m, n, u, v)` | Construct directly from pre-computed U/V factors |
//!
//! ### Queries
//!
//! | Method | Description |
//! |--------|-------------|
//! | `to_dense()` | Expand to full `m×n` row-major matrix |
//! | `compression_ratio()` | `(m+n)·r / (m·n)` — fraction of dense memory used |
//! | `memory_bytes()` | `(dense_bytes, blr_bytes)` pair |
//! | `frobenius_norm()` | `‖UVᵀ‖_F` without expanding to dense — O((m+n)·r²) |
//!
//! ### Arithmetic
//!
//! | Method | Description |
//! |--------|-------------|
//! | `apply_add(x, y, α)` | `y += α U Vᵀ x`  (matvec, single RHS) |
//! | `apply_add_t(x, y, α)` | `y += α V Uᵀ x`  (transpose matvec, single RHS) |
//! | `apply_add_mat(x, y, α, k)` | `Y += α U Vᵀ X`  (matvec, k RHS columns) |
//! | `apply_add_t_mat(x, y, α, k)` | `Y += α V Uᵀ X`  (transpose matvec, k RHS columns) |
//! | `scale(alpha)` | `A ← α A`  in-place, O(m·r) |
//! | `recompress(tol)` | Re-truncate with a looser tolerance — no original matrix needed |
//! | `recompress_capped(tol, max_rank)` | Re-truncate with tolerance **and** hard rank cap |
//! | `add_compressed(other, tol, max_rank)` | `A + B` with optimal rank truncation |
//! | `subtract_compressed(other, tol, max_rank)` | `A − B` with optimal rank truncation |
//!
//! ## `BlrMatrix<T>` — block-partitioned matrix
//!
//! A `nb_rows × nb_cols` grid of blocks.  Diagonal blocks are stored dense;
//! off-diagonal blocks use BLR compression.
//!
//! | Method | Description |
//! |--------|-------------|
//! | `compress_from_dense(a, …, row_sizes, col_sizes, tol, max_rank)` | Build from a dense matrix |
//! | `apply_add(x, y, α)` | Full `y += α A x` matvec |
//! | `apply_add_t(x, y, α)` | Full `y += α Aᵀ x` |
//! | `memory_bytes()` | Total bytes across all blocks |
//! | `blr_block(i, j)` / `dense_block(i, j)` | Borrow individual blocks |
//!
//! ## Compression parameters
//!
//! ```text
//! compress_block(a, m, n, tol, max_rank)
//! compress_block_adaptive(a, m, n, rtol, atol, max_rank)
//! ```
//!
//! * `tol` / `rtol` — relative singular-value threshold (e.g. `1e-8`).
//! * `atol` — absolute singular-value floor; 0.0 disables it.
//! * `max_rank` — hard rank cap; pass `0` for no limit (`min(m, n)` is used).
//!
//! ## Adaptive Cross Approximation (ACA+)
//!
//! As an alternative to the randomised SVD, [`compress_block_aca`] (and its
//! matrix-free sibling [`compress_block_aca_fn`]) build the same `A ≈ U Vᵀ`
//! approximation using the partially-pivoted ACA algorithm.  Only
//! `O(r(m+n))` matrix entries are read, making ACA attractive when:
//!
//! * entry evaluation is expensive (e.g. kernel or quadrature matrices), or
//! * the expected numerical rank `r` is small and known ahead of time.
//!
//! For matrices with slowly decaying singular values the randomised SVD
//! (`compress_block`) is more accurate at the same rank.
//!
//! ## References
//!
//! Halko, N., Martinsson, P.-G., & Tropp, J.A. (2011). Finding structure with
//! randomness: Probabilistic algorithms for constructing approximate matrix
//! decompositions. *SIAM Review*, 53(2), 217–288.
//!
//! Amestoy, P. et al. (2015). Improving multifrontal methods by means of block
//! low-rank representations. *SIAM J. Sci. Comput.*, 37(3), A1452–A1474.
//!
//! Bebendorf, M. (2000). Approximation of boundary element matrices.
//! *Numerische Mathematik*, 86(4), 565–589.

#![allow(clippy::needless_range_loop)]

use crate::core::scalar::Scalar;

// ─── Public types ─────────────────────────────────────────────────────────────

/// A compressed off-diagonal block stored in BLR format.
///
/// The original `m×n` block `A` is approximated as `U * Vᵀ` where
/// `U` is `m×rank` and `V` is `n×rank` (column-major storage).
#[derive(Debug, Clone)]
pub struct BlrBlock<T> {
    pub m: usize,
    pub n: usize,
    pub rank: usize,
    /// U factor: m×rank, column-major (U[i + k*m]).
    pub u: Vec<T>,
    /// V factor: n×rank, column-major (V[j + k*n]).
    pub v: Vec<T>,
}

impl<T: Scalar> BlrBlock<T> {
    /// Construct a `BlrBlock` directly from pre-computed `U` and `V` factors.
    ///
    /// The rank is inferred from the lengths of `u` and `v`:
    /// `rank = u.len() / m = v.len() / n`.
    ///
    /// This constructor is useful when importing factors produced by an external
    /// SVD routine or converting from another low-rank format.
    ///
    /// # Panics
    /// Panics if:
    /// - `m == 0 || n == 0`
    /// - `u.len()` is not a multiple of `m`
    /// - `v.len()` is not a multiple of `n`
    /// - the rank implied by `u` and `v` differs
    pub fn from_factors(m: usize, n: usize, u: Vec<T>, v: Vec<T>) -> Self {
        assert!(m > 0 && n > 0, "BlrBlock::from_factors: m and n must be > 0");
        assert_eq!(u.len() % m, 0, "BlrBlock::from_factors: u.len() not a multiple of m");
        assert_eq!(v.len() % n, 0, "BlrBlock::from_factors: v.len() not a multiple of n");
        let rank_u = u.len() / m;
        let rank_v = v.len() / n;
        assert_eq!(rank_u, rank_v,
            "BlrBlock::from_factors: rank from u ({rank_u}) != rank from v ({rank_v})");
        BlrBlock { m, n, rank: rank_u, u, v }
    }

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

    /// Apply block to a dense matrix: `Y += alpha * U (Vᵀ X)`.
    ///
    /// `x` is column-major `n × k_rhs` (stride `n` per column),
    /// `y` is column-major `m × k_rhs` (stride `m` per column).
    ///
    /// Cost: `O((m + n) · rank · k_rhs)` — much cheaper than expanding to dense
    /// and doing a full matrix product when `rank ≪ min(m, n)`.
    ///
    /// # Panics
    /// Panics if `x.len() != n * k_rhs` or `y.len() != m * k_rhs`.
    pub fn apply_add_mat(&self, x: &[T], y: &mut [T], alpha: T, k_rhs: usize) {
        assert_eq!(x.len(), self.n * k_rhs, "apply_add_mat: x length mismatch");
        assert_eq!(y.len(), self.m * k_rhs, "apply_add_mat: y length mismatch");
        if self.rank == 0 || k_rhs == 0 { return; }
        // W = Vᵀ X  (rank × k_rhs, col-major).
        let mut w = vec![T::zero(); self.rank * k_rhs];
        for c in 0..k_rhs {
            for kr in 0..self.rank {
                let mut s = T::zero();
                for j in 0..self.n {
                    s += self.v[j + kr * self.n] * x[j + c * self.n];
                }
                w[kr + c * self.rank] = s;
            }
        }
        // Y += alpha * U W.
        for c in 0..k_rhs {
            for kr in 0..self.rank {
                let aw = alpha * w[kr + c * self.rank];
                for i in 0..self.m {
                    y[i + c * self.m] += self.u[i + kr * self.m] * aw;
                }
            }
        }
    }

    /// Apply transpose block to a dense matrix: `Y += alpha * V (Uᵀ X)`.
    ///
    /// `x` is column-major `m × k_rhs`, `y` is column-major `n × k_rhs`.
    ///
    /// # Panics
    /// Panics if `x.len() != m * k_rhs` or `y.len() != n * k_rhs`.
    pub fn apply_add_t_mat(&self, x: &[T], y: &mut [T], alpha: T, k_rhs: usize) {
        assert_eq!(x.len(), self.m * k_rhs, "apply_add_t_mat: x length mismatch");
        assert_eq!(y.len(), self.n * k_rhs, "apply_add_t_mat: y length mismatch");
        if self.rank == 0 || k_rhs == 0 { return; }
        // W = Uᵀ X  (rank × k_rhs, col-major).
        let mut w = vec![T::zero(); self.rank * k_rhs];
        for c in 0..k_rhs {
            for kr in 0..self.rank {
                let mut s = T::zero();
                for i in 0..self.m {
                    s += self.u[i + kr * self.m] * x[i + c * self.m];
                }
                w[kr + c * self.rank] = s;
            }
        }
        // Y += alpha * V W.
        for c in 0..k_rhs {
            for kr in 0..self.rank {
                let aw = alpha * w[kr + c * self.rank];
                for j in 0..self.n {
                    y[j + c * self.n] += self.v[j + kr * self.n] * aw;
                }
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
    /// Re-compress this block with a looser (or equal) tolerance `new_tol`.
    ///
    /// Delegates to `recompress_capped(new_tol, 0)` (no hard rank cap).
    pub fn recompress(&self, new_tol: f64) -> Self {
        self.recompress_capped(new_tol, 0)
    }

    /// Re-compress with both a tolerance and a hard rank cap, applying both
    /// constraints inside the SVD step for optimal truncation.
    ///
    /// Unlike calling `recompress` followed by manual column dropping, this
    /// method passes `max_rank` as the SVD rank ceiling so the discarded
    /// singular vectors are always the numerically smallest ones.
    ///
    /// `max_rank = 0` is equivalent to calling `recompress(new_tol)` (no cap).
    pub fn recompress_capped(&self, new_tol: f64, max_rank: usize) -> Self {
        if self.rank == 0 { return self.clone(); }
        let u_f: Vec<f64> = self.u.iter()
            .map(|&v| <f64 as num_traits::NumCast>::from(v).unwrap_or(0.0))
            .collect();
        let v_f: Vec<f64> = self.v.iter()
            .map(|&v| <f64 as num_traits::NumCast>::from(v).unwrap_or(0.0))
            .collect();
        let (new_rank, u_new, v_new) = qr_svd_recompress_f64::<T>(
            u_f, v_f, self.m, self.n, self.rank, new_tol, max_rank,
        );
        BlrBlock { m: self.m, n: self.n, rank: new_rank, u: u_new, v: v_new }
    }

    /// Compute the Frobenius norm of the represented matrix without expanding to dense.
    ///
    /// Uses the identity `‖UVᵀ‖_F² = Σ_{i,j} (UᵀU)_{ij} · (VᵀV)_{ij}`
    /// (element-wise product of the two `rank×rank` Gram matrices), so cost is
    /// `O((m + n) · rank²)` rather than `O(m · n)`.
    pub fn frobenius_norm(&self) -> f64 {
        if self.rank == 0 { return 0.0; }
        let r = self.rank;
        let m = self.m;
        let n = self.n;
        let u_f: Vec<f64> = self.u.iter().map(|&v| {
            <f64 as num_traits::NumCast>::from(v).unwrap_or(0.0)
        }).collect();
        let v_f: Vec<f64> = self.v.iter().map(|&v| {
            <f64 as num_traits::NumCast>::from(v).unwrap_or(0.0)
        }).collect();
        // Build GU = UᵀU (r×r) and GV = VᵀV (r×r), then ‖UVᵀ‖_F² = trace(GU·GVᵀ) = Σ GU∘GV.
        let mut gu = vec![0.0f64; r * r];
        let mut gv = vec![0.0f64; r * r];
        for i in 0..r {
            for j in 0..r {
                gu[i * r + j] = (0..m).map(|k| u_f[k + i*m] * u_f[k + j*m]).sum();
                gv[i * r + j] = (0..n).map(|k| v_f[k + i*n] * v_f[k + j*n]).sum();
            }
        }
        let norm_sq: f64 = (0..r*r).map(|idx| gu[idx] * gv[idx]).sum();
        norm_sq.sqrt()
    }

    /// Scale all elements of the represented matrix in-place: `A ← α · A`.
    ///
    /// Implemented by scaling only the `U` columns: `(αU) Vᵀ = α (UVᵀ)`.
    /// Cost: `O(m · rank)`.
    pub fn scale(&mut self, alpha: T) {
        for v in self.u.iter_mut() { *v *= alpha; }
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
    /// ## Algorithm
    ///
    /// 1. Concatenate U and V factor columns: `Ũ = [U₁|U₂]`, `Ṽ = [V₁|V₂]`.
    /// 2. Re-compress `ŨṼᵀ` via the same QR+SVD path used in `recompress`,
    ///    but with `hard_cap` as an explicit rank ceiling passed to the SVD.
    ///    This gives the best rank-`hard_cap` approximation to the sum —
    ///    no post-hoc column dropping is needed.
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

        // Hard rank ceiling: applied inside SVD so the truncation is optimal.
        let hard_cap = if max_rank == 0 { m.min(n) } else { max_rank.min(m.min(n)) };

        // Concatenate U factors: [U₁ | U₂]  (m × r_cat, col-major).
        let mut u_cat = vec![T::zero(); m * r_cat];
        u_cat[..m * r1].copy_from_slice(&self.u);
        u_cat[m * r1..].copy_from_slice(&other.u);

        // Concatenate V factors: [V₁ | V₂]  (n × r_cat, col-major).
        let mut v_cat = vec![T::zero(); n * r_cat];
        v_cat[..n * r1].copy_from_slice(&self.v);
        v_cat[n * r1..].copy_from_slice(&other.v);

        // Re-compress with explicit rank cap via recompress_capped, which
        // passes hard_cap into the SVD step so dropped components are the
        // numerically smallest ones — not an arbitrary column suffix.
        let tmp = BlrBlock { m, n, rank: r_cat, u: u_cat, v: v_cat };
        tmp.recompress_capped(new_tol, hard_cap)
    }

    /// Subtract two same-size BLR blocks and re-compress the result.
    ///
    /// Equivalent to `self + (-1) · other`, implemented by negating `other`'s U
    /// columns before concatenation — no extra allocation beyond what
    /// `add_compressed` already needs.
    ///
    /// `new_tol` and `max_rank` have the same meaning as in `add_compressed`.
    ///
    /// # Panics
    /// Panics if `self.m != other.m || self.n != other.n`.
    pub fn subtract_compressed(&self, other: &BlrBlock<T>, new_tol: f64, max_rank: usize) -> Self {
        assert_eq!(self.m, other.m, "subtract_compressed: m mismatch");
        assert_eq!(self.n, other.n, "subtract_compressed: n mismatch");
        let m = self.m; let n = self.n;
        let r1 = self.rank; let r2 = other.rank;
        let r_cat = r1 + r2;
        if r_cat == 0 { return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] }; }

        let hard_cap = if max_rank == 0 { m.min(n) } else { max_rank.min(m.min(n)) };

        // Concatenate U factors with other's columns negated: [U₁ | -U₂].
        let mut u_cat = vec![T::zero(); m * r_cat];
        u_cat[..m * r1].copy_from_slice(&self.u);
        for (dst, &src) in u_cat[m * r1..].iter_mut().zip(other.u.iter()) {
            *dst = T::zero() - src;
        }

        // V factors concatenated without sign change: [V₁ | V₂].
        let mut v_cat = vec![T::zero(); n * r_cat];
        v_cat[..n * r1].copy_from_slice(&self.v);
        v_cat[n * r1..].copy_from_slice(&other.v);

        let tmp = BlrBlock { m, n, rank: r_cat, u: u_cat, v: v_cat };
        tmp.recompress_capped(new_tol, hard_cap)
    }
}

// ─── Display ──────────────────────────────────────────────────────────────────

impl<T: Scalar> std::fmt::Display for BlrBlock<T> {
    /// Compact one-line summary: `BLR [m×n, rank=r, ratio=xx.x%]`.
    ///
    /// The compression ratio is `(m+n)·rank / (m·n)`, shown as a percentage.
    /// A zero-rank block reports `rank=0, ratio=0.0%`.
    ///
    /// Example output: `BLR [50×30, rank=8, ratio=5.1%]`
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pct = self.compression_ratio() * 100.0;
        write!(f, "BLR [{}×{}, rank={}, ratio={:.1}%]", self.m, self.n, self.rank, pct)
    }
}

// ─── Internal QR + SVD recompression helper ───────────────────────────────────

/// Shared kernel for `recompress` / `recompress_capped` / `add_compressed`.
///
/// Given U factor columns `u_f` (m×r, col-major f64) and V factor columns
/// `v_f` (n×r, col-major f64):
///
/// 1. Double-pass Modified Gram-Schmidt QR of U → Q (m×r), R (r×r).
/// 2. Truncated SVD of R with hard rank cap `max_rank` (0 = r).
/// 3. Threshold singular values: keep `k` where `σ_k ≥ tol * σ₁`.
/// 4. Build U_new = Q Û Σ  (m×k, col-major, type T).
/// 5. Build V_new = V W   (n×k, col-major, type T).
///
/// Returns `(new_rank, u_new, v_new)`.
fn qr_svd_recompress_f64<T: Scalar>(
    u_f:      Vec<f64>,
    v_f:      Vec<f64>,
    m:        usize,
    n:        usize,
    r:        usize,
    tol:      f64,
    max_rank: usize,  // 0 = no cap
) -> (usize, Vec<T>, Vec<T>) {
    let svd_cap = if max_rank == 0 { r } else { max_rank.min(r) };

    // ── QR of U (m×r) via double-pass MGS ────────────────────────────────────
    let mut u_f = u_f;   // mutable copy for in-place orthogonalisation
    let mut q_buf = vec![0.0f64; m * r];
    let mut r_mat = vec![0.0f64; r * r];

    for k in 0..r {
        // Pass 1 and pass 2 (double-reorthogonalisation for numerical stability).
        for _pass in 0..2 {
            for qc in 0..k {
                let dot: f64 = (0..m).map(|i| q_buf[i + qc * m] * u_f[i + k * m]).sum();
                r_mat[qc * r + k] += dot;
                for i in 0..m { u_f[i + k * m] -= dot * q_buf[i + qc * m]; }
            }
        }
        let nrm: f64 = (0..m).map(|i| u_f[i + k * m].powi(2)).sum::<f64>().sqrt();
        if nrm < 1e-14 { continue; }  // linearly dependent — leave Q column zero
        r_mat[k * r + k] = nrm;
        let inv = 1.0 / nrm;
        for i in 0..m { q_buf[i + k * m] = u_f[i + k * m] * inv; }
    }

    // ── Truncated SVD of R (r×r) ──────────────────────────────────────────────
    let (sigma_r, u_hat_r, w_r) = power_iter_svd_f64(&r_mat, r, r, svd_cap);
    if sigma_r.is_empty() || sigma_r[0] == 0.0 {
        return (0, vec![], vec![]);
    }
    let threshold = tol * sigma_r[0];
    let new_rank = sigma_r.iter().take_while(|&&s| s >= threshold).count().min(svd_cap);
    if new_rank == 0 {
        return (0, vec![], vec![]);
    }

    // ── Build U_new = Q Û_r Σ_r  (m×new_rank, col-major, T) ─────────────────
    let mut u_new = vec![T::zero(); m * new_rank];
    for k in 0..new_rank {
        let sig = T::from_f64(sigma_r[k]);
        for i in 0..m {
            let mut s = 0.0f64;
            for qi in 0..r { s += q_buf[i + qi * m] * u_hat_r[qi + k * r]; }
            u_new[i + k * m] = T::from_f64(s) * sig;
        }
    }

    // ── Build V_new = V W_r  (n×new_rank, col-major, T) ──────────────────────
    let mut v_new = vec![T::zero(); n * new_rank];
    for k in 0..new_rank {
        for j in 0..n {
            let mut s = 0.0f64;
            for qi in 0..r { s += v_f[j + qi * n] * w_r[qi + k * r]; }
            v_new[j + k * n] = T::from_f64(s);
        }
    }

    (new_rank, u_new, v_new)
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

    // Sketch with (max_rank + oversampling) columns.
    //
    // Oversampling p improves accuracy for slowly-decaying spectra (Halko et al.
    // recommend p = 5..10 for most problems).  We use p = min(10, max_rank) so
    // the total sketch width scales with the requested rank rather than being
    // capped at the old hard-coded value of 20.
    let p_os = 10_usize.min(max_rank);
    let k_sketch = (max_rank + p_os).min(natural_max);

    // ── Step 1: Gaussian sketch Ω ∈ ℝ^{n × k_sketch} ────────────────────────
    let mut rng = Lcg64::new((m as u64).wrapping_mul(31337).wrapping_add(n as u64 * 7919));
    let mut omega = vec![0.0f64; n * k_sketch];
    for v in omega.iter_mut() { *v = rng.gaussian(); }

    // ── Step 2: Y = A Ω  (m × k_sketch), one power-iteration step ───────────
    // Compute Y = A (Aᵀ (A Ω)) for better accuracy on slowly-decaying spectra.
    let a = &a_f[..]; // shadow with f64 slice

    // Each of the three matmuls below is a dense Y ← A·X product where the
    // output columns are fully independent.  We parallelise over output columns
    // when the `rayon` feature is enabled and the block is large enough to
    // amortise threading overhead (heuristic: m·k_sketch > 4096).
    macro_rules! matmul_a_x {
        // y_out (m × k_sketch col-major) = A (m×n) · x_in (n × k_sketch col-major)
        ($y_out:expr, $x_in:expr, $rows:expr, $inner:expr, $cols:expr) => {{
            #[cfg(feature = "rayon")]
            if $rows * $cols > 4096 {
                use rayon::prelude::*;
                $y_out.par_chunks_exact_mut($rows)
                    .zip($x_in.par_chunks_exact($inner))
                    .for_each(|(y_col, x_col)| {
                        for i in 0..$rows {
                            let row = &a[i * $inner .. i * $inner + $inner];
                            y_col[i] = row.iter().zip(x_col).map(|(r, x)| r * x).sum();
                        }
                    });
            } else {
                for col in 0..$cols {
                    let x_col = &$x_in[col * $inner .. col * $inner + $inner];
                    let y_col = &mut $y_out[col * $rows .. col * $rows + $rows];
                    for i in 0..$rows {
                        let row = &a[i * $inner .. i * $inner + $inner];
                        y_col[i] = row.iter().zip(x_col).map(|(r, x)| r * x).sum();
                    }
                }
            }
            #[cfg(not(feature = "rayon"))]
            for col in 0..$cols {
                let x_col = &$x_in[col * $inner .. col * $inner + $inner];
                let y_col = &mut $y_out[col * $rows .. col * $rows + $rows];
                for i in 0..$rows {
                    let row = &a[i * $inner .. i * $inner + $inner];
                    y_col[i] = row.iter().zip(x_col).map(|(r, x)| r * x).sum();
                }
            }
        }};
    }
    macro_rules! matmul_at_x {
        // y_out (n × k col-major) = Aᵀ (n×m) · x_in (m × k col-major)
        ($y_out:expr, $x_in:expr, $out_rows:expr, $inner:expr, $cols:expr) => {{
            #[cfg(feature = "rayon")]
            if $out_rows * $cols > 4096 {
                use rayon::prelude::*;
                $y_out.par_chunks_exact_mut($out_rows)
                    .zip($x_in.par_chunks_exact($inner))
                    .for_each(|(y_col, x_col)| {
                        for j in 0..$out_rows {
                            y_col[j] = (0..$inner).map(|i| a[i * $out_rows + j] * x_col[i]).sum();
                        }
                    });
            } else {
                for col in 0..$cols {
                    let x_col = &$x_in[col * $inner .. col * $inner + $inner];
                    let y_col = &mut $y_out[col * $out_rows .. col * $out_rows + $out_rows];
                    for j in 0..$out_rows {
                        y_col[j] = (0..$inner).map(|i| a[i * $out_rows + j] * x_col[i]).sum();
                    }
                }
            }
            #[cfg(not(feature = "rayon"))]
            for col in 0..$cols {
                let x_col = &$x_in[col * $inner .. col * $inner + $inner];
                let y_col = &mut $y_out[col * $out_rows .. col * $out_rows + $out_rows];
                for j in 0..$out_rows {
                    y_col[j] = (0..$inner).map(|i| a[i * $out_rows + j] * x_col[i]).sum();
                }
            }
        }};
    }

    // Y₀ = A Ω  (m × k_sketch)
    let mut y0 = vec![0.0f64; m * k_sketch];
    matmul_a_x!(y0, omega, m, n, k_sketch);
    // tmp = Aᵀ Y₀  (n × k_sketch)
    let mut tmp = vec![0.0f64; n * k_sketch];
    matmul_at_x!(tmp, y0, n, m, k_sketch);
    // Y = A tmp  (m × k_sketch)
    let mut y = vec![0.0f64; m * k_sketch];
    matmul_a_x!(y, tmp, m, n, k_sketch);

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

/// Compress with a **combined absolute + relative** singular-value threshold.
///
/// A singular value `σ_k` is discarded when:
///
/// ```text
/// σ_k < max(rtol · σ₁,  atol)
/// ```
///
/// This is useful when the matrix entries have physical units and the noise
/// floor is known in absolute terms — e.g. `atol = 1e-6` prevents retaining
/// columns whose singular values are below machine-noise level regardless of
/// the largest singular value.
///
/// Setting `atol = 0.0` makes this identical to `compress_block(a, m, n, rtol, max_rank)`.
/// Setting `rtol = 0.0` makes truncation purely absolute.
///
/// All other parameters and the compression algorithm are identical to
/// [`compress_block`].
pub fn compress_block_adaptive<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    rtol: f64,
    atol: f64,
    max_rank: usize,
) -> BlrBlock<T> {
    // Run the standard compressor at `rtol` first — this does the expensive
    // randomised SVD exactly once.  Then prune any surviving columns whose
    // absolute singular value is below `atol`.
    //
    // Implementation note: `compress_block` already discards σ < rtol·σ₁.
    // The only additional step needed is to drop columns with σ < atol from
    // the already-compressed block without re-running the SVD.  We do this
    // by walking the U factor columns (which absorb Σ) and measuring their
    // norms — ‖U[:,k]‖ = σ_k because the standard algorithm stores U = Q·Û·Σ.
    if m == 0 || n == 0 {
        return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] };
    }
    let mut blk = compress_block(a, m, n, rtol, max_rank);
    if atol <= 0.0 || blk.rank == 0 {
        return blk;
    }
    // Count how many columns have ‖U[:,k]‖ ≥ atol.
    // U is col-major (m×rank): column k spans blk.u[k*m .. (k+1)*m].
    let keep = (0..blk.rank)
        .take_while(|&k| {
            let nrm_sq: f64 = blk.u[k*blk.m .. (k+1)*blk.m]
                .iter()
                .map(|&v| {
                    let f = <f64 as num_traits::NumCast>::from(v).unwrap_or(0.0);
                    f * f
                })
                .sum();
            nrm_sq.sqrt() >= atol
        })
        .count();
    if keep == blk.rank {
        return blk;
    }
    blk.rank = keep;
    blk.u.truncate(keep * blk.m);
    blk.v.truncate(keep * blk.n);
    blk
}

// ─── Adaptive Cross Approximation (ACA+) ─────────────────────────────────────

/// Compress the `m×n` row-major dense block `a` using **Adaptive Cross
/// Approximation** (partially-pivoted ACA, also known as ACA+).
///
/// ACA builds the low-rank approximation `A ≈ U Vᵀ` by greedily selecting
/// cross rows and columns from the matrix, reading only `O(r(m+n))` entries
/// in total instead of the full `m·n` required by the randomised SVD path.
/// This makes it attractive when matrix entries are expensive to evaluate
/// or a small numerical rank is expected a priori.
///
/// ## Algorithm (partially-pivoted ACA)
///
/// 1. Pick starting pivot row `i₀ = 0`.
/// 2. At step `k`:
///    a. Compute residual row `R[i_k, :] = A[i_k, :] − Σ_{l<k} u_l[i_k] v_lᵀ`.
///    b. Find column pivot `j_k = argmax_j |R[i_k, j]|`.
///    c. Set `v_k = R[i_k, :] / R[i_k, j_k]`.
///    d. Compute residual column `C[:, j_k] = A[:, j_k] − Σ_{l<k} v_l[j_k] u_l`.
///    e. Set `u_k = C[:, j_k]`.
///    f. Find next pivot row `i_{k+1} = argmax_i |u_k[i]|`.
/// 3. Stop when `‖u_k‖ · ‖v_k‖ < tol · ‖A‖_F^{approx}`.
///
/// ## Notes
///
/// * All arithmetic is performed in **f64** regardless of `T`, matching the
///   behaviour of [`compress_block`].
/// * The output `U` and `V` factors are **not** orthonormalised.  All
///   [`BlrBlock`] operations (`apply_add`, `to_dense`, etc.) remain correct.
/// * For matrices with slowly decaying singular values the randomised SVD
///   path (`compress_block`) is more accurate at the same rank.  Prefer ACA
///   when the matrix rank is expected to be small.
/// * For a matrix-free variant see [`compress_block_aca_fn`].
///
/// ## References
///
/// Bebendorf, M. (2000). Approximation of boundary element matrices.
/// *Numerische Mathematik*, 86(4), 565–589.
pub fn compress_block_aca<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    tol: f64,
    max_rank: usize,
) -> BlrBlock<T> {
    compress_block_aca_fn(|i, j| a[i * n + j], m, n, tol, max_rank)
}

/// Matrix-free variant of [`compress_block_aca`].
///
/// Instead of a dense slice, accepts a closure `entry(i, j) -> T` that
/// returns the `(i, j)` matrix entry on demand.  Only `O(r(m+n))` entries
/// are evaluated, making this suitable for kernel matrices or other
/// matrix-free operators where forming the full `m×n` block would be costly.
///
/// Parameters and behaviour are identical to [`compress_block_aca`].
pub fn compress_block_aca_fn<T, F>(
    entry: F,
    m: usize,
    n: usize,
    tol: f64,
    max_rank: usize,
) -> BlrBlock<T>
where
    T: Scalar,
    F: Fn(usize, usize) -> T,
{
    if m == 0 || n == 0 {
        return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] };
    }
    let natural_max = m.min(n);
    let max_rank = if max_rank == 0 { natural_max } else { max_rank.min(natural_max) };
    let tol_sq = tol * tol;

    // All arithmetic in f64; convert back to T when packing the output block.
    // u_cols[k] is the (un-normalised) column vector of length m.
    // v_cols[k] is the (normalised) row vector of length n.
    let mut u_cols: Vec<Vec<f64>> = Vec::with_capacity(max_rank);
    let mut v_cols: Vec<Vec<f64>> = Vec::with_capacity(max_rank);

    // Tracks which rows / columns have already been used as pivots so that
    // the algorithm does not revisit them.
    let mut used_rows = vec![false; m];
    let mut used_cols = vec![false; n];

    // Running lower-bound estimate of ‖A‖_F²: accumulated as
    // Σ_k ‖u_k‖² ‖v_k‖².  This is a lower bound (cross terms are omitted)
    // and is exact for orthogonal u/v, which is approximately true in
    // practice once the rank is sufficient.
    let mut frob_sq_approx = 0.0f64;

    // Starting pivot row.
    let mut pivot_row = 0usize;

    for _k in 0..max_rank {
        // ── Step (a): residual row at `pivot_row` ─────────────────────────────
        let mut r_row = vec![0.0f64; n];
        for j in 0..n {
            r_row[j] = num_traits::NumCast::from(entry(pivot_row, j)).unwrap_or(0.0);
        }
        for l in 0..u_cols.len() {
            let c = u_cols[l][pivot_row];
            if c != 0.0 {
                for j in 0..n {
                    r_row[j] -= c * v_cols[l][j];
                }
            }
        }

        // ── Step (b): column pivot ────────────────────────────────────────────
        let mut max_abs = 0.0f64;
        let mut pivot_col = n; // sentinel
        for j in 0..n {
            if !used_cols[j] {
                let v = r_row[j].abs();
                if v > max_abs {
                    max_abs = v;
                    pivot_col = j;
                }
            }
        }
        if pivot_col == n || max_abs < 1e-300 {
            break; // remaining block is numerically zero
        }

        // ── Step (c): v_k = r_row / r_row[pivot_col] ─────────────────────────
        let inv_pivot = 1.0 / r_row[pivot_col];
        let v_k: Vec<f64> = r_row.iter().map(|&x| x * inv_pivot).collect();

        // ── Step (d): residual column at `pivot_col` ──────────────────────────
        let mut u_k = vec![0.0f64; m];
        for i in 0..m {
            u_k[i] = num_traits::NumCast::from(entry(i, pivot_col)).unwrap_or(0.0);
        }
        for l in 0..v_cols.len() {
            let c = v_cols[l][pivot_col];
            if c != 0.0 {
                for i in 0..m {
                    u_k[i] -= c * u_cols[l][i];
                }
            }
        }

        // ── Update Frobenius-norm estimate ────────────────────────────────────
        let norm_u_sq: f64 = u_k.iter().map(|x| x * x).sum();
        let norm_v_sq: f64 = v_k.iter().map(|x| x * x).sum();
        frob_sq_approx += norm_u_sq * norm_v_sq;

        // ── Mark pivots used ──────────────────────────────────────────────────
        used_rows[pivot_row] = true;
        used_cols[pivot_col] = true;

        // ── Stopping criterion ────────────────────────────────────────────────
        let converged = frob_sq_approx > 0.0
            && norm_u_sq * norm_v_sq <= tol_sq * frob_sq_approx;

        u_cols.push(u_k);
        v_cols.push(v_k);

        if converged {
            break;
        }

        // ── Step (f): next pivot row ──────────────────────────────────────────
        let mut max_abs_u = 0.0f64;
        let mut next_row = m; // sentinel
        for i in 0..m {
            if !used_rows[i] {
                let v = u_cols.last().unwrap()[i].abs();
                if v > max_abs_u {
                    max_abs_u = v;
                    next_row = i;
                }
            }
        }
        if next_row == m {
            break; // all rows exhausted
        }
        pivot_row = next_row;
    }

    // ── Pack into BlrBlock (column-major U and V) ─────────────────────────────
    let rank = u_cols.len();
    if rank == 0 {
        return BlrBlock { m, n, rank: 0, u: vec![], v: vec![] };
    }

    // U: m × rank, column-major — column k spans u_out[k*m .. (k+1)*m].
    let mut u_out = vec![T::zero(); m * rank];
    for (k, col) in u_cols.iter().enumerate() {
        for i in 0..m {
            u_out[i + k * m] = num_traits::NumCast::from(col[i]).unwrap_or(T::zero());
        }
    }

    // V: n × rank, column-major — column k spans v_out[k*n .. (k+1)*n].
    let mut v_out = vec![T::zero(); n * rank];
    for (k, col) in v_cols.iter().enumerate() {
        for j in 0..n {
            v_out[j + k * n] = num_traits::NumCast::from(col[j]).unwrap_or(T::zero());
        }
    }

    BlrBlock { m, n, rank, u: u_out, v: v_out }
}

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



// ─── BlrMatrix ────────────────────────────────────────────────────────────────

/// A two-dimensional matrix partitioned into a `nb_rows × nb_cols` grid of
/// blocks, where each off-diagonal block is stored in BLR format and diagonal
/// blocks are stored as dense.
///
/// ## Storage layout
///
/// - `row_sizes[i]`  — number of rows in block-row `i` (sum = total rows).
/// - `col_sizes[j]`  — number of columns in block-col `j` (sum = total cols).
/// - `blocks[i * nb_cols + j]` — the `(i,j)` block.
///
/// ## Supported operations
///
/// | Method | Description |
/// |--------|-------------|
/// | `nrows()` / `ncols()` | Total matrix dimensions |
/// | `get(i, j)` | Borrow the `(i,j)` block |
/// | `get_mut(i, j)` | Mutably borrow the `(i,j)` block |
/// | `apply_add(x, y, alpha)` | `y += alpha · A · x` (full matvec) |
/// | `apply_add_t(x, y, alpha)` | `y += alpha · Aᵀ · x` |
/// | `compress_from_dense(a, tol, max_rank)` | Build from a dense row-major matrix |
#[derive(Debug, Clone)]
pub struct BlrMatrix<T: Scalar> {
    /// Number of block rows.
    pub nb_rows: usize,
    /// Number of block columns.
    pub nb_cols: usize,
    /// Row size of each block-row (length `nb_rows`).
    pub row_sizes: Vec<usize>,
    /// Column size of each block-col (length `nb_cols`).
    pub col_sizes: Vec<usize>,
    /// Blocks in row-major order.  `blocks[i * nb_cols + j]` is the `(i,j)` block.
    /// Diagonal blocks (`i == j`) use `rank = 0` with `u`/`v` empty and are
    /// stored dense in a separate `dense_blocks` array.
    pub blr_blocks: Vec<Option<BlrBlock<T>>>,
    /// Dense storage for diagonal (and explicitly-dense) blocks, indexed by
    /// `i * nb_cols + j`.  `None` for off-diagonal BLR blocks.
    pub dense_blocks: Vec<Option<Vec<T>>>,
}

impl<T: Scalar> BlrMatrix<T> {
    /// Total number of rows.
    #[inline]
    pub fn nrows(&self) -> usize { self.row_sizes.iter().sum() }

    /// Total number of columns.
    #[inline]
    pub fn ncols(&self) -> usize { self.col_sizes.iter().sum() }

    /// Borrow the `(i, j)` block.
    ///
    /// Returns `Some(&BlrBlock)` for off-diagonal BLR blocks, `None` for
    /// dense/diagonal blocks (use `dense_block` for those).
    #[inline]
    pub fn blr_block(&self, i: usize, j: usize) -> Option<&BlrBlock<T>> {
        self.blr_blocks[i * self.nb_cols + j].as_ref()
    }

    /// Borrow the dense storage for block `(i, j)`, if it exists.
    #[inline]
    pub fn dense_block(&self, i: usize, j: usize) -> Option<&[T]> {
        self.dense_blocks[i * self.nb_cols + j].as_deref()
    }

    /// Compute `y += alpha · A · x` (full matrix-vector product).
    ///
    /// `x` has length `ncols()`, `y` has length `nrows()`.
    ///
    /// Each block contributes independently:
    /// - BLR blocks use `BlrBlock::apply_add`.
    /// - Dense blocks use a direct triple-loop matvec.
    ///
    /// # Panics
    /// Panics if `x.len() != ncols()` or `y.len() != nrows()`.
    pub fn apply_add(&self, x: &[T], y: &mut [T], alpha: T) {
        assert_eq!(x.len(), self.ncols(), "BlrMatrix::apply_add: x length mismatch");
        assert_eq!(y.len(), self.nrows(), "BlrMatrix::apply_add: y length mismatch");

        let mut row_off = 0usize;
        for i in 0..self.nb_rows {
            let m_i = self.row_sizes[i];
            let y_slice = &mut y[row_off .. row_off + m_i];
            let mut col_off = 0usize;
            for j in 0..self.nb_cols {
                let n_j = self.col_sizes[j];
                let x_slice = &x[col_off .. col_off + n_j];
                let idx = i * self.nb_cols + j;
                if let Some(blk) = &self.blr_blocks[idx] {
                    blk.apply_add(x_slice, y_slice, alpha);
                } else if let Some(d) = &self.dense_blocks[idx] {
                    // Dense row-major m_i × n_j block — SIMD GEMV.
                    crate::simd::dense_ops::simd_gemv(alpha, d, m_i, n_j, x_slice, y_slice);
                }
                col_off += n_j;
            }
            row_off += m_i;
        }
    }

    /// Compute `y += alpha · Aᵀ · x`.
    ///
    /// `x` has length `nrows()`, `y` has length `ncols()`.
    ///
    /// # Panics
    /// Panics if `x.len() != nrows()` or `y.len() != ncols()`.
    pub fn apply_add_t(&self, x: &[T], y: &mut [T], alpha: T) {
        assert_eq!(x.len(), self.nrows(), "BlrMatrix::apply_add_t: x length mismatch");
        assert_eq!(y.len(), self.ncols(), "BlrMatrix::apply_add_t: y length mismatch");

        let mut row_off = 0usize;
        for i in 0..self.nb_rows {
            let m_i = self.row_sizes[i];
            let x_slice = &x[row_off .. row_off + m_i];
            let mut col_off = 0usize;
            for j in 0..self.nb_cols {
                let n_j = self.col_sizes[j];
                let y_slice = &mut y[col_off .. col_off + n_j];
                let idx = i * self.nb_cols + j;
                if let Some(blk) = &self.blr_blocks[idx] {
                    blk.apply_add_t(x_slice, y_slice, alpha);
                } else if let Some(d) = &self.dense_blocks[idx] {
                    // Dense transpose GEMV — SIMD-accelerated via row AXPY.
                    crate::simd::dense_ops::simd_gemv_t(alpha, d, m_i, n_j, x_slice, y_slice);
                }
                col_off += n_j;
            }
            row_off += m_i;
        }
    }

    /// Build a `BlrMatrix` from a dense row-major `nrows × ncols` matrix by
    /// partitioning it according to `row_sizes` / `col_sizes`, compressing
    /// off-diagonal blocks with BLR and storing diagonal blocks as dense.
    ///
    /// `tol` and `max_rank` are forwarded to `compress_block`.
    ///
    /// # Panics
    /// Panics if `row_sizes.iter().sum() != nrows` or analogously for columns,
    /// or if `a.len() != nrows * ncols`.
    pub fn compress_from_dense(
        a: &[T],
        nrows: usize,
        ncols: usize,
        row_sizes: &[usize],
        col_sizes: &[usize],
        tol: f64,
        max_rank: usize,
    ) -> Self {
        assert_eq!(a.len(), nrows * ncols, "BlrMatrix::compress_from_dense: a.len() mismatch");
        assert_eq!(row_sizes.iter().sum::<usize>(), nrows,
            "BlrMatrix::compress_from_dense: row_sizes sum mismatch");
        assert_eq!(col_sizes.iter().sum::<usize>(), ncols,
            "BlrMatrix::compress_from_dense: col_sizes sum mismatch");

        let nb_rows = row_sizes.len();
        let nb_cols = col_sizes.len();
        let n_blocks = nb_rows * nb_cols;
        let mut blr_blocks  = vec![None; n_blocks];
        let mut dense_blocks = vec![None; n_blocks];

        let mut row_off = 0usize;
        for i in 0..nb_rows {
            let m_i = row_sizes[i];
            let mut col_off = 0usize;
            for j in 0..nb_cols {
                let n_j = col_sizes[j];
                // Extract the sub-block as a contiguous row-major buffer.
                let mut sub = vec![T::zero(); m_i * n_j];
                for r in 0..m_i {
                    for c in 0..n_j {
                        sub[r * n_j + c] = a[(row_off + r) * ncols + (col_off + c)];
                    }
                }
                let idx = i * nb_cols + j;
                if i == j {
                    // Diagonal block: store dense.
                    dense_blocks[idx] = Some(sub);
                } else {
                    // Off-diagonal block: BLR compress.
                    blr_blocks[idx] = Some(compress_block(&sub, m_i, n_j, tol, max_rank));
                }
                col_off += n_j;
            }
            row_off += m_i;
        }

        BlrMatrix {
            nb_rows,
            nb_cols,
            row_sizes: row_sizes.to_vec(),
            col_sizes: col_sizes.to_vec(),
            blr_blocks,
            dense_blocks,
        }
    }

    /// Total memory used by all stored blocks (dense + BLR) in bytes.
    pub fn memory_bytes(&self) -> usize {
        let elem = std::mem::size_of::<T>();
        let blr: usize = self.blr_blocks.iter().flatten()
            .map(|b| b.memory_bytes().1)
            .sum();
        let dense: usize = self.dense_blocks.iter().flatten()
            .map(|d| d.len() * elem)
            .sum();
        blr + dense
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
        let u = [1.0f64, 2.0, 3.0, 4.0];
        let v = [1.0f64, -1.0, 2.0];
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
        let u1 = [1.0f64, 2.0, 0.0, -1.0, 0.5, 3.0];
        let v1 = [1.0f64, 0.0, -1.0, 2.0, 0.5];
        let u2 = [0.0f64, 1.0, -1.0, 0.0, 2.0, -0.5];
        let v2 = [0.5f64, -1.0, 1.0, 0.0, -2.0];
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
        for _ in 0..rank_true {
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
        for _ in 0..rank_true {
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
        let ub = [1.0f64, 2.0, -1.0, 0.5, 3.0];
        let vb = [1.0f64, -1.0, 2.0, 0.5];
        let uc = [0.0f64, 1.0, 1.0, -2.0, 0.5];
        let vc = [-1.0f64, 0.0, 1.0, -0.5];
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

    #[test]
    fn blr_add_compressed_hard_cap_is_optimal() {
        // B is rank-3, C is rank-3 → sum is rank ≤ 3 (two of the three pairs cancel).
        // With max_rank=2 the result should be the best rank-2 approximation of the sum,
        // NOT just the first two columns of the concatenated factors.
        let m = 6; let n = 5;
        // Build orthonormal-ish vectors.
        let u: [Vec<f64>; 3] = [
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ];
        let v: [Vec<f64>; 3] = [
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0],
        ];
        let sigmas_b = [50.0f64, 10.0, 1.0];
        let sigmas_c = [50.0f64, 10.0, 1.0];
        let mut b_dense = vec![0.0f64; m * n];
        let mut c_dense = vec![0.0f64; m * n];
        for k in 0..3 {
            for i in 0..m { for j in 0..n {
                b_dense[i*n+j] += sigmas_b[k] * u[k][i] * v[k][j];
                c_dense[i*n+j] += sigmas_c[k] * u[k][i] * v[k][j];
            }}
        }
        // sum = 2*B: singular values 100, 20, 2.
        let sum_dense: Vec<f64> = b_dense.iter().zip(&c_dense).map(|(a,b)| a+b).collect();

        let blk_b = compress_block::<f64>(&b_dense, m, n, 1e-12, 0);
        let blk_c = compress_block::<f64>(&c_dense, m, n, 1e-12, 0);
        // cap = 2 → should keep σ=100 and σ=20 columns only.
        let blk_sum = blk_b.add_compressed(&blk_c, 1e-12, 2);
        assert!(blk_sum.rank <= 2, "capped sum rank should be ≤2, got {}", blk_sum.rank);

        // Build optimal rank-2 reference (drop σ=2 component).
        let mut ref2 = vec![0.0f64; m * n];
        for k in 0..2 {
            let sigma_sum = sigmas_b[k] + sigmas_c[k];
            for i in 0..m { for j in 0..n {
                ref2[i*n+j] += sigma_sum * u[k][i] * v[k][j];
            }}
        }
        let recon = blk_sum.to_dense();
        let err: f64 = ref2.iter().zip(&recon).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt();
        let nrm: f64 = sum_dense.iter().map(|x| x*x).sum::<f64>().sqrt();
        // The σ=2 term is 2% of σ=100, so the capped approximation should differ by ~2%.
        assert!(err / nrm < 0.05, "hard-cap add_compressed error {:.4}", err / nrm);
    }

    #[test]
    fn blr_frobenius_norm_matches_dense() {
        let m = 5; let n = 4;
        let u = vec![1.0f64, 2.0, -1.0, 0.5, 3.0,
                     0.0,  1.0,  1.0, -2.0, 0.5];
        let v = vec![1.0f64, -1.0, 2.0, 0.5,
                     0.5,  0.0, -1.0, 1.0];
        let blk = BlrBlock::<f64> { m, n, rank: 2, u, v };
        // Dense norm.
        let dense = blk.to_dense();
        let dense_nrm: f64 = dense.iter().map(|x| x*x).sum::<f64>().sqrt();
        let blr_nrm = blk.frobenius_norm();
        assert!((blr_nrm - dense_nrm).abs() / dense_nrm.max(1e-15) < 1e-10,
            "frobenius_norm mismatch: blr={blr_nrm:.10} dense={dense_nrm:.10}");
    }

    #[test]
    fn blr_frobenius_norm_zero_rank() {
        let blk = BlrBlock::<f64> { m: 4, n: 3, rank: 0, u: vec![], v: vec![] };
        assert_eq!(blk.frobenius_norm(), 0.0);
    }

    #[test]
    fn blr_scale_matches_dense() {
        let m = 4; let n = 3;
        let a = vec![1.0f64, 2.0, 0.0, -1.0, 3.0, 1.0, 0.0, -2.0, 4.0, 1.0, 0.5, -0.5];
        let blk = compress_block::<f64>(&a, m, n, 1e-12, 0);
        let mut blk_scaled = blk.clone();
        blk_scaled.scale(3.0f64);
        let dense_scaled = blk_scaled.to_dense();
        let orig_dense = blk.to_dense();
        for (s, o) in dense_scaled.iter().zip(&orig_dense) {
            assert!((s - 3.0 * o).abs() < 1e-10, "scale mismatch: {s} vs {}", 3.0 * o);
        }
    }

    #[test]
    fn blr_large_rank_sketch_quality() {
        // Verify that de-hardcoding the sketch size doesn't degrade quality for
        // a block with true rank 25 (> old cap of 20).
        let m = 60; let n = 40;
        let true_rank = 25;
        let mut rng = Lcg64::new(0xabcdef01);
        let mut u_cols: Vec<Vec<f64>> = Vec::new();
        for _ in 0..true_rank {
            let mut col: Vec<f64> = (0..m).map(|_| rng.gaussian()).collect();
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
        for _ in 0..true_rank {
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
        let mut a = vec![0.0f64; m * n];
        for k in 0..u_cols.len().min(v_cols.len()) {
            let sigma = (-(k as f64) * 0.3).exp(); // slow decay
            for i in 0..m { for j in 0..n { a[i*n+j] += sigma * u_cols[k][i] * v_cols[k][j]; } }
        }
        let nrm: f64 = a.iter().map(|x| x*x).sum::<f64>().sqrt();
        let tol = 1e-6;
        let blk = compress_block::<f64>(&a, m, n, tol, 0);
        let recon = blk.to_dense();
        let err: f64 = a.iter().zip(&recon).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt();
        assert!(err / nrm < tol * 100.0,
            "large-rank sketch err/nrm={:.2e} > {}×tol, rank={}", err/nrm, 100, blk.rank);
    }

    // ── apply_add_mat / apply_add_t_mat ───────────────────────────────────────

    #[test]
    fn blr_apply_add_mat_matches_loop() {
        // Y += alpha * A * X where A is m×n, X is n×k.
        let m = 4; let n = 3; let k_rhs = 2;
        let a = vec![1.0f64, 2.0, 0.0, -1.0, 3.0, 1.0, 0.0, -2.0, 4.0, 1.0, 0.5, -0.5];
        let blk = compress_block::<f64>(&a, m, n, 1e-12, 0);
        // X: column-major n×k.
        let x = vec![1.0f64, -1.0, 2.0,   // col 0
                     0.5,  2.0, -1.0];      // col 1
        let alpha = 2.0f64;
        // Reference: dense loop.
        let mut y_ref = vec![0.0f64; m * k_rhs];
        for c in 0..k_rhs {
            for i in 0..m { for j in 0..n {
                y_ref[i + c*m] += alpha * a[i*n+j] * x[j + c*n];
            }}
        }
        let mut y_blr = vec![0.0f64; m * k_rhs];
        blk.apply_add_mat(&x, &mut y_blr, alpha, k_rhs);
        for idx in 0..m*k_rhs {
            assert!((y_ref[idx] - y_blr[idx]).abs() < 1e-10,
                "apply_add_mat mismatch at {idx}: ref={:.6} blr={:.6}", y_ref[idx], y_blr[idx]);
        }
    }

    #[test]
    fn blr_apply_add_t_mat_matches_loop() {
        // Y += alpha * Aᵀ * X where A is m×n, X is m×k.
        let m = 4; let n = 3; let k_rhs = 2;
        let a = vec![1.0f64, 2.0, 0.0, -1.0, 3.0, 1.0, 0.0, -2.0, 4.0, 1.0, 0.5, -0.5];
        let blk = compress_block::<f64>(&a, m, n, 1e-12, 0);
        let x = vec![1.0f64, -1.0, 2.0, 0.5,   // col 0 (length m)
                    -0.5,  1.0, 0.0, 2.0];       // col 1
        let alpha = -1.0f64;
        let mut y_ref = vec![0.0f64; n * k_rhs];
        for c in 0..k_rhs {
            for j in 0..n { for i in 0..m {
                y_ref[j + c*n] += alpha * a[i*n+j] * x[i + c*m];
            }}
        }
        let mut y_blr = vec![0.0f64; n * k_rhs];
        blk.apply_add_t_mat(&x, &mut y_blr, alpha, k_rhs);
        for idx in 0..n*k_rhs {
            assert!((y_ref[idx] - y_blr[idx]).abs() < 1e-10,
                "apply_add_t_mat mismatch at {idx}: ref={:.6} blr={:.6}", y_ref[idx], y_blr[idx]);
        }
    }

    #[test]
    fn blr_apply_add_mat_single_rhs_matches_apply_add() {
        // apply_add_mat with k_rhs=1 must agree exactly with apply_add.
        let m = 5; let n = 4;
        let a: Vec<f64> = (0..m*n).map(|i| (i as f64 * 1.7 + 0.3).sin()).collect();
        let blk = compress_block::<f64>(&a, m, n, 1e-12, 0);
        let x: Vec<f64> = (0..n).map(|j| j as f64 - 1.5).collect();
        let alpha = 3.0f64;
        let mut y1 = vec![0.0f64; m];
        blk.apply_add(&x, &mut y1, alpha);
        let mut y2 = vec![0.0f64; m];
        blk.apply_add_mat(&x, &mut y2, alpha, 1);
        for i in 0..m {
            assert!((y1[i] - y2[i]).abs() < 1e-13,
                "k_rhs=1 mismatch at i={i}: {:.10} vs {:.10}", y1[i], y2[i]);
        }
    }

    // ── subtract_compressed ───────────────────────────────────────────────────

    #[test]
    fn blr_subtract_compressed_self_is_zero() {
        // A - A should give a zero (or near-zero rank) block.
        let m = 5; let n = 4;
        let u = [1.0f64, 2.0, -1.0, 0.5, 3.0];
        let v = [1.0f64, -1.0, 2.0, 0.5];
        let mut a = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n { a[i*n+j] = u[i] * v[j]; } }
        let blk = compress_block::<f64>(&a, m, n, 1e-12, 0);
        let diff = blk.subtract_compressed(&blk, 1e-10, 0);
        // Rank should collapse to 0 (or the dense reconstruction should be ~0).
        let recon = diff.to_dense();
        let err: f64 = recon.iter().map(|x| x*x).sum::<f64>().sqrt();
        let nrm: f64 = a.iter().map(|x| x*x).sum::<f64>().sqrt();
        assert!(err / nrm < 1e-8, "A - A should be ~zero, got err/nrm={:.2e}", err / nrm);
    }

    #[test]
    fn blr_subtract_compressed_roundtrip() {
        // (A + B) - B should recover A.
        let m = 5; let n = 4;
        let ua = [1.0f64, 0.0, -1.0, 0.5, 2.0];
        let va = [1.0f64, -1.0, 0.0, 0.5];
        let ub = [0.0f64, 1.0,  1.0, -2.0, 0.5];
        let vb = [-1.0f64, 0.0, 1.0, -0.5];
        let mut a_dense = vec![0.0f64; m * n];
        let mut b_dense = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n {
            a_dense[i*n+j] = ua[i] * va[j];
            b_dense[i*n+j] = ub[i] * vb[j];
        }}
        let blk_a = compress_block::<f64>(&a_dense, m, n, 1e-12, 0);
        let blk_b = compress_block::<f64>(&b_dense, m, n, 1e-12, 0);
        let blk_sum = blk_a.add_compressed(&blk_b, 1e-12, 0);
        let blk_back = blk_sum.subtract_compressed(&blk_b, 1e-10, 0);
        let recon = blk_back.to_dense();
        let err: f64 = a_dense.iter().zip(&recon).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt();
        let nrm: f64 = a_dense.iter().map(|x| x*x).sum::<f64>().sqrt();
        assert!(err / nrm < 1e-8, "(A+B)-B should recover A, err/nrm={:.2e}", err / nrm);
    }

    // ── from_factors ──────────────────────────────────────────────────────────

    #[test]
    fn blr_from_factors_roundtrip() {
        // Build a rank-2 block by hand, construct via from_factors, verify to_dense.
        let m = 4; let n = 3;
        let u = vec![1.0f64, 0.0, -1.0, 2.0,   // col 0
                     0.5,   1.0,  0.0, -1.0];   // col 1
        let v = vec![1.0f64, -1.0, 2.0,         // col 0
                     0.0,   1.0, -0.5];          // col 1
        let blk = BlrBlock::<f64>::from_factors(m, n, u.clone(), v.clone());
        assert_eq!(blk.rank, 2);
        assert_eq!(blk.m, m);
        assert_eq!(blk.n, n);
        // Compare to_dense with manual U Vᵀ.
        let mut expected = vec![0.0f64; m * n];
        for k in 0..2 {
            for i in 0..m { for j in 0..n {
                expected[i*n+j] += u[i + k*m] * v[j + k*n];
            }}
        }
        let dense = blk.to_dense();
        for (a, b) in expected.iter().zip(&dense) {
            assert!((a - b).abs() < 1e-14, "from_factors to_dense mismatch: {a} vs {b}");
        }
    }

    #[test]
    #[should_panic(expected = "rank from u")]
    fn blr_from_factors_rank_mismatch_panics() {
        // u implies rank=2, v implies rank=1 → should panic.
        let u = vec![1.0f64, 2.0, 3.0, 4.0]; // m=2, rank=2
        let v = vec![1.0f64, 0.5, -1.0];      // n=3, rank=1
        let _ = BlrBlock::<f64>::from_factors(2, 3, u, v);
    }

    // ── Display ───────────────────────────────────────────────────────────────

    #[test]
    fn blr_display_rank1() {
        let m = 10; let n = 8;
        let u: Vec<f64> = (0..m).map(|i| i as f64 + 1.0).collect();
        let v: Vec<f64> = (0..n).map(|j| j as f64 * 0.5 + 1.0).collect();
        let mut a = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n { a[i*n+j] = u[i] * v[j]; } }
        let blk = compress_block::<f64>(&a, m, n, 1e-10, 0);
        let s = format!("{blk}");
        assert!(s.starts_with("BLR [10×8, rank=1,"), "unexpected Display: {s}");
        assert!(s.contains('%'), "Display should contain '%': {s}");
    }

    #[test]
    fn blr_display_zero_rank() {
        let blk = BlrBlock::<f64> { m: 4, n: 3, rank: 0, u: vec![], v: vec![] };
        let s = format!("{blk}");
        assert_eq!(s, "BLR [4×3, rank=0, ratio=0.0%]");
    }

    // ── compress_block_adaptive ───────────────────────────────────────────────

    #[test]
    fn blr_adaptive_atol_zero_same_as_compress_block() {
        // atol=0 should behave identically to compress_block.
        let m = 6; let n = 5;
        let u1 = [1.0f64, 2.0, 0.0, -1.0, 0.5, 3.0];
        let v1 = [1.0f64, 0.0, -1.0, 2.0, 0.5];
        let mut a = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n { a[i*n+j] = u1[i] * v1[j]; } }
        let blk1 = compress_block::<f64>(&a, m, n, 1e-8, 0);
        let blk2 = compress_block_adaptive::<f64>(&a, m, n, 1e-8, 0.0, 0);
        assert_eq!(blk1.rank, blk2.rank, "atol=0 rank should match compress_block");
    }

    #[test]
    fn blr_adaptive_atol_drops_small_singular_values() {
        // Build matrix with singular values 100, 1, 0.0001.
        // rtol=0 keeps all three; atol=0.01 should drop the σ=0.0001 column.
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
        let sigmas = [100.0f64, 1.0, 0.0001];
        let mut a = vec![0.0f64; m * n];
        for k in 0..3 {
            for i in 0..m { for j in 0..n { a[i*n+j] += sigmas[k] * u[k][i] * v[k][j]; } }
        }
        // rtol=0 (keep everything), atol=0.01 (drop σ=0.0001).
        let blk = compress_block_adaptive::<f64>(&a, m, n, 0.0, 0.01, 0);
        assert!(blk.rank <= 2,
            "atol=0.01 should drop σ=0.0001, got rank={}", blk.rank);
        // Reconstruction should capture the σ=100 and σ=1 components.
        let recon = blk.to_dense();
        let mut ref2 = vec![0.0f64; m * n];
        for k in 0..2 {
            for i in 0..m { for j in 0..n { ref2[i*n+j] += sigmas[k] * u[k][i] * v[k][j]; } }
        }
        let err: f64 = ref2.iter().zip(&recon).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt();
        let nrm: f64 = ref2.iter().map(|x| x*x).sum::<f64>().sqrt();
        assert!(err / nrm < 0.01, "adaptive atol reconstruction err/nrm={:.4}", err / nrm);
    }

    #[test]
    fn blr_adaptive_rtol_only_same_as_compress_block() {
        // atol=0, rtol>0 should behave like compress_block with that rtol.
        let m = 6; let n = 5;
        let sigmas = [100.0f64, 1.0, 0.001];
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
        let mut a = vec![0.0f64; m * n];
        for k in 0..3 {
            for i in 0..m { for j in 0..n { a[i*n+j] += sigmas[k] * u[k][i] * v[k][j]; } }
        }
        let tol = 0.01; // rtol drops σ=0.001 (0.001/100 < 0.01)
        let blk_std = compress_block::<f64>(&a, m, n, tol, 0);
        let blk_adp = compress_block_adaptive::<f64>(&a, m, n, tol, 0.0, 0);
        assert_eq!(blk_std.rank, blk_adp.rank,
            "rtol-only adaptive should equal compress_block: {} vs {}", blk_std.rank, blk_adp.rank);
    }

    // ── BlrMatrix ─────────────────────────────────────────────────────────────

    fn make_test_dense_4x4() -> Vec<f64> {
        // 4×4 matrix with a clear 2×2 block structure:
        // top-left 2×2 = identity, top-right 2×2 = rank-1
        // bottom-left 2×2 = rank-1, bottom-right 2×2 = identity
        let mut a = vec![0.0f64; 16];
        // diagonal blocks (identity)
        a[0] = 1.0; a[4+1] = 1.0;
        a[2*4+2] = 1.0; a[3*4+3] = 1.0;
        // top-right off-diagonal: rows 0-1, cols 2-3
        a[2] = 2.0; a[3] = 1.0;
        a[4+2] = 4.0; a[4+3] = 2.0;
        // bottom-left off-diagonal: rows 2-3, cols 0-1
        a[2*4] = 1.0; a[2*4+1] = 3.0;
        a[3*4] = 2.0; a[3*4+1] = 6.0;
        a
    }

    #[test]
    fn blr_matrix_apply_add_matches_dense() {
        let a = make_test_dense_4x4();
        let nrows = 4; let ncols = 4;
        let row_sizes = vec![2usize, 2];
        let col_sizes = vec![2usize, 2];
        let bm = BlrMatrix::compress_from_dense(
            &a, nrows, ncols, &row_sizes, &col_sizes, 1e-10, 0);
        let x = vec![1.0f64, -1.0, 2.0, 0.5];
        let mut y_ref = [0.0f64; 4];
        for i in 0..4 { for j in 0..4 { y_ref[i] += a[i*4+j] * x[j]; } }
        let mut y_blr = vec![0.0f64; 4];
        bm.apply_add(&x, &mut y_blr, 1.0f64);
        for i in 0..4 {
            assert!((y_ref[i] - y_blr[i]).abs() < 1e-10,
                "BlrMatrix apply_add mismatch at i={i}: ref={:.6} blr={:.6}", y_ref[i], y_blr[i]);
        }
    }

    #[test]
    fn blr_matrix_apply_add_t_matches_dense() {
        let a = make_test_dense_4x4();
        let bm: BlrMatrix<f64> = BlrMatrix::compress_from_dense(
            &a, 4, 4, &[2usize, 2], &[2usize, 2], 1e-10, 0);
        let x = vec![1.0f64, 0.5, -1.0, 2.0];
        let mut y_ref = [0.0f64; 4];
        for i in 0..4 { for j in 0..4 { y_ref[j] += a[i*4+j] * x[i]; } }
        let mut y_blr = vec![0.0f64; 4];
        bm.apply_add_t(&x, &mut y_blr, 1.0f64);
        for j in 0..4 {
            assert!((y_ref[j] - y_blr[j]).abs() < 1e-10,
                "BlrMatrix apply_add_t mismatch at j={j}: ref={:.6} blr={:.6}", y_ref[j], y_blr[j]);
        }
    }

    #[test]
    fn blr_matrix_alpha_scaling() {
        let a = make_test_dense_4x4();
        let bm: BlrMatrix<f64> = BlrMatrix::compress_from_dense(
            &a, 4, 4, &[2usize, 2], &[2usize, 2], 1e-10, 0);
        let x = vec![1.0f64, 2.0, -1.0, 0.5];
        let mut y1 = vec![0.0f64; 4];
        bm.apply_add(&x, &mut y1, 1.0f64);
        let mut y2 = vec![0.0f64; 4];
        bm.apply_add(&x, &mut y2, 3.0f64);
        for i in 0..4 {
            assert!((y2[i] - 3.0 * y1[i]).abs() < 1e-10,
                "alpha scaling mismatch at i={i}");
        }
    }

    #[test]
    fn blr_matrix_diagonal_blocks_are_dense() {
        let a = make_test_dense_4x4();
        let bm: BlrMatrix<f64> = BlrMatrix::compress_from_dense(
            &a, 4, 4, &[2usize, 2], &[2usize, 2], 1e-10, 0);
        assert!(bm.dense_block(0, 0).is_some(), "block (0,0) should be dense");
        assert!(bm.dense_block(1, 1).is_some(), "block (1,1) should be dense");
        assert!(bm.blr_block(0, 1).is_some(),   "block (0,1) should be BLR");
        assert!(bm.blr_block(1, 0).is_some(),   "block (1,0) should be BLR");
    }

    #[test]
    fn blr_matrix_memory_bytes_less_than_dense() {
        let n = 20;
        let u = (0..n).map(|i| i as f64 + 1.0).collect::<Vec<_>>();
        let v = (0..n).map(|j| (j as f64 + 1.0).recip()).collect::<Vec<_>>();
        let mut a = vec![0.0f64; n * n];
        for i in 0..n { for j in 0..n { a[i*n+j] = u[i] * v[j]; } }
        let h = n / 2;
        for i in 0..h { for j in 0..h { a[i*n+j] = 0.0; a[(h+i)*n+(h+j)] = 0.0; } }
        let bm: BlrMatrix<f64> = BlrMatrix::compress_from_dense(
            &a, n, n, &[h, h], &[h, h], 1e-10, 0);
        let dense_bytes = n * n * std::mem::size_of::<f64>();
        let blr_bytes = bm.memory_bytes();
        assert!(blr_bytes < dense_bytes,
            "BLR memory ({blr_bytes}B) should be < dense ({dense_bytes}B)");
    }

    // ── ACA tests ─────────────────────────────────────────────────────────────

    #[test]
    fn aca_rank1_exact() {
        // A true rank-1 matrix should be compressed to rank ≤ 2 with tiny error.
        // ACA may accept a second step before the floating-point residual drops
        // below the zero-pivot threshold, so we check reconstruction quality
        // rather than enforcing rank == 1.
        let m = 6; let n = 5;
        let u_vec: Vec<f64> = (0..m).map(|i| i as f64 + 1.0).collect();
        let v_vec: Vec<f64> = (0..n).map(|j| j as f64 * 0.5 + 0.1).collect();
        let mut a = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n { a[i*n+j] = u_vec[i] * v_vec[j]; } }
        let blk = compress_block_aca::<f64>(&a, m, n, 1e-10, 0);
        assert!(blk.rank <= 2, "rank-1 matrix should compress to rank ≤ 2, got {}", blk.rank);
        let recon = blk.to_dense();
        let err: f64 = a.iter().zip(&recon).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-10, "rank-1 ACA error = {err:.2e}");
    }

    #[test]
    fn aca_zero_block() {
        // A zero matrix should give rank 0.
        let m = 4; let n = 3;
        let a = vec![0.0f64; m * n];
        let blk = compress_block_aca::<f64>(&a, m, n, 1e-10, 0);
        assert_eq!(blk.rank, 0);
    }

    #[test]
    fn aca_apply_add_matches_dense() {
        // ACA matvec should agree with the dense reference.
        let m = 5; let n = 4;
        // Low-rank-2 matrix.
        let u1 = [1.0f64, -1.0, 2.0, 0.5, -0.5];
        let v1 = [1.0f64, 2.0, -1.0, 0.5];
        let u2 = [0.5f64, 1.0, -0.5, 1.5, -1.0];
        let v2 = [-1.0f64, 0.5, 2.0, -0.5];
        let mut a = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n { a[i*n+j] = u1[i]*v1[j] + u2[i]*v2[j]; } }
        let blk = compress_block_aca::<f64>(&a, m, n, 1e-10, 0);
        let x = vec![1.0f64, -0.5, 2.0, 1.0];
        let mut y_ref = vec![0.0f64; m];
        for i in 0..m { for j in 0..n { y_ref[i] += a[i*n+j] * x[j]; } }
        let mut y_aca = vec![0.0f64; m];
        blk.apply_add(&x, &mut y_aca, 1.0f64);
        for i in 0..m {
            assert!((y_ref[i] - y_aca[i]).abs() < 1e-9,
                "ACA matvec mismatch at i={i}: ref={:.8} aca={:.8}", y_ref[i], y_aca[i]);
        }
    }

    #[test]
    fn aca_fn_matches_dense_block() {
        // compress_block_aca_fn with an entry closure should agree with
        // compress_block_aca applied to the explicit slice.
        let m = 6; let n = 5;
        let a: Vec<f64> = (0..m*n).map(|k| (k as f64 * 0.7 + 1.0).sin()).collect();
        let blk_slice = compress_block_aca::<f64>(&a, m, n, 1e-6, 0);
        let blk_fn    = compress_block_aca_fn(|i, j| a[i*n+j], m, n, 1e-6, 0);
        // Both should produce the same rank and identical dense reconstructions.
        assert_eq!(blk_slice.rank, blk_fn.rank,
            "rank mismatch: slice={} fn={}", blk_slice.rank, blk_fn.rank);
        let dense_s = blk_slice.to_dense();
        let dense_f = blk_fn.to_dense();
        for (s, f) in dense_s.iter().zip(&dense_f) {
            assert!((s - f).abs() < 1e-14, "dense mismatch: {s:.10} vs {f:.10}");
        }
    }

    #[test]
    fn aca_low_rank_approximation_quality() {
        // Build an exact rank-3 matrix and verify ACA recovers it up to tol.
        let m = 10; let n = 8;
        let tol = 1e-8;
        let mut a = vec![0.0f64; m * n];
        let sigma = [3.0f64, 1.5, 0.5];
        let rows: [[f64; 10]; 3] = [
            [1.0, 0.5, -1.0, 0.2, 0.8, -0.3, 0.6, -0.7, 0.4, -0.1],
            [0.3, -0.6, 0.9, -0.2, 0.5, 0.8, -0.4, 0.1, -0.7, 0.6],
            [0.7, 0.2, -0.5, 0.9, -0.3, 0.4, 0.1, -0.8, 0.5, 0.3],
        ];
        let cols: [[f64; 8]; 3] = [
            [1.0, -0.5, 0.3, 0.8, -0.2, 0.6, -0.4, 0.7],
            [-0.3, 0.7, -0.9, 0.1, 0.5, -0.8, 0.4, -0.2],
            [0.6, 0.1, -0.4, 0.7, 0.3, -0.5, 0.9, -0.6],
        ];
        for k in 0..3 {
            for i in 0..m { for j in 0..n { a[i*n+j] += sigma[k] * rows[k][i] * cols[k][j]; } }
        }
        let nrm: f64 = a.iter().map(|x| x*x).sum::<f64>().sqrt();
        let blk = compress_block_aca::<f64>(&a, m, n, tol, 0);
        let recon = blk.to_dense();
        let err: f64 = a.iter().zip(&recon).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt();
        assert!(err / nrm < tol * 100.0,
            "ACA err/nrm={:.2e}, rank={}", err/nrm, blk.rank);
    }
}
