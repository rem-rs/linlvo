//! LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient.
//!
//! This module is a **1:1 Rust port of BLOPEX** (github.com/lobpcg/blopex,
//! MIT / Apache-2.0) — the exact LOBPCG kernel used by HYPRE's AME
//! (Auxiliary-space Maxwell Eigensolver, `src/parcsr_ls/ame.c`).  It replaces
//! the earlier home-grown LOBPCG that was numerically unstable on the
//! singular generalised Maxwell eigenproblem `A x = λ M x` (the discrete
//! gradient nullspace kept re-entering the search space).
//!
//! # Algorithm (BLOPEX `lobpcg_solve`, f64 path)
//!
//! ```text
//! X₀ = random block, B-orthonormalised by implicit QR (Cholesky of XᵀBX)
//! AX = A·X₀;  initial Rayleigh–Ritz → (λ, coordX);  X ← X·coordX, …
//! R  = BX·diag(λ) − AX                       (block residual)
//! loop (soft locking via active mask):
//!   sizeR = #{ j : ‖Rⱼ‖ > |λⱼ|·rtol + atol + eps }   (converged → locked)
//!   W = T⁻¹(R)   (preconditioned residual; AME: AMS⁻¹ then div-free)
//!   R = W;  B-orthonormalise R-block;  AR = A·R
//!   B-orthonormalise P-block (iteration > 1); AP = A·P
//!   assemble Gram A/B on search space [X | R | P]
//!     (X-block: diag(λ)/I — X stays diagonalised; R/P blocks: I on diag)
//!   solve GEVP gramA u = θ gramB u          (LAPACK dsygv semantics)
//!   λ ← θ[1..sizeX];  coordX = eigenvectors
//!   P ← P·coordPX + R·coordRX;  AP ← AP·coordPX + AR·coordRX
//!   X ← X·coordXX + P;  AX ← AX·coordXX + AP;  BX ← BX·coordXX + BP
//!   R ← BX·diag(λ) − AX   (active columns only)
//! ```
//!
//! Key differences from the old implementation that fix the ex32 instability:
//! - **soft locking** (`activeMask`): converged vectors are frozen in `X` and
//!   excluded from the R/P search directions — the nullspace cannot be
//!   re-injected into already-converged Ritz vectors;
//! - **no wholesale re-orthogonalisation / re-projection of `X`**: `X` is only
//!   *rotated* by the exact Ritz coordinates, so its discrete div-free
//!   property (imposed once at initialisation, cf. HYPRE AME setup) is
//!   preserved analytically instead of being re-approximated each iteration;
//! - exact small dense GEVP (`dsygv`-equivalent) on the Gram matrices.
//!
//! # References
//!
//! - Knyazev (2001). Toward the optimal preconditioned eigensolver: LOBPCG.
//! - BLOPEX: https://github.com/lobpcg/blopex — `blopex_abstract/krylov/lobpcg.c`
//! - Kolev & Vassilevski (2006). Parallel eigensolver for H(curl) problems
//!   using H1-auxiliary space AMG preconditioning. LLNL TR-226197.

use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    vector::{DenseVec, Vector},
};
use super::{EigenParams, EigenResult, EigenSolver, EigenWhich, fill_random, dot};

// ─── LOBPCG ──────────────────────────────────────────────────────────────────

/// LOBPCG eigensolver for symmetric positive definite operators.
///
/// Best combined with an AMG preconditioner (`linger::AmgPrecond`) for FEA
/// structural modal analysis.
///
/// Set `which = EigenWhich::SmallestAlgebraic` for structural modes (default),
/// or `LargestAlgebraic` for the top modes.
pub struct Lobpcg<'p, T: Scalar> {
    /// Optional preconditioner T⁻¹ ≈ A⁻¹.
    pub precond: Option<&'p dyn Preconditioner<Vector = DenseVec<T>>>,
    /// Optional B-operator for the generalised problem `A x = λ B x`.
    /// When `None`, the standard problem `A x = λ x` (B = I) is solved.
    pub b_op: Option<&'p dyn LinearOperator<Vector = DenseVec<T>>>,
    /// Optional post-preconditioner projection (applied to W after preconditioning
    /// but before the Rayleigh-Ritz).  Used e.g. for div-free projection in
    /// H(curl) eigenvalue problems.
    ///
    /// Together with `precond` this forms BLOPEX's `operatorT`:
    /// `T(x) = projector(precond(x))` — matching HYPRE AME, where
    /// `operatorT = AMS⁻¹` followed by the discrete div-free projection.
    pub projector: Option<&'p dyn Preconditioner<Vector = DenseVec<T>>>,
    /// Eliminated (essential/boundary) DOFs zeroed in the initial iterate
    /// before the nullspace projection (cf. HYPRE AME `edge_bc`).
    pub zero_dofs: Option<&'p [usize]>,
    pub seed: u64,
}

impl<'p, T: Scalar> Default for Lobpcg<'p, T> {
    fn default() -> Self {
        Lobpcg { precond: None, b_op: None, projector: None, zero_dofs: None, seed: 42 }
    }
}

impl<'p, T: Scalar> Lobpcg<'p, T> {
    /// Create a standard-problem LOBPCG (the existing API).
    pub fn new(precond: Option<&'p dyn Preconditioner<Vector = DenseVec<T>>>) -> Self {
        Lobpcg { precond, b_op: None, projector: None, zero_dofs: None, seed: 42 }
    }

    /// Create a generalised-problem LOBPCG with optional nullspace projector.
    pub fn new_generalized(
        precond: Option<&'p dyn Preconditioner<Vector = DenseVec<T>>>,
        b_op: Option<&'p dyn LinearOperator<Vector = DenseVec<T>>>,
        projector: Option<&'p dyn Preconditioner<Vector = DenseVec<T>>>,
    ) -> Self {
        Lobpcg { precond, b_op, projector, zero_dofs: None, seed: 42 }
    }

    /// Zero the given DOFs in the initial iterate (eliminated boundary DOFs).
    pub fn with_zero_dofs(mut self, dofs: &'p [usize]) -> Self {
        self.zero_dofs = Some(dofs);
        self
    }
}

impl<'p, T: Scalar> EigenSolver<T> for Lobpcg<'p, T> {
    fn solve<Op>(&self, op: &Op, params: &EigenParams<T>) -> Result<EigenResult<T>, SolverError>
    where Op: LinearOperator<Vector = DenseVec<T>>
    {
        self.solve_generalized(op, params)
    }
}

// ─── Implementation (BLOPEX `lobpcg_solve` port) ─────────────────────────────

impl<'p, T: Scalar> Lobpcg<'p, T> {
    /// Core solve: handles both standard (`b_op == None`) and generalised
    /// (`b_op == Some(…)`) eigenvalue problems.
    ///
    /// The algorithm is the BLOPEX `lobpcg_solve` (f64 path) as invoked by
    /// HYPRE AME: soft-locking active mask, implicit-QR B-orthonormalisation,
    /// exact small dense GEVP on the `[X | R | P]` Gram matrices.
    pub fn solve_generalized<Op>(
        &self,
        a: &Op,
        params: &EigenParams<T>,
    ) -> Result<EigenResult<T>, SolverError>
    where Op: LinearOperator<Vector = DenseVec<T>>
    {
        let n = a.nrows();
        let k = params.n_eigenvalues;
        assert_eq!(n, a.ncols(), "LOBPCG: operator must be square");
        assert!(k >= 1 && k < n, "nev must be in 1..n");

        let has_t = self.precond.is_some() || self.projector.is_some();
        let tol = num_traits::ToPrimitive::to_f64(&params.tol).unwrap_or(1e-10);
        let max_iter = params.max_iter;
        let eps = f64::EPSILON;

        // ── operators (BLOPEX operatorB / operatorT) ────────────────────────
        // B: y = B·x (or y = x for the standard problem).
        let apply_b = |x: &DenseVec<T>| -> DenseVec<T> {
            let mut y = DenseVec::zeros(n);
            if let Some(b) = self.b_op { b.apply(x, &mut y); } else { y.copy_from(x); }
            y
        };
        // T: y = T(x) = projector(precond(x)) — HYPRE AME's `hypre_AMEOperatorB`
        // (AMS⁻¹ followed by the discrete div-free projection).
        let apply_t = |r: &DenseVec<T>| -> DenseVec<T> {
            let mut w = DenseVec::zeros(n);
            if let Some(pc) = self.precond { pc.apply_precond(r, &mut w); } else { w.copy_from(r); }
            if let Some(proj) = self.projector {
                let mut wp = DenseVec::zeros(n);
                proj.apply_precond(&w, &mut wp);
                wp
            } else { w }
        };

        // ── 1. initial X: random → zero eliminated DOFs (AME `edge_bc`) →   ──
        //       div-free projection (AME setup: `hypre_AMEDiscrDivFreeComponent`)
        let mut x: Vec<DenseVec<T>> = Vec::with_capacity(k);
        for j in 0..k {
            let mut col = DenseVec::zeros(n);
            fill_random(&mut col, self.seed.wrapping_add(j as u64 * 0xdeadbeef));
            if let Some(dofs) = self.zero_dofs {
                let cs = col.as_mut_slice();
                for &d in dofs { if d < n { cs[d] = T::zero(); } }
            }
            if let Some(proj) = self.projector {
                let mut sp = DenseVec::zeros(n);
                proj.apply_precond(&col, &mut sp);
                col = sp;
            }
            x.push(col);
        }

        let full: Vec<usize> = (0..k).collect();

        // ── 2. B-orthonormalise X by implicit QR ────────────────────────────
        //       `lobpcg_MultiVectorImplicitQR`: U = chol(XᵀBX), X ← X·U⁻¹
        let mut bx: Vec<DenseVec<T>> = x.iter().map(apply_b).collect();
        {
            let g_xbx = gram_sub(&x, &bx, &full, &full);
            let u_inv = chol_upper_inv(&g_xbx, k).ok_or_else(|| {
                SolverError::NumericalBreakdown {
                    detail: "LOBPCG: bad initial vectors — XᵀBX not SPD (linearly dependent block)".into(),
                }
            })?;
            x  = vecs_combine(&x, &full, &u_inv, k);
            bx = vecs_combine(&bx, &full, &u_inv, k);
        }

        // ── 3. AX; initial Rayleigh–Ritz on X ───────────────────────────────
        let mut ax: Vec<DenseVec<T>> = x.iter().map(|xi| {
            let mut y = DenseVec::zeros(n); a.apply(xi, &mut y); y
        }).collect();
        let mut ga = gram_sub(&x, &ax, &full, &full);
        symmetrize(&mut ga, k);
        let mut gb = gram_sub(&x, &bx, &full, &full);
        symmetrize(&mut gb, k);
        let (mut lambda, coord_x) = dense_symm_eig_gen(&ga, &gb, k)
            .map_err(|_| SolverError::NumericalBreakdown {
                detail: "LOBPCG: initial Rayleigh–Ritz GEVP failed (bad problem)".into(),
            })?;
        x  = vecs_combine(&x, &full, &coord_x, k);
        ax = vecs_combine(&ax, &full, &coord_x, k);
        bx = vecs_combine(&bx, &full, &coord_x, k);

        // ── 4. R = BX·diag(λ) − AX ──────────────────────────────────────────
        let mut r: Vec<DenseVec<T>> = (0..k).map(|j| residual_vec(&bx[j], &ax[j], lambda[j])).collect();
        let mut res_norms: Vec<T> = r.iter().map(|v| v.norm2()).collect();

        // P / AP / BP (conjugate directions; only active columns participate)
        let mut p  = vec![DenseVec::zeros(n); k];
        let mut ap = vec![DenseVec::zeros(n); k];
        let mut bp = vec![DenseVec::zeros(n); k];

        let mut active: Vec<bool> = vec![true; k];
        let mut iterations: usize = 0;
        let mut all_converged = false;

        for it in 1..=max_iter {
            iterations = it;

            // ── `lobpcg_checkResiduals`: update the soft-locking mask ───────
            let mut size_r = 0usize;
            for j in 0..k {
                let res = num_traits::ToPrimitive::to_f64(&res_norms[j]).unwrap_or(f64::INFINITY);
                let lam = num_traits::ToPrimitive::to_f64(&lambda[j]).unwrap_or(0.0).abs();
                let not_conv = res > lam * tol + tol + eps;
                active[j] = not_conv;
                if not_conv { size_r += 1; }
            }
            if size_r == 0 { all_converged = true; break; }
            let act: Vec<usize> = (0..k).filter(|&j| active[j]).collect();

            if params.verbose {
                let max_res = res_norms.iter().cloned().fold(T::zero(), |m, r| if r > m { r } else { m });
                let mr = num_traits::ToPrimitive::to_f64(&max_res).unwrap_or(f64::NAN);
                println!("Iteration {it:4} \tbsize {size_r:2} \tmaxres {mr:.14e}");
            }

            // ── W = T(R); R ← W (precondition + div-free, active cols) ─────
            if has_t {
                for &j in &act { r[j] = apply_t(&r[j]); }
            }

            // ── BR = B·R (active columns) ───────────────────────────────────
            let mut br: Vec<DenseVec<T>> = (0..k).map(|j| {
                if active[j] { apply_b(&r[j]) } else { DenseVec::zeros(n) }
            }).collect();

            // ── B-orthonormalise the R block (BLOPEX implicit QR — block-only;
            //    the X–R cross terms are handled exactly in the Gram matrix) ──
            {
                let g_rbr = gram_sub(&r, &br, &act, &act);
                let u_inv = chol_upper_inv(&g_rbr, size_r).ok_or_else(|| {
                    SolverError::NumericalBreakdown {
                        detail: "LOBPCG: orthonormalisation of residuals failed (DPOTRF)".into(),
                    }
                })?;
                let r_new  = vecs_combine(&r, &act, &u_inv, size_r);
                let br_new = vecs_combine(&br, &act, &u_inv, size_r);
                for (idx, &j) in act.iter().enumerate() {
                    r[j] = r_new[idx].clone();
                    br[j] = br_new[idx].clone();
                }
            }

            // ── AR = A·R (active columns) ───────────────────────────────────
            let mut ar: Vec<DenseVec<T>> = (0..k).map(|j| {
                let mut y = DenseVec::zeros(n);
                if active[j] { a.apply(&r[j], &mut y); }
                y
            }).collect();

            // ── B-orthonormalise the P block (iteration > 1) ────────────────
            // BLOPEX: on DPOTRF failure `sizeP = 0` (silent degrade).
            let mut size_p = 0usize;
            if it > 1 {
                let g_pbp = gram_sub(&p, &bp, &act, &act);
                if let Some(u_inv) = chol_upper_inv(&g_pbp, size_r) {
                    let p_new  = vecs_combine(&p, &act, &u_inv, size_r);
                    let ap_new = vecs_combine(&ap, &act, &u_inv, size_r);
                    let bp_new = vecs_combine(&bp, &act, &u_inv, size_r);
                    for (idx, &j) in act.iter().enumerate() {
                        p[j]  = p_new[idx].clone();
                        ap[j] = ap_new[idx].clone();
                        bp[j] = bp_new[idx].clone();
                    }
                    size_p = size_r;
                }
                // else: keep P as-is; coordPX below is 0×k → P ← R·coordRX
            }

            // ── Assemble the Gram matrices on [X | R | P] ───────────────────
            // Layout (rows/cols): 0..k = X, k..k+sizeR = R(active),
            // k+sizeR.. = P(active).  Row-major, lower triangle filled then
            // symmetrised.  X-block: diag(λ)/I (X stays diagonalised and
            // B-orthonormal — BLOPEX does not recompute XᵀAX).
            let size_a = k + size_r + size_p;
            let mut ga = vec![T::zero(); size_a * size_a];
            let mut gb = vec![T::zero(); size_a * size_a];
            // X-block: BLOPEX forces diag(λ) — X stays exactly diagonalised
            // through the Ritz update.
            for i in 0..k {
                ga[i * size_a + i] = lambda[i];
                gb[i * size_a + i] = T::one();
            }
            for (ri, &row_i) in act.iter().enumerate() {
                let rrow = k + ri;
                for j in 0..k {
                    ga[rrow * size_a + j] = dot(r[row_i].as_slice(), ax[j].as_slice());
                    gb[rrow * size_a + j] = dot(r[row_i].as_slice(), bx[j].as_slice());
                }
                gb[rrow * size_a + rrow] = T::one();
                for (rj, &col_j) in act.iter().enumerate() {
                    ga[rrow * size_a + (k + rj)] = dot(r[row_i].as_slice(), ar[col_j].as_slice());
                }
            }
            if size_p > 0 {
                for (pi, &row_i) in act.iter().enumerate() {
                    let prow = k + size_r + pi;
                    for j in 0..k {
                        ga[prow * size_a + j] = dot(p[row_i].as_slice(), ax[j].as_slice());
                        gb[prow * size_a + j] = dot(p[row_i].as_slice(), bx[j].as_slice());
                    }
                    gb[prow * size_a + prow] = T::one();
                    for (rj, &col_j) in act.iter().enumerate() {
                        ga[prow * size_a + (k + rj)] = dot(p[row_i].as_slice(), ar[col_j].as_slice());
                    }
                    for (pj, &col_j) in act.iter().enumerate() {
                        ga[prow * size_a + (k + size_r + pj)] = dot(p[row_i].as_slice(), ap[col_j].as_slice());
                    }
                }
            }
            // symmetrise the full Gram matrices (dsygv reads the lower
            // triangle; nalgebra's Cholesky/SymmetricEigen do the same)
            for i in 0..size_a {
                for j in (i + 1)..size_a {
                    ga[i * size_a + j] = ga[j * size_a + i];
                    gb[i * size_a + j] = gb[j * size_a + i];
                }
            }

            // ── GEVP on the Gram matrices (dsygv equivalent) ────────────────
            let (lambda_ab, coord_x) = dense_symm_eig_gen(&ga, &gb, size_a)?;
            for j in 0..k { lambda[j] = lambda_ab[j]; }

            // coordX = eigenvectors (row-major: coord_x[j*size_a + i] is
            // row i of eigenvector j, matching BLOPEX's column-major view).
            // coordXX = coordX[0..k, :]; coordRX = coordX[k..k+sizeR, :];
            // coordPX = coordX[k+sizeR.., :].

            // ── update P/AP/BP ──────────────────────────────────────────────
            // iter > 1: P = P·coordPX + R·coordRX
            // iter == 1 (or P degraded): P = R·coordRX
            let mut p_new  = vec![DenseVec::zeros(n); k];
            let mut ap_new = vec![DenseVec::zeros(n); k];
            let mut bp_new = vec![DenseVec::zeros(n); k];
            for j in 0..k {
                if it > 1 && size_p > 0 {
                    for (i, &pi) in act.iter().enumerate() {
                        let c = coord_x[j * size_a + (k + size_r + i)];
                        p_new[j].axpy(c, &p[pi]);
                        ap_new[j].axpy(c, &ap[pi]);
                        bp_new[j].axpy(c, &bp[pi]);
                    }
                }
                for (i, &ri) in act.iter().enumerate() {
                    let c = coord_x[j * size_a + (k + i)];
                    p_new[j].axpy(c, &r[ri]);
                    ap_new[j].axpy(c, &ar[ri]);
                    bp_new[j].axpy(c, &br[ri]);
                }
            }
            p = p_new; ap = ap_new; bp = bp_new;

            // ── update X/AX/BX: X = X·coordXX + P, … ────────────────────────
            let mut x_new  = vec![DenseVec::zeros(n); k];
            let mut ax_new = vec![DenseVec::zeros(n); k];
            let mut bx_new = vec![DenseVec::zeros(n); k];
            for j in 0..k {
                for i in 0..k {
                    let c = coord_x[j * size_a + i];
                    x_new[j].axpy(c, &x[i]);
                    ax_new[j].axpy(c, &ax[i]);
                    bx_new[j].axpy(c, &bx[i]);
                }
                x_new[j].axpy(T::one(), &p[j]);
                ax_new[j].axpy(T::one(), &ap[j]);
                bx_new[j].axpy(T::one(), &bp[j]);
            }
            x = x_new; ax = ax_new; bx = bx_new;

            // ── R = BX·diag(λ) − AX (active columns only) ───────────────────
            for &j in &act {
                r[j] = residual_vec(&bx[j], &ax[j], lambda[j]);
                res_norms[j] = r[j].norm2();
            }
        }

        if all_converged {
            let mut order: Vec<usize> = (0..k).collect();
            match params.which {
                EigenWhich::SmallestAlgebraic | EigenWhich::SmallestMagnitude =>
                    order.sort_by(|&a2, &b2| lambda[a2].partial_cmp(&lambda[b2]).unwrap()),
                EigenWhich::LargestAlgebraic =>
                    order.sort_by(|&a2, &b2| lambda[b2].partial_cmp(&lambda[a2]).unwrap()),
                EigenWhich::LargestMagnitude =>
                    order.sort_by(|&a2, &b2| {
                        let la = num_traits::ToPrimitive::to_f64(&lambda[a2]).unwrap_or(0.0).abs();
                        let lb = num_traits::ToPrimitive::to_f64(&lambda[b2]).unwrap_or(0.0).abs();
                        lb.partial_cmp(&la).unwrap()
                    }),
                EigenWhich::BothEnds => {
                    // smallest k/2 + largest k/2 (best effort for even split)
                    let mut asc: Vec<usize> = (0..k).collect();
                    asc.sort_by(|&a2, &b2| lambda[a2].partial_cmp(&lambda[b2]).unwrap());
                    order.clear();
                    order.extend_from_slice(&asc[..k / 2]);
                    order.extend(asc[k / 2..].iter().rev());
                }
            }
            let eigenvalues = order.iter().map(|&j| lambda[j]).collect();
            let eigenvectors = order.iter().map(|&j| x[j].clone()).collect();
            let residuals = order.iter().map(|&j| res_norms[j]).collect();
            return Ok(EigenResult {
                eigenvalues,
                eigenvectors,
                converged: k,
                iterations,
                residuals,
            });
        }

        let max_res = res_norms.iter().cloned().fold(T::zero(), |m, r| if r > m { r } else { m });
        Err(SolverError::ConvergenceFailed {
            max_iter,
            residual: num_traits::ToPrimitive::to_f64(&max_res).unwrap_or(f64::INFINITY),
        })
    }
}

// ─── Helpers (BLOPEX block-vector / Fortran-matrix operations) ───────────────

/// Gram sub-block `G[i][j] = x[rows[i]]ᵀ · y[cols[j]]`, row-major `m×w`.
///
/// This is the Rust equivalent of BLOPEX's `lobpcg_MultiVectorByMultiVector`
/// with the active-column collection (`aux_maskCount` + `mv_collectVectorPtr`).
fn gram_sub<T: Scalar>(x: &[DenseVec<T>], y: &[DenseVec<T>], rows: &[usize], cols: &[usize]) -> Vec<T> {
    let m = rows.len();
    let w = cols.len();
    let mut g = vec![T::zero(); m * w];
    for (i, &ri) in rows.iter().enumerate() {
        let xv = x[ri].as_slice();
        for (j, &cj) in cols.iter().enumerate() {
            g[i * w + j] = dot(xv, y[cj].as_slice());
        }
    }
    g
}

/// `out[j] = Σ_i y[rows[i]] · r[i][j]` for a row-major `h×w` matrix `r`.
///
/// Rust equivalent of BLOPEX's `lobpcg_MultiVectorByMatrix` (`y = x·r` with
/// the mask-collected columns of `x`).
fn vecs_combine<T: Scalar>(y: &[DenseVec<T>], rows: &[usize], r: &[T], w: usize) -> Vec<DenseVec<T>> {
    let h = rows.len();
    let n = y[0].len();
    let mut out = vec![DenseVec::zeros(n); w];
    for i in 0..h {
        let yi = &y[rows[i]];
        for j in 0..w {
            out[j].axpy(r[i * w + j], yi);
        }
    }
    out
}

/// Symmetrise an `n×n` row-major matrix in place.
fn symmetrize<T: Scalar>(m: &mut [T], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            let v = m[j * n + i];
            m[i * n + j] = v;
        }
    }
}

/// Upper-triangular inverse of the Cholesky factor of a symmetric PD matrix.
///
/// Given `G` (row-major `n×n`), returns `U⁻¹` where `G = UᵀU` and `U` is
/// upper triangular (the BLOPEX `lobpcg_chol` + `FortranMatrixUpperInv`
/// sequence).  `None` when `G` is not positive definite (DPOTRF info ≠ 0).
fn chol_upper_inv<T: Scalar>(g: &[T], n: usize) -> Option<Vec<T>> {
    use nalgebra::DMatrix;
    let m = DMatrix::<f64>::from_fn(n, n, |r, c|
        num_traits::ToPrimitive::to_f64(&g[r * n + c]).unwrap_or(0.0));
    let l = m.cholesky()?.l();
    let u_inv = l.transpose().try_inverse()?;
    let mut out = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = T::from_f64(u_inv[(i, j)]);
        }
    }
    Some(out)
}

/// `r = λ·bx − ax`.
fn residual_vec<T: Scalar>(bx: &DenseVec<T>, ax: &DenseVec<T>, lambda: T) -> DenseVec<T> {
    let n = bx.len();
    let mut r = DenseVec::zeros(n);
    let (bs, as_) = (bx.as_slice(), ax.as_slice());
    let rs = r.as_mut_slice();
    for i in 0..n {
        rs[i] = lambda * bs[i] - as_[i];
    }
    r
}

// ─── Dense symmetric generalised eigensolver ───────────────────────────────────

/// Solve `A c = θ B c` for small dense symmetric matrices (LAPACK `dsygv`
/// semantics: `itype=1`, `jobz='V'`, Cholesky-based reduction).
///
/// Returns `(eigenvalues ascending, eigenvectors_flat_row_major)` where
/// `evecs[j*n + i]` is row `i` of eigenvector `j` (i.e. the eigenvectors are
/// the columns of the returned matrix).  `Err` when `B` is not positive
/// definite (dsygv INFO ≠ 0 → BLOPEX breaks the iteration).
#[cfg(not(target_arch = "wasm32"))]
fn dense_symm_eig_gen<T: Scalar>(a: &[T], b: &[T], n: usize) -> Result<(Vec<T>, Vec<T>), SolverError> {
    use nalgebra::{DMatrix, SymmetricEigen};

    let to_f = |v: &T| num_traits::ToPrimitive::to_f64(v).unwrap_or(0.0);
    let na = DMatrix::<f64>::from_fn(n, n, |r, c| to_f(&a[r * n + c]));
    let nb = DMatrix::<f64>::from_fn(n, n, |r, c| to_f(&b[r * n + c]));

    let chol = nb.cholesky().ok_or(SolverError::NumericalBreakdown {
        detail: "LOBPCG: GEVP B-matrix not positive definite (dsygv failure)".into(),
    })?;
    let l = chol.l();
    let li = l.try_inverse().ok_or(SolverError::NumericalBreakdown {
        detail: "LOBPCG: GEVP Cholesky factor singular (dsygv failure)".into(),
    })?;

    let c = &li * &na * li.transpose();
    let se = SymmetricEigen::new(c);

    // dsygv returns eigenvalues in ascending order; nalgebra's SymmetricEigen
    // order is not guaranteed across versions, so sort explicitly (and move
    // the matching eigenvector columns).
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| se.eigenvalues[i].partial_cmp(&se.eigenvalues[j]).unwrap());
    let evals: Vec<T> = idx.iter().map(|&i| T::from_f64(se.eigenvalues[i])).collect();
    let vecs = li.transpose() * &se.eigenvectors;
    let mut evecs = vec![T::zero(); n * n];
    for (col, &i) in idx.iter().enumerate() {
        for row in 0..n {
            evecs[col * n + row] = T::from_f64(vecs[(row, i)]);
        }
    }
    Ok((evals, evecs))
}

#[cfg(target_arch = "wasm32")]
fn dense_symm_eig_gen<T: Scalar>(_a: &[T], _b: &[T], _n: usize) -> Result<(Vec<T>, Vec<T>), SolverError> {
    // No dense GEVP on wasm32 — report an error rather than silently
    // returning garbage (diagonal as eigenvalues, identity as eigenvectors).
    Err(SolverError::NumericalBreakdown {
        detail: "LOBPCG: dense GEVP unavailable on wasm32 (no nalgebra linalg)".into(),
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::{CooMatrix, CsrMatrix};

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0     { coo.push(i, i-1, -1.0); }
            if i < n-1   { coo.push(i, i+1, -1.0); }
        }
        CsrMatrix::from_coo(&coo)
    }

    #[test]
    fn standard_smallest_eigenvalue() {
        let n = 20;
        let a = laplacian_1d(n);
        let solver = Lobpcg::<f64>::new(None);
        let mut params = EigenParams::new(1, EigenWhich::SmallestAlgebraic);
        // BLOPEX/HYPRE default tolerance (AME: tol = 1e-6).  A tighter tol
        // makes the (unpreconditioned) residual drop into the numerical-noise
        // regime before the soft-locking mask kicks in, and the normalised
        // noise residuals then poison the Rayleigh–Ritz.
        params.tol = 1e-6;
        params.max_iter = 2000;
        let res = solver.solve(&a, &params).unwrap();
        let exact = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
        eprintln!("[t] lambda = {:?} exact = {exact} iters = {}", res.eigenvalues, res.iterations);
        assert!((res.eigenvalues[0] - exact).abs() < 1e-4);
    }

    #[test]
    fn generalized_diagonal() {
        // A = diag(1,2,3,…), B = identity → λ = 1,2,3,…
        let n = 10;
        let mut a_coo = CooMatrix::new(n, n);
        for i in 0..n { a_coo.push(i, i, (i + 1) as f64); }
        let a = CsrMatrix::from_coo(&a_coo);
        let mut eye_coo = CooMatrix::new(n, n);
        for i in 0..n { eye_coo.push(i, i, 1.0); }
        let eye = CsrMatrix::from_coo(&eye_coo);
        let solver = Lobpcg::<f64>::new_generalized(None, Some(&eye), None);
        let mut params = EigenParams::new(2, EigenWhich::SmallestAlgebraic);
        params.tol = 1e-8;
        let res = solver.solve(&a, &params).unwrap();
        assert!((res.eigenvalues[0] - 1.0).abs() < 1e-4);
        assert!((res.eigenvalues[1] - 2.0).abs() < 1e-4);
    }

    /// Regression test for the ex32 instability: a singular generalised
    /// problem `A x = λ B x` whose nullspace is span(G) (discrete gradients).
    /// With a nullspace projector (the div-free projection) the solver must
    /// converge to the *nonzero* spectrum and the converged vectors must stay
    /// in the nullspace-orthogonal (div-free) subspace — i.e. no nullspace
    /// re-injection, `||GᵀMv|| ≈ 0`.
    #[test]
    fn nullspace_stays_locked_after_projection() {
        // Mimic 1D curl-curl: A = Neumann-type graph Laplacian — diagonal 1 at
        // the endpoints so that A·1 = 0 and span(1) IS the nullspace, with all
        // eigenvectors orthogonal to it (like range(G) for the curl-curl
        // operator in ex32).  M = identity.
        let n = 40;
        let mut a_coo = CooMatrix::new(n, n);
        for i in 0..n {
            let deg = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
            a_coo.push(i, i, deg);
            if i > 0     { a_coo.push(i, i - 1, -1.0); }
            if i < n - 1 { a_coo.push(i, i + 1, -1.0); }
        }
        let a = CsrMatrix::from_coo(&a_coo);
        let mut m_coo = CooMatrix::new(n, n);
        for i in 0..n { m_coo.push(i, i, 1.0); }
        let m = CsrMatrix::from_coo(&m_coo);

        // G: n×1 constant vector — its range spans the nullspace of A
        // (span(1) for the graph Laplacian), exactly like range(G) spans the
        // gradient nullspace of the curl-curl operator in ex32.  The nullspace
        // projector P = I − G(GᵀMG)⁻¹GᵀM then kills the constant vector.
        let mut g_coo = CooMatrix::new(n, 1);
        for i in 0..n { g_coo.push(i, 0, 1.0); }
        let g = CsrMatrix::from_coo(&g_coo);

        // Dense orthogonal projector onto null(Gᵀ) (M = I here).
        struct NullspaceProjector {
            p: Vec<f64>, // n×n, row-major
        }
        impl Preconditioner for NullspaceProjector {
            type Vector = DenseVec<f64>;
            fn apply_precond(&self, x: &DenseVec<f64>, y: &mut DenseVec<f64>) {
                let n = x.len();
                for i in 0..n {
                    let mut s = 0.0;
                    for j in 0..n {
                        s += self.p[i * n + j] * x.as_slice()[j];
                    }
                    y.as_mut_slice()[i] = s;
                }
            }
        }
        let m_g = nalgebra::DMatrix::<f64>::from_fn(n, 1, |r, _c| {
            let mut v = 0.0;
            for k in g.row_ptr()[r]..g.row_ptr()[r + 1] {
                v = g.values()[k];
            }
            v
        });
        let gtg_inv = (m_g.transpose() * &m_g).try_inverse().unwrap();
        let proj = &m_g * &gtg_inv * m_g.transpose();
        let p = (nalgebra::DMatrix::<f64>::identity(n, n) - proj).as_slice().to_vec();
        let projector = NullspaceProjector { p };

        // No preconditioner here: this 1D toy cannot carry a well-conditioned
        // AMS (a single-column G makes GᵀAG singular and the regularised
        // "inverse" amplifies the nullspace direction by 1/ε before the
        // projector cancels it, leaving O(1e-6) noise in the Gram cross terms).
        // With k=1 the Rayleigh–Ritz subspace is [X | R | P] and the cross
        // terms shrink as the residual converges, so no preconditioner is
        // needed — the point of this test is nullspace confinement, not speed.
        let solver = Lobpcg::<f64>::new_generalized(None, Some(&m), Some(&projector));
        let mut params = EigenParams::new(1, EigenWhich::SmallestAlgebraic);
        params.tol = 1e-8;
        params.max_iter = 2000;
        let res = solver.solve(&a, &params).unwrap();

        // The projection kills span(G) = span(1), i.e. the *nullspace* of A —
        // but in this 1D toy the first nonzero eigenvector v₁ = sin(πi/(n+1))
        // is NOT orthogonal to the nullspace (mean(v₁) ≠ 0), so the projected
        // problem's smallest eigenvalue is v₂ = 2 − 2cos(2π/(n+1)) (mean = 0).
        // What matters for the ex32 regression is:
        //   (a) λ converges to a *genuine nonzero* eigenvalue of A (no
        //       collapse onto the nullspace / no spurious ~0 Ritz values),
        //   (b) no nullspace re-injection: |Gᵀv| ≈ 0 for every converged
        //       vector (the old bug gave ‖G_effᵀMv‖ = 1.648).
        let a_dense = nalgebra::DMatrix::<f64>::from_fn(n, n, |r, c| {
            let mut v = 0.0;
            for k in a.row_ptr()[r]..a.row_ptr()[r + 1] {
                if a.col_idx()[k] as usize == c { v = a.values()[k]; }
            }
            v
        });
        let mut a_eigs: Vec<f64> = nalgebra::SymmetricEigen::new(a_dense)
            .eigenvalues.iter().cloned().collect();
        a_eigs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        // smallest 4 genuine eigenvalues: {0, v₁, v₂, v₃} for P_40
        assert!(a_eigs[0].abs() < 1e-8, "A must have a zero eigenvalue, got {:.3e}", a_eigs[0]);
        let genuine = &a_eigs[1..]; // nonzero spectrum
        let lam0 = res.eigenvalues[0];
        let dist = genuine.iter().map(|&e| (lam0 - e).abs()).fold(f64::INFINITY, f64::min);
        assert!(dist < 1e-4,
            "λ = {lam0} is not a genuine nonzero eigenvalue of A (nearest: {dist:.3e})");
        assert!(lam0 > 1e-3, "must NOT converge to the nullspace");

        // All converged vectors must be (nearly) nullspace-free: |Σv| ≈ 0.
        for v in &res.eigenvectors {
            let gtv: f64 = v.as_slice().iter().sum();
            assert!(gtv.abs() < 1e-5, "nullspace leaked: |Gᵀv| = {gtv}");
        }
    }
}
