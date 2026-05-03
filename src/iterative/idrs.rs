//! IDR(s) — Induced Dimension Reduction Krylov solver.
//!
//! A short-recurrence method for **non-symmetric** (and symmetric) square
//! systems.  IDR(s) builds a decreasing sequence of residuals that lie in
//! (nested) affine subspaces, converging in at most ⌈n/s⌉ outer steps in
//! exact arithmetic.
//!
//! s=1 gives BiCGSTAB-like behaviour; larger s reduces matvec count per unit
//! convergence at the cost of more inner products per step.
//!
//! ## Algorithm
//!
//! Implements Algorithm IDR(s) from:
//! > van Gijzen, M.B., and Sonneveld, P. (2011).
//! > *Algorithm 913: An elegant IDR(s) variant that efficiently exploits
//! > biorthogonality properties.*
//! > ACM Trans. Math. Software, 38(1), 5:1–5:19.
//!
//! Uses bi-orthogonalization to maintain lower-triangular M matrix.
//! At inner step k, solves M[k:s, k:s] * c = f[k:s] (forward substitution),
//! then updates `v = r - G[:,k:s]*c` and `U[:,k] = omega*uhat + U[:,k:s]*c`.
//!
//! ## Analogs
//!
//! PETSc: `KSPSetType(ksp, KSPIDRS)`

#![allow(clippy::needless_range_loop)]
use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
};

// ─── Public struct ────────────────────────────────────────────────────────────

/// IDR(s) solver for general (non-symmetric) systems.
pub struct Idrs<T> {
    s: usize,
    max_restarts: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Idrs<T> {
    pub fn new(s: usize) -> Self {
        Idrs { s: s.max(1), max_restarts: 3, _phantom: std::marker::PhantomData }
    }

    /// Set maximum number of shadow-space restarts on breakdown.
    pub fn with_max_restarts(mut self, n: usize) -> Self {
        self.max_restarts = n;
        self
    }
}

impl<T: Scalar> Default for Idrs<T> {
    fn default() -> Self { Self::new(4) }
}

// ─── KrylovSolver impl ────────────────────────────────────────────────────────

impl<T: Scalar> KrylovSolver for Idrs<T> {
    type Vector  = DenseVec<T>;

    fn solve(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        params: &SolverParams,
    ) -> Result<SolverResult, SolverError> {
        let n = b.len();
        if op.nrows() != n || op.ncols() != x.len() {
            return Err(SolverError::DimensionMismatch {
                op_rows: op.nrows(), op_cols: op.ncols(), rhs_len: n,
            });
        }

        let s        = self.s;
        let tol      = T::from_f64(params.rtol);
        let atol     = T::from_f64(params.atol);
        let max_iter = params.max_iter;

        let norm_b   = b.norm2();
        let norm_b_f = if norm_b == T::zero() { T::one() } else { norm_b };
        let mut residual_history: Vec<f64> = Vec::new();

        // Pre-allocated scratch buffers for preconditioner application (avoids
        // per-call Vec allocation on the hot path).
        let mut pc_in  = DenseVec::zeros(n);
        let mut pc_out = DenseVec::zeros(n);
        // Scratch buffer for u_k matvec (avoids clone per inner step).
        let mut u_dense = DenseVec::zeros(n);
        let mut g_k_dense = DenseVec::zeros(n);

        // ── Euclidean norm of a slice ─────────────────────────────────────────
        let rnorm = |r: &[T]| -> T {
            let mut s2 = T::zero();
            for &v in r { s2 += v * v; }
            s2.sqrt()
        };

        // ── Initial residual r = b − A x ──────────────────────────────────────
        let mut r = vec![T::zero(); n];
        {
            let mut ax = DenseVec::zeros(n);
            op.apply(x, &mut ax);
            crate::simd::dense_ops::simd_sub(b.as_slice(), ax.as_slice(), &mut r);
        }

        if rnorm(&r) <= atol || rnorm(&r) <= tol * norm_b_f {
            return Ok(SolverResult { converged: true, iterations: 0,
                final_residual: to_f64(rnorm(&r) / norm_b_f),
                residual_history, history: None });
        }

        // ── Build s orthonormal shadow vectors P (random, deterministic seed) ──
        let mut p_rows: Vec<Vec<T>> = (0..s).map(|i| {
            let mut row = vec![T::zero(); n];
            let mut state: u64 = 0x123456789abcdefu64.wrapping_add(i as u64 * 2654435769);
            for v in row.iter_mut() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let f = ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
                *v = T::from_f64(f);
            }
            row
        }).collect();
        // Orthonormalise via modified Gram-Schmidt.
        for i in 0..s {
            for k in 0..i {
                let dot = dot_slice(&p_rows[k], &p_rows[i]);
                let pk = p_rows[k].clone();
                for j in 0..n { p_rows[i][j] -= dot * pk[j]; }
            }
            let nrm2: T = p_rows[i].iter().map(|&v| v * v).fold(T::zero(), |a, b| a + b);
            let nrm = nrm2.sqrt();
            if nrm > T::from_f64(1e-14) {
                for v in &mut p_rows[i] { *v /= nrm; }
            }
        }

        // ── IDR(s) main loop (van Gijzen & Sonneveld 2011, Algorithm 913) ─────
        //
        // G[:,k], U[:,k]: s column vectors of length n.
        let mut g_cols: Vec<Vec<T>> = vec![vec![T::zero(); n]; s];
        let mut u_cols: Vec<Vec<T>> = vec![vec![T::zero(); n]; s];

        // M: s×s lower-triangular matrix. M[i][j] = p_i · G[:,j].
        // After biorthogonalization M[i][j] = 0 for i < j.
        // Initialised to identity so that M[k:s, k:s] is non-singular from
        // the very first iteration (G=0 case degenerates; we skip if M[k,k]≈0).
        let mut m_mat: Vec<Vec<T>> = vec![vec![T::zero(); s]; s];
        for i in 0..s { m_mat[i][i] = T::one(); }

        // f[i] = p_i · r  (updated incrementally throughout).
        let mut f: Vec<T> = (0..s).map(|i| dot_slice(&p_rows[i], &r)).collect();

        let mut iters = 0usize;
        let mut converged = false;
        let mut omega = T::one();
        let mut restarts = 0usize;
        let max_restarts = self.max_restarts;

        // Helper: rebuild shadow space P with a given seed offset.
        let build_shadow = |seed_offset: u64| -> Vec<Vec<T>> {
            let mut rows: Vec<Vec<T>> = (0..s).map(|i| {
                let mut row = vec![T::zero(); n];
                let mut state: u64 = 0x123456789abcdefu64
                    .wrapping_add(i as u64 * 2654435769)
                    .wrapping_add(seed_offset);
                for v in row.iter_mut() {
                    state = state.wrapping_mul(6364136223846793005)
                                 .wrapping_add(1442695040888963407);
                    let f = ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
                    *v = T::from_f64(f);
                }
                row
            }).collect();
            // Orthonormalise.
            for i in 0..s {
                for k in 0..i {
                    let dot = dot_slice(&rows[k], &rows[i]);
                    let pk = rows[k].clone();
                    for j in 0..n { rows[i][j] -= dot * pk[j]; }
                }
                let nrm2: T = rows[i].iter().map(|&v| v * v).fold(T::zero(), |a, b| a + b);
                let nrm = nrm2.sqrt();
                if nrm > T::from_f64(1e-14) {
                    for v in &mut rows[i] { *v /= nrm; }
                }
            }
            rows
        };
        let mut near_zero_count = 0usize;

        'outer: loop {
            for k in 0..s {
                // ── Forward substitution: M[k:s, k:s] * c = f[k:s] ──────────
                // c has length (s - k).
                let sub = s - k;
                let mut c = vec![T::zero(); sub];
                for row in 0..sub {
                    // absolute indices: k+row, k+col
                    let mut val = f[k + row];
                    for col in 0..row {
                        val -= m_mat[k + row][k + col] * c[col];
                    }
                    let diag = m_mat[k + row][k + row];
                    c[row] = if near_zero(diag) { T::zero() } else { val / diag };
                }

                // ── v = r − G[:,k:s] * c ─────────────────────────────────────
                let mut v = r.clone();
                for j in 0..sub {
                    let cj = c[j];
                    if cj == T::zero() { continue; }
                    let col = k + j;
                    for l in 0..n { v[l] -= cj * g_cols[col][l]; }
                }

                // ── uhat = precond(v) ─────────────────────────────────────────
                let mut uhat = vec![T::zero(); n];
                pc_in.as_mut_slice().copy_from_slice(&v);
                if let Some(pc) = precond {
                    pc.apply_precond(&pc_in, &mut pc_out);
                } else {
                    pc_out.copy_from(&pc_in);
                }
                uhat.copy_from_slice(pc_out.as_slice());

                // ── U[:,k] = omega * uhat + U[:,k:s] * c ─────────────────────
                let mut u_k = vec![T::zero(); n];
                for l in 0..n { u_k[l] = omega * uhat[l]; }
                for j in 0..sub {
                    let cj = c[j];
                    if cj == T::zero() { continue; }
                    let col = k + j;
                    for l in 0..n { u_k[l] += cj * u_cols[col][l]; }
                }

                // ── G[:,k] = A * U[:,k] ───────────────────────────────────────
                u_dense.as_mut_slice().copy_from_slice(&u_k);
                op.apply(&u_dense, &mut g_k_dense);
                let mut g_k = g_k_dense.as_slice().to_vec();

                // ── Bi-orthogonalise G[:,k] and U[:,k] against G[:,0..k-1] ───
                // After: p_i · G[:,k] = 0 for i < k.
                for i in 0..k {
                    let mii = m_mat[i][i];
                    if near_zero(mii) { continue; }
                    let alpha = dot_slice(&p_rows[i], &g_k) / mii;
                    for l in 0..n {
                        g_k[l] -= alpha * g_cols[i][l];
                        u_k[l] -= alpha * u_cols[i][l];
                    }
                }

                // ── M[i,k] = p_i · G[:,k]  for i = k..s ─────────────────────
                // (M[i,k] = 0 for i < k by biorthogonalization.)
                for i in k..s {
                    m_mat[i][k] = dot_slice(&p_rows[i], &g_k);
                }

                // ── beta = f[k] / M[k,k] ─────────────────────────────────────
                let m_kk = m_mat[k][k];
                if near_zero(m_kk) {
                    g_cols[k] = g_k;
                    u_cols[k] = u_k;
                    iters += 1;
                    near_zero_count += 1;
                    // If too many consecutive near-zero pivots, trigger restart.
                    if near_zero_count >= s && restarts < max_restarts {
                        restarts += 1;
                        near_zero_count = 0;
                        // Recompute r exactly from current x.
                        {
                            let mut ax = DenseVec::zeros(n);
                            op.apply(x, &mut ax);
                            crate::simd::dense_ops::simd_sub(b.as_slice(), ax.as_slice(), &mut r);
                        }
                        // Rebuild shadow space with new seed.
                        p_rows = build_shadow(restarts as u64 * 1337);
                        // Reset G, U, M, f.
                        g_cols = vec![vec![T::zero(); n]; s];
                        u_cols = vec![vec![T::zero(); n]; s];
                        m_mat = vec![vec![T::zero(); s]; s];
                        for i in 0..s { m_mat[i][i] = T::one(); }
                        f = (0..s).map(|i| dot_slice(&p_rows[i], &r)).collect();
                        omega = T::one();
                    }
                    continue;
                }
                near_zero_count = 0;
                let beta = f[k] / m_kk;

                // ── r -= beta * G[:,k],  x += beta * U[:,k] ──────────────────
                for l in 0..n { r[l] -= beta * g_k[l]; }
                { let xs = x.as_mut_slice(); for l in 0..n { xs[l] += beta * u_k[l]; } }

                // ── Update f ─────────────────────────────────────────────────
                // f[i] -= beta * M[i,k]  for i = k+1..s  (incremental update)
                for i in (k + 1)..s { f[i] -= beta * m_mat[i][k]; }
                // f[k] recomputed exactly from new r.
                f[k] = dot_slice(&p_rows[k], &r);

                // ── Store columns ─────────────────────────────────────────────
                g_cols[k] = g_k;
                u_cols[k] = u_k;

                iters += 1;
                {
                    let rn = rnorm(&r);
                    residual_history.push(to_f64(rn / norm_b_f));
                    if params.verbose == VerboseLevel::Iterations {
                        let res_f = to_f64(rn / norm_b_f);
                        println!("    IDR({s}) iter {:4}  ‖r‖/‖b‖ = {res_f:.6e}", iters);
                    }
                }
                if rnorm(&r) <= atol || rnorm(&r) <= tol * norm_b_f {
                    converged = true;
                    break 'outer;
                }
                if iters >= max_iter { break 'outer; }
            }

            if iters >= max_iter { break; }

            // ── Omega step (GMRES-1 minimisation) ────────────────────────────
            let mut uhat2 = vec![T::zero(); n];
            pc_in.as_mut_slice().copy_from_slice(&r);
            if let Some(pc) = precond {
                pc.apply_precond(&pc_in, &mut pc_out);
            } else {
                pc_out.copy_from(&pc_in);
            }
            uhat2.copy_from_slice(pc_out.as_slice());
            let mut t = DenseVec::zeros(n);
            op.apply(&pc_out, &mut t);

            let tr = dot_slice(t.as_slice(), &r);
            let tt = dot_slice(t.as_slice(), t.as_slice());
            omega = if tt.abs() < T::machine_epsilon() * T::from_f64(1e3) {
                T::from_f64(1e-8)
            } else {
                tr / tt
            };

            for l in 0..n { r[l] -= omega * t.as_slice()[l]; }
            { let xs = x.as_mut_slice(); for l in 0..n { xs[l] += omega * uhat2[l]; } }

            iters += 1;
            {
                let rn = rnorm(&r);
                residual_history.push(to_f64(rn / norm_b_f));
                if params.verbose == VerboseLevel::Iterations {
                    let res_f = to_f64(rn / norm_b_f);
                    println!("    IDR({s}) iter {:4}  ‖r‖/‖b‖ = {res_f:.6e}", iters);
                }
            }
            if rnorm(&r) <= atol || rnorm(&r) <= tol * norm_b_f {
                converged = true;
                break;
            }
            if iters >= max_iter { break; }

            // ── Fully recompute f = P r at end of each outer pass ─────────────
            for i in 0..s { f[i] = dot_slice(&p_rows[i], &r); }
        }

        let rn = rnorm(&r);
        let final_res = to_f64(rn / norm_b_f);
        if params.verbose != VerboseLevel::Silent {
            if converged {
                println!("IDR({s}): converged in {iters} iterations, rel_res={final_res:.3e}");
            } else {
                println!("IDR({s}): NOT converged after {iters} iterations, rel_res={final_res:.3e}");
            }
        }
        Ok(SolverResult { converged, iterations: iters, final_residual: final_res, residual_history, history: None })
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Near-zero pivot check: use sqrt(eps) to handle f32 without over-suppression.
fn near_zero<T: Scalar>(v: T) -> bool {
    v.abs() < T::machine_epsilon().sqrt()
}

fn dot_slice<T: Scalar>(a: &[T], b: &[T]) -> T {
    crate::simd::dense_ops::simd_dot(a, b)
}

fn to_f64<T: Scalar>(v: T) -> f64 {
    num_traits::ToPrimitive::to_f64(&v).unwrap_or(f64::INFINITY)
}
