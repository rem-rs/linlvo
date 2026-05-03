//! GMRES(m) for complex linear systems.
//!
//! Solves `A x = b` where `A` is a `LinearOperator` over `Complex<T>`.
//!
//! The algorithm is identical to real GMRES with two key differences:
//! - Modified Gram-Schmidt uses the **Hermitian** inner product
//!   `⟨u, v⟩ = Σ conj(uᵢ) vᵢ`.
//! - Givens rotations are **complex**: given `a, b ∈ ℂ`,
//!   the plane rotation `[c, s; −conj(s), c]` with real `c` and complex `s`
//!   annihilates `b` while preserving the 2-norm.
//!
//! # Stopping criterion
//! Converge when `‖r‖ / ‖b‖ ≤ rtol` OR `‖r‖ ≤ atol`.
//!
//! # Usage
//! ```ignore
//! use linger::iterative::complex_gmres::ComplexGmres;
//! use linger::core::{DenseMatrix, vector::DenseVec};
//! use num_complex::Complex;
//!
//! let a: DenseMatrix<Complex<f64>> = /* ... */;
//! let b: DenseVec<Complex<f64>> = /* ... */;
//! let mut x: DenseVec<Complex<f64>> = DenseVec::zero_like(&b);
//! let solver = ComplexGmres::<f64>::new(30);
//! let result = solver.solve(&a, &b, &mut x, 1e-8, 0.0, 500).unwrap();
//! ```

use crate::core::{
    operator::LinearOperator,
    scalar::Scalar,
    vector::{DenseVec, Vector},
};
use crate::core::error::SolverError;
use num_complex::Complex;
use num_traits::NumCast;

/// Convert a real scalar `T` to `f64` for reporting purposes.
#[inline]
fn to_f64<T: Scalar>(v: T) -> f64 {
    <f64 as NumCast>::from(v).unwrap_or(0.0)
}

/// Result returned by [`ComplexGmres`].
#[derive(Debug, Clone)]
pub struct ComplexGmresResult {
    /// Number of iterations actually performed.
    pub iters: usize,
    /// Final residual norm `‖r‖`.
    pub residual_norm: f64,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Residual history (one entry per outer restart, then the final value).
    pub residual_history: Vec<f64>,
}

/// Reusable scratch buffers for repeated solves on the same problem size.
pub struct ComplexGmresWorkspace<T: Scalar> {
    restart: usize,
    r: DenseVec<Complex<T>>,
    v: Vec<DenseVec<Complex<T>>>,
    w: DenseVec<Complex<T>>,
    /// Upper-Hessenberg matrix columns (h[j] has j+2 entries).
    h: Vec<Vec<Complex<T>>>,
    /// Real cosines of Givens rotations.
    cs: Vec<T>,
    /// Complex sines of Givens rotations.
    sn: Vec<Complex<T>>,
    /// Right-hand side of the projected least-squares problem.
    g: Vec<Complex<T>>,
}

impl<T: Scalar> ComplexGmresWorkspace<T> {
    pub fn new(n: usize, restart: usize) -> Self {
        let m = restart.max(1);
        let zero_c = Complex::new(T::zero(), T::zero());
        ComplexGmresWorkspace {
            restart: m,
            r: vec![zero_c; n].into(),
            v: (0..=m).map(|_| vec![zero_c; n].into()).collect(),
            w: vec![zero_c; n].into(),
            h: (0..m).map(|_| vec![zero_c; m + 1]).collect(),
            cs: vec![T::zero(); m],
            sn: vec![zero_c; m],
            g: vec![zero_c; m + 1],
        }
    }

    fn ensure_shape(&mut self, n: usize, restart: usize) {
        let m = restart.max(1);
        if self.restart != m || self.r.len() != n {
            *self = Self::new(n, m);
        }
    }
}

/// Complex GMRES(m) solver with restart.
///
/// Works for any `LinearOperator<Vector = DenseVec<Complex<T>>>`.
pub struct ComplexGmres<T: Scalar> {
    /// Number of Krylov vectors before restart.
    pub restart: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> ComplexGmres<T> {
    /// Create a new solver with the given restart parameter.
    pub fn new(restart: usize) -> Self {
        ComplexGmres { restart: restart.max(1), _phantom: std::marker::PhantomData }
    }

    /// Solve `A x = b` using complex GMRES.
    ///
    /// # Arguments
    /// - `op`      — linear operator (must be square)
    /// - `b`       — right-hand side
    /// - `x`       — solution (in/out, used as initial guess)
    /// - `rtol`    — relative residual tolerance
    /// - `atol`    — absolute residual tolerance (0 to disable)
    /// - `max_iter`— maximum total iterations (across restarts)
    pub fn solve(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<Complex<T>>>,
        b: &DenseVec<Complex<T>>,
        x: &mut DenseVec<Complex<T>>,
        rtol: f64,
        atol: f64,
        max_iter: usize,
    ) -> Result<ComplexGmresResult, SolverError> {
        let mut ws = ComplexGmresWorkspace::new(b.len(), self.restart);
        self.solve_with_workspace(op, b, x, rtol, atol, max_iter, &mut ws)
    }

    /// Solve using caller-supplied scratch space (avoids re-allocation on
    /// repeated solves with the same problem size).
    pub fn solve_with_workspace(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<Complex<T>>>,
        b: &DenseVec<Complex<T>>,
        x: &mut DenseVec<Complex<T>>,
        rtol: f64,
        atol: f64,
        max_iter: usize,
        ws: &mut ComplexGmresWorkspace<T>,
    ) -> Result<ComplexGmresResult, SolverError> {
        let n = b.len();
        if op.nrows() != n || op.ncols() != n || x.len() != n {
            return Err(SolverError::DimensionMismatch {
                op_rows: op.nrows(),
                op_cols: op.ncols(),
                rhs_len: n,
            });
        }
        ws.ensure_shape(n, self.restart);

        let m = self.restart;
        let zero_c = Complex::new(T::zero(), T::zero());
        let norm_b = b.norm2();
        // Treat zero RHS as a trivially converged system.
        if norm_b < T::machine_epsilon() {
            x.fill(zero_c);
            return Ok(ComplexGmresResult {
                iters: 0,
                residual_norm: 0.0,
                converged: true,
                residual_history: vec![0.0],
            });
        }

        let tol = T::from_f64(f64::max(rtol * to_f64(norm_b), atol));
        let mut residual_history: Vec<f64> = Vec::new();
        let mut total_iters = 0usize;

        'outer: loop {
            // r = b - A x
            op.apply(x, &mut ws.r);
            let r_sl = ws.r.as_mut_slice();
            let b_sl = b.as_slice();
            for i in 0..n {
                r_sl[i] = b_sl[i] - r_sl[i];
            }

            let beta = ws.r.norm2();
            residual_history.push(to_f64(beta));

            if beta <= tol || total_iters >= max_iter {
                let converged = beta <= tol;
                return Ok(ComplexGmresResult {
                    iters: total_iters,
                    residual_norm: to_f64(beta),
                    converged,
                    residual_history,
                });
            }

            // v₀ = r / β
            let inv_beta = Complex::new(T::one() / beta, T::zero());
            ws.v[0].copy_from(&ws.r);
            ws.v[0].scale(inv_beta);

            // g = [β, 0, ..., 0]
            ws.g.fill(zero_c);
            ws.g[0] = Complex::new(beta, T::zero());

            // Reset Givens buffers.
            for c in ws.cs.iter_mut() { *c = T::zero(); }
            for s in ws.sn.iter_mut() { *s = zero_c; }

            let mut j_final = 0usize;
            for j in 0..m {
                if total_iters >= max_iter {
                    j_final = j;
                    break;
                }

                // w = A vⱼ
                let vj = ws.v[j].clone();
                op.apply(&vj, &mut ws.w);

                // Modified Gram-Schmidt
                for &mut ref col in ws.h.iter_mut().take(j + 1).collect::<Vec<_>>().iter_mut() {
                    let _ = col; // avoid unused warning below
                }
                for i in 0..=j {
                    // hᵢⱼ = ⟨vᵢ, w⟩
                    let h_ij = ws.v[i].dot(&ws.w);
                    ws.h[j][i] = h_ij;
                    // w -= hᵢⱼ · vᵢ
                    let vi = ws.v[i].clone();
                    ws.w.axpy(-h_ij, &vi);
                }

                // h_{j+1,j} = ‖w‖
                let h_next = ws.w.norm2();
                ws.h[j][j + 1] = Complex::new(h_next, T::zero());

                // Apply previous Givens rotations to column j of H.
                for i in 0..j {
                    let h_i   = ws.h[j][i];
                    let h_ip1 = ws.h[j][i + 1];
                    let c = ws.cs[i];
                    let s = ws.sn[i];
                    ws.h[j][i]     = Complex::new(c, T::zero()) * h_i + s * h_ip1;
                    ws.h[j][i + 1] = -s.conj() * h_i + Complex::new(c, T::zero()) * h_ip1;
                }

                // Compute new Givens rotation to zero h_{j+1,j}.
                let (c_j, s_j) = complex_givens(ws.h[j][j], ws.h[j][j + 1]);
                ws.cs[j] = c_j;
                ws.sn[j] = s_j;

                // Apply to column j.
                let h_jj = ws.h[j][j];
                ws.h[j][j] = Complex::new(c_j, T::zero()) * h_jj + s_j * ws.h[j][j + 1];
                ws.h[j][j + 1] = zero_c; // annihilated

                // Apply rotation to g.
                let g_j   = ws.g[j];
                let g_jp1 = ws.g[j + 1];
                ws.g[j]     = Complex::new(c_j, T::zero()) * g_j + s_j * g_jp1;
                ws.g[j + 1] = -s_j.conj() * g_j + Complex::new(c_j, T::zero()) * g_jp1;

                let r_norm = ws.g[j + 1].norm();
                total_iters += 1;
                j_final = j + 1;

                // Build next basis vector if not converging.
                if r_norm > tol.into() && h_next > T::machine_epsilon() && j + 1 < m {
                    let inv_h = Complex::new(T::one() / h_next, T::zero());
                    let w_clone = ws.w.clone();
                    ws.v[j + 1].copy_from(&w_clone);
                    ws.v[j + 1].scale(inv_h);
                } else {
                    // Converged or Krylov basis collapsed.
                    update_solution(x, &ws.v, &ws.h, &ws.g, j_final, n);
                    let converged = r_norm <= tol.into();
                    residual_history.push(to_f64(r_norm));
                    if converged || total_iters >= max_iter {
                        return Ok(ComplexGmresResult {
                            iters: total_iters,
                            residual_norm: to_f64(r_norm),
                            converged,
                            residual_history,
                        });
                    }
                    continue 'outer;
                }
            }

            // End of inner loop — update x and restart.
            update_solution(x, &ws.v, &ws.h, &ws.g, j_final, n);
        }
    }
}

// ─── Givens rotation for complex scalars ─────────────────────────────────────

/// Compute complex Givens rotation `(c, s)` with real `c ≥ 0` and complex `s`
/// satisfying the **unitary** conditions:
///
/// ```text
/// [ c       s  ] [ a ]   [ R ]
/// [ -conj(s) c ] [ b ] = [ 0 ]
/// ```
///
/// where `R = (a / |a|) * sqrt(|a|² + |b|²)` (same phase as `a`).
///
/// Derivation: from `-conj(s)*a + c*b = 0` → `s = (a/|a|) * conj(b) / r`.
#[inline]
fn complex_givens<T: Scalar>(a: Complex<T>, b: Complex<T>) -> (T, Complex<T>) {
    let norm_a = a.norm();
    let norm_b = b.norm();
    let zero = T::zero();
    let zero_c = Complex::new(zero, zero);
    if norm_b < T::machine_epsilon() {
        // b ≈ 0 → no rotation needed
        return (T::one(), zero_c);
    }
    let r = (norm_a * norm_a + norm_b * norm_b).sqrt();
    if norm_a < T::machine_epsilon() {
        // a ≈ 0 → pure 90° rotation; use -conj(b)/|b| to zero b
        return (zero, -(b.conj() / Complex::new(norm_b, zero)));
    }
    let c = norm_a / r;
    // s = sign(a) * conj(b) / r = (a / |a|) * conj(b) / r
    let sign_a = a / Complex::new(norm_a, zero);
    let s = sign_a * b.conj() * Complex::new(T::one() / r, zero);
    (c, s)
}

// ─── Back-substitution / solution update ────────────────────────────────────

/// Back-substitute the triangular system H y = g (upper-triangular, `k` rows)
/// and update `x += V y`.
fn update_solution<T: Scalar>(
    x: &mut DenseVec<Complex<T>>,
    v: &[DenseVec<Complex<T>>],
    h: &[Vec<Complex<T>>],
    g: &[Complex<T>],
    k: usize,
    _n: usize,
) {
    if k == 0 { return; }
    let zero_c = Complex::new(T::zero(), T::zero());
    let mut y = vec![zero_c; k];

    // Back-substitution: H[0..k][0..k] y = g[0..k]
    for i in (0..k).rev() {
        let mut rhs = g[i];
        for j in (i + 1)..k {
            rhs -= h[j][i] * y[j];
        }
        let h_ii = h[i][i];
        if h_ii.norm() > T::machine_epsilon() {
            y[i] = rhs / h_ii;
        }
    }

    // x += V · y
    for (j, &yj) in y.iter().enumerate() {
        let vj = v[j].clone();
        x.axpy(yj, &vj);
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dense::DenseMatrix;
    use num_complex::Complex;

    type C64 = Complex<f64>;

    fn c(re: f64, im: f64) -> C64 {
        Complex::new(re, im)
    }

    /// Build a simple complex diagonally-dominant matrix:
    /// A = (D + i·ε·I) where D is tridiagonal Laplacian.
    fn complex_laplacian(n: usize, shift: f64) -> DenseMatrix<C64> {
        DenseMatrix::from_fn(n, n, |i, j| {
            if i == j {
                c(2.0 + shift, shift)
            } else if i + 1 == j || j + 1 == i {
                c(-1.0, 0.0)
            } else {
                c(0.0, 0.0)
            }
        })
    }

    #[test]
    fn complex_gmres_identity() {
        let n = 4;
        let a = DenseMatrix::<C64>::from_fn(n, n, |i, j| {
            if i == j { c(1.0, 0.0) } else { c(0.0, 0.0) }
        });
        let b: DenseVec<C64> = (0..n).map(|i| c(i as f64, -(i as f64))).collect::<Vec<_>>().into();
        let mut x: DenseVec<C64> = vec![c(0.0, 0.0); n].into();
        let solver = ComplexGmres::<f64>::new(10);
        let res = solver.solve(&a, &b, &mut x, 1e-10, 0.0, 100).unwrap();
        assert!(res.converged, "should converge on identity: {:?}", res);
        for (xi, bi) in x.as_slice().iter().zip(b.as_slice()) {
            assert!((xi - bi).norm() < 1e-10);
        }
    }

    #[test]
    fn complex_gmres_tridiagonal_real() {
        let n = 20;
        let a = complex_laplacian(n, 0.0);
        let b: DenseVec<C64> = vec![c(1.0, 0.0); n].into();
        let mut x: DenseVec<C64> = vec![c(0.0, 0.0); n].into();
        let solver = ComplexGmres::<f64>::new(20);
        let res = solver.solve(&a, &b, &mut x, 1e-10, 0.0, 500).unwrap();
        assert!(res.converged, "tridiagonal real: {:?}", res);
        // Verify A x ≈ b
        let mut ax: DenseVec<C64> = vec![c(0.0, 0.0); n].into();
        a.apply(&x, &mut ax);
        let err: f64 = ax.as_slice().iter().zip(b.as_slice()).map(|(ai, bi)| (ai - bi).norm()).sum();
        assert!(err < 1e-8, "residual too large: {err}");
    }

    #[test]
    fn complex_gmres_imaginary_shift() {
        // Solve (A + i·σ·I) x = b with σ = 1.0 (strictly non-Hermitian)
        let n = 15;
        let sigma = 1.0;
        let a = complex_laplacian(n, sigma);
        // b with both real and imaginary parts
        let b: DenseVec<C64> = (0..n).map(|i| c((i + 1) as f64, -(i as f64))).collect::<Vec<_>>().into();
        let mut x: DenseVec<C64> = vec![c(0.0, 0.0); n].into();
        let solver = ComplexGmres::<f64>::new(15);
        let res = solver.solve(&a, &b, &mut x, 1e-10, 0.0, 500).unwrap();
        assert!(res.converged, "imaginary shift: {:?}", res);
        // Verify A x ≈ b
        let mut ax: DenseVec<C64> = vec![c(0.0, 0.0); n].into();
        a.apply(&x, &mut ax);
        let err: f64 = ax.as_slice().iter().zip(b.as_slice()).map(|(ai, bi)| (ai - bi).norm()).sum();
        assert!(err < 1e-8, "residual too large: {err}");
    }

    #[test]
    fn complex_gmres_zero_rhs() {
        let n = 5;
        let a = complex_laplacian(n, 0.1);
        let b: DenseVec<C64> = vec![c(0.0, 0.0); n].into();
        let mut x: DenseVec<C64> = vec![c(1.0, 2.0); n].into();
        let solver = ComplexGmres::<f64>::new(5);
        let res = solver.solve(&a, &b, &mut x, 1e-10, 0.0, 100).unwrap();
        assert!(res.converged, "zero RHS: {:?}", res);
        for xi in x.as_slice() {
            assert!(xi.norm() < 1e-10, "expected zero solution, got {xi:?}");
        }
    }

    #[test]
    fn complex_givens_annihilates_b() {
        let a = c(3.0, 4.0);
        let b = c(1.0, -2.0);
        let (c_cos, s_sin) = complex_givens(a, b);
        // Verify: [c, s; -conj(s), c] [a; b] → second component = 0
        let rb = -s_sin.conj() * a + Complex::new(c_cos, 0.0) * b;
        assert!(rb.norm() < 1e-12, "b not annihilated: {rb:?}");
        // First component: R = sign(a) * r, same phase as a
        let ra =  Complex::new(c_cos, 0.0) * a + s_sin * b;
        let r_expected = (a.norm_sqr() + b.norm_sqr()).sqrt();
        assert!((ra.norm() - r_expected).abs() < 1e-12, "|R| mismatch: {} vs {}", ra.norm(), r_expected);
        // c² + |s|² = 1
        let unit = c_cos * c_cos + s_sin.norm_sqr();
        assert!((unit - 1.0).abs() < 1e-12, "not unitary: {unit}");
    }
}
