//! ex06 — Richardson iteration: first end-to-end KrylovSolver proof.
//!
//! **Purpose**: Validate the complete solver pipeline by implementing the
//! simplest possible iterative method — stationary Richardson iteration —
//! as a `KrylovSolver`.  This confirms that:
//!   - `CsrMatrix<f64>: LinearOperator`
//!   - `KrylovSolver::solve` signature is usable
//!   - `SolverParams` / `SolverResult` types work end-to-end
//!
//! **Algorithm** (Richardson, ω-damped):
//!   x_{k+1} = x_k + ω·(b − A·x_k)
//!   Converges when ω·ρ(A) < 2, i.e. ω < 2/λ_max.
//!   For 1-D Poisson: λ_max ≈ 4·cos²(π/(2(n+1))) < 4, so ω = 0.15 is safe.
//!
//! **HYPRE analog**
//!   `HYPRE_SolverCreate` / a custom relaxation sweep applied as a KSP.
//!
//! **PETSc analog**
//!   `KSPSetType(ksp, KSPRICHARDSON)` with `KSPRichardsonSetScale`.
//!
//! ─────────────────────────────────────────────────────────────────────────────
//! Sprint 2 will replace this with a proper CG / GMRES implementation.
//! The Richardson solver here is intentionally minimal — it exists only to
//! exercise the trait infrastructure, not to be numerically efficient.

use linger::{
    core::{
        operator::LinearOperator,
        preconditioner::Preconditioner,
        solver::{KrylovSolver, SolverParams, SolverResult, VerboseLevel},
    },
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, SolverError, Vector,
};
use std::time::Instant;

// ─── Richardson solver ───────────────────────────────────────────────────────

/// Stationary Richardson iteration:  x ← x + ω·M⁻¹·(b − A·x)
///
/// When `precond = None`, M = I (identity).
/// When `precond = Some(m)`, applies `z = M⁻¹·r` before the update.
struct RichardsonSolver {
    omega: f64,
}

impl KrylovSolver for RichardsonSolver {
    type Vector   = DenseVec<f64>;
    type Operator = CsrMatrix<f64>;

    fn solve(
        &self,
        op:     &CsrMatrix<f64>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<f64>>>,
        b:      &DenseVec<f64>,
        x:      &mut DenseVec<f64>,
        params: &SolverParams,
    ) -> Result<SolverResult, SolverError> {

        // Dimension checks (mirror PETSc's KSPCheckNullSpace / VecCheckSizes)
        if op.nrows() != b.len() || op.ncols() != x.len() {
            return Err(SolverError::DimensionMismatch {
                op_rows: op.nrows(),
                op_cols: op.ncols(),
                rhs_len: b.len(),
            });
        }

        let n       = b.len();
        let norm_b  = b.norm2();
        let norm_b  = if norm_b == 0.0 { 1.0 } else { norm_b };

        // Zero initial guess fast-path (skip first SpMV)
        let all_zero = x.as_slice().iter().all(|&v| v == 0.0);
        let mut r = if all_zero {
            b.clone()  // r = b  (since A·0 = 0)
        } else {
            let mut r = b.zero_like();
            op.apply(x, &mut r);                       // r = A·x
            // r = b - A·x
            for (ri, &bi) in r.as_mut_slice().iter_mut().zip(b.as_slice()) {
                *ri = bi - *ri;
            }
            r
        };

        let mut history = if params.verbose == VerboseLevel::Iterations {
            Some(Vec::new())
        } else {
            None
        };

        let mut z = DenseVec::zeros(n);   // preconditioned residual

        for k in 0..params.max_iter {
            // z = M⁻¹ · r  (or z = r if no preconditioner)
            match precond {
                Some(m) => m.apply_precond(&r, &mut z),
                None    => z.copy_from(&r),
            }

            // x += ω·z
            x.axpy(self.omega, &z);

            // r = b - A·x  (recompute every step for stability)
            let mut ax = b.zero_like();
            op.apply(x, &mut ax);
            for (ri, (&bi, &axi)) in r.as_mut_slice().iter_mut()
                .zip(b.as_slice().iter().zip(ax.as_slice()))
            {
                *ri = bi - axi;
            }

            let res = r.norm2() / norm_b;

            if let Some(ref mut h) = history {
                h.push(res);
            }
            if params.verbose == VerboseLevel::Iterations {
                println!("    iter {:4}  ‖r‖/‖b‖ = {res:.6e}", k + 1);
            }

            // Convergence check
            if res < params.rtol || r.norm2() < params.atol {
                let final_residual = res as f64;
                if params.verbose != VerboseLevel::Silent {
                    println!("  Converged at iter {}  ‖r‖/‖b‖ = {res:.3e}", k + 1);
                }
                return Ok(SolverResult {
                    converged: true,
                    iterations: k + 1,
                    final_residual,
                    history,
                });
            }
        }

        let final_residual = r.norm2() / norm_b;
        Err(SolverError::ConvergenceFailed {
            max_iter: params.max_iter,
            residual: final_residual,
        })
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn sep(title: &str) {
    println!("\n━━━━ {title} ━━━━━");
}

fn poisson_1d(n: usize) -> (CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
    let mut coo = CooMatrix::with_capacity(n, n, 3 * n - 2);
    for i in 0..n {
        if i > 0     { coo.push(i, i - 1, -1.0); }
        coo.push(i, i, 2.0);
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let pi = std::f64::consts::PI;
    let x_exact: Vec<f64> = (0..n)
        .map(|i| (pi * (i + 1) as f64 / (n + 1) as f64).sin())
        .collect();
    let mut b = vec![0.0f64; n];
    a.spmv(&x_exact, &mut b);
    (a, x_exact, b)
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    sep("ex06: Richardson iteration  (PETSc KSPRICHARDSON analog)");

    // ── Test 1: diagonal 4×4 SPD system (converges in ~1 step) ───────────────
    println!("\n  ── Test 1: diagonal 4×4 system ──────────────────────────");
    {
        // A = diag(1, 2, 3, 4),  b = [1, 2, 3, 4]ᵀ  →  x* = [1, 1, 1, 1]ᵀ
        let mut coo: CooMatrix<f64> = CooMatrix::new(4, 4);
        for i in 0..4 { coo.push(i, i, (i + 1) as f64); }
        let a = CsrMatrix::from_coo(&coo);
        let b = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let mut x = DenseVec::zeros(4);

        // ω must be < 2/λ_max = 2/4 = 0.5;  ω=0.2 is safe
        let solver = RichardsonSolver { omega: 0.2 };
        let params = SolverParams {
            rtol: 1e-10, max_iter: 10_000, verbose: VerboseLevel::Silent,
            ..Default::default()
        };
        let result = solver.solve(&a, None, &b, &mut x, &params).unwrap();

        println!("  converged={} iters={}  ‖r‖/‖b‖={:.3e}",
            result.converged, result.iterations, result.final_residual);
        assert!(result.converged);
        for (xi, &expected) in x.as_slice().iter().zip(&[1.0, 1.0, 1.0, 1.0]) {
            assert!((xi - expected).abs() < 1e-8,
                "solution mismatch: got {xi}, expected {expected}");
        }
        println!("  solution x = {:?}", x.as_slice());
        println!("  diagonal system ✓");
    }

    // ── Test 2: 1-D Poisson n=50, unpreconditioned ────────────────────────────
    //
    // Richardson converges slowly (κ(A)≈1000 for n=50) but must converge.
    // PETSc KSPRICHARDSON shows similar convergence rate.
    println!("\n  ── Test 2: 1-D Poisson n=50, ω=0.15 ────────────────────");
    {
        let n = 50;
        let (a, x_exact, b) = poisson_1d(n);
        let b_vec = DenseVec::from_vec(b.clone());
        let mut x = DenseVec::zeros(n);

        // ω=0.15 < 2/λ_max ≈ 2/4 = 0.5  →  converges for any initial guess
        let solver = RichardsonSolver { omega: 0.15 };
        let params = SolverParams {
            rtol: 1e-8, max_iter: 50_000, verbose: VerboseLevel::Silent,
            ..Default::default()
        };

        let t0 = Instant::now();
        let result = solver.solve(&a, None, &b_vec, &mut x, &params).unwrap();
        let elapsed = t0.elapsed();

        println!("  converged={}  iters={}  ‖r‖/‖b‖={:.3e}  time={elapsed:.2?}",
            result.converged, result.iterations, result.final_residual);
        assert!(result.converged,
            "Richardson did not converge (iters={}, final={:.3e})",
            result.iterations, result.final_residual);

        // Check solution accuracy
        let err: f64 = x.as_slice().iter().zip(&x_exact)
            .map(|(&xi, &xe)| (xi - xe).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm_xe: f64 = x_exact.iter().map(|&v| v * v).sum::<f64>().sqrt();
        println!("  ‖x_computed − x_exact‖/‖x_exact‖ = {:.3e}", err / norm_xe);
        assert!(err / norm_xe < 1e-6, "solution error too large: {:.3e}", err / norm_xe);
        println!("  solution accuracy ✓");

        // ── Convergence rate estimate ──────────────────────────────────────
        // Theoretical: ρ = |1 − ω·λ_min|  (dominant mode for Richardson)
        let lam_max = 4.0 * (std::f64::consts::PI * n as f64 / (2*(n+1)) as f64).sin().powi(2);
        let lam_min = 4.0 * (std::f64::consts::PI / (2*(n+1)) as f64).sin().powi(2);
        let rho_theory = (1.0 - 0.15 * lam_min).abs().max((1.0 - 0.15 * lam_max).abs());
        println!("  λ_min={lam_min:.4e}  λ_max={lam_max:.4e}  κ={:.2e}", lam_max / lam_min);
        println!("  theoretical ρ = {rho_theory:.6}  (→ slow convergence, needs preconditioning)");
    }

    // ── Test 3: dimension mismatch error path ─────────────────────────────────
    println!("\n  ── Test 3: dimension mismatch error ─────────────────────");
    {
        let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
        for i in 0..3 { coo.push(i, i, 1.0); }
        let a = CsrMatrix::from_coo(&coo);
        let b = DenseVec::from_vec(vec![1.0, 2.0]);   // wrong size!
        let mut x = DenseVec::zeros(3);
        let solver  = RichardsonSolver { omega: 0.5 };
        let params  = SolverParams::default();
        match solver.solve(&a, None, &b, &mut x, &params) {
            Err(SolverError::DimensionMismatch { .. }) => println!("  DimensionMismatch ✓"),
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    // ── Test 4: convergence-failure error path ────────────────────────────────
    println!("\n  ── Test 4: convergence-failure (max_iter=5) ─────────────");
    {
        let n = 50;
        let (a, _, b) = poisson_1d(n);
        let b_vec   = DenseVec::from_vec(b);
        let mut x   = DenseVec::zeros(n);
        let solver  = RichardsonSolver { omega: 0.15 };
        let params  = SolverParams { rtol: 1e-8, max_iter: 5, ..Default::default() };
        match solver.solve(&a, None, &b_vec, &mut x, &params) {
            Err(SolverError::ConvergenceFailed { max_iter, residual }) => {
                println!("  ConvergenceFailed: max_iter={max_iter}  residual={residual:.3e}  ✓");
            }
            other => panic!("expected ConvergenceFailed, got {other:?}"),
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!();
    println!("  ╔═══════════════════════════════════════════════════════╗");
    println!("  ║  Full KrylovSolver pipeline validated.                ║");
    println!("  ║  Sprint 2 will replace RichardsonSolver with CG,     ║");
    println!("  ║  GMRES, and BiCGSTAB for O(n) convergence.           ║");
    println!("  ╚═══════════════════════════════════════════════════════╝");

    println!("\n  OK\n");
}
