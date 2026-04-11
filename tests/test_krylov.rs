//! Sprint 2 — Krylov solver tests.
//!
//! Covers CG, GMRES, BiCGSTAB, and MINRES on canonical test problems.

mod common;

use linger::{
    iterative::{BiCgStab, ConjugateGradient, Gmres, Minres},
    precond::{Ilu0Precond, JacobiPrecond, SsorPrecond},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, SolverError, SolverParams, VerboseLevel,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn diag_spd(diag: &[f64]) -> CsrMatrix<f64> {
    let n = diag.len();
    let mut coo = CooMatrix::new(n, n);
    for (i, &v) in diag.iter().enumerate() { coo.push(i, i, v); }
    CsrMatrix::from_coo(&coo)
}

fn default_params(rtol: f64, max_iter: usize) -> SolverParams {
    SolverParams { rtol, max_iter, verbose: VerboseLevel::Silent, ..Default::default() }
}

fn solution_error(x: &[f64], x_exact: &[f64]) -> f64 {
    let err: f64 = x.iter().zip(x_exact).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let norm: f64 = x_exact.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 { err / norm } else { err }
}

// ─── CG tests ────────────────────────────────────────────────────────────────

#[test]
fn cg_diagonal_4x4() {
    // A = diag(1,2,3,4),  b = [1,2,3,4]  →  x* = [1,1,1,1]
    let a = diag_spd(&[1.0, 2.0, 3.0, 4.0]);
    let b = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut x = DenseVec::zeros(4);
    let cg = ConjugateGradient::<f64>::default();
    let params = default_params(1e-12, 100);
    let res = cg.solve(&a, None, &b, &mut x, &params).unwrap();
    assert!(res.converged, "CG did not converge on diagonal 4×4");
    for &xi in x.as_slice() {
        assert!((xi - 1.0).abs() < 1e-10, "CG solution error: {xi}");
    }
}

#[test]
fn cg_poisson_1d_unpreconditioned() {
    let n = 50;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let cg = ConjugateGradient::<f64>::default();
    let params = default_params(1e-10, 500);
    let res = cg.solve(&a, None, &b_vec, &mut x, &params).unwrap();
    assert!(res.converged, "CG didn't converge on 1D Poisson n=50; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn cg_poisson_1d_jacobi_precond() {
    let n = 100;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let jac = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let cg = ConjugateGradient::<f64>::default();
    let params = default_params(1e-10, 300);
    let res = cg.solve(&a, Some(&jac), &b_vec, &mut x, &params).unwrap();
    assert!(res.converged, "PCG (Jacobi) didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn cg_poisson_1d_ilu0_precond() {
    let n = 100;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let ilu = Ilu0Precond::<f64>::from_csr(&a).unwrap();
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let cg = ConjugateGradient::<f64>::default();
    let params = default_params(1e-10, 200);
    let res = cg.solve(&a, Some(&ilu), &b_vec, &mut x, &params).unwrap();
    assert!(res.converged, "PCG (ILU0) didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    // ILU(0) is exact on tridiagonal → should converge in 1 iteration
    assert!(res.iterations <= 5, "ILU(0) on tridiagonal should need ≤5 iters, got {}", res.iterations);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-10);
}

#[test]
fn cg_dimension_mismatch() {
    let a = diag_spd(&[1.0, 2.0, 3.0]);
    let b = DenseVec::from_vec(vec![1.0, 2.0]);  // wrong size
    let mut x = DenseVec::zeros(3);
    let cg = ConjugateGradient::<f64>::default();
    let params = default_params(1e-10, 100);
    match cg.solve(&a, None, &b, &mut x, &params) {
        Err(SolverError::DimensionMismatch { .. }) => {}
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

#[test]
fn cg_convergence_failure() {
    // Use a constant RHS (not an eigenvector) so CG needs many iterations.
    let n = 50;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    // b = [1,1,...,1] — has contributions from all eigenvectors, needs many CG steps.
    let b_vec = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    let cg = ConjugateGradient::<f64>::default();
    let params = default_params(1e-12, 2);  // intentionally too few iterations
    match cg.solve(&a, None, &b_vec, &mut x, &params) {
        Err(SolverError::ConvergenceFailed { .. }) => {}
        Ok(r) if !r.converged => {}
        other => panic!("expected failure, got {other:?}"),
    }
}

#[test]
fn cg_zero_operator_reports_breakdown() {
    // A = 0 has no unique solution for non-zero b; CG should report numerical breakdown.
    let n = 6;
    let a = CsrMatrix::from_coo(&CooMatrix::<f64>::new(n, n));
    let b = DenseVec::from_vec(vec![1.0; n]);
    let mut x = DenseVec::zeros(n);
    let cg = ConjugateGradient::<f64>::default();
    let params = default_params(1e-12, 20);

    match cg.solve(&a, None, &b, &mut x, &params) {
        Err(SolverError::NumericalBreakdown { .. }) => {}
        other => panic!("expected NumericalBreakdown on zero operator, got {other:?}"),
    }
}

// ─── GMRES tests ─────────────────────────────────────────────────────────────

#[test]
fn gmres_diagonal_4x4() {
    let a = diag_spd(&[1.0, 2.0, 3.0, 4.0]);
    let b = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut x = DenseVec::zeros(4);
    let gmres = Gmres::<f64>::new(10);
    let params = default_params(1e-12, 50);
    let res = gmres.solve(&a, None, &b, &mut x, &params).unwrap();
    assert!(res.converged, "GMRES did not converge on diagonal 4×4");
    for &xi in x.as_slice() {
        assert!((xi - 1.0).abs() < 1e-9, "GMRES solution error: {xi}");
    }
}

#[test]
fn gmres_poisson_1d() {
    let n = 50;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let gmres = Gmres::<f64>::new(30);
    let params = default_params(1e-10, 500);
    let res = gmres.solve(&a, None, &b_vec, &mut x, &params).unwrap();
    assert!(res.converged, "GMRES didn't converge on 1D Poisson n=50; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn gmres_nonsymmetric_convdiff() {
    let n = 30;
    let (a, x_exact, b) = common::make_nonsymmetric_convdiff::<f64>(n, 5.0);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let gmres = Gmres::<f64>::new(40);
    let params = default_params(1e-10, 500);
    let res = gmres.solve(&a, None, &b_vec, &mut x, &params).unwrap();
    assert!(res.converged, "GMRES didn't converge on conv-diff; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-7);
}

#[test]
fn gmres_nonfinite_rhs_reports_breakdown() {
    // Non-finite RHS should be rejected as numerical breakdown.
    let a = diag_spd(&[1.0, 2.0, 3.0]);
    let b = DenseVec::from_vec(vec![1.0, f64::NAN, 3.0]);
    let mut x = DenseVec::zeros(3);
    let gmres = Gmres::<f64>::new(10);
    let params = default_params(1e-10, 30);

    match gmres.solve(&a, None, &b, &mut x, &params) {
        Err(SolverError::NumericalBreakdown { .. }) => {}
        other => panic!("expected NumericalBreakdown for non-finite RHS, got {other:?}"),
    }
}

// ─── BiCGSTAB tests ──────────────────────────────────────────────────────────

#[test]
fn bicgstab_diagonal_4x4() {
    let a = diag_spd(&[1.0, 2.0, 3.0, 4.0]);
    let b = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut x = DenseVec::zeros(4);
    let solver = BiCgStab::<f64>::new();
    let params = default_params(1e-12, 50);
    let res = solver.solve(&a, None, &b, &mut x, &params).unwrap();
    assert!(res.converged, "BiCGSTAB did not converge on diagonal 4×4");
    for &xi in x.as_slice() {
        assert!((xi - 1.0).abs() < 1e-9, "BiCGSTAB solution error: {xi}");
    }
}

#[test]
fn bicgstab_poisson_1d() {
    let n = 50;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let solver = BiCgStab::<f64>::new();
    let params = default_params(1e-10, 500);
    let res = solver.solve(&a, None, &b_vec, &mut x, &params).unwrap();
    assert!(res.converged, "BiCGSTAB didn't converge on 1D Poisson n=50; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn bicgstab_nonsymmetric_convdiff() {
    // Use n=20, Pe=1 — small enough that BiCGSTAB converges well before Krylov exhaustion.
    let n = 20;
    let (a, x_exact, b) = common::make_nonsymmetric_convdiff::<f64>(n, 1.0);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let solver = BiCgStab::<f64>::new();
    let params = default_params(1e-10, 500);
    let res = solver.solve(&a, None, &b_vec, &mut x, &params).unwrap();
    assert!(res.converged, "BiCGSTAB didn't converge on conv-diff Pe=1; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-7);
}

// ─── MINRES tests ─────────────────────────────────────────────────────────────

#[test]
fn minres_diagonal_4x4() {
    let a = diag_spd(&[1.0, 2.0, 3.0, 4.0]);
    let b = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut x = DenseVec::zeros(4);
    let solver = Minres::<f64>::new();
    let params = default_params(1e-12, 100);
    let res = solver.solve(&a, None, &b, &mut x, &params).unwrap();
    assert!(res.converged, "MINRES did not converge on diagonal 4×4");
    for &xi in x.as_slice() {
        assert!((xi - 1.0).abs() < 1e-8, "MINRES solution error: {xi}");
    }
}

#[test]
fn minres_poisson_1d() {
    let n = 50;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let solver = Minres::<f64>::new();
    let params = default_params(1e-10, 500);
    let res = solver.solve(&a, None, &b_vec, &mut x, &params).unwrap();
    assert!(res.converged, "MINRES didn't converge on 1D Poisson n=50; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

// ─── Preconditioned CG accuracy comparison ────────────────────────────────────

#[test]
fn pcg_ssor_vs_unpreconditioned_fewer_iters() {
    // SSOR-preconditioned CG should converge in fewer iterations than unpreconditioned.
    // Use large n so κ≈O(n²) is high enough for SSOR to provide clear improvement.
    let n = 200;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(vec![1.0f64; n]);
    let cg = ConjugateGradient::<f64>::default();
    let params = default_params(1e-10, 2000);

    let mut x1 = DenseVec::zeros(n);
    let r1 = cg.solve(&a, None, &b_vec, &mut x1, &params).unwrap();

    let ssor = SsorPrecond::<f64>::from_csr(&a, 1.0).unwrap();
    let mut x2 = DenseVec::zeros(n);
    let r2 = cg.solve(&a, Some(&ssor), &b_vec, &mut x2, &params).unwrap();

    assert!(r1.converged && r2.converged);
    assert!(r2.iterations < r1.iterations,
        "SSOR-PCG ({} iters) should be faster than unpreconditioned CG ({} iters)",
        r2.iterations, r1.iterations);
}

// ─── residual_history tests (L1) ─────────────────────────────────────────────

#[test]
fn residual_history_always_populated_cg() {
    let n = 20;
    let (a, _, b_vec_raw) = common::make_poisson_1d::<f64>(n);
    let b = DenseVec::from_vec(b_vec_raw);
    let cg = ConjugateGradient::<f64>::default();
    let params = default_params(1e-10, 500);
    let mut x = DenseVec::zeros(n);
    let result = cg.solve(&a, None, &b, &mut x, &params).unwrap();
    assert!(result.converged);
    // history must have exactly iterations entries
    assert_eq!(result.residual_history.len(), result.iterations);
    // residuals must be non-increasing (CG on SPD, with tolerance)
    for w in result.residual_history.windows(2) {
        assert!(w[1] <= w[0] * 1.01, "residual should be non-increasing: {} -> {}", w[0], w[1]);
    }
    // last entry should match final_residual
    assert!((result.residual_history.last().unwrap() - result.final_residual).abs() < 1e-14);
}

#[test]
fn residual_history_always_populated_gmres() {
    let n = 20;
    let (a, _, b_vec_raw) = common::make_poisson_1d::<f64>(n);
    let b = DenseVec::from_vec(b_vec_raw);
    let gmres = Gmres::<f64>::default();
    let params = default_params(1e-10, 500);
    let mut x = DenseVec::zeros(n);
    let result = gmres.solve(&a, None, &b, &mut x, &params).unwrap();
    assert!(result.converged);
    assert!(!result.residual_history.is_empty());
    assert_eq!(result.residual_history.len(), result.iterations);
}

#[test]
fn residual_history_always_populated_bicgstab() {
    let n = 20;
    let (a, _, b_vec_raw) = common::make_poisson_1d::<f64>(n);
    let b = DenseVec::from_vec(b_vec_raw);
    let solver = BiCgStab::<f64>::default();
    let params = default_params(1e-10, 500);
    let mut x = DenseVec::zeros(n);
    let result = solver.solve(&a, None, &b, &mut x, &params).unwrap();
    assert!(result.converged);
    assert!(!result.residual_history.is_empty());
}

#[test]
fn residual_history_independent_of_verbose() {
    // residual_history must be populated even when verbose = Silent
    let n = 15;
    let (a, _, b_vec_raw) = common::make_poisson_1d::<f64>(n);
    let b = DenseVec::from_vec(b_vec_raw);
    let cg = ConjugateGradient::<f64>::default();
    let params_silent = SolverParams { verbose: VerboseLevel::Silent, ..default_params(1e-10, 200) };
    let params_verbose = SolverParams { verbose: VerboseLevel::Iterations, ..default_params(1e-10, 200) };

    let mut x1 = DenseVec::zeros(n);
    let r1 = cg.solve(&a, None, &b, &mut x1, &params_silent).unwrap();
    let mut x2 = DenseVec::zeros(n);
    let r2 = cg.solve(&a, None, &b, &mut x2, &params_verbose).unwrap();

    // Both should have residual_history
    assert!(!r1.residual_history.is_empty());
    assert!(!r2.residual_history.is_empty());
    // history (Option) should be None for silent, Some for verbose
    assert!(r1.history.is_none());
    assert!(r2.history.is_some());
    // residual_history should match history when verbose
    assert_eq!(r1.residual_history, r2.residual_history);
}
