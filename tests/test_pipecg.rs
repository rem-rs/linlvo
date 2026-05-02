//! Integration tests for the PipeCg pipelined conjugate-gradient solver.
//!
//! Verifies:
//! - Convergence matches standard CG on SPD problems.
//! - Preconditioned PIPECG converges with Jacobi and ILU(0).
//! - Result is the same whether preconditioner is present or not.

mod common;

use linger::{
    iterative::{ConjugateGradient, PipeCg},
    precond::{Ilu0Precond, JacobiPrecond},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, SolverParams, VerboseLevel,
};

fn default_params(rtol: f64, max_iter: usize) -> SolverParams {
    SolverParams { rtol, max_iter, verbose: VerboseLevel::Silent, ..Default::default() }
}

fn rel_error(x: &[f64], x_exact: &[f64]) -> f64 {
    let err: f64 = x.iter().zip(x_exact).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let nrm: f64 = x_exact.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if nrm > 0.0 { err / nrm } else { err }
}

// ─── 1-D Poisson ─────────────────────────────────────────────────────────────

#[test]
fn pipecg_poisson_1d_n20_converges() {
    let (a, x_exact, b_vec) = common::make_poisson_1d::<f64>(20);
    let b = DenseVec::from_vec(b_vec);
    let mut x = DenseVec::zeros(20);
    let r = PipeCg::new()
        .solve(&a, None, &b, &mut x, &default_params(1e-10, 200))
        .expect("PipeCg solve failed");
    assert!(r.converged, "PipeCg did not converge: iter={}, res={:.3e}", r.iterations, r.final_residual);
    assert!(rel_error(x.as_slice(), &x_exact) < 1e-9,
        "solution error = {:.3e}", rel_error(x.as_slice(), &x_exact));
}

// ─── Agrees with standard CG (Poisson problem) ──────────────────────────────

#[test]
fn pipecg_matches_cg_on_poisson_1d() {
    // Both CG and PipeCg should reach comparable accuracy on 1D Poisson.
    let (a, x_exact, b_vec) = common::make_poisson_1d::<f64>(30);
    let b = DenseVec::from_vec(b_vec);
    let params = default_params(1e-10, 300);

    let mut x_cg = DenseVec::zeros(30);
    let mut x_pcg = DenseVec::zeros(30);

    ConjugateGradient::new(10).solve(&a, None, &b, &mut x_cg, &params).unwrap();
    PipeCg::new().solve(&a, None, &b, &mut x_pcg, &params).unwrap();

    assert!(rel_error(x_cg.as_slice(),  &x_exact) < 1e-9, "CG error: {:.3e}", rel_error(x_cg.as_slice(), &x_exact));
    assert!(rel_error(x_pcg.as_slice(), &x_exact) < 1e-9, "PipeCg error: {:.3e}", rel_error(x_pcg.as_slice(), &x_exact));
}

// ─── Preconditioned PIPECG ────────────────────────────────────────────────────

#[test]
fn pipecg_jacobi_precond_poisson_1d_n30() {
    let (a, x_exact, b_vec) = common::make_poisson_1d::<f64>(30);
    let b = DenseVec::from_vec(b_vec);
    let pc = JacobiPrecond::from_csr(&a).expect("Jacobi setup");
    let mut x = DenseVec::zeros(30);
    let r = PipeCg::new()
        .solve(&a, Some(&pc), &b, &mut x, &default_params(1e-10, 100))
        .expect("PipeCg + Jacobi failed");
    assert!(r.converged, "PipeCg+Jacobi did not converge: res={:.3e}", r.final_residual);
    assert!(rel_error(x.as_slice(), &x_exact) < 1e-9);
}

#[test]
fn pipecg_ilu0_precond_poisson_1d_n40() {
    let (a, x_exact, b_vec) = common::make_poisson_1d::<f64>(40);
    let b = DenseVec::from_vec(b_vec);
    let pc = Ilu0Precond::from_csr(&a).expect("ILU(0) setup");
    let mut x = DenseVec::zeros(40);
    let r = PipeCg::new()
        .solve(&a, Some(&pc), &b, &mut x, &default_params(1e-10, 100))
        .expect("PipeCg + ILU(0) failed");
    assert!(r.converged, "PipeCg+ILU(0) did not converge: res={:.3e}", r.final_residual);
    assert!(rel_error(x.as_slice(), &x_exact) < 1e-9);
}

// ─── Dimension mismatch error ─────────────────────────────────────────────────

#[test]
fn pipecg_dimension_mismatch_returns_error() {
    let mut coo = CooMatrix::new(3, 3);
    coo.push(0, 0, 1.0); coo.push(1, 1, 1.0); coo.push(2, 2, 1.0);
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![1.0; 3]);
    let mut x = DenseVec::zeros(4); // wrong size
    let err = PipeCg::new().solve(&a, None, &b, &mut x, &default_params(1e-10, 10));
    assert!(err.is_err(), "expected DimensionMismatch error");
}

// ─── f32 scalar ───────────────────────────────────────────────────────────────

#[test]
fn pipecg_f32_poisson_1d_converges() {
    // ILU(0) is exact for tridiagonal, so the preconditioned system has κ≈1
    // and PIPECG converges in very few iterations even in f32.
    let (a, x_exact, b_vec) = common::make_poisson_1d::<f32>(20);
    let b = DenseVec::from_vec(b_vec);
    let ilu = Ilu0Precond::from_csr(&a).expect("ILU0 f32");
    let params = SolverParams { rtol: 1e-5, max_iter: 50, verbose: VerboseLevel::Silent, ..Default::default() };
    let mut x = DenseVec::zeros(20);
    let r = PipeCg::new().solve(&a, Some(&ilu), &b, &mut x, &params).expect("f32 PipeCg");
    assert!(r.converged, "f32 PipeCg did not converge");
    let err: f32 = x.as_slice().iter().zip(&x_exact).map(|(&a, &b)| (a-b).powi(2)).sum::<f32>().sqrt();
    assert!(err < 1e-3, "f32 solution error = {err:.3e}");
}

// ─── Already-converged initial guess ─────────────────────────────────────────

#[test]
fn pipecg_already_converged_zero_iters() {
    let (a, x_exact, _b_vec) = common::make_poisson_1d::<f64>(10);
    let b = {
        let mut bv = DenseVec::zeros(10);
        use linger::core::operator::LinearOperator;
        a.apply(&DenseVec::from_vec(x_exact.clone()), &mut bv);
        bv
    };
    // Provide the exact solution as initial guess.
    let mut x = DenseVec::from_vec(x_exact);
    let r = PipeCg::new()
        .solve(&a, None, &b, &mut x, &default_params(1e-12, 100))
        .unwrap();
    assert!(r.converged);
    assert_eq!(r.iterations, 0, "should return 0 iterations when initial residual is below tolerance");
}
