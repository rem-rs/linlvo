//! Integration tests for F4: IDR(s) Krylov solver.

use linger::{
    iterative::Idrs,
    precond::{JacobiPrecond, Ilu0Precond},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, SolverParams, VerboseLevel,
};

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn nonsymmetric_tridiag(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 4.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -2.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn rel_res(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    use linger::LinearOperator;
    let n = a.nrows();
    let mut ax = DenseVec::zeros(n);
    a.apply(x, &mut ax);
    let nr: f64 = ax.as_slice().iter().zip(b.as_slice()).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.as_slice().iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    if nb < 1e-300 { nr } else { nr / nb }
}

fn default_params() -> SolverParams {
    SolverParams { rtol: 1e-8, max_iter: 2000, verbose: VerboseLevel::Silent, ..Default::default() }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// 1. IDR(4) converges on symmetric 1D Laplacian.
#[test]
fn idrs4_laplacian_1d() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let res = Idrs::<f64>::new(4).solve(&a, None, &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "IDR(4) did not converge: rel_res={}", res.final_residual);
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 2. IDR(1) converges (BiCGSTAB-like).
#[test]
fn idrs1_laplacian_1d() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let res = Idrs::<f64>::new(1).solve(&a, None, &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "IDR(1) did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 3. IDR(8) converges.
#[test]
fn idrs8_laplacian_1d() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let res = Idrs::<f64>::new(8).solve(&a, None, &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "IDR(8) did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 4. IDR(4) converges on a non-symmetric system.
#[test]
fn idrs4_nonsymmetric() {
    let n = 40;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect::<Vec<_>>());
    let mut x = DenseVec::zeros(n);

    let res = Idrs::<f64>::new(4).solve(&a, None, &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "IDR(4) non-symmetric did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 5. IDR(4) with Jacobi preconditioner.
#[test]
fn idrs4_jacobi_precond() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let jac = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let res = Idrs::<f64>::new(4)
        .solve(&a, Some(&jac), &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "IDR(4)+Jacobi did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 6. IDR(4) with ILU(0) preconditioner.
#[test]
fn idrs4_ilu0_precond() {
    let n = 100;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let ilu = Ilu0Precond::<f64>::from_csr(&a).unwrap();
    let res = Idrs::<f64>::new(4)
        .solve(&a, Some(&ilu), &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "IDR(4)+ILU0 did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 7. Dimension mismatch returns an error.
#[test]
fn idrs_dimension_mismatch() {
    use linger::SolverError;
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n + 1]);
    let mut x = DenseVec::zeros(n);
    let result = Idrs::<f64>::new(4).solve(&a, None, &b, &mut x, &default_params());
    assert!(matches!(result, Err(SolverError::DimensionMismatch { .. })));
}

/// 8. Already converged (x = 0, b = 0) returns immediately.
#[test]
fn idrs_zero_rhs() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::zeros(n);
    let mut x = DenseVec::zeros(n);
    let res = Idrs::<f64>::new(4).solve(&a, None, &b, &mut x, &default_params()).unwrap();
    assert!(res.converged);
    assert_eq!(res.iterations, 0);
}

/// 9. IDR(4) uses fewer iterations than IDR(1) for large s (trend).
#[test]
fn idrs_higher_s_fewer_iters() {
    // n=100 is a safe size where the random shadow space doesn't degenerate.
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);

    let mut x1 = DenseVec::zeros(n);
    let res1 = Idrs::<f64>::new(1).solve(&a, None, &b, &mut x1, &default_params()).unwrap();

    let mut x4 = DenseVec::zeros(n);
    let res4 = Idrs::<f64>::new(4).solve(&a, None, &b, &mut x4, &default_params()).unwrap();

    assert!(res1.converged && res4.converged);
    // IDR(4) should use fewer matvecs than IDR(1) on this problem.
    assert!(res4.iterations < res1.iterations,
        "IDR(4) iters={} not < IDR(1) iters={}", res4.iterations, res1.iterations);
}

/// 10. f32 precision: IDR(4) converges for the 1D Laplacian.
#[test]
fn idrs4_f32() {
    let n = 30;
    let mut coo = CooMatrix::<f32>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0f32);
        if i > 0     { coo.push(i, i-1, -1.0f32); }
        if i+1 < n   { coo.push(i, i+1, -1.0f32); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![1.0f32; n]);
    let mut x = DenseVec::zeros(n);

    // rtol=1e-4: f32 machine eps ~1.2e-7, cond(A)~365, so effective floor ~4e-5.
    // 1e-4 is safely above the f32 precision floor for this matrix.
    let params = SolverParams { rtol: 1e-4, max_iter: 500, verbose: VerboseLevel::Silent, ..Default::default() };
    let res = Idrs::<f32>::new(4).solve(&a, None, &b, &mut x, &params).unwrap();
    assert!(res.converged, "IDR(4) f32 did not converge");
}

/// 11. IDR(s) with_max_restarts(0) converges normally (no restart needed for good seed).
#[test]
fn idrs_max_restarts_zero_still_converges() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    let res = Idrs::<f64>::new(4)
        .with_max_restarts(0)
        .solve(&a, None, &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "IDR(4) no-restart did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 12. with_max_restarts builder method is accessible and increases convergence
///     robustness (smoke test).
#[test]
fn idrs_with_max_restarts_accessible() {
    let n = 50;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    let res = Idrs::<f64>::new(4)
        .with_max_restarts(5)
        .solve(&a, None, &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "IDR(4) max_restarts=5 did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}
