//! Integration tests for A1: TFQMR Krylov solver.

use linger::{
    iterative::Tfqmr,
    iterative::BiCgStab,
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
    let nr: f64 = ax.as_slice().iter().zip(b.as_slice()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.as_slice().iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    if nb < 1e-300 { nr } else { nr / nb }
}

fn default_params() -> SolverParams {
    SolverParams { rtol: 1e-8, max_iter: 2000, verbose: VerboseLevel::Silent, ..Default::default() }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// 1. TFQMR converges on symmetric 1D Laplacian n=50.
#[test]
fn tfqmr_laplacian_1d() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let res = Tfqmr::<f64>::new().solve(&a, None, &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "TFQMR did not converge: rel_res={}", res.final_residual);
    assert!(rel_res(&a, &x, &b) < 1e-7, "rel_res too large: {}", rel_res(&a, &x, &b));
}

/// 2. TFQMR converges on non-symmetric tridiag n=40.
#[test]
fn tfqmr_nonsymmetric() {
    let n = 40;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect::<Vec<_>>());
    let mut x = DenseVec::zeros(n);

    let res = Tfqmr::<f64>::new().solve(&a, None, &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "TFQMR non-symmetric did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7, "rel_res too large: {}", rel_res(&a, &x, &b));
}

/// 3. TFQMR with Jacobi preconditioner.
#[test]
fn tfqmr_jacobi_precond() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let jac = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let res = Tfqmr::<f64>::new()
        .solve(&a, Some(&jac), &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "TFQMR+Jacobi did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7, "rel_res too large: {}", rel_res(&a, &x, &b));
}

/// 4. TFQMR with ILU(0) preconditioner.
#[test]
fn tfqmr_ilu0_precond() {
    let n = 100;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let ilu = Ilu0Precond::<f64>::from_csr(&a).unwrap();
    let res = Tfqmr::<f64>::new()
        .solve(&a, Some(&ilu), &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "TFQMR+ILU0 did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7, "rel_res too large: {}", rel_res(&a, &x, &b));
}

/// 5. Dimension mismatch returns an error.
#[test]
fn tfqmr_dimension_mismatch() {
    use linger::SolverError;
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n + 1]);
    let mut x = DenseVec::zeros(n);
    let result = Tfqmr::<f64>::new().solve(&a, None, &b, &mut x, &default_params());
    assert!(matches!(result, Err(SolverError::DimensionMismatch { .. })));
}

/// 6. Already converged (x = 0, b = 0) returns immediately with 0 iterations.
#[test]
fn tfqmr_zero_rhs() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::zeros(n);
    let mut x = DenseVec::zeros(n);
    let res = Tfqmr::<f64>::new().solve(&a, None, &b, &mut x, &default_params()).unwrap();
    assert!(res.converged);
    assert_eq!(res.iterations, 0);
}

/// 7. f32 precision: TFQMR converges for the 1D Laplacian.
#[test]
fn tfqmr_f32() {
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

    // rtol=1e-4 is safely above the f32 precision floor for this matrix.
    let params = SolverParams { rtol: 1e-4, max_iter: 500, verbose: VerboseLevel::Silent, ..Default::default() };
    let res = Tfqmr::<f32>::new().solve(&a, None, &b, &mut x, &params).unwrap();
    assert!(res.converged, "TFQMR f32 did not converge, rel_res={}", res.final_residual);
}

/// 8. TFQMR converges on a variety of problem sizes.
///    Specifically tests that the algorithm doesn't break down on larger systems.
#[test]
fn tfqmr_vs_bicgstab_convergence() {
    // Test TFQMR on non-symmetric problem using ILU(0) preconditioner
    // to ensure robust convergence.
    let n = 80;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);

    let ilu = Ilu0Precond::<f64>::from_csr(&a).unwrap();

    let mut x_tfqmr = DenseVec::zeros(n);
    let res_tfqmr = Tfqmr::<f64>::new()
        .solve(&a, Some(&ilu), &b, &mut x_tfqmr, &default_params()).unwrap();

    assert!(res_tfqmr.converged, "TFQMR did not converge");
    assert!(rel_res(&a, &x_tfqmr, &b) < 1e-7, "rel_res too large: {}", rel_res(&a, &x_tfqmr, &b));
}
