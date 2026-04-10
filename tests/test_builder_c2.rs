//! Sprint C — SolverBuilder integration tests.
//!
//! Verifies the high-level builder API (C2) for all solver combinations.
//! WASM types are compile-only tested here (no wasm-pack/browser runtime).

mod common;

use linger::{
    builder::{SolverBuilder, SolveMethod, DirectBackend, PrecondChoice, Ordering, solve_auto},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

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

fn relative_residual(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    use linger::core::operator::LinearOperator;
    let mut ax = DenseVec::zeros(a.nrows());
    a.apply(x, &mut ax);
    let r: f64  = ax.as_slice().iter().zip(b.as_slice()).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.as_slice().iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    if nb < 1e-300 { r } else { r / nb }
}

// ─── Direct solve via builder ─────────────────────────────────────────────────

#[test]
fn builder_direct_lu() {
    let n = 10;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = SolverBuilder::new()
        .method(SolveMethod::Direct(DirectBackend::Lu))
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

#[test]
fn builder_direct_cholesky() {
    let n = 12;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());
    let x = SolverBuilder::new()
        .method(SolveMethod::Direct(DirectBackend::Cholesky))
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

#[test]
fn builder_direct_multifrontal() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = SolverBuilder::new()
        .method(SolveMethod::Direct(DirectBackend::Multifrontal))
        .ordering(Ordering::Natural)
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

// ─── Krylov solvers via builder ───────────────────────────────────────────────

#[test]
fn builder_cg_no_precond() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = SolverBuilder::new()
        .method(SolveMethod::Cg)
        .rtol(1e-10)
        .max_iter(100)
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9);
}

#[test]
fn builder_gmres_no_precond() {
    let n = 10;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = SolverBuilder::new()
        .method(SolveMethod::Gmres { restart: 20 })
        .rtol(1e-10)
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9);
}

#[test]
fn builder_bicgstab() {
    let n = 10;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = SolverBuilder::new()
        .method(SolveMethod::BiCgStab)
        .rtol(1e-10)
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9);
}

// ─── Krylov + preconditioners via builder ────────────────────────────────────

#[test]
fn builder_gmres_ilu0() {
    let n = 10;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = SolverBuilder::new()
        .method(SolveMethod::Gmres { restart: 20 })
        .precond(PrecondChoice::Ilu0)
        .rtol(1e-10)
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9);
}

#[test]
fn builder_cg_icc0() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = SolverBuilder::new()
        .method(SolveMethod::Cg)
        .precond(PrecondChoice::Icc0)
        .rtol(1e-10)
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9);
}

#[test]
fn builder_cg_jacobi() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = SolverBuilder::new()
        .method(SolveMethod::Cg)
        .precond(PrecondChoice::Jacobi)
        .rtol(1e-10)
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9);
}

#[test]
fn builder_gmres_direct_precond_multifrontal() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = SolverBuilder::new()
        .method(SolveMethod::Gmres { restart: 20 })
        .precond(PrecondChoice::DirectLu(DirectBackend::Multifrontal))
        .rtol(1e-10)
        .max_iter(10)
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9,
        "residual = {:.3e}", relative_residual(&a, &x, &b));
}

#[test]
fn builder_gmres_direct_precond_lu() {
    let n = 10;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = SolverBuilder::new()
        .method(SolveMethod::Gmres { restart: 20 })
        .precond(PrecondChoice::DirectLu(DirectBackend::Lu))
        .rtol(1e-10)
        .max_iter(10)
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9);
}

// ─── Convenience API ─────────────────────────────────────────────────────────

#[test]
fn solve_auto_spd() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = solve_auto(&a, &b, true).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

#[test]
fn solve_auto_general() {
    let n = 10;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = solve_auto(&a, &b, false).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

// ─── Builder solve_into variant ───────────────────────────────────────────────

#[test]
fn builder_solve_into() {
    let n = 8;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    SolverBuilder::new()
        .method(SolveMethod::Direct(DirectBackend::Multifrontal))
        .solve_into(&a, &b, &mut x).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

// ─── Dimension mismatch error ─────────────────────────────────────────────────

#[test]
fn builder_dimension_mismatch_returns_err() {
    let n = 5;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n + 1]); // wrong size
    let result = SolverBuilder::new()
        .method(SolveMethod::Direct(DirectBackend::Lu))
        .solve(&a, &b);
    assert!(result.is_err());
}

#[test]
fn builder_hpc_ams_preset() {
    let (g, a) = common::make_chain_graph(31, 1e-3);
    let n = a.nrows();
    let x_exact: Vec<f64> = (1..=n)
        .map(|k| (std::f64::consts::PI * k as f64 / (n + 1) as f64).sin())
        .collect();
    let mut b_raw = vec![0.0f64; n];
    a.spmv(&x_exact, &mut b_raw);
    let b = DenseVec::from_vec(b_raw);

    let x = SolverBuilder::new()
        .hpc_ams(std::sync::Arc::new(g))
        .solve(&a, &b)
        .unwrap();

    assert!(relative_residual(&a, &x, &b) < 1e-7);
}

#[test]
fn builder_hpc_ads_preset() {
    let (g, c, a) = common::make_rect_complex(4, 4, 1e-3);
    let b = DenseVec::from_vec(vec![1.0f64; a.nrows()]);

    let x = SolverBuilder::new()
        .hpc_ads(std::sync::Arc::new(c), std::sync::Arc::new(g))
        .max_iter(600)
        .solve(&a, &b)
        .unwrap();

    assert!(x.as_slice().iter().all(|v| v.is_finite()));
    assert!(x.as_slice().iter().any(|v| v.abs() > 1e-14));
}
