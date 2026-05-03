//! Integration tests: Matrix Market IO + SolverBuilder end-to-end flows.

use linger::{
    builder::{BuilderPrecondReport, SolveMethod, SolverBuilder},
    sparse::{read_matrix_market_str, write_matrix_market_str},
    DenseVec, LinearOperator,
};

fn relative_residual(a: &linger::sparse::CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let mut ax = DenseVec::zeros(a.nrows());
    a.apply(x, &mut ax);
    let num = ax
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&lhs, &rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let den = b.as_slice().iter().map(|&v| v.powi(2)).sum::<f64>().sqrt();
    if den < 1e-300 { num } else { num / den }
}

#[test]
fn matrix_market_spd_direct_builder() {
    let mtx = "%%MatrixMarket matrix coordinate real symmetric
5 5 9
1 1 2.0
2 1 -1.0
2 2 2.0
3 2 -1.0
3 3 2.0
4 3 -1.0
4 4 2.0
5 4 -1.0
5 5 2.0
";

    let a = read_matrix_market_str(mtx).unwrap();
    let b = DenseVec::from_vec(vec![1.0_f64; a.nrows()]);

    let x = SolverBuilder::new()
        .method(SolveMethod::Direct(linger::builder::DirectBackend::Cholesky))
        .solve(&a, &b)
        .unwrap();

    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

#[test]
fn matrix_market_general_gmres_with_report() {
    let mtx = "%%MatrixMarket matrix coordinate real general
6 6 16
1 1 4.0
1 2 -2.0
2 1 -1.0
2 2 4.0
2 3 -2.0
3 2 -1.0
3 3 4.0
3 4 -2.0
4 3 -1.0
4 4 4.0
4 5 -2.0
5 4 -1.0
5 5 4.0
5 6 -2.0
6 5 -1.0
6 6 4.0
";

    let a = read_matrix_market_str(mtx).unwrap();
    let b = DenseVec::from_vec((1..=a.nrows()).map(|i| i as f64).collect());

    let (x, report) = SolverBuilder::new()
        .method(SolveMethod::Gmres { restart: 20 })
        .precond(linger::builder::PrecondChoice::Ilu0)
        .rtol(1e-10)
        .max_iter(100)
        .solve_with_report(&a, &b)
        .unwrap();

    assert!(relative_residual(&a, &x, &b) < 1e-9);

    match report.method {
        SolveMethod::Gmres { restart } => assert_eq!(restart, 20),
        _ => panic!("expected GMRES solve method in report"),
    }
    match report.precond {
        BuilderPrecondReport::Ilu0 => {}
        _ => panic!("expected ILU0 preconditioner report"),
    }
    let krylov = report.krylov.expect("expected Krylov report for GMRES");
    assert!(krylov.converged);
}

#[test]
fn matrix_market_roundtrip_then_builder_solve() {
    let input = "%%MatrixMarket matrix coordinate real symmetric
7 7 13
1 1 2.0
2 1 -1.0
2 2 2.0
3 2 -1.0
3 3 2.0
4 3 -1.0
4 4 2.0
5 4 -1.0
5 5 2.0
6 5 -1.0
6 6 2.0
7 6 -1.0
7 7 2.0
";

    let a = read_matrix_market_str(input).unwrap();
    let encoded = write_matrix_market_str(&a).unwrap();
    let b = read_matrix_market_str(&encoded).unwrap();

    let rhs = DenseVec::from_vec(vec![1.0_f64; b.nrows()]);
    let x = SolverBuilder::new()
        .method(SolveMethod::Cg)
        .precond(linger::builder::PrecondChoice::Jacobi)
        .rtol(1e-10)
        .max_iter(200)
        .solve(&b, &rhs)
        .unwrap();

    assert!(relative_residual(&b, &x, &rhs) < 1e-9);
}