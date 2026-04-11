//! ex08 - sparse direct solvers and direct-solver preconditioning.

use linger::{
    direct::{DirectOptions, DirectSolver, DirectSolverPrecond, SparseCholesky, SparseLu, ordering::OrderingMethod},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, Gmres, KrylovSolver, LinearOperator, SolverParams, VerboseLevel,
};

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0 {
            coo.push(i, i - 1, -1.0);
        }
        if i + 1 < n {
            coo.push(i, i + 1, -1.0);
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn nonsymmetric_tridiag(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 4.0);
        if i > 0 {
            coo.push(i, i - 1, -1.0);
        }
        if i + 1 < n {
            coo.push(i, i + 1, -2.0);
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn relative_residual(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let mut ax = DenseVec::zeros(a.nrows());
    a.apply(x, &mut ax);
    let norm_r = ax
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&lhs, &rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm_b = b.as_slice().iter().map(|&v| v.powi(2)).sum::<f64>().sqrt();
    if norm_b == 0.0 { norm_r } else { norm_r / norm_b }
}

fn main() {
    println!("ex08: sparse direct solvers");

    let n_spd = 32;
    let a_spd = laplacian_1d(n_spd);
    let b_spd = DenseVec::from_vec(vec![1.0_f64; n_spd]);
    let mut x_spd = DenseVec::zeros(n_spd);

    let mut cholesky = SparseCholesky::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Rcm,
        ..Default::default()
    });
    cholesky.factor(&a_spd).unwrap();
    cholesky.solve(&b_spd, &mut x_spd).unwrap();
    let chol_res = relative_residual(&a_spd, &x_spd, &b_spd);

    println!(
        "  SparseCholesky: n={} rel_res={:.3e}",
        n_spd,
        chol_res
    );
    assert!(chol_res < 1e-10);

    let n_ns = 20;
    let a_ns = nonsymmetric_tridiag(n_ns);
    let b_ns = DenseVec::from_vec((1..=n_ns).map(|i| i as f64).collect());
    let mut x_lu = DenseVec::zeros(n_ns);

    let mut lu = SparseLu::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Natural,
        ..Default::default()
    });
    lu.factor(&a_ns).unwrap();
    lu.solve(&b_ns, &mut x_lu).unwrap();
    let lu_res = relative_residual(&a_ns, &x_lu, &b_ns);

    println!("  SparseLu:       n={} rel_res={:.3e}", n_ns, lu_res);
    assert!(lu_res < 1e-10);

    let precond = DirectSolverPrecond::new(
        SparseLu::<f64>::new(DirectOptions {
            ordering: OrderingMethod::Natural,
            ..Default::default()
        }),
        &a_ns,
    )
    .unwrap();

    let gmres = Gmres::new(20);
    let params = SolverParams {
        rtol: 1e-10,
        max_iter: 100,
        verbose: VerboseLevel::Silent,
        ..Default::default()
    };
    let mut x_gmres = DenseVec::zeros(n_ns);
    let gmres_res = gmres
        .solve(&a_ns, Some(&precond), &b_ns, &mut x_gmres, &params)
        .unwrap();
    let gmres_rel = relative_residual(&a_ns, &x_gmres, &b_ns);

    println!(
        "  GMRES + LU preconditioner: iters={} rel_res={:.3e}",
        gmres_res.iterations,
        gmres_rel
    );

    assert!(gmres_res.converged);
    assert!(gmres_rel < 1e-10);

    println!("  OK");
}