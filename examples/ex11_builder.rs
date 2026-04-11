//! ex11 - high-level SolverBuilder workflows.

use linger::{
    builder::{solve_auto, DirectBackend, Ordering, PrecondChoice, SolveMethod, SolverBuilder},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, LinearOperator,
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
    let num = ax
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&lhs, &rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let den = b.as_slice().iter().map(|&v| v.powi(2)).sum::<f64>().sqrt();
    if den == 0.0 { num } else { num / den }
}

fn main() {
    println!("ex11: SolverBuilder workflows");

    let n_spd = 32;
    let a_spd = laplacian_1d(n_spd);
    let b_spd = DenseVec::from_vec(vec![1.0_f64; n_spd]);

    let x_direct = SolverBuilder::new()
        .method(SolveMethod::Direct(DirectBackend::Cholesky))
        .ordering(Ordering::Rcm)
        .solve(&a_spd, &b_spd)
        .unwrap();
    let direct_rel = relative_residual(&a_spd, &x_direct, &b_spd);

    let x_cg = SolverBuilder::new()
        .method(SolveMethod::Cg)
        .precond(PrecondChoice::Jacobi)
        .rtol(1e-10)
        .max_iter(500)
        .solve(&a_spd, &b_spd)
        .unwrap();
    let cg_rel = relative_residual(&a_spd, &x_cg, &b_spd);

    let x_auto = solve_auto(&a_spd, &b_spd, true).unwrap();
    let auto_rel = relative_residual(&a_spd, &x_auto, &b_spd);

    println!("  SPD direct Cholesky: rel_res={:.3e}", direct_rel);
    println!("  SPD CG + Jacobi:     rel_res={:.3e}", cg_rel);
    println!("  SPD solve_auto:      rel_res={:.3e}", auto_rel);

    assert!(direct_rel < 1e-10);
    assert!(cg_rel < 1e-9);
    assert!(auto_rel < 1e-10);

    let n_ns = 24;
    let a_ns = nonsymmetric_tridiag(n_ns);
    let b_ns = DenseVec::from_vec((1..=n_ns).map(|i| i as f64).collect());

    let x_gmres = SolverBuilder::new()
        .method(SolveMethod::Gmres { restart: 20 })
        .precond(PrecondChoice::Ilu0)
        .rtol(1e-10)
        .max_iter(100)
        .solve(&a_ns, &b_ns)
        .unwrap();
    let gmres_rel = relative_residual(&a_ns, &x_gmres, &b_ns);

    let x_direct_pre = SolverBuilder::new()
        .method(SolveMethod::Gmres { restart: 20 })
        .precond(PrecondChoice::DirectLu(DirectBackend::Lu))
        .ordering(Ordering::Natural)
        .rtol(1e-10)
        .max_iter(10)
        .solve(&a_ns, &b_ns)
        .unwrap();
    let direct_pre_rel = relative_residual(&a_ns, &x_direct_pre, &b_ns);

    let x_auto_general = solve_auto(&a_ns, &b_ns, false).unwrap();
    let auto_general_rel = relative_residual(&a_ns, &x_auto_general, &b_ns);

    println!("  nonsym GMRES + ILU0: rel_res={:.3e}", gmres_rel);
    println!("  nonsym GMRES + LU:   rel_res={:.3e}", direct_pre_rel);
    println!("  nonsym solve_auto:   rel_res={:.3e}", auto_general_rel);

    assert!(gmres_rel < 1e-9);
    assert!(direct_pre_rel < 1e-10);
    assert!(auto_general_rel < 1e-10);

    println!("  OK");
}