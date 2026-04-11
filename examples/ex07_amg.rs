//! ex07 - AMG-preconditioned CG on a 2-D Poisson problem.

use linger::{
    sparse::{CooMatrix, CsrMatrix},
    AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, ConjugateGradient,
    DenseVec, KrylovSolver, SolverParams, VerboseLevel,
};

fn poisson_2d(nx: usize, ny: usize) -> CsrMatrix<f64> {
    let n = nx * ny;
    let dof = |i: usize, j: usize| i * ny + j;

    let mut coo = CooMatrix::with_capacity(n, n, 5 * n);
    for i in 0..nx {
        for j in 0..ny {
            let row = dof(i, j);
            coo.push(row, row, 4.0);
            if i > 0 {
                coo.push(row, dof(i - 1, j), -1.0);
            }
            if i + 1 < nx {
                coo.push(row, dof(i + 1, j), -1.0);
            }
            if j > 0 {
                coo.push(row, dof(i, j - 1), -1.0);
            }
            if j + 1 < ny {
                coo.push(row, dof(i, j + 1), -1.0);
            }
        }
    }

    CsrMatrix::from_coo(&coo)
}

fn relative_residual(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let mut ax = vec![0.0_f64; a.nrows()];
    a.spmv(x.as_slice(), &mut ax);
    let num = ax
        .iter()
        .zip(b.as_slice())
        .map(|(&got, &want)| (got - want).powi(2))
        .sum::<f64>()
        .sqrt();
    let den = b.as_slice().iter().map(|&v| v * v).sum::<f64>().sqrt();
    if den == 0.0 { num } else { num / den }
}

fn main() {
    let (nx, ny) = (32, 32);
    let a = poisson_2d(nx, ny);
    let n = nx * ny;
    let b = DenseVec::from_vec(vec![1.0_f64; n]);

    println!("ex07: AMG-preconditioned CG");
    println!("  system: 2-D Poisson on {nx}x{ny}, n={n}, nnz={}", a.nnz());
    println!("  rhs: constant one-vector to excite multiple error modes");

    let cg = ConjugateGradient::<f64>::default();
    let params = SolverParams {
        rtol: 1e-9,
        max_iter: 4_000,
        verbose: VerboseLevel::Silent,
        ..Default::default()
    };

    let mut x_plain = DenseVec::zeros(n);
    let plain = cg.solve(&a, None, &b, &mut x_plain, &params).unwrap();
    let plain_rel = relative_residual(&a, &x_plain, &b);

    let sa_hier = AmgHierarchy::build(
        a.clone(),
        AmgConfig {
            coarse_threshold: 4,
            ..Default::default()
        },
    );
    let sa = AmgPrecond::new(sa_hier);
    let mut x_sa = DenseVec::zeros(n);
    let sa_res = cg.solve(&a, Some(&sa), &b, &mut x_sa, &params).unwrap();
    let sa_rel = relative_residual(&a, &x_sa, &b);

    let rs_hier = AmgHierarchy::build(
        a.clone(),
        AmgConfig {
            strategy: CoarsenStrategy::RugeStüben,
            coarse_threshold: 4,
            ..Default::default()
        },
    );
    let rs = AmgPrecond::new(rs_hier);
    let mut x_rs = DenseVec::zeros(n);
    let rs_res = cg.solve(&a, Some(&rs), &b, &mut x_rs, &params).unwrap();
    let rs_rel = relative_residual(&a, &x_rs, &b);

    println!(
        "  plain CG:  iters={} rel_res={:.3e} rel_err={:.3e}",
        plain.iterations,
        plain.final_residual,
        plain_rel
    );
    println!(
        "  SA-AMG:    iters={} rel_res={:.3e} rel_err={:.3e}",
        sa_res.iterations,
        sa_res.final_residual,
        sa_rel
    );
    println!(
        "  RS-AMG:    iters={} rel_res={:.3e} rel_err={:.3e}",
        rs_res.iterations,
        rs_res.final_residual,
        rs_rel
    );

    assert!(plain.converged);
    assert!(sa_res.converged);
    assert!(rs_res.converged);
    assert!(plain_rel < 1e-8);
    assert!(sa_rel < 1e-8);
    assert!(rs_rel < 1e-8);
    assert!(sa_res.iterations < plain.iterations);
    assert!(rs_res.iterations < plain.iterations);

    println!("  OK");
}