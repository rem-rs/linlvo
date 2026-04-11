//! ex05 - preconditioners and preconditioned CG on a 1-D Poisson system.

use linger::{
    sparse::{CooMatrix, CsrMatrix},
    ConjugateGradient, DenseVec, Ilu0Precond, JacobiPrecond, KrylovSolver,
    Preconditioner, SolverParams, VerboseLevel,
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

fn manufactured_system(a: &CsrMatrix<f64>) -> (Vec<f64>, DenseVec<f64>) {
    let n = a.nrows();
    let pi = std::f64::consts::PI;
    let x_exact: Vec<f64> = (0..n)
        .map(|i| (pi * (i + 1) as f64 / (n + 1) as f64).sin())
        .collect();

    let mut b = vec![0.0_f64; n];
    a.spmv(&x_exact, &mut b);
    (x_exact, DenseVec::from_vec(b))
}

fn rel_l2_error(x: &DenseVec<f64>, x_exact: &[f64]) -> f64 {
    let num = x
        .as_slice()
        .iter()
        .zip(x_exact)
        .map(|(&got, &want)| (got - want).powi(2))
        .sum::<f64>()
        .sqrt();
    let den = x_exact.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if den == 0.0 { num } else { num / den }
}

fn main() {
    let n = 100;
    let a = laplacian_1d(n);
    let (x_exact, b) = manufactured_system(&a);

    println!("ex05: preconditioners + CG");
    println!("  system: 1-D Poisson, n={n}, nnz={}", a.nnz());

    let jacobi = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let probe = DenseVec::from_vec(vec![1.0_f64; n]);
    let mut scaled = DenseVec::zeros(n);
    jacobi.apply_precond(&probe, &mut scaled);
    let jacobi_head = &scaled.as_slice()[0..4.min(n)];
    println!("  Jacobi on all-ones RHS -> first entries = {:?}", jacobi_head);
    assert!(scaled.as_slice().iter().all(|&v| (v - 0.5).abs() < 1e-14));

    let cg = ConjugateGradient::<f64>::default();
    let params = SolverParams {
        rtol: 1e-10,
        max_iter: 2_000,
        verbose: VerboseLevel::Silent,
        ..Default::default()
    };

    let mut x_plain = DenseVec::zeros(n);
    let plain = cg.solve(&a, None, &b, &mut x_plain, &params).unwrap();
    let plain_err = rel_l2_error(&x_plain, &x_exact);

    let ilu0 = Ilu0Precond::<f64>::from_csr(&a).unwrap();
    let mut x_ilu0 = DenseVec::zeros(n);
    let precond = cg.solve(&a, Some(&ilu0), &b, &mut x_ilu0, &params).unwrap();
    let precond_err = rel_l2_error(&x_ilu0, &x_exact);

    println!(
        "  plain CG:    converged={} iters={} rel_res={:.3e} rel_err={:.3e}",
        plain.converged,
        plain.iterations,
        plain.final_residual,
        plain_err
    );
    println!(
        "  ILU0-PCG:    converged={} iters={} rel_res={:.3e} rel_err={:.3e}",
        precond.converged,
        precond.iterations,
        precond.final_residual,
        precond_err
    );

    assert!(plain.converged);
    assert!(precond.converged);
    assert!(plain_err < 1e-8);
    assert!(precond_err < 1e-10);
    assert!(precond.iterations <= plain.iterations);

    println!("  OK");
}