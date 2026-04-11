//! ex09 - eigenvalue solvers on the 1-D Laplacian spectrum.

use linger::{
    sparse::{CooMatrix, CsrMatrix},
    EigenParams, EigenSolver, EigenWhich, Lobpcg, PowerIter,
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

fn laplacian_eigenvalue(n: usize, k: usize) -> f64 {
    let theta = k as f64 * std::f64::consts::PI / (n + 1) as f64;
    2.0 - 2.0 * theta.cos()
}

fn main() {
    let n = 20;
    let a = laplacian_1d(n);

    println!("ex09: eigen solvers");
    println!("  system: 1-D Laplacian, n={n}, nnz={}", a.nnz());

    let mut power_params = EigenParams::new(1, EigenWhich::LargestMagnitude);
    power_params.tol = 1e-9;
    power_params.max_iter = 3_000;
    let power = PowerIter::default().solve(&a, &power_params).unwrap();
    let largest_exact = laplacian_eigenvalue(n, n);

    println!(
        "  PowerIter largest: lambda={:.8} residual={:.3e}",
        power.eigenvalues[0],
        power.residuals[0]
    );
    assert!((power.eigenvalues[0] - largest_exact).abs() < 1e-7);

    let mut lobpcg_params = EigenParams::new(3, EigenWhich::SmallestAlgebraic);
    lobpcg_params.tol = 1e-7;
    lobpcg_params.max_iter = 3_000;
    let lobpcg = Lobpcg::default().solve(&a, &lobpcg_params).unwrap();

    let mut smallest = lobpcg.eigenvalues.clone();
    smallest.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    let expected: Vec<f64> = (1..=3).map(|k| laplacian_eigenvalue(n, k)).collect();

    println!("  LOBPCG smallest 3:");
    for (idx, (&got, &want)) in smallest.iter().zip(&expected).enumerate() {
        println!(
            "    lambda[{}] = {:.8}  exact = {:.8}  abs_err = {:.3e}",
            idx,
            got,
            want,
            (got - want).abs()
        );
        assert!((got - want).abs() < 1e-5);
    }

    println!("  OK");
}