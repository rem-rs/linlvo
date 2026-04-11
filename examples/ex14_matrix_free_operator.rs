//! ex14 - custom matrix-free LinearOperator with eigen solvers.

use linger::{
    core::operator::LinearOperator,
    DenseVec, EigenParams, EigenSolver, EigenWhich, InverseIter, PowerIter,
};

struct Laplacian1dOp {
    n: usize,
}

impl LinearOperator for Laplacian1dOp {
    type Vector = DenseVec<f64>;

    fn apply(&self, x: &DenseVec<f64>, y: &mut DenseVec<f64>) {
        let xs = x.as_slice();
        let ys = y.as_mut_slice();
        for i in 0..self.n {
            let mut sum = 2.0 * xs[i];
            if i > 0 {
                sum -= xs[i - 1];
            }
            if i + 1 < self.n {
                sum -= xs[i + 1];
            }
            ys[i] = sum;
        }
    }

    fn nrows(&self) -> usize { self.n }
    fn ncols(&self) -> usize { self.n }
}

fn laplacian_eigenvalue(n: usize, k: usize) -> f64 {
    let theta = k as f64 * std::f64::consts::PI / (n + 1) as f64;
    2.0 - 2.0 * theta.cos()
}

fn residual_norm(op: &Laplacian1dOp, x: &DenseVec<f64>, lambda: f64) -> f64 {
    let mut ax = DenseVec::zeros(op.nrows());
    op.apply(x, &mut ax);
    ax.as_slice()
        .iter()
        .zip(x.as_slice())
        .map(|(&axi, &xi)| (axi - lambda * xi).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn main() {
    let n = 20;
    let op = Laplacian1dOp { n };

    println!("ex14: matrix-free LinearOperator");
    println!("  operator: 1-D Laplacian, n={} (no CSR assembled)", n);

    let mut probe_out = DenseVec::zeros(n);
    let probe_in = DenseVec::from_vec(vec![1.0_f64; n]);
    op.apply(&probe_in, &mut probe_out);
    println!(
        "  apply(all-ones): first entries={:?}",
        &probe_out.as_slice()[0..4.min(n)]
    );
    assert!((probe_out[0] - 1.0).abs() < 1e-14);
    assert!((probe_out[n - 1] - 1.0).abs() < 1e-14);

    let mut power_params = EigenParams::new(1, EigenWhich::LargestMagnitude);
    power_params.tol = 1e-9;
    power_params.max_iter = 3_000;
    let power = PowerIter::default().solve(&op, &power_params).unwrap();
    let lambda_max = power.eigenvalues[0];
    let lambda_max_exact = laplacian_eigenvalue(n, n);
    let power_res = residual_norm(&op, &power.eigenvectors[0], lambda_max);
    println!(
        "  PowerIter largest: lambda={:.8} exact={:.8} residual={:.3e}",
        lambda_max,
        lambda_max_exact,
        power_res
    );
    assert!((lambda_max - lambda_max_exact).abs() < 1e-7);

    let mut inverse_params = EigenParams::new(1, EigenWhich::SmallestMagnitude);
    inverse_params.tol = 1e-8;
    inverse_params.max_iter = 200;
    let smallest = InverseIter::<f64>::default().solve(&op, &inverse_params).unwrap();
    let lambda_min = smallest.eigenvalues[0];
    let lambda_min_exact = laplacian_eigenvalue(n, 1);
    let inverse_res = residual_norm(&op, &smallest.eigenvectors[0], lambda_min);
    println!(
        "  InverseIter smallest: lambda={:.8} exact={:.8} residual={:.3e}",
        lambda_min,
        lambda_min_exact,
        inverse_res
    );
    assert!((lambda_min - lambda_min_exact).abs() < 1e-6);

    println!("  OK");
}