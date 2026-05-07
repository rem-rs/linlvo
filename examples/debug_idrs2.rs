use linger::{iterative::Idrs, sparse::{CooMatrix, CsrMatrix}, DenseVec, KrylovSolver, SolverParams, VerboseLevel};

fn main() {
    let n = 50usize;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i-1, -1.0); }
        if i+1 < n   { coo.push(i, i+1, -1.0); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![1.0f64; n]);

    let params = SolverParams { rtol: 1e-12, max_iter: 500, verbose: VerboseLevel::Iterations, ..Default::default() };
    let mut x = DenseVec::zeros(n);
    let res = Idrs::<f64>::new(4).solve(&a, None, &b, &mut x, &params).unwrap();
    println!("IDR(4) n={}: converged={} iters={} final={:.3e}", n, res.converged, res.iterations, res.final_residual);
    // Print last 10 history values
    let h = &res.residual_history;
    let start = if h.len() > 20 { h.len() - 20 } else { 0 };
    println!("last history: {:?}", &h[start..]);
}
