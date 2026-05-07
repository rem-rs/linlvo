use linger::{iterative::Idrs, sparse::{CooMatrix, CsrMatrix}, DenseVec, KrylovSolver, SolverParams, VerboseLevel};
fn main() {
    let n = 20usize;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i-1, -1.0); }
        if i+1 < n   { coo.push(i, i+1, -1.0); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let params = SolverParams { rtol: 1e-8, max_iter: 200, verbose: VerboseLevel::Iterations, ..Default::default() };
    
    let mut x = DenseVec::zeros(n);
    let res = Idrs::<f64>::new(1).solve(&a, None, &b, &mut x, &params).unwrap();
    println!("IDR(1) n={}: converged={} iters={}", n, res.converged, res.iterations);
    let h = &res.residual_history;
    for (i, v) in h.iter().enumerate() { println!("  iter {}: {:.6e}", i, v); }
}
