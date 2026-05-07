use linger::{iterative::{Idrs, BiCgStab}, sparse::{CooMatrix, CsrMatrix}, DenseVec, KrylovSolver, SolverParams, VerboseLevel};
fn main() {
    let n = 5usize;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i-1, -1.0); }
        if i+1 < n   { coo.push(i, i+1, -1.0); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![1.0f64; n]);

    let params = SolverParams { rtol: 1e-8, max_iter: 100, verbose: VerboseLevel::Iterations, ..Default::default() };

    let mut x1 = DenseVec::zeros(n);
    let res1 = Idrs::<f64>::new(2).solve(&a, None, &b, &mut x1, &params).unwrap();
    println!("IDR(2) n={}: converged={} iters={} final={:.3e}", n, res1.converged, res1.iterations, res1.final_residual);
    println!("history: {:?}", &res1.residual_history);
}
