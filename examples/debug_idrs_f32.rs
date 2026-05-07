use linger::{iterative::Idrs, sparse::{CooMatrix, CsrMatrix}, DenseVec, KrylovSolver, SolverParams, VerboseLevel};
fn main() {
    let n = 30usize;
    let mut coo = CooMatrix::<f32>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0f32);
        if i > 0     { coo.push(i, i-1, -1.0f32); }
        if i+1 < n   { coo.push(i, i+1, -1.0f32); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![1.0f32; n]);
    let mut x = DenseVec::zeros(n);
    let params = SolverParams { rtol: 1e-5, max_iter: 2000, verbose: VerboseLevel::Iterations, ..Default::default() };
    let res = Idrs::<f32>::new(4).solve(&a, None, &b, &mut x, &params).unwrap();
    println!("converged={} iters={} final={:.3e}", res.converged, res.iterations, res.final_residual);
    let h = &res.residual_history;
    let tail = if h.len() > 10 { &h[h.len()-10..] } else { h.as_slice() };
    println!("last residuals: {:?}", tail);
}
