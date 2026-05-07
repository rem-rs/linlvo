use linger::{iterative::{Idrs, BiCgStab}, sparse::{CooMatrix, CsrMatrix}, DenseVec, KrylovSolver, SolverParams, VerboseLevel};
fn main() {
    let n = 200usize;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i-1, -1.0); }
        if i+1 < n   { coo.push(i, i+1, -1.0); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let params = SolverParams { rtol: 1e-8, max_iter: 5000, verbose: VerboseLevel::Silent, ..Default::default() };
    
    let mut x1 = DenseVec::zeros(n);
    let r1 = Idrs::<f64>::new(1).solve(&a, None, &b, &mut x1, &params).unwrap();
    println!("IDR(1): converged={} iters={} final={:.3e}", r1.converged, r1.iterations, r1.final_residual);
    
    let mut x4 = DenseVec::zeros(n);
    let r4 = Idrs::<f64>::new(4).solve(&a, None, &b, &mut x4, &params).unwrap();
    println!("IDR(4): converged={} iters={} final={:.3e}", r4.converged, r4.iterations, r4.final_residual);
    
    let mut xb = DenseVec::zeros(n);
    let rb = BiCgStab::<f64>::default().solve(&a, None, &b, &mut xb, &params).unwrap();
    println!("BiCGSTAB: converged={} iters={} final={:.3e}", rb.converged, rb.iterations, rb.final_residual);
}
