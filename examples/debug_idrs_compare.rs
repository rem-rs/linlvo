use linger::{iterative::Idrs, sparse::{CooMatrix, CsrMatrix}, DenseVec, KrylovSolver, SolverParams, VerboseLevel};
fn main() {
    for n in [50, 80, 100] {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0     { coo.push(i, i-1, -1.0); }
            if i+1 < n   { coo.push(i, i+1, -1.0); }
        }
        let a = CsrMatrix::from_coo(&coo);
        let b = DenseVec::from_vec(vec![1.0f64; n]);
        let params = SolverParams { rtol: 1e-8, max_iter: 2000, verbose: VerboseLevel::Silent, ..Default::default() };
        let mut x1 = DenseVec::zeros(n);
        let r1 = Idrs::<f64>::new(1).solve(&a, None, &b, &mut x1, &params).unwrap();
        let mut x4 = DenseVec::zeros(n);
        let r4 = Idrs::<f64>::new(4).solve(&a, None, &b, &mut x4, &params).unwrap();
        println!("n={}: IDR(1) iters={} conv={}, IDR(4) iters={} conv={}", n, r1.iterations, r1.converged, r4.iterations, r4.converged);
    }
}
