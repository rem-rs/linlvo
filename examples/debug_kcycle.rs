use linger::{
    amg::{AmgConfig, AmgHierarchy, CycleType},
    sparse::CsrMatrix,
    DenseVec, LinearOperator, Vector,
};
use linger::sparse::CooMatrix;

fn make_poisson_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i-1, -1.0); }
        if i+1 < n   { coo.push(i, i+1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn rel_res(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let n = b.len();
    let mut ax = DenseVec::zeros(n);
    a.apply(x, &mut ax);
    let r: f64 = ax.as_slice().iter().zip(b.as_slice()).map(|(&ai, &bi)| (ai-bi).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.as_slice().iter().map(|&v| v*v).sum::<f64>().sqrt();
    if nb > 0.0 { r/nb } else { r }
}

fn main() {
    let n = 100;
    let a = make_poisson_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let config = linger::amg::AmgConfig { coarse_threshold: 4, ..Default::default() };
    let hier = AmgHierarchy::build(a.clone(), config.clone());

    println!("n_levels = {}", hier.n_levels());
    for (i, lv) in hier.levels.iter().enumerate() {
        println!("  level {i}: nrows={}", lv.a.nrows());
    }

    // V-cycle
    let mut xv = DenseVec::zeros(n);
    hier.apply_cycle(&b, &mut xv, CycleType::V);
    println!("V-cycle 1 rel_res = {:.4e}", rel_res(&a, &xv, &b));
    hier.apply_cycle(&b, &mut xv, CycleType::V);
    println!("V-cycle 2 rel_res = {:.4e}", rel_res(&a, &xv, &b));

    // K-cycle
    let hier2 = AmgHierarchy::build(a.clone(), config);
    let mut xk = DenseVec::zeros(n);
    hier2.apply_cycle(&b, &mut xk, CycleType::K { inner_iters: 2 });
    println!("K-cycle 1 rel_res = {:.4e}", rel_res(&a, &xk, &b));
    hier2.apply_cycle(&b, &mut xk, CycleType::K { inner_iters: 2 });
    println!("K-cycle 2 rel_res = {:.4e}", rel_res(&a, &xk, &b));
}

fn test_inner() {
    let n = 10;
    let a = make_poisson_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let config = linger::amg::AmgConfig { coarse_threshold: 4, ..Default::default() };
    let hier = AmgHierarchy::build(a.clone(), config);
    println!("inner test n_levels={}", hier.n_levels());
    for (i,lv) in hier.levels.iter().enumerate() { println!("  lv{i} nrows={}", lv.a.nrows()); }
    // One V-cycle from scratch
    let mut xv = DenseVec::zeros(n);
    hier.apply_cycle(&b, &mut xv, CycleType::V);
    println!("V-1 rr={:.4e}", rel_res(&a, &xv, &b));
    // One K-cycle
    let config2 = linger::amg::AmgConfig { coarse_threshold: 4, ..Default::default() };
    let hier2 = AmgHierarchy::build(a.clone(), config2);
    let mut xk = DenseVec::zeros(n);
    hier2.apply_cycle(&b, &mut xk, CycleType::K { inner_iters: 1 });
    println!("K-1 rr={:.4e}", rel_res(&a, &xk, &b));
}
