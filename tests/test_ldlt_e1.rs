//! Integration tests for SparseLdlt — sparse LDLᵀ factorisation (E1).

use linger::{
    direct::{DirectOptions, DirectSolver, SparseLdlt},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, OrderingMethod,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn laplacian_2d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut coo = CooMatrix::new(nn, nn);
    for i in 0..n {
        for j in 0..n {
            let id = i * n + j;
            coo.push(id, id, 4.0);
            if j > 0 { coo.push(id, id - 1, -1.0); coo.push(id - 1, id, -1.0); }
            if i > 0 { coo.push(id, id - n, -1.0); coo.push(id - n, id, -1.0); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut r = vec![0.0f64; a.nrows()];
    a.spmv(x, &mut r);
    r.iter().zip(b).map(|(ri, bi)| (ri - bi).powi(2)).sum::<f64>().sqrt()
}

// ─── SPD tests ────────────────────────────────────────────────────────────────

#[test]
fn ldlt_spd_poisson_1d_n50() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut ldlt = SparseLdlt::<f64>::default();
    ldlt.factor(&a).unwrap();
    let mut x = DenseVec::zeros(n);
    ldlt.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-10);
    assert!(ldlt.is_positive_definite());
}

#[test]
fn ldlt_spd_poisson_2d_n6() {
    let a = laplacian_2d(6); // 36×36
    let n = a.nrows();
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut ldlt = SparseLdlt::<f64>::default();
    ldlt.factor(&a).unwrap();
    let mut x = DenseVec::zeros(n);
    ldlt.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-9);
}

// ─── Indefinite tests ─────────────────────────────────────────────────────────

#[test]
fn ldlt_indefinite_diagonal() {
    // A = diag(3, -2, 5, -1)
    let n = 4;
    let diag = [3.0, -2.0, 5.0, -1.0];
    let mut coo = CooMatrix::new(n, n);
    for (i, &d) in diag.iter().enumerate() { coo.push(i, i, d); }
    let a = CsrMatrix::from_coo(&coo);
    let b_vals = [3.0, -4.0, 10.0, -2.0];
    // x_exact = b ./ diag = [1, 2, 2, 2]
    let b = DenseVec::from_vec(b_vals.to_vec());
    let mut ldlt = SparseLdlt::<f64>::default();
    ldlt.factor(&a).unwrap();
    let mut x = DenseVec::zeros(n);
    ldlt.solve(&b, &mut x).unwrap();
    let xs = x.as_slice();
    assert!((xs[0] - 1.0).abs() < 1e-13);
    assert!((xs[1] - 2.0).abs() < 1e-13);
    assert!((xs[2] - 2.0).abs() < 1e-13);
    assert!((xs[3] - 2.0).abs() < 1e-13);
    assert!(!ldlt.is_positive_definite());
}

#[test]
fn ldlt_indefinite_2x2_saddle_point() {
    // Saddle-point-like: [[A, B],[Bᵀ, 0]] style, simple 2×2
    // A = [[1, 1], [1, -1]] — indefinite
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 0, 1.0); coo.push(0, 1, 1.0);
    coo.push(1, 0, 1.0); coo.push(1, 1, -1.0);
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![2.0, 0.0]);
    let mut ldlt = SparseLdlt::<f64>::default();
    ldlt.factor(&a).unwrap();
    let mut x = DenseVec::zeros(2);
    ldlt.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-13);
    assert!(!ldlt.is_positive_definite());
}

// ─── Ordering tests ──────────────────────────────────────────────────────────

#[test]
fn ldlt_natural_ordering() {
    let n = 20;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let opts = DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() };
    let mut ldlt = SparseLdlt::<f64>::new(opts);
    ldlt.factor(&a).unwrap();
    let mut x = DenseVec::zeros(n);
    ldlt.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-10);
}

#[test]
fn ldlt_rcm_ordering() {
    let a = laplacian_2d(5); // 25×25
    let n = a.nrows();
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let opts = DirectOptions { ordering: OrderingMethod::Rcm, ..Default::default() };
    let mut ldlt = SparseLdlt::<f64>::new(opts);
    ldlt.factor(&a).unwrap();
    let mut x = DenseVec::zeros(n);
    ldlt.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-9);
}

#[test]
fn ldlt_nd_ordering() {
    let a = laplacian_2d(5); // 25×25
    let n = a.nrows();
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let opts = DirectOptions { ordering: OrderingMethod::NodeNd, ..Default::default() };
    let mut ldlt = SparseLdlt::<f64>::new(opts);
    ldlt.factor(&a).unwrap();
    let mut x = DenseVec::zeros(n);
    ldlt.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-9);
}

// ─── reuse_symbolic test ──────────────────────────────────────────────────────

#[test]
fn ldlt_reuse_symbolic() {
    let n = 15;
    let a = laplacian_1d(n);
    let opts = DirectOptions { reuse_symbolic: true, ..Default::default() };
    let mut ldlt = SparseLdlt::<f64>::new(opts);
    ldlt.factor(&a).unwrap();

    // Second factorization reuses the symbolic phase.
    ldlt.factorize(&a).unwrap();
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    ldlt.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-10);
}

// ─── preconditioner use ───────────────────────────────────────────────────────

#[test]
fn ldlt_as_precond_cg() {
    use linger::{ConjugateGradient, KrylovSolver, SolverParams, VerboseLevel};
    let n = 30;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);

    // Factorize A and use as direct preconditioner (M⁻¹ = A⁻¹).
    use linger::direct::DirectSolverPrecond;
    let ldlt_precond = DirectSolverPrecond::new(SparseLdlt::<f64>::default(), &a).unwrap();

    let cg = ConjugateGradient::<f64>::default();
    let params = SolverParams { rtol: 1e-12, max_iter: 100, verbose: VerboseLevel::Silent, ..Default::default() };
    let mut x = DenseVec::zeros(n);
    let result = cg.solve(&a, Some(&ldlt_precond), &b, &mut x, &params).unwrap();
    assert!(result.converged);
    assert!(result.iterations <= 3, "Should converge in ≤3 iters with exact precond, got {}", result.iterations);
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-10);
}

#[test]
fn ldlt_d_vals_sign_indefinite() {
    // For indefinite matrix, some D values should be negative.
    let n = 6;
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, if i % 2 == 0 { 3.0 } else { -3.0 });
    }
    let a = CsrMatrix::from_coo(&coo);
    let mut ldlt = SparseLdlt::<f64>::default();
    ldlt.factor(&a).unwrap();
    let d = ldlt.d_vals();
    let has_negative = d.iter().any(|&v| v < 0.0);
    let has_positive = d.iter().any(|&v| v > 0.0);
    assert!(has_negative && has_positive);
}
