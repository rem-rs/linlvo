//! Integration tests for F1: SupernodalSparseLu.

use linger::{
    direct::{DirectSolver, DirectOptions, SupernodalSparseLu, SparseLu, ordering::OrderingMethod},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec,
    core::operator::LinearOperator,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

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
            if j > 0   { coo.push(id, id - 1, -1.0); coo.push(id - 1, id, -1.0); }
            if i > 0   { coo.push(id, id - n, -1.0); coo.push(id - n, id, -1.0); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let n = b.len();
    let xv = DenseVec::from_vec(x.to_vec());
    let mut ax = DenseVec::zeros(n);
    a.apply(&xv, &mut ax);
    let res: f64 = ax.as_slice().iter().zip(b)
        .map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
    let nrm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    res / nrm.max(1e-300)
}

fn rel_err(x: &[f64], x_ref: &[f64]) -> f64 {
    let diff: f64 = x.iter().zip(x_ref).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
    let nrm:  f64 = x_ref.iter().map(|v| v*v).sum::<f64>().sqrt();
    diff / nrm.max(1e-300)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// 1. Exact solve: 3×3 symmetric positive-definite matrix.
#[test]
fn sn_lu_exact_solve_small() {
    let mut coo = CooMatrix::<f64>::new(3, 3);
    coo.push(0,0,4.0); coo.push(0,1,1.0);
    coo.push(1,0,1.0); coo.push(1,1,3.0); coo.push(1,2,1.0);
    coo.push(2,1,1.0); coo.push(2,2,5.0);
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![5.0, 10.0, 6.0]);
    let mut solver = SupernodalSparseLu::<f64>::default();
    solver.factor(&a).unwrap();
    let mut x = DenseVec::zeros(3);
    solver.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-10,
        "small SPD residual too large");
}

/// 2. Matches SparseLu on 1D Laplacian (n=30).
#[test]
fn sn_lu_matches_sparse_lu_1d() {
    let n = 30;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);

    // Reference solution via SparseLu.
    let mut ref_solver = SparseLu::<f64>::default();
    ref_solver.factor(&a).unwrap();
    let mut x_ref = DenseVec::zeros(n);
    ref_solver.solve(&b, &mut x_ref).unwrap();

    // Supernodal solver (sn_target = 8).
    let mut sn_solver = SupernodalSparseLu::<f64>::new(DirectOptions::default(), 8);
    sn_solver.factor(&a).unwrap();
    let mut x_sn = DenseVec::zeros(n);
    sn_solver.solve(&b, &mut x_sn).unwrap();

    assert!(rel_err(x_sn.as_slice(), x_ref.as_slice()) < 1e-9,
        "supernodal vs SparseLu relative error = {}",
        rel_err(x_sn.as_slice(), x_ref.as_slice()));
}

/// 3. Supernode count for tridiagonal matrix (chain e-tree).
///    With sn_target = 8 and n = 32: expect exactly 4 supernodes.
#[test]
fn sn_lu_snode_count_chain() {
    let n = 32;
    let a = laplacian_1d(n);
    let mut solver = SupernodalSparseLu::<f64>::new(
        DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() }, 8
    );
    solver.analyze(&a).unwrap();
    assert_eq!(solver.snode_count(), 4,
        "expected 4 supernodes for n=32 tridiagonal with sn_target=8");
}

/// 4. Larger 1D Laplacian (n=100) with sn_target=16.
#[test]
fn sn_lu_laplacian_1d_n100() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut solver = SupernodalSparseLu::<f64>::new(DirectOptions::default(), 16);
    solver.factor(&a).unwrap();
    let mut x = DenseVec::zeros(n);
    solver.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-9,
        "n=100 residual too large");
}

/// 5. Width-1 supernodes match the scalar SparseLu exactly.
#[test]
fn sn_lu_width1_matches_sparse_lu() {
    let n = 20;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);

    let mut ref_solver = SparseLu::<f64>::new(
        DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() }
    );
    ref_solver.factor(&a).unwrap();
    let mut x_ref = DenseVec::zeros(n);
    ref_solver.solve(&b, &mut x_ref).unwrap();

    let mut sn = SupernodalSparseLu::<f64>::new(
        DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() }, 1
    );
    sn.factor(&a).unwrap();
    assert_eq!(sn.snode_count(), n,
        "width=1 must give n={n} supernodes");
    let mut x_sn = DenseVec::zeros(n);
    sn.solve(&b, &mut x_sn).unwrap();

    assert!(rel_err(x_sn.as_slice(), x_ref.as_slice()) < 1e-12,
        "width=1 vs SparseLu err = {}", rel_err(x_sn.as_slice(), x_ref.as_slice()));
}

/// 6. RCM ordering with supernodal solver.
#[test]
fn sn_lu_rcm_ordering() {
    let n = 25;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut solver = SupernodalSparseLu::<f64>::new(
        DirectOptions { ordering: OrderingMethod::Rcm, ..Default::default() }, 6
    );
    solver.factor(&a).unwrap();
    let mut x = DenseVec::zeros(n);
    solver.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-9,
        "RCM residual too large");
}

/// 7. 2D Laplacian (small grid n=5×5).
#[test]
fn sn_lu_laplacian_2d_5x5() {
    let n = 5;
    let a = laplacian_2d(n);
    let nn = n * n;
    let b = DenseVec::from_vec(vec![1.0f64; nn]);
    let mut solver = SupernodalSparseLu::<f64>::new(DirectOptions::default(), 4);
    solver.factor(&a).unwrap();
    let mut x = DenseVec::zeros(nn);
    solver.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-9,
        "2D Laplacian 5×5 residual too large");
}

/// 8. reuse_symbolic gives identical results across two factorize calls.
#[test]
fn sn_lu_reuse_symbolic_consistent() {
    let n = 20;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut solver = SupernodalSparseLu::<f64>::new(
        DirectOptions { reuse_symbolic: true, ..Default::default() }, 4
    );
    solver.factor(&a).unwrap();
    let mut x1 = DenseVec::zeros(n);
    solver.solve(&b, &mut x1).unwrap();

    solver.factor(&a).unwrap();
    let mut x2 = DenseVec::zeros(n);
    solver.solve(&b, &mut x2).unwrap();

    assert!(rel_err(x1.as_slice(), x2.as_slice()) < 1e-12,
        "reuse_symbolic produced inconsistent results");
}

/// 9. Solution is finite for all entries.
#[test]
fn sn_lu_solution_is_finite() {
    let n = 40;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut solver = SupernodalSparseLu::<f64>::new(DirectOptions::default(), 8);
    solver.factor(&a).unwrap();
    let mut x = DenseVec::zeros(n);
    solver.solve(&b, &mut x).unwrap();
    assert!(x.as_slice().iter().all(|v| v.is_finite()),
        "solution contains non-finite values");
}

/// 10. Supernode count with sn_target=1 is always n.
#[test]
fn sn_lu_width1_snode_count() {
    for &n in &[5usize, 10, 20] {
        let a = laplacian_1d(n);
        let mut solver = SupernodalSparseLu::<f64>::new(
            DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() }, 1
        );
        solver.analyze(&a).unwrap();
        assert_eq!(solver.snode_count(), n,
            "n={n} sn_target=1 must give n supernodes");
    }
}
