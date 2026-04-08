//! Integration tests for F3: SupernodalSparseCholesky.

use linger::{
    direct::{DirectSolver, DirectOptions, SupernodalSparseCholesky},
    direct::ordering::OrderingMethod,
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, LinearOperator,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -1.0); }
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

fn rel_res(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let mut ax = DenseVec::zeros(a.nrows());
    a.apply(x, &mut ax);
    let norm_r: f64 = ax.as_slice().iter().zip(b.as_slice())
        .map(|(av, bv)| (av - bv).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.as_slice().iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    if norm_b < 1e-300 { norm_r } else { norm_r / norm_b }
}

fn default_solver() -> SupernodalSparseCholesky<f64> {
    SupernodalSparseCholesky::new(
        DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() },
        8,
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// 1. Small 3×3 diagonal SPD system.
#[test]
fn sn_chol_tiny_diagonal() {
    let mut coo = CooMatrix::<f64>::new(3, 3);
    coo.push(0, 0, 4.0); coo.push(1, 1, 9.0); coo.push(2, 2, 16.0);
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![4.0, 9.0, 16.0]); // solution = [1,1,1]
    let mut x = DenseVec::zeros(3);

    let mut s = default_solver();
    s.factor(&a).unwrap();
    s.solve(&b, &mut x).unwrap();

    let rel = rel_res(&a, &x, &b);
    assert!(rel < 1e-12, "rel residual = {rel:.3e}");
}

/// 2. 1D Laplacian n=10: residual < 1e-10.
#[test]
fn sn_chol_laplacian_1d_n10() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let mut s = default_solver();
    s.factor(&a).unwrap();
    s.solve(&b, &mut x).unwrap();

    assert!(rel_res(&a, &x, &b) < 1e-10);
}

/// 3. 1D Laplacian n=100 with sn_target=16: residual < 1e-10.
#[test]
fn sn_chol_laplacian_1d_n100() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let mut s = SupernodalSparseCholesky::new(
        DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() }, 16,
    );
    s.factor(&a).unwrap();
    s.solve(&b, &mut x).unwrap();

    assert!(rel_res(&a, &x, &b) < 1e-10);
}

/// 4. Matches SparseCholesky on a 1D Laplacian.
#[test]
fn sn_chol_matches_scalar_cholesky() {
    use linger::direct::SparseCholesky;
    let n = 20;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect::<Vec<_>>());

    let mut sn  = default_solver();
    let mut sc  = SparseCholesky::<f64>::new(
        DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() },
    );
    sn.factor(&a).unwrap();
    sc.factor(&a).unwrap();

    let mut x_sn = DenseVec::zeros(n);
    let mut x_sc = DenseVec::zeros(n);
    sn.solve(&b, &mut x_sn).unwrap();
    sc.solve(&b, &mut x_sc).unwrap();

    for i in 0..n {
        let diff = (x_sn.as_slice()[i] - x_sc.as_slice()[i]).abs();
        assert!(diff < 1e-12, "solution mismatch at i={i}: diff={diff:.3e}");
    }
}

/// 5. Non-SPD matrix → SingularMatrix error.
#[test]
fn sn_chol_not_spd_returns_error() {
    use linger::SolverError;
    let mut coo = CooMatrix::<f64>::new(2, 2);
    coo.push(0, 0, -1.0); coo.push(1, 1, 2.0); // negative diagonal
    let a = CsrMatrix::from_coo(&coo);
    let mut s = default_solver();
    let result = s.factor(&a);
    assert!(matches!(result, Err(SolverError::SingularMatrix { .. })));
}

/// 6. sn_target=1 matches scalar left-looking Cholesky exactly.
#[test]
fn sn_chol_sn1_matches_scalar() {
    use linger::direct::SparseCholesky;
    let n = 30;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);

    let mut sn = SupernodalSparseCholesky::<f64>::new(
        DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() }, 1,
    );
    let mut sc = SparseCholesky::<f64>::new(
        DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() },
    );
    sn.factor(&a).unwrap();
    sc.factor(&a).unwrap();

    let mut x_sn = DenseVec::zeros(n);
    let mut x_sc = DenseVec::zeros(n);
    sn.solve(&b, &mut x_sn).unwrap();
    sc.solve(&b, &mut x_sc).unwrap();

    for i in 0..n {
        let diff = (x_sn.as_slice()[i] - x_sc.as_slice()[i]).abs();
        assert!(diff < 1e-12, "diff at i={i}: {diff:.3e}");
    }
}

/// 7. 2D Laplacian 8×8 (64 DOF): residual < 1e-10.
#[test]
fn sn_chol_laplacian_2d_8x8() {
    let n = 8;
    let a = laplacian_2d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n * n]);
    let mut x = DenseVec::zeros(n * n);

    let mut s = SupernodalSparseCholesky::new(
        DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() }, 8,
    );
    s.factor(&a).unwrap();
    s.solve(&b, &mut x).unwrap();

    assert!(rel_res(&a, &x, &b) < 1e-10);
}

/// 8. RCM ordering + supernodal Cholesky: residual < 1e-10.
#[test]
fn sn_chol_rcm_ordering() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let mut s = SupernodalSparseCholesky::new(
        DirectOptions { ordering: OrderingMethod::Rcm, ..Default::default() }, 8,
    );
    s.factor(&a).unwrap();
    s.solve(&b, &mut x).unwrap();

    assert!(rel_res(&a, &x, &b) < 1e-10);
}

/// 9. Supernode count for tridiagonal is ⌈n / sn_target⌉.
#[test]
fn sn_chol_snode_count_chain() {
    // Tridiagonal 1D Laplacian (n=32) with sn_target=8 and Natural ordering:
    // e-tree is a chain (parent[j]=j+1), so we get 32/8 = 4 supernodes.
    use linger::direct::etree::elimination_tree;
    use linger::direct::ordering::permute_symmetric;
    let n = 32;
    let a = laplacian_1d(n);
    let perm: Vec<usize> = (0..n).collect(); // Natural
    let b = permute_symmetric(&a, &perm);
    let parent = elimination_tree(&b);

    // Manually replicate supernode building logic.
    let sn_target = 8usize;
    let mut count = 0usize;
    let mut j = 0usize;
    while j < n {
        let mut size = 1usize;
        while size < sn_target && j + size < n && parent[j + size - 1] == j + size {
            size += 1;
        }
        count += 1;
        j += size;
    }
    assert_eq!(count, 4, "expected 4 supernodes for n=32 sn_target=8, got {count}");
}

/// 10. Multiple RHS reuse (reset_factors + re-factorize).
#[test]
fn sn_chol_reuse_symbolic() {
    let n = 20;
    let a = laplacian_1d(n);
    let b1 = DenseVec::from_vec(vec![1.0f64; n]);
    let b2 = DenseVec::from_vec((1..=n).map(|i| i as f64).collect::<Vec<_>>());

    let mut s = SupernodalSparseCholesky::new(
        DirectOptions {
            ordering: OrderingMethod::Natural,
            reuse_symbolic: true,
            ..Default::default()
        },
        8,
    );
    s.factor(&a).unwrap();
    let mut x1 = DenseVec::zeros(n);
    s.solve(&b1, &mut x1).unwrap();

    s.reset_factors();
    s.factor(&a).unwrap();
    let mut x2 = DenseVec::zeros(n);
    s.solve(&b2, &mut x2).unwrap();

    assert!(rel_res(&a, &x1, &b1) < 1e-10);
    assert!(rel_res(&a, &x2, &b2) < 1e-10);
}
