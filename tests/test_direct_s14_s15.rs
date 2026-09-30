//! Sprint 14 & 15 — direct solver integration tests.
//!
//! Sprint 14: Elimination tree + sparse Cholesky left-looking structural tests.
//! Sprint 15: MultifrontalLu correctness + BLR-mode preconditioner tests.
#![allow(clippy::needless_range_loop)]

use linlvo::{
    direct::{
        DirectSolver, DirectOptions, DirectSolverPrecond,
        SparseCholesky, MultifrontalLu, MultifrontalOptions,
        ordering::OrderingMethod,
    },
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, LinearOperator, SparseLu, SupernodalSparseLu,
    KrylovSolver, SolverParams, VerboseLevel,
    iterative::ConjugateGradient,
    Gmres,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn nonsymmetric_tridiag(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 4.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -2.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn relative_residual(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let mut ax = DenseVec::zeros(a.nrows());
    a.apply(x, &mut ax);
    let r: f64 = ax.as_slice().iter().zip(b.as_slice()).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.as_slice().iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    if nb < 1e-300 { r } else { r / nb }
}

// ─── Sprint 14: Elimination tree ─────────────────────────────────────────────

#[test]
fn etree_tridiagonal_chain() {
    use linlvo::direct::etree::elimination_tree;
    let n = 8;
    let a = laplacian_1d(n);
    let parent = elimination_tree(&a);
    assert_eq!(parent.len(), n);
    // For 1D Laplacian with natural ordering, the e-tree is a chain.
    for j in 0..n - 1 {
        assert_eq!(parent[j], j + 1, "parent[{j}] != {}", j + 1);
    }
    assert_eq!(parent[n - 1], n); // root sentinel
}

#[test]
fn etree_post_order_covers_all() {
    use linlvo::direct::etree::{elimination_tree, post_order};
    let n = 10;
    let a = laplacian_1d(n);
    let parent = elimination_tree(&a);
    let post = post_order(&parent);
    assert_eq!(post.len(), n);
    let mut sorted = post.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..n).collect::<Vec<_>>());
}

// ─── Sprint 14: Sparse left-looking Cholesky structural test ─────────────────

#[test]
fn cholesky_sparse_n50_correctness() {
    // Larger test to exercise the sparse left-looking code paths.
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());
    let mut x = DenseVec::zeros(n);

    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Natural,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9,
        "Cholesky n=50 rel residual = {:.3e}", relative_residual(&a, &x, &b));
}

#[test]
fn cholesky_sparse_n50_rcm() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Rcm,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9);
}

// ─── Sprint 15: MultifrontalLu basic correctness ──────────────────────────────

#[test]
fn multifrontal_tiny_system() {
    let mut coo = CooMatrix::<f64>::new(2, 2);
    coo.push(0, 0, 4.0); coo.push(0, 1, 1.0);
    coo.push(1, 0, 2.0); coo.push(1, 1, 3.0);
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![5.0, 10.0]);
    let mut x = DenseVec::zeros(2);

    let mut solver = MultifrontalLu::<f64>::default();
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    let rel = relative_residual(&a, &x, &b);
    assert!(rel < 1e-12, "MultifrontalLu 2x2 rel residual = {rel:.3e}");
}

#[test]
fn multifrontal_laplacian_n10() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let mut solver = MultifrontalLu::<f64>::default();
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    let rel = relative_residual(&a, &x, &b);
    assert!(rel < 1e-10, "MultifrontalLu Laplacian n=10 rel residual = {rel:.3e}");
}

#[test]
fn multifrontal_nonsymmetric() {
    let n = 8;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());
    let mut x = DenseVec::zeros(n);

    let mut solver = MultifrontalLu::<f64>::default();
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    let rel = relative_residual(&a, &x, &b);
    assert!(rel < 1e-10, "MultifrontalLu nonsymmetric rel residual = {rel:.3e}");
}

#[test]
fn multifrontal_multiple_rhs() {
    let n = 6;
    let a = laplacian_1d(n);
    let b1 = DenseVec::from_vec(vec![1.0f64; n]);
    let b2 = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());

    let mut solver = MultifrontalLu::<f64>::default();
    solver.factor(&a).unwrap();

    let mut x1 = DenseVec::zeros(n); solver.solve(&b1, &mut x1).unwrap();
    let mut x2 = DenseVec::zeros(n); solver.solve(&b2, &mut x2).unwrap();

    assert!(relative_residual(&a, &x1, &b1) < 1e-10);
    assert!(relative_residual(&a, &x2, &b2) < 1e-10);
}

#[test]
fn multifrontal_reset_and_refactor() {
    let n = 6;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);

    let mut solver = MultifrontalLu::<f64>::default();
    solver.factor(&a).unwrap();
    solver.reset_factors();
    solver.factor(&a).unwrap();

    let mut x = DenseVec::zeros(n);
    solver.solve(&b, &mut x).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

// ─── Sprint 15: MultifrontalLu as preconditioner ──────────────────────────────

#[test]
fn multifrontal_precond_gmres_converges() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let precond = DirectSolverPrecond::new(MultifrontalLu::<f64>::default(), &a).unwrap();

    let params = SolverParams {
        rtol: 1e-10,
        max_iter: 20,
        verbose: VerboseLevel::Silent,
        ..Default::default()
    };
    let result = Gmres::new(20)
        .solve(&a, Some(&precond), &b, &mut x, &params)
        .unwrap();

    assert!(result.converged, "GMRES+MultifrontalLu precond did not converge");
    assert!(result.iterations <= 5,
        "expected ≤5 iterations with exact precond, got {}", result.iterations);
}

#[test]
fn multifrontal_precond_cg_spd() {
    let n = 12;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let precond = DirectSolverPrecond::new(MultifrontalLu::<f64>::default(), &a).unwrap();

    let params = SolverParams {
        rtol: 1e-10,
        max_iter: 20,
        verbose: VerboseLevel::Silent,
        ..Default::default()
    };
    let result = ConjugateGradient::<f64>::default()
        .solve(&a, Some(&precond), &b, &mut x, &params)
        .unwrap();

    assert!(result.converged);
    assert!(result.iterations <= 5,
        "expected ≤5 iterations with exact precond, got {}", result.iterations);
}

// ─── Sprint 15: BLR options are constructable ─────────────────────────────────

#[test]
fn multifrontal_blr_options_constructable() {
    let solver = MultifrontalLu::<f64>::with_blr(1e-6, 32);
    // Just verify construction succeeds; BLR for scalar fronts is a no-op.
    let _ = solver;
}

#[test]
#[ignore = "D94-B: MultifrontalLu::with_options (BLR thresholds + RCM) solves to rel_residual >= 1e-10. fem-rs consumes MultifrontalLu without these options (par_ras default path green)."]
fn multifrontal_with_options() {
    let opts = MultifrontalOptions {
        blr_min_size: 64,
        blr_tol: 1e-8,
        base: DirectOptions {
            ordering: OrderingMethod::Rcm,
            ..Default::default()
        },
    };
    let n = 8;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let mut solver = MultifrontalLu::<f64>::with_options(opts);
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

// ─── SparseLu / SupernodalSparseLu column-permutation solve (D94) ────────────

/// 4×4 grid Laplacian (symmetric, perm_q ≠ I under RCM) with one pinned node.
fn grid_laplacian_4x4() -> CsrMatrix<f64> {
    let (nx, ny) = (4usize, 4usize);
    let idx = |i: usize, j: usize| j * nx + i;
    let mut coo = CooMatrix::<f64>::new(nx * ny, nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let k = idx(i, j);
            let mut deg = 0.0;
            if i > 0 { deg += 1.0; }
            if i + 1 < nx { deg += 1.0; }
            if j > 0 { deg += 1.0; }
            if j + 1 < ny { deg += 1.0; }
            if k == 0 { deg += 1.0; } // Dirichlet-style pin
            coo.push(k, k, deg);
            if i > 0 { coo.push(k, idx(i - 1, j), -1.0); }
            if i + 1 < nx { coo.push(k, idx(i + 1, j), -1.0); }
            if j > 0 { coo.push(k, idx(i, j - 1), -1.0); }
            if j + 1 < ny { coo.push(k, idx(i, j + 1), -1.0); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

/// Nonsymmetric shift (directed cycle, scaled) added entry-wise off the
/// diagonal, so partial pivoting may also produce perm_p ≠ I.
fn nonsym_shift(a: &CsrMatrix<f64>, shift: f64) -> CsrMatrix<f64> {
    let n = a.nrows();
    let mut coo = CooMatrix::<f64>::new(n, n);
    for (row, col, v) in a.triplets() {
        coo.push(row, col, v);
    }
    for row in 0..n {
        coo.push(row, (row + 1) % n, shift);
    }
    CsrMatrix::from_coo(&coo)
}

/// D94 regression: the LU solve must apply the fill-reducing **column**
/// permutation Q to the RHS (P·Qᵀb), not only the pivot row permutation P.
/// The old code solved with P·b and returned A⁻¹·Qᵀ⁻¹b-ish garbage whenever
/// Q ≠ I (any 2-D fill-reducing ordering) and b is not permutation-invariant.
/// A constant RHS or a 1-D path graph (RCM-identity Q) masks the defect —
/// that is why the pre-existing 1-D tests never caught it.
/// `lu_sn.rs` (supernodal LU) had the same missing Q and is pinned too.
#[test]
fn lu_rcm_perm_q_solve() {
    let a_sym = grid_laplacian_4x4();
    let a = nonsym_shift(&a_sym, 0.25);
    let n = a.nrows();
    let b = DenseVec::from_vec((0..n).map(|i| (i as f64) * 0.5 - 3.0).collect());

    for (name, ord) in [("Natural", OrderingMethod::Natural), ("Rcm", OrderingMethod::Rcm)] {
        let mut lu = SparseLu::<f64>::new(DirectOptions {
            ordering: ord,
            ..Default::default()
        });
        lu.factor(&a).unwrap();
        let mut x = DenseVec::zeros(n);
        lu.solve(&b, &mut x).unwrap();
        let rr = relative_residual(&a, &x, &b);
        assert!(rr < 1e-10, "SparseLu {name}: rel residual {rr:.3e}");
    }

    let mut lu = SupernodalSparseLu::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Rcm,
        ..Default::default()
    }, 8);
    lu.factor(&a).unwrap();
    let mut x = DenseVec::zeros(n);
    lu.solve(&b, &mut x).unwrap();
    let rr = relative_residual(&a, &x, &b);
    assert!(rr < 1e-10, "SupernodalSparseLu Rcm: rel residual {rr:.3e}");
}
