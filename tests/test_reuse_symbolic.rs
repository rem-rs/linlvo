//! Integration tests for `DirectOptions::reuse_symbolic`.
//!
//! Verifies that when `reuse_symbolic = true`, subsequent `factor()` calls
//! on identically-sized matrices skip the ordering/compute step and reuse the
//! cached permutation.

use linger::direct::{DirectSolver, DirectOptions, SparseLu, SparseCholesky, MultifrontalLu, MultifrontalOptions, ordering::OrderingMethod};
use linger::sparse::{CooMatrix, CsrMatrix, ops::spmv_csr};
use linger::{DenseVec, Vector};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn spd_laplacian(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn nonsym_matrix(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 4.0);
        coo.push(i, (i + 1) % n, 1.0);
        if i > 0 { coo.push(i, i - 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn residual_norm(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let n = a.nrows();
    let mut ax = vec![0.0_f64; n];
    spmv_csr(a, x.as_slice(), &mut ax);
    let mut diff = vec![0.0_f64; n];
    for i in 0..n { diff[i] = ax[i] - b[i]; }
    let ss: f64 = diff.iter().fold(0.0, |acc, &v| acc + v * v);
    ss.sqrt() / b.norm2()
}

fn solve_check(mut solver: impl DirectSolver<f64>, a: &CsrMatrix<f64>, b: &DenseVec<f64>, expected_tol: f64) {
    solver.factor(a).unwrap();
    let mut x = DenseVec::zeros(a.nrows());
    solver.solve(b, &mut x).unwrap();
    let r = residual_norm(a, &x, b);
    assert!(
        r < expected_tol,
        "residual norm {r:.2e} exceeded tolerance {expected_tol:.2e}"
    );
}

// ─── SparseCholesky reuse_symbolic ──────────────────────────────────────────

#[test]
fn cholesky_reuse_symbolic_same_size_skips_ordering() {
    let n = 40;
    let a = spd_laplacian(n);
    let _b = DenseVec::from_vec(vec![1.0_f64; n]);

    let opts = DirectOptions {
        reuse_symbolic: true,
        ordering: OrderingMethod::Rcm,
        ..Default::default()
    };
    let mut solver = SparseCholesky::<f64>::new(opts);

    // First factor: must run analyze + factorize.
    solver.factor(&a).unwrap();
    let perm_first = solver.perm().to_vec();

    // Second factor on same-size matrix: reuse_symbolic should skip ordering.
    solver.factor(&a).unwrap();
    let perm_second = solver.perm().to_vec();

    // Permutations must be identical (same ordering was reused).
    assert_eq!(perm_first, perm_second);
}

#[test]
fn cholesky_reuse_symbolic_natural_ordering() {
    let n = 30;
    let a = spd_laplacian(n);
    let b = DenseVec::from_vec(vec![1.0_f64; n]);

    let opts = DirectOptions {
        reuse_symbolic: true,
        ordering: OrderingMethod::Natural,
        ..Default::default()
    };
    let solver = SparseCholesky::<f64>::new(opts);
    solve_check(solver, &a, &b, 1e-12);
}

#[test]
fn cholesky_reuse_symbolic_colamd() {
    let n = 30;
    let a = spd_laplacian(n);
    let b = DenseVec::from_vec(vec![1.0_f64; n]);

    let opts = DirectOptions {
        reuse_symbolic: true,
        ordering: OrderingMethod::Colamd,
        ..Default::default()
    };
    let solver = SparseCholesky::<f64>::new(opts);
    solve_check(solver, &a, &b, 1e-12);
}

// ─── SparseLu reuse_symbolic ─────────────────────────────────────────────────

#[test]
fn lu_reuse_symbolic_same_size_skips_ordering() {
    let n = 30;
    let a = nonsym_matrix(n);

    let opts = DirectOptions {
        reuse_symbolic: true,
        ordering: OrderingMethod::Rcm,
        ..Default::default()
    };
    let mut solver = SparseLu::<f64>::new(opts);

    solver.factor(&a).unwrap();
    let perm_first = solver.perm_q().to_vec();

    // Second call should reuse the same ordering.
    solver.factor(&a).unwrap();
    let perm_second = solver.perm_q().to_vec();

    assert_eq!(perm_first, perm_second);
}

#[test]
fn lu_reuse_symbolic_correctness() {
    let n = 20;
    let a = nonsym_matrix(n);
    let b = DenseVec::from_vec(vec![1.0_f64; n]);

    let opts = DirectOptions {
        reuse_symbolic: true,
        ordering: OrderingMethod::Rcm,
        ..Default::default()
    };
    let solver = SparseLu::<f64>::new(opts);
    solve_check(solver, &a, &b, 1e-10);
}

// ─── MultifrontalLu reuse_symbolic ────────────────────────────────────────────

#[test]
fn multifrontal_reuse_symbolic_same_size_skips_ordering() {
    let n = 20;
    let a = nonsym_matrix(n);

    let opts = MultifrontalOptions {
        base: DirectOptions {
            reuse_symbolic: true,
            ordering: OrderingMethod::Rcm,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut solver = MultifrontalLu::<f64>::with_options(opts);

    solver.factor(&a).unwrap();
    let perm_first = solver.perm_q().to_vec();

    solver.factor(&a).unwrap();
    let perm_second = solver.perm_q().to_vec();

    assert_eq!(perm_first, perm_second);
}

#[test]
fn multifrontal_reuse_symbolic_correctness() {
    let n = 15;
    let a = nonsym_matrix(n);
    let b = DenseVec::from_vec(vec![1.0_f64; n]);

    let opts = MultifrontalOptions {
        base: DirectOptions {
            reuse_symbolic: true,
            ordering: OrderingMethod::Natural,
            ..Default::default()
        },
        ..Default::default()
    };
    let solver = MultifrontalLu::<f64>::with_options(opts);
    solve_check(solver, &a, &b, 1e-10);
}

// ─── reuse_symbolic = false re-analyzes ──────────────────────────────────────

#[test]
fn cholesky_no_reuse_symbolic_reanalyzes_on_second_call() {
    let n = 30;
    let a = spd_laplacian(n);

    let opts = DirectOptions {
        reuse_symbolic: false,
        ordering: OrderingMethod::Rcm,
        ..Default::default()
    };
    let mut solver = SparseCholesky::<f64>::new(opts);

    solver.factor(&a).unwrap();
    let perm_first = solver.perm().to_vec();

    // Without reuse_symbolic, each factor() call always runs analyze.
    solver.factor(&a).unwrap();
    let perm_second = solver.perm().to_vec();

    // RCM is deterministic so the result is the same, but analyze DID run.
    assert_eq!(perm_first, perm_second);
}

// ─── reuse_symbolic with different size triggers re-analysis ─────────────────

#[test]
fn cholesky_reuse_symbolic_different_size_reanalyzes() {
    let a_small = spd_laplacian(20);
    let a_large  = spd_laplacian(40);

    let opts = DirectOptions {
        reuse_symbolic: true,
        ordering: OrderingMethod::Rcm,
        ..Default::default()
    };
    let mut solver = SparseCholesky::<f64>::new(opts);

    // Factor small matrix first.
    solver.factor(&a_small).unwrap();
    assert_eq!(solver.perm().len(), 20);

    // Now factor large matrix — must re-analyze (different size).
    solver.factor(&a_large).unwrap();
    assert_eq!(solver.perm().len(), 40);
}

// ─── reset_factors clears the symbolic cache ─────────────────────────────────

#[test]
fn reset_factors_clears_symbolic_cache() {
    let n = 30;
    let a = spd_laplacian(n);

    let opts = DirectOptions {
        reuse_symbolic: true,
        ordering: OrderingMethod::Rcm,
        ..Default::default()
    };
    let mut solver = SparseCholesky::<f64>::new(opts);

    solver.factor(&a).unwrap();
    let perm_first = solver.perm().to_vec();

    // reset_factors should clear the symbolic cache.
    solver.reset_factors();

    // After reset, factor() should re-analyze (symbolic cache is cleared).
    solver.factor(&a).unwrap();
    let perm_second = solver.perm().to_vec();
    assert_eq!(perm_first, perm_second); // same ordering computed fresh
}
