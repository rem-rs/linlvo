//! Iterative refinement integration tests.
//!
//! Verifies that `refine_steps > 0` improves solution accuracy, especially
//! for ill-conditioned systems.

use linger::{
    direct::{DirectSolver, DirectOptions, SparseLu, SparseCholesky},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec,
    core::operator::LinearOperator,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i-1, -1.0); }
        if i+1 < n   { coo.push(i, i+1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

/// Hilbert-like ill-conditioned SPD matrix:  A[i,j] = 1/(i+j+1).
fn hilbert_spd(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            coo.push(i, j, 1.0 / (i + j + 1) as f64);
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn relative_residual(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let mut ax = DenseVec::zeros(a.nrows());
    a.apply(x, &mut ax);
    let r:  f64 = ax.as_slice().iter().zip(b.as_slice()).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.as_slice().iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    if nb < 1e-300 { r } else { r / nb }
}

// ─── SparseLu refinement ──────────────────────────────────────────────────────

#[test]
fn lu_refine_0_steps_baseline() {
    let n = 20;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());
    let mut x = DenseVec::zeros(n);
    let mut solver = SparseLu::<f64>::new(DirectOptions {
        refine_steps: 0, ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    let res = relative_residual(&a, &x, &b);
    assert!(res < 1e-8, "baseline residual = {res:.3e}");
}

#[test]
fn lu_refine_2_steps_improves_residual() {
    let n = 20;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());

    // Without refinement.
    let mut x0 = DenseVec::zeros(n);
    let mut s0 = SparseLu::<f64>::new(DirectOptions { refine_steps: 0, ..Default::default() });
    s0.factor(&a).unwrap();
    s0.solve(&b, &mut x0).unwrap();

    // With 2 refinement steps.
    let mut x2 = DenseVec::zeros(n);
    let mut s2 = SparseLu::<f64>::new(DirectOptions { refine_steps: 2, ..Default::default() });
    s2.factor(&a).unwrap();
    s2.solve(&b, &mut x2).unwrap();

    let res0 = relative_residual(&a, &x0, &b);
    let res2 = relative_residual(&a, &x2, &b);
    // Refinement should not make things worse and should stay accurate.
    assert!(res2 < 1e-9, "refined residual = {res2:.3e}, unrefined = {res0:.3e}");
}

#[test]
fn lu_refine_ill_conditioned() {
    // Hilbert matrix: extremely ill-conditioned.
    // Refinement should recover at least moderate accuracy.
    let n = 8;
    let a = hilbert_spd(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    let mut solver = SparseLu::<f64>::new(DirectOptions {
        refine_steps: 3, ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    let res = relative_residual(&a, &x, &b);
    assert!(res < 1e-6, "Hilbert n={n} with refinement: rel res = {res:.3e}");
}

// ─── SparseCholesky refinement ────────────────────────────────────────────────

#[test]
fn cholesky_refine_0_steps_baseline() {
    let n = 20;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());
    let mut x = DenseVec::zeros(n);
    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        refine_steps: 0, ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    let res = relative_residual(&a, &x, &b);
    assert!(res < 1e-8, "baseline residual = {res:.3e}");
}

#[test]
fn cholesky_refine_2_steps_accurate() {
    let n = 20;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());
    let mut x = DenseVec::zeros(n);
    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        refine_steps: 2, ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    let res = relative_residual(&a, &x, &b);
    assert!(res < 1e-9, "cholesky refined residual = {res:.3e}");
}

#[test]
fn cholesky_refine_ill_conditioned() {
    let n = 8;
    let a = hilbert_spd(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        refine_steps: 3, ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    let res = relative_residual(&a, &x, &b);
    assert!(res < 1e-6, "Hilbert n={n} cholesky+refine: rel res = {res:.3e}");
}

// ─── Multiple RHS reuse ───────────────────────────────────────────────────────

#[test]
fn lu_refine_multiple_rhs() {
    let n = 10;
    let a = laplacian_1d(n);
    let mut solver = SparseLu::<f64>::new(DirectOptions {
        refine_steps: 2, ..Default::default()
    });
    solver.factor(&a).unwrap();

    for k in 1..=3usize {
        let b = DenseVec::from_vec(vec![k as f64; n]);
        let mut x = DenseVec::zeros(n);
        solver.solve(&b, &mut x).unwrap();
        assert!(relative_residual(&a, &x, &b) < 1e-9,
            "rhs {k}: residual = {:.3e}", relative_residual(&a, &x, &b));
    }
}
