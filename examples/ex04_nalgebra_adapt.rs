//! ex04 — nalgebra-sparse CSR matrix through the NalgebraCsrOp adapter.
//!
//! **Purpose**: Verify that a matrix assembled with `nalgebra_sparse` gives
//! identical SpMV results to linger's own `CsrMatrix`, validating the
//! `NalgebraCsrOp` adapter layer.
//!
//! **PETSc analog**
//!   `MatCreateSeqAIJ` / `MatSetValues` (native PETSc format)  vs.
//!   wrapping an external sparse format through `MatCreateShell`.
//!
//! **Why this matters**
//!   FEA frameworks that already use nalgebra for element matrices can hand
//!   the assembled global stiffness matrix directly to linger without a copy.

use linger::{
    sparse::adapt_nalgebra::NalgebraCsrOp,
    sparse::{CooMatrix as LingerCoo, CsrMatrix as LingerCsr},
    DenseVec, LinearOperator,
};
use nalgebra_sparse::CooMatrix as NaCoo;

// ─── helpers ─────────────────────────────────────────────────────────────────

fn sep(title: &str) {
    println!("\n━━━━ {title} ━━━━━");
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |s, &x| s + x * x).sqrt()
}

// Build 1-D Poisson via nalgebra's COO → CSR
fn na_poisson_1d(n: usize) -> nalgebra_sparse::CsrMatrix<f64> {
    let mut coo = NaCoo::<f64>::new(n, n);
    for i in 0..n {
        if i > 0     { coo.push(i, i - 1, -1.0); }
        coo.push(i, i,  2.0);
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    nalgebra_sparse::CsrMatrix::from(&coo)
}

// Build same 1-D Poisson via linger's own CooMatrix
fn linger_poisson_1d(n: usize) -> LingerCsr<f64> {
    let mut coo = LingerCoo::with_capacity(n, n, 3 * n - 2);
    for i in 0..n {
        if i > 0     { coo.push(i, i - 1, -1.0); }
        coo.push(i, i,  2.0);
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    LingerCsr::from_coo(&coo)
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    sep("ex04: nalgebra-sparse adapter  (PETSc MatCreateShell analogy)");

    for &n in &[10usize, 100, 1_000] {
        println!("\n  ── n = {n} ──────────────────────────────────────");

        // Build both representations
        let na_csr    = na_poisson_1d(n);
        let linger_csr = linger_poisson_1d(n);

        println!("  nalgebra CSR:  {}×{}  nnz={}",
            na_csr.nrows(), na_csr.ncols(), na_csr.nnz());
        println!("  linger   CSR:  {}×{}  nnz={}",
            linger_csr.nrows(), linger_csr.ncols(), linger_csr.nnz());

        assert_eq!(na_csr.nrows(), linger_csr.nrows());
        assert_eq!(na_csr.nnz(),   linger_csr.nnz());

        // Wrap nalgebra matrix as a linger LinearOperator
        let op = NalgebraCsrOp::<f64>::new(na_csr);

        println!("  NalgebraCsrOp: {}×{}", op.nrows(), op.ncols());

        // Random-ish test vector (deterministic: x[i] = sin(i))
        let x_vec: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let x = DenseVec::from_vec(x_vec.clone());

        // Compute y_na = NalgebraCsrOp · x
        let mut y_na = DenseVec::zeros(n);
        op.apply(&x, &mut y_na);

        // Compute y_li = linger CsrMatrix · x
        let mut y_li = vec![0.0f64; n];
        linger_csr.spmv(&x_vec, &mut y_li);

        // Compare
        let diff: f64 = y_na.as_slice().iter().zip(&y_li)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = norm2(&y_li);
        let rel = if scale > 0.0 { diff / scale } else { diff };

        println!("  ‖y_nalgebra − y_linger‖/‖y‖ = {rel:.3e}");
        assert!(rel < 1e-13, "adapter mismatch: {rel:.3e}");
        println!("  SpMV match  ✓");

        // Verify op.nrows / op.ncols
        assert_eq!(op.nrows(), n);
        assert_eq!(op.ncols(), n);
        println!("  dimensions  ✓");
    }

    // ── Demonstrate trait-object usage ────────────────────────────────────────
    //
    // Sprint 2 will pass &dyn LinearOperator<Vector=DenseVec<f64>> to solvers.
    // This block shows the pattern compiles correctly.
    println!("\n  ── trait-object (dyn LinearOperator) ───────────────────");
    {
        let n   = 20;
        let na  = na_poisson_1d(n);
        let op  = NalgebraCsrOp::<f64>::new(na);

        // Erase to trait object
        let dyn_op: &dyn LinearOperator<Vector = DenseVec<f64>> = &op;
        let x  = DenseVec::from_vec(vec![1.0f64; n]);
        let mut y = DenseVec::zeros(n);
        dyn_op.apply(&x, &mut y);
        // A·𝟏: for 1-D Poisson, interior rows give 0, endpoints give 1
        assert!((y[0]   - 1.0).abs() < 1e-14, "A·𝟏 first row");
        assert!((y[n-1] - 1.0).abs() < 1e-14, "A·𝟏 last row");
        println!("  A·𝟏 first = {:.1}, last = {:.1}  ✓", y[0], y[n-1]);
    }

    println!("\n  OK\n");
}
