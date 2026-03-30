//! ex05 — faer sparse matrix through the FaerSparseOp adapter.
//!
//! **Purpose**: Verify that a `faer::sparse::SparseColMat` (CSC format) wrapped
//! with `FaerSparseOp` gives identical SpMV results to linger's own `CsrMatrix`.
//!
//! **PETSc analog**
//!   Wrapping faer corresponds to using PETSc's `MatCreateShell` or
//!   `MatCreateSeqAIJWithArrays` with an externally-managed storage.
//!
//! **Why this matters**
//!   faer is the high-performance dense/sparse linear-algebra backend;
//!   linger leverages it for direct solvers (Sprint 3) and eventually for
//!   block-sparse formats. The adapter ensures the same Krylov interface.

use linger::{
    sparse::adapt_faer::FaerSparseOp,
    sparse::{CooMatrix as LingerCoo, CsrMatrix as LingerCsr},
    DenseVec, LinearOperator,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn sep(title: &str) {
    println!("\n━━━━ {title} ━━━━━");
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |s, &x| s + x * x).sqrt()
}

fn linger_poisson_1d(n: usize) -> LingerCsr<f64> {
    let mut coo = LingerCoo::with_capacity(n, n, 3 * n - 2);
    for i in 0..n {
        if i > 0     { coo.push(i, i - 1, -1.0); }
        coo.push(i, i, 2.0);
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    LingerCsr::from_coo(&coo)
}

/// Build a `faer` CSC sparse matrix for the 1-D Poisson problem.
///
/// faer 0.21's `try_new_from_triplets` takes a `&[Triplet<R, C, T>]` slice —
/// analogous to HYPRE's `HYPRE_IJMatrixAssemble` or PETSc's `MatAssemblyEnd`.
fn faer_poisson_1d(n: usize) -> faer::sparse::SparseColMat<usize, f64> {
    use faer::sparse::Triplet;
    let mut entries: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(3 * n);
    for i in 0..n {
        if i > 0 {
            entries.push(Triplet { row: i, col: i - 1, val: -1.0 });
        }
        entries.push(Triplet { row: i, col: i, val: 2.0 });
        if i < n - 1 {
            entries.push(Triplet { row: i, col: i + 1, val: -1.0 });
        }
    }
    faer::sparse::SparseColMat::try_new_from_triplets(n, n, &entries)
        .expect("faer: failed to construct sparse matrix from triplets")
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    sep("ex05: faer sparse adapter  (PETSc MatCreateShell analogy)");

    for &n in &[10usize, 100, 500] {
        println!("\n  ── n = {n} ──────────────────────────────────────");

        // Build both representations
        let linger_csr = linger_poisson_1d(n);
        let faer_csc   = faer_poisson_1d(n);

        println!("  linger CSR:  {}×{}  nnz={}", linger_csr.nrows(), linger_csr.ncols(), linger_csr.nnz());
        let faer_nnz = faer_csc.as_ref().col_ptr().last().copied().unwrap_or(0);
        println!("  faer   CSC:  {}×{}  nnz={}", faer_csc.nrows(), faer_csc.ncols(), faer_nnz);

        assert_eq!(faer_csc.nrows(), n);
        assert_eq!(faer_nnz, linger_csr.nnz());

        // Wrap faer matrix as linger LinearOperator
        let op = FaerSparseOp::<f64>::new(faer_csc);
        println!("  FaerSparseOp: {}×{}", op.nrows(), op.ncols());

        // Test vector: x[i] = sin(0.1·i)
        let x_vec: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let x = DenseVec::from_vec(x_vec.clone());

        // y_faer = FaerSparseOp · x
        let mut y_faer = DenseVec::zeros(n);
        op.apply(&x, &mut y_faer);

        // y_linger = linger CsrMatrix · x
        let mut y_linger = vec![0.0f64; n];
        linger_csr.spmv(&x_vec, &mut y_linger);

        let diff: f64 = y_faer.as_slice().iter().zip(&y_linger)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = norm2(&y_linger);
        let rel = if scale > 0.0 { diff / scale } else { diff };

        println!("  ‖y_faer − y_linger‖/‖y‖ = {rel:.3e}");
        assert!(rel < 1e-13, "faer adapter mismatch: {rel:.3e}");
        println!("  SpMV match  ✓");
    }

    // ── Trait-object usage preview ─────────────────────────────────────────────
    println!("\n  ── trait-object (dyn LinearOperator) ───────────────────");
    {
        let n = 20;
        let faer_csc = faer_poisson_1d(n);
        let op = FaerSparseOp::<f64>::new(faer_csc);

        let dyn_op: &dyn LinearOperator<Vector = DenseVec<f64>> = &op;
        let x = DenseVec::from_vec(vec![1.0f64; n]);
        let mut y = DenseVec::zeros(n);
        dyn_op.apply(&x, &mut y);
        // A·𝟏 for 1-D Poisson: y[0]=1, y[n-1]=1, interior=0
        assert!((y[0]   - 1.0).abs() < 1e-14);
        assert!((y[n-1] - 1.0).abs() < 1e-14);
        println!("  A·𝟏 first = {:.1}, last = {:.1}  ✓", y[0], y[n-1]);
    }

    println!("\n  OK\n");
}
