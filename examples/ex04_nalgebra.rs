//! ex04 — direct use of `nalgebra_sparse::CsrMatrix` as a linger `LinearOperator`.
//!
//! **Purpose**: Verify that a matrix assembled with `nalgebra_sparse` gives
//! identical SpMV results to linger's own `CsrMatrix`, without a wrapper type.

use linger::{
    sparse::{CooMatrix as LingerCoo, CsrMatrix as LingerCsr},
    DenseVec, LinearOperator,
};
use nalgebra_sparse::CooMatrix as NaCoo;

fn sep(title: &str) {
    println!("\n━━━━ {title} ━━━━━");
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |s, &x| s + x * x).sqrt()
}

fn na_poisson_1d(n: usize) -> nalgebra_sparse::CsrMatrix<f64> {
    let mut coo = NaCoo::<f64>::new(n, n);
    for i in 0..n {
        if i > 0 {
            coo.push(i, i - 1, -1.0);
        }
        coo.push(i, i, 2.0);
        if i < n - 1 {
            coo.push(i, i + 1, -1.0);
        }
    }
    nalgebra_sparse::CsrMatrix::from(&coo)
}

fn linger_poisson_1d(n: usize) -> LingerCsr<f64> {
    let mut coo = LingerCoo::with_capacity(n, n, 3 * n - 2);
    for i in 0..n {
        if i > 0 {
            coo.push(i, i - 1, -1.0);
        }
        coo.push(i, i, 2.0);
        if i < n - 1 {
            coo.push(i, i + 1, -1.0);
        }
    }
    LingerCsr::from_coo(&coo)
}

fn main() {
    sep("ex04: direct nalgebra CSR integration");

    for &n in &[10usize, 100, 1_000] {
        println!("\n  ── n = {n} ──────────────────────────────────────");

        let na_csr = na_poisson_1d(n);
        let linger_csr = linger_poisson_1d(n);

        println!(
            "  nalgebra CSR:  {}×{}  nnz={}",
            na_csr.nrows(),
            na_csr.ncols(),
            na_csr.nnz()
        );
        println!(
            "  linger   CSR:  {}×{}  nnz={}",
            linger_csr.nrows(),
            linger_csr.ncols(),
            linger_csr.nnz()
        );

        assert_eq!(na_csr.nrows(), linger_csr.nrows());
        assert_eq!(na_csr.nnz(), linger_csr.nnz());

        let x_vec: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let x = DenseVec::from_vec(x_vec.clone());

        let mut y_na = DenseVec::zeros(n);
        na_csr.apply(&x, &mut y_na);

        let mut y_li = vec![0.0f64; n];
        linger_csr.spmv(&x_vec, &mut y_li);

        let diff: f64 = y_na
            .as_slice()
            .iter()
            .zip(&y_li)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = norm2(&y_li);
        let rel = if scale > 0.0 { diff / scale } else { diff };

        println!("  ‖y_nalgebra − y_linger‖/‖y‖ = {rel:.3e}");
        assert!(rel < 1e-13, "direct nalgebra integration mismatch: {rel:.3e}");
        println!("  SpMV match  ✓");

        assert_eq!(na_csr.nrows(), n);
        assert_eq!(na_csr.ncols(), n);
        println!("  dimensions  ✓");
    }

    println!("\n  ── trait-object (dyn LinearOperator) ───────────────────");
    {
        let n = 20;
        let na = na_poisson_1d(n);

        let dyn_op: &dyn LinearOperator<Vector = DenseVec<f64>> = &na;
        let x = DenseVec::from_vec(vec![1.0f64; n]);
        let mut y = DenseVec::zeros(n);
        dyn_op.apply(&x, &mut y);

        assert!((y[0] - 1.0).abs() < 1e-14, "A·1 first row");
        assert!((y[n - 1] - 1.0).abs() < 1e-14, "A·1 last row");
        println!("  A·1 first = {:.1}, last = {:.1}  ✓", y[0], y[n - 1]);
    }

    println!("\n  OK\n");
}