//! Sprint 7 — eigenvalue solver tests.
//!
//! All tests use small matrices with known eigenvalues so results can be
//! verified analytically.  Tolerances are set relative to the eigenvalue
//! magnitude so the suite is numerically robust.

use linger::{
    eigen::{EigenParams, EigenSolver, EigenWhich, InverseIter, PowerIter, RayleighQuotientIter, SubspaceIter},
    sparse::{CooMatrix, CsrMatrix},
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn diag_csr(diag: &[f64]) -> CsrMatrix<f64> {
    let n = diag.len();
    let mut coo = CooMatrix::new(n, n);
    for (i, &v) in diag.iter().enumerate() {
        coo.push(i, i, v);
    }
    CsrMatrix::from_coo(&coo)
}

/// 2×2 symmetric:  [[2, 1], [1, 2]]  →  eigenvalues 1, 3
fn sym2x2() -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 0, 2.0); coo.push(0, 1, 1.0);
    coo.push(1, 0, 1.0); coo.push(1, 1, 2.0);
    CsrMatrix::from_coo(&coo)
}

/// Tridiagonal 1D Laplacian of size n (eigenvalues 2 − 2 cos(kπ/(n+1))).
fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

// ─── PowerIter ───────────────────────────────────────────────────────────────

#[test]
fn power_iter_diagonal_largest() {
    // diag(1,2,3,4) → dominant eigenvalue = 4
    let a = diag_csr(&[1.0, 2.0, 3.0, 4.0]);
    let params = EigenParams::new(1, EigenWhich::LargestMagnitude);
    let res = PowerIter::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 1);
    assert!((res.eigenvalues[0] - 4.0).abs() < 1e-8,
        "expected λ≈4, got {}", res.eigenvalues[0]);
}

#[test]
fn power_iter_sym2x2_dominant() {
    // [[2,1],[1,2]] → dominant eigenvalue = 3
    let a = sym2x2();
    let params = EigenParams::new(1, EigenWhich::LargestMagnitude);
    let res = PowerIter::default().solve(&a, &params).unwrap();
    assert!((res.eigenvalues[0] - 3.0).abs() < 1e-8,
        "expected λ≈3, got {}", res.eigenvalues[0]);
    // eigenvector should satisfy ‖Ax − λx‖ < tol
    assert!(res.residuals[0] < 1e-8);
}

#[test]
fn power_iter_laplacian_10_largest() {
    // Largest eigenvalue of 1D Laplacian n=10: 2 − 2cos(10π/11) ≈ 3.9190
    let n = 10;
    let a = laplacian_1d(n);
    let lam_exact = 2.0 - 2.0 * (10.0 * std::f64::consts::PI / 11.0).cos();
    let params = EigenParams::new(1, EigenWhich::LargestMagnitude);
    let res = PowerIter::default().solve(&a, &params).unwrap();
    assert!((res.eigenvalues[0] - lam_exact).abs() < 1e-7,
        "expected λ≈{lam_exact:.6}, got {:.6}", res.eigenvalues[0]);
}

// ─── SubspaceIter ─────────────────────────────────────────────────────────────

#[test]
fn subspace_iter_diagonal_top3() {
    // diag(1,2,3,4,5) → top 3 eigenvalues: 5, 4, 3
    let a = diag_csr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut params = EigenParams::new(3, EigenWhich::LargestMagnitude);
    params.max_iter = 2000;
    let res = SubspaceIter::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 3);
    let mut got: Vec<f64> = res.eigenvalues.to_vec();
    got.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let expected = [5.0, 4.0, 3.0];
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-6, "expected {e}, got {g}");
    }
}

#[test]
fn subspace_iter_sym2x2_both() {
    // [[2,1],[1,2]] → eigenvalues 1, 3
    let a = sym2x2();
    let mut params = EigenParams::new(2, EigenWhich::LargestMagnitude);
    params.max_iter = 5000;
    let res = SubspaceIter::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 2);
    let mut got: Vec<f64> = res.eigenvalues.to_vec();
    got.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((got[0] - 3.0).abs() < 1e-6, "got {}", got[0]);
    assert!((got[1] - 1.0).abs() < 1e-6, "got {}", got[1]);
}

// ─── InverseIter ─────────────────────────────────────────────────────────────

#[test]
fn inverse_iter_diagonal_smallest() {
    // diag(1,2,3,4) → smallest eigenvalue = 1 (shift = 0)
    let a = diag_csr(&[1.0, 2.0, 3.0, 4.0]);
    let mut params = EigenParams::new(1, EigenWhich::SmallestMagnitude);
    params.tol = 1e-8;
    let res = InverseIter::<f64>::default().solve(&a, &params).unwrap();
    assert!((res.eigenvalues[0] - 1.0).abs() < 1e-7,
        "expected λ≈1, got {}", res.eigenvalues[0]);
}

#[test]
fn inverse_iter_with_shift() {
    // diag(1,2,3,4) with shift 2.9 → should converge to nearest = 3
    let a = diag_csr(&[1.0, 2.0, 3.0, 4.0]);
    let solver = InverseIter::<f64>::new(2.9);
    let params = EigenParams::new(1, EigenWhich::SmallestMagnitude);
    let res = solver.solve(&a, &params).unwrap();
    assert!((res.eigenvalues[0] - 3.0).abs() < 1e-6,
        "expected λ≈3, got {}", res.eigenvalues[0]);
}

#[test]
fn inverse_iter_sym2x2_smallest() {
    // [[2,1],[1,2]] → smallest eigenvalue = 1
    let a = sym2x2();
    let params = EigenParams::new(1, EigenWhich::SmallestMagnitude);
    let res = InverseIter::<f64>::default().solve(&a, &params).unwrap();
    assert!((res.eigenvalues[0] - 1.0).abs() < 1e-7,
        "expected λ≈1, got {}", res.eigenvalues[0]);
}

// ─── RayleighQuotientIter ────────────────────────────────────────────────────

#[test]
fn rqi_diagonal_converges_fast() {
    // shift 3.1 → nearest eigenvalue is unambiguously 3
    let a = diag_csr(&[1.0, 2.0, 3.0, 4.0]);
    let solver = RayleighQuotientIter::<f64>::new(3.1);
    let mut params = EigenParams::new(1, EigenWhich::SmallestMagnitude);
    params.max_iter = 50;
    let res = solver.solve(&a, &params).unwrap();
    assert!((res.eigenvalues[0] - 3.0).abs() < 1e-8,
        "expected λ≈3, got {}", res.eigenvalues[0]);
}

#[test]
fn rqi_sym2x2_dominant() {
    // Start with shift near 3; converge to λ=3
    let a = sym2x2();
    let solver = RayleighQuotientIter::<f64>::new(2.7);
    let mut params = EigenParams::new(1, EigenWhich::LargestMagnitude);
    params.max_iter = 30;
    let res = solver.solve(&a, &params).unwrap();
    assert!((res.eigenvalues[0] - 3.0).abs() < 1e-8,
        "expected λ≈3, got {}", res.eigenvalues[0]);
}

// ─── f32 smoke ───────────────────────────────────────────────────────────────

#[test]
fn power_iter_f32_diagonal() {
    let mut coo = CooMatrix::<f32>::new(3, 3);
    coo.push(0, 0, 5.0); coo.push(1, 1, 3.0); coo.push(2, 2, 1.0);
    let a = CsrMatrix::from_coo(&coo);
    let mut params = EigenParams::<f32>::new(1, EigenWhich::LargestMagnitude);
    params.tol = 1e-5;
    let res = PowerIter::default().solve(&a, &params).unwrap();
    assert!((res.eigenvalues[0] - 5.0_f32).abs() < 1e-4,
        "expected λ≈5, got {}", res.eigenvalues[0]);
}
