//! Sprint 8 / 9 / 10 — eigenvalue solver integration tests.
//!
//! Covers: LanczosIter, ArnoldiIter (S8), GeneralizedEigen / ShiftInvertLanczos (S9),
//!         KrylovSchur, Lobpcg (S10).
//!
//! All matrices have known eigenvalues so results are verified analytically.

use linger::{
    eigen::{
        EigenParams, EigenSolver, EigenWhich,
        LanczosIter, ArnoldiIter,
        GeneralizedEigen, ShiftInvertLanczos,
        KrylovSchur, Lobpcg,
    },
    sparse::{CooMatrix, CsrMatrix},
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn diag_csr(diag: &[f64]) -> CsrMatrix<f64> {
    let n = diag.len();
    let mut coo = CooMatrix::new(n, n);
    for (i, &v) in diag.iter().enumerate() { coo.push(i, i, v); }
    CsrMatrix::from_coo(&coo)
}

/// 1D Laplacian (tridiagonal −1 / 2 / −1), eigenvalues 2−2cos(kπ/(n+1)).
fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

/// Smallest k exact eigenvalues of 1D Laplacian size n (sorted ascending).
fn laplacian_1d_eigs(n: usize, k: usize) -> Vec<f64> {
    let mut vals: Vec<f64> = (1..=n)
        .map(|j| 2.0 - 2.0 * (j as f64 * std::f64::consts::PI / (n + 1) as f64).cos())
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals.truncate(k);
    vals
}

fn sort_asc(mut v: Vec<f64>) -> Vec<f64> { v.sort_by(|a,b| a.partial_cmp(b).unwrap()); v }
fn sort_desc(mut v: Vec<f64>) -> Vec<f64> { v.sort_by(|a,b| b.partial_cmp(a).unwrap()); v }

// ═══════════════════════════════════════════════════════════════════════════
// Sprint 8 — LanczosIter
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lanczos_diagonal_largest_3() {
    let a = diag_csr(&[1.0, 3.0, 5.0, 2.0, 4.0]);
    let params = EigenParams::new(3, EigenWhich::LargestAlgebraic);
    let res = LanczosIter::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 3);
    let got = sort_desc(res.eigenvalues);
    let exp = [5.0, 4.0, 3.0];
    for (g, e) in got.iter().zip(&exp) {
        assert!((g - e).abs() < 1e-7, "LM: expected {e}, got {g}");
    }
}

#[test]
fn lanczos_diagonal_smallest_2() {
    let a = diag_csr(&[1.0, 3.0, 5.0, 2.0, 4.0]);
    let mut params = EigenParams::new(2, EigenWhich::SmallestAlgebraic);
    params.tol = 1e-8;
    let res = LanczosIter::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 2);
    let got = sort_asc(res.eigenvalues);
    assert!((got[0] - 1.0).abs() < 1e-6, "got {}", got[0]);
    assert!((got[1] - 2.0).abs() < 1e-6, "got {}", got[1]);
}

#[test]
fn lanczos_laplacian_20_top3() {
    let n = 20;
    let a = laplacian_1d(n);
    let mut params = EigenParams::new(3, EigenWhich::LargestAlgebraic);
    params.tol = 1e-7;
    let res = LanczosIter::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 3);
    // Largest 3: indices n, n-1, n-2
    let mut exp: Vec<f64> = (1..=n)
        .map(|j| 2.0 - 2.0 * (j as f64 * std::f64::consts::PI / (n + 1) as f64).cos())
        .collect();
    exp.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let got = sort_desc(res.eigenvalues);
    for (g, e) in got.iter().zip(&exp[..3]) {
        assert!((g - e).abs() < 1e-5, "expected {e:.6}, got {g:.6}");
    }
}

#[test]
fn lanczos_residuals_small() {
    // All residuals must satisfy the tolerance
    let a = laplacian_1d(15);
    let params = EigenParams::new(4, EigenWhich::LargestAlgebraic);
    let res = LanczosIter::default().solve(&a, &params).unwrap();
    for r in &res.residuals {
        assert!(*r < 1e-7, "residual {r:.3e} too large");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sprint 8 — ArnoldiIter
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn arnoldi_diagonal_largest_3() {
    let a = diag_csr(&[1.0, 3.0, 5.0, 2.0, 4.0]);
    let params = EigenParams::new(3, EigenWhich::LargestMagnitude);
    let res = ArnoldiIter::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 3);
    let got = sort_desc(res.eigenvalues);
    for (g, e) in got.iter().zip(&[5.0, 4.0, 3.0]) {
        assert!((g - e).abs() < 1e-6, "expected {e}, got {g}");
    }
}

#[test]
fn arnoldi_laplacian_10_top2() {
    let n = 10;
    let a = laplacian_1d(n);
    let mut params = EigenParams::new(2, EigenWhich::LargestMagnitude);
    params.tol = 1e-7;
    let res = ArnoldiIter::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 2);
    let mut exp: Vec<f64> = (1..=n)
        .map(|j| 2.0 - 2.0 * (j as f64 * std::f64::consts::PI / (n + 1) as f64).cos())
        .collect();
    exp.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let got = sort_desc(res.eigenvalues);
    for (g, e) in got.iter().zip(&exp[..2]) {
        assert!((g - e).abs() < 1e-5, "expected {e:.6}, got {g:.6}");
    }
}

#[test]
fn arnoldi_nonsymmetric_upper_triangular() {
    // A = [[3, 1], [0, 1]]  eigenvalues = 3, 1
    let mut coo = CooMatrix::<f64>::new(2, 2);
    coo.push(0, 0, 3.0); coo.push(0, 1, 1.0); coo.push(1, 1, 1.0);
    let a = CsrMatrix::from_coo(&coo);
    let params = EigenParams::new(1, EigenWhich::LargestMagnitude);
    let res = ArnoldiIter::default().solve(&a, &params).unwrap();
    assert!((res.eigenvalues[0] - 3.0).abs() < 1e-6,
        "expected λ≈3, got {}", res.eigenvalues[0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Sprint 9 — ShiftInvertLanczos
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn shift_invert_lanczos_smallest_2() {
    // Shift near 0 → picks smallest eigenvalues
    let a = diag_csr(&[0.5, 1.0, 2.0, 3.0, 4.0]);
    let solver = ShiftInvertLanczos::<f64>::new(0.0);
    let mut params = EigenParams::new(2, EigenWhich::LargestMagnitude); // LM in ν = SM in λ
    params.tol = 1e-7;
    let res = solver.solve(&a, &params).unwrap();
    let got = sort_asc(res.eigenvalues);
    assert!((got[0] - 0.5).abs() < 1e-5, "got {}", got[0]);
    assert!((got[1] - 1.0).abs() < 1e-5, "got {}", got[1]);
}

#[test]
fn shift_invert_lanczos_interior() {
    // Shift = 2.5 → picks eigenvalues nearest 2.5 = {2, 3}
    let a = diag_csr(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let solver = ShiftInvertLanczos::<f64>::new(2.5);
    let mut params = EigenParams::new(2, EigenWhich::LargestMagnitude);
    params.tol = 1e-7;
    let res = solver.solve(&a, &params).unwrap();
    let got = sort_asc(res.eigenvalues);
    assert!((got[0] - 2.0).abs() < 1e-5, "got {}", got[0]);
    assert!((got[1] - 3.0).abs() < 1e-5, "got {}", got[1]);
}

#[test]
fn generalized_eigen_diagonal() {
    // A = diag(2,4,6), B = diag(1,2,3)  → λ = A_ii/B_ii = 2,2,2
    let a = diag_csr(&[2.0, 4.0, 6.0]);
    let b = diag_csr(&[1.0, 2.0, 3.0]);
    let solver = GeneralizedEigen::<f64>::symmetric(1.9, false); // shift near 2
    let mut params = EigenParams::new(2, EigenWhich::SmallestMagnitude);
    params.tol = 1e-6;
    let res = solver.solve_generalized(&a, &b, &params).unwrap();
    for lam in &res.eigenvalues {
        assert!((lam - 2.0).abs() < 1e-4, "expected λ≈2, got {lam}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sprint 10 — KrylovSchur
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn krylov_schur_diagonal_top3() {
    let a = diag_csr(&[1.0, 3.0, 5.0, 2.0, 4.0]);
    let params = EigenParams::new(3, EigenWhich::LargestMagnitude);
    let res = KrylovSchur::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 3);
    let got = sort_desc(res.eigenvalues);
    for (g, e) in got.iter().zip(&[5.0, 4.0, 3.0]) {
        assert!((g - e).abs() < 1e-6, "expected {e}, got {g}");
    }
}

#[test]
fn krylov_schur_laplacian_20_top3() {
    let n = 20;
    let a = laplacian_1d(n);
    let mut params = EigenParams::new(3, EigenWhich::LargestMagnitude);
    params.tol = 1e-7;
    let res = KrylovSchur::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 3);
    let mut exp: Vec<f64> = (1..=n)
        .map(|j| 2.0 - 2.0 * (j as f64 * std::f64::consts::PI / (n + 1) as f64).cos())
        .collect();
    exp.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let got = sort_desc(res.eigenvalues);
    for (g, e) in got.iter().zip(&exp[..3]) {
        assert!((g - e).abs() < 1e-5, "KS: expected {e:.6}, got {g:.6}");
    }
}

#[test]
fn krylov_schur_nonsymmetric() {
    // Upper triangular with eigenvalues on diagonal
    let mut coo = CooMatrix::<f64>::new(3, 3);
    coo.push(0, 0, 5.0); coo.push(0, 1, 2.0); coo.push(0, 2, 1.0);
    coo.push(1, 1, 3.0); coo.push(1, 2, 1.0);
    coo.push(2, 2, 1.0);
    let a = CsrMatrix::from_coo(&coo);
    let params = EigenParams::new(1, EigenWhich::LargestMagnitude);
    let res = KrylovSchur::default().solve(&a, &params).unwrap();
    assert!((res.eigenvalues[0] - 5.0).abs() < 1e-6,
        "expected λ≈5, got {}", res.eigenvalues[0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Sprint 10 — LOBPCG
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lobpcg_diagonal_smallest_3() {
    let a = diag_csr(&[1.0, 3.0, 5.0, 2.0, 4.0]);
    let mut params = EigenParams::new(3, EigenWhich::SmallestAlgebraic);
    params.tol = 1e-8;
    params.max_iter = 2000;
    let res = Lobpcg::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 3);
    let got = sort_asc(res.eigenvalues);
    for (g, e) in got.iter().zip(&[1.0, 2.0, 3.0]) {
        assert!((g - e).abs() < 1e-6, "LOBPCG: expected {e}, got {g}");
    }
}

#[test]
fn lobpcg_laplacian_10_bottom3() {
    let n = 10;
    let a = laplacian_1d(n);
    let expected = laplacian_1d_eigs(n, 3);
    let mut params = EigenParams::new(3, EigenWhich::SmallestAlgebraic);
    params.tol = 1e-7;
    params.max_iter = 3000;
    let res = Lobpcg::default().solve(&a, &params).unwrap();
    assert_eq!(res.converged, 3);
    let got = sort_asc(res.eigenvalues);
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-5, "LOBPCG Laplacian: expected {e:.6}, got {g:.6}");
    }
}

#[test]
fn lobpcg_single_eig_spd() {
    // 4×4 SPD diagonal — single smallest
    let a = diag_csr(&[0.3, 1.2, 2.5, 4.0]);
    let mut params = EigenParams::new(1, EigenWhich::SmallestAlgebraic);
    params.tol = 1e-9;
    let res = Lobpcg::default().solve(&a, &params).unwrap();
    assert!((res.eigenvalues[0] - 0.3).abs() < 1e-7,
        "expected λ≈0.3, got {}", res.eigenvalues[0]);
}
