//! Sprint 11 / 12 — integration tests.
//!
//! Covers:
//! - S11: ComplexScalar trait, LanczosSvd
//! - S12: QuadraticEigen (QEP), NepNewton

use linger::{
    Complex,
    LanczosSvd,
    QuadraticEigen,
    NonlinearOperator, NepNewton,
    EigenParams, EigenWhich,
    DenseVec, Vector,
    sparse::{CooMatrix, CsrMatrix},
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn diag_csr(diag: &[f64]) -> CsrMatrix<f64> {
    let n = diag.len();
    let mut coo = CooMatrix::new(n, n);
    for (i, &v) in diag.iter().enumerate() { coo.push(i, i, v); }
    CsrMatrix::from_coo(&coo)
}

/// 1D Laplacian (tridiagonal −1 / 2 / −1).
fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

// ═══════════════════════════════════════════════════════════════════════════
// Sprint 11 — ComplexScalar
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn complex_scalar_ops() {
    use linger::ComplexScalar;
    let z: Complex<f64> = Complex::new(3.0, 4.0);
    // |z| = 5
    assert!((ComplexScalar::abs(z) - 5.0).abs() < 1e-12, "abs");
    // conj(z) = 3 - 4i
    let c = ComplexScalar::conj(z);
    assert!((c.re - 3.0).abs() < 1e-12 && (c.im + 4.0).abs() < 1e-12, "conj");
    // re / im
    assert_eq!(ComplexScalar::real(z), 3.0);
    assert_eq!(ComplexScalar::imag(z), 4.0);
    // from_f64
    let w: Complex<f64> = ComplexScalar::from_f64(2.5);
    assert_eq!(w.re, 2.5); assert_eq!(w.im, 0.0);
    // sqrt(−1) = i
    let mi: Complex<f64> = Complex::new(-1.0, 0.0);
    let sq = ComplexScalar::sqrt(mi);
    assert!(sq.re.abs() < 1e-12 && (sq.im - 1.0).abs() < 1e-12, "sqrt(-1)={sq:?}");
    // machine_epsilon
    assert_eq!(<Complex<f64> as ComplexScalar>::machine_epsilon(), f64::EPSILON);
    // Real types are also ComplexScalar
    assert_eq!(<f64 as ComplexScalar>::imag(1.5_f64), 0.0);
    assert_eq!(<f64 as ComplexScalar>::conj(2.0_f64), 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Sprint 11 — LanczosSvd
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn svd_diagonal_3x3() {
    // A = diag(3, 2, 1) — singular values = diagonal entries (sorted desc)
    let a = diag_csr(&[3.0, 1.0, 2.0]);
    let svd = LanczosSvd::default();
    // Request 2 (LanczosIter requires nev < n, so max is n-1 = 2 for 3x3)
    let res = svd.solve(&a, 2, 1e-9, 500, true).unwrap();
    let sv = res.singular_values.clone();
    // Should be ≈ [3, 2] descending
    assert!((sv[0] - 3.0).abs() < 1e-6, "σ₁={}", sv[0]);
    assert!((sv[1] - 2.0).abs() < 1e-6, "σ₂={}", sv[1]);
}

#[test]
fn svd_diagonal_residuals() {
    let a = diag_csr(&[4.0, 2.0, 1.0, 0.5]);
    let svd = LanczosSvd::default();
    let res = svd.solve(&a, 3, 1e-9, 500, true).unwrap();
    // ‖Aᵀ uᵢ − σᵢ vᵢ‖ < tol for all converged pairs
    for (i, r) in res.residuals.iter().enumerate() {
        assert!(*r < 1e-6, "residual[{}] = {:.3e}", i, r);
    }
}

#[test]
fn svd_laplacian_top2() {
    // For symmetric A, singular values = |eigenvalues|
    let n = 10;
    let a = laplacian_1d(n);
    let svd = LanczosSvd::default();
    let res = svd.solve(&a, 2, 1e-9, 1000, false).unwrap();
    // Largest 2 eigenvalues of Laplacian (which equal singular values for SPD)
    let mut exp_eigs: Vec<f64> = (1..=n)
        .map(|j| 2.0 - 2.0 * (j as f64 * std::f64::consts::PI / (n + 1) as f64).cos())
        .collect();
    exp_eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let sv = &res.singular_values;
    for (i, (&got, &exp)) in sv.iter().zip(&exp_eigs[..2]).enumerate() {
        assert!((got - exp).abs() < 1e-5,
            "σ[{}]: expected {:.6}, got {:.6}", i, exp, got);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sprint 12 — QuadraticEigen (QEP)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn qep_overdamped_1dof() {
    // (K + λC + λ²M) x = 0, scalars k=1, c=5, m=1
    // λ = (-5 ± √(25-4))/2 = (-5 ± √21)/2 ≈ -0.21 and -4.79
    // These are real eigenvalues (overdamped), Arnoldi should find them.
    let k = diag_csr(&[1.0]);
    let c = diag_csr(&[5.0]);
    let m = diag_csr(&[1.0]);
    let qep = QuadraticEigen::new(1);
    let mut params = EigenParams::new(1, EigenWhich::LargestMagnitude);
    params.max_iter = 500;
    let res = qep.solve(&k, &c, &m, &params).unwrap();
    // The largest-magnitude real eigenvalue of the 2×2 companion should be ≈ -4.79
    let lam = res.eigenvalues[0];
    // Verify with QEP residual
    assert!(res.residuals[0] < 0.1,
        "QEP residual = {:.3e}, λ = {:.4}", res.residuals[0], lam);
}

#[test]
fn qep_overdamped_2dof() {
    // K = diag(1,4), C = diag(5,8), M = diag(1,1)
    // Each DOF independent: λ²+5λ+1=0 and λ²+8λ+4=0
    // Roots: λ₁ = (-5±√21)/2 ≈ -0.21, -4.79
    //        λ₂ = (-8±√48)/2 ≈ -0.54, -7.46
    let k = diag_csr(&[1.0, 4.0]);
    let c = diag_csr(&[5.0, 8.0]);
    let m = diag_csr(&[1.0, 1.0]);
    let qep = QuadraticEigen::new(2);
    let mut params = EigenParams::new(2, EigenWhich::LargestMagnitude);
    params.max_iter = 1000;
    let res = qep.solve(&k, &c, &m, &params).unwrap();
    for (i, r) in res.residuals.iter().enumerate() {
        assert!(*r < 0.1, "QEP residual[{}] = {:.3e}", i, r);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sprint 12 — NepNewton
// ═══════════════════════════════════════════════════════════════════════════

/// T(λ) = A − λ I  (standard eigenvalue problem as a NEP)
struct StandardNep {
    /// diagonal entries of A
    diag: Vec<f64>,
}

impl NonlinearOperator<f64> for StandardNep {
    fn nrows(&self) -> usize { self.diag.len() }

    fn apply_t(&self, lam: f64, v: &DenseVec<f64>, out: &mut DenseVec<f64>) {
        let vs = v.as_slice();
        let os = out.as_mut_slice();
        for (i, (&d, &vi)) in self.diag.iter().zip(vs.iter()).enumerate() {
            os[i] = (d - lam) * vi;
        }
    }

    fn apply_dt(&self, _lam: f64, v: &DenseVec<f64>, out: &mut DenseVec<f64>) {
        // T'(λ) = -I, so T'(λ) v = -v
        let vs = v.as_slice();
        let os = out.as_mut_slice();
        for (i, &vi) in vs.iter().enumerate() { os[i] = -vi; }
    }
}

#[test]
fn nep_newton_linear_diagonal() {
    // T(λ) = diag(3,1,2) − λ I, largest eigenvalue = 3.  Shift near 3.
    let nep_op = StandardNep { diag: vec![3.0, 1.0, 2.0] };
    let solver = NepNewton::<f64>::new(2.9, 1e-9, 200);
    let (lam, x) = solver.solve(&nep_op).unwrap();
    assert!((lam - 3.0).abs() < 1e-6, "expected λ≈3, got {}", lam);
    // x should be the unit vector e₀
    assert!(x[0].abs() > 0.99, "x[0] should be ≈ 1, got {}", x[0]);
}

#[test]
fn nep_newton_linear_smallest() {
    // Shift near smallest eigenvalue = 1
    let nep_op = StandardNep { diag: vec![3.0, 1.0, 2.0] };
    let solver = NepNewton::<f64>::new(1.1, 1e-9, 200);
    let (lam, _x) = solver.solve(&nep_op).unwrap();
    assert!((lam - 1.0).abs() < 1e-6, "expected λ≈1, got {}", lam);
}

#[test]
fn nep_newton_convergence_check() {
    // Verify final residual ‖T(λ) x‖ < tol
    let diag = vec![5.0, 2.0, 1.0, 4.0, 3.0];
    let nep_op = StandardNep { diag: diag.clone() };
    for &target in &[5.0_f64, 4.0, 3.0, 2.0, 1.0] {
        let shift = target + 0.1;
        let solver = NepNewton::<f64>::new(shift, 1e-9, 300);
        let (lam, x) = solver.solve(&nep_op).unwrap();
        // Check residual: T(λ) x = (A - λI) x
        let mut r = DenseVec::zeros(diag.len());
        nep_op.apply_t(lam, &x, &mut r);
        let res = r.norm2();
        assert!(res < 1e-7, "target={target}, λ={lam:.6}, ‖r‖={res:.3e}");
        assert!((lam - target).abs() < 1e-5,
            "expected λ≈{target}, got {lam}");
    }
}
