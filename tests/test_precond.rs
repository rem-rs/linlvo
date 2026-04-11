//! Sprint 2 — preconditioner tests.
//!
//! Verifies that Jacobi, SOR, SSOR, and ILU(0) produce correct results
//! when applied to known systems.

mod common;

use linger::{
    precond::{Ilu0Precond, JacobiPrecond, SorPrecond, SsorPrecond},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, Preconditioner,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn diag_matrix(diag: &[f64]) -> CsrMatrix<f64> {
    let n = diag.len();
    let mut coo = CooMatrix::new(n, n);
    for (i, &v) in diag.iter().enumerate() { coo.push(i, i, v); }
    CsrMatrix::from_coo(&coo)
}

// ─── Jacobi ──────────────────────────────────────────────────────────────────

#[test]
fn jacobi_diagonal_system() {
    // A = diag(1, 2, 3, 4)  →  M⁻¹ = diag(1, 0.5, 1/3, 0.25)
    let a = diag_matrix(&[1.0, 2.0, 3.0, 4.0]);
    let jac = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let x = DenseVec::from_vec(vec![4.0, 4.0, 3.0, 4.0]);
    let mut y = DenseVec::zeros(4);
    jac.apply_precond(&x, &mut y);
    // y[i] = x[i] / a[i,i]
    assert!((y[0] - 4.0).abs() < 1e-14);
    assert!((y[1] - 2.0).abs() < 1e-14);
    assert!((y[2] - 1.0).abs() < 1e-14);
    assert!((y[3] - 1.0).abs() < 1e-14);
}

#[test]
fn jacobi_near_zero_diagonal_fails() {
    let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
    coo.push(0, 0, 1.0);
    coo.push(1, 1, 1e-100); // near-zero
    coo.push(2, 2, 1.0);
    let a = CsrMatrix::from_coo(&coo);
    assert!(JacobiPrecond::<f64>::from_csr(&a).is_err());
}

#[test]
fn jacobi_on_poisson_1d() {
    let (a, _, _) = common::make_poisson_1d::<f64>(10);
    let jac = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    // All diagonal entries are 2 → inv_diag = 0.5
    let x = DenseVec::from_vec(vec![1.0f64; 10]);
    let mut y = DenseVec::zeros(10);
    jac.apply_precond(&x, &mut y);
    for &v in y.as_slice() {
        assert!((v - 0.5).abs() < 1e-14, "expected 0.5, got {v}");
    }
}

// ─── SOR ─────────────────────────────────────────────────────────────────────

#[test]
fn sor_omega_out_of_range_fails() {
    let a = diag_matrix(&[1.0, 2.0]);
    assert!(SorPrecond::<f64>::from_csr(&a, 0.0).is_err());
    assert!(SorPrecond::<f64>::from_csr(&a, 2.0).is_err());
    assert!(SorPrecond::<f64>::from_csr(&a, -0.5).is_err());
}

#[test]
fn sor_diagonal_system_omega1() {
    // For diagonal A, SOR with ω=1 is identical to Jacobi.
    let a = diag_matrix(&[2.0, 4.0, 1.0]);
    let sor = SorPrecond::<f64>::from_csr(&a, 1.0).unwrap();
    let x = DenseVec::from_vec(vec![2.0, 4.0, 3.0]);
    let mut y = DenseVec::zeros(3);
    sor.apply_precond(&x, &mut y);
    assert!((y[0] - 1.0).abs() < 1e-14);
    assert!((y[1] - 1.0).abs() < 1e-14);
    assert!((y[2] - 3.0).abs() < 1e-14);
}

// ─── SSOR ─────────────────────────────────────────────────────────────────────

#[test]
fn ssor_omega_out_of_range_fails() {
    let a = diag_matrix(&[1.0, 2.0]);
    assert!(SsorPrecond::<f64>::from_csr(&a, 0.0).is_err());
    assert!(SsorPrecond::<f64>::from_csr(&a, 2.0).is_err());
}

#[test]
fn ssor_diagonal_system() {
    // For diagonal A, SSOR with ω=1:
    // Phase 1: forward solve  (D) z₁ = r  → z₁ = r / d
    // Phase 2: diagonal scale z₂ = (1/(2-1)) · d · z₁ = r
    // Phase 3: backward solve (D) z  = z₂ → z = z₂ / d = r / d
    // So SSOR with ω=1 on diagonal = Jacobi.
    let a = diag_matrix(&[2.0, 4.0]);
    let ssor = SsorPrecond::<f64>::from_csr(&a, 1.0).unwrap();
    let x = DenseVec::from_vec(vec![2.0, 4.0]);
    let mut y = DenseVec::zeros(2);
    ssor.apply_precond(&x, &mut y);
    assert!((y[0] - 1.0).abs() < 1e-13, "y[0]={}", y[0]);
    assert!((y[1] - 1.0).abs() < 1e-13, "y[1]={}", y[1]);
}

#[test]
fn ssor_is_spd_for_spd_input() {
    // For SPD A, SSOR is also SPD (positive definite preconditioner).
    // Verify: x^T (SSOR^{-1}) x = <x, M^{-1}x> > 0 for all nonzero x.
    let (a, _, _) = common::make_poisson_1d::<f64>(5);
    let ssor = SsorPrecond::<f64>::from_csr(&a, 1.0).unwrap();
    let x = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
    let mut y = DenseVec::zeros(5);
    ssor.apply_precond(&x, &mut y);
    let xty: f64 = x.as_slice().iter().zip(y.as_slice()).map(|(&a, &b)| a * b).sum();
    assert!(xty > 0.0, "SSOR must be positive definite; x^T M^{{-1}} x = {xty}");
}

// ─── ILU(0) ──────────────────────────────────────────────────────────────────

#[test]
fn ilu0_diagonal_system_exact() {
    // For a diagonal system, ILU(0) = exact inverse.
    let a = diag_matrix(&[2.0, 3.0, 5.0]);
    let ilu = Ilu0Precond::<f64>::from_csr(&a).unwrap();
    let b = DenseVec::from_vec(vec![4.0, 9.0, 10.0]);
    let mut y = DenseVec::zeros(3);
    ilu.apply_precond(&b, &mut y);
    assert!((y[0] - 2.0).abs() < 1e-14, "y[0]={}", y[0]);
    assert!((y[1] - 3.0).abs() < 1e-14, "y[1]={}", y[1]);
    assert!((y[2] - 2.0).abs() < 1e-14, "y[2]={}", y[2]);
}

#[test]
fn ilu0_missing_diagonal_fails() {
    // Matrix with no diagonal entry in row 1.
    let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
    coo.push(0, 0, 2.0);
    coo.push(1, 0, -1.0); // row 1 has no diagonal
    coo.push(2, 2, 2.0);
    let a = CsrMatrix::from_coo(&coo);
    assert!(Ilu0Precond::<f64>::from_csr(&a).is_err());
}

#[test]
fn ilu0_tridiagonal_residual_small() {
    // ILU(0) on 1-D Poisson tridiagonal is exact (no fill-in).
    // Solve A (ILU^{-1} b) ≈ b — the preconditioned system has small residual.
    let n = 10;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let ilu = Ilu0Precond::<f64>::from_csr(&a).unwrap();

    // For a tridiagonal, ILU(0) is the exact LU factorisation.
    // So for b = A e_0, applying ILU^{-1} should give back e_0.
    let mut b = vec![0.0f64; n];
    b[0] = 1.0; // b = e_0 = first standard basis vector
    // A e_0 = [2, -1, 0, ...] but we set b = e_0 directly.
    // ILU^{-1} (A e_0) = e_0  iff ILU = A (exact)
    // Instead: verify ‖A (ILU^{-1} e_i) - e_i‖ is small for random b.
    let b_vec = DenseVec::from_vec(b.clone());
    let mut y = DenseVec::zeros(n);
    ilu.apply_precond(&b_vec, &mut y);

    // Compute A y; should ≈ b
    let mut ay = vec![0.0f64; n];
    a.spmv(y.as_slice(), &mut ay);
    let err: f64 = ay.iter().zip(&b).map(|(&ai, &bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
    assert!(err < 1e-12, "ILU(0) should be exact on tridiagonal; err={err:.3e}");
}
