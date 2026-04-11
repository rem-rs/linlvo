//! Sprint 13 — direct solver integration tests.
//!
//! Covers:
//! - `SparseLu`:     exact factorisation, partial pivoting, ordering
//! - `SparseCholesky`: SPD factorisation
//! - `DirectSolverPrecond`: direct solver as Krylov preconditioner
//! - Ordering: RCM and COLAMD are valid permutations

use linger::{
    direct::{DirectSolver, DirectOptions, DirectSolverPrecond, SparseLu, SparseCholesky,
             ordering::{OrderingMethod, rcm, colamd, permute_symmetric}},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, LinearOperator,
    KrylovSolver, SolverError, SolverParams, VerboseLevel,
    iterative::ConjugateGradient,
    Gmres,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

/// 1-D Laplacian (tridiagonal −1 / 2 / −1), which is SPD.
fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

/// Non-symmetric tridiagonal: upper diagonal = 2, lower = 1.
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
    let ax_s = ax.as_slice();
    let b_s  = b.as_slice();
    let norm_r: f64 = ax_s.iter().zip(b_s).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b_s.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    if norm_b < 1e-300 { norm_r } else { norm_r / norm_b }
}

// ═══════════════════════════════════════════════════════════════════════════
// SparseLu — basic correctness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lu_tiny_system() {
    // 2×2: [4 1; 2 3] x = [5; 10]  → x = [1; 1] ... verify Ax ≈ b
    let mut coo = CooMatrix::<f64>::new(2, 2);
    coo.push(0, 0, 4.0); coo.push(0, 1, 1.0);
    coo.push(1, 0, 2.0); coo.push(1, 1, 3.0);
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![5.0, 10.0]);
    let mut x = DenseVec::zeros(2);

    let mut solver = SparseLu::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Natural,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    let rel = relative_residual(&a, &x, &b);
    assert!(rel < 1e-12, "relative residual = {rel:.3e}");
}

#[test]
fn lu_laplacian_1d_n10() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let mut solver = SparseLu::<f64>::default();
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    let rel = relative_residual(&a, &x, &b);
    assert!(rel < 1e-10, "rel residual = {rel:.3e}");
}

#[test]
fn lu_nonsymmetric() {
    let n = 8;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());
    let mut x = DenseVec::zeros(n);

    let mut solver = SparseLu::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Natural,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    let rel = relative_residual(&a, &x, &b);
    assert!(rel < 1e-10, "rel residual = {rel:.3e}");
}

#[test]
fn lu_singular_matrix_reports_breakdown() {
    // Duplicate rows => rank-deficient matrix.
    let mut coo = CooMatrix::<f64>::new(3, 3);
    coo.push(0, 0, 1.0); coo.push(0, 1, 2.0);
    coo.push(1, 0, 1.0); coo.push(1, 1, 2.0); // duplicate of row 0
    coo.push(2, 2, 1.0);
    let a = CsrMatrix::from_coo(&coo);

    let mut solver = SparseLu::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Natural,
        ..Default::default()
    });

    match solver.factor(&a) {
        Err(SolverError::NumericalBreakdown { .. }) => {}
        other => panic!("expected NumericalBreakdown for singular LU factorization, got {other:?}"),
    }
}

#[test]
fn lu_multiple_rhs() {
    let n = 6;
    let a = laplacian_1d(n);
    let b1 = DenseVec::from_vec(vec![1.0f64; n]);
    let b2 = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());

    let mut solver = SparseLu::<f64>::default();
    solver.factor(&a).unwrap();

    let mut x1 = DenseVec::zeros(n);
    let mut x2 = DenseVec::zeros(n);
    solver.solve(&b1, &mut x1).unwrap();
    solver.solve(&b2, &mut x2).unwrap();

    assert!(relative_residual(&a, &x1, &b1) < 1e-10);
    assert!(relative_residual(&a, &x2, &b2) < 1e-10);
}

#[test]
fn lu_with_rcm_ordering() {
    let n = 15;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let mut solver = SparseLu::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Rcm,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    let rel = relative_residual(&a, &x, &b);
    assert!(rel < 1e-10, "rel residual with RCM = {rel:.3e}");
}

#[test]
fn lu_with_colamd_ordering() {
    let n = 12;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let mut solver = SparseLu::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Colamd,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    let rel = relative_residual(&a, &x, &b);
    assert!(rel < 1e-10, "rel residual with COLAMD = {rel:.3e}");
}

#[test]
fn lu_reset_and_refactor() {
    // Factor, reset, then factor again with the same matrix.
    let n = 6;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);

    let mut solver = SparseLu::<f64>::default();
    solver.factor(&a).unwrap();
    solver.reset_factors();
    solver.factor(&a).unwrap();

    let mut x = DenseVec::zeros(n);
    solver.solve(&b, &mut x).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

#[test]
fn lu_f32_precision() {
    let n = 6;
    let mut coo = CooMatrix::<f32>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0f32);
        if i > 0     { coo.push(i, i - 1, -1.0f32); }
        if i + 1 < n { coo.push(i, i + 1, -1.0f32); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![1.0f32; n]);
    let mut x = DenseVec::zeros(n);

    let mut solver = SparseLu::<f32>::default();
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    // Compute relative residual in f32.
    let mut ax = DenseVec::zeros(n);
    a.apply(&x, &mut ax);
    let r: f32 = ax.as_slice().iter().zip(b.as_slice()).map(|(a,b)| (a-b).powi(2)).sum::<f32>().sqrt();
    let nb: f32 = b.as_slice().iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
    assert!(r / nb < 1e-4, "f32 relative residual = {:.3e}", r / nb);
}

// ═══════════════════════════════════════════════════════════════════════════
// SparseCholesky
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cholesky_laplacian_1d_n8() {
    let n = 8;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let mut solver = SparseCholesky::<f64>::default();
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    let rel = relative_residual(&a, &x, &b);
    assert!(rel < 1e-10, "Cholesky rel residual = {rel:.3e}");
}

#[test]
fn cholesky_diagonal_spd() {
    // A = diag(1, 2, 3, 4) — trivially SPD.
    let diag = [1.0, 2.0, 3.0, 4.0];
    let n = diag.len();
    let mut coo = CooMatrix::<f64>::new(n, n);
    for (i, &d) in diag.iter().enumerate() { coo.push(i, i, d); }
    let a = CsrMatrix::from_coo(&coo);

    let b = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut x = DenseVec::zeros(n);

    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Natural,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    // For diagonal A: x[i] = b[i] / diag[i] = 1 for all i.
    for (i, &xi) in x.as_slice().iter().enumerate() {
        assert!((xi - 1.0).abs() < 1e-12, "x[{i}] = {xi}");
    }
}

#[test]
fn cholesky_with_rcm() {
    let n = 12;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());
    let mut x = DenseVec::zeros(n);

    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Rcm,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();

    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

#[test]
fn cholesky_not_spd_returns_error() {
    // A = [−1, 0; 0, 2] is not SPD (negative diagonal).
    let mut coo = CooMatrix::<f64>::new(2, 2);
    coo.push(0, 0, -1.0);
    coo.push(1, 1,  2.0);
    let a = CsrMatrix::from_coo(&coo);

    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Natural,
        ..Default::default()
    });
    assert!(solver.factor(&a).is_err(), "expected error for non-SPD matrix");
}

// ═══════════════════════════════════════════════════════════════════════════
// DirectSolverPrecond — wrapping LU as preconditioner
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lu_precond_gmres_converges() {
    // GMRES + exact LU precond should converge in 1 iteration.
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let precond = DirectSolverPrecond::new(SparseLu::<f64>::default(), &a).unwrap();

    let params = SolverParams {
        rtol: 1e-10,
        max_iter: 20,
        verbose: VerboseLevel::Silent,
        ..Default::default()
    };
    let result = Gmres::new(20)
        .solve(&a, Some(&precond), &b, &mut x, &params)
        .unwrap();

    assert!(result.converged, "GMRES+LU precond did not converge");
    assert!(result.iterations <= 5,
        "expected ≤5 iterations with exact precond, got {}", result.iterations);
}

#[test]
fn cholesky_precond_cg_converges() {
    // CG + exact Cholesky precond on SPD system: should converge in 1 iteration.
    let n = 12;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let precond = DirectSolverPrecond::new(SparseCholesky::<f64>::default(), &a).unwrap();

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
        "expected ≤5 iterations with Cholesky precond, got {}", result.iterations);
}

// ═══════════════════════════════════════════════════════════════════════════
// Ordering — validity tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rcm_valid_permutation() {
    let a = laplacian_1d(20);
    let perm = rcm(&a);
    assert_eq!(perm.len(), 20);
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..20usize).collect::<Vec<_>>());
}

#[test]
fn colamd_valid_permutation() {
    let a = laplacian_1d(20);
    let perm = colamd(&a);
    assert_eq!(perm.len(), 20);
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..20usize).collect::<Vec<_>>());
}

#[test]
fn permute_symmetric_preserves_matrix() {
    // Permuting with the identity permutation should give the same matrix.
    let n = 5;
    let a = laplacian_1d(n);
    let perm: Vec<usize> = (0..n).collect();
    let ap = permute_symmetric(&a, &perm);

    // Check all entries match.
    for i in 0..n {
        for k in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let j = a.col_idx()[k];
            let v = a.values()[k];
            // Find (i, j) in ap.
            let found = (ap.row_ptr()[i]..ap.row_ptr()[i + 1])
                .any(|kk| ap.col_idx()[kk] == j && (ap.values()[kk] - v).abs() < 1e-15);
            assert!(found, "entry ({i},{j}) not preserved after identity permutation");
        }
    }
}
