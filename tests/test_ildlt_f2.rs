//! Integration tests for F2: ILDLᵀ(0) preconditioner.

use linger::{
    IldltPrecond, Preconditioner,
    iterative::ConjugateGradient,
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, SolverParams, VerboseLevel,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn params_silent(rtol: f64, max_iter: usize) -> SolverParams {
    SolverParams { rtol, max_iter, verbose: VerboseLevel::Silent, ..Default::default() }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// 1. ILDLᵀ preconditioned CG converges on 1D Laplacian.
#[test]
fn ildlt_pcg_converges_1d() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let precond = IldltPrecond::<f64>::from_csr(&a).unwrap();
    let p = params_silent(1e-8, 500);
    let mut x = DenseVec::zeros(n);
    let result = ConjugateGradient::<f64>::default()
        .solve(&a, Some(&precond), &b, &mut x, &p)
        .unwrap();
    assert!(result.converged,
        "ILDLt-PCG did not converge in {} iters", result.iterations);
}

/// 2. ILDLᵀ preconditioned CG converges faster than no preconditioner.
///    (Checks iteration count is reduced.)
#[test]
fn ildlt_fewer_iters_than_unpreconditioned() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let p = params_silent(1e-8, 2000);

    // Without preconditioner.
    let mut x_none = DenseVec::zeros(n);
    let r_none = ConjugateGradient::<f64>::default()
        .solve(&a, None, &b, &mut x_none, &p)
        .unwrap();

    // With ILDLᵀ preconditioner.
    let precond = IldltPrecond::<f64>::from_csr(&a).unwrap();
    let mut x_pre = DenseVec::zeros(n);
    let r_pre = ConjugateGradient::<f64>::default()
        .solve(&a, Some(&precond), &b, &mut x_pre, &p)
        .unwrap();

    assert!(r_pre.converged, "ILDLt-PCG did not converge");
    assert!(r_none.converged, "unpreconditioned CG did not converge");
    assert!(r_pre.iterations <= r_none.iterations,
        "ILDLt-PCG ({} iters) did not improve over unpreconditioned ({} iters)",
        r_pre.iterations, r_none.iterations);
}

/// 3. D entries are positive for SPD matrix.
#[test]
fn ildlt_d_positive_for_spd() {
    let n = 20;
    let a = laplacian_1d(n);
    let precond = IldltPrecond::<f64>::from_csr(&a).unwrap();
    assert!(precond.d().iter().all(|&v| v > 0.0),
        "D should be positive for SPD tridiagonal");
}

/// 4. Preconditioner is linear: M⁻¹(alpha*x) = alpha * M⁻¹(x).
#[test]
fn ildlt_linearity() {
    let n = 15;
    let a = laplacian_1d(n);
    let precond = IldltPrecond::<f64>::from_csr(&a).unwrap();
    let alpha = 2.5f64;
    let x_vec: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3 - 1.5).collect();
    let ax_vec: Vec<f64> = x_vec.iter().map(|&v| alpha * v).collect();

    let x  = DenseVec::from_vec(x_vec);
    let ax = DenseVec::from_vec(ax_vec);
    let mut y  = DenseVec::zeros(n);
    let mut ay = DenseVec::zeros(n);
    precond.apply_precond(&x, &mut y);
    precond.apply_precond(&ax, &mut ay);

    let err = y.as_slice().iter().zip(ay.as_slice())
        .map(|(yi, ayi)| (alpha * yi - ayi).abs())
        .fold(0.0f64, f64::max);
    assert!(err < 1e-12, "ILDLt not linear: max err={err}");
}

/// 5. Output is always finite.
#[test]
fn ildlt_output_finite() {
    let n = 30;
    let a = laplacian_1d(n);
    let precond = IldltPrecond::<f64>::from_csr(&a).unwrap();
    let x = DenseVec::from_vec(vec![1.0f64; n]);
    let mut y = DenseVec::zeros(n);
    precond.apply_precond(&x, &mut y);
    assert!(y.as_slice().iter().all(|v| v.is_finite()),
        "ILDLt output contains non-finite");
}

/// 6. nrows() matches matrix size.
#[test]
fn ildlt_nrows_matches() {
    for &n in &[5usize, 10, 20] {
        let a = laplacian_1d(n);
        let precond = IldltPrecond::<f64>::from_csr(&a).unwrap();
        assert_eq!(precond.nrows(), n, "nrows mismatch for n={n}");
    }
}

/// 7. Works on larger system (n=200).
#[test]
fn ildlt_pcg_n200() {
    let n = 200;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let precond = IldltPrecond::<f64>::from_csr(&a).unwrap();
    let p = params_silent(1e-8, 1000);
    let mut x = DenseVec::zeros(n);
    let result = ConjugateGradient::<f64>::default()
        .solve(&a, Some(&precond), &b, &mut x, &p)
        .unwrap();
    assert!(result.converged,
        "ILDLt-PCG n=200 did not converge in {} iters", result.iterations);
}

/// 8. Error on non-square matrix.
#[test]
fn ildlt_error_non_square() {
    let mut coo = CooMatrix::<f64>::new(3, 4);
    for i in 0..3 { coo.push(i, i, 1.0); }
    let a = CsrMatrix::from_coo(&coo);
    assert!(IldltPrecond::<f64>::from_csr(&a).is_err(),
        "expected error for non-square matrix");
}
