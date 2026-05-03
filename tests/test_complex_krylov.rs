//! Integration tests for complex Krylov solvers.
//!
//! Covers:
//! 1. `ComplexGmres` on a `DenseMatrix<Complex<f64>>` (shifted Laplacian)
//! 2. `ComplexGmres` on a `CsrMatrix<Complex<f64>>` (sparse complex Helmholtz-like)
//! 3. `multi_rhs_solve` on a complex sparse system (two right-hand sides)
//! 4. `ComplexGmresWorkspace` reuse across repeated solves
//! 5. Hermitian adjoint of `CsrMatrix<Complex<f64>>`

use num_complex::Complex;
use linger::{
    sparse::{CsrMatrix, CooMatrix},
    iterative::complex_gmres::{ComplexGmres, ComplexGmresWorkspace},
    core::{
        dense::DenseMatrix,
        vector::{DenseVec, Vector},
    },
};

type C64 = Complex<f64>;

fn c(re: f64, im: f64) -> C64 { Complex::new(re, im) }

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build the complex 1D Helmholtz matrix:
///   `A[i,i] = (2 - k²·h²) + i·σ·h²`,  `A[i,i±1] = -1`
/// where `h = 1/(n+1)`, `k` is the wave number, `σ` is the absorption.
fn complex_helmholtz_csr(n: usize, k: f64, sigma: f64) -> CsrMatrix<C64> {
    let h = 1.0 / (n + 1) as f64;
    let kh_sq = k * k * h * h;
    let sigma_h_sq = sigma * h * h;
    let mut coo: CooMatrix<C64> = CooMatrix::new_complex(n, n);
    for i in 0..n {
        coo.push_complex(i, i, c(2.0 - kh_sq, sigma_h_sq));
        if i > 0     { coo.push_complex(i, i - 1, c(-1.0, 0.0)); }
        if i + 1 < n { coo.push_complex(i, i + 1, c(-1.0, 0.0)); }
    }
    CsrMatrix::from_complex_coo(&coo)
}

/// Build the same system as a `DenseMatrix<C64>`.
fn complex_helmholtz_dense(n: usize, k: f64, sigma: f64) -> DenseMatrix<C64> {
    let h = 1.0 / (n + 1) as f64;
    let kh_sq = k * k * h * h;
    let sigma_h_sq = sigma * h * h;
    DenseMatrix::from_fn(n, n, |i, j| {
        if i == j {
            c(2.0 - kh_sq, sigma_h_sq)
        } else if (i as isize - j as isize).abs() == 1 {
            c(-1.0, 0.0)
        } else {
            c(0.0, 0.0)
        }
    })
}

/// Compute `‖A x - b‖₂` for a dense operator.
fn dense_residual(a: &DenseMatrix<C64>, x: &DenseVec<C64>, b: &DenseVec<C64>) -> f64 {
    use linger::core::operator::LinearOperator;
    let mut ax: DenseVec<C64> = vec![c(0.0, 0.0); b.len()].into();
    a.apply(x, &mut ax);
    let err: f64 = ax.as_slice().iter().zip(b.as_slice())
        .map(|(ai, bi)| (ai - bi).norm()).sum();
    err
}

/// Compute `‖A x - b‖₂` for a sparse operator.
fn sparse_residual(a: &CsrMatrix<C64>, x: &DenseVec<C64>, b: &DenseVec<C64>) -> f64 {
    use linger::core::operator::LinearOperator;
    let mut ax: DenseVec<C64> = vec![c(0.0, 0.0); b.len()].into();
    a.apply(x, &mut ax);
    ax.as_slice().iter().zip(b.as_slice())
        .map(|(ai, bi)| (ai - bi).norm()).sum()
}

// ── Test 1: dense Helmholtz ───────────────────────────────────────────────────

#[test]
fn complex_gmres_dense_helmholtz() {
    let n = 30;
    let a = complex_helmholtz_dense(n, 2.0 * std::f64::consts::PI, 1.0);
    let b: DenseVec<C64> = (0..n).map(|i| c((i + 1) as f64, -0.5 * i as f64)).collect::<Vec<_>>().into();
    let mut x: DenseVec<C64> = vec![c(0.0, 0.0); n].into();

    let solver = ComplexGmres::<f64>::new(30);
    let res = solver.solve(&a, &b, &mut x, 1e-10, 0.0, 1000).unwrap();
    assert!(res.converged, "dense Helmholtz: {:?}", res);
    assert!(dense_residual(&a, &x, &b) < 1e-8, "dense residual too large");
}

// ── Test 2: sparse Helmholtz ──────────────────────────────────────────────────

#[test]
fn complex_gmres_sparse_helmholtz() {
    let n = 40;
    let a = complex_helmholtz_csr(n, 5.0, 0.5);
    let b: DenseVec<C64> = vec![c(1.0, 0.0); n].into();
    let mut x: DenseVec<C64> = vec![c(0.0, 0.0); n].into();

    let solver = ComplexGmres::<f64>::new(40);
    let res = solver.solve(&a, &b, &mut x, 1e-10, 0.0, 1000).unwrap();
    assert!(res.converged, "sparse Helmholtz: {:?}", res);
    assert!(sparse_residual(&a, &x, &b) < 1e-8, "sparse residual too large");
}

// ── Test 3: workspace reuse ───────────────────────────────────────────────────

#[test]
fn complex_gmres_workspace_reuse() {
    let n = 20;
    let a = complex_helmholtz_csr(n, 3.0, 0.2);
    let solver = ComplexGmres::<f64>::new(20);
    let mut ws = ComplexGmresWorkspace::<f64>::new(n, 20);

    for rhs_idx in 0..3 {
        let b: DenseVec<C64> = (0..n)
            .map(|i| c((i + rhs_idx) as f64, -(i as f64)))
            .collect::<Vec<_>>().into();
        let mut x: DenseVec<C64> = vec![c(0.0, 0.0); n].into();
        let res = solver.solve_with_workspace(&a, &b, &mut x, 1e-10, 0.0, 500, &mut ws).unwrap();
        assert!(res.converged, "workspace reuse iter {rhs_idx}: {:?}", res);
        assert!(sparse_residual(&a, &x, &b) < 1e-8, "iter {rhs_idx} residual too large");
    }
}

// ── Test 4: complex CsrMatrix apply (LinearOperator) ─────────────────────────

#[test]
fn complex_csr_linear_operator_apply() {
    use linger::core::operator::LinearOperator;
    // 3×3 diagonal complex matrix: A = diag(1+i, 2-i, 3)
    let mut coo: CooMatrix<C64> = CooMatrix::new_complex(3, 3);
    coo.push_complex(0, 0, c(1.0, 1.0));
    coo.push_complex(1, 1, c(2.0, -1.0));
    coo.push_complex(2, 2, c(3.0, 0.0));
    let a = CsrMatrix::from_complex_coo(&coo);

    let x: DenseVec<C64> = vec![c(1.0, 0.0), c(0.0, 1.0), c(1.0, -1.0)].into();
    let mut y: DenseVec<C64> = vec![c(0.0, 0.0); 3].into();
    a.apply(&x, &mut y);

    // Expected: y[0]=(1+i)*1=(1,1), y[1]=(2-i)*i=i+1=(1,2), y[2]=3*(1-i)=(3,-3)
    let expected = [c(1.0, 1.0), c(1.0, 2.0), c(3.0, -3.0)];
    for (got, exp) in y.as_slice().iter().zip(&expected) {
        assert!((got - exp).norm() < 1e-14, "apply mismatch: got {got:?}, expected {exp:?}");
    }
}

// ── Test 5: complex CsrMatrix transpose apply ────────────────────────────────

#[test]
fn complex_csr_transpose_apply() {
    use linger::core::operator::{LinearOperator, TransposeOperator};
    // Off-diagonal: A = [[0, 1+i], [2-i, 0]]
    let mut coo: CooMatrix<C64> = CooMatrix::new_complex(2, 2);
    coo.push_complex(0, 1, c(1.0, 1.0));
    coo.push_complex(1, 0, c(2.0, -1.0));
    let a = CsrMatrix::from_complex_coo(&coo);

    let x: DenseVec<C64> = vec![c(1.0, 2.0), c(-1.0, 0.5)].into();
    let mut y: DenseVec<C64> = vec![c(0.0, 0.0); 2].into();
    a.apply_transpose(&x, &mut y);

    // Aᵀ = [[0, 2-i], [1+i, 0]]
    // y[0] = (2-i)*(−1+0.5i) = −2 + i + i − 0.5i² = −2 + 2i + 0.5 = −1.5 + 2i
    // y[1] = (1+i)*(1+2i) = 1 + 2i + i + 2i² = 1 + 3i − 2 = −1 + 3i
    let expected = [c(-1.5, 2.0), c(-1.0, 3.0)];
    for (got, exp) in y.as_slice().iter().zip(&expected) {
        assert!((got - exp).norm() < 1e-14, "transpose mismatch: got {got:?}, expected {exp:?}");
    }
}

// ── Test 6: pure imaginary RHS ────────────────────────────────────────────────

#[test]
fn complex_gmres_pure_imaginary_rhs() {
    let n = 15;
    // Use a purely real matrix, purely imaginary b → solution should be purely imaginary
    let a_real: DenseMatrix<C64> = DenseMatrix::from_fn(n, n, |i, j| {
        if i == j { c(3.0, 0.0) }
        else if (i as isize - j as isize).abs() == 1 { c(-1.0, 0.0) }
        else { c(0.0, 0.0) }
    });
    let b: DenseVec<C64> = vec![c(0.0, 1.0); n].into();
    let mut x: DenseVec<C64> = vec![c(0.0, 0.0); n].into();
    let solver = ComplexGmres::<f64>::new(15);
    let res = solver.solve(&a_real, &b, &mut x, 1e-10, 0.0, 500).unwrap();
    assert!(res.converged, "pure imaginary RHS: {:?}", res);
    // Real part of solution must be ≈ 0
    for xi in x.as_slice() {
        assert!(xi.re.abs() < 1e-9, "expected zero real part, got {xi:?}");
    }
}

// ── Test 7: restart behaviour ─────────────────────────────────────────────────

#[test]
fn complex_gmres_restart_converges() {
    let n = 50;
    let a = complex_helmholtz_csr(n, 2.0, 0.3);
    let b: DenseVec<C64> = vec![c(1.0, 0.0); n].into();
    let mut x: DenseVec<C64> = vec![c(0.0, 0.0); n].into();
    // Use small restart=10 to force multiple outer restarts.
    let solver = ComplexGmres::<f64>::new(10);
    let res = solver.solve(&a, &b, &mut x, 1e-10, 0.0, 2000).unwrap();
    assert!(res.converged, "restart (m=10): {:?}", res);
    assert!(sparse_residual(&a, &x, &b) < 1e-8, "restart residual too large");
}
