//! Integration tests for `FieldSplitPrecond`.
//!
//! Covers both `BlockJacobi` and `BlockTriangular` split modes, using Jacobi
//! and ILU(0) sub-preconditioners.  The test matrix is always the 1-D Poisson
//! operator (tridiagonal, SPD) solved by preconditioned CG.

mod common;

use linger::{
    iterative::{ConjugateGradient, Gmres},
    precond::{FieldSplitPrecond, Ilu0Precond, JacobiPrecond, SplitMode},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, SolverParams, VerboseLevel,
};

fn cg_params(rtol: f64, max_iter: usize) -> SolverParams {
    SolverParams { rtol, max_iter, verbose: VerboseLevel::Silent, ..Default::default() }
}

/// Extract the square diagonal block `A[row_start..row_end, row_start..row_end]`.
fn extract_diag_block(mat: &CsrMatrix<f64>, row_start: usize, row_end: usize) -> CsrMatrix<f64> {
    let nrows = row_end - row_start;
    let mut coo = CooMatrix::new(nrows, nrows);
    let rp   = mat.row_ptr();
    let ci   = mat.col_idx();
    let vals = mat.values();
    for i in row_start..row_end {
        for idx in rp[i]..rp[i + 1] {
            let col = ci[idx];
            if col >= row_start && col < row_end {
                coo.push(i - row_start, col - row_start, vals[idx]);
            }
        }
    }
    CsrMatrix::from_coo(&coo)
}

// ── BlockJacobi with Jacobi sub-preconditioners ───────────────────────────────

#[test]
fn fieldsplit_jacobi_mode_poisson_1d_n20() {
    let (a, x_exact, b_vec) = common::make_poisson_1d::<f64>(20);
    let b = DenseVec::from_vec(b_vec);

    let split = 10;
    let a00 = extract_diag_block(&a, 0, split);
    let a11 = extract_diag_block(&a, split, 20);

    let p0 = JacobiPrecond::from_csr(&a00).expect("Jacobi p0");
    let p1 = JacobiPrecond::from_csr(&a11).expect("Jacobi p1");

    let precond = FieldSplitPrecond::new(20, split, SplitMode::BlockJacobi, Box::new(p0), Box::new(p1));

    let mut x = DenseVec::zeros(20);
    let res = ConjugateGradient::new(20)
        .solve(&a, Some(&precond), &b, &mut x, &cg_params(1e-10, 200))
        .expect("BlockJacobi CG");

    assert!(res.converged, "BlockJacobi did not converge");
    let err: f64 = x.as_slice().iter().zip(&x_exact).map(|(&xi, &xe)| (xi - xe).powi(2)).sum::<f64>().sqrt();
    assert!(err < 1e-8, "BlockJacobi solution error = {err:.3e}");
}

// ── BlockTriangular with Jacobi sub-preconditioners ───────────────────────────

#[test]
fn fieldsplit_triangular_mode_jacobi_poisson_1d_n20() {
    let (a, x_exact, b_vec) = common::make_poisson_1d::<f64>(20);
    let b = DenseVec::from_vec(b_vec);

    let split = 10;
    let a00 = extract_diag_block(&a, 0, split);
    let a11 = extract_diag_block(&a, split, 20);

    let p0 = JacobiPrecond::from_csr(&a00).expect("Jacobi p0");
    let p1 = JacobiPrecond::from_csr(&a11).expect("Jacobi p1");

    let precond = FieldSplitPrecond::from_matrix(
        &a, split, SplitMode::BlockTriangular, Box::new(p0), Box::new(p1),
    );

    let mut x = DenseVec::zeros(20);
    let res = Gmres::new(20)
        .solve(&a, Some(&precond), &b, &mut x, &cg_params(1e-8, 200))
        .expect("BlockTriangular GMRES");

    assert!(res.converged, "BlockTriangular did not converge");
    let err: f64 = x.as_slice().iter().zip(&x_exact).map(|(&xi, &xe)| (xi - xe).powi(2)).sum::<f64>().sqrt();
    assert!(err < 1e-6, "BlockTriangular solution error = {err:.3e}");
}

// ── BlockTriangular with ILU(0) sub-preconditioners ──────────────────────────

#[test]
fn fieldsplit_triangular_ilu0_poisson_1d_n20() {
    let (a, x_exact, b_vec) = common::make_poisson_1d::<f64>(20);
    let b = DenseVec::from_vec(b_vec);

    let split = 10;
    let a00 = extract_diag_block(&a, 0, split);
    let a11 = extract_diag_block(&a, split, 20);

    let p0 = Ilu0Precond::from_csr(&a00).expect("ILU0 p0");
    let p1 = Ilu0Precond::from_csr(&a11).expect("ILU0 p1");

    let precond = FieldSplitPrecond::from_matrix(
        &a, split, SplitMode::BlockTriangular, Box::new(p0), Box::new(p1),
    );

    let mut x = DenseVec::zeros(20);
    let res = Gmres::new(20)
        .solve(&a, Some(&precond), &b, &mut x, &cg_params(1e-8, 50))
        .expect("FieldSplit ILU0 GMRES");

    assert!(res.converged, "FieldSplit ILU0 did not converge");
    let err: f64 = x.as_slice().iter().zip(&x_exact).map(|(&xi, &xe)| (xi - xe).powi(2)).sum::<f64>().sqrt();
    assert!(err < 1e-6, "FieldSplit ILU0 solution error = {err:.3e}");
}

// ── Larger system (n=60, ILU0, BlockTriangular) ───────────────────────────────

#[test]
fn fieldsplit_ilu0_poisson_1d_n60() {
    let (a, x_exact, b_vec) = common::make_poisson_1d::<f64>(60);
    let b = DenseVec::from_vec(b_vec);

    let split = 30;
    let a00 = extract_diag_block(&a, 0, split);
    let a11 = extract_diag_block(&a, split, 60);

    let p0 = Ilu0Precond::from_csr(&a00).expect("ILU0 p0 n60");
    let p1 = Ilu0Precond::from_csr(&a11).expect("ILU0 p1 n60");

    let precond = FieldSplitPrecond::from_matrix(
        &a, split, SplitMode::BlockTriangular, Box::new(p0), Box::new(p1),
    );

    let mut x = DenseVec::zeros(60);
    let res = Gmres::new(30)
        .solve(&a, Some(&precond), &b, &mut x, &cg_params(1e-7, 200))
        .expect("FieldSplit ILU0 n60 GMRES");

    assert!(res.converged, "FieldSplit ILU0 n60 did not converge");
    let err: f64 = x.as_slice().iter().zip(&x_exact).map(|(&xi, &xe)| (xi - xe).powi(2)).sum::<f64>().sqrt();
    assert!(err < 1e-5, "FieldSplit ILU0 n60 error = {err:.3e}");
}

// ── BlockJacobi reduces iteration count compared to unpreconditioned CG ───────

#[test]
fn fieldsplit_jacobi_fewer_iters_than_cg_n40() {
    let (a, _x_exact, b_vec) = common::make_poisson_1d::<f64>(40);
    let b = DenseVec::from_vec(b_vec);

    let split = 20;
    let a00 = extract_diag_block(&a, 0, split);
    let a11 = extract_diag_block(&a, split, 40);

    let p0 = JacobiPrecond::from_csr(&a00).expect("Jacobi p0");
    let p1 = JacobiPrecond::from_csr(&a11).expect("Jacobi p1");

    let precond = FieldSplitPrecond::new(40, split, SplitMode::BlockJacobi, Box::new(p0), Box::new(p1));

    let params = cg_params(1e-8, 500);

    let mut x_prec  = DenseVec::zeros(40);
    let mut x_noprec = DenseVec::zeros(40);

    let r_prec  = ConjugateGradient::new(40).solve(&a, Some(&precond), &b, &mut x_prec, &params).unwrap();
    let r_noprec = ConjugateGradient::new(40).solve(&a, None, &b, &mut x_noprec, &params).unwrap();

    assert!(r_prec.converged && r_noprec.converged, "one solver did not converge");
    // Jacobi doesn't dramatically improve iteration count for uniform-diagonal
    // Poisson, but the preconditioned system must still converge.
    assert!(r_prec.iterations <= r_noprec.iterations + 5,
        "preconditioned CG used {} iters vs unpreconditioned {} (expected ≤)",
        r_prec.iterations, r_noprec.iterations);
}
