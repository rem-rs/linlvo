//! Integration tests for A4: Block Jacobi preconditioner.

use linger::{
    precond::{JacobiPrecond, BlockJacobiPrecond},
    iterative::{Idrs, BiCgStab},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, Preconditioner, SolverParams, VerboseLevel,
};

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

/// A 2n×2n block-diagonal system: 2×2 blocks with [4,-1; -1,3] repeated.
fn block_diagonal_system(n_blocks: usize) -> CsrMatrix<f64> {
    let n = n_blocks * 2;
    let mut coo = CooMatrix::new(n, n);
    for b in 0..n_blocks {
        let off = b * 2;
        coo.push(off,     off,     4.0);
        coo.push(off,     off + 1, -1.0);
        coo.push(off + 1, off,     -1.0);
        coo.push(off + 1, off + 1, 3.0);
    }
    CsrMatrix::from_coo(&coo)
}

/// A 3n×3n block-diagonal system: 3×3 blocks with diag=[6,6,6] + off-diag=-1.
fn block3_system(n_blocks: usize) -> CsrMatrix<f64> {
    let n = n_blocks * 3;
    let mut coo = CooMatrix::new(n, n);
    for b in 0..n_blocks {
        let off = b * 3;
        for i in 0..3 {
            coo.push(off + i, off + i, 6.0);
            for j in 0..3 {
                if i != j { coo.push(off + i, off + j, -1.0); }
            }
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn rel_res(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    use linger::LinearOperator;
    let n = a.nrows();
    let mut ax = DenseVec::zeros(n);
    a.apply(x, &mut ax);
    let nr: f64 = ax.as_slice().iter().zip(b.as_slice()).map(|(a, b)| (a-b).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.as_slice().iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    if nb < 1e-300 { nr } else { nr / nb }
}

fn default_params() -> SolverParams {
    SolverParams { rtol: 1e-8, max_iter: 2000, verbose: VerboseLevel::Silent, ..Default::default() }
}

/// 1. Block size 1 is identical to point Jacobi.
#[test]
fn block_jacobi_size1_equals_point_jacobi() {
    let n = 20;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);

    let jac = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let bjac = BlockJacobiPrecond::<f64>::from_csr(&a, 1).unwrap();

    let mut y_jac  = DenseVec::zeros(n);
    let mut y_bjac = DenseVec::zeros(n);
    jac.apply_precond(&b, &mut y_jac);
    bjac.apply_precond(&b, &mut y_bjac);

    for i in 0..n {
        let diff = (y_jac.as_slice()[i] - y_bjac.as_slice()[i]).abs();
        assert!(diff < 1e-12, "block(1) ≠ point Jacobi at i={i}: diff={diff}");
    }
}

/// 2. 2×2 blocks on a block-diagonal system.
#[test]
fn block_jacobi_2x2_blocks() {
    let n_blocks = 50;
    let a = block_diagonal_system(n_blocks);
    let n = n_blocks * 2;
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let bjac = BlockJacobiPrecond::<f64>::from_csr(&a, 2).unwrap();
    let res = Idrs::<f64>::new(4).solve(&a, Some(&bjac), &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "Block(2)+IDR(4) did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 3. 3×3 blocks on a block-diagonal system.
#[test]
fn block_jacobi_3x3_blocks() {
    let n_blocks = 30;
    let a = block3_system(n_blocks);
    let n = n_blocks * 3;
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());
    let mut x = DenseVec::zeros(n);

    let bjac = BlockJacobiPrecond::<f64>::from_csr(&a, 3).unwrap();
    let res = BiCgStab::<f64>::new().solve(&a, Some(&bjac), &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "Block(3)+BiCGSTAB did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 3b. 4×4 blocks hit the small-block specialised solve path.
#[test]
fn block_jacobi_4x4_blocks() {
    let n_blocks = 20;
    let n = n_blocks * 4;
    let mut coo = CooMatrix::new(n, n);
    for b in 0..n_blocks {
        let off = b * 4;
        for i in 0..4 {
            coo.push(off + i, off + i, 8.0);
            for j in 0..4 {
                if i != j {
                    coo.push(off + i, off + j, -1.0);
                }
            }
        }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let bjac = BlockJacobiPrecond::<f64>::from_csr(&a, 4).unwrap();
    let res = BiCgStab::<f64>::new().solve(&a, Some(&bjac), &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "Block(4)+BiCGSTAB did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 4. Non-divisible n returns SolverError.
#[test]
fn block_jacobi_non_divisible_n_error() {
    use linger::SolverError;
    let a = laplacian_1d(10);  // n=10
    // block_size=3 → 10 not divisible by 3
    let result = BlockJacobiPrecond::<f64>::from_csr(&a, 3);
    assert!(matches!(result, Err(SolverError::PrecondSetupFailed { .. })));
}

/// 5. Works as preconditioner with IDR(4).
#[test]
fn block_jacobi_with_idrs4() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let bjac = BlockJacobiPrecond::<f64>::from_csr(&a, 2).unwrap();
    let res = Idrs::<f64>::new(4).solve(&a, Some(&bjac), &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "Block(2)+IDR(4) n=100 did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 6. Works as preconditioner with BiCGSTAB.
#[test]
fn block_jacobi_with_bicgstab() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);

    let bjac = BlockJacobiPrecond::<f64>::from_csr(&a, 5).unwrap();
    let res = BiCgStab::<f64>::new().solve(&a, Some(&bjac), &b, &mut x, &default_params()).unwrap();
    assert!(res.converged, "Block(5)+BiCGSTAB did not converge");
    assert!(rel_res(&a, &x, &b) < 1e-7);
}

/// 7. Singular block returns SolverError::SingularMatrix.
#[test]
fn block_jacobi_singular_block_error() {
    use linger::SolverError;
    // Build a 4×4 matrix with a singular 2×2 block in the top-left.
    let n = 4;
    let mut coo = CooMatrix::<f64>::new(n, n);
    // Block 0: [1,1; 1,1] — singular (determinant = 0)
    coo.push(0, 0, 1.0); coo.push(0, 1, 1.0);
    coo.push(1, 0, 1.0); coo.push(1, 1, 1.0);
    // Block 1: [2,0; 0,2] — non-singular
    coo.push(2, 2, 2.0);
    coo.push(3, 3, 2.0);
    let a = CsrMatrix::from_coo(&coo);
    let result = BlockJacobiPrecond::<f64>::from_csr(&a, 2);
    assert!(matches!(result, Err(SolverError::SingularMatrix { .. })));
}
