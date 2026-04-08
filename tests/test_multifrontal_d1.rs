//! Integration tests for D1: BLR-compressed MultifrontalLu.

use linger::{
    direct::{
        DirectSolver, DirectOptions, DirectSolverPrecond,
        MultifrontalLu, MultifrontalOptions,
        ordering::OrderingMethod,
        compress_block,
    },
    iterative::ConjugateGradient,
    sparse::{CooMatrix, CsrMatrix},
    DenseVec,
    core::{vector::Vector, operator::LinearOperator},
    KrylovSolver, SolverParams, VerboseLevel,
    Gmres,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0   { coo.push(i, i-1, -1.0); }
        if i < n-1 { coo.push(i, i+1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let n = b.len();
    let xv = DenseVec::from_vec(x.to_vec());
    let mut ax = DenseVec::zeros(n);
    a.apply(&xv, &mut ax);
    let res: f64 = ax.as_slice().iter().zip(b)
        .map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
    let nrm: f64 = b.iter().map(|v| v*v).sum::<f64>().sqrt();
    res / nrm.max(1e-300)
}

fn rel_err(x: &[f64], x_ref: &[f64]) -> f64 {
    let diff: f64 = x.iter().zip(x_ref).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
    let nrm: f64 = x_ref.iter().map(|v| v*v).sum::<f64>().sqrt();
    diff / nrm.max(1e-300)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// 1. Exact (BLR disabled) solve satisfies residual criterion.
#[test]
fn multifrontal_exact_solve_small() {
    let mut coo = CooMatrix::<f64>::new(3, 3);
    coo.push(0,0,4.0); coo.push(0,1,1.0);
    coo.push(1,0,1.0); coo.push(1,1,3.0); coo.push(1,2,1.0);
    coo.push(2,1,1.0); coo.push(2,2,5.0);
    let a = CsrMatrix::from_coo(&coo);
    let b = DenseVec::from_vec(vec![5.0, 10.0, 6.0]);
    let mut solver = MultifrontalLu::<f64>::default();
    solver.factor(&a).unwrap();
    let mut x = DenseVec::zeros(3);
    solver.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-10,
        "exact solve residual = {}", residual(&a, x.as_slice(), b.as_slice()));
}

/// 2. BLR-compressed solve used as GMRES preconditioner converges.
/// (GMRES is used instead of CG since the BLR preconditioner may not be SPD.)
#[test]
fn multifrontal_blr_as_precond_gmres_converges() {
    let n = 20;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    // Exact reference.
    let mut x_ref = DenseVec::zeros(n);
    {
        let mut exact = MultifrontalLu::<f64>::default();
        exact.factor(&a).unwrap();
        exact.solve(&b, &mut x_ref).unwrap();
    }
    // BLR with moderate compression: blr_min_size=10, tight tol.
    let blr_solver = MultifrontalLu::<f64>::with_blr(1e-8, 10);
    let precond = DirectSolverPrecond::new(blr_solver, &a).unwrap();
    let params = SolverParams { rtol: 1e-9, max_iter: 300, verbose: VerboseLevel::Silent, ..Default::default() };
    let mut x = DenseVec::zeros(n);
    let result = Gmres::new(10).solve(&a, Some(&precond), &b, &mut x, &params).unwrap();
    assert!(result.converged,
        "GMRES with BLR precond did not converge in {} iters", result.iterations);
    assert!(rel_err(x.as_slice(), x_ref.as_slice()) < 1e-7,
        "BLR precond GMRES solution error = {}", rel_err(x.as_slice(), x_ref.as_slice()));
}

/// 3. blr_factor_count and blr_compressed_count are sensible.
#[test]
fn multifrontal_blr_diagnostics() {
    let n = 16;
    let a = laplacian_1d(n);
    let opts = MultifrontalOptions {
        base: DirectOptions::default(),
        blr_min_size: 4,
        blr_tol: 1e-6,
    };
    let mut solver = MultifrontalLu::<f64>::with_options(opts);
    solver.factor(&a).unwrap();
    // With block size 4, n=16 → 4 supernodal blocks.
    assert_eq!(solver.blr_factor_count(), 4,
        "expected 4 BLR factor blocks for n=16, blr_min=4");
    assert!(solver.blr_compressed_count() <= solver.blr_factor_count());
}

/// 4. compress_block produces a proper low-rank approximation.
/// Uses a clearly rank-1 matrix (dominant singular value ≫ others).
#[test]
fn compress_block_rank_and_accuracy() {
    // Build a rank-1 matrix: A = u v^T.
    let m = 8; let n = 6;
    let u: Vec<f64> = (0..m).map(|i| (i as f64 + 1.0)).collect();
    let v: Vec<f64> = (0..n).map(|j| (j as f64 * 0.5 + 1.0)).collect();
    let mut a_mat = vec![0.0f64; m * n];
    for i in 0..m { for j in 0..n { a_mat[i*n+j] = u[i] * v[j]; } }
    let blk = compress_block::<f64>(&a_mat, m, n, 1e-8);
    assert!(blk.rank <= 1, "rank-1 block got rank={}", blk.rank);
    let recon = blk.to_dense();
    let err: f64 = a_mat.iter().zip(&recon).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt();
    let nrm: f64 = a_mat.iter().map(|v| v*v).sum::<f64>().sqrt();
    assert!(err / nrm < 1e-8, "BLR reconstruction error = {}", err/nrm);
}

/// 5. Large 1D Poisson exact solve (n=50).
#[test]
fn multifrontal_exact_poisson_n50() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut solver = MultifrontalLu::<f64>::default();
    solver.factor(&a).unwrap();
    let mut x = DenseVec::zeros(n);
    solver.solve(&b, &mut x).unwrap();
    assert!(residual(&a, x.as_slice(), b.as_slice()) < 1e-10,
        "poisson n=50 residual too large");
}

/// 6. BLR preconditioned GMRES on 1D Poisson n=50.
#[test]
fn multifrontal_blr_poisson_n50_as_precond() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    // Tight BLR tolerance → near-exact preconditioner → fast convergence.
    let blr_solver = MultifrontalLu::<f64>::with_blr(1e-8, 16);
    let precond = DirectSolverPrecond::new(blr_solver, &a).unwrap();
    let params = SolverParams { rtol: 1e-8, max_iter: 500, verbose: VerboseLevel::Silent, ..Default::default() };
    let mut x = DenseVec::zeros(n);
    let result = Gmres::new(20).solve(&a, Some(&precond), &b, &mut x, &params).unwrap();
    assert!(result.converged,
        "BLR-precond GMRES on n=50 Poisson did not converge in {} iters", result.iterations);
}

/// 7. with_blr constructor + basic solve is finite.
#[test]
fn multifrontal_with_blr_constructor() {
    let n = 8;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut solver = MultifrontalLu::<f64>::with_blr(1e-6, 3);
    solver.factor(&a).unwrap();
    assert!(solver.blr_factor_count() > 0, "expected BLR factors after factorize");
    let mut x = DenseVec::zeros(n);
    solver.solve(&b, &mut x).unwrap();
    assert!(x.as_slice().iter().all(|v| v.is_finite()), "solution contains non-finite values");
}

/// 8. reuse_symbolic works with BLR enabled.
#[test]
fn multifrontal_blr_reuse_symbolic() {
    let n = 12;
    let a = laplacian_1d(n);
    let opts = MultifrontalOptions {
        base: DirectOptions { reuse_symbolic: true, ..Default::default() },
        blr_min_size: 4,
        blr_tol: 1e-6,
    };
    let mut solver = MultifrontalLu::<f64>::with_options(opts);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    solver.factor(&a).unwrap();
    let mut x1 = DenseVec::zeros(n);
    solver.solve(&b, &mut x1).unwrap();
    solver.factor(&a).unwrap();
    let mut x2 = DenseVec::zeros(n);
    solver.solve(&b, &mut x2).unwrap();
    assert!(rel_err(x1.as_slice(), x2.as_slice()) < 1e-10,
        "reuse_symbolic gave inconsistent results");
}

/// 9. blr_tol=0 → full-rank BLR; the BLR solve path is consistent across calls.
#[test]
fn multifrontal_blr_tol0_consistent() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    // Two runs with tol=0 should give identical results.
    let mut blr1 = MultifrontalLu::<f64>::with_blr(0.0, 3);
    blr1.factor(&a).unwrap();
    let mut x1 = DenseVec::zeros(n);
    blr1.solve(&b, &mut x1).unwrap();
    let mut blr2 = MultifrontalLu::<f64>::with_blr(0.0, 3);
    blr2.factor(&a).unwrap();
    let mut x2 = DenseVec::zeros(n);
    blr2.solve(&b, &mut x2).unwrap();
    assert!(rel_err(x1.as_slice(), x2.as_slice()) < 1e-14,
        "blr_tol=0 runs gave different results");
    // All values should be finite.
    assert!(x1.as_slice().iter().all(|v| v.is_finite()));
}

/// 10. Ordering variants (RCM, ND) with BLR enabled, using GMRES.
#[test]
fn multifrontal_blr_rcm_nd_ordering() {
    let n = 20;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    for ordering in [OrderingMethod::Rcm, OrderingMethod::NodeNd] {
        let opts = MultifrontalOptions {
            base: DirectOptions { ordering: ordering.clone(), ..Default::default() },
            blr_min_size: 10,
            blr_tol: 1e-8,
        };
        let blr_solver = MultifrontalLu::<f64>::with_options(opts);
        let precond = DirectSolverPrecond::new(blr_solver, &a).unwrap();
        let params = SolverParams { rtol: 1e-9, max_iter: 300, verbose: VerboseLevel::Silent, ..Default::default() };
        let mut x = DenseVec::zeros(n);
        let result = Gmres::new(20).solve(&a, Some(&precond), &b, &mut x, &params).unwrap();
        assert!(result.converged,
            "GMRES with BLR+{ordering:?} did not converge after {} iters", result.iterations);
    }
}
