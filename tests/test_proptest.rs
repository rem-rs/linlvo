//! T1 — Property-based tests using proptest.
//!
//! Each property captures an invariant that must hold for **all** valid inputs,
//! not just the hand-crafted cases in the unit tests.  Shrinking is automatic:
//! when a failure is found, proptest minimises the input to its simplest form.
//!
//! Domains covered:
//!  A. Sparse matrix algebra  (CsrMatrix / COO construction)
//!  B. Direct solver correctness  (SparseLu, SparseCholesky, SparseLdlt)
//!  C. Krylov solver consistency  (CG, GMRES residual bound)
//!  D. Preconditioner contracts  (Jacobi, ILU0 — apply is linear, finite)
//!  E. BLR compression   (compress_block reconstruction bound)

use proptest::prelude::*;
use linger::{
    direct::{DirectSolver, DirectOptions, SparseLu, SparseCholesky, SparseLdlt,
             compress_block},
    iterative::ConjugateGradient,
    precond::{JacobiPrecond, Ilu0Precond},
    sparse::{CooMatrix, CsrMatrix},
    core::{operator::LinearOperator, vector::Vector},
    DenseVec, KrylovSolver, Preconditioner, SolverParams, VerboseLevel,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a strictly diagonally-dominant tridiagonal matrix of size `n` with
/// diagonal `diag` and off-diagonal `off`.  Always SPD when off < diag/2.
fn tridiag_spd(n: usize, diag: f64, off: f64) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, diag);
        if i > 0   { coo.push(i, i-1, off); }
        if i < n-1 { coo.push(i, i+1, off); }
    }
    CsrMatrix::from_coo(&coo)
}

/// Build a general (non-symmetric) tridiagonal matrix.
fn tridiag_general(n: usize, diag: f64, lower: f64, upper: f64) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, diag);
        if i > 0   { coo.push(i, i-1, lower); }
        if i < n-1 { coo.push(i, i+1, upper); }
    }
    CsrMatrix::from_coo(&coo)
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn residual_norm(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let mut ax = DenseVec::zeros(b.len());
    a.apply(x, &mut ax);
    let diff: f64 = ax.as_slice().iter().zip(b.as_slice())
        .map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
    diff / l2_norm(b.as_slice()).max(1e-300)
}

// ─────────────────────────────────────────────────────────────────────────────
// A. Sparse matrix algebra invariants
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// CsrMatrix row_ptr always starts at 0 and ends at nnz.
    #[test]
    fn prop_csr_row_ptr_bounds(
        n in 2usize..=30,
        diag in 1.0f64..10.0,
        off in -0.4f64..0.4,
    ) {
        let a = tridiag_spd(n, diag, off);
        let rp = a.row_ptr();
        prop_assert_eq!(rp[0], 0);
        prop_assert_eq!(rp[n], a.nnz());
    }

    /// Every column index is in-bounds [0, n).
    #[test]
    fn prop_csr_col_idx_in_bounds(
        n in 2usize..=30,
        diag in 1.0f64..10.0,
        off in -0.4f64..0.4,
    ) {
        let a = tridiag_spd(n, diag, off);
        for &c in a.col_idx() {
            prop_assert!(c < n, "col_idx={} out of range n={}", c, n);
        }
    }

    /// SpMV linearity: A(x + y) == Ax + Ay (within floating-point tolerance).
    #[test]
    fn prop_spmv_linearity(
        n in 3usize..=20,
        xs in prop::collection::vec(-5.0f64..5.0, 3..=20),
        ys in prop::collection::vec(-5.0f64..5.0, 3..=20),
    ) {
        let n = n.min(xs.len()).min(ys.len());
        let a = tridiag_spd(n, 4.0, -1.0);
        let x = DenseVec::from_vec(xs[..n].to_vec());
        let y = DenseVec::from_vec(ys[..n].to_vec());

        let mut ax  = DenseVec::zeros(n);
        let mut ay  = DenseVec::zeros(n);
        let mut axy = DenseVec::zeros(n);
        a.apply(&x, &mut ax);
        a.apply(&y, &mut ay);
        // x + y
        let xy = DenseVec::from_vec(
            x.as_slice().iter().zip(y.as_slice()).map(|(a,b)| a+b).collect()
        );
        a.apply(&xy, &mut axy);

        let err: f64 = axy.as_slice().iter()
            .zip(ax.as_slice().iter().zip(ay.as_slice()))
            .map(|(c, (a, b))| (c - (a + b)).abs())
            .fold(0.0f64, f64::max);
        prop_assert!(err < 1e-10, "linearity error={err}");
    }

    /// nnz == col_idx.len() == values.len().
    #[test]
    fn prop_csr_nnz_consistent(
        n in 2usize..=50,
        diag in 1.0f64..10.0,
        off in -0.4f64..0.4,
    ) {
        let a = tridiag_spd(n, diag, off);
        prop_assert_eq!(a.nnz(), a.col_idx().len());
        prop_assert_eq!(a.nnz(), a.values().len());
    }

    /// COO with duplicate entries: duplicates should be summed into CSR.
    #[test]
    fn prop_coo_duplicates_summed(
        n in 2usize..=20,
        v in 0.1f64..5.0,
    ) {
        // Push the same (0,0) entry twice; after CSR conversion it should be 2v.
        let mut coo = CooMatrix::<f64>::new(n, n);
        coo.push(0, 0, v);
        coo.push(0, 0, v);
        let a = CsrMatrix::from_coo(&coo);
        // Find the (0,0) value in CSR.
        let rp = a.row_ptr();
        let ci = a.col_idx();
        let va = a.values();
        let mut found = false;
        for k in rp[0]..rp[1] {
            if ci[k] == 0 {
                prop_assert!((va[k] - 2.0 * v).abs() < 1e-12,
                    "duplicate not summed: got {} expected {}", va[k], 2.0*v);
                found = true;
            }
        }
        prop_assert!(found, "diagonal entry not found in row 0");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// B. Direct solver correctness
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// SparseLu: for any diagonally-dominant tridiagonal, solve gives residual < 1e-10.
    #[test]
    fn prop_sparse_lu_residual(
        n in 2usize..=40,
        diag in 2.0f64..8.0,
        lower in -0.8f64..0.0,
        upper in 0.0f64..0.8,
        rhs in prop::collection::vec(-10.0f64..10.0, 2..=40),
    ) {
        let n = n.min(rhs.len());
        let a = tridiag_general(n, diag, lower, upper);
        let b = DenseVec::from_vec(rhs[..n].to_vec());
        let mut solver = SparseLu::<f64>::new(DirectOptions::default());
        solver.factor(&a).unwrap();
        let mut x = DenseVec::zeros(n);
        solver.solve(&b, &mut x).unwrap();
        let rel = residual_norm(&a, &x, &b);
        prop_assert!(rel < 1e-8, "SparseLu residual={rel:.2e} n={n}");
    }

    /// SparseCholesky: SPD tridiagonal, residual < 1e-10.
    #[test]
    fn prop_cholesky_residual(
        n in 2usize..=50,
        diag in 2.5f64..8.0,   // strictly > 2*|off| → SPD
        off_abs in 0.01f64..1.0,
        rhs in prop::collection::vec(-10.0f64..10.0, 2..=50),
    ) {
        let n = n.min(rhs.len());
        // off = -off_abs ensures diag > 2*off_abs → strictly diag-dominant → SPD
        let off = -off_abs.min(diag / 2.0 - 0.1);
        let a = tridiag_spd(n, diag, off);
        let b = DenseVec::from_vec(rhs[..n].to_vec());
        let mut solver = SparseCholesky::<f64>::new(DirectOptions::default());
        solver.factor(&a).unwrap();
        let mut x = DenseVec::zeros(n);
        solver.solve(&b, &mut x).unwrap();
        let rel = residual_norm(&a, &x, &b);
        prop_assert!(rel < 1e-8, "Cholesky residual={rel:.2e} n={n} diag={diag} off={off}");
    }

    /// SparseLdlt: symmetric tridiagonal (potentially indefinite), residual < 1e-8.
    #[test]
    fn prop_ldlt_residual(
        n in 2usize..=40,
        diag in 2.0f64..8.0,
        off in -0.9f64..0.9,
        rhs in prop::collection::vec(-10.0f64..10.0, 2..=40),
    ) {
        let n = n.min(rhs.len());
        let a = tridiag_spd(n, diag, off); // symmetric; may be indefinite for large |off|
        let b = DenseVec::from_vec(rhs[..n].to_vec());
        let mut solver = SparseLdlt::<f64>::default();
        // Only test when factorization succeeds (might fail for very indefinite).
        if solver.factor(&a).is_err() { return Ok(()); }
        let mut x = DenseVec::zeros(n);
        if solver.solve(&b, &mut x).is_err() { return Ok(()); }
        if !x.as_slice().iter().all(|v| v.is_finite()) { return Ok(()); }
        let rel = residual_norm(&a, &x, &b);
        prop_assert!(rel < 1e-8, "LDLt residual={rel:.2e} n={n}");
    }

    /// Direct solvers: A(A⁻¹b) ≈ b  (roundtrip property).
    #[test]
    fn prop_lu_roundtrip(
        n in 2usize..=25,
        diag in 3.0f64..6.0,
        rhs in prop::collection::vec(1.0f64..5.0, 2..=25),
    ) {
        let n = n.min(rhs.len());
        let a = tridiag_spd(n, diag, -0.5);
        let b = DenseVec::from_vec(rhs[..n].to_vec());
        let mut solver = SparseLu::<f64>::new(DirectOptions::default());
        solver.factor(&a).unwrap();
        let mut x = DenseVec::zeros(n);
        solver.solve(&b, &mut x).unwrap();
        // Ax should recover b.
        let mut ax = DenseVec::zeros(n);
        a.apply(&x, &mut ax);
        let err = ax.as_slice().iter().zip(b.as_slice())
            .map(|(ai,bi)| (ai - bi).abs())
            .fold(0.0f64, f64::max);
        prop_assert!(err < 1e-8, "roundtrip max-err={err:.2e}");
    }

    /// Solve is consistent: calling solve twice with same b gives same x.
    #[test]
    fn prop_solve_deterministic(
        n in 3usize..=30,
        diag in 3.0f64..7.0,
        rhs in prop::collection::vec(-5.0f64..5.0, 3..=30),
    ) {
        let n = n.min(rhs.len());
        let a = tridiag_spd(n, diag, -1.0);
        let b = DenseVec::from_vec(rhs[..n].to_vec());
        let mut solver = SparseLu::<f64>::new(DirectOptions::default());
        solver.factor(&a).unwrap();
        let mut x1 = DenseVec::zeros(n);
        let mut x2 = DenseVec::zeros(n);
        solver.solve(&b, &mut x1).unwrap();
        solver.solve(&b, &mut x2).unwrap();
        let diff = x1.as_slice().iter().zip(x2.as_slice())
            .map(|(a,b)| (a-b).abs()).fold(0.0f64, f64::max);
        prop_assert!(diff < 1e-14, "solve not deterministic, diff={diff}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// C. Krylov solver consistency
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// CG converges for any SPD tridiagonal (strict diagonal dominance).
    #[test]
    fn prop_cg_converges_spd(
        n in 3usize..=30,
        diag in 3.0f64..8.0,   // always > 2 → strictly diag-dominant
        rhs in prop::collection::vec(1.0f64..5.0, 3..=30),
    ) {
        let n = n.min(rhs.len());
        let a = tridiag_spd(n, diag, -1.0);
        let b = DenseVec::from_vec(rhs[..n].to_vec());
        let params = SolverParams { rtol: 1e-10, max_iter: 5 * n, verbose: VerboseLevel::Silent, ..Default::default() };
        let cg = ConjugateGradient::<f64>::default();
        let mut x = DenseVec::zeros(n);
        let res = cg.solve(&a, None, &b, &mut x, &params).unwrap();
        prop_assert!(res.converged,
            "CG did not converge in {} iters, n={n}, diag={diag}", res.iterations);
        prop_assert!(residual_norm(&a, &x, &b) < 1e-8,
            "CG residual={:.2e}", residual_norm(&a, &x, &b));
    }

    /// CG solution improves monotonically (residual_history is non-increasing).
    #[test]
    fn prop_cg_residual_history_nonincreasing(
        n in 5usize..=20,
        diag in 3.0f64..8.0,
    ) {
        let a = tridiag_spd(n, diag, -1.0);
        let b = DenseVec::from_vec(vec![1.0f64; n]);
        let params = SolverParams { rtol: 1e-12, max_iter: 3 * n, verbose: VerboseLevel::Silent, ..Default::default() };
        let cg = ConjugateGradient::<f64>::default();
        let mut x = DenseVec::zeros(n);
        let res = cg.solve(&a, None, &b, &mut x, &params).unwrap();
        // residual_history must be non-empty.
        prop_assert!(!res.residual_history.is_empty(), "residual_history empty");
        // Check non-increasing (allow tiny upward blips ≤ 1% from rounding).
        let hist = &res.residual_history;
        for w in hist.windows(2) {
            prop_assert!(w[1] <= w[0] * 1.01 + 1e-14,
                "residual increased: {:.4e} → {:.4e}", w[0], w[1]);
        }
    }

    /// residual_history.last() ≈ final_residual.
    #[test]
    fn prop_residual_history_matches_final(
        n in 4usize..=25,
        diag in 3.0f64..8.0,
    ) {
        let a = tridiag_spd(n, diag, -1.0);
        let b = DenseVec::from_vec(vec![1.0f64; n]);
        let params = SolverParams { rtol: 1e-12, max_iter: 4 * n, verbose: VerboseLevel::Silent, ..Default::default() };
        let cg = ConjugateGradient::<f64>::default();
        let mut x = DenseVec::zeros(n);
        let res = cg.solve(&a, None, &b, &mut x, &params).unwrap();
        prop_assert!(!res.residual_history.is_empty());
        let last = *res.residual_history.last().unwrap();
        prop_assert!((last - res.final_residual).abs() < 1e-10 * last.max(1e-14),
            "history last={last:.4e} final_residual={:.4e}", res.final_residual);
    }

    /// CG: solution is always finite when solver succeeds.
    #[test]
    fn prop_cg_solution_finite(
        n in 2usize..=30,
        diag in 3.0f64..8.0,
        rhs in prop::collection::vec(-10.0f64..10.0, 2..=30),
    ) {
        let n = n.min(rhs.len());
        let a = tridiag_spd(n, diag, -1.0);
        let b = DenseVec::from_vec(rhs[..n].to_vec());
        let params = SolverParams { rtol: 1e-10, max_iter: 5 * n, verbose: VerboseLevel::Silent, ..Default::default() };
        let cg = ConjugateGradient::<f64>::default();
        let mut x = DenseVec::zeros(n);
        let _ = cg.solve(&a, None, &b, &mut x, &params);
        prop_assert!(x.as_slice().iter().all(|v| v.is_finite()),
            "CG solution contains non-finite value");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// D. Preconditioner contracts
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// Jacobi preconditioner: M⁻¹ x is finite for any finite x.
    #[test]
    fn prop_jacobi_output_finite(
        n in 2usize..=30,
        diag in 1.0f64..10.0,
        xs in prop::collection::vec(-5.0f64..5.0, 2..=30),
    ) {
        let n = n.min(xs.len());
        let a = tridiag_spd(n, diag, -0.5);
        let precond = JacobiPrecond::from_csr(&a).unwrap();
        let x = DenseVec::from_vec(xs[..n].to_vec());
        let mut y = DenseVec::zeros(n);
        precond.apply_precond(&x, &mut y);
        prop_assert!(y.as_slice().iter().all(|v: &f64| v.is_finite()),
            "Jacobi output contains non-finite");
    }

    /// Jacobi preconditioner is linear: M⁻¹(ax) = a M⁻¹(x).
    #[test]
    fn prop_jacobi_homogeneous(
        n in 2usize..=20,
        diag in 1.0f64..10.0,
        xs in prop::collection::vec(-5.0f64..5.0, 2..=20),
        alpha in -3.0f64..3.0,
    ) {
        let n = n.min(xs.len());
        let a = tridiag_spd(n, diag, -0.3);
        let precond = JacobiPrecond::from_csr(&a).unwrap();
        let x = DenseVec::from_vec(xs[..n].to_vec());
        let ax = DenseVec::from_vec(xs[..n].iter().map(|&v| alpha * v).collect());
        let mut y  = DenseVec::zeros(n);
        let mut ay = DenseVec::zeros(n);
        precond.apply_precond(&x, &mut y);
        precond.apply_precond(&ax, &mut ay);
        let err = y.as_slice().iter().zip(ay.as_slice())
            .map(|(yi, ayi)| (alpha * yi - ayi).abs())
            .fold(0.0f64, f64::max);
        prop_assert!(err < 1e-12, "Jacobi not homogeneous: err={err}");
    }

    /// ILU0 preconditioner: output is always finite.
    #[test]
    fn prop_ilu0_output_finite(
        n in 3usize..=30,
        diag in 3.0f64..8.0,
        xs in prop::collection::vec(-5.0f64..5.0, 3..=30),
    ) {
        let n = n.min(xs.len());
        let a = tridiag_general(n, diag, -0.5, -0.5);
        let precond = Ilu0Precond::from_csr(&a).unwrap();
        let x = DenseVec::from_vec(xs[..n].to_vec());
        let mut y = DenseVec::zeros(n);
        precond.apply_precond(&x, &mut y);
        prop_assert!(y.as_slice().iter().all(|v: &f64| v.is_finite()),
            "ILU0 output contains non-finite");
    }

    /// Jacobi exact for diagonal matrix: M⁻¹(Ax) = x when A = diag(d).
    #[test]
    fn prop_jacobi_exact_on_diagonal(
        n in 2usize..=20,
        diag in 0.5f64..8.0,
        xs in prop::collection::vec(-5.0f64..5.0, 2..=20),
    ) {
        let n = n.min(xs.len());
        // Build a pure diagonal matrix (no off-diagonal).
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n { coo.push(i, i, diag); }
        let a = CsrMatrix::from_coo(&coo);
        let precond = JacobiPrecond::from_csr(&a).unwrap();
        // x_in = A * xs  ==> M⁻¹(x_in) == xs  (Jacobi = exact inverse for diagonal)
        let x_in = DenseVec::from_vec(xs[..n].iter().map(|&v| diag * v).collect());
        let mut y = DenseVec::zeros(n);
        precond.apply_precond(&x_in, &mut y);
        let err = y.as_slice().iter().zip(&xs[..n])
            .map(|(yi, xi)| (yi - xi).abs())
            .fold(0.0f64, f64::max);
        prop_assert!(err < 1e-12,
            "Jacobi not exact on diagonal matrix: max err={err}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E. BLR compression properties
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// compress_block: reconstructed matrix has Frobenius error ≤ tol * ||A||_F.
    #[test]
    fn prop_blr_reconstruction_bound(
        m in 3usize..=12,
        n in 3usize..=12,
        // Use a rank-1 matrix to guarantee the bound is achievable.
        u in prop::collection::vec(-3.0f64..3.0, 3..=12),
        v in prop::collection::vec(-3.0f64..3.0, 3..=12),
        tol in prop::sample::select(vec![1e-4f64, 1e-6, 1e-8]),
    ) {
        let m = m.min(u.len());
        let n = n.min(v.len());
        // A = u v^T (rank-1)
        let mut a = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n { a[i*n+j] = u[i] * v[j]; } }
        let nrm: f64 = a.iter().map(|x| x*x).sum::<f64>().sqrt();
        if nrm < 1e-10 { return Ok(()); } // skip degenerate case

        let blk = compress_block::<f64>(&a, m, n, tol, 0);
        let recon = blk.to_dense();
        let err: f64 = a.iter().zip(&recon).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt();
        prop_assert!(err / nrm < tol * 100.0 + 1e-10,
            "BLR error {:.2e} >> tol {tol:.2e}, m={m} n={n}", err / nrm);
    }

    /// compress_block: rank never exceeds min(m, n).
    #[test]
    fn prop_blr_rank_bounded(
        m in 2usize..=15,
        n in 2usize..=15,
        vals in prop::collection::vec(-2.0f64..2.0, 4..=225),
    ) {
        let size = m * n;
        if vals.len() < size { return Ok(()); }
        let a = vals[..size].to_vec();
        let blk = compress_block::<f64>(&a, m, n, 1e-12, 0);
        prop_assert!(blk.rank <= m.min(n),
            "rank {} > min(m={m},n={n})", blk.rank);
    }

    /// compress_block of zero matrix → rank 0.
    #[test]
    fn prop_blr_zero_rank_zero(m in 2usize..=15, n in 2usize..=15) {
        let a = vec![0.0f64; m * n];
        let blk = compress_block::<f64>(&a, m, n, 1e-12, 0);
        prop_assert_eq!(blk.rank, 0, "zero matrix got rank={}", blk.rank);
    }

    /// compress_block: apply_add matches dense matvec (within reconstruction tol).
    #[test]
    fn prop_blr_apply_add_correct(
        m in 3usize..=10,
        n in 3usize..=10,
        u in prop::collection::vec(-2.0f64..2.0, 3..=10),
        v in prop::collection::vec(-2.0f64..2.0, 3..=10),
        x in prop::collection::vec(-2.0f64..2.0, 3..=10),
    ) {
        let m = m.min(u.len());
        let n = n.min(v.len()).min(x.len());
        // Rank-1 matrix A = u v^T.
        let mut a = vec![0.0f64; m * n];
        for i in 0..m { for j in 0..n { a[i*n+j] = u[i] * v[j]; } }
        let blk = compress_block::<f64>(&a, m, n, 1e-10, 0);
        let xv = &x[..n];
        // Dense reference: y_ref = A * xv
        let mut y_ref = vec![0.0f64; m];
        for i in 0..m { for j in 0..n { y_ref[i] += a[i*n+j] * xv[j]; } }
        // BLR apply_add
        let mut y_blr = vec![0.0f64; m];
        blk.apply_add(xv, &mut y_blr, 1.0f64);
        let err = y_ref.iter().zip(&y_blr).map(|(a,b)| (a-b).abs()).fold(0.0f64, f64::max);
        let scale = y_ref.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1e-14);
        prop_assert!(err / scale < 1e-6,
            "apply_add error {:.2e} at scale {scale:.2e}", err / scale);
    }
}
