//! Integration tests for AMS and ADS preconditioners.
//!
//! Test topology:
//! - AMS tests use a 1-D chain graph: A = GGᵀ + δI, δ = 1e-3.
//! - ADS tests use a 2-D rectangular de Rham complex: A = CCᵀ + δI.

mod common;

use linger::{
    iterative::Gmres,
    precond::{AdsPrecond, AdsConfig, AmsPrecond, AmsConfig, AuxSpaceSolver},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, LinearOperator, Preconditioner, SolverParams, VerboseLevel,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Group A — AmsPrecond construction errors
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn ams_rejects_nonsquare_a() {
    // A is 3×4 — not square
    let mut ca = CooMatrix::new(3, 4);
    for i in 0..3 { ca.push(i, i, 2.0_f64); }
    let a = CsrMatrix::from_coo(&ca);
    let (g, _) = common::make_chain_graph(4, 1e-3); // G: 3×4
    let res = AmsPrecond::new(&a, &g, AmsConfig::default());
    assert!(res.is_err(), "expected error for non-square A");
}

#[test]
fn ams_rejects_g_wrong_nrows() {
    // A is 5×5 but G has 3 rows — mismatch
    let mut ca = CooMatrix::new(5, 5);
    for i in 0..5 { ca.push(i, i, 2.0_f64); }
    let a = CsrMatrix::from_coo(&ca);
    let (g, _) = common::make_chain_graph(4, 1e-3); // G: 3×4
    let res = AmsPrecond::new(&a, &g, AmsConfig::default());
    assert!(res.is_err(), "expected error for G.nrows ≠ A.nrows");
}

#[test]
fn ams_rejects_near_zero_diagonal() {
    // A has a zero diagonal at row 1
    let mut ca = CooMatrix::new(3, 3);
    ca.push(0, 0, 2.0_f64);
    ca.push(1, 1, 0.0_f64); // zero diagonal
    ca.push(2, 2, 2.0_f64);
    let a = CsrMatrix::from_coo(&ca);
    let (g, _) = common::make_chain_graph(4, 1e-3); // G: 3×4
    let res = AmsPrecond::new(&a, &g, AmsConfig::default());
    assert!(res.is_err(), "expected error for near-zero diagonal");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group B — AmsPrecond correctness
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn ams_applies_nontrivially() {
    // chain n=5: A is 4×4, G is 4×5
    let (g, a) = common::make_chain_graph(5, 1e-3);
    let p = AmsPrecond::new(&a, &g, AmsConfig::default()).unwrap();
    let n = a.nrows();
    let x = DenseVec::from_vec(vec![1.0_f64; n]);
    let mut y = DenseVec::zeros(n);
    p.apply_precond(&x, &mut y);
    let ys = y.as_slice();
    assert!(
        ys.iter().any(|&v| v.abs() > 1e-15),
        "AMS output should be non-zero"
    );
    assert!(
        ys.iter().all(|&v| v.is_finite()),
        "AMS output should be finite"
    );
}

#[test]
fn ams_with_ilu0_node_solver() {
    // Use ILU(0) as the nodal coarse solver.
    // Need a larger shift so GᵀAG is non-singular for ILU(0).
    let (g, a) = common::make_chain_graph(5, 0.5);
    let config = AmsConfig {
        node_solver: AuxSpaceSolver::Ilu0,
        ..Default::default()
    };
    let p = AmsPrecond::new(&a, &g, config).unwrap();
    let n = a.nrows();
    let x = DenseVec::from_vec(vec![1.0_f64; n]);
    let mut y = DenseVec::zeros(n);
    p.apply_precond(&x, &mut y);
    assert!(
        y.as_slice().iter().any(|&v| v.abs() > 1e-15),
        "AMS-ILU0 output should be non-zero"
    );
}

#[test]
fn ams_gradient_in_null_space() {
    // A gradient field x = G·v lies in the null space of A (for pure edge Laplacian).
    // The AMS preconditioner should handle it without producing NaN/Inf.
    let (g, a) = common::make_chain_graph(6, 1e-3);
    let n_nodes = g.ncols();
    // v = [1, 2, 3, 4, 5, 6]
    let v = DenseVec::from_vec((1..=n_nodes).map(|i| i as f64).collect::<Vec<_>>());
    // x = G v  (a gradient field)
    let mut x = DenseVec::zeros(a.nrows());
    g.apply(&v, &mut x);

    let p = AmsPrecond::new(&a, &g, AmsConfig::default()).unwrap();
    let mut y = DenseVec::zeros(a.nrows());
    p.apply_precond(&x, &mut y);

    let ys = y.as_slice();
    assert!(ys.iter().all(|&v| v.is_finite()), "AMS on gradient field should be finite");
}

#[test]
fn ams_gmres_convergence() {
    // Chain n=21: 20 edges, SPD system with δ=1e-3 shift.
    // GMRES + AMS should converge in far fewer iterations than unpreconditioned.
    let (g, a) = common::make_chain_graph(21, 1e-3);
    let n = a.nrows();

    // RHS: b = A * x_exact  where x_exact = sin(π k / n)
    let x_exact: Vec<f64> = (1..=n)
        .map(|k| (std::f64::consts::PI * k as f64 / (n + 1) as f64).sin())
        .collect();
    let mut b_raw = vec![0.0f64; n];
    a.spmv(&x_exact, &mut b_raw);
    let b = DenseVec::from_vec(b_raw);

    let precond = AmsPrecond::new(&a, &g, AmsConfig::default()).unwrap();
    let mut x = DenseVec::zeros(n);
    let params = SolverParams {
        rtol: 1e-8,
        max_iter: 200,
        verbose: VerboseLevel::Silent,
        ..Default::default()
    };
    let result = Gmres::new(30)
        .solve(&a, Some(&precond), &b, &mut x, &params)
        .unwrap();

    assert!(
        result.converged,
        "GMRES+AMS did not converge; iters={}, res={:.2e}",
        result.iterations, result.final_residual
    );
    assert!(
        result.iterations < 50,
        "GMRES+AMS took too many iterations: {}",
        result.iterations
    );

    // Verify solution quality
    let rel_err = common::relative_residual(&a, x.as_slice(), b.as_slice());
    assert!(rel_err < 1e-7, "relative residual too large: {rel_err:.2e}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group C — AdsPrecond construction errors
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn ads_rejects_c_wrong_nrows() {
    // A is 4×4 (4 faces), C has 3 rows — should be 4
    let mut ca = CooMatrix::new(4, 4);
    for i in 0..4 { ca.push(i, i, 2.0_f64); }
    let a = CsrMatrix::from_coo(&ca);

    // C: 3 rows (wrong), 2 cols (n_edges)
    let mut cc = CooMatrix::new(3, 2);
    cc.push(0, 0, 1.0_f64); cc.push(1, 1, 1.0); cc.push(2, 0, -1.0);
    let c = CsrMatrix::from_coo(&cc);

    // G: 2 rows (n_edges), 3 cols (n_nodes)
    let mut cg = CooMatrix::new(2, 3);
    cg.push(0, 0, -1.0_f64); cg.push(0, 1, 1.0);
    cg.push(1, 1, -1.0); cg.push(1, 2, 1.0);
    let g = CsrMatrix::from_coo(&cg);

    let res = AdsPrecond::new(&a, &c, &g, AdsConfig::default());
    assert!(res.is_err(), "expected error for C.nrows ≠ A.nrows");
}

#[test]
fn ads_rejects_g_wrong_nrows() {
    // C has 2 cols (n_edges=2), G has 3 rows — should be 2
    let (g_chain, a_chain) = common::make_chain_graph(3, 1e-2); // 2×3, A 2×2

    // Artificially make a C that has 2 rows (=n_faces), 2 cols (=n_edges=2)
    let mut cc = CooMatrix::new(2, 2);
    cc.push(0, 0, 1.0_f64); cc.push(0, 1, -1.0);
    cc.push(1, 0, -1.0); cc.push(1, 1, 1.0);
    let c = CsrMatrix::from_coo(&cc);

    // g_chain has 2 rows (n_edges) and 3 cols (n_nodes) — correct for this C.
    // But let's use a_chain which is 2×2 (n_edges×n_edges, not n_faces×n_faces).
    // → C.nrows (2) = A.nrows (2) OK, but we need a G with wrong nrows to trigger error.
    let mut cg_bad = CooMatrix::new(3, 2); // 3 rows, but n_edges = 2
    cg_bad.push(0, 0, -1.0_f64); cg_bad.push(0, 1, 1.0);
    cg_bad.push(1, 0, -1.0); cg_bad.push(1, 1, 1.0);
    cg_bad.push(2, 0, 1.0);
    let g_bad = CsrMatrix::from_coo(&cg_bad);

    let res = AdsPrecond::new(&a_chain, &c, &g_bad, AdsConfig::default());
    assert!(res.is_err(), "expected error for G.nrows ≠ C.ncols");
    let _ = (g_chain,); // suppress unused warning
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group D — AdsPrecond correctness
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn ads_applies_nontrivially() {
    // rect 3×3: 9 nodes, 12 edges, 4 faces
    let (g, c, a) = common::make_rect_complex(3, 3, 1e-3);
    let p = AdsPrecond::new(&a, &c, &g, AdsConfig::default()).unwrap();
    let n = a.nrows();
    let x = DenseVec::from_vec(vec![1.0_f64; n]);
    let mut y = DenseVec::zeros(n);
    p.apply_precond(&x, &mut y);
    let ys = y.as_slice();
    assert!(
        ys.iter().any(|&v| v.abs() > 1e-15),
        "ADS output should be non-zero"
    );
    assert!(
        ys.iter().all(|&v| v.is_finite()),
        "ADS output should be finite"
    );
}

#[test]
fn ads_curl_field_correction() {
    // x = C·e (a curl field) lies in the range of C.
    // ADS should handle it without NaN and produce a bounded output.
    let (g, c, a) = common::make_rect_complex(3, 3, 1e-3);
    let n_edges = c.ncols();
    // e = [1, -1, 1, -1, ...] (alternating)
    let e = DenseVec::from_vec(
        (0..n_edges).map(|k| if k % 2 == 0 { 1.0_f64 } else { -1.0 }).collect::<Vec<_>>()
    );
    let mut x = DenseVec::zeros(a.nrows());
    c.apply(&e, &mut x);

    let p = AdsPrecond::new(&a, &c, &g, AdsConfig::default()).unwrap();
    let mut y = DenseVec::zeros(a.nrows());
    p.apply_precond(&x, &mut y);
    assert!(
        y.as_slice().iter().all(|&v| v.is_finite()),
        "ADS on curl field should produce finite output"
    );
}

#[test]
fn ads_gmres_convergence() {
    // rect 4×4: 16 nodes, 24 edges, 9 faces; A = CCᵀ + δI
    let (g, c, a) = common::make_rect_complex(4, 4, 1e-3);
    let n = a.nrows();

    let b = DenseVec::from_vec(vec![1.0_f64; n]);
    let precond = AdsPrecond::new(&a, &c, &g, AdsConfig::default()).unwrap();
    let mut x = DenseVec::zeros(n);
    let params = SolverParams {
        rtol: 1e-8,
        max_iter: 300,
        verbose: VerboseLevel::Silent,
        ..Default::default()
    };
    let result = Gmres::new(30)
        .solve(&a, Some(&precond), &b, &mut x, &params)
        .unwrap();

    assert!(
        result.converged,
        "GMRES+ADS did not converge; iters={}, res={:.2e}",
        result.iterations, result.final_residual
    );
    assert!(
        result.iterations < 60,
        "GMRES+ADS took too many iterations: {}",
        result.iterations
    );

    let rel_err = common::relative_residual(&a, x.as_slice(), b.as_slice());
    assert!(rel_err < 1e-7, "ADS relative residual too large: {rel_err:.2e}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group E — Algebraic consistency (de Rham complex properties)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn discrete_complex_curl_of_gradient_is_zero() {
    // The key algebraic identity C·G = 0 must hold for ADS to be correct.
    let (g, c, _) = common::make_rect_complex(4, 4, 0.0);
    let cg = c.matmat(&g);
    for (_, _, v) in cg.triplets() {
        assert!(
            v.abs() < 1e-12,
            "C·G should be zero everywhere, got {v}"
        );
    }
}
