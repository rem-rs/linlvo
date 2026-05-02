//! Integration tests for A3: AMG K-cycle.
#![allow(clippy::needless_range_loop)]

mod common;

use linger::{
    amg::{AmgConfig, AmgHierarchy, AmgPrecond, CycleType},
    iterative::ConjugateGradient,
    sparse::CsrMatrix,
    DenseVec, KrylovSolver, SolverParams, VerboseLevel, Vector,
    LinearOperator,
};

fn rel_residual(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let n = b.len();
    let mut ax = DenseVec::zeros(n);
    a.apply(x, &mut ax);
    let r: f64 = ax.as_slice().iter().zip(b.as_slice()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.as_slice().iter().map(|&v| v * v).sum::<f64>().sqrt();
    if nb > 0.0 { r / nb } else { r }
}

fn amg_config() -> AmgConfig {
    AmgConfig {
        coarse_threshold: 4,
        pre_sweeps: 1,
        post_sweeps: 1,
        ..Default::default()
    }
}

fn cg_params() -> SolverParams {
    SolverParams { rtol: 1e-8, max_iter: 500, verbose: VerboseLevel::Silent, ..Default::default() }
}

/// 1. K-cycle via AMG preconditioned CG converges on 1D Laplacian (basic smoke test).
#[test]
fn kcycle_laplacian_1d_converges() {
    let n = 100;
    let (a, _, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);

    let hier = AmgHierarchy::build(a.clone(), amg_config());
    let prec = AmgPrecond::new(hier).with_cycle(CycleType::K { inner_iters: 2 });
    let mut x = DenseVec::zeros(n);
    let res = ConjugateGradient::<f64>::default()
        .solve(&a, Some(&prec), &b_vec, &mut x, &cg_params())
        .unwrap();
    assert!(res.converged, "K-cycle+CG did not converge");
    assert!(rel_residual(&a, &x, &b_vec) < 1e-7);
}

/// 2. K{inner_iters=1} converges, K{inner_iters=2} converges.
#[test]
fn kcycle_inner_iters_variants() {
    let n = 80;
    let (a, _, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);

    for inner in [1usize, 2] {
        let hier = AmgHierarchy::build(a.clone(), amg_config());
        let prec = AmgPrecond::new(hier).with_cycle(CycleType::K { inner_iters: inner });
        let mut x = DenseVec::zeros(n);
        let res = ConjugateGradient::<f64>::default()
            .solve(&a, Some(&prec), &b_vec, &mut x, &cg_params())
            .unwrap();
        assert!(res.converged, "K{{inner={inner}}} did not converge");
        assert!(rel_residual(&a, &x, &b_vec) < 1e-7);
    }
}

/// 3. K-cycle as AmgPrecond + CG converges.
#[test]
fn kcycle_as_precond_cg_converges() {
    let n = 200;
    let (a, _, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);

    let hier = AmgHierarchy::build(a.clone(), amg_config());
    let prec = AmgPrecond::new(hier).with_cycle(CycleType::K { inner_iters: 2 });

    let mut x = DenseVec::zeros(n);
    let res = ConjugateGradient::<f64>::default()
        .solve(&a, Some(&prec), &b_vec, &mut x, &cg_params())
        .unwrap();
    assert!(res.converged, "K-cycle+CG did not converge");
    assert!(rel_residual(&a, &x, &b_vec) < 1e-7);
}

/// 4. Repeated K-cycles applied as iterative solver converge in few iterations.
#[test]
fn kcycle_multi_cycle_convergence_rate() {
    let n = 200;
    let (a, _, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);

    // Use K-cycle as a standalone iterative solver (repeated application).
    // Each K-cycle uses the current residual b - A*x and adds a correction.
    let hier = AmgHierarchy::build(a.clone(), amg_config());
    let mut x = DenseVec::zeros(n);

    for _iter in 0..30 {
        // Compute current residual.
        let mut ax = DenseVec::zeros(n);
        a.apply(&x, &mut ax);
        let r: Vec<f64> = b_vec.as_slice().iter().zip(ax.as_slice()).map(|(bi, ai)| bi - ai).collect();
        let r_vec = DenseVec::from_vec(r);

        // Apply one K-cycle as a preconditioner (correction from 0).
        let mut e = DenseVec::zeros(n);
        hier.apply_cycle(&r_vec, &mut e, CycleType::K { inner_iters: 2 });

        // x += correction.
        let xs = x.as_mut_slice();
        for i in 0..n { xs[i] += e.as_slice()[i]; }

        if rel_residual(&a, &x, &b_vec) < 1e-8 { break; }
    }

    assert!(rel_residual(&a, &x, &b_vec) < 1e-4,
        "K-cycle iterative solver residual too large: {}", rel_residual(&a, &x, &b_vec));
}

/// 5. K-cycle doesn't produce NaN (robustness check).
#[test]
fn kcycle_no_nan() {
    let n = 50;
    let (a, _, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);

    let hier = AmgHierarchy::build(a.clone(), amg_config());
    let prec = AmgPrecond::new(hier).with_cycle(CycleType::K { inner_iters: 2 });
    let mut x = DenseVec::zeros(n);
    let res = ConjugateGradient::<f64>::default()
        .solve(&a, Some(&prec), &b_vec, &mut x, &cg_params())
        .unwrap();
    let rr = rel_residual(&a, &x, &b_vec);
    assert!(!rr.is_nan(), "K-cycle produced NaN");
    assert!(!rr.is_infinite(), "K-cycle produced Inf");
    assert!(res.converged);
}

/// 6. inner_iters=0 falls back to single V-cycle behaviour (both converge via PCG).
#[test]
fn kcycle_inner_iters_zero_fallback() {
    let n = 100;
    let (a, _, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);

    // K{inner_iters=0} should behave like a V-cycle.
    let hier_k = AmgHierarchy::build(a.clone(), amg_config());
    let prec_k = AmgPrecond::new(hier_k).with_cycle(CycleType::K { inner_iters: 0 });
    let mut x_k = DenseVec::zeros(n);
    let res_k = ConjugateGradient::<f64>::default()
        .solve(&a, Some(&prec_k), &b_vec, &mut x_k, &cg_params())
        .unwrap();

    let hier_v = AmgHierarchy::build(a.clone(), amg_config());
    let prec_v = AmgPrecond::new(hier_v).with_cycle(CycleType::V);
    let mut x_v = DenseVec::zeros(n);
    let res_v = ConjugateGradient::<f64>::default()
        .solve(&a, Some(&prec_v), &b_vec, &mut x_v, &cg_params())
        .unwrap();

    assert!(res_k.converged, "K{{0}}+CG did not converge");
    assert!(res_v.converged, "V-cycle+CG did not converge");
    assert!(rel_residual(&a, &x_k, &b_vec) < 1e-7);
    assert!(rel_residual(&a, &x_v, &b_vec) < 1e-7);
}
