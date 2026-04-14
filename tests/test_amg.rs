//! Sprint 4 — AMG tests.
//!
//! Covers SA-AMG and RS-AMG setup, V-cycle standalone convergence, and
//! AMG-preconditioned CG on 1D/2D Poisson problems.

mod common;

use linger::{
    amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, CycleType, SmootherType},
    iterative::{ConjugateGradient, Gmres},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, SolverParams, VerboseLevel, Vector,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn default_params(rtol: f64, max_iter: usize) -> SolverParams {
    SolverParams { rtol, max_iter, verbose: VerboseLevel::Silent, ..Default::default() }
}

fn solution_error(x: &[f64], x_exact: &[f64]) -> f64 {
    let e: f64 = x.iter().zip(x_exact).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let n: f64 = x_exact.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if n > 0.0 { e / n } else { e }
}

fn rel_residual(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let n = b.len();
    let mut ax = DenseVec::zeros(n);
    use linger::core::operator::LinearOperator;
    a.apply(x, &mut ax);
    let r: f64 = ax.as_slice().iter().zip(b.as_slice()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.as_slice().iter().map(|&v| v * v).sum::<f64>().sqrt();
    if nb > 0.0 { r / nb } else { r }
}

// ─── hierarchy build tests ────────────────────────────────────────────────────

#[test]
fn amg_builds_hierarchy_sa() {
    let n = 50;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let config = AmgConfig { coarse_threshold: 4, ..Default::default() };
    let hier = AmgHierarchy::build(a, config);
    assert!(hier.n_levels() >= 2, "SA-AMG should build at least 2 levels for n=50");
    assert!(hier.levels.last().unwrap().a.nrows() <= 10);
}

#[test]
fn amg_builds_hierarchy_rs() {
    let n = 50;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let config = AmgConfig {
        strategy: CoarsenStrategy::RugeStüben,
        coarse_threshold: 4,
        ..Default::default()
    };
    let hier = AmgHierarchy::build(a, config);
    assert!(hier.n_levels() >= 2, "RS-AMG should build at least 2 levels for n=50");
}

// ─── standalone V-cycle convergence ──────────────────────────────────────────

#[test]
fn vcycle_applies_nontrivially() {
    // Verify that the V-cycle preconditioner applies a non-trivial operator:
    // M⁻¹ b should be non-zero and have a bounded norm.
    let n = 30;
    let (a, _, b_vec) = common::make_poisson_1d::<f64>(n);
    let b = DenseVec::from_vec(b_vec);

    let config = AmgConfig { coarse_threshold: 4, ..Default::default() };
    let hier = AmgHierarchy::build(a, config);

    let mut y = DenseVec::zeros(n);
    hier.apply_cycle(&b, &mut y, CycleType::V);

    let y_norm: f64 = y.as_slice().iter().map(|&v| v * v).sum::<f64>().sqrt();
    let b_norm: f64 = b.as_slice().iter().map(|&v| v * v).sum::<f64>().sqrt();
    // M⁻¹ b should be non-zero and not wildly large.
    assert!(y_norm > 1e-10 * b_norm, "V-cycle output should be non-trivial");
    assert!(y_norm < 1e6 * b_norm, "V-cycle output should not blow up");
}

#[test]
fn vcycle_converges_standalone() {
    // Repeated V-cycles should converge to rtol=1e-8.
    let n = 30;
    let (a, x_exact, b_vec) = common::make_poisson_1d::<f64>(n);
    let b = DenseVec::from_vec(b_vec);

    let config = AmgConfig { pre_sweeps: 2, post_sweeps: 2, coarse_threshold: 4, ..Default::default() };
    let hier = AmgHierarchy::build(a.clone(), config);

    let mut x = DenseVec::zeros(n);
    for _ in 0..30 {
        let r = rel_residual(&a, &x, &b);
        if r < 1e-8 { break; }
        // Correction cycle: apply to residual, add to x.
        let nn = b.len();
        let mut ax = DenseVec::zeros(nn);
        use linger::core::operator::LinearOperator;
        a.apply(&x, &mut ax);
        let mut res = DenseVec::zeros(nn);
        let rs = res.as_mut_slice();
        for i in 0..nn { rs[i] = b.as_slice()[i] - ax.as_slice()[i]; }
        let mut e = DenseVec::zeros(nn);
        hier.apply_cycle(&res, &mut e, CycleType::V);
        let xs = x.as_mut_slice();
        let es = e.as_slice();
        for i in 0..nn { xs[i] += es[i]; }
    }

    let final_res = rel_residual(&a, &x, &b);
    assert!(final_res < 1e-6, "Standalone V-cycle should converge: final_res={final_res:.3e}");
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-5);
}

// ─── AMG-preconditioned CG ────────────────────────────────────────────────────

#[test]
fn amg_pcg_poisson_1d_sa() {
    let n = 100;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);

    let config = AmgConfig { coarse_threshold: 4, ..Default::default() };
    let hier  = AmgHierarchy::build(a.clone(), config);
    let precond = AmgPrecond::new(hier);

    let cg  = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&precond), &b_vec, &mut x, &default_params(1e-10, 200)).unwrap();

    assert!(res.converged,
        "SA-AMG PCG didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn amg_pcg_poisson_1d_rs() {
    let n = 100;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);

    let config = AmgConfig {
        strategy: CoarsenStrategy::RugeStüben,
        coarse_threshold: 4,
        ..Default::default()
    };
    let hier    = AmgHierarchy::build(a.clone(), config);
    let precond = AmgPrecond::new(hier);

    let cg  = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&precond), &b_vec, &mut x, &default_params(1e-10, 200)).unwrap();

    assert!(res.converged,
        "RS-AMG PCG didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn amg_pcg_poisson_2d() {
    let nx = 16;
    let ny = 16;
    let (a, x_exact, b) = common::make_poisson_2d::<f64>(nx, ny);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(nx * ny);

    let config = AmgConfig { coarse_threshold: 4, ..Default::default() };
    let hier    = AmgHierarchy::build(a.clone(), config);
    let precond = AmgPrecond::new(hier);

    let cg  = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&precond), &b_vec, &mut x, &default_params(1e-9, 300)).unwrap();

    assert!(res.converged,
        "AMG PCG didn't converge on 2D Poisson 16×16; iters={}, rel={:.3e}",
        res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-7);
}

#[test]
fn amg_pcg_fewer_iters_than_unpreconditioned() {
    // AMG-PCG should need significantly fewer iterations than plain CG.
    // Use constant rhs [1,1,...] which exercises all eigenmodes.
    let n = 200;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(vec![1.0f64; n]);
    let params = default_params(1e-9, 2000);
    let cg = ConjugateGradient::<f64>::default();

    let mut x1 = DenseVec::zeros(n);
    let r1 = cg.solve(&a, None, &b_vec, &mut x1, &params).unwrap();

    let config  = AmgConfig { coarse_threshold: 4, ..Default::default() };
    let hier    = AmgHierarchy::build(a.clone(), config);
    let precond = AmgPrecond::new(hier);
    let mut x2  = DenseVec::zeros(n);
    let r2 = cg.solve(&a, Some(&precond), &b_vec, &mut x2, &params).unwrap();

    assert!(r1.converged && r2.converged);
    assert!(r2.iterations < r1.iterations / 2,
        "AMG-PCG ({} iters) should be >2× faster than CG ({} iters)",
        r2.iterations, r1.iterations);
}

// ─── W-cycle test ─────────────────────────────────────────────────────────────

#[test]
fn amg_wcycle_converges() {
    let n = 50;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);

    let config  = AmgConfig { coarse_threshold: 4, ..Default::default() };
    let hier    = AmgHierarchy::build(a.clone(), config);
    let precond = AmgPrecond::new(hier).with_cycle(CycleType::W);

    let cg  = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&precond), &b_vec, &mut x, &default_params(1e-10, 100)).unwrap();

    assert!(res.converged, "W-cycle AMG PCG didn't converge; iters={}", res.iterations);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

// ─── GS smoother ─────────────────────────────────────────────────────────────

#[test]
fn amg_pcg_gs_smoother() {
    let n = 100;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);

    let config = AmgConfig {
        smoother: SmootherType::SymmetricGaussSeidel,
        coarse_threshold: 4,
        ..Default::default()
    };
    let hier    = AmgHierarchy::build(a.clone(), config);
    let precond = AmgPrecond::new(hier);

    let cg  = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&precond), &b_vec, &mut x, &default_params(1e-10, 200)).unwrap();

    assert!(res.converged, "AMG(SGS)-PCG didn't converge; iters={}", res.iterations);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn amg_air_gmres_nonsymmetric_convdiff_1d() {
    // AIR path should be usable on a nonsymmetric system.
    let n = 120;
    let (a, x_exact, b) = common::make_nonsymmetric_convdiff::<f64>(n, 8.0);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);

    let config = AmgConfig {
        strategy: CoarsenStrategy::Air,
        coarse_threshold: 4,
        ..Default::default()
    };
    let hier = AmgHierarchy::build(a.clone(), config);
    let precond = AmgPrecond::new(hier);

    let solver = Gmres::<f64>::new(30);
    let res = solver
        .solve(&a, Some(&precond), &b_vec, &mut x, &default_params(1e-8, 250))
        .unwrap();

    assert!(res.converged,
        "AIR-AMG GMRES didn't converge on nonsymmetric convdiff; iters={}, rel={:.3e}",
        res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-6);
}
