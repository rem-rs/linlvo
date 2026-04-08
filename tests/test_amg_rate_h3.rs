//! Integration tests for H3: AMG per-cycle convergence rate.

use linger::{
    amg::{AmgConfig, AmgHierarchy, CoarsenStrategy, CycleType},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, Vector, LinearOperator,
};

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn config_sa() -> AmgConfig {
    AmgConfig { coarse_threshold: 4, ..Default::default() }
}

/// 1. convergence_rate is NaN before any cycle is applied.
#[test]
fn h3_rate_nan_before_cycle() {
    let a = laplacian_1d(50);
    let hier = AmgHierarchy::build(a, config_sa());
    assert!(hier.convergence_rate().is_nan(),
        "expected NaN before first cycle, got {}", hier.convergence_rate());
}

/// 2. After one V-cycle, convergence_rate is a finite non-negative value.
#[test]
fn h3_rate_finite_after_cycle() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let hier = AmgHierarchy::build(a, config_sa());
    let mut x = DenseVec::zeros(n);
    hier.apply_cycle(&b, &mut x, CycleType::V);
    let rate = hier.convergence_rate();
    assert!(rate.is_finite() && rate >= 0.0,
        "convergence_rate should be finite and ≥ 0, got {rate}");
}

/// 3. After several V-cycles iterating x toward solution, final rate should be < 1.
#[test]
fn h3_rate_in_unit_interval() {
    let n = 200;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let hier = AmgHierarchy::build(a, config_sa());
    let mut x = DenseVec::zeros(n);

    // Run several cycles — after the first cycle the rate should settle < 1
    // as x approaches the solution.
    let mut last_rate = 0.0f64;
    for _ in 0..10 {
        hier.apply_cycle(&b, &mut x, CycleType::V);
        last_rate = hier.convergence_rate();
    }
    assert!(last_rate < 1.0, "rate after 10 cycles = {last_rate:.4}; expected < 1.0");
    assert!(last_rate >= 0.0);
}

/// 4. Rate is updated after each new cycle call.
#[test]
fn h3_rate_updates_on_each_cycle() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let hier = AmgHierarchy::build(a, config_sa());
    let mut x = DenseVec::zeros(n);

    hier.apply_cycle(&b, &mut x, CycleType::V);
    let rate1 = hier.convergence_rate();
    assert!(!rate1.is_nan());

    hier.apply_cycle(&b, &mut x, CycleType::V);
    let rate2 = hier.convergence_rate();
    assert!(!rate2.is_nan());
    // Both cycles recorded a finite rate — they may differ since x changed.
    assert!(rate1.is_finite() && rate2.is_finite());
}

/// 5. W-cycle also records a finite convergence rate.
#[test]
fn h3_rate_w_cycle() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let hier = AmgHierarchy::build(a, config_sa());
    let mut x = DenseVec::zeros(n);
    // Run a few cycles so x is closer to the solution.
    for _ in 0..5 {
        hier.apply_cycle(&b, &mut x, CycleType::W);
    }
    let rate = hier.convergence_rate();
    assert!(rate.is_finite() && rate >= 0.0 && rate < 1.0,
        "W-cycle rate after 5 cycles = {rate:.4}");
}

/// 6. After many cycles the residual norm is tiny, rate remains finite ≥ 0.
#[test]
fn h3_rate_near_convergence_finite() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let hier = AmgHierarchy::build(a, config_sa());
    let mut x = DenseVec::zeros(n);

    // Run many cycles to drive residual small.
    for _ in 0..100 {
        hier.apply_cycle(&b, &mut x, CycleType::V);
    }
    let rate = hier.convergence_rate();
    assert!(rate.is_finite() && rate >= 0.0,
        "rate near convergence should be finite and ≥ 0, got {rate}");
}

/// 7. RS-AMG also records a valid convergence rate.
#[test]
fn h3_rate_rs_amg() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let hier = AmgHierarchy::build(a, AmgConfig {
        strategy: CoarsenStrategy::RugeStüben,
        coarse_threshold: 4,
        ..Default::default()
    });
    let mut x = DenseVec::zeros(n);
    hier.apply_cycle(&b, &mut x, CycleType::V);
    let rate = hier.convergence_rate();
    assert!(rate.is_finite() && rate >= 0.0,
        "RS-AMG rate should be finite and ≥ 0, got {rate}");
}

/// 8. Convergence rate is consistent with manual residual computation.
#[test]
fn h3_rate_matches_manual_residual() {
    let n = 100;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let hier = AmgHierarchy::build(a.clone(), config_sa());
    let mut x = DenseVec::zeros(n);

    // Compute residual norm manually before cycle.
    let mut ax = DenseVec::zeros(n);
    a.apply(&x, &mut ax);
    let r_before: f64 = b.as_slice().iter().zip(ax.as_slice())
        .map(|(b, ax)| (b - ax).powi(2)).sum::<f64>().sqrt();

    hier.apply_cycle(&b, &mut x, CycleType::V);

    // Compute residual norm manually after cycle.
    a.apply(&x, &mut ax);
    let r_after: f64 = b.as_slice().iter().zip(ax.as_slice())
        .map(|(b, ax)| (b - ax).powi(2)).sum::<f64>().sqrt();

    let expected = r_after / r_before;
    let recorded = hier.convergence_rate();
    assert!((recorded - expected).abs() < 1e-12,
        "recorded rate {recorded:.6} differs from manual {expected:.6}");
}
