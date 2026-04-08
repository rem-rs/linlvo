//! Integration tests for H2: AMG level diagnostics.

use linger::{
    amg::{AmgConfig, AmgHierarchy, CoarsenStrategy},
    sparse::{CooMatrix, CsrMatrix},
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

fn laplacian_2d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut coo = CooMatrix::new(nn, nn);
    for i in 0..n {
        for j in 0..n {
            let id = i * n + j;
            coo.push(id, id, 4.0);
            if j > 0   { coo.push(id, id - 1, -1.0); coo.push(id - 1, id, -1.0); }
            if i > 0   { coo.push(id, id - n, -1.0); coo.push(id - n, id, -1.0); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn config_sa(coarse_threshold: usize) -> AmgConfig {
    AmgConfig { coarse_threshold, ..Default::default() }
}

fn config_rs(coarse_threshold: usize) -> AmgConfig {
    AmgConfig {
        strategy: CoarsenStrategy::RugeStüben,
        coarse_threshold,
        ..Default::default()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// 1. level_ndof(0) returns the fine-level problem size.
#[test]
fn amg_diag_level_ndof_fine() {
    let n = 100;
    let a = laplacian_1d(n);
    let hier = AmgHierarchy::build(a, config_sa(4));
    assert_eq!(hier.level_ndof(0), Some(n),
        "level 0 ndof should be {n}");
}

/// 2. level_nnz(0) returns the number of fine-level non-zeros.
#[test]
fn amg_diag_level_nnz_fine() {
    let n = 50;
    let a = laplacian_1d(n);
    // Tridiagonal 1D Laplacian has 3n - 2 non-zeros.
    let expected_nnz = 3 * n - 2;
    let hier = AmgHierarchy::build(a, config_sa(4));
    assert_eq!(hier.level_nnz(0), Some(expected_nnz),
        "level 0 nnz should be {expected_nnz}");
}

/// 3. level_ndof for out-of-bounds level returns None.
#[test]
fn amg_diag_level_ndof_oob() {
    let a = laplacian_1d(50);
    let hier = AmgHierarchy::build(a, config_sa(4));
    assert_eq!(hier.level_ndof(hier.n_levels()), None,
        "out-of-bounds level should return None");
}

/// 4. Coarsest level is_coarsest = true in level_info.
#[test]
fn amg_diag_coarsest_flag() {
    let n = 100;
    let a = laplacian_1d(n);
    let hier = AmgHierarchy::build(a, config_sa(4));
    let infos = hier.level_info();
    assert!(!infos.is_empty());
    // Only the last level should be coarsest.
    let coarsest_count = infos.iter().filter(|l| l.is_coarsest).count();
    assert_eq!(coarsest_count, 1,
        "exactly one level should be coarsest; got {coarsest_count}");
    assert!(infos.last().unwrap().is_coarsest,
        "last level must be coarsest");
}

/// 5. Levels are strictly decreasing in ndof.
#[test]
fn amg_diag_levels_decreasing_ndof() {
    let n = 200;
    let a = laplacian_1d(n);
    let hier = AmgHierarchy::build(a, config_sa(4));
    let infos = hier.level_info();
    for w in infos.windows(2) {
        assert!(w[0].ndof > w[1].ndof,
            "level {} ndof={} should be > level {} ndof={}",
            w[0].level, w[0].ndof, w[1].level, w[1].ndof);
    }
}

/// 6. operator_complexity ≥ 1.0 (each extra level adds nnz).
#[test]
fn amg_diag_operator_complexity_gte_1() {
    let n = 200;
    let a = laplacian_1d(n);
    let hier = AmgHierarchy::build(a, config_sa(4));
    let oc = hier.operator_complexity();
    assert!(oc >= 1.0,
        "operator_complexity={oc} must be ≥ 1.0");
}

/// 7. grid_complexity ≥ 1.0.
#[test]
fn amg_diag_grid_complexity_gte_1() {
    let n = 200;
    let a = laplacian_1d(n);
    let hier = AmgHierarchy::build(a, config_sa(4));
    let gc = hier.grid_complexity();
    assert!(gc >= 1.0,
        "grid_complexity={gc} must be ≥ 1.0");
}

/// 8. coarsen_ratios() has n_levels-1 entries, all ≥ 1.
#[test]
fn amg_diag_coarsen_ratios_valid() {
    let n = 100;
    let a = laplacian_1d(n);
    let hier = AmgHierarchy::build(a, config_sa(4));
    let ratios = hier.coarsen_ratios();
    assert_eq!(ratios.len(), hier.n_levels() - 1,
        "coarsen_ratios length should be n_levels-1");
    for (i, &r) in ratios.iter().enumerate() {
        assert!(r > 1.0,
            "coarsen_ratio[{i}]={r} should be > 1.0 (coarsening reduces DOFs)");
    }
}

/// 9. Level info for 2D Laplacian (more levels expected).
#[test]
fn amg_diag_2d_laplacian_multi_level() {
    let n = 16;
    let a = laplacian_2d(n); // 256 DOF
    let hier = AmgHierarchy::build(a, config_sa(4));
    assert!(hier.n_levels() >= 3,
        "2D 16×16 Laplacian should have ≥3 levels, got {}", hier.n_levels());
    let infos = hier.level_info();
    assert_eq!(infos[0].ndof, 256, "finest level should have 256 DOF");
}

/// 10. RS-AMG diagnostics: operator complexity is finite and ≥ 1.
#[test]
fn amg_diag_rs_operator_complexity() {
    let n = 100;
    let a = laplacian_1d(n);
    let hier = AmgHierarchy::build(a, config_rs(4));
    let oc = hier.operator_complexity();
    assert!(oc.is_finite() && oc >= 1.0,
        "RS-AMG operator_complexity={oc} should be finite and ≥ 1.0");
}

/// 11. print_info does not panic.
#[test]
fn amg_diag_print_info_no_panic() {
    let a = laplacian_1d(50);
    let hier = AmgHierarchy::build(a, config_sa(4));
    hier.print_info(); // should not panic
}

/// 12. Single-level hierarchy (n ≤ coarse_threshold) has complexity 1.0.
#[test]
fn amg_diag_single_level_complexity() {
    let n = 5;
    let a = laplacian_1d(n);
    let hier = AmgHierarchy::build(a, AmgConfig { coarse_threshold: 10, ..Default::default() });
    assert_eq!(hier.n_levels(), 1, "should be single level");
    assert!((hier.operator_complexity() - 1.0).abs() < 1e-12,
        "single-level OC should be exactly 1.0");
    assert!((hier.grid_complexity() - 1.0).abs() < 1e-12,
        "single-level GC should be exactly 1.0");
    assert_eq!(hier.coarsen_ratios().len(), 0,
        "single-level hierarchy has no coarsen ratios");
}
