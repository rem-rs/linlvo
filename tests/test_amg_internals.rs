//! Tests for AMG sub-modules (strength, coarsen_rs, coarsen_agg) and ILU(k)
//! with fill levels > 1.
//!
//! These cover code paths exercised only indirectly by the end-to-end AMG tests.

mod common;

use linger::{
    amg::{AmgConfig, AmgHierarchy},
    sparse::{CooMatrix, CsrMatrix},
    IlukPrecond, DenseVec,
};

// Re-import the internal AMG functions we want to test directly.
// They are `pub` within the `amg` module.
use linger::amg::{CoarsenStrategy};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn csr_from_triplets(nrows: usize, ncols: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(nrows, ncols);
    for &(r, c, v) in entries {
        coo.push(r, c, v);
    }
    CsrMatrix::from_coo(&coo)
}

fn poisson_1d(n: usize) -> CsrMatrix<f64> {
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    a
}

// ─── strong_connections ───────────────────────────────────────────────────────
//
// We test through the public AMG hierarchy interface by observing what the
// setup phase produces, and also directly via the re-exported function.

use linger::amg::strength::strong_connections;

#[test]
fn strong_connections_all_offdiag_for_poisson() {
    // 1D Poisson: diagonal = 2, off-diagonal = -1.
    // For theta = 0.25: cutoff = 0.25 * 1.0 = 0.25; all off-diagonals (|val|=1.0) qualify.
    let a = poisson_1d(6);
    let s = strong_connections(&a, 0.25);

    // Every off-diagonal entry should appear in s.
    let a_offdiag_nnz = a.nnz() - a.nrows();  // subtract diagonal entries
    assert_eq!(s.nnz(), a_offdiag_nnz,
        "All off-diagonals of Poisson should be strong at theta=0.25");
}

#[test]
fn strong_connections_no_connections_high_theta() {
    // theta=2.0: cutoff = 2.0 * max_off = 2.0 * 1.0 = 2.0.
    // No entry has |a_ij| >= 2.0, so strong graph is empty.
    let a = poisson_1d(6);
    let s = strong_connections(&a, 2.0);
    assert_eq!(s.nnz(), 0, "No strong connections expected for theta=2.0");
}

#[test]
fn strong_connections_diagonal_only_matrix() {
    // Diagonal matrix has no off-diagonal entries → strong graph is always empty.
    let a = csr_from_triplets(4, 4, &[(0,0,1.0),(1,1,2.0),(2,2,3.0),(3,3,4.0)]);
    let s = strong_connections(&a, 0.25);
    assert_eq!(s.nnz(), 0);
}

#[test]
fn strong_connections_asymmetric_row() {
    // A row where one off-diagonal is large and another is small.
    // Row 0: [4, -4, -0.1]  → max_off = 4.0, cutoff(theta=0.5) = 2.0
    // Only col 1 (|val|=4.0 >= 2.0) should be a strong connection.
    let a = csr_from_triplets(3, 3, &[
        (0, 0, 4.0), (0, 1, -4.0), (0, 2, -0.1),
        (1, 1, 4.0), (1, 0, -4.0),
        (2, 2, 4.0), (2, 0, -0.1),
    ]);
    let s = strong_connections(&a, 0.5);
    // Row 0: strong connection to col 1 only (|-4.0| >= 0.5*4.0=2.0 ✓; |-0.1| < 2.0 ✗)
    let rp = s.row_ptr();
    let ci = s.col_idx();
    let row0: Vec<usize> = ci[rp[0]..rp[1]].to_vec();
    assert!(row0.contains(&1), "col 1 must be a strong connection from row 0");
    assert!(!row0.contains(&2), "col 2 must NOT be a strong connection from row 0");
}

#[test]
fn strong_connections_dimensions_match_input() {
    let a = poisson_1d(10);
    let s = strong_connections(&a, 0.25);
    assert_eq!(s.nrows(), a.nrows());
    assert_eq!(s.ncols(), a.ncols());
}

// ─── RS coarsening ────────────────────────────────────────────────────────────

use linger::amg::coarsen_rs::{NodeType, coarse_index_map, rs_coarsen};

#[test]
fn rs_coarsen_all_nodes_decided() {
    let a = poisson_1d(8);
    let s = strong_connections(&a, 0.25);
    let status = rs_coarsen::<f64>(&s);
    assert_eq!(status.len(), 8);
    for &st in &status {
        assert_ne!(st, NodeType::Undecided, "All nodes must be decided after RS coarsening");
    }
}

#[test]
fn rs_coarsen_has_coarse_and_fine_nodes() {
    // For n > 1, RS should produce at least one C-point and one F-point.
    let a = poisson_1d(8);
    let s = strong_connections(&a, 0.25);
    let status = rs_coarsen::<f64>(&s);
    let n_coarse = status.iter().filter(|&&s| s == NodeType::Coarse).count();
    let n_fine   = status.iter().filter(|&&s| s == NodeType::Fine).count();
    assert!(n_coarse > 0, "Must have at least one C-point");
    assert!(n_fine   > 0, "Must have at least one F-point");
}

#[test]
fn rs_coarsen_coarsens_by_roughly_half() {
    // For 1D Poisson (fully connected path graph), RS should produce ~n/2 C-points.
    let n = 20;
    let a = poisson_1d(n);
    let s = strong_connections(&a, 0.25);
    let status = rs_coarsen::<f64>(&s);
    let n_coarse = status.iter().filter(|&&s| s == NodeType::Coarse).count();
    // Allow range [n/4, 3*n/4] to be robust.
    assert!(n_coarse >= n / 4 && n_coarse <= 3 * n / 4,
        "Expected ~n/2 C-points for 1D path, got {n_coarse}/{n}");
}

#[test]
fn coarse_index_map_correct_count() {
    let statuses = vec![
        NodeType::Coarse,
        NodeType::Fine,
        NodeType::Coarse,
        NodeType::Fine,
        NodeType::Coarse,
    ];
    let (nc, map) = coarse_index_map(&statuses);
    assert_eq!(nc, 3, "Should have 3 coarse nodes");
    assert_eq!(map[0], 0);
    assert_eq!(map[1], usize::MAX, "F-point should map to usize::MAX");
    assert_eq!(map[2], 1);
    assert_eq!(map[3], usize::MAX);
    assert_eq!(map[4], 2);
}

#[test]
fn coarse_index_map_all_fine() {
    let statuses = vec![NodeType::Fine; 4];
    let (nc, map) = coarse_index_map(&statuses);
    assert_eq!(nc, 0);
    assert!(map.iter().all(|&m| m == usize::MAX));
}

// ─── SA aggregation ───────────────────────────────────────────────────────────

use linger::amg::coarsen_agg::{build_aggregates, tentative_prolongation};

#[test]
fn build_aggregates_covers_all_nodes() {
    // Every node must be assigned to some aggregate.
    let a = poisson_1d(10);
    let s = strong_connections(&a, 0.25);
    let agg_id = build_aggregates::<f64>(&s);

    assert_eq!(agg_id.len(), 10);
    for (i, &id) in agg_id.iter().enumerate() {
        assert_ne!(id, usize::MAX, "Node {i} has no aggregate");
    }
}

#[test]
fn build_aggregates_count_reasonable() {
    // For 1D Poisson n=12, greedy SA should produce between 1 and n aggregates.
    let n = 12;
    let a = poisson_1d(n);
    let s = strong_connections(&a, 0.25);
    let agg_id = build_aggregates::<f64>(&s);
    let n_agg = agg_id.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    assert!(n_agg >= 1 && n_agg <= n,
        "Expected 1..={n} aggregates, got {n_agg}");
}

#[test]
fn build_aggregates_disconnected_graph_each_own_aggregate() {
    // Diagonal-only matrix → no strong connections → every node is its own aggregate.
    let a = csr_from_triplets(4, 4, &[(0,0,1.0),(1,1,1.0),(2,2,1.0),(3,3,1.0)]);
    let s = strong_connections(&a, 0.25);
    let agg_id = build_aggregates::<f64>(&s);
    // With no off-diagonal strong connections, each seed forms a singleton aggregate.
    let n_agg = agg_id.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    assert_eq!(n_agg, 4, "4 disconnected nodes → 4 singletons");
    // All nodes assigned.
    assert!(agg_id.iter().all(|&id| id != usize::MAX));
}

#[test]
fn tentative_prolongation_shape() {
    // P₀ must have shape n_fine × n_agg and exactly one nonzero per row.
    let a = poisson_1d(8);
    let s = strong_connections(&a, 0.25);
    let agg_id  = build_aggregates::<f64>(&s);
    let n_coarse = agg_id.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    let p0: CsrMatrix<f64> = tentative_prolongation::<f64>(&agg_id, n_coarse);

    assert_eq!(p0.nrows(), 8, "P₀ rows must equal n_fine");
    assert_eq!(p0.ncols(), n_coarse, "P₀ cols must equal n_agg");
    assert_eq!(p0.nnz(), 8, "Each fine node maps to exactly one coarse DOF");
}

#[test]
fn tentative_prolongation_unit_values() {
    // Every nonzero in P₀ should be 1.0.
    let a = poisson_1d(6);
    let s = strong_connections(&a, 0.25);
    let agg_id   = build_aggregates::<f64>(&s);
    let n_coarse = agg_id.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    let p0: CsrMatrix<f64> = tentative_prolongation::<f64>(&agg_id, n_coarse);

    for (_, _, v) in p0.triplets() {
        assert!((v - 1.0).abs() < 1e-14, "All P₀ values must be 1.0, got {v}");
    }
}

#[test]
fn tentative_prolongation_spmv_extracts_aggregate() {
    // P₀ * e_k (standard basis for aggregate k) should give 1.0 for nodes in k,
    // 0.0 for nodes outside k.  Verify for the first aggregate.
    let agg_id   = vec![0usize, 0, 1, 1];   // 4 nodes, 2 aggregates
    let n_coarse = 2;
    let p0: CsrMatrix<f64> = tentative_prolongation::<f64>(&agg_id, n_coarse);

    // P₀ * [1, 0] should give [1, 1, 0, 0]
    let mut y = vec![0.0f64; 4];
    p0.spmv(&[1.0, 0.0], &mut y);
    assert_eq!(y, vec![1.0, 1.0, 0.0, 0.0]);

    // P₀ * [0, 1] should give [0, 0, 1, 1]
    p0.spmv(&[0.0, 1.0], &mut y);
    assert_eq!(y, vec![0.0, 0.0, 1.0, 1.0]);
}

// ─── AMG hierarchy properties ─────────────────────────────────────────────────

#[test]
fn amg_hierarchy_levels_decrease_monotonically() {
    // Each level must be strictly smaller than the previous.
    let a = poisson_1d(30);
    let config = AmgConfig { coarse_threshold: 4, ..Default::default() };
    let hier = AmgHierarchy::build(a, config);

    for (l, pair) in hier.levels.windows(2).enumerate() {
        assert!(pair[1].a.nrows() < pair[0].a.nrows(),
            "Level {} (n={}) must be smaller than level {} (n={})",
            l + 1, pair[1].a.nrows(), l, pair[0].a.nrows());
    }
}

#[test]
fn amg_hierarchy_rs_levels_decrease_monotonically() {
    let a = poisson_1d(30);
    let config = AmgConfig {
        strategy: CoarsenStrategy::RugeStüben,
        coarse_threshold: 4,
        ..Default::default()
    };
    let hier = AmgHierarchy::build(a, config);

    for (l, pair) in hier.levels.windows(2).enumerate() {
        assert!(pair[1].a.nrows() < pair[0].a.nrows(),
            "RS level {} must be coarser than level {}", l + 1, l);
    }
}

#[test]
fn amg_hierarchy_air_levels_decrease_monotonically() {
    let a = poisson_1d(30);
    let config = AmgConfig {
        strategy: CoarsenStrategy::Air,
        coarse_threshold: 4,
        ..Default::default()
    };
    let hier = AmgHierarchy::build(a, config);

    assert!(hier.levels.len() >= 2, "AIR hierarchy should have at least 2 levels");
    for (l, pair) in hier.levels.windows(2).enumerate() {
        assert!(pair[1].a.nrows() < pair[0].a.nrows(),
            "AIR level {} must be coarser than level {}", l + 1, l);
    }
}

#[test]
fn amg_hierarchy_coarsest_level_below_threshold() {
    let threshold = 5;
    let a = poisson_1d(50);
    let config = AmgConfig { coarse_threshold: threshold, ..Default::default() };
    let hier = AmgHierarchy::build(a, config);
    let coarsest = hier.levels.last().unwrap();
    assert!(coarsest.a.nrows() <= threshold,
        "Coarsest level n={} must be <= threshold={threshold}", coarsest.a.nrows());
}

// ─── ILU(k) higher fill levels ────────────────────────────────────────────────

#[test]
fn iluk_k2_converges_on_poisson_1d() {
    use linger::{ConjugateGradient, KrylovSolver, SolverParams, VerboseLevel};
    let n = 50;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let ilu2 = IlukPrecond::<f64>::from_csr(&a, 2).unwrap();
    let params = SolverParams { rtol: 1e-8, max_iter: 200, verbose: VerboseLevel::Silent, ..Default::default() };

    let mut x = DenseVec::zeros(n);
    let res = ConjugateGradient::<f64>::default()
        .solve(&a, Some(&ilu2), &b, &mut x, &params)
        .unwrap();
    assert!(res.converged, "ILU(2)-PCG should converge");
}

#[test]
fn iluk_k2_no_more_iters_than_k1() {
    // ILU(2) has more fill than ILU(1), so it should converge in ≤ iterations.
    use linger::{ConjugateGradient, KrylovSolver, SolverParams, VerboseLevel};
    let n = 40;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let params = SolverParams { rtol: 1e-8, max_iter: 500, verbose: VerboseLevel::Silent, ..Default::default() };

    let ilu1 = IlukPrecond::<f64>::from_csr(&a, 1).unwrap();
    let ilu2 = IlukPrecond::<f64>::from_csr(&a, 2).unwrap();

    let mut x1 = DenseVec::zeros(n);
    let mut x2 = DenseVec::zeros(n);
    let r1 = ConjugateGradient::<f64>::default().solve(&a, Some(&ilu1), &b, &mut x1, &params).unwrap();
    let r2 = ConjugateGradient::<f64>::default().solve(&a, Some(&ilu2), &b, &mut x2, &params).unwrap();

    assert!(r1.converged && r2.converged);
    assert!(r2.iterations <= r1.iterations + 2,
        "ILU(2) ({} iters) should not need more than ILU(1) ({} iters) + small slack",
        r2.iterations, r1.iterations);
}

#[test]
fn iluk_k0_identical_to_ilu0_on_poisson() {
    // ILU(k=0) should give the same preconditioner as ILU(0).
    use linger::{ConjugateGradient, Ilu0Precond, KrylovSolver, SolverParams, VerboseLevel};
    let n = 20;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let params = SolverParams { rtol: 1e-8, max_iter: 200, verbose: VerboseLevel::Silent, ..Default::default() };

    let ilu0  = Ilu0Precond::<f64>::from_csr(&a).unwrap();
    let iluk0 = IlukPrecond::<f64>::from_csr(&a, 0).unwrap();

    let mut x0 = DenseVec::zeros(n);
    let mut xk = DenseVec::zeros(n);
    let r0 = ConjugateGradient::<f64>::default().solve(&a, Some(&ilu0),  &b, &mut x0, &params).unwrap();
    let rk = ConjugateGradient::<f64>::default().solve(&a, Some(&iluk0), &b, &mut xk, &params).unwrap();

    // Iteration counts should match (both use exact tridiagonal factorization).
    assert_eq!(r0.iterations, rk.iterations,
        "ILU(0) and ILU(k=0) should converge in the same number of iterations");
}
