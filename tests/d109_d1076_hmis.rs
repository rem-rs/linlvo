//! D1076 pins: the hypre-faithful `CoarsenStrategy::HmisAms` (HMIS(10) +
//! aggressive `Create2ndS` second shot + multipass / extended-i interpolation)
//! used by the AMS Pi-block solvers (`AmsPrecond::with_pi`).
//!
//! Semantics verified against hypre 2.28.0 itself (round-109, `tmp/d109a/`):
//! the level-0 multipass prolongator is bitwise identical to hypre's on the
//! tesla A_Pi operators and the V-cycle arm matches to ~1e-13 — provided the
//! level smoother is PLAIN symmetric Gauss-Seidel (hypre relax-8's l1 comes
//! from `ComputeL1Norms` option 4, which degenerates to the signed diagonal
//! on one rank; `par_amg_setup.c:3280` + `ams.c:678-692`).

use linlvo::amg::{AmgConfig, AmgHierarchy, CoarsenStrategy, CycleType, SmootherType};
use linlvo::core::vector::Vector as _;
use linlvo::sparse::{CooMatrix, CsrMatrix};

/// 2-D Poisson (5-point) operator on an n×n grid, CSR.
fn poisson_2d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n * n, n * n);
    for y in 0..n {
        for x in 0..n {
            let i = y * n + x;
            coo.push(i, i, 4.0);
            if x > 0 {
                coo.push(i, i - 1, -1.0);
            }
            if x + 1 < n {
                coo.push(i, i + 1, -1.0);
            }
            if y > 0 {
                coo.push(i, i - n, -1.0);
            }
            if y + 1 < n {
                coo.push(i, i + n, -1.0);
            }
        }
    }
    CsrMatrix::from_coo(&coo)
}

/// The HmisAms configuration used by `AmsPrecond::with_pi` (see ams.rs).
fn pi_config() -> AmgConfig {
    AmgConfig {
        strategy: CoarsenStrategy::HmisAms,
        smoother: SmootherType::SymmetricGaussSeidel,
        coarse_threshold: 2,
        max_levels: 25,
        coarsest_sweeps: Some(1),
        ..AmgConfig::default()
    }
}

/// Classical measure-0 strength: signed comparisons (positive off-diagonal
/// entries are never strong on non-negative diagonals), the
/// `max_row_sum = 0.9` weakening, and unit values.
#[test]
fn hmis_classical_strength_signed_semantics() {
    // Row 0: diag 4, off-diagonals -1.5 (strong at θ=0.25: -1.5 < 0.25·(-1.5)?)
    // Constructed explicitly below; θ = 0.25, min off-diag = -1.5 →
    // strong iff a_ij < -0.375: -1.5 strong, +0.5 never.
    let mut coo = CooMatrix::<f64>::new(3, 3);
    coo.push(0, 0, 4.0);
    coo.push(0, 1, -1.5);
    coo.push(0, 2, 0.5); // positive entry: never strong
    coo.push(1, 1, 4.0);
    coo.push(1, 0, -1.5);
    coo.push(2, 2, 1.0);
    let a = CsrMatrix::from_coo(&coo);
    let s = linlvo::amg::hmis::classical_strength(&a, 0.25);
    let row0: Vec<usize> = (0..s.row_ptr()[1])
        .map(|k| s.col_idx()[k])
        .collect();
    assert_eq!(row0, vec![1], "only the sufficiently-negative connection is strong");

    // Row-sum filter: |row_sum| = |4 - 1 - 1| = 2 ≤ 0.9·4 → not filtered;
    // now make a row whose |row_sum| dominates the diagonal.
    let mut coo2 = CooMatrix::<f64>::new(2, 2);
    coo2.push(0, 0, 1.0);
    coo2.push(0, 1, -3.0); // |row_sum| = 2 > 0.9·1 → all weak
    coo2.push(1, 1, 1.0);
    coo2.push(1, 0, -3.0);
    let a2 = CsrMatrix::from_coo(&coo2);
    let s2 = linlvo::amg::hmis::classical_strength(&a2, 0.25);
    assert_eq!(s2.row_ptr()[1] - s2.row_ptr()[0], 0, "ill-scaled row: all dependencies weak");
}

/// The aggressive second shot fires on level 0 only and strictly coarsens
/// more than the plain HMIS first pass; every hierarchy level coarsens.
#[test]
fn hmis_aggressive_second_shot_and_hierarchy() {
    let a = poisson_2d(24); // 576 dof
    let cfg = pi_config();
    let s = linlvo::amg::hmis::classical_strength(&a, cfg.theta);
    let cf1 = linlvo::amg::hmis::coarsen_hmis::<f64>(&s, false);
    let c_first = cf1.iter().filter(|&&m| m > 0).count();
    assert!(c_first > 0 && c_first < a.nrows());

    let s2 = linlvo::amg::hmis::create_second_strength::<f64>(&s, &cf1)
        .expect("coarse graph exists");
    let cfn = linlvo::amg::hmis::coarsen_hmis::<f64>(&s2, true);
    let c_second = cfn.iter().filter(|&&m| m > 0).count();
    assert!(
        c_second < c_first,
        "aggressive second shot must cut the C set ({c_second} !< {c_first})"
    );

    let mut cf = cf1.clone();
    linlvo::amg::hmis::correct_cf_marker(&mut cf, &cfn);
    assert_eq!(cf.iter().filter(|&&m| m > 0).count(), c_second);

    let hier = AmgHierarchy::<f64>::build(a.clone(), cfg);
    let sizes: Vec<usize> = hier.level_info().iter().map(|l| l.ndof).collect();
    assert!(sizes.len() >= 2, "hierarchy must coarsen: {sizes:?}");
    for w in sizes.windows(2) {
        assert!(w[1] < w[0], "levels must shrink: {sizes:?}");
    }
    assert!(*sizes.last().unwrap() >= 2, "MinCoarseSize 2 respected: {sizes:?}");
}

/// Multipass + extended-i interpolation weights: every assigned F row carries
/// nonzero weights (guards the pass-ordering regression where pass-k rows
/// read unfinished pass-(k−1) rows and came out all-zero), the extended-i
/// truncation keeps at most PMax=4 entries per coarse-level row, and the
/// V-cycle is a sane preconditioner.
#[test]
fn hmis_interpolation_weights_and_cycle_quality() {
    let a = poisson_2d(24);
    let cfg = pi_config();
    let hier = AmgHierarchy::<f64>::build(a.clone(), cfg);

    for (l, level) in hier.levels.iter().enumerate() {
        let Some(p) = level.p.as_ref() else {
            continue; // coarsest level: no prolongation
        };
        for i in 0..p.nrows() {
            let start = p.row_ptr()[i];
            let end = p.row_ptr()[i + 1];
            assert!(end - start > 0, "level {l}: empty P row {i}");
            let wsum: f64 = (start..end).map(|k| p.values()[k]).sum();
            assert!(wsum.is_finite() && wsum != 0.0, "level {l}: P row {i} weight sum {wsum}");
            if l > 0 {
                assert!(
                    end - start <= 4,
                    "extended-i PMax=4 violated: level {l} row {i} has {} entries",
                    end - start
                );
            }
        }
    }

    // V-cycle quality: one cycle from zero must contract the residual.
    let n = a.nrows();
    let b: Vec<f64> = (0..n).map(|i| ((i % 7) as f64 - 3.0).sin() + 1.0).collect();
    let bvec = linlvo::core::vector::DenseVec::from_vec(b);
    let mut y = linlvo::core::vector::DenseVec::zeros(n);
    hier.apply_cycle(&bvec, &mut y, CycleType::V);
    let rate = hier.convergence_rate();
    assert!(
        rate.is_finite() && rate < 1.0,
        "V-cycle must contract the residual, rate = {rate}"
    );
}
