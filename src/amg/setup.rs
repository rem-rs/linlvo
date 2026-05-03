//! AMG hierarchy setup phase.
//!
//! Builds a multilevel hierarchy of operators:
//!   Level 0: A₀ = A (finest)
//!   Level k: Aₖ = Pₖ₋₁ᵀ Aₖ₋₁ Pₖ₋₁  (Galerkin coarse-grid operator)
//!
//! Supports two coarsening strategies:
//! - **RS-AMG**: classical Ruge-Stüben with direct interpolation.
//! - **SA-AMG**: smoothed aggregation with smoothed prolongation.

use crate::amg::{
    air::air_restriction_diag,
    coarsen_agg::{build_aggregates, tentative_prolongation},
    coarsen_rs::rs_coarsen,
    interpolation::{rs_interpolation, smooth_prolongation},
    smoother::{SmootherType},
    strength::strong_connections,
};
use crate::core::scalar::Scalar;
use crate::sparse::CsrMatrix;

/// Coarsening strategy selection.
#[derive(Clone, Debug)]
pub enum CoarsenStrategy {
    /// Classical Ruge–Stüben C/F splitting.
    RugeStüben,
    /// Smoothed aggregation.
    SmoothedAggregation,
    /// AIR baseline: RS C/F splitting + diagonal-A_ff ideal-restriction approximation.
    Air,
}

/// Configuration for the AMG setup phase.
#[derive(Clone, Debug)]
pub struct AmgConfig {
    /// Strong-connection threshold θ (default 0.25).
    pub theta: f64,
    /// Coarsening strategy.
    pub strategy: CoarsenStrategy,
    /// Pre/post smoother type.
    pub smoother: SmootherType,
    /// Number of pre-smoothing sweeps.
    pub pre_sweeps: usize,
    /// Number of post-smoothing sweeps.
    pub post_sweeps: usize,
    /// Stop coarsening when coarse size ≤ this value.
    pub coarse_threshold: usize,
    /// Maximum number of levels.
    pub max_levels: usize,
    /// Smoothed-aggregation damping factor (fraction of 4/3·ω).
    pub sa_omega: f64,
}

impl Default for AmgConfig {
    fn default() -> Self {
        AmgConfig {
            theta:            0.25,
            strategy:         CoarsenStrategy::SmoothedAggregation,
            smoother:         SmootherType::WeightedJacobi { omega: 0.667 },
            pre_sweeps:       1,
            post_sweeps:      1,
            coarse_threshold: 10,
            max_levels:       20,
            sa_omega:         0.667,
        }
    }
}

/// One level in the AMG hierarchy.
#[derive(Clone)]
pub struct AmgLevel<T> {
    /// Operator at this level.
    pub a: CsrMatrix<T>,
    /// Prolongation to the next finer level (None for coarsest).
    pub p: Option<CsrMatrix<T>>,
    /// Restriction = Pᵀ (computed on demand).
    pub r: Option<CsrMatrix<T>>,
    /// Cached spectral radius estimate ρ(D⁻¹A) for Chebyshev smoother.
    pub spectral_radius: Option<T>,
}

/// Full AMG hierarchy.
pub struct AmgHierarchy<T> {
    pub levels:  Vec<AmgLevel<T>>,
    pub config:  AmgConfig,
    /// Convergence rate from the most recent `apply_cycle` call:
    /// ‖r_after‖ / ‖r_before‖ at the finest level.
    /// `NaN` if `apply_cycle` has never been called.
    ///
    /// Uses `AtomicU64` (bit-cast) so `AmgHierarchy` remains `Sync`
    /// (required by the `Preconditioner` trait bound via `AmgPrecond`).
    pub last_cycle_rate: std::sync::atomic::AtomicU64,
}

impl<T: Clone> Clone for AmgHierarchy<T> {
    fn clone(&self) -> Self {
        AmgHierarchy {
            levels: self.levels.clone(),
            config: self.config.clone(),
            last_cycle_rate: std::sync::atomic::AtomicU64::new(
                self.last_cycle_rate.load(std::sync::atomic::Ordering::Relaxed)
            ),
        }
    }
}

impl<T: Scalar> AmgHierarchy<T> {
    /// Build the AMG hierarchy for operator `a`.
    pub fn build(a: CsrMatrix<T>, config: AmgConfig) -> Self {
        let mut levels: Vec<AmgLevel<T>> = Vec::new();
        let mut a_curr = Some(a);

        // Check whether we need to precompute spectral radii for Chebyshev.
        let need_spectral = matches!(&config.smoother, SmootherType::Chebyshev { .. });

        for _ in 0..config.max_levels {
            let a_now = a_curr.take().unwrap();
            let n = a_now.nrows();
            if n <= config.coarse_threshold {
                levels.push(AmgLevel { a: a_now, p: None, r: None, spectral_radius: None });
                break;
            }

            // Precompute spectral radius of D^{-1}A if needed (20 power iterations).
            let sr = if need_spectral {
                Some(crate::simd::smoother::estimate_spectral_radius(&a_now, 20))
            } else {
                None
            };

            // Strong connection graph.
            let s = strong_connections(&a_now, config.theta);

            // Build prolongation P and (optionally) custom restriction R.
            let (p, r_custom) = match &config.strategy {
                CoarsenStrategy::RugeStüben => {
                    let status = rs_coarsen::<T>(&s);
                    (rs_interpolation(&a_now, &status), None)
                }
                CoarsenStrategy::SmoothedAggregation => {
                    let agg_id   = build_aggregates::<T>(&s);
                    let n_coarse = agg_id.iter().copied().max().map(|m| m + 1).unwrap_or(1);
                    let p0 = tentative_prolongation::<T>(&agg_id, n_coarse);
                    (smooth_prolongation(&a_now, &p0, config.sa_omega), None)
                }
                CoarsenStrategy::Air => {
                    let status = rs_coarsen::<T>(&s);
                    let p = rs_interpolation(&a_now, &status);
                    let r = air_restriction_diag(&a_now, &status);
                    (p, Some(r))
                }
            };

            let nc = p.ncols();
            if nc == 0 || nc >= n {
                // Coarsening failed to reduce the problem; stop here.
                levels.push(AmgLevel { a: a_now, p: None, r: None, spectral_radius: None });
                break;
            }

            // Restriction: default R=Pᵀ, AIR uses custom R.
            let r = r_custom.unwrap_or_else(|| p.transpose_csr());

            // Galerkin coarse-grid operator: Ac = R * A * P
            let a_coarse = r.matmat(&a_now.matmat(&p));

            levels.push(AmgLevel { a: a_now, p: Some(p), r: Some(r), spectral_radius: sr });
            a_curr = Some(a_coarse);
        }

        // If max_levels exhausted without breaking, push the remaining level.
        if let Some(a_remaining) = a_curr {
            levels.push(AmgLevel { a: a_remaining, p: None, r: None, spectral_radius: None });
        }

        AmgHierarchy { levels, config, last_cycle_rate: std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()) }
    }

    pub fn n_levels(&self) -> usize { self.levels.len() }

    // ─── Level diagnostics ────────────────────────────────────────────────────

    /// Number of degrees of freedom at level `l` (0 = finest).
    ///
    /// Returns `None` if `l >= n_levels()`.
    pub fn level_ndof(&self, l: usize) -> Option<usize> {
        self.levels.get(l).map(|lev| lev.a.nrows())
    }

    /// Number of non-zeros in the operator at level `l`.
    ///
    /// Returns `None` if `l >= n_levels()`.
    pub fn level_nnz(&self, l: usize) -> Option<usize> {
        self.levels.get(l).map(|lev| lev.a.col_idx().len())
    }

    /// Operator complexity: ratio of total non-zeros across all levels to
    /// the finest-level non-zeros.
    ///
    /// A value close to 1 indicates little overhead; typical values are 1.2–2.0
    /// for SA-AMG on structured problems.  Returns `0.0` if the hierarchy is
    /// empty.
    ///
    /// Formula: `sum_{l} nnz(A_l) / nnz(A_0)`.
    pub fn operator_complexity(&self) -> f64 {
        if self.levels.is_empty() { return 0.0; }
        let fine_nnz = self.level_nnz(0).unwrap_or(1) as f64;
        if fine_nnz == 0.0 { return 0.0; }
        let total: f64 = self.levels.iter()
            .map(|lev| lev.a.col_idx().len() as f64)
            .sum();
        total / fine_nnz
    }

    /// Grid complexity: ratio of total DOFs across all levels to the finest
    /// level DOFs.
    ///
    /// Formula: `sum_{l} n_l / n_0`.
    pub fn grid_complexity(&self) -> f64 {
        if self.levels.is_empty() { return 0.0; }
        let fine_n = self.level_ndof(0).unwrap_or(1) as f64;
        if fine_n == 0.0 { return 0.0; }
        let total: f64 = self.levels.iter()
            .map(|lev| lev.a.nrows() as f64)
            .sum();
        total / fine_n
    }

    /// Summary of the AMG hierarchy as a vector of [`LevelInfo`] structs,
    /// one per level (finest to coarsest).
    ///
    /// Use this for logging, debugging, or asserting hierarchy properties in tests.
    pub fn level_info(&self) -> Vec<LevelInfo> {
        self.levels.iter().enumerate().map(|(l, lev)| {
            let ndof = lev.a.nrows();
            let nnz  = lev.a.col_idx().len();
            let avg_nnz_per_row = if ndof == 0 { 0.0 } else { nnz as f64 / ndof as f64 };
            let has_prolongation = lev.p.is_some();
            LevelInfo {
                level: l,
                ndof,
                nnz,
                avg_nnz_per_row,
                is_coarsest: !has_prolongation,
            }
        }).collect()
    }

    /// Print a compact table of level information to stdout.
    ///
    /// Example output:
    /// ```text
    /// AMG hierarchy: 4 levels
    ///   level  n_dof     nnz   avg_nz/row  coarsest?
    ///       0   1000    2998         3.00      no
    ///       1    333    1001         3.00      no
    ///       2    111     333         3.00      no
    ///       3     37     109         2.95     yes
    /// operator complexity: 1.48    grid complexity: 1.48
    /// ```
    pub fn print_info(&self) {
        let infos = self.level_info();
        println!("AMG hierarchy: {} levels", infos.len());
        println!("  {:>6}  {:>6}  {:>7}  {:>12}  {:>9}", "level", "n_dof", "nnz", "avg_nz/row", "coarsest?");
        for info in &infos {
            println!("  {:>6}  {:>6}  {:>7}  {:>12.2}  {:>9}",
                info.level, info.ndof, info.nnz, info.avg_nnz_per_row,
                if info.is_coarsest { "yes" } else { "no" });
        }
        println!("operator complexity: {:.2}    grid complexity: {:.2}",
            self.operator_complexity(), self.grid_complexity());
    }

    /// Coarsening ratios between consecutive levels.
    ///
    /// `coarsen_ratios()[l]` = n_{l} / n_{l+1}  (≥ 1.0, since coarsening reduces).
    /// An empty `Vec` is returned for a single-level hierarchy.
    pub fn coarsen_ratios(&self) -> Vec<f64> {
        self.levels.windows(2).map(|w| {
            let n_fine   = w[0].a.nrows() as f64;
            let n_coarse = w[1].a.nrows() as f64;
            if n_coarse == 0.0 { f64::INFINITY } else { n_fine / n_coarse }
        }).collect()
    }

    /// Per-cycle convergence rate from the most recent [`apply_cycle`] call.
    ///
    /// Returns `‖r_after‖₂ / ‖r_before‖₂` where the residual is computed at
    /// the finest level: `r = b - A x`.  A value close to 0 indicates rapid
    /// convergence; a value close to 1 indicates slow convergence.
    ///
    /// Returns `NaN` if `apply_cycle` has not yet been called.
    pub fn convergence_rate(&self) -> f64 {
        f64::from_bits(self.last_cycle_rate.load(std::sync::atomic::Ordering::Relaxed))
    }
}

// ─── Diagnostic types ─────────────────────────────────────────────────────────

/// Per-level AMG diagnostic data.
///
/// Returned by [`AmgHierarchy::level_info`].
#[derive(Debug, Clone, PartialEq)]
pub struct LevelInfo {
    /// Level index (0 = finest).
    pub level:           usize,
    /// Number of degrees of freedom at this level.
    pub ndof:            usize,
    /// Number of non-zeros in the operator A at this level.
    pub nnz:             usize,
    /// Average non-zeros per row: `nnz / ndof`.
    pub avg_nnz_per_row: f64,
    /// True iff this is the coarsest level (no prolongation stored).
    pub is_coarsest:     bool,
}
