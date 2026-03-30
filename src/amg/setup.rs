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
    coarsen_agg::{build_aggregates, tentative_prolongation},
    coarsen_rs::rs_coarsen,
    interpolation::{rs_interpolation, smooth_prolongation},
    smoother::SmootherType,
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
pub struct AmgLevel<T> {
    /// Operator at this level.
    pub a: CsrMatrix<T>,
    /// Prolongation to the next finer level (None for coarsest).
    pub p: Option<CsrMatrix<T>>,
    /// Restriction = Pᵀ (computed on demand).
    pub r: Option<CsrMatrix<T>>,
}

/// Full AMG hierarchy.
pub struct AmgHierarchy<T> {
    pub levels:  Vec<AmgLevel<T>>,
    pub config:  AmgConfig,
}

impl<T: Scalar> AmgHierarchy<T> {
    /// Build the AMG hierarchy for operator `a`.
    pub fn build(a: CsrMatrix<T>, config: AmgConfig) -> Self {
        let mut levels: Vec<AmgLevel<T>> = Vec::new();
        let mut a_curr = Some(a);

        for _ in 0..config.max_levels {
            let a_now = a_curr.take().unwrap();
            let n = a_now.nrows();
            if n <= config.coarse_threshold {
                levels.push(AmgLevel { a: a_now, p: None, r: None });
                break;
            }

            // Strong connection graph.
            let s = strong_connections(&a_now, config.theta);

            // Build prolongation P.
            let p = match &config.strategy {
                CoarsenStrategy::RugeStüben => {
                    let status = rs_coarsen::<T>(&s);
                    rs_interpolation(&a_now, &status)
                }
                CoarsenStrategy::SmoothedAggregation => {
                    let agg_id   = build_aggregates::<T>(&s);
                    let n_coarse = agg_id.iter().copied().max().map(|m| m + 1).unwrap_or(1);
                    let p0 = tentative_prolongation::<T>(&agg_id, n_coarse);
                    smooth_prolongation(&a_now, &p0, config.sa_omega)
                }
            };

            let nc = p.ncols();
            if nc == 0 || nc >= n {
                // Coarsening failed to reduce the problem; stop here.
                levels.push(AmgLevel { a: a_now, p: None, r: None });
                break;
            }

            // R = Pᵀ
            let r = p.transpose_csr();

            // Galerkin coarse-grid operator: Ac = R * A * P
            let a_coarse = r.matmat(&a_now.matmat(&p));

            levels.push(AmgLevel { a: a_now, p: Some(p), r: Some(r) });
            a_curr = Some(a_coarse);
        }

        // If max_levels exhausted without breaking, push the remaining level.
        if let Some(a_remaining) = a_curr {
            levels.push(AmgLevel { a: a_remaining, p: None, r: None });
        }

        AmgHierarchy { levels, config }
    }

    pub fn n_levels(&self) -> usize { self.levels.len() }
}
