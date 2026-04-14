//! Algebraic Multigrid (AMG) — setup, cycle, and preconditioner wrapper.

pub mod coarsen_agg;
pub mod coarsen_rs;
pub mod cycle;
pub mod air;
pub mod interpolation;
pub mod setup;
pub mod smoother;
pub mod strength;

pub use setup::{AmgConfig, AmgHierarchy, CoarsenStrategy, AmgLevel, LevelInfo};
pub use smoother::SmootherType;
pub use cycle::CycleType;

use crate::core::{preconditioner::Preconditioner, scalar::Scalar, vector::{DenseVec, Vector}};

/// AMG preconditioner (wraps the hierarchy and applies one V- or W-cycle).
pub struct AmgPrecond<T> {
    pub hier:  AmgHierarchy<T>,
    pub cycle: CycleType,
}

impl<T: Scalar> AmgPrecond<T> {
    pub fn new(hier: AmgHierarchy<T>) -> Self {
        AmgPrecond { hier, cycle: CycleType::V }
    }

    pub fn with_cycle(mut self, cycle: CycleType) -> Self {
        self.cycle = cycle; self
    }
}

impl<T: Scalar> Preconditioner for AmgPrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        // Start from zero initial guess (preconditioner convention).
        let n = x.len();
        let mut tmp = DenseVec::zeros(n);
        self.hier.apply_cycle(x, &mut tmp, self.cycle);
        y.copy_from(&tmp);
    }
}
