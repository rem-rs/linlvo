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

use crate::core::{preconditioner::Preconditioner, scalar::Scalar, vector::DenseVec};
use std::sync::Mutex;

/// AMG preconditioner (wraps the hierarchy and applies one V- or W-cycle).
pub struct AmgPrecond<T: Scalar> {
    pub hier:  AmgHierarchy<T>,
    pub cycle: CycleType,
    scratch: Mutex<cycle::CycleWorkspace<T>>,
}

impl<T: Scalar> AmgPrecond<T> {
    pub fn new(hier: AmgHierarchy<T>) -> Self {
        let scratch = Mutex::new(cycle::CycleWorkspace::new(&hier));
        AmgPrecond { hier, cycle: CycleType::V, scratch }
    }

    pub fn with_cycle(mut self, cycle: CycleType) -> Self {
        self.cycle = cycle; self
    }
}

impl<T: Scalar> Preconditioner for AmgPrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        // Start from zero initial guess (preconditioner convention).
        y.as_mut_slice().fill(T::zero());
        let mut scratch = self.scratch.lock().unwrap_or_else(|poison| poison.into_inner());
        self.hier.apply_cycle_with_workspace(x, y, self.cycle, &mut scratch);
    }
}
