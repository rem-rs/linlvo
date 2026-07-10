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

use crate::core::{preconditioner::Preconditioner, scalar::ComplexScalar, vector::DenseVec};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrd};

/// Global counter for unique AmgPrecond instance IDs.
static AMGPRECOND_ID: AtomicU64 = AtomicU64::new(1);

/// AMG preconditioner (wraps the hierarchy and applies one V- or W-cycle).
///
/// Workspace memory is allocated per-thread via `thread_local!` so that
/// concurrent preconditioner applications (e.g., inside a rayon parallel
/// region) never contend on a shared lock.
pub struct AmgPrecond<T: ComplexScalar> {
    pub hier:  AmgHierarchy<T>,
    pub cycle: CycleType,
    /// Unique ID — used to key the per-thread workspace so that multiple
    /// `AmgPrecond<T>` instances of different sizes don't share the same entry.
    id: u64,
}

impl<T: ComplexScalar> AmgPrecond<T> {
    pub fn new(hier: AmgHierarchy<T>) -> Self {
        AmgPrecond {
            hier,
            cycle: CycleType::V,
            id: AMGPRECOND_ID.fetch_add(1, AtomicOrd::Relaxed),
        }
    }

    pub fn with_cycle(mut self, cycle: CycleType) -> Self {
        self.cycle = cycle; self
    }
}

impl<T: ComplexScalar> Preconditioner for AmgPrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        // Start from zero initial guess (preconditioner convention).
        y.as_mut_slice().fill(T::zero());
        // Each thread gets its own workspace — no locking needed.
        thread_local! {
            // Keyed by (TypeId, instance_id) so multiple AmgPrecond<f64>
            // instances of different hierarchy sizes don't share a workspace.
            static WS: RefCell<std::collections::HashMap<
                (std::any::TypeId, u64),
                Box<dyn std::any::Any>,
            >> = RefCell::new(std::collections::HashMap::new());
        }
        WS.with(|ws| {
            let mut map = ws.borrow_mut();
            let key = (std::any::TypeId::of::<T>(), self.id);
            let ws_entry = map
                .entry(key)
                .or_insert_with(|| {
                    Box::new(cycle::CycleWorkspace::<T>::new(&self.hier))
                });
            let workspace = ws_entry
                .downcast_mut::<cycle::CycleWorkspace<T>>()
                .expect("AmgPrecond: workspace type mismatch");
            self.hier.apply_cycle_with_workspace(x, y, self.cycle, workspace);
        });
    }
}
