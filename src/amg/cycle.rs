//! AMG V-cycle and W-cycle.
//!
//! The cycle is applied recursively:
//!
//! ```text
//! V-cycle(level, x, b):
//!   if coarsest: x = A⁻¹ b  (dense solve or many iterations)
//!   else:
//!     pre-smooth(A, x, b, ν₁)
//!     r = b - A x
//!     e = 0;  V-cycle(level+1, e, R r)   [coarse correction]
//!     x += P e
//!     post-smooth(A, x, b, ν₂)
//! ```
//!
//! W-cycle calls the coarse level **twice** instead of once.

use crate::amg::{setup::AmgHierarchy, smoother::smooth};
use crate::core::{operator::LinearOperator, scalar::Scalar, vector::{DenseVec, Vector}};

/// Number of coarse-level recursions per cycle level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CycleType { V, W }

impl<T: Scalar> AmgHierarchy<T> {
    /// Apply one AMG cycle as a preconditioner:  `x ← M⁻¹ b`  (x starts at 0).
    pub fn apply_cycle(&self, b: &DenseVec<T>, x: &mut DenseVec<T>, cycle: CycleType) {
        vcycle(self, 0, b, x, cycle);
    }
}

fn vcycle<T: Scalar>(
    hier:  &AmgHierarchy<T>,
    level: usize,
    b:     &DenseVec<T>,
    x:     &mut DenseVec<T>,
    cycle: CycleType,
) {
    let lv = &hier.levels[level];

    // Coarsest level: solve approximately with many Jacobi sweeps.
    if lv.p.is_none() {
        smooth(&lv.a, x, b, &hier.config.smoother, 50);
        return;
    }

    let p = lv.p.as_ref().unwrap();
    let r = lv.r.as_ref().unwrap();

    // Pre-smooth.
    smooth(&lv.a, x, b, &hier.config.smoother, hier.config.pre_sweeps);

    // Residual: res = b - A x.
    let n = b.len();
    let mut ax  = DenseVec::zeros(n);
    lv.a.apply(x, &mut ax);
    let mut res = DenseVec::zeros(n);
    {
        let rs  = res.as_mut_slice();
        let bs  = b.as_slice();
        let axs = ax.as_slice();
        for i in 0..n { rs[i] = bs[i] - axs[i]; }
    }

    // Restrict: r_c = R * res.
    let nc = r.nrows();
    let mut res_c = DenseVec::zeros(nc);
    r.apply(&res, &mut res_c);

    // Coarse-grid correction.
    let mut e_c = DenseVec::zeros(nc);
    let n_coarse_calls = if cycle == CycleType::W { 2 } else { 1 };
    for _ in 0..n_coarse_calls {
        vcycle(hier, level + 1, &res_c, &mut e_c, cycle);
    }

    // Prolongate and update: x += P e_c.
    let mut pe = DenseVec::zeros(n);
    p.apply(&e_c, &mut pe);
    {
        let xs  = x.as_mut_slice();
        let pes = pe.as_slice();
        for i in 0..n { xs[i] += pes[i]; }
    }

    // Post-smooth.
    smooth(&lv.a, x, b, &hier.config.smoother, hier.config.post_sweeps);
}
