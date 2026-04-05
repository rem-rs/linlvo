//! AMG V-cycle, W-cycle, and F-cycle.
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
//! F-cycle calls the coarse level twice: first with V, then with F (recursive).

use crate::amg::{setup::AmgHierarchy, smoother::{smooth_with_hint, SmootherType}};
use crate::core::{operator::LinearOperator, scalar::Scalar, vector::{DenseVec, Vector}};

/// Number of coarse-level recursions per cycle level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CycleType { V, W, F }

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

    // Coarsest level: solve approximately with many sweeps.
    // Use weighted Jacobi on the coarsest level regardless of the configured
    // smoother -- polynomial smoothers like Chebyshev can diverge here because
    // eigenvalue bounds may be very loose on small coarse operators.
    if lv.p.is_none() {
        let coarse_smoother = match &hier.config.smoother {
            SmootherType::Chebyshev { .. } => SmootherType::WeightedJacobi { omega: 0.667 },
            other => other.clone(),
        };
        smooth_with_hint(&lv.a, x, b, &coarse_smoother, 50, None);
        return;
    }

    let p = lv.p.as_ref().unwrap();
    let r = lv.r.as_ref().unwrap();

    // Pre-smooth (pass cached spectral radius hint for Chebyshev).
    smooth_with_hint(&lv.a, x, b, &hier.config.smoother, hier.config.pre_sweeps, lv.spectral_radius);

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
    match cycle {
        CycleType::V => {
            vcycle(hier, level + 1, &res_c, &mut e_c, CycleType::V);
        }
        CycleType::W => {
            vcycle(hier, level + 1, &res_c, &mut e_c, CycleType::W);
            vcycle(hier, level + 1, &res_c, &mut e_c, CycleType::W);
        }
        CycleType::F => {
            // F-cycle: first call with V, then call with F (recursive).
            vcycle(hier, level + 1, &res_c, &mut e_c, CycleType::V);
            vcycle(hier, level + 1, &res_c, &mut e_c, CycleType::F);
        }
    }

    // Prolongate and update: x += P e_c.
    let mut pe = DenseVec::zeros(n);
    p.apply(&e_c, &mut pe);
    {
        let xs  = x.as_mut_slice();
        let pes = pe.as_slice();
        for i in 0..n { xs[i] += pes[i]; }
    }

    // Post-smooth (pass cached spectral radius hint for Chebyshev).
    smooth_with_hint(&lv.a, x, b, &hier.config.smoother, hier.config.post_sweeps, lv.spectral_radius);
}
