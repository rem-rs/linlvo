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
use crate::core::{operator::LinearOperator, scalar::{Scalar}, vector::{DenseVec, Vector}};

struct LevelScratch<T: Scalar> {
    ax: DenseVec<T>,
    res: DenseVec<T>,
    coarse_rhs: DenseVec<T>,
    coarse_x: DenseVec<T>,
    pe: DenseVec<T>,
}

impl<T: Scalar> LevelScratch<T> {
    fn new(n: usize, nc: usize) -> Self {
        Self {
            ax: DenseVec::zeros(n),
            res: DenseVec::zeros(n),
            coarse_rhs: DenseVec::zeros(nc),
            coarse_x: DenseVec::zeros(nc),
            pe: DenseVec::zeros(n),
        }
    }
}

pub(crate) struct CycleWorkspace<T: Scalar> {
    levels: Vec<LevelScratch<T>>,
}

impl<T: Scalar> CycleWorkspace<T> {
    pub(crate) fn new(hier: &AmgHierarchy<T>) -> Self {
        let levels = hier.levels
            .iter()
            .map(|level| {
                let n = level.a.nrows();
                let nc = level.p.as_ref().map(|p| p.ncols()).unwrap_or(0);
                LevelScratch::new(n, nc)
            })
            .collect();
        Self { levels }
    }
}

fn residual_norm_into<T: Scalar>(
    a: &crate::sparse::CsrMatrix<T>,
    x: &DenseVec<T>,
    b: &DenseVec<T>,
    ax: &mut DenseVec<T>,
) -> f64 {
    a.apply(x, ax);
    let bs = b.as_slice();
    let axs = ax.as_slice();
    let nrm2 = (0..b.len())
        .map(|i| {
            let diff = bs[i] - axs[i];
            diff * diff
        })
        .fold(T::zero(), |acc, value| acc + value)
        .sqrt();
    num_traits::ToPrimitive::to_f64(&nrm2).unwrap_or(f64::INFINITY)
}

/// Number of coarse-level recursions per cycle level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CycleType {
    V,
    W,
    F,
    /// K-cycle: uses `inner_iters` steps of preconditioned CG at the coarse
    /// level with the next-level V-cycle as preconditioner.  Gives better
    /// convergence than W-cycle at similar or lower cost for harder problems.
    K { inner_iters: usize },
}

impl<T: Scalar> AmgHierarchy<T> {
    /// Apply one AMG cycle as a preconditioner:  `x ← M⁻¹ b`  (x starts at 0).
    ///
    /// Records ‖b - A x_after‖ / ‖b - A x_before‖ in `self.last_cycle_rate`
    /// (accessible via [`convergence_rate()`]).
    pub fn apply_cycle(&self, b: &DenseVec<T>, x: &mut DenseVec<T>, cycle: CycleType) {
        let mut workspace = CycleWorkspace::new(self);
        self.apply_cycle_with_workspace(b, x, cycle, &mut workspace);
    }

    pub(crate) fn apply_cycle_with_workspace(
        &self,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
        cycle: CycleType,
        workspace: &mut CycleWorkspace<T>,
    ) {
        let a0 = &self.levels[0].a;

        let r_before = residual_norm_into(a0, x, b, &mut workspace.levels[0].ax);

        vcycle(self, &mut workspace.levels, 0, b, x, cycle);

        let r_after = residual_norm_into(a0, x, b, &mut workspace.levels[0].ax);

        let rate = if r_before < 1e-300 { 0.0 } else { r_after / r_before };
        self.last_cycle_rate.store(
            rate.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

fn vcycle<T: Scalar>(
    hier:  &AmgHierarchy<T>,
    workspace: &mut [LevelScratch<T>],
    level: usize,
    b:     &DenseVec<T>,
    x:     &mut DenseVec<T>,
    cycle: CycleType,
) {
    let lv = &hier.levels[level];
    let (scratch, child_workspace) = workspace.split_first_mut().expect("workspace must cover all AMG levels");

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
    lv.a.apply(x, &mut scratch.ax);
    crate::simd::dense_ops::simd_sub(b.as_slice(), scratch.ax.as_slice(), scratch.res.as_mut_slice());

    // Restrict: r_c = R * res.
    r.apply(&scratch.res, &mut scratch.coarse_rhs);

    // Coarse-grid correction.
    scratch.coarse_x.as_mut_slice().fill(T::zero());
    match cycle {
        CycleType::V => {
            vcycle(hier, child_workspace, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, CycleType::V);
        }
        CycleType::W => {
            vcycle(hier, child_workspace, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, CycleType::W);
            vcycle(hier, child_workspace, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, CycleType::W);
        }
        CycleType::F => {
            // F-cycle: first call with V, then call with F (recursive).
            vcycle(hier, child_workspace, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, CycleType::V);
            vcycle(hier, child_workspace, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, CycleType::F);
        }
        CycleType::K { inner_iters } => {
            // K-cycle: run `inner_iters` steps of flexible CG on the coarse
            // system, using vcycle(level+1, V) as preconditioner.
            // inner_iters=0: fall back to V-cycle (same as CycleType::V).
            if inner_iters == 0 {
                vcycle(hier, child_workspace, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, CycleType::V);
            } else {
                inner_cg_solve(hier, child_workspace, level + 1, &scratch.coarse_rhs, &mut scratch.coarse_x, inner_iters);
            }
        }
    }

    // Prolongate and update: x += P e_c.
    p.apply(&scratch.coarse_x, &mut scratch.pe);
    {
        let xs  = x.as_mut_slice();
        let pes = scratch.pe.as_slice();
        for i in 0..n { xs[i] += pes[i]; }
    }

    // Post-smooth (pass cached spectral radius hint for Chebyshev).
    smooth_with_hint(&lv.a, x, b, &hier.config.smoother, hier.config.post_sweeps, lv.spectral_radius);
}

/// Inner preconditioned CG for the K-cycle coarse correction.
///
/// Solves `A_coarse * e = b_coarse` for exactly `max_iter` iterations
/// using vcycle(level, V) as the preconditioner.  No convergence check
/// is performed — the fixed iteration count is intentional for the K-cycle.
fn inner_cg_solve<T: Scalar>(
    hier:     &AmgHierarchy<T>,
    workspace: &mut [LevelScratch<T>],
    level:    usize,
    b:        &DenseVec<T>,
    x:        &mut DenseVec<T>,
    max_iter: usize,
) {
    let lv = &hier.levels[level];
    let n  = b.len();

    // r = b - A*x
    let mut ax = DenseVec::zeros(n);
    lv.a.apply(x, &mut ax);
    let mut r = DenseVec::zeros(n);
    crate::simd::dense_ops::simd_sub(b.as_slice(), ax.as_slice(), r.as_mut_slice());

    // z = M^{-1} r  (one V-cycle applied to r, not b)
    let mut z = DenseVec::zeros(n);
    vcycle(hier, workspace, level, &r, &mut z, CycleType::V);

    let mut p   = z.clone();
    let mut v   = DenseVec::zeros(n);
    let mut rho = dot_dense(&r, &z);

    for _ in 0..max_iter {
        // v = A * p
        lv.a.apply(&p, &mut v);

        let pv = dot_dense(&p, &v);
        if pv.abs() < T::machine_epsilon() * T::from_f64(1e4) { break; }
        let alpha = rho / pv;

        // x += alpha * p
        {
            let xs = x.as_mut_slice();
            let ps = p.as_slice();
            for i in 0..n { xs[i] += alpha * ps[i]; }
        }
        // r -= alpha * v
        {
            let rs = r.as_mut_slice();
            let vs = v.as_slice();
            for i in 0..n { rs[i] -= alpha * vs[i]; }
        }

        // z = M^{-1} r (V-cycle at this level)
    z.as_mut_slice().fill(T::zero());
        vcycle(hier, workspace, level, &r, &mut z, CycleType::V);

        let rho_new = dot_dense(&r, &z);
        if rho.abs() < T::machine_epsilon() * T::from_f64(1e4) { break; }
        let beta = rho_new / rho;
        rho = rho_new;

        // p = z + beta * p
        {
            let ps = p.as_mut_slice();
            let zs = z.as_slice();
            for i in 0..n { ps[i] = zs[i] + beta * ps[i]; }
        }
    }
}

fn dot_dense<T: Scalar>(a: &DenseVec<T>, b: &DenseVec<T>) -> T {
    a.as_slice().iter().zip(b.as_slice().iter())
        .fold(T::zero(), |s, (&ai, &bi)| s + ai * bi)
}
