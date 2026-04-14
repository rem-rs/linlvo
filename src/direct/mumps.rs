//! MUMPS-compatible direct solver entrypoint.
//!
//! This module provides a stable `MumpsSolver` API in linger so upper layers can
//! depend on factor/solve/reuse behavior today. The current implementation uses
//! linger's native multifrontal LU core and keeps the same 3-phase direct-solver
//! lifecycle (`analyze -> factorize -> solve`).
//!
//! The intent is compatibility, not an external MUMPS dependency: downstream
//! crates can keep a MUMPS-shaped entrypoint while executing on linger's own
//! direct-solver implementation.

use crate::core::{error::SolverError, scalar::Scalar, vector::DenseVec};
use crate::direct::{DirectOptions, DirectSolver, MultifrontalLu, MultifrontalOptions};
use crate::sparse::CsrMatrix;

/// MUMPS-facing compatibility wrapper over linger's native multifrontal solver.
///
/// Baseline behavior:
/// - non-wasm targets: native multifrontal implementation
/// - wasm targets: unsupported (same as other native direct backends)
pub struct MumpsSolver<T: Scalar> {
    inner: MultifrontalLu<T>,
}

impl<T: Scalar> Default for MumpsSolver<T> {
    fn default() -> Self {
        Self::with_options(DirectOptions::default())
    }
}

impl<T: Scalar> MumpsSolver<T> {
    /// Create a new MUMPS solver with direct-solver options.
    pub fn with_options(base: DirectOptions) -> Self {
        let opts = MultifrontalOptions {
            base,
            ..Default::default()
        };
        Self {
            inner: MultifrontalLu::with_options(opts),
        }
    }

    /// Returns whether this build advertises the MUMPS-compatibility feature flag.
    pub fn has_mumps_feature() -> bool {
        cfg!(feature = "mumps")
    }
}

impl<T: Scalar> DirectSolver<T> for MumpsSolver<T> {
    fn analyze(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        if cfg!(target_arch = "wasm32") {
            return Err(SolverError::PrecondSetupFailed {
                reason: "MUMPS-compatible native direct path is unavailable on wasm32 targets".to_string(),
            });
        }
        self.inner.analyze(a)
    }

    fn factorize(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        self.inner.factorize(a)
    }

    fn solve(&self, b: &DenseVec<T>, x: &mut DenseVec<T>) -> Result<(), SolverError> {
        self.inner.solve(b, x)
    }

    fn solve_multi(&self, bs: &[DenseVec<T>], xs: &mut [DenseVec<T>]) -> Result<(), SolverError> {
        self.inner.solve_multi(bs, xs)
    }

    fn reset_factors(&mut self) {
        self.inner.reset_factors();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::DirectSolver;
    use crate::sparse::CooMatrix;

    fn poisson_1d_3x3() -> CsrMatrix<f64> {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.push(0, 0, 2.0);
        coo.push(0, 1, -1.0);
        coo.push(1, 0, -1.0);
        coo.push(1, 1, 2.0);
        coo.push(1, 2, -1.0);
        coo.push(2, 1, -1.0);
        coo.push(2, 2, 2.0);
        CsrMatrix::from_coo(&coo)
    }

    #[test]
    fn mumps_solver_solves_single_rhs() {
        if cfg!(target_arch = "wasm32") {
            return;
        }

        let a = poisson_1d_3x3();
        let mut s = MumpsSolver::<f64>::default();
        s.factor(&a).unwrap();

        let b = DenseVec::from_vec(vec![1.0, 0.0, 1.0]);
        let mut x = DenseVec::zeros(3);
        s.solve(&b, &mut x).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
        assert!((x[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mumps_solver_reuses_factor_for_multi_rhs() {
        if cfg!(target_arch = "wasm32") {
            return;
        }

        let a = poisson_1d_3x3();
        let mut s = MumpsSolver::<f64>::default();
        s.factor(&a).unwrap();

        let bs = vec![
            DenseVec::from_vec(vec![1.0, 0.0, 1.0]),
            DenseVec::from_vec(vec![0.0, 1.0, 0.0]),
        ];
        let mut xs = vec![DenseVec::zeros(3), DenseVec::zeros(3)];
        s.solve_multi(&bs, &mut xs).unwrap();

        assert!((xs[0][0] - 1.0).abs() < 1e-10);
        assert!((xs[0][1] - 1.0).abs() < 1e-10);
        assert!((xs[0][2] - 1.0).abs() < 1e-10);

        assert!((xs[1][0] - 0.5).abs() < 1e-10);
        assert!((xs[1][1] - 1.0).abs() < 1e-10);
        assert!((xs[1][2] - 0.5).abs() < 1e-10);
    }
}
