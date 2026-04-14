//! MKL-compatible direct solver entrypoint.
//!
//! This module provides a stable `MklSolver` API in linger so upper layers can
//! depend on factor/solve/reuse behavior today. The current implementation uses
//! linger's native multifrontal LU core and keeps the same 3-phase direct-solver
//! lifecycle (`analyze -> factorize -> solve`).
//!
//! When an external MKL backend is introduced, this struct becomes the single
//! integration point without API churn to downstream crates.

use crate::core::{error::SolverError, scalar::Scalar, vector::DenseVec};
use crate::direct::{DirectOptions, DirectSolver, MultifrontalLu, MultifrontalOptions};
use crate::sparse::CsrMatrix;

/// MKL-facing direct solver wrapper.
///
/// Baseline behavior:
/// - non-wasm targets: native multifrontal implementation
/// - wasm targets: unsupported (same as external native backends)
pub struct MklSolver<T: Scalar> {
    inner: MultifrontalLu<T>,
}

impl<T: Scalar> Default for MklSolver<T> {
    fn default() -> Self {
        Self::with_options(DirectOptions::default())
    }
}

impl<T: Scalar> MklSolver<T> {
    /// Create a new MKL solver with direct-solver options.
    pub fn with_options(base: DirectOptions) -> Self {
        let opts = MultifrontalOptions {
            base,
            ..Default::default()
        };
        Self {
            inner: MultifrontalLu::with_options(opts),
        }
    }

    /// Returns whether this build has a native MKL feature flag enabled.
    pub fn has_mkl_feature() -> bool {
        cfg!(feature = "mkl")
    }
}

impl<T: Scalar> DirectSolver<T> for MklSolver<T> {
    fn analyze(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        if cfg!(target_arch = "wasm32") {
            return Err(SolverError::PrecondSetupFailed {
                reason: "MKL direct backend is unavailable on wasm32 targets".to_string(),
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
    fn mkl_solver_solves_single_rhs() {
        if cfg!(target_arch = "wasm32") {
            return;
        }

        let a = poisson_1d_3x3();
        let mut s = MklSolver::<f64>::default();
        s.factor(&a).unwrap();

        let b = DenseVec::from_vec(vec![1.0, 0.0, 1.0]);
        let mut x = DenseVec::zeros(3);
        s.solve(&b, &mut x).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
        assert!((x[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mkl_solver_reuses_factor_for_multi_rhs() {
        if cfg!(target_arch = "wasm32") {
            return;
        }

        let a = poisson_1d_3x3();
        let mut s = MklSolver::<f64>::default();
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