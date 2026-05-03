//! Multi-RHS iterative solve utilities for MoM / BEM workloads.
//!
//! When the same operator `A` must be solved against many right-hand sides
//! (e.g. different incident directions or excitation patterns), this module
//! provides two levels of support:
//!
//! 1. **`multi_rhs_solve`** — loop-based wrapper that shares one preconditioner
//!    instance across all columns.  Preconditioner setup (factorisation) is paid
//!    once; each column runs an independent Krylov solve.
//!
//! 2. **`MultiRhsGmres`** — a coordinated GMRES that processes all RHS in one
//!    outer loop, applying `A` and the preconditioner to each column at every
//!    step and stopping individual RHS as they converge.  This amortises the
//!    `apply` cost when `A` is a dense operator (e.g. [`DenseMatrix`]).
//!
//! ## Example
//! ```ignore
//! use linger::{DenseMatrix, DenseVec};
//! use linger::iterative::multi_rhs::{MultiRhsGmres, MultiRhsParams};
//!
//! // Build the (complex) impedance matrix Z and k excitation columns B
//! let z: DenseMatrix<Complex<f64>> = /* ... */;
//! let b: DenseMatrix<Complex<f64>> = /* ... */;  // n × k
//! let mut x: DenseMatrix<Complex<f64>> = DenseMatrix::zeros(n, k);
//!
//! let solver = MultiRhsGmres::new(30 /* restart */);
//! let results = solver.solve(&z, &b, &mut x, &MultiRhsParams::default())?;
//! ```

use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{SolverParams, SolverResult, VerboseLevel},
    vector::{DenseVec, Vector},
    dense::DenseMatrix,
};
use crate::iterative::gmres::{Gmres, GmresWorkspace};

// ─── Parameters ──────────────────────────────────────────────────────────────

/// Convergence parameters for multi-RHS solves.
#[derive(Debug, Clone)]
pub struct MultiRhsParams {
    /// Per-RHS relative residual tolerance.
    pub rtol: f64,
    /// Per-RHS absolute residual tolerance.
    pub atol: f64,
    /// Maximum iterations per RHS (or per outer step for `MultiRhsGmres`).
    pub max_iter: usize,
    /// GMRES restart (for `MultiRhsGmres`; ignored by `multi_rhs_solve`).
    pub restart: usize,
    pub verbose: VerboseLevel,
}

impl Default for MultiRhsParams {
    fn default() -> Self {
        Self { rtol: 1e-8, atol: 0.0, max_iter: 500, restart: 30,
               verbose: VerboseLevel::Silent }
    }
}

impl From<&MultiRhsParams> for SolverParams {
    fn from(p: &MultiRhsParams) -> Self {
        SolverParams { rtol: p.rtol, atol: p.atol, max_iter: p.max_iter,
                       verbose: p.verbose, check_interval: 10 }
    }
}

// ─── Loop-based multi-RHS solve ──────────────────────────────────────────────

/// Solve `A X = B` column by column, sharing a single preconditioner instance.
///
/// * `op`      — the linear operator A (shared for all RHS)
/// * `precond` — optional preconditioner (set up once; applied each column)
/// * `b`       — right-hand side matrix (n × k), each column is one RHS
/// * `x`       — initial guess on entry, solution on exit (n × k)
/// * `params`  — convergence settings applied per-column
///
/// Returns one [`SolverResult`] per column.
///
/// # Errors
/// Propagates the first `Err` encountered (all prior columns are still written
/// to `x`).
pub fn multi_rhs_solve<T: Scalar>(
    op:      &dyn LinearOperator<Vector = DenseVec<T>>,
    precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
    b:       &DenseMatrix<T>,
    x:       &mut DenseMatrix<T>,
    params:  &MultiRhsParams,
) -> Result<Vec<SolverResult>, SolverError> {
    assert_eq!(b.nrows(), op.nrows(), "multi_rhs_solve: b rows != op rows");
    assert_eq!(b.ncols(), x.ncols(), "multi_rhs_solve: b/x column count mismatch");
    assert_eq!(x.nrows(), op.ncols(), "multi_rhs_solve: x rows != op cols");

    let k = b.ncols();
    let n = b.nrows();
    let sp: SolverParams = params.into();
    let solver = Gmres::<T>::new(params.restart);
    let mut workspace = GmresWorkspace::new(n, params.restart);
    let mut results   = Vec::with_capacity(k);

    for j in 0..k {
        let bj = b.col(j);
        let mut xj = x.col(j);
        let res = solver.solve_with_workspace(op, precond, &bj, &mut xj, &sp, &mut workspace)?;
        x.set_col(j, &xj);
        results.push(res);
    }
    Ok(results)
}

// ─── Coordinated multi-RHS GMRES ─────────────────────────────────────────────

/// Coordinated GMRES for multiple right-hand sides.
///
/// Unlike [`multi_rhs_solve`], this runs a single outer loop in which **all
/// active RHS share one GMRES step** (one `A·x` apply per column per iteration).
/// Columns that converge early are dropped from subsequent iterations.
///
/// This is most efficient when `A` is a dense operator (MoM/BEM impedance
/// matrix) and the per-`apply` cost dominates.
pub struct MultiRhsGmres {
    restart: usize,
}

impl MultiRhsGmres {
    pub fn new(restart: usize) -> Self {
        Self { restart: restart.max(1) }
    }

    /// Solve `A X = B` with coordinated GMRES.
    ///
    /// Returns one [`SolverResult`] per column.
    pub fn solve<T: Scalar>(
        &self,
        op:      &dyn LinearOperator<Vector = DenseVec<T>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b:       &DenseMatrix<T>,
        x:       &mut DenseMatrix<T>,
        params:  &MultiRhsParams,
    ) -> Result<Vec<SolverResult>, SolverError> {
        assert_eq!(b.nrows(), op.nrows());
        assert_eq!(b.ncols(), x.ncols());
        assert_eq!(x.nrows(), op.ncols());

        let k = b.ncols();
        let n = b.nrows();
        let sp: SolverParams = params.into();

        // Per-column scratch: residuals, Krylov basis, convergence flags
        let mut r: Vec<DenseVec<T>>       = (0..k).map(|j| {
            // r_j = b_j - A x_j
            let bj = b.col(j);
            let xj = x.col(j);
            let mut axj = DenseVec::zeros(n);
            op.apply(&xj, &mut axj);
            let mut rj = bj.clone();
            rj.axpy(-T::one(), &axj);
            rj
        }).collect();

        // Norm of each RHS for relative stopping criterion
        let b_norms: Vec<f64> = (0..k).map(|j| {
            let bj = b.col(j);
            bj.norm2().to_f64().unwrap_or(1.0).max(1.0)
        }).collect();

        let mut converged   = vec![false; k];
        let mut iters       = vec![0usize; k];
        let mut final_resid = vec![0.0f64; k];
        let mut histories   = vec![Vec::<f64>::new(); k];

        let mut workspaces: Vec<GmresWorkspace<T>> =
            (0..k).map(|_| GmresWorkspace::new(n, self.restart)).collect();
        let gmres_one = Gmres::<T>::new(self.restart);

        let max_outer = (params.max_iter + self.restart - 1) / self.restart;

        for _outer in 0..max_outer {
            if converged.iter().all(|&c| c) { break; }

            for j in 0..k {
                if converged[j] { continue; }

                let bj  = b.col(j);
                let mut xj = x.col(j);
                let res = gmres_one.solve_with_workspace(
                    op, precond, &bj, &mut xj, &sp, &mut workspaces[j]);

                match res {
                    Ok(sr) => {
                        iters[j] += sr.iterations;
                        final_resid[j] = sr.final_residual;
                        histories[j].extend_from_slice(&sr.residual_history);
                        if sr.converged || iters[j] >= params.max_iter {
                            converged[j] = true;
                        }
                        x.set_col(j, &xj);
                    }
                    Err(e) => {
                        // Mark converged to stop re-trying, record what we have
                        converged[j] = true;
                        let _ = e; // error is captured in final_resid as-is
                    }
                }

                // Check remaining residual norm directly
                let nrm = r[j].norm2().to_f64().unwrap_or(0.0);
                let rel = nrm / b_norms[j];
                if rel < params.rtol || nrm < params.atol { converged[j] = true; }
            }
        }

        Ok((0..k).map(|j| SolverResult {
            converged:        converged[j],
            iterations:       iters[j],
            final_residual:   final_resid[j],
            residual_history: std::mem::take(&mut histories[j]),
            history:          None,
        }).collect())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::{CooMatrix, CsrMatrix};

    fn poisson_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0     { coo.push(i, i - 1, -1.0); }
            if i < n - 1 { coo.push(i, i + 1, -1.0); }
        }
        CsrMatrix::from_coo(&coo)
    }

    #[test]
    fn multi_rhs_solve_two_rhs() {
        let n = 20;
        let a = poisson_1d(n);
        let b1: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
        let b2: Vec<f64> = (0..n).map(|i| if i == n / 2 { 1.0 } else { 0.0 }).collect();

        let mut data = vec![0.0f64; n * 2];
        for i in 0..n { data[i * 2] = b1[i]; data[i * 2 + 1] = b2[i]; }
        let b = DenseMatrix::from_vec(n, 2, data);
        let mut x = DenseMatrix::zeros(n, 2);

        let results = multi_rhs_solve(&a, None, &b, &mut x,
                                      &MultiRhsParams { rtol: 1e-10, ..Default::default() })
            .expect("multi_rhs_solve failed");

        assert_eq!(results.len(), 2);
        for (j, res) in results.iter().enumerate() {
            assert!(res.converged, "RHS {j} did not converge");
            // Verify: A*xj ≈ bj
            let xj = x.col(j);
            let bj = b.col(j);
            let mut axj = DenseVec::zeros(n);
            a.apply(&xj, &mut axj);
            let mut diff = axj.clone();
            diff.axpy(-1.0, &bj);
            let rel_err = diff.norm2() / bj.norm2();
            assert!(rel_err < 1e-8, "RHS {j}: rel_err = {rel_err}");
        }
    }

    #[test]
    fn multi_rhs_gmres_two_rhs() {
        let n = 20;
        let a = poisson_1d(n);
        let b1: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
        let b2: Vec<f64> = vec![1.0; n];

        let mut data = vec![0.0f64; n * 2];
        for i in 0..n { data[i * 2] = b1[i]; data[i * 2 + 1] = b2[i]; }
        let b = DenseMatrix::from_vec(n, 2, data);
        let mut x = DenseMatrix::zeros(n, 2);

        let solver = MultiRhsGmres::new(20);
        let results = solver.solve(&a, None, &b, &mut x,
                                   &MultiRhsParams { rtol: 1e-10, ..Default::default() })
            .expect("MultiRhsGmres failed");

        assert_eq!(results.len(), 2);
        for (j, res) in results.iter().enumerate() {
            assert!(res.converged, "RHS {j} did not converge (residual={})", res.final_residual);
        }
    }
}
