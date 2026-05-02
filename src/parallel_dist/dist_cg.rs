//! Distributed conjugate-gradient solver.
//!
//! Operates on a single partition's owned unknowns and communicates ghost
//! values through a [`HaloExchange`] backend.  Works for symmetric positive
//! definite systems.
//!
//! # Global dot-product reduction
//! For a correct distributed CG, inner products (`<r, r>`, `<p, Ap>`) must be
//! *global* sums across all ranks.  Pass a [`GlobalReduce`] implementation:
//!
//! | Scenario | Implementation |
//! |----------|---------------|
//! | Single process | [`LocalReduce`] (no-op) |
//! | Multi-rank MPI | [`MpiReduce`](crate::parallel_dist::mpi_halo::MpiReduce) |

use crate::core::scalar::Scalar;
use crate::parallel_dist::dist_csr::DistCsrMatrix;
use crate::parallel_dist::halo::{HaloError, HaloExchange};
pub use crate::parallel_dist::mpi_halo::{GlobalReduce, LocalReduce};

/// Convergence or error result from the distributed CG solver.
#[derive(Debug, Clone)]
pub struct DistCgResult {
    /// Number of iterations performed.
    pub iters: usize,
    /// Final (local) residual norm `‖r‖₂`.
    pub residual_norm: f64,
    /// Whether the solver converged within the requested tolerance.
    pub converged: bool,
}

/// Parameters for [`dist_cg`].
#[derive(Debug, Clone)]
pub struct DistCgParams {
    /// Relative residual tolerance: converge when `‖r‖/‖b‖ ≤ rtol`.
    pub rtol: f64,
    /// Absolute residual tolerance.
    pub atol: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
}

impl Default for DistCgParams {
    fn default() -> Self {
        DistCgParams { rtol: 1e-8, atol: 0.0, max_iter: 1000 }
    }
}

/// Solve `A x = b` on a single partition using distributed CG.
///
/// # Arguments
/// - `a`         — distributed CSR matrix for this rank
/// - `halo`      — halo exchange backend (single-process or MPI)
/// - `reduce`    — global reduction backend ([`LocalReduce`] or
///   [`MpiReduce`](crate::parallel_dist::mpi_halo::MpiReduce))
/// - `b_owned`   — right-hand side for owned unknowns
/// - `x_owned`   — solution vector (in/out, used as initial guess)
/// - `params`    — convergence and iteration limits
///
/// Returns [`HaloError`] if the halo exchange fails during iteration.
pub fn dist_cg<T, E, R>(
    a: &DistCsrMatrix<T>,
    halo: &E,
    reduce: &R,
    b_owned: &[T],
    x_owned: &mut [T],
    params: &DistCgParams,
) -> Result<DistCgResult, HaloError>
where
    T: Scalar + Into<f64>,
    E: HaloExchange<T>,
    R: GlobalReduce,
{
    let n = b_owned.len();
    assert_eq!(x_owned.len(), n, "x and b must have the same length");

    // r = b - A x
    let mut ax = vec![T::zero(); n];
    a.spmv_with_halo(x_owned, halo, &mut ax)?;
    let mut r: Vec<T> = b_owned.iter().zip(ax.iter()).map(|(&b, &ax)| b - ax).collect();
    let mut p = r.clone();

    // Global norm of b for relative-tolerance check.
    let b_norm_sq = reduce.allreduce_sum(dot_f64(b_owned, b_owned));
    let b_norm = b_norm_sq.sqrt();
    let tol = f64::max(params.rtol * b_norm, params.atol);

    // Global <r, r>.
    let mut rtr = reduce.allreduce_sum(dot_f64(&r, &r));

    for iter in 0..params.max_iter {
        let r_norm = rtr.sqrt();
        if r_norm <= tol {
            return Ok(DistCgResult { iters: iter, residual_norm: r_norm, converged: true });
        }

        // Ap = A * p
        let mut ap = vec![T::zero(); n];
        a.spmv_with_halo(&p, halo, &mut ap)?;

        // Global <p, Ap>.
        let pap = reduce.allreduce_sum(dot_f64(&p, &ap));
        if pap == 0.0 { break; }

        let alpha_f = rtr / pap;
        let alpha = T::from_f64(alpha_f);

        // x += alpha * p
        for (xi, &pi) in x_owned.iter_mut().zip(p.iter()) {
            *xi += alpha * pi;
        }
        // r -= alpha * Ap
        for (ri, &api) in r.iter_mut().zip(ap.iter()) {
            *ri -= alpha * api;
        }

        // Global <r_new, r_new>.
        let rtr_new = reduce.allreduce_sum(dot_f64(&r, &r));
        let beta = T::from_f64(rtr_new / rtr);
        rtr = rtr_new;

        // p = r + beta * p
        for (pi, &ri) in p.iter_mut().zip(r.iter()) {
            *pi = ri + beta * *pi;
        }
    }

    Ok(DistCgResult {
        iters: params.max_iter,
        residual_norm: rtr.sqrt(),
        converged: false,
    })
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn dot_f64<T: Scalar + Into<f64>>(a: &[T], b: &[T]) -> f64 {
    a.iter().zip(b.iter()).fold(0.0, |acc, (&x, &y)| acc + x.into() * y.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel_dist::{LocalHaloExchange, DistCsrMatrix};
    use crate::sparse::CsrMatrix;

    fn poisson_1d(n: usize) -> CsrMatrix<f64> {
        use crate::sparse::CooMatrix;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0     { coo.push(i, i-1, -1.0); }
            if i + 1 < n { coo.push(i, i+1, -1.0); }
        }
        CsrMatrix::from_coo(&coo)
    }

    #[test]
    fn dist_cg_single_rank() {
        let n = 16;
        let global = poisson_1d(n);
        let dist = DistCsrMatrix::from_global_csr_block_partition(&global, 1, 0).unwrap();
        let halo = LocalHaloExchange::new(vec![1.0_f64; n]);

        let b = vec![1.0_f64; n];
        let mut x = vec![0.0_f64; n];
        let reduce = LocalReduce;
        let res = dist_cg(&dist, &halo, &reduce, &b, &mut x, &DistCgParams::default()).unwrap();
        assert!(res.converged, "dist_cg did not converge: {:?}", res);
    }
}
