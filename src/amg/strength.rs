//! Strong-connection graph for AMG.
//!
//! Entry (i, j) is a **strong connection** from i to j if
//!
//! ```text
//! |a_{ij}| >= theta * max_{k != i} |a_{ik}|
//! ```
//!
//! where `theta` (typically 0.25) controls how aggressively connections are
//! pruned.  The result is stored as a CSR matrix of booleans (unit values).
//!
//! Row computations are independent and are parallelised with Rayon when the
//! `rayon` feature is enabled.
//!
//! **Reference**: Ruge & Stüben, §3.1; Saad §12.5.

use crate::core::scalar::{ComplexScalar, Scalar};
use crate::sparse::CsrMatrix;
use num_traits::Zero;

/// Compute the strong-connection matrix S for `a` with threshold `theta`.
///
/// Returns a CSR matrix whose nonzero pattern marks strong off-diagonal
/// connections.  Diagonal entries are excluded.
///
/// When compiled with `feature = "rayon"`, row computations run in parallel.
pub fn strong_connections<T: ComplexScalar>(a: &CsrMatrix<T>, theta: f64) -> CsrMatrix<T> {
    let n       = a.nrows();
    let theta_t = <T::Real as Scalar>::from_f64(theta);

    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vs = a.values();

    // Each row is independent: compute the list of strong neighbours per row.
    let compute_row = |i: usize| -> Vec<(usize, T)> {
        let mut max_off = T::Real::zero();
        for k in rp[i]..rp[i + 1] {
            if ci[k] != i {
                let v = vs[k].abs();
                if v > max_off { max_off = v; }
            }
        }
        let cutoff = theta_t * max_off;

        let mut row = Vec::new();
        for k in rp[i]..rp[i + 1] {
            let j = ci[k];
            if j != i && vs[k].abs() >= cutoff && cutoff > T::Real::zero() {
                row.push((j, T::one()));
            }
        }
        row
    };

    #[cfg(feature = "rayon")]
    let rows: Vec<Vec<(usize, T)>> = {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(compute_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let rows: Vec<Vec<(usize, T)>> = (0..n).map(compute_row).collect();

    // Assemble CSR sequentially from per-row results.
    let nnz: usize = rows.iter().map(|r| r.len()).sum();
    let mut s_rp  = vec![0usize; n + 1];
    let mut s_ci  = Vec::with_capacity(nnz);
    let mut s_val = Vec::with_capacity(nnz);
    for (i, row) in rows.iter().enumerate() {
        s_rp[i + 1] = s_rp[i] + row.len();
        for &(j, v) in row {
            s_ci.push(j);
            s_val.push(v);
        }
    }

    CsrMatrix::from_raw(n, n, s_rp, s_ci, s_val)
}
