//! Smoothed Aggregation (SA-AMG) coarsening.
//!
//! Builds aggregates by a greedy algorithm:
//! 1. Pick an unaggregated node i; assign it and all its strongly-connected
//!    unaggregated neighbours to a new aggregate.
//! 2. Repeat until all nodes belong to an aggregate.
//!
//! Each aggregate forms one coarse DOF.  The tentative prolongation maps
//! aggregate k → coarse DOF k with unit coefficients.
//!
//! **Reference**: Vaněk, Mandel & Brezina, Computing 56 (1996).

use crate::core::scalar::Scalar;
use crate::sparse::CsrMatrix;

/// Build aggregates from the strong-connection graph `s`.
///
/// Returns `agg_id[i]` = aggregate index for fine node i (0-based).
pub fn build_aggregates<T: Scalar>(s: &CsrMatrix<T>) -> Vec<usize> {
    let n  = s.nrows();
    let rp = s.row_ptr();
    let ci = s.col_idx();

    let mut agg_id  = vec![usize::MAX; n];
    let mut n_agg   = 0usize;

    for seed in 0..n {
        if agg_id[seed] != usize::MAX { continue; }

        // Start new aggregate from seed.
        agg_id[seed] = n_agg;

        // Add strongly-connected unaggregated neighbours.
        for k in rp[seed]..rp[seed + 1] {
            let j = ci[k];
            if agg_id[j] == usize::MAX {
                agg_id[j] = n_agg;
            }
        }
        n_agg += 1;
    }

    agg_id
}

/// Build the **tentative prolongation** P₀ from aggregates.
///
/// P₀[i, k] = 1 if node i belongs to aggregate k, else 0.
/// Returns P₀ as a CSR matrix of size n_fine × n_coarse.
pub fn tentative_prolongation<T: Scalar>(
    agg_id: &[usize],
    n_coarse: usize,
) -> CsrMatrix<T> {
    let n_fine = agg_id.len();
    let mut row_ptr = vec![0usize; n_fine + 1];
    let mut col_idx = Vec::with_capacity(n_fine);
    let mut values  = Vec::with_capacity(n_fine);

    for (i, &k) in agg_id.iter().enumerate() {
        col_idx.push(k);
        values.push(T::one());
        row_ptr[i + 1] = col_idx.len();
    }

    CsrMatrix::from_raw(n_fine, n_coarse, row_ptr, col_idx, values)
}
