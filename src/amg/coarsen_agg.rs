//! Smoothed Aggregation (SA-AMG) coarsening.
//!
//! Builds aggregates by a greedy algorithm:
//! 1. Pick an unaggregated node i; assign it and all its strongly-connected
//!    unaggregated neighbours to a new aggregate.
//! 2. Repeat until all nodes belong to an aggregate.
//!
//! **Singleton absorption (round-107 D1000-LOR)**: a node whose strong-connection
//! row is empty forms a singleton aggregate.  Its tentative-prolongation column
//! is the unit vector `e_i`, and after one Jacobi smoothing step
//! `(I − ω D⁻¹ A) e_i ≈ (1−ω) e_i` collapses towards zero for diagonally
//! dominant rows — the column (and with it the Galerkin coarse operator
//! `A_c = Pᵀ A P`) becomes numerically rank-deficient, the coarsest LU
//! amplifies the near-nullspace, and the whole V-cycle preconditioner turns
//! indefinite (measured on the LOR-elasticity x-block at n = 16: a 1.1e-2
//! column norm produced λmin(A_c) = 2.5e-6 two levels down and
//! λmin(B_cycle) = −6.4e+2).  Following the hypre aggregation semantics
//! ("any remaining unaggregated point is added to the aggregate of its
//! strongest connected neighbour"), every singleton whose node has *any*
//! off-diagonal raw connection is absorbed into that neighbour's aggregate;
//! truly isolated rows (e.g. DIAG_ONE Dirichlet rows with zeroed off-diagonal
//! structure, or a diagonal-only matrix) keep their own aggregate.  Aggregate
//! ids are compacted afterwards, so `P₀` never has empty columns.
//!
//! Each aggregate forms one coarse DOF.  The tentative prolongation maps
//! aggregate k → coarse DOF k with unit coefficients.
//!
//! **Reference**: Vaněk, Mandel & Brezina, Computing 56 (1996).

#![allow(clippy::needless_range_loop)]
use crate::core::scalar::ComplexScalar;
use crate::sparse::CsrMatrix;
use num_traits::Zero;

/// Build aggregates from the matrix `a` and its strong-connection graph `s`.
///
/// Returns `agg_id[i]` = aggregate index for fine node i (0-based, compact).
pub fn build_aggregates<T: ComplexScalar>(a: &CsrMatrix<T>, s: &CsrMatrix<T>) -> Vec<usize> {
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

    // ── Singleton absorption (D1000-LOR, hypre aggregation semantics) ──────
    // A singleton aggregate whose node has an off-diagonal raw connection is
    // absorbed into the strongest-connected neighbour's aggregate.  Iterate
    // until no more absorption happens (two singleton nodes that are each
    // other's strongest neighbour merge into one 2-node aggregate this way).
    let a_rp = a.row_ptr();
    let a_ci = a.col_idx();
    let a_vs = a.values();
    loop {
        let mut sizes = vec![0usize; n_agg];
        for &id in agg_id.iter() { sizes[id] += 1; }
        let mut changed = false;
        for i in 0..n {
            if sizes[agg_id[i]] != 1 { continue; }
            // Strongest raw off-diagonal connection of node i.
            let mut best_j = usize::MAX;
            let mut best_w = <T::Real as Zero>::zero();
            for k in a_rp[i]..a_rp[i + 1] {
                let j = a_ci[k];
                if j != i {
                    let w = a_vs[k].abs();
                    if w > best_w {
                        best_w = w;
                        best_j = j;
                    }
                }
            }
            if best_j != usize::MAX && agg_id[best_j] != agg_id[i] {
                sizes[agg_id[i]] -= 1;
                agg_id[i] = agg_id[best_j];
                sizes[agg_id[i]] += 1;
                changed = true;
            }
        }
        if !changed { break; }
    }

    // Compact aggregate ids: absorption may have emptied some ids, and P₀
    // must not have empty columns (they would make A_c rank-deficient).
    let mut remap = vec![usize::MAX; n_agg];
    let mut next = 0usize;
    for id in agg_id.iter_mut() {
        if remap[*id] == usize::MAX {
            remap[*id] = next;
            next += 1;
        }
        *id = remap[*id];
    }

    agg_id
}

/// Build the **tentative prolongation** P₀ from aggregates.
///
/// P₀[i, k] = 1 if node i belongs to aggregate k, else 0.
/// Returns P₀ as a CSR matrix of size n_fine × n_coarse.
pub fn tentative_prolongation<T: ComplexScalar>(
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
