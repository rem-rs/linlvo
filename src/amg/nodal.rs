//! Nodal (system) AMG for vector problems — hypre `SetNodal(1)` style.
//!
//! For a vector FE problem stored in **byNODES** layout (dof `c*n_s + i`
//! belongs to component `c` of physical node `i`, `n_s = n/dim`), the
//! classical scalar RS coarsening treats every scalar DOF independently and
//! produces non-SPD coarse operators on strongly-coupled block systems
//! (e.g. linear elasticity, cf. MFEM ex12).  Hypre's nodal AMG instead:
//!
//! 1. builds a **nodal strength graph**: node `i` strongly connects to node
//!    `j` when the block row sum `Σ_{c,d} |A[c·n_s+i, d·n_s+j]|` exceeds
//!    `θ · max_k` of the block row (all components move together);
//! 2. runs the standard RS C/F splitting on that nodal graph (every DOF of a
//!    node gets the same status);
//! 3. interpolates **block-diagonally**: component `c` of an F node is
//!    interpolated only from component `c` of its C neighbours, with the
//!    scalar RS weight `-a_ij^cc / (a_ii^cc + Σ_F a_ik^cc)`.
//!
//! The result is a symmetric, SPD-friendly preconditioner for block systems
//! (each level's coarse operator is the Galerkin product `Pᵀ A P`).

#![allow(clippy::needless_range_loop)]

use crate::core::scalar::{ComplexScalar, Scalar};
use crate::sparse::CsrMatrix;
use num_traits::Zero;

use super::coarsen_rs::coarse_index_map;

/// Nodal strength-of-connection matrix.
///
/// Returns an `N×N` boolean CSR (`N = n/dim` physical nodes) where entry
/// `(i, j)` is 1 iff node `i` strongly connects to node `j`:
/// `block_row_sum(i,j) >= theta * max_k block_row_sum(i,k)`.
pub fn nodal_strong_connections<T: ComplexScalar>(
    a: &CsrMatrix<T>,
    dim: usize,
    theta: f64,
) -> CsrMatrix<T> {
    let n = a.nrows();
    let n_s = n / dim;
    debug_assert_eq!(n_s * dim, n, "byNODES layout requires n divisible by dim");

    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vs = a.values();

    let theta_t = <T::Real as Scalar>::from_f64(theta);

    // Block row sums: strength(i -> k) = Σ_{c,d} |A[c*n_s+i, d*n_s+k]|
    let mut rows: Vec<Vec<usize>> = Vec::with_capacity(n_s);
    for i in 0..n_s {
        // Accumulate per-node column sums over the dim×dim block.
        let mut col_sum = vec![T::Real::zero(); n_s];
        for c in 0..dim {
            let row = c * n_s + i;
            for k in rp[row]..rp[row + 1] {
                let j = ci[k];
                let node_j = j % n_s;
                col_sum[node_j] += vs[k].abs();
            }
        }
        let mut max_off = T::Real::zero();
        for (k, &v) in col_sum.iter().enumerate() {
            if k != i && v > max_off {
                max_off = v;
            }
        }
        let cutoff = theta_t * max_off;
        let mut row = Vec::new();
        for (k, &v) in col_sum.iter().enumerate() {
            if k != i && v >= cutoff && cutoff > T::Real::zero() {
                row.push(k);
            }
        }
        rows.push(row);
    }

    let nnz: usize = rows.iter().map(|r| r.len()).sum();
    let mut s_rp = vec![0usize; n_s + 1];
    let mut s_ci = Vec::with_capacity(nnz);
    let mut s_val = Vec::with_capacity(nnz);
    for (i, row) in rows.iter().enumerate() {
        s_rp[i + 1] = s_rp[i] + row.len();
        for &j in row {
            s_ci.push(j);
            s_val.push(T::one());
        }
    }
    CsrMatrix::from_raw(n_s, n_s, s_rp, s_ci, s_val)
}

/// Block-diagonal prolongation for a byNODES vector problem.
///
/// `status` is the **nodal** C/F classification (length `n_s`).  The
/// prolongator is `n × (n_c * dim)` where `n_c` is the number of C nodes:
/// component `c` of F node `i` interpolates from component `c` of its
/// strongly-connected C neighbours with the scalar RS weight
/// `-a_ij^cc / (a_ii^cc + Σ_{F} a_ik^cc)`.
pub fn nodal_rs_interpolation<T: ComplexScalar>(
    a: &CsrMatrix<T>,
    status: &[crate::amg::coarsen_rs::NodeType],
    dim: usize,
) -> CsrMatrix<T> {
    let n = a.nrows();
    let n_s = n / dim;
    let (nc, c_map) = coarse_index_map(status);
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vs = a.values();

    // P row index: c*n_s + i  (component c, node i)
    // P col index: c*nc + c_map[j]  (component c, coarse node j)
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx: Vec<usize> = Vec::new();
    let mut values: Vec<T> = Vec::new();

    for c in 0..dim {
        for i in 0..n_s {
            let row = c * n_s + i;
            match status[i] {
                crate::amg::coarsen_rs::NodeType::Coarse => {
                    col_idx.push(c * nc + c_map[i]);
                    values.push(T::one());
                }
                crate::amg::coarsen_rs::NodeType::Fine | crate::amg::coarsen_rs::NodeType::Undecided => {
                    // Gather C neighbours of the same component and the
                    // fine-neighbour sum for the denominator.
                    let mut a_ii = T::zero();
                    let mut f_sum = T::zero();
                    let mut c_entries: Vec<(usize, T)> = Vec::new();
                    for k in rp[row]..rp[row + 1] {
                        let j = ci[k];
                        if j == row {
                            a_ii = vs[k];
                        } else if j / n_s == c {
                            // Same-component neighbour only (block-diagonal
                            // interpolation): node = j - c*n_s.
                            let node_j = j - c * n_s;
                            if status[node_j] == crate::amg::coarsen_rs::NodeType::Coarse {
                                c_entries.push((c_map[node_j], vs[k]));
                            } else {
                                f_sum += vs[k];
                            }
                        }
                    }
                    let denom = a_ii + f_sum;
                    if denom.abs() < T::machine_epsilon() || c_entries.is_empty() {
                        // Leave the row empty (no interpolation) — coarse
                        // correction skips this DOF.
                        row_ptr[row + 1] = col_idx.len();
                        continue;
                    }
                    for (cj, a_ij) in c_entries {
                        let col = c * nc + cj;
                        col_idx.push(col);
                        values.push(T::zero() - a_ij / denom);
                    }
                }
            }
            row_ptr[row + 1] = col_idx.len();
        }
    }

    CsrMatrix::from_raw(n, nc * dim, row_ptr, col_idx, values)
}
