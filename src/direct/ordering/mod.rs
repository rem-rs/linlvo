//! Fill-reducing reordering algorithms for sparse direct solvers.
//!
//! A good permutation P is applied before factorisation so that the reordered
//! matrix P A Pᵀ produces far less fill in L and U.  For structured FEA meshes
//! RCM reduces bandwidth (and thus fill) by 5-50×; COLAMD is better for
//! unstructured matrices; and Nested Dissection (ND) achieves 2-5× fewer
//! non-zeros than COLAMD for large unstructured 3-D FEA problems.

#![allow(clippy::needless_range_loop)]

pub mod rcm;
pub mod colamd;
pub mod nd;

pub use rcm::rcm;
pub use colamd::colamd;
pub use nd::nd;

/// Fill-reducing ordering strategy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum OrderingMethod {
    /// No reordering — original matrix order.
    Natural,
    /// Reverse Cuthill-McKee: minimises bandwidth, fast O(n + nnz).
    #[default]
    Rcm,
    /// Column Approximate Minimum Degree (COLAMD): better for unstructured
    /// matrices; O(nnz) expected.
    Colamd,
    /// Multilevel Nested Dissection (pure-Rust METIS NodeND equivalent).
    ///
    /// For large unstructured FEA meshes this typically achieves 2-5× fewer
    /// non-zeros in L/U compared to COLAMD, at the cost of a longer analysis
    /// phase.  Fully WASM-compatible (zero external dependencies).
    NodeNd,
}

/// Apply a symmetric permutation `perm` to a CSR matrix, returning a new CSR
/// matrix representing `A[perm, perm]`.
///
/// `perm[i] = j` means row/column `j` of the original becomes row/column `i`
/// in the permuted matrix (i.e. `perm` maps *new* index → *old* index).
pub fn permute_symmetric<T: crate::core::scalar::Scalar>(
    a: &crate::sparse::CsrMatrix<T>,
    perm: &[usize],
) -> crate::sparse::CsrMatrix<T> {
    let n = a.nrows();
    assert_eq!(n, perm.len(), "perm length must equal matrix size");
    assert_eq!(n, a.ncols(), "matrix must be square for symmetric permutation");

    // Build inverse permutation: inv_perm[old] = new
    let mut inv_perm = vec![0usize; n];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        inv_perm[old_idx] = new_idx;
    }

    // Assemble permuted COO then convert to CSR.
    let mut coo = crate::sparse::CooMatrix::new(n, n);
    for new_i in 0..n {
        let old_i = perm[new_i];
        for k in a.row_ptr()[old_i]..a.row_ptr()[old_i + 1] {
            let old_j = a.col_idx()[k];
            let new_j = inv_perm[old_j];
            coo.push(new_i, new_j, a.values()[k]);
        }
    }
    crate::sparse::CsrMatrix::from_coo(&coo)
}

/// Given a permutation `perm` (new→old), return the inverse permutation
/// `inv` where `inv[old] = new`.
pub fn invert_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        inv[old_idx] = new_idx;
    }
    inv
}
