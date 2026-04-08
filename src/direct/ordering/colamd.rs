//! Column Approximate Minimum Degree (COLAMD) reordering.
//!
//! COLAMD computes a column ordering that approximately minimises the fill-in
//! produced by sparse Gaussian elimination.  It is more effective than RCM for
//! unstructured matrices but slower for regular-grid problems.
//!
//! ## Algorithm
//!
//! This implementation follows the simplified AMD (Approximate Minimum Degree)
//! ordering strategy applied symmetrically to `A + Aᵀ`, which is appropriate
//! for symmetric or nearly-symmetric FEA matrices.
//!
//! The core idea is a greedy elimination order:
//!   1. Maintain a *degree* estimate for each node (# of fill entries the
//!      elimination of this node would cause).
//!   2. Always eliminate the node with the smallest degree (minimum-degree
//!      heuristic).
//!   3. After elimination, update the degrees of neighbouring nodes.
//!
//! For FEA matrices this approximation typically reduces fill by 3-10× over
//! natural order for 3-D unstructured meshes.
//!
//! ## Reference
//!
//! Amestoy, P. R., Davis, T. A., Duff, I. S. (1996).  *An approximate minimum
//! degree ordering algorithm.*  SIAM J. Matrix Anal. Appl. 17(4):886-905.

#![allow(clippy::needless_range_loop)]
use crate::core::scalar::Scalar;
use crate::sparse::CsrMatrix;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// Compute an approximate minimum degree permutation for `a`.
///
/// Returns `perm` where `perm[i]` = original index placed at position `i`.
///
/// Works on the symmetric graph of `A + Aᵀ` so it is valid for both symmetric
/// and non-symmetric matrices.
///
/// # Example
/// ```
/// use linger::direct::ordering::colamd;
/// use linger::sparse::{CooMatrix, CsrMatrix};
///
/// let mut coo: CooMatrix<f64> = CooMatrix::new(4, 4);
/// for i in 0..4usize { coo.push(i, i, 2.0); }
/// coo.push(0, 1, -1.0); coo.push(1, 0, -1.0);
/// coo.push(1, 2, -1.0); coo.push(2, 1, -1.0);
/// coo.push(2, 3, -1.0); coo.push(3, 2, -1.0);
/// let csr = CsrMatrix::from_coo(&coo);
/// let perm = colamd(&csr);
/// assert_eq!(perm.len(), 4);
/// ```
pub fn colamd<T: Scalar>(a: &CsrMatrix<T>) -> Vec<usize> {
    let n = a.nrows();
    if n == 0 { return vec![]; }

    // Build symmetric adjacency (union of A and Aᵀ patterns).
    let adj = build_symmetric_adj(a);

    amd_order(n, &adj)
}

// ─── AMD greedy ordering ─────────────────────────────────────────────────────

/// Greedy Approximate Minimum Degree ordering.
///
/// Uses a min-heap keyed on the *external degree* of each node.
/// After elimination of node `v`, the degrees of `v`'s uneliminated
/// neighbours are updated using the *absorbed* degree model:
///   d(w) ← |adj(w) ∪ adj(v)| - |eliminated| - 1
///
/// Exposed as `pub(crate)` so the nested-dissection base case can reuse it
/// without duplicating the implementation.
pub(crate) fn amd_order(n: usize, adj: &[Vec<usize>]) -> Vec<usize> {
    // Initial degree = number of uneliminated neighbours.
    let mut degree: Vec<usize> = adj.iter().map(|nbrs| nbrs.len()).collect();
    let mut eliminated = vec![false; n];
    let mut perm = Vec::with_capacity(n);

    // Min-heap of (degree, node).  We use lazy deletion: entries in the heap
    // may be stale; we re-check after popping.
    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    for i in 0..n {
        heap.push(Reverse((degree[i], i)));
    }

    // Maintain adjacency sets as sorted Vecs for easy union/intersection.
    let mut adj_live: Vec<Vec<usize>> = adj.to_vec();

    while perm.len() < n {
        // Pop the minimum-degree non-eliminated node.
        let node = loop {
            let Reverse((d, v)) = match heap.pop() {
                Some(x) => x,
                None => break 0,  // heap empty — shouldn't happen
            };
            if !eliminated[v] && degree[v] == d {
                break v;
            }
            // Stale entry: re-insert with updated degree if not eliminated.
            if !eliminated[v] {
                heap.push(Reverse((degree[v], v)));
            }
        };

        eliminated[node] = true;
        perm.push(node);

        // Collect the uneliminated neighbours of `node`.
        let nbrs: Vec<usize> = adj_live[node]
            .iter()
            .copied()
            .filter(|&w| !eliminated[w])
            .collect();

        // Update degrees: for each neighbour w, the new adj(w) is the union
        // of adj(w) and adj(node) minus `node` itself minus eliminated nodes.
        for &w in &nbrs {
            // New adjacency for w = union(adj(w), nbrs) \ {w, node}.
            let mut new_adj: Vec<usize> = adj_live[w]
                .iter()
                .chain(nbrs.iter())
                .copied()
                .filter(|&v| v != w && v != node && !eliminated[v])
                .collect();
            new_adj.sort_unstable();
            new_adj.dedup();
            degree[w] = new_adj.len();
            adj_live[w] = new_adj;
            heap.push(Reverse((degree[w], w)));
        }
    }

    // Any nodes not yet in perm (disconnected) append in natural order.
    for i in 0..n {
        if !eliminated[i] { perm.push(i); }
    }

    perm
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn build_symmetric_adj<T: Scalar>(a: &CsrMatrix<T>) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        for k in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let j = a.col_idx()[k];
            if j == i { continue; }
            adj[i].push(j);
            adj[j].push(i);
        }
    }

    for nbrs in adj.iter_mut() {
        nbrs.sort_unstable();
        nbrs.dedup();
    }

    adj
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::{CooMatrix, CsrMatrix};

    fn tridiagonal(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0     { coo.push(i, i - 1, -1.0); }
            if i + 1 < n { coo.push(i, i + 1, -1.0); }
        }
        CsrMatrix::from_coo(&coo)
    }

    #[test]
    fn colamd_is_permutation() {
        let n = 8;
        let a = tridiagonal(n);
        let perm = colamd(&a);
        assert_eq!(perm.len(), n);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn colamd_empty() {
        let a: CsrMatrix<f64> = CsrMatrix::from_coo(&CooMatrix::new(0, 0));
        assert_eq!(colamd(&a), vec![]);
    }
}
