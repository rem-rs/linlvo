//! Elimination tree for sparse LU/Cholesky factorisation.
//!
//! The elimination tree (e-tree) encodes the dependency structure of the
//! column factorisation: column `j` depends on column `parent[j]`.  It is
//! the key data structure for both:
//!
//! - Symbolic factorisation (predict non-zero patterns of L and U).
//! - Multifrontal methods (Sprint 15): the tree drives the order in which
//!   frontal matrices are assembled and eliminated.
//!
//! ## Algorithm
//!
//! For a symmetric positive definite matrix (Cholesky) or the pattern of
//! `AᵀA` (LU), the e-tree is computed by the Liu (1986) algorithm:
//!
//! For each column j (0-indexed):
//!   - For each non-zero (i, j) with i > j (lower triangle):
//!     - Walk the path from i to the root of the current tree.
//!     - Set parent of every node along the path to j.
//!     - Use path-compression to keep the walk O(α(n)) amortised.
//!
//! ## Reference
//!
//! Gilbert, J.R., Ng, E.G., and Peierls, B.W. (1994).
//! *Sparse partial pivoting in time proportional to arithmetic.*
//! SIAM J. Sci. Comput., 15(5), 1075-1091.

#![allow(clippy::needless_range_loop)]
use crate::core::scalar::Scalar;
use crate::sparse::CsrMatrix;

/// Compute the elimination tree of a symmetric (or structurally symmetric)
/// matrix `a` in CSR format.
///
/// Returns `parent[j]` = parent of node `j` in the elimination tree.
/// For the root node(s), `parent[j] = n` (sentinel for "no parent").
///
/// Only the upper-triangular part (row i, col j with j > i) is used.
/// This is equivalent to the e-tree of `Aᵀ A` for unsymmetric matrices,
/// and to the Cholesky e-tree for SPD matrices.
pub fn elimination_tree<T: Scalar>(a: &CsrMatrix<T>) -> Vec<usize> {
    let n = a.nrows();
    let mut parent = vec![n; n];     // n = no parent (root sentinel)
    let mut ancestor = vec![n; n];   // path-compression ancestor

    for i in 0..n {
        for k in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let j = a.col_idx()[k];
            if j >= i { continue; } // only lower triangle (j < i)

            // Walk from j up to the root, path-compressing.
            let mut r = j;
            loop {
                let anc = ancestor[r];
                if anc == n || anc == i {
                    // r is a root or already points to i — set parent.
                    if parent[r] == n {
                        parent[r] = i;
                    }
                    ancestor[r] = i;
                    break;
                }
                ancestor[r] = i; // path compression
                r = anc;
            }
        }
    }

    parent
}

/// Post-order traversal of the elimination tree.
///
/// Returns `post[k]` = the k-th node in post-order (children before parent).
/// Post-ordering is used to schedule frontal matrix eliminations in the
/// multifrontal method (Sprint 15) and for efficient reach-set computation.
pub fn post_order(parent: &[usize]) -> Vec<usize> {
    let n = parent.len();
    // Build children lists.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut roots = Vec::new();
    for j in 0..n {
        if parent[j] < n {
            children[parent[j]].push(j);
        } else {
            roots.push(j);
        }
    }

    let mut post = Vec::with_capacity(n);
    // Iterative DFS post-order.
    let mut stack: Vec<(usize, usize)> = roots.iter().map(|&r| (r, 0)).collect();
    while let Some((node, ci)) = stack.last_mut() {
        let node = *node;
        if *ci < children[node].len() {
            let child = children[node][*ci];
            *ci += 1;
            stack.push((child, 0));
        } else {
            post.push(node);
            stack.pop();
        }
    }
    post
}

/// Compute the column counts of the Cholesky factor L.
///
/// `col_count[j]` = number of non-zeros in column j of L (including diagonal).
/// Used to pre-allocate sparse factor storage before numerical factorisation.
///
/// Uses the mark-and-sweep algorithm of Gilbert, Li, Ng, Peierls (1994):
/// for each row `i`, walk upward from each lower-triangle seed j via parent
/// links, marking visited nodes and incrementing their count, until a
/// previously-marked node is encountered (which means the rest of the path
/// was already counted for this row).
pub fn col_counts<T: Scalar>(a: &CsrMatrix<T>, parent: &[usize]) -> Vec<usize> {
    let n = a.nrows();
    let mut count = vec![1usize; n]; // diagonal always non-zero
    let mut mark = vec![n; n]; // mark[j] = i means j already visited for row i

    for i in 0..n {
        mark[i] = i; // mark the diagonal (so walks stop here)
        for k in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let mut j = a.col_idx()[k];
            if j >= i { continue; } // only strict lower triangle

            // Walk upward in the etree from j toward i, counting unvisited nodes.
            while mark[j] != i {
                mark[j] = i;
                count[j] += 1;
                j = parent[j]; // parent[j] > j in this etree convention
                if j >= n { break; } // reached sentinel root
            }
        }
    }
    count
}

/// Path-halving find: returns the root of `x`'s set, with path compression.
#[allow(dead_code)]
fn find_root(ancestor: &mut [usize], mut x: usize) -> usize {
    loop {
        let p = ancestor[x];
        if p == x { return x; }
        // Path halving: ancestor[x] = ancestor[p]
        let pp = ancestor[p];
        ancestor[x] = pp;
        x = p;
    }
}

#[allow(dead_code)]
fn find_root_u(anc: &mut [usize], mut x: usize) -> usize {
    loop {
        let p = anc[x];
        if p == x { return x; }
        let pp = anc[p];
        anc[x] = pp;
        x = p;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

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
    fn etree_tridiagonal_is_chain() {
        // For a tridiagonal n×n SPD matrix the e-tree is a chain:
        // parent[j] = j+1 for j < n-1, parent[n-1] = n (root).
        let n = 5;
        let a = tridiagonal(n);
        let parent = elimination_tree(&a);
        assert_eq!(parent.len(), n);
        for j in 0..n - 1 {
            assert_eq!(parent[j], j + 1, "parent[{j}] should be {}", j + 1);
        }
        assert_eq!(parent[n - 1], n, "root parent should be sentinel {n}");
    }

    #[test]
    fn post_order_full_coverage() {
        let n = 5;
        let a = tridiagonal(n);
        let parent = elimination_tree(&a);
        let post = post_order(&parent);
        assert_eq!(post.len(), n);
        let mut sorted = post.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn etree_single_node() {
        let mut coo = CooMatrix::new(1, 1);
        coo.push(0, 0, 1.0f64);
        let a = CsrMatrix::from_coo(&coo);
        let parent = elimination_tree(&a);
        assert_eq!(parent, vec![1]);
    }
}
