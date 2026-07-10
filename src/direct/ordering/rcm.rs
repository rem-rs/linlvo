//! Reverse Cuthill-McKee (RCM) reordering.
//!
//! RCM minimises the *bandwidth* of a sparse matrix, which directly bounds the
//! number of non-zeros in the LU factors.  For structured FEA meshes (e.g.
//! 1-D / 2-D regular grids) the bandwidth reduction is 5-50×.
//!
//! ## Algorithm
//!
//! 1. Build the symmetric adjacency graph of `A + Aᵀ`.
//! 2. Choose a starting node with small degree (peripheral heuristic via
//!    pseudo-peripheral node search).
//! 3. BFS-level traversal; within each level, sort neighbours by ascending
//!    degree before enqueuing.
//! 4. Reverse the resulting ordering to get RCM.
//!
//! ## Reference
//!
//! Cuthill, E. and McKee, J. (1969).  *Reducing the bandwidth of sparse
//! symmetric matrices.*  Proceedings of the ACM National Conference.

#![allow(clippy::needless_range_loop)]
use crate::core::scalar::Scalar;
use crate::sparse::CsrMatrix;
use std::collections::VecDeque;

/// Compute the Reverse Cuthill-McKee permutation for matrix `a`.
///
/// Returns a permutation vector `perm` of length `n` where `perm[i]` is the
/// original row/column index that should become row/column `i` in the reordered
/// matrix.  In other words, the reordered matrix is `A[perm, perm]`.
///
/// The permutation is computed from the symmetric graph of `A + Aᵀ`, so it
/// works for both symmetric and non-symmetric matrices.
///
/// # Example
/// ```
/// use linger::direct::ordering::rcm;
/// use linger::sparse::{CooMatrix, CsrMatrix};
///
/// let mut coo: CooMatrix<f64> = CooMatrix::new(4, 4);
/// for i in 0..4usize { coo.push(i, i, 2.0); }
/// coo.push(0, 1, -1.0); coo.push(1, 0, -1.0);
/// coo.push(1, 2, -1.0); coo.push(2, 1, -1.0);
/// coo.push(2, 3, -1.0); coo.push(3, 2, -1.0);
/// let csr = CsrMatrix::from_coo(&coo);
/// let perm = rcm(&csr);
/// assert_eq!(perm.len(), 4);
/// ```
pub fn rcm<T: Scalar>(a: &CsrMatrix<T>) -> Vec<usize> {
    let n = a.nrows();
    if n == 0 { return vec![]; }

    // Build symmetric adjacency lists (union of A and Aᵀ adjacency).
    let adj = build_symmetric_adj(a);

    // Degree of each node in the symmetric graph.
    let degree: Vec<usize> = adj.iter().map(|nbrs| nbrs.len()).collect();

    let mut visited = vec![false; n];
    let mut result: Vec<usize> = Vec::with_capacity(n);

    // Iterate over connected components (handles disconnected graphs).
    for seed in 0..n {
        if visited[seed] { continue; }
        // For the starting node of each component, pick the minimum-degree
        // unvisited node reachable from `seed` (pseudo-peripheral heuristic).
        let start = find_start_in_component(seed, &adj, &degree, &visited);
        let component_order = cuthill_mckee_from(start, &adj, &degree, &mut visited);
        // Reverse within each component for RCM.
        result.extend(component_order.into_iter().rev());
    }

    result
}

// ─── Internals ───────────────────────────────────────────────────────────────

/// Run the (unreversed) Cuthill-McKee BFS from `start`, marking nodes visited.
fn cuthill_mckee_from(
    start: usize,
    adj: &[Vec<usize>],
    degree: &[usize],
    visited: &mut [bool],
) -> Vec<usize> {
    let mut order = Vec::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    visited[start] = true;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        order.push(node);
        // Collect unvisited neighbours, sort by degree (ascending).
        let mut nbrs: Vec<usize> = adj[node]
            .iter()
            .copied()
            .filter(|&nb| !visited[nb])
            .collect();
        nbrs.sort_unstable_by_key(|&nb| degree[nb]);
        for nb in nbrs {
            visited[nb] = true;
            queue.push_back(nb);
        }
    }
    order
}

/// Find a good starting node in the connected component containing `seed`.
///
/// Performs one BFS from `seed` and returns the minimum-degree node in the
/// last level (a simple pseudo-peripheral heuristic).
fn find_start_in_component(
    seed: usize,
    adj: &[Vec<usize>],
    degree: &[usize],
    visited: &[bool],
) -> usize {
    let n = adj.len();
    let mut tmp_visited = vec![false; n];
    // Copy the real visited state so we don't traverse already-placed nodes.
    tmp_visited.copy_from_slice(visited);

    let mut last_level: Vec<usize> = vec![seed];
    let mut queue: VecDeque<usize> = VecDeque::new();
    tmp_visited[seed] = true;
    queue.push_back(seed);

    while !queue.is_empty() {
        let level_size = queue.len();
        last_level.clear();
        for _ in 0..level_size {
            let node = queue.pop_front().unwrap();
            last_level.push(node);
            for &nb in &adj[node] {
                if !tmp_visited[nb] {
                    tmp_visited[nb] = true;
                    queue.push_back(nb);
                }
            }
        }
    }

    *last_level.iter().min_by_key(|&&v| degree[v]).unwrap()
}

/// Build symmetric adjacency lists from `A` (union of row and column patterns).
fn build_symmetric_adj<T: Scalar>(a: &CsrMatrix<T>) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        for k in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let j = a.col_idx()[k];
            if j == i { continue; } // skip diagonal
            // Add both directions for symmetry.
            adj[i].push(j);
            adj[j].push(i);
        }
    }

    // Deduplicate each adjacency list.
    for nbrs in adj.iter_mut() {
        nbrs.sort_unstable();
        nbrs.dedup();
    }

    adj
}

// ─── Proper pseudo-peripheral search ────────────────────────────────────────

/// Find a pseudo-peripheral node via two BFS passes (George & Liu 1981).
///
/// Starting from `seed`, do a BFS; take the last-level node with the
/// minimum degree as a new start; repeat until no improvement in the
/// last-level width.
#[allow(dead_code)]
pub(crate) fn pseudo_peripheral(seed: usize, adj: &[Vec<usize>]) -> usize {
    let n = adj.len();
    let mut current = seed;
    let degree: Vec<usize> = adj.iter().map(|v| v.len()).collect();

    loop {
        // BFS from current.
        let mut visited = vec![false; n];
        let mut levels: Vec<Vec<usize>> = Vec::new();
        let mut queue: VecDeque<usize> = VecDeque::new();
        visited[current] = true;
        queue.push_back(current);

        while !queue.is_empty() {
            let mut level_nodes = Vec::new();
            for _ in 0..queue.len() {
                let node = queue.pop_front().unwrap();
                level_nodes.push(node);
                for &nb in &adj[node] {
                    if !visited[nb] {
                        visited[nb] = true;
                        queue.push_back(nb);
                    }
                }
            }
            levels.push(level_nodes);
        }

        let last_level = levels.last().unwrap();
        let next = *last_level.iter().min_by_key(|&&v| degree[v]).unwrap();

        if next == current { break; }
        current = next;
    }
    current
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
    fn rcm_is_permutation() {
        let n = 8;
        let a = tridiagonal(n);
        let perm = rcm(&a);
        assert_eq!(perm.len(), n);
        // Check it is a valid permutation (each index appears exactly once).
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn rcm_empty() {
        let a: CsrMatrix<f64> = CsrMatrix::from_coo(&CooMatrix::new(0, 0));
        assert_eq!(rcm(&a), Vec::<usize>::new());
    }

    #[test]
    fn rcm_single() {
        let mut coo = CooMatrix::new(1, 1);
        coo.push(0, 0, 1.0);
        let a = CsrMatrix::from_coo(&coo);
        assert_eq!(rcm(&a), vec![0]);
    }
}
