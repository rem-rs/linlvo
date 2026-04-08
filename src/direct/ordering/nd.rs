//! Multilevel Nested Dissection (ND) ordering — pure-Rust implementation.
//!
//! This is an independent pure-Rust implementation of the multilevel nested
//! dissection algorithm used in METIS NodeND.  It achieves the same goal
//! (fill-reducing vertex ordering for sparse direct solvers) with zero external
//! dependencies, making it fully compatible with `wasm32-unknown-unknown`.
//!
//! ## Algorithm overview
//!
//! ```text
//! nd_order(G):
//!   if |G| ≤ BASE_SIZE:
//!       return amd_small(G)                 // exact AMD for small subproblems
//!
//!   G_c = hem_coarsen(G, target≈60)         // Heavy-Edge Matching coarsening
//!   color[] = bfs_bipartition(G_c)          // BFS two-coloring from two poles
//!   (A, B, Sep) = extract_separator(G_c, color)
//!   fm_refine(G_c, A, B, Sep)               // Fiduccia-Mattheyses refinement
//!   project back to fine graph
//!
//!   nd_order(G[A]) ++ nd_order(G[B]) ++ Sep  // recursive
//! ```
//!
//! ## Fill reduction
//!
//! For 2-D FEA meshes (Laplacian-type), ND typically achieves 2-4× fewer
//! non-zeros in L/U compared to natural order, and is competitive with COLAMD
//! on irregular meshes.
//!
//! ## Reference
//!
//! Karypis, G. and Kumar, V. (1998).  *A fast and high quality multilevel
//! scheme for partitioning irregular graphs.*  SIAM J. Sci. Comput., 20(1),
//! 359-392.
//!
//! George, A. and Liu, J. W. H. (1981).  *Computer Solution of Large Sparse
//! Positive Definite Systems.*  Prentice-Hall.

use crate::core::scalar::Scalar;
use crate::sparse::CsrMatrix;
use std::collections::VecDeque;

// ─── Tuning constants ────────────────────────────────────────────────────────

/// Below this size, use AMD directly (no recursion).
const BASE_SIZE: usize = 120;
/// Target coarse graph size.
const COARSE_TARGET: usize = 60;
/// Maximum recursion depth.
const MAX_DEPTH: usize = 64;
/// Maximum Fiduccia-Mattheyses passes.
const MAX_FM_PASSES: usize = 5;
/// Imbalance tolerance: |A| and |B| may differ by at most this fraction of n.
const BALANCE_SLACK: f64 = 0.25;

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute a Nested Dissection ordering for the symmetric graph of `a`.
///
/// Returns `perm` where `perm[i]` = the original node index placed at
/// position `i` in the new ordering.  Plugs in identically to `rcm` and
/// `colamd`.
///
/// # Example
/// ```
/// use linger::direct::ordering::nd;
/// use linger::sparse::{CooMatrix, CsrMatrix};
///
/// let mut coo: CooMatrix<f64> = CooMatrix::new(4, 4);
/// for i in 0..4usize { coo.push(i, i, 2.0); }
/// coo.push(0,1,-1.0); coo.push(1,0,-1.0);
/// coo.push(1,2,-1.0); coo.push(2,1,-1.0);
/// coo.push(2,3,-1.0); coo.push(3,2,-1.0);
/// let csr = CsrMatrix::from_coo(&coo);
/// let perm = nd(&csr);
/// assert_eq!(perm.len(), 4);
/// ```
pub fn nd<T: Scalar>(a: &CsrMatrix<T>) -> Vec<usize> {
    let n = a.nrows();
    if n == 0 { return vec![]; }

    let global_adj = build_symmetric_adj(a);
    let all_nodes: Vec<usize> = (0..n).collect();
    let mut perm = Vec::with_capacity(n);
    nd_recurse(&global_adj, &all_nodes, 0, &mut perm);

    // Any remaining nodes not reached (should not happen for connected graphs,
    // but guard against missing disconnected components via the loop in
    // nd_recurse's base case).
    debug_assert_eq!(perm.len(), n, "ND produced wrong-length permutation");
    perm
}

// ─── Coarsening ──────────────────────────────────────────────────────────────

struct Coarsening {
    /// Coarse graph adjacency (locally indexed 0..nc).
    coarse_adj:     Vec<Vec<usize>>,
    /// Maps fine local index → coarse local index.
    fine_to_coarse: Vec<usize>,
    /// Maps coarse local index → list of fine local indices.
    coarse_to_fine: Vec<Vec<usize>>,
}

/// One round of Heavy-Edge Matching coarsening.
/// Greedily matches each unmatched node with the unmatched neighbour that has
/// the smallest degree (ties: smallest index).  Unmatched nodes become
/// singleton supernodes.
fn hem_coarsen(local_adj: &[Vec<usize>]) -> Coarsening {
    let nf = local_adj.len();
    let mut matched = vec![false; nf];
    let mut fine_to_coarse = vec![usize::MAX; nf];
    let mut coarse_to_fine: Vec<Vec<usize>> = Vec::new();

    // Process nodes in order; greedy minimum-degree matching.
    for v in 0..nf {
        if matched[v] { continue; }
        // Find best unmatched neighbour (min degree, then min index).
        let best = local_adj[v].iter()
            .copied()
            .filter(|&w| !matched[w])
            .min_by_key(|&w| (local_adj[w].len(), w));

        let cid = coarse_to_fine.len();
        if let Some(w) = best {
            matched[v] = true;
            matched[w] = true;
            fine_to_coarse[v] = cid;
            fine_to_coarse[w] = cid;
            coarse_to_fine.push(vec![v, w]);
        } else {
            matched[v] = true;
            fine_to_coarse[v] = cid;
            coarse_to_fine.push(vec![v]);
        }
    }

    let nc = coarse_to_fine.len();

    // Build coarse adjacency: for each coarse node c, take the union of
    // adj(fine nodes in c), map to coarse IDs, remove self-loops, dedup.
    let mut coarse_adj: Vec<Vec<usize>> = vec![Vec::new(); nc];
    for c in 0..nc {
        for &fv in &coarse_to_fine[c] {
            for &fw in &local_adj[fv] {
                let cw = fine_to_coarse[fw];
                if cw != c {
                    coarse_adj[c].push(cw);
                }
            }
        }
        coarse_adj[c].sort_unstable();
        coarse_adj[c].dedup();
    }

    Coarsening { coarse_adj, fine_to_coarse, coarse_to_fine }
}

/// Coarsen repeatedly until `|coarse| <= target` or no further contraction.
fn hem_coarsen_multilevel(local_adj: &[Vec<usize>], target: usize) -> Vec<Coarsening> {
    let mut levels: Vec<Coarsening> = Vec::new();
    let mut current: &[Vec<usize>];
    // We need ownership so use a temporary Vec.
    let mut owned: Vec<Vec<usize>> = local_adj.to_vec();

    loop {
        let n = owned.len();
        if n <= target { break; }
        let c = hem_coarsen(&owned);
        let nc = c.coarse_adj.len();
        owned = c.coarse_adj.clone();
        levels.push(c);
        // Stop if no progress (contraction ratio < 5%).
        if nc as f64 > n as f64 * 0.95 { break; }
    }
    levels
}

// ─── Bipartition ─────────────────────────────────────────────────────────────

/// BFS two-coloring from two pseudo-peripheral poles u and v.
///
/// Returns `color[i]` ∈ {0, 1, 2} where 2 = uncolored (separator candidate).
/// In practice almost all nodes get color 0 or 1.
fn bfs_bipartition(local_adj: &[Vec<usize>]) -> Vec<u8> {
    let n = local_adj.len();
    if n == 0 { return vec![]; }
    if n == 1 { return vec![0]; }

    // Find pseudo-peripheral pair (u, v) via two BFS passes.
    let u = bfs_farthest(0, local_adj);
    let v = bfs_farthest(u, local_adj);

    // Simultaneous BFS from u (color=0) and v (color=1).
    // First-arrival wins.
    let mut color = vec![u8::MAX; n];
    let mut queue: VecDeque<(usize, u8)> = VecDeque::new();
    color[u] = 0; queue.push_back((u, 0));
    color[v] = 1; queue.push_back((v, 1));

    while let Some((node, c)) = queue.pop_front() {
        for &nb in &local_adj[node] {
            if color[nb] == u8::MAX {
                color[nb] = c;
                queue.push_back((nb, c));
            }
        }
    }

    // Handle disconnected components: assign uncolored nodes to side 0.
    for c in color.iter_mut() {
        if *c == u8::MAX { *c = 0; }
    }

    color
}

/// Return the farthest node from `start` via BFS (last node visited).
fn bfs_farthest(start: usize, adj: &[Vec<usize>]) -> usize {
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    visited[start] = true;
    queue.push_back(start);
    let mut last = start;
    while let Some(v) = queue.pop_front() {
        last = v;
        for &nb in &adj[v] {
            if !visited[nb] {
                visited[nb] = true;
                queue.push_back(nb);
            }
        }
    }
    last
}

// ─── Separator extraction ─────────────────────────────────────────────────────

/// Extract a vertex separator from a 2-colored graph.
///
/// Strategy: identify all cross-edges (edges between color-0 and color-1 nodes).
/// Move boundary nodes from the smaller side into the separator until no
/// cross-edges remain.
///
/// Returns `(part_a, part_b, sep)` — three disjoint sets covering all nodes,
/// all in local (0-indexed) coordinates.
fn extract_vertex_separator(
    local_adj: &[Vec<usize>],
    color: &[u8],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = local_adj.len();

    // side[i] ∈ {0=A, 1=B, 2=Sep}
    let mut side: Vec<u8> = color.to_vec();

    // Find border nodes on side A that have a B-neighbour and move them to Sep.
    // Repeat until no cross-edges exist.
    loop {
        let mut moved = false;
        for v in 0..n {
            if side[v] != 0 { continue; }
            let has_b_nbr = local_adj[v].iter().any(|&w| side[w] == 1);
            if has_b_nbr {
                side[v] = 2;
                moved = true;
            }
        }
        if !moved { break; }
    }

    let mut part_a: Vec<usize> = Vec::new();
    let mut part_b: Vec<usize> = Vec::new();
    let mut sep:    Vec<usize> = Vec::new();
    for v in 0..n {
        match side[v] {
            0 => part_a.push(v),
            1 => part_b.push(v),
            _ => sep.push(v),
        }
    }
    (part_a, part_b, sep)
}

// ─── FM refinement ───────────────────────────────────────────────────────────

/// Fiduccia-Mattheyses greedy separator refinement.
///
/// Attempts to reduce `|sep|` while maintaining |A| ≈ |B| by moving boundary
/// nodes between the separator and the two partitions.
fn fm_refine(
    local_adj: &[Vec<usize>],
    part_a: &mut Vec<usize>,
    part_b: &mut Vec<usize>,
    sep:    &mut Vec<usize>,
) {
    let n = local_adj.len();
    if n == 0 { return; }

    // Build side assignment array: 0=A, 1=B, 2=Sep.
    let mut side = vec![0u8; n];
    for &v in part_a.iter() { side[v] = 0; }
    for &v in part_b.iter() { side[v] = 1; }
    for &v in sep.iter()    { side[v] = 2; }

    let total = n;

    for _pass in 0..MAX_FM_PASSES {
        let mut any_move = false;

        // Try to move separator nodes back into A or B if they have no
        // neighbours in the opposite side (would remove them from separator
        // without reintroducing cross-edges).
        let sep_snap: Vec<usize> = sep.clone();
        for &v in &sep_snap {
            let has_a = local_adj[v].iter().any(|&w| side[w] == 0);
            let has_b = local_adj[v].iter().any(|&w| side[w] == 1);

            if !has_a && !has_b {
                // Isolated in separator — assign to the smaller side.
                let target = if part_a.len() <= part_b.len() { 0u8 } else { 1u8 };
                side[v] = target;
                any_move = true;
            } else if !has_b {
                // Only touches A — safe to move to A.
                let new_size_a = part_a.len() + 1;
                let new_size_b = part_b.len();
                let slack = (BALANCE_SLACK * total as f64) as usize + 1;
                if new_size_a <= new_size_b + slack {
                    side[v] = 0;
                    any_move = true;
                }
            } else if !has_a {
                // Only touches B — safe to move to B.
                let new_size_b = part_b.len() + 1;
                let new_size_a = part_a.len();
                let slack = (BALANCE_SLACK * total as f64) as usize + 1;
                if new_size_b <= new_size_a + slack {
                    side[v] = 1;
                    any_move = true;
                }
            }
        }

        // Rebuild part_a, part_b, sep from side[].
        part_a.clear(); part_b.clear(); sep.clear();
        for v in 0..n {
            match side[v] {
                0 => part_a.push(v),
                1 => part_b.push(v),
                _ => sep.push(v),
            }
        }

        if !any_move { break; }
    }
}

// ─── Local adjacency for a subset ────────────────────────────────────────────

/// Build locally-indexed adjacency for the subset `nodes` of the global graph.
///
/// Only edges where *both* endpoints are in `nodes` are included.
/// Returns `local_adj[i]` = locally-indexed neighbours of `nodes[i]`.
fn induced_local_adj(global_adj: &[Vec<usize>], nodes: &[usize]) -> Vec<Vec<usize>> {
    let m = nodes.len();
    // Inverse map: global → local (use sentinel usize::MAX for non-members).
    let n = global_adj.len();
    let mut global_to_local = vec![usize::MAX; n];
    for (local, &g) in nodes.iter().enumerate() {
        global_to_local[g] = local;
    }

    let mut local_adj: Vec<Vec<usize>> = vec![Vec::new(); m];
    for (local_i, &g) in nodes.iter().enumerate() {
        for &nb_g in &global_adj[g] {
            let nb_l = global_to_local[nb_g];
            if nb_l != usize::MAX {
                local_adj[local_i].push(nb_l);
            }
        }
        // Already sorted and deduped because global_adj was built that way.
    }
    local_adj
}

// ─── Recursive ND ─────────────────────────────────────────────────────────────

fn nd_recurse(
    global_adj: &[Vec<usize>],
    nodes: &[usize],
    depth: usize,
    perm: &mut Vec<usize>,
) {
    let n = nodes.len();

    // Base case: use AMD directly.
    if n <= BASE_SIZE || depth >= MAX_DEPTH {
        let local_adj = induced_local_adj(global_adj, nodes);
        let local_perm = amd_small(&local_adj);
        for local_i in local_perm {
            perm.push(nodes[local_i]);
        }
        return;
    }

    // ── Coarsen ──────────────────────────────────────────────────────────────
    let local_adj = induced_local_adj(global_adj, nodes);
    let coarsenings = hem_coarsen_multilevel(&local_adj, COARSE_TARGET);

    // If no coarsening happened (very small or dense graph), fall back to AMD.
    if coarsenings.is_empty() {
        let local_perm = amd_small(&local_adj);
        for local_i in local_perm {
            perm.push(nodes[local_i]);
        }
        return;
    }

    // Coarsest level adjacency.
    let coarse_adj = &coarsenings.last().unwrap().coarse_adj;

    // ── Partition coarsest graph ─────────────────────────────────────────────
    let color = bfs_bipartition(coarse_adj);
    let (mut ca, mut cb, mut csep) = extract_vertex_separator(coarse_adj, &color);
    fm_refine(coarse_adj, &mut ca, &mut cb, &mut csep);

    // ── Uncoarsen: project partition back to fine level ──────────────────────
    // At each level, expand coarse node membership to fine nodes.
    let mut fine_a:   Vec<bool> = vec![false; coarse_adj.len()];
    let mut fine_b:   Vec<bool> = vec![false; coarse_adj.len()];
    let mut fine_sep: Vec<bool> = vec![false; coarse_adj.len()];
    for &c in &ca   { fine_a[c]   = true; }
    for &c in &cb   { fine_b[c]   = true; }
    for &c in &csep { fine_sep[c] = true; }

    // Walk coarsening levels in reverse.
    let mut n_fine = n;
    let mut a_mask:   Vec<bool>;
    let mut b_mask:   Vec<bool>;
    let mut sep_mask: Vec<bool>;

    {
        // Start at the coarsest resolution and work backwards.
        let mut cur_a   = fine_a;
        let mut cur_b   = fine_b;
        let mut cur_sep = fine_sep;
        let mut cur_size = coarse_adj.len();

        for level in coarsenings.iter().rev() {
            let nf = level.coarse_to_fine.len(); // this is NOT the fine size
            // Actually fine_to_coarse.len() == fine size for this level.
            let fine_size = level.fine_to_coarse.len();
            let mut next_a   = vec![false; fine_size];
            let mut next_b   = vec![false; fine_size];
            let mut next_sep = vec![false; fine_size];
            for f in 0..fine_size {
                let c = level.fine_to_coarse[f];
                if c < cur_size {
                    if cur_a[c]   { next_a[f]   = true; }
                    if cur_b[c]   { next_b[f]   = true; }
                    if cur_sep[c] { next_sep[f] = true; }
                }
            }
            cur_a   = next_a;
            cur_b   = next_b;
            cur_sep = next_sep;
            cur_size = fine_size;
            n_fine = fine_size;
            let _ = nf;
        }
        a_mask   = cur_a;
        b_mask   = cur_b;
        sep_mask = cur_sep;
    }

    // Convert masks to local-index lists.
    // Nodes not assigned to A, B, or Sep (shouldn't happen, but guard):
    // assign to the smaller side.
    let mut nodes_a:   Vec<usize> = Vec::new();
    let mut nodes_b:   Vec<usize> = Vec::new();
    let mut nodes_sep: Vec<usize> = Vec::new();
    for i in 0..n_fine {
        if sep_mask[i] {
            nodes_sep.push(i);
        } else if a_mask[i] {
            nodes_a.push(i);
        } else if b_mask[i] {
            nodes_b.push(i);
        } else {
            // Unassigned: put in the smaller partition.
            if nodes_a.len() <= nodes_b.len() {
                nodes_a.push(i);
            } else {
                nodes_b.push(i);
            }
        }
    }

    // If separator is empty or one partition is empty, fall back to AMD.
    if nodes_sep.is_empty() || nodes_a.is_empty() || nodes_b.is_empty() {
        let local_perm = amd_small(&local_adj);
        for li in local_perm {
            perm.push(nodes[li]);
        }
        return;
    }

    // ── Map local indices back to global nodes ────────────────────────────────
    let global_a:   Vec<usize> = nodes_a.iter().map(|&li| nodes[li]).collect();
    let global_b:   Vec<usize> = nodes_b.iter().map(|&li| nodes[li]).collect();
    let global_sep: Vec<usize> = nodes_sep.iter().map(|&li| nodes[li]).collect();

    // ── Recurse ───────────────────────────────────────────────────────────────
    nd_recurse(global_adj, &global_a, depth + 1, perm);
    nd_recurse(global_adj, &global_b, depth + 1, perm);

    // Separator last.
    for g in global_sep {
        perm.push(g);
    }
}

// ─── AMD base case ───────────────────────────────────────────────────────────

/// Small-graph AMD.  Reuses the `pub(crate) amd_order` from sibling module.
fn amd_small(local_adj: &[Vec<usize>]) -> Vec<usize> {
    let n = local_adj.len();
    if n == 0 { return vec![]; }
    super::colamd::amd_order(n, local_adj)
}

// ─── Graph utilities ─────────────────────────────────────────────────────────

pub(super) fn build_symmetric_adj<T: Scalar>(a: &CsrMatrix<T>) -> Vec<Vec<usize>> {
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

    /// 2D Laplacian on an n×n grid.
    fn laplacian_2d(n: usize) -> CsrMatrix<f64> {
        let nn = n * n;
        let mut coo = CooMatrix::new(nn, nn);
        for i in 0..n {
            for j in 0..n {
                let id = i * n + j;
                coo.push(id, id, 4.0);
                if j > 0     { coo.push(id, id - 1, -1.0); coo.push(id - 1, id, -1.0); }
                if i > 0     { coo.push(id, id - n, -1.0); coo.push(id - n, id, -1.0); }
            }
        }
        CsrMatrix::from_coo(&coo)
    }

    fn is_permutation(perm: &[usize], n: usize) -> bool {
        if perm.len() != n { return false; }
        let mut seen = vec![false; n];
        for &v in perm { if v >= n || seen[v] { return false; } seen[v] = true; }
        true
    }

    #[test]
    fn nd_is_permutation_tridiag_n20() {
        let a = tridiagonal(20);
        let perm = nd(&a);
        assert!(is_permutation(&perm, 20), "nd produced invalid permutation for tridiag n=20");
    }

    #[test]
    fn nd_is_permutation_grid_4x4() {
        let a = laplacian_2d(4);
        let perm = nd(&a);
        assert!(is_permutation(&perm, 16), "nd produced invalid permutation for 4x4 grid");
    }

    #[test]
    fn nd_is_permutation_large() {
        let a = tridiagonal(300);
        let perm = nd(&a);
        assert!(is_permutation(&perm, 300));
    }

    #[test]
    fn nd_empty() {
        let a: CsrMatrix<f64> = CsrMatrix::from_coo(&CooMatrix::new(0, 0));
        assert_eq!(nd(&a), vec![]);
    }

    #[test]
    fn nd_single_node() {
        let mut coo = CooMatrix::new(1, 1);
        coo.push(0, 0, 1.0);
        assert_eq!(nd(&CsrMatrix::from_coo(&coo)), vec![0]);
    }

    #[test]
    fn nd_two_nodes() {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 2.0); coo.push(1, 1, 2.0);
        coo.push(0, 1, -1.0); coo.push(1, 0, -1.0);
        let perm = nd(&CsrMatrix::from_coo(&coo));
        assert!(is_permutation(&perm, 2));
    }
}
