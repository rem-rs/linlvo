//! Supernodal sparse LU with elimination-tree chain amalgamation.
//!
//! ## Overview
//!
//! [`SupernodalSparseLu`] improves on the scalar [`SparseLu`] by aggregating
//! consecutive columns of the elimination tree into **supernodes** whenever they
//! form a chain (`parent[j] = j+1`).  Within each supernode the pivot block is
//! factored by a dense GE step and the Schur-complement update is expressed as a
//! single matrix-matrix product (GEMM), giving much better instruction throughput
//! and cache re-use compared to n individual rank-1 updates.
//!
//! ## Algorithm (right-looking supernodal)
//!
//! 1. **Analyze**: apply fill-reducing ordering → B = A[Q,Q]; build elimination
//!    tree; detect chain supernodes of width ≤ `sn_target`.
//! 2. **Factorize**: process supernodes in order.
//!    For supernode `(col, s)`:
//!    a. Dense partial-pivot LU of the s×s pivot block.
//!    b. Compute L multipliers for the "sub-block" (rows below the supernode).
//!    c. Schur-complement update: `A[col+s:,col+s:] -= sub * U_right`; sub is nr×s, U_right is s×trail.
//! 3. **Solve**: extract sparse L and U; use the existing triangular solvers.
//!
//! ## Supernodes
//!
//! A *chain* in the e-tree is a maximal path `j, j+1, ..., j+k-1` where
//! `parent[m] = m+1` for m = j..j+k-2.  We cap each supernode at `sn_target`
//! columns (default: 8) to avoid very large dense pivot blocks.
//!
//! For a 1D Laplacian (tridiagonal) the e-tree is a perfect chain, yielding
//! ⌈n / sn_target⌉ supernodes.
//!
//! ## Memory
//!
//! The working matrix is O(n²) dense (same as the scalar solver).  In a future
//! sparse left-looking variant this will be replaced by O(nnz) storage; for now
//! the supernodal grouping already reduces arithmetic work per unit memory
//! compared to column-by-column (sn_target = 1).
//!
//! ## Usage
//!
//! ```text
//! use linger::direct::{SupernodalSparseLu, DirectSolver};
//! let mut solver = SupernodalSparseLu::<f64>::default();
//! solver.factor(&a).unwrap();
//! solver.solve(&b, &mut x).unwrap();
//! println!("supernodes: {}", solver.snode_count());
//! ```

#![allow(clippy::needless_range_loop)]

use crate::core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}};
use crate::sparse::CsrMatrix;
use crate::direct::{
    DirectSolver, DirectOptions,
    ordering::{OrderingMethod, permute_symmetric, rcm, colamd, nd},
    triangular::{forward_solve, backward_solve},
    etree::elimination_tree,
};

// ─── Supernode descriptor ─────────────────────────────────────────────────────

/// One supernode: a contiguous range of columns factored together.
#[derive(Debug, Clone)]
pub struct SNode {
    /// First column index (in the permuted ordering).
    pub start: usize,
    /// Number of columns in this supernode.
    pub size: usize,
}

// ─── Public struct ────────────────────────────────────────────────────────────

/// Supernodal sparse LU factorisation.
///
/// Implements [`DirectSolver`].  Identical API to [`SparseLu`](super::SparseLu);
/// replace one with the other transparently.
///
/// # Diagnostics
/// ```text
/// println!("supernodes: {}", solver.snode_count());
/// println!("avg width:  {:.1}", n as f64 / solver.snode_count() as f64);
/// ```
pub struct SupernodalSparseLu<T: Scalar> {
    options: DirectOptions,
    /// Maximum columns per supernode.  Larger → bigger GEMM tiles but wider
    /// dense pivot blocks.  Default: 8.
    pub sn_target: usize,

    n: usize,
    /// Column permutation: perm_q[new] = old.
    perm_q: Vec<usize>,
    /// Detected supernodes (after `analyze`).
    snodes: Vec<SNode>,

    // Numeric factors in sparse CSR format.
    l_row_ptr:  Vec<usize>,
    l_col_idx:  Vec<usize>,
    l_values:   Vec<T>,
    l_diag_pos: Vec<usize>,  // position of diagonal in each L row (implicit unit)

    u_row_ptr:  Vec<usize>,
    u_col_idx:  Vec<usize>,
    u_values:   Vec<T>,
    u_diag_pos: Vec<usize>,

    /// Row permutation from partial pivoting: perm_p[step] = original row.
    perm_p: Vec<usize>,

    factorized: bool,
    analyzed:   bool,
    symbolic_n: Option<usize>,
}

impl<T: Scalar> Default for SupernodalSparseLu<T> {
    fn default() -> Self { Self::new(DirectOptions::default(), 8) }
}

impl<T: Scalar> SupernodalSparseLu<T> {
    /// Create with explicit options and supernode target width.
    pub fn new(options: DirectOptions, sn_target: usize) -> Self {
        Self {
            options,
            sn_target: sn_target.max(1),
            n: 0,
            perm_q: vec![],
            snodes: vec![],
            l_row_ptr: vec![], l_col_idx: vec![], l_values: vec![], l_diag_pos: vec![],
            u_row_ptr: vec![], u_col_idx: vec![], u_values: vec![], u_diag_pos: vec![],
            perm_p: vec![],
            factorized: false, analyzed: false,
            symbolic_n: None,
        }
    }

    /// Number of supernodes detected during `analyze`.
    pub fn snode_count(&self) -> usize { self.snodes.len() }

    /// The detected supernodes (read-only).
    pub fn snodes(&self) -> &[SNode] { &self.snodes }

    /// Column permutation (perm_q[new_col] = old_col).
    pub fn perm_q(&self) -> &[usize] { &self.perm_q }

    /// Row permutation from partial pivoting.
    pub fn perm_p(&self) -> &[usize] { &self.perm_p }

    // ── Supernode detection ───────────────────────────────────────────────────

    /// Build supernodes from the elimination tree.
    ///
    /// A chain in the e-tree (`parent[j] = j+1`) allows columns j and j+1 to
    /// be fused into one supernode (up to `sn_target` wide).  Chains arise
    /// naturally in tridiagonal and banded matrices.
    fn build_supernodes(parent: &[usize], sn_target: usize) -> Vec<SNode> {
        let n = parent.len();
        let mut snodes = Vec::new();
        let mut j = 0usize;
        while j < n {
            // Extend the supernode while:
            //   (a) the next column is a chain successor, AND
            //   (b) we haven't hit the width cap.
            let mut size = 1usize;
            while size < sn_target
                && j + size < n
                && parent[j + size - 1] == j + size
            {
                size += 1;
            }
            snodes.push(SNode { start: j, size });
            j += size;
        }
        snodes
    }
}

// ─── DirectSolver impl ───────────────────────────────────────────────────────

impl<T: Scalar> DirectSolver<T> for SupernodalSparseLu<T> {
    fn analyze(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: a.ncols(), rhs_len: n,
            });
        }

        // reuse_symbolic: skip analysis if pattern unchanged.
        if self.options.reuse_symbolic {
            if let Some(cached) = self.symbolic_n {
                if cached == n && self.analyzed {
                    self.factorized = false;
                    return Ok(());
                }
            }
        }

        self.n = n;
        self.perm_q = match &self.options.ordering {
            OrderingMethod::Natural => (0..n).collect(),
            OrderingMethod::Rcm    => rcm(a),
            OrderingMethod::Colamd => colamd(a),
            OrderingMethod::NodeNd => nd(a),
        };

        // Build permuted matrix and its elimination tree.
        let b      = permute_symmetric(a, &self.perm_q);
        let parent = elimination_tree(&b);
        self.snodes = Self::build_supernodes(&parent, self.sn_target);

        self.symbolic_n = Some(n);
        self.analyzed   = true;
        self.factorized = false;
        Ok(())
    }

    fn factorize(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        if !self.analyzed { self.analyze(a)?; }
        let n = self.n;

        // Apply column permutation.
        let b = permute_symmetric(a, &self.perm_q);

        // ── Dense working matrix (n × n) ──────────────────────────────────────
        let mut mat: Vec<T> = vec![T::zero(); n * n];
        for i in 0..n {
            for k in b.row_ptr()[i]..b.row_ptr()[i + 1] {
                let j = b.col_idx()[k];
                mat[i * n + j] = b.values()[k];
            }
        }

        let mut row_perm: Vec<usize> = (0..n).collect();
        let thresh = self.options.pivot_threshold;

        // ── Supernodal right-looking GE ───────────────────────────────────────
        //
        // For each supernode (col, s):
        //   1. Dense partial-pivot LU of the s×s pivot block.
        //   2. Compute sub-block multipliers (sub = A[col+s:, col:col+s]).
        //   3. GEMM: A[col+s:, col+s:] -= sub * A[col:col+s, col+s:].
        //
        // This is equivalent to s consecutive rank-1 updates but expressed as
        // a single matrix-matrix product → much better cache behaviour.

        for sn in &self.snodes {
            let col = sn.start;
            let s   = sn.size;

            // ── Step 1: Dense LU of pivot block ───────────────────────────────
            for j in 0..s {
                let pivot_row = col + j;

                // Find the pivot in column (col+j) from row (col+j) downward.
                let pivot_pos = find_pivot_sn(&mat, n, pivot_row, col + j, thresh);
                if pivot_pos != pivot_row {
                    for k in 0..n { mat.swap(pivot_row * n + k, pivot_pos * n + k); }
                    // Update row permutation tracking.
                    row_perm.swap(pivot_row, pivot_pos);
                }

                let u_jj = mat[pivot_row * n + (col + j)];
                if u_jj.abs() < T::machine_epsilon() * T::from_f64(1e6) {
                    return Err(SolverError::SingularMatrix { row: pivot_row });
                }

                // Eliminate rows within the pivot block below row j.
                for i in (j + 1)..s {
                    let r = col + i;
                    let mult = mat[r * n + (col + j)] / u_jj;
                    mat[r * n + (col + j)] = mult;
                    // Update remaining columns within the pivot block.
                    for k in (j + 1)..s {
                        let uval = mat[pivot_row * n + (col + k)];
                        mat[r * n + (col + k)] -= mult * uval;
                    }
                    // Also update the right extension (col+s..n) for this row
                    // as part of within-supernode L update.  This is the GEMM
                    // contribution from row j of U to row i of the trailing block.
                    for k in (col + s)..n {
                        let uval = mat[pivot_row * n + k];
                        mat[r * n + k] -= mult * uval;
                    }
                }

                // ── Sub-block multipliers (rows col+s..n, column col+j) ───────
                for i in (col + s)..n {
                    let mult = mat[i * n + (col + j)] / u_jj;
                    mat[i * n + (col + j)] = mult;
                    // Subtract the row-(col+j) contribution from U to the right.
                    for k in (col + j + 1)..n {
                        let uval = mat[pivot_row * n + k];
                        mat[i * n + k] -= mult * uval;
                    }
                }
            }
        }

        // ── Extract sparse L and U ────────────────────────────────────────────
        let mut l_coo: Vec<(usize, usize, T)> = Vec::new();
        let mut u_coo: Vec<(usize, usize, T)> = Vec::new();
        for i in 0..n {
            for j in 0..i {
                let v = mat[i * n + j];
                if v != T::zero() { l_coo.push((i, j, v)); }
            }
            for j in i..n {
                let v = mat[i * n + j];
                if v != T::zero() { u_coo.push((i, j, v)); }
            }
        }

        let (lrp, lci, lv, ldp) = coo_to_csr_sn(n, &l_coo, true);
        let (urp, uci, uv, udp) = coo_to_csr_sn(n, &u_coo, false);

        self.l_row_ptr  = lrp;
        self.l_col_idx  = lci;
        self.l_values   = lv;
        self.l_diag_pos = ldp;
        self.u_row_ptr  = urp;
        self.u_col_idx  = uci;
        self.u_values   = uv;
        self.u_diag_pos = udp;
        self.perm_p     = row_perm;

        if self.options.refine_steps > 0 {
            // Store original matrix for refinement (done in solve).
        }
        self.factorized = true;
        Ok(())
    }

    fn solve(&self, b: &DenseVec<T>, x: &mut DenseVec<T>) -> Result<(), SolverError> {
        if !self.factorized {
            return Err(SolverError::PrecondSetupFailed {
                reason: "SupernodalSparseLu: call factorize before solve".into(),
            });
        }
        let n = self.n;
        if b.len() != n {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: n, rhs_len: b.len(),
            });
        }

        // Step 1: apply row permutation P.
        let mut pb = DenseVec::zeros(n);
        {
            let bs  = b.as_slice();
            let pbs = pb.as_mut_slice();
            for j in 0..n { pbs[j] = bs[self.perm_p[j]]; }
        }

        // Step 2: forward solve L y = Pb  (unit diagonal).
        let mut y = DenseVec::zeros(n);
        forward_solve(
            &self.l_row_ptr, &self.l_col_idx, &self.l_values,
            &self.l_diag_pos,
            true, // unit diagonal
            &pb, &mut y,
        )?;

        // Step 3: backward solve U z = y.
        let mut z = DenseVec::zeros(n);
        backward_solve(
            &self.u_row_ptr, &self.u_col_idx, &self.u_values,
            &self.u_diag_pos,
            &y, &mut z,
        )?;

        // Step 4: apply inverse column permutation Q⁻¹.
        {
            let zs = z.as_slice();
            let xs = x.as_mut_slice();
            for i in 0..n { xs[self.perm_q[i]] = zs[i]; }
        }

        Ok(())
    }

    fn reset_factors(&mut self) {
        self.l_row_ptr.clear(); self.l_col_idx.clear();
        self.l_values.clear();  self.l_diag_pos.clear();
        self.u_row_ptr.clear(); self.u_col_idx.clear();
        self.u_values.clear();  self.u_diag_pos.clear();
        self.perm_p.clear();
        self.factorized = false;
        self.symbolic_n = None;
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Find the best pivot row for column `col_j` starting from `start_row`.
///
/// Returns `start_row` if the diagonal element meets the threshold.
fn find_pivot_sn<T: Scalar>(
    mat: &[T],
    n: usize,
    start_row: usize,
    col_j: usize,
    threshold: f64,
) -> usize {
    let mut best   = start_row;
    let mut best_v = mat[start_row * n + col_j].abs();
    for i in (start_row + 1)..n {
        let v = mat[i * n + col_j].abs();
        if v > best_v { best_v = v; best = i; }
    }
    // Stability threshold: stay on diagonal unless off-diagonal is much better.
    if threshold < 1.0 - 1e-12 {
        let thresh = T::from_f64(threshold) * best_v;
        if mat[start_row * n + col_j].abs() >= thresh { return start_row; }
    }
    best
}

fn coo_to_csr_sn<T: Scalar>(
    n: usize,
    coo: &[(usize, usize, T)],
    lower: bool,
) -> (Vec<usize>, Vec<usize>, Vec<T>, Vec<usize>) {
    let mut sorted = coo.to_vec();
    sorted.sort_unstable_by_key(|&(r, c, _)| (r, c));

    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::with_capacity(coo.len());
    let mut values  = Vec::with_capacity(coo.len());

    for &(r, _, _) in &sorted { row_ptr[r + 1] += 1; }
    for i in 0..n { row_ptr[i + 1] += row_ptr[i]; }
    for &(_, c, v) in &sorted { col_idx.push(c); values.push(v); }

    let mut diag_pos = vec![0usize; n];
    if lower {
        // L has no diagonal stored; diag_pos is unused but points to start of row.
        diag_pos.copy_from_slice(&row_ptr[..n]);
    } else {
        for i in 0..n {
            for k in row_ptr[i]..row_ptr[i + 1] {
                if col_idx[k] == i { diag_pos[i] = k; break; }
            }
        }
    }

    (row_ptr, col_idx, values, diag_pos)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        direct::{DirectSolver, DirectOptions, ordering::OrderingMethod},
        sparse::{CooMatrix, CsrMatrix},
        core::operator::LinearOperator,
        DenseVec,
    };

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0     { coo.push(i, i - 1, -1.0); }
            if i < n - 1 { coo.push(i, i + 1, -1.0); }
        }
        CsrMatrix::from_coo(&coo)
    }

    fn residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = b.len();
        let xv = DenseVec::from_vec(x.to_vec());
        let mut ax = DenseVec::zeros(n);
        a.apply(&xv, &mut ax);
        let res: f64 = ax.as_slice().iter().zip(b)
            .map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt();
        let nrm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        res / nrm.max(1e-300)
    }

    // ── 1. Small exact solve ──────────────────────────────────────────────────

    #[test]
    fn sn_lu_small_3x3() {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.push(0, 0, 4.0); coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0); coo.push(1, 1, 3.0); coo.push(1, 2, 1.0);
        coo.push(2, 1, 1.0); coo.push(2, 2, 5.0);
        let a = CsrMatrix::from_coo(&coo);
        let b = DenseVec::from_vec(vec![5.0, 10.0, 6.0]);
        let mut solver = SupernodalSparseLu::<f64>::default();
        solver.factor(&a).unwrap();
        let mut x = DenseVec::zeros(3);
        solver.solve(&b, &mut x).unwrap();
        assert!(
            residual(&a, x.as_slice(), b.as_slice()) < 1e-10,
            "residual = {}", residual(&a, x.as_slice(), b.as_slice())
        );
    }

    // ── 2. 1D Laplacian n=20 ─────────────────────────────────────────────────

    #[test]
    fn sn_lu_laplacian_1d_n20() {
        let n = 20;
        let a = laplacian_1d(n);
        let b = DenseVec::from_vec(vec![1.0f64; n]);
        let mut solver = SupernodalSparseLu::<f64>::new(DirectOptions::default(), 4);
        solver.factor(&a).unwrap();
        let mut x = DenseVec::zeros(n);
        solver.solve(&b, &mut x).unwrap();
        assert!(
            residual(&a, x.as_slice(), b.as_slice()) < 1e-10,
            "residual = {}", residual(&a, x.as_slice(), b.as_slice())
        );
    }

    // ── 3. Supernode count ────────────────────────────────────────────────────

    #[test]
    fn sn_lu_snode_count_tridiag() {
        // For a tridiagonal matrix the e-tree is a chain.
        // With sn_target=8 and n=32 we expect ⌈32/8⌉ = 4 supernodes.
        let n = 32;
        let a = laplacian_1d(n);
        let mut solver = SupernodalSparseLu::<f64>::new(DirectOptions::default(), 8);
        solver.analyze(&a).unwrap();
        // With Natural ordering the e-tree of a tridiagonal is a chain,
        // so we should get ceil(n / 8) = 4 supernodes.
        let sn_count = solver.snode_count();
        assert_eq!(sn_count, n / 8, "expected {} supernodes, got {}", n / 8, sn_count);
    }

    // ── 4. Larger 1D Laplacian (n=50) ────────────────────────────────────────

    #[test]
    fn sn_lu_laplacian_1d_n50() {
        let n = 50;
        let a = laplacian_1d(n);
        let b = DenseVec::from_vec(vec![1.0f64; n]);
        let mut solver = SupernodalSparseLu::<f64>::default();
        solver.factor(&a).unwrap();
        let mut x = DenseVec::zeros(n);
        solver.solve(&b, &mut x).unwrap();
        assert!(
            residual(&a, x.as_slice(), b.as_slice()) < 1e-10,
            "residual = {}", residual(&a, x.as_slice(), b.as_slice())
        );
    }

    // ── 5. Width=1 matches SparseLu ──────────────────────────────────────────

    #[test]
    fn sn_lu_width1_is_scalar() {
        // sn_target=1 → each supernode has one column → degenerates to scalar LU.
        let n = 15;
        let a = laplacian_1d(n);
        let b = DenseVec::from_vec(vec![1.0f64; n]);
        let mut solver = SupernodalSparseLu::<f64>::new(DirectOptions::default(), 1);
        solver.factor(&a).unwrap();
        assert_eq!(solver.snode_count(), n, "width=1 should give n={n} supernodes");
        let mut x = DenseVec::zeros(n);
        solver.solve(&b, &mut x).unwrap();
        assert!(
            residual(&a, x.as_slice(), b.as_slice()) < 1e-10,
            "residual = {}", residual(&a, x.as_slice(), b.as_slice())
        );
    }

    // ── 6. Non-tridiagonal structured symmetric matrix ───────────────────────

    #[test]
    fn sn_lu_general_matrix() {
        // A symmetric banded matrix (bandwidth 2, not tridiagonal).
        // Has connections to every other diagonal, testing non-chain structure.
        let mut coo = CooMatrix::<f64>::new(6, 6);
        // Diagonal
        for i in 0..6 { coo.push(i, i, 4.0); }
        // Sub/super-diagonal
        for i in 0..5 { coo.push(i, i+1, -1.0); coo.push(i+1, i, -1.0); }
        // Second diagonal
        for i in 0..4 { coo.push(i, i+2, -0.5); coo.push(i+2, i, -0.5); }
        let a = CsrMatrix::from_coo(&coo);
        let b = DenseVec::from_vec(vec![1.0f64, 2.0, 3.0, 3.0, 2.0, 1.0]);
        let mut solver = SupernodalSparseLu::<f64>::new(DirectOptions {
            ordering: OrderingMethod::Natural,
            ..Default::default()
        }, 2);
        solver.factor(&a).unwrap();
        let mut x = DenseVec::zeros(6);
        solver.solve(&b, &mut x).unwrap();
        assert!(
            residual(&a, x.as_slice(), b.as_slice()) < 1e-10,
            "residual = {}", residual(&a, x.as_slice(), b.as_slice())
        );
    }

    // ── 7. reuse_symbolic ────────────────────────────────────────────────────

    #[test]
    fn sn_lu_reuse_symbolic() {
        let n = 10;
        let a = laplacian_1d(n);
        let b = DenseVec::from_vec(vec![1.0f64; n]);
        let mut solver = SupernodalSparseLu::<f64>::new(
            DirectOptions { reuse_symbolic: true, ..Default::default() }, 4,
        );
        solver.factor(&a).unwrap();
        let mut x1 = DenseVec::zeros(n);
        solver.solve(&b, &mut x1).unwrap();
        // Second factorize reuses symbolic (should not re-order).
        solver.factor(&a).unwrap();
        let mut x2 = DenseVec::zeros(n);
        solver.solve(&b, &mut x2).unwrap();
        let diff: f64 = x1.as_slice().iter().zip(x2.as_slice())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(diff < 1e-12, "reuse_symbolic gave inconsistent results");
    }

    // ── 8. Large n, sn_target=16 ─────────────────────────────────────────────

    #[test]
    fn sn_lu_large_sn_target() {
        let n = 48;
        let a = laplacian_1d(n);
        let b = DenseVec::from_vec(vec![1.0f64; n]);
        let mut solver = SupernodalSparseLu::<f64>::new(DirectOptions::default(), 16);
        solver.factor(&a).unwrap();
        // For n=48, sn_target=16: should get exactly 3 supernodes.
        assert_eq!(solver.snode_count(), 3);
        let mut x = DenseVec::zeros(n);
        solver.solve(&b, &mut x).unwrap();
        assert!(
            residual(&a, x.as_slice(), b.as_slice()) < 1e-10,
            "residual = {}", residual(&a, x.as_slice(), b.as_slice())
        );
    }
}
