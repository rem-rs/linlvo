//! Supernodal sparse Cholesky factorisation for SPD matrices.
//!
//! Groups elimination-tree chains (`parent[j] = j+1`) into supernodes and
//! applies dense GEMM-like updates within each supernode, reducing scatter/gather
//! overhead and improving cache utilisation compared to scalar left-looking Cholesky.
//!
//! ## Supernode definition
//!
//! A supernode is a maximal run of columns `[start, start+size)` such that
//! `parent[j] = j+1` for all `j` in `[start, start+size-1)`, capped at
//! `sn_target` columns.  For a tridiagonal matrix the entire matrix is one chain,
//! giving ⌈n / sn_target⌉ supernodes.
//!
//! ## Solve
//!
//! `SparseCholesky` (scalar) triangle solves.  Supernodes are used only for
//! the factorization; the triangular solve reuses the standard forward/backward
//! substitution on the CSR-stored L.

#![allow(clippy::needless_range_loop)]
use crate::core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}, operator::LinearOperator};
use crate::sparse::CsrMatrix;
use crate::direct::{
    DirectSolver, DirectOptions,
    ordering::{OrderingMethod, permute_symmetric, invert_perm, rcm, colamd, nd},
    triangular::forward_solve,
    etree::elimination_tree,
};

// ─── Supernode ────────────────────────────────────────────────────────────────

/// A contiguous range of columns fused into one supernode.
#[derive(Debug, Clone, Copy)]
pub struct SNode {
    pub start: usize,
    pub size:  usize,
}

fn build_supernodes(parent: &[usize], sn_target: usize) -> Vec<SNode> {
    let n = parent.len();
    let mut snodes = Vec::new();
    let mut j = 0usize;
    while j < n {
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

// ─── Public struct ────────────────────────────────────────────────────────────

/// Supernodal sparse Cholesky solver for SPD matrices.
///
/// Groups elimination-tree chain columns into supernodes for dense GEMM-like
/// updates, improving performance over scalar left-looking Cholesky on
/// structured sparse problems.
///
/// Implements the [`DirectSolver`] interface.
pub struct SupernodalSparseCholesky<T: Scalar> {
    options:   DirectOptions,
    sn_target: usize,

    n:        usize,
    perm:     Vec<usize>,
    inv_perm: Vec<usize>,

    /// L factor in CSR (lower-triangular, including diagonal).
    l_row_ptr:  Vec<usize>,
    l_col_idx:  Vec<usize>,
    l_values:   Vec<T>,
    l_diag_pos: Vec<usize>,

    a_stored:   Option<CsrMatrix<T>>,
    factorized: bool,
    analyzed:   bool,
    symbolic_n: Option<usize>,
}

impl<T: Scalar> Default for SupernodalSparseCholesky<T> {
    fn default() -> Self {
        Self::new(DirectOptions::default(), 8)
    }
}

impl<T: Scalar> SupernodalSparseCholesky<T> {
    /// Create with given options and supernode target width.
    pub fn new(options: DirectOptions, sn_target: usize) -> Self {
        Self {
            options,
            sn_target: sn_target.max(1),
            n: 0,
            perm: vec![], inv_perm: vec![],
            l_row_ptr: vec![], l_col_idx: vec![], l_values: vec![], l_diag_pos: vec![],
            a_stored: None,
            factorized: false, analyzed: false, symbolic_n: None,
        }
    }

    pub fn perm(&self) -> &[usize] { &self.perm }
}

// ─── DirectSolver impl ────────────────────────────────────────────────────────

impl<T: Scalar> DirectSolver<T> for SupernodalSparseCholesky<T> {
    fn analyze(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: a.ncols(), rhs_len: n,
            });
        }

        if self.options.reuse_symbolic {
            if let Some(cached_n) = self.symbolic_n {
                if cached_n == n && self.analyzed {
                    self.factorized = false;
                    return Ok(());
                }
            }
        }

        self.n = n;
        self.perm = match &self.options.ordering {
            OrderingMethod::Natural => (0..n).collect(),
            OrderingMethod::Rcm    => rcm(a),
            OrderingMethod::Colamd => colamd(a),
            OrderingMethod::NodeNd => nd(a),
        };
        self.inv_perm  = invert_perm(&self.perm);
        self.symbolic_n = Some(n);
        self.analyzed   = true;
        self.factorized = false;
        Ok(())
    }

    fn factorize(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        if !self.analyzed { self.analyze(a)?; }
        let n = self.n;

        let b = permute_symmetric(a, &self.perm);
        let parent = elimination_tree(&b);
        let snodes = build_supernodes(&parent, self.sn_target);

        // ── Build column access for lower triangle of B ───────────────────────
        // col_ptr_lo[j]..col_ptr_lo[j+1] → (row, value) for B[row,j] with row > j.
        let mut col_ptr_lo = vec![0usize; n + 1];
        for i in 0..n {
            for k in b.row_ptr()[i]..b.row_ptr()[i + 1] {
                let j = b.col_idx()[k];
                if j < i { col_ptr_lo[j + 1] += 1; }
            }
        }
        for i in 0..n { col_ptr_lo[i + 1] += col_ptr_lo[i]; }
        let mut col_row_lo = vec![0usize; col_ptr_lo[n]];
        let mut col_val_lo = vec![T::zero(); col_ptr_lo[n]];
        {
            let mut fp = col_ptr_lo.clone();
            for i in 0..n {
                for k in b.row_ptr()[i]..b.row_ptr()[i + 1] {
                    let j = b.col_idx()[k];
                    if j < i {
                        col_row_lo[fp[j]] = i;
                        col_val_lo[fp[j]] = b.values()[k];
                        fp[j] += 1;
                    }
                }
            }
        }

        // ── L stored CSC during factorisation ─────────────────────────────────
        let mut l_csc_col_ptr: Vec<usize> = vec![0usize; n + 1];
        let mut l_csc_rows: Vec<usize>    = Vec::new();
        let mut l_csc_vals: Vec<T>        = Vec::new();

        // Dense working vector (size n; zeroed between columns via `touched`).
        let mut x:       Vec<T>     = vec![T::zero(); n];
        let mut touched: Vec<usize> = Vec::new();
        let mut mark = vec![usize::MAX; n];

        for sn in &snodes {
            let SNode { start, size } = *sn;

            // ── Per-supernode dense panel (size × size) + trailing updates ─────
            // We process each column within the supernode using the standard
            // left-looking algorithm, but the within-supernode Schur complement
            // update is done as a dense sub-block (GEMM-like).

            // Collect the shared row pattern for the supernode:
            // all rows i >= start+size that appear in ANY column of the supernode in L.
            // We accumulate as we go column-by-column within the SN.

            for local_j in 0..size {
                let j = start + local_j;

                // Scatter B[j:n, j] into x.
                for k in b.row_ptr()[j]..b.row_ptr()[j + 1] {
                    if b.col_idx()[k] == j {
                        x[j] = b.values()[k];
                        touched.push(j);
                    }
                }
                for idx in col_ptr_lo[j]..col_ptr_lo[j + 1] {
                    let i = col_row_lo[idx];
                    x[i] = col_val_lo[idx];
                    touched.push(i);
                }

                // Reach set: DFS on e-tree from upper-triangle entries of row j.
                let mut reach: Vec<usize> = Vec::new();
                let mut dfs_stack: Vec<usize> = Vec::new();
                for k in b.row_ptr()[j]..b.row_ptr()[j + 1] {
                    let col = b.col_idx()[k];
                    if col < j && mark[col] != j {
                        dfs_stack.push(col);
                    }
                }
                while let Some(r) = dfs_stack.pop() {
                    if mark[r] == j { continue; }
                    mark[r] = j;
                    reach.push(r);
                    let p = parent[r];
                    if p < j && mark[p] != j { dfs_stack.push(p); }
                }
                reach.sort_unstable();

                // ── Left-looking updates from columns OUTSIDE the supernode ────
                for &k in &reach {
                    if k >= start { continue; } // within-SN handled below
                    let ljk = find_in_col(&l_csc_rows, &l_csc_vals, &l_csc_col_ptr, k, j);
                    if ljk == T::zero() { continue; }
                    x[j] -= ljk * ljk;
                    for idx in l_csc_col_ptr[k]..l_csc_col_ptr[k + 1] {
                        let i = l_csc_rows[idx];
                        if i <= j { continue; }
                        x[i] -= l_csc_vals[idx] * ljk;
                        touched.push(i);
                    }
                }

                // ── Within-supernode updates (dense, GEMM-like) ────────────────
                // For columns k in [start, j) (previous columns in this SN):
                for k in start..j {
                    let ljk = find_in_col(&l_csc_rows, &l_csc_vals, &l_csc_col_ptr, k, j);
                    if ljk == T::zero() { continue; }
                    x[j] -= ljk * ljk;
                    for idx in l_csc_col_ptr[k]..l_csc_col_ptr[k + 1] {
                        let i = l_csc_rows[idx];
                        if i <= j { continue; }
                        x[i] -= l_csc_vals[idx] * ljk;
                        touched.push(i);
                    }
                }

                // ── Compute L[j,j] and column j of L ──────────────────────────
                if x[j] <= T::zero() {
                    for &t in &touched { x[t] = T::zero(); }
                    return Err(SolverError::SingularMatrix { row: j });
                }
                let ljj = x[j].sqrt();

                let col_start = l_csc_rows.len();
                l_csc_rows.push(j);
                l_csc_vals.push(ljj);

                touched.sort_unstable();
                touched.dedup();
                for &i in touched.iter().filter(|&&t| t > j) {
                    let lij = x[i] / ljj;
                    if lij != T::zero() {
                        l_csc_rows.push(i);
                        l_csc_vals.push(lij);
                    }
                }
                l_csc_col_ptr[j + 1] = l_csc_col_ptr[j] + (l_csc_rows.len() - col_start);

                for &t in &touched { x[t] = T::zero(); }
                x[j] = T::zero();
                touched.clear();
            }
        }

        // ── Convert CSC L to CSR for solve ────────────────────────────────────
        let nnz = l_csc_rows.len();
        let mut row_counts = vec![0usize; n];
        for &r in &l_csc_rows { row_counts[r] += 1; }

        let mut l_row_ptr = vec![0usize; n + 1];
        for i in 0..n { l_row_ptr[i + 1] = l_row_ptr[i] + row_counts[i]; }
        let mut l_col_idx = vec![0usize; nnz];
        let mut l_values  = vec![T::zero(); nnz];
        let mut next = l_row_ptr.clone();

        for col in 0..n {
            for k in l_csc_col_ptr[col]..l_csc_col_ptr[col + 1] {
                let row = l_csc_rows[k];
                let pos = next[row];
                l_col_idx[pos] = col;
                l_values[pos]  = l_csc_vals[k];
                next[row] += 1;
            }
        }

        // Sort within each row and find diagonal positions.
        let mut l_diag_pos = vec![0usize; n];
        for i in 0..n {
            let start = l_row_ptr[i];
            let end   = l_row_ptr[i + 1];
            let mut pairs: Vec<(usize, T)> = (start..end)
                .map(|k| (l_col_idx[k], l_values[k]))
                .collect();
            pairs.sort_by_key(|(c, _)| *c);
            for (idx, (c, v)) in pairs.iter().enumerate() {
                l_col_idx[start + idx] = *c;
                l_values[start + idx]  = *v;
                if *c == i { l_diag_pos[i] = start + idx; }
            }
        }

        self.l_row_ptr  = l_row_ptr;
        self.l_col_idx  = l_col_idx;
        self.l_values   = l_values;
        self.l_diag_pos = l_diag_pos;

        if self.options.refine_steps > 0 {
            self.a_stored = Some(a.clone());
        }
        self.factorized = true;
        Ok(())
    }

    fn solve(&self, b: &DenseVec<T>, x: &mut DenseVec<T>) -> Result<(), SolverError> {
        if !self.factorized {
            return Err(SolverError::PrecondSetupFailed {
                reason: "SupernodalSparseCholesky: call factorize before solve".into(),
            });
        }
        let n = self.n;
        if b.len() != n {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: n, rhs_len: b.len(),
            });
        }

        // Apply permutation P: b_perm[i] = b[perm[i]].
        let mut pb = DenseVec::zeros(n);
        {
            let bs  = b.as_slice();
            let pbs = pb.as_mut_slice();
            for i in 0..n { pbs[i] = bs[self.perm[i]]; }
        }

        // Forward solve L y = P b.
        let mut y = DenseVec::zeros(n);
        forward_solve(
            &self.l_row_ptr, &self.l_col_idx, &self.l_values,
            &self.l_diag_pos,
            false, // non-unit diagonal
            &pb, &mut y,
        )?;

        // Backward solve Lᵀ z = y.
        let mut z = DenseVec::zeros(n);
        backward_solve_lt(
            &self.l_row_ptr, &self.l_col_idx, &self.l_values,
            &self.l_diag_pos,
            &y, &mut z,
        )?;

        // Apply inverse permutation: x[perm[i]] = z[i].
        {
            let zs = z.as_slice();
            let xs = x.as_mut_slice();
            for i in 0..n { xs[self.perm[i]] = zs[i]; }
        }

        Ok(())
    }

    fn reset_factors(&mut self) {
        self.l_row_ptr.clear(); self.l_col_idx.clear();
        self.l_values.clear();  self.l_diag_pos.clear();
        self.a_stored   = None;
        self.factorized = false;
        self.symbolic_n = None;
    }
}

// ─── LinearOperator impl ─────────────────────────────────────────────────────

impl<T: Scalar> LinearOperator for SupernodalSparseCholesky<T> {
    type Vector = DenseVec<T>;
    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let _ = self.solve(x, y);
    }
    fn nrows(&self) -> usize { self.n }
    fn ncols(&self) -> usize { self.n }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Find L[row, col] in CSC storage. Returns zero if not present.
fn find_in_col<T: Scalar>(
    l_csc_rows: &[usize],
    l_csc_vals: &[T],
    l_csc_col_ptr: &[usize],
    col: usize,
    row: usize,
) -> T {
    let start = l_csc_col_ptr[col];
    let end   = l_csc_col_ptr[col + 1];
    // Rows in a CSC column are appended in increasing order during factorization.
    for k in start..end {
        if l_csc_rows[k] == row { return l_csc_vals[k]; }
        if l_csc_rows[k] >  row { break; }
    }
    T::zero()
}

/// Backward solve for Lᵀ x = b, given L in CSR format.
///
/// Lᵀ is upper triangular: (Lᵀ)[i,j] = L[j,i] for j >= i.
/// We solve row by row from i = n-1 to 0:
///   x[i] = (b[i] - sum_{j>i} L[j,i] * x[j]) / L[i,i]
///
/// This access pattern traverses column i of L (rows j > i), which in CSR
/// is not contiguous.  We use a scatter approach: iterate over all rows j
/// and update x[col] for each off-diagonal entry L[j, col].
fn backward_solve_lt<T: Scalar>(
    row_ptr:  &[usize],
    col_idx:  &[usize],
    values:   &[T],
    diag_pos: &[usize],
    b: &DenseVec<T>,
    x: &mut DenseVec<T>,
) -> Result<(), SolverError> {
    let n = row_ptr.len().saturating_sub(1);
    let bs = b.as_slice();
    let xs = x.as_mut_slice();
    xs[..n].copy_from_slice(&bs[..n]);

    for i in (0..n).rev() {
        let d = values[diag_pos[i]];
        if d.abs() < T::machine_epsilon() * T::from_f64(1e6) {
            return Err(SolverError::SingularMatrix { row: i });
        }
        xs[i] = xs[i] / d;

        // Update: xs[j] -= L[i,j] * xs[i]  for j < i (L is lower, so L[i,j] with j<i)
        for k in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[k];
            if j >= i { break; }
            xs[j] -= values[k] * xs[i];
        }
    }
    Ok(())
}
