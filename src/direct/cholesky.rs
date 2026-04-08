//! Sparse Cholesky factorisation for symmetric positive definite (SPD) matrices.
//!
//! Computes `P A Pᵀ = L Lᵀ` where:
//! - `P` is a fill-reducing row/column permutation
//! - `L` is lower triangular with positive diagonal
//!
//! Only the lower-triangular part of `A` is read (including diagonal).
//!
//! ## Algorithm: Left-looking sparse Cholesky
//!
//! Column-by-column, using the elimination tree to determine the non-zero
//! pattern of each column of L before computing values.  Memory is O(nnz(L)).
//!
//! For column j:
//! 1. Scatter A[j:n, j] into a dense working vector x.
//! 2. Find the reach set: columns k < j where L[j,k] != 0, via DFS on etree.
//! 3. For each k in the reach set (topological order):
//!    x[j] -= L[j,k]^2  and  x[i] -= L[i,k]*L[j,k] for i > j in col k of L.
//! 4. L[j,j] = sqrt(x[j]).  L[i,j] = x[i] / L[j,j].
//!
//! ## Reference
//!
//! Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems.* SIAM.

#![allow(clippy::needless_range_loop)]
use crate::core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}, operator::LinearOperator};
use crate::sparse::CsrMatrix;
use crate::direct::{
    DirectSolver, DirectOptions,
    ordering::{OrderingMethod, permute_symmetric, invert_perm, rcm, colamd, nd},
    triangular::forward_solve,
    etree::elimination_tree,
};

// ─── Public struct ────────────────────────────────────────────────────────────

/// Sparse Cholesky factorisation solver for SPD matrices.
///
/// Implements the three-phase [`DirectSolver`] interface.
pub struct SparseCholesky<T: Scalar> {
    options: DirectOptions,

    n: usize,
    perm:     Vec<usize>,   // perm[new] = old
    inv_perm: Vec<usize>,   // inv_perm[old] = new

    /// L factor in CSR (lower-triangular, including diagonal).
    l_row_ptr:  Vec<usize>,
    l_col_idx:  Vec<usize>,
    l_values:   Vec<T>,
    l_diag_pos: Vec<usize>,

    /// Stored copy of the factored matrix A (needed for iterative refinement).
    a_stored: Option<CsrMatrix<T>>,

    factorized: bool,
    analyzed:   bool,

    /// Cached symbolic ordering size — used by reuse_symbolic.
    symbolic_n: Option<usize>,
}

impl<T: Scalar> Default for SparseCholesky<T> {
    fn default() -> Self { Self::new(DirectOptions::default()) }
}

impl<T: Scalar> SparseCholesky<T> {
    pub fn new(options: DirectOptions) -> Self {
        Self {
            options, n: 0,
            perm: vec![], inv_perm: vec![],
            l_row_ptr: vec![], l_col_idx: vec![], l_values: vec![], l_diag_pos: vec![],
            a_stored: None,
            factorized: false, analyzed: false,
            symbolic_n: None,
        }
    }

    /// Returns the permutation `P` (perm[new] = old) after analysis.
    pub fn perm(&self) -> &[usize] { &self.perm }
}

// ─── DirectSolver impl ───────────────────────────────────────────────────────

impl<T: Scalar> DirectSolver<T> for SparseCholesky<T> {
    fn analyze(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: a.ncols(), rhs_len: n,
            });
        }

        // reuse_symbolic: skip ordering if already analyzed with the same size.
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
        self.inv_perm = invert_perm(&self.perm);
        self.symbolic_n = Some(n);
        self.analyzed   = true;
        self.factorized = false;
        Ok(())
    }

    fn factorize(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        if !self.analyzed { self.analyze(a)?; }
        let n = self.n;

        // Apply symmetric permutation.
        let b = permute_symmetric(a, &self.perm);

        // Compute elimination tree for the permuted matrix.
        let parent = elimination_tree(&b);

        // Build lower-triangular column access for B:
        // col_ptr_lo[j]..col_ptr_lo[j+1] → (row, value) for entries B[row, j] with row > j.
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
            let mut fill_pos = col_ptr_lo.clone();
            for i in 0..n {
                for k in b.row_ptr()[i]..b.row_ptr()[i + 1] {
                    let j = b.col_idx()[k];
                    if j < i {
                        let p = fill_pos[j];
                        col_row_lo[p] = i;
                        col_val_lo[p] = b.values()[k];
                        fill_pos[j] += 1;
                    }
                }
            }
        }

        // ── Left-looking sparse Cholesky ──────────────────────────────────────
        // L stored column-by-column in CSC during factorization.
        let mut l_csc_col_ptr: Vec<usize> = vec![0usize; n + 1];
        let mut l_csc_rows:    Vec<usize> = Vec::new();
        let mut l_csc_vals:    Vec<T>     = Vec::new();

        // Dense working vector (size n; only j..n used per column j).
        let mut x: Vec<T>           = vec![T::zero(); n];
        let mut touched: Vec<usize> = Vec::new();
        let mut mark = vec![usize::MAX; n]; // mark[k] = j → visited in column j

        for j in 0..n {
            // ── Scatter A[j:n, j] into x ─────────────────────────────────────
            // Diagonal entry B[j, j].
            for k in b.row_ptr()[j]..b.row_ptr()[j + 1] {
                if b.col_idx()[k] == j {
                    x[j] = b.values()[k];
                    touched.push(j);
                    break;
                }
            }
            // Sub-diagonal entries B[i, j] for i > j.
            for idx in col_ptr_lo[j]..col_ptr_lo[j + 1] {
                let i = col_row_lo[idx];
                x[i] = col_val_lo[idx];
                touched.push(i);
            }

            // ── Compute reach set via DFS from lower entries of row j ─────────
            // Lower entries of row j: B[j, k] with k < j.
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
                if p < j && mark[p] != j {
                    dfs_stack.push(p);
                }
            }
            reach.sort_unstable(); // topological order

            // ── Left-looking updates ──────────────────────────────────────────
            for &k in &reach {
                // Find L[j, k] in column k of L (CSC).
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

            // ── Compute L[j,j] and column j of L ─────────────────────────────
            if x[j] <= T::zero() {
                for &t in &touched { x[t] = T::zero(); }
                return Err(SolverError::SingularMatrix { row: j });
            }
            let ljj = x[j].sqrt();

            let col_start = l_csc_rows.len();
            l_csc_rows.push(j); // diagonal
            l_csc_vals.push(ljj);

            // Sub-diagonal: only indices that are in touched and > j.
            // Collect unique sub-diagonal touched entries.
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

            // Clear x.
            for &t in &touched { x[t] = T::zero(); }
            x[j] = T::zero();
            touched.clear();
        }

        // ── Convert L from CSC to CSR ─────────────────────────────────────────
        let nnz = l_csc_rows.len();
        let mut l_row_ptr  = vec![0usize; n + 1];
        let mut l_col_idx  = vec![0usize; nnz];
        let mut l_values   = vec![T::zero(); nnz];
        let mut l_diag_pos = vec![0usize; n];

        for &r in &l_csc_rows { l_row_ptr[r + 1] += 1; }
        for i in 0..n { l_row_ptr[i + 1] += l_row_ptr[i]; }

        let mut pos = l_row_ptr.clone();
        for j in 0..n {
            for k in l_csc_col_ptr[j]..l_csc_col_ptr[j + 1] {
                let r = l_csc_rows[k];
                let v = l_csc_vals[k];
                let p = pos[r];
                l_col_idx[p] = j;
                l_values[p]  = v;
                pos[r] += 1;
            }
        }

        for i in 0..n {
            for k in l_row_ptr[i]..l_row_ptr[i + 1] {
                if l_col_idx[k] == i { l_diag_pos[i] = k; break; }
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
                reason: "SparseCholesky: call factorize before solve".into(),
            });
        }
        let n = self.n;
        if b.len() != n {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: n, rhs_len: b.len(),
            });
        }

        // Step 1: apply permutation P to b.
        let mut pb = DenseVec::zeros(n);
        {
            let bs  = b.as_slice();
            let pbs = pb.as_mut_slice();
            for i in 0..n { pbs[i] = bs[self.perm[i]]; }
        }

        // Step 2: forward solve L y = Pb (L has non-unit diagonal).
        let mut y = DenseVec::zeros(n);
        forward_solve(
            &self.l_row_ptr, &self.l_col_idx, &self.l_values,
            &self.l_diag_pos,
            false, // non-unit diagonal
            &pb, &mut y,
        )?;

        // Step 3: backward solve Lᵀ z = y.
        let mut z = DenseVec::zeros(n);
        backward_solve_lt(
            n,
            &self.l_row_ptr, &self.l_col_idx, &self.l_values,
            &self.l_diag_pos,
            &y, &mut z,
        )?;

        // Step 4: apply inverse permutation Pᵀ to z → x.
        {
            let zs = z.as_slice();
            let xs = x.as_mut_slice();
            for i in 0..n { xs[self.perm[i]] = zs[i]; }
        }

        // Step 5: iterative refinement — x_{k+1} = x_k + A^{-1}(b - A x_k)
        if self.options.refine_steps > 0 {
            if let Some(ref a) = self.a_stored {
                for _ in 0..self.options.refine_steps {
                    // Compute residual r = b - A x.
                    let mut r = DenseVec::zeros(n);
                    a.apply(x, &mut r);
                    {
                        let rs = r.as_mut_slice();
                        let bs = b.as_slice();
                        for i in 0..n { rs[i] = bs[i] - rs[i]; }
                    }

                    // Apply permutation P to r.
                    let mut pr = DenseVec::zeros(n);
                    {
                        let rs  = r.as_slice();
                        let prs = pr.as_mut_slice();
                        for i in 0..n { prs[i] = rs[self.perm[i]]; }
                    }

                    // Forward solve L dy = Pr.
                    let mut dy = DenseVec::zeros(n);
                    forward_solve(
                        &self.l_row_ptr, &self.l_col_idx, &self.l_values,
                        &self.l_diag_pos, false, &pr, &mut dy,
                    )?;

                    // Backward solve Lᵀ dz = dy.
                    let mut dz = DenseVec::zeros(n);
                    backward_solve_lt(
                        n,
                        &self.l_row_ptr, &self.l_col_idx, &self.l_values,
                        &self.l_diag_pos, &dy, &mut dz,
                    )?;

                    // x += P^{-T} dz  (same as xs[perm[i]] += dz[i])
                    {
                        let dzs = dz.as_slice();
                        let xs  = x.as_mut_slice();
                        for i in 0..n { xs[self.perm[i]] += dzs[i]; }
                    }
                }
            }
        }

        Ok(())
    }

    fn reset_factors(&mut self) {
        self.l_row_ptr.clear(); self.l_col_idx.clear();
        self.l_values.clear();  self.l_diag_pos.clear();
        self.a_stored = None;
        self.factorized = false;
        self.symbolic_n = None;
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Find value at `row` in column `col` of a CSC matrix. Returns zero if absent.
#[inline]
fn find_in_col<T: Scalar>(
    rows: &[usize],
    vals: &[T],
    col_ptr: &[usize],
    col: usize,
    row: usize,
) -> T {
    for k in col_ptr[col]..col_ptr[col + 1] {
        if rows[k] == row { return vals[k]; }
    }
    T::zero()
}

// ─── Backward solve Lᵀ x = b ─────────────────────────────────────────────────

/// Solve `Lᵀ x = b` given lower-triangular `L` stored in CSR.
fn backward_solve_lt<T: Scalar>(
    n: usize,
    l_row_ptr:  &[usize],
    l_col_idx:  &[usize],
    l_values:   &[T],
    l_diag_pos: &[usize],
    b: &DenseVec<T>,
    x: &mut DenseVec<T>,
) -> Result<(), SolverError> {
    let bs = b.as_slice();
    let xs = x.as_mut_slice();
    xs[..n].copy_from_slice(&bs[..n]);

    for i in (0..n).rev() {
        let l_ii = l_values[l_diag_pos[i]];
        if l_ii.abs() < T::machine_epsilon() * T::from_f64(1e6) {
            return Err(SolverError::SingularMatrix { row: i });
        }
        xs[i] /= l_ii;
        let xi = xs[i];
        for k in l_row_ptr[i]..l_diag_pos[i] {
            let j = l_col_idx[k];
            xs[j] -= l_values[k] * xi;
        }
    }
    Ok(())
}
