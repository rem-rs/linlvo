//! Sparse Cholesky factorisation for symmetric positive definite (SPD) matrices.
//!
//! Computes `P A Pᵀ = L Lᵀ` where:
//! - `P` is a fill-reducing row/column permutation
//! - `L` is lower triangular with positive diagonal
//!
//! Only the lower-triangular part of `A` is read (including diagonal).
//!
//! ## Algorithm
//!
//! Left-looking dense Cholesky, column by column:
//!   `L[j,j] = sqrt(A[j,j] − Σ_{k<j} L[j,k]²)`
//!   `L[i,j] = (A[i,j] − Σ_{k<j} L[i,k] L[j,k]) / L[j,j]`  for i > j
//!
//! ## Reference
//!
//! Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems.* SIAM.

use crate::core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}};
use crate::sparse::CsrMatrix;
use crate::direct::{
    DirectSolver, DirectOptions,
    ordering::{OrderingMethod, permute_symmetric, invert_perm, rcm, colamd},
    triangular::forward_solve,
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

    factorized: bool,
    analyzed:   bool,
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
            factorized: false, analyzed: false,
        }
    }
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
        self.n = n;
        self.perm = match &self.options.ordering {
            OrderingMethod::Natural => (0..n).collect(),
            OrderingMethod::Rcm    => rcm(a),
            OrderingMethod::Colamd => colamd(a),
        };
        self.inv_perm = invert_perm(&self.perm);
        self.analyzed   = true;
        self.factorized = false;
        Ok(())
    }

    fn factorize(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        if !self.analyzed { self.analyze(a)?; }
        let n = self.n;

        // Apply symmetric permutation.
        let b = permute_symmetric(a, &self.perm);

        // Dense n×n lower-triangular factorisation.
        // Store only lower triangle (column-major for cache efficiency on column access).
        // We use a flat row-major array: mat[i*n+j], i >= j.
        let mut l: Vec<T> = vec![T::zero(); n * n];

        // Scatter lower-triangular part of B into l.
        for i in 0..n {
            for k in b.row_ptr()[i]..b.row_ptr()[i + 1] {
                let j = b.col_idx()[k];
                if j <= i {
                    l[i * n + j] = b.values()[k];
                }
            }
        }

        // Left-looking Cholesky column by column.
        for j in 0..n {
            // Update l[j, j]: subtract sum of l[j, k]^2 for k < j.
            for k in 0..j {
                let ljk = l[j * n + k];
                l[j * n + j] -= ljk * ljk;
            }
            if l[j * n + j] <= T::zero() {
                return Err(SolverError::SingularMatrix { row: j });
            }
            let ljj = l[j * n + j].sqrt();
            l[j * n + j] = ljj;

            // Update l[i, j] for i > j.
            for i in (j + 1)..n {
                for k in 0..j {
                    let lik = l[i * n + k];
                    let ljk = l[j * n + k];
                    l[i * n + j] -= lik * ljk;
                }
                l[i * n + j] = l[i * n + j] / ljj;
            }
        }

        // Extract sparse L from the dense array (non-zeros only).
        let mut l_coo: Vec<(usize, usize, T)> = Vec::new();
        for i in 0..n {
            for j in 0..=i {
                let v = l[i * n + j];
                if v != T::zero() {
                    l_coo.push((i, j, v));
                }
            }
        }

        let (rp, ci, v, dp) = coo_to_csr_lower(n, &l_coo);
        self.l_row_ptr  = rp;
        self.l_col_idx  = ci;
        self.l_values   = v;
        self.l_diag_pos = dp;
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

        Ok(())
    }

    fn reset_factors(&mut self) {
        self.l_row_ptr.clear(); self.l_col_idx.clear();
        self.l_values.clear();  self.l_diag_pos.clear();
        self.factorized = false;
    }
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
        xs[i] = xs[i] / l_ii;
        let xi = xs[i];
        // Subtract L[i, j] * xi from xs[j] for each j < i (column j of Lᵀ row i).
        for k in l_row_ptr[i]..l_diag_pos[i] {
            let j = l_col_idx[k];
            xs[j] -= l_values[k] * xi;
        }
    }
    Ok(())
}

// ─── COO → CSR helper ────────────────────────────────────────────────────────

fn coo_to_csr_lower<T: Scalar>(
    n: usize,
    coo: &[(usize, usize, T)],
) -> (Vec<usize>, Vec<usize>, Vec<T>, Vec<usize>) {
    let mut sorted = coo.to_vec();
    sorted.sort_unstable_by_key(|&(r, c, _)| (r, c));

    let mut row_ptr  = vec![0usize; n + 1];
    let mut col_idx  = Vec::with_capacity(coo.len());
    let mut values   = Vec::with_capacity(coo.len());
    let mut diag_pos = vec![0usize; n];

    for &(r, _, _) in &sorted { row_ptr[r + 1] += 1; }
    for i in 0..n { row_ptr[i + 1] += row_ptr[i]; }
    for &(_, c, v) in &sorted { col_idx.push(c); values.push(v); }

    for i in 0..n {
        for k in row_ptr[i]..row_ptr[i + 1] {
            if col_idx[k] == i { diag_pos[i] = k; break; }
        }
    }

    (row_ptr, col_idx, values, diag_pos)
}
