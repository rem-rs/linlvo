//! Sparse LU factorisation with column-reordering and partial pivoting.
//!
//! Computes `P A Q = L U` where:
//! - `Q` is a fill-reducing column permutation (ordering heuristic)
//! - `P` is a row permutation from partial pivoting (numerical stability)
//! - `L` is unit lower triangular
//! - `U` is upper triangular
//!
//! ## Algorithm
//!
//! For the n×n system, we factorise the reordered matrix `B = A Q` column by
//! column.  At column `j` we:
//! 1. Scatter column `j` of `B` into a dense working vector `x`.
//! 2. Apply the previously computed L columns: for `k < j`, subtract
//!    `l[j,k] * u[k, k:]` from `x` (equivalent to forward-substituting
//!    with the L factor constructed so far).
//! 3. Choose the pivot row (maximum magnitude in `x[j:]`).
//! 4. Store `u[j, j:] = x[j:]` (row j of U).
//! 5. Store `l[i, j] = x[i] / u[j,j]` for `i > j` (column j of L).
//!
//! This is a right-looking, dense-working-vector scheme; it is correct but
//! O(n * fill) rather than the O(fill) of full Gilbert-Peierls.  For the
//! problem sizes targeted by this module (up to ~10^4 DOF) it is fast enough
//! and much easier to implement correctly.

use crate::core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}};
use crate::sparse::CsrMatrix;
use crate::direct::{
    DirectSolver, DirectOptions,
    ordering::{OrderingMethod, permute_symmetric, invert_perm, rcm, colamd},
    triangular::{forward_solve, backward_solve},
};

// ─── Public struct ────────────────────────────────────────────────────────────

/// Sparse LU factorisation solver.
///
/// Implements the three-phase [`DirectSolver`] interface.
///
/// # Example
/// ```
/// use linger::direct::{SparseLu, DirectSolver};
/// use linger::sparse::{CooMatrix, CsrMatrix};
/// use linger::{DenseVec};
///
/// let mut coo = CooMatrix::<f64>::new(3, 3);
/// coo.push(0, 0, 4.0); coo.push(0, 1, 1.0);
/// coo.push(1, 0, 2.0); coo.push(1, 1, 3.0); coo.push(1, 2, 1.0);
/// coo.push(2, 1, 1.0); coo.push(2, 2, 5.0);
/// let a = CsrMatrix::from_coo(&coo);
///
/// let b = DenseVec::from_vec(vec![5.0, 10.0, 6.0]);
/// let mut x = DenseVec::zeros(3);
///
/// let mut solver = SparseLu::<f64>::default();
/// solver.factor(&a).unwrap();
/// solver.solve(&b, &mut x).unwrap();
/// ```
pub struct SparseLu<T: Scalar> {
    options: DirectOptions,

    n: usize,
    /// Column permutation: perm_q[new_col] = old_col.
    perm_q: Vec<usize>,

    // Numeric factors stored in separate CSR structures.
    /// L factor (unit lower-triangular, diagonal implicit = 1).
    l_row_ptr:  Vec<usize>,
    l_col_idx:  Vec<usize>,
    l_values:   Vec<T>,
    l_diag_pos: Vec<usize>,

    /// U factor (upper-triangular, diagonal stored).
    u_row_ptr:  Vec<usize>,
    u_col_idx:  Vec<usize>,
    u_values:   Vec<T>,
    u_diag_pos: Vec<usize>,

    /// Row permutation from partial pivoting: perm_p[step] = original_row.
    perm_p: Vec<usize>,

    factorized: bool,
    analyzed:   bool,
}

impl<T: Scalar> Default for SparseLu<T> {
    fn default() -> Self { Self::new(DirectOptions::default()) }
}

impl<T: Scalar> SparseLu<T> {
    pub fn new(options: DirectOptions) -> Self {
        Self {
            options, n: 0,
            perm_q: vec![],
            l_row_ptr: vec![], l_col_idx: vec![], l_values: vec![], l_diag_pos: vec![],
            u_row_ptr: vec![], u_col_idx: vec![], u_values: vec![], u_diag_pos: vec![],
            perm_p: vec![],
            factorized: false, analyzed: false,
        }
    }
}

// ─── DirectSolver impl ───────────────────────────────────────────────────────

impl<T: Scalar> DirectSolver<T> for SparseLu<T> {
    fn analyze(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: a.ncols(), rhs_len: n,
            });
        }
        self.n = n;
        self.perm_q = match &self.options.ordering {
            OrderingMethod::Natural => (0..n).collect(),
            OrderingMethod::Rcm    => rcm(a),
            OrderingMethod::Colamd => colamd(a),
        };
        self.analyzed   = true;
        self.factorized = false;
        Ok(())
    }

    fn factorize(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        if !self.analyzed { self.analyze(a)?; }
        let n = self.n;

        // Apply symmetric permutation B = P_q A P_q^T so that the reordered
        // matrix has better fill characteristics.
        let b = permute_symmetric(a, &self.perm_q);

        // Dense n×n working matrix — right-looking LU on B.
        // We store the entire dense matrix in row-major order and factorise
        // it in place, then extract the sparse L and U at the end.
        // This is O(n²) memory but correct and simple; for n ≤ 10^4 this is
        // at most 800 MB — acceptable for the target problem size range.
        // For larger n, use the SuperLU FFI backend (Sprint 14).
        let mut mat: Vec<T> = vec![T::zero(); n * n];
        let mut pivot_row = vec![0usize; n]; // pivot_row[j] = row chosen at step j

        // Scatter B into the dense matrix.
        for i in 0..n {
            for k in b.row_ptr()[i]..b.row_ptr()[i + 1] {
                let j = b.col_idx()[k];
                mat[i * n + j] = b.values()[k];
            }
        }

        // Row permutation tracking.
        let mut row_perm: Vec<usize> = (0..n).collect(); // row_perm[pos] = original_row
        let mut row_pos:  Vec<usize> = (0..n).collect(); // row_pos[orig] = current_pos

        let thresh = self.options.pivot_threshold;

        for j in 0..n {
            // Find pivot in column j, rows j..n (in the permuted ordering).
            let pivot_pos = find_pivot_col(&mat, n, j, thresh);
            pivot_row[j] = row_perm[pivot_pos];

            if pivot_pos != j {
                // Swap rows j and pivot_pos in mat.
                for k in 0..n {
                    mat.swap(j * n + k, pivot_pos * n + k);
                }
                row_perm.swap(j, pivot_pos);
                row_pos[row_perm[j]]         = j;
                row_pos[row_perm[pivot_pos]] = pivot_pos;
            }

            let u_jj = mat[j * n + j];
            if u_jj.abs() < T::machine_epsilon() * T::from_f64(1e6) {
                return Err(SolverError::SingularMatrix { row: j });
            }

            // Compute multipliers and update submatrix.
            for i in (j + 1)..n {
                let mult = mat[i * n + j] / u_jj;
                mat[i * n + j] = mult; // store L[i,j] in place
                for k in (j + 1)..n {
                    let uval = mat[j * n + k];
                    mat[i * n + k] -= mult * uval;
                }
            }
        }

        // Extract L and U from the dense factored matrix.
        let mut l_coo: Vec<(usize, usize, T)> = Vec::new();
        let mut u_coo: Vec<(usize, usize, T)> = Vec::new();

        for i in 0..n {
            for j in 0..=i {
                let v = mat[i * n + j];
                if j < i && v != T::zero() {
                    l_coo.push((i, j, v));
                }
                // diagonal of L is implicitly 1 — not stored
            }
            for j in i..n {
                let v = mat[i * n + j];
                if v != T::zero() {
                    u_coo.push((i, j, v));
                }
            }
        }

        // Build CSR factors.
        let (lrp, lci, lv, ldp) = coo_to_csr(n, &l_coo, true);
        let (urp, uci, uv, udp) = coo_to_csr(n, &u_coo, false);

        self.l_row_ptr  = lrp;
        self.l_col_idx  = lci;
        self.l_values   = lv;
        self.l_diag_pos = ldp;
        self.u_row_ptr  = urp;
        self.u_col_idx  = uci;
        self.u_values   = uv;
        self.u_diag_pos = udp;
        self.perm_p     = row_perm;
        self.factorized = true;
        Ok(())
    }

    fn solve(&self, b: &DenseVec<T>, x: &mut DenseVec<T>) -> Result<(), SolverError> {
        if !self.factorized {
            return Err(SolverError::PrecondSetupFailed {
                reason: "SparseLu: call factorize before solve".into(),
            });
        }
        let n = self.n;
        if b.len() != n {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: n, rhs_len: b.len(),
            });
        }

        // Step 1: apply row permutation P to b.
        // perm_p[j] = original row placed at step j.
        // The solve is: (PAQ)(Q⁻¹x) = Pb, i.e. LU z = Pb where x = Q z.
        let mut pb = DenseVec::zeros(n);
        {
            let bs  = b.as_slice();
            let pbs = pb.as_mut_slice();
            for j in 0..n {
                pbs[j] = bs[self.perm_p[j]];
            }
        }

        // Step 2: forward solve L y = Pb.
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

        // Step 4: apply inverse column permutation Q⁻¹ z → x.
        // perm_q[new] = old, so x[old] = z[new]  ↔  x[perm_q[i]] = z[i].
        {
            let zs = z.as_slice();
            let xs = x.as_mut_slice();
            for i in 0..n {
                xs[self.perm_q[i]] = zs[i];
            }
        }

        Ok(())
    }

    fn reset_factors(&mut self) {
        self.l_row_ptr.clear(); self.l_col_idx.clear(); self.l_values.clear();
        self.u_row_ptr.clear(); self.u_col_idx.clear(); self.u_values.clear();
        self.perm_p.clear();
        self.factorized = false;
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Find the pivot row index in column `j` of the dense matrix (rows `j..n`).
fn find_pivot_col<T: Scalar>(mat: &[T], n: usize, j: usize, threshold: f64) -> usize {
    let mut best   = j;
    let mut best_v = mat[j * n + j].abs();
    for i in (j + 1)..n {
        let v = mat[i * n + j].abs();
        if v > best_v { best_v = v; best = i; }
    }
    if threshold < 1.0 - 1e-12 {
        let thresh = T::from_f64(threshold) * best_v;
        if mat[j * n + j].abs() >= thresh { return j; }
    }
    best
}

/// Convert COO to CSR.  `lower = true` → lower-triangular, skip diagonal.
/// Returns `(row_ptr, col_idx, values, diag_pos)`.
fn coo_to_csr<T: Scalar>(
    n: usize,
    coo: &[(usize, usize, T)],
    lower: bool,
) -> (Vec<usize>, Vec<usize>, Vec<T>, Vec<usize>) {
    // Sort by (row, col).
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
        // Unit lower-triangular: diagonal is implicit; point diag_pos[i] to
        // the start of row i (used only as a sentinel).
        for i in 0..n { diag_pos[i] = row_ptr[i]; }
    } else {
        // Upper-triangular: locate diagonal.
        for i in 0..n {
            for k in row_ptr[i]..row_ptr[i + 1] {
                if col_idx[k] == i { diag_pos[i] = k; break; }
            }
        }
    }

    (row_ptr, col_idx, values, diag_pos)
}
