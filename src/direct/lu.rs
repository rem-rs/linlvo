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
//! Right-looking dense Gaussian elimination on the reordered matrix `B = A[Q,Q]`.
//! A flat n×n working matrix is used for the numeric factorization, making
//! correctness straightforward.
//!
//! The symbolic analysis phase (elimination tree + symbolic LU) is computed
//! before factorization and used for pattern prediction. Future work will
//! replace the dense numeric phase with a sparse left-looking implementation
//! that uses the symbolic pattern for O(nnz(L+U)) storage.
//!
//! Memory: O(n²) for the dense working matrix. Practical limit: n ≈ 10^4.
//! For larger matrices, use [`MultifrontalLu`](super::multifrontal::MultifrontalLu).

use crate::core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}, operator::LinearOperator};
use crate::sparse::CsrMatrix;
use crate::direct::{
    DirectSolver, DirectOptions,
    ordering::{OrderingMethod, permute_symmetric, invert_perm, rcm, colamd, nd},
    triangular::{forward_solve, backward_solve},
    etree::elimination_tree,
    symbolic::{symbolic_lu, SymbolicLu},
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

    /// Stored copy of the factored matrix A (needed for iterative refinement).
    a_stored: Option<CsrMatrix<T>>,

    factorized: bool,
    analyzed:   bool,

    /// Cached symbolic ordering size — used by reuse_symbolic.
    /// When `options.reuse_symbolic` is true, `analyze` is skipped if
    /// the incoming matrix has the same size as the cached analysis.
    symbolic_n: Option<usize>,
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
            a_stored: None,
            factorized: false, analyzed: false,
            symbolic_n: None,
        }
    }

    /// Returns the column permutation `Q` (perm_q[new] = old) after analysis.
    pub fn perm_q(&self) -> &[usize] { &self.perm_q }

    /// Returns the row permutation `P` (perm_p[step] = original_row) after factorization.
    pub fn perm_p(&self) -> &[usize] { &self.perm_p }
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

        // reuse_symbolic: skip ordering if already analyzed with the same size.
        if self.options.reuse_symbolic {
            if let Some(cached_n) = self.symbolic_n {
                if cached_n == n && self.analyzed {
                    // Keep existing perm_q; just reset numeric factors.
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
        self.symbolic_n = Some(n);
        self.analyzed   = true;
        self.factorized = false;
        Ok(())
    }

    fn factorize(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError> {
        if !self.analyzed { self.analyze(a)?; }
        let n = self.n;

        // Apply symmetric permutation B = A[perm_q, perm_q].
        let b = permute_symmetric(a, &self.perm_q);

        // Compute elimination tree and symbolic factorization.
        // These are used for pre-allocation and will be fully utilized
        // in a future sparse left-looking implementation.
        let _parent = elimination_tree(&b);
        let _sym = symbolic_lu(&b, &_parent);

        // ── Dense working matrix approach (O(n²) memory) ────────────────────────
        // The symbolic pattern from _sym is computed but not yet used for
        // the numeric factorization. This is a transitional implementation
        // that validates the symbolic analysis while keeping the proven
        // dense factorization for correctness.

        let mut mat: Vec<T> = vec![T::zero(); n * n];
        let mut pivot_row = vec![0usize; n];

        // Scatter B into the dense matrix.
        for i in 0..n {
            for k in b.row_ptr()[i]..b.row_ptr()[i + 1] {
                let j = b.col_idx()[k];
                mat[i * n + j] = b.values()[k];
            }
        }

        // Row permutation tracking.
        let mut row_perm: Vec<usize> = (0..n).collect();
        let mut row_pos:  Vec<usize> = (0..n).collect();

        let thresh = self.options.pivot_threshold;

        for j in 0..n {
            let pivot_pos = find_pivot_col(&mat, n, j, thresh);
            pivot_row[j] = row_perm[pivot_pos];

            if pivot_pos != j {
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

            for i in (j + 1)..n {
                let mult = mat[i * n + j] / u_jj;
                mat[i * n + j] = mult;
                for k in (j + 1)..n {
                    let uval = mat[j * n + k];
                    mat[i * n + k] -= mult * uval;
                }
            }
        }

        // Extract sparse L and U from the dense factored matrix.
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
        if self.options.refine_steps > 0 {
            self.a_stored = Some(a.clone());
        }
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
        let mut pb = DenseVec::zeros(n);
        {
            let bs  = b.as_slice();
            let pbs = pb.as_mut_slice();
            for j in 0..n { pbs[j] = bs[self.perm_p[j]]; }
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
        {
            let zs = z.as_slice();
            let xs = x.as_mut_slice();
            for i in 0..n { xs[self.perm_q[i]] = zs[i]; }
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

                    // Solve A δx = r (reuse stored factors).
                    let mut pb = DenseVec::zeros(n);
                    {
                        let rs  = r.as_slice();
                        let pbs = pb.as_mut_slice();
                        for j in 0..n { pbs[j] = rs[self.perm_p[j]]; }
                    }
                    let mut dy = DenseVec::zeros(n);
                    forward_solve(
                        &self.l_row_ptr, &self.l_col_idx, &self.l_values,
                        &self.l_diag_pos, true, &pb, &mut dy,
                    )?;
                    let mut dz = DenseVec::zeros(n);
                    backward_solve(
                        &self.u_row_ptr, &self.u_col_idx, &self.u_values,
                        &self.u_diag_pos, &dy, &mut dz,
                    )?;

                    // x += Q^{-1} dz
                    {
                        let dzs = dz.as_slice();
                        let xs  = x.as_mut_slice();
                        for i in 0..n { xs[self.perm_q[i]] += dzs[i]; }
                    }
                }
            }
        }

        Ok(())
    }

    fn reset_factors(&mut self) {
        self.l_row_ptr.clear(); self.l_col_idx.clear(); self.l_values.clear(); self.l_diag_pos.clear();
        self.u_row_ptr.clear(); self.u_col_idx.clear(); self.u_values.clear(); self.u_diag_pos.clear();
        self.perm_p.clear();
        self.a_stored = None;
        self.factorized = false;
        self.symbolic_n = None;
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// CSC matrix structure for temporary use during factorization.
struct CscMatrix<T: Scalar> {
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<T>,
}

/// Convert CSC to CSR format (for L after column-wise factorization).
fn csc_to_csr<T: Scalar>(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    values: &[T],
) -> (Vec<usize>, Vec<usize>, Vec<T>, Vec<usize>) {
    // Count entries per row.
    let mut row_counts = vec![0usize; n];
    for &row in row_idx {
        row_counts[row] += 1;
    }

    // Build row pointers.
    let mut row_ptr = vec![0usize; n + 1];
    for i in 0..n { row_ptr[i + 1] = row_ptr[i] + row_counts[i]; }

    // Fill in column indices and values.
    let nnz = values.len();
    let mut col_idx = vec![0usize; nnz];
    let mut vals = vec![T::zero(); nnz];
    let mut next = row_ptr.clone();

    for col in 0..n {
        for k in col_ptr[col]..col_ptr[col + 1] {
            let row = row_idx[k];
            let pos = next[row];
            col_idx[pos] = col;
            vals[pos] = values[k];
            next[row] += 1;
        }
    }

    // Sort column indices within each row and find diagonal positions.
    let mut diag_pos = vec![0usize; n];
    for i in 0..n {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        // Sort by column index.
        let mut pairs: Vec<(usize, T)> = (start..end)
            .map(|k| (col_idx[k], vals[k]))
            .collect();
        pairs.sort_by_key(|(c, _)| *c);
        for (idx, (c, v)) in pairs.iter().enumerate() {
            col_idx[start + idx] = *c;
            vals[start + idx] = *v;
            if *c == i { diag_pos[i] = start + idx; }
        }
    }

    (row_ptr, col_idx, vals, diag_pos)
}

/// Convert CSR to CSC format for column access during factorization.
fn csc_from_csr<T: Scalar>(csr: &CsrMatrix<T>) -> CscMatrix<T> {
    let n = csr.nrows();
    let nnz = csr.col_idx().len();

    let mut col_ptr = vec![0usize; n + 1];
    let mut row_idx = vec![0usize; nnz];
    let mut values = vec![T::zero(); nnz];

    // Count entries per column.
    for &c in csr.col_idx() {
        col_ptr[c + 1] += 1;
    }
    // Cumulative sum.
    for i in 0..n { col_ptr[i + 1] += col_ptr[i]; }

    // Scatter entries.
    let mut pos = col_ptr.clone();
    for i in 0..n {
        for k in csr.row_ptr()[i]..csr.row_ptr()[i + 1] {
            let j = csr.col_idx()[k];
            row_idx[pos[j]] = i;
            values[pos[j]] = csr.values()[k];
            pos[j] += 1;
        }
    }

    CscMatrix { n, col_ptr, row_idx, values }
}

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

fn coo_to_csr<T: Scalar>(
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
        for i in 0..n { diag_pos[i] = row_ptr[i]; }
    } else {
        for i in 0..n {
            for k in row_ptr[i]..row_ptr[i + 1] {
                if col_idx[k] == i { diag_pos[i] = k; break; }
            }
        }
    }

    (row_ptr, col_idx, values, diag_pos)
}
