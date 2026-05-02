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
//! Sparse right-looking Gaussian elimination with partial pivoting.
//! The working matrix is maintained as `n` sparse rows (sorted `Vec<(col, val)>`).
//! Each elimination step updates only the non-zero entries, giving O(nnz(L+U))
//! memory throughout — no O(n²) dense matrix is allocated.
//!
//! Memory: O(nnz(L) + nnz(U)).  Practical limit: n ≈ 10^6 for sparse matrices.
//! For very large fill or dense problems, use [`MultifrontalLu`](super::multifrontal::MultifrontalLu).

#![allow(clippy::needless_range_loop)]
use crate::core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}, operator::LinearOperator};
use crate::sparse::CsrMatrix;
use crate::direct::{
    DirectSolver, DirectOptions,
    ordering::{OrderingMethod, permute_symmetric, rcm, colamd, nd},
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

        // ── Sparse right-looking GE with partial pivoting (O(nnz) memory) ───────
        //
        // The working matrix is stored as n sparse rows (sorted Vec<(col, val)>).
        // Each pivot step:
        //   1. Find pivot row (max |entry at col j| in rows j..n-1).
        //   2. Swap rows.
        //   3. For each row i > j with a non-zero at col j:
        //      mult = row_i[j] / row_j[j];  store mult in row_i[j];
        //      row_i[k] -= mult * row_j[k] for k > j  (sparse axpy).
        //
        // Memory: O(nnz(L) + nnz(U)) throughout — no n×n allocation.

        // Scatter B into sparse rows.
        let mut rows: Vec<Vec<(usize, T)>> = (0..n).map(|_| Vec::new()).collect();
        for i in 0..n {
            for k in b.row_ptr()[i]..b.row_ptr()[i + 1] {
                rows[i].push((b.col_idx()[k], b.values()[k]));
            }
            rows[i].sort_unstable_by_key(|&(c, _)| c);
        }

        let mut row_perm: Vec<usize> = (0..n).collect();
        let thresh = self.options.pivot_threshold;

        for j in 0..n {
            // ── Partial pivot: find row with largest |entry at col j| in j..n ───
            let pivot_pos = find_pivot_sparse(&rows, n, j, thresh);
            if pivot_pos != j {
                rows.swap(j, pivot_pos);
                row_perm.swap(j, pivot_pos);
            }

            let u_jj = get_entry(&rows[j], j).unwrap_or(T::zero());
            if u_jj.abs() < T::machine_epsilon() * T::from_f64(1e6) {
                return Err(SolverError::NumericalBreakdown {
                    detail: format!(
                        "SparseLu: pivot too small at step {} (|u_jj|={:.3e}); matrix may be singular/ill-conditioned, try NodeNd/Colamd ordering, pivot threshold tuning, or iterative fallback",
                        j,
                        num_traits::ToPrimitive::to_f64(&u_jj.abs()).unwrap_or(f64::INFINITY),
                    ),
                });
            }

            // ── Update rows below the pivot ──────────────────────────────────────
            // Snapshot the pivot row's upper part (cols > j) for the axpy.
            let pivot_upper: Vec<(usize, T)> = rows[j]
                .iter()
                .filter(|&&(c, _)| c > j)
                .cloned()
                .collect();

            for i in (j + 1)..n {
                if let Some(mult) = get_entry(&rows[i], j) {
                    let mult = mult / u_jj;
                    set_entry(&mut rows[i], j, mult); // in-place: store L multiplier
                    sparse_axpy(&mut rows[i], &pivot_upper, -mult, j);
                }
            }
        }

        // ── Extract sparse L and U from factored rows ────────────────────────────
        let mut l_coo: Vec<(usize, usize, T)> = Vec::new();
        let mut u_coo: Vec<(usize, usize, T)> = Vec::new();

        for i in 0..n {
            for &(j, v) in &rows[i] {
                if v != T::zero() {
                    if j < i { l_coo.push((i, j, v)); }
                    else      { u_coo.push((i, j, v)); }
                }
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

        if x.as_slice().iter().any(|v| !v.is_finite()) {
            return Err(SolverError::NumericalBreakdown {
                detail: "SparseLu: non-finite solution entries detected after triangular solves; check conditioning/scaling and pivot strategy".into(),
            });
        }

        // Step 5: iterative refinement — x_{k+1} = x_k + A^{-1}(b - A x_k)
        if self.options.refine_steps > 0 {
            if let Some(ref a) = self.a_stored {
                let mut r = DenseVec::zeros(n);
                let mut pb = DenseVec::zeros(n);
                let mut dy = DenseVec::zeros(n);
                let mut dz = DenseVec::zeros(n);
                for _ in 0..self.options.refine_steps {
                    // Compute residual r = b - A x.
                    a.apply(x, &mut r);
                    {
                        let rs = r.as_mut_slice();
                        let bs = b.as_slice();
                        for i in 0..n { rs[i] = bs[i] - rs[i]; }
                    }

                    // Solve A δx = r (reuse stored factors).
                    {
                        let rs  = r.as_slice();
                        let pbs = pb.as_mut_slice();
                        for j in 0..n { pbs[j] = rs[self.perm_p[j]]; }
                    }
                    forward_solve(
                        &self.l_row_ptr, &self.l_col_idx, &self.l_values,
                        &self.l_diag_pos, true, &pb, &mut dy,
                    )?;
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

                    if x.as_slice().iter().any(|v| !v.is_finite()) {
                        return Err(SolverError::NumericalBreakdown {
                            detail: "SparseLu: non-finite values during iterative refinement; disable refine_steps or improve scaling".into(),
                        });
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

// ─── Sparse row helpers ───────────────────────────────────────────────────────

/// Get the value at column `col` from a sorted sparse row, or None.
fn get_entry<T: Scalar>(row: &[(usize, T)], col: usize) -> Option<T> {
    row.binary_search_by_key(&col, |&(c, _)| c)
        .ok()
        .map(|idx| row[idx].1)
}

/// Set the value at column `col` in a sorted sparse row (must already exist).
fn set_entry<T: Scalar>(row: &mut [(usize, T)], col: usize, val: T) {
    if let Ok(idx) = row.binary_search_by_key(&col, |&(c, _)| c) {
        row[idx].1 = val;
    }
}

/// Sparse axpy: row += scale * pivot_upper, only for cols > skip_below.
/// Merges new entries from pivot_upper into the sorted sparse row.
fn sparse_axpy<T: Scalar>(
    row: &mut Vec<(usize, T)>,
    pivot_upper: &[(usize, T)],
    scale: T,
    skip_below: usize,
) {
    if scale == T::zero() || pivot_upper.is_empty() { return; }
    // Merge pivot_upper entries into row (both sorted by col).
    // Collect positions for update-in-place.
    let mut inserts: Vec<(usize, T)> = Vec::new();
    for &(c, pv) in pivot_upper {
        if c <= skip_below { continue; }
        let delta = scale * pv;
        match row.binary_search_by_key(&c, |&(cc, _)| cc) {
            Ok(idx) => row[idx].1 += delta,
            Err(_)  => inserts.push((c, delta)),
        }
    }
    if !inserts.is_empty() {
        row.extend_from_slice(&inserts);
        row.sort_unstable_by_key(|&(c, _)| c);
    }
}

/// Find pivot row: row with max |entry at col j| among rows[j..n].
/// Respects threshold pivoting: if diagonal is large enough, prefer it.
fn find_pivot_sparse<T: Scalar>(
    rows: &[Vec<(usize, T)>],
    n: usize,
    j: usize,
    threshold: f64,
) -> usize {
    let mut best   = j;
    let mut best_v = get_entry(&rows[j], j).unwrap_or(T::zero()).abs();
    for i in (j + 1)..n {
        let v = get_entry(&rows[i], j).unwrap_or(T::zero()).abs();
        if v > best_v { best_v = v; best = i; }
    }
    if threshold < 1.0 - 1e-12 {
        let diag_v = get_entry(&rows[j], j).unwrap_or(T::zero()).abs();
        let thresh = T::from_f64(threshold) * best_v;
        if diag_v >= thresh { return j; }
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
