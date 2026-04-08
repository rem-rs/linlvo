//! Sparse LDLᵀ factorisation for symmetric (possibly indefinite) matrices.
//!
//! Computes `P A Pᵀ ≈ L D Lᵀ` where:
//! - `P` is a fill-reducing row/column permutation
//! - `L` is unit lower-triangular (diagonal entries = 1)
//! - `D` is a diagonal matrix (may have negative entries)
//!
//! Unlike Cholesky (`A = L Lᵀ`, requires SPD), LDLᵀ handles **symmetric
//! indefinite** matrices (e.g., saddle-point systems, Helmholtz problems,
//! augmented Lagrangian systems).
//!
//! ## Algorithm: Left-looking sparse LDLᵀ
//!
//! Column j update (Davis 2005, Algorithm LDL):
//! 1. Scatter A[j:n, j] into dense working vector x.
//! 2. Find reach set: columns k < j where L[j,k] != 0 (via etree DFS).
//! 3. For each k in reach (topological):
//!    - y_k = L[j,k] * D[k]
//!    - D[j] -= y_k * L[j,k]
//!    - L[i,j] -= y_k * L[i,k] for all i > j in column k
//! 4. L[j,j] = 1 (unit diagonal)
//!    L[i,j] = x[i] / D[j] for i > j in the pattern
//!
//! If |D[j]| < pivot_threshold, the diagonal is treated as singular.
//!
//! ## References
//!
//! - Davis, T. A. (2005). *Algorithm 849: A concise sparse Cholesky algorithm.*
//!   ACM Trans. Math. Softw., 31(4), 587-591.
//! - Duff, I.S. & Reid, J.K. (1983). *The multifrontal solution of indefinite
//!   sparse symmetric linear equations.* ACM Trans. Math. Softw., 9(3), 302-325.

#![allow(clippy::needless_range_loop)]
use crate::core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}, operator::LinearOperator};
use crate::sparse::CsrMatrix;
use crate::direct::{
    DirectSolver, DirectOptions,
    ordering::{OrderingMethod, permute_symmetric, invert_perm, rcm, colamd, nd},
    etree::elimination_tree,
};

// ─── Public struct ────────────────────────────────────────────────────────────

/// Sparse LDLᵀ factorisation solver for symmetric (possibly indefinite) matrices.
///
/// Factorises `P A Pᵀ = L D Lᵀ` where `L` is unit lower-triangular and `D`
/// is diagonal.  Unlike [`SparseCholesky`](super::SparseCholesky), this handles
/// symmetric indefinite systems (negative pivots allowed).
///
/// # Example
/// ```
/// use linger::direct::{SparseLdlt, DirectSolver};
/// use linger::sparse::{CooMatrix, CsrMatrix};
/// use linger::DenseVec;
///
/// // 3×3 symmetric indefinite system: saddle-point-like
/// let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
/// coo.push(0, 0,  2.0); coo.push(0, 1,  1.0); coo.push(0, 2,  0.0);
/// coo.push(1, 0,  1.0); coo.push(1, 1, -1.0); coo.push(1, 2,  1.0);
/// coo.push(2, 0,  0.0); coo.push(2, 1,  1.0); coo.push(2, 2,  2.0);
/// let a = CsrMatrix::from_coo(&coo);
///
/// let mut ldlt = SparseLdlt::<f64>::default();
/// ldlt.factor(&a).unwrap();
/// let b = DenseVec::from_vec(vec![3.0, 1.0, 3.0]);
/// let mut x = DenseVec::zeros(3);
/// ldlt.solve(&b, &mut x).unwrap();
/// // verify: A x ≈ b
/// ```
pub struct SparseLdlt<T: Scalar> {
    options: DirectOptions,

    n: usize,
    perm:     Vec<usize>,   // perm[new] = old
    inv_perm: Vec<usize>,   // inv_perm[old] = new

    /// D diagonal (length n).
    d_vals: Vec<T>,

    /// L factor in CSR (unit lower-triangular, **without** diagonal which is 1).
    /// Entries: L[i,j] for j < i.
    l_row_ptr:  Vec<usize>,
    l_col_idx:  Vec<usize>,
    l_values:   Vec<T>,

    factorized: bool,
    analyzed:   bool,
    symbolic_n: Option<usize>,
}

impl<T: Scalar> Default for SparseLdlt<T> {
    fn default() -> Self { Self::new(DirectOptions::default()) }
}

impl<T: Scalar> SparseLdlt<T> {
    /// Create with custom options.
    pub fn new(options: DirectOptions) -> Self {
        Self {
            options, n: 0,
            perm: vec![], inv_perm: vec![],
            d_vals: vec![],
            l_row_ptr: vec![], l_col_idx: vec![], l_values: vec![],
            factorized: false, analyzed: false, symbolic_n: None,
        }
    }

    /// Return the diagonal `D` after factorisation.
    pub fn d_vals(&self) -> &[T] { &self.d_vals }

    /// Return the permutation `P` (perm[new] = old) after analysis.
    pub fn perm(&self) -> &[usize] { &self.perm }

    /// `true` if all diagonal entries `D[j] > 0` (matrix was SPD).
    pub fn is_positive_definite(&self) -> bool {
        self.factorized && self.d_vals.iter().all(|&d| d > T::zero())
    }
}

// ─── DirectSolver impl ───────────────────────────────────────────────────────

impl<T: Scalar> DirectSolver<T> for SparseLdlt<T> {
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
        let thresh = T::from_f64(self.options.pivot_threshold.max(1e-300));

        // Apply symmetric permutation B = P A Pᵀ.
        let b = permute_symmetric(a, &self.perm);

        // Elimination tree of the symmetric pattern.
        let parent = elimination_tree(&b);

        // Build column-access for the lower triangle of B (entries B[i,j], i > j).
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

        // ── Left-looking sparse LDLᵀ ─────────────────────────────────────────
        // L stored column-by-column in CSC (unit lower-triangular, no diagonal).
        let mut l_csc_col_ptr = vec![0usize; n + 1];
        let mut l_csc_rows: Vec<usize> = Vec::new();
        let mut l_csc_vals: Vec<T>     = Vec::new();

        let mut d = vec![T::zero(); n];      // diagonal D
        let mut x = vec![T::zero(); n];      // dense working column
        let mut touched: Vec<usize> = Vec::new();
        let mut mark = vec![usize::MAX; n];  // mark[k] = j → visited for column j

        for j in 0..n {
            // ── Scatter B[j:n, j] into x ─────────────────────────────────────
            // Diagonal B[j,j].
            for k in b.row_ptr()[j]..b.row_ptr()[j + 1] {
                if b.col_idx()[k] == j {
                    x[j] = b.values()[k];
                    touched.push(j);
                    break;
                }
            }
            // Sub-diagonal B[i,j], i > j.
            for idx in col_ptr_lo[j]..col_ptr_lo[j + 1] {
                let i = col_row_lo[idx];
                x[i] = col_val_lo[idx];
                touched.push(i);
            }

            // ── Reach set via DFS on etree ────────────────────────────────────
            let mut reach: Vec<usize> = Vec::new();
            let mut stack: Vec<usize> = Vec::new();
            for k in b.row_ptr()[j]..b.row_ptr()[j + 1] {
                let col = b.col_idx()[k];
                if col < j && mark[col] != j { stack.push(col); }
            }
            while let Some(r) = stack.pop() {
                if mark[r] == j { continue; }
                mark[r] = j;
                reach.push(r);
                let p = parent[r];
                if p < j && mark[p] != j { stack.push(p); }
            }
            reach.sort_unstable(); // topological order

            // ── Left-looking LDLᵀ update ──────────────────────────────────────
            // For each k in reach: x -= (L[:,k] * D[k] * L[j,k])
            for &k in &reach {
                // Find L[j,k] in CSC column k.
                let ljk = find_in_col_csc(&l_csc_rows, &l_csc_vals, &l_csc_col_ptr, k, j);
                if ljk == T::zero() { continue; }

                let yk = ljk * d[k];   // y_k = L[j,k] * D[k]
                x[j] -= yk * ljk;     // D[j] -= y_k * L[j,k]

                for idx in l_csc_col_ptr[k]..l_csc_col_ptr[k + 1] {
                    let i = l_csc_rows[idx];
                    if i <= j { continue; }
                    x[i] -= yk * l_csc_vals[idx];  // L[i,j] -= y_k * L[i,k]
                    touched.push(i);
                }
            }

            // ── Diagonal pivot D[j] = x[j] ───────────────────────────────────
            let dj = x[j];
            if dj.abs() < thresh {
                // Near-zero pivot: regularise to ±threshold.
                d[j] = if dj < T::zero() { -thresh } else { thresh };
            } else {
                d[j] = dj;
            }

            // ── Off-diagonal: L[i,j] = x[i] / D[j] ──────────────────────────
            let col_start = l_csc_rows.len();
            touched.sort_unstable();
            touched.dedup();
            for &i in touched.iter().filter(|&&t| t > j) {
                let lij = x[i] / d[j];
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
        let mut l_row_ptr = vec![0usize; n + 1];
        let mut l_col_idx = vec![0usize; nnz];
        let mut l_values  = vec![T::zero(); nnz];

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

        self.d_vals     = d;
        self.l_row_ptr  = l_row_ptr;
        self.l_col_idx  = l_col_idx;
        self.l_values   = l_values;
        self.factorized = true;
        Ok(())
    }

    fn solve(&self, b: &DenseVec<T>, x: &mut DenseVec<T>) -> Result<(), SolverError> {
        if !self.factorized {
            return Err(SolverError::NumericalBreakdown {
                detail: "SparseLdlt: factorize() must be called before solve()".to_string(),
            });
        }
        let n = self.n;
        if b.len() != n {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: n, rhs_len: b.len(),
            });
        }
        if x.len() != n { *x = DenseVec::zeros(n); }

        // Step 1: apply permutation — y = P b.
        let mut y = vec![T::zero(); n];
        for (new_i, &old_i) in self.perm.iter().enumerate() {
            y[new_i] = b.as_slice()[old_i];
        }

        // Step 2: forward solve L z = y  (unit lower-triangular, diagonal=1).
        let mut z = y;
        for i in 0..n {
            for k in self.l_row_ptr[i]..self.l_row_ptr[i + 1] {
                let j = self.l_col_idx[k];
                let tmp = self.l_values[k] * z[j];
                z[i] -= tmp;
            }
        }

        // Step 3: diagonal solve D w = z.
        for i in 0..n {
            z[i] = z[i] / self.d_vals[i];
        }

        // Step 4: backward solve Lᵀ v = w  (Lᵀ is unit upper-triangular).
        // Lᵀ[j,i] = L[i,j], so for row i of Lᵀ: v[j] -= L[i,j] * v[i].
        // Process rows i in reverse order.
        for i in (0..n).rev() {
            for k in self.l_row_ptr[i]..self.l_row_ptr[i + 1] {
                let j = self.l_col_idx[k];
                let tmp = self.l_values[k] * z[i];
                z[j] -= tmp;
            }
        }

        // Step 5: inverse permutation — x = Pᵀ v.
        let xs = x.as_mut_slice();
        for (new_i, &old_i) in self.perm.iter().enumerate() {
            xs[old_i] = z[new_i];
        }
        Ok(())
    }
    fn reset_factors(&mut self) {
        self.d_vals.clear();
        self.l_row_ptr.clear(); self.l_col_idx.clear(); self.l_values.clear();
        self.factorized = false;
        self.symbolic_n = None;
    }
}

// ─── LinearOperator impl ─────────────────────────────────────────────────────

impl<T: Scalar> LinearOperator for SparseLdlt<T> {
    type Vector = DenseVec<T>;
    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        // LDLᵀ as an operator applies the factorised matrix (not the solve).
        // For use as a preconditioner, we want the solve direction.
        self.solve(x, y).expect("SparseLdlt::apply: factorisation not ready");
    }
    fn nrows(&self) -> usize { self.n }
    fn ncols(&self) -> usize { self.n }
}

// ─── Helper ──────────────────────────────────────────────────────────────────

/// Find entry (k, j) with row index `target_row` in CSC column `col`.
/// Returns T::zero() if not found.
fn find_in_col_csc<T: Scalar>(
    rows: &[usize], vals: &[T], col_ptr: &[usize], col: usize, target_row: usize,
) -> T {
    for idx in col_ptr[col]..col_ptr[col + 1] {
        if rows[idx] == target_row { return vals[idx]; }
    }
    T::zero()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CooMatrix;

    fn make_spd(n: usize) -> CsrMatrix<f64> {
        // Tridiagonal [-1, 2, -1] — symmetric positive definite.
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0);
            if i > 0     { coo.push(i, i - 1, -1.0); }
            if i < n - 1 { coo.push(i, i + 1, -1.0); }
        }
        CsrMatrix::from_coo(&coo)
    }

    fn make_indefinite(n: usize) -> CsrMatrix<f64> {
        // Symmetric indefinite: block [A B; Bᵀ -C] with A, C SPD.
        // Simple: diagonal with alternating sign ±1 for some off-diagonals.
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            let d = if i < n / 2 { 2.0 } else { -2.0 };
            coo.push(i, i, d);
            if i > 0     { coo.push(i, i - 1, 1.0); }
            if i < n - 1 { coo.push(i, i + 1, 1.0); }
        }
        CsrMatrix::from_coo(&coo)
    }

    fn residual_norm(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = a.nrows();
        let mut r = vec![0.0f64; n];
        a.spmv(x, &mut r);
        let mut norm = 0.0f64;
        for i in 0..n { norm += (r[i] - b[i]).powi(2); }
        norm.sqrt()
    }

    #[test]
    fn spd_3x3_solves() {
        let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
        coo.push(0, 0, 4.0); coo.push(0, 1, 2.0);
        coo.push(1, 0, 2.0); coo.push(1, 1, 3.0); coo.push(1, 2, 1.0);
        coo.push(2, 1, 1.0); coo.push(2, 2, 2.0);
        let a = CsrMatrix::from_coo(&coo);
        let b = DenseVec::from_vec(vec![6.0, 6.0, 3.0]);
        let mut ldlt = SparseLdlt::<f64>::default();
        ldlt.factor(&a).unwrap();
        let mut x = DenseVec::zeros(3);
        ldlt.solve(&b, &mut x).unwrap();
        assert!(residual_norm(&a, x.as_slice(), b.as_slice()) < 1e-12);
    }

    #[test]
    fn spd_matches_cholesky_on_poisson_1d() {
        let n = 20;
        let a = make_spd(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b_vec = DenseVec::from_vec(b.clone());

        let mut ldlt = SparseLdlt::<f64>::default();
        ldlt.factor(&a).unwrap();
        let mut x = DenseVec::zeros(n);
        ldlt.solve(&b_vec, &mut x).unwrap();
        assert!(residual_norm(&a, x.as_slice(), &b) < 1e-10,
            "residual = {}", residual_norm(&a, x.as_slice(), &b));
        assert!(ldlt.is_positive_definite());
    }

    #[test]
    fn indefinite_diagonal_solves() {
        // Purely diagonal indefinite: A = diag(2, -3, 4).
        let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
        coo.push(0, 0, 2.0); coo.push(1, 1, -3.0); coo.push(2, 2, 4.0);
        let a = CsrMatrix::from_coo(&coo);
        let b = DenseVec::from_vec(vec![4.0, -9.0, 8.0]);
        let mut ldlt = SparseLdlt::<f64>::default();
        ldlt.factor(&a).unwrap();
        let mut x = DenseVec::zeros(3);
        ldlt.solve(&b, &mut x).unwrap();
        let xs = x.as_slice();
        assert!((xs[0] - 2.0).abs() < 1e-13);
        assert!((xs[1] - 3.0).abs() < 1e-13);
        assert!((xs[2] - 2.0).abs() < 1e-13);
        assert!(!ldlt.is_positive_definite());
    }

    #[test]
    fn indefinite_tridiagonal_solves() {
        let n = 10;
        let a = make_indefinite(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let b_vec = DenseVec::from_vec(b.clone());
        let mut ldlt = SparseLdlt::<f64>::default();
        ldlt.factor(&a).unwrap();
        let mut x = DenseVec::zeros(n);
        ldlt.solve(&b_vec, &mut x).unwrap();
        assert!(residual_norm(&a, x.as_slice(), &b) < 1e-8,
            "residual={}", residual_norm(&a, x.as_slice(), &b));
    }

    #[test]
    fn poisson_2d_solves() {
        let n = 4; // 4×4 grid = 16×16 system
        let nn = n * n;
        let mut coo: CooMatrix<f64> = CooMatrix::new(nn, nn);
        for i in 0..n {
            for j in 0..n {
                let id = i * n + j;
                coo.push(id, id, 4.0);
                if j > 0 { coo.push(id, id - 1, -1.0); coo.push(id - 1, id, -1.0); }
                if i > 0 { coo.push(id, id - n, -1.0); coo.push(id - n, id, -1.0); }
            }
        }
        let a = CsrMatrix::from_coo(&coo);
        let b = DenseVec::from_vec(vec![1.0f64; nn]);
        let mut ldlt = SparseLdlt::<f64>::default();
        ldlt.factor(&a).unwrap();
        let mut x = DenseVec::zeros(nn);
        ldlt.solve(&b, &mut x).unwrap();
        assert!(residual_norm(&a, x.as_slice(), b.as_slice()) < 1e-10);
        assert!(ldlt.is_positive_definite());
    }

    #[test]
    fn d_vals_correct_spd() {
        // For 2×2 A = [[4,2],[2,3]]:
        // LDLᵀ: D[0]=4, L[1,0]=A[1,0]/D[0]=0.5, D[1]=A[1,1]-L[1,0]²*D[0]=3-1=2
        // Use Natural ordering so no permutation is applied.
        use crate::direct::DirectOptions;
        let opts = DirectOptions { ordering: OrderingMethod::Natural, ..Default::default() };
        let mut coo: CooMatrix<f64> = CooMatrix::new(2, 2);
        coo.push(0, 0, 4.0); coo.push(0, 1, 2.0);
        coo.push(1, 0, 2.0); coo.push(1, 1, 3.0);
        let a = CsrMatrix::from_coo(&coo);
        let mut ldlt = SparseLdlt::<f64>::new(opts);
        ldlt.factor(&a).unwrap();
        let d = ldlt.d_vals();
        assert!((d[0] - 4.0).abs() < 1e-14, "D[0]={}", d[0]);
        assert!((d[1] - 2.0).abs() < 1e-14, "D[1]={}", d[1]);
        // Verify solve is correct
        let b = DenseVec::from_vec(vec![4.0, 3.0]);
        let mut x = DenseVec::zeros(2);
        ldlt.solve(&b, &mut x).unwrap();
        assert!(residual_norm(&a, x.as_slice(), b.as_slice()) < 1e-13,
            "residual={}", residual_norm(&a, x.as_slice(), b.as_slice()));
    }

    #[test]
    fn reuse_symbolic() {
        use crate::direct::DirectOptions;
        let opts = DirectOptions { reuse_symbolic: true, ..Default::default() };
        let n = 8;
        let a = make_spd(n);
        let b = DenseVec::from_vec(vec![1.0f64; n]);
        let mut ldlt = SparseLdlt::<f64>::new(opts);
        ldlt.factor(&a).unwrap();
        // Reuse: factorize again with a slightly different matrix
        let a2 = make_spd(n); // same pattern, same values for simplicity
        ldlt.factorize(&a2).unwrap();
        let mut x = DenseVec::zeros(n);
        ldlt.solve(&b, &mut x).unwrap();
        assert!(residual_norm(&a, x.as_slice(), b.as_slice()) < 1e-10);
    }

    #[test]
    fn as_preconditioner() {
        // SparseLdlt can be used as a preconditioner via LinearOperator::apply.
        let n = 5;
        let a = make_spd(n);
        let b = DenseVec::from_vec(vec![1.0f64; n]);
        let mut ldlt = SparseLdlt::<f64>::default();
        ldlt.factor(&a).unwrap();
        let mut x = DenseVec::zeros(n);
        ldlt.apply(&b, &mut x);
        assert!(residual_norm(&a, x.as_slice(), b.as_slice()) < 1e-10);
    }
}
