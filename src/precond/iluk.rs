//! ILU(k) — Incomplete LU with level-of-fill k.
//!
//! Extends ILU(0) by allowing fill-in entries up to level k.  The level of
//! entry (i, j) is defined as:
//!   lev(i, j) = 0                          if a_{ij} ≠ 0
//!   lev(i, j) = min_k( lev(i,k) + lev(k,j) + 1 )   otherwise
//!
//! Only entries with lev(i, j) ≤ k are kept in the factorisation.
//! k = 0 reproduces ILU(0); k = ∞ gives the exact LU factorisation.
//!
//! **Algorithm**: Saad, "Iterative Methods for Sparse Linear Systems", §10.3.
//!
//! **Analogs**
//!   PETSc: `PCILU` with `PCFactorSetLevels(pc, k)`
//!   HYPRE: `HYPRE_EuclidCreate` with `HYPRE_EuclidSetLevel(solver, k)`

#![allow(clippy::needless_range_loop)]
use std::collections::BTreeMap;

use crate::core::{
    error::SolverError, preconditioner::Preconditioner, scalar::{ComplexScalar, Scalar}, vector::DenseVec,
};
use crate::sparse::CsrMatrix;

/// ILU(k) preconditioner.
pub struct IlukPrecond<T> {
    nrows:    usize,
    row_ptr:  Vec<usize>,
    col_idx:  Vec<usize>,
    lu_val:   Vec<T>,
    diag_pos: Vec<usize>,
}

impl<T: ComplexScalar> IlukPrecond<T> {
    /// Compute ILU(k) factorisation of `mat` with fill level `fill_level`.
    pub fn from_csr(mat: &CsrMatrix<T>, fill_level: usize) -> Result<Self, SolverError> {
        let n = mat.nrows();
        if mat.ncols() != n {
            return Err(SolverError::PrecondSetupFailed {
                reason: "ILU(k) requires a square matrix".into(),
            });
        }

        let rp = mat.row_ptr();
        let ci = mat.col_idx();
        let vs = mat.values();

        // ── Symbolic phase ────────────────────────────────────────────────────
        // levels[i]: col → fill-level for entries in row i
        let mut levels: Vec<BTreeMap<usize, usize>> = (0..n)
            .map(|i| {
                let mut m = BTreeMap::new();
                for k in rp[i]..rp[i + 1] {
                    m.insert(ci[k], 0usize);
                }
                m
            })
            .collect();

        for i in 1..n {
            // collect lower-triangle pivots (k < i) present in row i
            let pivots: Vec<(usize, usize)> =
                levels[i].range(..i).map(|(&k, &lv)| (k, lv)).collect();

            for (k, lev_ik) in pivots {
                if lev_ik >= fill_level {
                    continue;
                }
                // propagate fill from row k's upper part into row i
                let k_upper: Vec<(usize, usize)> =
                    levels[k].range((k + 1)..).map(|(&j, &lv)| (j, lv)).collect();
                for (j, lev_kj) in k_upper {
                    let new_lev = lev_ik + lev_kj + 1;
                    if new_lev <= fill_level {
                        let e = levels[i].entry(j).or_insert(new_lev);
                        if new_lev < *e {
                            *e = new_lev;
                        }
                    }
                }
            }
        }

        // ── Build extended CSR from symbolic pattern ───────────────────────
        let mut new_rp = vec![0usize; n + 1];
        for i in 0..n {
            new_rp[i + 1] = new_rp[i] + levels[i].len();
        }
        let nnz = new_rp[n];
        let mut new_ci  = Vec::with_capacity(nnz);
        let mut lu_val  = Vec::with_capacity(nnz);
        let mut diag_pos = vec![0usize; n];

        for i in 0..n {
            let mut found = false;
            for &j in levels[i].keys() {
                let pos = new_ci.len();
                if j == i {
                    diag_pos[i] = pos;
                    found = true;
                }
                new_ci.push(j);
                lu_val.push(orig_val(rp, ci, vs, i, j));
            }
            if !found {
                return Err(SolverError::PrecondSetupFailed {
                    reason: format!("ILU(k): missing diagonal at row {i}"),
                });
            }
        }

        // ── Numeric phase: ILU factorisation on extended pattern ──────────
        let tol = T::machine_epsilon() * <T::Real as Scalar>::from_f64(1e6);
        for i in 1..n {
            let i_start = new_rp[i];
            let i_end   = new_rp[i + 1];

            for pos_ik in i_start..i_end {
                let k = new_ci[pos_ik];
                if k >= i {
                    break;
                }
                let u_kk = lu_val[diag_pos[k]];
                if u_kk.abs() < tol {
                    return Err(SolverError::SingularMatrix { row: k });
                }
                let factor = lu_val[pos_ik] / u_kk;
                lu_val[pos_ik] = factor;

                let k_end = new_rp[k + 1];
                for pos_kj in (diag_pos[k] + 1)..k_end {
                    let j = new_ci[pos_kj];
                    if let Some(pos_ij) = find_in_row(&new_ci, i_start, i_end, j) {
                        let v = lu_val[pos_kj];
                        lu_val[pos_ij] -= factor * v;
                    }
                }
            }
        }

        Ok(IlukPrecond { nrows: n, row_ptr: new_rp, col_idx: new_ci, lu_val, diag_pos })
    }
}

impl<T: ComplexScalar> Preconditioner for IlukPrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n = self.nrows;
        let xs = x.as_slice();
        let ys = y.as_mut_slice();

        // Forward solve  L z = x  (unit lower triangular, diagonal implicit = 1)
        for i in 0..n {
            let mut s = xs[i];
            for k in self.row_ptr[i]..self.diag_pos[i] {
                s -= self.lu_val[k] * ys[self.col_idx[k]];
            }
            ys[i] = s;
        }

        // Backward solve  U y = z
        for i in (0..n).rev() {
            let mut s = ys[i];
            for k in (self.diag_pos[i] + 1)..self.row_ptr[i + 1] {
                s -= self.lu_val[k] * ys[self.col_idx[k]];
            }
            ys[i] = s / self.lu_val[self.diag_pos[i]];
        }
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Look up the value of entry (i, j) in the original CSR matrix; return 0 if absent.
fn orig_val<T: ComplexScalar>(rp: &[usize], ci: &[usize], vs: &[T], i: usize, j: usize) -> T {
    for pos in rp[i]..rp[i + 1] {
        match ci[pos].cmp(&j) {
            std::cmp::Ordering::Equal   => return vs[pos],
            std::cmp::Ordering::Greater => break,
            std::cmp::Ordering::Less    => {}
        }
    }
    T::zero()
}

fn find_in_row(ci: &[usize], start: usize, end: usize, j: usize) -> Option<usize> {
    for pos in start..end {
        match ci[pos].cmp(&j) {
            std::cmp::Ordering::Equal   => return Some(pos),
            std::cmp::Ordering::Greater => return None,
            std::cmp::Ordering::Less    => {}
        }
    }
    None
}
