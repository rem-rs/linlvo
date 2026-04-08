//! ILUT — Incomplete LU with dual threshold (drop tolerance + fill bound).
//!
//! ILUT(τ, p) generalises ILU(0) in two ways:
//! - **Drop tolerance** τ: entries smaller than τ · ‖row‖₂ are discarded.
//! - **Fill bound** p: at most p off-diagonal entries per row are kept in each
//!   of L and U (the p entries with largest absolute value).
//!
//! τ = 0 / p = ∞ reproduces the exact LU; τ = ∞ degenerates to diagonal.
//!
//! **Algorithm**: Saad, §10.4.2 (Algorithm 10.6).
//!
//! **Analogs**
//!   PETSc: `PCILU` with `PCFactorSetFill` + `PCFactorSetDropTolerance`
//!   HYPRE: `HYPRE_EuclidCreate` with threshold parameter

use crate::core::{
    error::SolverError, preconditioner::Preconditioner, scalar::Scalar, vector::DenseVec,
};
use crate::sparse::CsrMatrix;

/// ILUT(τ, p) preconditioner.
///
/// Stores separate lower (L) and upper (U) factor rows, each with at most
/// `p` off-diagonal nonzeros.
pub struct IlutPrecond<T> {
    nrows:   usize,
    l_rows:  Vec<Vec<(usize, T)>>,  // (col, val), col < row, sorted ascending
    u_rows:  Vec<Vec<(usize, T)>>,  // (col, val), col > row, sorted ascending
    u_diag:  Vec<T>,                // u[i, i]
}

impl<T: Scalar> IlutPrecond<T> {
    /// Compute ILUT factorisation.
    ///
    /// * `tau`    — relative drop tolerance (e.g. 0.01)
    /// * `p_fill` — max off-diagonal fill per row in L and in U
    pub fn from_csr(mat: &CsrMatrix<T>, tau: f64, p_fill: usize) -> Result<Self, SolverError> {
        let n = mat.nrows();
        if mat.ncols() != n {
            return Err(SolverError::PrecondSetupFailed {
                reason: "ILUT requires a square matrix".into(),
            });
        }

        let rp = mat.row_ptr();
        let ci = mat.col_idx();
        let vs = mat.values();

        let mut l_rows: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        let mut u_rows: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        let mut u_diag: Vec<T>               = vec![T::zero(); n];

        // Dense working row; reused and zeroed each iteration.
        let mut w: Vec<T>    = vec![T::zero(); n];
        // Active column set (sorted).
        let mut active: Vec<usize> = Vec::new();

        let eps = T::machine_epsilon() * T::from_f64(1e6);

        for i in 0..n {
            // ── Initialise w from row i of A ──────────────────────────────
            active.clear();
            for pos in rp[i]..rp[i + 1] {
                let j = ci[pos];
                w[j] = vs[pos];
                active.push(j);
            }
            active.sort_unstable();

            // Row norm for threshold (computed before elimination).
            let row_norm: f64 = active
                .iter()
                .map(|&j| to_f64_sq(w[j]))
                .sum::<f64>()
                .sqrt();
            let thresh = T::from_f64(tau * row_norm);

            // ── Elimination pass ─────────────────────────────────────────
            let lower_cols: Vec<usize> = active.iter().copied().filter(|&j| j < i).collect();
            for k in lower_cols {
                if w[k].abs() < thresh {
                    w[k] = T::zero();
                    continue;
                }
                let ukk = u_diag[k];
                if ukk.abs() < eps {
                    zero_active(&mut w, &active);
                    return Err(SolverError::SingularMatrix { row: k });
                }
                let mult = w[k] / ukk;
                w[k] = mult;

                // w[j] -= mult * u[k, j]  for j > k
                for &(j, u_kj) in &u_rows[k] {
                    if w[j] == T::zero() {
                        active.push(j);
                    }
                    w[j] -= mult * u_kj;
                }
            }
            active.sort_unstable();
            active.dedup();

            // ── Apply threshold and p-fill to L (j < i) ──────────────────
            let mut l_entries: Vec<(usize, T)> = active
                .iter()
                .filter(|&&j| j < i)
                .filter_map(|&j| {
                    let v = w[j];
                    if v.abs() >= thresh {
                        Some((j, v))
                    } else {
                        w[j] = T::zero();
                        None
                    }
                })
                .collect();
            top_p_sort(&mut l_entries, p_fill);

            // ── Apply threshold and p-fill to U off-diagonal (j > i) ─────
            let mut u_entries: Vec<(usize, T)> = active
                .iter()
                .filter(|&&j| j > i)
                .filter_map(|&j| {
                    let v = w[j];
                    if v.abs() >= thresh {
                        Some((j, v))
                    } else {
                        w[j] = T::zero();
                        None
                    }
                })
                .collect();
            top_p_sort(&mut u_entries, p_fill);

            // Diagonal of U (always kept).
            let diag_val = w[i];
            if diag_val.abs() < eps {
                zero_active(&mut w, &active);
                return Err(SolverError::SingularMatrix { row: i });
            }
            u_diag[i] = diag_val;

            l_rows[i] = l_entries;
            u_rows[i] = u_entries;

            // ── Zero w for next row ───────────────────────────────────────
            zero_active(&mut w, &active);
            w[i] = T::zero();
        }

        Ok(IlutPrecond { nrows: n, l_rows, u_rows, u_diag })
    }
}

impl<T: Scalar> Preconditioner for IlutPrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n = self.nrows;
        let xs = x.as_slice();
        let ys = y.as_mut_slice();

        // Forward solve  L z = x  (unit lower triangular)
        for i in 0..n {
            let mut s = xs[i];
            for &(k, l_ik) in &self.l_rows[i] {
                s -= l_ik * ys[k];
            }
            ys[i] = s;
        }

        // Backward solve  U y = z
        for i in (0..n).rev() {
            let mut s = ys[i];
            for &(k, u_ik) in &self.u_rows[i] {
                s -= u_ik * ys[k];
            }
            ys[i] = s / self.u_diag[i];
        }
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn to_f64_sq<T: Scalar>(v: T) -> f64 {
    let f = num_traits::ToPrimitive::to_f64(&v).unwrap_or(0.0);
    f * f
}

fn zero_active<T: Scalar>(w: &mut [T], active: &[usize]) {
    for &j in active {
        w[j] = T::zero();
    }
}

/// Keep the top `p` entries by absolute value; sort remaining by column index.
fn top_p_sort<T: Scalar>(entries: &mut Vec<(usize, T)>, p: usize) {
    if entries.len() > p {
        entries.sort_unstable_by(|a, b| {
            b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(p);
    }
    entries.sort_unstable_by_key(|&(j, _)| j);
}
