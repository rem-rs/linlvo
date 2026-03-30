//! Interpolation (prolongation) operator construction for AMG.
//!
//! ## Direct interpolation (for RS-AMG)
//!
//! Given the C/F splitting and the strong-connection graph, builds a
//! prolongation operator P of size n_fine × n_coarse:
//!
//! - For C-points: `P[c, coarse_id(c)] = 1`.
//! - For F-points i: interpolate from strongly connected C-neighbours:
//!   ```text
//!   P[i, k] = -a_{ij} / (a_{ii} + sum_{m∈F∩S(i)} a_{im})
//!   ```
//!   for each C-neighbour j with coarse index k.
//!
//! ## Smoothed prolongation (for SA-AMG)
//!
//! Applies one step of Jacobi smoothing to the tentative prolongator P₀:
//! ```text
//! P = (I - ω D⁻¹ A) P₀
//! ```
//! where D = diag(A) and ω = 4/(3 ρ(D⁻¹ A)).  We approximate the spectral
//! radius ρ by the Gershgorin bound: max_i (|a_{ii}|⁻¹ Σ_{j≠i} |a_{ij}|).
//!
//! Row computations in both routines are independent and are parallelised with
//! Rayon when the `rayon` feature is enabled.

use crate::core::scalar::Scalar;
use crate::sparse::CsrMatrix;
use crate::amg::coarsen_rs::{NodeType, coarse_index_map};

// ─── RS direct interpolation ─────────────────────────────────────────────────

/// Build P for classical RS-AMG.
///
/// When compiled with `feature = "rayon"`, rows are computed in parallel.
pub fn rs_interpolation<T: Scalar>(
    a:      &CsrMatrix<T>,
    status: &[NodeType],
) -> CsrMatrix<T> {
    let n           = a.nrows();
    let (nc, c_map) = coarse_index_map(status);
    let rp          = a.row_ptr();
    let ci          = a.col_idx();
    let vs          = a.values();

    let compute_row = |i: usize| -> Vec<(usize, T)> {
        match status[i] {
            NodeType::Coarse => {
                vec![(c_map[i], T::one())]
            }
            NodeType::Fine | NodeType::Undecided => {
                let mut a_ii      = T::zero();
                let mut f_sum     = T::zero();
                let mut c_entries: Vec<(usize, T)> = Vec::new();

                for k in rp[i]..rp[i + 1] {
                    let j = ci[k];
                    if j == i {
                        a_ii = vs[k];
                    } else if status[j] == NodeType::Coarse {
                        c_entries.push((c_map[j], vs[k]));
                    } else {
                        f_sum += vs[k];
                    }
                }

                let denom = a_ii + f_sum;
                if denom.abs() < T::machine_epsilon() || c_entries.is_empty() {
                    return Vec::new();
                }

                let mut row: Vec<(usize, T)> = c_entries
                    .into_iter()
                    .map(|(coarse_j, a_ij)| (coarse_j, -a_ij / denom))
                    .collect();
                row.sort_unstable_by_key(|&(j, _)| j);
                row
            }
        }
    };

    #[cfg(feature = "rayon")]
    let p_rows: Vec<Vec<(usize, T)>> = {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(compute_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let p_rows: Vec<Vec<(usize, T)>> = (0..n).map(compute_row).collect();

    pack_csr(n, nc, p_rows)
}

// ─── SA smoothed prolongation ─────────────────────────────────────────────────

/// Apply one Jacobi-smoothing step to tentative prolongator P₀:
///   P = (I - ω D⁻¹ A) P₀
///
/// `omega_factor`: fraction of 4/3; use 0.667 (default) for standard SA.
///
/// When compiled with `feature = "rayon"`, rows are computed in parallel.
pub fn smooth_prolongation<T: Scalar>(
    a:            &CsrMatrix<T>,
    p0:           &CsrMatrix<T>,
    omega_factor: f64,
) -> CsrMatrix<T> {
    let n  = a.nrows();
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vs = a.values();

    // Gershgorin spectral radius estimate of D⁻¹A (sequential — O(nnz)).
    let mut rho = T::zero();
    for i in 0..n {
        let mut d   = T::zero();
        let mut off = T::zero();
        for k in rp[i]..rp[i + 1] {
            if ci[k] == i { d = vs[k].abs(); } else { off += vs[k].abs(); }
        }
        if d > T::zero() {
            let r = off / d;
            if r > rho { rho = r; }
        }
    }
    let omega = if rho > T::zero() {
        T::from_f64(omega_factor * 4.0 / 3.0) / rho
    } else {
        T::from_f64(omega_factor * 2.0 / 3.0)
    };

    let diag   = a.diag();
    let nc     = p0.ncols();
    let p0_rp  = p0.row_ptr();
    let p0_ci  = p0.col_idx();
    let p0_vs  = p0.values();

    // Each row of P is computed independently: read-only access to a, p0, diag.
    let compute_row = |i: usize| -> Vec<(usize, T)> {
        let inv_di = if diag[i].abs() > T::machine_epsilon() {
            T::one() / diag[i]
        } else {
            T::zero()
        };

        // Accumulate P₀ row i.
        let mut acc: std::collections::HashMap<usize, T> =
            std::collections::HashMap::new();
        for k in p0_rp[i]..p0_rp[i + 1] {
            *acc.entry(p0_ci[k]).or_insert(T::zero()) += p0_vs[k];
        }

        // Subtract ω * inv_di * (A row i) * P₀.
        for ka in rp[i]..rp[i + 1] {
            let j     = ci[ka];
            let coeff = omega * inv_di * vs[ka];
            for kb in p0_rp[j]..p0_rp[j + 1] {
                let col = p0_ci[kb];
                *acc.entry(col).or_insert(T::zero()) -= coeff * p0_vs[kb];
            }
        }

        let mut row: Vec<(usize, T)> = acc
            .into_iter()
            .filter(|&(_, v)| v.abs() > T::machine_epsilon() * T::from_f64(1e-3))
            .collect();
        row.sort_unstable_by_key(|&(j, _)| j);
        row
    };

    #[cfg(feature = "rayon")]
    let p_rows: Vec<Vec<(usize, T)>> = {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(compute_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let p_rows: Vec<Vec<(usize, T)>> = (0..n).map(compute_row).collect();

    pack_csr(n, nc, p_rows)
}

// ─── helper ───────────────────────────────────────────────────────────────────

fn pack_csr<T: Scalar>(
    nrows: usize,
    ncols: usize,
    rows:  Vec<Vec<(usize, T)>>,
) -> CsrMatrix<T> {
    let nnz: usize  = rows.iter().map(|r| r.len()).sum();
    let mut row_ptr = vec![0usize; nrows + 1];
    let mut col_idx = Vec::with_capacity(nnz);
    let mut values  = Vec::with_capacity(nnz);
    for (i, row) in rows.iter().enumerate() {
        row_ptr[i + 1] = row_ptr[i] + row.len();
        for &(j, v) in row {
            col_idx.push(j);
            values.push(v);
        }
    }
    CsrMatrix::from_raw(nrows, ncols, row_ptr, col_idx, values)
}
