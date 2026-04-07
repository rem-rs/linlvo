//! Sparse triangular solve routines.
//!
//! Both routines operate on a CSR-stored triangular factor together with a
//! diagonal-position index and an optional row/column permutation.

use crate::core::{error::SolverError, scalar::Scalar, vector::DenseVec};

// ─── Forward solve  L x = b ──────────────────────────────────────────────────

/// Sparse forward substitution: solve `L x = b` in-place (`x ← b` on entry).
///
/// `L` is stored in CSR format.  Entries with `col_idx < row` are the
/// strict lower-triangular part; the diagonal entry is provided separately via
/// `diag_val` (allows unit-diagonal factors where the diagonal is implicit).
///
/// `perm` maps external row indices to internal order.  Pass `None` if no
/// permutation was applied.
pub(crate) fn forward_solve_csr(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[f64],
    diag_val: &[f64],    // diag_val[i] = L[i,i] (1.0 for unit-lower)
    b: &[f64],
    x: &mut [f64],
    perm: Option<&[usize]>,   // x[perm[i]] updated in step i; None = identity
) -> Result<(), SolverError> {
    // Apply permutation to RHS on entry.
    if let Some(p) = perm {
        for i in 0..n { x[i] = b[p[i]]; }
    } else {
        x[..n].copy_from_slice(&b[..n]);
    }
    for i in 0..n {
        // sum lower-triangular contributions already computed
        let mut s = x[i];
        for k in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[k];
            if j >= i { break; }
            s -= values[k] * x[j];
        }
        let d = diag_val[i];
        if d.abs() < f64::EPSILON * 1e6 {
            return Err(SolverError::SingularMatrix { row: i });
        }
        x[i] = s / d;
    }
    Ok(())
}

/// Sparse backward substitution: solve `U x = b` in-place (`x ← b` on entry).
///
/// `U` is upper-triangular; entries with `col_idx > row` are the strict
/// upper part; the diagonal is in `diag_val`.
pub(crate) fn backward_solve_csr(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[f64],
    diag_val: &[f64],
    b: &[f64],
    x: &mut [f64],
    perm: Option<&[usize]>,   // permutation applied to result
) -> Result<(), SolverError> {
    x[..n].copy_from_slice(&b[..n]);
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[k];
            if j <= i { continue; }
            s -= values[k] * x[j];
        }
        let d = diag_val[i];
        if d.abs() < f64::EPSILON * 1e6 {
            return Err(SolverError::SingularMatrix { row: i });
        }
        x[i] = s / d;
    }
    // Apply inverse permutation to result.
    if let Some(p) = perm {
        let tmp: Vec<f64> = x[..n].to_vec();
        for i in 0..n { x[p[i]] = tmp[i]; }
    }
    Ok(())
}

// ─── Generic wrappers (public in the direct module) ──────────────────────────

/// Generic CSR forward solve `L x = b` for any `T: Scalar`.
///
/// `L` is stored as the lower-triangular portion of a CSR matrix.
/// `diag_pos[i]` is the index into `col_idx`/`values` of `L[i,i]`.
/// `unit_diag = true` treats `L[i,i] = 1` regardless of stored values.
pub fn forward_solve<T: Scalar>(
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[T],
    diag_pos: &[usize],
    unit_diag: bool,
    b: &DenseVec<T>,
    x: &mut DenseVec<T>,
) -> Result<(), SolverError> {
    let n = row_ptr.len().saturating_sub(1);
    let bs = b.as_slice();
    let xs = x.as_mut_slice();
    xs[..n].copy_from_slice(&bs[..n]);

    for i in 0..n {
        let mut s = xs[i];
        for k in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[k];
            if j >= i { break; }
            s -= values[k] * xs[j];
        }
        if unit_diag {
            xs[i] = s;
        } else {
            let d = values[diag_pos[i]];
            if d.abs() < T::machine_epsilon() * T::from_f64(1e6) {
                return Err(SolverError::SingularMatrix { row: i });
            }
            xs[i] = s / d;
        }
    }
    Ok(())
}

/// Generic CSR backward solve `U x = b`.
///
/// `U` is the upper-triangular portion of a CSR matrix.
/// `diag_pos[i]` is the index of `U[i,i]` in `col_idx`/`values`.
pub fn backward_solve<T: Scalar>(
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[T],
    diag_pos: &[usize],
    b: &DenseVec<T>,
    x: &mut DenseVec<T>,
) -> Result<(), SolverError> {
    let n = row_ptr.len().saturating_sub(1);
    let bs = b.as_slice();
    let xs = x.as_mut_slice();
    xs[..n].copy_from_slice(&bs[..n]);

    for i in (0..n).rev() {
        let mut s = xs[i];
        for k in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[k];
            if j <= i { continue; }
            s -= values[k] * xs[j];
        }
        let d = values[diag_pos[i]];
        if d.abs() < T::machine_epsilon() * T::from_f64(1e6) {
            return Err(SolverError::SingularMatrix { row: i });
        }
        xs[i] = s / d;
    }
    Ok(())
}
