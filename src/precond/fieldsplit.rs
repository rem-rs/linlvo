#![allow(clippy::needless_range_loop)]
//! FieldSplit preconditioner for multi-physics / block-structured systems.
//!
//! Decomposes the global DOF set into two disjoint fields `F₀` and `F₁` and
//! applies one of two block strategies:
//!
//! - **Block-Jacobi** (additive): solve each diagonal block independently and
//!   add the contributions.  Convergence rate independent of off-diagonal
//!   coupling; cheap but less effective.
//!
//! - **Block-Triangular** (multiplicative, lower): apply `P₀⁻¹` to field 0,
//!   correct field 1 for the off-diagonal contribution, then apply `P₁⁻¹` to
//!   field 1.  Better convergence for strongly coupled problems.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use linger::{
//!     sparse::CsrMatrix,
//!     precond::fieldsplit::{FieldSplitPrecond, SplitMode},
//!     precond::{JacobiPrecond, Ilu0Precond},
//!     DenseVec,
//! };
//! # use linger::core::preconditioner::Preconditioner;
//! # fn example(p0: JacobiPrecond<f64>, p1: JacobiPrecond<f64>) {
//! // Split 6×6 system: DOFs 0..3 → field 0, DOFs 3..6 → field 1
//! let split_point = 3_usize;
//! let prec: FieldSplitPrecond<f64> = FieldSplitPrecond::new(
//!     6,                    // total DOFs
//!     split_point,          // first DOF index of field 1
//!     SplitMode::BlockJacobi,
//!     Box::new(p0),         // Preconditioner for block (0,0)
//!     Box::new(p1),         // Preconditioner for block (1,1)
//! );
//! # }
//! ```
//!
//! ## Limitations
//!
//! - Currently supports **contiguous** 2-field splits only
//!   (`DOFs 0..split` and `split..n`).  Arbitrary index sets will be added in
//!   a future release.
//! - Off-diagonal correction in `BlockTriangular` mode is applied using the
//!   **raw** off-diagonal block of the original matrix stored at construction
//!   time.  It is the caller's responsibility to rebuild the preconditioner if
//!   the matrix changes significantly.
//!
//! ## Analogs
//!
//! PETSc: `PCFIELDSPLIT` with `PC_COMPOSITE_ADDITIVE` /
//! `PC_COMPOSITE_MULTIPLICATIVE`.

use std::sync::Arc;
use crate::{
    core::{preconditioner::Preconditioner, scalar::ComplexScalar, vector::DenseVec},
    sparse::CsrMatrix,
};

// ─── Public types ─────────────────────────────────────────────────────────────

/// Split strategy for `FieldSplitPrecond`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitMode {
    /// Block-Jacobi (additive): `P⁻¹ ≈ diag(P₀⁻¹, P₁⁻¹)`.
    BlockJacobi,
    /// Lower block-triangular (multiplicative): solve field 0, correct, solve field 1.
    BlockTriangular,
}

/// Two-field split preconditioner.
///
/// Type parameters:
/// - `T`: scalar type (usually `f64`)
pub struct FieldSplitPrecond<T: ComplexScalar> {
    split:  usize,
    n:      usize,
    mode:   SplitMode,
    p0:     Box<dyn Preconditioner<Vector = DenseVec<T>> + Send + Sync>,
    p1:     Box<dyn Preconditioner<Vector = DenseVec<T>> + Send + Sync>,
    /// Off-diagonal block A₁₀ (rows of field 1, cols of field 0).
    /// Only populated in `BlockTriangular` mode.
    off_a10: Option<Arc<OffDiag<T>>>,
}

/// Compact representation of an off-diagonal rectangular sub-block.
struct OffDiag<T> {
    /// Number of rows (size of field 1 = n - split).
    nrows: usize,
    /// Number of cols (size of field 0 = split).
    #[allow(dead_code)]
    ncols: usize,
    /// CSR row pointers (length nrows+1).
    rowptr: Vec<usize>,
    /// CSR column indices (zero-based within field 0, i.e. col - split = col_local).
    colidx: Vec<usize>,
    values: Vec<T>,
}

impl<T: ComplexScalar> OffDiag<T> {
    /// Apply `y_local += A₁₀ * x_local` where `x_local` is from field 0.
    fn apply_add(&self, x0: &[T], y1: &mut [T]) {
        for i in 0..self.nrows {
            let mut acc = T::zero();
            for idx in self.rowptr[i]..self.rowptr[i + 1] {
                acc += self.values[idx] * x0[self.colidx[idx]];
            }
            y1[i] += acc;
        }
    }
}

// ─── Construction ─────────────────────────────────────────────────────────────

impl<T: ComplexScalar> FieldSplitPrecond<T> {
    /// Build a FieldSplit preconditioner from two sub-preconditioners.
    ///
    /// - `n`: total number of DOFs.
    /// - `split`: first DOF index of field 1 (field 0 = `0..split`, field 1 =
    ///   `split..n`).  Must satisfy `0 < split < n`.
    /// - `mode`: additive (Block-Jacobi) or multiplicative (Block-Triangular).
    /// - `p0`: preconditioner for the `split × split` top-left block.
    /// - `p1`: preconditioner for the `(n-split) × (n-split)` bottom-right block.
    pub fn new(
        n:     usize,
        split: usize,
        mode:  SplitMode,
        p0:    Box<dyn Preconditioner<Vector = DenseVec<T>> + Send + Sync>,
        p1:    Box<dyn Preconditioner<Vector = DenseVec<T>> + Send + Sync>,
    ) -> Self {
        FieldSplitPrecond { split, n, mode, p0, p1, off_a10: None }
    }

    /// Build a FieldSplit preconditioner with an explicit matrix for extracting
    /// the off-diagonal block (needed for `BlockTriangular` mode).
    ///
    /// In `BlockJacobi` mode the matrix is ignored (pass `None`).
    pub fn from_matrix(
        mat:   &CsrMatrix<T>,
        split: usize,
        mode:  SplitMode,
        p0:    Box<dyn Preconditioner<Vector = DenseVec<T>> + Send + Sync>,
        p1:    Box<dyn Preconditioner<Vector = DenseVec<T>> + Send + Sync>,
    ) -> Self {
        let n = mat.nrows();
        let off_a10 = if mode == SplitMode::BlockTriangular {
            Some(Arc::new(extract_off_diag(mat, split)))
        } else {
            None
        };
        FieldSplitPrecond { split, n, mode, p0, p1, off_a10 }
    }
}

// ─── Preconditioner impl ──────────────────────────────────────────────────────

impl<T: ComplexScalar> Preconditioner for FieldSplitPrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, r: &DenseVec<T>, z: &mut DenseVec<T>) {
        let n      = self.n;
        let split  = self.split;
        let n1     = n - split;

        let r0 = DenseVec::from_vec(r.as_slice()[..split].to_vec());
        let r1 = DenseVec::from_vec(r.as_slice()[split..].to_vec());

        let mut z0 = DenseVec::zeros(split);
        let mut z1 = DenseVec::zeros(n1);

        match self.mode {
            SplitMode::BlockJacobi => {
                // z₀ = P₀⁻¹ r₀,  z₁ = P₁⁻¹ r₁  (independently)
                self.p0.apply_precond(&r0, &mut z0);
                self.p1.apply_precond(&r1, &mut z1);
            }
            SplitMode::BlockTriangular => {
                // 1. z₀ = P₀⁻¹ r₀
                self.p0.apply_precond(&r0, &mut z0);

                // 2. r̃₁ = r₁ − A₁₀ z₀
                let mut correction = vec![T::zero(); n1];
                if let Some(off) = &self.off_a10 {
                    off.apply_add(z0.as_slice(), &mut correction);
                }
                let r1c_vals: Vec<T> = r1.as_slice().iter().zip(correction.iter())
                    .map(|(&ri, &ci)| ri - ci)
                    .collect();
                let r1c = DenseVec::from_vec(r1c_vals);

                // 3. z₁ = P₁⁻¹ r̃₁
                self.p1.apply_precond(&r1c, &mut z1);
            }
        }

        // Write back into z.
        let zs = z.as_mut_slice();
        zs[..split].copy_from_slice(z0.as_slice());
        zs[split..].copy_from_slice(z1.as_slice());
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Extract the off-diagonal block A₁₀ (rows split..n, cols 0..split).
fn extract_off_diag<T: ComplexScalar>(mat: &CsrMatrix<T>, split: usize) -> OffDiag<T> {
    let n    = mat.nrows();
    let n1   = n - split;
    let rp   = mat.row_ptr();
    let ci   = mat.col_idx();
    let vals = mat.values();

    let mut rowptr = Vec::with_capacity(n1 + 1);
    let mut colidx = Vec::new();
    let mut values = Vec::new();
    rowptr.push(0usize);

    for i in split..n {
        for idx in rp[i]..rp[i + 1] {
            let col = ci[idx];
            if col < split {
                colidx.push(col);
                values.push(vals[idx]);
            }
        }
        rowptr.push(colidx.len());
    }

    OffDiag { nrows: n1, ncols: split, rowptr, colidx, values }
}
