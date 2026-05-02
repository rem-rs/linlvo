//! Block Jacobi preconditioner.
//!
//! Inverts `block_size × block_size` dense blocks on the diagonal of A.
//! Useful for multi-DOF-per-node FEA problems (e.g. 3D elasticity with 3
//! DOF/node) where the diagonal block structure aligns with physical coupling.
//!
//! Each block is factorised in-place with dense LU (partial pivoting) at
//! construction time.  Application is forward/back substitution per block.
//!
//! **Analogs**
//!   PETSc: `PCBJACOBI`

use crate::core::{
    error::SolverError, preconditioner::Preconditioner, scalar::Scalar, vector::DenseVec,
};
use crate::sparse::CsrMatrix;

/// Block Jacobi preconditioner.
///
/// Stores LU-factored dense blocks for each diagonal block of the matrix.
pub struct BlockJacobiPrecond<T> {
    block_size: usize,
    n_blocks:   usize,
    /// Each block stored column-major as Vec<T> of length block_size².
    /// In-place LU with partial pivoting.
    blocks:     Vec<Vec<T>>,
    pivots:     Vec<Vec<usize>>,
}

impl<T: Scalar> BlockJacobiPrecond<T> {
    /// Build from a CSR matrix.
    ///
    /// Returns `Err` if:
    /// - `n` is not divisible by `block_size`
    /// - any diagonal block is singular (pivot < sqrt(ε))
    pub fn from_csr(mat: &CsrMatrix<T>, block_size: usize) -> Result<Self, SolverError> {
        let n = mat.nrows();
        if block_size == 0 {
            return Err(SolverError::PrecondSetupFailed {
                reason: "block_size must be > 0".into(),
            });
        }
        if n % block_size != 0 {
            return Err(SolverError::PrecondSetupFailed {
                reason: format!("n={n} not divisible by block_size={block_size}"),
            });
        }
        let n_blocks = n / block_size;
        let bs = block_size;

        let mut blocks = Vec::with_capacity(n_blocks);
        let mut pivots = Vec::with_capacity(n_blocks);

        for b in 0..n_blocks {
            let row0 = b * bs;
            let col0 = b * bs;

            // Extract block b (row-major into a flat vec).
            let mut blk = vec![T::zero(); bs * bs];
            for i in 0..bs {
                let row = row0 + i;
                let rp = mat.row_ptr();
                for idx in rp[row]..rp[row + 1] {
                    let col = mat.col_idx()[idx];
                    if col >= col0 && col < col0 + bs {
                        let j = col - col0;
                        blk[i * bs + j] = mat.values()[idx];
                    }
                }
            }

            // LU factorise in-place (row-major).
            let mut piv = vec![0usize; bs];
            dense_lu_factor(&mut blk, &mut piv, bs).map_err(|row| {
                SolverError::SingularMatrix { row: row0 + row }
            })?;

            blocks.push(blk);
            pivots.push(piv);
        }

        Ok(BlockJacobiPrecond { block_size, n_blocks, blocks, pivots })
    }
}

impl<T: Scalar> Preconditioner for BlockJacobiPrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let bs = self.block_size;
        let xs = x.as_slice();
        let ys = y.as_mut_slice();

        for b in 0..self.n_blocks {
            let off = b * bs;
            // Copy the source block to output and solve in-place to avoid per-block allocations.
            ys[off..off + bs].copy_from_slice(&xs[off..off + bs]);
            match bs {
                1 => dense_lu_solve_1x1(&self.blocks[b], &mut ys[off..off + bs]),
                2 => dense_lu_solve_small::<T, 2>(&self.blocks[b], &self.pivots[b], &mut ys[off..off + bs]),
                3 => dense_lu_solve_small::<T, 3>(&self.blocks[b], &self.pivots[b], &mut ys[off..off + bs]),
                4 => dense_lu_solve_small::<T, 4>(&self.blocks[b], &self.pivots[b], &mut ys[off..off + bs]),
                _ => dense_lu_solve(&self.blocks[b], &self.pivots[b], &mut ys[off..off + bs], bs),
            }
        }
    }
}

// ─── Dense LU helpers (row-major, in-place, partial pivoting) ────────────────

/// Factorise a bs×bs dense matrix (row-major) in-place with partial pivoting.
/// Stores L (strictly lower) and U (upper) packed into `block`.
/// Returns Err(row) if a zero pivot is encountered.
fn dense_lu_factor<T: Scalar>(
    block:  &mut [T],
    pivots: &mut [usize],
    bs:     usize,
) -> Result<(), usize> {
    for k in 0..bs {
        // Find the pivot in column k below row k.
        let mut max_abs = T::zero();
        let mut max_row = k;
        for i in k..bs {
            let v = block[i * bs + k].abs();
            if v > max_abs {
                max_abs = v;
                max_row = i;
            }
        }
        pivots[k] = max_row;

        // Swap rows k and max_row.
        if max_row != k {
            for j in 0..bs {
                block.swap(k * bs + j, max_row * bs + j);
            }
        }

        let pivot = block[k * bs + k];
        if pivot.abs() < T::machine_epsilon().sqrt() {
            return Err(k);
        }
        let inv_pivot = T::one() / pivot;

        // Eliminate below.
        for i in (k + 1)..bs {
            let factor = block[i * bs + k] * inv_pivot;
            block[i * bs + k] = factor; // store multiplier in L
            for j in (k + 1)..bs {
                let u_kj = block[k * bs + j];
                block[i * bs + j] -= factor * u_kj;
            }
        }
    }
    Ok(())
}

/// Forward/back substitution using the LU factorisation in `block`.
/// Overwrites `rhs` with the solution.
fn dense_lu_solve<T: Scalar>(block: &[T], pivots: &[usize], rhs: &mut [T], bs: usize) {
    // Apply row permutations (forward pass).
    for (k, &piv) in pivots[..bs].iter().enumerate() {
        rhs.swap(k, piv);
    }
    // Forward substitution: L * y = rhs (L has 1s on diagonal, stored below).
    for i in 1..bs {
        for j in 0..i {
            let lij = block[i * bs + j];
            rhs[i] -= lij * rhs[j];
        }
    }
    // Back substitution: U * x = y.
    for i in (0..bs).rev() {
        for j in (i + 1)..bs {
            let uij = block[i * bs + j];
            rhs[i] -= uij * rhs[j];
        }
        rhs[i] /= block[i * bs + i];
    }
}

#[inline]
fn dense_lu_solve_1x1<T: Scalar>(block: &[T], rhs: &mut [T]) {
    rhs[0] /= block[0];
}

#[inline]
fn dense_lu_solve_small<T: Scalar, const BS: usize>(
    block: &[T],
    pivots: &[usize],
    rhs: &mut [T],
) {
    let mut vals = [T::zero(); BS];
    vals.copy_from_slice(&rhs[..BS]);

    for (k, &pivot) in pivots.iter().take(BS).enumerate() {
        vals.swap(k, pivot);
    }
    for i in 1..BS {
        for j in 0..i {
            vals[i] -= block[i * BS + j] * vals[j];
        }
    }
    for i in (0..BS).rev() {
        for j in (i + 1)..BS {
            vals[i] -= block[i * BS + j] * vals[j];
        }
        vals[i] /= block[i * BS + i];
    }

    rhs[..BS].copy_from_slice(&vals);
}
