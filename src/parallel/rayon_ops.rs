//! Rayon-parallelised sparse and dense-vector operations.
//!
//! Each function has two code paths:
//! - **`feature = "rayon"`**: uses `rayon::prelude::*` for data-parallel loops.
//! - **fallback**: delegates to the scalar implementation in `crate::sparse::ops`.
//!
//! The parallel SpMV partitions the CSR row index space across Rayon threads.
//! Each thread independently accumulates a contiguous row range — no
//! synchronisation is needed because rows of A×x are independent.
//!
//! **Analogs**
//!   PETSc: `MatMult` (distributed-memory via MPI, shared-memory via OpenMP)
//!   HYPRE: `hypre_ParCSRMatrixMatvec` (MPI + OpenMP hybrid)

use crate::core::scalar::Scalar;
use crate::sparse::CsrMatrix;

// ─── SpMV ─────────────────────────────────────────────────────────────────────

/// Parallel `y ← A · x` for CSR matrices.
///
/// Falls back to serial when the `rayon` feature is disabled or `n` is small.
pub fn parallel_spmv<T: Scalar + Send + Sync>(
    mat: &CsrMatrix<T>,
    x:   &[T],
    y:   &mut [T],
) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;

        let rp = mat.row_ptr();
        let ci = mat.col_idx();
        let vs = mat.values();
        let n = mat.nrows();

        // Rayon overhead dominates for very small systems; keep serial path fast.
        if n < 256 {
            mat.spmv(x, y);
            return;
        }

        let threads = rayon::current_num_threads().max(1);
        let nnz = vs.len().max(1);
        let avg_nnz_per_row = (nnz / n.max(1)).max(1);
        let target_chunks = threads * 8;
        let target_nnz_per_chunk = (nnz + target_chunks - 1) / target_chunks;
        let mut chunk_rows = (target_nnz_per_chunk / avg_nnz_per_row).max(32);
        chunk_rows = chunk_rows.min(1024).min(n.max(1));

        y.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, y_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                for (local_i, yi) in y_chunk.iter_mut().enumerate() {
                    let i = row_start + local_i;
                    let mut sum = T::zero();
                    for k in rp[i]..rp[i + 1] {
                        sum += vs[k] * x[ci[k]];
                    }
                    *yi = sum;
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        mat.spmv(x, y);
    }
}

/// Parallel `y ← α·A·x + β·y` for CSR matrices.
pub fn parallel_spmv_add<T: Scalar + Send + Sync>(
    mat:   &CsrMatrix<T>,
    alpha: T,
    x:     &[T],
    beta:  T,
    y:     &mut [T],
) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;

        let rp = mat.row_ptr();
        let ci = mat.col_idx();
        let vs = mat.values();
        let n = mat.nrows();

        if n < 256 {
            mat.spmv_add(alpha, x, beta, y);
            return;
        }

        let threads = rayon::current_num_threads().max(1);
        let nnz = vs.len().max(1);
        let avg_nnz_per_row = (nnz / n.max(1)).max(1);
        let target_chunks = threads * 8;
        let target_nnz_per_chunk = (nnz + target_chunks - 1) / target_chunks;
        let mut chunk_rows = (target_nnz_per_chunk / avg_nnz_per_row).max(32);
        chunk_rows = chunk_rows.min(1024).min(n.max(1));

        y.par_chunks_mut(chunk_rows)
            .enumerate()
            .for_each(|(chunk_idx, y_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                for (local_i, yi) in y_chunk.iter_mut().enumerate() {
                    let i = row_start + local_i;
                    let mut sum = T::zero();
                    for k in rp[i]..rp[i + 1] {
                        sum += vs[k] * x[ci[k]];
                    }
                    *yi = alpha * sum + beta * *yi;
                }
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        mat.spmv_add(alpha, x, beta, y);
    }
}

// ─── Dense-vector AXPY ───────────────────────────────────────────────────────

/// Parallel `y += alpha * x`.
pub fn parallel_axpy<T: Scalar + Send + Sync>(alpha: T, x: &[T], y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        y.par_iter_mut().zip(x.par_iter()).for_each(|(yi, &xi)| {
            *yi += alpha * xi;
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        crate::sparse::ops::axpy(alpha, x, y);
    }
}

/// Parallel `y = alpha * x + beta * y`.
pub fn parallel_axpby<T: Scalar + Send + Sync>(alpha: T, x: &[T], beta: T, y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        y.par_iter_mut().zip(x.par_iter()).for_each(|(yi, &xi)| {
            *yi = alpha * xi + beta * *yi;
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        crate::sparse::ops::axpby(alpha, x, beta, y);
    }
}

// ─── Dot product / norm ───────────────────────────────────────────────────────

/// Parallel Euclidean inner product `<x, y>`.
pub fn parallel_dot<T: Scalar + Send + Sync>(x: &[T], y: &[T]) -> T {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        x.par_iter()
            .zip(y.par_iter())
            .map(|(&a, &b)| a * b)
            .reduce(|| T::zero(), |acc, v| acc + v)
    }

    #[cfg(not(feature = "rayon"))]
    {
        crate::sparse::ops::dot(x, y)
    }
}

/// Parallel Euclidean 2-norm `√(Σ xᵢ²)`.
pub fn parallel_norm2<T: Scalar + Send + Sync>(x: &[T]) -> T {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        x.par_iter()
            .map(|&v| v * v)
            .reduce(|| T::zero(), |acc, v| acc + v)
            .sqrt()
    }

    #[cfg(not(feature = "rayon"))]
    {
        crate::sparse::ops::norm2(x)
    }
}
