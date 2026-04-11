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

const PARALLEL_SPMV_MIN_ROWS: usize = 256;
const PARALLEL_VECTOR_MIN_LEN: usize = 2048;
const PARALLEL_TARGET_CHUNKS_PER_THREAD: usize = 8;
const PARALLEL_SPMV_MIN_CHUNK_ROWS: usize = 32;
const PARALLEL_SPMV_MAX_CHUNK_ROWS: usize = 1024;

#[cfg(feature = "rayon")]
fn balanced_row_chunks(
    row_ptr: &[usize],
    nrows: usize,
    target_chunks: usize,
) -> Vec<(usize, usize)> {
    if nrows == 0 {
        return Vec::new();
    }

    let total_nnz = row_ptr[nrows].saturating_sub(row_ptr[0]).max(1);
    let target_chunks = target_chunks.max(1);
    let target_nnz_per_chunk = total_nnz.div_ceil(target_chunks);
    let mut chunks = Vec::with_capacity(target_chunks.min(nrows));
    let mut row_start = 0;

    while row_start < nrows {
        let mut row_end = (row_start + PARALLEL_SPMV_MIN_CHUNK_ROWS).min(nrows);
        while row_end < nrows && row_end - row_start < PARALLEL_SPMV_MAX_CHUNK_ROWS {
            let nnz = row_ptr[row_end] - row_ptr[row_start];
            if nnz >= target_nnz_per_chunk {
                break;
            }
            row_end += 1;
        }
        if row_end == row_start {
            row_end += 1;
        }
        chunks.push((row_start, row_end));
        row_start = row_end;
    }

    chunks
}

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
        if n < PARALLEL_SPMV_MIN_ROWS {
            mat.spmv(x, y);
            return;
        }

        let target_chunks = rayon::current_num_threads().max(1) * PARALLEL_TARGET_CHUNKS_PER_THREAD;
        let row_chunks = balanced_row_chunks(rp, n, target_chunks);

        let mut y_chunks: Vec<(usize, &mut [T])> = Vec::with_capacity(row_chunks.len());
        let mut rest = y;
        let mut current = 0;
        for (row_start, row_end) in row_chunks {
            debug_assert_eq!(row_start, current);
            let (chunk, tail) = rest.split_at_mut(row_end - row_start);
            y_chunks.push((row_start, chunk));
            rest = tail;
            current = row_end;
        }

        y_chunks.into_par_iter().for_each(|(row_start, y_chunk)| {
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

        if n < PARALLEL_SPMV_MIN_ROWS {
            mat.spmv_add(alpha, x, beta, y);
            return;
        }

        let target_chunks = rayon::current_num_threads().max(1) * PARALLEL_TARGET_CHUNKS_PER_THREAD;
        let row_chunks = balanced_row_chunks(rp, n, target_chunks);

        let mut y_chunks: Vec<(usize, &mut [T])> = Vec::with_capacity(row_chunks.len());
        let mut rest = y;
        let mut current = 0;
        for (row_start, row_end) in row_chunks {
            debug_assert_eq!(row_start, current);
            let (chunk, tail) = rest.split_at_mut(row_end - row_start);
            y_chunks.push((row_start, chunk));
            rest = tail;
            current = row_end;
        }

        y_chunks.into_par_iter().for_each(|(row_start, y_chunk)| {
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
        if x.len() < PARALLEL_VECTOR_MIN_LEN {
            crate::sparse::ops::axpy(alpha, x, y);
            return;
        }
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
        if x.len() < PARALLEL_VECTOR_MIN_LEN {
            crate::sparse::ops::axpby(alpha, x, beta, y);
            return;
        }
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
        if x.len() < PARALLEL_VECTOR_MIN_LEN {
            return crate::sparse::ops::dot(x, y);
        }
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
        if x.len() < PARALLEL_VECTOR_MIN_LEN {
            return crate::sparse::ops::norm2(x);
        }
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
