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
const PARALLEL_SPMV_MIN_NNZ: usize = 32_768;
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

#[cfg(feature = "rayon")]
#[inline(always)]
unsafe fn csr_row_dot_unchecked<T: Scalar>(
    col_idx: &[usize],
    values: &[T],
    x: &[T],
    start: usize,
    end: usize,
) -> T {
    // Use SIMD-accelerated dot product when available (SIMD module will dispatch)
    crate::simd::simd_row_dot(col_idx, values, x, start, end)
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
        if n < PARALLEL_SPMV_MIN_ROWS || vs.len() < PARALLEL_SPMV_MIN_NNZ {
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
                *yi = unsafe { csr_row_dot_unchecked(ci, vs, x, rp[i], rp[i + 1]) };
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

        if n < PARALLEL_SPMV_MIN_ROWS || vs.len() < PARALLEL_SPMV_MIN_NNZ {
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
                let sum = unsafe { csr_row_dot_unchecked(ci, vs, x, rp[i], rp[i + 1]) };
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
            crate::simd::dense_ops::simd_axpy(alpha, x, y);
            return;
        }
        // Split into equal chunks so each thread gets a contiguous slice,
        // then apply SIMD within each chunk (Rayon × SIMD two-level parallelism).
        let nthreads = rayon::current_num_threads().max(1);
        let chunk_size = ((x.len() + nthreads - 1) / nthreads).max(256);
        x.par_chunks(chunk_size)
            .zip(y.par_chunks_mut(chunk_size))
            .for_each(|(xc, yc)| {
                crate::simd::dense_ops::simd_axpy(alpha, xc, yc);
            });
        return;
    }

    #[cfg(not(feature = "rayon"))]
    crate::simd::dense_ops::simd_axpy(alpha, x, y);
}

/// Parallel `y = alpha * x + beta * y`.
pub fn parallel_axpby<T: Scalar + Send + Sync>(alpha: T, x: &[T], beta: T, y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        if x.len() < PARALLEL_VECTOR_MIN_LEN {
            crate::simd::dense_ops::simd_axpby(alpha, x, beta, y);
            return;
        }
        let nthreads = rayon::current_num_threads().max(1);
        let chunk_size = ((x.len() + nthreads - 1) / nthreads).max(256);
        x.par_chunks(chunk_size)
            .zip(y.par_chunks_mut(chunk_size))
            .for_each(|(xc, yc)| {
                crate::simd::dense_ops::simd_axpby(alpha, xc, beta, yc);
            });
        return;
    }

    #[cfg(not(feature = "rayon"))]
    crate::simd::dense_ops::simd_axpby(alpha, x, beta, y);
}

// ─── Dot product / norm ───────────────────────────────────────────────────────

/// Parallel Euclidean inner product `<x, y>`.
pub fn parallel_dot<T: Scalar + Send + Sync>(x: &[T], y: &[T]) -> T {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        if x.len() < PARALLEL_VECTOR_MIN_LEN {
            return crate::simd::dense_ops::simd_dot(x, y);
        }
        let nthreads = rayon::current_num_threads().max(1);
        let chunk_size = ((x.len() + nthreads - 1) / nthreads).max(256);
        return x.par_chunks(chunk_size)
            .zip(y.par_chunks(chunk_size))
            .map(|(xc, yc)| crate::simd::dense_ops::simd_dot(xc, yc))
            .reduce(|| T::zero(), |a, b| a + b);
    }

    #[cfg(not(feature = "rayon"))]
    crate::simd::dense_ops::simd_dot(x, y)
}

/// Parallel Euclidean 2-norm `√(Σ xᵢ²)`.
pub fn parallel_norm2<T: Scalar + Send + Sync>(x: &[T]) -> T {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        if x.len() < PARALLEL_VECTOR_MIN_LEN {
            return crate::simd::dense_ops::simd_norm2(x);
        }
        let nthreads = rayon::current_num_threads().max(1);
        let chunk_size = ((x.len() + nthreads - 1) / nthreads).max(256);
        // Compute sum-of-squares in parallel using SIMD per chunk, then sqrt once.
        let ss = x.par_chunks(chunk_size)
            .map(|xc| {
                // norm2^2 for each chunk via dot(xc, xc)
                let n = crate::simd::dense_ops::simd_norm2(xc);
                n * n
            })
            .reduce(|| T::zero(), |a, b| a + b);
        return ss.sqrt();
    }

    #[cfg(not(feature = "rayon"))]
    crate::simd::dense_ops::simd_norm2(x)
}
