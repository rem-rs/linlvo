//! Criterion benchmarks for SpMV operations.
//!
//! Measures serial CSR SpMV, parallel CSR SpMV, and BSR block SpMV
//! at several problem sizes representative of FEA applications.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use linger::{
    parallel::{parallel_spmv, parallel_spmv_add},
    sparse::{BsrBuilder, CooMatrix, CsrMatrix},
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn make_poisson_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn make_poisson_2d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut coo = CooMatrix::new(nn, nn);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            coo.push(row, row, 4.0);
            if i > 0     { coo.push(row, (i-1)*n+j, -1.0); }
            if i < n-1   { coo.push(row, (i+1)*n+j, -1.0); }
            if j > 0     { coo.push(row, i*n+j-1,   -1.0); }
            if j < n-1   { coo.push(row, i*n+j+1,   -1.0); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

/// Build a BSR matrix from a 2-DOF block Poisson-like stiffness system.
fn make_bsr_2dof(n_blocks: usize) -> linger::BsrMatrix<f64> {
    let r = 2;
    let c = 2;
    let mut builder = BsrBuilder::new(n_blocks, n_blocks, r, c);
    // Diagonal blocks = [4,-1; -1,4], off-diagonal = [-1,0; 0,-1]
    let diag_block = vec![4.0, -1.0, -1.0, 4.0];
    let off_block  = vec![-1.0, 0.0, 0.0, -1.0];
    for I in 0..n_blocks {
        builder.push_block(I, I, diag_block.clone());
        if I > 0          { builder.push_block(I, I - 1, off_block.clone()); }
        if I < n_blocks-1 { builder.push_block(I, I + 1, off_block.clone()); }
    }
    builder.build()
}

// ─── Serial CSR SpMV ─────────────────────────────────────────────────────────

fn bench_spmv_serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_serial_csr");

    for &n in &[100usize, 500, 1000, 5000] {
        let a = make_poisson_1d(n);
        let x = vec![1.0f64; n];
        let mut y = vec![0.0f64; n];

        group.throughput(Throughput::Elements(a.nnz() as u64));
        group.bench_with_input(BenchmarkId::new("1d_poisson", n), &n, |b, _| {
            b.iter(|| {
                a.spmv(black_box(&x), black_box(&mut y));
            });
        });
    }

    for &n in &[16usize, 32, 64] {
        let a = make_poisson_2d(n);
        let nn = n * n;
        let x  = vec![1.0f64; nn];
        let mut y = vec![0.0f64; nn];

        group.throughput(Throughput::Elements(a.nnz() as u64));
        group.bench_with_input(BenchmarkId::new("2d_poisson", n), &n, |b, _| {
            b.iter(|| {
                a.spmv(black_box(&x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

// ─── Parallel CSR SpMV ───────────────────────────────────────────────────────

fn bench_spmv_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_parallel_csr");

    for &n in &[1000usize, 5000, 10000] {
        let a = make_poisson_1d(n);
        let x = vec![1.0f64; n];
        let mut y = vec![0.0f64; n];

        group.throughput(Throughput::Elements(a.nnz() as u64));
        group.bench_with_input(BenchmarkId::new("1d_poisson", n), &n, |b, _| {
            b.iter(|| {
                parallel_spmv(black_box(&a), black_box(&x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

// ─── BSR SpMV ────────────────────────────────────────────────────────────────

fn bench_spmv_bsr(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_bsr");

    for &n_blocks in &[100usize, 500, 1000] {
        let bsr  = make_bsr_2dof(n_blocks);
        let cols = bsr.ncols();
        let rows = bsr.nrows();
        let x    = vec![1.0f64; cols];
        let mut y = vec![0.0f64; rows];

        group.throughput(Throughput::Elements(bsr.nnz_stored() as u64));
        group.bench_with_input(BenchmarkId::new("2dof_blocks", n_blocks), &n_blocks, |b, _| {
            b.iter(|| {
                bsr.spmv(black_box(&x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

// ─── BSR vs CSR comparison ────────────────────────────────────────────────────

fn bench_spmv_bsr_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_bsr_parallel");

    for &n_blocks in &[500usize, 2000] {
        let bsr  = make_bsr_2dof(n_blocks);
        let cols = bsr.ncols();
        let rows = bsr.nrows();
        let x    = vec![1.0f64; cols];
        let mut y = vec![0.0f64; rows];

        group.throughput(Throughput::Elements(bsr.nnz_stored() as u64));
        group.bench_with_input(BenchmarkId::new("2dof_blocks", n_blocks), &n_blocks, |b, _| {
            b.iter(|| {
                bsr.spmv_parallel(black_box(&x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_spmv_serial,
    bench_spmv_parallel,
    bench_spmv_bsr,
    bench_spmv_bsr_parallel,
);
criterion_main!(benches);
