//! Benchmark for SIMD SpMV vs scalar implementations.
//!
//! Compares performance of SIMD-accelerated sparse matrix-vector product
//! with scalar fallback implementation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use linger::sparse::{CooMatrix, CsrMatrix};

#[path = "baseline.rs"]
mod baseline;

fn make_poisson_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0 { coo.push(i, i - 1, -1.0); }
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
            if i > 0 { coo.push(row, (i - 1) * n + j, -1.0); }
            if i < n - 1 { coo.push(row, (i + 1) * n + j, -1.0); }
            if j > 0 { coo.push(row, i * n + j - 1, -1.0); }
            if j < n - 1 { coo.push(row, i * n + j + 1, -1.0); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn bench_spmv_simd(c: &mut Criterion) {
    // 1D Poisson matrices at various sizes to test SIMD effectiveness
    let mut group = c.benchmark_group("spmv_simd_1d");
    group.throughput(Throughput::Bytes(500 * 8 * 2)); // Rough estimate

    for &n in &[500, 1000, 5000, 10000] {
        let a = black_box(make_poisson_1d(n));
        let x = black_box(vec![1.0_f64; n]);
        let mut y = vec![0.0_f64; n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                a.spmv(&x, &mut y);
                black_box(&y);
            });
        });
    }
    group.finish();

    // 2D Poisson matrices (more complex sparsity pattern)
    let mut group = c.benchmark_group("spmv_simd_2d");
    for &n in &[16, 32, 64] {
        let nn = n * n;
        let a = black_box(make_poisson_2d(n));
        let x = black_box(vec![1.0_f64; nn]);
        let mut y = vec![0.0_f64; nn];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", n, n)),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    a.spmv(&x, &mut y);
                    black_box(&y);
                });
            },
        );
    }
    group.finish();
}

fn bench_spmv_add_simd(c: &mut Criterion) {
    // Benchmark for the α·A·x + β·y variant
    let mut group = c.benchmark_group("spmv_add_simd");

    for &n in &[1000, 5000, 10000] {
        let a = black_box(make_poisson_1d(n));
        let x = black_box(vec![1.0_f64; n]);
        let mut y = vec![2.0_f64; n];

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                a.spmv_add(2.0, &x, 0.5, &mut y);
                black_box(&y);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_spmv_simd, bench_spmv_add_simd);
criterion_main!(benches);
