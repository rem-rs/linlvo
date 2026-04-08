//! Criterion benchmarks for sparse direct solvers.
//!
//! Measures factorization and solve time for SparseLu, SparseCholesky, and
//! MultifrontalLu across different problem sizes and orderings.
//!
//! Small-scale (n ≤ 400): all three solvers, various orderings.
//! Medium-scale (n = 1000–5000): SparseCholesky and MultifrontalLu with RCM/NodeNd.
//! Large-scale 2D (grid ≤ 70×70 ≈ 4900 DOF): ordering comparison on structured grids.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use linger::{
    direct::{
        DirectSolver, DirectOptions, SparseLu, SparseCholesky, MultifrontalLu, MultifrontalOptions,
        ordering::OrderingMethod,
    },
    sparse::{CooMatrix, CsrMatrix},
    DenseVec,
};

// ─── matrix generators ────────────────────────────────────────────────────────

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn laplacian_2d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut coo = CooMatrix::new(nn, nn);
    for i in 0..n {
        for j in 0..n {
            let id = i * n + j;
            coo.push(id, id, 4.0);
            if j > 0 { coo.push(id, id - 1, -1.0); coo.push(id - 1, id, -1.0); }
            if i > 0 { coo.push(id, id - n, -1.0); coo.push(id - n, id, -1.0); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn ones_rhs(n: usize) -> DenseVec<f64> {
    DenseVec::from_vec(vec![1.0f64; n])
}

// ─── SparseLu factorization ───────────────────────────────────────────────────

fn bench_lu_factorize(c: &mut Criterion) {
    let mut group = c.benchmark_group("SparseLu/factorize");

    for &n in &[50usize, 100, 200] {
        let a = laplacian_1d(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("1D/n", n), &n, |b, _| {
            b.iter(|| {
                let mut s = SparseLu::<f64>::new(DirectOptions {
                    ordering: OrderingMethod::Rcm,
                    ..Default::default()
                });
                s.factor(black_box(&a)).unwrap();
                black_box(s)
            });
        });
    }

    // 2D Laplacian (more fill, exercises ordering impact).
    for &n in &[8usize, 12, 16] {
        let a = laplacian_2d(n);
        let nn = n * n;
        group.throughput(Throughput::Elements(nn as u64));
        group.bench_with_input(BenchmarkId::new("2D/n", n), &n, |b, _| {
            b.iter(|| {
                let mut s = SparseLu::<f64>::new(DirectOptions {
                    ordering: OrderingMethod::Rcm,
                    ..Default::default()
                });
                s.factor(black_box(&a)).unwrap();
                black_box(s)
            });
        });
    }

    group.finish();
}

// ─── SparseLu solve ───────────────────────────────────────────────────────────

fn bench_lu_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("SparseLu/solve");

    for &n in &[50usize, 100, 200] {
        let a = laplacian_1d(n);
        let b = ones_rhs(n);
        let mut solver = SparseLu::<f64>::new(DirectOptions {
            ordering: OrderingMethod::Rcm,
            ..Default::default()
        });
        solver.factor(&a).unwrap();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("1D/n", n), &n, |bench, _| {
            bench.iter(|| {
                let mut x = DenseVec::zeros(n);
                solver.solve(black_box(&b), black_box(&mut x)).unwrap();
                black_box(x)
            });
        });
    }

    group.finish();
}

// ─── SparseCholesky factorization ─────────────────────────────────────────────

fn bench_cholesky_factorize(c: &mut Criterion) {
    let mut group = c.benchmark_group("SparseCholesky/factorize");

    for &n in &[50usize, 100, 200, 400] {
        let a = laplacian_1d(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("1D/n", n), &n, |b, _| {
            b.iter(|| {
                let mut s = SparseCholesky::<f64>::new(DirectOptions {
                    ordering: OrderingMethod::Rcm,
                    ..Default::default()
                });
                s.factor(black_box(&a)).unwrap();
                black_box(s)
            });
        });
    }

    // 2D Laplacian.
    for &n in &[8usize, 12, 16] {
        let a = laplacian_2d(n);
        let nn = n * n;
        group.throughput(Throughput::Elements(nn as u64));
        group.bench_with_input(BenchmarkId::new("2D/n", n), &n, |b, _| {
            b.iter(|| {
                let mut s = SparseCholesky::<f64>::new(DirectOptions {
                    ordering: OrderingMethod::Rcm,
                    ..Default::default()
                });
                s.factor(black_box(&a)).unwrap();
                black_box(s)
            });
        });
    }

    group.finish();
}

// ─── MultifrontalLu factorization ─────────────────────────────────────────────

fn bench_multifrontal_factorize(c: &mut Criterion) {
    let mut group = c.benchmark_group("MultifrontalLu/factorize");

    for &n in &[30usize, 60, 100] {
        let a = laplacian_1d(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("1D/n", n), &n, |b, _| {
            b.iter(|| {
                let mut s = MultifrontalLu::<f64>::with_options(MultifrontalOptions {
                    base: DirectOptions {
                        ordering: OrderingMethod::Rcm,
                        ..Default::default()
                    },
                    ..Default::default()
                });
                s.factor(black_box(&a)).unwrap();
                black_box(s)
            });
        });
    }

    group.finish();
}

// ─── Ordering comparison ──────────────────────────────────────────────────────

fn bench_ordering_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cholesky/ordering");
    let n = 16;
    let a = laplacian_2d(n);
    let nn = n * n;
    group.throughput(Throughput::Elements(nn as u64));

    for (name, ordering) in &[
        ("Natural", OrderingMethod::Natural),
        ("Rcm",     OrderingMethod::Rcm),
        ("Colamd",  OrderingMethod::Colamd),
        ("NodeNd",  OrderingMethod::NodeNd),
    ] {
        let ord = ordering.clone();
        group.bench_with_input(BenchmarkId::new("2D", name), name, |b, _| {
            b.iter(|| {
                let mut s = SparseCholesky::<f64>::new(DirectOptions {
                    ordering: ord.clone(),
                    ..Default::default()
                });
                s.factor(black_box(&a)).unwrap();
                black_box(s)
            });
        });
    }

    group.finish();
}

// ─── Iterative refinement overhead ────────────────────────────────────────────

fn bench_refinement_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("SparseLu/refinement");
    let n = 100;
    let a = laplacian_1d(n);
    let b = ones_rhs(n);

    for &steps in &[0usize, 1, 2, 3] {
        group.bench_with_input(BenchmarkId::new("refine_steps", steps), &steps, |bench, &s| {
            let mut solver = SparseLu::<f64>::new(DirectOptions {
                ordering: OrderingMethod::Rcm,
                refine_steps: s,
                ..Default::default()
            });
            solver.factor(&a).unwrap();
            bench.iter(|| {
                let mut x = DenseVec::zeros(n);
                solver.solve(black_box(&b), black_box(&mut x)).unwrap();
                black_box(x)
            });
        });
    }

    group.finish();
}

// ─── SparseCholesky medium-scale (1D Laplacian, n = 500–5000) ────────────────

fn bench_cholesky_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("SparseCholesky/medium");
    // Use longer measurement time for larger problems.
    group.sample_size(10);

    for &n in &[500usize, 1000, 2000, 5000] {
        let a = laplacian_1d(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("1D_Rcm/n", n), &n, |b, _| {
            b.iter(|| {
                let mut s = SparseCholesky::<f64>::new(DirectOptions {
                    ordering: OrderingMethod::Rcm,
                    ..Default::default()
                });
                s.factor(black_box(&a)).unwrap();
                black_box(s)
            });
        });
        group.bench_with_input(BenchmarkId::new("1D_NodeNd/n", n), &n, |b, _| {
            b.iter(|| {
                let mut s = SparseCholesky::<f64>::new(DirectOptions {
                    ordering: OrderingMethod::NodeNd,
                    ..Default::default()
                });
                s.factor(black_box(&a)).unwrap();
                black_box(s)
            });
        });
    }

    group.finish();
}

// ─── SparseCholesky medium-scale solve (factor + solve together) ──────────────

fn bench_cholesky_solve_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("SparseCholesky/solve_medium");
    group.sample_size(10);

    for &n in &[500usize, 1000, 2000, 5000] {
        let a = laplacian_1d(n);
        let b = ones_rhs(n);
        let mut solver = SparseCholesky::<f64>::new(DirectOptions {
            ordering: OrderingMethod::Rcm,
            ..Default::default()
        });
        solver.factor(&a).unwrap();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("1D_Rcm/n", n), &n, |bench, _| {
            bench.iter(|| {
                let mut x = DenseVec::zeros(n);
                solver.solve(black_box(&b), black_box(&mut x)).unwrap();
                black_box(x)
            });
        });
    }

    group.finish();
}

// ─── MultifrontalLu medium-scale (1D Laplacian, n = 200–2000) ────────────────

fn bench_multifrontal_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("MultifrontalLu/medium");
    group.sample_size(10);

    for &n in &[200usize, 500, 1000, 2000] {
        let a = laplacian_1d(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("1D_Rcm/n", n), &n, |b, _| {
            b.iter(|| {
                let mut s = MultifrontalLu::<f64>::with_options(MultifrontalOptions {
                    base: DirectOptions {
                        ordering: OrderingMethod::Rcm,
                        ..Default::default()
                    },
                    ..Default::default()
                });
                s.factor(black_box(&a)).unwrap();
                black_box(s)
            });
        });
    }

    group.finish();
}

// ─── Ordering comparison on medium 2D grids ──────────────────────────────────

fn bench_ordering_medium_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cholesky/ordering_medium_2d");
    group.sample_size(10);

    // Grid sizes: 20×20=400, 32×32=1024, 50×50=2500, 70×70=4900
    for &n in &[20usize, 32, 50, 70] {
        let a = laplacian_2d(n);
        let nn = n * n;
        group.throughput(Throughput::Elements(nn as u64));

        for (name, ordering) in &[
            ("Rcm",    OrderingMethod::Rcm),
            ("Colamd", OrderingMethod::Colamd),
            ("NodeNd", OrderingMethod::NodeNd),
        ] {
            let ord = ordering.clone();
            group.bench_with_input(
                BenchmarkId::new(format!("grid{}x{}", n, n), name),
                name,
                |b, _| {
                    b.iter(|| {
                        let mut s = SparseCholesky::<f64>::new(DirectOptions {
                            ordering: ord.clone(),
                            ..Default::default()
                        });
                        s.factor(black_box(&a)).unwrap();
                        black_box(s)
                    });
                },
            );
        }
    }

    group.finish();
}

// ─── Criterion entry points ───────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_lu_factorize,
    bench_lu_solve,
    bench_cholesky_factorize,
    bench_cholesky_medium,
    bench_multifrontal_factorize,
    bench_multifrontal_medium,
    bench_ordering_comparison,
    bench_ordering_medium_2d,
    bench_refinement_overhead,
    bench_cholesky_solve_medium,
);
criterion_main!(benches);
