//! Criterion benchmarks for AMG setup and solve phases.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linger::{
    amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, CycleType},
    iterative::ConjugateGradient,
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, SolverParams, VerboseLevel,
};

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
            if i > 0   { coo.push(row, (i-1)*n+j, -1.0); }
            if i < n-1 { coo.push(row, (i+1)*n+j, -1.0); }
            if j > 0   { coo.push(row, i*n+j-1,   -1.0); }
            if j < n-1 { coo.push(row, i*n+j+1,   -1.0); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn params(rtol: f64, max_iter: usize) -> SolverParams {
    SolverParams { rtol, max_iter, verbose: VerboseLevel::Silent, ..Default::default() }
}

// ─── AMG setup phase ─────────────────────────────────────────────────────────

fn bench_amg_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("amg_setup");

    for &n in &[100usize, 500, 1000] {
        let a = make_poisson_1d(n);
        let config_sa = AmgConfig { coarse_threshold: 4, ..Default::default() };
        let config_rs = AmgConfig {
            strategy: CoarsenStrategy::RugeStüben,
            coarse_threshold: 4,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("SA_1d", n), &n, |b, _| {
            b.iter(|| {
                let _ = AmgHierarchy::build(black_box(a.clone()), config_sa.clone());
            });
        });

        group.bench_with_input(BenchmarkId::new("RS_1d", n), &n, |b, _| {
            b.iter(|| {
                let _ = AmgHierarchy::build(black_box(a.clone()), config_rs.clone());
            });
        });
    }

    for &n in &[16usize, 32] {
        let a = make_poisson_2d(n);
        let config = AmgConfig { coarse_threshold: 4, ..Default::default() };
        group.bench_with_input(BenchmarkId::new("SA_2d", n), &n, |b, _| {
            b.iter(|| {
                let _ = AmgHierarchy::build(black_box(a.clone()), config.clone());
            });
        });
    }

    group.finish();
}

// ─── AMG-PCG solve ────────────────────────────────────────────────────────────

fn bench_amg_pcg(c: &mut Criterion) {
    let mut group = c.benchmark_group("amg_pcg_solve");
    let p = params(1e-8, 500);

    for &n in &[100usize, 500, 1000] {
        let a = make_poisson_1d(n);
        let b = DenseVec::from_vec(vec![1.0f64; n]);

        // Pre-build hierarchy outside the timed loop.
        let config  = AmgConfig { coarse_threshold: 4, ..Default::default() };
        let hier    = AmgHierarchy::build(a.clone(), config);
        let precond = AmgPrecond::new(hier);
        let cg      = ConjugateGradient::<f64>::default();

        group.bench_with_input(BenchmarkId::new("SA_1d", n), &n, |bench, _| {
            bench.iter(|| {
                let mut x = DenseVec::zeros(n);
                cg.solve(black_box(&a), Some(&precond), black_box(&b), black_box(&mut x), &p).unwrap();
            });
        });
    }

    // 2D Poisson.
    for &n in &[16usize, 32] {
        let a  = make_poisson_2d(n);
        let nn = n * n;
        let b  = DenseVec::from_vec(vec![1.0f64; nn]);

        let config  = AmgConfig { coarse_threshold: 4, ..Default::default() };
        let hier    = AmgHierarchy::build(a.clone(), config);
        let precond = AmgPrecond::new(hier);
        let cg      = ConjugateGradient::<f64>::default();

        group.bench_with_input(BenchmarkId::new("SA_2d", n), &n, |bench, _| {
            bench.iter(|| {
                let mut x = DenseVec::zeros(nn);
                cg.solve(black_box(&a), Some(&precond), black_box(&b), black_box(&mut x), &p).unwrap();
            });
        });
    }

    group.finish();
}

// ─── V-cycle throughput ───────────────────────────────────────────────────────

fn bench_vcycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("amg_vcycle");

    for &n in &[200usize, 1000] {
        let a      = make_poisson_1d(n);
        let b      = DenseVec::from_vec(vec![1.0f64; n]);
        let config = AmgConfig { coarse_threshold: 4, ..Default::default() };
        let hier   = AmgHierarchy::build(a, config);

        group.bench_with_input(BenchmarkId::new("V_1d", n), &n, |bench, _| {
            bench.iter(|| {
                let mut x = DenseVec::zeros(n);
                hier.apply_cycle(black_box(&b), black_box(&mut x), CycleType::V);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_amg_setup, bench_amg_pcg, bench_vcycle);
criterion_main!(benches);
