//! Criterion benchmarks for AMG setup and solve phases.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linger::{
    amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, CycleType},
    iterative::ConjugateGradient,
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, SolverParams, VerboseLevel,
};

#[path = "baseline.rs"]
mod baseline;

const AMG_SETUP_1D_SIZES: [usize; 3] = [100, 500, 1000];
const AMG_SETUP_2D_GRIDS: [usize; 2] = [16, 32];
const AMG_PCG_1D_SIZES: [usize; 3] = [100, 500, 1000];
const AMG_PCG_2D_GRIDS: [usize; 2] = [16, 32];
const AMG_VCYCLE_1D_SIZES: [usize; 2] = [200, 1000];

fn emit_baseline_manifest() {
    baseline::print_baseline_manifest(&[
        "BASELINE|bench=amg|group=amg_setup|cases=[SA_1d,RS_1d]|sizes=[100,500,1000]",
        "BASELINE|bench=amg|group=amg_setup|case=SA_2d|grid_sizes=[16,32]",
        "BASELINE|bench=amg|group=amg_pcg_solve|case=SA_1d|sizes=[100,500,1000]",
        "BASELINE|bench=amg|group=amg_pcg_solve|case=SA_2d|grid_sizes=[16,32]",
        "BASELINE|bench=amg|group=amg_vcycle|case=V_1d|sizes=[200,1000]",
    ]);
}

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
    emit_baseline_manifest();
    let mut group = c.benchmark_group("amg_setup");

    for &n in &AMG_SETUP_1D_SIZES {
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

    for &n in &AMG_SETUP_2D_GRIDS {
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
    emit_baseline_manifest();
    let mut group = c.benchmark_group("amg_pcg_solve");
    let p = params(1e-8, 500);

    for &n in &AMG_PCG_1D_SIZES {
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
    for &n in &AMG_PCG_2D_GRIDS {
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
    emit_baseline_manifest();
    let mut group = c.benchmark_group("amg_vcycle");

    for &n in &AMG_VCYCLE_1D_SIZES {
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
