//! Criterion benchmarks for Krylov solvers and preconditioners.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use linger::{
    direct::{DirectSolverPrecond, SparseLdlt},
    iterative::{BiCgStab, ConjugateGradient, Gmres},
    precond::{Icc0Precond, Ilu0Precond, IlukPrecond, IldltPrecond, JacobiPrecond},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, SolverParams, VerboseLevel,
};

#[path = "baseline.rs"]
mod baseline;

const SOLVER_SIZES: [usize; 3] = [100, 500, 1000];
const PCG_PRECOND_SIZE: usize = 500;

fn emit_baseline_manifest() {
    baseline::print_baseline_manifest(&[
        "BASELINE|bench=krylov|group=krylov_solvers|solvers=[CG,GMRES30,BiCGSTAB]|sizes=[100,500,1000]",
        "BASELINE|bench=krylov|group=pcg_preconditioners|n=500|preconds=[none,jacobi,ilu0,ilu1,icc0,ildlt0,ldlt_exact]",
    ]);
}

fn make_poisson_1d(n: usize) -> (CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b = vec![1.0f64; n];
    let x = vec![0.0f64; n];
    (a, b, x)
}

fn solver_params(rtol: f64, max_iter: usize) -> SolverParams {
    SolverParams { rtol, max_iter, verbose: VerboseLevel::Silent, ..Default::default() }
}

// ─── Solver comparison ────────────────────────────────────────────────────────

fn bench_solvers(c: &mut Criterion) {
    emit_baseline_manifest();
    let mut group = c.benchmark_group("krylov_solvers");
    let params = solver_params(1e-8, 2000);

    for &n in &SOLVER_SIZES {
        let (a, b_vec, _) = make_poisson_1d(n);
        let b = DenseVec::from_vec(b_vec);

        group.bench_with_input(BenchmarkId::new("CG", n), &n, |bench, _| {
            bench.iter(|| {
                let mut x = DenseVec::zeros(n);
                ConjugateGradient::<f64>::default()
                    .solve(black_box(&a), None, black_box(&b), black_box(&mut x), &params)
                    .unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("GMRES30", n), &n, |bench, _| {
            bench.iter(|| {
                let mut x = DenseVec::zeros(n);
                Gmres::<f64>::new(30)
                    .solve(black_box(&a), None, black_box(&b), black_box(&mut x), &params)
                    .unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("BiCGSTAB", n), &n, |bench, _| {
            bench.iter(|| {
                let mut x = DenseVec::zeros(n);
                BiCgStab::<f64>::new()
                    .solve(black_box(&a), None, black_box(&b), black_box(&mut x), &params)
                    .unwrap();
            });
        });
    }

    group.finish();
}

// ─── Preconditioner quality (iteration counts) ────────────────────────────────

fn bench_preconditioners(c: &mut Criterion) {
    emit_baseline_manifest();
    let mut group = c.benchmark_group("pcg_preconditioners");
    let params = solver_params(1e-8, 2000);
    let n = PCG_PRECOND_SIZE;
    let (a, b_vec, _) = make_poisson_1d(n);
    let b = DenseVec::from_vec(b_vec);

    group.bench_function("none", |bench| {
        bench.iter(|| {
            let mut x = DenseVec::zeros(n);
            ConjugateGradient::<f64>::default()
                .solve(black_box(&a), None, black_box(&b), black_box(&mut x), &params)
                .unwrap();
        });
    });

    group.bench_function("jacobi", |bench| {
        let jac = JacobiPrecond::<f64>::from_csr(&a).unwrap();
        bench.iter(|| {
            let mut x = DenseVec::zeros(n);
            ConjugateGradient::<f64>::default()
                .solve(black_box(&a), Some(&jac), black_box(&b), black_box(&mut x), &params)
                .unwrap();
        });
    });

    group.bench_function("ilu0", |bench| {
        let ilu = Ilu0Precond::<f64>::from_csr(&a).unwrap();
        bench.iter(|| {
            let mut x = DenseVec::zeros(n);
            ConjugateGradient::<f64>::default()
                .solve(black_box(&a), Some(&ilu), black_box(&b), black_box(&mut x), &params)
                .unwrap();
        });
    });

    group.bench_function("ilu1", |bench| {
        let ilu = IlukPrecond::<f64>::from_csr(&a, 1).unwrap();
        bench.iter(|| {
            let mut x = DenseVec::zeros(n);
            ConjugateGradient::<f64>::default()
                .solve(black_box(&a), Some(&ilu), black_box(&b), black_box(&mut x), &params)
                .unwrap();
        });
    });

    group.bench_function("icc0", |bench| {
        let icc = Icc0Precond::<f64>::from_csr(&a).unwrap();
        bench.iter(|| {
            let mut x = DenseVec::zeros(n);
            ConjugateGradient::<f64>::default()
                .solve(black_box(&a), Some(&icc), black_box(&b), black_box(&mut x), &params)
                .unwrap();
        });
    });

    // ILDLᵀ(0) — incomplete LDLᵀ for symmetric matrices.
    group.bench_function("ildlt0", |bench| {
        let ildlt = IldltPrecond::<f64>::from_csr(&a).unwrap();
        bench.iter(|| {
            let mut x = DenseVec::zeros(n);
            ConjugateGradient::<f64>::default()
                .solve(black_box(&a), Some(&ildlt), black_box(&b), black_box(&mut x), &params)
                .unwrap();
        });
    });

    // SparseLdlt (exact sparse LDLᵀ) as preconditioner.
    group.bench_function("ldlt_exact", |bench| {
        let precond = DirectSolverPrecond::new(SparseLdlt::<f64>::default(), &a).unwrap();
        bench.iter(|| {
            let mut x = DenseVec::zeros(n);
            ConjugateGradient::<f64>::default()
                .solve(black_box(&a), Some(&precond), black_box(&b), black_box(&mut x), &params)
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_solvers, bench_preconditioners);
criterion_main!(benches);
