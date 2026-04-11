use std::time::Instant;

use linger::{
    iterative::ConjugateGradient,
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, SolverParams, VerboseLevel,
};

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0 {
            coo.push(i, i - 1, -1.0);
        }
        if i + 1 < n {
            coo.push(i, i + 1, -1.0);
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let q = q.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn summarize(mut samples_ms: Vec<f64>) -> (f64, f64, f64) {
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    let p50 = percentile(&samples_ms, 0.50);
    let p95 = percentile(&samples_ms, 0.95);
    (mean, p50, p95)
}

fn bench_spmv_1d_n5000() -> (f64, f64, f64) {
    let n = 5000;
    let a = laplacian_1d(n);
    let x = vec![1.0_f64; n];
    let mut y = vec![0.0_f64; n];

    for _ in 0..20 {
        a.spmv(&x, &mut y);
    }

    let rounds = 8;
    let reps_per_round = 200;
    let mut samples = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        let t0 = Instant::now();
        for _ in 0..reps_per_round {
            a.spmv(&x, &mut y);
        }
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        samples.push(dt / reps_per_round as f64);
    }

    summarize(samples)
}

fn bench_cg_1d_n1000() -> (f64, f64, f64) {
    let n = 1000;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0_f64; n]);
    let cg = ConjugateGradient::<f64>::default();
    let params = SolverParams {
        rtol: 1e-8,
        max_iter: 4000,
        verbose: VerboseLevel::Silent,
        ..Default::default()
    };

    for _ in 0..2 {
        let mut x = DenseVec::zeros(n);
        let _ = cg.solve(&a, None, &b, &mut x, &params);
    }

    let rounds = 6;
    let reps_per_round = 8;
    let mut samples = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        let t0 = Instant::now();
        for _ in 0..reps_per_round {
            let mut x = DenseVec::zeros(n);
            let result = cg
                .solve(&a, None, &b, &mut x, &params)
                .expect("perf_guard: CG solve failed");
            assert!(result.converged, "perf_guard: CG did not converge");
        }
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        samples.push(dt / reps_per_round as f64);
    }

    summarize(samples)
}

fn main() {
    let (spmv_mean_ms, spmv_p50_ms, spmv_p95_ms) = bench_spmv_1d_n5000();
    let (cg_mean_ms, cg_p50_ms, cg_p95_ms) = bench_cg_1d_n1000();

    // Keep legacy keys for backward compatibility while adding robust quantiles.
    println!("spmv_1d_n5000_ms={spmv_mean_ms:.6}");
    println!("cg_1d_n1000_ms={cg_mean_ms:.6}");

    println!("spmv_1d_n5000_p50_ms={spmv_p50_ms:.6}");
    println!("spmv_1d_n5000_p95_ms={spmv_p95_ms:.6}");
    println!("cg_1d_n1000_p50_ms={cg_p50_ms:.6}");
    println!("cg_1d_n1000_p95_ms={cg_p95_ms:.6}");
}
