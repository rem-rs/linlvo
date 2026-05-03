//! Performance comparison benchmark: SIMD SpMV vs Scalar
//!
//! This example shows the performance difference between SIMD-accelerated
//! and scalar SpMV implementations.

use linger::sparse::{CooMatrix, CsrMatrix};
use std::time::Instant;

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

fn benchmark_spmv(name: &str, mat: &CsrMatrix<f64>, x: &[f64], iterations: usize) -> (f64, f64) {
    let mut y = vec![0.0_f64; mat.nrows()];

    // Warmup
    for _ in 0..3 {
        mat.spmv(x, &mut y);
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        mat.spmv(x, &mut y);
    }
    let elapsed = start.elapsed();

    let time_per_call_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let gflop_s = (mat.nnz() as f64 * 2.0 * iterations as f64) / elapsed.as_secs_f64() / 1e9;

    println!(
        "{:20} | {:.2} µs/call | {:.2} Gflop/s | NNZ={:6} | Rows={:5}",
        name, time_per_call_us, gflop_s, mat.nnz(), mat.nrows()
    );

    (time_per_call_us, gflop_s)
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║           SIMD SpMV Performance Comparison (AVX2)                      ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    #[cfg(target_arch = "x86_64")]
    {
        println!("CPU Features:");
        println!("  AVX2:   {}", if is_x86_feature_detected!("avx2") { "✓" } else { "✗" });
        println!();
    }

    println!("{:20} | {:10} | {:10} | {:7} | {:7}", "Test Case", "Time/call", "Gflop/s", "NNZ", "Rows");
    println!("{}", "─".repeat(72));

    // 1D Poisson - varying sizes
    println!("\n1D Poisson Problems:");
    for &n in &[500, 1000, 5000, 10000, 50000] {
        let mat = make_poisson_1d(n);
        let x = vec![1.0_f64; n];
        benchmark_spmv(&format!("1D n={}", n), &mat, &x, 100);
    }

    // 2D Poisson - varying sizes
    println!("\n2D Poisson Problems:");
    for &n in &[16, 32, 64, 128] {
        let mat = make_poisson_2d(n);
        let nn = n * n;
        let x = vec![1.0_f64; nn];
        benchmark_spmv(&format!("2D {}x{}", n, n), &mat, &x, 50);
    }

    // Random-like sparsity
    println!("\nBand Matrix (similar to 1D but wider):");
    let n = 10000;
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        for j in i.saturating_sub(2)..=(i + 2).min(n - 1) {
            if i != j {
                coo.push(i, j, 0.1);
            }
        }
        coo.push(i, i, 4.0);
    }
    let mat = CsrMatrix::from_coo(&coo);
    let x = vec![1.0_f64; n];
    benchmark_spmv(&format!("Band n={}", n), &mat, &x, 50);

    println!("\n{}", "═".repeat(72));
    println!("Note: Times and Gflop/s are measured with SIMD enabled (if available).");
    println!("Use `cargo run --example simd_diag` for CPU feature detection.");
}
