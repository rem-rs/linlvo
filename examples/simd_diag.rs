//! Diagnostic tool to check SIMD feature detection and basic performance.

use linger::sparse::{CooMatrix, CsrMatrix};

fn main() {
    println!("=== SIMD SpMV Diagnostics ===\n");

    // Check CPU features
    #[cfg(target_arch = "x86_64")]
    {
        println!("CPU Feature Detection:");
        println!("  AVX2:   {}", if is_x86_feature_detected!("avx2") { "✓ YES" } else { "✗ NO" });
        println!("  AVX512: {}", if is_x86_feature_detected!("avx512f") { "✓ YES" } else { "✗ NO" });
        println!("  SSE4.2: {}", if is_x86_feature_detected!("sse4.2") { "✓ YES" } else { "✗ NO" });
        println!();
    }

    // Create test matrices
    let n = 1000;
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0 { coo.push(i, i - 1, -1.0); }
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    let mat = CsrMatrix::from_coo(&coo);

    println!("Test Matrix (1D Poisson, n={}):", n);
    println!("  Rows:  {}", mat.nrows());
    println!("  Cols:  {}", mat.ncols());
    println!("  NNZ:   {}", mat.nnz());
    println!("  Density: {:.2}%", (mat.nnz() as f64 / (mat.nrows() as f64 * mat.ncols() as f64)) * 100.0);
    println!();

    // Run timing tests
    println!("SpMV Performance (10 iterations):");
    let x = vec![1.0_f64; n];
    let mut y = vec![0.0_f64; n];

    let start = std::time::Instant::now();
    for _ in 0..10 {
        mat.spmv(&x, &mut y);
    }
    let elapsed = start.elapsed();

    println!("  Total time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Per call:   {:.3} µs", elapsed.as_secs_f64() * 1_000_000.0 / 10.0);
    println!("  Gflop/s:    {:.2}", (mat.nnz() as f64 * 2.0 * 10.0) / elapsed.as_secs_f64() / 1e9);
    println!();

    println!("Result verification:");
    println!("  y[0]   = {:.6}", y[0]);
    println!("  y[n/2] = {:.6}", y[n / 2]);
    println!("  y[n-1] = {:.6}", y[n - 1]);
}
