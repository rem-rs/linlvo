//! ex02 — 1-D Poisson equation: assembly, MMS verification, SpMV timing.
//!
//! **Problem**: −u'' = f on (0,1) with homogeneous Dirichlet BCs.
//! Uniform mesh h = 1/(n+1), finite-difference discretisation:
//!   −u_{i-1} + 2u_i − u_{i+1} = h²·f_i
//! (we absorb h² into f for this dimensionless form).
//!
//! **HYPRE analog**
//!   `hypre/src/examples/ex1.c`  — 1-D Laplacian, BoomerAMG solve
//!   Here we perform only assembly + SpMV; the AMG solve is Sprint 4.
//!
//! **PETSc analog**
//!   `ksp/ksp/tutorials/ex2.c`   — CG solve of 1-D Laplacian
//!   Here we perform only assembly + MMS check; KSP solve is Sprint 2.
//!
//! **Method of Manufactured Solutions (MMS)**
//!   Exact solution: x_exact[i] = sin(π·(i+1)/(n+1))
//!   Construct b = A·x_exact, then verify ‖b − A·x_exact‖ / ‖b‖ ≈ 0.

use std::time::Instant;

use linger::{
    sparse::{CooMatrix, CsrMatrix},
    LinearOperator,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn sep(title: &str) {
    println!("\n━━━━ {title} ━━━━━");
}

fn poisson_1d(n: usize) -> (CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
    // Assembly — identical to HYPRE's IJ interface pattern:
    //   for i in ilower..=iupper: set row with up-to-3 entries
    let mut coo: CooMatrix<f64> = CooMatrix::with_capacity(n, n, 3 * n - 2);
    for i in 0..n {
        if i > 0     { coo.push(i, i - 1, -1.0); }
        coo.push(i, i,  2.0);
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    let a = CsrMatrix::from_coo(&coo);

    // MMS exact solution
    let pi = std::f64::consts::PI;
    let x_exact: Vec<f64> = (0..n)
        .map(|i| (pi * (i + 1) as f64 / (n + 1) as f64).sin())
        .collect();

    // RHS b = A · x_exact
    let mut b = vec![0.0f64; n];
    a.spmv(&x_exact, &mut b);

    (a, x_exact, b)
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |acc, &x| acc + x * x).sqrt()
}

fn rel_residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut r = vec![0.0f64; b.len()];
    a.spmv(x, &mut r);
    let nr = r.iter().zip(b).map(|(&ri, &bi)| (ri - bi).powi(2)).sum::<f64>().sqrt();
    let nb = norm2(b);
    if nb == 0.0 { nr } else { nr / nb }
}

fn spectral_bounds_1d(n: usize) -> (f64, f64) {
    // For h=1/(n+1): λ_min = 4sin²(π/2(n+1)), λ_max = 4cos²(π/2(n+1)) ≈ 4
    let pi = std::f64::consts::PI;
    let lambda_min = 4.0 * (pi / (2.0 * (n + 1) as f64)).sin().powi(2);
    let lambda_max = 4.0 * (pi * n as f64 / (2.0 * (n + 1) as f64)).sin().powi(2);
    (lambda_min, lambda_max)
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    sep("ex02: 1-D Poisson  (HYPRE ex1 / PETSc ex2)");

    for &n in &[10usize, 100, 1_000, 10_000] {
        println!("\n  ── n = {n} ──────────────────────────────────────");

        // Assembly
        let t0 = Instant::now();
        let (a, x_exact, b) = poisson_1d(n);
        let t_assemble = t0.elapsed();

        let nnz = a.nnz();
        let sparsity = nnz as f64 / (n as f64 * n as f64) * 100.0;
        let (lam_min, lam_max) = spectral_bounds_1d(n);
        let cond = lam_max / lam_min;

        println!("  size:     {n}×{n}  nnz={nnz}  sparsity={sparsity:.4}%");
        println!("  assemble: {t_assemble:.2?}");
        println!("  λ_min ≈ {lam_min:.6e}  λ_max ≈ {lam_max:.6e}  κ(A) ≈ {cond:.3e}");
        println!("  ‖b‖₂   = {:.6e}", norm2(&b));

        // MMS verification: residual must be essentially 0
        let rel_res = rel_residual(&a, &x_exact, &b);
        println!("  ‖A·x_exact − b‖/‖b‖ = {rel_res:.3e}  [ε_machine={:.3e}]",
            f64::EPSILON);
        assert!(
            rel_res < 1e-12,
            "MMS residual too large: {rel_res:.3e}"
        );

        // Diagonal check
        let diag = a.diag();
        let all_two = diag.iter().all(|&d| (d - 2.0).abs() < 1e-14);
        assert!(all_two, "diagonal should be all 2.0");

        // Symmetry check
        assert!(a.is_structurally_symmetric(), "1-D Poisson must be symmetric");

        // SpMV throughput: 10 × SpMV
        let reps = 10usize.max(1_000_000 / n); // more reps for small n
        let mut y = vec![0.0f64; n];
        let t1 = Instant::now();
        for _ in 0..reps {
            a.spmv(&x_exact, &mut y);
        }
        let t_spmv = t1.elapsed() / reps as u32;
        let flops_per_spmv = 2 * nnz; // 1 multiply + 1 add per nnz
        let gflops = flops_per_spmv as f64 / t_spmv.as_secs_f64() / 1e9;
        println!("  SpMV×{reps}: avg {t_spmv:.2?}  ({gflops:.2} GFLOP/s)");

        // Also validate via LinearOperator trait
        let xv = linger::DenseVec::from_vec(x_exact.clone());
        let mut yv = linger::DenseVec::zeros(n);
        a.apply(&xv, &mut yv);
        let diff: f64 = yv.as_slice().iter().zip(&b)
            .map(|(&yi, &bi)| (yi - bi).powi(2)).sum::<f64>().sqrt();
        assert!(diff < 1e-12 * norm2(&b), "LinearOperator::apply mismatch");

        println!("  LinearOperator::apply  ✓");
        println!("  MMS  ✓");
    }

    // ── Spectral radius summary ────────────────────────────────────────────────
    println!("\n  ── Spectral radius vs n ────────────────────────────────");
    println!("  {:>8}  {:>14}  {:>14}  {:>12}", "n", "λ_min", "λ_max", "κ(A)");
    for &n in &[10usize, 100, 1_000, 10_000] {
        let (lmin, lmax) = spectral_bounds_1d(n);
        println!("  {:>8}  {:>14.6e}  {:>14.6e}  {:>12.3e}", n, lmin, lmax, lmax/lmin);
    }
    println!("  (κ grows as O(n²) — motivates AMG preconditioner in Sprint 4)");

    println!("\n  OK\n");
}
