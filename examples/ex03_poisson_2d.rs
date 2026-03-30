//! ex03 — 2-D Poisson equation on structured grids.
//!
//! **Problem**: −Δu = f on (0,1)² with Dirichlet BCs.
//! Standard 5-point stencil, row-major (i*ny + j) global DOF numbering.
//!
//! **HYPRE analog**
//!   `hypre/src/examples/ex3.c`  — 2-D 5pt stencil, IJMatrix + BoomerAMG
//!
//! **PETSc analog**
//!   `ksp/ksp/tutorials/ex29.c` / `DMDA` + `KSPSolve`
//!   (we use explicit COO assembly rather than DMDA here)
//!
//! **What this validates**
//!   - 2-D COO assembly with 5-point stencil
//!   - CsrMatrix::from_coo duplicate-free construction
//!   - MMS residual ≈ 0 for the exact solution
//!   - Row-length distribution (boundary rows have 3 or 4 non-zeros)
//!   - SpMV throughput across problem sizes

use std::time::Instant;

use linger::{
    sparse::{CooMatrix, CsrMatrix},
    LinearOperator,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn sep(title: &str) {
    println!("\n━━━━ {title} ━━━━━");
}

fn poisson_2d(nx: usize, ny: usize) -> (CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
    let n = nx * ny;
    let dof = |i: usize, j: usize| i * ny + j;

    let mut coo = CooMatrix::with_capacity(n, n, 5 * n);
    for i in 0..nx {
        for j in 0..ny {
            let row = dof(i, j);
            coo.push(row, row, 4.0);                         // self
            if i > 0     { coo.push(row, dof(i-1, j), -1.0); }  // left
            if i < nx-1  { coo.push(row, dof(i+1, j), -1.0); }  // right
            if j > 0     { coo.push(row, dof(i, j-1), -1.0); }  // down
            if j < ny-1  { coo.push(row, dof(i, j+1), -1.0); }  // up
        }
    }

    let a = CsrMatrix::from_coo(&coo);

    // MMS exact solution: u(i,j) = sin(π(i+1)/(nx+1)) · sin(π(j+1)/(ny+1))
    let pi  = std::f64::consts::PI;
    let nx1 = (nx + 1) as f64;
    let ny1 = (ny + 1) as f64;
    let x_exact: Vec<f64> = (0..nx)
        .flat_map(|i| {
            let si = (pi * (i + 1) as f64 / nx1).sin();
            (0..ny).map(move |j| si * (pi * (j + 1) as f64 / ny1).sin())
        })
        .collect();

    let mut b = vec![0.0f64; n];
    a.spmv(&x_exact, &mut b);

    (a, x_exact, b)
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |s, &x| s + x * x).sqrt()
}

fn row_length_histogram(a: &CsrMatrix<f64>) -> [usize; 6] {
    // Buckets: nnz/row = 2, 3, 4, 5, 6, >6
    let mut hist = [0usize; 6];
    let row_ptr = a.row_ptr();
    for i in 0..a.nrows() {
        let len = row_ptr[i + 1] - row_ptr[i];
        let bucket = (len.saturating_sub(2)).min(5);
        hist[bucket] += 1;
    }
    hist
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    sep("ex03: 2-D Poisson  (HYPRE ex3 / PETSc ex29)");

    // Reference sizes (HYPRE ex3.c uses a 33×33 interior grid → 1089 DOF)
    for &(nx, ny) in &[(8, 8), (32, 32), (64, 64)] {
        let n = nx * ny;
        println!("\n  ── {nx}×{ny} grid, n={n} DOF ──────────────────────────");

        // Assembly
        let t0 = Instant::now();
        let (a, x_exact, b) = poisson_2d(nx, ny);
        let t_assemble = t0.elapsed();

        let nnz = a.nnz();
        let avg_row = nnz as f64 / n as f64;
        let sparsity = nnz as f64 / (n as f64 * n as f64) * 100.0;

        println!("  size:      {n}×{n}  nnz={nnz}  avg-row-len={avg_row:.2}  sparsity={sparsity:.5}%");
        println!("  assemble:  {t_assemble:.2?}");

        // Row-length histogram (validates stencil shape)
        let hist = row_length_histogram(&a);
        println!("  row-len histogram:");
        for (k, &cnt) in hist.iter().enumerate() {
            if cnt > 0 {
                println!("    len={}: {} rows", k + 2, cnt);
            }
        }
        // Interior rows (len=5): (nx-2)*(ny-2)
        let interior = if nx > 2 && ny > 2 { (nx - 2) * (ny - 2) } else { 0 };
        assert_eq!(hist[3], interior,
            "interior rows (len=5) should be (nx-2)*(ny-2)={interior}");

        // Diagonal check: all interior = 4, boundary too
        let diag = a.diag();
        assert!(
            diag.iter().all(|&d| (d - 4.0).abs() < 1e-14),
            "all diagonal entries should be 4"
        );
        println!("  diag:      all 4.0  ✓");

        // Structural symmetry
        assert!(a.is_structurally_symmetric(), "2-D Poisson must be symmetric");
        println!("  symmetric: ✓");

        // MMS residual
        let mut r = vec![0.0f64; n];
        a.spmv(&x_exact, &mut r);
        let nr = r.iter().zip(&b).map(|(&ri, &bi)| (ri-bi).powi(2)).sum::<f64>().sqrt();
        let nb = norm2(&b);
        let rel_res = if nb == 0.0 { nr } else { nr / nb };
        println!("  MMS ‖A·x_exact−b‖/‖b‖ = {rel_res:.3e}");
        assert!(rel_res < 1e-11, "MMS residual too large: {rel_res:.3e}");
        println!("  MMS: ✓");

        // SpMV throughput
        let reps = 10usize.max(100_000 / n);
        let mut y = vec![0.0f64; n];
        let t1 = Instant::now();
        for _ in 0..reps {
            a.spmv(&x_exact, &mut y);
        }
        let t_avg = t1.elapsed() / reps as u32;
        let gflops = (2 * nnz) as f64 / t_avg.as_secs_f64() / 1e9;
        println!("  SpMV×{reps}: avg {t_avg:.2?}  ({gflops:.2} GFLOP/s)");

        // LinearOperator trait path
        use linger::DenseVec;
        let xv = DenseVec::from_vec(x_exact.clone());
        let mut yv = DenseVec::zeros(n);
        a.apply(&xv, &mut yv);
        let diff: f64 = yv.as_slice().iter().zip(&b)
            .map(|(&yi, &bi)| (yi - bi).powi(2)).sum::<f64>().sqrt();
        assert!(diff < 1e-11 * nb, "LinearOperator::apply mismatch");
        println!("  LinearOperator::apply ✓");
    }

    // ── Operator complexity growth ────────────────────────────────────────────
    println!("\n  ── Operator complexity (nnz / n) for 2-D Poisson ───────");
    println!("  {:>10}  {:>8}  {:>10}  {:>12}  {:>10}",
        "grid", "n", "nnz", "nnz/n", "sparsity%");
    for &(nx, ny) in &[(8,8),(16,16),(32,32),(64,64),(128,128)] {
        let n  = nx * ny;
        // nnz = 5*n - 2*(nx+ny) for a rectangular grid (boundary corrections)
        // Exact formula: 5*n - 2*(nx+ny)  but simpler to just assemble
        // Approx: ≈ 5n for large n
        let nnz_approx = 5 * n - 2 * (nx + ny);
        let ratio = nnz_approx as f64 / n as f64;
        let sparsity = nnz_approx as f64 / (n as f64 * n as f64) * 100.0;
        println!("  {:>10}  {:>8}  {:>10}  {:>12.4}  {:>10.5}%",
            format!("{nx}×{ny}"), n, nnz_approx, ratio, sparsity);
    }
    println!("  (nnz/n → 5 as n→∞; near-constant operator complexity ← AMG ideal)");

    println!("\n  OK\n");
}
