//! Sprint 5 tests — parallel ops, BSR format, and WASM-safe build.

mod common;

use linger::{
    parallel::{parallel_axpy, parallel_axpby, parallel_dot, parallel_norm2, parallel_spmv, parallel_spmv_add},
    sparse::{BsrBuilder, CsrMatrix},
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn make_poisson_1d(n: usize) -> CsrMatrix<f64> {
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    a
}

fn serial_spmv(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrows()];
    a.spmv(x, &mut y);
    y
}

// ─── parallel SpMV ───────────────────────────────────────────────────────────

#[test]
fn parallel_spmv_matches_serial_1d() {
    let n = 200;
    let a = make_poisson_1d(n);
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

    let y_serial = serial_spmv(&a, &x);

    let mut y_par = vec![0.0f64; n];
    parallel_spmv(&a, &x, &mut y_par);

    for i in 0..n {
        let diff = (y_par[i] - y_serial[i]).abs();
        assert!(diff < 1e-14, "parallel_spmv mismatch at {i}: par={} serial={}", y_par[i], y_serial[i]);
    }
}

#[test]
fn parallel_spmv_matches_serial_2d() {
    let (a, _, b_vec) = common::make_poisson_2d::<f64>(16, 16);
    let n = a.nrows();
    let x: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

    let mut y_serial = vec![0.0; n];
    a.spmv(&x, &mut y_serial);

    let mut y_par = vec![0.0f64; n];
    parallel_spmv(&a, &x, &mut y_par);

    let diff = y_par.iter().zip(&y_serial).map(|(&p, &s)| (p - s).powi(2)).sum::<f64>().sqrt();
    assert!(diff < 1e-12, "parallel_spmv 2D mismatch: diff={diff:.3e}");
    let _ = b_vec;
}

#[test]
fn parallel_spmv_add_matches_serial() {
    let n = 100;
    let a = make_poisson_1d(n);
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let alpha = 2.0f64;
    let beta  = 0.5f64;
    let y0: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();

    // Serial reference.
    let mut y_serial = y0.clone();
    a.spmv_add(alpha, &x, beta, &mut y_serial);

    // Parallel.
    let mut y_par = y0.clone();
    parallel_spmv_add(&a, alpha, &x, beta, &mut y_par);

    let diff = y_par.iter().zip(&y_serial).map(|(&p, &s)| (p - s).powi(2)).sum::<f64>().sqrt();
    assert!(diff < 1e-12, "parallel_spmv_add mismatch: diff={diff:.3e}");
}

// ─── parallel dense vector ops ───────────────────────────────────────────────

#[test]
fn parallel_axpy_correctness() {
    let n = 1000;
    let alpha = 3.7f64;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut y_par: Vec<f64>    = (0..n).map(|i| (i as f64) * 0.5).collect();
    let mut y_serial: Vec<f64> = y_par.clone();

    // Serial reference.
    for i in 0..n { y_serial[i] += alpha * x[i]; }

    parallel_axpy(alpha, &x, &mut y_par);

    let diff = y_par.iter().zip(&y_serial).map(|(&p, &s)| (p - s).powi(2)).sum::<f64>().sqrt();
    assert!(diff < 1e-12, "parallel_axpy mismatch: diff={diff:.3e}");
}

#[test]
fn parallel_axpby_correctness() {
    let n = 500;
    let (alpha, beta) = (2.0f64, -1.5f64);
    let x: Vec<f64>          = (0..n).map(|i| (i as f64).sin()).collect();
    let mut y_par: Vec<f64>  = (0..n).map(|i| (i as f64).cos()).collect();
    let mut y_ref             = y_par.clone();

    for i in 0..n { y_ref[i] = alpha * x[i] + beta * y_ref[i]; }
    parallel_axpby(alpha, &x, beta, &mut y_par);

    let diff = y_par.iter().zip(&y_ref).map(|(&p, &s)| (p - s).powi(2)).sum::<f64>().sqrt();
    assert!(diff < 1e-12, "parallel_axpby mismatch: diff={diff:.3e}");
}

#[test]
fn parallel_dot_correctness() {
    let n     = 1000;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).recip()).collect();
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let serial: f64 = x.iter().zip(&y).map(|(&a, &b)| a * b).sum();
    let par     = parallel_dot(&x, &y);

    assert!((par - serial).abs() < 1e-10, "parallel_dot mismatch: par={par} serial={serial}");
}

#[test]
fn parallel_norm2_correctness() {
    let n   = 500;
    let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let serial: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let par     = parallel_norm2(&x);
    assert!((par - serial).abs() < 1e-12, "parallel_norm2 mismatch: par={par} serial={serial}");
}

// ─── BSR format ───────────────────────────────────────────────────────────────

#[test]
fn bsr_build_and_spmv_1x1_matches_csr() {
    // 1×1 blocks → BSR reduces to standard CSR.
    let n = 10;
    let a = make_poisson_1d(n);
    let x: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    let mut y_csr = vec![0.0f64; n];
    a.spmv(&x, &mut y_csr);

    // Build BSR with 1×1 blocks from triplets.
    let mut builder = BsrBuilder::<f64>::new(n, n, 1, 1);
    for (r, c, v) in a.triplets() {
        builder.push_block(r, c, vec![v]);
    }
    let bsr = builder.build();

    let mut y_bsr = vec![0.0f64; n];
    bsr.spmv(&x, &mut y_bsr);

    let diff = y_bsr.iter().zip(&y_csr).map(|(&b, &c)| (b - c).powi(2)).sum::<f64>().sqrt();
    assert!(diff < 1e-14, "BSR 1×1 spmv mismatch: diff={diff:.3e}");
}

#[test]
fn bsr_2x2_block_spmv_correctness() {
    // Hand-built 2×2 block system: 2 block-rows, 2 block-cols, 2×2 blocks.
    // A_block = [[1,2],[3,4]] on diagonal, [[0,0],[0,0]] off-diagonal.
    let block = vec![1.0f64, 2.0, 3.0, 4.0];
    let mut builder = BsrBuilder::<f64>::new(2, 2, 2, 2);
    builder.push_block(0, 0, block.clone());
    builder.push_block(1, 1, block.clone());
    let bsr = builder.build();

    assert_eq!(bsr.nrows(), 4);
    assert_eq!(bsr.ncols(), 4);
    assert_eq!(bsr.n_blocks(), 2);

    let x = vec![1.0f64, 2.0, 3.0, 4.0];
    let mut y = vec![0.0f64; 4];
    bsr.spmv(&x, &mut y);

    // Block 0: [1,2;3,4]*[1,2]' = [5, 11]
    // Block 1: [1,2;3,4]*[3,4]' = [11, 25]
    assert!((y[0] - 5.0).abs() < 1e-14);
    assert!((y[1] - 11.0).abs() < 1e-14);
    assert!((y[2] - 11.0).abs() < 1e-14);
    assert!((y[3] - 25.0).abs() < 1e-14);
}

#[test]
fn bsr_to_csr_roundtrip() {
    // Build a BSR matrix and convert to CSR; verify SpMV gives same result.
    let n_blocks = 20;
    let r = 3;
    let c = 3;
    let diag_block: Vec<f64> = (0..r*c).map(|k| if k / c == k % c { 4.0 } else { -1.0 }).collect();
    let mut builder = BsrBuilder::<f64>::new(n_blocks, n_blocks, r, c);
    for i in 0..n_blocks {
        builder.push_block(i, i, diag_block.clone());
    }
    let bsr = builder.build();
    let csr = bsr.to_csr();

    let n  = bsr.nrows();
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).recip()).collect();
    let mut y_bsr = vec![0.0f64; n];
    let mut y_csr = vec![0.0f64; n];
    bsr.spmv(&x, &mut y_bsr);
    csr.spmv(&x, &mut y_csr);

    let diff = y_bsr.iter().zip(&y_csr).map(|(&b, &c)| (b - c).powi(2)).sum::<f64>().sqrt();
    assert!(diff < 1e-13, "BSR→CSR roundtrip mismatch: diff={diff:.3e}");
}

#[test]
fn bsr_parallel_spmv_matches_serial() {
    let n_blocks = 50;
    let r = 2;
    let c = 2;
    let block = vec![4.0f64, -1.0, -1.0, 4.0];
    let off   = vec![-1.0f64, 0.0, 0.0, -1.0];
    let mut builder = BsrBuilder::<f64>::new(n_blocks, n_blocks, r, c);
    for i in 0..n_blocks {
        builder.push_block(i, i, block.clone());
        if i > 0          { builder.push_block(i, i - 1, off.clone()); }
        if i < n_blocks-1 { builder.push_block(i, i + 1, off.clone()); }
    }
    let bsr = builder.build();
    let n   = bsr.nrows();
    let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

    let mut y_serial = vec![0.0f64; n];
    bsr.spmv(&x, &mut y_serial);

    let mut y_par = vec![0.0f64; n];
    bsr.spmv_parallel(&x, &mut y_par);

    let diff = y_par.iter().zip(&y_serial).map(|(&p, &s)| (p - s).powi(2)).sum::<f64>().sqrt();
    assert!(diff < 1e-13, "BSR parallel spmv mismatch: diff={diff:.3e}");
}

#[test]
fn bsr_builder_merges_duplicate_blocks() {
    // Two identical (0,0) blocks should be summed.
    let mut builder = BsrBuilder::<f64>::new(1, 1, 2, 2);
    builder.push_block(0, 0, vec![1.0, 0.0, 0.0, 2.0]);
    builder.push_block(0, 0, vec![3.0, 0.0, 0.0, 4.0]);
    let bsr = builder.build();

    assert_eq!(bsr.n_blocks(), 1);
    assert_eq!(bsr.block_vals[0], 4.0); // 1+3
    assert_eq!(bsr.block_vals[3], 6.0); // 2+4
}

// ─── WASM-compatible build check (compile-time only) ─────────────────────────

// The WASM build is validated by `cargo build --target wasm32-unknown-unknown
// --features wasm` in CI.  No runtime test is possible from a native test binary.
// We add a simple smoke-test that exercises the same code paths used by the WASM
// wrapper without wasm-bindgen.
#[test]
fn wasm_api_smoke_test_via_native_types() {
    use linger::{iterative::ConjugateGradient, DenseVec, KrylovSolver, SolverParams, VerboseLevel};

    let n = 10;
    let a = make_poisson_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    let params = SolverParams {
        rtol: 1e-8,
        max_iter: 200,
        verbose: VerboseLevel::Silent,
        ..Default::default()
    };
    let res = ConjugateGradient::<f64>::default()
        .solve(&a, None, &b, &mut x, &params)
        .unwrap();
    assert!(res.converged);
}
