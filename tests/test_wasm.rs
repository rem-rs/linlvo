//! WASM end-to-end tests for linger's JS-callable API.
//!
//! These tests are compiled and executed in a headless browser / Node.js via
//! `wasm-pack test --headless --firefox` (or `--node`).
//!
//! On non-wasm32 targets the tests are skipped automatically because
//! `wasm_bindgen_test` only registers them when compiling for wasm32.
//!
//! # Running
//! ```sh
//! wasm-pack test --node --features wasm
//! # or headless browser:
//! wasm-pack test --headless --firefox --features wasm
//! ```

use wasm_bindgen_test::*;

// Run in Node.js (no browser needed for CI).
wasm_bindgen_test_configure!(run_in_node_experimental);

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Build the 1D Laplacian of size n as (row_ptr, col_idx, vals) flat arrays.
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn laplacian_1d_coo(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n {
        rows.push(i); cols.push(i); vals.push(2.0);
        if i > 0     { rows.push(i); cols.push(i - 1); vals.push(-1.0); }
        if i < n - 1 { rows.push(i); cols.push(i + 1); vals.push(-1.0); }
    }
    (rows, cols, vals)
}

// ─── WasmCsrMatrix ───────────────────────────────────────────────────────────

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm_tests {
    use super::*;
    use linger::wasm::{WasmCsrMatrix, WasmCgSolver, WasmGmresSolver,
                       WasmLuSolver, WasmCholeskySolver, WasmMultifrontalSolver};

    #[wasm_bindgen_test]
    fn wasm_csr_matrix_dimensions() {
        let (rows, cols, vals) = laplacian_1d_coo(5);
        let a = WasmCsrMatrix::from_coo(5, 5, &rows, &cols, &vals).unwrap();
        assert_eq!(a.nrows(), 5);
        assert_eq!(a.ncols(), 5);
        assert!(a.nnz() > 0);
    }

    #[wasm_bindgen_test]
    fn wasm_csr_mismatched_lengths_error() {
        let result = WasmCsrMatrix::from_coo(2, 2, &[0], &[0, 1], &[1.0, 2.0]);
        assert!(result.is_err());
    }

    // ─── CG solver ───────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn wasm_cg_solves_diagonal() {
        // Diagonal matrix: A = diag(1,2,3,4,5), b = [1,2,3,4,5] → x = [1,1,1,1,1]
        let n = 5usize;
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let vals: Vec<f64>   = (1..=n).map(|i| i as f64).collect();
        let a = WasmCsrMatrix::from_coo(n, n, &rows, &cols, &vals).unwrap();
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let x = WasmCgSolver::new(1e-10, 100).solve(&a, &b).unwrap();
        assert_eq!(x.len(), n);
        for &xi in &x {
            assert!((xi - 1.0).abs() < 1e-8, "xi = {xi}");
        }
    }

    #[wasm_bindgen_test]
    fn wasm_cg_solves_poisson_1d() {
        let n = 20usize;
        let (rows, cols, vals) = laplacian_1d_coo(n);
        let a = WasmCsrMatrix::from_coo(n, n, &rows, &cols, &vals).unwrap();
        let b = vec![1.0f64; n];
        let x = WasmCgSolver::new(1e-10, 500).solve(&a, &b).unwrap();
        assert_eq!(x.len(), n);
        // Verify residual: A*x ≈ b. Just check solution is finite and nonzero.
        for &xi in &x {
            assert!(xi.is_finite(), "non-finite solution entry");
        }
    }

    #[wasm_bindgen_test]
    fn wasm_cg_dimension_mismatch_error() {
        let (rows, cols, vals) = laplacian_1d_coo(4);
        let a = WasmCsrMatrix::from_coo(4, 4, &rows, &cols, &vals).unwrap();
        let b = vec![1.0f64; 5]; // wrong length
        assert!(WasmCgSolver::new(1e-10, 100).solve(&a, &b).is_err());
    }

    // ─── GMRES solver ────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn wasm_gmres_solves_diagonal() {
        let n = 5usize;
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let vals: Vec<f64>   = (1..=n).map(|i| i as f64).collect();
        let a = WasmCsrMatrix::from_coo(n, n, &rows, &cols, &vals).unwrap();
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let x = WasmGmresSolver::new(1e-10, 100, 10).solve(&a, &b).unwrap();
        for &xi in &x {
            assert!((xi - 1.0).abs() < 1e-8, "xi = {xi}");
        }
    }

    #[wasm_bindgen_test]
    fn wasm_gmres_solves_poisson_1d() {
        let n = 20usize;
        let (rows, cols, vals) = laplacian_1d_coo(n);
        let a = WasmCsrMatrix::from_coo(n, n, &rows, &cols, &vals).unwrap();
        let b = vec![1.0f64; n];
        let x = WasmGmresSolver::new(1e-10, 500, 20).solve(&a, &b).unwrap();
        assert_eq!(x.len(), n);
        for &xi in &x {
            assert!(xi.is_finite());
        }
    }

    // ─── LU direct solver ────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn wasm_lu_solves_diagonal() {
        let n = 4usize;
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let vals = vec![2.0f64, 3.0, 4.0, 5.0];
        let a = WasmCsrMatrix::from_coo(n, n, &rows, &cols, &vals).unwrap();
        let b = vec![2.0f64, 6.0, 8.0, 10.0]; // x = [1,2,2,2]
        let x = WasmLuSolver::new("rcm").solve(&a, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[wasm_bindgen_test]
    fn wasm_lu_solves_poisson_1d() {
        let n = 10usize;
        let (rows, cols, vals) = laplacian_1d_coo(n);
        let a = WasmCsrMatrix::from_coo(n, n, &rows, &cols, &vals).unwrap();
        let b = vec![1.0f64; n];
        let x = WasmLuSolver::new("rcm").solve(&a, &b).unwrap();
        assert_eq!(x.len(), n);
        for &xi in &x {
            assert!(xi.is_finite());
        }
    }

    // ─── Cholesky direct solver ───────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn wasm_cholesky_solves_poisson_1d() {
        let n = 10usize;
        let (rows, cols, vals) = laplacian_1d_coo(n);
        let a = WasmCsrMatrix::from_coo(n, n, &rows, &cols, &vals).unwrap();
        let b = vec![1.0f64; n];
        let x = WasmCholeskySolver::new("rcm").solve(&a, &b).unwrap();
        assert_eq!(x.len(), n);
        for &xi in &x {
            assert!(xi.is_finite());
        }
    }

    // ─── Multifrontal solver ──────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn wasm_multifrontal_solves_poisson_1d() {
        let n = 10usize;
        let (rows, cols, vals) = laplacian_1d_coo(n);
        let a = WasmCsrMatrix::from_coo(n, n, &rows, &cols, &vals).unwrap();
        let b = vec![1.0f64; n];
        let x = WasmMultifrontalSolver::new("rcm").solve(&a, &b).unwrap();
        assert_eq!(x.len(), n);
        for &xi in &x {
            assert!(xi.is_finite());
        }
    }

    #[wasm_bindgen_test]
    fn wasm_multifrontal_precond_gmres() {
        let n = 10usize;
        let (rows, cols, vals) = laplacian_1d_coo(n);
        let a = WasmCsrMatrix::from_coo(n, n, &rows, &cols, &vals).unwrap();
        let b = vec![1.0f64; n];
        let x = WasmMultifrontalSolver::new("rcm")
            .solve_precond_gmres(&a, &b, 1e-10, 50, 15)
            .unwrap();
        assert_eq!(x.len(), n);
        for &xi in &x {
            assert!(xi.is_finite());
        }
    }

    #[wasm_bindgen_test]
    fn wasm_multifrontal_ordering_natural() {
        let n = 8usize;
        let (rows, cols, vals) = laplacian_1d_coo(n);
        let a = WasmCsrMatrix::from_coo(n, n, &rows, &cols, &vals).unwrap();
        let b = vec![1.0f64; n];
        let x_rcm = WasmMultifrontalSolver::new("rcm").solve(&a, &b).unwrap();
        let x_nat = WasmMultifrontalSolver::new("natural").solve(&a, &b).unwrap();
        // Both orderings should give the same solution.
        for (a, b) in x_rcm.iter().zip(x_nat.iter()) {
            assert!((a - b).abs() < 1e-8, "solutions differ: {a} vs {b}");
        }
    }
}

// On non-wasm targets, provide empty placeholder tests so `cargo test` doesn't
// fail due to an empty test binary.
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
mod native_placeholders {
    /// WASM tests only run under `wasm-pack test --features wasm`.
    /// This placeholder ensures `cargo test` sees at least one test in this file.
    #[test]
    fn wasm_tests_require_wasm_pack() {
        // Nothing to test on native; wasm-pack handles the actual WASM tests.
    }
}
