//! Tests for sparse matrix structure operations.
//!
//! Covers: CsrMatrix (matmat, transpose_csr, is_structurally_symmetric, diag,
//! triplets, from_coo duplicate merging) and CscMatrix (spmv, to_csr, transpose).

mod common;

use linger::sparse::{CooMatrix, CscMatrix, CsrMatrix};

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Build a small dense-ish matrix from explicit (row, col, val) triplets.
fn csr_from_triplets(nrows: usize, ncols: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(nrows, ncols);
    for &(r, c, v) in entries {
        coo.push(r, c, v);
    }
    CsrMatrix::from_coo(&coo)
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// ─── from_coo ────────────────────────────────────────────────────────────────

#[test]
fn csr_from_coo_merges_duplicates() {
    // Push (0,0) twice; values should be summed.
    let mut coo: CooMatrix<f64> = CooMatrix::new(2, 2);
    coo.push(0, 0, 1.5);
    coo.push(0, 0, 0.5);  // duplicate → sum = 2.0
    coo.push(1, 1, 3.0);
    let csr = CsrMatrix::from_coo(&coo);

    assert_eq!(csr.nnz(), 2);
    // Verify via triplets.
    let trips: Vec<_> = csr.triplets().collect();
    assert_eq!(trips.len(), 2);
    let (r, c, v) = trips[0];
    assert_eq!((r, c), (0, 0));
    assert!((v - 2.0).abs() < 1e-14, "expected 2.0, got {v}");
}

#[test]
fn csr_from_coo_empty() {
    let coo: CooMatrix<f64> = CooMatrix::new(5, 4);
    let csr = CsrMatrix::from_coo(&coo);
    assert_eq!(csr.nrows(), 5);
    assert_eq!(csr.ncols(), 4);
    assert_eq!(csr.nnz(), 0);
    assert_eq!(csr.triplets().count(), 0);
}

#[test]
fn csr_from_coo_out_of_order_entries() {
    // Entries pushed in reverse order; CSR must be sorted row-then-column.
    let mut coo: CooMatrix<f64> = CooMatrix::new(3, 3);
    coo.push(2, 2, 3.0);
    coo.push(0, 0, 1.0);
    coo.push(1, 1, 2.0);
    let csr = CsrMatrix::from_coo(&coo);

    let trips: Vec<_> = csr.triplets().collect();
    assert_eq!(trips, vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
}

// ─── triplets ────────────────────────────────────────────────────────────────

#[test]
fn csr_triplets_round_trip() {
    // Build a 3×3 matrix and verify every (r, c, v) is recovered.
    let entries = vec![(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (2, 0, 4.0), (2, 2, 5.0)];
    let csr = csr_from_triplets(3, 3, &entries);

    let mut recovered: Vec<(usize, usize, f64)> = csr.triplets().collect();
    recovered.sort_unstable_by_key(|&(r, c, _)| (r, c));
    assert_eq!(recovered, entries);
}

// ─── diag ─────────────────────────────────────────────────────────────────────

#[test]
fn csr_diag_square() {
    // A = [[2, -1, 0], [-1, 3, -1], [0, -1, 4]]
    let csr = csr_from_triplets(3, 3, &[
        (0, 0, 2.0), (0, 1, -1.0),
        (1, 0, -1.0), (1, 1, 3.0), (1, 2, -1.0),
        (2, 1, -1.0), (2, 2, 4.0),
    ]);
    let d = csr.diag();
    assert_eq!(d, vec![2.0, 3.0, 4.0]);
}

#[test]
fn csr_diag_missing_entry_is_zero() {
    // Row 1 has no diagonal entry.
    let csr = csr_from_triplets(3, 3, &[
        (0, 0, 5.0),
        (1, 0, -1.0), (1, 2, -1.0), // no (1,1)
        (2, 2, 7.0),
    ]);
    let d = csr.diag();
    assert_eq!(d.len(), 3);
    assert!((d[0] - 5.0).abs() < 1e-14);
    assert_eq!(d[1], 0.0, "missing diagonal must be zero");
    assert!((d[2] - 7.0).abs() < 1e-14);
}

#[test]
fn csr_diag_nonsquare_length() {
    // 2×5 matrix: diag length should be min(2, 5) = 2.
    let csr = csr_from_triplets(2, 5, &[(0, 0, 1.0), (1, 1, 2.0)]);
    assert_eq!(csr.diag().len(), 2);

    // 5×2 matrix: diag length = min(5, 2) = 2.
    let csr2 = csr_from_triplets(5, 2, &[(0, 0, 1.0), (1, 1, 2.0)]);
    assert_eq!(csr2.diag().len(), 2);
}

// ─── is_structurally_symmetric ────────────────────────────────────────────────

#[test]
fn csr_is_structurally_symmetric_true() {
    // Symmetric tridiagonal matrix.
    let (a, _, _) = common::make_poisson_1d::<f64>(6);
    assert!(a.is_structurally_symmetric());
}

#[test]
fn csr_is_structurally_symmetric_false_missing_lower() {
    // Upper-triangular tridiagonal (missing sub-diagonal).
    let csr = csr_from_triplets(3, 3, &[
        (0, 0, 2.0), (0, 1, -1.0),
        (1, 1, 2.0), (1, 2, -1.0),
        (2, 2, 2.0),
    ]);
    assert!(!csr.is_structurally_symmetric());
}

#[test]
fn csr_is_structurally_symmetric_false_nonsquare() {
    // Non-square → cannot be symmetric.
    let csr = csr_from_triplets(2, 3, &[(0, 0, 1.0), (1, 1, 1.0)]);
    assert!(!csr.is_structurally_symmetric());
}

#[test]
fn csr_is_structurally_symmetric_diagonal_only() {
    // Pure diagonal: each entry (i, i) trivially has a matching (i, i).
    let csr = csr_from_triplets(4, 4, &[(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0), (3, 3, 4.0)]);
    assert!(csr.is_structurally_symmetric());
}

// ─── matmat ───────────────────────────────────────────────────────────────────

#[test]
fn csr_matmat_known_result() {
    // A = [[1, 2], [3, 4]]   B = [[5, 6], [7, 8]]
    // C = A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let a = csr_from_triplets(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)]);
    let b = csr_from_triplets(2, 2, &[(0, 0, 5.0), (0, 1, 6.0), (1, 0, 7.0), (1, 1, 8.0)]);
    let c = a.matmat(&b);

    assert_eq!(c.nrows(), 2);
    assert_eq!(c.ncols(), 2);

    // Verify via SpMV: C * [1, 0] = col 0 of C = [19, 43]
    let mut y = vec![0.0f64; 2];
    c.spmv(&[1.0, 0.0], &mut y);
    assert!((y[0] - 19.0).abs() < 1e-12, "C[0,0]={}", y[0]);
    assert!((y[1] - 43.0).abs() < 1e-12, "C[1,0]={}", y[1]);

    // C * [0, 1] = col 1 of C = [22, 50]
    c.spmv(&[0.0, 1.0], &mut y);
    assert!((y[0] - 22.0).abs() < 1e-12, "C[0,1]={}", y[0]);
    assert!((y[1] - 50.0).abs() < 1e-12, "C[1,1]={}", y[1]);
}

#[test]
fn csr_matmat_identity_right() {
    // A * I = A  for any matrix A.
    let (a, _, _) = common::make_poisson_1d::<f64>(8);
    let n = a.nrows();

    // Build I as CSR.
    let identity = csr_from_triplets(n, n, &(0..n).map(|i| (i, i, 1.0)).collect::<Vec<_>>());

    let c = a.matmat(&identity);

    // A * I should reproduce A: compare SpMV on a random-ish vector.
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let mut ya = vec![0.0f64; n];
    let mut yc = vec![0.0f64; n];
    a.spmv(&x, &mut ya);
    c.spmv(&x, &mut yc);

    let diff = ya.iter().zip(&yc).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
    assert!(diff < 1e-13, "A*I != A: diff={diff:.3e}");
}

#[test]
fn csr_matmat_identity_left() {
    // I * A = A.
    let (a, _, _) = common::make_poisson_1d::<f64>(8);
    let n = a.nrows();
    let identity = csr_from_triplets(n, n, &(0..n).map(|i| (i, i, 1.0)).collect::<Vec<_>>());
    let c = identity.matmat(&a);

    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).cos()).collect();
    let mut ya = vec![0.0f64; n];
    let mut yc = vec![0.0f64; n];
    a.spmv(&x, &mut ya);
    c.spmv(&x, &mut yc);

    let diff = ya.iter().zip(&yc).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
    assert!(diff < 1e-13, "I*A != A: diff={diff:.3e}");
}

#[test]
fn csr_matmat_ata_is_spd() {
    // AᵀA is symmetric positive semi-definite for any A.
    // Verify: for Poisson A, AᵀA * x · x > 0 for nonzero x.
    let (a, _, _) = common::make_poisson_1d::<f64>(6);
    let at = a.transpose_csr();
    let ata = at.matmat(&a);  // AᵀA

    let x: Vec<f64> = (0..6).map(|i| (i + 1) as f64).collect();
    let mut y = vec![0.0f64; 6];
    ata.spmv(&x, &mut y);
    let xty: f64 = x.iter().zip(&y).map(|(&xi, &yi)| xi * yi).sum();
    assert!(xty > 0.0, "AᵀA must be positive semi-definite; xᵀAᵀAx = {xty}");
}

// ─── transpose_csr ───────────────────────────────────────────────────────────

#[test]
fn csr_transpose_csr_dimensions() {
    // For an m×n matrix, (Aᵀ) should be n×m.
    let csr = csr_from_triplets(3, 5, &[(0, 1, 1.0), (1, 3, 2.0), (2, 4, 3.0)]);
    let at = csr.transpose_csr();
    assert_eq!(at.nrows(), 5);
    assert_eq!(at.ncols(), 3);
    assert_eq!(at.nnz(), csr.nnz());
}

#[test]
fn csr_transpose_csr_roundtrip() {
    // (Aᵀ)ᵀ == A: SpMV with a random vector should give the same result.
    let (a, _, _) = common::make_poisson_1d::<f64>(7);
    let n = a.nrows();
    let att = a.transpose_csr().transpose_csr();

    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5).sin()).collect();
    let mut ya  = vec![0.0f64; n];
    let mut ytt = vec![0.0f64; n];
    a.spmv(&x, &mut ya);
    att.spmv(&x, &mut ytt);

    let diff = ya.iter().zip(&ytt).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
    assert!(diff < 1e-13, "(Aᵀ)ᵀ != A: diff={diff:.3e}");
}

#[test]
fn csr_transpose_csr_spmv_correctness() {
    // Hand-verify Aᵀ * x for a small non-symmetric matrix.
    // A = [[1, 2, 3], [4, 5, 6]]  (2×3)
    // Aᵀ = [[1, 4], [2, 5], [3, 6]]  (3×2)
    // Aᵀ * [1, 1] = [5, 7, 9]
    let a = csr_from_triplets(2, 3, &[
        (0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0),
        (1, 0, 4.0), (1, 1, 5.0), (1, 2, 6.0),
    ]);
    let at = a.transpose_csr();

    let mut y = vec![0.0f64; 3];
    at.spmv(&[1.0, 1.0], &mut y);
    assert!((y[0] - 5.0).abs() < 1e-13, "y[0]={}", y[0]);
    assert!((y[1] - 7.0).abs() < 1e-13, "y[1]={}", y[1]);
    assert!((y[2] - 9.0).abs() < 1e-13, "y[2]={}", y[2]);
}

#[test]
fn csr_transpose_csr_matches_csc_transpose() {
    // `transpose_csr()` and `transpose().to_csr()` must give the same SpMV.
    let (a, _, _) = common::make_poisson_1d::<f64>(9);
    let n = a.nrows();
    let at_csr = a.transpose_csr();
    let at_via_csc = a.transpose().to_csr();  // CscMatrix → back to CSR

    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut y1 = vec![0.0f64; n];
    let mut y2 = vec![0.0f64; n];
    at_csr.spmv(&x, &mut y1);
    at_via_csc.spmv(&x, &mut y2);

    let diff = norm2(&y1.iter().zip(&y2).map(|(&a, &b)| a - b).collect::<Vec<_>>());
    assert!(diff < 1e-13, "transpose_csr vs transpose().to_csr mismatch: diff={diff:.3e}");
}

// ─── CSC ─────────────────────────────────────────────────────────────────────

#[test]
fn csc_spmv_matches_manual() {
    // A = [[1, 2], [3, 4]]
    // Aᵀ (as CSC of A) computes Aᵀ * x.
    // Aᵀ * [1, 0] = [1, 2],   Aᵀ * [0, 1] = [3, 4]
    let a = csr_from_triplets(2, 2, &[
        (0, 0, 1.0), (0, 1, 2.0),
        (1, 0, 3.0), (1, 1, 4.0),
    ]);
    let at_csc: CscMatrix<f64> = a.transpose();  // CscMatrix representing Aᵀ

    let mut y = vec![0.0f64; 2];
    at_csc.spmv(&[1.0, 0.0], &mut y);
    assert!((y[0] - 1.0).abs() < 1e-14 && (y[1] - 2.0).abs() < 1e-14,
        "Aᵀ * e0: expected [1,2], got {:?}", y);

    at_csc.spmv(&[0.0, 1.0], &mut y);
    assert!((y[0] - 3.0).abs() < 1e-14 && (y[1] - 4.0).abs() < 1e-14,
        "Aᵀ * e1: expected [3,4], got {:?}", y);
}

#[test]
fn csc_spmv_matches_csr_transposed() {
    // For symmetric Poisson A, Aᵀ = A, so CSC::spmv should match CSR::spmv.
    let (a, _, _) = common::make_poisson_1d::<f64>(8);
    let n = a.nrows();
    let at: CscMatrix<f64> = a.transpose();

    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let mut y_csr = vec![0.0f64; n];
    let mut y_csc = vec![0.0f64; n];
    a.spmv(&x, &mut y_csr);
    at.spmv(&x, &mut y_csc);

    let diff = norm2(&y_csr.iter().zip(&y_csc).map(|(&a, &b)| a - b).collect::<Vec<_>>());
    assert!(diff < 1e-13,
        "Symmetric matrix: CSR.spmv vs CSC(Aᵀ).spmv mismatch: diff={diff:.3e}");
}

#[test]
fn csc_accessors_consistent() {
    // Verify col_ptr, row_idx, values have correct sizes.
    let a = csr_from_triplets(3, 4, &[(0, 1, 2.0), (1, 3, 5.0), (2, 0, 1.0)]);
    let csc: CscMatrix<f64> = a.transpose();
    // Aᵀ is 4×3, so CSC has ncols = 4, nrows = 3... wait.
    // Actually: csr is 3×4, transpose() returns CscMatrix representing Aᵀ.
    // CscMatrix(nrows=4, ncols=3, col_ptr.len=4, ...)
    // Let me verify the internal consistency rules.
    assert_eq!(csc.col_ptr().len(), csc.ncols() + 1);
    assert_eq!(csc.row_idx().len(), csc.nnz());
    assert_eq!(csc.values().len(), csc.nnz());
}

#[test]
fn csc_to_csr_roundtrip() {
    // csc.to_csr() should reproduce the original CSR data.
    let (a, _, _) = common::make_poisson_1d::<f64>(6);
    let n = a.nrows();
    let csc: CscMatrix<f64> = a.transpose();
    let back = csc.to_csr();  // this is Aᵀ as CSR

    // back is Aᵀ; applying it should match transpose_csr().spmv.
    let at_csr = a.transpose_csr();
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut y1 = vec![0.0f64; n];
    let mut y2 = vec![0.0f64; n];
    back.spmv(&x, &mut y1);
    at_csr.spmv(&x, &mut y2);

    let diff = norm2(&y1.iter().zip(&y2).map(|(&a, &b)| a - b).collect::<Vec<_>>());
    assert!(diff < 1e-13, "CSC.to_csr() roundtrip mismatch: diff={diff:.3e}");
}

#[test]
fn csc_transpose_returns_original() {
    // csc.transpose() returns a CsrMatrix. For symmetric A (Poisson),
    // this should be equal to A again.
    let (a, _, _) = common::make_poisson_1d::<f64>(5);
    let n = a.nrows();
    let csc: CscMatrix<f64> = a.transpose();
    let recovered: CsrMatrix<f64> = csc.transpose(); // (Aᵀ)ᵀ = A as CSR

    let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let mut ya = vec![0.0f64; n];
    let mut yr = vec![0.0f64; n];
    a.spmv(&x, &mut ya);
    recovered.spmv(&x, &mut yr);

    let diff = norm2(&ya.iter().zip(&yr).map(|(&a, &b)| a - b).collect::<Vec<_>>());
    assert!(diff < 1e-13, "CSC.transpose() != A: diff={diff:.3e}");
}

// ─── spmv_add ─────────────────────────────────────────────────────────────────

#[test]
fn csr_spmv_add_alpha1_beta0_equals_spmv() {
    // spmv_add(1, x, 0, y) should be identical to spmv(x, y).
    let (a, _, _) = common::make_poisson_1d::<f64>(8);
    let n = a.nrows();
    let x: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

    let mut y1 = vec![0.0f64; n];
    let mut y2 = vec![1e10f64; n];  // initial value irrelevant for beta=0
    a.spmv(&x, &mut y1);
    a.spmv_add(1.0, &x, 0.0, &mut y2);

    let diff = norm2(&y1.iter().zip(&y2).map(|(&a, &b)| a - b).collect::<Vec<_>>());
    assert!(diff < 1e-12, "spmv_add(1,x,0,y) != spmv: diff={diff:.3e}");
}

#[test]
fn csr_spmv_add_accumulates_correctly() {
    // y = 2*A*x + 3*y_init; verify by computing reference manually.
    let n = 5;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let x    = vec![1.0f64; n];
    let y0   = vec![1.0f64; n];

    let mut ax = vec![0.0f64; n];
    a.spmv(&x, &mut ax);
    let reference: Vec<f64> = ax.iter().zip(&y0).map(|(&ax, &y)| 2.0 * ax + 3.0 * y).collect();

    let mut y = y0.clone();
    a.spmv_add(2.0, &x, 3.0, &mut y);

    let diff = norm2(&y.iter().zip(&reference).map(|(&a, &b)| a - b).collect::<Vec<_>>());
    assert!(diff < 1e-12, "spmv_add accumulation wrong: diff={diff:.3e}");
}
