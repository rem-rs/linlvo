//! Integration tests for ELL and DIA sparse matrix formats (H1).

use linger::{DiaMatrix, EllMatrix};
use linger::sparse::{CooMatrix, CsrMatrix};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn laplacian_2d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut coo = CooMatrix::new(nn, nn);
    for i in 0..n {
        for j in 0..n {
            let id = i * n + j;
            coo.push(id, id, 4.0);
            if j > 0 { coo.push(id, id - 1, -1.0); coo.push(id - 1, id, -1.0); }
            if i > 0 { coo.push(id, id - n, -1.0); coo.push(id - n, id, -1.0); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

// ─── ELL tests ────────────────────────────────────────────────────────────────

#[test]
fn ell_from_csr_dimensions() {
    let csr = laplacian_1d(10);
    let ell = EllMatrix::<f64>::from_csr(&csr);
    assert_eq!(ell.nrows(), 10);
    assert_eq!(ell.ncols(), 10);
    assert_eq!(ell.max_nnz_per_row(), 3); // interior rows have 3 entries
}

#[test]
fn ell_nnz_matches_csr() {
    let csr = laplacian_1d(8);
    let ell = EllMatrix::<f64>::from_csr(&csr);
    assert_eq!(ell.nnz(), csr.nnz());
}

#[test]
fn ell_spmv_all_ones() {
    let n = 6usize;
    let csr = laplacian_1d(n);
    let ell = EllMatrix::<f64>::from_csr(&csr);
    let x = vec![1.0f64; n];
    let mut y = vec![0.0f64; n];
    ell.spmv(&x, &mut y);
    // boundary: 2-1=1; interior: -1+2-1=0
    assert!((y[0] - 1.0).abs() < 1e-14);
    assert!((y[1]).abs() < 1e-14);
    assert!((y[n - 1] - 1.0).abs() < 1e-14);
}

#[test]
fn ell_spmv_matches_csr_random_like() {
    let csr = laplacian_1d(12);
    let ell = EllMatrix::<f64>::from_csr(&csr);
    let x: Vec<f64> = (1..=12).map(|i| i as f64 * 0.5).collect();
    let mut y_csr = vec![0.0f64; 12];
    let mut y_ell = vec![0.0f64; 12];
    csr.spmv(&x, &mut y_csr);
    ell.spmv(&x, &mut y_ell);
    for (a, b) in y_csr.iter().zip(y_ell.iter()) {
        assert!((a - b).abs() < 1e-12, "mismatch: CSR={a}, ELL={b}");
    }
}

#[test]
fn ell_spmv_2d_laplacian() {
    let csr = laplacian_2d(8); // 64×64
    let ell = EllMatrix::<f64>::from_csr(&csr);
    let n = csr.nrows();
    let x = vec![1.0f64; n];
    let mut y_csr = vec![0.0f64; n];
    let mut y_ell = vec![0.0f64; n];
    csr.spmv(&x, &mut y_csr);
    ell.spmv(&x, &mut y_ell);
    for (a, b) in y_csr.iter().zip(y_ell.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn ell_to_csr_roundtrip_values() {
    let csr = laplacian_1d(9);
    let ell = EllMatrix::<f64>::from_csr(&csr);
    let csr2 = ell.to_csr();
    // SpMV on both should give identical results.
    let x: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let mut y1 = vec![0.0f64; 9];
    let mut y2 = vec![0.0f64; 9];
    csr.spmv(&x, &mut y1);
    csr2.spmv(&x, &mut y2);
    for (a, b) in y1.iter().zip(y2.iter()) {
        assert!((a - b).abs() < 1e-13);
    }
}

#[test]
fn ell_storage_layout_column_major() {
    // 2×2 identity — max_nnz_per_row = 1
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 0, 3.0);
    coo.push(1, 1, 7.0);
    let ell = EllMatrix::from_coo(&coo);
    assert_eq!(ell.max_nnz_per_row(), 1);
    // col_idx[0 + 0*2] = 0 (row 0, diag-slot 0)
    // col_idx[1 + 0*2] = 1 (row 1, diag-slot 0)
    assert_eq!(ell.col_idx()[0], 0);
    assert_eq!(ell.col_idx()[1], 1);
    assert!((ell.values()[0] - 3.0f64).abs() < 1e-14);
    assert!((ell.values()[1] - 7.0f64).abs() < 1e-14);
}

#[test]
fn ell_from_coo_convenience() {
    let n = 5usize;
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n { coo.push(i, i, 2.0); }
    let ell = EllMatrix::from_coo(&coo);
    assert_eq!(ell.nnz(), n);
}

// ─── DIA tests ────────────────────────────────────────────────────────────────

#[test]
fn dia_tridiagonal_offsets() {
    let csr = laplacian_1d(8);
    let dia = DiaMatrix::<f64>::from_csr(&csr);
    assert_eq!(dia.num_diags(), 3);
    assert_eq!(dia.offsets(), &[-1isize, 0, 1]);
}

#[test]
fn dia_nnz_matches_csr() {
    let csr = laplacian_1d(7);
    let dia = DiaMatrix::<f64>::from_csr(&csr);
    assert_eq!(dia.nnz(), csr.nnz());
}

#[test]
fn dia_spmv_all_ones() {
    let n = 6usize;
    let csr = laplacian_1d(n);
    let dia = DiaMatrix::<f64>::from_csr(&csr);
    let x = vec![1.0f64; n];
    let mut y = vec![0.0f64; n];
    dia.spmv(&x, &mut y);
    assert!((y[0] - 1.0).abs() < 1e-14);
    assert!((y[1]).abs() < 1e-14);
    assert!((y[n - 1] - 1.0).abs() < 1e-14);
}

#[test]
fn dia_spmv_matches_csr() {
    let csr = laplacian_1d(12);
    let dia = DiaMatrix::<f64>::from_csr(&csr);
    let x: Vec<f64> = (1..=12).map(|i| i as f64 * 0.3).collect();
    let mut y_csr = vec![0.0f64; 12];
    let mut y_dia = vec![0.0f64; 12];
    csr.spmv(&x, &mut y_csr);
    dia.spmv(&x, &mut y_dia);
    for (a, b) in y_csr.iter().zip(y_dia.iter()) {
        assert!((a - b).abs() < 1e-12, "mismatch: CSR={a}, DIA={b}");
    }
}

#[test]
fn dia_spmv_2d_laplacian() {
    let csr = laplacian_2d(6); // 36×36, diagonals at -6,-1,0,+1,+6
    let dia = DiaMatrix::<f64>::from_csr(&csr);
    assert_eq!(dia.num_diags(), 5);
    let n = csr.nrows();
    let x = vec![1.0f64; n];
    let mut y_csr = vec![0.0f64; n];
    let mut y_dia = vec![0.0f64; n];
    csr.spmv(&x, &mut y_csr);
    dia.spmv(&x, &mut y_dia);
    for (a, b) in y_csr.iter().zip(y_dia.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn dia_to_csr_roundtrip() {
    let csr = laplacian_1d(9);
    let dia = DiaMatrix::<f64>::from_csr(&csr);
    let csr2 = dia.to_csr();
    let x: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let mut y1 = vec![0.0f64; 9];
    let mut y2 = vec![0.0f64; 9];
    csr.spmv(&x, &mut y1);
    csr2.spmv(&x, &mut y2);
    for (a, b) in y1.iter().zip(y2.iter()) {
        assert!((a - b).abs() < 1e-13);
    }
}

#[test]
fn dia_rectangular_spmv() {
    // 3×5 with 3 entries on diagonal offset +1
    let mut coo = CooMatrix::new(3, 5);
    coo.push(0, 1, 1.0);
    coo.push(1, 2, 2.0);
    coo.push(2, 3, 3.0);
    let dia = DiaMatrix::from_coo(&coo);
    assert_eq!(dia.offsets(), &[1isize]);
    let x = vec![0.0f64, 1.0, 1.0, 1.0, 0.0];
    let mut y = vec![0.0f64; 3];
    dia.spmv(&x, &mut y);
    assert!((y[0] - 1.0).abs() < 1e-14);
    assert!((y[1] - 2.0).abs() < 1e-14);
    assert!((y[2] - 3.0).abs() < 1e-14);
}

#[test]
fn dia_empty_matrix() {
    let coo: CooMatrix<f64> = CooMatrix::new(0, 0);
    let dia = DiaMatrix::from_coo(&coo);
    assert_eq!(dia.num_diags(), 0);
    assert_eq!(dia.nnz(), 0);
}

// ─── Cross-format consistency ────────────────────────────────────────────────

#[test]
fn ell_dia_csr_all_agree_2d() {
    let csr = laplacian_2d(5); // 25×25
    let ell = EllMatrix::<f64>::from_csr(&csr);
    let dia = DiaMatrix::<f64>::from_csr(&csr);
    let n = csr.nrows();
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
    let mut y_csr = vec![0.0f64; n];
    let mut y_ell = vec![0.0f64; n];
    let mut y_dia = vec![0.0f64; n];
    csr.spmv(&x, &mut y_csr);
    ell.spmv(&x, &mut y_ell);
    dia.spmv(&x, &mut y_dia);
    for i in 0..n {
        assert!((y_csr[i] - y_ell[i]).abs() < 1e-12, "ELL mismatch at {i}");
        assert!((y_csr[i] - y_dia[i]).abs() < 1e-12, "DIA mismatch at {i}");
    }
}
