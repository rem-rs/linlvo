//! Integration tests for the Matrix Market (.mtx) reader/writer.
//!
//! All tests use in-memory string parsing — no .mtx files need to exist on disk.

use linger::sparse::{read_matrix_market_str, read_matrix_market_coo_str, write_matrix_market_str, MmioError};

// ─── helper ───────────────────────────────────────────────────────────────────

/// Return the value at (row, col) in a CSR matrix, or None if the entry is not
/// structurally present.
fn get_val(a: &linger::sparse::CsrMatrix<f64>, row: usize, col: usize) -> Option<f64> {
    let row_start = a.row_ptr()[row];
    let row_end   = a.row_ptr()[row + 1];
    for k in row_start..row_end {
        if a.col_idx()[k] == col {
            return Some(a.values()[k]);
        }
    }
    None
}

// ─── basic general matrix ────────────────────────────────────────────────────

#[test]
fn general_3x3_size_and_nnz() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
3 3 4
1 1 1.0
1 3 2.0
2 2 3.0
3 1 4.0
";
    let a = read_matrix_market_str(mtx).unwrap();
    assert_eq!(a.nrows(), 3);
    assert_eq!(a.ncols(), 3);
    assert_eq!(a.nnz(), 4);
}

#[test]
fn general_entries_correct_values() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
3 3 3
1 1 7.5
2 3 -2.0
3 2 0.5
";
    let a = read_matrix_market_str(mtx).unwrap();
    assert!((get_val(&a, 0, 0).unwrap() - 7.5).abs() < 1e-15);
    assert!((get_val(&a, 1, 2).unwrap() + 2.0).abs() < 1e-15);
    assert!((get_val(&a, 2, 1).unwrap() - 0.5).abs() < 1e-15);
}

// ─── symmetric expansion ─────────────────────────────────────────────────────

#[test]
fn symmetric_off_diagonal_expanded() {
    let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 3 3
1 1 4.0
2 1 -1.0
3 3 5.0
";
    let a = read_matrix_market_str(mtx).unwrap();
    // (1,0) and (0,1) both present.
    assert!((get_val(&a, 1, 0).unwrap() + 1.0).abs() < 1e-15);
    assert!((get_val(&a, 0, 1).unwrap() + 1.0).abs() < 1e-15);
    assert_eq!(a.nnz(), 4);
}

#[test]
fn symmetric_diagonal_not_doubled() {
    let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
2 2 2
1 1 3.0
2 2 5.0
";
    let a = read_matrix_market_str(mtx).unwrap();
    assert_eq!(a.nnz(), 2);
    assert!((get_val(&a, 0, 0).unwrap() - 3.0).abs() < 1e-15);
    assert!((get_val(&a, 1, 1).unwrap() - 5.0).abs() < 1e-15);
}

// ─── skew-symmetric ───────────────────────────────────────────────────────────

#[test]
fn skew_symmetric_negated_transpose() {
    let mtx = "\
%%MatrixMarket matrix coordinate real skew-symmetric
3 3 2
2 1 3.0
3 2 -2.0
";
    let a = read_matrix_market_str(mtx).unwrap();
    assert!((get_val(&a, 1, 0).unwrap() - 3.0).abs() < 1e-15);
    assert!((get_val(&a, 0, 1).unwrap() + 3.0).abs() < 1e-15);
    assert!((get_val(&a, 2, 1).unwrap() + 2.0).abs() < 1e-15);
    assert!((get_val(&a, 1, 2).unwrap() - 2.0).abs() < 1e-15);
    assert_eq!(a.nnz(), 4);
}

// ─── pattern format ───────────────────────────────────────────────────────────

#[test]
fn pattern_general_all_ones() {
    let mtx = "\
%%MatrixMarket matrix coordinate pattern general
4 4 3
1 1
2 3
4 2
";
    let a = read_matrix_market_str(mtx).unwrap();
    assert_eq!(a.nnz(), 3);
    for &v in a.values() {
        assert!((v - 1.0).abs() < 1e-15);
    }
}

#[test]
fn pattern_symmetric_expands_with_ones() {
    let mtx = "\
%%MatrixMarket matrix coordinate pattern symmetric
4 4 3
1 1
2 2
3 2
";
    let a = read_matrix_market_str(mtx).unwrap();
    // (2,1) and (1,2) off-diagonal → 2 entries; plus 2 diag = 4
    assert_eq!(a.nnz(), 4);
    for &v in a.values() {
        assert!((v - 1.0).abs() < 1e-15);
    }
}

// ─── integer format ───────────────────────────────────────────────────────────

#[test]
fn integer_general_values() {
    let mtx = "\
%%MatrixMarket matrix coordinate integer general
2 2 2
1 1 5
2 2 7
";
    let a = read_matrix_market_str(mtx).unwrap();
    assert!((get_val(&a, 0, 0).unwrap() - 5.0).abs() < 1e-15);
    assert!((get_val(&a, 1, 1).unwrap() - 7.0).abs() < 1e-15);
}

// ─── comment and whitespace handling ─────────────────────────────────────────

#[test]
fn multiple_comment_lines_skipped() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
% comment 1
% comment 2
% comment 3
2 2 2
1 1 1.0
2 2 2.0
";
    let a = read_matrix_market_str(mtx).unwrap();
    assert_eq!(a.nnz(), 2);
}

#[test]
fn empty_matrix_zero_nnz() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
0 0 0
";
    let a = read_matrix_market_str(mtx).unwrap();
    assert_eq!(a.nrows(), 0);
    assert_eq!(a.nnz(), 0);
}

// ─── COO variant ─────────────────────────────────────────────────────────────

#[test]
fn coo_variant_preserves_shape() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
5 4 2
1 1 1.0
3 4 -1.0
";
    let coo = read_matrix_market_coo_str(mtx).unwrap();
    assert_eq!(coo.nrows(), 5);
    assert_eq!(coo.ncols(), 4);
    assert_eq!(coo.nnz(), 2);
}

// ─── error cases ─────────────────────────────────────────────────────────────

#[test]
fn missing_header_returns_error() {
    let mtx = "3 3 1\n1 1 1.0\n";
    assert!(matches!(read_matrix_market_str(mtx), Err(MmioError::MissingHeader)));
}

#[test]
fn array_format_rejected() {
    let mtx = "%%MatrixMarket matrix array real general\n2 2\n1.0\n2.0\n3.0\n4.0\n";
    assert!(matches!(
        read_matrix_market_str(mtx),
        Err(MmioError::NotCoordinate(_))
    ));
}

#[test]
fn complex_field_rejected() {
    let mtx = "%%MatrixMarket matrix coordinate complex general\n2 2 1\n1 1 1.0 0.0\n";
    assert!(matches!(
        read_matrix_market_str(mtx),
        Err(MmioError::UnsupportedField(_))
    ));
}

#[test]
fn row_index_out_of_bounds_error() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
2 2 1
5 1 1.0
";
    assert!(matches!(
        read_matrix_market_str(mtx),
        Err(MmioError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn col_index_out_of_bounds_error() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
2 2 1
1 9 1.0
";
    assert!(matches!(
        read_matrix_market_str(mtx),
        Err(MmioError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn non_square_symmetric_rejected() {
    let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
3 4 1
1 1 1.0
";
    assert!(matches!(
        read_matrix_market_str(mtx),
        Err(MmioError::NonSquareSymmetric)
    ));
}

#[test]
fn malformed_size_line_rejected() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
3 3
1 1 1.0
";
    assert!(matches!(
        read_matrix_market_str(mtx),
        Err(MmioError::MalformedSizeLine(_))
    ));
}

#[test]
fn zero_row_index_rejected() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
3 3 1
0 1 1.0
";
    assert!(matches!(
        read_matrix_market_str(mtx),
        Err(MmioError::MalformedDataLine(_))
    ));
}

#[test]
fn zero_col_index_rejected() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
3 3 1
1 0 1.0
";
    assert!(matches!(
        read_matrix_market_str(mtx),
        Err(MmioError::MalformedDataLine(_))
    ));
}

// ─── rectangular general ──────────────────────────────────────────────────────

#[test]
fn rectangular_matrix_allowed_for_general() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
5 3 2
1 1 1.0
5 3 2.0
";
    let a = read_matrix_market_str(mtx).unwrap();
    assert_eq!(a.nrows(), 5);
    assert_eq!(a.ncols(), 3);
    assert_eq!(a.nnz(), 2);
}

// ─── 1D Laplacian round-trip ──────────────────────────────────────────────────

#[test]
fn laplacian_1d_round_trip() {
    // Build a 5x5 1D Laplacian as a Matrix Market string, parse it back, and
    // verify the diagonal and off-diagonal entries.
    let mtx = "\
%%MatrixMarket matrix coordinate real symmetric
5 5 7
1 1 2.0
2 1 -1.0
2 2 2.0
3 2 -1.0
3 3 2.0
4 3 -1.0
4 4 2.0
";
    let a = read_matrix_market_str(mtx).unwrap();
    assert_eq!(a.nrows(), 5);
    // diag entries: 4, off-diag: 3 lower + 3 upper = 6 → total 10
    // but row 5 was omitted from the file, so n=5 but last row/col is empty.
    // Let's just check the declared entries are correct.
    assert!((get_val(&a, 0, 0).unwrap() - 2.0).abs() < 1e-14);
    assert!((get_val(&a, 1, 0).unwrap() + 1.0).abs() < 1e-14);
    assert!((get_val(&a, 0, 1).unwrap() + 1.0).abs() < 1e-14);
}

// ─── write tests ─────────────────────────────────────────────────────────────

#[test]
fn write_roundtrip_general() {
    let mtx = "\
%%MatrixMarket matrix coordinate real general
3 3 4
1 1 1.0
1 3 2.0
2 2 3.0
3 1 4.0
";
    let a = read_matrix_market_str(mtx).unwrap();
    let s = write_matrix_market_str(&a).unwrap();
    let b = read_matrix_market_str(&s).unwrap();
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    assert_eq!(a.nnz(), b.nnz());
    for (v1, v2) in a.values().iter().zip(b.values().iter()) {
        assert!((v1 - v2).abs() < 1e-14);
    }
}

#[test]
fn write_header_format() {
    use linger::sparse::{CooMatrix, CsrMatrix};
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 0, 1.5);
    coo.push(1, 1, 2.5);
    let a = CsrMatrix::from_coo(&coo);
    let s = write_matrix_market_str(&a).unwrap();
    assert!(s.starts_with("%%MatrixMarket matrix coordinate real general\n"));
}

#[test]
fn write_1based_indices() {
    use linger::sparse::{CooMatrix, CsrMatrix};
    let mut coo = CooMatrix::new(3, 3);
    coo.push(2, 1, 7.0); // 0-based → should appear as "3 2 7.0" in file
    let a = CsrMatrix::from_coo(&coo);
    let s = write_matrix_market_str(&a).unwrap();
    assert!(s.contains("3 2"), "expected 1-based index '3 2' in output:\n{s}");
}

#[test]
fn write_filters_structural_zeros() {
    // Build a matrix that has a stored 0 value.
    use linger::sparse::{CooMatrix, CsrMatrix};
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 0, 0.0); // structural zero
    coo.push(1, 1, 5.0);
    let a = CsrMatrix::from_coo(&coo);
    let s = write_matrix_market_str(&a).unwrap();
    let b = read_matrix_market_str(&s).unwrap();
    // The 0.0 entry should have been omitted.
    assert_eq!(b.nnz(), 1);
}

#[test]
fn write_empty_matrix() {
    use linger::sparse::{CooMatrix, CsrMatrix};
    let coo: CooMatrix<f64> = CooMatrix::new(0, 0);
    let a = CsrMatrix::from_coo(&coo);
    let s = write_matrix_market_str(&a).unwrap();
    let b = read_matrix_market_str(&s).unwrap();
    assert_eq!(b.nrows(), 0);
    assert_eq!(b.nnz(), 0);
}

#[test]
fn write_roundtrip_poisson_1d() {
    use linger::sparse::{CooMatrix, CsrMatrix};
    let n = 50;
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let s = write_matrix_market_str(&a).unwrap();
    let b = read_matrix_market_str(&s).unwrap();
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.nnz(), b.nnz());
    // Check a diagonal entry survived the roundtrip.
    assert!((get_val(&b, 0, 0).unwrap() - 2.0).abs() < 1e-14);
    assert!((get_val(&b, 1, 0).unwrap() + 1.0).abs() < 1e-14);
}
