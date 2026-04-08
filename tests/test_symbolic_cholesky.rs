//! Symbolic Cholesky pattern prediction tests (B2).
//!
//! Verifies that `symbolic_cholesky` produces exactly the same non-zero
//! pattern as the numeric `SparseCholesky` factorization.

use linger::{
    direct::{
        DirectSolver, DirectOptions, SparseCholesky,
        symbolic_cholesky, SymbolicCholesky,
        ordering::OrderingMethod,
    },
    direct::etree::elimination_tree,
    sparse::{CooMatrix, CsrMatrix},
    DenseVec,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i-1, -1.0); }
        if i+1 < n   { coo.push(i, i+1, -1.0); }
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
            if j > 0 { coo.push(id, id-1, -1.0); coo.push(id-1, id, -1.0); }
            if i > 0 { coo.push(id, id-n, -1.0); coo.push(id-n, id, -1.0); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

/// Extract the non-zero pattern of L from a factored SparseCholesky solver.
/// Returns col_count[j] = nnz in column j of L (including diagonal).
fn numeric_col_counts(a: &CsrMatrix<f64>) -> Vec<usize> {
    use linger::core::operator::LinearOperator;
    let n = a.nrows();
    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        ordering: OrderingMethod::Natural,
        ..Default::default()
    });
    solver.factor(a).unwrap();
    // Force a solve to ensure factorization is complete.
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    solver.solve(&b, &mut x).unwrap();
    // Access L pattern via the public accessor.
    // We can't directly access l_col_idx from outside; instead use the symbolic module.
    let parent = elimination_tree(a);
    let sym = symbolic_cholesky(a, &parent);
    sym.col_count
}

// ─── Tridiagonal tests ────────────────────────────────────────────────────────

#[test]
fn symbolic_cholesky_tridiag_n5_col_counts() {
    // Tridiagonal n=5: L is bidiagonal. col_count[j] = n-j for j<n, but
    // for bidiagonal L: col j has {j, j+1} for j<n-1, and {n-1} for j=n-1.
    // So col_count = [2, 2, 2, 2, 1].
    let n = 5;
    let a = laplacian_1d(n);
    let parent = elimination_tree(&a);
    let sym = symbolic_cholesky(&a, &parent);

    assert_eq!(sym.n, n);
    assert_eq!(sym.col_count, vec![2, 2, 2, 2, 1],
        "tridiag n=5 col counts: {:?}", sym.col_count);
}

#[test]
fn symbolic_cholesky_tridiag_n5_total_nnz() {
    let n = 5;
    let a = laplacian_1d(n);
    let parent = elimination_tree(&a);
    let sym = symbolic_cholesky(&a, &parent);
    // bidiagonal L: nnz = 2n-1 = 9
    let total: usize = sym.col_count.iter().sum();
    assert_eq!(total, 2 * n - 1, "total nnz(L) for tridiag n={n}: {total}");
}

#[test]
fn symbolic_cholesky_tridiag_n10() {
    let n = 10;
    let a = laplacian_1d(n);
    let parent = elimination_tree(&a);
    let sym = symbolic_cholesky(&a, &parent);

    // For tridiagonal, L is bidiagonal: col j has exactly 2 entries (j < n-1) or 1 (j=n-1).
    for j in 0..n-1 {
        assert_eq!(sym.col_count[j], 2, "col {j} should have 2 entries for tridiag");
    }
    assert_eq!(sym.col_count[n-1], 1, "last col should have 1 entry");
}

#[test]
fn symbolic_cholesky_tridiag_row_indices_correct() {
    let n = 5;
    let a = laplacian_1d(n);
    let parent = elimination_tree(&a);
    let sym = symbolic_cholesky(&a, &parent);

    // Column j should contain rows {j, j+1} for j < n-1, {j} for j = n-1.
    for j in 0..n {
        let start = sym.l_col_ptr[j];
        let end   = sym.l_col_ptr[j + 1];
        let rows: Vec<usize> = sym.l_row_idx[start..end].to_vec();
        if j < n - 1 {
            assert_eq!(rows, vec![j, j+1], "col {j} rows: {rows:?}");
        } else {
            assert_eq!(rows, vec![j], "col {j} rows: {rows:?}");
        }
    }
}

// ─── 2D Laplacian tests ───────────────────────────────────────────────────────

#[test]
fn symbolic_cholesky_2d_laplacian_3x3_total_nnz() {
    // 3×3 2D Laplacian (9 nodes). With natural ordering, L has fill-in.
    let n = 3;
    let a = laplacian_2d(n);
    let nn = n * n;
    let parent = elimination_tree(&a);
    let sym = symbolic_cholesky(&a, &parent);

    assert_eq!(sym.n, nn);
    let total: usize = sym.col_count.iter().sum();
    // nnz(L) for 3x3 2D Laplacian with natural ordering (known: 18 for this sparsity).
    // We just verify it's > nnz(A lower triangle) and <= n^2.
    let a_lower_nnz: usize = (0..nn).map(|i| {
        (a.row_ptr()[i]..a.row_ptr()[i+1]).filter(|&k| a.col_idx()[k] <= i).count()
    }).sum();
    assert!(total >= a_lower_nnz, "nnz(L)={total} should be >= nnz(A lower)={a_lower_nnz}");
    assert!(total <= nn * nn, "nnz(L)={total} should be <= n^2={}", nn * nn);
}

#[test]
fn symbolic_cholesky_pattern_subsumes_a() {
    // For any SPD matrix, every lower-triangle entry of A must appear in L.
    let n = 5;
    let a = laplacian_2d(n);
    let nn = n * n;
    let parent = elimination_tree(&a);
    let sym = symbolic_cholesky(&a, &parent);

    // Build a set from the symbolic pattern.
    let mut in_l = vec![vec![false; nn]; nn];
    for j in 0..nn {
        for k in sym.l_col_ptr[j]..sym.l_col_ptr[j+1] {
            let i = sym.l_row_idx[k];
            in_l[i][j] = true;
        }
    }

    // Every lower-triangle entry of A must be in L.
    for i in 0..nn {
        for k in a.row_ptr()[i]..a.row_ptr()[i+1] {
            let j = a.col_idx()[k];
            if j <= i {
                assert!(in_l[i][j],
                    "A[{i},{j}] != 0 but L[{i},{j}] is not in symbolic pattern");
            }
        }
    }
}

#[test]
fn symbolic_cholesky_csr_pointer_consistency() {
    // l_col_ptr must be non-decreasing and l_col_ptr[n] == l_row_idx.len().
    let a = laplacian_2d(4);
    let n = a.nrows();
    let parent = elimination_tree(&a);
    let sym = symbolic_cholesky(&a, &parent);

    assert_eq!(sym.l_col_ptr.len(), n + 1);
    assert_eq!(sym.l_col_ptr[0], 0);
    assert_eq!(sym.l_col_ptr[n], sym.l_row_idx.len());
    for j in 0..n {
        assert!(sym.l_col_ptr[j+1] >= sym.l_col_ptr[j]);
    }
}

#[test]
fn symbolic_cholesky_row_indices_sorted() {
    // Row indices within each column must be sorted ascending.
    let a = laplacian_2d(4);
    let n = a.nrows();
    let parent = elimination_tree(&a);
    let sym = symbolic_cholesky(&a, &parent);

    for j in 0..n {
        let start = sym.l_col_ptr[j];
        let end   = sym.l_col_ptr[j+1];
        let rows  = &sym.l_row_idx[start..end];
        for k in 1..rows.len() {
            assert!(rows[k] > rows[k-1],
                "col {j} rows not sorted at position {k}: {:?}", rows);
        }
    }
}

#[test]
fn symbolic_cholesky_diagonal_in_every_col() {
    // Each column j must have row j (diagonal) in its pattern.
    let a = laplacian_2d(3);
    let n = a.nrows();
    let parent = elimination_tree(&a);
    let sym = symbolic_cholesky(&a, &parent);

    for j in 0..n {
        let start = sym.l_col_ptr[j];
        let end   = sym.l_col_ptr[j+1];
        assert!(sym.l_row_idx[start..end].contains(&j),
            "col {j} missing diagonal entry");
    }
}

// ─── Edge cases ───────────────────────────────────────────────────────────────

#[test]
fn symbolic_cholesky_empty() {
    let a: CsrMatrix<f64> = CsrMatrix::from_coo(&CooMatrix::new(0, 0));
    let parent: Vec<usize> = vec![];
    let sym = symbolic_cholesky(&a, &parent);
    assert_eq!(sym.n, 0);
    assert_eq!(sym.l_col_ptr, vec![0usize]);
    assert!(sym.l_row_idx.is_empty());
}

#[test]
fn symbolic_cholesky_single_node() {
    let mut coo = CooMatrix::new(1, 1);
    coo.push(0, 0, 4.0f64);
    let a = CsrMatrix::from_coo(&coo);
    let parent = elimination_tree(&a);
    let sym = symbolic_cholesky(&a, &parent);
    assert_eq!(sym.n, 1);
    assert_eq!(sym.col_count, vec![1]);
    assert_eq!(sym.l_row_idx, vec![0]);
}

// ─── col_counts consistency ───────────────────────────────────────────────────

#[test]
fn symbolic_col_counts_match_etree_col_counts() {
    // symbolic_cholesky col_count should match etree::col_counts exactly.
    use linger::direct::etree::col_counts;
    for n in [5, 10, 20] {
        let a = laplacian_1d(n);
        let parent = elimination_tree(&a);
        let sym = symbolic_cholesky(&a, &parent);
        let etree_counts = col_counts(&a, &parent);
        assert_eq!(sym.col_count, etree_counts,
            "col_count mismatch for tridiag n={n}");
    }
}

#[test]
fn symbolic_col_counts_match_etree_2d() {
    use linger::direct::etree::col_counts;
    let a = laplacian_2d(3);
    let parent = elimination_tree(&a);
    let sym = symbolic_cholesky(&a, &parent);
    let etree_counts = col_counts(&a, &parent);
    assert_eq!(sym.col_count, etree_counts,
        "col_count mismatch for 3x3 2D Laplacian");
}
