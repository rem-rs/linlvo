//! E4: Property-based tests for sparse format round-trips and SpMV linearity.

use linger::sparse::{CooMatrix, CsrMatrix};
use proptest::prelude::*;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build an n×n dense matrix from COO entries (for reference computation).
fn to_dense(entries: &[(usize, usize, f64)], n: usize) -> Vec<f64> {
    let mut d = vec![0.0f64; n * n];
    for &(r, c, v) in entries {
        d[r * n + c] += v;
    }
    d
}

/// Dense matrix-vector product.
fn dense_matvec(dense: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    (0..n).map(|r| (0..n).map(|c| dense[r * n + c] * x[c]).sum()).collect()
}

/// Euclidean norm.
fn norm2(v: &[f64]) -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() }

/// Generate a random n×n CSR matrix with `nnz` nonzeros in [1, n²].
fn arb_csr_matrix(n: usize, nnz: usize) -> impl Strategy<Value = (CsrMatrix<f64>, Vec<(usize, usize, f64)>)> {
    proptest::collection::vec(
        (0..n, 0..n, -10.0f64..=10.0f64),
        1..=nnz,
    ).prop_map(move |entries| {
        let mut coo = CooMatrix::<f64>::new(n, n);
        for (r, c, v) in &entries {
            coo.push(*r, *c, *v);
        }
        let mat = CsrMatrix::from_coo(&coo);
        (mat, entries)
    })
}

// ─── Test 1: COO → CSR → dense equals original dense ─────────────────────────

proptest! {
    #[test]
    fn coo_to_csr_to_dense_matches(
        entries in proptest::collection::vec((0usize..8, 0usize..8, -10.0f64..=10.0f64), 1..=30),
    ) {
        let n = 8usize;
        let ref_dense = to_dense(&entries, n);

        let mut coo = CooMatrix::<f64>::new(n, n);
        for &(r, c, v) in &entries { coo.push(r, c, v); }
        let csr = CsrMatrix::from_coo(&coo);

        // Check validate passes.
        prop_assert!(csr.validate().is_ok(), "CSR from COO failed validate()");

        // Reconstruct dense from CSR triplets.
        let mut csr_dense = vec![0.0f64; n * n];
        for (r, c, v) in csr.triplets() { csr_dense[r * n + c] += v; }

        let diff = norm2(&ref_dense.iter().zip(&csr_dense).map(|(a, b)| a - b).collect::<Vec<_>>());
        prop_assert!(diff < 1e-10, "COO→CSR→dense mismatch: diff={diff:.3e}");
    }
}

// ─── Test 2: CSR transpose twice = identity ───────────────────────────────────

proptest! {
    #[test]
    fn csr_transpose_twice_identity(
        entries in proptest::collection::vec((0usize..6, 0usize..6, -5.0f64..=5.0f64), 1..=20),
    ) {
        let n = 6usize;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for &(r, c, v) in &entries { coo.push(r, c, v); }
        let a = CsrMatrix::from_coo(&coo);
        let at  = a.transpose_csr();
        let att = at.transpose_csr();

        // A and Aᵀᵀ should have the same sparsity pattern and values.
        prop_assert_eq!(a.nnz(), att.nnz());
        // Compare SpMV: A*x == Aᵀᵀ*x for random x=[1,2,...,n].
        let x_vec: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let xd = linger::DenseVec::from_vec(x_vec.clone());
        let mut ax  = linger::DenseVec::zeros(n);
        let mut attx = linger::DenseVec::zeros(n);
        use linger::LinearOperator;
        a.apply(&xd, &mut ax);
        att.apply(&xd, &mut attx);
        let diff = norm2(&ax.as_slice().iter().zip(attx.as_slice()).map(|(a, b)| a - b).collect::<Vec<_>>());
        prop_assert!(diff < 1e-10, "Aᵀᵀ ≠ A: diff={diff:.3e}");
    }
}

// ─── Test 3: COO with duplicates → no dup col_idx per row after CSR ──────────

proptest! {
    #[test]
    fn coo_duplicates_merged_in_csr(
        entries in proptest::collection::vec((0usize..5, 0usize..5, -3.0f64..=3.0f64), 5..=25),
    ) {
        let n = 5usize;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for &(r, c, v) in &entries { coo.push(r, c, v); }
        let csr = CsrMatrix::from_coo(&coo);
        // validate() checks strictly increasing col_idx per row.
        prop_assert!(csr.validate().is_ok(), "CSR with random duplicates failed validate()");
    }
}

// ─── Test 4: SpMV linearity: A*(x+y) = A*x + A*y ────────────────────────────

proptest! {
    #[test]
    fn spmv_linearity_addition(
        entries in proptest::collection::vec((0usize..7, 0usize..7, -5.0f64..=5.0f64), 1..=25),
        x_vals in proptest::collection::vec(-10.0f64..=10.0f64, 7usize..=7),
        y_vals in proptest::collection::vec(-10.0f64..=10.0f64, 7usize..=7),
    ) {
        let n = 7usize;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for &(r, c, v) in &entries { coo.push(r, c, v); }
        let a = CsrMatrix::from_coo(&coo);

        use linger::LinearOperator;
        let xd = linger::DenseVec::from_vec(x_vals.clone());
        let yd = linger::DenseVec::from_vec(y_vals.clone());
        let xy: Vec<f64> = x_vals.iter().zip(&y_vals).map(|(x, y)| x + y).collect();
        let xyd = linger::DenseVec::from_vec(xy);

        let mut ax  = linger::DenseVec::zeros(n);
        let mut ay  = linger::DenseVec::zeros(n);
        let mut axy = linger::DenseVec::zeros(n);
        a.apply(&xd,  &mut ax);
        a.apply(&yd,  &mut ay);
        a.apply(&xyd, &mut axy);

        // A*(x+y) should equal A*x + A*y
        let diff_vals: Vec<f64> = axy.as_slice().iter()
            .zip(ax.as_slice()).zip(ay.as_slice())
            .map(|((axy, ax), ay)| axy - ax - ay)
            .collect();
        let diff = norm2(&diff_vals);
        prop_assert!(diff < 1e-9, "SpMV linearity violation: ‖A(x+y)-Ax-Ay‖={diff:.3e}");
    }
}

// ─── Test 5: SpMV distributivity: (A+D)*x = A*x + D*x ───────────────────────

proptest! {
    #[test]
    fn spmv_distributivity_diagonal_plus_sparse(
        entries in proptest::collection::vec((0usize..6, 0usize..6, -4.0f64..=4.0f64), 1..=20),
        diag  in proptest::collection::vec(0.5f64..=5.0f64, 6usize..=6),
        x_vals in proptest::collection::vec(-5.0f64..=5.0f64, 6usize..=6),
    ) {
        let n = 6usize;
        use linger::LinearOperator;

        let mut coo_a = CooMatrix::<f64>::new(n, n);
        for &(r, c, v) in &entries { coo_a.push(r, c, v); }
        let a = CsrMatrix::from_coo(&coo_a);

        // D is a diagonal matrix.
        let mut coo_d = CooMatrix::<f64>::new(n, n);
        for (i, &d) in diag.iter().enumerate() { coo_d.push(i, i, d); }
        let d_mat = CsrMatrix::from_coo(&coo_d);

        // Build A+D via COO.
        let mut coo_sum = CooMatrix::<f64>::new(n, n);
        for (r, c, v) in a.triplets()    { coo_sum.push(r, c, v); }
        for (r, c, v) in d_mat.triplets() { coo_sum.push(r, c, v); }
        let apd = CsrMatrix::from_coo(&coo_sum);

        let xd = linger::DenseVec::from_vec(x_vals.clone());
        let mut ax   = linger::DenseVec::zeros(n);
        let mut dx   = linger::DenseVec::zeros(n);
        let mut apdx = linger::DenseVec::zeros(n);
        a.apply(&xd,   &mut ax);
        d_mat.apply(&xd,   &mut dx);
        apd.apply(&xd, &mut apdx);

        let diff_vals: Vec<f64> = apdx.as_slice().iter()
            .zip(ax.as_slice()).zip(dx.as_slice())
            .map(|((apdx, ax), dx)| apdx - ax - dx)
            .collect();
        let diff = norm2(&diff_vals);
        prop_assert!(diff < 1e-9, "SpMV distributivity violation: ‖(A+D)x-Ax-Dx‖={diff:.3e}");
    }
}
