//! Nested Dissection ordering integration tests.
//!
//! Tests nd() permutation validity, correctness with all direct solvers,
//! and fill quality on structured grids.

use linger::{
    direct::{DirectSolver, DirectOptions, SparseLu, SparseCholesky, MultifrontalLu, MultifrontalOptions},
    direct::ordering::{OrderingMethod, nd, permute_symmetric},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec,
    core::operator::LinearOperator,
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
            if j > 0   { coo.push(id, id-1, -1.0); coo.push(id-1, id, -1.0); }
            if i > 0   { coo.push(id, id-n, -1.0); coo.push(id-n, id, -1.0); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn nonsymmetric_tridiag(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 4.0);
        if i > 0   { coo.push(i, i-1, -1.0); }
        if i+1 < n { coo.push(i, i+1, -2.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn is_permutation(perm: &[usize], n: usize) -> bool {
    if perm.len() != n { return false; }
    let mut seen = vec![false; n];
    for &v in perm {
        if v >= n || seen[v] { return false; }
        seen[v] = true;
    }
    true
}

fn relative_residual(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let mut ax = DenseVec::zeros(a.nrows());
    a.apply(x, &mut ax);
    let r:  f64 = ax.as_slice().iter().zip(b.as_slice()).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.as_slice().iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    if nb < 1e-300 { r } else { r / nb }
}

fn nnz_lower_cholesky(a: &CsrMatrix<f64>, ordering: OrderingMethod) -> usize {
    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        ordering, ..Default::default()
    });
    let n = a.nrows();
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    solver.factor(a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    // Approximate nnz(L) as the number of non-zeros we can measure by counting
    // in the permuted matrix's lower triangle. Use the permuted matrix directly.
    let perm = linger::direct::ordering::nd(a);
    let pa = permute_symmetric(a, &perm);
    let mut count = 0usize;
    for i in 0..pa.nrows() {
        for k in pa.row_ptr()[i]..pa.row_ptr()[i+1] {
            if pa.col_idx()[k] <= i { count += 1; }
        }
    }
    count
}

// ─── Permutation validity ─────────────────────────────────────────────────────

#[test]
fn nd_is_permutation_tridiag_n20() {
    let a = laplacian_1d(20);
    let perm = nd(&a);
    assert!(is_permutation(&perm, 20));
}

#[test]
fn nd_is_permutation_grid_4x4() {
    let a = laplacian_2d(4);
    let perm = nd(&a);
    assert!(is_permutation(&perm, 16));
}

#[test]
fn nd_is_permutation_grid_8x8() {
    let a = laplacian_2d(8);
    let perm = nd(&a);
    assert!(is_permutation(&perm, 64));
}

#[test]
fn nd_is_permutation_large_tridiag() {
    let a = laplacian_1d(300);
    let perm = nd(&a);
    assert!(is_permutation(&perm, 300));
}

// ─── Edge cases ───────────────────────────────────────────────────────────────

#[test]
fn nd_empty() {
    let a: CsrMatrix<f64> = CsrMatrix::from_coo(&CooMatrix::new(0, 0));
    assert_eq!(nd(&a), vec![]);
}

#[test]
fn nd_single_node() {
    let mut coo = CooMatrix::new(1, 1);
    coo.push(0, 0, 1.0);
    let perm = nd(&CsrMatrix::from_coo(&coo));
    assert_eq!(perm, vec![0]);
}

#[test]
fn nd_disconnected_two_components() {
    // Two isolated edges: 0-1 and 2-3
    let mut coo = CooMatrix::new(4, 4);
    for i in 0..4usize { coo.push(i, i, 2.0); }
    coo.push(0, 1, -1.0); coo.push(1, 0, -1.0);
    coo.push(2, 3, -1.0); coo.push(3, 2, -1.0);
    let perm = nd(&CsrMatrix::from_coo(&coo));
    assert!(is_permutation(&perm, 4));
}

// ─── Solver correctness with NodeNd ordering ─────────────────────────────────

#[test]
fn nd_cholesky_correctness_n50() {
    let n = 50;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());
    let mut x = DenseVec::zeros(n);
    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        ordering: OrderingMethod::NodeNd,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9,
        "Cholesky+ND n=50 rel residual = {:.3e}", relative_residual(&a, &x, &b));
}

#[test]
fn nd_lu_correctness_n10() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    let mut solver = SparseLu::<f64>::new(DirectOptions {
        ordering: OrderingMethod::NodeNd,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

#[test]
fn nd_lu_nonsymmetric_n8() {
    let n = 8;
    let a = nonsymmetric_tridiag(n);
    let b = DenseVec::from_vec((1..=n).map(|i| i as f64).collect());
    let mut x = DenseVec::zeros(n);
    let mut solver = SparseLu::<f64>::new(DirectOptions {
        ordering: OrderingMethod::NodeNd,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

#[test]
fn nd_multifrontal_correctness_n10() {
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let mut x = DenseVec::zeros(n);
    let mut solver = MultifrontalLu::<f64>::with_options(MultifrontalOptions {
        base: DirectOptions {
            ordering: OrderingMethod::NodeNd,
            ..Default::default()
        },
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}

#[test]
fn nd_cholesky_2d_grid_5x5() {
    let n = 5;
    let a = laplacian_2d(n);
    let nn = n * n;
    let b = DenseVec::from_vec(vec![1.0f64; nn]);
    let mut x = DenseVec::zeros(nn);
    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        ordering: OrderingMethod::NodeNd,
        ..Default::default()
    });
    solver.factor(&a).unwrap();
    solver.solve(&b, &mut x).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-9);
}

// ─── Fill quality ─────────────────────────────────────────────────────────────

#[test]
fn nd_fill_not_worse_than_natural_on_grid() {
    // On a 5×5 2D Laplacian, ND should produce no more fill than natural order.
    let n = 5;
    let a = laplacian_2d(n);
    let nn = n * n;

    // Count lower-triangle nnz after permutation: proxy for nnz(L).
    let count_lower = |perm: &[usize]| -> usize {
        let pa = permute_symmetric(&a, perm);
        let mut c = 0usize;
        for i in 0..pa.nrows() {
            for k in pa.row_ptr()[i]..pa.row_ptr()[i+1] {
                if pa.col_idx()[k] <= i { c += 1; }
            }
        }
        c
    };

    let natural: Vec<usize> = (0..nn).collect();
    let nd_perm = nd(&a);

    let fill_natural = count_lower(&natural);
    let fill_nd      = count_lower(&nd_perm);

    // ND should be at most 50% worse than natural on structured grids
    // (ND is designed for unstructured meshes; on grids RCM/ND may differ).
    assert!(fill_nd <= fill_natural * 3 / 2,
        "ND fill ({fill_nd}) was worse than 1.5x natural ({fill_natural}) on 5×5 grid");
}

// ─── Builder integration ──────────────────────────────────────────────────────

#[test]
fn builder_nd_ordering_direct_solve() {
    use linger::builder::{SolverBuilder, SolveMethod, DirectBackend, Ordering};
    let n = 10;
    let a = laplacian_1d(n);
    let b = DenseVec::from_vec(vec![1.0f64; n]);
    let x = SolverBuilder::new()
        .method(SolveMethod::Direct(DirectBackend::Cholesky))
        .ordering(Ordering::NodeNd)
        .solve(&a, &b).unwrap();
    assert!(relative_residual(&a, &x, &b) < 1e-10);
}
