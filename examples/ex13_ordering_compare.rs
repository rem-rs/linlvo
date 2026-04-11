//! ex13 - compare fill-reducing orderings for direct solvers.

use linger::{
    direct::{ordering::{colamd, nd, permute_symmetric, rcm}, DirectOptions, DirectSolver, SparseCholesky},
    OrderingMethod,
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, LinearOperator,
};

fn laplacian_2d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n;
    let mut coo = CooMatrix::new(nn, nn);
    for i in 0..n {
        for j in 0..n {
            let id = i * n + j;
            coo.push(id, id, 4.0);
            if j > 0 {
                coo.push(id, id - 1, -1.0);
                coo.push(id - 1, id, -1.0);
            }
            if i > 0 {
                coo.push(id, id - n, -1.0);
                coo.push(id - n, id, -1.0);
            }
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn profile_metrics(a: &CsrMatrix<f64>) -> (usize, usize) {
    let mut bandwidth = 0;
    let mut profile = 0;
    for i in 0..a.nrows() {
        let mut first_col = i;
        for k in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let j = a.col_idx()[k];
            let diff = i.abs_diff(j);
            bandwidth = bandwidth.max(diff);
            if j <= i {
                first_col = first_col.min(j);
            }
        }
        profile += i - first_col;
    }
    (bandwidth, profile)
}

fn relative_residual(a: &CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let mut ax = DenseVec::zeros(a.nrows());
    a.apply(x, &mut ax);
    let num = ax
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&lhs, &rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let den = b.as_slice().iter().map(|&v| v.powi(2)).sum::<f64>().sqrt();
    if den == 0.0 { num } else { num / den }
}

fn run_solver(ordering: OrderingMethod, a: &CsrMatrix<f64>, b: &DenseVec<f64>) -> f64 {
    let mut solver = SparseCholesky::<f64>::new(DirectOptions {
        ordering,
        ..Default::default()
    });
    let mut x = DenseVec::zeros(a.nrows());
    solver.factor(a).unwrap();
    solver.solve(b, &mut x).unwrap();
    relative_residual(a, &x, b)
}

fn main() {
    let n = 8;
    let a = laplacian_2d(n);
    let b = DenseVec::from_vec(vec![1.0_f64; a.nrows()]);
    let natural: Vec<usize> = (0..a.nrows()).collect();
    let rcm_perm = rcm(&a);
    let colamd_perm = colamd(&a);
    let nd_perm = nd(&a);

    println!("ex13: ordering comparison");
    println!("  system: 2-D Laplacian on {}x{}, n={}, nnz={}", n, n, a.nrows(), a.nnz());

    let orderings = [
        ("Natural", natural, OrderingMethod::Natural),
        ("RCM", rcm_perm, OrderingMethod::Rcm),
        ("COLAMD", colamd_perm, OrderingMethod::Colamd),
        ("NodeNd", nd_perm, OrderingMethod::NodeNd),
    ];

    let natural_pa = permute_symmetric(&a, &(0..a.nrows()).collect::<Vec<_>>());
    let (natural_bw, natural_profile) = profile_metrics(&natural_pa);

    for (name, perm, ordering) in orderings {
        let pa = permute_symmetric(&a, &perm);
        let (bw, profile) = profile_metrics(&pa);
        let rel = run_solver(ordering, &a, &b);
        println!(
            "  {:>7}: bandwidth={} ({:.3}x) profile={} ({:.3}x) rel_res={:.3e}",
            name,
            bw,
            bw as f64 / natural_bw as f64,
            profile,
            profile as f64 / natural_profile as f64,
            rel
        );
        assert!(rel < 1e-9);
    }

    println!("  OK");
}