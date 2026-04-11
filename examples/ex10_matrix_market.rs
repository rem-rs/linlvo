//! ex10 - Matrix Market read/write round-trip and solve.

use std::{env, fs};

use linger::{
    sparse::{read_matrix_market, read_matrix_market_coo_str, read_matrix_market_str, write_matrix_market, write_matrix_market_str},
    ConjugateGradient, DenseVec, KrylovSolver, SolverParams, VerboseLevel,
};

fn rel_residual(a: &linger::sparse::CsrMatrix<f64>, x: &DenseVec<f64>, b: &DenseVec<f64>) -> f64 {
    let mut ax = vec![0.0_f64; a.nrows()];
    a.spmv(x.as_slice(), &mut ax);
    let num = ax
        .iter()
        .zip(b.as_slice())
        .map(|(&lhs, &rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let den = b.as_slice().iter().map(|&v| v * v).sum::<f64>().sqrt();
    if den == 0.0 { num } else { num / den }
}

fn main() {
    let mtx = "%%MatrixMarket matrix coordinate real symmetric
% 1-D Laplacian with lower triangle stored only
5 5 9
1 1 2.0
2 1 -1.0
2 2 2.0
3 2 -1.0
3 3 2.0
4 3 -1.0
4 4 2.0
5 4 -1.0
5 5 2.0
";

    println!("ex10: Matrix Market round-trip");

    let coo = read_matrix_market_coo_str(mtx).unwrap();
    let a = read_matrix_market_str(mtx).unwrap();
    println!("  parsed: {}x{} nnz={} (expanded from symmetric input)", a.nrows(), a.ncols(), a.nnz());
    assert_eq!(coo.nrows(), 5);
    assert_eq!(a.nnz(), 13);

    let mmio_string = write_matrix_market_str(&a).unwrap();
    let roundtrip = read_matrix_market_str(&mmio_string).unwrap();
    println!("  string round-trip: output bytes={}", mmio_string.len());
    assert_eq!(roundtrip.nrows(), a.nrows());
    assert_eq!(roundtrip.ncols(), a.ncols());
    assert_eq!(roundtrip.nnz(), a.nnz());

    let temp_path = env::temp_dir().join("linger_ex10_matrix_market.mtx");
    write_matrix_market(&temp_path, &a).unwrap();
    let from_file = read_matrix_market(&temp_path).unwrap();
    let metadata = fs::metadata(&temp_path).unwrap();
    println!("  file round-trip: {} bytes -> {}", metadata.len(), temp_path.display());
    fs::remove_file(&temp_path).unwrap();
    assert_eq!(from_file.nnz(), a.nnz());

    let b = DenseVec::from_vec(vec![1.0_f64; a.nrows()]);
    let mut x = DenseVec::zeros(a.nrows());
    let result = ConjugateGradient::<f64>::default()
        .solve(
            &from_file,
            None,
            &b,
            &mut x,
            &SolverParams {
                rtol: 1e-10,
                max_iter: 100,
                verbose: VerboseLevel::Silent,
                ..Default::default()
            },
        )
        .unwrap();
    let rel = rel_residual(&from_file, &x, &b);

    println!(
        "  CG on parsed matrix: converged={} iters={} rel_res={:.3e}",
        result.converged,
        result.iterations,
        rel
    );
    assert!(result.converged);
    assert!(rel < 1e-9);

    println!("  OK");
}