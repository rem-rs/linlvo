//! MPI distributed conjugate gradient — multi-rank demonstration.
//!
//! Build and run with:
//! ```text
//! mpiexec -n 2 cargo run --features mpi --example ex_mpi_dist_cg
//! ```

fn main() {
    let universe = mpi::initialize().expect("MPI initialisation failed");
    let world = universe.world();
    let rank   = world.rank() as usize;
    let nranks = world.size() as usize;

    let n_global: usize = 8 * nranks;
    let (row_ptr, col_ind, values) = build_poisson_1d(n_global);

    use linger::sparse::CsrMatrix;
    use linger::parallel_dist::{
        DistCsrMatrix, dist_cg, DistCgParams,
        mpi_halo::{MpiHaloExchange, MpiReduce},
    };

    let global_csr = CsrMatrix::from_raw(n_global, n_global, row_ptr, col_ind, values);
    let dist = DistCsrMatrix::from_global_csr_block_partition(&global_csr, nranks, rank)
        .expect("failed to distribute matrix");

    let halo   = MpiHaloExchange::from_dist_matrix(&dist, &world);
    let reduce = MpiReduce;

    let local_n = dist.layout().local_size();
    let b: Vec<f64> = (0..local_n).map(|i| {
        let gi = dist.layout().owned_global_range.start + i;
        if gi == 0 || gi == n_global - 1 { 1.0 } else { 0.0 }
    }).collect();
    let mut x = vec![0.0_f64; local_n];

    let result = dist_cg(&dist, &halo, &reduce, &b, &mut x, &DistCgParams::default())
        .expect("dist_cg failed");

    if rank == 0 {
        println!(
            "Converged in {} iterations, final residual {:.3e}",
            result.iters, result.residual_norm
        );
    }

    let mut ax = vec![0.0_f64; local_n];
    dist.spmv_with_halo(&x, &halo, &mut ax).expect("spmv failed");
    let local_res: f64 = ax.iter().zip(b.iter())
        .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();

    use mpi::traits::*;
    use mpi::collective::SystemOperation;
    let mut global_res = 0.0_f64;
    world.all_reduce_into(&local_res, &mut global_res, &SystemOperation::max());

    if rank == 0 {
        println!("Global residual (max over ranks): {:.3e}", global_res);
        assert!(global_res < 1e-8, "residual too large: {:.3e}", global_res);
        println!("OK");
    }
}

fn build_poisson_1d(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut row_ptr = vec![0usize];
    let mut col_ind = Vec::new();
    let mut values  = Vec::new();
    for i in 0..n {
        if i == 0 || i == n - 1 {
            col_ind.push(i); values.push(1.0);
        } else {
            col_ind.push(i - 1); values.push(-1.0);
            col_ind.push(i);     values.push(2.0);
            col_ind.push(i + 1); values.push(-1.0);
        }
        row_ptr.push(col_ind.len());
    }
    (row_ptr, col_ind, values)
}
