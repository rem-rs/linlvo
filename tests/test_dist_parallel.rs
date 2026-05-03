//! Integration tests for distributed parallel scaffolding (Sprint 6).
//!
//! These tests run in single-process mode using `LocalHaloExchange` and
//! `LocalReduce`, which allows testing the distributed-CG algorithm and
//! halo-exchange infrastructure without MPI.
//!
//! Covered scenarios:
//! 1. `block_partition` — covers all DOFs exactly once across ranks.
//! 2. `PartitionLayout` — correct owned/ghost metadata.
//! 3. `DistCsrMatrix::from_global_csr_block_partition` — single rank.
//! 4. `DistCsrMatrix::spmv_with_halo` — correctness vs. global CSR.
//! 5. `dist_cg` — convergence on 1D Poisson (n=16).
//! 6. `dist_cg` — correctness: solution matches dense reference.
//! 7. `dist_cg` — early stopping if initial x is already the solution.
//! 8. `LocalHaloExchange::exchange` — returns correct ghost values.
//! 9. `HaloPlan` — validates duplicate-rank check.
//! 10. `dist_cg` on larger 1D problem (n=128).

use linger::parallel_dist::{
    block_partition, PartitionLayout,
    DistCsrMatrix, LocalHaloExchange,
    dist_cg, DistCgParams,
    LocalReduce,
    HaloExchange, HaloPlan, NeighborHaloPlan,
};
use linger::sparse::{CooMatrix, CsrMatrix};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn poisson_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

/// Solve the global system with a simple Jacobi-style iteration reference
/// (just used to check the distributed solver agrees).
fn dense_solve_diag_dom(a: &CsrMatrix<f64>, b: &[f64], tol: f64) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    // Extract diagonal
    let rp = a.row_ptr();
    let ci = a.col_idx();
    let vl = a.values();
    // Build full dense system and use Gauss-Seidel to get a reference solution
    for _ in 0..10_000 {
        for i in 0..n {
            let mut s = b[i];
            let mut diag = 1.0;
            for k in rp[i]..rp[i + 1] {
                let j = ci[k];
                if j == i { diag = vl[k]; }
                else       { s -= vl[k] * x[j]; }
            }
            x[i] = s / diag;
        }
        let residual: f64 = (0..n)
            .map(|i| {
                let mut ax = 0.0;
                for k in rp[i]..rp[i + 1] { ax += vl[k] * x[ci[k]]; }
                (ax - b[i]).abs()
            })
            .sum();
        if residual < tol { break; }
    }
    x
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn block_partition_full_coverage() {
    for n in [1usize, 7, 16, 100] {
        for p in [1usize, 2, 3, 4, 7, 13] {
            let mut marks = vec![0u8; n];
            for rank in 0..p {
                let r = block_partition(n, p, rank);
                for i in r { marks[i] += 1; }
            }
            assert!(marks.iter().all(|&m| m == 1), "n={n} p={p}: not fully covered");
        }
    }
}

#[test]
fn partition_layout_metadata() {
    let layout = PartitionLayout::new(20, 5..12, vec![4, 12, 13]).unwrap();
    assert_eq!(layout.local_size(), 7);
    assert_eq!(layout.ghost_size(), 3);
    assert!(layout.owns_global(7));
    assert!(!layout.owns_global(12));
    assert!(!layout.owns_global(4));
}

#[test]
fn dist_csr_single_rank_structure() {
    let n = 8;
    let global = poisson_1d(n);
    let dist = DistCsrMatrix::from_global_csr_block_partition(&global, 1, 0).unwrap();
    assert_eq!(dist.layout().local_size(), n);
    assert_eq!(dist.layout().ghost_size(), 0);
    assert_eq!(dist.local_mat().nrows(), n);
}

#[test]
fn dist_csr_spmv_matches_global() {
    let n = 16;
    let global = poisson_1d(n);
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();

    // Reference: global spmv
    let mut y_ref = vec![0.0_f64; n];
    {
        let rp = global.row_ptr();
        let ci = global.col_idx();
        let vl = global.values();
        for i in 0..n {
            for k in rp[i]..rp[i + 1] { y_ref[i] += vl[k] * x[ci[k]]; }
        }
    }

    // Distributed: single partition with local halo (trivial exchange)
    let dist = DistCsrMatrix::from_global_csr_block_partition(&global, 1, 0).unwrap();
    let halo = LocalHaloExchange::new(x.clone());
    let mut y_dist = vec![0.0_f64; n];
    dist.spmv_with_halo(&x, &halo, &mut y_dist).unwrap();

    for i in 0..n {
        assert!((y_dist[i] - y_ref[i]).abs() < 1e-12, "row {i}: {} vs {}", y_dist[i], y_ref[i]);
    }
}

#[test]
fn dist_cg_converges_poisson_16() {
    let n = 16;
    let global = poisson_1d(n);
    let dist = DistCsrMatrix::from_global_csr_block_partition(&global, 1, 0).unwrap();
    let b = vec![1.0_f64; n];
    let halo = LocalHaloExchange::new(vec![1.0; n]); // ghost values; will be refreshed by spmv
    let mut x = vec![0.0_f64; n];
    let reduce = LocalReduce;

    let res = dist_cg(&dist, &halo, &reduce, &b, &mut x, &DistCgParams::default()).unwrap();
    assert!(res.converged, "dist_cg did not converge on n=16: {:?}", res);
}

#[test]
fn dist_cg_solution_matches_reference() {
    let n = 16;
    let global = poisson_1d(n);
    let b = vec![1.0_f64; n];

    // Reference solution via Gauss-Seidel
    let x_ref = dense_solve_diag_dom(&global, &b, 1e-10);

    // Distributed CG solution
    let dist = DistCsrMatrix::from_global_csr_block_partition(&global, 1, 0).unwrap();
    let halo = LocalHaloExchange::new(vec![0.0; n]);
    let mut x = vec![0.0_f64; n];
    let res = dist_cg(&dist, &halo, &LocalReduce, &b, &mut x, &DistCgParams::default()).unwrap();
    assert!(res.converged, "dist_cg did not converge: {:?}", res);

    let max_err: f64 = x.iter().zip(&x_ref).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(max_err < 1e-7, "max solution error: {max_err}");
}

#[test]
fn dist_cg_already_solved_zero_iterations() {
    let n = 8;
    let global = poisson_1d(n);
    let dist = DistCsrMatrix::from_global_csr_block_partition(&global, 1, 0).unwrap();

    let b = vec![1.0_f64; n];
    // Compute the actual solution first
    let halo = LocalHaloExchange::new(vec![0.0; n]);
    let mut x = vec![0.0_f64; n];
    let _ = dist_cg(&dist, &halo, &LocalReduce, &b, &mut x, &DistCgParams::default()).unwrap();

    // Now use x as the initial guess — should converge in 0 or 1 iterations
    let halo2 = LocalHaloExchange::new(x.clone());
    let params = DistCgParams { rtol: 1e-8, atol: 0.0, max_iter: 1000 };
    let res2 = dist_cg(&dist, &halo2, &LocalReduce, &b, &mut x, &params).unwrap();
    assert!(res2.converged, "should still converge: {:?}", res2);
    assert!(res2.iters <= 2, "expected few iterations, got {}", res2.iters);
}

#[test]
fn local_halo_exchange_returns_correct_values() {
    let global = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0];
    let halo = LocalHaloExchange::new(global.clone());
    let indices = vec![0usize, 2, 4];
    let mut out = vec![0.0_f64; 3];
    halo.exchange(&indices, &mut out).unwrap();
    assert_eq!(out, [10.0, 30.0, 50.0]);
}

#[test]
fn halo_plan_rejects_duplicate_rank() {
    let result = HaloPlan::new(vec![
        NeighborHaloPlan { neighbor_rank: 1, send_globals: vec![0], recv_globals: vec![5] },
        NeighborHaloPlan { neighbor_rank: 1, send_globals: vec![1], recv_globals: vec![6] },
    ]);
    assert!(result.is_err(), "duplicate rank should be rejected");
}

#[test]
fn halo_plan_unique_ranks_accepted() {
    let plan = HaloPlan::new(vec![
        NeighborHaloPlan { neighbor_rank: 0, send_globals: vec![3], recv_globals: vec![7] },
        NeighborHaloPlan { neighbor_rank: 2, send_globals: vec![4], recv_globals: vec![8] },
    ]).unwrap();
    assert_eq!(plan.total_send_len(), 2);
    assert_eq!(plan.total_recv_len(), 2);
}

#[test]
fn dist_cg_converges_larger_problem() {
    let n = 128;
    let global = poisson_1d(n);
    let dist = DistCsrMatrix::from_global_csr_block_partition(&global, 1, 0).unwrap();
    let b: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let halo = LocalHaloExchange::new(vec![0.0; n]);
    let mut x = vec![0.0_f64; n];
    let params = DistCgParams { rtol: 1e-8, atol: 0.0, max_iter: 2000 };
    let res = dist_cg(&dist, &halo, &LocalReduce, &b, &mut x, &params).unwrap();
    assert!(res.converged, "n=128 did not converge: {:?}", res);
    // Verify residual
    let rp = global.row_ptr();
    let ci = global.col_idx();
    let vl = global.values();
    let res_norm: f64 = (0..n)
        .map(|i| {
            let mut ax = 0.0;
            for k in rp[i]..rp[i+1] { ax += vl[k] * x[ci[k]]; }
            (ax - b[i]).powi(2)
        })
        .sum::<f64>()
        .sqrt();
    assert!(res_norm < 1e-6, "residual norm too large: {res_norm}");
}
