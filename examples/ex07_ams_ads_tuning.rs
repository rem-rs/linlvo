//! ex07 — AMS/ADS tuning sweep (CSV output)
//!
//! Runs small parameter sweeps for auxiliary-space preconditioners and prints
//! one CSV row per run.
//!
//! Usage:
//!   cargo run --example ex07_ams_ads_tuning -- --mode ams
//!   cargo run --example ex07_ams_ads_tuning -- --mode ads
//!   cargo run --example ex07_ams_ads_tuning -- --mode both

use std::sync::Arc;
use std::time::Instant;

use linger::{
    amg::AmgConfig,
    builder::{BuilderPrecondReport, PrecondChoice, SolveMethod, SolverBuilder},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Ams,
    Ads,
    Both,
}

fn parse_mode() -> Mode {
    let mut mode = Mode::Both;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        if arg == "--mode" {
            if let Some(v) = it.next() {
                mode = match v.as_str() {
                    "ams" => Mode::Ams,
                    "ads" => Mode::Ads,
                    _ => Mode::Both,
                };
            }
        }
    }
    mode
}

fn make_chain_graph(n_nodes: usize, delta: f64) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
    let n_edges = n_nodes - 1;
    let mut cg = CooMatrix::new(n_edges, n_nodes);
    for e in 0..n_edges {
        cg.push(e, e, -1.0);
        cg.push(e, e + 1, 1.0);
    }
    let g = CsrMatrix::from_coo(&cg);

    let g_t = g.transpose_csr();
    let gg_t = g.matmat(&g_t);
    let mut ca = CooMatrix::new(n_edges, n_edges);
    for (i, j, v) in gg_t.triplets() {
        ca.push(i, j, v);
    }
    for i in 0..n_edges {
        ca.push(i, i, delta);
    }
    let a = CsrMatrix::from_coo(&ca);
    (g, a)
}

fn make_rect_complex(nx: usize, ny: usize, delta: f64) -> (CsrMatrix<f64>, CsrMatrix<f64>, CsrMatrix<f64>) {
    let n_nodes = nx * ny;
    let n_h = nx * (ny - 1);
    let n_v = (nx - 1) * ny;
    let n_edges = n_h + n_v;
    let n_faces = (nx - 1) * (ny - 1);

    let node = |i: usize, j: usize| i * ny + j;
    let h_edge = |i: usize, j: usize| i * (ny - 1) + j;
    let v_edge = |i: usize, j: usize| n_h + i * ny + j;

    let mut cg = CooMatrix::new(n_edges, n_nodes);
    for i in 0..nx {
        for j in 0..(ny - 1) {
            cg.push(h_edge(i, j), node(i, j), -1.0);
            cg.push(h_edge(i, j), node(i, j + 1), 1.0);
        }
    }
    for i in 0..(nx - 1) {
        for j in 0..ny {
            cg.push(v_edge(i, j), node(i, j), -1.0);
            cg.push(v_edge(i, j), node(i + 1, j), 1.0);
        }
    }
    let g = CsrMatrix::from_coo(&cg);

    let mut cc = CooMatrix::new(n_faces, n_edges);
    let face = |i: usize, j: usize| i * (ny - 1) + j;
    for i in 0..(nx - 1) {
        for j in 0..(ny - 1) {
            let f = face(i, j);
            cc.push(f, h_edge(i, j), 1.0);
            cc.push(f, v_edge(i, j + 1), 1.0);
            cc.push(f, h_edge(i + 1, j), -1.0);
            cc.push(f, v_edge(i, j), -1.0);
        }
    }
    let c = CsrMatrix::from_coo(&cc);

    let c_t = c.transpose_csr();
    let cc_t = c.matmat(&c_t);
    let mut ca = CooMatrix::new(n_faces, n_faces);
    for (i, j, v) in cc_t.triplets() {
        ca.push(i, j, v);
    }
    for i in 0..n_faces {
        ca.push(i, i, delta);
    }
    let a = CsrMatrix::from_coo(&ca);
    (g, c, a)
}

fn run_ams_sweep() {
    let n_nodes = 121usize;
    let (g, a) = make_chain_graph(n_nodes, 1e-3);
    let g = Arc::new(g);

    let n = a.nrows();
    let x_exact: Vec<f64> = (1..=n)
        .map(|k| (std::f64::consts::PI * k as f64 / (n + 1) as f64).sin())
        .collect();
    let mut b_raw = vec![0.0f64; n];
    a.spmv(&x_exact, &mut b_raw);
    let b = DenseVec::from_vec(b_raw);

    let thetas = [0.15_f64, 0.25, 0.40];
    let coarse_thresholds = [16_usize, 32, 64];
    let restarts = [30_usize, 50];

    println!("family,theta,coarse_threshold,restart,max_levels,converged,iterations,final_residual,solve_ms,aux_n_levels,aux_op_cx,aux_grid_cx");

    for &theta in &thetas {
        for &ct in &coarse_thresholds {
            for &restart in &restarts {
                let cfg = linger::AmsConfig {
                    smoother_omega: 0.667,
                    node_solver: linger::AuxSpaceSolver::Amg(AmgConfig {
                        theta,
                        coarse_threshold: ct,
                        max_levels: 30,
                        ..AmgConfig::default()
                    }),
                };

                let builder = SolverBuilder::new()
                    .method(SolveMethod::Gmres { restart })
                    .precond(PrecondChoice::Ams { g: g.clone(), config: cfg })
                    .rtol(1e-8)
                    .max_iter(800);

                let t0 = Instant::now();
                let res = builder.solve_with_report(&a, &b);
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;

                match res {
                    Ok((_x, report)) => {
                        let (mut n_levels, mut op_cx, mut grid_cx) = (0usize, 0.0f64, 0.0f64);
                        if let BuilderPrecondReport::Ams(p) = report.precond {
                            if let linger::AuxSolverProfile::Amg(amg) = p.node_solver {
                                n_levels = amg.n_levels;
                                op_cx = amg.operator_complexity;
                                grid_cx = amg.grid_complexity;
                            }
                        }
                        if let Some(k) = report.krylov {
                            println!(
                                "ams,{theta:.2},{ct},{restart},30,{},{},{:.3e},{:.3},{},{:.4},{:.4}",
                                k.converged,
                                k.iterations,
                                k.final_residual,
                                elapsed_ms,
                                n_levels,
                                op_cx,
                                grid_cx,
                            );
                        }
                    }
                    Err(e) => {
                        println!(
                            "ams,{theta:.2},{ct},{restart},30,false,-1,NaN,{elapsed_ms:.3},0,0.0,0.0 # error={e}"
                        );
                    }
                }
            }
        }
    }
}

fn run_ads_sweep() {
    let (g, c, a) = make_rect_complex(10, 10, 1e-3);
    let g = Arc::new(g);
    let c = Arc::new(c);
    let b = DenseVec::from_vec(vec![1.0_f64; a.nrows()]);

    let thetas = [0.15_f64, 0.25, 0.40];
    let coarse_thresholds = [16_usize, 32, 64];
    let restarts = [30_usize, 50];

    println!("family,theta,coarse_threshold,restart,max_levels,converged,iterations,final_residual,solve_ms,edge_n_levels,edge_op_cx,edge_grid_cx,node_n_levels,node_op_cx,node_grid_cx");

    for &theta in &thetas {
        for &ct in &coarse_thresholds {
            for &restart in &restarts {
                let amg = AmgConfig {
                    theta,
                    coarse_threshold: ct,
                    max_levels: 30,
                    ..AmgConfig::default()
                };
                let cfg = linger::AdsConfig {
                    smoother_omega: 0.667,
                    edge_solver: linger::AuxSpaceSolver::Amg(amg.clone()),
                    node_solver: linger::AuxSpaceSolver::Amg(amg),
                };

                let builder = SolverBuilder::new()
                    .method(SolveMethod::Gmres { restart })
                    .precond(PrecondChoice::Ads {
                        c: c.clone(),
                        g: g.clone(),
                        config: cfg,
                    })
                    .rtol(1e-8)
                    .max_iter(1200);

                let t0 = Instant::now();
                let res = builder.solve_with_report(&a, &b);
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;

                match res {
                    Ok((_x, report)) => {
                        let (mut en, mut eop, mut egc) = (0usize, 0.0f64, 0.0f64);
                        let (mut nn, mut nop, mut ngc) = (0usize, 0.0f64, 0.0f64);
                        if let BuilderPrecondReport::Ads(p) = report.precond {
                            if let linger::AuxSolverProfile::Amg(amg) = p.edge_solver {
                                en = amg.n_levels;
                                eop = amg.operator_complexity;
                                egc = amg.grid_complexity;
                            }
                            if let linger::AuxSolverProfile::Amg(amg) = p.node_solver {
                                nn = amg.n_levels;
                                nop = amg.operator_complexity;
                                ngc = amg.grid_complexity;
                            }
                        }
                        if let Some(k) = report.krylov {
                            println!(
                                "ads,{theta:.2},{ct},{restart},30,{},{},{:.3e},{:.3},{},{:.4},{:.4},{},{:.4},{:.4}",
                                k.converged,
                                k.iterations,
                                k.final_residual,
                                elapsed_ms,
                                en,
                                eop,
                                egc,
                                nn,
                                nop,
                                ngc,
                            );
                        }
                    }
                    Err(e) => {
                        println!(
                            "ads,{theta:.2},{ct},{restart},30,false,-1,NaN,{elapsed_ms:.3},0,0.0,0.0,0,0.0,0.0 # error={e}"
                        );
                    }
                }
            }
        }
    }
}

fn main() {
    match parse_mode() {
        Mode::Ams => run_ams_sweep(),
        Mode::Ads => run_ads_sweep(),
        Mode::Both => {
            run_ams_sweep();
            run_ads_sweep();
        }
    }
}
