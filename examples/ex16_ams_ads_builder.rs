//! ex16 - SolverBuilder AMS/ADS presets with diagnostics.

use std::sync::Arc;

use linger::{
    builder::{BuilderPrecondReport, SolverBuilder},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, LinearOperator,
};

fn make_chain_graph(n_nodes: usize, delta: f64) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
    let n_edges = n_nodes - 1;
    let mut cg = CooMatrix::new(n_edges, n_nodes);
    for edge in 0..n_edges {
        cg.push(edge, edge, -1.0);
        cg.push(edge, edge + 1, 1.0);
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
    (g, CsrMatrix::from_coo(&ca))
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
    let face = |i: usize, j: usize| i * (ny - 1) + j;

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

    (g, c, CsrMatrix::from_coo(&ca))
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
    if den < 1e-300 { num } else { num / den }
}

fn main() {
    println!("ex16: SolverBuilder AMS/ADS presets");

    let (g_ams, a_ams) = make_chain_graph(31, 1e-3);
    let n_ams = a_ams.nrows();
    let x_exact: Vec<f64> = (1..=n_ams)
        .map(|k| (std::f64::consts::PI * k as f64 / (n_ams + 1) as f64).sin())
        .collect();
    let mut rhs_ams = vec![0.0_f64; n_ams];
    a_ams.spmv(&x_exact, &mut rhs_ams);
    let b_ams = DenseVec::from_vec(rhs_ams);

    let (x_ams, report_ams) = SolverBuilder::new()
        .hpc_ams(Arc::new(g_ams))
        .solve_with_report(&a_ams, &b_ams)
        .unwrap();
    let rel_ams = relative_residual(&a_ams, &x_ams, &b_ams);

    match report_ams.precond {
        BuilderPrecondReport::Ams(profile) => {
            println!(
                "  AMS: rel_res={:.3e} edges={} nodes={} a_node_nnz={}",
                rel_ams,
                profile.n_edges,
                profile.n_nodes,
                profile.a_node_nnz
            );
        }
        _ => panic!("expected AMS report"),
    }
    let krylov_ams = report_ams.krylov.expect("expected AMS Krylov report");
    println!(
        "  AMS Krylov: converged={} iters={} final_res={:.3e}",
        krylov_ams.converged,
        krylov_ams.iterations,
        krylov_ams.final_residual
    );
    assert!(krylov_ams.converged);
    assert!(rel_ams < 1e-7);

    let (g_ads, c_ads, a_ads) = make_rect_complex(4, 4, 1e-3);
    let b_ads = DenseVec::from_vec(vec![1.0_f64; a_ads.nrows()]);
    let (x_ads, report_ads) = SolverBuilder::new()
        .hpc_ads(Arc::new(c_ads), Arc::new(g_ads))
        .max_iter(600)
        .solve_with_report(&a_ads, &b_ads)
        .unwrap();
    let rel_ads = relative_residual(&a_ads, &x_ads, &b_ads);
    let x_ads_slice = x_ads.as_slice();

    match report_ads.precond {
        BuilderPrecondReport::Ads(profile) => {
            println!(
                "  ADS: rel_res={:.3e} faces={} edges={} nodes={}",
                rel_ads,
                profile.n_faces,
                profile.n_edges,
                profile.n_nodes
            );
        }
        _ => panic!("expected ADS report"),
    }
    let krylov_ads = report_ads.krylov.expect("expected ADS Krylov report");
    println!(
        "  ADS Krylov: converged={} iters={} final_res={:.3e}",
        krylov_ads.converged,
        krylov_ads.iterations,
        krylov_ads.final_residual
    );
    assert!(krylov_ads.converged);
    assert!(x_ads_slice.iter().all(|v| v.is_finite()));
    assert!(x_ads_slice.iter().any(|v| v.abs() > 1e-14));

    println!("  OK");
}