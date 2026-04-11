//! ex12 - AMS and ADS auxiliary-space preconditioners.

use linger::{
    iterative::Gmres,
    precond::{AdsConfig, AdsPrecond, AmsConfig, AmsPrecond},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, LinearOperator, Preconditioner, SolverParams, VerboseLevel,
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
    if den == 0.0 { num } else { num / den }
}

fn main() {
    println!("ex12: AMS and ADS preconditioners");

    let (g_ams, a_ams) = make_chain_graph(21, 1e-3);
    let ams = AmsPrecond::new(&a_ams, &g_ams, AmsConfig::default()).unwrap();
    let ams_probe = DenseVec::from_vec(vec![1.0_f64; a_ams.nrows()]);
    let mut ams_apply = DenseVec::zeros(a_ams.nrows());
    ams.apply_precond(&ams_probe, &mut ams_apply);
    println!(
        "  AMS apply: n_edges={} first entries={:?}",
        a_ams.nrows(),
        &ams_apply.as_slice()[0..4.min(a_ams.nrows())]
    );
    assert!(ams_apply.as_slice().iter().all(|v| v.is_finite()));

    let ams_exact: Vec<f64> = (1..=a_ams.nrows())
        .map(|k| (std::f64::consts::PI * k as f64 / (a_ams.nrows() + 1) as f64).sin())
        .collect();
    let mut ams_rhs = vec![0.0_f64; a_ams.nrows()];
    a_ams.spmv(&ams_exact, &mut ams_rhs);
    let ams_b = DenseVec::from_vec(ams_rhs);
    let mut ams_x = DenseVec::zeros(a_ams.nrows());
    let ams_result = Gmres::new(30)
        .solve(
            &a_ams,
            Some(&ams),
            &ams_b,
            &mut ams_x,
            &SolverParams {
                rtol: 1e-8,
                max_iter: 200,
                verbose: VerboseLevel::Silent,
                ..Default::default()
            },
        )
        .unwrap();
    let ams_rel = relative_residual(&a_ams, &ams_x, &ams_b);
    println!("  GMRES + AMS: iters={} rel_res={:.3e}", ams_result.iterations, ams_rel);
    assert!(ams_result.converged);
    assert!(ams_rel < 1e-7);

    let (g_ads, c_ads, a_ads) = make_rect_complex(4, 4, 1e-3);
    let ads = AdsPrecond::new(&a_ads, &c_ads, &g_ads, AdsConfig::default()).unwrap();
    let ads_probe = DenseVec::from_vec(vec![1.0_f64; a_ads.nrows()]);
    let mut ads_apply = DenseVec::zeros(a_ads.nrows());
    ads.apply_precond(&ads_probe, &mut ads_apply);
    println!(
        "  ADS apply: n_faces={} first entries={:?}",
        a_ads.nrows(),
        &ads_apply.as_slice()[0..4.min(a_ads.nrows())]
    );
    assert!(ads_apply.as_slice().iter().all(|v| v.is_finite()));

    let ads_b = DenseVec::from_vec(vec![1.0_f64; a_ads.nrows()]);
    let mut ads_x = DenseVec::zeros(a_ads.nrows());
    let ads_result = Gmres::new(30)
        .solve(
            &a_ads,
            Some(&ads),
            &ads_b,
            &mut ads_x,
            &SolverParams {
                rtol: 1e-8,
                max_iter: 300,
                verbose: VerboseLevel::Silent,
                ..Default::default()
            },
        )
        .unwrap();
    let ads_rel = relative_residual(&a_ads, &ads_x, &ads_b);
    println!("  GMRES + ADS: iters={} rel_res={:.3e}", ads_result.iterations, ads_rel);
    assert!(ads_result.converged);
    assert!(ads_rel < 1e-7);

    let cg = c_ads.matmat(&g_ads);
    let complex_zero = cg.triplets().all(|(_, _, v)| v.abs() < 1e-12);
    println!("  de Rham identity C*G=0: {}", complex_zero);
    assert!(complex_zero);

    println!("  OK");
}