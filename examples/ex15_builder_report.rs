//! ex15 - SolverBuilder solve_with_report diagnostics.

use linger::{
    builder::{BuilderPrecondReport, SolveMethod, SolverBuilder},
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, LinearOperator,
};

fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0 {
            coo.push(i, i - 1, -1.0);
        }
        if i + 1 < n {
            coo.push(i, i + 1, -1.0);
        }
    }
    CsrMatrix::from_coo(&coo)
}

fn nonsymmetric_tridiag(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 4.0);
        if i > 0 {
            coo.push(i, i - 1, -1.0);
        }
        if i + 1 < n {
            coo.push(i, i + 1, -2.0);
        }
    }
    CsrMatrix::from_coo(&coo)
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

fn precond_name(p: &BuilderPrecondReport) -> &'static str {
    match p {
        BuilderPrecondReport::None => "none",
        BuilderPrecondReport::Jacobi => "jacobi",
        BuilderPrecondReport::Ilu0 => "ilu0",
        BuilderPrecondReport::Icc0 => "icc0",
        BuilderPrecondReport::DirectLu { .. } => "direct-lu",
        BuilderPrecondReport::Ams(_) => "ams",
        BuilderPrecondReport::Ads(_) => "ads",
        BuilderPrecondReport::FieldSplit { .. } => "fieldsplit",
    }
}

fn main() {
    println!("ex15: SolverBuilder solve_with_report");

    let a_spd = laplacian_1d(32);
    let b_spd = DenseVec::from_vec(vec![1.0_f64; a_spd.nrows()]);
    let (x_spd, report_spd) = SolverBuilder::new()
        .method(SolveMethod::Cg)
        .precond(linger::builder::PrecondChoice::Jacobi)
        .rtol(1e-10)
        .max_iter(300)
        .solve_with_report(&a_spd, &b_spd)
        .unwrap();
    let rel_spd = relative_residual(&a_spd, &x_spd, &b_spd);
    println!(
        "  SPD solve: precond={} rel_res={:.3e}",
        precond_name(&report_spd.precond),
        rel_spd
    );

    let a_ns = nonsymmetric_tridiag(24);
    let b_ns = DenseVec::from_vec((1..=a_ns.nrows()).map(|i| i as f64).collect());
    let (x_ns, report_ns) = SolverBuilder::new()
        .method(SolveMethod::Gmres { restart: 20 })
        .precond(linger::builder::PrecondChoice::Ilu0)
        .rtol(1e-10)
        .max_iter(120)
        .solve_with_report(&a_ns, &b_ns)
        .unwrap();
    let rel_ns = relative_residual(&a_ns, &x_ns, &b_ns);
    println!(
        "  nonsym solve: precond={} rel_res={:.3e}",
        precond_name(&report_ns.precond),
        rel_ns
    );

    if let Some(k) = report_ns.krylov {
        println!(
            "  GMRES report: converged={} iters={} final_res={:.3e}",
            k.converged,
            k.iterations,
            k.final_residual
        );
        assert!(k.converged);
    }

    assert!(rel_spd < 1e-9);
    assert!(rel_ns < 1e-9);
    println!("  OK");
}