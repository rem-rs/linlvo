mod common;

use linger::{
    iterative::{CgWorkspace, ConjugateGradient, Gmres, GmresWorkspace},
    DenseVec,
};

#[test]
fn cg_fixed_iters_runs_exact_count_with_workspace() {
    let n = 64;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let b = DenseVec::from_vec(vec![1.0; n]);
    let cg = ConjugateGradient::<f64>::default();
    let mut workspace = CgWorkspace::new(n);

    let mut x4 = DenseVec::zeros(n);
    let res4 = cg
        .solve_fixed_iters_with_workspace(&a, None, &b, &mut x4, 4, &mut workspace)
        .expect("fixed-iteration CG should not break down on Poisson");

    let mut x8 = DenseVec::zeros(n);
    let res8 = cg
        .solve_fixed_iters_with_workspace(&a, None, &b, &mut x8, 8, &mut workspace)
        .expect("fixed-iteration CG should not break down on Poisson");

    assert_eq!(res4.iterations, 4);
    assert_eq!(res8.iterations, 8);
    assert_eq!(res4.residual_history.len(), 4);
    assert_eq!(res8.residual_history.len(), 8);
    assert!(res8.final_residual < res4.final_residual,
        "CG fixed-iteration residual should decrease: 4 iters {:.3e}, 8 iters {:.3e}",
        res4.final_residual, res8.final_residual);
}

#[test]
fn gmres_fixed_iters_runs_exact_count_with_workspace() {
    let n = 40;
    let (a, _, b_exact) = common::make_nonsymmetric_convdiff::<f64>(n, 5.0);
    let b = DenseVec::from_vec(b_exact);
    let gmres = Gmres::<f64>::new(5);
    let mut workspace = GmresWorkspace::new(n, 5);

    let mut x3 = DenseVec::zeros(n);
    let res3 = gmres
        .solve_fixed_iters_with_workspace(&a, None, &b, &mut x3, 3, &mut workspace)
        .expect("fixed-iteration GMRES should not break down on convection-diffusion");

    let mut x6 = DenseVec::zeros(n);
    let res6 = gmres
        .solve_fixed_iters_with_workspace(&a, None, &b, &mut x6, 6, &mut workspace)
        .expect("fixed-iteration GMRES should not break down across restart boundary");

    assert_eq!(res3.iterations, 3);
    assert_eq!(res6.iterations, 6);
    assert_eq!(res3.residual_history.len(), 3);
    assert_eq!(res6.residual_history.len(), 6);
    assert!(res6.final_residual < res3.final_residual,
        "GMRES fixed-iteration residual should decrease: 3 iters {:.3e}, 6 iters {:.3e}",
        res3.final_residual, res6.final_residual);
}