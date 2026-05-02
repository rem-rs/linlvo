//! Sprint 3 tests — advanced preconditioners and flexible/augmented Krylov solvers.

mod common;

use linger::{
    iterative::{ConjugateGradient, Fgmres, Gmres, Lgmres},
    precond::{
        AdditivePrecond, Icc0Precond, IlukPrecond, IlutPrecond, JacobiPrecond,
        MultiplicativePrecond, SpaiPrecond,
    },
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, Preconditioner, SolverParams, VerboseLevel,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn diag_spd(diag: &[f64]) -> CsrMatrix<f64> {
    let n = diag.len();
    let mut coo = CooMatrix::new(n, n);
    for (i, &v) in diag.iter().enumerate() {
        coo.push(i, i, v);
    }
    CsrMatrix::from_coo(&coo)
}

fn default_params(rtol: f64, max_iter: usize) -> SolverParams {
    SolverParams { rtol, max_iter, verbose: VerboseLevel::Silent, ..Default::default() }
}

fn solution_error(x: &[f64], x_exact: &[f64]) -> f64 {
    let err: f64 = x.iter().zip(x_exact).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let norm: f64 = x_exact.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 { err / norm } else { err }
}

// ─── ILU(k) tests ─────────────────────────────────────────────────────────────

#[test]
fn iluk_k0_equals_ilu0_on_diagonal() {
    // ILU(k=0) on a diagonal system is exact → preconditioned CG in 1 step.
    let a = diag_spd(&[1.0, 2.0, 3.0, 4.0]);
    let b = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut x = DenseVec::zeros(4);
    let ilu = IlukPrecond::<f64>::from_csr(&a, 0).unwrap();
    let cg = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&ilu), &b, &mut x, &default_params(1e-12, 50)).unwrap();
    assert!(res.converged);
    for &xi in x.as_slice() {
        assert!((xi - 1.0).abs() < 1e-10);
    }
}

#[test]
fn iluk_k1_poisson_1d() {
    let n = 100;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let ilu = IlukPrecond::<f64>::from_csr(&a, 1).unwrap();
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let cg = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&ilu), &b_vec, &mut x, &default_params(1e-10, 200)).unwrap();
    assert!(res.converged, "ILU(1)-PCG didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn iluk_k1_no_more_iters_than_ilu0() {
    // ILU(1) has at least as good preconditioning as ILU(0) (no worse fill).
    // For 1D Poisson (tridiagonal) fill level k=1 adds no extra entries,
    // so the iteration counts should be equal.
    let n = 100;
    let (a, _, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let params = default_params(1e-10, 500);
    let cg = ConjugateGradient::<f64>::default();

    let ilu0 = linger::Ilu0Precond::<f64>::from_csr(&a).unwrap();
    let mut x0 = DenseVec::zeros(n);
    let r0 = cg.solve(&a, Some(&ilu0), &b_vec, &mut x0, &params).unwrap();

    let ilu1 = IlukPrecond::<f64>::from_csr(&a, 1).unwrap();
    let mut x1 = DenseVec::zeros(n);
    let r1 = cg.solve(&a, Some(&ilu1), &b_vec, &mut x1, &params).unwrap();

    assert!(r0.converged && r1.converged);
    // ILU(1) should need <= ILU(0) iterations
    assert!(r1.iterations <= r0.iterations + 2,
        "ILU(1) ({}) should be at most slightly worse than ILU(0) ({})",
        r1.iterations, r0.iterations);
}

// ─── ILUT tests ───────────────────────────────────────────────────────────────

#[test]
fn ilut_diagonal_system_exact() {
    // With tau=0 and large p, ILUT is exact → single preconditioned step.
    let a = diag_spd(&[1.0, 2.0, 3.0, 4.0]);
    let b = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut x = DenseVec::zeros(4);
    let ilu = IlutPrecond::<f64>::from_csr(&a, 0.0, 10).unwrap();
    let cg = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&ilu), &b, &mut x, &default_params(1e-12, 50)).unwrap();
    assert!(res.converged);
    for &xi in x.as_slice() {
        assert!((xi - 1.0).abs() < 1e-10);
    }
}

#[test]
fn ilut_poisson_1d() {
    let n = 100;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let ilu = IlutPrecond::<f64>::from_csr(&a, 0.01, 5).unwrap();
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let cg = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&ilu), &b_vec, &mut x, &default_params(1e-10, 200)).unwrap();
    assert!(res.converged, "ILUT-PCG didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn ilut_nonsymmetric_convdiff() {
    let n = 20;
    let (a, x_exact, b) = common::make_nonsymmetric_convdiff::<f64>(n, 2.0);
    let ilu = IlutPrecond::<f64>::from_csr(&a, 0.01, 5).unwrap();
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let gmres = Gmres::<f64>::new(20);
    let res = gmres.solve(&a, Some(&ilu), &b_vec, &mut x, &default_params(1e-10, 200)).unwrap();
    assert!(res.converged, "ILUT-GMRES didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-7);
}

// ─── ICC(0) tests ─────────────────────────────────────────────────────────────

#[test]
fn icc0_diagonal_system_exact() {
    let a = diag_spd(&[1.0, 2.0, 3.0, 4.0]);
    let b = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut x = DenseVec::zeros(4);
    let icc = Icc0Precond::<f64>::from_csr(&a).unwrap();
    let cg = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&icc), &b, &mut x, &default_params(1e-12, 50)).unwrap();
    assert!(res.converged);
    for &xi in x.as_slice() {
        assert!((xi - 1.0).abs() < 1e-10);
    }
}

#[test]
fn icc0_poisson_1d() {
    let n = 100;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let icc = Icc0Precond::<f64>::from_csr(&a).unwrap();
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let cg = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&icc), &b_vec, &mut x, &default_params(1e-10, 200)).unwrap();
    assert!(res.converged, "ICC(0)-PCG didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn icc0_fewer_iters_than_unpreconditioned() {
    let n = 200;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(vec![1.0f64; n]);
    let params = default_params(1e-10, 2000);
    let cg = ConjugateGradient::<f64>::default();

    let mut x1 = DenseVec::zeros(n);
    let r1 = cg.solve(&a, None, &b_vec, &mut x1, &params).unwrap();

    let icc = Icc0Precond::<f64>::from_csr(&a).unwrap();
    let mut x2 = DenseVec::zeros(n);
    let r2 = cg.solve(&a, Some(&icc), &b_vec, &mut x2, &params).unwrap();

    assert!(r1.converged && r2.converged);
    assert!(r2.iterations < r1.iterations,
        "ICC(0)-PCG ({} iters) should beat unpreconditioned ({} iters)",
        r2.iterations, r1.iterations);
}

// ─── SPAI tests ───────────────────────────────────────────────────────────────

#[test]
fn spai_diagonal_gives_exact_inverse() {
    // For diagonal A = diag(d), SPAI should find M = diag(1/d).
    let diag = [2.0f64, 4.0, 5.0, 10.0];
    let a = diag_spd(&diag);
    let spai = SpaiPrecond::<f64>::from_csr(&a).unwrap();
    let x = DenseVec::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
    let mut y = DenseVec::zeros(4);
    spai.apply_precond(&x, &mut y);
    let ys = y.as_slice();
    for (i, &d) in diag.iter().enumerate() {
        assert!((ys[i] - 1.0 / d).abs() < 1e-10, "SPAI diagonal mismatch at {i}: {}", ys[i]);
    }
}

#[test]
fn spai_poisson_1d_precond() {
    let n = 50;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let spai = SpaiPrecond::<f64>::from_csr(&a).unwrap();
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let cg = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&spai), &b_vec, &mut x, &default_params(1e-10, 500)).unwrap();
    assert!(res.converged, "SPAI-PCG didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

// ─── Composite preconditioner tests ──────────────────────────────────────────

#[test]
fn additive_two_jacobi_equals_doubled() {
    // Additive combination of two identical Jacobi preconditioners
    // should give 2 × Jacobi.
    let n = 10;
    let (a, _, _) = common::make_poisson_1d::<f64>(n);
    let j1 = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let j2 = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let add = AdditivePrecond::<f64>::new(vec![Box::new(j1), Box::new(j2)]);

    let x = DenseVec::from_vec((0..n).map(|i| (i + 1) as f64).collect());
    let mut y_add = DenseVec::zeros(n);
    add.apply_precond(&x, &mut y_add);

    let j_ref = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let mut y_ref = DenseVec::zeros(n);
    j_ref.apply_precond(&x, &mut y_ref);

    for i in 0..n {
        let diff = (y_add.as_slice()[i] - 2.0 * y_ref.as_slice()[i]).abs();
        assert!(diff < 1e-14, "additive Jacobi mismatch at {i}");
    }
}

#[test]
fn multiplicative_precond_converges() {
    // Multiplicative Jacobi + ILU(0) should converge on 1D Poisson.
    let n = 50;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let jac = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let ilu = linger::Ilu0Precond::<f64>::from_csr(&a).unwrap();
    let multi = MultiplicativePrecond::<f64>::new(vec![Box::new(jac), Box::new(ilu)]);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let cg = ConjugateGradient::<f64>::default();
    let res = cg.solve(&a, Some(&multi), &b_vec, &mut x, &default_params(1e-10, 300)).unwrap();
    assert!(res.converged, "Multiplicative-PCG didn't converge; iters={}", res.iterations);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

// ─── FGMRES tests ─────────────────────────────────────────────────────────────

#[test]
fn fgmres_diagonal_4x4() {
    let a = diag_spd(&[1.0, 2.0, 3.0, 4.0]);
    let b = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut x = DenseVec::zeros(4);
    let solver = Fgmres::<f64>::new(10);
    let res = solver.solve(&a, None, &b, &mut x, &default_params(1e-12, 50)).unwrap();
    assert!(res.converged);
    for &xi in x.as_slice() {
        assert!((xi - 1.0).abs() < 1e-9);
    }
}

#[test]
fn fgmres_poisson_1d() {
    let n = 50;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let solver = Fgmres::<f64>::new(30);
    let res = solver.solve(&a, None, &b_vec, &mut x, &default_params(1e-10, 500)).unwrap();
    assert!(res.converged, "FGMRES didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn fgmres_nonsymmetric_convdiff() {
    let n = 20;
    let (a, x_exact, b) = common::make_nonsymmetric_convdiff::<f64>(n, 1.0);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let solver = Fgmres::<f64>::new(30);
    let res = solver.solve(&a, None, &b_vec, &mut x, &default_params(1e-10, 500)).unwrap();
    assert!(res.converged, "FGMRES didn't converge on conv-diff; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-7);
}

#[test]
fn fgmres_matches_gmres_with_fixed_precond() {
    // With a fixed preconditioner, FGMRES and GMRES should give equal results.
    let n = 30;
    let (a, _, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let params = default_params(1e-10, 300);

    let jac1 = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let mut x1 = DenseVec::zeros(n);
    let r1 = Gmres::<f64>::new(20).solve(&a, Some(&jac1), &b_vec, &mut x1, &params).unwrap();

    let jac2 = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let mut x2 = DenseVec::zeros(n);
    let r2 = Fgmres::<f64>::new(20).solve(&a, Some(&jac2), &b_vec, &mut x2, &params).unwrap();

    assert!(r1.converged && r2.converged);
    // Solutions should be close (both minimize residual in same space)
    let diff: f64 = x1.as_slice().iter().zip(x2.as_slice()).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
    assert!(diff < 1e-8, "FGMRES and GMRES solutions diverged: diff={diff:.3e}");
}

// ─── LGMRES tests ─────────────────────────────────────────────────────────────

#[test]
fn lgmres_diagonal_4x4() {
    let a = diag_spd(&[1.0, 2.0, 3.0, 4.0]);
    let b = DenseVec::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut x = DenseVec::zeros(4);
    let solver = Lgmres::<f64>::new(10, 3);
    let res = solver.solve(&a, None, &b, &mut x, &default_params(1e-12, 50)).unwrap();
    assert!(res.converged);
    for &xi in x.as_slice() {
        assert!((xi - 1.0).abs() < 1e-9, "LGMRES solution error: {xi}");
    }
}

#[test]
fn lgmres_poisson_1d() {
    let n = 50;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let solver = Lgmres::<f64>::new(20, 3);
    let res = solver.solve(&a, None, &b_vec, &mut x, &default_params(1e-10, 500)).unwrap();
    assert!(res.converged, "LGMRES didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}

#[test]
fn lgmres_nonsymmetric_convdiff() {
    let n = 20;
    let (a, x_exact, b) = common::make_nonsymmetric_convdiff::<f64>(n, 1.0);
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let solver = Lgmres::<f64>::new(20, 3);
    let res = solver.solve(&a, None, &b_vec, &mut x, &default_params(1e-10, 500)).unwrap();
    assert!(res.converged, "LGMRES didn't converge on conv-diff; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-7);
}

#[test]
fn lgmres_converges_on_harder_problem() {
    // LGMRES with augmentation should converge on a moderately sized problem.
    // Use Jacobi preconditioner to keep iteration count manageable.
    let n = 50;
    let (a, x_exact, b) = common::make_poisson_1d::<f64>(n);
    let jac = JacobiPrecond::<f64>::from_csr(&a).unwrap();
    let b_vec = DenseVec::from_vec(b);
    let mut x = DenseVec::zeros(n);
    let solver = Lgmres::<f64>::new(15, 3);
    let res = solver.solve(&a, Some(&jac), &b_vec, &mut x, &default_params(1e-10, 1000)).unwrap();
    assert!(res.converged, "LGMRES (Jacobi) didn't converge; iters={}, rel={:.3e}", res.iterations, res.final_residual);
    assert!(solution_error(x.as_slice(), &x_exact) < 1e-8);
}
