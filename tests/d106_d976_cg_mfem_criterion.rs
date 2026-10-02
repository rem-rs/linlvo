//! D976: linlvo `ConjugateGradient` must stop with **MFEM `CGSolver` semantics**.
//!
//! MFEM `CGSolver::Mult` (`linalg/solvers.cpp:869-1053`, MFEM 4.10) computes the
//! convergence threshold **once** from the initial residual energy
//! `nom0 = (B r₀, r₀)` (or `‖r₀‖²` without a preconditioner):
//!
//! ```text
//! r0 = max(nom0 * rel_tol * rel_tol, abs_tol * abs_tol);   // :915
//! ... betanom = Dot(r, z);                                  // :957  (z = B r)
//! if (Monitor(i,...) || betanom <= r0) { converged = 1; }   // :977
//! ```
//!
//! i.e. the *preconditioned energy ratio* must reach **rel_tol²**, not rel_tol.
//! The old linlvo test `(B r, r)/(B r₀, r₀) < rtol` stopped six orders of
//! magnitude (in the preconditioned norm √ratio) too early — the D976 symptom
//! (rtol 1e-12 requested, true residual ~1e-7..1e-6 at stop).
//!
//! C++ oracle (MFEM 4.10 serial, `$HOME/mfem410_ser`, probe
//! `tmp/d106kernel/d976_cg_probe.cpp`): the identical system below — 5-point
//! 2-D Poisson, 30×30 interior grid (N=900), `GSSmoother(SYMMETRIC,1)`
//! preconditioner (linlvo's `GaussSeidelSmoother` is the 1:1 match), b = A·1,
//! x₀ = 0, `SetRelTol(1e-12)`, max 500 — reports
//!
//! ```text
//!    Iteration :   0  (B r, r) = 57292.6 ...
//!    Iteration :  43  (B r, r) = 2.00721e-20
//!    Average reduction factor = 0.519558
//!    TRUE rel residual at stop = 6.817629e-13
//! ```
//!
//! final energy ratio 2.0e-20 / 5.729e4 ≈ 3.5e-25 ≤ rtol² — MFEM really does
//! iterate to the squared tolerance.

use linlvo::core::solver::{KrylovSolver, SolverParams, VerboseLevel};
use linlvo::core::vector::DenseVec;
use linlvo::{LinearOperator, Preconditioner};
use linlvo::iterative::ConjugateGradient;
use linlvo::precond::GaussSeidelSmoother;
use linlvo::sparse::{CooMatrix, CsrMatrix};

/// 5-point 2-D Poisson, `m × m` interior grid, scaled by 1/h².
fn poisson_2d(m: usize) -> CsrMatrix<f64> {
    let n = m * m;
    let h2 = 1.0 / ((m as f64 + 1.0) * (m as f64 + 1.0));
    let mut coo: CooMatrix<f64> = CooMatrix::new(n, n);
    let id = |i: usize, j: usize| j * m + i;
    for j in 0..m {
        for i in 0..m {
            let k = id(i, j);
            coo.push(k, k, 4.0 / h2);
            if i > 0 { coo.push(k, id(i - 1, j), -1.0 / h2); }
            if i < m - 1 { coo.push(k, id(i + 1, j), -1.0 / h2); }
            if j > 0 { coo.push(k, id(i, j - 1), -1.0 / h2); }
            if j < m - 1 { coo.push(k, id(i, j + 1), -1.0 / h2); }
        }
    }
    CsrMatrix::from_coo(&coo)
}

#[test]
fn d976_rtol_1e_12_drives_true_residual_to_rtol_scale() {
    let m = 30;
    let a = poisson_2d(m);
    let n = m * m;

    // b = A · 1  (exact solution = all ones), x0 = 0.
    let ones = vec![1.0_f64; n];
    let mut b = vec![0.0_f64; n];
    a.spmv(&ones, &mut b);
    let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();

    let gs = GaussSeidelSmoother::from_csr(&a).unwrap();
    let mut x = DenseVec::from_vec(vec![0.0_f64; n]);
    let params = SolverParams {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 500,
        verbose: VerboseLevel::Silent,
        ..SolverParams::default()
    };
    let res = ConjugateGradient::<f64>::default()
        .solve(
            &a,
            Some(&gs as &dyn Preconditioner<Vector = DenseVec<f64>>),
            &DenseVec::from_vec(b.clone()),
            &mut x,
            &params,
        )
        .expect("CG must converge within 500 iterations");

    // True residual ‖b − A x‖ / ‖b‖ measured from scratch.
    let mut ax = DenseVec::zeros(n);
    LinearOperator::apply(&a, &x, &mut ax);
    let r2: f64 = b
        .iter()
        .zip(ax.as_slice())
        .map(|(&bi, &ai)| (bi - ai) * (bi - ai))
        .sum();
    let rel = r2.sqrt() / b_norm;

    // MFEM oracle: 43 iterations, true rel residual 6.8e-13 at stop.
    assert!(
        rel <= 1e-11,
        "D976: CG stopped with true rel residual {rel:.3e} after {} iterations \
         (rtol=1e-12 requested; MFEM semantics would drive it to ~1e-12)",
        res.iterations
    );
    assert!(
        (40..=46).contains(&res.iterations),
        "D976: iterations {} outside the MFEM oracle band [40,46] (C++: 43)",
        res.iterations
    );
}

#[test]
fn d976_energy_ratio_at_stop_le_rtol_squared() {
    let m = 24;
    let a = poisson_2d(m);
    let n = m * m;
    let ones = vec![1.0_f64; n];
    let mut b = vec![0.0_f64; n];
    a.spmv(&ones, &mut b);

    let gs = GaussSeidelSmoother::from_csr(&a).unwrap();
    let params = SolverParams {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 500,
        verbose: VerboseLevel::Iterations,
        ..SolverParams::default()
    };
    // Capture the printed energy trajectory by replaying the CG recurrence
    // through the solver: VerboseLevel::Iterations prints per-iteration (B r, r)
    // — asserted indirectly below via the residual history instead: with the
    // MFEM criterion the *last* recorded true residual must be ≤ ~rtol, i.e.
    // the preconditioned-energy ratio reached rtol².
    let x0 = vec![0.0_f64; n];
    let res = ConjugateGradient::<f64>::default()
        .solve(
            &a,
            Some(&gs as &dyn Preconditioner<Vector = DenseVec<f64>>),
            &DenseVec::from_vec(b.clone()),
            &mut DenseVec::from_vec(x0.clone()),
            &params,
        )
        .expect("CG must converge");

    let last = *res.residual_history.last().expect("non-empty history");
    assert!(
        last <= params.rtol * 10.0,
        "D976: last history residual {last:.3e} not at rtol={} scale — \
         the solver still stops on the un-squared energy ratio",
        params.rtol
    );
}
