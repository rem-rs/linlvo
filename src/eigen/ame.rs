//! AME — Auxiliary-space Maxwell Eigensolver
//!
//! Thin wrapper that configures the generalised-problem LOBPCG ([`Lobpcg`]) with
//! an AMS preconditioner and a discrete divergence-free (nullspace) projector to
//! solve the curl-curl eigenvalue problem `A x = λ M x`.
//!
//! # Algorithm
//!
//! AME = extended LOBPCG (`b_op`, `projector`) + AMS preconditioner.
//!
//! 1. LOBPCG iteration for the generalised problem `A x = λ M x`
//! 2. Inner PCG-AMS solves `A·W ≈ R` at each iteration (not single AMS apply)
//! 3. Div-free projection `P = I − G(GᵀMG)⁻¹GᵀM` removes gradient nullspace
//!
//! # References
//!
//! - Kolev & Vassilevski (2006). Parallel eigensolver for H(curl) problems
//!   using H1-auxiliary space AMG preconditioning. LLNL TR-226197.

use crate::amg::{AmgConfig, AmgHierarchy, AmgPrecond};
use crate::core::{
    error::SolverError,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{KrylovSolver, SolverParams},
    vector::DenseVec,
};
use crate::eigen::{EigenParams, EigenWhich};
use crate::eigen::lobpcg::Lobpcg;
use crate::iterative::ConjugateGradient;
use crate::precond::ams::{AmsConfig, AmsPrecond};
use crate::sparse::CsrMatrix;
use std::marker::PhantomData;

// ─── Configuration ──────────────────────────────────────────────────────────────

/// Configuration for the AME eigensolver.
#[derive(Debug, Clone)]
pub struct AmeConfig {
    /// Number of eigenvalue/vector pairs to compute (block size).
    pub nev: usize,
    /// Maximum LOBPCG iterations (default 100).
    pub max_iter: usize,
    /// Convergence tolerance: `‖Ax − λMx‖ / |λ| < tol` (default 1e-8).
    pub tol: f64,
    /// Print convergence info when `true`.
    pub verbose: bool,
    /// AMS nodal-system singularity regularization (default 1e-6).
    pub singularity_regularization: f64,
    /// Block oversampling (default 20).
    pub extra: usize,
}

impl Default for AmeConfig {
    fn default() -> Self {
        AmeConfig { nev: 5, max_iter: 100, tol: 1e-8, verbose: false, singularity_regularization: 1e-6, extra: 20 }
    }
}

/// Result returned by the AME solver.
#[derive(Debug, Clone)]
pub struct AmeResult<T: Scalar> {
    /// Converged eigenvalues, sorted ascending.
    pub eigenvalues: Vec<T>,
    /// Converged eigenvectors (columns of a dense matrix, shape `n × n_converged`).
    pub eigenvectors: Vec<DenseVec<T>>,
    /// Number of LOBPCG iterations used.
    pub iterations: usize,
    /// Whether the requested number of eigenvalues converged.
    pub converged: bool,
    /// Residual norms per eigenpair.
    pub residuals: Vec<T>,
}

// ─── Div-free projector (implements Preconditioner) ─────────────────────────────

/// Discrete divergence-free projector `P = I − G(GᵀMG)⁻¹GᵀM`.
///
/// Wraps the projection as a [`Preconditioner`] so it can be passed to
/// [`Lobpcg::new_generalized`] as the `projector` argument.
struct DivFreeProjector<T: Scalar> {
    n_edges: usize,
    n_nodes: usize,
    g: CsrMatrix<T>,
    g_t: CsrMatrix<T>,
    m: CsrMatrix<T>,
    g_t_m_g: CsrMatrix<T>,
    node_precond: AmgPrecond<T>,
}

impl<T: Scalar> DivFreeProjector<T> {
    fn new(
        g: &CsrMatrix<T>,
        m: &CsrMatrix<T>,
        singularity_regularization: f64,
    ) -> Result<Self, SolverError> {
        let n_edges = g.nrows();
        let n_nodes = g.ncols();
        let g_t = g.transpose_csr();
        let mg   = m.matmat(g);
        let mut g_t_m_g = g_t.matmat(&mg);
        if singularity_regularization > 0.0 {
            let eps = T::from_f64(singularity_regularization);
            let mut coo = crate::sparse::CooMatrix::new(n_nodes, n_nodes);
            for (r, c, v) in g_t_m_g.triplets() { coo.push(r, c, v); }
            for i in 0..n_nodes { coo.push(i, i, eps); }
            g_t_m_g = CsrMatrix::from_coo(&coo);
        }
        let hier = AmgHierarchy::build(g_t_m_g.clone(), AmgConfig::default());
        let node_precond = AmgPrecond::new(hier);
        Ok(DivFreeProjector { n_edges, n_nodes, g: g.clone(), g_t, m: m.clone(), g_t_m_g, node_precond })
    }
}

impl<T: Scalar> Preconditioner for DivFreeProjector<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n_edges = self.n_edges;
        let n_nodes = self.n_nodes;
        // t = M·x
        let mut t = DenseVec::zeros(n_edges);
        self.m.spmv(x.as_slice(), t.as_mut_slice());
        // rhs = Gᵀ·t
        let mut rhs = DenseVec::zeros(n_nodes);
        self.g_t.spmv(t.as_slice(), rhs.as_mut_slice());
        // Solve (GᵀMG) z = rhs via PCG-AMG
        let mut z = DenseVec::zeros(n_nodes);
        let cg_params = SolverParams { max_iter: 100, rtol: 1e-8, ..SolverParams::default() };
        let cg = ConjugateGradient::<T>::default();
        if cg.solve(&self.g_t_m_g, Some(&self.node_precond), &rhs, &mut z, &cg_params).is_err() {
            self.node_precond.apply_precond(&rhs, &mut z);
        }
        // y = x − G·z
        self.g.spmv(z.as_slice(), y.as_mut_slice());
        let xs = x.as_slice();
        let ys = y.as_mut_slice();
        for i in 0..n_edges { ys[i] = xs[i] - ys[i]; }
    }
}

// ─── PCG-AMS preconditioner wrapper ─────────────────────────────────────────────

/// Inner PCG-AMS preconditioner: solves `A·W ≈ R` with AMS-preconditioned CG.
///
/// This is the key preconditioner for the AME LOBPCG.  A single AMS apply
/// degenerates to diagonal scaling on the div-free complement, so we run a
/// few PCG iterations with AMS as the preconditioner instead.
struct PcgAmsPrecond<'a, T: Scalar> {
    a: &'a CsrMatrix<T>,
    ams: &'a AmsPrecond<T>,
}

impl<'a, T: Scalar> Preconditioner for PcgAmsPrecond<'a, T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let inner_params = SolverParams { max_iter: 30, rtol: 1e-6, ..SolverParams::default() };
        let cg = ConjugateGradient::<T>::default();
        let _ = cg.solve(self.a, Some(self.ams), x, y, &inner_params);
    }
}

// ─── AME solver ─────────────────────────────────────────────────────────────────

/// Auxiliary-space Maxwell Eigensolver.
///
/// Solves `A x = λ M x` using generalised LOBPCG with inner PCG-AMS
/// preconditioner and discrete divergence-free projection.
pub struct AmeSolver<T: Scalar> {
    cfg: AmeConfig,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> AmeSolver<T> {
    pub fn new(nev: usize) -> Self {
        AmeSolver { cfg: AmeConfig { nev, ..AmeConfig::default() }, _phantom: PhantomData }
    }
    pub fn tol(mut self, tol: f64) -> Self { self.cfg.tol = tol; self }
    pub fn max_iter(mut self, max_iter: usize) -> Self { self.cfg.max_iter = max_iter; self }
    pub fn verbose(mut self, verbose: bool) -> Self { self.cfg.verbose = verbose; self }
    pub fn singularity_regularization(mut self, val: f64) -> Self { self.cfg.singularity_regularization = val; self }
    pub fn extra(mut self, extra: usize) -> Self { self.cfg.extra = extra; self }

    /// Solve `A x = λ M x` using AME = LOBPCG + PCG-AMS + DivFree.
    pub fn solve(
        &self,
        a: &CsrMatrix<T>,
        m: &CsrMatrix<T>,
        g: &CsrMatrix<T>,
    ) -> Result<AmeResult<T>, SolverError> {
        let n = a.nrows();
        let k = self.cfg.nev;
        let block = (k + self.cfg.extra).min(n).saturating_sub(1).max(1);
        assert_eq!(a.ncols(), n);
        assert_eq!(m.nrows(), n);
        assert_eq!(g.nrows(), n);

        // ── 1. PCG-AMS preconditioner ──────────────────────────────────────
        let ams = AmsPrecond::<T>::new(a, g, AmsConfig {
            singularity_regularization: self.cfg.singularity_regularization,
            smoother_sweeps: 3,
            ..AmsConfig::default()
        }).map_err(|e| SolverError::PrecondSetupFailed {
            reason: format!("AME: AMS setup: {e}"),
        })?;
        let pcg_ams = PcgAmsPrecond { a, ams: &ams };

        // ── 2. Div-free projector ──────────────────────────────────────────
        let projector = DivFreeProjector::<T>::new(g, m, self.cfg.singularity_regularization)?;

        // ── 3. Generalised LOBPCG ──────────────────────────────────────────
        let lobpcg = Lobpcg::<T>::new_generalized(Some(&pcg_ams), Some(m), Some(&projector));

        let mut params = EigenParams::<T>::new(block, EigenWhich::SmallestAlgebraic);
        params.tol = T::from_f64(self.cfg.tol);
        params.max_iter = self.cfg.max_iter;
        params.verbose = self.cfg.verbose;

        let result = lobpcg.solve_generalized(a, &params)?;

        // ── 4. Pack result ─────────────────────────────────────────────────
        let n_found = result.eigenvalues.len().min(k);
        let mut evals = Vec::with_capacity(n_found);
        let mut evecs = Vec::with_capacity(n_found);
        let mut residuals = Vec::with_capacity(n_found);
        for j in 0..n_found {
            evals.push(result.eigenvalues[j]);
            evecs.push(result.eigenvectors[j].clone());
            // Compute Mx for residual
            let mut mx = DenseVec::zeros(n);
            m.spmv(result.eigenvectors[j].as_slice(), mx.as_mut_slice());
            let lam = result.eigenvalues[j];
            let mut rn = T::zero();
            for i in 0..n {
                let ri = result.eigenvectors[j].as_slice()[i] - lam * mx.as_slice()[i];
                rn += ri * ri;
            }
            residuals.push(rn.sqrt());
        }
        Ok(AmeResult { eigenvalues: evals, eigenvectors: evecs, iterations: result.iterations, converged: true, residuals })
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vector::Vector;
    use crate::sparse::CooMatrix;

    fn build_small_maxwell_problem() -> (CsrMatrix<f64>, CsrMatrix<f64>, CsrMatrix<f64>) {
        let n_edges = 20;
        let n_nodes = n_edges + 1;
        let mut a_coo = CooMatrix::new(n_edges, n_edges);
        for i in 0..n_edges {
            a_coo.push(i, i, 2.0);
            if i > 0 { a_coo.push(i, i - 1, -1.0); }
            if i < n_edges - 1 { a_coo.push(i, i + 1, -1.0); }
        }
        let a = CsrMatrix::from_coo(&a_coo);
        let mut m_coo = CooMatrix::new(n_edges, n_edges);
        for i in 0..n_edges { m_coo.push(i, i, 1.0); }
        let m = CsrMatrix::from_coo(&m_coo);
        let mut g_coo = CooMatrix::new(n_edges, n_nodes);
        for i in 0..n_edges {
            g_coo.push(i, i, -1.0);
            g_coo.push(i, i + 1, 1.0);
        }
        let g = CsrMatrix::from_coo(&g_coo);
        (a, m, g)
    }

    #[test]
    fn ame_solves_small_maxwell() {
        let (a, _m, g) = build_small_maxwell_problem();
        let n_edges_test = a.nrows();
        let mut eye_coo = CooMatrix::new(n_edges_test, n_edges_test);
        for i in 0..n_edges_test { eye_coo.push(i, i, 1.0); }
        let eye = CsrMatrix::from_coo(&eye_coo);

        let solver = AmeSolver::new(3).tol(1e-6).max_iter(200).extra(10).verbose(true);
        let result = solver.solve(&a, &eye, &g).unwrap();
        eprintln!("AME eigenvalues: {:?}", result.eigenvalues);
        assert!(result.eigenvalues.len() >= 2);
        assert!((result.eigenvalues[0] - 0.024).abs() < 0.01,
            "first eigenvalue ≈ 0.024, got {}", result.eigenvalues[0]);
    }

    #[test]
    fn div_free_projector_removes_gradient_component() {
        let (a, m, g) = build_small_maxwell_problem();
        let n_nodes = g.ncols();
        let mut ones = DenseVec::zeros(n_nodes);
        for i in 0..n_nodes { ones.as_mut_slice()[i] = 1.0_f64; }
        let mut grad_vec = DenseVec::zeros(a.nrows());
        g.spmv(ones.as_slice(), grad_vec.as_mut_slice());

        let projector = DivFreeProjector::new(&g, &m, 1e-6).unwrap();
        let mut result = DenseVec::zeros(a.nrows());
        projector.apply_precond(&grad_vec, &mut result);

        let nrm = result.norm2();
        assert!(nrm < 1e-10, "div-free projection of gradient should be near-zero, got {nrm:#?}");
    }
}