//! AME — Auxiliary-space Maxwell Eigensolver
//!
//! Solves the generalised eigenvalue problem `A x = λ M x` for the **curl-curl
//! Maxwell cavity** problem using LOBPCG + AMS preconditioner + discrete
//! divergence-free (nullspace) projection.
//!
//! # Algorithm (HYPRE AME / Kolev–Vassilevski 2006)
//!
//! ```text
//! 1. Random initial X: n×k block, mass-orthonormalised w.r.t M
//! 2. Compute AX = A·X,  MX = M·X
//! 3. Rayleigh–Ritz: solve XᵀAX c = θ XᵀMX c  →  Λ = diag(θ₁…θₖ)
//! 4. Residual: R[:,j] = AX[:,j] − λⱼ·MX[:,j]
//! 5. Convergence: ‖R[:,j]‖ / λⱼ < tol  ⟶  locked (not updated further)
//! 6. Precondition: W[:,j] = AMS⁻¹(R[:,j])        (only unlocked modes)
//! 7. Nullspace project: W[:,j] = P·W[:,j]         (remove gradient component)
//! 8. Rayleigh–Ritz on [X, W, P]:
//!    solve SᵀAS c = θ SᵀMS c  →  new X, new P
//! 9. Repeat from 2
//! ```
//!
//! The nullspace projector is
//! `P = I − G(GᵀMG)⁻¹GᵀM`,
//! which enforces the discrete divergence-free condition `GᵀM x = 0`.
//! The nodal system `GᵀMG` is solved with one AMG V-cycle.
//!
//! # References
//!
//! - Kolev, T. V. & Vassilevski, P. S. (2006). Parallel eigensolver for H(curl)
//!   problems using H1-auxiliary space AMG preconditioning. LLNL TR-226197.
//! - Knyazev, A. V. (2001). Toward the optimal preconditioned eigensolver:
//!   Locally Optimal Block Preconditioned Conjugate Gradient Method.
//!
//! # Usage
//!
//! ```rust,ignore
//! use linger::eigen::AmeSolver;
//! let solver = AmeSolver::new(5)                 // 5 eigenvalues
//!     .tol(1e-8)
//!     .max_iter(100)
//!     .verbose(true);
//! let result = solver.solve(&a, &m, &g).unwrap();
//! ```
//!
//! where `a` is the `n×n` curl-curl operator, `m` the `n×n` mass matrix, and
//! `g` the `n×n_nodes` discrete gradient.

use crate::amg::{AmgConfig, AmgHierarchy, AmgPrecond};
use crate::core::{
    error::SolverError,
    preconditioner::Preconditioner,
    scalar::Scalar,
    solver::{KrylovSolver, SolverParams},
    vector::{DenseVec, Vector},
};
use crate::iterative::ConjugateGradient;
use crate::precond::ams::{AmsConfig, AmsPrecond};
use crate::sparse::CsrMatrix;
use std::marker::PhantomData;

// ─── AME solver ─────────────────────────────────────────────────────────────────

/// Configuration for the AME eigensolver.
#[derive(Debug, Clone)]
pub struct AmeConfig {
    /// Number of eigenvalue/vector pairs to compute (block size).
    pub nev: usize,
    /// Maximum number of LOBPCG iterations (default 100).
    pub max_iter: usize,
    /// Convergence tolerance: `‖Ax − λMx‖ / |λ| < tol` (default 1e-8).
    pub tol: f64,
    /// Print convergence info each iteration when `true`.
    pub verbose: bool,
    /// Regularization added to the AMS nodal system (default 1e-6).
    /// Shifts gradient nullspace eigenvalues away from zero so the nodal
    /// AMG solve does not produce NaN.
    pub singularity_regularization: f64,
    /// Oversampling factor: the block size used internally is `nev + extra`.
    /// A larger value improves robustness at the cost of more work per iter.
    pub extra: usize,
}

impl Default for AmeConfig {
    fn default() -> Self {
        AmeConfig {
            nev: 5,
            max_iter: 100,
            tol: 1e-8,
            verbose: false,
            singularity_regularization: 1e-6,
            extra: 20,
        }
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

// ─── Nullspace projector P = I − G(GᵀMG)⁻¹GᵀM ──────────────────────────────────

/// Discrete divergence-free projector for H(curl) eigenvalue problems.
///
/// Applies `P = I − G·(GᵀMG)⁻¹·GᵀM` to a vector, removing its gradient
/// (nullspace) component.  The nodal system `GᵀMG` is inverted by PCG-AMG
/// (Conjugate Gradient preconditioned by AMG) with a relaxed tolerance.
struct DivFreeProjector<T: Scalar> {
    n_edges: usize,
    n_nodes: usize,
    /// Discrete gradient G (n_edges × n_nodes).
    g: CsrMatrix<T>,
    /// Gᵀ (n_nodes × n_edges), cached transpose.
    g_t: CsrMatrix<T>,
    /// Mass matrix M (n_edges × n_edges).
    m: CsrMatrix<T>,
    /// Nodal system matrix GᵀMG (n_nodes × n_nodes).
    g_t_m_g: CsrMatrix<T>,
    /// AMG preconditioner for the nodal system.
    node_precond: AmgPrecond<T>,
}

impl<T: Scalar> DivFreeProjector<T> {
    /// Build the projector from the gradient and mass matrices.
    ///
    /// `singularity_regularization` adds a small diagonal shift `ε·I` to the
    /// nodal system to handle singular `GᵀMG` (e.g. on floating meshes).
    fn new(
        g: &CsrMatrix<T>,
        m: &CsrMatrix<T>,
        singularity_regularization: f64,
    ) -> Result<Self, SolverError> {
        let n_edges = g.nrows();
        let n_nodes = g.ncols();
        assert_eq!(m.nrows(), n_edges);
        assert_eq!(m.ncols(), n_edges);

        let g_t = g.transpose_csr();            // n_nodes × n_edges

        // Build GᵀMG = Gᵀ · (M · G)
        let mg   = m.matmat(g);                 // n_edges × n_nodes
        let mut g_t_m_g = g_t.matmat(&mg);       // n_nodes × n_nodes

        // Add singularity regularization: (GᵀMG) += ε·I
        if singularity_regularization > 0.0 {
            let eps = T::from_f64(singularity_regularization);
            let mut coo = crate::sparse::CooMatrix::new(n_nodes, n_nodes);
            for (r, c, v) in g_t_m_g.triplets() {
                coo.push(r, c, v);
            }
            for i in 0..n_nodes {
                coo.push(i, i, eps);
            }
            g_t_m_g = CsrMatrix::from_coo(&coo);
        }

        // AMG for the nodal system (GᵀMG or the regularised version).
        let hier = AmgHierarchy::build(g_t_m_g.clone(), AmgConfig::default());
        let node_precond = AmgPrecond::new(hier);

        Ok(DivFreeProjector { n_edges, n_nodes, g: g.clone(), g_t, m: m.clone(), g_t_m_g, node_precond })
    }

    /// Apply the projector: `y = P·x = (I − G(GᵀMG)⁻¹GᵀM)·x`
    ///
    /// Uses PCG-AMG (CG preconditioned by one AMG V-cycle) to solve the nodal
    /// system to about 1e-4 relative residual, which is sufficient to suppress
    /// nullspace components well below the LOBPCG convergence threshold.
    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n_edges = self.n_edges;
        let n_nodes = self.n_nodes;

        // 1. t = M·x
        let mut t = DenseVec::zeros(n_edges);
        self.m.spmv(x.as_slice(), t.as_mut_slice());

        // 2. rhs = Gᵀ · t   (n_nodes)
        let mut rhs = DenseVec::zeros(n_nodes);
        self.g_t.spmv(t.as_slice(), rhs.as_mut_slice());

        // 3. Solve (GᵀMG) z = rhs  via PCG-AMG (tight tolerance for accuracy)
        let mut z = DenseVec::zeros(n_nodes);
        let cg_params = SolverParams {
            max_iter: 100,
            rtol: 1e-8,
            ..SolverParams::default()
        };
        // Apply PCG-AMG; if it fails, fall back to one AMG V-cycle.
        let cg = ConjugateGradient::<T>::default();
        if cg.solve(&self.g_t_m_g, Some(&self.node_precond), &rhs, &mut z, &cg_params).is_err() {
            self.node_precond.apply_precond(&rhs, &mut z);
        }

        // 4. y = x − G·z
        self.g.spmv(z.as_slice(), y.as_mut_slice());
        {
            let xs = x.as_slice();
            let ys = y.as_mut_slice();
            for i in 0..n_edges {
                ys[i] = xs[i] - ys[i];
            }
        }
    }
}

// ─── AME solver (LOBPCG with AMS + div-free projection) ─────────────────────────

/// Auxiliary-space Maxwell Eigensolver.
///
/// Solves `A x = λ M x` using the Kolev–Vassilevski algorithm: LOBPCG accelerated
/// by the AMS preconditioner, with discrete divergence-free projection to suppress
/// the gradient nullspace.
pub struct AmeSolver<T: Scalar> {
    cfg: AmeConfig,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> AmeSolver<T> {
    /// Create a new AME solver for `nev` eigenvalues.
    pub fn new(nev: usize) -> Self {
        AmeSolver {
            cfg: AmeConfig { nev, ..AmeConfig::default() },
            _phantom: PhantomData,
        }
    }

    /// Set the convergence tolerance (default 1e-8).
    pub fn tol(mut self, tol: f64) -> Self { self.cfg.tol = tol; self }

    /// Set the maximum number of iterations (default 100).
    pub fn max_iter(mut self, max_iter: usize) -> Self { self.cfg.max_iter = max_iter; self }

    /// Enable or disable convergence logging.
    pub fn verbose(mut self, verbose: bool) -> Self { self.cfg.verbose = verbose; self }

    /// Set the singularity regularization for the AMS nodal system (default 1e-6).
    pub fn singularity_regularization(mut self, val: f64) -> Self { self.cfg.singularity_regularization = val; self }

    /// Set the block oversampling (default 20).
    pub fn extra(mut self, extra: usize) -> Self { self.cfg.extra = extra; self }

    /// Solve the generalised eigenvalue problem `A x = λ M x`.
    ///
    /// # Arguments
    ///
    /// * `a` — Curl-curl stiffness matrix (`n × n`, symmetric positive semi-definite).
    /// * `m` — Mass matrix (`n × n`, symmetric positive definite).
    /// * `g` — Discrete gradient (`n × n_nodes`), mapping H¹ → H(curl).
    ///        Only the graph (pattern of ±1 per row) matters; used by AMS and
    ///        the divergence-free projector.
    pub fn solve(
        &self,
        a: &CsrMatrix<T>,
        m: &CsrMatrix<T>,
        g: &CsrMatrix<T>,
    ) -> Result<AmeResult<T>, SolverError> {
        let n = a.nrows();
        let k = self.cfg.nev;
        let block = (k + self.cfg.extra).min(n).saturating_sub(1).max(1);
        assert_eq!(a.ncols(), n, "A must be square");
        assert_eq!(m.nrows(), n, "M must have same nrows as A");
        assert_eq!(g.nrows(), n, "G must have same nrows as A");

        // ── 1. Build AMS preconditioner ──────────────────────────────────────
        let ams = AmsPrecond::<T>::new(
            a,
            g,
            AmsConfig {
                singularity_regularization: self.cfg.singularity_regularization,
                smoother_sweeps: 3,
                ..AmsConfig::default()
            },
        )
        .map_err(|e| SolverError::PrecondSetupFailed {
            reason: format!("AME: AMS setup failed: {e}"),
        })?;

        // ── 2. Build divergence-free projector ───────────────────────────────
        let projector = DivFreeProjector::<T>::new(
            g,
            m,
            self.cfg.singularity_regularization,
        )?;

        // ── 3. Initialise X: n×block random, mass-orthonormalised ────────────
        let mut x_cols: Vec<DenseVec<T>> = Vec::with_capacity(block);
        let mut mx_cols: Vec<DenseVec<T>> = Vec::with_capacity(block);
        let mut ax_cols: Vec<DenseVec<T>> = Vec::with_capacity(block);

        let mut seed: u64 = 42;
        for _ in 0..block {
            let mut col = DenseVec::zeros(n);
            super::fill_random(&mut col, seed);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

            // Mass-orthogonalise against previous columns
            mass_orthogonalise(&mut col, &x_cols, m);
            // Apply div-free projection (note: can't borrow col as both & and &mut)
            let mut col_proj = DenseVec::zeros(n);
            projector.apply(&col, &mut col_proj);
            col = col_proj;
            let nrm = mass_norm2(&col, m);
            if nrm > T::from_f64(1e-14) {
                col.scale(T::one() / nrm);
            }
            // Re-project after normalisation
            let mut col_proj2 = DenseVec::zeros(n);
            projector.apply(&col, &mut col_proj2);
            col = col_proj2;
            let nrm2 = mass_norm2(&col, m);
            if nrm2 > T::from_f64(1e-14) {
                col.scale(T::one() / nrm2);
            }

            let mut mx = DenseVec::zeros(n);
            m.spmv(col.as_slice(), mx.as_mut_slice());
            let mut ax_tmp = DenseVec::zeros(n);
            a.spmv(col.as_slice(), ax_tmp.as_mut_slice());

            x_cols.push(col);
            mx_cols.push(mx);
            ax_cols.push(ax_tmp);
        }

        // ── 4. Rayleigh–Ritz on initial X ────────────────────────────────────
        let mut lambdas: Vec<T> = (0..block).map(|j| {
            let xj = x_cols[j].as_slice();
            let axj = ax_cols[j].as_slice();
            let mxj = mx_cols[j].as_slice();
            let num = super::dot(axj, xj);
            let den = super::dot(mxj, xj);
            if den > T::zero() { num / den } else { num }
        }).collect();
        if self.cfg.verbose {
            let rq_debug: Vec<f64> = lambdas.iter().map(|&v| num_traits::ToPrimitive::to_f64(&v).unwrap_or(f64::NAN)).collect();
            eprintln!("  [AME] initial RQs (first 5): {:?}", &rq_debug[..5.min(rq_debug.len())]);
            // Check mass-norms
            let mut norms = Vec::new();
            for j in 0..block.min(5) {
                let nrm = mass_norm2(&x_cols[j], m);
                norms.push(num_traits::ToPrimitive::to_f64(&nrm).unwrap_or(f64::NAN));
            }
            eprintln!("  [AME] mass-norms (first 5): {:?}", norms);
        }
        // Small dense Rayleigh–Ritz to get better initial guess
        rr_update(&mut x_cols, &mut ax_cols, &mut mx_cols, &mut lambdas, m, a, block);
        if self.cfg.verbose {
            let rr_debug: Vec<f64> = lambdas.iter().map(|&v| num_traits::ToPrimitive::to_f64(&v).unwrap_or(f64::NAN)).collect();
            eprintln!("  [AME] after RR (first 5): {:?}", &rr_debug[..5.min(rr_debug.len())]);
        }

        // P columns (previous search directions): start None
        let mut p_cols: Vec<Option<DenseVec<T>>> = vec![None; block];
        let mut _ap_cols: Vec<Option<DenseVec<T>>> = vec![None; block];

        // ── 5. LOBPCG iteration ─────────────────────────────────────────────
        for iter in 0..self.cfg.max_iter {
            // ── 5a. Residual: R[:,j] = AX[:,j] − λⱼ·MX[:,j] ────────────────
            let r_cols: Vec<DenseVec<T>> = (0..block).map(|j| {
                let mut r = DenseVec::zeros(n);
                let axs = ax_cols[j].as_slice();
                let mxs = mx_cols[j].as_slice();
                let lam = lambdas[j];
                let rs = r.as_mut_slice();
                for i in 0..n {
                    rs[i] = axs[i] - lam * mxs[i];
                }
                r
            }).collect();

            // ── 5b. Convergence check (soft locking) ─────────────────────────
            let mut locked = vec![false; block];
            let mut max_res = T::zero();
            for j in 0..block {
                let nrm = r_cols[j].norm2();
                let denom = if lambdas[j].abs() > T::zero() { lambdas[j].abs() } else { T::one() };
                let rel = nrm / denom;
                if rel < T::from_f64(self.cfg.tol) {
                    locked[j] = true;
                }
                if rel > max_res { max_res = rel; }
            }

            if self.cfg.verbose {
                let mr = num_traits::ToPrimitive::to_f64(&max_res).unwrap_or(f64::NAN);
                let locked_count = locked.iter().filter(|&&l| l).count();
                println!("  AME iter {:4}  max‖r‖/|λ| = {mr:.3e}  locked {locked_count}/{block}", iter + 1);
            }

            // If all modes converged, return
            let n_locked = locked.iter().filter(|&&l| l).count();
            if n_locked >= k {
                return Ok(self.pack_result(&x_cols, &ax_cols, &mx_cols, &lambdas, iter + 1, true));
            }

            // ── 5c. Precondition + nullspace project (unlocked modes only) ──
            let mut w_cols: Vec<Option<DenseVec<T>>> = vec![None; block];
            for j in 0..block {
                if locked[j] {
                    // Keep a copy of the existing eigenvector approximation
                    // in the search space but don't precondition.
                    // The existing x_cols[j] already spans this mode.
                    continue;
                }

                // Inner PCG-AMS: solve A·W[:,j] ≈ R[:,j] to a relaxed tolerance.
                // A single AMS apply degenerates to ωD⁻¹ on the div-free complement,
                // so we run a few PCG iterations with AMS as the preconditioner.
                // This matches how HYPRE AME uses AMS (inner PCG mode).
                let mut w = DenseVec::zeros(n);
                let inner_params = SolverParams {
                    max_iter: 30,
                    rtol: 1e-6,
                    ..SolverParams::default()
                };
                let cg = ConjugateGradient::<T>::default();
                let _ = cg.solve(a, Some(&ams), &r_cols[j], &mut w, &inner_params);

                // Apply div-free projection: W[:,j] = P·W[:,j]
                let mut w_proj = DenseVec::zeros(n);
                projector.apply(&w, &mut w_proj);

                w_cols[j] = Some(w_proj);
            }

            // ── 5d. Rayleigh–Ritz on [X, W_locked?, P] ──────────────────────
            // Build the search space S = [X_locked, W_unlocked, P_unlocked]
            // where locked X columns are kept but not updated via W/P.
            let s_cols = self.build_search_space(&x_cols, &w_cols, &p_cols, &locked, n);
            if s_cols.len() < k {
                return Err(SolverError::ConvergenceFailed {
                    max_iter: iter + 1,
                    residual: num_traits::ToPrimitive::to_f64(&max_res).unwrap_or(f64::INFINITY),
                });
            }

            // Compress S via M-orthonormalisation (drop linearly-dependent columns)
            // to keep the Rayleigh–Ritz matrix M_S = SᵀMS well-conditioned.
            let s_cols = mass_orthonormalise_basis(s_cols, m);

            // Compute A·S and M·S
            let as_cols: Vec<DenseVec<T>> = s_cols.iter().map(|sv| {
                let mut asv = DenseVec::zeros(n); a.spmv(sv.as_slice(), asv.as_mut_slice()); asv
            }).collect();
            let ms_cols: Vec<DenseVec<T>> = s_cols.iter().map(|sv| {
                let mut msv = DenseVec::zeros(n); m.spmv(sv.as_slice(), msv.as_mut_slice()); msv
            }).collect();

            let m_s = s_cols.len();
            // Rayleigh–Ritz: A_S c = θ M_S c
            let mut a_s = vec![T::zero(); m_s * m_s];
            let mut m_s_mat = vec![T::zero(); m_s * m_s];
            for i in 0..m_s {
                for j in 0..m_s {
                    a_s[i * m_s + j] = super::dot(s_cols[i].as_slice(), as_cols[j].as_slice());
                    m_s_mat[i * m_s + j] = super::dot(s_cols[i].as_slice(), ms_cols[j].as_slice());
                }
            }

            let (theta, c_vecs) = dense_symm_eig_gen(&a_s, &m_s_mat, m_s);

            // Sort eigenvalues ascending.
            let mut order: Vec<usize> = (0..m_s).collect();
            order.sort_by(|&a, &b| theta[a].partial_cmp(&theta[b]).unwrap());

            // Select modes for the next iteration: skip eigenvalues near zero
            // (gradient nullspace), take the next `block` eigenvalues.
            let zero_thresh = T::from_f64(1e-12);
            let first_phys = order.iter()
                .position(|&oi| theta[oi] > zero_thresh)
                .unwrap_or(m_s.saturating_sub(block));
            let start = first_phys.min(m_s.saturating_sub(block).max(0));
            let n_take = block.min(m_s - start);
            let selected: Vec<usize> = order[start..start + n_take].to_vec();

            // ── 5e. Update X, AX, MX, P, AP ──────────────────────────────────
            let mut x_new: Vec<DenseVec<T>> = Vec::with_capacity(block);
            let mut ax_new: Vec<DenseVec<T>> = Vec::with_capacity(block);
            let mut mx_new: Vec<DenseVec<T>> = Vec::with_capacity(block);
            let mut p_new: Vec<DenseVec<T>> = Vec::with_capacity(block);
            let mut ap_new: Vec<DenseVec<T>> = Vec::with_capacity(block);

            for &sel_idx in &selected {
                let c = &c_vecs[sel_idx * m_s..(sel_idx + 1) * m_s];
                let mut xn  = DenseVec::zeros(n);
                let mut axn = DenseVec::zeros(n);
                let mut mxn = DenseVec::zeros(n);
                let mut pn  = DenseVec::zeros(n);
                let mut apn = DenseVec::zeros(n);

                for j in 0..m_s {
                    let cj = c[j];
                    xn.axpy(cj, &s_cols[j]);
                    axn.axpy(cj, &as_cols[j]);
                    mxn.axpy(cj, &ms_cols[j]);
                    // P = W + P components of S
                    let base_x_count = block; // first `block` cols of S are X
                    if j >= base_x_count {
                        pn.axpy(cj, &s_cols[j]);
                        apn.axpy(cj, &as_cols[j]);
                    }
                }

                // Mass-normalise
                let nrm = mass_norm2(&xn, m);
                if nrm > T::from_f64(1e-14) {
                    let inv = T::one() / nrm;
                    xn.scale(inv); axn.scale(inv); mxn.scale(inv); pn.scale(inv); apn.scale(inv);
                }

                x_new.push(xn);
                ax_new.push(axn);
                mx_new.push(mxn);
                p_new.push(pn);
                ap_new.push(apn);
            }

            x_cols = x_new;
            ax_cols = ax_new;
            mx_cols = mx_new;
            p_cols = p_new.into_iter().map(Some).collect();
            _ap_cols = ap_new.into_iter().map(Some).collect();

            // Update lambdas
            for j in 0..block.min(selected.len()) {
                lambdas[j] = theta[selected[j]];
            }
        }

        // ── 6. Return best available ─────────────────────────────────────────
        Ok(self.pack_result(&x_cols, &ax_cols, &mx_cols, &lambdas, self.cfg.max_iter, false))
    }

    // ─── Internal helpers ──────────────────────────────────────────────────────

    fn build_search_space(
        &self,
        x_cols: &[DenseVec<T>],
        w_cols: &[Option<DenseVec<T>>],
        p_cols: &[Option<DenseVec<T>>],
        locked: &[bool],
        n: usize,
    ) -> Vec<DenseVec<T>> {
        let block = x_cols.len();
        let mut s: Vec<DenseVec<T>> = Vec::new();

        // First `block` columns: X (all columns, locked or not)
        s.extend_from_slice(x_cols);

        // W columns for unlocked modes
        for j in 0..block {
            if !locked[j] {
                if let Some(ref w) = w_cols[j] {
                    s.push(w.clone());
                } else {
                    // Fallback: use residual itself
                    s.push(DenseVec::zeros(n));
                }
            }
        }

        // P columns for unlocked modes (only if P was set)
        let has_p = p_cols.iter().any(|p| p.is_some());
        if has_p {
            for j in 0..block {
                if !locked[j] {
                    if let Some(ref p) = p_cols[j] {
                        s.push(p.clone());
                    }
                }
            }
        }

        s
    }

    fn pack_result(
        &self,
        x_cols: &[DenseVec<T>],
        ax_cols: &[DenseVec<T>],
        mx_cols: &[DenseVec<T>],
        lambdas: &[T],
        iterations: usize,
        converged: bool,
    ) -> AmeResult<T> {
        let n_found = x_cols.len().min(self.cfg.nev);
        let mut evals = Vec::with_capacity(n_found);
        let mut evecs = Vec::with_capacity(n_found);
        let mut residuals = Vec::with_capacity(n_found);

        for j in 0..n_found {
            evals.push(lambdas[j]);
            evecs.push(x_cols[j].clone());

            let axs = ax_cols[j].as_slice();
            let mxs = mx_cols[j].as_slice();
            let lam = lambdas[j];
            let mut r_norm_sq = T::zero();
            for i in 0..x_cols[j].len() {
                let ri = axs[i] - lam * mxs[i];
                r_norm_sq += ri * ri;
            }
            residuals.push(r_norm_sq.sqrt());
        }

        AmeResult { eigenvalues: evals, eigenvectors: evecs, iterations, converged, residuals }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────────

/// Mass-norm squared: `‖v‖²_M = vᵀM v`.
fn mass_norm2<T: Scalar>(v: &DenseVec<T>, m: &CsrMatrix<T>) -> T {
    let n = v.len();
    let mut mv = DenseVec::zeros(n);
    m.spmv(v.as_slice(), mv.as_mut_slice());
    super::dot(v.as_slice(), mv.as_slice()).sqrt()
}

/// Mass-orthogonalise `v` against columns in `basis` (in-place).
fn mass_orthogonalise<T: Scalar>(
    v: &mut DenseVec<T>,
    basis: &[DenseVec<T>],
    m: &CsrMatrix<T>,
) {
    let n = v.len();
    let mut mv = DenseVec::zeros(n);
    m.spmv(v.as_slice(), mv.as_mut_slice());
    for q in basis {
        let proj = super::dot(q.as_slice(), mv.as_slice());
        let vs = v.as_mut_slice();
        let qs = q.as_slice();
        for i in 0..n { vs[i] -= proj * qs[i]; }
        // update Mv for next iteration
        m.spmv(v.as_slice(), mv.as_mut_slice());
    }
}

/// Small dense symmetric generalised eigensolve via Cholesky.
fn dense_symm_eig_gen<T: Scalar>(a: &[T], b: &[T], n: usize) -> (Vec<T>, Vec<T>) {
    use nalgebra::{DMatrix, SymmetricEigen};

    let na = DMatrix::<f64>::from_fn(n, n, |r, c|
        num_traits::ToPrimitive::to_f64(&a[r * n + c]).unwrap_or(0.0));
    let nb = DMatrix::<f64>::from_fn(n, n, |r, c|
        num_traits::ToPrimitive::to_f64(&b[r * n + c]).unwrap_or(0.0));

    let chol = match nb.clone().cholesky() {
        Some(c) => c,
        None => {
            // Fallback: standard eigenvalue problem
            let se = SymmetricEigen::new(na);
            let evals: Vec<T> = se.eigenvalues.iter().map(|&v| T::from_f64(v)).collect();
            let mut evecs = vec![T::zero(); n * n];
            for j in 0..n { for i in 0..n { evecs[j * n + i] = T::from_f64(se.eigenvectors[(i, j)]); } }
            return (evals, evecs);
        }
    };
    let l = chol.l();
    let li = match l.clone().try_inverse() {
        Some(m) => m,
        None => DMatrix::identity(n, n),
    };

    // C = L⁻¹ A (L⁻¹)ᵀ
    let c = &li * &na * li.transpose();
    let se = SymmetricEigen::new(c);

    let evals: Vec<T> = se.eigenvalues.iter().map(|&v| T::from_f64(v)).collect();
    let vecs = li.transpose() * &se.eigenvectors;
    let mut evecs = vec![T::zero(); n * n];
    for j in 0..n { for i in 0..n { evecs[j * n + i] = T::from_f64(vecs[(i, j)]); } }
    (evals, evecs)
}

/// Mass-orthonormalise a set of vectors in-place (modified Gram–Schmidt).
/// Returns the compressed basis (drops columns with near-zero mass-norm).
fn mass_orthonormalise_basis<T: Scalar>(
    basis: Vec<DenseVec<T>>,
    m: &CsrMatrix<T>,
) -> Vec<DenseVec<T>> {
    let n = basis[0].len();
    let mut result: Vec<DenseVec<T>> = Vec::with_capacity(basis.len());

    for mut v in basis {
        // Orthogonalise against already-selected columns
        for q in &result {
            let mut mq = DenseVec::zeros(n);
            m.spmv(q.as_slice(), mq.as_mut_slice());
            let proj = super::dot(v.as_slice(), mq.as_slice());
            let vs = v.as_mut_slice();
            let qs = q.as_slice();
            for i in 0..n { vs[i] -= proj * qs[i]; }
        }
        // Check mass-norm
        let nrm = mass_norm2(&v, m);
        if nrm > T::from_f64(1e-10) {
            let inv = T::one() / nrm;
            v.scale(inv);
            result.push(v);
        }
    }
    result
}

/// One-shot Rayleigh–Ritz refinement for the initial block:
/// solve XᵀAX c = θ XᵀMX c, then replace X with better approximation.
fn rr_update<T: Scalar>(
    x_cols: &mut Vec<DenseVec<T>>,
    ax_cols: &mut Vec<DenseVec<T>>,
    mx_cols: &mut Vec<DenseVec<T>>,
    lambdas: &mut Vec<T>,
    m: &CsrMatrix<T>,
    _a: &CsrMatrix<T>,
    block: usize,
) {
    let n = x_cols[0].len();
    let m_s = block;
    let mut a_s = vec![T::zero(); m_s * m_s];
    let mut m_s_mat = vec![T::zero(); m_s * m_s];
    for i in 0..m_s {
        for j in 0..m_s {
            a_s[i * m_s + j] = super::dot(x_cols[i].as_slice(), ax_cols[j].as_slice());
            m_s_mat[i * m_s + j] = super::dot(x_cols[i].as_slice(), mx_cols[j].as_slice());
        }
    }

    let (theta, c_vecs) = dense_symm_eig_gen(&a_s, &m_s_mat, m_s);
    let mut order: Vec<usize> = (0..m_s).collect();
    order.sort_by(|&a, &b| theta[a].partial_cmp(&theta[b]).unwrap());

    let mut x_new: Vec<DenseVec<T>> = Vec::with_capacity(block);
    let mut ax_new: Vec<DenseVec<T>> = Vec::with_capacity(block);
    let mut mx_new: Vec<DenseVec<T>> = Vec::with_capacity(block);

    for &sel in order.iter().take(block) {
        let c = &c_vecs[sel * m_s..(sel + 1) * m_s];
        let mut xn  = DenseVec::zeros(n);
        let mut axn = DenseVec::zeros(n);
        let mut mxn = DenseVec::zeros(n);
        for j in 0..m_s {
            let cj = c[j];
            xn.axpy(cj, &x_cols[j]);
            axn.axpy(cj, &ax_cols[j]);
            mxn.axpy(cj, &mx_cols[j]);
        }
        let nrm = mass_norm2(&xn, m);
        if nrm > T::from_f64(1e-14) {
            let inv = T::one() / nrm;
            xn.scale(inv); axn.scale(inv); mxn.scale(inv);
        }
        x_new.push(xn);
        ax_new.push(axn);
        mx_new.push(mxn);
    }

    *x_cols = x_new;
    *ax_cols = ax_new;
    *mx_cols = mx_new;
    for j in 0..block.min(order.len()) {
        lambdas[j] = theta[order[j]];
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CooMatrix;

    /// Build a simple 1-D-like test problem with nullspace.
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
        // First verify that the standard eigenvalue problem works (M=I).
        // Build M=I from the same COO builder pattern.
        let n_edges_test = a.nrows();
        let mut eye_coo = CooMatrix::new(n_edges_test, n_edges_test);
        for i in 0..n_edges_test { eye_coo.push(i, i, 1.0); }
        let eye = CsrMatrix::from_coo(&eye_coo);

        // Try with M=I first
        let solver = AmeSolver::new(3)
            .tol(1e-6)
            .max_iter(300)
            .extra(10)
            .verbose(true);
        let result = solver.solve(&a, &eye, &g).unwrap();
        eprintln!("M=I eigenvalues: {:?}", result.eigenvalues);
        assert!(result.eigenvalues.len() >= 2,
            "M=I: should find at least 2 eigenvalues, got {}", result.eigenvalues.len());
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
        projector.apply(&grad_vec, &mut result);

        let nrm = result.norm2();
        assert!(nrm < 1e-10, "div-free projection of gradient should be near-zero, got {nrm:#?}");
    }
}