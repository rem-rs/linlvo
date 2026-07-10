//! Auxiliary-space Maxwell Solver (AMS) preconditioner.
//!
//! Implements the 2-term Hiptmair-Xu auxiliary-space preconditioner for
//! H(curl) edge-element discretisations of Maxwell-type problems:
//!
//! ```text
//! M_AMS⁻¹ x  ≈  ω D_A⁻¹ x  +  G · P_v⁻¹ · Gᵀ x
//! ```
//!
//! where
//! - `D_A` is the diagonal of the edge stiffness matrix `A`,
//! - `G`   is the discrete gradient matrix (nodes → edges, user-supplied),
//! - `P_v` is an approximate solver for the nodal Laplacian `GᵀAG`.
//!
//! The coarse nodal solve `P_v` can be either AMG (recommended for large
//! problems) or ILU(0) (suitable for small/medium problems).
//!
//! ## Usage
//!
//! ```text
//! use linger::precond::{AmsPrecond, AmsConfig, AuxSpaceSolver};
//!
//! // G: discrete gradient, n_edges × n_nodes, user-assembled
//! let config = AmsConfig::default();  // AMG coarse solve, ω = 0.667
//! let precond = AmsPrecond::new(&a_edge, &g, config)?;
//!
//! // Use as a Krylov preconditioner:
//! ConjugateGradient::default().solve(&a_edge, Some(&precond), &b, &mut x, &params)?;
//! ```
//!
//! ## References
//!
//! Hiptmair, R. & Xu, J. (2007). Nodal auxiliary space preconditioning in
//! H(curl) and H(div) spaces. *SIAM J. Numer. Anal.*, 45(6), 2483–2509.
//!
//! Kolev, T.V. & Vassilevski, P.S. (2009). Parallel auxiliary space AMG for
//! H(curl) problems. *J. Comput. Math.*, 27(5), 604–623.

#![allow(clippy::needless_range_loop)]

use crate::amg::{AmgConfig, AmgHierarchy, AmgPrecond};
use crate::core::{
    error::SolverError,
    operator::TransposeOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    vector::DenseVec,
};
use crate::precond::ilu0::Ilu0Precond;
use crate::sparse::{CooMatrix, CsrMatrix};

/// Profiling summary for an AMG auxiliary-space solve.
#[derive(Debug, Clone)]
pub struct AuxAmgProfile {
    /// Number of AMG levels.
    pub n_levels: usize,
    /// AMG operator complexity.
    pub operator_complexity: f64,
    /// AMG grid complexity.
    pub grid_complexity: f64,
}

/// Profiling summary for an auxiliary-space solver backend.
#[derive(Debug, Clone)]
pub enum AuxSolverProfile {
    /// Algebraic multigrid backend diagnostics.
    Amg(AuxAmgProfile),
    /// ILU(0) backend diagnostics.
    Ilu0 {
        /// Matrix size of the auxiliary-space operator.
        n: usize,
        /// Stored non-zeros of the auxiliary-space operator.
        nnz: usize,
    },
}

// ─── AuxSpaceSolver ──────────────────────────────────────────────────────────

/// Choice of solver for the auxiliary-space coarse problem.
///
/// Used by both [`AmsPrecond`] (nodal solve) and [`AdsPrecond`] (edge and
/// nodal solves).
///
/// ## ILU(0) caveat
///
/// When `A = GGᵀ` the coarse operator `GᵀAG` is singular (its null space
/// is spanned by constant node vectors).  AMG handles this gracefully;
/// ILU(0) will return `PrecondSetupFailed` due to a zero pivot.  In practice
/// always add a small diagonal shift `δI` to `A` before constructing the
/// preconditioner if using `Ilu0`.
#[derive(Debug, Clone)]
pub enum AuxSpaceSolver {
    /// Algebraic multigrid (recommended).  Uses the project's existing AMG
    /// hierarchy; effective for the scalar Laplacian-like coarse operators.
    Amg(AmgConfig),
    /// Incomplete LU with zero fill-in.  Fast setup; suitable for problems
    /// where the coarse operator is small (n_nodes ≲ 5 000) and non-singular.
    Ilu0,
}

impl Default for AuxSpaceSolver {
    fn default() -> Self { Self::Amg(AmgConfig::default()) }
}

// ─── AmsConfig ───────────────────────────────────────────────────────────────

/// Configuration for [`AmsPrecond`].
#[derive(Debug, Clone)]
pub struct AmsConfig {
    /// Damping weight ω for the edge-space Jacobi smoother.
    ///
    /// Typical value: 2/3 ≈ 0.667 (optimal for model problems).
    pub smoother_omega: f64,
    /// Number of pre/post-smoothing sweeps (power/BiCG-stable iterations).
    /// More sweeps improve h-independence at the cost of more SpMV calls.
    /// Default: 1 (one Jacobi step).  Recommended: 3–5 for strong scaling.
    pub smoother_sweeps: usize,
    /// Approximate solver for the nodal Laplacian `GᵀAG`.
    pub node_solver: AuxSpaceSolver,
    /// Regularization added to the diagonal of the nodal system `GᵀAG`.
    /// Set to a small positive value (e.g. 10⁻⁶) when the auxiliary space
    /// is singular (e.g. curl-curl eigenvalue problems where gradient fields
    /// map to the nullspace).  Zero (default) means no regularization.
    ///
    /// This is similar to MFEM's `SetSingularProblem()` which tells AMS to
    /// handle the H¹ nodal operator nullspace internally.
    pub singularity_regularization: f64,
}

impl Default for AmsConfig {
    fn default() -> Self {
        AmsConfig {
            smoother_omega: 0.667,
            smoother_sweeps: 1,
            node_solver: AuxSpaceSolver::default(),
            singularity_regularization: 0.0,
        }
    }
}

impl AmsConfig {
    /// HPC-oriented default for auxiliary-space Maxwell solves.
    ///
    /// Uses 3 Jacobi smoothing sweeps for better h-independence,
    /// SA-AMG with larger coarse threshold for the node solve.
    pub fn hpc_default() -> Self {
        AmsConfig {
            smoother_omega: 0.667,
            smoother_sweeps: 3,
            node_solver: AuxSpaceSolver::Amg(AmgConfig {
                coarse_threshold: 64,
                max_levels: 30,
                ..AmgConfig::default()
            }),
            singularity_regularization: 0.0,
        }
    }
}

/// Lightweight setup diagnostics for [`AmsPrecond`].
#[derive(Debug, Clone)]
pub struct AmsProfile {
    /// Number of edge DOFs.
    pub n_edges: usize,
    /// Number of node DOFs.
    pub n_nodes: usize,
    /// Non-zeros in the fine operator `A`.
    pub a_nnz: usize,
    /// Non-zeros in the discrete gradient `G`.
    pub g_nnz: usize,
    /// Non-zeros in the assembled coarse operator `G^T A G`.
    pub a_node_nnz: usize,
    /// Auxiliary-space backend profile for the nodal solve.
    pub node_solver: AuxSolverProfile,
}

// ─── AmsPrecond ──────────────────────────────────────────────────────────────

/// AMS preconditioner for H(curl) edge-element Maxwell problems.
///
/// Constructed via [`AmsPrecond::new`]; implements [`Preconditioner`] and can
/// be passed directly to any [`KrylovSolver`](crate::KrylovSolver).
///
/// # Multi-sweep smoothing
///
/// When `smoother_sweeps > 1`, the preconditioner applies `smoother_sweeps`
/// sweeps of a preconditioned Richardson iteration:
///
/// ```text
/// y⁰ = 0
/// for l = 1…K:
///   rˡ = x - A·yˡ⁻¹
///   yˡ = yˡ⁻¹ + ω·D⁻¹·rˡ  +  G·P_v⁻¹·Gᵀ·rˡ
/// y = yᵏ
/// ```
///
/// More sweeps improve h-independence and robustness for Maxwell eigenvalue
/// problems at the cost of additional SpMV per preconditioner application.
pub struct AmsPrecond<T: Scalar> {
    n_edges: usize,
    n_nodes: usize,
    /// Edge stiffness matrix A (stored for multi-sweep residual).
    a: CsrMatrix<T>,
    /// Precomputed ω / d_i for each edge i (avoids division in apply).
    scaled_inv_diag: Vec<T>,
    /// Discrete gradient G: n_edges × n_nodes (column-sparse in practice).
    g: CsrMatrix<T>,
    /// Number of smoother sweeps to apply.
    smoother_sweeps: usize,
    /// Approximate solver for the nodal coarse problem GᵀAG.
    node_precond: Box<dyn Preconditioner<Vector = DenseVec<T>>>,
    /// Setup diagnostics for observability and tuning.
    profile: AmsProfile,
}

impl<T: Scalar> AmsPrecond<T> {
    /// Build the AMS preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a`      — Edge stiffness matrix, square `n_edges × n_edges`.
    /// * `g`      — Discrete gradient matrix, `n_edges × n_nodes`.
    ///   Each row has exactly two non-zeros: −1 at the tail node
    ///   and +1 at the head node (standard FE convention).
    /// * `config` — Smoother weight and coarse-solver choice.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::PrecondSetupFailed`] if:
    /// - `a` is not square,
    /// - `g.nrows() ≠ a.nrows()`,
    /// - `g.ncols() == 0` (no node DOFs),
    /// - a diagonal entry of `a` is near-zero (< ε · 10⁶),
    /// - the coarse-solver setup fails (e.g. ILU(0) on a singular `GᵀAG`).
    pub fn new(
        a:      &CsrMatrix<T>,
        g:      &CsrMatrix<T>,
        config: AmsConfig,
    ) -> Result<Self, SolverError> {
        let n_edges = a.nrows();
        let n_nodes = g.ncols();

        // ── 1. Dimension checks ──────────────────────────────────────────────
        if a.ncols() != n_edges {
            return Err(SolverError::PrecondSetupFailed {
                reason: format!("AMS: A must be square, got {}×{}", n_edges, a.ncols()),
            });
        }
        if g.nrows() != n_edges {
            return Err(SolverError::PrecondSetupFailed {
                reason: format!(
                    "AMS: G must have nrows = n_edges = {n_edges}, got {}",
                    g.nrows()
                ),
            });
        }
        if n_nodes == 0 {
            return Err(SolverError::PrecondSetupFailed {
                reason: "AMS: G has zero columns (no node DOFs)".into(),
            });
        }

        // ── 2. Edge smoother: ω / d_i ────────────────────────────────────────
        let omega = T::from_f64(config.smoother_omega);
        let tol   = T::machine_epsilon() * T::from_f64(1e6);
        let diag  = a.diag();
        let scaled_inv_diag: Vec<T> = diag
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d.abs() < tol {
                    Err(SolverError::PrecondSetupFailed {
                        reason: format!("AMS: near-zero diagonal in A at row {i}: {d:?}"),
                    })
                } else {
                    Ok(omega / d)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // ── 3. Coarse operator: A_node = GᵀAG ───────────────────────────────
        let g_t    = g.transpose_csr();   // n_nodes × n_edges
        let ag     = a.matmat(g);         // n_edges × n_nodes
        let mut a_node = g_t.matmat(&ag);     // n_nodes × n_nodes

        // When `singularity_regularization > 0`, add ε·GᵀG to the nodal
        // system to shift nullspace eigenvalues away from zero, preventing NaN
        // in AMG/ILU(0) coarse solves for singular problems (e.g. curl-curl
        // eigenvalue problems).
        let eps_f64 = config.singularity_regularization;
        if eps_f64 > 0.0 {
            let eps = T::from_f64(eps_f64);
            // GᵀG has the same dimensions as GᵀAG: n_nodes × n_nodes.
            // The `g_t` transpose from step 3.1 is still live.
            let gtg = g_t.matmat(g);       // n_nodes × n_nodes
            // Merge a_node + ε·GᵀG via COO (CooMatrix handles duplicate summing).
            let mut coo = CooMatrix::new(n_nodes, n_nodes);
            for (r, c, v) in a_node.triplets() {
                coo.push(r, c, v);
            }
            for (r, c, v) in gtg.triplets() {
                coo.push(r, c, eps * v);
            }
            a_node = CsrMatrix::from_coo(&coo);
        }

        // ── 4. Coarse solver ─────────────────────────────────────────────────
        let a_node_nnz = a_node.nnz();
        let (node_precond, node_solver_profile) = build_aux_solver(a_node, &config.node_solver)?;

        let profile = AmsProfile {
            n_edges,
            n_nodes,
            a_nnz: a.nnz(),
            g_nnz: g.nnz(),
            a_node_nnz,
            node_solver: node_solver_profile,
        };

        Ok(AmsPrecond {
            n_edges,
            n_nodes,
            a: a.clone(),
            scaled_inv_diag,
            g: g.clone(),
            smoother_sweeps: config.smoother_sweeps,
            node_precond,
            profile,
        })
    }

    /// Setup-time profile for diagnostics and performance tuning.
    pub fn profile(&self) -> &AmsProfile { &self.profile }
}

impl<T: Scalar> Preconditioner for AmsPrecond<T> {
    type Vector = DenseVec<T>;

    /// Apply the AMS preconditioner.
    ///
    /// When `smoother_sweeps == 1` this is the standard Hiptmair-Xu
    /// preconditioner `M⁻¹ ≈ ω·D⁻¹ + G·P_v⁻¹·Gᵀ`.
    ///
    /// When `smoother_sweeps > 1`, multi-sweep Richardson is used (see
    /// struct-level docs), which gives better h-independence and robustness
    /// for Maxwell eigenvalue problems.
    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n_edges = self.n_edges;
        let n_nodes = self.n_nodes;

        // y = 0
        for ys in y.as_mut_slice().iter_mut().take(n_edges) {
            *ys = T::zero();
        }

        // Temporary vectors reused across sweeps.
        let mut r = DenseVec::zeros(n_edges);
        let mut t_node = DenseVec::zeros(n_nodes);
        let mut s_node = DenseVec::zeros(n_nodes);
        let mut corr = DenseVec::zeros(n_edges);

        for _ in 0..self.smoother_sweeps {
            // ── r = x - A·y ────────────────────────────────────────────────
            self.a.spmv_add(T::one(), y.as_slice(), T::zero(), r.as_mut_slice());
            {
                let xs = x.as_slice();
                let rs = r.as_mut_slice();
                for i in 0..n_edges {
                    rs[i] = xs[i] - rs[i];
                }
            }

            // ── corr = ω·D⁻¹·r  (edge smoother) ────────────────────────────
            {
                let rs = r.as_slice();
                let cs = corr.as_mut_slice();
                for i in 0..n_edges {
                    cs[i] = self.scaled_inv_diag[i] * rs[i];
                }
            }

            // ── corr += G·P_v⁻¹·Gᵀ·r  (coarse auxiliary-space correction) ─
            self.g.apply_transpose(&r, &mut t_node);
            self.node_precond.apply_precond(&t_node, &mut s_node);
            self.g.spmv_add(
                T::one(),
                s_node.as_slice(),
                T::one(),
                corr.as_mut_slice(),
            );

            // ── y += corr ───────────────────────────────────────────────────
            {
                let cs = corr.as_slice();
                let ys = y.as_mut_slice();
                for i in 0..n_edges {
                    ys[i] = ys[i] + cs[i];
                }
            }
        }
    }
}

// ─── Shared helper ────────────────────────────────────────────────────────────

/// Build a boxed coarse-space solver from the given operator and strategy.
///
/// `pub(super)` so that `ads.rs` can call it without duplicating the match.
#[allow(clippy::type_complexity)]
pub(super) fn build_aux_solver<T: Scalar>(
    mat:    CsrMatrix<T>,
    solver: &AuxSpaceSolver,
) -> Result<(Box<dyn Preconditioner<Vector = DenseVec<T>>>, AuxSolverProfile), SolverError> {
    match solver {
        AuxSpaceSolver::Amg(cfg) => {
            let hier = AmgHierarchy::build(mat, cfg.clone());
            let profile = AuxSolverProfile::Amg(AuxAmgProfile {
                n_levels: hier.n_levels(),
                operator_complexity: hier.operator_complexity().max(1.0),
                grid_complexity: hier.grid_complexity().max(1.0),
            });
            Ok((Box::new(AmgPrecond::new(hier)), profile))
        }
        AuxSpaceSolver::Ilu0 => {
            let n = mat.nrows();
            let nnz = mat.nnz();
            let ilu = Ilu0Precond::from_csr(&mat).map_err(|e| {
                SolverError::PrecondSetupFailed {
                    reason: format!("AMS/ADS auxiliary ILU(0) setup failed: {e}"),
                }
            })?;
            Ok((Box::new(ilu), AuxSolverProfile::Ilu0 { n, nnz }))
        }
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::{CooMatrix, CsrMatrix};

    /// Build a 1-D chain graph (n_nodes nodes, n_edges = n_nodes-1 edges).
    /// Returns (G, A) where A = GGᵀ + delta*I.
    fn chain_graph(n_nodes: usize, delta: f64) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
        let n_edges = n_nodes - 1;
        let mut cg = CooMatrix::new(n_edges, n_nodes);
        for e in 0..n_edges {
            cg.push(e, e,     -1.0);
            cg.push(e, e + 1,  1.0);
        }
        let g = CsrMatrix::from_coo(&cg);
        // A = G Gᵀ + delta * I
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

    #[test]
    fn ams_rejects_nonsquare_a() {
        // A is 3×4 (non-square)
        let mut ca = CooMatrix::new(3, 4);
        ca.push(0, 0, 1.0_f64); ca.push(1, 1, 1.0); ca.push(2, 2, 1.0);
        let a = CsrMatrix::from_coo(&ca);
        let mut cg = CooMatrix::new(3, 2);
        cg.push(0, 0, -1.0_f64); cg.push(0, 1, 1.0);
        cg.push(1, 0, -1.0); cg.push(1, 1, 1.0);
        cg.push(2, 0, -1.0); cg.push(2, 1, 1.0);
        let g = CsrMatrix::from_coo(&cg);
        assert!(AmsPrecond::new(&a, &g, AmsConfig::default()).is_err());
    }

    #[test]
    fn ams_rejects_g_wrong_nrows() {
        // A is 4×4 but G has 3 rows
        let mut ca = CooMatrix::new(4, 4);
        for i in 0..4 { ca.push(i, i, 2.0_f64); }
        let a = CsrMatrix::from_coo(&ca);
        let mut cg = CooMatrix::new(3, 2);
        cg.push(0, 0, -1.0_f64); cg.push(0, 1, 1.0);
        cg.push(1, 0, -1.0); cg.push(1, 1, 1.0);
        cg.push(2, 0, -1.0); cg.push(2, 1, 1.0);
        let g = CsrMatrix::from_coo(&cg);
        assert!(AmsPrecond::new(&a, &g, AmsConfig::default()).is_err());
    }

    #[test]
    fn ams_rejects_near_zero_diagonal() {
        // A has a zero on the diagonal at row 1
        let mut ca = CooMatrix::new(2, 2);
        ca.push(0, 0, 2.0_f64);
        ca.push(1, 1, 0.0_f64); // zero diagonal
        let a = CsrMatrix::from_coo(&ca);
        let mut cg = CooMatrix::new(2, 3);
        cg.push(0, 0, -1.0_f64); cg.push(0, 1, 1.0);
        cg.push(1, 1, -1.0); cg.push(1, 2, 1.0);
        let g = CsrMatrix::from_coo(&cg);
        assert!(AmsPrecond::new(&a, &g, AmsConfig::default()).is_err());
    }

    #[test]
    fn ams_applies_chain_graph() {
        let (g, a) = chain_graph(6, 1e-3);
        let p = AmsPrecond::new(&a, &g, AmsConfig::default()).unwrap();
        let n = a.nrows();
        let x = DenseVec::from_vec(vec![1.0f64; n]);
        let mut y = DenseVec::zeros(n);
        p.apply_precond(&x, &mut y);
        let ys = y.as_slice();
        assert!(ys.iter().any(|&v| v.abs() > 1e-15), "output should be non-zero");
        assert!(ys.iter().all(|&v| v.is_finite()), "output should be finite");
    }

    #[test]
    fn ams_ilu0_node_solver() {
        let (g, a) = chain_graph(5, 0.1); // larger shift → non-singular GᵀAG
        let config = AmsConfig {
            node_solver: AuxSpaceSolver::Ilu0,
            ..Default::default()
        };
        let p = AmsPrecond::new(&a, &g, config).unwrap();
        let n = a.nrows();
        let x = DenseVec::from_vec(vec![1.0f64; n]);
        let mut y = DenseVec::zeros(n);
        p.apply_precond(&x, &mut y);
        assert!(y.as_slice().iter().any(|&v| v.abs() > 1e-15));
    }
}
