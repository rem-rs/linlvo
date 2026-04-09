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
    operator::{LinearOperator, TransposeOperator},
    preconditioner::Preconditioner,
    scalar::Scalar,
    vector::DenseVec,
};
use crate::precond::ilu0::Ilu0Precond;
use crate::sparse::CsrMatrix;

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
    /// Approximate solver for the nodal Laplacian `GᵀAG`.
    pub node_solver: AuxSpaceSolver,
}

impl Default for AmsConfig {
    fn default() -> Self {
        AmsConfig {
            smoother_omega: 0.667,
            node_solver: AuxSpaceSolver::default(),
        }
    }
}

// ─── AmsPrecond ──────────────────────────────────────────────────────────────

/// AMS preconditioner for H(curl) edge-element Maxwell problems.
///
/// Constructed via [`AmsPrecond::new`]; implements [`Preconditioner`] and can
/// be passed directly to any [`KrylovSolver`](crate::KrylovSolver).
pub struct AmsPrecond<T: Scalar> {
    n_edges: usize,
    n_nodes: usize,
    /// Precomputed ω / d_i for each edge i (avoids division in apply).
    scaled_inv_diag: Vec<T>,
    /// Discrete gradient G: n_edges × n_nodes (column-sparse in practice).
    g: CsrMatrix<T>,
    /// Approximate solver for the nodal coarse problem GᵀAG.
    node_precond: Box<dyn Preconditioner<Vector = DenseVec<T>>>,
}

impl<T: Scalar> AmsPrecond<T> {
    /// Build the AMS preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a`      — Edge stiffness matrix, square `n_edges × n_edges`.
    /// * `g`      — Discrete gradient matrix, `n_edges × n_nodes`.
    ///              Each row has exactly two non-zeros: −1 at the tail node
    ///              and +1 at the head node (standard FE convention).
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
        // Pattern mirrors amg/setup.rs: r.matmat(&a_now.matmat(&p))
        let g_t    = g.transpose_csr();   // n_nodes × n_edges
        let ag     = a.matmat(g);         // n_edges × n_nodes
        let a_node = g_t.matmat(&ag);     // n_nodes × n_nodes

        // ── 4. Coarse solver ─────────────────────────────────────────────────
        let node_precond = build_aux_solver(a_node, &config.node_solver)?;

        Ok(AmsPrecond {
            n_edges,
            n_nodes,
            scaled_inv_diag,
            g: g.clone(),
            node_precond,
        })
    }
}

impl<T: Scalar> Preconditioner for AmsPrecond<T> {
    type Vector = DenseVec<T>;

    /// Apply the 2-term AMS preconditioner: `y ← ω D_A⁻¹ x + G P_v⁻¹ Gᵀ x`.
    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        // ── Term 1: edge smoother  y[i] ← ω d_i⁻¹ x[i] ─────────────────────
        {
            let xs = x.as_slice();
            let ys = y.as_mut_slice();
            for i in 0..self.n_edges {
                ys[i] = self.scaled_inv_diag[i] * xs[i];
            }
        }

        // ── Term 2: gradient auxiliary-space correction ───────────────────────
        // 2a. t_node ← Gᵀ x
        let mut t_node = DenseVec::zeros(self.n_nodes);
        self.g.apply_transpose(x, &mut t_node);

        // 2b. s_node ← P_v⁻¹ t_node
        let mut s_node = DenseVec::zeros(self.n_nodes);
        self.node_precond.apply_precond(&t_node, &mut s_node);

        // 2c. e_edge ← G s_node
        let mut e_edge = DenseVec::zeros(self.n_edges);
        self.g.apply(&s_node, &mut e_edge);

        // 2d. y += e_edge
        {
            let ys = y.as_mut_slice();
            let es = e_edge.as_slice();
            for i in 0..self.n_edges {
                ys[i] += es[i];
            }
        }
    }
}

// ─── Shared helper ────────────────────────────────────────────────────────────

/// Build a boxed coarse-space solver from the given operator and strategy.
///
/// `pub(super)` so that `ads.rs` can call it without duplicating the match.
pub(super) fn build_aux_solver<T: Scalar>(
    mat:    CsrMatrix<T>,
    solver: &AuxSpaceSolver,
) -> Result<Box<dyn Preconditioner<Vector = DenseVec<T>>>, SolverError> {
    match solver {
        AuxSpaceSolver::Amg(cfg) => {
            let hier = AmgHierarchy::build(mat, cfg.clone());
            Ok(Box::new(AmgPrecond::new(hier)))
        }
        AuxSpaceSolver::Ilu0 => {
            let ilu = Ilu0Precond::from_csr(&mat).map_err(|e| {
                SolverError::PrecondSetupFailed {
                    reason: format!("AMS/ADS auxiliary ILU(0) setup failed: {e}"),
                }
            })?;
            Ok(Box::new(ilu))
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
