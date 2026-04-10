//! Auxiliary-space Divergence Solver (ADS) preconditioner.
//!
//! Implements the 3-term Hiptmair-Xu auxiliary-space preconditioner for
//! H(div) face-element discretisations (Darcy flow, mixed-form Maxwell):
//!
//! ```text
//! M_ADS⁻¹ x  ≈  ω D_A⁻¹ x  +  C · P_e⁻¹ · Cᵀ x  +  C G · P_v⁻¹ · Gᵀ Cᵀ x
//! ```
//!
//! where
//! - `D_A` is the diagonal of the face stiffness matrix `A`,
//! - `C`   is the discrete curl matrix (edges → faces, user-supplied),
//! - `G`   is the discrete gradient matrix (nodes → edges, user-supplied),
//! - `P_e` is an approximate solver for the edge Laplacian `CᵀAC`,
//! - `P_v` is an approximate solver for the node Laplacian `Gᵀ(CᵀAC)G`.
//!
//! The key algebraic identity `C G = 0` (curl-of-gradient is zero) holds
//! exactly for any valid discrete de Rham complex and is what makes the
//! auxiliary-space decomposition spectrally equivalent to `A`.
//!
//! ## Usage
//!
//! ```text
//! use linger::precond::{AdsPrecond, AdsConfig, AuxSpaceSolver};
//!
//! // C: discrete curl, n_faces × n_edges
//! // G: discrete gradient, n_edges × n_nodes
//! let config = AdsConfig::default();  // AMG for both coarse solves
//! let precond = AdsPrecond::new(&a_face, &c, &g, config)?;
//!
//! Gmres::new(30).solve(&a_face, Some(&precond), &b, &mut x, &params)?;
//! ```
//!
//! ## References
//!
//! Hiptmair, R. & Xu, J. (2007). Nodal auxiliary space preconditioning in
//! H(curl) and H(div) spaces. *SIAM J. Numer. Anal.*, 45(6), 2483–2509.

#![allow(clippy::needless_range_loop)]

use crate::core::{
    error::SolverError,
    operator::{LinearOperator, TransposeOperator},
    preconditioner::Preconditioner,
    scalar::Scalar,
    vector::DenseVec,
};
use crate::precond::ams::{AuxSolverProfile, AuxSpaceSolver, build_aux_solver};
use crate::sparse::CsrMatrix;

// ─── AdsConfig ───────────────────────────────────────────────────────────────

/// Configuration for [`AdsPrecond`].
#[derive(Debug, Clone)]
pub struct AdsConfig {
    /// Damping weight ω for the face-space Jacobi smoother.
    ///
    /// Typical value: 2/3 ≈ 0.667.
    pub smoother_omega: f64,
    /// Approximate solver for the edge Laplacian `CᵀAC`.
    pub edge_solver: AuxSpaceSolver,
    /// Approximate solver for the node Laplacian `Gᵀ(CᵀAC)G`.
    pub node_solver: AuxSpaceSolver,
}

impl Default for AdsConfig {
    fn default() -> Self {
        AdsConfig {
            smoother_omega: 0.667,
            edge_solver:    AuxSpaceSolver::default(),
            node_solver:    AuxSpaceSolver::default(),
        }
    }
}

impl AdsConfig {
    /// HPC-oriented default for H(div) auxiliary-space solves.
    pub fn hpc_default() -> Self {
        let amg = crate::amg::AmgConfig {
            coarse_threshold: 64,
            max_levels: 30,
            ..crate::amg::AmgConfig::default()
        };
        AdsConfig {
            smoother_omega: 0.667,
            edge_solver: AuxSpaceSolver::Amg(amg.clone()),
            node_solver: AuxSpaceSolver::Amg(amg),
        }
    }
}

/// Lightweight setup diagnostics for [`AdsPrecond`].
#[derive(Debug, Clone)]
pub struct AdsProfile {
    /// Number of face DOFs.
    pub n_faces: usize,
    /// Number of edge DOFs.
    pub n_edges: usize,
    /// Number of node DOFs.
    pub n_nodes: usize,
    /// Non-zeros in the fine operator `A`.
    pub a_nnz: usize,
    /// Non-zeros in the discrete curl `C`.
    pub c_nnz: usize,
    /// Non-zeros in the discrete gradient `G`.
    pub g_nnz: usize,
    /// Non-zeros in `A_edge = C^T A C`.
    pub a_edge_nnz: usize,
    /// Non-zeros in `A_node = G^T A_edge G`.
    pub a_node_nnz: usize,
    /// Auxiliary-space backend profile for the edge solve.
    pub edge_solver: AuxSolverProfile,
    /// Auxiliary-space backend profile for the node solve.
    pub node_solver: AuxSolverProfile,
}

// ─── AdsPrecond ──────────────────────────────────────────────────────────────

/// ADS preconditioner for H(div) face-element problems.
///
/// Constructed via [`AdsPrecond::new`]; implements [`Preconditioner`] and can
/// be passed directly to any [`KrylovSolver`](crate::KrylovSolver).
pub struct AdsPrecond<T: Scalar> {
    n_faces: usize,
    n_edges: usize,
    n_nodes: usize,
    /// Precomputed ω / d_i for each face i.
    scaled_inv_diag: Vec<T>,
    /// Discrete curl C: n_faces × n_edges.
    c: CsrMatrix<T>,
    /// Discrete gradient G: n_edges × n_nodes.
    g: CsrMatrix<T>,
    /// Approximate solver for the edge coarse problem CᵀAC.
    edge_precond: Box<dyn Preconditioner<Vector = DenseVec<T>>>,
    /// Approximate solver for the node coarse problem Gᵀ(CᵀAC)G.
    node_precond: Box<dyn Preconditioner<Vector = DenseVec<T>>>,
    /// Setup diagnostics for observability and tuning.
    profile: AdsProfile,
}

impl<T: Scalar> AdsPrecond<T> {
    /// Build the ADS preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a`      — Face stiffness matrix, square `n_faces × n_faces`.
    /// * `c`      — Discrete curl matrix, `n_faces × n_edges`.
    /// * `g`      — Discrete gradient matrix, `n_edges × n_nodes`.
    /// * `config` — Smoother weight and coarse-solver choices.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::PrecondSetupFailed`] if:
    /// - `a` is not square,
    /// - `c.nrows() ≠ a.nrows()`,
    /// - `g.nrows() ≠ c.ncols()` (n_edges mismatch),
    /// - a diagonal entry of `a` is near-zero,
    /// - either coarse-solver setup fails.
    pub fn new(
        a:      &CsrMatrix<T>,
        c:      &CsrMatrix<T>,
        g:      &CsrMatrix<T>,
        config: AdsConfig,
    ) -> Result<Self, SolverError> {
        let n_faces = a.nrows();
        let n_edges = c.ncols();
        let n_nodes = g.ncols();

        // ── 1. Dimension checks ──────────────────────────────────────────────
        if a.ncols() != n_faces {
            return Err(SolverError::PrecondSetupFailed {
                reason: format!("ADS: A must be square, got {}×{}", n_faces, a.ncols()),
            });
        }
        if c.nrows() != n_faces {
            return Err(SolverError::PrecondSetupFailed {
                reason: format!(
                    "ADS: C must have nrows = n_faces = {n_faces}, got {}",
                    c.nrows()
                ),
            });
        }
        if g.nrows() != n_edges {
            return Err(SolverError::PrecondSetupFailed {
                reason: format!(
                    "ADS: G must have nrows = n_edges = {n_edges}, got {}",
                    g.nrows()
                ),
            });
        }
        if n_nodes == 0 {
            return Err(SolverError::PrecondSetupFailed {
                reason: "ADS: G has zero columns (no node DOFs)".into(),
            });
        }

        // ── 2. Face smoother: ω / d_i ────────────────────────────────────────
        let omega = T::from_f64(config.smoother_omega);
        let tol   = T::machine_epsilon() * T::from_f64(1e6);
        let diag  = a.diag();
        let scaled_inv_diag: Vec<T> = diag
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d.abs() < tol {
                    Err(SolverError::PrecondSetupFailed {
                        reason: format!("ADS: near-zero diagonal in A at row {i}: {d:?}"),
                    })
                } else {
                    Ok(omega / d)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // ── 3. Edge coarse operator: A_edge = CᵀAC ──────────────────────────
        let c_t    = c.transpose_csr();       // n_edges × n_faces
        let ac     = a.matmat(c);             // n_faces × n_edges
        let a_edge = c_t.matmat(&ac);         // n_edges × n_edges

        // ── 4. Node coarse operator: A_node = Gᵀ A_edge G ───────────────────
        let g_t    = g.transpose_csr();       // n_nodes × n_edges
        let a_e_g  = a_edge.matmat(g);        // n_edges × n_nodes
        let a_node = g_t.matmat(&a_e_g);      // n_nodes × n_nodes

        // ── 5. Coarse solvers ─────────────────────────────────────────────────
        let a_edge_nnz = a_edge.nnz();
        let a_node_nnz = a_node.nnz();
        let (edge_precond, edge_solver_profile) = build_aux_solver(a_edge, &config.edge_solver)?;
        let (node_precond, node_solver_profile) = build_aux_solver(a_node, &config.node_solver)?;

        let profile = AdsProfile {
            n_faces,
            n_edges,
            n_nodes,
            a_nnz: a.nnz(),
            c_nnz: c.nnz(),
            g_nnz: g.nnz(),
            a_edge_nnz,
            a_node_nnz,
            edge_solver: edge_solver_profile,
            node_solver: node_solver_profile,
        };

        Ok(AdsPrecond {
            n_faces,
            n_edges,
            n_nodes,
            scaled_inv_diag,
            c: c.clone(),
            g: g.clone(),
            edge_precond,
            node_precond,
            profile,
        })
    }

    /// Setup-time profile for diagnostics and performance tuning.
    pub fn profile(&self) -> &AdsProfile { &self.profile }
}

impl<T: Scalar> Preconditioner for AdsPrecond<T> {
    type Vector = DenseVec<T>;

    /// Apply the 3-term ADS preconditioner:
    /// `y ← ω D_A⁻¹ x + C P_e⁻¹ Cᵀ x + C G P_v⁻¹ Gᵀ Cᵀ x`.
    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        // ── Term 1: face smoother  y[i] ← ω d_i⁻¹ x[i] ─────────────────────
        {
            let xs = x.as_slice();
            let ys = y.as_mut_slice();
            for i in 0..self.n_faces {
                ys[i] = self.scaled_inv_diag[i] * xs[i];
            }
        }

        // ── Shared: t_edge ← Cᵀ x  (reused by both Terms 2 and 3) ───────────
        let mut t_edge = DenseVec::zeros(self.n_edges);
        self.c.apply_transpose(x, &mut t_edge);

        // ── Term 2: curl correction  y += C P_e⁻¹ Cᵀ x ──────────────────────
        let mut s_edge = DenseVec::zeros(self.n_edges);
        self.edge_precond.apply_precond(&t_edge, &mut s_edge);

        let mut e_face = DenseVec::zeros(self.n_faces);
        self.c.apply(&s_edge, &mut e_face);

        {
            let ys = y.as_mut_slice();
            let es = e_face.as_slice();
            for i in 0..self.n_faces { ys[i] += es[i]; }
        }

        // ── Term 3: gradient correction  y += C G P_v⁻¹ Gᵀ Cᵀ x ─────────────
        // Reuse t_edge (= Cᵀ x) as input for the node path.
        let mut t_node = DenseVec::zeros(self.n_nodes);
        self.g.apply_transpose(&t_edge, &mut t_node);   // Gᵀ (Cᵀ x)

        let mut s_node = DenseVec::zeros(self.n_nodes);
        self.node_precond.apply_precond(&t_node, &mut s_node);

        let mut f_edge = DenseVec::zeros(self.n_edges);
        self.g.apply(&s_node, &mut f_edge);              // G s_node

        let mut e2_face = DenseVec::zeros(self.n_faces);
        self.c.apply(&f_edge, &mut e2_face);             // C G s_node

        {
            let ys  = y.as_mut_slice();
            let e2s = e2_face.as_slice();
            for i in 0..self.n_faces { ys[i] += e2s[i]; }
        }
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::{CooMatrix, CsrMatrix};

    /// Build a 2-D rectangular complex: nx×ny nodes.
    /// Returns (G: n_edges×n_nodes, C: n_faces×n_edges, A = CCᵀ+delta*I).
    ///
    /// Node numbering: node(i,j) = i*(ny) + j,  i in 0..nx, j in 0..ny.
    /// Horizontal edge h(i,j): node(i,j)→node(i,j+1),  index = i*(ny-1)+j.
    /// Vertical   edge v(i,j): node(i,j)→node(i+1,j),  index = (nx-1)*ny + i*ny+j.
    /// Face f(i,j): for i in 0..nx-1, j in 0..ny-1, clockwise boundary.
    fn rect_complex(nx: usize, ny: usize, delta: f64)
        -> (CsrMatrix<f64>, CsrMatrix<f64>, CsrMatrix<f64>)
    {
        let n_nodes = nx * ny;
        let n_h     = (nx) * (ny - 1);  // horizontal edges
        let n_v     = (nx - 1) * ny;    // vertical edges
        let n_edges = n_h + n_v;
        let n_faces = (nx - 1) * (ny - 1);

        let node = |i: usize, j: usize| i * ny + j;
        let h_edge = |i: usize, j: usize| i * (ny - 1) + j;
        let v_edge = |i: usize, j: usize| n_h + i * ny + j;

        // G: gradient (n_edges × n_nodes)
        let mut cg = CooMatrix::new(n_edges, n_nodes);
        for i in 0..nx {
            for j in 0..(ny - 1) {
                cg.push(h_edge(i, j), node(i, j),     -1.0);
                cg.push(h_edge(i, j), node(i, j + 1),  1.0);
            }
        }
        for i in 0..(nx - 1) {
            for j in 0..ny {
                cg.push(v_edge(i, j), node(i, j),       -1.0);
                cg.push(v_edge(i, j), node(i + 1, j),    1.0);
            }
        }
        let g = CsrMatrix::from_coo(&cg);

        // C: curl (n_faces × n_edges), each face bounded by 4 edges
        // Face f(i,j) corners: node(i,j), node(i,j+1), node(i+1,j+1), node(i+1,j)
        // Positive orientation: bottom(+), right(+), top(-), left(-)
        let mut cc = CooMatrix::new(n_faces, n_edges);
        let face = |i: usize, j: usize| i * (ny - 1) + j;
        for i in 0..(nx - 1) {
            for j in 0..(ny - 1) {
                let f = face(i, j);
                cc.push(f, h_edge(i,     j    ),  1.0); // bottom
                cc.push(f, v_edge(i,     j + 1),  1.0); // right
                cc.push(f, h_edge(i + 1, j    ), -1.0); // top (reversed)
                cc.push(f, v_edge(i,     j    ), -1.0); // left (reversed)
            }
        }
        let c = CsrMatrix::from_coo(&cc);

        // A = C Cᵀ + delta * I
        let c_t  = c.transpose_csr();
        let cc_t = c.matmat(&c_t);
        let mut ca = CooMatrix::new(n_faces, n_faces);
        for (i, j, v) in cc_t.triplets() {
            ca.push(i, j, v);
        }
        for i in 0..n_faces {
            ca.push(i, i, delta);
        }
        let a = CsrMatrix::from_coo(&ca);
        (g, c, a)
    }

    #[test]
    fn ads_rejects_nonsquare_a() {
        let mut ca = CooMatrix::new(3, 4);
        ca.push(0, 0, 1.0_f64); ca.push(1, 1, 1.0); ca.push(2, 2, 1.0);
        let a = CsrMatrix::from_coo(&ca);
        let mut cc = CooMatrix::new(3, 2);
        cc.push(0, 0, 1.0_f64); cc.push(1, 1, 1.0); cc.push(2, 0, 1.0);
        let c = CsrMatrix::from_coo(&cc);
        let mut cg = CooMatrix::new(2, 3);
        cg.push(0, 0, -1.0_f64); cg.push(0, 1, 1.0);
        cg.push(1, 1, -1.0); cg.push(1, 2, 1.0);
        let g = CsrMatrix::from_coo(&cg);
        assert!(AdsPrecond::new(&a, &c, &g, AdsConfig::default()).is_err());
    }

    #[test]
    fn ads_rejects_c_wrong_nrows() {
        // A is 4×4, C has 3 rows (should be 4)
        let mut ca = CooMatrix::new(4, 4);
        for i in 0..4 { ca.push(i, i, 2.0_f64); }
        let a = CsrMatrix::from_coo(&ca);
        let mut cc = CooMatrix::new(3, 2);
        cc.push(0, 0, 1.0_f64); cc.push(1, 0, -1.0); cc.push(2, 1, 1.0);
        let c = CsrMatrix::from_coo(&cc);
        let mut cg = CooMatrix::new(2, 3);
        cg.push(0, 0, -1.0_f64); cg.push(0, 1, 1.0);
        cg.push(1, 1, -1.0); cg.push(1, 2, 1.0);
        let g = CsrMatrix::from_coo(&cg);
        assert!(AdsPrecond::new(&a, &c, &g, AdsConfig::default()).is_err());
    }

    #[test]
    fn ads_rejects_g_wrong_nrows() {
        // A is 4×4, C is 4×3, G has 2 rows (should be 3 = n_edges)
        let mut ca = CooMatrix::new(4, 4);
        for i in 0..4 { ca.push(i, i, 2.0_f64); }
        let a = CsrMatrix::from_coo(&ca);
        let mut cc = CooMatrix::new(4, 3);
        cc.push(0, 0, 1.0_f64); cc.push(1, 1, 1.0); cc.push(2, 2, 1.0); cc.push(3, 0, -1.0);
        let c = CsrMatrix::from_coo(&cc);
        let mut cg = CooMatrix::new(2, 3); // wrong: should be 3 rows
        cg.push(0, 0, -1.0_f64); cg.push(0, 1, 1.0);
        cg.push(1, 1, -1.0); cg.push(1, 2, 1.0);
        let g = CsrMatrix::from_coo(&cg);
        assert!(AdsPrecond::new(&a, &c, &g, AdsConfig::default()).is_err());
    }

    #[test]
    fn ads_applies_nontrivially() {
        let (_g, c, a) = rect_complex(3, 3, 1e-3);
        // Build G for the 3×3 grid (same helper)
        let (g, _, _) = rect_complex(3, 3, 1e-3);
        let p = AdsPrecond::new(&a, &c, &g, AdsConfig::default()).unwrap();
        let n = a.nrows();
        let x = DenseVec::from_vec(vec![1.0f64; n]);
        let mut y = DenseVec::zeros(n);
        p.apply_precond(&x, &mut y);
        let ys = y.as_slice();
        assert!(ys.iter().any(|&v| v.abs() > 1e-15), "output should be non-zero");
        assert!(ys.iter().all(|&v| v.is_finite()), "output should be finite");
    }

    #[test]
    fn ads_curl_of_gradient_is_zero() {
        // Verify that the discrete complex satisfies C G = 0 (algebraic identity).
        // This is a sanity check on the test geometry, not on AdsPrecond itself.
        let (g, c, _a) = rect_complex(3, 3, 1e-3);
        let cg = c.matmat(&g);
        for (_, _, v) in cg.triplets() {
            assert!(v.abs() < 1e-12, "C*G should be zero, got {v}");
        }
    }
}
