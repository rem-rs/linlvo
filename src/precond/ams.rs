//! Auxiliary-space Maxwell Solver (AMS) preconditioner.
//!
//! Implements the 2-term Hiptmair-Xu auxiliary-space preconditioner for
//! H(curl) edge-element discretisations of Maxwell-type problems:
//!
//! ```text
//! M_AMS⁻¹ x  ≈  S_A⁻¹ x  +  G · P_v⁻¹ · Gᵀ x
//! ```
//!
//! where
//! - `S_A⁻¹` is an approximate inverse of the edge stiffness matrix `A`
//!   (weighted Jacobi `ω D_A⁻¹` or symmetric Gauss-Seidel),
//! - `G`   is the discrete gradient matrix (nodes → edges, user-supplied),
//! - `P_v` is an approximate solver for the nodal Laplacian `GᵀAG`.
//!
//! The coarse nodal solve `P_v` can be either AMG (recommended for large
//! problems) or ILU(0) (suitable for small/medium problems).
//!
//! ## Usage
//!
//! ```text
//! use linlvo::precond::{AmsPrecond, AmsConfig, AuxSpaceSolver};
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

use crate::amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy};
use crate::core::vector::Vector as _;
use crate::core::{
    error::SolverError,
    operator::TransposeOperator,
    preconditioner::Preconditioner,
    scalar::{ComplexScalar, Scalar},
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
    /// Edge-space smoother variant (MFEM HypreAMS uses symmetric
    /// Gauss-Seidel with weight 1.0 by default).
    pub edge_smoother: AmsEdgeSmoother,
    /// Two-level cycle structure (additive Hiptmair-Xu or multiplicative
    /// V(1,1); HYPRE AMS default is the V(1,1) cycle).
    pub cycle: AmsCycle,
    /// Enable the 3-D face (curl) auxiliary space `Pi = [Pi_x, Pi_y, Pi_z]`
    /// (HYPRE AMS's `HYPRE_AMSSetInterpolations`).
    ///
    /// Requires vertex coordinates (via [`AmsPrecond::with_coords`]); when
    /// enabled together with [`AmsCycle::MultiplicativeV11`] the cycle becomes
    /// the full 9-step HYPRE AMS `cycle_type = 13` structure
    /// `GS → Pi_x → Pi_y → Pi_z → nodal → Pi_z → Pi_y → Pi_x → GS`.
    /// Default: `false` (two-space AMS, as before).
    pub face_space: bool,
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
    /// Declare the edge system itself singular — a curl-curl problem with no
    /// mass term and no essential BCs, i.e. exactly the system
    /// `HypreAMS::SetSingularProblem()` declares (MFEM `linalg/hypre.hpp:2047`
    /// = `HYPRE_AMSSetBetaPoissonMatrix(ams, NULL)`: AMS builds the nodal
    /// `GᵀAG` itself and solves the *unshifted* singular A).
    ///
    /// When `true`, a zero (or near-zero) diagonal entry of `A` no longer
    /// fails the setup: its Jacobi scale factor is set to zero — the row gets
    /// no Jacobi edge smoothing, mirroring hypre's relaxation which skips
    /// zero-pivot rows (the symmetric Gauss-Seidel arm already does).
    /// PCG remains responsible for kernel compatibility: with a right-hand
    /// side in `range(A)` the Krylov iterates stay in `range(A)` and the
    /// convergent representative is the min-norm one — no `δI` shift on `A`
    /// is needed or wanted (a shift perturbs the solved system by `O(δ)`).
    ///
    /// **Cycle effect** (hypre 2.28 `ams.c:3688`, `hypre_AMSSolve`): a singular
    /// problem switches the block-Pi multiplicative cycle from the
    /// three-space `034515430` (GS → Pi_x → Pi_y → Pi_z → **nodal** → Pi_z →
    /// Pi_y → Pi_x → GS) to `0345430` — the nodal gradient correction is
    /// **dropped entirely** (hypre does not even build `B_G` when
    /// `beta_is_zero`, `ams.c:3144`).  The flag only changes the cycle when
    /// face blocks are present (constructor [`AmsPrecond::with_pi`]); the
    /// historical two-space cycle keeps its nodal correction unconditionally.
    ///
    /// Default `false` (every previous consumer keeps the strict setup).
    pub singular_problem: bool,
}

impl Default for AmsConfig {
    fn default() -> Self {
        AmsConfig {
            smoother_omega: 0.667,
            smoother_sweeps: 1,
            // HYPRE AMS default `rlx_type = 2` (symmetric Gauss-Seidel,
            // `rlx_weight = 1.0`).  Weighted Jacobi with ω = 2/3 diverges on
            // curl-curl operators (ρ(D⁻¹A) ≫ 3), which makes the additive /
            // V(1,1) cycle indefinite and breaks preconditioned CG.
            edge_smoother: AmsEdgeSmoother::SymmetricGaussSeidel,
            // HYPRE AMS default `cycle_type = 13` (symmetric multiplicative
            // V(1,1) cycle: GS → nodal → GS).
            cycle: AmsCycle::MultiplicativeV11,
            face_space: false,
            node_solver: AuxSpaceSolver::default(),
            // Shift the nodal operator GᵀAG away from singularity: with
            // Dirichlet-type BCs the boundary-node rows of GᵀAG are (nearly)
            // zero, and the AMG ω·D⁻¹ smoother then amplifies the nullspace
            // into an indefinite coarse correction, breaking PCG.
            singularity_regularization: 1e-6,
            // `A` is assumed nonsingular unless the caller declares the
            // curl-curl singular problem (see [`AmsConfig::singular_problem`]).
            singular_problem: false,
        }
    }
}

/// Edge-space smoother for the AMS/ADS auxiliary-space preconditioner.
#[derive(Debug, Clone)]
pub enum AmsEdgeSmoother {
    /// Weighted Jacobi `ω·D⁻¹` (classical Hiptmair-Xu form).
    WeightedJacobi,
    /// Symmetric Gauss-Seidel (forward + backward sweep) — the default of
    /// HYPRE AMS (`rlx_type = 2`, `rlx_weight = 1.0`).
    SymmetricGaussSeidel,
    /// hypre AMS `rlx_type = 2` semantics (ams.c `hypre_ParCSRRelax`): the
    /// "offd-l1-scaled" symmetric Gauss-Seidel — for relax types 1–4 hypre
    /// AMSSetup computes the **row l1 norms of A** (ams.c:3041-3053) and the
    /// hybrid-SOR relax divides by them, NOT by the diagonal: each update is
    /// `x_i += (b_i - Σ_j a_ij x_j)/‖row_i‖₁`.  On singular curl-curl systems
    /// rows with ‖row‖₁ >> |a_ii| make the plain (diag-scaled) SGS arm
    /// amplify; the l1 scaling stays bounded (hypre-faithful robustness).
    L1ScaledSymmetricGaussSeidel,
}

impl Default for AmsEdgeSmoother {
    fn default() -> Self {
        Self::WeightedJacobi
    }
}

/// Two-level cycle structure for the AMS preconditioner.
#[derive(Debug, Clone)]
pub enum AmsCycle {
    /// Additive Hiptmair-Xu form `S⁻¹ + G·P_v⁻¹·Gᵀ`, iterated `smoother_sweeps`
    /// times (Richardson).
    Additive,
    /// Symmetric multiplicative V(1,1) cycle: pre-GS → nodal coarse
    /// correction → post-GS (HYPRE AMS default `cycle_type = 13`).
    MultiplicativeV11,
}

impl Default for AmsCycle {
    fn default() -> Self {
        Self::Additive
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
            edge_smoother: AmsEdgeSmoother::WeightedJacobi,
            cycle: AmsCycle::Additive,
            face_space: false,
            node_solver: AuxSpaceSolver::Amg(AmgConfig {
                coarse_threshold: 64,
                max_levels: 30,
                ..AmgConfig::default()
            }),
            singularity_regularization: 0.0,
            singular_problem: false,
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
/// # Multi-sweep smoothing (additive)
///
/// With [`AmsCycle::Additive`] and `smoother_sweeps > 1`, the preconditioner
/// applies `smoother_sweeps` sweeps of a preconditioned Richardson iteration:
///
/// ```text
/// y⁰ = 0
/// for l = 1…K:
///   rˡ = x - A·yˡ⁻¹
///   yˡ = yˡ⁻¹ + S⁻¹·rˡ  +  G·P_v⁻¹·Gᵀ·rˡ
/// y = yᵏ
/// ```
///
/// More sweeps improve h-independence and robustness for Maxwell eigenvalue
/// problems at the cost of additional SpMV per preconditioner application.
///
/// # Multiplicative V(1,1) cycle
///
/// With [`AmsCycle::MultiplicativeV11`] the preconditioner applies the
/// standard symmetric two-level cycle (pre-GS → nodal coarse correction →
/// post-GS), matching HYPRE AMS's default `cycle_type = 13` structure:
///
/// ```text
/// y ← S⁻¹·x                (one symmetric GS sweep from zero)
/// r = x - A·y
/// y += G·P_v⁻¹·Gᵀ·r
/// r = x - A·y
/// y += S⁻¹·r               (one symmetric GS sweep from zero)
/// ```
pub struct AmsPrecond<T: ComplexScalar> {
    n_edges: usize,
    n_nodes: usize,
    /// Edge stiffness matrix A (stored for multi-sweep residual).
    a: CsrMatrix<T>,
    /// Edge-space smoother variant (Jacobi ω·D⁻¹ or symmetric GS).
    edge_smoother: AmsEdgeSmoother,
    /// Cycle structure (additive Hiptmair-Xu or multiplicative V(1,1)).
    cycle: AmsCycle,
    /// Precomputed ω / d_i for each edge i (avoids division in apply).
    scaled_inv_diag: Vec<T>,
    /// Discrete gradient G: n_edges × n_nodes (column-sparse in practice).
    g: CsrMatrix<T>,
    /// Number of smoother sweeps to apply.
    smoother_sweeps: usize,
    /// Approximate solver for the nodal coarse problem GᵀAG.
    node_precond: Box<dyn Preconditioner<Vector = DenseVec<T>>>,
    /// Face (curl) auxiliary space: `(Pi_d, B_d)` for d = x, y, z, where
    /// `B_d ≈ (Pi_dᵀ A Pi_d)⁻¹`.  Empty when the face space is off.
    face_blocks: Vec<(CsrMatrix<T>, Box<dyn Preconditioner<Vector = DenseVec<T>>>)>,
    /// Whether the edge system was declared singular
    /// ([`AmsConfig::singular_problem`]): with face blocks present this drops
    /// the nodal correction from the cycle (hypre `beta_is_zero` → `0345430`).
    singular_problem: bool,
    /// Setup diagnostics for observability and tuning.
    profile: AmsProfile,
}

impl<T: ComplexScalar> AmsPrecond<T> {
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
        Self::build(a, g, None, None, config)
    }

    /// Build the AMS preconditioner with **user-supplied** face
    /// (curl) interpolation blocks, the analogue of hypre's
    /// `HYPRE_AMSSetInterpolations(ams, Pi_x, Pi_y, Pi_z)`.
    ///
    /// This is the mechanism MFEM's `HypreAMS` uses for every non-trivial
    /// space: when the edge space is higher order or the mesh is curved,
    /// `HypreAMS::MakeGradientAndInterpolation` (MFEM `linalg/hypre.cpp`)
    /// assembles the identity interpolator `id_ND : [H¹]³ → H(curl)` and hands
    /// its three component blocks to hypre — hypre then runs the block-Pi
    /// multiplicative cycle (`cycle_type = 13`: `034515430`, or `0345430`
    /// with [`AmsConfig::singular_problem`]) with `B_Pi_d = AMG(Pi_dᵀ A Pi_d)`
    /// on each block.  Unlike [`Self::with_coords`] (hypre's internal
    /// lowest-order coordinate construction, only valid for straight first-
    /// order meshes), the caller supplies the FE-exact interpolations.
    ///
    /// `pi` holds 1–3 rectangular matrices `Pi_d` of shape
    /// `n_edges × n_face_dofs` (columns in node-based order).  Zero rows of
    /// the assembled coarse operators `Pi_dᵀ A Pi_d` are fixed to unit
    /// diagonal, matching hypre's `hypre_ParCSRMatrixFixZeroRows`.  The
    /// [`AmsConfig::face_space`] flag (coordinate construction) is ignored by
    /// this constructor.
    ///
    /// See [`Self::new`] for the common arguments and error conditions.
    pub fn with_pi(
        a:      &CsrMatrix<T>,
        g:      &CsrMatrix<T>,
        pi:     &[CsrMatrix<T>],
        config: AmsConfig,
    ) -> Result<Self, SolverError> {
        if pi.is_empty() || pi.len() > 3 {
            return Err(SolverError::PrecondSetupFailed {
                reason: format!("AMS: with_pi expects 1–3 Pi blocks, got {}", pi.len()),
            });
        }
        Self::build(a, g, None, Some(pi), config)
    }

    /// Build the AMS preconditioner with the 3-D face (curl) auxiliary space.
    ///
    /// `coords` are the physical coordinates of the `n_nodes` H¹ vertices,
    /// row-major `[x, y, z, ...]` (length `3·n_nodes`; 2-D layouts of length
    /// `2·n_nodes` disable the face space).  With `AmsConfig::face_space` and
    /// [`AmsCycle::MultiplicativeV11`] this reproduces HYPRE AMS's full
    /// `cycle_type = 13` structure (three Pi blocks + nodal space).
    ///
    /// See [`Self::new`] for the common arguments and error conditions.
    pub fn with_coords(
        a:      &CsrMatrix<T>,
        g:      &CsrMatrix<T>,
        coords: &[f64],
        config: AmsConfig,
    ) -> Result<Self, SolverError> {
        Self::build(a, g, Some(coords), None, config)
    }

    fn build(
        a:      &CsrMatrix<T>,
        g:      &CsrMatrix<T>,
        coords: Option<&[f64]>,
        pi:     Option<&[CsrMatrix<T>]>,
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
        // `singular_problem` (MFEM `HypreAMS::SetSingularProblem` =
        // `HYPRE_AMSSetBetaPoissonMatrix(NULL)`, linalg/hypre.hpp:2047) declares
        // the unshifted curl-curl system: zero-pivot rows get a zero Jacobi
        // scale (no edge smoothing on that row — hypre's relaxation skips
        // them; the symmetric Gauss-Seidel arm already does) instead of
        // failing the setup.  Without the flag the strict check stands.
        let omega = T::from_real(<T::Real as Scalar>::from_f64(config.smoother_omega));
        let tol   = T::machine_epsilon() * <T::Real as Scalar>::from_f64(1e6);
        let diag  = a.diag();
        let scaled_inv_diag: Vec<T> = diag
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d.abs() < tol {
                    if config.singular_problem {
                        Ok(T::zero())
                    } else {
                        Err(SolverError::PrecondSetupFailed {
                            reason: format!("AMS: near-zero diagonal in A at row {i}: {d:?}"),
                        })
                    }
                } else {
                    Ok(omega / d)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // ── 3. Coarse operator: A_node = GᵀAG ───────────────────────────────
        let g_t    = g.transpose_csr();   // n_nodes × n_edges
        let ag     = a.matmat(g);         // n_edges × n_nodes
        let mut a_node = g_t.matmat(&ag);     // n_nodes × n_nodes

        // When `singularity_regularization > 0`, add ε·I to the nodal
        // system to shift *all* eigenvalues away from zero — in particular
        // the exact nullspace of GᵀAG (the constant vector, since G·1 = 0,
        // and curl-curl vanishes on gradients so GᵀAG ≈ δ·GᵀMG there).
        // Note that adding ε·GᵀG (the previous implementation) does *not*
        // regularize that mode because GᵀG·1 = 0 as well; the singular
        // nodal AMG then amplifies any nullspace component of Gᵀr into a
        // huge coarse correction (observed on one rank of pex34 np2 -hex).
        // The constant component of the coarse solution is harmless: it is
        // annihilated by the left multiplication with G in the correction.
        let eps_f64 = config.singularity_regularization;
        if eps_f64 > 0.0 {
            let eps = T::from_f64(eps_f64);
            // Merge a_node + ε·I via COO (diagonal entries sum up).
            let mut coo = CooMatrix::new(n_nodes, n_nodes);
            for (r, c, v) in a_node.triplets() {
                coo.push(r, c, v);
            }
            for d in 0..n_nodes {
                coo.push(d, d, eps);
            }
            a_node = CsrMatrix::from_coo(&coo);
        }

        // ── 4. Coarse solver ─────────────────────────────────────────────────
        // On the `with_pi` path (the MFEM HypreAMS structure) the coarse
        // operators are generally SINGULAR: hypre deliberately avoids an
        // exact coarsest solve there ("Generally, don't use exact solve on
        // the coarsest level (matrix may be singular)", linalg/hypre.cpp —
        // `SetCycleRelaxType(amg_rlx_type, 3)` for both B_G and B_Pi).  An
        // LU-based coarsest solve amplifies the near-nullspace by 1/ε and
        // blows up PCG (measured: tesla `-cr` order-1).  Force
        // relaxation-based coarsest solves for the user-Pi path unless the
        // caller already chose one.
        let mut node_solver = config.node_solver.clone();
        if pi.is_some() {
            if let AuxSpaceSolver::Amg(cfg) = &mut node_solver {
                if cfg.coarsest_sweeps.is_none() {
                    cfg.coarsest_sweeps = Some(1);
                }
            }
        }
        // The Pi-block AMG (`B_Pi_d ≈ (Pi_dᵀ A Pi_d)⁻¹`) runs the hypre
        // BoomerAMG configuration MFEM 4.10 `HypreAMS::MakeSolver` sets for
        // every B_Pi: coarsen_type 10 (HMIS) + aggressive second shot
        // (`agg_num_levels = 1`, `Create2ndS` + `CoarsenHMIS(S2, measure+3)`)
        // + interp_type 6 (extended-i, PMax 4) — [`CoarsenStrategy::HmisAms`].
        // The Galerkin operators `A_Pi_d = Pi_dᵀ A Pi_d` carry the curl-curl
        // near-nullspace (on straight meshes `Pi_d·x̂_d` is a DISCRETE
        // gradient, so λ_min(A_Pi) ~ round-off); the historical SA-era guard
        // (`coarse_threshold ≥ 64`, D957) existed because deep aggregation
        // collapses such operators to a 1×1 coarsest whose single entry IS
        // the near-zero eigenvalue (measured, tesla inline-hex o2: level-8
        // 1×1 entry 2.18e-9 → each Pi arm amplifies by ~5e8).  The HMIS
        // aggressive hierarchy is shallow by construction — that is exactly
        // what hypre's second-shot coarsening buys — so hypre's own bounds
        // apply: MinCoarseSize 2 (ams.c:3226), MaxLevels 25 (ams.c:3215).
        let mut pi_solver = node_solver.clone();
        if pi.is_some() {
            if let AuxSpaceSolver::Amg(cfg) = &mut pi_solver {
                cfg.strategy = CoarsenStrategy::HmisAms;
                cfg.coarse_threshold = 2;
                cfg.max_levels = 25;
                // hypre BoomerAMG relax type 8 ("l1" SSOR) computes its scale
                // factors with hypre_ParCSRComputeL1Norms OPTION 4
                // (par_amg_setup.c:3280, truncation per Remark 6.2,
                // ams.c:678-692): l1_i starts at |a_ii| and the off-diagonal
                // contributions only come from the processor-offd block —
                // on ONE RANK the result degenerates to the (sign-fixed)
                // diagonal, i.e. plain symmetric Gauss-Seidel
                // (par_amg_setup.c:3265-3285 picks option 4 for coarsest
                // type-8 as well).  The earlier "option 1, full row sum"
                // reading (r108 ledger) was wrong: option 1 is the AMS-scope
                // helper, not the BoomerAMG relax-8 path.
                cfg.smoother = crate::amg::SmootherType::SymmetricGaussSeidel;
            }
        }
        let a_node_nnz = a_node.nnz();
        let (node_precond, node_solver_profile) = build_aux_solver(a_node, &node_solver)?;

        // ── 5. Face (curl) auxiliary space: Pi = [Pi_x, Pi_y, Pi_z] ─────────
        // Two construction paths:
        // * `with_pi` — user-supplied FE-exact interpolation blocks (the MFEM
        //   `HYPRE_AMSSetInterpolations` path).  Coarse operator per block:
        //   `A_Pid = Pi_dᵀ·A·Pi_d` with zero rows fixed to unit diagonal
        //   (hypre `hypre_ParCSRMatrixFixZeroRows`, ams.c:3272).
        // * `with_coords` + `face_space` — hypre's internal lowest-order
        //   construction `Pi_d(e, v) = 0.5·|G(e,v)|·(Gᵀx_d)[e]`.
        let mut face_blocks: Vec<(CsrMatrix<T>, Box<dyn Preconditioner<Vector = DenseVec<T>>>)> =
            Vec::new();
        if let Some(pi) = pi {
            for (d, pid) in pi.iter().enumerate() {
                if pid.nrows() != n_edges {
                    return Err(SolverError::PrecondSetupFailed {
                        reason: format!(
                            "AMS: Pi block {d} must have nrows = n_edges = {n_edges}, got {}",
                            pid.nrows()
                        ),
                    });
                }
                let a_pid = crate::amg::hmis::rap_hypre_order(pid, a);
                // Zero rows → unit diagonal (hypre `FixZeroRows`); NO ε·I
                // shift — hypre's BoomerAMG runs unshifted on the (generally
                // singular) A_Pi, and the earlier measured shift here made
                // things *worse*: a 10⁻⁶ eigenvalue turns the relaxation
                // coarsest solve into a 10⁺⁶-amplified near-null mode
                // (tesla `-cr` order-1 divergence).  The relaxation-based
                // coarsest solve (`coarsest_sweeps`) is the hypre-faithful
                // nullspace handling.
                let a_pid = fix_zero_rows(&a_pid);
                let (b_pid, _) = build_aux_solver(a_pid, &pi_solver)?;
                face_blocks.push((pid.clone(), b_pid));
            }
        } else if config.face_space {
            let dim = coords.map(|c| c.len() / n_nodes).unwrap_or(0);
            let coords = coords.ok_or_else(|| SolverError::PrecondSetupFailed {
                reason: "AMS: face_space requires vertex coordinates (AmsPrecond::with_coords)"
                    .into(),
            })?;
            if !(dim == 3 && coords.len() == dim * n_nodes) {
                return Err(SolverError::PrecondSetupFailed {
                    reason: format!(
                        "AMS: face_space needs 3-D coordinates, got {} entries for {n_nodes} nodes",
                        coords.len()
                    ),
                });
            }
            let g_rp = g.row_ptr();
            let g_ci = g.col_idx();
            let g_vals = g.values();
            for d in 0..3 {
                // t_d[e] = (Gᵀ x_d)[e] = Σ_v G(e,v)·x_d(v)  (edge vector component);
                // Pi_d[e, v] = 0.5·t_d[e] for both vertices v of edge e.
                let mut coo = CooMatrix::new(n_edges, n_nodes);
                for e in 0..n_edges {
                    let mut t = T::zero();
                    for p in g_rp[e]..g_rp[e + 1] {
                        let v = g_ci[p] as usize;
                        t += g_vals[p] * T::from_f64(coords[v * dim + d]);
                    }
                    let t_half = t * T::from_f64(0.5);
                    for p in g_rp[e]..g_rp[e + 1] {
                        let v = g_ci[p] as usize;
                        coo.push(e, v, t_half);
                    }
                }
                let pid = CsrMatrix::from_coo(&coo); // n_edges × n_nodes
                let pid_t = pid.transpose_csr();
                let a_pid = fix_zero_rows(&pid_t.matmat(&a.matmat(&pid))); // n_nodes × n_nodes
                let (b_pid, _) = build_aux_solver(a_pid, &node_solver)?;
                face_blocks.push((pid, b_pid));
            }
        }

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
            edge_smoother: config.edge_smoother.clone(),
            cycle: config.cycle.clone(),
            scaled_inv_diag,
            g: g.clone(),
            smoother_sweeps: config.smoother_sweeps,
            node_precond,
            face_blocks,
            singular_problem: config.singular_problem,
            profile,
        })
    }

    /// Setup-time profile for diagnostics and performance tuning.
    pub fn profile(&self) -> &AmsProfile { &self.profile }
}

impl<T: ComplexScalar> Preconditioner for AmsPrecond<T> {
    type Vector = DenseVec<T>;

    /// Apply the AMS preconditioner.
    ///
    /// With [`AmsCycle::Additive`] this is the standard Hiptmair-Xu
    /// preconditioner `M⁻¹ ≈ S⁻¹ + G·P_v⁻¹·Gᵀ` (iterated `smoother_sweeps`
    /// times via Richardson when `smoother_sweeps > 1`).
    ///
    /// With [`AmsCycle::MultiplicativeV11`] the symmetric two-level V(1,1)
    /// cycle of HYPRE AMS is applied: pre-GS → nodal correction → post-GS.
    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n_edges = self.n_edges;
        let n_nodes = self.n_nodes;

        // y = 0
        for ys in y.as_mut_slice().iter_mut().take(n_edges) {
            *ys = T::zero();
        }

        if matches!(self.cycle, AmsCycle::MultiplicativeV11) {
            // ── Symmetric multiplicative V(1,1) cycle (HYPRE AMS) ───────────
            let mut r = DenseVec::zeros(n_edges);

            // Pre-smoothing: y ← S⁻¹·x (one symmetric GS sweep from zero).
            self.edge_solve(x, y);

            if self.face_blocks.is_empty() {
                // Two-space cycle: GS → nodal → GS.
                self.residual_of(x, y, &mut r);
                self.apply_coarse(&self.g, &*self.node_precond, &r, y);
                self.residual_of(x, y, &mut r);
                let mut post = DenseVec::zeros(n_edges);
                self.edge_solve(&r, &mut post);
                {
                    let ps = post.as_slice();
                    let ys = y.as_mut_slice();
                    for i in 0..n_edges {
                        ys[i] = ys[i] + ps[i];
                    }
                }
                return;
            }

            // Full HYPRE AMS block-Pi multiplicative cycle.  hypre 2.28
            // `hypre_AMSSolve` (ams.c:3688) switches on `beta_is_zero`
            // (= [`AmsConfig::singular_problem`]):
            // * non-singular: `cycle_type = 13` → "034515430"
            //   GS → Pi_x → Pi_y → Pi_z → nodal → Pi_z → Pi_y → Pi_x → GS
            // * singular (SetSingularProblem): → "0345430" — the nodal
            //   gradient correction is dropped (hypre never builds B_G).
            let use_nodal = !self.singular_problem;
            let dbg = std::env::var("LINLVO_AMS_DEBUG").is_ok();
            let rayleigh = |tag: &str, v: &DenseVec<T>| {
                if !dbg {
                    return;
                }
                // Temporary diagnostic (LINLVO_AMS_DEBUG): ||v||² and the
                // Rayleigh quotient (v,Av)/(v,v) — Debug formatting since
                // Scalar::Real does not implement formatting traits.
                let v2 = v
                    .as_slice()
                    .iter()
                    .fold(<T::Real as crate::core::scalar::Scalar>::from_f64(0.0), |acc, vv| {
                        acc + (*vv * T::conj(*vv)).real()
                    });
                let mut av = DenseVec::zeros(v.len());
                self.a.spmv(v.as_slice(), av.as_mut_slice());
                let vav = v
                    .as_slice()
                    .iter()
                    .zip(av.as_slice().iter())
                    .fold(<T::Real as crate::core::scalar::Scalar>::from_f64(0.0), |acc, (vv, avv)| {
                        acc + (*avv * T::conj(*vv)).real()
                    });
                eprintln!("[ams] {tag}: ||v||^2 = {:?}  rayleigh = {:?}", v2, vav / v2);
            };
            rayleigh("after GS", y);
            for (idx, (pid, b)) in self.face_blocks.iter().enumerate() {
                self.residual_of(x, y, &mut r);
                self.apply_coarse(pid, &**b, &r, y);
                rayleigh(&format!("after Pi arm {idx}"), y);
            }
            if use_nodal {
                self.residual_of(x, y, &mut r);
                self.apply_coarse(&self.g, &*self.node_precond, &r, y);
            }
            for (pid, b) in self.face_blocks.iter().rev() {
                self.residual_of(x, y, &mut r);
                self.apply_coarse(pid, &**b, &r, y);
            }
            self.residual_of(x, y, &mut r);
            let mut post = DenseVec::zeros(n_edges);
            self.edge_solve(&r, &mut post);
            {
                let ps = post.as_slice();
                let ys = y.as_mut_slice();
                for i in 0..n_edges {
                    ys[i] = ys[i] + ps[i];
                }
            }
            return;
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

            // ── corr = smoother(A, r)  (edge smoother) ─────────────────────
            self.edge_solve(&r, &mut corr);

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

impl<T: ComplexScalar> AmsPrecond<T> {
    /// Compute `r = x - A·y` (edge residual).
    fn residual_of(&self, x: &DenseVec<T>, y: &DenseVec<T>, r: &mut DenseVec<T>) {
        let n_edges = self.n_edges;
        self.a.spmv_add(T::one(), y.as_slice(), T::zero(), r.as_mut_slice());
        let xs = x.as_slice();
        let rs = r.as_mut_slice();
        for i in 0..n_edges {
            rs[i] = xs[i] - rs[i];
        }
    }

    /// Apply one coarse-space correction: `y += P·B⁻¹·Pᵀ·r`.
    ///
    /// The temporary vectors are sized from `p.ncols()` — the Pi blocks may be
    /// rectangular (`n_edges × n_face_dofs`) with more columns than the nodal
    /// `G` block's vertex count.
    fn apply_coarse(
        &self,
        p: &CsrMatrix<T>,
        b: &dyn Preconditioner<Vector = DenseVec<T>>,
        r: &DenseVec<T>,
        y: &mut DenseVec<T>,
    ) {
        let n_cols = p.ncols();
        let mut t_node = DenseVec::zeros(n_cols);
        let mut s_node = DenseVec::zeros(n_cols);
        p.apply_transpose(r, &mut t_node);
        b.apply_precond(&t_node, &mut s_node);
        p.spmv_add(T::one(), s_node.as_slice(), T::one(), y.as_mut_slice());
    }

    /// Solve `b_out ≈ S⁻¹·b_in` with the configured edge smoother
    /// (weighted Jacobi or one symmetric GS sweep from zero).
    fn edge_solve(&self, b_in: &DenseVec<T>, b_out: &mut DenseVec<T>) {
        let n_edges = self.n_edges;
        match &self.edge_smoother {
            AmsEdgeSmoother::WeightedJacobi => {
                let bs = b_in.as_slice();
                let os = b_out.as_mut_slice();
                for i in 0..n_edges {
                    os[i] = self.scaled_inv_diag[i] * bs[i];
                }
            }
            AmsEdgeSmoother::SymmetricGaussSeidel => {
                // Symmetric GS solve of A·b_out = b_in (forward + backward),
                // matching HYPRE AMS's default rlx_type = 2 smoother.
                for c in b_out.as_mut_slice().iter_mut().take(n_edges) {
                    *c = T::zero();
                }
                crate::amg::smoother::smooth_with_hint(
                    &self.a,
                    b_out,
                    b_in,
                    &crate::amg::smoother::SmootherType::SymmetricGaussSeidel,
                    1,
                    None,
                );
            }
            AmsEdgeSmoother::L1ScaledSymmetricGaussSeidel => {
                // hypre rlx_type = 2 exact semantics: l1-scaled symmetric GS
                // (ams.c:3041-3053 computes the row l1 norms of A for relax
                // types 1-4; `hypre_BoomerAMGRelaxHybridSOR` divides by them).
                for c in b_out.as_mut_slice().iter_mut().take(n_edges) {
                    *c = T::zero();
                }
                crate::simd::smoother::l1_sgs_smooth(&self.a, b_out, b_in, 1);
            }
        }
    }
}

// ─── Shared helper ────────────────────────────────────────────────────────────

/// Replace all-zero rows of `mat` with a unit diagonal row.
///
/// hypre `hypre_ParCSRMatrixFixZeroRows` (used on `A_Pi` in `hypre_AMSSetup`,
/// ams.c:3272: "Make sure that A_Pix has no zero rows"): a zero row would
/// break the AMG l1-based smoothers; fixing it to the identity row leaves the
/// preconditioner consistent on the affected dofs.
fn fix_zero_rows<T: ComplexScalar>(mat: &CsrMatrix<T>) -> CsrMatrix<T> {
    let tol = T::machine_epsilon() * <T::Real as Scalar>::from_f64(1e6);
    let mut coo = CooMatrix::new(mat.nrows(), mat.ncols());
    let mut zero_rows = vec![true; mat.nrows()];
    for (r, c, v) in mat.triplets() {
        if v.abs() > tol {
            zero_rows[r] = false;
        }
        coo.push(r, c, v);
    }
    let mut fixed = false;
    for (r, z) in zero_rows.iter().enumerate() {
        if *z {
            coo.push(r, r, T::one());
            fixed = true;
        }
    }
    if fixed {
        CsrMatrix::from_coo(&coo)
    } else {
        mat.clone()
    }
}

/// Build a boxed coarse-space solver from the given operator and strategy.
///
/// `pub(super)` so that `ads.rs` can call it without duplicating the match.
#[allow(clippy::type_complexity)]
pub(super) fn build_aux_solver<T: ComplexScalar>(
    mat:    CsrMatrix<T>,
    solver: &AuxSpaceSolver,
) -> Result<(Box<dyn Preconditioner<Vector = DenseVec<T>>>, AuxSolverProfile), SolverError> {
    match solver {
        AuxSpaceSolver::Amg(cfg) => {
            let hier = AmgHierarchy::build(mat, cfg.clone());
            if std::env::var("FEMRS_AUXAMG_DEBUG").as_deref() == Ok("1") {
                let sizes: Vec<usize> = hier.level_info().iter().map(|l| l.ndof).collect();
                eprintln!(
                    "aux-AMG: n={:5} levels={:2} sizes={:?} op_cx={:.2}",
                    sizes.first().copied().unwrap_or(0),
                    hier.n_levels(),
                    sizes,
                    hier.operator_complexity()
                );
            }
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

    /// Face (curl) auxiliary space: rejects missing / non-3-D coordinates,
    /// and applies cleanly when enabled with 3-D vertex coordinates.
    #[test]
    fn ams_face_space_coords_validation_and_apply() {
        let (g, a) = chain_graph(6, 1e-3);

        // face_space without coordinates → error.
        let cfg_missing = AmsConfig { face_space: true, ..Default::default() };
        assert!(AmsPrecond::new(&a, &g, cfg_missing).is_err());

        // face_space with 2-D coordinates (wrong length) → error.
        let coords2d: Vec<f64> = (0..g.ncols() * 2).map(|i| i as f64).collect();
        let cfg = AmsConfig {
            face_space: true,
            cycle: AmsCycle::MultiplicativeV11,
            edge_smoother: AmsEdgeSmoother::SymmetricGaussSeidel,
            ..Default::default()
        };
        assert!(AmsPrecond::with_coords(&a, &g, &coords2d, cfg.clone()).is_err());

        // 3-D coordinates along a line → builds and applies.
        let coords3d: Vec<f64> = (0..g.ncols())
            .flat_map(|i| vec![i as f64, 0.0, 0.0])
            .collect();
        let p = AmsPrecond::with_coords(&a, &g, &coords3d, cfg).unwrap();
        let n = a.nrows();
        let x = DenseVec::from_vec(vec![1.0f64; n]);
        let mut y = DenseVec::zeros(n);
        p.apply_precond(&x, &mut y);
        let ys = y.as_slice();
        assert!(ys.iter().any(|&v| v.abs() > 1e-15), "output should be non-zero");
        assert!(ys.iter().all(|&v| v.is_finite()), "output should be finite");
    }
}
