use super::{operator::LinearOperator, preconditioner::Preconditioner, vector::Vector};
use crate::core::error::SolverError;

// ─── Verbosity ────────────────────────────────────────────────────────────────

/// Controls how much the solver prints during iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerboseLevel {
    /// No output.
    #[default]
    Silent,
    /// Print final convergence summary.
    Summary,
    /// Print residual at every iteration.
    Iterations,
}

// ─── SolverParams ─────────────────────────────────────────────────────────────

/// Convergence and behaviour parameters shared by all Krylov solvers.
#[derive(Debug, Clone)]
pub struct SolverParams {
    /// Relative residual tolerance: converge when `‖r‖/‖b‖ < rtol`.
    pub rtol: f64,
    /// Absolute residual tolerance: converge when `‖r‖ < atol`.
    pub atol: f64,
    /// Maximum number of iterations (outer iterations for restarted methods).
    pub max_iter: usize,
    /// Verbosity level.
    pub verbose: VerboseLevel,
    /// Interval (in iterations) at which the true residual is recomputed to
    /// guard against floating-point drift.  `0` disables recomputation.
    pub check_interval: usize,
}

impl Default for SolverParams {
    fn default() -> Self {
        SolverParams {
            rtol: 1e-8,
            atol: 0.0,
            max_iter: 1_000,
            verbose: VerboseLevel::Silent,
            check_interval: 10,
        }
    }
}

// ─── SolverResult ─────────────────────────────────────────────────────────────

/// Outcome of a single `KrylovSolver::solve` call.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// `true` if the solver reached the requested tolerance.
    pub converged: bool,
    /// Total number of matrix-vector products performed.
    pub iterations: usize,
    /// `‖b − A·x‖₂ / ‖b‖₂` at exit (or `‖b − A·x‖₂` if `‖b‖ = 0`).
    pub final_residual: f64,
    /// Per-iteration residual history (populated only when
    /// `verbose == VerboseLevel::Iterations`).
    pub history: Option<Vec<f64>>,
}

// ─── KrylovSolver ─────────────────────────────────────────────────────────────

/// Common interface for all Krylov iterative solvers.
///
/// Implementations include [`ConjugateGradient`], [`Gmres`], [`BiCgStab`], etc.
///
/// # Example skeleton
/// ```ignore
/// let solver = ConjugateGradient::new();
/// let result = solver.solve(&op, Some(&jacobi), &b, &mut x, &params)?;
/// assert!(result.converged);
/// ```
pub trait KrylovSolver: Send + Sync {
    type Vector: Vector;
    type Operator: LinearOperator<Vector = Self::Vector>;

    /// Solve  `A · x = b`  with optional preconditioning.
    ///
    /// * `op`     — the linear operator A
    /// * `precond` — optional preconditioner M⁻¹
    /// * `b`      — right-hand side
    /// * `x`      — initial guess on entry, solution on exit
    /// * `params` — convergence and verbosity settings
    ///
    /// # Errors
    /// Returns [`SolverError::ConvergenceFailed`] if `max_iter` is reached
    /// without satisfying the tolerance, or a numerical-breakdown error if the
    /// iteration cannot continue.
    fn solve(
        &self,
        op: &Self::Operator,
        precond: Option<&dyn Preconditioner<Vector = Self::Vector>>,
        b: &Self::Vector,
        x: &mut Self::Vector,
        params: &SolverParams,
    ) -> Result<SolverResult, SolverError>;
}
