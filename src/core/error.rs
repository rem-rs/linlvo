/// All errors that can be returned by linger's public API.
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    /// Encountered a (near-)singular pivot during factorisation.
    #[error("singular matrix detected at row {row}; try scaling, different ordering, or iterative fallback")]
    SingularMatrix { row: usize },

    /// Krylov iteration did not reach the requested tolerance.
    #[error("failed to converge after {max_iter} iterations, residual = {residual:.3e}; try stronger preconditioner, larger restart/max_iter, or looser tolerance")]
    ConvergenceFailed { max_iter: usize, residual: f64 },

    /// Operator and right-hand-side dimensions are incompatible.
    #[error(
        "dimension mismatch: operator is {op_rows}×{op_cols}, \
         rhs has {rhs_len} entries"
    )]
    DimensionMismatch {
        op_rows: usize,
        op_cols: usize,
        rhs_len: usize,
    },

    /// Preconditioner setup failed (e.g. zero diagonal during ILU).
    #[error("preconditioner setup failed: {reason}; try diagonal scaling, ordering changes, or fallback preconditioner")]
    PrecondSetupFailed { reason: String },

    /// Numerical breakdown in the iteration (e.g. zero inner product in CG).
    #[error("numerical breakdown: {detail}")]
    NumericalBreakdown { detail: String },
}
