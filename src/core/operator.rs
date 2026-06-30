use super::vector::Vector;

/// A linear map  `y ← A·x`.
///
/// Implemented by sparse/dense matrices and matrix-free operators alike.
/// All implementations must be `Send + Sync` to allow multi-threaded solvers.
pub trait LinearOperator: Send + Sync {
    type Vector: Vector;

    /// Compute  `y = A · x`.
    fn apply(&self, x: &Self::Vector, y: &mut Self::Vector);

    /// Number of rows of the operator.
    fn nrows(&self) -> usize;

    /// Number of columns of the operator.
    fn ncols(&self) -> usize;

    /// Block apply: compute `y_j = A · x_j` for j = 0..k.
    ///
    /// The default implementation calls [`apply`] for each column.
    /// Override this when a single allreduce can replace k individual ones.
    fn block_apply(
        &self,
        x_columns: &[&Self::Vector],
        y_columns: &mut [&mut Self::Vector],
    ) {
        for j in 0..x_columns.len() {
            self.apply(x_columns[j], y_columns[j]);
        }
    }
}

/// Extension trait for operators that also support transposed application
/// `y ← Aᵀ·x`.
pub trait TransposeOperator: LinearOperator {
    /// Compute  `y = Aᵀ · x`.
    fn apply_transpose(&self, x: &Self::Vector, y: &mut Self::Vector);
}
