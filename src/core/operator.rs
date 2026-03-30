use super::vector::Vector;

/// A linear map  `y ← A·x`.
///
/// Implemented by sparse/dense matrices and matrix-free operators alike.
/// All implementations must be `Send + Sync` to allow multi-threaded solvers.
pub trait LinearOperator: Send + Sync {
    type Vector: Vector;

    /// Compute  `y = A · x`.
    ///
    /// # Panics
    /// Implementations may panic if dimension constraints are violated.
    fn apply(&self, x: &Self::Vector, y: &mut Self::Vector);

    /// Number of rows of the operator.
    fn nrows(&self) -> usize;

    /// Number of columns of the operator.
    fn ncols(&self) -> usize;
}
