use super::{operator::LinearOperator, vector::Vector};

/// An approximate inverse of a linear operator, used to accelerate Krylov methods.
///
/// Implementors expose `apply_precond` which computes  `y ← M⁻¹ · x`.
///
/// The optional `setup` method is called once before the solve phase to
/// analyse the operator (e.g. compute ILU factors, build AMG hierarchy).
/// It is separate from the constructor so that the same preconditioner
/// *struct* can be reused across multiple right-hand sides.
pub trait Preconditioner: Send + Sync {
    type Vector: Vector;

    /// Apply the approximate inverse:  `y ← M⁻¹ · x`.
    fn apply_precond(&self, x: &Self::Vector, y: &mut Self::Vector);

    /// (Optional) Analyse `op` and build internal data structures.
    ///
    /// Default implementation is a no-op, suitable for stateless
    /// preconditioners like Jacobi when the diagonal is already stored.
    fn setup(&mut self, _op: &dyn LinearOperator<Vector = Self::Vector>) {}
}
