//! Composite preconditioners — additive and multiplicative combinations.
//!
//! ## Additive
//!
//! `y = M₁⁻¹ x + M₂⁻¹ x + …`
//!
//! Useful as a parallel preconditioner or Schwarz smoother.  The sum of
//! approximate inverses is itself an approximate inverse.
//!
//! ## Multiplicative
//!
//! `y = Mₖ⁻¹ … M₂⁻¹ M₁⁻¹ x`
//!
//! Applies preconditioners sequentially, each acting on the result of the
//! previous.  Equivalent to a multiplicative Schwarz iteration.
//!
//! **Analogs**
//!   PETSc: `PCCompositeSetType(pc, PC_COMPOSITE_ADDITIVE)` /
//!          `PC_COMPOSITE_MULTIPLICATIVE`
//!   HYPRE: composite preconditioners via custom setup

use crate::core::{preconditioner::Preconditioner, scalar::ComplexScalar, vector::{DenseVec, Vector}};

/// Additive composite preconditioner.
///
/// `apply_precond(x) = ∑ᵢ Mᵢ⁻¹ x`
pub struct AdditivePrecond<T: ComplexScalar> {
    preconditioners: Vec<Box<dyn Preconditioner<Vector = DenseVec<T>>>>,
}

impl<T: ComplexScalar> AdditivePrecond<T> {
    pub fn new(preconditioners: Vec<Box<dyn Preconditioner<Vector = DenseVec<T>>>>) -> Self {
        AdditivePrecond { preconditioners }
    }
}

impl<T: ComplexScalar> Preconditioner for AdditivePrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n = x.len();
        let ys = y.as_mut_slice();
        for v in ys.iter_mut() { *v = T::zero(); }

        let mut tmp = DenseVec::zeros(n);
        for p in &self.preconditioners {
            p.apply_precond(x, &mut tmp);
            let ts = tmp.as_slice();
            for i in 0..n {
                ys[i] += ts[i];
            }
        }
    }
}

/// Multiplicative composite preconditioner.
///
/// `apply_precond(x) = Mₖ⁻¹ … M₁⁻¹ x`
pub struct MultiplicativePrecond<T: ComplexScalar> {
    preconditioners: Vec<Box<dyn Preconditioner<Vector = DenseVec<T>>>>,
}

impl<T: ComplexScalar> MultiplicativePrecond<T> {
    pub fn new(preconditioners: Vec<Box<dyn Preconditioner<Vector = DenseVec<T>>>>) -> Self {
        MultiplicativePrecond { preconditioners }
    }
}

impl<T: ComplexScalar> Preconditioner for MultiplicativePrecond<T> {
    type Vector = DenseVec<T>;

    fn apply_precond(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        let n = x.len();
        // Pass through each preconditioner in sequence.
        let mut buf_a = x.clone();
        let mut buf_b = DenseVec::zeros(n);
        for (idx, p) in self.preconditioners.iter().enumerate() {
            if idx % 2 == 0 {
                p.apply_precond(&buf_a, &mut buf_b);
            } else {
                p.apply_precond(&buf_b, &mut buf_a);
            }
        }
        // Result is in whichever buffer was last written.
        if self.preconditioners.len() % 2 == 1 {
            y.copy_from(&buf_b);
        } else {
            y.copy_from(&buf_a);
        }
    }
}
