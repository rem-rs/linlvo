use super::scalar::{ComplexScalar, Scalar};
use num_traits::{Float, Zero};

/// Abstract dense-vector interface required by all Krylov solvers.
///
/// The scalar type is [`ComplexScalar`], which is implemented by both real
/// types (`f32`, `f64`) and complex types (`Complex<f32>`, `Complex<f64>`).
/// All existing real-scalar code is unaffected: for `T: Scalar`, `T`
/// automatically satisfies `ComplexScalar<Real = T>`.
///
/// The default concrete type is [`DenseVec<T>`].
pub trait Vector: Clone + Send + Sync {
    /// Element type - a real or complex scalar.
    type Scalar: ComplexScalar;

    /// Number of elements.
    fn len(&self) -> usize;

    /// Returns `true` if the vector has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Hermitian inner product `<self, other> = Σ conj(selfᵢ) · otherᵢ`.
    ///
    /// For real scalars this reduces to the standard dot product because
    /// `conj(x) = x`.
    fn dot(&self, other: &Self) -> Self::Scalar;

    /// `self += alpha * x`  (BLAS-1 AXPY).
    fn axpy(&mut self, alpha: Self::Scalar, x: &Self);

    /// `self *= alpha`.
    fn scale(&mut self, alpha: Self::Scalar);

    /// Euclidean 2-norm `√(Σ |xᵢ|²)`.
    ///
    /// Returns a **real** value even for complex vectors.
    fn norm2(&self) -> <Self::Scalar as ComplexScalar>::Real;

    /// Allocate a zero vector with the same length as `self`.
    fn zero_like(&self) -> Self;

    /// Fill every element with `alpha`.
    fn fill(&mut self, alpha: Self::Scalar);

    /// Copy elements from `src` into `self`.
    ///
    /// # Panics
    /// Panics if `self.len() != src.len()`.
    fn copy_from(&mut self, src: &Self);

    /// Read-only slice view of the underlying storage.
    fn as_slice(&self) -> &[Self::Scalar];

    /// Mutable slice view of the underlying storage.
    fn as_mut_slice(&mut self) -> &mut [Self::Scalar];
}

// ─── DenseVec<T> ─────────────────────────────────────────────────────────────

/// Heap-allocated dense vector - the default [`Vector`] implementation.
///
/// Supports both real (`f32`, `f64`) and complex (`Complex<f32>`,
/// `Complex<f64>`) element types.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseVec<T>(Vec<T>);

// ── Utility methods for all scalar types ─────────────────────────────────────

impl<T: ComplexScalar> DenseVec<T> {
    /// Create a zero vector of length `n`.
    pub fn zeros(n: usize) -> Self {
        DenseVec(vec![T::zero(); n])
    }

    /// Create from an existing `Vec<T>`.
    pub fn from_vec(v: Vec<T>) -> Self {
        DenseVec(v)
    }

    /// Read-only slice view.
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the vector has no elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Mutable slice view.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    /// Consume the wrapper and return the inner `Vec<T>`.
    pub fn into_vec(self) -> Vec<T> {
        self.0
    }

    /// Element-wise difference `self − other` -> new vector.
    pub fn sub(&self, other: &Self) -> Self {
        debug_assert_eq!(self.0.len(), other.0.len(), "DenseVec::sub: length mismatch");
        let out_data: Vec<T> = self.0.iter().zip(other.0.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        DenseVec(out_data)
    }

    /// Element-wise product `self ⊙ other` -> new vector (Hadamard product).
    pub fn hadamard(&self, other: &Self) -> Self {
        debug_assert_eq!(self.0.len(), other.0.len(), "DenseVec::hadamard: length mismatch");
        let out_data: Vec<T> = self.0.iter().zip(other.0.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        DenseVec(out_data)
    }

    /// Infinity-norm: `max |xᵢ|`.
    pub fn max_abs(&self) -> T::Real {
        self.0.iter().fold(T::Real::zero(), |m: T::Real, v| {
            let a = v.abs();
            if a > m { a } else { m }
        })
    }

    /// L1-norm: `Σ |xᵢ|`.
    pub fn l1_norm(&self) -> T::Real {
        self.0.iter().fold(T::Real::zero(), |s, v| s + v.abs())
    }
}

// ─── Unified Vector impl for all ComplexScalar types ─────────────────────────
//
// A single generic impl covers both real (f32/f64) and complex
// (Complex<f32>/Complex<f64>) element types.  The `dot` method computes the
// Hermitian inner product `Σ conj(selfᵢ)·otherᵢ`; for real scalars `conj` is
// a no-op, so this reduces to the standard dot product with zero overhead.
//
// SIMD acceleration for real types was previously provided via
// `simd::dense_ops::par_*` functions.  Those are still available for direct
// call sites that need maximum throughput; the `Vector` trait methods use
// scalar loops for portability across real and complex types.

impl<T: ComplexScalar> Vector for DenseVec<T> {
    type Scalar = T;

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    /// Hermitian inner product: `Σ conj(selfᵢ) · otherᵢ`.
    ///
    /// For real scalars `conj` is a no-op, so this is the standard dot product.
    fn dot(&self, other: &Self) -> T {
        debug_assert_eq!(self.0.len(), other.0.len());
        self.0
            .iter()
            .zip(other.0.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a.conj() * b)
    }

    fn axpy(&mut self, alpha: T, x: &Self) {
        debug_assert_eq!(self.0.len(), x.0.len());
        for (y_i, &x_i) in self.0.iter_mut().zip(x.0.iter()) {
            *y_i += alpha * x_i;
        }
    }

    fn scale(&mut self, alpha: T) {
        for v in self.0.iter_mut() {
            *v *= alpha;
        }
    }

    /// Euclidean norm `√(Σ |xᵢ|²)` - always returns a real value.
    fn norm2(&self) -> T::Real {
        let sq: T::Real = self.0.iter().fold(T::Real::zero(), |s, v| {
            let a = v.abs();
            s + a * a
        });
        <T::Real as Float>::sqrt(sq)
    }

    fn zero_like(&self) -> Self {
        DenseVec(vec![T::zero(); self.0.len()])
    }

    fn fill(&mut self, alpha: T) {
        for v in self.0.iter_mut() {
            *v = alpha;
        }
    }

    fn copy_from(&mut self, src: &Self) {
        assert_eq!(self.0.len(), src.0.len(), "DenseVec::copy_from: length mismatch");
        self.0.copy_from_slice(&src.0);
    }

    fn as_slice(&self) -> &[T] { &self.0 }
    fn as_mut_slice(&mut self) -> &mut [T] { &mut self.0 }
}

impl<T: ComplexScalar> std::ops::Index<usize> for DenseVec<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}

impl<T: ComplexScalar> std::ops::IndexMut<usize> for DenseVec<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}

impl<T: ComplexScalar> From<Vec<T>> for DenseVec<T> {
    fn from(v: Vec<T>) -> Self {
        DenseVec(v)
    }
}

impl<T: ComplexScalar> From<DenseVec<T>> for Vec<T> {
    fn from(d: DenseVec<T>) -> Self {
        d.0
    }
}
