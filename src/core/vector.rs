use super::scalar::{ComplexScalar, Scalar};
use num_complex::Complex;

/// Abstract dense-vector interface required by all Krylov solvers.
///
/// The scalar type is [`ComplexScalar`], which is implemented by both real
/// types (`f32`, `f64`) and complex types (`Complex<f32>`, `Complex<f64>`).
/// All existing real-scalar code is unaffected: for `T: Scalar`, `T`
/// automatically satisfies `ComplexScalar<Real = T>`.
///
/// The default concrete type is [`DenseVec<T>`].
pub trait Vector: Clone + Send + Sync {
    /// Element type — a real or complex scalar.
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

/// Heap-allocated dense vector — the default [`Vector`] implementation.
///
/// Supports both real (`f32`, `f64`) and complex (`Complex<f32>`,
/// `Complex<f64>`) element types.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseVec<T>(Vec<T>);

// ── Utility methods for real scalars ─────────────────────────────────────────

impl<T: Scalar> DenseVec<T> {
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

    /// Mutable slice view.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    /// Consume the wrapper and return the inner `Vec<T>`.
    pub fn into_vec(self) -> Vec<T> {
        self.0
    }

    /// Element-wise difference `self − other` → new vector.
    pub fn sub(&self, other: &Self) -> Self {
        debug_assert_eq!(self.0.len(), other.0.len(), "DenseVec::sub: length mismatch");
        let mut out = DenseVec(vec![T::zero(); self.0.len()]);
        crate::simd::dense_ops::simd_sub(&self.0, &other.0, &mut out.0);
        out
    }

    /// Element-wise product `self ⊙ other` → new vector (Hadamard product).
    pub fn hadamard(&self, other: &Self) -> Self {
        debug_assert_eq!(self.0.len(), other.0.len(), "DenseVec::hadamard: length mismatch");
        let mut out = DenseVec(vec![T::zero(); self.0.len()]);
        crate::simd::dense_ops::simd_hadamard(&self.0, &other.0, &mut out.0);
        out
    }

    /// Infinity-norm: `max |xᵢ|`.
    pub fn max_abs(&self) -> T {
        self.0.iter().fold(T::zero(), |m, &v| if v.abs() > m { v.abs() } else { m })
    }

    /// L1-norm: `Σ |xᵢ|`.
    pub fn l1_norm(&self) -> T {
        self.0.iter().fold(T::zero(), |s, &v| s + v.abs())
    }
}

impl<T: Scalar> Vector for DenseVec<T> {
    type Scalar = T;

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    fn dot(&self, other: &Self) -> T {
        debug_assert_eq!(self.0.len(), other.0.len());
        crate::simd::dense_ops::simd_dot(&self.0, &other.0)
    }

    fn axpy(&mut self, alpha: T, x: &Self) {
        debug_assert_eq!(self.0.len(), x.0.len());
        crate::simd::dense_ops::simd_axpy(alpha, &x.0, &mut self.0);
    }

    fn scale(&mut self, alpha: T) {
        crate::simd::dense_ops::simd_scale(alpha, &mut self.0);
    }

    /// Returns the Euclidean norm.  For `T: Scalar`, `T::Real = T`, so the
    /// return type is still `T` — identical to the previous API.
    fn norm2(&self) -> T {
        crate::simd::dense_ops::simd_norm2(&self.0)
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

impl<T: Scalar> std::ops::Index<usize> for DenseVec<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}

impl<T: Scalar> std::ops::IndexMut<usize> for DenseVec<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}

impl<T: Scalar> From<Vec<T>> for DenseVec<T> {
    fn from(v: Vec<T>) -> Self {
        DenseVec(v)
    }
}

impl<T: Scalar> From<DenseVec<T>> for Vec<T> {
    fn from(d: DenseVec<T>) -> Self {
        d.0
    }
}

// ─── DenseVec<Complex<T>> ────────────────────────────────────────────────────

/// `Vector` implementation for complex dense vectors.
///
/// The `Scalar` associated type is `Complex<T>` and `norm2` returns `T`
/// (the real-valued Euclidean norm).
impl<T: Scalar> Vector for DenseVec<Complex<T>> {
    type Scalar = Complex<T>;

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    /// Hermitian inner product: `Σ conj(selfᵢ) · otherᵢ`.
    fn dot(&self, other: &Self) -> Complex<T> {
        debug_assert_eq!(self.0.len(), other.0.len());
        self.0
            .iter()
            .zip(other.0.iter())
            .fold(Complex::new(T::zero(), T::zero()), |acc, (&a, &b)| acc + a.conj() * b)
    }

    fn axpy(&mut self, alpha: Complex<T>, x: &Self) {
        debug_assert_eq!(self.0.len(), x.0.len());
        for (y_i, &x_i) in self.0.iter_mut().zip(x.0.iter()) {
            *y_i += alpha * x_i;
        }
    }

    fn scale(&mut self, alpha: Complex<T>) {
        for v in self.0.iter_mut() {
            *v *= alpha;
        }
    }

    /// Euclidean norm `√(Σ |zᵢ|²)` — always real-valued.
    fn norm2(&self) -> T {
        let sq: T = self.0.iter().map(|v| v.norm_sqr()).fold(T::zero(), |a, b| a + b);
        sq.sqrt()
    }

    fn zero_like(&self) -> Self {
        DenseVec(vec![Complex::new(T::zero(), T::zero()); self.0.len()])
    }

    fn fill(&mut self, alpha: Complex<T>) {
        for v in self.0.iter_mut() {
            *v = alpha;
        }
    }

    fn copy_from(&mut self, src: &Self) {
        assert_eq!(self.0.len(), src.0.len(), "DenseVec::copy_from: length mismatch");
        self.0.copy_from_slice(&src.0);
    }

    fn as_slice(&self) -> &[Complex<T>] { &self.0 }
    fn as_mut_slice(&mut self) -> &mut [Complex<T>] { &mut self.0 }
}

impl<T: Scalar> std::ops::Index<usize> for DenseVec<Complex<T>> {
    type Output = Complex<T>;
    fn index(&self, i: usize) -> &Complex<T> {
        &self.0[i]
    }
}

impl<T: Scalar> std::ops::IndexMut<usize> for DenseVec<Complex<T>> {
    fn index_mut(&mut self, i: usize) -> &mut Complex<T> {
        &mut self.0[i]
    }
}

impl<T: Scalar> From<Vec<Complex<T>>> for DenseVec<Complex<T>> {
    fn from(v: Vec<Complex<T>>) -> Self {
        DenseVec(v)
    }
}

impl<T: Scalar> From<DenseVec<Complex<T>>> for Vec<Complex<T>> {
    fn from(d: DenseVec<Complex<T>>) -> Self {
        d.0
    }
}

// ── Accessor methods for Complex<T> vectors ───────────────────────────────────
//
// NOTE: No inherent `as_slice`/`as_mut_slice`/`len` here — those exist only in
// `impl<T: Scalar> DenseVec<T>` above.  For complex vectors, access elements
// through the `Vector` trait methods or the `Index`/`IndexMut` impls.
// The `len()` method is available through `Vector::len()`.

impl<T: Scalar> DenseVec<Complex<T>> {}
