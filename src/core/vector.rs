use super::scalar::Scalar;

/// Abstract dense-vector interface required by all Krylov solvers.
///
/// The default concrete type is [`DenseVec<T>`].
pub trait Vector: Clone + Send + Sync {
    type Scalar: Scalar;

    /// Number of elements.
    fn len(&self) -> usize;

    /// Returns `true` if the vector has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Euclidean inner product `<self, other>`.
    fn dot(&self, other: &Self) -> Self::Scalar;

    /// `self += alpha * x`  (BLAS-1 AXPY).
    fn axpy(&mut self, alpha: Self::Scalar, x: &Self);

    /// `self *= alpha`.
    fn scale(&mut self, alpha: Self::Scalar);

    /// Euclidean 2-norm  `√(Σ xᵢ²)`.
    fn norm2(&self) -> Self::Scalar;

    /// Allocate a zero vector with the same length as `self`.
    fn zero_like(&self) -> Self;

    /// Fill every element with `alpha`.
    fn fill(&mut self, alpha: Self::Scalar);

    /// Copy elements from `src` into `self`.
    ///
    /// # Panics
    /// Panics if `self.len() != src.len()`.
    fn copy_from(&mut self, src: &Self);
}

// ─── DenseVec<T> ─────────────────────────────────────────────────────────────

/// Heap-allocated dense vector — the default [`Vector`] implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseVec<T>(Vec<T>);

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
}

impl<T: Scalar> Vector for DenseVec<T> {
    type Scalar = T;

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    fn dot(&self, other: &Self) -> T {
        debug_assert_eq!(self.0.len(), other.0.len());
        self.0
            .iter()
            .zip(other.0.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }

    fn axpy(&mut self, alpha: T, x: &Self) {
        debug_assert_eq!(self.0.len(), x.0.len());
        for (y_i, &x_i) in self.0.iter_mut().zip(x.0.iter()) {
            *y_i += alpha * x_i;
        }
    }

    fn scale(&mut self, alpha: T) {
        for y_i in self.0.iter_mut() {
            *y_i *= alpha;
        }
    }

    fn norm2(&self) -> T {
        let ss = self.0.iter().fold(T::zero(), |acc, &v| acc + v * v);
        ss.sqrt()
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
