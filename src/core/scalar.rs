use num_traits::{Float, NumAssign, One, Zero};
use std::fmt::Debug;

/// Numeric scalar bound used by all linger algorithms.
///
/// Every algorithm is generic over `T: Scalar`, giving unified f32/f64 support.
///
/// # Examples
/// ```
/// use linger::Scalar;
/// fn eps<T: Scalar>() -> T { T::machine_epsilon() }
/// assert!(eps::<f64>() < 1e-10);
/// assert!(eps::<f32>() < 1e-5);
/// ```
pub trait Scalar:
    Float + NumAssign + Zero + One + Copy + Debug + Send + Sync + 'static
{
    /// Machine epsilon: the smallest `e` such that `1 + e != 1`.
    fn machine_epsilon() -> Self;

    /// Convert a plain `f64` literal to `Self` (panics if out of range).
    ///
    /// Intended only for test/benchmark helpers where the value is known valid.
    fn from_f64(v: f64) -> Self {
        <Self as num_traits::NumCast>::from(v)
            .expect("Scalar::from_f64: value out of range")
    }
}

impl Scalar for f64 {
    #[inline]
    fn machine_epsilon() -> Self {
        f64::EPSILON
    }
}

impl Scalar for f32 {
    #[inline]
    fn machine_epsilon() -> Self {
        f32::EPSILON
    }
}

// ─── ComplexScalar ────────────────────────────────────────────────────────────

/// Scalar trait covering both real (`f32`, `f64`) and complex
/// (`Complex<f32>`, `Complex<f64>`) values.
///
/// Unlike [`Scalar`] this does **not** require `Float` (which `Complex` cannot
/// implement) or `PartialOrd` / `signum` (undefined for complex numbers).
///
/// The associated type [`ComplexScalar::Real`] allows algorithms to extract
/// real-valued norms and residuals from complex vectors.
pub trait ComplexScalar:
    NumAssign + Zero + One + Copy + Debug + Send + Sync + 'static
{
    /// The real (floating-point) component type.
    type Real: Scalar;

    fn from_f64(v: f64) -> Self;
    fn from_real(r: Self::Real) -> Self;
    fn real(self) -> Self::Real;
    fn imag(self) -> Self::Real;
    /// Modulus |z|.
    fn abs(self) -> Self::Real;
    fn conj(self) -> Self;
    fn sqrt(self) -> Self;
    fn is_finite(self) -> bool;
    fn machine_epsilon() -> Self::Real;
}

/// Every `Scalar` (real floating-point type) is automatically a `ComplexScalar`
/// with `Real = Self`.
impl<T: Scalar> ComplexScalar for T {
    type Real = T;

    #[inline] fn from_f64(v: f64) -> Self { <T as Scalar>::from_f64(v) }
    #[inline] fn from_real(r: T) -> Self { r }
    #[inline] fn real(self) -> T { self }
    #[inline] fn imag(self) -> T { T::zero() }
    #[inline] fn abs(self) -> T { <T as Float>::abs(self) }
    #[inline] fn conj(self) -> T { self }
    #[inline] fn sqrt(self) -> T { <T as Float>::sqrt(self) }
    #[inline] fn is_finite(self) -> bool { <T as Float>::is_finite(self) }
    #[inline] fn machine_epsilon() -> T { T::machine_epsilon() }
}

impl ComplexScalar for num_complex::Complex<f64> {
    type Real = f64;

    #[inline] fn from_f64(v: f64) -> Self { num_complex::Complex::new(v, 0.0) }
    #[inline] fn from_real(r: f64) -> Self { num_complex::Complex::new(r, 0.0) }
    #[inline] fn real(self) -> f64 { self.re }
    #[inline] fn imag(self) -> f64 { self.im }
    #[inline] fn abs(self) -> f64 { num_complex::Complex::norm(self) }
    #[inline] fn conj(self) -> Self { num_complex::Complex::conj(&self) }
    #[inline] fn sqrt(self) -> Self { num_complex::Complex::sqrt(self) }
    #[inline] fn is_finite(self) -> bool { self.re.is_finite() && self.im.is_finite() }
    #[inline] fn machine_epsilon() -> f64 { f64::EPSILON }
}

impl ComplexScalar for num_complex::Complex<f32> {
    type Real = f32;

    #[inline] fn from_f64(v: f64) -> Self { num_complex::Complex::new(v as f32, 0.0) }
    #[inline] fn from_real(r: f32) -> Self { num_complex::Complex::new(r, 0.0) }
    #[inline] fn real(self) -> f32 { self.re }
    #[inline] fn imag(self) -> f32 { self.im }
    #[inline] fn abs(self) -> f32 { num_complex::Complex::norm(self) }
    #[inline] fn conj(self) -> Self { num_complex::Complex::conj(&self) }
    #[inline] fn sqrt(self) -> Self { num_complex::Complex::sqrt(self) }
    #[inline] fn is_finite(self) -> bool { self.re.is_finite() && self.im.is_finite() }
    #[inline] fn machine_epsilon() -> f32 { f32::EPSILON }
}
