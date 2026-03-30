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
