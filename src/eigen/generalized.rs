//! Generalized Eigenvalue Problem + Spectral Transforms — Sprint 9.
//!
//! Solves `A x = λ B x` by reduction to a standard problem via
//! spectral transforms.
//!
//! **Shift-Invert (SI) transform:**
//! ```text
//! (A − σB)⁻¹ B x = ν x,   ν = 1/(λ − σ)
//! ```
//! The largest ν correspond to λ closest to σ.  The inner linear system
//! `(A − σB) y = B x` is solved with the matrix-free GMRES from Sprint 7.
//!
//! **Buckling transform:**
//! ```text
//! (A − σB)⁻¹ A x = ν x
//! ```
//! (for structural buckling problems)
//!
//! Both transforms wrap the composed operator and delegate to [`LanczosIter`]
//! (symmetric) or [`ArnoldiIter`] (non-symmetric).

use crate::core::{error::SolverError, operator::LinearOperator, scalar::Scalar, vector::{DenseVec, Vector}};
use crate::eigen::{EigenParams, EigenResult, EigenSolver, EigenWhich, matfree_gmres};
use crate::eigen::lanczos::LanczosIter;
use crate::eigen::arnoldi::ArnoldiIter;

// ─── Spectral transform trait ─────────────────────────────────────────────────

/// A spectral transform maps a generalised problem `(A, B)` to a standard
/// one whose extreme eigenvalues correspond to desired eigenvalues of `(A,B)`.
pub trait SpectralTransform<T: Scalar>: Send + Sync {
    /// Apply the transformed operator: `y ← T(A,B) · x`.
    fn apply_transformed(
        &self,
        x: &DenseVec<T>,
        y: &mut DenseVec<T>,
    );

    /// Map a transformed eigenvalue `ν` back to the original eigenvalue `λ`.
    fn recover_lambda(&self, nu: T) -> T;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

// ─── ShiftInvertST ────────────────────────────────────────────────────────────

/// Shift-Invert spectral transform for `Ax = λBx`.
///
/// Presents the operator `(A − σB)⁻¹B` to a standard eigensolver.
/// The inner system `(A − σB)y = Bx` is solved iteratively (GMRES).
///
/// With `b = None` (standard problem `Ax = λx`, B = I), this reduces to
/// `(A − σI)⁻¹`, identical to [`InverseIter`] but reusable with Lanczos/Arnoldi.
pub struct ShiftInvertST<'a, T, A, B>
where
    T: Scalar,
    A: LinearOperator<Vector = DenseVec<T>>,
    B: LinearOperator<Vector = DenseVec<T>>,
{
    pub a:          &'a A,
    pub b:          Option<&'a B>, // None → B = I
    pub shift:      T,
    pub inner_rtol: T,
    pub inner_max_iter: usize,
}

impl<'a, T, A, B> ShiftInvertST<'a, T, A, B>
where
    T: Scalar,
    A: LinearOperator<Vector = DenseVec<T>>,
    B: LinearOperator<Vector = DenseVec<T>>,
{
    pub fn new(a: &'a A, b: Option<&'a B>, shift: T) -> Self {
        ShiftInvertST {
            a, b, shift,
            inner_rtol: T::from_f64(1e-10),
            inner_max_iter: 500,
        }
    }
}

/// Operator wrapping `(A − σB)` for the inner system.
struct ShiftedPencil<'a, T, A, B>
where
    T: Scalar,
    A: LinearOperator<Vector = DenseVec<T>>,
    B: LinearOperator<Vector = DenseVec<T>>,
{
    a: &'a A,
    b: Option<&'a B>,
    shift: T,
    n: usize,
}

impl<'a, T, A, B> LinearOperator for ShiftedPencil<'a, T, A, B>
where
    T: Scalar,
    A: LinearOperator<Vector = DenseVec<T>>,
    B: LinearOperator<Vector = DenseVec<T>>,
{
    type Vector = DenseVec<T>;
    fn nrows(&self) -> usize { self.n }
    fn ncols(&self) -> usize { self.n }

    /// y = (A − σB) x
    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        self.a.apply(x, y);                          // y = Ax
        if let Some(b) = self.b {
            let mut bx = DenseVec::zeros(self.n);
            b.apply(x, &mut bx);                     // bx = Bx
            let ys = y.as_mut_slice();
            let bs = bx.as_slice();
            for i in 0..self.n { ys[i] -= self.shift * bs[i]; }
        } else {
            let ys = y.as_mut_slice();
            let xs = x.as_slice();
            for i in 0..self.n { ys[i] -= self.shift * xs[i]; }
        }
    }
}

unsafe impl<'a, T: Scalar, A: LinearOperator<Vector=DenseVec<T>>+Send,
            B: LinearOperator<Vector=DenseVec<T>>+Send> Send
    for ShiftedPencil<'a, T, A, B> {}
unsafe impl<'a, T: Scalar, A: LinearOperator<Vector=DenseVec<T>>+Sync,
            B: LinearOperator<Vector=DenseVec<T>>+Sync> Sync
    for ShiftedPencil<'a, T, A, B> {}

/// Operator presenting `(A − σB)⁻¹ B` to an eigensolver.
pub(crate) struct ShiftInvertOp<'a, T, A, B>
where
    T: Scalar,
    A: LinearOperator<Vector = DenseVec<T>>,
    B: LinearOperator<Vector = DenseVec<T>>,
{
    pencil: ShiftedPencil<'a, T, A, B>,
    b:      Option<&'a B>,
    inner_rtol: T,
    inner_max_iter: usize,
    n: usize,
}

impl<'a, T, A, B> LinearOperator for ShiftInvertOp<'a, T, A, B>
where
    T: Scalar,
    A: LinearOperator<Vector = DenseVec<T>>,
    B: LinearOperator<Vector = DenseVec<T>>,
{
    type Vector = DenseVec<T>;
    fn nrows(&self) -> usize { self.n }
    fn ncols(&self) -> usize { self.n }

    /// y ← (A − σB)⁻¹ B x
    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) {
        // rhs = B x  (or x if B = I)
        let rhs = if let Some(b) = self.b {
            let mut bx = DenseVec::zeros(self.n);
            b.apply(x, &mut bx);
            bx
        } else {
            x.clone()
        };
        // solve (A − σB) y = rhs
        y.fill(T::zero());
        matfree_gmres(&self.pencil, &rhs, y, self.inner_rtol, self.inner_max_iter, 30).ok();
    }
}

unsafe impl<'a, T: Scalar, A: LinearOperator<Vector=DenseVec<T>>+Send,
            B: LinearOperator<Vector=DenseVec<T>>+Send> Send
    for ShiftInvertOp<'a, T, A, B> {}
unsafe impl<'a, T: Scalar, A: LinearOperator<Vector=DenseVec<T>>+Sync,
            B: LinearOperator<Vector=DenseVec<T>>+Sync> Sync
    for ShiftInvertOp<'a, T, A, B> {}

// ─── GeneralizedEigen ─────────────────────────────────────────────────────────

/// Generalized eigenvalue solver for `A x = λ B x`.
///
/// Uses Shift-Invert to find eigenvalues near `shift`, then delegates to
/// [`LanczosIter`] (symmetric) or [`ArnoldiIter`] (non-symmetric).
///
/// # Example
/// ```no_run
/// use linger::{
///     eigen::{EigenParams, EigenSolver, EigenWhich},
///     eigen::generalized::GeneralizedEigen,
///     sparse::{CooMatrix, CsrMatrix},
/// };
///
/// // K x = λ M x  (stiffness / mass)
/// let k: CsrMatrix<f64> = todo!();
/// let m: CsrMatrix<f64> = todo!();
/// let solver = GeneralizedEigen::symmetric(0.0, false);
/// let params = EigenParams::new(3, EigenWhich::SmallestMagnitude);
/// let res = solver.solve_generalized(&k, &m, &params).unwrap();
/// ```
pub struct GeneralizedEigen<T: Scalar> {
    /// Spectral shift σ.
    pub shift: T,
    /// `true` → use Lanczos (symmetric), `false` → Arnoldi (general).
    pub symmetric: bool,
    /// Krylov space size (None → auto).
    pub ncv: Option<usize>,
    pub inner_rtol: T,
    pub inner_max_iter: usize,
}

impl<T: Scalar> GeneralizedEigen<T> {
    pub fn symmetric(shift: T, _unused: bool) -> Self {
        GeneralizedEigen {
            shift,
            symmetric: true,
            ncv: None,
            inner_rtol: T::from_f64(1e-10),
            inner_max_iter: 500,
        }
    }
    pub fn nonsymmetric(shift: T) -> Self {
        GeneralizedEigen { shift, symmetric: false, ncv: None,
            inner_rtol: T::from_f64(1e-10), inner_max_iter: 500 }
    }

    /// Solve `A x = λ B x`.
    pub fn solve_generalized<A, B>(
        &self,
        a: &A,
        b: &B,
        params: &EigenParams<T>,
    ) -> Result<EigenResult<T>, SolverError>
    where
        A: LinearOperator<Vector = DenseVec<T>>,
        B: LinearOperator<Vector = DenseVec<T>>,
    {
        let n = a.nrows();
        let pencil = ShiftedPencil { a, b: Some(b), shift: self.shift, n };
        let op = ShiftInvertOp {
            pencil,
            b: Some(b),
            inner_rtol: self.inner_rtol,
            inner_max_iter: self.inner_max_iter,
            n,
        };

        // Params for the standard sub-problem: LM (largest ν = closest λ to σ)
        let inner_params = EigenParams {
            n_eigenvalues: params.n_eigenvalues,
            which: EigenWhich::LargestMagnitude,
            tol: params.tol,
            max_iter: params.max_iter,
            verbose: params.verbose,
        };

        let mut result = if self.symmetric {
            let l = LanczosIter { ncv: self.ncv, seed: 42 };
            EigenSolver::<T>::solve(&l, &op, &inner_params)?
        } else {
            let a_iter = ArnoldiIter { ncv: self.ncv, seed: 42 };
            EigenSolver::<T>::solve(&a_iter, &op, &inner_params)?
        };

        // Recover original eigenvalues: λ = σ + 1/ν
        for lam in result.eigenvalues.iter_mut() {
            if lam.abs() > T::from_f64(1e-14) {
                *lam = self.shift + T::one() / *lam;
            }
        }

        // Recompute residuals against original A, B
        let mut final_residuals = Vec::with_capacity(result.eigenvalues.len());
        for (lam, x) in result.eigenvalues.iter().zip(result.eigenvectors.iter()) {
            let mut ax = DenseVec::zeros(n);
            a.apply(x, &mut ax);
            let mut bx = DenseVec::zeros(n);
            b.apply(x, &mut bx);
            // ‖Ax − λBx‖
            let mut rn = T::zero();
            let axs = ax.as_slice();
            let bxs = bx.as_slice();
            for i in 0..n { let d = axs[i] - *lam * bxs[i]; rn += d * d; }
            final_residuals.push(rn.sqrt());
        }
        result.residuals = final_residuals;

        Ok(result)
    }
}

// ─── Generic identity operator ────────────────────────────────────────────────

struct IdentityOpG<T: Scalar> { n: usize, _p: std::marker::PhantomData<T> }
#[allow(dead_code)]
impl<T: Scalar> IdentityOpG<T> { fn new(n: usize) -> Self { IdentityOpG { n, _p: std::marker::PhantomData } } }
impl<T: Scalar> LinearOperator for IdentityOpG<T> {
    type Vector = DenseVec<T>;
    fn nrows(&self) -> usize { self.n }
    fn ncols(&self) -> usize { self.n }
    fn apply(&self, x: &DenseVec<T>, y: &mut DenseVec<T>) { y.copy_from(x); }
}
unsafe impl<T: Scalar + Send> Send for IdentityOpG<T> {}
unsafe impl<T: Scalar + Sync> Sync for IdentityOpG<T> {}

/// Convenience: solve standard `A x = λ x` near `shift` using Shift-Invert
/// Lanczos.  Equivalent to [`InverseIter`] but builds a full Krylov space,
/// giving `nev` eigenvalues in one pass.
pub struct ShiftInvertLanczos<T: Scalar> {
    pub shift: T,
    pub ncv: Option<usize>,
    pub inner_rtol: T,
    pub inner_max_iter: usize,
}

impl<T: Scalar> ShiftInvertLanczos<T> {
    pub fn new(shift: T) -> Self {
        ShiftInvertLanczos {
            shift,
            ncv: None,
            inner_rtol: T::from_f64(1e-10),
            inner_max_iter: 500,
        }
    }
}

impl<T: Scalar> EigenSolver<T> for ShiftInvertLanczos<T> {
    fn solve<Op>(&self, op: &Op, params: &EigenParams<T>) -> Result<EigenResult<T>, SolverError>
    where Op: LinearOperator<Vector = DenseVec<T>>
    {
        let n = op.nrows();

        let pencil: ShiftedPencil<'_, T, Op, IdentityOpG<T>> =
            ShiftedPencil { a: op, b: None, shift: self.shift, n };
        let si_op: ShiftInvertOp<'_, T, Op, IdentityOpG<T>> = ShiftInvertOp {
            pencil,
            b: None,
            inner_rtol: self.inner_rtol,
            inner_max_iter: self.inner_max_iter,
            n,
        };

        let inner_params = EigenParams {
            n_eigenvalues: params.n_eigenvalues,
            which: EigenWhich::LargestMagnitude,
            tol: params.tol,
            max_iter: params.max_iter,
            verbose: params.verbose,
        };

        let lanczos = LanczosIter { ncv: self.ncv, seed: 42 };
        let mut result = EigenSolver::<T>::solve(&lanczos, &si_op, &inner_params)?;

        // Recover λ = σ + 1/ν
        for lam in result.eigenvalues.iter_mut() {
            if lam.abs() > T::from_f64(1e-14) {
                *lam = self.shift + T::one() / *lam;
            }
        }
        // Recompute residuals vs original A
        let mut final_res = Vec::new();
        for (lam, x) in result.eigenvalues.iter().zip(result.eigenvectors.iter()) {
            let mut ax = DenseVec::zeros(n);
            op.apply(x, &mut ax);
            let mut rn = T::zero();
            let axs = ax.as_slice();
            let xs  = x.as_slice();
            for i in 0..n { let d = axs[i] - *lam * xs[i]; rn += d * d; }
            final_res.push(rn.sqrt());
        }
        result.residuals = final_res;
        Ok(result)
    }
}
