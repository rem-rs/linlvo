//! Dense LU factorisation using BLAS/LAPACK backend.
//!
//! When the `blas` feature is active, uses the system LAPACK library
//! (OpenBLAS / Accelerate / netlib) via Fortran-style FFI.
//! When only `blas-oxiblas` is active, falls back to pure-Rust nalgebra.
//! When no BLAS feature is active, also uses nalgebra.
//!
//! The BLAS backend is selected through linlvo's Cargo features:
//! - `blas-openblas` / `blas-openblas-system` → OpenBLAS zgetrf/zgetrs
//! - `blas-accelerate` → macOS Accelerate framework
//! - `blas-netlib` → reference netlib LAPACK
//! - `blas-oxiblas` → pure Rust OxiBLAS (no accelerated complex LU)
//! - (none) → pure Rust nalgebra::LU

use num_complex::Complex64;

/// Dense LU factorization result.
///
/// Stores the LU factors in-place and the pivot indices,
/// ready for use with [`DenseLu::solve`].
pub struct DenseLu {
    /// LU factors (L below diagonal, U on and above, compact storage).
    pub a: Vec<Complex64>,
    /// Pivot indices (0-based, Fortran convention: row k swapped with ipiv[k]).
    pub ipiv: Vec<i32>,
    /// Matrix dimension.
    pub n: usize,
}

impl DenseLu {
    /// Factorize a dense n×n matrix via LAPACK `zgetrf`.
    ///
    /// When the `blas` feature is active, uses the LAPACK implementation
    /// provided by the selected backend (OpenBLAS, Accelerate, etc.).
    /// Otherwise falls back to pure-Rust `nalgebra::LU`.
    ///
    /// The input matrix `a` is consumed and overwritten with the LU factors.
    #[cfg(feature = "blas")]
    pub fn factorize(a: Vec<Complex64>, n: usize) -> Result<Self, String> {
        let n_i32 = n as i32;
        let mut a = a;
        let mut ipiv = vec![0i32; n];
        let mut info: i32 = 0;

        unsafe {
            zgetrf_(&n_i32, &n_i32, a.as_mut_ptr(), &n_i32, ipiv.as_mut_ptr(), &mut info);
        }

        if info != 0 {
            return Err(format!("zgetrf failed with info={} (singular at row {})", info, info));
        }

        Ok(DenseLu { a, ipiv, n })
    }

    /// Solve A·x = b using the LU factors from [`DenseLu::factorize`].
    #[cfg(feature = "blas")]
    pub fn solve(&self, b: &[Complex64]) -> Vec<Complex64> {
        let n_i32 = self.n as i32;
        let mut x = b.to_vec();
        let one: i32 = 1;
        let mut info: i32 = 0;
        let trans: i8 = b'N' as i8;

        unsafe {
            zgetrs_(
                &trans, &n_i32, &one,
                self.a.as_ptr(), &n_i32, self.ipiv.as_ptr(),
                x.as_mut_ptr(), &n_i32, &mut info,
            );
        }

        x
    }

    #[cfg(not(feature = "blas"))]
    pub fn factorize(a: Vec<Complex64>, n: usize) -> Result<Self, String> {
        use nalgebra::DMatrix;
        // Store the *original* matrix; `solve` re-factorises via nalgebra.
        // (Storing the LU factors here and re-factorising them in `solve`
        // performed a double decomposition and returned wrong results.)
        let matrix = DMatrix::<Complex64>::from_column_slice(n, n, &a);
        if !matrix.clone().lu().is_invertible() {
            return Err(format!("DenseLu: singular matrix (n={n})"));
        }
        let mut ipiv = vec![0i32; n];
        for i in 0..n {
            ipiv[i] = (i + 1) as i32; // identity permutation (no actual pivot info from nalgebra)
        }
        Ok(DenseLu { a, ipiv, n })
    }

    #[cfg(not(feature = "blas"))]
    pub fn solve(&self, b: &[Complex64]) -> Vec<Complex64> {
        use nalgebra::{DMatrix, DVector};
        let n = self.n;
        let m = DMatrix::<Complex64>::from_column_slice(n, n, &self.a);
        let lu = m.lu();
        let b_vec = DVector::from_vec(b.to_vec());
        match lu.solve(&b_vec) {
            Some(x) => x.as_slice().to_vec(),
            None => vec![Complex64::ZERO; n],
        }
    }
}

#[cfg(feature = "blas")]
extern "C" {
    fn zgetrf_(
        m: *const i32, n: *const i32, a: *mut Complex64,
        lda: *const i32, ipiv: *mut i32, info: *mut i32,
    );
    fn zgetrs_(
        trans: *const i8, n: *const i32, nrhs: *const i32,
        a: *const Complex64, lda: *const i32, ipiv: *const i32,
        b: *mut Complex64, ldb: *const i32, info: *mut i32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_identity() {
        let n = 3;
        let mut a = vec![Complex64::ZERO; n * n];
        for i in 0..n { a[i * n + i] = Complex64::new(1.0, 0.0); }
        let lu = DenseLu::factorize(a, n).unwrap();
        let b = vec![Complex64::new(2.0, 0.0); n];
        let x = lu.solve(&b);
        for i in 0..n {
            assert!((x[i].re - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn small_2x2() {
        let n = 2;
        let a = vec![
            Complex64::new(4.0, 0.0), Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0),
        ];
        let lu = DenseLu::factorize(a, n).unwrap();
        let b = vec![Complex64::new(5.0, 0.0), Complex64::new(10.0, 0.0)];
        let x = lu.solve(&b);
        // Column-major (LAPACK) convention: a = [4,2,1,3] is [[4,1],[2,3]],
        // so the system is 4x+y=5, 2x+3y=10  ⇒  x=0.5, y=3.0.
        assert!((x[0].re - 0.5).abs() < 1e-10, "x[0] = {}", x[0].re);
        assert!((x[1].re - 3.0).abs() < 1e-10, "x[1] = {}", x[1].re);
    }
}
