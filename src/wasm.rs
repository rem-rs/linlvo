//! WASM / wasm-bindgen public interface for linger.
//!
//! Exposes a minimal JS-callable API for solving sparse linear systems in the
//! browser or Node.js.  Only available when compiled with `feature = "wasm"`.
//!
//! **Typical JS usage**:
//! ```js
//! import init, {
//!   WasmCsrMatrix, WasmCgSolver, WasmGmresSolver,
//!   WasmLuSolver, WasmCholeskySolver, WasmLdltSolver, WasmMultifrontalSolver,
//! } from './linger_wasm.js';
//! await init();
//!
//! // Iterative solver (CG)
//! const A = WasmCsrMatrix.from_coo(n, n, rows, cols, vals);
//! const x = new WasmCgSolver(1e-8, 500).solve(A, b);
//!
//! // Direct solver (LU for non-symmetric; Cholesky for SPD; LDLT for symmetric indefinite)
//! const x2 = new WasmLuSolver("rcm").solve(A, b);
//! const x3 = new WasmCholeskySolver("rcm").solve(A, b);
//! const x4 = new WasmLdltSolver("nd").solve(A, b);
//!
//! // Multifrontal LU (large systems, or with BLR compression as preconditioner)
//! const x5 = new WasmMultifrontalSolver("rcm").solve(A, b);
//! const x6 = new WasmMultifrontalSolver("rcm").solve_precond_gmres(A, b, 1e-10, 100, 30);
//!
//! // Multiple right-hand sides — matrix factored once
//! const x_flat = new WasmLdltSolver("rcm").solve_many(A, b_flat, num_rhs);
//! const x_lu   = new WasmLuSolver("rcm").solve_many(A, b_flat, num_rhs);
//! const x_chol = new WasmCholeskySolver("rcm").solve_many(A, b_flat, num_rhs);
//! ```
//!
//! **Analogs**
//!   PETSc.js: https://petsc.org/release/overview/petsc4py/
//!   HYPRE-WASM: (not publicly available)

#[cfg(feature = "wasm")]
mod wasm_impl {
    use wasm_bindgen::prelude::*;

    use crate::{
        iterative::ConjugateGradient,
        sparse::{CooMatrix, CsrMatrix},
        DenseVec, KrylovSolver, SolverParams, VerboseLevel,
        direct::DirectSolver as _,
    };

    /// WASM-exported sparse CSR matrix (f64).
    #[wasm_bindgen]
    pub struct WasmCsrMatrix {
        inner: CsrMatrix<f64>,
    }

    #[wasm_bindgen]
    impl WasmCsrMatrix {
        /// Build from COO triplets.
        ///
        /// `rows`, `cols`, `vals` must all have the same length.
        #[wasm_bindgen(constructor)]
        pub fn from_coo(
            nrows: usize,
            ncols: usize,
            rows:  &[usize],
            cols:  &[usize],
            vals:  &[f64],
        ) -> Result<WasmCsrMatrix, JsValue> {
            if rows.len() != cols.len() || rows.len() != vals.len() {
                return Err(JsValue::from_str("rows, cols, vals must have equal length"));
            }
            let mut coo = CooMatrix::with_capacity(nrows, ncols, rows.len());
            for i in 0..rows.len() {
                coo.push(rows[i], cols[i], vals[i]);
            }
            Ok(WasmCsrMatrix { inner: CsrMatrix::from_coo(&coo) })
        }

        /// Number of rows.
        pub fn nrows(&self) -> usize { self.inner.nrows() }
        /// Number of columns.
        pub fn ncols(&self) -> usize { self.inner.ncols() }
        /// Number of stored nonzeros.
        pub fn nnz(&self) -> usize  { self.inner.nnz() }
    }

    /// WASM-exported Conjugate Gradient solver.
    #[wasm_bindgen]
    pub struct WasmCgSolver {
        rtol:     f64,
        max_iter: usize,
    }

    #[wasm_bindgen]
    impl WasmCgSolver {
        #[wasm_bindgen(constructor)]
        pub fn new(rtol: f64, max_iter: usize) -> WasmCgSolver {
            WasmCgSolver { rtol, max_iter }
        }

        /// Solve `A x = b`.  Returns the solution as a `Float64Array`.
        pub fn solve(&self, a: &WasmCsrMatrix, b: &[f64]) -> Result<Vec<f64>, JsValue> {
            let n = a.inner.nrows();
            if b.len() != n {
                return Err(JsValue::from_str("b.length must equal A.nrows"));
            }
            let b_vec  = DenseVec::from_vec(b.to_vec());
            let mut x  = DenseVec::zeros(n);
            let params = SolverParams {
                rtol:     self.rtol,
                max_iter: self.max_iter,
                verbose:  VerboseLevel::Silent,
                ..Default::default()
            };
            let cg  = ConjugateGradient::<f64>::default();
            cg.solve(&a.inner, None, &b_vec, &mut x, &params)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(x.as_slice().to_vec())
        }
    }

    /// WASM-exported GMRES solver.
    #[wasm_bindgen]
    pub struct WasmGmresSolver {
        rtol:     f64,
        max_iter: usize,
        restart:  usize,
    }

    #[wasm_bindgen]
    impl WasmGmresSolver {
        #[wasm_bindgen(constructor)]
        pub fn new(rtol: f64, max_iter: usize, restart: usize) -> WasmGmresSolver {
            WasmGmresSolver { rtol, max_iter, restart }
        }

        /// Solve `A x = b` (works for non-symmetric systems).
        pub fn solve(&self, a: &WasmCsrMatrix, b: &[f64]) -> Result<Vec<f64>, JsValue> {
            use crate::iterative::Gmres;
            let n = a.inner.nrows();
            if b.len() != n {
                return Err(JsValue::from_str("b.length must equal A.nrows"));
            }
            let b_vec  = DenseVec::from_vec(b.to_vec());
            let mut x  = DenseVec::zeros(n);
            let params = SolverParams {
                rtol:     self.rtol,
                max_iter: self.max_iter,
                verbose:  VerboseLevel::Silent,
                ..Default::default()
            };
            let gmres = Gmres::<f64>::new(self.restart);
            gmres.solve(&a.inner, None, &b_vec, &mut x, &params)
                 .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(x.as_slice().to_vec())
        }
    }

    // ─── Direct solvers ───────────────────────────────────────────────────────

    fn parse_ordering(s: &str) -> crate::direct::ordering::OrderingMethod {
        match s.to_lowercase().as_str() {
            "natural" => crate::direct::ordering::OrderingMethod::Natural,
            "colamd"  => crate::direct::ordering::OrderingMethod::Colamd,
            "nd" | "nodend" | "node_nd" => crate::direct::ordering::OrderingMethod::NodeNd,
            _         => crate::direct::ordering::OrderingMethod::Rcm,
        }
    }

    /// WASM-exported sparse LDLᵀ solver for symmetric (possibly indefinite) matrices.
    ///
    /// Unlike Cholesky, this handles symmetric matrices with negative eigenvalues
    /// (saddle-point problems, symmetric indefinite systems).
    #[wasm_bindgen]
    pub struct WasmLdltSolver {
        ordering: String,
    }

    #[wasm_bindgen]
    impl WasmLdltSolver {
        /// Create a new LDLᵀ solver.
        ///
        /// `ordering`: fill-reducing permutation — `"rcm"` (default), `"colamd"`,
        /// `"nd"` (nested dissection), or `"natural"`.
        /// Use for **symmetric** (not necessarily positive definite) matrices.
        #[wasm_bindgen(constructor)]
        pub fn new(ordering: &str) -> WasmLdltSolver {
            WasmLdltSolver { ordering: ordering.to_string() }
        }

        /// Solve `A x = b` using sparse LDLᵀ factorization.
        pub fn solve(&self, a: &WasmCsrMatrix, b: &[f64]) -> Result<Vec<f64>, JsValue> {
            use crate::direct::{SparseLdlt, DirectSolver, DirectOptions};
            let n = a.inner.nrows();
            if b.len() != n {
                return Err(JsValue::from_str("b.length must equal A.nrows"));
            }
            let b_vec = DenseVec::from_vec(b.to_vec());
            let mut x = DenseVec::zeros(n);
            let mut solver = SparseLdlt::<f64>::new(DirectOptions {
                ordering: parse_ordering(&self.ordering),
                ..Default::default()
            });
            solver.factor(&a.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
            solver.solve(&b_vec, &mut x).map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(x.as_slice().to_vec())
        }

        /// Solve for multiple right-hand sides, returning results packed into a
        /// flat `Float64Array` of length `n * num_rhs` (row-major: first `n`
        /// values are `x₀`, next `n` are `x₁`, etc.).
        ///
        /// `b_flat` must have length `n * num_rhs`.
        pub fn solve_many(
            &self,
            a:      &WasmCsrMatrix,
            b_flat: &[f64],
            num_rhs: usize,
        ) -> Result<Vec<f64>, JsValue> {
            use crate::direct::{SparseLdlt, DirectSolver, DirectOptions};
            let n = a.inner.nrows();
            if b_flat.len() != n * num_rhs {
                return Err(JsValue::from_str(
                    "b_flat.length must equal A.nrows * num_rhs",
                ));
            }
            let bs: Vec<DenseVec<f64>> = (0..num_rhs)
                .map(|i| DenseVec::from_vec(b_flat[i * n..(i + 1) * n].to_vec()))
                .collect();
            let mut xs: Vec<DenseVec<f64>> =
                (0..num_rhs).map(|_| DenseVec::zeros(n)).collect();
            let mut solver = SparseLdlt::<f64>::new(DirectOptions {
                ordering: parse_ordering(&self.ordering),
                ..Default::default()
            });
            solver.factor(&a.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
            solver.solve_multi(&bs, &mut xs)
                  .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let mut out = Vec::with_capacity(n * num_rhs);
            for x in &xs { out.extend_from_slice(x.as_slice()); }
            Ok(out)
        }
    }

    /// WASM-exported sparse LU solver (Gilbert-Peierls, general square systems).
    #[wasm_bindgen]
    pub struct WasmLuSolver {
        ordering: String,
    }

    #[wasm_bindgen]
    impl WasmLuSolver {
        /// Create a new LU solver.
        ///
        /// `ordering`: fill-reducing permutation — `"rcm"` (default), `"colamd"`, or `"natural"`.
        #[wasm_bindgen(constructor)]
        pub fn new(ordering: &str) -> WasmLuSolver {
            WasmLuSolver { ordering: ordering.to_string() }
        }

        /// Solve `A x = b` using sparse LU factorization.
        /// Suitable for general (non-symmetric) square matrices.
        pub fn solve(&self, a: &WasmCsrMatrix, b: &[f64]) -> Result<Vec<f64>, JsValue> {
            use crate::direct::{SparseLu, DirectSolver, DirectOptions};
            let n = a.inner.nrows();
            if b.len() != n {
                return Err(JsValue::from_str("b.length must equal A.nrows"));
            }
            let b_vec = DenseVec::from_vec(b.to_vec());
            let mut x = DenseVec::zeros(n);
            let mut solver = SparseLu::<f64>::new(DirectOptions {
                ordering: parse_ordering(&self.ordering),
                ..Default::default()
            });
            solver.factor(&a.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
            solver.solve(&b_vec, &mut x).map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(x.as_slice().to_vec())
        }

        /// Solve for multiple right-hand sides, returning results packed into a
        /// flat `Float64Array` of length `n * num_rhs` (row-major: first `n`
        /// values are `x₀`, next `n` are `x₁`, etc.).
        ///
        /// `b_flat` must have length `n * num_rhs`.  The matrix is factored once.
        pub fn solve_many(
            &self,
            a:       &WasmCsrMatrix,
            b_flat:  &[f64],
            num_rhs: usize,
        ) -> Result<Vec<f64>, JsValue> {
            use crate::direct::{SparseLu, DirectSolver, DirectOptions};
            let n = a.inner.nrows();
            if b_flat.len() != n * num_rhs {
                return Err(JsValue::from_str(
                    "b_flat.length must equal A.nrows * num_rhs",
                ));
            }
            let bs: Vec<DenseVec<f64>> = (0..num_rhs)
                .map(|i| DenseVec::from_vec(b_flat[i * n..(i + 1) * n].to_vec()))
                .collect();
            let mut xs: Vec<DenseVec<f64>> =
                (0..num_rhs).map(|_| DenseVec::zeros(n)).collect();
            let mut solver = SparseLu::<f64>::new(DirectOptions {
                ordering: parse_ordering(&self.ordering),
                ..Default::default()
            });
            solver.factor(&a.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
            solver.solve_multi(&bs, &mut xs)
                  .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let mut out = Vec::with_capacity(n * num_rhs);
            for x in &xs { out.extend_from_slice(x.as_slice()); }
            Ok(out)
        }
    }

    /// WASM-exported sparse Cholesky solver (left-looking, SPD matrices only).
    #[wasm_bindgen]
    pub struct WasmCholeskySolver {
        ordering: String,
    }

    #[wasm_bindgen]
    impl WasmCholeskySolver {
        /// Create a new Cholesky solver.
        ///
        /// `ordering`: `"rcm"` (default), `"colamd"`, or `"natural"`.
        /// Only use for **symmetric positive definite** matrices.
        #[wasm_bindgen(constructor)]
        pub fn new(ordering: &str) -> WasmCholeskySolver {
            WasmCholeskySolver { ordering: ordering.to_string() }
        }

        /// Solve `A x = b` using sparse Cholesky.
        pub fn solve(&self, a: &WasmCsrMatrix, b: &[f64]) -> Result<Vec<f64>, JsValue> {
            use crate::direct::{SparseCholesky, DirectSolver, DirectOptions};
            let n = a.inner.nrows();
            if b.len() != n {
                return Err(JsValue::from_str("b.length must equal A.nrows"));
            }
            let b_vec = DenseVec::from_vec(b.to_vec());
            let mut x = DenseVec::zeros(n);
            let mut solver = SparseCholesky::<f64>::new(DirectOptions {
                ordering: parse_ordering(&self.ordering),
                ..Default::default()
            });
            solver.factor(&a.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
            solver.solve(&b_vec, &mut x).map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(x.as_slice().to_vec())
        }

        /// Solve for multiple right-hand sides, returning results packed into a
        /// flat `Float64Array` of length `n * num_rhs` (row-major: first `n`
        /// values are `x₀`, next `n` are `x₁`, etc.).
        ///
        /// `b_flat` must have length `n * num_rhs`.  The matrix is factored once.
        /// Only valid for **symmetric positive definite** matrices.
        pub fn solve_many(
            &self,
            a:       &WasmCsrMatrix,
            b_flat:  &[f64],
            num_rhs: usize,
        ) -> Result<Vec<f64>, JsValue> {
            use crate::direct::{SparseCholesky, DirectSolver, DirectOptions};
            let n = a.inner.nrows();
            if b_flat.len() != n * num_rhs {
                return Err(JsValue::from_str(
                    "b_flat.length must equal A.nrows * num_rhs",
                ));
            }
            let bs: Vec<DenseVec<f64>> = (0..num_rhs)
                .map(|i| DenseVec::from_vec(b_flat[i * n..(i + 1) * n].to_vec()))
                .collect();
            let mut xs: Vec<DenseVec<f64>> =
                (0..num_rhs).map(|_| DenseVec::zeros(n)).collect();
            let mut solver = SparseCholesky::<f64>::new(DirectOptions {
                ordering: parse_ordering(&self.ordering),
                ..Default::default()
            });
            solver.factor(&a.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
            solver.solve_multi(&bs, &mut xs)
                  .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let mut out = Vec::with_capacity(n * num_rhs);
            for x in &xs { out.extend_from_slice(x.as_slice()); }
            Ok(out)
        }
    }

    /// WASM-exported multifrontal LU solver with optional BLR compression.
    ///
    /// When used as an exact solver (`blr_tol = 0`), this is the most robust
    /// direct solver in the library.  When BLR is enabled it becomes an
    /// approximate solver suitable as a preconditioner.
    #[wasm_bindgen]
    pub struct WasmMultifrontalSolver {
        ordering:     String,
        blr_min_size: usize,
        blr_tol:      f64,
    }

    #[wasm_bindgen]
    impl WasmMultifrontalSolver {
        /// Create an exact multifrontal solver (BLR disabled).
        ///
        /// `ordering`: `"rcm"` (default), `"colamd"`, or `"natural"`.
        #[wasm_bindgen(constructor)]
        pub fn new(ordering: &str) -> WasmMultifrontalSolver {
            WasmMultifrontalSolver {
                ordering:     ordering.to_string(),
                blr_min_size: usize::MAX,
                blr_tol:      1e-8,
            }
        }

        /// Create an approximate multifrontal solver with BLR compression.
        ///
        /// - `blr_tol`: singular value truncation threshold (e.g. `1e-6`).
        /// - `blr_min_size`: minimum front size to apply BLR (e.g. `32`).
        pub fn with_blr(ordering: &str, blr_tol: f64, blr_min_size: usize)
            -> WasmMultifrontalSolver
        {
            WasmMultifrontalSolver {
                ordering: ordering.to_string(),
                blr_min_size,
                blr_tol,
            }
        }

        fn build_solver(&self) -> crate::direct::MultifrontalLu<f64> {
            use crate::direct::{MultifrontalLu, MultifrontalOptions, DirectOptions};
            MultifrontalLu::<f64>::with_options(MultifrontalOptions {
                base: DirectOptions {
                    ordering: parse_ordering(&self.ordering),
                    ..Default::default()
                },
                blr_min_size: self.blr_min_size,
                blr_tol: self.blr_tol,
            })
        }

        /// Solve `A x = b` using multifrontal LU.
        pub fn solve(&self, a: &WasmCsrMatrix, b: &[f64]) -> Result<Vec<f64>, JsValue> {
            use crate::direct::DirectSolver;
            let n = a.inner.nrows();
            if b.len() != n {
                return Err(JsValue::from_str("b.length must equal A.nrows"));
            }
            let b_vec = DenseVec::from_vec(b.to_vec());
            let mut x = DenseVec::zeros(n);
            let mut solver = self.build_solver();
            solver.factor(&a.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
            solver.solve(&b_vec, &mut x).map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(x.as_slice().to_vec())
        }

        /// Solve using GMRES preconditioned by this multifrontal factorization.
        ///
        /// Useful when `blr_tol > 0` (approximate factorization) or for
        /// ill-conditioned systems where iterative refinement helps.
        ///
        /// - `rtol`: relative tolerance for GMRES convergence.
        /// - `max_iter`: maximum GMRES outer iterations.
        /// - `restart`: GMRES restart parameter (Krylov subspace size).
        pub fn solve_precond_gmres(
            &self,
            a: &WasmCsrMatrix,
            b: &[f64],
            rtol: f64,
            max_iter: usize,
            restart: usize,
        ) -> Result<Vec<f64>, JsValue> {
            use crate::direct::{DirectSolver, DirectSolverPrecond};
            use crate::iterative::Gmres;
            let n = a.inner.nrows();
            if b.len() != n {
                return Err(JsValue::from_str("b.length must equal A.nrows"));
            }
            let b_vec  = DenseVec::from_vec(b.to_vec());
            let mut x  = DenseVec::zeros(n);
            let precond = DirectSolverPrecond::new(self.build_solver(), &a.inner)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let params = SolverParams {
                rtol,
                max_iter,
                verbose: VerboseLevel::Silent,
                ..Default::default()
            };
            Gmres::<f64>::new(restart)
                .solve(&a.inner, Some(&precond), &b_vec, &mut x, &params)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(x.as_slice().to_vec())
        }
    }

    /// Initialise panic hook for better WASM error messages.
    #[wasm_bindgen(start)]
    pub fn wasm_init() {
        #[cfg(feature = "wasm")]
        console_error_panic_hook::set_once();
    }
}

// Re-export at crate level when feature is active.
#[cfg(feature = "wasm")]
pub use wasm_impl::*;
