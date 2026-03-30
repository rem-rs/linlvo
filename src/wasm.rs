//! WASM / wasm-bindgen public interface for linger.
//!
//! Exposes a minimal JS-callable API for solving sparse linear systems in the
//! browser or Node.js.  Only available when compiled with `feature = "wasm"`.
//!
//! **Typical JS usage**:
//! ```js
//! import init, { WasmCsrMatrix, WasmCgSolver } from './linger_wasm.js';
//! await init();
//!
//! const A = WasmCsrMatrix.from_coo(n, n, rows, cols, vals);
//! const solver = new WasmCgSolver(1e-8, 500);
//! const x = solver.solve(A, b);   // Float64Array
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
