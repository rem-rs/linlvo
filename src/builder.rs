//! High-level `SolverBuilder` — unified entry point for all linger solvers.
//!
//! `SolverBuilder` provides a fluent, self-documenting API that unifies Krylov
//! iterative and direct solvers under a single interface.  It handles the
//! `analyze → factorize → solve` lifecycle automatically and wires up
//! preconditioners transparently.
//!
//! # Quick examples
//!
//! ```text
//! use linger::builder::{SolverBuilder, SolveMethod, DirectBackend};
//!
//! // Simple direct solve (multifrontal LU, RCM ordering)
//! let x = SolverBuilder::new()
//!     .method(SolveMethod::Direct(DirectBackend::Multifrontal))
//!     .solve(&a, &b)?;
//!
//! // GMRES with ILU(0) preconditioner
//! let x = SolverBuilder::new()
//!     .method(SolveMethod::Gmres { restart: 30 })
//!     .precond(PrecondChoice::Ilu0)
//!     .rtol(1e-10)
//!     .max_iter(200)
//!     .solve(&a, &b)?;
//!
//! // GMRES with exact MultifrontalLu preconditioner
//! let x = SolverBuilder::new()
//!     .method(SolveMethod::Gmres { restart: 20 })
//!     .precond(PrecondChoice::DirectLu(DirectBackend::Multifrontal))
//!     .solve(&a, &b)?;
//!
//! // CG for SPD (default settings)
//! let x = SolverBuilder::new()
//!     .method(SolveMethod::Cg)
//!     .precond(PrecondChoice::Icc0)
//!     .solve(&a, &b)?;
//! ```

use crate::{
    core::{error::SolverError, scalar::Scalar, vector::{DenseVec, Vector}},
    sparse::CsrMatrix,
    iterative::{ConjugateGradient, Gmres, BiCgStab, Minres, Fgmres, Lgmres, Idrs, Tfqmr, PipeCg},
    KrylovSolver, SolverParams, SolverResult, VerboseLevel,
    direct::{DirectOptions, DirectSolver, DirectSolverPrecond, ordering::OrderingMethod},
    core::preconditioner::Preconditioner,
};

// ─── Public enums ─────────────────────────────────────────────────────────────

/// Which backend to use for direct solvers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectBackend {
    /// Gilbert-Peierls sparse LU (general square, dense working storage).
    Lu,
    /// Left-looking sparse Cholesky (symmetric positive definite only).
    Cholesky,
    /// Multifrontal LU with optional BLR compression (general square).
    Multifrontal,
    /// MUMPS-compatible direct path implemented by linger's own multifrontal solver.
    Mumps,
    /// MKL-compatible direct path implemented by linger's own multifrontal solver.
    Mkl,
}

/// External backend families coordinated across subprojects.
///
/// Note: these selections are a contract-level API in C1. Execution remains
/// on the native linger path until per-backend wiring lands in later stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalBackend {
    /// Pure-Rust HYPRE-equivalent track.
    HypreRs,
    /// Pure-Rust PETSc-equivalent track (canonical ID: petsc-rs).
    PetscRs,
    /// Legacy compatibility name for the PETSc-equivalent track.
    PetscFfi,
    /// Compatibility request ID for a MUMPS-shaped direct-solver contract.
    ///
    /// linger resolves this to its native multifrontal replacement path rather
    /// than an external MUMPS dependency.
    Mumps,
    /// Compatibility request ID for an MKL-shaped direct-solver contract.
    ///
    /// linger resolves this to its native multifrontal replacement path rather
    /// than an external MKL dependency.
    Mkl,
}

/// Compile-time capability snapshot for optional solver backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub hypre_rs: bool,
    /// Pure-Rust PETSc-equivalent track (`petsc-rs` feature).
    pub petsc_rs: bool,
    /// Legacy placeholder for C-lib PETSc binding (`petsc-ffi` feature).
    pub petsc_ffi: bool,
    /// Whether the MUMPS-compatibility profile is advertised by this build.
    pub mumps: bool,
    pub mkl: bool,
    pub wasm_target: bool,
}

impl BackendCapabilities {
    /// Detect capabilities from compile-time feature flags.
    pub fn detect() -> Self {
        Self {
            hypre_rs: cfg!(feature = "hypre-rs"),
            petsc_rs: cfg!(feature = "petsc-rs"),
            petsc_ffi: cfg!(feature = "petsc-ffi"),
            mumps: cfg!(feature = "mumps"),
            mkl: cfg!(feature = "mkl"),
            wasm_target: cfg!(target_arch = "wasm32"),
        }
    }

    /// Returns `true` if either the pure-Rust or FFI PETSc track is enabled.
    pub fn has_any_petsc(&self) -> bool {
        self.petsc_rs || self.petsc_ffi
    }
}

/// Effective execution route for the current solve request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveBackend {
    /// Native linger implementation path.
    NativeLinger,
    /// Selected external backend path.
    External(ExternalBackend),
}

/// Result of external-backend selection and fallback policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSelectionReport {
    pub requested: Option<ExternalBackend>,
    pub effective: EffectiveBackend,
    pub capabilities: BackendCapabilities,
    pub note: String,
}

/// Top-level solver selection.
#[derive(Debug, Clone)]
pub enum SolveMethod {
    /// Conjugate Gradient — symmetric positive definite systems.
    Cg,
    /// GMRES with specified restart parameter.
    Gmres {
        /// Krylov subspace restart dimension.  Typical values: 20–50.
        restart: usize,
    },
    /// BiCGSTAB — general non-symmetric, less memory than GMRES.
    BiCgStab,
    /// MINRES — symmetric (possibly indefinite) systems.
    Minres,
    /// FGMRES — flexible GMRES; allows varying preconditioner per iteration.
    Fgmres {
        /// Krylov subspace restart dimension.  Typical values: 20–50.
        restart: usize,
    },
    /// LGMRES — GMRES with Krylov subspace recycling across restarts.
    Lgmres {
        /// Inner Krylov dimension per restart.
        restart: usize,
        /// Number of augmentation vectors retained from previous restarts.
        aug_dim: usize,
    },
    /// IDR(s) — Induced Dimension Reduction; robust for non-symmetric systems.
    Idrs {
        /// Shadow space dimension.  `s=4` is a good default.
        s: usize,
    },
    /// TFQMR — Transpose-Free QMR; smooth convergence for non-symmetric systems.
    Tfqmr,
    /// PIPECG — Pipelined CG; one global dot-product per iter (hides all-reduce latency).
    ///
    /// Drop-in for [`Cg`](SolveMethod::Cg) on SPD systems.  Particularly
    /// beneficial in distributed/MPI environments where global reductions are
    /// expensive.
    PipeCg,
    /// Exact direct solve (no Krylov outer iteration).
    Direct(DirectBackend),
}

/// Preconditioner choice for Krylov methods.
#[derive(Clone)]
pub enum PrecondChoice {
    /// No preconditioner.
    None,
    /// Jacobi (diagonal scaling).
    Jacobi,
    /// ILU(0) — zero-fill incomplete LU.
    Ilu0,
    /// ICC(0) — zero-fill incomplete Cholesky (SPD only).
    Icc0,
    /// Exact direct-solver preconditioner.
    ///
    /// Uses the given direct backend to factor `A` and applies the exact
    /// triangular solve at each preconditioner application.  One Krylov
    /// iteration is usually sufficient (exact precond = direct solve).
    DirectLu(DirectBackend),
    /// AMS auxiliary-space preconditioner for H(curl) / edge-element problems.
    ///
    /// Requires the discrete gradient matrix `G` (n_edges × n_nodes).
    /// Stored as `Arc` so that `PrecondChoice` remains cheaply `Clone`.
    Ams {
        /// Discrete gradient matrix G (n_edges × n_nodes), in `f64`.
        g:      std::sync::Arc<crate::sparse::CsrMatrix<f64>>,
        /// AMS configuration (smoother weight, coarse-solver choice).
        config: crate::precond::ams::AmsConfig,
    },
    /// ADS auxiliary-space preconditioner for H(div) / face-element problems.
    ///
    /// Requires the discrete curl `C` (n_faces × n_edges) and gradient `G`
    /// (n_edges × n_nodes) matrices.
    Ads {
        /// Discrete curl matrix C (n_faces × n_edges), in `f64`.
        c:      std::sync::Arc<crate::sparse::CsrMatrix<f64>>,
        /// Discrete gradient matrix G (n_edges × n_nodes), in `f64`.
        g:      std::sync::Arc<crate::sparse::CsrMatrix<f64>>,
        /// ADS configuration.
        config: crate::precond::ads::AdsConfig,
    },
    /// Two-field FieldSplit preconditioner.
    ///
    /// Decomposes the DOF set at `split` and applies separate sub-preconditioners
    /// to each block.  If `block_triangular = true`, the lower off-diagonal block
    /// is extracted and used for a correction sweep.
    ///
    /// Sub-preconditioners are wrapped in `Arc` so this variant remains `Clone`.
    FieldSplit {
        /// Total number of DOFs.
        n: usize,
        /// Index of first DOF in field 1 (field 0 = 0..split).
        split: usize,
        /// Use lower block-triangular sweep; otherwise block-Jacobi (additive).
        block_triangular: bool,
        /// Sub-preconditioner for field 0 (top-left block).
        p0: std::sync::Arc<dyn crate::core::preconditioner::Preconditioner<Vector = DenseVec<f64>> + Send + Sync>,
        /// Sub-preconditioner for field 1 (bottom-right block).
        p1: std::sync::Arc<dyn crate::core::preconditioner::Preconditioner<Vector = DenseVec<f64>> + Send + Sync>,
    },
}

impl std::fmt::Debug for PrecondChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrecondChoice::None           => write!(f, "None"),
            PrecondChoice::Jacobi         => write!(f, "Jacobi"),
            PrecondChoice::Ilu0           => write!(f, "Ilu0"),
            PrecondChoice::Icc0           => write!(f, "Icc0"),
            PrecondChoice::DirectLu(b)    => write!(f, "DirectLu({b:?})"),
            PrecondChoice::Ams { .. }     => write!(f, "Ams {{ .. }}"),
            PrecondChoice::Ads { .. }     => write!(f, "Ads {{ .. }}"),
            PrecondChoice::FieldSplit { n, split, block_triangular, .. } =>
                write!(f, "FieldSplit {{ n: {n}, split: {split}, block_triangular: {block_triangular} }}"),
        }
    }
}

/// Fill-reducing permutation exposed at builder level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ordering {
    /// No permutation.
    Natural,
    /// Reverse Cuthill-McKee (bandwidth reduction, good for SPD).
    Rcm,
    /// Approximate Minimum Degree via COLAMD (fill reduction, non-symmetric).
    Colamd,
    /// Multilevel Nested Dissection — best fill reduction for large unstructured meshes.
    NodeNd,
}

/// Structured preconditioner diagnostics emitted by [`SolverBuilder`].
#[derive(Debug, Clone)]
pub enum BuilderPrecondReport {
    /// No preconditioner.
    None,
    /// Jacobi preconditioner.
    Jacobi,
    /// ILU(0) preconditioner.
    Ilu0,
    /// ICC(0) preconditioner.
    Icc0,
    /// Exact direct-solver preconditioner.
    DirectLu { backend: DirectBackend },
    /// AMS preconditioner with setup profile.
    Ams(crate::precond::ams::AmsProfile),
    /// ADS preconditioner with setup profile.
    Ads(crate::precond::ads::AdsProfile),
    /// FieldSplit preconditioner.
    FieldSplit {
        /// Boundary index between the two fields.
        split: usize,
        /// Whether a block-triangular sweep was used.
        block_triangular: bool,
    },
}

/// Structured solve diagnostics emitted by [`SolverBuilder::solve_with_report`].
#[derive(Debug, Clone)]
pub struct BuilderSolveReport {
    /// Method used for this solve.
    pub method: SolveMethod,
    /// Selected preconditioner and its setup profile (if available).
    pub precond: BuilderPrecondReport,
    /// Krylov iteration result. `None` for direct solves.
    pub krylov: Option<SolverResult>,
}

// ─── Builder ─────────────────────────────────────────────────────────────────

/// Fluent builder for configuring and running linear solvers.
///
/// All fields have sensible defaults:
/// - method: `Gmres { restart: 30 }`
/// - precond: `None`
/// - ordering: `Rcm`
/// - rtol: `1e-8`
/// - max_iter: `1000`
/// - verbose: silent
#[derive(Debug, Clone)]
pub struct SolverBuilder {
    method:   SolveMethod,
    precond:  PrecondChoice,
    ordering: Ordering,
    rtol:     f64,
    atol:     f64,
    max_iter: usize,
    verbose:  bool,
    external_backend: Option<ExternalBackend>,
}

impl Default for SolverBuilder {
    fn default() -> Self {
        SolverBuilder {
            method:   SolveMethod::Gmres { restart: 30 },
            precond:  PrecondChoice::None,
            ordering: Ordering::Rcm,
            rtol:     1e-8,
            atol:     0.0,
            max_iter: 1000,
            verbose:  false,
            external_backend: None,
        }
    }
}

impl SolverBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self { Self::default() }

    /// Set the solution method.
    pub fn method(mut self, m: SolveMethod) -> Self { self.method = m; self }

    /// Set the preconditioner (for Krylov methods).
    pub fn precond(mut self, p: PrecondChoice) -> Self { self.precond = p; self }

    /// Set the fill-reducing ordering for direct solvers / direct preconditioners.
    pub fn ordering(mut self, o: Ordering) -> Self { self.ordering = o; self }

    /// Set the relative residual tolerance for Krylov convergence.
    pub fn rtol(mut self, v: f64) -> Self { self.rtol = v; self }

    /// Set the absolute residual tolerance for Krylov convergence.
    pub fn atol(mut self, v: f64) -> Self { self.atol = v; self }

    /// Set the maximum number of Krylov iterations.
    pub fn max_iter(mut self, v: usize) -> Self { self.max_iter = v; self }

    /// Enable iteration-by-iteration convergence logging.
    pub fn verbose(mut self) -> Self { self.verbose = true; self }

    /// Apply HPC-oriented Krylov defaults for large sparse problems.
    ///
    /// Sets:
    /// - GMRES restart = 50
    /// - rtol = 1e-8
    /// - atol = 0.0
    /// - max_iter = 400
    pub fn hpc_krylov_defaults(mut self) -> Self {
        self.method = SolveMethod::Gmres { restart: 50 };
        self.rtol = 1e-8;
        self.atol = 0.0;
        self.max_iter = 400;
        self
    }

    /// One-shot HPC preset for H(curl) systems using AMS.
    ///
    /// Uses [`AmsConfig::hpc_default`] and [`Self::hpc_krylov_defaults`].
    pub fn hpc_ams(
        mut self,
        g: std::sync::Arc<crate::sparse::CsrMatrix<f64>>,
    ) -> Self {
        self = self.hpc_krylov_defaults();
        self.precond = PrecondChoice::Ams {
            g,
            config: crate::precond::ams::AmsConfig::hpc_default(),
        };
        self
    }

    /// One-shot HPC preset for H(div) systems using ADS.
    ///
    /// Uses [`AdsConfig::hpc_default`] and [`Self::hpc_krylov_defaults`].
    pub fn hpc_ads(
        mut self,
        c: std::sync::Arc<crate::sparse::CsrMatrix<f64>>,
        g: std::sync::Arc<crate::sparse::CsrMatrix<f64>>,
    ) -> Self {
        self = self.hpc_krylov_defaults();
        self.precond = PrecondChoice::Ads {
            c,
            g,
            config: crate::precond::ads::AdsConfig::hpc_default(),
        };
        self
    }

    /// Request an external backend route.
    ///
    /// In C1 this is an interface-freeze API: unsupported or not-yet-wired
    /// requests deterministically fall back to native linger execution.
    pub fn external_backend(mut self, b: ExternalBackend) -> Self {
        self.external_backend = Some(b);
        self
    }

    /// Return compile-time backend capability information.
    pub fn backend_capabilities() -> BackendCapabilities {
        BackendCapabilities::detect()
    }

    /// Resolve the currently requested external backend into an effective route.
    pub fn backend_selection_report(&self) -> BackendSelectionReport {
        let caps = BackendCapabilities::detect();
        resolve_external_backend(self.external_backend, caps)
    }

    // ─── solve ────────────────────────────────────────────────────────────────

    /// Solve `A x = b`.
    ///
    /// Returns `x` on success, or a [`SolverError`] describing the failure.
    pub fn solve<T: Scalar>(&self, a: &CsrMatrix<T>, b: &DenseVec<T>)
        -> Result<DenseVec<T>, SolverError>
    {
        let n = a.nrows();
        if b.len() != n {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: a.ncols(), rhs_len: b.len(),
            });
        }
        let (x, _) = self.solve_with_report(a, b)?;
        Ok(x)
    }

    /// Solve `A x = b` and return structured diagnostics.
    pub fn solve_with_report<T: Scalar>(
        &self,
        a: &CsrMatrix<T>,
        b: &DenseVec<T>,
    ) -> Result<(DenseVec<T>, BuilderSolveReport), SolverError> {
        let n = a.nrows();
        if b.len() != n {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: a.ncols(), rhs_len: b.len(),
            });
        }
        let mut x = DenseVec::zeros(n);
        let report = self.solve_into_with_report(a, b, &mut x)?;
        Ok((x, report))
    }

    /// Solve `A x = b`, storing the result in the pre-allocated `x`.
    pub fn solve_into<T: Scalar>(
        &self,
        a:  &CsrMatrix<T>,
        b:  &DenseVec<T>,
        x:  &mut DenseVec<T>,
    ) -> Result<(), SolverError> {
        self.solve_into_with_report(a, b, x).map(|_| ())
    }

    /// Solve `A x = b` for multiple right-hand sides.
    ///
    /// For direct methods, the matrix is factored **once** and the triangular
    /// solve is applied to every column of `bs`.  For Krylov methods each
    /// right-hand side runs an independent iteration starting from `xs[i]` as
    /// the initial guess.
    ///
    /// Returns a `Vec<DenseVec<T>>` with one solution per right-hand side.
    ///
    /// # Errors
    /// Returns the first error encountered, leaving already-computed solutions
    /// in the returned vector.
    pub fn solve_many<T: Scalar>(
        &self,
        a:  &CsrMatrix<T>,
        bs: &[DenseVec<T>],
    ) -> Result<Vec<DenseVec<T>>, SolverError> {
        let n = a.nrows();
        if bs.iter().any(|b| b.len() != n) {
            return Err(SolverError::DimensionMismatch {
                op_rows: n, op_cols: a.ncols(), rhs_len: bs.first().map_or(0, |b| b.len()),
            });
        }
        let mut xs: Vec<DenseVec<T>> = (0..bs.len()).map(|_| DenseVec::zeros(n)).collect();
        self.solve_many_into(a, bs, &mut xs)?;
        Ok(xs)
    }

    /// Solve `A x = b` for multiple right-hand sides, writing results into `xs`.
    ///
    /// For direct backends, the matrix is factored once and `solve_multi` is
    /// called to reuse the factors.  For Krylov backends each system is solved
    /// independently (parallel execution not assumed here — callers may call
    /// concurrently if needed).
    pub fn solve_many_into<T: Scalar>(
        &self,
        a:  &CsrMatrix<T>,
        bs: &[DenseVec<T>],
        xs: &mut [DenseVec<T>],
    ) -> Result<(), SolverError> {
        if bs.len() != xs.len() {
            return Err(SolverError::DimensionMismatch {
                op_rows: bs.len(), op_cols: 1, rhs_len: xs.len(),
            });
        }
        match &self.method {
            SolveMethod::Direct(backend) => self.run_direct_many(backend, a, bs, xs),
            _ => {
                for (b, x) in bs.iter().zip(xs.iter_mut()) {
                    self.run_krylov_into(a, b, x)?;
                }
                Ok(())
            }
        }
    }

    // ─── internal: direct many ────────────────────────────────────────────────

    fn run_direct_many<T: Scalar>(
        &self,
        backend: &DirectBackend,
        a:  &CsrMatrix<T>,
        bs: &[DenseVec<T>],
        xs: &mut [DenseVec<T>],
    ) -> Result<(), SolverError> {
        use crate::direct::{SparseLu, SparseCholesky, MultifrontalLu, MultifrontalOptions,
                            MumpsSolver, MklSolver};
        macro_rules! factor_and_solve_many {
            ($solver:expr) => {{
                let mut s = $solver;
                s.factor(a)?;
                s.solve_multi(bs, xs)
            }};
        }
        match backend {
            DirectBackend::Lu =>
                factor_and_solve_many!(SparseLu::<T>::new(self.direct_opts())),
            DirectBackend::Cholesky =>
                factor_and_solve_many!(SparseCholesky::<T>::new(self.direct_opts())),
            DirectBackend::Multifrontal =>
                factor_and_solve_many!(MultifrontalLu::<T>::with_options(
                    MultifrontalOptions { base: self.direct_opts(), ..Default::default() }
                )),
            DirectBackend::Mumps =>
                factor_and_solve_many!(MumpsSolver::<T>::with_options(self.direct_opts())),
            DirectBackend::Mkl =>
                factor_and_solve_many!(MklSolver::<T>::with_options(self.direct_opts())),
        }
    }

    // ─── internal: krylov single (no report) ─────────────────────────────────

    fn run_krylov_into<T: Scalar>(
        &self,
        a:      &CsrMatrix<T>,
        b:      &DenseVec<T>,
        x:      &mut DenseVec<T>,
    ) -> Result<(), SolverError> {
        self.run_krylov_with_report(a, b, x).map(|_| ())
    }

    /// Solve `A x = b` into `x` and return structured diagnostics.
    pub fn solve_into_with_report<T: Scalar>(
        &self,
        a:  &CsrMatrix<T>,
        b:  &DenseVec<T>,
        x:  &mut DenseVec<T>,
    ) -> Result<BuilderSolveReport, SolverError> {
        let backend_report = self.backend_selection_report();
        if self.verbose && self.external_backend.is_some() {
            eprintln!("[linger::SolverBuilder] {}", backend_report.note);
        }
        match &self.method {
            SolveMethod::Direct(backend) => {
                self.run_direct(backend, a, b, x)?;
                Ok(BuilderSolveReport {
                    method: self.method.clone(),
                    precond: BuilderPrecondReport::None,
                    krylov: None,
                })
            }
            _ => self.run_krylov_with_report(a, b, x),
        }
    }

    // ─── internal: direct ─────────────────────────────────────────────────────

    fn direct_opts(&self) -> DirectOptions {
        DirectOptions {
            ordering: match self.ordering {
                Ordering::Natural => OrderingMethod::Natural,
                Ordering::Rcm     => OrderingMethod::Rcm,
                Ordering::Colamd  => OrderingMethod::Colamd,
                Ordering::NodeNd  => OrderingMethod::NodeNd,
            },
            ..Default::default()
        }
    }

    fn krylov_params(&self) -> SolverParams {
        SolverParams {
            rtol:     self.rtol,
            atol:     self.atol,
            max_iter: self.max_iter,
            verbose:  if self.verbose { VerboseLevel::Iterations } else { VerboseLevel::Silent },
            ..Default::default()
        }
    }

    fn run_direct<T: Scalar>(
        &self,
        backend: &DirectBackend,
        a: &CsrMatrix<T>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
    ) -> Result<(), SolverError> {
        use crate::direct::{SparseLu, SparseCholesky, MultifrontalLu, MultifrontalOptions, MumpsSolver, MklSolver};
        match backend {
            DirectBackend::Lu => {
                let mut s = SparseLu::<T>::new(self.direct_opts());
                s.factor(a)?;
                s.solve(b, x)
            }
            DirectBackend::Cholesky => {
                let mut s = SparseCholesky::<T>::new(self.direct_opts());
                s.factor(a)?;
                s.solve(b, x)
            }
            DirectBackend::Multifrontal => {
                let mut s = MultifrontalLu::<T>::with_options(MultifrontalOptions {
                    base: self.direct_opts(),
                    ..Default::default()
                });
                s.factor(a)?;
                s.solve(b, x)
            }
            DirectBackend::Mumps => {
                let mut s = MumpsSolver::<T>::with_options(self.direct_opts());
                s.factor(a)?;
                s.solve(b, x)
            }
            DirectBackend::Mkl => {
                let mut s = MklSolver::<T>::with_options(self.direct_opts());
                s.factor(a)?;
                s.solve(b, x)
            }
        }
    }

    // ─── internal: Krylov ─────────────────────────────────────────────────────

    fn run_krylov_with_report<T: Scalar>(
        &self,
        a: &CsrMatrix<T>,
        b: &DenseVec<T>,
        x: &mut DenseVec<T>,
    ) -> Result<BuilderSolveReport, SolverError> {
        let params = self.krylov_params();
        match &self.precond {
            PrecondChoice::None => {
                let result = self.dispatch_krylov_result::<T>(a, None, b, x, &params)?;
                Ok(BuilderSolveReport {
                    method: self.method.clone(),
                    precond: BuilderPrecondReport::None,
                    krylov: Some(result),
                })
            }
            PrecondChoice::Jacobi => {
                let p = crate::JacobiPrecond::from_csr(a)?;
                let result = self.dispatch_krylov_result(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, &params)?;
                Ok(BuilderSolveReport {
                    method: self.method.clone(),
                    precond: BuilderPrecondReport::Jacobi,
                    krylov: Some(result),
                })
            }
            PrecondChoice::Ilu0 => {
                let p = crate::Ilu0Precond::from_csr(a)?;
                let result = self.dispatch_krylov_result(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, &params)?;
                Ok(BuilderSolveReport {
                    method: self.method.clone(),
                    precond: BuilderPrecondReport::Ilu0,
                    krylov: Some(result),
                })
            }
            PrecondChoice::Icc0 => {
                let p = crate::Icc0Precond::from_csr(a)?;
                let result = self.dispatch_krylov_result(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, &params)?;
                Ok(BuilderSolveReport {
                    method: self.method.clone(),
                    precond: BuilderPrecondReport::Icc0,
                    krylov: Some(result),
                })
            }
            PrecondChoice::DirectLu(backend) => {
                let result = self.run_krylov_direct_precond(backend, a, b, x, &params)?;
                Ok(BuilderSolveReport {
                    method: self.method.clone(),
                    precond: BuilderPrecondReport::DirectLu { backend: backend.clone() },
                    krylov: Some(result),
                })
            }
            PrecondChoice::Ams { g, config } => {
                let g_t = cast_csr_f64_to::<T>(g);
                let p = crate::precond::ams::AmsPrecond::new(a, &g_t, config.clone())?;
                if self.verbose {
                    print_ams_profile_summary(p.profile());
                }
                let result = self.dispatch_krylov_result(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, &params)?;
                Ok(BuilderSolveReport {
                    method: self.method.clone(),
                    precond: BuilderPrecondReport::Ams(p.profile().clone()),
                    krylov: Some(result),
                })
            }
            PrecondChoice::Ads { c, g, config } => {
                let c_t = cast_csr_f64_to::<T>(c);
                let g_t = cast_csr_f64_to::<T>(g);
                let p = crate::precond::ads::AdsPrecond::new(a, &c_t, &g_t, config.clone())?;
                if self.verbose {
                    print_ads_profile_summary(p.profile());
                }
                let result = self.dispatch_krylov_result(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, &params)?;
                Ok(BuilderSolveReport {
                    method: self.method.clone(),
                    precond: BuilderPrecondReport::Ads(p.profile().clone()),
                    krylov: Some(result),
                })
            }
            PrecondChoice::FieldSplit { n, split, block_triangular, p0, p1 } => {
                use crate::precond::fieldsplit::{FieldSplitPrecond, SplitMode};
                // FieldSplitPrecond is fixed to f64 due to trait-object storage.
                // Attempt f64 downcast; return error for other types.
                let a_f64 = cast_csr_to_f64(a).ok_or_else(|| SolverError::PrecondSetupFailed {
                    reason: "PrecondChoice::FieldSplit only supports f64 matrices".into(),
                })?;
                let mode = if *block_triangular { SplitMode::BlockTriangular } else { SplitMode::BlockJacobi };
                // Wrap Arc preconditioners as Box — clone the Arc and wrap.
                let p0_box: Box<dyn Preconditioner<Vector = DenseVec<f64>> + Send + Sync> =
                    Box::new(ArcPrecond(std::sync::Arc::clone(p0)));
                let p1_box: Box<dyn Preconditioner<Vector = DenseVec<f64>> + Send + Sync> =
                    Box::new(ArcPrecond(std::sync::Arc::clone(p1)));
                let p = if *block_triangular {
                    FieldSplitPrecond::from_matrix(&a_f64, *split, mode, p0_box, p1_box)
                } else {
                    FieldSplitPrecond::new(*n, *split, mode, p0_box, p1_box)
                };
                // Run in f64 only.
                let a64 = a_f64;
                let b64 = cast_dense_to_f64(b).ok_or_else(|| SolverError::PrecondSetupFailed {
                    reason: "PrecondChoice::FieldSplit: scalar mismatch".into(),
                })?;
                let mut x64 = DenseVec::<f64>::zeros(x.len());
                let result = {
                    let params64 = params.clone();
                    self.dispatch_krylov_result(&a64, Some(&p as &dyn Preconditioner<Vector=DenseVec<f64>>), &b64, &mut x64, &params64)?
                };
                copy_f64_to_dense(x, &x64);
                Ok(BuilderSolveReport {
                    method: self.method.clone(),
                    precond: BuilderPrecondReport::FieldSplit { split: *split, block_triangular: *block_triangular },
                    krylov: Some(result),
                })
            }
        }
    }

    fn dispatch_krylov_result<T: Scalar>(
        &self,
        a:       &CsrMatrix<T>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<T>>>,
        b:       &DenseVec<T>,
        x:       &mut DenseVec<T>,
        params:  &SolverParams,
    ) -> Result<SolverResult, SolverError> {
        match &self.method {
            SolveMethod::Cg => {
                ConjugateGradient::<T>::default().solve(a, precond, b, x, params)
            }
            SolveMethod::Gmres { restart } => {
                Gmres::<T>::new(*restart).solve(a, precond, b, x, params)
            }
            SolveMethod::BiCgStab => {
                BiCgStab::<T>::default().solve(a, precond, b, x, params)
            }
            SolveMethod::Minres => {
                Minres::<T>::default().solve(a, precond, b, x, params)
            }
            SolveMethod::Fgmres { restart } => {
                Fgmres::<T>::new(*restart).solve(a, precond, b, x, params)
            }
            SolveMethod::Lgmres { restart, aug_dim } => {
                Lgmres::<T>::new(*restart, *aug_dim).solve(a, precond, b, x, params)
            }
            SolveMethod::Idrs { s } => {
                Idrs::<T>::new(*s).solve(a, precond, b, x, params)
            }
            SolveMethod::Tfqmr => {
                Tfqmr::<T>::default().solve(a, precond, b, x, params)
            }
            SolveMethod::PipeCg => {
                PipeCg::<T>::default().solve(a, precond, b, x, params)
            }
            SolveMethod::Direct(_) => unreachable!(),
        }
    }

    fn run_krylov_direct_precond<T: Scalar>(
        &self,
        backend: &DirectBackend,
        a:       &CsrMatrix<T>,
        b:       &DenseVec<T>,
        x:       &mut DenseVec<T>,
        params:  &SolverParams,
    ) -> Result<SolverResult, SolverError> {
        use crate::direct::{SparseLu, SparseCholesky, MultifrontalLu, MultifrontalOptions, MumpsSolver, MklSolver};
        match backend {
            DirectBackend::Lu => {
                let s = SparseLu::<T>::new(self.direct_opts());
                let p = DirectSolverPrecond::new(s, a)?;
                self.dispatch_krylov_result(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, params)
            }
            DirectBackend::Cholesky => {
                let s = SparseCholesky::<T>::new(self.direct_opts());
                let p = DirectSolverPrecond::new(s, a)?;
                self.dispatch_krylov_result(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, params)
            }
            DirectBackend::Multifrontal => {
                let s = MultifrontalLu::<T>::with_options(MultifrontalOptions {
                    base: self.direct_opts(),
                    ..Default::default()
                });
                let p = DirectSolverPrecond::new(s, a)?;
                self.dispatch_krylov_result(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, params)
            }
            DirectBackend::Mumps => {
                let s = MumpsSolver::<T>::with_options(self.direct_opts());
                let p = DirectSolverPrecond::new(s, a)?;
                self.dispatch_krylov_result(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, params)
            }
            DirectBackend::Mkl => {
                let s = MklSolver::<T>::with_options(self.direct_opts());
                let p = DirectSolverPrecond::new(s, a)?;
                self.dispatch_krylov_result(a, Some(&p as &dyn Preconditioner<Vector=DenseVec<T>>), b, x, params)
            }
        }
    }
}

// ─── Convenience functions ────────────────────────────────────────────────────

/// Solve `A x = b` with automatic method selection.
///
/// Uses Cholesky if `spd_hint` is true, otherwise MultifrontalLu.  Intended as
/// the simplest possible entry point for users who don't want to configure a
/// full `SolverBuilder`.
///
/// For more control use [`SolverBuilder`] directly.
pub fn solve_auto<T: Scalar>(
    a:        &CsrMatrix<T>,
    b:        &DenseVec<T>,
    spd_hint: bool,
) -> Result<DenseVec<T>, SolverError> {
    let backend = if spd_hint { DirectBackend::Cholesky } else { DirectBackend::Multifrontal };
    SolverBuilder::new()
        .method(SolveMethod::Direct(backend))
        .solve(a, b)
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Convert a `CsrMatrix<f64>` to `CsrMatrix<T>` by casting each value.
///
/// Used by the AMS/ADS builder arms to convert user-supplied `f64` matrices
/// into the solver's working scalar type `T` (`f32` or `f64`).
fn cast_csr_f64_to<T: Scalar>(m: &crate::sparse::CsrMatrix<f64>) -> crate::sparse::CsrMatrix<T> {
    crate::sparse::CsrMatrix::from_raw(
        m.nrows(),
        m.ncols(),
        m.row_ptr().to_vec(),
        m.col_idx().to_vec(),
        m.values().iter().map(|&v| T::from_f64(v)).collect(),
    )
}

/// Try to interpret a `CsrMatrix<T>` as `CsrMatrix<f64>`.
/// Returns `None` if `T` is not `f64`.
fn cast_csr_to_f64<T: Scalar>(m: &crate::sparse::CsrMatrix<T>) -> Option<crate::sparse::CsrMatrix<f64>> {
    let vals: Option<Vec<f64>> = m.values().iter()
        .map(|v| num_traits::ToPrimitive::to_f64(v))
        .collect();
    vals.map(|vs| crate::sparse::CsrMatrix::from_raw(
        m.nrows(),
        m.ncols(),
        m.row_ptr().to_vec(),
        m.col_idx().to_vec(),
        vs,
    ))
}

fn cast_dense_to_f64<T: Scalar>(v: &DenseVec<T>) -> Option<DenseVec<f64>> {
    let vals: Option<Vec<f64>> = v.as_slice().iter()
        .map(|x| num_traits::ToPrimitive::to_f64(x))
        .collect();
    vals.map(DenseVec::from_vec)
}

fn copy_f64_to_dense<T: Scalar>(dst: &mut DenseVec<T>, src: &DenseVec<f64>) {
    for (d, &s) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
        *d = T::from_f64(s);
    }
}

/// Wrapper that lets an `Arc<dyn Preconditioner>` implement `Preconditioner`
/// (required because `Box<dyn Preconditioner>` isn't `Clone`, but `Arc` is).
struct ArcPrecond<V: crate::core::vector::Vector>(
    std::sync::Arc<dyn Preconditioner<Vector = V> + Send + Sync>,
);

impl<V: crate::core::vector::Vector> Preconditioner for ArcPrecond<V> {
    type Vector = V;
    fn apply_precond(&self, x: &V, y: &mut V) { self.0.apply_precond(x, y) }
}

fn print_ams_profile_summary(profile: &crate::precond::ams::AmsProfile) {
    println!(
        "[AMS] edges={} nodes={} nnz(A)={} nnz(G)={} nnz(G^T A G)={} aux={}",
        profile.n_edges,
        profile.n_nodes,
        profile.a_nnz,
        profile.g_nnz,
        profile.a_node_nnz,
        format_aux_solver_profile(&profile.node_solver),
    );
}

fn print_ads_profile_summary(profile: &crate::precond::ads::AdsProfile) {
    println!(
        "[ADS] faces={} edges={} nodes={} nnz(A)={} nnz(C)={} nnz(G)={} nnz(C^T A C)={} nnz(G^T A_e G)={} edge_aux={} node_aux={}",
        profile.n_faces,
        profile.n_edges,
        profile.n_nodes,
        profile.a_nnz,
        profile.c_nnz,
        profile.g_nnz,
        profile.a_edge_nnz,
        profile.a_node_nnz,
        format_aux_solver_profile(&profile.edge_solver),
        format_aux_solver_profile(&profile.node_solver),
    );
}

fn format_aux_solver_profile(profile: &crate::precond::ams::AuxSolverProfile) -> String {
    match profile {
        crate::precond::ams::AuxSolverProfile::Amg(amg) => {
            format!(
                "AMG(levels={}, op_cx={:.2}, grid_cx={:.2})",
                amg.n_levels, amg.operator_complexity, amg.grid_complexity
            )
        }
        crate::precond::ams::AuxSolverProfile::Ilu0 { n, nnz } => {
            format!("ILU0(n={}, nnz={})", n, nnz)
        }
    }
}

fn resolve_external_backend(
    requested: Option<ExternalBackend>,
    caps: BackendCapabilities,
) -> BackendSelectionReport {
    match requested {
        None => BackendSelectionReport {
            requested: None,
            effective: EffectiveBackend::NativeLinger,
            capabilities: caps,
            note: "No external backend requested; using native linger path.".to_string(),
        },
        Some(ExternalBackend::HypreRs) => {
            if caps.hypre_rs {
                BackendSelectionReport {
                    requested,
                    effective: EffectiveBackend::NativeLinger,
                    capabilities: caps,
                    note: "Requested hypre-rs. Feature is enabled; C1 keeps execution on native linger while parity wiring is staged in later milestones.".to_string(),
                }
            } else {
                BackendSelectionReport {
                    requested,
                    effective: EffectiveBackend::NativeLinger,
                    capabilities: caps,
                    note: "Requested hypre-rs, but feature hypre-rs is disabled. Falling back to native linger path.".to_string(),
                }
            }
        }
        Some(ExternalBackend::PetscRs) | Some(ExternalBackend::PetscFfi) => {
            if caps.wasm_target {
                BackendSelectionReport {
                    requested,
                    effective: EffectiveBackend::NativeLinger,
                    capabilities: caps,
                    note: "Requested petsc-rs on wasm32 target. External solver backends are unsupported on wasm; using native linger path.".to_string(),
                }
            } else if caps.petsc_rs {
                BackendSelectionReport {
                    requested,
                    effective: EffectiveBackend::NativeLinger,
                    capabilities: caps,
                    note: "Requested petsc-rs. Capability is enabled; execution remains on native linger until external backend wiring is completed.".to_string(),
                }
            } else {
                BackendSelectionReport {
                    requested,
                    effective: EffectiveBackend::NativeLinger,
                    capabilities: caps,
                    note: "Requested petsc-rs, but capability is disabled. Falling back to native linger path.".to_string(),
                }
            }
        }
        Some(ExternalBackend::Mumps) => {
            if caps.wasm_target {
                BackendSelectionReport {
                    requested,
                    effective: EffectiveBackend::NativeLinger,
                    capabilities: caps,
                    note: "Requested mumps on wasm32 target. linger exposes this as a native compatibility path, but direct native backends are unavailable on wasm; using baseline native linger path.".to_string(),
                }
            } else if caps.mumps {
                BackendSelectionReport {
                    requested,
                    effective: EffectiveBackend::NativeLinger,
                    capabilities: caps,
                    note: "Requested mumps. Feature is enabled; linger provides a MUMPS-compatible contract via its native multifrontal replacement path (SolverBuilder::Direct(DirectBackend::Mumps)).".to_string(),
                }
            } else {
                BackendSelectionReport {
                    requested,
                    effective: EffectiveBackend::NativeLinger,
                    capabilities: caps,
                    note: "Requested mumps, but the compatibility flag is disabled. Falling back to native linger path; linger does not depend on an external MUMPS backend.".to_string(),
                }
            }
        }
        Some(ExternalBackend::Mkl) => {
            if caps.wasm_target {
                BackendSelectionReport {
                    requested,
                    effective: EffectiveBackend::NativeLinger,
                    capabilities: caps,
                    note: "Requested mkl on wasm32 target. linger exposes this as a native compatibility path, but direct native backends are unavailable on wasm; using baseline native linger path.".to_string(),
                }
            } else if caps.mkl {
                BackendSelectionReport {
                    requested,
                    effective: EffectiveBackend::NativeLinger,
                    capabilities: caps,
                    note: "Requested mkl. Feature is enabled; linger provides an MKL-compatible contract via its native multifrontal replacement path (SolverBuilder::Direct(DirectBackend::Mkl)).".to_string(),
                }
            } else {
                BackendSelectionReport {
                    requested,
                    effective: EffectiveBackend::NativeLinger,
                    capabilities: caps,
                    note: "Requested mkl, but the compatibility flag is disabled. Falling back to native linger path; linger does not depend on an external MKL backend.".to_string(),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_report_defaults_to_native() {
        let rep = SolverBuilder::new().backend_selection_report();
        assert_eq!(rep.requested, None);
        assert_eq!(rep.effective, EffectiveBackend::NativeLinger);
    }

    #[test]
    fn backend_report_hypre_rs_with_feature_off_falls_back() {
        let rep = SolverBuilder::new()
            .external_backend(ExternalBackend::HypreRs)
            .backend_selection_report();
        if !rep.capabilities.hypre_rs {
            assert_eq!(rep.effective, EffectiveBackend::NativeLinger);
            assert!(rep.note.contains("hypre-rs"));
        }
    }

    #[test]
    fn backend_report_petsc_feature_off_falls_back() {
        let rep = SolverBuilder::new()
            .external_backend(ExternalBackend::PetscRs)
            .backend_selection_report();
        if !rep.capabilities.petsc_rs {
            assert_eq!(rep.effective, EffectiveBackend::NativeLinger);
            assert!(rep.note.contains("petsc-rs"));
        }
    }

    #[test]
    fn backend_report_mumps_feature_off_falls_back() {
        let rep = SolverBuilder::new()
            .external_backend(ExternalBackend::Mumps)
            .backend_selection_report();
        assert_eq!(rep.effective, EffectiveBackend::NativeLinger);
        assert!(rep.note.contains("mumps"));
    }

    #[test]
    fn direct_backend_mumps_solves_system() {
        use crate::sparse::CooMatrix;

        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.push(0, 0, 2.0);
        coo.push(0, 1, -1.0);
        coo.push(1, 0, -1.0);
        coo.push(1, 1, 2.0);
        coo.push(1, 2, -1.0);
        coo.push(2, 1, -1.0);
        coo.push(2, 2, 2.0);
        let a = CsrMatrix::from_coo(&coo);

        let b = DenseVec::from_vec(vec![1.0, 0.0, 1.0]);
        let x = SolverBuilder::new()
            .method(SolveMethod::Direct(DirectBackend::Mumps))
            .solve(&a, &b)
            .unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
        assert!((x[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn direct_backend_mkl_solves_system() {
        use crate::sparse::CooMatrix;

        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.push(0, 0, 2.0);
        coo.push(0, 1, -1.0);
        coo.push(1, 0, -1.0);
        coo.push(1, 1, 2.0);
        coo.push(1, 2, -1.0);
        coo.push(2, 1, -1.0);
        coo.push(2, 2, 2.0);
        let a = CsrMatrix::from_coo(&coo);

        let b = DenseVec::from_vec(vec![1.0, 0.0, 1.0]);
        let x = SolverBuilder::new()
            .method(SolveMethod::Direct(DirectBackend::Mkl))
            .solve(&a, &b)
            .unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
        assert!((x[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn backend_report_mkl_feature_off_falls_back() {
        let rep = SolverBuilder::new()
            .external_backend(ExternalBackend::Mkl)
            .backend_selection_report();
        assert_eq!(rep.effective, EffectiveBackend::NativeLinger);
        assert!(rep.note.contains("mkl"));
    }
}
