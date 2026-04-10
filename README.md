# linger

Pure-Rust sparse linear system solver library for FEA (Finite Element Analysis).

Provides Krylov iterative methods, algebraic multigrid (AMG), and a rich preconditioner library. The core solver layer compiles to WebAssembly.

---

## Feature flags

| Flag | Default | Effect |
|------|---------|--------|
| `rayon` | **on** | Parallel SpMV, AXPY, dot, norm2, and AMG setup phases via Rayon |
| `wasm` | off | Enables `wasm-bindgen` JS interface (`WasmCsrMatrix`, `WasmCgSolver`, `WasmGmresSolver`) |
| `mpi` | off | Placeholder for distributed-memory support (not yet implemented) |
| `hypre-ffi` / `petsc-ffi` / `mumps` / `mkl` | off | Placeholders for FFI backends (Sprint 6) |

```toml
# Cargo.toml — add linger as a dependency
linger = { path = ".", features = ["rayon"] }

# Without parallelism (e.g., embedding in a single-threaded context)
linger = { path = ".", default-features = false }
```

---

## Quick start

```rust
use linger::{
    iterative::ConjugateGradient,
    sparse::{CooMatrix, CsrMatrix},
    DenseVec, KrylovSolver, SolverParams, VerboseLevel,
};

// 1. Assemble matrix in COO format, then convert to CSR.
let n = 100;
let mut coo: CooMatrix<f64> = CooMatrix::new(n, n);
for i in 0..n {
    coo.push(i, i, 2.0);
    if i > 0     { coo.push(i, i - 1, -1.0); }
    if i < n - 1 { coo.push(i, i + 1, -1.0); }
}
let a = CsrMatrix::from_coo(&coo);

// 2. Set up right-hand side and initial guess.
let b = DenseVec::from_vec(vec![1.0_f64; n]);
let mut x = DenseVec::zeros(n);

// 3. Solve with Conjugate Gradient.
let params = SolverParams {
    rtol: 1e-8,
    max_iter: 500,
    verbose: VerboseLevel::Silent,
    ..Default::default()
};
let result = ConjugateGradient::<f64>::default()
    .solve(&a, None, &b, &mut x, &params)
    .unwrap();

assert!(result.converged);
println!("Converged in {} iterations, residual = {:.3e}",
    result.iterations, result.final_residual);
```

### With AMG preconditioning

```rust
use linger::{
    amg::{AmgConfig, AmgHierarchy, AmgPrecond},
    iterative::ConjugateGradient,
    KrylovSolver, SolverParams,
};

let config = AmgConfig { coarse_threshold: 10, ..Default::default() };
let hier   = AmgHierarchy::build(a.clone(), config);
let precond = AmgPrecond::new(hier);

let mut x = DenseVec::zeros(n);
let result = ConjugateGradient::<f64>::default()
    .solve(&a, Some(&precond), &b, &mut x, &params)
    .unwrap();
```

### AMS/ADS parameter sweep (CSV)

Use the tuning example to generate structured CSV rows for AMS/ADS settings
(`theta`, `coarse_threshold`, `restart`) including convergence, iterations,
residual, elapsed time, and AMG complexity metrics.

```bash
# AMS only
cargo run --example ex07_ams_ads_tuning -- --mode ams

# ADS only
cargo run --example ex07_ams_ads_tuning -- --mode ads

# Both families
cargo run --example ex07_ams_ads_tuning -- --mode both
```

Pipe the output to a file for post-processing:

```bash
cargo run --example ex07_ams_ads_tuning -- --mode both > ams_ads_sweep.csv
```

### With `nalgebra_sparse::CsrMatrix`

On native targets, `nalgebra_sparse::CsrMatrix<T>` implements `LinearOperator`
directly, so it can be passed to solvers without a wrapper.

Note: nalgebra is an integration layer in this project, not an algorithmic backend.
The Krylov solvers, preconditioners, AMG, and the default vector/matrix path are
still implemented on top of linger's own `DenseVec` and sparse matrix types.
The nalgebra integration only allows an already-assembled
`nalgebra_sparse::CsrMatrix<T>` to be used as a `LinearOperator` on native
builds. The wasm path does not use nalgebra; it uses linger's own COO/CSR types
through `WasmCsrMatrix`.

```rust
use linger::{DenseVec, KrylovSolver, SolverParams, iterative::ConjugateGradient};
use nalgebra_sparse::CooMatrix;

let n = 8;
let mut coo = CooMatrix::<f64>::new(n, n);
for i in 0..n {
    coo.push(i, i, 2.0);
    if i > 0 {
        coo.push(i, i - 1, -1.0);
    }
    if i + 1 < n {
        coo.push(i, i + 1, -1.0);
    }
}

let a = nalgebra_sparse::CsrMatrix::from(&coo);
let b = DenseVec::from_vec(vec![1.0; n]);
let mut x = DenseVec::zeros(n);

let result = ConjugateGradient::<f64>::default()
    .solve(&a, None, &b, &mut x, &SolverParams::default())
    .unwrap();

assert!(result.converged);
```

---

## Module map

```
linger/
├── core/
│   ├── scalar.rs          Scalar trait (f32/f64) + ComplexScalar trait (Complex<f32/f64>)
│   ├── vector.rs          Vector trait + DenseVec<T>
│   ├── operator.rs        LinearOperator trait + TransposeOperator trait
│   ├── preconditioner.rs  Preconditioner trait
│   ├── solver.rs          KrylovSolver trait, SolverParams, SolverResult
│   └── error.rs           SolverError enum
├── sparse/
│   ├── coo.rs             CooMatrix<T>  — assembly format
│   ├── csr.rs             CsrMatrix<T>  — primary operator (impl LinearOperator + TransposeOperator)
│   ├── csc.rs             CscMatrix<T>  — obtained via csr.transpose()
│   ├── bsr.rs             BsrMatrix<T>  — block sparse row + BsrBuilder
│   ├── ops.rs             SpMV helpers
│   └── nalgebra.rs        direct LinearOperator impl for nalgebra_sparse::CsrMatrix (native only)
├── iterative/
│   ├── cg.rs              Conjugate Gradient (SPD systems)
│   ├── minres.rs          MINRES (symmetric indefinite)
│   ├── gmres.rs           GMRES(m) (general)
│   ├── bicgstab.rs        BiCGSTAB (non-symmetric)
│   ├── fgmres.rs          Flexible GMRES (variable preconditioner)
│   ├── lgmres.rs          LGMRES (augmented Krylov)
│   ├── idrs.rs            IDR(s) — short-recurrence, non-symmetric, with auto-restart
│   └── tfqmr.rs           TFQMR — Transpose-Free QMR (Freund 1993)
├── precond/
│   ├── jacobi.rs          JacobiPrecond — diagonal scaling
│   ├── block_jacobi.rs    BlockJacobiPrecond — dense LU per diagonal block
│   ├── sor.rs             SOR / SSOR
│   ├── ilu0.rs            ILU(0)
│   ├── iluk.rs            ILU(k) — level-of-fill
│   ├── ilut.rs            ILUT(tau, p) — dual threshold
│   ├── icc.rs             ICC(0) — incomplete Cholesky
│   ├── spai.rs            SPAI — sparse approximate inverse
│   ├── composite.rs       AdditivePrecond / MultiplicativePrecond
│   ├── ams.rs             AmsPrecond — auxiliary-space Maxwell solver (H(curl))
│   └── ads.rs             AdsPrecond — auxiliary-space divergence solver (H(div))
├── amg/
│   ├── strength.rs        Strong-connection graph (θ threshold)
│   ├── coarsen_rs.rs      Ruge–Stüben C/F splitting
│   ├── coarsen_agg.rs     Smoothed Aggregation (SA-AMG) greedy aggregation
│   ├── interpolation.rs   RS direct interpolation / SA smoothed prolongation
│   ├── smoother.rs        Weighted Jacobi / Gauss-Seidel sweeps
│   ├── cycle.rs           V-cycle / W-cycle / K-cycle
│   └── setup.rs           AmgHierarchy::build (Galerkin RAP)
├── direct/
│   ├── blr.rs             BlrBlock<T> — Block Low-Rank compression (randomised SVD)
│   ├── lu.rs              SparseLu — Gilbert-Peierls + partial pivoting
│   ├── lu_sn.rs           SupernodalSparseLu — supernodal LU
│   ├── cholesky.rs        SparseCholesky — left-looking incomplete Cholesky
│   ├── cholesky_sn.rs     SupernodalSparseCholesky
│   ├── ldlt.rs            SparseLdlt — left-looking sparse LDLᵀ
│   ├── multifrontal.rs    MultifrontalLu — multifrontal LU with optional BLR compression
│   ├── symbolic.rs        SymbolicCholesky / SymbolicLu — fill-pattern analysis
│   ├── etree.rs           Elimination tree + post-order traversal
│   ├── triangular.rs      forward_solve / backward_solve
│   └── ordering/          RCM / COLAMD / nested-dissection fill-reducing orderings
├── eigen/
│   ├── power.rs           PowerIter — largest-magnitude single eigenpair
│   ├── subspace.rs        SubspaceIter — k largest eigenpairs
│   ├── inverse.rs         InverseIter, RayleighQuotientIter
│   ├── lanczos.rs         LanczosIter (IRLM) — symmetric operators
│   ├── arnoldi.rs         ArnoldiIter (IRAM) — general operators
│   ├── generalized.rs     GeneralizedEigen (Ax=λBx), ShiftInvertLanczos
│   ├── krylov_schur.rs    KrylovSchur — robust restart (Stewart 2001)
│   ├── lobpcg.rs          Lobpcg — block CG for SPD (Knyazev 2001)
│   ├── svd.rs             LanczosSvd — partial SVD via Lanczos on AᵀA
│   ├── qep.rs             QuadraticEigen — (K+λC+λ²M)x=0 via companion linearisation
│   └── nep.rs             NonlinearOperator trait + NepNewton
├── parallel/
│   └── rayon_ops.rs       parallel_spmv, parallel_axpy, parallel_dot, …
└── wasm.rs                WasmCsrMatrix, WasmCgSolver, WasmGmresSolver
```

---

## Eigenvalue solvers

All eigenvalue algorithms implement `EigenSolver<T>` and accept any `LinearOperator`.

```rust
use linger::{
    EigenParams, EigenWhich,
    LanczosIter, ArnoldiIter, KrylovSchur, Lobpcg,
    LanczosSvd, QuadraticEigen,
    NonlinearOperator, NepNewton,
};
```

### Standard eigenvalue problems (`Ax = λx`)

| Struct | Best for | Notes |
|--------|----------|-------|
| `PowerIter` | Largest-magnitude single eigenpair | Simple, no restarts |
| `SubspaceIter` | k largest eigenpairs | Orthogonal iteration |
| `InverseIter` | Nearest to a shift | Shift-invert via matrix-free GMRES |
| `RayleighQuotientIter` | Single eigenpair, cubic convergence | Adaptive shift |
| `LanczosIter` | k eigenpairs, **symmetric** operators | IRLM; thick restart |
| `ArnoldiIter` | k eigenpairs, any operator | IRAM; full Hessenberg |
| `KrylovSchur` | k eigenpairs, any operator (robust) | Stewart 2001; deflation |
| `Lobpcg` | k smallest, **SPD** operators | Best with AMG precond |

```rust
let a = laplacian_csr(200);
let params = EigenParams::new(6, EigenWhich::SmallestAlgebraic);
let res = LanczosIter::default().solve(&a, &params).unwrap();
println!("λ = {:?}", res.eigenvalues);
```

### Generalised eigenvalue problems (`Ax = λBx`)

```rust
use linger::{GeneralizedEigen, ShiftInvertLanczos};

// ShiftInvertLanczos: shift near σ → targets eigenvalues closest to σ
let solver = ShiftInvertLanczos::<f64>::new(0.0);  // σ = 0 → smallest eigenvalues
let res = solver.solve(&a, &params).unwrap();
```

### Singular Value Decomposition (SVD)

`LanczosSvd` computes the k largest singular values via Lanczos on AᵀA.
Requires the operator to implement [`TransposeOperator`] — `CsrMatrix` does.

```rust
let svd = LanczosSvd::default();
let res = svd.solve(&a, /*k=*/4, /*tol=*/1e-10, /*max_iter=*/500, /*vecs=*/true).unwrap();
println!("σ = {:?}", res.singular_values);
// res.left_vectors  → U columns
// res.right_vectors → V columns
```

### Quadratic Eigenvalue Problem — QEP (`(K + λC + λ²M)x = 0`)

Structural dynamics modal analysis with damping.  Linearises to a 2n × 2n
companion standard EVP and delegates to `ArnoldiIter`.

```rust
let qep = QuadraticEigen::new(4);   // 4 eigenpairs
let mut params = EigenParams::new(4, EigenWhich::LargestMagnitude);
let res = qep.solve(&k_mat, &c_mat, &m_mat, &params).unwrap();
```

### Nonlinear Eigenvalue Problem — NEP (`T(λ)x = 0`)

```rust
struct MyNep { /* ... */ }

impl NonlinearOperator<f64> for MyNep {
    fn nrows(&self) -> usize { /* ... */ }
    fn apply_t(&self, lam: f64, v: &DenseVec<f64>, out: &mut DenseVec<f64>) { /* T(λ)v */ }
    // apply_dt: defaults to central finite difference — override for exact derivative
}

let solver = NepNewton::new(/*shift=*/2.9, /*tol=*/1e-9, /*max_iter=*/200);
let (lam, x) = solver.solve(&my_nep).unwrap();
```

### `ComplexScalar` trait

The `ComplexScalar` trait extends numeric support to `Complex<f32>` and
`Complex<f64>`.  Every `Scalar` type also implements `ComplexScalar`
(real numbers are a special case).

```rust
use linger::{Complex, ComplexScalar};

let z: Complex<f64> = Complex::new(3.0, 4.0);
let modulus: f64 = ComplexScalar::abs(z);   // 5.0
let conj            = ComplexScalar::conj(z); // 3 - 4i
let re: f64         = ComplexScalar::real(z); // 3.0
```

---

### `CsrMatrix<T>`

The primary operator. Implement once, use everywhere.

```rust
// Build
let csr = CsrMatrix::from_coo(&coo);
let csr = CsrMatrix::from_raw(nrows, ncols, row_ptr, col_idx, values);
//   └─ from_raw checks col_idx bounds in debug builds (panics if any col_idx ≥ ncols)

// Query
csr.nrows()  csr.ncols()  csr.nnz()
csr.row_ptr()  csr.col_idx()  csr.values()
csr.triplets()          // Iterator<(row, col, val)>
csr.validate()          // Result<(), String> — check structural correctness

// Operations
csr.spmv(x, y)                          // y = A·x
csr.spmv_add(alpha, x, beta, y)         // y = α·A·x + β·y
csr.diag()                              // Vec<T>
csr.transpose()                         // CscMatrix<T>
csr.transpose_csr()                     // CsrMatrix<T>   (needed for AMG)
csr.matmat(&b)                          // CsrMatrix<T>   (C = A·B)
csr.is_structurally_symmetric()         // bool
```

Implements `LinearOperator`, so it can be passed directly to any `KrylovSolver`.

### `DenseVec<T>`

```rust
DenseVec::zeros(n)
DenseVec::from_vec(vec)
dv.as_slice()    dv.as_mut_slice()    dv.into_vec()
```

Implements the `Vector` trait (dot, axpy, norm2, scale, …).

### `SolverParams`

```rust
SolverParams {
    rtol:           1e-8,           // relative residual tolerance
    atol:           0.0,            // absolute residual tolerance
    max_iter:       1_000,
    verbose:        VerboseLevel::Silent,   // Silent | Summary | Iterations
    check_interval: 10,             // recompute true residual every N iters
    ..Default::default()
}
```

### `SolverResult`

```rust
result.converged          // bool
result.iterations         // usize
result.final_residual     // f64 — ‖b − Ax‖ / ‖b‖
result.residual_history   // Vec<f64> — per-iteration residuals (always populated; moved out, not cloned)
result.history            // Option<Vec<f64>> — same, only Some when verbose = Iterations
```

---

## Solvers

All solvers implement `KrylovSolver<Operator = CsrMatrix<T>, Vector = DenseVec<T>>`.

| Struct | Best for | Constructor | Notes |
|--------|----------|-------------|-------|
| `ConjugateGradient` | SPD systems | `::default()` | |
| `Minres` | Symmetric indefinite | `::default()` | |
| `Gmres` | General (non-symmetric) | `::new(restart)` | Krylov basis pre-allocated outside restart loop |
| `BiCgStab` | Non-symmetric, large | `::new()` | |
| `Fgmres` | Variable preconditioner | `::new(restart)` | |
| `Lgmres` | Augmented Krylov | `::new(restart, aug_dim)` | |
| `Idrs` | Non-symmetric, short recurrence | `::new(s)` — s=4 recommended | Hot-path allocations eliminated |
| `Tfqmr` | Non-symmetric, breakdown-robust | `::new()` | |

`Idrs` uses s shadow vectors; larger s → fewer iterations, more work per step (s=1 ≈ BiCGSTAB, s=4 typical). It auto-restarts with a fresh shadow space on near-breakdown, configurable via `.with_max_restarts(n)`.

`Tfqmr` (Transpose-Free QMR, Freund 1993) uses 2 matrix-vector products per outer step and avoids the omega denominator that causes BiCGSTAB breakdown.

### Performance notes

- **GMRES**: The Arnoldi basis (`m+1` vectors of size `n`), preconditioner scratch (`z`, `w`, `mz`), and `Ax` scratch are allocated once before the restart loop and reused each cycle — no per-restart heap allocations for these buffers.
- **All solvers**: On convergence or early exit, `residual_history` is moved out of the solver (via `std::mem::take`) rather than cloned — zero extra allocation on the return path.
- **IDR(s)**: Preconditioner application and inner-loop SpMV reuse pre-allocated `DenseVec` scratch buffers, eliminating the O(s · n_iter) transient allocations that existed in earlier versions.

```rust
// Signature (same for all)
solver.solve(&a, precond_opt, &b, &mut x, &params) -> Result<SolverResult, SolverError>
```

---

## Preconditioners

All implement `Preconditioner<Vector = DenseVec<T>>`.

| Struct | Constructor | Notes |
|--------|-------------|-------|
| `JacobiPrecond` | `::from_csr(&a)` | Diagonal scaling |
| `SorPrecond` | `::from_csr(&a, omega)` | 0 < ω < 2 |
| `SsorPrecond` | `::from_csr(&a, omega)` | Symmetric SOR |
| `Ilu0Precond` | `::from_csr(&a)` | ILU(0), exact on tridiagonals |
| `IlukPrecond` | `::from_csr(&a, k)` | ILU(k), k=0,1,2,… |
| `IlutPrecond` | `::from_csr(&a, tau, p)` | Dual threshold drop |
| `Icc0Precond` | `::from_csr(&a)` | Incomplete Cholesky, SPD only |
| `SpaiPrecond` | `::from_csr(&a)` | Sparse approximate inverse |
| `AdditivePrecond` | `::new(vec_of_preconds)` | Sums M⁻¹ applications |
| `MultiplicativePrecond` | `::new(vec_of_preconds)` | Composes M⁻¹ applications |
| `BlockJacobiPrecond` | `::from_csr(&a, block_size)` | Dense LU per diagonal block; ideal for multi-DOF-per-node FEA |
| `AmgPrecond` | `AmgPrecond::new(hierarchy)` | AMG V-cycle as preconditioner |
| `AmsPrecond` | `::new(&a, &g, config)` | Auxiliary-space Maxwell solver — H(curl) / edge elements |
| `AdsPrecond` | `::new(&a, &c, &g, config)` | Auxiliary-space divergence solver — H(div) / face elements |

All constructors return `Result<_, SolverError>`.

```rust
// Typical usage
let precond = IlukPrecond::<f64>::from_csr(&a, 1).unwrap();
solver.solve(&a, Some(&precond), &b, &mut x, &params)?;

// Block Jacobi — ideal when DOF are grouped in fixed-size blocks (e.g. 3D elasticity)
let bjac = BlockJacobiPrecond::<f64>::from_csr(&a, 3).unwrap();  // 3×3 blocks
Idrs::<f64>::new(4).solve(&a, Some(&bjac), &b, &mut x, &params)?;
```

---

## Auxiliary-space preconditioners (AMS / ADS)

Pure-Rust implementations of the Hiptmair-Xu auxiliary-space framework for
edge- and face-element FEA problems.

### AMS — H(curl) / edge elements (Maxwell)

```text
M_AMS⁻¹ x  ≈  ω D_A⁻¹ x  +  G · P_v⁻¹ · Gᵀ x
```

| Term | Meaning |
|------|---------|
| `ω D_A⁻¹ x` | Weighted Jacobi smoother on the edge space |
| `G · P_v⁻¹ · Gᵀ x` | AMG (or ILU(0)) solve on the nodal Laplacian `GᵀAG` |

```rust
use linger::precond::{AmsPrecond, AmsConfig, AuxSpaceSolver};

// G: discrete gradient matrix (n_edges × n_nodes), user-assembled
let config = AmsConfig::default();          // AMG coarse solve, ω = 0.667
let precond = AmsPrecond::new(&a_edge, &g, config)?;

ConjugateGradient::default()
    .solve(&a_edge, Some(&precond), &b, &mut x, &params)?;
```

### ADS — H(div) / face elements (Darcy, mixed Maxwell)

```text
M_ADS⁻¹ x  ≈  ω D_A⁻¹ x  +  C · P_e⁻¹ · Cᵀ x  +  C G · P_v⁻¹ · Gᵀ Cᵀ x
```

| Term | Meaning |
|------|---------|
| `ω D_A⁻¹ x` | Weighted Jacobi smoother on the face space |
| `C · P_e⁻¹ · Cᵀ x` | AMG solve on the edge Laplacian `CᵀAC` |
| `C G · P_v⁻¹ · Gᵀ Cᵀ x` | AMG solve on the nodal Laplacian `Gᵀ(CᵀAC)G` |

```rust
use linger::precond::{AdsPrecond, AdsConfig};

// C: discrete curl (n_faces × n_edges), G: discrete gradient (n_edges × n_nodes)
let config = AdsConfig::default();          // AMG for both coarse solves
let precond = AdsPrecond::new(&a_face, &c, &g, config)?;

Gmres::new(30).solve(&a_face, Some(&precond), &b, &mut x, &params)?;
```

### Coarse-solver choice

Both `AmsConfig` and `AdsConfig` accept `AuxSpaceSolver` for each coarse level:

```rust
use linger::precond::{AmsConfig, AuxSpaceSolver};
use linger::amg::AmgConfig;

// AMG (default, recommended for large problems)
let config = AmsConfig { node_solver: AuxSpaceSolver::Amg(AmgConfig::default()), ..Default::default() };

// ILU(0) (fast setup, suitable for small/medium non-singular coarse problems)
let config = AmsConfig { node_solver: AuxSpaceSolver::Ilu0, ..Default::default() };
```

> **Note:** ILU(0) will fail with `PrecondSetupFailed` if the coarse operator is
> singular.  This can happen when `A = GGᵀ` (pure edge Laplacian) has no
> diagonal shift.  Add a small regularisation `δI` to `A` before constructing
> the preconditioner, or use AMG instead.

### Via `SolverBuilder`

```rust
use linger::builder::{SolverBuilder, SolveMethod, PrecondChoice};
use linger::precond::{AmsConfig};
use std::sync::Arc;

let precond = PrecondChoice::Ams {
    g:      Arc::new(g_f64),          // f64 gradient matrix
    config: AmsConfig::default(),
};
let x = SolverBuilder::new()
    .method(SolveMethod::Gmres { restart: 30 })
    .precond(precond)
    .solve(&a, &b)?;
```

---

## Algebraic Multigrid (AMG)

```rust
use linger::amg::{AmgConfig, AmgHierarchy, AmgPrecond, CoarsenStrategy, SmootherType};

let config = AmgConfig {
    theta:            0.25,                              // strong-connection threshold
    strategy:         CoarsenStrategy::SmoothedAggregation, // or RugeStüben
    smoother:         SmootherType::WeightedJacobi { omega: 0.667 },
    pre_sweeps:       1,
    post_sweeps:      1,
    coarse_threshold: 10,                                // stop coarsening below this size
    max_levels:       20,
    sa_omega:         0.667,
};

let hier    = AmgHierarchy::build(a.clone(), config);
let precond = AmgPrecond::new(hier);
```

### Standalone AMG solve (V-cycle / W-cycle / K-cycle)

```rust
use linger::amg::CycleType;
let b_dv = DenseVec::from_vec(b.clone());
let mut x_dv = DenseVec::zeros(n);
hier.apply_cycle(&b_dv, &mut x_dv, CycleType::V);
hier.apply_cycle(&b_dv, &mut x_dv, CycleType::W);
hier.apply_cycle(&b_dv, &mut x_dv, CycleType::K { inner_iters: 2 });
```

The K-cycle uses inner preconditioned CG (with the next-level V-cycle as preconditioner) as the coarse correction. It gives better convergence than W-cycle for heterogeneous or harder problems. Because it is a **variable preconditioner**, use it with `AmgPrecond` + a flexible outer method, or as a standalone iterative solver — not with standard CG.

```rust
// K-cycle as AMG preconditioner (use with FGMRES or standalone)
let precond = AmgPrecond::new(hier).with_cycle(CycleType::K { inner_iters: 2 });
```

---

## Parallel operations (feature = "rayon")

```rust
use linger::{parallel_spmv, parallel_spmv_add, parallel_axpy,
             parallel_axpby, parallel_dot, parallel_norm2};

parallel_spmv(&a, &x, &mut y);                       // y = A·x
parallel_spmv_add(&a, alpha, &x, beta, &mut y);      // y = α·A·x + β·y
parallel_axpy(alpha, &x, &mut y);                    // y += α·x
parallel_axpby(alpha, &x, beta, &mut y);             // y = α·x + β·y
let d = parallel_dot(&x, &y);
let n = parallel_norm2(&x);
```

When `rayon` is disabled these functions silently fall back to scalar paths — the API is identical.

AMG setup phases (`strong_connections`, `rs_interpolation`, `smooth_prolongation`) also use `par_iter` when `rayon` is enabled.

---

## Block Sparse Row (BSR) format

```rust
use linger::{BsrBuilder, BsrMatrix};

let mut builder = BsrBuilder::<f64>::new(n_block_rows, n_block_cols, r, c);
builder.push_block(br, bc, block_values_row_major);  // duplicate blocks are summed
let bsr: BsrMatrix<f64> = builder.build();

bsr.spmv(&x, &mut y);           // serial block SpMV
bsr.spmv_parallel(&x, &mut y);  // parallel block SpMV (rayon feature)
let csr = bsr.to_csr();          // convert to CsrMatrix
```

---

## WebAssembly interface (feature = "wasm")

Build:
```bash
# Core (no native nalgebra integration, no rayon)
cargo build --target wasm32-unknown-unknown --no-default-features

# Full JS interface
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm
```

JavaScript usage:
```js
import init, { WasmCsrMatrix, WasmCgSolver, WasmGmresSolver } from './linger_wasm.js';
await init();

const A = WasmCsrMatrix.from_coo(n, n, rowsU32, colsU32, valsF64);
const solver = new WasmCgSolver(1e-8, 500);
const x = solver.solve(A, b);   // Float64Array

const gmres = new WasmGmresSolver(1e-8, 500, 30);
const x2 = gmres.solve(A, b);
```

Note: native `nalgebra` integration is excluded from wasm32 builds. The core CSR/COO/CSC and solver stack remain wasm-compatible.

---

## Running tests and benchmarks

```bash
# All tests (509 tests across 33 suites)
cargo test

# Individual suites (selection)
cargo test --test test_sparse_ops           # CSR/CSC structure + validate (30 tests)
cargo test --test test_sparse_proptest_e4   # proptest round-trip + SpMV linearity (5 tests)
cargo test --test test_krylov               # Core Krylov solvers (19 tests)
cargo test --test test_precond              # Basic preconditioners (11 tests)
cargo test --test test_sprint3              # Advanced precond + FGMRES/LGMRES (21 tests)
cargo test --test test_idrs_f4              # IDR(s) solver incl. restart (12 tests)
cargo test --test test_tfqmr_a1             # TFQMR solver (8 tests)
cargo test --test test_block_jacobi_a4      # Block Jacobi preconditioner (7 tests)
cargo test --test test_ams_ads              # AMS/ADS auxiliary-space preconditioners (13 tests)
cargo test --test test_amg                  # AMG hierarchy and cycles (10 tests)
cargo test --test test_amg_kcycle_a3        # AMG K-cycle (6 tests)
cargo test --test test_amg_internals        # AMG sub-modules + ILU(k) (22 tests)
cargo test --test test_parallel             # Parallel ops + BSR format (13 tests)
cargo test --test test_eigen                # Eigenvalue solvers (11 tests)
cargo test --test test_eigen_s8_s10         # Eigenvalue solvers Sprint 8-10 (16 tests)
cargo test --test test_eigen_s11_s12        # SVD, QEP, NEP, ComplexScalar (9 tests)

# Criterion benchmarks (HTML report in target/criterion/)
cargo bench --bench bench_spmv
cargo bench --bench bench_krylov
cargo bench --bench bench_amg

# WASM build verification
cargo build --target wasm32-unknown-unknown --no-default-features
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm
```

---

## Error handling

All public APIs return `Result<_, SolverError>`:

```rust
pub enum SolverError {
    SingularMatrix { row: usize },
    ConvergenceFailed { max_iter: usize, residual: f64 },
    DimensionMismatch { op_rows, op_cols, rhs_len },
    PrecondSetupFailed { reason: String },
    NumericalBreakdown { detail: String },
}
```

`ConvergenceFailed` is returned when `max_iter` is reached without satisfying `rtol`/`atol`. If you only care about the best-effort solution, `result.converged == false` is not fatal; the solution `x` is still the best available iterate.

---

## Design constraints for agents

1. **All algorithms are generic over `T: Scalar`** (`f32` and `f64` both work).
2. **No global mutable state.** Preconditioners implement `&self` apply — safe for concurrent use.
3. **No `std::thread::spawn` in library code.** Parallelism flows exclusively through Rayon's `par_iter` and is gated by `#[cfg(feature = "rayon")]`.
4. **No `std::time::Instant` in the core library** — safe for wasm32 compilation.
5. **Direct `nalgebra_sparse::CsrMatrix` support** is gated to `cfg(not(target_arch = "wasm32"))`. Use linger's own sparse formats in wasm-targeted code.
6. **Matrix construction is always COO → CSR.** Never construct `CsrMatrix` by hand; use `CooMatrix::push` then `CsrMatrix::from_coo`. Duplicate entries are summed automatically.
7. **`from_raw` is for internal use.** Prefer `from_coo` unless you have pre-validated CSR arrays. In debug builds, `from_raw` panics if any `col_idx` value is ≥ `ncols` — this protects the `unsafe get_unchecked` calls in `spmv` from out-of-bounds access.

---

## Test infrastructure

`tests/common/mod.rs` exposes shared helpers used by all test suites:

```rust
// Returns (A, x_exact, b) where b = A * x_exact
common::make_poisson_1d::<f64>(n)           // 1D Poisson tridiagonal
common::make_poisson_2d::<f64>(nx, ny)      // 2D Poisson 5-point stencil
common::make_nonsymmetric_convdiff::<f64>(n, peclet)  // upwind convection-diffusion

// AMS/ADS test geometries
common::make_chain_graph(n_nodes, delta)       // 1-D edge complex: (G, A=GGᵀ+δI)
common::make_rect_complex(nx, ny, delta)       // 2-D face complex: (G, C, A=CCᵀ+δI)

// ‖Ax − b‖₂ / ‖b‖₂
common::relative_residual(&a, x.as_slice(), &b)
```

---

## Direct solvers

All direct solvers implement the `DirectSolver<T>` trait:

```rust
solver.analyze(&a)?;   // fill-reducing reorder + symbolic factorisation
solver.factorize(&a)?; // numerical factorisation (reuse analysis if pattern unchanged)
solver.solve(&b, &mut x)?;
// — or in one call —
solver.factor(&a)?;
solver.solve_multi(&bs, &mut xs)?;   // multiple right-hand sides
```

### Available solvers

| Struct | Algorithm | Suitable for |
|--------|-----------|--------------|
| `SparseLu` | Gilbert-Peierls LU + partial pivoting | General square matrices |
| `SupernodalSparseLu` | Supernodal LU | General; better cache use on large problems |
| `SparseCholesky` | Left-looking sparse Cholesky | SPD matrices |
| `SupernodalSparseCholesky` | Supernodal Cholesky | SPD; improved cache blocking |
| `SparseLdlt` | Left-looking sparse LDLᵀ | Symmetric indefinite |
| `MultifrontalLu` | Multifrontal LU + optional BLR compression | General; approximate via BLR for preconditioning |

### Fill-reducing orderings

```rust
use linger::direct::ordering::{rcm, colamd, nd, OrderingMethod};

let perm = rcm(&a);    // Reverse Cuthill-McKee
let perm = colamd(&a); // Column approximate minimum degree
let perm = nd(&a);     // Nested dissection (best fill for 2D/3D FEA)
```

Pass the ordering via `DirectOptions`:

```rust
use linger::direct::{SparseLu, DirectOptions};
use linger::direct::ordering::OrderingMethod;

let mut solver = SparseLu::<f64>::default();
solver.options.ordering = OrderingMethod::NestedDissection;
solver.factor(&a)?;
solver.solve(&b, &mut x)?;
```

### Direct solver as preconditioner

Any `DirectSolver` can be wrapped in `DirectSolverPrecond` for use with Krylov methods:

```rust
use linger::direct::{SparseLu, DirectSolverPrecond};

let precond = DirectSolverPrecond::new(SparseLu::<f64>::default(), &a)?;
Gmres::new(30).solve(&a, Some(&precond), &b, &mut x, &params)?;
```

### `MultifrontalLu` with BLR compression

`MultifrontalLu` supports approximate factorisation via Block Low-Rank (BLR)
compression of off-diagonal frontal blocks.  BLR trades accuracy for memory and
arithmetic savings (typically 2–5× for FEA problems).  Use it as a
high-quality preconditioner rather than an exact solver.

```rust
use linger::direct::multifrontal::{MultifrontalLu, MultifrontalOptions};

let opts = MultifrontalOptions {
    blr_min_size: 16,   // compress fronts larger than this
    blr_tol: 1e-6,      // relative singular-value threshold
    ..Default::default()
};
let mut solver = MultifrontalLu::<f64>::with_options(opts);
solver.factor(&a)?;
solver.solve(&b, &mut x)?;
```

Setting `blr_min_size = usize::MAX` disables BLR entirely (exact factorisation).

### `BlrBlock<T>` — low-level BLR API

`BlrBlock` is also available as a standalone compression primitive:

```rust
use linger::direct::{BlrBlock, compress_block};

// Compress a row-major m×n dense block with tolerance 1e-8.
// max_rank = 0 means no hard cap (uses min(m, n)).
let blk: BlrBlock<f64> = compress_block(&dense, m, n, 1e-8, /*max_rank=*/ 0);

println!("rank={}, compression={:.1}%", blk.rank, blk.compression_ratio() * 100.0);
let (dense_bytes, blr_bytes) = blk.memory_bytes();

// Matrix-vector products
blk.apply_add(&x, &mut y, alpha);    // y += α A x
blk.apply_add_t(&x, &mut y, alpha);  // y += α Aᵀ x  (transpose)

// Recompress with a looser tolerance (no access to original matrix needed)
let blk2 = blk.recompress(1e-4);

// Add two same-size BLR blocks and recompress
let blk_sum = blk_a.add_compressed(&blk_b, 1e-6, /*max_rank=*/ 0);
```

---

## References

1. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.)
2. Trottenberg, Oosterlee & Schüller (2001). *Multigrid*
3. Falgout & Yang (2002). *hypre: A Library of High Performance Preconditioners*
4. Balay et al. *PETSc Users Manual* (ANL-95/11 Rev 3.20)
5. Freund, R.W. (1993). A transpose-free quasi-minimal residual algorithm for non-Hermitian linear systems. *SIAM J. Sci. Comput.*, 14(2), 470–482. (TFQMR)
6. van Gijzen, M.B. & Sonneveld, P. (2011). Algorithm 913: An elegant IDR(s) variant that efficiently exploits biorthogonality properties. *ACM Trans. Math. Software*, 38(1). (IDR(s))
7. Hiptmair, R. & Xu, J. (2007). Nodal auxiliary space preconditioning in H(curl) and H(div) spaces. *SIAM J. Numer. Anal.*, 45(6), 2483–2509. (AMS/ADS)
8. Kolev, T.V. & Vassilevski, P.S. (2009). Parallel auxiliary space AMG for H(curl) problems. *J. Comput. Math.*, 27(5), 604–623. (AMS)
