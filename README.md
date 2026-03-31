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

---

## Module map

```
linger/
├── core/
│   ├── scalar.rs          Scalar trait (f32 / f64 generic bound)
│   ├── vector.rs          Vector trait + DenseVec<T>
│   ├── operator.rs        LinearOperator trait
│   ├── preconditioner.rs  Preconditioner trait
│   ├── solver.rs          KrylovSolver trait, SolverParams, SolverResult
│   └── error.rs           SolverError enum
├── sparse/
│   ├── coo.rs             CooMatrix<T>  — assembly format
│   ├── csr.rs             CsrMatrix<T>  — primary operator format
│   ├── csc.rs             CscMatrix<T>  — obtained via csr.transpose()
│   ├── bsr.rs             BsrMatrix<T>  — block sparse row + BsrBuilder
│   ├── ops.rs             SpMV helpers
│   ├── adapt_nalgebra.rs  nalgebra CsrMatrix → NalgebraCsrOp  (native only)
│   └── adapt_faer.rs      faer SparseColMat → FaerSparseOp    (native only)
├── iterative/
│   ├── cg.rs              Conjugate Gradient (SPD systems)
│   ├── minres.rs          MINRES (symmetric indefinite)
│   ├── gmres.rs           GMRES(m) (general)
│   ├── bicgstab.rs        BiCGSTAB (non-symmetric)
│   ├── fgmres.rs          Flexible GMRES (variable preconditioner)
│   └── lgmres.rs          LGMRES (augmented Krylov)
├── precond/
│   ├── jacobi.rs          Jacobi / Block Jacobi
│   ├── sor.rs             SOR / SSOR
│   ├── ilu0.rs            ILU(0)
│   ├── iluk.rs            ILU(k) — level-of-fill
│   ├── ilut.rs            ILUT(tau, p) — dual threshold
│   ├── icc.rs             ICC(0) — incomplete Cholesky
│   ├── spai.rs            SPAI — sparse approximate inverse
│   └── composite.rs       AdditivePrecond / MultiplicativePrecond
├── amg/
│   ├── strength.rs        Strong-connection graph (θ threshold)
│   ├── coarsen_rs.rs      Ruge–Stüben C/F splitting
│   ├── coarsen_agg.rs     Smoothed Aggregation (SA-AMG) greedy aggregation
│   ├── interpolation.rs   RS direct interpolation / SA smoothed prolongation
│   ├── smoother.rs        Weighted Jacobi / Gauss-Seidel sweeps
│   ├── cycle.rs           V-cycle / W-cycle
│   └── setup.rs           AmgHierarchy::build (Galerkin RAP)
├── parallel/
│   └── rayon_ops.rs       parallel_spmv, parallel_axpy, parallel_dot, …
└── wasm.rs                WasmCsrMatrix, WasmCgSolver, WasmGmresSolver
```

---

## Core types

### `CsrMatrix<T>`

The primary operator. Implement once, use everywhere.

```rust
// Build
let csr = CsrMatrix::from_coo(&coo);
let csr = CsrMatrix::from_raw(nrows, ncols, row_ptr, col_idx, values);

// Query
csr.nrows()  csr.ncols()  csr.nnz()
csr.row_ptr()  csr.col_idx()  csr.values()
csr.triplets()          // Iterator<(row, col, val)>

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
result.converged        // bool
result.iterations       // usize
result.final_residual   // f64 — ‖b − Ax‖ / ‖b‖
result.history          // Option<Vec<f64>> — per-iteration residuals
```

---

## Solvers

All solvers implement `KrylovSolver<Operator = CsrMatrix<T>, Vector = DenseVec<T>>`.

| Struct | Best for | Constructor |
|--------|----------|-------------|
| `ConjugateGradient` | SPD systems | `::default()` |
| `Minres` | Symmetric indefinite | `::default()` |
| `Gmres` | General (non-symmetric) | `::new(restart)` |
| `BiCgStab` | Non-symmetric, large | `::new()` |
| `Fgmres` | Variable preconditioner | `::new(restart)` |
| `Lgmres` | Augmented Krylov | `::new(restart, aug_dim)` |

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
| `AmgPrecond` | `AmgPrecond::new(hierarchy)` | AMG V-cycle as preconditioner |

All constructors return `Result<_, SolverError>`.

```rust
// Typical usage
let precond = IlukPrecond::<f64>::from_csr(&a, 1).unwrap();
solver.solve(&a, Some(&precond), &b, &mut x, &params)?;
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

### Standalone AMG solve (V-cycle / W-cycle)

```rust
use linger::amg::CycleType;
let b_dv = DenseVec::from_vec(b.clone());
let mut x_dv = DenseVec::zeros(n);
hier.apply_cycle(&b_dv, &mut x_dv, CycleType::V);
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
# Core (no adapters, no rayon)
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

Note: `nalgebra`/`faer` adapters are automatically excluded from wasm32 builds (they depend on threading and `getrandom`).

---

## Running tests and benchmarks

```bash
# All tests (121 tests across 8 suites)
cargo test

# Individual suites
cargo test --test test_sparse_ops       # CSR/CSC structure operations (26 tests)
cargo test --test test_krylov           # Core Krylov solvers (15 tests)
cargo test --test test_precond          # Basic preconditioners (11 tests)
cargo test --test test_sprint3          # Advanced precond + FGMRES/LGMRES (21 tests)
cargo test --test test_amg              # AMG hierarchy and cycles (10 tests)
cargo test --test test_amg_internals    # AMG sub-modules + ILU(k) (22 tests)
cargo test --test test_parallel         # Parallel ops + BSR format (13 tests)

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
5. **`nalgebra`/`faer` adapters** (`adapt_nalgebra.rs`, `adapt_faer.rs`) are gated to `cfg(not(target_arch = "wasm32"))`. Do not import them in wasm-targeted code.
6. **Matrix construction is always COO → CSR.** Never construct `CsrMatrix` by hand; use `CooMatrix::push` then `CsrMatrix::from_coo`. Duplicate entries are summed automatically.
7. **`from_raw` is for internal use.** Prefer `from_coo` unless you have pre-validated CSR arrays.

---

## Test infrastructure

`tests/common/mod.rs` exposes shared helpers used by all test suites:

```rust
// Returns (A, x_exact, b) where b = A * x_exact
common::make_poisson_1d::<f64>(n)           // 1D Poisson tridiagonal
common::make_poisson_2d::<f64>(nx, ny)      // 2D Poisson 5-point stencil
common::make_nonsymmetric_convdiff::<f64>(n, peclet)  // upwind convection-diffusion

// ‖Ax − b‖₂ / ‖b‖₂
common::relative_residual(&a, x.as_slice(), &b)
```

---

## References

1. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.)
2. Trottenberg, Oosterlee & Schüller (2001). *Multigrid*
3. Falgout & Yang (2002). *hypre: A Library of High Performance Preconditioners*
4. Balay et al. *PETSc Users Manual* (ANL-95/11 Rev 3.20)
