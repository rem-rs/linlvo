# External Solver Requirements (fem-rs)

Date: 2026-04-13
Scope: required capabilities for `linger` external solver backends used by `fem-rs`.

Cross-project scope update: this roadmap must be coordinated with `vendor/reed`
and `vendor/jsmpi` integration paths.

## Goal

Provide production-grade solver integration with pure-Rust default paths:

- `hypre-rs` parity track (pure Rust, no external hypre FFI dependency)
- `petsc-rs` parity track (pure Rust, no external PETSc FFI dependency)
- `mkl` compatibility profile remains native to `linger` and is not an external-backend delivery target
- `mumps` compatibility profile remains native to `linger` and is not an external-backend delivery target

Ownership split:

- `linger`: backend contracts + pure-Rust solver cores (`hypre-rs`, `petsc-rs`, native `mumps`-compatible direct path)
- `reed`: GPU execution path and operator/export integration with `linger`
- `jsmpi`: wasm/browser runtime constraints and fallback reporting

## 1. Common Integration Requirements

- [ ] Stable backend abstraction in `linger` builder API (select backend without changing call sites)
- [ ] Unified error mapping from C/Fortran return codes to `SolverError`
- [ ] Explicit MPI communicator ownership and lifetime rules
- [ ] Feature-gated compile path with clean fallback to pure-Rust implementations
- [ ] Reusable symbolic/numeric factorization handles
- [ ] Multi-RHS solve support where backend provides it
- [ ] CI matrix entries for each enabled backend (build + smoke solve)

## 2. HYPRE-Equivalent (`hypre-rs`) Requirements (Pure Rust)

- [ ] ParCSR-style matrix/vector bridge (import/export from `linger` CSR partition view)
- [ ] BoomerAMG-equivalent setup/apply lifecycle in pure Rust
- [ ] Core BoomerAMG options: coarsening, interpolation, relax/smoother, cycle type
- [ ] AIR/Advective AMG option path for non-symmetric problems
- [ ] AMS wrapper for H(curl) systems
- [ ] ADS wrapper for H(div) systems
- [ ] Device policy hooks (CPU/GPU-aware behavior passthrough)
- [ ] No dependency on external hypre C library in default or advanced paths

## 3. PETSc-Equivalent (`petsc-rs`) Requirements (Pure Rust)

- [ ] KSP-equivalent lifecycle (`create -> set operators -> solve -> destroy`) in pure Rust
- [ ] PC options passthrough (AMG/ILU/fieldsplit)
- [ ] Matrix bridge for AIJ-equivalent and shell/operator mode
- [ ] Optional nonlinear path hooks (SNES) for future extension
- [ ] Optional eigen path handoff (SLEPc) reserved extension point
- [ ] No dependency on external PETSc C library in default or advanced paths

## 4. Native Direct-Compatibility Requirements (`mumps` / `mkl`)

- [ ] Sparse factorization bridge from `linger` matrix storage
- [ ] Numeric factor reuse across solves
- [ ] Multi-RHS solve support
- [ ] Ordering and pivoting controls exposed in builder config
- [ ] Deterministic cleanup and resource release

## 6. Delivery Order

1. `hypre-rs` minimal BoomerAMG-equivalent path
2. `hypre-rs` AMS/ADS and AIR options
3. native direct-compatibility hardening (`mumps` / `mkl`)
4. `petsc-rs` KSP/PC parity path

## 7. Done Criteria

- [ ] Each external backend has at least one integration test solving Poisson-like SPD system
- [ ] `fem-rs` can select backend through `linger` without app-level API changes
- [ ] Disabled feature path compiles cleanly and reports clear runtime errors

## 8. Milestone Execution Board

### M1 - Foundation + Minimal HYPRE-Equivalent (Pure Rust)

Target: establish common backend skeleton and deliver first usable pure-Rust BoomerAMG-equivalent path.

- [ ] Define backend-neutral solver handle traits in `linger` builder layer
- [ ] Implement shared backend error adapter (backend code -> `SolverError`)
- [ ] Add `hypre-rs` capability gate with clear runtime capability reporting
- [ ] Implement minimal ParCSR bridge for partitioned CSR data
- [ ] Implement BoomerAMG-equivalent `setup/apply` minimal path
- [ ] Add smoke integration test: Poisson SPD solve with BoomerAMG preconditioning

Exit criteria:

- [ ] `cargo test` passes with pure-Rust default settings
- [ ] `cargo test --features hypre-rs` passes in configured environment

### M2 - HYPRE-Equivalent Advanced + Direct Solvers

Target: cover production options and native direct-compatibility hardening.

- [ ] Expose BoomerAMG option set (coarsening/interp/relax/cycle)
- [ ] Add AIR option path for non-symmetric systems
- [ ] Add AMS wrapper path
- [ ] Add ADS wrapper path
- [ ] Harden native `mumps`/`mkl`-compatible paths in docs/examples as replacement contracts
- [ ] Keep builder/runtime reporting aligned with native replacement semantics

Exit criteria:

- [ ] HYPRE-equivalent advanced options validated by integration tests
- [ ] Native `mumps`/`mkl`-compatible replacement paths remain selectable from `SolverBuilder`

### M3 - MKL + PETSc-Equivalent + CI Matrix

Target: complete PETSc-equivalent path and operationalize compatibility matrix in CI.

- [ ] Implement PETSc-equivalent KSP/PC path (AIJ-equivalent + shell mode) in pure Rust
- [ ] Add backend capability flags reporting (`supports_multi_rhs`, `supports_nonsym_amg`, etc.)
- [ ] Add CI jobs for feature combinations with backend smoke tests
- [ ] Add backend selection examples used by `fem-rs` integration tests

Exit criteria:

- [ ] All backend feature gates compile independently
- [ ] At least one integration test per backend passes in CI with backend available

## 9. Implementation Notes

- Prefer introducing one external backend at a time behind a stable builder-facing API.
- Keep pure-Rust default behavior unchanged when optional FFI features are disabled.
- Any backend-specific matrix conversion should be isolated in dedicated bridge modules.

## 10. Cross-Subproject Coordination

This document is maintained in `linger`, but delivery is considered complete only when
integration points in `reed` and `jsmpi` are satisfied.

### 10.1 reed Integration Requirements

- [ ] Provide stable operator/matrix export bridge from `reed` objects into `linger` backend selection path
- [ ] Preserve current backend resource naming and selection behavior for CPU and future GPU paths
- [ ] Preserve `mkl` compatibility resource naming while routing to linger-native reporting
- [ ] Own GPU backend implementation milestones and publish capability matrix consumed by linger contract
- [ ] Add at least one `reed` integration test that selects an external solver backend via `linger`

### 10.2 jsmpi Integration Requirements

- [ ] Define wasm/browser compatibility policy for external backends (unsupported vs fallback)
- [ ] Ensure distributed execution path can route through `jsmpi` transport where native MPI is unavailable
- [ ] Add browser-oriented smoke test plan documenting expected fallback behavior

### 10.3 Coordination Exit Criteria

- [ ] `linger` backend selection API is consumable by `reed` without app-level API churn
- [ ] `jsmpi` constraints and fallback rules are documented for each backend feature
- [ ] Cross-project integration checklist is referenced by all three subprojects during milestone reviews
