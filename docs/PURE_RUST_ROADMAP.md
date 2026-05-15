# Pure Rust Solver Roadmap (linger)

Date: 2026-04-19

## Positioning

`linger` is a pure Rust sparse linear solver library.

This roadmap tracks only native solver capabilities and scale hardening.
No external-solver equivalence tracks are part of the active plan.

## Current Scope

- Krylov iterative solvers (CG/GMRES/FGMRES/BiCGSTAB/MINRES/LGMRES/IDR(s)/TFQMR)
- AMG hierarchy and cycle variants
- Native direct solver compatibility routes (`mumps` / `mkl` naming)
- WASM-compatible core solver layer
- Distributed-memory foundation (`mpi` placeholder, staged implementation)

## Near-Term Priorities

1. Large-scale robustness hardening for AMG options (AIR/AMS/ADS already landed as baseline pieces)
2. Distributed-memory solver path completion and benchmark gates
3. Native direct-solver factor reuse and multi-RHS optimization
4. CI matrix hardening for baseline + `mumps` + `mkl` profiles

## Exit Criteria for Current Cycle

- Deterministic solver behavior across supported targets
- Repeatable benchmark reports for convergence and runtime trends
- Stable CI for default and compatibility feature profiles
