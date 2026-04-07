# 特征值求解能力对比：linger vs ARPACK-NG vs SLEPc

> 文档版本：2026-04-07  
> linger 版本：v0.1.0（Sprint 12 完成后）

---

## 一、现状概述

| 维度 | ARPACK-NG | SLEPc | linger (v0.1.0) |
|------|-----------|-------|-----------------|
| **定位** | 大规模稀疏特征值 | HPC 特征值/SVD/矩阵函数 | 稀疏线性方程组 + 特征值（FEA 场景） |
| **特征值能力** | 是 | 是 | Sprint 7–12 全实现 |
| **核心语言** | Fortran/C | C/Fortran | Rust |
| **成熟度** | 极成熟（30+ 年） | 极成熟（20+ 年） | v0.1.0，开发中 |

---

## 二、特征值问题类型对比

| 问题类型 | 数学形式 | ARPACK-NG | SLEPc | linger |
|---------|---------|-----------|-------|--------|
| 标准特征值（SEP） | `Ax = λx` | ✓ | ✓ | ✓ (S7) |
| 广义特征值（GEP） | `Ax = λBx` | ✓ | ✓ | ✓ (S9) |
| 对称广义特征值 | A、B 对称正定 | ✓ | ✓ | ✓ (S9) |
| 不定广义特征值 | B 不定 | ✗ | ✓（GHIEP）| ✗ |
| 二次特征值（QEP） | `(K+λC+λ²M)x=0` | ✗ | ✓ | ✓ (S12) |
| 非线性特征值（NEP） | `T(λ)x = 0` | ✗ | ✓ | ✓ 基础 (S12) |
| 结构化特征值（Hamiltonian 等） | 保结构问题 | ✗ | ✓ | ✗ |
| 奇异值分解（SVD） | `A = UΣVᵀ` | 基础 | ✓（GSVD、HSVD）| ✓ (S11) |
| 矩阵函数作用 | `y = f(A)v` | ✗ | ✓（MFN 模块）| ✗ |

---

## 三、求解算法对比

### 3.1 特征值求解算法

| 算法 | ARPACK-NG | SLEPc | linger | 适用场景 |
|------|-----------|-------|--------|---------|
| 幂迭代（Power Iteration）| 基础 | ✓ | ✓ (S7) | 最大模单特征值 |
| 反幂迭代（Inverse Iteration）| — | ✓ | ✓ (S7) | 最小模/指定移位附近 |
| Rayleigh 商迭代（RQI）| — | ✓ | ✓ (S7) | 快速收敛单特征值，三次收敛 |
| 正交（子空间）迭代 | — | ✓ | ✓ (S7) | 多个最大特征值 |
| Implicitly Restarted Arnoldi (IRAM) | ✓ | ✓ | ✓ (S8) | 通用非对称稀疏 |
| Implicitly Restarted Lanczos (IRLM) | ✓ | ✓ | ✓ (S8) | 对称稀疏 |
| Krylov-Schur | ✗ | ✓（默认）| ✓ (S10) | 通用，最鲁棒 |
| Jacobi-Davidson (JD) | ✗ | ✓ | ✗ | 内部特征值 |
| Generalized Davidson (GD) | ✗ | ✓ | ✗ | 预条件对称问题 |
| LOBPCG | ✗ | ✓ | ✓ (S10) | 大规模对称正定（FEA 模态） |
| 轮廓积分（CISS） | ✗ | ✓ | ✗ | 区域内所有特征值 |
| TOAR / STOAR | ✗ | ✓（PEP）| 部分 (S12) | 多项式特征值（线性化） |
| NLEIGS | ✗ | ✓（NEP）| 基础 Newton (S12) | 非线性特征值 |
| Shift-Invert 变换 | ✓ | ✓（ST 对象）| ✓ (S7/S9) | 内部特征值加速 |

### 3.2 SVD 算法

| 算法 | ARPACK-NG | SLEPc | linger |
|------|-----------|-------|--------|
| Lanczos SVD（via AᵀA） | 部分 | ✓ | ✓ (S11) |
| Golub-Reinsch 双对角 | ✗ | ✓ | ✗（规划） |
| 广义 SVD（GSVD） | ✗ | ✓ | ✗ |

### 3.3 线性系统求解算法（linger 当前优势）

| 算法 | linger | ARPACK-NG | SLEPc |
|------|--------|-----------|-------|
| CG | ✓ | RCI 用户提供 | 通过 KSP |
| GMRES(m) | ✓ | RCI 用户提供 | 通过 KSP |
| BiCGStab | ✓ | RCI 用户提供 | 通过 KSP |
| MINRES | ✓ | — | 通过 KSP |
| FGMRES | ✓ | — | 通过 KSP |
| LGMRES | ✓ | — | 通过 KSP |
| 矩阵自由 GMRES（eigen 内部）| ✓ (S7) | — | — |

---

## 四、矩阵类型与数据类型对比

| 特性 | ARPACK-NG | SLEPc | linger |
|------|-----------|-------|--------|
| 实数（f32/f64）| f32、f64 | f64（编译时）| ✓（泛型 Scalar trait，f32/f64）|
| 复数 | ✓ | ✓ | ✓ ComplexScalar trait (S11) |
| 对称矩阵优化 | ✓ | ✓ | 部分（ICC、CG、Lanczos）|
| 非对称矩阵 | ✓ | ✓ | ✓ |
| 稀疏（CSR/CSC）| 透明（RCI）| ✓（AIJ=CSR）| ✓（CSR、CSC）|
| 稀疏（COO/BSR）| 透明 | ✓ | ✓（含 BSR）|
| 密集矩阵 | ✗ | ✓ | ✗ |
| 带状矩阵 | ✓（专用例程）| ✓ | ✗ |
| nalgebra 矩阵 | ✗ | ✗ | ✓（LinearOperator impl）|

---

## 五、并行与性能对比

| 能力 | ARPACK-NG | SLEPc | linger |
|------|-----------|-------|--------|
| 共享内存（多线程）| ✗（库本身无）| 通过 PETSc | ✓（Rayon；SpMV/AXPY 等）|
| 分布式内存（MPI）| ✓（PARPACK）| ✓（PETSc 基础）| 占位符，未实现（S13 规划）|
| GPU（CUDA/HIP）| ✗ | ✓ | ✗ |
| WASM | ✗ | ✗ | ✓（wasm32-unknown-unknown）|
| 谱切片并行 | ✗ | ✓ | ✗ |

---

## 六、生态与接口对比

| 能力 | ARPACK-NG | SLEPc | linger |
|------|-----------|-------|--------|
| Fortran | ✓（原生）| ✓ | ✗ |
| C/C++ | ✓ | ✓ | ✗ |
| Rust | ✗ | ✗ | ✓（原生）|
| Python | ✓（SciPy 后端）| ✓（slepc4py）| ✗ |
| JavaScript/WASM | ✗ | ✗ | ✓ |
| nalgebra 集成 | ✗ | ✗ | ✓（LinearOperator）|
| 外部库封装 | — | ARPACK、PRIMME、FEAST 等 | ✗ |

---

## 七、linger 开发规划

### 路线图总览

```
Sprint 7  ── 基础特征值框架 + 幂法族            [已完成]
Sprint 8  ── Krylov 子空间：Lanczos / Arnoldi  [已完成]
Sprint 9  ── 广义特征值 + Shift-Invert 框架    [已完成]
Sprint 10 ── Krylov-Schur、LOBPCG             [已完成]
Sprint 11 ── 复数支持 (ComplexScalar) + SVD   [已完成]
Sprint 12 ── QEP (二次特征值) + 基础 NEP       [已完成]
Sprint 13 ── MPI 并行 + GPU 后端（按需）
```

---

### Sprint 7（已完成）— 基础特征值框架

| 组件 | 文件 | 说明 |
|------|------|------|
| 核心 trait 与类型 | `eigen/mod.rs` | `EigenSolver`、`EigenParams`、`EigenResult`、`EigenWhich` |
| 矩阵自由 GMRES | `eigen/mod.rs` | `matfree_gmres()`，用于 InverseIter 内部线性系统 |
| 幂迭代 | `eigen/power.rs` | `PowerIter`，最大模单特征值 |
| 正交子空间迭代 | `eigen/subspace.rs` | `SubspaceIter`，k 个最大特征值 |
| 反幂迭代 | `eigen/inverse.rs` | `InverseIter`，最小模/指定移位附近 |
| Rayleigh 商迭代 | `eigen/inverse.rs` | `RayleighQuotientIter`，三次收敛，自适应移位 |
| 测试 | `tests/test_eigen.rs` | 11 个测试，覆盖 f32/f64，对角、2×2、1D Laplacian |

---

### Sprint 8（已完成）— Krylov 子空间特征值法

| 组件 | 文件 | 说明 |
|------|------|------|
| 隐式重启 Lanczos（IRLM）| `eigen/lanczos.rs` | 对称问题；完全重正交化；厚重启 |
| 隐式重启 Arnoldi（IRAM）| `eigen/arnoldi.rs` | 非对称问题；Francis QR |
| 测试 | `tests/test_eigen_s8_s10.rs` | 对角矩阵 + 1D Laplacian |

---

### Sprint 9（已完成）— 广义特征值 + Shift-Invert 框架

| 组件 | 文件 | 说明 |
|------|------|------|
| Shift-Invert Lanczos | `eigen/generalized.rs` | `ShiftInvertLanczos`，内部谱移位 |
| 广义特征值 | `eigen/generalized.rs` | `GeneralizedEigen`，`Ax=λBx` 变换为标准 EVP |

---

### Sprint 10（已完成）— 高级算法

| 算法 | 文件 | 说明 |
|------|------|------|
| Krylov-Schur | `eigen/krylov_schur.rs` | 比 IRAM 更稳定；对称/非对称通用 |
| LOBPCG | `eigen/lobpcg.rs` | FEA 模态分析黄金组合：LOBPCG + AMG |

---

### Sprint 11（已完成）— ComplexScalar trait + SVD

| 组件 | 文件 | 说明 |
|------|------|------|
| `ComplexScalar` trait | `core/scalar.rs` | `Complex<f32/f64>` + 实数 blanket impl；`type Real: Scalar` 关联类型 |
| `TransposeOperator` trait | `core/operator.rs` | `apply_transpose(x, y)`；`CsrMatrix` 实现 |
| `LanczosSvd` / `SvdResult` | `eigen/svd.rs` | Lanczos on AᵀA；σᵢ = √λᵢ；左/右奇异向量可选 |
| 测试 | `tests/test_eigen_s11_s12.rs` | `complex_scalar_ops`、`svd_diagonal_3x3`、`svd_laplacian_top2` 等 |

**公开 API 新增**：
```rust
use linger::{Complex, ComplexScalar, TransposeOperator, LanczosSvd, SvdResult};
```

---

### Sprint 12（已完成）— QEP + 基础 NEP 框架

| 组件 | 文件 | 说明 |
|------|------|------|
| `QuadraticEigen` | `eigen/qep.rs` | `(K+λC+λ²M)x=0` 伴随线性化 → 2n×2n → `ArnoldiIter` |
| `NonlinearOperator` trait | `eigen/nep.rs` | `apply_t(λ, v, out)` + `apply_dt`（有限差分默认） |
| `NepNewton` | `eigen/nep.rs` | Rayleigh 泛函更新 λ + 正则化反迭代更新 x |
| 测试 | `tests/test_eigen_s11_s12.rs` | 过阻尼 QEP、Newton 收敛验证（5 个特征值） |

**公开 API 新增**：
```rust
use linger::{QuadraticEigen, NonlinearOperator, NepNewton};
```

---

## 八、linger 相对于 ARPACK-NG / SLEPc 的差异化优势

1. **Shift-Invert 内置集成**：InverseIter 通过矩阵自由 GMRES 实现 Shift-Invert，零额外依赖
2. **AMG + LOBPCG 组合**：内置 AMG 预条件器直接加速大规模 FEA 模态分析
3. **ComplexScalar 统一泛型**：单套代码覆盖 f32/f64/Complex<f64> — 无需分别编译
4. **QEP + NEP 框架**：阻尼结构动力学 `(K+λC+λ²M)x=0` 一行代码求解
5. **WASM 运行时**：浏览器端特征值求解，ARPACK-NG 和 SLEPc 均无此能力
6. **Rust 安全性**：无 FFI 开销、内存安全、零 UB
7. **nalgebra 零成本集成**：nalgebra 矩阵直接实现 `LinearOperator`，无需任何包装

---

## 一、现状概述

| 维度 | ARPACK-NG | SLEPc | linger (v0.1.0) |
|------|-----------|-------|-----------------|
| **定位** | 大规模稀疏特征值 | HPC 特征值/SVD/矩阵函数 | 稀疏线性方程组（FEA 场景） |
| **特征值能力** | 是 | 是 | Sprint 7 起部分支持 |
| **核心语言** | Fortran/C | C/Fortran | Rust |
| **成熟度** | 极成熟（30+ 年） | 极成熟（20+ 年） | v0.1.0，开发中 |

---

## 二、特征值问题类型对比

| 问题类型 | 数学形式 | ARPACK-NG | SLEPc | linger |
|---------|---------|-----------|-------|--------|
| 标准特征值（SEP） | `Ax = λx` | ✓ | ✓ | ✓ (S7) |
| 广义特征值（GEP） | `Ax = λBx` | ✓ | ✓ | ✗ (S9) |
| 对称广义特征值 | A、B 对称正定 | ✓ | ✓ | ✗ (S9) |
| 不定广义特征值 | B 不定 | ✗ | ✓（GHIEP）| ✗ |
| 多项式特征值（PEP） | `(A₀+λA₁+…+λᵈAᵈ)x=0` | ✗ | ✓ | ✗ (S12) |
| 非线性特征值（NEP） | `T(λ)x = 0` | ✗ | ✓ | ✗ (S12) |
| 结构化特征值（Hamiltonian 等） | 保结构问题 | ✗ | ✓ | ✗ |
| 奇异值分解（SVD） | `A = UΣVᵀ` | 基础 | ✓（GSVD、HSVD）| ✗ (S11) |
| 矩阵函数作用 | `y = f(A)v` | ✗ | ✓（MFN 模块）| ✗ |

括号内为计划实现的 Sprint 编号。

---

## 三、求解算法对比

### 3.1 特征值求解算法

| 算法 | ARPACK-NG | SLEPc | linger | 适用场景 |
|------|-----------|-------|--------|---------|
| 幂迭代（Power Iteration）| 基础 | ✓ | ✓ (S7) | 最大模单特征值 |
| 反幂迭代（Inverse Iteration）| — | ✓ | ✓ (S7) | 最小模/指定移位附近 |
| Rayleigh 商迭代（RQI）| — | ✓ | ✓ (S7) | 快速收敛单特征值，三次收敛 |
| 正交（子空间）迭代 | — | ✓ | ✓ (S7) | 多个最大特征值 |
| Implicitly Restarted Arnoldi (IRAM) | ✓ | ✓ | ✗ (S8) | 通用非对称稀疏 |
| Implicitly Restarted Lanczos (IRLM) | ✓ | ✓ | ✗ (S8) | 对称稀疏 |
| Krylov-Schur | ✗ | ✓（默认）| ✗ (S10) | 通用，最鲁棒 |
| Jacobi-Davidson (JD) | ✗ | ✓ | ✗ (S10) | 内部特征值 |
| Generalized Davidson (GD) | ✗ | ✓ | ✗ (S10) | 预条件对称问题 |
| LOBPCG | ✗ | ✓ | ✗ (S10) | 大规模对称正定（FEA 模态） |
| 轮廓积分（CISS） | ✗ | ✓ | ✗ | 区域内所有特征值 |
| TOAR / STOAR | ✗ | ✓（PEP）| ✗ (S12) | 多项式特征值 |
| NLEIGS | ✗ | ✓（NEP）| ✗ (S12) | 非线性特征值 |
| Shift-Invert 变换 | ✓ | ✓（ST 对象）| 内置于 InverseIter (S7) | 内部特征值加速 |

### 3.2 线性系统求解算法（linger 当前优势）

linger 拥有完整的迭代线性求解器栈，这些与特征值求解深度集成（InverseIter 直接复用矩阵自由 GMRES）：

| 算法 | linger | ARPACK-NG | SLEPc |
|------|--------|-----------|-------|
| CG | ✓ | RCI 用户提供 | 通过 KSP |
| GMRES(m) | ✓ | RCI 用户提供 | 通过 KSP |
| BiCGStab | ✓ | RCI 用户提供 | 通过 KSP |
| MINRES | ✓ | — | 通过 KSP |
| FGMRES | ✓ | — | 通过 KSP |
| LGMRES | ✓ | — | 通过 KSP |
| 矩阵自由 GMRES（eigen 内部）| ✓ (S7) | — | — |

---

## 四、矩阵类型与数据类型对比

| 特性 | ARPACK-NG | SLEPc | linger |
|------|-----------|-------|--------|
| 实数（f32/f64）| f32、f64 | f64（编译时）| ✓（泛型 Scalar trait，f32/f64）|
| 复数 | ✓ | ✓ | ✗（S11 规划）|
| 对称矩阵优化 | ✓ | ✓ | 部分（ICC、CG、Lanczos）|
| 非对称矩�� | ✓ | ✓ | ✓ |
| 稀疏（CSR/CSC）| 透明（RCI）| ✓（AIJ=CSR）| ✓（CSR、CSC）|
| 稀疏（COO/BSR）| 透明 | ✓ | ✓（含 BSR）|
| 密集矩阵 | ✗ | ✓ | ✗ |
| 带状矩阵 | ✓（专用例程）| ✓ | ✗ |
| nalgebra 矩阵 | ✗ | ✗ | ✓（LinearOperator impl）|

---

## 五、并行与性能对比

| 能力 | ARPACK-NG | SLEPc | linger |
|------|-----------|-------|--------|
| 共享内存（多线程）| ✗（库本身无）| 通过 PETSc | ✓（Rayon；SpMV/AXPY 等）|
| 分布式内存（MPI）| ✓（PARPACK）| ✓（PETSc 基础）| 占位符，未实现（S13 规划）|
| GPU（CUDA/HIP）| ✗ | ✓ | ✗ |
| WASM | ✗ | ✗ | ✓（wasm32-unknown-unknown）|
| 谱切片并行 | ✗ | ✓ | ✗ |

---

## 六、生态与接口对比

| 能力 | ARPACK-NG | SLEPc | linger |
|------|-----------|-------|--------|
| Fortran | ✓（原生）| ✓ | ✗ |
| C/C++ | ✓ | ✓ | ✗ |
| Rust | ✗ | ✗ | ✓（原生）|
| Python | ✓（SciPy 后端）| ✓（slepc4py）| ✗ |
| JavaScript/WASM | ✗ | ✗ | ✓ |
| nalgebra 集成 | ✗ | ✗ | ✓（LinearOperator）|
| 外部库封装 | — | ARPACK、PRIMME、FEAST 等 | ✗ |

---

## 七、linger 开发规划

### 路线图总览

```
Sprint 7  ── 基础特征值框架 + 幂法族            [已完成]
Sprint 8  ── Krylov 子空间：Lanczos / Arnoldi
Sprint 9  ── 广义特征值 + Shift-Invert 框架
Sprint 10 ── Krylov-Schur、LOBPCG、JD
Sprint 11 ── 复数支持 + SVD
Sprint 12 ── PEP/NEP（二次特征值 QEP）
Sprint 13 ── MPI 并行 + GPU 后端（按需）
```

---

### Sprint 7（已完成）— 基础特征值框架

**新增模块**：`src/eigen/`

| 组件 | 文件 | 说明 |
|------|------|------|
| 核心 trait 与类型 | `eigen/mod.rs` | `EigenSolver`、`EigenParams`、`EigenResult`、`EigenWhich` |
| 矩阵自由 GMRES | `eigen/mod.rs` | `matfree_gmres()`，用于 InverseIter 内部线性系统 |
| 幂迭代 | `eigen/power.rs` | `PowerIter`，最大模单特征值 |
| 正交子空间迭代 | `eigen/subspace.rs` | `SubspaceIter`，k 个最大特征值 |
| 反幂迭代 | `eigen/inverse.rs` | `InverseIter`，最小模/指定移位附近 |
| Rayleigh 商迭代 | `eigen/inverse.rs` | `RayleighQuotientIter`，三次收敛，自适应移位 |
| 测试 | `tests/test_eigen.rs` | 11 个测试，覆盖 f32/f64，对角、2×2、1D Laplacian |

**公开 API（通过 `linger::` 根重导出）**：
```rust
use linger::{
    EigenParams, EigenResult, EigenSolver, EigenWhich,
    PowerIter, SubspaceIter, InverseIter, RayleighQuotientIter,
};
```

---

### Sprint 8 — Krylov 子空间特征值法

**目标**：达到 ARPACK-NG 核心能力

| 组件 | 文件 | 说明 |
|------|------|------|
| Lanczos 三对角化 | `eigen/lanczos.rs` | 含完全/选择性重正交化 |
| 隐式重启 Lanczos（IRLM）| `eigen/lanczos.rs` | 对标 ARPACK `dsaupd`；对称问题 |
| Arnoldi 上 Hessenberg 化 | `eigen/arnoldi.rs` | 非对称问题 |
| 隐式重启 Arnoldi（IRAM）| `eigen/arnoldi.rs` | 对标 ARPACK `dnaupd` |

---

### Sprint 9 — 广义特征值 + Shift-Invert 框架

**目标**：支持 `Ax = λBx`，统一谱变换抽象

```rust
pub trait SpectralTransform<T: Scalar> { ... }
pub struct ShiftInvertST<T, Solver: KrylovSolver<T>> { shift: T, solver: Solver }
pub struct GeneralizedEigenSolver { ... }
```

**关键**：`ShiftInvertST` 复用 linger 已有线性求解器，求解 `(A − σB)⁻¹B` 的最大特征值。

---

### Sprint 10 — 高级算法

| 算法 | 优先级 | 说明 |
|------|--------|------|
| Krylov-Schur | P0 | 比 IRAM 更稳定；对称/非对称通用 |
| LOBPCG | P0 | FEA 模态分析黄金组合：LOBPCG + AMG |
| Jacobi-Davidson | P1 | 内部特征值，带预条件修正方程 |

> **战略价值**：LOBPCG + linger 内置 AMG 预条件器是 FEA 结构模态分析的最优路径，这是 linger 对比 ARPACK-NG 的差异化功能。

---

### Sprint 11 — 复数支持与 SVD

扩展 `Scalar` trait 为 `ComplexScalar`，实现 SVD（通过 Golub-Reinsch 双对角化）。

---

### Sprint 12 — 多项式与非线性特征值

| 功能 | 对标 SLEPc | FEA 场景 |
|------|-----------|---------|
| 二次特征值（QEP）| `PEP_TOAR` | 阻尼结构动力学：`(K + λC + λ²M)x = 0` |
| TOAR（二阶 Arnoldi）| `PEP_TOAR` | 保结构、节省内存 |
| NLEIGS | `NEP_NLEIGS` | 有理插值，无需导数 |

---

## 八、linger 相对于 ARPACK-NG / SLEPc 的差异化优势

1. **Shift-Invert 内置集成**：InverseIter 通过矩阵自由 GMRES 实现 Shift-Invert，零额外依赖，已在 Sprint 7 交付
2. **AMG + LOBPCG 组合**（Sprint 10）：内置 AMG 预条件器直接加速大规模 FEA 模态分析
3. **WASM 运行时**：浏览器端特征值求解，ARPACK-NG 和 SLEPc 均无此能力
4. **Rust 安全性**：无 FFI 开销、内存安全、零 UB
5. **泛型精度**：同一套代码 f32/f64 无缝切换，适合嵌入式/实时仿真
6. **nalgebra 零成本集成**：nalgebra 矩阵直接实现 `LinearOperator`，无需任何包装
