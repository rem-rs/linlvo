# 技术规范文档：linger

**项目定位**：Rust 通用有限元分析框架（linger）的线性方程组求解子库
**版本**：v0.1.0-draft
**日期**：2026-03-30

---

## 1. 项目概述

### 1.1 目标

`linger` 是一个纯 Rust 实现的线性方程组求解库，提供与 HYPRE、PETSc 功能等价的算法体系，同时以 trait 抽象兼容 `nalgebra` 和 `faer` 矩阵类型。其核心目标是：

- 为 FEA 装配产生的大规模稀疏线性系统提供高效求解
- 提供结构清晰的 Krylov 迭代器 + 预条件器组合框架
- 支持代数多重网格（AMG）等 FEA 特有的高效预条件策略
- **支持编译为 WebAssembly（wasm32-unknown-unknown）**：核心 solver 层（`core/`、`sparse/`、`iterative/`、`precond/`）可在浏览器或 WASM 运行时中运行，用于 Web 端轻量 FEA 工具或可视化仿真
- 长期目标：提供可选的 HYPRE/PETSc C 库 FFI 绑定作为后端

### 1.2 参照系统

| 参照系统 | 主要贡献 | 本库对应实现策略 |
|---------|---------|----------------|
| HYPRE   | BoomerAMG、ParaSails、Euclid ILU、结构化网格多重网格 | 纯 Rust 实现等价算法；可选 FFI feature |
| PETSc   | KSP（Krylov）、PC（预条件）、Mat/Vec 抽象、并行分布式对象 | trait 体系 + rayon 共享内存并行；MPI 留接口 |

### 1.3 约束与假设

- Rust Edition 2021，MSRV = 1.80
- 初期目标：单节点共享内存并行（rayon）
- 矩阵规模假设：中等规模（$10^4$–$10^7$ 自由度）；WASM 场景下以小规模（< $10^4$ DOF）为主
- 数值精度：默认 `f64`，泛型支持 `f32`
- 禁止 unsafe 的范围：核心 solver trait 层全部 safe；FFI 绑定层隔离在单独 crate
- **WASM 兼容性约束**：
  - `rayon` 并行在 WASM 目标下自动禁用（`#[cfg(not(target_arch = "wasm32"))]`）
  - FFI 后端（hypre-ffi、petsc-ffi）在 WASM 目标下不可用
  - 核心算法层禁止依赖系统线程、文件 I/O 等不可移植 API
  - `std::time` 计时在 WASM 下替换为可选 feature 或忽略

---

## 2. 功能范围（Feature Scope）

### 2.1 模块划分

```
linger
├── core/           # 抽象 trait 与基础类型
├── sparse/         # 稀疏矩阵格式与 BLAS
├── direct/         # 直接法求解器
├── iterative/      # Krylov 迭代法
├── precond/        # 预条件器
├── amg/            # 代数多重网格
├── parallel/       # 并行基础设施
└── ffi/            # 可选：HYPRE/PETSc FFI（feature-gated）
```

### 2.2 功能清单（对标 HYPRE + PETSc）

#### 2.2.1 稀疏矩阵格式（对标 PETSc Mat）

| 格式 | 说明 | 优先级 |
|------|------|--------|
| CSR  | Compressed Sparse Row，通用稀疏格式 | P0 |
| CSC  | Compressed Sparse Column | P0 |
| COO  | Coordinate format，装配阶段使用 | P0 |
| BSR  | Block Sparse Row，FEA 块结构优化 | P1 |
| 对称格式（CSS） | 对称问题节省存储 | P1 |
| 分布式 CSR | MPI 分块行存储 | P2 |

#### 2.2.2 直接求解器（对标 PETSc PCLU/PCCHOLESKY）

| 求解器 | 说明 | 优先级 |
|--------|------|--------|
| Dense LU | 通过 faer/nalgebra | P0 |
| Dense Cholesky | 通过 faer/nalgebra | P0 |
| Sparse LU（KLU 算法） | 适合中等规模、电路/结构问题 | P1 |
| Sparse Cholesky（supernodal） | 对称正定大规模问题 | P1 |
| MUMPS FFI | 可选外部库绑定 | P2 |
| SuperLU FFI | 可选外部库绑定 | P2 |

#### 2.2.3 Krylov 迭代法（对标 PETSc KSP、HYPRE Krylov）

| 方法 | 适用问题 | 优先级 |
|------|---------|--------|
| CG（Conjugate Gradient） | 对称正定 | P0 |
| MINRES | 对称不定 | P0 |
| GMRES(m) | 非对称，重启版本 | P0 |
| BiCGSTAB | 非对称，稳定化 | P0 |
| FGMRES | 非对称，flexible（变预条件） | P1 |
| LGMRES | 改进 GMRES，复用 Krylov 子空间 | P1 |
| PIPECG | 流水线 CG，减少全局规约通信 | P1 |
| GCROT(m,k) | 循环 GMRES 变体 | P2 |
| LSQR / LSMR | 最小二乘问题 | P2 |

#### 2.2.4 预条件器（对标 PETSc PC、HYPRE preconditioners）

| 预条件器 | 说明 | 优先级 |
|---------|------|--------|
| Jacobi（对角缩放） | 最基础 | P0 |
| Block Jacobi | 块对角 | P0 |
| SOR / SSOR | 逐次超松弛 | P0 |
| ILU(0) | 零填充不完全 LU | P0 |
| ILU(k) | k 级填充 ILU | P1 |
| ILUT | 阈值 ILU（对标 HYPRE Euclid） | P1 |
| ICC(0) / ICC(k) | 不完全 Cholesky | P1 |
| SPAI | 稀疏近似逆（对标 HYPRE ParaSails） | P1 |
| AMG 作为预条件 | 嵌套 AMG 预条件 | P1 |
| Composite PC | 加性/乘性组合（对标 PETSc PCCOMPOSITE） | P1 |
| Fieldsplit PC | 块分裂（对标 PETSc PCFIELDSPLIT） | P2 |

#### 2.2.5 代数多重网格（对标 HYPRE BoomerAMG）

| 组件 | 说明 | 优先级 |
|------|------|--------|
| 经典 Ruge-Stüben 粗化 | C/F 分裂、强连接图 | P1 |
| Smoothed Aggregation（SA-AMG） | 聚合型，适合弹性问题 | P1 |
| 插值算子构造（直接/标准/扩展+i） | | P1 |
| 平滑器：Jacobi、Gauss-Seidel、Chebyshev | | P1 |
| V-cycle、W-cycle、F-cycle | | P1 |
| 全近似格式（FAS，非线性 MG） | | P2 |
| 几何 MG（GMG） | 结合网格层次 | P2 |

#### 2.2.6 并行基础设施

| 功能 | 说明 | 优先级 |
|------|------|--------|
| rayon 数据并行 | SpMV、向量操作并行化 | P0 |
| 线程安全 solver 接口 | `Send + Sync` 约束 | P0 |
| MPI 接口抽象（trait） | 为分布式预留 | P1 |
| rsmpi 集成 | 实际 MPI 绑定 | P2 |

---

## 3. 接口设计规范

### 3.1 核心 Trait 体系

```rust
/// 向量抽象
pub trait Vector: Clone + Send + Sync {
    type Scalar: Scalar;
    fn len(&self) -> usize;
    fn dot(&self, other: &Self) -> Self::Scalar;
    fn axpy(&mut self, alpha: Self::Scalar, x: &Self);
    fn scale(&mut self, alpha: Self::Scalar);
    fn norm2(&self) -> Self::Scalar;
    fn zero_like(&self) -> Self;
}

/// 线性算子抽象（矩阵或矩阵自由算子）
pub trait LinearOperator: Send + Sync {
    type Vector: Vector;
    /// y = A * x
    fn apply(&self, x: &Self::Vector, y: &mut Self::Vector);
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

/// 预条件器：线性算子的特化
pub trait Preconditioner: LinearOperator {
    /// M^{-1} x -> y
    fn apply_precond(&self, x: &Self::Vector, y: &mut Self::Vector);
    /// 可选：更新内部状态（如 AMG 的 setup 阶段）
    fn setup(&mut self, op: &dyn LinearOperator<Vector = Self::Vector>) {}
}

/// Krylov 求解器 trait
pub trait KrylovSolver: Send + Sync {
    type Vector: Vector;
    type Operator: LinearOperator<Vector = Self::Vector>;

    fn solve(
        &self,
        op: &Self::Operator,
        precond: Option<&dyn Preconditioner<Vector = Self::Vector>>,
        b: &Self::Vector,
        x: &mut Self::Vector,
        params: &SolverParams,
    ) -> SolverResult;
}

/// 求解参数
pub struct SolverParams {
    pub rtol: f64,          // 相对残差容忍度
    pub atol: f64,          // 绝对残差容忍度
    pub max_iter: usize,
    pub verbose: VerboseLevel,
}

/// 求解结果
pub struct SolverResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
    pub history: Option<Vec<f64>>,
}
```

### 3.2 矩阵适配层

nalgebra 和 faer 矩阵类型通过 newtype wrapper 实现 `LinearOperator` trait，不侵入上游库：

```rust
// faer 稀疏矩阵适配
pub struct FaerSparseAdapter<T>(faer::sparse::SparseColMat<usize, T>);
impl<T: Scalar> LinearOperator for FaerSparseAdapter<T> { ... }

// nalgebra 稀疏矩阵适配
pub struct NalgebraCsrAdapter<T>(nalgebra_sparse::CsrMatrix<T>);
impl<T: Scalar> LinearOperator for NalgebraCsrAdapter<T> { ... }
```

### 3.3 Builder 模式的求解器配置

对标 PETSc 的 `KSPSetType` + `KSPSetPC` 组合：

```rust
let solver = SolverBuilder::new()
    .method(KrylovMethod::GMRES { restart: 30 })
    .preconditioner(PrecondType::ILU { fill_level: 1 })
    .rtol(1e-8)
    .max_iter(1000)
    .build::<f64>()?;

let result = solver.solve(&matrix, &rhs, &mut x)?;
```

---

## 4. 依赖规划

### 4.1 必要依赖

| crate | 用途 | 版本约束 |
|-------|------|---------|
| `nalgebra` | 密集矩阵/向量，核心 FEA 数值 | >= 0.33 |
| `nalgebra-sparse` | CSR/CSC 格式 | >= 0.10 |
| `faer` | 高性能密集/稀疏线性代数 | >= 0.21 |
| `rayon` | 数据并行 | >= 1.10 |
| `thiserror` | 错误类型 | >= 2.0 |
| `num-traits` | 泛型数值 trait | >= 0.2 |

### 4.2 可选依赖（feature-gated）

| crate/库 | Feature 名 | 用途 |
|---------|-----------|------|
| `rsmpi` | `mpi` | MPI 分布式并行 |
| HYPRE C 库 | `hypre-ffi` | BoomerAMG 外部后端 |
| PETSc C 库 | `petsc-ffi` | KSP/PC 外部后端 |
| MUMPS | `mumps` | 大规模稀疏直接法 |
| `intel-mkl-src` | `mkl` | MKL 加速 BLAS |
| `wasm-bindgen` | `wasm` | WASM JS 绑定（暴露 solver API 给 JavaScript） |
| `console_error_panic_hook` | `wasm` | WASM 环境下 panic 信息重定向到 `console.error` |

### 4.3 Cargo.toml 结构

```toml
[package]
name = "linger"
version = "0.1.0"
edition = "2021"
rust-version = "1.80"

[features]
default = ["rayon"]
mpi = ["dep:rsmpi"]
hypre-ffi = ["dep:hypre-sys"]
petsc-ffi = ["dep:petsc-sys"]
mumps = ["dep:mumps-sys"]
mkl = ["dep:intel-mkl-src"]
wasm = ["dep:wasm-bindgen", "dep:console_error_panic_hook"]

[dependencies]
nalgebra = { version = "0.33", features = ["sparse"] }
faer = "0.21"
rayon = { version = "1.10", optional = true }
thiserror = "2"
num-traits = "0.2"

[dependencies.nalgebra-sparse]
version = "0.10"
```

---

## 5. 性能目标

| 场景 | 规模 | 目标性能 |
|------|------|---------|
| CSR SpMV（单线程） | 10^6 行，1% 密度 | > 1 GFLOP/s |
| CG + AMG 预条件 | 10^6 DOF，Poisson 方程 | < 10s 到 1e-8 |
| GMRES(30) + ILU(1) | 10^5 DOF | < 1s |
| 并行 SpMV（8 线程） | 10^7 行 | 线性加速比 > 6x |

---

## 6. 质量要求

- 每个 solver/precond 必须有对应的收敛性测试（基于已知解的制造解 MMS）
- 数值精度：最终残差 $\|b - Ax\| / \|b\| < \epsilon_{tol}$，测试中验证
- CI：每次 PR 运行完整测试套件 + benchmark 回归检测
- 文档：每个公开 API 必须有 rustdoc 示例

---

## 7. 阶段里程碑

| 里程碑 | 内容 | 条件 |
|--------|------|------|
| M1 | 稀疏矩阵格式 + trait 抽象 + nalgebra/faer 适配 | P0 完成 |
| M2 | CG / GMRES / BiCGSTAB + Jacobi/ILU(0) | M1 完成 |
| M3 | ILUT / SPAI / AMG（SA-AMG） | M2 完成 |
| M4 | rayon 并行化 + 性能基准 | M2 完成 |
| M5 | HYPRE/PETSc FFI 可选后端 | M3 完成 |
