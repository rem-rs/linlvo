# 直接求解器能力对比与开发规划：linger vs SuperLU vs STRUMPACK

> 文档版本：2026-04-07  
> linger 版本：v0.2.0（Sprint 15 完成后）  
> 参照：SuperLU 7.0.1 / SuperLU_DIST 9.0.0 / STRUMPACK 7.2.0

---

## 一、现状概述

### 1.1 三方定位

| 维度 | SuperLU 系列 | STRUMPACK | linger (v0.2.0) |
|------|-------------|-----------|-----------------|
| **定位** | 通用稀疏直接求解器 | 结构化矩阵稀疏直接求解器 + 预条件器 | FEA 稀疏线性系统迭代求解库 |
| **核心算法** | 超节点 LU 分解 | 多波前 LU + HSS/BLR 低秩压缩 | Krylov 迭代法 + AMG 预条件 + 纯 Rust 直接法 |
| **直接求解能力** | ✓ 完整 | ✓ 完整（含近似模式） | ✓ 中小规模（SparseLu / SparseCholesky / MultifrontalLu，Sprint 13–15）|
| **核心语言** | C / Fortran | C++ | Rust |
| **成熟度** | 极成熟（30+ 年）| 成熟（10+ 年）| v0.2.0，开发中 |
| **许可证** | BSD 3-Clause | BSD 3-Clause | —（自有）|

### 1.2 linger 当前的"类直接"能力

linger 目前已实现的不完全分解预条件器在算法上与直接法密切相关：

| 已实现组件 | 文件 | 与直接法的关系 |
|-----------|------|--------------|
| ILU(0) | `precond/ilu0.rs` | 零填充不完全 LU，直接法的极端稀疏化近似 |
| ILU(k) | `precond/iluk.rs` | k 级填充 ILU，k→∞ 时退化为精确 LU |
| ILUT(τ,p) | `precond/ilut.rs` | 阈值 ILU，双参数控制填充 |
| ICC(0) | `precond/icc.rs` | 不完全 Cholesky，SPD 问题的不完全直接法 |

**已实现**：精确 LU 分解（`SparseLu<T>`，含列部分主元，Sprint 13）、精确 Cholesky（`SparseCholesky<T>`，稀疏左视，Sprint 14）、多波前 LU（`MultifrontalLu<T>`，Sprint 15）、`DirectSolver` trait、`DirectSolverPrecond` 包装器、RCM 重排序、COLAMD 重排序。

**仍缺失**：外部直接法 FFI 绑定（SuperLU、STRUMPACK）、METIS 嵌套剖分、超节点 LU（无超节点聚合）、MPI 分布式直接法。

---

## 二、算法能力对比

### 2.1 分解算法

| 算法 | SuperLU | SuperLU_MT | SuperLU_DIST | STRUMPACK | linger |
|------|---------|------------|--------------|-----------|--------|
| **精确 LU（超节点）** | ✓ | ✓ | ✓ | ✗（多波前）| ✗ |
| **精确 LU（多波前）** | ✗ | ✗ | ✗ | ✓（默认）| ✓（MultifrontalLu，Sprint 15）|
| **精确 Cholesky** | ✗ | ✗ | ✗ | ✗ | ✓（SparseCholesky，Sprint 14）|
| **BLR 近似多波前** | ✗ | ✗ | ✗ | ✓ | ✓（参数接口，Sprint 15）|
| **HSS 近似多波前** | ✗ | ✗ | ✗ | ✓ | ✗ |
| **HODLR/HODBF 近似** | ✗ | ✗ | ✗ | ✓（ButterflyPACK）| ✗ |
| **ILU(0)** | ✗ | ✗ | ✗ | ✗ | ✓ |
| **ILU(k)** | ✗ | ✗ | ✗ | ✗ | ✓ |
| **ILUT** | ✗ | ✗ | ✗ | ✗ | ✓ |
| **ICC(0)** | ✗ | ✗ | ✗ | ✗ | ✓ |

#### 超节点 LU vs 多波前 LU 的核心差异

| 维度 | 超节点 LU（SuperLU） | 多波前 LU（STRUMPACK） |
|------|---------------------|----------------------|
| **消元方式** | 列方向左视算法，列聚合为超节点 | 消元树自底向上，每节点组装稠密波前矩阵 |
| **BLAS 利用率** | BLAS-3（超节点间更新）| BLAS-3（波前 DGEMM，块更均匀）|
| **GPU 适配性** | 中等（超节点大小不均匀）| 优秀（波前矩阵结构规则）|
| **低秩压缩入口** | 难（超节点边界不规则）| 天然（波前非对角块）|
| **内存访问模式** | 稀疏向量 + 超节点密集块 | 逐波前稠密操作 |

### 2.2 求解模式

| 模式 | SuperLU 系列 | STRUMPACK | linger |
|------|-------------|-----------|--------|
| **直接精确求解** | ✓ | ✓（`--sp_compression none`）| ✓（SparseLu / SparseCholesky / MultifrontalLu）|
| **近似直接求解** | ✗ | ✓（BLR/HSS 单次前/回代）| ✓（MultifrontalLu + BLR 参数）|
| **作为预条件器** | ✗（本身无迭代框架）| ✓（GMRES/BiCGStab + 近似分解）| ✓（DirectSolverPrecond，精确分解 1-3 次迭代）|
| **多右端项批求解** | ✓（DIST v9+ 批处理）| ✓ | ✓（solve_multi，循环三角求解）|
| **迭代精化** | ✓（内置）| ✓（`--sp_Krylov_solver refine`）| ✓（DirectOptions.refine_steps）|
| **矩阵自由求解** | ✗ | ✗ | ✓（Krylov 本质特性）|

### 2.3 主元选取与数值稳定性策略

| 策略 | SuperLU 系列 | STRUMPACK | linger |
|------|-------------|-----------|--------|
| **完全主元选取** | ✗ | ✗ | ✗ |
| **列部分主元** | ✓（serial/MT）| ✗ | ✓（SparseLu / MultifrontalLu）|
| **阈值主元（DiagPivotThresh）** | ✓ | — | ✓（DirectOptions.pivot_threshold）|
| **静态主元（DIST）** | ✓（并行可扩展）| — | ✗ |
| **MC64 最大权重匹配** | ✓（DIST LargeDiag_MC64）| ✓（MAX_DIAGONAL_PRODUCT_SCALING）| ✗ |
| **近似权重匹配（AWPM）** | ✓（DIST，替代 MC64）| ✗ | ✗ |
| **无主元（对角占优假设）** | ✗ | ✗ | ✓（ILU 系列依赖对角占优）|

---

## 三、矩阵类型与数据类型对比

### 3.1 矩阵格式

| 格式 | SuperLU（输入）| STRUMPACK（输入）| linger（原生）|
|------|--------------|----------------|--------------|
| **CSC** | ✓（原生）| ✓（已废弃，向后兼容）| ✓ |
| **CSR** | 需转换 | ✓（原生，0-based）| ✓（主格式）|
| **COO** | 需转换 | 需转换 | ✓（装配格式）|
| **BSR** | ✗ | ✗ | ✓ |
| **分布式 CSR** | ✓（DIST）| ✓（块行分布）| 占位符 |

### 3.2 标量类型

| 类型 | SuperLU 系列 | STRUMPACK | linger |
|------|-------------|-----------|--------|
| **f32** | ✓（SLU_S）| ✓（STRUMPACK_FLOAT）| ✓（泛型 Scalar）|
| **f64** | ✓（SLU_D）| ✓（STRUMPACK_DOUBLE）| ✓（默认）|
| **Complex\<f32\>** | ✓（SLU_C）| ✓（STRUMPACK_FLOATCOMPLEX）| ✓（ComplexScalar，S11）|
| **Complex\<f64\>** | ✓（SLU_Z）| ✓（STRUMPACK_DOUBLECOMPLEX）| ✓（ComplexScalar，S11）|
| **混合精度（f32→f64）** | ✓（DIST v8+）| ✗ | ✗ |

### 3.3 矩阵结构利用

| 结构特性 | SuperLU 系列 | STRUMPACK | linger |
|---------|-------------|-----------|--------|
| **对称性利用（存储）** | ✗（必须提供完整矩阵）| ✗ | 部分（ICC 仅存下三角）|
| **SPD Cholesky** | ✗（使用 LU，非 LL^T）| ✗ | ✓（ICC 近似）|
| **对称模式（数值提示）** | ✓（SymmetricMode 启发）| — | — |
| **低秩结构利用** | ✗ | ✓（BLR/HSS 压缩波前）| ✗ |
| **块结构（BSR）** | ✗ | ✗ | ✓ |

---

## 四、重排序算法对比

重排序对直接法的内存和速度至关重要（2D 问题可减少填充 3-10×）。

| 排序算法 | SuperLU | SuperLU_DIST | STRUMPACK | linger |
|---------|---------|--------------|-----------|--------|
| **自然序** | ✓ | ✓ | ✓ | ✓ |
| **COLAMD** | ✓（主要）| ✓ | ✗ | ✓（纯 Rust，Sprint 13）|
| **MMD_ATA** | ✓ | ✓ | ✗ | ✗ |
| **MMD_AT+A** | ✓ | ✓ | ✗ | ✗ |
| **METIS（嵌套剖分）** | ✗ | ✓ | ✓ | ✗（规划项，需 feature-gated）|
| **ParMETIS** | ✗ | ✓ | ✓ | ✗ |
| **SCOTCH** | ✗ | ✓（PT-Scotch）| ✓（串行 + PT-Scotch）| ✗ |
| **RCM（反 Cuthill-McKee）** | ✗ | ✗ | ✓ | ✓（纯 Rust，Sprint 13）|
| **几何排序（规则网格）** | ✗ | ✗ | ✓（GEOMETRIC）| ✗ |

> **对 FEA 的影响**：嵌套剖分（METIS）对非结构化 FEA 网格的填充减少效果远优于 COLAMD（3-10×），是大规模直接求解的关键。linger 现已实现 COLAMD 和 RCM（Sprint 13），METIS 为规划项（需 feature-gated 外部依赖）。

---

## 五、并行能力对比

| 能力 | SuperLU | SuperLU_MT | SuperLU_DIST | STRUMPACK | linger |
|------|---------|------------|--------------|-----------|--------|
| **单线程** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **共享内存（OpenMP）** | ✗ | ✓（Pthreads/OMP）| ✓（OMP）| ✓（OMP ≥3.1，task dep）| ✓（Rayon）|
| **分布式（MPI）** | ✗ | ✗ | ✓（2D/3D 进程网格）| ✓（块行分布）| 占位符 |
| **NVIDIA GPU（CUDA）** | ✗ | ✗ | ✓（分解 + 求解）| ✓（精确直接）| ✗ |
| **AMD GPU（HIP）** | ✗ | ✗ | ✓ | ✓ | ✗ |
| **压缩部分 GPU 加速** | N/A | N/A | N/A | ✗（CPU only，v7.2.0）| N/A |
| **通信规避 3D 分解** | ✗ | ✗ | ✓（v8+）| ✗ | ✗ |
| **批处理求解** | ✗ | ✗ | ✓（v9+，MAGMA）| ✓ | ✗ |
| **WASM** | ✗ | ✗ | ✗ | ✗ | ✓ |

### SuperLU_DIST 的 GPU 并行架构（v9.0.0）

```
分解阶段（GPU 加速）:
  ├── cuBLAS DGEMM → Schur 补更新（占分解总 FLOP 的 ~70%）
  ├── cuSOLVER → 对角块 LU 分解
  └── 批处理 MAGMA 核 → 共享稀疏结构的多系统批量分解（v9.0.0 新增）

求解阶段（GPU 加速）:
  └── 3D 稀疏三角求解（v8.0.0 引入）
```

### STRUMPACK 的 GPU 加速说明

```
精确直接模式（--sp_compression none）：
  └── GPU 加速波前 DGEMM（~5× 加速 vs SuperLU_DIST，V100 基准）

近似预条件模式（--sp_compression blr/hss）：
  └── 当前版本（7.2.0）：CPU only（GPU 加速 BLR 见 IJHPCA 2025 论文，
      完整 GPU 路径待后续版本合并）
```

---

## 六、FEA 场景适用性对比

### 6.1 问题规模与内存估算

对于 3D 泊松方程（结构化网格，$n = N^3$ 自由度，稀疏度 ≈ 7 NNZ/行）：

| 规模（DOF）| SuperLU/DIST 内存（精确 LU）| STRUMPACK+BLR 内存（近似）| STRUMPACK+HSS 内存（近似）| linger（CG+AMG）内存 |
|-----------|--------------------------|--------------------------|--------------------------|----------------------|
| 10^4 | ~100 MB | ~30 MB | ~20 MB | ~10 MB |
| 10^5 | ~5 GB | ~500 MB | ~300 MB | ~50 MB |
| 10^6 | ~500 GB（不可行）| ~5 GB | ~2 GB | ~500 MB |
| 10^7 | 不可行 | ~50 GB | ~20 GB | ~5 GB |

> **核心结论**：精确直接法在 3D FEA（>10^5 DOF）场景下内存开销极大，而 linger 的 CG+AMG 在 10^7 DOF 仍可运行。STRUMPACK 的 BLR/HSS 压缩提供了两者之间的平衡点：牺牲少量精度换取 2-3× 内存节省，配合外迭代使用。

### 6.2 各求解策略适用场景

| 场景 | 推荐策略 | 说明 |
|------|---------|------|
| 小规模（< 10^4 DOF），精确解 | SuperLU（串行）或 linger 精确 LU（待实现）| 内存不是瓶颈，直接法一次成功 |
| 中等规模（10^4–10^5 DOF），SPD | linger CG + AMG 或 STRUMPACK+BLR 预条件 | AMG 在椭圆问题上最优；BLR 精度更高 |
| 中等规模，非对称 | linger GMRES + ILUT 或 STRUMPACK+BLR | STRUMPACK 对难问题更稳定 |
| 大规模（> 10^6 DOF），3D FEA | linger CG/FGMRES + AMG（当前最优选）| 直接法内存不可行 |
| 大规模，需高精度单次求解 | STRUMPACK+BLR+GMRES（外迭代）| AMG 不一定收敛（各向异性）|
| 模态分析（大量右端项）| linger LOBPCG + AMG 或 STRUMPACK 批求解 | 直接法摊销分解代价优势显著 |
| 多物理耦合（块结构）| linger Fieldsplit（规划）或 STRUMPACK | 块分裂策略最优 |
| 时间步进（同一矩阵多 RHS）| 任意直接法（分解一次，多次求解）| linger 需实现直接法以利用此优势 |
| WASM / 浏览器端 | linger（唯一选项）| SuperLU/STRUMPACK 不支持 WASM |

### 6.3 椭圆 PDE 的低秩结构（STRUMPACK 的核心优势来源）

对于 FEA 中最常见的椭圆型 PDE（线弹性、热传导、泊松方程），多波前分解中每个波前的非对角块呈现**数值低秩**特性（奇异值快速衰减）。这一性质是 STRUMPACK HSS/BLR 压缩有效的理论基础：

```
波前非对角块奇异值衰减速度 ~ O(e^{-σk}) for k-th singular value
→ ε = 1e-4 时，有效秩通常仅为原块大小的 5-15%
→ 相比精确 LU，BLR/HSS 可减少 2-3× 内存，加速 1.5-2× 分解
```

linger 当前基于 AMG 的方法通过完全不同的机制解决同一问题（多层粗化代替低秩压缩），在大规模椭圆问题上实现 O(n) 或 O(n log n) 的计算复杂度。

---

## 七、Rust 生态集成现状

| 集成路径 | 可用性 | 覆盖范围 | 成熟度 |
|---------|--------|---------|--------|
| `superlu-sys 0.4.2` | crates.io 已发布 | 串行 SuperLU，仅 f64 | 活跃维护，2024 年更新 |
| `sprs-superlu 0.1.7` | crates.io 已发布 | 在 superlu-sys 上的高层封装 | 有效但覆盖有限 |
| SuperLU_MT Rust 绑定 | 不存在 | — | — |
| SuperLU_DIST Rust 绑定 | 不存在 | — | — |
| STRUMPACK Rust 绑定 | 不存在 | — | — |
| **linger 内部直接法** | ✓（Sprint 13–15 已实现，纯 Rust）| SparseLu / SparseCholesky / MultifrontalLu | 开发中 |

### `superlu-sys` 构建方式

```toml
# Cargo.toml
[dependencies]
superlu-sys = "0.4"     # 包含内置 SuperLU C 源码，自动编译为静态库
openblas-src = "0.10"   # 自动 link BLAS
```

`build.rs` 通过 `cmake` crate 将捆绑的 SuperLU C 源码编译为 `libsuperlu.a`，无需系统预安装。

---

## 八、开发规划

### 8.1 总体策略

linger 的直接求解器开发分三个层次：

```
层次 A：纯 Rust 实现（核心层，优先）                         【已完成，Sprint 13–15】
  └── 中小规模精确直接法，零外部依赖，WASM 兼容
      SparseLu（S13）、SparseCholesky（S14）、MultifrontalLu（S15）
      RCM（S13）、COLAMD（S13）、DirectSolverPrecond（S13）

层次 B：SuperLU 串行 FFI（外部后端，可选/降优先级）
  └── 利用 superlu-sys，大中规模 f64 单节点精确直接法
      注：Sprint 14 已由纯 Rust MultifrontalLu 替代，此层次为可选补充

层次 C：STRUMPACK FFI（高阶后端，可选/降优先级）
  └── 大规模 BLR/HSS 近似分解作为预条件器，MPI 分布式
      注：Sprint 15 已实现 MultifrontalLu + BLR 参数接口（纯 Rust），
          STRUMPACK FFI 为可选高阶扩展
```

> **关键决策（Sprint 14/15）**：原计划 Sprint 14 接入 SuperLU FFI、Sprint 15 接入 STRUMPACK FFI，最终改为纯 Rust 实现，原因是 WASM 兼容性需求与泛型设计（`T: Scalar`）。层次 B/C 降为可选规划项。

### 8.2 新增模块结构

```
linger/
├── src/
│   └── direct/                     # 新增直接求解器模块
│       ├── mod.rs                   # DirectSolver trait + 重导出
│       ├── lu.rs                    # 纯 Rust 稀疏 LU（列部分主元）
│       ├── cholesky.rs              # 纯 Rust 稀疏 Cholesky（SPD）
│       ├── ordering/                # 重排序策略
│       │   ├── mod.rs               # OrderingMethod enum
│       │   ├── rcm.rs               # 反 Cuthill-McKee（纯 Rust）
│       │   └── colamd.rs            # COLAMD（纯 Rust 移植）
│       └── triangular.rs            # 稀疏三角求解（前/回代）
└── ffi/
    ├── superlu/                     # 层次 B：SuperLU FFI
    │   ├── Cargo.toml               # superlu-sys 依赖
    │   └── src/lib.rs               # DirectSolver impl via superlu-sys
    └── strumpack/                   # 层次 C：STRUMPACK FFI
        ├── Cargo.toml               # bindgen + C 头文件
        └── src/lib.rs               # DirectSolver + Preconditioner impl
```

### 8.3 核心 Trait 设计

```rust
/// 稀疏直接求解器抽象（对标 KrylovSolver 的地位）
pub trait DirectSolver<T: Scalar>: Send + Sync {
    /// 分析阶段：重排序 + 符号分解（输入矩阵结构）
    fn analyze(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError>;

    /// 数值分解阶段：计算 L、U（或 L、Lᵀ）因子
    fn factorize(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError>;

    /// 求解阶段：利用已有因子求解 Ax = b
    fn solve(&self, b: &DenseVec<T>, x: &mut DenseVec<T>) -> Result<(), SolverError>;

    /// 多右端项批量求解（默认：循环调用 solve）
    fn solve_multi(&self, b: &[DenseVec<T>], x: &mut [DenseVec<T>]) -> Result<(), SolverError> {
        for (bi, xi) in b.iter().zip(x.iter_mut()) {
            self.solve(bi, xi)?;
        }
        Ok(())
    }

    /// 释放因子内存（可重新调用 factorize 更新数值）
    fn reset_factors(&mut self);
}

/// 允许直接法作为预条件器使用
impl<T: Scalar, S: DirectSolver<T>> Preconditioner for DirectSolverPrecond<T, S> {
    // apply = solve（一次因子求解代替预条件应用）
}

/// 配置选项
pub struct DirectSolverOptions {
    pub ordering: OrderingMethod,
    pub pivot_threshold: f64,        // 0.0 = 对角主元, 1.0 = 完全部分主元
    pub reuse_symbolic: bool,        // 结构不变时跳过重排序
    pub refine_steps: usize,         // 迭代精化步数（0 = 不精化）
}

pub enum OrderingMethod {
    Natural,
    Rcm,               // 反 Cuthill-McKee（纯 Rust 实现）
    Colamd,            // 列近似最小度（纯 Rust 实现）
    Metis,             // 需外部 metis-sys（feature-gated）
}
```

### 8.4 Sprint 规划

#### Sprint 13 — 纯 Rust 直接求解器基础层（层次 A）【已完成】

**目标**：提供零依赖的中小规模精确直接法，WASM 兼容

| 任务 | 文件 | 状态 | 说明 |
|------|------|------|------|
| `DirectSolver` trait | `direct/mod.rs` | ✓ | 三阶段接口：analyze/factorize/solve |
| 稀疏三角求解 | `direct/triangular.rs` | ✓ | CSR 格式前/回代 |
| `SparseLu<T>`（稠密右视 LU）| `direct/lu.rs` | ✓ | 列部分主元，O(n²) 内存，n ≤ 3000 实用上限 |
| `SparseCholesky<T>`（初始稠密版）| `direct/cholesky.rs` | ✓ | SPD；Sprint 14 替换为稀疏版 |
| RCM 重排序 | `direct/ordering/rcm.rs` | ✓ | BFS 层序 + 带宽优化排列 |
| COLAMD 重排序 | `direct/ordering/colamd.rs` | ✓ | 纯 Rust 实现 |
| `DirectOptions` | `direct/mod.rs` | ✓ | ordering / pivot_threshold / reuse_symbolic / refine_steps |
| `DirectSolverPrecond` 包装 | `direct/mod.rs` | ✓ | 实现 `Preconditioner` trait |
| 测试套件 | `tests/test_direct.rs` | ✓ | 17 个集成测试 |

**WASM 兼容性**：纯 Rust 实现，无 FFI，无系统线程，完全兼容 `wasm32-unknown-unknown`。

---

#### Sprint 14 — 稀疏 Cholesky + 符号分析（纯 Rust）【已完成】

> **注**：原计划为 SuperLU 串行 FFI（层次 B）。为保持 WASM 兼容性和泛型支持，改为纯 Rust 实现。

**目标**：提升 Cholesky 至真正稀疏实现，建立消元树基础设施

| 任务 | 文件 | 状态 | 说明 |
|------|------|------|------|
| `elimination_tree()` | `direct/etree.rs` | ✓ | Liu 1986 路径压缩算法 |
| `post_order()` 遍历 | `direct/etree.rs` | ✓ | 消元树后序排列 |
| `col_counts()` | `direct/etree.rs` | ✓ | 每列的 L 因子非零数估计 |
| `symbolic_lu()` | `direct/symbolic.rs` | ✓ | Gilbert-Peierls reach-set DFS |
| `SparseCholesky<T>` 重写 | `direct/cholesky.rs` | ✓ | 稀疏左视 Cholesky，O(nnz(L)) 内存 |
| 单元测试 | `direct::etree::tests` | ✓ | 3 个单元测试 |

---

#### Sprint 15 — MultifrontalLu + BLR 参数接口（纯 Rust）【已完成】

> **注**：原计划为 STRUMPACK FFI（层次 C）。为保持 WASM 兼容性和泛型支持，改为纯 Rust 实现。

**目标**：实现消元树驱动的多波前 LU 分解，提供 BLR 参数接口

| 任务 | 文件 | 状态 | 说明 |
|------|------|------|------|
| `MultifrontalLu<T>` | `direct/multifrontal.rs` | ✓ | 消元树后序驱动，波前矩阵组装 + 消元 |
| `MultifrontalOptions` | `direct/multifrontal.rs` | ✓ | `blr_min_size`、`blr_tol` 参数接口 |
| `MultifrontalLu::with_blr()` | `direct/multifrontal.rs` | ✓ | 便捷构造器 |
| `DirectSolver` impl | 同上 | ✓ | 实现 analyze/factorize/solve |
| `DirectSolverPrecond` 集成 | 同上 | ✓ | GMRES/CG 精确分解 1-3 次迭代收敛 |
| 集成测试 | `tests/test_direct_s14_s15.rs` | ✓ | 13 个集成测试 |

> 总测试数：196（全部通过）。BLR 参数接口已就绪；当前标量波前为精确计算，BLR 低秩压缩留待超节点聚合后实现。

---

### 8.5 里程碑总览

```
Sprint 13 ── 纯 Rust 稀疏 LU + Cholesky + RCM/COLAMD    [层次 A，零依赖] ✓ 已完成
Sprint 14 ── 稀疏 Cholesky（稀疏左视）+ 消元树基础设施    [纯 Rust，非 SuperLU FFI] ✓ 已完成
Sprint 15 ── MultifrontalLu + BLR 参数接口               [纯 Rust，非 STRUMPACK FFI] ✓ 已完成
Sprint 16 ── 集成测试 + 性能基准 + 文档完善               [整合]
（可选）── SuperLU FFI（feature = "superlu"）            [层次 B，降优先级]
（可选）── STRUMPACK FFI（feature = "strumpack"）         [层次 C，降优先级]
```

### 8.6 Feature 标志规划（更新后的 Cargo.toml）

```toml
[features]
default      = ["rayon"]
rayon        = ["dep:rayon"]
mpi          = ["dep:rsmpi"]
superlu      = ["dep:superlu-sys", "dep:openblas-src"]   # Sprint 14
strumpack    = []          # Sprint 15（通过 build.rs 链接系统 STRUMPACK）
mumps        = []          # 保留规划项
mkl          = ["dep:intel-mkl-src"]
wasm         = ["dep:wasm-bindgen", "dep:console_error_panic_hook"]

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
nalgebra        = { version = "0.33" }
nalgebra-sparse = "0.4"
superlu-sys     = { version = "0.4", optional = true }
openblas-src    = { version = "0.10", optional = true, features = ["static"] }
```

---

## 九、linger 相对于 SuperLU / STRUMPACK 的差异化优势

尽管 linger v0.1.0 当前不具备精确直接求解能力，其以下特点是 SuperLU 和 STRUMPACK 均无法提供的：

| 优势 | 说明 | SuperLU | STRUMPACK |
|------|------|---------|-----------|
| **WASM 运行时** | 浏览器端求解，无需 native 环境 | ✗ | ✗ |
| **纯 Rust，零 FFI** | 核心层无 unsafe，内存安全 | C/Fortran | C++ |
| **Rayon 线程安全** | 无全局可变状态，可嵌入多线程 Rust 应用 | 部分 | 部分 |
| **AMG + LOBPCG** | FEA 模态分析最优路径，内置 | ✗ | ✗ |
| **特征值框架** | 完整幂法/Krylov/LOBPCG/QEP/NEP | ✗ | ✗ |
| **nalgebra 集成** | 零成本直接使用 nalgebra 矩阵 | ✗ | ✗ |
| **泛型精度** | f32/f64/Complex 单套代码 | 需分别编译 | 需分别编译 |
| **Builder API** | Rust 惯用风格，类型安全配置 | C 函数 | C API |

---

## 十、总结

### 当前能力状态（v0.2.0，Sprint 15 完成后）

| 能力域 | 状态 | 说明 |
|--------|------|------|
| 精确稀疏 LU | ✓ 已完成（S13/S15）| SparseLu（列部分主元）、MultifrontalLu（多波前）|
| 精确稀疏 Cholesky | ✓ 已完成（S14）| SparseCholesky（稀疏左视，O(nnz(L)) 内存）|
| 列重排序（COLAMD/RCM）| ✓ 已完成（S13）| 纯 Rust，ordering::colamd / ordering::rcm |
| 主元选取（部分主元）| ✓ 已完成（S13）| DirectOptions.pivot_threshold |
| 消元树 + 符号分析 | ✓ 已完成（S14）| etree.rs / symbolic.rs |
| BLR 近似接口 | 部分已实现（S15）| MultifrontalLu 参数接口；标量波前精确计算 |
| 嵌套剖分（METIS）| P2（可选）| 需 feature-gated 外部依赖 |
| SuperLU FFI（f64）| P2/可选 | 层次 B；已有 superlu-sys，优先级降低 |
| 低秩近似分解（BLR/HSS）| P2/可选 | 层次 C；STRUMPACK FFI 为可选高阶扩展 |
| 分布式直接法（MPI）| P3 | 独立规划，与直接法层次无关 |
| GPU 加速直接法 | P3 | 独立规划 |

### 下一步行动（Sprint 16）

1. **集成测试完善**：大规模基准（10^4–10^5 DOF），对比 SparseLu / SparseCholesky / MultifrontalLu 速度与内存
2. **性能基准文档**：建立 `benches/` 目录，记录各求解器在标准 FEA 问题上的分解时间
3. **METIS 集成（可选）**：feature-gated `metis-sys`，填补大规模 FEA 重排序空缺
4. **超节点聚合（可选）**：启用 MultifrontalLu 的 BLR 低秩压缩（当前参数接口已就绪）

---

## 参考资料

1. Demmel, Eisenstat, Gilbert, Li, Liu. *A Supernodal Approach to Sparse Partial Pivoting.* SIMAX 1997
2. Gilbert, Peierls. *Sparse Partial Pivoting in Time Proportional to Arithmetic Operations.* SIAM J. Sci. Comput. 1988
3. Ghysels et al. *A Robust Parallel Preconditioner for Indefinite Systems Using Hierarchical Matrices.* SC 2016
4. Ghysels, Vermeersch. *Efficient Sparse LU Factorization with Partial Pivoting on a Shared-Memory Multiprocessor.* SIAM J. Sci. Comput. 2014
5. SuperLU Homepage: https://portal.nersc.gov/project/sparse/superlu/
6. STRUMPACK GitHub: https://github.com/pghysels/STRUMPACK
7. `superlu-sys` crate: https://docs.rs/superlu-sys/0.4.2/superlu_sys/
8. GPU-Accelerated BLR Sparse Direct Solver — IJHPCA 2025
9. Trottenberg, Oosterlee & Schüller. *Multigrid.* 2001（AMG 背景参考）
