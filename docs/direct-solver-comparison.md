# 直接求解器能力对比与开发规划：linger vs SuperLU vs STRUMPACK

> 文档版本：2026-04-07  
> linger 版本：v0.1.0（Sprint 12 完成后）  
> 参照：SuperLU 7.0.1 / SuperLU_DIST 9.0.0 / STRUMPACK 7.2.0

---

## 一、现状概述

### 1.1 三方定位

| 维度 | SuperLU 系列 | STRUMPACK | linger (v0.1.0) |
|------|-------------|-----------|-----------------|
| **定位** | 通用稀疏直接求解器 | 结构化矩阵稀疏直接求解器 + 预条件器 | FEA 稀疏线性系统迭代求解库 |
| **核心算法** | 超节点 LU 分解 | 多波前 LU + HSS/BLR 低秩压缩 | Krylov 迭代法 + AMG 预条件 |
| **直接求解能力** | ✓ 完整 | ✓ 完整（含近似模式） | ✗ 无（仅不完全分解预条件器） |
| **核心语言** | C / Fortran | C++ | Rust |
| **成熟度** | 极成熟（30+ 年）| 成熟（10+ 年）| v0.1.0，开发中 |
| **许可证** | BSD 3-Clause | BSD 3-Clause | —（自有）|

### 1.2 linger 当前的"类直接"能力

linger 目前已实现的不完全分解预条件器在算法上与直接法密切相关：

| 已实现组件 | 文件 | 与直接法的关系 |
|-----------|------|--------------|
| ILU(0) | `precond/ilu0.rs` | 零填充不完全 LU，直接法的极端稀疏化近似 |
| ILU(k) | `precond/iluk.rs` | k 级填充 ILU，k→∞ 时退化为精确 LU |
| ILUT(τ,p) | `precond/ilut.rs` | 阈值 ILU，双参数控制填充 |
| ICC(0) | `precond/icc.rs` | 不完全 Cholesky，SPD 问题的不完全直接法 |

**缺失**：精确 LU 分解（含主元选取）、精确 Cholesky、直接求解器 trait、外部直接法 FFI 绑定。

---

## 二、算法能力对比

### 2.1 分解算法

| 算法 | SuperLU | SuperLU_MT | SuperLU_DIST | STRUMPACK | linger |
|------|---------|------------|--------------|-----------|--------|
| **精确 LU（超节点）** | ✓ | ✓ | ✓ | ✗（多波前）| ✗ |
| **精确 LU（多波前）** | ✗ | ✗ | ✗ | ✓（默认）| ✗ |
| **精确 Cholesky** | ✗ | ✗ | ✗ | ✗ | ✗ |
| **BLR 近似多波前** | ✗ | ✗ | ✗ | ✓ | ✗ |
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
| **直接精确求解** | ✓ | ✓（`--sp_compression none`）| ✗ |
| **近似直接求解** | ✗ | ✓（BLR/HSS 单次前/回代）| ✗ |
| **作为预条件器** | ✗（本身无迭代框架）| ✓（GMRES/BiCGStab + 近似分解）| ✓（ILU 系列 + AMG）|
| **多右端项批求解** | ✓（DIST v9+ 批处理）| ✓ | 循环调用 Krylov |
| **迭代精化** | ✓（内置）| ✓（`--sp_Krylov_solver refine`）| 用户自行实现 |
| **矩阵自由求解** | ✗ | ✗ | ✓（Krylov 本质特性）|

### 2.3 主元选取与数值稳定性策略

| 策略 | SuperLU 系列 | STRUMPACK | linger |
|------|-------------|-----------|--------|
| **完全主元选取** | ✗ | ✗ | ✗ |
| **列部分主元** | ✓（serial/MT）| ✗ | ✗ |
| **阈值主元（DiagPivotThresh）** | ✓ | — | ✗ |
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
| **自然序** | ✓ | ✓ | ✓ | ✓（唯一选项）|
| **COLAMD** | ✓（主要）| ✓ | ✗ | ✗ |
| **MMD_ATA** | ✓ | ✓ | ✗ | ✗ |
| **MMD_AT+A** | ✓ | ✓ | ✗ | ✗ |
| **METIS（嵌套剖分）** | ✗ | ✓ | ✓ | ✗ |
| **ParMETIS** | ✗ | ✓ | ✓ | ✗ |
| **SCOTCH** | ✗ | ✓（PT-Scotch）| ✓（串行 + PT-Scotch）| ✗ |
| **RCM（反 Cuthill-McKee）** | ✗ | ✗ | ✓ | ✗ |
| **几何排序（规则网格）** | ✗ | ✗ | ✓（GEOMETRIC）| ✗ |

> **对 FEA 的影响**：嵌套剖分（METIS）对非结构化 FEA 网格的填充减少效果远优于 COLAMD（3-10×），是大规模直接求解的关键。linger 当前无重排序能力，这也是为何 ILU 系列只能作预条件而非直接求解。

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
| **linger 内部直接法** | 不存在（本文档规划目标）| — | — |

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
层次 A：纯 Rust 实现（核心层，优先）
  └── 中小规模精确直接法，零外部依赖，WASM 兼容

层次 B：SuperLU 串行 FFI（外部后端，中期）
  └── 利用 superlu-sys，大中规模 f64 单节点精确直接法

层次 C：STRUMPACK FFI（高阶后端，长期）
  └── 大规模 BLR/HSS 近似分解作为预条件器，MPI 分布式
```

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

#### Sprint 13 — 纯 Rust 直接求解器基础层（层次 A）

**目标**：提供零依赖的中小规模精确直接法，WASM 兼容

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| `DirectSolver` trait | `direct/mod.rs` | S | 三阶段接口：analyze/factorize/solve |
| 稀疏三角求解 | `direct/triangular.rs` | S | CSR 格式前/回代（已在 ILU apply 中有雏形）|
| 稀疏 LU（列部分主元）| `direct/lu.rs` | L | Gilbert-Peierls 算法；CSR→CSC 内部转换 |
| 稀疏 Cholesky | `direct/cholesky.rs` | M | 仅 SPD；符号阶段 + 数值 left-looking |
| RCM 重排序 | `direct/ordering/rcm.rs` | M | BFS 层序 + 带宽优化排列；CSR 输入 |
| COLAMD 重排序 | `direct/ordering/colamd.rs` | L | 移植或绑定（参考 SuiteSparse COLAMD 算法文献）|
| `DirectSolverPrecond` 包装 | `direct/mod.rs` | S | 实现 `Preconditioner` trait |
| 测试套件 | `tests/test_direct.rs` | M | Poisson 1D/2D MMS；对比 CG 收敛；残差验证 |

**关键算法参考**：  
Gilbert-Peierls 稀疏 LU（1988 ACM TOMS）：按列进行深度优先搜索确定非零结构，精确 O(nnz(L)+nnz(U)) 符号分解，避免超节点假设。

**WASM 兼容性**：纯 Rust 实现，无 FFI，无系统线程，完全兼容 `wasm32-unknown-unknown`。

**性能目标（Sprint 13）**：

| 场景 | 矩阵规模 | 目标 |
|------|---------|------|
| 稀疏 LU 分解（f64）| 10^3 DOF，Poisson | < 10ms |
| 稀疏 LU 分解（f64）| 10^4 DOF，Poisson | < 500ms |
| LU 三角求解（f64）| 10^4 DOF | < 5ms（分解后）|
| 稀疏 Cholesky（SPD）| 10^4 DOF | < 200ms |
| 作为预条件（vs ILU(0)）| 10^4 DOF，GMRES | 迭代次数减少 > 50% |

---

#### Sprint 14 — SuperLU 串行 FFI（层次 B）

**目标**：通过 `superlu-sys` 接入 SuperLU 串行版，覆盖更大规模和完整稳定性策略

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| Feature flag 定义 | `Cargo.toml` | S | `features = ["superlu"]` |
| `SuperLuSolver<T>` struct | `ffi/superlu/src/lib.rs` | M | 封装 `superlu-sys` API |
| `DirectSolver` impl | 同上 | M | analyze → `sp_preorder`；factorize → `dgstrf`；solve → `dgstrs` |
| f32/Complex 支持 | 同上 | M | `sgssv`、`cgssv`、`zgssv` 路径 |
| `OrderingMethod::Metis` | 可选 | S | SuperLU 内置 MMD/COLAMD |
| 构建脚本验证 | `ffi/superlu/build.rs` | S | 验证系统 OpenBLAS 或 MKL |
| 测试 | `tests/test_superlu_ffi.rs` | M | 对比 Sprint 13 纯 Rust LU 的解；大矩阵测试 |
| 示例 | `examples/superlu_fea.rs` | S | 2D Poisson，10^5 DOF，直接 vs CG+AMG 对比 |

**API 目标**：

```rust
// feature = "superlu"
use linger::ffi::superlu::SuperLuSolver;

let mut solver = SuperLuSolver::<f64>::new(SuperLuOptions {
    ordering: OrderingMethod::Colamd,
    pivot_threshold: 1.0,
    ..Default::default()
});

solver.analyze(&a)?;
solver.factorize(&a)?;
solver.solve(&b, &mut x)?;

// 同一分解，多个右端项
solver.solve(&b2, &mut x2)?;
```

**性能目标（Sprint 14）**：

| 场景 | 矩阵规模 | 目标 |
|------|---------|------|
| 稀疏 LU 分解（via SuperLU）| 10^5 DOF，Poisson | < 10s |
| 三角求解（已有分解）| 10^5 DOF | < 100ms |
| 多 RHS 摊销（100 个 RHS）| 10^5 DOF | 分解 < 10s + 每 RHS < 100ms |

---

#### Sprint 15 — STRUMPACK FFI（层次 C）

**目标**：通过 STRUMPACK C API 提供 BLR/HSS 近似分解预条件器，填补大规模 FEA 的能力空缺

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| `bindgen` 生成 C 绑定 | `ffi/strumpack/build.rs` | M | 绑定 `StrumpackSparseSolver.h` |
| Feature flag | `Cargo.toml` | S | `features = ["strumpack"]` |
| `StrumpackSolver<T>` | `ffi/strumpack/src/lib.rs` | L | 封装 init/set_matrix/reorder/factor/solve/destroy |
| `DirectSolver` impl | 同上 | M | 精确模式（compression=none）|
| `Preconditioner` impl | 同上 | L | BLR/HSS 近似分解 + 外迭代器接口 |
| 压缩参数暴露 | `StrumpackOptions` | M | compression type、tolerances、max_rank |
| METIS 集成 | CMake 辅助脚本 | M | 必须依赖，build.rs 中链接 |
| MPI 可选路径 | `ffi/strumpack/src/mpi.rs` | L | `cfg(feature = "mpi")` 下的分布式接口 |
| 测试 | `tests/test_strumpack_ffi.rs` | M | BLR 精度验证；大矩阵内存测试 |

**API 目标**：

```rust
// feature = "strumpack"
use linger::ffi::strumpack::{StrumpackSolver, CompressionType, StrumpackOptions};

// 模式一：精确直接求解
let mut solver = StrumpackSolver::<f64>::new(StrumpackOptions {
    compression: CompressionType::None,
    ordering: OrderingMethod::Metis,
    ..Default::default()
});
solver.analyze(&a)?;
solver.factorize(&a)?;
solver.solve(&b, &mut x)?;

// 模式二：BLR 近似分解作为 GMRES 预条件器
let mut strumpack_precond = StrumpackSolver::<f64>::new(StrumpackOptions {
    compression: CompressionType::Blr { rel_tol: 1e-4, abs_tol: 1e-10, max_rank: 512 },
    ..Default::default()
});
strumpack_precond.analyze(&a)?;
strumpack_precond.factorize(&a)?;  // 近似分解，速度快

// 外迭代：linger 自身的 GMRES 驱动
let result = Gmres::new(30).solve(&a, Some(&strumpack_precond), &b, &mut x, &params)?;
```

**性能目标（Sprint 15）**：

| 场景 | 矩阵规模 | 目标 |
|------|---------|------|
| BLR 近似分解（tol=1e-4）| 10^5 DOF，3D Poisson | < 5s，内存 < 2 GB |
| STRUMPACK+BLR+GMRES | 10^6 DOF | 收敛 1e-8，< 120s |
| 精确 LU vs SuperLU | 10^5 DOF | 速度相当（±50%）|

---

### 8.5 里程碑总览

```
Sprint 13 ── 纯 Rust 稀疏 LU + Cholesky + RCM/COLAMD    [层次 A，零依赖]
Sprint 14 ── SuperLU 串行 FFI（feature = "superlu"）      [层次 B，单节点大规模]
Sprint 15 ── STRUMPACK FFI（feature = "strumpack"）       [层次 C，近似分解预条件]
Sprint 16 ── 集成测试 + 性能基准 + 文档完善               [整合]
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

### 当前能力差距（v0.1.0）

| 能力域 | 优先级 | 与 SuperLU 的差距 | 与 STRUMPACK 的差距 |
|--------|--------|-------------------|---------------------|
| 精确稀疏 LU | P0（Sprint 13）| 核心缺失 | 核心缺失 |
| 精确稀疏 Cholesky | P0（Sprint 13）| 缺失 | 缺失 |
| 列重排序（COLAMD/RCM）| P0（Sprint 13）| 缺失 | 缺失 |
| 嵌套剖分（METIS）| P1（Sprint 13/14）| 缺失（DIST 有）| 缺失 |
| 主元选取（部分主元）| P0（Sprint 13）| 缺失 | 缺失 |
| SuperLU FFI（f64）| P1（Sprint 14）| `superlu-sys` 可用 | N/A |
| 低秩近似分解（BLR/HSS）| P2（Sprint 15）| 无对应 | STRUMPACK 独有能力 |
| 分布式直接法（MPI）| P3（Sprint 15+）| SuperLU_DIST | STRUMPACK+MPI |
| GPU 加速直接法 | P3（Sprint 15+）| SuperLU_DIST v9 | STRUMPACK GPU |

### 推荐的短期行动（Sprint 13 优先项）

1. **实现 `DirectSolver` trait**：建立三阶段接口（analyze/factorize/solve），作为整个直接法体系的基础
2. **Gilbert-Peierls 稀疏 LU**：参考 SuiteSparse/CSparse 的开源实现，移植到纯 Rust
3. **RCM 重排序**：BFS 实现，减少带宽，对直接法内存有 2-5× 改善（正则网格问题）
4. **`DirectSolverPrecond`**：使精确 LU 可直接作为任何 Krylov 求解器的预条件器，统一接口

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
