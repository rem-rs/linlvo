# 设计与开发文档：linger

**适用对象**：AI Agent 驱动的自动化开发流程
**文档版本**：v1.0.0
**crate 版本**：v0.2.0
**日期**：2026-04-08

---

## 0. 立即执行检查清单

发布前和大改动合并前，至少完成以下检查：

1. 版本一致性
  - `Cargo.toml` 的 `version`
  - `README.md` 的 Current status
  - 本文档头部 crate 版本
2. 特性路径回归
  - `cargo test --all-targets`
  - `cargo test --no-default-features --lib --tests`
  - `cargo test --no-default-features --features rayon --lib --tests`
  - `cargo test --no-default-features --features __native --lib --tests`
3. wasm 交叉编译
  - `cargo build --target wasm32-unknown-unknown --no-default-features --features wasm --lib`
4. 基准构建可用性
  - `cargo build --benches`
5. 基线清单一致性
  - `scripts/check_benchmark_manifest.sh`
  - 若基准规模有意变更：`scripts/check_benchmark_manifest.sh --write`
6. 轻量性能守护
  - `scripts/check_perf_guard.sh`
  - 若需要调整容差：`PERF_GUARD_TOLERANCE=0.60 scripts/check_perf_guard.sh`
  - 若需要分指标容差：`PERF_GUARD_TOLERANCE_MAP="spmv_1d_n5000_p50_ms=0.45,cg_1d_n1000_p95_ms=0.80" scripts/check_perf_guard.sh`
  - 若需指定对比基线：`PERF_GUARD_BASELINE_PATH=... scripts/check_perf_guard.sh`
  - 若基线机器变化：`scripts/check_perf_guard.sh --write`

---

## 1. 代码库结构

```
linger/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── src/
│   ├── lib.rs                    # crate root，re-exports 公开 API
│   ├── core/
│   │   ├── mod.rs
│   │   ├── scalar.rs             # Scalar trait（f32/f64）+ ComplexScalar trait（Complex<f32/f64>）
│   │   ├── vector.rs             # Vector trait + 默认实现
│   │   ├── operator.rs           # LinearOperator trait + TransposeOperator trait
│   │   ├── preconditioner.rs     # Preconditioner trait
│   │   ├── solver.rs             # KrylovSolver trait + SolverParams + SolverResult
│   │   └── error.rs              # SolverError 枚举
│   ├── sparse/
│   │   ├── mod.rs
│   │   ├── csr.rs                # 自有 CsrMatrix<T>（impl LinearOperator + TransposeOperator）
│   │   ├── csc.rs                # 自有 CscMatrix<T>
│   │   ├── coo.rs                # CooMatrix<T>（装配阶段）
│   │   ├── bsr.rs                # BlockSparseRow<T>
│   │   ├── ops.rs                # SpMV、稀疏 AXPY 等操作
│   │   ├── mmio.rs               # Matrix Market .mtx 文件读取
│   │   └── nalgebra.rs           # 直接为 nalgebra_sparse::CsrMatrix 实现 LinearOperator
│   ├── iterative/
│   │   ├── mod.rs
│   │   ├── cg.rs                 # Conjugate Gradient
│   │   ├── minres.rs             # MINRES
│   │   ├── gmres.rs              # GMRES(m)
│   │   ├── bicgstab.rs           # BiCGSTAB
│   │   ├── fgmres.rs             # Flexible GMRES
│   │   └── lgmres.rs             # LGMRES
│   ├── precond/
│   │   ├── mod.rs
│   │   ├── jacobi.rs             # Jacobi / Block Jacobi
│   │   ├── sor.rs                # SOR / SSOR
│   │   ├── ilu0.rs               # ILU(0)
│   │   ├── iluk.rs               # ILU(k)
│   │   ├── ilut.rs               # ILUT（带阈值）
│   │   ├── icc.rs                # ICC(0) / ICC(k)
│   │   ├── spai.rs               # Sparse Approximate Inverse
│   │   └── composite.rs          # 组合预条件器（加性/乘性）
│   ├── amg/
│   │   ├── mod.rs
│   │   ├── strength.rs           # 强连接图计算
│   │   ├── coarsen_rs.rs         # Ruge-Stüben C/F 分裂
│   │   ├── coarsen_agg.rs        # 聚合粗化（SA-AMG）
│   │   ├── interpolation.rs      # 插值算子构造
│   │   ├── smoother.rs           # 平滑器（Jacobi/GS/Chebyshev）
│   │   ├── cycle.rs              # V/W/F-cycle 实现
│   │   └── setup.rs              # AMG 层次建立（setup phase）
│   ├── eigen/
│   │   ├── mod.rs                # EigenSolver/Params/Result/Which + matfree_gmres + 辅助函数
│   │   ├── power.rs              # PowerIter（S7）
│   │   ├── subspace.rs           # SubspaceIter（S7）
│   │   ├── inverse.rs            # InverseIter、RayleighQuotientIter（S7）
│   │   ├── lanczos.rs            # LanczosIter IRLM（S8）
│   │   ├── arnoldi.rs            # ArnoldiIter IRAM（S8）
│   │   ├── generalized.rs        # GeneralizedEigen、ShiftInvertLanczos（S9）
│   │   ├── krylov_schur.rs       # KrylovSchur（S10）
│   │   ├── lobpcg.rs             # Lobpcg（S10）
│   │   ├── svd.rs                # LanczosSvd、SvdResult（S11）
│   │   ├── qep.rs                # QuadraticEigen（S12）
│   │   └── nep.rs                # NonlinearOperator trait、NepNewton（S12）
│   ├── direct/
│   │   ├── mod.rs                # DirectSolver trait + DirectOptions + DirectSolverPrecond
│   │   │                         # 新增导出：SymbolicCholesky, symbolic_cholesky
│   │   ├── etree.rs              # 消去树（elimination_tree）+ post_order + col_counts（已修正，S14/B2）
│   │   ├── symbolic.rs           # 符号 LU（symbolic_lu，S14）+ 符号 Cholesky（symbolic_cholesky，B2）
│   │   ├── lu.rs                 # SparseLu<T>：稠密右视 LU + 部分主元 + 迭代精化（A2）
│   │   ├── cholesky.rs           # SparseCholesky<T>：左视稀疏 Cholesky + 迭代精化（A2）
│   │   ├── multifrontal.rs       # MultifrontalLu<T>：消去树驱动多前沿 LU + BLR 选项（S15）
│   │   ├── triangular.rs         # forward_solve / backward_solve（CSR 三角求解）
│   │   └── ordering/
│   │       ├── mod.rs            # OrderingMethod + permute_symmetric + invert_perm
│   │       ├── rcm.rs            # Reverse Cuthill-McKee（S13）
│   │       ├── colamd.rs         # Column Approximate Minimum Degree（S13）
│   │       └── nd.rs             # NodeNd 多层嵌套剖分（B1，纯 Rust，无外部依赖）
│   ├── builder.rs                # SolverBuilder fluent API（C2）
│   ├── wasm.rs                   # WASM bindings（C1 扩展：WasmLuSolver 等）
│   ├── parallel/
│   │   ├── mod.rs
│   │   ├── rayon_ops.rs          # rayon 并行 SpMV、向量操作
│   │   └── mpi_stub.rs           # MPI trait 接口（暂 stub）
│   └── parallel_dist/            # 规划：纯 Rust 分布式并行（MPI 抽象 / halo / 分布式 CSR）
│       ├── mod.rs
│       ├── layout.rs             # 全局/局部 DOF 映射与分区元数据
│       ├── halo.rs               # halo 打包/交换接口
│       └── dist_csr.rs           # 分布式 CSR 结构与局部 SpMV
├── tests/
│   ├── common/
│   │   └── mod.rs                # 共用测试辅助（MMS 制造解）
│   ├── test_sparse_ops.rs
│   ├── test_krylov.rs
│   ├── test_precond.rs
│   ├── test_sprint3.rs
│   ├── test_amg.rs
│   ├── test_amg_internals.rs
│   ├── test_parallel.rs
│   ├── test_eigen.rs             # Sprint 7：幂法族 (11 tests)
│   ├── test_eigen_s8_s10.rs      # Sprint 8-10：Lanczos/Arnoldi/KS/LOBPCG (16 tests)
│   ├── test_eigen_s11_s12.rs     # Sprint 11-12：SVD/QEP/NEP/ComplexScalar (9 tests)
│   ├── test_direct.rs            # Sprint 13：SparseLu/SparseCholesky/RCM/COLAMD (17 tests)
│   ├── test_direct_s14_s15.rs   # Sprint 14-15：etree/MultifrontalLu/BLR (13 tests)
│   ├── test_ordering_nd.rs      # B1：NodeNd 嵌套剖分排序 (14 tests)
│   ├── test_builder_c2.rs       # C1+C2：SolverBuilder + WASM API (15 tests)
│   ├── test_refinement.rs       # A2：迭代精化 (7 tests)
│   ├── test_symbolic_cholesky.rs # B2：符号 Cholesky (13 tests)
│   ├── test_mmio.rs             # C3：Matrix Market 读取 (14 tests)
│   └── test_reuse_symbolic.rs   # B3：reuse_symbolic 缓存 (10 tests)
└── benches/
    ├── bench_spmv.rs
    ├── bench_krylov.rs
    ├── bench_amg.rs
    └── bench_direct.rs         # A1：直接法基准测试
```

---

## 2. 各模块详细设计

### 2.1 core/scalar.rs

定义数值泛型边界，所有算法对 `T: Scalar` 泛型。Sprint 11 新增 `ComplexScalar`：

```rust
use num_traits::{Float, NumAssign, Zero, One};

pub trait Scalar: Float + NumAssign + Zero + One + Copy + Debug + Send + Sync + 'static {
    fn machine_epsilon() -> Self;
    fn from_f64(v: f64) -> Self;
}

/// 覆盖复数类型的 trait（Scalar 自动实现 ComplexScalar）
pub trait ComplexScalar: NumAssign + Zero + One + Copy + Debug + Send + Sync + 'static {
    type Real: Scalar;   // 实部类型
    fn from_f64(v: f64) -> Self;
    fn real(self) -> Self::Real;
    fn imag(self) -> Self::Real;
    fn abs(self) -> Self::Real;  // 模
    fn conj(self) -> Self;
    fn sqrt(self) -> Self;
    fn is_finite(self) -> bool;
    fn machine_epsilon() -> Self::Real;
}

// blanket impl：所有 Scalar 自动实现 ComplexScalar（Real = Self）
impl<T: Scalar> ComplexScalar for T { ... }

// 显式 impl：
impl ComplexScalar for Complex<f64> { type Real = f64; ... }
impl ComplexScalar for Complex<f32> { type Real = f32; ... }
```

### 2.2 core/vector.rs

默认实现：`Vec<T>` 包装为 `DenseVec<T>`。当前 solver 路径统一使用 `DenseVec<T>` 作为向量类型。

关键操作：
- `dot`: 使用 rayon 并行归约（当 feature = "rayon"）
- `axpy`: `y += alpha * x`，BLAS level-1
- `norm2`: $\sqrt{\sum x_i^2}$，注意溢出防护

### 2.3 sparse/csr.rs

```rust
pub struct CsrMatrix<T> {
    nrows: usize,
    ncols: usize,
    row_ptr: Vec<usize>,   // 长度 nrows + 1
    col_idx: Vec<usize>,   // 长度 nnz
    values:  Vec<T>,       // 长度 nnz
}

impl<T: Scalar> CsrMatrix<T> {
    pub fn from_coo(coo: &CooMatrix<T>) -> Self { ... }
    pub fn spmv(&self, x: &[T], y: &mut [T]) { ... }        // y = A * x
    pub fn spmv_add(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) { ... } // y = alpha*A*x + beta*y
    pub fn transpose(&self) -> CscMatrix<T> { ... }
    pub fn diag(&self) -> Vec<T> { ... }
    pub fn ilu0(&self) -> IluFactors<T> { ... }               // 就地 ILU(0)
}
```

SpMV 并行策略（rayon）：按行分块，每块独立计算，无数据竞争。

### 2.4 iterative/cg.rs（CG 实现参考）

标准 Preconditioned CG 算法（对标 HYPRE/PETSc PCG）：

```
算法：Preconditioned Conjugate Gradient
输入：A, M（预条件器）, b, x0, tol, max_iter
---
r = b - A*x
z = M^{-1} * r
p = z
rz = <r, z>
for k = 1, 2, ..., max_iter:
    Ap = A * p
    alpha = rz / <p, Ap>
    x = x + alpha * p
    r = r - alpha * Ap
    if ||r||_2 / ||b||_2 < tol: break
    z = M^{-1} * r
    rz_new = <r, z>
    beta = rz_new / rz
    p = z + beta * p
    rz = rz_new
```

实现注意事项：
- 初始残差 $\|b\|_2 = 0$ 时直接返回（零右端项）
- 支持 `x0 = 0` 的快速路径（省略初始 SpMV）
- 每隔 `check_interval`（默认 10）次迭代重计算真实残差防止浮点漂移

### 2.5 iterative/gmres.rs（GMRES 实现参考）

GMRES(m) Arnoldi 过程（对标 HYPRE GMRES、PETSc KSPGMRES）：

```
算法：Restarted GMRES(m) with preconditioning
---
外层循环（重启）：
  r = M^{-1} * (b - A*x)
  beta = ||r||_2
  v[0] = r / beta
  内层循环 j = 0..m-1：
    w = M^{-1} * A * v[j]        （右预条件）
    改进 Gram-Schmidt 正交化：
      for i = 0..j:
        h[i,j] = <w, v[i]>
        w = w - h[i,j] * v[i]
    h[j+1,j] = ||w||_2
    v[j+1] = w / h[j+1,j]
    更新 Givens 旋转，最小化残差
  从 Hessenberg 最小二乘解更新 x
```

实现要点：
- Hessenberg 矩阵用 `Vec<Vec<T>>` 动态分配
- Givens 旋转在线更新，避免存储完整 QR
- 重启后保留解向量 x 累积更新

### 2.6 precond/ilut.rs（ILUT，对标 HYPRE Euclid）

ILUT(tau, p) 算法：
- `tau`：丢弃容忍度（相对于行范数）
- `p`：每行保留的最大填充数

```
for i = 0..n:
    取第 i 行到工作数组 w
    for k < i where w[k] != 0:
        w[k] /= L[k,k]（pivot）
        for j > k: w[j] -= w[k] * U[k,j]
        按阈值 tau * ||row||_2 丢弃 w[k]
    分割 w 为 L 部分（j < i）和 U 部分（j >= i）
    各自按幅值保留前 p 个元素
    存入 L[i,:] 和 U[i,:]
```

### 2.7 amg/ 模块设计

AMG 分三个阶段：

**Setup Phase**（`amg/setup.rs`）：
1. 计算强连接矩阵 S（`strength.rs`）：$|a_{ij}| \geq \theta \cdot \max_{k \neq i} |a_{ik}|$
2. 粗化（`coarsen_rs.rs` 或 `coarsen_agg.rs`）：确定 C/F 点集合
3. 构造插值算子 P（`interpolation.rs`）
4. 构造粗网格算子 $A_c = P^T A P$（Galerkin 投影）
5. 递归建立多层次层次，直到粗网格足够小（< `coarse_threshold`）

**Solve Phase**（`amg/cycle.rs`）：
- V-cycle：pre-smooth → recurse → post-smooth
- W-cycle：两次递归调用
- 支持作为独立求解器或作为 Krylov 预条件器

**平滑器**（`amg/smoother.rs`）：
- Weighted Jacobi（$\omega = 2/3$ 默认）
- Gauss-Seidel（红黑排序）
- Chebyshev 多项式平滑

**SA-AMG 聚合算法**（`amg/coarsen_agg.rs`）：
1. 构造聚合：贪心算法，未聚合节点找其强连接邻居聚合
2. 构造试探向量（tentative prolongation）
3. 通过平滑操作改善插值（smoothed aggregation）

---

### 2.8 eigen/ 模块设计（Sprint 7–12）

特征值子系统建立在统一的 trait 体系上：

```rust
pub trait EigenSolver<T: Scalar> {
    fn solve<Op: LinearOperator<Vector = DenseVec<T>>>(
        &self, op: &Op, params: &EigenParams<T>,
    ) -> Result<EigenResult<T>, SolverError>;
}
```

#### 算法分层

| 层次 | 算法 | Sprint | 适用场景 |
|------|------|--------|---------|
| 基础迭代 | PowerIter、SubspaceIter | S7 | 最大特征值，初始探测 |
| 移位反迭代 | InverseIter、RayleighQuotientIter | S7 | 指定区域单特征值 |
| Krylov 子空间 | LanczosIter (IRLM)、ArnoldiIter (IRAM) | S8 | 多特征值；对称/非对称 |
| 广义特征值 | ShiftInvertLanczos、GeneralizedEigen | S9 | `Ax = λBx` |
| 鲁棒重启 | KrylovSchur、Lobpcg | S10 | 生产级；FEA 模态 |
| SVD | LanczosSvd | S11 | 最大奇异值/向量 |
| QEP | QuadraticEigen | S12 | 阻尼结构动力学 |
| NEP | NepNewton | S12 | 非线性特征值，Newton |

#### core/operator.rs 扩展（S11）

```rust
pub trait TransposeOperator: LinearOperator {
    fn apply_transpose(&self, x: &Self::Vector, y: &mut Self::Vector);
}

// CsrMatrix<T> 实现 TransposeOperator（scatter-based Aᵀ x）
```

#### SVD 设计（eigen/svd.rs）

对 `AᵀA` 运行 `LanczosIter`，σᵢ = √λᵢ，左奇异向量 uᵢ = A vᵢ / σᵢ。
`AtAOperator` 包装器透明组合两次 `apply` + `apply_transpose`。

```rust
// 关键约束：Op 必须同时实现 LinearOperator 和 TransposeOperator
pub fn solve<T, Op>(&self, op: &Op, k: usize, tol: T, max_iter: usize, vecs: bool)
    -> Result<SvdResult<T>, SolverError>
where Op: LinearOperator<Vector = DenseVec<T>> + TransposeOperator
```

#### QEP 线性化（eigen/qep.rs）

伴随型线性化（companion form）：

```
A = [[0, I], [-K, -C]],  B = I

y = A x:
  x = [x₁; x₂]
  y[0..n] = x₂
  y[n..2n] = -Kx₁ - Cx₂
```

`QepCompanion<K, C>` 实现 `LinearOperator`，传给 `ArnoldiIter` 求 2n 维特征值。
结果截取前 n 分量作为物理特征向量。

#### NEP Newton（eigen/nep.rs）

```
每步：
  r = T(λ)x
  δλ = -(xᵀr) / (xᵀT'(λ)x)   — Rayleigh 泛函更新
  求解 T(λ+ε)w = x             — 正则化反迭代（ε = 1e-6·(1+|λ|)）
  x ← w / ‖w‖
  λ ← λ + δλ（带阻尼）
```

`T'(λ)` 默认使用中心有限差分；用户可重写 `apply_dt` 提供精确导数。

---

---

### 2.9 direct/ 模块设计（Sprint 13–15）

直接求解器子系统实现三阶段接口，与 Krylov 路径正交，可通过 `DirectSolverPrecond` 互通：

```rust
pub trait DirectSolver<T: Scalar>: Send + Sync {
    fn analyze(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError>;   // 符号分析
    fn factorize(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError>; // 数值分解
    fn solve(&self, b: &DenseVec<T>, x: &mut DenseVec<T>) -> Result<(), SolverError>;
    fn factor(&mut self, a: &CsrMatrix<T>) -> Result<(), SolverError>;   // analyze + factorize
    fn reset_factors(&mut self);
}
```

#### 求解器家族

| 结构体 | Sprint | 算法 | 内存 | 适用场景 |
|--------|--------|------|------|---------|
| `SparseLu<T>` | S13 | 稠密右视 LU + 行列置换 | O(n²) | 小型矩阵（n ≤ 3000） |
| `SparseCholesky<T>` | S14 | 左视稀疏 Cholesky | O(nnz(L)) | SPD 矩阵，任意规模 |
| `MultifrontalLu<T>` | S15 | 消去树驱动多前沿 LU | O(nnz(L+U)) | 通用，支持 BLR 近似 |

#### 重排序（ordering/）

| 算法 | 适用场景 |
|------|---------|
| `Natural` | 调试、测试用 |
| `Rcm`（默认）| 结构化 FEA 网格，带宽压缩 5-50× |
| `Colamd` | 非结构化矩阵，最小度近似 |

#### 消去树（etree.rs，Sprint 14）

`elimination_tree(a)` 使用 Liu (1986) 路径压缩算法，O(n α(n)) 时间：
- `parent[j]` = 消去树中节点 j 的父节点（n 为根哨兵）
- `post_order(parent)` 后序遍历（子节点先于父节点）
- `col_counts(a, parent)` 预测 L 各列非零数（用于预分配）

消去树是 Sprint 14 稀疏 Cholesky 和 Sprint 15 多前沿框架的共同基础。

#### 符号分解（symbolic.rs，Sprint 14）

`symbolic_lu(a, parent)` 基于 Gilbert-Peierls reach-set 算法：
- 对每列 j，从 A 的种子节点出发在消去树上做 DFS
- 到达集 = L 列 j 和 U 行 j 的精确非零模式
- 总工作量 O(nnz(L) + nnz(U))

#### 左视稀疏 Cholesky（cholesky.rs，Sprint 14）

```
for j = 0..n:
    x[j:n] ← A[j:n, j]                     # 从 A 散布
    reach ← DFS-etree({A 行 j 的下三角元素})  # O(|reach|)
    for k in reach:
        x[j] -= L[j,k]²
        x[i] -= L[i,k]·L[j,k] for i>j in col k of L
    L[j,j] = √x[j];  L[i,j] = x[i]/L[j,j]
```

内存 O(nnz(L))，相比 S13 稠密版 O(n²) 大幅降低。

#### 多前沿 LU（multifrontal.rs，Sprint 15）

`MultifrontalLu<T>` 按消去树后序逐节点处理：
- 每个节点对应一个"前沿矩阵"
- 将父节点的 Schur 补贡献传递给父前沿
- `MultifrontalOptions { blr_min_size, blr_tol }` 控制 BLR 压缩
  - `blr_min_size = usize::MAX`（默认）：精确分解
  - 有限值：前沿非对角块用截断 SVD 压缩，适合作为预条件器

#### DirectSolverPrecond 包装器

任意 `DirectSolver` 均可包装为 Krylov 预条件器：

```rust
let precond = DirectSolverPrecond::new(MultifrontalLu::<f64>::default(), &a)?;
Gmres::new(30).solve(&a, Some(&precond), &b, &mut x, &params)?;
// 精确分解时通常 1–3 次迭代收敛
```

---

```rust
// src/sparse/nalgebra.rs
use nalgebra_sparse::CsrMatrix as NaCsr;

impl<T: Scalar + nalgebra::RealField> LinearOperator for NaCsr<T> {
    type Vector = DenseVec<T>;

    fn apply(&self, x: &Self::Vector, y: &mut Self::Vector) {
        // 逐行遍历 nalgebra_sparse CSR 结构
        for (i, row) in self.row_iter().enumerate() { ... }
    }
    fn nrows(&self) -> usize { self.nrows() }
    fn ncols(&self) -> usize { self.ncols() }
}
```

---

## 4. 测试策略

### 4.1 制造解（MMS）测试框架

```rust
// tests/common/mod.rs

/// 构造已知解的测试系统：A * x_exact = b
pub fn make_poisson_1d<T: Scalar>(n: usize) -> (CsrMatrix<T>, Vec<T>, Vec<T>) {
    // 1D Poisson：三对角矩阵 [-1, 2, -1]
    // x_exact = sin(pi * i / n)
    // b = A * x_exact
    ...
}

pub fn make_poisson_2d<T: Scalar>(nx: usize, ny: usize) -> (...) { ... }

pub fn make_nonsymmetric_convdiff<T: Scalar>(n: usize, peclet: T) -> (...) { ... }
```

### 4.2 每个 Solver 的验证矩阵

| 求解器 | 测试矩阵 | 验证指标 |
|--------|---------|---------|
| CG | SPD Poisson 1D/2D | 收敛 + 残差 < 1e-10 |
| MINRES | 不定矩阵（鞍点） | 收敛 |
| GMRES | 对流扩散（非对称） | 收敛 + 迭代次数合理 |
| AMG | 各向同性/各向异性 Poisson | 网格无关收敛 |
| ILUT | 大 Peclet 数对流 | 预条件质量 |

### 4.3 回归测试

- 每个 solver 记录 "黄金" 迭代次数，回归时允许 ±10% 浮动
- 性能基准：Criterion.rs，CI 中记录 benchmark 结果，检测 > 20% 退化

---

## 5. AI Agent 开发指引

### 5.1 任务分解原则

每个 Agent 任务应聚焦单一模块，遵循以下粒度：

**可并行的独立任务**：
- `core/` 全部 trait 定义（无外部依赖）
- `sparse/csr.rs` + `sparse/coo.rs`（无 solver 依赖）
- `iterative/cg.rs`（仅依赖 core traits）
- `iterative/gmres.rs`（仅依赖 core traits）
- `precond/jacobi.rs`（仅依赖 sparse/csr）

**有依赖顺序的任务**：
```
core/ → sparse/ → iterative/ + precond/ → amg/ → parallel_dist/
                                        ↘

```

### 5.2 各模块 Agent 任务描述模板

#### 任务：实现 `iterative/cg.rs`

**输入契约**：
- `core/operator.rs` 中的 `LinearOperator` trait 已定义
- `core/preconditioner.rs` 中的 `Preconditioner` trait 已定义
- `core/solver.rs` 中的 `SolverParams`、`SolverResult`、`KrylovSolver` trait 已定义

**输出要求**：
- `pub struct ConjugateGradient<T: Scalar>` 实现 `KrylovSolver` trait
- 支持无预条件（`None`）和有预条件两种路径
- 函数签名：`fn solve(&self, op, precond, b, x, params) -> SolverResult`
- 内部实现：标准 PCG 算法，参见设计文档 §2.4
- 零右端项快速路径

**测试要求**：
- 在 `tests/test_krylov.rs` 中添加：
  1. 10阶 SPD 对角矩阵（精确解已知）
  2. 100 阶 1D Poisson 矩阵，无预条件
  3. 100 阶 1D Poisson 矩阵，Jacobi 预条件
  - 验证：收敛 + `result.final_residual < 1e-10`

---

#### 任务：实现 `amg/` 模块（SA-AMG）

**输入契约**：
- `sparse/csr.rs` 中 `CsrMatrix<T>` 已实现，包括 `spmv`、`transpose`
- `precond/` 中 Jacobi 平滑器已实现

**输出要求**：
- `pub struct SmoothedAggregationAmg<T>` 实现 `Preconditioner` trait
- `AmgBuilder` 提供参数配置：`coarsening_threshold`、`smoother`、`max_levels`、`coarse_solver`
- `setup()` 方法建立多层次层次
- `apply_precond()` 执行一次 V-cycle
- V-cycle 实现参见设计文档 §2.7

**测试要求**：
- 2D Poisson（32×32 网格，约 1024 DOF）：AMG 独立求解，10 次 V-cycle 后残差缩减 > 10^6
- AMG 作为 CG 预条件：迭代次数 < 20（无预条件需要 ~200）
- 层次信息打印：每层 DOF 数、算子复杂度

---

#### 任务：实现 `precond/ilut.rs`

**输入契约**：
- `sparse/csr.rs` 中 `CsrMatrix<T>` 已实现

**输出要求**：
- `pub struct IluT<T>` 实现 `Preconditioner` trait
- 参数：`tau: T`（丢弃阈值），`p: usize`（每行最大填充）
- `from_csr(mat: &CsrMatrix<T>, tau: T, p: usize) -> Result<Self, SolverError>`
- 三角求解：前代（L solve）+ 后代（U solve）
- 算法参见设计文档 §2.6

**测试要求**：
- 100 阶 Poisson 矩阵：ILU(0,0) 等价于 ILU(0)，验证两者结果接近
- 对流扩散矩阵（Peclet=10）：ILUT 作为 GMRES 预条件，与无预条件对比迭代次数

---

### 5.3 错误处理规范

所有公开 API 返回 `Result<_, SolverError>`：

```rust
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    #[error("singular matrix detected at row {row}")]
    SingularMatrix { row: usize },

    #[error("failed to converge after {max_iter} iterations, residual = {residual:.3e}")]
    ConvergenceFailed { max_iter: usize, residual: f64 },

    #[error("dimension mismatch: operator is {op_rows}x{op_cols}, rhs has {rhs_len} entries")]
    DimensionMismatch { op_rows: usize, op_cols: usize, rhs_len: usize },

    #[error("preconditioner setup failed: {reason}")]
    PrecondSetupFailed { reason: String },

    #[error("numerical breakdown: {detail}")]
    NumericalBreakdown { detail: String },
}
```

### 5.4 并行化规范

- 所有循环中含 SpMV、向量 AXPY 的操作，在长度 > `PARALLEL_THRESHOLD`（默认 1024）时启用 rayon
- AMG 的粗化、插值构造中的行操作可并行
- 严禁在预条件器的 `apply` 中使用全局可变状态（确保 `&self` 接口）
- ILU 因下三角/上三角依赖不可直接并行；可用 level-scheduling 或 block Jacobi 替代
- rayon 并行调用须用 `#[cfg(not(target_arch = "wasm32"))]` 条件编译隔离，WASM 下退化为单线程串行路径

### 5.5 WebAssembly（WASM）兼容性规范

**可编译为 WASM 是 linger 的明确目标**，适用于 Web 端轻量仿真、在线 FEA 工具等场景。

**模块可用性矩阵**：

| 模块 | wasm32 支持 | 备注 |
|------|------------|------|
| `core/` | ✅ 完全支持 | 纯数学 trait，无系统依赖 |
| `sparse/` | ✅ 完全支持 | 纯内存操作 |
| `iterative/` | ✅ 完全支持 | 算法层无系统调用 |
| `precond/` | ✅ 完全支持 | 同上 |
| `amg/` | ✅ 完全支持 | setup/cycle 均可在 WASM 运行 |
| `parallel/rayon_ops.rs` | ⚠️ 禁用 | `cfg` 条件编译屏蔽 |
| `direct/` | ✅ 完全支持 | 纯 Rust，零 FFI 依赖，明确 WASM 兼容目标 |

**编码约束**：
- 所有模块禁止直接调用 `std::thread::spawn`，并行路径统一通过 rayon feature 隔离
- 不使用 `std::time::Instant`（在 wasm32 不稳定）；计时功能移到 `cfg(not(target_arch = "wasm32"))` 分支
- `feature = "wasm"` 启用时，通过 `wasm-bindgen` 暴露简化的 JS 友好接口：

```rust
// 示例：通过 wasm-bindgen 暴露的 JS 接口
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn solve_cg_js(
    row_ptr: &[usize], col_idx: &[usize], values: &[f64],
    rhs: &[f64], tol: f64, max_iter: usize,
) -> Vec<f64> { ... }
```

**验证目标**：
- `cargo build --target wasm32-unknown-unknown` 核心 crate（不带 rayon feature）编译通过
- `wasm-pack test --headless --firefox` 运行基本 solver 测试

### 5.6 代码风格约定

- 公开 API 必须有 `///` rustdoc 文档，包含 `# Errors` 和 `# Examples` 节
- 内部辅助函数用 `#[inline]` 修饰热路径
- 避免运行时 `panic!`；使用 `Result` 传播错误
- 数组越界：在 debug 模式用 `assert!`，在 release 模式用 `get_unchecked`（明确注释 unsafe 不变量）
- benchmark 文件中每个 bench group 用 `criterion_group!` 组织

---

## 6. 开发路线图（Sprint 计划）

### Sprint 1（M1）：基础设施 ✅ 已完成
- [x] `core/` 全部 trait 定义（`Scalar`、`Vector`、`LinearOperator`、`Preconditioner`、`KrylovSolver`、`SolverError`）
- [x] `sparse/csr.rs`、`coo.rs`、`csc.rs`（含 `matmat`、`transpose_csr`）
- [x] `sparse/ops.rs`（SpMV、spmv_add、对角提取、三元组迭代）
- [x] `sparse/nalgebra.rs`（直接为 `nalgebra_sparse::CsrMatrix` 实现 `LinearOperator`）
- [x] `core/error.rs`
- [x] 基础测试框架（`tests/common/`，含 MMS Poisson 1D/2D、对流扩散矩阵生成器）

### Sprint 2（M2）：核心求解器 ✅ 已完成
- [x] `precond/jacobi.rs`（Jacobi、Block Jacobi）
- [x] `precond/sor.rs`（SOR、SSOR）
- [x] `precond/ilu0.rs`（ILU(0)）
- [x] `iterative/cg.rs`（PCG，含零右端快速路径）
- [x] `iterative/gmres.rs`（GMRES(m)，Givens 旋转在线更新）
- [x] `iterative/bicgstab.rs`（BiCGSTAB，含近收敛时 rho≈0 保护）
- [x] `iterative/minres.rs`
- [x] 10 项集成测试（`tests/test_krylov.rs`）

### Sprint 3（M3）：高级预条件器 ✅ 已完成
- [x] `precond/iluk.rs`（ILU(k)，BTreeMap 符号相位 + 填充级别传播）
- [x] `precond/ilut.rs`（ILUT(tau,p)，密集工作行 + 双阈值丢弃）
- [x] `precond/icc.rs`（ICC(0)，左视行序 Cholesky）
- [x] `precond/spai.rs`（静态模式 SPAI，逐列 QR 最小二乘）
- [x] `precond/composite.rs`（`AdditivePrecond`、`MultiplicativePrecond`）
- [x] `iterative/fgmres.rs`（存储预条件向量 z，支持变预条件器）
- [x] `iterative/lgmres.rs`（循环增广向量 FIFO，含 Jacobi 预条件测试）
- [x] 21 项集成测试（`tests/test_sprint3.rs`）

### Sprint 4（M4）：代数多重网格 ✅ 已完成
- [x] `amg/strength.rs`（强连接图，theta 阈值）
- [x] `amg/coarsen_rs.rs`（Ruge-Stüben C/F 分裂，lambda 贪心）
- [x] `amg/coarsen_agg.rs`（SA-AMG 贪心聚合 + 试探延拓算子）
- [x] `amg/interpolation.rs`（RS 直接插值 + SA 平滑插值，Gershgorin 谱估计）
- [x] `amg/smoother.rs`（Jacobi、GS 前向/后向各向平滑）
- [x] `amg/cycle.rs`（V-cycle / W-cycle，递归实现）
- [x] `amg/setup.rs`（多层次建立，Galerkin $A_c = R A P$，Option::take 避免移动冲突）
- [x] `AmgPrecond` 实现 `Preconditioner` trait，可插入任意 Krylov 求解器
- [x] 10 项集成测试（`tests/test_amg.rs`，含 SA/RS 层次建立、V/W-cycle、AMG-PCG 1D/2D）

### Sprint 5（M5）：并行 + 性能 + WASM ✅ 已完成
- [x] `parallel/rayon_ops.rs`（`parallel_spmv`、`parallel_spmv_add`、`parallel_axpy`、`parallel_axpby`、`parallel_dot`、`parallel_norm2`；`cfg(feature="rayon")` 门控，不带时退化串行）
- [x] AMG setup 并行化：`amg/strength.rs` 的 `strong_connections`、`amg/interpolation.rs` 的 `rs_interpolation` 和 `smooth_prolongation` 均采用 collect-then-assemble 模式，per-row 计算在 `cfg(feature="rayon")` 下通过 `into_par_iter().map().collect()` 并行执行
- [x] `sparse/bsr.rs`（`BsrMatrix<T>`：块行指针/列索引/值；`BsrBuilder` 支持重复块累加；串行/并行 SpMV；`to_csr` 转换）
- [x] Criterion.rs benchmark 套件（`benches/bench_spmv.rs`、`bench_krylov.rs`、`bench_amg.rs`）
- [x] `src/wasm.rs`（`WasmCsrMatrix`、`WasmCgSolver`、`WasmGmresSolver`；`feature="wasm"` + wasm-bindgen）
- [x] WASM 目标编译验证：`cargo build --target wasm32-unknown-unknown --no-default-features` 和 `--features wasm` 均通过。解决方案：将 `nalgebra` 移入 `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]`，直接 nalgebra 集成代码仅在非 wasm32 目标编译
- [x] 13 项集成测试（`tests/test_parallel.rs`）
- [ ] `wasm-pack test` 浏览器集成测试（需要 Node.js / wasm-pack 环境，超出当前 CI 范围）

**Sprint 5 完成时状态**：73 项测试全部通过，`cargo build --benches` 编译通过，wasm32 双模式编译通过。

### Sprint 6（M6，可选）：纯 Rust 分布式并行基础
- [ ] `parallel_dist/layout.rs`：分区元数据（owned/ghost）
- [ ] `parallel_dist/halo.rs`：邻域通信抽象（先 trait，后接 rsmpi）
- [ ] `parallel_dist/dist_csr.rs`：分布式 CSR + 局部 SpMV + halo 合并
- [ ] 对比基准：单机多线程 vs 分布式（弱扩展/强扩展）

### Sprint 7（M7）：特征值求解器基础 ✅ 已完成
- [x] `eigen/power.rs`（PowerIter）
- [x] `eigen/subspace.rs`（SubspaceIter）
- [x] `eigen/inverse.rs`（InverseIter、RayleighQuotientIter）
- [x] 11 项集成测试（`tests/test_eigen.rs`）

### Sprint 8–10（M8–M10）：Lanczos / Arnoldi / KS / LOBPCG ✅ 已完成
- [x] `eigen/lanczos.rs`（LanczosIter IRLM）
- [x] `eigen/arnoldi.rs`（ArnoldiIter IRAM）
- [x] `eigen/generalized.rs`（GeneralizedEigen、ShiftInvertLanczos）
- [x] `eigen/krylov_schur.rs`（KrylovSchur）
- [x] `eigen/lobpcg.rs`（Lobpcg）
- [x] 16 项集成测试（`tests/test_eigen_s8_s10.rs`）

### Sprint 11–12（M11–M12）：ComplexScalar + SVD + QEP + NEP ✅ 已完成
- [x] `core/scalar.rs`：`ComplexScalar` trait（Complex<f64/f32>）
- [x] `core/operator.rs`：`TransposeOperator` trait
- [x] `eigen/svd.rs`：`LanczosSvd`、`SvdResult`（对 AᵀA 运行 Lanczos）
- [x] `eigen/qep.rs`：`QuadraticEigen`（伴随型线性化 → ArnoldiIter）
- [x] `eigen/nep.rs`：`NonlinearOperator` trait、`NepNewton`（阻尼 Newton）
- [x] 9 项集成测试（`tests/test_eigen_s11_s12.rs`）

### Sprint 13（M13）：稀疏直接求解器基础 ✅ 已完成
- [x] `direct/mod.rs`：`DirectSolver` trait、`DirectOptions`、`DirectSolverPrecond`
- [x] `direct/triangular.rs`：`forward_solve` / `backward_solve`（CSR，unit/non-unit 对角）
- [x] `direct/ordering/rcm.rs`：Reverse Cuthill-McKee，伪周边节点启发式
- [x] `direct/ordering/colamd.rs`：Column AMD，懒删除最小堆
- [x] `direct/lu.rs`：`SparseLu<T>`，稠密右视 LU + 部分选主元 + 行列置换
- [x] `direct/cholesky.rs`（初版）：稠密左视 Cholesky，O(n²)
- [x] 17 项集成测试（`tests/test_direct.rs`）

### Sprint 14（M14）：消去树 + 稀疏 Cholesky ✅ 已完成
- [x] `direct/etree.rs`：`elimination_tree`（Liu 1986 路径压缩）、`post_order`、`col_counts`（mark-and-sweep，Davis 2006，已修正）
- [x] `direct/symbolic.rs`：`symbolic_lu`（Gilbert-Peierls reach-set，DFS on etree）
- [x] `direct/cholesky.rs`：重写为左视稀疏 Cholesky，O(nnz(L)) 内存
- [x] 消去树单元测试（`direct::etree::tests`，3 项）

### Sprint 15（M15）：多前沿 LU + BLR 预条件器 ✅ 已完成
- [x] `direct/multifrontal.rs`：`MultifrontalLu<T>`，消去树后序驱动
- [x] `MultifrontalOptions { blr_min_size, blr_tol }`，BLR 参数预留接口
- [x] `MultifrontalLu::with_blr(tol, min_size)` 便捷构造
- [x] `MultifrontalLu` 可包装为 `DirectSolverPrecond`（GMRES/CG 1–3 次迭代收敛）
- [x] 13 项集成测试（`tests/test_direct_s14_s15.rs`）
- [x] lib.rs 新增公开 re-export：`MultifrontalLu`、`MultifrontalOptions`

### B1：纯 Rust 嵌套剖分（NodeNd）✅ 已完成
- [x] `direct/ordering/nd.rs`：完整多层嵌套剖分
  - HEM 多层粗化（Heavy-Edge Matching，target=60）
  - BFS 双极着色（双端 BFS 找最远点对）
  - 顶点分隔符提取 + FM 贪心细化（最多 5 轮，balance_slack=0.20）
  - 递归嵌套剖分（MAX_DEPTH=64，小图退回 AMD）
- [x] `OrderingMethod::NodeNd` enum 变体，三个求解器（SparseLu/SparseCholesky/MultifrontalLu）集成
- [x] `Ordering::NodeNd` 在 `builder.rs` SolverBuilder 中映射
- [x] 14 项集成测试（`tests/test_ordering_nd.rs`）

### C1+C2：WASM 直接求解器 API + SolverBuilder ✅ 已完成
- [x] `src/wasm.rs`：`WasmLuSolver`、`WasmCholeskySolver`、`WasmMultifrontalSolver`（wasm-bindgen）
- [x] `src/builder.rs`：`SolverBuilder` fluent API
  - `SolveMethod::{Cg, Gmres{restart}, BiCgStab, Direct(DirectBackend)}`
  - `PrecondChoice::{None, Jacobi, Ilu0, Icc0, DirectLu(DirectBackend)}`
  - `Ordering::{Natural, Rcm, Colamd, NodeNd}`
  - `solve_auto(a, b, spd_hint)` 便捷入口
- [x] 15 项集成测试（`tests/test_builder_c2.rs`）

### A2：迭代精化激活 ✅ 已完成
- [x] `direct/lu.rs`：`SparseLu::solve()` 末尾循环：`r=b−Ax` → 复用 L,U 求解 `Aδx=r` → `x+=δx`
- [x] `direct/cholesky.rs`：`SparseCholesky::solve()` 同上，SPD 路径
- [x] `refine_steps>0` 时在 `factorize()` 中保存 A 副本（`a_stored: Option<CsrMatrix<T>>`）
- [x] 7 项集成测试（`tests/test_refinement.rs`）

### B2：符号 Cholesky 精确模式预测 ✅ 已完成
- [x] `direct/symbolic.rs`：新增 `symbolic_cholesky()` 函数（左视 DFS 达到集，O(nnz(L))）
- [x] `SymbolicCholesky` 结构体：CSC 格式 L 列模式 + `col_count` + `parent`
- [x] `direct/etree.rs`：修正 `col_counts()` 算法（替换为正确的 mark-and-sweep，Davis 2006 Algorithm 4.7）
- [x] `direct/mod.rs`：公开导出 `SymbolicCholesky`、`symbolic_cholesky`
- [x] 13 项集成测试（`tests/test_symbolic_cholesky.rs`）

### C3：Matrix Market 文件读取器 ✅ 已完成
- [x] `sparse/mmio.rs`：`read_matrix_market()`、`read_matrix_market_coo()`
  - 解析 `%%MatrixMarket matrix coordinate real general/symmetric/pattern` 头
  - 支持对称矩阵自动展开（下三角→完整）
  - 1-based 索引转 0-based
- [x] `sparse/mod.rs`：公开导出 `mmio` 模块及函数
- [x] `lib.rs`：公开 re-export `read_matrix_market`、`read_matrix_market_coo`、`MmioError`
- [x] 14 项集成测试（`tests/test_mmio.rs`）

### B3：reuse_symbolic 性能优化 ✅ 已完成
- [x] `direct/lu.rs`：`SparseLu` 新增 `symbolic_n` 缓存字段
  - `analyze()` 当 `reuse_symbolic=true` 且矩阵尺寸匹配时跳过重排序
- [x] `direct/cholesky.rs`：`SparseCholesky` 同上
- [x] `direct/multifrontal.rs`：`MultifrontalLu` 同上
- [x] 新增公开访问器：`SparseLu::perm_q()`、`SparseLu::perm_p()`、`SparseCholesky::perm()`、`MultifrontalLu::perm_q()`
- [x] 10 项集成测试（`tests/test_reuse_symbolic.rs`）

### A3：稀疏右视 LU（Symbolic 基础）✅ 已完成
- [x] `direct/symbolic.rs`：修复 `symbolic_lu()` L 列模式计算
  - 原实现仅用 A 的直接下三角项，现通过 etree 父链接传播 fill（Gilbert-Peierls）
- [x] `direct/lu.rs`：引入 `elimination_tree`、`symbolic_lu` 导入
  - `factorize()` 现在先计算 symbolic 模式（为后续 O(nnz) 左视数值分解做准备）

### A1：直接法性能基准 ✅ 已完成
- [x] `benches/bench_direct.rs`：6 组 Criterion 基准
  - `SparseLu/factorize`、`SparseLu/solve`（1D/2D Laplacian）
  - `SparseCholesky/factorize`（1D/2D，n=50-400）
  - `MultifrontalLu/factorize`（1D，n=30-100）
  - `Cholesky/ordering`（Natural/Rcm/Colamd/NodeNd 对比，16×16 网格）
  - `SparseLu/refinement`（refine_steps 0-3 开销对比）
- [x] `Cargo.toml` 新增 `[[bench]] name = "bench_direct" harness = false`

**当前状态**：全部测试通过（`cargo test`）。`cargo bench --bench bench_direct -- --test` 全部 Success。

---

## 7. 与 linger 主项目的集成接口

`linger` 作为独立 crate，通过以下接口供 FEA 主项目调用：

```rust
// linger 主项目中的典型使用
use linger_solver::{
    SolverBuilder, KrylovMethod, PrecondType,
};

// FEA 全局刚度矩阵（装配完成的 nalgebra CSR）
let stiffness: nalgebra_sparse::CsrMatrix<f64> = assemble_stiffness(&mesh, &material);

let solver = SolverBuilder::new()
    .method(KrylovMethod::CG)
    .preconditioner(PrecondType::AMG {
        coarsening: CoarseningStrategy::SmoothedAggregation,
        smoother: SmootherType::WeightedJacobi { omega: 0.67 },
        max_levels: 10,
    })
    .rtol(1e-8)
    .max_iter(500)
    .build::<f64>()?;

let mut displacement = vec![0.0_f64; ndof];
let result = solver.solve(&stiffness, &force_vector, &mut displacement)?;

println!("Converged in {} iterations, residual = {:.3e}",
    result.iterations, result.final_residual);
```

---

## 8. 参考文献

1. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.)
2. Trottenberg, U., Oosterlee, C., & Schüller, A. (2001). *Multigrid*
3. Falgout, R. D., & Yang, U. M. (2002). *hypre: A Library of High Performance Preconditioners*
4. Balay, S. et al. *PETSc Users Manual* (ANL-95/11 Rev 3.20)
5. Dongarra, J. et al. (1990). *Templates for the Solution of Linear Systems*
6. Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems.* SIAM.
7. Gilbert, J.R., Ng, E.G., and Peierls, B.W. (1994). *Sparse partial pivoting in time proportional to arithmetic.* SIAM J. Sci. Comput., 15(5), 1075–1091.
8. Liu, J.W.H. (1986). *A compact row storage scheme for Cholesky factors using elimination trees.* ACM Trans. Math. Softw., 12(2), 127–148.
9. Duff, I.S. and Reid, J.K. (1983). *The multifrontal solution of indefinite sparse symmetric linear equations.* ACM Trans. Math. Softw., 9(3), 302–325.
10. Amestoy, P.R. et al. (2015). *Improving multifrontal methods by means of block low-rank representations.* SIAM J. Sci. Comput., 37(3), A1452–A1474.
11. Cuthill, E. and McKee, J. (1969). *Reducing the bandwidth of sparse symmetric matrices.* Proc. ACM National Conference.
