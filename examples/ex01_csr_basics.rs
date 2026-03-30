//! ex01 — CSR matrix construction, SpMV, and structural queries.
//!
//! **Purpose**: Validate the core sparse-matrix pipeline end-to-end:
//!   COO assembly → CSR conversion → SpMV → transpose → diagonal extraction.
//!
//! **HYPRE analog**
//!   `HYPRE_IJMatrixCreate` / `HYPRE_IJMatrixSetValues`
//!   `HYPRE_ParCSRMatrixMatvec`           (parcsr_mv/par_csr_matvec.c)
//!
//! **PETSc analog**
//!   `MatCreate` / `MatSetType(MATAIJ)` / `MatSetValues` / `MatAssemblyBegin`
//!   `MatMult`                            (mat/impls/aij/seq/aij.c)

use linger::{
    sparse::{CooMatrix, CsrMatrix},
    LinearOperator,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn sep(title: &str) {
    println!("\n━━━━ {title} ━━━━━");
}

fn check(label: &str, ok: bool) {
    let mark = if ok { "✓" } else { "✗ FAIL" };
    println!("  {mark}  {label}");
    assert!(ok, "check failed: {label}");
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    sep("ex01: CSR basics  (HYPRE ParCSRMatvec / PETSc MatMult)");

    // ── 1. Assemble a 5×5 tridiagonal matrix via COO ─────────────────────────
    //
    //     [ 2 -1  0  0  0 ]
    //     [-1  2 -1  0  0 ]
    // A = [ 0 -1  2 -1  0 ]
    //     [ 0  0 -1  2 -1 ]
    //     [ 0  0  0 -1  2 ]
    //
    // PETSc would call MatSetValues with ADD_VALUES; we push COO triplets.
    let n = 5usize;
    let mut coo: CooMatrix<f64> = CooMatrix::with_capacity(n, n, 3 * n - 2);
    for i in 0..n {
        if i > 0     { coo.push(i, i - 1, -1.0); }
        coo.push(i, i, 2.0);
        if i < n - 1 { coo.push(i, i + 1, -1.0); }
    }

    println!("\n  Assembly  (COO → CSR)");
    println!("  COO: {}×{}, {} raw triplets", coo.nrows(), coo.ncols(), coo.nnz());

    let a: CsrMatrix<f64> = CsrMatrix::from_coo(&coo);
    println!("  CSR: {}×{}, {} nnz", a.nrows(), a.ncols(), a.nnz());

    check("nrows == 5",   a.nrows() == 5);
    check("ncols == 5",   a.ncols() == 5);
    check("nnz == 13",    a.nnz() == 13);   // 5 diag + 4 lower + 4 upper

    // ── 2. SpMV  y = A · e₁  (first unit vector) ─────────────────────────────
    //
    // A·e₁ = first column of A = [2, -1, 0, 0, 0]ᵀ
    println!("\n  SpMV  y = A·e₁");
    let x = vec![1.0f64, 0.0, 0.0, 0.0, 0.0];
    let mut y = vec![0.0f64; n];
    a.spmv(&x, &mut y);
    println!("  y = {:?}", y);
    check("y[0] == 2",   (y[0] -  2.0).abs() < 1e-15);
    check("y[1] == -1",  (y[1] - -1.0).abs() < 1e-15);
    check("y[2] == 0",   y[2].abs() < 1e-15);

    // ── 3. Via LinearOperator trait (DenseVec path) ───────────────────────────
    println!("\n  LinearOperator::apply  (DenseVec path)");
    use linger::DenseVec;
    let xv = DenseVec::from_vec(vec![0.0f64, 0.0, 1.0, 0.0, 0.0]); // e₂ (0-indexed)
    let mut yv = DenseVec::zeros(n);
    a.apply(&xv, &mut yv);  // A·e₂ = [0, -1, 2, -1, 0]ᵀ
    println!("  A·e₂ = {:?}", yv.as_slice());
    check("y[1] == -1", (yv[1] - -1.0).abs() < 1e-15);
    check("y[2] ==  2", (yv[2] -  2.0).abs() < 1e-15);
    check("y[3] == -1", (yv[3] - -1.0).abs() < 1e-15);

    // ── 4. spmv_add  y = α·A·x + β·y ─────────────────────────────────────────
    println!("\n  spmv_add  y = 2·A·e₁ + 3·y  (where y = A·e₂)");
    let x_e1 = vec![1.0f64, 0.0, 0.0, 0.0, 0.0];
    let mut y2 = vec![0.0f64, -1.0, 2.0, -1.0, 0.0]; // A·e₂
    a.spmv_add(2.0, &x_e1, 3.0, &mut y2);
    // Expected: 2·[2,-1,0,0,0] + 3·[0,-1,2,-1,0] = [4, -5, 6, -3, 0]
    println!("  y = {:?}", y2);
    check("y[0] ==  4", (y2[0] -  4.0).abs() < 1e-15);
    check("y[1] == -5", (y2[1] - -5.0).abs() < 1e-15);
    check("y[2] ==  6", (y2[2] -  6.0).abs() < 1e-15);
    check("y[3] == -3", (y2[3] - -3.0).abs() < 1e-15);

    // ── 5. Diagonal extraction ────────────────────────────────────────────────
    println!("\n  Diagonal extraction");
    let diag = a.diag();
    println!("  diag = {:?}", diag);
    check("all diag == 2", diag.iter().all(|&d| (d - 2.0).abs() < 1e-15));

    // ── 6. Transpose (CSR → CSC) ──────────────────────────────────────────────
    //
    // A is symmetric, so Aᵀ·x == A·x for all x.
    println!("\n  Transpose  Aᵀ·e₁  (CSC SpMV)");
    let at = a.transpose();
    check("Aᵀ nrows == 5", at.nrows() == 5);
    check("Aᵀ ncols == 5", at.ncols() == 5);
    check("Aᵀ nnz == 13",  at.nnz() == 13);

    let xe1 = vec![1.0f64, 0.0, 0.0, 0.0, 0.0];
    let mut yt = vec![0.0f64; n];
    at.spmv(&xe1, &mut yt);           // Aᵀ·e₁ = A·e₁ (symmetric)
    println!("  Aᵀ·e₁ = {:?}", yt);
    check("Aᵀ·e₁ == A·e₁", yt.iter().zip(y.iter()).all(|(a, b)| (a - b).abs() < 1e-15));

    // ── 7. Structural symmetry check ─────────────────────────────────────────
    println!("\n  Structural symmetry");
    check("A is structurally symmetric", a.is_structurally_symmetric());

    // ── 8. Duplicate COO entries are summed ───────────────────────────────────
    println!("\n  Duplicate-entry summing");
    let mut coo2: CooMatrix<f64> = CooMatrix::new(2, 2);
    coo2.push(0, 0, 1.0);
    coo2.push(0, 0, 1.0); // duplicate: should sum to 2.0
    coo2.push(0, 1, -1.0);
    coo2.push(1, 0, -1.0);
    coo2.push(1, 1, 1.0);
    let a2 = CsrMatrix::from_coo(&coo2);
    check("nnz == 4 (duplicates merged)", a2.nnz() == 4);
    check("diag[0] == 2 (1+1 summed)",    (a2.diag()[0] - 2.0).abs() < 1e-15);

    println!("\n  OK\n");
}
