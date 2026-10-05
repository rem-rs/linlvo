//! D1215 pins (round 115, `tmp/d115a/`): `hmis::rap_hypre_order` is
//! bitwise-faithful to hypre's `hypre_BoomerAMGBuildCoarseOperatorKT`
//! two-stage Galerkin accumulation (`parcsr_ls/par_rap.c:16-22`, single-rank
//! value pass at `par_rap.c:1608-1985`).
//!
//! Measured on the tesla ball-quad o2 level-0 pair (d115a): an independent
//! plain-C transcription of the KT value pass equals `rap_hypre_order`
//! bit-for-bit on A(1460², 141448 nnz) × Pi(1460×517) → A_Pi 29521/29521
//! entries bitwise, and equals real hypre 2.28 bit-for-bit (29521/29521)
//! once hypre's matrix sees the same A *storage order* — hypre's own IJ
//! assembly normalizes rows to `[diagonal, off-diagonals in insertion
//! order]`, and the C++/MFEM handoff order (hypre PtAP emission over the
//! MFEM linked-list local assembly) is a third order again.  The RAP
//! algorithm itself is exact; the tesla o2 7/6-vs-8/7 residual lives in
//! that upstream storage order (D1230), not here.
//!
//! These pins freeze the two-stage semantics: the RA first-touch order
//! consumed by stage 2 is part of the contract, so any change to the
//! traversal (e.g. sorting RA columns) shows up as a last-ulp change that
//! HMIS strength ties amplify into different coarse grids.

use linlvo::amg::hmis::rap_hypre_order;
use linlvo::sparse::CsrMatrix;

/// Bit-exact map of a RAP output: `(row, col) -> f64 bits`.
fn entry_bits(m: &CsrMatrix<f64>) -> Vec<((usize, usize), u64)> {
    let mut v: Vec<((usize, usize), u64)> = (0..m.nrows())
        .flat_map(|r| {
            (m.row_ptr()[r]..m.row_ptr()[r + 1])
                .map(move |k| ((r, m.col_idx()[k]), m.values()[k].to_bits()))
        })
        .collect();
    v.sort_unstable();
    v
}

/// Independent transcription of hypre's KT value pass (single rank): R = Pᵀ
/// with stable-ascending rows (the single-thread `hypre_CSRMatrixTranspose`
/// counting sort), then per coarse row `ic`: stage 1 RA create-or-add over
/// (i1 in R row ic, stored order) × (A row i1, stored order), stage 2 RAP
/// create-or-add consuming RA in first-touch order × (P row i2, stored
/// order).  Returns `(row, col, value bits)` in first-touch order per row.
fn kt_replica(p: &CsrMatrix<f64>, a: &CsrMatrix<f64>) -> Vec<((usize, usize), u64)> {
    // stable transpose by hand (NOT linlvo's transpose_csr — independence)
    let (n, m) = (p.nrows(), p.ncols());
    let mut counts = vec![0usize; m];
    for &c in p.col_idx() {
        counts[c] += 1;
    }
    let mut rt_starts = vec![0usize; m + 1];
    for j in 0..m {
        rt_starts[j + 1] = rt_starts[j] + counts[j];
    }
    let mut rt_r = vec![0usize; p.nnz()];
    let mut rt_v = vec![0.0_f64; p.nnz()];
    let mut cur = rt_starts[..m].to_vec();
    for i in 0..n {
        for k in p.row_ptr()[i]..p.row_ptr()[i + 1] {
            let pos = cur[p.col_idx()[k]];
            rt_r[pos] = i;
            rt_v[pos] = p.values()[k];
            cur[p.col_idx()[k]] += 1;
        }
    }

    let mut out = Vec::new();
    for ic in 0..m {
        // stage 1: RA row, first-touch create-or-add
        let mut ra_j: Vec<usize> = Vec::new();
        let mut ra_v: Vec<f64> = Vec::new();
        for k1 in rt_starts[ic]..rt_starts[ic + 1] {
            let i1 = rt_r[k1];
            let v1 = rt_v[k1];
            for k2 in a.row_ptr()[i1]..a.row_ptr()[i1 + 1] {
                let i2 = a.col_idx()[k2];
                let term = v1 * a.values()[k2];
                match ra_j.iter().position(|&x| x == i2) {
                    Some(idx) => ra_v[idx] += term,
                    None => {
                        ra_j.push(i2);
                        ra_v.push(term);
                    }
                }
            }
        }
        // stage 2: RAP row, create-or-add in RA first-touch order
        let mut rap_j: Vec<usize> = Vec::new();
        let mut rap_v: Vec<f64> = Vec::new();
        for (i2, &ra) in ra_j.iter().zip(ra_v.iter()) {
            for k3 in p.row_ptr()[*i2]..p.row_ptr()[*i2 + 1] {
                let k = p.col_idx()[k3];
                let term = ra * p.values()[k3];
                match rap_j.iter().position(|&x| x == k) {
                    Some(idx) => rap_v[idx] += term,
                    None => {
                        rap_j.push(k);
                        rap_v.push(term);
                    }
                }
            }
        }
        for (k, v) in rap_j.into_iter().zip(rap_v.into_iter()) {
            out.push(((ic, k), v.to_bits()));
        }
    }
    out.sort_unstable();
    out
}

/// Deterministic LCG (no external rng dependency).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn f64_unit(&mut self) -> f64 {
        (self.next() % 1_000_000) as f64 / 1_000_000.0 - 0.5
    }
}

/// Random SPD-ish A (n×n, dominant diagonal + a few off-diagonals per row)
/// and rectangular P (n×m, 2–3 nonzeros per row, overlapping columns).
/// Returns the entries so the caller can emit any row storage order.
fn random_case(
    seed: u64,
    n: usize,
    m: usize,
) -> (Vec<(usize, usize, f64)>, Vec<(usize, usize, f64)>) {
    let mut g = Lcg(seed | 1);
    let mut a = Vec::new();
    for i in 0..n {
        a.push((i, i, 4.0 + g.f64_unit()));
        for _ in 0..3 {
            let j = (g.next() as usize) % n;
            if j != i {
                a.push((i, j, g.f64_unit() * 2.0));
                a.push((j, i, g.f64_unit() * 2.0)); // keep A structurally symmetric
            }
        }
    }
    let mut p = Vec::new();
    for i in 0..n {
        let deg = 2 + (g.next() as usize) % 2;
        let base = (g.next() as usize) % m;
        for d in 0..deg {
            let j = (base + d * (1 + (g.next() as usize) % 3)) % m;
            let v = g.f64_unit();
            if !p.iter().any(|&(r, c, _)| r == i && c == j) {
                p.push((i, j, v));
            }
        }
    }
    (a, p)
}

/// Emit entries as a CSR with rows in the given internal order:
/// `sorted` (ascending columns) vs `diag_first` (the `[diagonal, rest]`
/// layout hypre's IJ assembly produces).
fn to_csr(
    nrows: usize,
    ncols: usize,
    mut entries: Vec<(usize, usize, f64)>,
    diag_first: bool,
) -> CsrMatrix<f64> {
    entries.sort_by_key(|&(r, c, _)| (r, if diag_first && c == r { 0 } else { 1 }, c));
    let mut row_ptr = vec![0usize; nrows + 1];
    for &(r, _, _) in &entries {
        row_ptr[r + 1] += 1;
    }
    for i in 0..nrows {
        row_ptr[i + 1] += row_ptr[i];
    }
    let col_idx: Vec<usize> = entries.iter().map(|&(_, c, _)| c).collect();
    let values: Vec<f64> = entries.iter().map(|&(_, _, v)| v).collect();
    CsrMatrix::from_raw(nrows, ncols, row_ptr, col_idx, values)
}

/// `rap_hypre_order` == the KT two-stage value pass, bit-for-bit, on random
/// inputs, and the outputs are (as sets) the algebraic PᵀAP sparsity.
#[test]
fn d1215_rap_matches_kt_two_stage_bitwise() {
    for seed in [1_u64, 2, 3, 7, 42] {
        let (a, p) = random_case(seed, 40, 13);
        let a_sorted = to_csr(40, 40, a.clone(), false);
        let p_sorted = to_csr(40, 13, p.clone(), false);
        let got = entry_bits(&rap_hypre_order(&p_sorted, &a_sorted));
        let want = kt_replica(&p_sorted, &a_sorted);
        assert_eq!(got, want, "seed {seed}: RAP != KT replica (sorted order)");
    }
}

/// The A *storage order* is part of the RAP float contract: the same
/// entries with rows laid out `[diagonal, off-diagonals]` (hypre IJ / C++
/// handoff family) vs fully sorted can round differently — and
/// `rap_hypre_order` must track whichever order it is handed bit-for-bit
/// (this is why D1215's tesla residual is an input-order question, not an
/// algorithm question).  Searched deterministically for a witness case.
#[test]
fn d1215_rap_tracks_input_storage_order() {
    let mut witness = None;
    for seed in 1..200_u64 {
        let (a, p) = random_case(seed, 24, 9);
        let a_sorted = to_csr(24, 24, a.clone(), false);
        let a_dfirst = to_csr(24, 24, a.clone(), true);
        let p_csr = to_csr(24, 9, p.clone(), false);
        let s = kt_replica(&p_csr, &a_sorted);
        let d = kt_replica(&p_csr, &a_dfirst);
        if s != d {
            witness = Some((seed, a_sorted, a_dfirst, p_csr, s, d));
            break;
        }
    }
    let (seed, a_sorted, a_dfirst, p_csr, want_sorted, want_dfirst) =
        witness.expect("no storage-order witness found in seeds 1..200");
    assert_eq!(
        entry_bits(&rap_hypre_order(&p_csr, &a_sorted)),
        want_sorted,
        "seed {seed}: sorted-order RAP != KT replica"
    );
    assert_eq!(
        entry_bits(&rap_hypre_order(&p_csr, &a_dfirst)),
        want_dfirst,
        "seed {seed}: diag-first-order RAP != KT replica"
    );
}
