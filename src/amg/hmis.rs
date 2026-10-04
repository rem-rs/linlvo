//! hypre BoomerAMG HMIS coarsening + aggressive second shot + interpolation —
//! serial (one-rank) port of the exact configuration MFEM 4.10
//! `HypreAMS::MakeSolver` hands to hypre 2.28 for the AMS Pi-block solvers.
//!
//! Every routine cites its hypre source (2.28.0):
//!
//! * [`classical_strength`] — `hypre_BoomerAMGCreateSHost`, measure 0
//!   (`parcsr_ls/par_strength.c:75-400`): row `i` depends on `j` iff
//!   `a_ij < θ·min_{k≠i}(a_ik)` when `a_ii ≥ 0` (signed comparison — positive
//!   entries are never strong), or `a_ij > θ·max_{k≠i}(a_ik)` when `a_ii < 0`;
//!   rows with `|row_sum| > |a_ii|·max_row_sum` have all dependencies made
//!   weak (`par_strength.c:316-321`).  BoomerAMG's default `max_row_sum = 0.9`
//!   (`parcsr_ls/par_amg.c:165`) is active — MFEM does not override it.
//! * [`coarsen_hmis`] — HMIS is `hypre_BoomerAMGCoarsenHMIS` →
//!   `hypre_BoomerAMGCoarsenRuge(S, A, measure, 10, …)` (`par_coarsen.c:2846`),
//!   which sets `f_pnt = Z_PT` and `coarsen_type = 11` (`par_coarsen.c:1148`)
//!   and runs ONLY the Ruge first pass (measure = number of undecided
//!   influences = row sums of Sᵀ, `par_coarsen.c:1102-1110`; greedy list with
//!   measure updates `par_coarsen.c:1280-1370`).  Isolated rows become
//!   `SC_PT` (strong coarse) when `agg_2` (measure 3/4, `par_coarsen.c:992`),
//!   else `SF_PT`; measure-0 points become `Z_PT`.  The serial measure
//!   carries no randomness.  Ties are broken FIFO by insertion order
//!   (`utilities/amg_linklist.c:196-206`: equal-measure entries are appended
//!   at the bucket tail; the pick is the bucket head), reproduced here via
//!   `(max measure, min insertion seq)`.
//! * [`create_second_strength`] — `hypre_BoomerAMGCreate2ndS`
//!   (`par_strength.c:1795, 2380-2456`) with `num_paths = 1` (BoomerAMG
//!   default, `par_amg.c:183`): the coarse-point graph `S2` where row `ic`
//!   (a C-point `i1`) is connected to every C-point reachable in ONE hop
//!   (`S(i1)`) or TWO hops (`S(j), j ∈ S(i1)`, excluding `ic` itself) —
//!   the "S*S+2S" pattern restricted to coarse points.
//! * [`correct_cf_marker`] — `hypre_BoomerAMGCorrectCFMarker`
//!   (`par_strength.c:3064-3088`): each first-pass C-point receives the
//!   second-pass verdict (and may be demoted to F/Z).
//! * [`interp_multipass`] — multipass interpolation, the aggressive-level
//!   interpolation: `level < agg_num_levels = 1` uses `agg_interp_type`
//!   (BoomerAMG default 4 = `hypre_BoomerAMGBuildMultipass`,
//!   `par_amg_setup.c:1695-1770`, `par_amg.c:186`) with no truncation
//!   (`agg_trunc_factor = 0`, `agg_P_max_elmts = 0`, `par_amg.c:167,176`)
//!   and `sep_weight = 0` for interp_type 6 (`par_amg_setup.c:338-347`).
//!   Serial algorithm (`par_multi_interp.c:16-1900`): C points → unit rows;
//!   F points get a pass number (pass 1 = has a strong C neighbour; pass k =
//!   has a strong neighbour of pass k−1; at most `max_num_passes = 10`
//!   passes, `par_multi_interp.c:127,525`); the pass-k interpolatory set is
//!   the union of the pass-(k−1) sets of its pass-(k−1) strong neighbours;
//!   pass-1 weights are the raw `a_ij` of the strong C-columns, pass-k
//!   weights accumulate `a_ij·P[j,c]` over pass-(k−1) strong neighbours `j`,
//!   `sum_N` collects everything else except `SF_PT` columns (pass-1: every
//!   non-`SF` off-diagonal), and the row is scaled by
//!   `alfa = −sum_N / (sum_C·a_ii)`.  `Z_PT`/`SF_PT`/never-assigned rows
//!   stay EMPTY.
//! * [`interp_ext_pi`] — extended+i classical interpolation, interp_type 6 =
//!   `hypre_BoomerAMGBuildExtPIInterp` (`par_amg_setup.c:2234-2239`,
//!   `par_lr_interp.c:1024-1860`): classical modified interpolation whose
//!   interpolatory set adds, beyond the strong C-neighbours of `i`, the
//!   strong C-neighbours of every non-`SF` strong neighbour (`F` **and**
//!   `Z`) — NO common-C requirement, which is what distinguishes
//!   "extended+i" from interp 0; two-pass weight accumulation with diagonal
//!   distribution over strong-F rows (same-sign connections only, `sgn`
//!   from `a_i1i1`, `par_lr_interp.c:1555-1699`), then division by
//!   `−diagonal` when it is nonzero (`par_lr_interp.c:1799-1807`).
//! * [`truncate_pmax`] — `hypre_BoomerAMGInterpTruncation(P, 0, P_max_elmts)`
//!   (`par_interp.c:2632-2650`) → `hypre_ParCSRMatrixTruncate(tol=0,
//!   max_row_elmts, rescale=1, nrm_type=0)` (`parcsr_mv/par_csr_matrix.c:2266`,
//!   keep/drop/rescale at 2398-2432): keep the `max_row_elmts` largest-|·|
//!   entries per row and rescale the row so its (signed) sum is preserved.
//!   MFEM sets `P_max_elmts = 4` (`linalg/hypre.cpp`, `MakeSolver`:
//!   `amg_Pmax = 4`).

use crate::core::scalar::{ComplexScalar, Scalar};
use crate::sparse::{CooMatrix, CsrMatrix};
use num_traits::Zero;

/// hypre C/F marker values (`par_coarsen_device.c:12-16`, `par_cgc_coarsen.c:19-23`).
pub(crate) const C_PT: i32 = 1;
pub(crate) const F_PT: i32 = -1;
pub(crate) const Z_PT: i32 = -2;
pub(crate) const SF_PT: i32 = -3;
pub(crate) const SC_PT: i32 = 4;

/// BoomerAMG default `max_row_sum` (`parcsr_ls/par_amg.c:165`).
const MAX_ROW_SUM: f64 = 0.9;
/// `max_num_passes` of multipass interpolation (`par_multi_interp.c:127`).
const MAX_NUM_PASSES: usize = 10;
/// MFEM `HypreAMS::MakeSolver` `amg_Pmax` (linalg/hypre.cpp).
const AMS_PMAX: usize = 4;

// ─── Strength (measure 0) ─────────────────────────────────────────────────────

/// Classical (measure-0) strength matrix of `a` — serial port of
/// `hypre_BoomerAMGCreateSHost` with `num_functions = 1` (pattern-only,
/// unit values; all consumers use the S pattern).  Signed comparisons act on
/// the real part of the entries (the port targets real-valued operators).
pub fn classical_strength<T: ComplexScalar>(
    a:     &CsrMatrix<T>,
    theta: f64,
) -> CsrMatrix<T> {
    let n     = a.nrows();
    let rp    = a.row_ptr();
    let ci    = a.col_idx();
    let vs    = a.values();
    let theta = <T::Real as Scalar>::from_f64(theta);
    let max_row_sum = <T::Real as Scalar>::from_f64(MAX_ROW_SUM);

    let compute_row = |i: usize| -> Vec<(usize, T)> {
        let diag = {
            let mut d = T::zero();
            for k in rp[i]..rp[i + 1] {
                if ci[k] == i {
                    d = vs[k];
                    break;
                }
            }
            d
        };
        let diag_r = diag.real();

        // row_scale = max/min of SIGNED off-diagonal values
        // (par_strength.c:257-303); row_sum = diag + Σ off-diag (:255).
        let mut row_scale = T::Real::zero();
        let mut row_sum = diag_r;
        let mut first = true;
        for k in rp[i]..rp[i + 1] {
            if ci[k] == i {
                continue;
            }
            let v = vs[k].real();
            row_sum += v;
            if diag_r < T::Real::zero() {
                if first || v > row_scale {
                    row_scale = v;
                }
            } else if first || v < row_scale {
                row_scale = v;
            }
            first = false;
        }

        // Ill-scaled rows: all dependencies weak (par_strength.c:316-321).
        if MAX_ROW_SUM < 1.0 && row_sum.abs() > diag_r.abs() * max_row_sum {
            return Vec::new();
        }

        let mut row = Vec::new();
        for k in rp[i]..rp[i + 1] {
            let j = ci[k];
            if j == i {
                continue;
            }
            let strong = if diag_r < T::Real::zero() {
                vs[k].real() > theta * row_scale
            } else {
                vs[k].real() < theta * row_scale
            };
            if strong {
                row.push((j, T::one()));
            }
        }
        row
    };

    pack_rows(n, n, (0..n).map(compute_row).collect())
}

// ─── HMIS coarsening (Ruge first pass, type 11) ───────────────────────────────

/// Serial HMIS coarsening: the `CoarsenRuge` first pass with `f_pnt = Z_PT`
/// (`coarsen_type 10 → Z_PT/11`, `par_coarsen.c:1148-1151`) and no second
/// pass (`par_coarsen.c:1379-1387`).  `agg_2` selects the `SC_PT` handling of
/// isolated rows for the aggressive second shot (measure 3,
/// `par_coarsen.c:992,1207-1212`).
pub fn coarsen_hmis<T: ComplexScalar>(s: &CsrMatrix<T>, agg_2: bool) -> Vec<i32> {
    let n   = s.nrows();
    let rp  = s.row_ptr();
    let ci  = s.col_idx();

    // measure[i] = #influences = row sum of Sᵀ (par_coarsen.c:1102-1110).
    let st  = s.transpose_csr();
    let trp = st.row_ptr();
    let tci = st.col_idx();

    let mut cf = vec![0_i32; n]; // UNDECIDED
    let mut lam: Vec<i64> = (0..n).map(|i| (trp[i + 1] - trp[i]) as i64).collect();

    // Init: isolated rows → SF_PT (agg_2: SC_PT), measure 0
    // (par_coarsen.c:1202-1216).  (The `cut_factor` dense-row marking is
    // skipped: BoomerAMG default `coarsen_cut_factor = 0`, par_amg.c:160.)
    for j in 0..n {
        if rp[j + 1] - rp[j] == 0 {
            cf[j] = if agg_2 { SC_PT } else { SF_PT };
            lam[j] = 0;
        }
    }

    // The greedy loop keeps the invariant that a point is on hypre's lists
    // iff it is UNDECIDED with measure > 0; equal-measure entries are
    // appended at the bucket tail (amg_linklist.c:196-206) so the pick is
    // (max measure, earliest insertion).  `seq` records the insertion order
    // of the last (re-)entry, which makes a plain scan over `in_list`
    // equivalent to the linked-list structure.
    let mut in_list = vec![false; n];
    let mut seq = vec![0_u64; n];
    let mut next_seq: u64 = 1;

    // Initial listing (par_coarsen.c:1219-1262): measure-0 undecided points
    // become f_pnt = Z_PT and increment the measures of their (undecided)
    // S-neighbours — already-processed neighbours (`nabor < j`) are removed
    // from the lists first and re-entered with the new measure.
    for j in 0..n {
        if cf[j] == SF_PT || cf[j] == SC_PT {
            continue;
        }
        if lam[j] > 0 {
            in_list[j] = true;
            seq[j] = next_seq;
            next_seq += 1;
        } else {
            cf[j] = Z_PT;
            for k in rp[j]..rp[j + 1] {
                let nabor = ci[k];
                if cf[nabor] != SF_PT && cf[nabor] != SC_PT {
                    if nabor < j && in_list[nabor] {
                        in_list[nabor] = false;
                    }
                    lam[nabor] += 1;
                    if nabor < j {
                        in_list[nabor] = true;
                        seq[nabor] = next_seq;
                        next_seq += 1;
                    }
                }
            }
        }
    }

    // Bump the measure of every undecided S-neighbour of `j`
    // (remove + re-enter, par_coarsen.c:1245-1257 and 1350-1360).
    fn bump(
        j:       usize,
        rp:      &[usize],
        ci:      &[usize],
        cf:      &[i32],
        lam:     &mut [i64],
        in_list: &mut [bool],
        seq:     &mut [u64],
        next_seq: &mut u64,
    ) {
        for k in rp[j]..rp[j + 1] {
            let nb = ci[k];
            if cf[nb] == 0 {
                in_list[nb] = false;
                lam[nb] += 1;
                in_list[nb] = true;
                seq[nb] = *next_seq;
                *next_seq += 1;
            }
        }
    }

    // Main greedy loop (par_coarsen.c:1280-1370).
    loop {
        // Pick: max measure, FIFO tie-break (earliest seq).
        let mut best: Option<usize> = None;
        for i in 0..n {
            if in_list[i] {
                best = match best {
                    None => Some(i),
                    Some(b) => {
                        if lam[i] > lam[b] || (lam[i] == lam[b] && seq[i] < seq[b]) {
                            Some(i)
                        } else {
                            Some(b)
                        }
                    }
                };
            }
        }
        let c = match best {
            Some(c) => c,
            None => break,
        };

        cf[c] = C_PT;
        lam[c] = 0;
        in_list[c] = false;

        // Undecided points that strongly depend on c → F
        // (par_coarsen.c:1288-1318).
        for k in trp[c]..trp[c + 1] {
            let j = tci[k];
            if cf[j] == 0 {
                cf[j] = F_PT;
                in_list[j] = false;
                bump(j, rp, ci, &cf, &mut lam, &mut in_list, &mut seq, &mut next_seq);
            }
        }
        // Undecided points c depends on lose a measure; at measure ≤ 0 they
        // turn F with the same neighbour bump (par_coarsen.c:1320-1365).
        for k in rp[c]..rp[c + 1] {
            let j = ci[k];
            if cf[j] == 0 {
                in_list[j] = false;
                lam[j] -= 1;
                if lam[j] > 0 {
                    in_list[j] = true;
                    seq[j] = next_seq;
                    next_seq += 1;
                } else {
                    cf[j] = F_PT;
                    bump(j, rp, ci, &cf, &mut lam, &mut in_list, &mut seq, &mut next_seq);
                }
            }
        }
    }

    // SC → C conversion (par_coarsen.c:1373-1377).
    if agg_2 {
        for v in cf.iter_mut() {
            if *v == SC_PT {
                *v = C_PT;
            }
        }
    }
    cf
}

// ─── Aggressive second shot ───────────────────────────────────────────────────

/// `hypre_BoomerAMGCreate2ndS` with `num_paths = 1` (serial): the coarse-point
/// strength graph with one- and two-hop connections through C-points
/// (`par_strength.c:2380-2456`, "S*S+2S" comment at `par_strength.c:1790`).
/// Returns `None` when there are no coarse points.
pub fn create_second_strength<T: ComplexScalar>(
    s:  &CsrMatrix<T>,
    cf: &[i32],
) -> Option<CsrMatrix<T>> {
    let n  = s.nrows();
    let rp = s.row_ptr();
    let ci = s.col_idx();

    let mut fine_to_coarse = vec![usize::MAX; n];
    let mut coarse_to_fine = Vec::new();
    for (i, &m) in cf.iter().enumerate() {
        if m > 0 {
            fine_to_coarse[i] = coarse_to_fine.len();
            coarse_to_fine.push(i);
        }
    }
    let nc = coarse_to_fine.len();
    if nc == 0 {
        return None;
    }

    let rows: Vec<Vec<(usize, T)>> = (0..nc)
        .map(|ic| {
            let i1 = coarse_to_fine[ic];
            let mut marker = vec![false; nc];
            let mut row = Vec::new();
            // 1-hop: C-points in S(i1); 2-hop: C-points in S(j), j ∈ S(i1),
            // excluding the row's own coarse index (par_strength.c:2402-2421).
            for k1 in rp[i1]..rp[i1 + 1] {
                let i2 = ci[k1];
                if cf[i2] > 0 {
                    let index = fine_to_coarse[i2];
                    if !marker[index] {
                        marker[index] = true;
                        row.push((index, T::one()));
                    }
                }
                for k2 in rp[i2]..rp[i2 + 1] {
                    let i3 = ci[k2];
                    if cf[i3] > 0 {
                        let index = fine_to_coarse[i3];
                        if index != ic && !marker[index] {
                            marker[index] = true;
                            row.push((index, T::one()));
                        }
                    }
                }
            }
            row
        })
        .collect();

    Some(pack_rows(nc, nc, rows))
}

/// `hypre_BoomerAMGCorrectCFMarker` (`par_strength.c:3064-3088`): merge the
/// second-pass verdicts (`new_cf`, indexed over the first-pass C-points in
/// order) back into the full marker.
/// Galerkin operator `Pᵀ·A·P` accumulated in hypre's exact triple-loop order.
///
/// hypre `hypre_BoomerAMGBuildCoarseOperatorKT` (parcsr_ls/par_rap.c) treats
/// the first argument as the restriction RT and walks, per coarse row `ic`,
/// the entries of RT's own row `i1` (the implicit transpose), then A's row
/// `i1`, then P's row `i2`, folding each `v1·va·vp` into a marker-SPA slot in
/// exactly that traversal order.  The multiplication ORDER fixes the float
/// rounding of the coarse operator — and hypre's HMIS coarsening flips
/// near-tie strength decisions on those last-ulp differences (measured d110a:
/// a last-ulp-different `A_Pi` moved 10 of 20 level-0 C-points and one PCG
/// iteration), so the user-Pi path must reproduce it rather than use the
/// algebraically equivalent `(PᵀA)·P` matmat chain.  Row-major emission
/// (sorted columns) is ulp-irrelevant: hypre's first-touch column order only
/// changes non-discrete matvec summations.
pub fn rap_hypre_order<T: ComplexScalar>(p: &CsrMatrix<T>, a: &CsrMatrix<T>) -> CsrMatrix<T> {
    use std::collections::HashMap;
    // R_diag = Pᵀ (hypre transposes RT internally when keepTranspose = 0;
    // the transpose's row entries come out in ascending column order).
    let pt = p.transpose_csr();
    let n = pt.nrows();
    let rt_rp = pt.row_ptr();
    let rt_ci = pt.col_idx();
    let rt_va = pt.values();
    let a_rp = a.row_ptr();
    let a_ci = a.col_idx();
    let a_va = a.values();
    let p_rp = p.row_ptr();
    let p_ci = p.col_idx();
    let p_va = p.values();
    let mut coo = CooMatrix::new(n, n);
    let mut ra_i: Vec<usize> = Vec::new();
    let mut ra_v: Vec<T> = Vec::new();
    let mut ra_seen: HashMap<usize, usize> = HashMap::new();
    let mut rap_i: Vec<usize> = Vec::new();
    let mut rap_v: Vec<T> = Vec::new();
    let mut rap_seen: HashMap<usize, usize> = HashMap::new();
    for ic in 0..n {
        // Stage 1 (par_rap.c "compute row ic of RA"): RA[ic, i2] =
        // Σ_{(i1,v1)∈R_row(ic)} v1·A[i1, i2] — create-or-add, so the RA
        // columns are in first-touch order and each entry accumulates in
        // ascending i1 order.
        ra_i.clear();
        ra_v.clear();
        ra_seen.clear();
        for jj1 in rt_rp[ic]..rt_rp[ic + 1] {
            let i1 = rt_ci[jj1];
            let v1 = rt_va[jj1];
            for jj2 in a_rp[i1]..a_rp[i1 + 1] {
                let i2 = a_ci[jj2];
                match ra_seen.get(&i2) {
                    Some(&idx) => ra_v[idx] += v1 * a_va[jj2],
                    None => {
                        ra_seen.insert(i2, ra_i.len());
                        ra_i.push(i2);
                        ra_v.push(v1 * a_va[jj2]);
                    }
                }
            }
        }
        // Stage 2: RAP[ic, k] = Σ_{i2∈RA_row(ic)} RA[ic, i2]·P[i2, k] — the
        // RA entries are consumed in first-touch (not sorted) order, which is
        // part of the float accumulation order.
        rap_i.clear();
        rap_v.clear();
        rap_seen.clear();
        for (i2, &ra) in ra_i.iter().zip(ra_v.iter()) {
            for jj3 in p_rp[*i2]..p_rp[*i2 + 1] {
                let k = p_ci[jj3];
                match rap_seen.get(&k) {
                    Some(&idx) => rap_v[idx] += ra * p_va[jj3],
                    None => {
                        rap_seen.insert(k, rap_i.len());
                        rap_i.push(k);
                        rap_v.push(ra * p_va[jj3]);
                    }
                }
            }
        }
        for (k, v) in rap_i.iter().zip(rap_v.iter()) {
            coo.push(ic, *k, *v);
        }
    }
    CsrMatrix::from_coo(&coo)
}

pub fn correct_cf_marker(cf: &mut [i32], new_cf: &[i32]) {
    let mut cnt = 0;
    for v in cf.iter_mut() {
        if *v > 0 {
            if *v == 1 {
                *v = new_cf[cnt];
            } else {
                *v = C_PT;
            }
            cnt += 1;
        }
    }
}

// ─── Multipass interpolation (aggressive level) ───────────────────────────────

/// Multipass interpolation (`hypre_BoomerAMGBuildMultipass`, serial,
/// `weight_option = 0`, no truncation).  C rows are unit rows; F rows are
/// built pass by pass; `Z_PT`/`SF_PT` rows stay empty.
pub(crate) fn interp_multipass<T: ComplexScalar>(
    a:  &CsrMatrix<T>,
    cf: &[i32],
    s:  &CsrMatrix<T>,
) -> CsrMatrix<T> {
    let n   = a.nrows();
    let rp  = a.row_ptr();
    let ci  = a.col_idx();
    let vs  = a.values();
    let srp = s.row_ptr();
    let sci = s.col_idx();

    // C-point enumeration in index order (par_multi_interp.c:339-357).
    let mut fine_to_coarse = vec![usize::MAX; n];
    let mut coarse_to_fine = Vec::new();
    for (i, &m) in cf.iter().enumerate() {
        if m == C_PT {
            fine_to_coarse[i] = coarse_to_fine.len();
            coarse_to_fine.push(i);
        }
    }
    let nc = coarse_to_fine.len();

    // Pass assignment (par_multi_interp.c:434-553).  C → 0; F → 1, 2, …;
    // everything else (Z/SF) is never assigned and keeps an empty P row.
    let mut assigned = vec![-1_i32; n];
    // per pass (index 1..): per point: coarse columns of the interpolatory set.
    let mut pass_rows: Vec<Vec<Vec<usize>>> =
        vec![vec![Vec::new(); n]; MAX_NUM_PASSES + 1];

    // Pass 1: F points with a strong C neighbour.
    for i in 0..n {
        if cf[i] == F_PT {
            let mut row: Vec<usize> = Vec::new();
            for k in srp[i]..srp[i + 1] {
                let j = sci[k];
                if cf[j] == C_PT {
                    row.push(fine_to_coarse[j]);
                }
            }
            if !row.is_empty() {
                assigned[i] = 1;
                pass_rows[1][i] = row;
            }
        }
    }

    // Passes 2..MAX_NUM_PASSES (par_multi_interp.c:524-553): an F point joins
    // pass k when it has a strong neighbour of pass k−1.  Once a pass adds
    // nobody, no later pass can (monotone), which is equivalent to hypre's
    // `while (global_pass_array_size && pass < max_num_passes)`.
    let mut pass = 2;
    loop {
        if pass > MAX_NUM_PASSES {
            break;
        }
        // Clone of the previous pass' rows (kept in `pass_rows` for the
        // weight phase below); small per-level sizes make this cheap.
        let prev = pass_rows[pass - 1].clone();
        let mut any_new = false;
        for i in 0..n {
            if cf[i] == F_PT && assigned[i] < 0 {
                let mut row: Vec<usize> = Vec::new();
                let mut marker = vec![false; nc];
                for k in srp[i]..srp[i + 1] {
                    let j = sci[k];
                    if assigned[j] == pass as i32 - 1 {
                        for &c in &prev[j] {
                            if !marker[c] {
                                marker[c] = true;
                                row.push(c);
                            }
                        }
                    }
                }
                if !row.is_empty() {
                    assigned[i] = pass as i32;
                    pass_rows[pass][i] = row;
                    any_new = true;
                }
            }
        }
        if !any_new {
            break;
        }
        pass += 1;
    }

    // Weights (par_multi_interp.c:1620-1810, weight_option = 0).  Passes are
    // processed in pass order (pass-k rows read the FINISHED P rows of pass
    // k−1), as in hypre.  C rows are unit rows and are never read by the
    // accumulation (assigned[C] = 0 ≠ pass−1 ≥ 1).
    let max_pass = (0..n).filter(|&i| assigned[i] > 0).map(|i| assigned[i]).max().unwrap_or(0);
    let mut out_rows: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
    for i in 0..n {
        if cf[i] == C_PT {
            out_rows[i].push((fine_to_coarse[i], T::one()));
        }
    }
    for my_pass in 1..=max_pass {
        for i in 0..n {
            if cf[i] != F_PT || assigned[i] != my_pass {
                continue;
            }
            let cols = &pass_rows[my_pass as usize][i];
            let mut w = vec![T::zero(); cols.len()];
            let mut col_pos = vec![None; nc];
            for (pos, &c) in cols.iter().enumerate() {
                col_pos[c] = Some(pos);
            }
            let diagonal = diag_value(rp, ci, vs, i);
            let mut sum_c = T::Real::zero();
            let mut sum_n = T::Real::zero();
            if my_pass == 1 {
                // Raw a-values of the (strong C) interpolatory columns;
                // sum_N over every off-diagonal column that is not SF.
                for (pos, &c) in cols.iter().enumerate() {
                    let a_ij = a_value(rp, ci, vs, i, coarse_to_fine[c]);
                    w[pos] = a_ij;
                    sum_c += a_ij.real();
                }
                for k in rp[i]..rp[i + 1] {
                    let j = ci[k];
                    if j == i || cf[j] == SF_PT {
                        continue;
                    }
                    sum_n += vs[k].real();
                }
            } else {
                // Accumulate a_ij · P[j, c] over the STRONG neighbours j
                // assigned at pass k−1 (par_multi_interp.c:1755-1801:
                // `tmp_marker[j1] == i1` marks S-row neighbours only);
                // all other non-SF columns of row i feed sum_N only.
                let mut strong_nb = vec![false; n];
                for k in srp[i]..srp[i + 1] {
                    strong_nb[sci[k]] = true;
                }
                for k in rp[i]..rp[i + 1] {
                    let j = ci[k];
                    if j == i {
                        continue;
                    }
                    if assigned[j] == my_pass as i32 - 1 && strong_nb[j] {
                        let a_ij = vs[k];
                        for &(c, pj) in &out_rows[j] {
                            let alfa = a_ij * pj;
                            w[col_pos[c].expect("neighbour P columns ⊆ row set")] += alfa;
                            sum_c += alfa.real();
                            sum_n += alfa.real();
                        }
                    } else if cf[j] != SF_PT {
                        sum_n += vs[k].real();
                    }
                }
            }

            let denom = sum_c * diagonal.real();
            if denom != T::Real::zero() {
                let alfa = -sum_n / denom;
                let alfa_t = T::from_real(alfa);
                for wv in w.iter_mut() {
                    *wv *= alfa_t;
                }
            }
            for (pos, &c) in cols.iter().enumerate() {
                out_rows[i].push((c, w[pos]));
            }
        }
    }

    pack_rows(n, nc, out_rows)
}

// ─── Extended+i interpolation (interp_type 6) ─────────────────────────────────

/// Extended+i classical interpolation (`hypre_BoomerAMGBuildExtPIInterp`,
/// serial, `par_lr_interp.c:1024-1860`).
pub(crate) fn interp_ext_pi<T: ComplexScalar>(
    a:  &CsrMatrix<T>,
    cf: &[i32],
    s:  &CsrMatrix<T>,
) -> CsrMatrix<T> {
    let n   = a.nrows();
    let rp  = a.row_ptr();
    let ci  = a.col_idx();
    let vs  = a.values();
    let srp = s.row_ptr();
    let sci = s.col_idx();

    let mut fine_to_coarse = vec![usize::MAX; n];
    let mut nc = 0;
    for (i, &m) in cf.iter().enumerate() {
        if m > 0 {
            fine_to_coarse[i] = nc;
            nc += 1;
        }
    }

    let rows: Vec<Vec<(usize, T)>> = (0..n)
        .map(|i| {
            if cf[i] > 0 {
                return vec![(fine_to_coarse[i], T::one())];
            }
            if cf[i] == SF_PT {
                return Vec::new();
            }

            // Interpolatory set (structure pass, par_lr_interp.c:1461-1508
            // structure / 1542-1577 weights): strong C-neighbours of i, plus
            // strong C-neighbours of every non-SF strong neighbour i1
            // (extended+i: no common-C requirement).  `strong_f` marks the
            // neighbours that contributed 2-hop columns; they drive the
            // diagonal-distribution pass (par_lr_interp.c:1555-1699).
            let mut cols: Vec<usize> = Vec::new();
            let mut col_pos = vec![None; nc];
            let mut strong_f = vec![false; n];
            for k1 in srp[i]..srp[i + 1] {
                let i1 = sci[k1];
                if cf[i1] > 0 {
                    if col_pos[fine_to_coarse[i1]].is_none() {
                        col_pos[fine_to_coarse[i1]] = Some(cols.len());
                        cols.push(fine_to_coarse[i1]);
                    }
                } else if cf[i1] != SF_PT {
                    strong_f[i1] = true;
                    for k2 in srp[i1]..srp[i1 + 1] {
                        let k_ = sci[k2];
                        if cf[k_] > 0 && col_pos[fine_to_coarse[k_]].is_none() {
                            col_pos[fine_to_coarse[k_]] = Some(cols.len());
                            cols.push(fine_to_coarse[k_]);
                        }
                    }
                }
            }

            // Weight pass (par_lr_interp.c:1525-1807).
            let mut w = vec![T::zero(); cols.len()];
            let mut diagonal = diag_value(rp, ci, vs, i);
            for k in rp[i]..rp[i + 1] {
                let i1 = ci[k];
                if i1 == i {
                    continue;
                }
                let in_set = cf[i1] > 0 && col_pos[fine_to_coarse[i1]].is_some();
                if in_set {
                    // Interpolatory C column: accumulate a_i,i1.
                    w[col_pos[fine_to_coarse[i1]].unwrap()] += vs[k];
                } else if strong_f[i1] {
                    // Strong F neighbour: distribute a_i,i1 over its
                    // same-sign connections to the interpolatory set / i.
                    let sgn_is_neg = diag_value(rp, ci, vs, i1).real() < T::Real::zero();
                    let mut sum = T::Real::zero();
                    for k1 in rp[i1]..rp[i1 + 1] {
                        let i2 = ci[k1];
                        if i2 == i1 {
                            continue;
                        }
                        let i2_in_set = cf[i2] > 0 && col_pos[fine_to_coarse[i2]].is_some();
                        if i2_in_set || i2 == i {
                            let sgn_v = if sgn_is_neg { -vs[k1].real() } else { vs[k1].real() };
                            if sgn_v < T::Real::zero() {
                                sum += vs[k1].real();
                            }
                        }
                    }
                    if sum != T::Real::zero() {
                        let distribute = vs[k] / T::from_real(sum);
                        for k1 in rp[i1]..rp[i1 + 1] {
                            let i2 = ci[k1];
                            if i2 == i1 {
                                continue;
                            }
                            let sgn_v = if sgn_is_neg { -vs[k1].real() } else { vs[k1].real() };
                            if sgn_v < T::Real::zero() {
                                let i2_in_set =
                                    cf[i2] > 0 && col_pos[fine_to_coarse[i2]].is_some();
                                if i2_in_set {
                                    w[col_pos[fine_to_coarse[i2]].unwrap()] += distribute * vs[k1];
                                }
                                if i2 == i {
                                    diagonal += distribute * vs[k1];
                                }
                            }
                        }
                    } else {
                        diagonal += vs[k];
                    }
                } else if cf[i1] != SF_PT {
                    // Weakly connected (incl. Z columns): into the diagonal.
                    diagonal += vs[k];
                }
            }

            if diagonal != T::zero() {
                let minus_diagonal = T::zero() - diagonal;
                for wv in w.iter_mut() {
                    *wv /= minus_diagonal;
                }
            }
            cols.iter().enumerate().map(|(pos, &c)| (c, w[pos])).collect()
        })
        .collect();

    pack_rows(n, nc, rows)
}

// ─── PMax truncation ──────────────────────────────────────────────────────────

/// Truncate `p` to `AMS_PMAX` entries per row with signed row-sum rescaling
/// (`hypre_ParCSRMatrixTruncate`, rescale = 1, `parcsr_mv/par_csr_matrix.c:2398-2432`).
pub(crate) fn truncate_pmax<T: ComplexScalar>(p: &CsrMatrix<T>) -> CsrMatrix<T> {
    let n  = p.nrows();
    let nc = p.ncols();
    let rp = p.row_ptr();
    let ci = p.col_idx();
    let vs = p.values();

    let rows: Vec<Vec<(usize, T)>> = (0..n)
        .map(|i| {
            let len = rp[i + 1] - rp[i];
            if len <= AMS_PMAX {
                return (rp[i]..rp[i + 1]).map(|k| (ci[k], vs[k])).collect();
            }
            let mut row: Vec<(usize, T)> = (rp[i]..rp[i + 1]).map(|k| (ci[k], vs[k])).collect();
            let row_sum: T::Real = row.iter().fold(T::Real::zero(), |s, (_, v)| s + v.real());
            // Sort by |value| descending (hypre_qsort2_abs).
            row.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            row.truncate(AMS_PMAX);
            let kept: T::Real = row.iter().fold(T::Real::zero(), |s, (_, v)| s + v.real());
            if kept != T::Real::zero() && kept != row_sum {
                let scale = row_sum / kept;
                let scale_t = T::from_real(scale);
                for (_, v) in row.iter_mut() {
                    *v *= scale_t;
                }
            }
            row
        })
        .collect();

    pack_rows(n, nc, rows)
}

// ─── helpers ──────────────────────────────────────────────────────────────────

fn pack_rows<T: ComplexScalar>(nrows: usize, ncols: usize, rows: Vec<Vec<(usize, T)>>) -> CsrMatrix<T> {
    let nnz: usize = rows.iter().map(|r| r.len()).sum();
    let mut row_ptr = vec![0_usize; nrows + 1];
    let mut col_idx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    let mut acc = 0_usize;
    for (i, row) in rows.iter().enumerate() {
        row_ptr[i] = acc;
        acc += row.len();
        for &(j, v) in row {
            col_idx.push(j);
            values.push(v);
        }
    }
    row_ptr[nrows] = acc;
    CsrMatrix::from_raw(nrows, ncols, row_ptr, col_idx, values)
}

fn a_value<T: ComplexScalar>(rp: &[usize], ci: &[usize], vs: &[T], i: usize, j: usize) -> T {
    for k in rp[i]..rp[i + 1] {
        if ci[k] == j {
            return vs[k];
        }
    }
    T::zero()
}

fn diag_value<T: ComplexScalar>(rp: &[usize], ci: &[usize], vs: &[T], i: usize) -> T {
    a_value(rp, ci, vs, i, i)
}
