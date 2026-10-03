//! D1000 round-107 pins: AMG quality fixes in the SA setup phase.
//!
//! 1. Chebyshev smoother recursion (simd/smoother.rs): the direction update
//!    must realise the exact normalized-Chebyshev polynomial.  Pinned through
//!    the two-grid cycle spectrum in `crates/amg` (`amg_cg_chebyshev_smoother`);
//!    here we pin the polynomial itself against hypre `par_cheby.c` explicit
//!    standard coefficients on a diagonal operator.
//! 2. SA aggregation singleton absorption (coarsen_agg.rs): aggregates of
//!    size 1 whose node has an off-diagonal raw connection must be absorbed
//!    (hypre aggregation semantics), because their smoothed P₀ column
//!    collapses on diagonally dominant rows and the Galerkin coarse operator
//!    goes numerically rank-deficient (the D1000-LOR −4.12 root cause).
#![allow(clippy::needless_range_loop)]

mod common;

use linlvo::amg::coarsen_agg::build_aggregates;
use linlvo::amg::smoother::gershgorin_upper_scaled;
use linlvo::amg::{AmgConfig, AmgHierarchy, SmootherType};
use linlvo::simd::smoother::chebyshev_smooth;
use linlvo::sparse::{CooMatrix, CsrMatrix};
use linlvo::{DenseVec, LinearOperator};

fn poisson_1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0 { coo.push(i, i - 1, -1.0); }
        if i + 1 < n { coo.push(i, i + 1, -1.0); }
    }
    CsrMatrix::from_coo(&coo)
}

fn csr_from_triplets(n: usize, triplets: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, n);
    for &(i, j, v) in triplets {
        coo.push(i, j, v);
    }
    CsrMatrix::from_coo(&coo)
}

/// hypre `par_cheby.c` variant-0 (standard Chebyshev) explicit coefficients
/// for relax order `order` on the interval `[lower, upper]` — the truth
/// source the corrected recursion must reproduce.
fn hypre_standard_coefs(lower: f64, upper: f64, order: usize) -> Vec<f64> {
    let theta = (upper + lower) / 2.0;
    let delta = (upper - lower) / 2.0;
    let cheby_order = order - 1;
    let mut coefs = vec![0.0_f64; order];
    match cheby_order {
        0 => coefs[0] = 1.0 / theta,
        1 => {
            let den = delta * delta - 2.0 * theta * theta;
            coefs[0] = -4.0 * theta / den;
            coefs[1] = 2.0 / den;
        }
        2 => {
            let den = 3.0 * delta * delta * theta - 4.0 * theta.powi(3);
            coefs[0] = (3.0 * delta * delta - 12.0 * theta * theta) / den;
            coefs[1] = 12.0 * theta / den;
            coefs[2] = -4.0 / den;
        }
        _ => {
            let den = delta.powi(4) - 8.0 * delta * delta * theta * theta + 8.0 * theta.powi(4);
            coefs[0] = (32.0 * theta.powi(3) - 16.0 * delta * delta * theta) / den;
            coefs[1] = (8.0 * delta * delta - 48.0 * theta * theta) / den;
            coefs[2] = 32.0 * theta / den;
            coefs[3] = -8.0 / den;
        }
    }
    coefs
}

/// D1000 pin 1: the smoother realizes `Δx = s(D⁻¹A)·D⁻¹(b − A x)`; with a
/// unit-diagonal operator the polynomial acts on A itself, so driving one
/// call from x = 0 with an eigenvector b extracts `s(t)` directly.  The
/// realized `s` must equal the hypre `par_cheby.c` explicit standard
/// polynomial (the pre-fix ρ(ρ−1) recursion realized a different,
/// non-damping polynomial — `|R(λmax)| ≈ 1` instead of `1/T_d(τ₀)`).
#[test]
fn chebyshev_recursion_matches_hypre_explicit_coefs() {
    // A = I − 0.5·(1-D Laplacian): unit diagonal, eigenvalues
    // t_k = 1 − cos(kπ/(n+1)) with sine eigenvectors.
    let n = 8usize;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 1.0);
        if i > 0 { coo.push(i, i - 1, -0.5); }
        if i + 1 < n { coo.push(i, i + 1, -0.5); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let t = |k: usize| 1.0 - ((k as f64) * std::f64::consts::PI / (n as f64 + 1.0)).cos();

    for degree in [2usize, 3, 4] {
        let lower = 0.5_f64;
        let upper = 4.0_f64;
        let coefs = hypre_standard_coefs(lower, upper, degree);
        for k in 1..=n {
            // normalized sine eigenvector for t_k
            let mut b = DenseVec::zeros(n);
            let mut qnorm = 0.0_f64;
            for i in 0..n {
                let v = ((i as f64 + 1.0) * (k as f64) * std::f64::consts::PI / (n as f64 + 1.0)).sin();
                b.as_mut_slice()[i] = v;
                qnorm += v * v;
            }
            let qn = qnorm.sqrt();
            for v in b.as_mut_slice().iter_mut() { *v /= qn; }
            let mut x = DenseVec::zeros(n);
            chebyshev_smooth(&a, &mut x, &b, lower, upper, degree);
            let tk = t(k);
            let got = x.as_slice()[0] / b.as_slice()[0];
            let want: f64 = coefs.iter().enumerate().map(|(j, &c)| c * tk.powi(j as i32)).sum();
            assert!(
                (got - want).abs() < 5e-14 * (1.0 + want.abs()),
                "degree {degree}: s({tk:.6}) = {got:.17e} vs hypre explicit {want:.17e}"
            );
        }
    }
}

/// D1000 pin 2: aggregates with a single member whose node has any
/// off-diagonal raw connection must not exist post-fix.
#[test]
fn aggregation_absorbs_singletons_with_raw_connections() {
    // Asymmetric strength graph: row 0's off-diagonal mass is dominated by the
    // far entry a_05 = 1.0, so near entries are not "strong" for row 0 while
    // row 1's only off-diagonal neighbour is node 0.  The greedy pass then
    // seeds singleton aggregates on weakly-connected nodes that DO have raw
    // off-diagonal connections.
    let n = 8;
    let a = csr_from_triplets(
        n,
        &[
            (0, 0, 4.0),
            (0, 5, 1.0), // dominates row 0's off-diagonal max
            (0, 1, 0.1),
            (1, 1, 4.0),
            (1, 0, 0.1),
            (1, 2, 0.2),
            (2, 2, 4.0),
            (2, 1, 0.2),
            (2, 3, 2.0),
            (3, 3, 4.0),
            (3, 2, 2.0),
            (3, 4, 2.0),
            (4, 4, 4.0),
            (4, 3, 2.0),
            (4, 5, 2.0),
            (5, 5, 4.0),
            (5, 4, 2.0),
            (5, 0, 1.0),
            (6, 6, 4.0),
            (6, 7, 3.0),
            (7, 7, 4.0),
            (7, 6, 3.0),
        ],
    );
    let s = linlvo::amg::strength::strong_connections(&a, 0.25);
    let agg_id = build_aggregates::<f64>(&a, &s);
    let n_agg = agg_id.iter().copied().max().map(|m| m + 1).unwrap_or(0);

    // No singleton aggregate whose member has an off-diagonal raw connection.
    let mut sizes = vec![0usize; n_agg];
    for &id in agg_id.iter() {
        sizes[id] += 1;
    }
    let rp = a.row_ptr();
    let ci = a.col_idx();
    for i in 0..n {
        if sizes[agg_id[i]] == 1 {
            let has_raw_conn = (rp[i]..rp[i + 1]).any(|k| ci[k] != i);
            assert!(
                !has_raw_conn,
                "node {i} is a singleton aggregate but has raw off-diagonal connections"
            );
        }
    }
    // Every node still assigned; ids compact.
    assert!(agg_id.iter().all(|&id| id != usize::MAX));
    let used: std::collections::HashSet<usize> = agg_id.iter().copied().collect();
    assert_eq!(used.len(), n_agg, "aggregate ids must be compact");
}

/// D1000 pin 3 (the pathology end-to-end at the hierarchy level): a
/// diagonally dominant grid operator whose strength graph fragments must
/// produce a hierarchy whose Galerkin coarse operators stay SPD and whose
/// V-cycle preconditioner stays a contraction.  Before the absorption fix a
/// collapsed P column made A_c numerically singular (λmin ~ 1e-6 two levels
/// below the collapse) and the cycle operator indefinite.
#[test]
fn sa_hierarchy_with_weak_rows_stays_spd() {
    // Diagonally dominant 2-D-ish operator: strong diagonal (4), weak cross
    // links (0.05) plus a few strong links — the greedy fragments the weak
    // region into singletons pre-fix.
    let nx = 24usize;
    let n = nx * nx;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for j in 0..nx {
        for i in 0..nx {
            let r = j * nx + i;
            coo.push(r, r, 4.0);
            let w = if (i / 4) % 2 == 0 { 1.0 } else { 0.05 };
            if i > 0 { coo.push(r, r - 1, -w); }
            if i + 1 < nx { coo.push(r, r + 1, -w); }
            if j > 0 { coo.push(r, r - nx, -1.0); }
            if j + 1 < nx { coo.push(r, r + nx, -1.0); }
        }
    }
    let a = CsrMatrix::from_coo(&coo);

    let cfg = AmgConfig {
        smoother: SmootherType::WeightedJacobi { omega: 0.667 },
        ..AmgConfig::default()
    };
    let hier = AmgHierarchy::build(a.clone(), cfg);

    // Per-level SPD-ness of the Galerkin coarse operators (power iteration on
    // cI − A_l; c = 2·λmax + 1).  λmin must stay clearly positive.
    use linlvo::core::operator::LinearOperator;
    for (l, lev) in hier.levels.iter().enumerate() {
        let nl = lev.a.nrows();
        let la = &lev.a;
        let mut v: Vec<f64> = vec![1.0 / (nl as f64).sqrt(); nl];
        let mut lam_max = 0.0_f64;
        for _ in 0..60 {
            let mut w = DenseVec::zeros(nl);
            la.apply(&DenseVec::from_vec(v.clone()), &mut w);
            let ws = w.as_slice();
            let nw: f64 = ws.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nw < 1e-300 { break; }
            lam_max = nw;
            v.copy_from_slice(ws);
            for x in v.iter_mut() { *x /= nw; }
        }
        let c = 2.0 * lam_max + 1.0;
        let mut u: Vec<f64> = (0..nl).map(|i| ((i * 7919 + 17) % 101) as f64 / 101.0 - 0.5).collect();
        let nu: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in u.iter_mut() { *x /= nu; }
        let mut mu_max = 0.0_f64;
        for _ in 0..120 {
            let mut w = DenseVec::zeros(nl);
            la.apply(&DenseVec::from_vec(u.clone()), &mut w);
            let ws = w.as_slice();
            let nw: f64 = ws.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nw < 1e-300 { break; }
            mu_max = nw;
            for i in 0..nl { u[i] = (c * u[i] - ws[i]) / nw; }
        }
        let lam_min = c - mu_max;
        println!("level {l}: n={nl} λmin={lam_min:.3e} λmax={lam_max:.3e}");
        assert!(
            lam_min > 1e-10 * lam_max,
            "D1000: Galerkin operator at level {l} numerically singular: λmin={lam_min:.3e}"
        );
    }

    // V-cycle operator SPD-ness (columns of the cycle applied at level 0).
    let la0 = hier.levels[0].a.clone();
    let n0 = la0.nrows();
    let mut b = DenseVec::zeros(n0);
    let mut x = DenseVec::zeros(n0);
    // Rayleigh probes: ⟨M⁻¹r, r⟩ > 0 for a set of probe vectors.
    let probes: Vec<Vec<f64>> = vec![
        vec![1.0; n0],
        (0..n0).map(|i| ((i % 7) as f64) - 3.0).collect(),
        (0..n0).map(|i| ((i * 37 % 101) as f64) / 101.0 - 0.5).collect(),
    ];
    for p in &probes {
        b.as_mut_slice().copy_from_slice(p);
        x.as_mut_slice().fill(0.0);
        hier.apply_cycle(&b, &mut x, linlvo::amg::CycleType::V);
        // ⟨M⁻¹p, p⟩ with the cycle's own residual bookkeeping is not directly
        // the energy; measure via the rate + positivity of the incremental
        // energy ⟨x, p⟩ instead: x = M⁻¹ p ⇒ ⟨M⁻¹p, p⟩ = ⟨x, p⟩.
        let energy: f64 = x
            .as_slice()
            .iter()
            .zip(p.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(energy > 0.0, "D1000: V-cycle not PD: ⟨M⁻¹p, p⟩ = {energy:.3e}");
    }
    let _ = gershgorin_upper_scaled(&la0);
}
