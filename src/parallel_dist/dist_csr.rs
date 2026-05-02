use crate::core::scalar::Scalar;
use crate::parallel_dist::halo::HaloExchange;
use crate::parallel_dist::layout::{block_partition, PartitionLayout};
use crate::sparse::{CooMatrix, CsrMatrix};
use std::collections::BTreeSet;

/// Distributed CSR scaffold.
///
/// The local matrix uses an augmented column space:
/// - `[0, local_size)` indexes owned unknowns,
/// - `[local_size, local_size + ghost_size)` indexes ghost unknowns in the
///   same order as `layout.ghost_globals`.
#[derive(Debug, Clone)]
pub struct DistCsrMatrix<T: Scalar> {
    local_mat: CsrMatrix<T>,
    layout: PartitionLayout,
}

impl<T: Scalar> DistCsrMatrix<T> {
    /// Build a distributed CSR view for one partition.
    pub fn new(local_mat: CsrMatrix<T>, layout: PartitionLayout) -> Result<Self, String> {
        if local_mat.nrows() != layout.local_size() {
            return Err("local rows must equal owned partition size".into());
        }
        let expected_cols = layout.local_size() + layout.ghost_size();
        if local_mat.ncols() != expected_cols {
            return Err("local cols must equal local_size + ghost_size".into());
        }
        Ok(Self { local_mat, layout })
    }

    pub fn layout(&self) -> &PartitionLayout { &self.layout }

    pub fn local_mat(&self) -> &CsrMatrix<T> { &self.local_mat }

    /// Build one partition from a global CSR matrix using simple block rows.
    ///
    /// This is a scaffolding constructor for early distributed development and
    /// testing. It assumes one unknown per row and therefore requires
    /// `global.nrows() == global.ncols()`.
    pub fn from_global_csr_block_partition(
        global: &CsrMatrix<T>,
        nranks: usize,
        rank: usize,
    ) -> Result<Self, String> {
        if global.nrows() != global.ncols() {
            return Err("global matrix must be square for block-partition constructor".into());
        }

        let owned = block_partition(global.nrows(), nranks, rank);
        let local_size = owned.end - owned.start;

        let mut ghost_set = BTreeSet::new();
        for i in owned.clone() {
            let rs = global.row_ptr()[i];
            let re = global.row_ptr()[i + 1];
            for &gcol in &global.col_idx()[rs..re] {
                if gcol < owned.start || gcol >= owned.end {
                    ghost_set.insert(gcol);
                }
            }
        }
        let ghost_globals: Vec<usize> = ghost_set.into_iter().collect();

        let layout = PartitionLayout::new(global.nrows(), owned.clone(), ghost_globals.clone())?;

        let mut coo = CooMatrix::new(local_size, local_size + ghost_globals.len());
        for gi in owned.clone() {
            let li = gi - owned.start;
            let rs = global.row_ptr()[gi];
            let re = global.row_ptr()[gi + 1];
            for k in rs..re {
                let gcol = global.col_idx()[k];
                let lcol = if gcol >= owned.start && gcol < owned.end {
                    gcol - owned.start
                } else {
                    let pos = ghost_globals
                        .binary_search(&gcol)
                        .map_err(|_| "internal ghost mapping error")?;
                    local_size + pos
                };
                coo.push(li, lcol, global.values()[k]);
            }
        }

        let local_mat = CsrMatrix::from_coo(&coo);
        Self::new(local_mat, layout)
    }

    /// Local SpMV using owned + already-fetched ghost values.
    pub fn spmv_local(&self, x_owned: &[T], x_ghost: &[T], y_owned: &mut [T]) {
        let n_local = self.layout.local_size();
        debug_assert_eq!(x_owned.len(), n_local);
        debug_assert_eq!(x_ghost.len(), self.layout.ghost_size());
        debug_assert_eq!(y_owned.len(), n_local);

        let mut x = vec![T::zero(); n_local + x_ghost.len()];
        x[..n_local].copy_from_slice(x_owned);
        x[n_local..].copy_from_slice(x_ghost);
        self.local_mat.spmv(&x, y_owned);
    }

    /// SpMV with halo requests through an exchange backend.
    ///
    /// Calls [`HaloExchange::exchange_with_owned`], passing the owned values
    /// so that MPI backends can send them to neighbours.  Single-process
    /// backends (e.g. [`LocalHaloExchange`]) ignore the owned values and look
    /// up ghost values from their local copy of the global vector.
    ///
    /// [`LocalHaloExchange`]: crate::parallel_dist::halo::LocalHaloExchange
    pub fn spmv_with_halo<E: HaloExchange<T>>(
        &self,
        x_owned: &[T],
        halo: &E,
        y_owned: &mut [T],
    ) -> Result<(), crate::parallel_dist::halo::HaloError> {
        let mut x_ghost = vec![T::zero(); self.layout.ghost_size()];
        halo.exchange_with_owned(
            self.layout.owned_global_range.start,
            x_owned,
            &self.layout.ghost_globals,
            &mut x_ghost,
        )?;
        self.spmv_local(x_owned, &x_ghost, y_owned);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel_dist::halo::LocalHaloExchange;
    use crate::sparse::CooMatrix;

    #[test]
    fn dist_spmv_local_matches_manual_expectation() {
        // Partition owns global 1..3 (two rows), ghosts are [0, 3].
        // local column map: 0->g1, 1->g2, 2->g0, 3->g3
        let layout = PartitionLayout::new(4, 1..3, vec![0, 3]).unwrap();

        // Local rows represent a 1D Laplacian fragment:
        // row(g1): -x0 + 2x1 - x2
        // row(g2): -x1 + 2x2 - x3
        let mut coo = CooMatrix::new(2, 4);
        coo.push(0, 2, -1.0_f64); // g0 ghost
        coo.push(0, 0,  2.0);     // g1 local
        coo.push(0, 1, -1.0);     // g2 local
        coo.push(1, 0, -1.0);     // g1 local
        coo.push(1, 1,  2.0);     // g2 local
        coo.push(1, 3, -1.0);     // g3 ghost
        let local = CsrMatrix::from_coo(&coo);

        let dist = DistCsrMatrix::new(local, layout).unwrap();

        let x_owned = vec![20.0_f64, 30.0]; // g1, g2
        let halo = LocalHaloExchange::new(vec![10.0_f64, 20.0, 30.0, 40.0]); // g0..g3
        let mut y = vec![0.0_f64; 2];

        dist.spmv_with_halo(&x_owned, &halo, &mut y).unwrap();
        assert_eq!(y, vec![0.0_f64, 0.0_f64]);
    }

    #[test]
    fn dist_spmv_two_partitions_matches_global_spmv() {
        let n = 8usize;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0_f64);
            if i > 0 { coo.push(i, i - 1, -1.0); }
            if i + 1 < n { coo.push(i, i + 1, -1.0); }
        }
        let global = CsrMatrix::from_coo(&coo);

        let d0 = DistCsrMatrix::from_global_csr_block_partition(&global, 2, 0).unwrap();
        let d1 = DistCsrMatrix::from_global_csr_block_partition(&global, 2, 1).unwrap();

        let xg: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let halo = LocalHaloExchange::new(xg.clone());

        let x0 = &xg[d0.layout().owned_global_range.clone()];
        let x1 = &xg[d1.layout().owned_global_range.clone()];

        let mut y0 = vec![0.0_f64; d0.layout().local_size()];
        let mut y1 = vec![0.0_f64; d1.layout().local_size()];
        d0.spmv_with_halo(x0, &halo, &mut y0).unwrap();
        d1.spmv_with_halo(x1, &halo, &mut y1).unwrap();

        let mut yd = Vec::with_capacity(n);
        yd.extend_from_slice(&y0);
        yd.extend_from_slice(&y1);

        let mut yg = vec![0.0_f64; n];
        global.spmv(&xg, &mut yg);

        assert_eq!(yd.len(), yg.len());
        for (a, b) in yd.iter().zip(yg.iter()) {
            assert!((*a - *b).abs() < 1e-12, "distributed vs global mismatch: {a} vs {b}");
        }
    }
}
