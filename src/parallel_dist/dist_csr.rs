use crate::core::scalar::Scalar;
use crate::parallel_dist::halo::HaloExchange;
use crate::parallel_dist::layout::PartitionLayout;
use crate::sparse::CsrMatrix;

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
    pub fn spmv_with_halo<E: HaloExchange<T>>(
        &self,
        x_owned: &[T],
        halo: &E,
        y_owned: &mut [T],
    ) -> Result<(), crate::parallel_dist::halo::HaloError> {
        let mut x_ghost = vec![T::zero(); self.layout.ghost_size()];
        halo.exchange(&self.layout.ghost_globals, &mut x_ghost)?;
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
}
