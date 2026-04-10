use crate::core::scalar::Scalar;
use std::collections::HashSet;

/// Errors from halo exchange operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum HaloError {
    /// Requested global index does not exist in exchange source.
    #[error("missing halo value for global index {global_index}")]
    MissingValue { global_index: usize },
    /// Output buffer length mismatch.
    #[error("output length mismatch: expected {expected}, got {actual}")]
    OutputLenMismatch { expected: usize, actual: usize },
}

/// Abstract halo exchange interface.
///
/// A production implementation can route requests to MPI neighborhood
/// communication while keeping solver-side code unchanged.
pub trait HaloExchange<T: Scalar>: Send + Sync {
    /// Fill `out` with values corresponding to `global_indices`.
    fn exchange(&self, global_indices: &[usize], out: &mut [T]) -> Result<(), HaloError>;
}

/// Halo request/response plan for one neighboring rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighborHaloPlan {
    /// Neighbor rank id.
    pub neighbor_rank: usize,
    /// Global indices to send to this neighbor.
    pub send_globals: Vec<usize>,
    /// Global indices to receive from this neighbor.
    pub recv_globals: Vec<usize>,
}

/// Full halo communication plan for one rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HaloPlan {
    /// Neighbor plans in arbitrary order.
    pub neighbors: Vec<NeighborHaloPlan>,
}

impl HaloPlan {
    pub fn new(neighbors: Vec<NeighborHaloPlan>) -> Result<Self, String> {
        let mut seen = HashSet::new();
        for n in &neighbors {
            if !seen.insert(n.neighbor_rank) {
                return Err("duplicate neighbor rank in halo plan".into());
            }
        }
        Ok(Self { neighbors })
    }

    /// Total number of halo values received by this rank.
    pub fn total_recv_len(&self) -> usize {
        self.neighbors.iter().map(|n| n.recv_globals.len()).sum()
    }

    /// Total number of halo values sent by this rank.
    pub fn total_send_len(&self) -> usize {
        self.neighbors.iter().map(|n| n.send_globals.len()).sum()
    }
}

/// Single-process halo exchange adapter used for testing and scaffolding.
#[derive(Debug, Clone)]
pub struct LocalHaloExchange<T: Scalar> {
    global_values: Vec<T>,
}

impl<T: Scalar> LocalHaloExchange<T> {
    pub fn new(global_values: Vec<T>) -> Self {
        Self { global_values }
    }
}

impl<T: Scalar> HaloExchange<T> for LocalHaloExchange<T> {
    fn exchange(&self, global_indices: &[usize], out: &mut [T]) -> Result<(), HaloError> {
        if out.len() != global_indices.len() {
            return Err(HaloError::OutputLenMismatch {
                expected: global_indices.len(),
                actual: out.len(),
            });
        }
        for (i, &gidx) in global_indices.iter().enumerate() {
            let Some(&v) = self.global_values.get(gidx) else {
                return Err(HaloError::MissingValue { global_index: gidx });
            };
            out[i] = v;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_halo_exchange_returns_requested_values() {
        let h = LocalHaloExchange::new(vec![10.0_f64, 20.0, 30.0, 40.0]);
        let mut out = vec![0.0_f64; 2];
        h.exchange(&[3, 1], &mut out).unwrap();
        assert_eq!(out, vec![40.0, 20.0]);
    }

    #[test]
    fn halo_plan_rejects_duplicate_neighbors() {
        let plan = HaloPlan::new(vec![
            NeighborHaloPlan {
                neighbor_rank: 1,
                send_globals: vec![0],
                recv_globals: vec![4],
            },
            NeighborHaloPlan {
                neighbor_rank: 1,
                send_globals: vec![2],
                recv_globals: vec![6],
            },
        ]);
        assert!(plan.is_err());
    }
}
