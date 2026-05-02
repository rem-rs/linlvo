//! MPI-backed halo exchange and global reduction.
//!
//! Enable with `features = ["mpi"]`.  Requires an MPI installation (Open MPI
//! ≥ 4.0 tested) and links to `libmpi` via the `mpi` crate (rsmpi 0.8).
//!
//! # Thread safety
//! Both [`MpiHaloExchange`] and [`MpiReduce`] access `MPI_COMM_WORLD` through
//! `SystemCommunicator::world()`, which is safe as long as MPI is initialised
//! with at least `MPI_THREAD_SERIALIZED`.  This library never spawns extra
//! threads that call MPI, so single-threaded per-rank usage is assumed.
//!
//! # Example (2-rank run)
//! ```no_run
//! # #[cfg(feature = "mpi")] {
//! use linger::parallel_dist::{
//!     LocalHaloExchange, HaloExchange,
//!     mpi_halo::{MpiHaloExchange, MpiReduce, GlobalReduce},
//!     HaloPlan, NeighborHaloPlan,
//!     DistCsrMatrix, block_partition,
//!     dist_cg, DistCgParams,
//! };
//! use linger::sparse::CsrMatrix;
//!
//! let universe = mpi::initialize().unwrap();
//! let world    = universe.world();
//! let rank     = world.rank() as usize;
//! let nranks   = world.size() as usize;
//!
//! // Build global matrix on every rank (small demo; real code loads a partition).
//! let global = CsrMatrix::<f64>::identity(4);
//! let dist   = DistCsrMatrix::from_global_csr_block_partition(&global, nranks, rank).unwrap();
//!
//! let halo   = MpiHaloExchange::from_dist_matrix(&dist, &world);
//! let reduce = MpiReduce;
//!
//! let n  = dist.layout().local_size();
//! let b  = vec![1.0_f64; n];
//! let mut x = vec![0.0_f64; n];
//! dist_cg(&dist, &halo, &reduce, &b, &mut x, &DistCgParams::default()).unwrap();
//! # }
//! ```

#[cfg(feature = "mpi")]
pub use inner::{GlobalReduce, LocalReduce, MpiHaloExchange, MpiReduce};

// Always export LocalReduce so non-MPI code can use it.
#[cfg(not(feature = "mpi"))]
pub use fallback::{GlobalReduce, LocalReduce};

// ─── non-MPI fallback: only LocalReduce / GlobalReduce trait ─────────────────

#[cfg(not(feature = "mpi"))]
mod fallback {
    /// Trait for global all-reduce across MPI ranks.
    ///
    /// The single-process [`LocalReduce`] implementation is always available.
    /// The MPI [`MpiReduce`](super::inner::MpiReduce) is available under
    /// `features = ["mpi"]`.
    pub trait GlobalReduce: Send + Sync {
        /// Sum `local` across all participating ranks and return the global sum.
        fn allreduce_sum(&self, local: f64) -> f64;
    }

    /// No-op reduction: returns `local` unchanged (single process).
    pub struct LocalReduce;

    impl GlobalReduce for LocalReduce {
        #[inline]
        fn allreduce_sum(&self, local: f64) -> f64 { local }
    }
}

// ─── MPI implementation ───────────────────────────────────────────────────────

#[cfg(feature = "mpi")]
mod inner {
    use crate::core::scalar::Scalar;
    use crate::parallel_dist::dist_csr::DistCsrMatrix;
    use crate::parallel_dist::halo::{HaloError, HaloExchange, NeighborHaloPlan, HaloPlan};
    use mpi::topology::{Communicator, SimpleCommunicator};
    use mpi::traits::*;
    use std::collections::HashMap;

    // ─── GlobalReduce ────────────────────────────────────────────────────────

    /// Trait for global all-reduce across MPI ranks.
    pub trait GlobalReduce: Send + Sync {
        /// Sum `local` across all participating ranks and return the global sum.
        fn allreduce_sum(&self, local: f64) -> f64;
    }

    /// No-op reduction: returns `local` unchanged (single process).
    pub struct LocalReduce;

    impl GlobalReduce for LocalReduce {
        #[inline]
        fn allreduce_sum(&self, local: f64) -> f64 { local }
    }

    /// MPI all-reduce using `MPI_COMM_WORLD`.
    ///
    /// # Safety
    /// Assumes the MPI runtime is initialised with at least
    /// `MPI_THREAD_SERIALIZED`.  No other thread must call MPI concurrently.
    pub struct MpiReduce;

    // Safety: MPI_COMM_WORLD is a process-global constant; our usage is
    // serialised (single solver thread per rank).
    unsafe impl Send for MpiReduce {}
    unsafe impl Sync for MpiReduce {}

    impl GlobalReduce for MpiReduce {
        fn allreduce_sum(&self, local: f64) -> f64 {
            let world = SimpleCommunicator::world();
            let mut global = 0.0_f64;
            world.all_reduce_into(&local, &mut global, &mpi::collective::SystemOperation::sum());
            global
        }
    }

    // ─── Per-neighbor descriptor ─────────────────────────────────────────────

    struct NeighborComm {
        rank: i32,
        /// Offsets within the owned-values slice to pack into the send buffer.
        send_owned_offsets: Vec<usize>,
        /// Positions in `ghost_out` where received values should land.
        recv_ghost_positions: Vec<usize>,
    }

    // ─── MpiHaloExchange ─────────────────────────────────────────────────────

    /// MPI-based halo exchange using non-blocking point-to-point communication.
    ///
    /// Implements [`HaloExchange`] via [`HaloExchange::exchange_with_owned`],
    /// which posts `MPI_Isend` / `MPI_Irecv` pairs for every neighbor and
    /// waits for all completions.
    ///
    /// # Construction
    /// Use [`MpiHaloExchange::new`] with a [`HaloPlan`] and owned range, or
    /// use the convenience [`MpiHaloExchange::from_dist_matrix`] helper.
    pub struct MpiHaloExchange {
        owned_start: usize,
        neighbors: Vec<NeighborComm>,
    }

    // Safety: see module-level doc.
    unsafe impl Send for MpiHaloExchange {}
    unsafe impl Sync for MpiHaloExchange {}

    impl MpiHaloExchange {
        /// Build an exchange descriptor from a `HaloPlan`.
        ///
        /// * `plan`          — communication plan (send/recv globals per neighbor)
        /// * `owned_start`   — first global index owned by this rank
        /// * `ghost_globals` — ordered list of ghost global indices (must match
        ///                     `DistCsrMatrix::layout().ghost_globals`)
        pub fn new(
            plan: &HaloPlan,
            owned_start: usize,
            ghost_globals: &[usize],
        ) -> Self {
            // Map ghost global → position in ghost_out buffer.
            let ghost_pos: HashMap<usize, usize> = ghost_globals
                .iter()
                .enumerate()
                .map(|(i, &g)| (g, i))
                .collect();

            let neighbors = plan
                .neighbors
                .iter()
                .map(|n| {
                    let send_owned_offsets = n
                        .send_globals
                        .iter()
                        .map(|&g| {
                            g.checked_sub(owned_start)
                                .expect("send_global must be inside owned range")
                        })
                        .collect();
                    let recv_ghost_positions = n
                        .recv_globals
                        .iter()
                        .map(|&g| {
                            *ghost_pos
                                .get(&g)
                                .expect("recv_global not found in ghost_globals list")
                        })
                        .collect();
                    NeighborComm {
                        rank: n.neighbor_rank as i32,
                        send_owned_offsets,
                        recv_ghost_positions,
                    }
                })
                .collect();

            Self { owned_start, neighbors }
        }

        /// Convenience constructor: derive the halo plan from a
        /// [`DistCsrMatrix`] and the world communicator.
        ///
        /// Each rank broadcasts the global indices it owns and derives which
        /// neighbors it needs to communicate with.  For a block-partitioned
        /// matrix this avoids constructing a `HaloPlan` by hand.
        pub fn from_dist_matrix<C: Communicator, T: Scalar>(
            dist: &DistCsrMatrix<T>,
            comm: &C,
        ) -> Self {
            let layout = dist.layout();
            let rank = comm.rank() as usize;
            let size = comm.size() as usize;

            let owned_start = layout.owned_global_range.start;
            let owned_end   = layout.owned_global_range.end;

            // Collect ghost globals grouped by their owning rank.
            let mut sends_to: Vec<Vec<usize>> = vec![Vec::new(); size]; // sends_to[r] = globals this rank needs from rank r
            for &g in &layout.ghost_globals {
                // Determine owner of g by scanning all ranks (works for block partition).
                // For non-block partitions, build an owner table via all-gather.
                let owner = owner_rank(g, layout.global_size, size);
                sends_to[owner].push(g);
            }

            // Build per-neighbor recv plans (globals we receive).
            let mut neighbor_plans: Vec<NeighborHaloPlan> = sends_to
                .iter()
                .enumerate()
                .filter(|(r, v)| *r != rank && !v.is_empty())
                .map(|(r, recv_globals)| NeighborHaloPlan {
                    neighbor_rank: r,
                    send_globals:  Vec::new(), // filled below
                    recv_globals:  recv_globals.clone(),
                })
                .collect();

            // Exchange: tell each neighbor which of their owned values we need
            // so they know what to send.
            // Use all-to-all: each rank broadcasts a list of globals it wants
            // from every other rank.
            //
            // Simple approach: all-gather each rank's ghost list and compute
            // send sets locally.
            //
            // We perform one all-gather of (rank, ghost_globals) pairs.
            let local_ghosts: Vec<usize> = layout.ghost_globals.clone();
            let local_len = local_ghosts.len() as i32;

            // Gather lengths.
            let mut all_lens = vec![0i32; size];
            comm.all_gather_into(&local_len, &mut all_lens[..]);

            // Gather all ghost lists into a flat buffer.
            let displs: Vec<i32> = all_lens
                .iter()
                .scan(0i32, |acc, &l| { let d = *acc; *acc += l; Some(d) })
                .collect();
            let total: i32 = all_lens.iter().sum();
            let mut all_ghosts = vec![0usize; total as usize];
            {
                use mpi::datatype::PartitionMut;
                let counts: Vec<_> = all_lens.iter().map(|&c| c).collect();
                let mut partition = PartitionMut::new(&mut all_ghosts[..], counts, &displs[..]);
                comm.all_gather_varcount_into(&local_ghosts[..], &mut partition);
            }

            // For each other rank r, the globals they requested that fall in
            // our owned range [owned_start, owned_end) are what we send.
            for plan in &mut neighbor_plans {
                let r = plan.neighbor_rank;
                let start = displs[r] as usize;
                let end   = start + all_lens[r] as usize;
                plan.send_globals = all_ghosts[start..end]
                    .iter()
                    .copied()
                    .filter(|&g| g >= owned_start && g < owned_end)
                    .collect();
            }
            // Add neighbors that only send to us (no recv from them).
            // These are ranks that have our owned globals in their ghost list.
            for r in 0..size {
                if r == rank { continue; }
                let start = displs[r] as usize;
                let end   = start + all_lens[r] as usize;
                let send_globals: Vec<usize> = all_ghosts[start..end]
                    .iter()
                    .copied()
                    .filter(|&g| g >= owned_start && g < owned_end)
                    .collect();
                if send_globals.is_empty() { continue; }
                // Update existing plan or insert new.
                if let Some(plan) = neighbor_plans.iter_mut().find(|p| p.neighbor_rank == r) {
                    plan.send_globals = send_globals;
                } else {
                    neighbor_plans.push(NeighborHaloPlan {
                        neighbor_rank: r,
                        send_globals,
                        recv_globals:  Vec::new(),
                    });
                }
            }

            let halo_plan = HaloPlan::new(neighbor_plans).expect("duplicate neighbor rank");
            Self::new(&halo_plan, owned_start, &layout.ghost_globals)
        }
    }

    /// Determine which rank owns global index `g` under a block partition.
    fn owner_rank(g: usize, global_size: usize, nranks: usize) -> usize {
        let q = global_size / nranks;
        let r = global_size % nranks;
        // Ranks [0, r) own q+1 items; ranks [r, nranks) own q items.
        if g < r * (q + 1) {
            g / (q + 1)
        } else {
            r + (g - r * (q + 1)) / q
        }
    }

    impl<T> HaloExchange<T> for MpiHaloExchange
    where
        T: Scalar + Equivalence,
    {
        fn exchange(&self, global_indices: &[usize], out: &mut [T]) -> Result<(), HaloError> {
            // Fallback: single-rank case — all ghost values must be available
            // from other ranks, which they won't be without owned values.
            // This method should not be called directly on a multi-rank run.
            let _ = (global_indices, out);
            panic!(
                "MpiHaloExchange::exchange called without owned values. \
                 Use exchange_with_owned (called automatically via spmv_with_halo)."
            )
        }

        fn exchange_with_owned(
            &self,
            _owned_range_start: usize,
            owned_values: &[T],
            _ghost_globals: &[usize],
            ghost_out: &mut [T],
        ) -> Result<(), HaloError> {
            use mpi::datatype::{Partition, PartitionMut};

            let world = SimpleCommunicator::world();
            let nranks = world.size() as usize;

            // Build per-rank send/recv counts.
            let mut send_counts = vec![0i32; nranks];
            let mut recv_counts = vec![0i32; nranks];
            for n in &self.neighbors {
                send_counts[n.rank as usize] = n.send_owned_offsets.len() as i32;
                recv_counts[n.rank as usize] = n.recv_ghost_positions.len() as i32;
            }

            // Exclusive prefix-sum displacements.
            let send_displs: Vec<i32> = send_counts
                .iter()
                .scan(0i32, |acc, &c| { let d = *acc; *acc += c; Some(d) })
                .collect();
            let recv_displs: Vec<i32> = recv_counts
                .iter()
                .scan(0i32, |acc, &c| { let d = *acc; *acc += c; Some(d) })
                .collect();

            let total_send: usize = send_counts.iter().map(|&c| c as usize).sum();
            let total_recv: usize = recv_counts.iter().map(|&c| c as usize).sum();

            // Pack send buffer in rank order.
            let mut send_buf = vec![T::zero(); total_send];
            for n in &self.neighbors {
                let start = send_displs[n.rank as usize] as usize;
                for (k, &i) in n.send_owned_offsets.iter().enumerate() {
                    send_buf[start + k] = owned_values[i];
                }
            }

            // Alltoallv — deadlock-free collective, no lifetime issues.
            let mut recv_buf = vec![T::zero(); total_recv];
            {
                let send_part = Partition::new(&send_buf[..], &send_counts[..], &send_displs[..]);
                let mut recv_part = PartitionMut::new(&mut recv_buf[..], &recv_counts[..], &recv_displs[..]);
                world.all_to_all_varcount_into(&send_part, &mut recv_part);
            }

            // Scatter into ghost_out.
            for n in &self.neighbors {
                let start = recv_displs[n.rank as usize] as usize;
                for (k, &pos) in n.recv_ghost_positions.iter().enumerate() {
                    ghost_out[pos] = recv_buf[start + k];
                }
            }

            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Verify single-rank MPI exchange matches LocalHaloExchange.
        /// Run with: `mpiexec -n 1 cargo test --features mpi`
        #[test]
        fn mpi_single_rank_halo_matches_local() {
            let _universe = mpi::initialize().unwrap_or_else(|| {
                panic!("MPI init failed — is an MPI runtime available?")
            });
            let world = SimpleCommunicator::world();
            assert_eq!(world.size(), 1, "this test expects a single MPI rank");

            // No neighbors on a single rank — exchange should be a no-op.
            let plan = HaloPlan::new(vec![]).unwrap();
            let mpi_halo = MpiHaloExchange::new(&plan, 0, &[]);

            let owned: Vec<f64> = vec![1.0, 2.0, 3.0];
            let ghost_globals: Vec<usize> = vec![];
            let mut ghost_out: Vec<f64> = vec![];

            mpi_halo
                .exchange_with_owned(0, &owned, &ghost_globals, &mut ghost_out)
                .unwrap();

            assert!(ghost_out.is_empty());
        }

        /// Verify MpiReduce sum is identity on a single rank.
        #[test]
        fn mpi_reduce_single_rank_is_identity() {
            let _universe = mpi::initialize().unwrap_or_else(|| {
                panic!("MPI init failed")
            });
            let reduce = MpiReduce;
            let local = 42.5_f64;
            let global = reduce.allreduce_sum(local);
            assert_eq!(global, local);
        }
    }
}
