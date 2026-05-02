//! Pure-Rust distributed-memory scaffolding.
//!
//! This module provides the initial abstractions for distributed sparse linear
//! algebra without introducing external FFI backends.

pub mod layout;
pub mod halo;
pub mod dist_csr;
pub mod dist_cg;
pub mod mpi_halo;

pub use layout::{block_partition, PartitionLayout};
pub use halo::{
    HaloError, HaloExchange, LocalHaloExchange,
    HaloPlan, NeighborHaloPlan,
};
pub use dist_csr::DistCsrMatrix;
pub use dist_cg::{dist_cg, DistCgParams, DistCgResult};
pub use mpi_halo::{GlobalReduce, LocalReduce};

#[cfg(feature = "mpi")]
pub use mpi_halo::{MpiHaloExchange, MpiReduce};
