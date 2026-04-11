//! Pure-Rust distributed-memory scaffolding.
//!
//! This module provides the initial abstractions for distributed sparse linear
//! algebra without introducing external FFI backends.

pub mod layout;
pub mod halo;
pub mod dist_csr;

pub use layout::{block_partition, PartitionLayout};
pub use halo::{
	HaloError, HaloExchange, LocalHaloExchange,
	HaloPlan, NeighborHaloPlan,
};
pub use dist_csr::DistCsrMatrix;
