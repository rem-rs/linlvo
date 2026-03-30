#[cfg(not(target_arch = "wasm32"))]
pub mod adapt_faer;
#[cfg(not(target_arch = "wasm32"))]
pub mod adapt_nalgebra;
pub mod bsr;
pub mod coo;
pub mod csc;
pub mod csr;
pub mod ops;

pub use bsr::{BsrMatrix, BsrBuilder};
pub use coo::CooMatrix;
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
