#[cfg(not(target_arch = "wasm32"))]
mod nalgebra;
pub mod bsr;
pub mod coo;
pub mod csc;
pub mod csr;
pub mod dia;
pub mod ell;
pub mod mmio;
pub mod ops;

pub use bsr::{BsrMatrix, BsrBuilder};
pub use coo::CooMatrix;
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
pub use dia::DiaMatrix;
pub use ell::EllMatrix;
pub use mmio::{
    read_matrix_market, read_matrix_market_coo,
    read_matrix_market_str, read_matrix_market_coo_str,
    write_matrix_market, write_matrix_market_str,
    MmioError,
};
