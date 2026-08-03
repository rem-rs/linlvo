pub mod jacobi;
pub mod sor;
pub mod gs_smoother;
pub mod ilu0;
pub mod iluk;
pub mod ilut;
pub mod icc;
pub mod ildlt;
pub mod spai;
pub mod composite;
pub mod block_jacobi;
pub mod ams;
pub mod ads;
pub mod fieldsplit;

pub use jacobi::JacobiPrecond;
pub use gs_smoother::GaussSeidelSmoother;
pub use sor::{SorPrecond, SsorPrecond};
pub use ilu0::Ilu0Precond;
pub use iluk::IlukPrecond;
pub use ilut::IlutPrecond;
pub use icc::Icc0Precond;
pub use ildlt::IldltPrecond;
pub use spai::SpaiPrecond;
pub use composite::{AdditivePrecond, MultiplicativePrecond};
pub use block_jacobi::BlockJacobiPrecond;
pub use ams::{AmsPrecond, AmsConfig, AmsEdgeSmoother, AmsCycle, AmsProfile, AuxSpaceSolver, AuxSolverProfile, AuxAmgProfile};
pub use ads::{AdsPrecond, AdsConfig, AdsProfile};
pub use fieldsplit::{FieldSplitPrecond, SplitMode};

/// MFEM-compatible alias: block-diagonal preconditioner over contiguous fields.
///
/// Construct with `SplitMode::BlockJacobi` to match MFEM's
/// `BlockDiagonalPreconditioner` (used in ex5, ex4, Stokes examples, etc.),
/// or `SplitMode::BlockTriangular` for lower-triangular coupling.
pub type BlockDiagonalPreconditioner<T> = FieldSplitPrecond<T>;
