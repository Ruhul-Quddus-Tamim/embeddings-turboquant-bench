//! **TurboQuant vector index** — [`TurboQuantIndex::new`] (`Q_mse`: scalar Lloyd–Max per coordinate after rotation).
//!
//! Add/search use the in-crate Gaussian Lloyd–Max + [`crate::simd`] LUT path (bit-packed codes).
//! Snapshots: **v1** / **v2** (MSE with rotation **Q** on disk). Legacy **v3** snapshots are rejected; rebuild if needed.
pub mod csv_dataset;
pub mod disk;
pub mod index;
pub mod lloyd_max;
pub mod packing;
pub mod recall_report;
pub mod latency;
pub mod rotation;
pub mod search_blocked_neon;
pub mod simd;

pub use crate::index::TurboQuantIndex;
