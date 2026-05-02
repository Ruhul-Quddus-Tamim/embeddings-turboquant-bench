//! **TurboQuant-style vector index** — [`TurboQuantIndex::new`] (`Q_mse`) and
//! [`TurboQuantIndex::new_prod`] (`Q_prod`: `(b−1)`-bit MSE + QJL on the residual, Algorithm 2).
//!
//! Data-oblivious compression: Lloyd–Max scalar quantization on Gaussian-marginals after a fixed
//! random orthogonal rotation (`σ² = 1/d` per coordinate in the high‑d limit). Inner product in the
//! rotated domain approximates cosine for normalized embeddings.
pub mod csv_dataset;
pub mod disk;
pub mod index;
pub mod lloyd_max;
pub mod packing;
pub mod qjl;
pub mod rotation;
pub mod simd;

pub use crate::index::TurboQuantIndex;
