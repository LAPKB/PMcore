//! Likelihood scoring.
//!
//! This module provides estimator-facing observation scoring and likelihood
//! aggregation over generated predictions.

pub(crate) mod batch;
mod distributions;
pub use distributions::NormalDistributionError;
pub(crate) mod matrix;
pub(crate) mod objective;
pub(crate) mod observation;
pub use observation::AssayLikelihoodError;
pub(crate) mod particle;
pub(crate) mod residual;
