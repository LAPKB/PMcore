//! Parametric algorithm implementations for the unified estimation platform.
//!
//! Supported today:
//! - SAEM
//! - FOCEI
//!
//! IT2B remains intentionally deferred.

mod algorithm;
pub mod focei;
pub mod saem;

pub(crate) use algorithm::run_parametric_algorithm;
pub(crate) use algorithm::ParametricAlgorithmInput;
pub use algorithm::{dispatch_parametric_algorithm, ParametricAlgorithm, ParametricConfig};
pub use saem::{FSaemConfig, FSAEM};
