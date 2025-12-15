//! Data structures for population pharmacokinetic modeling
//!
//! This module contains core data structures used across PMcore algorithms.
//!
//! # Module Organization
//!
//! - [`nonparametric`]: Structures for non-parametric algorithms (Theta, Psi, Weights)
//! - [`parametric`]: Structures for parametric algorithms (Population, Individual, SufficientStats)

pub mod nonparametric;
pub mod parametric;

// Re-export submodules for backward compatibility
pub use nonparametric::psi;
pub use nonparametric::theta;
pub use nonparametric::weights;

// Re-export commonly used types from submodules
pub use nonparametric::{Psi, Theta, Weights};
pub use parametric::{Individual, IndividualEstimates, Population, SufficientStats};
