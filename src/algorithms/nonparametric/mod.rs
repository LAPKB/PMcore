//! Non-parametric algorithm implementations
//!
//! This module contains the trait definition and implementations for non-parametric
//! population pharmacokinetic algorithms. These algorithms estimate the population
//! distribution as a discrete set of support points with associated probability weights.
//!
//! # Available Algorithms
//!
//! - [`NPAG`](npag): Non-Parametric Adaptive Grid
//! - [`NPOD`](npod): Non-Parametric Optimal Design
//! - [`POSTPROB`](postprob): Posterior probability reweighting
//!
//! # Algorithm Trait
//!
//! All non-parametric algorithms implement the [`NPAlgorithm`] trait (aliased from `Algorithms`)
//! which defines the common interface for initialization, estimation, condensation, expansion,
//! and convergence evaluation.

// Algorithm implementations
pub mod npag;
pub mod npod;
pub mod postprob;

// Re-export algorithm structs
pub use npag::NPAG;
pub use npod::NPOD;
pub use postprob::POSTPROB;

// Re-export the NP algorithm trait from parent
pub use super::Algorithms as NPAlgorithm;
