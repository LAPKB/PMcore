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
pub mod npmap;
pub mod npod;

// Re-export algorithm structs
pub use npag::NPAG;
pub use npmap::NPMAP;
pub use npod::NPOD;

// Re-export per-algorithm configuration structs
pub use npag::NpagConfig;
pub use npmap::NpmapConfig;
pub use npod::NpodConfig;
