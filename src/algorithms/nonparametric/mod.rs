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
//! - [`NPSAH`](npsah): Non-Parametric Simulated Annealing Hybrid
//! - And others...
//!
//! # Algorithm Trait
//!
//! All non-parametric algorithms implement the [`NPAlgorithm`] trait (aliased from `Algorithms`)
//! which defines the common interface for initialization, estimation, condensation, expansion,
//! and convergence evaluation.

// Algorithm implementations
pub mod nexus;
pub mod npag;
pub mod npbo;
pub mod npcat;
pub mod npcma;
pub mod npod;
pub mod npopt;
pub mod nppso;
pub mod npsah;
pub mod npsah2;
pub mod npxo;
pub mod postprob;

// Re-export algorithm structs
pub use nexus::NEXUS;
pub use npag::NPAG;
pub use npbo::NPBO;
pub use npcat::NPCAT;
pub use npcma::NPCMA;
pub use npod::NPOD;
pub use npopt::NPOPT;
pub use nppso::NPPSO;
pub use npsah::NPSAH;
pub use npsah2::NPSAH2;
pub use npxo::NPXO;
pub use postprob::POSTPROB;

// Re-export the NP algorithm trait from parent
pub use super::Algorithms as NPAlgorithm;
