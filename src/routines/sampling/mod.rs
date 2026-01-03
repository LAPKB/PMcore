//! MCMC Sampling infrastructure for parametric algorithms
//!
//! This module provides the sampling machinery needed for algorithms like SAEM
//! that require samples from the conditional distribution p(η | y, θ).
//!
//! # Available Components
//!
//! - [`ChainState`]: State of an MCMC chain for a single subject
//! - [`KernelConfig`]: Configuration for MCMC kernels
//! - [`MapEstimator`]: MAP estimation for individual parameters
//!
//! # Parameter Transformations
//!
//! For parameter space transformations (φ ↔ ψ), use
//! [`crate::structs::parametric::ParameterTransform`].

pub mod kernels;
pub mod map;

// Core MCMC components
pub use kernels::{ChainState, KernelConfig};
pub use map::{MapConfig, MapEstimator};
