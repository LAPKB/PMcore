//! MCMC Sampling infrastructure for parametric algorithms
//!
//! This module provides the sampling machinery needed for algorithms like SAEM
//! that require samples from the conditional distribution p(η | y, θ).
//!
//! # Available Samplers
//!
//! - [`MetropolisHastings`]: Basic random-walk Metropolis-Hastings
//! - [`FSaemKernels`]: Four-kernel sampler for f-SAEM algorithm
//! - [`MapEstimator`]: MAP estimation for f-SAEM Kernel 4
//! - [`ParameterTransforms`]: Parameter space transformations (φ ↔ ψ)
//!
//! # f-SAEM Algorithm
//!
//! The f-SAEM (fast SAEM) algorithm uses four MCMC kernels for improved mixing:
//! 1. **Kernel 1**: Full proposals from prior N(0, Ω)
//! 2. **Kernel 2**: Component-wise random walk
//! 3. **Kernel 3**: Block random walk with adaptive grouping
//! 4. **Kernel 4**: MAP-based proposals using Laplace approximation
//!
//! # Parameter Transformations
//!
//! SAEM works internally in transformed (unconstrained) space φ where random
//! effects are normally distributed. Use [`ParameterTransforms`] to convert
//! between:
//! - φ (unconstrained): For MCMC sampling, sufficient statistics
//! - ψ (constrained): For likelihood computation, reporting
//!
//! # Usage
//!
//! Samplers are used in the E-step of SAEM to draw samples from the conditional
//! posterior distribution of individual parameters given observations and population parameters.

pub mod kernels;
pub mod map;
pub mod metropolis;
pub mod proposal;
pub mod transforms;

// Basic MH sampler
pub use metropolis::MetropolisHastings;
pub use proposal::{AdaptiveProposal, GaussianProposal, ProposalDistribution};

// f-SAEM components
pub use kernels::{AdaptiveScales, ChainState, FSaemKernels, KernelConfig, MapEstimate};
pub use map::{BatchMapEstimator, MapConfig, MapEstimator};

// Parameter transforms
pub use transforms::ParameterTransforms;
