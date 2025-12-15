//! MCMC Sampling infrastructure for parametric algorithms
//!
//! This module provides the sampling machinery needed for algorithms like SAEM
//! that require samples from the conditional distribution p(η | y, θ).
//!
//! # Available Samplers
//!
//! - [`MetropolisHastings`]: Basic random-walk Metropolis-Hastings
//! - Adaptive MCMC (planned)
//! - Hamiltonian Monte Carlo (future)
//!
//! # Usage
//!
//! Samplers are used in the E-step of SAEM to draw samples from the conditional
//! posterior distribution of individual parameters given observations and population parameters.

pub mod metropolis;
pub mod proposal;

pub use metropolis::MetropolisHastings;
pub use proposal::{ProposalDistribution, GaussianProposal, AdaptiveProposal};
