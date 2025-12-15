//! Parametric population structures
//!
//! This module contains structures specific to parametric algorithms (SAEM, FOCE, IT2B, etc.),
//! which represent population distributions as continuous distributions parameterized by
//! a mean vector (μ) and covariance matrix (Ω).
//!
//! # Key Structures
//!
//! - [`Population`]: Parametric population parameters (μ, Ω)
//! - [`Individual`]: Individual parameter estimates (empirical Bayes estimates)
//! - [`SufficientStats`]: Sufficient statistics for SAEM algorithm
//!
//! # Algorithm Support
//!
//! These structures support the following parametric algorithms:
//! - **SAEM**: Stochastic Approximation Expectation-Maximization
//! - **FOCE/FOCEI**: First-Order Conditional Estimation
//! - **FO**: First-Order (linearization) method
//! - **Laplacian**: Laplacian approximation method
//! - **IT2B**: Iterative Two-Stage Bayesian

pub mod individual;
pub mod population;
pub mod sufficient_stats;

pub use individual::{Individual, IndividualEstimates};
pub use population::Population;
pub use sufficient_stats::SufficientStats;
