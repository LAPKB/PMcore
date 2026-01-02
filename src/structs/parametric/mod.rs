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
//! - [`ParameterTransform`]: Parameter space transformations (log-normal, logit, probit)
//! - [`CovariateModel`]: Covariate effects on population parameters
//!
//! # Algorithm Support
//!
//! These structures support the following parametric algorithms:
//! - **SAEM**: Stochastic Approximation Expectation-Maximization
//! - **FOCE/FOCEI**: First-Order Conditional Estimation
//! - **FO**: First-Order (linearization) method
//! - **Laplacian**: Laplacian approximation method
//! - **IT2B**: Iterative Two-Stage Bayesian

pub mod covariate;
pub mod individual;
pub mod population;
pub mod sufficient_stats;
pub mod transform;

pub use covariate::CovariateModel;
pub use individual::{Individual, IndividualEstimates};
pub use population::{CovarianceStructure, Population};
pub use sufficient_stats::{StepSizeSchedule, SufficientStats};
pub use transform::ParameterTransform;
