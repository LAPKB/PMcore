//! Numerical integration methods for marginal likelihood estimation
//!
//! This module provides building blocks for computing marginal likelihoods in
//! mixed-effects models by integrating over random effects:
//!
//! - **Importance Sampling**: Monte Carlo estimate using proposal distribution
//! - **Gaussian Quadrature**: Deterministic numerical integration (planned)
//! - **Laplace Approximation**: Second-order approximation (planned, for FOCE)
//!
//! # Mathematical Background
//!
//! The marginal likelihood for subject i is:
//!
//! ```text
//! p(yᵢ | θ) = ∫ p(yᵢ | ηᵢ, θ) × p(ηᵢ | Ω) dηᵢ
//! ```
//!
//! The conditional likelihood `p(yᵢ | ηᵢ, θ)` is computed by pharmsol.
//! This module provides methods to numerically evaluate the integral.

mod importance_sampling;

pub use importance_sampling::{
    ImportanceSamplingConfig, ImportanceSamplingEstimator, SubjectConditionalPosterior,
};
