//! Numerical integration methods for marginal likelihood estimation.
//!
//! This module provides building blocks for computing marginal likelihoods in
//! mixed-effects models by integrating over random effects.

mod importance_sampling;

pub(crate) use importance_sampling::{
    ImportanceSamplingConfig, ImportanceSamplingEstimator, SubjectConditionalPosterior,
};
