//! Parametric algorithm implementations
//!
//! This module contains the trait definition and placeholder implementations for parametric
//! population pharmacokinetic algorithms. These algorithms estimate the population
//! distribution as a continuous distribution parameterized by mean (μ) and covariance (Ω).
//!
//! # Supported Algorithms
//!
//! - **SAEM**: Stochastic Approximation Expectation-Maximization
//! - **FOCE/FOCEI**: First-Order Conditional Estimation (with/without Interaction)
//! - **FO**: First-Order (linearization) method
//! - **Laplacian**: Laplacian approximation method
//! - **IT2B**: Iterative Two-Stage Bayesian
//!
//! # Algorithm Trait
//!
//! All parametric algorithms implement the [`ParametricAlgorithm`] trait, which defines
//! the common interface for population parameter estimation with continuous distributions.

mod algorithm;
pub mod foce;
pub mod saem;

pub use algorithm::{ParametricAlgorithm, ParametricConfig};
