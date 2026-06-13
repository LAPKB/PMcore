//! Parametric algorithm implementations.
//!
//! Parametric algorithms estimate the population distribution as a parametric model (for
//! example a multivariate normal) whose parameters are fitted to the data.
//!
//! # Algorithm Selection
//!
//! Use the [`ParametricAlgorithm`] enum to select and configure an algorithm. Each variant
//! wraps its algorithm-specific configuration struct (e.g. [`SaemConfig`]).
//!
//! Note: the parametric fitting machinery is not yet implemented. Constructing a problem and
//! calling [`fit_with`](crate::estimation::EstimationProblem::fit_with) with a
//! [`ParametricAlgorithm`] type-checks today, but running it will panic until the SAEM solver
//! is implemented.

pub mod saem_config;

pub use saem_config::SaemConfig;

use crate::algorithms::Algorithm;
use crate::estimation::{EstimationProblem, Parametric};
use crate::results::ParametricResult;
use anyhow::Result;
use pharmsol::prelude::simulator::Equation;

/// The parametric algorithms supported by PMcore.
///
/// Each variant wraps the configuration for one algorithm. Construct a variant with its
/// default configuration using [`saem`](Self::saem), then refine it with the chainable
/// setters:
///
/// ```no_run
/// use pmcore::prelude::*;
///
/// // Default SAEM configuration.
/// let algorithm = ParametricAlgorithm::saem();
///
/// // SAEM with a custom iteration schedule and seed.
/// let algorithm = ParametricAlgorithm::saem()
///     .k1_iterations(500)
///     .k2_iterations(200)
///     .seed(42);
/// ```
///
/// The setters are passthroughs to the wrapped configuration.
#[derive(Debug, Clone)]
pub enum ParametricAlgorithm {
    /// Stochastic Approximation Expectation-Maximization.
    Saem(SaemConfig),
}

impl Default for ParametricAlgorithm {
    fn default() -> Self {
        Self::saem()
    }
}

impl From<SaemConfig> for ParametricAlgorithm {
    fn from(config: SaemConfig) -> Self {
        Self::Saem(config)
    }
}

impl ParametricAlgorithm {
    /// The Stochastic Approximation Expectation-Maximization (SAEM) algorithm with its
    /// default configuration.
    pub fn saem() -> Self {
        Self::Saem(SaemConfig::default())
    }

    /// Number of exploration-phase (K1) iterations (SAEM only).
    pub fn k1_iterations(mut self, iterations: usize) -> Self {
        let Self::Saem(config) = &mut self;
        config.k1_iterations = iterations;
        self
    }

    /// Number of smoothing-phase (K2) iterations (SAEM only).
    pub fn k2_iterations(mut self, iterations: usize) -> Self {
        let Self::Saem(config) = &mut self;
        config.k2_iterations = iterations;
        self
    }

    /// Number of burn-in iterations (SAEM only).
    pub fn burn_in(mut self, burn_in: usize) -> Self {
        let Self::Saem(config) = &mut self;
        config.burn_in = burn_in;
        self
    }

    /// Number of MCMC chains (SAEM only).
    pub fn n_chains(mut self, n_chains: usize) -> Self {
        let Self::Saem(config) = &mut self;
        config.n_chains = n_chains;
        self
    }

    /// MCMC step size (SAEM only).
    pub fn mcmc_step_size(mut self, step_size: f64) -> Self {
        let Self::Saem(config) = &mut self;
        config.mcmc_step_size = step_size;
        self
    }

    /// Random-number-generator seed (SAEM only).
    pub fn seed(mut self, seed: u64) -> Self {
        let Self::Saem(config) = &mut self;
        config.seed = seed;
        self
    }
}

impl<E: Equation + Send + 'static> Algorithm<E, Parametric> for ParametricAlgorithm {
    type Output = ParametricResult<E>;

    fn fit(self, _problem: EstimationProblem<E, Parametric>) -> Result<Self::Output> {
        match self {
            Self::Saem(_config) => {
                unimplemented!("SAEM fitting is not yet implemented")
            }
        }
    }
}
