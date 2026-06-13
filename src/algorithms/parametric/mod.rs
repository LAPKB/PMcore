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
/// Use the constructors to select an algorithm with its default configuration:
///
/// ```no_run
/// use pmcore::prelude::*;
///
/// // Default SAEM configuration.
/// let algorithm = ParametricAlgorithm::saem();
/// ```
///
/// To customize an algorithm, build its configuration struct (which exposes only the
/// setters valid for that algorithm) and pass it directly to
/// [`fit_with`](crate::estimation::EstimationProblem::fit_with):
///
/// ```no_run
/// use pmcore::prelude::*;
///
/// // SAEM with a custom iteration schedule and seed.
/// let config = SaemConfig::new()
///     .k1_iterations(500)
///     .k2_iterations(200)
///     .seed(42);
/// // `problem.fit_with(config)` accepts the config directly.
/// ```
///
/// [`SaemConfig`] implements [`Algorithm`] by delegating to the matching enum variant, so
/// configs can be passed to `fit_with` without converting them first.
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

// `SaemConfig` delegates to the matching `ParametricAlgorithm` variant so it can be passed
// directly to `fit_with`, keeping its setters compile-time checked.
impl<E: Equation + Send + 'static> Algorithm<E, Parametric> for SaemConfig {
    type Output = ParametricResult<E>;

    fn fit(self, problem: EstimationProblem<E, Parametric>) -> Result<Self::Output> {
        ParametricAlgorithm::from(self).fit(problem)
    }
}
