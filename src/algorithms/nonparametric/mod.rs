//! Non-parametric algorithm implementations
//!
//! This module contains the trait definition and implementations for non-parametric
//! population pharmacokinetic algorithms. These algorithms estimate the population
//! distribution as a discrete set of support points with associated probability weights.
//!
//! # Available Algorithms
//!
//! - [`NPAG`](npag): Non-Parametric Adaptive Grid
//! - [`NPOD`](npod): Non-Parametric Optimal Design
//! - [`NPMAP`](npmap): Maximum a posteriori reweighting
//!
//! # Algorithm Selection
//!
//! Use the [`NonParametricAlgorithm`] enum to select and configure an algorithm. Each
//! variant wraps its algorithm-specific configuration struct (e.g. [`NpagConfig`]). The
//! internal execution state used while fitting implements the [`NonParametricRunner`]
//! trait, which defines the common interface for initialization, estimation, condensation,
//! expansion, and convergence evaluation.

// Shared error-model factor optimization
pub mod error_optim;

// Algorithm implementations
pub mod ncnpag;
pub mod npag;
pub mod npmap;
pub mod npod;

// Incremental, observable fitting (stepping handle + per-cycle observers).
pub mod controller;

// Re-export algorithm structs
pub use ncnpag::NCNPAG;
pub use npag::NPAG;
pub use npmap::NPMAP;
pub use npod::NPOD;

// Re-export per-algorithm configuration structs
pub use error_optim::ErrorOptimConfig;
pub use ncnpag::NcnpagConfig;
pub use npag::NpagConfig;
pub use npmap::NpmapConfig;
pub use npod::NpodConfig;

// Re-export the incremental fitting API
pub use controller::{CycleFlow, FitController, FitObserver};

use crate::algorithms::{Algorithm, NonParametricRunner};
use crate::estimation::nonparametric::NonParametricResult;
use crate::estimation::{EstimationProblem, NonParametric};
use anyhow::Result;
use pharmsol::prelude::simulator::Equation;

/// The non-parametric algorithms supported by PMcore.
///
/// Use the constructors to select an algorithm with its default configuration:
///
/// ```no_run
/// use pmcore::prelude::*;
///
/// // Default NPAG configuration.
/// let algorithm = NonParametricAlgorithm::npag();
/// ```
///
/// To customize an algorithm, build its configuration struct (which exposes only the
/// setters valid for that algorithm) and pass it directly to
/// [`fit_with`](crate::estimation::EstimationProblem::fit_with):
///
/// ```no_run
/// use pmcore::prelude::*;
///
/// // NPAG with a tighter convergence criterion and a cycle cap.
/// let config = NpagConfig::new().eps(0.1).max_cycles(500);
/// // `problem.fit_with(config)` accepts the config directly.
/// ```
///
/// Each configuration type ([`NpagConfig`], [`NpodConfig`], [`NpmapConfig`]) implements
/// [`Algorithm`] by delegating to the matching enum variant, so configs can be passed to
/// `fit_with` without converting them first.
#[derive(Debug, Clone)]
pub enum NonParametricAlgorithm {
    /// Non-Parametric Adaptive Grid.
    Npag(NpagConfig),
    /// Non-Parametric Optimal Design.
    Npod(NpodConfig),
    /// Non-parametric maximum a posteriori (posterior probability reweighting).
    Npmap(NpmapConfig),
    /// Non-collapsing NPAG (single-pass Bayesian reweighting of fixed support points).
    Ncnpag(NcnpagConfig),
}

impl Default for NonParametricAlgorithm {
    fn default() -> Self {
        Self::npag()
    }
}

impl From<NpagConfig> for NonParametricAlgorithm {
    fn from(config: NpagConfig) -> Self {
        Self::Npag(config)
    }
}

impl From<NpodConfig> for NonParametricAlgorithm {
    fn from(config: NpodConfig) -> Self {
        Self::Npod(config)
    }
}

impl From<NpmapConfig> for NonParametricAlgorithm {
    fn from(config: NpmapConfig) -> Self {
        Self::Npmap(config)
    }
}

impl From<NcnpagConfig> for NonParametricAlgorithm {
    fn from(config: NcnpagConfig) -> Self {
        Self::Ncnpag(config)
    }
}

impl NonParametricAlgorithm {
    /// The Non-Parametric Adaptive Grid (NPAG) algorithm with its default configuration.
    pub fn npag() -> Self {
        Self::Npag(NpagConfig::default())
    }

    /// The Non-Parametric Optimal Design (NPOD) algorithm with its default configuration.
    pub fn npod() -> Self {
        Self::Npod(NpodConfig::default())
    }

    /// The non-parametric maximum a posteriori (NPMAP) algorithm with its default
    /// configuration.
    pub fn npmap() -> Self {
        Self::Npmap(NpmapConfig::default())
    }

    /// The non-collapsing NPAG (NCNPAG) algorithm with its default configuration.
    pub fn ncnpag() -> Self {
        Self::Ncnpag(NcnpagConfig::default())
    }

    /// Build the internal, mutable execution state (runner) for this algorithm.
    ///
    /// Both [`fit`](Algorithm::fit) and the stepping
    /// [`FitController`](crate::algorithms::nonparametric::controller::FitController)
    /// build on this primitive.
    pub(crate) fn into_runner<E: Equation + Send + 'static>(
        self,
        problem: EstimationProblem<E, NonParametric>,
    ) -> Result<Box<dyn NonParametricRunner<E>>> {
        // `problem.prior` is the prior `Theta` (which also carries the parameter
        // space) and `problem.error_models` is strictly `AssayErrorModels`.
        let runner: Box<dyn NonParametricRunner<E>> = match self {
            Self::Npag(config) => Box::new(NPAG::from_parts(
                problem.model.equation,
                problem.data,
                problem.error_models,
                problem.prior,
                config,
            )?),
            Self::Npod(config) => Box::new(NPOD::from_parts(
                problem.model.equation,
                problem.data,
                problem.error_models,
                problem.prior,
                config,
            )?),
            Self::Npmap(config) => Box::new(NPMAP::from_parts(
                problem.model.equation,
                problem.data,
                problem.error_models,
                problem.prior,
                config,
            )?),
            Self::Ncnpag(config) => Box::new(NCNPAG::from_parts(
                problem.model.equation,
                problem.data,
                problem.error_models,
                problem.prior,
                config,
            )?),
        };
        Ok(runner)
    }
}

impl<E: Equation + Send + 'static> Algorithm<E, NonParametric> for NonParametricAlgorithm {
    type Output = NonParametricResult<E>;

    fn fit(self, problem: EstimationProblem<E, NonParametric>) -> Result<Self::Output> {
        let mut runner = self.into_runner(problem)?;
        runner.fit()
    }
}

// Each configuration struct delegates to its matching `NonParametricAlgorithm` variant so it
// can be passed directly to `fit_with`. This keeps the variant-specific setters on the config
// types (compile-time checked) while the enum remains the single source of fitting logic.
impl<E: Equation + Send + 'static> Algorithm<E, NonParametric> for NpagConfig {
    type Output = NonParametricResult<E>;

    fn fit(self, problem: EstimationProblem<E, NonParametric>) -> Result<Self::Output> {
        NonParametricAlgorithm::from(self).fit(problem)
    }
}

impl<E: Equation + Send + 'static> Algorithm<E, NonParametric> for NpodConfig {
    type Output = NonParametricResult<E>;

    fn fit(self, problem: EstimationProblem<E, NonParametric>) -> Result<Self::Output> {
        NonParametricAlgorithm::from(self).fit(problem)
    }
}

impl<E: Equation + Send + 'static> Algorithm<E, NonParametric> for NpmapConfig {
    type Output = NonParametricResult<E>;

    fn fit(self, problem: EstimationProblem<E, NonParametric>) -> Result<Self::Output> {
        NonParametricAlgorithm::from(self).fit(problem)
    }
}

impl<E: Equation + Send + 'static> Algorithm<E, NonParametric> for NcnpagConfig {
    type Output = NonParametricResult<E>;

    fn fit(self, problem: EstimationProblem<E, NonParametric>) -> Result<Self::Output> {
        NonParametricAlgorithm::from(self).fit(problem)
    }
}
