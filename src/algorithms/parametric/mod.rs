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
//! SAEM runs through the same [`Algorithm`] and
//! [`fit_with`](crate::estimation::EstimationProblem::fit_with) lifecycle as the
//! nonparametric algorithms. The cycle controller and result interfaces are
//! algorithm-neutral.

pub mod controller;
mod saem;
pub mod saem_config;

pub use controller::{CycleFlow, FitController, FitObserver, ParametricFitSnapshot};
use saem::SaemState;
pub use saem_config::{
    CovarianceStabilityConfig, LugsailConfig, MarkovSimulationVarianceConfig,
    OperationalConvergenceConfig, SaemConfig, SaemEstimatorPolicy,
};

use crate::algorithms::{Algorithm, Status, StopReason};
use crate::estimation::{EstimationProblem, Parametric};
use crate::results::{ParametricResult, SaemCycleDiagnostics};
use anyhow::Result;
use ndarray::Array2;
use pharmsol::prelude::simulator::Equation;

/// The SAEM operation that encountered a numerical failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalFailurePhase {
    /// The expectation step.
    Expectation,
    /// The maximization step.
    Maximization,
    /// Post-fit conditional-mode or result assembly.
    ResultAssembly,
}

impl std::fmt::Display for NumericalFailurePhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let phase = match self {
            Self::Expectation => "expectation",
            Self::Maximization => "maximization",
            Self::ResultAssembly => "result assembly",
        };
        f.write_str(phase)
    }
}

/// A numerical error that terminated a parametric fit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("numerical failure during {phase} at cycle {attempted_cycle}: {source_message}")]
pub struct NumericalFailure {
    attempted_cycle: usize,
    phase: NumericalFailurePhase,
    source_message: String,
}

impl NumericalFailure {
    pub(crate) fn new(
        attempted_cycle: usize,
        phase: NumericalFailurePhase,
        source_message: String,
    ) -> Self {
        Self {
            attempted_cycle,
            phase,
            source_message,
        }
    }

    /// The cycle being attempted when the failure occurred.
    pub fn attempted_cycle(&self) -> usize {
        self.attempted_cycle
    }

    /// The operation that encountered the failure.
    pub fn phase(&self) -> NumericalFailurePhase {
        self.phase
    }

    /// The original error message.
    pub fn source_message(&self) -> &str {
        &self.source_message
    }
}

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

    pub(crate) fn into_runner<E: Equation + Send + 'static>(
        self,
        problem: EstimationProblem<E, Parametric>,
    ) -> Result<Box<dyn ParametricRunner<E>>> {
        let runner: Box<dyn ParametricRunner<E>> = match self {
            Self::Saem(config) => Box::new(SaemState::from_problem(problem, &config)?),
        };
        Ok(runner)
    }
}

pub(crate) trait ParametricRunner<E: Equation + Send + 'static>: Send {
    fn step(&mut self) -> Result<Status>;
    fn request_stop(&mut self, reason: StopReason);
    fn cycle(&self) -> usize;
    fn status(&self) -> &Status;
    fn cycle_diagnostics(&self) -> &[SaemCycleDiagnostics];
    fn log_likelihood(&self) -> f64;
    fn population_parameters(&self) -> &[f64];
    fn covariate_betas(&self) -> Option<Vec<f64>>;
    fn random_effect_names(&self) -> &[String];
    fn iov_effect_names(&self) -> Option<&[String]>;
    fn eta_log_prior(&self) -> f64;
    fn kappa_log_prior(&self) -> f64;
    fn acceptance_rate(&self) -> Option<f64>;
    fn eta_block_acceptance_rate(&self) -> Option<f64>;
    fn kappa_acceptance_rate(&self) -> Option<f64>;
    fn rejected_proposals(&self) -> Option<usize>;
    fn non_finite_proposals(&self) -> Option<usize>;
    fn parameter_acceptance_rates(&self) -> Option<&[f64]>;
    fn proposal_step_sizes(&self) -> Option<&[f64]>;
    fn eta_block_step_sizes(&self) -> Option<&[f64]>;
    fn log_acceptance_ratios(&self) -> Option<&[f64]>;
    fn negative_log_likelihood(&self) -> f64;
    fn n_chains(&self) -> Option<usize>;
    fn omega(&self) -> Option<&Array2<f64>>;
    fn omega_iov(&self) -> Option<&Array2<f64>>;
    fn residual_sigmas(&self) -> &[f64];
    fn step_size(&self) -> f64;
    fn total_iterations(&self) -> usize;
    fn into_result(self: Box<Self>) -> Result<ParametricResult<E>>;

    fn log_posterior(&self) -> f64 {
        self.log_likelihood() + self.eta_log_prior() + self.kappa_log_prior()
    }

    fn n2ll(&self) -> f64 {
        2.0 * self.negative_log_likelihood()
    }

    fn omega_diagonal(&self) -> Option<Vec<f64>> {
        self.omega().map(|omega| {
            (0..omega.nrows())
                .map(|index| omega[[index, index]])
                .collect()
        })
    }
}

impl<E: Equation + Send + 'static> Algorithm<E, Parametric> for ParametricAlgorithm {
    type Output = ParametricResult<E>;

    fn fit(self, problem: EstimationProblem<E, Parametric>) -> Result<Self::Output> {
        FitController::new(self, problem)?.finish()
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
