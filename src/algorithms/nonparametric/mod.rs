//! Non-parametric algorithm implementations
//!
//! This module contains the trait definition and implementations for non-parametric
//! population pharmacokinetic algorithms. These algorithms estimate the population
//! distribution as a discrete set of support points with associated probability weights.
//!
//! # Available Algorithms
//!
//! - [NPAG](npag): Non-Parametric Adaptive Grid
//! - [NPOD](npod): Non-Parametric Optimal Design
//! - [NPMAP](npmap): Maximum a posteriori reweighting
//! - [NCNPAG](ncnpag): Non-collapsing NPAG
//!
//! # Algorithm Selection
//!
//! Use the [`NonParametricAlgorithm`] enum to select and configure an algorithm. Each
//! variant wraps its algorithm-specific configuration struct (e.g. [`NpagConfig`]).
//!
//! # Structure
//!
//! Every algorithm runs on the same engine (see the `fit` module): the state shared by all
//! algorithms (support points, likelihoods, weights, cycle log) lives in one place, and each
//! algorithm only supplies what a single cycle does to it.

// Shared error-model factor optimization
pub mod error_optim;

// The engine every algorithm runs on.
mod fit;

// Algorithm implementations
pub mod ncnpag;
pub mod npag;
pub mod npmap;
pub mod npod;

// Incremental, observable fitting (stepping handle + per-cycle observers).
pub mod controller;

// Re-export per-algorithm configuration structs
pub use error_optim::ErrorOptimConfig;
pub use ncnpag::NcnpagConfig;
pub use npag::NpagConfig;
pub use npmap::NpmapConfig;
pub use npod::NpodConfig;

// Re-export the incremental fitting API
pub use controller::{CycleFlow, FitController, FitObserver};

use crate::algorithms::Algorithm;
use crate::estimation::nonparametric::NonParametricResult;
use crate::estimation::{EstimationProblem, NonParametric};
use anyhow::Result;
use fit::{Fit, FitState, Runner};
use ncnpag::Ncnpag;
use npag::Npag;
use npmap::Npmap;
use npod::Npod;
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
/// Anything that converts into a `NonParametricAlgorithm`, including each configuration type
/// ([`NpagConfig`], [`NpodConfig`], [`NpmapConfig`], [`NcnpagConfig`]), implements
/// [`Algorithm`], so configs can be passed to `fit_with` without converting them first.
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

    /// Set up a fit of `problem` with this algorithm, ready to run its first cycle.
    ///
    /// Both [`fit`](Algorithm::fit) and the stepping
    /// [`FitController`](crate::algorithms::nonparametric::controller::FitController)
    /// build on this.
    pub(crate) fn start<E: Equation>(
        self,
        problem: EstimationProblem<E, NonParametric>,
    ) -> Result<Fit<E>> {
        let state = FitState::new(problem);
        let runner: Box<dyn Runner<E>> = match self {
            Self::Npag(config) => Box::new(Npag::new(config, &state.prior, &state.error_models)),
            Self::Npod(config) => Box::new(Npod::new(config, &state.error_models)),
            Self::Npmap(_) => Box::new(Npmap),
            Self::Ncnpag(config) => Box::new(Ncnpag::new(config)),
        };
        Fit::new(state, runner)
    }
}

/// Every non-parametric algorithm (or its configuration) can be fitted directly.
impl<E, A> Algorithm<E, NonParametric> for A
where
    E: Equation,
    A: Into<NonParametricAlgorithm>,
{
    type Output = NonParametricResult<E>;

    fn fit(self, problem: EstimationProblem<E, NonParametric>) -> Result<Self::Output> {
        let mut fit = self.into().start(problem)?;
        fit.run()?;
        fit.into_result()
    }
}
