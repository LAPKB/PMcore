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
//! - [`POSTPROB`](postprob): Posterior probability reweighting
//!
//! # Algorithm Selection
//!
//! Use the [`NonParametricAlgorithm`] enum to select and configure an algorithm. Each
//! variant wraps its algorithm-specific configuration struct (e.g. [`NpagConfig`]). The
//! internal execution state used while fitting implements the [`NonParametricRunner`]
//! trait, which defines the common interface for initialization, estimation, condensation,
//! expansion, and convergence evaluation.

// Algorithm implementations
pub mod npag;
pub mod npmap;
pub mod npod;

// Re-export algorithm structs
pub use npag::NPAG;
pub use npmap::NPMAP;
pub use npod::NPOD;

// Re-export per-algorithm configuration structs
pub use npag::NpagConfig;
pub use npmap::NpmapConfig;
pub use npod::NpodConfig;

use crate::algorithms::{Algorithm, NonParametricRunner};
use crate::estimation::nonparametric::NonParametricResult;
use crate::estimation::{EstimationProblem, NonParametric};
use anyhow::Result;
use pharmsol::prelude::simulator::Equation;

/// The non-parametric algorithms supported by PMcore.
///
/// Each variant wraps the configuration for one algorithm. Construct a variant with its
/// default configuration using [`npag`](Self::npag), [`npod`](Self::npod), or
/// [`npmap`](Self::npmap), then refine it with the chainable setters:
///
/// ```no_run
/// use pmcore::prelude::*;
///
/// // Default NPAG configuration.
/// let algorithm = NonParametricAlgorithm::npag();
///
/// // NPAG with a tighter convergence criterion and a cycle cap.
/// let algorithm = NonParametricAlgorithm::npag().eps(0.1).max_cycles(500);
/// ```
///
/// The setters are passthroughs to the wrapped configuration. A setter that does not apply
/// to the active variant (for example calling [`eps`](Self::eps) on an [`Npod`](Self::Npod)
/// value) is a no-op.
#[derive(Debug, Clone)]
pub enum NonParametricAlgorithm {
    /// Non-Parametric Adaptive Grid.
    Npag(NpagConfig),
    /// Non-Parametric Optimal Design.
    Npod(NpodConfig),
    /// Non-parametric maximum a posteriori (posterior probability reweighting).
    Npmap(NpmapConfig),
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

    /// Maximum number of cycles to run.
    ///
    /// Applies to the [`Npag`](Self::Npag) and [`Npod`](Self::Npod) variants.
    pub fn max_cycles(mut self, cycles: usize) -> Self {
        match &mut self {
            Self::Npag(config) => config.max_cycles = cycles,
            Self::Npod(config) => config.max_cycles = cycles,
            Self::Npmap(_) => {}
        }
        self
    }

    /// Whether to print progress information during fitting.
    ///
    /// Applies to the [`Npag`](Self::Npag) and [`Npod`](Self::Npod) variants.
    pub fn progress(mut self, progress: bool) -> Self {
        match &mut self {
            Self::Npag(config) => config.progress = progress,
            Self::Npod(config) => config.progress = progress,
            Self::Npmap(_) => {}
        }
        self
    }

    /// Initial convergence criterion (NPAG only).
    pub fn eps(mut self, eps: f64) -> Self {
        if let Self::Npag(config) = &mut self {
            config.eps = eps;
        }
        self
    }

    /// Minimum convergence criterion (NPAG only).
    pub fn min_eps(mut self, min_eps: f64) -> Self {
        if let Self::Npag(config) = &mut self {
            config.min_eps = min_eps;
        }
        self
    }

    /// Objective-function convergence tolerance (NPAG only).
    pub fn objective_tolerance(mut self, tolerance: f64) -> Self {
        if let Self::Npag(config) = &mut self {
            config.objective_tolerance = tolerance;
        }
        self
    }

    /// P(Y|L) convergence tolerance (NPAG only).
    pub fn pyl_tolerance(mut self, tolerance: f64) -> Self {
        if let Self::Npag(config) = &mut self {
            config.pyl_tolerance = tolerance;
        }
        self
    }

    /// Support-point pruning threshold (NPAG only).
    pub fn prune_threshold(mut self, threshold: f64) -> Self {
        if let Self::Npag(config) = &mut self {
            config.prune_threshold = threshold;
        }
        self
    }

    /// QR decomposition tolerance (NPAG only).
    pub fn qr_tolerance(mut self, tolerance: f64) -> Self {
        if let Self::Npag(config) = &mut self {
            config.qr_tolerance = tolerance;
        }
        self
    }

    /// Adaptive-grid tolerance (NPAG only).
    pub fn grid_tolerance(mut self, tolerance: f64) -> Self {
        if let Self::Npag(config) = &mut self {
            config.grid_tolerance = tolerance;
        }
        self
    }

    /// Initial error-model step size (NPAG only).
    pub fn error_step(mut self, step: f64) -> Self {
        if let Self::Npag(config) = &mut self {
            config.error_step = step;
        }
        self
    }

    /// Minimum error-model step size (NPAG only).
    pub fn min_error_step(mut self, step: f64) -> Self {
        if let Self::Npag(config) = &mut self {
            config.min_error_step = step;
        }
        self
    }

    /// Error-model step growth factor (NPAG only).
    pub fn error_step_growth(mut self, factor: f64) -> Self {
        if let Self::Npag(config) = &mut self {
            config.error_step_growth = factor;
        }
        self
    }

    /// Error-model step shrink factor (NPAG only).
    pub fn error_step_shrink(mut self, factor: f64) -> Self {
        if let Self::Npag(config) = &mut self {
            config.error_step_shrink = factor;
        }
        self
    }
}

impl<E: Equation + Send + 'static> Algorithm<E, NonParametric> for NonParametricAlgorithm {
    type Output = NonParametricResult<E>;

    fn fit(self, problem: EstimationProblem<E, NonParametric>) -> Result<Self::Output> {
        match self {
            Self::Npag(config) => {
                // `problem.prior` is the prior `Theta` (which also carries the parameter
                // space) and `problem.error_models` is strictly `AssayErrorModels`.
                let mut runner = NPAG::from_parts(
                    problem.model.equation,
                    problem.data,
                    problem.error_models,
                    problem.prior,
                    config,
                )?;
                NonParametricRunner::fit(&mut runner)
            }
            Self::Npod(config) => {
                let mut runner = NPOD::from_parts(
                    problem.model.equation,
                    problem.data,
                    problem.error_models,
                    problem.prior,
                    config,
                )?;
                NonParametricRunner::fit(&mut runner)
            }
            Self::Npmap(config) => {
                let mut runner = NPMAP::from_parts(
                    problem.model.equation,
                    problem.data,
                    problem.error_models,
                    problem.prior,
                    config,
                )?;
                NonParametricRunner::fit(&mut runner)
            }
        }
    }
}
