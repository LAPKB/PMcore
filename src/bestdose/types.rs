//! Core data types for the BestDose algorithm
//!
//! - [`BestDoseProblem`]: a dose-optimization problem over a parameter distribution.
//! - [`BestDoseOptions`]: optional forecasting settings.
//! - [`BestDoseResult`]: the optimal dosing subject and its cost.
//! - [`Target`]: concentration or AUC target.
//! - [`DoseRange`]: dose constraint specification.

use crate::estimation::nonparametric::{Theta, Weights};
use pharmsol::prelude::*;
use pharmsol::Equation;
use serde::{Deserialize, Serialize};

/// Target type for dose optimization.
///
/// Specifies whether the optimization targets drug concentrations at specific
/// times or Area Under the Curve (AUC) values.
///
/// # AUC Calculation Methods
///
/// - [`Target::AUCFromZero`]: integrates from time 0 to the observation time.
/// - [`Target::AUCFromLastDose`]: integrates from the most recent dose before
///   each observation to the observation time (e.g. steady-state AUCτ).
///
/// Both AUC methods use the trapezoidal rule on a dense time grid controlled by
/// [`BestDoseOptions::prediction_interval`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Target {
    /// Target concentrations at observation times.
    ///
    /// The optimizer finds doses to achieve specified concentration values at
    /// the observation times in the target subject.
    Concentration,

    /// Target cumulative AUC values from time 0.
    ///
    /// The optimizer finds doses to achieve specified cumulative AUC values
    /// calculated from the beginning of the dosing regimen (time 0).
    AUCFromZero,

    /// Target interval AUC values from the last dose to each observation.
    ///
    /// For each observation at time `t`, finds the most recent bolus/infusion
    /// before `t` and integrates from that dose time to `t`. If no dose exists
    /// before `t`, integrates from time 0.
    AUCFromLastDose,
}

/// Allowable dose range for optimization.
///
/// Doses outside `[min, max]` are penalized by the cost function, constraining
/// the optimizer to search within the range. The default range is
/// `[0.0, f64::MAX]` (effectively unbounded).
#[derive(Debug, Clone)]
pub struct DoseRange {
    pub(crate) min: f64,
    pub(crate) max: f64,
}

impl DoseRange {
    pub fn new(min: f64, max: f64) -> Self {
        DoseRange { min, max }
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
    }
}

impl Default for DoseRange {
    fn default() -> Self {
        DoseRange {
            min: 0.0,
            max: f64::MAX,
        }
    }
}

/// Optional forecasting settings for [`BestDoseProblem::optimize`].
///
/// Constructed with [`Default`] and overridden field-by-field, e.g.
/// `BestDoseOptions { prediction_interval: 0.1 }`.
#[derive(Debug, Clone)]
pub struct BestDoseOptions {
    /// Spacing of the dense grid used for AUC integration, in the model's time
    /// units (the same units as observation times and dose times).
    ///
    /// Smaller values give more accurate trapezoidal AUC at higher cost. Only
    /// used for the AUC targets; ignored for [`Target::Concentration`].
    pub prediction_interval: f64,
}

impl Default for BestDoseOptions {
    fn default() -> Self {
        Self {
            prediction_interval: 0.1,
        }
    }
}

/// A dose-optimization problem over a parameter distribution.
///
/// A `BestDoseProblem` pairs a model with a distribution over its parameters —
/// the support points ([`Theta`]) and their probability [`Weights`]. The
/// distribution is supplied by the caller: it may be a population fit, or a
/// patient-specific posterior computed with NCNPAG/NPMAP from past data.
///
/// Call [`optimize`](Self::optimize) to solve for the doses that best hit a
/// target profile.
///
/// # Example
/// ```rust,no_run,ignore
/// let problem = BestDoseProblem::new(eq, theta, weights)?;
/// let result = problem.optimize(
///     target,
///     Target::Concentration,
///     DoseRange::new(0.0, 300.0),
///     0.5,
///     BestDoseOptions::default(),
/// )?;
/// let optimal_subject = result.subject();
/// let cost = result.cost();
/// ```
#[derive(Clone)]
pub struct BestDoseProblem<E: Equation> {
    pub(crate) eq: E,
    pub(crate) theta: Theta,
    pub(crate) weights: Weights,
}

impl<E: Equation> BestDoseProblem<E> {
    /// Creates a dose-optimization problem from a model and a parameter
    /// distribution.
    ///
    /// Returns an error if the number of weights does not match the number of
    /// support points in `theta`.
    pub fn new(eq: E, theta: Theta, weights: Weights) -> anyhow::Result<Self> {
        if weights.len() != theta.matrix().nrows() {
            anyhow::bail!(
                "number of weights ({}) does not match the number of support points ({})",
                weights.len(),
                theta.matrix().nrows()
            );
        }
        Ok(Self { eq, theta, weights })
    }

    /// Solves for the optimal doses that hit `target`.
    ///
    /// Doses in the `target` subject with amount `0.0` are optimized; non-zero
    /// doses stay fixed. `bias` (λ ∈ [0, 1]) trades off minimizing the expected
    /// squared error over the distribution (λ = 0) against minimizing the error
    /// of the weighted-mean prediction (λ = 1).
    ///
    /// Returns the optimal dosing [`Subject`] and its cost. Predictions can be
    /// generated afterwards by simulating the returned subject.
    pub fn optimize(
        &self,
        target: Subject,
        target_type: Target,
        dose_range: DoseRange,
        bias: f64,
        options: BestDoseOptions,
    ) -> anyhow::Result<BestDoseResult> {
        let objective = BestDoseObjective {
            eq: self.eq.clone(),
            theta: self.theta.clone(),
            weights: self.weights.clone(),
            target,
            target_type,
            doserange: dose_range,
            bias_weight: bias,
            prediction_interval: options.prediction_interval,
        };
        crate::bestdose::optimization::optimize(&objective)
    }

    /// The support points of the distribution.
    pub fn theta(&self) -> &Theta {
        &self.theta
    }

    /// The probability weights of the distribution.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }
}

/// Internal objective assembled by [`BestDoseProblem::optimize`]. Holds
/// everything the Nelder-Mead cost function needs to evaluate a candidate dose
/// regimen.
#[derive(Clone)]
pub(crate) struct BestDoseObjective<E: Equation> {
    pub(crate) eq: E,
    pub(crate) theta: Theta,
    pub(crate) weights: Weights,
    pub(crate) target: Subject,
    pub(crate) target_type: Target,
    pub(crate) doserange: DoseRange,
    pub(crate) bias_weight: f64,
    pub(crate) prediction_interval: f64,
}

/// How well the optimal doses hit a single target observation.
///
/// `achieved` is the expected value under the distribution (the weighted mean
/// prediction across support points) — a concentration for
/// [`Target::Concentration`] or an AUC for the AUC targets.
#[derive(Debug, Clone, Copy)]
pub struct Achievement {
    /// Observation time.
    pub time: f64,
    /// Output equation index of the observation.
    pub outeq: usize,
    /// The requested target value at this observation.
    pub target: f64,
    /// The expected achieved value at the optimal doses.
    pub achieved: f64,
}

/// Result of a BestDose optimization: the optimal dosing subject, its cost, and
/// how well each target was achieved.
#[derive(Debug, Clone)]
pub struct BestDoseResult {
    pub(crate) subject: Subject,
    pub(crate) cost: f64,
    pub(crate) achievements: Vec<Achievement>,
}

impl BestDoseResult {
    /// The subject with optimal dose amounts substituted in.
    pub fn subject(&self) -> &Subject {
        &self.subject
    }

    /// The final cost function value (lower is better).
    pub fn cost(&self) -> f64 {
        self.cost
    }

    /// Per-observation targets and their expected achieved values at the optimal
    /// doses, in observation order.
    pub fn achievements(&self) -> &[Achievement] {
        &self.achievements
    }

    /// The optimal dose amounts, in the order they appear in the subject.
    pub fn doses(&self) -> Vec<f64> {
        self.subject
            .iter()
            .flat_map(|occ| {
                occ.events()
                    .iter()
                    .filter_map(|event| match event {
                        Event::Bolus(bolus) => Some(bolus.amount()),
                        Event::Infusion(infusion) => Some(infusion.amount()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}
