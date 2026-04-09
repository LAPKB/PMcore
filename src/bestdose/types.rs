//! Core data types for the BestDose algorithm
//!
//! This module defines the main structures used throughout the BestDose optimization:
//! - [`BestDosePosterior`]: Two-stage API entry point — compute posterior, then optimize
//! - [`BestDoseResult`]: Output structure containing optimal doses and predictions
//! - [`Target`]: Enum specifying concentration or AUC targets
//! - [`DoseRange`]: Dose constraint specification

use std::fmt::Display;

use crate::prelude::*;
use crate::routines::output::predictions::NPPredictions;
use crate::routines::settings::Settings;
use crate::structs::theta::Theta;
use crate::structs::weights::Weights;
use pharmsol::prelude::*;
use serde::{Deserialize, Serialize};

/// Target type for dose optimization
///
/// Specifies whether the optimization targets are drug concentrations at specific times
/// or Area Under the Curve (AUC) values.
///
/// # Examples
///
/// ```rust
/// use pmcore::bestdose::Target;
///
/// // Optimize to achieve target concentrations
/// let target_type = Target::Concentration;
///
/// // Optimize to achieve target cumulative AUC from time 0
/// let target_type = Target::AUCFromZero;
///
/// // Optimize to achieve target interval AUC from last dose
/// let target_type = Target::AUCFromLastDose;
/// ```
///
/// # AUC Calculation Methods
///
/// The algorithm supports two AUC calculation approaches:
///
/// ## AUCFromZero (Cumulative AUC)
/// - Integrates from time 0 to the observation time
/// - Useful for total drug exposure assessment
/// - Formula: `AUC(t) = ∫₀ᵗ C(τ) dτ`
///
/// ## AUCFromLastDose (Interval AUC)
/// - Integrates from the last dose time to the observation time
/// - Useful for steady-state dosing intervals (e.g., AUCτ)
/// - Formula: `AUC(t) = ∫ₜ_last_dose^t C(τ) dτ`
/// - Automatically finds the most recent bolus/infusion before each observation
///
/// Both methods use trapezoidal rule on a dense time grid controlled by `settings.predictions().idelta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Target {
    /// Target concentrations at observation times
    ///
    /// The optimizer finds doses to achieve specified concentration values
    /// at the observation times in the target subject.
    ///
    /// # Example Target Subject
    /// ```rust,ignore
    /// let target = Subject::builder("patient")
    ///     .bolus(0.0, 100.0, 0)        // Dose to optimize
    ///     .observation(12.0, 10.0, 0)  // Target: 10 mg/L at 12h
    ///     .observation(24.0, 5.0, 0)   // Target: 5 mg/L at 24h
    ///     .build();
    /// ```
    Concentration,

    /// Target cumulative AUC values from time 0
    ///
    /// The optimizer finds doses to achieve specified cumulative AUC values
    /// calculated from the beginning of the dosing regimen (time 0).
    ///
    /// # Example Target Subject
    /// ```rust,ignore
    /// let target = Subject::builder("patient")
    ///     .bolus(0.0, 100.0, 0)         // Dose to optimize
    ///     .bolus(12.0, 100.0, 0)        // Second dose to optimize
    ///     .observation(24.0, 400.0, 0)  // Target: AUC₀₋₂₄ = 400 mg·h/L
    ///     .build();
    /// ```
    ///
    /// # Time Grid Resolution
    ///
    /// Control the time grid density via settings:
    /// ```rust,ignore
    /// settings.predictions().idelta = 15;  // 15-minute intervals
    /// ```
    AUCFromZero,

    /// Target interval AUC values from last dose to observation
    ///
    /// The optimizer finds doses to achieve specified interval AUC values
    /// calculated from the most recent dose before each observation.
    /// This is particularly useful for steady-state dosing intervals (AUCτ).
    ///
    /// # Example Target Subject
    /// ```rust,ignore
    /// let target = Subject::builder("patient")
    ///     .bolus(0.0, 200.0, 0)         // Loading dose (fixed at 200 mg)
    ///     .bolus(12.0, 0.0, 0)          // Maintenance dose to optimize
    ///     .observation(24.0, 200.0, 0)  // Target: AUC₁₂₋₂₄ = 200 mg·h/L
    ///     .build();
    /// // The observation at t=24h targets AUC from t=12h (last dose) to t=24h
    /// ```
    ///
    /// # Behavior
    ///
    /// For each observation at time t:
    /// - Finds the most recent bolus or infusion before time t
    /// - Calculates AUC from that dose time to t
    /// - If no dose exists before t, integrates from time 0
    ///
    /// This allows different observations to have different integration intervals,
    /// each relative to their respective preceding dose.
    AUCFromLastDose,
}

/// Allowable dose range constraints
///
/// Specifies minimum and maximum allowable doses for optimization.
/// The Nelder-Mead optimizer will search within these bounds via penalty-based
/// constraint enforcement.
///
/// # Bounds Enforcement
///
/// When candidate doses violate the bounds, the cost function returns a large
/// penalty value proportional to the violation distance. This effectively
/// constrains the Nelder-Mead simplex to remain within the valid range.
///
/// # Examples
///
/// ```rust,ignore
/// use pmcore::bestdose::DoseRange;
///
/// // Large range: 0-1000 mg
/// let range = DoseRange::new(0.0, 1000.0);
///
/// // Narrow range: 50-150 mg
/// let range = DoseRange::new(50.0, 150.0);
///
/// // Access bounds
/// assert_eq!(range.min(), 0.0);
/// assert_eq!(range.max(), 1000.0);
/// ```
///
/// # Clinical Considerations
///
/// - Set bounds appropriate for your drug's clinical use
/// - Consider patient-specific factors (weight, renal function, etc.)
/// - If optimization hits a bound, consider widening the range
/// - Monitor the cost function value - sudden increases may indicate constraint violation
/// - Default range is `[0.0, f64::MAX]` (effectively unbounded)
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

/// The computed Bayesian posterior for a patient
///
/// This is the main public entry point for the two-stage BestDose API:
///
/// 1. **Stage 1: Posterior computation** ([`BestDosePosterior::compute()`])
///    - NPAGFULL11: Bayesian filtering of prior support points
///    - NPAGFULL: Local refinement of each filtered point
///
/// 2. **Stage 2: Dose optimization** ([`BestDosePosterior::optimize()`])
///    - Dual optimization (posterior vs uniform weights)
///    - Final predictions with optimal doses
///
/// The posterior can be reused across multiple `optimize()` calls with
/// different targets, dose ranges, or bias weights.
///
/// # Example
///
/// ```rust,no_run,ignore
/// use pmcore::bestdose::{BestDosePosterior, Target, DoseRange};
///
/// # fn example(population_theta: pmcore::structs::theta::Theta,
/// #            population_weights: pmcore::structs::weights::Weights,
/// #            past: pharmsol::prelude::Subject,
/// #            target: pharmsol::prelude::Subject,
/// #            eq: pharmsol::prelude::ODE,
/// #            settings: pmcore::routines::settings::Settings)
/// #            -> anyhow::Result<()> {
/// // Stage 1: Compute posterior (expensive, done once)
/// let posterior = BestDosePosterior::compute(
///     &population_theta,
///     &population_weights,
///     Some(past),
///     eq,
///     settings,
/// )?;
///
/// // Stage 2: Optimize doses (can be called multiple times)
/// let result = posterior.optimize(
///     target,
///     None,                            // No time offset
///     DoseRange::new(0.0, 1000.0),
///     0.5,                             // bias_weight
///     Target::Concentration,
/// )?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BestDosePosterior {
    /// Refined posterior support points (from NPAGFULL11 + NPAGFULL)
    pub(crate) theta: Theta,
    /// Posterior probability weights
    pub(crate) posterior: Weights,
    /// Filtered population weights (used for bias term in cost function)
    pub(crate) population_weights: Weights,
    /// Past patient data (stored for use in optimize() with time_offset)
    pub(crate) past_data: Option<Subject>,
    /// PK/PD model
    pub(crate) eq: ODE,
    /// Settings (used for prediction grid, error models, etc.)
    pub(crate) settings: Settings,
}

impl BestDosePosterior {
    /// Get the refined posterior support points (Θ)
    pub fn theta(&self) -> &Theta {
        &self.theta
    }

    /// Get the posterior probability weights
    pub fn posterior_weights(&self) -> &Weights {
        &self.posterior
    }

    /// Get the filtered population weights used for the bias term
    pub fn population_weights(&self) -> &Weights {
        &self.population_weights
    }

    /// Get the number of support points in the posterior
    pub fn n_support_points(&self) -> usize {
        self.theta.matrix().nrows()
    }
}

/// Internal optimization problem (not exposed in public API)
///
/// Contains all data needed for dose optimization.
/// Created internally by [`BestDosePosterior::optimize()`].
#[derive(Debug, Clone)]
pub(crate) struct BestDoseProblem {
    pub(crate) target: Subject,
    pub(crate) target_type: Target,
    pub(crate) population_weights: Weights,
    pub(crate) theta: Theta,
    pub(crate) posterior: Weights,
    pub(crate) eq: ODE,
    pub(crate) settings: Settings,
    pub(crate) doserange: DoseRange,
    pub(crate) bias_weight: f64,
}

/// Result from BestDose optimization
///
/// Contains the optimal doses and associated predictions from running
/// [`BestDosePosterior::optimize()`].
///
/// # Fields
///
/// - `doses`: Optimal dose amount(s) in the same order as doses in target subject
/// - `objf`: Final cost function value at optimal doses
/// - `status`: Optimization status (converged or max iterations)
/// - `predictions`: Concentration-time predictions using optimal doses
/// - `auc_predictions`: AUC values at observation times (only for AUC targets)
/// - `optimization_method`: Which method won: `Posterior` or `Uniform`
///
/// # Interpretation
///
/// ## Optimization Method
///
/// - **`Posterior`**: Patient-specific optimization won (uses posterior weights)
///   - Indicates patient differs from population or has sufficient history
///   - Doses are highly personalized
///
/// - **`Uniform`**: Population-based optimization won (uses uniform weights)
///   - Indicates patient is population-typical or has limited history
///   - Doses are more conservative/robust
///
/// ## Cost Function (`objf`)
///
/// Lower is better. The cost combines variance and bias:
/// ```text
/// Cost = (1-λ) × Variance + λ × Bias²
/// ```
///
/// # Examples
///
/// ## Extracting Results
///
/// ```rust,no_run,ignore
/// # use pmcore::bestdose::{BestDosePosterior, Target, DoseRange, BestDoseResult};
/// # fn example(posterior: BestDosePosterior,
/// #            target: pharmsol::prelude::Subject) -> anyhow::Result<()> {
/// let result = posterior.optimize(
///     target, None, DoseRange::new(0.0, 1000.0), 0.5, Target::Concentration,
/// )?;
///
/// // Single dose
/// println!("Optimal dose: {} mg", result.doses()[0]);
///
/// // Multiple doses
/// for (i, dose) in result.doses().iter().enumerate() {
///     println!("Dose {}: {} mg", i + 1, dose);
/// }
///
/// // Check which method was used
/// println!("Method: {}", result.optimization_method());
///
/// // For AUC targets
/// if let Some(auc_values) = result.auc_predictions() {
///     for (time, auc) in auc_values {
///         println!("AUC at t={:.1}h: {:.1} mg·h/L", time, auc);
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestDoseResult {
    /// Subject with optimal doses
    ///
    /// The [Subject] contains the same events as the target subject,
    /// but with the dose amounts updated to the optimal values.
    pub(crate) optimal_subject: Subject,

    /// Final cost function value
    ///
    /// Lower is better. Represents the weighted combination of variance
    /// (patient-specific error) and bias (deviation from population).
    pub(crate) objf: f64,

    /// Optimization status message
    ///
    /// Examples: "converged", "maximum iterations reached", etc.
    pub(crate) status: BestDoseStatus,

    /// Concentration-time predictions for optimal doses
    ///
    /// Contains predicted concentrations at observation times using the
    /// optimal doses. Predictions use the weights from the winning optimization
    /// method (posterior or uniform).
    pub(crate) preds: NPPredictions,

    /// AUC values at observation times
    ///
    /// Only populated when `target_type` is [`Target::AUC`].
    /// Each tuple contains `(time, cumulative_auc)`.
    ///
    /// For [`Target::Concentration`], this field is `None`.
    pub(crate) auc_predictions: Option<Vec<(f64, f64)>>,

    /// Which optimization method produced the best result
    ///
    /// - `"posterior"`: Patient-specific optimization (uses posterior weights)
    /// - `"uniform"`: Population-based optimization (uses uniform weights)
    ///
    /// The algorithm runs both optimizations and selects the one with lower cost.
    pub(crate) optimization_method: OptimalMethod,
}

impl BestDoseResult {
    /// Get the optimized subject
    pub fn optimal_subject(&self) -> &Subject {
        &self.optimal_subject
    }

    /// Get the dose amounts of the optimized subject
    ///
    /// This includes all doses (bolus and infusion) in the order they appear
    /// in the optimal subject, and returns their amounts as a vector of f64.
    pub fn doses(&self) -> Vec<f64> {
        self.optimal_subject()
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
            .collect::<Vec<f64>>()
    }

    /// Get the objective cost function value
    pub fn objf(&self) -> f64 {
        self.objf
    }

    /// Get the optimization status
    pub fn status(&self) -> &BestDoseStatus {
        &self.status
    }

    /// Get the concentration-time predictions
    pub fn predictions(&self) -> &NPPredictions {
        &self.preds
    }

    /// Get the AUC predictions, if available
    pub fn auc_predictions(&self) -> Option<Vec<(f64, f64)>> {
        self.auc_predictions.clone()
    }

    /// Get the optimization method used
    pub fn optimization_method(&self) -> OptimalMethod {
        self.optimization_method
    }
}

/// Optimization method used in BestDose
///
/// This returns the type of optimization method that produced the best result:
/// - `Posterior`: Patient-specific optimization using posterior weights
/// - `Uniform`: Population-based optimization using uniform weights
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq, Eq)]
pub enum OptimalMethod {
    Posterior,
    Uniform,
}

impl Display for OptimalMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimalMethod::Posterior => write!(f, "Posterior"),
            OptimalMethod::Uniform => write!(f, "Uniform"),
        }
    }
}

/// Status of the BestDose optimization
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq, Eq)]
pub enum BestDoseStatus {
    Converged,
    MaxIterations,
}
