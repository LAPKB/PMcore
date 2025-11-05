//! Core data types for the BestDose algorithm
//!
//! This module defines the main structures used throughout the BestDose optimization:
//! - [`BestDoseProblem`]: The complete optimization problem specification
//! - [`BestDoseResult`]: Output structure containing optimal doses and predictions
//! - [`Target`]: Enum specifying concentration or AUC targets
//! - [`DoseRange`]: Dose constraint specification

use crate::prelude::*;
use crate::routines::output::predictions::NPPredictions;
use crate::routines::settings::Settings;
use crate::structs::theta::Theta;
use crate::structs::weights::Weights;
use pharmsol::prelude::*;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
/// // Standard range: 0-1000 mg
/// let range = DoseRange::new(0.0, 1000.0);
///
/// // Narrow therapeutic window
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

/// The BestDose optimization problem
///
/// Contains all data needed for the three-stage BestDose algorithm.
/// Create via [`BestDoseProblem::new()`], then call [`.optimize()`](BestDoseProblem::optimize)
/// to run the full algorithm.
///
/// # Three-Stage Algorithm
///
/// 1. **Posterior Density Calculation** (automatic in `new()`)
///    - NPAGFULL11: Bayesian filtering of prior support points
///    - NPAGFULL: Local refinement of each filtered point
///
/// 2. **Dual Optimization** (automatic in `optimize()`)
///    - Optimization with posterior weights (patient-specific)
///    - Optimization with uniform weights (population-based)
///    - Selection of better result
///
/// 3. **Final Predictions** (automatic in `optimize()`)
///    - Concentration or AUC predictions with optimal doses
///
/// # Fields
///
/// ## Input Data
/// - `target`: Future dosing template with target observations
/// - `target_type`: [`Target::Concentration`] or [`Target::AUC`]
///
/// ## Population Prior
/// - `population_weights`: Filtered population probability weights (used for bias term)
///
/// ## Patient-Specific Posterior
/// - `theta`: Refined posterior support points (from NPAGFULL11 + NPAGFULL)
/// - `posterior`: Posterior probability weights
///
/// ## Model Components
/// - `eq`: Pharmacokinetic/pharmacodynamic ODE model
/// - `settings`: NPAG configuration settings (used for prediction grid)
///
/// ## Optimization Parameters
/// - `doserange`: Min/max dose constraints
/// - `bias_weight` (λ): Personalization parameter (0=personalized, 1=population)
///
/// # Example
///
/// ```rust,no_run,ignore
/// use pmcore::bestdose::{BestDoseProblem, Target, DoseRange};
///
/// # fn example(population_theta: pmcore::structs::theta::Theta,
/// #            population_weights: pmcore::structs::weights::Weights,
/// #            past: pharmsol::prelude::Subject,
/// #            target: pharmsol::prelude::Subject,
/// #            eq: pharmsol::prelude::ODE,
/// #            error_models: pharmsol::prelude::ErrorModels,
/// #            settings: pmcore::routines::settings::Settings)
/// #            -> anyhow::Result<()> {
/// let problem = BestDoseProblem::new(
///     &population_theta,
///     &population_weights,
///     Some(past),                      // Patient history
///     target,                          // Dosing template with targets
///     eq,
///     error_models,
///     DoseRange::new(0.0, 1000.0),
///     0.5,                             // Balanced personalization
///     settings,
///     500,                             // NPAGFULL cycles
///     Target::Concentration,
/// )?;
///
/// let result = problem.optimize()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BestDoseProblem {
    /// Target subject with dosing template and target observations
    ///
    /// This [Subject] defines the targets for optimization, including
    /// dose events (with amounts to be optimized) and observation events
    /// (with desired target values).
    ///
    /// For a `Target::Concentration`, observation values are target concentrations.
    /// For a `Target::AUC`, observation values are target cumulative AUC.
    ///
    /// Only doses with a value of `0.0` will be optimized; non-zero doses remain fixed.
    pub(crate) target: Subject,
    /// Target type for optimization
    ///
    /// Specifies whether to optimize for concentrations or AUC values.
    pub(crate) target_type: Target,

    /// The population prior weights ([Weights]), representing the probability of each support point in the population.
    pub(crate) population_weights: Weights,

    // Patient-specific posterior (from NPAGFULL11 + NPAGFULL)
    pub(crate) theta: Theta,
    pub(crate) posterior: Weights,

    // Model and settings
    pub(crate) eq: ODE,
    pub(crate) settings: Settings,

    // Optimization parameters
    pub(crate) doserange: DoseRange,
    pub(crate) bias_weight: f64, // λ: 0=personalized, 1=population
}

/// Result from BestDose optimization
///
/// Contains the optimal doses and associated predictions from running
/// [`BestDoseProblem::optimize()`].
///
/// # Fields
///
/// - `dose`: Optimal dose amount(s) in the same order as doses in target subject
/// - `objf`: Final cost function value at optimal doses
/// - `status`: Optimization status message (e.g., "converged", "max iterations")
/// - `preds`: Concentration-time predictions using optimal doses
/// - `auc_predictions`: AUC values at observation times (only for [`Target::AUC`])
/// - `optimization_method`: Which method won: `"posterior"` or `"uniform"`
///
/// # Interpretation
///
/// ## Optimization Method
///
/// - **"posterior"**: Patient-specific optimization won (uses posterior weights)
///   - Indicates patient differs from population or has sufficient history
///   - Doses are highly personalized
///
/// - **"uniform"**: Population-based optimization won (uses uniform weights)
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
/// # use pmcore::bestdose::BestDoseProblem;
/// # fn example(problem: BestDoseProblem) -> anyhow::Result<()> {
/// let result = problem.optimize()?;
///
/// // Single dose
/// println!("Optimal dose: {} mg", result.dose[0]);
///
/// // Multiple doses
/// for (i, &dose) in result.dose.iter().enumerate() {
///     println!("Dose {}: {} mg", i + 1, dose);
/// }
///
/// // Check which method was used
/// match result.optimization_method.as_str() {
///     "posterior" => println!("Patient-specific optimization"),
///     "uniform" => println!("Population-based optimization"),
///     _ => {}
/// }
///
/// // Access predictions
/// for pred in result.preds.iter() {
///     println!("t={:.1}h: {:.2} mg/L", pred.time(), pred.prediction());
/// }
///
/// // For AUC targets
/// if let Some(auc_values) = result.auc_predictions {
///     for (time, auc) in auc_values {
///         println!("AUC at t={:.1}h: {:.1} mg·h/L", time, auc);
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BestDoseResult {
    /// Optimal dose amount(s)
    ///
    /// Vector contains one element per dose in the target subject.
    /// Order matches the dose events in the target subject.
    pub optimal_subject: Subject,

    /// Final cost function value
    ///
    /// Lower is better. Represents the weighted combination of variance
    /// (patient-specific error) and bias (deviation from population).
    pub objf: f64,

    /// Optimization status message
    ///
    /// Examples: "converged", "maximum iterations reached", etc.
    pub status: String,

    /// Concentration-time predictions for optimal doses
    ///
    /// Contains predicted concentrations at observation times using the
    /// optimal doses. Predictions use the weights from the winning optimization
    /// method (posterior or uniform).
    pub preds: NPPredictions,

    /// AUC values at observation times
    ///
    /// Only populated when `target_type` is [`Target::AUC`].
    /// Each tuple contains `(time, cumulative_auc)`.
    ///
    /// For [`Target::Concentration`], this field is `None`.
    pub auc_predictions: Option<Vec<(f64, f64)>>,

    /// Which optimization method produced the best result
    ///
    /// - `"posterior"`: Patient-specific optimization (uses posterior weights)
    /// - `"uniform"`: Population-based optimization (uses uniform weights)
    ///
    /// The algorithm runs both optimizations and selects the one with lower cost.
    pub optimization_method: String,
}
