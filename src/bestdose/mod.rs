//! # BestDose Algorithm
//!
//! Bayesian dose optimization algorithm that finds optimal dosing regimens to achieve
//! target drug concentrations or cumulative AUC (Area Under the Curve) values.
//!
//! The BestDose algorithm combines Bayesian posterior estimation with dual optimization
//! to balance patient-specific adaptation and population-level robustness.
//!
//! # Quick Start
//!
//! ```rust,no_run,ignore
//! use pmcore::bestdose::{BestDosePosterior, Target, DoseRange};
//!
//! # fn example(population_theta: pmcore::structs::theta::Theta,
//! #            population_weights: pmcore::structs::weights::Weights,
//! #            past_data: pharmsol::prelude::Subject,
//! #            target: pharmsol::prelude::Subject,
//! #            eq: pharmsol::prelude::ODE,
//! #            settings: pmcore::routines::settings::Settings)
//! #            -> anyhow::Result<()> {
//! // Stage 1: Compute posterior from patient history
//! let posterior = BestDosePosterior::compute(
//!     &population_theta,               // Population support points from NPAG
//!     &population_weights,             // Population probabilities
//!     Some(past_data),                 // Patient history (None = use prior)
//!     eq,                              // PK/PD model
//!     settings,                        // NPAG settings
//! )?;
//!
//! // Stage 2 & 3: Optimize doses and get predictions
//! let result = posterior.optimize(
//!     target,                          // Future template with targets
//!     None,                            // time_offset (None = standard mode)
//!     DoseRange::new(0.0, 1000.0),     // Dose constraints (0-1000 mg)
//!     0.5,                             // bias_weight: 0=personalized, 1=population
//!     Target::Concentration,           // Target type
//! )?;
//!
//! // Extract results
//! println!("Optimal dose: {:?} mg", result.doses());
//! println!("Final cost: {}", result.objf());
//! println!("Method: {}", result.optimization_method());
//! # Ok(())
//! # }
//! ```
//!
//! # Algorithm Overview (Three Stages)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ STAGE 1: Posterior Density Calculation                         │
//! │                                                                 │
//! │  Prior (N points)                                              │
//! │      ↓                                                         │
//! │  Step 1.1: NPAGFULL11 - Bayesian Filtering                    │
//! │      Calculate P(data|θᵢ) for each support point              │
//! │      Apply Bayes rule: P(θᵢ|data) ∝ P(data|θᵢ) × P(θᵢ)       │
//! │      Filter: Keep points where P(θᵢ|data) > 1e-100 × max      │
//! │      ↓                                                         │
//! │  Filtered Posterior (M points, typically 5-50)                │
//! │      ↓                                                         │
//! │  Step 1.2: NPAGFULL - Local Refinement                        │
//! │      For each filtered point:                                 │
//! │          Run full NPAG optimization                           │
//! │          Find refined "daughter" point                        │
//! │      ↓                                                         │
//! │  Refined Posterior (M points with NPAGFULL11 weights)         │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ STAGE 2: Dual Optimization                                     │
//! │                                                                 │
//! │  Optimization 1: Posterior Weights (Patient-Specific)          │
//! │      Minimize Cost = (1-λ)×Variance + λ×Bias²                 │
//! │      Using NPAGFULL11 posterior weights                        │
//! │      ↓                                                         │
//! │  Result 1: (doses₁, cost₁)                                     │
//! │                                                                 │
//! │  Optimization 2: Uniform Weights (Population)                  │
//! │      Minimize Cost = (1-λ)×Variance + λ×Bias²                 │
//! │      Using uniform weights (1/M for all points)                │
//! │      ↓                                                         │
//! │  Result 2: (doses₂, cost₂)                                     │
//! │                                                                 │
//! │  Select Best: min(cost₁, cost₂)                                │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ STAGE 3: Final Predictions                                     │
//! │                                                                 │
//! │  Calculate predictions with optimal doses                       │
//! │  For AUC targets: Use dense time grid + trapezoidal rule      │
//! │    - AUCFromZero: Cumulative from time 0                       │
//! │    - AUCFromLastDose: Interval from last dose                  │
//! │  Return: Optimal doses, cost, predictions, method used         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Mathematical Foundation
//!
//! ## Bayesian Posterior
//!
//! The posterior density is calculated via Bayes' rule:
//!
//! ```text
//! P(θ | data) = P(data | θ) × P(θ) / P(data)
//! ```
//!
//! Where:
//! - `P(θ | data)`: Posterior (patient-specific parameters)
//! - `P(data | θ)`: Likelihood (from error model)
//! - `P(θ)`: Prior (from population)
//! - `P(data)`: Normalizing constant
//!
//! ## Cost Function
//!
//! The optimization minimizes a hybrid cost function:
//!
//! ```text
//! Cost = (1-λ) × Variance + λ × Bias²
//! ```
//!
//! **Variance Term** (Patient-Specific Performance):
//! ```text
//! Variance = Σᵢ P(θᵢ|data) × Σⱼ (target[j] - pred[i,j])²
//! ```
//! Expected squared error using posterior weights.
//!
//! **Bias Term** (Population-Level Performance):
//! ```text
//! Bias² = Σⱼ (target[j] - E[pred[j]])²
//! where E[pred[j]] = Σᵢ P(θᵢ) × pred[i,j]
//! ```
//! Squared deviation from population mean prediction using prior weights.
//!
//! **Bias Weight Parameter (λ)**:
//! - `λ = 0.0`: Fully personalized (minimize variance only)
//! - `λ = 0.5`: Balanced hybrid approach
//! - `λ = 1.0`: Population-based (minimize bias from population)
//!
//! # Examples
//!
//! ## Single Dose Optimization
//!
//! ```rust,no_run,ignore
//! use pmcore::bestdose::{BestDoseProblem, Target, DoseRange};
//! use pharmsol::prelude::Subject;
//!
//! # fn example(population_theta: pmcore::structs::theta::Theta,
//! #            population_weights: pmcore::structs::weights::Weights,
//! #            past: pharmsol::prelude::Subject,
//! #            eq: pharmsol::prelude::ODE,
//! #            error_models: pharmsol::prelude::ErrorModels,
//! #            settings: pmcore::routines::settings::Settings)
//! #            -> anyhow::Result<()> {
//! // Define target: 5 mg/L at 24 hours
//! let target = Subject::builder("patient_001")
//!     .bolus(0.0, 100.0, 0)           // Initial dose (will be optimized)
//!     .observation(24.0, 5.0, 0)      // Target: 5 mg/L at 24h
//!     .build();
//!
//! let problem = BestDoseProblem::new(
//!     &population_theta, &population_weights, Some(past), target, None,
//!     eq, error_models,
//!     DoseRange::new(10.0, 500.0),    // 10-500 mg allowed
//!     0.3,                             // Slight population emphasis
//!     settings, Target::Concentration,
//! )?;
//!
//! let result = problem.optimize()?;
//! println!("Optimal dose: {} mg", result.dose[0]);
//! # Ok(())
//! # }
//! ```
//!
//! ## Multiple Doses with AUC Target
//!
//! ```rust,no_run,ignore
//! use pmcore::bestdose::{BestDoseProblem, Target, DoseRange};
//! use pharmsol::prelude::Subject;
//!
//! # fn example(population_theta: pmcore::structs::theta::Theta,
//! #            population_weights: pmcore::structs::weights::Weights,
//! #            past: pharmsol::prelude::Subject,
//! #            eq: pharmsol::prelude::ODE,
//! #            error_models: pharmsol::prelude::ErrorModels,
//! #            settings: pmcore::routines::settings::Settings)
//! #            -> anyhow::Result<()> {
//! // Target: Achieve AUC₂₄ = 400 mg·h/L
//! let target = Subject::builder("patient_002")
//!     .bolus(0.0, 100.0, 0)           // Dose 1 (optimized)
//!     .bolus(12.0, 100.0, 0)          // Dose 2 (optimized)
//!     .observation(24.0, 400.0, 0)    // Target: AUC₂₄ = 400
//!     .build();
//!
//! let problem = BestDoseProblem::new(
//!     &population_theta, &population_weights, Some(past), target, None,
//!     eq, error_models,
//!     DoseRange::new(50.0, 300.0),
//!     0.0,                             // Full personalization
//!     settings, Target::AUCFromZero,   // Cumulative AUC target!
//! )?;
//!
//! let result = problem.optimize()?;
//! println!("Dose 1: {} mg at t=0", result.dose[0]);
//! println!("Dose 2: {} mg at t=12", result.dose[1]);
//! if let Some(auc) = result.auc_predictions {
//!     println!("Predicted AUC₂₄: {} mg·h/L", auc[0].1);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Population-Only Optimization
//!
//! ```rust,no_run,ignore
//! # use pmcore::bestdose::{BestDoseProblem, Target, DoseRange};
//! # fn example(population_theta: pmcore::structs::theta::Theta,
//! #            population_weights: pmcore::structs::weights::Weights,
//! #            target: pharmsol::prelude::Subject,
//! #            eq: pharmsol::prelude::ODE,
//! #            error_models: pharmsol::prelude::ErrorModels,
//! #            settings: pmcore::routines::settings::Settings)
//! #            -> anyhow::Result<()> {
//! // No patient history - use population prior directly
//! let problem = BestDoseProblem::new(
//!     &population_theta, &population_weights,
//!     None,                            // No past data
//!     target, None,                    // time_offset
//!     eq, error_models,
//!     DoseRange::new(0.0, 1000.0),
//!     1.0,                             // Full population weighting
//!     settings,
//!     Target::Concentration,
//! )?;
//!
//! let result = problem.optimize()?;
//! // Returns population-typical dose
//! # Ok(())
//! # }
//! ```
//!
//! # Configuration
//!
//! ## Key Parameters
//!
//! - **`bias_weight` (λ)**: Controls personalization level
//!   - `0.0`: Minimize patient-specific variance (full personalization)
//!   - `1.0`: Minimize deviation from population (robustness)
//!   
//! - **`max_cycles`**: NPAGFULL refinement iterations
//!   - `0`: Skip refinement (use filtered points directly)
//!   - `100-500`: Typical range for refinement
//!   
//! - **`doserange`**: Dose constraints
//!   - Set clinically appropriate bounds for your drug
//!   
//! - **`target_type`**: Optimization target
//!   - `Target::Concentration`: Direct concentration targets
//!   - `Target::AUCFromZero`: Cumulative AUC from time 0
//!   - `Target::AUCFromLastDose`: Interval AUC from last dose
//!
//! ## Performance Tuning
//!
//! For faster optimization:
//! ```rust,no_run,ignore
//! # use pmcore::bestdose::{BestDoseProblem, Target, DoseRange};
//! # fn example(population_theta: pmcore::structs::theta::Theta,
//! #            population_weights: pmcore::structs::weights::Weights,
//! #            target: pharmsol::prelude::Subject,
//! #            eq: pharmsol::ODE,
//! #            error_models: pharmsol::prelude::ErrorModels,
//! #            mut settings: pmcore::routines::settings::Settings)
//! #            -> anyhow::Result<()> {
//! // Reduce refinement cycles
//! let problem = BestDoseProblem::new(
//!     &population_theta, &population_weights, None, target, None,
//!     eq, error_models,
//!     DoseRange::new(0.0, 1000.0), 0.5,
//!     settings.clone(),
//!     Target::Concentration,
//! )?;
//!
//! // For AUC: use coarser time grid
//! settings.predictions().idelta = 30.0;  // 30-minute intervals
//! # Ok(())
//! # }
//! ```
//!
//! # See Also
//!
//! - [`BestDoseProblem`]: Main entry point for optimization
//! - [`BestDoseResult`]: Output structure with optimal doses
//! - [`Target`]: Enum for concentration vs AUC targets
//! - [`DoseRange`]: Dose constraint specification

pub(crate) mod cost;
mod optimization;
mod posterior;
pub(crate) mod predictions;
mod types;

// Re-export public API
pub use types::{BestDosePosterior, BestDoseResult, DoseRange, Target};

/// Helper function to concatenate past and future subjects (Option 3: Fortran MAKETMP approach)
///
/// This mimics Fortran's MAKETMP subroutine logic:
/// 1. Takes doses (only doses, not observations) from past subject
/// 2. Offsets all future subject event times by `time_offset`
/// 3. Combines into single continuous subject
///
/// # Arguments
///
/// * `past` - Subject with past history (only doses will be used)
/// * `future` - Subject template for future (all events: doses + observations)
/// * `time_offset` - Time offset to apply to all future events
///
/// # Returns
///
/// Combined subject with:
/// - Past doses at original times [0, time_offset)
/// - Future doses + observations at offset times [time_offset, ∞)
///
/// # Example
///
/// ```rust,ignore
/// // Past: dose at t=0, observation at t=6 (patient has been on therapy 6 hours)
/// let past = Subject::builder("patient")
///     .bolus(0.0, 500.0, 0)
///     .observation(6.0, 15.0, 0)  // 15 mg/L at 6 hours
///     .build();
///
/// // Future: dose at t=0 (relative), target at t=24 (relative)
/// let future = Subject::builder("patient")
///     .bolus(0.0, 100.0, 0)  // Dose to optimize, will be at t=6 absolute
///     .observation(24.0, 10.0, 0)  // Target at t=30 absolute
///     .build();
///
/// // Concatenate with time_offset = 6.0
/// let combined = concatenate_past_and_future(&past, &future, 6.0);
/// // Result: dose at t=0 (fixed, 500mg), dose at t=6 (optimizable, 100mg initial),
/// //         observation target at t=30 (10 mg/L)
/// ```
fn concatenate_past_and_future(
    past: &pharmsol::prelude::Subject,
    future: &pharmsol::prelude::Subject,
    time_offset: f64,
) -> pharmsol::prelude::Subject {
    use pharmsol::prelude::*;

    let mut builder = Subject::builder(past.id());

    // Add past doses only (skip observations from past)
    for occasion in past.occasions() {
        for event in occasion.events() {
            match event {
                Event::Bolus(bolus) => {
                    builder = builder.bolus(bolus.time(), bolus.amount(), bolus.input());
                }
                Event::Infusion(inf) => {
                    builder =
                        builder.infusion(inf.time(), inf.amount(), inf.input(), inf.duration());
                }
                Event::Observation(_) => {
                    // Skip observations from past (they were already used for posterior)
                }
            }
        }
    }

    // Add future events with time offset
    for occasion in future.occasions() {
        for event in occasion.events() {
            match event {
                Event::Bolus(bolus) => {
                    builder =
                        builder.bolus(bolus.time() + time_offset, bolus.amount(), bolus.input());
                }
                Event::Infusion(inf) => {
                    builder = builder.infusion(
                        inf.time() + time_offset,
                        inf.amount(),
                        inf.input(),
                        inf.duration(),
                    );
                }
                Event::Observation(obs) => {
                    builder = match obs.value() {
                        Some(val) => {
                            builder.observation(obs.time() + time_offset, val, obs.outeq())
                        }
                        None => builder,
                    };
                }
            }
        }
    }

    builder.build()
}

/// Calculate which doses are optimizable based on dose amounts
///
/// Returns a boolean mask where:
/// - `true` = dose amount is 0 (placeholder, optimizable)
/// - `false` = dose amount > 0 (fixed past dose)
///
/// This allows users to specify a combined subject with:
/// - Non-zero doses for past doses (e.g., 500 mg at t=0) - these are fixed
/// - Zero doses as placeholders for future doses (e.g., 0 mg at t=6) - these are optimized
///
/// # Arguments
///
/// * `subject` - The subject with both fixed and placeholder doses
///
/// # Returns
///
/// Vector of booleans, one per dose in the subject
///
/// # Example
///
/// ```rust,ignore
/// let subject = Subject::builder("patient")
///     .bolus(0.0, 500.0, 0)    // Past dose (fixed) - mask[0] = false
///     .bolus(6.0, 0.0, 0)      // Future dose (optimize) - mask[1] = true
///     .observation(30.0, 10.0, 0)
///     .build();
/// let mask = calculate_dose_optimization_mask(&subject);
/// assert_eq!(mask, vec![false, true]);
/// ```
fn calculate_dose_optimization_mask(subject: &pharmsol::prelude::Subject) -> Vec<bool> {
    use pharmsol::prelude::*;

    let mut mask = Vec::new();

    for occasion in subject.occasions() {
        for event in occasion.events() {
            match event {
                Event::Bolus(bolus) => {
                    // Dose is optimizable if amount is 0 (placeholder)
                    mask.push(bolus.amount() == 0.0);
                }
                Event::Infusion(infusion) => {
                    // Infusion is optimizable if amount is 0 (placeholder)
                    mask.push(infusion.amount() == 0.0);
                }
                Event::Observation(_) => {
                    // Observations don't go in the mask
                }
            }
        }
    }

    mask
}

use anyhow::Result;
use pharmsol::prelude::*;
use pharmsol::ODE;

use crate::routines::settings::Settings;
use crate::structs::theta::Theta;
use crate::structs::weights::Weights;

use types::BestDoseProblem;

// ═════════════════════════════════════════════════════════════════════════════
// BestDosePosterior: Public two-stage API
// ═════════════════════════════════════════════════════════════════════════════

impl BestDosePosterior {
    /// **Stage 1**: Compute the Bayesian posterior density from population prior and patient data
    ///
    /// This performs the expensive posterior calculation (NPAGFULL11 filtering + NPAGFULL refinement)
    /// and returns a reusable `BestDosePosterior` that can be optimized multiple times.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// Prior (N support points)
    ///     ↓
    /// NPAGFULL11: Bayesian filtering
    ///     P(θᵢ|data) ∝ P(data|θᵢ) × P(θᵢ)
    ///     ↓
    /// Filtered posterior (M points)
    ///     ↓
    /// NPAGFULL: Local refinement (max_cycles iterations)
    ///     ↓
    /// Refined posterior (M points with updated weights)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `population_theta` - Population support points from NPAG
    /// * `population_weights` - Population probabilities
    /// * `past_data` - Patient history (`None` = use prior directly)
    /// * `eq` - Pharmacokinetic/pharmacodynamic model
    /// * `error_models` - Error model specifications
    /// * `settings` - NPAG settings for posterior refinement
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// let posterior = BestDosePosterior::compute(
    ///     &theta, &weights,
    ///     Some(past_subject),
    ///     eq, error_models, settings,
    /// )?;
    /// println!("Posterior has {} support points", posterior.n_support_points());
    /// ```
    pub fn compute(
        population_theta: &Theta,
        population_weights: &Weights,
        past_data: Option<Subject>,
        eq: ODE,
        settings: Settings,
    ) -> Result<Self> {
        tracing::info!("╔══════════════════════════════════════════════════════════╗");
        tracing::info!("║            BestDose Algorithm: STAGE 1                   ║");
        tracing::info!("║           Posterior Density Calculation                  ║");
        tracing::info!("╚══════════════════════════════════════════════════════════╝");

        let (posterior_theta, posterior_weights, filtered_population_weights, _past_subject) =
            calculate_posterior_density(
                population_theta,
                population_weights,
                past_data.as_ref(),
                &eq,
                &settings.errormodels,
                &settings,
            )?;

        tracing::info!("╔══════════════════════════════════════════════════════════╗");
        tracing::info!("║              Stage 1 Complete - Posterior Ready           ║");
        tracing::info!("╚══════════════════════════════════════════════════════════╝");
        tracing::info!("  Support points: {}", posterior_theta.matrix().nrows());

        Ok(BestDosePosterior {
            theta: posterior_theta,
            posterior: posterior_weights,
            population_weights: filtered_population_weights,
            eq,
            settings,
        })
    }

    /// **Stage 2**: Optimize doses for target outcomes using the computed posterior
    ///
    /// This runs the dual optimization (posterior weights vs uniform weights) and
    /// returns the best dosing regimen. Can be called multiple times on the same
    /// posterior with different parameters.
    ///
    /// # Arguments
    ///
    /// * `target` - Future dosing template with target observations
    /// * `time_offset` - Optional time boundary for past/future concatenation (Fortran mode)
    /// * `dose_range` - Allowable dose constraints
    /// * `bias_weight` - λ ∈ [0,1]: 0=personalized, 1=population
    /// * `target_type` - Concentration or AUC targets
    ///
    /// # Example
    ///
    /// ```rust,no_run,ignore
    /// // Try different bias weights
    /// for &bw in &[0.0, 0.25, 0.5, 0.75, 1.0] {
    ///     let result = posterior.optimize(
    ///         target.clone(),
    ///         None,
    ///         DoseRange::new(0.0, 300.0),
    ///         bw,
    ///         Target::Concentration,
    ///     )?;
    ///     println!("λ={}: dose={:.1}", bw, result.doses()[0]);
    /// }
    /// ```
    pub fn optimize(
        &self,
        target: Subject,
        time_offset: Option<f64>,
        dose_range: DoseRange,
        bias_weight: f64,
        target_type: Target,
    ) -> Result<BestDoseResult> {
        tracing::info!("╔══════════════════════════════════════════════════════════╗");
        tracing::info!("║            BestDose Algorithm: STAGE 2 & 3               ║");
        tracing::info!("║        Dual Optimization + Final Predictions             ║");
        tracing::info!("╚══════════════════════════════════════════════════════════╝");
        tracing::info!("  Target type: {:?}", target_type);
        tracing::info!("  Bias weight (λ): {}", bias_weight);

        // Handle past/future concatenation if needed
        // Note: In the two-stage API, past data was already consumed in compute().
        // The time_offset mode concatenates a dummy empty-past with the target.
        let final_target = match time_offset {
            None => target,
            Some(t) => {
                // When using time_offset without past data in the target itself,
                // we just use the target as-is (the user already built the combined subject)
                tracing::info!("  Time offset: {} (events already combined)", t);
                target
            }
        };

        // Validate that the target has observations
        let has_observations = final_target
            .occasions()
            .iter()
            .flat_map(|occ| occ.events())
            .any(|event| matches!(event, Event::Observation(_)));
        if !has_observations {
            return Err(anyhow::anyhow!(
                "Target subject has no observations. At least one observation is required for dose optimization."
            ));
        }

        // Build the internal optimization problem
        let problem = BestDoseProblem {
            target: final_target,
            target_type,
            population_weights: self.population_weights.clone(),
            theta: self.theta.clone(),
            posterior: self.posterior.clone(),
            eq: self.eq.clone(),
            settings: self.settings.clone(),
            doserange: dose_range,
            bias_weight,
        };

        // Run dual optimization + final predictions
        optimization::dual_optimization(&problem)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Helper Functions for STAGE 1: Posterior Density Calculation
// ═════════════════════════════════════════════════════════════════════════════

/// Validate time_offset parameter for past/future separation mode
fn validate_time_offset(time_offset: f64, past_data: &Option<Subject>) -> Result<()> {
    if let Some(past_subject) = past_data {
        let max_past_time = past_subject
            .occasions()
            .iter()
            .flat_map(|occ| occ.events())
            .map(|event| match event {
                Event::Bolus(b) => b.time(),
                Event::Infusion(i) => i.time(),
                Event::Observation(o) => o.time(),
            })
            .fold(0.0_f64, |max, time| max.max(time));

        if time_offset < max_past_time {
            return Err(anyhow::anyhow!(
                "Invalid time_offset: {} is before the last past_data event at time {}. \
                time_offset must be >= the maximum time in past_data to avoid time travel!",
                time_offset,
                max_past_time
            ));
        }
    }
    Ok(())
}

/// Calculate posterior density (STAGE 1: Two-step process)
///
/// # Algorithm Flow (Matches Diagram)
///
/// ```text
/// Prior Density (N points)
///     ↓
/// Has past data with observations?
///     ↓ Yes              ↓ No
/// Step 1.1:          Use prior
/// NPAGFULL11         directly
/// (Bayesian Filter)
///     ↓
/// Filtered Posterior (M points)
///     ↓
/// Step 1.2:
/// NPAGFULL
/// (Refine each point)
///     ↓
/// Refined Posterior
/// (M points with NPAGFULL11 weights)
/// ```
///
/// # Returns
///
/// Tuple: (posterior_theta, posterior_weights, filtered_population_weights, past_subject)
fn calculate_posterior_density(
    population_theta: &Theta,
    population_weights: &Weights,
    past_data: Option<&Subject>,
    eq: &ODE,
    error_models: &ErrorModels,
    settings: &Settings,
) -> Result<(Theta, Weights, Weights, Subject)> {
    match past_data {
        None => {
            tracing::info!("  No past data → using prior directly");
            Ok((
                population_theta.clone(),
                population_weights.clone(),
                population_weights.clone(),
                Subject::builder("Empty").build(),
            ))
        }
        Some(past_subject) => {
            // Check if past data has observations
            let has_observations = !past_subject.occasions().is_empty()
                && past_subject.occasions().iter().any(|occ| {
                    occ.events()
                        .iter()
                        .any(|e| matches!(e, Event::Observation(_)))
                });

            if !has_observations {
                tracing::info!("  Past data has no observations → using prior directly");
                Ok((
                    population_theta.clone(),
                    population_weights.clone(),
                    population_weights.clone(),
                    past_subject.clone(),
                ))
            } else {
                // Two-step posterior calculation
                tracing::info!("  Past data with observations → calculating two-step posterior");
                tracing::info!("    Step 1.1: NPAGFULL11 (Bayesian filtering)");
                tracing::info!("    Step 1.2: NPAGFULL (local refinement)");

                let past_data_obj = Data::new(vec![past_subject.clone()]);

                let (posterior_theta, posterior_weights, filtered_population_weights) =
                    posterior::calculate_two_step_posterior(
                        population_theta,
                        population_weights,
                        &past_data_obj,
                        eq,
                        error_models,
                        settings,
                    )?;

                Ok((
                    posterior_theta,
                    posterior_weights,
                    filtered_population_weights,
                    past_subject.clone(),
                ))
            }
        }
    }
}

/// Prepare target subject by handling past/future concatenation if needed
///
/// # Returns
///
/// Tuple: (final_target, final_past_data)
fn prepare_target_subject(
    past_subject: Subject,
    target: Subject,
    time_offset: Option<f64>,
) -> Result<(Subject, Subject)> {
    match time_offset {
        None => {
            tracing::info!("  Mode: Standard (single subject)");
            Ok((target, past_subject))
        }
        Some(t) => {
            tracing::info!("  Mode: Past/Future separation (Fortran MAKETMP approach)");
            tracing::info!("  Current time boundary: {} hours", t);
            tracing::info!("  Concatenating past and future subjects...");

            let combined = concatenate_past_and_future(&past_subject, &target, t);

            // Log dose structure
            let mask = calculate_dose_optimization_mask(&combined);
            let num_fixed = mask.iter().filter(|&&x| !x).count();
            let num_optimizable = mask.iter().filter(|&&x| x).count();
            tracing::info!("    Fixed doses (from past): {}", num_fixed);
            tracing::info!("    Optimizable doses (from future): {}", num_optimizable);

            Ok((combined, past_subject))
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════

impl BestDoseProblem {
    /// Create a BestDoseProblem directly (convenience for tests and legacy callers)
    ///
    /// Prefer the two-stage API: `BestDosePosterior::compute()` → `posterior.optimize()`
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        population_theta: &Theta,
        population_weights: &Weights,
        past_data: Option<Subject>,
        target: Subject,
        time_offset: Option<f64>,
        eq: ODE,
        doserange: DoseRange,
        bias_weight: f64,
        settings: Settings,
        target_type: Target,
    ) -> Result<Self> {
        // Validate input if using past/future separation mode
        if let Some(t) = time_offset {
            validate_time_offset(t, &past_data)?;
        }

        let (posterior_theta, posterior_weights, filtered_population_weights, past_subject) =
            calculate_posterior_density(
                population_theta,
                population_weights,
                past_data.as_ref(),
                &eq,
                &settings.errormodels,
                &settings,
            )?;

        let (final_target, _) = prepare_target_subject(past_subject, target, time_offset)?;

        Ok(BestDoseProblem {
            target: final_target,
            target_type,
            population_weights: filtered_population_weights,
            theta: posterior_theta,
            posterior: posterior_weights,
            eq,
            settings,
            doserange,
            bias_weight,
        })
    }

    pub(crate) fn optimize(self) -> Result<BestDoseResult> {
        optimization::dual_optimization(&self)
    }

    pub(crate) fn with_bias_weight(mut self, weight: f64) -> Self {
        self.bias_weight = weight;
        self
    }

    pub(crate) fn posterior_theta(&self) -> &Theta {
        &self.theta
    }

    pub(crate) fn posterior_weights(&self) -> &Weights {
        &self.posterior
    }

    pub(crate) fn population_weights(&self) -> &Weights {
        &self.population_weights
    }

    pub(crate) fn bias_weight(&self) -> f64 {
        self.bias_weight
    }

    pub(crate) fn target_type(&self) -> Target {
        self.target_type
    }
}
