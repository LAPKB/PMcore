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
//! use pmcore::bestdose::{BestDoseProblem, Target, DoseRange};
//!
//! # fn example(prior_theta: pmcore::structs::theta::Theta,
//! #            prior_weights: pmcore::structs::weights::Weights,
//! #            past_data: pharmsol::prelude::Subject,
//! #            target: pharmsol::prelude::Subject,
//! #            eq: pharmsol::prelude::ODE,
//! #            error_models: pharmsol::prelude::ErrorModels,
//! #            settings: pmcore::routines::settings::Settings)
//! #            -> anyhow::Result<()> {
//! // Create optimization problem
//! let problem = BestDoseProblem::new(
//!     &prior_theta,                    // Population support points from NPAG
//!     &prior_weights,                  // Population probabilities
//!     Some(past_data),                 // Patient history (None = use prior)
//!     target,                          // Future template with targets
//!     eq,                              // PK/PD model
//!     error_models,                    // Error specifications
//!     DoseRange::new(0.0, 1000.0),     // Dose constraints (0-1000 mg)
//!     0.5,                             // bias_weight: 0=personalized, 1=population
//!     settings,                        // NPAG settings
//!     500,                             // NPAGFULL refinement cycles
//!     Target::Concentration,           // Target type
//! )?;
//!
//! // Run optimization
//! let result = problem.optimize()?;
//!
//! // Extract results
//! println!("Optimal dose: {:?} mg", result.dose);
//! println!("Final cost: {}", result.objf);
//! println!("Method: {}", result.optimization_method);  // "posterior" or "uniform"
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
//! # fn example(prior_theta: pmcore::structs::theta::Theta,
//! #            prior_weights: pmcore::structs::weights::Weights,
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
//!     &prior_theta, &prior_weights, Some(past), target, eq, error_models,
//!     DoseRange::new(10.0, 500.0),    // 10-500 mg allowed
//!     0.3,                             // Slight population emphasis
//!     settings, 500, Target::Concentration,
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
//! # fn example(prior_theta: pmcore::structs::theta::Theta,
//! #            prior_weights: pmcore::structs::weights::Weights,
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
//!     &prior_theta, &prior_weights, Some(past), target, eq, error_models,
//!     DoseRange::new(50.0, 300.0),
//!     0.0,                             // Full personalization
//!     settings, 500, Target::AUC,      // AUC target!
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
//! # fn example(prior_theta: pmcore::structs::theta::Theta,
//! #            prior_weights: pmcore::structs::weights::Weights,
//! #            target: pharmsol::prelude::Subject,
//! #            eq: pharmsol::prelude::ODE,
//! #            error_models: pharmsol::prelude::ErrorModels,
//! #            settings: pmcore::routines::settings::Settings)
//! #            -> anyhow::Result<()> {
//! // No patient history - use population prior directly
//! let problem = BestDoseProblem::new(
//!     &prior_theta, &prior_weights,
//!     None,                            // No past data
//!     target, eq, error_models,
//!     DoseRange::new(0.0, 1000.0),
//!     1.0,                             // Full population weighting
//!     settings,
//!     0,                               // Skip refinement
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
//!   - `Target::AUC`: Cumulative AUC targets
//!
//! ## Performance Tuning
//!
//! For faster optimization:
//! ```rust,no_run,ignore
//! # use pmcore::bestdose::{BestDoseProblem, Target, DoseRange};
//! # fn example(prior_theta: pmcore::structs::theta::Theta,
//! #            prior_weights: pmcore::structs::weights::Weights,
//! #            target: pharmsol::prelude::Subject,
//! #            eq: pharmsol::ODE,
//! #            error_models: pharmsol::prelude::ErrorModels,
//! #            mut settings: pmcore::routines::settings::Settings)
//! #            -> anyhow::Result<()> {
//! // Reduce refinement cycles
//! let problem = BestDoseProblem::new(
//!     &prior_theta, &prior_weights, None, target, eq, error_models,
//!     DoseRange::new(0.0, 1000.0), 0.5,
//!     settings.clone(),
//!     100,                             // Faster: 100 instead of 500
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

pub mod cost;
mod optimization;
mod posterior;
pub mod predictions;
mod types;

// Re-export public API
pub use types::{BestDoseProblem, BestDoseResult, DoseRange, Target};

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
                Event::Infusion(_) => {
                    // Note: Infusions not currently supported in BestDose
                    // Don't add to mask
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
/// Tuple: (posterior_theta, posterior_weights, filtered_prior_weights, past_subject)
fn calculate_posterior_density(
    prior_theta: &Theta,
    prior_weights: &Weights,
    past_data: Option<&Subject>,
    eq: &ODE,
    error_models: &ErrorModels,
    settings: &Settings,
    max_cycles: usize,
) -> Result<(Theta, Weights, Weights, Subject)> {
    match past_data {
        None => {
            tracing::info!("  No past data → using prior directly");
            Ok((
                prior_theta.clone(),
                prior_weights.clone(),
                prior_weights.clone(),
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
                    prior_theta.clone(),
                    prior_weights.clone(),
                    prior_weights.clone(),
                    past_subject.clone(),
                ))
            } else {
                // Two-step posterior calculation
                tracing::info!("  Past data with observations → calculating two-step posterior");
                tracing::info!("    Step 1.1: NPAGFULL11 (Bayesian filtering)");
                tracing::info!("    Step 1.2: NPAGFULL (local refinement)");

                let past_data_obj = Data::new(vec![past_subject.clone()]);

                let (posterior_theta, posterior_weights, filtered_prior_weights) =
                    posterior::calculate_two_step_posterior(
                        prior_theta,
                        prior_weights,
                        &past_data_obj,
                        eq,
                        error_models,
                        settings,
                        max_cycles,
                    )?;

                Ok((
                    posterior_theta,
                    posterior_weights,
                    filtered_prior_weights,
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
    /// Create a new BestDose problem with automatic posterior calculation
    ///
    /// This is the main entry point for the BestDose algorithm.
    ///
    /// # Algorithm Structure (Matches Flowchart)
    ///
    /// ```text
    /// ┌─────────────────────────────────────────┐
    /// │ STAGE 1: Posterior Density Calculation  │
    /// │                                         │
    /// │  Prior Density (N points)              │
    /// │      ↓                                 │
    /// │  Has past data with observations?      │
    /// │      ↓ Yes          ↓ No              │
    /// │  Step 1.1:      Use prior             │
    /// │  NPAGFULL11     directly               │
    /// │  (Filter)                              │
    /// │      ↓                                 │
    /// │  Step 1.2:                             │
    /// │  NPAGFULL                              │
    /// │  (Refine)                              │
    /// │      ↓                                 │
    /// │  Posterior Density                     │
    /// └─────────────────────────────────────────┘
    /// ```
    ///
    /// # Parameters
    ///
    /// * `prior_theta` - Population support points from NPAG
    /// * `prior_weights` - Population probabilities
    /// * `past_data` - Patient history (None = use prior directly)
    /// * `target` - Future dosing template with targets
    /// * `time_offset` - Optional time offset for concatenation (None = standard mode, Some(t) = Fortran mode)
    /// * `eq` - Pharmacokinetic/pharmacodynamic model
    /// * `error_models` - Error model specifications
    /// * `doserange` - Allowable dose constraints
    /// * `bias_weight` - λ ∈ [0,1]: 0=personalized, 1=population
    /// * `settings` - NPAG settings for posterior refinement
    /// * `max_cycles` - NPAGFULL cycles (0=skip refinement, 500=default)
    /// * `target_type` - Concentration or AUC targets
    ///
    /// # Returns
    ///
    /// BestDoseProblem ready for `optimize()`
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prior_theta: &Theta,
        prior_weights: &Weights,
        past_data: Option<Subject>,
        target: Subject,
        time_offset: Option<f64>,
        eq: ODE,
        error_models: ErrorModels,
        doserange: DoseRange,
        bias_weight: f64,
        settings: Settings,
        max_cycles: usize,
        target_type: Target,
    ) -> Result<Self> {
        tracing::info!("╔══════════════════════════════════════════════════════════╗");
        tracing::info!("║            BestDose Algorithm: STAGE 1                   ║");
        tracing::info!("║           Posterior Density Calculation                  ║");
        tracing::info!("╚══════════════════════════════════════════════════════════╝");

        // Validate input if using past/future separation mode
        if let Some(t) = time_offset {
            validate_time_offset(t, &past_data)?;
        }

        // ═════════════════════════════════════════════════════════════
        // STAGE 1: Calculate Posterior Density
        // ═════════════════════════════════════════════════════════════
        let (posterior_theta, posterior_weights, filtered_prior_weights, past_subject) =
            calculate_posterior_density(
                prior_theta,
                prior_weights,
                past_data.as_ref(),
                &eq,
                &error_models,
                &settings,
                max_cycles,
            )?;

        // Handle past/future concatenation if needed
        let (final_target, final_past_data) =
            prepare_target_subject(past_subject, target, time_offset)?;

        tracing::info!("╔══════════════════════════════════════════════════════════╗");
        tracing::info!("║              Stage 1 Complete - Ready for Optimization   ║");
        tracing::info!("╚══════════════════════════════════════════════════════════╝");
        tracing::info!("  Support points: {}", posterior_theta.matrix().nrows());
        tracing::info!("  Target type: {:?}", target_type);
        tracing::info!("  Bias weight (λ): {}", bias_weight);

        Ok(BestDoseProblem {
            past_data: final_past_data,
            target: final_target,
            target_type,
            prior_theta: prior_theta.clone(),
            prior_weights: filtered_prior_weights,
            theta: posterior_theta,
            posterior: posterior_weights,
            eq,
            error_models,
            settings,
            doserange,
            bias_weight,
            time_offset,
        })
    }

    /// Run the complete BestDose optimization algorithm
    ///
    /// # Algorithm Flow (Matches Diagram!)
    ///
    /// ```text
    /// ┌─────────────────────────────────────────┐
    /// │ STAGE 1: Posterior Calculation          │
    /// │         [COMPLETED in new()]             │
    /// └────────────┬────────────────────────────┘
    ///              ↓
    /// ┌─────────────────────────────────────────┐
    /// │ STAGE 2: Dual Optimization              │
    /// │                                         │
    /// │  Optimization 1: Posterior Weights      │
    /// │    (Patient-specific)                   │
    /// │      ↓                                  │
    /// │  Result 1: (doses₁, cost₁)             │
    /// │                                         │
    /// │  Optimization 2: Uniform Weights        │
    /// │    (Population-based)                   │
    /// │      ↓                                  │
    /// │  Result 2: (doses₂, cost₂)             │
    /// │                                         │
    /// │  Select: min(cost₁, cost₂)             │
    /// └────────────┬────────────────────────────┘
    ///              ↓
    /// ┌─────────────────────────────────────────┐
    /// │ STAGE 3: Final Predictions              │
    /// │                                         │
    /// │  Calculate predictions with             │
    /// │  optimal doses and winning weights      │
    /// └─────────────────────────────────────────┘
    /// ```
    ///
    /// # Returns
    ///
    /// `BestDoseResult` containing:
    /// - `dose`: Optimal dose amount(s)
    /// - `objf`: Final cost function value
    /// - `preds`: Concentration-time predictions
    /// - `auc_predictions`: AUC values (if target_type is AUC)
    /// - `optimization_method`: "posterior" or "uniform"
    pub fn optimize(self) -> Result<BestDoseResult> {
        tracing::info!("╔══════════════════════════════════════════════════════════╗");
        tracing::info!("║            BestDose Algorithm: STAGE 2 & 3               ║");
        tracing::info!("║        Dual Optimization + Final Predictions             ║");
        tracing::info!("╚══════════════════════════════════════════════════════════╝");

        // STAGE 2 & 3: Dual optimization + predictions
        optimization::dual_optimization(&self)
    }

    /// Set the bias weight (lambda parameter)
    ///
    /// - λ = 0.0 (default): Full personalization (minimize patient-specific variance)
    /// - λ = 0.5: Balanced between individual and population
    /// - λ = 1.0: Population-based (minimize deviation from population mean)
    pub fn with_bias_weight(mut self, weight: f64) -> Self {
        self.bias_weight = weight;
        self
    }
}
