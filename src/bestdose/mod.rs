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
//! BestDose is a two-stage API that mirrors the estimation API's ergonomics:
//! compute a reusable posterior once (estimation), then optimize any number of
//! targets against it (forecasting).
//!
//! ```rust,no_run,ignore
//! use pmcore::bestdose::{BestDosePosterior, DoseRange, Prior, Target};
//! use pmcore::prelude::*;
//!
//! # fn example(eq: pharmsol::prelude::ODE,
//! #            error_models: pharmsol::prelude::AssayErrorModels,
//! #            parameter_space: pmcore::prelude::ParameterSpace<pmcore::prelude::BoundedParameter>,
//! #            past_data: pharmsol::prelude::Subject,
//! #            target: pharmsol::prelude::Subject)
//! #            -> anyhow::Result<()> {
//! // The prior carries its own parameter space (loaded from a previous NPAG run).
//! let prior = Prior::from_file("outputs/theta.csv", &parameter_space)?;
//!
//! // Stage 1 — estimation: compute the posterior once (reusable).
//! let posterior = BestDosePosterior::builder(eq, error_models, prior)
//!     .history(Some(past_data)) // None ⇒ use the population prior directly
//!     .refinement_cycles(500)
//!     .compute()?;
//!
//! // Stage 2 — forecasting: optimize doses for a target.
//! let result = posterior
//!     .optimize(target, Target::Concentration)
//!     .dose_range(DoseRange::new(0.0, 1000.0))
//!     .bias(0.5) // λ: 0 = personalized, 1 = population
//!     .run()?;
//!
//! println!("Optimal doses: {:?} mg", result.doses());
//! println!("Final cost: {}", result.objf());
//! println!("Method: {}", result.optimization_method()); // Posterior or Uniform
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
//! # fn example(population_theta: pmcore::estimation::nonparametric::Theta,
//! #            population_weights: pmcore::estimation::nonparametric::Weights,
//! #            past: pharmsol::prelude::Subject,
//! #            eq: pharmsol::prelude::ODE,
//! #            config: pmcore::bestdose::BestDoseConfig)
//! #            -> anyhow::Result<()> {
//! // Define target: 5 mg/L at 24 hours
//! let target = Subject::builder("patient_001")
//!     .bolus(0.0, 100.0, 0)           // Initial dose (will be optimized)
//!     .observation(24.0, 5.0, 0)      // Target: 5 mg/L at 24h
//!     .build();
//!
//! let problem = BestDoseProblem::new(
//!     &population_theta, &population_weights, Some(past), target, None,
//!     eq,
//!     DoseRange::new(10.0, 500.0),    // 10-500 mg allowed
//!     0.3,                             // Slight population emphasis
//!     config, Target::Concentration,
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
//! # fn example(population_theta: pmcore::estimation::nonparametric::Theta,
//! #            population_weights: pmcore::estimation::nonparametric::Weights,
//! #            past: pharmsol::prelude::Subject,
//! #            eq: pharmsol::prelude::ODE,
//! #            config: pmcore::bestdose::BestDoseConfig)
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
//!     eq,
//!     DoseRange::new(50.0, 300.0),
//!     0.0,                             // Full personalization
//!     config, Target::AUCFromZero,     // Cumulative AUC target!
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
//! # fn example(population_theta: pmcore::estimation::nonparametric::Theta,
//! #            population_weights: pmcore::estimation::nonparametric::Weights,
//! #            target: pharmsol::prelude::Subject,
//! #            eq: pharmsol::prelude::ODE,
//! #            config: pmcore::bestdose::BestDoseConfig)
//! #            -> anyhow::Result<()> {
//! // No patient history - use population prior directly
//! let problem = BestDoseProblem::new(
//!     &population_theta, &population_weights,
//!     None,                            // No past data
//!     target, None,                    // time_offset
//!     eq,
//!     DoseRange::new(0.0, 1000.0),
//!     1.0,                             // Full population weighting
//!     config,
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
//! - **`refinement_cycles`**: NPAGFULL refinement iterations
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
//! # fn example(population_theta: pmcore::estimation::nonparametric::Theta,
//! #            population_weights: pmcore::estimation::nonparametric::Weights,
//! #            target: pharmsol::prelude::Subject,
//! #            eq: pharmsol::ODE,
//! #            error_models: pharmsol::prelude::AssayErrorModels,
//! #            parameter_space: pmcore::prelude::ParameterSpace)
//! #            -> anyhow::Result<()> {
//! let config = pmcore::bestdose::BestDoseConfig::new(parameter_space, error_models)
//!     .with_refinement_cycles(100)
//!     .with_prediction_interval(30.0);
//!
//! let problem = BestDoseProblem::new(
//!     &population_theta, &population_weights, None, target, None,
//!     eq,
//!     DoseRange::new(0.0, 1000.0), 0.5,
//!     config,
//!     Target::Concentration,
//! )?;
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
pub use types::{
    BestDosePosterior, BestDosePosteriorBuilder, BestDoseResult, BestDoseStatus, DoseOptimization,
    DoseRange, OptimalMethod, OptimizationStrategy, Prior, Target,
};

// Internal (non-exported) types used to assemble and run the two stages.
use types::{BestDoseConfig, BestDoseProblem};

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

use anyhow::Result;
use pharmsol::prelude::*;
use pharmsol::ODE;

use crate::estimation::nonparametric::{Theta, Weights};

// ═════════════════════════════════════════════════════════════════════════════
// BestDosePosterior: Public two-stage API
// ═════════════════════════════════════════════════════════════════════════════

impl BestDosePosterior {
    /// Begins the estimation stage: build a reusable posterior from a population
    /// [`Prior`] and (optionally) patient history.
    ///
    /// The prior already carries its parameter space, and `error_models` is an
    /// explicit argument — mirroring `EstimationProblem::nonparametric`.
    ///
    /// # Example
    /// ```rust,ignore
    /// let posterior = BestDosePosterior::builder(eq, error_models, prior)
    ///     .history(Some(past_data))
    ///     .refinement_cycles(500)
    ///     .progress(false)
    ///     .compute()?;
    /// ```
    pub fn builder(
        eq: ODE,
        error_models: AssayErrorModels,
        prior: Prior,
    ) -> BestDosePosteriorBuilder {
        BestDosePosteriorBuilder {
            eq,
            prior,
            history: None,
            error_models,
            refinement_cycles: 500,
            progress: true,
        }
    }

    /// Begins the forecasting stage: optimize future doses against this posterior
    /// for a `target`.
    ///
    /// Returns a [`DoseOptimization`] builder; set forecasting options (dose
    /// range, bias, strategy, prediction interval, time offset) and call
    /// [`run`](DoseOptimization::run).
    ///
    /// # Example
    /// ```rust,ignore
    /// let optimal = posterior
    ///     .optimize(target, Target::Concentration)
    ///     .dose_range(DoseRange::new(0.0, 300.0))
    ///     .bias(0.3)
    ///     .run()?;
    /// ```
    pub fn optimize(&self, target: Subject, target_type: Target) -> DoseOptimization<'_> {
        DoseOptimization {
            posterior: self,
            target,
            target_type,
            dose_range: DoseRange::default(),
            bias_weight: 0.0,
            strategy: OptimizationStrategy::default(),
            prediction_interval: 0.12,
            time_offset: None,
        }
    }
}

impl BestDosePosteriorBuilder {
    /// Runs Stage 1: computes the reusable posterior density from the population
    /// prior and patient history.
    pub fn compute(self) -> Result<BestDosePosterior> {
        tracing::info!("Stage 1: posterior density calculation");

        let config = self.config();
        let BestDosePosteriorBuilder {
            eq, prior, history, ..
        } = self;

        let (posterior_theta, posterior_weights, filtered_population_weights, _past_subject) =
            calculate_posterior_density(
                prior.theta(),
                prior.weights(),
                history.as_ref(),
                &eq,
                config.error_models(),
                &config,
            )?;

        tracing::info!(
            "Stage 1 complete: posterior ready with {} support points",
            posterior_theta.matrix().nrows()
        );

        Ok(BestDosePosterior {
            theta: posterior_theta,
            posterior: posterior_weights,
            population_weights: filtered_population_weights,
            past_data: history,
            eq,
        })
    }
}

impl DoseOptimization<'_> {
    /// Runs Stage 2 (dual/posterior optimization) and Stage 3 (final predictions),
    /// returning the optimal dosing result.
    pub fn run(self) -> Result<BestDoseResult> {
        let posterior = self.posterior;

        tracing::info!(
            "Stage 2 & 3: optimization and final predictions (target: {:?}, bias weight: {})",
            self.target_type,
            self.bias_weight
        );

        if let Some(t) = self.time_offset {
            if t < 0.0 {
                return Err(anyhow::anyhow!(
                    "Invalid time_offset: {} is negative. \
                    time_offset must be >= 0 (it represents the gap after the last past event).",
                    t
                ));
            }
        }

        let effective_offset = self.time_offset.map(|t| {
            let max_past_time = posterior
                .past_data
                .as_ref()
                .map(|past| {
                    past.occasions()
                        .iter()
                        .flat_map(|occ| occ.events())
                        .map(|event| match event {
                            Event::Bolus(b) => b.time(),
                            Event::Infusion(i) => i.time(),
                            Event::Observation(o) => o.time(),
                        })
                        .fold(0.0_f64, |max, time| max.max(time))
                })
                .unwrap_or(0.0);
            max_past_time + t
        });

        let final_target = match effective_offset {
            None => self.target,
            Some(eff) => {
                tracing::debug!(
                    "Time offset gap: {:?} hours (effective absolute offset: {} hours)",
                    self.time_offset,
                    eff
                );
                match &posterior.past_data {
                    Some(past) => {
                        tracing::debug!("Concatenating past doses with offset target events");
                        concatenate_past_and_future(past, &self.target, eff)
                    }
                    None => {
                        tracing::debug!("No past data stored — offsetting target events only");
                        concatenate_past_and_future(
                            &Subject::builder("empty").build(),
                            &self.target,
                            eff,
                        )
                    }
                }
            }
        };

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

        let problem = BestDoseProblem {
            target: final_target,
            target_type: self.target_type,
            population_weights: posterior.population_weights.clone(),
            theta: posterior.theta.clone(),
            posterior: posterior.posterior.clone(),
            eq: posterior.eq.clone(),
            doserange: self.dose_range,
            bias_weight: self.bias_weight,
            prediction_interval: self.prediction_interval,
        };

        match self.strategy {
            OptimizationStrategy::Dual => optimization::dual_optimization(&problem),
            OptimizationStrategy::PosteriorOnly => optimization::posterior_optimization(&problem),
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Helper Functions for STAGE 1: Posterior Density Calculation
// ═════════════════════════════════════════════════════════════════════════════

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
    error_models: &AssayErrorModels,
    config: &BestDoseConfig,
) -> Result<(Theta, Weights, Weights, Subject)> {
    match past_data {
        None => {
            tracing::debug!("No past data, using prior directly");
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
                tracing::debug!("Past data has no observations, using prior directly");
                Ok((
                    population_theta.clone(),
                    population_weights.clone(),
                    population_weights.clone(),
                    past_subject.clone(),
                ))
            } else {
                // Two-step posterior calculation
                tracing::debug!("Past data with observations, calculating two-step posterior");

                let past_data_obj = Data::new(vec![past_subject.clone()]);

                let (posterior_theta, posterior_weights, filtered_population_weights) =
                    posterior::calculate_two_step_posterior(
                        population_theta,
                        population_weights,
                        &past_data_obj,
                        eq,
                        error_models,
                        config,
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
