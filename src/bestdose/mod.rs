//! # BestDose Algorithm
//!
//! Optimizes future dosing regimens to achieve target drug concentrations or AUCs.
//!
//! ## Algorithm Overview (Three Stages)
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
//! ## Usage Example
//!
//! ```rust,no_run
//! use pmcore::bestdose::BestDoseProblem;
//!
//! // Create problem with prior, past data, and targets
//! let problem = BestDoseProblem::new(
//!     &prior_theta,
//!     &prior_weights,
//!     Some(past_data),
//!     target_subject,
//!     eq,
//!     error_models,
//!     dose_range,
//!     bias_weight,
//!     settings,
//!     max_cycles,
//!     target_type,
//! )?;
//!
//! // Run complete three-stage algorithm
//! let result = problem.optimize()?;
//!
//! println!("Optimal dose: {:?}", result.dose);
//! println!("Method used: {}", result.optimization_method);
//! ```

mod cost;
mod optimization;
mod posterior;
mod predictions;
mod types;

// Re-export public API
pub use types::{BestDoseProblem, BestDoseResult, DoseRange, Target};

use anyhow::Result;
use pharmsol::prelude::*;
use pharmsol::ODE;

use crate::routines::settings::Settings;
use crate::structs::theta::Theta;
use crate::structs::weights::Weights;

impl BestDoseProblem {
    /// Create a new BestDose problem with automatic posterior calculation
    ///
    /// This is the main entry point for the BestDose algorithm.
    ///
    /// # Three-Stage Process
    ///
    /// 1. **Posterior Calculation** (if past data exists):
    ///    - NPAGFULL11: Bayesian filtering of prior
    ///    - NPAGFULL: Refinement of each filtered point
    ///
    /// 2. **Dual Optimization** (automatic in `optimize()`):
    ///    - Optimize with posterior weights
    ///    - Optimize with uniform weights
    ///    - Select better result
    ///
    /// 3. **Predictions** (automatic in `optimize()`):
    ///    - Calculate concentration or AUC predictions
    ///    - Return optimal doses and statistics
    ///
    /// # Parameters
    ///
    /// * `prior_theta` - Population support points from NPAG
    /// * `prior_weights` - Population probabilities
    /// * `past_data` - Patient history (None = use prior directly)
    /// * `target` - Future dosing template with targets
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
        eq: ODE,
        error_models: ErrorModels,
        doserange: DoseRange,
        bias_weight: f64,
        settings: Settings,
        max_cycles: usize,
        target_type: Target,
    ) -> Result<Self> {
        tracing::info!("╔══════════════════════════════════════════════════════════╗");
        tracing::info!("║            BestDose Algorithm Initialization             ║");
        tracing::info!("╚══════════════════════════════════════════════════════════╝");

        // Determine if we need posterior calculation
        let (posterior_theta, posterior_weights, filtered_prior_weights, final_past_data) =
            match &past_data {
                None => {
                    tracing::info!("No past data provided → using prior directly");
                    (
                        prior_theta.clone(),
                        prior_weights.clone(),
                        prior_weights.clone(), // No filtering when no past data
                        Subject::builder("Empty").build(),
                    )
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
                        tracing::info!("Past data has no observations → using prior directly");
                        (
                            prior_theta.clone(),
                            prior_weights.clone(),
                            prior_weights.clone(), // No filtering when no observations
                            past_subject.clone(),
                        )
                    } else {
                        // Calculate two-step posterior (NPAGFULL11 + NPAGFULL)
                        tracing::info!("Past data with observations → calculating posterior");
                        let past_data_obj = Data::new(vec![past_subject.clone()]);

                        let (post_theta, post_weights, filt_prior_weights) =
                            posterior::calculate_two_step_posterior(
                                prior_theta,
                                prior_weights,
                                &past_data_obj,
                                &eq,
                                &error_models,
                                &settings,
                                max_cycles,
                            )?;

                        // Return filtered prior weights to match filtered theta rows
                        (
                            post_theta,
                            post_weights,
                            filt_prior_weights,
                            past_subject.clone(),
                        )
                    }
                }
            };

        tracing::info!("Initialization complete:");
        tracing::info!("  Support points: {}", posterior_theta.matrix().nrows());
        tracing::info!("  Target type: {:?}", target_type);
        tracing::info!("  Bias weight (λ): {}", bias_weight);

        Ok(BestDoseProblem {
            past_data: final_past_data,
            target,
            target_type,
            prior_theta: prior_theta.clone(),
            prior_weights: filtered_prior_weights, // Use filtered prior weights that match filtered theta rows
            theta: posterior_theta,
            posterior: posterior_weights,
            eq,
            error_models,
            settings,
            doserange,
            bias_weight,
        })
    }

    /// Run the complete BestDose optimization algorithm
    ///
    /// # Algorithm Flow (Reads Like a Diagram!)
    ///
    /// ```text
    /// Stage 1: Posterior Calculation  [DONE in new()]
    ///     ↓
    /// Stage 2: Dual Optimization      [THIS METHOD]
    ///     Optimization 1: Posterior weights
    ///     Optimization 2: Uniform weights
    ///     Select best result
    ///     ↓
    /// Stage 3: Final Predictions      [THIS METHOD]
    ///     Calculate concentrations/AUCs
    ///     Return optimal doses
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
        tracing::info!("║              BestDose Optimization Starting              ║");
        tracing::info!("╚══════════════════════════════════════════════════════════╝");

        // Stage 2 & 3: Dual optimization + predictions
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

    /// Legacy alias for with_bias_weight
    #[deprecated(since = "0.16.0", note = "Use with_bias_weight instead")]
    pub fn bias(self, weight: f64) -> Self {
        self.with_bias_weight(weight)
    }
}
