//! Cost function calculation for BestDose optimization
//!
//! Implements the hybrid cost function that balances patient-specific performance
//! (variance) with population-level robustness (bias).
//!
//! # Cost Function
//!
//! ```text
//! Cost = (1-λ) × Variance + λ × Bias²
//! ```
//!
//! ## Variance Term (Patient-Specific)
//!
//! Expected squared prediction error using posterior weights:
//! ```text
//! Variance = Σᵢ posterior_weight[i] × Σⱼ (target[j] - pred[i,j])²
//! ```
//!
//! - Weighted by patient-specific posterior probabilities
//! - Minimizes expected error for this specific patient
//! - Emphasizes parameter values compatible with patient history
//!
//! ## Bias Term (Population-Level)
//!
//! Squared deviation from population mean prediction using prior weights:
//! ```text
//! Bias² = Σⱼ (target[j] - population_mean[j])²
//! where population_mean[j] = Σᵢ prior_weight[i] × pred[i,j]
//! ```
//!
//! - Weighted by population prior probabilities
//! - Minimizes deviation from population-typical behavior
//! - Provides robustness when patient history is limited
//!
//! ## Bias Weight Parameter (λ)
//!
//! - `λ = 0.0`: Pure personalization (minimize variance only)
//! - `λ = 0.5`: Balanced hybrid approach
//! - `λ = 1.0`: Pure population (minimize bias only)
//!
//! # Implementation Notes
//!
//! The cost function handles both concentration and AUC targets:
//! - **Concentration**: Simulates model at observation times directly
//! - **AUC**: Generates dense time grid and calculates cumulative AUC via trapezoidal rule
//!
//! See [`calculate_cost`] for the main implementation.

use anyhow::Result;

use crate::bestdose::predictions::{calculate_auc_at_times, calculate_dense_times};
use crate::bestdose::types::{BestDoseProblem, Target};
use pharmsol::prelude::*;
use pharmsol::Equation;

/// Calculate cost function for a candidate dose regimen
///
/// This is the core objective function minimized by the Nelder-Mead optimizer during
/// Stage 2 of the BestDose algorithm.
///
/// # Arguments
///
/// * `problem` - The [`BestDoseProblem`] containing all necessary data
/// * `candidate_doses` - Dose amounts to evaluate (only for optimizable doses)
///
/// # Returns
///
/// The cost value `(1-λ) × Variance + λ × Bias²` for the candidate doses.
/// Lower cost indicates better match to targets.
///
/// # Dose Masking
///
/// When `problem.current_time` is set (past/future separation), only doses where
/// `dose_optimization_mask[i] == true` are updated with values from `candidate_doses`.
/// Past doses (mask == false) remain at their historical values.
///
/// - **Standard mode**: All doses in `candidate_doses` → all doses updated
/// - **Fortran mode**: Only future doses in `candidate_doses` → only future doses updated
///
/// # Cost Function Details
///
/// ## Variance Term
///
/// Expected squared prediction error using posterior weights:
/// ```text
/// Variance = Σᵢ P(θᵢ|data) × Σⱼ (target[j] - pred[i,j])²
/// ```
///
/// For each support point θᵢ:
/// 1. Simulate model with candidate doses
/// 2. Calculate squared error at each observation time j
/// 3. Weight by posterior probability P(θᵢ|data)
///
/// ## Bias Term
///
/// Squared deviation from population mean:
/// ```text
/// Bias² = Σⱼ (target[j] - E[pred[j]])²
/// where E[pred[j]] = Σᵢ P(θᵢ) × pred[i,j]  (prior weights)
/// ```
///
/// The population mean uses **prior weights**, not posterior weights, to represent
/// population-typical behavior independent of patient-specific data.
///
/// ## Target Types
///
/// - **Concentration** ([`Target::Concentration`]):
///   Predictions are concentrations at observation times
///
/// - **AUC** ([`Target::AUC`]):
///   Predictions are cumulative AUC values calculated via trapezoidal rule
///   on a dense time grid (controlled by `settings.predictions().idelta`)
///
/// # Example
///
/// ```rust,ignore
/// // Internal use by optimizer
/// let cost = calculate_cost(&problem, &[100.0, 150.0])?;
/// ```
///
/// # Errors
///
/// Returns error if:
/// - Model simulation fails
/// - Prediction length doesn't match observation count
/// - AUC calculation fails (for AUC targets)
pub fn calculate_cost(problem: &BestDoseProblem, candidate_doses: &[f64]) -> Result<f64> {
    // Validate candidate_doses length matches expected optimizable dose count
    let expected_optimizable = problem
        .target
        .occasions()
        .iter()
        .flat_map(|occ| occ.events())
        .filter(|event| match event {
            Event::Bolus(b) => b.amount() == 0.0,
            Event::Infusion(inf) => inf.amount() == 0.0,
            _ => false,
        })
        .count();

    if candidate_doses.len() != expected_optimizable {
        return Err(anyhow::anyhow!(
            "Dose count mismatch: received {} candidate doses but expected {} optimizable doses",
            candidate_doses.len(),
            expected_optimizable
        ));
    }

    // Build target subject with candidate doses
    let mut target_subject = problem.target.clone();
    let mut optimizable_dose_number = 0; // Index into candidate_doses

    for occasion in target_subject.iter_mut() {
        for event in occasion.iter_mut() {
            match event {
                Event::Bolus(bolus) => {
                    // Only update if this dose is optimizable (amount == 0)
                    if bolus.amount() == 0.0 {
                        bolus.set_amount(candidate_doses[optimizable_dose_number]);
                        optimizable_dose_number += 1;
                    }
                    // If not optimizable (amount > 0), keep original amount
                }
                Event::Infusion(infusion) => {
                    // Only update if this dose is optimizable (amount == 0)
                    if infusion.amount() == 0.0 {
                        infusion.set_amount(candidate_doses[optimizable_dose_number]);
                        optimizable_dose_number += 1;
                    }
                    // If not optimizable (amount > 0), keep original amount
                }
                Event::Observation(_) => {}
            }
        }
    }

    // Extract target values and observation times
    let obs_times: Vec<f64> = target_subject
        .occasions()
        .iter()
        .flat_map(|occ| occ.events())
        .filter_map(|event| match event {
            Event::Observation(obs) => Some(obs.time()),
            _ => None,
        })
        .collect();

    // Validate that target has observations
    if obs_times.is_empty() {
        return Err(anyhow::anyhow!(
            "Target subject has no observations. At least one observation is required for dose optimization."
        ));
    }

    let obs_vec: Vec<f64> = target_subject
        .occasions()
        .iter()
        .flat_map(|occ| occ.events())
        .filter_map(|event| match event {
            Event::Observation(obs) => obs.value(),
            _ => None,
        })
        .collect();

    let n_obs = obs_vec.len();

    // Accumulators
    let mut variance = 0.0_f64; // Expected squared error E[(target - pred)²]
    let mut y_bar = vec![0.0_f64; n_obs]; // Population mean predictions

    // Calculate variance (using posterior weights) and population mean (using prior weights)

    for ((row, post_prob), prior_prob) in problem
        .theta
        .matrix()
        .row_iter()
        .zip(problem.posterior.iter()) // Posterior from NPAGFULL11 (patient-specific)
        .zip(problem.prior_weights.iter())
    // Prior (population)
    {
        let spp = row.iter().copied().collect::<Vec<f64>>();

        // Get predictions based on target type
        let preds_i: Vec<f64> = match problem.target_type {
            Target::Concentration => {
                // Simulate at observation times only
                let pred = problem.eq.simulate_subject(&target_subject, &spp, None)?;
                pred.0.flat_predictions()
            }
            Target::AUC => {
                // For AUC: simulate at dense time grid and calculate cumulative AUC
                let idelta = problem.settings.predictions().idelta;
                let start_time = 0.0; // Future starts at 0
                let end_time = obs_times.last().copied().unwrap_or(0.0);

                // Generate dense time grid
                let dense_times =
                    calculate_dense_times(start_time, end_time, &obs_times, idelta as usize);

                // Create temporary subject with dense time points for simulation
                let subject_id = target_subject.id().to_string();
                let mut builder = Subject::builder(&subject_id);

                // Add all doses from original subject
                for occasion in target_subject.occasions() {
                    for event in occasion.events() {
                        match event {
                            Event::Bolus(bolus) => {
                                builder =
                                    builder.bolus(bolus.time(), bolus.amount(), bolus.input());
                            }
                            Event::Infusion(infusion) => {
                                builder = builder.infusion(
                                    infusion.time(),
                                    infusion.amount(),
                                    infusion.input(),
                                    infusion.duration(),
                                );
                            }
                            Event::Observation(_) => {} // Skip original observations
                        }
                    }
                }

                // Collect observations with (time, outeq) pairs to preserve original order
                let obs_time_outeq: Vec<(f64, usize)> = target_subject
                    .occasions()
                    .iter()
                    .flat_map(|occ| occ.events())
                    .filter_map(|event| match event {
                        Event::Observation(obs) => Some((obs.time(), obs.outeq())),
                        _ => None,
                    })
                    .collect();

                let mut unique_outeqs: Vec<usize> =
                    obs_time_outeq.iter().map(|(_, outeq)| *outeq).collect();
                unique_outeqs.sort();
                unique_outeqs.dedup();

                // Add observations at dense times (with dummy values for timing only)
                for outeq in unique_outeqs.iter() {
                    for &t in &dense_times {
                        builder = builder.missing_observation(t, *outeq);
                    }
                }

                let dense_subject = builder.build();

                // Simulate at dense times
                let pred = problem.eq.simulate_subject(&dense_subject, &spp, None)?;
                let dense_predictions_with_outeq = pred.0.predictions();

                // Group predictions by outeq using the Prediction struct
                let mut outeq_predictions: std::collections::HashMap<usize, Vec<f64>> =
                    std::collections::HashMap::new();

                for prediction in dense_predictions_with_outeq {
                    outeq_predictions
                        .entry(prediction.outeq())
                        .or_insert_with(Vec::new)
                        .push(prediction.prediction());
                }

                // Calculate AUC for each outeq separately
                let mut outeq_aucs: std::collections::HashMap<usize, Vec<f64>> =
                    std::collections::HashMap::new();

                for &outeq in unique_outeqs.iter() {
                    let outeq_preds = outeq_predictions.get(&outeq).ok_or_else(|| {
                        anyhow::anyhow!("Missing predictions for outeq {}", outeq)
                    })?;

                    // Get observation times for this outeq only
                    let outeq_obs_times: Vec<f64> = obs_time_outeq
                        .iter()
                        .filter(|(_, o)| *o == outeq)
                        .map(|(t, _)| *t)
                        .collect();

                    // Calculate AUC at observation times for this outeq
                    let aucs = calculate_auc_at_times(&dense_times, outeq_preds, &outeq_obs_times);
                    outeq_aucs.insert(outeq, aucs);
                }

                // Build final AUC vector in original observation order
                let mut result_aucs = Vec::with_capacity(obs_time_outeq.len());
                let mut outeq_counters: std::collections::HashMap<usize, usize> =
                    std::collections::HashMap::new();

                for (_, outeq) in obs_time_outeq.iter() {
                    let aucs = outeq_aucs
                        .get(outeq)
                        .ok_or_else(|| anyhow::anyhow!("Missing AUC for outeq {}", outeq))?;

                    let counter = outeq_counters.entry(*outeq).or_insert(0);
                    if *counter < aucs.len() {
                        result_aucs.push(aucs[*counter]);
                        *counter += 1;
                    } else {
                        return Err(anyhow::anyhow!(
                            "AUC index out of bounds for outeq {}",
                            outeq
                        ));
                    }
                }

                result_aucs
            }
        };

        if preds_i.len() != n_obs {
            return Err(anyhow::anyhow!(
                "prediction length ({}) != observation length ({})",
                preds_i.len(),
                n_obs
            ));
        }

        // Calculate variance term: weighted by POSTERIOR probability
        let mut sumsq_i = 0.0_f64;
        for (j, &obs_val) in obs_vec.iter().enumerate() {
            let pj = preds_i[j];
            let se = (obs_val - pj).powi(2);
            sumsq_i += se;
            // Calculate population mean using PRIOR probabilities
            y_bar[j] += prior_prob * pj;
        }

        variance += post_prob * sumsq_i; // Weighted by posterior
    }

    // Calculate bias term: squared difference from population mean
    let mut bias = 0.0_f64;
    for (j, &obs_val) in obs_vec.iter().enumerate() {
        bias += (obs_val - y_bar[j]).powi(2);
    }

    // Final cost: (1-λ)×Variance + λ×Bias²
    // λ=0: Full personalization (minimize variance)
    // λ=1: Population-based (minimize bias from population)
    let cost = (1.0 - problem.bias_weight) * variance + problem.bias_weight * bias;

    Ok(cost)
}
