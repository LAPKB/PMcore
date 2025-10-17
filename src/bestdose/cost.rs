//! Cost function calculation for BestDose optimization
//!
//! Implements the weighted cost function:
//! Cost = (1-λ)×Variance + λ×Bias²
//!
//! Where:
//! - Variance: Expected squared prediction error (patient-specific)
//! - Bias: Squared deviation from population mean prediction
//! - λ (bias_weight): 0=personalized, 1=population-based

use anyhow::Result;

use crate::bestdose::predictions::{calculate_auc_at_times, calculate_dense_times};
use crate::bestdose::types::{BestDoseProblem, Target};
use pharmsol::prelude::*;
use pharmsol::Equation;

/// Calculate cost function for a candidate dose regimen
///
/// This is the core objective function minimized by the Nelder-Mead optimizer.
///
/// Cost = (1-λ)×Variance + λ×Bias²
///
/// Where:
/// - Variance = Σᵢ posterior_weight[i] × (target - pred[i])²
///   (expected squared error across patient-specific posterior)
/// - Bias² = Σⱼ (target[j] - population_mean[j])²
///   (squared deviation from population mean prediction)
/// - population_mean[j] = Σᵢ prior_weight[i] × pred[i,j]
pub fn calculate_cost(problem: &BestDoseProblem, candidate_doses: &[f64]) -> Result<f64> {
    // Build target subject with candidate doses
    let mut target_subject = problem.target.clone();
    let mut dose_number = 0;

    for occasion in target_subject.iter_mut() {
        for event in occasion.iter_mut() {
            match event {
                Event::Bolus(bolus) => {
                    bolus.set_amount(candidate_doses[dose_number]);
                    dose_number += 1;
                }
                Event::Infusion(infusion) => {
                    infusion.set_amount(candidate_doses[dose_number]);
                    dose_number += 1;
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
                                builder = builder.bolus(bolus.time(), bolus.amount(), 0);
                            }
                            Event::Infusion(_infusion) => {
                                tracing::warn!("Infusions not yet supported in AUC mode");
                            }
                            Event::Observation(_) => {} // Skip original observations
                        }
                    }
                }

                // Add observations at dense times (with dummy values for timing only)
                for &t in &dense_times {
                    builder = builder.observation(t, -99.0, 0);
                }

                let dense_subject = builder.build();

                // Simulate at dense times
                let pred = problem.eq.simulate_subject(&dense_subject, &spp, None)?;
                let dense_predictions = pred.0.flat_predictions();

                // Calculate AUC at observation times
                calculate_auc_at_times(&dense_times, &dense_predictions, &obs_times)
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
