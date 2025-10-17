//! Prediction calculations for BestDose
//!
//! Handles:
//! - Dense time grid generation for AUC calculations
//! - Trapezoidal AUC integration
//! - Final predictions with optimal doses

use anyhow::Result;
use faer::Mat;

use crate::bestdose::types::{BestDoseProblem, Target};
use crate::routines::output::posterior::Posterior;
use crate::routines::output::predictions::NPPredictions;
use crate::structs::weights::Weights;
use pharmsol::prelude::*;
use pharmsol::Equation;

/// Generate dense time grid for AUC calculations
///
/// Creates a grid with:
/// - Observation times from the target
/// - Intermediate points at `idelta` intervals
/// - All times sorted and deduplicated
///
/// # Arguments
/// * `start_time` - Start of time range
/// * `end_time` - End of time range
/// * `obs_times` - Required observation times (always included)
/// * `idelta` - Time step for dense grid (minutes)
///
/// # Returns
/// Sorted, unique time vector suitable for AUC calculation
pub fn calculate_dense_times(
    start_time: f64,
    end_time: f64,
    obs_times: &[f64],
    idelta: usize,
) -> Vec<f64> {
    let idelta_hours = (idelta as f64) / 60.0;
    let mut times = Vec::new();

    // Add observation times
    times.extend_from_slice(obs_times);

    // Add regular grid points
    let mut t = start_time;
    while t <= end_time {
        times.push(t);
        t += idelta_hours;
    }

    // Ensure end time is included
    if !times.contains(&end_time) {
        times.push(end_time);
    }

    // Sort and deduplicate
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Remove duplicates with tolerance
    let tolerance = 1e-10;
    let mut unique_times = Vec::new();
    let mut last_time = f64::NEG_INFINITY;

    for &t in &times {
        if (t - last_time).abs() > tolerance {
            unique_times.push(t);
            last_time = t;
        }
    }

    unique_times
}

/// Calculate cumulative AUC at target times using trapezoidal rule
///
/// Takes dense concentration predictions and calculates cumulative AUC
/// from the first time point. AUC values at target observation times
/// are extracted and returned.
///
/// # Arguments
/// * `dense_times` - Dense time grid (must include all `target_times`)
/// * `dense_predictions` - Concentration predictions at `dense_times`
/// * `target_times` - Observation times where AUC should be extracted
///
/// # Returns
/// Vector of AUC values at `target_times`
pub fn calculate_auc_at_times(
    dense_times: &[f64],
    dense_predictions: &[f64],
    target_times: &[f64],
) -> Vec<f64> {
    assert_eq!(dense_times.len(), dense_predictions.len());

    let mut target_aucs = Vec::with_capacity(target_times.len());
    let mut auc = 0.0;
    let mut target_idx = 0;
    let tolerance = 1e-10;

    for i in 1..dense_times.len() {
        // Update cumulative AUC using trapezoidal rule
        let dt = dense_times[i] - dense_times[i - 1];
        let avg_conc = (dense_predictions[i] + dense_predictions[i - 1]) / 2.0;
        auc += avg_conc * dt;

        // Check if current time matches next target time
        if target_idx < target_times.len() {
            if (dense_times[i] - target_times[target_idx]).abs() < tolerance {
                target_aucs.push(auc);
                target_idx += 1;
            }
        }
    }

    target_aucs
}

/// Calculate predictions for optimal doses
///
/// This generates the final NPPredictions structure with the optimal doses
/// and appropriate weights (posterior or uniform depending on which optimization won).
pub fn calculate_final_predictions(
    problem: &BestDoseProblem,
    optimal_doses: &[f64],
    weights: &Weights,
) -> Result<(NPPredictions, Option<Vec<(f64, f64)>>)> {
    // Build subject with optimal doses
    let mut target_with_optimal = problem.target.clone();
    let mut dose_number = 0;

    for occasion in target_with_optimal.iter_mut() {
        for event in occasion.iter_mut() {
            match event {
                Event::Bolus(bolus) => {
                    bolus.set_amount(optimal_doses[dose_number]);
                    dose_number += 1;
                }
                Event::Infusion(infusion) => {
                    infusion.set_amount(optimal_doses[dose_number]);
                    dose_number += 1;
                }
                Event::Observation(_) => {}
            }
        }
    }

    // Create posterior matrix for predictions
    let posterior_matrix = Mat::from_fn(1, weights.weights().nrows(), |_row, col| {
        *weights.weights().get(col)
    });
    let posterior = Posterior::from(posterior_matrix);

    // Calculate concentration predictions
    let concentration_preds = NPPredictions::calculate(
        &problem.eq,
        &Data::new(vec![target_with_optimal.clone()]),
        problem.theta.clone(),
        weights,
        &posterior,
        0.0,
        0.0,
    )?;

    // Calculate AUC predictions if in AUC mode
    let auc_predictions = if matches!(problem.target_type, Target::AUC) {
        let obs_times: Vec<f64> = target_with_optimal
            .occasions()
            .iter()
            .flat_map(|occ| occ.events())
            .filter_map(|event| match event {
                Event::Observation(obs) => Some(obs.time()),
                _ => None,
            })
            .collect();

        let idelta = problem.settings.predictions().idelta;
        let start_time = 0.0;
        let end_time = obs_times.last().copied().unwrap_or(0.0);
        let dense_times = calculate_dense_times(start_time, end_time, &obs_times, idelta as usize);

        let subject_id = target_with_optimal.id().to_string();
        let mut builder = Subject::builder(&subject_id);

        let mut dose_number = 0;
        for occasion in target_with_optimal.occasions() {
            for event in occasion.events() {
                match event {
                    Event::Bolus(bolus) => {
                        builder = builder.bolus(bolus.time(), optimal_doses[dose_number], 0);
                        dose_number += 1;
                    }
                    Event::Infusion(_) => {
                        tracing::warn!("Infusions not fully supported in AUC mode");
                    }
                    Event::Observation(_) => {}
                }
            }
        }

        for &t in &dense_times {
            builder = builder.observation(t, -99.0, 0);
        }

        let dense_subject = builder.build();
        let mut mean_aucs = vec![0.0; obs_times.len()];

        for (row, weight) in problem.theta.matrix().row_iter().zip(weights.iter()) {
            let spp = row.iter().copied().collect::<Vec<f64>>();
            let pred = problem.eq.simulate_subject(&dense_subject, &spp, None)?;
            let dense_concentrations = pred.0.flat_predictions();
            let aucs = calculate_auc_at_times(&dense_times, &dense_concentrations, &obs_times);

            for (i, &auc) in aucs.iter().enumerate() {
                mean_aucs[i] += weight * auc;
            }
        }

        Some(obs_times.into_iter().zip(mean_aucs.into_iter()).collect())
    } else {
        None
    };

    Ok((concentration_preds, auc_predictions))
}
