//! Cost function calculation for BestDose optimization
//!
//! Implements the hybrid cost function that trades off hitting the target on
//! average against the spread of outcomes across the parameter distribution.
//! Also enforces dose-range constraints through penalty-based bounds checking.
//!
//! # Cost Function
//!
//! Everything is computed from a single distribution over parameters — the
//! support points and their probability weights `w` (see `BestDoseObjective`).
//! Let `p[i,j]` be the prediction for support point `i` at observation `j`, and
//! `t[j]` the target.
//!
//! ```text
//! Cost = {
//!   (1-λ) × Variance + λ × Bias²,  if doses within bounds
//!   1e12 + violation² × 1e6,        if any dose violates bounds
//! }
//! ```
//!
//! ## Variance term — expected squared error
//!
//! ```text
//! Variance = Σᵢ wᵢ Σⱼ (t[j] - p[i,j])²   =   E_w[(t - p)²]
//! ```
//!
//! ## Bias term — squared error of the mean prediction
//!
//! ```text
//! Bias² = Σⱼ (t[j] - ȳ[j])²,   where ȳ[j] = Σᵢ wᵢ p[i,j]   (the weighted mean)
//! ```
//!
//! ## Bias weight parameter (λ)
//!
//! Using the decomposition `E_w[(t-p)²] = (t - E_w[p])² + Var_w(p)`, the cost
//! simplifies to:
//!
//! ```text
//! Cost = (t - E_w[p])² + (1-λ) · Var_w(p)
//! ```
//!
//! So λ controls how strongly the *spread* of predicted outcomes across the
//! distribution is penalized:
//!
//! - `λ = 0.0`: minimize the full expected squared error — hit the target on
//!   average **and** keep the prediction spread small (robust across all
//!   plausible parameter values).
//! - `λ = 1.0`: only the weighted-mean prediction has to hit the target; the
//!   spread is ignored.
//! - `0 < λ < 1`: interpolates the variance penalty.
//!
//! Note: λ is independent of whether `w` is a population distribution or a
//! patient-specific posterior — that choice is made upstream by the caller.
//!
//! # Implementation Notes
//!
//! The cost function handles both concentration and AUC targets:
//! - **Concentration**: Simulates model at observation times directly
//! - **AUC**: Generates dense time grid and calculates AUC via trapezoidal rule
//!
//! See `evaluate` for the main implementation.

use anyhow::Result;

use crate::bestdose::predictions::{
    calculate_auc_at_times, calculate_dense_times, calculate_interval_auc_per_observation,
};
use crate::bestdose::types::{Achievement, BestDoseObjective, Target};
use pharmsol::prelude::*;
use pharmsol::Equation;
use pharmsol::Predictions;

/// Cost together with the per-observation target achievements at a candidate
/// dose regimen.
pub(crate) struct Evaluation {
    pub cost: f64,
    pub achievements: Vec<Achievement>,
}

/// Calculate cost function for a candidate dose regimen
///
/// This is the core objective function minimized by the Nelder-Mead optimizer.
///
/// # Arguments
///
/// * `problem` - The [`BestDoseObjective`] containing the model, distribution,
///   target, and optimization settings
/// * `candidate_doses` - Dose amounts to evaluate (only for optimizable doses)
///
/// # Returns
///
/// The cost value `(1-λ) × Variance + λ × Bias²` for the candidate doses.
/// Lower cost indicates a better match to targets.
///
/// # Dose Masking
///
/// Only doses with `amount == 0.0` in the target subject are considered optimizable.
/// Doses with non-zero amounts remain fixed at their specified values.
///
/// The `candidate_doses` parameter contains only the optimizable doses, which are
/// substituted into the target subject before simulation.
///
/// # Cost Function Details
///
/// Both terms are computed from the single distribution `w` (support points and
/// their weights). For each support point the model is simulated with the
/// candidate doses to obtain `p[i,j]`.
///
/// - **Variance** (expected squared error): `Σᵢ wᵢ Σⱼ (t[j] - p[i,j])²`
/// - **Bias²** (error of the weighted mean): `Σⱼ (t[j] - Σᵢ wᵢ p[i,j])²`
///
/// `λ` (`problem.bias_weight`) trades between them; equivalently
/// `Cost = (t - E_w[p])² + (1-λ)·Var_w(p)`, so `λ` sets how much predictive
/// spread across the distribution is penalized.
///
/// ## Target Types
///
/// - **Concentration** ([`Target::Concentration`]): predictions are
///   concentrations at observation times.
/// - **AUC** ([`Target::AUCFromZero`] / [`Target::AUCFromLastDose`]):
///   predictions are AUC values computed via the trapezoidal rule on a dense
///   time grid (controlled by the prediction interval).
///
/// # Errors
///
/// Returns an error if:
/// - Model simulation fails
/// - Prediction length doesn't match observation count
/// - AUC calculation fails (for AUC targets)
pub(crate) fn calculate_cost<E: Equation>(
    problem: &BestDoseObjective<E>,
    candidate_doses: &[f64],
) -> Result<f64> {
    Ok(evaluate(problem, candidate_doses)?.cost)
}

/// Evaluate a candidate dose regimen, returning both the cost and the expected
/// achieved value at each target observation.
pub(crate) fn evaluate<E: Equation>(
    problem: &BestDoseObjective<E>,
    candidate_doses: &[f64],
) -> Result<Evaluation> {
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

    // Check bounds and return penalty if violated
    // This constrains the Nelder-Mead optimizer to search within the specified DoseRange
    let min_dose = problem.doserange.min;
    let max_dose = problem.doserange.max;

    for &dose in candidate_doses {
        if dose < min_dose || dose > max_dose {
            // Return a large penalty cost to push the optimizer back into bounds
            // The penalty grows quadratically with distance from the nearest bound
            let violation = if dose < min_dose {
                min_dose - dose
            } else {
                dose - max_dose
            };
            return Ok(Evaluation {
                cost: 1e12 + violation.powi(2) * 1e6,
                achievements: Vec::new(),
            });
        }
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

    let obs_outeqs: Vec<usize> = target_subject
        .occasions()
        .iter()
        .flat_map(|occ| occ.events())
        .filter_map(|event| match event {
            Event::Observation(obs) => Some(obs.outeq_index().unwrap_or(0)),
            _ => None,
        })
        .collect();

    let n_obs = obs_vec.len();

    // Accumulators
    let mut variance = 0.0_f64; // Expected squared error E[(target - pred)²]
    let mut y_bar = vec![0.0_f64; n_obs]; // Weighted-mean predictions

    // Both cost terms are computed from the single distribution weights.
    for (row, prob) in problem
        .theta
        .matrix()
        .row_iter()
        .zip(problem.weights.iter())
    {
        let spp = row.iter().copied().collect::<Vec<f64>>();

        // Get predictions based on target type
        let preds_i: Vec<f64> = match problem.target_type {
            Target::Concentration => {
                // Simulate at observation times only
                let pred = problem
                    .eq
                    .simulate_subject_dense(&target_subject, &spp, None)?;
                pred.0
                    .get_predictions()
                    .iter()
                    .map(|p| p.prediction())
                    .collect()
            }
            Target::AUCFromZero => {
                // For AUC: simulate at dense time grid and calculate cumulative AUC
                let idelta = problem.prediction_interval;
                let start_time = 0.0; // Future starts at 0
                let end_time = obs_times.last().copied().unwrap_or(0.0);

                // Generate dense time grid
                let dense_times = calculate_dense_times(start_time, end_time, &obs_times, idelta);

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
                        Event::Observation(obs) => Some(
                            obs.outeq_index()
                                .map(|outeq| (obs.time(), outeq))
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "BestDose AUC calculations require numeric observation output labels; got `{}`",
                                        obs.outeq()
                                    )
                                }),
                        ),
                        _ => None,
                    })
                    .collect::<Result<Vec<_>>>()?;

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
                let pred = problem
                    .eq
                    .simulate_subject_dense(&dense_subject, &spp, None)?;
                let dense_predictions_with_outeq = pred.0.get_predictions();

                // Group predictions by outeq using the Prediction struct
                let mut outeq_predictions: std::collections::HashMap<usize, Vec<f64>> =
                    std::collections::HashMap::new();

                for prediction in dense_predictions_with_outeq {
                    outeq_predictions
                        .entry(prediction.outeq())
                        .or_default()
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
            Target::AUCFromLastDose => {
                // For interval AUC: simulate at dense time grid and calculate AUC from last dose
                let idelta = problem.prediction_interval;
                let end_time = obs_times.last().copied().unwrap_or(0.0);

                // Generate dense time grid from 0 to end_time (need full grid for intervals)
                let dense_times = calculate_dense_times(0.0, end_time, &obs_times, idelta);

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
                        Event::Observation(obs) => Some(
                            obs.outeq_index()
                                .map(|outeq| (obs.time(), outeq))
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "BestDose AUC calculations require numeric observation output labels; got `{}`",
                                        obs.outeq()
                                    )
                                }),
                        ),
                        _ => None,
                    })
                    .collect::<Result<Vec<_>>>()?;

                let mut unique_outeqs: Vec<usize> =
                    obs_time_outeq.iter().map(|(_, outeq)| *outeq).collect();
                unique_outeqs.sort();
                unique_outeqs.dedup();

                // Add observations at dense times
                for outeq in unique_outeqs.iter() {
                    for &t in &dense_times {
                        builder = builder.missing_observation(t, *outeq);
                    }
                }

                let dense_subject = builder.build();

                // Simulate at dense times
                let pred = problem
                    .eq
                    .simulate_subject_dense(&dense_subject, &spp, None)?;
                let dense_predictions_with_outeq = pred.0.get_predictions();

                // Group predictions by outeq
                let mut outeq_predictions: std::collections::HashMap<usize, Vec<f64>> =
                    std::collections::HashMap::new();

                for prediction in dense_predictions_with_outeq {
                    outeq_predictions
                        .entry(prediction.outeq())
                        .or_default()
                        .push(prediction.prediction());
                }

                // Calculate interval AUC for each outeq separately
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

                    // Calculate interval AUC at observation times for this outeq
                    let aucs = calculate_interval_auc_per_observation(
                        &target_subject,
                        &dense_times,
                        outeq_preds,
                        &outeq_obs_times,
                    );
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

        // Calculate variance term: weighted by the distribution probability
        let mut sumsq_i = 0.0_f64;
        for (j, &obs_val) in obs_vec.iter().enumerate() {
            let pj = preds_i[j];
            let se = (obs_val - pj).powi(2);
            sumsq_i += se;
            // Weighted-mean prediction
            y_bar[j] += prob * pj;
        }

        variance += prob * sumsq_i; // Weighted by the distribution
    }

    // Bias term: squared error of the weighted-mean prediction.
    let mut bias = 0.0_f64;
    for (j, &obs_val) in obs_vec.iter().enumerate() {
        bias += (obs_val - y_bar[j]).powi(2);
    }

    // Cost = (1-λ)×Variance + λ×Bias²  ≡  Bias² + (1-λ)·Var_w(pred).
    // λ sets how much predictive spread across the distribution is penalized:
    // λ=0 minimizes the full expected squared error; λ=1 only aims the mean.
    let cost = (1.0 - problem.bias_weight) * variance + problem.bias_weight * bias;

    // Expected achieved value at each observation is the weighted-mean prediction.
    let achievements = obs_times
        .iter()
        .zip(obs_outeqs.iter())
        .zip(obs_vec.iter())
        .zip(y_bar.iter())
        .map(|(((&time, &outeq), &target), &achieved)| Achievement {
            time,
            outeq,
            target,
            achieved,
        })
        .collect();

    Ok(Evaluation { cost, achievements })
}

#[cfg(test)]
mod tests {
    use super::calculate_cost;
    use crate::bestdose::types::{BestDoseObjective, DoseRange, Target};
    use crate::estimation::nonparametric::{Theta, Weights};
    use crate::model::{BoundedParameter, ParameterSpace};
    use pharmsol::prelude::*;

    fn one_compartment() -> pharmsol::ODE {
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke, _v);
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _, _| lag! {},
            |_p, _, _| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, v);
                y[0] = x[0] / v;
            },
        )
    }

    fn single_point_theta() -> Theta {
        let params = ParameterSpace::<BoundedParameter>::new()
            .add("ke", 0.1, 0.5)
            .add("v", 40.0, 60.0);
        let mat = faer::Mat::from_fn(1, 2, |_r, c| if c == 0 { 0.3 } else { 50.0 });
        Theta::from_parts(mat, params).unwrap()
    }

    fn problem_with(target: Subject) -> BestDoseObjective<pharmsol::ODE> {
        BestDoseObjective {
            target,
            target_type: Target::Concentration,
            theta: single_point_theta(),
            weights: Weights::uniform(1),
            eq: one_compartment(),
            doserange: DoseRange::new(10.0, 300.0),
            bias_weight: 0.5,
            prediction_interval: 0.12,
        }
    }

    #[test]
    fn dose_count_mismatch_is_rejected() {
        // Two optimizable doses, so a single candidate dose must be rejected.
        let target = Subject::builder("test_patient")
            .bolus(0.0, 0.0, 0)
            .bolus(6.0, 0.0, 0)
            .observation(2.0, 5.0, 0)
            .observation(8.0, 3.0, 0)
            .build();
        let problem = problem_with(target);

        let wrong = calculate_cost(&problem, &[100.0]);
        assert!(wrong.is_err(), "wrong dose count should fail");
        assert!(wrong.unwrap_err().to_string().contains("mismatch"));

        let correct = calculate_cost(&problem, &[100.0, 150.0]);
        assert!(correct.is_ok(), "correct dose count should succeed");
    }

    #[test]
    fn empty_observations_are_rejected() {
        let target = Subject::builder("test_patient").bolus(0.0, 0.0, 0).build();
        let problem = problem_with(target);

        let result = calculate_cost(&problem, &[100.0]);
        assert!(result.is_err(), "no observations should fail");
        assert!(result.unwrap_err().to_string().contains("no observations"));
    }
}
