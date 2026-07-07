//! AUC / dense-grid helpers used by the cost function.
//!
//! For the AUC targets, concentrations are simulated on a dense time grid and
//! integrated with the trapezoidal rule:
//!
//! ```text
//! AUC(t) = Σᵢ (C[i] + C[i-1]) / 2 × (t[i] - t[i-1])
//! ```

use pharmsol::prelude::*;

/// Find the time of the last dose (bolus or infusion) before a given observation
/// time. Returns `0.0` if no dose exists before `obs_time`.
pub fn find_last_dose_time_before(subject: &Subject, obs_time: f64) -> f64 {
    let mut last_dose_time = 0.0;

    for occasion in subject.occasions() {
        for event in occasion.events() {
            let event_time = match event {
                Event::Bolus(b) => Some(b.time()),
                Event::Infusion(i) => Some(i.time()),
                Event::Observation(_) => None,
            };

            if let Some(t) = event_time {
                if t < obs_time && t > last_dose_time {
                    last_dose_time = t;
                }
            }
        }
    }

    last_dose_time
}

/// Generate a dense time grid for AUC calculations.
///
/// The grid contains the observation times plus regular points spaced `idelta`
/// apart (in the model's time units), sorted and deduplicated. A non-positive
/// `idelta` disables the regular grid, leaving only the observation times.
pub fn calculate_dense_times(
    start_time: f64,
    end_time: f64,
    obs_times: &[f64],
    idelta: f64,
) -> Vec<f64> {
    let mut times = Vec::new();

    times.extend_from_slice(obs_times);

    if idelta > 0.0 {
        let mut t = start_time;
        while t <= end_time {
            times.push(t);
            t += idelta;
        }
    }

    if !times.contains(&end_time) {
        times.push(end_time);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

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

/// Calculate cumulative AUC at target times using the trapezoidal rule.
///
/// Integrates from the first dense time point and extracts the cumulative AUC at
/// each of `target_times`.
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
        let dt = dense_times[i] - dense_times[i - 1];
        let avg_conc = (dense_predictions[i] + dense_predictions[i - 1]) / 2.0;
        auc += avg_conc * dt;

        if target_idx < target_times.len()
            && (dense_times[i] - target_times[target_idx]).abs() < tolerance
        {
            target_aucs.push(auc);
            target_idx += 1;
        }
    }

    target_aucs
}

/// Calculate interval AUC for each observation independently.
///
/// For each observation at time `t`, integrates from the last dose before `t` to
/// `t` (e.g. dosing-interval AUCτ at steady state).
pub fn calculate_interval_auc_per_observation(
    subject: &Subject,
    dense_times: &[f64],
    dense_predictions: &[f64],
    obs_times: &[f64],
) -> Vec<f64> {
    assert_eq!(dense_times.len(), dense_predictions.len());

    let mut interval_aucs = Vec::with_capacity(obs_times.len());
    let tolerance = 1e-10;

    for &obs_time in obs_times {
        let last_dose_time = find_last_dose_time_before(subject, obs_time);

        let start_idx = dense_times
            .iter()
            .position(|&t| (t - last_dose_time).abs() < tolerance || t > last_dose_time)
            .unwrap_or(0);

        let end_idx = dense_times
            .iter()
            .position(|&t| (t - obs_time).abs() < tolerance || t > obs_time)
            .unwrap_or(dense_times.len() - 1);

        let mut auc = 0.0;
        for i in (start_idx + 1)..=end_idx.min(dense_times.len() - 1) {
            let dt = dense_times[i] - dense_times[i - 1];
            let avg_conc = (dense_predictions[i] + dense_predictions[i - 1]) / 2.0;
            auc += avg_conc * dt;
        }

        interval_aucs.push(auc);
    }

    interval_aucs
}
