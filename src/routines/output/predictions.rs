use anyhow::{bail, Result};
use pharmsol::{prelude::simulator::Prediction, Censor, Data, Predictions as PredTrait};
use serde::{Deserialize, Serialize};

use crate::{
    routines::output::{posterior::Posterior, weighted_median},
    structs::{theta::Theta, weights::Weights},
};

/// Container for the multiple model estimated predictions
///
/// Each row contains the predictions for a single time point for a single subject
/// It includes the population and posterior mean and median predictions
/// These are defined by the mean and median of the prediction for each model, weighted by the population or posterior weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NPPredictionRow {
    /// The subject ID
    id: String,
    /// The time of the prediction
    time: f64,
    /// The output equation number
    outeq: usize,
    /// The occasion of the prediction
    block: usize,
    /// The observed value, if any
    obs: Option<f64>,
    /// Censored observation flag
    cens: Censor,
    /// The population mean prediction
    pop_mean: f64,
    /// The population median prediction
    pop_median: f64,
    /// The posterior mean prediction
    post_mean: f64,
    /// The posterior median prediction
    post_median: f64,
}

impl NPPredictionRow {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn outeq(&self) -> usize {
        self.outeq
    }
    pub fn block(&self) -> usize {
        self.block
    }
    pub fn obs(&self) -> Option<f64> {
        self.obs
    }
    pub fn pop_mean(&self) -> f64 {
        self.pop_mean
    }
    pub fn pop_median(&self) -> f64 {
        self.pop_median
    }
    pub fn post_mean(&self) -> f64 {
        self.post_mean
    }
    pub fn post_median(&self) -> f64 {
        self.post_median
    }

    pub fn censoring(&self) -> Censor {
        self.cens
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NPPredictions {
    predictions: Vec<NPPredictionRow>,
}

impl IntoIterator for NPPredictions {
    type Item = NPPredictionRow;
    type IntoIter = std::vec::IntoIter<NPPredictionRow>;

    fn into_iter(self) -> Self::IntoIter {
        self.predictions.into_iter()
    }
}

impl Default for NPPredictions {
    fn default() -> Self {
        NPPredictions::new()
    }
}

impl NPPredictions {
    pub fn new() -> Self {
        NPPredictions {
            predictions: Vec::new(),
        }
    }

    /// Add a [NPPredictionRow] to the predictions
    pub fn add(&mut self, row: NPPredictionRow) {
        self.predictions.push(row);
    }

    /// Get a reference to the predictions
    pub fn predictions(&self) -> &[NPPredictionRow] {
        &self.predictions
    }

    /// Calculate the population and posterior predictions
    ///
    /// # Arguments
    /// * `equation` - The equation to use for simulation
    /// * `data` - The data to use for simulation
    /// * `theta` - The theta values for the simulation
    /// * `w` - The weights for the simulation
    /// * `posterior` - The posterior probabilities for the simulation
    /// * `idelta` - The delta for the simulation
    /// * `tad` - The time after dose for the simulation
    /// # Returns
    /// A Result containing the NPPredictions or an error
    pub fn calculate(
        equation: &impl pharmsol::prelude::simulator::Equation,
        data: &Data,
        theta: &Theta,
        w: &Weights,
        posterior: &Posterior,
        idelta: f64,
        tad: f64,
    ) -> Result<Self> {
        // Create a new NPPredictions instance
        let mut container = NPPredictions::new();

        // Expand data
        let data = data.clone().expand(idelta, tad);
        let subjects = data.subjects();

        if subjects.len() != posterior.matrix().nrows() {
            bail!("Number of subjects and number of posterior means do not match");
        };

        // Iterate over each subject and then each support point
        for subject in subjects.iter().enumerate() {
            let (subject_index, subject) = subject;

            // Container for predictions for this subject
            // This will hold predictions for each support point
            // The outer vector is for each support point
            // The inner vector is for the vector of predictions for that support point
            let mut predictions: Vec<Vec<Prediction>> = Vec::new();

            // And each support points
            for spp in theta.matrix().row_iter() {
                // Simulate the subject with the current support point
                let spp_values = spp.iter().cloned().collect::<Vec<f64>>();
                let pred = equation
                    .simulate_subject(subject, &spp_values, None)?
                    .0
                    .get_predictions();
                predictions.push(pred);
            }

            if predictions.is_empty() {
                continue; // Skip this subject if no predictions are available
            }

            // Calculate population mean using
            let mut pop_mean: Vec<f64> = vec![0.0; predictions.first().unwrap().len()];
            for outer_pred in predictions.iter().enumerate() {
                let (i, outer_pred) = outer_pred;
                for inner_pred in outer_pred.iter().enumerate() {
                    let (j, pred) = inner_pred;
                    pop_mean[j] += pred.prediction() * w[i];
                }
            }

            // Calculate population median
            let mut pop_median: Vec<f64> = Vec::new();
            for j in 0..predictions.first().unwrap().len() {
                let mut values: Vec<f64> = Vec::new();
                let mut weights: Vec<f64> = Vec::new();

                for (i, outer_pred) in predictions.iter().enumerate() {
                    values.push(outer_pred[j].prediction());
                    weights.push(w[i]);
                }

                let median_val = weighted_median(&values, &weights);
                pop_median.push(median_val);
            }

            // Calculate posterior mean
            let mut posterior_mean: Vec<f64> = vec![0.0; predictions.first().unwrap().len()];
            for outer_pred in predictions.iter().enumerate() {
                let (i, outer_pred) = outer_pred;
                for inner_pred in outer_pred.iter().enumerate() {
                    let (j, pred) = inner_pred;
                    posterior_mean[j] += pred.prediction() * posterior.matrix()[(subject_index, i)];
                }
            }

            // Calculate posterior median
            let mut posterior_median: Vec<f64> = Vec::new();
            for j in 0..predictions.first().unwrap().len() {
                let mut values: Vec<f64> = Vec::new();
                let mut weights: Vec<f64> = Vec::new();

                for (i, outer_pred) in predictions.iter().enumerate() {
                    values.push(outer_pred[j].prediction());
                    weights.push(posterior.matrix()[(subject_index, i)]);
                }

                let median_val = weighted_median(&values, &weights);
                posterior_median.push(median_val);
            }

            // Iterate over the aggregated predictions (one row per timepoint per subject)
            // Use the first support point predictions to get time, outeq, block, and obs info
            if let Some(first_spp_preds) = predictions.first() {
                for (j, p) in first_spp_preds.iter().enumerate() {
                    let row = NPPredictionRow {
                        id: subject.id().clone(),
                        time: p.time(),
                        outeq: p.outeq(),
                        block: p.occasion(),
                        obs: p.observation(),
                        cens: p.censoring(),
                        pop_mean: pop_mean[j],
                        pop_median: pop_median[j],
                        post_mean: posterior_mean[j],
                        post_median: posterior_median[j],
                    };
                    container.add(row);
                }
            }
        }

        Ok(container)
    }

    /// Compute prediction performance metrics for all prediction types
    ///
    /// Only uncensored observations (`Censor::None`) with a non-`None` observed value are included.
    /// Returns `None` if there are no valid observation-prediction pairs.
    pub fn metrics(&self) -> Option<PredictionMetrics> {
        let cap = self.predictions.len();
        let mut obs_vals = Vec::with_capacity(cap);
        let mut pop_mean_vals = Vec::with_capacity(cap);
        let mut pop_median_vals = Vec::with_capacity(cap);
        let mut post_mean_vals = Vec::with_capacity(cap);
        let mut post_median_vals = Vec::with_capacity(cap);
        let mut subject_ids = std::collections::HashSet::new();

        for row in &self.predictions {
            if row.cens != Censor::None {
                continue;
            }
            if let Some(o) = row.obs {
                obs_vals.push(o);
                pop_mean_vals.push(row.pop_mean);
                pop_median_vals.push(row.pop_median);
                post_mean_vals.push(row.post_mean);
                post_median_vals.push(row.post_median);
                subject_ids.insert(row.id.clone());
            }
        }

        if obs_vals.is_empty() {
            return None;
        }

        Some(PredictionMetrics {
            n_subjects: subject_ids.len(),
            pop_mean: ErrorMetrics::compute(&obs_vals, &pop_mean_vals),
            pop_median: ErrorMetrics::compute(&obs_vals, &pop_median_vals),
            post_mean: ErrorMetrics::compute(&obs_vals, &post_mean_vals),
            post_median: ErrorMetrics::compute(&obs_vals, &post_median_vals),
        })
    }
}

/// Metrics for a single prediction type (e.g. population mean, posterior median)
///
/// Percentage metrics (`bias_pct`, `imprecision_pct`, `rmse_pct`) are computed only
/// for observation-prediction pairs where obs > 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Number of observations used
    pub n: usize,
    /// Number of observations excluded from percentage metrics (obs <= 0)
    pub n_excluded: usize,
    /// Bias: mean(pred - obs)
    pub bias: f64,
    /// Imprecision: standard deviation of (pred - obs)
    pub imprecision: f64,
    /// Root mean squared error: sqrt(mean((pred - obs)²))
    pub rmse: f64,
    /// Coefficient of determination (R²)
    pub r_squared: f64,
    /// Relative bias (%): mean((pred - obs) / obs) * 100, for obs > 0
    pub bias_pct: f64,
    /// Relative imprecision (%): SD of ((pred - obs) / obs) * 100, for obs > 0
    pub imprecision_pct: f64,
    /// Relative RMSE (%): sqrt(mean(((pred - obs) / obs)²)) * 100, for obs > 0
    pub rmse_pct: f64,
}

impl ErrorMetrics {
    /// Compute error metrics from paired observations and predictions.
    ///
    /// Percentage metrics only include pairs where obs > 0.
    fn compute(obs: &[f64], pred: &[f64]) -> Self {
        let n = obs.len();
        assert_eq!(n, pred.len());

        if n == 0 {
            return ErrorMetrics {
                n: 0,
                n_excluded: 0,
                bias: f64::NAN,
                imprecision: f64::NAN,
                rmse: f64::NAN,
                r_squared: f64::NAN,
                bias_pct: f64::NAN,
                imprecision_pct: f64::NAN,
                rmse_pct: f64::NAN,
            };
        }

        let nf = n as f64;
        let mut sum_err = 0.0;
        let mut sum_sq_err = 0.0;
        let mut rel_errors: Vec<f64> = Vec::new();

        for (&o, &p) in obs.iter().zip(pred.iter()) {
            let err = p - o;
            sum_err += err;
            sum_sq_err += err * err;
            if o > 0.0 {
                rel_errors.push(err / o);
            }
        }

        let bias = sum_err / nf;
        let imprecision = (sum_sq_err / nf - bias * bias).max(0.0).sqrt();
        let rmse = (sum_sq_err / nf).sqrt();

        // R²: 1 - SS_res / SS_tot
        let obs_mean = obs.iter().sum::<f64>() / nf;
        let ss_tot: f64 = obs.iter().map(|&o| (o - obs_mean).powi(2)).sum();
        let r_squared = if ss_tot > 0.0 {
            1.0 - sum_sq_err / ss_tot
        } else {
            f64::NAN
        };

        let (bias_pct, imprecision_pct, rmse_pct) = if !rel_errors.is_empty() {
            let n_rel = rel_errors.len() as f64;
            let mean_rel: f64 = rel_errors.iter().sum::<f64>() / n_rel;
            let mean_sq_rel: f64 = rel_errors.iter().map(|e| e * e).sum::<f64>() / n_rel;
            let var_rel = (mean_sq_rel - mean_rel * mean_rel).max(0.0);
            (
                mean_rel * 100.0,
                var_rel.sqrt() * 100.0,
                mean_sq_rel.sqrt() * 100.0,
            )
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };

        let n_excluded = n - rel_errors.len();

        ErrorMetrics {
            n,
            n_excluded,
            bias,
            imprecision,
            rmse,
            r_squared,
            bias_pct,
            imprecision_pct,
            rmse_pct,
        }
    }
}

/// Prediction performance metrics for all prediction types
///
/// Contains [ErrorMetrics] for each of the four prediction types computed by `NPPredictions`:
/// population mean, population median, posterior mean, and posterior median.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetrics {
    /// Number of unique subjects included
    pub n_subjects: usize,
    /// Metrics for population mean predictions
    pub pop_mean: ErrorMetrics,
    /// Metrics for population median predictions
    pub pop_median: ErrorMetrics,
    /// Metrics for posterior mean predictions
    pub post_mean: ErrorMetrics,
    /// Metrics for posterior median predictions
    pub post_median: ErrorMetrics,
}

impl std::fmt::Display for PredictionMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = [
            &self.pop_mean,
            &self.pop_median,
            &self.post_mean,
            &self.post_median,
        ];
        let w = 14; // column width

        write!(
            f,
            "Prediction Metrics ({} subjects, {} observations",
            self.n_subjects, m[0].n
        )?;
        if m[0].n_excluded > 0 {
            write!(
                f,
                ", {} with obs <= 0 excluded from relative metrics",
                m[0].n_excluded
            )?;
        }
        writeln!(f, ")")?;
        writeln!(
            f,
            "{:<16}{:>w$}{:>w$}{:>w$}{:>w$}",
            "", "Pop. Mean", "Pop. Median", "Post. Mean", "Post. Median",
        )?;

        writeln!(
            f,
            "{:<16}{:>w$.4}{:>w$.4}{:>w$.4}{:>w$.4}",
            "Bias", m[0].bias, m[1].bias, m[2].bias, m[3].bias,
        )?;
        writeln!(
            f,
            "{:<16}{:>w$.4}{:>w$.4}{:>w$.4}{:>w$.4}",
            "Imprecision", m[0].imprecision, m[1].imprecision, m[2].imprecision, m[3].imprecision,
        )?;
        writeln!(
            f,
            "{:<16}{:>w$.4}{:>w$.4}{:>w$.4}{:>w$.4}",
            "RMSE", m[0].rmse, m[1].rmse, m[2].rmse, m[3].rmse,
        )?;
        writeln!(
            f,
            "{:<16}{:>w$.4}{:>w$.4}{:>w$.4}{:>w$.4}",
            "R²", m[0].r_squared, m[1].r_squared, m[2].r_squared, m[3].r_squared,
        )?;

        // Percentage metrics
        writeln!(
            f,
            "{:<16}{:>w$.2}{:>w$.2}{:>w$.2}{:>w$.2}",
            "Bias%", m[0].bias_pct, m[1].bias_pct, m[2].bias_pct, m[3].bias_pct,
        )?;
        writeln!(
            f,
            "{:<16}{:>w$.2}{:>w$.2}{:>w$.2}{:>w$.2}",
            "Imprecision%",
            m[0].imprecision_pct,
            m[1].imprecision_pct,
            m[2].imprecision_pct,
            m[3].imprecision_pct,
        )?;
        write!(
            f,
            "{:<16}{:>w$.2}{:>w$.2}{:>w$.2}{:>w$.2}",
            "RMSE%", m[0].rmse_pct, m[1].rmse_pct, m[2].rmse_pct, m[3].rmse_pct,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build an NPPredictions from raw tuples.
    /// Each tuple is (id, obs, cens, pop_mean, pop_median, post_mean, post_median).
    fn make_predictions(rows: &[(&str, Option<f64>, Censor, f64, f64, f64, f64)]) -> NPPredictions {
        let mut preds = NPPredictions::new();
        for (id, obs, cens, pm, pmed, ptm, ptmed) in rows {
            preds.add(NPPredictionRow {
                id: id.to_string(),
                time: 0.0,
                outeq: 1,
                block: 1,
                obs: *obs,
                cens: *cens,
                pop_mean: *pm,
                pop_median: *pmed,
                post_mean: *ptm,
                post_median: *ptmed,
            });
        }
        preds
    }

    // ── ErrorMetrics::compute ───────────────────────────────────────────

    #[test]
    fn error_metrics_empty_input() {
        let m = ErrorMetrics::compute(&[], &[]);
        assert_eq!(m.n, 0);
        assert_eq!(m.n_excluded, 0);
        assert!(m.bias.is_nan());
        assert!(m.rmse.is_nan());
    }

    #[test]
    fn error_metrics_perfect_predictions() {
        let obs = vec![1.0, 2.0, 3.0];
        let pred = vec![1.0, 2.0, 3.0];
        let m = ErrorMetrics::compute(&obs, &pred);

        assert_eq!(m.n, 3);
        assert_eq!(m.n_excluded, 0);
        assert!((m.bias).abs() < 1e-12);
        assert!((m.imprecision).abs() < 1e-12);
        assert!((m.rmse).abs() < 1e-12);
        assert!((m.r_squared - 1.0).abs() < 1e-12);
        assert!((m.bias_pct).abs() < 1e-12);
        assert!((m.imprecision_pct).abs() < 1e-12);
        assert!((m.rmse_pct).abs() < 1e-12);
    }

    #[test]
    fn error_metrics_constant_offset() {
        // pred = obs + 1 for all points
        let obs = vec![2.0, 4.0, 6.0];
        let pred = vec![3.0, 5.0, 7.0];
        let m = ErrorMetrics::compute(&obs, &pred);

        assert_eq!(m.n, 3);
        assert_eq!(m.n_excluded, 0);
        assert!((m.bias - 1.0).abs() < 1e-12);
        assert!((m.rmse - 1.0).abs() < 1e-12);
        // Imprecision (SD of errors) should be 0 since all errors are identical
        assert!(m.imprecision.abs() < 1e-12);
    }

    #[test]
    fn error_metrics_excludes_non_positive_obs_from_pct() {
        let obs = vec![0.0, -1.0, 2.0, 4.0];
        let pred = vec![0.5, -0.5, 2.5, 4.5];
        let m = ErrorMetrics::compute(&obs, &pred);

        assert_eq!(m.n, 4);
        assert_eq!(m.n_excluded, 2); // obs=0.0 and obs=-1.0

        // Absolute metrics use all 4 pairs
        assert!((m.bias - 0.5).abs() < 1e-12);

        // Percentage metrics use only obs=2.0 and obs=4.0
        // rel errors: 0.5/2.0 = 0.25, 0.5/4.0 = 0.125
        let expected_bias_pct = (0.25 + 0.125) / 2.0 * 100.0;
        assert!((m.bias_pct - expected_bias_pct).abs() < 1e-10);
    }

    #[test]
    fn error_metrics_all_non_positive_obs() {
        let obs = vec![0.0, -1.0];
        let pred = vec![0.5, -0.5];
        let m = ErrorMetrics::compute(&obs, &pred);

        assert_eq!(m.n, 2);
        assert_eq!(m.n_excluded, 2);
        assert!(m.bias_pct.is_nan());
        assert!(m.imprecision_pct.is_nan());
        assert!(m.rmse_pct.is_nan());
    }

    #[test]
    fn error_metrics_r_squared_constant_obs() {
        // All obs the same → SS_tot = 0 → R² = NaN
        let obs = vec![5.0, 5.0, 5.0];
        let pred = vec![5.0, 6.0, 4.0];
        let m = ErrorMetrics::compute(&obs, &pred);
        assert!(m.r_squared.is_nan());
    }

    #[test]
    fn metrics_returns_none_when_no_observations() {
        let preds = make_predictions(&[("S1", None, Censor::None, 1.0, 1.0, 1.0, 1.0)]);
        assert!(preds.metrics().is_none());
    }

    #[test]
    fn metrics_skips_censored_rows() {
        let preds = make_predictions(&[("S1", Some(10.0), Censor::BLOQ, 10.0, 10.0, 10.0, 10.0)]);
        assert!(preds.metrics().is_none());
    }

    #[test]
    fn metrics_counts_subjects() {
        let preds = make_predictions(&[
            ("S1", Some(1.0), Censor::None, 1.0, 1.0, 1.0, 1.0),
            ("S1", Some(2.0), Censor::None, 2.0, 2.0, 2.0, 2.0),
            ("S2", Some(3.0), Censor::None, 3.0, 3.0, 3.0, 3.0),
        ]);
        let m = preds.metrics().unwrap();
        assert_eq!(m.n_subjects, 2);
        assert_eq!(m.pop_mean.n, 3);
    }

    #[test]
    fn metrics_routes_predictions_correctly() {
        // Use distinct prediction values per type so we can verify routing
        let preds = make_predictions(&[("S1", Some(10.0), Censor::None, 11.0, 12.0, 13.0, 14.0)]);
        let m = preds.metrics().unwrap();

        assert!((m.pop_mean.bias - 1.0).abs() < 1e-12);
        assert!((m.pop_median.bias - 2.0).abs() < 1e-12);
        assert!((m.post_mean.bias - 3.0).abs() < 1e-12);
        assert!((m.post_median.bias - 4.0).abs() < 1e-12);
    }

    #[test]
    fn display_header_without_exclusions() {
        let preds = make_predictions(&[("S1", Some(1.0), Censor::None, 1.0, 1.0, 1.0, 1.0)]);
        let m = preds.metrics().unwrap();
        let output = format!("{}", m);
        assert!(output.contains("1 subjects"));
        assert!(output.contains("1 observations"));
        assert!(!output.contains("excluded"));
    }

    #[test]
    fn display_header_with_exclusions() {
        let preds = make_predictions(&[
            ("S1", Some(0.0), Censor::None, 0.5, 0.5, 0.5, 0.5),
            ("S1", Some(2.0), Censor::None, 2.5, 2.5, 2.5, 2.5),
        ]);
        let m = preds.metrics().unwrap();
        let output = format!("{}", m);
        assert!(output.contains("2 observations"));
        assert!(output.contains("1 with obs <= 0 excluded from relative metrics"));
    }
}
