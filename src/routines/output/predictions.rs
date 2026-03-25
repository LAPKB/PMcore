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
        let mut obs_vals = Vec::new();
        let mut pop_mean_vals = Vec::new();
        let mut pop_median_vals = Vec::new();
        let mut post_mean_vals = Vec::new();
        let mut post_median_vals = Vec::new();
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
