//! Parametric algorithm predictions
//!
//! This module defines prediction structures for parametric population pharmacokinetic
//! algorithms (SAEM, FOCE, etc.). Unlike NP algorithms which use discrete support points,
//! parametric methods generate predictions from:
//!
//! - **Population predictions (PPRED)**: Using population mean parameters (μ)
//! - **Individual predictions (IPRED)**: Using individual parameter estimates (ψᵢ)
//!
//! # R saemix Correspondence
//!
//! | PMcore | R saemix | Description |
//! |--------|----------|-------------|
//! | `ppred` | `ppred` | Population prediction with population parameters |
//! | `ipred` | `ipred` | Individual prediction with MAP estimates |
//! | `icpred` | `icpred` | Individual prediction with conditional mean |
//! | `res` | `ires` | Individual residual (obs - ipred) |
//! | `wres` | `iwres` | Individual weighted residual |

use anyhow::{Context, Result};
use csv::WriterBuilder;
use pharmsol::{Censor, Data, Equation, Predictions as PredTrait};
use serde::{Deserialize, Serialize};

use crate::routines::output::OutputFile;
use crate::routines::settings::Settings;
use crate::structs::parametric::{IndividualEstimates, Population};

/// A single prediction row for parametric algorithms
///
/// Contains predictions from both population and individual parameters,
/// along with residuals and weighted residuals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricPredictionRow {
    /// Subject identifier
    id: String,
    /// Time of prediction
    time: f64,
    /// Output equation number (for multi-output models)
    outeq: usize,
    /// Occasion/block number
    block: usize,
    /// Observed value (None if prediction-only time point)
    obs: Option<f64>,
    /// Censoring flag
    cens: Censor,
    /// Population prediction using μ (population mean parameters)
    ppred: f64,
    /// Individual prediction using individual estimates (EBEs/MAP)
    ipred: f64,
    /// Individual residual: obs - ipred
    ires: Option<f64>,
    /// Individual weighted residual: ires / σ(ipred)
    iwres: Option<f64>,
}

impl ParametricPredictionRow {
    /// Get the subject ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the time
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get the output equation
    pub fn outeq(&self) -> usize {
        self.outeq
    }

    /// Get the block/occasion
    pub fn block(&self) -> usize {
        self.block
    }

    /// Get the observation
    pub fn obs(&self) -> Option<f64> {
        self.obs
    }

    /// Get the censoring flag
    pub fn censoring(&self) -> Censor {
        self.cens
    }

    /// Get population prediction
    pub fn ppred(&self) -> f64 {
        self.ppred
    }

    /// Get individual prediction
    pub fn ipred(&self) -> f64 {
        self.ipred
    }

    /// Get individual residual
    pub fn ires(&self) -> Option<f64> {
        self.ires
    }

    /// Get individual weighted residual
    pub fn iwres(&self) -> Option<f64> {
        self.iwres
    }
}

/// Container for parametric model predictions
///
/// Stores predictions for all subjects at all time points, including
/// both population and individual predictions with residuals.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParametricPredictions {
    predictions: Vec<ParametricPredictionRow>,
}

impl IntoIterator for ParametricPredictions {
    type Item = ParametricPredictionRow;
    type IntoIter = std::vec::IntoIter<ParametricPredictionRow>;

    fn into_iter(self) -> Self::IntoIter {
        self.predictions.into_iter()
    }
}

impl ParametricPredictions {
    /// Create an empty predictions container
    pub fn new() -> Self {
        Self {
            predictions: Vec::new(),
        }
    }

    /// Add a prediction row
    pub fn add(&mut self, row: ParametricPredictionRow) {
        self.predictions.push(row);
    }

    /// Get all predictions
    pub fn predictions(&self) -> &[ParametricPredictionRow] {
        &self.predictions
    }

    /// Get the number of predictions
    pub fn len(&self) -> usize {
        self.predictions.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.predictions.is_empty()
    }

    /// Calculate predictions for all subjects
    ///
    /// # Arguments
    ///
    /// * `equation` - The model equation
    /// * `data` - The input data
    /// * `population` - Population parameters (μ, Ω)
    /// * `individual_estimates` - Individual parameter estimates (EBEs)
    /// * `sigma` - Residual error standard deviation (for IWRES calculation)
    /// * `idelta` - Time increment for prediction grid expansion
    /// * `tad` - Time after dose for prediction expansion
    ///
    /// # Returns
    ///
    /// A `ParametricPredictions` containing predictions for all subjects
    pub fn calculate<E: Equation>(
        equation: &E,
        data: &Data,
        population: &Population,
        individual_estimates: &IndividualEstimates,
        sigma: Option<f64>,
        idelta: f64,
        tad: f64,
    ) -> Result<Self> {
        let mut container = Self::new();

        // Expand data with prediction grid
        let expanded_data = data.clone().expand(idelta, tad);
        let subjects = expanded_data.subjects();

        // Get population mean parameters as a vector (already in ψ space)
        let mu: Vec<f64> = (0..population.npar()).map(|i| population.mu()[i]).collect();

        // Match subjects with individual estimates
        for subject in subjects.iter() {
            // Find individual estimates for this subject
            let individual = individual_estimates
                .iter()
                .find(|ind| ind.subject_id() == subject.id());

            // Get individual parameters (fall back to population mean if no individual estimate)
            // Note: Individual::psi() returns φ (unconstrained) for SAEM, so we need to transform.
            // For LogNormal parameters (most common), ψ = exp(φ)
            // TODO: Store transforms in result and apply them properly here
            let psi: Vec<f64> = match individual {
                Some(ind) => (0..ind.npar()).map(|i| ind.psi()[i].exp()).collect(),
                None => mu.clone(),
            };

            // Simulate with population parameters (PPRED)
            let ppred_result = equation
                .simulate_subject(subject, &mu, None)
                .context(format!(
                    "Failed to simulate subject {} with population parameters",
                    subject.id()
                ))?;
            let ppred_vec = ppred_result.0.get_predictions();

            // Simulate with individual parameters (IPRED)
            let ipred_result = equation
                .simulate_subject(subject, &psi, None)
                .context(format!(
                    "Failed to simulate subject {} with individual parameters",
                    subject.id()
                ))?;
            let ipred_vec = ipred_result.0.get_predictions();

            // Create prediction rows
            for (ppred, ipred) in ppred_vec.iter().zip(ipred_vec.iter()) {
                let obs = ppred.observation();

                // Calculate residuals if observation exists
                let (ires, iwres) = if let Some(y) = obs {
                    let res = y - ipred.prediction();
                    let wres = sigma.map(|s| {
                        if s > 0.0 {
                            res / s
                        } else {
                            f64::NAN
                        }
                    });
                    (Some(res), wres)
                } else {
                    (None, None)
                };

                let row = ParametricPredictionRow {
                    id: subject.id().clone(),
                    time: ppred.time(),
                    outeq: ppred.outeq(),
                    block: ppred.occasion(),
                    obs,
                    cens: ppred.censoring(),
                    ppred: ppred.prediction(),
                    ipred: ipred.prediction(),
                    ires,
                    iwres,
                };

                container.add(row);
            }
        }

        Ok(container)
    }

    /// Write predictions to a CSV file
    pub fn write(&self, settings: &Settings) -> Result<()> {
        let outputfile = OutputFile::new(&settings.output().path, "pred.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(&outputfile.file);

        // Header
        writer.write_record([
            "id", "time", "outeq", "block", "obs", "cens", "ppred", "ipred", "ires", "iwres",
        ])?;

        // Data rows
        for row in &self.predictions {
            writer.write_record([
                row.id.clone(),
                row.time.to_string(),
                row.outeq.to_string(),
                row.block.to_string(),
                row.obs.map_or("NA".to_string(), |v| v.to_string()),
                format!("{:?}", row.cens),
                row.ppred.to_string(),
                row.ipred.to_string(),
                row.ires.map_or("NA".to_string(), |v| format!("{:.6}", v)),
                row.iwres.map_or("NA".to_string(), |v| format!("{:.6}", v)),
            ])?;
        }

        writer.flush()?;
        tracing::debug!("Predictions written to {:?}", outputfile.relative_path());

        Ok(())
    }
}

/// Summary statistics for predictions (useful for GOF plots)
#[derive(Debug, Clone, Default, Serialize)]
pub struct PredictionSummary {
    /// Number of observations
    pub n_obs: usize,
    /// Mean PPRED
    pub mean_ppred: f64,
    /// Mean IPRED
    pub mean_ipred: f64,
    /// Mean absolute residual
    pub mean_abs_ires: f64,
    /// Root mean squared error (IPRED)
    pub rmse_ipred: f64,
    /// Correlation between obs and IPRED
    pub corr_obs_ipred: f64,
}

impl ParametricPredictions {
    /// Calculate summary statistics for the predictions
    pub fn summary(&self) -> PredictionSummary {
        let obs_rows: Vec<_> = self.predictions.iter().filter(|r| r.obs.is_some()).collect();
        let n = obs_rows.len();

        if n == 0 {
            return PredictionSummary::default();
        }

        let sum_ppred: f64 = obs_rows.iter().map(|r| r.ppred).sum();
        let sum_ipred: f64 = obs_rows.iter().map(|r| r.ipred).sum();
        let sum_abs_ires: f64 = obs_rows.iter().filter_map(|r| r.ires.map(|v| v.abs())).sum();
        let sum_sq_ires: f64 = obs_rows.iter().filter_map(|r| r.ires.map(|v| v * v)).sum();

        // Correlation calculation
        let obs_vec: Vec<f64> = obs_rows.iter().filter_map(|r| r.obs).collect();
        let ipred_vec: Vec<f64> = obs_rows.iter().map(|r| r.ipred).collect();

        let mean_obs = obs_vec.iter().sum::<f64>() / n as f64;
        let mean_ipred = ipred_vec.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_obs = 0.0;
        let mut var_ipred = 0.0;

        for (o, p) in obs_vec.iter().zip(ipred_vec.iter()) {
            let do_ = o - mean_obs;
            let dp = p - mean_ipred;
            cov += do_ * dp;
            var_obs += do_ * do_;
            var_ipred += dp * dp;
        }

        let corr = if var_obs > 0.0 && var_ipred > 0.0 {
            cov / (var_obs.sqrt() * var_ipred.sqrt())
        } else {
            0.0
        };

        PredictionSummary {
            n_obs: n,
            mean_ppred: sum_ppred / n as f64,
            mean_ipred: sum_ipred / n as f64,
            mean_abs_ires: sum_abs_ires / n as f64,
            rmse_ipred: (sum_sq_ires / n as f64).sqrt(),
            corr_obs_ipred: corr,
        }
    }
}
