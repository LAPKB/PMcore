//! Parametric algorithm predictions.

use anyhow::{Context, Result};
use csv::WriterBuilder;
use pharmsol::{Censor, Data, Equation, Predictions as PredTrait};
use serde::{Deserialize, Serialize};

use crate::estimation::parametric::{IndividualEstimates, Population};
use crate::output::OutputFile;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricPredictionRow {
    id: String,
    time: f64,
    outeq: usize,
    block: usize,
    obs: Option<f64>,
    cens: Censor,
    ppred: f64,
    ipred: f64,
    ires: Option<f64>,
    iwres: Option<f64>,
}

impl ParametricPredictionRow {
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

    pub fn censoring(&self) -> Censor {
        self.cens
    }

    pub fn ppred(&self) -> f64 {
        self.ppred
    }

    pub fn ipred(&self) -> f64 {
        self.ipred
    }

    pub fn ires(&self) -> Option<f64> {
        self.ires
    }

    pub fn iwres(&self) -> Option<f64> {
        self.iwres
    }
}

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
    pub fn new() -> Self {
        Self {
            predictions: Vec::new(),
        }
    }

    pub fn add(&mut self, row: ParametricPredictionRow) {
        self.predictions.push(row);
    }

    pub fn predictions(&self) -> &[ParametricPredictionRow] {
        &self.predictions
    }

    pub fn len(&self) -> usize {
        self.predictions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.predictions.is_empty()
    }

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
        let expanded_data = data.clone().expand(idelta, tad);
        let subjects = expanded_data.subjects();

        let mu: Vec<f64> = (0..population.npar()).map(|i| population.mu()[i]).collect();

        for subject in subjects.iter() {
            let individual = individual_estimates
                .iter()
                .find(|ind| ind.subject_id() == subject.id());

            let psi: Vec<f64> = match individual {
                Some(ind) => (0..ind.npar()).map(|i| ind.psi()[i].exp()).collect(),
                None => mu.clone(),
            };

            let ppred_result = equation
                .simulate_subject(subject, &mu, None)
                .context(format!(
                    "Failed to simulate subject {} with population parameters",
                    subject.id()
                ))?;
            let ppred_vec = ppred_result.0.get_predictions();

            let ipred_result = equation
                .simulate_subject(subject, &psi, None)
                .context(format!(
                    "Failed to simulate subject {} with individual parameters",
                    subject.id()
                ))?;
            let ipred_vec = ipred_result.0.get_predictions();

            for (ppred, ipred) in ppred_vec.iter().zip(ipred_vec.iter()) {
                let obs = ppred.observation();
                let (ires, iwres) = if let Some(y) = obs {
                    let res = y - ipred.prediction();
                    let wres = sigma.map(|s| if s > 0.0 { res / s } else { f64::NAN });
                    (Some(res), wres)
                } else {
                    (None, None)
                };

                container.add(ParametricPredictionRow {
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
                });
            }
        }

        Ok(container)
    }

    pub fn write(&self, folder: &str) -> Result<()> {
        let outputfile = OutputFile::new(folder, "pred.csv")?;
        let mut writer = WriterBuilder::new()
            .has_headers(true)
            .from_writer(outputfile.file());

        writer.write_record([
            "id", "time", "outeq", "block", "obs", "cens", "ppred", "ipred", "ires", "iwres",
        ])?;

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

#[derive(Debug, Clone, Default, Serialize)]
pub struct PredictionSummary {
    pub n_obs: usize,
    pub mean_ppred: f64,
    pub mean_ipred: f64,
    pub mean_abs_ires: f64,
    pub rmse_ipred: f64,
    pub corr_obs_ipred: f64,
}

impl ParametricPredictions {
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

        let obs_vec: Vec<f64> = obs_rows.iter().filter_map(|r| r.obs).collect();
        let ipred_vec: Vec<f64> = obs_rows.iter().map(|r| r.ipred).collect();

        let mean_obs = obs_vec.iter().sum::<f64>() / n as f64;
        let mean_ipred = ipred_vec.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_obs = 0.0;
        let mut var_ipred = 0.0;

        for (obs, ipred) in obs_vec.iter().zip(ipred_vec.iter()) {
            let d_obs = obs - mean_obs;
            let d_ipred = ipred - mean_ipred;
            cov += d_obs * d_ipred;
            var_obs += d_obs * d_obs;
            var_ipred += d_ipred * d_ipred;
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