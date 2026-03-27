//! Parametric algorithm predictions.

use anyhow::{Context, Result};
use pharmsol::{Censor, Data, Equation, Predictions as PredTrait};
use serde::{Deserialize, Serialize};

use crate::estimation::parametric::{IndividualEstimates, Population};

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
                Some(ind) => (0..ind.npar()).map(|i| ind.psi()[i]).collect(),
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
        let obs_rows: Vec<_> = self
            .predictions
            .iter()
            .filter(|r| r.obs.is_some())
            .collect();
        let n = obs_rows.len();

        if n == 0 {
            return PredictionSummary::default();
        }

        let sum_ppred: f64 = obs_rows.iter().map(|r| r.ppred).sum();
        let sum_ipred: f64 = obs_rows.iter().map(|r| r.ipred).sum();
        let sum_abs_ires: f64 = obs_rows
            .iter()
            .filter_map(|r| r.ires.map(|v| v.abs()))
            .sum();
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

#[cfg(test)]
mod tests {
    use super::ParametricPredictions;
    use crate::estimation::parametric::{Individual, IndividualEstimates, Population};
    use crate::model::{ParameterSpace, ParameterSpec};
    use crate::prelude::*;
    use anyhow::Result;
    use faer::{Col, Mat};
    use pharmsol::{Data, Subject};

    fn equation() -> equation::ODE {
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke);
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, v);
                y[0] = x[0] / v;
            },
        )
    }

    fn data() -> Data {
        let subject = Subject::builder("1")
            .bolus(0.0, 100.0, 0)
            .observation(1.0, 10.0, 0)
            .build();
        Data::new(vec![subject])
    }

    #[test]
    fn calculate_uses_canonical_psi_space_individual_parameters() -> Result<()> {
        let parameters = ParameterSpace::new()
            .add(ParameterSpec::bounded("ke", 0.1, 1.0))
            .add(ParameterSpec::bounded("v", 1.0, 20.0));
        let population = Population::new(
            Col::from_fn(2, |index| if index == 0 { 0.5 } else { 10.0 }),
            Mat::from_fn(2, 2, |row, col| if row == col { 0.1 } else { 0.0 }),
            parameters,
        )?;
        let individuals = IndividualEstimates::from_vec(vec![Individual::new(
            "1",
            Col::from_fn(2, |_| 0.0),
            Col::from_fn(2, |index| if index == 0 { 0.5 } else { 10.0 }),
        )?]);

        let predictions =
            ParametricPredictions::calculate(&equation(), &data(), &population, &individuals, None, 1.0, 0.0)?;

        let first_observation = predictions
            .predictions()
            .iter()
            .find(|row| row.obs().is_some())
            .expect("prediction row with observation");

        assert!((first_observation.ipred() - first_observation.ppred()).abs() < 1e-12);
        Ok(())
    }
}
