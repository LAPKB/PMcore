use anyhow::{bail, Result};
use pharmsol::{prelude::simulator::Prediction, Censor, Data, Predictions as PredTrait};
use serde::{Deserialize, Serialize};

use crate::{
    estimation::nonparametric::{theta::Theta, weights::Weights},
    estimation::nonparametric::{weighted_median, Posterior},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NPPredictionRow {
    id: String,
    time: f64,
    outeq: usize,
    block: usize,
    obs: Option<f64>,
    cens: Censor,
    pop_mean: f64,
    pop_median: f64,
    post_mean: f64,
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

    pub fn add(&mut self, row: NPPredictionRow) {
        self.predictions.push(row);
    }

    pub fn predictions(&self) -> &[NPPredictionRow] {
        &self.predictions
    }

    pub fn calculate(
        equation: &impl pharmsol::prelude::simulator::Equation,
        data: &Data,
        theta: &Theta,
        w: &Weights,
        posterior: &Posterior,
        idelta: f64,
        tad: f64,
    ) -> Result<Self> {
        let mut container = NPPredictions::new();

        let data = data.clone().expand(idelta, tad);
        let subjects = data.subjects();

        if subjects.len() != posterior.matrix().nrows() {
            bail!("Number of subjects and number of posterior means do not match");
        };

        for (subject_index, subject) in subjects.iter().enumerate() {
            let mut predictions: Vec<Vec<Prediction>> = Vec::new();

            for spp in theta.matrix().row_iter() {
                let spp_values = spp.iter().cloned().collect::<Vec<f64>>();
                let pred = equation
                    .simulate_subject_dense(subject, &spp_values, None)?
                    .0
                    .get_predictions();
                predictions.push(pred);
            }

            if predictions.is_empty() {
                continue;
            }

            let mut pop_mean: Vec<f64> = vec![0.0; predictions.first().unwrap().len()];
            for (i, outer_pred) in predictions.iter().enumerate() {
                for (j, pred) in outer_pred.iter().enumerate() {
                    pop_mean[j] += pred.prediction() * w[i];
                }
            }

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

            let mut posterior_mean: Vec<f64> = vec![0.0; predictions.first().unwrap().len()];
            for (i, outer_pred) in predictions.iter().enumerate() {
                for (j, pred) in outer_pred.iter().enumerate() {
                    posterior_mean[j] += pred.prediction() * posterior.matrix()[(subject_index, i)];
                }
            }

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
}
