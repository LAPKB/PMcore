use anyhow::{bail, Result};
use faer::Col;
use pharmsol::{prelude::simulator::Prediction, Data, Event, Predictions as PredTrait};
use serde::{Deserialize, Serialize};

use crate::{
    routines::output::{posterior::Posterior, weighted_median},
    structs::theta::Theta,
};

// Structure for the output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NPPredictionRow {
    id: String,
    time: f64,
    outeq: usize,
    block: usize,
    obs: Option<f64>,
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
}

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

    pub fn calculate(
        equation: &impl pharmsol::prelude::simulator::Equation,
        data: &Data,
        theta: Theta,
        w: &Col<f64>,
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

            // Get a vector of occasions for this subject, for each predictions
            let occasions = subject
                .occasions()
                .iter()
                .flat_map(|o| {
                    o.events()
                        .iter()
                        .filter_map(|e| {
                            if let Event::Observation(_obs) = e {
                                Some(o.index())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<usize>>();

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

            for pred in predictions.iter().enumerate() {
                let (_, preds) = pred;
                for (j, p) in preds.iter().enumerate() {
                    let row = NPPredictionRow {
                        id: subject.id().clone(),
                        time: p.time(),
                        outeq: p.outeq(),
                        block: occasions[j],
                        obs: p.observation(),
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
