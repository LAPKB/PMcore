use anyhow::Result;
use faer::Mat;
use pharmsol::{
    prelude::simulator::SubjectPredictions, Data, Equation, ErrorModel, Predictions, Subject,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde_json::error;
use std::fmt::Error;

use crate::structs::theta::Theta;

pub struct PredictionsContainer {
    pub matrix: Mat<f64>,
    pub times: Vec<f64>,
    pub probs: Vec<f64>,
}

impl PredictionsContainer {
    fn matrix(&self) -> &Mat<f64> {
        &self.matrix
    }

    fn nsub(&self) -> usize {
        self.matrix.ncols()
    }
    fn nout(&self) -> usize {
        self.matrix.nrows()
    }
}

struct CostMatrix {
    matrix: Option<Mat<f64>>,
    auc: f64,
    cmax: f64,
    cmin: f64,
}

impl CostMatrix {
    pub fn new(auc: f64, cmax: f64, cmin: f64) -> Self {
        !unimplemented!()
    }
}

/// The results of a multiple-model optimization
///
///
#[derive(Debug)]
pub struct MmoptResult {
    // Optimal sample times
    pub times: Vec<f64>,
    // Bayes risk
    pub risk: f64,
}

pub fn mmopt(
    theta: &Theta,
    subject: &Subject,
    equation: impl Equation,
    errormodel: ErrorModel,
    nsamp: usize,
) -> Result<MmoptResult> {
    // Check that subject contains only one Occasion
    if subject.occasions().len() != 1 {
        return Err(anyhow::anyhow!("Subject must contain only one Occasion"));
    }

    // Generate predictions
    let predictions = theta
        .matrix()
        .row_iter()
        .map(|theta_row| {
            let support_point: Vec<f64> = theta_row.iter().cloned().collect();
            let predictions = equation
                .estimate_predictions(&subject, &support_point)
                .get_predictions();
            predictions
        })
        .collect::<Vec<_>>();

    // Times vector
    let times = predictions[0].iter().map(|p| p.time()).collect::<Vec<_>>();

    // Generate prediction matrix
    let pred_matrix = Mat::from_fn(predictions[0].len(), theta.nspp(), |i, j| {
        predictions[j][i].prediction().to_owned()
    });

    // Generate sample candidate indices
    let candidate_indices = generate_combinations(times.len(), nsamp);

    // em
    let e = errormodel.;

    let (best_combo, min_risk) = candidate_indices
        .par_iter()
        .map(|combo| {
            let mut risk = 0.0;
            // Compare the i-th and the j-th subject predictions
            for i in 0..theta.nspp() {
                for j in 0..theta.nspp() {
                    if i != j {
                        let i_obs: Vec<f64> = pred_matrix
                            .col(i)
                            .iter()
                            .enumerate()
                            .filter_map(|(k, &x)| if combo.contains(&k) { Some(x) } else { None })
                            .collect();

                        let j_obs: Vec<f64> = pred_matrix
                            .col(j)
                            .iter()
                            .enumerate()
                            .filter_map(|(k, &x)| if combo.contains(&k) { Some(x) } else { None })
                            .collect();

                        let i_var: Vec<f64> =
                            i_obs.iter().map(|&x| errormodel.(x)).collect();
                        let j_var: Vec<f64> =
                            j_obs.iter().map(|&x| errorpoly.variance(x)).collect();

                        let sum_k_ijn: f64 = i_obs
                            .iter()
                            .zip(j_obs.iter())
                            .zip(i_var.iter())
                            .zip(j_var.iter())
                            .map(|(((y_i, y_j), i_var), j_var)| {
                                let denominator = i_var + j_var;
                                let term1 = (y_i - y_j).powi(2) / (4.0 * denominator);
                                let term2 = 0.5 * ((i_var + j_var) / 2.0).ln();
                                let term3 = -0.25 * (i_var * j_var).ln();
                                term1 + term2 + term3
                            })
                            .collect::<Vec<f64>>()
                            .iter()
                            .sum::<f64>();

                        let prob_i = predictions.probs[i];
                        let prob_j = predictions.probs[j];
                        let cost = cost_matrix.matrix[(i, j)];
                        let risk_component = prob_i * prob_j * (-sum_k_ijn).exp() * cost;
                        risk += risk_component;
                    }
                }
            }

            (combo.clone(), risk)
        })
        .min_by(|(_, risk_a), (_, risk_b)| risk_a.partial_cmp(risk_b).unwrap())
        .unwrap();

    let res = MmoptResult {
        best_combo_indices: best_combo.clone(),
        best_combo_times: best_combo
            .iter()
            .map(|&index| predictions.times[index])
            .collect(),
        min_risk,
    };

    Ok(res)
}

fn generate_combinations(m: usize, n: usize) -> Vec<Vec<usize>> {
    fn backtrack(
        m: usize,
        n: usize,
        start: usize,
        current: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == n {
            results.push(current.clone());
            return;
        }

        for i in start..m {
            current.push(i);
            backtrack(m, n, i + 1, current, results);
            current.pop();
        }
    }

    let mut results = Vec::new();
    let mut current = Vec::new();
    backtrack(m, n, 0, &mut current, &mut results);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combinations() {
        let m = 5;
        let n = 3;
        let combinations = generate_combinations(m, n);
        assert_eq!(combinations.len(), 10);
        assert_eq!(combinations[0], vec![0, 1, 2]);
        assert_eq!(combinations[1], vec![0, 1, 3]);
        assert_eq!(combinations[2], vec![0, 1, 4]);
        assert_eq!(combinations[3], vec![0, 2, 3]);
        assert_eq!(combinations[4], vec![0, 2, 4]);
        assert_eq!(combinations[5], vec![0, 3, 4]);
        assert_eq!(combinations[6], vec![1, 2, 3]);
        assert_eq!(combinations[7], vec![1, 2, 4]);
        assert_eq!(combinations[8], vec![1, 3, 4]);
        assert_eq!(combinations[9], vec![2, 3, 4]);
    }
}
