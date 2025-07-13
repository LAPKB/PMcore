use anyhow::{Ok, Result};
use faer::Mat;
use pharmsol::{Equation, ErrorModel, Predictions, Subject};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::structs::theta::Theta;

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

/// Perform multiple-model optimization to determine optimal sample times
pub fn mmopt(
    theta: &Theta,
    subject: &Subject,
    equation: impl Equation,
    errormodel: ErrorModel,
    nsamp: usize,
    weights: Vec<f64>,
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
                .unwrap()
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

    let (best_combo, min_risk) = candidate_indices
        .par_iter()
        .map(|combo| {
            let risk =
                calculate_risk(combo, &pred_matrix, theta, &errormodel, weights.clone()).unwrap();
            (combo.clone(), risk)
        })
        .min_by(|(_, risk_a), (_, risk_b)| risk_a.partial_cmp(risk_b).unwrap())
        .unwrap();

    let optimal_times = best_combo.iter().map(|&i| times[i]).collect();
    Ok(MmoptResult {
        times: optimal_times,
        risk: min_risk,
    })
}

/// Calculate the risk for a specific combination of sample times
fn calculate_risk(
    combo: &[usize],
    pred_matrix: &Mat<f64>,
    theta: &Theta,
    errormodel: &ErrorModel,
    weights: Vec<f64>,
) -> Result<f64> {
    let nspp = theta.nspp();

    let risk = (0..nspp)
        .flat_map(|i| (0..nspp).map(move |j| (i, j)))
        .filter(|(i, j)| i != j)
        .map(|(i, j)| {
            // Extract observations for the selected time points
            let i_obs: Vec<f64> = combo.iter().map(|&k| pred_matrix[(k, i)]).collect();

            let j_obs: Vec<f64> = combo.iter().map(|&k| pred_matrix[(k, j)]).collect();

            // Calculate the sum of log-likelihood differences
            let sum_k_ijn: f64 = i_obs
                .iter()
                .zip(j_obs.iter())
                .map(|(&y_i, &y_j)| {
                    let i_var = errormodel.variance_from_value(y_i).unwrap();
                    let j_var = errormodel.variance_from_value(y_j).unwrap();
                    let denominator = i_var + j_var;

                    let term1 = (y_i - y_j).powi(2) / (4.0 * denominator);
                    let term2 = 0.5 * (denominator / 2.0).ln();
                    let term3 = -0.25 * (i_var * j_var).ln();

                    term1 + term2 + term3
                })
                .sum();

            // For now, assume unit cost matrix (cost = 1.0 for all pairs)
            // This can be parameterized later if needed
            let cost = 1.0;

            weights[i] * weights[j] * (-sum_k_ijn).exp() * cost
        })
        .sum();

    Ok(risk)
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
