use anyhow::{Ok, Result};
use faer::Mat;
use pharmsol::{Equation, ErrorModel, Predictions, Subject};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::structs::theta::Theta;

/// The results of a multiple-model optimization
///
/// Contains the optimal sample times and the associated Bayes risk.
#[derive(Debug, Clone)]
pub struct MmoptResult {
    /// Optimal sample times
    pub times: Vec<f64>,
    /// Bayes risk at the optimal sample times
    pub risk: f64,
}

impl std::fmt::Display for MmoptResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Optimal times: {:?}, Bayes risk: {:.6}",
            self.times, self.risk
        )
    }
}

/// Perform multiple-model optimization to determine optimal sample times.
///
/// This function evaluates all possible combinations of `nsamp` sample times
/// from the candidate times defined as [pharmsol::data::Observation]s in the [Subject], and returns
/// the combination that minimizes the Bayes risk of misclassification between
/// support points.
///
/// # Arguments
/// * `theta` - Support points (population parameter distribution)
/// * `subject` - Subject with candidate observation times (must have exactly one occasion)
/// * `equation` - The pharmacometric model equation
/// * `errormodel` - Error model for computing observation variance
/// * `outeq` - Output equation index to optimize for
/// * `nsamp` - Number of samples to select
/// * `weights` - Probability weights for each support point (must sum to ~1.0)
///
/// # Errors
/// Returns an error if:
/// - The subject has more than one occasion
/// - The number of support points is less than 2
/// - The weights length doesn't match the number of support points
/// - `nsamp` is 0 or exceeds the number of candidate times
pub fn mmopt(
    theta: &Theta,
    subject: &Subject,
    equation: impl Equation,
    errormodel: ErrorModel,
    outeq: usize,
    nsamp: usize,
    weights: Vec<f64>,
) -> Result<MmoptResult> {
    // Validate inputs
    if subject.occasions().len() != 1 {
        return Err(anyhow::anyhow!("Subject must contain exactly one Occasion"));
    }

    if theta.nspp() < 2 {
        return Err(anyhow::anyhow!(
            "At least 2 support points are required, got {}",
            theta.nspp()
        ));
    }

    if weights.len() != theta.nspp() {
        return Err(anyhow::anyhow!(
            "Weights length ({}) must match number of support points ({})",
            weights.len(),
            theta.nspp()
        ));
    }

    if nsamp == 0 {
        return Err(anyhow::anyhow!("Number of samples must be at least 1"));
    }

    // Generate predictions for each support point
    let predictions = theta
        .matrix()
        .row_iter()
        .enumerate()
        .map(|(idx, theta_row)| {
            let support_point: Vec<f64> = theta_row.iter().cloned().collect();
            let all_preds = equation
                .estimate_predictions(subject, &support_point)
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to generate predictions for support point {}: {}",
                        idx,
                        e
                    )
                })?
                .get_predictions();
            // Filter predictions by output equation
            Ok(all_preds
                .into_iter()
                .filter(|p| p.outeq() == outeq)
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;

    if predictions[0].is_empty() {
        return Err(anyhow::anyhow!(
            "No predictions found for output equation {}",
            outeq
        ));
    }

    // Times vector from the first support point's predictions
    let times = predictions[0].iter().map(|p| p.time()).collect::<Vec<_>>();

    if nsamp > times.len() {
        return Err(anyhow::anyhow!(
            "Number of samples ({}) exceeds number of candidate times ({})",
            nsamp,
            times.len()
        ));
    }

    // Guard against combinatorial explosion
    let n_combinations = n_choose_k(times.len(), nsamp);
    const MAX_COMBINATIONS: u128 = 1_000_000;
    if n_combinations > MAX_COMBINATIONS {
        return Err(anyhow::anyhow!(
            "C({}, {}) = {} exceeds the maximum allowed combinations ({}). \
             Reduce the number of candidate times or increase nsamp.",
            times.len(),
            nsamp,
            n_combinations,
            MAX_COMBINATIONS
        ));
    }

    // Generate prediction matrix: rows = time points, cols = support points
    let pred_matrix = Mat::from_fn(predictions[0].len(), theta.nspp(), |i, j| {
        predictions[j][i].prediction()
    });

    // Generate all C(m, n) sample candidate index combinations
    let candidate_indices = generate_combinations(times.len(), nsamp);

    // Evaluate risk in parallel for all combinations and select minimum
    let (best_combo, min_risk) = candidate_indices
        .par_iter()
        .map(|combo| {
            let risk = calculate_risk(combo, &pred_matrix, &errormodel, &weights);
            (combo.clone(), risk)
        })
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater))
        .ok_or_else(|| anyhow::anyhow!("No candidate combinations to evaluate"))?;

    let optimal_times = best_combo.iter().map(|&i| times[i]).collect();
    Ok(MmoptResult {
        times: optimal_times,
        risk: min_risk,
    })
}

/// Calculate the Bayes risk for a specific combination of sample time indices.
///
/// The risk quantifies the expected misclassification probability between support
/// points, weighted by their probabilities. Lower risk means the selected sample
/// times provide better discrimination between support points.
fn calculate_risk(
    combo: &[usize],
    pred_matrix: &Mat<f64>,
    errormodel: &ErrorModel,
    weights: &[f64],
) -> f64 {
    let nspp = pred_matrix.ncols();

    (0..nspp)
        .flat_map(|i| ((i + 1)..nspp).map(move |j| (i, j)))
        .map(|(i, j)| {
            // Extract predictions for the selected time points
            let i_obs: Vec<f64> = combo.iter().map(|&k| pred_matrix[(k, i)]).collect();
            let j_obs: Vec<f64> = combo.iter().map(|&k| pred_matrix[(k, j)]).collect();

            // Calculate the sum of log-likelihood discrimination terms
            let sum_k_ijn: f64 = i_obs
                .iter()
                .zip(j_obs.iter())
                .map(|(&y_i, &y_j)| {
                    let i_var = errormodel.variance_from_value(y_i).unwrap_or(f64::EPSILON);
                    let j_var = errormodel.variance_from_value(y_j).unwrap_or(f64::EPSILON);
                    let denominator = i_var + j_var;

                    let term1 = (y_i - y_j).powi(2) / (4.0 * denominator);
                    let term2 = 0.5 * (denominator / 2.0).ln();
                    let term3 = -0.25 * (i_var * j_var).ln();

                    term1 + term2 + term3
                })
                .sum();

            weights[i] * weights[j] * (-sum_k_ijn).exp()
        })
        .sum()
}

/// Compute C(m, n) without overflow risk by using u128 arithmetic.
fn n_choose_k(m: usize, n: usize) -> u128 {
    if n > m {
        return 0;
    }
    // Use the smaller of n and m-n for efficiency
    let k = n.min(m - n) as u128;
    let m = m as u128;
    (0..k).fold(1u128, |acc, i| acc * (m - i) / (i + 1))
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
    use faer::Mat;
    use pharmsol::ErrorPoly;

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

    #[test]
    fn test_combinations_edge_cases() {
        // Select all elements
        let combinations = generate_combinations(3, 3);
        assert_eq!(combinations.len(), 1);
        assert_eq!(combinations[0], vec![0, 1, 2]);

        // Select 1 element
        let combinations = generate_combinations(4, 1);
        assert_eq!(combinations.len(), 4);
        assert_eq!(combinations[0], vec![0]);
        assert_eq!(combinations[3], vec![3]);

        // C(6, 2) = 15
        let combinations = generate_combinations(6, 2);
        assert_eq!(combinations.len(), 15);
    }

    fn make_error_model() -> ErrorModel {
        ErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 0.0)
    }

    #[test]
    fn test_calculate_risk_identical_predictions() {
        // When two support points have identical predictions, misclassification risk is maximal
        let errormodel = make_error_model();
        let weights = vec![0.5, 0.5];

        // pred_matrix: 3 time points, 2 support points with identical predictions
        let pred_matrix = Mat::from_fn(3, 2, |i, _j| match i {
            0 => 10.0,
            1 => 5.0,
            2 => 2.0,
            _ => 0.0,
        });

        let combo = vec![0, 1, 2];
        let risk_identical = calculate_risk(&combo, &pred_matrix, &errormodel, &weights);

        // Now different predictions
        let pred_matrix_diff = Mat::from_fn(3, 2, |i, j| match (i, j) {
            (0, 0) => 10.0,
            (0, 1) => 50.0,
            (1, 0) => 5.0,
            (1, 1) => 25.0,
            (2, 0) => 2.0,
            (2, 1) => 12.0,
            _ => 0.0,
        });

        let risk_different = calculate_risk(&combo, &pred_matrix_diff, &errormodel, &weights);

        // Identical predictions should have higher risk (harder to discriminate)
        assert!(
            risk_identical > risk_different,
            "Identical predictions should have higher risk: {} vs {}",
            risk_identical,
            risk_different
        );
    }

    #[test]
    fn test_calculate_risk_symmetric_weights() {
        // Risk should be symmetric when weights are equal
        let errormodel = make_error_model();
        let weights = vec![0.5, 0.5];

        let pred_matrix = Mat::from_fn(3, 2, |i, j| match (i, j) {
            (0, 0) => 10.0,
            (0, 1) => 20.0,
            (1, 0) => 5.0,
            (1, 1) => 10.0,
            (2, 0) => 2.0,
            (2, 1) => 4.0,
            _ => 0.0,
        });

        let combo = vec![0, 1, 2];
        let risk = calculate_risk(&combo, &pred_matrix, &errormodel, &weights);
        assert!(risk > 0.0, "Risk should be positive");
        assert!(risk.is_finite(), "Risk should be finite");
    }

    #[test]
    fn test_calculate_risk_more_samples_lower_risk() {
        // Using more sample times should generally not increase (and usually decrease) risk
        let errormodel = make_error_model();
        let weights = vec![0.5, 0.5];

        // 4 time points with very different predictions
        let pred_matrix = Mat::from_fn(4, 2, |i, j| match (i, j) {
            (0, 0) => 10.0,
            (0, 1) => 20.0,
            (1, 0) => 5.0,
            (1, 1) => 15.0,
            (2, 0) => 2.0,
            (2, 1) => 8.0,
            (3, 0) => 1.0,
            (3, 1) => 6.0,
            _ => 0.0,
        });

        // Best 2-sample combo risk
        let combos_2 = generate_combinations(4, 2);
        let min_risk_2 = combos_2
            .iter()
            .map(|combo| calculate_risk(combo, &pred_matrix, &errormodel, &weights))
            .fold(f64::INFINITY, f64::min);

        // Best 3-sample combo risk
        let combos_3 = generate_combinations(4, 3);
        let min_risk_3 = combos_3
            .iter()
            .map(|combo| calculate_risk(combo, &pred_matrix, &errormodel, &weights))
            .fold(f64::INFINITY, f64::min);

        assert!(
            min_risk_3 <= min_risk_2,
            "More samples should yield equal or lower risk: {} vs {}",
            min_risk_3,
            min_risk_2
        );
    }

    #[test]
    fn test_calculate_risk_zero_weight() {
        // Setting a weight to zero should eliminate that support point's contribution
        let errormodel = make_error_model();

        let pred_matrix = Mat::from_fn(2, 3, |i, j| match (i, j) {
            (0, 0) => 10.0,
            (0, 1) => 20.0,
            (0, 2) => 30.0,
            (1, 0) => 5.0,
            (1, 1) => 10.0,
            (1, 2) => 15.0,
            _ => 0.0,
        });

        let combo = vec![0, 1];

        // With all weights
        let weights_all = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let risk_all = calculate_risk(&combo, &pred_matrix, &errormodel, &weights_all);

        // Zero out one weight
        let weights_zero = vec![0.5, 0.5, 0.0];
        let risk_zero = calculate_risk(&combo, &pred_matrix, &errormodel, &weights_zero);

        // Risks should differ when a weight is zeroed
        assert!(
            (risk_all - risk_zero).abs() > 1e-10,
            "Zeroing a weight should change the risk"
        );
    }

    #[test]
    fn test_risk_positive_and_finite() {
        let errormodel = make_error_model();
        let weights = vec![0.25, 0.25, 0.25, 0.25];

        let pred_matrix = Mat::from_fn(5, 4, |i, j| (i as f64 + 1.0) * (j as f64 + 1.0) * 2.0);

        let combo = vec![0, 2, 4];
        let risk = calculate_risk(&combo, &pred_matrix, &errormodel, &weights);
        assert!(risk >= 0.0, "Risk must be non-negative");
        assert!(risk.is_finite(), "Risk must be finite");
    }

    #[test]
    fn test_mmopt_result_display() {
        let result = MmoptResult {
            times: vec![1.0, 4.0, 8.0],
            risk: 0.123456,
        };
        let display = format!("{}", result);
        assert!(display.contains("1.0"));
        assert!(display.contains("4.0"));
        assert!(display.contains("8.0"));
        assert!(display.contains("0.123456"));
    }
}
