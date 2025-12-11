//! Mathematical utility functions for numerical stability
//!
//! This module provides stable implementations of common numerical operations.

/// Compute the log-sum-exp of a slice of values in a numerically stable way.
///
/// The log-sum-exp is defined as: `log(sum(exp(x_i)))` for all elements `x_i`.
///
/// This implementation uses the "shift by max" trick to avoid overflow:
/// `logsumexp(x) = max(x) + log(sum(exp(x_i - max(x))))`
///
/// # Arguments
/// * `values` - A slice of f64 values (typically log-likelihoods)
///
/// # Returns
/// The log-sum-exp of the values. Returns `f64::NEG_INFINITY` if all values are `-inf`.
///
/// # Example
/// ```ignore
/// let log_probs = vec![-1.0, -2.0, -3.0];
/// let result = logsumexp(&log_probs);
/// // result ≈ log(exp(-1) + exp(-2) + exp(-3)) ≈ -0.407
/// ```
#[inline]
pub fn logsumexp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if max_val.is_infinite() && max_val.is_sign_negative() {
        // All values are -inf, return -inf
        f64::NEG_INFINITY
    } else if max_val.is_infinite() && max_val.is_sign_positive() {
        // At least one value is +inf
        f64::INFINITY
    } else {
        max_val
            + values
                .iter()
                .map(|&x| (x - max_val).exp())
                .sum::<f64>()
                .ln()
    }
}

/// Compute the weighted log-sum-exp: `logsumexp(log_values + log_weights)`.
///
/// This computes `log(sum(values_i * weights_i))` when `log_values` contains log-likelihoods.
/// Equivalent to `log(sum(exp(log_values_i) * weights_i))`.
///
/// # Arguments
/// * `log_values` - A slice of log-values (e.g., log-likelihoods)
/// * `log_weights` - A slice of log-weights (should be same length as log_values)
///
/// # Returns
/// The weighted log-sum-exp. Panics if slices have different lengths.
#[inline]
pub fn logsumexp_weighted(log_values: &[f64], log_weights: &[f64]) -> f64 {
    assert_eq!(
        log_values.len(),
        log_weights.len(),
        "log_values and log_weights must have the same length"
    );

    let combined: Vec<f64> = log_values
        .iter()
        .zip(log_weights.iter())
        .map(|(&lv, &lw)| lv + lw)
        .collect();

    logsumexp(&combined)
}

/// Compute log-sum-exp for each row of a matrix represented as a closure.
///
/// # Arguments
/// * `nrows` - Number of rows
/// * `ncols` - Number of columns
/// * `get_value` - Closure that returns the value at (row, col)
///
/// # Returns
/// A vector of logsumexp values, one per row.
pub fn logsumexp_rows<F>(nrows: usize, ncols: usize, get_value: F) -> Vec<f64>
where
    F: Fn(usize, usize) -> f64,
{
    (0..nrows)
        .map(|i| {
            let row: Vec<f64> = (0..ncols).map(|j| get_value(i, j)).collect();
            logsumexp(&row)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logsumexp_basic() {
        let values = vec![-1.0, -2.0, -3.0];
        let result = logsumexp(&values);
        // log(exp(-1) + exp(-2) + exp(-3)) ≈ -0.4076
        let expected = ((-1.0_f64).exp() + (-2.0_f64).exp() + (-3.0_f64).exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_single_value() {
        let values = vec![-5.0];
        let result = logsumexp(&values);
        assert!((result - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_empty() {
        let values: Vec<f64> = vec![];
        let result = logsumexp(&values);
        assert!(result.is_infinite() && result.is_sign_negative());
    }

    #[test]
    fn test_logsumexp_all_neg_inf() {
        let values = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        let result = logsumexp(&values);
        assert!(result.is_infinite() && result.is_sign_negative());
    }

    #[test]
    fn test_logsumexp_with_neg_inf() {
        // logsumexp([-inf, 0]) = log(0 + 1) = 0
        let values = vec![f64::NEG_INFINITY, 0.0];
        let result = logsumexp(&values);
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_large_values() {
        // Test numerical stability with large values
        let values = vec![1000.0, 1001.0, 1002.0];
        let result = logsumexp(&values);
        // Should be close to 1002 + log(exp(-2) + exp(-1) + 1) ≈ 1002.41
        let expected = 1002.0 + ((-2.0_f64).exp() + (-1.0_f64).exp() + 1.0).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_very_negative() {
        // Test with very negative values that would underflow with naive implementation
        let values = vec![-1000.0, -1001.0, -1002.0];
        let result = logsumexp(&values);
        let expected = -1000.0 + (1.0 + (-1.0_f64).exp() + (-2.0_f64).exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_weighted() {
        let log_values = vec![-1.0, -2.0];
        let log_weights = vec![0.0, 0.0]; // weights = 1
        let result = logsumexp_weighted(&log_values, &log_weights);
        let expected = logsumexp(&log_values);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_rows() {
        let matrix = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
        let result = logsumexp_rows(2, 2, |i, j| matrix[i][j]);

        let expected_0 = logsumexp(&[-1.0, -2.0]);
        let expected_1 = logsumexp(&[-3.0, -4.0]);

        assert!((result[0] - expected_0).abs() < 1e-10);
        assert!((result[1] - expected_1).abs() < 1e-10);
    }
}
