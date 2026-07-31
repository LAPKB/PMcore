//! Frozen-kernel Markov simulation-variance algebra for averaged SAEM.
//!
//! This deliberately small module implements non-overlapping multivariate
//! batch means and the Vats/Flegal lugsail combination.  It never repairs,
//! projects, regularizes, or replaces a reported matrix.

use anyhow::{bail, Result};
use ndarray::Array2;

use crate::algorithms::parametric::LugsailConfig;
use crate::estimation::parametric::covariance::cholesky_lower;

pub(crate) fn batch_means(samples: &[Vec<f64>], batch_size: usize) -> Result<Array2<f64>> {
    if samples.is_empty() || batch_size == 0 || !samples.len().is_multiple_of(batch_size) {
        bail!("batch-means samples must be non-empty and divisible by batch size");
    }
    let batches = samples.len() / batch_size;
    if batches < 2 {
        bail!("batch means require at least two batches");
    }
    let width = samples[0].len();
    if samples
        .iter()
        .any(|sample| sample.len() != width || sample.iter().any(|x| !x.is_finite()))
    {
        bail!("batch-means samples must have one finite coordinate width");
    }
    let mut overall = vec![0.0; width];
    for sample in samples {
        for j in 0..width {
            overall[j] += sample[j] / samples.len() as f64;
        }
    }
    let mut result = Array2::zeros((width, width));
    for batch in samples.chunks(batch_size) {
        let mut mean = vec![0.0; width];
        for sample in batch {
            for j in 0..width {
                mean[j] += sample[j] / batch_size as f64;
            }
        }
        for row in 0..width {
            for column in 0..width {
                result[[row, column]] +=
                    (mean[row] - overall[row]) * (mean[column] - overall[column]);
            }
        }
    }
    result.mapv_inplace(|value| value * batch_size as f64 / (batches - 1) as f64);
    Ok(result)
}

pub(crate) fn lugsail_batch_means(
    samples: &[Vec<f64>],
    batch_size: usize,
    lugsail: LugsailConfig,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    let coarse = batch_means(samples, batch_size)?;
    let fine = batch_means(samples, batch_size / lugsail.r)?;
    let lrv = (&coarse - &fine * lugsail.c) / (1.0 - lugsail.c);
    Ok((coarse, fine, lrv))
}

/// Independent chain averages have covariance C^-2 sum_c Lambda_c.
#[allow(dead_code)]
pub(crate) fn combine_independent_chain_lrvs(chains: &[Array2<f64>]) -> Result<Array2<f64>> {
    let Some(first) = chains.first() else {
        bail!("at least one chain LRV is required");
    };
    let shape = first.dim();
    if shape.0 != shape.1 || chains.iter().any(|matrix| matrix.dim() != shape) {
        bail!("chain LRV matrices must have one square dimension");
    }
    let mut combined = Array2::zeros(shape);
    for chain in chains {
        combined += chain;
    }
    combined /= (chains.len() * chains.len()) as f64;
    Ok(combined)
}

/// Scale a sum of `C_d` independent-chain LRVs both for the diagnostic-chain
/// mean (`sum / C_d^2`) and for the operational fit-chain mean
/// (`sum / (C_d C_f)`).
pub(crate) fn scale_lrv_sum(
    sum: &Array2<f64>,
    diagnostic_chains: usize,
    fit_chains: usize,
) -> (Array2<f64>, Array2<f64>) {
    debug_assert!(diagnostic_chains > 0 && fit_chains > 0);
    let diagnostic = sum / (diagnostic_chains as f64 * diagnostic_chains as f64);
    let operational = sum / (diagnostic_chains as f64 * fit_chains as f64);
    (diagnostic, operational)
}

/// Compute Iobs^-1 Lambda Iobs^-T and its Cesaro-average covariance Xi/n_avg.
pub(crate) fn transform_simulation_variance(
    information: &Array2<f64>,
    lambda: &Array2<f64>,
    n_avg: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    if n_avg == 0 || information.dim() != lambda.dim() || information.nrows() != information.ncols()
    {
        bail!("simulation-variance transform dimensions are invalid");
    }
    let inverse = inverse_spd_no_jitter(information)?;
    let n = inverse.nrows();
    let mut xi = Array2::zeros((n, n));
    for row in 0..n {
        for column in 0..=row {
            let mut value = 0.0;
            for left in 0..n {
                for right in 0..n {
                    value +=
                        inverse[[row, left]] * lambda[[left, right]] * inverse[[column, right]];
                }
            }
            xi[[row, column]] = value;
            xi[[column, row]] = value;
        }
    }
    let covariance = &xi / n_avg as f64;
    Ok((xi, covariance))
}

fn inverse_spd_no_jitter(matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let lower = cholesky_lower(matrix)?;
    let n = lower.len();
    let mut inverse = Array2::zeros((n, n));
    for column in 0..n {
        let mut y = vec![0.0; n];
        for row in 0..n {
            let rhs = usize::from(row == column) as f64;
            y[row] = (rhs - (0..row).map(|k| lower[row][k] * y[k]).sum::<f64>()) / lower[row][row];
        }
        let mut x = vec![0.0; n];
        for row in (0..n).rev() {
            x[row] = (y[row] - ((row + 1)..n).map(|k| lower[k][row] * x[k]).sum::<f64>())
                / lower[row][row];
        }
        for row in 0..n {
            inverse[[row, column]] = x[row];
        }
    }
    Ok(inverse)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatrixClassification {
    EligiblePsd,
    NonFinite,
    NonSymmetric,
    Indefinite,
}

pub(crate) fn classify_psd(matrix: &Array2<f64>) -> MatrixClassification {
    if matrix.nrows() != matrix.ncols() || matrix.iter().any(|value| !value.is_finite()) {
        return MatrixClassification::NonFinite;
    }
    let n = matrix.nrows();
    for row in 0..n {
        for column in 0..row {
            let scale = matrix[[row, column]]
                .abs()
                .max(matrix[[column, row]].abs())
                .max(1.0);
            if (matrix[[row, column]] - matrix[[column, row]]).abs() > 64.0 * f64::EPSILON * scale {
                return MatrixClassification::NonSymmetric;
            }
        }
    }
    // Symmetric Jacobi eigenvalues are used only for classification. The input
    // and every reported matrix remain byte-for-byte unmodified.
    let mut work = matrix.clone();
    for _ in 0..(50 * n.max(1) * n.max(1)) {
        let mut p = 0;
        let mut q = 0;
        let mut largest = 0.0;
        for row in 0..n {
            for column in 0..row {
                if work[[row, column]].abs() > largest {
                    largest = work[[row, column]].abs();
                    p = row;
                    q = column;
                }
            }
        }
        if largest <= 64.0 * f64::EPSILON * work.iter().fold(1.0_f64, |s, x| s.max(x.abs())) {
            break;
        }
        let angle = 0.5 * (2.0 * work[[p, q]]).atan2(work[[q, q]] - work[[p, p]]);
        let (sin, cos) = angle.sin_cos();
        for k in 0..n {
            if k == p || k == q {
                continue;
            }
            let kp = work[[k, p]];
            let kq = work[[k, q]];
            work[[k, p]] = cos * kp - sin * kq;
            work[[p, k]] = work[[k, p]];
            work[[k, q]] = sin * kp + cos * kq;
            work[[q, k]] = work[[k, q]];
        }
        let pp = work[[p, p]];
        let qq = work[[q, q]];
        let pq = work[[p, q]];
        work[[p, p]] = cos * cos * pp - 2.0 * sin * cos * pq + sin * sin * qq;
        work[[q, q]] = sin * sin * pp + 2.0 * sin * cos * pq + cos * cos * qq;
        work[[p, q]] = 0.0;
        work[[q, p]] = 0.0;
    }
    let scale = matrix.iter().fold(1.0_f64, |s, x| s.max(x.abs()));
    if (0..n).any(|index| work[[index, index]] < -256.0 * f64::EPSILON * scale) {
        MatrixClassification::Indefinite
    } else {
        MatrixClassification::EligiblePsd
    }
}

pub(crate) fn rows(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.rows().into_iter().map(|row| row.to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_and_two_dimensional_batch_means_are_hand_calculated() {
        let scalar = batch_means(&[vec![1.0], vec![3.0], vec![5.0], vec![7.0]], 2).unwrap();
        assert_eq!(scalar[[0, 0]], 16.0);
        let two = batch_means(
            &[
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![5.0, 8.0],
                vec![7.0, 10.0],
            ],
            2,
        )
        .unwrap();
        assert_eq!(two, ndarray::array![[16.0, 24.0], [24.0, 36.0]]);
    }

    #[test]
    fn over_lugsail_and_chain_scaling_are_exact() {
        let samples = [
            vec![1.0],
            vec![3.0],
            vec![5.0],
            vec![7.0],
            vec![9.0],
            vec![11.0],
        ];
        let (b, fine, lrv) =
            lugsail_batch_means(&samples, 3, LugsailConfig::over_lugsail_bartlett()).unwrap();
        assert_eq!(b[[0, 0]], 54.0);
        assert_eq!(fine[[0, 0]], 14.0);
        assert_eq!(lrv[[0, 0]], 94.0);
        assert_eq!(
            combine_independent_chain_lrvs(&[lrv.clone(), lrv]).unwrap()[[0, 0]],
            47.0
        );
    }

    #[test]
    fn unequal_diagnostic_and_fit_chain_scaling_is_exact() {
        // Hard-coded sum of three 2x2 chain LRVs with C_d=3 and C_f=2.
        let sum = ndarray::array![[18.0, 9.0], [9.0, 36.0]];
        let (diagnostic, operational) = scale_lrv_sum(&sum, 3, 2);
        assert_eq!(diagnostic, ndarray::array![[2.0, 1.0], [1.0, 4.0]]);
        assert_eq!(operational, ndarray::array![[3.0, 1.5], [1.5, 6.0]]);
    }

    #[test]
    fn transform_and_indefinite_classification_retain_raw_values() {
        let information = ndarray::array![[2.0, 0.0], [0.0, 4.0]];
        let lambda = ndarray::array![[8.0, 4.0], [4.0, 8.0]];
        let (xi, covariance) = transform_simulation_variance(&information, &lambda, 5).unwrap();
        for (actual, expected) in xi.iter().zip([2.0, 0.5, 0.5, 0.5]) {
            assert!((actual - expected).abs() < 1e-14);
        }
        for (actual, expected) in covariance.iter().zip([0.4, 0.1, 0.1, 0.1]) {
            assert!((actual - expected).abs() < 1e-14);
        }
        let indefinite = ndarray::array![[1.0, 2.0], [2.0, 1.0]];
        assert_eq!(classify_psd(&indefinite), MatrixClassification::Indefinite);
        assert_eq!(indefinite[[0, 1]], 2.0);
    }

    #[test]
    fn per_chain_indefiniteness_survives_psd_aggregate_cancellation() {
        let first = ndarray::array![[-1.0, 0.0], [0.0, 3.0]];
        let second = ndarray::array![[3.0, 0.0], [0.0, -1.0]];
        assert_eq!(classify_psd(&first), MatrixClassification::Indefinite);
        assert_eq!(classify_psd(&second), MatrixClassification::Indefinite);
        let combined = combine_independent_chain_lrvs(&[first.clone(), second.clone()]).unwrap();
        assert_eq!(combined, ndarray::array![[0.5, 0.0], [0.0, 0.5]]);
        assert_eq!(classify_psd(&combined), MatrixClassification::EligiblePsd);
        assert_eq!(first[[0, 0]], -1.0);
        assert_eq!(second[[1, 1]], -1.0);
    }

    #[test]
    fn transformed_indefinite_matrices_are_finite_and_remain_indefinite() {
        let information = ndarray::array![[2.0, 0.0], [0.0, 4.0]];
        let lambda = ndarray::array![[-2.0, 0.0], [0.0, 4.0]];
        let (xi, covariance) = transform_simulation_variance(&information, &lambda, 2).unwrap();
        for (actual, expected) in xi.iter().zip([-0.5, 0.0, 0.0, 0.25]) {
            assert!((actual - expected).abs() < 1e-14);
        }
        for (actual, expected) in covariance.iter().zip([-0.25, 0.0, 0.0, 0.125]) {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert_eq!(classify_psd(&xi), MatrixClassification::Indefinite);
        assert_eq!(classify_psd(&covariance), MatrixClassification::Indefinite);
        assert!(xi
            .iter()
            .chain(covariance.iter())
            .all(|value| value.is_finite()));
    }

    #[test]
    fn finite_symmetry_and_psd_are_classified_without_repair() {
        assert_eq!(
            classify_psd(&ndarray::array![[1.0, 1.0], [1.0, 1.0]]),
            MatrixClassification::EligiblePsd
        );
        assert_eq!(
            classify_psd(&ndarray::array![[1.0, 0.5], [0.25, 1.0]]),
            MatrixClassification::NonSymmetric
        );
        assert_eq!(
            classify_psd(&ndarray::array![[f64::NAN]]),
            MatrixClassification::NonFinite
        );
    }
}
