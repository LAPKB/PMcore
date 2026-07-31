use statrs::function::erf::erfc;
use thiserror::Error;

const LOG_2PI: f64 = 1.8378770664093453_f64;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Invalid inputs to PMcore's normal-distribution scoring primitives.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum NormalDistributionError {
    /// The standard deviation was non-finite, zero, or negative.
    #[error("normal standard deviation must be finite and greater than zero, got {0}")]
    InvalidSigma(f64),
    /// The observation or prediction was non-finite.
    #[error("normal observation and prediction must be finite")]
    NonFiniteInput,
}

fn standardized(obs: f64, pred: f64, sigma: f64) -> Result<f64, NormalDistributionError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(NormalDistributionError::InvalidSigma(sigma));
    }
    if !obs.is_finite() || !pred.is_finite() {
        return Err(NormalDistributionError::NonFiniteInput);
    }
    Ok((obs - pred) / sigma)
}

/// Log of the standard-normal upper-tail probability for non-negative `z`.
///
/// `erfc` evaluates the tail directly through the range where it is
/// representable. The Mills-ratio expansion avoids underflow beyond that range.
fn log_standard_normal_upper_tail(z: f64) -> f64 {
    debug_assert!(z >= 0.0);
    let direct = 0.5 * erfc(z / SQRT_2);
    if direct > 0.0 {
        return direct.ln();
    }

    let inverse_square = 1.0 / (z * z);
    let mills_series = 1.0
        - inverse_square
            * (1.0
                - inverse_square
                    * (3.0
                        - inverse_square
                            * (15.0 - inverse_square * (105.0 - inverse_square * 945.0))));
    -0.5 * z * z - z.ln() - 0.5 * LOG_2PI + mills_series.ln()
}

#[inline(always)]
pub(crate) fn log_normal_pdf(
    obs: f64,
    pred: f64,
    sigma: f64,
) -> Result<f64, NormalDistributionError> {
    let z = standardized(obs, pred, sigma)?;
    Ok(-0.5 * LOG_2PI - sigma.ln() - 0.5 * z * z)
}

#[inline(always)]
pub(crate) fn log_normal_cdf(
    obs: f64,
    pred: f64,
    sigma: f64,
) -> Result<f64, NormalDistributionError> {
    let z = standardized(obs, pred, sigma)?;
    if z <= 0.0 {
        Ok(log_standard_normal_upper_tail(-z))
    } else {
        Ok((-log_standard_normal_upper_tail(z).exp()).ln_1p())
    }
}

#[inline(always)]
pub(crate) fn log_normal_ccdf(
    obs: f64,
    pred: f64,
    sigma: f64,
) -> Result<f64, NormalDistributionError> {
    let z = standardized(obs, pred, sigma)?;
    if z >= 0.0 {
        Ok(log_standard_normal_upper_tail(z))
    } else {
        Ok((-log_standard_normal_upper_tail(-z).exp()).ln_1p())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_matches_standard_normal_at_mean() {
        let ll = log_normal_pdf(0.0, 0.0, 1.0).unwrap();
        assert!((ll + 0.5 * LOG_2PI).abs() < 1e-12);
    }

    #[test]
    fn cdf_and_survival_are_symmetric() {
        for z in [-100.0, -37.0, -8.0, 0.0, 8.0, 37.0, 100.0] {
            assert_eq!(
                log_normal_cdf(z, 0.0, 1.0).unwrap(),
                log_normal_ccdf(-z, 0.0, 1.0).unwrap()
            );
        }
    }

    #[test]
    fn survival_is_finite_across_cancellation_and_extreme_tails() {
        for z in [8.0, 12.0, 20.0, 37.0, 40.0, 100.0, 1_000.0] {
            let value = log_normal_ccdf(z, 0.0, 1.0).unwrap();
            assert!(value.is_finite(), "z={z}, log survival={value}");
            assert!(value < 0.0);
        }
    }

    #[test]
    fn invalid_sigma_is_typed() {
        for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                log_normal_pdf(0.0, 0.0, sigma),
                Err(NormalDistributionError::InvalidSigma(value)) if value.to_bits() == sigma.to_bits()
            ));
            assert!(matches!(
                log_normal_ccdf(0.0, 0.0, sigma),
                Err(NormalDistributionError::InvalidSigma(value)) if value.to_bits() == sigma.to_bits()
            ));
        }
    }

    #[test]
    fn non_finite_pdf_inputs_are_typed() {
        for (observation, prediction) in [(f64::NAN, 0.0), (0.0, f64::INFINITY)] {
            assert_eq!(
                log_normal_pdf(observation, prediction, 1.0),
                Err(NormalDistributionError::NonFiniteInput)
            );
        }
    }
}
