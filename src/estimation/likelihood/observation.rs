use pharmsol::prelude::simulator::Prediction;
use pharmsol::Censor;
use pharmsol::Predictions;

use crate::{AssayErrorModels, ErrorModelError};

use super::distributions::{
    log_normal_ccdf, log_normal_cdf, log_normal_pdf, NormalDistributionError,
};

/// Typed scoring error for assay observation likelihood.
///
/// This is the public error type returned by [`crate::AssayErrorModels::log_likelihood`].
/// It keeps a valid-but-impossible score ([`AssayLikelihoodError::Impossible`],
/// a log-likelihood of negative infinity) distinct from invalid inputs such as
/// an out-of-range standard deviation or non-finite observation/prediction
/// ([`AssayLikelihoodError::Distribution`]).
#[derive(Debug, thiserror::Error)]
pub enum AssayLikelihoodError {
    /// The assay error model itself was invalid for the scored prediction.
    #[error("invalid assay error model")]
    ErrorModel(#[from] ErrorModelError),
    /// The normal distribution rejected the standardized inputs (invalid sigma
    /// or non-finite observation/prediction).
    #[error(transparent)]
    Distribution(#[from] NormalDistributionError),
    /// The log-likelihood is negative infinity: a valid but impossible score.
    #[error("assay log-likelihood is negative infinity")]
    Impossible,
    /// The log-likelihood is NaN or positive infinity, which is never valid.
    #[error("assay log-likelihood is NaN or positive infinity: {0}")]
    InvalidScore(f64),
}

/// Apply assay error-model likelihood logic to one pharmsol prediction DTO.
///
/// This preserves the current Pmetrics/non-parametric semantics: sigma is
/// observation/assay based and comes from [`AssayErrorModels`]. The likelihood
/// math and non-finite handling live here rather than in pharmsol.
#[inline]
pub(crate) fn assay_error_model_log_likelihood(
    prediction: &Prediction,
    error_models: &AssayErrorModels,
) -> std::result::Result<f64, AssayLikelihoodError> {
    let Some(obs) = prediction.observation() else {
        return Ok(0.0);
    };

    let sigma = error_models.sigma(prediction)?;

    let log_lik = match prediction.censoring() {
        Censor::None => log_normal_pdf(obs, prediction.prediction(), sigma)?,
        Censor::BLOQ => log_normal_cdf(obs, prediction.prediction(), sigma)?,
        Censor::ALOQ => log_normal_ccdf(obs, prediction.prediction(), sigma)?,
    };

    if log_lik.is_finite() {
        Ok(log_lik)
    } else if log_lik == f64::NEG_INFINITY {
        Err(AssayLikelihoodError::Impossible)
    } else {
        Err(AssayLikelihoodError::InvalidScore(log_lik))
    }
}

pub(crate) fn assay_error_model_log_likelihoods<P>(
    predictions: &P,
    error_models: &AssayErrorModels,
) -> std::result::Result<f64, AssayLikelihoodError>
where
    P: Predictions,
{
    let mut total = 0.0;
    let mut error = None;
    predictions.for_each_prediction(|prediction| {
        if error.is_some() {
            return;
        }
        match assay_error_model_log_likelihood(prediction, error_models) {
            Ok(ll) => total += ll,
            Err(err) => error = Some(err),
        }
    });

    match error {
        Some(err) => Err(err),
        None => Ok(total),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_observations_score_zero() {
        let prediction = Prediction::default();
        assert_eq!(
            assay_error_model_log_likelihood(&prediction, &AssayErrorModels::new()).unwrap(),
            0.0
        );
    }
}
