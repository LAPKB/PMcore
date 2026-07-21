use pharmsol::prelude::simulator::Prediction;
use pharmsol::Predictions;

use crate::{ResidualErrorModel, ResidualErrorModels};

use super::distributions::log_normal_pdf;

#[inline]
fn residual_log_likelihood_values(
    model: &ResidualErrorModel,
    observation: f64,
    prediction: f64,
) -> f64 {
    match model {
        ResidualErrorModel::Exponential { .. } => {
            if !observation.is_finite()
                || observation <= 0.0
                || !prediction.is_finite()
                || prediction <= 0.0
            {
                return f64::NEG_INFINITY;
            }
            // Fit log(y) = log(f) + sigma*epsilon. The Jacobian
            // converts the transformed Gaussian density back to the original
            // observation scale; it is constant during latent-state MCMC but
            // required for an honest likelihood value.
            log_normal_pdf(observation.ln(), prediction.ln(), model.sigma(prediction))
                .unwrap_or(f64::NEG_INFINITY)
                - observation.ln()
        }
        ResidualErrorModel::Constant { .. }
        | ResidualErrorModel::Proportional { .. }
        | ResidualErrorModel::Combined { .. }
        | ResidualErrorModel::CorrelatedCombined { .. } => {
            log_normal_pdf(observation, prediction, model.sigma(prediction))
                .unwrap_or(f64::NEG_INFINITY)
        }
    }
}

/// Apply parametric residual-error-model likelihood semantics to one prediction.
///
/// The Gaussian normalization term is retained through `log_normal_pdf`.
/// Proportional coefficients are SD-scale values and use `b * abs(prediction)`;
/// negative predictions remain signed in the residual, matching the direct
/// raw-prediction E-step. Exponential error uses a positive-only lognormal
/// observation model and includes the original-scale Jacobian. Censoring is not
/// handled on this path.
#[inline]
pub(crate) fn residual_error_model_log_likelihood(
    prediction: &Prediction,
    error_models: &ResidualErrorModels,
) -> f64 {
    let Some(obs) = prediction.observation() else {
        return 0.0;
    };

    let Some(model) = error_models.get(prediction.outeq()) else {
        return f64::NEG_INFINITY;
    };

    residual_log_likelihood_values(model, obs, prediction.prediction())
}

pub(crate) fn residual_error_model_log_likelihoods<P>(
    predictions: &P,
    error_models: &ResidualErrorModels,
) -> f64
where
    P: Predictions,
{
    let mut total = 0.0;
    let mut failed = false;
    predictions.for_each_prediction(|prediction| {
        if failed {
            return;
        }
        let ll = residual_error_model_log_likelihood(prediction, error_models);
        if ll.is_finite() {
            total += ll;
        } else {
            failed = true;
        }
    });

    if failed {
        f64::NEG_INFINITY
    } else {
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_sigma_is_sd_scaled_symmetric_and_floored_at_zero() {
        let model = ResidualErrorModel::proportional(0.1);

        assert_eq!(model.sigma(10.0), 1.0);
        assert_eq!(model.sigma(-10.0), 1.0);
        assert_eq!(model.sigma(0.0), f64::EPSILON.sqrt());
    }

    #[test]
    fn correlated_combined_likelihood_uses_exact_signed_variance_and_log_term() {
        let model = ResidualErrorModel::correlated_combined(0.7, 0.2, -0.35);
        for prediction in [-3.0_f64, 0.0, 2.5] {
            let observation = prediction + 0.4;
            let variance = 0.7_f64.powi(2)
                + 2.0 * -0.35 * 0.7 * 0.2 * prediction
                + 0.2_f64.powi(2) * prediction.powi(2);
            let expected = -0.5
                * ((2.0 * std::f64::consts::PI).ln() + variance.ln() + 0.4_f64.powi(2) / variance);
            assert!(
                (residual_log_likelihood_values(&model, observation, prediction) - expected).abs()
                    < 1e-12
            );
        }

        let ordinary = ResidualErrorModel::combined(0.7, 0.2);
        let independent = ResidualErrorModel::correlated_combined(0.7, 0.2, 0.0);
        for prediction in [-3.0_f64, 0.0, 2.5] {
            let observation = prediction + 0.4;
            assert_eq!(
                residual_log_likelihood_values(&ordinary, observation, prediction),
                residual_log_likelihood_values(&independent, observation, prediction)
            );
        }
    }

    #[test]
    fn exponential_likelihood_is_lognormal_with_original_scale_jacobian() {
        let model = ResidualErrorModel::exponential(0.25);
        let observation: f64 = 12.0;
        let prediction: f64 = 10.0;
        let expected =
            log_normal_pdf(observation.ln(), prediction.ln(), 0.25).unwrap() - observation.ln();

        assert!(
            (residual_log_likelihood_values(&model, observation, prediction) - expected).abs()
                < 1e-12
        );
        assert_eq!(
            residual_log_likelihood_values(&model, 0.0, prediction),
            f64::NEG_INFINITY
        );
        assert_eq!(
            residual_log_likelihood_values(&model, observation, -prediction),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn exponential_likelihood_uses_the_canonical_scale_floor() {
        let observation: f64 = 12.0;
        let prediction: f64 = 10.0;
        let floor = f64::EPSILON.sqrt();
        let expected =
            log_normal_pdf(observation.ln(), prediction.ln(), floor).unwrap() - observation.ln();

        for sigma in [floor / 2.0, floor] {
            let model = ResidualErrorModel::exponential(sigma);
            assert_eq!(
                residual_log_likelihood_values(&model, observation, prediction),
                expected
            );
        }

        let above_floor = floor * 2.0;
        let model = ResidualErrorModel::exponential(above_floor);
        let expected_above = log_normal_pdf(observation.ln(), prediction.ln(), above_floor)
            .unwrap()
            - observation.ln();
        assert_eq!(
            residual_log_likelihood_values(&model, observation, prediction),
            expected_above
        );
    }

    #[test]
    fn exponential_fixed_trace_likelihood_matches_reference_checkpoint() {
        let model = ResidualErrorModel::exponential(0.156_764_356_228_701_02);
        let observations = [12.0, 8.0, 4.5, 1.2];
        let predictions = [10.0, 7.5, 5.0, 1.5];
        let log_likelihood = observations
            .into_iter()
            .zip(predictions)
            .map(|(observation, prediction)| {
                residual_log_likelihood_values(&model, observation, prediction)
            })
            .sum::<f64>();

        assert!((log_likelihood - -4.514_455_210_348_959).abs() < 1e-14);
    }

    #[test]
    fn missing_observations_score_zero() {
        assert_eq!(
            residual_error_model_log_likelihood(
                &Prediction::default(),
                &ResidualErrorModels::new(),
            ),
            0.0
        );
    }
}
