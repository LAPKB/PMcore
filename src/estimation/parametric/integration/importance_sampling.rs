//! Importance Sampling for Marginal Likelihood Estimation
//!
//! This module implements importance sampling (IS) for estimating the marginal
//! log-likelihood in mixed-effects models, matching R saemix's `llis.saemix` function.

use anyhow::Result;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Col, Mat};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StudentT};
use statrs::function::gamma::ln_gamma;

use crate::estimation::parametric::ParameterTransform;
use pharmsol::{Equation, Predictions, ResidualErrorModels, Subject};

#[derive(Debug, Clone)]
pub struct ImportanceSamplingConfig {
    pub n_samples: usize,
    pub nu: f64,
    pub seed: u64,
}

impl Default for ImportanceSamplingConfig {
    fn default() -> Self {
        Self {
            n_samples: 5000,
            nu: 4.0,
            seed: 123456,
        }
    }
}

impl ImportanceSamplingConfig {
    pub fn saemix_defaults() -> Self {
        Self::default()
    }

    pub fn with_n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    pub fn with_nu(mut self, nu: f64) -> Self {
        self.nu = nu;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

#[derive(Debug, Clone)]
pub struct SubjectConditionalPosterior {
    pub mean: Col<f64>,
    pub variance: Vec<f64>,
}

impl SubjectConditionalPosterior {
    pub fn from_mcmc_samples(
        eta_samples: &[Col<f64>],
        mu_phi: &Col<f64>,
        omega: &Mat<f64>,
    ) -> Self {
        let n_params = mu_phi.nrows();
        let n_samples = eta_samples.len();

        if n_samples == 0 {
            return Self {
                mean: mu_phi.clone(),
                variance: (0..n_params).map(|j| omega[(j, j)].max(1e-6)).collect(),
            };
        }

        let mut mean_eta = vec![0.0; n_params];
        for eta in eta_samples {
            for j in 0..n_params {
                mean_eta[j] += eta[j];
            }
        }
        for value in &mut mean_eta {
            *value /= n_samples as f64;
        }

        let mean = Col::from_fn(n_params, |j| mu_phi[j] + mean_eta[j]);

        let variance = if n_samples > 1 {
            let mut var = vec![0.0; n_params];
            for eta in eta_samples {
                for j in 0..n_params {
                    let diff = eta[j] - mean_eta[j];
                    var[j] += diff * diff;
                }
            }
            for value in &mut var {
                *value = (*value / (n_samples - 1) as f64).max(1e-6);
            }
            var
        } else {
            (0..n_params).map(|j| omega[(j, j)].max(1e-6)).collect()
        };

        Self { mean, variance }
    }

    pub fn new(mean: Col<f64>, variance: Vec<f64>) -> Self {
        Self { mean, variance }
    }
}

pub struct ImportanceSamplingEstimator<'a, E: Equation> {
    config: ImportanceSamplingConfig,
    equation: &'a E,
    error_models: &'a ResidualErrorModels,
    transforms: &'a [ParameterTransform],
    mu_phi: &'a Col<f64>,
    #[allow(dead_code)]
    omega: &'a Mat<f64>,
    omega_inv: Mat<f64>,
    log_prior_const: f64,
}

impl<'a, E: Equation> ImportanceSamplingEstimator<'a, E> {
    pub fn new(
        config: ImportanceSamplingConfig,
        equation: &'a E,
        error_models: &'a ResidualErrorModels,
        transforms: &'a [ParameterTransform],
        mu_phi: &'a Col<f64>,
        omega: &'a Mat<f64>,
    ) -> Result<Self> {
        let omega_inv = omega
            .llt(faer::Side::Lower)
            .map_err(|_| anyhow::anyhow!("Omega not positive definite"))?
            .inverse();
        let n_params = mu_phi.nrows();
        let log_det_omega = omega.determinant().ln();
        let log_prior_const = log_det_omega + (n_params as f64) * (2.0 * std::f64::consts::PI).ln();

        Ok(Self {
            config,
            equation,
            error_models,
            transforms,
            mu_phi,
            omega,
            omega_inv,
            log_prior_const,
        })
    }

    pub fn estimate_subject_ll(
        &self,
        subject: &Subject,
        conditional: &SubjectConditionalPosterior,
        rng: &mut impl rand::Rng,
    ) -> f64 {
        let n_params = self.mu_phi.nrows();
        let n_samples = self.config.n_samples;
        let nu = self.config.nu;

        let t_dist = StudentT::new(nu).expect("Invalid nu for StudentT");
        let mut log_weights = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let r: Vec<f64> = (0..n_params).map(|_| t_dist.sample(rng)).collect();
            let phi_sample = Col::from_fn(n_params, |j| {
                conditional.mean[j] + conditional.variance[j].sqrt() * r[j]
            });
            let psi_sample: Vec<f64> = (0..n_params)
                .map(|j| self.transforms[j].phi_to_psi(phi_sample[j]))
                .collect();

            let log_lik = match self.equation.estimate_predictions(subject, &psi_sample) {
                Ok(predictions) => {
                    let obs_pred_pairs = predictions
                        .get_predictions()
                        .into_iter()
                        .filter_map(|pred| pred.observation().map(|obs| (pred.outeq(), obs, pred.prediction())));
                    self.error_models.total_log_likelihood(obs_pred_pairs)
                }
                Err(_) => continue,
            };

            let eta = Col::from_fn(n_params, |j| phi_sample[j] - self.mu_phi[j]);
            let mut quad_form = 0.0;
            for j in 0..n_params {
                for k in 0..n_params {
                    quad_form += eta[j] * self.omega_inv[(j, k)] * eta[k];
                }
            }
            let log_prior = -0.5 * (quad_form + self.log_prior_const);

            let mut log_proposal = 0.0;
            for j in 0..n_params {
                let log_t_pdf = ln_gamma((nu + 1.0) / 2.0)
                    - ln_gamma(nu / 2.0)
                    - 0.5 * (nu * std::f64::consts::PI).ln()
                    - ((nu + 1.0) / 2.0) * (1.0 + r[j] * r[j] / nu).ln();
                log_proposal += log_t_pdf;
            }
            log_proposal -= 0.5 * conditional.variance.iter().map(|v| v.ln()).sum::<f64>();

            log_weights.push(log_lik + log_prior - log_proposal);
        }

        if log_weights.is_empty() {
            return f64::NEG_INFINITY;
        }

        let max_weight = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !max_weight.is_finite() {
            return f64::NEG_INFINITY;
        }

        let sum_exp: f64 = log_weights.iter().map(|&w| (w - max_weight).exp()).sum();
        max_weight + sum_exp.ln() - (log_weights.len() as f64).ln()
    }

    pub fn estimate_minus2ll(
        &self,
        subjects: Vec<&Subject>,
        conditionals: &[SubjectConditionalPosterior],
    ) -> f64 {
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        let mut total_ll = 0.0;

        for (subject, conditional) in subjects.iter().zip(conditionals.iter()) {
            total_ll += self.estimate_subject_ll(subject, conditional, &mut rng);
        }

        -2.0 * total_ll
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ImportanceSamplingConfig::default();
        assert_eq!(config.n_samples, 5000);
        assert_eq!(config.nu, 4.0);
    }

    #[test]
    fn test_conditional_posterior_from_empty() {
        let mu_phi = Col::from_fn(2, |i| (i + 1) as f64);
        let omega = Mat::from_fn(2, 2, |i, j| if i == j { 0.5 } else { 0.0 });

        let conditional = SubjectConditionalPosterior::from_mcmc_samples(&[], &mu_phi, &omega);

        assert_eq!(conditional.mean[0], 1.0);
        assert_eq!(conditional.mean[1], 2.0);
        assert!((conditional.variance[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_conditional_posterior_from_samples() {
        let mu_phi = Col::from_fn(2, |_| 0.0);
        let omega = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });

        let samples = vec![
            Col::from_fn(2, |_| 1.0),
            Col::from_fn(2, |_| 2.0),
            Col::from_fn(2, |_| 3.0),
        ];

        let conditional = SubjectConditionalPosterior::from_mcmc_samples(&samples, &mu_phi, &omega);

        assert!((conditional.mean[0] - 2.0).abs() < 1e-6);
        assert!((conditional.variance[0] - 1.0).abs() < 1e-6);
    }
}