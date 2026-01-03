//! Importance Sampling for Marginal Likelihood Estimation
//!
//! This module implements importance sampling (IS) for estimating the marginal
//! log-likelihood in mixed-effects models, matching R saemix's `llis.saemix` function.
//!
//! # Algorithm
//!
//! For each subject i, we estimate:
//!
//! ```text
//! p(yᵢ | θ) ≈ (1/M) Σₘ [p(yᵢ | φᵢₘ) × p(φᵢₘ | θ) / q(φᵢₘ)]
//! ```
//!
//! where:
//! - φᵢₘ ~ q(·) is drawn from a proposal distribution (Student's t)
//! - p(yᵢ | φᵢₘ) is the data likelihood
//! - p(φᵢₘ | θ) is the population prior (multivariate normal)
//! - q(φᵢₘ) is the proposal density
//!
//! # Reference
//!
//! Kuhn, E., & Lavielle, M. (2005). Maximum likelihood estimation in nonlinear
//! mixed effects models. Computational Statistics & Data Analysis, 49(4), 1020-1038.

use anyhow::Result;
use faer::{Col, Mat};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StudentT};
use statrs::function::gamma::ln_gamma;

use crate::structs::parametric::transform::ParameterTransform;
use pharmsol::{Equation, Predictions, ResidualErrorModels, Subject};

/// Configuration for importance sampling likelihood estimation
#[derive(Debug, Clone)]
pub struct ImportanceSamplingConfig {
    /// Number of Monte Carlo samples per subject
    pub n_samples: usize,
    /// Degrees of freedom for Student's t proposal distribution
    pub nu: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for ImportanceSamplingConfig {
    fn default() -> Self {
        Self {
            n_samples: 5000,
            nu: 4.0, // R saemix default
            seed: 123456,
        }
    }
}

impl ImportanceSamplingConfig {
    /// Create with R saemix defaults
    pub fn saemix_defaults() -> Self {
        Self::default()
    }

    /// Create with custom number of samples
    pub fn with_n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set degrees of freedom for t-distribution
    pub fn with_nu(mut self, nu: f64) -> Self {
        self.nu = nu;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Conditional posterior information for a single subject
///
/// Contains the mean and variance of the conditional distribution p(φᵢ | yᵢ, θ)
/// estimated from MCMC samples.
#[derive(Debug, Clone)]
pub struct SubjectConditionalPosterior {
    /// Conditional mean of φ for this subject
    pub mean: Col<f64>,
    /// Conditional variance (diagonal) of φ for this subject
    pub variance: Vec<f64>,
}

impl SubjectConditionalPosterior {
    /// Create from MCMC chain states (eta values)
    ///
    /// # Arguments
    ///
    /// * `eta_samples` - MCMC samples of η (random effects)
    /// * `mu_phi` - Population mean in φ space
    /// * `omega` - Population covariance matrix (for fallback variance)
    pub fn from_mcmc_samples(
        eta_samples: &[Col<f64>],
        mu_phi: &Col<f64>,
        omega: &Mat<f64>,
    ) -> Self {
        let n_params = mu_phi.nrows();
        let n_samples = eta_samples.len();

        if n_samples == 0 {
            // No samples - use population prior
            return Self {
                mean: mu_phi.clone(),
                variance: (0..n_params).map(|j| omega[(j, j)].max(1e-6)).collect(),
            };
        }

        // Compute mean of eta samples
        let mut mean_eta = vec![0.0; n_params];
        for eta in eta_samples {
            for j in 0..n_params {
                mean_eta[j] += eta[j];
            }
        }
        for j in 0..n_params {
            mean_eta[j] /= n_samples as f64;
        }

        // Conditional mean: φ = μ + η
        let mean = Col::from_fn(n_params, |j| mu_phi[j] + mean_eta[j]);

        // Compute variance of eta samples
        let variance = if n_samples > 1 {
            let mut var = vec![0.0; n_params];
            for eta in eta_samples {
                for j in 0..n_params {
                    let diff = eta[j] - mean_eta[j];
                    var[j] += diff * diff;
                }
            }
            for j in 0..n_params {
                var[j] = (var[j] / (n_samples - 1) as f64).max(1e-6);
            }
            var
        } else {
            // Single sample - use population variance
            (0..n_params).map(|j| omega[(j, j)].max(1e-6)).collect()
        };

        Self { mean, variance }
    }

    /// Create directly from mean and variance estimates
    pub fn new(mean: Col<f64>, variance: Vec<f64>) -> Self {
        Self { mean, variance }
    }
}

/// Importance Sampling Likelihood Estimator
///
/// Computes the marginal log-likelihood using importance sampling with a
/// Student's t proposal distribution centered at the conditional posterior.
pub struct ImportanceSamplingEstimator<'a, E: Equation> {
    /// Configuration
    config: ImportanceSamplingConfig,
    /// Equation for likelihood computation
    equation: &'a E,
    /// Residual error models
    error_models: &'a ResidualErrorModels,
    /// Parameter transforms
    transforms: &'a [ParameterTransform],
    /// Population mean in φ space
    mu_phi: &'a Col<f64>,
    /// Population covariance matrix (kept for potential adaptive IS extensions)
    #[allow(dead_code)]
    omega: &'a Mat<f64>,
    /// Precomputed Ω⁻¹
    omega_inv: Mat<f64>,
    /// Precomputed log|Ω| + p×log(2π)
    log_prior_const: f64,
}

impl<'a, E: Equation> ImportanceSamplingEstimator<'a, E> {
    /// Create a new importance sampling estimator
    ///
    /// # Arguments
    ///
    /// * `config` - IS configuration
    /// * `equation` - Model equation
    /// * `error_models` - Residual error models
    /// * `transforms` - Parameter transforms (φ ↔ ψ)
    /// * `mu_phi` - Population mean in unconstrained space
    /// * `omega` - Population covariance matrix
    pub fn new(
        config: ImportanceSamplingConfig,
        equation: &'a E,
        error_models: &'a ResidualErrorModels,
        transforms: &'a [ParameterTransform],
        mu_phi: &'a Col<f64>,
        omega: &'a Mat<f64>,
    ) -> Result<Self> {
        // Use faer's Cholesky-based inversion directly
        let omega_inv = omega
            .llt(faer::Side::Lower)
            .map_err(|_| anyhow::anyhow!("Omega not positive definite"))?
            .inverse();
        let n_params = mu_phi.nrows();

        // log|Ω| + p×log(2π) for multivariate normal normalization
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

    /// Estimate log-likelihood for a single subject
    ///
    /// # Arguments
    ///
    /// * `subject` - Subject data
    /// * `conditional` - Conditional posterior from MCMC
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Estimated log p(y | θ) for this subject
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
            // Sample r ~ t(nu) for each parameter
            let r: Vec<f64> = (0..n_params).map(|_| t_dist.sample(rng)).collect();

            // φ = conditional_mean + sqrt(conditional_var) × r
            let phi_sample = Col::from_fn(n_params, |j| {
                conditional.mean[j] + conditional.variance[j].sqrt() * r[j]
            });

            // Transform to ψ space for model evaluation
            let psi_sample: Vec<f64> = (0..n_params)
                .map(|j| self.transforms[j].phi_to_psi(phi_sample[j]))
                .collect();

            // Compute log p(y | φ) - data likelihood
            let log_lik = match self.equation.estimate_predictions(subject, &psi_sample) {
                Ok(predictions) => {
                    let obs_pred_pairs =
                        predictions
                            .get_predictions()
                            .into_iter()
                            .filter_map(|pred| {
                                pred.observation()
                                    .map(|obs| (pred.outeq(), obs, pred.prediction()))
                            });
                    self.error_models.total_log_likelihood(obs_pred_pairs)
                }
                Err(_) => continue, // Skip failed predictions
            };

            // Compute log p(φ | Ω) - population prior (multivariate normal)
            let eta = Col::from_fn(n_params, |j| phi_sample[j] - self.mu_phi[j]);
            let mut quad_form = 0.0;
            for j in 0..n_params {
                for k in 0..n_params {
                    quad_form += eta[j] * self.omega_inv[(j, k)] * eta[k];
                }
            }
            let log_prior = -0.5 * (quad_form + self.log_prior_const);

            // Compute log q(φ) - proposal density (product of scaled t-distributions)
            let mut log_proposal = 0.0;
            for j in 0..n_params {
                // log pdf of t(nu) at r[j]
                let log_t_pdf = ln_gamma((nu + 1.0) / 2.0)
                    - ln_gamma(nu / 2.0)
                    - 0.5 * (nu * std::f64::consts::PI).ln()
                    - ((nu + 1.0) / 2.0) * (1.0 + r[j] * r[j] / nu).ln();
                log_proposal += log_t_pdf;
            }
            // Adjust for scale: subtract 0.5 × Σlog(cond_var)
            log_proposal -= 0.5 * conditional.variance.iter().map(|v| v.ln()).sum::<f64>();

            // Importance weight: log[p(y|φ) × p(φ|Ω) / q(φ)]
            let log_weight = log_lik + log_prior - log_proposal;
            log_weights.push(log_weight);
        }

        // Log-sum-exp for numerical stability
        if log_weights.is_empty() {
            return f64::NEG_INFINITY;
        }

        let max_weight = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_weight.is_finite() {
            return f64::NEG_INFINITY;
        }

        let sum_exp: f64 = log_weights.iter().map(|&w| (w - max_weight).exp()).sum();
        max_weight + sum_exp.ln() - (log_weights.len() as f64).ln()
    }

    /// Estimate total -2LL across all subjects
    ///
    /// # Arguments
    ///
    /// * `subjects` - Slice of subject references
    /// * `conditionals` - Conditional posteriors for each subject
    ///
    /// # Returns
    ///
    /// -2 × log-likelihood (for comparison with other methods)
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

        // Should use population mean and variance
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

        // Mean of eta should be 2.0, so conditional mean = mu + 2 = 2
        assert!((conditional.mean[0] - 2.0).abs() < 1e-6);

        // Variance should be sample variance of [1, 2, 3] = 1.0
        assert!((conditional.variance[0] - 1.0).abs() < 1e-6);
    }
}
