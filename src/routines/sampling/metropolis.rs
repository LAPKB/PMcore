//! Metropolis-Hastings MCMC sampler
//!
//! This module implements the Metropolis-Hastings algorithm for sampling from
//! the conditional distribution p(η | y, θ) in mixed-effects models.
//!
//! # Algorithm
//!
//! The Metropolis-Hastings algorithm generates samples from a target distribution
//! π(η) ∝ p(y | η) × p(η | θ) by:
//!
//! 1. Propose η' ~ q(η' | η)
//! 2. Compute acceptance probability: α = min(1, [π(η') q(η | η')] / [π(η) q(η' | η)])
//! 3. Accept η' with probability α, otherwise keep η
//!
//! # Usage
//!
//! ```ignore
//! use pmcore::routines::sampling::MetropolisHastings;
//!
//! let sampler = MetropolisHastings::new(proposal, target_log_density);
//! let samples = sampler.sample(initial, n_samples, &mut rng)?;
//! ```

use anyhow::Result;
use faer::Col;
use rand::Rng;
use rand_distr::{Distribution, Uniform};

use super::proposal::ProposalDistribution;

/// Metropolis-Hastings MCMC sampler
pub struct MetropolisHastings<P: ProposalDistribution> {
    /// Proposal distribution
    proposal: P,
    /// Number of burn-in samples to discard
    burn_in: usize,
    /// Thinning interval (keep every nth sample)
    thin: usize,
    /// Number of accepted proposals (for diagnostics)
    n_accepted: usize,
    /// Total number of proposals
    n_total: usize,
}

impl<P: ProposalDistribution> MetropolisHastings<P> {
    /// Create a new Metropolis-Hastings sampler
    pub fn new(proposal: P) -> Self {
        Self {
            proposal,
            burn_in: 0,
            thin: 1,
            n_accepted: 0,
            n_total: 0,
        }
    }

    /// Set the burn-in period
    pub fn with_burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Set the thinning interval
    pub fn with_thinning(mut self, thin: usize) -> Self {
        self.thin = thin.max(1);
        self
    }

    /// Get the acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.n_total == 0 {
            0.0
        } else {
            self.n_accepted as f64 / self.n_total as f64
        }
    }

    /// Reset acceptance statistics
    pub fn reset_statistics(&mut self) {
        self.n_accepted = 0;
        self.n_total = 0;
    }

    /// Get a reference to the proposal distribution
    pub fn proposal(&self) -> &P {
        &self.proposal
    }

    /// Get a mutable reference to the proposal distribution
    pub fn proposal_mut(&mut self) -> &mut P {
        &mut self.proposal
    }

    /// Run the sampler for a single subject
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial state of the chain
    /// * `n_samples` - Number of samples to return (after burn-in and thinning)
    /// * `log_target` - Function computing log π(η) (unnormalized log posterior)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Vector of samples from the target distribution
    pub fn sample<F>(
        &mut self,
        initial: Col<f64>,
        n_samples: usize,
        log_target: F,
        rng: &mut impl Rng,
    ) -> Result<Vec<Col<f64>>>
    where
        F: Fn(&Col<f64>) -> f64,
    {
        let uniform = Uniform::new(0.0_f64, 1.0).expect("Failed to create uniform distribution");

        let mut current = initial;
        let mut current_log_prob = log_target(&current);
        let mut samples = Vec::with_capacity(n_samples);

        // Total iterations needed
        let total_iterations = self.burn_in + n_samples * self.thin;

        for iter in 0..total_iterations {
            // Propose new state
            let proposed = self.proposal.propose(&current, rng);
            let proposed_log_prob = log_target(&proposed);

            // Compute log acceptance probability
            let log_alpha = proposed_log_prob - current_log_prob;

            // Add proposal ratio if not symmetric
            let log_alpha = if self.proposal.is_symmetric() {
                log_alpha
            } else {
                log_alpha + self.proposal.log_density(&current, &proposed)
                    - self.proposal.log_density(&proposed, &current)
            };

            // Accept or reject
            self.n_total += 1;
            let u: f64 = uniform.sample(rng);
            if log_alpha.is_finite() && u.ln() < log_alpha {
                current = proposed;
                current_log_prob = proposed_log_prob;
                self.n_accepted += 1;
            }

            // Store sample if past burn-in and on thinning interval
            if iter >= self.burn_in && (iter - self.burn_in) % self.thin == 0 {
                samples.push(current.clone());
            }
        }

        Ok(samples)
    }

    /// Run the sampler and return only the final sample
    ///
    /// Useful for SAEM where only one sample per iteration is needed.
    pub fn sample_one<F>(
        &mut self,
        initial: Col<f64>,
        _n_iterations: usize,
        log_target: F,
        rng: &mut impl Rng,
    ) -> Result<Col<f64>>
    where
        F: Fn(&Col<f64>) -> f64,
    {
        let samples = self.sample(initial, 1, log_target, rng)?;
        Ok(samples.into_iter().next().unwrap_or_else(|| Col::zeros(0)))
    }

    /// Adapt the proposal distribution based on collected samples
    pub fn adapt(&mut self, samples: &[Col<f64>]) {
        let acceptance_rate = self.acceptance_rate();
        self.proposal.adapt(samples, acceptance_rate);
        self.reset_statistics();
    }
}

/// Builder for creating log-target functions for mixed-effects models
///
/// The target distribution is:
/// π(η) ∝ p(y | η, θ) × p(η | Ω)
///
/// where p(η | Ω) = N(0, Ω) is the prior on random effects.
pub struct LogTargetBuilder {
    /// Population covariance matrix inverse (Ω⁻¹)
    omega_inv: Mat<f64>,
    /// Log determinant of Ω (for normalization, optional)
    log_det_omega: f64,
}

use faer::Mat;

impl LogTargetBuilder {
    /// Create a new log-target builder from population covariance
    pub fn new(omega: &Mat<f64>) -> Result<Self> {
        // Compute Ω⁻¹ using Cholesky decomposition
        let n = omega.nrows();

        // For now, use simple matrix inversion (should use Cholesky for numerical stability)
        // TODO: Use proper Cholesky decomposition from faer

        // Simple diagonal approximation for now
        let omega_inv = Mat::from_fn(n, n, |i, j| {
            if i == j && omega[(i, i)] > 0.0 {
                1.0 / omega[(i, i)]
            } else {
                0.0
            }
        });

        // Compute log determinant (sum of log of diagonal for diagonal matrix)
        let log_det = (0..n).map(|i| omega[(i, i)].ln()).sum();

        Ok(Self {
            omega_inv,
            log_det_omega: log_det,
        })
    }

    /// Build the log-target function
    ///
    /// Returns a closure that computes:
    /// log π(η) = log_likelihood(η) - 0.5 × η' Ω⁻¹ η - 0.5 × log|Ω| + const
    pub fn build<F>(&self, log_likelihood: F) -> impl Fn(&Col<f64>) -> f64
    where
        F: Fn(&Col<f64>) -> f64,
    {
        let omega_inv = self.omega_inv.clone();
        let log_det = self.log_det_omega;

        move |eta: &Col<f64>| {
            // Log-likelihood contribution
            let ll = log_likelihood(eta);

            // Prior contribution: -0.5 × η' Ω⁻¹ η
            let n = eta.nrows();
            let mut quadratic_form = 0.0;
            for i in 0..n {
                for j in 0..n {
                    quadratic_form += eta[i] * omega_inv[(i, j)] * eta[j];
                }
            }
            let log_prior = -0.5 * quadratic_form - 0.5 * log_det;

            ll + log_prior
        }
    }
}

/// MCMC diagnostics and convergence assessment
#[derive(Debug, Clone)]
pub struct MCMCDiagnostics {
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// Effective sample size (estimated)
    pub effective_sample_size: f64,
    /// Sample mean
    pub mean: Col<f64>,
    /// Sample standard deviation
    pub std: Col<f64>,
}

impl MCMCDiagnostics {
    /// Compute diagnostics from a set of samples
    pub fn from_samples(samples: &[Col<f64>], acceptance_rate: f64) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let n_samples = samples.len() as f64;
        let n_params = samples[0].nrows();

        // Compute mean
        let mean = Col::from_fn(n_params, |j| {
            samples.iter().map(|s| s[j]).sum::<f64>() / n_samples
        });

        // Compute standard deviation
        let std = Col::from_fn(n_params, |j| {
            let variance = samples.iter().map(|s| (s[j] - mean[j]).powi(2)).sum::<f64>()
                / (n_samples - 1.0);
            variance.sqrt()
        });

        // Estimate effective sample size (simplified - assumes independent samples)
        // A proper implementation would use autocorrelation
        let effective_sample_size = n_samples * acceptance_rate;

        Some(Self {
            acceptance_rate,
            effective_sample_size,
            mean,
            std,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routines::sampling::GaussianProposal;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_metropolis_hastings_sampling() {
        // Sample from a standard normal distribution
        let proposal = GaussianProposal::uniform(1, 1.0);
        let mut sampler = MetropolisHastings::new(proposal).with_burn_in(100);

        let initial = Col::from_fn(1, |_| 0.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Target: N(0, 1)
        let log_target = |x: &Col<f64>| -0.5 * x[0] * x[0];

        let samples = sampler.sample(initial, 1000, log_target, &mut rng).unwrap();

        assert_eq!(samples.len(), 1000);

        // Check mean is close to 0
        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.2, "Mean should be close to 0, got {}", mean);

        // Check acceptance rate is reasonable
        let acc_rate = sampler.acceptance_rate();
        assert!(
            acc_rate > 0.1 && acc_rate < 0.9,
            "Acceptance rate {} should be reasonable",
            acc_rate
        );
    }

    #[test]
    fn test_log_target_builder() {
        // Simple 2D case with diagonal covariance
        let omega = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let builder = LogTargetBuilder::new(&omega).unwrap();

        // Constant log-likelihood
        let log_likelihood = |_eta: &Col<f64>| 0.0;
        let log_target = builder.build(log_likelihood);

        // At η = 0, prior should be at maximum
        let eta_zero = Col::from_fn(2, |_| 0.0);
        let eta_one = Col::from_fn(2, |_| 1.0);

        let lp_zero = log_target(&eta_zero);
        let lp_one = log_target(&eta_one);

        // log p(0) should be greater than log p(1) for N(0,I)
        assert!(lp_zero > lp_one);
    }
}
