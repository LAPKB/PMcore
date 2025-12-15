//! Proposal distributions for MCMC sampling
//!
//! This module defines proposal distributions used in Metropolis-Hastings
//! and other MCMC algorithms.

use faer::{Col, Mat};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Trait for MCMC proposal distributions
///
/// A proposal distribution q(θ' | θ) suggests new parameter values
/// given the current state.
pub trait ProposalDistribution: Send + Sync {
    /// Generate a proposal given the current state
    fn propose(&self, current: &Col<f64>, rng: &mut impl Rng) -> Col<f64>;

    /// Compute log q(proposed | current)
    ///
    /// For symmetric proposals (like Gaussian random walk), this equals
    /// log q(current | proposed), and the ratio cancels in M-H acceptance.
    fn log_density(&self, proposed: &Col<f64>, current: &Col<f64>) -> f64;

    /// Check if the proposal is symmetric
    ///
    /// If true, the proposal ratio q(θ|θ')/q(θ'|θ) = 1 and can be ignored
    /// in the Metropolis-Hastings acceptance probability.
    fn is_symmetric(&self) -> bool;

    /// Update proposal parameters (for adaptive MCMC)
    fn adapt(&mut self, _samples: &[Col<f64>], _acceptance_rate: f64) {
        // Default: no adaptation
    }
}

/// Gaussian random-walk proposal distribution
///
/// Proposes θ' = θ + ε where ε ~ N(0, Σ)
#[derive(Debug, Clone)]
pub struct GaussianProposal {
    /// Proposal standard deviations for each parameter
    scale: Col<f64>,
    /// Covariance matrix (diagonal by default)
    covariance: Option<Mat<f64>>,
}

impl GaussianProposal {
    /// Create a new Gaussian proposal with given scales
    pub fn new(scale: Col<f64>) -> Self {
        Self {
            scale,
            covariance: None,
        }
    }

    /// Create from a covariance matrix
    pub fn from_covariance(covariance: Mat<f64>) -> Self {
        let n = covariance.nrows();
        let scale = Col::from_fn(n, |i| covariance[(i, i)].sqrt());
        Self {
            scale,
            covariance: Some(covariance),
        }
    }

    /// Create with uniform scale for all parameters
    pub fn uniform(n_params: usize, scale: f64) -> Self {
        Self {
            scale: Col::from_fn(n_params, |_| scale),
            covariance: None,
        }
    }

    /// Get the proposal scales
    pub fn scale(&self) -> &Col<f64> {
        &self.scale
    }

    /// Set the proposal scales
    pub fn set_scale(&mut self, scale: Col<f64>) {
        self.scale = scale;
        self.covariance = None; // Reset to diagonal
    }
}

impl ProposalDistribution for GaussianProposal {
    fn propose(&self, current: &Col<f64>, rng: &mut impl Rng) -> Col<f64> {
        let n = current.nrows();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // For now, use diagonal covariance (independent proposals)
        Col::from_fn(n, |i| {
            let eps: f64 = normal.sample(rng);
            current[i] + eps * self.scale[i]
        })
    }

    fn log_density(&self, _proposed: &Col<f64>, _current: &Col<f64>) -> f64 {
        // For symmetric Gaussian proposal, we don't need the actual density
        // since it cancels in the M-H ratio
        0.0
    }

    fn is_symmetric(&self) -> bool {
        true // Gaussian random walk is symmetric
    }
}

/// Adaptive Gaussian proposal distribution
///
/// Adapts the proposal covariance based on the history of accepted samples
/// using the Haario et al. (2001) adaptive MCMC algorithm.
#[derive(Debug, Clone)]
pub struct AdaptiveProposal {
    /// Base Gaussian proposal
    base: GaussianProposal,
    /// Target acceptance rate
    target_acceptance: f64,
    /// Adaptation scale factor
    adaptation_rate: f64,
    /// Minimum number of samples before adapting
    min_samples: usize,
    /// Current adaptation iteration
    iteration: usize,
    /// Running mean of samples
    sample_mean: Option<Col<f64>>,
    /// Running covariance of samples
    sample_covariance: Option<Mat<f64>>,
}

impl AdaptiveProposal {
    /// Create a new adaptive proposal
    pub fn new(initial_scale: Col<f64>) -> Self {
        Self {
            base: GaussianProposal::new(initial_scale),
            target_acceptance: 0.234, // Optimal for high dimensions
            adaptation_rate: 0.1,
            min_samples: 100,
            iteration: 0,
            sample_mean: None,
            sample_covariance: None,
        }
    }

    /// Set the target acceptance rate
    pub fn with_target_acceptance(mut self, rate: f64) -> Self {
        self.target_acceptance = rate.clamp(0.1, 0.5);
        self
    }

    /// Update running statistics with a new sample
    fn update_statistics(&mut self, sample: &Col<f64>) {
        self.iteration += 1;
        let n = sample.nrows();

        match (&mut self.sample_mean, &mut self.sample_covariance) {
            (Some(mean), Some(cov)) => {
                // Welford's online algorithm for mean and covariance
                let k = self.iteration as f64;
                let delta = Col::from_fn(n, |i| sample[i] - mean[i]);

                // Update mean
                for i in 0..n {
                    mean[i] += delta[i] / k;
                }

                // Update covariance (simplified diagonal version)
                for i in 0..n {
                    let delta2 = sample[i] - mean[i];
                    cov[(i, i)] += delta[i] * delta2 - cov[(i, i)] / k;
                }
            }
            _ => {
                // Initialize
                self.sample_mean = Some(sample.clone());
                self.sample_covariance = Some(Mat::zeros(n, n));
            }
        }
    }
}

impl ProposalDistribution for AdaptiveProposal {
    fn propose(&self, current: &Col<f64>, rng: &mut impl Rng) -> Col<f64> {
        self.base.propose(current, rng)
    }

    fn log_density(&self, proposed: &Col<f64>, current: &Col<f64>) -> f64 {
        self.base.log_density(proposed, current)
    }

    fn is_symmetric(&self) -> bool {
        true
    }

    fn adapt(&mut self, samples: &[Col<f64>], acceptance_rate: f64) {
        // Update statistics with new samples
        for sample in samples {
            self.update_statistics(sample);
        }

        // Only adapt after enough samples
        if self.iteration < self.min_samples {
            return;
        }

        // Adjust scale based on acceptance rate
        let adjustment = if acceptance_rate < self.target_acceptance {
            // Too many rejections: decrease scale
            1.0 - self.adaptation_rate
        } else {
            // Too many acceptances: increase scale
            1.0 + self.adaptation_rate
        };

        // Apply adjustment
        let n = self.base.scale.nrows();
        let new_scale = Col::from_fn(n, |i| self.base.scale[i] * adjustment);
        self.base.set_scale(new_scale);

        // Optionally use empirical covariance (Roberts & Rosenthal, 2009)
        if let Some(ref cov) = self.sample_covariance {
            // Scale empirical covariance by 2.38²/d (optimal for Gaussian targets)
            let d = n as f64;
            let optimal_scale = 2.38_f64.powi(2) / d;

            let scaled_cov = Mat::from_fn(n, n, |i, j| cov[(i, j)] * optimal_scale);
            self.base = GaussianProposal::from_covariance(scaled_cov);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_gaussian_proposal() {
        let scale = Col::from_fn(2, |_| 0.1);
        let proposal = GaussianProposal::new(scale);

        let current = Col::from_fn(2, |i| i as f64);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let proposed = proposal.propose(&current, &mut rng);

        // Proposed should be close to current (within a few standard deviations)
        for i in 0..2 {
            assert!((proposed[i] - current[i]).abs() < 1.0);
        }

        assert!(proposal.is_symmetric());
    }

    #[test]
    fn test_adaptive_proposal() {
        let scale = Col::from_fn(2, |_| 0.1);
        let mut proposal = AdaptiveProposal::new(scale);

        // Simulate some samples
        let samples: Vec<Col<f64>> = (0..200)
            .map(|i| Col::from_fn(2, |j| (i + j) as f64 * 0.01))
            .collect();

        // Adapt with high acceptance rate - should increase scale
        let initial_scale = proposal.base.scale[0];
        proposal.adapt(&samples, 0.5);
        assert!(proposal.base.scale[0] > initial_scale);
    }
}
