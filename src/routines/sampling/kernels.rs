//! Core MCMC types for SAEM Algorithm
//!
//! This module provides the core types used for MCMC sampling in parametric algorithms.

use anyhow::Result;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Col, Mat};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Configuration for the MCMC kernels in SAEM
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Number of iterations for Kernel 1 (prior-based proposals)
    pub n_kernel1: usize,
    /// Number of iterations for Kernel 2 (component-wise random walk)
    pub n_kernel2: usize,
    /// Number of iterations for Kernel 3 (block random walk)
    pub n_kernel3: usize,
    /// Number of iterations for Kernel 4 (MAP-based proposals)
    pub n_kernel4: usize,
    /// Number of iterations to use MAP kernel (after this, revert to standard)
    pub map_iterations: usize,
    /// Step size for random walk adaptation
    pub rw_step_size: f64,
    /// Target acceptance probability for adaptive proposals
    pub target_acceptance: f64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            // Match R saemix defaults: nbiter.mcmc = c(2, 2, 2, 0)
            n_kernel1: 2,
            n_kernel2: 2,
            n_kernel3: 2,
            n_kernel4: 0,           // Disabled by default in R saemix
            map_iterations: 0,      // Not used when n_kernel4 = 0
            rw_step_size: 0.4,      // Match R saemix stepsize.rw
            target_acceptance: 0.4, // Match R saemix proba.mcmc
        }
    }
}

/// State of the MCMC chain for a single subject
#[derive(Debug, Clone)]
pub struct ChainState {
    /// Current random effects (η)
    pub eta: Col<f64>,
    /// Log-likelihood contribution: log p(y | η)
    pub log_likelihood: f64,
    /// Log-prior contribution: log p(η | Ω)
    pub log_prior: f64,
}

impl Default for ChainState {
    fn default() -> Self {
        Self {
            eta: Col::zeros(0),
            log_likelihood: f64::NEG_INFINITY,
            log_prior: f64::NEG_INFINITY,
        }
    }
}

impl ChainState {
    /// Create a new chain state
    pub fn new(eta: Col<f64>) -> Self {
        Self {
            eta,
            log_likelihood: f64::NEG_INFINITY,
            log_prior: f64::NEG_INFINITY,
        }
    }

    /// Get the log-posterior (unnormalized)
    pub fn log_posterior(&self) -> f64 {
        self.log_likelihood + self.log_prior
    }
}

/// MAP estimation result for a single subject
#[derive(Debug, Clone)]
pub struct MapEstimate {
    /// MAP estimate of η
    pub eta: Col<f64>,
    /// Approximate posterior covariance (inverse Hessian)
    pub covariance: Mat<f64>,
    /// Cholesky factor of covariance for sampling
    pub chol_cov: Mat<f64>,
    /// Inverse covariance for proposal density
    pub inv_cov: Mat<f64>,
}

impl MapEstimate {
    /// Create from MAP estimate and covariance
    pub fn new(eta: Col<f64>, covariance: Mat<f64>) -> Result<Self> {
        let n = eta.nrows();

        // Ensure positive definiteness by adding small diagonal if needed
        let mut cov = covariance.clone();
        let eps = 1e-6;
        for i in 0..n {
            if cov[(i, i)] < eps {
                cov[(i, i)] = eps;
            }
        }

        // Compute Cholesky factorization and inverse using faer's llt method
        let (chol_cov, inv_cov) = match cov.llt(faer::Side::Lower) {
            Ok(llt) => (llt.L().to_owned(), llt.inverse()),
            Err(_) => {
                // Fallback: use diagonal for both L and inverse
                let mut l = Mat::zeros(n, n);
                let mut inv = Mat::zeros(n, n);
                for i in 0..n {
                    let diag = cov[(i, i)].sqrt().max(eps);
                    l[(i, i)] = diag;
                    inv[(i, i)] = 1.0 / cov[(i, i)].max(eps);
                }
                (l, inv)
            }
        };

        Ok(Self {
            eta,
            covariance: cov,
            chol_cov,
            inv_cov,
        })
    }

    /// Sample from the MAP-based proposal distribution
    pub fn sample(&self, rng: &mut impl Rng) -> Col<f64> {
        let n = self.eta.nrows();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Generate standard normal vector
        let z: Vec<f64> = (0..n).map(|_| normal.sample(rng)).collect();

        // Transform: η_proposed = η_MAP + L * z
        let mut result = self.eta.clone();
        for i in 0..n {
            for j in 0..=i {
                result[i] += self.chol_cov[(i, j)] * z[j];
            }
        }
        result
    }

    /// Compute log proposal density: log q(η | η_MAP, Γ)
    pub fn log_density(&self, eta: &Col<f64>) -> f64 {
        let n = eta.nrows();
        let diff: Vec<f64> = (0..n).map(|i| eta[i] - self.eta[i]).collect();

        // -0.5 * (η - η_MAP)ᵀ Γ⁻¹ (η - η_MAP)
        let mut quad = 0.0;
        for i in 0..n {
            for j in 0..n {
                quad += diff[i] * self.inv_cov[(i, j)] * diff[j];
            }
        }

        -0.5 * quad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_state() {
        let eta = Col::from_fn(3, |i| i as f64 * 0.1);
        let state = ChainState::new(eta);
        assert!(state.log_posterior() == f64::NEG_INFINITY);
    }
}
