//! MCMC Kernels for f-SAEM Algorithm
//!
//! This module implements the four MCMC kernels used in the f-SAEM
//! (fast Stochastic Approximation Expectation-Maximization) algorithm.
//!
//! # Algorithm Background
//!
//! The SAEM algorithm uses MCMC sampling in the E-step to draw samples from
//! the conditional distribution p(ηᵢ | yᵢ, θ). The f-SAEM variant uses a
//! combination of four kernels to improve mixing:
//!
//! 1. **Kernel 1 (Prior)**: Full multivariate proposals from the prior N(0, Ω)
//! 2. **Kernel 2 (Component-wise)**: Single-component random walk updates
//! 3. **Kernel 3 (Block)**: Block random walk with adaptive grouping
//! 4. **Kernel 4 (MAP-based)**: Proposals centered at MAP estimate with
//!    covariance from Laplace approximation (the "fast" in f-SAEM)
//!
//! # References
//!
//! - Kuhn & Lavielle (2005). "Maximum likelihood estimation in nonlinear
//!   mixed effects models." Computational Statistics & Data Analysis.
//! - Comets et al. (2017). "Parameter estimation in nonlinear mixed effect
//!   models using saemix." Journal of Statistical Software.

use anyhow::Result;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Col, Mat};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

/// Configuration for the MCMC kernels in f-SAEM
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
            eta: Col::zeros(0), // Empty column
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

/// Adaptive random walk scales for Kernel 2 and 3
#[derive(Debug, Clone)]
pub struct AdaptiveScales {
    /// Per-parameter scales for component-wise updates
    component_scales: Col<f64>,
    /// Block scales for different block sizes
    block_scales: Vec<Col<f64>>,
    /// Acceptance counts for adaptation
    component_accepts: Vec<usize>,
    component_totals: Vec<usize>,
}

impl AdaptiveScales {
    /// Create new adaptive scales initialized from Ω
    pub fn new(omega: &Mat<f64>, rw_init: f64) -> Self {
        let n = omega.nrows();

        // Initialize component scales from sqrt(diag(Ω))
        let component_scales = Col::from_fn(n, |i| omega[(i, i)].sqrt() * rw_init);

        // Initialize block scales for different block sizes (1 to n)
        let block_scales = (1..=n)
            .map(|_size| {
                // Each column corresponds to a parameter
                Col::from_fn(n, |i| omega[(i, i)].sqrt() * rw_init)
            })
            .collect();

        Self {
            component_scales,
            block_scales,
            component_accepts: vec![0; n],
            component_totals: vec![0; n],
        }
    }

    /// Get scale for component j
    pub fn component_scale(&self, j: usize) -> f64 {
        self.component_scales[j]
    }

    /// Get scales for a block of parameters
    pub fn block_scale(&self, indices: &[usize], block_size: usize) -> Col<f64> {
        let block_idx = (block_size - 1).min(self.block_scales.len() - 1);
        Col::from_fn(indices.len(), |i| self.block_scales[block_idx][indices[i]])
    }

    /// Record acceptance for component j
    pub fn record_component(&mut self, j: usize, accepted: bool) {
        self.component_totals[j] += 1;
        if accepted {
            self.component_accepts[j] += 1;
        }
    }

    /// Adapt scales based on acceptance rates
    pub fn adapt(&mut self, step_size: f64, target_rate: f64) {
        let n = self.component_scales.nrows();
        for j in 0..n {
            if self.component_totals[j] > 0 {
                let rate = self.component_accepts[j] as f64 / self.component_totals[j] as f64;
                // Adapt scale: increase if acceptance too high, decrease if too low
                let factor = 1.0 + step_size * (rate - target_rate);
                self.component_scales[j] *= factor;

                // Update block scales too
                for block_scale in &mut self.block_scales {
                    block_scale[j] *= factor;
                }
            }
        }

        // Reset counters
        for j in 0..n {
            self.component_accepts[j] = 0;
            self.component_totals[j] = 0;
        }
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
    /// Create from MAP estimate and Hessian
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

/// MCMC sampler implementing the f-SAEM kernels
#[derive(Clone)]
pub struct FSaemKernels {
    /// Configuration
    config: KernelConfig,
    /// Number of parameters
    n_params: usize,
    /// Population covariance matrix Ω
    omega: Mat<f64>,
    /// Inverse of Ω
    omega_inv: Mat<f64>,
    /// Cholesky factor of Ω (for Kernel 1)
    chol_omega: Mat<f64>,
    /// Adaptive scales for random walk kernels
    adaptive_scales: AdaptiveScales,
    /// Current iteration
    iteration: usize,
}

impl FSaemKernels {
    /// Create a new f-SAEM kernel sampler
    pub fn new(omega: Mat<f64>, config: KernelConfig) -> Result<Self> {
        let n = omega.nrows();

        // Compute Cholesky of Ω and Ω⁻¹ using faer's built-in methods
        let llt = omega
            .llt(faer::Side::Lower)
            .map_err(|_| anyhow::anyhow!("Omega not positive definite"))?;
        let chol_omega = llt.L().to_owned();
        let omega_inv = llt.inverse();

        // Initialize adaptive scales
        let adaptive_scales = AdaptiveScales::new(&omega, config.rw_step_size);

        Ok(Self {
            config,
            n_params: n,
            omega,
            omega_inv,
            chol_omega,
            adaptive_scales,
            iteration: 0,
        })
    }

    /// Update the population covariance matrix
    pub fn update_omega(&mut self, omega: Mat<f64>) -> Result<()> {
        let llt = omega
            .llt(faer::Side::Lower)
            .map_err(|_| anyhow::anyhow!("Omega not positive definite"))?;
        self.chol_omega = llt.L().to_owned();
        self.omega_inv = llt.inverse();
        self.omega = omega;
        Ok(())
    }

    /// Set the current iteration (for deciding whether to use MAP kernel)
    pub fn set_iteration(&mut self, iter: usize) {
        self.iteration = iter;
    }

    /// Run all kernels for a single subject
    ///
    /// # Arguments
    ///
    /// * `state` - Current chain state
    /// * `mean_phi` - Population mean for this subject (including covariate effects)
    /// * `log_likelihood_fn` - Function computing log p(y | η + mean_phi)
    /// * `map_estimate` - Optional MAP estimate for Kernel 4
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Updated chain state
    pub fn run_kernels<F>(
        &mut self,
        mut state: ChainState,
        mean_phi: &Col<f64>,
        log_likelihood_fn: F,
        map_estimate: Option<&MapEstimate>,
        rng: &mut impl Rng,
    ) -> ChainState
    where
        F: Fn(&Col<f64>) -> f64,
    {
        let uniform = Uniform::new(0.0_f64, 1.0).unwrap();

        // Ensure initial log values are computed
        if state.log_likelihood.is_infinite() && state.log_likelihood < 0.0 {
            let phi = add_vectors(&state.eta, mean_phi);
            state.log_likelihood = log_likelihood_fn(&phi);
            state.log_prior = self.log_prior(&state.eta);
        }

        // Kernel 1: Full proposals from prior
        for _ in 0..self.config.n_kernel1 {
            let proposed_eta = self.sample_from_prior(rng);
            let phi = add_vectors(&proposed_eta, mean_phi);
            let proposed_ll = log_likelihood_fn(&phi);

            // For kernel 1, prior ratio is 1 (both from prior), only likelihood matters
            let log_alpha = proposed_ll - state.log_likelihood;

            if log_alpha.is_finite() && uniform.sample(rng).ln() < log_alpha {
                state.eta = proposed_eta;
                state.log_likelihood = proposed_ll;
                state.log_prior = self.log_prior(&state.eta);
            }
        }

        // Kernel 2: Component-wise random walk
        for _ in 0..self.config.n_kernel2 {
            for j in 0..self.n_params {
                let scale = self.adaptive_scales.component_scale(j);
                let normal = Normal::new(0.0, scale).unwrap();

                let mut proposed_eta = state.eta.clone();
                proposed_eta[j] += normal.sample(rng);

                let phi = add_vectors(&proposed_eta, mean_phi);
                let proposed_ll = log_likelihood_fn(&phi);
                let proposed_prior = self.log_prior(&proposed_eta);

                let log_alpha =
                    (proposed_ll + proposed_prior) - (state.log_likelihood + state.log_prior);

                let accepted = log_alpha.is_finite() && uniform.sample(rng).ln() < log_alpha;
                self.adaptive_scales.record_component(j, accepted);

                if accepted {
                    state.eta = proposed_eta;
                    state.log_likelihood = proposed_ll;
                    state.log_prior = proposed_prior;
                }
            }
        }

        // Kernel 3: Block random walk
        let block_size = (self.iteration % (self.n_params - 1)) + 2;
        let block_size = block_size.min(self.n_params);

        for _ in 0..self.config.n_kernel3 {
            // Select random block of parameters
            let indices: Vec<usize> = if block_size < self.n_params {
                let mut idx: Vec<usize> = (0..self.n_params).collect();
                // Fisher-Yates shuffle for first block_size elements
                for i in 0..block_size {
                    let j = i + (rng.random_range(0..(self.n_params - i)));
                    idx.swap(i, j);
                }
                idx[..block_size].to_vec()
            } else {
                (0..self.n_params).collect()
            };

            let scales = self.adaptive_scales.block_scale(&indices, block_size);
            let mut proposed_eta = state.eta.clone();

            for (k, &j) in indices.iter().enumerate() {
                let normal = Normal::new(0.0, scales[k]).unwrap();
                proposed_eta[j] += normal.sample(rng);
            }

            let phi = add_vectors(&proposed_eta, mean_phi);
            let proposed_ll = log_likelihood_fn(&phi);
            let proposed_prior = self.log_prior(&proposed_eta);

            let log_alpha =
                (proposed_ll + proposed_prior) - (state.log_likelihood + state.log_prior);

            if log_alpha.is_finite() && uniform.sample(rng).ln() < log_alpha {
                state.eta = proposed_eta;
                state.log_likelihood = proposed_ll;
                state.log_prior = proposed_prior;
            }
        }

        // Kernel 4: MAP-based proposals (only if we have a MAP estimate and within map_iterations)
        if self.config.n_kernel4 > 0
            && self.iteration < self.config.map_iterations
            && map_estimate.is_some()
        {
            let map_est = map_estimate.unwrap();

            for _ in 0..self.config.n_kernel4 {
                let proposed_eta = map_est.sample(rng);
                let phi = add_vectors(&proposed_eta, mean_phi);
                let proposed_ll = log_likelihood_fn(&phi);
                let proposed_prior = self.log_prior(&proposed_eta);

                // Compute proposal densities (asymmetric!)
                let log_q_proposed_given_current = map_est.log_density(&proposed_eta);
                let log_q_current_given_proposed = map_est.log_density(&state.eta);

                let log_alpha = (proposed_ll + proposed_prior + log_q_current_given_proposed)
                    - (state.log_likelihood + state.log_prior + log_q_proposed_given_current);

                if log_alpha.is_finite() && uniform.sample(rng).ln() < log_alpha {
                    state.eta = proposed_eta;
                    state.log_likelihood = proposed_ll;
                    state.log_prior = proposed_prior;
                }
            }
        }

        state
    }

    /// Adapt the random walk scales
    pub fn adapt_scales(&mut self) {
        self.adaptive_scales
            .adapt(self.config.rw_step_size, self.config.target_acceptance);
    }

    /// Sample from the prior N(0, Ω)
    fn sample_from_prior(&self, rng: &mut impl Rng) -> Col<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z: Vec<f64> = (0..self.n_params).map(|_| normal.sample(rng)).collect();

        // Transform: η = L * z where Ω = LLᵀ
        let mut eta = Col::zeros(self.n_params);
        for i in 0..self.n_params {
            for j in 0..=i {
                eta[i] += self.chol_omega[(i, j)] * z[j];
            }
        }
        eta
    }

    /// Compute log prior: log p(η | Ω) = -0.5 * ηᵀ Ω⁻¹ η
    fn log_prior(&self, eta: &Col<f64>) -> f64 {
        let n = eta.nrows();
        let mut quad = 0.0;
        for i in 0..n {
            for j in 0..n {
                quad += eta[i] * self.omega_inv[(i, j)] * eta[j];
            }
        }
        -0.5 * quad
    }

    /// Get number of parameters
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Get configuration
    pub fn config(&self) -> &KernelConfig {
        &self.config
    }

    /// Get adaptive scales for diagnostics
    pub fn adaptive_scales(&self) -> &AdaptiveScales {
        &self.adaptive_scales
    }
}

// ============== Helper Functions ==============

/// Add two column vectors
fn add_vectors(a: &Col<f64>, b: &Col<f64>) -> Col<f64> {
    let n = a.nrows();
    Col::from_fn(n, |i| a[i] + b[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cholesky_faer() {
        // Simple 2x2 positive definite matrix
        let mat = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.5 });

        // Use faer's built-in Cholesky
        let l = mat.llt(faer::Side::Lower).unwrap().L().to_owned();

        // Verify L * Lᵀ = mat
        for i in 0..2 {
            for j in 0..2 {
                let mut sum: f64 = 0.0;
                for k in 0..2 {
                    sum += l[(i, k)] * l[(j, k)];
                }
                assert!((sum - mat[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_inversion_faer() {
        let mat = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.5 });

        // Use faer's built-in inversion via Cholesky
        let inv = mat.llt(faer::Side::Lower).unwrap().inverse();

        // Verify mat * inv = I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum: f64 = 0.0;
                for k in 0..2 {
                    sum += mat[(i, k)] * inv[(k, j)];
                }
                let expected: f64 = if i == j { 1.0 } else { 0.0 };
                assert!((sum - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_chain_state() {
        let eta = Col::from_fn(3, |i| i as f64 * 0.1);
        let state = ChainState::new(eta);
        assert!(state.log_posterior() == f64::NEG_INFINITY);
    }
}
