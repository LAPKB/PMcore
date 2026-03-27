//! f-SAEM (fast Stochastic Approximation Expectation-Maximization) Algorithm
//!
//! This module implements the f-SAEM algorithm for maximum likelihood estimation
//! in nonlinear mixed-effects models.
//!
//! # Algorithm Overview
//!
//! f-SAEM is an enhanced version of SAEM that uses four MCMC kernels for improved
//! mixing and faster convergence:
//!
//! 1. **Kernel 1 (Prior)**: Full multivariate proposals from N(0, Ω)
//! 2. **Kernel 2 (Component-wise)**: Single-component adaptive random walk
//! 3. **Kernel 3 (Block)**: Block random walk with varying block sizes
//! 4. **Kernel 4 (MAP-based)**: Proposals centered at MAP with Laplace covariance
//!
//! # Algorithm Phases
//!
//! ## Phase 1: Burn-in (K₁ iterations)
//! - Step size γₖ = 1 (full updates)
//! - Simulated annealing on variance (floor shrinking)
//! - All four MCMC kernels active
//!
//! ## Phase 2: Estimation (K₂ iterations)  
//! - Decreasing step size γₖ = 1/(k - K₁)
//! - Sufficient statistics converge to true expectations
//! - MAP kernel may be disabled for efficiency
//!
//! # Mathematical Background
//!
//! SAEM replaces the intractable E-step with stochastic approximation:
//!
//! ```text
//! E-step: Draw φ⁽ᵏ⁾ ~ p(φ | y, θ⁽ᵏ⁻¹⁾) using MCMC
//! SA-step: sₖ = sₖ₋₁ + γₖ(S(φ⁽ᵏ⁾) - sₖ₋₁)
//! M-step: θ⁽ᵏ⁾ = argmax_θ Q(θ, sₖ)
//! ```
//!
//! For normal random effects, sufficient statistics are:
//! - S₁ = Σᵢ φᵢ (sum of parameters)
//! - S₂ = Σᵢ φᵢφᵢᵀ (sum of outer products)
//!
//! And the M-step has closed-form solutions:
//! - μ = S₁ / n
//! - Ω = S₂ / n - μμᵀ
//!
//! # References
//!
//! - Kuhn & Lavielle (2005). "Maximum likelihood estimation in nonlinear
//!   mixed effects models." Computational Statistics & Data Analysis.
//! - Comets et al. (2017). "Parameter estimation in nonlinear mixed effect
//!   models using saemix." Journal of Statistical Software.
//! - Lavielle, M. (2015). "Mixed Effects Models for the Population Approach."
//!   Chapman & Hall/CRC.

use anyhow::Result;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Col, Mat};
use ndarray::Array2;
use pharmsol::{Data, Equation, Event, ResidualErrorModels};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

use crate::algorithms::{Status, StopReason};
use crate::api::SaemConfig;
use crate::estimation::parametric::{
    approximate_objective_from_individuals, assemble_saem_result, batch_log_likelihood_from_eta,
    blended_subject_covariate_m_step, covariate_state, ensure_positive_definite_covariance,
    estimate_initial_sigma_sq, initialize_population_in_phi_space, log_priors_from_eta_matrix,
    occasion_covariate_maps, phi_to_psi, recenter_individual_estimates, saem_posthoc_likelihood,
    sample_eta_from_population, subject_covariate_maps, subject_log_prior_from_eta,
    subject_mean_phi, transform_label, update_residual_error_from_individuals, ChainState,
    Individual, IndividualEstimates, KernelConfig, ParameterTransform, PhiVector, Population,
    SaemResultInput, SufficientStats,
};
use crate::model::{CovariateModel, CovariateSpec};
use crate::output::shared::RunConfiguration;

use super::algorithm::{ParametricAlgorithm, ParametricAlgorithmInput, ParametricConfig};

/// f-SAEM algorithm configuration
///
/// This structure can be constructed from [`SaemConfig`] to ensure
/// consistency with user-facing configuration.
#[derive(Debug, Clone)]
pub struct FSaemConfig {
    /// Base parametric algorithm configuration
    pub base: ParametricConfig,
    /// MCMC kernel configuration
    pub kernel_config: KernelConfig,
    /// Number of MCMC chains per subject
    pub n_chains: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Number of pure burn-in iterations with γ=0 (nbiter.burn in R saemix)
    /// During this phase, MCMC runs but statistics are not accumulated
    /// R saemix default: 5
    pub n_pure_burn: usize,
    /// Number of SA iterations with γ=1 (exploration phase after burn-in)
    /// R saemix: this is K₁ - nbiter.burn, where K₁ = nbiter.saemix[1]
    /// R saemix default: 300 - 5 = 295
    pub n_sa: usize,
    /// Total number of iterations (K₁ + K₂)
    /// R saemix default: 300 + 100 = 400
    pub n_iterations: usize,
    /// Simulated annealing decay rate for variance floor (alpha.sa in R)
    /// R saemix default: 0.97
    pub sa_alpha: f64,
    /// Minimum variance for simulated annealing
    pub sa_min_var: f64,
    /// Number of SA iterations for variance floor decay (nbiter.sa in R)
    /// R saemix default: K₁/2 = 150
    pub n_sa_variance: usize,
}

impl FSaemConfig {
    /// Get total burn-in iterations (pure burn + SA phase)
    /// This is equivalent to nbiter.saemix[1] in R
    pub fn n_burn_in(&self) -> usize {
        self.n_pure_burn + self.n_sa
    }
}

impl Default for FSaemConfig {
    /// Default configuration matching R saemix defaults exactly
    ///
    /// R saemix defaults:
    /// - nbiter.saemix = c(300, 100) → K₁=300, K₂=100, total=400
    /// - nbiter.burn = 5
    /// - nbiter.sa = K₁/2 = 150 (for variance floor decay)
    /// - alpha.sa = 0.97
    /// - nb.chains = 1
    /// - nbiter.mcmc = c(2, 2, 2, 0)
    fn default() -> Self {
        Self {
            base: ParametricConfig::default(),
            kernel_config: KernelConfig::default(),
            n_chains: 1,
            seed: 123456, // Match R saemix default seed
            // R saemix: nbiter.burn = 5 (pure burn-in with γ=0)
            n_pure_burn: 5,
            // R saemix: K₁ = 300 (exploration phase with γ=1)
            // n_sa = K₁ - n_pure_burn = 300 - 5 = 295
            n_sa: 295,
            // R saemix: K₁ + K₂ = 300 + 100 = 400
            n_iterations: 400,
            sa_alpha: 0.97,
            sa_min_var: 1e-6,
            // R saemix: nbiter.sa = K₁/2 = 150 (variance floor decay iterations)
            n_sa_variance: 150,
        }
    }
}

impl FSaemConfig {
    /// Create internal configuration from the user-facing SAEM config.
    pub fn from_saem_config(config: &SaemConfig) -> Self {
        let k1 = config.k1_iterations;
        let k2 = config.k2_iterations;

        Self {
            base: ParametricConfig {
                max_iterations: k1 + k2,
                burn_in: config.burn_in,
                n_chains: config.n_chains,
                n_samples: config.mcmc_iterations,
                parameter_tolerance: 1e-4,
                objective_tolerance: 1e-4,
                use_annealing: config.sa_iterations > 0,
                initial_temperature: 1.0,
            },
            kernel_config: KernelConfig {
                n_kernel1: 2,
                n_kernel2: 2,
                n_kernel3: 2,
                n_kernel4: if config.n_kernels >= 4 { 0 } else { 0 }, // Kernel 4 disabled by default
                map_iterations: 0,
                rw_step_size: config.mcmc_step_size,
                target_acceptance: 0.4,
                rw_init: 0.5,
            },
            n_chains: config.n_chains,
            seed: config.seed,
            n_pure_burn: config.burn_in,
            n_sa: k1.saturating_sub(config.burn_in),
            n_iterations: k1 + k2,
            sa_alpha: config.sa_cooling_factor,
            sa_min_var: config.omega_min_variance,
            // R saemix: nbiter.sa = K₁/2 for variance floor decay
            n_sa_variance: if config.sa_iterations > 0 {
                config.sa_iterations
            } else {
                k1 / 2
            },
        }
    }

    /// Create configuration with auto-scaling for small datasets
    ///
    /// R saemix automatically increases chains when N < 50:
    /// nb.chains = ceiling(50/N)
    pub fn from_saem_config_with_auto_chains(saem_config: &SaemConfig, n_subjects: usize) -> Self {
        let mut config = Self::from_saem_config(saem_config);

        // R saemix: if N < 50, auto-increase chains
        if n_subjects < 50 && saem_config.n_chains == 1 {
            config.n_chains = ((50.0 / n_subjects as f64).ceil() as usize).max(1);
            tracing::info!(
                "Auto-scaled MCMC chains from 1 to {} for small dataset (N={})",
                config.n_chains,
                n_subjects
            );
        }

        config
    }
}

/// f-SAEM algorithm state
pub struct FSAEM<E: Equation> {
    /// Pharmacokinetic/pharmacodynamic model
    equation: E,
    /// Population data
    data: Data,
    /// Run configuration derived from the unified API surface
    run_configuration: RunConfiguration,
    /// Current population parameters (μ, Ω)
    population: Population,
    /// Individual parameter estimates from last iteration
    individual_estimates: IndividualEstimates,
    /// Accumulated sufficient statistics
    sufficient_stats: SufficientStats,
    /// Current iteration number
    iteration: usize,
    /// Current objective function value (-2LL approximation)
    objf: f64,
    /// Previous objective function value (for convergence)
    prev_objf: f64,
    /// Algorithm status
    status: Status,
    /// f-SAEM specific configuration
    config: FSaemConfig,
    /// Chain states for each subject and chain
    chain_states: Vec<Vec<ChainState>>,
    /// Current residual error variance (σ² estimated in M-step)
    sigma_sq: f64,
    /// Sufficient statistic for residual error (statrese in R saemix)
    /// This is Σ(weighted residuals²) and gets updated via stochastic approximation
    statrese: f64,
    /// Residual error models for prediction-based sigma calculation
    /// Used for both M-step residual weighting and likelihood computation
    /// Uses pharmsol's ResidualErrorModels which computes sigma from prediction
    residual_error_models: ResidualErrorModels,
    /// Variance floor for simulated annealing
    variance_floor: Col<f64>,
    /// Random number generator
    rng: ChaCha8Rng,
    /// Parameter history for convergence diagnostics
    param_history: Vec<Vec<f64>>,
    /// Objective function history (per-iteration approximate -2LL)
    objf_history: Vec<f64>,
    /// Parameter transforms (φ ↔ ψ conversions)
    /// Maps between unconstrained space (φ) and constrained space (ψ)
    transforms: Vec<ParameterTransform>,
    /// Adaptive proposal scales for Kernel 2 & 3 random walks (domega2 in R saemix)
    /// Initialized as sqrt(diag(Ω)) * rw_init and adapted based on acceptance rates
    /// This decouples proposal width from current Ω, preventing collapse
    domega2: Col<f64>,
    /// Structured subject-level covariate model used to compute subject-specific mean φ.
    subject_covariate_model: Option<CovariateModel>,
    /// Structured subject-level covariate values keyed by covariate name.
    subject_covariates: Vec<HashMap<String, f64>>,
    /// Structured occasion-level covariate model preserved in fitted state for IOV-ready paths.
    occasion_covariate_model: Option<CovariateModel>,
    /// Structured occasion-level covariate values keyed by covariate name.
    occasion_covariates: Vec<HashMap<String, f64>>,
}

impl<E: Equation + Send + 'static> FSAEM<E> {
    /// Create a new f-SAEM algorithm instance from the unified SAEM config.
    ///
    /// This constructor reads configuration from [`SaemConfig`] to ensure
    /// the algorithm behaves according to user-specified parameters.
    /// For small datasets (N < 50), MCMC chains are automatically scaled
    /// following R saemix behavior.
    pub(crate) fn create(input: ParametricAlgorithmInput<E>) -> Result<Box<Self>> {
        let n_subjects = input.data.subjects().len();
        let config =
            FSaemConfig::from_saem_config_with_auto_chains(input.saem_config(), n_subjects);
        Self::create_with_config(input, config)
    }

    /// Create with custom configuration (advanced users)
    ///
    /// Use this when you need fine-grained control over algorithm parameters
    /// beyond what [`SaemConfig`] provides.
    pub(crate) fn create_with_config(
        input: ParametricAlgorithmInput<E>,
        config: FSaemConfig,
    ) -> Result<Box<Self>> {
        let run_configuration = input.run_configuration();
        // Initialize population from the unified parameter space.
        let population = input.initial_population()?;
        let transforms = input.parameter_transforms();
        let ParametricAlgorithmInput {
            equation,
            data,
            covariates,
            structured_covariates,
            residual_error_models,
            ..
        } = input;
        let (
            subject_covariate_model,
            occasion_covariate_model,
            subject_covariates,
            occasion_covariates,
        ) = match covariates {
            CovariateSpec::InEquation => (None, None, Vec::new(), Vec::new()),
            CovariateSpec::Structured(spec) => (
                spec.subject_effects,
                spec.occasion_effects,
                subject_covariate_maps(&structured_covariates),
                occasion_covariate_maps(&structured_covariates),
            ),
        };
        let n_params = population.npar();
        let n_subjects = data.subjects().len();

        // Initialize sufficient statistics
        let sufficient_stats = SufficientStats::new(n_params);

        // Initialize residual error variance
        let sigma_sq = estimate_initial_sigma_sq(&residual_error_models);
        // Initialize statrese - sufficient statistic for residual error
        // Start with initial sigma² * n_obs (will be divided back to get sigma)
        let n_obs_estimate = data
            .subjects()
            .iter()
            .map(|s| {
                s.occasions()
                    .iter()
                    .flat_map(|o| o.events())
                    .filter(|e| matches!(e, Event::Observation(_)))
                    .count()
            })
            .sum::<usize>();
        let statrese = sigma_sq * n_obs_estimate as f64;

        // Initialize chain states (one set of chains per subject)
        let chain_states: Vec<Vec<ChainState>> = (0..n_subjects)
            .map(|_| {
                (0..config.n_chains)
                    .map(|_| ChainState::new(Col::zeros(n_params)))
                    .collect()
            })
            .collect();

        // Initialize variance floor for simulated annealing
        let variance_floor = Col::from_fn(n_params, |i| population.omega()[(i, i)]);

        // Initialize adaptive proposal scales (domega2 in R saemix)
        // R saemix: domega2 = sqrt(diag(omega.eta)) * rw.init
        let domega2 = Col::from_fn(n_params, |i| {
            population.omega()[(i, i)].sqrt() * config.kernel_config.rw_init
        });

        // Random number generator
        let rng = ChaCha8Rng::seed_from_u64(config.seed);

        Ok(Box::new(Self {
            equation,
            data,
            run_configuration,
            population,
            individual_estimates: IndividualEstimates::new(),
            sufficient_stats,
            iteration: 0,
            objf: f64::INFINITY,
            prev_objf: f64::INFINITY,
            status: Status::Continue,
            config,
            chain_states,
            sigma_sq,
            statrese,
            residual_error_models,
            variance_floor,
            rng,
            param_history: Vec::new(),
            objf_history: Vec::new(),
            transforms,
            domega2,
            subject_covariate_model,
            subject_covariates,
            occasion_covariate_model,
            occasion_covariates,
        }))
    }

    /// Check if currently in burn-in phase (Phase 1 + Phase 2)
    /// - Phase 1: Pure burn-in (γ=0)
    /// - Phase 2: SA phase (γ=1)
    /// Check if currently in pure burn-in phase (γ=0, no stat updates)
    pub fn is_pure_burn_in(&self) -> bool {
        self.iteration <= self.config.n_pure_burn
    }

    pub fn is_burn_in(&self) -> bool {
        self.iteration <= self.config.n_burn_in()
    }

    /// Check if currently in SA phase (γ=1 with simulated annealing)
    pub fn is_sa_phase(&self) -> bool {
        self.iteration > self.config.n_pure_burn && self.iteration <= self.config.n_burn_in()
    }

    /// Check if variance floor should be applied (R saemix: kiter <= nbiter.sa)
    ///
    /// In R saemix, `nbiter.sa` controls how long the variance floor decays.
    /// This is separate from the step size schedule.
    pub fn is_variance_floor_active(&self) -> bool {
        self.iteration <= self.config.n_sa_variance
    }

    /// Get the current step size γₖ
    ///
    /// Matches R saemix reference implementation:
    /// - Phase 1 (kiter <= nbiter.burn): γ = 0 (pure MCMC, no stat update)
    /// - Phase 2 (kiter <= nbiter.sa): γ = 1 (SA phase with annealing)
    /// - Phase 3 (kiter > nbiter.sa): γ = 1/(k - n_burn_in + 1) (stochastic approx)
    pub fn current_step_size(&self) -> f64 {
        if self.is_pure_burn_in() {
            // Phase 1: Pure burn-in - no statistics update (γ=0)
            0.0
        } else if self.is_sa_phase() {
            // Phase 2: SA phase - full updates (γ=1) with simulated annealing
            1.0
        } else {
            // Phase 3: Stochastic approximation - decreasing step size
            let post_burnin = self.iteration - self.config.n_burn_in();
            1.0 / (post_burnin as f64).max(1.0)
        }
    }

    /// Run E-step: Sample from p(φ | y, θ) using vectorized MCMC kernels
    ///
    /// This implementation matches R saemix's approach:
    /// 1. All proposals are generated for all subjects at once
    /// 2. Likelihoods are computed in parallel batch
    /// 3. Accept/reject is vectorized
    fn e_step_impl(&mut self) -> Result<()> {
        let n_params = self.population.npar();
        let n_subjects = self.data.len();

        // Get Cholesky of Ω and Ω⁻¹ using faer's built-in methods
        let omega = self.population.omega();
        let llt = omega
            .llt(faer::Side::Lower)
            .map_err(|_| anyhow::anyhow!("Omega not positive definite"))?;
        let chol_omega = llt.L().to_owned();
        let omega_inv = llt.inverse();

        // Subject-specific population means in φ space.
        let mean_phi = subject_mean_phi(
            self.population.mu(),
            self.data.subjects().len(),
            self.subject_covariate_model.as_ref(),
            &self.subject_covariates,
        );

        // Current η for all subjects (N × P matrix)
        // η = φ - μ, so φ = μ + η
        let mut eta_matrix: Array2<f64> = Array2::zeros((n_subjects, n_params));
        for i in 0..n_subjects {
            if !self.chain_states[i].is_empty() {
                for j in 0..n_params {
                    eta_matrix[[i, j]] = self.chain_states[i][0].eta[j];
                }
            }
        }

        // Current log-likelihood for all subjects
        let mut current_ll = batch_log_likelihood_from_eta(
            &self.equation,
            &self.data,
            &self.residual_error_models,
            &self.transforms,
            &eta_matrix,
            &mean_phi,
        )?;

        // Current log-prior for all subjects: -0.5 * η' Ω⁻¹ η
        let mut current_log_prior = log_priors_from_eta_matrix(&eta_matrix, &omega_inv);

        let normal = Normal::new(0.0, 1.0).unwrap();

        // Kernel 1: Full proposals from prior (vectorized)
        for _ in 0..self.config.kernel_config.n_kernel1 {
            // Propose η ~ N(0, Ω) for all subjects
            let mut proposed_eta = Array2::zeros((n_subjects, n_params));
            for i in 0..n_subjects {
                // Generate standard normal vector
                let z: Vec<f64> = (0..n_params)
                    .map(|_| normal.sample(&mut self.rng))
                    .collect();
                // Transform by Cholesky: η = L * z
                for j in 0..n_params {
                    let mut sum = 0.0;
                    for k in 0..=j {
                        sum += chol_omega[(j, k)] * z[k];
                    }
                    proposed_eta[[i, j]] = sum;
                }
            }

            // Compute likelihoods for all proposals in parallel
            let proposed_ll = batch_log_likelihood_from_eta(
                &self.equation,
                &self.data,
                &self.residual_error_models,
                &self.transforms,
                &proposed_eta,
                &mean_phi,
            )?;

            // Accept/reject (vectorized)
            for i in 0..n_subjects {
                // For kernel 1, prior ratio is 1 (both from prior)
                let log_alpha = proposed_ll[i] - current_ll[i];
                let u: f64 = self.rng.random();
                if log_alpha.is_finite() && u.ln() < log_alpha {
                    for j in 0..n_params {
                        eta_matrix[[i, j]] = proposed_eta[[i, j]];
                    }
                    current_ll[i] = proposed_ll[i];
                    // Update prior (all from N(0,Ω), so log_prior is the same form)
                    current_log_prior[i] = subject_log_prior_from_eta(i, &eta_matrix, &omega_inv);
                }
            }
        }

        // Kernel 2: Component-wise random walk with adaptive scaling (vectorized)
        // Uses self.domega2 (persistent) instead of recomputing from omega each iteration
        if self.config.kernel_config.n_kernel2 > 0 {
            let mut n_accepted = vec![0usize; n_params];
            let mut n_total = vec![0usize; n_params];

            for _ in 0..self.config.kernel_config.n_kernel2 {
                for param_idx in 0..n_params {
                    // Propose perturbation for this component for all subjects
                    let mut proposed_eta = eta_matrix.clone();
                    for i in 0..n_subjects {
                        let perturbation = normal.sample(&mut self.rng) * self.domega2[param_idx];
                        proposed_eta[[i, param_idx]] += perturbation;
                    }

                    // Compute likelihoods for all proposals
                    let proposed_ll = batch_log_likelihood_from_eta(
                        &self.equation,
                        &self.data,
                        &self.residual_error_models,
                        &self.transforms,
                        &proposed_eta,
                        &mean_phi,
                    )?;

                    // Compute proposed log-priors
                    let proposed_log_prior = log_priors_from_eta_matrix(&proposed_eta, &omega_inv);

                    // Accept/reject
                    for i in 0..n_subjects {
                        let log_alpha = (proposed_ll[i] + proposed_log_prior[i])
                            - (current_ll[i] + current_log_prior[i]);
                        let u: f64 = self.rng.random();
                        if log_alpha.is_finite() && u.ln() < log_alpha {
                            eta_matrix[[i, param_idx]] = proposed_eta[[i, param_idx]];
                            current_ll[i] = proposed_ll[i];
                            current_log_prior[i] = proposed_log_prior[i];
                            n_accepted[param_idx] += 1;
                        }
                        n_total[param_idx] += 1;
                    }
                }
            }

            // Adapt domega2 based on acceptance rates (R saemix adaptive scaling)
            // domega2 *= 1 + stepsize.rw * (acc_rate - proba.mcmc)
            let stepsize_rw = self.config.kernel_config.rw_step_size;
            let target = self.config.kernel_config.target_acceptance;
            for param_idx in 0..n_params {
                if n_total[param_idx] > 0 {
                    let acc_rate = n_accepted[param_idx] as f64 / n_total[param_idx] as f64;
                    self.domega2[param_idx] *= 1.0 + stepsize_rw * (acc_rate - target);
                }
            }
        }

        // Kernel 3: Block random walk with adaptive scaling (vectorized)
        let block_size = (self.iteration % n_params.max(2).saturating_sub(1)).max(1) + 1;
        let block_size = block_size.min(n_params);

        if self.config.kernel_config.n_kernel3 > 0 {
            let mut n_accepted = vec![0usize; n_params];
            let mut n_total = vec![0usize; n_params];

            for _ in 0..self.config.kernel_config.n_kernel3 {
                // Select block of parameters
                let block_indices: Vec<usize> = if block_size < n_params {
                    let mut indices: Vec<usize> = (0..n_params).collect();
                    // Shuffle first block_size elements using Fisher-Yates
                    for k in 0..block_size {
                        let u: f64 = self.rng.random();
                        let remaining = n_params - k;
                        let swap_offset = (u * remaining as f64).floor() as usize;
                        let swap_idx = k + swap_offset.min(remaining - 1);
                        indices.swap(k, swap_idx);
                    }
                    indices[..block_size].to_vec()
                } else {
                    (0..n_params).collect()
                };

                // Propose perturbations for the block using adaptive domega2
                let mut proposed_eta = eta_matrix.clone();
                for i in 0..n_subjects {
                    for &j in &block_indices {
                        let perturbation = normal.sample(&mut self.rng) * self.domega2[j];
                        proposed_eta[[i, j]] += perturbation;
                    }
                }

                // Compute likelihoods
                let proposed_ll = batch_log_likelihood_from_eta(
                    &self.equation,
                    &self.data,
                    &self.residual_error_models,
                    &self.transforms,
                    &proposed_eta,
                    &mean_phi,
                )?;

                // Compute proposed log-priors
                let proposed_log_prior = log_priors_from_eta_matrix(&proposed_eta, &omega_inv);

                // Accept/reject
                for i in 0..n_subjects {
                    let log_alpha = (proposed_ll[i] + proposed_log_prior[i])
                        - (current_ll[i] + current_log_prior[i]);
                    let u: f64 = self.rng.random();
                    if log_alpha.is_finite() && u.ln() < log_alpha {
                        for &j in &block_indices {
                            eta_matrix[[i, j]] = proposed_eta[[i, j]];
                            n_accepted[j] += 1;
                        }
                        current_ll[i] = proposed_ll[i];
                        current_log_prior[i] = proposed_log_prior[i];
                    }
                    for &j in &block_indices {
                        n_total[j] += 1;
                    }
                }
            }

            // Adapt domega2 based on acceptance rates for kernel 3
            let stepsize_rw = self.config.kernel_config.rw_step_size;
            let target = self.config.kernel_config.target_acceptance;
            for param_idx in 0..n_params {
                if n_total[param_idx] > 0 {
                    let acc_rate = n_accepted[param_idx] as f64 / n_total[param_idx] as f64;
                    self.domega2[param_idx] *= 1.0 + stepsize_rw * (acc_rate - target);
                }
            }
        }

        // Update chain states with final η
        for i in 0..n_subjects {
            let eta = Col::from_fn(n_params, |j| eta_matrix[[i, j]]);
            if self.chain_states[i].is_empty() {
                self.chain_states[i].push(ChainState {
                    eta: eta.clone(),
                    log_likelihood: current_ll[i],
                    log_prior: current_log_prior[i],
                });
            } else {
                self.chain_states[i][0].eta = eta.clone();
                self.chain_states[i][0].log_likelihood = current_ll[i];
                self.chain_states[i][0].log_prior = current_log_prior[i];
            }
        }

        // Accumulate sufficient statistics
        let mut new_stats = SufficientStats::new(n_params);
        let mut individuals = Vec::with_capacity(n_subjects);

        for i in 0..n_subjects {
            let phi = Col::from_fn(n_params, |j| mean_phi[i][j] + eta_matrix[[i, j]]);
            new_stats.accumulate(&phi)?;

            let eta = Col::from_fn(n_params, |j| eta_matrix[[i, j]]);
            let subject_id = self.data.subjects()[i].id().clone();
            // Note: We store phi (unconstrained) in Individual. The field is named "psi"
            // but for SAEM it contains the φ values. Transform to ψ when needed.
            let individual = Individual::new(subject_id, eta, phi)?;
            individuals.push(individual);
        }

        // Stochastic approximation update
        let step_size = self.current_step_size();
        self.sufficient_stats
            .stochastic_update(&new_stats, step_size)?;

        // Update individual estimates
        self.individual_estimates = IndividualEstimates::from_vec(individuals);

        Ok(())
    }

    /// Run M-step: Update population parameters from sufficient statistics
    ///
    /// During pure burn-in (γ=0), we skip parameter updates entirely.
    /// This matches R saemix behavior where nbiter.burn is pure MCMC exploration.
    fn m_step_impl(&mut self) -> Result<()> {
        // During pure burn-in, don't update parameters - just explore
        if self.is_pure_burn_in() {
            // Still track history for monitoring
            let mut params = Vec::new();
            for i in 0..self.population.npar() {
                params.push(self.population.mu()[i]);
            }

            for i in 0..self.population.npar() {
                params.push(self.population.omega()[(i, i)]);
            }
            params.push(self.sigma_sq.sqrt());
            self.param_history.push(params);
            // Compute approximate objf even during burn-in for monitoring
            self.objf = approximate_objective_from_individuals(
                &self.equation,
                &self.data,
                &self.residual_error_models,
                &self.transforms,
                &self.population,
                &self.individual_estimates,
                &subject_mean_phi(
                    self.population.mu(),
                    self.data.subjects().len(),
                    self.subject_covariate_model.as_ref(),
                    &self.subject_covariates,
                ),
            );
            self.objf_history.push(self.objf);
            return Ok(());
        }

        let (mu, omega) = if self.subject_covariate_model.is_some() {
            self.compute_covariate_aware_m_step()?
        } else {
            self.sufficient_stats.compute_m_step()?
        };

        // Apply simulated annealing to variance during SA variance phase
        // R saemix: applies floor decay for nbiter.sa iterations (typically K₁/2)
        let omega_constrained = if self.is_variance_floor_active() {
            self.apply_simulated_annealing(&omega)
        } else {
            omega
        };

        // Update population parameters
        self.population.update_mu(mu)?;

        // Ensure Omega remains positive definite before updating
        // R saemix uses cutoff on diagonal: domega <- cutoff(diag(omega), .Machine$double.eps)
        let omega_pd = ensure_positive_definite_covariance(&omega_constrained);
        self.population.update_omega(omega_pd)?;

        if self.subject_covariate_model.is_some() {
            let subject_means = subject_mean_phi(
                self.population.mu(),
                self.data.subjects().len(),
                self.subject_covariate_model.as_ref(),
                &self.subject_covariates,
            );
            self.recenter_subject_effects(&subject_means)?;
        }

        // Update residual error (simplified - could be more sophisticated)
        self.update_residual_error()?;

        // Store parameter history
        let mut params = Vec::new();
        for i in 0..self.population.npar() {
            params.push(self.population.mu()[i]);
        }
        for i in 0..self.population.npar() {
            params.push(self.population.omega()[(i, i)]);
        }
        params.push(self.sigma_sq.sqrt());
        self.param_history.push(params);

        // Update objective function
        self.prev_objf = self.objf;
        self.objf = approximate_objective_from_individuals(
            &self.equation,
            &self.data,
            &self.residual_error_models,
            &self.transforms,
            &self.population,
            &self.individual_estimates,
            &subject_mean_phi(
                self.population.mu(),
                self.data.subjects().len(),
                self.subject_covariate_model.as_ref(),
                &self.subject_covariates,
            ),
        );
        self.objf_history.push(self.objf);

        Ok(())
    }

    /// Apply simulated annealing to variance (prevent premature convergence)
    ///
    /// Matches R saemix: diag.omega = max(diag.omega.new, floor * alpha)
    /// This ensures variance doesn't collapse too quickly during burn-in
    fn apply_simulated_annealing(&mut self, omega: &Mat<f64>) -> Mat<f64> {
        let n = omega.nrows();
        let mut omega_sa = omega.clone();

        // R formula: take max of new variance vs decayed floor
        for i in 0..n {
            let decayed_floor = self.variance_floor[i] * self.config.sa_alpha;
            // New variance should be at least the decayed floor
            omega_sa[(i, i)] = omega[(i, i)].max(decayed_floor);
            // Update floor for next iteration (don't let it go below minimum)
            self.variance_floor[i] = decayed_floor.max(self.config.sa_min_var);
        }

        omega_sa
    }

    /// Update residual error variance from residuals
    ///
    /// Uses **prediction-based** weighting to match R saemix behavior:
    /// - For constant error: σ² = Σ(y - f)² / n
    /// - For proportional error: σ² = Σ(y - f)² / f² / n  (f = prediction)
    /// - For combined error: uses current σ estimate for weighting
    ///
    /// Follows R saemix approach:
    /// 1. Compute current residuals: statr = Σ(weighted_res²)
    /// 2. Update sufficient statistic via SA: statrese = statrese + γ*(statr - statrese)
    /// 3. Compute sig² = statrese / nobs
    /// 4. During SA phase: pres = max(pres * alpha, sqrt(sig²))
    /// 5. After SA phase: normal stochastic approximation
    fn update_residual_error(&mut self) -> Result<()> {
        let step_size = self.current_step_size();
        let use_annealed_sigma_floor = self.is_variance_floor_active();
        let allow_sigma_update = !self.is_pure_burn_in();
        let update = update_residual_error_from_individuals(
            &self.equation,
            &self.data,
            &mut self.residual_error_models,
            &self.transforms,
            &self.individual_estimates,
            step_size,
            self.sigma_sq,
            self.statrese,
            use_annealed_sigma_floor,
            self.config.sa_alpha,
            allow_sigma_update,
        )?;

        self.sigma_sq = update.sigma_sq;
        self.statrese = update.statrese;
        Ok(())
    }

    /// Get parameter history for convergence diagnostics
    pub fn param_history(&self) -> &Vec<Vec<f64>> {
        &self.param_history
    }

    /// Get current residual error standard deviation
    pub fn sigma(&self) -> f64 {
        self.sigma_sq.sqrt()
    }
}

impl<E: Equation + Send + 'static> ParametricAlgorithm<E> for FSAEM<E> {
    fn equation(&self) -> &E {
        &self.equation
    }

    fn data(&self) -> &Data {
        &self.data
    }

    fn population(&self) -> &Population {
        &self.population
    }

    fn population_mut(&mut self) -> &mut Population {
        &mut self.population
    }

    fn individual_estimates(&self) -> &IndividualEstimates {
        &self.individual_estimates
    }

    fn iteration(&self) -> usize {
        self.iteration
    }

    fn increment_iteration(&mut self) -> usize {
        self.iteration += 1;
        self.iteration
    }

    fn objective_function(&self) -> f64 {
        self.objf
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    fn initialize(&mut self) -> Result<()> {
        tracing::info!("Initializing f-SAEM algorithm");
        tracing::info!(
            "Configuration: {} pure burn-in, {} SA, {} total iterations, {} chains",
            self.config.n_pure_burn,
            self.config.n_sa,
            self.config.n_iterations,
            self.config.n_chains
        );

        let initialized_population =
            initialize_population_in_phi_space(&mut self.population, &self.transforms)?;
        let mu_psi = initialized_population.mu_psi;
        let mu_phi = initialized_population.mu_phi;
        let omega_phi = initialized_population.omega_phi;

        if let Some(model) = self.subject_covariate_model.as_mut() {
            let initialize_intercepts =
                (0..model.beta().nrows()).all(|index| model.beta()[index].abs() < 1e-12);
            if initialize_intercepts {
                let intercepts = (0..self.population.npar())
                    .map(|index| self.population.mu()[index])
                    .collect::<Vec<_>>();
                model.set_intercepts(&intercepts)?;
            }
        }

        let n_params = self.population.npar();

        // Show transforms being used
        let transform_names: Vec<&str> = self.transforms.iter().map(transform_label).collect();

        // Print to stderr for real-time feedback
        eprintln!(
            "\n╔══════════════════════════════════════════════════════════════════════════════╗"
        );
        eprintln!(
            "║                            f-SAEM Algorithm                                  ║"
        );
        eprintln!(
            "╠══════════════════════════════════════════════════════════════════════════════╣"
        );
        eprintln!(
            "║ Phases: {} burn-in → {} SA → {} estimation │ {} chains │ {} subjects        ",
            self.config.n_pure_burn,
            self.config.n_sa,
            self.config.n_iterations - self.config.n_burn_in(),
            self.config.n_chains,
            self.data.subjects().len()
        );
        eprintln!("║ Transforms: {:?}", transform_names);
        eprintln!(
            "║ Initial μ(ψ): {:?}",
            mu_psi
                .as_slice()
                .iter()
                .map(|value| format!("{:.4}", value))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "║ Initial μ(φ): {:?}",
            mu_phi
                .as_slice()
                .iter()
                .map(|value| format!("{:.4}", value))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "║ Initial ω²(φ): {:?}",
            (0..n_params)
                .map(|i| format!("{:.4}", omega_phi[(i, i)]))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "╚══════════════════════════════════════════════════════════════════════════════╝\n"
        );

        // Initialize chain states from population distribution
        let subjects = self.data.subjects();

        for i in 0..subjects.len() {
            for chain_idx in 0..self.config.n_chains {
                // Sample initial η from N(0, Ω)
                let eta = sample_eta_from_population(&self.population, &mut self.rng);
                self.chain_states[i][chain_idx] = ChainState::new(eta);
            }
        }

        // Initialize sufficient statistics
        self.sufficient_stats = SufficientStats::new(n_params);

        // Initialize variance floor
        for i in 0..n_params {
            self.variance_floor[i] = self.population.omega()[(i, i)];
        }

        Ok(())
    }

    fn e_step(&mut self) -> Result<()> {
        self.e_step_impl()
    }

    fn m_step(&mut self) -> Result<()> {
        self.m_step_impl()
    }

    fn evaluate(&mut self) -> Result<Status> {
        // Check for stop file
        if std::path::Path::new("stop").exists() {
            self.status = Status::Stop(StopReason::Stopped);
            return Ok(self.status.clone());
        }

        // Check max iterations
        if self.iteration >= self.config.n_iterations {
            self.status = Status::Stop(StopReason::MaxCycles);
            return Ok(self.status.clone());
        }

        // Only check convergence after burn-in
        if !self.is_burn_in() && self.iteration > self.config.n_burn_in() + 10 {
            // Check objective function convergence
            let objf_change = (self.objf - self.prev_objf).abs() / (1.0 + self.prev_objf.abs());
            if objf_change < self.config.base.objective_tolerance {
                self.status = Status::Stop(StopReason::Converged);
                return Ok(self.status.clone());
            }
        }

        self.status = Status::Continue;
        Ok(self.status.clone())
    }

    fn log_iteration(&mut self) {
        use std::io::Write;

        let phase = if self.is_pure_burn_in() {
            "burn-in"
        } else if self.is_sa_phase() {
            "SA"
        } else {
            "est"
        };

        // Log via tracing (for file output)
        tracing::info!(
            "f-SAEM iter {} ({}): -2LL ≈ {:.4}, γ = {:.4}, σ = {:.4}",
            self.iteration,
            phase,
            self.objf,
            self.current_step_size(),
            self.sigma()
        );

        // Also print progress to stdout for real-time feedback (every 10 iterations or key moments)
        let should_print = self.iteration == 1
            || self.iteration == self.config.n_pure_burn
            || self.iteration == self.config.n_burn_in()
            || self.iteration % 10 == 0
            || self.iteration == self.config.n_iterations;

        if should_print {
            // Transform μ from φ space to ψ space for display
            // μ in storage is in φ space, but users want to see ψ (natural scale)
            let mu_phi = PhiVector::from(self.population.mu());
            let mu_psi = phi_to_psi(&self.transforms, &mu_phi);

            // Format parameter values (showing ψ = natural scale)
            let mu_str: Vec<String> = mu_psi
                .as_slice()
                .iter()
                .map(|value| format!("{:.4}", value))
                .collect();
            let omega_diag_str: Vec<String> = self
                .population
                .variances_as_vec()
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect();

            eprintln!(
                "[SAEM {:>4}/{}] {:>7} | -2LL: {:>12.4} | γ: {:.3} | σ: {:.4} | μ(ψ): [{}] | ω²: [{}]",
                self.iteration,
                self.config.n_iterations,
                phase,
                self.objf,
                self.current_step_size(),
                self.sigma(),
                mu_str.join(", "),
                omega_diag_str.join(", ")
            );
            let _ = std::io::stderr().flush();
        }

        if self.iteration % 10 == 0 || self.iteration <= 5 {
            // Show both φ and ψ spaces in debug
            let mu_phi = PhiVector::from(self.population.mu());
            let mu_psi = phi_to_psi(&self.transforms, &mu_phi);
            tracing::debug!("  μ(φ): {:?}", self.population.mu_as_vec());
            tracing::debug!("  μ(ψ): {:?}", mu_psi.as_slice());
            tracing::debug!("  diag(Ω): {:?}", self.population.variances_as_vec());
        }
    }

    fn into_result(&self) -> Result<crate::estimation::parametric::ParametricWorkspace<E>> {
        let eta_samples_by_subject = self
            .chain_states
            .iter()
            .map(|chain_states| chain_states.iter().map(|state| state.eta.clone()).collect())
            .collect::<Vec<Vec<Col<f64>>>>();
        let (likelihood_estimates, minus2ll) = saem_posthoc_likelihood(
            &self.equation,
            &self.data,
            &self.residual_error_models,
            &self.transforms,
            &self.population,
            &self.individual_estimates,
            &eta_samples_by_subject,
            self.config.seed,
        )?;
        tracing::info!("-2LL computed by importance sampling: {:.4}", minus2ll);

        assemble_saem_result(SaemResultInput {
            equation: &self.equation,
            data: &self.data,
            population: &self.population,
            individual_estimates: &self.individual_estimates,
            objf: minus2ll,
            iterations: self.iteration,
            status: &self.status,
            run_configuration: self.run_configuration.clone(),
            likelihood_estimates,
            param_history: &self.param_history,
            objf_history: &self.objf_history,
            burn_in_iterations: self.config.n_burn_in(),
            sigma_sq: self.sigma_sq,
            transforms: &self.transforms,
            covariates: Some(covariate_state(
                self.subject_covariate_model.as_ref(),
                &self.subject_covariates,
                self.occasion_covariate_model.as_ref(),
                &self.occasion_covariates,
            )),
        })
    }

    fn sufficient_stats(&self) -> Option<&SufficientStats> {
        Some(&self.sufficient_stats)
    }
}

impl<E: Equation + Send + 'static> FSAEM<E> {
    fn compute_covariate_aware_m_step(&mut self) -> Result<(Col<f64>, Mat<f64>)> {
        let step_size = self.current_step_size();
        let Some(model) = self.subject_covariate_model.clone() else {
            return self.sufficient_stats.compute_m_step();
        };

        let (updated_model, _subject_means, mu, omega) = blended_subject_covariate_m_step(
            &model,
            &self.subject_covariates,
            &self.individual_estimates,
            &self.population,
            step_size,
        )?;
        self.subject_covariate_model = Some(updated_model);

        Ok((mu, omega))
    }

    fn recenter_subject_effects(&mut self, subject_means: &[Col<f64>]) -> Result<()> {
        self.individual_estimates =
            recenter_individual_estimates(&self.individual_estimates, subject_means)?;

        for (subject_index, individual) in self.individual_estimates.iter().enumerate() {
            if let Some(chain_state) = self
                .chain_states
                .get_mut(subject_index)
                .and_then(|states| states.get_mut(0))
            {
                chain_state.eta = individual.eta().clone();
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parametric::estimate_beta;
    use crate::estimation::parametric::StepSizeSchedule;

    #[test]
    fn test_cholesky_faer() {
        // Test that faer's built-in Cholesky works correctly
        let mat = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.5 });

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
    fn test_invert_faer() {
        // Test that faer's built-in inversion works correctly
        let mat = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.5 });

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
    fn test_step_size_schedule() {
        let schedule = StepSizeSchedule::new_saem(100, 200);

        // During burn-in: step_size should be 1.0
        assert_eq!(schedule.step_size(1), 1.0);
        assert_eq!(schedule.step_size(50), 1.0);
        assert_eq!(schedule.step_size(99), 1.0);

        // At burn-in boundary: step_size(100) = 1/(100-100+1) = 1.0
        assert_eq!(schedule.step_size(100), 1.0);

        // After burn-in: step_size should decrease
        // step_size(110) = 1/(110-100+1) = 1/11
        // step_size(150) = 1/(150-100+1) = 1/51
        assert!(schedule.step_size(150) < schedule.step_size(110));
        assert!((schedule.step_size(110) - 1.0 / 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_beta_recovers_subject_covariate_effects() {
        let model = CovariateModel::new(vec!["CL", "V"], vec!["WT"], vec![vec![true], vec![false]])
            .unwrap();

        let subject_covariates = vec![
            HashMap::from([(String::from("WT"), 60.0)]),
            HashMap::from([(String::from("WT"), 80.0)]),
        ];
        let individuals = IndividualEstimates::from_vec(vec![
            Individual::new(
                "1",
                Col::from_fn(2, |_| 0.0),
                Col::from_fn(2, |index| if index == 0 { 11.0 } else { 50.0 }),
            )
            .unwrap(),
            Individual::new(
                "2",
                Col::from_fn(2, |_| 0.0),
                Col::from_fn(2, |index| if index == 0 { 13.0 } else { 50.0 }),
            )
            .unwrap(),
        ]);

        let beta = estimate_beta(&model, &subject_covariates, &individuals).unwrap();

        assert!((beta[0] - 5.0).abs() < 1e-5);
        assert!((beta[1] - 0.1).abs() < 1e-6);
        assert!((beta[2] - 50.0).abs() < 1e-5);
    }
}
