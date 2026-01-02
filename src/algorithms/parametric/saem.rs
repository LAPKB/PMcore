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

use anyhow::{bail, Result};
use faer::{Col, Mat};
use ndarray::Array2;
use pharmsol::{Data, Equation, Event, Predictions, ResidualErrorModels, Subject};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

use crate::algorithms::{Status, StopReason};
use crate::routines::output::ParametricResult;
use crate::routines::sampling::{ChainState, KernelConfig};
use crate::routines::settings::Settings;
use crate::structs::parametric::{
    Individual, IndividualEstimates, ParameterTransform, Population, SufficientStats,
};

use super::algorithm::{ParametricAlgorithm, ParametricConfig};
use crate::routines::settings::SaemSettings;

/// f-SAEM algorithm configuration
///
/// This structure can be constructed from [`SaemSettings`] to ensure
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
    /// Create configuration from user-facing SaemSettings
    ///
    /// This ensures the algorithm uses the settings specified by the user.
    pub fn from_settings(settings: &SaemSettings) -> Self {
        let k1 = settings.k1_iterations;
        let k2 = settings.k2_iterations;

        Self {
            base: ParametricConfig {
                max_iterations: k1 + k2,
                burn_in: settings.burn_in,
                n_chains: settings.n_chains,
                n_samples: settings.mcmc_iterations,
                parameter_tolerance: 1e-4,
                objective_tolerance: 1e-4,
                use_annealing: settings.sa_iterations > 0,
                initial_temperature: 1.0,
            },
            kernel_config: KernelConfig {
                n_kernel1: 2,
                n_kernel2: 2,
                n_kernel3: 2,
                n_kernel4: if settings.n_kernels >= 4 { 0 } else { 0 }, // Kernel 4 disabled by default
                map_iterations: 0,
                rw_step_size: settings.mcmc_step_size,
                target_acceptance: 0.4,
            },
            n_chains: settings.n_chains,
            seed: settings.seed,
            n_pure_burn: settings.burn_in,
            n_sa: k1.saturating_sub(settings.burn_in),
            n_iterations: k1 + k2,
            sa_alpha: settings.sa_cooling_factor,
            sa_min_var: settings.omega_min_variance,
            // R saemix: nbiter.sa = K₁/2 for variance floor decay
            n_sa_variance: if settings.sa_iterations > 0 {
                settings.sa_iterations
            } else {
                k1 / 2
            },
        }
    }

    /// Create configuration with auto-scaling for small datasets
    ///
    /// R saemix automatically increases chains when N < 50:
    /// nb.chains = ceiling(50/N)
    pub fn from_settings_with_auto_chains(settings: &SaemSettings, n_subjects: usize) -> Self {
        let mut config = Self::from_settings(settings);

        // R saemix: if N < 50, auto-increase chains
        if n_subjects < 50 && settings.n_chains == 1 {
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
    /// Algorithm settings
    settings: Settings,
    /// Pharmacokinetic/pharmacodynamic model
    equation: E,
    /// Population data
    data: Data,
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
    /// Parameter transforms (φ ↔ ψ conversions)
    /// Maps between unconstrained space (φ) and constrained space (ψ)
    transforms: Vec<ParameterTransform>,
}

impl<E: Equation + Send + 'static> FSAEM<E> {
    /// Create a new f-SAEM algorithm instance using settings
    ///
    /// This constructor reads configuration from [`SaemSettings`] to ensure
    /// the algorithm behaves according to user-specified parameters.
    /// For small datasets (N < 50), MCMC chains are automatically scaled
    /// following R saemix behavior.
    pub fn create(settings: Settings, equation: E, data: Data) -> Result<Self> {
        let n_subjects = data.subjects().len();
        let config =
            FSaemConfig::from_settings_with_auto_chains(settings.advanced().saem(), n_subjects);
        Self::create_with_config(settings, equation, data, config)
    }

    /// Create with custom configuration (advanced users)
    ///
    /// Use this when you need fine-grained control over algorithm parameters
    /// beyond what [`SaemSettings`] provides.
    pub fn create_with_config(
        settings: Settings,
        equation: E,
        data: Data,
        config: FSaemConfig,
    ) -> Result<Self> {
        // Initialize population from parameter settings
        let population = Population::from_parameters(settings.parameters().clone())?;
        let n_params = population.npar();
        let n_subjects = data.subjects().len();

        // Initialize parameter transforms from settings
        // Get transform codes from SAEM settings, defaulting to LogNormal (1) for all
        let transform_codes = settings.advanced().saem.get_transforms(n_params);
        let transforms: Vec<ParameterTransform> = transform_codes
            .iter()
            .map(|&code| ParameterTransform::from_saemix_code(code))
            .collect();

        // Initialize sufficient statistics
        let sufficient_stats = SufficientStats::new(n_params);

        // Initialize residual error variance
        let sigma_sq = estimate_initial_sigma_sq(&settings);
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

        // Random number generator
        let rng = ChaCha8Rng::seed_from_u64(config.seed);

        // Initialize residual error models for prediction-based sigma calculation
        // This is required for parametric algorithms
        let residual_error_models = settings
            .residual_error()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("SAEM requires residual_error to be set in settings"))?;

        Ok(Self {
            settings,
            equation,
            data,
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
            transforms,
        })
    }

    /// Check if currently in burn-in phase (Phase 1 + Phase 2)
    /// - Phase 1: Pure burn-in (γ=0)
    /// - Phase 2: SA phase (γ=1)
    pub fn is_burn_in(&self) -> bool {
        self.iteration <= self.config.n_burn_in()
    }

    /// Check if currently in pure burn-in phase (γ=0, no stat updates)
    pub fn is_pure_burn_in(&self) -> bool {
        self.iteration <= self.config.n_pure_burn
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

        // Get Cholesky of Ω for sampling from prior
        let omega = self.population.omega();
        let chol_omega = cholesky_lower(omega)?;

        // Get Ω⁻¹ for computing prior density
        let omega_inv = invert_symmetric(omega)?;

        // Population mean in φ space
        let mean_phi = self.population.mu();

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
        let mut current_ll = self.compute_batch_log_likelihood(&eta_matrix, mean_phi)?;

        // Current log-prior for all subjects: -0.5 * η' Ω⁻¹ η
        let mut current_log_prior: Vec<f64> = (0..n_subjects)
            .map(|i| {
                let mut quad = 0.0;
                for j in 0..n_params {
                    for k in 0..n_params {
                        quad += eta_matrix[[i, j]] * omega_inv[(j, k)] * eta_matrix[[i, k]];
                    }
                }
                -0.5 * quad
            })
            .collect();

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
            let proposed_ll = self.compute_batch_log_likelihood(&proposed_eta, mean_phi)?;

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
                    let mut quad = 0.0;
                    for j in 0..n_params {
                        for k in 0..n_params {
                            quad += eta_matrix[[i, j]] * omega_inv[(j, k)] * eta_matrix[[i, k]];
                        }
                    }
                    current_log_prior[i] = -0.5 * quad;
                }
            }
        }

        // Kernel 2: Component-wise random walk (vectorized)
        let domega = Col::from_fn(n_params, |j| {
            omega[(j, j)].sqrt() * self.config.kernel_config.rw_step_size
        });

        for _ in 0..self.config.kernel_config.n_kernel2 {
            for param_idx in 0..n_params {
                // Propose perturbation for this component for all subjects
                let mut proposed_eta = eta_matrix.clone();
                for i in 0..n_subjects {
                    let perturbation = normal.sample(&mut self.rng) * domega[param_idx];
                    proposed_eta[[i, param_idx]] += perturbation;
                }

                // Compute likelihoods for all proposals
                let proposed_ll = self.compute_batch_log_likelihood(&proposed_eta, mean_phi)?;

                // Compute proposed log-priors
                let proposed_log_prior: Vec<f64> = (0..n_subjects)
                    .map(|i| {
                        let mut quad = 0.0;
                        for j in 0..n_params {
                            for k in 0..n_params {
                                quad +=
                                    proposed_eta[[i, j]] * omega_inv[(j, k)] * proposed_eta[[i, k]];
                            }
                        }
                        -0.5 * quad
                    })
                    .collect();

                // Accept/reject
                for i in 0..n_subjects {
                    let log_alpha = (proposed_ll[i] + proposed_log_prior[i])
                        - (current_ll[i] + current_log_prior[i]);
                    let u: f64 = self.rng.random();
                    if log_alpha.is_finite() && u.ln() < log_alpha {
                        eta_matrix[[i, param_idx]] = proposed_eta[[i, param_idx]];
                        current_ll[i] = proposed_ll[i];
                        current_log_prior[i] = proposed_log_prior[i];
                    }
                }
            }
        }

        // Kernel 3: Block random walk (vectorized)
        let block_size = (self.iteration % n_params.max(2).saturating_sub(1)).max(1) + 1;
        let block_size = block_size.min(n_params);

        for _ in 0..self.config.kernel_config.n_kernel3 {
            // Select block of parameters
            let block_indices: Vec<usize> = if block_size < n_params {
                let mut indices: Vec<usize> = (0..n_params).collect();
                // Shuffle first block_size elements using Fisher-Yates
                for k in 0..block_size {
                    // Generate random index in remaining range using uniform
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

            // Propose perturbations for the block
            let mut proposed_eta = eta_matrix.clone();
            for i in 0..n_subjects {
                for &j in &block_indices {
                    let perturbation = normal.sample(&mut self.rng) * domega[j];
                    proposed_eta[[i, j]] += perturbation;
                }
            }

            // Compute likelihoods
            let proposed_ll = self.compute_batch_log_likelihood(&proposed_eta, mean_phi)?;

            // Compute proposed log-priors
            let proposed_log_prior: Vec<f64> = (0..n_subjects)
                .map(|i| {
                    let mut quad = 0.0;
                    for j in 0..n_params {
                        for k in 0..n_params {
                            quad += proposed_eta[[i, j]] * omega_inv[(j, k)] * proposed_eta[[i, k]];
                        }
                    }
                    -0.5 * quad
                })
                .collect();

            // Accept/reject
            for i in 0..n_subjects {
                let log_alpha = (proposed_ll[i] + proposed_log_prior[i])
                    - (current_ll[i] + current_log_prior[i]);
                let u: f64 = self.rng.random();
                if log_alpha.is_finite() && u.ln() < log_alpha {
                    for &j in &block_indices {
                        eta_matrix[[i, j]] = proposed_eta[[i, j]];
                    }
                    current_ll[i] = proposed_ll[i];
                    current_log_prior[i] = proposed_log_prior[i];
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
            let phi = Col::from_fn(n_params, |j| mean_phi[j] + eta_matrix[[i, j]]);
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

    /// Compute log-likelihoods for all subjects in parallel batch
    /// Uses prediction-based sigma (parametric formulation) to match R saemix
    fn compute_batch_log_likelihood(
        &self,
        eta_matrix: &Array2<f64>,
        mean_phi: &Col<f64>,
    ) -> Result<Vec<f64>> {
        let n_subjects = eta_matrix.nrows();
        let n_params = eta_matrix.ncols();

        // Build parameter matrix in ψ space (constrained) for each subject
        let mut psi_params = Array2::<f64>::zeros((n_subjects, n_params));
        for i in 0..n_subjects {
            for j in 0..n_params {
                let phi_ij = mean_phi[j] + eta_matrix[[i, j]];
                psi_params[[i, j]] = self.transforms[j].phi_to_psi(phi_ij);
            }
        }

        // Use pharmsol's log_likelihood_batch for parallel computation
        pharmsol::prelude::simulator::log_likelihood_batch(
            &self.equation,
            &self.data,
            &psi_params,
            &self.residual_error_models,
        )
        .map_err(|e| anyhow::anyhow!("Likelihood computation failed: {}", e))
    }

    /// Transform parameters from unconstrained (φ) to constrained (ψ) space
    ///
    /// This applies the inverse transform h⁻¹(φ) = ψ for each parameter.
    /// The model expects parameters in the constrained space (e.g., positive clearance).
    fn phi_to_psi(&self, phi: &Col<f64>) -> Col<f64> {
        Col::from_fn(phi.nrows(), |i| self.transforms[i].phi_to_psi(phi[i]))
    }

    /// Transform parameters from constrained (ψ) to unconstrained (φ) space
    ///
    /// This applies the transform h(ψ) = φ for each parameter.
    /// MCMC sampling happens in the unconstrained space.
    #[allow(dead_code)]
    fn psi_to_phi(&self, psi: &Col<f64>) -> Col<f64> {
        Col::from_fn(psi.nrows(), |i| self.transforms[i].psi_to_phi(psi[i]))
    }

    /// Compute log-likelihood for a single subject given parameters φ (unconstrained)
    ///
    /// This transforms φ → ψ before evaluating the model, ensuring parameters
    /// stay in their valid domains (e.g., positive clearance).
    ///
    /// NOTE: No Jacobian correction is applied. For MCMC in φ space with prior
    /// p(φ|Ω) = N(μ_φ, Ω), we only need the likelihood p(y|ψ(φ)).
    #[allow(dead_code)]
    fn compute_individual_log_likelihood(&self, subject: &Subject, phi: &Col<f64>) -> f64 {
        // Transform from unconstrained (φ) to constrained (ψ) space
        let psi = self.phi_to_psi(phi);

        // Convert Col<f64> to Vec<f64>
        let params: Vec<f64> = (0..psi.nrows()).map(|i| psi[i]).collect();

        // Simulate to get predictions
        let predictions = match self.equation.estimate_predictions(subject, &params) {
            Ok(preds) => preds,
            Err(_) => return f64::NEG_INFINITY,
        };

        // Extract (outeq, observation, prediction) tuples and compute log-likelihood
        let obs_pred_pairs = predictions
            .get_predictions()
            .into_iter()
            .filter_map(|pred| {
                pred.observation()
                    .map(|obs| (pred.outeq(), obs, pred.prediction()))
            });

        self.residual_error_models
            .total_log_likelihood(obs_pred_pairs)
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
            return Ok(());
        }

        // Get M-step estimates from sufficient statistics
        let (mu, omega) = self.sufficient_stats.compute_m_step()?;

        // Apply simulated annealing to variance during SA variance phase
        // R saemix: applies floor decay for nbiter.sa iterations (typically K₁/2)
        let omega_constrained = if self.is_variance_floor_active() {
            self.apply_simulated_annealing(&omega)
        } else {
            omega
        };

        // Update population parameters
        self.population.update_mu(mu)?;
        self.population.update_omega(omega_constrained)?;

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
        self.objf = self.compute_approximate_objective();

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
        let subjects: Vec<_> = self.data.subjects().to_vec();
        let mut sum_weighted_sq_residuals = 0.0;
        let mut n_obs = 0;

        for (i, subject) in subjects.iter().enumerate() {
            if let Some(individual) = self.individual_estimates.get(i) {
                // Transform φ (unconstrained) to ψ (constrained) for model evaluation
                let phi = individual.psi();
                let params: Vec<f64> = (0..phi.nrows())
                    .map(|j| self.transforms[j].phi_to_psi(phi[j]))
                    .collect();

                // Get predictions
                if let Ok(predictions) = self.equation.estimate_predictions(&subject, &params) {
                    // Collect observations from all occasions
                    let observations: Vec<_> = subject
                        .occasions()
                        .iter()
                        .flat_map(|occ| occ.events().iter())
                        .filter_map(|event| {
                            if let Event::Observation(obs) = event {
                                obs.value().map(|v| (v, obs.outeq()))
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Compute weighted squared residuals using PREDICTION-based weighting
                    for ((obs_value, outeq), pred) in observations
                        .iter()
                        .zip(predictions.get_predictions().iter())
                    {
                        let prediction = pred.prediction();

                        // Use residual error model to compute weighted residual
                        // This uses PREDICTION for sigma, matching R saemix
                        if let Some(error_model) = self.residual_error_models.get(*outeq) {
                            let weighted_sq_res =
                                error_model.weighted_squared_residual(*obs_value, prediction);
                            sum_weighted_sq_residuals += weighted_sq_res;
                        } else {
                            // Fallback: unweighted for missing error model
                            let residual = obs_value - prediction;
                            sum_weighted_sq_residuals += residual * residual;
                        }
                        n_obs += 1;
                    }
                }
            }
        }

        if n_obs > 0 {
            // Step 1: Current iteration's residual statistic
            let statr = sum_weighted_sq_residuals;

            // Step 2: Update sufficient statistic via stochastic approximation
            // R saemix: suffStat$statrese <- suffStat$statrese + stepsize * (statr/nchains - suffStat$statrese)
            // We have 1 chain, so statr/nchains = statr
            let step_size = self.current_step_size();
            if step_size > 0.0 {
                self.statrese = self.statrese + step_size * (statr - self.statrese);
            }

            // Step 3: Compute sig² from sufficient statistic
            let sig2 = self.statrese / n_obs as f64;

            // Step 4: Update sigma with SA floor
            // R saemix: applies SA decay for nbiter.sa iterations
            if self.is_variance_floor_active() {
                // During SA variance phase: pres = max(pres * alpha, sqrt(sig2))
                // R saemix applies SA decay to the *parameter* (pres), not the statistic
                let decayed_sigma = self.sigma_sq.sqrt() * self.config.sa_alpha;
                let new_sigma = sig2.sqrt();
                self.sigma_sq = decayed_sigma.max(new_sigma).powi(2);
            } else if !self.is_pure_burn_in() {
                // After SA phase: use the smoothed sufficient statistic directly
                self.sigma_sq = sig2;
            }
            // During pure burn-in (gamma=0): don't update sigma

            // Update residual error models with new sigma
            self.sync_error_models_with_sigma();
        }

        Ok(())
    }

    /// Synchronize residual error models with the estimated σ
    fn sync_error_models_with_sigma(&mut self) {
        let sigma = self.sigma_sq.sqrt();
        self.residual_error_models.update_sigma(sigma);
    }

    /// Compute approximate objective function (approximate -2LL)
    fn compute_approximate_objective(&self) -> f64 {
        // Use importance sampling approximation
        // This is a simplified version - full implementation would use
        // Monte Carlo integration over the random effects

        let subjects = self.data.subjects();
        let mut total_ll = 0.0;

        for (i, subject) in subjects.iter().enumerate() {
            if let Some(individual) = self.individual_estimates.get(i) {
                // Use current individual estimate for approximation
                let ll = self.compute_individual_log_likelihood(subject, individual.psi());

                // Add prior contribution
                let eta = individual.eta();
                let n = eta.nrows();
                let omega_inv = match invert_symmetric(self.population.omega()) {
                    Ok(inv) => inv,
                    Err(_) => continue,
                };

                let mut prior_term = 0.0;
                for j in 0..n {
                    for k in 0..n {
                        prior_term += eta[j] * omega_inv[(j, k)] * eta[k];
                    }
                }

                total_ll += ll - 0.5 * prior_term;
            }
        }

        -2.0 * total_ll
    }

    /// Compute -2LL using Importance Sampling (matching R saemix llis.saemix)
    ///
    /// This method samples from the conditional posterior p(φᵢ|yᵢ, θ̂) and uses
    /// importance sampling to estimate the marginal likelihood p(y|θ̂).
    ///
    /// The importance sampling estimate is:
    /// p(yᵢ|θ) ≈ (1/M) Σₘ [p(yᵢ|φᵢₘ) * p(φᵢₘ|θ) / q(φᵢₘ)]
    ///
    /// where q is the proposal distribution (conditional posterior with t-distribution tails).
    /// Uses the same algorithm as R saemix's llis.saemix function.
    pub fn compute_ll_importance_sampling(&self, n_samples: usize) -> f64 {
        use rand_distr::StudentT;
        use statrs::function::gamma::ln_gamma;

        let subjects = self.data.subjects();
        let n_params = self.population.npar();

        // Degrees of freedom for t-distribution (matching R saemix default nu.is = 4)
        let nu_is: f64 = 4.0;

        // Get population parameters
        let mu_phi = self.population.mu();
        let omega = self.population.omega();
        let omega_inv = match invert_symmetric(omega) {
            Ok(inv) => inv,
            Err(_) => return f64::NEG_INFINITY,
        };

        // Compute log|Omega| + p*log(2π) for population prior (multivariate normal)
        let log_det_omega = omega.determinant().ln();
        let log_prior_const = log_det_omega + (n_params as f64) * (2.0 * std::f64::consts::PI).ln();

        // RNG for sampling
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed + 12345);

        // StudentT distribution for sampling (like R saemix's trnd.mlx)
        let t_dist = StudentT::new(nu_is).unwrap();

        let mut total_ll = 0.0;

        for (i, subject) in subjects.iter().enumerate() {
            // Get conditional posterior parameters for this subject in φ space
            // In R saemix: mtild.phiM1 = cond.mean.phi (conditional mean from MCMC)
            //
            // The chain states contain eta (random effects) in φ space
            // cond_mean = mu_phi + eta (or just the individual's mean eta from the chain)
            let cond_mean: Col<f64> =
                if !self.chain_states.is_empty() && !self.chain_states[i].is_empty() {
                    // Compute mean of chain states (which are in φ space as eta)
                    let chain = &self.chain_states[i];
                    let n_chain = chain.len();

                    let mut mean_eta: Vec<f64> = vec![0.0; n_params];
                    for state in chain.iter() {
                        for j in 0..n_params {
                            mean_eta[j] = mean_eta[j] + state.eta[j];
                        }
                    }
                    for j in 0..n_params {
                        mean_eta[j] = mean_eta[j] / n_chain as f64;
                    }
                    // cond_mean in φ space = mu_phi + mean(eta)
                    Col::from_fn(n_params, |j| mu_phi[j] + mean_eta[j])
                } else if let Some(ind) = self.individual_estimates.get(i) {
                    // Transform psi to phi
                    let psi = ind.psi();
                    Col::from_fn(n_params, |j| self.transforms[j].psi_to_phi(psi[j]))
                } else {
                    Col::from_fn(n_params, |j| mu_phi[j])
                };

            // Compute conditional variance from chain states in φ space
            // In R saemix: stild.phiM1 = sqrt(cond.var.phi)
            let cond_var: Vec<f64> =
                if !self.chain_states.is_empty() && !self.chain_states[i].is_empty() {
                    let chain = &self.chain_states[i];
                    let n_chain = chain.len();

                    if n_chain > 1 {
                        let mut mean: Vec<f64> = vec![0.0; n_params];
                        for state in chain.iter() {
                            for j in 0..n_params {
                                mean[j] = mean[j] + state.eta[j];
                            }
                        }
                        for j in 0..n_params {
                            mean[j] = mean[j] / n_chain as f64;
                        }

                        let mut var: Vec<f64> = vec![0.0; n_params];
                        for state in chain.iter() {
                            for j in 0..n_params {
                                let diff: f64 = state.eta[j] - mean[j];
                                var[j] = var[j] + diff * diff;
                            }
                        }
                        for j in 0..n_params {
                            let val: f64 = var[j] / (n_chain - 1) as f64;
                            var[j] = val.max(1e-6);
                        }
                        var
                    } else {
                        (0..n_params).map(|j| omega[(j, j)].max(1e-6)).collect()
                    }
                } else {
                    (0..n_params).map(|j| omega[(j, j)].max(1e-6)).collect()
                };

            let mut weights = Vec::with_capacity(n_samples);

            for _ in 0..n_samples {
                // Sample r from t-distribution (like R saemix's trnd.mlx)
                let r: Vec<f64> = (0..n_params).map(|_| t_dist.sample(&mut rng)).collect();

                // phiM1 = mtild.phiM1 + stild.phiM1 * r
                let phi_sample: Col<f64> =
                    Col::from_fn(n_params, |j| cond_mean[j] + cond_var[j].sqrt() * r[j]);

                // Transform to ψ space
                let psi_sample: Vec<f64> = (0..n_params)
                    .map(|j| self.transforms[j].phi_to_psi(phi_sample[j]))
                    .collect();

                // Compute log p(y|φ) - the likelihood (e1 in R saemix)
                let log_lik = {
                    let predictions = match self.equation.estimate_predictions(subject, &psi_sample)
                    {
                        Ok(preds) => preds,
                        Err(_) => {
                            // On error, reject this sample
                            continue;
                        }
                    };
                    let obs_pred_pairs =
                        predictions
                            .get_predictions()
                            .into_iter()
                            .filter_map(|pred| {
                                pred.observation()
                                    .map(|obs| (pred.outeq(), obs, pred.prediction()))
                            });
                    self.residual_error_models
                        .total_log_likelihood(obs_pred_pairs)
                };

                // Compute log p(φ|Ω) - the population prior (e2 in R saemix)
                // d2 = (-0.5)*(rowSums(dphiM*(dphiM%*%IOmega.phi1)) + c2)
                let eta: Col<f64> = Col::from_fn(n_params, |j| phi_sample[j] - mu_phi[j]);
                let mut quad_form = 0.0;
                for j in 0..n_params {
                    for k in 0..n_params {
                        quad_form += eta[j] * omega_inv[(j, k)] * eta[k];
                    }
                }
                let log_prior = -0.5 * (quad_form + log_prior_const);

                // Compute log q(φ) - the proposal density using t-distribution (e3 in R saemix)
                // pitild.phi1 = rowSums(log(tpdf.mlx(r, nu.is)))
                // e3 = pitild.phi1 - 0.5*rowSums(log(cond.var.phi1))
                //
                // log pdf of t-distribution: log(Γ((ν+1)/2)) - log(Γ(ν/2)) - 0.5*log(ν*π) - ((ν+1)/2)*log(1 + r²/ν)
                let mut log_proposal: f64 = 0.0;
                for j in 0..n_params {
                    // Log pdf of t-distribution at r[j]
                    let r_val: f64 = r[j];
                    let log_t_pdf: f64 = ln_gamma((nu_is + 1.0) / 2.0)
                        - ln_gamma(nu_is / 2.0)
                        - 0.5 * (nu_is * std::f64::consts::PI).ln()
                        - ((nu_is + 1.0) / 2.0) * (1.0 + r_val * r_val / nu_is).ln();
                    log_proposal = log_proposal + log_t_pdf;
                }
                // Adjust for the scale (stild.phiM1 = sqrt(cond.var))
                // The proposal is actually t-distributed with scale sqrt(cond_var)
                // log q(φ) = log t_pdf(r) - 0.5*sum(log(cond_var))
                log_proposal -= 0.5 * cond_var.iter().map(|v| v.ln()).sum::<f64>();

                // Importance weight: exp(log_lik + log_prior - log_proposal)
                // sume = e1 + e2 - e3
                let log_weight = log_lik + log_prior - log_proposal;
                weights.push(log_weight);
            }

            // Log-sum-exp for numerical stability
            // meana = rowMeans(exp(sume))
            // LL = sum(log(meana))
            let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_weight.is_finite() {
                let sum_exp: f64 = weights.iter().map(|&w| (w - max_weight).exp()).sum();
                let log_marginal = max_weight + sum_exp.ln() - (n_samples as f64).ln();
                total_ll += log_marginal;
            }
        }

        -2.0 * total_ll
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
    fn new(settings: Settings, equation: E, data: Data) -> Result<Box<Self>> {
        Ok(Box::new(Self::create(settings, equation, data)?))
    }

    fn settings(&self) -> &Settings {
        &self.settings
    }

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

        // Transform initial μ from ψ (natural) to φ (unconstrained) space
        // Population was initialized with midpoints of parameter bounds (ψ space)
        let mu_psi = self.population.mu().clone();
        let mu_phi = self.psi_to_phi(&mu_psi);
        if let Err(e) = self.population.update_mu(mu_phi.clone()) {
            tracing::warn!("Failed to update initial μ to φ space: {}", e);
        }

        // Transform Ω to φ space as well
        // For LogNormal: Ω in ψ space → Ω in φ space
        // We use CV² = exp(ω²_φ) - 1, so ω²_φ = ln(1 + CV²)
        // Initialize with CV ≈ 50% which gives ω²_φ ≈ 0.22
        let n_params = self.population.npar();
        let mut omega_phi = self.population.omega().clone();
        for i in 0..n_params {
            match self.transforms[i] {
                ParameterTransform::LogNormal => {
                    // For LogNormal, use ω²_φ = ln(1 + CV²) with CV ≈ 0.5
                    // This gives reasonable starting variance in φ space
                    let cv: f64 = 0.5; // 50% CV
                    omega_phi[(i, i)] = (1.0 + cv * cv).ln();
                }
                ParameterTransform::Logit { .. } | ParameterTransform::Probit { .. } => {
                    // For bounded transforms, use moderate variance in φ space
                    omega_phi[(i, i)] = 1.0;
                }
                ParameterTransform::None => {
                    // Keep the ψ-space variance (it's already appropriate)
                }
            }
            // Zero out off-diagonals for safety
            for j in 0..n_params {
                if i != j {
                    omega_phi[(i, j)] = 0.0;
                }
            }
        }
        if let Err(e) = self.population.update_omega(omega_phi.clone()) {
            tracing::warn!("Failed to update initial Ω to φ space: {}", e);
        }

        // Show transforms being used
        let transform_names: Vec<&str> = self
            .transforms
            .iter()
            .map(|t| match t {
                ParameterTransform::None => "Normal",
                ParameterTransform::LogNormal => "LogNormal",
                ParameterTransform::Logit { .. } => "Logit",
                ParameterTransform::Probit { .. } => "Probit",
            })
            .collect();

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
            (0..mu_psi.nrows())
                .map(|i| format!("{:.4}", mu_psi[i]))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "║ Initial μ(φ): {:?}",
            (0..mu_phi.nrows())
                .map(|i| format!("{:.4}", mu_phi[i]))
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
                let eta = self.sample_from_prior();
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
            let mu_phi = self.population.mu();
            let mu_psi = self.phi_to_psi(mu_phi);

            // Format parameter values (showing ψ = natural scale)
            let mu_str: Vec<String> = (0..mu_psi.nrows())
                .map(|i| format!("{:.4}", mu_psi[i]))
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
            let mu_psi = self.phi_to_psi(self.population.mu());
            let psi_vec: Vec<f64> = (0..mu_psi.nrows()).map(|i| mu_psi[i]).collect();
            tracing::debug!("  μ(φ): {:?}", self.population.mu_as_vec());
            tracing::debug!("  μ(ψ): {:?}", psi_vec);
            tracing::debug!("  diag(Ω): {:?}", self.population.variances_as_vec());
        }
    }

    fn into_result(&self) -> Result<ParametricResult<E>> {
        use crate::routines::output::ResidualErrorEstimates;

        // Compute -2LL using importance sampling (matching R saemix)
        // R saemix uses 100*K samples where K is typically 100, so 10000 total
        let n_is_samples = 10000; // Number of importance sampling samples
        let minus2ll = self.compute_ll_importance_sampling(n_is_samples);
        tracing::info!("-2LL computed by importance sampling: {:.4}", minus2ll);

        // Construct iteration log from parameter history
        let mut iteration_log = crate::routines::output::ParametricIterationLog::new();
        for (i, params) in self.param_history.iter().enumerate() {
            // Reconstruct a minimal population state for logging
            let n = self.population.npar();
            let mu = Col::from_fn(n, |j| params.get(j).copied().unwrap_or(0.0));
            let omega = Mat::from_fn(n, n, |r, c| {
                if r == c {
                    params.get(n + r).copied().unwrap_or(0.0)
                } else {
                    0.0
                }
            });

            if let Ok(pop) = Population::new(mu, omega, self.population.parameters().clone()) {
                let status_str = if i < self.config.n_burn_in() {
                    Status::Continue
                } else {
                    self.status.clone()
                };
                iteration_log.log_iteration(i + 1, minus2ll, &pop, &status_str);
            }
        }

        // Transform population from φ (unconstrained) space back to ψ (natural) space
        // SAEM operates internally in φ space, but results should be in ψ space
        let mu_phi = self.population.mu();
        let mu_psi = self.phi_to_psi(mu_phi);

        // Create a new population in ψ space
        // Note: Ω in φ space and ψ space have different interpretations
        // For LogNormal: Var(ψ) = E[ψ]² * (exp(ω²_φ) - 1)
        // For now, we report ω² in φ space (consistent with how NLME models typically report)
        let population_psi = Population::new(
            mu_psi,
            self.population.omega().clone(),
            self.population.parameters().clone(),
        )?;

        // Get the estimated residual error (σ = sqrt(σ²))
        // The residual error model type depends on what was configured in settings
        let sigma_estimate = self.sigma_sq.sqrt();

        // Determine sigma type from the residual error models
        // For now we report sigma as additive (most common for SAEM)
        // TODO: Properly detect the error model type from settings
        let sigma = ResidualErrorEstimates::additive(sigma_estimate);

        // Construct result object with sigma
        let mut result = ParametricResult::with_sigma(
            self.equation.clone(),
            self.data.clone(),
            population_psi,
            self.individual_estimates.clone(),
            minus2ll,
            self.iteration,
            self.status.clone(),
            self.settings.clone(),
            iteration_log,
            sigma,
        );

        // Set the likelihood estimates with IS value
        let mut ll_estimates = crate::routines::output::parametric::LikelihoodEstimates::new();
        ll_estimates.ll_importance_sampling = Some(-minus2ll / 2.0);
        ll_estimates.is_n_samples = Some(n_is_samples);
        result.set_likelihood_estimates(ll_estimates);

        Ok(result)
    }

    fn sufficient_stats(&self) -> Option<&SufficientStats> {
        Some(&self.sufficient_stats)
    }
}

impl<E: Equation + Send + 'static> FSAEM<E> {
    /// Sample from prior distribution N(0, Ω)
    fn sample_from_prior(&mut self) -> Col<f64> {
        use rand_distr::{Distribution, Normal};

        let n = self.population.npar();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Generate standard normal vector
        let z: Vec<f64> = (0..n).map(|_| normal.sample(&mut self.rng)).collect();

        // Compute Cholesky of Ω
        let omega = self.population.omega();
        let chol = match cholesky_lower(omega) {
            Ok(l) => l,
            Err(_) => {
                // Fallback to diagonal
                let mut l = Mat::zeros(n, n);
                for i in 0..n {
                    l[(i, i)] = omega[(i, i)].sqrt().max(1e-6);
                }
                l
            }
        };

        // Transform: η = L * z
        let mut eta = Col::zeros(n);
        for i in 0..n {
            for j in 0..=i {
                eta[i] += chol[(i, j)] * z[j];
            }
        }
        eta
    }
}

// ============== Helper Functions ==============

/// Compute Cholesky decomposition (lower triangular)
fn cholesky_lower(mat: &Mat<f64>) -> Result<Mat<f64>> {
    let n = mat.nrows();
    let mut l = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = mat[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                if sum <= 0.0 {
                    bail!("Matrix not positive definite at index {}", i);
                }
                l[(i, j)] = sum.sqrt();
            } else {
                if l[(j, j)].abs() < 1e-10 {
                    bail!("Near-zero diagonal in Cholesky at index {}", j);
                }
                l[(i, j)] = sum / l[(j, j)];
            }
        }
    }

    Ok(l)
}

/// Invert a symmetric positive definite matrix
fn invert_symmetric(mat: &Mat<f64>) -> Result<Mat<f64>> {
    let n = mat.nrows();
    let l = cholesky_lower(mat)?;

    // Invert L
    let mut l_inv = Mat::zeros(n, n);
    for i in 0..n {
        l_inv[(i, i)] = 1.0 / l[(i, i)];
        for j in (i + 1)..n {
            let mut sum = 0.0;
            for k in i..j {
                sum -= l[(j, k)] * l_inv[(k, i)];
            }
            l_inv[(j, i)] = sum / l[(j, j)];
        }
    }

    // A⁻¹ = L⁻ᵀ L⁻¹
    let mut inv = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in i.max(j)..n {
                sum += l_inv[(k, i)] * l_inv[(k, j)];
            }
            inv[(i, j)] = sum;
            inv[(j, i)] = sum;
        }
    }

    Ok(inv)
}

/// Estimate initial residual error variance from data
fn estimate_initial_sigma_sq(settings: &Settings) -> f64 {
    // Use error model initial values if available
    let error_models = settings.errormodels();
    if let Some((_outeq, error_model)) = error_models.iter().next() {
        // Get error polynomial coefficients
        if let Ok(poly) = error_model.errorpoly() {
            // For combined error c0 + c1*f, use c0² as rough estimate of additive variance
            let c0 = poly.c0();
            (c0 * c0).max(0.01)
        } else {
            1.0
        }
    } else {
        1.0 // Default
    }
}

// Type alias for backward compatibility
pub type SAEM<E> = FSAEM<E>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::parametric::sufficient_stats::StepSizeSchedule;

    #[test]
    fn test_cholesky() {
        let mat = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.5 });

        let l = cholesky_lower(&mat).unwrap();

        // Verify L * Lᵀ = mat
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += l[(i, k)] * l[(j, k)];
                }
                assert!((sum - mat[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_invert_symmetric() {
        let mat = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.5 });

        let inv = invert_symmetric(&mat).unwrap();

        // Verify mat * inv = I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += mat[(i, k)] * inv[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
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
}
