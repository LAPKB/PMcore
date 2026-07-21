use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::estimation::parametric::{
    marginal_likelihood::MarginalLikelihoodConfig, residual::RESIDUAL_OPTIMIZER_MAX_SIGMA,
};

/// Lugsail batch-means parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LugsailConfig {
    pub r: usize,
    pub c: f64,
}

impl LugsailConfig {
    pub fn new(r: usize, c: f64) -> Self {
        Self { r, c }
    }

    /// High/extreme-MCMC over-lugsail convenience: Bartlett q=1, r=3, c=0.5.
    pub fn over_lugsail_bartlett() -> Self {
        Self::new(3, 0.5)
    }
}

/// Explicit post-fit frozen-kernel simulation-variance and rank-diagnostic budget.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MarkovSimulationVarianceConfig {
    pub seed: u64,
    pub warmup_transitions: usize,
    pub draws_per_chain: usize,
    pub batch_size: usize,
    pub lugsail: LugsailConfig,
    pub diagnostic_chains: usize,
    pub max_trace_bytes: usize,
}

impl MarkovSimulationVarianceConfig {
    pub fn new(
        seed: u64,
        warmup_transitions: usize,
        draws_per_chain: usize,
        batch_size: usize,
        lugsail: LugsailConfig,
        diagnostic_chains: usize,
        max_trace_bytes: usize,
    ) -> Self {
        Self {
            seed,
            warmup_transitions,
            draws_per_chain,
            batch_size,
            lugsail,
            diagnostic_chains,
            max_trace_bytes,
        }
    }
}

/// Caller-declared policy for detecting sustained covariance-boundary stalls.
///
/// The margin threshold is dimensionless and relative to the declared initial
/// covariance. No threshold is inferred from observed fit results.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CovarianceStabilityConfig {
    pub minimum_relative_spd_margin: f64,
    pub rejection_window: usize,
}

impl CovarianceStabilityConfig {
    pub fn new(minimum_relative_spd_margin: f64, rejection_window: usize) -> Self {
        Self {
            minimum_relative_spd_margin,
            rejection_window,
        }
    }
}

/// Explicit opt-in PMcore operational convergence policy.
///
/// This is an operational stopping rule, not proof of mathematical convergence,
/// stationarity, model correctness, or uncertainty. There is deliberately no
/// `Default`; schedule, precision, confidence, and stationarity thresholds are
/// caller choices.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationalConvergenceConfig {
    pub first_eligible_averaged_iteration: usize,
    pub check_interval: usize,
    pub max_rhat: f64,
    pub min_bulk_ess: f64,
    pub min_average_bulk_ess_per_split_chain: f64,
    pub relative_fixed_width_epsilon: f64,
    pub confidence_level: f64,
    /// PMcore operational-policy threshold; the literature supplies none.
    pub max_newton_displacement: f64,
    /// PMcore operational-policy threshold; the literature supplies none.
    pub max_newton_displacement_mc_sd: f64,
}

impl OperationalConvergenceConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        first_eligible_averaged_iteration: usize,
        check_interval: usize,
        max_rhat: f64,
        min_bulk_ess: f64,
        min_average_bulk_ess_per_split_chain: f64,
        relative_fixed_width_epsilon: f64,
        confidence_level: f64,
        max_newton_displacement: f64,
        max_newton_displacement_mc_sd: f64,
    ) -> Self {
        Self {
            first_eligible_averaged_iteration,
            check_interval,
            max_rhat,
            min_bulk_ess,
            min_average_bulk_ess_per_split_chain,
            relative_fixed_width_epsilon,
            confidence_level,
            max_newton_displacement,
            max_newton_displacement_mc_sd,
        }
    }

    /// Literature-guided Vehtari and Gong/Flegal policy with caller-supplied
    /// schedule, fixed-width, confidence, and PMcore stationarity thresholds.
    pub fn literature_guided(
        first_eligible_averaged_iteration: usize,
        check_interval: usize,
        relative_fixed_width_epsilon: f64,
        confidence_level: f64,
        max_newton_displacement: f64,
        max_newton_displacement_mc_sd: f64,
    ) -> Self {
        Self::new(
            first_eligible_averaged_iteration,
            check_interval,
            1.01,
            400.0,
            50.0,
            relative_fixed_width_epsilon,
            confidence_level,
            max_newton_displacement,
            max_newton_displacement_mc_sd,
        )
    }

    pub fn first_eligible_averaged_iteration(mut self, iteration: usize) -> Self {
        self.first_eligible_averaged_iteration = iteration;
        self
    }

    pub fn check_interval(mut self, interval: usize) -> Self {
        self.check_interval = interval;
        self
    }

    pub fn max_rhat(mut self, value: f64) -> Self {
        self.max_rhat = value;
        self
    }

    pub fn min_bulk_ess(mut self, value: f64) -> Self {
        self.min_bulk_ess = value;
        self
    }

    pub fn min_average_bulk_ess_per_split_chain(mut self, value: f64) -> Self {
        self.min_average_bulk_ess_per_split_chain = value;
        self
    }

    pub fn relative_fixed_width_epsilon(mut self, value: f64) -> Self {
        self.relative_fixed_width_epsilon = value;
        self
    }

    pub fn confidence_level(mut self, value: f64) -> Self {
        self.confidence_level = value;
        self
    }

    pub fn max_newton_displacement(mut self, value: f64) -> Self {
        self.max_newton_displacement = value;
        self
    }

    pub fn max_newton_displacement_mc_sd(mut self, value: f64) -> Self {
        self.max_newton_displacement_mc_sd = value;
        self
    }
}

/// Final-estimate policy for SAEM.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum SaemEstimatorPolicy {
    /// Use the final stochastic-approximation iterate.
    #[default]
    TerminalIterate,
    /// Use an unweighted average of completed smoothing-phase iterates.
    AveragedIterates { alpha: f64 },
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields, default)]
pub struct SaemConfig {
    /// Total pre-smoothing (K1) iterations, including `burn_in`.
    ///
    /// The exploration-phase count is `k1_iterations - burn_in`.
    pub k1_iterations: usize,
    /// Number of smoothing-phase (K2) iterations.
    pub k2_iterations: usize,
    /// Number of burn-in iterations within the total K1 period.
    pub burn_in: usize,
    /// Residual variance-floor protection duration. Zero uses the runtime's
    /// automatic K1/2 duration; a positive value sets the duration explicitly.
    pub sa_iterations: usize,
    pub sa_cooling_factor: f64,
    pub rw_init: f64,
    pub n_chains: usize,
    pub mcmc_iterations: usize,
    pub eta_block_iterations: usize,
    pub adapt_interval: usize,
    /// Maximum early covariance stabilization fraction. For covariate IIV,
    /// this under-relaxes the accepted exploration Ω/GEM displacement. For
    /// centered non-covariate IIV and IOV statistics, it limits their early
    /// stochastic-approximation updates.
    pub omega_sa_max_step: f64,
    /// Minimum accepted estimated IIV variance. Every estimated initial Ω
    /// diagonal must already be at least this floor; fixed diagonals are exempt.
    pub omega_min_variance: f64,
    /// Minimum accepted estimated IOV variance. Every estimated initial Ω_IOV
    /// diagonal must already be at least this floor; fixed diagonals are exempt.
    pub omega_iov_min_variance: f64,
    pub residual_min_sigma: f64,
    pub residual_optimizer_max_iterations: usize,
    pub compute_map: bool,
    pub map_max_iterations: usize,
    pub map_sd_tolerance: f64,
    pub map_initial_step: f64,
    pub seed: u64,
    /// Final-estimate and smoothing-gain policy.
    pub estimator_policy: SaemEstimatorPolicy,
    /// Optional post-fit frozen-kernel Markov simulation-variance diagnostic.
    pub markov_simulation_variance: Option<MarkovSimulationVarianceConfig>,
    /// Optional caller-declared covariance-boundary diagnostic policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub covariance_stability: Option<CovarianceStabilityConfig>,
    /// Optional explicit operational convergence policy. A joint passing
    /// checkpoint may terminate with `Converged`; disabled, failed, or
    /// ineligible finite schedules retain `MaxCycles`. Operational convergence
    /// requires an explicit covariance-stability policy.
    pub operational_convergence: Option<OperationalConvergenceConfig>,
    /// Optional explicit post-fit population marginal-likelihood calculation.
    pub marginal_likelihood: Option<MarginalLikelihoodConfig>,
}

impl Default for SaemConfig {
    fn default() -> Self {
        Self {
            k1_iterations: 300,
            k2_iterations: 100,
            burn_in: 5,
            sa_iterations: 0,
            sa_cooling_factor: 0.97,
            rw_init: 0.5,
            n_chains: 1,
            mcmc_iterations: 1,
            // Disabled by default; opt in to the block-mixture kernel.
            eta_block_iterations: 0,
            adapt_interval: 50,
            // Guard against one-draw correlated Ω collapse in exploration.
            omega_sa_max_step: 0.1,
            omega_min_variance: 1e-6,
            omega_iov_min_variance: 1e-8,
            // Uses the established 1e-12 residual-variance guard on the SD scale.
            residual_min_sigma: 1e-6,
            residual_optimizer_max_iterations: 200,
            compute_map: true,
            map_max_iterations: 200,
            map_sd_tolerance: 1e-8,
            map_initial_step: 0.1,
            seed: 123456,
            estimator_policy: SaemEstimatorPolicy::TerminalIterate,
            markov_simulation_variance: None,
            covariance_stability: None,
            operational_convergence: None,
            marginal_likelihood: None,
        }
    }
}

impl SaemConfig {
    /// Creates a new `SaemConfig` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total pre-smoothing (K1) iterations, including burn-in.
    ///
    /// The exploration-phase count is `iterations - burn_in`.
    pub fn k1_iterations(mut self, iterations: usize) -> Self {
        self.k1_iterations = iterations;
        self
    }

    /// Number of smoothing-phase (K2) iterations.
    pub fn k2_iterations(mut self, iterations: usize) -> Self {
        self.k2_iterations = iterations;
        self
    }

    /// Number of burn-in iterations within the total K1 period.
    pub fn burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Number of MCMC chains.
    pub fn n_chains(mut self, n_chains: usize) -> Self {
        self.n_chains = n_chains;
        self
    }

    /// Initial random-walk scale multiplier.
    pub fn mcmc_step_size(mut self, step_size: f64) -> Self {
        self.rw_init = step_size;
        self
    }

    /// Number of MCMC proposal sweeps per E-step.
    pub fn mcmc_iterations(mut self, iterations: usize) -> Self {
        self.mcmc_iterations = iterations;
        self
    }

    /// Number of Ω-scaled η block proposals per subject-chain and E-step.
    ///
    /// Each proposal precedes the componentwise decorrelation sweep and uses
    /// `eta' = eta + scale_subject * chol(Omega) * z`, matching the reference
    /// block kernel. The default is zero, which disables the block kernel.
    pub fn eta_block_iterations(mut self, iterations: usize) -> Self {
        self.eta_block_iterations = iterations;
        self
    }

    /// Number of E-steps between proposal-scale adaptations.
    pub fn adapt_interval(mut self, iterations: usize) -> Self {
        self.adapt_interval = iterations;
        self
    }

    /// Maximum early covariance stabilization fraction.
    ///
    /// Covariate IIV uses this as an exploration-only cap on the accepted
    /// mask-aware Ω/GEM displacement, after constructing coherent raw moments.
    /// Centered non-covariate IIV and IOV statistics retain their existing
    /// burn-in/exploration stochastic-approximation interpretation.
    pub fn omega_sa_max_step(mut self, step_size: f64) -> Self {
        self.omega_sa_max_step = step_size;
        self
    }

    /// Minimum accepted estimated IOV variance.
    ///
    /// Every estimated initial Ω_IOV diagonal must already be at least this
    /// value. Fixed diagonals are exempt.
    pub fn omega_iov_min_variance(mut self, variance: f64) -> Self {
        self.omega_iov_min_variance = variance;
        self
    }

    /// Minimum estimated residual standard deviation.
    pub fn residual_min_sigma(mut self, sigma: f64) -> Self {
        self.residual_min_sigma = sigma;
        self
    }

    /// Maximum iterations for each joint residual-parameter optimization.
    pub fn residual_optimizer_max_iterations(mut self, iterations: usize) -> Self {
        self.residual_optimizer_max_iterations = iterations;
        self
    }

    /// Enable or disable posthoc conditional-mode estimation.
    pub fn compute_map(mut self, compute: bool) -> Self {
        self.compute_map = compute;
        self
    }

    /// Maximum Nelder-Mead iterations for each subject's posthoc mode.
    pub fn map_max_iterations(mut self, iterations: usize) -> Self {
        self.map_max_iterations = iterations;
        self
    }

    /// Simplex objective-standard-deviation tolerance for posthoc modes.
    pub fn map_sd_tolerance(mut self, tolerance: f64) -> Self {
        self.map_sd_tolerance = tolerance;
        self
    }

    /// Initial simplex displacement as a fraction of each random-effect SD.
    pub fn map_initial_step(mut self, step: f64) -> Self {
        self.map_initial_step = step;
        self
    }

    /// Random-number-generator seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Select the final-estimate policy directly.
    pub fn estimator_policy(mut self, policy: SaemEstimatorPolicy) -> Self {
        self.estimator_policy = policy;
        self
    }

    /// Use smoothing gain `s^-alpha` and the Cesaro average of smoothing iterates.
    pub fn averaged_iterates(self, alpha: f64) -> Self {
        self.estimator_policy(SaemEstimatorPolicy::AveragedIterates { alpha })
    }

    /// Enable the explicitly-budgeted post-fit frozen-kernel diagnostic.
    pub fn markov_simulation_variance(mut self, config: MarkovSimulationVarianceConfig) -> Self {
        self.markov_simulation_variance = Some(config);
        self
    }

    /// Enable caller-declared covariance-boundary and rejection diagnostics.
    pub fn covariance_stability(mut self, policy: CovarianceStabilityConfig) -> Self {
        self.covariance_stability = Some(policy);
        self
    }

    /// Enable explicit operational checkpoints and joint stopping criteria.
    pub fn operational_convergence(mut self, criteria: OperationalConvergenceConfig) -> Self {
        self.operational_convergence = Some(criteria);
        self
    }

    /// Request the explicit post-fit population marginal likelihood.
    pub fn marginal_likelihood(mut self, config: MarginalLikelihoodConfig) -> Self {
        self.marginal_likelihood = Some(config);
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let total_iterations = self
            .k1_iterations
            .checked_add(self.k2_iterations)
            .ok_or_else(|| {
                anyhow::anyhow!("SAEM k1_iterations + k2_iterations must not overflow")
            })?;
        if total_iterations == 0 {
            anyhow::bail!("SAEM k1_iterations + k2_iterations must be greater than zero");
        }
        if self.burn_in > self.k1_iterations {
            anyhow::bail!("SAEM burn_in must not exceed k1_iterations");
        }
        if let Some(policy) = self.covariance_stability {
            if !policy.minimum_relative_spd_margin.is_finite()
                || policy.minimum_relative_spd_margin <= 0.0
                || policy.minimum_relative_spd_margin >= 1.0
            {
                anyhow::bail!(
                    "SAEM covariance-stability minimum relative SPD margin must be finite and in (0, 1)"
                );
            }
            if policy.rejection_window == 0 {
                anyhow::bail!("SAEM covariance-stability rejection window must be positive");
            }
            let active_cycles = total_iterations - self.burn_in;
            if policy.rejection_window > active_cycles {
                anyhow::bail!(
                    "SAEM covariance-stability rejection window must be reachable during covariance-active cycles"
                );
            }
        }
        if let SaemEstimatorPolicy::AveragedIterates { alpha } = self.estimator_policy {
            if !alpha.is_finite() || alpha <= 0.5 || alpha >= 1.0 {
                anyhow::bail!(
                    "SAEM averaged-iterate alpha must be finite and strictly between 0.5 and 1.0"
                );
            }
            if self.k2_iterations == 0 {
                anyhow::bail!("SAEM averaged iterates require k2_iterations greater than zero");
            }
        }
        if let Some(diagnostic) = self.markov_simulation_variance {
            if !matches!(
                self.estimator_policy,
                SaemEstimatorPolicy::AveragedIterates { .. }
            ) {
                anyhow::bail!("SAEM Markov simulation variance requires averaged iterates");
            }
            if self.k2_iterations == 0 {
                anyhow::bail!(
                    "SAEM Markov simulation variance requires a completed-capable K2 phase"
                );
            }
            let samples = diagnostic.draws_per_chain;
            let batch = diagnostic.batch_size;
            let r = diagnostic.lugsail.r;
            let c = diagnostic.lugsail.c;
            if samples == 0 || batch == 0 || samples % batch != 0 || samples / batch < 2 {
                anyhow::bail!(
                    "SAEM Markov simulation variance draws must form at least two complete batches"
                );
            }
            if r == 0 || batch % r != 0 || batch / r == 0 {
                anyhow::bail!("SAEM Markov simulation variance lugsail r must divide batch size with b/r >= 1");
            }
            if !c.is_finite() || !(0.0..1.0).contains(&c) {
                anyhow::bail!(
                    "SAEM Markov simulation variance lugsail c must be finite and in [0, 1)"
                );
            }
            if diagnostic.diagnostic_chains == 0 {
                anyhow::bail!(
                    "SAEM Markov simulation variance diagnostic_chains must be at least 1"
                );
            }
            if diagnostic.max_trace_bytes == 0 {
                anyhow::bail!(
                    "SAEM Markov simulation variance max_trace_bytes must be greater than zero"
                );
            }
        }
        if let Some(policy) = self.operational_convergence {
            let _covariance_stability = self.covariance_stability.ok_or_else(|| {
                anyhow::anyhow!(
                    "SAEM operational convergence requires an explicit covariance-stability policy"
                )
            })?;
            if !matches!(
                self.estimator_policy,
                SaemEstimatorPolicy::AveragedIterates { .. }
            ) {
                anyhow::bail!("SAEM operational convergence requires averaged iterates");
            }
            if policy.first_eligible_averaged_iteration == 0
                || policy.first_eligible_averaged_iteration > self.k2_iterations
            {
                anyhow::bail!("SAEM operational convergence first eligible averaged iteration must be reachable in K2");
            }
            if policy.check_interval == 0 {
                anyhow::bail!("SAEM operational convergence check interval must be positive");
            }
            if !policy.max_rhat.is_finite() || policy.max_rhat <= 1.0 {
                anyhow::bail!(
                    "SAEM operational convergence max Rhat must be finite and greater than 1"
                );
            }
            for (name, value) in [
                ("min bulk ESS", policy.min_bulk_ess),
                (
                    "min average bulk ESS per split chain",
                    policy.min_average_bulk_ess_per_split_chain,
                ),
                (
                    "relative fixed-width epsilon",
                    policy.relative_fixed_width_epsilon,
                ),
                ("max Newton displacement", policy.max_newton_displacement),
                (
                    "max Newton-displacement MC SD",
                    policy.max_newton_displacement_mc_sd,
                ),
            ] {
                if !value.is_finite() || value <= 0.0 {
                    anyhow::bail!(
                        "SAEM operational convergence {name} must be finite and positive"
                    );
                }
            }
            if !policy.confidence_level.is_finite()
                || !(0.0..1.0).contains(&policy.confidence_level)
            {
                anyhow::bail!(
                    "SAEM operational convergence confidence level must be finite and in (0, 1)"
                );
            }
            let normal = statrs::distribution::Normal::new(0.0, 1.0)
                .expect("standard normal parameters are valid");
            let one_sided = policy.confidence_level + (1.0 - policy.confidence_level) / 2.0;
            let z = statrs::distribution::ContinuousCDF::inverse_cdf(&normal, one_sided);
            if !z.is_finite() || z <= 0.0 {
                anyhow::bail!(
                    "SAEM operational convergence confidence level produces an unusable normal quantile"
                );
            }
            let implied_minimum_ess = 4.0 * z * z / policy.relative_fixed_width_epsilon.powi(2);
            if !implied_minimum_ess.is_finite() || implied_minimum_ess <= 0.0 {
                anyhow::bail!(
                    "SAEM operational convergence confidence/epsilon produce an unusable implied minimum ESS"
                );
            }
            let diagnostic = self.markov_simulation_variance.ok_or_else(|| {
                anyhow::anyhow!("SAEM operational convergence requires Markov diagnostics")
            })?;
            if diagnostic.diagnostic_chains < 4 {
                anyhow::bail!("SAEM operational convergence requires at least 4 diagnostic chains");
            }
            if diagnostic.draws_per_chain % 2 != 0 {
                anyhow::bail!("SAEM operational convergence requires an even retained draw count");
            }
            if diagnostic.lugsail != LugsailConfig::over_lugsail_bartlett() {
                anyhow::bail!(
                    "SAEM operational termination requires over-lugsail Bartlett r=3,c=0.5"
                );
            }
        }
        if let Some(config) = self.marginal_likelihood {
            config.validate()?;
        }
        if !self.rw_init.is_finite() || self.rw_init <= 0.0 {
            anyhow::bail!("SAEM rw_init must be finite and positive");
        }
        if self.n_chains == 0 {
            anyhow::bail!("SAEM n_chains must be greater than zero");
        }
        if self.mcmc_iterations == 0 {
            anyhow::bail!("SAEM mcmc_iterations must be greater than zero");
        }
        if self.adapt_interval == 0 {
            anyhow::bail!("SAEM adapt_interval must be greater than zero");
        }
        if !self.sa_cooling_factor.is_finite()
            || self.sa_cooling_factor <= 0.0
            || self.sa_cooling_factor > 1.0
        {
            anyhow::bail!("SAEM sa_cooling_factor must be finite and in (0, 1]");
        }
        if !self.omega_sa_max_step.is_finite()
            || self.omega_sa_max_step <= 0.0
            || self.omega_sa_max_step > 1.0
        {
            anyhow::bail!("SAEM omega_sa_max_step must be finite and in (0, 1]");
        }
        for (name, variance) in [
            ("omega_min_variance", self.omega_min_variance),
            ("omega_iov_min_variance", self.omega_iov_min_variance),
        ] {
            if !variance.is_finite() || variance <= 0.0 {
                anyhow::bail!("SAEM {name} must be finite and positive");
            }
        }
        if !self.residual_min_sigma.is_finite()
            || self.residual_min_sigma <= 0.0
            || self.residual_min_sigma >= RESIDUAL_OPTIMIZER_MAX_SIGMA
        {
            anyhow::bail!(
                "SAEM residual_min_sigma must be finite, positive, and below {RESIDUAL_OPTIMIZER_MAX_SIGMA}"
            );
        }
        if self.residual_optimizer_max_iterations == 0 {
            anyhow::bail!("SAEM residual_optimizer_max_iterations must be greater than zero");
        }
        if self.compute_map {
            if self.map_max_iterations == 0 {
                anyhow::bail!("SAEM map_max_iterations must be greater than zero");
            }
            if !self.map_sd_tolerance.is_finite() || self.map_sd_tolerance <= 0.0 {
                anyhow::bail!("SAEM map_sd_tolerance must be finite and positive");
            }
            if !self.map_initial_step.is_finite() || self.map_initial_step <= 0.0 {
                anyhow::bail!("SAEM map_initial_step must be finite and positive");
            }
        }
        Ok(())
    }

    pub fn total_iterations(&self) -> usize {
        self.k1_iterations + self.k2_iterations
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CovarianceStabilityConfig, LugsailConfig, MarkovSimulationVarianceConfig,
        OperationalConvergenceConfig, SaemConfig, SaemEstimatorPolicy,
        RESIDUAL_OPTIMIZER_MAX_SIGMA,
    };
    use crate::estimation::MarginalLikelihoodConfig;

    #[test]
    fn mcmc_step_size_sets_rw_init() {
        let config = SaemConfig::new().mcmc_step_size(0.75);
        assert_eq!(config.rw_init, 0.75);
    }

    #[test]
    fn eta_block_kernel_is_opt_in_and_serialized_configuration_is_operational() {
        assert_eq!(SaemConfig::default().eta_block_iterations, 0);
        let config = SaemConfig::new().eta_block_iterations(2);
        assert_eq!(config.eta_block_iterations, 2);
        let decoded: SaemConfig = serde_json::from_str(r#"{"eta_block_iterations":3}"#).unwrap();
        assert_eq!(decoded.eta_block_iterations, 3);
    }

    #[test]
    fn invalid_operational_values_fail_closed() {
        let invalid = [
            (
                "empty schedule",
                SaemConfig::new().k1_iterations(0).k2_iterations(0),
            ),
            (
                "schedule overflow",
                SaemConfig {
                    k1_iterations: usize::MAX,
                    k2_iterations: 1,
                    ..SaemConfig::default()
                },
            ),
            ("burn-in", SaemConfig::new().k1_iterations(1).burn_in(2)),
            (
                "random-walk scale",
                SaemConfig::new().mcmc_step_size(f64::NAN),
            ),
            ("chains", SaemConfig::new().n_chains(0)),
            ("MCMC iterations", SaemConfig::new().mcmc_iterations(0)),
            ("adaptation interval", SaemConfig::new().adapt_interval(0)),
            (
                "SA cooling",
                SaemConfig {
                    sa_cooling_factor: 0.0,
                    ..SaemConfig::default()
                },
            ),
            ("omega step", SaemConfig::new().omega_sa_max_step(1.01)),
            (
                "omega floor",
                SaemConfig {
                    omega_min_variance: 0.0,
                    ..SaemConfig::default()
                },
            ),
            (
                "IOV omega floor",
                SaemConfig::new().omega_iov_min_variance(f64::INFINITY),
            ),
            ("residual floor", SaemConfig::new().residual_min_sigma(-1.0)),
            (
                "residual optimizer bound",
                SaemConfig::new().residual_min_sigma(RESIDUAL_OPTIMIZER_MAX_SIGMA),
            ),
            (
                "residual optimizer",
                SaemConfig::new().residual_optimizer_max_iterations(0),
            ),
            ("MAP iterations", SaemConfig::new().map_max_iterations(0)),
            (
                "MAP tolerance",
                SaemConfig::new().map_sd_tolerance(f64::NAN),
            ),
            ("MAP step", SaemConfig::new().map_initial_step(0.0)),
        ];

        for (case, config) in invalid {
            assert!(config.validate().is_err(), "{case} must fail validation");
        }

        assert!(SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(0)
            .burn_in(0)
            .eta_block_iterations(0)
            .validate()
            .is_ok());
        assert!(SaemConfig::new()
            .compute_map(false)
            .map_max_iterations(0)
            .map_sd_tolerance(f64::NAN)
            .map_initial_step(0.0)
            .validate()
            .is_ok());
    }

    #[test]
    fn covariance_stability_is_explicit_and_invalid_policies_fail_closed() {
        assert!(SaemConfig::default().covariance_stability.is_none());
        let policy = CovarianceStabilityConfig::new(0.01, 2);
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(1)
            .burn_in(0)
            .covariance_stability(policy);
        assert!(config.validate().is_ok());
        let encoded = serde_json::to_string(&config).unwrap();
        assert!(encoded.contains("covariance_stability"));
        assert_eq!(
            serde_json::from_str::<SaemConfig>(&encoded)
                .unwrap()
                .covariance_stability,
            Some(policy)
        );
        for invalid in [
            CovarianceStabilityConfig::new(0.0, 1),
            CovarianceStabilityConfig::new(1.0, 1),
            CovarianceStabilityConfig::new(f64::NAN, 1),
            CovarianceStabilityConfig::new(0.01, 0),
            CovarianceStabilityConfig::new(0.01, 3),
        ] {
            assert!(SaemConfig::new()
                .k1_iterations(1)
                .k2_iterations(1)
                .burn_in(0)
                .covariance_stability(invalid)
                .validate()
                .is_err());
        }
    }

    #[test]
    fn marginal_likelihood_config_is_explicit_and_invalid_values_fail_closed() {
        assert!(SaemConfig::default().marginal_likelihood.is_none());
        for config in [
            MarginalLikelihoodConfig::new(1, 1, 3, 1.0),
            MarginalLikelihoodConfig::new(2, 1, 2, 1.0),
            MarginalLikelihoodConfig::new(2, 1, 3, 0.0),
            MarginalLikelihoodConfig::new(2, 1, 3, f64::NAN),
        ] {
            assert!(SaemConfig::new()
                .marginal_likelihood(config)
                .validate()
                .is_err());
        }
        let valid = MarginalLikelihoodConfig::new(2, 9, 3, 0.5);
        assert!(SaemConfig::new()
            .marginal_likelihood(valid)
            .validate()
            .is_ok());
    }

    #[test]
    fn estimator_policy_defaults_and_validates_alpha_schedule() {
        assert_eq!(
            SaemConfig::default().estimator_policy,
            SaemEstimatorPolicy::TerminalIterate
        );
        for alpha in [0.5, 1.0, f64::NAN, f64::INFINITY] {
            assert!(SaemConfig::new()
                .averaged_iterates(alpha)
                .validate()
                .is_err());
        }
        assert!(SaemConfig::new()
            .averaged_iterates(0.75)
            .k2_iterations(0)
            .validate()
            .is_err());
        assert!(SaemConfig::new().averaged_iterates(0.75).validate().is_ok());
    }

    #[test]
    fn markov_variance_config_is_explicit_and_all_invalid_shapes_fail() {
        assert_eq!(
            LugsailConfig::over_lugsail_bartlett(),
            LugsailConfig::new(3, 0.5)
        );
        let valid = MarkovSimulationVarianceConfig::new(
            7,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            2,
            1024,
        );
        assert!(SaemConfig::new()
            .averaged_iterates(0.75)
            .markov_simulation_variance(valid)
            .validate()
            .is_ok());
        assert!(SaemConfig::new()
            .markov_simulation_variance(valid)
            .validate()
            .is_err());
        for invalid in [
            MarkovSimulationVarianceConfig::new(1, 0, 0, 6, LugsailConfig::new(3, 0.5), 2, 1024),
            MarkovSimulationVarianceConfig::new(1, 0, 10, 6, LugsailConfig::new(3, 0.5), 2, 1024),
            MarkovSimulationVarianceConfig::new(1, 0, 6, 6, LugsailConfig::new(3, 0.5), 2, 1024),
            MarkovSimulationVarianceConfig::new(1, 0, 12, 0, LugsailConfig::new(3, 0.5), 2, 1024),
            MarkovSimulationVarianceConfig::new(1, 0, 12, 6, LugsailConfig::new(0, 0.5), 2, 1024),
            MarkovSimulationVarianceConfig::new(1, 0, 12, 6, LugsailConfig::new(4, 0.5), 2, 1024),
            MarkovSimulationVarianceConfig::new(1, 0, 12, 6, LugsailConfig::new(3, -0.1), 2, 1024),
            MarkovSimulationVarianceConfig::new(1, 0, 12, 6, LugsailConfig::new(3, 1.0), 2, 1024),
            MarkovSimulationVarianceConfig::new(
                1,
                0,
                12,
                6,
                LugsailConfig::new(3, f64::NAN),
                2,
                1024,
            ),
            MarkovSimulationVarianceConfig::new(1, 0, 12, 6, LugsailConfig::new(3, 0.5), 0, 1024),
            MarkovSimulationVarianceConfig::new(1, 0, 12, 6, LugsailConfig::new(3, 0.5), 2, 0),
        ] {
            assert!(SaemConfig::new()
                .averaged_iterates(0.75)
                .markov_simulation_variance(invalid)
                .validate()
                .is_err());
        }
    }

    fn operational_policy() -> OperationalConvergenceConfig {
        OperationalConvergenceConfig::literature_guided(2, 3, 0.05, 0.95, 0.1, 0.02)
    }

    #[test]
    fn operational_convergence_literature_guided_fixes_only_vehtari_values() {
        let policy = operational_policy();
        assert_eq!(policy.max_rhat, 1.01);
        assert_eq!(policy.min_bulk_ess, 400.0);
        assert_eq!(policy.min_average_bulk_ess_per_split_chain, 50.0);
        assert_eq!(policy.relative_fixed_width_epsilon, 0.05);
        assert_eq!(policy.confidence_level, 0.95);
    }

    #[test]
    fn operational_convergence_serialization_has_no_policy_defaults() {
        assert_eq!(SaemConfig::default().operational_convergence, None);
        let policy = operational_policy().check_interval(4).max_rhat(1.02);
        let config = SaemConfig::new()
            .covariance_stability(CovarianceStabilityConfig::new(0.01, 2))
            .operational_convergence(policy);
        let decoded: SaemConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(decoded.operational_convergence, Some(policy));
        assert!(
            serde_json::from_str::<OperationalConvergenceConfig>(r#"{"max_rhat":1.01}"#).is_err()
        );
    }

    #[test]
    fn operational_convergence_validation_requires_complete_eligible_policy() {
        let markov = MarkovSimulationVarianceConfig::new(
            7,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            4,
            1024,
        );
        let base = || {
            SaemConfig::new()
                .k2_iterations(10)
                .averaged_iterates(0.75)
                .markov_simulation_variance(markov)
                .covariance_stability(CovarianceStabilityConfig::new(0.01, 2))
        };
        assert!(base()
            .operational_convergence(operational_policy())
            .validate()
            .is_ok());
        // The five-cycle window is not available at the first K2 checkpoint,
        // but it is reachable later in the declared eleven active cycles.
        assert!(SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(10)
            .burn_in(0)
            .averaged_iterates(0.75)
            .markov_simulation_variance(markov)
            .covariance_stability(CovarianceStabilityConfig::new(0.01, 5))
            .operational_convergence(operational_policy())
            .validate()
            .is_ok());
        assert!(SaemConfig::new()
            .averaged_iterates(0.75)
            .operational_convergence(operational_policy())
            .validate()
            .is_err());
        assert!(SaemConfig::new()
            .k2_iterations(10)
            .averaged_iterates(0.75)
            .markov_simulation_variance(markov)
            .operational_convergence(operational_policy())
            .validate()
            .is_err());
        for invalid in [
            operational_policy().first_eligible_averaged_iteration(0),
            operational_policy().first_eligible_averaged_iteration(11),
            operational_policy().check_interval(0),
            operational_policy().max_rhat(1.0),
            operational_policy().min_bulk_ess(0.0),
            operational_policy().min_average_bulk_ess_per_split_chain(0.0),
            operational_policy().relative_fixed_width_epsilon(0.0),
            operational_policy().relative_fixed_width_epsilon(f64::MIN_POSITIVE),
            operational_policy().confidence_level(1.0),
            operational_policy().confidence_level(f64::from_bits(1.0_f64.to_bits() - 1)),
            operational_policy().max_newton_displacement(f64::NAN),
            operational_policy().max_newton_displacement_mc_sd(0.0),
        ] {
            assert!(base().operational_convergence(invalid).validate().is_err());
        }
    }

    #[test]
    fn removed_unwired_options_fail_closed_during_deserialization() {
        for json in [
            r#"{"compute_fim":true}"#,
            r#"{"compute_ll_is":true}"#,
            r#"{"compute_ll_gq":true}"#,
            r#"{"use_gibbs":true}"#,
            r#"{"n_kernels":4}"#,
            r#"{"transform_par":[1,1]}"#,
            r#"{"fix_seed":false}"#,
        ] {
            let error = serde_json::from_str::<SaemConfig>(json)
                .expect_err("unwired SAEM option must not deserialize silently");
            assert!(error.to_string().contains("unknown field"));
        }
    }
}
