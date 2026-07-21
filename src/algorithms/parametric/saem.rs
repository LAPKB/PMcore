use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use argmin::{
    core::{CostFunction, Error as ArgminError, Executor},
    solver::neldermead::NelderMead,
};
use ndarray::Array2;
use pharmsol::{Data, Equation, Event, Subject};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::algorithms::{Status, StopReason};
use crate::estimation::likelihood::batch::{
    parametric_occasion_log_likelihood, parametric_subject_log_likelihood,
};
use crate::estimation::likelihood::objective::parametric_subject_log_likelihoods;
use crate::estimation::parametric::conditional_uncertainty::{
    conditional_mode_curvature, ConditionalModeMetadata, JointLatentCoordinate,
    JointLatentCoordinateKind,
};
use crate::estimation::parametric::covariance::{
    cholesky_lower, relative_spd_margin, worst_contrast,
};
use crate::estimation::parametric::covariates::{
    rebase_eta, solve_covariate_gls, subject_centered_omega, CovariateGlsProblem, CovariateModel,
};
use crate::estimation::parametric::individual::{
    individual_phi, individual_phi_from_subject_mean, individual_psi,
    individual_psi_from_subject_mean, occasion_psi, occasion_psi_from_subject_mean, population_phi,
    population_psi,
};
use crate::estimation::parametric::information::{
    derive_population_uncertainty, CompleteDerivative, InformationLayout, InformationRecursion,
};
use crate::estimation::parametric::marginal_likelihood::{
    calculate_population_marginal_likelihood, unavailable_population_marginal_likelihood,
    MarginalLikelihoodDiagnostics, MarginalLikelihoodFailureReason, MarginalLikelihoodStatus,
    MarginalSubject,
};
use crate::estimation::parametric::markov_variance::{
    classify_psd, lugsail_batch_means, rows, scale_lrv_sum, transform_simulation_variance,
    MatrixClassification,
};
use crate::estimation::parametric::posterior::{
    eta_log_prior_from_omega, eta_log_priors, SubjectPosteriorScore,
};
use crate::estimation::parametric::posthoc::optimize_conditional_mode;
use crate::estimation::parametric::prior::CovarianceUpdateResult;
use crate::estimation::parametric::rank_diagnostics::{
    bulk_ess, folded_split_rhat, rank_normalized_split_rhat, RankDiagnosticError,
};
use crate::estimation::parametric::residual::{
    combined_additive_sigma_collapsed, optimize_combined_residual,
    optimize_correlated_combined_residual, primary_sigma_parameter, primary_sigma_parameters,
    residual_statistics_for_subject, update_estimated_combined_residual_model,
    update_estimated_correlated_combined_residual_model,
    update_estimated_simple_residual_model_with_sigma, ResidualSufficientStatistics,
};
use crate::estimation::parametric::shrinkage::{
    derive_eta_map_shrinkage, derive_eta_posterior_mean_shrinkage, derive_kappa_map_shrinkage,
    derive_kappa_posterior_mean_shrinkage, ShrinkageDiagnostics,
};
use crate::estimation::parametric::sufficient::{
    CovariateSufficientStatistics, PhiSufficientStatistics,
};
use crate::estimation::parametric::{CovarianceUpdateStatus, ResolvedOmega};
use crate::estimation::{EstimationProblem, Parametric, ParametricErrorModels};
use crate::model::{ParameterScale, UnboundedParameter};
use crate::ResidualErrorModel;

use crate::results::{
    derive_information_criteria, CovarianceCycleUpdateDiagnostics, CovarianceCycleUpdateOutcome,
    CovarianceUpdateNotAttemptedReason, DiagnosticTraceCoordinate, InformationCoordinateKind,
    InformationDiagnostics, InformationStatus, MarkovSimulationVarianceChainDiagnostics,
    MarkovSimulationVarianceDiagnostics, MarkovSimulationVarianceStatus, OccasionKappaEstimate,
    OperationalConvergenceCheck, OperationalConvergenceCriterion,
    OperationalConvergenceCriterionStatus, OperationalConvergenceDiagnostics,
    OperationalConvergenceOutcome, ParametricResult, ParametricWarning, RankDiagnosticStatus,
    RankMixingDiagnostic, RankMixingDiagnostics, ResidualCycleDiagnostics, ResidualErrorEstimate,
    SaemCycleDiagnostics, SaemEstimatorMetadata, SaemPhase, SubjectConditionalMode,
    SubjectEtaEstimate,
};

use super::{
    CovarianceStabilityConfig, NumericalFailure, NumericalFailurePhase,
    OperationalConvergenceConfig, ParametricRunner, SaemConfig, SaemEstimatorPolicy,
};

fn pending_covariance_update_diagnostics(
    phase: SaemPhase,
    configured: bool,
    has_estimated_entries: bool,
) -> CovarianceCycleUpdateDiagnostics {
    let reason = if !configured {
        CovarianceUpdateNotAttemptedReason::NotConfigured
    } else if !has_estimated_entries {
        CovarianceUpdateNotAttemptedReason::NoEstimatedEntries
    } else if phase == SaemPhase::BurnIn {
        CovarianceUpdateNotAttemptedReason::BurnIn
    } else {
        CovarianceUpdateNotAttemptedReason::UpdateInactive
    };
    CovarianceCycleUpdateDiagnostics::not_attempted(reason)
}

fn completed_covariance_update_diagnostics(
    proposal: &Array2<f64>,
    update: &CovarianceUpdateResult,
) -> Result<CovarianceCycleUpdateDiagnostics> {
    let outcome = match update.status {
        CovarianceUpdateStatus::Accepted => CovarianceCycleUpdateOutcome::Accepted,
        CovarianceUpdateStatus::NoOp => CovarianceCycleUpdateOutcome::NoOp,
        CovarianceUpdateStatus::Rejected => CovarianceCycleUpdateOutcome::Rejected {
            reason: update.rejection_reason.ok_or_else(|| {
                anyhow!("rejected covariance update lacks a typed diagnostic reason")
            })?,
        },
    };
    Ok(CovarianceCycleUpdateDiagnostics {
        proposal: Some(proposal.clone()),
        solved_target: update.solved_target.clone(),
        outcome,
        accepted_fraction: update.accepted_fraction,
        attempted_fractions: update.attempted_fractions.clone(),
        trial_rejections: update.trial_rejections.clone(),
    })
}

const COMPONENT_TARGET_ACCEPTANCE: f64 = 0.44;
const ETA_BLOCK_TARGET_ACCEPTANCE: f64 = 0.40;
const KAPPA_BLOCK_TARGET_ACCEPTANCE: f64 = 0.40;
const PROPOSAL_SCALE_INCREASE: f64 = 1.1;
const MARKOV_VARIANCE_ASSUMPTIONS: &str = concat!(
    "diagnostic only: prior draws at frozen averaged Omega/Omega_IOV; ",
    "per-chain seed = config.seed.wrapping_add(i).wrapping_mul(0x9E3779B97F4A7C15); ",
    "frozen-kernel stationarity, adequate mixing, the Poisson equation, and the ",
    "controlled-Markov averaged-SA CLT are unverified; lugsail batch means alone is not a ",
    "mixing diagnostic; failure detection (non-finite, ",
    "constant, stuck, byte overflow, non-positive tau) is not a convergence claim; ",
    "literature recommendations for R̂ and ESS are referenced but no threshold "
);

#[derive(Clone)]
struct FrozenDiagnosticState {
    etas: Vec<Vec<Vec<f64>>>,
    kappas: Vec<Vec<Vec<Vec<f64>>>>,
}

struct DiagnosticCandidate {
    population_parameters: Vec<f64>,
    covariate_model: Option<CovariateModel>,
    omega: Array2<f64>,
    omega_iov: Option<Array2<f64>>,
    error_models: ParametricErrorModels,
}

#[derive(Debug, Clone)]
struct NonIivCoordinateLayout {
    population_indices: Vec<usize>,
    covariate_indices: Vec<usize>,
}

impl NonIivCoordinateLayout {
    fn len(&self) -> usize {
        self.population_indices.len() + self.covariate_indices.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

type NonIivCandidateComponents = (Vec<f64>, Option<CovariateModel>, Option<Vec<Vec<f64>>>);

fn parameters_are_strictly_in_domain(values: &[f64], scales: &[ParameterScale]) -> bool {
    values.len() == scales.len()
        && values.iter().zip(scales).all(|(value, scale)| {
            value.is_finite()
                && match scale {
                    ParameterScale::Identity => true,
                    ParameterScale::Log => *value > 0.0,
                    ParameterScale::Logit { lower, upper }
                    | ParameterScale::Probit { lower, upper } => *value > *lower && *value < *upper,
                }
        })
}

fn non_iiv_candidate_improves(current: f64, candidate: f64) -> bool {
    candidate.is_finite() && candidate < current
}

struct NonIivPopulationCost<'a, E: Equation> {
    state: &'a SaemState<E>,
    layout: &'a NonIivCoordinateLayout,
}

impl<E: Equation> CostFunction for NonIivPopulationCost<'_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, coordinates: &Self::Param) -> std::result::Result<f64, ArgminError> {
        Ok(self
            .state
            .non_iiv_observation_nll(self.layout, coordinates)
            .unwrap_or(NON_IIV_OPTIMIZER_PENALTY))
    }
}

const NON_IIV_OPTIMIZER_MAX_ITERATIONS: u64 = 100;
const NON_IIV_OPTIMIZER_PENALTY: f64 = 1e100;
const NON_IIV_OPTIMIZER_SD_TOLERANCE: f64 = 1e-8;
const PROPOSAL_SCALE_DECREASE: f64 = 0.9;
const MIN_PROPOSAL_SCALE: f64 = 1e-6;
const MAX_PROPOSAL_SCALE: f64 = 5.0;

/// SAEM iteration schedule derived from [`SaemConfig`].
///
/// This uses the established high-level split: a pure burn-in
/// region, an exploration region with full stochastic approximation updates,
/// then a smoothing region with decreasing step size.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SaemSchedule {
    pub(crate) pure_burn_in: usize,
    pub(crate) exploration_iterations: usize,
    pub(crate) smoothing_iterations: usize,
    pub(crate) total_iterations: usize,
    pub(crate) variance_floor_iterations: usize,
    pub(crate) annealing_alpha: f64,
    pub(crate) omega_sa_max_step: f64,
    pub(crate) minimum_variance: f64,
    pub(crate) minimum_iov_variance: f64,
    pub(crate) minimum_residual_sigma: f64,
    pub(crate) averaging_alpha: Option<f64>,
}

impl SaemSchedule {
    pub(crate) fn from_config(config: &SaemConfig) -> Self {
        let pure_burn_in = config.burn_in;
        let exploration_iterations = config.k1_iterations.saturating_sub(pure_burn_in);
        let smoothing_iterations = config.k2_iterations;
        let total_iterations = config.k1_iterations + config.k2_iterations;
        let variance_floor_iterations = if config.sa_iterations > 0 {
            config.sa_iterations
        } else {
            config.k1_iterations / 2
        };

        Self {
            pure_burn_in,
            exploration_iterations,
            smoothing_iterations,
            total_iterations,
            variance_floor_iterations,
            annealing_alpha: config.sa_cooling_factor,
            omega_sa_max_step: config.omega_sa_max_step,
            minimum_variance: config.omega_min_variance,
            minimum_iov_variance: config.omega_iov_min_variance,
            minimum_residual_sigma: config.residual_min_sigma,
            averaging_alpha: match config.estimator_policy {
                SaemEstimatorPolicy::TerminalIterate => None,
                SaemEstimatorPolicy::AveragedIterates { alpha } => Some(alpha),
            },
        }
    }

    pub(crate) fn stochastic_approximation_step(&self, iteration: usize) -> f64 {
        if iteration <= self.pure_burn_in {
            0.0
        } else if iteration <= self.pure_burn_in + self.exploration_iterations {
            1.0
        } else {
            let smoothing_iteration = iteration
                .saturating_sub(self.pure_burn_in + self.exploration_iterations)
                .max(1);
            match self.averaging_alpha {
                Some(alpha) => (smoothing_iteration as f64).powf(-alpha),
                None => 1.0 / smoothing_iteration as f64,
            }
        }
    }

    /// Stochastic-approximation step for Ω/Ω_IOV sufficient statistics.
    ///
    /// Covariance learning is damped during both pure chain
    /// warm-up and exploration so one un-equilibrated draw cannot overwrite a
    /// correlated covariance. The cap is lifted in smoothing.
    pub(crate) fn covariance_step(&self, iteration: usize) -> f64 {
        if iteration <= self.pure_burn_in + self.exploration_iterations {
            self.omega_sa_max_step.min(1.0)
        } else {
            self.stochastic_approximation_step(iteration)
        }
    }

    pub(crate) fn covariance_update_active(&self, iteration: usize) -> bool {
        iteration > self.pure_burn_in
    }

    pub(crate) fn phase(&self, iteration: usize) -> SaemPhase {
        if iteration <= self.pure_burn_in {
            SaemPhase::BurnIn
        } else if iteration <= self.pure_burn_in + self.exploration_iterations {
            SaemPhase::Exploration
        } else {
            SaemPhase::Smoothing
        }
    }

    /// Guard an estimated residual SD against early collapse.
    ///
    /// During simulated annealing, PMcore cools the previous residual SD by
    /// `alpha.sa` and takes the larger of that value and the M-step candidate.
    /// The configured residual floor always applies. Fixed residual models are
    /// left untouched.
    pub(crate) fn guarded_residual_sigma(
        &self,
        iteration: usize,
        previous: f64,
        candidate: f64,
    ) -> f64 {
        let mut guarded = candidate.max(self.minimum_residual_sigma);
        if iteration <= self.variance_floor_iterations {
            guarded = guarded.max(previous * self.annealing_alpha);
        }
        guarded
    }
}

fn covariate_omega_update_maximum_fraction(
    has_covariates: bool,
    phase: SaemPhase,
    covariance_step: f64,
) -> f64 {
    if has_covariates && phase == SaemPhase::Exploration {
        covariance_step
    } else {
        1.0
    }
}

fn applied_combined_residual_component(
    schedule: &SaemSchedule,
    iteration: usize,
    previous: f64,
    candidate: f64,
    estimated: bool,
) -> f64 {
    if !estimated {
        return previous;
    }
    let guarded_candidate = candidate.max(schedule.minimum_residual_sigma);
    if iteration <= schedule.variance_floor_iterations {
        return guarded_candidate.max(previous * schedule.annealing_alpha);
    }
    if schedule.phase(iteration) != SaemPhase::Smoothing {
        return guarded_candidate;
    }
    let gamma = schedule.stochastic_approximation_step(iteration);
    previous + gamma * (guarded_candidate - previous)
}

/// Immutable SAEM setup computed once before the iterations begin.
///
/// Parameter metadata, random/IOV effect indices, the resolved omega
/// specification, and initial subject-conditioned log-likelihoods are all
/// resolved here so the runner state only carries mutable estimation state.
#[derive(Debug, Clone)]
pub(crate) struct SaemInitialization {
    pub(crate) schedule: SaemSchedule,
    pub(crate) n_chains: usize,
    pub(crate) parameter_names: Vec<String>,
    pub(crate) parameter_scales: Vec<ParameterScale>,
    pub(crate) estimated_parameters: Vec<bool>,
    pub(crate) random_effect_indices: Vec<usize>,
    pub(crate) random_effect_names: Vec<String>,
    pub(crate) omega: ResolvedOmega,
    pub(crate) iov_effect_indices: Vec<usize>,
    pub(crate) iov_effect_names: Vec<String>,
    pub(crate) omega_iov: Option<ResolvedOmega>,
    pub(crate) occasion_counts: Vec<usize>,
    pub(crate) subject_ids: Vec<String>,
    pub(crate) observation_count: usize,
    pub(crate) initial_population_parameters: Vec<f64>,
    pub(crate) initial_subject_log_likelihoods: Vec<f64>,
    pub(crate) initial_negative_log_likelihood: f64,
    pub(crate) covariate_model: Option<CovariateModel>,
    pub(crate) initial_subject_mu_phi: Option<Vec<Vec<f64>>>,
    pub(crate) initial_residual_values: Vec<Vec<f64>>,
    pub(crate) initial_residual_estimated: Vec<Vec<bool>>,
}

fn applied_correlated_residual_correlation(
    schedule: &SaemSchedule,
    iteration: usize,
    previous: f64,
    candidate: f64,
    estimated: bool,
) -> f64 {
    if !estimated {
        return previous;
    }
    if schedule.phase(iteration) != SaemPhase::Smoothing {
        return candidate;
    }
    let gamma = schedule.stochastic_approximation_step(iteration);
    previous + gamma * (candidate - previous)
}

fn validate_initial_estimated_variance_floor(
    covariance_name: &str,
    floor_name: &str,
    omega: &ResolvedOmega,
    minimum_variance: f64,
) -> Result<()> {
    for (index, effect_name) in omega.names().iter().enumerate() {
        let initial_variance = omega.initial()[[index, index]];
        if omega.estimated_mask()[[index, index]] && initial_variance < minimum_variance {
            anyhow::bail!(
                "SAEM initial {covariance_name} variance for estimated effect '{effect_name}' ({initial_variance}) is below configured {floor_name} ({minimum_variance})"
            );
        }
    }
    Ok(())
}

impl SaemInitialization {
    pub(crate) fn create<E>(
        problem: &EstimationProblem<E, Parametric>,
        config: &SaemConfig,
    ) -> Result<Self>
    where
        E: Equation,
    {
        config.validate()?;
        let omega = problem.prior.resolved_omega().clone();
        let n_subjects = problem.data.subjects().len();
        let initial_row = initial_parameter_row(problem.parameters().iter());
        let random_effect_indices = problem
            .parameters()
            .iter()
            .enumerate()
            .filter_map(|(index, parameter)| parameter.random_effect.then_some(index))
            .collect::<Vec<_>>();
        let random_effect_names = random_effect_indices
            .iter()
            .map(|index| problem.parameters().items[*index].name.clone())
            .collect();
        let (iov_effect_indices, iov_effect_names, omega_iov) = problem
            .prior
            .resolved_iov()
            .map(|iov| {
                (
                    iov.parameter_indices().to_vec(),
                    iov.omega().names().to_vec(),
                    Some(iov.omega().clone()),
                )
            })
            .unwrap_or_else(|| (Vec::new(), Vec::new(), None));
        validate_initial_estimated_variance_floor(
            "Omega",
            "omega_min_variance",
            &omega,
            config.omega_min_variance,
        )?;
        if let Some(omega_iov) = omega_iov.as_ref() {
            validate_initial_estimated_variance_floor(
                "Omega_IOV",
                "omega_iov_min_variance",
                omega_iov,
                config.omega_iov_min_variance,
            )?;
        }
        if config.marginal_likelihood.is_some()
            && (!random_effect_indices.is_empty() || !iov_effect_indices.is_empty())
            && !config.compute_map
        {
            anyhow::bail!(
                "N2 with latent dimensions requires compute_map=true; conditional modes are not enabled"
            );
        }
        let covariate_model = problem.covariates().cloned();
        let initial_population_phi = population_phi(
            &initial_row,
            &problem
                .parameters()
                .iter()
                .map(|parameter| parameter.scale)
                .collect::<Vec<_>>(),
        )?;
        let initial_subject_population = covariate_model
            .as_ref()
            .map(|model| {
                model.subject_population_parameters(
                    &initial_population_phi,
                    &problem
                        .parameters()
                        .iter()
                        .map(|parameter| parameter.scale)
                        .collect::<Vec<_>>(),
                )
            })
            .transpose()?;
        let initial_subject_mu_phi = initial_subject_population.as_ref().map(|rows| {
            rows.iter()
                .map(|row| row.phi().to_vec())
                .collect::<Vec<_>>()
        });
        let initial_individual_parameters = match initial_subject_population.as_ref() {
            Some(rows) => {
                Array2::from_shape_fn((n_subjects, initial_row.len()), |(i, j)| rows[i].psi()[j])
            }
            None => Array2::from_shape_fn((n_subjects, initial_row.len()), |(_, j)| initial_row[j]),
        };
        let initial_subject_log_likelihoods =
            parametric_subject_log_likelihoods(problem, &initial_individual_parameters)?;
        if let Some((subject_index, _)) = initial_subject_log_likelihoods
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            let subject = problem.data.subjects()[subject_index];
            if let Ok(statistics) = residual_statistics_for_subject(
                &problem.model.equation,
                subject,
                &initial_row,
                problem.error_models.models(),
            ) {
                for (output_index, _) in problem.error_models.models().iter() {
                    let Some(statistic) = statistics.output(output_index) else {
                        continue;
                    };
                    if statistic.exponential_domain_violation_count > 0 {
                        let output = problem
                            .error_models
                            .output_name(output_index)
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("output_{output_index}"));
                        anyhow::bail!(
                            "initial conditional likelihood is non-finite for subject '{}' because exponential residual model output '{}' has {} non-positive or non-finite observation/prediction pair(s); exponential errors require positive finite observations and predictions",
                            subject.id(),
                            output,
                            statistic.exponential_domain_violation_count
                        );
                    }
                }
            }
            anyhow::bail!(
                "initial conditional likelihood is non-finite for subject '{}'; verify parameter values, predictions, observations, and residual-model domain",
                subject.id()
            );
        }
        let initial_negative_log_likelihood =
            negative_log_likelihood(&initial_subject_log_likelihoods);
        Ok(Self {
            schedule: SaemSchedule::from_config(config),
            n_chains: n_chains(config, n_subjects),
            parameter_names: problem.parameters().names(),
            parameter_scales: problem
                .parameters()
                .iter()
                .map(|parameter| parameter.scale)
                .collect(),
            estimated_parameters: problem
                .parameters()
                .iter()
                .map(|parameter| parameter.estimate)
                .collect(),
            random_effect_indices,
            random_effect_names,
            omega,
            iov_effect_indices,
            iov_effect_names,
            omega_iov,
            occasion_counts: problem
                .data
                .subjects()
                .iter()
                .map(|subject| subject.occasions().len())
                .collect(),
            subject_ids: problem
                .data
                .subjects()
                .iter()
                .map(|subject| subject.id().clone())
                .collect(),
            observation_count: count_observations(&problem.data),
            initial_population_parameters: initial_row,
            initial_subject_log_likelihoods,
            initial_negative_log_likelihood,
            covariate_model,
            initial_subject_mu_phi,
            initial_residual_values: Vec::new(),
            initial_residual_estimated: Vec::new(),
        })
    }
}

#[derive(Debug, Clone)]
struct SaemIterateAverage {
    population_phi: Vec<f64>,
    covariate_betas: Option<Vec<f64>>,
    omega: Array2<f64>,
    omega_iov: Option<Array2<f64>>,
    residual_model_width: usize,
    residual_models: Vec<(usize, ResidualErrorModel)>,
    start_cycle: usize,
    count: usize,
}

// ─── Operational convergence lifecycle ────────────────────────────────────
//
// Result types live in `crate::results::fit_result`.
// `OperationalConvergenceConfig` is the source of truth for settings.

/// Domain-separation constant for deterministic per-checkpoint seeds.
///
/// Its fixed bytes are combined with the SAEM seed via wrapping addition.
const OPERATIONAL_CHECKPOINT_SEED_DOMAIN: u64 = 0x4E31_4F50_4352_4954;

/// Per-cycle SAEM estimation state.
///
/// MCMC chains, stochastic-approximation sufficient statistics, and the
/// current population / omega / sigma estimates are updated in-place.
#[derive(Debug)]
pub(crate) struct SaemState<E: Equation> {
    equation: E,
    data: Data,
    error_models: ParametricErrorModels,
    config: SaemConfig,
    pub(crate) initialization: SaemInitialization,
    cycle: usize,
    status: Status,
    numerical_failure: Option<NumericalFailure>,
    etas: Vec<Vec<Vec<f64>>>,
    kappas: Vec<Vec<Vec<Vec<f64>>>>,
    population_parameters: Vec<f64>,
    omega: Array2<f64>,
    omega_iov: Option<Array2<f64>>,
    iiv_second_moment: Array2<f64>,
    iov_second_moment: Option<Array2<f64>>,
    sufficient_statistics: PhiSufficientStatistics,
    covariate_statistics: Option<CovariateSufficientStatistics>,
    subject_mu_phi: Option<Vec<Vec<f64>>>,
    covariate_model: Option<CovariateModel>,
    residual_statistics: ResidualSufficientStatistics,
    residual_sigmas: Vec<f64>,
    information: InformationRecursion,
    proposal_step_sizes: Vec<f64>,
    eta_block_step_sizes: Vec<f64>,
    kappa_proposal_step_sizes: Vec<f64>,
    mcmc_iterations: usize,
    eta_block_iterations: usize,
    adapt_interval: usize,
    residual_optimizer_max_iterations: usize,
    compute_map: bool,
    map_max_iterations: usize,
    map_sd_tolerance: f64,
    map_initial_step: f64,
    steps_since_adapt: usize,
    adaptation_accept_counts: Vec<usize>,
    adaptation_proposal_counts: Vec<usize>,
    eta_block_adaptation_accept_counts: Vec<usize>,
    eta_block_adaptation_proposal_counts: Vec<usize>,
    kappa_adaptation_accept_counts: Vec<usize>,
    kappa_adaptation_proposal_counts: Vec<usize>,
    rng: StdRng,
    subject_log_likelihoods: Vec<f64>,
    subject_log_priors: Vec<f64>,
    subject_kappa_log_priors: Vec<f64>,
    last_log_acceptance_ratios: Vec<f64>,
    last_acceptance_rate: Option<f64>,
    last_eta_block_acceptance_rate: Option<f64>,
    last_kappa_acceptance_rate: Option<f64>,
    last_rejected_proposals: Option<usize>,
    last_non_finite_proposals: Option<usize>,
    last_parameter_acceptance_rates: Vec<f64>,
    cycle_diagnostics: Vec<SaemCycleDiagnostics>,
    negative_log_likelihood: f64,
    iterate_average: Option<SaemIterateAverage>,
    operational_settings: Option<OperationalConvergenceConfig>,
    operational_diagnostics: OperationalConvergenceDiagnostics,
}

impl<E: Equation> SaemState<E> {
    pub(crate) fn from_problem(
        problem: EstimationProblem<E, Parametric>,
        config: &SaemConfig,
    ) -> Result<Self> {
        let mut initialization = SaemInitialization::create(&problem, config)?;
        let EstimationProblem {
            model,
            data,
            error_models,
            ..
        } = problem;
        // Capture immutable initial residual values and estimated masks before
        // any SAEM cycle modifies them.
        let mut initial_residual_values = Vec::new();
        let mut initial_residual_estimated = Vec::new();
        for (outeq, model) in error_models.models().iter() {
            let estimate = error_models.is_estimated(outeq);
            let combined = error_models.combined_component_estimated(outeq);
            let correlated = error_models.correlated_combined_component_estimated(outeq);
            let (additive, proportional, correlation) =
                if matches!(model, ResidualErrorModel::CorrelatedCombined { .. }) {
                    (correlated[0], correlated[1], Some(correlated[2]))
                } else {
                    (combined[0], combined[1], None)
                };
            let components = crate::results::parametric_output::residual_components(
                *model,
                estimate,
                Some(additive),
                Some(proportional),
                correlation,
            );
            initial_residual_values.push(components.iter().map(|c| c.1).collect());
            initial_residual_estimated.push(components.iter().map(|c| c.2).collect());
        }
        initialization.initial_residual_values = initial_residual_values;
        initialization.initial_residual_estimated = initial_residual_estimated;
        Ok(Self::new(
            model.equation,
            data,
            error_models,
            initialization,
            config,
        ))
    }

    pub(crate) fn new(
        equation: E,
        data: Data,
        error_models: ParametricErrorModels,
        initialization: SaemInitialization,
        config: &SaemConfig,
    ) -> Self {
        let n_random_effects = initialization.random_effect_indices.len();
        let etas = zero_etas(
            initialization.subject_ids.len(),
            initialization.n_chains,
            n_random_effects,
        );
        let kappas = zero_kappas(
            &initialization.occasion_counts,
            initialization.n_chains,
            initialization.iov_effect_indices.len(),
        );
        let population_parameters = initialization.initial_population_parameters.clone();
        let omega = initialization.omega.initial().clone();
        let iiv_second_moment = omega.clone();
        let omega_iov = initialization
            .omega_iov
            .as_ref()
            .map(|omega| omega.initial().clone());
        let iov_second_moment = omega_iov.clone();
        let initial_subject_phi = zero_eta_subject_phi(&population_parameters, &initialization)
            .expect("initial population parameters should produce valid phi statistics");
        let mut sufficient_statistics =
            PhiSufficientStatistics::from_subject_phi(&initial_subject_phi)
                .expect("initial phi statistics should be valid");
        for (eta_row, parameter_row) in initialization.random_effect_indices.iter().enumerate() {
            for (eta_col, parameter_col) in initialization.random_effect_indices.iter().enumerate()
            {
                sufficient_statistics.second_moment[[*parameter_row, *parameter_col]] +=
                    omega[[eta_row, eta_col]];
            }
        }
        let subject_mu_phi = initialization.initial_subject_mu_phi.clone();
        let covariate_model = initialization.covariate_model.clone();
        let covariate_statistics = subject_mu_phi.as_ref().map(|means| {
            let expected_phi = means
                .iter()
                .map(|mean| {
                    initialization
                        .random_effect_indices
                        .iter()
                        .map(|index| mean[*index])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut global_second_moment = Array2::zeros((n_random_effects, n_random_effects));
            for mean in &expected_phi {
                for row in 0..n_random_effects {
                    for column in 0..n_random_effects {
                        global_second_moment[[row, column]] +=
                            mean[row] * mean[column] / expected_phi.len() as f64;
                    }
                }
            }
            global_second_moment += &omega;
            CovariateSufficientStatistics {
                expected_phi,
                global_second_moment,
            }
        });
        let subject_log_priors = eta_log_priors(&etas, &omega, 0)
            .expect("validated initial omega should produce finite eta priors");
        let subject_kappa_log_priors = omega_iov
            .as_ref()
            .map(|omega| {
                kappas
                    .iter()
                    .map(|subject_chains| {
                        subject_chains[0]
                            .iter()
                            .map(|kappa| eta_log_prior_from_omega(kappa, omega))
                            .collect::<Result<Vec<_>>>()
                            .map(|priors| priors.into_iter().sum())
                    })
                    .collect::<Result<Vec<_>>>()
                    .expect("validated initial omega_iov should produce finite kappa priors")
            })
            .unwrap_or_else(|| vec![0.0; initialization.subject_ids.len()]);
        let residual_statistics = ResidualSufficientStatistics::zero(error_models.models().len());
        let residual_sigmas = primary_sigma_parameters(error_models.models());
        let proposal_step_sizes = initial_proposal_step_sizes(&omega, config.rw_init);
        let eta_block_step_sizes = if config.eta_block_iterations > 0 {
            vec![config.rw_init; initialization.subject_ids.len()]
        } else {
            Vec::new()
        };
        let kappa_proposal_step_sizes = omega_iov
            .as_ref()
            .map(|_| vec![config.rw_init; initialization.subject_ids.len()])
            .unwrap_or_default();
        let mcmc_iterations = config.mcmc_iterations;
        let eta_block_iterations = config.eta_block_iterations;
        let adapt_interval = config.adapt_interval;
        let steps_since_adapt = 0;
        let adaptation_accept_counts = vec![0; n_random_effects];
        let adaptation_proposal_counts = vec![0; n_random_effects];
        let eta_block_adaptation_accept_counts = vec![0; eta_block_step_sizes.len()];
        let eta_block_adaptation_proposal_counts = vec![0; eta_block_step_sizes.len()];
        let kappa_adaptation_accept_counts = vec![0; initialization.subject_ids.len()];
        let kappa_adaptation_proposal_counts = vec![0; initialization.subject_ids.len()];
        let rng = StdRng::seed_from_u64(config.seed);
        let last_log_acceptance_ratios = vec![0.0; initialization.subject_ids.len()];
        let last_acceptance_rate = None;
        let last_parameter_acceptance_rates = vec![0.0; n_random_effects];
        let covariate_effect_names = covariate_model
            .as_ref()
            .map(|model| {
                model
                    .estimates()
                    .iter()
                    .map(|estimate| estimate.name().to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let covariate_estimated = covariate_model
            .as_ref()
            .map(|model| {
                model
                    .estimates()
                    .iter()
                    .map(|estimate| estimate.estimated())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let information_layout = InformationLayout::new(
            &initialization.parameter_names,
            &initialization.estimated_parameters,
            &covariate_effect_names,
            &covariate_estimated,
            &initialization.random_effect_names,
            initialization.omega.structural_mask(),
            initialization.omega.estimated_mask(),
            &initialization.iov_effect_names,
            initialization
                .omega_iov
                .as_ref()
                .map(|omega| omega.structural_mask()),
            initialization
                .omega_iov
                .as_ref()
                .map(|omega| omega.estimated_mask()),
            &error_models,
        )
        .expect("validated SAEM metadata must produce an information layout");
        let mut information = InformationRecursion::new(information_layout);
        let has_non_iiv_population =
            initialization
                .estimated_parameters
                .iter()
                .enumerate()
                .any(|(index, estimated)| {
                    *estimated && !initialization.random_effect_indices.contains(&index)
                });
        let has_non_iiv_covariate = covariate_model.as_ref().is_some_and(|model| {
            model
                .estimates()
                .iter()
                .enumerate()
                .any(|(index, estimate)| {
                    estimate.estimated()
                        && !initialization
                            .random_effect_indices
                            .contains(&model.parameter_indices()[index])
                })
        });
        if has_non_iiv_population || has_non_iiv_covariate {
            information.mark_unavailable(InformationStatus::Unsupported(
                "structural observation sensitivities are unavailable for estimated non-IIV population or covariate coordinates"
                    .to_string(),
            ));
        }

        Self {
            equation,
            data,
            error_models,
            config: config.clone(),
            etas,
            kappas,
            population_parameters,
            omega,
            omega_iov,
            iiv_second_moment,
            iov_second_moment,
            sufficient_statistics,
            covariate_statistics,
            subject_mu_phi,
            covariate_model,
            residual_statistics,
            residual_sigmas,
            information,
            proposal_step_sizes,
            eta_block_step_sizes,
            kappa_proposal_step_sizes,
            mcmc_iterations,
            eta_block_iterations,
            adapt_interval,
            residual_optimizer_max_iterations: config.residual_optimizer_max_iterations,
            compute_map: config.compute_map,
            map_max_iterations: config.map_max_iterations,
            map_sd_tolerance: config.map_sd_tolerance,
            map_initial_step: config.map_initial_step,
            steps_since_adapt,
            adaptation_accept_counts,
            adaptation_proposal_counts,
            eta_block_adaptation_accept_counts,
            eta_block_adaptation_proposal_counts,
            kappa_adaptation_accept_counts,
            kappa_adaptation_proposal_counts,
            rng,
            subject_log_likelihoods: initialization.initial_subject_log_likelihoods.clone(),
            subject_log_priors,
            subject_kappa_log_priors,
            last_log_acceptance_ratios,
            last_acceptance_rate,
            last_eta_block_acceptance_rate: None,
            last_kappa_acceptance_rate: None,
            last_rejected_proposals: None,
            last_non_finite_proposals: None,
            last_parameter_acceptance_rates,
            cycle_diagnostics: Vec::with_capacity(initialization.schedule.total_iterations),
            negative_log_likelihood: initialization.initial_negative_log_likelihood,
            iterate_average: None,
            operational_settings: config.operational_convergence,
            operational_diagnostics: OperationalConvergenceDiagnostics {
                config: config.operational_convergence,
                ..OperationalConvergenceDiagnostics::default()
            },
            initialization,
            cycle: 0,
            status: Status::Continue,
            numerical_failure: None,
        }
    }

    fn e_step(&mut self) -> Result<()> {
        let mut eta_accepted = 0usize;
        let mut eta_rejected = 0usize;
        let mut eta_non_finite = 0usize;
        let mut eta_proposed = 0usize;
        let mut eta_block_accepted = 0usize;
        let mut eta_block_rejected = 0usize;
        let mut eta_block_non_finite = 0usize;
        let mut eta_block_proposed = 0usize;
        let mut kappa_accepted = 0usize;
        let mut kappa_rejected = 0usize;
        let mut kappa_non_finite = 0usize;
        let mut kappa_proposed = 0usize;
        let eta_step_sizes_before = self.proposal_step_sizes.clone();
        let eta_block_step_sizes_before = self.eta_block_step_sizes.clone();
        let kappa_step_sizes_before = self.kappa_proposal_step_sizes.clone();
        let kappa_subject_count = if self.omega_iov.is_some() {
            self.initialization.subject_ids.len()
        } else {
            0
        };
        let mut kappa_subject_accept_counts = vec![0usize; kappa_subject_count];
        let mut kappa_subject_proposal_counts = vec![0usize; kappa_subject_count];
        let eta_block_subject_count = if self.eta_block_iterations > 0 {
            self.initialization.subject_ids.len()
        } else {
            0
        };
        let mut eta_block_subject_accept_counts = vec![0usize; eta_block_subject_count];
        let mut eta_block_subject_proposal_counts = vec![0usize; eta_block_subject_count];
        let n_parameters = self.initialization.random_effect_indices.len();
        let mut subject_log_acceptance_sums = vec![0.0; self.initialization.subject_ids.len()];
        let mut subject_proposal_counts = vec![0usize; self.initialization.subject_ids.len()];
        let mut parameter_accept_counts = vec![0usize; n_parameters];
        let mut parameter_proposal_counts = vec![0usize; n_parameters];

        // Compound-kernel order: Omega-scaled eta blocks first, followed by
        // component eta walks and occasion-level kappa blocks. Eta blocks are
        // opt-in.
        for _ in 0..self.eta_block_iterations {
            for subject_index in 0..self.initialization.subject_ids.len() {
                for chain_index in 0..self.initialization.n_chains {
                    let current_eta = self.etas[subject_index][chain_index].clone();
                    let proposed_eta = self.block_random_walk_eta(&current_eta, subject_index)?;
                    let log_acceptance_ratio = self.proposal_log_acceptance_ratio(
                        subject_index,
                        chain_index,
                        &proposed_eta,
                    )?;
                    subject_log_acceptance_sums[subject_index] += log_acceptance_ratio;
                    subject_proposal_counts[subject_index] += 1;
                    eta_block_subject_proposal_counts[subject_index] += 1;
                    self.eta_block_adaptation_proposal_counts[subject_index] += 1;
                    eta_block_proposed += 1;
                    eta_proposed += 1;
                    if !log_acceptance_ratio.is_finite() {
                        eta_block_non_finite += 1;
                        eta_non_finite += 1;
                    }
                    if self.accept_proposal(log_acceptance_ratio) {
                        self.etas[subject_index][chain_index] = proposed_eta;
                        eta_block_subject_accept_counts[subject_index] += 1;
                        self.eta_block_adaptation_accept_counts[subject_index] += 1;
                        eta_block_accepted += 1;
                        eta_accepted += 1;
                    } else {
                        eta_block_rejected += 1;
                        eta_rejected += 1;
                    }
                }
            }
        }

        for _ in 0..self.mcmc_iterations {
            for subject_index in 0..self.initialization.subject_ids.len() {
                for chain_index in 0..self.initialization.n_chains {
                    for parameter_index in 0..n_parameters {
                        let current_eta = self.etas[subject_index][chain_index].clone();
                        let proposed_eta =
                            self.component_random_walk_eta(&current_eta, parameter_index);
                        let log_acceptance_ratio = self.proposal_log_acceptance_ratio(
                            subject_index,
                            chain_index,
                            &proposed_eta,
                        )?;
                        subject_log_acceptance_sums[subject_index] += log_acceptance_ratio;
                        subject_proposal_counts[subject_index] += 1;
                        parameter_proposal_counts[parameter_index] += 1;
                        eta_proposed += 1;
                        if !log_acceptance_ratio.is_finite() {
                            eta_non_finite += 1;
                        }
                        if self.accept_proposal(log_acceptance_ratio) {
                            self.etas[subject_index][chain_index] = proposed_eta;
                            parameter_accept_counts[parameter_index] += 1;
                            eta_accepted += 1;
                        } else {
                            eta_rejected += 1;
                        }
                    }

                    // Gibbs sweep over occasion-specific κ blocks. Every
                    // proposal is evaluated against the full subject posterior,
                    // keeping η and all other occasions fixed.
                    if self.omega_iov.is_some() {
                        for occasion_index in 0..self.kappas[subject_index][chain_index].len() {
                            let current_kappa =
                                self.kappas[subject_index][chain_index][occasion_index].clone();
                            let proposed_kappa =
                                self.block_random_walk_kappa(&current_kappa, subject_index)?;
                            let log_acceptance_ratio = self.kappa_proposal_log_acceptance_ratio(
                                subject_index,
                                chain_index,
                                occasion_index,
                                &proposed_kappa,
                            )?;
                            subject_log_acceptance_sums[subject_index] += log_acceptance_ratio;
                            subject_proposal_counts[subject_index] += 1;
                            kappa_proposed += 1;
                            kappa_subject_proposal_counts[subject_index] += 1;
                            self.kappa_adaptation_proposal_counts[subject_index] += 1;
                            if !log_acceptance_ratio.is_finite() {
                                kappa_non_finite += 1;
                            }
                            if self.accept_proposal(log_acceptance_ratio) {
                                self.kappas[subject_index][chain_index][occasion_index] =
                                    proposed_kappa;
                                kappa_accepted += 1;
                                kappa_subject_accept_counts[subject_index] += 1;
                                self.kappa_adaptation_accept_counts[subject_index] += 1;
                            } else {
                                kappa_rejected += 1;
                            }
                        }
                    }
                }
            }
        }

        self.refresh_subject_scores_from_chains()?;
        self.last_log_acceptance_ratios = subject_log_acceptance_sums
            .into_iter()
            .zip(subject_proposal_counts)
            .map(|(sum, count)| if count > 0 { sum / count as f64 } else { 0.0 })
            .collect();
        let proposed = eta_proposed + kappa_proposed;
        let accepted = eta_accepted + kappa_accepted;
        self.last_acceptance_rate = if proposed > 0 {
            Some(accepted as f64 / proposed as f64)
        } else {
            None
        };
        self.last_eta_block_acceptance_rate = if self.eta_block_iterations > 0 {
            Some(eta_block_accepted as f64 / eta_block_proposed.max(1) as f64)
        } else {
            None
        };
        self.last_kappa_acceptance_rate = if self.omega_iov.is_some() {
            Some(kappa_accepted as f64 / kappa_proposed.max(1) as f64)
        } else {
            None
        };
        self.last_rejected_proposals = Some(eta_rejected + kappa_rejected);
        self.last_non_finite_proposals = Some(eta_non_finite + kappa_non_finite);
        self.last_parameter_acceptance_rates = parameter_accept_counts
            .iter()
            .zip(parameter_proposal_counts.iter())
            .map(|(accepted, proposed)| {
                if *proposed > 0 {
                    *accepted as f64 / *proposed as f64
                } else {
                    0.0
                }
            })
            .collect();
        for parameter_index in 0..n_parameters {
            self.adaptation_accept_counts[parameter_index] +=
                parameter_accept_counts[parameter_index];
            self.adaptation_proposal_counts[parameter_index] +=
                parameter_proposal_counts[parameter_index];
        }
        self.steps_since_adapt += 1;
        self.adapt_proposal_step_sizes();
        let phase = self.initialization.schedule.phase(self.cycle);
        let omega_update = pending_covariance_update_diagnostics(
            phase,
            true,
            self.initialization.omega.has_estimated_entries(),
        );
        let omega_iov_update = pending_covariance_update_diagnostics(
            phase,
            self.initialization.omega_iov.is_some(),
            self.initialization
                .omega_iov
                .as_ref()
                .is_some_and(ResolvedOmega::has_estimated_entries),
        );
        self.cycle_diagnostics.push(SaemCycleDiagnostics {
            iteration: self.cycle,
            phase,
            stochastic_approximation_step: self
                .initialization
                .schedule
                .stochastic_approximation_step(self.cycle),
            covariance_step: self.initialization.schedule.covariance_step(self.cycle),
            eta_proposals: eta_proposed,
            eta_accepted,
            eta_rejected,
            eta_non_finite,
            eta_parameter_acceptance_rates: self.last_parameter_acceptance_rates.clone(),
            eta_proposal_step_sizes_before_adaptation: eta_step_sizes_before,
            eta_proposal_step_sizes_after_adaptation: self.proposal_step_sizes.clone(),
            eta_block_proposals: eta_block_proposed,
            eta_block_accepted,
            eta_block_rejected,
            eta_block_non_finite,
            eta_block_subject_acceptance_rates: eta_block_subject_accept_counts
                .iter()
                .zip(eta_block_subject_proposal_counts.iter())
                .map(|(accepted, proposed)| {
                    if *proposed > 0 {
                        *accepted as f64 / *proposed as f64
                    } else {
                        0.0
                    }
                })
                .collect(),
            eta_block_step_sizes_before_adaptation: eta_block_step_sizes_before,
            eta_block_step_sizes_after_adaptation: self.eta_block_step_sizes.clone(),
            kappa_proposals: kappa_proposed,
            kappa_accepted,
            kappa_rejected,
            kappa_non_finite,
            kappa_subject_acceptance_rates: kappa_subject_accept_counts
                .iter()
                .zip(kappa_subject_proposal_counts.iter())
                .map(|(accepted, proposed)| {
                    if *proposed > 0 {
                        *accepted as f64 / *proposed as f64
                    } else {
                        0.0
                    }
                })
                .collect(),
            kappa_proposal_step_sizes_before_adaptation: kappa_step_sizes_before,
            kappa_proposal_step_sizes_after_adaptation: self.kappa_proposal_step_sizes.clone(),
            simulated_annealing_active: self.cycle
                <= self.initialization.schedule.variance_floor_iterations,
            population_parameters: self.population_parameters.clone(),
            omega: self.omega.clone(),
            omega_iov: self.omega_iov.clone(),
            residual_error_estimates: self.residual_error_estimates(),
            residual_diagnostics: Vec::new(),
            conditional_negative_log_likelihood: self.negative_log_likelihood,
            eta_log_prior: self.subject_log_priors.iter().sum(),
            kappa_log_prior: self.subject_kappa_log_priors.iter().sum(),
            omega_update_rejected: false,
            omega_iov_update_rejected: false,
            omega_update,
            omega_iov_update,
            omega_relative_spd_margin: None,
            omega_iov_relative_spd_margin: None,
            covariate_betas: self.covariate_model.as_ref().map(|model| {
                model
                    .estimates()
                    .iter()
                    .map(|estimate| estimate.estimate())
                    .collect()
            }),
            covariate_beta_estimated: self.covariate_model.as_ref().map(|model| {
                model
                    .estimates()
                    .iter()
                    .map(|estimate| estimate.estimated())
                    .collect()
            }),
        });
        self.negative_log_likelihood = negative_log_likelihood(&self.subject_log_likelihoods);
        Ok(())
    }

    fn m_step(&mut self) -> Result<()> {
        let parameter_step = self
            .initialization
            .schedule
            .stochastic_approximation_step(self.cycle);
        let covariance_step = self.initialization.schedule.covariance_step(self.cycle);
        if self.covariate_model.is_some() {
            let observed = self.current_covariate_statistics()?;
            self.covariate_statistics
                .as_mut()
                .expect("covariate model has initialized statistics")
                .stochastic_update(&observed, parameter_step)?;
        } else {
            let observed_statistics = self.current_phi_statistics()?;
            self.sufficient_statistics.stochastic_update_with_steps(
                &observed_statistics,
                parameter_step,
                covariance_step,
            )?;
        }

        if let Some(second_moment) = self.iov_second_moment.as_mut() {
            let observed_second_moment = covariance_from_kappas(&self.kappas)?;
            *second_moment =
                &*second_moment + &((&observed_second_moment - &*second_moment) * covariance_step);
        }

        // Pure burn-in warms the latent chains and their centered covariance
        // statistics while theta, Omega, Omega_IOV, and sigma remain fixed. Raw
        // covariate phi moments remain unchanged, matching their zero SA gain.
        if parameter_step == 0.0 {
            let observed_second_moment = second_moment_from_etas(&self.etas)?;
            self.iiv_second_moment = &self.iiv_second_moment
                + &((&observed_second_moment - &self.iiv_second_moment) * covariance_step);
            self.finalize_cycle_diagnostics()?;
            return Ok(());
        }

        let pre_update_residual_evidence = self.current_residual_statistics_and_information()?;
        if self.covariate_model.is_some() {
            // The raw first and second phi moments already share the SAEM gain.
            // Keep their centered covariance candidate coherent; exploration
            // robustness is applied later to the accepted Omega iterate rather
            // than introducing a second sufficient-statistic recursion.
            self.iiv_second_moment = self.update_covariate_population_and_recenter_etas()?;
        } else {
            self.update_population_and_recenter_etas()?;
            let observed_second_moment = second_moment_from_etas(&self.etas)?;
            self.iiv_second_moment = &self.iiv_second_moment
                + &((&observed_second_moment - &self.iiv_second_moment) * covariance_step);
        }

        self.update_non_iiv_population(parameter_step)?;
        let (observed_residual_statistics, information_replicates) = pre_update_residual_evidence;
        match information_replicates {
            Ok(replicates) => self.information.update(&replicates, parameter_step),
            Err(reason) => self
                .information
                .mark_unavailable(information_failure_status(reason)),
        }
        let mut residual_diagnostics = self
            .error_models
            .models()
            .iter()
            .map(|(output_index, _)| {
                let statistic = observed_residual_statistics
                    .output(output_index)
                    .unwrap_or_default();
                ResidualCycleDiagnostics {
                    output: self
                        .error_models
                        .output_name(output_index)
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("output_{output_index}")),
                    output_index,
                    prediction_evaluation_count: statistic.observation_count,
                    proportional_floor_count: statistic.proportional_floor_count,
                    non_finite_prediction_count: statistic.non_finite_prediction_count,
                    exponential_domain_violation_count: statistic
                        .exponential_domain_violation_count,
                    update_rejected: false,
                    optimizer_objective: None,
                    optimizer_converged: None,
                    optimizer_iterations: None,
                    optimizer_termination: None,
                    combined_additive_collapse_warning: false,
                }
            })
            .collect::<Vec<_>>();
        let residual_observations = (0..self.error_models.len())
            .map(|output_index| {
                observed_residual_statistics
                    .observations(output_index)
                    .unwrap_or_default()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        self.residual_statistics = self
            .residual_statistics
            .stochastic_update(observed_residual_statistics, parameter_step);

        if self
            .initialization
            .schedule
            .covariance_update_active(self.cycle)
        {
            if self.initialization.omega.has_estimated_entries() {
                let phase = self.initialization.schedule.phase(self.cycle);
                let update = if self.covariate_model.is_some() && phase == SaemPhase::Exploration {
                    self.initialization
                        .omega
                        .update_with_status_and_max_fraction(
                            &self.omega,
                            &self.iiv_second_moment,
                            self.initialization.schedule.minimum_variance,
                            covariate_omega_update_maximum_fraction(true, phase, covariance_step),
                        )?
                } else {
                    // Preserve the established floor-after-interpolation path
                    // for non-covariate IIV and for uncapped covariate smoothing.
                    self.initialization.omega.update_with_status(
                        &self.omega,
                        &self.iiv_second_moment,
                        self.initialization.schedule.minimum_variance,
                    )?
                };
                let status = update.status;
                let update_diagnostics =
                    completed_covariance_update_diagnostics(&self.iiv_second_moment, &update)?;
                self.omega = update.matrix;
                if let Some(diagnostics) = self.cycle_diagnostics.last_mut() {
                    diagnostics.omega_update_rejected = status == CovarianceUpdateStatus::Rejected;
                    diagnostics.omega_update = update_diagnostics;
                }
            }
            if let (Some(specification), Some(omega_iov), Some(second_moment)) = (
                self.initialization.omega_iov.as_ref(),
                self.omega_iov.as_mut(),
                self.iov_second_moment.as_ref(),
            ) {
                if specification.has_estimated_entries() {
                    let update = specification.update_with_status(
                        omega_iov,
                        second_moment,
                        self.initialization.schedule.minimum_iov_variance,
                    )?;
                    let status = update.status;
                    let update_diagnostics =
                        completed_covariance_update_diagnostics(second_moment, &update)?;
                    *omega_iov = update.matrix;
                    if let Some(diagnostics) = self.cycle_diagnostics.last_mut() {
                        diagnostics.omega_iov_update_rejected =
                            status == CovarianceUpdateStatus::Rejected;
                        diagnostics.omega_iov_update = update_diagnostics;
                    }
                }
            }
        }
        for residual_diagnostic in &mut residual_diagnostics {
            let outeq = residual_diagnostic.output_index;
            if !self.error_models.is_estimated(outeq) {
                continue;
            }
            let Some(model) = self.error_models.models().get(outeq).copied() else {
                residual_diagnostic.update_rejected = true;
                continue;
            };
            if let ResidualErrorModel::Combined { a, b } = model {
                match optimize_combined_residual(
                    &residual_observations[outeq],
                    a,
                    b,
                    self.error_models.combined_component_estimated(outeq),
                    self.initialization.schedule.minimum_residual_sigma,
                    self.residual_optimizer_max_iterations as u64,
                ) {
                    Ok(solution) => {
                        let component_estimated =
                            self.error_models.combined_component_estimated(outeq);
                        let additive_sd = applied_combined_residual_component(
                            &self.initialization.schedule,
                            self.cycle,
                            a,
                            solution.additive_sd,
                            component_estimated[0],
                        );
                        let proportional_sd = applied_combined_residual_component(
                            &self.initialization.schedule,
                            self.cycle,
                            b,
                            solution.proportional_sd,
                            component_estimated[1],
                        );
                        residual_diagnostic.combined_additive_collapse_warning =
                            combined_additive_sigma_collapsed(additive_sd, component_estimated[0]);
                        update_estimated_combined_residual_model(
                            &mut self.error_models,
                            outeq,
                            additive_sd,
                            proportional_sd,
                        );
                        residual_diagnostic.optimizer_objective = Some(solution.objective);
                        residual_diagnostic.optimizer_converged = Some(solution.converged);
                        residual_diagnostic.optimizer_iterations = Some(solution.iterations);
                        residual_diagnostic.optimizer_termination = Some(solution.termination);
                    }
                    Err(error) => {
                        residual_diagnostic.update_rejected = true;
                        residual_diagnostic.optimizer_termination = Some(error.to_string());
                    }
                }
                continue;
            }
            if let ResidualErrorModel::CorrelatedCombined { a, b, rho } = model {
                match optimize_correlated_combined_residual(
                    &residual_observations[outeq],
                    a,
                    b,
                    rho,
                    self.error_models
                        .correlated_combined_component_estimated(outeq),
                    self.initialization.schedule.minimum_residual_sigma,
                    self.residual_optimizer_max_iterations as u64,
                ) {
                    Ok(solution) => {
                        let component_estimated = self
                            .error_models
                            .correlated_combined_component_estimated(outeq);
                        let additive_sd = applied_combined_residual_component(
                            &self.initialization.schedule,
                            self.cycle,
                            a,
                            solution.additive_sd,
                            component_estimated[0],
                        );
                        let proportional_sd = applied_combined_residual_component(
                            &self.initialization.schedule,
                            self.cycle,
                            b,
                            solution.proportional_sd,
                            component_estimated[1],
                        );
                        let correlation = applied_correlated_residual_correlation(
                            &self.initialization.schedule,
                            self.cycle,
                            rho,
                            solution.correlation,
                            component_estimated[2],
                        );
                        if !correlation.is_finite() || correlation <= -1.0 || correlation >= 1.0 {
                            residual_diagnostic.update_rejected = true;
                            residual_diagnostic.optimizer_termination = Some(
                                "correlated-combined residual update left (-1, 1)".to_string(),
                            );
                            continue;
                        }
                        residual_diagnostic.combined_additive_collapse_warning =
                            combined_additive_sigma_collapsed(additive_sd, component_estimated[0]);
                        update_estimated_correlated_combined_residual_model(
                            &mut self.error_models,
                            outeq,
                            additive_sd,
                            proportional_sd,
                            correlation,
                        );
                        residual_diagnostic.optimizer_objective = Some(solution.objective);
                        residual_diagnostic.optimizer_converged = Some(solution.converged);
                        residual_diagnostic.optimizer_iterations = Some(solution.iterations);
                        residual_diagnostic.optimizer_termination = Some(solution.termination);
                    }
                    Err(error) => {
                        residual_diagnostic.update_rejected = true;
                        residual_diagnostic.optimizer_termination = Some(error.to_string());
                    }
                }
                continue;
            }
            let Some(candidate_sigma) = self
                .residual_statistics
                .output(outeq)
                .and_then(|statistic| statistic.sigma())
            else {
                residual_diagnostic.update_rejected = true;
                continue;
            };
            let previous_sigma = primary_sigma_parameter(&model);
            let sigma = self.initialization.schedule.guarded_residual_sigma(
                self.cycle,
                previous_sigma,
                candidate_sigma,
            );
            update_estimated_simple_residual_model_with_sigma(&mut self.error_models, outeq, sigma);
        }
        if let Some(diagnostics) = self.cycle_diagnostics.last_mut() {
            diagnostics.residual_diagnostics = residual_diagnostics;
        }
        self.residual_sigmas = primary_sigma_parameters(self.error_models.models());
        self.refresh_subject_scores_from_chains()?;
        self.negative_log_likelihood = negative_log_likelihood(&self.subject_log_likelihoods);
        self.update_iterate_average()?;
        self.finalize_cycle_diagnostics()?;
        Ok(())
    }

    fn update_iterate_average(&mut self) -> Result<()> {
        if self.initialization.schedule.phase(self.cycle) != SaemPhase::Smoothing
            || !matches!(
                self.config.estimator_policy,
                SaemEstimatorPolicy::AveragedIterates { .. }
            )
        {
            return Ok(());
        }
        let population_phi = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        let residual_models = self
            .error_models
            .models()
            .iter()
            .map(|(output_index, model)| (output_index, *model))
            .collect::<Vec<_>>();
        let residual_model_width = self.error_models.models().len();
        let Some(average) = self.iterate_average.as_mut() else {
            self.iterate_average = Some(SaemIterateAverage {
                population_phi,
                covariate_betas: self.covariate_model.as_ref().map(|model| {
                    model
                        .estimates()
                        .iter()
                        .map(|estimate| estimate.estimate())
                        .collect()
                }),
                omega: self.omega.clone(),
                omega_iov: self.omega_iov.clone(),
                residual_model_width,
                residual_models,
                start_cycle: self.cycle,
                count: 1,
            });
            return Ok(());
        };
        let next_count = average.count + 1;
        for (index, value) in population_phi.iter().copied().enumerate() {
            if self.initialization.estimated_parameters[index] {
                average.population_phi[index] =
                    incremental_average(average.population_phi[index], value, next_count);
            }
        }
        if let (Some(average_betas), Some(model)) = (
            average.covariate_betas.as_mut(),
            self.covariate_model.as_ref(),
        ) {
            for (index, estimate) in model.estimates().iter().enumerate() {
                if estimate.estimated() {
                    average_betas[index] =
                        incremental_average(average_betas[index], estimate.estimate(), next_count);
                }
            }
        }
        average_covariance(
            &mut average.omega,
            &self.omega,
            self.initialization.omega.estimated_mask(),
            next_count,
        );
        if let (Some(average_iov), Some(current_iov), Some(specification)) = (
            average.omega_iov.as_mut(),
            self.omega_iov.as_ref(),
            self.initialization.omega_iov.as_ref(),
        ) {
            average_covariance(
                average_iov,
                current_iov,
                specification.estimated_mask(),
                next_count,
            );
        }
        if residual_model_width != average.residual_model_width
            || residual_models.len() != average.residual_models.len()
        {
            anyhow::bail!("residual output declarations changed while accumulating SAEM averages");
        }
        for ((average_output_index, previous), (output_index, current)) in
            average.residual_models.iter_mut().zip(residual_models)
        {
            if *average_output_index != output_index {
                anyhow::bail!(
                    "residual output declarations changed while accumulating SAEM averages"
                );
            }
            let estimated = self.error_models.is_estimated(output_index);
            let components = self.error_models.combined_component_estimated(output_index);
            let correlated_components = self
                .error_models
                .correlated_combined_component_estimated(output_index);
            *previous = average_residual_model(
                *previous,
                current,
                estimated,
                components,
                correlated_components,
                next_count,
            )?;
        }
        average.count = next_count;
        Ok(())
    }

    fn install_iterate_average(&mut self) -> Result<SaemEstimatorMetadata> {
        let policy = self.config.estimator_policy;
        let Some(average) = self.iterate_average.clone() else {
            tracing::info!("averaged SAEM estimate was not available; retaining terminal iterate");
            return Ok(SaemEstimatorMetadata {
                policy,
                ..SaemEstimatorMetadata::default()
            });
        };
        let terminal_phi = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        validate_average_population(&average.population_phi, &self.initialization)?;
        validate_average_covariance(&average.omega, &self.initialization.omega, "Omega")?;
        if let (Some(matrix), Some(specification)) = (
            average.omega_iov.as_ref(),
            self.initialization.omega_iov.as_ref(),
        ) {
            validate_average_covariance(matrix, specification, "Omega_IOV")?;
        }
        validate_average_residuals(
            average.residual_model_width,
            &average.residual_models,
            &self.error_models,
        )?;

        self.population_parameters = population_psi(
            &average.population_phi,
            &self.initialization.parameter_scales,
        )?;
        if let (Some(model), Some(beta_values), Some(old_means)) = (
            self.covariate_model.as_ref(),
            average.covariate_betas.as_ref(),
            self.subject_mu_phi.as_ref(),
        ) {
            let averaged_model = model.with_estimates(beta_values)?;
            let new_rows = averaged_model.subject_population_parameters(
                &average.population_phi,
                &self.initialization.parameter_scales,
            )?;
            let new_means = new_rows
                .iter()
                .map(|row| row.phi().to_vec())
                .collect::<Vec<_>>();
            for (subject_index, chains) in self.etas.iter_mut().enumerate() {
                let old_random = self
                    .initialization
                    .random_effect_indices
                    .iter()
                    .map(|index| old_means[subject_index][*index])
                    .collect::<Vec<_>>();
                let new_random = self
                    .initialization
                    .random_effect_indices
                    .iter()
                    .map(|index| new_means[subject_index][*index])
                    .collect::<Vec<_>>();
                for eta in chains {
                    rebase_eta(eta, &old_random, &new_random)?;
                }
            }
            self.covariate_model = Some(averaged_model);
            self.subject_mu_phi = Some(new_means);
        } else {
            for (eta_index, parameter_index) in self
                .initialization
                .random_effect_indices
                .iter()
                .copied()
                .enumerate()
            {
                let shift = terminal_phi[parameter_index] - average.population_phi[parameter_index];
                for subject_chains in &mut self.etas {
                    for eta in subject_chains {
                        eta[eta_index] += shift;
                    }
                }
            }
        }
        self.omega = average.omega;
        self.omega_iov = average.omega_iov;
        for (output_index, model) in average.residual_models {
            match model {
                ResidualErrorModel::Combined { a, b } => update_estimated_combined_residual_model(
                    &mut self.error_models,
                    output_index,
                    a,
                    b,
                ),
                ResidualErrorModel::CorrelatedCombined { a, b, rho } => {
                    update_estimated_correlated_combined_residual_model(
                        &mut self.error_models,
                        output_index,
                        a,
                        b,
                        rho,
                    )
                }
                ResidualErrorModel::Constant { .. }
                | ResidualErrorModel::Proportional { .. }
                | ResidualErrorModel::Exponential { .. } => {
                    update_estimated_simple_residual_model_with_sigma(
                        &mut self.error_models,
                        output_index,
                        primary_sigma_parameter(&model),
                    )
                }
            }
        }
        self.residual_sigmas = primary_sigma_parameters(self.error_models.models());
        self.refresh_subject_scores_from_chains()?;
        self.negative_log_likelihood = negative_log_likelihood(&self.subject_log_likelihoods);
        tracing::info!(
            start_cycle = average.start_cycle,
            averaged_iterations = average.count,
            "installed averaged SAEM estimate"
        );
        Ok(SaemEstimatorMetadata {
            policy,
            average_applied: true,
            averaging_start_cycle: Some(average.start_cycle),
            averaged_iterations: average.count,
        })
    }

    fn residual_error_estimates(&self) -> Vec<ResidualErrorEstimate> {
        self.error_models
            .models()
            .iter()
            .map(|(output_index, model)| {
                let model = *model;
                let combined_components =
                    self.error_models.combined_component_estimated(output_index);
                let correlated_components = self
                    .error_models
                    .correlated_combined_component_estimated(output_index);
                let is_combined = matches!(model, ResidualErrorModel::Combined { .. });
                let is_correlated = matches!(model, ResidualErrorModel::CorrelatedCombined { .. });
                ResidualErrorEstimate {
                    output: self
                        .error_models
                        .output_name(output_index)
                        .map(str::to_owned)
                        .expect("declared residual models have output names"),
                    output_index,
                    model,
                    estimated: self.error_models.is_estimated(output_index),
                    combined_additive_estimated: if is_combined {
                        Some(combined_components[0])
                    } else {
                        is_correlated.then_some(correlated_components[0])
                    },
                    combined_proportional_estimated: if is_combined {
                        Some(combined_components[1])
                    } else {
                        is_correlated.then_some(correlated_components[1])
                    },
                    correlation_estimated: is_correlated.then_some(correlated_components[2]),
                }
            })
            .collect()
    }

    fn finalize_cycle_diagnostics(&mut self) -> Result<()> {
        let population_parameters = self.population_parameters.clone();
        let omega = self.omega.clone();
        let omega_iov = self.omega_iov.clone();
        let residual_error_estimates = self.residual_error_estimates();
        let conditional_negative_log_likelihood = self.negative_log_likelihood;
        let eta_log_prior = self.subject_log_priors.iter().sum();
        let kappa_log_prior = self.subject_kappa_log_priors.iter().sum();
        let (omega_relative_spd_margin, omega_iov_relative_spd_margin) =
            if self.config.covariance_stability.is_some() {
                let initial_omega = self.initialization.omega.initial();
                let omega_margin = (initial_omega.nrows() > 0)
                    .then(|| relative_spd_margin(&omega, initial_omega))
                    .transpose()?;
                let omega_iov_margin =
                    match (self.initialization.omega_iov.as_ref(), omega_iov.as_ref()) {
                        (Some(specification), Some(matrix))
                            if specification.initial().nrows() > 0 =>
                        {
                            Some(relative_spd_margin(matrix, specification.initial())?)
                        }
                        _ => None,
                    };
                (omega_margin, omega_iov_margin)
            } else {
                (None, None)
            };
        let covariate_betas = self.covariate_model.as_ref().map(|model| {
            model
                .estimates()
                .iter()
                .map(|estimate| estimate.estimate())
                .collect()
        });
        let covariate_beta_estimated = self.covariate_model.as_ref().map(|model| {
            model
                .estimates()
                .iter()
                .map(|estimate| estimate.estimated())
                .collect()
        });
        if let Some(diagnostics) = self.cycle_diagnostics.last_mut() {
            diagnostics.population_parameters = population_parameters;
            diagnostics.omega = omega;
            diagnostics.omega_iov = omega_iov;
            diagnostics.omega_relative_spd_margin = omega_relative_spd_margin;
            diagnostics.omega_iov_relative_spd_margin = omega_iov_relative_spd_margin;
            diagnostics.residual_error_estimates = residual_error_estimates;
            diagnostics.conditional_negative_log_likelihood = conditional_negative_log_likelihood;
            diagnostics.eta_log_prior = eta_log_prior;
            diagnostics.kappa_log_prior = kappa_log_prior;
            diagnostics.covariate_betas = covariate_betas;
            diagnostics.covariate_beta_estimated = covariate_beta_estimated;
        }
        Ok(())
    }

    fn update_population_and_recenter_etas(&mut self) -> Result<Vec<f64>> {
        let old_population_phi = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        let mut new_population_phi = old_population_phi.clone();
        for (parameter_index, parameter_phi) in new_population_phi.iter_mut().enumerate() {
            if self.initialization.estimated_parameters[parameter_index]
                && self
                    .initialization
                    .random_effect_indices
                    .contains(&parameter_index)
            {
                *parameter_phi = self.sufficient_statistics.mean_phi[parameter_index];
            }
        }

        for (eta_index, parameter_index) in self
            .initialization
            .random_effect_indices
            .iter()
            .copied()
            .enumerate()
        {
            let realized_shift =
                new_population_phi[parameter_index] - old_population_phi[parameter_index];
            for subject_chains in &mut self.etas {
                for eta in subject_chains {
                    eta[eta_index] -= realized_shift;
                }
            }
        }
        self.population_parameters =
            population_psi(&new_population_phi, &self.initialization.parameter_scales)?;
        Ok(new_population_phi)
    }

    fn update_covariate_population_and_recenter_etas(&mut self) -> Result<Array2<f64>> {
        let model = self
            .covariate_model
            .as_ref()
            .expect("covariate update requires a resolved model")
            .clone();
        let statistics = self
            .covariate_statistics
            .as_ref()
            .expect("covariate update requires sufficient statistics")
            .clone();
        let q = self.initialization.random_effect_indices.len();
        let old_population_phi = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        let old_subject_mu = self
            .subject_mu_phi
            .as_ref()
            .expect("covariate update requires subject means")
            .clone();

        let free_intercepts = self
            .initialization
            .random_effect_indices
            .iter()
            .copied()
            .filter(|index| self.initialization.estimated_parameters[*index])
            .collect::<Vec<_>>();
        let free_effects = model
            .estimates()
            .iter()
            .enumerate()
            .filter_map(|(index, estimate)| {
                (estimate.estimated()
                    && self
                        .initialization
                        .random_effect_indices
                        .contains(&model.parameter_indices()[index]))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        let width = free_intercepts.len() + free_effects.len();
        let random_row = self
            .initialization
            .random_effect_indices
            .iter()
            .enumerate()
            .map(|(row, parameter)| (*parameter, row))
            .collect::<BTreeMap<_, _>>();
        let mut designs = Vec::with_capacity(model.subject_design().len());
        let mut offsets = Vec::with_capacity(model.subject_design().len());
        for subject in model.subject_design() {
            let mut design = Array2::zeros((q, width));
            let mut offset = vec![0.0; q];
            for (row, parameter_index) in self
                .initialization
                .random_effect_indices
                .iter()
                .copied()
                .enumerate()
            {
                if let Some(column) = free_intercepts
                    .iter()
                    .position(|index| *index == parameter_index)
                {
                    design[[row, column]] = 1.0;
                } else {
                    offset[row] = old_population_phi[parameter_index];
                }
            }
            for (effect_index, value) in subject.values().iter().copied().enumerate() {
                let parameter_index = model.parameter_indices()[effect_index];
                let Some(&row) = random_row.get(&parameter_index) else {
                    continue;
                };
                if let Some(effect_column) =
                    free_effects.iter().position(|index| *index == effect_index)
                {
                    design[[row, free_intercepts.len() + effect_column]] = value;
                } else {
                    offset[row] += value * model.estimates()[effect_index].estimate();
                }
            }
            designs.push(design);
            offsets.push(offset);
        }

        let solution = if width == 0 {
            Vec::new()
        } else {
            solve_covariate_gls(CovariateGlsProblem {
                design: &designs,
                expected_phi: &statistics.expected_phi,
                offset: &offsets,
                omega: &self.omega,
            })?
        };
        let mut new_population_phi = old_population_phi;
        for (column, parameter_index) in free_intercepts.iter().copied().enumerate() {
            new_population_phi[parameter_index] = solution[column];
        }
        let mut beta_values = model
            .estimates()
            .iter()
            .map(|estimate| estimate.estimate())
            .collect::<Vec<_>>();
        for (column, effect_index) in free_effects.iter().copied().enumerate() {
            beta_values[effect_index] = solution[free_intercepts.len() + column];
        }
        let updated_model = model.with_estimates(&beta_values)?;
        let subject_population = updated_model.subject_population_parameters(
            &new_population_phi,
            &self.initialization.parameter_scales,
        )?;
        let new_subject_mu = subject_population
            .iter()
            .map(|row| row.phi().to_vec())
            .collect::<Vec<_>>();
        for (subject_index, subject_chains) in self.etas.iter_mut().enumerate() {
            let old_random = self
                .initialization
                .random_effect_indices
                .iter()
                .map(|index| old_subject_mu[subject_index][*index])
                .collect::<Vec<_>>();
            let new_random = self
                .initialization
                .random_effect_indices
                .iter()
                .map(|index| new_subject_mu[subject_index][*index])
                .collect::<Vec<_>>();
            for eta in subject_chains {
                rebase_eta(eta, &old_random, &new_random)?;
            }
        }
        let subject_mu_random = new_subject_mu
            .iter()
            .map(|mean| {
                self.initialization
                    .random_effect_indices
                    .iter()
                    .map(|index| mean[*index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let candidate = if q == 0 {
            Array2::zeros((0, 0))
        } else {
            subject_centered_omega(
                &statistics.global_second_moment,
                &statistics.expected_phi,
                &subject_mu_random,
            )?
        };
        self.population_parameters =
            population_psi(&new_population_phi, &self.initialization.parameter_scales)?;
        self.subject_mu_phi = Some(new_subject_mu);
        self.covariate_model = Some(updated_model);
        Ok(candidate)
    }

    fn adapt_proposal_step_sizes(&mut self) {
        if self.steps_since_adapt < self.adapt_interval {
            return;
        }

        for parameter_index in 0..self.proposal_step_sizes.len() {
            let proposed = self.adaptation_proposal_counts[parameter_index].max(1);
            let acceptance_rate =
                self.adaptation_accept_counts[parameter_index] as f64 / proposed as f64;
            self.proposal_step_sizes[parameter_index] = adapt_component_step_size(
                self.proposal_step_sizes[parameter_index],
                acceptance_rate,
            );
            self.adaptation_accept_counts[parameter_index] = 0;
            self.adaptation_proposal_counts[parameter_index] = 0;
        }
        for subject_index in 0..self.eta_block_step_sizes.len() {
            let proposed = self.eta_block_adaptation_proposal_counts[subject_index].max(1);
            let acceptance_rate =
                self.eta_block_adaptation_accept_counts[subject_index] as f64 / proposed as f64;
            self.eta_block_step_sizes[subject_index] = adapt_block_step_size(
                self.eta_block_step_sizes[subject_index],
                acceptance_rate,
                ETA_BLOCK_TARGET_ACCEPTANCE,
            );
            self.eta_block_adaptation_accept_counts[subject_index] = 0;
            self.eta_block_adaptation_proposal_counts[subject_index] = 0;
        }
        for subject_index in 0..self.kappa_proposal_step_sizes.len() {
            let proposed = self.kappa_adaptation_proposal_counts[subject_index].max(1);
            let acceptance_rate =
                self.kappa_adaptation_accept_counts[subject_index] as f64 / proposed as f64;
            self.kappa_proposal_step_sizes[subject_index] = adapt_block_step_size(
                self.kappa_proposal_step_sizes[subject_index],
                acceptance_rate,
                KAPPA_BLOCK_TARGET_ACCEPTANCE,
            );
            self.kappa_adaptation_accept_counts[subject_index] = 0;
            self.kappa_adaptation_proposal_counts[subject_index] = 0;
        }
        self.steps_since_adapt = 0;
    }

    fn component_random_walk_eta(
        &mut self,
        current_eta: &[f64],
        parameter_index: usize,
    ) -> Vec<f64> {
        let mut proposed_eta = current_eta.to_vec();
        proposed_eta[parameter_index] +=
            self.proposal_step_sizes[parameter_index] * self.standard_normal();
        proposed_eta
    }

    fn block_random_walk_eta(
        &mut self,
        current_eta: &[f64],
        subject_index: usize,
    ) -> Result<Vec<f64>> {
        let lower = cholesky_lower(&self.omega)?;
        let standard_normals = (0..current_eta.len())
            .map(|_| self.standard_normal())
            .collect::<Vec<_>>();
        correlated_random_walk(
            current_eta,
            &lower,
            &standard_normals,
            self.eta_block_step_sizes[subject_index],
        )
    }

    fn block_random_walk_kappa(
        &mut self,
        current_kappa: &[f64],
        subject_index: usize,
    ) -> Result<Vec<f64>> {
        let omega_iov = self
            .omega_iov
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("kappa proposal requires configured omega_iov"))?;
        let lower = cholesky_lower(omega_iov)?;
        let standard_normals = (0..current_kappa.len())
            .map(|_| self.standard_normal())
            .collect::<Vec<_>>();
        correlated_random_walk(
            current_kappa,
            &lower,
            &standard_normals,
            self.kappa_proposal_step_sizes[subject_index],
        )
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2 = self.rng.random::<f64>();
        (-2.0_f64 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn accept_proposal(&mut self, log_acceptance_ratio: f64) -> bool {
        if !log_acceptance_ratio.is_finite() {
            return false;
        }
        if log_acceptance_ratio >= 0.0 {
            return true;
        }
        self.rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < log_acceptance_ratio
    }

    fn individual_parameters(&self, subject_index: usize, chain_index: usize) -> Vec<f64> {
        self.individual_parameters_from_eta(subject_index, &self.etas[subject_index][chain_index])
            .expect("stored eta should match parameter dimensions")
    }

    fn individual_parameters_from_eta(
        &self,
        subject_index: usize,
        eta: &[f64],
    ) -> Result<Vec<f64>> {
        match self.subject_mu_phi.as_ref() {
            Some(means) => individual_psi_from_subject_mean(
                &means[subject_index],
                &self.initialization.parameter_scales,
                &self.initialization.random_effect_indices,
                eta,
            ),
            None => individual_psi(
                &self.population_parameters,
                &self.initialization.parameter_scales,
                &self.initialization.random_effect_indices,
                eta,
            ),
        }
    }

    fn individual_phi(&self, subject_index: usize, chain_index: usize) -> Result<Vec<f64>> {
        match self.subject_mu_phi.as_ref() {
            Some(means) => individual_phi_from_subject_mean(
                &means[subject_index],
                &self.initialization.random_effect_indices,
                &self.etas[subject_index][chain_index],
            ),
            None => individual_phi(
                &self.population_parameters,
                &self.initialization.parameter_scales,
                &self.initialization.random_effect_indices,
                &self.etas[subject_index][chain_index],
            ),
        }
    }

    fn current_phi_statistics(&self) -> Result<PhiSufficientStatistics> {
        let mut subject_phi = Vec::with_capacity(
            self.initialization.subject_ids.len() * self.initialization.n_chains,
        );
        for subject_index in 0..self.initialization.subject_ids.len() {
            for chain_index in 0..self.initialization.n_chains {
                subject_phi.push(self.individual_phi(subject_index, chain_index)?);
            }
        }
        PhiSufficientStatistics::from_subject_phi(&subject_phi)
    }

    fn current_covariate_statistics(&self) -> Result<CovariateSufficientStatistics> {
        let mut subjects = Vec::with_capacity(self.initialization.subject_ids.len());
        for subject_index in 0..self.initialization.subject_ids.len() {
            let mut chains = Vec::with_capacity(self.initialization.n_chains);
            for chain_index in 0..self.initialization.n_chains {
                let phi = self.individual_phi(subject_index, chain_index)?;
                chains.push(
                    self.initialization
                        .random_effect_indices
                        .iter()
                        .map(|index| phi[*index])
                        .collect(),
                );
            }
            subjects.push(chains);
        }
        CovariateSufficientStatistics::from_subject_chains(&subjects)
    }

    fn current_residual_statistics_and_information(
        &self,
    ) -> Result<(
        ResidualSufficientStatistics,
        std::result::Result<Vec<CompleteDerivative>, String>,
    )> {
        let mut total = ResidualSufficientStatistics::zero(self.error_models.len());
        let layout = self.information.layout();
        let mut replicates = (0..self.initialization.n_chains)
            .map(|_| CompleteDerivative::zero(layout.len()))
            .collect::<Vec<_>>();
        let mut information_error = None;
        // Preserve the established subject-major/chain-minor prediction and
        // accumulation order so this diagnostic cannot alter fit trajectories.
        for subject_index in 0..self.initialization.subject_ids.len() {
            let subject = self.data.subjects()[subject_index];
            for (chain_index, derivative) in replicates.iter_mut().enumerate() {
                if information_error.is_none() {
                    let derivative_result = match self.covariate_model.as_ref() {
                        Some(model) => derivative.add_covariate_population_prior(
                            &self.etas[subject_index][chain_index],
                            &self.omega,
                            &self.initialization.random_effect_indices,
                            model.parameter_indices(),
                            model.subject_design()[subject_index].values(),
                            layout,
                        ),
                        None => derivative.add_population_prior(
                            &self.etas[subject_index][chain_index],
                            &self.omega,
                            &self.initialization.random_effect_indices,
                            layout,
                        ),
                    };
                    if let Err(error) = derivative_result {
                        information_error = Some(error.to_string());
                    }
                }
                if self.omega_iov.is_none() {
                    let parameters = self.individual_parameters(subject_index, chain_index);
                    let predictions = self
                        .equation
                        .estimate_predictions_dense(subject, &parameters)?;
                    total.add_assign(&ResidualSufficientStatistics::from_predictions(
                        &predictions,
                        self.error_models.models(),
                    ));
                    if information_error.is_none() {
                        if let Err(error) =
                            derivative.add_predictions(&predictions, &self.error_models, layout)
                        {
                            information_error = Some(error.to_string());
                        }
                    }
                    continue;
                }
                for (occasion, kappa) in subject
                    .occasions()
                    .iter()
                    .zip(&self.kappas[subject_index][chain_index])
                {
                    if information_error.is_none() {
                        if let Some(omega_iov) = self.omega_iov.as_ref() {
                            if let Err(error) = derivative.add_iov_prior(kappa, omega_iov, layout) {
                                information_error = Some(error.to_string());
                            }
                        }
                    }
                    let parameters = match self.subject_mu_phi.as_ref() {
                        Some(means) => occasion_psi_from_subject_mean(
                            &means[subject_index],
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            &self.etas[subject_index][chain_index],
                            &self.initialization.iov_effect_indices,
                            kappa,
                        ),
                        None => occasion_psi(
                            &self.population_parameters,
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            &self.etas[subject_index][chain_index],
                            &self.initialization.iov_effect_indices,
                            kappa,
                        ),
                    }?;
                    let occasion_subject =
                        Subject::from_occasions(subject.id().to_owned(), vec![occasion.clone()]);
                    let predictions = self
                        .equation
                        .estimate_predictions_dense(&occasion_subject, &parameters)?;
                    total.add_assign(&ResidualSufficientStatistics::from_predictions(
                        &predictions,
                        self.error_models.models(),
                    ));
                    if information_error.is_none() {
                        if let Err(error) =
                            derivative.add_predictions(&predictions, &self.error_models, layout)
                        {
                            information_error = Some(error.to_string());
                        }
                    }
                }
            }
        }
        Ok((
            total,
            match information_error {
                Some(error) => Err(error),
                None => Ok(replicates),
            },
        ))
    }

    #[cfg(test)]
    fn current_residual_statistics(&self) -> Result<ResidualSufficientStatistics> {
        self.current_residual_statistics_and_information()
            .map(|(statistics, _)| statistics)
    }

    fn refresh_subject_scores_from_chains(&mut self) -> Result<()> {
        let n_chains = self.initialization.n_chains as f64;
        let mut subject_log_likelihoods = vec![0.0; self.initialization.subject_ids.len()];
        let mut subject_log_priors = vec![0.0; self.initialization.subject_ids.len()];
        let mut subject_kappa_log_priors = vec![0.0; self.initialization.subject_ids.len()];
        for subject_index in 0..self.initialization.subject_ids.len() {
            for chain_index in 0..self.initialization.n_chains {
                let score = self.score_subject_latents(
                    subject_index,
                    &self.etas[subject_index][chain_index],
                    &self.kappas[subject_index][chain_index],
                )?;
                subject_log_likelihoods[subject_index] += score.log_likelihood / n_chains;
                subject_log_priors[subject_index] += score.eta_log_prior / n_chains;
                subject_kappa_log_priors[subject_index] += score.kappa_log_prior / n_chains;
            }
        }
        self.subject_log_likelihoods = subject_log_likelihoods;
        self.subject_log_priors = subject_log_priors;
        self.subject_kappa_log_priors = subject_kappa_log_priors;
        Ok(())
    }

    fn score_subject_latents(
        &self,
        subject_index: usize,
        eta: &[f64],
        kappas: &[Vec<f64>],
    ) -> Result<SubjectPosteriorScore> {
        self.score_subject_latents_at(subject_index, eta, kappas, None)
    }

    fn non_iiv_coordinate_layout(&self) -> NonIivCoordinateLayout {
        let population_indices = self
            .initialization
            .estimated_parameters
            .iter()
            .enumerate()
            .filter_map(|(index, estimated)| {
                (*estimated && !self.initialization.random_effect_indices.contains(&index))
                    .then_some(index)
            })
            .collect();
        let covariate_indices = self
            .covariate_model
            .as_ref()
            .map(|model| {
                model
                    .estimates()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, estimate)| {
                        (estimate.estimated()
                            && !self
                                .initialization
                                .random_effect_indices
                                .contains(&model.parameter_indices()[index]))
                        .then_some(index)
                    })
                    .collect()
            })
            .unwrap_or_default();
        NonIivCoordinateLayout {
            population_indices,
            covariate_indices,
        }
    }

    fn non_iiv_population_update_active(&self, parameter_step: f64) -> bool {
        let Some(post_burn_start) = self.initialization.schedule.pure_burn_in.checked_add(1) else {
            return false;
        };
        let first_active_cycle = self
            .initialization
            .schedule
            .variance_floor_iterations
            .max(post_burn_start);
        parameter_step.is_finite() && parameter_step > 0.0 && self.cycle >= first_active_cycle
    }

    fn pack_non_iiv_coordinates(&self, layout: &NonIivCoordinateLayout) -> Result<Vec<f64>> {
        let population = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        let mut coordinates = layout
            .population_indices
            .iter()
            .map(|index| population[*index])
            .collect::<Vec<_>>();
        if let Some(model) = self.covariate_model.as_ref() {
            coordinates.extend(
                layout
                    .covariate_indices
                    .iter()
                    .map(|index| model.estimates()[*index].estimate()),
            );
        }
        Ok(coordinates)
    }

    fn non_iiv_candidate_components(
        &self,
        layout: &NonIivCoordinateLayout,
        coordinates: &[f64],
    ) -> Result<NonIivCandidateComponents> {
        if coordinates.len() != layout.len() || coordinates.iter().any(|value| !value.is_finite()) {
            anyhow::bail!("non-IIV population coordinate width or value is invalid");
        }
        let mut population = population_phi(
            &self.population_parameters,
            &self.initialization.parameter_scales,
        )?;
        for (coordinate, parameter_index) in coordinates
            .iter()
            .copied()
            .zip(layout.population_indices.iter().copied())
        {
            population[parameter_index] = coordinate;
        }
        let population_parameters =
            population_psi(&population, &self.initialization.parameter_scales)?;
        if !parameters_are_strictly_in_domain(
            &population_parameters,
            &self.initialization.parameter_scales,
        ) {
            anyhow::bail!("non-IIV population candidate violates its declared parameter domain");
        }

        let covariate_model = match self.covariate_model.as_ref() {
            Some(model) => {
                let mut values = model
                    .estimates()
                    .iter()
                    .map(|estimate| estimate.estimate())
                    .collect::<Vec<_>>();
                for (coordinate, effect_index) in coordinates[layout.population_indices.len()..]
                    .iter()
                    .copied()
                    .zip(layout.covariate_indices.iter().copied())
                {
                    values[effect_index] = coordinate;
                }
                Some(model.with_estimates(&values)?)
            }
            None if layout.covariate_indices.is_empty() => None,
            None => anyhow::bail!("non-IIV covariate coordinates lack a covariate model"),
        };
        let subject_rows = covariate_model
            .as_ref()
            .map(|model| {
                model.subject_population_parameters(
                    &population,
                    &self.initialization.parameter_scales,
                )
            })
            .transpose()?;
        if subject_rows.as_ref().is_some_and(|rows| {
            rows.iter().any(|row| {
                !parameters_are_strictly_in_domain(row.psi(), &self.initialization.parameter_scales)
            })
        }) {
            anyhow::bail!("non-IIV covariate candidate violates a declared parameter domain");
        }
        let subject_means = subject_rows.map(|rows| {
            rows.into_iter()
                .map(|row| row.phi().to_vec())
                .collect::<Vec<_>>()
        });
        Ok((population_parameters, covariate_model, subject_means))
    }

    fn non_iiv_observation_nll(
        &self,
        layout: &NonIivCoordinateLayout,
        coordinates: &[f64],
    ) -> Result<f64> {
        let (population_parameters, _covariate_model, subject_means) =
            self.non_iiv_candidate_components(layout, coordinates)?;
        let chain_count = self.initialization.n_chains;
        if chain_count == 0 {
            anyhow::bail!("non-IIV observation objective requires at least one chain");
        }
        let mut objective = 0.0;
        for subject_index in 0..self.initialization.subject_ids.len() {
            let subject = self.data.subjects()[subject_index];
            let subject_mean = subject_means
                .as_ref()
                .map(|means| means[subject_index].as_slice());
            for chain_index in 0..chain_count {
                let eta = &self.etas[subject_index][chain_index];
                let log_likelihood = if self.omega_iov.is_none() {
                    let parameters = match subject_mean {
                        Some(mean) => individual_psi_from_subject_mean(
                            mean,
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            eta,
                        ),
                        None => individual_psi(
                            &population_parameters,
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            eta,
                        ),
                    }?;
                    parametric_subject_log_likelihood(
                        &self.equation,
                        subject,
                        &parameters,
                        self.error_models.models(),
                    )
                } else {
                    let kappas = &self.kappas[subject_index][chain_index];
                    if kappas.len() != subject.occasions().len() {
                        anyhow::bail!("non-IIV objective kappa/occasion dimension mismatch");
                    }
                    let mut value = 0.0;
                    for (occasion, kappa) in subject.occasions().iter().zip(kappas) {
                        let parameters = match subject_mean {
                            Some(mean) => occasion_psi_from_subject_mean(
                                mean,
                                &self.initialization.parameter_scales,
                                &self.initialization.random_effect_indices,
                                eta,
                                &self.initialization.iov_effect_indices,
                                kappa,
                            ),
                            None => occasion_psi(
                                &population_parameters,
                                &self.initialization.parameter_scales,
                                &self.initialization.random_effect_indices,
                                eta,
                                &self.initialization.iov_effect_indices,
                                kappa,
                            ),
                        }?;
                        let occasion_value = parametric_occasion_log_likelihood(
                            &self.equation,
                            subject.id(),
                            occasion,
                            &parameters,
                            self.error_models.models(),
                        );
                        if !occasion_value.is_finite() {
                            anyhow::bail!("non-IIV observation objective is non-finite");
                        }
                        value += occasion_value;
                    }
                    value
                };
                if !log_likelihood.is_finite() {
                    anyhow::bail!("non-IIV observation objective is non-finite");
                }
                objective -= log_likelihood / chain_count as f64;
            }
        }
        if !objective.is_finite() {
            anyhow::bail!("non-IIV observation objective is non-finite");
        }
        Ok(objective)
    }

    fn update_non_iiv_population(&mut self, parameter_step: f64) -> Result<bool> {
        let layout = self.non_iiv_coordinate_layout();
        if layout.is_empty() || !self.non_iiv_population_update_active(parameter_step) {
            return Ok(false);
        }

        let initial = self.pack_non_iiv_coordinates(&layout)?;
        let initial_objective = self.non_iiv_observation_nll(&layout, &initial)?;
        if !initial_objective.is_finite() {
            anyhow::bail!("current non-IIV observation objective is non-finite");
        }

        let mut simplex = Vec::with_capacity(initial.len() + 1);
        simplex.push(initial.clone());
        for coordinate in 0..initial.len() {
            let mut point = initial.clone();
            point[coordinate] += 0.1 * initial[coordinate].abs().max(1.0);
            simplex.push(point);
        }
        let solver = NelderMead::new(simplex).with_sd_tolerance(NON_IIV_OPTIMIZER_SD_TOLERANCE)?;
        let execution = Executor::new(
            NonIivPopulationCost {
                state: self,
                layout: &layout,
            },
            solver,
        )
        .configure(|state| state.max_iters(NON_IIV_OPTIMIZER_MAX_ITERATIONS))
        .run();
        let result = match execution {
            Ok(result) => result,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "Non-IIV population optimizer failed; retaining current state"
                );
                return Ok(false);
            }
        };
        let Some(candidate) = result.state.best_param.as_ref() else {
            return Ok(false);
        };
        let candidate_objective = match self.non_iiv_observation_nll(&layout, candidate) {
            Ok(value) if value.is_finite() => value,
            _ => return Ok(false),
        };
        if !non_iiv_candidate_improves(initial_objective, candidate_objective) {
            return Ok(false);
        }

        let applied = initial
            .iter()
            .zip(candidate)
            .map(|(current, target)| current + parameter_step * (target - current))
            .collect::<Vec<_>>();
        match self.non_iiv_observation_nll(&layout, &applied) {
            Ok(value) if value.is_finite() => {}
            _ => return Ok(false),
        }

        let (population_parameters, covariate_model, subject_means) =
            self.non_iiv_candidate_components(&layout, &applied)?;
        self.population_parameters = population_parameters;
        self.covariate_model = covariate_model;
        self.subject_mu_phi = subject_means;
        Ok(true)
    }

    fn score_subject_latents_at(
        &self,
        subject_index: usize,
        eta: &[f64],
        kappas: &[Vec<f64>],
        candidate: Option<&DiagnosticCandidate>,
    ) -> Result<SubjectPosteriorScore> {
        if eta.len() != self.initialization.random_effect_indices.len() {
            anyhow::bail!(
                "eta has {} values but there are {} random effects",
                eta.len(),
                self.initialization.random_effect_indices.len()
            );
        }

        let subject = self.data.subjects()[subject_index];
        let population_parameters = candidate
            .map_or(self.population_parameters.as_slice(), |value| {
                value.population_parameters.as_slice()
            });
        let omega = candidate.map_or(&self.omega, |value| &value.omega);
        let omega_iov = candidate.map_or(self.omega_iov.as_ref(), |value| value.omega_iov.as_ref());
        let error_models = candidate.map_or(&self.error_models, |value| &value.error_models);
        let candidate_covariates = candidate
            .and_then(|value| value.covariate_model.as_ref())
            .or(self.covariate_model.as_ref());
        let calculated_subject_mu = if candidate.is_some() {
            candidate_covariates
                .map(|model| {
                    let phi = population_phi(
                        population_parameters,
                        &self.initialization.parameter_scales,
                    )?;
                    Ok::<_, anyhow::Error>(
                        model.subject_population_parameters(
                            &phi,
                            &self.initialization.parameter_scales,
                        )?[subject_index]
                            .phi()
                            .to_vec(),
                    )
                })
                .transpose()?
        } else {
            None
        };
        let subject_mu = calculated_subject_mu.as_deref().or_else(|| {
            self.subject_mu_phi
                .as_ref()
                .map(|means| means[subject_index].as_slice())
        });
        let eta_log_prior = eta_log_prior_from_omega(eta, omega)?;
        if omega_iov.is_none() {
            let parameters = match subject_mu {
                Some(mean) => individual_psi_from_subject_mean(
                    mean,
                    &self.initialization.parameter_scales,
                    &self.initialization.random_effect_indices,
                    eta,
                ),
                None => individual_psi(
                    population_parameters,
                    &self.initialization.parameter_scales,
                    &self.initialization.random_effect_indices,
                    eta,
                ),
            }?;
            return Ok(SubjectPosteriorScore {
                log_likelihood: parametric_subject_log_likelihood(
                    &self.equation,
                    subject,
                    &parameters,
                    error_models.models(),
                ),
                eta_log_prior,
                kappa_log_prior: 0.0,
            });
        }

        if kappas.len() != subject.occasions().len() {
            anyhow::bail!(
                "subject '{}' has {} occasions but {} kappa states",
                subject.id(),
                subject.occasions().len(),
                kappas.len()
            );
        }
        let omega_iov = omega_iov.expect("checked above");
        let mut log_likelihood = 0.0;
        let mut kappa_log_prior = 0.0;
        for (occasion, kappa) in subject.occasions().iter().zip(kappas) {
            let parameters = match subject_mu {
                Some(mean) => occasion_psi_from_subject_mean(
                    mean,
                    &self.initialization.parameter_scales,
                    &self.initialization.random_effect_indices,
                    eta,
                    &self.initialization.iov_effect_indices,
                    kappa,
                ),
                None => occasion_psi(
                    population_parameters,
                    &self.initialization.parameter_scales,
                    &self.initialization.random_effect_indices,
                    eta,
                    &self.initialization.iov_effect_indices,
                    kappa,
                ),
            }?;
            let occasion_log_likelihood = parametric_occasion_log_likelihood(
                &self.equation,
                subject.id(),
                occasion,
                &parameters,
                error_models.models(),
            );
            if !occasion_log_likelihood.is_finite() {
                log_likelihood = f64::NEG_INFINITY;
            } else if log_likelihood.is_finite() {
                log_likelihood += occasion_log_likelihood;
            }
            kappa_log_prior += eta_log_prior_from_omega(kappa, omega_iov)?;
        }

        Ok(SubjectPosteriorScore {
            log_likelihood,
            eta_log_prior,
            kappa_log_prior,
        })
    }

    fn proposal_log_acceptance_ratio(
        &self,
        subject_index: usize,
        chain_index: usize,
        proposed_eta: &[f64],
    ) -> Result<f64> {
        let current = self.score_subject_latents(
            subject_index,
            &self.etas[subject_index][chain_index],
            &self.kappas[subject_index][chain_index],
        )?;
        let proposed = self.score_subject_latents(
            subject_index,
            proposed_eta,
            &self.kappas[subject_index][chain_index],
        )?;
        Ok(current.log_acceptance_ratio(proposed))
    }

    fn kappa_proposal_log_acceptance_ratio(
        &self,
        subject_index: usize,
        chain_index: usize,
        occasion_index: usize,
        proposed_kappa: &[f64],
    ) -> Result<f64> {
        let current_kappas = &self.kappas[subject_index][chain_index];
        let current = self.score_subject_latents(
            subject_index,
            &self.etas[subject_index][chain_index],
            current_kappas,
        )?;
        let mut proposed_kappas = current_kappas.clone();
        proposed_kappas[occasion_index] = proposed_kappa.to_vec();
        let proposed = self.score_subject_latents(
            subject_index,
            &self.etas[subject_index][chain_index],
            &proposed_kappas,
        )?;
        Ok(current.log_acceptance_ratio(proposed))
    }

    fn markov_variance_diagnostics(
        &self,
        estimator: &SaemEstimatorMetadata,
        information: &InformationDiagnostics,
    ) -> MarkovSimulationVarianceDiagnostics {
        self.markov_variance_diagnostics_with_seed(estimator, information, None, None)
    }

    /// Frozen-kernel diagnostic with an optional deterministic seed override.
    ///
    /// The override gives each operational checkpoint its own deterministic
    /// stream; `None` preserves the exact
    /// post-fit path seeded by the diagnostic configuration.
    fn markov_variance_diagnostics_with_seed(
        &self,
        estimator: &SaemEstimatorMetadata,
        information: &InformationDiagnostics,
        seed_override: Option<u64>,
        candidate: Option<&DiagnosticCandidate>,
    ) -> MarkovSimulationVarianceDiagnostics {
        let Some(config) = self.config.markov_simulation_variance else {
            return MarkovSimulationVarianceDiagnostics::disabled();
        };
        let diagnostic_seed = seed_override.unwrap_or(config.seed);
        let cd = config.diagnostic_chains;
        let cf = self.initialization.n_chains;
        let mut diagnostic = MarkovSimulationVarianceDiagnostics {
            config: Some(config),
            coordinates: information.coordinates.clone(),
            chain_count: cd,
            n_avg: estimator.averaged_iterations,
            chains: Vec::new(),
            grand_score_mean: Vec::new(),
            lambda: Vec::new(),
            lambda_status: MarkovSimulationVarianceStatus::AssumptionsUnverified,
            xi: Vec::new(),
            xi_status: MarkovSimulationVarianceStatus::AssumptionsUnverified,
            simulation_covariance: Vec::new(),
            simulation_covariance_status: MarkovSimulationVarianceStatus::AssumptionsUnverified,
            status: MarkovSimulationVarianceStatus::AssumptionsUnverified,
            assumptions: MARKOV_VARIANCE_ASSUMPTIONS.into(),
            rank_diagnostics: RankMixingDiagnostics {
                diagnostic_chains: cd,
                draws_per_chain: config.draws_per_chain,
                original_chains: cf,
                traces: Vec::new(),
                lrv_per_chain: Vec::new(),
                lrv_chain_statuses: Vec::new(),
                diagnostic_mean_lrv: None,
                operational_lrv: None,
                max_trace_bytes: 0,
                accounted_peak_trace_bytes_required: 0,
                accounted_peak_trace_bytes_used: 0,
                worst_rhat: None,
                min_bulk_ess: None,
                min_avg_ess_per_split_chain: None,
                assumptions: MARKOV_VARIANCE_ASSUMPTIONS.into(),
                status: RankDiagnosticStatus::Disabled,
            },
        };
        let width = information.coordinates.len();
        let information_eligible = estimator.average_applied
            && matches!(information.status, InformationStatus::Available)
            && width > 0;
        let observed_information = if information_eligible {
            match matrix_from_rows(&information.observed_information, width) {
                Ok(matrix) => Some(matrix),
                Err(_) => {
                    diagnostic.xi_status = MarkovSimulationVarianceStatus::CoordinateMismatch;
                    None
                }
            }
        } else {
            None
        };
        if self.initialization.random_effect_indices.is_empty()
            && self.initialization.iov_effect_indices.is_empty()
        {
            let zero = Array2::zeros((width, width));
            diagnostic.lambda = rows(&zero);
            diagnostic.lambda_status = MarkovSimulationVarianceStatus::ExactZeroNoLatentState;
            diagnostic.xi = rows(&zero);
            diagnostic.xi_status = MarkovSimulationVarianceStatus::ExactZeroNoLatentState;
            diagnostic.simulation_covariance = rows(&zero);
            diagnostic.simulation_covariance_status =
                MarkovSimulationVarianceStatus::ExactZeroNoLatentState;
            diagnostic.status = MarkovSimulationVarianceStatus::ExactZeroNoLatentState;
            diagnostic.rank_diagnostics.status = RankDiagnosticStatus::NoLatent;
            diagnostic
                .rank_diagnostics
                .lrv_chain_statuses
                .fill(RankDiagnosticStatus::NoLatent);
            diagnostic.rank_diagnostics.max_trace_bytes = config.max_trace_bytes;
            return diagnostic;
        }

        // ── Pre-execution byte-cap check (checked) ───────────────────────
        let trace_shape = self
            .initialization
            .random_effect_indices
            .len()
            .checked_mul(self.initialization.subject_ids.len())
            .and_then(|n_eta| {
                self.initialization
                    .occasion_counts
                    .iter()
                    .try_fold(0usize, |total, count| total.checked_add(*count))
                    .and_then(|occasions| {
                        occasions
                            .checked_mul(self.initialization.iov_effect_indices.len())
                            .and_then(|n_kappa| width.checked_add(n_eta)?.checked_add(n_kappa))
                    })
            });
        let Some(n_traces) = trace_shape else {
            mark_diagnostic_failure(
                &mut diagnostic,
                RankDiagnosticStatus::TraceMemoryAccountingOverflow,
                MarkovSimulationVarianceStatus::TraceMemoryAccountingOverflow,
            );
            diagnostic.rank_diagnostics.max_trace_bytes = config.max_trace_bytes;
            return diagnostic;
        };
        // Deterministic requested-capacity accounting. `traces` is nested
        // coordinate-major storage, so its heap-resident Vec headers count in
        // addition to every f64 leaf payload. The peak adds the larger of:
        // (a) one nested draw-major score view, or (b) a conservative upper
        // bound for the live rank/folding/ESS workspaces. The latter is eight
        // payload-widths per retained draw (including the 24-byte ranked tuple)
        // plus sixteen Vec headers per chain. This upper-bounds all capacities
        // explicitly requested by the current rank helpers; allocator metadata
        // and allocator size-class rounding are intentionally not claimed.
        let vec_header = std::mem::size_of::<Vec<f64>>();
        let f64_bytes = std::mem::size_of::<f64>();
        let accounted = cd
            .checked_mul(config.draws_per_chain)
            .and_then(|samples_per_coordinate| {
                samples_per_coordinate
                    .checked_mul(n_traces)
                    .and_then(|values| values.checked_mul(f64_bytes))
                    .and_then(|leaf_payload| {
                        n_traces
                            .checked_mul(cd)
                            .and_then(|headers| headers.checked_mul(vec_header))
                            .and_then(|leaf_headers| leaf_payload.checked_add(leaf_headers))
                    })
                    .and_then(|bytes| {
                        n_traces
                            .checked_mul(vec_header)
                            .and_then(|middle_headers| bytes.checked_add(middle_headers))
                    })
                    .and_then(|persistent_bytes| {
                        config
                            .draws_per_chain
                            .checked_mul(width)
                            .and_then(|values| values.checked_mul(f64_bytes))
                            .and_then(|payload| {
                                config
                                    .draws_per_chain
                                    .checked_mul(vec_header)
                                    .and_then(|headers| payload.checked_add(headers))
                            })
                            .and_then(|score_transient_bytes| {
                                samples_per_coordinate
                                    .checked_mul(8 * f64_bytes)
                                    .and_then(|payload| {
                                        cd.checked_mul(16)
                                            .and_then(|headers| headers.checked_mul(vec_header))
                                            .and_then(|headers| payload.checked_add(headers))
                                    })
                                    .and_then(|rank_transient_bytes| {
                                        persistent_bytes
                                            .checked_add(
                                                score_transient_bytes.max(rank_transient_bytes),
                                            )
                                            .map(|required_bytes| {
                                                (
                                                    persistent_bytes,
                                                    score_transient_bytes,
                                                    required_bytes,
                                                )
                                            })
                                    })
                            })
                    })
            });
        let Some((persistent_bytes, score_transient_bytes, required_bytes)) = accounted else {
            mark_diagnostic_failure(
                &mut diagnostic,
                RankDiagnosticStatus::TraceMemoryAccountingOverflow,
                MarkovSimulationVarianceStatus::TraceMemoryAccountingOverflow,
            );
            diagnostic.rank_diagnostics.max_trace_bytes = config.max_trace_bytes;
            return diagnostic;
        };
        diagnostic
            .rank_diagnostics
            .accounted_peak_trace_bytes_required = required_bytes;
        diagnostic.rank_diagnostics.max_trace_bytes = config.max_trace_bytes;
        if required_bytes > config.max_trace_bytes {
            mark_diagnostic_failure(
                &mut diagnostic,
                RankDiagnosticStatus::TraceByteCapExceeded,
                MarkovSimulationVarianceStatus::InvalidConfiguration(format!(
                    "diagnostic trace accounted peak requires {required_bytes} bytes, exceeding cap {}",
                    config.max_trace_bytes
                )),
            );
            return diagnostic;
        }
        diagnostic.rank_diagnostics.lrv_per_chain = vec![None; cd];
        diagnostic.rank_diagnostics.lrv_chain_statuses =
            vec![RankDiagnosticStatus::Unavailable; cd];

        // ── Trace coordinate metadata ─────────────────────────────────────
        let mut trace_coords: Vec<DiagnosticTraceCoordinate> = Vec::with_capacity(n_traces);
        for coord in &information.coordinates {
            trace_coords.push(DiagnosticTraceCoordinate::Score {
                index: coord.index,
                name: coord.name.clone(),
                kind: coord.kind.clone(),
            });
        }
        for subject_id in &self.initialization.subject_ids {
            for (eff_idx, name) in self.initialization.random_effect_names.iter().enumerate() {
                trace_coords.push(DiagnosticTraceCoordinate::Eta {
                    subject: subject_id.clone(),
                    effect_index: eff_idx,
                    effect_name: name.clone(),
                });
            }
        }
        if !self.initialization.iov_effect_indices.is_empty() {
            for (subject_idx, subject_id) in self.initialization.subject_ids.iter().enumerate() {
                for occasion in self.data.subjects()[subject_idx].occasions() {
                    for (eff_idx, name) in self.initialization.iov_effect_names.iter().enumerate() {
                        trace_coords.push(DiagnosticTraceCoordinate::Kappa {
                            subject: subject_id.clone(),
                            occasion_index: occasion.index(),
                            effect_index: eff_idx,
                            effect_name: name.clone(),
                        });
                    }
                }
            }
        }

        // ── Cd < 2 → still execute frozen chains, LRV, and Xi ────────────
        // Only per-trace rank diagnostics are unavailable (TooFewChains).
        let rank_possible = cd >= 2;

        // ── Fresh prior-drawn chains ──────────────────────────────────────
        let omega = candidate.map_or(&self.omega, |value| &value.omega);
        let omega_iov = candidate.map_or(self.omega_iov.as_ref(), |value| value.omega_iov.as_ref());
        let omega_lower = match cholesky_lower(omega) {
            Ok(lower) => lower,
            Err(_) => {
                mark_diagnostic_failure(
                    &mut diagnostic,
                    RankDiagnosticStatus::InvalidVariance,
                    MarkovSimulationVarianceStatus::Indefinite,
                );
                return diagnostic;
            }
        };
        let iov_lower = if self.initialization.iov_effect_indices.is_empty() {
            None
        } else {
            match omega_iov.map(cholesky_lower) {
                Some(Ok(lower)) => Some(lower),
                Some(Err(_)) | None => {
                    mark_diagnostic_failure(
                        &mut diagnostic,
                        RankDiagnosticStatus::InvalidVariance,
                        MarkovSimulationVarianceStatus::Indefinite,
                    );
                    return diagnostic;
                }
            }
        };

        // Canonical storage: [score_0..score_{w-1}, eta_0.., kappa_0..].
        // A draw-major score view is created one chain at a time for LRV and
        // released before the next chain.
        let mut traces: Vec<Vec<Vec<f64>>> = (0..n_traces)
            .map(|_| vec![Vec::with_capacity(config.draws_per_chain); cd])
            .collect();
        diagnostic.rank_diagnostics.accounted_peak_trace_bytes_used = persistent_bytes;
        let score_eligible =
            width > 0 && matches!(information.status, InformationStatus::Available);

        // Initialize Cd independent diagnostic chains with domain-separated seeds.
        // Seed derivation: per-chain seed = base.wrapping_add(i).wrapping_mul(GOLDEN_RATIO)
        // where GOLDEN_RATIO = 0x9E3779B97F4A7C15 (2^64 / φ) and base is the
        // configured diagnostic seed or the deterministic checkpoint override.
        let mut chain_states: Vec<FrozenDiagnosticState> = (0..cd)
            .map(|chain| {
                let chain_seed = diagnostic_seed
                    .wrapping_add(chain as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15);
                let mut chain_rng = StdRng::seed_from_u64(chain_seed);
                FrozenDiagnosticState {
                    etas: self.draw_prior_etas(&omega_lower, &mut chain_rng),
                    kappas: self.draw_prior_kappas(iov_lower.as_deref(), &mut chain_rng),
                }
            })
            .collect();
        // Independent RNG streams for transitions (offset by +1 to separate
        // from prior-initialization streams).
        let mut chain_rngs: Vec<StdRng> = (0..cd)
            .map(|chain| {
                let chain_seed = diagnostic_seed
                    .wrapping_add(chain as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(1);
                StdRng::seed_from_u64(chain_seed)
            })
            .collect();
        let mut chain_counts = vec![(0usize, 0usize, 0usize); cd];

        // ── Warmup ────────────────────────────────────────────────────────
        for _ in 0..config.warmup_transitions {
            for chain in 0..cd {
                let mut single = [chain_counts[chain]];
                if self
                    .frozen_diagnostic_transition(
                        &mut chain_states[chain],
                        &mut chain_rngs[chain],
                        &mut single,
                        candidate,
                    )
                    .is_err()
                {
                    mark_diagnostic_failure(
                        &mut diagnostic,
                        RankDiagnosticStatus::Unavailable,
                        MarkovSimulationVarianceStatus::UnsupportedScore(
                            "frozen diagnostic warmup transition failed".into(),
                        ),
                    );
                    return diagnostic;
                }
                chain_counts[chain] = single[0];
            }
        }
        begin_retained_transition_accounting(&mut chain_counts);

        // ── Single retained-draw pass: transition → collect traces ──────
        for _ in 0..config.draws_per_chain {
            for chain in 0..cd {
                let mut single = [chain_counts[chain]];
                if self
                    .frozen_diagnostic_transition(
                        &mut chain_states[chain],
                        &mut chain_rngs[chain],
                        &mut single,
                        candidate,
                    )
                    .is_err()
                {
                    mark_diagnostic_failure(
                        &mut diagnostic,
                        RankDiagnosticStatus::Unavailable,
                        MarkovSimulationVarianceStatus::UnsupportedScore(
                            "frozen retained diagnostic transition failed".into(),
                        ),
                    );
                    return diagnostic;
                }
                chain_counts[chain] = single[0];

                // Score failure never discards independently valid latent draws.
                let score = if score_eligible {
                    match self.frozen_complete_score(&chain_states[chain], 0, candidate) {
                        Ok(values) if values.len() == width => Some(values),
                        Ok(_) | Err(_) => None,
                    }
                } else {
                    None
                };
                for coord_idx in 0..width {
                    traces[coord_idx][chain]
                        .push(score.as_ref().map_or(f64::NAN, |values| values[coord_idx]));
                }

                // Collect eta coordinates: subject-major, coordinate-major.
                let mut trace_idx = width;
                for subject_etas in &chain_states[chain].etas {
                    for eta_coord in &subject_etas[0] {
                        traces[trace_idx][chain].push(*eta_coord);
                        trace_idx += 1;
                    }
                }

                // Collect kappa coordinates.
                for subject_kappas in &chain_states[chain].kappas {
                    for kappa_vec in &subject_kappas[0] {
                        for kappa_coord in kappa_vec {
                            traces[trace_idx][chain].push(*kappa_coord);
                            trace_idx += 1;
                        }
                    }
                }
            }
        }

        // Preserve the raw grand complete-score mean used by the invariant
        // stationarity diagnostic. Any non-finite score leaves it unavailable.
        if score_eligible {
            let denominator = (cd * config.draws_per_chain) as f64;
            let means = (0..width)
                .map(|coordinate| {
                    traces[coordinate].iter().flatten().copied().sum::<f64>() / denominator
                })
                .collect::<Vec<_>>();
            if means.iter().all(|value| value.is_finite()) {
                diagnostic.grand_score_mean = means;
            }
        }

        // ── Per-chain score LRV from transient draw-major views ─────────
        let mut lrv_matrices: Vec<Option<Array2<f64>>> = Vec::with_capacity(cd);
        for chain in 0..cd {
            let (proposals, accepts, state_changes) = chain_counts[chain];
            let score_view = (0..config.draws_per_chain)
                .map(|draw| (0..width).map(|coord| traces[coord][chain][draw]).collect())
                .collect::<Vec<Vec<f64>>>();
            diagnostic.rank_diagnostics.accounted_peak_trace_bytes_used = persistent_bytes
                .checked_add(score_transient_bytes)
                .unwrap_or(required_bytes);
            let lrv_result = if score_eligible {
                match lugsail_batch_means(&score_view, config.batch_size, config.lugsail) {
                    Ok(value) => Some(value),
                    Err(_) => {
                        diagnostic.xi_status = MarkovSimulationVarianceStatus::UnsupportedScore(
                            "per-chain score LRV failed".into(),
                        );
                        None
                    }
                }
            } else {
                None
            };
            if let Some((coarse, fine, lrv)) = lrv_result {
                let classification = classify_psd(&lrv);
                let lrv_status = markov_matrix_status(classification);
                diagnostic
                    .chains
                    .push(MarkovSimulationVarianceChainDiagnostics {
                        chain,
                        bm_batch: rows(&coarse),
                        bm_batch_over_r: rows(&fine),
                        lugsail_lrv: rows(&lrv),
                        status: lrv_status,
                        proposals,
                        accepts,
                        state_changes,
                    });
                diagnostic.rank_diagnostics.lrv_per_chain[chain] = Some(rows(&lrv));
                diagnostic.rank_diagnostics.lrv_chain_statuses[chain] = match classification {
                    MatrixClassification::EligiblePsd => RankDiagnosticStatus::Available,
                    MatrixClassification::NonFinite => RankDiagnosticStatus::NonFiniteDraws,
                    MatrixClassification::NonSymmetric | MatrixClassification::Indefinite => {
                        RankDiagnosticStatus::InvalidVariance
                    }
                };
                lrv_matrices.push(Some(lrv));
            } else {
                diagnostic
                    .chains
                    .push(MarkovSimulationVarianceChainDiagnostics {
                        chain,
                        bm_batch: Vec::new(),
                        bm_batch_over_r: Vec::new(),
                        lugsail_lrv: Vec::new(),
                        status: MarkovSimulationVarianceStatus::UnsupportedScore(
                            "complete-score trace or information unavailable".into(),
                        ),
                        proposals,
                        accepts,
                        state_changes,
                    });
                diagnostic.rank_diagnostics.lrv_per_chain[chain] = None;
                diagnostic.rank_diagnostics.lrv_chain_statuses[chain] =
                    RankDiagnosticStatus::ScoreUnavailable;
                lrv_matrices.push(None);
            }
        }

        let stuck_chain = chain_counts
            .iter()
            .enumerate()
            .find(|(_, count)| count.2 == 0)
            .map(|(chain, _)| chain);

        // Aggregate only when every chain has an eligible score LRV.
        let all_lrvs_available = lrv_matrices.len() == cd
            && lrv_matrices.iter().all(Option::is_some)
            && diagnostic
                .rank_diagnostics
                .lrv_chain_statuses
                .iter()
                .all(|status| matches!(status, RankDiagnosticStatus::Available));
        if all_lrvs_available {
            let mut lrv_sum = Array2::zeros((width, width));
            for lrv in &lrv_matrices {
                lrv_sum += lrv
                    .as_ref()
                    .expect("all per-chain LRV matrices were checked available");
            }
            let (diag_mean, operational) = scale_lrv_sum(&lrv_sum, cd, cf);
            diagnostic.rank_diagnostics.diagnostic_mean_lrv = Some(rows(&diag_mean));
            diagnostic.lambda = rows(&diag_mean);
            diagnostic.lambda_status = markov_matrix_status(classify_psd(&diag_mean));

            // Cd != Cf is intentional: operational scale is Σ/(Cd*Cf).
            diagnostic.rank_diagnostics.operational_lrv = Some(rows(&operational));
            if let Some(observed_information) = observed_information.as_ref() {
                match transform_simulation_variance(
                    observed_information,
                    &operational,
                    estimator.averaged_iterations,
                ) {
                    Ok((xi, covariance)) => {
                        diagnostic.xi = rows(&xi);
                        diagnostic.xi_status = markov_matrix_status(classify_psd(&xi));
                        diagnostic.simulation_covariance = rows(&covariance);
                        diagnostic.simulation_covariance_status =
                            markov_matrix_status(classify_psd(&covariance));
                    }
                    Err(_) => {
                        diagnostic.xi_status = MarkovSimulationVarianceStatus::NonFinite;
                        diagnostic.simulation_covariance_status =
                            MarkovSimulationVarianceStatus::NonFinite;
                    }
                }
            } else {
                diagnostic.xi_status = MarkovSimulationVarianceStatus::InformationUnavailable(
                    format!("{:?}", information.status),
                );
                diagnostic.simulation_covariance_status = diagnostic.xi_status.clone();
            }
        } else {
            let failure = if !score_eligible {
                MarkovSimulationVarianceStatus::InformationUnavailable(format!(
                    "{:?}",
                    information.status
                ))
            } else {
                diagnostic
                    .chains
                    .iter()
                    .map(|chain| &chain.status)
                    .find(|status| {
                        !matches!(
                            status,
                            MarkovSimulationVarianceStatus::AssumptionsUnverified
                        )
                    })
                    .cloned()
                    .unwrap_or_else(|| {
                        MarkovSimulationVarianceStatus::UnsupportedScore(
                            "one or more configured diagnostic-chain score LRVs failed".into(),
                        )
                    })
            };
            diagnostic.lambda_status = failure.clone();
            diagnostic.xi_status = failure.clone();
            diagnostic.simulation_covariance_status = failure;
        }

        // ── Rank/mixing diagnostics from traces ─────────────────────────
        // The prechecked rank workspace is the accounted peak whenever rank
        // diagnostics execute; no allocator-specific byte claim is made.
        if rank_possible {
            diagnostic.rank_diagnostics.accounted_peak_trace_bytes_used = required_bytes;
            diagnostic.rank_diagnostics.traces =
                self.rank_diagnostics_from_traces(cd, &traces, &trace_coords);
        } else {
            diagnostic.rank_diagnostics.traces = trace_coords
                .iter()
                .map(|coord| RankMixingDiagnostic {
                    trace: coord.clone(),
                    rank_rhat: None,
                    rank_rhat_status: RankDiagnosticStatus::TooFewChains,
                    folded_rhat: None,
                    folded_rhat_status: RankDiagnosticStatus::TooFewChains,
                    max_rhat: None,
                    max_rhat_status: RankDiagnosticStatus::TooFewChains,
                    bulk_ess: None,
                    bulk_ess_status: RankDiagnosticStatus::TooFewChains,
                    avg_ess_per_split_chain: None,
                    tau: None,
                    status: RankDiagnosticStatus::TooFewChains,
                })
                .collect();
        }

        // ── Aggregate per-coordinate worst/min across traces ────────────
        diagnostic.rank_diagnostics.worst_rhat =
            worst_valid_max_rhat(&diagnostic.rank_diagnostics.traces);
        diagnostic.rank_diagnostics.min_bulk_ess = diagnostic
            .rank_diagnostics
            .traces
            .iter()
            .filter_map(|t| t.bulk_ess)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        diagnostic.rank_diagnostics.min_avg_ess_per_split_chain = diagnostic
            .rank_diagnostics
            .traces
            .iter()
            .filter_map(|t| t.avg_ess_per_split_chain)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Aggregate status: ineligible if any coordinate or LRV is non-available.
        let any_coord_non_available = diagnostic
            .rank_diagnostics
            .traces
            .iter()
            .any(|t| !matches!(t.status, RankDiagnosticStatus::Available));
        let any_lrv_non_available = diagnostic
            .rank_diagnostics
            .lrv_chain_statuses
            .iter()
            .any(|s| !matches!(s, RankDiagnosticStatus::Available));
        if rank_possible
            && (any_coord_non_available || any_lrv_non_available || stuck_chain.is_some())
        {
            diagnostic.rank_diagnostics.status = if diagnostic
                .rank_diagnostics
                .traces
                .iter()
                .any(|trace| matches!(trace.status, RankDiagnosticStatus::Available))
            {
                RankDiagnosticStatus::PartialAvailability
            } else {
                RankDiagnosticStatus::Unavailable
            };
        } else if !rank_possible {
            diagnostic.rank_diagnostics.status = RankDiagnosticStatus::TooFewChains;
        } else {
            diagnostic.rank_diagnostics.status = RankDiagnosticStatus::Available;
        }

        // ── Final aggregate markov status ───────────────────────────────
        diagnostic.status = if let Some(chain) = stuck_chain {
            MarkovSimulationVarianceStatus::StuckChain { chain }
        } else if !estimator.average_applied {
            MarkovSimulationVarianceStatus::AverageNotApplied
        } else if !matches!(information.status, InformationStatus::Available) {
            MarkovSimulationVarianceStatus::InformationUnavailable(format!(
                "{:?}",
                information.status
            ))
        } else {
            diagnostic
                .chains
                .iter()
                .map(|chain| &chain.status)
                .chain([
                    &diagnostic.lambda_status,
                    &diagnostic.xi_status,
                    &diagnostic.simulation_covariance_status,
                ])
                .find(|status| {
                    !matches!(
                        status,
                        MarkovSimulationVarianceStatus::AssumptionsUnverified
                    )
                })
                .cloned()
                .unwrap_or(MarkovSimulationVarianceStatus::AssumptionsUnverified)
        };
        diagnostic
    }

    /// Build per-coordinate rank/mixing diagnostics from collected trace chains.
    fn rank_diagnostics_from_traces(
        &self,
        cd: usize,
        traces: &[Vec<Vec<f64>>],
        trace_coords: &[DiagnosticTraceCoordinate],
    ) -> Vec<RankMixingDiagnostic> {
        trace_coords
            .iter()
            .enumerate()
            .map(|(idx, coord)| {
                let chains = &traces[idx];
                let rank_result = rank_normalized_split_rhat(chains);
                let folded_result = folded_split_rhat(chains);
                let ess_result = bulk_ess(chains);

                let rank_rhat = match rank_result.as_ref() {
                    Ok(value) => Some(*value),
                    Err(_) => None,
                };
                let folded_rhat = match folded_result.as_ref() {
                    Ok(value) => Some(*value),
                    Err(_) => None,
                };
                let (bulk_ess, tau) = match ess_result.as_ref() {
                    Ok((ess, tau)) => (Some(*ess), Some(*tau)),
                    Err(_) => (None, None),
                };
                let avg_ess_per_split_chain = bulk_ess.map(|ess| ess / (2.0 * cd as f64));

                let score_unavailable = matches!(coord, DiagnosticTraceCoordinate::Score { .. })
                    && chains.iter().flatten().any(|draw| !draw.is_finite());
                let statistic_status = |result: Result<(), &RankDiagnosticError>| {
                    if score_unavailable {
                        RankDiagnosticStatus::ScoreUnavailable
                    } else {
                        result
                            .map(|()| RankDiagnosticStatus::Available)
                            .unwrap_or_else(rank_diagnostic_error_status)
                    }
                };
                let rank_rhat_status = statistic_status(rank_result.as_ref().map(|_| ()));
                let folded_rhat_status = statistic_status(folded_result.as_ref().map(|_| ()));
                let max_rhat = match (rank_rhat, folded_rhat) {
                    (Some(rank), Some(folded)) => Some(rank.max(folded)),
                    _ => None,
                };
                let max_rhat_status = if matches!(rank_rhat_status, RankDiagnosticStatus::Available)
                    && matches!(folded_rhat_status, RankDiagnosticStatus::Available)
                {
                    RankDiagnosticStatus::Available
                } else if !matches!(rank_rhat_status, RankDiagnosticStatus::Available) {
                    rank_rhat_status.clone()
                } else {
                    folded_rhat_status.clone()
                };
                let bulk_ess_status = statistic_status(ess_result.as_ref().map(|_| ()));
                let statuses = [&rank_rhat_status, &folded_rhat_status, &bulk_ess_status];
                let available = statuses
                    .iter()
                    .filter(|status| matches!(status, RankDiagnosticStatus::Available))
                    .count();
                let status = if available == statuses.len() {
                    RankDiagnosticStatus::Available
                } else if available > 0 {
                    RankDiagnosticStatus::PartialAvailability
                } else if statuses.iter().all(|status| *status == statuses[0]) {
                    statuses[0].clone()
                } else {
                    RankDiagnosticStatus::Unavailable
                };

                RankMixingDiagnostic {
                    trace: coord.clone(),
                    rank_rhat,
                    rank_rhat_status,
                    folded_rhat,
                    folded_rhat_status,
                    max_rhat,
                    max_rhat_status,
                    bulk_ess,
                    bulk_ess_status,
                    avg_ess_per_split_chain,
                    tau,
                    status,
                }
            })
            .collect()
    }

    /// Draw initial η vectors from N(0, Omega) for fresh diagnostic chains.
    fn draw_prior_etas(&self, omega_lower: &[Vec<f64>], rng: &mut StdRng) -> Vec<Vec<Vec<f64>>> {
        let n_eta = self.initialization.random_effect_indices.len();
        if n_eta == 0 {
            return vec![vec![Vec::new(); 1]; self.initialization.subject_ids.len()];
        }
        self.initialization
            .subject_ids
            .iter()
            .map(|_| {
                let normals: Vec<f64> = (0..n_eta)
                    .map(|_| diagnostic_standard_normal(rng))
                    .collect();
                let eta = (0..n_eta)
                    .map(|row| {
                        (0..=row)
                            .map(|col| omega_lower[row][col] * normals[col])
                            .sum()
                    })
                    .collect::<Vec<_>>();
                vec![eta]
            })
            .collect()
    }

    /// Draw initial κ vectors from N(0, Omega_IOV) for fresh diagnostic chains.
    fn draw_prior_kappas(
        &self,
        iov_lower: Option<&[Vec<f64>]>,
        rng: &mut StdRng,
    ) -> Vec<Vec<Vec<Vec<f64>>>> {
        let Some(iov_lower) = iov_lower else {
            return vec![vec![Vec::new(); 1]; self.initialization.subject_ids.len()];
        };
        let n_kappa = self.initialization.iov_effect_indices.len();
        self.initialization
            .occasion_counts
            .iter()
            .map(|&n_occasions| {
                let kappas: Vec<Vec<f64>> = (0..n_occasions)
                    .map(|_| {
                        let normals: Vec<f64> = (0..n_kappa)
                            .map(|_| diagnostic_standard_normal(rng))
                            .collect();
                        (0..n_kappa)
                            .map(|row| {
                                (0..=row)
                                    .map(|col| iov_lower[row][col] * normals[col])
                                    .sum()
                            })
                            .collect()
                    })
                    .collect();
                vec![kappas]
            })
            .collect()
    }

    fn frozen_diagnostic_transition(
        &self,
        state: &mut FrozenDiagnosticState,
        rng: &mut StdRng,
        counts: &mut [(usize, usize, usize)],
        candidate: Option<&DiagnosticCandidate>,
    ) -> std::result::Result<(), String> {
        for _ in 0..self.eta_block_iterations {
            for subject in 0..self.initialization.subject_ids.len() {
                let omega = candidate.map_or(&self.omega, |value| &value.omega);
                let lower = cholesky_lower(omega).map_err(|error| error.to_string())?;
                for (chain, count) in counts.iter_mut().enumerate() {
                    let current = state.etas[subject][chain].clone();
                    let normals = (0..current.len())
                        .map(|_| diagnostic_standard_normal(rng))
                        .collect::<Vec<_>>();
                    let proposed = correlated_random_walk(
                        &current,
                        &lower,
                        &normals,
                        self.eta_block_step_sizes[subject],
                    )
                    .map_err(|error| error.to_string())?;
                    let current_score = self
                        .score_subject_latents_at(
                            subject,
                            &current,
                            &state.kappas[subject][chain],
                            candidate,
                        )
                        .map_err(|error| error.to_string())?;
                    let proposed_score = self
                        .score_subject_latents_at(
                            subject,
                            &proposed,
                            &state.kappas[subject][chain],
                            candidate,
                        )
                        .map_err(|error| error.to_string())?;
                    count.0 += 1;
                    if diagnostic_accept(rng, current_score.log_acceptance_ratio(proposed_score)) {
                        count.1 += 1;
                        if proposed != current {
                            count.2 += 1;
                        }
                        state.etas[subject][chain] = proposed;
                    }
                }
            }
        }
        for _ in 0..self.mcmc_iterations {
            for subject in 0..self.initialization.subject_ids.len() {
                for (chain, count) in counts.iter_mut().enumerate() {
                    for parameter in 0..self.initialization.random_effect_indices.len() {
                        let current = state.etas[subject][chain].clone();
                        let mut proposed = current.clone();
                        proposed[parameter] +=
                            self.proposal_step_sizes[parameter] * diagnostic_standard_normal(rng);
                        let current_score = self
                            .score_subject_latents_at(
                                subject,
                                &current,
                                &state.kappas[subject][chain],
                                candidate,
                            )
                            .map_err(|error| error.to_string())?;
                        let proposed_score = self
                            .score_subject_latents_at(
                                subject,
                                &proposed,
                                &state.kappas[subject][chain],
                                candidate,
                            )
                            .map_err(|error| error.to_string())?;
                        count.0 += 1;
                        if diagnostic_accept(
                            rng,
                            current_score.log_acceptance_ratio(proposed_score),
                        ) {
                            count.1 += 1;
                            if proposed != current {
                                count.2 += 1;
                            }
                            state.etas[subject][chain] = proposed;
                        }
                    }
                    let omega_iov =
                        candidate.map_or(self.omega_iov.as_ref(), |value| value.omega_iov.as_ref());
                    if let Some(omega_iov) = omega_iov {
                        let lower = cholesky_lower(omega_iov).map_err(|error| error.to_string())?;
                        for occasion in 0..state.kappas[subject][chain].len() {
                            let current = state.kappas[subject][chain][occasion].clone();
                            let normals = (0..current.len())
                                .map(|_| diagnostic_standard_normal(rng))
                                .collect::<Vec<_>>();
                            let proposed = correlated_random_walk(
                                &current,
                                &lower,
                                &normals,
                                self.kappa_proposal_step_sizes[subject],
                            )
                            .map_err(|error| error.to_string())?;
                            let current_score = self
                                .score_subject_latents_at(
                                    subject,
                                    &state.etas[subject][chain],
                                    &state.kappas[subject][chain],
                                    candidate,
                                )
                                .map_err(|error| error.to_string())?;
                            let mut proposed_kappas = state.kappas[subject][chain].clone();
                            proposed_kappas[occasion] = proposed.clone();
                            let proposed_score = self
                                .score_subject_latents_at(
                                    subject,
                                    &state.etas[subject][chain],
                                    &proposed_kappas,
                                    candidate,
                                )
                                .map_err(|error| error.to_string())?;
                            count.0 += 1;
                            if diagnostic_accept(
                                rng,
                                current_score.log_acceptance_ratio(proposed_score),
                            ) {
                                count.1 += 1;
                                if proposed != current {
                                    count.2 += 1;
                                }
                                state.kappas[subject][chain][occasion] = proposed;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // ─── Operational convergence ─────────────────────────────────────────

    /// Evaluate an operational convergence checkpoint if one is due.
    fn evaluate_operational_convergence(
        &mut self,
        iteration: usize,
        scheduled: bool,
        mandatory_final: bool,
    ) -> Result<()> {
        let Some(settings) = self.operational_settings else {
            return Ok(());
        };
        // Only check during smoothing, unless this is a mandatory final check.
        if !mandatory_final && self.initialization.schedule.phase(iteration) != SaemPhase::Smoothing
        {
            return Ok(());
        }
        let Some(ref average) = self.iterate_average else {
            return Ok(());
        };
        let n_averaged = average.count;
        if n_averaged < settings.first_eligible_averaged_iteration {
            return Ok(());
        }

        // Cadence: periodic checkpoints are evaluated every check_interval
        // iterations starting from first_eligible_averaged_iteration.
        if scheduled && !mandatory_final {
            let smoothing_start = self.initialization.schedule.pure_burn_in
                + self.initialization.schedule.exploration_iterations
                + 1;
            let smoothing_offset = iteration.saturating_sub(smoothing_start) + 1;
            if smoothing_offset < settings.first_eligible_averaged_iteration
                || !(smoothing_offset - settings.first_eligible_averaged_iteration)
                    .is_multiple_of(settings.check_interval)
            {
                return Ok(());
            }
        }

        // Defensive caching: if this is a mandatory final check and we already
        // evaluated at this iteration, reuse instead of rerunning.
        if mandatory_final {
            if let Some(last) = self.operational_diagnostics.checks.last() {
                if last.iteration == iteration {
                    self.operational_diagnostics.final_check_reused = true;
                    return Ok(());
                }
            }
        }

        // Build the deterministic per-checkpoint seed.
        let checkpoint_seed = self
            .config
            .markov_simulation_variance
            .expect("operational policy validation requires Markov diagnostics")
            .seed
            .wrapping_add(OPERATIONAL_CHECKPOINT_SEED_DOMAIN)
            .wrapping_add(iteration as u64);

        // Two-sided standard normal quantile.
        let z_quantile = normal_two_sided_z(settings.confidence_level);

        let implied_averaged_iterations =
            Some(4.0 * z_quantile * z_quantile / settings.relative_fixed_width_epsilon.powi(2));

        let info = self.information.diagnostics();
        let avg_psi = match population_psi(
            &average.population_phi,
            &self.initialization.parameter_scales,
        ) {
            Ok(psi) => psi,
            Err(_) => {
                self.record_ineligible_checkpoint(
                    iteration,
                    n_averaged,
                    scheduled,
                    mandatory_final,
                    checkpoint_seed,
                    z_quantile,
                    implied_averaged_iterations,
                    Vec::new(),
                    "averaged population psi conversion failed".to_string(),
                );
                return Ok(());
            }
        };
        let mut candidate_error_models = self.error_models.clone();
        for (output_index, model) in &average.residual_models {
            match *model {
                ResidualErrorModel::Combined { a, b } => update_estimated_combined_residual_model(
                    &mut candidate_error_models,
                    *output_index,
                    a,
                    b,
                ),
                ResidualErrorModel::CorrelatedCombined { a, b, rho } => {
                    update_estimated_correlated_combined_residual_model(
                        &mut candidate_error_models,
                        *output_index,
                        a,
                        b,
                        rho,
                    )
                }
                ResidualErrorModel::Constant { .. }
                | ResidualErrorModel::Proportional { .. }
                | ResidualErrorModel::Exponential { .. } => {
                    update_estimated_simple_residual_model_with_sigma(
                        &mut candidate_error_models,
                        *output_index,
                        primary_sigma_parameter(model),
                    )
                }
            }
        }
        let candidate_covariate_model = match (
            self.covariate_model.as_ref(),
            average.covariate_betas.as_ref(),
        ) {
            (Some(model), Some(values)) => Some(model.with_estimates(values)?),
            (None, None) => None,
            _ => anyhow::bail!("averaged covariate metadata dimension mismatch"),
        };
        let candidate = DiagnosticCandidate {
            population_parameters: avg_psi,
            covariate_model: candidate_covariate_model,
            omega: average.omega.clone(),
            omega_iov: average.omega_iov.clone(),
            error_models: candidate_error_models,
        };
        let candidate_free_coordinates = match operational_free_coordinates(&info, average) {
            Ok(values) if !values.is_empty() => values,
            Ok(_) => {
                self.record_ineligible_checkpoint(
                    iteration,
                    n_averaged,
                    scheduled,
                    mandatory_final,
                    checkpoint_seed,
                    z_quantile,
                    implied_averaged_iterations,
                    Vec::new(),
                    "no free coordinates".to_string(),
                );
                return Ok(());
            }
            Err(error) => {
                self.record_ineligible_checkpoint(
                    iteration,
                    n_averaged,
                    scheduled,
                    mandatory_final,
                    checkpoint_seed,
                    z_quantile,
                    implied_averaged_iterations,
                    Vec::new(),
                    error.to_string(),
                );
                return Ok(());
            }
        };
        if self.initialization.random_effect_indices.is_empty()
            && self.initialization.iov_effect_indices.is_empty()
        {
            self.record_ineligible_checkpoint(
                iteration,
                n_averaged,
                scheduled,
                mandatory_final,
                checkpoint_seed,
                z_quantile,
                implied_averaged_iterations,
                candidate_free_coordinates,
                "no latent coordinates".to_string(),
            );
            return Ok(());
        }

        let diagnostic_metadata = SaemEstimatorMetadata {
            policy: self.config.estimator_policy,
            average_applied: true,
            averaging_start_cycle: Some(average.start_cycle),
            averaged_iterations: n_averaged,
        };

        let markov = self.markov_variance_diagnostics_with_seed(
            &diagnostic_metadata,
            &info,
            Some(checkpoint_seed),
            Some(&candidate),
        );

        let rank = &markov.rank_diagnostics;
        let simulation_sd_fraction = operational_simulation_sd_fraction(&info, &markov);
        let fixed_width = simulation_sd_fraction.map(|fraction| 2.0 * z_quantile * fraction);
        let fixed_width_ratio =
            fixed_width.map(|width| width / settings.relative_fixed_width_epsilon);
        let newton_value = newton_displacement(&info, &markov).filter(|value| value.is_finite());
        let newton_mc_sd =
            newton_displacement_mc_sd(&info, &markov).filter(|value| value.is_finite());
        let matrix_valid = matches!(info.status, InformationStatus::Available)
            && matches!(
                markov.lambda_status,
                MarkovSimulationVarianceStatus::AssumptionsUnverified
            )
            && matches!(
                markov.xi_status,
                MarkovSimulationVarianceStatus::AssumptionsUnverified
            )
            && matches!(
                markov.simulation_covariance_status,
                MarkovSimulationVarianceStatus::AssumptionsUnverified
            );
        let every_chain_moved =
            !markov.chains.is_empty() && markov.chains.iter().all(|chain| chain.state_changes > 0);
        let every_trace_valid = !rank.traces.is_empty()
            && rank.traces.iter().all(|trace| {
                trace.rank_rhat.is_some()
                    && trace.folded_rhat.is_some()
                    && trace.max_rhat.is_some()
                    && trace.bulk_ess.is_some()
                    && matches!(trace.rank_rhat_status, RankDiagnosticStatus::Available)
                    && matches!(trace.folded_rhat_status, RankDiagnosticStatus::Available)
                    && matches!(trace.max_rhat_status, RankDiagnosticStatus::Available)
                    && matches!(trace.bulk_ess_status, RankDiagnosticStatus::Available)
            });
        let covariance_policy = self
            .config
            .covariance_stability
            .expect("operational policy validation requires covariance stability");
        let omega_boundary = covariance_boundary_rejection_summary(
            &self.cycle_diagnostics,
            covariance_policy,
            false,
        );
        let omega_iov_boundary =
            covariance_boundary_rejection_summary(&self.cycle_diagnostics, covariance_policy, true);
        let covariance_active_cycles =
            iteration.saturating_sub(self.initialization.schedule.pure_burn_in);
        let covariance_window_available =
            covariance_active_cycles >= covariance_policy.rejection_window;
        let boundary_criterion = |name: &str, longest_run: usize| {
            if covariance_window_available {
                evaluate_criterion(
                    name,
                    Some(longest_run as f64),
                    covariance_policy.rejection_window as f64,
                    |observed| observed < covariance_policy.rejection_window as f64,
                )
            } else {
                OperationalConvergenceCriterion {
                    name: name.to_string(),
                    observed: Some(longest_run as f64),
                    threshold: covariance_policy.rejection_window as f64,
                    status: OperationalConvergenceCriterionStatus::Unavailable(format!(
                        "covariance-stability window requires {} active cycles; {covariance_active_cycles} completed",
                        covariance_policy.rejection_window
                    )),
                }
            }
        };
        let criteria: Vec<OperationalConvergenceCriterion> = vec![
            evaluate_criterion(
                "valid_information_and_matrices",
                Some(matrix_valid as u8 as f64),
                1.0,
                |value| value == 1.0,
            ),
            evaluate_criterion(
                "every_diagnostic_chain_moved",
                Some(every_chain_moved as u8 as f64),
                1.0,
                |value| value == 1.0,
            ),
            evaluate_criterion(
                "every_rank_diagnostic_valid",
                Some(every_trace_valid as u8 as f64),
                1.0,
                |value| value == 1.0,
            ),
            evaluate_criterion("max_rhat", rank.worst_rhat, settings.max_rhat, |observed| {
                observed < settings.max_rhat
            }),
            evaluate_criterion(
                "min_bulk_ess",
                rank.min_bulk_ess,
                settings.min_bulk_ess,
                |observed| observed > settings.min_bulk_ess,
            ),
            evaluate_criterion(
                "min_average_bulk_ess_per_split_chain",
                rank.min_avg_ess_per_split_chain,
                settings.min_average_bulk_ess_per_split_chain,
                |observed| observed >= settings.min_average_bulk_ess_per_split_chain,
            ),
            evaluate_criterion(
                "worst_simulation_sd_fraction",
                simulation_sd_fraction,
                settings.relative_fixed_width_epsilon / (2.0 * z_quantile),
                |observed| 2.0 * z_quantile * observed <= settings.relative_fixed_width_epsilon,
            ),
            evaluate_criterion(
                "relative_fixed_width",
                fixed_width,
                settings.relative_fixed_width_epsilon,
                |observed| observed <= settings.relative_fixed_width_epsilon,
            ),
            evaluate_criterion(
                "newton_displacement",
                newton_value,
                settings.max_newton_displacement,
                |observed| observed <= settings.max_newton_displacement,
            ),
            evaluate_criterion(
                "newton_displacement_mc_sd",
                newton_mc_sd,
                settings.max_newton_displacement_mc_sd,
                |observed| observed <= settings.max_newton_displacement_mc_sd,
            ),
            boundary_criterion("omega_boundary_rejection_run", omega_boundary.longest_run),
            boundary_criterion(
                "omega_iov_boundary_rejection_run",
                omega_iov_boundary.longest_run,
            ),
        ];

        let mut ineligible_reasons = criteria
            .iter()
            .filter_map(|criterion| match &criterion.status {
                OperationalConvergenceCriterionStatus::Unavailable(reason) => {
                    Some(format!("{}: {reason}", criterion.name))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if !matrix_valid {
            ineligible_reasons.push("information or matrix validation failed".to_string());
        }
        if !every_trace_valid {
            ineligible_reasons.push("one or more rank diagnostics unavailable".to_string());
        }
        if !every_chain_moved {
            ineligible_reasons
                .push("one or more retained diagnostic chains did not move".to_string());
        }
        let failed_criteria = criteria
            .iter()
            .filter(|criterion| {
                matches!(
                    criterion.status,
                    OperationalConvergenceCriterionStatus::NotSatisfied
                )
            })
            .map(|criterion| criterion.name.clone())
            .collect::<Vec<_>>();
        let outcome = if !ineligible_reasons.is_empty() {
            OperationalConvergenceOutcome::Ineligible {
                reasons: ineligible_reasons,
            }
        } else if !failed_criteria.is_empty() {
            OperationalConvergenceOutcome::Failed {
                criteria: failed_criteria,
            }
        } else {
            OperationalConvergenceOutcome::Passed
        };

        let passed = matches!(outcome, OperationalConvergenceOutcome::Passed);
        self.operational_diagnostics.final_status = Some(outcome.clone());
        self.operational_diagnostics.worst_rhat = rank.worst_rhat;
        self.operational_diagnostics.min_bulk_ess = rank.min_bulk_ess;
        self.operational_diagnostics.fixed_width_ratio = fixed_width_ratio;
        self.operational_diagnostics.fixed_width_epsilon =
            Some(settings.relative_fixed_width_epsilon);
        self.operational_diagnostics.implied_minimum_ess = implied_averaged_iterations;
        self.operational_diagnostics.newton_displacement = newton_value;
        self.operational_diagnostics.newton_displacement_mc_sd = newton_mc_sd;
        let checkpoint = OperationalConvergenceCheck {
            iteration,
            averaged_iterations: n_averaged,
            scheduled,
            mandatory_final,
            checkpoint_seed: Some(checkpoint_seed),
            z_quantile: Some(z_quantile),
            implied_minimum_ess: implied_averaged_iterations,
            candidate_free_coordinates,
            information: Some(info),
            criteria,
            outcome,
            markov: Some(markov),
        };

        self.operational_diagnostics.checks.push(checkpoint);

        // Terminate early if converged and this was a scheduled check.
        if passed {
            self.operational_diagnostics.used_for_termination = true;
            self.status = Status::Stop(StopReason::Converged);
        }

        Ok(())
    }

    /// Record an ineligible checkpoint (candidate unavailable).
    #[allow(clippy::too_many_arguments)]
    fn record_ineligible_checkpoint(
        &mut self,
        iteration: usize,
        averaged_iterations: usize,
        scheduled: bool,
        mandatory_final: bool,
        checkpoint_seed: u64,
        z_quantile: f64,
        implied_averaged_iterations: Option<f64>,
        candidate_free_coordinates: Vec<f64>,
        reason: String,
    ) {
        let settings = self
            .operational_settings
            .expect("ineligible operational checkpoint requires configured settings");
        let unavailable = |name: &str, threshold: f64| OperationalConvergenceCriterion {
            name: name.to_string(),
            observed: None,
            threshold,
            status: OperationalConvergenceCriterionStatus::Unavailable(reason.clone()),
        };
        let criteria = vec![
            unavailable("candidate_available", 1.0),
            unavailable("valid_information_and_matrices", 1.0),
            unavailable("every_diagnostic_chain_moved", 1.0),
            unavailable("every_rank_diagnostic_valid", 1.0),
            unavailable("max_rhat", settings.max_rhat),
            unavailable("min_bulk_ess", settings.min_bulk_ess),
            unavailable(
                "min_average_bulk_ess_per_split_chain",
                settings.min_average_bulk_ess_per_split_chain,
            ),
            unavailable(
                "worst_simulation_sd_fraction",
                settings.relative_fixed_width_epsilon / (2.0 * z_quantile),
            ),
            unavailable(
                "relative_fixed_width",
                settings.relative_fixed_width_epsilon,
            ),
            unavailable("newton_displacement", settings.max_newton_displacement),
            unavailable(
                "newton_displacement_mc_sd",
                settings.max_newton_displacement_mc_sd,
            ),
        ];
        let outcome = OperationalConvergenceOutcome::Ineligible {
            reasons: vec![reason],
        };
        self.operational_diagnostics.final_status = Some(outcome.clone());
        self.operational_diagnostics
            .checks
            .push(OperationalConvergenceCheck {
                iteration,
                averaged_iterations,
                scheduled,
                mandatory_final,
                checkpoint_seed: Some(checkpoint_seed),
                z_quantile: Some(z_quantile),
                implied_minimum_ess: implied_averaged_iterations,
                candidate_free_coordinates,
                information: None,
                criteria,
                outcome,
                markov: None,
            });
    }

    fn frozen_complete_score(
        &self,
        state: &FrozenDiagnosticState,
        chain: usize,
        candidate: Option<&DiagnosticCandidate>,
    ) -> std::result::Result<Vec<f64>, String> {
        let layout = self.information.layout();
        let population_parameters = candidate
            .map_or(self.population_parameters.as_slice(), |value| {
                value.population_parameters.as_slice()
            });
        let omega = candidate.map_or(&self.omega, |value| &value.omega);
        let omega_iov = candidate.map_or(self.omega_iov.as_ref(), |value| value.omega_iov.as_ref());
        let error_models = candidate.map_or(&self.error_models, |value| &value.error_models);
        let mut derivative = CompleteDerivative::zero(layout.len());
        for subject_index in 0..self.initialization.subject_ids.len() {
            let covariate_model = candidate
                .and_then(|value| value.covariate_model.as_ref())
                .or(self.covariate_model.as_ref());
            match covariate_model {
                Some(model) => derivative.add_covariate_population_prior(
                    &state.etas[subject_index][chain],
                    omega,
                    &self.initialization.random_effect_indices,
                    model.parameter_indices(),
                    model.subject_design()[subject_index].values(),
                    layout,
                ),
                None => derivative.add_population_prior(
                    &state.etas[subject_index][chain],
                    omega,
                    &self.initialization.random_effect_indices,
                    layout,
                ),
            }
            .map_err(|error| error.to_string())?;
            let calculated_mu = if candidate.is_some() {
                covariate_model
                    .map(|model| {
                        let phi = population_phi(
                            population_parameters,
                            &self.initialization.parameter_scales,
                        )?;
                        Ok::<_, anyhow::Error>(
                            model.subject_population_parameters(
                                &phi,
                                &self.initialization.parameter_scales,
                            )?[subject_index]
                                .phi()
                                .to_vec(),
                        )
                    })
                    .transpose()
                    .map_err(|error| error.to_string())?
            } else {
                None
            };
            let subject_mu = calculated_mu.as_deref().or_else(|| {
                self.subject_mu_phi
                    .as_ref()
                    .map(|means| means[subject_index].as_slice())
            });
            let subject = self.data.subjects()[subject_index];
            if let Some(omega_iov) = omega_iov {
                let occasions = subject.occasions();
                let kappas = &state.kappas[subject_index][chain];
                if occasions.len() != kappas.len() {
                    return Err(format!(
                        "subject {} has {} occasions but {} diagnostic kappa states",
                        subject.id(),
                        occasions.len(),
                        kappas.len()
                    ));
                }
                for (occasion, kappa) in occasions.iter().zip(kappas) {
                    derivative
                        .add_iov_prior(kappa, omega_iov, layout)
                        .map_err(|error| error.to_string())?;
                    let parameters = match subject_mu {
                        Some(mean) => occasion_psi_from_subject_mean(
                            mean,
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            &state.etas[subject_index][chain],
                            &self.initialization.iov_effect_indices,
                            kappa,
                        ),
                        None => occasion_psi(
                            population_parameters,
                            &self.initialization.parameter_scales,
                            &self.initialization.random_effect_indices,
                            &state.etas[subject_index][chain],
                            &self.initialization.iov_effect_indices,
                            kappa,
                        ),
                    }
                    .map_err(|error| error.to_string())?;
                    let occasion_subject =
                        Subject::from_occasions(subject.id().to_owned(), vec![occasion.clone()]);
                    let predictions = self
                        .equation
                        .estimate_predictions_dense(&occasion_subject, &parameters)
                        .map_err(|error| error.to_string())?;
                    derivative
                        .add_predictions_strict(&predictions, error_models, layout)
                        .map_err(|error| error.to_string())?;
                }
            } else {
                let parameters = match subject_mu {
                    Some(mean) => individual_psi_from_subject_mean(
                        mean,
                        &self.initialization.parameter_scales,
                        &self.initialization.random_effect_indices,
                        &state.etas[subject_index][chain],
                    ),
                    None => individual_psi(
                        population_parameters,
                        &self.initialization.parameter_scales,
                        &self.initialization.random_effect_indices,
                        &state.etas[subject_index][chain],
                    ),
                }
                .map_err(|error| error.to_string())?;
                let predictions = self
                    .equation
                    .estimate_predictions_dense(subject, &parameters)
                    .map_err(|error| error.to_string())?;
                derivative
                    .add_predictions_strict(&predictions, error_models, layout)
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(derivative.score)
    }
}

fn diagnostic_standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn diagnostic_accept(rng: &mut StdRng, ratio: f64) -> bool {
    ratio.is_finite() && (ratio >= 0.0 || rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < ratio)
}

fn begin_retained_transition_accounting(counts: &mut [(usize, usize, usize)]) {
    counts.fill((0, 0, 0));
}

fn mark_diagnostic_failure(
    diagnostic: &mut MarkovSimulationVarianceDiagnostics,
    rank_status: RankDiagnosticStatus,
    markov_status: MarkovSimulationVarianceStatus,
) {
    diagnostic.rank_diagnostics.status = rank_status.clone();
    diagnostic
        .rank_diagnostics
        .lrv_chain_statuses
        .fill(rank_status);
    diagnostic.lambda_status = markov_status.clone();
    diagnostic.xi_status = markov_status.clone();
    diagnostic.simulation_covariance_status = markov_status.clone();
    diagnostic.status = markov_status;
}

fn markov_matrix_status(classification: MatrixClassification) -> MarkovSimulationVarianceStatus {
    match classification {
        MatrixClassification::EligiblePsd => MarkovSimulationVarianceStatus::AssumptionsUnverified,
        MatrixClassification::NonFinite => MarkovSimulationVarianceStatus::NonFinite,
        MatrixClassification::NonSymmetric => MarkovSimulationVarianceStatus::NonSymmetric,
        MatrixClassification::Indefinite => MarkovSimulationVarianceStatus::Indefinite,
    }
}

fn worst_valid_max_rhat(traces: &[RankMixingDiagnostic]) -> Option<f64> {
    traces
        .iter()
        .filter(|trace| matches!(trace.max_rhat_status, RankDiagnosticStatus::Available))
        .filter_map(|trace| trace.max_rhat)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn rank_diagnostic_error_status(error: &RankDiagnosticError) -> RankDiagnosticStatus {
    match error {
        RankDiagnosticError::NoChains => RankDiagnosticStatus::NoChains,
        RankDiagnosticError::TooFewChains { .. } => RankDiagnosticStatus::TooFewChains,
        RankDiagnosticError::UnequalChainLengths { .. } => {
            RankDiagnosticStatus::UnequalChainLengths
        }
        RankDiagnosticError::OddChainLength { .. } => RankDiagnosticStatus::OddDraws,
        RankDiagnosticError::NonFiniteDraw => RankDiagnosticStatus::NonFiniteDraws,
        RankDiagnosticError::TooFewDraws { .. } => RankDiagnosticStatus::TooFewDraws,
        RankDiagnosticError::ConstantDraws => RankDiagnosticStatus::ConstantDraws,
        RankDiagnosticError::InvalidVariance => RankDiagnosticStatus::InvalidVariance,
        RankDiagnosticError::NonPositiveTau { .. } => RankDiagnosticStatus::NonPositiveTau,
    }
}

fn matrix_from_rows(values: &[Vec<f64>], width: usize) -> Result<Array2<f64>> {
    if values.len() != width || values.iter().any(|row| row.len() != width) {
        anyhow::bail!("matrix coordinate width mismatch");
    }
    Ok(Array2::from_shape_vec(
        (width, width),
        values.iter().flatten().copied().collect(),
    )?)
}

impl<E: Equation + Send + 'static> ParametricRunner<E> for SaemState<E> {
    fn step(&mut self) -> Result<Status> {
        if self.status.is_stop() {
            return Ok(self.status.clone());
        }

        if self.cycle >= self.initialization.schedule.total_iterations {
            self.status = Status::Stop(StopReason::MaxCycles);
            return Ok(self.status.clone());
        }

        self.cycle += 1;
        if let Err(error) = self.e_step() {
            let failure = NumericalFailure::new(
                self.cycle,
                NumericalFailurePhase::Expectation,
                format!("{error:#}"),
            );
            self.status = Status::Stop(StopReason::NumericalFailure);
            self.numerical_failure = Some(failure.clone());
            return Err(failure.into());
        }
        // m_step also accumulates damped covariance sufficient statistics
        // during pure burn-in while leaving theta, Omega, Omega_IOV, and sigma
        // unchanged, so it must run in every schedule phase.
        if let Err(error) = self.m_step() {
            let failure = NumericalFailure::new(
                self.cycle,
                NumericalFailurePhase::Maximization,
                format!("{error:#}"),
            );
            self.status = Status::Stop(StopReason::NumericalFailure);
            self.numerical_failure = Some(failure.clone());
            return Err(failure.into());
        }

        if self.cycle >= self.initialization.schedule.total_iterations {
            self.status = Status::Stop(StopReason::MaxCycles);
            let scheduled = self
                .operational_settings
                .zip(self.iterate_average.as_ref())
                .is_some_and(|(policy, average)| {
                    average.count >= policy.first_eligible_averaged_iteration
                        && (average.count - policy.first_eligible_averaged_iteration)
                            .is_multiple_of(policy.check_interval)
                });
            self.evaluate_operational_convergence(self.cycle, scheduled, true)?;
        } else {
            self.evaluate_operational_convergence(self.cycle, true, false)?;
        }

        Ok(self.status.clone())
    }

    fn request_stop(&mut self, reason: StopReason) {
        if self.status.is_continue() && self.numerical_failure.is_none() {
            self.status = Status::Stop(reason);
        }
    }

    fn cycle(&self) -> usize {
        self.cycle
    }

    fn status(&self) -> &Status {
        &self.status
    }

    fn cycle_diagnostics(&self) -> &[SaemCycleDiagnostics] {
        &self.cycle_diagnostics
    }

    fn log_likelihood(&self) -> f64 {
        self.subject_log_likelihoods.iter().sum()
    }

    fn population_parameters(&self) -> &[f64] {
        &self.population_parameters
    }

    fn covariate_betas(&self) -> Option<Vec<f64>> {
        self.covariate_model.as_ref().map(|model| {
            model
                .estimates()
                .iter()
                .map(|estimate| estimate.estimate())
                .collect()
        })
    }

    fn random_effect_names(&self) -> &[String] {
        &self.initialization.random_effect_names
    }

    fn iov_effect_names(&self) -> Option<&[String]> {
        (!self.initialization.iov_effect_names.is_empty())
            .then_some(&self.initialization.iov_effect_names)
    }

    fn eta_log_prior(&self) -> f64 {
        self.subject_log_priors.iter().sum()
    }

    fn kappa_log_prior(&self) -> f64 {
        self.subject_kappa_log_priors.iter().sum()
    }

    fn acceptance_rate(&self) -> Option<f64> {
        self.last_acceptance_rate
    }

    fn eta_block_acceptance_rate(&self) -> Option<f64> {
        self.last_eta_block_acceptance_rate
    }

    fn kappa_acceptance_rate(&self) -> Option<f64> {
        self.last_kappa_acceptance_rate
    }

    fn rejected_proposals(&self) -> Option<usize> {
        self.last_rejected_proposals
    }

    fn non_finite_proposals(&self) -> Option<usize> {
        self.last_non_finite_proposals
    }

    fn parameter_acceptance_rates(&self) -> Option<&[f64]> {
        self.last_acceptance_rate
            .map(|_| self.last_parameter_acceptance_rates.as_slice())
    }

    fn proposal_step_sizes(&self) -> Option<&[f64]> {
        Some(&self.proposal_step_sizes)
    }

    fn eta_block_step_sizes(&self) -> Option<&[f64]> {
        (self.eta_block_iterations > 0).then_some(self.eta_block_step_sizes.as_slice())
    }

    fn log_acceptance_ratios(&self) -> Option<&[f64]> {
        Some(&self.last_log_acceptance_ratios)
    }

    fn negative_log_likelihood(&self) -> f64 {
        self.negative_log_likelihood
    }

    fn n_chains(&self) -> Option<usize> {
        self.etas
            .first()
            .map(|subject_chains| subject_chains.len())
            .or(Some(self.initialization.n_chains))
    }

    fn omega(&self) -> Option<&Array2<f64>> {
        Some(&self.omega)
    }

    fn omega_iov(&self) -> Option<&Array2<f64>> {
        self.omega_iov.as_ref()
    }

    fn residual_sigmas(&self) -> &[f64] {
        &self.residual_sigmas
    }

    fn step_size(&self) -> f64 {
        self.initialization
            .schedule
            .stochastic_approximation_step(self.cycle)
    }

    fn total_iterations(&self) -> usize {
        self.initialization.schedule.total_iterations
    }

    fn into_result(mut self: Box<Self>) -> Result<ParametricResult<E>> {
        if let Some(failure) = self.numerical_failure.as_ref() {
            return Err(failure.clone().into());
        }

        let result_cycle = self.cycle;
        let estimator_metadata = match self.config.estimator_policy {
            SaemEstimatorPolicy::TerminalIterate => SaemEstimatorMetadata::default(),
            SaemEstimatorPolicy::AveragedIterates { .. } => {
                self.install_iterate_average().map_err(|error| {
                    NumericalFailure::new(
                        result_cycle,
                        NumericalFailurePhase::ResultAssembly,
                        format!("{error:#}"),
                    )
                })?
            }
        };
        let information_diagnostics = self.information.diagnostics();
        let population_uncertainty = derive_population_uncertainty(&information_diagnostics);
        let markov_simulation_variance = if self.operational_settings.is_some() {
            self.operational_diagnostics
                .checks
                .last()
                .and_then(|check| check.markov.clone())
                .unwrap_or_else(MarkovSimulationVarianceDiagnostics::disabled)
        } else {
            self.markov_variance_diagnostics(&estimator_metadata, &information_diagnostics)
        };
        let (conditional_modes, conditional_mode_error) = match conditional_modes(&self) {
            Ok(modes) => (modes, None),
            Err(error) if self.config.marginal_likelihood.is_some() => {
                (Vec::new(), Some(format!("{error:#}")))
            }
            Err(error) => {
                return Err(NumericalFailure::new(
                    result_cycle,
                    NumericalFailurePhase::ResultAssembly,
                    format!("{error:#}"),
                )
                .into())
            }
        };
        let marginal_likelihood = calculate_result_marginal_likelihood(
            &self,
            &conditional_modes,
            conditional_mode_error.as_deref(),
        );
        let information_criteria = derive_information_criteria(
            marginal_likelihood.as_ref(),
            &information_diagnostics.coordinates,
            self.initialization.subject_ids.len(),
        );
        let eta_chain_means = self
            .initialization
            .subject_ids
            .iter()
            .enumerate()
            .map(|(subject_index, subject_id)| {
                Ok(SubjectEtaEstimate {
                    subject_id: subject_id.clone(),
                    values: mean_vectors(
                        self.etas[subject_index].iter().map(|eta| eta.as_slice()),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|error| {
                NumericalFailure::new(
                    result_cycle,
                    NumericalFailurePhase::ResultAssembly,
                    format!("{error:#}"),
                )
            })?;
        let mut kappa_chain_means = Vec::new();
        if self.omega_iov.is_some() {
            for (subject_index, subject_id) in self.initialization.subject_ids.iter().enumerate() {
                for (occasion_position, occasion) in self.data.subjects()[subject_index]
                    .occasions()
                    .iter()
                    .enumerate()
                {
                    kappa_chain_means.push(OccasionKappaEstimate {
                        subject_id: subject_id.clone(),
                        occasion_index: occasion.index(),
                        values: mean_vectors(
                            self.kappas[subject_index]
                                .iter()
                                .map(|chain| chain[occasion_position].as_slice()),
                        )
                        .map_err(|error| {
                            NumericalFailure::new(
                                result_cycle,
                                NumericalFailurePhase::ResultAssembly,
                                format!("{error:#}"),
                            )
                        })?,
                    });
                }
            }
        }
        let eta_variances = (0..self.omega.nrows())
            .map(|index| self.omega[[index, index]])
            .collect::<Vec<_>>();
        let eta_posterior_rows = eta_chain_means
            .iter()
            .map(|estimate| estimate.values.clone())
            .collect::<Vec<_>>();
        let eta_map_rows = (!conditional_modes.is_empty()).then(|| {
            conditional_modes
                .iter()
                .map(|mode| mode.eta.clone())
                .collect::<Vec<_>>()
        });
        let kappa_variances = self
            .omega_iov
            .as_ref()
            .map(|omega| {
                (0..omega.nrows())
                    .map(|index| omega[[index, index]])
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let kappa_posterior_rows = kappa_chain_means
            .iter()
            .map(|estimate| estimate.values.clone())
            .collect::<Vec<_>>();
        let kappa_map_rows = (!conditional_modes.is_empty()).then(|| {
            conditional_modes
                .iter()
                .flat_map(|mode| mode.kappas.iter().map(|kappa| kappa.values.clone()))
                .collect::<Vec<_>>()
        });
        let shrinkage = ShrinkageDiagnostics {
            eta_posterior_mean: derive_eta_posterior_mean_shrinkage(
                &self.initialization.random_effect_names,
                &eta_variances,
                &eta_posterior_rows,
            ),
            eta_map: derive_eta_map_shrinkage(
                &self.initialization.random_effect_names,
                &eta_variances,
                eta_map_rows.as_deref(),
            ),
            kappa_posterior_mean: derive_kappa_posterior_mean_shrinkage(
                &self.initialization.iov_effect_names,
                &kappa_variances,
                &kappa_posterior_rows,
            ),
            kappa_map: derive_kappa_map_shrinkage(
                &self.initialization.iov_effect_names,
                &kappa_variances,
                kappa_map_rows.as_deref(),
            ),
        };
        let residual_error_estimates = self.residual_error_estimates();
        let mut warnings =
            parametric_warnings(&self.cycle_diagnostics, self.config.covariance_stability);
        if let Some(diagnostics) = marginal_likelihood.as_ref() {
            match &diagnostics.status {
                MarginalLikelihoodStatus::Unavailable { failures } => {
                    warnings.push(ParametricWarning::MarginalLikelihoodUnavailable {
                        subjects: failures
                            .iter()
                            .map(|failure| failure.subject_id.clone())
                            .collect(),
                    });
                }
                MarginalLikelihoodStatus::AvailableWithNonconvergedModes { subjects } => {
                    warnings.push(ParametricWarning::MarginalLikelihoodNonconvergedModes {
                        subjects: subjects.clone(),
                    });
                }
                MarginalLikelihoodStatus::Available => {}
            }
        }
        let omega_structural_mask = self.initialization.omega.structural_mask().clone();
        let omega_estimated_mask = self.initialization.omega.estimated_mask().clone();
        let omega_iov_structural_mask = self
            .initialization
            .omega_iov
            .as_ref()
            .map(|omega| omega.structural_mask().clone());
        let omega_iov_estimated_mask = self
            .initialization
            .omega_iov
            .as_ref()
            .map(|omega| omega.estimated_mask().clone());
        let individual_estimates = if conditional_modes.is_empty() {
            self.initialization
                .subject_ids
                .iter()
                .enumerate()
                .map(|(subject_index, subject_id)| {
                    (
                        subject_id.clone(),
                        self.individual_parameters(subject_index, 0),
                    )
                })
                .collect()
        } else {
            conditional_modes
                .iter()
                .map(|mode| (mode.subject_id.clone(), mode.parameters.clone()))
                .collect()
        };

        let SaemState {
            equation,
            data,
            config,
            negative_log_likelihood,
            initialization,
            cycle,
            status,
            population_parameters,
            omega,
            omega_iov,
            residual_sigmas,
            cycle_diagnostics,
            operational_diagnostics,
            covariate_model,
            ..
        } = *self;

        Ok(ParametricResult {
            equation,
            data,
            config,
            effective_n_chains: initialization.n_chains,
            objective_function: 2.0 * negative_log_likelihood,
            converged: status.converged(),
            termination_reason: status.stop_reason().cloned(),
            iterations: cycle,
            subject_count: initialization.subject_ids.len(),
            observation_count: initialization.observation_count,
            parameter_names: initialization.parameter_names,
            parameter_scales: initialization.parameter_scales,
            estimated_parameters: initialization.estimated_parameters,
            population_initial: initialization.initial_population_parameters.clone(),
            population_estimates: population_parameters,
            random_effect_indices: initialization.random_effect_indices,
            random_effect_names: initialization.random_effect_names,
            omega,
            omega_structural_mask,
            omega_estimated_mask,
            omega_initial: initialization.omega.initial().clone(),
            iov_effect_indices: initialization.iov_effect_indices,
            iov_effect_names: initialization.iov_effect_names,
            omega_iov,
            omega_iov_structural_mask,
            omega_iov_estimated_mask,
            omega_iov_initial: initialization
                .omega_iov
                .as_ref()
                .map(|omega| omega.initial().clone()),
            residual_sigmas,
            residual_error_estimates,
            residual_initial_values: initialization.initial_residual_values.clone(),
            residual_initial_estimated: initialization.initial_residual_estimated.clone(),
            eta_chain_means,
            kappa_chain_means,
            conditional_modes,
            shrinkage,
            cycle_diagnostics,
            warnings,
            information_diagnostics,
            population_uncertainty,
            markov_simulation_variance,
            operational_diagnostics,
            marginal_likelihood,
            information_criteria,
            estimator_metadata,
            individual_estimates,
            covariate_model,
        })
    }
}

// ─── Operational convergence helpers ────────────────────────────────────

/// Two-sided standard normal quantile for confidence level `p` ∈ (0, 1).
///
/// Returns z such that P(|Z| ≤ z) = p, i.e. z = Φ⁻¹(p + (1-p)/2).
fn normal_two_sided_z(p: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let norm = Normal::new(0.0, 1.0).expect("standard normal parameters are valid");
    let one_sided = p + (1.0 - p) / 2.0;
    norm.inverse_cdf(one_sided)
}

/// Evaluate one operational convergence criterion.
fn evaluate_criterion(
    name: &str,
    observed: Option<f64>,
    threshold: f64,
    predicate: impl FnOnce(f64) -> bool,
) -> OperationalConvergenceCriterion {
    let status = match observed {
        Some(value) if value.is_finite() && predicate(value) => {
            OperationalConvergenceCriterionStatus::Satisfied
        }
        Some(value) if value.is_finite() => OperationalConvergenceCriterionStatus::NotSatisfied,
        Some(_) => OperationalConvergenceCriterionStatus::Unavailable(
            "observed value is non-finite".to_string(),
        ),
        None => OperationalConvergenceCriterionStatus::Unavailable(
            "criterion could not be evaluated".to_string(),
        ),
    };
    OperationalConvergenceCriterion {
        name: name.to_string(),
        observed,
        threshold,
        status,
    }
}

fn operational_free_coordinates(
    information: &InformationDiagnostics,
    average: &SaemIterateAverage,
) -> Result<Vec<f64>> {
    information
        .coordinates
        .iter()
        .map(|coordinate| match &coordinate.kind {
            InformationCoordinateKind::Population { parameter_index } => average
                .population_phi
                .get(*parameter_index)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("population coordinate out of range")),
            InformationCoordinateKind::CovariateEffect { effect_index } => average
                .covariate_betas
                .as_ref()
                .and_then(|values| values.get(*effect_index))
                .copied()
                .ok_or_else(|| anyhow::anyhow!("covariate coordinate out of range")),
            InformationCoordinateKind::Omega { row, column } => average
                .omega
                .get((*row, *column))
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Omega coordinate out of range")),
            InformationCoordinateKind::OmegaIov { row, column } => average
                .omega_iov
                .as_ref()
                .and_then(|matrix| matrix.get((*row, *column)))
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Omega_IOV coordinate out of range")),
            InformationCoordinateKind::Residual {
                output_index,
                component,
            } => {
                let model = average
                    .residual_models
                    .iter()
                    .find(|(index, _)| index == output_index)
                    .map(|(_, model)| model)
                    .ok_or_else(|| anyhow::anyhow!("residual coordinate output unavailable"))?;
                match (model, component.as_str()) {
                    (ResidualErrorModel::Constant { a }, "sigma") => Ok(*a),
                    (ResidualErrorModel::Exponential { sigma }, "sigma") => Ok(*sigma),
                    (ResidualErrorModel::Proportional { b }, "proportional") => Ok(*b),
                    (ResidualErrorModel::Combined { a, .. }, "additive")
                    | (ResidualErrorModel::CorrelatedCombined { a, .. }, "additive") => Ok(*a),
                    (ResidualErrorModel::Combined { b, .. }, "proportional")
                    | (ResidualErrorModel::CorrelatedCombined { b, .. }, "proportional") => Ok(*b),
                    (ResidualErrorModel::CorrelatedCombined { rho, .. }, "correlation") => Ok(*rho),
                    _ => anyhow::bail!("residual coordinate component mismatch"),
                }
            }
        })
        .collect()
}

fn operational_simulation_sd_fraction(
    information: &InformationDiagnostics,
    markov: &MarkovSimulationVarianceDiagnostics,
) -> Option<f64> {
    let width = information.coordinates.len();
    let observed = matrix_from_rows(&information.observed_information, width).ok()?;
    let covariance = matrix_from_rows(&markov.simulation_covariance, width).ok()?;
    worst_contrast(&observed, &covariance).ok()
}

fn solve_spd(matrix: &Array2<f64>, rhs: &[f64]) -> Option<Vec<f64>> {
    if matrix.nrows() != matrix.ncols() || matrix.nrows() != rhs.len() {
        return None;
    }
    let lower = cholesky_lower(matrix).ok()?;
    let n = rhs.len();
    let mut y = vec![0.0; n];
    for row in 0..n {
        let subtotal = (0..row)
            .map(|column| lower[row][column] * y[column])
            .sum::<f64>();
        y[row] = (rhs[row] - subtotal) / lower[row][row];
    }
    let mut result = vec![0.0; n];
    for row in (0..n).rev() {
        let subtotal = ((row + 1)..n)
            .map(|column| lower[column][row] * result[column])
            .sum::<f64>();
        result[row] = (y[row] - subtotal) / lower[row][row];
    }
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
}

/// Invariant Newton displacement `sqrt(g^T Iobs^-1 g)`.
fn newton_displacement(
    info: &InformationDiagnostics,
    markov: &MarkovSimulationVarianceDiagnostics,
) -> Option<f64> {
    let width = info.coordinates.len();
    if width == 0 || markov.grand_score_mean.len() != width {
        return None;
    }
    let observed = matrix_from_rows(&info.observed_information, width).ok()?;
    let displacement = solve_spd(&observed, &markov.grand_score_mean)?;
    let squared = markov
        .grand_score_mean
        .iter()
        .zip(&displacement)
        .map(|(score, step)| score * step)
        .sum::<f64>();
    (squared.is_finite() && squared >= 0.0).then(|| squared.sqrt())
}

/// Worst-direction Newton-step MC SD from diagnostic-mean LRV/draws.
fn newton_displacement_mc_sd(
    info: &InformationDiagnostics,
    markov: &MarkovSimulationVarianceDiagnostics,
) -> Option<f64> {
    let width = info.coordinates.len();
    let draws = markov.config?.draws_per_chain;
    if width == 0 || draws == 0 {
        return None;
    }
    let observed = matrix_from_rows(&info.observed_information, width).ok()?;
    let mut score_covariance =
        matrix_from_rows(markov.rank_diagnostics.diagnostic_mean_lrv.as_ref()?, width).ok()?;
    score_covariance /= draws as f64;
    let mut inverse = Array2::zeros((width, width));
    for column in 0..width {
        let mut unit = vec![0.0; width];
        unit[column] = 1.0;
        let solved = solve_spd(&observed, &unit)?;
        for row in 0..width {
            inverse[[row, column]] = solved[row];
        }
    }
    let mut mapped = Array2::zeros((width, width));
    for row in 0..width {
        for column in 0..=row {
            let mut value = 0.0;
            for left in 0..width {
                for right in 0..width {
                    value += inverse[[row, left]]
                        * score_covariance[[left, right]]
                        * inverse[[column, right]];
                }
            }
            mapped[[row, column]] = value;
            mapped[[column, row]] = value;
        }
    }
    worst_contrast(&observed, &mapped).ok()
}

fn incremental_average(previous: f64, current: f64, count: usize) -> f64 {
    previous + (current - previous) / count as f64
}

fn average_covariance(
    average: &mut Array2<f64>,
    current: &Array2<f64>,
    estimated_mask: &Array2<bool>,
    count: usize,
) {
    for row in 0..average.nrows() {
        for col in 0..=row {
            if estimated_mask[[row, col]] {
                let value = incremental_average(average[[row, col]], current[[row, col]], count);
                average[[row, col]] = value;
                average[[col, row]] = value;
            }
        }
    }
}

fn average_residual_model(
    previous: ResidualErrorModel,
    current: ResidualErrorModel,
    estimated: bool,
    components: [bool; 2],
    correlated_components: [bool; 3],
    count: usize,
) -> Result<ResidualErrorModel> {
    let averaged = match (previous, current) {
        (ResidualErrorModel::Constant { a }, ResidualErrorModel::Constant { a: current }) => {
            ResidualErrorModel::Constant {
                a: if estimated {
                    incremental_average(a, current, count)
                } else {
                    a
                },
            }
        }
        (
            ResidualErrorModel::Proportional { b },
            ResidualErrorModel::Proportional { b: current },
        ) => ResidualErrorModel::Proportional {
            b: if estimated {
                incremental_average(b, current, count)
            } else {
                b
            },
        },
        (
            ResidualErrorModel::Exponential { sigma },
            ResidualErrorModel::Exponential { sigma: current },
        ) => ResidualErrorModel::Exponential {
            sigma: if estimated {
                incremental_average(sigma, current, count)
            } else {
                sigma
            },
        },
        (
            ResidualErrorModel::Combined { a, b },
            ResidualErrorModel::Combined {
                a: current_a,
                b: current_b,
            },
        ) => ResidualErrorModel::Combined {
            a: if components[0] {
                incremental_average(a, current_a, count)
            } else {
                a
            },
            b: if components[1] {
                incremental_average(b, current_b, count)
            } else {
                b
            },
        },
        (
            ResidualErrorModel::CorrelatedCombined { a, b, rho },
            ResidualErrorModel::CorrelatedCombined {
                a: current_a,
                b: current_b,
                rho: current_rho,
            },
        ) => ResidualErrorModel::CorrelatedCombined {
            a: if correlated_components[0] {
                incremental_average(a, current_a, count)
            } else {
                a
            },
            b: if correlated_components[1] {
                incremental_average(b, current_b, count)
            } else {
                b
            },
            rho: if correlated_components[2] {
                incremental_average(rho, current_rho, count)
            } else {
                rho
            },
        },
        _ => anyhow::bail!("residual family changed while accumulating SAEM averages"),
    };
    Ok(averaged)
}

fn validate_average_population(values: &[f64], initialization: &SaemInitialization) -> Result<()> {
    let initial = population_phi(
        &initialization.initial_population_parameters,
        &initialization.parameter_scales,
    )?;
    if values.len() != initial.len() || values.iter().any(|value| !value.is_finite()) {
        anyhow::bail!("averaged population phi values must be finite and retain their width");
    }
    for index in 0..values.len() {
        if !initialization.estimated_parameters[index] && values[index] != initial[index] {
            anyhow::bail!("averaged population phi changed fixed coordinate {index}");
        }
    }
    Ok(())
}

fn validate_average_covariance(
    matrix: &Array2<f64>,
    specification: &ResolvedOmega,
    label: &str,
) -> Result<()> {
    if matrix.raw_dim() != specification.initial().raw_dim() {
        anyhow::bail!("averaged {label} has an invalid shape");
    }
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            let value = matrix[[row, col]];
            if !value.is_finite() || value != matrix[[col, row]] {
                anyhow::bail!("averaged {label} must be finite and symmetric");
            }
            if !specification.structural_mask()[[row, col]] && value != 0.0 {
                anyhow::bail!("averaged {label} changed a structural zero");
            }
            if !specification.estimated_mask()[[row, col]]
                && value != specification.initial()[[row, col]]
            {
                anyhow::bail!("averaged {label} changed a fixed entry");
            }
        }
    }
    cholesky_lower(matrix)
        .map(|_| ())
        .map_err(|error| anyhow::anyhow!("averaged {label} is not positive definite: {error}"))
}

fn validate_average_residuals(
    original_width: usize,
    models: &[(usize, ResidualErrorModel)],
    declarations: &ParametricErrorModels,
) -> Result<()> {
    if original_width != declarations.models().len()
        || models.len() != declarations.models().iter().count()
    {
        anyhow::bail!("averaged residual output collection changed");
    }
    for ((output, model), (declared_output, terminal)) in models.iter().copied().zip(
        declarations
            .models()
            .iter()
            .map(|(index, model)| (index, *model)),
    ) {
        if output != declared_output || output >= original_width {
            anyhow::bail!("averaged residual output indices changed");
        }
        let output_name = declarations
            .output_name(output)
            .ok_or_else(|| anyhow::anyhow!("averaged residual output {output} has no name"))?;
        let components = declarations.combined_component_estimated(output);
        if !declarations.is_estimated(output) && model != terminal {
            anyhow::bail!(
                "averaged residual model changed fixed output '{output_name}' at index {output}"
            );
        }
        if let (
            ResidualErrorModel::Combined { a, b },
            ResidualErrorModel::Combined {
                a: terminal_a,
                b: terminal_b,
            },
        ) = (model, terminal)
        {
            if (!components[0] && a != terminal_a) || (!components[1] && b != terminal_b) {
                anyhow::bail!(
                    "averaged residual model changed a fixed component for output '{output_name}' at index {output}"
                );
            }
        }
        let correlated_components = declarations.correlated_combined_component_estimated(output);
        if let (
            ResidualErrorModel::CorrelatedCombined { a, b, rho },
            ResidualErrorModel::CorrelatedCombined {
                a: terminal_a,
                b: terminal_b,
                rho: terminal_rho,
            },
        ) = (model, terminal)
        {
            if (!correlated_components[0] && a != terminal_a)
                || (!correlated_components[1] && b != terminal_b)
                || (!correlated_components[2] && rho != terminal_rho)
            {
                anyhow::bail!(
                    "averaged correlated-combined model changed a fixed component for output '{output_name}' at index {output}"
                );
            }
        }
        let valid = match model {
            ResidualErrorModel::Constant { a } => a.is_finite() && a > 0.0,
            ResidualErrorModel::Proportional { b } => b.is_finite() && b > 0.0,
            ResidualErrorModel::Exponential { sigma } => sigma.is_finite() && sigma > 0.0,
            ResidualErrorModel::Combined { a, b } => {
                a.is_finite()
                    && b.is_finite()
                    && a >= 0.0
                    && b >= 0.0
                    && (!components[0] || a > 0.0)
                    && (!components[1] || b > 0.0)
            }
            ResidualErrorModel::CorrelatedCombined { a, b, rho } => {
                a.is_finite()
                    && a > 0.0
                    && b.is_finite()
                    && b > 0.0
                    && rho.is_finite()
                    && rho > -1.0
                    && rho < 1.0
            }
        };
        if !valid {
            anyhow::bail!(
                "averaged residual model for output '{output_name}' at index {output} is outside its domain"
            );
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct WarningCount {
    first_iteration: Option<usize>,
    cycles: usize,
    count: usize,
}

impl WarningCount {
    fn record_cycle(&mut self, iteration: usize) {
        self.first_iteration.get_or_insert(iteration);
        self.cycles += 1;
    }

    fn record_count(&mut self, iteration: usize, count: usize) {
        if count == 0 {
            return;
        }
        self.first_iteration.get_or_insert(iteration);
        self.count += count;
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct CovarianceBoundaryRejectionSummary {
    first_iteration: Option<usize>,
    longest_run: usize,
}

fn covariance_boundary_rejection_summary(
    cycles: &[SaemCycleDiagnostics],
    policy: CovarianceStabilityConfig,
    iov: bool,
) -> CovarianceBoundaryRejectionSummary {
    let mut summary = CovarianceBoundaryRejectionSummary::default();
    let mut current_run = 0usize;
    let mut current_start = None;
    for cycle in cycles {
        let (rejected, margin) = if iov {
            (
                cycle.omega_iov_update_rejected,
                cycle.omega_iov_relative_spd_margin,
            )
        } else {
            (cycle.omega_update_rejected, cycle.omega_relative_spd_margin)
        };
        if rejected && margin.is_some_and(|value| value <= policy.minimum_relative_spd_margin) {
            if current_run == 0 {
                current_start = Some(cycle.iteration);
            }
            current_run += 1;
            summary.longest_run = summary.longest_run.max(current_run);
            if current_run >= policy.rejection_window && summary.first_iteration.is_none() {
                summary.first_iteration = current_start;
            }
        } else {
            current_run = 0;
            current_start = None;
        }
    }
    summary
}

fn parametric_warnings(
    cycles: &[SaemCycleDiagnostics],
    covariance_stability: Option<CovarianceStabilityConfig>,
) -> Vec<ParametricWarning> {
    let mut omega = WarningCount::default();
    let mut omega_iov = WarningCount::default();
    let mut eta_non_finite = WarningCount::default();
    let mut eta_block_non_finite = WarningCount::default();
    let mut kappa_non_finite = WarningCount::default();
    let mut residual_rejected = BTreeMap::<String, WarningCount>::new();
    let mut proportional_floor = BTreeMap::<String, WarningCount>::new();
    let mut residual_non_finite = BTreeMap::<String, WarningCount>::new();
    let mut exponential_domain = BTreeMap::<String, WarningCount>::new();
    let mut additive_collapse = BTreeMap::<String, WarningCount>::new();
    let mut optimizer_not_converged = BTreeMap::<String, WarningCount>::new();

    for cycle in cycles {
        if cycle.omega_update_rejected {
            omega.record_cycle(cycle.iteration);
        }
        if cycle.omega_iov_update_rejected {
            omega_iov.record_cycle(cycle.iteration);
        }
        eta_non_finite.record_count(cycle.iteration, cycle.eta_non_finite);
        eta_block_non_finite.record_count(cycle.iteration, cycle.eta_block_non_finite);
        kappa_non_finite.record_count(cycle.iteration, cycle.kappa_non_finite);
        for residual in &cycle.residual_diagnostics {
            if residual.update_rejected {
                residual_rejected
                    .entry(residual.output.clone())
                    .or_default()
                    .record_cycle(cycle.iteration);
            }
            proportional_floor
                .entry(residual.output.clone())
                .or_default()
                .record_count(cycle.iteration, residual.proportional_floor_count);
            residual_non_finite
                .entry(residual.output.clone())
                .or_default()
                .record_count(cycle.iteration, residual.non_finite_prediction_count);
            exponential_domain
                .entry(residual.output.clone())
                .or_default()
                .record_count(cycle.iteration, residual.exponential_domain_violation_count);
            if residual.combined_additive_collapse_warning {
                additive_collapse
                    .entry(residual.output.clone())
                    .or_default()
                    .record_cycle(cycle.iteration);
            }
            if residual.optimizer_converged == Some(false) {
                optimizer_not_converged
                    .entry(residual.output.clone())
                    .or_default()
                    .record_cycle(cycle.iteration);
            }
        }
    }

    let mut warnings = Vec::new();
    if let Some(first_iteration) = omega.first_iteration {
        warnings.push(ParametricWarning::OmegaUpdateRejected {
            first_iteration,
            cycles: omega.cycles,
        });
    }
    if let Some(first_iteration) = omega_iov.first_iteration {
        warnings.push(ParametricWarning::OmegaIovUpdateRejected {
            first_iteration,
            cycles: omega_iov.cycles,
        });
    }
    if let Some(policy) = covariance_stability {
        let omega_boundary = covariance_boundary_rejection_summary(cycles, policy, false);
        if let Some(first_iteration) = omega_boundary.first_iteration {
            warnings.push(ParametricWarning::OmegaBoundaryRejection {
                first_iteration,
                longest_run: omega_boundary.longest_run,
            });
        }
        let omega_iov_boundary = covariance_boundary_rejection_summary(cycles, policy, true);
        if let Some(first_iteration) = omega_iov_boundary.first_iteration {
            warnings.push(ParametricWarning::OmegaIovBoundaryRejection {
                first_iteration,
                longest_run: omega_iov_boundary.longest_run,
            });
        }
    }
    if let Some(first_iteration) = eta_non_finite.first_iteration {
        warnings.push(ParametricWarning::EtaNonFiniteProposals {
            first_iteration,
            count: eta_non_finite.count,
        });
    }
    if let Some(first_iteration) = eta_block_non_finite.first_iteration {
        warnings.push(ParametricWarning::EtaBlockNonFiniteProposals {
            first_iteration,
            count: eta_block_non_finite.count,
        });
    }
    if let Some(first_iteration) = kappa_non_finite.first_iteration {
        warnings.push(ParametricWarning::KappaNonFiniteProposals {
            first_iteration,
            count: kappa_non_finite.count,
        });
    }
    for (output, warning) in residual_rejected {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::ResidualUpdateRejected {
                output,
                first_iteration,
                cycles: warning.cycles,
            });
        }
    }
    for (output, warning) in proportional_floor {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::ProportionalPredictionFloor {
                output,
                first_iteration,
                count: warning.count,
            });
        }
    }
    for (output, warning) in residual_non_finite {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::NonFiniteResidualPrediction {
                output,
                first_iteration,
                count: warning.count,
            });
        }
    }
    for (output, warning) in exponential_domain {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::ExponentialDomainViolation {
                output,
                first_iteration,
                count: warning.count,
            });
        }
    }
    for (output, warning) in additive_collapse {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::CombinedAdditiveCollapse {
                output,
                first_iteration,
                cycles: warning.cycles,
            });
        }
    }
    for (output, warning) in optimizer_not_converged {
        if let Some(first_iteration) = warning.first_iteration {
            warnings.push(ParametricWarning::ResidualOptimizerNotConverged {
                output,
                first_iteration,
                cycles: warning.cycles,
            });
        }
    }
    warnings
}

fn calculate_result_marginal_likelihood<E: Equation>(
    state: &SaemState<E>,
    conditional_modes: &[SubjectConditionalMode],
    conditional_mode_error: Option<&str>,
) -> Option<MarginalLikelihoodDiagnostics> {
    let config = state.config.marginal_likelihood?;
    let n_eta = state.initialization.random_effect_indices.len();
    let n_kappa = state.initialization.iov_effect_indices.len();
    let latent = n_eta > 0 || n_kappa > 0;
    let occasion_indices = state
        .data
        .subjects()
        .iter()
        .map(|subject| {
            if n_kappa == 0 {
                Vec::new()
            } else {
                subject
                    .occasions()
                    .iter()
                    .map(|occasion| occasion.index())
                    .collect()
            }
        })
        .collect::<Vec<Vec<usize>>>();
    let mut flattened_modes = Vec::with_capacity(state.initialization.subject_ids.len());
    let mut converged = Vec::with_capacity(state.initialization.subject_ids.len());
    let mut validation_failures = Vec::with_capacity(state.initialization.subject_ids.len());

    for (subject_index, subject_id) in state.initialization.subject_ids.iter().enumerate() {
        if !latent {
            flattened_modes.push(Vec::new());
            converged.push(None);
            validation_failures.push(None);
            continue;
        }
        let Some(mode) = conditional_modes.get(subject_index) else {
            flattened_modes.push(Vec::new());
            converged.push(None);
            validation_failures.push(Some(
                MarginalLikelihoodFailureReason::MissingConditionalMode,
            ));
            continue;
        };
        let mut validation_failure = None;
        if mode.subject_id != *subject_id {
            validation_failure.get_or_insert(MarginalLikelihoodFailureReason::SubjectIdMismatch {
                expected: subject_id.clone(),
                actual: mode.subject_id.clone(),
            });
        }
        if mode.eta.len() != n_eta {
            validation_failure.get_or_insert(MarginalLikelihoodFailureReason::EtaWidthMismatch {
                expected: n_eta,
                actual: mode.eta.len(),
            });
        }
        if mode.kappas.len() != occasion_indices[subject_index].len() {
            validation_failure.get_or_insert(MarginalLikelihoodFailureReason::KappaCountMismatch {
                expected: occasion_indices[subject_index].len(),
                actual: mode.kappas.len(),
            });
        }
        for (position, kappa) in mode.kappas.iter().enumerate() {
            if let Some(expected) = occasion_indices[subject_index].get(position) {
                if kappa.occasion_index != *expected {
                    validation_failure.get_or_insert(
                        MarginalLikelihoodFailureReason::KappaOccasionMismatch {
                            position,
                            expected: *expected,
                            actual: kappa.occasion_index,
                        },
                    );
                }
            }
            if kappa.values.len() != n_kappa {
                validation_failure.get_or_insert(
                    MarginalLikelihoodFailureReason::KappaWidthMismatch {
                        position,
                        expected: n_kappa,
                        actual: kappa.values.len(),
                    },
                );
            }
        }
        let mut flattened = mode.eta.clone();
        for kappa in &mode.kappas {
            flattened.extend_from_slice(&kappa.values);
        }
        if flattened.iter().any(|value| !value.is_finite()) {
            validation_failure
                .get_or_insert(MarginalLikelihoodFailureReason::NonFiniteModeCoordinate);
        }
        flattened_modes.push(flattened);
        converged.push(Some(mode.converged));
        validation_failures.push(validation_failure);
    }

    let curvature_covariances = conditional_modes
        .iter()
        .map(|mode| {
            mode.uncertainty
                .latent_covariance
                .as_ref()
                .and_then(|rows| matrix_from_rows(rows, rows.len()).ok())
        })
        .collect::<Vec<_>>();
    let subjects = state
        .initialization
        .subject_ids
        .iter()
        .enumerate()
        .map(|(index, subject_id)| MarginalSubject {
            subject_id,
            occasion_indices: &occasion_indices[index],
            mode: &flattened_modes[index],
            mode_converged: converged[index],
            eta_dimension: n_eta,
            kappa_dimension: n_kappa,
            validation_failure: validation_failures[index].clone(),
            curvature_availability: conditional_modes
                .get(index)
                .map(|mode| &mode.uncertainty.status),
            curvature_covariance: curvature_covariances.get(index).and_then(Option::as_ref),
        })
        .collect::<Vec<_>>();
    if let Some(error) = conditional_mode_error {
        return Some(unavailable_population_marginal_likelihood(
            config,
            &subjects,
            MarginalLikelihoodFailureReason::ConditionalModeCalculationFailed(format!(
                "global conditional mode calculation failed: {error}"
            )),
        ));
    }
    Some(calculate_population_marginal_likelihood(
        config,
        &subjects,
        &state.omega,
        state.omega_iov.as_ref(),
        |subject_index, eta, kappas| {
            state
                .score_subject_latents(subject_index, eta, kappas)
                .map(SubjectPosteriorScore::log_posterior)
        },
    ))
}

fn conditional_modes<E: Equation>(state: &SaemState<E>) -> Result<Vec<SubjectConditionalMode>> {
    if !state.compute_map {
        return Ok(Vec::new());
    }

    let n_eta = state.initialization.random_effect_indices.len();
    let n_kappa = state.initialization.iov_effect_indices.len();
    if n_eta == 0 && n_kappa == 0 {
        return Ok(Vec::new());
    }
    let mut modes = Vec::with_capacity(state.initialization.subject_ids.len());
    for (subject_index, subject_id) in state.initialization.subject_ids.iter().enumerate() {
        let eta_start = mean_vectors(state.etas[subject_index].iter().map(|eta| eta.as_slice()))?;
        let occasion_count = if state.omega_iov.is_some() {
            state.data.subjects()[subject_index].occasions().len()
        } else {
            0
        };
        let mut kappa_start = Vec::with_capacity(occasion_count);
        for occasion_position in 0..occasion_count {
            kappa_start.push(mean_vectors(
                state.kappas[subject_index]
                    .iter()
                    .map(|chain| chain[occasion_position].as_slice()),
            )?);
        }
        let mut initial = eta_start;
        for kappa in &kappa_start {
            initial.extend_from_slice(kappa);
        }

        let step_fraction = state.map_initial_step;
        let mut scales = (0..n_eta)
            .map(|index| state.omega[[index, index]].sqrt() * step_fraction)
            .collect::<Vec<_>>();
        if let Some(omega_iov) = state.omega_iov.as_ref() {
            for _ in 0..occasion_count {
                scales.extend(
                    (0..n_kappa).map(|index| omega_iov[[index, index]].sqrt() * step_fraction),
                );
            }
        }
        for scale in &mut scales {
            *scale = scale.max(1e-6);
        }

        let solution = optimize_conditional_mode(
            initial,
            &scales,
            state.map_max_iterations as u64,
            state.map_sd_tolerance,
            |coordinates| {
                let (eta, kappas) = unflatten_latents(coordinates, n_eta, occasion_count, n_kappa);
                match state.score_subject_latents(subject_index, eta, &kappas) {
                    Ok(score) if score.log_posterior().is_finite() => -score.log_posterior(),
                    _ => f64::INFINITY,
                }
            },
        )?;
        let mut coordinates = (0..n_eta)
            .map(|index| JointLatentCoordinate {
                index,
                name: format!("eta:{}", state.initialization.random_effect_names[index]),
                kind: JointLatentCoordinateKind::Eta {
                    parameter_index: state.initialization.random_effect_indices[index],
                },
                prior_sd: state.omega[[index, index]].sqrt(),
            })
            .collect::<Vec<_>>();
        if let Some(omega_iov) = state.omega_iov.as_ref() {
            for occasion_position in 0..occasion_count {
                let occasion_index =
                    state.data.subjects()[subject_index].occasions()[occasion_position].index();
                for effect_index in 0..n_kappa {
                    coordinates.push(JointLatentCoordinate {
                        index: n_eta + occasion_position * n_kappa + effect_index,
                        name: format!(
                            "kappa:{occasion_index}:{}",
                            state.initialization.iov_effect_names[effect_index]
                        ),
                        kind: JointLatentCoordinateKind::Kappa {
                            occasion_index,
                            effect_index,
                            parameter_index: state.initialization.iov_effect_indices[effect_index],
                        },
                        prior_sd: omega_iov[[effect_index, effect_index]].sqrt(),
                    });
                }
            }
        }
        let prior_sds = coordinates
            .iter()
            .map(|coordinate| coordinate.prior_sd)
            .collect::<Vec<_>>();
        let mode_metadata = ConditionalModeMetadata {
            converged: solution.converged,
            iterations: solution.iterations,
            objective_value: solution.objective,
            termination_message: solution.termination.clone(),
        };
        let uncertainty = conditional_mode_curvature(
            &solution.coordinates,
            &prior_sds,
            &coordinates,
            &mode_metadata,
            |coordinates| {
                let (eta, kappas) = unflatten_latents(coordinates, n_eta, occasion_count, n_kappa);
                match state.score_subject_latents(subject_index, eta, &kappas) {
                    Ok(score) if score.log_posterior().is_finite() => -score.log_posterior(),
                    _ => f64::INFINITY,
                }
            },
        );
        let (eta, kappas) =
            unflatten_latents(&solution.coordinates, n_eta, occasion_count, n_kappa);
        let parameters = state.individual_parameters_from_eta(subject_index, eta)?;
        let kappa_estimates = kappas
            .into_iter()
            .enumerate()
            .map(|(occasion_position, values)| OccasionKappaEstimate {
                subject_id: subject_id.clone(),
                occasion_index: state.data.subjects()[subject_index].occasions()[occasion_position]
                    .index(),
                values,
            })
            .collect();
        modes.push(SubjectConditionalMode {
            subject_id: subject_id.clone(),
            eta: eta.to_vec(),
            kappas: kappa_estimates,
            parameters,
            objective: solution.objective,
            converged: solution.converged,
            iterations: solution.iterations,
            termination: solution.termination,
            uncertainty,
        });
    }
    Ok(modes)
}

fn unflatten_latents(
    coordinates: &[f64],
    n_eta: usize,
    occasion_count: usize,
    n_kappa: usize,
) -> (&[f64], Vec<Vec<f64>>) {
    let eta = &coordinates[..n_eta];
    let kappas = (0..occasion_count)
        .map(|occasion| {
            let start = n_eta + occasion * n_kappa;
            coordinates[start..start + n_kappa].to_vec()
        })
        .collect();
    (eta, kappas)
}

fn mean_vectors<'a>(vectors: impl IntoIterator<Item = &'a [f64]>) -> Result<Vec<f64>> {
    let mut vectors = vectors.into_iter();
    let Some(first) = vectors.next() else {
        anyhow::bail!("cannot summarize random effects without chains");
    };
    let mut mean = first.to_vec();
    let mut count = 1usize;
    for vector in vectors {
        if vector.len() != mean.len() {
            anyhow::bail!("random-effect chains have inconsistent dimensions");
        }
        for (sum, value) in mean.iter_mut().zip(vector) {
            *sum += value;
        }
        count += 1;
    }
    for value in &mut mean {
        *value /= count as f64;
    }
    Ok(mean)
}

fn zero_etas(n_subjects: usize, n_chains: usize, n_parameters: usize) -> Vec<Vec<Vec<f64>>> {
    vec![vec![vec![0.0; n_parameters]; n_chains]; n_subjects]
}

fn zero_kappas(
    occasion_counts: &[usize],
    n_chains: usize,
    n_kappa: usize,
) -> Vec<Vec<Vec<Vec<f64>>>> {
    occasion_counts
        .iter()
        .map(|&n_occasions| vec![vec![vec![0.0; n_kappa]; n_occasions]; n_chains])
        .collect()
}

fn second_moment_from_etas(etas: &[Vec<Vec<f64>>]) -> Result<Array2<f64>> {
    let mut samples = etas.iter().flat_map(|subject_chains| subject_chains.iter());
    let Some(first) = samples.next() else {
        anyhow::bail!("cannot update omega without subject-chain samples");
    };
    let dimension = first.len();
    let mut second_moment = Array2::zeros((dimension, dimension));
    let mut count = 0usize;
    for eta in std::iter::once(first).chain(samples) {
        if eta.len() != dimension {
            anyhow::bail!("eta samples have inconsistent dimensions");
        }
        for row in 0..dimension {
            for col in 0..dimension {
                second_moment[[row, col]] += eta[row] * eta[col];
            }
        }
        count += 1;
    }
    second_moment.mapv_inplace(|value| value / count as f64);
    Ok(second_moment)
}

fn covariance_from_kappas(kappas: &[Vec<Vec<Vec<f64>>>]) -> Result<Array2<f64>> {
    let mut samples = kappas
        .iter()
        .flat_map(|subject_chains| subject_chains.iter())
        .flat_map(|chains| chains.iter());
    let Some(first) = samples.next() else {
        anyhow::bail!("cannot update omega_iov without occasion samples");
    };
    let dimension = first.len();
    let mut covariance = Array2::zeros((dimension, dimension));
    let mut count = 0usize;
    for kappa in std::iter::once(first).chain(samples) {
        if kappa.len() != dimension {
            anyhow::bail!("kappa samples have inconsistent dimensions");
        }
        for row in 0..dimension {
            for col in 0..dimension {
                covariance[[row, col]] += kappa[row] * kappa[col];
            }
        }
        count += 1;
    }
    covariance.mapv_inplace(|value| value / count as f64);
    Ok(covariance)
}

fn correlated_random_walk(
    current: &[f64],
    lower: &[Vec<f64>],
    standard_normals: &[f64],
    scale: f64,
) -> Result<Vec<f64>> {
    anyhow::ensure!(
        lower.len() == current.len()
            && standard_normals.len() == current.len()
            && lower
                .iter()
                .enumerate()
                .all(|(row, values)| values.len() > row),
        "correlated random-walk dimensions do not match"
    );
    Ok((0..current.len())
        .map(|row| {
            let perturbation = (0..=row)
                .map(|column| lower[row][column] * standard_normals[column])
                .sum::<f64>();
            current[row] + scale * perturbation
        })
        .collect())
}

fn initial_proposal_step_sizes(omega: &Array2<f64>, rw_init: f64) -> Vec<f64> {
    (0..omega.nrows())
        .map(|index| omega[[index, index]].max(f64::EPSILON).sqrt() * rw_init)
        .collect()
}

fn adapt_component_step_size(current: f64, acceptance_rate: f64) -> f64 {
    adapt_block_step_size(current, acceptance_rate, COMPONENT_TARGET_ACCEPTANCE)
}

fn adapt_block_step_size(current: f64, acceptance_rate: f64, target: f64) -> f64 {
    if acceptance_rate > target {
        (current * PROPOSAL_SCALE_INCREASE).min(MAX_PROPOSAL_SCALE)
    } else {
        (current * PROPOSAL_SCALE_DECREASE).max(MIN_PROPOSAL_SCALE)
    }
}

fn zero_eta_subject_phi(
    population_parameters: &[f64],
    initialization: &SaemInitialization,
) -> Result<Vec<Vec<f64>>> {
    let phi = population_phi(population_parameters, &initialization.parameter_scales)?;
    Ok(vec![phi; initialization.subject_ids.len()])
}

fn negative_log_likelihood(subject_log_likelihoods: &[f64]) -> f64 {
    if subject_log_likelihoods.iter().any(|ll| !ll.is_finite()) {
        f64::INFINITY
    } else {
        -subject_log_likelihoods.iter().sum::<f64>()
    }
}

fn count_observations(data: &Data) -> usize {
    data.subjects()
        .iter()
        .flat_map(|subject| subject.occasions())
        .flat_map(|occasion| occasion.events())
        .filter(|event| matches!(event, Event::Observation(_)))
        .count()
}

fn n_chains(config: &SaemConfig, n_subjects: usize) -> usize {
    if n_subjects > 0 && n_subjects < 50 && config.n_chains == 1 {
        ((50.0 / n_subjects as f64).ceil() as usize).max(1)
    } else {
        config.n_chains
    }
}

fn initial_parameter_row<'a>(
    parameters: impl IntoIterator<Item = &'a UnboundedParameter>,
) -> Vec<f64> {
    parameters
        .into_iter()
        .map(initial_parameter_value)
        .collect()
}

fn initial_parameter_value(parameter: &UnboundedParameter) -> f64 {
    if let Some(initial) = parameter.initial {
        return initial;
    }

    match parameter.scale {
        ParameterScale::Identity | ParameterScale::Log => 1.0,
        ParameterScale::Logit { lower, upper } | ParameterScale::Probit { lower, upper } => {
            0.5 * (lower + upper)
        }
    }
}

fn information_failure_status(reason: String) -> InformationStatus {
    if reason.contains("censored") {
        InformationStatus::Unsupported(reason)
    } else if reason.contains("non-finite") {
        InformationStatus::NonFinite
    } else {
        InformationStatus::Ineligible(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parametric::transforms::{phi_to_psi, psi_to_phi};
    use crate::estimation::parametric::ParametricPrior;
    use crate::estimation::{EstimationProblem, Iov, Omega, ParametricErrorModel};
    use crate::model::Parameter;
    use crate::results::{
        FitResult, PopulationUncertaintyDiagnostics, PopulationUncertaintyRegularization,
        PopulationUncertaintyStatus,
    };
    use pharmsol::prelude::*;
    use pharmsol::SubjectBuilderExt;

    #[test]
    fn finite_improvement_is_eligible_without_a_convergence_flag() {
        assert!(non_iiv_candidate_improves(10.0, 9.0));
        assert!(!non_iiv_candidate_improves(10.0, 10.0));
        assert!(!non_iiv_candidate_improves(10.0, f64::NAN));
    }

    #[test]
    fn censored_information_failure_has_explicit_unsupported_status() {
        let reason = "analytic information is unsupported for censored observations".to_string();
        assert_eq!(
            information_failure_status(reason.clone()),
            InformationStatus::Unsupported(reason)
        );
    }

    fn one_compartment_metadata() -> pharmsol::equation::ModelMetadata {
        equation::metadata::new("one_compartment_saem")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["0"])
            .route(equation::Route::bolus("0").to_state("central"))
    }

    fn one_compartment() -> pharmsol::ODE {
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke);
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, v);
                y[0] = x[0] / v;
            },
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(one_compartment_metadata())
        .unwrap()
    }

    fn sparse_second_output_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        let equation = equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke);
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, v);
                y[0] = x[0];
                y[1] = x[0] / v;
            },
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(2)
        .with_metadata(
            equation::metadata::new("sparse_second_output")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["unmeasured", "measured"])
                .route(equation::Route::bolus("dose").to_state("central")),
        )
        .unwrap();
        let data = Data::new(vec![Subject::builder("sparse")
            .bolus(0.0, 100.0, "dose")
            .observation(1.0, 8.0, "measured")
            .observation(2.0, 6.0, "measured")
            .build()]);

        EstimationProblem::parametric(equation, data)
            .parameter(
                Parameter::log("ke")
                    .with_initial(0.2)
                    .fixed()
                    .without_random_effect(),
            )
            .parameter(
                Parameter::log("v")
                    .with_initial(10.0)
                    .fixed()
                    .without_random_effect(),
            )
            .error_model("measured", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    fn mixed_residual_output_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        let equation = equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke);
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, v);
                y[0] = x[0] / v;
                y[1] = x[0] / v;
            },
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(2)
        .with_metadata(
            equation::metadata::new("mixed_residual_outputs")
                .parameters(["ke", "v"])
                .states(["central"])
                .outputs(["fixed", "mixed"])
                .route(equation::Route::bolus("dose").to_state("central")),
        )
        .expect("mixed residual equation metadata should validate");
        let data = Data::new(vec![Subject::builder("mixed")
            .bolus(0.0, 100.0, "dose")
            .observation(1.0, 8.5, "fixed")
            .observation(2.0, 6.5, "fixed")
            .observation(1.0, 8.0, "mixed")
            .observation(2.0, 6.0, "mixed")
            .build()]);

        EstimationProblem::parametric(equation, data)
            .parameter(
                Parameter::log("ke")
                    .with_initial(0.2)
                    .fixed()
                    .without_random_effect(),
            )
            .parameter(
                Parameter::log("v")
                    .with_initial(10.0)
                    .fixed()
                    .without_random_effect(),
            )
            .error_model(
                "fixed",
                ParametricErrorModel::new(ResidualErrorModel::constant(0.5)).fixed(),
            )
            .error_model(
                "mixed",
                ParametricErrorModel::new(ResidualErrorModel::combined(0.0, 0.1))
                    .fixed_combined_additive(),
            )
            .build()
            .expect("mixed residual output problem should validate")
    }

    fn data() -> Data {
        Data::new(vec![
            Subject::builder("s1")
                .bolus(0.0, 100.0, "0")
                .observation(1.0, 12.0, "0")
                .observation(4.0, 4.0, "0")
                .build(),
            Subject::builder("s2")
                .bolus(0.0, 80.0, "0")
                .observation(0.5, 9.0, "0")
                .observation(3.0, 2.5, "0")
                .build(),
        ])
    }

    fn covariate_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        let subjects = [-1.0, 0.0, 1.0]
            .into_iter()
            .enumerate()
            .map(|(index, wt)| {
                Subject::builder(format!("cov{index}"))
                    .covariate("wt", 0.0, wt)
                    .covariate("sex", 0.0, if index == 2 { 1.0 } else { 0.0 })
                    .bolus(0.0, 100.0, "0")
                    .observation(1.0, 8.0 + index as f64, "0")
                    .build()
            })
            .collect();
        EstimationProblem::parametric(one_compartment(), Data::new(subjects))
            .parameter(Parameter::log("ke").with_initial(0.2))
            .parameter(Parameter::log("v").with_initial(10.0))
            .covariate_effect(
                crate::estimation::parametric::CovariateEffect::continuous("ke", "wt", 0.0)
                    .with_initial(0.0),
            )
            .covariate_effect(
                crate::estimation::parametric::CovariateEffect::categorical("v", "sex", 0.0, 1.0)
                    .with_initial(0.0),
            )
            .error_model(
                "0",
                ParametricErrorModel::new(ResidualErrorModel::constant(1.0)).fixed(),
            )
            .build()
            .unwrap()
    }

    fn fixed_covariate_iiv_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        let subjects = [-1.0, 1.0]
            .into_iter()
            .enumerate()
            .map(|(index, wt)| {
                Subject::builder(format!("fixed-cov-iiv-{index}"))
                    .covariate("wt", 0.0, wt)
                    .bolus(0.0, 100.0, "0")
                    .observation(1.0, 5.0 + index as f64, "0")
                    .build()
            })
            .collect();
        EstimationProblem::parametric(one_compartment(), Data::new(subjects))
            .parameter(Parameter::log("ke").with_initial(0.2).fixed())
            .parameter(
                Parameter::log("v")
                    .with_initial(10.0)
                    .fixed()
                    .without_random_effect(),
            )
            .omega(Omega::diagonal([("ke", 1.0)]))
            .covariate_effect(
                crate::estimation::parametric::CovariateEffect::continuous("ke", "wt", 0.0)
                    .with_initial(0.0)
                    .fixed(),
            )
            .error_model(
                "0",
                ParametricErrorModel::new(ResidualErrorModel::constant(1.0)).fixed(),
            )
            .build()
            .unwrap()
    }

    fn fixed_covariate_without_iiv_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        let subjects = [0.0, 1.0]
            .into_iter()
            .enumerate()
            .map(|(index, wt)| {
                Subject::builder(format!("fixed-cov-{index}"))
                    .covariate("wt", 0.0, wt)
                    .bolus(0.0, 100.0, "0")
                    .observation(1.0, 5.0 + index as f64, "0")
                    .build()
            })
            .collect();
        EstimationProblem::parametric(one_compartment(), Data::new(subjects))
            .parameter(
                Parameter::log("ke")
                    .with_initial(0.2)
                    .fixed()
                    .without_random_effect(),
            )
            .parameter(
                Parameter::log("v")
                    .with_initial(10.0)
                    .fixed()
                    .without_random_effect(),
            )
            .covariate_effect(
                crate::estimation::parametric::CovariateEffect::continuous("ke", "wt", 0.0)
                    .with_initial(0.2)
                    .fixed(),
            )
            .error_model(
                "0",
                ParametricErrorModel::new(ResidualErrorModel::constant(1.0)).fixed(),
            )
            .build()
            .unwrap()
    }

    fn problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        EstimationProblem::parametric(one_compartment(), data())
            .parameter(Parameter::log("ke").with_initial(0.2))
            .parameter(Parameter::log("v").with_initial(10.0))
            .error_model(
                "0",
                ParametricErrorModel::new(ResidualErrorModel::combined(0.5, 0.1)).fixed(),
            )
            .build()
            .unwrap()
    }

    fn constant_error_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        EstimationProblem::parametric(one_compartment(), data())
            .parameter(Parameter::log("ke").with_initial(0.2))
            .parameter(Parameter::log("v").with_initial(10.0))
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    fn partial_iiv_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        EstimationProblem::parametric(one_compartment(), data())
            .parameter(Parameter::log("ke").with_initial(0.2))
            .parameter(
                Parameter::log("v")
                    .with_initial(10.0)
                    .fixed()
                    .without_random_effect(),
            )
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    fn iov_data() -> Data {
        Data::new(vec![Subject::builder("s1")
            .bolus(0.0, 100.0, "0")
            .observation(1.0, 12.0, "0")
            .reset()
            .bolus(0.0, 100.0, "0")
            .observation(1.0, 10.0, "0")
            .build()])
    }

    fn iov_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        EstimationProblem::parametric(one_compartment(), iov_data())
            .parameter(Parameter::log("ke").with_initial(0.2))
            .parameter(Parameter::log("v").with_initial(10.0))
            .iov(Iov::diagonal([("ke", 0.1)]))
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    fn markov_iov_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        EstimationProblem::parametric(one_compartment(), iov_data())
            .parameter(Parameter::log("ke").with_initial(0.2).fixed())
            .parameter(
                Parameter::log("v")
                    .with_initial(10.0)
                    .fixed()
                    .without_random_effect(),
            )
            .omega(Omega::new().fixed_variance("ke", 0.1))
            .iov(Iov::new().fixed_variance("ke", 0.1))
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    fn uneven_iov_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        let data = Data::new(vec![
            Subject::builder("one")
                .bolus(0.0, 100.0, "0")
                .observation(1.0, 12.0, "0")
                .build(),
            Subject::builder("two")
                .bolus(0.0, 100.0, "0")
                .observation(1.0, 12.0, "0")
                .reset()
                .bolus(0.0, 100.0, "0")
                .observation(1.0, 10.0, "0")
                .build(),
            Subject::builder("three")
                .bolus(0.0, 100.0, "0")
                .observation(1.0, 12.0, "0")
                .reset()
                .bolus(0.0, 100.0, "0")
                .observation(1.0, 10.0, "0")
                .reset()
                .bolus(0.0, 100.0, "0")
                .observation(1.0, 11.0, "0")
                .build(),
        ]);
        EstimationProblem::parametric(one_compartment(), data)
            .parameter(Parameter::log("ke").with_initial(0.2))
            .parameter(Parameter::log("v").with_initial(10.0))
            .iov(Iov::diagonal([("ke", 0.1)]))
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    fn configured_iov_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        EstimationProblem::parametric(one_compartment(), iov_data())
            .parameter(Parameter::log("ke").with_initial(0.2))
            .parameter(Parameter::log("v").with_initial(10.0))
            .iov(
                Iov::diagonal([("ke", 0.10)])
                    .fixed_variance("v", 0.20)
                    .fixed_covariance("ke", "v", 0.05),
            )
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    fn ordered_metadata_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        EstimationProblem::parametric(one_compartment(), iov_data())
            .parameter(Parameter::real("ke").with_initial(0.2))
            .parameter(
                Parameter::log("v")
                    .with_initial(10.0)
                    .fixed()
                    .without_random_effect(),
            )
            .iov(Iov::diagonal([("v", 0.20)]))
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    fn configured_omega_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        EstimationProblem::parametric(one_compartment(), data())
            .parameter(Parameter::log("ke").with_initial(0.2))
            .parameter(Parameter::log("v").with_initial(10.0))
            .omega(Omega::diagonal([("ke", 0.25)]).fixed_variance("v", 0.5))
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    fn correlated_omega_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        EstimationProblem::parametric(one_compartment(), data())
            .parameter(Parameter::log("ke").with_initial(0.2))
            .parameter(Parameter::log("v").with_initial(10.0))
            .omega(Omega::diagonal([("ke", 0.25), ("v", 0.25)]).covariance("ke", "v", 0.20))
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    fn fixed_population_iiv_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        EstimationProblem::parametric(one_compartment(), data())
            .parameter(Parameter::log("ke").with_initial(0.2).fixed())
            .parameter(
                Parameter::log("v")
                    .with_initial(10.0)
                    .fixed()
                    .without_random_effect(),
            )
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    fn fixed_no_iiv_problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        EstimationProblem::parametric(one_compartment(), data())
            .parameter(
                Parameter::log("ke")
                    .with_initial(0.2)
                    .fixed()
                    .without_random_effect(),
            )
            .parameter(
                Parameter::log("v")
                    .with_initial(10.0)
                    .fixed()
                    .without_random_effect(),
            )
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap()
    }

    #[test]
    fn initialization_builds_initial_objective() {
        let initialization =
            SaemInitialization::create(&problem(), &SaemConfig::default()).unwrap();

        assert_eq!(
            initialization.initial_population_parameters,
            vec![0.2, 10.0]
        );
        assert_eq!(initialization.initial_subject_log_likelihoods.len(), 2);
        assert!(initialization.initial_negative_log_likelihood.is_finite());
    }

    #[test]
    fn initialization_rejects_estimated_iiv_variance_below_floor() {
        let mut config = SaemConfig::new();
        config.omega_min_variance = 0.3;

        let error = SaemInitialization::create(&configured_omega_problem(), &config)
            .unwrap_err()
            .to_string();

        assert!(error.contains(
            "initial Omega variance for estimated effect 'ke' (0.25) is below configured omega_min_variance (0.3)"
        ));
    }

    #[test]
    fn initialization_rejects_estimated_iov_variance_below_floor() {
        let config = SaemConfig::new().omega_iov_min_variance(0.11);

        let error = SaemInitialization::create(&configured_iov_problem(), &config)
            .unwrap_err()
            .to_string();

        assert!(error.contains(
            "initial Omega_IOV variance for estimated effect 'ke' (0.1) is below configured omega_iov_min_variance (0.11)"
        ));
    }

    #[test]
    fn initialization_floor_exempts_fixed_covariance_diagonals() {
        let problem = EstimationProblem::parametric(one_compartment(), data())
            .parameter(Parameter::log("ke").with_initial(0.2))
            .parameter(Parameter::log("v").with_initial(10.0))
            .omega(Omega::diagonal([("ke", 0.25)]).fixed_variance("v", 0.01))
            .error_model("0", ResidualErrorModel::constant(1.0))
            .build()
            .unwrap();
        let mut config = SaemConfig::new();
        config.omega_min_variance = 0.1;

        let initialization = SaemInitialization::create(&problem, &config).unwrap();

        assert_eq!(initialization.omega.initial()[[0, 0]], 0.25);
        assert_eq!(initialization.omega.initial()[[1, 1]], 0.01);
        assert!(!initialization.omega.estimated_mask()[[1, 1]]);
    }

    #[test]
    fn schedule_counts_real_internal_phases() {
        let config = SaemConfig::new()
            .burn_in(100)
            .k1_iterations(400)
            .k2_iterations(700);
        let schedule = SaemSchedule::from_config(&config);
        let counts = (1..=schedule.total_iterations).fold([0_usize; 3], |mut counts, cycle| {
            match schedule.phase(cycle) {
                SaemPhase::BurnIn => counts[0] += 1,
                SaemPhase::Exploration => counts[1] += 1,
                SaemPhase::Smoothing => counts[2] += 1,
            }
            counts
        });

        assert_eq!(counts, [100, 300, 700]);
        assert_eq!(schedule.total_iterations, 1100);
    }

    #[test]
    fn covariate_omega_cap_applies_only_during_exploration() {
        assert_eq!(
            covariate_omega_update_maximum_fraction(true, SaemPhase::BurnIn, 0.1),
            1.0
        );
        assert_eq!(
            covariate_omega_update_maximum_fraction(true, SaemPhase::Exploration, 0.1),
            0.1
        );
        assert_eq!(
            covariate_omega_update_maximum_fraction(true, SaemPhase::Smoothing, 0.1),
            1.0
        );
        assert_eq!(
            covariate_omega_update_maximum_fraction(false, SaemPhase::Exploration, 0.1),
            1.0
        );
    }

    #[derive(Debug)]
    struct CommonMomentCycle {
        expected_phi: Vec<Vec<f64>>,
        global_second_moment: Array2<f64>,
        beta: Vec<f64>,
        subject_means: Vec<Vec<f64>>,
        covariance_target: Array2<f64>,
        omega: Array2<f64>,
    }

    fn common_moment_cycle(
        statistics: &mut CovariateSufficientStatistics,
        observed: &CovariateSufficientStatistics,
        gain: f64,
        designs: &[Array2<f64>],
        current_omega: &Array2<f64>,
        omega_specification: &ResolvedOmega,
    ) -> Result<CommonMomentCycle> {
        statistics.stochastic_update(observed, gain)?;
        let offsets = vec![vec![0.0]; designs.len()];
        let beta = solve_covariate_gls(CovariateGlsProblem {
            design: designs,
            expected_phi: &statistics.expected_phi,
            offset: &offsets,
            omega: current_omega,
        })?;
        let subject_means = designs
            .iter()
            .map(|design| vec![design[[0, 0]] * beta[0] + design[[0, 1]] * beta[1]])
            .collect::<Vec<_>>();
        let covariance_target = subject_centered_omega(
            &statistics.global_second_moment,
            &statistics.expected_phi,
            &subject_means,
        )?;
        let omega = omega_specification
            .update_with_status(current_omega, &covariance_target, 1e-6)?
            .matrix;
        Ok(CommonMomentCycle {
            expected_phi: statistics.expected_phi.clone(),
            global_second_moment: statistics.global_second_moment.clone(),
            beta,
            subject_means,
            covariance_target,
            omega,
        })
    }

    fn assert_nested_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (actual_value, expected_value) in actual_row.iter().zip(expected_row) {
                assert!((actual_value - expected_value).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn common_gain_raw_moments_are_coherent_cycle_by_cycle() {
        let designs = [-1.0, 0.0, 1.0]
            .into_iter()
            .map(|covariate| ndarray::array![[1.0, covariate]])
            .collect::<Vec<_>>();
        let parameters = [Parameter::log("x")].into_iter().collect();
        let prior =
            ParametricPrior::new(parameters, Some(Omega::diagonal([("x", 1.0)])), None).unwrap();
        let mut current_omega = prior.omega().clone();
        let mut statistics = CovariateSufficientStatistics {
            expected_phi: vec![vec![0.0]; 3],
            global_second_moment: ndarray::array![[1.0]],
        };
        let exploration_observed = CovariateSufficientStatistics::from_subject_chains(&[
            vec![vec![-1.4], vec![-0.6]],
            vec![vec![-0.4], vec![0.4]],
            vec![vec![0.6], vec![1.4]],
        ])
        .unwrap();
        let first_smoothing_observed = CovariateSufficientStatistics::from_subject_chains(&[
            vec![vec![-1.5], vec![-0.5]],
            vec![vec![0.5], vec![1.5]],
            vec![vec![2.5], vec![3.5]],
        ])
        .unwrap();
        let second_smoothing_observed = CovariateSufficientStatistics::from_subject_chains(&[
            vec![vec![-3.0], vec![-1.0]],
            vec![vec![-1.0], vec![1.0]],
            vec![vec![1.0], vec![3.0]],
        ])
        .unwrap();

        let burn = common_moment_cycle(
            &mut statistics,
            &exploration_observed,
            0.0,
            &designs,
            &current_omega,
            prior.resolved_omega(),
        )
        .unwrap();
        assert_eq!(burn.expected_phi, vec![vec![0.0]; 3]);
        assert_eq!(burn.global_second_moment, ndarray::array![[1.0]]);
        assert_eq!(burn.beta, vec![0.0, 0.0]);
        assert_eq!(burn.subject_means, vec![vec![0.0]; 3]);
        assert_eq!(burn.covariance_target, ndarray::array![[1.0]]);
        assert_eq!(burn.omega, ndarray::array![[1.0]]);

        let exploration = common_moment_cycle(
            &mut statistics,
            &exploration_observed,
            1.0,
            &designs,
            &current_omega,
            prior.resolved_omega(),
        )
        .unwrap();
        assert_nested_close(
            &exploration.expected_phi,
            &[vec![-1.0], vec![0.0], vec![1.0]],
        );
        assert!((exploration.global_second_moment[[0, 0]] - 62.0 / 75.0).abs() <= 1e-12);
        assert!((exploration.beta[0] - 0.0).abs() <= 1e-12);
        assert!((exploration.beta[1] - 1.0).abs() <= 1e-12);
        assert_nested_close(&exploration.subject_means, &exploration.expected_phi);
        assert!((exploration.covariance_target[[0, 0]] - 0.16).abs() <= 1e-12);
        assert!((exploration.omega[[0, 0]] - 0.16).abs() <= 1e-12);
        current_omega = exploration.omega.clone();

        let first_smoothing = common_moment_cycle(
            &mut statistics,
            &first_smoothing_observed,
            1.0,
            &designs,
            &current_omega,
            prior.resolved_omega(),
        )
        .unwrap();
        assert_nested_close(
            &first_smoothing.expected_phi,
            &[vec![-1.0], vec![1.0], vec![3.0]],
        );
        assert!((first_smoothing.global_second_moment[[0, 0]] - 47.0 / 12.0).abs() <= 1e-12);
        assert!((first_smoothing.beta[0] - 1.0).abs() <= 1e-12);
        assert!((first_smoothing.beta[1] - 2.0).abs() <= 1e-12);
        assert_nested_close(
            &first_smoothing.subject_means,
            &first_smoothing.expected_phi,
        );
        assert!((first_smoothing.covariance_target[[0, 0]] - 0.25).abs() <= 1e-12);
        assert!((first_smoothing.omega[[0, 0]] - 0.25).abs() <= 1e-12);
        current_omega = first_smoothing.omega.clone();

        let second_smoothing = common_moment_cycle(
            &mut statistics,
            &second_smoothing_observed,
            0.5,
            &designs,
            &current_omega,
            prior.resolved_omega(),
        )
        .unwrap();
        assert_nested_close(
            &second_smoothing.expected_phi,
            &[vec![-1.5], vec![0.5], vec![2.5]],
        );
        assert!((second_smoothing.global_second_moment[[0, 0]] - 91.0 / 24.0).abs() <= 1e-12);
        assert!((second_smoothing.beta[0] - 0.5).abs() <= 1e-12);
        assert!((second_smoothing.beta[1] - 2.0).abs() <= 1e-12);
        assert_nested_close(
            &second_smoothing.subject_means,
            &second_smoothing.expected_phi,
        );
        assert!((second_smoothing.covariance_target[[0, 0]] - 0.875).abs() <= 1e-12);
        assert!((second_smoothing.omega[[0, 0]] - 0.875).abs() <= 1e-12);

        for cycle in [burn, exploration, first_smoothing, second_smoothing] {
            let mean_square = cycle
                .expected_phi
                .iter()
                .map(|row| row[0] * row[0])
                .sum::<f64>()
                / cycle.expected_phi.len() as f64;
            assert!(cycle.global_second_moment[[0, 0]] + 1e-12 >= mean_square);
            assert!(cycle.covariance_target[[0, 0]] >= -1e-12);
        }
    }

    #[test]
    fn coherent_covariance_target_precedes_structured_gem_constraints() {
        let coherent_target = ndarray::array![[0.002, 0.0], [0.0, 0.04]];
        assert!(cholesky_lower(&coherent_target).is_ok());
        let parameters = [Parameter::log("ke"), Parameter::log("v")]
            .into_iter()
            .collect();
        let prior = ParametricPrior::new(
            parameters,
            Some(
                Omega::new()
                    .variance("ke", 0.02)
                    .fixed_variance("v", 0.04)
                    .fixed_covariance("ke", "v", 0.012),
            ),
            None,
        )
        .unwrap();

        let constrained = prior
            .resolved_omega()
            .update_with_status(prior.omega(), &coherent_target, 0.0)
            .unwrap();

        assert_eq!(coherent_target[[0, 0]], 0.002);
        assert!((constrained.matrix[[0, 0]] - 0.0092).abs() <= 1e-10);
        assert_eq!(constrained.matrix[[0, 1]], 0.012);
        assert_eq!(constrained.matrix[[1, 1]], 0.04);
        assert_ne!(constrained.matrix, coherent_target);
    }

    #[test]
    fn covariate_update_uses_common_moments_and_no_second_smoothing_gain() {
        let mut statistics =
            CovariateSufficientStatistics::from_subject_chains(&[vec![vec![0.0], vec![2.0]]])
                .unwrap();
        let exploration_observed =
            CovariateSufficientStatistics::from_subject_chains(&[vec![vec![2.0], vec![4.0]]])
                .unwrap();
        statistics
            .stochastic_update(&exploration_observed, 1.0)
            .unwrap();
        assert_eq!(statistics.expected_phi, vec![vec![3.0]]);
        assert_eq!(statistics.global_second_moment, ndarray::array![[10.0]]);
        let exploration_variance = statistics.global_second_moment[[0, 0]]
            - statistics.expected_phi[0][0] * statistics.expected_phi[0][0];
        let exploration_candidate = ndarray::array![[exploration_variance]];
        assert_eq!(exploration_candidate, ndarray::array![[1.0]]);

        let parameters = [Parameter::log("x")].into_iter().collect();
        let prior =
            ParametricPrior::new(parameters, Some(Omega::diagonal([("x", 0.25)])), None).unwrap();
        let exploration = prior
            .resolved_omega()
            .update_with_status_and_max_fraction(
                prior.omega(),
                &exploration_candidate,
                0.0,
                covariate_omega_update_maximum_fraction(true, SaemPhase::Exploration, 0.1),
            )
            .unwrap();
        assert!((exploration.matrix[[0, 0]] - 0.325).abs() <= 1e-12);

        let smoothing_observed =
            CovariateSufficientStatistics::from_subject_chains(&[vec![vec![4.0], vec![6.0]]])
                .unwrap();
        statistics
            .stochastic_update(&smoothing_observed, 0.5)
            .unwrap();
        assert_eq!(statistics.expected_phi, vec![vec![4.0]]);
        assert_eq!(statistics.global_second_moment, ndarray::array![[18.0]]);
        let smoothing_variance = statistics.global_second_moment[[0, 0]]
            - statistics.expected_phi[0][0] * statistics.expected_phi[0][0];
        let smoothing_candidate = ndarray::array![[smoothing_variance]];
        assert_eq!(smoothing_candidate, ndarray::array![[2.0]]);

        let smoothing = prior
            .resolved_omega()
            .update_with_status(&exploration.matrix, &smoothing_candidate, 0.0)
            .unwrap();
        assert_eq!(smoothing.matrix, smoothing_candidate);
    }

    #[test]
    fn covariate_state_m_step_caps_exploration_and_does_not_resmooth_omega() {
        let config = SaemConfig::new()
            .n_chains(2)
            .mcmc_iterations(1)
            .burn_in(1)
            .k1_iterations(2)
            .k2_iterations(2)
            .omega_sa_max_step(0.1)
            .compute_map(false);
        let mut state = SaemState::from_problem(fixed_covariate_iiv_problem(), &config).unwrap();

        for subject_chains in &mut state.etas {
            subject_chains[0][0] = 2.0;
            subject_chains[1][0] = -2.0;
        }
        state.cycle = 2;
        assert_eq!(
            state.initialization.schedule.phase(state.cycle),
            SaemPhase::Exploration
        );
        assert_eq!(
            state
                .initialization
                .schedule
                .stochastic_approximation_step(state.cycle),
            1.0
        );
        state.m_step().unwrap();

        assert!((state.iiv_second_moment[[0, 0]] - 4.0).abs() <= 1e-12);
        assert!((state.omega[[0, 0]] - 1.3).abs() <= 1e-12);

        for subject_chains in &mut state.etas {
            subject_chains[0][0] = 4.0;
            subject_chains[1][0] = -4.0;
        }
        state.cycle = 4;
        assert_eq!(
            state.initialization.schedule.phase(state.cycle),
            SaemPhase::Smoothing
        );
        assert_eq!(
            state
                .initialization
                .schedule
                .stochastic_approximation_step(state.cycle),
            0.5
        );
        state.m_step().unwrap();

        // The common raw history moves from variance 4 toward 16 with gain 0.5,
        // giving 10. Omega installs that coherent target directly. Applying the
        // smoothing gain a second time would instead leave Omega below 10.
        assert!((state.iiv_second_moment[[0, 0]] - 10.0).abs() <= 1e-12);
        assert!((state.omega[[0, 0]] - 10.0).abs() <= 1e-12);
    }

    #[test]
    fn schedule_splits_burn_in_exploration_and_smoothing() {
        let config = SaemConfig::new()
            .k1_iterations(300)
            .k2_iterations(100)
            .burn_in(5);
        let schedule = SaemSchedule::from_config(&config);

        assert_eq!(schedule.pure_burn_in, 5);
        assert_eq!(schedule.exploration_iterations, 295);
        assert_eq!(schedule.smoothing_iterations, 100);
        assert_eq!(schedule.total_iterations, 400);
        assert_eq!(schedule.variance_floor_iterations, 150);
        assert_eq!(schedule.minimum_residual_sigma, 1e-6);
        assert_eq!(schedule.stochastic_approximation_step(1), 0.0);
        assert_eq!(schedule.stochastic_approximation_step(6), 1.0);
        assert_eq!(schedule.stochastic_approximation_step(301), 1.0);
        assert_eq!(schedule.stochastic_approximation_step(302), 0.5);
        assert_eq!(schedule.covariance_step(1), 0.1);
        assert_eq!(schedule.covariance_step(6), 0.1);
        assert_eq!(schedule.covariance_step(300), 0.1);
        assert_eq!(schedule.covariance_step(301), 1.0);
        assert_eq!(schedule.covariance_step(302), 0.5);
        assert!(!schedule.covariance_update_active(5));
        assert!(schedule.covariance_update_active(6));
        assert_eq!(schedule.guarded_residual_sigma(1, 1.0, 0.1), 0.97);
        assert_eq!(schedule.guarded_residual_sigma(151, 1.0, 0.1), 0.1);
        assert_eq!(schedule.guarded_residual_sigma(151, 1.0, 0.0), 1e-6);
    }

    #[test]
    fn averaged_schedule_uses_alpha_only_during_smoothing() {
        let schedule = SaemSchedule::from_config(
            &SaemConfig::new()
                .k1_iterations(3)
                .burn_in(1)
                .k2_iterations(4)
                .averaged_iterates(0.75),
        );
        assert_eq!(schedule.stochastic_approximation_step(1), 0.0);
        assert_eq!(schedule.stochastic_approximation_step(2), 1.0);
        assert_eq!(schedule.stochastic_approximation_step(3), 1.0);
        assert_eq!(schedule.stochastic_approximation_step(4), 1.0);
        assert_eq!(
            schedule.stochastic_approximation_step(5),
            2.0_f64.powf(-0.75)
        );
        assert_eq!(
            schedule.stochastic_approximation_step(7),
            4.0_f64.powf(-0.75)
        );
    }

    #[test]
    fn averaged_result_uses_only_completed_smoothing_iterates() {
        let config = SaemConfig::new()
            .k1_iterations(2)
            .burn_in(1)
            .k2_iterations(3)
            .averaged_iterates(0.75)
            .compute_map(false)
            .seed(9981);
        let result = problem().fit_with(config).unwrap();
        let metadata = result.estimator_metadata();
        assert!(metadata.average_applied);
        assert_eq!(metadata.averaging_start_cycle, Some(3));
        assert_eq!(metadata.averaged_iterations, 3);
        assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));

        let smoothing = &result.cycle_diagnostics()[2..];
        for parameter_index in 0..result.population_parameters().len() {
            if !result.estimated_parameters()[parameter_index] {
                continue;
            }
            let expected = smoothing
                .iter()
                .map(|cycle| {
                    population_phi(&cycle.population_parameters, result.parameter_scales()).unwrap()
                        [parameter_index]
                })
                .sum::<f64>()
                / smoothing.len() as f64;
            let installed =
                population_phi(result.population_parameters(), result.parameter_scales()).unwrap()
                    [parameter_index];
            assert!((installed - expected).abs() < 1e-12);
        }
        for row in 0..result.omega().nrows() {
            for col in 0..result.omega().ncols() {
                let expected = smoothing
                    .iter()
                    .map(|cycle| cycle.omega[[row, col]])
                    .sum::<f64>()
                    / smoothing.len() as f64;
                assert!((result.omega()[[row, col]] - expected).abs() < 1e-12);
            }
        }
        cholesky_lower(result.omega()).unwrap();
    }

    #[test]
    fn averaged_iov_installation_is_canonical_and_preserves_latent_coordinates() {
        let config = SaemConfig::new()
            .n_chains(2)
            .mcmc_iterations(2)
            .k1_iterations(1)
            .k2_iterations(3)
            .burn_in(0)
            .averaged_iterates(0.75)
            .compute_map(false)
            .seed(71_004);
        let mut state = SaemState::from_problem(configured_iov_problem(), &config)
            .expect("averaged IOV state should initialize");
        while matches!(state.status, Status::Continue) {
            state.step().expect("averaged IOV cycle should complete");
        }
        let cycle_records = state.cycle_diagnostics.clone();
        let smoothing = &cycle_records[1..];
        let terminal_phi = population_phi(
            &state.population_parameters,
            &state.initialization.parameter_scales,
        )
        .expect("terminal population phi should be valid");
        let terminal_absolute_phi = state
            .etas
            .iter()
            .map(|chains| {
                chains
                    .iter()
                    .map(|eta| {
                        state
                            .initialization
                            .random_effect_indices
                            .iter()
                            .enumerate()
                            .map(|(eta_index, parameter_index)| {
                                terminal_phi[*parameter_index] + eta[eta_index]
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let terminal_kappas = state.kappas.clone();
        let average = state
            .iterate_average
            .clone()
            .expect("completed smoothing average");

        let metadata = state
            .install_iterate_average()
            .expect("averaged IOV state should install");
        assert!(metadata.average_applied);
        assert_eq!(metadata.averaging_start_cycle, Some(2));
        assert_eq!(metadata.averaged_iterations, 3);
        assert_eq!(state.cycle_diagnostics, cycle_records);
        assert_eq!(state.kappas, terminal_kappas);

        let installed_phi = population_phi(
            &state.population_parameters,
            &state.initialization.parameter_scales,
        )
        .expect("installed population phi should be valid");
        assert_eq!(installed_phi, average.population_phi);
        for (subject_index, chains) in state.etas.iter().enumerate() {
            for (chain_index, eta) in chains.iter().enumerate() {
                for (eta_index, parameter_index) in state
                    .initialization
                    .random_effect_indices
                    .iter()
                    .copied()
                    .enumerate()
                {
                    assert!(
                        (installed_phi[parameter_index] + eta[eta_index]
                            - terminal_absolute_phi[subject_index][chain_index][eta_index])
                            .abs()
                            < 1e-14
                    );
                }
            }
        }

        let omega_iov = state.omega_iov.as_ref().expect("installed Omega_IOV");
        let iov_specification = state
            .initialization
            .omega_iov
            .as_ref()
            .expect("IOV specification");
        assert_eq!(omega_iov, &average.omega_iov.expect("averaged Omega_IOV"));
        for row in 0..omega_iov.nrows() {
            for col in 0..omega_iov.ncols() {
                let expected = if iov_specification.estimated_mask()[[row, col]] {
                    smoothing
                        .iter()
                        .map(|cycle| {
                            cycle
                                .omega_iov
                                .as_ref()
                                .expect("smoothing cycle should retain Omega_IOV")[[row, col]]
                        })
                        .sum::<f64>()
                        / smoothing.len() as f64
                } else {
                    iov_specification.initial()[[row, col]]
                };
                assert!((omega_iov[[row, col]] - expected).abs() < 1e-12);
            }
        }

        let n_chains = state.initialization.n_chains as f64;
        let mut direct_likelihoods = vec![0.0; state.initialization.subject_ids.len()];
        let mut direct_eta_priors = vec![0.0; state.initialization.subject_ids.len()];
        let mut direct_kappa_priors = vec![0.0; state.initialization.subject_ids.len()];
        for subject_index in 0..state.initialization.subject_ids.len() {
            for chain_index in 0..state.initialization.n_chains {
                let score = state
                    .score_subject_latents(
                        subject_index,
                        &state.etas[subject_index][chain_index],
                        &state.kappas[subject_index][chain_index],
                    )
                    .expect("installed latent score should be directly calculable");
                direct_likelihoods[subject_index] += score.log_likelihood / n_chains;
                direct_eta_priors[subject_index] += score.eta_log_prior / n_chains;
                direct_kappa_priors[subject_index] += score.kappa_log_prior / n_chains;
            }
        }
        assert_eq!(state.subject_log_likelihoods, direct_likelihoods);
        assert_eq!(state.subject_log_priors, direct_eta_priors);
        assert_eq!(state.subject_kappa_log_priors, direct_kappa_priors);
        assert_eq!(
            state.negative_log_likelihood,
            negative_log_likelihood(&direct_likelihoods)
        );
    }

    #[test]
    fn frozen_markov_diagnostic_is_repeatable_and_canonical_result_is_unchanged() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};

        let base = SaemConfig::new()
            .k1_iterations(100)
            .k2_iterations(50)
            .burn_in(1)
            .n_chains(2)
            .eta_block_iterations(1)
            .compute_map(true)
            .seed(91)
            .averaged_iterates(0.75);
        let diagnostic_config = MarkovSimulationVarianceConfig::new(
            700,
            2,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            2,
            64 * 1024,
        );
        let disabled = markov_iov_problem().fit_with(base.clone()).unwrap();
        let enabled = markov_iov_problem()
            .fit_with(base.clone().markov_simulation_variance(diagnostic_config))
            .unwrap();
        let repeated = markov_iov_problem()
            .fit_with(base.clone().markov_simulation_variance(diagnostic_config))
            .unwrap();
        let changed_seed = markov_iov_problem()
            .fit_with(
                base.clone()
                    .markov_simulation_variance(MarkovSimulationVarianceConfig::new(
                        701,
                        2,
                        12,
                        6,
                        LugsailConfig::over_lugsail_bartlett(),
                        2,
                        64 * 1024,
                    )),
            )
            .unwrap();

        assert_eq!(
            enabled.markov_simulation_variance(),
            repeated.markov_simulation_variance()
        );
        assert_ne!(
            enabled.markov_simulation_variance(),
            changed_seed.markov_simulation_variance()
        );
        assert_ne!(
            enabled.markov_simulation_variance().status,
            MarkovSimulationVarianceStatus::Disabled
        );
        assert!(!enabled.markov_simulation_variance().chains.is_empty());
        // One subject, one eta block, one component eta, and two occasion-kappa
        // blocks are attempted in that exact compound-kernel order per retained
        // transition. Warmup attempts are absent from the exported count.
        assert!(enabled
            .markov_simulation_variance()
            .chains
            .iter()
            .all(|chain| chain.proposals == 12 * (1 + 1 + 2)));
        assert_eq!(
            enabled.population_parameters(),
            disabled.population_parameters()
        );
        assert_eq!(enabled.omega(), disabled.omega());
        assert_eq!(enabled.omega_iov(), disabled.omega_iov());
        assert_eq!(
            enabled.residual_error_estimates(),
            disabled.residual_error_estimates()
        );
        assert_eq!(enabled.eta_chain_means(), disabled.eta_chain_means());
        assert_eq!(enabled.kappa_chain_means(), disabled.kappa_chain_means());
        assert!(!enabled.conditional_modes().is_empty());
        assert_eq!(enabled.conditional_modes(), disabled.conditional_modes());
        assert_eq!(
            enabled.information_diagnostics(),
            disabled.information_diagnostics()
        );
        assert_eq!(enabled.cycle_diagnostics(), disabled.cycle_diagnostics());
        assert_eq!(enabled.warnings(), disabled.warnings());
        assert_eq!(enabled.conditional_n2ll(), disabled.conditional_n2ll());
        assert_eq!(enabled.termination_reason(), disabled.termination_reason());
        assert_eq!(
            enabled.population_parameters(),
            changed_seed.population_parameters()
        );
        assert_eq!(enabled.omega(), changed_seed.omega());
        assert_eq!(
            enabled.residual_error_estimates(),
            changed_seed.residual_error_estimates()
        );
        assert_eq!(enabled.eta_chain_means(), changed_seed.eta_chain_means());
        assert_eq!(
            enabled.cycle_diagnostics(),
            changed_seed.cycle_diagnostics()
        );
        assert_eq!(enabled.warnings(), changed_seed.warnings());
        assert_eq!(enabled.conditional_n2ll(), changed_seed.conditional_n2ll());
        let enabled_predictions = enabled.population_predictions(0.0, 0.0).unwrap();
        let disabled_predictions = disabled.population_predictions(0.0, 0.0).unwrap();
        assert_eq!(enabled_predictions.len(), disabled_predictions.len());
        for (actual, expected) in enabled_predictions.iter().zip(&disabled_predictions) {
            assert_prediction_points_equal(actual, expected);
        }
        let enabled_conditional = enabled.conditional_predictions(0.0, 0.0).unwrap();
        let disabled_conditional = disabled.conditional_predictions(0.0, 0.0).unwrap();
        assert_eq!(enabled_conditional.len(), disabled_conditional.len());
        for (actual, expected) in enabled_conditional.iter().zip(&disabled_conditional) {
            assert_prediction_points_equal(actual, expected);
        }
    }

    #[test]
    fn rank_diagnostics_computed_for_multiple_chains_and_iov() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        use crate::results::{DiagnosticTraceCoordinate, RankDiagnosticStatus};

        let base = SaemConfig::new()
            .k1_iterations(30)
            .k2_iterations(20)
            .burn_in(1)
            .n_chains(2)
            .eta_block_iterations(1)
            .compute_map(false)
            .seed(91)
            .averaged_iterates(0.75);
        let diag = MarkovSimulationVarianceConfig::new(
            700,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            2,
            1024 * 1024,
        );
        let result = markov_iov_problem()
            .fit_with(base.clone().markov_simulation_variance(diag))
            .unwrap();
        let rank = &result.markov_simulation_variance().rank_diagnostics;
        assert_eq!(rank.diagnostic_chains, 2);
        assert_eq!(rank.draws_per_chain, 12);
        assert_eq!(rank.original_chains, 2);
        assert_eq!(rank.status, RankDiagnosticStatus::Available);
        assert!(!rank.traces.is_empty());
        // First trace is a score coordinate.
        assert!(matches!(
            rank.traces[0].trace,
            DiagnosticTraceCoordinate::Score { .. }
        ));
        let score_count = result.information_diagnostics().coordinates.len();
        let eta_count = result
            .eta_chain_means()
            .iter()
            .map(|estimate| estimate.values.len())
            .sum::<usize>();
        let kappa_count = result
            .kappa_chain_means()
            .iter()
            .map(|estimate| estimate.values.len())
            .sum::<usize>();
        assert_eq!(rank.traces.len(), score_count + eta_count + kappa_count);
        for (trace, coordinate) in rank
            .traces
            .iter()
            .take(score_count)
            .zip(&result.information_diagnostics().coordinates)
        {
            assert!(matches!(
                &trace.trace,
                DiagnosticTraceCoordinate::Score { index, .. } if *index == coordinate.index
            ));
        }
        assert!(rank
            .traces
            .iter()
            .skip(score_count)
            .take(eta_count)
            .all(|trace| matches!(trace.trace, DiagnosticTraceCoordinate::Eta { .. })));
        assert!(rank
            .traces
            .iter()
            .skip(score_count + eta_count)
            .all(|trace| matches!(trace.trace, DiagnosticTraceCoordinate::Kappa { .. })));
        assert!(rank.diagnostic_mean_lrv.is_some());
        assert!(rank.operational_lrv.is_some());

        // Repeatability: same seed produces identical rank diagnostics.
        let repeated = markov_iov_problem()
            .fit_with(base.clone().markov_simulation_variance(diag))
            .unwrap();
        assert_eq!(
            result.markov_simulation_variance().rank_diagnostics,
            repeated.markov_simulation_variance().rank_diagnostics
        );

        // Canonical result is unchanged by rank diagnostic presence.
        let disabled = markov_iov_problem().fit_with(base).unwrap();
        assert_eq!(
            result.population_parameters(),
            disabled.population_parameters()
        );
        assert_eq!(result.omega(), disabled.omega());
        assert_eq!(result.conditional_n2ll(), disabled.conditional_n2ll());
        assert_eq!(result.termination_reason(), disabled.termination_reason());
        assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
    }

    #[test]
    fn score_failure_does_not_discard_valid_eta_rank_diagnostics() {
        use crate::results::{
            DiagnosticTraceCoordinate, InformationCoordinateKind, RankDiagnosticStatus,
        };

        let state = SaemState::from_problem(problem(), &SaemConfig::default()).unwrap();
        let traces = vec![
            vec![vec![f64::NAN; 8], vec![f64::NAN; 8]],
            vec![
                vec![1.0, 4.0, 2.0, 3.0, 2.0, 4.0, 1.0, 3.0],
                vec![2.1, 3.1, 1.1, 4.1, 3.1, 1.1, 4.1, 2.1],
            ],
        ];
        let coordinates = vec![
            DiagnosticTraceCoordinate::Score {
                index: 0,
                name: "score".into(),
                kind: InformationCoordinateKind::Population { parameter_index: 0 },
            },
            DiagnosticTraceCoordinate::Eta {
                subject: "1".into(),
                effect_index: 0,
                effect_name: "CL".into(),
            },
        ];
        let diagnostics = state.rank_diagnostics_from_traces(2, &traces, &coordinates);
        assert_eq!(
            diagnostics[0].rank_rhat_status,
            RankDiagnosticStatus::ScoreUnavailable
        );
        assert!(diagnostics[0].rank_rhat.is_none());
        assert_eq!(
            diagnostics[1].rank_rhat_status,
            RankDiagnosticStatus::Available
        );
        assert!(diagnostics[1].rank_rhat.is_some());
    }

    #[test]
    fn multimodal_latent_trace_is_detected_while_mixed_score_trace_passes() {
        use crate::results::{
            DiagnosticTraceCoordinate, InformationCoordinateKind, RankDiagnosticStatus,
        };

        let state = SaemState::from_problem(problem(), &SaemConfig::default()).unwrap();
        let traces = vec![
            vec![
                vec![1.0, 4.0, 2.0, 3.0, 2.0, 4.0, 1.0, 3.0],
                vec![2.1, 3.1, 1.1, 4.1, 3.1, 1.1, 4.1, 2.1],
            ],
            vec![
                vec![-10.0, -9.0, -11.0, -8.0, -9.5, -8.5, -10.5, -7.5],
                vec![8.0, 11.0, 9.0, 10.0, 8.5, 10.5, 7.5, 9.5],
            ],
        ];
        let coordinates = vec![
            DiagnosticTraceCoordinate::Score {
                index: 0,
                name: "score".into(),
                kind: InformationCoordinateKind::Population { parameter_index: 0 },
            },
            DiagnosticTraceCoordinate::Eta {
                subject: "1".into(),
                effect_index: 0,
                effect_name: "CL".into(),
            },
        ];
        let diagnostics = state.rank_diagnostics_from_traces(2, &traces, &coordinates);
        assert_eq!(
            diagnostics[0].rank_rhat_status,
            RankDiagnosticStatus::Available
        );
        assert!(diagnostics[0].rank_rhat.is_some_and(|rhat| rhat < 1.1));
        assert_eq!(
            diagnostics[1].rank_rhat_status,
            RankDiagnosticStatus::Available
        );
        assert!(diagnostics[1].rank_rhat.is_some_and(|rhat| rhat > 1.1));
    }

    #[test]
    fn rank_coordinate_retains_valid_rhats_when_bulk_ess_is_unavailable() {
        use crate::results::{DiagnosticTraceCoordinate, RankDiagnosticStatus};

        let state = SaemState::from_problem(problem(), &SaemConfig::default()).unwrap();
        let traces = vec![vec![vec![1.0, 2.0, 4.0, 3.0], vec![1.5, 2.5, 4.5, 3.5]]];
        let coordinates = vec![DiagnosticTraceCoordinate::Eta {
            subject: "1".into(),
            effect_index: 0,
            effect_name: "CL".into(),
        }];
        let diagnostics = state.rank_diagnostics_from_traces(2, &traces, &coordinates);
        let diagnostic = &diagnostics[0];
        assert!(diagnostic.rank_rhat.is_some());
        assert_eq!(diagnostic.rank_rhat_status, RankDiagnosticStatus::Available);
        assert!(diagnostic.folded_rhat.is_some());
        assert_eq!(
            diagnostic.folded_rhat_status,
            RankDiagnosticStatus::Available
        );
        assert!(diagnostic.bulk_ess.is_none());
        assert!(diagnostic.tau.is_none());
        assert_eq!(
            diagnostic.bulk_ess_status,
            RankDiagnosticStatus::TooFewDraws
        );
        assert_eq!(diagnostic.status, RankDiagnosticStatus::PartialAvailability);
    }

    #[test]
    fn derived_max_rhat_requires_both_rank_and_folded_components() {
        use crate::results::{DiagnosticTraceCoordinate, RankDiagnosticStatus};

        let state = SaemState::from_problem(problem(), &SaemConfig::default()).unwrap();
        let traces = vec![vec![
            vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            vec![2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0],
        ]];
        let coordinates = vec![DiagnosticTraceCoordinate::Eta {
            subject: "1".into(),
            effect_index: 0,
            effect_name: "CL".into(),
        }];

        let diagnostics = state.rank_diagnostics_from_traces(2, &traces, &coordinates);
        let diagnostic = &diagnostics[0];
        assert!(diagnostic.rank_rhat.is_some());
        assert_eq!(diagnostic.rank_rhat_status, RankDiagnosticStatus::Available);
        assert!(diagnostic.folded_rhat.is_none());
        assert_eq!(
            diagnostic.folded_rhat_status,
            RankDiagnosticStatus::ConstantDraws
        );
        assert!(diagnostic.max_rhat.is_none());
        assert_eq!(
            diagnostic.max_rhat_status,
            RankDiagnosticStatus::ConstantDraws
        );
        assert_eq!(worst_valid_max_rhat(&diagnostics), None);
    }

    #[test]
    fn rank_diagnostics_available_when_markov_config_enabled() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        use crate::results::RankDiagnosticStatus;

        let base = SaemConfig::new()
            .k1_iterations(100)
            .k2_iterations(50)
            .burn_in(1)
            .n_chains(2)
            .eta_block_iterations(1)
            .compute_map(false)
            .seed(77)
            .averaged_iterates(0.75);
        let diag = MarkovSimulationVarianceConfig::new(
            42,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            2,
            1024 * 1024,
        );
        let result = markov_iov_problem()
            .fit_with(base.clone().markov_simulation_variance(diag))
            .unwrap();
        let rank = &result.markov_simulation_variance().rank_diagnostics;
        // Rank diagnostics object is always present when markov config enabled;
        // status reflects whether data supported valid computation.
        assert_eq!(rank.diagnostic_chains, 2);
        assert_eq!(rank.original_chains, 2);
        assert!(!matches!(rank.status, RankDiagnosticStatus::Disabled));
    }

    #[test]
    fn one_diagnostic_chain_retains_markov_lrv_but_rank_is_unavailable() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        use crate::results::RankDiagnosticStatus;

        let config = SaemConfig::new()
            .k1_iterations(30)
            .k2_iterations(20)
            .burn_in(1)
            .n_chains(2)
            .eta_block_iterations(1)
            .compute_map(false)
            .seed(93)
            .averaged_iterates(0.75)
            .markov_simulation_variance(MarkovSimulationVarianceConfig::new(
                702,
                0,
                12,
                6,
                LugsailConfig::over_lugsail_bartlett(),
                1,
                1024 * 1024,
            ));
        let result = markov_iov_problem().fit_with(config).unwrap();
        let markov = result.markov_simulation_variance();
        assert_eq!(
            markov.rank_diagnostics.status,
            RankDiagnosticStatus::TooFewChains
        );
        assert_eq!(markov.chains.len(), 1);
        assert!(!markov.lambda.is_empty());
        assert!(markov.rank_diagnostics.operational_lrv.is_some());
        assert!(markov.rank_diagnostics.traces.iter().all(|trace| {
            trace.status == RankDiagnosticStatus::TooFewChains
                && trace.rank_rhat.is_none()
                && trace.bulk_ess.is_none()
        }));
        assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
        assert!(!result.converged());
    }

    #[test]
    fn rank_diagnostics_trace_byte_cap_exceeded_is_reported() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        use crate::results::RankDiagnosticStatus;

        let base = SaemConfig::new()
            .k1_iterations(100)
            .k2_iterations(50)
            .burn_in(1)
            .n_chains(2)
            .eta_block_iterations(1)
            .compute_map(false)
            .seed(91)
            .averaged_iterates(0.75);
        let tiny_cap = MarkovSimulationVarianceConfig::new(
            700,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            2,
            1, // 1 byte cap — guaranteed to be exceeded
        );
        let result = markov_iov_problem()
            .fit_with(base.clone().markov_simulation_variance(tiny_cap))
            .unwrap();
        let rank = &result.markov_simulation_variance().rank_diagnostics;
        assert_eq!(rank.status, RankDiagnosticStatus::TraceByteCapExceeded);
        assert!(rank.traces.is_empty());
        assert!(rank.diagnostic_mean_lrv.is_none());
        assert!(rank.operational_lrv.is_none());
        assert_eq!(rank.max_trace_bytes, 1);
        assert!(rank.accounted_peak_trace_bytes_required > rank.max_trace_bytes);
        assert_eq!(rank.accounted_peak_trace_bytes_used, 0);
        let markov = result.markov_simulation_variance();
        assert!(matches!(
            markov.status,
            MarkovSimulationVarianceStatus::InvalidConfiguration(_)
        ));
        assert_eq!(markov.lambda_status, markov.status);
        assert_eq!(markov.xi_status, markov.status);
        assert_eq!(markov.simulation_covariance_status, markov.status);
        assert!(markov.chains.is_empty());
        // Canonical result is unchanged.
        assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));

        let generous = MarkovSimulationVarianceConfig::new(
            700,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            2,
            1024 * 1024,
        );
        let measured = markov_iov_problem()
            .fit_with(base.clone().markov_simulation_variance(generous))
            .unwrap();
        let measured_rank = &measured.markov_simulation_variance().rank_diagnostics;
        let trace_count = measured_rank.traces.len();
        let score_width = measured.markov_simulation_variance().coordinates.len();
        let vec_header = std::mem::size_of::<Vec<f64>>();
        let persistent_bytes = 2 * 12 * trace_count * std::mem::size_of::<f64>()
            + trace_count * 2 * vec_header
            + trace_count * vec_header;
        let score_transient_bytes = score_width * 12 * std::mem::size_of::<f64>() + 12 * vec_header;
        let rank_transient_bytes = 2 * 12 * 8 * std::mem::size_of::<f64>() + 2 * 16 * vec_header;
        let expected_bytes = persistent_bytes + score_transient_bytes.max(rank_transient_bytes);
        assert_eq!(
            measured_rank.accounted_peak_trace_bytes_required,
            expected_bytes
        );
        assert_eq!(
            measured_rank.accounted_peak_trace_bytes_used,
            expected_bytes
        );

        let exact_cap = MarkovSimulationVarianceConfig::new(
            700,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            2,
            expected_bytes,
        );
        let exact = markov_iov_problem()
            .fit_with(base.clone().markov_simulation_variance(exact_cap))
            .unwrap();
        assert_eq!(
            exact
                .markov_simulation_variance()
                .rank_diagnostics
                .accounted_peak_trace_bytes_used,
            expected_bytes
        );
        assert!(!exact
            .markov_simulation_variance()
            .rank_diagnostics
            .traces
            .is_empty());

        let under_cap = MarkovSimulationVarianceConfig::new(
            700,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            2,
            expected_bytes - 1,
        );
        let rejected = markov_iov_problem()
            .fit_with(base.clone().markov_simulation_variance(under_cap))
            .unwrap();
        assert_eq!(
            rejected
                .markov_simulation_variance()
                .rank_diagnostics
                .status,
            RankDiagnosticStatus::TraceByteCapExceeded
        );
        assert_eq!(
            rejected
                .markov_simulation_variance()
                .rank_diagnostics
                .accounted_peak_trace_bytes_used,
            0
        );

        let overflow = MarkovSimulationVarianceConfig::new(
            700,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            usize::MAX / 2 + 1,
            usize::MAX,
        );
        let overflowed = markov_iov_problem()
            .fit_with(base.markov_simulation_variance(overflow))
            .unwrap();
        let overflowed = overflowed.markov_simulation_variance();
        assert_eq!(
            overflowed.rank_diagnostics.status,
            RankDiagnosticStatus::TraceMemoryAccountingOverflow
        );
        assert_eq!(
            overflowed.status,
            MarkovSimulationVarianceStatus::TraceMemoryAccountingOverflow
        );
        assert_eq!(
            overflowed
                .rank_diagnostics
                .accounted_peak_trace_bytes_required,
            0
        );
        assert_eq!(
            overflowed.rank_diagnostics.accounted_peak_trace_bytes_used,
            0
        );
        assert!(overflowed.chains.is_empty());
    }

    #[test]
    fn operational_and_frozen_iov_transitions_preserve_compound_kernel_order() {
        let seed = 0x5eed;
        let mut operational = SaemState::from_problem(
            markov_iov_problem(),
            &SaemConfig::new()
                .n_chains(2)
                .mcmc_iterations(1)
                .eta_block_iterations(1)
                .adapt_interval(50)
                .seed(seed),
        )
        .unwrap();
        let initial_eta_scales = operational.proposal_step_sizes.clone();
        let initial_eta_block_scales = operational.eta_block_step_sizes.clone();
        let initial_kappa_scales = operational.kappa_proposal_step_sizes.clone();
        let mut frozen = FrozenDiagnosticState {
            etas: operational.etas.clone(),
            kappas: operational.kappas.clone(),
        };
        let mut frozen_rng = StdRng::seed_from_u64(seed);
        let mut frozen_counts = vec![(0, 0, 0); operational.initialization.n_chains];

        // This single compound transition is order-sensitive: eta blocks consume
        // the stream first, followed by component etas and then occasion kappas.
        operational
            .frozen_diagnostic_transition(&mut frozen, &mut frozen_rng, &mut frozen_counts, None)
            .unwrap();
        operational.e_step().unwrap();

        assert_eq!(operational.etas, frozen.etas);
        assert_eq!(operational.kappas, frozen.kappas);
        assert_eq!(operational.proposal_step_sizes, initial_eta_scales);
        assert_eq!(operational.eta_block_step_sizes, initial_eta_block_scales);
        assert_eq!(operational.kappa_proposal_step_sizes, initial_kappa_scales);

        let diagnostics = operational.cycle_diagnostics.last().unwrap();
        let frozen_proposals = frozen_counts.iter().map(|count| count.0).sum::<usize>();
        let frozen_accepts = frozen_counts.iter().map(|count| count.1).sum::<usize>();
        let frozen_changes = frozen_counts.iter().map(|count| count.2).sum::<usize>();
        assert_eq!(diagnostics.eta_block_proposals, 2);
        assert_eq!(diagnostics.eta_proposals, 4);
        assert_eq!(diagnostics.kappa_proposals, 4);
        assert_eq!(frozen_proposals, 8);
        assert_eq!(
            frozen_accepts,
            diagnostics.eta_accepted + diagnostics.kappa_accepted
        );
        assert_eq!(frozen_changes, frozen_accepts);
        assert_eq!(
            diagnostics.eta_rejected + diagnostics.kappa_rejected,
            frozen_proposals - frozen_accepts
        );
        assert_eq!(diagnostics.eta_non_finite, 0);
        assert_eq!(diagnostics.kappa_non_finite, 0);

        let operational_continuation = operational.rng.random::<u64>();
        let frozen_continuation = frozen_rng.random::<u64>();
        assert_eq!(operational_continuation, frozen_continuation);
    }

    #[test]
    fn warmup_movement_cannot_satisfy_retained_movement_accounting() {
        let mut counts = [(12, 7, 4), (8, 1, 1)];
        begin_retained_transition_accounting(&mut counts);
        // Retained proposals that are accepted without an actual state change
        // still leave the chain eligible for the exact stuck guard.
        counts[0].0 += 3;
        counts[0].1 += 3;
        assert_eq!(counts, [(3, 3, 0), (0, 0, 0)]);
        let stuck: Vec<_> = counts
            .iter()
            .enumerate()
            .filter_map(|(chain, count)| (count.2 == 0).then_some(chain))
            .collect();
        assert_eq!(stuck, [0, 1]);
    }

    #[test]
    fn no_latent_state_reports_exact_zero_markov_variance() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};

        let result = fixed_no_iiv_problem()
            .fit_with(
                SaemConfig::new()
                    .k1_iterations(30)
                    .k2_iterations(20)
                    .burn_in(1)
                    .compute_map(false)
                    .averaged_iterates(0.75)
                    .markov_simulation_variance(MarkovSimulationVarianceConfig::new(
                        4,
                        100,
                        12,
                        6,
                        LugsailConfig::over_lugsail_bartlett(),
                        2,
                        1024,
                    )),
            )
            .unwrap();
        let diagnostic = result.markov_simulation_variance();
        assert_eq!(
            diagnostic.status,
            MarkovSimulationVarianceStatus::ExactZeroNoLatentState
        );
        assert!(diagnostic.chains.is_empty());
        assert_eq!(
            diagnostic.rank_diagnostics.status,
            RankDiagnosticStatus::NoLatent
        );
        assert!(diagnostic.rank_diagnostics.traces.is_empty());
        assert!(diagnostic
            .lambda
            .iter()
            .flatten()
            .all(|value| *value == 0.0));
        assert!(diagnostic.xi.iter().flatten().all(|value| *value == 0.0));
        assert!(diagnostic
            .simulation_covariance
            .iter()
            .flatten()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn explicit_terminal_policy_preserves_default_trajectory() {
        let base = SaemConfig::new()
            .k1_iterations(2)
            .burn_in(1)
            .k2_iterations(2)
            .compute_map(false)
            .seed(7788);
        let default = problem().fit_with(base.clone()).unwrap();
        let explicit = problem()
            .fit_with(base.estimator_policy(SaemEstimatorPolicy::TerminalIterate))
            .unwrap();
        assert_eq!(default.cycle_diagnostics(), explicit.cycle_diagnostics());
        assert_eq!(
            default.population_parameters(),
            explicit.population_parameters()
        );
        assert_eq!(default.omega(), explicit.omega());
        assert_eq!(default.conditional_n2ll(), explicit.conditional_n2ll());
        assert_eq!(default.termination_reason(), Some(&StopReason::MaxCycles));
        assert_eq!(explicit.termination_reason(), Some(&StopReason::MaxCycles));
    }

    fn residual_phase_schedule() -> SaemSchedule {
        let mut schedule = SaemSchedule::from_config(
            &SaemConfig::new()
                .burn_in(0)
                .k1_iterations(4)
                .k2_iterations(3),
        );
        schedule.variance_floor_iterations = 1;
        schedule
    }

    #[test]
    fn combined_residual_component_anneals_during_configured_period() {
        let schedule = residual_phase_schedule();
        let applied = applied_combined_residual_component(&schedule, 1, 1.0, 0.1, true);
        assert_eq!(applied, schedule.annealing_alpha);
    }

    #[test]
    fn combined_residual_component_replaces_directly_in_remaining_exploration() {
        let schedule = residual_phase_schedule();
        assert_eq!(
            applied_combined_residual_component(&schedule, 2, 1.0, 0.1, true),
            0.1
        );
    }

    #[test]
    fn combined_residual_component_smooths_in_k2() {
        let schedule = residual_phase_schedule();
        assert_eq!(
            applied_combined_residual_component(&schedule, 6, 1.0, 0.2, true),
            0.6
        );
    }

    #[test]
    fn combined_residual_component_preserves_fixed_value() {
        let schedule = residual_phase_schedule();
        assert_eq!(
            applied_combined_residual_component(&schedule, 1, 1.0, 0.1, false),
            1.0
        );
        assert_eq!(
            applied_combined_residual_component(&schedule, 6, 1.0, 0.1, false),
            1.0
        );
    }

    #[test]
    fn burn_in_warms_covariance_statistics_without_updating_parameters() {
        let config = SaemConfig::new()
            .n_chains(1)
            .burn_in(2)
            .k1_iterations(4)
            .omega_sa_max_step(0.1);
        let mut state = SaemState::from_problem(problem(), &config).unwrap();
        for subject_chains in &mut state.etas {
            subject_chains[0].fill(2.0);
        }
        let initial_population = state.population_parameters.clone();
        let initial_omega = state.omega.clone();
        let initial_iiv_second_moment = state.iiv_second_moment.clone();
        let initial_phi_second_moment = state.sufficient_statistics.second_moment.clone();

        state.step().unwrap();

        assert_eq!(state.cycle, 1);
        assert_eq!(state.cycle_diagnostics[0].phase, SaemPhase::BurnIn);
        assert_eq!(state.population_parameters, initial_population);
        assert_eq!(state.omega, initial_omega);
        assert_ne!(state.iiv_second_moment, initial_iiv_second_moment);
        assert_ne!(
            state.sufficient_statistics.second_moment,
            initial_phi_second_moment
        );
    }

    #[test]
    fn chain_count_auto_scales_for_small_datasets() {
        assert_eq!(n_chains(&SaemConfig::default(), 2), 25);
        assert_eq!(n_chains(&SaemConfig::new().n_chains(3), 2), 3);
        assert_eq!(n_chains(&SaemConfig::default(), 100), 1);
    }

    #[test]
    fn result_retains_requested_config_and_separate_effective_chain_count() {
        let config = SaemConfig::new()
            .n_chains(1)
            .k1_iterations(1)
            .k2_iterations(0)
            .burn_in(1)
            .compute_map(false)
            .seed(9876);
        let serialized_config = serde_json::to_value(&config).unwrap();
        let state = SaemState::from_problem(problem(), &config).unwrap();

        let result = Box::new(state).into_result().unwrap();

        assert_eq!(result.config().n_chains, 1);
        assert_eq!(result.effective_n_chains(), 25);
        assert_eq!(
            serde_json::to_value(result.config()).unwrap(),
            serialized_config
        );
    }

    #[test]
    fn result_parameter_metadata_preserves_declaration_order() {
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(0)
            .burn_in(1)
            .compute_map(false);
        let state = SaemState::from_problem(ordered_metadata_problem(), &config).unwrap();

        let result = Box::new(state).into_result().unwrap();

        assert_eq!(result.parameter_names(), ["ke", "v"]);
        assert_eq!(
            result.parameter_scales(),
            [ParameterScale::Identity, ParameterScale::Log]
        );
        assert_eq!(result.estimated_parameters(), [true, false]);
        assert_eq!(result.random_effect_indices(), [0]);
        assert_eq!(result.random_effect_names(), ["ke"]);
        assert_eq!(result.iov_effect_indices(), [1]);
        assert_eq!(result.iov_effect_names(), ["v"]);
    }

    #[test]
    fn result_retains_exact_symmetric_iiv_covariance_masks() {
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(0)
            .burn_in(1)
            .compute_map(false);
        let configured =
            Box::new(SaemState::from_problem(configured_omega_problem(), &config).unwrap())
                .into_result()
                .unwrap();
        let correlated =
            Box::new(SaemState::from_problem(correlated_omega_problem(), &config).unwrap())
                .into_result()
                .unwrap();

        assert_eq!(configured.random_effect_names(), ["ke", "v"]);
        assert_eq!(
            configured.omega_structural_mask(),
            &ndarray::array![[true, false], [false, true]]
        );
        assert_eq!(
            configured.omega_estimated_mask(),
            &ndarray::array![[true, false], [false, false]]
        );
        assert_eq!(
            correlated.omega_structural_mask(),
            &ndarray::array![[true, true], [true, true]]
        );
        assert_eq!(
            correlated.omega_estimated_mask(),
            &ndarray::array![[true, true], [true, true]]
        );
    }

    #[test]
    fn result_retains_ordered_iov_masks_and_none_without_iov() {
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(0)
            .burn_in(1)
            .compute_map(false);
        let iov = Box::new(SaemState::from_problem(configured_iov_problem(), &config).unwrap())
            .into_result()
            .unwrap();
        let no_iov = Box::new(SaemState::from_problem(problem(), &config).unwrap())
            .into_result()
            .unwrap();

        assert_eq!(iov.iov_effect_indices(), [0, 1]);
        assert_eq!(iov.iov_effect_names(), ["ke", "v"]);
        assert_eq!(
            iov.omega_iov_structural_mask(),
            Some(&ndarray::array![[true, true], [true, true]])
        );
        assert_eq!(
            iov.omega_iov_estimated_mask(),
            Some(&ndarray::array![[true, false], [false, false]])
        );
        assert_eq!(no_iov.omega_iov_structural_mask(), None);
        assert_eq!(no_iov.omega_iov_estimated_mask(), None);
    }

    #[test]
    fn state_initializes_zero_eta_chains() {
        let state = SaemState::from_problem(problem(), &SaemConfig::default()).unwrap();

        assert_eq!(state.etas.len(), 2);
        assert_eq!(state.etas[0].len(), 25);
        assert_eq!(state.etas[0][0], vec![0.0, 0.0]);
        assert_eq!(state.etas[1][24], vec![0.0, 0.0]);
        assert_eq!(state.omega_diagonal(), Some(vec![1.0, 1.0]));
    }

    #[test]
    fn covariate_state_joint_gls_rebases_eta_and_builds_subject_omega() {
        let mut state = SaemState::from_problem(
            covariate_problem(),
            &SaemConfig::new().n_chains(2).compute_map(false),
        )
        .unwrap();
        let intercept = [0.2_f64.ln(), 10.0_f64.ln()];
        let beta = 0.35;
        let expected_phi = [-1.0, 0.0, 1.0]
            .into_iter()
            .map(|design| vec![intercept[0] + beta * design, intercept[1]])
            .collect::<Vec<_>>();
        let desired_omega = ndarray::array![[0.4, 0.1], [0.1, 0.3]];
        let mut second = desired_omega.clone();
        for mean in &expected_phi {
            for row in 0..2 {
                for column in 0..2 {
                    second[[row, column]] += mean[row] * mean[column] / 3.0;
                }
            }
        }
        let old_means = state.subject_mu_phi.clone().unwrap();
        for chains in &mut state.etas {
            for eta in chains {
                eta[0] = 0.1;
                eta[1] = -0.2;
            }
        }
        let absolute_before = old_means
            .iter()
            .map(|mean| vec![mean[0] + 0.1, mean[1] - 0.2])
            .collect::<Vec<_>>();
        state.covariate_statistics = Some(CovariateSufficientStatistics {
            expected_phi,
            global_second_moment: second,
        });

        let candidate = state
            .update_covariate_population_and_recenter_etas()
            .unwrap();
        let model = state.covariate_model.as_ref().unwrap();
        assert!((model.estimates()[0].estimate() - beta).abs() < 1e-10);
        assert!((candidate[[0, 0]] - desired_omega[[0, 0]]).abs() < 1e-10);
        assert!((candidate[[0, 1]] - desired_omega[[0, 1]]).abs() < 1e-10);
        for (subject, mean) in state.subject_mu_phi.as_ref().unwrap().iter().enumerate() {
            for coordinate in 0..2 {
                assert!(
                    (mean[coordinate] + state.etas[subject][0][coordinate]
                        - absolute_before[subject][coordinate])
                        .abs()
                        < 1e-10
                );
            }
        }
    }

    #[test]
    fn covariate_fit_executes_and_retains_subject_population_parameters() {
        let result = covariate_problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(2)
                    .mcmc_iterations(1)
                    .burn_in(1)
                    .k1_iterations(2)
                    .k2_iterations(2)
                    .averaged_iterates(0.75)
                    .compute_map(false),
            )
            .unwrap();
        assert!(result.estimator_metadata().average_applied);
        assert_eq!(result.covariate_estimates().unwrap().len(), 2);
        assert!(result.covariate_estimates().unwrap()[0].estimate() < 0.0);
        assert_eq!(
            result
                .covariate_subject_population_parameters()
                .unwrap()
                .unwrap()
                .len(),
            3
        );
        assert!(result.cycle_diagnostics().iter().all(|cycle| cycle
            .covariate_betas
            .as_ref()
            .is_some_and(|values| values.len() == 2)));
        let tables = result.tables(1.0, 0.0).unwrap();
        assert_eq!(tables.covariate_effects.len(), 2);
        assert_eq!(tables.subject_covariates.len(), 6);
        assert_eq!(tables.subject_population_parameters.len(), 6);

        let directory =
            std::env::temp_dir().join(format!("pmcore-schema7-covariate-{}", std::process::id()));
        result.write_outputs(&directory, 1.0, 0.0).unwrap();
        let record =
            crate::results::ParametricResultRecord::read_json(directory.join("result.json"))
                .unwrap();
        assert_eq!(record.schema_version, 9);
        assert_eq!(record.source_metadata.covariate_effects.len(), 2);
        let warm = record
            .warm_start_problem(one_compartment(), result.data().clone())
            .unwrap();
        let warm_estimates = warm
            .covariates()
            .unwrap()
            .estimates()
            .iter()
            .map(|estimate| estimate.estimate())
            .collect::<Vec<_>>();
        let result_estimates = result
            .covariate_estimates()
            .unwrap()
            .iter()
            .map(|estimate| estimate.estimate())
            .collect::<Vec<_>>();
        assert!(warm_estimates
            .iter()
            .zip(result_estimates)
            .all(|(warm, result)| (warm - result).abs() <= 2.0 * f64::EPSILON));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn fixed_covariate_without_iiv_executes_subject_specific_predictions() {
        let result = fixed_covariate_without_iiv_problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(1)
                    .mcmc_iterations(1)
                    .burn_in(0)
                    .k1_iterations(1)
                    .k2_iterations(0)
                    .compute_map(false),
            )
            .unwrap();
        assert!(result.random_effect_names().is_empty());
        let means = result
            .covariate_subject_population_parameters()
            .unwrap()
            .unwrap();
        assert!((means[0].psi()[0] - 0.2).abs() < 1e-12);
        assert!((means[1].psi()[0] - 0.2 * 0.2_f64.exp()).abs() < 1e-12);
        let predictions = result.population_predictions(0.0, 0.0).unwrap();
        assert_ne!(
            predictions[0].predictions()[0].prediction(),
            predictions[1].predictions()[0].prediction()
        );
    }

    #[test]
    fn explicit_iiv_mask_controls_eta_and_omega_dimensions() {
        let mut state =
            SaemState::from_problem(partial_iiv_problem(), &SaemConfig::new().n_chains(2)).unwrap();

        assert_eq!(state.initialization.random_effect_indices, vec![0]);
        assert_eq!(state.initialization.random_effect_names, vec!["ke"]);
        assert!(state
            .etas
            .iter()
            .flat_map(|subject_chains| subject_chains.iter())
            .all(|eta| eta.len() == 1));
        assert_eq!(state.omega.dim(), (1, 1));
        assert_eq!(state.proposal_step_sizes.len(), 1);

        state.etas[0][0][0] = 2.0_f64.ln();
        let individual = state.individual_parameters(0, 0);
        assert!((individual[0] - 0.4).abs() < 1e-12);
        assert!((individual[1] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn all_fixed_parameters_support_zero_dimensional_iiv() {
        let config = SaemConfig::new()
            .n_chains(1)
            .burn_in(1)
            .k1_iterations(1)
            .k2_iterations(1);
        let state = SaemState::from_problem(fixed_no_iiv_problem(), &config).expect(
            "fixed population plus estimated residual error should support zero-dimensional IIV",
        );
        assert!(state.initialization.random_effect_names.is_empty());
        assert!(state.omega.is_empty());
        assert!(state.iiv_second_moment.is_empty());
        assert!(state
            .etas
            .iter()
            .all(|chains| chains.iter().all(Vec::is_empty)));

        let result = fixed_no_iiv_problem().fit_with(config).unwrap();
        assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
        assert_eq!(result.iterations(), 2);
        assert!(result.objf().is_finite());
        assert!(result.conditional_modes().is_empty());
        assert_eq!(result.omega_structural_mask().dim(), (0, 0));
        assert_eq!(result.omega_estimated_mask().dim(), (0, 0));
        assert!(result.omega_structural_mask().is_empty());
        assert!(result.omega_estimated_mask().is_empty());
        assert_eq!(result.omega_iov_structural_mask(), None);
        assert_eq!(result.omega_iov_estimated_mask(), None);
        assert!(result
            .eta_chain_means()
            .iter()
            .all(|estimate| estimate.values.is_empty()));
        assert!(result.kappa_chain_means().is_empty());
    }

    #[test]
    fn iov_state_tracks_one_kappa_per_subject_occasion_and_chain() {
        let state = SaemState::from_problem(iov_problem(), &SaemConfig::new().n_chains(2)).unwrap();

        assert_eq!(state.initialization.iov_effect_names, vec!["ke"]);
        assert_eq!(state.omega_iov, Some(ndarray::array![[0.1]]));
        assert_eq!(state.kappas.len(), 1);
        assert_eq!(state.kappas[0].len(), 2);
        assert_eq!(state.kappas[0][0], vec![vec![0.0], vec![0.0]]);
    }

    #[test]
    fn uneven_occasion_counts_preserve_kappa_shapes_order_and_named_lookup() {
        let result = uneven_iov_problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(2)
                    .mcmc_iterations(1)
                    .burn_in(0)
                    .k1_iterations(2)
                    .k2_iterations(0)
                    .compute_map(false),
            )
            .unwrap();

        assert_eq!(result.kappa_chain_means().len(), 6);
        assert!(result.kappa_chain_mean("one", 0).is_some());
        assert!(result.kappa_chain_mean("one", 1).is_none());
        assert!(result.kappa_chain_mean("two", 0).is_some());
        assert!(result.kappa_chain_mean("two", 1).is_some());
        assert!(result.kappa_chain_mean("three", 0).is_some());
        assert!(result.kappa_chain_mean("three", 1).is_some());
        assert!(result.kappa_chain_mean("three", 2).is_some());
        assert!(result.eta_chain_mean("two").is_some());
        assert!(result.eta_chain_mean("missing").is_none());
        assert!(result.conditional_mode("two").is_none());
        assert!(result
            .cycle_diagnostics()
            .iter()
            .all(|cycle| cycle.kappa_proposals == 12));
    }

    #[test]
    fn iov_scores_per_occasion_kappa_prior_and_conditional_proposal() {
        let state = SaemState::from_problem(iov_problem(), &SaemConfig::new().n_chains(2)).unwrap();
        let score = state
            .score_subject_latents(0, &state.etas[0][0], &state.kappas[0][0])
            .unwrap();

        assert!((score.log_likelihood - state.subject_log_likelihoods[0]).abs() < 1e-12);
        assert!((score.kappa_log_prior - state.subject_kappa_log_priors[0]).abs() < 1e-12);
        assert!(score.kappa_log_prior.is_finite());
        assert_eq!(
            state
                .kappa_proposal_log_acceptance_ratio(0, 0, 0, &[0.0])
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn iov_controller_exposes_kappa_covariance_and_runs_conditional_mcmc() {
        let mut controller = iov_problem()
            .fit_controller(
                SaemConfig::new()
                    .n_chains(2)
                    .k1_iterations(2)
                    .k2_iterations(0)
                    .burn_in(2),
            )
            .unwrap();

        assert_eq!(
            controller.iov_effect_names(),
            Some(["ke".to_string()].as_slice())
        );
        assert_eq!(controller.omega_iov(), Some(&ndarray::array![[0.1]]));
        assert!(controller.kappa_log_prior().is_finite());
        assert_eq!(
            controller.log_posterior(),
            controller.likelihood() + controller.eta_log_prior() + controller.kappa_log_prior()
        );

        controller.step().unwrap();
        assert!(controller.likelihood().is_finite());
        assert!(controller.kappa_log_prior().is_finite());
        assert!(controller.acceptance_rate().is_some());
        assert!(controller
            .kappa_acceptance_rate()
            .is_some_and(|rate| (0.0..=1.0).contains(&rate)));
    }

    #[test]
    fn correlated_random_walk_reuses_one_standard_normal_vector() {
        let proposed =
            correlated_random_walk(&[1.0, 2.0], &[vec![2.0], vec![1.0, 3.0]], &[0.5, -1.0], 0.2)
                .unwrap();

        assert!((proposed[0] - 1.2).abs() < 1e-12);
        assert!((proposed[1] - 1.5).abs() < 1e-12);
        assert!(correlated_random_walk(&[0.0], &[vec![1.0]], &[0.0, 1.0], 1.0).is_err());
    }

    #[test]
    fn eta_block_proposal_uses_covariance_scale_and_adaptation() {
        let lower = vec![vec![1.0], vec![0.8, 0.6]];
        let normals = [[0.5, -1.0], [-0.25, 0.75], [1.2, 0.1], [-0.8, -0.4]];
        let uniforms = [0.2_f64, 0.9, 0.4, 0.7];
        let expected_trace = [
            [0.65, -0.3],
            [0.525, -0.175],
            [0.525, -0.175],
            [0.525, -0.175],
        ];
        let expected_ratios = [
            -0.9451955782312924,
            0.6944515306122447,
            -2.211747363945578,
            -0.4124850340136057,
        ];
        let expected_accepts = [true, true, false, false];
        let expected_scales = [0.55, 0.495];
        let expected_checkpoint_counts = [(2, 2), (0, 2)];
        let log_likelihood = |eta: &[f64]| {
            -0.5 * ((eta[0] - 0.3) / 0.5).powi(2) - 0.5 * ((eta[1] + 0.1) / 0.7).powi(2)
        };
        let log_prior = |eta: &[f64]| {
            -0.5 / (1.0 - 0.8_f64.powi(2))
                * (eta[0].powi(2) - 1.6 * eta[0] * eta[1] + eta[1].powi(2))
        };

        let mut eta = vec![0.4, -0.2];
        let mut scale = 0.5;
        let mut accepted = 0;
        let mut proposed = 0;
        let mut scale_index = 0;
        for (step, (z, uniform)) in normals.iter().zip(uniforms).enumerate() {
            let proposal = correlated_random_walk(&eta, &lower, z, scale).unwrap();
            let reference = [
                eta[0] + scale * lower[0][0] * z[0],
                eta[1] + scale * (lower[1][0] * z[0] + lower[1][1] * z[1]),
            ];
            assert!((proposal[0] - reference[0]).abs() < 1e-15);
            assert!((proposal[1] - reference[1]).abs() < 1e-15);

            let current_score = SubjectPosteriorScore {
                log_likelihood: log_likelihood(&eta),
                eta_log_prior: log_prior(&eta),
                kappa_log_prior: 0.0,
            };
            let proposed_score = SubjectPosteriorScore {
                log_likelihood: log_likelihood(&proposal),
                eta_log_prior: log_prior(&proposal),
                kappa_log_prior: 0.0,
            };
            let ratio = current_score.log_acceptance_ratio(proposed_score);
            let reference_ratio = proposed_score.log_posterior() - current_score.log_posterior();
            assert!((ratio - reference_ratio).abs() < 1e-15);
            assert!((ratio - expected_ratios[step]).abs() < 1e-12);

            let accept = ratio >= 0.0 || uniform.ln() < ratio;
            assert_eq!(accept, expected_accepts[step]);
            proposed += 1;
            if accept {
                eta = proposal;
                accepted += 1;
            }
            assert!((eta[0] - expected_trace[step][0]).abs() < 1e-12);
            assert!((eta[1] - expected_trace[step][1]).abs() < 1e-12);

            if (step + 1) % 2 == 0 {
                assert_eq!(
                    (accepted, proposed),
                    expected_checkpoint_counts[scale_index]
                );
                scale = adapt_block_step_size(
                    scale,
                    accepted as f64 / proposed as f64,
                    ETA_BLOCK_TARGET_ACCEPTANCE,
                );
                assert!((scale - expected_scales[scale_index]).abs() < 1e-12);
                scale_index += 1;
                accepted = 0;
                proposed = 0;
            }
        }

        let eta_unchanged = [0.7, -0.3];
        let kappa_0_unchanged = [0.1, 0.2];
        let kappa_1 = correlated_random_walk(
            &[-0.2, 0.4],
            &[vec![0.5], vec![0.1, 0.4]],
            &[-0.5, 0.25],
            0.3,
        )
        .unwrap();
        assert_eq!(eta_unchanged, [0.7, -0.3]);
        assert_eq!(kappa_0_unchanged, [0.1, 0.2]);
        assert!((kappa_1[0] + 0.275).abs() < 1e-12);
        assert!((kappa_1[1] - 0.415).abs() < 1e-12);
    }

    #[test]
    fn eta_block_kernel_runs_before_component_sweep_and_records_diagnostics() {
        let mut state = SaemState::from_problem(
            problem(),
            &SaemConfig::new()
                .n_chains(2)
                .mcmc_iterations(1)
                .eta_block_iterations(2)
                .adapt_interval(50)
                .seed(2024),
        )
        .unwrap();

        state.e_step().unwrap();

        let diagnostics = state.cycle_diagnostics.last().unwrap();
        assert_eq!(diagnostics.eta_block_proposals, 2 * 2 * 2);
        assert_eq!(
            diagnostics.eta_block_accepted + diagnostics.eta_block_rejected,
            diagnostics.eta_block_proposals
        );
        assert_eq!(diagnostics.eta_proposals, 2 * 2 * 2 + 2 * 2 * 2);
        assert_eq!(diagnostics.eta_block_subject_acceptance_rates.len(), 2);
        assert_eq!(
            diagnostics.eta_block_step_sizes_before_adaptation,
            vec![0.5, 0.5]
        );
        assert_eq!(
            diagnostics.eta_block_step_sizes_after_adaptation,
            vec![0.5, 0.5]
        );
    }

    #[test]
    fn controller_exposes_opt_in_eta_block_acceptance_and_scales() {
        let mut controller = problem()
            .fit_controller(
                SaemConfig::new()
                    .n_chains(2)
                    .eta_block_iterations(1)
                    .k1_iterations(1)
                    .k2_iterations(0)
                    .burn_in(1),
            )
            .unwrap();

        assert_eq!(
            controller.eta_block_step_sizes(),
            Some([0.5, 0.5].as_slice())
        );
        assert_eq!(controller.eta_block_acceptance_rate(), None);
        controller.step().unwrap();
        assert!(controller
            .eta_block_acceptance_rate()
            .is_some_and(|rate| (0.0..=1.0).contains(&rate)));
    }

    #[test]
    fn eta_block_scale_adapts_per_subject_toward_acceptance_target() {
        let mut state = SaemState::from_problem(
            problem(),
            &SaemConfig::new()
                .n_chains(1)
                .eta_block_iterations(1)
                .adapt_interval(1),
        )
        .unwrap();
        assert_eq!(state.eta_block_step_sizes, vec![0.5, 0.5]);

        state.eta_block_adaptation_accept_counts = vec![1, 0];
        state.eta_block_adaptation_proposal_counts = vec![1, 1];
        state.steps_since_adapt = 1;
        state.adapt_proposal_step_sizes();
        assert_eq!(state.eta_block_step_sizes, vec![0.55, 0.45]);
        assert_eq!(state.eta_block_adaptation_accept_counts, vec![0, 0]);
        assert_eq!(state.eta_block_adaptation_proposal_counts, vec![0, 0]);
    }

    #[test]
    fn kappa_block_scale_adapts_per_subject_toward_acceptance_target() {
        let mut state = SaemState::from_problem(
            iov_problem(),
            &SaemConfig::new().n_chains(2).adapt_interval(1),
        )
        .unwrap();
        assert_eq!(state.kappa_proposal_step_sizes, vec![0.5]);

        state.kappa_adaptation_accept_counts[0] = 1;
        state.kappa_adaptation_proposal_counts[0] = 1;
        state.steps_since_adapt = 1;
        state.adapt_proposal_step_sizes();
        assert!((state.kappa_proposal_step_sizes[0] - 0.55).abs() < 1e-12);

        state.kappa_adaptation_accept_counts[0] = 0;
        state.kappa_adaptation_proposal_counts[0] = 1;
        state.steps_since_adapt = 1;
        state.adapt_proposal_step_sizes();
        assert!((state.kappa_proposal_step_sizes[0] - 0.495).abs() < 1e-12);
    }

    #[test]
    fn iov_second_moment_weights_each_occasion_chain_sample_equally() {
        let kappas = vec![
            vec![vec![vec![1.0, 2.0]]],
            vec![vec![vec![3.0, 4.0], vec![5.0, 6.0]]],
        ];

        let covariance = covariance_from_kappas(&kappas).unwrap();

        assert!((covariance[[0, 0]] - 35.0 / 3.0).abs() < 1e-12);
        assert!((covariance[[0, 1]] - 44.0 / 3.0).abs() < 1e-12);
        assert!((covariance[[1, 0]] - 44.0 / 3.0).abs() < 1e-12);
        assert!((covariance[[1, 1]] - 56.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn iov_m_step_updates_omega_from_all_occasions() {
        let mut state = SaemState::from_problem(
            iov_problem(),
            &SaemConfig::new()
                .n_chains(2)
                .burn_in(0)
                .omega_sa_max_step(1.0),
        )
        .unwrap();
        state.cycle = 1;
        state.e_step().unwrap();
        for kappas in &mut state.kappas[0] {
            kappas[0][0] = 0.2;
            kappas[1][0] = -0.1;
        }

        state.m_step().unwrap();

        assert!((state.omega_iov.as_ref().unwrap()[[0, 0]] - 0.025).abs() < 1e-12);
        assert!(
            !state
                .cycle_diagnostics
                .last()
                .unwrap()
                .omega_iov_update_rejected
        );
    }

    #[test]
    fn covariance_update_status_drives_iiv_and_iov_cycle_rejection_diagnostics() {
        let config = SaemConfig::new()
            .n_chains(2)
            .burn_in(0)
            .omega_sa_max_step(1.0);
        let mut state = SaemState::from_problem(iov_problem(), &config).unwrap();
        state.cycle = 1;
        state.e_step().unwrap();
        state.iiv_second_moment.fill(f64::NAN);
        state.iov_second_moment.as_mut().unwrap().fill(f64::NAN);

        state.m_step().unwrap();

        let diagnostics = state.cycle_diagnostics.last().unwrap();
        assert!(diagnostics.omega_update_rejected);
        assert!(diagnostics.omega_iov_update_rejected);
    }

    #[test]
    fn iov_second_moment_uses_saem_smoothing_step() {
        let config = SaemConfig::new()
            .n_chains(2)
            .burn_in(0)
            .omega_sa_max_step(1.0)
            .k1_iterations(1)
            .k2_iterations(2);
        let mut state = SaemState::from_problem(iov_problem(), &config).unwrap();
        for kappas in &mut state.kappas[0] {
            kappas[0][0] = 0.2;
            kappas[1][0] = -0.1;
        }
        state.cycle = 1;
        state.m_step().unwrap();

        for kappas in &mut state.kappas[0] {
            kappas[0][0] = 0.2;
            kappas[1][0] = 0.2;
        }
        state.cycle = 3; // first smoothing iteration after K1: γ = 1/2
        state.m_step().unwrap();

        assert!((state.omega_iov.as_ref().unwrap()[[0, 0]] - 0.0325).abs() < 1e-12);
    }

    #[test]
    fn iov_m_step_preserves_fixed_entries_and_positive_definiteness_jointly() {
        let config = SaemConfig::new()
            .n_chains(2)
            .burn_in(0)
            .omega_sa_max_step(1.0);
        let mut state = SaemState::from_problem(configured_iov_problem(), &config).unwrap();
        state.cycle = 1;
        for chain in &mut state.kappas[0] {
            for kappa in chain {
                kappa[0] = 0.3;
                kappa[1] = 1.0;
            }
        }

        state.m_step().unwrap();

        let omega_iov = state.omega_iov.as_ref().unwrap();
        // With fixed b=.20 and c=.05, the exact constrained profile optimum is
        // S11 - 2(c/b)S12 + c²/b + (c²/b²)S22 = .015.
        assert!((omega_iov[[0, 0]] - 0.015).abs() < 1e-12);
        assert_eq!(omega_iov[[0, 1]], 0.05);
        assert_eq!(omega_iov[[1, 0]], 0.05);
        assert_eq!(omega_iov[[1, 1]], 0.20);
        assert!(omega_iov[[0, 0]] * omega_iov[[1, 1]] - omega_iov[[0, 1]].powi(2) > 0.0);
    }

    #[test]
    fn state_uses_declared_initial_omega() {
        let state =
            SaemState::from_problem(configured_omega_problem(), &SaemConfig::new().n_chains(2))
                .unwrap();

        assert_eq!(state.omega, ndarray::array![[0.25, 0.0], [0.0, 0.5]]);
        assert_eq!(state.proposal_step_sizes, vec![0.25, 0.25 * 2.0_f64.sqrt()]);
    }

    #[test]
    fn individual_parameters_add_eta_in_phi_space() {
        let mut state = SaemState::from_problem(problem(), &SaemConfig::new().n_chains(1)).unwrap();

        let initial = state.individual_parameters(0, 0);
        assert!((initial[0] - 0.2).abs() < 1e-12);
        assert!((initial[1] - 10.0).abs() < 1e-12);

        state.etas[0][0][0] = 2.0_f64.ln();
        state.etas[0][0][1] = 0.5_f64.ln();
        let individual = state.individual_parameters(0, 0);

        assert!((individual[0] - 0.4).abs() < 1e-12);
        assert!((individual[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn bounded_transforms_round_trip() {
        let logit = ParameterScale::Logit {
            lower: 0.0,
            upper: 1.0,
        };
        let probit = ParameterScale::Probit {
            lower: 0.0,
            upper: 1.0,
        };

        assert!((phi_to_psi(psi_to_phi(0.25, logit), logit) - 0.25).abs() < 1e-12);
        assert!((phi_to_psi(psi_to_phi(0.25, probit), probit) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn parametric_fit_controller_steps_like_nonparametric_controller() {
        let config = SaemConfig::new()
            .k1_iterations(2)
            .k2_iterations(1)
            .burn_in(1);
        let mut controller = problem().fit_controller(config).unwrap();

        assert_eq!(controller.cycle(), 0);
        assert!(controller.status().is_continue());
        assert!(controller.likelihood().is_finite());
        assert_eq!(controller.population_parameters(), &[0.2, 10.0]);
        assert_eq!(controller.random_effect_names(), &["ke", "v"]);
        assert_eq!(controller.iov_effect_names(), None);
        assert_eq!(controller.omega_iov(), None);
        assert_eq!(controller.residual_sigmas(), &[0.5]);
        assert_eq!(controller.acceptance_rate(), None);
        assert_eq!(controller.kappa_acceptance_rate(), None);
        assert_eq!(controller.rejected_proposals(), None);
        assert_eq!(controller.non_finite_proposals(), None);
        assert_eq!(controller.parameter_acceptance_rates(), None);
        assert_eq!(
            controller.proposal_step_sizes(),
            Some([0.5, 0.5].as_slice())
        );
        assert!(controller.eta_log_prior().is_finite());
        assert_eq!(
            controller.log_posterior(),
            controller.likelihood() + controller.eta_log_prior()
        );
        assert!(controller.negative_log_likelihood().is_finite());
        assert_eq!(
            controller.negative_log_likelihood(),
            -controller.likelihood()
        );
        assert!(controller.n2ll().is_finite());
        assert_eq!(controller.n_chains(), Some(25));
        assert_eq!(
            controller.omega(),
            Some(&ndarray::array![[1.0, 0.0], [0.0, 1.0]])
        );
        assert_eq!(controller.omega_diagonal(), Some(vec![1.0, 1.0]));
        assert_eq!(
            controller.log_acceptance_ratios(),
            Some([0.0, 0.0].as_slice())
        );
        assert_eq!(controller.total_iterations(), 3);
        assert_eq!(controller.step_size(), 0.0);

        assert!(controller.step().unwrap().is_continue());
        assert_eq!(controller.cycle(), 1);
        assert_eq!(controller.step_size(), 0.0);
        assert_eq!(controller.population_parameters(), &[0.2, 10.0]);
        assert_eq!(
            controller.omega(),
            Some(&ndarray::array![[1.0, 0.0], [0.0, 1.0]])
        );
        assert!(controller.acceptance_rate().is_some());
        assert_eq!(controller.kappa_acceptance_rate(), None);
        assert!(controller.rejected_proposals().is_some());
        assert_eq!(controller.non_finite_proposals(), Some(0));
        let parameter_acceptance_rates = controller.parameter_acceptance_rates().unwrap();
        assert_eq!(parameter_acceptance_rates.len(), 2);
        assert!(parameter_acceptance_rates
            .iter()
            .all(|rate| (0.0..=1.0).contains(rate)));
        assert!(controller.step().unwrap().is_continue());
        assert_eq!(controller.cycle(), 2);
        assert_eq!(controller.step_size(), 1.0);
        assert!(controller.step().unwrap().is_stop());
        assert_eq!(controller.cycle(), 3);
    }

    #[test]
    fn aborted_controller_preserves_typed_termination_reason() {
        let mut controller = problem()
            .fit_controller(SaemConfig::new().compute_map(false))
            .unwrap();
        controller.step().unwrap();
        controller.request_stop();

        let result = controller.into_result().unwrap();

        assert!(!result.converged());
        assert_eq!(result.termination_reason(), Some(&StopReason::Aborted));
        assert_ne!(result.termination_reason(), Some(&StopReason::MaxCycles));
        assert_ne!(
            result.termination_reason(),
            Some(&StopReason::NumericalFailure)
        );
        assert_eq!(result.iterations(), 1);
    }

    #[test]
    fn expectation_numerical_failure_stops_and_blocks_result() {
        let mut state = SaemState::from_problem(
            problem(),
            &SaemConfig::new()
                .n_chains(1)
                .mcmc_iterations(1)
                .compute_map(false),
        )
        .unwrap();
        state.omega[[0, 0]] = f64::NAN;

        let error = state.step().unwrap_err();
        let failure = error
            .downcast_ref::<NumericalFailure>()
            .expect("step error should retain its numerical failure type")
            .clone();

        assert_eq!(failure.attempted_cycle(), 1);
        assert_eq!(failure.phase(), NumericalFailurePhase::Expectation);
        assert!(!failure.source_message().is_empty());
        assert_eq!(state.status, Status::Stop(StopReason::NumericalFailure));
        assert_eq!(
            state.step().unwrap(),
            Status::Stop(StopReason::NumericalFailure)
        );

        let result_error = Box::new(state).into_result().unwrap_err();
        assert_eq!(
            result_error.downcast_ref::<NumericalFailure>(),
            Some(&failure)
        );
    }

    #[test]
    fn maximization_numerical_failure_stops_fit() {
        let mut state = SaemState::from_problem(
            problem(),
            &SaemConfig::new()
                .n_chains(1)
                .mcmc_iterations(1)
                .compute_map(false),
        )
        .unwrap();
        state.sufficient_statistics.mean_phi.pop();

        let error = state.step().unwrap_err();
        let failure = error
            .downcast_ref::<NumericalFailure>()
            .expect("step error should retain its numerical failure type");

        assert_eq!(failure.attempted_cycle(), 1);
        assert_eq!(failure.phase(), NumericalFailurePhase::Maximization);
        assert!(!failure.source_message().is_empty());
        assert_eq!(state.status, Status::Stop(StopReason::NumericalFailure));
    }

    #[test]
    fn result_assembly_numerical_failure_returns_no_result() {
        let mut state =
            SaemState::from_problem(problem(), &SaemConfig::new().n_chains(1).compute_map(false))
                .unwrap();
        state.etas[0].clear();

        let error = Box::new(state).into_result().unwrap_err();
        let failure = error
            .downcast_ref::<NumericalFailure>()
            .expect("result error should retain its numerical failure type");

        assert_eq!(failure.attempted_cycle(), 0);
        assert_eq!(failure.phase(), NumericalFailurePhase::ResultAssembly);
        assert!(!failure.source_message().is_empty());
    }

    #[test]
    fn proposal_score_uses_pmcore_likelihood_and_eta_prior() {
        let state = SaemState::from_problem(problem(), &SaemConfig::new().n_chains(1)).unwrap();
        let current_eta = state.etas[0][0].clone();
        let score = state
            .score_subject_latents(0, &current_eta, &state.kappas[0][0])
            .unwrap();

        assert_eq!(score.log_likelihood, state.subject_log_likelihoods[0]);
        assert_eq!(score.eta_log_prior, state.subject_log_priors[0]);
        assert_eq!(
            state
                .proposal_log_acceptance_ratio(0, 0, &current_eta)
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn component_random_walk_changes_only_selected_eta() {
        let mut state =
            SaemState::from_problem(problem(), &SaemConfig::new().n_chains(2).seed(2024)).unwrap();
        let current = vec![1.0, 2.0];

        let proposed = state.component_random_walk_eta(&current, 1);

        assert_eq!(proposed[0], current[0]);
        assert_ne!(proposed[1], current[1]);
    }

    #[test]
    fn component_scale_adaptation_uses_acceptance_bands_and_clamps() {
        assert!((adapt_component_step_size(1.0, 0.45) - 1.1).abs() < 1e-12);
        assert!((adapt_component_step_size(1.0, 0.44) - 0.9).abs() < 1e-12);
        assert_eq!(adapt_component_step_size(5.0, 1.0), 5.0);
        assert_eq!(adapt_component_step_size(1e-6, 0.0), 1e-6);
    }

    #[test]
    fn component_scale_adaptation_waits_for_interval_and_resets_counts() {
        let mut state =
            SaemState::from_problem(problem(), &SaemConfig::new().n_chains(2).adapt_interval(2))
                .unwrap();
        state.adaptation_accept_counts = vec![9, 1];
        state.adaptation_proposal_counts = vec![10, 10];
        state.steps_since_adapt = 1;

        state.adapt_proposal_step_sizes();
        assert_eq!(state.proposal_step_sizes, vec![0.5, 0.5]);

        state.steps_since_adapt = 2;
        state.adapt_proposal_step_sizes();
        assert_eq!(state.proposal_step_sizes, vec![0.55, 0.45]);
        assert_eq!(state.adaptation_accept_counts, vec![0, 0]);
        assert_eq!(state.adaptation_proposal_counts, vec![0, 0]);
        assert_eq!(state.steps_since_adapt, 0);
    }

    #[test]
    fn e_step_runs_seeded_random_walk_for_all_chains_and_records_acceptance_rate() {
        let config = SaemConfig::new().n_chains(3).mcmc_iterations(2).seed(2024);
        let mut state = SaemState::from_problem(problem(), &config).unwrap();
        let initial_etas = state.etas.clone();

        state.e_step().unwrap();

        let acceptance_rate = state.acceptance_rate().unwrap();
        assert!((0.0..=1.0).contains(&acceptance_rate));
        assert_eq!(state.last_log_acceptance_ratios.len(), 2);
        assert_eq!(state.last_parameter_acceptance_rates.len(), 2);
        assert!(state
            .last_parameter_acceptance_rates
            .iter()
            .all(|rate| (0.0..=1.0).contains(rate)));
        assert!(state
            .last_log_acceptance_ratios
            .iter()
            .all(|value| value.is_finite()));
        assert_ne!(state.etas, initial_etas);
        assert!(state
            .etas
            .iter()
            .flat_map(|subject_chains| subject_chains.iter())
            .all(|eta| eta.len() == 2));
    }

    #[test]
    fn cycle_diagnostics_separate_eta_kappa_counts_and_schedule_phases() {
        let config = SaemConfig::new()
            .n_chains(2)
            .mcmc_iterations(1)
            .burn_in(1)
            .k1_iterations(2)
            .k2_iterations(1);
        let mut state = SaemState::from_problem(iov_problem(), &config).unwrap();

        state.step().unwrap();
        state.step().unwrap();
        state.step().unwrap();

        assert_eq!(state.cycle_diagnostics.len(), 3);
        assert_eq!(state.cycle_diagnostics[0].phase, SaemPhase::BurnIn);
        assert_eq!(state.cycle_diagnostics[1].phase, SaemPhase::Exploration);
        assert_eq!(state.cycle_diagnostics[2].phase, SaemPhase::Smoothing);
        for diagnostics in &state.cycle_diagnostics {
            assert_eq!(diagnostics.eta_proposals, 4);
            assert_eq!(
                diagnostics.eta_accepted + diagnostics.eta_rejected,
                diagnostics.eta_proposals
            );
            assert_eq!(diagnostics.kappa_proposals, 4);
            assert_eq!(
                diagnostics.kappa_accepted + diagnostics.kappa_rejected,
                diagnostics.kappa_proposals
            );
            assert_eq!(diagnostics.eta_parameter_acceptance_rates.len(), 2);
            assert_eq!(
                diagnostics.eta_proposal_step_sizes_before_adaptation.len(),
                2
            );
            assert_eq!(
                diagnostics.eta_proposal_step_sizes_after_adaptation.len(),
                2
            );
            assert_eq!(diagnostics.kappa_subject_acceptance_rates.len(), 1);
            assert_eq!(
                diagnostics
                    .kappa_proposal_step_sizes_before_adaptation
                    .len(),
                1
            );
            assert_eq!(
                diagnostics.kappa_proposal_step_sizes_after_adaptation.len(),
                1
            );
        }
        assert_eq!(
            state.cycle_diagnostics[0].stochastic_approximation_step,
            0.0
        );
        assert_eq!(state.cycle_diagnostics[0].covariance_step, 0.1);
    }

    #[test]
    fn warning_aggregation_preserves_kind_output_first_cycle_and_counts() {
        let config = SaemConfig::new()
            .n_chains(1)
            .mcmc_iterations(1)
            .burn_in(0)
            .k1_iterations(1)
            .k2_iterations(0);
        let mut state = SaemState::from_problem(problem(), &config).unwrap();
        state.step().unwrap();
        let cycle = &mut state.cycle_diagnostics[0];
        cycle.omega_update_rejected = true;
        cycle.eta_non_finite = 2;
        cycle.eta_block_non_finite = 7;
        let residual = &mut cycle.residual_diagnostics[0];
        residual.update_rejected = true;
        residual.proportional_floor_count = 3;
        residual.non_finite_prediction_count = 4;
        residual.exponential_domain_violation_count = 5;
        residual.combined_additive_collapse_warning = true;
        residual.optimizer_converged = Some(false);

        let warnings = parametric_warnings(&state.cycle_diagnostics, None);

        assert!(warnings.contains(&ParametricWarning::OmegaUpdateRejected {
            first_iteration: 1,
            cycles: 1,
        }));
        assert!(
            warnings.contains(&ParametricWarning::EtaNonFiniteProposals {
                first_iteration: 1,
                count: 2,
            })
        );
        assert!(
            warnings.contains(&ParametricWarning::EtaBlockNonFiniteProposals {
                first_iteration: 1,
                count: 7,
            })
        );
        assert!(
            warnings.contains(&ParametricWarning::ResidualUpdateRejected {
                output: "0".to_owned(),
                first_iteration: 1,
                cycles: 1,
            })
        );
        assert!(
            warnings.contains(&ParametricWarning::ProportionalPredictionFloor {
                output: "0".to_owned(),
                first_iteration: 1,
                count: 3,
            })
        );
        assert!(
            warnings.contains(&ParametricWarning::NonFiniteResidualPrediction {
                output: "0".to_owned(),
                first_iteration: 1,
                count: 4,
            })
        );
        assert!(
            warnings.contains(&ParametricWarning::ExponentialDomainViolation {
                output: "0".to_owned(),
                first_iteration: 1,
                count: 5,
            })
        );
        assert!(
            warnings.contains(&ParametricWarning::CombinedAdditiveCollapse {
                output: "0".to_owned(),
                first_iteration: 1,
                cycles: 1,
            })
        );
        assert!(
            warnings.contains(&ParametricWarning::ResidualOptimizerNotConverged {
                output: "0".to_owned(),
                first_iteration: 1,
                cycles: 1,
            })
        );
    }

    #[test]
    fn covariance_stability_records_fixed_iiv_and_iov_margins_and_output_rows() {
        let result = markov_iov_problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(1)
                    .mcmc_iterations(1)
                    .burn_in(0)
                    .k1_iterations(1)
                    .k2_iterations(0)
                    .covariance_stability(CovarianceStabilityConfig::new(0.01, 1)),
            )
            .unwrap();
        let cycle = &result.cycle_diagnostics()[0];
        assert!((cycle.omega_relative_spd_margin.unwrap() - 1.0).abs() < 1e-12);
        assert!((cycle.omega_iov_relative_spd_margin.unwrap() - 1.0).abs() < 1e-12);

        let tables = result.tables(0.0, 0.0).unwrap();
        let stability_rows = tables
            .statistics
            .iter()
            .filter(|row| row.kind == "covariance_stability")
            .collect::<Vec<_>>();
        assert_eq!(stability_rows.len(), 2);
        assert!(stability_rows.iter().any(|row| {
            row.name == "omega_relative_spd_margin"
                && row.value.is_some_and(|value| (value - 1.0).abs() < 1e-12)
        }));
        assert!(stability_rows.iter().any(|row| {
            row.name == "omega_iov_relative_spd_margin"
                && row.value.is_some_and(|value| (value - 1.0).abs() < 1e-12)
        }));
    }

    #[test]
    fn covariance_boundary_rejection_requires_a_complete_consecutive_window() {
        let config = SaemConfig::new()
            .n_chains(1)
            .mcmc_iterations(1)
            .burn_in(0)
            .k1_iterations(1)
            .k2_iterations(0);
        let mut state = SaemState::from_problem(problem(), &config).unwrap();
        state.step().unwrap();
        let base = state.cycle_diagnostics[0].clone();
        let policy = CovarianceStabilityConfig::new(0.01, 3);
        let pattern = [
            (1, 0.005, true),
            (2, 0.004, true),
            (3, 0.02, true),
            (4, 0.003, true),
            (5, 0.002, true),
            (6, 0.001, true),
        ];
        let cycles = pattern
            .into_iter()
            .map(|(iteration, margin, rejected)| {
                let mut cycle = base.clone();
                cycle.iteration = iteration;
                cycle.omega_relative_spd_margin = Some(margin);
                cycle.omega_update_rejected = rejected;
                cycle
            })
            .collect::<Vec<_>>();

        assert_eq!(
            covariance_boundary_rejection_summary(&cycles[..2], policy, false),
            CovarianceBoundaryRejectionSummary {
                first_iteration: None,
                longest_run: 2,
            }
        );
        assert_eq!(
            covariance_boundary_rejection_summary(&cycles, policy, false),
            CovarianceBoundaryRejectionSummary {
                first_iteration: Some(4),
                longest_run: 3,
            }
        );
        let warnings = parametric_warnings(&cycles, Some(policy));
        assert!(
            warnings.contains(&ParametricWarning::OmegaBoundaryRejection {
                first_iteration: 4,
                longest_run: 3,
            })
        );

        let mut mismatched_iov = base.clone();
        mismatched_iov.omega_iov_relative_spd_margin = Some(0.005);
        mismatched_iov.omega_update_rejected = true;
        mismatched_iov.omega_iov_update_rejected = false;
        assert_eq!(
            covariance_boundary_rejection_summary(&[mismatched_iov.clone()], policy, true),
            CovarianceBoundaryRejectionSummary::default()
        );
        mismatched_iov.omega_iov_update_rejected = true;
        assert_eq!(
            covariance_boundary_rejection_summary(&[mismatched_iov], policy, true).longest_run,
            1
        );

        let iov_cycles = (1..=3)
            .map(|iteration| {
                let mut cycle = base.clone();
                cycle.iteration = iteration;
                cycle.omega_iov_relative_spd_margin = Some(policy.minimum_relative_spd_margin);
                cycle.omega_iov_update_rejected = true;
                cycle
            })
            .collect::<Vec<_>>();
        assert_eq!(
            covariance_boundary_rejection_summary(&iov_cycles, policy, true),
            CovarianceBoundaryRejectionSummary {
                first_iteration: Some(1),
                longest_run: 3,
            }
        );
        assert!(parametric_warnings(&iov_cycles, Some(policy)).contains(
            &ParametricWarning::OmegaIovBoundaryRejection {
                first_iteration: 1,
                longest_run: 3,
            }
        ));

        let criterion = evaluate_criterion(
            "omega_boundary_rejection_run",
            Some(3.0),
            policy.rejection_window as f64,
            |observed| observed < policy.rejection_window as f64,
        );
        assert_eq!(
            criterion.status,
            OperationalConvergenceCriterionStatus::NotSatisfied
        );
    }

    #[test]
    fn m_step_recenters_etas_before_updating_iiv_second_moment() {
        let mut state = SaemState::from_problem(
            problem(),
            &SaemConfig::new()
                .n_chains(1)
                .burn_in(0)
                .omega_sa_max_step(0.1),
        )
        .unwrap();
        state.cycle = 1;
        for subject_chains in &mut state.etas {
            for eta in subject_chains {
                eta[0] = 2.0_f64.ln();
            }
        }
        let individual_before = state.individual_parameters(0, 0);

        state.m_step().unwrap();

        let individual_after = state.individual_parameters(0, 0);
        assert!((individual_before[0] - individual_after[0]).abs() < 1e-12);
        assert!(state
            .etas
            .iter()
            .flat_map(|subject_chains| subject_chains.iter())
            .all(|eta| eta[0].abs() < 1e-12));
        assert!((state.population_parameters[0] - 0.4).abs() < 1e-12);
        assert!((state.population_parameters[1] - 10.0).abs() < 1e-12);
        let information = state.information.diagnostics();
        let ke_coordinate = information
            .coordinates
            .iter()
            .position(|coordinate| coordinate.name == "phi:ke")
            .unwrap();
        // Two pre-M-step absolute phi values each differ from the old
        // population by ln(2). Post-update or un-recentered evaluation would
        // give a different score (zero or double-counted population shift).
        assert!((information.delta[ke_coordinate] - 2.0 * 2.0_f64.ln()).abs() < 1e-12);
        let expected_omega = ndarray::array![[0.9, 0.0], [0.0, 0.9]];
        assert!(state
            .iiv_second_moment
            .iter()
            .zip(expected_omega.iter())
            .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
        assert!(state
            .omega
            .iter()
            .zip(expected_omega.iter())
            .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
    }

    #[test]
    fn exploration_covariance_cap_prevents_one_draw_rank_one_collapse() {
        fn correlation(omega: &Array2<f64>) -> f64 {
            omega[[0, 1]] / (omega[[0, 0]] * omega[[1, 1]]).sqrt()
        }

        let make_state = |omega_sa_max_step| {
            SaemState::from_problem(
                correlated_omega_problem(),
                &SaemConfig::new()
                    .n_chains(1)
                    .burn_in(0)
                    .omega_sa_max_step(omega_sa_max_step),
            )
            .unwrap()
        };
        let mut guarded = make_state(0.1);
        let mut uncapped = make_state(1.0);
        for state in [&mut guarded, &mut uncapped] {
            state.cycle = 1;
            state.etas[0][0] = vec![2.0, 2.0];
            state.etas[1][0] = vec![-2.0, -2.0];
            state.m_step().unwrap();
            assert!(state.omega[[0, 0]] >= state.initialization.schedule.minimum_variance);
            assert!(state.omega[[1, 1]] >= state.initialization.schedule.minimum_variance);
            assert!(
                state.omega[[0, 0]] * state.omega[[1, 1]] - state.omega[[0, 1]].powi(2) > 0.0,
                "omega: {:?}",
                state.omega
            );
        }

        let guarded_correlation = correlation(&guarded.omega);
        let uncapped_correlation = correlation(&uncapped.omega);
        assert!(guarded_correlation < 0.85);
        assert!(uncapped_correlation > 0.85);
        assert!(uncapped_correlation - guarded_correlation > 0.05);
    }

    #[test]
    fn m_step_preserves_fixed_omega_and_structural_zeros() {
        let mut state = SaemState::from_problem(
            configured_omega_problem(),
            &SaemConfig::new()
                .n_chains(2)
                .burn_in(0)
                .omega_sa_max_step(1.0),
        )
        .unwrap();
        state.cycle = 1;
        for (subject_index, subject_chains) in state.etas.iter_mut().enumerate() {
            let sign = if subject_index == 0 { 1.0 } else { -1.0 };
            for eta in subject_chains {
                eta[0] = sign;
                eta[1] = 2.0 * sign;
            }
        }

        state.m_step().unwrap();

        assert!((state.omega[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((state.omega[[1, 1]] - 0.5).abs() < 1e-12);
        assert_eq!(state.omega[[0, 1]], 0.0);
        assert_eq!(state.omega[[1, 0]], 0.0);
    }

    #[test]
    fn fixed_population_effect_is_not_updated_and_omega_uses_fixed_center() {
        let mut state = SaemState::from_problem(
            fixed_population_iiv_problem(),
            &SaemConfig::new()
                .n_chains(2)
                .burn_in(0)
                .omega_sa_max_step(1.0),
        )
        .unwrap();
        state.cycle = 1;
        for subject_chains in &mut state.etas {
            for eta in subject_chains {
                eta[0] = 2.0_f64.ln();
            }
        }

        state.m_step().unwrap();

        assert!((state.population_parameters[0] - 0.2).abs() < 1e-12);
        assert!(state
            .etas
            .iter()
            .flat_map(|subject_chains| subject_chains.iter())
            .all(|eta| (eta[0] - 2.0_f64.ln()).abs() < 1e-12));
        assert!((state.omega[[0, 0]] - 2.0_f64.ln().powi(2)).abs() < 1e-12);
        let individual = state.individual_parameters(0, 0);
        assert!((individual[0] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn m_step_updates_simple_residual_sigma_from_statrese() {
        let mut state = SaemState::from_problem(
            constant_error_problem(),
            &SaemConfig::new().n_chains(1).burn_in(0),
        )
        .unwrap();
        state.cycle = 1;
        let candidate_sigma = state
            .current_residual_statistics()
            .unwrap()
            .output(0)
            .and_then(|statistic| statistic.sigma())
            .unwrap();
        let expected_sigma = state.initialization.schedule.guarded_residual_sigma(
            state.cycle,
            state.residual_sigmas[0],
            candidate_sigma,
        );

        state.m_step().unwrap();

        assert!((state.residual_sigmas[0] - expected_sigma).abs() < 1e-12);
        assert_eq!(
            state.error_models.get(0),
            Some(&ResidualErrorModel::constant(expected_sigma))
        );
    }

    #[test]
    fn sparse_second_output_reports_only_declared_residual_model() {
        let result = sparse_second_output_problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(1)
                    .k1_iterations(1)
                    .k2_iterations(0)
                    .burn_in(0)
                    .compute_map(false),
            )
            .unwrap();

        assert_eq!(result.residual_sigmas().len(), 1);
        assert_eq!(result.residual_error_estimates().len(), 1);
        assert_eq!(result.residual_error_estimates()[0].output, "measured");
        assert_eq!(result.residual_error_estimates()[0].output_index, 1);
        assert_eq!(result.cycle_diagnostics().len(), 1);
        assert_eq!(result.cycle_diagnostics()[0].residual_diagnostics.len(), 1);
        assert_eq!(
            result.cycle_diagnostics()[0].residual_diagnostics[0].output,
            "measured"
        );
        assert_eq!(
            result.cycle_diagnostics()[0].residual_diagnostics[0].output_index,
            1
        );
    }

    #[test]
    fn averaged_sparse_second_output_preserves_index_name_and_arithmetic_mean() {
        let result = sparse_second_output_problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(1)
                    .k1_iterations(1)
                    .k2_iterations(3)
                    .burn_in(0)
                    .averaged_iterates(0.75)
                    .compute_map(false)
                    .seed(71_002),
            )
            .expect("averaged sparse-output fit should complete");

        let metadata = result.estimator_metadata();
        assert!(metadata.average_applied);
        assert_eq!(metadata.averaging_start_cycle, Some(2));
        assert_eq!(metadata.averaged_iterations, 3);
        let estimate = result
            .residual_error_estimates()
            .first()
            .expect("sparse residual estimate");
        assert_eq!(
            (estimate.output_index, estimate.output.as_str()),
            (1, "measured")
        );
        let smoothing = &result.cycle_diagnostics()[1..];
        let expected = smoothing
            .iter()
            .map(|cycle| {
                let residual = cycle
                    .residual_error_estimates
                    .first()
                    .expect("sparse cycle residual");
                assert_eq!(
                    (residual.output_index, residual.output.as_str()),
                    (1, "measured")
                );
                primary_sigma_parameter(&residual.model)
            })
            .sum::<f64>()
            / smoothing.len() as f64;
        assert!((primary_sigma_parameter(&estimate.model) - expected).abs() < 1e-12);
    }

    #[test]
    fn averaged_multi_output_residuals_preserve_fixed_and_fixed_zero_components() {
        let result = mixed_residual_output_problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(1)
                    .k1_iterations(1)
                    .k2_iterations(3)
                    .burn_in(0)
                    .averaged_iterates(0.75)
                    .compute_map(false)
                    .seed(71_003),
            )
            .expect("averaged mixed-output fit should complete");
        let estimates = result.residual_error_estimates();
        assert_eq!(estimates.len(), 2);
        assert_eq!(
            (estimates[0].output_index, estimates[0].output.as_str()),
            (0, "fixed")
        );
        assert_eq!(estimates[0].model, ResidualErrorModel::constant(0.5));
        assert!(!estimates[0].estimated);
        assert_eq!(
            (estimates[1].output_index, estimates[1].output.as_str()),
            (1, "mixed")
        );
        assert_eq!(estimates[1].combined_additive_estimated, Some(false));
        assert_eq!(estimates[1].combined_proportional_estimated, Some(true));
        let ResidualErrorModel::Combined { a, b } = estimates[1].model else {
            panic!("expected combined residual model");
        };
        assert_eq!(a, 0.0);
        let smoothing = &result.cycle_diagnostics()[1..];
        let expected_b = smoothing
            .iter()
            .map(|cycle| match cycle.residual_error_estimates[1].model {
                ResidualErrorModel::Combined { a, b } => {
                    assert_eq!(a, 0.0);
                    b
                }
                _ => panic!("expected combined cycle residual model"),
            })
            .sum::<f64>()
            / smoothing.len() as f64;
        assert!((b - expected_b).abs() < 1e-12);
        assert!(result.cycle_diagnostics().iter().all(|cycle| {
            cycle.residual_error_estimates[0].model == ResidualErrorModel::constant(0.5)
        }));
    }

    #[test]
    fn correlated_residual_averaging_preserves_fixed_components_and_rejects_family_changes() {
        let averaged = average_residual_model(
            ResidualErrorModel::correlated_combined(0.3, 0.1, 0.2),
            ResidualErrorModel::correlated_combined(0.5, 0.2, -0.4),
            true,
            [true, true],
            [false, true, true],
            2,
        )
        .unwrap();
        let ResidualErrorModel::CorrelatedCombined { a, b, rho } = averaged else {
            panic!("expected correlated-combined average")
        };
        assert_eq!(a, 0.3);
        assert!((b - 0.15).abs() < 1e-15);
        assert!((rho + 0.1).abs() < 1e-15);
        assert!(average_residual_model(
            averaged,
            ResidualErrorModel::combined(0.3, 0.15),
            true,
            [true, true],
            [true, true, true],
            3,
        )
        .is_err());
    }

    fn assert_prediction_points_equal(
        actual: &pharmsol::simulator::prediction::SubjectPredictions,
        expected: &pharmsol::simulator::prediction::SubjectPredictions,
    ) {
        assert_eq!(actual.predictions().len(), expected.predictions().len());
        for (actual, expected) in actual.predictions().iter().zip(expected.predictions()) {
            assert_eq!(actual.time(), expected.time());
            assert_eq!(actual.observation(), expected.observation());
            assert_eq!(actual.prediction(), expected.prediction());
            assert_eq!(actual.outeq(), expected.outeq());
            assert_eq!(actual.errorpoly(), expected.errorpoly());
            assert_eq!(actual.state(), expected.state());
            assert_eq!(actual.occasion(), expected.occasion());
            assert_eq!(actual.censoring(), expected.censoring());
        }
    }

    #[test]
    fn population_predictions_match_direct_execution_and_metadata() {
        let result = problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(1)
                    .k1_iterations(1)
                    .k2_iterations(0)
                    .burn_in(1),
            )
            .unwrap();
        let predictions = result.population_predictions(0.25, 0.0).unwrap();
        let expanded = result.data().clone().expand(0.25, 0.0);

        assert_eq!(predictions.len(), expanded.subjects().len());
        assert_eq!(expanded.subjects()[0].id(), "s1");
        assert_eq!(expanded.subjects()[1].id(), "s2");
        for (subject, actual) in expanded.subjects().iter().zip(&predictions) {
            let expected = result
                .equation()
                .estimate_predictions_dense(subject, result.population_parameters())
                .unwrap();
            assert_prediction_points_equal(actual, &expected);
        }
    }

    #[test]
    fn fixed_zero_latent_conditional_predictions_equal_population_predictions() {
        let result = fixed_no_iiv_problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(1)
                    .k1_iterations(1)
                    .k2_iterations(0)
                    .burn_in(1),
            )
            .unwrap();

        assert!(result.conditional_modes().is_empty());
        let population = result.population_predictions(0.25, 0.0).unwrap();
        let conditional = result.conditional_predictions(0.25, 0.0).unwrap();
        assert_eq!(conditional.len(), population.len());
        for (conditional, population) in conditional.iter().zip(&population) {
            assert_prediction_points_equal(conditional, population);
        }
    }

    #[test]
    fn iov_conditional_predictions_use_each_occasion_kappa_in_order() {
        let mut result = iov_problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(1)
                    .k1_iterations(1)
                    .k2_iterations(0)
                    .burn_in(1),
            )
            .unwrap();
        result.conditional_modes[0].eta.fill(0.0);
        result.conditional_modes[0].kappas[0].values[0] = -0.2;
        result.conditional_modes[0].kappas[1].values[0] = 0.3;

        let actual = result.conditional_predictions(0.25, 0.0).unwrap();
        assert_eq!(actual.len(), 1);
        let expanded = result.data().clone().expand(0.25, 0.0);
        let subject = &expanded.subjects()[0];
        let mode = &result.conditional_modes()[0];
        let mut expected_points = Vec::new();
        for (occasion, kappa) in subject.occasions().iter().zip(&mode.kappas) {
            let parameters = occasion_psi(
                result.population_parameters(),
                &result.parameter_scales,
                &result.random_effect_indices,
                &mode.eta,
                &result.iov_effect_indices,
                &kappa.values,
            )
            .unwrap();
            let occasion_subject =
                Subject::from_occasions(subject.id().clone(), vec![occasion.clone()]);
            for mut prediction in result
                .equation()
                .estimate_predictions_dense(&occasion_subject, &parameters)
                .unwrap()
                .predictions()
                .iter()
                .cloned()
            {
                *prediction.mut_occasion() = occasion.index();
                expected_points.push(prediction);
            }
        }
        let expected = pharmsol::simulator::prediction::SubjectPredictions::from(expected_points);
        assert_prediction_points_equal(&actual[0], &expected);
        assert!(actual[0]
            .predictions()
            .windows(2)
            .any(|pair| pair[0].occasion() != pair[1].occasion()));
        let occasion_predictions = subject
            .occasions()
            .iter()
            .map(|occasion| {
                actual[0]
                    .predictions()
                    .iter()
                    .find(|prediction| {
                        prediction.occasion() == occasion.index()
                            && prediction.observation().is_some()
                    })
                    .unwrap()
                    .prediction()
            })
            .collect::<Vec<_>>();
        assert_ne!(occasion_predictions[0], occasion_predictions[1]);
    }

    #[test]
    fn e_step_rescores_chain_zero_parameters() {
        let mut state = SaemState::from_problem(problem(), &SaemConfig::new().n_chains(1)).unwrap();
        let initial = state.log_likelihood();

        state.etas[0][0][0] = 2.0_f64.ln();
        state.e_step().unwrap();

        assert!(state.log_likelihood().is_finite());
        assert_ne!(state.log_likelihood(), initial);
        assert_eq!(state.negative_log_likelihood(), -state.log_likelihood());
    }

    #[test]
    fn iov_result_retains_named_omega_iov() {
        let result = iov_problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(2)
                    .k1_iterations(1)
                    .k2_iterations(0)
                    .burn_in(1),
            )
            .unwrap();

        assert_eq!(result.iov_effect_names(), &["ke"]);
        assert_eq!(result.omega_iov(), Some(&ndarray::array![[0.1]]));
        assert_eq!(result.conditional_modes().len(), 1);
        assert_eq!(result.conditional_modes()[0].kappas.len(), 2);
        assert!(result.conditional_modes()[0].objective.is_finite());
    }

    #[test]
    fn result_reports_final_chain_means_for_eta_and_kappa() {
        let mut state =
            SaemState::from_problem(iov_problem(), &SaemConfig::new().n_chains(2)).unwrap();
        state.etas[0][0][0] = 0.2;
        state.etas[0][1][0] = 0.4;
        state.kappas[0][0][0][0] = -0.2;
        state.kappas[0][1][0][0] = 0.4;
        state.kappas[0][0][1][0] = 0.1;
        state.kappas[0][1][1][0] = 0.3;

        let result = Box::new(state).into_result().unwrap();

        assert_eq!(result.eta_chain_means().len(), 1);
        assert!((result.eta_chain_means()[0].values[0] - 0.3).abs() < 1e-12);
        assert_eq!(result.kappa_chain_means().len(), 2);
        assert_eq!(result.kappa_chain_means()[0].occasion_index, 0);
        assert!((result.kappa_chain_means()[0].values[0] - 0.1).abs() < 1e-12);
        assert_eq!(result.kappa_chain_means()[1].occasion_index, 1);
        assert!((result.kappa_chain_means()[1].values[0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn result_retains_immutable_cycle_diagnostics() {
        let config = SaemConfig::new()
            .n_chains(1)
            .mcmc_iterations(1)
            .burn_in(1)
            .k1_iterations(1)
            .k2_iterations(1)
            .compute_map(false);
        let mut state = SaemState::from_problem(problem(), &config).unwrap();
        state.step().unwrap();
        state.step().unwrap();

        let result = Box::new(state).into_result().unwrap();

        assert_eq!(result.parameter_names(), ["ke", "v"]);
        assert_eq!(result.data().subjects().len(), 2);
        assert_eq!(
            result
                .equation()
                .metadata()
                .expect("retained equation metadata")
                .outputs()[0]
                .name(),
            "0"
        );
        assert_eq!(result.cycle_diagnostics().len(), 2);
        assert_eq!(result.cycle_diagnostics()[0].iteration, 1);
        assert_eq!(result.cycle_diagnostics()[0].phase, SaemPhase::BurnIn);
        assert_eq!(result.cycle_diagnostics()[1].iteration, 2);
        assert_eq!(result.cycle_diagnostics()[1].phase, SaemPhase::Smoothing);
        assert_eq!(
            result.cycle_diagnostics()[0].population_parameters,
            vec![0.2, 10.0]
        );
        let final_cycle = &result.cycle_diagnostics()[1];
        assert_eq!(
            final_cycle.population_parameters,
            result.population_parameters()
        );
        assert_eq!(&final_cycle.omega, result.omega());
        assert_eq!(final_cycle.omega_iov.as_ref(), result.omega_iov());
        assert_eq!(
            final_cycle.residual_error_estimates,
            result.residual_error_estimates()
        );
        assert!(final_cycle.conditional_negative_log_likelihood.is_finite());
        assert!(final_cycle.eta_log_prior.is_finite());
        assert!(final_cycle.kappa_log_prior.is_finite());
    }

    #[test]
    fn conditional_modes_can_be_disabled_without_relabeling_chain_means() {
        let result = problem()
            .fit_with(
                SaemConfig::new()
                    .n_chains(2)
                    .k1_iterations(1)
                    .k2_iterations(0)
                    .burn_in(1)
                    .compute_map(false),
            )
            .unwrap();

        assert!(result.conditional_modes().is_empty());
        assert_eq!(result.eta_chain_means().len(), 2);
        let error = result.conditional_predictions(0.25, 0.0).unwrap_err();
        assert_eq!(
            error.to_string(),
            "conditional predictions require conditional modes; rerun with compute_map(true)"
        );
    }

    #[test]
    fn population_uncertainty_wires_analytical_fit_summary_without_changing_estimates() {
        let equation = analytical! {
            name: "population_uncertainty_summary_fixture",
            params: [ke, v],
            states: [central],
            outputs: [cp],
            routes: [infusion(iv) -> central],
            structure: one_compartment,
            out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
        };
        let data = Data::new(vec![
            Subject::builder("uncertainty-1")
                .infusion(0.0, 100.0, "iv", 0.5)
                .observation(1.0, 4.8, "cp")
                .observation(3.0, 3.0, "cp")
                .build(),
            Subject::builder("uncertainty-2")
                .infusion(0.0, 120.0, "iv", 0.5)
                .observation(1.0, 5.4, "cp")
                .observation(3.0, 3.2, "cp")
                .build(),
        ]);
        let problem = EstimationProblem::parametric(equation, data)
            .parameter(Parameter::log("ke").with_initial(0.25))
            .parameter(
                Parameter::log("v")
                    .with_initial(20.0)
                    .fixed()
                    .without_random_effect(),
            )
            .omega(Omega::new().fixed_variance("ke", 0.09))
            .error_model(
                "cp",
                ParametricErrorModel::new(ResidualErrorModel::constant(0.4)).fixed(),
            )
            .build()
            .expect("population uncertainty analytical fixture");
        let mut result = problem
            .fit_with(
                SaemConfig::new()
                    .seed(0x6a_2026)
                    .n_chains(2)
                    .mcmc_iterations(1)
                    .burn_in(1)
                    .k1_iterations(1)
                    .k2_iterations(0)
                    .compute_map(false),
            )
            .expect("population uncertainty analytical fit");
        let estimates_before = result.population_parameters().to_vec();
        let objective_before = result.objf();
        assert_eq!(estimates_before, vec![0.25, 20.0]);
        assert_eq!(result.estimated_parameters(), &[true, false]);
        assert_eq!(
            result.population_uncertainty(),
            &derive_population_uncertainty(result.information_diagnostics())
        );

        let coordinates = result.information_diagnostics().coordinates.clone();
        assert_eq!(coordinates.len(), 1);
        assert_eq!(
            coordinates[0].kind,
            InformationCoordinateKind::Population { parameter_index: 0 }
        );
        result.population_uncertainty = PopulationUncertaintyDiagnostics {
            coordinates,
            free_covariance: Some(vec![vec![0.04]]),
            free_standard_errors: Some(vec![0.2]),
            spectral_condition_number: Some(1.0),
            status: PopulationUncertaintyStatus::Available,
            regularization: PopulationUncertaintyRegularization::None,
        };

        let summary = result.population_summary();
        assert_eq!(result.population_parameters(), estimates_before);
        assert_eq!(result.objf().to_bits(), objective_before.to_bits());
        assert_eq!(
            summary
                .parameters
                .iter()
                .map(|parameter| parameter.estimate)
                .collect::<Vec<_>>(),
            estimates_before
        );
        assert!(
            (summary.parameters[0]
                .sd
                .expect("free log-scale parameter SE")
                - 0.2 * estimates_before[0])
                .abs()
                < 1e-12
        );
        assert!(
            (summary.parameters[0]
                .cv_percent
                .expect("free log-scale parameter CV")
                - 20.0)
                .abs()
                < 1e-12
        );
        assert_eq!(summary.parameters[1].sd, None);
        assert_eq!(summary.parameters[1].cv_percent, None);
    }

    #[test]
    fn initialization_result_is_non_converged_snapshot() {
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(1)
            .burn_in(1);
        let result = problem().fit_with(config).unwrap();
        let summary = result.summary();

        assert!(!result.converged());
        assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
        assert_ne!(result.termination_reason(), Some(&StopReason::Aborted));
        assert_ne!(
            result.termination_reason(),
            Some(&StopReason::NumericalFailure)
        );
        assert_eq!(result.iterations(), 2);
        assert_eq!(summary.subject_count, 2);
        assert_eq!(summary.observation_count, 4);
        assert_eq!(summary.parameter_count, 2);
        assert!(result.objf().is_finite());
        assert_eq!(result.population_parameters().len(), 2);
        assert_eq!(result.random_effect_names(), &["ke", "v"]);
        assert_eq!(result.omega().dim(), (2, 2));
        assert_eq!(result.residual_sigmas().len(), 1);
        assert_eq!(result.eta_chain_means().len(), 2);
        assert!(result.kappa_chain_means().is_empty());
        assert_eq!(result.conditional_modes().len(), 2);
        assert!(result
            .conditional_modes()
            .iter()
            .all(|mode| mode.objective.is_finite()));
        assert_eq!(result.population_summary().parameters.len(), 2);
        assert_eq!(result.individual_summaries().len(), 2);
    }

    // ─── Operational convergence tests ───────────────────────────────────

    #[test]
    fn operational_convergence_disabled_when_config_is_none() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        let markov = MarkovSimulationVarianceConfig::new(
            7,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            4,
            1024 * 1024,
        );
        let config = SaemConfig::new()
            .k1_iterations(2)
            .k2_iterations(2)
            .burn_in(0)
            .averaged_iterates(0.75)
            .markov_simulation_variance(markov)
            .compute_map(false)
            .seed(42);
        let result = problem().fit_with(config).unwrap();
        let ops = result.operational_diagnostics();
        assert!(ops.checks.is_empty());
        assert!(!ops.used_for_termination);
        assert!(!ops.final_check_reused);
        assert_eq!(result.termination_reason(), Some(&StopReason::MaxCycles));
    }

    #[test]
    fn operational_convergence_records_checkpoints_when_configured() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        let markov = MarkovSimulationVarianceConfig::new(
            7,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            4,
            1024 * 1024,
        );
        let oc = OperationalConvergenceConfig::literature_guided(1, 1, 1.0, 0.95, 0.1, 0.02);
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(3)
            .burn_in(0)
            .averaged_iterates(0.75)
            .markov_simulation_variance(markov)
            .covariance_stability(CovarianceStabilityConfig::new(0.01, 2))
            .operational_convergence(oc)
            .compute_map(false)
            .seed(43);
        let result = problem().fit_with(config).unwrap();
        let ops = result.operational_diagnostics();
        // Should have at least one checkpoint (smoothing phase produces checkpoints)
        assert!(!ops.checks.is_empty(), "expected at least one checkpoint");
        // Each checkpoint should have all fields populated
        for check in &ops.checks {
            assert!(check.checkpoint_seed.is_some());
            assert!(check.z_quantile.is_some());
            assert!(check.implied_minimum_ess.is_some());
            assert!(!check.criteria.is_empty());
            assert!(check.markov.is_some());
            assert_eq!(
                check.averaged_iterations,
                check.markov.as_ref().unwrap().n_avg
            );
        }
    }

    #[test]
    fn operational_convergence_has_exact_criterion_names() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        let markov = MarkovSimulationVarianceConfig::new(
            7,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            4,
            1024 * 1024,
        );
        let oc = OperationalConvergenceConfig::literature_guided(1, 1, 1.0, 0.95, 0.1, 0.02);
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(3)
            .burn_in(0)
            .averaged_iterates(0.75)
            .markov_simulation_variance(markov)
            .covariance_stability(CovarianceStabilityConfig::new(0.01, 2))
            .operational_convergence(oc)
            .compute_map(false)
            .seed(44);
        let result = problem().fit_with(config).unwrap();
        let ops = result.operational_diagnostics();
        assert!(!ops.checks.is_empty());
        let first_check = &ops.checks[0];
        let names: Vec<&str> = first_check
            .criteria
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert!(names.contains(&"max_rhat"));
        assert!(names.contains(&"min_bulk_ess"));
        assert!(names.contains(&"min_average_bulk_ess_per_split_chain"));
        assert!(names.contains(&"relative_fixed_width"));
        assert!(names.contains(&"newton_displacement"));
        assert!(names.contains(&"newton_displacement_mc_sd"));
        assert!(names.contains(&"omega_boundary_rejection_run"));
        assert!(names.contains(&"omega_iov_boundary_rejection_run"));
    }

    #[test]
    fn covariance_boundary_rejection_blocks_converged_stop_reason() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        let markov = MarkovSimulationVarianceConfig::new(
            7,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            4,
            1024 * 1024,
        );
        let oc = OperationalConvergenceConfig::literature_guided(1, 1, 1.0, 0.95, 100.0, 100.0);
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(2)
            .burn_in(0)
            .averaged_iterates(0.75)
            .markov_simulation_variance(markov)
            .covariance_stability(CovarianceStabilityConfig::new(0.99, 1))
            .operational_convergence(oc)
            .compute_map(false)
            .seed(47);
        let mut state = SaemState::from_problem(problem(), &config).unwrap();
        state.step().unwrap();
        state.cycle_diagnostics[0].omega_relative_spd_margin = Some(0.5);
        state.cycle_diagnostics[0].omega_update_rejected = true;

        state.step().unwrap();

        let check = state
            .operational_diagnostics
            .checks
            .last()
            .expect("operational checkpoint");
        let boundary = check
            .criteria
            .iter()
            .find(|criterion| criterion.name == "omega_boundary_rejection_run")
            .expect("Omega boundary criterion");
        assert_eq!(
            boundary.status,
            OperationalConvergenceCriterionStatus::NotSatisfied
        );
        assert_ne!(state.status, Status::Stop(StopReason::Converged));
    }

    #[test]
    fn iov_boundary_rejection_blocks_converged_stop_reason() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        let markov = MarkovSimulationVarianceConfig::new(
            7,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            4,
            1024 * 1024,
        );
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(2)
            .burn_in(0)
            .averaged_iterates(0.75)
            .markov_simulation_variance(markov)
            .covariance_stability(CovarianceStabilityConfig::new(0.99, 1))
            .operational_convergence(OperationalConvergenceConfig::literature_guided(
                1, 1, 1.0, 0.95, 100.0, 100.0,
            ))
            .compute_map(false)
            .seed(48);
        let mut state = SaemState::from_problem(iov_problem(), &config).unwrap();
        state.step().unwrap();
        state.cycle_diagnostics[0].omega_iov_relative_spd_margin = Some(0.5);
        state.cycle_diagnostics[0].omega_iov_update_rejected = true;
        state.step().unwrap();

        let check = state
            .operational_diagnostics
            .checks
            .last()
            .expect("operational checkpoint");
        let boundary = check
            .criteria
            .iter()
            .find(|criterion| criterion.name == "omega_iov_boundary_rejection_run")
            .expect("Omega_IOV boundary criterion");
        assert_eq!(
            boundary.status,
            OperationalConvergenceCriterionStatus::NotSatisfied
        );
        assert_ne!(state.status, Status::Stop(StopReason::Converged));
    }

    #[test]
    fn operational_convergence_waits_for_complete_covariance_window() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        let markov = MarkovSimulationVarianceConfig::new(
            7,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            4,
            1024 * 1024,
        );
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(5)
            .burn_in(0)
            .averaged_iterates(0.75)
            .markov_simulation_variance(markov)
            .covariance_stability(CovarianceStabilityConfig::new(0.01, 5))
            .operational_convergence(OperationalConvergenceConfig::literature_guided(
                1, 1, 1.0, 0.95, 100.0, 100.0,
            ))
            .compute_map(false)
            .seed(49);
        let mut state = SaemState::from_problem(problem(), &config).unwrap();
        state.step().unwrap();
        state.step().unwrap();

        let first = state
            .operational_diagnostics
            .checks
            .last()
            .expect("first operational checkpoint");
        let first_boundary = first
            .criteria
            .iter()
            .find(|criterion| criterion.name == "omega_boundary_rejection_run")
            .expect("Omega boundary criterion");
        assert!(matches!(
            first_boundary.status,
            OperationalConvergenceCriterionStatus::Unavailable(_)
        ));
        assert!(matches!(
            first.outcome,
            OperationalConvergenceOutcome::Ineligible { .. }
        ));
        assert_ne!(state.status, Status::Stop(StopReason::Converged));

        while state.cycle < 5 && !state.status.is_stop() {
            state.step().unwrap();
        }
        let eligible = state
            .operational_diagnostics
            .checks
            .last()
            .expect("fifth-cycle operational checkpoint");
        assert_eq!(eligible.iteration, 5);
        let eligible_boundary = eligible
            .criteria
            .iter()
            .find(|criterion| criterion.name == "omega_boundary_rejection_run")
            .expect("Omega boundary criterion");
        assert_eq!(
            eligible_boundary.status,
            OperationalConvergenceCriterionStatus::Satisfied
        );
    }

    #[test]
    fn operational_convergence_final_checkpoint_runs_once_with_truthful_flags() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        let markov = MarkovSimulationVarianceConfig::new(
            7,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            4,
            1024 * 1024,
        );
        // check_interval=1 means every smoothing iteration is a checkpoint,
        // so the last scheduled checkpoint and the mandatory final will overlap.
        let oc = OperationalConvergenceConfig::literature_guided(1, 1, 1.0, 0.95, 0.1, 0.02);
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(2)
            .burn_in(0)
            .averaged_iterates(0.75)
            .markov_simulation_variance(markov)
            .covariance_stability(CovarianceStabilityConfig::new(0.01, 2))
            .operational_convergence(oc)
            .compute_map(false)
            .seed(45);
        let result = problem().fit_with(config).unwrap();
        let ops = result.operational_diagnostics();
        assert!(!ops.final_check_reused);
        let final_check = ops.checks.last().expect("final checkpoint");
        assert!(final_check.scheduled);
        assert!(final_check.mandatory_final);
        assert_eq!(
            ops.checks
                .iter()
                .filter(|check| check.iteration == final_check.iteration)
                .count(),
            1
        );
    }

    #[test]
    fn operational_convergence_checkpoint_seed_is_deterministic_and_global_seed_is_unchanged() {
        use crate::algorithms::parametric::{LugsailConfig, MarkovSimulationVarianceConfig};
        let markov = MarkovSimulationVarianceConfig::new(
            7,
            0,
            12,
            6,
            LugsailConfig::over_lugsail_bartlett(),
            4,
            1024 * 1024,
        );
        let oc = OperationalConvergenceConfig::literature_guided(1, 1, 1.0, 0.95, 0.1, 0.02);
        let config = SaemConfig::new()
            .k1_iterations(1)
            .k2_iterations(3)
            .burn_in(0)
            .averaged_iterates(0.75)
            .markov_simulation_variance(markov)
            .covariance_stability(CovarianceStabilityConfig::new(0.01, 2))
            .operational_convergence(oc)
            .compute_map(false)
            .seed(46);
        let result1 = problem().fit_with(config.clone()).unwrap();
        let result2 = problem().fit_with(config).unwrap();

        let ops1 = result1.operational_diagnostics();
        let ops2 = result2.operational_diagnostics();
        assert_eq!(ops1.checks.len(), ops2.checks.len());
        for (c1, c2) in ops1.checks.iter().zip(ops2.checks.iter()) {
            assert_eq!(c1.checkpoint_seed, c2.checkpoint_seed);
            assert_eq!(c1.z_quantile, c2.z_quantile);
            assert_eq!(c1.outcome, c2.outcome);
        }
        // Canonical fit result must be unchanged by operational convergence
        assert_eq!(
            result1.population_parameters(),
            result2.population_parameters()
        );
        assert_eq!(result1.omega(), result2.omega());
        assert_eq!(result1.conditional_n2ll(), result2.conditional_n2ll());
    }

    #[test]
    fn normal_two_sided_z_covers_common_confidence_levels() {
        use statrs::distribution::{ContinuousCDF, Normal};
        let norm = Normal::new(0.0, 1.0).unwrap();
        for p in [0.90, 0.95, 0.99] {
            let expected = norm.inverse_cdf(p + (1.0 - p) / 2.0);
            let actual = normal_two_sided_z(p);
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn gong_flegal_fixed_width_and_implied_ess_are_exact() {
        let z = normal_two_sided_z(0.95);
        let epsilon = 0.05;
        let implied = 4.0 * z * z / (epsilon * epsilon);
        assert!((implied - 6146.34).abs() < 0.1);
        let boundary_fraction = epsilon / (2.0 * z);
        assert!(2.0 * z * boundary_fraction <= epsilon);
        assert!(2.0 * z * (boundary_fraction + 1e-12) > epsilon);
    }

    #[test]
    fn evaluate_criterion_detects_satisfied_not_satisfied_and_unavailable() {
        let satisfied = evaluate_criterion("test", Some(0.5), 1.0, |v| v <= 1.0);
        assert_eq!(
            satisfied.status,
            OperationalConvergenceCriterionStatus::Satisfied
        );
        assert_eq!(satisfied.observed, Some(0.5));

        let not_satisfied = evaluate_criterion("test", Some(2.0), 1.0, |v| v <= 1.0);
        assert_eq!(
            not_satisfied.status,
            OperationalConvergenceCriterionStatus::NotSatisfied
        );
        assert_eq!(not_satisfied.observed, Some(2.0));

        let unavailable_none = evaluate_criterion("test", None, 1.0, |v| v <= 1.0);
        assert!(matches!(
            unavailable_none.status,
            OperationalConvergenceCriterionStatus::Unavailable(_)
        ));
        assert_eq!(unavailable_none.observed, None);

        let unavailable_nan = evaluate_criterion("test", Some(f64::NAN), 1.0, |v| v <= 1.0);
        assert!(matches!(
            unavailable_nan.status,
            OperationalConvergenceCriterionStatus::Unavailable(_)
        ));
    }

    #[test]
    fn newton_displacement_requires_matching_dimensions() {
        let empty_info = InformationDiagnostics {
            coordinates: vec![],
            recursion_cycles: 0,
            delta: vec![],
            g: vec![],
            expected_complete_hessian: vec![],
            observed_hessian: vec![],
            observed_information: vec![],
            status: InformationStatus::Available,
        };
        let empty_markov = MarkovSimulationVarianceDiagnostics::disabled();
        assert_eq!(newton_displacement(&empty_info, &empty_markov), None);
        assert_eq!(newton_displacement_mc_sd(&empty_info, &empty_markov), None);
    }
}
