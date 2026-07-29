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
    CompleteDerivative, InformationLayout, InformationRecursion,
};
use crate::estimation::parametric::marginal_likelihood::{
    calculate_population_marginal_likelihood, unavailable_population_marginal_likelihood,
    MarginalLikelihoodDiagnostics, MarginalLikelihoodFailureReason, MarginalSubject,
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
use crate::estimation::parametric::sufficient::{
    CovariateSufficientStatistics, PhiSufficientStatistics,
};
use crate::estimation::parametric::{CovarianceUpdateStatus, ResolvedOmega};
use crate::estimation::{EstimationProblem, Parametric, ParametricErrorModels};
use crate::model::{ParameterScale, UnboundedParameter};
use crate::ResidualErrorModel;

use crate::results::{
    CovarianceCycleUpdateDiagnostics, CovarianceCycleUpdateOutcome,
    CovarianceUpdateNotAttemptedReason, DiagnosticTraceCoordinate, InformationCoordinateKind,
    InformationDiagnostics, InformationStatus, MarkovSimulationVarianceChainDiagnostics,
    MarkovSimulationVarianceDiagnostics, MarkovSimulationVarianceStatus, OccasionKappaEstimate,
    OperationalConvergenceCheck, OperationalConvergenceCriterion,
    OperationalConvergenceCriterionStatus, OperationalConvergenceDiagnostics,
    OperationalConvergenceOutcome, ParametricWarning, RankDiagnosticStatus, RankMixingDiagnostic,
    RankMixingDiagnostics, ResidualCycleDiagnostics, ResidualErrorEstimate, SaemCycleDiagnostics,
    SaemEstimatorMetadata, SaemPhase, SubjectConditionalMode,
};

use super::{
    CovarianceStabilityConfig, NumericalFailure, OperationalConvergenceConfig, SaemConfig,
    SaemEstimatorPolicy,
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
                &problem.error_models,
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

mod state;

pub(crate) use state::SaemState;
