use anyhow::{bail, Context};
use ndarray::Array2;
use pharmsol::simulator::prediction::SubjectPredictions;
use pharmsol::{Data, Equation, Event, Subject};
use serde::{Deserialize, Serialize};

use crate::estimation::parametric::{
    covariates::{
        CovariateEffect, CovariateEstimate, CovariateModel, CovariateMstepError,
        SubjectCovariateDesign, SubjectCovariateValue, SubjectPopulationParameters,
    },
    individual::{occasion_psi, occasion_psi_from_subject_mean, population_phi},
    marginal_likelihood::{MarginalLikelihoodDiagnostics, MarginalLikelihoodStatus},
    transforms::{phi_to_psi_derivative, psi_to_phi},
    ConditionalCurvatureDiagnostics, ShrinkageDiagnostics,
};
use crate::model::ParameterScale;
use crate::ResidualErrorModel;

use crate::algorithms::parametric::{
    MarkovSimulationVarianceConfig, OperationalConvergenceConfig, SaemConfig, SaemEstimatorPolicy,
};
use crate::algorithms::StopReason;
use crate::estimation::nonparametric::NonParametricResult;
use crate::results::{
    FitSummary, IndividualSummary, InformationCriteriaDiagnostics, ParameterSummary,
    PopulationSummary,
};

/// A shared trait for the output of any estimation algorithm.
pub trait FitResult {
    fn objf(&self) -> f64;
    fn converged(&self) -> bool;
    fn summary(&self) -> FitSummary;
    fn population_summary(&self) -> PopulationSummary;
    fn individual_summaries(&self) -> Vec<IndividualSummary>;
}

/// SAEM schedule phase associated with an immutable cycle diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SaemPhase {
    BurnIn,
    Exploration,
    Smoothing,
}

/// Per-output residual M-step diagnostics for one SAEM cycle.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResidualCycleDiagnostics {
    pub output: String,
    pub output_index: usize,
    pub prediction_evaluation_count: usize,
    pub proportional_floor_count: usize,
    pub non_finite_prediction_count: usize,
    pub exponential_domain_violation_count: usize,
    pub update_rejected: bool,
    pub optimizer_objective: Option<f64>,
    pub optimizer_converged: Option<bool>,
    pub optimizer_iterations: Option<u64>,
    pub optimizer_termination: Option<String>,
    pub combined_additive_collapse_warning: bool,
}

/// Why a covariance update was not attempted during a cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CovarianceUpdateNotAttemptedReason {
    BurnIn,
    UpdateInactive,
    NoEstimatedEntries,
    NotConfigured,
}

/// Terminal reason for rejecting a covariance proposal before or after trial steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CovarianceUpdateRejectionReason {
    CandidateNotFiniteSymmetric,
    CurrentObjectiveUnavailable,
    ConstrainedSolveFailed,
    BacktrackingExhausted,
}

/// Reason one deterministic covariance trial was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CovarianceTrialRejectionReason {
    VarianceFloorInfeasible,
    NotPositiveDefinite,
    ObjectiveUnavailable,
    ObjectiveIncrease,
}

/// Terminal classification of one cycle's covariance update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CovarianceCycleUpdateOutcome {
    NotAttempted {
        reason: CovarianceUpdateNotAttemptedReason,
    },
    Accepted,
    NoOp,
    Rejected {
        reason: CovarianceUpdateRejectionReason,
    },
}

/// Complete proposal and acceptance diagnostics for one covariance M-step.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CovarianceCycleUpdateDiagnostics {
    /// Coherent covariance second-moment proposal supplied to the updater.
    pub proposal: Option<Array2<f64>>,
    /// Mask-aware solved target after any capped-path variance floor.
    pub solved_target: Option<Array2<f64>>,
    pub outcome: CovarianceCycleUpdateOutcome,
    /// Accepted interpolation fraction; absent when no trial was accepted.
    pub accepted_fraction: Option<f64>,
    /// Fractions evaluated in deterministic order.
    pub attempted_fractions: Vec<f64>,
    /// One entry for every rejected trial, aligned with the corresponding
    /// prefix of `attempted_fractions` before an accepted trial, if any.
    pub trial_rejections: Vec<CovarianceTrialRejectionReason>,
}

impl CovarianceCycleUpdateDiagnostics {
    pub(crate) fn not_attempted(reason: CovarianceUpdateNotAttemptedReason) -> Self {
        Self {
            proposal: None,
            solved_target: None,
            outcome: CovarianceCycleUpdateOutcome::NotAttempted { reason },
            accepted_fraction: None,
            attempted_fractions: Vec::new(),
            trial_rejections: Vec::new(),
        }
    }
}

/// MCMC, covariance, and residual diagnostics captured after one complete SAEM cycle.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SaemCycleDiagnostics {
    pub iteration: usize,
    pub phase: SaemPhase,
    pub stochastic_approximation_step: f64,
    pub covariance_step: f64,
    pub eta_proposals: usize,
    pub eta_accepted: usize,
    pub eta_rejected: usize,
    pub eta_non_finite: usize,
    pub eta_parameter_acceptance_rates: Vec<f64>,
    pub eta_proposal_step_sizes_before_adaptation: Vec<f64>,
    pub eta_proposal_step_sizes_after_adaptation: Vec<f64>,
    pub eta_block_proposals: usize,
    pub eta_block_accepted: usize,
    pub eta_block_rejected: usize,
    pub eta_block_non_finite: usize,
    pub eta_block_subject_acceptance_rates: Vec<f64>,
    pub eta_block_step_sizes_before_adaptation: Vec<f64>,
    pub eta_block_step_sizes_after_adaptation: Vec<f64>,
    pub kappa_proposals: usize,
    pub kappa_accepted: usize,
    pub kappa_rejected: usize,
    pub kappa_non_finite: usize,
    pub kappa_subject_acceptance_rates: Vec<f64>,
    pub kappa_proposal_step_sizes_before_adaptation: Vec<f64>,
    pub kappa_proposal_step_sizes_after_adaptation: Vec<f64>,
    pub simulated_annealing_active: bool,
    pub population_parameters: Vec<f64>,
    pub omega: Array2<f64>,
    pub omega_iov: Option<Array2<f64>>,
    pub residual_error_estimates: Vec<ResidualErrorEstimate>,
    pub residual_diagnostics: Vec<ResidualCycleDiagnostics>,
    pub conditional_negative_log_likelihood: f64,
    pub eta_log_prior: f64,
    pub kappa_log_prior: f64,
    pub omega_update_rejected: bool,
    pub omega_iov_update_rejected: bool,
    /// Detailed Ω proposal and acceptance record for this cycle.
    pub omega_update: CovarianceCycleUpdateDiagnostics,
    /// Detailed Ω_IOV proposal and acceptance record for this cycle.
    pub omega_iov_update: CovarianceCycleUpdateDiagnostics,
    /// Dimensionless generalized SPD margin relative to the declared initial Ω.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub omega_relative_spd_margin: Option<f64>,
    /// Dimensionless generalized SPD margin relative to the declared initial Ω_IOV.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub omega_iov_relative_spd_margin: Option<f64>,
    /// Current covariate coefficient values (beta) after this cycle's M-step.
    ///
    /// Present only when a covariate model was declared. The vector is in
    /// canonical declaration order and includes both estimated and fixed
    /// coefficients.
    pub covariate_betas: Option<Vec<f64>>,
    /// Fixed/free status aligned with `covariate_betas`.
    pub covariate_beta_estimated: Option<Vec<bool>>,
}

impl SaemCycleDiagnostics {
    /// Residual M-step diagnostics for a named model output in this cycle.
    pub fn residual_diagnostic(&self, output: &str) -> Option<&ResidualCycleDiagnostics> {
        self.residual_diagnostics
            .iter()
            .find(|diagnostic| diagnostic.output == output)
    }
}

/// Named final residual-error estimate for one model output.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResidualErrorEstimate {
    pub output: String,
    pub output_index: usize,
    pub model: ResidualErrorModel,
    pub estimated: bool,
    /// Additive-component status for ordinary or correlated combined models.
    pub combined_additive_estimated: Option<bool>,
    /// Proportional-component status for ordinary or correlated combined models.
    pub combined_proportional_estimated: Option<bool>,
    /// Correlation-component status for the correlated-combined model.
    pub correlation_estimated: Option<bool>,
}

/// Structured warning aggregated from immutable parametric cycle diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParametricWarning {
    OmegaUpdateRejected {
        first_iteration: usize,
        cycles: usize,
    },
    OmegaIovUpdateRejected {
        first_iteration: usize,
        cycles: usize,
    },
    OmegaBoundaryRejection {
        first_iteration: usize,
        longest_run: usize,
    },
    OmegaIovBoundaryRejection {
        first_iteration: usize,
        longest_run: usize,
    },
    EtaNonFiniteProposals {
        first_iteration: usize,
        count: usize,
    },
    EtaBlockNonFiniteProposals {
        first_iteration: usize,
        count: usize,
    },
    KappaNonFiniteProposals {
        first_iteration: usize,
        count: usize,
    },
    ResidualUpdateRejected {
        output: String,
        first_iteration: usize,
        cycles: usize,
    },
    ProportionalPredictionFloor {
        output: String,
        first_iteration: usize,
        count: usize,
    },
    NonFiniteResidualPrediction {
        output: String,
        first_iteration: usize,
        count: usize,
    },
    ExponentialDomainViolation {
        output: String,
        first_iteration: usize,
        count: usize,
    },
    CombinedAdditiveCollapse {
        output: String,
        first_iteration: usize,
        cycles: usize,
    },
    ResidualOptimizerNotConverged {
        output: String,
        first_iteration: usize,
        cycles: usize,
    },
    MarginalLikelihoodUnavailable {
        subjects: Vec<String>,
    },
    MarginalLikelihoodNonconvergedModes {
        subjects: Vec<String>,
    },
}

/// Kind and exact source indices of one free information coordinate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InformationCoordinateKind {
    Population {
        parameter_index: usize,
    },
    CovariateEffect {
        effect_index: usize,
    },
    Omega {
        row: usize,
        column: usize,
    },
    OmegaIov {
        row: usize,
        column: usize,
    },
    Residual {
        output_index: usize,
        component: String,
    },
}

/// One deterministic free coordinate used by complete-data derivatives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InformationCoordinate {
    pub index: usize,
    pub name: String,
    pub kind: InformationCoordinateKind,
}

/// Scientific and numerical availability of the observed-information diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "reason", rename_all = "snake_case")]
pub enum InformationStatus {
    Available,
    NoFreeCoordinates,
    NonFinite,
    ObservedInformationNotPositiveDefinite,
    Unsupported(String),
    Ineligible(String),
}

/// Immutable complete-data score/Hessian stochastic-approximation diagnostics.
///
/// This is diagnostic only. The persistent-MCMC assumptions required for an
/// inferential interpretation have not been verified. It is the source for the
/// separately classified population uncertainty result, not a marginal-likelihood
/// result or evidence of convergence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InformationDiagnostics {
    pub coordinates: Vec<InformationCoordinate>,
    pub recursion_cycles: usize,
    pub delta: Vec<f64>,
    pub g: Vec<Vec<f64>>,
    pub expected_complete_hessian: Vec<Vec<f64>>,
    pub observed_hessian: Vec<Vec<f64>>,
    pub observed_information: Vec<Vec<f64>>,
    pub status: InformationStatus,
}

/// Explicit classification of why population uncertainty is unavailable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", content = "detail", rename_all = "snake_case")]
pub enum PopulationUncertaintyUnavailableReason {
    SourceUnavailable(InformationStatus),
    NonFinite,
    ObservedInformationNotPositiveDefinite,
    InversionFailed,
}

/// Top-level status of a population uncertainty derivation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "reason", rename_all = "snake_case")]
pub enum PopulationUncertaintyStatus {
    Available,
    Unavailable(PopulationUncertaintyUnavailableReason),
}

/// Regularization applied while deriving population uncertainty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PopulationUncertaintyRegularization {
    None,
}

/// Free-coordinate population uncertainty derived from the observed information.
///
/// The covariance and standard errors are in estimation (φ) space. Natural-scale
/// (ψ) parameter standard errors are computed separately via the first-order delta
/// method. No confidence intervals or stronger inferential claims are provided; the
/// required persistent-MCMC assumptions have not been verified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopulationUncertaintyDiagnostics {
    /// Canonical free-coordinate indices, names, and kinds.
    pub coordinates: Vec<InformationCoordinate>,
    /// Unregularized free-coordinate observed-information inverse (φ-space covariance).
    /// Rows are coordinates in the same order as [`Self::coordinates`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_covariance: Option<Vec<Vec<f64>>>,
    /// Diagonal square-root of [`Self::free_covariance`] (φ-space SEs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_standard_errors: Option<Vec<f64>>,
    /// Spectral condition number (λ_max / λ_min) of the observed-information matrix.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spectral_condition_number: Option<f64>,
    /// Derivation outcome.
    pub status: PopulationUncertaintyStatus,
    /// Explicit regularization classification (always [`PopulationUncertaintyRegularization::None`]).
    pub regularization: PopulationUncertaintyRegularization,
}

impl PopulationUncertaintyDiagnostics {
    pub(crate) fn unavailable(reason: PopulationUncertaintyUnavailableReason) -> Self {
        Self {
            coordinates: Vec::new(),
            free_covariance: None,
            free_standard_errors: None,
            spectral_condition_number: None,
            status: PopulationUncertaintyStatus::Unavailable(reason),
            regularization: PopulationUncertaintyRegularization::None,
        }
    }
}

/// Scientific/numerical status of the optional frozen-kernel diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "detail", rename_all = "snake_case")]
pub enum MarkovSimulationVarianceStatus {
    Disabled,
    AverageNotApplied,
    NoFreeCoordinates,
    ExactZeroNoLatentState,
    InformationUnavailable(String),
    InvalidConfiguration(String),
    /// Checked trace-memory accounting overflowed `usize` before allocation.
    TraceMemoryAccountingOverflow,
    CoordinateMismatch,
    UnsupportedScore(String),
    NonFinite,
    NonSymmetric,
    Indefinite,
    StuckChain {
        chain: usize,
    },
    /// Finite algebra was produced, but stationarity, mixing, and the Markov
    /// Poisson-equation/CLT assumptions were not verified.
    AssumptionsUnverified,
}

/// Batch-means output and movement accounting for one independent chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkovSimulationVarianceChainDiagnostics {
    pub chain: usize,
    pub bm_batch: Vec<Vec<f64>>,
    pub bm_batch_over_r: Vec<Vec<f64>>,
    pub lugsail_lrv: Vec<Vec<f64>>,
    /// Classification of this chain's raw lugsail LRV.
    pub status: MarkovSimulationVarianceStatus,
    /// Retained-transition proposals only; diagnostic warmup is excluded.
    pub proposals: usize,
    /// Retained-transition accepts only; diagnostic warmup is excluded.
    pub accepts: usize,
    /// Retained-transition actual state changes only; diagnostic warmup is excluded.
    pub state_changes: usize,
}

/// Diagnostic-only frozen-kernel simulation variance for the Cesaro estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkovSimulationVarianceDiagnostics {
    pub config: Option<MarkovSimulationVarianceConfig>,
    pub coordinates: Vec<InformationCoordinate>,
    pub chain_count: usize,
    pub n_avg: usize,
    pub chains: Vec<MarkovSimulationVarianceChainDiagnostics>,
    /// Grand complete-score mean across all retained independent diagnostic draws.
    pub grand_score_mean: Vec<f64>,
    pub lambda: Vec<Vec<f64>>,
    pub lambda_status: MarkovSimulationVarianceStatus,
    pub xi: Vec<Vec<f64>>,
    pub xi_status: MarkovSimulationVarianceStatus,
    pub simulation_covariance: Vec<Vec<f64>>,
    pub simulation_covariance_status: MarkovSimulationVarianceStatus,
    pub status: MarkovSimulationVarianceStatus,
    pub assumptions: String,
    /// Rank/mixing convergence diagnostics from independent prior-drawn chains.
    /// Always present; status indicates availability.
    pub rank_diagnostics: RankMixingDiagnostics,
}

impl MarkovSimulationVarianceDiagnostics {
    pub(crate) fn disabled() -> Self {
        Self {
            config: None,
            coordinates: Vec::new(),
            chain_count: 0,
            n_avg: 0,
            chains: Vec::new(),
            grand_score_mean: Vec::new(),
            lambda: Vec::new(),
            lambda_status: MarkovSimulationVarianceStatus::Disabled,
            xi: Vec::new(),
            xi_status: MarkovSimulationVarianceStatus::Disabled,
            simulation_covariance: Vec::new(),
            simulation_covariance_status: MarkovSimulationVarianceStatus::Disabled,
            status: MarkovSimulationVarianceStatus::Disabled,
            assumptions: "diagnostic disabled; no stationarity, mixing, Poisson-equation, or Markov-CLT claim".into(),
            rank_diagnostics: RankMixingDiagnostics {
                diagnostic_chains: 0,
                draws_per_chain: 0,
                original_chains: 0,
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
                assumptions: String::new(),
                status: RankDiagnosticStatus::Disabled,
            },
        }
    }
}

/// Immutable status of a rank/mixing diagnostic coordinate or aggregate.
///
/// Variants carry explicit reasons; no silent fallback hides the cause.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "detail", rename_all = "snake_case")]
pub enum RankDiagnosticStatus {
    Disabled,
    /// The requested diagnostic has no latent coordinates to trace.
    NoLatent,
    /// A complete-score coordinate could not be evaluated; latent traces remain independent.
    ScoreUnavailable,
    Unavailable,
    /// Some coordinate diagnostics are valid while others are unavailable.
    PartialAvailability,
    NoChains,
    /// Fewer than two diagnostic chains: split-R̂ requires ≥ 2 chains.
    TooFewChains,
    /// Chains did not retain the same number of draws.
    UnequalChainLengths,
    /// Too few retained draws for the requested statistic.
    TooFewDraws,
    /// The retained draw count cannot be split evenly.
    OddDraws,
    /// max_trace_bytes would be exceeded by the accounted trace-memory peak.
    TraceByteCapExceeded,
    /// Checked trace-memory accounting overflowed `usize` before allocation.
    TraceMemoryAccountingOverflow,
    /// At least one draw or derived folded value is non-finite.
    NonFiniteDraws,
    /// The coordinate is constant (W = 0): R̂/ESS is undefined.
    ConstantDraws,
    /// A required variance was invalid.
    InvalidVariance,
    /// Integrated autocorrelation time τ ≤ 0: ESS undefined.
    NonPositiveTau,
    /// Valid rank diagnostics were computed; no convergence claim is made.
    Available,
}

/// Per-coordinate trace metadata: identifies the quantity being traced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum DiagnosticTraceCoordinate {
    /// One coordinate of the score vector (information coordinate).
    Score {
        index: usize,
        name: String,
        kind: InformationCoordinateKind,
    },
    /// One η element for a specific subject and random effect.
    Eta {
        subject: String,
        effect_index: usize,
        effect_name: String,
    },
    /// One κ element for a specific subject, occasion, and random effect.
    Kappa {
        subject: String,
        occasion_index: usize,
        effect_index: usize,
        effect_name: String,
    },
}

/// Per-coordinate rank/mixing diagnostic with split-R̂ and bulk ESS.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankMixingDiagnostic {
    /// Metadata identifying the traced coordinate.
    pub trace: DiagnosticTraceCoordinate,
    /// Rank-normalized split-R̂, or None if inapplicable.
    pub rank_rhat: Option<f64>,
    /// Exact status of the rank-normalized split-R̂ computation.
    pub rank_rhat_status: RankDiagnosticStatus,
    /// Folded split-R̂, or None if inapplicable.
    pub folded_rhat: Option<f64>,
    /// Exact status of the folded split-R̂ computation.
    pub folded_rhat_status: RankDiagnosticStatus,
    /// Maximum of rank and folded split-R̂, present only when both components succeed.
    pub max_rhat: Option<f64>,
    /// Derived status of the maximum; available only when both component R̂ values succeed.
    pub max_rhat_status: RankDiagnosticStatus,
    /// Bulk effective sample size (total across all split chains).
    pub bulk_ess: Option<f64>,
    /// Exact status of the bulk ESS computation.
    pub bulk_ess_status: RankDiagnosticStatus,
    /// Average ESS per split chain = bulk_ess / (2 × diagnostic_chains).
    pub avg_ess_per_split_chain: Option<f64>,
    /// Integrated autocorrelation time τ.
    pub tau: Option<f64>,
    /// Per-coordinate status.
    pub status: RankDiagnosticStatus,
}

/// Aggregate rank/mixing convergence diagnostics from independent prior-drawn chains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankMixingDiagnostics {
    /// Number of diagnostic chains drawn from the installed prior (Cd).
    pub diagnostic_chains: usize,
    /// Number of draws retained per diagnostic chain (post-warmup).
    pub draws_per_chain: usize,
    /// Number of original fit chains (Cf), for LRV scaling.
    pub original_chains: usize,
    /// Per-coordinate mixing diagnostics.
    pub traces: Vec<RankMixingDiagnostic>,
    /// Index-preserving per-chain LRV matrices. A failed configured chain is
    /// retained as `None` at its original diagnostic-chain index.
    pub lrv_per_chain: Vec<Option<Vec<Vec<f64>>>>,
    /// Per-chain LRV matrix status classification.
    pub lrv_chain_statuses: Vec<RankDiagnosticStatus>,
    /// Diagnostic-mean LRV: Σ(LRV_i) / Cd² (full matrix).
    pub diagnostic_mean_lrv: Option<Vec<Vec<f64>>>,
    /// Operational LRV: Σ(LRV_i) / (Cd × Cf) (full matrix); used for Xi/n_avg.
    pub operational_lrv: Option<Vec<Vec<f64>>>,
    /// Configured maximum trace storage bytes (from config).
    pub max_trace_bytes: usize,
    /// Conservative deterministic upper bound on simultaneously requested
    /// trace-buffer bytes, including nested `Vec` headers/capacities and the
    /// largest score/rank transient workspace.
    pub accounted_peak_trace_bytes_required: usize,
    /// Accounted trace-buffer peak reached; zero when rejected before trace
    /// allocation. This is requested-capacity accounting, not allocator metadata.
    pub accounted_peak_trace_bytes_used: usize,
    /// Worst complete maximum of rank-normalized and folded split-R̂ across
    /// coordinates that have both valid component values.
    pub worst_rhat: Option<f64>,
    /// Minimum bulk ESS across all coordinates.
    pub min_bulk_ess: Option<f64>,
    /// Minimum average ESS per split chain across all coordinates.
    pub min_avg_ess_per_split_chain: Option<f64>,
    /// Assumptions text recorded at diagnostic construction.
    pub assumptions: String,
    /// Aggregate status: the worst status across all coordinates.
    pub status: RankDiagnosticStatus,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SaemEstimatorMetadata {
    pub policy: SaemEstimatorPolicy,
    pub average_applied: bool,
    pub averaging_start_cycle: Option<usize>,
    pub averaged_iterations: usize,
}

impl Default for SaemEstimatorMetadata {
    fn default() -> Self {
        Self {
            policy: SaemEstimatorPolicy::TerminalIterate,
            average_applied: false,
            averaging_start_cycle: None,
            averaged_iterations: 0,
        }
    }
}

/// Final-chain posterior mean of a subject-level η vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubjectEtaEstimate {
    pub subject_id: String,
    pub values: Vec<f64>,
}

/// Final-chain posterior mean of an occasion-level κ vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OccasionKappaEstimate {
    pub subject_id: String,
    pub occasion_index: usize,
    pub values: Vec<f64>,
}

/// Joint posthoc conditional mode for one subject.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubjectConditionalMode {
    pub subject_id: String,
    pub eta: Vec<f64>,
    pub kappas: Vec<OccasionKappaEstimate>,
    pub parameters: Vec<f64>,
    pub objective: f64,
    pub converged: bool,
    pub iterations: u64,
    pub termination: String,
    /// One strict joint eta/kappa curvature, reused by opt-in importance sampling.
    pub uncertainty: ConditionalCurvatureDiagnostics,
}

#[derive(Debug, Clone)]
pub struct ParametricResult<E: Equation> {
    pub(crate) equation: E,
    pub(crate) data: Data,
    pub(crate) config: SaemConfig,
    pub(crate) effective_n_chains: usize,
    pub(crate) objective_function: f64,
    pub(crate) converged: bool,
    pub(crate) termination_reason: Option<StopReason>,
    pub(crate) iterations: usize,
    pub(crate) subject_count: usize,
    pub(crate) observation_count: usize,
    pub(crate) parameter_names: Vec<String>,
    pub(crate) parameter_scales: Vec<ParameterScale>,
    pub(crate) estimated_parameters: Vec<bool>,
    pub(crate) population_initial: Vec<f64>,
    pub(crate) population_estimates: Vec<f64>,
    pub(crate) random_effect_indices: Vec<usize>,
    pub(crate) random_effect_names: Vec<String>,
    pub(crate) omega: Array2<f64>,
    pub(crate) omega_structural_mask: Array2<bool>,
    pub(crate) omega_estimated_mask: Array2<bool>,
    pub(crate) omega_initial: Array2<f64>,
    pub(crate) iov_effect_indices: Vec<usize>,
    pub(crate) iov_effect_names: Vec<String>,
    pub(crate) omega_iov: Option<Array2<f64>>,
    pub(crate) omega_iov_structural_mask: Option<Array2<bool>>,
    pub(crate) omega_iov_estimated_mask: Option<Array2<bool>>,
    pub(crate) omega_iov_initial: Option<Array2<f64>>,
    pub(crate) residual_sigmas: Vec<f64>,
    pub(crate) residual_error_estimates: Vec<ResidualErrorEstimate>,
    pub(crate) residual_initial_values: Vec<Vec<f64>>,
    pub(crate) residual_initial_estimated: Vec<Vec<bool>>,
    pub(crate) eta_chain_means: Vec<SubjectEtaEstimate>,
    pub(crate) kappa_chain_means: Vec<OccasionKappaEstimate>,
    pub(crate) conditional_modes: Vec<SubjectConditionalMode>,
    pub(crate) shrinkage: ShrinkageDiagnostics,
    pub(crate) cycle_diagnostics: Vec<SaemCycleDiagnostics>,
    pub(crate) warnings: Vec<ParametricWarning>,
    pub(crate) information_diagnostics: InformationDiagnostics,
    pub(crate) population_uncertainty: PopulationUncertaintyDiagnostics,
    pub(crate) markov_simulation_variance: MarkovSimulationVarianceDiagnostics,
    pub(crate) operational_diagnostics: OperationalConvergenceDiagnostics,
    pub(crate) marginal_likelihood: Option<MarginalLikelihoodDiagnostics>,
    pub(crate) information_criteria: InformationCriteriaDiagnostics,
    pub(crate) estimator_metadata: SaemEstimatorMetadata,
    pub(crate) individual_estimates: Vec<(String, Vec<f64>)>,
    pub(crate) covariate_model: Option<CovariateModel>,
}

impl<E: Equation> ParametricResult<E> {
    /// Structural model retained for prediction and follow-up runs.
    pub fn equation(&self) -> &E {
        &self.equation
    }

    /// Original estimation dataset.
    pub fn data(&self) -> &Data {
        &self.data
    }

    /// Original validated requested SAEM configuration.
    ///
    /// In particular, [`SaemConfig::n_chains`] remains the requested chain
    /// count and is not replaced by the effective auto-scaled count.
    pub fn config(&self) -> &SaemConfig {
        &self.config
    }

    /// Effective chain count used by the fit after automatic scaling.
    pub fn effective_n_chains(&self) -> usize {
        self.effective_n_chains
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Complete-data observed-information diagnostic accumulated during SAEM.
    ///
    /// This does not provide standard errors, a marginal likelihood, or
    /// convergence evidence. Persistent-MCMC assumptions remain unverified.
    pub fn information_diagnostics(&self) -> &InformationDiagnostics {
        &self.information_diagnostics
    }

    /// Free-coordinate population uncertainty from the observed information.
    ///
    /// Returns covariance, standard errors, and spectral condition number in
    /// estimation (φ) space. Natural-scale (ψ) parameter standard errors are
    /// available through [`PopulationSummary`] when uncertainty is available.
    pub fn population_uncertainty(&self) -> &PopulationUncertaintyDiagnostics {
        &self.population_uncertainty
    }

    /// Optional frozen-kernel Markov diagnostics, including LRV simulation-
    /// variance matrices and rank-normalized/folded mixing diagnostics.
    ///
    /// These diagnostics remain non-proof: they do not establish stationarity,
    /// mathematical convergence, model correctness, or valid uncertainty.
    pub fn markov_simulation_variance(&self) -> &MarkovSimulationVarianceDiagnostics {
        &self.markov_simulation_variance
    }

    pub fn operational_convergence(&self) -> &OperationalConvergenceDiagnostics {
        &self.operational_diagnostics
    }

    /// Immutable operational convergence lifecycle diagnostics.
    ///
    /// Empty when no operational convergence criteria were configured.
    pub fn operational_diagnostics(&self) -> &OperationalConvergenceDiagnostics {
        &self.operational_diagnostics
    }

    /// Rank-normalized/folded split-Rhat and bulk-ESS mixing diagnostics.
    ///
    /// These detect unreliable frozen diagnostic chains but do not prove
    /// convergence or alter the fit termination reason.
    pub fn rank_mixing_diagnostics(&self) -> &RankMixingDiagnostics {
        &self.markov_simulation_variance.rank_diagnostics
    }

    /// Final-estimate policy and smoothing-average application details.
    pub fn estimator_metadata(&self) -> &SaemEstimatorMetadata {
        &self.estimator_metadata
    }

    /// Why the parametric fit stopped.
    ///
    /// A completed fixed SAEM schedule reports [`StopReason::MaxCycles`]; this
    /// must not be interpreted as statistical convergence.
    pub fn termination_reason(&self) -> Option<&StopReason> {
        self.termination_reason.as_ref()
    }

    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    pub fn parameter_scales(&self) -> &[ParameterScale] {
        &self.parameter_scales
    }

    pub fn estimated_parameters(&self) -> &[bool] {
        &self.estimated_parameters
    }

    pub fn population_parameters(&self) -> &[f64] {
        &self.population_estimates
    }

    /// Generate predictions at the final population parameters.
    ///
    /// The retained data are cloned and expanded for this call; predictions are
    /// not cached on the result.
    pub fn population_predictions(
        &self,
        idelta: f64,
        tad: f64,
    ) -> anyhow::Result<Vec<SubjectPredictions>>
    where
        E: pharmsol::equation::EquationTypes<P = SubjectPredictions>,
    {
        self.validate_prediction_metadata()?;
        let expanded = self.data.clone().expand(idelta, tad);
        self.validate_expanded_subjects(&expanded)?;
        let population_phi = self
            .covariate_model
            .as_ref()
            .map(|_| population_phi(&self.population_estimates, &self.parameter_scales))
            .transpose()?;
        let subject_population = self
            .covariate_model
            .as_ref()
            .zip(population_phi.as_ref())
            .map(|(model, phi)| model.subject_population_parameters(phi, &self.parameter_scales))
            .transpose()?;
        expanded
            .subjects()
            .iter()
            .enumerate()
            .map(|(subject_index, subject)| {
                let parameters = subject_population
                    .as_ref()
                    .map(|rows| rows[subject_index].psi())
                    .unwrap_or(&self.population_estimates);
                self.equation
                    .estimate_predictions_dense(subject, parameters)
                    .with_context(|| {
                        format!(
                            "population prediction failed for subject '{}'",
                            subject.id()
                        )
                    })
            })
            .collect()
    }

    /// Generate predictions at each subject's final posthoc conditional mode.
    ///
    /// For IOV fits, each retained occasion is simulated independently using
    /// its reconstructed eta/kappa parameter vector.
    pub fn conditional_predictions(
        &self,
        idelta: f64,
        tad: f64,
    ) -> anyhow::Result<Vec<SubjectPredictions>>
    where
        E: pharmsol::equation::EquationTypes<P = SubjectPredictions>,
    {
        self.validate_prediction_metadata()?;
        if self.random_effect_indices.is_empty() && self.iov_effect_indices.is_empty() {
            return self.population_predictions(idelta, tad);
        }
        if self.conditional_modes.is_empty() {
            bail!(
                "conditional predictions require conditional modes; rerun with compute_map(true)"
            );
        }

        let expanded = self.data.clone().expand(idelta, tad);
        self.validate_expanded_subjects(&expanded)?;
        if self.conditional_modes.len() != expanded.subjects().len() {
            bail!(
                "conditional mode count {} does not match subject count {}",
                self.conditional_modes.len(),
                expanded.subjects().len()
            );
        }

        let population_phi = self
            .covariate_model
            .as_ref()
            .map(|_| population_phi(&self.population_estimates, &self.parameter_scales))
            .transpose()?;
        let subject_population = self
            .covariate_model
            .as_ref()
            .zip(population_phi.as_ref())
            .map(|(model, phi)| model.subject_population_parameters(phi, &self.parameter_scales))
            .transpose()?;
        expanded
            .subjects()
            .iter()
            .zip(&self.conditional_modes)
            .enumerate()
            .map(|(subject_index, (subject, mode))| {
                if mode.subject_id != *subject.id() {
                    bail!(
                        "conditional mode subject '{}' does not match retained subject '{}'",
                        mode.subject_id,
                        subject.id()
                    );
                }
                if mode.parameters.len() != self.population_estimates.len() {
                    bail!(
                        "conditional mode for subject '{}' has parameter width {} but expected {}",
                        subject.id(),
                        mode.parameters.len(),
                        self.population_estimates.len()
                    );
                }
                if mode.eta.len() != self.random_effect_indices.len() {
                    bail!(
                        "conditional mode for subject '{}' has eta width {} but expected {}",
                        subject.id(),
                        mode.eta.len(),
                        self.random_effect_indices.len()
                    );
                }

                if self.iov_effect_indices.is_empty() {
                    if !mode.kappas.is_empty() {
                        bail!(
                            "conditional mode for non-IOV subject '{}' unexpectedly has kappas",
                            subject.id()
                        );
                    }
                    return self
                        .equation
                        .estimate_predictions_dense(subject, &mode.parameters)
                        .with_context(|| {
                            format!(
                                "conditional prediction failed for subject '{}'",
                                subject.id()
                            )
                        });
                }

                if mode.kappas.len() != subject.occasions().len() {
                    bail!(
                        "conditional mode for subject '{}' has {} occasions but retained data have {}",
                        subject.id(),
                        mode.kappas.len(),
                        subject.occasions().len()
                    );
                }
                let mut combined = Vec::new();
                for (occasion, kappa) in subject.occasions().iter().zip(&mode.kappas) {
                    if kappa.subject_id != *subject.id() {
                        bail!(
                            "conditional kappa subject '{}' does not match retained subject '{}'",
                            kappa.subject_id,
                            subject.id()
                        );
                    }
                    if kappa.occasion_index != occasion.index() {
                        bail!(
                            "conditional kappa occasion {} does not match retained occasion {} for subject '{}'",
                            kappa.occasion_index,
                            occasion.index(),
                            subject.id()
                        );
                    }
                    if kappa.values.len() != self.iov_effect_indices.len() {
                        bail!(
                            "conditional kappa for subject '{}' occasion {} has width {} but expected {}",
                            subject.id(),
                            occasion.index(),
                            kappa.values.len(),
                            self.iov_effect_indices.len()
                        );
                    }
                    let parameters = match subject_population.as_ref() {
                        Some(rows) => occasion_psi_from_subject_mean(
                            rows[subject_index].phi(),
                            &self.parameter_scales,
                            &self.random_effect_indices,
                            &mode.eta,
                            &self.iov_effect_indices,
                            &kappa.values,
                        ),
                        None => occasion_psi(
                            &self.population_estimates,
                            &self.parameter_scales,
                            &self.random_effect_indices,
                            &mode.eta,
                            &self.iov_effect_indices,
                            &kappa.values,
                        ),
                    }?;
                    let occasion_subject =
                        Subject::from_occasions(subject.id().clone(), vec![occasion.clone()]);
                    let predictions = self
                        .equation
                        .estimate_predictions_dense(&occasion_subject, &parameters)
                        .with_context(|| {
                            format!(
                                "conditional prediction failed for subject '{}' occasion {}",
                                subject.id(),
                                occasion.index()
                            )
                        })?;
                    let expected_points = occasion
                        .events()
                        .iter()
                        .filter(|event| matches!(event, Event::Observation(_)))
                        .count();
                    if predictions.predictions().len() != expected_points {
                        bail!(
                            "conditional prediction shape mismatch for subject '{}' occasion {}: generated {} points but expected {}",
                            subject.id(),
                            occasion.index(),
                            predictions.predictions().len(),
                            expected_points
                        );
                    }
                    for mut prediction in predictions.predictions().iter().cloned() {
                        *prediction.mut_occasion() = occasion.index();
                        combined.push(prediction);
                    }
                }
                Ok(SubjectPredictions::from(combined))
            })
            .collect()
    }

    fn validate_prediction_metadata(&self) -> anyhow::Result<()> {
        if self.population_estimates.len() != self.parameter_names.len() {
            bail!(
                "population parameter width {} does not match parameter-name width {}",
                self.population_estimates.len(),
                self.parameter_names.len()
            );
        }
        if self.parameter_scales.len() != self.population_estimates.len() {
            bail!(
                "parameter-scale width {} does not match population parameter width {}",
                self.parameter_scales.len(),
                self.population_estimates.len()
            );
        }
        Ok(())
    }

    fn validate_expanded_subjects(&self, expanded: &Data) -> anyhow::Result<()> {
        if expanded.subjects().len() != self.data.subjects().len() {
            bail!(
                "expanded subject count {} does not match retained subject count {}",
                expanded.subjects().len(),
                self.data.subjects().len()
            );
        }
        for (retained, expanded) in self.data.subjects().iter().zip(expanded.subjects()) {
            if retained.id() != expanded.id() {
                bail!(
                    "expanded subject '{}' does not match retained subject '{}'",
                    expanded.id(),
                    retained.id()
                );
            }
        }
        Ok(())
    }

    pub fn random_effect_indices(&self) -> &[usize] {
        &self.random_effect_indices
    }

    pub fn random_effect_names(&self) -> &[String] {
        &self.random_effect_names
    }

    pub fn omega(&self) -> &Array2<f64> {
        &self.omega
    }

    pub fn omega_structural_mask(&self) -> &Array2<bool> {
        &self.omega_structural_mask
    }

    pub fn omega_estimated_mask(&self) -> &Array2<bool> {
        &self.omega_estimated_mask
    }

    pub fn iov_effect_indices(&self) -> &[usize] {
        &self.iov_effect_indices
    }

    pub fn iov_effect_names(&self) -> &[String] {
        &self.iov_effect_names
    }

    pub fn omega_iov(&self) -> Option<&Array2<f64>> {
        self.omega_iov.as_ref()
    }

    pub fn omega_iov_structural_mask(&self) -> Option<&Array2<bool>> {
        self.omega_iov_structural_mask.as_ref()
    }

    pub fn omega_iov_estimated_mask(&self) -> Option<&Array2<bool>> {
        self.omega_iov_estimated_mask.as_ref()
    }

    /// Legacy positional primary sigma values.
    ///
    /// Prefer [`Self::residual_error_estimates`] when output names, complete
    /// error-model parameters, or fixed/estimated status are needed.
    pub fn residual_sigmas(&self) -> &[f64] {
        &self.residual_sigmas
    }

    /// Named residual-error models and their estimation status.
    pub fn residual_error_estimates(&self) -> &[ResidualErrorEstimate] {
        &self.residual_error_estimates
    }

    /// Final residual-error estimate for a named model output.
    pub fn residual_error_estimate(&self, output: &str) -> Option<&ResidualErrorEstimate> {
        self.residual_error_estimates
            .iter()
            .find(|estimate| estimate.output == output)
    }

    /// Final conditional negative log-likelihood at the retained latent chains.
    ///
    /// This is not a population marginal likelihood or a validated marginal OFV.
    pub fn conditional_negative_log_likelihood(&self) -> f64 {
        self.objective_function / 2.0
    }

    /// Twice the final conditional negative log-likelihood.
    ///
    /// This is the value currently returned through [`FitResult::objf`] for API
    /// compatibility; it must not be interpreted as a marginal population OFV.
    pub fn conditional_n2ll(&self) -> f64 {
        self.objective_function
    }

    /// Complete post-fit marginal-likelihood diagnostics, when requested.
    pub fn marginal_likelihood_diagnostics(&self) -> Option<&MarginalLikelihoodDiagnostics> {
        self.marginal_likelihood.as_ref()
    }

    /// Population marginal log likelihood, absent when disabled or unavailable.
    pub fn marginal_log_likelihood(&self) -> Option<f64> {
        self.marginal_likelihood
            .as_ref()
            .and_then(|diagnostics| diagnostics.log_marginal_likelihood)
    }

    /// Population marginal negative twice log likelihood, absent when unavailable.
    pub fn marginal_n2ll(&self) -> Option<f64> {
        self.marginal_likelihood
            .as_ref()
            .and_then(|diagnostics| diagnostics.n2ll)
    }

    /// Delta-method Monte Carlo standard error of the marginal N2LL.
    pub fn marginal_n2ll_mcse(&self) -> Option<f64> {
        self.marginal_likelihood
            .as_ref()
            .and_then(|diagnostics| diagnostics.n2ll_mcse)
    }

    /// Typed population marginal-likelihood status, when requested.
    pub fn marginal_likelihood_status(&self) -> Option<&MarginalLikelihoodStatus> {
        self.marginal_likelihood
            .as_ref()
            .map(|diagnostics| &diagnostics.status)
    }

    /// Immutable AIC/BIC diagnostics derived only from population marginal N2LL.
    pub fn information_criteria(&self) -> &InformationCriteriaDiagnostics {
        &self.information_criteria
    }

    /// Akaike information criterion, absent unless marginal N2LL is available.
    pub fn aic(&self) -> Option<f64> {
        self.information_criteria.aic
    }

    /// Bayesian information criterion using independent subjects as sample size.
    pub fn bic(&self) -> Option<f64> {
        self.information_criteria.bic
    }

    /// AIC Monte Carlo standard error, exactly the source marginal N2LL MCSE.
    pub fn aic_mcse(&self) -> Option<f64> {
        self.information_criteria.aic_mcse
    }

    /// BIC Monte Carlo standard error, exactly the source marginal N2LL MCSE.
    pub fn bic_mcse(&self) -> Option<f64> {
        self.information_criteria.bic_mcse
    }

    /// Number of free population-level coordinates used in both penalties.
    pub fn free_parameter_count(&self) -> usize {
        self.information_criteria.parameter_count.total
    }

    /// Subject-level η values averaged across final SAEM chains.
    ///
    /// These are posterior chain summaries, not posthoc conditional-mode EBEs.
    pub fn eta_chain_means(&self) -> &[SubjectEtaEstimate] {
        &self.eta_chain_means
    }

    /// Final-chain η mean for a named subject.
    pub fn eta_chain_mean(&self, subject_id: &str) -> Option<&SubjectEtaEstimate> {
        self.eta_chain_means
            .iter()
            .find(|estimate| estimate.subject_id == subject_id)
    }

    /// Occasion-level κ values averaged across final SAEM chains.
    ///
    /// These are posterior chain summaries, not posthoc conditional-mode EBEs.
    pub fn kappa_chain_means(&self) -> &[OccasionKappaEstimate] {
        &self.kappa_chain_means
    }

    /// Final-chain κ mean for a named subject and occasion index.
    pub fn kappa_chain_mean(
        &self,
        subject_id: &str,
        occasion_index: usize,
    ) -> Option<&OccasionKappaEstimate> {
        self.kappa_chain_means.iter().find(|estimate| {
            estimate.subject_id == subject_id && estimate.occasion_index == occasion_index
        })
    }

    /// Joint η/κ posthoc conditional modes under the final population fit.
    pub fn conditional_modes(&self) -> &[SubjectConditionalMode] {
        &self.conditional_modes
    }

    /// Joint posthoc conditional mode for a named subject.
    pub fn conditional_mode(&self, subject_id: &str) -> Option<&SubjectConditionalMode> {
        self.conditional_modes
            .iter()
            .find(|mode| mode.subject_id == subject_id)
    }

    /// Source-explicit eta/kappa posterior-mean and MAP shrinkage diagnostics.
    pub fn shrinkage(&self) -> &ShrinkageDiagnostics {
        &self.shrinkage
    }

    /// Immutable per-cycle SAEM MCMC and covariance diagnostics.
    pub fn cycle_diagnostics(&self) -> &[SaemCycleDiagnostics] {
        &self.cycle_diagnostics
    }

    /// Structured fit warnings aggregated from cycle-level diagnostics.
    pub fn warnings(&self) -> &[ParametricWarning] {
        &self.warnings
    }

    // ─── Covariate result accessors ─────────────────────────────────────

    /// Fully validated subject-static covariate population model, when declared.
    pub fn covariates(&self) -> Option<&CovariateModel> {
        self.covariate_model.as_ref()
    }

    /// Ordered covariate-effect declarations.
    pub fn covariate_declarations(&self) -> Option<&[CovariateEffect]> {
        self.covariate_model
            .as_ref()
            .map(|model| model.declarations())
    }

    /// Ordered covariate coefficient estimates.
    pub fn covariate_estimates(&self) -> Option<&[CovariateEstimate]> {
        self.covariate_model.as_ref().map(|model| model.estimates())
    }

    /// Subject-level covariate values extracted from the estimation dataset.
    pub fn covariate_subject_values(&self) -> Option<&[SubjectCovariateValue]> {
        self.covariate_model
            .as_ref()
            .map(|model| model.subject_values())
    }

    /// Subject-level design-matrix rows used by the population M-step.
    pub fn covariate_subject_design(&self) -> Option<&[SubjectCovariateDesign]> {
        self.covariate_model
            .as_ref()
            .map(|model| model.subject_design())
    }

    /// Subject-specific transformed (phi) and execution-space (psi) population
    /// parameters incorporating covariate effects.
    ///
    /// This derives from [`CovariateModel::subject_population_parameters`]
    /// using the retained population estimates and parameter scales.
    pub fn covariate_subject_population_parameters(
        &self,
    ) -> Result<Option<Vec<SubjectPopulationParameters>>, CovariateMstepError> {
        match self.covariate_model.as_ref() {
            Some(model) => {
                let phi = population_phi(&self.population_estimates, &self.parameter_scales)
                    .map_err(|_error| CovariateMstepError::NonFiniteSolution)?;
                model
                    .subject_population_parameters(&phi, &self.parameter_scales)
                    .map(Some)
            }
            None => Ok(None),
        }
    }
}

impl<E: Equation> FitResult for ParametricResult<E> {
    fn objf(&self) -> f64 {
        self.objective_function
    }

    fn converged(&self) -> bool {
        self.converged
    }

    fn summary(&self) -> FitSummary {
        FitSummary {
            objective_function: self.objective_function,
            converged: self.converged,
            iterations: self.iterations,
            subject_count: self.subject_count,
            observation_count: self.observation_count,
            parameter_count: self.parameter_names.len(),
            marginal_log_likelihood: self.marginal_log_likelihood(),
            marginal_n2ll: self.marginal_n2ll(),
            marginal_n2ll_mcse: self.marginal_n2ll_mcse(),
            marginal_likelihood_status: self.marginal_likelihood_status().cloned(),
            information_criteria: Some(self.information_criteria.clone()),
        }
    }

    fn population_summary(&self) -> PopulationSummary {
        let free_standard_errors: Option<&[f64]> = match &self.population_uncertainty.status {
            PopulationUncertaintyStatus::Available => {
                self.population_uncertainty.free_standard_errors.as_deref()
            }
            _ => None,
        };
        let parameters: Vec<ParameterSummary> = self
            .parameter_names
            .iter()
            .enumerate()
            .map(|(param_index, name)| {
                let estimate = self.population_estimates[param_index];
                let scale = self.parameter_scales[param_index];
                let is_estimated = self.estimated_parameters[param_index];
                let (sd, cv_percent) = if is_estimated {
                    if let Some(ses) = free_standard_errors {
                        // Find the population coordinate matching this parameter.
                        let coordinate_se = self
                            .population_uncertainty
                            .coordinates
                            .iter()
                            .find_map(|coord| match &coord.kind {
                                InformationCoordinateKind::Population { parameter_index }
                                    if *parameter_index == param_index =>
                                {
                                    ses.get(coord.index)
                                }
                                _ => None,
                            })
                            .copied();
                        if let Some(phi_se) = coordinate_se {
                            let phi = psi_to_phi(estimate, scale);
                            let abs_deriv = phi_to_psi_derivative(phi, scale).abs();
                            let psi_sd = phi_se * abs_deriv;
                            let sd = psi_sd.is_finite().then_some(psi_sd);
                            let cv_percent = sd.and_then(|sd| {
                                if estimate.is_finite() && estimate != 0.0 {
                                    let cv = 100.0 * sd / estimate.abs();
                                    cv.is_finite().then_some(cv)
                                } else {
                                    None
                                }
                            });
                            (sd, cv_percent)
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };
                ParameterSummary {
                    name: name.clone(),
                    estimate,
                    mean: None,
                    median: None,
                    sd,
                    cv_percent,
                }
            })
            .collect();
        PopulationSummary {
            parameters,
            information_criteria: Some(self.information_criteria.clone()),
            population_uncertainty: Some(self.population_uncertainty.clone()),
            shrinkage: Some(self.shrinkage.clone()),
        }
    }

    fn individual_summaries(&self) -> Vec<IndividualSummary> {
        self.individual_estimates
            .iter()
            .map(|(id, estimates)| IndividualSummary {
                id: id.clone(),
                parameter_names: self.parameter_names.clone(),
                estimates: estimates.clone(),
                standard_errors: None,
                conditional_uncertainty: self
                    .conditional_mode(id)
                    .map(|mode| mode.uncertainty.clone()),
            })
            .collect()
    }
}

use crate::estimation::nonparametric;

impl<E: Equation> FitResult for NonParametricResult<E> {
    fn objf(&self) -> f64 {
        self.objf()
    }

    fn converged(&self) -> bool {
        self.converged()
    }

    fn summary(&self) -> FitSummary {
        nonparametric::fit_summary(self)
    }

    fn population_summary(&self) -> PopulationSummary {
        nonparametric::population_summary(self)
    }

    fn individual_summaries(&self) -> Vec<IndividualSummary> {
        nonparametric::individual_summaries(self)
    }
}

// ─── Operational convergence result types ────────────────────────────────

/// Exact evaluation status of one operational convergence criterion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "detail", rename_all = "snake_case")]
pub enum OperationalConvergenceCriterionStatus {
    Satisfied,
    NotSatisfied,
    Unavailable(String),
}

/// One evaluated operational convergence criterion with its retained values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalConvergenceCriterion {
    /// Short deterministic criterion name.
    pub name: String,
    /// Observed value, present only when the criterion could be evaluated.
    pub observed: Option<f64>,
    /// Configured threshold against which the observed value is compared.
    pub threshold: f64,
    /// Per-criterion evaluation status.
    pub status: OperationalConvergenceCriterionStatus,
}

/// Joint outcome of one operational convergence checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", content = "detail", rename_all = "snake_case")]
pub enum OperationalConvergenceOutcome {
    Passed,
    Failed { criteria: Vec<String> },
    Ineligible { reasons: Vec<String> },
}

/// Immutable record of one operational convergence checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalConvergenceCheck {
    /// SAEM cycle at which the checkpoint was evaluated.
    pub iteration: usize,
    /// Number of averaged iterates contributing to the candidate.
    pub averaged_iterations: usize,
    /// Whether this checkpoint was triggered by the periodic schedule.
    pub scheduled: bool,
    /// Whether this checkpoint was the mandatory final evaluation.
    pub mandatory_final: bool,
    /// Deterministic per-checkpoint diagnostic seed; `None` when no frozen
    /// diagnostic configuration exists or the candidate was unavailable.
    pub checkpoint_seed: Option<u64>,
    /// Two-sided standard normal quantile for the configured confidence level.
    pub z_quantile: Option<f64>,
    /// Gong/Flegal implied minimum ESS `4*z²/epsilon²`; metadata only and
    /// never a substitute for rank ESS.
    pub implied_minimum_ess: Option<f64>,
    /// Averaged-candidate values in deterministic free-coordinate order;
    /// empty when the candidate coordinate mapping was unavailable.
    pub candidate_free_coordinates: Vec<f64>,
    /// Exact observed-information snapshot used by this checkpoint.
    pub information: Option<InformationDiagnostics>,
    /// Ordered per-criterion evaluations.
    pub criteria: Vec<OperationalConvergenceCriterion>,
    /// Joint checkpoint outcome.
    pub outcome: OperationalConvergenceOutcome,
    /// Immutable frozen-kernel diagnostics evaluated at the averaged
    /// candidate; `None` when the candidate itself was unavailable.
    pub markov: Option<MarkovSimulationVarianceDiagnostics>,
}

impl OperationalConvergenceCheck {
    /// Per-trace rank/mixing diagnostics collected across all coordinates
    /// of the frozen-kernel evaluation (if available).
    pub fn per_trace_diagnostics(&self) -> &[RankMixingDiagnostic] {
        self.markov
            .as_ref()
            .map(|markov| markov.rank_diagnostics.traces.as_slice())
            .unwrap_or(&[])
    }

    /// Worst rank-normalized/folded split-Rhat across all coordinates
    /// (if available).
    pub fn worst_rhat(&self) -> Option<f64> {
        self.markov
            .as_ref()
            .and_then(|markov| markov.rank_diagnostics.worst_rhat)
    }

    /// Minimum bulk ESS across all coordinates (if available).
    pub fn min_bulk_ess(&self) -> Option<f64> {
        self.markov
            .as_ref()
            .and_then(|markov| markov.rank_diagnostics.min_bulk_ess)
    }
}

/// Immutable operational convergence lifecycle record for one fit.
///
/// Retains every scheduled and mandatory-final checkpoint evaluation,
/// accompanied by exact policy-wording summary diagnostics. When no
/// operational convergence configuration was present, `checks` is empty and
/// `used_for_termination` is false.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct OperationalConvergenceDiagnostics {
    pub config: Option<OperationalConvergenceConfig>,
    /// Ordered checkpoint evaluations, one per scheduled or mandatory-final
    /// check. Empty when no criteria were configured or no checkpoint was
    /// reachable.
    pub checks: Vec<OperationalConvergenceCheck>,
    /// Whether the last mandatory final checkpoint reused an identical
    /// earlier check (defensive caching marker).
    pub final_check_reused: bool,
    /// Whether any checkpoint passed and supplied the final stop reason.
    pub used_for_termination: bool,
    pub final_status: Option<OperationalConvergenceOutcome>,
    pub worst_rhat: Option<f64>,
    pub min_bulk_ess: Option<f64>,
    pub fixed_width_ratio: Option<f64>,
    pub fixed_width_epsilon: Option<f64>,
    pub implied_minimum_ess: Option<f64>,
    pub newton_displacement: Option<f64>,
    pub newton_displacement_mc_sd: Option<f64>,
}

impl OperationalConvergenceDiagnostics {
    /// Exact policy-wording summary for this diagnostics record.
    ///
    /// Returns one or more lines describing the lifecycle state. The wording
    /// is public API and must not be rewritten without updating external
    /// consumers.
    pub fn warnings(&self) -> Vec<String> {
        match (&self.config, &self.final_status, self.used_for_termination) {
            (_, Some(OperationalConvergenceOutcome::Passed), true) => vec![
                "PMcore operational convergence criteria passed; this is not proof of mathematical convergence, stationarity, model correctness, or valid uncertainty; run an independent doubled-budget fit."
                    .to_string(),
            ],
            (Some(_), Some(OperationalConvergenceOutcome::Failed { .. }), false) => vec![
                "PMcore operational convergence criteria were evaluated but not satisfied; finite schedule completion remains MaxCycles and operational convergence was not established."
                    .to_string(),
            ],
            (Some(_), Some(OperationalConvergenceOutcome::Ineligible { .. }), false) => vec![
                "PMcore operational convergence criteria were evaluated but were ineligible; finite schedule completion remains MaxCycles and operational convergence was not established."
                    .to_string(),
            ],
            (Some(_), None, false) => vec![
                "PMcore operational convergence was configured, but no checkpoint was evaluated; operational convergence was not evaluated or established."
                    .to_string(),
            ],
            _ => Vec::new(),
        }
    }
}
