mod assembler;
mod compiler;
mod effects;
mod engine;
mod individual;
mod integration;
mod likelihood;
mod population;
mod posthoc;
mod predictions;
mod reporting;
mod sampling;
mod state;
mod statistics;
mod sufficient_stats;
mod summaries;
mod transforms;
mod uncertainty;
mod workspace;

pub(crate) use assembler::{
    assemble_focei_result, assemble_saem_result, FoceiResultInput, SaemResultInput,
};
pub use compiler::compile_model_state;
pub(crate) use effects::{
    blended_subject_covariate_m_step, covariance_from_individual_etas,
    covariance_from_subject_means, covariate_state, estimate_beta, occasion_covariate_maps,
    recenter_individual_estimates, subject_covariate_maps, subject_mean_phi,
};
pub use engine::{fit, ParametricEngine};
pub use individual::{Individual, IndividualEstimates};
pub(crate) use integration::{
    ImportanceSamplingConfig, ImportanceSamplingEstimator, SubjectConditionalPosterior,
};
pub use likelihood::{
    approximate_objective_from_individuals, batch_log_likelihood_from_eta,
    estimate_initial_sigma_sq, importance_sampling_likelihood_estimates,
    log_priors_from_eta_matrix, subject_conditionals_from_eta_samples, subject_log_prior_from_eta,
    subject_objective_from_eta, sync_error_models_with_sigma,
    update_residual_error_from_individuals, ResidualErrorUpdate,
};
pub(crate) use population::ensure_positive_definite_covariance;
pub use population::{CovarianceStructure, Population};
pub(crate) use posthoc::saem_posthoc_likelihood;
pub use posthoc::{aic, bic, cache_predictions, shrinkage, statistics, write_statistics};
pub use predictions::{ParametricPredictionRow, ParametricPredictions, PredictionSummary};
pub use reporting::{FimMethod, LikelihoodEstimates, ParametricIterationLog, UncertaintyEstimates};
pub(crate) use sampling::{sample_eta_from_population, ChainState, KernelConfig};
pub use state::{
    CovariateEffectsSnapshot, CovariateState, EtaTable, EtaVector, FixedEffects,
    IndividualEffectsState, KappaVector, OccasionKappa, OccasionKappaTable, ParametricModelState,
    ParametricTransformKind, PhiTable, PhiVector, PsiTable, PsiVector, RandomEffects,
    ResidualState, TransformSet,
};
pub use statistics::{ParametricStatistics, ResidualErrorEstimates};
pub use sufficient_stats::{StepSizeSchedule, SufficientStats};
pub use summaries::{fit_summary, individual_summaries, population_summary};
pub(crate) use transforms::initialize_population_in_phi_space;
pub use transforms::{
    default_phi_variance, phi_to_psi, phi_to_psi_vec, psi_to_phi, psi_to_phi_vec, transform_label,
    transforms_from_saemix_codes, ParameterTransform,
};
pub use uncertainty::{
    estimates as uncertainty_estimates, fim, fim_method, has_fim, has_standard_errors, se_mu,
};
pub use workspace::ParametricWorkspace;
