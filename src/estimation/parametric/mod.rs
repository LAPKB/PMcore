//! Shared parametric-estimation utilities.
//!
//! Algorithm implementations such as SAEM, FO, FOCE, and FOCEI should use these
//! modules for parameter transforms, covariance algebra, and η/Ω posterior
//! scoring instead of keeping algorithm-local copies.

pub(crate) mod conditional_uncertainty;
pub(crate) mod covariance;
pub mod covariates;
pub(crate) mod individual;
pub(crate) mod information;
pub mod marginal_likelihood;
pub(crate) mod markov_variance;
pub(crate) mod posterior;
pub(crate) mod posthoc;
pub(crate) mod prior;
pub(crate) mod rank_diagnostics;
pub(crate) mod residual;
pub(crate) mod shrinkage;
pub(crate) mod sufficient;
pub(crate) mod transforms;

pub use conditional_uncertainty::{
    ConditionalCurvatureAvailability, ConditionalCurvatureDiagnostics,
    ConditionalCurvatureRegularization, ConditionalCurvatureStatus,
    ConditionalCurvatureUnavailableReason, ConditionalModeMetadata, JointLatentCoordinate,
    JointLatentCoordinateKind,
};
pub use covariates::{
    rebase_eta, reject_constraints, solve_covariate_gls, subject_centered_omega, CovariateEffect,
    CovariateEffectFamily, CovariateEstimate, CovariateGlsProblem, CovariateModel,
    CovariateMstepError, CovariateValidationError, ParametricConstraint, SubjectCovariateDesign,
    SubjectCovariateValue, SubjectPopulationParameters,
};
pub use marginal_likelihood::{
    marginal_likelihood_subject_seed, MarginalLikelihoodConfig, MarginalLikelihoodDiagnostics,
    MarginalLikelihoodFailureReason, MarginalLikelihoodMethod, MarginalLikelihoodProposal,
    MarginalLikelihoodStatus, MarginalLikelihoodSubjectFailure, ProposalScaleSource,
    SubjectMarginalLikelihoodDiagnostics, N2_SEED_DOMAIN,
};
pub(crate) use prior::{CovarianceUpdateStatus, ResolvedOmega};
pub use prior::{Iov, Omega, ParametricPrior};
pub use shrinkage::{
    EtaMapShrinkage, EtaPosteriorMeanShrinkage, KappaMapShrinkage, KappaPosteriorMeanShrinkage,
    ShrinkageDiagnostics, ShrinkageUnavailableReason, ShrinkageValue,
};
