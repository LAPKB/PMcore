//! Population pharmacokinetic estimation algorithms and result types.
//!
//! PMcore provides nonparametric algorithms, SAEM for deterministic analytical
//! and ODE models, and explicit SDE particle filtering. Models and data are
//! supplied through [pharmsol], configured with [`estimation::EstimationProblem`],
//! and run with an algorithm from [`algorithms`]. Generic SDE fitting through
//! [`estimation::EstimationProblem`] is unsupported.

/// Provides the various algorithms used within the framework
pub mod algorithms;

/// Estimation problems and the non-parametric and parametric fitting families.
pub mod estimation;

/// Model-domain types: parameters, parameter spaces, and metadata.
pub mod model;

/// Result and summary types shared across algorithms.
pub mod results;

/// Logging utilities.
pub mod logs;

// Re-export commonly used items
pub use anyhow::Result;
pub use std::collections::HashMap;

#[allow(deprecated)]
pub use estimation::{
    AssayErrorModel, AssayErrorModels, AssayLikelihoodError, BoundAssayErrorModels,
    ConditionalCurvatureDiagnostics, ConditionalCurvatureRegularization,
    ConditionalCurvatureStatus, ConditionalCurvatureUnavailableReason, ConditionalModeMetadata,
    CovariateEffect, CovariateEffectFamily, CovariateEstimate, CovariateGlsProblem, CovariateModel,
    CovariateMstepError, CovariateValidationError, ErrorModel, ErrorModelError, ErrorPoly,
    EtaMapShrinkage, EtaPosteriorMeanShrinkage, Factor, JointLatentCoordinate,
    JointLatentCoordinateKind, KappaMapShrinkage, KappaPosteriorMeanShrinkage,
    MarginalLikelihoodConfig, MarginalLikelihoodDiagnostics, MarginalLikelihoodFailureReason,
    MarginalLikelihoodMethod, MarginalLikelihoodProposal, MarginalLikelihoodStatus,
    MarginalLikelihoodSubjectFailure, NormalDistributionError, ParametricConstraint,
    ProposalScaleSource, ResidualErrorModel, ResidualErrorModels, SdeParticleConfig,
    SdeParticleError, SdeParticleFilter, SdeParticleRecord, SdeParticleResult,
    ShrinkageDiagnostics, ShrinkageUnavailableReason, ShrinkageValue, SubjectCovariateDesign,
    SubjectCovariateValue, SubjectPopulationParameters,
};

/// Dose optimization and forecasting (BestDose).
pub mod bestdose;

/// SDE-based Inter-Occasion Variability optimization.
pub mod iov;

/// A collection of commonly used items to simplify imports.
pub mod prelude {
    pub use super::logs::Logger;
    pub use super::HashMap;
    pub use super::Result;
    pub use crate::algorithms;
    pub use crate::algorithms::Algorithm;

    pub use crate::estimation::NonParametric;
    pub use crate::estimation::Parametric;
    #[allow(deprecated)]
    pub use crate::estimation::{
        AssayErrorModel, AssayErrorModels, AssayLikelihoodError, BoundAssayErrorModels,
        ConditionalCurvatureDiagnostics, ConditionalCurvatureRegularization,
        ConditionalCurvatureStatus, ConditionalCurvatureUnavailableReason, ConditionalModeMetadata,
        CovarianceStabilityConfig, CovariateEffect, CovariateEffectFamily, CovariateEstimate,
        CovariateGlsProblem, CovariateModel, CovariateMstepError, CovariateValidationError,
        ErrorModel, ErrorModelError, ErrorModels, ErrorPoly, EstimationProblem, EtaMapShrinkage,
        EtaPosteriorMeanShrinkage, Factor, FitProgress, Iov, JointLatentCoordinate,
        JointLatentCoordinateKind, KappaMapShrinkage, KappaPosteriorMeanShrinkage, LugsailConfig,
        MarginalLikelihoodConfig, MarginalLikelihoodDiagnostics, MarginalLikelihoodFailureReason,
        MarginalLikelihoodMethod, MarginalLikelihoodProposal, MarginalLikelihoodStatus,
        MarginalLikelihoodSubjectFailure, MarkovSimulationVarianceConfig, NcnpagConfig,
        NonParametricAlgorithm, NonparametricCycleProgress, NormalDistributionError, NpagConfig,
        NpmapConfig, NpodConfig, Omega, OperationalConvergenceConfig, ParametricAlgorithm,
        ParametricConstraint, ParametricErrorModel, ParametricErrorModels, ParametricPrior,
        ProposalScaleSource, ResidualErrorModel, ResidualErrorModels, SaemConfig,
        SaemEstimatorPolicy, SdeParticleConfig, SdeParticleError, SdeParticleFilter,
        SdeParticleRecord, SdeParticleResult, ShrinkageDiagnostics, ShrinkageUnavailableReason,
        ShrinkageValue, SubjectCovariateDesign, SubjectCovariateValue,
        SubjectMarginalLikelihoodDiagnostics, SubjectPopulationParameters,
    };

    pub use crate::model::parameter_space::{
        BoundedParameter, Parameter, ParameterScale, ParameterSpace, UnboundedParameter,
    };

    pub use crate::algorithms::nonparametric::{CycleFlow, FitController, FitObserver};
    pub use crate::algorithms::parametric::{
        CycleFlow as ParametricCycleFlow, FitController as ParametricFitController,
        FitObserver as ParametricFitObserver, NumericalFailure, NumericalFailurePhase,
        ParametricFitSnapshot,
    };
    pub use crate::estimation::nonparametric::{
        CycleLog, NPCycle, NPPredictions, NonParametricResult, Posterior, Psi, Theta, Weights,
    };
    pub use crate::iov::{DiffusionConfig, DiffusionOptimize, DiffusionResult};
    pub use crate::model::{EquationMetadataSource, ModelMetadata};
    pub use crate::results::{
        CovarianceCycleUpdateDiagnostics, CovarianceCycleUpdateOutcome,
        CovarianceTrialRejectionReason, CovarianceUpdateNotAttemptedReason,
        CovarianceUpdateRejectionReason, DiagnosticTraceCoordinate, FitResult, FitSummary,
        IndividualEffectRow, IndividualParameterRow, IndividualSummary,
        InformationCriteriaDiagnostics, InformationCriteriaParameterCount, InformationCriteriaRow,
        InformationCriteriaSampleSizeConvention, InformationCriteriaStatus,
        InformationCriteriaUnavailableReason, IterationRow, MarginalLikelihoodRow,
        MarkovSimulationVarianceChainDiagnostics, MarkovSimulationVarianceDiagnostics,
        MarkovSimulationVarianceStatus, OccasionKappaEstimate, OmegaRow,
        OperationalConvergenceCheck, OperationalConvergenceCriterion,
        OperationalConvergenceCriterionStatus, OperationalConvergenceDiagnostics,
        OperationalConvergenceOutcome, ParameterSummary, ParametricResult, ParametricResultRecord,
        ParametricResultTables, ParametricSourceCovariance, ParametricSourceEffect,
        ParametricSourceMetadata, ParametricSourceParameter, ParametricSourceResidual,
        ParametricWarning, ParametricWarningRecord, PopulationParameterRow, PopulationSummary,
        PopulationUncertaintyDiagnostics, PopulationUncertaintyRegularization,
        PopulationUncertaintyStatus, PopulationUncertaintyUnavailableReason, PredictionRow,
        RankDiagnosticStatus, RankMixingDiagnostic, RankMixingDiagnostics,
        ResidualCycleDiagnostics, ResidualErrorEstimate, ResidualErrorRow, SaemCycleDiagnostics,
        SaemEstimatorMetadata, SaemPhase, StatisticRow, SubjectConditionalMode, SubjectEtaEstimate,
    };

    // pharmsol: re-export the crate itself and its curated prelude.
    pub use pharmsol;
    pub use pharmsol::prelude::*;

    // Items required by downstream code that are not part of `pharmsol::prelude`.
    pub use pharmsol::equation::{EquationTypes, Predictions};
    pub use pharmsol::optimize::effect::get_e2;
    pub use pharmsol::{ODE, SDE};

    // Organized submodules mirroring pharmsol's grouping.
    pub mod simulator {
        pub use pharmsol::prelude::simulator::*;
    }
    pub mod data {
        pub use pharmsol::prelude::data::*;
    }
    pub mod models {
        pub use pharmsol::prelude::models::*;
    }

    // macros
    pub use pharmsol::{fa, fetch_cov, fetch_params, lag};
}
