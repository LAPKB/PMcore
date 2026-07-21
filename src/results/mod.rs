mod fit_result;
mod information_criteria;
pub(crate) mod parametric_output;
mod summary;

pub use fit_result::{
    CovarianceCycleUpdateDiagnostics, CovarianceCycleUpdateOutcome, CovarianceTrialRejectionReason,
    CovarianceUpdateNotAttemptedReason, CovarianceUpdateRejectionReason, DiagnosticTraceCoordinate,
    FitResult, InformationCoordinate, InformationCoordinateKind, InformationDiagnostics,
    InformationStatus, MarkovSimulationVarianceChainDiagnostics,
    MarkovSimulationVarianceDiagnostics, MarkovSimulationVarianceStatus, OccasionKappaEstimate,
    OperationalConvergenceCheck, OperationalConvergenceCriterion,
    OperationalConvergenceCriterionStatus, OperationalConvergenceDiagnostics,
    OperationalConvergenceOutcome, ParametricResult, ParametricWarning,
    PopulationUncertaintyDiagnostics, PopulationUncertaintyRegularization,
    PopulationUncertaintyStatus, PopulationUncertaintyUnavailableReason, RankDiagnosticStatus,
    RankMixingDiagnostic, RankMixingDiagnostics, ResidualCycleDiagnostics, ResidualErrorEstimate,
    SaemCycleDiagnostics, SaemEstimatorMetadata, SaemPhase, SubjectConditionalMode,
    SubjectEtaEstimate,
};

pub(crate) use information_criteria::derive_information_criteria;
pub use information_criteria::{
    InformationCriteriaDiagnostics, InformationCriteriaParameterCount,
    InformationCriteriaSampleSizeConvention, InformationCriteriaStatus,
    InformationCriteriaUnavailableReason,
};
pub use parametric_output::{
    IndividualEffectRow, IndividualParameterRow, InformationCriteriaRow, IterationRow,
    MarginalLikelihoodRow, OmegaRow, ParametricResultRecord, ParametricResultTables,
    ParametricSourceCovariance, ParametricSourceEffect, ParametricSourceMetadata,
    ParametricSourceParameter, ParametricSourceResidual, ParametricWarningRecord,
    PopulationParameterRow, PredictionRow, ResidualErrorRow, StatisticRow,
    PARAMETRIC_RESULT_SCHEMA_VERSION,
};
pub use summary::{FitSummary, IndividualSummary, ParameterSummary, PopulationSummary};
