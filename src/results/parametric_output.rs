use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};
use pharmsol::simulator::prediction::{Prediction, SubjectPredictions};
use pharmsol::{Censor, Data, Equation};
use serde::{Deserialize, Serialize};

use crate::algorithms::parametric::SaemConfig;
use crate::algorithms::StopReason;
use crate::estimation::parametric::{
    covariates::{CovariateEffect, CovariateEffectFamily},
    individual::{
        individual_psi, individual_psi_from_subject_mean, occasion_psi,
        occasion_psi_from_subject_mean,
    },
    marginal_likelihood::{
        MarginalLikelihoodDiagnostics, MarginalLikelihoodMethod, MarginalLikelihoodStatus,
        ProposalScaleSource,
    },
    shrinkage::{
        derive_eta_map_shrinkage, derive_eta_posterior_mean_shrinkage, derive_kappa_map_shrinkage,
        derive_kappa_posterior_mean_shrinkage, ShrinkageDiagnostics, ShrinkageValue,
    },
    transforms::{phi_to_psi, psi_to_phi},
};
use crate::estimation::{EstimationProblem, Iov, Omega, Parametric, ParametricErrorModel};
use crate::model::{EquationMetadataSource, ParameterScale, UnboundedParameter};
use crate::results::{
    derive_information_criteria, DiagnosticTraceCoordinate, InformationCoordinate,
    InformationCoordinateKind, InformationCriteriaDiagnostics,
    InformationCriteriaSampleSizeConvention, InformationCriteriaStatus, InformationDiagnostics,
    MarkovSimulationVarianceDiagnostics, MarkovSimulationVarianceStatus,
    OperationalConvergenceDiagnostics, OperationalConvergenceOutcome, ParametricResult,
    ParametricWarning, PopulationUncertaintyDiagnostics, RankDiagnosticStatus,
    SaemEstimatorMetadata, SaemPhase, SubjectConditionalMode,
};
use crate::ResidualErrorModel;

pub const PARAMETRIC_RESULT_SCHEMA_VERSION: u32 = 9;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopulationParameterRow {
    pub name: String,
    pub estimate: f64,
    pub scale: String,
    pub estimated: bool,
    pub iiv: bool,
    pub iov: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OmegaRow {
    pub row: String,
    pub column: String,
    pub estimate: f64,
    pub structural: bool,
    pub estimated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualErrorRow {
    pub output: String,
    pub output_index: usize,
    pub family: String,
    pub component: String,
    pub estimate: f64,
    pub estimated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndividualEffectRow {
    pub subject: String,
    pub source: String,
    pub effect_kind: String,
    pub parameter: String,
    pub occasion: Option<usize>,
    pub value: f64,
    pub mode_converged: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndividualParameterRow {
    pub subject: String,
    pub occasion: Option<usize>,
    pub parameter: String,
    pub value: f64,
    pub source: String,
    pub mode_converged: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IterationRow {
    pub cycle: usize,
    pub phase: String,
    pub conditional_n2ll: f64,
    pub sa_step: f64,
    pub covariance_step: f64,
    pub eta_proposals: usize,
    pub eta_accepted: usize,
    pub eta_rejected: usize,
    pub eta_nonfinite: usize,
    pub eta_block_proposals: usize,
    pub eta_block_accepted: usize,
    pub eta_block_rejected: usize,
    pub eta_block_nonfinite: usize,
    pub kappa_proposals: usize,
    pub kappa_accepted: usize,
    pub kappa_rejected: usize,
    pub kappa_nonfinite: usize,
    pub omega_update_rejected: bool,
    pub omega_iov_update_rejected: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatisticRow {
    pub cycle: usize,
    pub kind: String,
    pub name: String,
    pub row: Option<String>,
    pub column: Option<String>,
    pub output_index: Option<usize>,
    pub component: Option<String>,
    pub value: Option<f64>,
    /// Availability of the corresponding information diagnostic. Ordinary
    /// cycle statistics leave this empty.
    pub status: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginalLikelihoodRow {
    pub scope: String,
    pub subject: Option<String>,
    pub method: String,
    pub status: String,
    pub samples_per_subject: usize,
    pub seed: Option<u64>,
    pub degrees_of_freedom: u32,
    pub covariance_scale_multiplier: f64,
    pub proposal_scale_source: String,
    pub dimension: usize,
    pub occasion_indices: String,
    pub mode: String,
    pub mode_converged: Option<bool>,
    pub log_marginal_likelihood: Option<f64>,
    pub n2ll: Option<f64>,
    pub n2ll_mcse: Option<f64>,
    pub effective_sample_size: Option<f64>,
    pub effective_sample_fraction: Option<f64>,
    pub zero_weight_count: usize,
    pub failure: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InformationCriteriaRow {
    pub status: String,
    pub sample_size_convention: String,
    pub subject_count: usize,
    pub population_parameter_count: usize,
    pub covariate_parameter_count: usize,
    pub omega_parameter_count: usize,
    pub omega_iov_parameter_count: usize,
    pub residual_parameter_count: usize,
    pub free_parameter_count: usize,
    pub source_marginal_n2ll: Option<f64>,
    pub source_marginal_n2ll_mcse: Option<f64>,
    pub aic: Option<f64>,
    pub bic: Option<f64>,
    pub aic_mcse: Option<f64>,
    pub bic_mcse: Option<f64>,
    pub failure_reason: Option<String>,
}

/// Legacy CSV projection used only for fits without covariate effects.
#[derive(Serialize)]
struct NoEffectInformationCriteriaRow {
    status: String,
    sample_size_convention: String,
    subject_count: usize,
    population_parameter_count: usize,
    omega_parameter_count: usize,
    omega_iov_parameter_count: usize,
    residual_parameter_count: usize,
    free_parameter_count: usize,
    source_marginal_n2ll: Option<f64>,
    source_marginal_n2ll_mcse: Option<f64>,
    aic: Option<f64>,
    bic: Option<f64>,
    aic_mcse: Option<f64>,
    bic_mcse: Option<f64>,
    failure_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionRow {
    pub subject: String,
    pub time: f64,
    pub output_index: usize,
    pub block: usize,
    pub observation: Option<f64>,
    pub censoring: String,
    pub population_prediction: f64,
    pub conditional_prediction: Option<f64>,
    pub conditional_source: Option<String>,
}

/// One canonical transformed-space covariate coefficient.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CovariateEffectRow {
    pub order: usize,
    pub name: String,
    pub family: String,
    pub parameter: String,
    pub parameter_index: usize,
    pub covariate: String,
    pub center: Option<f64>,
    pub reference: Option<f64>,
    pub level: Option<f64>,
    pub initial: f64,
    pub estimate: f64,
    pub estimated: bool,
}

/// One exact subject-static covariate value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubjectCovariateRow {
    pub subject: String,
    pub subject_index: usize,
    pub covariate: String,
    pub value: f64,
}

/// One subject/parameter population mean in transformed and execution space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubjectPopulationParameterRow {
    pub subject: String,
    pub subject_index: usize,
    pub parameter: String,
    pub parameter_index: usize,
    pub phi: f64,
    pub psi: f64,
}

/// Parseable, equation-free tables for a parametric fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricResultTables {
    pub population: Vec<PopulationParameterRow>,
    pub omega: Vec<OmegaRow>,
    pub omega_iov: Option<Vec<OmegaRow>>,
    pub residual_error: Vec<ResidualErrorRow>,
    pub individual_effects: Vec<IndividualEffectRow>,
    pub individual_parameters: Vec<IndividualParameterRow>,
    pub iterations: Vec<IterationRow>,
    pub statistics: Vec<StatisticRow>,
    pub marginal_likelihood: Vec<MarginalLikelihoodRow>,
    pub information_criteria: Vec<InformationCriteriaRow>,
    pub predictions: Vec<PredictionRow>,
    /// Required in schema 7, including an explicit empty vector for no-effect fits.
    pub covariate_effects: Vec<CovariateEffectRow>,
    /// Required in schema 7, including an explicit empty vector for no-effect fits.
    pub subject_covariates: Vec<SubjectCovariateRow>,
    /// Required in schema 7, including an explicit empty vector for no-effect fits.
    pub subject_population_parameters: Vec<SubjectPopulationParameterRow>,
}

impl ParametricResultTables {
    pub fn write_population(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.population,
            &["name", "estimate", "scale", "estimated", "iiv", "iov"],
        )
    }

    pub fn write_omega(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.omega,
            &["row", "column", "estimate", "structural", "estimated"],
        )
    }

    pub fn write_omega_iov(&self, path: impl AsRef<Path>) -> Result<()> {
        let rows = self
            .omega_iov
            .as_ref()
            .context("Omega_IOV is not present in this result")?;
        write_csv(
            path.as_ref(),
            rows,
            &["row", "column", "estimate", "structural", "estimated"],
        )
    }

    pub fn write_residual_error(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.residual_error,
            &[
                "output",
                "output_index",
                "family",
                "component",
                "estimate",
                "estimated",
            ],
        )
    }

    pub fn write_individual_effects(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.individual_effects,
            &[
                "subject",
                "source",
                "effect_kind",
                "parameter",
                "occasion",
                "value",
                "mode_converged",
            ],
        )
    }

    pub fn write_individual_parameters(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.individual_parameters,
            &[
                "subject",
                "occasion",
                "parameter",
                "value",
                "source",
                "mode_converged",
            ],
        )
    }

    pub fn write_iterations(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.iterations,
            &[
                "cycle",
                "phase",
                "conditional_n2ll",
                "sa_step",
                "covariance_step",
                "eta_proposals",
                "eta_accepted",
                "eta_rejected",
                "eta_nonfinite",
                "eta_block_proposals",
                "eta_block_accepted",
                "eta_block_rejected",
                "eta_block_nonfinite",
                "kappa_proposals",
                "kappa_accepted",
                "kappa_rejected",
                "kappa_nonfinite",
                "omega_update_rejected",
                "omega_iov_update_rejected",
            ],
        )
    }

    pub fn write_statistics(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.statistics,
            &[
                "cycle",
                "kind",
                "name",
                "row",
                "column",
                "output_index",
                "component",
                "value",
                "status",
            ],
        )
    }

    pub fn write_marginal_likelihood(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.marginal_likelihood,
            &[
                "scope",
                "subject",
                "method",
                "status",
                "samples_per_subject",
                "seed",
                "degrees_of_freedom",
                "covariance_scale_multiplier",
                "proposal_scale_source",
                "dimension",
                "occasion_indices",
                "mode",
                "mode_converged",
                "log_marginal_likelihood",
                "n2ll",
                "n2ll_mcse",
                "effective_sample_size",
                "effective_sample_fraction",
                "zero_weight_count",
                "failure",
            ],
        )
    }

    pub fn write_information_criteria(&self, path: impl AsRef<Path>) -> Result<()> {
        let headers = [
            "status",
            "sample_size_convention",
            "subject_count",
            "population_parameter_count",
            "omega_parameter_count",
            "omega_iov_parameter_count",
            "residual_parameter_count",
            "free_parameter_count",
            "source_marginal_n2ll",
            "source_marginal_n2ll_mcse",
            "aic",
            "bic",
            "aic_mcse",
            "bic_mcse",
            "failure_reason",
        ];
        if self
            .information_criteria
            .iter()
            .all(|row| row.covariate_parameter_count == 0)
        {
            let rows = self
                .information_criteria
                .iter()
                .map(|row| NoEffectInformationCriteriaRow {
                    status: row.status.clone(),
                    sample_size_convention: row.sample_size_convention.clone(),
                    subject_count: row.subject_count,
                    population_parameter_count: row.population_parameter_count,
                    omega_parameter_count: row.omega_parameter_count,
                    omega_iov_parameter_count: row.omega_iov_parameter_count,
                    residual_parameter_count: row.residual_parameter_count,
                    free_parameter_count: row.free_parameter_count,
                    source_marginal_n2ll: row.source_marginal_n2ll,
                    source_marginal_n2ll_mcse: row.source_marginal_n2ll_mcse,
                    aic: row.aic,
                    bic: row.bic,
                    aic_mcse: row.aic_mcse,
                    bic_mcse: row.bic_mcse,
                    failure_reason: row.failure_reason.clone(),
                })
                .collect::<Vec<_>>();
            write_csv(path.as_ref(), &rows, &headers)
        } else {
            write_csv(
                path.as_ref(),
                &self.information_criteria,
                &[
                    "status",
                    "sample_size_convention",
                    "subject_count",
                    "population_parameter_count",
                    "covariate_parameter_count",
                    "omega_parameter_count",
                    "omega_iov_parameter_count",
                    "residual_parameter_count",
                    "free_parameter_count",
                    "source_marginal_n2ll",
                    "source_marginal_n2ll_mcse",
                    "aic",
                    "bic",
                    "aic_mcse",
                    "bic_mcse",
                    "failure_reason",
                ],
            )
        }
    }

    pub fn write_predictions(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.predictions,
            &[
                "subject",
                "time",
                "output_index",
                "block",
                "observation",
                "censoring",
                "population_prediction",
                "conditional_prediction",
                "conditional_source",
            ],
        )
    }

    pub fn write_covariate_effects(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.covariate_effects,
            &[
                "order",
                "name",
                "family",
                "parameter",
                "parameter_index",
                "covariate",
                "center",
                "reference",
                "level",
                "initial",
                "estimate",
                "estimated",
            ],
        )
    }

    pub fn write_subject_covariates(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.subject_covariates,
            &["subject", "subject_index", "covariate", "value"],
        )
    }

    pub fn write_subject_population_parameters(&self, path: impl AsRef<Path>) -> Result<()> {
        write_csv(
            path.as_ref(),
            &self.subject_population_parameters,
            &[
                "subject",
                "subject_index",
                "parameter",
                "parameter_index",
                "phi",
                "psi",
            ],
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParametricWarningRecord {
    pub kind: String,
    pub output: Option<String>,
    pub first_cycle: usize,
    pub count: usize,
    pub subjects: Option<Vec<String>>,
}

/// Ordered population declaration retained independently of output tables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricSourceParameter {
    pub name: String,
    pub parameter_index: usize,
    pub scale: String,
    /// Immutable natural-scale value declared before the first SAEM cycle.
    pub initial: f64,
    pub estimate: f64,
    pub estimated: bool,
}

/// Ordered population parameter selected for an IIV or IOV effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricSourceEffect {
    pub parameter_index: usize,
    pub parameter_name: String,
}

/// Canonical covariance declaration and final value snapshot.
///
/// Names and the complete ordered matrix deliberately duplicate the result
/// table so schema-9 readers can bind and validate this source independently.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricSourceCovariance {
    pub dimension: usize,
    pub names: Vec<String>,
    pub values: Vec<Vec<f64>>,
    pub structural_mask: Vec<Vec<bool>>,
    pub estimated_mask: Vec<Vec<bool>>,
    /// Immutable initial covariance as declared before the first SAEM cycle.
    /// Fixed entries must equal their final values; free entries must be finite
    /// but may differ.
    pub initial_values: Vec<Vec<f64>>,
}

/// Ordered residual declaration and its exact final component snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricSourceResidual {
    pub output: String,
    pub output_index: usize,
    pub family: String,
    pub components: Vec<String>,
    pub values: Vec<f64>,
    pub estimated_mask: Vec<bool>,
    /// Immutable initial component values as declared before the first SAEM cycle.
    /// Fixed components must equal their final values; free components must be
    /// finite but may differ.
    pub initial_values: Vec<f64>,
    /// Immutable initial estimated mask as declared before the first SAEM cycle.
    pub initial_estimated_mask: Vec<bool>,
}

/// One ordered covariate declaration and its final coefficient estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricSourceCovariateEffect {
    pub order: usize,
    pub name: String,
    pub family: String,
    pub parameter: String,
    pub parameter_index: usize,
    pub covariate: String,
    pub center: Option<f64>,
    pub reference: Option<f64>,
    pub level: Option<f64>,
    pub initial: f64,
    pub estimate: f64,
    pub estimated: bool,
}

/// One subject design row retained independently of output tables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricSourceSubjectDesign {
    pub subject: String,
    pub subject_index: usize,
    pub values: Vec<f64>,
}

/// Canonical scientific declarations and resolved schema-9 snapshot.
///
/// This snapshot is generated directly from immutable [`ParametricResult`]
/// metadata. It is independent schema consistency evidence, not a
/// cryptographic signature and not continuation state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParametricSourceMetadata {
    pub parameters: Vec<ParametricSourceParameter>,
    pub random_effects: Vec<ParametricSourceEffect>,
    pub omega: ParametricSourceCovariance,
    pub iov_effects: Vec<ParametricSourceEffect>,
    pub omega_iov: Option<ParametricSourceCovariance>,
    pub residual_outputs: Vec<ParametricSourceResidual>,
    /// Required explicit empty fields for no-effect schema-9 records.
    pub covariate_effects: Vec<ParametricSourceCovariateEffect>,
    pub subject_covariates: Vec<SubjectCovariateRow>,
    pub subject_design: Vec<ParametricSourceSubjectDesign>,
    pub subject_population_parameters: Vec<SubjectPopulationParameterRow>,
}

/// Versioned, equation-free persisted parametric result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricResultRecord {
    pub schema_version: u32,
    pub fit_family: String,
    pub algorithm: String,
    pub config: SaemConfig,
    pub effective_n_chains: usize,
    pub termination: Option<StopReason>,
    pub objective_kind: String,
    pub conditional_n2ll: f64,
    pub subject_count: usize,
    pub marginal_likelihood: Option<MarginalLikelihoodDiagnostics>,
    pub information_criteria: InformationCriteriaDiagnostics,
    pub source_metadata: ParametricSourceMetadata,
    pub warnings: Vec<ParametricWarningRecord>,
    pub tables: ParametricResultTables,
    pub information_diagnostics: InformationDiagnostics,
    pub population_uncertainty: PopulationUncertaintyDiagnostics,
    pub conditional_modes: Vec<SubjectConditionalMode>,
    pub shrinkage: ShrinkageDiagnostics,
    pub markov_simulation_variance: MarkovSimulationVarianceDiagnostics,
    pub operational_convergence: OperationalConvergenceDiagnostics,
    pub estimator_metadata: SaemEstimatorMetadata,
}

fn equal_with_roundoff(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 64.0 * f64::EPSILON * left.abs().max(right.abs()).max(1.0)
}

fn stable_number(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        value.to_string()
    }
}

fn validate_persisted_marginal_likelihood(record: &ParametricResultRecord) -> Result<()> {
    match (
        record.config.marginal_likelihood,
        record.marginal_likelihood.as_ref(),
    ) {
        (None, None) => {
            if record.tables.marginal_likelihood != marginal_likelihood_rows(None) {
                bail!("persisted disabled marginal_likelihood table is inconsistent");
            }
            let actual_statistics = record
                .tables
                .statistics
                .iter()
                .filter(|row| row.kind.starts_with("marginal_likelihood"))
                .cloned()
                .collect::<Vec<_>>();
            let mut expected_statistics = Vec::new();
            marginal_likelihood_statistics(
                record.tables.iterations.len(),
                None,
                &mut expected_statistics,
            );
            if actual_statistics != expected_statistics {
                bail!("persisted disabled marginal-likelihood statistics are inconsistent");
            }
        }
        (None, Some(_)) => bail!("persisted N2 diagnostics are present while N2 is disabled"),
        (Some(_), None) => bail!("persisted N2 diagnostics are required when N2 is configured"),
        (Some(config), Some(diagnostics)) => {
            if diagnostics.config != config {
                bail!("persisted N2 configuration does not match the retained fit configuration");
            }
            let mut expected_subjects = Vec::<String>::new();
            for row in &record.tables.individual_parameters {
                if expected_subjects.last() != Some(&row.subject)
                    && !expected_subjects.contains(&row.subject)
                {
                    expected_subjects.push(row.subject.clone());
                }
            }
            let actual_subjects = diagnostics
                .subjects
                .iter()
                .map(|subject| subject.subject_id.clone())
                .collect::<Vec<_>>();
            if actual_subjects != expected_subjects {
                bail!("persisted N2 subject ordering does not match result subject ordering");
            }
            let available = matches!(
                diagnostics.status,
                MarginalLikelihoodStatus::Available
                    | MarginalLikelihoodStatus::AvailableWithNonconvergedModes { .. }
            );
            let totals = [
                diagnostics.log_marginal_likelihood,
                diagnostics.n2ll,
                diagnostics.n2ll_mcse,
            ];
            if available {
                if !totals.iter().all(Option::is_some) {
                    bail!("persisted available N2 requires every population value");
                }
            } else if totals.iter().any(Option::is_some) {
                bail!("persisted unavailable N2 must omit every population value");
            }
            if totals.into_iter().flatten().any(|value| !value.is_finite()) {
                bail!("persisted N2 population values must be finite when present");
            }
            if let MarginalLikelihoodStatus::Unavailable { failures } = &diagnostics.status {
                if failures.is_empty() {
                    bail!("persisted unavailable N2 status must retain failures");
                }
            }
            let eta_dimension = record
                .tables
                .omega
                .iter()
                .filter(|row| row.row == row.column)
                .count();
            let kappa_dimension = record
                .tables
                .omega_iov
                .as_ref()
                .map(|rows| rows.iter().filter(|row| row.row == row.column).count())
                .unwrap_or(0);
            for (subject_index, subject) in diagnostics.subjects.iter().enumerate() {
                let mut expected_occasions = Vec::new();
                for effect in record.tables.individual_effects.iter().filter(|effect| {
                    effect.subject == subject.subject_id && effect.effect_kind == "kappa"
                }) {
                    if let Some(occasion) = effect.occasion {
                        if !expected_occasions.contains(&occasion) {
                            expected_occasions.push(occasion);
                        }
                    }
                }
                let expected_dimension = eta_dimension + expected_occasions.len() * kappa_dimension;
                if subject.occasion_indices != expected_occasions
                    || subject.dimension != expected_dimension
                    || subject.mode.iter().any(|value| !value.is_finite())
                {
                    bail!(
                        "persisted N2 metadata for subject '{}' is malformed",
                        subject.subject_id
                    );
                }
                let subject_available = subject.failure.is_none();
                let subject_values = [
                    subject.log_marginal_likelihood,
                    subject.n2ll,
                    subject.effective_sample_size,
                    subject.effective_sample_fraction,
                    subject.var_log,
                    subject.n2ll_mcse,
                ];
                if subject_available {
                    if subject.mode.len() != expected_dimension
                        || ![
                            subject.log_marginal_likelihood,
                            subject.n2ll,
                            subject.var_log,
                            subject.n2ll_mcse,
                        ]
                        .iter()
                        .all(Option::is_some)
                    {
                        bail!("persisted available N2 subject is incomplete");
                    }
                } else if subject_values.iter().any(Option::is_some) {
                    bail!("persisted failed N2 subject must omit every numerical result");
                }
                if subject_values
                    .into_iter()
                    .flatten()
                    .any(|value| !value.is_finite())
                    || subject.zero_weight_count > subject.samples
                {
                    bail!("persisted N2 subject diagnostics contain invalid numeric values");
                }
                if subject_available {
                    let (Some(log_likelihood), Some(n2ll), Some(variance), Some(mcse)) = (
                        subject.log_marginal_likelihood,
                        subject.n2ll,
                        subject.var_log,
                        subject.n2ll_mcse,
                    ) else {
                        bail!("persisted available N2 subject is incomplete");
                    };
                    if variance < 0.0
                        || mcse < 0.0
                        || !equal_with_roundoff(n2ll, -2.0 * log_likelihood)
                        || !equal_with_roundoff(mcse * mcse, 4.0 * variance)
                    {
                        bail!("persisted N2 subject algebra is inconsistent");
                    }
                }
                match subject.method {
                    MarginalLikelihoodMethod::ExactNoLatent => {
                        if subject.dimension != 0
                            || !subject.occasion_indices.is_empty()
                            || !subject.mode.is_empty()
                            || subject.mode_converged.is_some()
                            || subject.samples != 0
                            || subject.seed.is_some()
                            || subject.proposal_scale_source
                                != ProposalScaleSource::NotApplicableNoLatent
                            || subject.effective_sample_size.is_some()
                            || subject.effective_sample_fraction.is_some()
                            || subject.zero_weight_count != 0
                            || (subject_available
                                && (subject.var_log != Some(0.0) || subject.n2ll_mcse != Some(0.0)))
                        {
                            bail!("persisted exact no-latent N2 diagnostics are inconsistent");
                        }
                    }
                    MarginalLikelihoodMethod::StudentTImportanceSampling => {
                        let expected_seed = crate::estimation::parametric::marginal_likelihood::marginal_likelihood_subject_seed(
                            config.seed,
                            subject_index,
                        );
                        let mode_unavailable = matches!(
                            subject.failure.as_ref(),
                            Some(
                                crate::estimation::parametric::marginal_likelihood::MarginalLikelihoodFailureReason::MissingConditionalMode
                                    | crate::estimation::parametric::marginal_likelihood::MarginalLikelihoodFailureReason::ConditionalModeCalculationFailed(_)
                            )
                        );
                        let mode_metadata_inconsistent = if mode_unavailable {
                            !subject.mode.is_empty() || subject.mode_converged.is_some()
                        } else {
                            subject.mode_converged.is_none()
                        };
                        let expected_proposal_source = match config.proposal {
                            crate::estimation::parametric::MarginalLikelihoodProposal::FinalRawOmegaBlocks => {
                                ProposalScaleSource::FinalRawOmegaBlocks
                            }
                            crate::estimation::parametric::MarginalLikelihoodProposal::ConditionalModeCurvature => {
                                ProposalScaleSource::ConditionalModeCurvature
                            }
                        };
                        if subject.dimension == 0
                            || subject.samples != config.samples_per_subject
                            || subject.seed != Some(expected_seed)
                            || subject.proposal_scale_source != expected_proposal_source
                            || mode_metadata_inconsistent
                        {
                            bail!("persisted stochastic N2 diagnostics have inconsistent proposal metadata");
                        }
                        if subject_available {
                            let ess = subject.effective_sample_size.ok_or_else(|| {
                                anyhow::anyhow!(
                                    "persisted available stochastic N2 diagnostics require ESS"
                                )
                            })?;
                            let fraction = subject.effective_sample_fraction.ok_or_else(|| {
                                anyhow::anyhow!("persisted available stochastic N2 diagnostics require ESS fraction")
                            })?;
                            if ess <= 0.0
                                || ess > subject.samples as f64
                                || fraction <= 0.0
                                || fraction > 1.0
                                || !equal_with_roundoff(fraction, ess / subject.samples as f64)
                            {
                                bail!("persisted available stochastic N2 ESS is out of range or inconsistent");
                            }
                        }
                    }
                }
            }
            if available {
                let mut summed_log = 0.0;
                let mut summed_variance = 0.0;
                for subject in &diagnostics.subjects {
                    summed_log += subject.log_marginal_likelihood.ok_or_else(|| {
                        anyhow::anyhow!("persisted available N2 subject is incomplete")
                    })?;
                    summed_variance += subject.var_log.ok_or_else(|| {
                        anyhow::anyhow!("persisted available N2 subject is incomplete")
                    })?;
                }
                let expected_n2ll = -2.0 * summed_log;
                let expected_mcse = 2.0 * summed_variance.sqrt();
                let persisted_log = diagnostics.log_marginal_likelihood.ok_or_else(|| {
                    anyhow::anyhow!("persisted available N2 population is incomplete")
                })?;
                let persisted_n2ll = diagnostics.n2ll.ok_or_else(|| {
                    anyhow::anyhow!("persisted available N2 population is incomplete")
                })?;
                let persisted_mcse = diagnostics.n2ll_mcse.ok_or_else(|| {
                    anyhow::anyhow!("persisted available N2 population is incomplete")
                })?;
                if !summed_log.is_finite()
                    || !summed_variance.is_finite()
                    || !expected_n2ll.is_finite()
                    || !expected_mcse.is_finite()
                    || !equal_with_roundoff(persisted_log, summed_log)
                    || !equal_with_roundoff(persisted_n2ll, expected_n2ll)
                    || !equal_with_roundoff(persisted_mcse, expected_mcse)
                {
                    bail!("persisted N2 population algebra is inconsistent");
                }
            }
            let failed_subjects = diagnostics
                .subjects
                .iter()
                .filter_map(|subject| {
                    subject.failure.clone().map(|reason| {
                        crate::estimation::parametric::marginal_likelihood::MarginalLikelihoodSubjectFailure {
                            subject_id: subject.subject_id.clone(),
                            reason,
                        }
                    })
                })
                .collect::<Vec<_>>();
            let nonconverged_subjects = diagnostics
                .subjects
                .iter()
                .filter(|subject| {
                    subject.mode_converged == Some(false) && subject.failure.is_none()
                })
                .map(|subject| subject.subject_id.clone())
                .collect::<Vec<_>>();
            match &diagnostics.status {
                MarginalLikelihoodStatus::Available => {
                    if !failed_subjects.is_empty() || !nonconverged_subjects.is_empty() {
                        bail!("persisted available N2 status conflicts with subject statuses");
                    }
                }
                MarginalLikelihoodStatus::AvailableWithNonconvergedModes { subjects } => {
                    if !failed_subjects.is_empty() || *subjects != nonconverged_subjects {
                        bail!(
                            "persisted nonconverged-mode N2 status conflicts with subject statuses"
                        );
                    }
                }
                MarginalLikelihoodStatus::Unavailable { failures } => {
                    if *failures != failed_subjects {
                        bail!("persisted unavailable N2 failures conflict with subject failures");
                    }
                }
            }
            let expected_rows = marginal_likelihood_rows(record.marginal_likelihood.as_ref());
            if record.tables.marginal_likelihood.len() != expected_rows.len()
                || record
                    .tables
                    .marginal_likelihood
                    .iter()
                    .zip(&expected_rows)
                    .any(|(actual, expected)| !marginal_likelihood_row_equal(actual, expected))
            {
                bail!("persisted marginal_likelihood table does not match N2 diagnostics");
            }
            let actual_statistics = record
                .tables
                .statistics
                .iter()
                .filter(|row| row.kind.starts_with("marginal_likelihood"))
                .cloned()
                .collect::<Vec<_>>();
            let mut expected_statistics = Vec::new();
            marginal_likelihood_statistics(
                record.tables.iterations.len(),
                record.marginal_likelihood.as_ref(),
                &mut expected_statistics,
            );
            if actual_statistics.len() != expected_statistics.len()
                || actual_statistics
                    .iter()
                    .zip(&expected_statistics)
                    .any(|(actual, expected)| !statistic_row_equal(actual, expected))
            {
                bail!("persisted marginal-likelihood statistics do not match N2 diagnostics");
            }
        }
    }
    Ok(())
}

fn validate_persisted_covariate_statistics(record: &ParametricResultRecord) -> Result<()> {
    let rows = record
        .tables
        .statistics
        .iter()
        .filter(|row| row.kind == "covariate_effect")
        .collect::<Vec<_>>();
    let final_rows = record
        .tables
        .statistics
        .iter()
        .filter(|row| row.kind == "covariate_effect_final")
        .collect::<Vec<_>>();
    let effects = &record.source_metadata.covariate_effects;
    if effects.is_empty() {
        if !rows.is_empty() || !final_rows.is_empty() {
            bail!("no-effect record contains covariate coefficient statistics");
        }
        return Ok(());
    }
    if rows.len() != record.tables.iterations.len() * effects.len() {
        bail!("persisted covariate coefficient statistics have the wrong shape");
    }
    for (cycle_index, iteration) in record.tables.iterations.iter().enumerate() {
        for (effect_index, effect) in effects.iter().enumerate() {
            let row = rows[cycle_index * effects.len() + effect_index];
            if row.cycle != iteration.cycle
                || row.name != effect.name
                || row.row.is_some()
                || row.column.is_some()
                || row.output_index.is_some()
                || row.component.is_some()
                || row.status.as_deref()
                    != Some(if effect.estimated {
                        "estimated"
                    } else {
                        "fixed"
                    })
                || row.value.is_none_or(|value| !value.is_finite())
            {
                bail!("persisted covariate coefficient statistic is malformed or reordered");
            }
            if !effect.estimated && row.value != Some(effect.initial) {
                bail!("persisted fixed covariate coefficient changed during a cycle");
            }
        }
    }
    if final_rows.len() != effects.len() {
        bail!("persisted final covariate coefficient statistics have the wrong shape");
    }
    for (row, effect) in final_rows.iter().zip(effects) {
        if row.cycle
            != record
                .tables
                .iterations
                .last()
                .map_or(0, |iteration| iteration.cycle)
            || row.name != effect.name
            || row.row.is_some()
            || row.column.is_some()
            || row.output_index.is_some()
            || row.component.is_some()
            || row.status.as_deref()
                != Some(if effect.estimated {
                    "estimated"
                } else {
                    "fixed"
                })
            || row
                .value
                .is_none_or(|value| !equal_with_roundoff(value, effect.estimate))
        {
            bail!("persisted final covariate coefficient statistic disagrees with its estimate");
        }
    }
    Ok(())
}

fn validate_persisted_information_criteria(record: &ParametricResultRecord) -> Result<()> {
    validate_source_snapshot(&record.source_metadata)?;
    validate_persisted_source_metadata(&record.tables, &record.source_metadata)?;
    validate_persisted_information_shape(&record.information_diagnostics)?;
    if !record.source_metadata.covariate_effects.is_empty()
        && record.source_metadata.subject_design.len() != record.subject_count
    {
        bail!("persisted covariate subject order/count does not match subject_count");
    }
    validate_persisted_covariate_statistics(record)?;
    validate_final_source_statistics(record)?;
    if let Some(marginal) = record.marginal_likelihood.as_ref() {
        if marginal.subjects.len() != record.subject_count {
            bail!(
                "persisted marginal-likelihood subject count does not match persisted data subject count"
            );
        }
    }
    let expected_coordinates = expected_information_coordinates(&record.source_metadata)?;
    if record.information_diagnostics.coordinates != expected_coordinates {
        bail!("persisted information coordinates do not match persisted free-parameter metadata");
    }
    let expected = derive_information_criteria(
        record.marginal_likelihood.as_ref(),
        &record.information_diagnostics.coordinates,
        record.subject_count,
    );
    if !information_criteria_equal(&record.information_criteria, &expected) {
        bail!("persisted information criteria do not match their N2, coordinate, and subject-count sources");
    }
    let expected_rows = information_criteria_rows(&expected);
    if record.tables.information_criteria.len() != 1
        || !information_criteria_row_equal(
            &record.tables.information_criteria[0],
            &expected_rows[0],
        )
    {
        bail!("persisted information-criteria table does not match diagnostics");
    }
    let actual_statistics = record
        .tables
        .statistics
        .iter()
        .filter(|row| {
            row.kind == "information_criteria_status"
                || row.kind == "information_criteria"
                || row.kind == "information_criteria_metadata"
        })
        .collect::<Vec<_>>();
    let mut expected_statistics = Vec::new();
    information_criteria_statistics(
        record.tables.iterations.len(),
        &expected,
        &mut expected_statistics,
    );
    if actual_statistics.len() != expected_statistics.len()
        || actual_statistics
            .iter()
            .zip(&expected_statistics)
            .any(|(actual, expected)| !statistic_row_equal(actual, expected))
    {
        bail!("persisted information-criteria statistics do not match diagnostics");
    }
    Ok(())
}

fn validate_final_source_statistics(record: &ParametricResultRecord) -> Result<()> {
    let final_cycle = record
        .tables
        .iterations
        .last()
        .map(|row| row.cycle)
        .unwrap_or_default();
    let final_rows = record
        .tables
        .statistics
        .iter()
        .filter(|row| row.cycle == final_cycle)
        .collect::<Vec<_>>();
    let suffix = if record.estimator_metadata.averaged_iterations > 0 {
        "_final"
    } else {
        ""
    };
    for parameter in &record.source_metadata.parameters {
        let matches = final_rows
            .iter()
            .filter(|row| row.kind == format!("theta{suffix}") && row.name == parameter.name)
            .collect::<Vec<_>>();
        if matches.len() != 1
            || !matches[0]
                .value
                .is_some_and(|value| equal_with_roundoff(value, parameter.estimate))
        {
            bail!("persisted final theta statistic disagrees with source snapshot");
        }
    }
    for (kind, covariance) in std::iter::once(("omega", &record.source_metadata.omega)).chain(
        record
            .source_metadata
            .omega_iov
            .as_ref()
            .map(|covariance| ("omega_iov", covariance)),
    ) {
        for row in 0..covariance.dimension {
            for column in 0..=row {
                let matches = final_rows
                    .iter()
                    .filter(|entry| {
                        entry.kind == format!("{kind}{suffix}")
                            && entry.row.as_deref() == Some(covariance.names[row].as_str())
                            && entry.column.as_deref() == Some(covariance.names[column].as_str())
                    })
                    .collect::<Vec<_>>();
                if matches.len() != 1
                    || !matches[0].value.is_some_and(|value| {
                        equal_with_roundoff(value, covariance.values[row][column])
                    })
                {
                    bail!("persisted final {kind} statistic disagrees with source snapshot");
                }
            }
        }
    }
    for residual in &record.source_metadata.residual_outputs {
        for (component, expected) in residual.components.iter().zip(&residual.values) {
            let matches = final_rows
                .iter()
                .filter(|row| {
                    row.kind == format!("residual{suffix}")
                        && row.name == residual.output
                        && row.output_index == Some(residual.output_index)
                        && row.component.as_deref() == Some(component.as_str())
                })
                .collect::<Vec<_>>();
            if matches.len() != 1
                || !matches[0]
                    .value
                    .is_some_and(|value| equal_with_roundoff(value, *expected))
            {
                bail!("persisted final residual statistic disagrees with source snapshot");
            }
        }
    }
    Ok(())
}

fn validate_source_snapshot(source: &ParametricSourceMetadata) -> Result<()> {
    let mut parameter_names = std::collections::HashSet::new();
    for (parameter_index, parameter) in source.parameters.iter().enumerate() {
        if parameter.name.is_empty()
            || parameter.parameter_index != parameter_index
            || !parameter.initial.is_finite()
            || !parameter.estimate.is_finite()
            || (!parameter.estimated && !equal_with_roundoff(parameter.initial, parameter.estimate))
            || !parameter_names.insert(parameter.name.as_str())
        {
            bail!("source population parameter declarations must be finite, ordered, non-empty, unique, and preserve fixed initial values");
        }
        let scale = parse_scale(&parameter.scale)?;
        if !psi_to_phi(parameter.initial, scale).is_finite()
            || !psi_to_phi(parameter.estimate, scale).is_finite()
        {
            bail!("source population parameter initial/final value violates its scale domain");
        }
    }
    validate_source_effects(&source.random_effects, &source.parameters, "IIV")?;
    validate_source_covariance(
        &source.omega,
        &source
            .random_effects
            .iter()
            .map(|effect| effect.parameter_name.as_str())
            .collect::<Vec<_>>(),
        "Omega",
    )?;
    validate_source_effects(&source.iov_effects, &source.parameters, "IOV")?;
    match (&source.omega_iov, source.iov_effects.is_empty()) {
        (None, true) => {}
        (Some(covariance), false) => validate_source_covariance(
            covariance,
            &source
                .iov_effects
                .iter()
                .map(|effect| effect.parameter_name.as_str())
                .collect::<Vec<_>>(),
            "Omega_IOV",
        )?,
        (Some(_), true) => bail!("source Omega_IOV exists without IOV effects"),
        (None, false) => bail!("source IOV effects require Omega_IOV"),
    }
    let mut output_names = std::collections::HashSet::new();
    let mut output_indices = std::collections::HashSet::new();
    let mut previous_index = None;
    for residual in &source.residual_outputs {
        if residual.output.is_empty()
            || !output_names.insert(residual.output.as_str())
            || !output_indices.insert(residual.output_index)
            || previous_index.is_some_and(|index| index >= residual.output_index)
        {
            bail!("source residual outputs must be unique and in output-index order");
        }
        previous_index = Some(residual.output_index);
        let expected = residual_component_names(&residual.family)?;
        if residual.components != expected
            || residual.values.len() != expected.len()
            || residual.estimated_mask.len() != expected.len()
            || residual.values.iter().any(|value| !value.is_finite())
        {
            bail!("source residual component declaration/value snapshot is malformed");
        }
        validate_residual_values(&residual.family, &residual.values, &residual.estimated_mask)?;
        // Validate immutable initial residual declarations.
        if residual.initial_values.len() != expected.len()
            || residual.initial_estimated_mask.len() != expected.len()
            || residual
                .initial_values
                .iter()
                .any(|value| !value.is_finite())
        {
            bail!("source residual initial values/mask for '{}' must match the component width and be finite", residual.output);
        }
        // The estimation status is immutable across the fit: initial and
        // current masks must agree.
        if residual.initial_estimated_mask != residual.estimated_mask {
            bail!(
                "source residual initial and final estimated masks for '{}' disagree",
                residual.output
            );
        }
        // Fixed components must preserve their declared initial value; free
        // components must have finite initials (may differ from finals).
        for (component_index, (initial, estimated)) in residual
            .initial_values
            .iter()
            .zip(&residual.estimated_mask)
            .enumerate()
        {
            if !estimated && !equal_with_roundoff(*initial, residual.values[component_index]) {
                bail!(
                        "source residual '{}' fixed component '{}' initial ({}) does not match final ({})",
                        residual.output,
                        residual.components[component_index],
                        initial,
                        residual.values[component_index],
                    );
            }
        }
        validate_residual_values(
            &residual.family,
            &residual.initial_values,
            &residual.initial_estimated_mask,
        )?;
    }
    validate_source_covariates(source)
}

fn validate_source_covariates(source: &ParametricSourceMetadata) -> Result<()> {
    if source.covariate_effects.is_empty() {
        if !source.subject_covariates.is_empty()
            || !source.subject_design.is_empty()
            || !source.subject_population_parameters.is_empty()
        {
            bail!("no-effect source metadata requires explicit empty N5 fields");
        }
        return Ok(());
    }
    let mut effect_names = std::collections::HashSet::new();
    let mut covariate_families = std::collections::HashMap::<&str, (&str, Option<f64>)>::new();
    for (order, effect) in source.covariate_effects.iter().enumerate() {
        if effect.order != order
            || effect.parameter_index >= source.parameters.len()
            || effect.parameter != source.parameters[effect.parameter_index].name
            || effect.name.is_empty()
            || effect.covariate.is_empty()
            || !effect.initial.is_finite()
            || !effect.estimate.is_finite()
            || (!effect.estimated && !equal_with_roundoff(effect.initial, effect.estimate))
        {
            bail!("source covariate-effect declaration is malformed or out of order");
        }
        let expected_name = match effect.family.as_str() {
            "continuous"
                if effect.center.is_some_and(f64::is_finite)
                    && effect.reference.is_none()
                    && effect.level.is_none() =>
            {
                format!("beta:{}:{}", effect.parameter, effect.covariate)
            }
            "categorical"
                if effect.center.is_none()
                    && effect.reference.is_some_and(f64::is_finite)
                    && effect.level.is_some_and(f64::is_finite)
                    && effect.reference != effect.level =>
            {
                format!(
                    "beta:{}:{}:{}",
                    effect.parameter,
                    effect.covariate,
                    stable_number(effect.level.unwrap())
                )
            }
            _ => bail!("source covariate-effect family metadata is malformed"),
        };
        if effect.name != expected_name || !effect_names.insert(effect.name.as_str()) {
            bail!("source covariate-effect canonical name is inconsistent or duplicated");
        }
        let family_key = if effect.family == "continuous" {
            ("continuous", effect.center)
        } else {
            ("categorical", effect.reference)
        };
        if covariate_families
            .insert(effect.covariate.as_str(), family_key)
            .is_some_and(|prior| prior != family_key)
        {
            bail!("source covariate family/center/reference declarations are inconsistent");
        }
    }
    let subject_count = source.subject_design.len();
    if subject_count == 0 {
        bail!("covariate source metadata requires subject design rows");
    }
    let parameter_count = source.parameters.len();
    if source.subject_population_parameters.len() != subject_count * parameter_count {
        bail!("source subject population parameter table has the wrong shape");
    }
    let mut covariate_names = source
        .covariate_effects
        .iter()
        .map(|effect| effect.covariate.as_str())
        .collect::<Vec<_>>();
    covariate_names.sort_unstable();
    covariate_names.dedup();
    if source.subject_covariates.len() != subject_count * covariate_names.len() {
        bail!("source subject covariate table has the wrong shape");
    }
    for (subject_index, design) in source.subject_design.iter().enumerate() {
        if design.subject.is_empty()
            || design.subject_index != subject_index
            || design.values.len() != source.covariate_effects.len()
        {
            bail!("source subject design order or width is malformed");
        }
        let subject_rows = &source.subject_covariates
            [subject_index * covariate_names.len()..(subject_index + 1) * covariate_names.len()];
        for (row, covariate) in subject_rows.iter().zip(&covariate_names) {
            if row.subject != design.subject
                || row.subject_index != subject_index
                || row.covariate != **covariate
                || !row.value.is_finite()
            {
                bail!("source subject covariate order/value metadata is malformed");
            }
        }
        let values = subject_rows
            .iter()
            .map(|row| (row.covariate.as_str(), row.value))
            .collect::<std::collections::HashMap<_, _>>();
        for (effect_index, effect) in source.covariate_effects.iter().enumerate() {
            let value = *values
                .get(effect.covariate.as_str())
                .context("source subject covariate value is missing")?;
            let expected = match effect.family.as_str() {
                "continuous" => value - effect.center.unwrap(),
                "categorical" => {
                    if value == effect.level.unwrap() {
                        1.0
                    } else if value == effect.reference.unwrap()
                        || source.covariate_effects.iter().any(|candidate| {
                            candidate.parameter == effect.parameter
                                && candidate.covariate == effect.covariate
                                && candidate.level == Some(value)
                        })
                    {
                        0.0
                    } else {
                        bail!("source subject has an unknown or incompletely declared categorical level");
                    }
                }
                _ => unreachable!(),
            };
            if !value.is_finite() || design.values[effect_index] != expected {
                bail!("source subject design does not reconstruct exactly from static values");
            }
        }
        let mut phi = source
            .parameters
            .iter()
            .map(|parameter| {
                parse_scale(&parameter.scale).map(|scale| psi_to_phi(parameter.estimate, scale))
            })
            .collect::<Result<Vec<_>>>()?;
        for (effect, design_value) in source.covariate_effects.iter().zip(&design.values) {
            phi[effect.parameter_index] += effect.estimate * design_value;
        }
        for (parameter_index, phi_value) in phi.iter().copied().enumerate() {
            let row = &source.subject_population_parameters
                [subject_index * parameter_count + parameter_index];
            let scale = parse_scale(&source.parameters[parameter_index].scale)?;
            if row.subject != design.subject
                || row.subject_index != subject_index
                || row.parameter_index != parameter_index
                || row.parameter != source.parameters[parameter_index].name
                || !equal_with_roundoff(row.phi, phi_value)
                || !equal_with_roundoff(row.psi, phi_to_psi(phi_value, scale))
            {
                bail!("source subject population phi/psi rows are inconsistent");
            }
        }
    }
    Ok(())
}

fn validate_source_effects(
    effects: &[ParametricSourceEffect],
    parameters: &[ParametricSourceParameter],
    label: &str,
) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for effect in effects {
        if effect.parameter_index >= parameters.len()
            || !seen.insert(effect.parameter_index)
            || effect.parameter_name != parameters[effect.parameter_index].name
        {
            bail!("source {label} indices/names are out of bounds, duplicated, or inconsistent");
        }
    }
    Ok(())
}

fn validate_source_covariance(
    covariance: &ParametricSourceCovariance,
    expected_names: &[&str],
    label: &str,
) -> Result<()> {
    let dimension = expected_names.len();
    if covariance.dimension != dimension
        || covariance.names.len() != dimension
        || covariance.values.len() != dimension
        || covariance.structural_mask.len() != dimension
        || covariance.estimated_mask.len() != dimension
        || covariance.values.iter().any(|row| row.len() != dimension)
        || covariance
            .structural_mask
            .iter()
            .chain(&covariance.estimated_mask)
            .any(|row| row.len() != dimension)
        || covariance
            .names
            .iter()
            .map(String::as_str)
            .ne(expected_names.iter().copied())
        || covariance.names.iter().any(|name| name.is_empty())
    {
        bail!("source {label} names, values, or masks do not match their declared dimension/order");
    }
    for row in 0..dimension {
        for column in 0..dimension {
            let value = covariance.values[row][column];
            if !value.is_finite()
                || value != covariance.values[column][row]
                || (row == column && !covariance.structural_mask[row][column])
                || covariance.structural_mask[row][column]
                    != covariance.structural_mask[column][row]
                || covariance.estimated_mask[row][column] != covariance.estimated_mask[column][row]
                || (covariance.estimated_mask[row][column]
                    && !covariance.structural_mask[row][column])
                || (!covariance.structural_mask[row][column] && value != 0.0)
            {
                bail!("source {label} values/masks must be finite, symmetric, structurally consistent, and estimated only where structural");
            }
        }
    }
    // Strict Cholesky: a nonempty persisted covariance is accepted only when
    // every pivot is finite and positive, without jitter or matrix repair.
    let mut lower = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = covariance.values[row][column];
            for (row_value, column_value) in
                lower[row][..column].iter().zip(&lower[column][..column])
            {
                value -= row_value * column_value;
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    bail!("source {label} final covariance must be strictly positive definite");
                }
                lower[row][column] = value.sqrt();
            } else {
                lower[row][column] = value / lower[column][column];
                if !lower[row][column].is_finite() {
                    bail!("source {label} final covariance must be strictly positive definite");
                }
            }
        }
    }
    // Immutable initial covariance is validated independently and without
    // repair: exact finite symmetry, exact declared structural zeros, positive
    // diagonals, and strict Cholesky SPD are all required.
    if covariance.initial_values.len() != dimension
        || covariance
            .initial_values
            .iter()
            .any(|row| row.len() != dimension)
    {
        bail!("source {label} initial_values do not match the declared dimension");
    }
    for row in 0..dimension {
        for column in 0..dimension {
            let initial = covariance.initial_values[row][column];
            if !initial.is_finite() {
                bail!("source {label} initial_values must be finite");
            }
            if initial != covariance.initial_values[column][row] {
                bail!("source {label} initial_values must be exactly symmetric");
            }
            if !covariance.structural_mask[row][column] && initial != 0.0 {
                bail!("source {label} initial structural-zero entries must be exactly zero");
            }
            if row == column && initial <= 0.0 {
                bail!("source {label} initial variances must be strictly positive");
            }
            if !covariance.estimated_mask[row][column]
                && !equal_with_roundoff(initial, covariance.values[row][column])
            {
                bail!("source {label} fixed entry ({row},{column}) initial does not match final value");
            }
        }
    }
    let mut initial_lower = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = covariance.initial_values[row][column];
            for (row_value, column_value) in initial_lower[row][..column]
                .iter()
                .zip(&initial_lower[column][..column])
            {
                value -= row_value * column_value;
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    bail!("source {label} initial covariance must be strictly positive definite");
                }
                initial_lower[row][column] = value.sqrt();
            } else {
                initial_lower[row][column] = value / initial_lower[column][column];
                if !initial_lower[row][column].is_finite() {
                    bail!("source {label} initial covariance must be strictly positive definite");
                }
            }
        }
    }
    Ok(())
}

fn validate_persisted_source_metadata(
    tables: &ParametricResultTables,
    source: &ParametricSourceMetadata,
) -> Result<()> {
    if tables.population.len() != source.parameters.len() {
        bail!("persisted population table width does not match source metadata");
    }
    for (index, (parameter, declaration)) in
        tables.population.iter().zip(&source.parameters).enumerate()
    {
        if parameter.name != declaration.name
            || declaration.parameter_index != index
            || parameter.scale != declaration.scale
            || !declaration.initial.is_finite()
            || !equal_with_roundoff(parameter.estimate, declaration.estimate)
            || parameter.estimated != declaration.estimated
            || parameter.iiv
                != source
                    .random_effects
                    .iter()
                    .any(|effect| effect.parameter_index == index)
            || parameter.iov
                != source
                    .iov_effects
                    .iter()
                    .any(|effect| effect.parameter_index == index)
            || !parameter.estimate.is_finite()
        {
            bail!("persisted population table disagrees with source metadata");
        }
        parse_scale(&parameter.scale)?;
    }
    validate_covariance_table(
        &tables.omega,
        &source.random_effects,
        &source.omega,
        "Omega",
    )?;
    match (&tables.omega_iov, &source.omega_iov) {
        (None, None) => {}
        (Some(rows), Some(covariance)) => {
            validate_covariance_table(rows, &source.iov_effects, covariance, "Omega_IOV")?
        }
        _ => bail!("persisted Omega_IOV table disagrees with source metadata"),
    }
    validate_persisted_residual_metadata(&tables.residual_error)?;
    let expected_rows = source
        .residual_outputs
        .iter()
        .flat_map(|residual| {
            residual
                .components
                .iter()
                .zip(&residual.values)
                .zip(&residual.estimated_mask)
                .map(move |((component, value), estimated)| {
                    (
                        residual.output.as_str(),
                        residual.output_index,
                        residual.family.as_str(),
                        component.as_str(),
                        *value,
                        *estimated,
                    )
                })
        })
        .collect::<Vec<_>>();
    if tables.residual_error.len() != expected_rows.len()
        || tables
            .residual_error
            .iter()
            .zip(expected_rows)
            .any(|(row, expected)| {
                (
                    row.output.as_str(),
                    row.output_index,
                    row.family.as_str(),
                    row.component.as_str(),
                    row.estimated,
                ) != (expected.0, expected.1, expected.2, expected.3, expected.5)
                    || !equal_with_roundoff(row.estimate, expected.4)
            })
    {
        bail!("persisted residual table disagrees with source metadata");
    }
    let expected_effect_rows = source
        .covariate_effects
        .iter()
        .map(|row| CovariateEffectRow {
            order: row.order,
            name: row.name.clone(),
            family: row.family.clone(),
            parameter: row.parameter.clone(),
            parameter_index: row.parameter_index,
            covariate: row.covariate.clone(),
            center: row.center,
            reference: row.reference,
            level: row.level,
            initial: row.initial,
            estimate: row.estimate,
            estimated: row.estimated,
        })
        .collect::<Vec<_>>();
    if tables.covariate_effects != expected_effect_rows
        || tables.subject_covariates != source.subject_covariates
        || tables.subject_population_parameters != source.subject_population_parameters
    {
        bail!("persisted N5 tables disagree with the independent source snapshot");
    }
    Ok(())
}

fn validate_covariance_table(
    rows: &[OmegaRow],
    effects: &[ParametricSourceEffect],
    covariance: &ParametricSourceCovariance,
    label: &str,
) -> Result<()> {
    let expected_len = covariance.dimension * (covariance.dimension + 1) / 2;
    if rows.len() != expected_len {
        bail!("persisted {label} table has the wrong lower-triangle width");
    }
    let mut offset = 0;
    for row in 0..covariance.dimension {
        for column in 0..=row {
            let entry = &rows[offset];
            if entry.row != effects[row].parameter_name
                || entry.column != effects[column].parameter_name
                || entry.structural != covariance.structural_mask[row][column]
                || entry.estimated != covariance.estimated_mask[row][column]
                || !equal_with_roundoff(entry.estimate, covariance.values[row][column])
            {
                bail!("persisted {label} table disagrees with source metadata");
            }
            offset += 1;
        }
    }
    let names = effects
        .iter()
        .map(|effect| effect.parameter_name.clone())
        .collect::<Vec<_>>();
    covariance_declaration(rows, &names, label)?;
    Ok(())
}

fn residual_component_names(family: &str) -> Result<Vec<String>> {
    match family {
        "constant" | "exponential" => Ok(vec!["sigma".to_string()]),
        "proportional" => Ok(vec!["proportional".to_string()]),
        "combined" => Ok(vec!["additive".to_string(), "proportional".to_string()]),
        "correlated_combined" => Ok(vec![
            "additive".to_string(),
            "proportional".to_string(),
            "correlation".to_string(),
        ]),
        family => bail!("unknown source residual-error family '{family}'"),
    }
}

fn validate_residual_values(family: &str, values: &[f64], estimated: &[bool]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) || values.len() != estimated.len() {
        bail!("source residual values/fixed status are malformed");
    }
    match family {
        "constant" | "proportional" | "exponential" if values == [values[0]] => {
            if values[0] <= 0.0 {
                bail!("source residual scale must be strictly positive");
            }
        }
        "combined" if values.len() == 2 => {
            if values.iter().any(|value| *value < 0.0)
                || values.iter().all(|value| *value == 0.0)
                || values
                    .iter()
                    .zip(estimated)
                    .any(|(value, estimated)| *estimated && *value <= 0.0)
            {
                bail!("source combined residual components are invalid");
            }
        }
        "correlated_combined" if values.len() == 3 => {
            if values[0] <= 0.0 || values[1] <= 0.0 || values[2] <= -1.0 || values[2] >= 1.0 {
                bail!("source correlated-combined residual components are invalid");
            }
        }
        _ => bail!("source residual family/value width is malformed"),
    }
    Ok(())
}

fn validate_persisted_residual_metadata(rows: &[ResidualErrorRow]) -> Result<()> {
    let mut seen_outputs = std::collections::HashSet::new();
    let mut previous_output_index = None;
    let mut offset = 0;
    while offset < rows.len() {
        let first = &rows[offset];
        if first.output.is_empty()
            || !seen_outputs.insert((first.output_index, first.output.as_str()))
        {
            bail!("persisted residual outputs must be non-empty and unique");
        }
        if previous_output_index.is_some_and(|previous| previous >= first.output_index) {
            bail!("persisted residual outputs are not in canonical output-index order");
        }
        previous_output_index = Some(first.output_index);
        let expected_components: &[&str] = match first.family.as_str() {
            "constant" | "exponential" => &["sigma"],
            "proportional" => &["proportional"],
            "combined" => &["additive", "proportional"],
            "correlated_combined" => &["additive", "proportional", "correlation"],
            family => bail!("unknown persisted residual-error family '{family}'"),
        };
        if offset + expected_components.len() > rows.len() {
            bail!("persisted residual-error family has missing components");
        }
        let group = &rows[offset..offset + expected_components.len()];
        for (row, expected_component) in group.iter().zip(expected_components) {
            if row.output_index != first.output_index
                || row.output != first.output
                || row.family != first.family
                || row.component != *expected_component
                || !row.estimate.is_finite()
            {
                bail!("persisted residual metadata is duplicated, reordered, or malformed");
            }
        }
        match first.family.as_str() {
            "constant" | "proportional" | "exponential" => {
                if group[0].estimate <= 0.0 {
                    bail!("persisted residual scale must be strictly positive");
                }
            }
            "combined" => {
                if group.iter().any(|row| row.estimate < 0.0)
                    || group.iter().all(|row| row.estimate == 0.0)
                    || group.iter().any(|row| row.estimated && row.estimate <= 0.0)
                {
                    bail!("persisted combined residual components are invalid");
                }
            }
            "correlated_combined" => {
                if group[0].estimate <= 0.0
                    || group[1].estimate <= 0.0
                    || group[2].estimate <= -1.0
                    || group[2].estimate >= 1.0
                {
                    bail!("persisted correlated-combined residual components are invalid");
                }
            }
            _ => unreachable!(),
        }
        offset += expected_components.len();
    }
    Ok(())
}

fn validate_persisted_information_shape(information: &InformationDiagnostics) -> Result<()> {
    let p = information.coordinates.len();
    let square = |matrix: &[Vec<f64>]| matrix.len() == p && matrix.iter().all(|row| row.len() == p);
    if information.delta.len() != p
        || !square(&information.g)
        || !square(&information.expected_complete_hessian)
        || !square(&information.observed_hessian)
        || !square(&information.observed_information)
    {
        bail!("persisted information diagnostics do not have exact p-dimensional shapes");
    }
    if information
        .coordinates
        .iter()
        .enumerate()
        .any(|(index, coordinate)| coordinate.index != index)
    {
        bail!("persisted information coordinate indices are not contiguous");
    }
    if matches!(
        information.status,
        crate::results::InformationStatus::Available
    ) && p == 0
    {
        bail!("persisted available information diagnostics require free coordinates");
    }
    if matches!(
        information.status,
        crate::results::InformationStatus::NoFreeCoordinates
    ) && p != 0
    {
        bail!("persisted no-free-coordinate information status has nonzero width");
    }
    if !matches!(
        information.status,
        crate::results::InformationStatus::NonFinite
    ) {
        let finite = information.delta.iter().all(|value| value.is_finite())
            && information
                .g
                .iter()
                .chain(&information.expected_complete_hessian)
                .chain(&information.observed_hessian)
                .chain(&information.observed_information)
                .flatten()
                .all(|value| value.is_finite());
        if !finite {
            bail!("persisted information diagnostics retain nonfinite values for their status");
        }
    }
    Ok(())
}

fn expected_information_coordinates(
    source: &ParametricSourceMetadata,
) -> Result<Vec<InformationCoordinate>> {
    let mut coordinates = Vec::new();
    let push = |coordinates: &mut Vec<InformationCoordinate>, name, kind| {
        coordinates.push(InformationCoordinate {
            index: coordinates.len(),
            name,
            kind,
        });
    };
    for (parameter_index, parameter) in source.parameters.iter().enumerate() {
        if parameter.estimated {
            push(
                &mut coordinates,
                format!("phi:{}", parameter.name),
                InformationCoordinateKind::Population { parameter_index },
            );
        }
    }
    for effect in &source.covariate_effects {
        if effect.estimated {
            push(
                &mut coordinates,
                effect.name.clone(),
                InformationCoordinateKind::CovariateEffect {
                    effect_index: effect.order,
                },
            );
        }
    }
    append_source_covariance_coordinates(
        &mut coordinates,
        &source.random_effects,
        &source.omega,
        false,
    );
    if let Some(covariance) = &source.omega_iov {
        append_source_covariance_coordinates(
            &mut coordinates,
            &source.iov_effects,
            covariance,
            true,
        );
    }
    for residual in &source.residual_outputs {
        for (component, estimated) in residual.components.iter().zip(&residual.estimated_mask) {
            if *estimated {
                push(
                    &mut coordinates,
                    format!("residual:{}:{component}", residual.output),
                    InformationCoordinateKind::Residual {
                        output_index: residual.output_index,
                        component: component.clone(),
                    },
                );
            }
        }
    }
    Ok(coordinates)
}

fn append_source_covariance_coordinates(
    coordinates: &mut Vec<InformationCoordinate>,
    effects: &[ParametricSourceEffect],
    covariance: &ParametricSourceCovariance,
    iov: bool,
) {
    for row in 0..covariance.dimension {
        for column in 0..=row {
            if !covariance.estimated_mask[row][column] {
                continue;
            }
            let prefix = if iov { "omega_iov" } else { "omega" };
            coordinates.push(InformationCoordinate {
                index: coordinates.len(),
                name: format!(
                    "{prefix}:{}:{}",
                    effects[row].parameter_name, effects[column].parameter_name
                ),
                kind: if iov {
                    InformationCoordinateKind::OmegaIov { row, column }
                } else {
                    InformationCoordinateKind::Omega { row, column }
                },
            });
        }
    }
}

fn information_criteria_equal(
    actual: &InformationCriteriaDiagnostics,
    expected: &InformationCriteriaDiagnostics,
) -> bool {
    actual.status == expected.status
        && actual.parameter_count == expected.parameter_count
        && actual.sample_size_convention == expected.sample_size_convention
        && actual.subject_count == expected.subject_count
        && optional_float_equal(actual.source_marginal_n2ll, expected.source_marginal_n2ll)
        && optional_float_equal(
            actual.source_marginal_n2ll_mcse,
            expected.source_marginal_n2ll_mcse,
        )
        && optional_float_equal(actual.aic, expected.aic)
        && optional_float_equal(actual.bic, expected.bic)
        && optional_float_equal(actual.aic_mcse, expected.aic_mcse)
        && optional_float_equal(actual.bic_mcse, expected.bic_mcse)
}

fn information_criteria_row_equal(
    actual: &InformationCriteriaRow,
    expected: &InformationCriteriaRow,
) -> bool {
    actual.status == expected.status
        && actual.sample_size_convention == expected.sample_size_convention
        && actual.subject_count == expected.subject_count
        && actual.population_parameter_count == expected.population_parameter_count
        && actual.covariate_parameter_count == expected.covariate_parameter_count
        && actual.omega_parameter_count == expected.omega_parameter_count
        && actual.omega_iov_parameter_count == expected.omega_iov_parameter_count
        && actual.residual_parameter_count == expected.residual_parameter_count
        && actual.free_parameter_count == expected.free_parameter_count
        && optional_float_equal(actual.source_marginal_n2ll, expected.source_marginal_n2ll)
        && optional_float_equal(
            actual.source_marginal_n2ll_mcse,
            expected.source_marginal_n2ll_mcse,
        )
        && optional_float_equal(actual.aic, expected.aic)
        && optional_float_equal(actual.bic, expected.bic)
        && optional_float_equal(actual.aic_mcse, expected.aic_mcse)
        && optional_float_equal(actual.bic_mcse, expected.bic_mcse)
        && actual.failure_reason == expected.failure_reason
}

fn marginal_likelihood_row_equal(
    actual: &MarginalLikelihoodRow,
    expected: &MarginalLikelihoodRow,
) -> bool {
    actual.scope == expected.scope
        && actual.subject == expected.subject
        && actual.method == expected.method
        && actual.status == expected.status
        && actual.samples_per_subject == expected.samples_per_subject
        && actual.seed == expected.seed
        && actual.degrees_of_freedom == expected.degrees_of_freedom
        && equal_with_roundoff(
            actual.covariance_scale_multiplier,
            expected.covariance_scale_multiplier,
        )
        && actual.proposal_scale_source == expected.proposal_scale_source
        && actual.dimension == expected.dimension
        && actual.occasion_indices == expected.occasion_indices
        && json_float_vec_equal(&actual.mode, &expected.mode)
        && actual.mode_converged == expected.mode_converged
        && optional_float_equal(
            actual.log_marginal_likelihood,
            expected.log_marginal_likelihood,
        )
        && optional_float_equal(actual.n2ll, expected.n2ll)
        && optional_float_equal(actual.n2ll_mcse, expected.n2ll_mcse)
        && optional_float_equal(actual.effective_sample_size, expected.effective_sample_size)
        && optional_float_equal(
            actual.effective_sample_fraction,
            expected.effective_sample_fraction,
        )
        && actual.zero_weight_count == expected.zero_weight_count
        && actual.failure == expected.failure
}

fn json_float_vec_equal(actual: &str, expected: &str) -> bool {
    let Ok(actual) = serde_json::from_str::<Vec<f64>>(actual) else {
        return false;
    };
    let Ok(expected) = serde_json::from_str::<Vec<f64>>(expected) else {
        return false;
    };
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| equal_with_roundoff(*actual, expected))
}

fn statistic_row_equal(actual: &StatisticRow, expected: &StatisticRow) -> bool {
    actual.cycle == expected.cycle
        && actual.kind == expected.kind
        && actual.name == expected.name
        && actual.row == expected.row
        && actual.column == expected.column
        && actual.output_index == expected.output_index
        && actual.component == expected.component
        && optional_float_equal(actual.value, expected.value)
        && actual.status == expected.status
}

fn optional_float_equal(actual: Option<f64>, expected: Option<f64>) -> bool {
    match (actual, expected) {
        (Some(actual), Some(expected)) => equal_with_roundoff(actual, expected),
        (None, None) => true,
        _ => false,
    }
}

fn persisted_effect_matrix(
    rows: &[IndividualEffectRow],
    source: &str,
    effect_kind: &str,
    effect_names: &[String],
) -> Result<Vec<Vec<f64>>> {
    let mut grouped =
        std::collections::BTreeMap::<(String, Option<usize>), Vec<Option<f64>>>::new();
    for row in rows
        .iter()
        .filter(|row| row.source == source && row.effect_kind == effect_kind)
    {
        let effect_index = effect_names
            .iter()
            .position(|name| name == &row.parameter)
            .context("persisted individual-effect row has an unknown effect name")?;
        let values = grouped
            .entry((row.subject.clone(), row.occasion))
            .or_insert_with(|| vec![None; effect_names.len()]);
        if values[effect_index].replace(row.value).is_some() {
            bail!("persisted individual-effect rows contain a duplicate effect");
        }
    }
    grouped
        .into_values()
        .map(|values| {
            values
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .context("persisted individual-effect row is missing an effect")
        })
        .collect()
}

fn shrinkage_value_equal(actual: &ShrinkageValue, expected: &ShrinkageValue) -> bool {
    match (actual, expected) {
        (
            ShrinkageValue::Available {
                value: actual_value,
                unit_count: actual_count,
                denominator_documentation: actual_documentation,
            },
            ShrinkageValue::Available {
                value: expected_value,
                unit_count: expected_count,
                denominator_documentation: expected_documentation,
            },
        ) => {
            equal_with_roundoff(*actual_value, *expected_value)
                && actual_count == expected_count
                && actual_documentation == expected_documentation
        }
        (
            ShrinkageValue::Unavailable {
                reason: actual_reason,
            },
            ShrinkageValue::Unavailable {
                reason: expected_reason,
            },
        ) => actual_reason == expected_reason,
        _ => false,
    }
}

fn is_n6_statistic(kind: &str) -> bool {
    kind.starts_with("population_uncertainty_")
        || kind.starts_with("conditional_curvature_")
        || matches!(
            kind,
            "conditional_latent_se"
                | "conditional_hessian"
                | "conditional_latent_covariance"
                | "eta_shrinkage_posterior_mean"
                | "eta_shrinkage_map"
                | "kappa_shrinkage_posterior_mean"
                | "kappa_shrinkage_map"
        )
}

fn validate_persisted_n6(record: &ParametricResultRecord) -> Result<()> {
    let expected_population =
        crate::estimation::parametric::information::derive_population_uncertainty(
            &record.information_diagnostics,
        );
    if record.population_uncertainty != expected_population {
        bail!("persisted population uncertainty is inconsistent with observed information");
    }
    let population_status = serde_json::to_string(&record.population_uncertainty.status)?;
    let status_rows = record
        .tables
        .statistics
        .iter()
        .filter(|row| row.kind == "population_uncertainty_status")
        .collect::<Vec<_>>();
    if status_rows.len() != 1
        || status_rows[0].value.is_some()
        || status_rows[0].status.as_deref() != Some(population_status.as_str())
    {
        bail!("persisted population uncertainty status row is inconsistent");
    }
    let regularization_rows = record
        .tables
        .statistics
        .iter()
        .filter(|row| row.kind == "population_uncertainty_regularization")
        .collect::<Vec<_>>();
    if regularization_rows.len() != 1
        || regularization_rows[0].value.is_some()
        || regularization_rows[0].status.as_deref() != Some("none")
    {
        bail!("persisted population uncertainty regularization row is inconsistent");
    }
    let covariance_rows = record
        .tables
        .statistics
        .iter()
        .filter(|row| row.kind == "population_uncertainty_covariance_phi")
        .collect::<Vec<_>>();
    match record.population_uncertainty.free_covariance.as_ref() {
        Some(covariance) => {
            let width = record.population_uncertainty.coordinates.len();
            if covariance_rows.len() != width * width {
                bail!("persisted population uncertainty covariance rows are incomplete");
            }
            for (row_index, values) in covariance.iter().enumerate() {
                for (column_index, expected) in values.iter().enumerate() {
                    let offset = row_index * width + column_index;
                    let row = covariance_rows[offset];
                    if row.row.as_deref()
                        != Some(
                            record.population_uncertainty.coordinates[row_index]
                                .name
                                .as_str(),
                        )
                        || row.column.as_deref()
                            != Some(
                                record.population_uncertainty.coordinates[column_index]
                                    .name
                                    .as_str(),
                            )
                        || row
                            .value
                            .is_none_or(|actual| !equal_with_roundoff(actual, *expected))
                    {
                        bail!("persisted population uncertainty covariance row is inconsistent");
                    }
                }
            }
        }
        None if !covariance_rows.is_empty() => {
            bail!("persisted unavailable population uncertainty contains covariance rows");
        }
        None => {}
    }

    if !record.conditional_modes.is_empty()
        && record.conditional_modes.len() != record.subject_count
    {
        bail!("persisted conditional-mode count does not match subject count");
    }
    let eta_width = record.source_metadata.random_effects.len();
    let kappa_width = record.source_metadata.iov_effects.len();
    for mode in &record.conditional_modes {
        let dimension = eta_width + mode.kappas.len() * kappa_width;
        let diagnostics = &mode.uncertainty;
        if diagnostics.coordinates.len() != dimension
            || diagnostics
                .coordinates
                .iter()
                .enumerate()
                .any(|(index, coordinate)| coordinate.index != index)
            || diagnostics.mode_metadata.converged != mode.converged
            || diagnostics.mode_metadata.iterations != mode.iterations
            || diagnostics.mode_metadata.termination_message != mode.termination
            || !equal_with_roundoff(diagnostics.mode_metadata.objective_value, mode.objective)
        {
            bail!("persisted conditional curvature metadata is inconsistent");
        }
        if mode.eta.len() != eta_width
            || mode.kappas.iter().any(|kappa| {
                kappa.subject_id != mode.subject_id || kappa.values.len() != kappa_width
            })
        {
            bail!("persisted conditional mode latent widths are inconsistent");
        }
        for (effect_index, effect) in record.source_metadata.random_effects.iter().enumerate() {
            let coordinate = &diagnostics.coordinates[effect_index];
            let expected_sd =
                record.source_metadata.omega.values[effect_index][effect_index].sqrt();
            if coordinate.name != format!("eta:{}", effect.parameter_name)
                || !matches!(
                    coordinate.kind,
                    crate::estimation::parametric::JointLatentCoordinateKind::Eta {
                        parameter_index
                    } if parameter_index == effect.parameter_index
                )
                || !equal_with_roundoff(coordinate.prior_sd, expected_sd)
            {
                bail!("persisted eta curvature coordinate metadata is inconsistent");
            }
        }
        let omega_iov = record.source_metadata.omega_iov.as_ref();
        for (occasion_position, kappa) in mode.kappas.iter().enumerate() {
            for (effect_index, effect) in record.source_metadata.iov_effects.iter().enumerate() {
                let coordinate_index = eta_width + occasion_position * kappa_width + effect_index;
                let coordinate = &diagnostics.coordinates[coordinate_index];
                let expected_sd = omega_iov
                    .and_then(|omega| omega.values.get(effect_index))
                    .and_then(|row| row.get(effect_index))
                    .copied()
                    .context("persisted kappa coordinate lacks an Omega_IOV variance")?
                    .sqrt();
                if coordinate.name
                    != format!("kappa:{}:{}", kappa.occasion_index, effect.parameter_name)
                    || !matches!(
                        coordinate.kind,
                        crate::estimation::parametric::JointLatentCoordinateKind::Kappa {
                            occasion_index,
                            effect_index: actual_effect_index,
                            parameter_index,
                        } if occasion_index == kappa.occasion_index
                            && actual_effect_index == effect_index
                            && parameter_index == effect.parameter_index
                    )
                    || !equal_with_roundoff(coordinate.prior_sd, expected_sd)
                {
                    bail!("persisted kappa curvature coordinate metadata is inconsistent");
                }
            }
        }
        let expected_status = serde_json::to_string(&diagnostics.status)?;
        let status_rows = record
            .tables
            .statistics
            .iter()
            .filter(|row| row.kind == "conditional_curvature_status" && row.name == mode.subject_id)
            .collect::<Vec<_>>();
        if status_rows.len() != 1
            || status_rows[0].value.is_some()
            || status_rows[0].status.as_deref() != Some(expected_status.as_str())
        {
            bail!("persisted conditional curvature status row is inconsistent");
        }
        let regularization_rows = record
            .tables
            .statistics
            .iter()
            .filter(|row| {
                row.kind == "conditional_curvature_regularization" && row.name == mode.subject_id
            })
            .collect::<Vec<_>>();
        if regularization_rows.len() != 1
            || regularization_rows[0].value.is_some()
            || regularization_rows[0].status.as_deref() != Some("none")
        {
            bail!("persisted conditional curvature regularization row is inconsistent");
        }
        match diagnostics.status {
            crate::estimation::parametric::ConditionalCurvatureStatus::Available => {
                let square = |matrix: &[Vec<f64>]| {
                    matrix.len() == dimension
                        && matrix.iter().all(|row| {
                            row.len() == dimension && row.iter().all(|value| value.is_finite())
                        })
                };
                if diagnostics
                    .hessian
                    .as_deref()
                    .is_none_or(|matrix| !square(matrix))
                    || diagnostics
                        .latent_covariance
                        .as_deref()
                        .is_none_or(|matrix| !square(matrix))
                    || diagnostics
                        .latent_standard_errors
                        .as_ref()
                        .is_none_or(|values| {
                            values.len() != dimension
                                || values
                                    .iter()
                                    .any(|value| !value.is_finite() || *value <= 0.0)
                        })
                    || diagnostics.finite_difference_steps.len() != dimension
                    || diagnostics
                        .finite_difference_steps
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
                    || diagnostics
                        .spectral_condition_number
                        .is_none_or(|value| !value.is_finite() || value < 1.0)
                {
                    bail!("persisted available conditional curvature has malformed numerics");
                }
            }
            crate::estimation::parametric::ConditionalCurvatureStatus::Unavailable(_) => {
                if diagnostics.hessian.is_some()
                    || diagnostics.latent_covariance.is_some()
                    || diagnostics.latent_standard_errors.is_some()
                    || diagnostics.spectral_condition_number.is_some()
                {
                    bail!("persisted unavailable conditional curvature contains numerics");
                }
            }
        }
    }

    let eta_names = record
        .source_metadata
        .random_effects
        .iter()
        .map(|effect| effect.parameter_name.clone())
        .collect::<Vec<_>>();
    let kappa_names = record
        .source_metadata
        .iov_effects
        .iter()
        .map(|effect| effect.parameter_name.clone())
        .collect::<Vec<_>>();
    if record.shrinkage.eta_posterior_mean.len() != eta_names.len()
        || record.shrinkage.eta_map.len() != eta_names.len()
        || record.shrinkage.kappa_posterior_mean.len() != kappa_names.len()
        || record.shrinkage.kappa_map.len() != kappa_names.len()
        || record
            .shrinkage
            .eta_posterior_mean
            .iter()
            .zip(&eta_names)
            .any(|(value, name)| value.effect != *name)
        || record
            .shrinkage
            .eta_map
            .iter()
            .zip(&eta_names)
            .any(|(value, name)| value.effect != *name)
        || record
            .shrinkage
            .kappa_posterior_mean
            .iter()
            .zip(&kappa_names)
            .any(|(value, name)| value.effect != *name)
        || record
            .shrinkage
            .kappa_map
            .iter()
            .zip(&kappa_names)
            .any(|(value, name)| value.effect != *name)
    {
        bail!("persisted shrinkage effect ordering is inconsistent");
    }

    let eta_variances = (0..eta_names.len())
        .map(|index| record.source_metadata.omega.values[index][index])
        .collect::<Vec<_>>();
    let kappa_variances = record
        .source_metadata
        .omega_iov
        .as_ref()
        .map(|omega| {
            (0..kappa_names.len())
                .map(|index| omega.values[index][index])
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let eta_posterior_rows = persisted_effect_matrix(
        &record.tables.individual_effects,
        "chain_mean",
        "eta",
        &eta_names,
    )?;
    let eta_map_rows = persisted_effect_matrix(
        &record.tables.individual_effects,
        "conditional_mode",
        "eta",
        &eta_names,
    )?;
    let kappa_posterior_rows = persisted_effect_matrix(
        &record.tables.individual_effects,
        "chain_mean",
        "kappa",
        &kappa_names,
    )?;
    let kappa_map_rows = persisted_effect_matrix(
        &record.tables.individual_effects,
        "conditional_mode",
        "kappa",
        &kappa_names,
    )?;
    let expected_shrinkage = ShrinkageDiagnostics {
        eta_posterior_mean: derive_eta_posterior_mean_shrinkage(
            &eta_names,
            &eta_variances,
            &eta_posterior_rows,
        ),
        eta_map: derive_eta_map_shrinkage(
            &eta_names,
            &eta_variances,
            (!record.conditional_modes.is_empty()).then_some(eta_map_rows.as_slice()),
        ),
        kappa_posterior_mean: derive_kappa_posterior_mean_shrinkage(
            &kappa_names,
            &kappa_variances,
            &kappa_posterior_rows,
        ),
        kappa_map: derive_kappa_map_shrinkage(
            &kappa_names,
            &kappa_variances,
            (!record.conditional_modes.is_empty()).then_some(kappa_map_rows.as_slice()),
        ),
    };
    macro_rules! validate_shrinkage_group {
        ($actual:expr, $expected:expr) => {
            if $actual.len() != $expected.len()
                || $actual.iter().zip($expected).any(|(actual, expected)| {
                    actual.effect != expected.effect
                        || !shrinkage_value_equal(&actual.shrinkage, &expected.shrinkage)
                })
            {
                bail!("persisted shrinkage values are inconsistent with retained effects");
            }
        };
    }
    validate_shrinkage_group!(
        &record.shrinkage.eta_posterior_mean,
        &expected_shrinkage.eta_posterior_mean
    );
    validate_shrinkage_group!(&record.shrinkage.eta_map, &expected_shrinkage.eta_map);
    validate_shrinkage_group!(
        &record.shrinkage.kappa_posterior_mean,
        &expected_shrinkage.kappa_posterior_mean
    );
    validate_shrinkage_group!(&record.shrinkage.kappa_map, &expected_shrinkage.kappa_map);

    let cycle = record
        .tables
        .iterations
        .last()
        .map_or(0, |iteration| iteration.cycle);
    let expected_statistics = n6_uncertainty_statistic_rows(
        cycle,
        &record.population_uncertainty,
        &record.conditional_modes,
        &record.shrinkage,
    );
    let actual_statistics = record
        .tables
        .statistics
        .iter()
        .filter(|row| is_n6_statistic(&row.kind))
        .collect::<Vec<_>>();
    if actual_statistics.len() != expected_statistics.len()
        || actual_statistics
            .iter()
            .zip(&expected_statistics)
            .any(|(actual, expected)| !statistic_row_equal(actual, expected))
    {
        bail!("persisted N6 statistic rows are inconsistent");
    }
    Ok(())
}

impl ParametricResultRecord {
    pub fn read_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("failed to open parametric result '{}'", path.display()))?;
        let raw: serde_json::Value = serde_json::from_reader(file)
            .with_context(|| format!("failed to parse parametric result '{}'", path.display()))?;
        let object = raw
            .as_object()
            .context("parametric result JSON must be an object")?;
        for required in [
            "marginal_likelihood",
            "information_criteria",
            "subject_count",
            "population_uncertainty",
            "conditional_modes",
            "shrinkage",
        ] {
            if !object.contains_key(required) {
                bail!("schema-9 parametric result requires the {required} field");
            }
        }
        object
            .get("source_metadata")
            .and_then(serde_json::Value::as_object)
            .context("schema-9 parametric result requires an object source_metadata field")?;
        let config = object
            .get("config")
            .and_then(serde_json::Value::as_object)
            .context("schema-9 parametric result requires an object config field")?;
        if !config.contains_key("marginal_likelihood") {
            bail!("schema-9 parametric result requires config.marginal_likelihood");
        }
        let record: Self = serde_json::from_value(raw)
            .with_context(|| format!("failed to parse parametric result '{}'", path.display()))?;
        if record.schema_version != PARAMETRIC_RESULT_SCHEMA_VERSION {
            bail!(
                "unsupported parametric result schema version {}",
                record.schema_version
            );
        }
        if record.fit_family != "parametric" || record.algorithm != "saem" {
            bail!("JSON record is not a parametric SAEM result");
        }
        if record.objective_kind != "conditional_n2ll" {
            bail!(
                "unsupported parametric objective kind '{}'",
                record.objective_kind
            );
        }
        record
            .config
            .validate()
            .context("schema-9 parametric result contains invalid retained SAEM configuration")?;
        validate_persisted_marginal_likelihood(&record)?;
        validate_persisted_information_criteria(&record)?;
        validate_persisted_n6(&record)?;
        Ok(record)
    }

    /// Reconstruct a typed problem initialized from this persisted fit.
    ///
    /// This starts a new run from the final scientific estimates. Sampler,
    /// adaptation, cycle, and random-number-generator state are not persisted
    /// or continued. The retained [`SaemConfig`] describes the parent run only.
    pub fn warm_start_problem<E>(
        &self,
        equation: E,
        data: Data,
    ) -> Result<EstimationProblem<E, Parametric>>
    where
        E: Equation + EquationMetadataSource,
    {
        self.validate_warm_start_header()?;
        problem_from_tables(equation, data, &self.tables)
    }

    fn validate_warm_start_header(&self) -> Result<()> {
        if self.schema_version != PARAMETRIC_RESULT_SCHEMA_VERSION {
            bail!(
                "unsupported parametric result schema version {}",
                self.schema_version
            );
        }
        if self.fit_family != "parametric" {
            bail!("result fit family '{}' is not parametric", self.fit_family);
        }
        if self.algorithm != "saem" {
            bail!("result algorithm '{}' is not SAEM", self.algorithm);
        }
        self.config.validate()?;
        validate_persisted_marginal_likelihood(self)?;
        validate_persisted_information_criteria(self)?;
        validate_persisted_n6(self)?;
        Ok(())
    }

    fn write_json(&self, path: &Path) -> Result<()> {
        create_parent_dir(path)?;
        let file =
            File::create(path).with_context(|| format!("failed to create '{}'", path.display()))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("failed to write '{}'", path.display()))
    }
}

#[derive(Serialize)]
struct ParametricManifest {
    schema_version: u32,
    fit_family: &'static str,
    algorithm: &'static str,
    objective_kind: &'static str,
    termination: Option<StopReason>,
    operational_convergence: OperationalConvergenceDiagnostics,
    estimator_metadata: SaemEstimatorMetadata,
    marginal_likelihood: Option<MarginalLikelihoodDiagnostics>,
    information_criteria: InformationCriteriaDiagnostics,
    files: Vec<String>,
}

impl<E: Equation> ParametricResult<E> {
    /// Build a new typed problem initialized from this fit's final estimates.
    ///
    /// The parent result is borrowed and remains unchanged. This is a new run,
    /// not an exact continuation: latent chains, proposal adaptation, cycle
    /// state, and random-number-generator streams are not carried forward.
    pub fn warm_start_problem(&self) -> Result<EstimationProblem<E, Parametric>>
    where
        E: Clone + EquationMetadataSource,
    {
        problem_from_tables(
            self.equation.clone(),
            self.data.clone(),
            &self.base_tables()?,
        )
    }

    /// Fit a new SAEM run initialized from this result's final estimates.
    ///
    /// `config`, including its seed and schedule, belongs entirely to the new
    /// run. The parent configuration and sampler state are not reused.
    pub fn fit_next(&self, config: SaemConfig) -> Result<ParametricResult<E>>
    where
        E: Clone + EquationMetadataSource + Send + 'static,
    {
        self.warm_start_problem()?.fit_with(config)
    }

    pub fn tables(&self, idelta: f64, tad: f64) -> Result<ParametricResultTables>
    where
        E: pharmsol::equation::EquationTypes<P = SubjectPredictions>,
    {
        let mut tables = self.base_tables()?;
        tables.predictions = self.prediction_rows(idelta, tad)?;
        Ok(tables)
    }

    pub fn write_json(&self, path: impl AsRef<Path>, idelta: f64, tad: f64) -> Result<()>
    where
        E: pharmsol::equation::EquationTypes<P = SubjectPredictions>,
    {
        self.record(self.tables(idelta, tad)?)?
            .write_json(path.as_ref())
    }

    pub fn write_outputs(&self, directory: impl AsRef<Path>, idelta: f64, tad: f64) -> Result<()>
    where
        E: pharmsol::equation::EquationTypes<P = SubjectPredictions>,
    {
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory).with_context(|| {
            format!(
                "failed to create output directory '{}'",
                directory.display()
            )
        })?;
        let tables = self.tables(idelta, tad)?;
        let mut files = vec![
            "population.csv".to_string(),
            "omega.csv".to_string(),
            "residual_error.csv".to_string(),
            "individual_effects.csv".to_string(),
            "individual_parameters.csv".to_string(),
            "iterations.csv".to_string(),
            "statistics.csv".to_string(),
            "marginal_likelihood.csv".to_string(),
            "information_criteria.csv".to_string(),
            "predictions.csv".to_string(),
            "covariate_effects.csv".to_string(),
            "subject_covariates.csv".to_string(),
            "subject_population_parameters.csv".to_string(),
        ];
        tables.write_population(directory.join("population.csv"))?;
        tables.write_omega(directory.join("omega.csv"))?;
        if tables.omega_iov.is_some() {
            tables.write_omega_iov(directory.join("omega_iov.csv"))?;
            files.push("omega_iov.csv".to_string());
        }
        tables.write_residual_error(directory.join("residual_error.csv"))?;
        tables.write_individual_effects(directory.join("individual_effects.csv"))?;
        tables.write_individual_parameters(directory.join("individual_parameters.csv"))?;
        tables.write_iterations(directory.join("iterations.csv"))?;
        tables.write_statistics(directory.join("statistics.csv"))?;
        tables.write_marginal_likelihood(directory.join("marginal_likelihood.csv"))?;
        tables.write_information_criteria(directory.join("information_criteria.csv"))?;
        tables.write_predictions(directory.join("predictions.csv"))?;
        tables.write_covariate_effects(directory.join("covariate_effects.csv"))?;
        tables.write_subject_covariates(directory.join("subject_covariates.csv"))?;
        tables.write_subject_population_parameters(
            directory.join("subject_population_parameters.csv"),
        )?;

        let record = self.record(tables)?;
        record.write_json(&directory.join("result.json"))?;
        files.push("result.json".to_string());
        files.push("manifest.json".to_string());
        let manifest = ParametricManifest {
            schema_version: PARAMETRIC_RESULT_SCHEMA_VERSION,
            fit_family: "parametric",
            algorithm: "saem",
            objective_kind: "conditional_n2ll",
            termination: self.termination_reason().cloned(),
            operational_convergence: self.operational_convergence().clone(),
            estimator_metadata: self.estimator_metadata().clone(),
            marginal_likelihood: self.marginal_likelihood_diagnostics().cloned(),
            information_criteria: self.information_criteria().clone(),
            files,
        };
        let manifest_path = directory.join("manifest.json");
        let file = File::create(&manifest_path)
            .with_context(|| format!("failed to create '{}'", manifest_path.display()))?;
        serde_json::to_writer_pretty(file, &manifest)
            .with_context(|| format!("failed to write '{}'", manifest_path.display()))?;
        tracing::info!(directory = %directory.display(), "wrote parametric result outputs");
        Ok(())
    }

    fn record(&self, tables: ParametricResultTables) -> Result<ParametricResultRecord> {
        let source_metadata = self.source_metadata(&tables)?;
        validate_source_snapshot(&source_metadata)?;
        validate_persisted_source_metadata(&tables, &source_metadata)?;
        Ok(ParametricResultRecord {
            schema_version: PARAMETRIC_RESULT_SCHEMA_VERSION,
            fit_family: "parametric".to_string(),
            algorithm: "saem".to_string(),
            config: self.config.clone(),
            effective_n_chains: self.effective_n_chains,
            termination: self.termination_reason.clone(),
            objective_kind: "conditional_n2ll".to_string(),
            conditional_n2ll: self.conditional_n2ll(),
            subject_count: self.data.subjects().len(),
            marginal_likelihood: self.marginal_likelihood_diagnostics().cloned(),
            information_criteria: self.information_criteria().clone(),
            source_metadata,
            warnings: self.warnings.iter().map(warning_record).collect(),
            tables,
            information_diagnostics: self.information_diagnostics().clone(),
            population_uncertainty: self.population_uncertainty().clone(),
            conditional_modes: self.conditional_modes().to_vec(),
            shrinkage: self.shrinkage().clone(),
            markov_simulation_variance: self.markov_simulation_variance().clone(),
            operational_convergence: self.operational_diagnostics().clone(),
            estimator_metadata: self.estimator_metadata().clone(),
        })
    }

    fn source_metadata(&self, tables: &ParametricResultTables) -> Result<ParametricSourceMetadata> {
        let effects = |indices: &[usize], names: &[String]| {
            indices
                .iter()
                .zip(names)
                .map(|(parameter_index, parameter_name)| ParametricSourceEffect {
                    parameter_index: *parameter_index,
                    parameter_name: parameter_name.clone(),
                })
                .collect()
        };
        let covariance =
            |names: &[String],
             values: &ndarray::Array2<f64>,
             structural: &ndarray::Array2<bool>,
             estimated: &ndarray::Array2<bool>,
             initial: &ndarray::Array2<f64>| ParametricSourceCovariance {
                dimension: values.nrows(),
                names: names.to_vec(),
                values: numeric_matrix_rows(values),
                structural_mask: boolean_mask_rows(structural),
                estimated_mask: boolean_mask_rows(estimated),
                initial_values: numeric_matrix_rows(initial),
            };
        if self.population_initial.len() != tables.population.len() {
            bail!("result population initial declarations do not match the parameter width");
        }
        if self.residual_initial_values.len() != self.residual_error_estimates.len()
            || self.residual_initial_estimated.len() != self.residual_error_estimates.len()
        {
            bail!("result residual initial declarations do not match the output width");
        }
        let omega_iov = match (
            self.omega_iov.as_ref(),
            self.omega_iov_structural_mask.as_ref(),
            self.omega_iov_estimated_mask.as_ref(),
            self.omega_iov_initial.as_ref(),
        ) {
            (None, None, None, None) => None,
            (Some(values), Some(structural), Some(estimated), Some(initial)) => Some(covariance(
                &self.iov_effect_names,
                values,
                structural,
                estimated,
                initial,
            )),
            _ => bail!("result Omega_IOV final and immutable initial declarations are incomplete"),
        };
        Ok(ParametricSourceMetadata {
            parameters: tables
                .population
                .iter()
                .enumerate()
                .map(|(parameter_index, row)| ParametricSourceParameter {
                    name: row.name.clone(),
                    parameter_index,
                    scale: row.scale.clone(),
                    initial: self.population_initial[parameter_index],
                    estimate: row.estimate,
                    estimated: row.estimated,
                })
                .collect(),
            random_effects: effects(&self.random_effect_indices, &self.random_effect_names),
            omega: covariance(
                &self.random_effect_names,
                &self.omega,
                &self.omega_structural_mask,
                &self.omega_estimated_mask,
                &self.omega_initial,
            ),
            iov_effects: effects(&self.iov_effect_indices, &self.iov_effect_names),
            omega_iov,
            residual_outputs: self
                .residual_error_estimates
                .iter()
                .enumerate()
                .map(|(estimate_index, estimate)| {
                    let components = residual_components(
                        estimate.model,
                        estimate.estimated,
                        estimate.combined_additive_estimated,
                        estimate.combined_proportional_estimated,
                        estimate.correlation_estimated,
                    );
                    let initial_values = self.residual_initial_values[estimate_index].clone();
                    let initial_estimated = self.residual_initial_estimated[estimate_index].clone();
                    ParametricSourceResidual {
                        output: estimate.output.clone(),
                        output_index: estimate.output_index,
                        family: residual_family(estimate.model).to_string(),
                        components: components
                            .iter()
                            .map(|(name, _, _)| (*name).to_string())
                            .collect(),
                        values: components.iter().map(|(_, value, _)| *value).collect(),
                        estimated_mask: components
                            .iter()
                            .map(|(_, _, estimated)| *estimated)
                            .collect(),
                        initial_values,
                        initial_estimated_mask: initial_estimated,
                    }
                })
                .collect(),
            covariate_effects: tables
                .covariate_effects
                .iter()
                .map(|row| ParametricSourceCovariateEffect {
                    order: row.order,
                    name: row.name.clone(),
                    family: row.family.clone(),
                    parameter: row.parameter.clone(),
                    parameter_index: row.parameter_index,
                    covariate: row.covariate.clone(),
                    center: row.center,
                    reference: row.reference,
                    level: row.level,
                    initial: row.initial,
                    estimate: row.estimate,
                    estimated: row.estimated,
                })
                .collect(),
            subject_covariates: tables.subject_covariates.clone(),
            subject_design: self
                .covariates()
                .map(|model| {
                    model
                        .subject_design()
                        .iter()
                        .enumerate()
                        .map(|(subject_index, row)| ParametricSourceSubjectDesign {
                            subject: row.subject().to_string(),
                            subject_index,
                            values: row.values().to_vec(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            subject_population_parameters: tables.subject_population_parameters.clone(),
        })
    }

    fn base_tables(&self) -> Result<ParametricResultTables> {
        self.validate_output_metadata()?;
        let population = self
            .parameter_names
            .iter()
            .enumerate()
            .map(|(index, name)| PopulationParameterRow {
                name: name.clone(),
                estimate: self.population_estimates[index],
                scale: scale_text(self.parameter_scales[index]),
                estimated: self.estimated_parameters[index],
                iiv: self.random_effect_indices.contains(&index),
                iov: self.iov_effect_indices.contains(&index),
            })
            .collect();
        let omega = covariance_rows(
            &self.random_effect_names,
            &self.omega,
            &self.omega_structural_mask,
            &self.omega_estimated_mask,
            "Omega",
        )?;
        let omega_iov = match (
            self.omega_iov.as_ref(),
            self.omega_iov_structural_mask.as_ref(),
            self.omega_iov_estimated_mask.as_ref(),
        ) {
            (Some(matrix), Some(structural), Some(estimated)) => Some(covariance_rows(
                &self.iov_effect_names,
                matrix,
                structural,
                estimated,
                "Omega_IOV",
            )?),
            (None, None, None) => None,
            _ => bail!("Omega_IOV matrix and masks must either all be present or all be absent"),
        };
        let residual_error = residual_rows(&self.residual_error_estimates);
        let individual_effects = self.individual_effect_rows()?;
        let individual_parameters = self.individual_parameter_rows()?;
        let iterations = self.iteration_rows();
        let mut statistics = self.statistic_rows()?;
        if self.estimator_metadata().averaged_iterations > 0 {
            for (name, value) in self.parameter_names.iter().zip(&self.population_estimates) {
                statistics.push(statistic(
                    self.iterations,
                    "theta_final",
                    name,
                    None,
                    None,
                    None,
                    None,
                    *value,
                ));
            }
            append_covariance_statistics(
                &mut statistics,
                self.iterations,
                "omega_final",
                &self.random_effect_names,
                &self.omega,
            )?;
            if let Some(matrix) = self.omega_iov.as_ref() {
                append_covariance_statistics(
                    &mut statistics,
                    self.iterations,
                    "omega_iov_final",
                    &self.iov_effect_names,
                    matrix,
                )?;
            }
            for residual in &self.residual_error_estimates {
                for component in residual_components(
                    residual.model,
                    residual.estimated,
                    residual.combined_additive_estimated,
                    residual.combined_proportional_estimated,
                    residual.correlation_estimated,
                ) {
                    statistics.push(statistic(
                        self.iterations,
                        "residual_final",
                        &residual.output,
                        None,
                        None,
                        Some(residual.output_index),
                        Some(component.0),
                        component.1,
                    ));
                }
            }
        }
        marginal_likelihood_statistics(
            self.iterations,
            self.marginal_likelihood_diagnostics(),
            &mut statistics,
        );
        information_statistics(
            self.iterations,
            self.information_diagnostics(),
            &mut statistics,
        );
        n6_uncertainty_statistics(self, &mut statistics);
        information_criteria_statistics(
            self.iterations,
            self.information_criteria(),
            &mut statistics,
        );
        markov_variance_statistics(
            self.iterations,
            self.markov_simulation_variance(),
            &mut statistics,
        );
        operational_convergence_statistics(
            self.iterations,
            self.operational_diagnostics(),
            &mut statistics,
        );
        let covariate_effects = self.covariate_effect_rows()?;
        for effect in &covariate_effects {
            statistics.push(StatisticRow {
                cycle: self.iterations,
                kind: "covariate_effect_final".to_string(),
                name: effect.name.clone(),
                row: None,
                column: None,
                output_index: None,
                component: None,
                value: Some(effect.estimate),
                status: Some(
                    if effect.estimated {
                        "estimated"
                    } else {
                        "fixed"
                    }
                    .to_string(),
                ),
            });
        }
        let subject_covariates = self.subject_covariate_rows();
        let subject_population_parameters = self.subject_population_parameter_rows()?;
        Ok(ParametricResultTables {
            population,
            omega,
            omega_iov,
            residual_error,
            individual_effects,
            individual_parameters,
            iterations,
            statistics,
            marginal_likelihood: marginal_likelihood_rows(self.marginal_likelihood_diagnostics()),
            information_criteria: information_criteria_rows(self.information_criteria()),
            predictions: Vec::new(),
            covariate_effects,
            subject_covariates,
            subject_population_parameters,
        })
    }

    fn covariate_effect_rows(&self) -> Result<Vec<CovariateEffectRow>> {
        let Some(model) = self.covariates() else {
            return Ok(Vec::new());
        };
        model
            .declarations()
            .iter()
            .zip(model.estimates())
            .enumerate()
            .map(|(order, (declaration, estimate))| {
                let (family, center, reference, level) = match declaration.family() {
                    CovariateEffectFamily::Continuous { center } => {
                        ("continuous", Some(center), None, None)
                    }
                    CovariateEffectFamily::Categorical { reference, level } => {
                        ("categorical", None, Some(reference), Some(level))
                    }
                };
                Ok(CovariateEffectRow {
                    order,
                    name: declaration.name(),
                    family: family.to_string(),
                    parameter: declaration.parameter().to_string(),
                    parameter_index: model.parameter_indices()[order],
                    covariate: declaration.covariate().to_string(),
                    center,
                    reference,
                    level,
                    initial: declaration
                        .initial()
                        .context("validated covariate declaration lacks initial coefficient")?,
                    estimate: estimate.estimate(),
                    estimated: estimate.estimated(),
                })
            })
            .collect()
    }

    fn subject_covariate_rows(&self) -> Vec<SubjectCovariateRow> {
        let Some(model) = self.covariates() else {
            return Vec::new();
        };
        let subject_indices = self
            .data
            .subjects()
            .iter()
            .enumerate()
            .map(|(index, subject)| (subject.id().as_str(), index))
            .collect::<std::collections::HashMap<_, _>>();
        model
            .subject_values()
            .iter()
            .map(|row| SubjectCovariateRow {
                subject: row.subject().to_string(),
                subject_index: subject_indices[row.subject()],
                covariate: row.covariate().to_string(),
                value: row.value(),
            })
            .collect()
    }

    fn subject_population_parameter_rows(&self) -> Result<Vec<SubjectPopulationParameterRow>> {
        let Some(rows) = self
            .covariate_subject_population_parameters()
            .map_err(|error| anyhow::anyhow!(error))?
        else {
            return Ok(Vec::new());
        };
        let mut output = Vec::with_capacity(rows.len() * self.parameter_names.len());
        for (subject_index, row) in rows.iter().enumerate() {
            for parameter_index in 0..self.parameter_names.len() {
                output.push(SubjectPopulationParameterRow {
                    subject: row.subject().to_string(),
                    subject_index,
                    parameter: self.parameter_names[parameter_index].clone(),
                    parameter_index,
                    phi: row.phi()[parameter_index],
                    psi: row.psi()[parameter_index],
                });
            }
        }
        Ok(output)
    }

    fn validate_output_metadata(&self) -> Result<()> {
        let width = self.parameter_names.len();
        if self.population_estimates.len() != width
            || self.parameter_scales.len() != width
            || self.estimated_parameters.len() != width
        {
            bail!("population parameter names, estimates, scales, and estimated flags must have equal lengths");
        }
        validate_effect_indices(
            &self.random_effect_indices,
            &self.random_effect_names,
            width,
            "IIV",
        )?;
        validate_effect_indices(
            &self.iov_effect_indices,
            &self.iov_effect_names,
            width,
            "IOV",
        )?;
        if self.cycle_diagnostics.len() != self.iterations {
            bail!(
                "cycle diagnostic count {} does not match reported iteration count {}",
                self.cycle_diagnostics.len(),
                self.iterations
            );
        }
        Ok(())
    }

    fn individual_effect_rows(&self) -> Result<Vec<IndividualEffectRow>> {
        let mut rows = Vec::new();
        for eta in &self.eta_chain_means {
            if eta.values.len() != self.random_effect_names.len() {
                bail!(
                    "eta width for subject '{}' does not match IIV names",
                    eta.subject_id
                );
            }
            for (parameter, value) in self.random_effect_names.iter().zip(&eta.values) {
                rows.push(IndividualEffectRow {
                    subject: eta.subject_id.clone(),
                    source: "chain_mean".to_string(),
                    effect_kind: "eta".to_string(),
                    parameter: parameter.clone(),
                    occasion: None,
                    value: *value,
                    mode_converged: None,
                });
            }
        }
        for kappa in &self.kappa_chain_means {
            if kappa.values.len() != self.iov_effect_names.len() {
                bail!(
                    "kappa width for subject '{}' occasion {} does not match IOV names",
                    kappa.subject_id,
                    kappa.occasion_index
                );
            }
            for (parameter, value) in self.iov_effect_names.iter().zip(&kappa.values) {
                rows.push(IndividualEffectRow {
                    subject: kappa.subject_id.clone(),
                    source: "chain_mean".to_string(),
                    effect_kind: "kappa".to_string(),
                    parameter: parameter.clone(),
                    occasion: Some(kappa.occasion_index),
                    value: *value,
                    mode_converged: None,
                });
            }
        }
        for mode in &self.conditional_modes {
            if mode.eta.len() != self.random_effect_names.len() {
                bail!(
                    "conditional eta width for subject '{}' does not match IIV names",
                    mode.subject_id
                );
            }
            for (parameter, value) in self.random_effect_names.iter().zip(&mode.eta) {
                rows.push(IndividualEffectRow {
                    subject: mode.subject_id.clone(),
                    source: "conditional_mode".to_string(),
                    effect_kind: "eta".to_string(),
                    parameter: parameter.clone(),
                    occasion: None,
                    value: *value,
                    mode_converged: Some(mode.converged),
                });
            }
            for kappa in &mode.kappas {
                if kappa.subject_id != mode.subject_id
                    || kappa.values.len() != self.iov_effect_names.len()
                {
                    bail!(
                        "conditional kappa metadata for subject '{}' is inconsistent",
                        mode.subject_id
                    );
                }
                for (parameter, value) in self.iov_effect_names.iter().zip(&kappa.values) {
                    rows.push(IndividualEffectRow {
                        subject: mode.subject_id.clone(),
                        source: "conditional_mode".to_string(),
                        effect_kind: "kappa".to_string(),
                        parameter: parameter.clone(),
                        occasion: Some(kappa.occasion_index),
                        value: *value,
                        mode_converged: Some(mode.converged),
                    });
                }
            }
        }
        Ok(rows)
    }

    fn individual_parameter_rows(&self) -> Result<Vec<IndividualParameterRow>> {
        let subjects = self.data.subjects();
        if self.eta_chain_means.len() != subjects.len() {
            bail!(
                "eta chain-mean count {} does not match subject count {}",
                self.eta_chain_means.len(),
                subjects.len()
            );
        }

        let subject_means = self
            .covariate_subject_population_parameters()
            .map_err(|error| anyhow::anyhow!(error))?;
        if let Some(means) = &subject_means {
            if means.len() != subjects.len() {
                bail!(
                    "resolved subject population mean count {} does not match subject count {}",
                    means.len(),
                    subjects.len()
                );
            }
        }

        let mut rows = Vec::new();
        let mut chain_kappas = self.kappa_chain_means.iter();
        for (subject_index, (subject, eta)) in
            subjects.iter().zip(&self.eta_chain_means).enumerate()
        {
            let subject_mu_phi = subject_means.as_ref().map(|means| &means[subject_index]);
            if let Some(mean) = subject_mu_phi {
                if mean.subject() != subject.id() {
                    bail!(
                        "resolved subject population mean '{}' does not match subject '{}'",
                        mean.subject(),
                        subject.id()
                    );
                }
            }
            validate_eta(
                subject.id(),
                eta.subject_id.as_str(),
                eta.values.len(),
                self.random_effect_indices.len(),
                "chain mean",
            )?;
            if self.iov_effect_indices.is_empty() {
                let values = if let Some(mean) = subject_mu_phi {
                    individual_psi_from_subject_mean(
                        mean.phi(),
                        &self.parameter_scales,
                        &self.random_effect_indices,
                        &eta.values,
                    )?
                } else {
                    individual_psi(
                        &self.population_estimates,
                        &self.parameter_scales,
                        &self.random_effect_indices,
                        &eta.values,
                    )?
                };
                push_parameter_rows(
                    &mut rows,
                    subject.id(),
                    None,
                    &self.parameter_names,
                    &values,
                    "chain_mean",
                    None,
                )?;
            } else {
                for occasion in subject.occasions() {
                    let kappa = chain_kappas.next().with_context(|| {
                        format!(
                            "missing chain-mean kappa for subject '{}' occasion {}",
                            subject.id(),
                            occasion.index()
                        )
                    })?;
                    validate_kappa(
                        subject.id(),
                        occasion.index(),
                        kappa,
                        self.iov_effect_indices.len(),
                        "chain mean",
                    )?;
                    let values = if let Some(mean) = subject_mu_phi {
                        occasion_psi_from_subject_mean(
                            mean.phi(),
                            &self.parameter_scales,
                            &self.random_effect_indices,
                            &eta.values,
                            &self.iov_effect_indices,
                            &kappa.values,
                        )?
                    } else {
                        occasion_psi(
                            &self.population_estimates,
                            &self.parameter_scales,
                            &self.random_effect_indices,
                            &eta.values,
                            &self.iov_effect_indices,
                            &kappa.values,
                        )?
                    };
                    push_parameter_rows(
                        &mut rows,
                        subject.id(),
                        Some(occasion.index()),
                        &self.parameter_names,
                        &values,
                        "chain_mean",
                        None,
                    )?;
                }
            }
        }
        if let Some(extra) = chain_kappas.next() {
            bail!(
                "unexpected chain-mean kappa for subject '{}' occasion {}",
                extra.subject_id,
                extra.occasion_index
            );
        }

        if !self.conditional_modes.is_empty() && self.conditional_modes.len() != subjects.len() {
            bail!(
                "conditional mode count {} does not match subject count {}",
                self.conditional_modes.len(),
                subjects.len()
            );
        }
        for (subject_index, (subject, mode)) in
            subjects.iter().zip(&self.conditional_modes).enumerate()
        {
            let subject_mu_phi = subject_means.as_ref().map(|means| &means[subject_index]);
            validate_eta(
                subject.id(),
                mode.subject_id.as_str(),
                mode.eta.len(),
                self.random_effect_indices.len(),
                "conditional mode",
            )?;
            if self.iov_effect_indices.is_empty() {
                if !mode.kappas.is_empty() {
                    bail!(
                        "conditional mode for non-IOV subject '{}' unexpectedly has kappas",
                        subject.id()
                    );
                }
                let values = if let Some(mean) = subject_mu_phi {
                    individual_psi_from_subject_mean(
                        mean.phi(),
                        &self.parameter_scales,
                        &self.random_effect_indices,
                        &mode.eta,
                    )?
                } else {
                    mode.parameters.clone()
                };
                push_parameter_rows(
                    &mut rows,
                    subject.id(),
                    None,
                    &self.parameter_names,
                    &values,
                    "conditional_mode",
                    Some(mode.converged),
                )?;
            } else {
                if mode.kappas.len() != subject.occasions().len() {
                    bail!(
                        "conditional mode for subject '{}' has {} kappas but retained data have {} occasions",
                        subject.id(),
                        mode.kappas.len(),
                        subject.occasions().len()
                    );
                }
                for (occasion, kappa) in subject.occasions().iter().zip(&mode.kappas) {
                    validate_kappa(
                        subject.id(),
                        occasion.index(),
                        kappa,
                        self.iov_effect_indices.len(),
                        "conditional mode",
                    )?;
                    let values = if let Some(mean) = subject_mu_phi {
                        occasion_psi_from_subject_mean(
                            mean.phi(),
                            &self.parameter_scales,
                            &self.random_effect_indices,
                            &mode.eta,
                            &self.iov_effect_indices,
                            &kappa.values,
                        )?
                    } else {
                        occasion_psi(
                            &self.population_estimates,
                            &self.parameter_scales,
                            &self.random_effect_indices,
                            &mode.eta,
                            &self.iov_effect_indices,
                            &kappa.values,
                        )?
                    };
                    push_parameter_rows(
                        &mut rows,
                        subject.id(),
                        Some(occasion.index()),
                        &self.parameter_names,
                        &values,
                        "conditional_mode",
                        Some(mode.converged),
                    )?;
                }
            }
        }
        Ok(rows)
    }

    fn iteration_rows(&self) -> Vec<IterationRow> {
        self.cycle_diagnostics
            .iter()
            .map(|cycle| IterationRow {
                cycle: cycle.iteration,
                phase: phase_text(cycle.phase).to_string(),
                conditional_n2ll: 2.0 * cycle.conditional_negative_log_likelihood,
                sa_step: cycle.stochastic_approximation_step,
                covariance_step: cycle.covariance_step,
                eta_proposals: cycle.eta_proposals,
                eta_accepted: cycle.eta_accepted,
                eta_rejected: cycle.eta_rejected,
                eta_nonfinite: cycle.eta_non_finite,
                eta_block_proposals: cycle.eta_block_proposals,
                eta_block_accepted: cycle.eta_block_accepted,
                eta_block_rejected: cycle.eta_block_rejected,
                eta_block_nonfinite: cycle.eta_block_non_finite,
                kappa_proposals: cycle.kappa_proposals,
                kappa_accepted: cycle.kappa_accepted,
                kappa_rejected: cycle.kappa_rejected,
                kappa_nonfinite: cycle.kappa_non_finite,
                omega_update_rejected: cycle.omega_update_rejected,
                omega_iov_update_rejected: cycle.omega_iov_update_rejected,
            })
            .collect()
    }

    fn statistic_rows(&self) -> Result<Vec<StatisticRow>> {
        let mut rows = Vec::new();
        for cycle in &self.cycle_diagnostics {
            if cycle.population_parameters.len() != self.parameter_names.len() {
                bail!(
                    "population width in cycle {} is inconsistent",
                    cycle.iteration
                );
            }
            for (name, value) in self
                .parameter_names
                .iter()
                .zip(&cycle.population_parameters)
            {
                rows.push(statistic(
                    cycle.iteration,
                    "theta",
                    name,
                    None,
                    None,
                    None,
                    None,
                    *value,
                ));
            }
            match (&cycle.covariate_betas, &cycle.covariate_beta_estimated) {
                (None, None) if self.covariates().is_none() => {}
                (Some(values), Some(estimated))
                    if values.len() == estimated.len()
                        && values.len()
                            == self.covariates().map_or(0, |model| model.estimates().len()) =>
                {
                    for (effect_index, value) in values.iter().enumerate() {
                        rows.push(StatisticRow {
                            cycle: cycle.iteration,
                            kind: "covariate_effect".to_string(),
                            name: self
                                .covariates()
                                .expect("covariate cycle metadata validated")
                                .estimates()[effect_index]
                                .name()
                                .to_string(),
                            row: None,
                            column: None,
                            output_index: None,
                            component: None,
                            value: Some(*value),
                            status: Some(
                                if estimated[effect_index] {
                                    "estimated"
                                } else {
                                    "fixed"
                                }
                                .to_string(),
                            ),
                        });
                    }
                }
                _ => bail!(
                    "covariate coefficient diagnostics in cycle {} are inconsistent",
                    cycle.iteration
                ),
            }
            append_covariance_statistics(
                &mut rows,
                cycle.iteration,
                "omega",
                &self.random_effect_names,
                &cycle.omega,
            )?;
            if let Some(margin) = cycle.omega_relative_spd_margin {
                rows.push(statistic(
                    cycle.iteration,
                    "covariance_stability",
                    "omega_relative_spd_margin",
                    None,
                    None,
                    None,
                    None,
                    margin,
                ));
            }
            if let Some(matrix) = cycle.omega_iov.as_ref() {
                append_covariance_statistics(
                    &mut rows,
                    cycle.iteration,
                    "omega_iov",
                    &self.iov_effect_names,
                    matrix,
                )?;
            }
            if let Some(margin) = cycle.omega_iov_relative_spd_margin {
                rows.push(statistic(
                    cycle.iteration,
                    "covariance_stability",
                    "omega_iov_relative_spd_margin",
                    None,
                    None,
                    None,
                    None,
                    margin,
                ));
            }
            for residual in &cycle.residual_error_estimates {
                for component in residual_components(
                    residual.model,
                    residual.estimated,
                    residual.combined_additive_estimated,
                    residual.combined_proportional_estimated,
                    residual.correlation_estimated,
                ) {
                    rows.push(statistic(
                        cycle.iteration,
                        "residual",
                        &residual.output,
                        None,
                        None,
                        Some(residual.output_index),
                        Some(component.0),
                        component.1,
                    ));
                }
            }
        }
        Ok(rows)
    }

    fn prediction_rows(&self, idelta: f64, tad: f64) -> Result<Vec<PredictionRow>>
    where
        E: pharmsol::equation::EquationTypes<P = SubjectPredictions>,
    {
        let population = self.population_predictions(idelta, tad)?;
        let conditional =
            if self.random_effect_indices.is_empty() && self.iov_effect_indices.is_empty() {
                Some((population.clone(), "population"))
            } else if self.conditional_modes.is_empty() {
                None
            } else {
                Some((
                    self.conditional_predictions(idelta, tad)?,
                    "conditional_mode",
                ))
            };
        if population.len() != self.data.subjects().len() {
            bail!("population prediction subject count does not match retained data");
        }
        if let Some((conditional, _)) = conditional.as_ref() {
            if conditional.len() != population.len() {
                bail!("conditional and population prediction subject counts differ");
            }
        }
        let subjects = self.data.subjects();
        let mut rows = Vec::new();
        for (subject_index, predictions) in population.iter().enumerate() {
            let subject = subjects
                .get(subject_index)
                .context("prediction subject index exceeds retained data")?;
            let conditional_predictions = conditional.as_ref().map(|value| &value.0[subject_index]);
            if let Some(other) = conditional_predictions {
                if other.predictions().len() != predictions.predictions().len() {
                    bail!("prediction count mismatch for subject '{}'", subject.id());
                }
            }
            for (point_index, point) in predictions.predictions().iter().enumerate() {
                let conditional_point =
                    conditional_predictions.map(|values| &values.predictions()[point_index]);
                if let Some(other) = conditional_point {
                    validate_prediction_pair(subject.id(), point, other)?;
                }
                rows.push(PredictionRow {
                    subject: subject.id().clone(),
                    time: point.time(),
                    output_index: point.outeq(),
                    block: point.occasion(),
                    observation: point.observation(),
                    censoring: censor_text(point.censoring()).to_string(),
                    population_prediction: point.prediction(),
                    conditional_prediction: conditional_point.map(Prediction::prediction),
                    conditional_source: conditional.as_ref().map(|value| value.1.to_string()),
                });
            }
        }
        Ok(rows)
    }
}

fn validate_effect_indices(
    indices: &[usize],
    names: &[String],
    width: usize,
    label: &str,
) -> Result<()> {
    if indices.len() != names.len() {
        bail!("{label} indices and names have different lengths");
    }
    let mut seen = vec![false; width];
    for index in indices {
        if *index >= width {
            bail!("{label} parameter index {index} exceeds parameter width {width}");
        }
        if seen[*index] {
            bail!("{label} parameter index {index} is duplicated");
        }
        seen[*index] = true;
    }
    Ok(())
}

fn boolean_mask_rows(mask: &ndarray::Array2<bool>) -> Vec<Vec<bool>> {
    (0..mask.nrows())
        .map(|row| {
            (0..mask.ncols())
                .map(|column| mask[[row, column]])
                .collect()
        })
        .collect()
}

fn numeric_matrix_rows(matrix: &ndarray::Array2<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[[row, column]])
                .collect()
        })
        .collect()
}

fn covariance_rows(
    names: &[String],
    matrix: &ndarray::Array2<f64>,
    structural: &ndarray::Array2<bool>,
    estimated: &ndarray::Array2<bool>,
    label: &str,
) -> Result<Vec<OmegaRow>> {
    let n = names.len();
    if matrix.dim() != (n, n) || structural.dim() != (n, n) || estimated.dim() != (n, n) {
        bail!("{label} matrix and masks must be square with width {n}");
    }
    let mut rows = Vec::new();
    for row in 0..n {
        for column in 0..=row {
            rows.push(OmegaRow {
                row: names[row].clone(),
                column: names[column].clone(),
                estimate: matrix[[row, column]],
                structural: structural[[row, column]],
                estimated: estimated[[row, column]],
            });
        }
    }
    Ok(rows)
}

fn residual_rows(estimates: &[crate::results::ResidualErrorEstimate]) -> Vec<ResidualErrorRow> {
    let mut rows = Vec::new();
    for estimate in estimates {
        let family = residual_family(estimate.model).to_string();
        for (component, value, estimated) in residual_components(
            estimate.model,
            estimate.estimated,
            estimate.combined_additive_estimated,
            estimate.combined_proportional_estimated,
            estimate.correlation_estimated,
        ) {
            rows.push(ResidualErrorRow {
                output: estimate.output.clone(),
                output_index: estimate.output_index,
                family: family.clone(),
                component: component.to_string(),
                estimate: value,
                estimated,
            });
        }
    }
    rows
}

pub(crate) fn residual_components(
    model: ResidualErrorModel,
    estimated: bool,
    additive_estimated: Option<bool>,
    proportional_estimated: Option<bool>,
    correlation_estimated: Option<bool>,
) -> Vec<(&'static str, f64, bool)> {
    match model {
        ResidualErrorModel::Constant { a } => vec![("sigma", a, estimated)],
        ResidualErrorModel::Proportional { b } => vec![("proportional", b, estimated)],
        ResidualErrorModel::Combined { a, b } => vec![
            ("additive", a, additive_estimated.unwrap_or(estimated)),
            (
                "proportional",
                b,
                proportional_estimated.unwrap_or(estimated),
            ),
        ],
        ResidualErrorModel::CorrelatedCombined { a, b, rho } => vec![
            ("additive", a, additive_estimated.unwrap_or(estimated)),
            (
                "proportional",
                b,
                proportional_estimated.unwrap_or(estimated),
            ),
            (
                "correlation",
                rho,
                correlation_estimated.unwrap_or(estimated),
            ),
        ],
        ResidualErrorModel::Exponential { sigma } => vec![("sigma", sigma, estimated)],
    }
}

fn residual_family(model: ResidualErrorModel) -> &'static str {
    match model {
        ResidualErrorModel::Constant { .. } => "constant",
        ResidualErrorModel::Proportional { .. } => "proportional",
        ResidualErrorModel::Combined { .. } => "combined",
        ResidualErrorModel::CorrelatedCombined { .. } => "correlated_combined",
        ResidualErrorModel::Exponential { .. } => "exponential",
    }
}

fn problem_from_tables<E>(
    equation: E,
    data: Data,
    tables: &ParametricResultTables,
) -> Result<EstimationProblem<E, Parametric>>
where
    E: Equation + EquationMetadataSource,
{
    let mut names = std::collections::HashSet::new();
    let mut parameters = Vec::with_capacity(tables.population.len());
    let mut random_effect_names = Vec::new();
    let mut iov_membership_names = Vec::new();
    for row in &tables.population {
        if row.name.is_empty() || !names.insert(row.name.as_str()) {
            bail!("population parameter names must be non-empty and unique");
        }
        if !row.estimate.is_finite() {
            bail!("population estimate for '{}' must be finite", row.name);
        }
        let parameter = UnboundedParameter::new(row.name.clone(), parse_scale(&row.scale)?)
            .with_initial(row.estimate)
            .with_estimate(row.estimated)
            .with_random_effect(row.iiv);
        if row.iiv {
            random_effect_names.push(row.name.clone());
        }
        if row.iov {
            iov_membership_names.push(row.name.clone());
        }
        parameters.push(parameter);
    }
    if parameters.is_empty() {
        bail!("warm-start population table must contain at least one parameter");
    }

    let omega = covariance_declaration(&tables.omega, &random_effect_names, "Omega")?;
    let iov = match (&tables.omega_iov, iov_membership_names.is_empty()) {
        (None, true) => None,
        (Some(_), true) => bail!("Omega_IOV rows are present but no population parameter has IOV"),
        (None, false) => bail!("IOV parameters are present but Omega_IOV rows are missing"),
        (Some(rows), false) => {
            let ordered_names = covariance_names(rows, &iov_membership_names, "Omega_IOV")?;
            Some(iov_declaration(rows, &ordered_names)?)
        }
    };
    let residuals = residual_declarations(&tables.residual_error, &equation)?;
    let covariate_effects = tables
        .covariate_effects
        .iter()
        .enumerate()
        .map(|(order, row)| {
            if row.order != order
                || !row.estimate.is_finite()
                || row.parameter_index >= tables.population.len()
                || row.parameter != tables.population[row.parameter_index].name
            {
                bail!("warm-start covariate effect metadata is malformed");
            }
            let effect = match row.family.as_str() {
                "continuous" => CovariateEffect::continuous(
                    row.parameter.clone(),
                    row.covariate.clone(),
                    row.center
                        .context("continuous covariate effect requires a center")?,
                ),
                "categorical" => CovariateEffect::categorical(
                    row.parameter.clone(),
                    row.covariate.clone(),
                    row.reference
                        .context("categorical covariate effect requires a reference")?,
                    row.level
                        .context("categorical covariate effect requires a level")?,
                ),
                family => bail!("unknown warm-start covariate effect family '{family}'"),
            }
            .with_initial(row.estimate);
            Ok(if row.estimated {
                effect
            } else {
                effect.fixed()
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut builder = EstimationProblem::parametric(equation, data)
        .parameters(parameters)
        .omega(omega)
        .covariate_effects(covariate_effects);
    if let Some(iov) = iov {
        builder = builder.iov(iov);
    }
    for (output, declaration) in residuals {
        builder = builder.error_model(output, declaration);
    }
    builder.build()
}

fn covariance_names(rows: &[OmegaRow], membership: &[String], label: &str) -> Result<Vec<String>> {
    let expected_len = membership.len() * (membership.len() + 1) / 2;
    if rows.len() != expected_len {
        bail!(
            "{label} lower-triangle row count {} does not match expected {expected_len}",
            rows.len()
        );
    }
    let mut names = Vec::with_capacity(membership.len());
    let mut offset = 0;
    for row_index in 0..membership.len() {
        let diagonal = &rows[offset + row_index];
        if diagonal.row != diagonal.column
            || !membership.iter().any(|name| name == &diagonal.row)
            || names.iter().any(|name| name == &diagonal.row)
        {
            bail!(
                "{label} diagonal row {} must name one unique declared effect",
                offset + row_index
            );
        }
        names.push(diagonal.row.clone());
        offset += row_index + 1;
    }
    Ok(names)
}

fn covariance_declaration(rows: &[OmegaRow], names: &[String], label: &str) -> Result<Omega> {
    let expected_len = names.len() * (names.len() + 1) / 2;
    if rows.len() != expected_len {
        bail!(
            "{label} lower-triangle row count {} does not match expected {expected_len}",
            rows.len()
        );
    }
    let mut omega = Omega::new();
    let mut offset = 0;
    for row_index in 0..names.len() {
        for column_index in 0..=row_index {
            let row = &rows[offset];
            offset += 1;
            if row.row != names[row_index] || row.column != names[column_index] {
                bail!(
                    "{label} row {} must be lower-triangle entry ('{}', '{}'), found ('{}', '{}')",
                    offset - 1,
                    names[row_index],
                    names[column_index],
                    row.row,
                    row.column
                );
            }
            if !row.estimate.is_finite() {
                bail!(
                    "{label} entry ('{}', '{}') must be finite",
                    row.row,
                    row.column
                );
            }
            if !row.structural {
                if row.estimated || row.estimate != 0.0 {
                    bail!(
                        "non-structural {label} entry ('{}', '{}') must be zero and not estimated",
                        row.row,
                        row.column
                    );
                }
                continue;
            }
            if row_index == column_index && row.estimate <= 0.0 {
                bail!("{label} variance for '{}' must be positive", row.row);
            }
            omega = match (row_index == column_index, row.estimated) {
                (true, true) => omega.variance(row.row.clone(), row.estimate),
                (true, false) => omega.fixed_variance(row.row.clone(), row.estimate),
                (false, true) => {
                    omega.covariance(row.row.clone(), row.column.clone(), row.estimate)
                }
                (false, false) => {
                    omega.fixed_covariance(row.row.clone(), row.column.clone(), row.estimate)
                }
            };
        }
    }
    Ok(omega)
}

fn iov_declaration(rows: &[OmegaRow], names: &[String]) -> Result<Iov> {
    covariance_declaration(rows, names, "Omega_IOV")?;
    let mut iov = Iov::new();
    for row in rows.iter().filter(|row| row.structural) {
        let diagonal = row.row == row.column;
        iov = match (diagonal, row.estimated) {
            (true, true) => iov.variance(row.row.clone(), row.estimate),
            (true, false) => iov.fixed_variance(row.row.clone(), row.estimate),
            (false, true) => iov.covariance(row.row.clone(), row.column.clone(), row.estimate),
            (false, false) => {
                iov.fixed_covariance(row.row.clone(), row.column.clone(), row.estimate)
            }
        };
    }
    Ok(iov)
}

fn residual_declarations<E>(
    rows: &[ResidualErrorRow],
    equation: &E,
) -> Result<Vec<(String, ParametricErrorModel)>>
where
    E: Equation + EquationMetadataSource,
{
    if rows.is_empty() {
        bail!("warm-start residual-error table must not be empty");
    }
    let metadata = equation
        .equation_metadata()
        .context("warm-start equation must provide output metadata")?;
    let mut declarations = Vec::new();
    let mut output_indices = std::collections::HashSet::new();
    let mut offset = 0;
    while offset < rows.len() {
        let first = &rows[offset];
        if first.output.is_empty() {
            bail!("residual output {} has an empty name", first.output_index);
        }
        if !output_indices.insert(first.output_index) {
            bail!(
                "residual output index {} is declared more than once",
                first.output_index
            );
        }
        let family = first.family.as_str();
        let component_count = match family {
            "combined" => 2,
            "correlated_combined" => 3,
            _ => 1,
        };
        if offset + component_count > rows.len() {
            bail!("residual output '{}' has missing components", first.output);
        }
        let group = &rows[offset..offset + component_count];
        for row in group {
            let expected_output = metadata.outputs().get(row.output_index).with_context(|| {
                format!(
                    "residual output index {} exceeds equation output metadata",
                    row.output_index
                )
            })?;
            if expected_output.name() != row.output {
                bail!(
                    "residual output index {} is named '{}' in the table but '{}' in equation metadata",
                    row.output_index,
                    row.output,
                    expected_output.name()
                );
            }
        }
        if group.iter().any(|row| {
            row.output_index != first.output_index
                || row.output != first.output
                || row.family != first.family
                || !row.estimate.is_finite()
        }) {
            bail!(
                "residual output '{}' must have one coherent family with finite components",
                first.output
            );
        }
        let declaration = match family {
            "constant" if group[0].component == "sigma" && group[0].estimate > 0.0 => {
                ParametricErrorModel::new(ResidualErrorModel::constant(group[0].estimate))
                    .with_estimate(group[0].estimated)
            }
            "proportional"
                if group[0].component == "proportional" && group[0].estimate > 0.0 =>
            {
                ParametricErrorModel::new(ResidualErrorModel::proportional(group[0].estimate))
                    .with_estimate(group[0].estimated)
            }
            "exponential" if group[0].component == "sigma" && group[0].estimate > 0.0 => {
                ParametricErrorModel::new(ResidualErrorModel::exponential(group[0].estimate))
                    .with_estimate(group[0].estimated)
            }
            "combined"
                if group[0].component == "additive"
                    && group[1].component == "proportional"
                    && group.iter().all(|row| row.estimate >= 0.0)
                    && group.iter().all(|row| !row.estimated || row.estimate > 0.0)
                    && group.iter().any(|row| row.estimate > 0.0) =>
            {
                ParametricErrorModel::new(ResidualErrorModel::combined(
                    group[0].estimate,
                    group[1].estimate,
                ))
                .with_combined_additive_estimate(group[0].estimated)
                .with_combined_proportional_estimate(group[1].estimated)
            }
            "correlated_combined"
                if group[0].component == "additive"
                    && group[1].component == "proportional"
                    && group[2].component == "correlation"
                    && group[0].estimate > 0.0
                    && group[1].estimate > 0.0
                    && group[2].estimate > -1.0
                    && group[2].estimate < 1.0 =>
            {
                ParametricErrorModel::new(ResidualErrorModel::correlated_combined(
                    group[0].estimate,
                    group[1].estimate,
                    group[2].estimate,
                ))
                .with_correlated_combined_additive_estimate(group[0].estimated)
                .with_correlated_combined_proportional_estimate(group[1].estimated)
                .with_correlated_combined_correlation_estimate(group[2].estimated)
            }
            "constant" | "proportional" | "exponential" => bail!(
                "residual output '{}' requires a finite strictly positive component for family '{}'",
                first.output,
                family
            ),
            "combined" => bail!(
                "residual output '{}' requires finite non-negative combined components, estimated components must be positive, and both components cannot be zero",
                first.output
            ),
            "correlated_combined" => bail!(
                "residual output '{}' requires finite positive additive/proportional components and correlation strictly inside (-1, 1)",
                first.output
            ),
            _ => bail!("unknown residual-error family '{family}'"),
        };
        declarations.push((first.output.clone(), declaration));
        offset += component_count;
    }
    Ok(declarations)
}

fn parse_scale(text: &str) -> Result<ParameterScale> {
    match text {
        "identity" => Ok(ParameterScale::Identity),
        "log" => Ok(ParameterScale::Log),
        _ => {
            let (scale, bounds) = text
                .split_once('(')
                .and_then(|(scale, rest)| rest.strip_suffix(')').map(|bounds| (scale, bounds)))
                .ok_or_else(|| anyhow::anyhow!("unknown parameter scale '{text}'"))?;
            let (lower, upper) = bounds
                .split_once(',')
                .ok_or_else(|| anyhow::anyhow!("invalid parameter scale '{text}'"))?;
            let lower = lower
                .parse::<f64>()
                .with_context(|| format!("invalid lower bound in parameter scale '{text}'"))?;
            let upper = upper
                .parse::<f64>()
                .with_context(|| format!("invalid upper bound in parameter scale '{text}'"))?;
            if !lower.is_finite() || !upper.is_finite() || lower >= upper {
                bail!("invalid finite ordered bounds in parameter scale '{text}'");
            }
            match scale {
                "logit" => Ok(ParameterScale::Logit { lower, upper }),
                "probit" => Ok(ParameterScale::Probit { lower, upper }),
                _ => bail!("unknown parameter scale '{text}'"),
            }
        }
    }
}

fn validate_eta(
    expected_subject: &str,
    actual_subject: &str,
    actual_width: usize,
    expected_width: usize,
    source: &str,
) -> Result<()> {
    if actual_subject != expected_subject {
        bail!(
            "{source} eta subject '{}' does not match retained subject '{}'",
            actual_subject,
            expected_subject
        );
    }
    if actual_width != expected_width {
        bail!(
            "{source} eta for subject '{}' has width {} but expected {}",
            expected_subject,
            actual_width,
            expected_width
        );
    }
    Ok(())
}

fn validate_kappa(
    expected_subject: &str,
    expected_occasion: usize,
    kappa: &crate::results::OccasionKappaEstimate,
    expected_width: usize,
    source: &str,
) -> Result<()> {
    if kappa.subject_id != expected_subject {
        bail!(
            "{source} kappa subject '{}' does not match retained subject '{}'",
            kappa.subject_id,
            expected_subject
        );
    }
    if kappa.occasion_index != expected_occasion {
        bail!(
            "{source} kappa occasion {} does not match retained occasion {} for subject '{}'",
            kappa.occasion_index,
            expected_occasion,
            expected_subject
        );
    }
    if kappa.values.len() != expected_width {
        bail!(
            "{source} kappa for subject '{}' occasion {} has width {} but expected {}",
            expected_subject,
            expected_occasion,
            kappa.values.len(),
            expected_width
        );
    }
    Ok(())
}

fn push_parameter_rows(
    rows: &mut Vec<IndividualParameterRow>,
    subject: &str,
    occasion: Option<usize>,
    names: &[String],
    values: &[f64],
    source: &str,
    converged: Option<bool>,
) -> Result<()> {
    if names.len() != values.len() {
        bail!("individual parameter width for subject '{subject}' does not match parameter names");
    }
    for (parameter, value) in names.iter().zip(values) {
        rows.push(IndividualParameterRow {
            subject: subject.to_string(),
            occasion,
            parameter: parameter.clone(),
            value: *value,
            source: source.to_string(),
            mode_converged: converged,
        });
    }
    Ok(())
}

fn append_covariance_statistics(
    rows: &mut Vec<StatisticRow>,
    cycle: usize,
    kind: &str,
    names: &[String],
    matrix: &ndarray::Array2<f64>,
) -> Result<()> {
    let n = names.len();
    if matrix.dim() != (n, n) {
        bail!("{kind} width in cycle {cycle} is inconsistent");
    }
    for row in 0..n {
        for column in 0..=row {
            rows.push(statistic(
                cycle,
                kind,
                "",
                Some(names[row].clone()),
                Some(names[column].clone()),
                None,
                None,
                matrix[[row, column]],
            ));
        }
    }
    Ok(())
}

fn marginal_likelihood_rows(
    diagnostics: Option<&MarginalLikelihoodDiagnostics>,
) -> Vec<MarginalLikelihoodRow> {
    let Some(diagnostics) = diagnostics else {
        return vec![MarginalLikelihoodRow {
            scope: "total".to_string(),
            subject: None,
            method: "disabled".to_string(),
            status: "disabled".to_string(),
            samples_per_subject: 0,
            seed: None,
            degrees_of_freedom: 0,
            covariance_scale_multiplier: 0.0,
            proposal_scale_source: "not_applicable".to_string(),
            dimension: 0,
            occasion_indices: "[]".to_string(),
            mode: "[]".to_string(),
            mode_converged: None,
            log_marginal_likelihood: None,
            n2ll: None,
            n2ll_mcse: None,
            effective_sample_size: None,
            effective_sample_fraction: None,
            zero_weight_count: 0,
            failure: None,
        }];
    };
    let status = marginal_status_text(&diagnostics.status);
    let mut rows = vec![MarginalLikelihoodRow {
        scope: "total".to_string(),
        subject: None,
        method: "population".to_string(),
        status: status.clone(),
        samples_per_subject: diagnostics.config.samples_per_subject,
        seed: Some(diagnostics.config.seed),
        degrees_of_freedom: diagnostics.config.degrees_of_freedom,
        covariance_scale_multiplier: diagnostics.config.covariance_scale_multiplier,
        proposal_scale_source: "mixed_by_subject".to_string(),
        dimension: diagnostics
            .subjects
            .iter()
            .map(|subject| subject.dimension)
            .sum(),
        occasion_indices: "[]".to_string(),
        mode: "[]".to_string(),
        mode_converged: None,
        log_marginal_likelihood: diagnostics.log_marginal_likelihood,
        n2ll: diagnostics.n2ll,
        n2ll_mcse: diagnostics.n2ll_mcse,
        effective_sample_size: None,
        effective_sample_fraction: None,
        zero_weight_count: diagnostics
            .subjects
            .iter()
            .map(|subject| subject.zero_weight_count)
            .sum(),
        failure: match &diagnostics.status {
            MarginalLikelihoodStatus::Unavailable { failures } => {
                serde_json::to_string(failures).ok()
            }
            _ => None,
        },
    }];
    rows.extend(diagnostics.subjects.iter().map(|subject| {
        MarginalLikelihoodRow {
            scope: "subject".to_string(),
            subject: Some(subject.subject_id.clone()),
            method: marginal_method_text(subject.method).to_string(),
            status: if subject.failure.is_some() {
                "unavailable".to_string()
            } else if subject.mode_converged == Some(false) {
                "available_with_nonconverged_mode".to_string()
            } else {
                "available".to_string()
            },
            samples_per_subject: subject.samples,
            seed: subject.seed,
            degrees_of_freedom: diagnostics.config.degrees_of_freedom,
            covariance_scale_multiplier: diagnostics.config.covariance_scale_multiplier,
            proposal_scale_source: proposal_scale_source_text(subject.proposal_scale_source)
                .to_string(),
            dimension: subject.dimension,
            occasion_indices: serde_json::to_string(&subject.occasion_indices)
                .expect("occasion indices serialize"),
            mode: serde_json::to_string(&subject.mode).expect("mode coordinates serialize"),
            mode_converged: subject.mode_converged,
            log_marginal_likelihood: subject.log_marginal_likelihood,
            n2ll: subject.n2ll,
            n2ll_mcse: subject.n2ll_mcse,
            effective_sample_size: subject.effective_sample_size,
            effective_sample_fraction: subject.effective_sample_fraction,
            zero_weight_count: subject.zero_weight_count,
            failure: subject
                .failure
                .as_ref()
                .and_then(|reason| serde_json::to_string(reason).ok()),
        }
    }));
    rows
}

fn marginal_likelihood_statistics(
    iterations: usize,
    diagnostics: Option<&MarginalLikelihoodDiagnostics>,
    rows: &mut Vec<StatisticRow>,
) {
    let status = diagnostics
        .map(|diagnostics| marginal_status_text(&diagnostics.status))
        .unwrap_or_else(|| "disabled".to_string());
    rows.push(StatisticRow {
        cycle: iterations,
        kind: "marginal_likelihood_status".to_string(),
        name: "population".to_string(),
        row: None,
        column: None,
        output_index: None,
        component: None,
        value: None,
        status: Some(status.clone()),
    });
    let Some(diagnostics) = diagnostics else {
        return;
    };
    for (name, value) in [
        (
            "log_marginal_likelihood",
            diagnostics.log_marginal_likelihood,
        ),
        ("marginal_n2ll", diagnostics.n2ll),
        ("marginal_n2ll_mcse", diagnostics.n2ll_mcse),
    ] {
        rows.push(StatisticRow {
            cycle: iterations,
            kind: "marginal_likelihood".to_string(),
            name: name.to_string(),
            row: None,
            column: None,
            output_index: None,
            component: None,
            value,
            status: Some(status.clone()),
        });
    }
    for subject in &diagnostics.subjects {
        rows.push(StatisticRow {
            cycle: iterations,
            kind: "marginal_likelihood_subject_status".to_string(),
            name: subject.subject_id.clone(),
            row: None,
            column: None,
            output_index: None,
            component: None,
            value: subject.n2ll,
            status: Some(if subject.failure.is_some() {
                "unavailable".to_string()
            } else {
                "available".to_string()
            }),
        });
    }
}

fn marginal_status_text(status: &MarginalLikelihoodStatus) -> String {
    match status {
        MarginalLikelihoodStatus::Available => "available",
        MarginalLikelihoodStatus::AvailableWithNonconvergedModes { .. } => {
            "available_with_nonconverged_modes"
        }
        MarginalLikelihoodStatus::Unavailable { .. } => "unavailable",
    }
    .to_string()
}

fn marginal_method_text(method: MarginalLikelihoodMethod) -> &'static str {
    match method {
        MarginalLikelihoodMethod::ExactNoLatent => "exact_no_latent",
        MarginalLikelihoodMethod::StudentTImportanceSampling => "student_t_importance_sampling",
    }
}

fn proposal_scale_source_text(source: ProposalScaleSource) -> &'static str {
    match source {
        ProposalScaleSource::FinalRawOmegaBlocks => "final_raw_omega_blocks",
        ProposalScaleSource::ConditionalModeCurvature => "conditional_mode_curvature",
        ProposalScaleSource::NotApplicableNoLatent => "not_applicable_no_latent",
    }
}

fn information_criteria_rows(
    diagnostics: &InformationCriteriaDiagnostics,
) -> Vec<InformationCriteriaRow> {
    vec![InformationCriteriaRow {
        status: information_criteria_status_text(&diagnostics.status).to_string(),
        sample_size_convention: information_criteria_convention_text(
            diagnostics.sample_size_convention,
        )
        .to_string(),
        subject_count: diagnostics.subject_count,
        population_parameter_count: diagnostics.parameter_count.population,
        covariate_parameter_count: diagnostics.parameter_count.covariate,
        omega_parameter_count: diagnostics.parameter_count.omega,
        omega_iov_parameter_count: diagnostics.parameter_count.omega_iov,
        residual_parameter_count: diagnostics.parameter_count.residual,
        free_parameter_count: diagnostics.parameter_count.total,
        source_marginal_n2ll: diagnostics.source_marginal_n2ll,
        source_marginal_n2ll_mcse: diagnostics.source_marginal_n2ll_mcse,
        aic: diagnostics.aic,
        bic: diagnostics.bic,
        aic_mcse: diagnostics.aic_mcse,
        bic_mcse: diagnostics.bic_mcse,
        failure_reason: match &diagnostics.status {
            InformationCriteriaStatus::Unavailable { reason } => serde_json::to_string(reason).ok(),
            _ => None,
        },
    }]
}

fn information_criteria_statistics(
    iterations: usize,
    diagnostics: &InformationCriteriaDiagnostics,
    rows: &mut Vec<StatisticRow>,
) {
    let status = information_criteria_status_text(&diagnostics.status).to_string();
    rows.push(StatisticRow {
        cycle: iterations,
        kind: "information_criteria_status".to_string(),
        name: "population".to_string(),
        row: None,
        column: None,
        output_index: None,
        component: None,
        value: None,
        status: Some(status.clone()),
    });
    for (name, value) in [
        ("source_marginal_n2ll", diagnostics.source_marginal_n2ll),
        (
            "source_marginal_n2ll_mcse",
            diagnostics.source_marginal_n2ll_mcse,
        ),
        ("aic", diagnostics.aic),
        ("bic", diagnostics.bic),
        ("aic_mcse", diagnostics.aic_mcse),
        ("bic_mcse", diagnostics.bic_mcse),
    ] {
        rows.push(StatisticRow {
            cycle: iterations,
            kind: "information_criteria".to_string(),
            name: name.to_string(),
            row: None,
            column: None,
            output_index: None,
            component: None,
            value,
            status: Some(status.clone()),
        });
    }
    for (name, value) in [
        (
            "population_parameter_count",
            diagnostics.parameter_count.population,
        ),
        ("omega_parameter_count", diagnostics.parameter_count.omega),
        (
            "omega_iov_parameter_count",
            diagnostics.parameter_count.omega_iov,
        ),
        (
            "residual_parameter_count",
            diagnostics.parameter_count.residual,
        ),
        ("free_parameter_count", diagnostics.parameter_count.total),
        ("independent_subject_count", diagnostics.subject_count),
    ] {
        rows.push(StatisticRow {
            cycle: iterations,
            kind: "information_criteria_metadata".to_string(),
            name: name.to_string(),
            row: None,
            column: None,
            output_index: None,
            component: None,
            value: Some(value as f64),
            status: Some(status.clone()),
        });
    }
}

fn information_criteria_status_text(status: &InformationCriteriaStatus) -> &'static str {
    match status {
        InformationCriteriaStatus::NotRequested => "not_requested",
        InformationCriteriaStatus::Available => "available",
        InformationCriteriaStatus::AvailableWithNonconvergedModes { .. } => {
            "available_with_nonconverged_modes"
        }
        InformationCriteriaStatus::Unavailable { .. } => "unavailable",
    }
}

fn information_criteria_convention_text(
    convention: InformationCriteriaSampleSizeConvention,
) -> &'static str {
    match convention {
        InformationCriteriaSampleSizeConvention::IndependentSubjects => "independent_subjects",
    }
}

fn n6_uncertainty_statistic_rows(
    cycle: usize,
    population: &PopulationUncertaintyDiagnostics,
    conditional_modes: &[SubjectConditionalMode],
    shrinkage: &ShrinkageDiagnostics,
) -> Vec<StatisticRow> {
    let mut rows = Vec::new();
    let population_status = serde_json::to_string(&population.status)
        .unwrap_or_else(|_| "population_uncertainty_status_unserializable".to_string());
    rows.push(StatisticRow {
        cycle,
        kind: "population_uncertainty_status".into(),
        name: "observed_information_inverse".into(),
        row: None,
        column: None,
        output_index: None,
        component: None,
        value: None,
        status: Some(population_status.clone()),
    });
    rows.push(StatisticRow {
        cycle,
        kind: "population_uncertainty_regularization".into(),
        name: "regularization".into(),
        row: None,
        column: None,
        output_index: None,
        component: None,
        value: None,
        status: Some("none".into()),
    });
    if let Some(covariance) = population.free_covariance.as_ref() {
        for (row_index, row_values) in covariance.iter().enumerate() {
            for (column_index, value) in row_values.iter().enumerate() {
                rows.push(StatisticRow {
                    cycle,
                    kind: "population_uncertainty_covariance_phi".into(),
                    name: "free_coordinate_covariance".into(),
                    row: population
                        .coordinates
                        .get(row_index)
                        .map(|coordinate| coordinate.name.clone()),
                    column: population
                        .coordinates
                        .get(column_index)
                        .map(|coordinate| coordinate.name.clone()),
                    output_index: None,
                    component: None,
                    value: Some(*value),
                    status: Some(population_status.clone()),
                });
            }
        }
    }
    if let Some(condition) = population.spectral_condition_number {
        rows.push(StatisticRow {
            cycle,
            kind: "population_uncertainty_condition".into(),
            name: "spectral_condition_number".into(),
            row: None,
            column: None,
            output_index: None,
            component: None,
            value: Some(condition),
            status: Some(population_status.clone()),
        });
    }
    if let Some(standard_errors) = population.free_standard_errors.as_ref() {
        for (coordinate, standard_error) in population.coordinates.iter().zip(standard_errors) {
            rows.push(StatisticRow {
                cycle,
                kind: "population_uncertainty_se_phi".into(),
                name: coordinate.name.clone(),
                row: None,
                column: None,
                output_index: None,
                component: None,
                value: Some(*standard_error),
                status: Some(population_status.clone()),
            });
        }
    }

    for mode in conditional_modes {
        let diagnostics = &mode.uncertainty;
        let status = serde_json::to_string(&diagnostics.status)
            .unwrap_or_else(|_| "conditional_curvature_status_unserializable".to_string());
        rows.push(StatisticRow {
            cycle,
            kind: "conditional_curvature_status".into(),
            name: mode.subject_id.clone(),
            row: None,
            column: None,
            output_index: None,
            component: None,
            value: None,
            status: Some(status.clone()),
        });
        rows.push(StatisticRow {
            cycle,
            kind: "conditional_curvature_regularization".into(),
            name: mode.subject_id.clone(),
            row: None,
            column: None,
            output_index: None,
            component: None,
            value: None,
            status: Some("none".into()),
        });
        for (coordinate, step) in diagnostics
            .coordinates
            .iter()
            .zip(&diagnostics.finite_difference_steps)
        {
            rows.push(StatisticRow {
                cycle,
                kind: "conditional_curvature_step".into(),
                name: mode.subject_id.clone(),
                row: Some(coordinate.name.clone()),
                column: None,
                output_index: None,
                component: None,
                value: Some(*step),
                status: Some(status.clone()),
            });
        }
        if let Some(standard_errors) = diagnostics.latent_standard_errors.as_ref() {
            for (coordinate, standard_error) in diagnostics.coordinates.iter().zip(standard_errors)
            {
                rows.push(StatisticRow {
                    cycle,
                    kind: "conditional_latent_se".into(),
                    name: mode.subject_id.clone(),
                    row: Some(coordinate.name.clone()),
                    column: None,
                    output_index: None,
                    component: None,
                    value: Some(*standard_error),
                    status: Some(status.clone()),
                });
            }
        }
        for (kind, matrix) in [
            ("conditional_hessian", diagnostics.hessian.as_ref()),
            (
                "conditional_latent_covariance",
                diagnostics.latent_covariance.as_ref(),
            ),
        ] {
            if let Some(matrix) = matrix {
                for (row_index, row_values) in matrix.iter().enumerate() {
                    for (column_index, value) in row_values.iter().enumerate() {
                        rows.push(StatisticRow {
                            cycle,
                            kind: kind.into(),
                            name: mode.subject_id.clone(),
                            row: diagnostics
                                .coordinates
                                .get(row_index)
                                .map(|coordinate| coordinate.name.clone()),
                            column: diagnostics
                                .coordinates
                                .get(column_index)
                                .map(|coordinate| coordinate.name.clone()),
                            output_index: None,
                            component: None,
                            value: Some(*value),
                            status: Some(status.clone()),
                        });
                    }
                }
            }
        }
        if let Some(condition) = diagnostics.spectral_condition_number {
            rows.push(StatisticRow {
                cycle,
                kind: "conditional_curvature_condition".into(),
                name: mode.subject_id.clone(),
                row: None,
                column: None,
                output_index: None,
                component: None,
                value: Some(condition),
                status: Some(status),
            });
        }
    }

    let mut push_shrinkage =
        |kind: &str, effect: &str, value: &crate::estimation::parametric::ShrinkageValue| {
            let (numeric, status) = match value {
                crate::estimation::parametric::ShrinkageValue::Available { value, .. } => {
                    (Some(*value), "available".to_string())
                }
                crate::estimation::parametric::ShrinkageValue::Unavailable { reason } => (
                    None,
                    serde_json::to_string(reason)
                        .unwrap_or_else(|_| "shrinkage_status_unserializable".to_string()),
                ),
            };
            rows.push(StatisticRow {
                cycle,
                kind: kind.into(),
                name: effect.into(),
                row: None,
                column: None,
                output_index: None,
                component: None,
                value: numeric,
                status: Some(status),
            });
        };
    for value in &shrinkage.eta_posterior_mean {
        push_shrinkage(
            "eta_shrinkage_posterior_mean",
            &value.effect,
            &value.shrinkage,
        );
    }
    for value in &shrinkage.eta_map {
        push_shrinkage("eta_shrinkage_map", &value.effect, &value.shrinkage);
    }
    for value in &shrinkage.kappa_posterior_mean {
        push_shrinkage(
            "kappa_shrinkage_posterior_mean",
            &value.effect,
            &value.shrinkage,
        );
    }
    for value in &shrinkage.kappa_map {
        push_shrinkage("kappa_shrinkage_map", &value.effect, &value.shrinkage);
    }
    rows
}

fn n6_uncertainty_statistics<E: Equation>(
    result: &ParametricResult<E>,
    rows: &mut Vec<StatisticRow>,
) {
    rows.extend(n6_uncertainty_statistic_rows(
        result.iterations(),
        result.population_uncertainty(),
        result.conditional_modes(),
        result.shrinkage(),
    ));
}

fn information_statistics(
    iterations: usize,
    diagnostics: &InformationDiagnostics,
    rows: &mut Vec<StatisticRow>,
) {
    let status = information_status_text(&diagnostics.status);
    for (index, value) in diagnostics.delta.iter().enumerate() {
        let name = diagnostics
            .coordinates
            .get(index)
            .map(|coordinate| coordinate.name.as_str())
            .unwrap_or("");
        let mut row = statistic(
            iterations,
            "information_delta",
            name,
            None,
            None,
            None,
            None,
            *value,
        );
        row.status = Some(status.clone());
        rows.push(row);
    }
    for (kind, matrix) in [
        ("information_g", &diagnostics.g),
        (
            "information_complete_hessian",
            &diagnostics.expected_complete_hessian,
        ),
        (
            "information_observed_hessian",
            &diagnostics.observed_hessian,
        ),
        ("observed_information", &diagnostics.observed_information),
    ] {
        for (row_index, values) in matrix.iter().enumerate() {
            for (column_index, value) in values.iter().enumerate() {
                let row_name = diagnostics
                    .coordinates
                    .get(row_index)
                    .map(|coordinate| coordinate.name.clone());
                let column_name = diagnostics
                    .coordinates
                    .get(column_index)
                    .map(|coordinate| coordinate.name.clone());
                let mut row = statistic(
                    iterations,
                    kind,
                    "",
                    row_name,
                    column_name,
                    None,
                    None,
                    *value,
                );
                row.status = Some(status.clone());
                rows.push(row);
            }
        }
    }
}

fn markov_variance_statistics(
    iterations: usize,
    diagnostics: &MarkovSimulationVarianceDiagnostics,
    rows: &mut Vec<StatisticRow>,
) {
    push_markov_metadata(
        iterations,
        "markov_status",
        "aggregate",
        0.0,
        &diagnostics.status,
        rows,
    );
    push_markov_metadata(
        iterations,
        "markov_chain_count",
        "retained_chains",
        diagnostics.chain_count as f64,
        &diagnostics.status,
        rows,
    );
    push_markov_metadata(
        iterations,
        "markov_n_avg",
        "averaged_iterations",
        diagnostics.n_avg as f64,
        &diagnostics.status,
        rows,
    );
    if let Some(config) = diagnostics.config {
        for (name, value) in [
            ("seed", config.seed as f64),
            ("warmup_transitions", config.warmup_transitions as f64),
            ("draws_per_chain", config.draws_per_chain as f64),
            ("batch_size", config.batch_size as f64),
            ("lugsail_r", config.lugsail.r as f64),
            ("lugsail_c", config.lugsail.c),
            ("diagnostic_chains", config.diagnostic_chains as f64),
            ("max_trace_bytes", config.max_trace_bytes as f64),
        ] {
            push_markov_metadata(
                iterations,
                "markov_config",
                name,
                value,
                &diagnostics.status,
                rows,
            );
        }
    }
    for chain in &diagnostics.chains {
        let chain_name = format!("chain_{}", chain.chain);
        for (name, value) in [
            ("retained_proposals", chain.proposals),
            ("retained_accepts", chain.accepts),
            ("retained_state_changes", chain.state_changes),
        ] {
            push_markov_metadata(
                iterations,
                "markov_chain_count",
                &format!("{chain_name}_{name}"),
                value as f64,
                &chain.status,
                rows,
            );
        }
        let chain_status = markov_variance_status_text(&chain.status);
        for (kind, matrix) in [
            ("markov_bm_batch", &chain.bm_batch),
            ("markov_bm_batch_over_r", &chain.bm_batch_over_r),
            ("markov_lugsail_lrv", &chain.lugsail_lrv),
        ] {
            append_markov_matrix(
                iterations,
                kind,
                &chain_name,
                matrix,
                &diagnostics.coordinates,
                &chain_status,
                rows,
            );
        }
    }
    for (kind, matrix, matrix_status) in [
        (
            "markov_combined_lambda",
            &diagnostics.lambda,
            &diagnostics.lambda_status,
        ),
        ("markov_xi", &diagnostics.xi, &diagnostics.xi_status),
        (
            "markov_xi_over_n_avg",
            &diagnostics.simulation_covariance,
            &diagnostics.simulation_covariance_status,
        ),
    ] {
        append_markov_matrix(
            iterations,
            kind,
            "",
            matrix,
            &diagnostics.coordinates,
            &markov_variance_status_text(matrix_status),
            rows,
        );
    }

    // ── Rank/mixing diagnostic statistics ──
    {
        let rank = &diagnostics.rank_diagnostics;
        // Always emit aggregate metadata even when disabled.
        let rank_status_text = rank_diagnostic_status_text(&rank.status);
        for (name, value) in [
            ("rank_status", 0.0),
            ("diagnostic_chains", rank.diagnostic_chains as f64),
            ("fit_chains", rank.original_chains as f64),
            ("draws_per_chain", rank.draws_per_chain as f64),
            ("max_trace_bytes", rank.max_trace_bytes as f64),
            (
                "accounted_peak_trace_bytes_required",
                rank.accounted_peak_trace_bytes_required as f64,
            ),
            (
                "accounted_peak_trace_bytes_used",
                rank.accounted_peak_trace_bytes_used as f64,
            ),
        ] {
            push_rank_metadata(
                iterations,
                "markov_rank_status",
                name,
                value,
                &rank.status,
                rows,
            );
        }
        for (name, value) in [
            ("worst_rhat", rank.worst_rhat),
            ("min_bulk_ess", rank.min_bulk_ess),
            (
                "min_avg_ess_per_split_chain",
                rank.min_avg_ess_per_split_chain,
            ),
        ] {
            if let Some(value) = value {
                push_rank_metadata(
                    iterations,
                    "markov_rank_status",
                    name,
                    value,
                    &rank.status,
                    rows,
                );
            }
        }

        // Per-chain LRV status is always emitted at the configured chain
        // index; matrix rows follow only when that same chain has a matrix.
        for chain_idx in 0..rank.diagnostic_chains {
            let chain_name = format!("diagnostic_chain_{chain_idx}");
            let chain_status = rank
                .lrv_chain_statuses
                .get(chain_idx)
                .cloned()
                .unwrap_or(RankDiagnosticStatus::Unavailable);
            push_rank_metadata(
                iterations,
                "markov_rank_lrv_chain_status",
                &chain_name,
                0.0,
                &chain_status,
                rows,
            );
            if let Some(Some(lrv_matrix)) = rank.lrv_per_chain.get(chain_idx) {
                append_markov_matrix(
                    iterations,
                    "markov_rank_lrv_per_chain",
                    &chain_name,
                    lrv_matrix,
                    &diagnostics.coordinates,
                    rank_diagnostic_status_text(&chain_status),
                    rows,
                );
            }
        }

        // Diagnostic-mean LRV matrix.
        if let Some(ref lrv) = rank.diagnostic_mean_lrv {
            append_markov_matrix(
                iterations,
                "markov_rank_lrv_diagnostic_mean",
                "",
                lrv,
                &diagnostics.coordinates,
                rank_status_text,
                rows,
            );
        }
        // Operational LRV matrix.
        if let Some(ref lrv) = rank.operational_lrv {
            append_markov_matrix(
                iterations,
                "markov_rank_lrv_operational",
                "",
                lrv,
                &diagnostics.coordinates,
                rank_status_text,
                rows,
            );
        }

        // Per-coordinate rank diagnostics. Use each trace's actual status.
        for trace in &rank.traces {
            let coord_status = rank_diagnostic_status_text(&trace.status);
            let label = diagnostic_trace_label(&trace.trace);
            for (statistic_name, statistic_status) in [
                ("rank_rhat", &trace.rank_rhat_status),
                ("folded_rhat", &trace.folded_rhat_status),
                ("max_rhat", &trace.max_rhat_status),
                ("bulk_ess", &trace.bulk_ess_status),
            ] {
                push_rank_metadata(
                    iterations,
                    "markov_rank_statistic_status",
                    &format!("{statistic_name}:{label}"),
                    0.0,
                    statistic_status,
                    rows,
                );
            }
            if let Some(value) = trace.rank_rhat {
                let mut row = statistic(
                    iterations,
                    "markov_rank_rhat",
                    &format!("rank_rhat:{label}"),
                    None,
                    None,
                    None,
                    None,
                    value,
                );
                row.status = Some(rank_diagnostic_status_text(&trace.rank_rhat_status).to_string());
                rows.push(row);
            }
            if let Some(value) = trace.folded_rhat {
                let mut row = statistic(
                    iterations,
                    "markov_rank_rhat",
                    &format!("folded_rhat:{label}"),
                    None,
                    None,
                    None,
                    None,
                    value,
                );
                row.status =
                    Some(rank_diagnostic_status_text(&trace.folded_rhat_status).to_string());
                rows.push(row);
            }
            if let Some(value) = trace.max_rhat {
                let mut row = statistic(
                    iterations,
                    "markov_rank_rhat",
                    &format!("max_rhat:{label}"),
                    None,
                    None,
                    None,
                    None,
                    value,
                );
                row.status = Some(rank_diagnostic_status_text(&trace.max_rhat_status).to_string());
                rows.push(row);
            }
            if let Some(value) = trace.bulk_ess {
                let mut row = statistic(
                    iterations,
                    "markov_rank_ess",
                    &format!("bulk_ess:{label}"),
                    None,
                    None,
                    None,
                    None,
                    value,
                );
                row.status = Some(rank_diagnostic_status_text(&trace.bulk_ess_status).to_string());
                rows.push(row);
            }
            if let Some(value) = trace.avg_ess_per_split_chain {
                let mut row = statistic(
                    iterations,
                    "markov_rank_ess",
                    &format!("avg_ess_per_split_chain:{label}"),
                    None,
                    None,
                    None,
                    None,
                    value,
                );
                row.status = Some(rank_diagnostic_status_text(&trace.bulk_ess_status).to_string());
                rows.push(row);
            }
            if let Some(value) = trace.tau {
                let mut row = statistic(
                    iterations,
                    "markov_rank_ess",
                    &format!("tau:{label}"),
                    None,
                    None,
                    None,
                    None,
                    value,
                );
                row.status = Some(rank_diagnostic_status_text(&trace.bulk_ess_status).to_string());
                rows.push(row);
            }
            // Per-coordinate aggregate status (PartialAvailability when only a
            // subset of rank Rhat, folded Rhat, and ESS succeeded).
            let mut row = statistic(
                iterations,
                "markov_rank_coord_status",
                &format!("status:{label}"),
                None,
                None,
                None,
                None,
                0.0,
            );
            row.status = Some(coord_status.to_string());
            rows.push(row);
        }
    }
}

fn operational_convergence_statistics(
    iterations: usize,
    diagnostics: &OperationalConvergenceDiagnostics,
    rows: &mut Vec<StatisticRow>,
) {
    // Always emit lifecycle metadata. When `checks` is empty the diagnostics
    // are explicitly absent.
    push_operational_metadata(
        iterations,
        "operational_convergence_checkpoints",
        "total_checkpoints",
        diagnostics.checks.len() as f64,
        rows,
    );
    let flag = if diagnostics.used_for_termination {
        1.0
    } else {
        0.0
    };
    push_operational_metadata(
        iterations,
        "operational_convergence_flag",
        "used_for_termination",
        flag,
        rows,
    );
    let flag = if diagnostics.final_check_reused {
        1.0
    } else {
        0.0
    };
    push_operational_metadata(
        iterations,
        "operational_convergence_flag",
        "final_check_reused",
        flag,
        rows,
    );

    for (check_index, check) in diagnostics.checks.iter().enumerate() {
        let row_start = rows.len();
        let prefix = format!("check_{check_index}");

        // Checkpoint-level metadata.
        push_operational_metadata(
            iterations,
            "operational_convergence_check",
            &format!("{prefix}:iteration"),
            check.iteration as f64,
            rows,
        );
        push_operational_metadata(
            iterations,
            "operational_convergence_check",
            &format!("{prefix}:averaged_iterations"),
            check.averaged_iterations as f64,
            rows,
        );
        let flag = if check.scheduled { 1.0 } else { 0.0 };
        push_operational_metadata(
            iterations,
            "operational_convergence_flag",
            &format!("{prefix}:scheduled"),
            flag,
            rows,
        );
        let flag = if check.mandatory_final { 1.0 } else { 0.0 };
        push_operational_metadata(
            iterations,
            "operational_convergence_flag",
            &format!("{prefix}:mandatory_final"),
            flag,
            rows,
        );
        let outcome_value = match &check.outcome {
            OperationalConvergenceOutcome::Passed => 1.0,
            OperationalConvergenceOutcome::Failed { .. } => 0.0,
            OperationalConvergenceOutcome::Ineligible { .. } => -1.0,
        };
        push_operational_metadata(
            iterations,
            "operational_convergence_outcome",
            &format!("{prefix}:outcome"),
            outcome_value,
            rows,
        );
        // Numeric checkpoint fields. Absent values (None) are not emitted;
        // only finite available values produce rows.
        if let Some(z) = check.z_quantile {
            if z.is_finite() {
                push_operational_metadata(
                    iterations,
                    "operational_convergence_check",
                    &format!("{prefix}:z_quantile"),
                    z,
                    rows,
                );
            }
        }
        if let Some(imp) = check.implied_minimum_ess {
            if imp.is_finite() {
                push_operational_metadata(
                    iterations,
                    "operational_convergence_check",
                    &format!("{prefix}:implied_minimum_ess"),
                    imp,
                    rows,
                );
            }
        }
        if let Some(seed) = check.checkpoint_seed {
            push_operational_metadata(
                iterations,
                "operational_convergence_check",
                &format!("{prefix}:checkpoint_seed_high_u32"),
                (seed >> 32) as u32 as f64,
                rows,
            );
            push_operational_metadata(
                iterations,
                "operational_convergence_check",
                &format!("{prefix}:checkpoint_seed_low_u32"),
                seed as u32 as f64,
                rows,
            );
        }
        // Averaged-candidate free coordinates. Retain only finite values.
        for (coord_index, value) in check.candidate_free_coordinates.iter().enumerate() {
            if value.is_finite() {
                push_operational_metadata(
                    iterations,
                    "operational_convergence_coordinate",
                    &format!("{prefix}:free_coordinate_{coord_index}"),
                    *value,
                    rows,
                );
            }
        }
        // Per-criterion evaluations.
        for (criterion_index, criterion) in check.criteria.iter().enumerate() {
            let crit_prefix = format!("{prefix}:crit_{criterion_index}");
            let status_value = match criterion.status {
                crate::results::OperationalConvergenceCriterionStatus::Satisfied => 1.0,
                crate::results::OperationalConvergenceCriterionStatus::NotSatisfied => 0.0,
                crate::results::OperationalConvergenceCriterionStatus::Unavailable(_) => -1.0,
            };
            push_operational_metadata(
                iterations,
                "operational_convergence_criterion_status",
                &format!("{crit_prefix}:{}:status", criterion.name),
                status_value,
                rows,
            );
            if let Some(row) = rows.last_mut() {
                row.status = Some(match &criterion.status {
                    crate::results::OperationalConvergenceCriterionStatus::Satisfied => {
                        "satisfied".to_string()
                    }
                    crate::results::OperationalConvergenceCriterionStatus::NotSatisfied => {
                        "not_satisfied".to_string()
                    }
                    crate::results::OperationalConvergenceCriterionStatus::Unavailable(reason) => {
                        format!("unavailable: {reason}")
                    }
                });
            }
            if let Some(observed) = criterion.observed {
                if observed.is_finite() {
                    push_operational_metadata(
                        iterations,
                        "operational_convergence_criterion",
                        &format!("{crit_prefix}:observed"),
                        observed,
                        rows,
                    );
                }
            }
            if criterion.threshold.is_finite() {
                push_operational_metadata(
                    iterations,
                    "operational_convergence_criterion",
                    &format!("{crit_prefix}:threshold"),
                    criterion.threshold,
                    rows,
                );
            }
        }
        // Per-trace rank diagnostics and matrix availability from the exact
        // frozen-kernel object serialized in JSON. Status rows are emitted even
        // when an optional numeric or matrix value is absent; their zero value
        // is an availability code, never a fabricated diagnostic value.
        if let Some(ref markov) = check.markov {
            let rank = &markov.rank_diagnostics;
            for trace in &rank.traces {
                let label = diagnostic_trace_label(&trace.trace);
                for (name, status) in [
                    ("rank_rhat", &trace.rank_rhat_status),
                    ("folded_rhat", &trace.folded_rhat_status),
                    ("max_rhat", &trace.max_rhat_status),
                    ("bulk_ess", &trace.bulk_ess_status),
                    ("coordinate_aggregate", &trace.status),
                ] {
                    push_rank_metadata(
                        iterations,
                        "operational_convergence_trace_status",
                        &format!("{prefix}:{name}:{label}"),
                        0.0,
                        status,
                        rows,
                    );
                }
                if let Some(value) = trace.rank_rhat {
                    let mut row = statistic(
                        iterations,
                        "operational_convergence_rhat",
                        &format!("{prefix}:rank_rhat:{label}"),
                        None,
                        None,
                        None,
                        None,
                        value,
                    );
                    row.status =
                        Some(rank_diagnostic_status_text(&trace.rank_rhat_status).to_string());
                    rows.push(row);
                }
                if let Some(value) = trace.folded_rhat {
                    let mut row = statistic(
                        iterations,
                        "operational_convergence_rhat",
                        &format!("{prefix}:folded_rhat:{label}"),
                        None,
                        None,
                        None,
                        None,
                        value,
                    );
                    row.status =
                        Some(rank_diagnostic_status_text(&trace.folded_rhat_status).to_string());
                    rows.push(row);
                }
                if let Some(value) = trace.max_rhat {
                    let mut row = statistic(
                        iterations,
                        "operational_convergence_rhat",
                        &format!("{prefix}:max_rhat:{label}"),
                        None,
                        None,
                        None,
                        None,
                        value,
                    );
                    row.status =
                        Some(rank_diagnostic_status_text(&trace.max_rhat_status).to_string());
                    rows.push(row);
                }
                if let Some(value) = trace.bulk_ess {
                    let mut row = statistic(
                        iterations,
                        "operational_convergence_ess",
                        &format!("{prefix}:bulk_ess:{label}"),
                        None,
                        None,
                        None,
                        None,
                        value,
                    );
                    row.status =
                        Some(rank_diagnostic_status_text(&trace.bulk_ess_status).to_string());
                    rows.push(row);
                }
                if let Some(value) = trace.avg_ess_per_split_chain {
                    let mut row = statistic(
                        iterations,
                        "operational_convergence_ess",
                        &format!("{prefix}:avg_ess_per_split_chain:{label}"),
                        None,
                        None,
                        None,
                        None,
                        value,
                    );
                    row.status =
                        Some(rank_diagnostic_status_text(&trace.bulk_ess_status).to_string());
                    rows.push(row);
                }
            }
            // Aggregate rank values each receive an explicit status row,
            // regardless of whether the corresponding numeric exists.
            for name in ["worst_rhat", "min_bulk_ess", "min_avg_ess_per_split_chain"] {
                push_rank_metadata(
                    iterations,
                    "operational_convergence_rank_aggregate_status",
                    &format!("{prefix}:{name}"),
                    0.0,
                    &rank.status,
                    rows,
                );
            }

            // Every configured rank LRV chain retains its JSON index and status.
            for chain_index in 0..rank.diagnostic_chains {
                let chain_status = rank
                    .lrv_chain_statuses
                    .get(chain_index)
                    .cloned()
                    .unwrap_or(RankDiagnosticStatus::Unavailable);
                let chain_name = format!("{prefix}:diagnostic_chain_{chain_index}");
                push_rank_metadata(
                    iterations,
                    "operational_convergence_lrv_chain_status",
                    &chain_name,
                    0.0,
                    &chain_status,
                    rows,
                );
                if let Some(Some(matrix)) = rank.lrv_per_chain.get(chain_index) {
                    append_markov_matrix(
                        iterations,
                        "operational_convergence_lrv_per_chain",
                        &chain_name,
                        matrix,
                        &markov.coordinates,
                        rank_diagnostic_status_text(&chain_status),
                        rows,
                    );
                }
            }
            for (name, matrix) in [
                ("diagnostic_mean", rank.diagnostic_mean_lrv.as_ref()),
                ("operational", rank.operational_lrv.as_ref()),
            ] {
                push_rank_metadata(
                    iterations,
                    "operational_convergence_lrv_aggregate_status",
                    &format!("{prefix}:{name}"),
                    0.0,
                    &rank.status,
                    rows,
                );
                if let Some(matrix) = matrix {
                    append_markov_matrix(
                        iterations,
                        "operational_convergence_lrv_aggregate",
                        &format!("{prefix}:{name}"),
                        matrix,
                        &markov.coordinates,
                        rank_diagnostic_status_text(&rank.status),
                        rows,
                    );
                }
            }

            // Information-mapped matrices used by eligibility mirror the JSON
            // status exactly, whether or not matrix cells exist.
            for (name, matrix, status) in [
                ("lambda", &markov.lambda, &markov.lambda_status),
                ("xi", &markov.xi, &markov.xi_status),
                (
                    "simulation_covariance",
                    &markov.simulation_covariance,
                    &markov.simulation_covariance_status,
                ),
            ] {
                push_markov_metadata(
                    iterations,
                    "operational_convergence_matrix_status",
                    &format!("{prefix}:{name}"),
                    0.0,
                    status,
                    rows,
                );
                append_markov_matrix(
                    iterations,
                    "operational_convergence_matrix",
                    &format!("{prefix}:{name}"),
                    matrix,
                    &markov.coordinates,
                    &markov_variance_status_text(status),
                    rows,
                );
            }

            // Checkpoint worst/meta from markov.
            if let Some(value) = rank.worst_rhat {
                let mut row = statistic(
                    iterations,
                    "operational_convergence_rhat",
                    &format!("{prefix}:worst_rhat"),
                    None,
                    None,
                    None,
                    None,
                    value,
                );
                row.status =
                    Some(rank_diagnostic_status_text(&markov.rank_diagnostics.status).to_string());
                rows.push(row);
            }
            if let Some(value) = rank.min_bulk_ess {
                let mut row = statistic(
                    iterations,
                    "operational_convergence_ess",
                    &format!("{prefix}:min_bulk_ess"),
                    None,
                    None,
                    None,
                    None,
                    value,
                );
                row.status =
                    Some(rank_diagnostic_status_text(&markov.rank_diagnostics.status).to_string());
                rows.push(row);
            }
            if let Some(value) = rank.min_avg_ess_per_split_chain {
                let mut row = statistic(
                    iterations,
                    "operational_convergence_ess",
                    &format!("{prefix}:min_avg_ess_per_split_chain"),
                    None,
                    None,
                    None,
                    None,
                    value,
                );
                row.status = Some(rank_diagnostic_status_text(&rank.status).to_string());
                rows.push(row);
            }
        } else {
            for name in ["lambda", "xi", "simulation_covariance"] {
                let mut row = statistic(
                    iterations,
                    "operational_convergence_matrix_status",
                    &format!("{prefix}:{name}"),
                    None,
                    None,
                    None,
                    None,
                    0.0,
                );
                row.status = Some("unavailable: no frozen-kernel diagnostic".to_string());
                rows.push(row);
            }
            for name in ["diagnostic_mean", "operational"] {
                let mut row = statistic(
                    iterations,
                    "operational_convergence_lrv_aggregate_status",
                    &format!("{prefix}:{name}"),
                    None,
                    None,
                    None,
                    None,
                    0.0,
                );
                row.status = Some("unavailable: no frozen-kernel diagnostic".to_string());
                rows.push(row);
            }
        }
        for row in &mut rows[row_start..] {
            row.cycle = check.iteration;
        }
    }
}

fn push_operational_metadata(
    iterations: usize,
    kind: &str,
    name: &str,
    value: f64,
    rows: &mut Vec<StatisticRow>,
) {
    rows.push(statistic(
        iterations, kind, name, None, None, None, None, value,
    ));
}

fn push_markov_metadata(
    iterations: usize,
    kind: &str,
    name: &str,
    value: f64,
    status: &MarkovSimulationVarianceStatus,
    rows: &mut Vec<StatisticRow>,
) {
    let mut row = statistic(iterations, kind, name, None, None, None, None, value);
    row.status = Some(markov_variance_status_text(status));
    rows.push(row);
}

fn push_rank_metadata(
    iterations: usize,
    kind: &str,
    name: &str,
    value: f64,
    status: &RankDiagnosticStatus,
    rows: &mut Vec<StatisticRow>,
) {
    let mut row = statistic(iterations, kind, name, None, None, None, None, value);
    row.status = Some(rank_diagnostic_status_text(status).to_string());
    rows.push(row);
}

fn append_markov_matrix(
    iterations: usize,
    kind: &str,
    name: &str,
    matrix: &[Vec<f64>],
    coordinates: &[crate::results::InformationCoordinate],
    status: &str,
    rows: &mut Vec<StatisticRow>,
) {
    for (row_index, values) in matrix.iter().enumerate() {
        for (column_index, value) in values.iter().enumerate() {
            let mut row = statistic(
                iterations,
                kind,
                name,
                coordinates.get(row_index).map(|value| value.name.clone()),
                coordinates
                    .get(column_index)
                    .map(|value| value.name.clone()),
                None,
                None,
                *value,
            );
            row.status = Some(status.to_string());
            rows.push(row);
        }
    }
}

fn markov_variance_status_text(status: &MarkovSimulationVarianceStatus) -> String {
    match status {
        MarkovSimulationVarianceStatus::Disabled => "disabled".into(),
        MarkovSimulationVarianceStatus::AverageNotApplied => "average_not_applied".into(),
        MarkovSimulationVarianceStatus::NoFreeCoordinates => "no_free_coordinates".into(),
        MarkovSimulationVarianceStatus::ExactZeroNoLatentState => {
            "exact_zero_no_latent_state".into()
        }
        MarkovSimulationVarianceStatus::InformationUnavailable(reason) => {
            format!("information_unavailable: {reason}")
        }
        MarkovSimulationVarianceStatus::InvalidConfiguration(reason) => {
            format!("invalid_configuration: {reason}")
        }
        MarkovSimulationVarianceStatus::TraceMemoryAccountingOverflow => {
            "trace_memory_accounting_overflow".into()
        }
        MarkovSimulationVarianceStatus::CoordinateMismatch => "coordinate_mismatch".into(),
        MarkovSimulationVarianceStatus::UnsupportedScore(reason) => {
            format!("unsupported_score: {reason}")
        }
        MarkovSimulationVarianceStatus::NonFinite => "non_finite".into(),
        MarkovSimulationVarianceStatus::NonSymmetric => "non_symmetric".into(),
        MarkovSimulationVarianceStatus::Indefinite => "indefinite".into(),
        MarkovSimulationVarianceStatus::StuckChain { chain } => format!("stuck_chain: {chain}"),
        MarkovSimulationVarianceStatus::AssumptionsUnverified => "assumptions_unverified".into(),
    }
}

fn information_status_text(status: &crate::results::InformationStatus) -> String {
    use crate::results::InformationStatus;

    match status {
        InformationStatus::Available => "available".to_string(),
        InformationStatus::NoFreeCoordinates => "no_free_coordinates".to_string(),
        InformationStatus::NonFinite => "non_finite".to_string(),
        InformationStatus::ObservedInformationNotPositiveDefinite => {
            "observed_information_not_positive_definite".to_string()
        }
        InformationStatus::Unsupported(reason) => format!("unsupported: {reason}"),
        InformationStatus::Ineligible(reason) => format!("ineligible: {reason}"),
    }
}

fn rank_diagnostic_status_text(status: &RankDiagnosticStatus) -> &str {
    match status {
        RankDiagnosticStatus::Disabled => "disabled",
        RankDiagnosticStatus::NoLatent => "no_latent",
        RankDiagnosticStatus::ScoreUnavailable => "score_unavailable",
        RankDiagnosticStatus::Unavailable => "unavailable",
        RankDiagnosticStatus::PartialAvailability => "partial_availability",
        RankDiagnosticStatus::NoChains => "no_chains",
        RankDiagnosticStatus::TooFewChains => "too_few_chains",
        RankDiagnosticStatus::UnequalChainLengths => "unequal_chain_lengths",
        RankDiagnosticStatus::TooFewDraws => "too_few_draws",
        RankDiagnosticStatus::OddDraws => "odd_draws",
        RankDiagnosticStatus::TraceByteCapExceeded => "trace_byte_cap_exceeded",
        RankDiagnosticStatus::TraceMemoryAccountingOverflow => "trace_memory_accounting_overflow",
        RankDiagnosticStatus::NonFiniteDraws => "non_finite_draws",
        RankDiagnosticStatus::ConstantDraws => "constant_draws",
        RankDiagnosticStatus::InvalidVariance => "invalid_variance",
        RankDiagnosticStatus::NonPositiveTau => "non_positive_tau",
        RankDiagnosticStatus::Available => "available",
    }
}

fn diagnostic_trace_label(trace: &DiagnosticTraceCoordinate) -> String {
    match trace {
        DiagnosticTraceCoordinate::Score { index, name, .. } => {
            format!("score[{index}]:{name}")
        }
        DiagnosticTraceCoordinate::Eta {
            subject,
            effect_index,
            effect_name,
        } => {
            format!("eta:{subject}:{effect_index}:{effect_name}")
        }
        DiagnosticTraceCoordinate::Kappa {
            subject,
            occasion_index,
            effect_index,
            effect_name,
        } => {
            format!("kappa:{subject}:{occasion_index}:{effect_index}:{effect_name}")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn statistic(
    cycle: usize,
    kind: &str,
    name: &str,
    row: Option<String>,
    column: Option<String>,
    output_index: Option<usize>,
    component: Option<&str>,
    value: f64,
) -> StatisticRow {
    StatisticRow {
        cycle,
        kind: kind.to_string(),
        name: name.to_string(),
        row,
        column,
        output_index,
        component: component.map(str::to_string),
        value: Some(value),
        status: None,
    }
}

fn validate_prediction_pair(
    subject: &str,
    population: &Prediction,
    conditional: &Prediction,
) -> Result<()> {
    if population.time() != conditional.time()
        || population.outeq() != conditional.outeq()
        || population.occasion() != conditional.occasion()
        || population.observation() != conditional.observation()
        || population.censoring() != conditional.censoring()
    {
        bail!("population and conditional prediction metadata mismatch for subject '{subject}' at time {}", population.time());
    }
    Ok(())
}

fn scale_text(scale: ParameterScale) -> String {
    match scale {
        ParameterScale::Identity => "identity".to_string(),
        ParameterScale::Log => "log".to_string(),
        ParameterScale::Logit { lower, upper } => format!("logit({lower},{upper})"),
        ParameterScale::Probit { lower, upper } => format!("probit({lower},{upper})"),
    }
}

fn phase_text(phase: SaemPhase) -> &'static str {
    match phase {
        SaemPhase::BurnIn => "burn_in",
        SaemPhase::Exploration => "exploration",
        SaemPhase::Smoothing => "smoothing",
    }
}

fn censor_text(censor: Censor) -> &'static str {
    match censor {
        Censor::None => "none",
        Censor::BLOQ => "bloq",
        Censor::ALOQ => "aloq",
    }
}

fn warning_record(warning: &ParametricWarning) -> ParametricWarningRecord {
    match warning {
        ParametricWarning::OmegaUpdateRejected {
            first_iteration,
            cycles,
        } => warning_values("omega_update_rejected", None, *first_iteration, *cycles),
        ParametricWarning::OmegaIovUpdateRejected {
            first_iteration,
            cycles,
        } => warning_values("omega_iov_update_rejected", None, *first_iteration, *cycles),
        ParametricWarning::OmegaBoundaryRejection {
            first_iteration,
            longest_run,
        } => warning_values(
            "omega_boundary_rejection_run",
            None,
            *first_iteration,
            *longest_run,
        ),
        ParametricWarning::OmegaIovBoundaryRejection {
            first_iteration,
            longest_run,
        } => warning_values(
            "omega_iov_boundary_rejection_run",
            None,
            *first_iteration,
            *longest_run,
        ),
        ParametricWarning::EtaNonFiniteProposals {
            first_iteration,
            count,
        } => warning_values("eta_nonfinite_proposals", None, *first_iteration, *count),
        ParametricWarning::EtaBlockNonFiniteProposals {
            first_iteration,
            count,
        } => warning_values(
            "eta_block_nonfinite_proposals",
            None,
            *first_iteration,
            *count,
        ),
        ParametricWarning::KappaNonFiniteProposals {
            first_iteration,
            count,
        } => warning_values("kappa_nonfinite_proposals", None, *first_iteration, *count),
        ParametricWarning::ResidualUpdateRejected {
            output,
            first_iteration,
            cycles,
        } => warning_values(
            "residual_update_rejected",
            Some(output.clone()),
            *first_iteration,
            *cycles,
        ),
        ParametricWarning::ProportionalPredictionFloor {
            output,
            first_iteration,
            count,
        } => warning_values(
            "proportional_prediction_floor",
            Some(output.clone()),
            *first_iteration,
            *count,
        ),
        ParametricWarning::NonFiniteResidualPrediction {
            output,
            first_iteration,
            count,
        } => warning_values(
            "nonfinite_residual_prediction",
            Some(output.clone()),
            *first_iteration,
            *count,
        ),
        ParametricWarning::ExponentialDomainViolation {
            output,
            first_iteration,
            count,
        } => warning_values(
            "exponential_domain_violation",
            Some(output.clone()),
            *first_iteration,
            *count,
        ),
        ParametricWarning::CombinedAdditiveCollapse {
            output,
            first_iteration,
            cycles,
        } => warning_values(
            "combined_additive_collapse",
            Some(output.clone()),
            *first_iteration,
            *cycles,
        ),
        ParametricWarning::ResidualOptimizerNotConverged {
            output,
            first_iteration,
            cycles,
        } => warning_values(
            "residual_optimizer_not_converged",
            Some(output.clone()),
            *first_iteration,
            *cycles,
        ),
        ParametricWarning::MarginalLikelihoodUnavailable { subjects } => ParametricWarningRecord {
            kind: "marginal_likelihood_unavailable".to_string(),
            output: None,
            first_cycle: 0,
            count: subjects.len(),
            subjects: Some(subjects.clone()),
        },
        ParametricWarning::MarginalLikelihoodNonconvergedModes { subjects } => {
            ParametricWarningRecord {
                kind: "marginal_likelihood_nonconverged_modes".to_string(),
                output: None,
                first_cycle: 0,
                count: subjects.len(),
                subjects: Some(subjects.clone()),
            }
        }
    }
}

fn warning_values(
    kind: &str,
    output: Option<String>,
    first_cycle: usize,
    count: usize,
) -> ParametricWarningRecord {
    ParametricWarningRecord {
        kind: kind.to_string(),
        output,
        first_cycle,
        count,
        subjects: None,
    }
}

fn write_csv<T: Serialize>(path: &Path, rows: &[T], headers: &[&str]) -> Result<()> {
    create_parent_dir(path)?;
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("failed to create '{}'", path.display()))?;
    if rows.is_empty() {
        writer
            .write_record(headers)
            .with_context(|| format!("failed to write headers to '{}'", path.display()))?;
    } else {
        for row in rows {
            writer
                .serialize(row)
                .with_context(|| format!("failed to serialize row to '{}'", path.display()))?;
        }
    }
    writer
        .flush()
        .with_context(|| format!("failed to flush '{}'", path.display()))
}

fn create_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory '{}'", parent.display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_rank_and_index_preserving_lrv_rows_keep_exact_statuses() {
        use crate::results::RankMixingDiagnostic;

        let mut diagnostics = MarkovSimulationVarianceDiagnostics::disabled();
        diagnostics.rank_diagnostics.diagnostic_chains = 2;
        diagnostics.rank_diagnostics.status = RankDiagnosticStatus::PartialAvailability;
        diagnostics.rank_diagnostics.lrv_per_chain = vec![None, Some(vec![vec![2.0]])];
        diagnostics.rank_diagnostics.lrv_chain_statuses = vec![
            RankDiagnosticStatus::ScoreUnavailable,
            RankDiagnosticStatus::Available,
        ];
        diagnostics.rank_diagnostics.traces = vec![RankMixingDiagnostic {
            trace: DiagnosticTraceCoordinate::Eta {
                subject: "subject-1".into(),
                effect_index: 0,
                effect_name: "CL".into(),
            },
            rank_rhat: Some(1.01),
            rank_rhat_status: RankDiagnosticStatus::Available,
            folded_rhat: None,
            folded_rhat_status: RankDiagnosticStatus::ConstantDraws,
            max_rhat: None,
            max_rhat_status: RankDiagnosticStatus::ConstantDraws,
            bulk_ess: None,
            bulk_ess_status: RankDiagnosticStatus::NonPositiveTau,
            avg_ess_per_split_chain: None,
            tau: None,
            status: RankDiagnosticStatus::PartialAvailability,
        }];

        let json = serde_json::to_string(&diagnostics).unwrap();
        let decoded: MarkovSimulationVarianceDiagnostics = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.rank_diagnostics.lrv_per_chain.len(), 2);
        assert!(decoded.rank_diagnostics.lrv_per_chain[0].is_none());
        assert_eq!(
            decoded.rank_diagnostics.lrv_per_chain[1],
            Some(vec![vec![2.0]])
        );
        assert_eq!(
            decoded.rank_diagnostics.lrv_chain_statuses,
            vec![
                RankDiagnosticStatus::ScoreUnavailable,
                RankDiagnosticStatus::Available,
            ]
        );

        let mut rows = Vec::new();
        markov_variance_statistics(7, &decoded, &mut rows);
        let status_rows = rows
            .iter()
            .filter(|row| row.kind == "markov_rank_lrv_chain_status")
            .collect::<Vec<_>>();
        assert_eq!(status_rows.len(), 2);
        assert_eq!(status_rows[0].name, "diagnostic_chain_0");
        assert_eq!(status_rows[0].status.as_deref(), Some("score_unavailable"));
        assert_eq!(status_rows[1].name, "diagnostic_chain_1");
        assert_eq!(status_rows[1].status.as_deref(), Some("available"));
        let matrix_rows = rows
            .iter()
            .filter(|row| row.kind == "markov_rank_lrv_per_chain")
            .collect::<Vec<_>>();
        assert_eq!(matrix_rows.len(), 1);
        assert_eq!(matrix_rows[0].name, "diagnostic_chain_1");

        let statistic_status = |name: &str| {
            rows.iter()
                .find(|row| {
                    row.kind == "markov_rank_statistic_status" && row.name.starts_with(name)
                })
                .and_then(|row| row.status.as_deref())
        };
        assert_eq!(statistic_status("rank_rhat:"), Some("available"));
        assert_eq!(statistic_status("folded_rhat:"), Some("constant_draws"));
        assert_eq!(statistic_status("max_rhat:"), Some("constant_draws"));
        assert_eq!(statistic_status("bulk_ess:"), Some("non_positive_tau"));
        let value_rows = rows
            .iter()
            .filter(|row| row.kind == "markov_rank_rhat" && row.name.starts_with("rank_rhat:"))
            .collect::<Vec<_>>();
        assert_eq!(value_rows.len(), 1);
        assert_eq!(value_rows[0].value, Some(1.01));
        assert_eq!(value_rows[0].status.as_deref(), Some("available"));
        assert!(!rows
            .iter()
            .any(|row| { row.kind == "markov_rank_rhat" && row.name.starts_with("max_rhat:") }));
        assert!(!rows
            .iter()
            .any(|row| { row.kind == "markov_rank_ess" && row.name.starts_with("bulk_ess:") }));

        let csv_path = std::env::temp_dir().join(format!(
            "pmcore-failed-chain-{}-statistics.csv",
            std::process::id()
        ));
        write_csv(
            &csv_path,
            &rows,
            &[
                "cycle",
                "kind",
                "name",
                "row",
                "column",
                "output_index",
                "component",
                "value",
                "status",
            ],
        )
        .unwrap();
        let csv_rows = csv::Reader::from_path(&csv_path)
            .unwrap()
            .deserialize::<StatisticRow>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        std::fs::remove_file(&csv_path).unwrap();
        let csv_status_rows = csv_rows
            .iter()
            .filter(|row| row.kind == "markov_rank_lrv_chain_status")
            .collect::<Vec<_>>();
        assert_eq!(csv_status_rows.len(), 2);
        assert_eq!(csv_status_rows[0].name, "diagnostic_chain_0");
        assert_eq!(
            csv_status_rows[0].status.as_deref(),
            Some("score_unavailable")
        );
        assert_eq!(csv_status_rows[1].name, "diagnostic_chain_1");
        assert_eq!(csv_status_rows[1].status.as_deref(), Some("available"));
        let csv_matrix_rows = csv_rows
            .iter()
            .filter(|row| row.kind == "markov_rank_lrv_per_chain")
            .collect::<Vec<_>>();
        assert_eq!(csv_matrix_rows.len(), 1);
        assert_eq!(csv_matrix_rows[0].name, "diagnostic_chain_1");
        assert_eq!(csv_matrix_rows[0].value, Some(2.0));
    }

    #[test]
    fn every_markov_outcome_has_an_explicit_neutral_status_row() {
        let outcomes = vec![
            MarkovSimulationVarianceStatus::Disabled,
            MarkovSimulationVarianceStatus::AverageNotApplied,
            MarkovSimulationVarianceStatus::NoFreeCoordinates,
            MarkovSimulationVarianceStatus::InformationUnavailable("test".into()),
            MarkovSimulationVarianceStatus::InvalidConfiguration("test".into()),
            MarkovSimulationVarianceStatus::TraceMemoryAccountingOverflow,
            MarkovSimulationVarianceStatus::CoordinateMismatch,
            MarkovSimulationVarianceStatus::UnsupportedScore("test".into()),
            MarkovSimulationVarianceStatus::NonFinite,
            MarkovSimulationVarianceStatus::NonSymmetric,
            MarkovSimulationVarianceStatus::Indefinite,
            MarkovSimulationVarianceStatus::StuckChain { chain: 2 },
            MarkovSimulationVarianceStatus::ExactZeroNoLatentState,
            MarkovSimulationVarianceStatus::AssumptionsUnverified,
        ];
        for outcome in outcomes {
            let mut rows = Vec::new();
            push_markov_metadata(9, "markov_status", "aggregate", 0.0, &outcome, &mut rows);
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].kind, "markov_status");
            assert_eq!(rows[0].value, Some(0.0));
            assert_eq!(
                rows[0].status.as_deref(),
                Some(markov_variance_status_text(&outcome).as_str())
            );
        }
    }

    #[test]
    fn operational_convergence_schema_roundtrip_preserves_all_fields() {
        use crate::results::{
            OperationalConvergenceCheck, OperationalConvergenceCriterion,
            OperationalConvergenceCriterionStatus, OperationalConvergenceDiagnostics,
            OperationalConvergenceOutcome,
        };

        let diagnostic = OperationalConvergenceDiagnostics {
            checks: vec![OperationalConvergenceCheck {
                iteration: 42,
                averaged_iterations: 10,
                scheduled: true,
                mandatory_final: false,
                checkpoint_seed: Some(12345),
                z_quantile: Some(1.96),
                implied_minimum_ess: Some(100.0),
                candidate_free_coordinates: vec![0.5, -0.3],
                information: None,
                criteria: vec![
                    OperationalConvergenceCriterion {
                        name: "max_rhat".into(),
                        observed: Some(1.005),
                        threshold: 1.01,
                        status: OperationalConvergenceCriterionStatus::Satisfied,
                    },
                    OperationalConvergenceCriterion {
                        name: "min_bulk_ess".into(),
                        observed: Some(350.0),
                        threshold: 400.0,
                        status: OperationalConvergenceCriterionStatus::NotSatisfied,
                    },
                    OperationalConvergenceCriterion {
                        name: "fixed_width_ratio".into(),
                        observed: None,
                        threshold: 0.1,
                        status: OperationalConvergenceCriterionStatus::Unavailable(
                            "no diagnostic chains configured".into(),
                        ),
                    },
                ],
                outcome: OperationalConvergenceOutcome::Failed {
                    criteria: vec!["min_bulk_ess".into()],
                },
                markov: None,
            }],
            final_check_reused: false,
            used_for_termination: false,
            ..OperationalConvergenceDiagnostics::default()
        };

        let json = serde_json::to_string(&diagnostic).unwrap();
        let decoded: OperationalConvergenceDiagnostics = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.checks.len(), 1);
        let check = &decoded.checks[0];
        assert_eq!(check.iteration, 42);
        assert_eq!(check.averaged_iterations, 10);
        assert!(check.scheduled);
        assert!(!check.mandatory_final);
        assert_eq!(check.checkpoint_seed, Some(12345));
        assert_eq!(check.z_quantile, Some(1.96));
        assert_eq!(check.implied_minimum_ess, Some(100.0));
        assert_eq!(check.candidate_free_coordinates, vec![0.5, -0.3]);
        assert_eq!(check.criteria.len(), 3);

        // Criterion 0: satisfied
        assert_eq!(check.criteria[0].name, "max_rhat");
        assert_eq!(check.criteria[0].observed, Some(1.005));
        assert_eq!(check.criteria[0].threshold, 1.01);
        assert!(matches!(
            check.criteria[0].status,
            OperationalConvergenceCriterionStatus::Satisfied
        ));

        // Criterion 1: not satisfied
        assert_eq!(check.criteria[1].name, "min_bulk_ess");
        assert_eq!(check.criteria[1].observed, Some(350.0));
        assert_eq!(check.criteria[1].threshold, 400.0);
        assert!(matches!(
            check.criteria[1].status,
            OperationalConvergenceCriterionStatus::NotSatisfied
        ));

        // Criterion 2: unavailable
        assert_eq!(check.criteria[2].name, "fixed_width_ratio");
        assert_eq!(check.criteria[2].observed, None);
        assert_eq!(check.criteria[2].threshold, 0.1);
        assert!(matches!(
            &check.criteria[2].status,
            OperationalConvergenceCriterionStatus::Unavailable(reason) if reason == "no diagnostic chains configured"
        ));

        assert!(matches!(
            check.outcome,
            OperationalConvergenceOutcome::Failed { .. }
        ));
        assert!(!decoded.final_check_reused);
        assert!(!decoded.used_for_termination);
    }

    #[test]
    fn operational_convergence_stat_rows_omit_absent_numerics() {
        use crate::results::{
            OperationalConvergenceCheck, OperationalConvergenceCriterion,
            OperationalConvergenceCriterionStatus, OperationalConvergenceDiagnostics,
            OperationalConvergenceOutcome,
        };

        let diagnostic = OperationalConvergenceDiagnostics {
            checks: vec![OperationalConvergenceCheck {
                iteration: 5,
                averaged_iterations: 2,
                scheduled: false,
                mandatory_final: true,
                checkpoint_seed: Some(42),
                z_quantile: None,
                implied_minimum_ess: None,
                candidate_free_coordinates: vec![f64::NAN, 1.0],
                information: None,
                criteria: vec![OperationalConvergenceCriterion {
                    name: "max_rhat".into(),
                    observed: None,
                    threshold: 1.01,
                    status: OperationalConvergenceCriterionStatus::Unavailable(
                        "no frozen diagnostic".into(),
                    ),
                }],
                outcome: OperationalConvergenceOutcome::Ineligible {
                    reasons: vec!["no frozen diagnostic".into()],
                },
                markov: None,
            }],
            final_check_reused: true,
            used_for_termination: false,
            ..OperationalConvergenceDiagnostics::default()
        };

        let mut rows = Vec::new();
        operational_convergence_statistics(10, &diagnostic, &mut rows);

        // Filter rows for interest.
        let by_kind = |kind: &str| -> Vec<&StatisticRow> {
            rows.iter().filter(|row| row.kind == kind).collect()
        };

        // Checkpoint count.
        let checkpoints = by_kind("operational_convergence_checkpoints");
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].value, Some(1.0));

        // Flags.
        let flags = by_kind("operational_convergence_flag");
        let used_termination = flags.iter().find(|row| row.name == "used_for_termination");
        assert!(used_termination.is_some());
        assert_eq!(used_termination.unwrap().value, Some(0.0));
        let reused = flags.iter().find(|row| row.name == "final_check_reused");
        assert_eq!(reused.unwrap().value, Some(1.0));
        let sched = flags.iter().find(|row| row.name == "check_0:scheduled");
        assert_eq!(sched.unwrap().value, Some(0.0));
        let final_flag = flags
            .iter()
            .find(|row| row.name == "check_0:mandatory_final");
        assert_eq!(final_flag.unwrap().value, Some(1.0));

        // Check metadata.
        let checks = by_kind("operational_convergence_check");
        let iter_row = checks.iter().find(|row| row.name == "check_0:iteration");
        assert_eq!(iter_row.unwrap().value, Some(5.0));
        let seed_high = checks
            .iter()
            .find(|row| row.name == "check_0:checkpoint_seed_high_u32");
        let seed_low = checks
            .iter()
            .find(|row| row.name == "check_0:checkpoint_seed_low_u32");
        assert_eq!(seed_high.unwrap().value, Some(0.0));
        assert_eq!(seed_low.unwrap().value, Some(42.0));
        // z_quantile and implied were None: absent.
        assert!(checks
            .iter()
            .find(|row| row.name == "check_0:z_quantile")
            .is_none());
        assert!(checks
            .iter()
            .find(|row| row.name == "check_0:implied_minimum_ess")
            .is_none());

        // Free coordinates: only the finite one (1.0) appears.
        let coords = by_kind("operational_convergence_coordinate");
        assert_eq!(coords.len(), 1);
        assert_eq!(coords[0].name, "check_0:free_coordinate_1");
        assert_eq!(coords[0].value, Some(1.0));

        // Criterion status row.
        let crit_status = by_kind("operational_convergence_criterion_status");
        assert_eq!(crit_status.len(), 1);
        assert_eq!(crit_status[0].name, "check_0:crit_0:max_rhat:status");
        assert_eq!(crit_status[0].value, Some(-1.0)); // Unavailable

        // Criterion observed absent, threshold present.
        let crit_values = by_kind("operational_convergence_criterion");
        assert_eq!(crit_values.len(), 1);
        assert_eq!(crit_values[0].name, "check_0:crit_0:threshold");
        assert_eq!(crit_values[0].value, Some(1.01));

        // Outcome.
        let outcomes = by_kind("operational_convergence_outcome");
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].name, "check_0:outcome");
        assert_eq!(outcomes[0].value, Some(-1.0));
    }

    #[test]
    fn empty_operational_convergence_emits_only_metadata() {
        let diagnostic = OperationalConvergenceDiagnostics::default();
        let mut rows = Vec::new();
        operational_convergence_statistics(10, &diagnostic, &mut rows);

        // Only metadata rows: checkpoints count and two flags.
        let by_kind = |kind: &str| -> Vec<&StatisticRow> {
            rows.iter().filter(|row| row.kind == kind).collect()
        };
        assert_eq!(by_kind("operational_convergence_checkpoints").len(), 1);
        assert_eq!(by_kind("operational_convergence_flag").len(), 2);
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn unavailable_marginal_rows_and_statistics_omit_numerics() {
        use crate::estimation::parametric::marginal_likelihood::{
            MarginalLikelihoodConfig, MarginalLikelihoodFailureReason,
            MarginalLikelihoodSubjectFailure, SubjectMarginalLikelihoodDiagnostics,
        };

        let failure = MarginalLikelihoodFailureReason::ScoringFailure("posthoc failed".into());
        let diagnostics = MarginalLikelihoodDiagnostics {
            config: MarginalLikelihoodConfig::new(16, 7, 5, 1.5),
            status: MarginalLikelihoodStatus::Unavailable {
                failures: vec![MarginalLikelihoodSubjectFailure {
                    subject_id: "subject".into(),
                    reason: failure.clone(),
                }],
            },
            log_marginal_likelihood: None,
            n2ll: None,
            n2ll_mcse: None,
            subjects: vec![SubjectMarginalLikelihoodDiagnostics {
                subject_id: "subject".into(),
                method: MarginalLikelihoodMethod::StudentTImportanceSampling,
                proposal_scale_source: ProposalScaleSource::FinalRawOmegaBlocks,
                seed: Some(
                    crate::estimation::parametric::marginal_likelihood::marginal_likelihood_subject_seed(
                        7, 0,
                    ),
                ),
                dimension: 1,
                occasion_indices: vec![],
                mode: vec![],
                mode_converged: Some(false),
                samples: 16,
                log_marginal_likelihood: None,
                n2ll: None,
                effective_sample_size: None,
                effective_sample_fraction: None,
                zero_weight_count: 0,
                var_log: None,
                n2ll_mcse: None,
                failure: Some(failure),
            }],
        };
        let rows = marginal_likelihood_rows(Some(&diagnostics));
        assert!(rows.iter().all(|row| {
            row.log_marginal_likelihood.is_none()
                && row.n2ll.is_none()
                && row.n2ll_mcse.is_none()
                && row.effective_sample_size.is_none()
                && row.effective_sample_fraction.is_none()
        }));
        let mut statistics = Vec::new();
        marginal_likelihood_statistics(3, Some(&diagnostics), &mut statistics);
        assert!(statistics.iter().all(|row| row.value.is_none()));
    }

    #[test]
    fn covariance_boundary_warnings_have_stable_output_kinds() {
        let omega = warning_record(&ParametricWarning::OmegaBoundaryRejection {
            first_iteration: 12,
            longest_run: 7,
        });
        assert_eq!(omega.kind, "omega_boundary_rejection_run");
        assert_eq!(omega.first_cycle, 12);
        assert_eq!(omega.count, 7);

        let omega_iov = warning_record(&ParametricWarning::OmegaIovBoundaryRejection {
            first_iteration: 21,
            longest_run: 4,
        });
        assert_eq!(omega_iov.kind, "omega_iov_boundary_rejection_run");
        assert_eq!(omega_iov.first_cycle, 21);
        assert_eq!(omega_iov.count, 4);
    }

    #[test]
    fn operational_convergence_warnings_have_exact_policy_wording() {
        use crate::algorithms::parametric::OperationalConvergenceConfig;

        assert!(OperationalConvergenceDiagnostics::default()
            .warnings()
            .is_empty());
        let configured = OperationalConvergenceDiagnostics {
            config: Some(OperationalConvergenceConfig::literature_guided(
                1, 1, 0.05, 0.95, 0.1, 0.02,
            )),
            ..OperationalConvergenceDiagnostics::default()
        };
        assert!(configured.warnings()[0].contains("no checkpoint was evaluated"));

        let failed = OperationalConvergenceDiagnostics {
            final_status: Some(OperationalConvergenceOutcome::Failed {
                criteria: vec!["max_rhat".into()],
            }),
            ..configured.clone()
        };
        assert!(failed.warnings()[0].contains("evaluated but not satisfied"));

        let ineligible = OperationalConvergenceDiagnostics {
            final_status: Some(OperationalConvergenceOutcome::Ineligible {
                reasons: vec!["constant draws".into()],
            }),
            ..configured.clone()
        };
        assert!(ineligible.warnings()[0].contains("evaluated but were ineligible"));

        let passed = OperationalConvergenceDiagnostics {
            final_status: Some(OperationalConvergenceOutcome::Passed),
            used_for_termination: true,
            ..configured
        };
        let warning = &passed.warnings()[0];
        assert!(warning.contains("PMcore operational convergence criteria passed"));
        assert!(warning.contains("not proof of mathematical convergence"));
        assert!(warning.contains("independent doubled-budget fit"));
    }

    // ─── Schema-7 immutable declaration tests ────────────────────────

    #[test]
    fn source_covariance_initial_values_are_validated() {
        // Fixed diagonal entry must have initial == final.
        let covariance = ParametricSourceCovariance {
            dimension: 1,
            names: vec!["ke".to_string()],
            values: vec![vec![0.5]],
            structural_mask: vec![vec![true]],
            estimated_mask: vec![vec![false]],
            initial_values: vec![vec![0.5]],
        };
        assert!(validate_source_covariance(&covariance, &["ke"], "Omega").is_ok());

        // Mismatched fixed entry.
        let bad_covariance = ParametricSourceCovariance {
            initial_values: vec![vec![0.3]],
            ..covariance.clone()
        };
        let err = validate_source_covariance(&bad_covariance, &["ke"], "Omega").unwrap_err();
        assert!(
            err.to_string().contains("initial does not match final"),
            "unexpected error: {err}"
        );

        // Free entry: initial may differ from final, but must be finite.
        let free_covariance = ParametricSourceCovariance {
            dimension: 1,
            names: vec!["ke".to_string()],
            values: vec![vec![0.6]],
            structural_mask: vec![vec![true]],
            estimated_mask: vec![vec![true]],
            initial_values: vec![vec![0.4]],
        };
        assert!(validate_source_covariance(&free_covariance, &["ke"], "Omega").is_ok());

        // Non-finite initial value is rejected.
        let nonfinite_covariance = ParametricSourceCovariance {
            initial_values: vec![vec![f64::NAN]],
            ..free_covariance.clone()
        };
        let err = validate_source_covariance(&nonfinite_covariance, &["ke"], "Omega").unwrap_err();
        assert!(
            err.to_string().contains("must be finite"),
            "unexpected error: {err}"
        );

        // Wrong dimension.
        let wrong_dim = ParametricSourceCovariance {
            initial_values: vec![vec![0.4, 0.0], vec![0.0, 0.5]],
            ..free_covariance
        };
        let err = validate_source_covariance(&wrong_dim, &["ke"], "Omega").unwrap_err();
        assert!(
            err.to_string()
                .contains("do not match the declared dimension"),
            "unexpected error: {err}"
        );

        let structural_zero = ParametricSourceCovariance {
            dimension: 2,
            names: vec!["ke".to_string(), "v".to_string()],
            values: vec![vec![0.5, 0.0], vec![0.0, 0.6]],
            structural_mask: vec![vec![true, false], vec![false, true]],
            estimated_mask: vec![vec![true, false], vec![false, true]],
            initial_values: vec![vec![0.4, 0.01], vec![0.01, 0.7]],
        };
        let error =
            validate_source_covariance(&structural_zero, &["ke", "v"], "Omega").unwrap_err();
        assert!(error.to_string().contains("structural-zero"));

        let non_spd = ParametricSourceCovariance {
            initial_values: vec![vec![0.4, 0.8], vec![0.8, 0.7]],
            structural_mask: vec![vec![true, true], vec![true, true]],
            estimated_mask: vec![vec![true, true], vec![true, true]],
            ..structural_zero
        };
        let error = validate_source_covariance(&non_spd, &["ke", "v"], "Omega").unwrap_err();
        assert!(error.to_string().contains("strictly positive definite"));
    }

    #[test]
    fn source_covariance_two_by_two_initial_validation() {
        // Mixed fixed/free two-dimensional Omega.
        let covariance = ParametricSourceCovariance {
            dimension: 2,
            names: vec!["ke".to_string(), "v".to_string()],
            values: vec![vec![0.25, 0.05], vec![0.05, 0.30]],
            structural_mask: vec![vec![true, true], vec![true, true]],
            estimated_mask: vec![vec![true, true], vec![true, false]],
            initial_values: vec![vec![0.20, 0.03], vec![0.03, 0.30]],
        };
        // The covariance and ke variance are free; only the v variance is
        // fixed, so its immutable initial and final values are both 0.30.
        assert!(validate_source_covariance(&covariance, &["ke", "v"], "Omega").is_ok());
    }

    #[test]
    fn source_residual_initial_values_are_validated() {
        // Fixed component: initial == final.
        let residual = ParametricSourceResidual {
            output: "cp".to_string(),
            output_index: 0,
            family: "constant".to_string(),
            components: vec!["sigma".to_string()],
            values: vec![0.5],
            estimated_mask: vec![false],
            initial_values: vec![0.5],
            initial_estimated_mask: vec![false],
        };
        let metadata = ParametricSourceMetadata {
            parameters: vec![],
            random_effects: vec![],
            omega: ParametricSourceCovariance {
                dimension: 0,
                names: vec![],
                values: vec![],
                structural_mask: vec![],
                estimated_mask: vec![],
                initial_values: vec![],
            },
            iov_effects: vec![],
            omega_iov: None,
            residual_outputs: vec![residual.clone()],
            covariate_effects: vec![],
            subject_covariates: vec![],
            subject_design: vec![],
            subject_population_parameters: vec![],
        };
        assert!(validate_source_snapshot(&metadata).is_ok());

        // Mismatched fixed component: change initial but leave final same.
        let bad_residual = ParametricSourceResidual {
            initial_values: vec![0.3],
            ..residual.clone()
        };
        let bad_metadata = ParametricSourceMetadata {
            residual_outputs: vec![bad_residual],
            ..metadata.clone()
        };
        let err = validate_source_snapshot(&bad_metadata).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("initial"),
            "error should mention initial, got: {msg}"
        );
        assert!(
            msg.contains("final") || msg.contains("match"),
            "error should reject mismatch, got: {msg}"
        );

        // Initial estimated mask disagrees.
        let mask_residual = ParametricSourceResidual {
            estimated_mask: vec![false],
            initial_estimated_mask: vec![true],
            ..residual.clone()
        };
        let mask_metadata = ParametricSourceMetadata {
            residual_outputs: vec![mask_residual],
            ..metadata.clone()
        };
        let err = validate_source_snapshot(&mask_metadata).unwrap_err();
        assert!(
            err.to_string()
                .contains("initial and final estimated masks"),
            "unexpected error: {err}"
        );

        // Non-finite initial value.
        let nonfinite_residual = ParametricSourceResidual {
            initial_values: vec![f64::NAN],
            ..residual
        };
        let nf_metadata = ParametricSourceMetadata {
            residual_outputs: vec![nonfinite_residual],
            ..metadata
        };
        let err = validate_source_snapshot(&nf_metadata).unwrap_err();
        assert!(
            err.to_string()
                .contains("must match the component width and be finite"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn source_residual_free_component_initial_may_differ_from_final() {
        let residual = ParametricSourceResidual {
            output: "cp".to_string(),
            output_index: 0,
            family: "constant".to_string(),
            components: vec!["sigma".to_string()],
            values: vec![0.8],
            estimated_mask: vec![true],
            initial_values: vec![0.5],
            initial_estimated_mask: vec![true],
        };
        let metadata = ParametricSourceMetadata {
            parameters: vec![],
            random_effects: vec![],
            omega: ParametricSourceCovariance {
                dimension: 0,
                names: vec![],
                values: vec![],
                structural_mask: vec![],
                estimated_mask: vec![],
                initial_values: vec![],
            },
            iov_effects: vec![],
            omega_iov: None,
            residual_outputs: vec![residual],
            covariate_effects: vec![],
            subject_covariates: vec![],
            subject_design: vec![],
            subject_population_parameters: vec![],
        };
        assert!(validate_source_snapshot(&metadata).is_ok());
    }

    #[test]
    fn source_covariance_initial_symmetry_is_checked() {
        let asymmetric = ParametricSourceCovariance {
            dimension: 2,
            names: vec!["ke".to_string(), "v".to_string()],
            values: vec![vec![0.25, 0.05], vec![0.05, 0.30]],
            structural_mask: vec![vec![true, true], vec![true, true]],
            estimated_mask: vec![vec![true, true], vec![true, true]],
            initial_values: vec![vec![0.20, 0.03], vec![0.04, 0.30]],
        };
        let err = validate_source_covariance(&asymmetric, &["ke", "v"], "Omega").unwrap_err();
        assert!(
            err.to_string().contains("symmetric"),
            "unexpected error: {err}"
        );
    }

    /// Run a real SAEM fit and verify that the source snapshot preserves
    /// immutable initial declarations: fixed residual components have
    /// initial == final, free omega variances have finite initials that
    /// may differ from finals.
    #[test]
    fn real_fit_preserves_immutable_initial_declarations() {
        use crate::estimation::{EstimationProblem, Omega};
        use crate::model::Parameter;
        use pharmsol::prelude::*;

        let equation = analytical! {
            name: "immutable_fixture",
            params: [ke, v],
            states: [central],
            outputs: [cp],
            routes: [infusion(iv) -> central],
            structure: one_compartment,
            out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
        };

        let data = Data::new(vec![
            Subject::builder("s1")
                .infusion(0.0, 100.0, "iv", 0.5)
                .observation(1.0, 4.8, "cp")
                .build(),
            Subject::builder("s2")
                .infusion(0.0, 100.0, "iv", 0.5)
                .observation(1.0, 5.1, "cp")
                .build(),
        ]);

        let omega = Omega::new().variance("ke", 0.04).fixed_variance("v", 0.09);

        let result = EstimationProblem::parametric(equation, data)
            .parameter(Parameter::log("ke").with_initial(0.30))
            .parameter(Parameter::log("v").with_initial(20.0).fixed())
            .omega(omega)
            .error_model(
                "cp",
                crate::estimation::ParametricErrorModel::new(ResidualErrorModel::constant(0.5))
                    .fixed(),
            )
            .build()
            .unwrap()
            .fit_with(
                SaemConfig::default()
                    .seed(42)
                    .burn_in(20)
                    .k1_iterations(30)
                    .k2_iterations(30)
                    .n_chains(2),
            )
            .unwrap();

        // Extract source metadata.
        let tables = result.tables(0.1, 1.0).unwrap();
        let record = result.record(tables).unwrap();
        let source = &record.source_metadata;

        assert_eq!(source.parameters[0].initial, 0.30);
        assert_eq!(source.parameters[1].initial, 20.0);
        assert!(!source.parameters[1].estimated);
        assert!(equal_with_roundoff(
            source.parameters[1].initial,
            source.parameters[1].estimate
        ));

        // Omega: ke-ke is free, v-v is fixed.
        assert_eq!(source.omega.dimension, 2);
        assert!(source.omega.estimated_mask[0][0]);
        assert!(!source.omega.estimated_mask[1][1]);
        // Fixed entry must have initial == final.
        assert!(
            (source.omega.values[1][1] - source.omega.initial_values[1][1]).abs()
                <= 64.0
                    * f64::EPSILON
                    * source.omega.initial_values[1][1]
                        .abs()
                        .max(source.omega.values[1][1].abs())
                        .max(1.0),
            "fixed v variance: initial {} != final {}",
            source.omega.initial_values[1][1],
            source.omega.values[1][1],
        );
        // Free entry: initial must be finite.
        assert!(source.omega.initial_values[0][0].is_finite());
        assert!(source.omega.initial_values[0][0] > 0.0);

        // Residual: sigma is fixed, must have initial == final.
        let residual = &source.residual_outputs[0];
        assert!(!residual.estimated_mask[0]);
        assert!(
            (residual.values[0] - residual.initial_values[0]).abs()
                <= 64.0
                    * f64::EPSILON
                    * residual.initial_values[0]
                        .abs()
                        .max(residual.values[0].abs())
                        .max(1.0),
            "fixed sigma: initial {} != final {}",
            residual.initial_values[0],
            residual.values[0],
        );
    }

    /// Tamper a persisted JSON record: change a fixed residual value.
    /// The reader must reject it.
    #[test]
    fn persisted_fixed_residual_tampering_is_rejected() {
        use crate::estimation::EstimationProblem;
        use crate::model::Parameter;
        use pharmsol::prelude::*;

        let equation = analytical! {
            name: "tamper_fixture",
            params: [ke, v],
            states: [central],
            outputs: [cp],
            routes: [infusion(iv) -> central],
            structure: one_compartment,
            out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
        };

        let data = Data::new(vec![
            Subject::builder("s1")
                .infusion(0.0, 100.0, "iv", 0.5)
                .observation(1.0, 4.8, "cp")
                .build(),
            Subject::builder("s2")
                .infusion(0.0, 100.0, "iv", 0.5)
                .observation(1.0, 5.1, "cp")
                .build(),
        ]);

        let result = EstimationProblem::parametric(equation, data)
            .parameter(Parameter::log("ke").with_initial(0.30).fixed())
            .parameter(Parameter::log("v").with_initial(20.0).fixed())
            .error_model(
                "cp",
                crate::estimation::ParametricErrorModel::new(ResidualErrorModel::constant(0.5))
                    .fixed(),
            )
            .build()
            .unwrap()
            .fit_with(
                SaemConfig::default()
                    .seed(42)
                    .burn_in(10)
                    .k1_iterations(10)
                    .k2_iterations(10)
                    .n_chains(2),
            )
            .unwrap();

        let tables = result.tables(0.1, 1.0).unwrap();
        let record = result.record(tables).unwrap();
        let mut json = serde_json::to_value(&record).expect("valid record should serialize");

        // Coordinate the mutable source/table/statistic snapshots while
        // leaving the immutable constant-error declaration unchanged.
        let original_final = json["source_metadata"]["residual_outputs"][0]["values"][0]
            .as_f64()
            .unwrap();
        let changed = original_final + 0.1;
        json["source_metadata"]["residual_outputs"][0]["values"][0] = serde_json::json!(changed);
        json["tables"]["residual_error"][0]["estimate"] = serde_json::json!(changed);
        for row in json["tables"]["statistics"].as_array_mut().unwrap() {
            if row["kind"] == "residual" && row["name"] == "cp" && row["component"] == "sigma" {
                row["value"] = serde_json::json!(changed);
            }
        }

        let tampered: Result<ParametricResultRecord, _> = serde_json::from_value(json.clone());
        assert!(tampered.is_ok(), "deserialization should succeed");
        let err = ParametricResultRecord::read_json_from_value(json).unwrap_err();
        assert!(
            err.to_string().contains("initial")
                && err.to_string().contains("final")
                && err.to_string().contains("sigma"),
            "unexpected error: {err}"
        );
    }

    /// Tamper a persisted JSON record: change a fixed Omega entry.
    /// The reader must reject it.
    #[test]
    fn persisted_fixed_omega_tampering_is_rejected() {
        use crate::estimation::{EstimationProblem, Omega};
        use crate::model::Parameter;
        use pharmsol::prelude::*;

        let equation = analytical! {
            name: "tamper_omega_fixture",
            params: [ke, v],
            states: [central],
            outputs: [cp],
            routes: [infusion(iv) -> central],
            structure: one_compartment,
            out: |x, _p, _t, _cov, y| { y[cp] = x[central] / v; },
        };

        let data = Data::new(vec![
            Subject::builder("s1")
                .infusion(0.0, 100.0, "iv", 0.5)
                .observation(1.0, 4.8, "cp")
                .build(),
            Subject::builder("s2")
                .infusion(0.0, 100.0, "iv", 0.5)
                .observation(1.0, 5.1, "cp")
                .build(),
        ]);

        let omega = Omega::new().variance("ke", 0.04).fixed_variance("v", 0.09);

        let result = EstimationProblem::parametric(equation, data)
            .parameter(Parameter::log("ke").with_initial(0.30))
            .parameter(Parameter::log("v").with_initial(20.0).fixed())
            .omega(omega)
            .error_model(
                "cp",
                crate::estimation::ParametricErrorModel::new(ResidualErrorModel::constant(0.5))
                    .fixed(),
            )
            .build()
            .unwrap()
            .fit_with(
                SaemConfig::default()
                    .seed(42)
                    .burn_in(10)
                    .k1_iterations(10)
                    .k2_iterations(10)
                    .n_chains(2),
            )
            .unwrap();

        let tables = result.tables(0.1, 1.0).unwrap();
        let record = result.record(tables).unwrap();
        let mut json = serde_json::to_value(&record).expect("valid record should serialize");

        // Tamper: change a fixed Omega entry final value.
        let omega = &mut json["source_metadata"]["omega"];
        let original = omega["values"][1][1].as_f64().unwrap();
        omega["values"][1][1] =
            serde_json::Value::Number(serde_json::Number::from_f64(original + 0.05).unwrap());

        let tampered: Result<ParametricResultRecord, _> = serde_json::from_value(json.clone());
        assert!(tampered.is_ok(), "deserialization should succeed");
        let err = ParametricResultRecord::read_json_from_value(json).unwrap_err();
        assert!(
            err.to_string().contains("initial") && err.to_string().contains("final"),
            "unexpected error: {err}"
        );
    }
}

/// Helper: read a ParametricResultRecord from a JSON value, performing
/// all the same validation as read_json without touching a file.
#[allow(clippy::items_after_test_module)]
#[cfg(test)]
impl ParametricResultRecord {
    fn read_json_from_value(raw: serde_json::Value) -> Result<Self> {
        let object = raw
            .as_object()
            .context("parametric result JSON must be an object")?;
        for required in [
            "marginal_likelihood",
            "information_criteria",
            "subject_count",
            "population_uncertainty",
            "conditional_modes",
            "shrinkage",
        ] {
            if !object.contains_key(required) {
                bail!("schema-9 parametric result requires the {required} field");
            }
        }
        object
            .get("source_metadata")
            .and_then(serde_json::Value::as_object)
            .context("schema-9 parametric result requires an object source_metadata field")?;
        let config = object
            .get("config")
            .and_then(serde_json::Value::as_object)
            .context("schema-9 parametric result requires an object config field")?;
        if !config.contains_key("marginal_likelihood") {
            bail!("schema-9 parametric result requires config.marginal_likelihood");
        }
        let record: Self =
            serde_json::from_value(raw).context("failed to parse parametric result")?;
        if record.schema_version != PARAMETRIC_RESULT_SCHEMA_VERSION {
            bail!(
                "unsupported parametric result schema version {}",
                record.schema_version
            );
        }
        if record.fit_family != "parametric" || record.algorithm != "saem" {
            bail!("JSON record is not a parametric SAEM result");
        }
        if record.objective_kind != "conditional_n2ll" {
            bail!(
                "unsupported parametric objective kind '{}'",
                record.objective_kind
            );
        }
        record
            .config
            .validate()
            .context("schema-9 parametric result contains invalid retained SAEM configuration")?;
        validate_persisted_marginal_likelihood(&record)?;
        validate_persisted_information_criteria(&record)?;
        validate_persisted_n6(&record)?;
        Ok(record)
    }
}
