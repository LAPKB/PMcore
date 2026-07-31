use serde::{Deserialize, Serialize};

use crate::estimation::parametric::{ConditionalCurvatureDiagnostics, ShrinkageDiagnostics};
use crate::estimation::MarginalLikelihoodStatus;
use crate::results::{InformationCriteriaDiagnostics, PopulationUncertaintyDiagnostics};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FitSummary {
    pub objective_function: f64,
    pub converged: bool,
    pub iterations: usize,
    pub subject_count: usize,
    pub observation_count: usize,
    pub parameter_count: usize,
    pub marginal_log_likelihood: Option<f64>,
    pub marginal_n2ll: Option<f64>,
    pub marginal_n2ll_mcse: Option<f64>,
    pub marginal_likelihood_status: Option<MarginalLikelihoodStatus>,
    pub information_criteria: Option<InformationCriteriaDiagnostics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PopulationSummary {
    pub parameters: Vec<ParameterSummary>,
    pub information_criteria: Option<InformationCriteriaDiagnostics>,
    /// Parametric observed-information covariance and conditioning diagnostics.
    pub population_uncertainty: Option<PopulationUncertaintyDiagnostics>,
    /// Source-explicit eta/kappa posterior-mean and MAP shrinkage.
    pub shrinkage: Option<ShrinkageDiagnostics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterSummary {
    pub name: String,
    pub estimate: f64,
    pub mean: Option<f64>,
    pub median: Option<f64>,
    pub sd: Option<f64>,
    pub cv_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndividualSummary {
    pub id: String,
    pub parameter_names: Vec<String>,
    pub estimates: Vec<f64>,
    pub standard_errors: Option<Vec<f64>>,
    /// Joint eta/kappa curvature in exact flattened subject/occasion order.
    pub conditional_uncertainty: Option<ConditionalCurvatureDiagnostics>,
}
