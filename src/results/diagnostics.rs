use std::collections::BTreeMap;

use pharmsol::Equation;
use serde::{Deserialize, Serialize};

use crate::estimation::nonparametric::NonparametricWorkspace;
use crate::estimation::parametric::ParametricWorkspace;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiagnosticsBundle {
    pub warnings: Vec<String>,
    pub deferred_features: Vec<String>,
    pub convergence_notes: Vec<String>,
    pub estimator_metadata: BTreeMap<String, String>,
}

pub(crate) fn nonparametric_diagnostics<E: Equation>(
    result: &NonparametricWorkspace<E>,
) -> DiagnosticsBundle {
    let mut convergence_notes = Vec::new();
    if result.converged() {
        convergence_notes.push("Estimator reported convergence.".to_string());
    } else {
        convergence_notes.push("Estimator stopped without convergence.".to_string());
    }

    let status = result
        .cycle_log()
        .cycles()
        .last()
        .map(|cycle| format!("{:?}", cycle.status()))
        .unwrap_or_else(|| "Continue".to_string());

    let mut estimator_metadata = BTreeMap::new();
    estimator_metadata.insert("algorithm".to_string(), format!("{:?}", result.algorithm()));
    estimator_metadata.insert("status".to_string(), status);
    estimator_metadata.insert(
        "outputs_requested".to_string(),
        result.should_write_outputs().to_string(),
    );
    estimator_metadata.insert(
        "support_point_count".to_string(),
        result.get_theta().nspp().to_string(),
    );
    estimator_metadata.insert(
        "prediction_cache".to_string(),
        if result.predictions().is_some() {
            "available".to_string()
        } else {
            "not_materialized".to_string()
        },
    );

    DiagnosticsBundle {
        warnings: Vec::new(),
        deferred_features: Vec::new(),
        convergence_notes,
        estimator_metadata,
    }
}

pub(crate) fn parametric_diagnostics<E: Equation>(
    result: &ParametricWorkspace<E>,
) -> DiagnosticsBundle {
    let mut warnings = Vec::new();
    let mut deferred_features = Vec::new();
    let mut convergence_notes = Vec::new();
    let mut estimator_metadata = BTreeMap::new();

    if result.converged() {
        convergence_notes.push("Estimator reported convergence.".to_string());
    } else {
        warnings.push("Estimator stopped without convergence.".to_string());
        convergence_notes.push(format!("Final status: {:?}", result.status()));
    }

    if result.state().variability.occasion.is_some() {
        deferred_features.push("occasion_inference".to_string());
        warnings.push(
            "Occasion variability is represented in the compiled and fitted state, but occasion-level inference remains deferred; occasion_kappa entries are structural placeholders.".to_string(),
        );
        estimator_metadata.insert(
            "occasion_inference".to_string(),
            "deferred".to_string(),
        );
    } else {
        estimator_metadata.insert(
            "occasion_inference".to_string(),
            "not_requested".to_string(),
        );
    }

    estimator_metadata.insert("algorithm".to_string(), format!("{:?}", result.algorithm()));
    estimator_metadata.insert("status".to_string(), format!("{:?}", result.status()));
    estimator_metadata.insert(
        "outputs_requested".to_string(),
        result.should_write_outputs().to_string(),
    );
    estimator_metadata.insert(
        "iteration_log_entries".to_string(),
        result.iteration_log().len().to_string(),
    );
    estimator_metadata.insert(
        "prediction_cache".to_string(),
        if result.predictions().is_some() {
            "available".to_string()
        } else {
            "not_materialized".to_string()
        },
    );
    estimator_metadata.insert(
        "residual_error_model".to_string(),
        if result.sigma().model_type.is_empty() {
            "none".to_string()
        } else {
            result.sigma().model_type.clone()
        },
    );
    estimator_metadata.insert(
        "residual_error_output".to_string(),
        if !result.should_write_outputs() {
            "disabled".to_string()
        } else if result.sigma().model_type.is_empty() {
            "not_available".to_string()
        } else {
            "expected".to_string()
        },
    );
    estimator_metadata.insert(
        "uncertainty_method".to_string(),
        result
            .uncertainty()
            .fim_method()
            .map(|method| format!("{:?}", method))
            .unwrap_or_else(|| "none".to_string()),
    );
    estimator_metadata.insert(
        "uncertainty_output".to_string(),
        if !result.should_write_outputs() {
            "disabled".to_string()
        } else if result.uncertainty().has_fim() || result.uncertainty().has_standard_errors() {
            "expected".to_string()
        } else {
            "not_available".to_string()
        },
    );
    estimator_metadata.insert(
        "likelihood_best_objf".to_string(),
        result.best_objf().to_string(),
    );
    estimator_metadata.insert(
        "objective_source".to_string(),
        if result.likelihoods().ll_gaussian_quadrature.is_some() {
            "gaussian_quadrature".to_string()
        } else if result.likelihoods().ll_importance_sampling.is_some() {
            "importance_sampling".to_string()
        } else if result.likelihoods().ll_linearization.is_some() {
            "linearization".to_string()
        } else {
            "algorithm_state".to_string()
        },
    );

    DiagnosticsBundle {
        warnings,
        deferred_features,
        convergence_notes,
        estimator_metadata,
    }
}
