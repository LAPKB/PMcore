use std::collections::BTreeMap;

use pharmsol::Equation;
use serde::{Deserialize, Serialize};

use crate::estimation::nonparametric::NonparametricWorkspace;

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
