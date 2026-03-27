use pharmsol::Equation;
use serde::{Deserialize, Serialize};

use crate::estimation::nonparametric::NonparametricWorkspace;
use crate::estimation::parametric::ParametricWorkspace;
use crate::results::{nonparametric_artifacts, parametric_artifacts};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PredictionsBundle {
    pub available: bool,
    pub row_count: Option<usize>,
    pub source: Option<String>,
    pub artifact: Option<String>,
}

pub(crate) fn nonparametric_predictions<E: Equation>(
    result: &NonparametricWorkspace<E>,
) -> PredictionsBundle {
    let artifact = nonparametric_artifacts(result)
        .files
        .into_iter()
        .find(|file| file == "predictions.csv");

    if let Some(predictions) = result.predictions() {
        return PredictionsBundle {
            available: true,
            row_count: Some(predictions.predictions().len()),
            source: Some("in_memory".to_string()),
            artifact,
        };
    }

    PredictionsBundle {
        available: artifact.is_some(),
        row_count: None,
        source: artifact.as_ref().map(|_| "artifact".to_string()),
        artifact,
    }
}

pub(crate) fn parametric_predictions<E: Equation>(
    result: &ParametricWorkspace<E>,
) -> PredictionsBundle {
    let artifact = parametric_artifacts(result)
        .files
        .into_iter()
        .find(|file| file == "predictions.csv");

    if let Some(predictions) = result.predictions() {
        return PredictionsBundle {
            available: true,
            row_count: Some(predictions.predictions().len()),
            source: Some("in_memory".to_string()),
            artifact,
        };
    }

    PredictionsBundle {
        available: artifact.is_some(),
        row_count: None,
        source: artifact.as_ref().map(|_| "artifact".to_string()),
        artifact,
    }
}
