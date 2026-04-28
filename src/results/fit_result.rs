use anyhow::Result;
use pharmsol::Equation;

use crate::estimation::nonparametric;
use crate::estimation::nonparametric::NonparametricWorkspace;
use crate::results::{
    nonparametric_artifacts, nonparametric_diagnostics, nonparametric_predictions, ArtifactIndex,
    DiagnosticsBundle, FitSummary, IndividualSummary, PopulationSummary, PredictionsBundle,
};

#[derive(Debug)]
pub enum FitResult<E: Equation> {
    Nonparametric(NonparametricWorkspace<E>),
}

impl<E: Equation> FitResult<E> {
    pub fn objf(&self) -> f64 {
        match self {
            Self::Nonparametric(result) => result.objf(),
        }
    }

    pub fn converged(&self) -> bool {
        match self {
            Self::Nonparametric(result) => result.converged(),
        }
    }

    pub fn write_outputs(&mut self) -> Result<()> {
        crate::output::write_result(self)
    }

    pub fn summary(&self) -> FitSummary {
        match self {
            Self::Nonparametric(result) => nonparametric::fit_summary(result),
        }
    }

    pub fn population_summary(&self) -> PopulationSummary {
        match self {
            Self::Nonparametric(result) => nonparametric::population_summary(result),
        }
    }

    pub fn individual_summaries(&self) -> Vec<IndividualSummary> {
        match self {
            Self::Nonparametric(result) => nonparametric::individual_summaries(result),
        }
    }

    pub fn diagnostics(&self) -> DiagnosticsBundle {
        match self {
            Self::Nonparametric(result) => nonparametric_diagnostics(result),
        }
    }

    pub fn predictions(&self) -> PredictionsBundle {
        match self {
            Self::Nonparametric(result) => nonparametric_predictions(result),
        }
    }

    pub fn artifacts(&self) -> ArtifactIndex {
        match self {
            Self::Nonparametric(result) => nonparametric_artifacts(result),
        }
    }

    pub fn as_nonparametric(&self) -> Option<&NonparametricWorkspace<E>> {
        match self {
            Self::Nonparametric(result) => Some(result),
        }
    }
}
