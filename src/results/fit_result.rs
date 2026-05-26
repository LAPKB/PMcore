use pharmsol::Equation;

use crate::estimation::nonparametric;
use crate::estimation::nonparametric::NonparametricWorkspace;
use crate::results::{FitSummary, IndividualSummary, PopulationSummary};

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

    pub fn as_nonparametric(&self) -> Option<&NonparametricWorkspace<E>> {
        match self {
            Self::Nonparametric(result) => Some(result),
        }
    }
}
