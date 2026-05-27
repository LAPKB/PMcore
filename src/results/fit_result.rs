use pharmsol::Equation;

use crate::estimation::nonparametric;
use crate::estimation::nonparametric::NonParametricResult;

use crate::results::{FitSummary, IndividualSummary, PopulationSummary};

#[derive(Debug)]
pub struct ParametricResult<E: Equation> {
    // Placeholder for future implementation
    _phantom: std::marker::PhantomData<E>,
}

#[derive(Debug)]
pub enum FitResult<E: Equation> {
    Nonparametric(NonParametricResult<E>),
    Parametric(ParametricResult<E>),
}

impl<E: Equation> FitResult<E> {
    pub fn objf(&self) -> f64 {
        match self {
            Self::Nonparametric(result) => result.objf(),
            _ => unimplemented!(), // Placeholder for ParametricResult
        }
    }

    pub fn converged(&self) -> bool {
        match self {
            Self::Nonparametric(result) => result.converged(),
            _ => unimplemented!(), // Placeholder for ParametricResult
        }
    }

    pub fn summary(&self) -> FitSummary {
        match self {
            Self::Nonparametric(result) => nonparametric::fit_summary(result),
            _ => unimplemented!(), // Placeholder for ParametricResult
        }
    }

    pub fn population_summary(&self) -> PopulationSummary {
        match self {
            Self::Nonparametric(result) => nonparametric::population_summary(result),
            _ => unimplemented!(), // Placeholder for ParametricResult
        }
    }

    pub fn individual_summaries(&self) -> Vec<IndividualSummary> {
        match self {
            Self::Nonparametric(result) => nonparametric::individual_summaries(result),
            _ => unimplemented!(), // Placeholder for ParametricResult
        }
    }

    pub fn as_nonparametric(&self) -> Option<&NonParametricResult<E>> {
        match self {
            Self::Nonparametric(result) => Some(result),
            _ => unimplemented!(), // Placeholder for ParametricResult
        }
    }
}
