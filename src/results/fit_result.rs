use pharmsol::Equation;

use crate::estimation::nonparametric::NonParametricResult;
use crate::results::{FitSummary, IndividualSummary, PopulationSummary};

/// A shared trait for the output of any estimation algorithm.
pub trait FitResult {
    /// The objective function minimized by the algorithm, `-2 × log-likelihood`
    /// for likelihood-based algorithms (lower is better).
    fn n2ll(&self) -> f64;
    /// Whether the fit stopped because it converged, rather than being cut short.
    fn converged(&self) -> bool;
    fn summary(&self) -> FitSummary;
    fn population_summary(&self) -> PopulationSummary;
    fn individual_summaries(&self) -> Vec<IndividualSummary>;
}

// TODO: Implement ParametricResult once parametric fitting is available.
#[derive(Debug)]
#[allow(unused)]
pub struct ParametricResult<E: Equation> {
    _phantom: std::marker::PhantomData<E>,
}

impl<E: Equation> FitResult for ParametricResult<E> {
    fn n2ll(&self) -> f64 {
        unimplemented!("Parametric result not yet implemented")
    }
    fn converged(&self) -> bool {
        unimplemented!()
    }
    fn summary(&self) -> FitSummary {
        unimplemented!()
    }
    fn population_summary(&self) -> PopulationSummary {
        unimplemented!()
    }
    fn individual_summaries(&self) -> Vec<IndividualSummary> {
        unimplemented!()
    }
}

use crate::estimation::nonparametric;

impl<E: Equation> FitResult for NonParametricResult<E> {
    fn n2ll(&self) -> f64 {
        NonParametricResult::n2ll(self)
    }

    fn converged(&self) -> bool {
        NonParametricResult::converged(self)
    }

    fn summary(&self) -> FitSummary {
        nonparametric::fit_summary(self)
    }

    fn population_summary(&self) -> PopulationSummary {
        nonparametric::population_summary(self)
    }

    fn individual_summaries(&self) -> Vec<IndividualSummary> {
        nonparametric::individual_summaries(self)
    }
}
