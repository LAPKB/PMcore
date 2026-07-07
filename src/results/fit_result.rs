use pharmsol::Equation;

use crate::estimation::nonparametric::NonParametricResult;
use crate::results::{FitSummary, IndividualSummary, PopulationSummary};

/// A shared trait for the output of any estimation algorithm.
pub trait FitResult {
    fn objf(&self) -> f64;
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
    fn objf(&self) -> f64 {
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
    fn objf(&self) -> f64 {
        self.objf() // Assuming the struct has this native method
    }

    fn converged(&self) -> bool {
        self.converged() // Assuming the struct has this native method
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
