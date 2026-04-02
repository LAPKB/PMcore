use anyhow::Result;
use pharmsol::equation::Equation;

use crate::api::estimation_problem::{EstimationMethod, EstimationProblem};
use crate::estimation::{nonparametric, parametric};
use crate::results::FitResult;

pub fn fit<E: Equation + Clone + Send + 'static>(
    problem: EstimationProblem<E>,
) -> Result<FitResult<E>> {
    if problem.runtime.logging.initialize {
        problem.initialize_logs()?;
    }

    let method = problem.method;
    let compiled = problem.compile()?;

    match method {
        EstimationMethod::Nonparametric(_) => nonparametric::fit(compiled),
        EstimationMethod::Parametric(_) => parametric::fit(compiled),
    }
}
