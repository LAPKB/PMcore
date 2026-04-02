use anyhow::Result;
use pharmsol::equation::Equation;

use crate::api::estimation_problem::EstimationProblem;
use crate::estimation::nonparametric;
use crate::results::FitResult;

pub fn fit<E: Equation + Clone + Send + 'static>(
    problem: EstimationProblem<E>,
) -> Result<FitResult<E>> {
    if problem.runtime.logging.initialize {
        problem.initialize_logs()?;
    }

    let compiled = problem.compile()?;
    nonparametric::fit(compiled)
}
