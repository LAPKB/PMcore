use anyhow::Result;
use pharmsol::Equation;

use crate::algorithms::{run_nonparametric_algorithm, NonparametricAlgorithmInput};
use crate::api::EstimationMethod;
use crate::compile::CompiledProblem;
use crate::estimation::nonparametric::workspace::NonparametricWorkspace;
use crate::results::FitResult;

#[derive(Debug, Default, Clone, Copy)]
pub struct NonparametricEngine;

impl NonparametricEngine {
    pub fn fit<E: Equation + Clone + Send + 'static>(
        problem: CompiledProblem<E>,
    ) -> Result<NonparametricWorkspace<E>> {
        let EstimationMethod::Nonparametric(method) = problem.method();
        let output = problem.output_plan().clone();
        let runtime = problem.runtime_options().clone();
        let (model, data) = problem.into_parts();
        let input = NonparametricAlgorithmInput::new(method, model, data, output, runtime);
        run_nonparametric_algorithm(input)
    }
}

pub fn fit<E: Equation + Clone + Send + 'static>(
    problem: CompiledProblem<E>,
) -> Result<FitResult<E>> {
    let workspace = NonparametricEngine::fit(problem)?;
    Ok(workspace.into_fit_result())
}
