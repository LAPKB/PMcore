use anyhow::Result;
use pharmsol::Equation;

use crate::algorithms::{
    run_nonparametric_algorithm, run_nonparametric_algorithm_with_progress,
    NonparametricAlgorithmInput,
};
use crate::api::{EstimationMethod, NonparametricCycleProgress};
use crate::compile::CompiledProblem;
use crate::estimation::nonparametric::workspace::NonparametricWorkspace;
use crate::results::FitResult;

#[derive(Debug, Default, Clone, Copy)]
pub struct NonparametricEngine;

impl NonparametricEngine {
    pub fn fit<E: Equation + Clone + Send + 'static>(
        problem: CompiledProblem<E>,
    ) -> Result<NonparametricWorkspace<E>> {
        let input = input_from_compiled_problem(problem)?;
        run_nonparametric_algorithm(input)
    }

    pub fn fit_with_progress<E, F>(
        problem: CompiledProblem<E>,
        mut on_progress: F,
    ) -> Result<NonparametricWorkspace<E>>
    where
        E: Equation + Clone + Send + 'static,
        F: FnMut(NonparametricCycleProgress),
    {
        let input = input_from_compiled_problem(problem)?;
        run_nonparametric_algorithm_with_progress(
            input,
            |cycle, objective, objective_delta, elapsed_ms, status| {
                on_progress(NonparametricCycleProgress {
                    cycle,
                    objective,
                    objective_delta,
                    elapsed_ms,
                    status,
                });
            },
        )
    }
}

fn input_from_compiled_problem<E: Equation + Clone + Send + 'static>(
    problem: CompiledProblem<E>,
) -> Result<NonparametricAlgorithmInput<E>> {
    let EstimationMethod::Nonparametric(method) = problem.method();
    let output = problem.output_plan().clone();
    let runtime = problem.runtime_options().clone();
    let (model, data) = problem.into_parts();
    Ok(NonparametricAlgorithmInput::new(
        method, model, data, output, runtime,
    ))
}

pub fn fit<E: Equation + Clone + Send + 'static>(
    problem: CompiledProblem<E>,
) -> Result<FitResult<E>> {
    let workspace = NonparametricEngine::fit(problem)?;
    Ok(workspace.into_fit_result())
}

pub fn fit_with_progress<E, F>(problem: CompiledProblem<E>, on_progress: F) -> Result<FitResult<E>>
where
    E: Equation + Clone + Send + 'static,
    F: FnMut(NonparametricCycleProgress),
{
    let workspace = NonparametricEngine::fit_with_progress(problem, on_progress)?;
    Ok(workspace.into_fit_result())
}
