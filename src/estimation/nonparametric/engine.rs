use anyhow::Result;
use pharmsol::Equation;

use crate::algorithms::{
    run_nonparametric_algorithm, run_nonparametric_algorithm_with_progress,
    run_nonparametric_algorithm_with_progress_and_control, NonparametricAlgorithmInput,
};
use crate::api::{FitControlSource, FitProgress};
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
        on_progress: F,
    ) -> Result<NonparametricWorkspace<E>>
    where
        E: Equation + Clone + Send + 'static,
        F: FnMut(FitProgress),
    {
        let input = input_from_compiled_problem(problem)?;
        run_nonparametric_algorithm_with_progress(input, on_progress)
    }

    pub fn fit_with_progress_and_control<E, F, C>(
        problem: CompiledProblem<E>,
        on_progress: F,
        next_control: C,
    ) -> Result<NonparametricWorkspace<E>>
    where
        E: Equation + Clone + Send + 'static,
        F: FnMut(FitProgress),
        C: FitControlSource,
    {
        let input = input_from_compiled_problem(problem)?;
        run_nonparametric_algorithm_with_progress_and_control(input, on_progress, next_control)
    }
}

fn input_from_compiled_problem<E: Equation + Clone + Send + 'static>(
    problem: CompiledProblem<E>,
) -> Result<NonparametricAlgorithmInput<E>> {
    let method = problem.method();
    let error_models = problem.error_models().models().clone();
    let output = problem.output_plan().clone();
    let runtime = problem.runtime_options().clone();
    let (model, data) = problem.into_parts();
    Ok(NonparametricAlgorithmInput::new(
        method,
        model,
        data,
        error_models,
        output,
        runtime,
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
    F: FnMut(FitProgress),
{
    let workspace = NonparametricEngine::fit_with_progress(problem, on_progress)?;
    Ok(workspace.into_fit_result())
}

pub fn fit_with_progress_and_control<E, F, C>(
    problem: CompiledProblem<E>,
    on_progress: F,
    next_control: C,
) -> Result<FitResult<E>>
where
    E: Equation + Clone + Send + 'static,
    F: FnMut(FitProgress),
    C: FitControlSource,
{
    let workspace = NonparametricEngine::fit_with_progress_and_control(
        problem,
        on_progress,
        next_control,
    )?;
    Ok(workspace.into_fit_result())
}
