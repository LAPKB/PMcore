use anyhow::Result;
use pharmsol::Equation;

use crate::algorithms::parametric::{run_parametric_algorithm, ParametricAlgorithmInput};
use crate::api::EstimationMethod;
use crate::compile::CompiledProblem;
use crate::estimation::parametric::compiler::compile_model_state;
use crate::estimation::parametric::workspace::ParametricWorkspace;
use crate::results::FitResult;

#[derive(Debug, Default, Clone, Copy)]
pub struct ParametricEngine;

impl ParametricEngine {
    pub fn fit<E: Equation + Clone + Send + 'static>(
        problem: CompiledProblem<E>,
    ) -> Result<ParametricWorkspace<E>> {
        let compiled_state = compile_model_state(&problem);
        let occasion_design = problem.design.occasions.clone();
        if !matches!(problem.method(), EstimationMethod::Parametric(_)) {
            anyhow::bail!(
                "parametric engine received non-parametric method: {:?}",
                problem.method()
            );
        }
        let input = ParametricAlgorithmInput::from_compiled_problem(problem)?;
        let workspace = run_parametric_algorithm(input)?;
        Ok(workspace.with_compiled_state(compiled_state, &occasion_design))
    }
}

pub fn fit<E: Equation + Clone + Send + 'static>(
    problem: CompiledProblem<E>,
) -> Result<FitResult<E>> {
    let workspace = ParametricEngine::fit(problem)?;
    Ok(workspace.into_fit_result())
}
