use anyhow::{bail, Result};
use pharmsol::equation::Equation;

use crate::api::EstimationProblem;
use crate::model::EquationMetadataSource;

pub fn validate_problem<E: Equation + EquationMetadataSource>(
    problem: &EstimationProblem<E>,
) -> Result<()> {
    if problem.model.parameters.is_empty() {
        bail!("estimation problem requires at least one parameter");
    }

    if problem.runtime.cycles == 0 {
        bail!("runtime cycles must be greater than zero");
    }

    if problem.model.output_count() == 0 {
        bail!("at least one equation output is required");
    }

    let error_models = problem.error_models.models();
    if error_models.iter().next().is_none() {
        bail!("at least one nonparametric error model is required");
    }

    problem.model.parameters.finite_ranges()?;

    Ok(())
}
