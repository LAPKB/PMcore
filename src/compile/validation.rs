use anyhow::{bail, Result};
use pharmsol::equation::Equation;

use crate::api::EstimationProblem;

pub fn validate_problem<E: Equation>(problem: &EstimationProblem<E>) -> Result<()> {
    if problem.model.parameters.is_empty() {
        bail!("estimation problem requires at least one parameter");
    }

    if problem.runtime.cycles == 0 {
        bail!("runtime cycles must be greater than zero");
    }

    if problem.model.observations.channels.is_empty() {
        bail!("at least one observation channel is required");
    }

    problem.model.parameters.finite_ranges()?;

    Ok(())
}
