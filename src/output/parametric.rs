use anyhow::Result;

use crate::estimation::parametric::{self as posthoc, ParametricWorkspace};

pub fn write_parametric_workspace_outputs<E: pharmsol::Equation>(result: &mut ParametricWorkspace<E>) -> Result<()> {
    let (idelta, tad) = result.prediction_interval();
    result.write_population()?;
    result.write_individual_estimates()?;
    result.write_iteration_log()?;
    posthoc::cache_predictions(result, idelta, tad)?;
    if let Some(predictions) = result.predictions() {
        predictions.write(result.output_folder())?;
    }
    posthoc::write_statistics(result)?;
    result.sigma().write(result.output_folder())?;
    result.write_covariates()?;
    Ok(())
}