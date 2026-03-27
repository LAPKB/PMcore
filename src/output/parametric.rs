use anyhow::Result;

use crate::estimation::parametric::{self as posthoc, ParametricWorkspace};
use crate::output::shared::shared_output_file_names;

pub(crate) fn output_file_names<E: pharmsol::Equation>(
    result: &ParametricWorkspace<E>,
) -> Vec<String> {
    let mut files = shared_output_file_names();
    files.extend(
        [
            "population.csv",
            "correlation.csv",
            "individual_parameters.csv",
            "individual_effects.csv",
            "iterations.csv",
            "statistics.csv",
            "shrinkage.csv",
            "residual_error.csv",
        ]
        .into_iter()
        .map(str::to_string),
    );

    if result.uncertainty().has_fim() || result.uncertainty().has_standard_errors() {
        files.push("uncertainty.csv".to_string());
    }

    let has_covariates = result
        .data()
        .subjects()
        .iter()
        .any(|subject| {
            subject
                .occasions()
                .iter()
                .any(|occasion| !occasion.covariates().covariates().is_empty())
        });
    if has_covariates {
        files.push("covariates.csv".to_string());
    }

    files.sort();
    files.dedup();
    files
}

pub fn write_parametric_workspace_outputs<E: pharmsol::Equation>(
    result: &mut ParametricWorkspace<E>,
) -> Result<()> {
    let (idelta, tad) = result.prediction_interval();
    result.write_population()?;
    result.write_individual_parameters()?;
    result.write_individual_effects()?;
    result.write_iteration_log()?;
    posthoc::cache_predictions(result, idelta, tad)?;
    posthoc::write_statistics(result)?;
    result.write_uncertainty()?;
    result.sigma().write(result.output_folder())?;
    result.write_covariates()?;
    Ok(())
}
