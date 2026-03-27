use anyhow::Result;

use crate::estimation::nonparametric::NonparametricWorkspace;
use crate::output::shared::shared_output_file_names;

pub(crate) fn output_file_names<E: pharmsol::Equation>(
    result: &NonparametricWorkspace<E>,
) -> Vec<String> {
    let mut files = shared_output_file_names();
    files.extend([
        "iterations.csv",
        "theta.csv",
        "posterior.csv",
    ]
    .into_iter()
    .map(str::to_string));

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

pub fn write_nonparametric_outputs<E: pharmsol::Equation>(
    result: &mut NonparametricWorkspace<E>,
) -> Result<()> {
    let parameter_names = result.get_theta().parameters().names();
    result
        .cycle_log()
        .write(result.output_folder(), &parameter_names)?;
    result.write_theta()?;
    result.write_covariates()?;
    result.write_posterior()?;
    let (idelta, tad) = result.prediction_interval();
    result.calculate_predictions(idelta, tad)?;
    Ok(())
}
