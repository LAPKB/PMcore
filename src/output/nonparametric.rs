use anyhow::Result;

use crate::estimation::nonparametric::NonparametricWorkspace;

pub fn write_nonparametric_outputs<E: pharmsol::Equation>(
    result: &mut NonparametricWorkspace<E>,
) -> Result<()> {
    let parameter_names = result.get_theta().parameters().names();
    result
        .cycle_log()
        .write(result.output_folder(), &parameter_names)?;
    result.write_theta()?;
    result.write_covs()?;
    result.write_posterior()?;
    let (idelta, tad) = result.prediction_interval();
    result.write_predictions(idelta, tad)?;
    Ok(())
}
