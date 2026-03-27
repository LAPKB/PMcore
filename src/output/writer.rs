use anyhow::Result;
use pharmsol::{Censor, Equation};
use serde::Serialize;

use crate::estimation::nonparametric as np_estimation;
use crate::estimation::nonparametric::NonparametricWorkspace;
use crate::estimation::parametric as param_estimation;
use crate::estimation::parametric::ParametricWorkspace;
use crate::output::{nonparametric as np_output, parametric, shared};
use crate::results::FitResult;
use crate::results::{
    nonparametric_diagnostics, parametric_diagnostics, FitSummary,
};

#[derive(Debug, Clone, Serialize)]
struct SharedPredictionRow {
    id: String,
    time: f64,
    outeq: usize,
    block: usize,
    obs: Option<f64>,
    cens: Censor,
    pred_population: f64,
    pred_individual: f64,
    residual_population: Option<f64>,
    residual_individual: Option<f64>,
    source_method: String,
}

pub fn write_result<E: Equation>(result: &mut FitResult<E>) -> Result<()> {
    match result {
        FitResult::Nonparametric(inner) => write_nonparametric_result(inner)?,
        FitResult::Parametric(inner) => write_parametric_workspace_result(inner)?,
    }

    Ok(())
}

pub fn write_nonparametric_result<E: Equation>(
    result: &mut NonparametricWorkspace<E>,
) -> Result<()> {
    if !result.should_write_outputs() {
        return Ok(());
    }

    let folder = result.output_folder().to_string();
    shared::write_settings(&folder, result.run_configuration())?;
    shared::write_summary(&folder, &nonparametric_summary(result))?;
    shared::write_diagnostics(&folder, &nonparametric_diagnostics(result))?;
    np_output::write_nonparametric_outputs(result)?;

    if let Some(predictions) = result.predictions() {
        let rows = predictions
            .predictions()
            .iter()
            .map(|row| SharedPredictionRow {
                id: row.id().to_string(),
                time: row.time(),
                outeq: row.outeq(),
                block: row.block(),
                obs: row.obs(),
                cens: row.censoring(),
                pred_population: row.pop_mean(),
                pred_individual: row.post_mean(),
                residual_population: row.obs().map(|obs| obs - row.pop_mean()),
                residual_individual: row.obs().map(|obs| obs - row.post_mean()),
                source_method: "nonparametric".to_string(),
            });
        shared::write_csv_rows(&folder, "predictions.csv", rows)?;
    }

    Ok(())
}

pub fn write_parametric_workspace_result<E: Equation>(
    result: &mut ParametricWorkspace<E>,
) -> Result<()> {
    if !result.should_write_outputs() {
        return Ok(());
    }

    let folder = result.output_folder().to_string();
    shared::write_settings(&folder, result.run_configuration())?;
    shared::write_summary(&folder, &parametric_summary(result))?;
    shared::write_diagnostics(&folder, &parametric_diagnostics(result))?;
    parametric::write_parametric_workspace_outputs(result)?;

    if let Some(predictions) = result.predictions() {
        let rows = predictions
            .predictions()
            .iter()
            .map(|row| SharedPredictionRow {
                id: row.id().to_string(),
                time: row.time(),
                outeq: row.outeq(),
                block: row.block(),
                obs: row.obs(),
                cens: row.censoring(),
                pred_population: row.ppred(),
                pred_individual: row.ipred(),
                residual_population: row.obs().map(|obs| obs - row.ppred()),
                residual_individual: row.ires(),
                source_method: "parametric".to_string(),
            });
        shared::write_csv_rows(&folder, "predictions.csv", rows)?;
    }

    Ok(())
}

fn nonparametric_summary<E: Equation>(result: &NonparametricWorkspace<E>) -> FitSummary {
    np_estimation::fit_summary(result)
}

fn parametric_summary<E: Equation>(result: &ParametricWorkspace<E>) -> FitSummary {
    param_estimation::fit_summary(result)
}
