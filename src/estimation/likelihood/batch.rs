#![allow(dead_code)] // wired when parametric algorithms cut over to PMcore scoring

use anyhow::{bail, Result};
use ndarray::{Array2, Axis};
use pharmsol::{Data, Equation, Occasion, Subject};

use crate::ResidualErrorModels;
use rayon::prelude::*;

use super::residual::residual_error_model_log_likelihoods;

/// Compute one parametric subject log-likelihood in PMcore.
///
/// pharmsol is used only to generate predictions; PMcore owns residual-error scoring.
pub(crate) fn parametric_subject_log_likelihood(
    equation: &impl Equation,
    subject: &Subject,
    parameter_row: &[f64],
    residual_error_models: &ResidualErrorModels,
) -> f64 {
    let predictions = match equation.estimate_predictions_dense(subject, parameter_row) {
        Ok(predictions) => predictions,
        Err(_) => return f64::NEG_INFINITY,
    };

    residual_error_model_log_likelihoods(&predictions, residual_error_models)
}

/// Score one occasion under its own κ-adjusted parameter vector.
///
/// An occasion is simulated as an independent pharmsol subject so its reset
/// state and occasion-local covariates remain intact. PMcore supplies the
/// occasion-specific ψ parameters and owns residual scoring.
pub(crate) fn parametric_occasion_log_likelihood(
    equation: &impl Equation,
    subject_id: &str,
    occasion: &Occasion,
    parameter_row: &[f64],
    residual_error_models: &ResidualErrorModels,
) -> f64 {
    let occasion_subject = Subject::from_occasions(subject_id.to_owned(), vec![occasion.clone()]);
    parametric_subject_log_likelihood(
        equation,
        &occasion_subject,
        parameter_row,
        residual_error_models,
    )
}

/// Compute parametric subject log-likelihoods in PMcore.
///
/// Each subject has one parameter row. pharmsol is used only to generate
/// predictions; PMcore owns residual-error scoring.
pub(crate) fn parametric_log_likelihood_batch(
    equation: &impl Equation,
    subjects: &Data,
    parameters: &Array2<f64>,
    residual_error_models: &ResidualErrorModels,
) -> Result<Vec<f64>> {
    let subject_refs = subjects.subjects();
    if parameters.nrows() != subject_refs.len() {
        bail!(
            "parameters has {} rows but there are {} subjects",
            parameters.nrows(),
            subject_refs.len()
        );
    }

    if let Some(flat_parameters) = parameters.as_slice() {
        let width = parameters.ncols();
        Ok(subject_refs
            .par_iter()
            .enumerate()
            .map(|(i, subject)| {
                let start = i * width;
                parametric_subject_log_likelihood(
                    equation,
                    subject,
                    &flat_parameters[start..start + width],
                    residual_error_models,
                )
            })
            .collect())
    } else {
        let parameter_rows = parameters
            .axis_iter(Axis(0))
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();

        Ok(subject_refs
            .par_iter()
            .enumerate()
            .map(|(i, subject)| {
                parametric_subject_log_likelihood(
                    equation,
                    subject,
                    &parameter_rows[i],
                    residual_error_models,
                )
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResidualErrorModel;
    use pharmsol::prelude::*;
    use pharmsol::SubjectBuilderExt;

    fn one_compartment_metadata() -> pharmsol::equation::ModelMetadata {
        equation::metadata::new("one_compartment_parametric_parity")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["0"])
            .route(equation::Route::bolus("0").to_state("central"))
    }

    fn one_compartment() -> pharmsol::ODE {
        equation::ODE::new(
            |x, p, _t, dx, b, _rateiv, _cov| {
                fetch_params!(p, ke);
                dx[0] = -ke * x[0] + b[0];
            },
            |_p, _t, _cov| lag! {},
            |_p, _t, _cov| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, v);
                y[0] = x[0] / v;
            },
        )
        .with_nstates(1)
        .with_ndrugs(1)
        .with_nout(1)
        .with_metadata(one_compartment_metadata())
        .unwrap()
    }

    fn data() -> Data {
        Data::new(vec![
            Subject::builder("s1")
                .bolus(0.0, 100.0, "0")
                .observation(1.0, 12.0, "0")
                .missing_observation(2.0, "0")
                .observation(4.0, 4.0, "0")
                .build(),
            Subject::builder("s2")
                .bolus(0.0, 80.0, "0")
                .observation(0.5, 9.0, "0")
                .observation(3.0, 2.5, "0")
                .build(),
        ])
    }

    #[test]
    fn batch_scores_one_parameter_row_per_subject() {
        let equation = one_compartment();
        let data = data();
        let parameters = ndarray::array![[0.15, 8.0], [0.30, 12.0]];
        let error_models =
            ResidualErrorModels::new().add(0, ResidualErrorModel::combined(0.5, 0.1));

        let scores = parametric_log_likelihood_batch(&equation, &data, &parameters, &error_models)
            .expect("pmcore batch");

        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|score| score.is_finite()));
    }
}
