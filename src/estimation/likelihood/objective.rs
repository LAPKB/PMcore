use anyhow::{bail, Result};
use ndarray::Array2;
use pharmsol::Equation;

use crate::estimation::{EstimationProblem, Parametric};

use super::batch::parametric_log_likelihood_batch;

/// Score one row of individual parameters per subject for a parametric problem.
///
/// pharmsol is used only for prediction generation. PMcore owns residual-error
/// scoring through the likelihood module.
pub(crate) fn parametric_subject_log_likelihoods<E>(
    problem: &EstimationProblem<E, Parametric>,
    individual_parameters: &Array2<f64>,
) -> Result<Vec<f64>>
where
    E: Equation,
{
    validate_parameter_width(problem, individual_parameters)?;

    parametric_log_likelihood_batch(
        &problem.model.equation,
        &problem.data,
        individual_parameters,
        problem.error_models.models(),
    )
}

fn validate_parameter_width<E>(
    problem: &EstimationProblem<E, Parametric>,
    individual_parameters: &Array2<f64>,
) -> Result<()>
where
    E: Equation,
{
    let expected = problem.parameters().len();
    if individual_parameters.ncols() != expected {
        bail!(
            "individual parameter matrix has {} columns but the parametric problem declares {} parameters",
            individual_parameters.ncols(),
            expected
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::likelihood::batch::parametric_log_likelihood_batch;
    use crate::estimation::ParametricErrorModel;
    use crate::model::Parameter;
    use crate::ResidualErrorModel;
    use pharmsol::prelude::*;
    use pharmsol::SubjectBuilderExt;

    fn one_compartment_metadata() -> pharmsol::equation::ModelMetadata {
        equation::metadata::new("one_compartment_parametric_objective")
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

    fn problem() -> EstimationProblem<pharmsol::ODE, Parametric> {
        let data = Data::new(vec![
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
        ]);

        EstimationProblem::parametric(one_compartment(), data)
            .parameter(Parameter::log("ke"))
            .parameter(Parameter::log("v"))
            .error_model(
                "0",
                ParametricErrorModel::new(ResidualErrorModel::combined(0.5, 0.1)).fixed(),
            )
            .build()
            .unwrap()
    }

    #[test]
    fn objective_uses_batch_subject_likelihoods() {
        let problem = problem();
        let parameters = ndarray::array![[0.15, 8.0], [0.30, 12.0]];

        let expected = parametric_log_likelihood_batch(
            &problem.model.equation,
            &problem.data,
            &parameters,
            problem.error_models.models(),
        )
        .unwrap();
        let actual = parametric_subject_log_likelihoods(&problem, &parameters).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn objective_rejects_wrong_parameter_width() {
        let problem = problem();
        let parameters = ndarray::array![[0.15], [0.30]];

        let err = parametric_subject_log_likelihoods(&problem, &parameters).unwrap_err();
        assert!(err.to_string().contains("declares 2 parameters"));
    }
}
