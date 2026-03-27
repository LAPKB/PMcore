use anyhow::Result;
use pharmsol::{AssayErrorModel, ErrorPoly};
use pmcore::prelude::*;

fn simple_equation() -> equation::ODE {
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
}

fn multi_subject_data() -> Data {
    let first = Subject::builder("1")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .build();

    let second = Subject::builder("2")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 9.0, 0)
        .observation(3.0, 7.0, 0)
        .build();

    Data::new(vec![first, second])
}

fn structured_covariate_data() -> Data {
    let subject = Subject::builder("1")
        .covariate("wt", 0.0, 70.0)
        .covariate("study_day", 0.0, 1.0)
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .reset()
        .covariate("wt", 0.0, 72.0)
        .covariate("study_day", 0.0, 2.0)
        .bolus(0.0, 100.0, 0)
        .observation(1.5, 8.0, 0)
        .build();

    Data::new(vec![subject])
}

fn simple_problem() -> Result<EstimationProblem<equation::ODE>> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

    EstimationProblem::builder(model, multi_subject_data())
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions,
        )))
        .output(OutputPlan::disabled())
        .build()
}

#[test]
fn test_compile_problem_builds_indexes() -> Result<()> {
    let compiled = simple_problem()?.compile()?;

    assert_eq!(compiled.design.subject_count(), 2);
    assert_eq!(compiled.design.occasion_count(), 2);
    assert_eq!(compiled.observation_index.len(), 4);
    assert_eq!(compiled.design.parameter_names, vec!["ke", "v"]);
    Ok(())
}

#[test]
fn test_compile_problem_builds_algorithm_settings() -> Result<()> {
    let compiled = simple_problem()?.compile()?;

    assert_eq!(compiled.method().algorithm(), Algorithm::NPAG);
    assert_eq!(compiled.design.parameter_names.len(), 2);
    assert!(!compiled.output_plan().write);
    Ok(())
}

#[test]
fn test_compile_problem_extracts_structured_covariate_values() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke", 0.1, 1.0))
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .covariates(CovariateSpec::Structured(CovariateEffectsSpec {
            subject_effects: Some(CovariateModel::new(
                vec!["ke", "v"],
                vec!["wt"],
                vec![vec![true], vec![false]],
            )?),
            occasion_effects: Some(CovariateModel::new(
                vec!["ke", "v"],
                vec!["study_day"],
                vec![vec![true], vec![false]],
            )?),
        }))
        .build()?;

    let compiled = EstimationProblem::builder(model, structured_covariate_data())
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions,
        )))
        .output(OutputPlan::disabled())
        .build()?
        .compile()?;

    assert_eq!(
        compiled.design.structured_covariates.subject_columns,
        vec!["wt"]
    );
    assert_eq!(
        compiled.design.structured_covariates.occasion_columns,
        vec!["study_day"]
    );
    assert_eq!(compiled.design.structured_covariates.subject_rows.len(), 1);
    assert_eq!(compiled.design.structured_covariates.occasion_rows.len(), 2);
    assert_eq!(
        compiled.design.structured_covariates.subject_rows[0].values,
        vec![Some(70.0)]
    );
    assert_eq!(
        compiled.design.structured_covariates.occasion_rows[0].values,
        vec![Some(1.0)]
    );
    assert_eq!(
        compiled.design.structured_covariates.occasion_rows[1].values,
        vec![Some(2.0)]
    );
    Ok(())
}
