use anyhow::Result;
use pharmsol::{AssayErrorModel, ErrorPoly, ResidualErrorModel, ResidualErrorModels};
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

fn simple_data() -> Data {
    let subject = Subject::builder("1")
        .covariate("wt", 0.0, 70.0)
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .build();

    Data::new(vec![subject])
}

fn multi_occasion_covariate_data() -> Data {
    let subject = Subject::builder("1")
        .covariate("wt", 0.0, 70.0)
        .covariate("study_day", 0.0, 1.0)
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .reset()
        .covariate("wt", 0.0, 70.0)
        .covariate("study_day", 0.0, 2.0)
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 8.0, 0)
        .build();

    Data::new(vec![subject])
}

#[test]
fn test_parametric_compiler_extracts_model_intent() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::combined(0.5, 0.1));
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?)
        .with_residual_error_models(residual_error);

    let model = ModelDefinition::builder(simple_equation())
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec {
                    name: "ke".to_string(),
                    domain: ParameterDomain::Positive {
                        lower: Some(0.0),
                        upper: Some(1.0),
                    },
                    transform: ModelParameterTransform::LogNormal,
                    initial: Some(0.4),
                    estimate: true,
                    variability: ParameterVariability::SubjectAndOccasion,
                })
                .add(ParameterSpec {
                    name: "v".to_string(),
                    domain: ParameterDomain::Bounded {
                        lower: 1.0,
                        upper: 20.0,
                    },
                    transform: ModelParameterTransform::Identity,
                    initial: None,
                    estimate: true,
                    variability: ParameterVariability::FixedOnly,
                }),
        )
        .observations(observations)
        .covariates(CovariateSpec::Structured(CovariateEffectsSpec {
            subject_effects: Some(CovariateModel::new(
                vec!["ke", "v"],
                vec!["wt"],
                vec![vec![true], vec![false]],
            )?),
            occasion_effects: None,
        }))
        .build()?;

    let compiled = EstimationProblem::builder(model, simple_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Saem(SaemOptions)))
        .output(OutputPlan::disabled())
        .build()?
        .compile()?;

    let state = compile_model_state(&compiled);

    assert_eq!(state.fixed_effects.parameter_names, vec!["ke", "v"]);
    assert_eq!(state.fixed_effects.population_mean.0, vec![0.4, 10.5]);
    assert_eq!(state.transforms.transforms[0], ParametricTransformKind::LogNormal);
    assert_eq!(state.transforms.transforms[1], ParametricTransformKind::Identity);
    assert_eq!(state.variability.subject.enabled_for, vec![true, false]);
    assert_eq!(
        state
            .variability
            .occasion
            .as_ref()
            .expect("occasion variability should be derived from parameter roles")
            .enabled_for,
        vec![true, false]
    );
    assert!(state.covariates.subject_effects.is_some());
    assert!(state.covariates.occasion_effects.is_none());
    assert_eq!(
        state.covariates.subject_effects.as_ref().unwrap().column_names,
        vec!["wt"]
    );
    assert_eq!(
        state.covariates.subject_effects.as_ref().unwrap().parameter_names,
        vec!["ke", "v"]
    );
    assert_eq!(
        state.covariates.subject_effects.as_ref().unwrap().covariate_mask,
        vec![vec![true], vec![false]]
    );
    assert_eq!(
        state.covariates.subject_effects.as_ref().unwrap().values,
        vec![vec![Some(70.0)]]
    );
    Ok(())
}

#[test]
fn test_parametric_compiler_extracts_occasion_covariates() -> Result<()> {
    let assay_error = AssayErrorModel::additive(ErrorPoly::new(0.0, 0.10, 0.0, 0.0), 2.0);
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::combined(0.5, 0.1));
    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(0, assay_error)?)
        .with_residual_error_models(residual_error);

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

    let compiled = EstimationProblem::builder(model, multi_occasion_covariate_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Saem(SaemOptions)))
        .output(OutputPlan::disabled())
        .build()?
        .compile()?;

    let state = compile_model_state(&compiled);
    let occasion = state
        .covariates
        .occasion_effects
        .as_ref()
        .expect("occasion covariates should be compiled into parametric state");

    assert_eq!(occasion.column_names, vec!["study_day"]);
    assert_eq!(occasion.parameter_names, vec!["ke", "v"]);
    assert_eq!(occasion.covariate_mask, vec![vec![true], vec![false]]);
    assert_eq!(occasion.values, vec![vec![Some(1.0)], vec![Some(2.0)]]);
    Ok(())
}