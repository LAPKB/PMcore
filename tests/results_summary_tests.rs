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
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .build();

    Data::new(vec![subject])
}

fn multi_occasion_data() -> Data {
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .reset()
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 9.0, 0)
        .observation(2.0, 7.5, 0)
        .build();

    Data::new(vec![subject])
}

#[test]
fn test_nonparametric_fit_result_summary_surface() -> Result<()> {
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

    let result = EstimationProblem::builder(model, simple_data())
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 1,
            cache: true,
            progress: false,
            idelta: 0.12,
            tad: 0.0,
            prior: None,
            ..RuntimeOptions::default()
        })
        .run()?;

    let summary = result.summary();

    assert_eq!(summary.parameter_count, 2);
    assert_eq!(summary.subject_count, 1);
    assert_eq!(summary.observation_count, 2);
    assert_eq!(result.population_summary().parameters.len(), 2);
    assert_eq!(result.individual_summaries().len(), 1);
    let diagnostics = result.diagnostics();
    assert_eq!(
        diagnostics.estimator_metadata.get("algorithm"),
        Some(&"NPAG".to_string())
    );
    assert_eq!(
        diagnostics.estimator_metadata.get("outputs_requested"),
        Some(&"false".to_string())
    );
    assert!(!diagnostics.convergence_notes.is_empty());
    assert!(diagnostics.deferred_features.is_empty());
    let predictions = result.predictions();
    assert!(!predictions.available);
    assert!(result.artifacts().files.is_empty());
    assert!(result.artifacts().expected_files.is_empty());
    Ok(())
}

#[test]
fn test_parametric_fit_result_diagnostics_expose_iov_boundary() -> Result<()> {
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
                    domain: ParameterDomain::Bounded {
                        lower: 0.1,
                        upper: 1.0,
                    },
                    transform: ModelParameterTransform::Identity,
                    initial: Some(0.4),
                    estimate: true,
                    variability: ParameterVariability::SubjectAndOccasion,
                })
                .add(ParameterSpec::bounded("v", 1.0, 20.0)),
        )
        .observations(observations)
        .build()?;

    let result = EstimationProblem::builder(model, multi_occasion_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Focei(
            FoceiOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 1,
            progress: false,
            ..RuntimeOptions::default()
        })
        .run()?;

    let diagnostics = result.diagnostics();
    assert!(diagnostics
        .warnings
        .iter()
        .any(|warning| warning.contains("occasion-level inference remains deferred")));
    assert_eq!(
        diagnostics.estimator_metadata.get("algorithm"),
        Some(&"FOCEI".to_string())
    );
    assert_eq!(
        diagnostics.estimator_metadata.get("occasion_inference"),
        Some(&"deferred".to_string())
    );
    assert_eq!(
        diagnostics.estimator_metadata.get("outputs_requested"),
        Some(&"false".to_string())
    );
    assert_eq!(
        diagnostics.estimator_metadata.get("residual_error_output"),
        Some(&"disabled".to_string())
    );
    assert_eq!(
        diagnostics.estimator_metadata.get("uncertainty_output"),
        Some(&"disabled".to_string())
    );
    assert!(diagnostics
        .deferred_features
        .iter()
        .any(|feature| feature == "occasion_inference"));
    Ok(())
}

#[test]
fn test_parametric_population_summary_uses_transform_aware_cv() -> Result<()> {
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
        .build()?;

    let result = EstimationProblem::builder(model, simple_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Focei(
            FoceiOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 3,
            progress: false,
            ..RuntimeOptions::default()
        })
        .run()?;

    let summary = result.population_summary();
    let diagnostics = result.diagnostics();
    let v_summary = summary
        .parameters
        .iter()
        .find(|parameter| parameter.name == "v")
        .expect("v should be present in the population summary");
    let expected_cv = 100.0 * v_summary.sd / v_summary.mean.abs();

    assert!((v_summary.cv_percent - expected_cv).abs() < 1e-10);
    assert!(v_summary.cv_percent.is_finite());
    assert_eq!(
        diagnostics.estimator_metadata.get("residual_error_model"),
        Some(&"combined".to_string())
    );
    assert_eq!(
        diagnostics.estimator_metadata.get("residual_error_output"),
        Some(&"disabled".to_string())
    );
    assert_eq!(
        diagnostics.estimator_metadata.get("uncertainty_output"),
        Some(&"disabled".to_string())
    );
    Ok(())
}
