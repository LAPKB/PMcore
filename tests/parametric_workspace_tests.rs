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
fn test_parametric_workspace_preserves_occasion_effect_slots() -> Result<()> {
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

    let compiled = EstimationProblem::builder(model, multi_occasion_data())
        .method(EstimationMethod::Parametric(ParametricMethod::Focei(
            FoceiOptions,
        )))
        .output(OutputPlan::disabled())
        .runtime(RuntimeOptions {
            cycles: 1,
            progress: false,
            ..RuntimeOptions::default()
        })
        .build()?
        .compile()?;

    let workspace = ParametricEngine::fit(compiled)?;
    let occasion_kappa = workspace
        .individuals()
        .occasion_kappa
        .as_ref()
        .expect("occasion effect slots should exist for occasion-enabled models");

    assert_eq!(occasion_kappa.0.len(), 2);
    assert_eq!(occasion_kappa.0[0].subject_index, 0);
    assert_eq!(occasion_kappa.0[0].occasion_index, 0);
    assert_eq!(occasion_kappa.0[1].occasion_index, 1);
    assert_eq!(occasion_kappa.0[0].values.0, vec![0.0, 0.0]);
    assert_eq!(
        workspace.state().variability.subject.enabled_for,
        vec![true, true]
    );
    assert_eq!(
        workspace
            .state()
            .variability
            .occasion
            .as_ref()
            .expect("occasion variability should be present")
            .enabled_for,
        vec![true, false]
    );
    Ok(())
}
