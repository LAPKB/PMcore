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

fn simple_data() -> Data {
    let subject = Subject::builder("1")
        .bolus(0.0, 100.0, 0)
        .observation(1.0, 10.0, 0)
        .observation(2.0, 8.0, 0)
        .build();

    Data::new(vec![subject])
}

#[test]
fn test_nonparametric_engine_returns_workspace() -> Result<()> {
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

    let compiled = EstimationProblem::builder(model, simple_data())
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
        .build()?
        .compile()?;

    let workspace = NonparametricEngine::fit(compiled)?;
    assert!(workspace.objf().is_finite());
    assert_eq!(workspace.get_theta().parameters().len(), 2);

    let fit_result = workspace.into_fit_result();
    assert_eq!(fit_result.population_summary().parameters.len(), 2);
    assert_eq!(fit_result.individual_summaries().len(), 1);
    Ok(())
}
