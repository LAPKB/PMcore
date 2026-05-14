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
    .with_nstates(1)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("nonparametric_engine")
            .parameters(["ke", "v"])
            .states(["central"])
            .outputs(["0"])
            .route(equation::Route::bolus("0").to_state("central")),
    )
    .expect("metadata attachment should validate")
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
    let compiled = EstimationProblem::builder(simple_equation(), simple_data())
        .parameter(Parameter::bounded("ke", 0.1, 1.0))?
        .parameter(Parameter::bounded("v", 1.0, 20.0))?
        .method(Npag::new())
        .error("0", assay_error)?
        .cycles(1)
        .progress(false)
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
