use anyhow::Ok;
use pmcore::prelude::*;
fn main() -> Result<()> {
    let sde = equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0);
            // let ke0 = 1.2;
            dx[1] = -x[1] + ke0;
            let ke = x[1];
            // user defined
            dx[0] = -ke * x[0];
        },
        |p, d| {
            fetch_params!(p, _ke0);
            d[1] = 0.1;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0);
            x[1] = ke0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0);
            y[0] = x[0] / 50.0;
        },
        10000,
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1);

    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "cp"))
        .with_assay_error_models(AssayErrorModels::new().add(
            0,
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.0, 0.0, 0.0), 0.0000757575757576),
        )?);

    let model = ModelDefinition::builder(sde)
        .parameters(ParameterSpace::new().add(ParameterSpec::bounded("ke0", 0.001, 2.0)))
        .observations(observations)
        .build()?;

    let data = data::read_pmetrics("examples/iov/test.csv").unwrap();
    let mut result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(NpagOptions)))
        .output(OutputPlan {
            write: true,
            path: Some("examples/iov/output".to_string()),
        })
        .runtime(RuntimeOptions {
            cycles: 100000,
            prior: Some(Prior::sobol(100, 347)),
            ..RuntimeOptions::default()
        })
        .run()
        .unwrap();
    result.write_outputs().unwrap();

    Ok(())
}
