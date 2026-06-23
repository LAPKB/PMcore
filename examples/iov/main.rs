use anyhow::Ok;
use pmcore::prelude::*;
fn main() -> Result<()> {
    let sde = sde! {
        name: "iov_sde",
        params: [ke0],
        states: [central, ke_state],
        outputs: [1],
        particles: 10000,
        routes: [
            bolus(1) -> central,
        ],
        drift: |x, _t, dx| {
            dx[ke_state] = -x[ke_state] + ke0;
            dx[central] = -x[ke_state] * x[central];
        },
        diffusion: |sigma| {
            sigma[ke_state] = 0.1;
        },
        init: |_t, x| {
            x[ke_state] = ke0;
        },
        out: |x, _t, y| {
            y[1] = x[central] / 50.0;
        },
    };

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
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions::default(),
        )))
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
