use pmcore::prelude::*;

fn main() {
    let sde = sde! {
        name: "new_iov_sde",
        params: [ke0, ske],
        states: [central, ke_state],
        outputs: [1],
        particles: 11,
        routes: [
            bolus(1) -> central,
        ],
        drift: |x, _t, dx| {
            dx[ke_state] = -x[ke_state] + ke0;
            dx[central] = -x[ke_state] * x[central];
        },
        diffusion: |sigma| {
            sigma[ke_state] = ske;
        },
        init: |_t, x| {
            x[ke_state] = ke0;
        },
        out: |x, _t, y| {
            y[1] = x[central] / 50.0;
        },
    };

    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "central"))
        .with_assay_error_models(
            AssayErrorModels::new()
                .add(
                    0,
                    AssayErrorModel::additive(
                        ErrorPoly::new(-0.00119, 0.44379, -0.45864, 0.16537),
                        0.0,
                    ),
                )
                .unwrap(),
        );

    let model = ModelDefinition::builder(sde)
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ke0", 0.0001, 2.4))
                .add(ParameterSpec::bounded("ske", 0.0001, 0.2)),
        )
        .observations(observations)
        .build()
        .unwrap();

    let data = data::read_pmetrics("examples/new_iov/data.csv").unwrap();
    let mut result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions::default(),
        )))
        .output(OutputPlan {
            write: true,
            path: Some("examples/new_iov/output".to_string()),
        })
        .runtime(RuntimeOptions {
            cycles: 1000,
            cache: true,
            prior: Some(Prior::sobol(100, 347)),
            ..RuntimeOptions::default()
        })
        .run()
        .unwrap();
    result.write_outputs().unwrap();
}
