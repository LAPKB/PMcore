use pmcore::prelude::*;

fn main() {
    let sde = sde! {
        name: "vanco_sde",
        params: [ka, ke0, kcp, kpc, vol, ske],
        covariates: [wt],
        states: [gut, central, peripheral, ke_state],
        outputs: [1],
        particles: 100,
        routes: [
            bolus(1) -> gut,
        ],
        drift: |x, _t, dx| {
            dx[ke_state] = -x[ke_state] + ke0;
            let ke = x[ke_state];
            dx[gut] = -ka * x[gut];
            dx[central] = ka * x[gut] - (ke + kcp) * x[central] + kpc * x[peripheral];
            dx[peripheral] = kcp * x[central] - kpc * x[peripheral];
        },
        diffusion: |sigma| {
            sigma[ke_state] = ske;
        },
        init: |_t, x| {
            x[ke_state] = ke0;
        },
        out: |x, _t, y| {
            y[1] = x[central] / (vol * wt);
        },
    };

    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(0, "central"))
        .with_assay_error_models(
            AssayErrorModels::new()
                .add(
                    0,
                    AssayErrorModel::additive(ErrorPoly::new(0.00119, 0.20, 0.0, 0.0), 0.0),
                )
                .unwrap(),
        );

    let model = ModelDefinition::builder(sde)
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ka", 0.0001, 2.4))
                .add(ParameterSpec::bounded("ke0", 0.0001, 2.7))
                .add(ParameterSpec::bounded("kcp", 0.0001, 2.4))
                .add(ParameterSpec::bounded("kpc", 0.0001, 2.4))
                .add(ParameterSpec::bounded("vol", 0.2, 12.0))
                .add(ParameterSpec::bounded("ske", 0.0001, 0.2)),
        )
        .observations(observations)
        .build()
        .unwrap();

    let data = data::read_pmetrics("examples/vanco_sde/vanco_clean.csv").unwrap();
    let mut result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions::default(),
        )))
        .output(OutputPlan {
            write: true,
            path: Some("examples/vanco_sde/output".to_string()),
        })
        .runtime(RuntimeOptions {
            cycles: usize::MAX,
            cache: true,
            prior: Some(Prior::sobol(100, 347)),
            ..RuntimeOptions::default()
        })
        .run()
        .unwrap();
    result.write_outputs().unwrap();
}
