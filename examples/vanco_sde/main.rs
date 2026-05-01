use pmcore::prelude::*;

fn main() {
    let sde = equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_params!(p, ka, ke0, kcp, kpc, _vol);
            dx[3] = -x[3] + ke0;
            let ke = x[3];
            // dbg!(x[3], ke0, dx[3]);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - (ke + kcp) * x[1] + kpc * x[2];
            dx[2] = kcp * x[1] - kpc * x[2];
        },
        |p, d| {
            fetch_params!(p, _ka, _ke0, _kcp, _kpc, _vol, ske);
            d[3] = ske;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, _ka, ke0, _kcp, _kpc, _vol);
            x[3] = ke0;
        },
        |x, p, t, cov, y| {
            fetch_params!(p, _ka, _ke0, _kcp, _kpc, vol);
            fetch_cov!(cov, t, wt);
            y[0] = x[1] / (vol * wt);
        },
        100,
    );

    // let ode = equation::ODE::new(
    //     |x, p, _t, dx, _rateiv, _cov| {
    //         fetch_params!(p, ka, ke0, kcp, kpc, _vol);
    //         dx[0] = -ka * x[0];
    //         dx[1] = ka * x[0] - (ke0 + kcp) * x[1] + kpc * x[2];
    //         dx[2] = kcp * x[1] - kpc * x[2];
    //     },
    //     |_p, _t, _cov| lag! {},
    //     |_p, _t, _cov| fa! {},
    //     |_p, _t, _cov, _x| {},
    //     |x, p, t, cov, y| {
    //         fetch_params!(p, _ka, _ke0, _kcp, _kpc, vol);
    //         fetch_cov!(cov, t, wt);
    //         y[0] = x[1] / (vol);
    //     },
    //     (3, 1),
    // );

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
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(NpagOptions)))
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
