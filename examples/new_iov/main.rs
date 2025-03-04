use pmcore::prelude::*;

fn main() {
    let sde = equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, _ske);
            // let ke0 = 1.2;
            dx[1] = -x[1] + ke0;
            let ke = x[1];
            // user defined
            dx[0] = -ke * x[0];
        },
        |p, d| {
            fetch_params!(p, _ke0, ske);
            d[1] = ske;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _ske);
            x[1] = ke0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0, _ske);
            y[0] = x[0] / 50.0;
        },
        (2, 1),
        11,
    );

    let params = Parameters::builder()
        .add("ke0", 0.0001, 2.4, false)
        .add("ske", 0.0001, 0.2, false)
        .build()
        .unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_model(
            ErrorModel::Additive,
            0.0,
            (-0.00119, 0.44379, -0.45864, 0.16537),
        )
        .build();

    settings.set_cycles(1000);
    settings.set_cache(true);
    settings.set_output_path("examples/new_iov/output");
    settings.set_prior(Prior {
        sampler: "sobol".to_string(),
        points: 100,
        seed: 347,
        file: None,
    });
    settings.set_output_write(true);
    setup_log(&settings).unwrap();
    let data = data::read_pmetrics("examples/new_iov/data.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, sde, data).unwrap();
    algorithm.initialize().unwrap();
    while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
}
