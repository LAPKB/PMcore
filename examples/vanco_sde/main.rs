use pmcore::prelude::{
    settings::{Parameters, Prior, Settings},
    *,
};

fn main() {
    let sde = equation::SDE::new(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ke0, kcp, kpc, _vol);
            dx[2] = -x[2] + ke0;
            let ke = x[2];
            // dbg!(x[3], ke0, dx[3]);

            dx[0] = -(ke + kcp) * x[0] + kpc * x[1] + rateiv[0];
            dx[1] = kcp * x[0] - kpc * x[1];
        },
        |p, d| {
            fetch_params!(p, _ke0, _kcp, _kpc, _vol, ske);
            d[2] = ske;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _kcp, _kpc, _vol);
            x[2] = ke0;
        },
        |x, p, t, cov, y| {
            fetch_params!(p, _ke0, _kcp, _kpc, vol);
            fetch_cov!(cov, t, wt);
            y[0] = x[0] / (vol * wt);
        },
        (3, 1),
        100,
    );

    // let ode = equation::ODE::new(
    //     |x, p, _t, dx, _rateiv, _cov| {
    //         fetch_params!(p, ka, ke0, kcp, kpc, _vol);
    //         dx[0] = -ka * x[0];
    //         dx[1] = ka * x[0] - (ke0 + kcp) * x[1] + kpc * x[2];
    //         dx[2] = kcp * x[1] - kpc * x[2];
    //     },
    //     |_p| lag! {},
    //     |_p| fa! {},
    //     |_p, _t, _cov, _x| {},
    //     |x, p, t, cov, y| {
    //         fetch_params!(p, _ka, _ke0, _kcp, _kpc, vol);
    //         fetch_cov!(cov, t, wt);
    //         y[0] = x[1] / (vol);
    //     },
    //     (3, 1),
    // );

    let mut settings = Settings::new();

    let params = Parameters::builder()
        // .add("ka", 0.0001, 2.4, false)
        .add("ke0", 0.0001, 5.0, false)
        .add("kcp", 0.0001, 5.0, false)
        .add("kpc", 0.0001, 5.0, false)
        .add("vol", 0.2, 50.0, false)
        .add("ske", 0.0001, 2.0, false)
        .build()
        .unwrap();

    settings.set_parameters(params);
    settings.set_cycles(usize::MAX);
    settings.set_cache(true);
    settings.set_error_poly((0.00119, 0.10, 0.0, 0.0));
    settings.set_error_value(2.5516439936509987);
    settings.set_error_type(ErrorType::Add);
    settings.set_output_path("examples/vanco_sde/output");
    settings.set_prior(Prior {
        sampler: "sobol".to_string(),
        points: 100,
        seed: 22,
        file: None,
    });
    settings.set_output_write(true);
    settings.set_log_level(settings::LogLevel::DEBUG);
    setup_log(&settings).unwrap();
    let data = data::read_pmetrics("examples/vanco_sde/vanco_clean.csv").unwrap();

    let mut algorithm = dispatch_algorithm(settings, sde, data).unwrap();
    algorithm.initialize().unwrap();
    while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
}
