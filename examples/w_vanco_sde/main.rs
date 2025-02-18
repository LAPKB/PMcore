use pmcore::prelude::{
    settings::{Parameters, Prior, Settings},
    *,
};

fn main() {
    //% cp ~/Documents/CHLA/IOV2024/vanco_PICU.csv ./test.csv
    //% cargo run --release --example vanco_sde

    let sde = equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            fetch_params!(p, ka, ke0, kcp, kpc, _vol, _ske);
            dx[3] = -x[3] + ke0;
            let ke = x[3]; // ke is a mean-reverting stochastic parameter
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - (ke + kcp) * x[1] + kpc * x[2];
            dx[2] = kcp * x[1] - kpc * x[2];
        },
        |p, d| {
            fetch_params!(p, _ka, _ke0, _kcp, _kpc, _vol, ske);
            d[3] = ske;
        },
        |_p| lag! {}, // remember println!() .. it's a macro ... {} empty function
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, _ka, ke0, _kcp, _kpc, _vol, _ske);
            //    x[0] = 0.0; // spot check suggests all subjects have a 0-dose at time 0
            //    x[1] = 0.0;
            x[3] = ke0;
        },
        |x, p, t, cov, y| {
            fetch_params!(p, _ka, _ke0, _kcp, _kpc, vol, _ske);
            fetch_cov!(cov, t, wt);
            y[0] = x[1] / (vol * wt);
        },
        (4, 1), // (input equations, output equations)
        11,
    );

    let mut settings = Settings::new();

    let params = Parameters::builder()
        .add("ka", 0.0001, 2.4, false)
        .add("ke0", 0.0001, 2.7, false)
        .add("kcp", 0.0001, 2.4, false)
        .add("kpc", 0.0001, 2.4, false)
        .add("vol", 0.2, 1.2, false)
        .add("ske", 0.0001, 0.2, false)
        .build()
        .unwrap();

    settings.set_parameters(params);
    settings.set_cycles(10);
    settings.set_cache(true);
    settings.set_error_poly((0.00119, 0.20, 0.0, 0.0));
    settings.set_error_value(1.0);
    settings.set_error_type(ErrorType::Add);
    settings.set_output_path("examples/w_vanco_sde/output");
    settings.set_prior(Prior {
        sampler: "sobol".to_string(),
        points: 10000,
        seed: 347,
        file: None,
    });
    settings.set_output_write(true);
    settings.set_log_level(settings::LogLevel::DEBUG);
    setup_log(&settings).unwrap();
    let data = data::read_pmetrics("examples/w_vanco_sde/test.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, sde, data).unwrap();
    algorithm.initialize().unwrap();
    while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
}
