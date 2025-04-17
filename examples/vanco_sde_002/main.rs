use pmcore::prelude::{
    settings::{Parameters, Prior, Settings},
    *,
};

pub(crate) fn main() {
    let sde = equation::SDE::new(
        |x, p, _t, dx, rateiv, cov| {
            fetch_params!(p, ke0, _v0, kcp, well);
            fetch_cov!(cov,_t,scr,wt);
            dx[0] = ke0 - x[0]; // x[0] moves toward ke0
            dx[1] = 0.0; // v0 - x[1];
            let ke = x[0];
            let vol = x[1];
            let norm_wt = wt/70.0;
            let a_ke = norm_wt.powf(0.75) * (0.4385789/scr).powf(0.873566) / (norm_wt * vol);
            dx[2] = rateiv[0] - ( a_ke * ke + kcp) * x[2] + ( well * kcp ) * x[3];
            dx[3] = kcp * x[2] - ( well * kcp ) * x[3];
        },
        |p, d| {
            fetch_params!(p, _ke0, _vol, _well, ske, _svol);
            d[0] = ske;
            //d[1] = svol;
            // the above increments MUST match the state increments of x
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, v0, _well, _ske, _svol);
            x[0] = ke0;
            x[1] = v0;
            x[2] = 0.0;
            x[3] = 0.0;
        },
        |x, _p, t, cov, y| {
            // fetch_params!(p, _ke0, _v0, _kcp, _well, _ske, _svol);
            fetch_cov!(cov, t, wt);
            let norm_wt = wt/70.0;
            let vol = x[1];
            y[0] = x[2] / (vol * norm_wt);
        },
        (4, 1),
        17,
    );

    let _ode = equation::ODE::new(
        |x, p, _t, dx, rateiv, cov| {
            fetch_params!(p, ke0, v0, kcp, well);
            fetch_cov!(cov,_t,scr,wt);
            let norm_wt = wt/70.0;
            dx[0] = rateiv[0] - (( norm_wt.powf(0.75) * (0.4385789/scr).powf(0.873566) / (norm_wt * v0) ) * ke0 + kcp) * x[0];
            dx[1] = kcp * x[0] - ( well * kcp ) * x[1]
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_params!(p, _ke0, v0, _kcp, _well);
            fetch_cov!(cov, t, wt);
            y[0] = x[0] / (v0 * wt / 70.0);
        },
         (2, 1),
    );

    let mut settings = Settings::new();

    let params = Parameters::builder()
        .add("ke0", 1.00, 7.25, false) // Paula's work: mean Ke = 0.137/Hr and mean Vol = 0.425L/kg in prenatal SoAmerican infants, 1 comp mdoel
        .add("v0", 25.000, 175.0, false)
        .add("kcp", 0.0001, 0.1, false)
        .add("well", 0.125, 16.00, false)
        .add("ske", 1e-7, 0.0001, false)
        // .add("svol",1e-7, 0.0001,false)
        .build()
        .unwrap();

    settings.set_parameters(params);
    settings.set_cycles(usize::MAX);
    settings.set_cache(true);
    settings.set_error_poly((0.05, 0.035, 0.0, 0.0));
    settings.set_error_value(10.0);
    // settings.set_error_value(2.5516439936509987);
    // settings.set_error_type(ErrorType::Add);
    settings.set_error_type(ErrorType::Prop);
    settings.set_output_path("examples/vanco_sde_002/output_sde_ske_tmp"); // *** SET OUTPUT DIRECTORY HERE ***
    settings.set_prior(Prior {
        sampler: "sobol".to_string(),
        points: 131073,
        seed: 347,
        file: Some(String::from("examples/vanco_sde_002/output_ode/theta_ske.csv")), // None,
    });
    settings.set_output_write(true);
    settings.set_log_level(settings::LogLevel::DEBUG);
    setup_log(&settings).unwrap();

    let data = data::read_pmetrics("examples/vanco_sde_002/vclean.csv").unwrap();

    let mut algorithm = dispatch_algorithm(settings, sde, data).unwrap();
    algorithm.initialize().unwrap();
    while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
}
