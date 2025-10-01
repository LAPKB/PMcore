use pmcore::prelude::{
    settings::{Parameters, Prior, Settings},
    *,
};

pub(crate) fn main() {
    let _sde = equation::SDE::new(
        |x, p, t, dx, rateiv, cov| {
            fetch_params!(p, fke0, kcp, fkpc, fvol0, mmpower, _svol);
            fetch_cov!(cov, t, age, weight, crcl);
            dx[3] = -x[3] + fvol0; // + fke0;

            let volume = x[3] * weight; 
            let clr0 = 0.06*(0.9001446 * crcl + 3.698084) ; // (weight/70.0).powf(0.75) * crcl;
            let pma = 0.8 + age;
            let mm50 : f64 = 1.0; // full function is expected at 2-3 YRO
            let clr = clr0
              *(pma.powf(mmpower)/(pma.powf(mmpower) + mm50.powf(mmpower)));
            // let ke = x[3] * clr / volume;
            let ke = fke0 * clr / volume;

            // let ke = x[3]; // naive model w/no covariates

            // dbg!(x[3], fke0, dx[3]); // data.csv has 0 duration input (always)
            // dx[0] = -ka * x[0]; // vancomycin is an IV drug ... not sure why this is in model?
            dx[1] = rateiv[0] - (ke + kcp) * x[1] + fkpc * kcp * x[2];
            dx[2] = kcp * (x[1] - fkpc * x[2]);
            //
            // I don't really want to know the probability of each path; I want to
            // know the area between the path and the mean.
            //
            dx[0] = x[3] - fvol0; // with enough samples this goes to zero
            dx[4] = (x[3] - fvol0).powf(2.0)/fvol0; // Chi^2

        },
        |p, d| {
            fetch_params!(p, _fke0, _kcp, _fkpc, _fvol0, _mmpower, svol);
            d[3] = svol;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, _fke0, _kcp, _fkpc, fvol0, _mmpower, _svol);
            x[3] = fvol0;
        },
        |x, p, t, cov, y| {
            fetch_params!(p, _fke0, _kcp, _fkpc, _fvol0, _mmpower, _svol);
            fetch_cov!(cov, t, weight);
            let volume = x[3] * weight; 
            y[0] = x[1] / (volume * weight);
            y[1] = x[0]; // -> 0
            y[2] = x[4]; // \sum Chi^2 // this will be usedin pharmsol likelihood
        },
        (5, 3), // 
        17,
    );

    let ode = equation::ODE::new(
        |x, p, t, dx, rateiv, cov| {
            fetch_params!(p // , ka
                , fke0, kcp, fkpc, fvol0, mmpower, _svol
              );
            // fetch_cov!(cov, t, scr, weight); // scr, age, height, weight, sex_m
            // let volume: f64 = vol * weight / 70.0;
            // let clr = (weight/70.0).powf(0.75) * (0.3406/scr).powf(1.7748);
            fetch_cov!(cov, t, age, weight, crcl); // cr, age, height, weight, sex_m, crcl
            // let volume: f64 = fvol0 * (11.0 + 0.29 * age + 0.33 * weight) * weight ; // vol * weight / 70.0;
            let volume = fvol0 * weight; 
            let clr0 = 0.06*(0.9001446 * crcl + 3.698084) ; // (weight/70.0).powf(0.75) * crcl;
            let pma = 0.8 + age;
            let mm50 : f64 = 1.0; // full functio:q!n is expected at 2-3 YRO
            let clr = clr0
              // *(weight/70.0 ).powf(0.75)
              *(pma.powf(mmpower)/(pma.powf(mmpower) + mm50.powf(mmpower)));
            let kel = fke0 * clr / volume;

            // dx[0] = -ka * x[0];
            dx[0] = rateiv[0] + fkpc * kcp * x[1] // + ka * x[0] // kpc *= kcp, kcp is a multiplier on kel
              - (kel + kcp) * x[0]
              ;
            dx[1] = kcp * ( x[0] - fkpc * x[1] );
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_params!(p // , _ka
                , _fke0, _kcp, _kpc, fvol0);
            fetch_cov!(cov, t, weight); // scr, age, weight, ht, male
            // let volume: f64 = fvol0 * (11.0 + 0.29 * age + 0.33 * weight) * weight ;
            let volume = fvol0 * weight;
            y[0] = x[0] / volume ;
        },
         (2, 1),
    );

    let mut settings = Settings::new();

    let params = Parameters::builder()
        // .add("ka", 0.0001, 2.4, false)
        .add("fke0", 0.05, 2.75, false)
        .add("kcp", 0.001, 1.0, false)
        .add("fkpc", 0.0005, 1.0, false)
        .add("fvol0", 0.01, 10.20, false) // was 0.0001 to 250
        .add("mmpower", 0.75, 3.5, false)
        .add("svol", 0.3e-26, 0.1e-20, false) // stochastic element on ke w/mean = fke0
        .build()
        .unwrap();

        /*
        "ke0", 0.05, 2.75
        "kcp", 0.001, 1.0
        "kpc", 0.0005, 0.45
        "vol", 10.0, 150.0
        */

    settings.set_algorithm(Algorithm::NPAG);
    
    settings.set_parameters(params);
    //    .set_error_model(ErrorModel::Proportional, 3.923582836468327, (0.125, 0.025, 0.0, 0.0))
    //    .build();

    // 1.752, -0.18, 0.005
    // +1.5 c(1.752, 0.08, 0,0) "works" w/good posterior mean, but population underfits the highs

    settings.set_cycles(usize::MAX);
    settings.set_cache(true);
    settings.set_error_poly((0.125, 0.025, 0.0, 0.0));
    settings.set_error_value(3.923582836468327);
    // settings.set_error_value(2.5516439936509987);
    // settings.set_error_type(ErrorType::Add);
    settings.set_error_type(ErrorType::Prop);
    settings.set_output_path("examples/vanco_sde/output.tmp_ODE"); // _sde_vol_20250723_fixed_sigma_inside_tf.tmp"); // *** SET OUTPUT DIRECTORY HERE ***
    settings.set_prior(Prior {
        sampler: "sobol".to_string(),
        points: 244, // E[t] = 6w(10^5), (2.5^6)
        seed: 347,
        file: None, // Some(String::from("examples/vanco_sde/theta_ode_20250722.csv")), // None,
    });
    settings.set_output_write(true);
    settings.set_log_level(settings::LogLevel::DEBUG);
    settings.set_idelta(1.0);
    setup_log(&settings).unwrap();

    let data = data::read_pmetrics("examples/vanco_sde/tmp0.csv").unwrap();

    let mut algorithm = dispatch_algorithm(settings, ode, data).unwrap();

    // TODO_wmy copy main.rs to the output directory before running:
    // cp examples/vanco_sde/main.rs ./examples/vanco_sde/output_sde_vol_20250723_fixed_sigma.tmp 
    // output_sde_vol_20250722.tmp
    // output_ode_vol_20250722.tmp

    algorithm.initialize().unwrap();
    while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
}
