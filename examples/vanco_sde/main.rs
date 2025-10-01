use pmcore::prelude::{
    settings::{Parameters, Prior, Settings},
    *,
};
// use rand::rng;
use rand_distr::{Distribution, Normal}; // to initialize the sde

pub(crate) fn main() {
    let sde = equation::SDE::new(
        |x, p, t, dx, rateiv, cov| {
            fetch_params!(p, fke0, kcp, fkpc, fvol0, _svol);
            fetch_cov!(cov, t, weight, crcl); // CR,AGE,HEIGHT,WEIGHT,SEX_M,CRCL
            dx[3] = -x[3] + fvol0; // + fke0;

            let volume = x[3] * weight; 
            // let clr0 = 0.06*(0.9001446 * crcl + 3.698084) ; // (weight/70.0).powf(0.75) * crcl;
            // let pma = 0.8 + age;
            // let mm50 : f64 = 1.0; // full function is expected at 2-3 YRO
            // let clr = clr0
            //  *(pma.powf(mmpower)/(pma.powf(mmpower) + mm50.powf(mmpower)));
            // let ke = x[3] * clr / volume;
            // let ke = fke0 * clr / volume;

            let clr = crcl // 
              * (weight/70.0 ).powf(0.75); // helped a small bit ... so left in, even though MN already put in crcl
            //  * (pma.powf(2.0) / (pma.powf(2.0) + mm50.powf(2.0)));
            let ke = fke0 * clr / volume;

            // dbg!(x[3], fke0, dx[3]); // data.csv has 0 duration input (always)
            // dx[0] = -ka * x[0]; // vancomycin is an IV drug ... not sure why this is in model?
            dx[1] = rateiv[0] - (ke + kcp) * x[1] + fkpc * kcp * x[2];
            dx[2] = kcp * (x[1] - fkpc * x[2]);
            //
            // The area between the path and the mean implies the bias of the path
            //
            dx[0] = x[3] - fvol0; // with enough samples this goes to zero
            //
            // I don't need to know the probability of each path, 
            //
            dx[4] = (x[3] - fvol0).powf(2.0)/fvol0; // Chi^2

        },
        |p, d| {
            fetch_params!(p, _fke0, _kcp, _fkpc, _fvol0, svol);
            d[3] = svol;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, _fke0, _kcp, _fkpc, fvol0, _svol);
            // let normal = Normal::new(0.0,svol).unwrap();
            /*
            let v = normal.sample(&mut rand::rng());
            x[3] = fvol0 + v;
            */
            x[3] = fvol0; 
        },
        |x, p, t, cov, y| {
            fetch_params!(p, _fke0, _kcp, _fkpc, _fvol0, _svol);
            fetch_cov!(cov, t, weight);
            let volume = x[3] * weight; 
            y[0] = x[1] / volume;
            y[1] = x[0]; // -> 0
            y[2] = x[4]; // \sum Chi^2 // this will be usedin pharmsol likelihood
        },
        (5, 3), // 
        31,
    );

    let _ode = equation::ODE::new(
        |x, p, t, dx, rateiv, cov| {
            fetch_params!(p // , ka
                , fke0, kcp, fkpc, fvol0 // , _mmpower, _svol
              );
            // fetch_cov!(cov, t, scr, weight); // scr, age, height, weight, sex_m
            // let volume: f64 = vol * weight / 70.0;
            // let clr = (weight/70.0).powf(0.75) * (0.3406/scr).powf(1.7748);
            fetch_cov!(cov, t, age, weight, crcl); // cr, age, height, weight, sex_m, crcl
            // let volume: f64 = fvol0 * (11.0 + 0.29 * age + 0.33 * weight) * weight ; // vol * weight / 70.0;
            let volume = fvol0 * weight; 
            let _pma = age; // MN calculated a crcl so, age and weight are already integrated
            let _mm50 : f64 = 0.75; // full functio:q!n is expected at 2-3 YRO
            // let clr0 = mmpower * crcl + svol; // 0.06741 * crcl; // + 0.33456;
            // let clr0 = 0.0624851 * crcl + 0.2528494; // mean optimized in (0.02, 0.12) (0.15, 0.65)
            // let clr0 = 0.05400868 * crcl + 0.221885; // above polynomial was better;
            let clr = crcl // 
              * (weight/70.0 ).powf(0.75); // helped a small bit ... so left in, even though MN already put in crcl
            //  * (pma.powf(2.0) / (pma.powf(2.0) + mm50.powf(2.0)));
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
        .add("fke0", 0.000005, 1.0, false) // (0.0005, 1.5) for no wt in clearance
        .add("kcp", 0.00125, 3.5, false) // (0.5, 5.45) better pop but post degrades a lot
        .add("fkpc", 0.1, 0.9, false)
        .add("fvol0", 0.005, 7.75, false) // (0.075,2.5) better pop but post degrades
        .add("svol", 0.0005, 0.0025, false) // stochastic element on ke w/mean = fke0
        .build()
        .unwrap();

  // .add("mmpower", 0.02, 0.12, false)
  // .add("mmpower", 0.01, 0.2, false)
  // .add("mmpower", 0.002, 0.12, false) // last run
       //  .add("svol", 0.15, 0.65, false) // stochastic element on ke w/mean = fke0

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
    settings.set_error_poly((0.250, 0.0625, 0.0, 0.0)); // ALREADY INCREASED!!! Just run
    settings.set_error_value(1.9437446456398102); // current ODE ... verify that the gamma optimization is off in npag.rs !!!
    // settings.set_error_value(2.5516439936509987);
    // settings.set_error_type(ErrorType::Add);
    settings.set_error_type(ErrorType::Prop);
    settings.set_output_path("examples/vanco_sde/output.tmp_SDE_big_BLQ"); // _lowpred"); // _sde_vol_20250723_fixed_sigma_inside_tf.tmp"); // *** SET OUTPUT DIRECTORY HERE ***
    settings.set_prior(Prior {
        sampler: "sobol".to_string(),
        points: 262144,// 244, // E[t] = 6w(10^5), (2.5^6)
        seed: 347,
        file: Some(String::from("examples/vanco_sde/output.tmp_ODE_big_BLQ/theta_ode2sde.csv")), // None,
    });
    settings.set_output_write(true);
    settings.set_log_level(settings::LogLevel::DEBUG);
    settings.set_idelta(1.0);
    setup_log(&settings).unwrap();

    let data = data::read_pmetrics("examples/vanco_sde/tmp0.csv").unwrap();
    // tmp0.csv").unwrap();

    let mut algorithm = dispatch_algorithm(settings, sde, data).unwrap();

    // TODO_wmy copy main.rs to the output directory before running:
    // cp examples/vanco_sde/main.rs ./examples/vanco_sde/output_sde_vol_20250723_fixed_sigma.tmp 
    // output_sde_vol_20250722.tmp
    // output_ode_vol_20250722.tmp

    algorithm.initialize().unwrap();
    while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
}
