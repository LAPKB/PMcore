use anyhow::Result;
use logger::setup_log;
use pmcore::prelude::*;
use settings::{Parameters, Settings};
// use rand_distr::weighted::WeightedIndex;
use rand_distr::{Distribution, Normal};

fn main() -> Result<()> {
    let eq = equation::ODE::new(
        |x, p, t, dx, rateiv, cov| {
            // automatically defined
            fetch_params!(p, ke0, kcp, kpc, _v0);
            fetch_cov!(cov,t,wt,crcl); // automatically interpolates, so you need t

            let k_e = ke0 * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            // let V = V0 * (wt/70)

            // user defined
            dx[0] = rateiv[0] - (k_e + kcp) * x[0] + kpc * x[1];
            dx[1] = kcp * x[0] - kpc * x[1]
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, x| {
            x[0] = 0.0;
            x[1] = 0.0; // pretty sure these are not necesary, but I like it.
        },
        |x, p, t, cov, y| {
            fetch_params!(p, _ke0, _kcp, _kpc, v0);
            fetch_cov!(cov,t,wt); // , _crcl); // automatically interpolates, so you need t

            // let k_e = ke0 * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            let vol = v0 * (wt/70.0);

            y[0] = x[0]/vol;
        },
        (2, 1),
    );

    let _eq = equation::SDE::new(
        |x, p, t, dx, rateiv, cov| {
            fetch_params!(p, ke0, kcp, kpc, v0, _ske, _svol);
            fetch_cov!(cov,t,wt,crcl);
            dx[0] = ke0 - x[0]; // mean reverting sde
            dx[1] = v0 - x[1];
            let ke = x[0]; // use ke = ke0, if SDE in only on volume.
            let _vol = x[1]* (wt/70.0);
            // let kpc = well * kcp;
            // let norm_wt = wt/70.0;
            // let kel = ke * norm_wt.powf(-0.25) * (0.2145/scr).powf(1.1776);
            let k_e = ke * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            dx[2] = rateiv[0] - ( k_e + kcp) * x[2] + kpc * x[3];
            dx[3] = kcp * x[2] - kpc * x[3];
        },
        |p, d| {
            fetch_params!(p, ke0, _kcp, _kpc, v0, ske, svol);
            d[0] = ske * ke0;
            d[1] = svol * v0;
            // the above increments MUST match the state increments of x
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _kcp, _kpc, v0, ske, svol);
            let normal_ke = Normal::new(ke0, ske*ke0).unwrap();
            x[0] = normal_ke.sample(&mut rand::rng()); // k0 +/- s1
            let normal_v = Normal::new(v0, svol*v0).unwrap();
            x[1] = normal_v.sample(&mut rand::rng()); // v0 +/- s2
            x[2] = 0.0;
            x[3] = 0.0;
        },
        |x, _p, t, cov, y| {
            // fetch_params!(p, _ke0, _kcp, _kpc, v0);
            fetch_cov!(cov,t,wt); // , crcl); // automatically interpolates, so you need t

            // let k_e = ke0 * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            let vol = x[1] * (wt/70.0);

            y[0] = x[2]/vol;
        },
        (4, 1),
        64,
    );

    let mut settings = Settings::new();

    let params = Parameters::builder()
        .add("ke0", 0.0001, 5.0, true)
        .add("kcp", 0.0001, 5.5, true)
        .add("kpc", 0.0001, 11.0, true)
        .add("v0", 0.1, 25.0, true)
        // .add("ske", 0.0001, 0.5, true)
        // .add("svol", 0.0001, 0.5, true) // SDE requires sigmas ... but ODE does not
        .build()
        .unwrap();

    settings.set_parameters(params);
    settings.set_prior_points(16384);
    settings.set_cycles(1000);
    settings.set_error_poly((0.01, 0.1, 0.0, 0.0)); // MN uses 0.1,0.15 ... CV%<0.1 is an acceptiable assay ... so 0, 0.1, ... is probably "right" to use for comparing SDE to ODE solutions
    settings.set_error_type(ErrorType::Add);
    settings.set_error_value(0.16);
    settings.set_idelta(1.0);

    // for ODE use this block:
    // /*
        settings.set_output_path("examples/vpicu_for_grant_prop/output_ode"); // THIS LINE OVERWRITES THIS DIRECTORY !!!
        settings.set_prior_sampler("sobol".to_string());
        settings.set_prior_points(16384);
        settings.set_prior_seed(347);
        // settings.set_prior(settings::Prior {
        //    sampler: "sobol".to_string(),
        //    points: 16384,
        //    seed: 347,
        //    file: None, // Some(String::from("examples/vpicu_for_grant_prop/output_ode/theta.csv")),
        // });
    // */

    // for SDE use this block (Is AG edited to expand only in dimensions of sigma? ___ YES ___):
    /*
        settings.set_output_path("examples/vpicu_for_grant_prop/output_sde_sigma_only"); // THIS LINE OVERWRITES THIS DIRECTORY !!!
        settings.set_prior_file(Some(String::from("examples/vpicu_for_grant_prop/output_sde/theta_w_sigma.csv")));
        //
        // to optimize ONLY the sigmas, edit src/routines/expansion/adaptive_grid.rs to only expand in the dimentions of sigma
        //
    */

    setup_log(&settings)?;
    let data = data::read_pmetrics("examples/vpicu_for_grant_prop/vpicu.csv")?;
    let mut algorithm = dispatch_algorithm(settings, eq, data)?;
    let result = algorithm.fit().unwrap();
    result.write_outputs()?;

    Ok(())
}
