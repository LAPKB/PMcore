use anyhow::Result;
use logger::setup_log;
use pmcore::prelude::*;
use settings::{Parameters, Settings};
fn main() -> Result<()> {
    let _eq = equation::ODE::new(
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
            fetch_cov!(cov,t,wt, crcl); // automatically interpolates, so you need t

            // let k_e = ke0 * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            let vol = v0 * (wt/70.0);

            y[0] = x[0]/vol;
        },
        (2, 1), // extra output equations are used in SDE to collect statistics
    );

    let eq = equation::SDE::new(
        |x, p, t, dx, rateiv, cov| {
            fetch_params!(p, ke0, kcp, kpc, v0, ske, svol);
            fetch_cov!(cov,t,wt,crcl);
            dx[0] = ke0 - x[0]; // mean reverting sde
            dx[1] = v0 - x[1];
            let ke = x[0]; // use ke = ke0, if SDE in only on volume.
            let _vol = x[1];
            // let kpc = well * kcp;
            // let norm_wt = wt/70.0;
            // let kel = ke * norm_wt.powf(-0.25) * (0.2145/scr).powf(1.1776);
            let k_e = ke0 * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            dx[2] = rateiv[0] - ( k_e + kcp) * x[2] + kpc * x[3];
            dx[3] = kcp * x[2] - kpc * x[3];
        },
        |p, d| {
            fetch_params!(p, _ke0, _kcp, _kpc, _v0, ske, svol);
            d[0] = ske;
            d[1] = svol;
            // the above increments MUST match the state increments of x
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _kcp, _kpc, v0, _ske, _svol);
            x[0] = ke0;
            x[1] = v0;
            x[2] = 0.0;
            x[3] = 0.0;
        },
        |x, p, t, cov, y| {
            fetch_params!(p, _ke0, _kcp, _kpc, v0);
            fetch_cov!(cov,t,wt, crcl); // automatically interpolates, so you need t

            // let k_e = ke0 * (wt/70.0).powf(-0.25) * (crcl/120.0); 
            let vol = v0 * (wt/70.0);

            y[0] = x[2]/vol;
        },
        (4, 1),
        100,
    );

    let mut settings = Settings::new();

    let params = Parameters::builder()
        .add("ke0", 0.0001, 5.0, true)
        .add("kcp", 0.0001, 4.0, true)
        .add("kpc", 0.0001, 10.0, true)
        .add("v0", 0.1, 12.0, true)
        .add("ske", 0.0001, 1.0, true)
        .add("svol", 0.0001, 1.0, true)
        .build()
        .unwrap();

    settings.set_parameters(params);
    settings.set_prior_points(4097);
    settings.set_cycles(1000);
    settings.set_error_poly((0.1, 0.15, 0.0, 0.0));
    settings.set_error_type(ErrorType::Add);
    settings.set_error_value(2.0);
    settings.set_output_path("examples/vpicu_for_grant_prop/output_sde");
    settings.set_idelta(1.0);

    setup_log(&settings)?;
    let data = data::read_pmetrics("examples/vpicu_for_grant_prop/vpicu.csv")?;
    let mut algorithm = dispatch_algorithm(settings, eq, data)?;
    let result = algorithm.fit().unwrap();
    result.write_outputs()?;

    Ok(())
}
