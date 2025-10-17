use pmcore::prelude::*;

pub(crate) fn main() -> Result<()> {
    let sde = equation::SDE::new(
        |x, p, _t, dx, rateiv, cov| {
            fetch_params!(p, v0, ke0, kcp, well, _ske);
            fetch_cov!(cov, _t, scr, wt);
            dx[0] = ke0 - x[0]; // mean reverting sde
                                // dx[4] = prob.(ke0 - x[0]).log; // sum of log(p(particle)*dt) = log pr of trajectory, I think????
                                // at t_f the particle filter removes some trials, the remaining trials ar either high
                                // probability (large X[4] or low probability low X[4]) ... low means IOV has occured during the
                                // event-event interval. Important for simulation to future. because future simulation requiers
                                // the support point, defining the intial state, changes.
            dx[1] = v0 - x[1];
            let ke = x[0]; // use ke = ke0, if SDE in only on volume.
            let _vol = x[1];
            let kpc = well * kcp;
            let norm_wt = wt / 70.0;
            let kel = ke * norm_wt.powf(-0.25) * (0.2145 / scr).powf(1.1776);
            dx[2] = rateiv[0] - (kel + kcp) * x[2] + kpc * x[3];
            dx[3] = kcp * x[2] - kpc * x[3];
        },
        |p, d| {
            // fetch_params!(p, _v0, _ke0, _kcp, _well, ske, _svol);
            fetch_params!(p, _v0, _ke0, _kcp, _well, ske, svol);
            d[0] = ske;
            d[1] = svol;
            // the above increments MUST match the state increments of x
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, v0, ke0, _kcp, _well, _ske, _svol);
            x[0] = ke0;
            x[1] = v0;
            x[2] = 0.0;
            x[3] = 0.0;
        },
        |x, _p, t, cov, y| {
            // fetch_params!(p, v0, _ke0, _kcp, _well, _ske, _svol);
            // fetch_params!(p, v0, _ke0, _kcp, _well, _ske); // macro that expands into an index into an array, i.e. let <name> = p[index]
            //fetch_cov!(cov, t, _scr,wt,_ht,_male);
            fetch_cov!(cov, t, wt); // this is a hash map ... let wt = cov.get("wt") ... NOT AN INDEX!
            let norm_wt = wt / 70.0;
            let vol = x[1];
            y[0] = x[2] / (vol * norm_wt);
        },
        (4, 1),
        100,
    );

    let _ode = equation::ODE::new(
        |x, p, _t, dx, rateiv, cov| {
            fetch_params!(p, _v0, ke0, kcp, well);
            fetch_cov!(cov, _t, scr, wt);
            let norm_wt = wt / 70.0;
            let kel = ke0 * norm_wt.powf(-0.25) * (0.2145 / scr).powf(1.1776);
            let kpc = well * kcp;
            dx[0] = rateiv[0] - (kel + kcp) * x[0] + kpc * x[1];
            dx[1] = kcp * x[0] - kpc * x[1];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_params!(p, v0, _ke0, _kcp, _well);
            fetch_cov!(cov, t, wt);
            let norm_wt = wt / 70.0;
            y[0] = x[0] / (v0 * norm_wt);
        },
        (2, 1),
    );

    let params = Parameters::new()
        .add("v0", 15.0, 200.0)
        .add("ke0", 0.0004, 0.9)
        .add("kcp", 0.00001, 0.15)
        .add("well", 0.00025, 20.00)
        .add("ske", 1e-27, 0.4)
        .add("svol", 1e-27, 1e-25);

    let em = ErrorModel::proportional(ErrorPoly::new(0.15, 0.075, 0.00001, 0.0), 2.803022, None);
    let ems = ErrorModels::new().add(0, em).unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(usize::MAX);
    settings.set_cache(true);
    settings.set_prior(Prior::File(
        "examples/vanco_sde_003/output_ode_sobol/theta_sde0.csv".to_string(),
    ));
    settings.set_output_path("examples/vanco_sde_003/output_sde_ske0pt2");
    settings.set_write_logs(true);
    settings.set_log_level(LogLevel::DEBUG);
    settings.initialize_logs()?;

    let data = data::read_pmetrics("examples/vanco_sde_003/vclean.csv")?;
    let mut algorithm = dispatch_algorithm(settings, sde, data)?;
    algorithm.initialize()?;
    while !algorithm.next_cycle()? {}
    let result = algorithm.into_npresult();
    result.write_outputs()?;

    Ok(())
}
