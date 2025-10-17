use pmcore::prelude::*;

pub(crate) fn main() -> Result<()> {
    let sde = equation::SDE::new(
        |x, p, _t, dx, rateiv, cov| {
            fetch_params!(p, ke0, _v0, kcp, well);
            fetch_cov!(cov, _t, scr, wt);
            dx[0] = ke0 - x[0]; // x[0] moves toward ke0
            dx[1] = 0.0; // v0 - x[1];
            let ke = x[0];
            let vol = x[1];
            let norm_wt = wt / 70.0;
            let a_ke = norm_wt.powf(0.75) * (0.4385789 / scr).powf(0.873566) / (norm_wt * vol);
            dx[2] = rateiv[0] - (a_ke * ke + kcp) * x[2] + (well * kcp) * x[3];
            dx[3] = kcp * x[2] - (well * kcp) * x[3];
        },
        |p, d| {
            fetch_params!(p, _ke0, _vol, _well, ske, _svol);
            d[0] = ske;
            //d[1] = svol;
            // the above increments MUST match the state increments of x
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
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
            let norm_wt = wt / 70.0;
            let vol = x[1];
            y[0] = x[2] / (vol * norm_wt);
        },
        (4, 1),
        17,
    );

    let _ode = equation::ODE::new(
        |x, p, _t, dx, rateiv, cov| {
            fetch_params!(p, ke0, v0, kcp, well);
            fetch_cov!(cov, _t, scr, wt);
            let norm_wt = wt / 70.0;
            dx[0] = rateiv[0]
                - ((norm_wt.powf(0.75) * (0.4385789 / scr).powf(0.873566) / (norm_wt * v0)) * ke0
                    + kcp)
                    * x[0];
            dx[1] = kcp * x[0] - (well * kcp) * x[1]
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_params!(p, _ke0, v0, _kcp, _well);
            fetch_cov!(cov, t, wt);
            y[0] = x[0] / (v0 * wt / 70.0);
        },
        (2, 1),
    );

    let params = Parameters::new()
        .add("ke0", 1.00, 7.25)
        .add("v0", 25.000, 175.0)
        .add("kcp", 0.0001, 0.1)
        .add("well", 0.125, 16.00)
        .add("ske", 1e-7, 0.0001);

    let em = ErrorModel::proportional(ErrorPoly::new(0.05, 0.035, 0.0, 0.0), 10.0, None);
    let ems = ErrorModels::new().add(0, em).unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(usize::MAX);
    settings.set_cache(true);
    settings.set_prior(Prior::File(
        "examples/vanco_sde_002/output_ode/theta_ske.csv".to_string(),
    ));
    settings.set_output_path("examples/vanco_sde_002/output_sde_ske_tmp");
    settings.set_write_logs(true);
    settings.set_log_level(LogLevel::DEBUG);
    settings.initialize_logs()?;

    let data = data::read_pmetrics("examples/vanco_sde_002/vclean.csv")?;
    let mut algorithm = dispatch_algorithm(settings, sde, data)?;
    algorithm.initialize()?;
    while !algorithm.next_cycle()? {}
    let result = algorithm.into_npresult();
    result.write_outputs()?;

    Ok(())
}
