use anyhow::Ok;
use pmcore::prelude::*;
fn main() -> Result<()> {
    let sde = equation::SDE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0);
            // let ke0 = 1.2;
            dx[1] = -x[1] + ke0;
            let ke = x[1];
            // user defined
            dx[0] = -ke * x[0];
        },
        |p, d| {
            fetch_params!(p, _ke0);
            d[1] = 0.1;
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0);
            x[1] = ke0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0);
            y[0] = x[0] / 50.0;
        },
        (2, 1),
        10000,
    );

    let params = Parameters::new().add("ke0", 0.001, 2.0);

    let em = ErrorModel::additive(ErrorPoly::new(0.0, 0.0, 0.0, 0.0), 0.0000757575757576, None);
    let ems = ErrorModels::new().add(0, em).unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(100000);
    settings.set_prior(Prior::sobol(100, 347));
    settings.set_output_path("examples/iov/output");
    settings.set_write_logs(true);
    settings.initialize_logs()?;

    let data = data::read_pmetrics("examples/iov/test.csv")?;
    let mut algorithm = dispatch_algorithm(settings, sde, data)?;
    algorithm.initialize()?;
    while !algorithm.next_cycle()? {}
    let result = algorithm.into_npresult();
    result.write_outputs()?;

    Ok(())
}
