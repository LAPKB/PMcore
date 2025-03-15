use anyhow::Result;
use logger::setup_log;
use pmcore::prelude::*;
use settings::{Parameters, Settings};
fn main() -> Result<()> {
    let eq = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    let mut settings = Settings::new();

    let params = Parameters::builder()
        .add("ke", 0.001, 3.0, true)
        .add("v", 25.0, 250.0, true)
        .build()
        .unwrap();

    settings.set_parameters(params);
    settings.set_cycles(1000);
    settings.set_error_poly((0.0, 0.5, 0.0, 0.0));
    settings.set_error_type(ErrorType::Add);
    settings.set_output_path("examples/bimodal_ke/output");

    setup_log(&settings)?;
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;
    let mut algorithm = dispatch_algorithm(settings, eq, data)?;
    let result = algorithm.fit()?;
    result.write_outputs()?;

    Ok(())
}
