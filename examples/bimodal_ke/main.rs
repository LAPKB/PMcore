use anyhow::Result;
use logger::setup_log;
use pmcore::prelude::*;

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

    let params = Parameters::new()
        .add("ke", 0.001, 3.0, false)
        .add("v", 25.0, 250.0, false);

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_model(ErrorModel::Additive, 0.0, (0.0, 0.5, 0.0, 0.0))
        .build();

    settings.set_cycles(1000);
    settings.enable_output_files("examples/bimodal_ke/output/");

    setup_log(&settings)?;
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;
    let mut algorithm = dispatch_algorithm(settings, eq, data)?;
    let result = algorithm.fit()?;
    result.write_outputs()?;

    Ok(())
}
