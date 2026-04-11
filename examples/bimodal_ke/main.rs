use anyhow::Result;
use pmcore::prelude::*;

fn main() -> Result<()> {
    let eq = ode! {
        diffeq: |x, p, _t, dx, b, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[1] + b[1];
        },
        out: |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[1] = x[0] / v;
        },
    }
    .with_solver(OdeSolver::ExplicitRk(ExplicitRkTableau::Tsit45));

    let params = Parameters::new()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0);

    let ems = AssayErrorModels::new().add(
        1,
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
    )?;

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(1000);
    settings.set_prior(Prior::sobol(2028, 22));
    settings.set_output_path("examples/bimodal_ke/output/");
    settings.set_write_logs(true);

    settings.write()?;

    // settings.enable_logs(stdout: bool, )
    settings.initialize_logs()?;
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;
    let mut algorithm = dispatch_algorithm(settings, eq, data)?;
    let mut result = algorithm.fit()?;
    result.write_outputs()?;

    Ok(())
}
