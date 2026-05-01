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
        10000,
    )
    .with_nstates(2)
    .with_ndrugs(1)
    .with_nout(1)
    .with_metadata(
        equation::metadata::new("iov")
            .parameters(["ke0"])
            .states(["central", "ke_latent"])
            .outputs(["1"])
            .route(equation::Route::bolus("1").to_state("central"))
            .particles(10000),
    )
    .unwrap();

    let data = data::read_pmetrics("examples/iov/test.csv").unwrap();
    EstimationProblem::builder(sde, data)
        .parameter(Parameter::bounded("ke0", 0.001, 2.0))?
        .method(Npag::new())
        .error(
            "1",
            AssayErrorModel::additive(ErrorPoly::new(0.0, 0.0, 0.0, 0.0), 0.0000757575757576),
        )?
        .output_dir("examples/iov/output")
        .cycles(100000)
        .prior(Prior::sobol(100, 347))
        .fit()
        .unwrap();

    Ok(())
}
