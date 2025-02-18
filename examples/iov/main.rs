use anyhow::Ok;
use pmcore::prelude::{
    settings::{Parameters, Prior, Settings},
    *,
};
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
        |_p| lag! {},
        |_p| fa! {},
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

    let mut settings = Settings::new();

    let params = Parameters::builder()
        .add("ke0", 0.001, 2.0, false)
        .build()
        .unwrap();
    settings.set_parameters(params);
    settings.set_cycles(100000);
    settings.set_error_poly((0.0, 0.0, 0.0, 0.0));
    settings.set_error_type(ErrorType::Add);
    settings.set_error_value(0.0000757575757576);
    settings.set_output_path("examples/iov/output");
    settings.set_prior(Prior {
        sampler: "sobol".to_string(),
        points: 100,
        seed: 347,
        file: None,
    });
    settings.set_output_write(true);

    setup_log(&settings)?;

    let data = data::read_pmetrics("examples/iov/test.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, sde, data).unwrap();
    algorithm.initialize().unwrap();
    while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();

    Ok(())
}
