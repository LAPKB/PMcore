use anyhow::Result;
use logger::setup_log;
use pmcore::prelude::*;
use settings::Parameters;
fn main() -> Result<()> {
    let eq = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );
    // let eq = equation::Analytical::new(
    //     one_compartment,
    //     |_p, _t, _cov| {},
    //     |_p| lag! {},
    //     |_p| fa! {},
    //     |_p, _t, _cov, _x| {},
    //     |x, p, _t, _cov, y| {
    //         fetch_params!(p, _ke, v);
    //         y[0] = x[0] / v;
    //     },
    //     (1, 1),
    // );
    // let eq = equation::ODENet::new(
    //     vec![dmatrix![-1.0], dmatrix![0.0]],
    //     vec![],
    //     vec![],
    //     vec![],
    //     vec![],
    //     vec![],
    //     vec![OutEq::new(0, Div(X(0), P(1)))],
    //     (1, 1),
    // );

    let mut settings = settings::read("examples/bimodal_ke/config.toml").unwrap();
    let parameters = Parameters::new()
        .add("ke", 0.001, 3.0, false)?
        .add("v", 25.0, 250.0, false)?
        .to_owned();
    settings.parameters = parameters;
    setup_log(&settings).unwrap();
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    let result = algorithm.fit().unwrap();
    // algorithm.initialize().unwrap();
    // while !algorithm.next_cycle().unwrap() {}
    // let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
    // println!("{:?}", result);
    // let _result = fit(eq, data, settings);
    Ok(())
}
