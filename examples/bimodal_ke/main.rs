use logger::setup_log;
use pmcore::prelude::*;
fn main() {
    // let eq = equation::ODE::new(
    //     |x, p, _t, dx, rateiv, _cov| {
    //         // fetch_cov!(cov, t, wt);
    //         fetch_params!(p, ke, _v);
    //         dx[0] = -ke * x[0] + rateiv[0];
    //     },
    //     |_p| lag! {},
    //     |_p| fa! {},
    //     |_p, _t, _cov, _x| {},
    //     |x, p, _t, _cov, y| {
    //         fetch_params!(p, _ke, v);
    //         y[0] = x[0] / v;
    //     },
    //     (1, 1),
    // );
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
    let eq = equation::ODENet::new(
        vec![dmatrix![-1.0], dmatrix![0.0]],
        vec![OutEq::new(0, 0, Op::Div(1))],
        (1, 1),
    );

    let settings = settings::read("examples/bimodal_ke/config.toml").unwrap();
    setup_log(&settings).unwrap();
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    // let result = algorithm.fit().unwrap();
    algorithm.initialize().unwrap();
    while !algorithm.next_cycle().unwrap() {}
    let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
    // println!("{:?}", result);
    // let _result = fit(eq, data, settings);
}
