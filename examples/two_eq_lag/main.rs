#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use core::panic;
use std::path::Path;

use data::read_pmetrics;
use logger::setup_log;
use ndarray::Array2;
use pmcore::prelude::{models::one_compartment_with_absorption, simulator::Equation, *};
use settings::Parameters;
use settings::Settings;

fn main() {
    let eq = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_cov!(cov, t,);
            fetch_params!(p, ka, ke, tlag, v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p| {
            fetch_params!(p, ka, ke, tlag, v);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, ka, ke, tlag, v);
            y[0] = x[1] / v;
        },
        (2, 1),
    );
    // let eq = Equation::new_analytical(
    //     one_compartment_with_absorption,
    //     |_p, _cov| {},
    //     |p| {
    //         fetch_params!(p, _ka, _ke, tlag, _v);
    //         lag! {0=>tlag}
    //     },
    //     |_p| fa! {},
    //     |_p, _t, _cov, _x| {},
    //     |x, p, _t, _cov, y| {
    //         fetch_params!(p, _ka, _ke, _tlag, v);
    //         y[0] = x[1] / v;
    //     },
    //     (2, 1),
    // );
    // let eq = equation::ODENet::new(
    //     vec![
    //         dmatrix![
    //             -1.0,0.;
    //             1.,0.
    //         ],
    //         dmatrix![
    //             0.,0.;
    //             0.,-1.
    //         ],
    //         dmatrix![
    //             0.0,0.0;
    //             0.0,0.0
    //         ],
    //         dmatrix![
    //             0.0,0.0;
    //             0.0,0.0
    //         ],
    //     ],
    //     vec![],
    //     vec![],
    //     vec![Lag::new(0, Op::Equal(P(2)))],
    //     vec![],
    //     vec![],
    //     vec![OutEq::new(0, Op::Div(X(1), P(3)))],
    //     (2, 1),
    // );

    let mut settings = Settings::new();

    let parameters = Parameters::new()
        .add("ka", 0.1, 0.3, false)
        .unwrap()
        .add("ke", 0.001, 0.1, false)
        .unwrap()
        .add("tlag", 0.0, 4.00, false)
        .unwrap()
        .add("v", 30.0, 120.0, false)
        .unwrap();

    settings.set_parameters(parameters);

    settings.set_gamlam(5.0);
    settings.set_error_poly((0.02, 0.05, -2e-04, 0.0));
    settings.set_prior_points(2129);
    settings.set_cycles(1000);

    setup_log(&settings).unwrap();
    let data = data::read_pmetrics("examples/two_eq_lag/two_eq_lag.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    let result = algorithm.fit().unwrap();
    // algorithm.initialize().unwrap();
    // while !algorithm.next_cycle().unwrap() {}
    // let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
}
