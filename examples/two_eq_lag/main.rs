#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use std::path::Path;

use data::read_pmetrics;
use pmcore::prelude::{models::one_compartment_with_absorption, simulator::Equation, *};

fn main() {
    let eq = Equation::new_ode(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_cov!(cov, t,);
            fetch_params!(p, ka, ke, _tlag, _v);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
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

    let settings = read_settings("examples/two_eq_lag/config.toml".to_string()).unwrap();
    let data = read_pmetrics(Path::new("examples/two_eq_lag/two_eq_lag.csv")).unwrap();
    let _result = fit(eq, data, settings);
}
