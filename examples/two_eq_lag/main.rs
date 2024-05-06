#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
use pmcore::{
    prelude::*,
    simulator::{analytical::one_compartment_with_absorption, Equation},
};

fn main() -> Result<()> {
    // let eq = Equation::new_ode_solvers(
    //     |x, p, _t, dx, rateiv, _cov| {
    //         fetch_params!(p, ke, ka, _v, _tlag);
    //         dx[0] = -ka * x[0];
    //         dx[1] = ka * x[0] - ke * x[1];
    //     },
    //     |p| {
    //         fetch_params!(p, _ke, _ka, _v, tlag);
    //         lag! {0=>tlag}
    //     },
    //     |_p| fa! {},
    //     |_p, _t, _cov, _x| {},
    //     |x, p, _t, _cov, y| {
    //         fetch_params!(p, _ke, v);
    //         y[0] = x[1] / v;
    //     },
    // );

    let eq = Equation::new_analytical(
        one_compartment_with_absorption,
        |_p, _cov| {},
        |p| {
            fetch_params!(p, _ke, _ka, _v, tlag);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, _ka, v, _tlag);
            y[0] = x[1] / v;
        },
    );
    let _result = start(eq, "examples/two_eq_lag/config.toml".to_string())?;

    Ok(())
}
