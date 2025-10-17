#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use pmcore::prelude::*;

fn main() {
    let eq = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_cov!(cov, t,);
            fetch_params!(p, ka, ke);
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0=>tlag}
        },
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            y[0] = x[1] / v;
        },
        (3, 1),
    );
    // let eq = Equation::new_analytical(
    //     one_compartment_with_absorption,
    //     |_p, _cov| {},
    //     |p, _t, _cov| {
    //         fetch_params!(p, _ka, _ke, tlag, _v);
    //         lag! {0=>tlag}
    //     },
    //     |_p, _t, _cov| fa! {},
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

    let params = Parameters::new()
        .add("ka", 0.1, 0.9)
        .add("ke", 0.001, 0.1)
        .add("tlag", 0.0, 4.0)
        .add("v", 30.0, 120.0);

    let em = ErrorModel::proportional(ErrorPoly::new(0.02, 0.05, -2e-04, 0.0), 5.0, None);
    let ems = ErrorModels::new().add(0, em).unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(1000);
    settings.set_prior(Prior::sobol(2129, 347));
    settings.set_cache(true);
    settings.set_write_logs(true);
    settings.initialize_logs().unwrap();

    let data = data::read_pmetrics("examples/two_eq_lag/two_eq_lag.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    let result = algorithm.fit().unwrap();
    // algorithm.initialize().unwrap();
    // while !algorithm.next_cycle().unwrap() {}
    // let result = algorithm.into_npresult();
    result.write_outputs().unwrap();
}
