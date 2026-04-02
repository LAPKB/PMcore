#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use pmcore::prelude::*;

fn main() {
    let eq = ode! {
        diffeq: |x, p, _t, dx, b, rateiv, _cov| {
            fetch_cov!(cov, t,);
            fetch_params!(p, ka, ke);
            dx[0] = -ka * x[0] + b[1];
            dx[1] = ka * x[0] - ke * x[1];
        },
        lag: |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {1=>tlag}
        },
        out: |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            y[1] = x[1] / v;
        },
    };
    // let eq = Equation::new_analytical(
    //     one_compartment_with_absorption,
    //     |_p, _cov| {},
    //     |p| {
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

    let observations = ObservationSpec::new()
        .add_channel(ObservationChannel::continuous(1, "cp"))
        .with_assay_error_models(
            AssayErrorModels::new()
                .add(
                    1,
                    AssayErrorModel::additive(
                        ErrorPoly::new(-0.00119, 0.44379, -0.45864, 0.16537),
                        0.0,
                    ),
                )
                .unwrap(),
        );

    let model = ModelDefinition::builder(eq)
        .parameters(
            ParameterSpace::new()
                .add(ParameterSpec::bounded("ka", 0.1, 0.9))
                .add(ParameterSpec::bounded("ke", 0.001, 0.1))
                .add(ParameterSpec::bounded("tlag", 0.0, 4.0))
                .add(ParameterSpec::bounded("v", 30.0, 120.0)),
        )
        .observations(observations)
        .build()
        .unwrap();

    let data = data::read_pmetrics("examples/two_eq_lag/two_eq_lag.csv").unwrap();
    let mut result = EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(NonparametricMethod::Npag(
            NpagOptions::default(),
        )))
        .run()
        .unwrap();
    result.write_outputs().unwrap();
}
