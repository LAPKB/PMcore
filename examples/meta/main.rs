#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use pmcore::{prelude::*, routines::settings};

fn main() {
    let eq = equation::ODE::new(
        |x, p, t, dx, b, rateiv, cov| {
            fetch_cov!(cov, t, wt, pkvisit);
            fetch_params!(p, cls, fm, k20, relv, theta1, theta2, vs);
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let ke = cl / v;
            let v2 = relv * v;
            dx[0] = rateiv[0] - ke * x[0] * (1.0 - fm) - fm * x[0] + b[0];
            dx[1] = fm * x[0] - k20 * x[1];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_cov!(cov, t, wt, pkvisit);
            fetch_params!(p, cls, fm, k20, relv, theta1, theta2, vs);
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let ke = cl / v;
            let v2 = relv * v;
            y[0] = x[0] / v;
            y[1] = x[1] / v2;
        },
        (2, 2),
    );

    let params = Parameters::new()
        .add("cls", 0.1, 10.0)
        .add("fm", 0.0, 1.0)
        .add("k20", 0.01, 1.0)
        .add("relv", 0.1, 1.0)
        .add("theta1", 0.1, 10.0)
        .add("theta2", 0.1, 10.0)
        .add("vs", 1.0, 10.0);

    let ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )
        .unwrap()
        .add(
            1,
            ErrorModel::proportional(ErrorPoly::new(1.0, 0.1, 0.0, 0.0), 5.0),
        )
        .unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPOD)
        .set_parameters(params)
        .set_error_models(ems)
        .build();
    settings.set_cycles(10000);

    settings.initialize_logs().unwrap();
    let data = data::read_pmetrics("examples/meta/meta.csv").unwrap();
    let mut algorithm = dispatch_algorithm(settings, eq, data).unwrap();
    // let result = algorithm.fit().unwrap();
    algorithm.initialize().unwrap();
    let mut result = algorithm.fit().unwrap();
    result.write_outputs().unwrap();
}
