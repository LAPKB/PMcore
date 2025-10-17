// use ndarray::array;
use pmcore::prelude::*;

fn main() {
    let _eq = equation::ODE::new(
        |x, p, _t, dx, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    //this exampled used to call OSAT to estimate_theta
    // let data = read_pmetrics("examples/bimodal_ke/bimodal_ke.csv").unwrap();
    // let theta = data.estimate_theta(&eq, &array![1.5, 125.0]);
    // dbg!(theta);
}
