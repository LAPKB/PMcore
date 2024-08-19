use pmcore::prelude::*;
fn main() {
    let sde = simulator::Equation::new_sde(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, _ske);
            let ke = x[1];
            let ke0 = 0.7;
            let ka = 0.3;
            dx[2] = -ke + ke0;
            // user defined
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p, d| {
            fetch_params!(p, ske);
            d[1] = ske;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, _ske);
            x[1] = 0.7
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ske);
            y[0] = x[1] / 50.0;
        },
        (3, 1),
    );

    // let ode = simulator::Equation::new_ode(
    //     |x, p, _t, dx, _rateiv, _cov| {
    //         // automatically defined
    //         fetch_params!(p, ke);
    //         // user defined
    //         dx[0] = -ke * x[0];
    //     },
    //     |_p| lag! {},
    //     |_p| fa! {},
    //     |_p, _t, _cov, _x| {},
    //     |x, _p, _t, _cov, y| {
    //         // fetch_params!(p, _ke0, v);
    //         y[0] = x[0] / 50.0;
    //     },
    //     (1, 1),
    // );

    let settings = settings::read("examples/iov/config.toml".to_string()).unwrap();
    let data = data::read_pmetrics("examples/iov/test.csv").unwrap();
    let _result = fit(sde, data, settings);
    // let _result = fit(ode, data, settings);
}
