use pmcore::prelude::*;
fn main() {
    let sde = simulator::Equation::new_sde(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ka, ke0, _ske, kcp, kpc);
            let ke = x[3];
            dx[3] = -ke + ke0;
            // user defined
            dx[0] = -ka * x[0];
            dx[1] = ka * x[0] - ke * x[1] - kcp * x[1] + kpc * x[2];
            dx[2] = kcp * x[1] - kpc * x[2];
        },
        |p, d| {
            fetch_params!(p, _ka, _ke0, ske, _kcp, _kpc);
            d[3] = ske;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, _ka, ke0, _ske, _kcp, _kpc);
            x[3] = ke0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke0, _ske, _kcp, _kpc);
            y[0] = x[1] / 1.0;
        },
        (4, 1),
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
    let data = data::read_pmetrics("examples/iov/sde_data.csv").unwrap();
    let _result = fit(sde, data, settings);
    // let _result = fit(ode, data, settings);
}
