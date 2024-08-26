use pmcore::prelude::*;
fn main() {
    let sde = simulator::Equation::new_sde(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0);
            if ke0 < 0.0 {
                panic!("ke0 must be positive");
            }
            dx[1] = -x[1] + ke0;
            let ke = x[1];
            // user defined
            dx[0] = -ke * x[0];
        },
        |p, d| {
            fetch_params!(p, _ke0);
            d[1] = 0.1;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0);
            x[1] = ke0
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0);
            y[0] = x[0] / 50.0;
        },
        (2, 1),
    );

    let settings = settings::read("examples/iov/config.toml".to_string()).unwrap();
    let data = data::read_pmetrics("examples/iov/test.csv").unwrap();
    let _result = fit(sde, data, settings);
    // let _result = fit(ode, data, settings);
}
