use pmcore::prelude::*;
fn main() {
    let sde = simulator::Equation::new_sde(
        |x, p, _t, dx, _rateiv, _cov| {
            // automatically defined
            fetch_params!(p, ke0, _v);
            let ke = x[1];
            dx[1] = -ke + ke0;
            // user defined
            dx[0] = -ke * x[0];
        },
        |p, d| {
            fetch_params!(p, _ke0, _v);
            d[1] = 0.1;
        },
        |_p| lag! {},
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(p, ke0, _v);
            x[1] = ke0
        },
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke0, v);
            y[0] = x[0] / v;
        },
        (2, 1),
    );

    let settings = settings::read("examples/iov/config.toml".to_string()).unwrap();
    let data = data::read_pmetrics("examples/iov/data.csv").unwrap();
    let _result = fit(sde, data, settings);
}
