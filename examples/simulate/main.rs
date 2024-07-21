use pmcore::prelude::{simulator::Equation, *};

fn main() {
    let eq = Equation::new_ode(
        |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, _v, ke);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, v, _ke);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    let settings = read_settings("examples/simulate/config.toml".to_string()).unwrap();
    let _result = simulate(eq, settings);
}
