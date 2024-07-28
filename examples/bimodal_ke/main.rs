use pmcore::prelude::{models::one_compartment, simulator::Equation, *};

fn main() {
    let method = "ode".to_string();

    let eq = match method.as_str() {
        "ode" => {
            Equation::new_ode(
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
            )
        }
        "analytical" => Equation::new_analytical(
            one_compartment,
            |_p, _t, _cov| {},
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, v, _ke);
                y[0] = x[0] / v;
            },
            (1, 1),
        ),
        _ => panic!("Method not found"),
    };

    let settings = settings::read("examples/bimodal_ke/config.toml").unwrap();
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv").unwrap();
    let _result = fit(eq, data, settings);
}
