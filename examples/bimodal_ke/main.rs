use pmcore::{
    prelude::*,
    simulator::{analytical::one_compartment, Equation},
};

fn main() -> Result<()> {
    let method = "analytical".to_string();

    let eq = match method.as_str() {
        "ode" => {
            Equation::new_ode(
                |x, p, _t, dx, rateiv, _cov| {
                    // fetch_cov!(cov, t, wt);
                    fetch_params!(p, ke, _v);
                    dx[0] = -ke * x[0] + rateiv[0];
                },
                |_p| lag! {},
                |_p| fa! {},
                |_p, _t, _cov, _x| {},
                |x, p, _t, _cov, y| {
                    fetch_params!(p, _ke, v);
                    y[0] = x[0] / v;
                },
                (1, 1),
            )
        }
        "analytical" => Equation::new_analytical(
            one_compartment,
            |p, _cov| {},
            |_p| lag! {},
            |_p| fa! {},
            |_p, _t, _cov, _x| {},
            |x, p, _t, _cov, y| {
                fetch_params!(p, _ke, v);
                y[0] = x[0] / v;
            },
            (1, 1),
        ),
        _ => panic!("Method not found"),
    };

    let _result = start(eq, "examples/bimodal_ke/config.toml".to_string())?;
    Ok(())
}
