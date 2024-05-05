use pmcore::{
    prelude::*,
    simulator::{analytical::one_compartment, Equation, V},
};

fn main() -> Result<()> {
    let eq = Equation::new_ode_solvers(
        |x, p, _t, dx, rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0];
        },
        |_p| lag! {},
        |_p| fa! {},
        |_p, _t, _cov| V::from_vec(vec![0.0, 0.0]),
        |x, p, _t, _cov| {
            fetch_params!(p, _ke, v);
            V::from_vec(vec![x[0] / v, 0.0])
        },
    );
    // let eq = Equation::new_analytical(
    //     one_compartment,
    //     |_p, _cov| {},
    //     |_p| lag! {},
    //     |_p| fa! {},
    //     |_p, _t, _cov| V::from_vec(vec![0.0]),
    //     |x, p, _t, _cov| {
    //         fetch_params!(p, _ke, v);
    //         V::from_vec(vec![x[0] / v])
    //     },
    // );
    let _result = start(eq, "examples/bimodal_ke/config.toml".to_string())?;
    Ok(())
}
