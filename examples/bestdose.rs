use anyhow::Result;
use pmcore::bestdose::{optimize_dose, DoseOptimizer};
use pmcore::prelude::*;
use pmcore::routines::initialization::sobol::generate;

fn main() -> Result<()> {
    // Example model
    let eq = equation::ODE::new(
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
    );

    // Example Theta
    let params = Parameters::new()
        .add("ke", 0.001, 3.0, false)
        .add("v", 25.0, 250.0, false);

    let theta = generate(&params, 24, 22)?;

    // Some observed data
    let subject = Subject::builder("Nikola Tesla")
        .bolus(0.0, 20.0, 0)
        .observation(12.0, 8.0, 0)
        .build();

    // Example usage
    let data = Data::new(vec![subject]); // Placeholder for actual data
    let theta = theta;
    let target_concentration = 10.0;
    let target_time = 5.0;
    let eq = eq;
    let min_dose = 0.0;
    let max_dose = 100.0;

    let problem = DoseOptimizer {
        data,
        theta,
        target_concentration,
        target_time,
        eq,
        min_dose,
        max_dose,
    };

    optimize_dose(problem)?;

    Ok(())
}
