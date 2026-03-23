use anyhow::Result;
use pmcore::mmopt::mmopt;
use pmcore::prelude::*;
use pmcore::structs::theta::Theta;

fn main() -> Result<()> {
    // Define a one-compartment PK model
    let eq = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + b[0];
        },
        |_p, _, _| lag! {},
        |_p, _, _| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    );

    // Population support points representing two distinct PK sub-populations
    let params = Parameters::new().add("ke", 0.1, 1.0).add("v", 30.0, 100.0);

    let mat = faer::Mat::from_fn(2, 2, |r, c| match (r, c) {
        (0, 0) => 0.3,  // ke: slow eliminator
        (0, 1) => 50.0, // v
        (1, 0) => 0.5,  // ke: fast eliminator
        (1, 1) => 60.0, // v
        _ => 0.0,
    });
    let theta = Theta::from_parts(mat, params)?;

    // Error model: additive with SD = 20% of observation
    let errormodel = ErrorModel::additive(ErrorPoly::new(0.0, 0.20, 0.0, 0.0), 0.0);

    // Create a subject with a dose and candidate observation times
    // The observations values are irrelevant — only their times matter
    let subject = Subject::builder("candidate")
        .bolus(0.0, 100.0, 0)
        .missing_observation(0.5, 0)
        .missing_observation(1.0, 0)
        .missing_observation(2.0, 0)
        .missing_observation(4.0, 0)
        .missing_observation(6.0, 0)
        .missing_observation(8.0, 0)
        .missing_observation(12.0, 0)
        .missing_observation(24.0, 0)
        .build();

    // Equal prior weights for both sub-populations
    let weights = vec![0.5, 0.5];

    // Find the optimal 2 sample times (out of 8 candidates)
    println!("Finding optimal 2 sample times from 8 candidates...\n");
    let result = mmopt(
        &theta,
        &subject,
        eq.clone(),
        errormodel.clone(),
        0,
        2,
        weights.clone(),
    )?;
    println!("  {}", result);

    // Compare with 3 samples
    println!("\nFinding optimal 3 sample times...\n");
    let result_3 = mmopt(&theta, &subject, eq, errormodel, 0, 3, weights)?;
    println!("  {}", result_3);

    println!(
        "\nRisk reduction from 2 → 3 samples: {:.2}%",
        (1.0 - result_3.risk / result.risk) * 100.0
    );

    Ok(())
}
