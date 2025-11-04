use anyhow::Result;
use pmcore::bestdose::{BestDoseProblem, DoseRange, Target};
use pmcore::prelude::*;
use pmcore::routines::initialization::parse_prior;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("BestDose with Dose Range Bounds - Example\n");
    println!("==========================================\n");

    // Simple one-compartment PK model
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

    let params = Parameters::new()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0);

    let ems = ErrorModels::new().add(
        0,
        ErrorModel::additive(ErrorPoly::new(0.0, 0.20, 0.0, 0.0), 0.0),
    )?;

    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems.clone())
        .build();

    settings.disable_output();

    // Load realistic prior from previous NPAG run
    println!("Loading prior from bimodal_ke example...");
    let (theta, prior) = parse_prior(
        &"examples/bimodal_ke/output/theta.csv".to_string(),
        &settings,
    )?;
    let weights = prior.as_ref().unwrap();

    println!("Prior: {} support points\n", theta.matrix().nrows());

    // Create a target requiring high dose
    println!("Target: Achieve 15 mg/L at 2h (requires high dose)");

    let target_data = Subject::builder("Target")
        .bolus(0.0, 0.0, 0) // Dose to be optimized
        .observation(2.0, 15.0, 0) // High target concentration
        .build();

    // Test with different dose ranges
    let dose_ranges = vec![
        (50.0, 200.0, "Narrow range (50-200 mg)"),
        (50.0, 500.0, "Medium range (50-500 mg)"),
        (50.0, 2000.0, "Wide range (50-2000 mg)"),
    ];

    println!("\nTesting optimization with different dose range constraints:\n");
    println!("{:<30} | {:>12} | {:>10}", "Range", "Optimal Dose", "Cost");
    println!("{}", "-".repeat(60));

    for (min, max, description) in dose_ranges {
        let problem = BestDoseProblem::new(
            &theta,
            weights,
            None,
            target_data.clone(),
            None,
            eq.clone(),
            ems.clone(),
            DoseRange::new(min, max),
            0.5,
            settings.clone(),
            Target::Concentration,
        )?;

        let result = problem.optimize()?;

        // Check if dose hit the bound
        let at_bound = if (result.dose[0] - max).abs() < 1.0 {
            " (at upper bound)"
        } else if (result.dose[0] - min).abs() < 1.0 {
            " (at lower bound)"
        } else {
            ""
        };

        println!(
            "{:<30} | {:>10.1} mg | {:>10.6}{}",
            description, result.dose[0], result.objf, at_bound
        );
    }

    println!("\n{}", "=".repeat(60));
    println!("\nKey Observations:");
    println!("  - Narrower ranges may constrain the optimizer to suboptimal doses");
    println!("  - When the optimizer hits a bound, consider widening the range");
    println!("  - The cost function increases when doses are constrained");
    println!("  - Bounds are enforced via penalty in the cost function");

    Ok(())
}
