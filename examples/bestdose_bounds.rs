use anyhow::Result;
use pmcore::bestdose::{BestDoseConfig, BestDosePosterior, DoseRange, Target};
use pmcore::prelude::*;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info,diffsol=off"))
        .init();

    println!("BestDose with Dose Range Bounds - Example\n");
    println!("==========================================\n");

    // Simple one-compartment PK model
    let eq = ode! {
        name: "bestdose_bounds_one_compartment",
        params: [ke, v],
        states: [central],
        outputs: [cp],
        routes: [
            bolus(dose) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[cp] = x[central] / v;
        },
    };

    let parameter_space = ParameterSpace::new()
        .add(ParameterSpec::bounded("ke", 0.001, 3.0))
        .add(ParameterSpec::bounded("v", 25.0, 250.0));

    let ems = AssayErrorModels::new().add(
        0,
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.20, 0.0, 0.0), 0.0),
    )?;

    let config = BestDoseConfig::new(parameter_space.clone(), ems.clone()).with_progress(false);

    // Load realistic prior from previous NPAG run
    println!("Loading prior from bimodal_ke example...");
    let (theta, prior) = read_prior("examples/bimodal_ke/output/theta.csv", &parameter_space)?;
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

    let posterior = BestDosePosterior::compute(&theta, weights, None, eq.clone(), config.clone())?;

    println!("\nTesting optimization with different dose range constraints:\n");
    println!("{:<30} | {:>12} | {:>10}", "Range", "Optimal Dose", "Cost");
    println!("{}", "-".repeat(60));

    for (min, max, description) in dose_ranges {
        let result = posterior.optimize(
            target_data.clone(),
            None,
            DoseRange::new(min, max),
            0.5,
            Target::Concentration,
        )?;

        let doses: Vec<f64> = result
            .optimal_subject()
            .iter()
            .flat_map(|occ| {
                occ.iter()
                    .filter(|event| matches!(event, Event::Bolus(_) | Event::Infusion(_)))
                    .map(|event| match event {
                        Event::Bolus(bolus) => bolus.amount(),
                        Event::Infusion(infusion) => infusion.amount(),
                        _ => 0.0,
                    })
            })
            .collect();

        // Check if dose hit the bound
        let at_bound = if (doses[0] - max).abs() < 1.0 {
            " (at upper bound)"
        } else if (doses[0] - min).abs() < 1.0 {
            " (at lower bound)"
        } else {
            ""
        };

        println!(
            "{:<30} | {:>10.1} mg | {:>10.6}{}",
            description,
            doses[0],
            result.objf(),
            at_bound
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
