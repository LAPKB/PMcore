use anyhow::Result;
use pmcore::bestdose::{BestDoseProblem, DoseRange};

use pmcore::prelude::*;
use pmcore::routines::initialization::parse_prior;

fn main() -> Result<()> {
    // Example model
    let eq = equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0];
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
        ErrorModel::additive(ErrorPoly::new(0.0, 0.1, 0.0, 0.0), 0.0, None),
    )?;

    // Make settings
    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_models(ems.clone())
        .build();

    settings.disable_output();

    // Generate a patient with known parameters
    // Ke = 0.5, V = 50
    // C(t) = Dose * exp(-ke * t) / V

    fn conc(t: f64) -> f64 {
        let dose = 150.0; // Example dose
        let ke = 0.5; // Elimination rate constant
        let v = 50.0; // Volume of distribution
        (dose * (-ke * t).exp()) / v
    }

    // Some observed data
    let subject = Subject::builder("Nikola Tesla")
        .bolus(0.0, 100.0, 0)
        .observation(2.0, conc(2.0), 0)
        .observation(4.0, conc(4.0), 0)
        .observation(6.0, conc(6.0), 0)
        .observation(12.0, conc(12.0), 0)
        .build();

    let past_data = subject.clone();

    let theta = parse_prior(
        &"examples/bimodal_ke/output/theta.csv".to_string(),
        &settings,
    )
    .unwrap();

    // Create target data (future dosing scenario we want to optimize)
    let target_data = Subject::builder("Target Patient")
        .bolus(0.0, 100.0, 0) // This dose will be optimized
        .observation(5.0, conc(5.0), 0) // Target observation at t=5.0
        .build();

    // Example usage
    let problem = BestDoseProblem {
        past_data: past_data.clone(),
        theta,
        target_data: target_data.clone(),
        eq: eq.clone(),
        doserange: DoseRange::new(10.0, 1000.0),
        bias_weight: 0.0,
        error_models: ems.clone(),
    };

    println!("Optimizing dose...");

    let bias_weights = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let mut results = Vec::new();

    for bias_weight in &bias_weights {
        let optimal = problem.clone().bias(*bias_weight).optimize()?;
        results.push((bias_weight, optimal));
    }

    // Print results
    for (bias_weight, optimal) in results {
        println!(
            "Bias weight: {:.1}\t\t Optimal dose: {:?}\t\t ln cost: {:.2}",
            bias_weight, optimal.dose, optimal.objf
        );
    }

    Ok(())
}
