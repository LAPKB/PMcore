use anyhow::Result;
use pmcore::bestdose; // bestdose new
                      // use pmcore::bestdose::bestdose_old as bestdose; // bestdose old

use pmcore::prelude::*;
use pmcore::routines::initialization::parse_prior;

fn main() -> Result<()> {
    // Example model
    let eq = equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            // fetch_cov!(cov, t, wt);
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

    fn conc(t: f64, dose: f64) -> f64 {
        let ke = 0.3406021231412888; // Elimination rate constant
        let v = 99.99475717544556; // Volume of distribution
        (dose * (-ke * t).exp()) / v
    }

    // Some observed data
    let subject = Subject::builder("Nikola Tesla")
        .bolus(0.0, 150.0, 0)
        .observation(2.0, conc(2.0, 150.0), 0)
        .observation(4.0, conc(4.0, 150.0), 0)
        .observation(6.0, conc(6.0, 150.0), 0)
        .bolus(12.0, 75.0, 0)
        .observation(14.0, conc(2.0, 75.0) + conc(14.0, 150.0), 0)
        .observation(16.0, conc(4.0, 75.0) + conc(16.0, 150.0), 0)
        .observation(18.0, conc(6.0, 75.0) + conc(18.0, 150.0), 0)
        .build();

    let past_data = subject.clone();

    let target_data = Subject::builder("Thomas Edison")
        .bolus(0.0, 0.0, 0)
        .observation(2.0, conc(2.0, 150.0), 0)
        .observation(4.0, conc(4.0, 150.0), 0)
        .observation(6.0, conc(6.0, 150.0), 0)
        .bolus(12.0, 0.0, 0)
        .observation(14.0, conc(2.0, 75.0) + conc(14.0, 150.0), 0)
        .observation(16.0, conc(4.0, 75.0) + conc(16.0, 150.0), 0)
        .observation(18.0, conc(6.0, 75.0) + conc(18.0, 150.0), 0)
        .build();

    let (theta, prior) = parse_prior(
        &"examples/bimodal_ke/output/theta.csv".to_string(),
        &settings,
    )
    .unwrap();

    // Example usage - using new() constructor which calculates NPAGFULL11 posterior
    // max_cycles controls NPAGFULL refinement:
    //   0 = NPAGFULL11 only (fast but less accurate)
    //   100 = moderate refinement
    //   500 = full refinement (Fortran default, slow but most accurate)
    let problem = bestdose::BestDoseProblem::new(
        &theta,
        &prior.unwrap(),
        Some(past_data.clone()), // Optional: past data for Bayesian updating
        target_data.clone(),
        None,
        eq.clone(),
        ems.clone(),
        bestdose::DoseRange::new(0.0, 300.0),
        0.0,
        settings.clone(),
        bestdose::Target::Concentration, // Target concentrations (not AUCs)
    )?;

    println!("Optimizing dose...");

    let bias_weights = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let mut results = Vec::new();

    for bias_weight in &bias_weights {
        println!("Running optimization with bias weight: {}", bias_weight);
        let optimal = problem.clone().with_bias_weight(*bias_weight).optimize()?;
        results.push((bias_weight, optimal));
    }

    // Print results
    for (bias_weight, optimal) in &results {
        let opt_doses = optimal
            .optimal_subject
            .iter()
            .flat_map(|occ| {
                occ.events()
                    .iter()
                    .filter_map(|event| match event {
                        Event::Bolus(bolus) => Some(bolus.amount()),
                        Event::Infusion(infusion) => Some(infusion.amount()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<f64>>();

        println!(
            "Bias weight: {:.2}\t\t Optimal dose: {:?}\t\tCost: {:.6}\t\tln Cost: {:.4}\t\tMethod: {}",
            bias_weight,
            opt_doses,
            optimal.objf,
            optimal.objf.ln(),
            optimal.optimization_method
        );
    }

    // Print concentration-time predictions for the optimal dose
    let optimal = &results.last().unwrap().1;
    println!("\nConcentration-time predictions for optimal dose:");
    for pred in optimal.preds.predictions().into_iter() {
        println!(
            "Time: {:.2} h, Observed: {:.2}, (Pop Mean: {:.4}, Pop Median: {:.4}, Post Mean: {:.4}, Post Median: {:.4})",
            pred.time(), pred.obs().unwrap_or(0.0), pred.pop_mean(), pred.pop_median(), pred.post_mean(), pred.post_median()
        );
    }

    Ok(())
}
