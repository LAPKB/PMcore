use anyhow::Result;
use pmcore::bestdose::{BestDoseConfig, BestDosePosterior, DoseRange, Target};
use pmcore::prelude::*;

fn main() -> Result<()> {
    // Example model
    let eq = ode! {
        name: "bestdose_one_compartment",
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

    let (theta, prior) = read_prior("examples/bimodal_ke/output/theta.csv", &parameter_space)?;

    let posterior = BestDosePosterior::compute(
        &theta,
        &prior.unwrap(),
        Some(past_data.clone()), // Optional: past data for Bayesian updating
        eq.clone(),
        config.clone(),
    )?;

    println!("Optimizing dose...");

    let bias_weights = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let mut results = Vec::new();

    for bias_weight in &bias_weights {
        println!("Running optimization with bias weight: {}", bias_weight);
        let optimal = posterior.optimize(
            target_data.clone(),
            None,
            DoseRange::new(0.0, 300.0),
            *bias_weight,
            Target::Concentration,
        )?;
        results.push((bias_weight, optimal));
    }

    // Print results
    for (bias_weight, optimal) in &results {
        let opt_doses = optimal.doses();

        println!(
            "Bias weight: {:.2}\t\t Optimal dose: {:?}\t\tCost: {:.6}\t\tln Cost: {:.4}\t\tMethod: {}",
            bias_weight,
            opt_doses,
            optimal.objf(),
            optimal.objf().ln(),
            optimal.optimization_method()
        );
    }

    // Print concentration-time predictions for the optimal dose
    let optimal = &results.last().unwrap().1;
    println!("\nConcentration-time predictions for optimal dose:");
    for pred in optimal.predictions().predictions().iter() {
        println!(
            "Time: {:.2} h, Observed: {:.2}, (Pop Mean: {:.4}, Pop Median: {:.4}, Post Mean: {:.4}, Post Median: {:.4})",
            pred.time(), pred.obs().unwrap_or(0.0), pred.pop_mean(), pred.pop_median(), pred.post_mean(), pred.post_median()
        );
    }

    Ok(())
}
