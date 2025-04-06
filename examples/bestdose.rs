use anyhow::Result;
use pmcore::bestdose::{optimize_dose, DoseOptimizer, DoseRange};
use pmcore::prelude::data::read_pmetrics;
use pmcore::prelude::*;

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

    let params = Parameters::new()
        .add("ke", 0.001, 3.0, false)
        .add("v", 25.0, 250.0, false);

    // Read BKE data
    let data = read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;

    // Make settings
    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::NPAG)
        .set_parameters(params)
        .set_error_model(ErrorModel::Additive, 0.0, (0.0, 0.05, 0.0, 0.0))
        .build();

    settings.disable_output();

    // Run NPAG
    let mut algorithm = dispatch_algorithm(settings, eq.clone(), data)?;

    println!("Running NPAG...");

    let result = algorithm.fit()?;
    println!("Finished NPAG...");
    let theta = result.get_theta().clone();

    // Some observed data
    let subject = Subject::builder("Nikola Tesla")
        .bolus(0.0, 20.0, 0)
        .observation(12.0, 8.0, 0)
        .build();

    let past_data = Data::new(vec![subject.clone()]);

    // Example usage
    let problem = DoseOptimizer {
        past_data: past_data.clone(),
        theta,
        target_concentration: 10.0,
        target_time: 5.0,
        eq: eq.clone(),
        doserange: DoseRange::new(0.0, 10000.0),
        bias_weight: 0.0,
    };

    println!("Optimizing dose...");
    let optimal = optimize_dose(problem)?;

    println!("Optimal dose: {:#?}", optimal);

    // Test different values of bias_weight
    let bias_weights = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let mut results = Vec::new();
    for bias_weight in &bias_weights {
        let problem = DoseOptimizer {
            past_data: past_data.clone(),
            theta: result.get_theta().clone(),
            target_concentration: 10.0,
            target_time: 5.0,
            eq: eq.clone(),
            doserange: DoseRange::new(0.0, 10000.0),
            bias_weight: *bias_weight,
        };

        let optimal = optimize_dose(problem)?;
        results.push((bias_weight, optimal));
    }

    // Print results
    for (bias_weight, optimal) in results {
        println!(
            "Bias weight: {}, Optimal dose: {:.2}, Objective function: {:.2}",
            bias_weight, optimal.dose, optimal.objf
        );
    }

    Ok(())
}
