use anyhow::Result;
use pmcore::bestdose::{BestDosePosterior, DoseRange, OptimizationStrategy, Prior, Target};
use pmcore::prelude::*;

fn main() -> Result<()> {
    // Example model
    let eq = ode! {
        name: "bestdose_one_compartment",
        params: [ke, v],
        states: [central],
        outputs: [outeq_0],
        routes: [
            bolus(input_0) -> central,
        ],
        diffeq: |x, _t, dx| {
            dx[central] = -ke * x[central];
        },
        out: |x, _t, y| {
            y[outeq_0] = x[central] / v;
        },
    };

    let parameter_space = ParameterSpace::<BoundedParameter>::new()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0);

    let ems = AssayErrorModels::new().add(
        0,
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.20, 0.0, 0.0), 0.0),
    )?;

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

    // Load the population prior produced by the `bimodal_ke` example.
    // Run `cargo run --example bimodal_ke` first to generate this file.
    let prior = Prior::from_file("outputs/bimodal_ke/theta.csv", &parameter_space)?;

    let posterior = BestDosePosterior::builder(eq.clone(), ems.clone(), prior)
        .history(Some(past_data.clone())) // Optional: past data for Bayesian updating
        .progress(false)
        .compute()?;

    println!("Optimizing dose...");

    let bias_weights = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let mut results = Vec::new();

    for bias_weight in &bias_weights {
        println!("Running optimization with bias weight: {}", bias_weight);
        let optimal = posterior
            .optimize(target_data.clone(), Target::Concentration)
            .dose_range(DoseRange::new(0.0, 300.0))
            .bias(*bias_weight)
            .run()?;
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

    // ── Reuse the same posterior across different dose-range constraints ──
    // The posterior is computed once and can be reused for any number of
    // forecasting scenarios. Here we hold the bias weight fixed and vary the
    // allowable dose range, illustrating how bounds constrain the optimum.
    println!("\n=== Dose-range constraints (posterior reused, bias = 0.5) ===");
    println!("{:<24} {:>18}", "Range", "Optimal doses");
    for (min, max) in [(0.0, 300.0), (50.0, 150.0), (200.0, 300.0)] {
        let result = posterior
            .optimize(target_data.clone(), Target::Concentration)
            .dose_range(DoseRange::new(min, max))
            .bias(0.5)
            .run()?;
        println!(
            "{:<24} {:>18}",
            format!("[{min}, {max}] mg"),
            format!("{:?}", result.doses())
        );
    }

    // ── Optimization strategy ──
    // By default BestDose runs a *dual* optimization (both posterior- and
    // population-weighted) and keeps the cheaper result. To force the
    // patient-specific path only, use `PosteriorOnly`.
    let posterior_only = posterior
        .optimize(target_data.clone(), Target::Concentration)
        .dose_range(DoseRange::new(0.0, 300.0))
        .bias(0.0)
        .strategy(OptimizationStrategy::PosteriorOnly)
        .run()?;
    println!(
        "\nPosterior-only (\u{3bb}=0) optimal doses: {:?} (method: {})",
        posterior_only.doses(),
        posterior_only.optimization_method()
    );

    // Print concentration-time predictions for the optimal dose
    let optimal = &results.last().unwrap().1;
    println!("\nConcentration-time predictions for optimal dose:");
    for pred in optimal.predictions().predictions().iter() {
        println!(
            "Time: {:.2} h, Observed: {:.2}, (Pop Mean: {:.4}, Pop Median: {:.4}, Post Mean: {:.4}, Post Median: {:.4})",
            pred.time(), pred.obs().unwrap_or(0.0), pred.pop_mean(), pred.pop_median(), pred.post_mean(), pred.post_median()
        );
    }

    // Print the posterior support points with their filtered population and posterior weights.
    let posterior_theta = posterior.theta();
    let posterior_weights = posterior.posterior_weights();
    let population_weights = posterior.population_weights();
    let param_names = posterior_theta.param_names();

    println!("\n=== Support Points Summary ===");
    println!("Number of support points: {}", posterior_theta.nspp());

    print!(
        "\n{:<8} {:<15} {:<15}",
        "Point", "Prior Weight", "Posterior Weight"
    );
    for name in &param_names {
        print!(" {:<15}", name);
    }
    println!();
    println!("{}", "-".repeat(40 + 16 * param_names.len()));

    for point_idx in 0..posterior_theta.nspp() {
        let row = posterior_theta.matrix().row(point_idx);

        print!(
            "{:<8} {:<15.6e} {:<15.6e}",
            point_idx, population_weights[point_idx], posterior_weights[point_idx]
        );

        for value in row.iter() {
            print!(" {:<15.6}", value);
        }

        println!();
    }

    Ok(())
}
