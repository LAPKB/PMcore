use anyhow::Result;
use pmcore::bestdose::{BestDosePosterior, DoseRange, OptimizationStrategy, Prior, Target};
use pmcore::prelude::*;

fn main() -> Result<()> {
    // ── Shared model, parameter space, and error model ──
    // A simple one-compartment model with bolus input:
    //   C(t) = Dose * exp(-ke * t) / V
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

    // Load the population prior (support points) from the local CSV.
    let prior = Prior::from_file("examples/bestdose/theta.csv", &parameter_space)?;
    println!(
        "Loaded prior with {} support points\n",
        prior.theta().matrix().nrows()
    );

    concentration_target(&eq, &ems, &prior)?;
    auc_from_zero_target(&eq, &ems, &prior)?;
    interval_auc_target(&eq, &ems, &prior)?;

    Ok(())
}

/// Concentration targeting with Bayesian updating from past data.
///
/// Demonstrates a bias-weight sweep, reusing a single posterior across
/// different dose-range constraints, and forcing the posterior-only strategy.
fn concentration_target(
    eq: &pmcore::prelude::ODE,
    ems: &AssayErrorModels,
    prior: &Prior,
) -> Result<()> {
    println!("═════════════════════════════════════════════════════════");
    println!("  Concentration target (with past data / Bayesian update)");
    println!("═════════════════════════════════════════════════════════\n");

    // Helper to synthesize observations from a known patient.
    fn conc(t: f64, dose: f64) -> f64 {
        let ke = 0.3406021231412888;
        let v = 99.99475717544556;
        (dose * (-ke * t).exp()) / v
    }

    // Past data used for Bayesian updating of the prior.
    let past_data = Subject::builder("Nikola Tesla")
        .bolus(0.0, 150.0, 0)
        .observation(2.0, conc(2.0, 150.0), 0)
        .observation(4.0, conc(4.0, 150.0), 0)
        .observation(6.0, conc(6.0, 150.0), 0)
        .bolus(12.0, 75.0, 0)
        .observation(14.0, conc(2.0, 75.0) + conc(14.0, 150.0), 0)
        .observation(16.0, conc(4.0, 75.0) + conc(16.0, 150.0), 0)
        .observation(18.0, conc(6.0, 75.0) + conc(18.0, 150.0), 0)
        .build();

    // Target profile: doses (set to 0) are optimized to hit the observations.
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

    let posterior = BestDosePosterior::builder(eq.clone(), ems.clone(), prior.clone())
        .history(Some(past_data))
        .progress(false)
        .compute()?;

    // Bias-weight sweep: 0.0 (pure population) → 1.0 (pure patient-specific).
    println!("Bias-weight sweep:");
    let bias_weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    for bias_weight in bias_weights {
        let optimal = posterior
            .optimize(target_data.clone(), Target::Concentration)
            .dose_range(DoseRange::new(0.0, 300.0))
            .bias(bias_weight)
            .run()?;
        println!(
            "  bias {:.2} | doses {:?} | cost {:.6} | method {}",
            bias_weight,
            optimal.doses(),
            optimal.objf(),
            optimal.optimization_method()
        );
    }

    // Reuse the same posterior across different dose-range constraints.
    println!("\nDose-range constraints (posterior reused, bias = 0.5):");
    println!("  {:<24} {:>18}", "Range", "Optimal doses");
    for (min, max) in [(0.0, 300.0), (50.0, 150.0), (200.0, 300.0)] {
        let result = posterior
            .optimize(target_data.clone(), Target::Concentration)
            .dose_range(DoseRange::new(min, max))
            .bias(0.5)
            .run()?;
        println!(
            "  {:<24} {:>18}",
            format!("[{min}, {max}] mg"),
            format!("{:?}", result.doses())
        );
    }

    // Force the patient-specific path only (PosteriorOnly strategy).
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

    // Concentration-time predictions for the last optimization.
    let optimal = posterior
        .optimize(target_data.clone(), Target::Concentration)
        .dose_range(DoseRange::new(0.0, 300.0))
        .bias(1.0)
        .run()?;
    println!("\nConcentration-time predictions for optimal dose:");
    for pred in optimal.predictions().predictions().iter() {
        println!(
            "  t {:5.2} h | obs {:6.2} | pop mean {:.4} | pop median {:.4} | post mean {:.4} | post median {:.4}",
            pred.time(),
            pred.obs().unwrap_or(0.0),
            pred.pop_mean(),
            pred.pop_median(),
            pred.post_mean(),
            pred.post_median()
        );
    }

    // Support-point summary with prior and posterior weights.
    let theta = posterior.theta();
    let posterior_weights = posterior.posterior_weights();
    let population_weights = posterior.population_weights();
    let param_names = theta.param_names();

    println!("\nSupport points: {}", theta.nspp());
    print!(
        "{:<8} {:<15} {:<15}",
        "Point", "Prior Weight", "Posterior Weight"
    );
    for name in &param_names {
        print!(" {:<15}", name);
    }
    println!();
    println!("{}", "-".repeat(40 + 16 * param_names.len()));
    for point_idx in 0..theta.nspp() {
        print!(
            "{:<8} {:<15.6e} {:<15.6e}",
            point_idx, population_weights[point_idx], posterior_weights[point_idx]
        );
        for value in theta.matrix().row(point_idx).iter() {
            print!(" {:<15.6}", value);
        }
        println!();
    }
    println!();

    Ok(())
}

/// AUC targeting integrated from time zero (`Target::AUCFromZero`).
fn auc_from_zero_target(
    eq: &pmcore::prelude::ODE,
    ems: &AssayErrorModels,
    prior: &Prior,
) -> Result<()> {
    println!("\n═════════════════════════════════════════════════════════");
    println!("  AUC target from zero (Target::AUCFromZero)");
    println!("═════════════════════════════════════════════════════════\n");

    println!("Target AUCs: AUC(0-6h) = 50.0, AUC(0-12h) = 80.0 mg*h/L\n");

    let target_data = Subject::builder("AUC target")
        .bolus(0.0, 0.0, 0) // Dose to be optimized
        .observation(6.0, 50.0, 0) // Target AUC at 6h
        .observation(12.0, 80.0, 0) // Target AUC at 12h
        .build();

    let posterior = BestDosePosterior::builder(eq.clone(), ems.clone(), prior.clone())
        .progress(false)
        .compute()?;

    let optimal = posterior
        .optimize(target_data, Target::AUCFromZero)
        .dose_range(DoseRange::new(100.0, 2000.0))
        .bias(0.8)
        .prediction_interval(60.0)
        .run()?;

    println!(
        "Optimal dose: {:.1} mg | cost {:.6}",
        optimal.doses()[0],
        optimal.objf()
    );

    if let Some(auc_preds) = &optimal.auc_predictions() {
        println!("\nAUC predictions:");
        let mut total_error = 0.0;
        for (time, auc) in auc_preds {
            let target = if (time - 6.0).abs() < 0.1 { 50.0 } else { 80.0 };
            let error_pct = ((auc - target) / target * 100.0).abs();
            total_error += error_pct;
            println!(
                "  t {:5.1}h | target {:6.1} | predicted {:6.2} | error {:5.1}%",
                time, target, auc, error_pct
            );
        }
        println!(
            "\n  Mean absolute error: {:.1}%",
            total_error / auc_preds.len() as f64
        );
    }

    Ok(())
}

/// Interval AUC targeting from the last dose (`Target::AUCFromLastDose`).
fn interval_auc_target(
    eq: &pmcore::prelude::ODE,
    ems: &AssayErrorModels,
    prior: &Prior,
) -> Result<()> {
    println!("\n═════════════════════════════════════════════════════════");
    println!("  Interval AUC target (Target::AUCFromLastDose)");
    println!("═════════════════════════════════════════════════════════\n");

    println!("Scenario: loading dose + optimized maintenance dose");
    println!("Target: AUC(12-24h) = 60.0 mg*h/L\n");

    let target_data = Subject::builder("Interval AUC target")
        .bolus(0.0, 200.0, 0) // Loading dose (fixed)
        .bolus(12.0, 0.0, 0) // Maintenance dose to be optimized
        .observation(24.0, 60.0, 0) // Target: AUC from t=12 to t=24
        .build();

    let posterior = BestDosePosterior::builder(eq.clone(), ems.clone(), prior.clone())
        .progress(false)
        .compute()?;

    let optimal = posterior
        .optimize(target_data, Target::AUCFromLastDose)
        .dose_range(DoseRange::new(50.0, 500.0))
        .bias(0.8)
        .prediction_interval(60.0)
        .run()?;

    println!(
        "Optimal maintenance dose (at t=12h): {:.1} mg | cost {:.6}",
        optimal.doses()[0],
        optimal.objf()
    );

    if let Some(auc_preds) = &optimal.auc_predictions() {
        println!("\nInterval AUC predictions:");
        for (time, auc) in auc_preds {
            let target = 60.0;
            let error_pct = ((auc - target) / target * 100.0).abs();
            println!(
                "  t {:5.1}h | target AUC(12-24) {:6.1} | predicted {:6.2} | error {:5.1}%",
                time, target, auc, error_pct
            );
        }
    }

    println!("\nKey difference:");
    println!("  - AUCFromZero:     integrates from t=0 to the observation");
    println!("  - AUCFromLastDose: integrates from the last dose to the observation");

    Ok(())
}
