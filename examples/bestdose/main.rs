use anyhow::Result;
use pmcore::bestdose::{BestDoseOptions, BestDoseProblem, DoseRange, Target};
use pmcore::prelude::*;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info,diffsol=off"))
        .init();

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

    // Load the population distribution (support points + weights) from the local CSV.
    let (theta, weights) = Theta::from_file("examples/bestdose/theta.csv", &parameter_space)?;
    let weights = weights.expect("theta.csv must contain a `prob` column with weights");
    println!(
        "Loaded population prior with {} support points\n",
        theta.matrix().nrows()
    );

    concentration_target(&eq, &ems, &theta, &weights)?;
    auc_from_zero_target(&eq, &theta, &weights)?;
    interval_auc_target(&eq, &theta, &weights)?;

    Ok(())
}

/// Concentration targeting with a patient-specific posterior (NCNPAG) from past data.
///
/// The population distribution is first updated to a patient-specific posterior
/// using the NCNPAG algorithm on the patient's history, then doses are optimized
/// against that posterior.
fn concentration_target(
    eq: &ODE,
    ems: &AssayErrorModels,
    theta: &Theta,
    _weights: &Weights,
) -> Result<()> {
    println!("════════════════════════════════════════════════════════");
    println!("  Concentration target (patient-specific posterior via NCNPAG)");
    println!("════════════════════════════════════════════════════════\n");

    // Helper to synthesize observations from a known patient.
    fn conc(t: f64, dose: f64) -> f64 {
        let ke = 0.3406021231412888;
        let v = 99.99475717544556;
        (dose * (-ke * t).exp()) / v
    }

    // Past data used to compute the patient-specific posterior.
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

    // Compute the patient-specific posterior over the population support points.
    let posterior = EstimationProblem::nonparametric(
        eq.clone(),
        Data::new(vec![past_data]),
        theta.clone(),
        ems.clone(),
    )?
    .fit_with(NcnpagConfig::default())?;

    let post_theta = posterior.get_theta().clone();
    let post_weights = posterior.weights().clone();
    println!(
        "NCNPAG posterior: {} support points\n",
        post_theta.matrix().nrows()
    );

    let problem = BestDoseProblem::new(eq.clone(), post_theta, post_weights)?;

    // Bias-weight sweep: 0.0 (personalized) → 1.0 (population-typical).
    println!("Bias-weight sweep:");
    for bias in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
        let result = problem.optimize(
            target_data.clone(),
            Target::Concentration,
            DoseRange::new(0.0, 300.0),
            bias,
            BestDoseOptions::default(),
        )?;
        println!(
            "  bias {:.2} | doses {:?} | cost {:.6}",
            bias,
            result.doses(),
            result.cost()
        );
    }

    // Reuse the same problem across different dose-range constraints.
    println!("\nDose-range constraints (bias = 0.5):");
    println!("  {:<24} {:>18}", "Range", "Optimal doses");
    for (min, max) in [(0.0, 300.0), (50.0, 150.0), (200.0, 300.0)] {
        let result = problem.optimize(
            target_data.clone(),
            Target::Concentration,
            DoseRange::new(min, max),
            0.5,
            BestDoseOptions::default(),
        )?;
        println!(
            "  {:<24} {:>18}",
            format!("[{min}, {max}] mg"),
            format!("{:?}", result.doses())
        );
    }

    Ok(())
}

/// AUC targeting integrated from time zero (`Target::AUCFromZero`) using the
/// population distribution directly (no past data).
fn auc_from_zero_target(eq: &ODE, theta: &Theta, weights: &Weights) -> Result<()> {
    println!("\n════════════════════════════════════════════════════════");
    println!("  AUC target from zero (Target::AUCFromZero)");
    println!("════════════════════════════════════════════════════════\n");

    println!("Target AUCs: AUC(0-6h) = 50.0, AUC(0-12h) = 80.0 mg*h/L\n");

    let target_data = Subject::builder("AUC target")
        .bolus(0.0, 0.0, 0) // Dose to be optimized
        .observation(6.0, 50.0, 0) // Target AUC at 6h
        .observation(12.0, 80.0, 0) // Target AUC at 12h
        .build();

    let problem = BestDoseProblem::new(eq.clone(), theta.clone(), weights.clone())?;
    let result = problem.optimize(
        target_data,
        Target::AUCFromZero,
        DoseRange::new(100.0, 2000.0),
        0.8,
        BestDoseOptions {
            prediction_interval: 0.1,
        },
    )?;

    println!(
        "Optimal dose: {:.1} mg | cost {:.6}",
        result.doses()[0],
        result.cost()
    );

    println!("\nAUC achievements:");
    for a in result.achievements() {
        let error_pct = ((a.achieved - a.target) / a.target * 100.0).abs();
        println!(
            "  t {:5.1}h | target {:6.1} | achieved {:6.2} | error {:5.1}%",
            a.time, a.target, a.achieved, error_pct
        );
    }

    Ok(())
}

/// Interval AUC targeting from the last dose (`Target::AUCFromLastDose`) using
/// the population distribution directly.
fn interval_auc_target(eq: &ODE, theta: &Theta, weights: &Weights) -> Result<()> {
    println!("\n════════════════════════════════════════════════════════");
    println!("  Interval AUC target (Target::AUCFromLastDose)");
    println!("════════════════════════════════════════════════════════\n");

    println!("Scenario: loading dose + optimized maintenance dose");
    println!("Target: AUC(12-24h) = 60.0 mg*h/L\n");

    let target_data = Subject::builder("Interval AUC target")
        .bolus(0.0, 200.0, 0) // Loading dose (fixed)
        .bolus(12.0, 0.0, 0) // Maintenance dose to be optimized
        .observation(24.0, 60.0, 0) // Target: AUC from t=12 to t=24
        .build();

    let problem = BestDoseProblem::new(eq.clone(), theta.clone(), weights.clone())?;
    let result = problem.optimize(
        target_data,
        Target::AUCFromLastDose,
        DoseRange::new(50.0, 500.0),
        0.8,
        BestDoseOptions {
            prediction_interval: 0.1,
        },
    )?;

    println!(
        "Optimal maintenance dose (at t=12h): {:.1} mg | cost {:.6}",
        result.doses()[1],
        result.cost()
    );

    println!("\nInterval AUC achievements:");
    for a in result.achievements() {
        let error_pct = ((a.achieved - a.target) / a.target * 100.0).abs();
        println!(
            "  t {:5.1}h | target AUC(12-24) {:6.1} | achieved {:6.2} | error {:5.1}%",
            a.time, a.target, a.achieved, error_pct
        );
    }

    Ok(())
}
