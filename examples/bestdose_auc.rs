use anyhow::Result;
use pmcore::bestdose::{BestDoseProblem, DoseRange, Target};
use pmcore::prelude::*;
use pmcore::routines::initialization::parse_prior;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("BestDose AUC Target - Minimal Example\n");
    println!("======================================\n");

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

    // Minimal parameter ranges
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
    settings.set_idelta(60.0); // 1 hour intervals for AUC calculation

    // Load realistic prior from previous NPAG run (47 support points)
    println!("Loading prior from bimodal_ke example...");
    let (theta, prior) = parse_prior(
        &"examples/bimodal_ke/output/theta.csv".to_string(),
        &settings,
    )?;
    let weights = prior.as_ref().unwrap();

    println!("Prior: {} support points\n", theta.matrix().nrows());

    // Target: achieve specific AUC values (simple targets)
    println!("Target AUCs:");
    println!("  AUC(0-6h) = 50.0 mg*h/L");
    println!("  AUC(0-12h) = 80.0 mg*h/L\n");

    let target_data = Subject::builder("Target")
        .bolus(0.0, 0.0, 0) // Dose to be optimized
        .observation(6.0, 50.0, 0) // Target AUC at 6h
        .observation(12.0, 80.0, 0) // Target AUC at 12h
        .build();

    println!("Creating BestDose problem with AUC targets...");
    let problem = BestDoseProblem::new(
        &theta,
        weights,
        None, // No past data - use prior directly
        target_data.clone(),
        None,
        eq.clone(),
        ems.clone(),
        DoseRange::new(100.0, 2000.0), // Wider range for AUC targets
        0.8,                           // for AUC targets higher bias_weight usually works best
        settings.clone(),
        Target::AUCFromZero, // Cumulative AUC from time 0
    )?;

    println!("Optimizing dose...\n");
    let optimal = problem.optimize()?;

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

    println!("=== RESULTS ===");
    println!("Optimal dose: {:.1} mg", opt_doses[0]);
    println!("Cost function: {:.6}", optimal.objf);

    if let Some(auc_preds) = &optimal.auc_predictions {
        println!("\nAUC Predictions:");
        let mut total_error = 0.0;
        for (time, auc) in auc_preds {
            // Find the target AUC for this time
            let target = if (*time - 6.0).abs() < 0.1 {
                50.0
            } else if (*time - 12.0).abs() < 0.1 {
                80.0
            } else {
                0.0
            };
            let error_pct = ((auc - target) / target * 100.0).abs();
            total_error += error_pct;
            println!(
                "  Time: {:5.1}h | Target: {:6.1} | Predicted: {:6.2} | Error: {:5.1}%",
                time, target, auc, error_pct
            );
        }
        println!(
            "\n  Mean absolute error: {:.1}%",
            total_error / auc_preds.len() as f64
        );
    } else {
        println!("\nConcentration Predictions:");
        for pred in optimal.preds.predictions() {
            println!(
                "  Time: {:5.1}h | Target: {:6.1} | Predicted: {:6.2}",
                pred.time(),
                pred.obs().unwrap_or(0.0),
                pred.post_mean()
            );
        }
    }

    // =========================================================================
    // EXAMPLE 2: Interval AUC (AUCFromLastDose)
    // =========================================================================
    println!("\n\n");
    println!("════════════════════════════════════════════════════════");
    println!("  EXAMPLE 2: Interval AUC (AUCFromLastDose)");
    println!("════════════════════════════════════════════════════════\n");

    println!("Scenario: Loading dose + maintenance dose");
    println!("Target: AUC₁₂₋₂₄ = 60.0 mg*h/L (interval from t=12 to t=24)\n");

    let target_interval = Subject::builder("Target_Interval")
        .bolus(0.0, 200.0, 0) // Loading dose (fixed)
        .bolus(12.0, 0.0, 0) // Maintenance dose to be optimized
        .observation(24.0, 60.0, 0) // Target: AUC from t=12 to t=24
        .build();

    println!("Creating BestDose problem with interval AUC target...");
    let problem_interval = BestDoseProblem::new(
        &theta,
        weights,
        None,
        target_interval.clone(),
        None,
        eq.clone(),
        ems.clone(),
        DoseRange::new(50.0, 500.0),
        0.8,
        settings.clone(),
        Target::AUCFromLastDose, // Interval AUC from last dose!
    )?;

    println!("Optimizing maintenance dose...\n");
    let optimal_interval = problem_interval.optimize()?;

    let doses: Vec<f64> = optimal_interval
        .optimal_subject
        .iter()
        .map(|occ| {
            occ.iter()
                .filter(|event| match event {
                    Event::Bolus(_) => true,
                    Event::Infusion(_) => true,
                    _ => false,
                })
                .map(|event| match event {
                    Event::Bolus(bolus) => bolus.amount(),
                    Event::Infusion(infusion) => infusion.amount(),
                    _ => 0.0,
                })
        })
        .flatten()
        .collect();

    println!("=== INTERVAL AUC RESULTS ===");
    println!("Optimal maintenance dose (at t=12h): {:.1} mg", doses[0]);
    println!("Cost function: {:.6}", optimal_interval.objf);

    if let Some(auc_preds) = &optimal_interval.auc_predictions {
        println!("\nInterval AUC Predictions:");
        for (time, auc) in auc_preds {
            let target = 60.0;
            let error_pct = ((auc - target) / target * 100.0).abs();
            println!(
                "  Time: {:5.1}h | Target AUC₁₂₋₂₄: {:6.1} | Predicted: {:6.2} | Error: {:5.1}%",
                time, target, auc, error_pct
            );
        }
    }

    println!("\n");
    println!("════════════════════════════════════════════════════════");
    println!("  KEY DIFFERENCE:");
    println!("  - AUCFromZero:     Integrates from t=0 to observation");
    println!("  - AUCFromLastDose: Integrates from last dose to observation");
    println!("════════════════════════════════════════════════════════");

    Ok(())
}
