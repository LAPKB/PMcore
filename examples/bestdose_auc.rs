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
        |x, p, _t, dx, _rateiv, _cov| {
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

    // Minimal parameter ranges
    let params = Parameters::new()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0);

    let ems = ErrorModels::new().add(
        0,
        ErrorModel::additive(ErrorPoly::new(0.0, 0.20, 0.0, 0.0), 0.0, None),
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
    let weights = prior.unwrap();

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
        &weights,
        None, // No past data - use prior directly
        target_data,
        eq,
        ems,
        DoseRange::new(100.0, 2000.0), // Wider range for AUC targets
        0.0,                           // bias_weight: 0.0 = full personalization
        settings,
        0, // No NPAGFULL refinement (no past data)
        Target::AUC,
    )?;

    println!("Optimizing dose...\n");
    let optimal = problem.optimize()?;

    println!("=== RESULTS ===");
    println!("Optimal dose: {:.1} mg", optimal.dose[0]);
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

    Ok(())
}
