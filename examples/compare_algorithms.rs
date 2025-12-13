//! Compare all non-parametric algorithms on the bimodal_ke dataset
//!
//! Run with: cargo run --release --example compare_algorithms

use anyhow::Result;
use pmcore::prelude::*;
use std::time::Instant;

fn create_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[0] + b[0];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[0] = x[0] / v;
        },
        (1, 1),
    )
}

fn create_settings(algorithm: Algorithm, output_path: &str) -> Settings {
    let params = Parameters::new()
        .add("ke", 0.001, 3.0)
        .add("v", 25.0, 250.0);

    let ems = ErrorModels::new()
        .add(
            0,
            ErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
        )
        .unwrap()
        .add(1, ErrorModel::None)
        .unwrap();

    let mut settings = Settings::builder()
        .set_algorithm(algorithm)
        .set_parameters(params)
        .set_error_models(ems)
        .build();

    settings.set_cycles(10000); // Limit cycles for comparison
    settings.set_prior(Prior::sobol(2028, 22));
    settings.set_output_path(output_path);
    settings.set_write_logs(false); // Disable file logging for cleaner comparison

    settings
}

fn run_algorithm(
    name: &str,
    algorithm: Algorithm,
    data: &Data,
) -> Result<(f64, usize, usize, std::time::Duration)> {
    let eq = create_equation();
    let output_path = format!("examples/bimodal_ke/output_{}/", name.to_lowercase());
    let settings = create_settings(algorithm, &output_path);

    println!("\n============================================================");
    println!("Running {}", name);
    println!("============================================================");

    let start = Instant::now();
    let mut alg = dispatch_algorithm(settings, eq, data.clone())?;
    let result = alg.fit()?;
    let duration = start.elapsed();

    let objf = result.objf();
    let n_spp = result.get_theta().nspp();
    let cycles = result.cycles();

    println!("\n{} Results:", name);
    println!("  -2LL (objective): {:.4}", objf);
    println!("  Support points:   {}", n_spp);
    println!("  Cycles:           {}", cycles);
    println!("  Time:             {:.2?}", duration);

    // Print support points summary
    let theta = result.get_theta();
    let weights = result.weights();
    println!("\n  Support points (ke, v, weight):");
    for (i, spp) in theta.matrix().row_iter().enumerate() {
        let w = if i < weights.len() { weights[i] } else { 0.0 };
        if w > 0.01 {
            // Only show points with > 1% weight
            println!("    [{:.4}, {:.2}] weight: {:.4}", spp[0], spp[1], w);
        }
    }

    Ok((objf, n_spp, cycles, duration))
}

fn main() -> Result<()> {
    // Initialize logging once
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     ALGORITHM COMPARISON: Bimodal Ke Dataset             ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;
    println!("\nDataset: {} subjects", data.len());

    let algorithms = [
        ("NPAG", Algorithm::NPAG),
        ("NPOD", Algorithm::NPOD),
        ("NPSAH", Algorithm::NPSAH),
        ("NPCAT", Algorithm::NPCAT),
    ];

    let mut results = Vec::new();

    for (name, alg) in &algorithms {
        match run_algorithm(name, *alg, &data) {
            Ok(result) => results.push((name.to_string(), result)),
            Err(e) => println!("  ERROR running {}: {}", name, e),
        }
    }

    // Summary table
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                           SUMMARY COMPARISON                             ║");
    println!("╠══════════╦══════════════╦══════════════╦════════╦════════════════════════╣");
    println!("║ Algorithm║    -2LL      ║ Support Pts  ║ Cycles ║        Time            ║");
    println!("╠══════════╬══════════════╬══════════════╬════════╬════════════════════════╣");

    for (name, (objf, n_spp, cycles, duration)) in &results {
        println!(
            "║ {:8} ║ {:12.4} ║ {:12} ║ {:6} ║ {:22.2?} ║",
            name, objf, n_spp, cycles, duration
        );
    }
    println!("╚══════════╩══════════════╩══════════════╩════════╩════════════════════════╝");

    // Find best result
    if let Some((best_name, (best_objf, _, _, _))) = results
        .iter()
        .min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
    {
        println!("\nBest -2LL: {} with {:.4}", best_name, best_objf);
    }

    if let Some((fastest_name, (_, _, _, fastest_time))) =
        results.iter().min_by(|a, b| a.1 .3.cmp(&b.1 .3))
    {
        println!("Fastest:   {} with {:?}", fastest_name, fastest_time);
    }

    Ok(())
}
