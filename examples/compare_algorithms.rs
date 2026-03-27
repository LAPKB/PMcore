//! Compare all non-parametric algorithms on the bimodal_ke dataset
//!
//! Run with: cargo run --release --example compare_algorithms

use anyhow::Result;
use pmcore::prelude::*;
use std::time::Instant;

fn build_problem<E: pharmsol::Equation + Clone>(
    equation: E,
    data: Data,
    method: NonparametricMethod,
    output_path: &str,
    initialize_logs: bool,
) -> Result<EstimationProblem<E>> {
    let parameters = ParameterSpace::new()
        .add(ParameterSpec::bounded("ke", 0.001, 3.0))
        .add(ParameterSpec::bounded("v", 25.0, 250.0));

    let assay_error_models = AssayErrorModels::new().add(
        1,
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.5, 0.0, 0.0), 0.0),
    )?;

    let observations = ObservationSpec::new()
        .with_assay_error_models(assay_error_models)
        .add_channel(ObservationChannel::continuous(1, "obs_1"));

    let model = ModelDefinition::builder(equation)
        .parameters(parameters)
        .observations(observations)
        .build()?;

    EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(method))
        .output(OutputPlan {
            write: true,
            path: Some(output_path.to_string()),
        })
        .runtime(RuntimeOptions {
            cycles: 10_000,
            prior: Some(Prior::sobol(2028, 22)),
            logging: LoggingOptions {
                initialize: initialize_logs,
                write: false,
                stdout: true,
                ..LoggingOptions::default()
            },
            ..RuntimeOptions::default()
        })
        .build()
}

fn create_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[1] = -ke * x[1] + rateiv[1] + b[1];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[1] = x[1] / v;
        },
    )
}

fn run_algorithm(
    name: &str,
    method: NonparametricMethod,
    data: &Data,
    initialize_logs: bool,
) -> Result<(f64, usize, usize, std::time::Duration)> {
    let eq = create_equation();
    let output_path = format!("examples/bimodal_ke/output_{}/", name.to_lowercase());

    println!("\n============================================================");
    println!("Running {}", name);
    println!("============================================================");

    let start = Instant::now();
    let fit_result = fit(build_problem(eq, data.clone(), method, &output_path, initialize_logs)?)?;
    let duration = start.elapsed();
    let result = fit_result
        .as_nonparametric()
        .expect("nonparametric comparison should yield a nonparametric result");

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
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     ALGORITHM COMPARISON: Bimodal Ke Dataset             ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;
    println!("\nDataset: {} subjects", data.len());

    let algorithms = [
        ("NPAG", NonparametricMethod::Npag(NpagOptions::default())),
        ("NPOD", NonparametricMethod::Npod(NpodOptions::default())),
        ("NPSAH", NonparametricMethod::Npsah(NpsahOptions::default())),
        ("NPSAH2", NonparametricMethod::Npsah2(Npsah2Options::default())),
        ("NPCAT", NonparametricMethod::Npcat(NpcatOptions::default())),
        ("NPOPT", NonparametricMethod::Npopt(NpoptOptions::default())),
        ("NPPSO", NonparametricMethod::Nppso(NppsoOptions::default())),
        ("NPXO", NonparametricMethod::Npxo(NpxoOptions::default())),
        ("NPBO", NonparametricMethod::Npbo(NpboOptions::default())),
        ("NPCMA", NonparametricMethod::Npcma(NpcmaOptions::default())),
        ("NEXUS", NonparametricMethod::Nexus(NexusOptions::default())),
    ];

    let mut results = Vec::new();

    for (index, (name, method)) in algorithms.iter().enumerate() {
        match run_algorithm(name, *method, &data, index == 0) {
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
