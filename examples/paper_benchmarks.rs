//! Comprehensive algorithm benchmarking for the paper
//!
//! Run with: cargo run --release --example paper_benchmarks -- [category]
//!
//! Categories:
//!   all       - Run all benchmarks (WARNING: takes many hours)
//!   a         - Reproducibility (bimodal_ke, 5 seeds, all algorithms)
//!   d         - Convergence speed (theophylline, 3D unimodal)
//!   e         - Lag time (two_eq_lag, 4D with tlag)
//!   f         - Multi-output with covariates (meta, 7D)
//!   g         - High-dimensional (neely, 10D)
//!   quick     - Quick sanity check (bimodal_ke, 1 seed, 3 algorithms)

use anyhow::Result;
use pmcore::prelude::*;
use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

#[derive(Clone)]
struct ExampleRunConfig {
    algorithm: Algorithm,
    parameter_space: ParameterSpace,
    assay_error_models: AssayErrorModels,
    output: OutputPlan,
    runtime: RuntimeOptions,
}

fn nonparametric_method(algorithm: Algorithm) -> Result<NonparametricMethod> {
    Ok(match algorithm {
        Algorithm::NPAG => NonparametricMethod::Npag(NpagOptions),
        Algorithm::NPBO => NonparametricMethod::Npbo(NpboOptions),
        Algorithm::NPCAT => NonparametricMethod::Npcat(NpcatOptions),
        Algorithm::NPCMA => NonparametricMethod::Npcma(NpcmaOptions),
        Algorithm::NPOD => NonparametricMethod::Npod(NpodOptions),
        Algorithm::NPOPT => NonparametricMethod::Npopt(NpoptOptions),
        Algorithm::NPPSO => NonparametricMethod::Nppso(NppsoOptions),
        Algorithm::NPSAH => NonparametricMethod::Npsah(NpsahOptions),
        Algorithm::NPSAH2 => NonparametricMethod::Npsah2(Npsah2Options),
        Algorithm::NPXO => NonparametricMethod::Npxo(NpxoOptions),
        Algorithm::NEXUS => NonparametricMethod::Nexus(NexusOptions),
        Algorithm::POSTPROB => NonparametricMethod::Postprob(PostProbOptions),
        other => anyhow::bail!("unsupported nonparametric algorithm: {:?}", other),
    })
}

fn build_problem<E: pharmsol::Equation + Clone>(
    equation: E,
    data: Data,
    config: &ExampleRunConfig,
) -> Result<EstimationProblem<E>> {
    let observations = config
        .assay_error_models
        .iter()
        .filter(|(_, model)| !matches!(model, AssayErrorModel::None))
        .fold(
            ObservationSpec::new().with_assay_error_models(config.assay_error_models.clone()),
            |spec, (outeq, _)| {
                spec.add_channel(ObservationChannel::continuous(
                    outeq,
                    format!("obs_{}", outeq),
                ))
            },
        );

    let model = ModelDefinition::builder(equation)
        .parameters(config.parameter_space.clone())
        .observations(observations)
        .build()?;

    EstimationProblem::builder(model, data)
        .method(EstimationMethod::Nonparametric(nonparametric_method(
            config.algorithm,
        )?))
        .output(config.output.clone())
        .runtime(config.runtime.clone())
        .build()
}

fn bounded_parameter_space(bounds: &[(&str, f64, f64)]) -> ParameterSpace {
    bounds.iter().fold(ParameterSpace::new(), |space, (name, lower, upper)| {
        space.add(ParameterSpec::bounded(*name, *lower, *upper))
    })
}

// ============================================================================
// MODELS
// ============================================================================

fn bimodal_ke_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, _b, rateiv, _cov| {
            fetch_params!(p, ke, _v);
            dx[0] = -ke * x[0] + rateiv[1];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ke, v);
            y[1] = x[0] / v;
        },
    )
}

fn theophylline_equation() -> equation::Analytical {
    equation::Analytical::new(
        one_compartment_with_absorption,
        |_p, _t, _cov| {},
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, v);
            y[0] = x[1] * 1000.0 / v;
        },
    )
}

fn two_eq_lag_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, _t, dx, b, _rateiv, _cov| {
            fetch_params!(p, ka, ke);
            dx[0] = -ka * x[0] + b[0];
            dx[1] = ka * x[0] - ke * x[1];
        },
        |p, _t, _cov| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0=>tlag}
        },
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, _t, _cov, y| {
            fetch_params!(p, _ka, _ke, _tlag, v);
            y[0] = x[1] / v;
        },
    )
}

fn meta_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, t, dx, _b, rateiv, cov| {
            fetch_cov!(cov, t, wt, pkvisit);
            fetch_params!(p, cls, fm, k20, relv, theta1, theta2, vs);
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let ke = cl / v;
            let _v2 = relv * v;
            dx[0] = rateiv[1] - ke * x[0] * (1.0 - fm) - fm * x[0];
            dx[1] = fm * x[0] - k20 * x[1];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_cov!(cov, t, wt, pkvisit);
            fetch_params!(p, cls, _fm, _k20, relv, theta1, theta2, vs);
            let _cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let v2 = relv * v;
            y[1] = x[0] / v;
            y[2] = x[1] / v2;
        },
    )
}

fn neely_equation() -> equation::ODE {
    equation::ODE::new(
        |x, p, t, dx, _b, rateiv, cov| {
            fetch_params!(p, cls, k30, k40, qs, vps, vs, fm1, fm2, theta1, theta2);
            fetch_cov!(cov, t, wt, pkvisit);
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let q = qs * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let vp = vps * (wt / 70.0);
            let ke = cl / v;
            let k12 = q / v;
            let k21 = q / vp;
            dx[0] = rateiv[1] - ke * x[0] * (1.0 - fm1 - fm2) - (fm1 + fm2) * x[0] - k12 * x[0]
                + k21 * x[1];
            dx[1] = k12 * x[0] - k21 * x[1];
            dx[2] = fm1 * x[0] - k30 * x[2];
            dx[3] = fm2 * x[0] - k40 * x[3];
        },
        |_p, _t, _cov| lag! {},
        |_p, _t, _cov| fa! {},
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_params!(p, cls, _k30, _k40, qs, vps, vs, _fm1, _fm2, theta1, theta2);
            fetch_cov!(cov, t, wt, pkvisit);
            let vfrac1 = 0.068202;
            let vfrac2 = 0.022569;
            let _cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let _q = qs * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let _vp = vps * (wt / 70.0);
            let vm1 = vfrac1 * v;
            let vm2 = vfrac2 * v;
            y[1] = x[0] / v;
            y[2] = x[2] / vm1;
            y[3] = x[3] / vm2;
        },
    )
}

// ============================================================================
// DATASET CONFIGURATIONS
// ============================================================================

#[derive(Clone)]
struct DatasetConfig {
    name: &'static str,
    data_path: &'static str,
    parameters: Vec<(&'static str, f64, f64)>,
    error_models: Vec<(usize, f64, f64, f64, bool)>, // (output, c0, c1, scale, is_proportional)
}

fn bimodal_ke_config() -> DatasetConfig {
    DatasetConfig {
        name: "bimodal_ke",
        data_path: "examples/bimodal_ke/bimodal_ke.csv",
        parameters: vec![("ke", 0.001, 3.0), ("v", 25.0, 250.0)],
        error_models: vec![(1, 0.0, 0.5, 0.0, false)],
    }
}

fn theophylline_config() -> DatasetConfig {
    DatasetConfig {
        name: "theophylline",
        data_path: "examples/theophylline/theophylline.csv",
        parameters: vec![("ka", 0.001, 3.0), ("ke", 0.001, 3.0), ("v", 0.001, 50.0)],
        error_models: vec![(0, 0.1, 0.1, 2.0, true)],
    }
}

fn two_eq_lag_config() -> DatasetConfig {
    DatasetConfig {
        name: "two_eq_lag",
        data_path: "examples/two_eq_lag/two_eq_lag.csv",
        parameters: vec![
            ("ka", 0.1, 0.9),
            ("ke", 0.001, 0.1),
            ("tlag", 0.0, 4.0),
            ("v", 30.0, 120.0),
        ],
        error_models: vec![(0, -0.00119, 0.44379, 0.0, false)],
    }
}

fn meta_config() -> DatasetConfig {
    DatasetConfig {
        name: "meta",
        data_path: "examples/meta/meta.csv",
        parameters: vec![
            ("cls", 0.1, 10.0),
            ("fm", 0.0, 1.0),
            ("k20", 0.01, 1.0),
            ("relv", 0.1, 1.0),
            ("theta1", 0.1, 10.0),
            ("theta2", 0.1, 10.0),
            ("vs", 1.0, 10.0),
        ],
        error_models: vec![(1, 1.0, 0.1, 5.0, true), (2, 1.0, 0.1, 5.0, true)],
    }
}

fn neely_config() -> DatasetConfig {
    DatasetConfig {
        name: "neely",
        data_path: "examples/neely/data.csv",
        parameters: vec![
            ("cls", 0.0, 0.4),
            ("k30", 0.0, 0.5),
            ("k40", 0.3, 1.5),
            ("qs", 0.0, 0.5),
            ("vps", 0.0, 5.0),
            ("vs", 0.0, 2.0),
            ("fm1", 0.0, 0.2),
            ("fm2", 0.0, 0.1),
            ("theta1", -4.0, 2.0),
            ("theta2", -2.0, 0.5),
        ],
        error_models: vec![
            (1, 1.0, 0.1, 5.0, true),
            (2, 1.0, 0.1, 5.0, true),
            (3, 1.0, 0.1, 5.0, true),
        ],
    }
}

// ============================================================================
// BENCHMARK RESULT
// ============================================================================

#[derive(Debug)]
struct BenchmarkResult {
    experiment: String,
    dataset: String,
    algorithm: String,
    seed: u64,
    cycles: usize,
    time_secs: f64,
    objf: f64,
    n_spp: usize,
    converged: bool,
}

impl BenchmarkResult {
    fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{:.2},{:.4},{},{}\n",
            self.experiment,
            self.dataset,
            self.algorithm,
            self.seed,
            self.cycles,
            self.time_secs,
            self.objf,
            self.n_spp,
            self.converged
        )
    }
}

// ============================================================================
// BENCHMARK RUNNER
// ============================================================================

fn create_config(
    algorithm: Algorithm,
    config: &DatasetConfig,
    seed: u64,
    max_cycles: usize,
    output_path: &str,
) -> ExampleRunConfig {
    let parameter_space = bounded_parameter_space(&config.parameters);

    let mut ems = AssayErrorModels::new();
    for (output, c0, c1, scale, is_proportional) in &config.error_models {
        if *is_proportional {
            ems = ems
                .add(
                    *output,
                    AssayErrorModel::proportional(ErrorPoly::new(*c0, *c1, 0.0, 0.0), *scale),
                )
                .unwrap();
        } else {
            ems = ems
                .add(
                    *output,
                    AssayErrorModel::additive(ErrorPoly::new(*c0, *c1, 0.0, 0.0), *scale),
                )
                .unwrap();
        }
    }

    ExampleRunConfig {
        algorithm,
        parameter_space,
        assay_error_models: ems,
        output: OutputPlan {
            write: true,
            path: Some(output_path.to_string()),
        },
        runtime: RuntimeOptions {
            cycles: max_cycles,
            cache: true,
            progress: true,
            idelta: 0.12,
            tad: 0.0,
            prior: Some(Prior::sobol(2028, seed as usize)),
            ..RuntimeOptions::default()
        },
    }
}

/// Run a single benchmark, writing theta files for post-hoc analysis
macro_rules! run_fit {
    ($settings:expr, $eq:expr, $data:expr) => {{
        let config = $settings;
        let equation = $eq;
        let fit_result = fit(build_problem(equation, $data.clone(), &config)?)?;
        let result = fit_result
            .as_nonparametric()
            .expect("benchmark fit should yield a nonparametric result");
        let _ = result.write_theta();
        (result.objf(), result.get_theta().nspp(), result.cycles())
    }};
}

fn run_single_benchmark(
    experiment: &str,
    dataset_config: &DatasetConfig,
    algorithm: Algorithm,
    algorithm_name: &str,
    seed: u64,
    max_cycles: usize,
    data: &Data,
) -> Result<BenchmarkResult> {
    let output_path = format!(
        "examples/paper_benchmarks/output/{}/{}_seed{}/",
        dataset_config.name, algorithm_name, seed
    );
    fs::create_dir_all(&output_path)?;

    let config = create_config(algorithm, dataset_config, seed, max_cycles, &output_path);

    println!(
        "  Running {} on {} (seed {})...",
        algorithm_name, dataset_config.name, seed
    );

    let start = Instant::now();

    let (objf, n_spp, cycles) = match dataset_config.name {
        "bimodal_ke" => run_fit!(config, bimodal_ke_equation(), data),
        "theophylline" => run_fit!(config, theophylline_equation(), data),
        "two_eq_lag" => run_fit!(config, two_eq_lag_equation(), data),
        "meta" => run_fit!(config, meta_equation(), data),
        "neely" => run_fit!(config, neely_equation(), data),
        _ => anyhow::bail!("Unknown dataset: {}", dataset_config.name),
    };

    let duration = start.elapsed();

    let result = BenchmarkResult {
        experiment: experiment.to_string(),
        dataset: dataset_config.name.to_string(),
        algorithm: algorithm_name.to_string(),
        seed,
        cycles,
        time_secs: duration.as_secs_f64(),
        objf,
        n_spp,
        converged: true,
    };

    println!(
        "    -> -2LL: {:.4}, cycles: {}, time: {:.2}s, spp: {}",
        result.objf, result.cycles, result.time_secs, result.n_spp
    );

    Ok(result)
}

// ============================================================================
// ALGORITHM SETS
// ============================================================================

fn all_algorithms() -> Vec<(&'static str, Algorithm)> {
    vec![
        ("NPAG", Algorithm::NPAG),
        ("NPOD", Algorithm::NPOD),
        ("NPSAH", Algorithm::NPSAH),
        ("NPSAH2", Algorithm::NPSAH2),
        ("NPCAT", Algorithm::NPCAT),
        ("NPOPT", Algorithm::NPOPT),
        ("NPPSO", Algorithm::NPPSO),
        ("NPXO", Algorithm::NPXO),
        ("NPBO", Algorithm::NPBO),
        ("NPCMA", Algorithm::NPCMA),
        ("NEXUS", Algorithm::NEXUS),
    ]
}

/// Algorithms competitive enough to test on harder problems
fn competitive_algorithms() -> Vec<(&'static str, Algorithm)> {
    vec![
        ("NPAG", Algorithm::NPAG),
        ("NPOD", Algorithm::NPOD),
        ("NPSAH", Algorithm::NPSAH),
        ("NPSAH2", Algorithm::NPSAH2),
        ("NPCAT", Algorithm::NPCAT),
        ("NPOPT", Algorithm::NPOPT),
        ("NPPSO", Algorithm::NPPSO),
        ("NEXUS", Algorithm::NEXUS),
    ]
}

// ============================================================================
// EXPERIMENT CATEGORIES
// ============================================================================

fn run_category_a(results_file: &mut File) -> Result<()> {
    println!("\n========================================");
    println!("CATEGORY A: Reproducibility & Multimodality");
    println!("bimodal_ke | 51 subj | 2D | bimodal ke");
    println!("========================================\n");

    let config = bimodal_ke_config();
    let data = data::read_pmetrics(config.data_path)?;
    let seeds = [42u64, 123, 456, 789, 1001];

    for (name, alg) in &all_algorithms() {
        for seed in &seeds {
            match run_single_benchmark("A", &config, *alg, name, *seed, 10000, &data) {
                Ok(r) => {
                    results_file.write_all(r.to_csv_row().as_bytes())?;
                    results_file.flush()?;
                }
                Err(e) => eprintln!("    ERROR: {}", e),
            }
        }
    }
    Ok(())
}

fn run_category_d(results_file: &mut File) -> Result<()> {
    println!("\n========================================");
    println!("CATEGORY D: Unimodal Convergence");
    println!("theophylline | 12 subj | 3D | analytical");
    println!("========================================\n");

    let config = theophylline_config();
    let data = data::read_pmetrics(config.data_path)?;
    let seeds = [42u64, 123, 456];

    for (name, alg) in &competitive_algorithms() {
        for seed in &seeds {
            match run_single_benchmark("D", &config, *alg, name, *seed, 500, &data) {
                Ok(r) => {
                    results_file.write_all(r.to_csv_row().as_bytes())?;
                    results_file.flush()?;
                }
                Err(e) => eprintln!("    ERROR: {}", e),
            }
        }
    }
    Ok(())
}

fn run_category_e(results_file: &mut File) -> Result<()> {
    println!("\n========================================");
    println!("CATEGORY E: Lag Time Estimation");
    println!("two_eq_lag | 20 subj | 4D | ODE + lag");
    println!("========================================\n");

    let config = two_eq_lag_config();
    let data = data::read_pmetrics(config.data_path)?;
    let seeds = [42u64, 123, 456];

    for (name, alg) in &competitive_algorithms() {
        for seed in &seeds {
            match run_single_benchmark("E", &config, *alg, name, *seed, 5000, &data) {
                Ok(r) => {
                    results_file.write_all(r.to_csv_row().as_bytes())?;
                    results_file.flush()?;
                }
                Err(e) => eprintln!("    ERROR: {}", e),
            }
        }
    }
    Ok(())
}

fn run_category_f(results_file: &mut File) -> Result<()> {
    println!("\n========================================");
    println!("CATEGORY F: Medium-dim + Covariates");
    println!("meta | 19 subj | 7D | 2 outputs | covariates");
    println!("========================================\n");

    let config = meta_config();
    let data = data::read_pmetrics(config.data_path)?;
    let seeds = [42u64, 123, 456];

    for (name, alg) in &competitive_algorithms() {
        for seed in &seeds {
            match run_single_benchmark("F", &config, *alg, name, *seed, 5000, &data) {
                Ok(r) => {
                    results_file.write_all(r.to_csv_row().as_bytes())?;
                    results_file.flush()?;
                }
                Err(e) => eprintln!("    ERROR: {}", e),
            }
        }
    }
    Ok(())
}

fn run_category_g(results_file: &mut File) -> Result<()> {
    println!("\n========================================");
    println!("CATEGORY G: High Dimensionality");
    println!("neely | 22 subj | 10D | 3 outputs | covariates");
    println!("========================================\n");

    let config = neely_config();
    let data = data::read_pmetrics(config.data_path)?;
    let seeds = [42u64, 123, 456];

    for (name, alg) in &competitive_algorithms() {
        for seed in &seeds {
            match run_single_benchmark("G", &config, *alg, name, *seed, 1000, &data) {
                Ok(r) => {
                    results_file.write_all(r.to_csv_row().as_bytes())?;
                    results_file.flush()?;
                }
                Err(e) => eprintln!("    ERROR: {}", e),
            }
        }
    }
    Ok(())
}

fn run_quick(results_file: &mut File) -> Result<()> {
    println!("\n========================================");
    println!("QUICK: Sanity Check");
    println!("========================================\n");

    let config = bimodal_ke_config();
    let data = data::read_pmetrics(config.data_path)?;
    let algorithms = vec![
        ("NPAG", Algorithm::NPAG),
        ("NPOD", Algorithm::NPOD),
        ("NPSAH2", Algorithm::NPSAH2),
    ];

    for (name, alg) in &algorithms {
        match run_single_benchmark("quick", &config, *alg, name, 42, 1000, &data) {
            Ok(r) => {
                results_file.write_all(r.to_csv_row().as_bytes())?;
                results_file.flush()?;
            }
            Err(e) => eprintln!("    ERROR: {}", e),
        }
    }
    Ok(())
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn,diffsol=off"))
        .with_target(false)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let category = args.get(1).map(|s| s.as_str()).unwrap_or("quick");

    fs::create_dir_all("examples/paper_benchmarks/output")?;

    let results_path = format!(
        "examples/paper_benchmarks/results_{}.csv",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );
    let mut results_file = File::create(&results_path)?;
    writeln!(
        results_file,
        "experiment,dataset,algorithm,seed,cycles,time_secs,objf,n_spp,converged"
    )?;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       PAPER BENCHMARKS: Algorithm Comparison             ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("\nResults will be saved to: {}", results_path);

    match category {
        "all" => {
            run_category_a(&mut results_file)?;
            run_category_d(&mut results_file)?;
            run_category_e(&mut results_file)?;
            run_category_f(&mut results_file)?;
            run_category_g(&mut results_file)?;
        }
        "a" => run_category_a(&mut results_file)?,
        "d" => run_category_d(&mut results_file)?,
        "e" => run_category_e(&mut results_file)?,
        "f" => run_category_f(&mut results_file)?,
        "g" => run_category_g(&mut results_file)?,
        "quick" | _ => run_quick(&mut results_file)?,
    }

    println!("\n========================================");
    println!("BENCHMARKS COMPLETE");
    println!("Results saved to: {}", results_path);
    println!("========================================");

    Ok(())
}
