//! Run SAEM on the bimodal_ke dataset
//!
//! This example demonstrates using the SAEM algorithm for a simple
//! one-compartment model with elimination rate constant (ke) and volume (v).
//!
//! Run with: cargo run --example bimodal_ke_saem --release

use anyhow::Result;
use pmcore::prelude::*;

use log::LevelFilter;

fn main() -> Result<()> {
    let _ = simple_logging::log_to_file("examples/analytical_saem_test/logs/pmcore.log", LevelFilter::Info);

    let data = data::read_pmetrics("examples/analytical_saem_test/converted_data_theo.csv")?;
    println!("Loaded {} subjects", data.len());

    // Create model
    let equation = analytical! {
        name: "theophylline",
        params: [ka, ke, v],
        states: [depot, central],
        outputs: [outeq_0],
        routes: [
            bolus(input_0) -> depot,
        ],
        structure: one_compartment_with_absorption,
        out: |x, _t, y| {
            y[outeq_0] = x[central] / v;
        },
    };

    let problem = EstimationProblem::parametric(equation, data)
        // Must currently follow the model metadata order: ka, ke, v.
        .parameter(Parameter::log("ka").with_initial(1.0))
        .parameter(Parameter::log("ke").with_initial(0.025))
        .parameter(Parameter::log("v").with_initial(20.0))
        .error_model("outeq_0", ResidualErrorModel::constant(1.0))
        .build()?;

    let config = SaemConfig::new()
        .seed(632545)
        .n_chains(25)
        .mcmc_iterations(2)
        .eta_block_iterations(0)
        .burn_in(100)
        .k1_iterations(300)
        .k2_iterations(150);

    let result = problem.fit_with(config)?;

    // Write all output files
    result.write_outputs("examples/analytical_saem_test/pmcore_output/", 0.0, 0.0)?;
    println!("\nOutput files written to: examples/analytical_saem_test/pmcore_output/");

    // Print comprehensive results summary (matching R saemix format)
    // print_saem_report(result);

    Ok(())
}



// /// Create analytical one-compartment model (much faster than ODE)
// fn create_equation() -> equation::Analytical {
//     equation::Analytical::new(
//         |x, p, t, _rateiv, _cov| {
//             let mut xout = x.clone();
//             let ka = p[0];
//             let ke = p[1];

//             xout[0] = x[0] * (-ka * t).exp();

//             xout[1] = x[1] * (-ke * t).exp()
//                 + ((ka * x[0]) / (ka - ke)) * ((-ke * t).exp() - (-ka * t).exp());

//             xout
//         },
//         |_p, _t, _cov| {},
//         |_p, _t, _cov| lag! {},
//         |_p, _t, _cov| fa! {},
//         |_p, _t, _cov, _x| {},
//         |x, p, _t, _cov, y| {
//             fetch_params!(p, _ka, _ke, v);
//             y[1] = x[1] / v;
//             y[1];
//         },
//     )
// }


// /// Print a comprehensive SAEM report matching R saemix output format
// fn print_saem_report<E: pharmsol::Equation>(result: &pmcore::prelude::ParametricWorkspace<E>) {
//     let n_subjects = result.data().len();
//     // Count observations from all occasions
//     let n_obs: usize = result
//         .data()
//         .subjects()
//         .iter()
//         .flat_map(|s| s.occasions())
//         .flat_map(|o| o.events())
//         .filter(|e| matches!(e, pharmsol::Event::Observation(_)))
//         .count();
//     let param_names = result.population().param_names();
//     let n_params = param_names.len();
//     let mu = result.mu();
//     let omega = result.omega();

//     println!("\n{}", "=".repeat(60));
//     println!("{:^60}", "SAEM Algorithm Results");
//     println!("{}", "=".repeat(60));

//     // Dataset characteristics
//     println!("\n{}", "-".repeat(60));
//     println!("{:^60}", "Data");
//     println!("{}", "-".repeat(60));
//     println!("  Number of subjects:     {}", n_subjects);
//     println!("  Number of observations: {}", n_obs);
//     println!(
//         "  Average obs/subject:    {:.1}",
//         n_obs as f64 / n_subjects as f64
//     );

//     // Algorithm info
//     println!("\n{}", "-".repeat(60));
//     println!("{:^60}", "Algorithm");
//     println!("{}", "-".repeat(60));
//     println!("  Iterations completed:   {}", result.iterations());
//     println!("  Status:                 {:?}", result.status());

//     // Fixed effects (population means)
//     println!("\n{}", "-".repeat(60));
//     println!("{:^60}", "Fixed Effects (Population Means)");
//     println!("{}", "-".repeat(60));
//     println!("  {:12} {:>12} {:>12}", "Parameter", "Estimate", "");
//     println!("  {:12} {:>12} {:>12}", "---------", "--------", "");
//     for (i, name) in param_names.iter().enumerate() {
//         println!("  {:12} {:>12.4}", name, mu[i]);
//     }

//     // Variance of random effects
//     println!("\n{}", "-".repeat(60));
//     println!("{:^60}", "Variance of Random Effects");
//     println!("{}", "-".repeat(60));
//     println!("  {:12} {:>12}", "Parameter", "Estimate");
//     println!("  {:12} {:>12}", "---------", "--------");
//     for (i, name) in param_names.iter().enumerate() {
//         let var = omega[(i, i)];
//         println!("  omega2.{:<4} {:>12.4}", name, var);
//     }

//     // Covariances (if any non-zero off-diagonal)
//     let mut has_covariances = false;
//     for i in 0..n_params {
//         for j in (i + 1)..n_params {
//             if omega[(i, j)].abs() > 1e-10 {
//                 if !has_covariances {
//                     println!("\n  Covariances:");
//                     has_covariances = true;
//                 }
//                 println!(
//                     "  cov.{}.{:<6} {:>12.4}",
//                     param_names[i],
//                     param_names[j],
//                     omega[(i, j)]
//                 );
//             }
//         }
//     }

//     // Correlation matrix
//     println!("\n{}", "-".repeat(60));
//     println!("{:^60}", "Correlation Matrix of Random Effects");
//     println!("{}", "-".repeat(60));

//     // Header
//     print!("  {:12}", "");
//     for name in &param_names {
//         print!(" {:>10}", name);
//     }
//     println!();

//     // Matrix rows
//     for i in 0..n_params {
//         print!("  {:12}", param_names[i]);
//         for j in 0..n_params {
//             let sd_i = omega[(i, i)].sqrt();
//             let sd_j = omega[(j, j)].sqrt();
//             let corr = if sd_i > 0.0 && sd_j > 0.0 {
//                 omega[(i, j)] / (sd_i * sd_j)
//             } else if i == j {
//                 1.0
//             } else {
//                 0.0
//             };
//             print!(" {:>10.4}", corr);
//         }
//         println!();
//     }

//     // Residual error
//     println!("\n{}", "-".repeat(60));
//     println!("{:^60}", "Residual Error");
//     println!("{}", "-".repeat(60));
//     println!("  σ estimates:          {:?}", result.sigma().as_vec());

//     // Statistical criteria
//     println!("\n{}", "-".repeat(60));
//     println!("{:^60}", "Statistical Criteria");
//     println!("{}", "-".repeat(60));

//     let _ll = -result.objf() / 2.0; // Convert -2LL to LL
//     let n_fixed = n_params;
//     let n_random = n_params; // diagonal omega
//     let n_resid = 1; // residual error
//     let n_total_params = n_fixed + n_random + n_resid;

//     let aic = result.objf() + 2.0 * n_total_params as f64;
//     let bic = result.objf() + (n_total_params as f64) * (n_subjects as f64).ln();

//     println!("  -2LL = {:.4}", result.objf());
//     println!("  AIC  = {:.4}", aic);
//     println!("  BIC  = {:.4}", bic);

//     // Individual parameters (first 10 subjects) on the canonical ψ-space result surface.
//     println!("\n{}", "-".repeat(60));
//     println!("{:^60}", "Individual Parameters (first 10 subjects)");
//     println!("{}", "-".repeat(60));

//     // Header
//     print!("  {:>4}", "ID");
//     for name in &param_names {
//         print!(" {:>12}", name);
//     }
//     println!();

//     // Get individual estimates
//     let individuals = result.individual_estimates();
//     let show_count = std::cmp::min(10, individuals.nsubjects());

//     for i in 0..show_count {
//         if let Some(ind) = individuals.get(i) {
//             print!("  {:>4}", i + 1);
//             for j in 0..n_params {
//                 print!(" {:>12.6}", ind.psi()[j]);
//             }
//             println!();
//         }
//     }

//     // Summary statistics for each parameter
//     println!("\n{}", "-".repeat(60));
//     println!("{:^60}", "Summary of Individual Estimates");
//     println!("{}", "-".repeat(60));

//     for (p, name) in param_names.iter().enumerate() {
//         let mut values: Vec<f64> = Vec::new();
//         for i in 0..individuals.nsubjects() {
//             if let Some(ind) = individuals.get(i) {
//                 values.push(ind.psi()[p]);
//             }
//         }

//         if !values.is_empty() {
//             values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
//             let n = values.len();
//             let min = values[0];
//             let max = values[n - 1];
//             let mean: f64 = values.iter().sum::<f64>() / n as f64;
//             let median = if n % 2 == 0 {
//                 (values[n / 2 - 1] + values[n / 2]) / 2.0
//             } else {
//                 values[n / 2]
//             };
//             let q1 = values[n / 4];
//             let q3 = values[3 * n / 4];
//             let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
//             let sd = variance.sqrt();

//             println!("\n  --- {} ---", name);
//             println!("    Min:    {:>10.5}", min);
//             println!("    1st Qu: {:>10.5}", q1);
//             println!("    Median: {:>10.5}", median);
//             println!("    Mean:   {:>10.5}", mean);
//             println!("    3rd Qu: {:>10.5}", q3);
//             println!("    Max:    {:>10.5}", max);
//             println!("    SD:     {:>10.5}", sd);
//         }
//     }

//     // Derived statistics
//     println!("\n{}", "-".repeat(60));
//     println!("{:^60}", "Population Parameter Summary");
//     println!("{}", "-".repeat(60));
//     println!("  {:12} {:>10} {:>10}", "Parameter", "SD(ω)", "CV(%)");
//     println!("  {:12} {:>10} {:>10}", "---------", "------", "-----");
//     let cvs = result.cv_percent();
//     for (i, name) in param_names.iter().enumerate() {
//         let omega2 = omega[(i, i)];
//         let sd = omega2.sqrt();
//         println!("  {:12} {:>10.4} {:>10.1}", name, sd, cvs[i]);
//     }

//     println!("\n{}", "=".repeat(60));
// }
