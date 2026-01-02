//! Run SAEM on the bimodal_ke dataset
//!
//! This example demonstrates using the SAEM algorithm for a simple
//! one-compartment model with elimination rate constant (ke) and volume (v).
//!
//! Run with: cargo run --example bimodal_ke_saem --release

use anyhow::Result;
use pharmsol::prelude::models::one_compartment;
use pharmsol::{ResidualErrorModel, ResidualErrorModels};
use pmcore::algorithms::parametric::dispatch_parametric_algorithm;
use pmcore::prelude::*;

/// Create analytical one-compartment model (much faster than ODE)
fn create_equation() -> equation::Analytical {
    equation::Analytical::new(
        one_compartment, // Analytical solution: x = x0*exp(-ke*t) + rateiv/ke*(1-exp(-ke*t))
        |_p, _t, _cov| {},
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

fn main() -> Result<()> {
    // Load data
    let data = data::read_pmetrics("examples/bimodal_ke/bimodal_ke.csv")?;
    println!("Loaded {} subjects", data.len());

    // Create model
    let eq = create_equation();

    // Parameter ranges
    // NPAG found: ke mean=0.191 (range 0.01-0.98), v mean=107 (range 67-209)
    // SAEM needs reasonable starting bounds since it initializes at midpoint
    // Use ranges that center near the expected values
    let params = Parameters::new()
        .add("ke", 0.01, 0.5) // Midpoint ~0.255
        .add("v", 50.0, 180.0); // Midpoint ~115

    // Residual error model for parametric algorithms (SAEM)
    // Uses PREDICTION-based sigma: σ = sqrt(a² + b²*f²)
    // For proportional error: σ = b * |f|, so use Proportional with b=0.1
    let residual_error = ResidualErrorModels::new().add(0, ResidualErrorModel::proportional(0.1));

    // Create SAEM settings - parametric algorithms only need ResidualErrorModels!
    // No ErrorModels required - SAEM computes likelihood using ResidualErrorModels directly
    let mut settings = Settings::builder()
        .set_algorithm(Algorithm::SAEM)
        .set_parameters(params)
        .set_residual_error(residual_error)
        .build();

    settings.set_output_path("examples/bimodal_ke_saem/output/");
    settings.set_write_logs(true);
    settings.initialize_logs()?;

    println!("Running SAEM algorithm...");

    // Run the algorithm
    let mut algorithm = dispatch_parametric_algorithm(settings, eq, data)?;
    let mut result = algorithm.fit()?;

    // Write all output files
    result.write_outputs()?;
    println!("\nOutput files written to: examples/bimodal_ke_saem/output/");

    // Print comprehensive results summary (matching R saemix format)
    print_saem_report(&result);

    Ok(())
}

/// Print a comprehensive SAEM report matching R saemix output format
fn print_saem_report<E: pharmsol::Equation>(
    result: &pmcore::routines::output::ParametricResult<E>,
) {
    let n_subjects = result.data().len();
    // Count observations from all occasions
    let n_obs: usize = result
        .data()
        .subjects()
        .iter()
        .flat_map(|s| s.occasions())
        .flat_map(|o| o.events())
        .filter(|e| matches!(e, pharmsol::Event::Observation(_)))
        .count();
    let param_names = result.population().param_names();
    let n_params = param_names.len();
    let mu = result.mu();
    let omega = result.omega();

    println!("\n{}", "=".repeat(60));
    println!("{:^60}", "SAEM Algorithm Results");
    println!("{}", "=".repeat(60));

    // Dataset characteristics
    println!("\n{}", "-".repeat(60));
    println!("{:^60}", "Data");
    println!("{}", "-".repeat(60));
    println!("  Number of subjects:     {}", n_subjects);
    println!("  Number of observations: {}", n_obs);
    println!(
        "  Average obs/subject:    {:.1}",
        n_obs as f64 / n_subjects as f64
    );

    // Algorithm info
    println!("\n{}", "-".repeat(60));
    println!("{:^60}", "Algorithm");
    println!("{}", "-".repeat(60));
    println!("  Iterations completed:   {}", result.iterations());
    println!("  Status:                 {:?}", result.status());

    // Fixed effects (population means)
    println!("\n{}", "-".repeat(60));
    println!("{:^60}", "Fixed Effects (Population Means)");
    println!("{}", "-".repeat(60));
    println!("  {:12} {:>12} {:>12}", "Parameter", "Estimate", "");
    println!("  {:12} {:>12} {:>12}", "---------", "--------", "");
    for (i, name) in param_names.iter().enumerate() {
        println!("  {:12} {:>12.4}", name, mu[i]);
    }

    // Variance of random effects
    println!("\n{}", "-".repeat(60));
    println!("{:^60}", "Variance of Random Effects");
    println!("{}", "-".repeat(60));
    println!("  {:12} {:>12}", "Parameter", "Estimate");
    println!("  {:12} {:>12}", "---------", "--------");
    for (i, name) in param_names.iter().enumerate() {
        let var = omega[(i, i)];
        println!("  omega2.{:<4} {:>12.4}", name, var);
    }

    // Covariances (if any non-zero off-diagonal)
    let mut has_covariances = false;
    for i in 0..n_params {
        for j in (i + 1)..n_params {
            if omega[(i, j)].abs() > 1e-10 {
                if !has_covariances {
                    println!("\n  Covariances:");
                    has_covariances = true;
                }
                println!(
                    "  cov.{}.{:<6} {:>12.4}",
                    param_names[i],
                    param_names[j],
                    omega[(i, j)]
                );
            }
        }
    }

    // Correlation matrix
    println!("\n{}", "-".repeat(60));
    println!("{:^60}", "Correlation Matrix of Random Effects");
    println!("{}", "-".repeat(60));

    // Header
    print!("  {:12}", "");
    for name in &param_names {
        print!(" {:>10}", name);
    }
    println!();

    // Matrix rows
    for i in 0..n_params {
        print!("  {:12}", param_names[i]);
        for j in 0..n_params {
            let sd_i = omega[(i, i)].sqrt();
            let sd_j = omega[(j, j)].sqrt();
            let corr = if sd_i > 0.0 && sd_j > 0.0 {
                omega[(i, j)] / (sd_i * sd_j)
            } else if i == j {
                1.0
            } else {
                0.0
            };
            print!(" {:>10.4}", corr);
        }
        println!();
    }

    // Residual error
    println!("\n{}", "-".repeat(60));
    println!("{:^60}", "Residual Error");
    println!("{}", "-".repeat(60));
    // Get sigma from settings or result (assuming additive error model)
    // Note: In SAEM, sigma is estimated during M-step
    println!("  σ (residual SD): Not directly available in result");
    println!("  (See iteration output for current σ estimate)");

    // Statistical criteria
    println!("\n{}", "-".repeat(60));
    println!("{:^60}", "Statistical Criteria");
    println!("{}", "-".repeat(60));

    let _ll = -result.objf() / 2.0; // Convert -2LL to LL
    let n_fixed = n_params;
    let n_random = n_params; // diagonal omega
    let n_resid = 1; // residual error
    let n_total_params = n_fixed + n_random + n_resid;

    let aic = result.objf() + 2.0 * n_total_params as f64;
    let bic = result.objf() + (n_total_params as f64) * (n_subjects as f64).ln();

    println!("  -2LL = {:.4}", result.objf());
    println!("  AIC  = {:.4}", aic);
    println!("  BIC  = {:.4}", bic);

    // Individual parameters (first 10 subjects)
    // Note: SAEM stores individual estimates in φ space (unconstrained)
    // For lognormal parameters, ψ = exp(φ)
    println!("\n{}", "-".repeat(60));
    println!("{:^60}", "Individual Parameters (first 10 subjects)");
    println!("{}", "-".repeat(60));

    // Header
    print!("  {:>4}", "ID");
    for name in &param_names {
        print!(" {:>12}", name);
    }
    println!();

    // Get individual estimates
    let individuals = result.individual_estimates();
    let show_count = std::cmp::min(10, individuals.nsubjects());

    for i in 0..show_count {
        if let Some(ind) = individuals.get(i) {
            print!("  {:>4}", i + 1);
            let phi = ind.psi(); // Note: psi() returns φ (unconstrained) for SAEM
            for j in 0..n_params {
                // Transform phi to psi for lognormal parameters
                let psi_val = phi[j].exp();
                print!(" {:>12.6}", psi_val);
            }
            println!();
        }
    }

    // Summary statistics for each parameter
    println!("\n{}", "-".repeat(60));
    println!("{:^60}", "Summary of Individual Estimates");
    println!("{}", "-".repeat(60));

    for (p, name) in param_names.iter().enumerate() {
        // Collect all individual values for this parameter (transformed to natural space)
        let mut values: Vec<f64> = Vec::new();
        for i in 0..individuals.nsubjects() {
            if let Some(ind) = individuals.get(i) {
                let phi = ind.psi();
                let psi_val = phi[p].exp(); // Transform to natural space
                values.push(psi_val);
            }
        }

        if !values.is_empty() {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = values.len();
            let min = values[0];
            let max = values[n - 1];
            let mean: f64 = values.iter().sum::<f64>() / n as f64;
            let median = if n % 2 == 0 {
                (values[n / 2 - 1] + values[n / 2]) / 2.0
            } else {
                values[n / 2]
            };
            let q1 = values[n / 4];
            let q3 = values[3 * n / 4];
            let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let sd = variance.sqrt();

            println!("\n  --- {} ---", name);
            println!("    Min:    {:>10.5}", min);
            println!("    1st Qu: {:>10.5}", q1);
            println!("    Median: {:>10.5}", median);
            println!("    Mean:   {:>10.5}", mean);
            println!("    3rd Qu: {:>10.5}", q3);
            println!("    Max:    {:>10.5}", max);
            println!("    SD:     {:>10.5}", sd);
        }
    }

    // Derived statistics
    println!("\n{}", "-".repeat(60));
    println!("{:^60}", "Population Parameter Summary");
    println!("{}", "-".repeat(60));
    println!("  {:12} {:>10} {:>10}", "Parameter", "SD(ω)", "CV(%)");
    println!("  {:12} {:>10} {:>10}", "---------", "------", "-----");
    for (i, name) in param_names.iter().enumerate() {
        let omega2 = omega[(i, i)];
        let sd = omega2.sqrt();
        // For lognormal: CV% = 100 * sqrt(exp(ω²) - 1) ≈ 100 * ω for small ω
        let cv = 100.0 * (omega2.exp() - 1.0).sqrt();
        println!("  {:12} {:>10.4} {:>10.1}", name, sd, cv);
    }

    println!("\n{}", "=".repeat(60));
}
