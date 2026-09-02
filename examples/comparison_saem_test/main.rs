//! Run SAEM on the bimodal_ke dataset
//!
//! This example demonstrates using the SAEM algorithm for a simple
//! one-compartment model with elimination rate constant (ke) and volume (v).
//!
//! Run with: cargo run --example bimodal_ke_saem --release

use anyhow::Result;
use pmcore::prelude::*;


// arguments (ka: f64, ke: f64, v: f64, trial_id: u64)
fn main() -> Result<()> {
    let data = data::read_pmetrics("examples/comparison_saem_test/data_pmcore.csv")?;
    println!("Loaded {} subjects", data.len());

    // Create model
    let equation = analytical! {
        name: "data",
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
        .parameter(Parameter::log("ke").with_initial(0.1))
        .parameter(Parameter::log("v").with_initial(60.0))
        .error_model("outeq_0", ResidualErrorModel::proportional(0.1))
        .build()?;

    let config = SaemConfig::new()
        .k1_iterations(300)
        .k2_iterations(100)
        .burn_in(5)
        
        .n_chains(1)
        .seed(632545)

        // sa_iterations: 0,
        // sa_cooling_factor: 0.97,
        // rw_init: 0.5,
        .mcmc_iterations(1)
        // Disabled by default; opt in to the block-mixture kernel.
        .eta_block_iterations(0)
        // .saemix_mcmc(None)
        .adapt_interval(50)
        // Guard against one-draw correlated Ω collapse in exploration.
        .omega_sa_max_step(0.1)
        // omega_min_variance: 1e-6,
        .omega_iov_min_variance(1e-8)
        // Uses the established 1e-12 residual-variance guard on the SD scale.
        .residual_min_sigma(1e-6)
        .residual_optimizer_max_iterations(200)
        .compute_map(true)
        .map_max_iterations(200)
        .map_sd_tolerance(1e-8)
        .map_initial_step(0.1)
        .estimator_policy(SaemEstimatorPolicy::TerminalIterate)
        // .markov_simulation_variance(None)
        // .covariance_stability(None)
        // .operational_convergence(None)
        // .marginal_likelihood(None)
        ;

    let result = problem.fit_with(config)?;

    // Write all output files
    result.write_outputs("examples/comparison_saem_test/outputs/pmcore_output", 0.0, 0.0)?;

    Ok(())
}
