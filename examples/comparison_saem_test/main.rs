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
        .seed(632545) // weird, typically you have something like "13" or "9", we use -17
        .n_chains(25)
        // .mcmc_iterations(2)
        .burn_in(5)
        .k1_iterations(300)
        .k2_iterations(150);

    let result = problem.fit_with(config)?;

    // Write all output files
    result.write_outputs("examples/comparison_saem_test/outputs/pmcore_output", 0.0, 0.0)?;

    Ok(())
}
