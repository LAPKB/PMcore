//! Run SAEM on the bimodal_ke dataset
//!
//! This example demonstrates using the SAEM algorithm for a simple
//! one-compartment model with elimination rate constant (ke) and volume (v).
//!
//! Run with: cargo run --example bimodal_ke_saem --release
//! 
//! run 128 times, pop truth mean/standard deviation, individual truths

use anyhow::Result;
use pmcore::prelude::*;
use rand::SeedableRng;

use std::fs::{remove_dir_all, File, OpenOptions, read_dir};
use std::path::Path;
use csv::{Reader, Writer};

use rand_distr::{Distribution, Normal};
use rand::rngs::StdRng;


const SEED: u64 = 17;
const TRIALS_PER_DATASET: u64 = 2;


#[derive(serde::Serialize)]
struct Entry<'a> {
    dataset_file_name: &'a str,
    ka: f64,
    ke: f64,
    v: f64,
    trial: u64
}

#[derive(serde::Deserialize)]
struct PopEntry {
    name: String,
    estimate: f64,
    #[serde(rename = "scale")]
    _scale: String,
    #[serde(rename = "estimated")]
    _estimated: bool,
    #[serde(rename = "iiv")]
    _iiv: bool,
    #[serde(rename = "iov")]
    _iov: bool
}

// arguments (ka: f64, ke: f64, v: f64, trial_id: u64)
fn main() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let ka_dist: Normal<f64> = Normal::new(0.8, 3.0*0.0064).unwrap();
    let ke_dist: Normal<f64> = Normal::new(0.18, 3.0*0.000324).unwrap();
    let v_dist: Normal<f64> = Normal::new(63.0, 39.69).unwrap();
    
    remove_dir_all("examples/analytical_saem_test/outputs/pmcore_output")?;

    let paths = read_dir("examples/analytical_saem_test/test_data/pmcore_data");

    let mut trial_id = 0;

    for entry in paths.unwrap() {

        let path = entry?.path();

        if !path.is_file() {
            continue;
        }

        println!("Starting trials with file: {:?}", path);

        for i in 0..TRIALS_PER_DATASET {

            pmcore_loop(path.to_str().unwrap(), "examples/analytical_saem_test/outputs/pmcore_output/run_data", 
                ka_dist.sample(&mut rng).max(0.000001),
                ke_dist.sample(&mut rng).max(0.000001),
                v_dist.sample(&mut rng).max(0.000001)
            )?;
            
            let output_file = File::open(Path::new("examples/analytical_saem_test/outputs/pmcore_output/run_data/population.csv"))?;
            let mut output_reader = Reader::from_reader(output_file);

            let mut final_ka: f64 = -1.0;
            let mut final_ke: f64 = -1.0;
            let mut final_v: f64 = -1.0;

            for entry in output_reader.deserialize() {
                let record: PopEntry = entry?;
                if record.name == "ka" { final_ka = record.estimate; }
                else if record.name == "ke" { final_ke = record.estimate; }
                else if record.name == "v" { final_v = record.estimate; }
            }
                
            let trace_file = OpenOptions::new()
                .write(true)
                .append(true)
                .create(true)
                .open("examples/analytical_saem_test/outputs/pmcore_output/pmcore_trace.csv")
                .unwrap();
            
            let mut trace_writer = Writer::from_writer(trace_file);
            if trial_id == 0 {
                trace_writer.write_record(["dataset_file_name", "ka", "ke", "v", "trial"])?;
            }

            trace_writer.serialize(Entry {
                dataset_file_name: path.to_str().unwrap_or("failed to read file name").strip_prefix("examples/analytical_saem_test/test_data/pmcore_data/").unwrap(),
                ka: final_ka,
                ke: final_ke,
                v: final_v,
                trial: i+1,
            })?;
            
            trial_id += 1;
        }
    }

    Ok(())
}

fn pmcore_loop(data: impl Into<String>, output: &str, ka: f64, ke: f64, v: f64) -> Result<()> {
    println!("Initial values - ka: {}, ke: {}, v: {}", ka, ke, v);
    // let data = data::read_pmetrics("examples/analytical_saem_test/converted_data_theo.csv")?;
    let data = data::read_pmetrics(data)?;
    // println!("Loaded {} subjects", data.len());

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

    let config = SaemConfig::new()
        .seed(632545)
        .n_chains(25)
        .mcmc_iterations(2)
        .eta_block_iterations(0)
        .burn_in(100)
        .k1_iterations(300)
        .k2_iterations(150);

    let problem = EstimationProblem::parametric(equation, data)
        // Must currently follow the model metadata order: ka 0.8 3*0.0064, ke 0.18 3*.000324, v 63.0 3*39.69 mean/variance
        .parameter(Parameter::log("ka").with_initial(ka))
        .parameter(Parameter::log("ke").with_initial(ke))
        .parameter(Parameter::log("v").with_initial(v))
        .error_model("outeq_0", ResidualErrorModel::constant(1.0))
        .build()?;

    let result = problem.fit_with(config)?;

    // Write all output files
    // let output_dir = "examples/analytical_saem_test/outputs/pmcore_output/";
    result.write_outputs(output, 0.0, 0.0)?;

    // Print comprehensive results summary (matching R saemix format)
    // print_saem_report(result);

    Ok(())
}
