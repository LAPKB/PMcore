use crate::algorithms::initialize_algorithm;
use crate::prelude::output::NPCycle;
use crate::prelude::{
    output::NPResult,
    predict::{Engine, Predict},
    *,
};
use csv::{ReaderBuilder, WriterBuilder};
use eyre::Result;

use ndarray::Array2;
use ndarray_csv::Array2Reader;
use predict::sim_obs;
use std::fs::File;
use std::thread::spawn;
use std::time::Instant;
use tokio::sync::mpsc::{self};

pub fn simulate<S>(engine: Engine<S>, settings_path: String) -> Result<()>
where
    S: Predict<'static> + std::marker::Sync + std::marker::Send + 'static + Clone,
{
    let settings = settings::simulator::read(settings_path);
    let theta_file = File::open(settings.paths.theta).unwrap();
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(theta_file);
    let theta: Array2<f64> = reader.deserialize_array2_dynamic().unwrap();
    let scenarios = datafile::parse(&settings.paths.data).unwrap();

    let ypred = sim_obs(&engine, &scenarios, &theta, false);

    let sim_file = File::create("simulation_output.csv").unwrap();
    let mut sim_writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(sim_file);
    sim_writer
        .write_record(["id", "point", "time", "sim_obs"])
        .unwrap();
    for (id, scenario) in scenarios.iter().enumerate() {
        let time = scenario.obs_times.clone();
        for (point, _spp) in theta.rows().into_iter().enumerate() {
            for (i, time) in time.iter().enumerate() {
                sim_writer.write_record(&[
                    id.to_string(),
                    point.to_string(),
                    time.to_string(),
                    ypred.get((id, point)).unwrap().get(i).unwrap().to_string(),
                ])?;
            }
        }
    }
    Ok(())
}
pub fn start<S>(engine: Engine<S>, settings_path: String) -> Result<NPResult>
where
    S: Predict<'static> + std::marker::Sync + std::marker::Send + 'static + Clone,
{
    let now = Instant::now();
    let settings = settings::run::read(settings_path);
    logger::setup_log(&settings);
    let (tx, rx) = mpsc::unbounded_channel::<NPCycle>();
    let mut algorithm = initialize_algorithm(engine.clone(), settings.clone(), tx);
    // Spawn new thread for TUI
    let settings_tui = settings.clone();
    if settings.parsed.config.tui {
        let _ui_handle = spawn(move || {
            start_ui(rx, settings_tui).expect("Failed to start TUI");
        });
    }

    let result = algorithm.fit();
    log::info!("Total time: {:.2?}", now.elapsed());

    let idelta = settings.parsed.config.idelta.unwrap_or(0.0);
    if let Some(write) = &settings.parsed.config.pmetrics_outputs {
        result.write_outputs(*write, &engine, idelta);
    }

    Ok(result)
}
