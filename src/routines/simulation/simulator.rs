use crate::prelude::*;
use csv::{ReaderBuilder, WriterBuilder};
use eyre::Result;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use predict::{sim_obs, Engine, Predict};
use std::fs::File;

pub fn simulate<S>(engine: Engine<S>, settings_path: String) -> Result<()>
where
    S: Predict + std::marker::Sync + std::marker::Send + 'static + Clone,
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
